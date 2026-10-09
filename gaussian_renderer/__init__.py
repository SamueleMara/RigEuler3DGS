#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# Extensions:
#   - render_feature_field(): arbitrary-channel per-Gaussian feature splatting
#   - render_semantic_field(): one-channel semantic/foreground splatting
#   - render_sam3_feature_field(): normalized dense SAM3 feature splatting
#   - render_sam3_affinity_field(): normalized SAM3 affinity-field splatting

import torch
import math
import torch.nn.functional as F
from diff_gaussian_rasterization import (
    GaussianRasterizationSettings,
    GaussianRasterizer,
)
from scene.gaussian_model import GaussianModel
from utils.sh_utils import eval_sh

# Keep network_gui import behavior compatible with original gaussian_renderer.
try:
    from . import network_gui
except Exception:
    network_gui = None


def _make_screenspace_points(pc: GaussianModel, detach_geometry: bool = False):
    base_xyz = pc.get_xyz.detach() if detach_geometry else pc.get_xyz
    screenspace_points = (
        torch.zeros_like(
            base_xyz,
            dtype=base_xyz.dtype,
            requires_grad=not detach_geometry,
            device="cuda",
        )
        + 0
    )
    if not detach_geometry:
        try:
            screenspace_points.retain_grad()
        except Exception:
            pass
    return screenspace_points


def _make_rasterizer(viewpoint_camera, pipe, bg_color, pc, scaling_modifier=1.0):
    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)

    raster_settings = GaussianRasterizationSettings(
        image_height=int(viewpoint_camera.image_height),
        image_width=int(viewpoint_camera.image_width),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg=bg_color,
        scale_modifier=scaling_modifier,
        viewmatrix=viewpoint_camera.world_view_transform,
        projmatrix=viewpoint_camera.full_proj_transform,
        sh_degree=pc.active_sh_degree,
        campos=viewpoint_camera.camera_center,
        prefiltered=False,
        debug=pipe.debug,
        antialiasing=pipe.antialiasing,
    )
    return GaussianRasterizer(raster_settings=raster_settings)


def _geometry_inputs(
    pc: GaussianModel, pipe, scaling_modifier=1.0, detach_geometry: bool = False
):
    scales = None
    rotations = None
    cov3D_precomp = None

    if pipe.compute_cov3D_python:
        cov3D_precomp = pc.get_covariance(scaling_modifier)
        if detach_geometry and cov3D_precomp is not None:
            cov3D_precomp = cov3D_precomp.detach()
    else:
        scales = pc.get_scaling
        rotations = pc.get_rotation
        if detach_geometry:
            scales = scales.detach()
            rotations = rotations.detach()

    return scales, rotations, cov3D_precomp


def render(
    viewpoint_camera,
    pc: GaussianModel,
    pipe,
    bg_color: torch.Tensor,
    scaling_modifier=1.0,
    separate_sh=False,
    override_color=None,
    use_trained_exp=False,
    means3D_override: torch.Tensor | None = None,
    rotations_override: torch.Tensor | None = None,
):
    """
    Render RGB exactly as the original 3DGS renderer.

    Background tensor (bg_color) must be on GPU and have 3 channels.
    """
    means3D = pc.get_xyz if means3D_override is None else means3D_override
    if means3D.ndim != 2 or means3D.shape != pc.get_xyz.shape:
        raise ValueError("means3D_override must match pc.get_xyz [N,3]")
    screenspace_points = (
        torch.zeros_like(
            means3D,
            dtype=means3D.dtype,
            requires_grad=means3D.requires_grad,
            device=means3D.device,
        )
        + 0
    )
    if means3D.requires_grad:
        try:
            screenspace_points.retain_grad()
        except Exception:
            pass
    rasterizer = _make_rasterizer(
        viewpoint_camera, pipe, bg_color, pc, scaling_modifier
    )

    means2D = screenspace_points
    opacity = pc.get_opacity
    scales, rotations, cov3D_precomp = _geometry_inputs(pc, pipe, scaling_modifier)
    if rotations_override is not None and not pipe.compute_cov3D_python:
        if rotations_override.shape != pc.get_rotation.shape:
            raise ValueError("rotations_override must match pc.get_rotation [N,4]")
        rotations = F.normalize(rotations_override, dim=-1).contiguous()

    shs = None
    colors_precomp = None
    dc = None
    if override_color is None:
        if pipe.convert_SHs_python:
            shs_view = pc.get_features.transpose(1, 2).view(
                -1, 3, (pc.max_sh_degree + 1) ** 2
            )
            dir_pp = means3D - viewpoint_camera.camera_center.repeat(
                pc.get_features.shape[0], 1
            )
            dir_pp_normalized = dir_pp / dir_pp.norm(dim=1, keepdim=True).clamp_min(
                1e-8
            )
            sh2rgb = eval_sh(pc.active_sh_degree, shs_view, dir_pp_normalized)
            colors_precomp = torch.clamp_min(sh2rgb + 0.5, 0.0)
        else:
            if separate_sh:
                dc, shs = pc.get_features_dc, pc.get_features_rest
            else:
                shs = pc.get_features
    else:
        colors_precomp = override_color

    if separate_sh:
        rendered_image, radii, depth_image = rasterizer(
            means3D=means3D,
            means2D=means2D,
            dc=dc,
            shs=shs,
            colors_precomp=colors_precomp,
            opacities=opacity,
            scales=scales,
            rotations=rotations,
            cov3D_precomp=cov3D_precomp,
        )
    else:
        rendered_image, radii, depth_image = rasterizer(
            means3D=means3D,
            means2D=means2D,
            shs=shs,
            colors_precomp=colors_precomp,
            opacities=opacity,
            scales=scales,
            rotations=rotations,
            cov3D_precomp=cov3D_precomp,
        )

    if use_trained_exp:
        exposure = pc.get_exposure_from_name(viewpoint_camera.image_name)
        rendered_image = (
            torch.matmul(rendered_image.permute(1, 2, 0), exposure[:3, :3]).permute(
                2, 0, 1
            )
            + exposure[:3, 3, None, None]
        )

    rendered_image = rendered_image.clamp(0, 1)
    return {
        "render": rendered_image,
        "viewspace_points": screenspace_points,
        "visibility_filter": (radii > 0).nonzero(),
        "radii": radii,
        "depth": depth_image,
    }


def render_feature_field(
    viewpoint_camera,
    pc: GaussianModel,
    pipe,
    features: torch.Tensor,
    bg_feature: torch.Tensor | None = None,
    scaling_modifier=1.0,
    detach_geometry: bool = False,
):
    """
    Render arbitrary C-channel per-Gaussian features with the same alpha
    compositing used for RGB color splats.

    features: [N, C]
    bg_feature: [C], defaults to zeros.
    """
    if features.ndim != 2:
        raise ValueError(f"features must be [N,C], got {tuple(features.shape)}")
    if features.shape[0] != pc.get_xyz.shape[0]:
        raise ValueError(
            f"features N={features.shape[0]} does not match Gaussians N={pc.get_xyz.shape[0]}"
        )

    if bg_feature is None:
        bg_feature = torch.zeros(
            features.shape[1], device=features.device, dtype=features.dtype
        )
    else:
        bg_feature = bg_feature.to(device=features.device, dtype=features.dtype)
    if bg_feature.numel() != features.shape[1]:
        raise ValueError(
            f"bg_feature must have {features.shape[1]} channels, got {bg_feature.numel()}"
        )

    original_channels = int(features.shape[1])
    if original_channels == 4:
        # The CUDA extension reserves raw 4-channel feature renders for
        # homogeneous position transport.  Pad arbitrary feature fields to keep
        # semantic/DINO/SAM3 on the generic feature path.
        features = torch.cat(
            (
                features,
                torch.zeros((features.shape[0], 1), device=features.device, dtype=features.dtype),
            ),
            dim=1,
        )
        bg_feature = torch.cat(
            (
                bg_feature,
                torch.zeros(1, device=bg_feature.device, dtype=bg_feature.dtype),
            ),
            dim=0,
        )

    screenspace_points = _make_screenspace_points(pc, detach_geometry=detach_geometry)
    rasterizer = _make_rasterizer(
        viewpoint_camera, pipe, bg_feature, pc, scaling_modifier
    )

    scales, rotations, cov3D_precomp = _geometry_inputs(
        pc, pipe, scaling_modifier, detach_geometry=detach_geometry
    )
    rendered_features, radii, depth_image = rasterizer.render_features(
        means3D=pc.get_xyz.detach() if detach_geometry else pc.get_xyz,
        means2D=screenspace_points,
        features=features.contiguous(),
        opacities=pc.get_opacity.detach() if detach_geometry else pc.get_opacity,
        scales=scales,
        rotations=rotations,
        cov3D_precomp=cov3D_precomp,
    )

    return {
        "render": rendered_features[:original_channels],
        "viewspace_points": screenspace_points,
        "visibility_filter": (radii > 0).nonzero(),
        "radii": radii,
        "depth": depth_image,
    }


def render_sam3_feature_field(
    viewpoint_camera,
    pc: GaussianModel,
    pipe,
    features: torch.Tensor | None = None,
    scaling_modifier=1.0,
    detach_geometry: bool = False,
):
    """Render normalized dense SAM3 descriptors through the CUDA feature path."""
    feature = pc.get_sam3_feature_field if features is None else features
    if not torch.is_tensor(feature) or feature.numel() == 0:
        raise RuntimeError("SAM3 feature field is not initialized")
    if feature.ndim != 2 or feature.shape[0] != pc.get_xyz.shape[0]:
        raise ValueError("SAM3 feature tensor must have shape [N,C]")
    normalized = F.normalize(feature, dim=-1, eps=1.0e-8)
    return render_feature_field(
        viewpoint_camera,
        pc,
        pipe,
        normalized,
        bg_feature=torch.zeros(normalized.shape[1], device=normalized.device, dtype=normalized.dtype),
        scaling_modifier=scaling_modifier,
        detach_geometry=detach_geometry,
    )


def render_sam3_affinity_field(
    viewpoint_camera,
    pc: GaussianModel,
    pipe,
    scaling_modifier=1.0,
):
    """Render alpha-normalized, unit per-Gaussian SAM3 affinity embeddings."""
    feature = pc.get_sam3_affinity
    if not torch.is_tensor(feature) or feature.numel() == 0:
        raise RuntimeError("SAM3 affinity field is not initialized")
    normalized = F.normalize(feature, dim=-1, eps=1.0e-8)
    pkg = render_feature_field(
        viewpoint_camera,
        pc,
        pipe,
        normalized,
        bg_feature=torch.zeros(normalized.shape[1], device=normalized.device, dtype=normalized.dtype),
        scaling_modifier=scaling_modifier,
        detach_geometry=True,
    )
    alpha_pkg = render_feature_field(
        viewpoint_camera,
        pc,
        pipe,
        torch.ones((normalized.shape[0], 1), device=normalized.device, dtype=normalized.dtype),
        bg_feature=torch.zeros(1, device=normalized.device, dtype=normalized.dtype),
        scaling_modifier=scaling_modifier,
        detach_geometry=True,
    )
    alpha = alpha_pkg["render"][0].detach().clamp(0.0, 1.0)
    pkg["feature"] = pkg["render"] / alpha.clamp_min(1.0e-4)[None]
    pkg["alpha"] = alpha
    return pkg


def render_semantic_field(
    viewpoint_camera,
    pc: GaussianModel,
    pipe,
    semantic_values: torch.Tensor | None = None,
    scaling_modifier=1.0,
    detach_geometry: bool = False,
):
    """
    Render a one-channel foreground/semantic probability image.

    By default uses sigmoid(pc.semantic_logits).
    """
    if semantic_values is None:
        semantic_values = pc.get_semantic_probability
    if semantic_values.ndim == 1:
        semantic_values = semantic_values[:, None]
    bg = torch.zeros(1, device=semantic_values.device, dtype=semantic_values.dtype)
    return render_feature_field(
        viewpoint_camera=viewpoint_camera,
        pc=pc,
        pipe=pipe,
        features=semantic_values,
        bg_feature=bg,
        scaling_modifier=scaling_modifier,
        detach_geometry=detach_geometry,
    )


def render_query_feature_field(
    viewpoint_camera,
    pc: GaussianModel,
    pipe,
    query_positions: torch.Tensor,
    features: torch.Tensor,
    query_rotations: torch.Tensor | None = None,
    bg_feature: torch.Tensor | None = None,
    scaling_modifier: float = 1.0,
):
    """Rasterize differentiable per-Gaussian features with fixed query geometry.

    Query positions, covariance, rotation and opacity are detached. Gradients
    flow only to ``features``. This is the bridge for projected-velocity losses
    that must not optimize splat ownership or camera geometry.
    """
    if query_positions.ndim != 2 or query_positions.shape[-1] != 3:
        raise ValueError(
            f"query_positions must be [N,3], got {tuple(query_positions.shape)}"
        )
    if features.ndim != 2:
        raise ValueError(f"features must be [N,C], got {tuple(features.shape)}")
    if (
        query_positions.shape[0] != pc.get_xyz.shape[0]
        or features.shape[0] != pc.get_xyz.shape[0]
    ):
        raise ValueError("query_positions and features must match the Gaussian count")

    if bg_feature is None:
        bg_feature = torch.zeros(
            features.shape[1], device=features.device, dtype=features.dtype
        )
    else:
        bg_feature = bg_feature.to(device=features.device, dtype=features.dtype)
    if bg_feature.numel() != features.shape[1]:
        raise ValueError(f"bg_feature must have {features.shape[1]} channels")

    device, dtype = features.device, features.dtype
    original_channels = int(features.shape[1])
    if original_channels == 4:
        features = torch.cat(
            (
                features,
                torch.zeros((features.shape[0], 1), device=device, dtype=dtype),
            ),
            dim=1,
        )
        bg_feature = torch.cat(
            (
                bg_feature,
                torch.zeros(1, device=bg_feature.device, dtype=bg_feature.dtype),
            ),
            dim=0,
        )

    query_positions = (
        query_positions.detach().to(device=device, dtype=dtype).contiguous()
    )
    screenspace_points = torch.zeros_like(query_positions, requires_grad=False)
    rasterizer = _make_rasterizer(
        viewpoint_camera, pipe, bg_feature, pc, scaling_modifier
    )

    if pipe.compute_cov3D_python:
        cov3D_precomp = (
            pc.get_covariance(scaling_modifier).detach().to(device=device, dtype=dtype)
        )
        scales = None
        rotations = None
    else:
        cov3D_precomp = None
        scales = pc.get_scaling.detach().to(device=device, dtype=dtype)
        rotations = (
            (
                pc.get_rotation.detach()
                if query_rotations is None
                else query_rotations.detach()
            )
            .to(device=device, dtype=dtype)
            .contiguous()
        )
        if rotations.shape[0] != query_positions.shape[0]:
            raise ValueError("query_rotations must match the Gaussian count")

    rendered_features, radii, depth_image = rasterizer.render_features(
        means3D=query_positions,
        means2D=screenspace_points,
        features=features.contiguous(),
        opacities=pc.get_opacity.detach().to(device=device, dtype=dtype),
        scales=scales,
        rotations=rotations,
        cov3D_precomp=cov3D_precomp,
    )
    return {
        "render": rendered_features[:original_channels],
        "viewspace_points": screenspace_points,
        "visibility_filter": (radii > 0).nonzero(),
        "radii": radii,
        "depth": depth_image,
    }


def _resize_observation_to_camera(
    value: torch.Tensor,
    *,
    channels: int,
    height: int,
    width: int,
    device,
):
    if value.ndim == 2:
        value = value[None]
    if value.ndim != 3 or value.shape[0] != channels:
        raise ValueError(f"observation must have shape [{channels},H,W]")
    value = value.to(device=device, dtype=torch.float32)
    if tuple(value.shape[-2:]) != (height, width):
        value = F.interpolate(
            value[None],
            size=(height, width),
            mode="bilinear",
            align_corners=True,
        )[0]
    return value.contiguous()


@torch.no_grad()
def render_observed_flow_stats(
    viewpoint_camera,
    pc: GaussianModel,
    pipe,
    query_positions: torch.Tensor,
    observed_flow: torch.Tensor,
    observed_confidence: torch.Tensor,
    group_ids: torch.Tensor | None = None,
    num_groups: int | None = None,
    query_rotations: torch.Tensor | None = None,
    scaling_modifier: float = 1.0,
    delta_time: float = 1.0,
    min_observation_weight: float = 0.0,
):
    """Accumulate rasterized optical-flow normal equations.

    The returned statistics are per Gaussian by default, or per caller-provided
    group/cell id.  Each contribution is weighted by the same query-frame
    ``T*alpha`` ownership used by the RGB rasterizer.
    """
    if query_positions.ndim != 2 or query_positions.shape[-1] != 3:
        raise ValueError(
            f"query_positions must be [N,3], got {tuple(query_positions.shape)}"
        )
    if query_positions.shape[0] != pc.get_xyz.shape[0]:
        raise ValueError(
            f"query position count {query_positions.shape[0]} does not match "
            f"Gaussian count {pc.get_xyz.shape[0]}"
        )

    height = int(viewpoint_camera.image_height)
    width = int(viewpoint_camera.image_width)
    device = query_positions.device
    flow = _resize_observation_to_camera(
        observed_flow, channels=2, height=height, width=width, device=device
    )
    confidence = _resize_observation_to_camera(
        observed_confidence, channels=1, height=height, width=width, device=device
    ).clamp(0.0, 1.0)

    background = torch.zeros(1, device=device, dtype=torch.float32)
    rasterizer = _make_rasterizer(
        viewpoint_camera, pipe, background, pc, scaling_modifier
    )
    screenspace_points = torch.zeros_like(query_positions, requires_grad=False)

    if pipe.compute_cov3D_python:
        cov3D_precomp = pc.get_covariance(scaling_modifier).detach()
        scales = None
        rotations = None
    else:
        cov3D_precomp = None
        scales = pc.get_scaling.detach()
        rotations = (
            pc.get_rotation.detach()
            if query_rotations is None
            else query_rotations.detach().contiguous()
        )
        if rotations.shape[0] != query_positions.shape[0]:
            raise ValueError("query_rotations must match the Gaussian count")

    rendered, normal6, rhs, y2, weight, count, radii = rasterizer.accumulate_flow_stats(
        means3D=query_positions.detach().contiguous(),
        observed_flow=flow,
        observed_confidence=confidence,
        opacities=pc.get_opacity.detach(),
        scales=scales,
        rotations=rotations,
        cov3D_precomp=cov3D_precomp,
        group_ids=group_ids,
        num_groups=num_groups,
        delta_time=delta_time,
        min_observation_weight=min_observation_weight,
    )
    return {
        "rendered": rendered,
        "normal6": normal6,
        "rhs": rhs,
        "y2": y2,
        "weight_sum": weight,
        "observation_count": count.to(torch.long),
        "radii": radii,
        "viewspace_points": screenspace_points,
        "visibility_filter": (radii > 0).nonzero(),
    }


@torch.no_grad()
def accumulate_pixel_features_to_gaussians(
    viewpoint_camera,
    pc: GaussianModel,
    pipe,
    pixel_features: torch.Tensor,
    pixel_confidence: torch.Tensor,
    *,
    query_positions: torch.Tensor | None = None,
    query_rotations: torch.Tensor | None = None,
    group_ids: torch.Tensor | None = None,
    num_groups: int | None = None,
    scaling_modifier: float = 1.0,
    min_observation_weight: float = 0.0,
):
    """Reverse-render arbitrary pixel evidence with exact ``T*alpha`` ownership."""
    positions = pc.get_xyz.detach() if query_positions is None else query_positions.detach()
    if positions.ndim != 2 or positions.shape != pc.get_xyz.shape:
        raise ValueError("query_positions must match pc.get_xyz [N,3]")
    height, width = int(viewpoint_camera.image_height), int(viewpoint_camera.image_width)
    features = _resize_observation_to_camera(
        pixel_features,
        channels=int(torch.as_tensor(pixel_features).shape[-3]),
        height=height,
        width=width,
        device=positions.device,
    )
    confidence = _resize_observation_to_camera(
        pixel_confidence, channels=1, height=height, width=width, device=positions.device
    ).clamp(0.0, 1.0)
    rasterizer = _make_rasterizer(
        viewpoint_camera, pipe,
        torch.zeros(1, device=positions.device, dtype=torch.float32),
        pc, scaling_modifier,
    )
    if pipe.compute_cov3D_python:
        cov3D_precomp = pc.get_covariance(scaling_modifier).detach()
        scales = rotations = None
    else:
        cov3D_precomp = None
        scales = pc.get_scaling.detach()
        rotations = pc.get_rotation.detach() if query_rotations is None else query_rotations.detach()
        rotations = F.normalize(rotations, dim=-1).contiguous()
    rendered, feature_sum, weight, count, raw_support, visible_mass, radii = rasterizer.accumulate_pixel_features_to_gaussians(
        means3D=positions.contiguous(),
        pixel_features=features,
        pixel_confidence=confidence,
        opacities=pc.get_opacity.detach(),
        scales=scales,
        rotations=rotations,
        cov3D_precomp=cov3D_precomp,
        group_ids=group_ids,
        num_groups=num_groups,
        min_observation_weight=min_observation_weight,
    )
    eps = 1.0e-8
    visibility_ratio = visible_mass / raw_support.clamp_min(eps)
    valid_fraction = weight / visible_mass.clamp_min(eps)
    return {
        "rendered": rendered,
        "feature_sum": feature_sum,
        "feature_mean": feature_sum / weight[:, None].clamp_min(eps),
        # Backward-compatible name: confidence-weighted visible ownership.
        "weight_sum": weight,
        "valid_visible_mass": weight,
        # Occlusion-aware alpha statistics from the native rasterizer.
        "raw_support": raw_support,
        "visible_mass": visible_mass,
        "visibility_ratio": visibility_ratio.clamp(0.0, 1.0),
        "valid_fraction": valid_fraction.clamp(0.0, 1.0),
        "projected": raw_support > eps,
        "alpha_visible": visible_mass > eps,
        "evidence_valid": weight > eps,
        "observation_count": count.to(torch.long),
        "radii": radii,
    }
