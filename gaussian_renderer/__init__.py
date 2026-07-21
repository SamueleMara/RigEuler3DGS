#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# Extensions:
#   - render_feature_field(): arbitrary-channel per-Gaussian feature splatting
#   - render_semantic_field(): one-channel semantic/foreground splatting

import torch
import math
from diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer
from scene.gaussian_model import GaussianModel
from utils.sh_utils import eval_sh

# Keep network_gui import behavior compatible with original gaussian_renderer.
try:
    from . import network_gui
except Exception:
    network_gui = None


def _make_screenspace_points(pc: GaussianModel, detach_geometry: bool = False):
    base_xyz = pc.get_xyz.detach() if detach_geometry else pc.get_xyz
    screenspace_points = torch.zeros_like(
        base_xyz,
        dtype=base_xyz.dtype,
        requires_grad=not detach_geometry,
        device="cuda",
    ) + 0
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


def _geometry_inputs(pc: GaussianModel, pipe, scaling_modifier=1.0, detach_geometry: bool = False):
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
):
    """
    Render RGB exactly as the original 3DGS renderer.

    Background tensor (bg_color) must be on GPU and have 3 channels.
    """
    screenspace_points = _make_screenspace_points(pc)
    rasterizer = _make_rasterizer(viewpoint_camera, pipe, bg_color, pc, scaling_modifier)

    means3D = pc.get_xyz
    means2D = screenspace_points
    opacity = pc.get_opacity
    scales, rotations, cov3D_precomp = _geometry_inputs(pc, pipe, scaling_modifier)

    shs = None
    colors_precomp = None
    dc = None
    if override_color is None:
        if pipe.convert_SHs_python:
            shs_view = pc.get_features.transpose(1, 2).view(-1, 3, (pc.max_sh_degree + 1) ** 2)
            dir_pp = pc.get_xyz - viewpoint_camera.camera_center.repeat(pc.get_features.shape[0], 1)
            dir_pp_normalized = dir_pp / dir_pp.norm(dim=1, keepdim=True).clamp_min(1e-8)
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
            torch.matmul(rendered_image.permute(1, 2, 0), exposure[:3, :3])
            .permute(2, 0, 1)
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
        raise ValueError(f"features N={features.shape[0]} does not match Gaussians N={pc.get_xyz.shape[0]}")

    if bg_feature is None:
        bg_feature = torch.zeros(features.shape[1], device=features.device, dtype=features.dtype)
    else:
        bg_feature = bg_feature.to(device=features.device, dtype=features.dtype)
    if bg_feature.numel() != features.shape[1]:
        raise ValueError(f"bg_feature must have {features.shape[1]} channels, got {bg_feature.numel()}")

    screenspace_points = _make_screenspace_points(pc, detach_geometry=detach_geometry)
    rasterizer = _make_rasterizer(viewpoint_camera, pipe, bg_feature, pc, scaling_modifier)

    scales, rotations, cov3D_precomp = _geometry_inputs(pc, pipe, scaling_modifier, detach_geometry=detach_geometry)
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
        "render": rendered_features,
        "viewspace_points": screenspace_points,
        "visibility_filter": (radii > 0).nonzero(),
        "radii": radii,
        "depth": depth_image,
    }


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
