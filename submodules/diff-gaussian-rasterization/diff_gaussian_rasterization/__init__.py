#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#

from typing import NamedTuple

import torch
import torch.nn as nn

from . import _C


def cpu_deep_copy_tuple(values):
    return tuple(
        value.detach().cpu().clone() if isinstance(value, torch.Tensor) else value
        for value in values
    )


def _empty(reference: torch.Tensor):
    return torch.empty(0, device=reference.device, dtype=reference.dtype)


class GaussianRasterizationSettings(NamedTuple):
    image_height: int
    image_width: int
    tanfovx: float
    tanfovy: float
    bg: torch.Tensor
    scale_modifier: float
    viewmatrix: torch.Tensor
    projmatrix: torch.Tensor
    sh_degree: int
    campos: torch.Tensor
    prefiltered: bool
    debug: bool
    antialiasing: bool


def rasterize_gaussians(
    means3D, means2D, sh, colors_precomp, opacities, scales, rotations,
    cov3Ds_precomp, raster_settings,
):
    return _RasterizeGaussians.apply(
        means3D, means2D, sh, colors_precomp, opacities, scales, rotations,
        cov3Ds_precomp, raster_settings,
    )


class _RasterizeGaussians(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx, means3D, means2D, sh, colors_precomp, opacities, scales,
        rotations, cov3Ds_precomp, raster_settings,
    ):
        args = (
            raster_settings.bg, means3D, colors_precomp, opacities, scales,
            rotations, raster_settings.scale_modifier, cov3Ds_precomp,
            raster_settings.viewmatrix, raster_settings.projmatrix,
            raster_settings.tanfovx, raster_settings.tanfovy,
            raster_settings.image_height, raster_settings.image_width, sh,
            raster_settings.sh_degree, raster_settings.campos,
            raster_settings.prefiltered, raster_settings.antialiasing,
            raster_settings.debug,
        )
        try:
            result = _C.rasterize_gaussians(*args)
        except Exception:
            if raster_settings.debug:
                torch.save(cpu_deep_copy_tuple(args), "snapshot_fw.dump")
            raise
        num_rendered, color, radii, geom, binning, image, invdepth = result
        ctx.raster_settings = raster_settings
        ctx.num_rendered = num_rendered
        ctx.save_for_backward(
            colors_precomp, means3D, opacities, scales, rotations,
            cov3Ds_precomp, radii, sh, geom, binning, image,
        )
        return color, radii, invdepth

    @staticmethod
    def backward(ctx, grad_color, _grad_radii, grad_invdepth):
        settings = ctx.raster_settings
        (
            colors, means3D, opacities, scales, rotations, cov3D, radii, sh,
            geom, binning, image,
        ) = ctx.saved_tensors
        if grad_invdepth is None:
            grad_invdepth = _empty(grad_color)
        args = (
            settings.bg, means3D, radii, colors, opacities, scales, rotations,
            settings.scale_modifier, cov3D, settings.viewmatrix,
            settings.projmatrix, settings.tanfovx, settings.tanfovy,
            grad_color, grad_invdepth, sh, settings.sh_degree, settings.campos,
            geom, ctx.num_rendered, binning, image, settings.antialiasing,
            settings.debug,
        )
        try:
            grads = _C.rasterize_gaussians_backward(*args)
        except Exception:
            if settings.debug:
                torch.save(cpu_deep_copy_tuple(args), "snapshot_bw.dump")
            raise
        (
            grad_means2D, grad_colors, grad_opacity, grad_means3D, grad_cov3D,
            grad_sh, grad_scales, grad_rotations,
        ) = grads
        return (
            grad_means3D, grad_means2D, grad_sh, grad_colors, grad_opacity,
            grad_scales, grad_rotations, grad_cov3D, None,
        )


def rasterize_gaussian_features(
    means3D, means2D, features, opacities, scales, rotations,
    cov3Ds_precomp, raster_settings,
):
    return _RasterizeGaussianFeatures.apply(
        means3D, means2D, features, opacities, scales, rotations,
        cov3Ds_precomp, raster_settings,
    )


class _RasterizeGaussianFeatures(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx, means3D, means2D, features, opacities, scales, rotations,
        cov3Ds_precomp, raster_settings,
    ):
        background = raster_settings.bg
        args = (
            background, means3D, features, opacities, scales, rotations,
            raster_settings.scale_modifier, cov3Ds_precomp,
            raster_settings.viewmatrix, raster_settings.projmatrix,
            raster_settings.tanfovx, raster_settings.tanfovy,
            raster_settings.image_height, raster_settings.image_width,
            raster_settings.campos, raster_settings.prefiltered,
            raster_settings.antialiasing, raster_settings.debug,
        )
        try:
            result = _C.rasterize_gaussian_features(*args)
        except Exception:
            if raster_settings.debug:
                torch.save(cpu_deep_copy_tuple(args), "snapshot_feature_fw.dump")
            raise
        num_rendered, output, radii, geom, binning, image, invdepth = result
        ctx.raster_settings = raster_settings
        ctx.num_rendered = num_rendered
        ctx.save_for_backward(
            means3D, features, opacities, scales, rotations, cov3Ds_precomp,
            radii, geom, binning, image,
        )
        return output, radii, invdepth

    @staticmethod
    def backward(ctx, grad_output, _grad_radii, grad_invdepth):
        settings = ctx.raster_settings
        (
            means3D, features, opacities, scales, rotations, cov3D, radii,
            geom, binning, image,
        ) = ctx.saved_tensors
        if grad_invdepth is None:
            grad_invdepth = _empty(grad_output)
        args = (
            settings.bg, means3D, radii, features, opacities, scales,
            rotations, settings.scale_modifier, cov3D, settings.viewmatrix,
            settings.projmatrix, settings.tanfovx, settings.tanfovy,
            grad_output, grad_invdepth, settings.campos, geom,
            ctx.num_rendered, binning, image, settings.antialiasing,
            settings.debug,
        )
        try:
            grads = _C.rasterize_gaussian_features_backward(*args)
        except Exception:
            if settings.debug:
                torch.save(cpu_deep_copy_tuple(args), "snapshot_feature_bw.dump")
            raise
        (
            grad_means2D, grad_features, grad_opacity, grad_means3D,
            grad_cov3D, _grad_sh, grad_scales, grad_rotations,
        ) = grads
        return (
            grad_means3D, grad_means2D, grad_features, grad_opacity,
            grad_scales, grad_rotations, grad_cov3D, None,
        )


def rasterize_gaussian_positions(
    query_means3D, means2D, target_positions, opacities, scales, rotations,
    cov3Ds_precomp, raster_settings,
):
    return _RasterizeGaussianPositions.apply(
        query_means3D, means2D, target_positions, opacities, scales, rotations,
        cov3Ds_precomp, raster_settings,
    )


class _RasterizeGaussianPositions(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx, query_means3D, means2D, target_positions, opacities, scales,
        rotations, cov3Ds_precomp, raster_settings,
    ):
        args = (
            query_means3D, target_positions, opacities, scales, rotations,
            raster_settings.scale_modifier, cov3Ds_precomp,
            raster_settings.viewmatrix, raster_settings.projmatrix,
            raster_settings.tanfovx, raster_settings.tanfovy,
            raster_settings.image_height, raster_settings.image_width,
            raster_settings.campos, raster_settings.prefiltered,
            raster_settings.antialiasing, raster_settings.debug,
        )
        try:
            result = _C.rasterize_gaussian_positions(*args)
        except Exception:
            if raster_settings.debug:
                torch.save(cpu_deep_copy_tuple(args), "snapshot_position_fw.dump")
            raise
        num_rendered, output, radii, geom, binning, image, invdepth = result
        ctx.raster_settings = raster_settings
        ctx.num_rendered = num_rendered
        ctx.save_for_backward(
            query_means3D, target_positions, opacities, scales, rotations,
            cov3Ds_precomp, radii, geom, binning, image,
        )
        return output, radii, invdepth

    @staticmethod
    def backward(ctx, grad_output, _grad_radii, grad_invdepth):
        settings = ctx.raster_settings
        (
            query_means3D, target_positions, opacities, scales, rotations,
            cov3D, radii, geom, binning, image,
        ) = ctx.saved_tensors
        if grad_invdepth is None:
            grad_invdepth = _empty(grad_output)
        args = (
            query_means3D, radii, target_positions, opacities, scales,
            rotations, settings.scale_modifier, cov3D, settings.viewmatrix,
            settings.projmatrix, settings.tanfovx, settings.tanfovy,
            grad_output, grad_invdepth, settings.campos, geom,
            ctx.num_rendered, binning, image, settings.antialiasing,
            settings.debug,
        )
        try:
            grads = _C.rasterize_gaussian_positions_backward(*args)
        except Exception:
            if settings.debug:
                torch.save(cpu_deep_copy_tuple(args), "snapshot_position_bw.dump")
            raise
        (
            grad_means2D, grad_target_positions, grad_opacity,
            grad_query_means3D, grad_cov3D, _grad_sh, grad_scales,
            grad_rotations,
        ) = grads
        return (
            grad_query_means3D, grad_means2D, grad_target_positions,
            grad_opacity, grad_scales, grad_rotations, grad_cov3D, None,
        )


def rasterize_gaussian_flow_stats(
    means3D, observed_flow, observed_confidence, group_ids, num_groups,
    opacities, scales, rotations, cov3Ds_precomp, raster_settings,
    delta_time: float = 1.0, min_observation_weight: float = 0.0,
):
    args = (
        means3D, observed_flow, observed_confidence, group_ids, int(num_groups),
        opacities, scales, rotations, raster_settings.scale_modifier,
        cov3Ds_precomp, raster_settings.viewmatrix, raster_settings.projmatrix,
        raster_settings.tanfovx, raster_settings.tanfovy,
        raster_settings.image_height, raster_settings.image_width,
        raster_settings.campos, raster_settings.prefiltered,
        raster_settings.antialiasing, float(delta_time),
        float(min_observation_weight), raster_settings.debug,
    )
    try:
        return _C.rasterize_gaussian_flow_stats(*args)
    except Exception:
        if raster_settings.debug:
            torch.save(cpu_deep_copy_tuple(args), "snapshot_flow_stats_fw.dump")
        raise


def rasterize_gaussian_pixel_feature_stats(
    means3D, pixel_features, pixel_confidence, group_ids, num_groups,
    opacities, scales, rotations, cov3Ds_precomp, raster_settings,
    min_observation_weight: float = 0.0,
):
    args = (
        means3D, pixel_features, pixel_confidence, group_ids, int(num_groups),
        opacities, scales, rotations, raster_settings.scale_modifier,
        cov3Ds_precomp, raster_settings.viewmatrix, raster_settings.projmatrix,
        raster_settings.tanfovx, raster_settings.tanfovy,
        raster_settings.image_height, raster_settings.image_width,
        raster_settings.campos, raster_settings.prefiltered,
        raster_settings.antialiasing, float(min_observation_weight),
        raster_settings.debug,
    )
    try:
        return _C.rasterize_gaussian_pixel_feature_stats(*args)
    except Exception:
        if raster_settings.debug:
            torch.save(cpu_deep_copy_tuple(args), "snapshot_pixel_feature_stats_fw.dump")
        raise


class GaussianRasterizer(nn.Module):
    def __init__(self, raster_settings):
        super().__init__()
        self.raster_settings = raster_settings

    def markVisible(self, positions):
        with torch.no_grad():
            return _C.mark_visible(
                positions,
                self.raster_settings.viewmatrix,
                self.raster_settings.projmatrix,
            )

    @staticmethod
    def _geometry_inputs(means3D, scales, rotations, cov3D_precomp):
        if ((scales is None or rotations is None) and cov3D_precomp is None) or (
            (scales is not None or rotations is not None) and cov3D_precomp is not None
        ):
            raise ValueError(
                "Provide exactly one of scales/rotations or cov3D_precomp"
            )
        empty = _empty(means3D)
        return (
            empty if scales is None else scales,
            empty if rotations is None else rotations,
            empty if cov3D_precomp is None else cov3D_precomp,
        )

    def forward(
        self, means3D, means2D, opacities, shs=None, colors_precomp=None,
        scales=None, rotations=None, cov3D_precomp=None, dc=None,
    ):
        if dc is not None:
            shs = dc if shs is None else torch.cat((dc, shs), dim=1)
        if (shs is None) == (colors_precomp is None):
            raise ValueError("Provide exactly one of shs or colors_precomp")
        scales, rotations, cov3D_precomp = self._geometry_inputs(
            means3D, scales, rotations, cov3D_precomp
        )
        empty = _empty(means3D)
        return rasterize_gaussians(
            means3D, means2D,
            empty if shs is None else shs,
            empty if colors_precomp is None else colors_precomp,
            opacities, scales, rotations, cov3D_precomp,
            self.raster_settings,
        )

    def render_features(
        self, means3D, means2D, features, opacities,
        scales=None, rotations=None, cov3D_precomp=None,
    ):
        if features.ndim != 2 or features.shape[0] != means3D.shape[0]:
            raise ValueError("features must have shape [N,C]")
        if self.raster_settings.bg.numel() != features.shape[1]:
            raise ValueError("raster background channels must match features")
        scales, rotations, cov3D_precomp = self._geometry_inputs(
            means3D, scales, rotations, cov3D_precomp
        )
        return rasterize_gaussian_features(
            means3D, means2D, features, opacities, scales, rotations,
            cov3D_precomp, self.raster_settings,
        )

    def render_positions(
        self, means3D, means2D, target_positions, opacities,
        scales=None, rotations=None, cov3D_precomp=None,
    ):
        if target_positions.ndim != 2 or target_positions.shape != means3D.shape:
            raise ValueError("target_positions must have shape [N,3]")
        if self.raster_settings.bg.numel() != 4:
            raise ValueError("position rasterization requires a four-channel background")
        scales, rotations, cov3D_precomp = self._geometry_inputs(
            means3D, scales, rotations, cov3D_precomp
        )
        return rasterize_gaussian_positions(
            means3D, means2D, target_positions, opacities, scales, rotations,
            cov3D_precomp, self.raster_settings,
        )

    @torch.no_grad()
    def accumulate_flow_stats(
        self, means3D, observed_flow, observed_confidence, opacities,
        scales=None, rotations=None, cov3D_precomp=None, group_ids=None,
        num_groups=None, delta_time: float = 1.0,
        min_observation_weight: float = 0.0,
    ):
        if observed_flow.ndim != 3 or observed_flow.shape[0] != 2:
            raise ValueError("observed_flow must have shape [2,H,W]")
        if observed_confidence.ndim == 2:
            observed_confidence = observed_confidence[None]
        if observed_confidence.ndim != 3 or observed_confidence.shape[0] != 1:
            raise ValueError("observed_confidence must have shape [1,H,W] or [H,W]")
        if tuple(observed_flow.shape[-2:]) != (
            self.raster_settings.image_height, self.raster_settings.image_width,
        ):
            raise ValueError("observed_flow resolution must match raster settings")
        if tuple(observed_confidence.shape[-2:]) != tuple(observed_flow.shape[-2:]):
            raise ValueError("observed_confidence resolution must match observed_flow")
        scales, rotations, cov3D_precomp = self._geometry_inputs(
            means3D, scales, rotations, cov3D_precomp
        )
        if group_ids is None:
            group_ids = torch.empty(0, device=means3D.device, dtype=torch.long)
            num_groups = means3D.shape[0]
        else:
            group_ids = group_ids.to(device=means3D.device, dtype=torch.long).contiguous()
            if group_ids.numel() != means3D.shape[0]:
                raise ValueError("group_ids must have one id per Gaussian")
            if num_groups is None:
                valid = group_ids[group_ids >= 0]
                num_groups = int(valid.max().item() + 1) if valid.numel() else 0
        return rasterize_gaussian_flow_stats(
            means3D.contiguous(),
            observed_flow.to(device=means3D.device, dtype=torch.float32).contiguous(),
            observed_confidence.to(device=means3D.device, dtype=torch.float32).contiguous(),
            group_ids,
            int(num_groups or 0),
            opacities,
            scales,
            rotations,
            cov3D_precomp,
            self.raster_settings,
            delta_time=delta_time,
            min_observation_weight=min_observation_weight,
        )

    @torch.no_grad()
    def accumulate_pixel_features_to_gaussians(
        self, means3D, pixel_features, pixel_confidence, opacities,
        scales=None, rotations=None, cov3D_precomp=None, group_ids=None,
        num_groups=None, min_observation_weight: float = 0.0,
    ):
        """Accumulate pixel evidence into Gaussians/groups.

        Returns ``(rendered, feature_sum, valid_weight, count, raw_support,
        visible_mass, radii)``. ``raw_support`` is ``sum(alpha)`` over the full
        sorted tile range, while ``visible_mass`` is the exact RGB ownership
        ``sum(T*alpha)``.  ``valid_weight`` additionally includes pixel
        confidence and is therefore observation-valid evidence.
        """
        if pixel_features.ndim != 3 or pixel_features.shape[0] < 1:
            raise ValueError("pixel_features must have shape [C,H,W]")
        if pixel_confidence.ndim == 2:
            pixel_confidence = pixel_confidence[None]
        if pixel_confidence.ndim != 3 or pixel_confidence.shape[0] != 1:
            raise ValueError("pixel_confidence must have shape [1,H,W] or [H,W]")
        expected = (self.raster_settings.image_height, self.raster_settings.image_width)
        if tuple(pixel_features.shape[-2:]) != expected or tuple(pixel_confidence.shape[-2:]) != expected:
            raise ValueError("pixel feature/confidence resolution must match raster settings")
        scales, rotations, cov3D_precomp = self._geometry_inputs(
            means3D, scales, rotations, cov3D_precomp
        )
        if group_ids is None:
            group_ids = torch.empty(0, device=means3D.device, dtype=torch.long)
            num_groups = means3D.shape[0]
        else:
            group_ids = group_ids.to(device=means3D.device, dtype=torch.long).contiguous()
            if group_ids.numel() != means3D.shape[0]:
                raise ValueError("group_ids must have one id per Gaussian")
            if num_groups is None:
                valid = group_ids[group_ids >= 0]
                num_groups = int(valid.max().item() + 1) if valid.numel() else 0
        return rasterize_gaussian_pixel_feature_stats(
            means3D.contiguous(),
            pixel_features.to(device=means3D.device, dtype=torch.float32).contiguous(),
            pixel_confidence.to(device=means3D.device, dtype=torch.float32).contiguous(),
            group_ids, int(num_groups or 0), opacities, scales, rotations,
            cov3D_precomp, self.raster_settings,
            min_observation_weight=min_observation_weight,
        )


# ---------------------------------------------------------------------------
# Sparse alpha-aware IEKF observation operators.
# CODEX_IMPLEMENT_SPARSE_ALPHA_AWARE_IEKF.md section 19.1.
#
# These are deliberately NOT torch.autograd.Functions: the estimator must use
# analytic Jacobians, and wrapping them in autograd would invite exactly the
# loss.backward() path section 36 forbids. They are plain tensor-in/tensor-out
# calls operating at a frozen linearization.
# ---------------------------------------------------------------------------
from dataclasses import dataclass
from typing import Optional


@dataclass
class IEKFLinearizationCache:
    """One frozen linearization (section 20.2/20.3).

    Everything the observation operators need is captured here, so sorting,
    tile membership, alpha values and Jacobian coefficients cannot change
    inside a PCG solve -- which would destroy conjugacy (section 20.3).
    """

    mu_c: torch.Tensor
    depth: torch.Tensor
    u: torch.Tensor
    Jp: torch.Tensor
    sigma_c: torch.Tensor
    conic: torch.Tensor
    opacity: torch.Tensor
    radii: torch.Tensor
    clamp_flags: torch.Tensor
    ranges: torch.Tensor
    point_list: torch.Tensor
    num_rendered: int
    # geometry/camera the cache was built at, needed by the tangent operators
    means3D: torch.Tensor
    viewmatrix: torch.Tensor
    projmatrix: torch.Tensor
    tanfovx: float
    tanfovy: float
    image_height: int
    image_width: int
    colors: torch.Tensor
    background: torch.Tensor
    eps_depth: float
    debug: bool = False

    @property
    def num_gaussians(self) -> int:
        return int(self.mu_c.shape[0])

    def _args(self):
        return (self.mu_c, self.depth, self.u, self.Jp, self.sigma_c, self.conic,
                self.opacity, self.radii, self.clamp_flags, self.ranges, self.point_list)


def iekf_build_linearization_cache(
    means3D, scales, rotations, opacity, colors, background,
    viewmatrix, projmatrix, tanfovx, tanfovy, image_height, image_width,
    scale_modifier: float = 1.0,
    cov3D_precomp: Optional[torch.Tensor] = None,
    antialiasing: bool = False,
    eps_depth: float = 1e-6,
    debug: bool = False,
) -> IEKFLinearizationCache:
    """Preprocess + binning + sort at the current state (section 21 Kernel A)."""
    if cov3D_precomp is None:
        cov3D_precomp = torch.empty(0, device=means3D.device, dtype=means3D.dtype)
    (mu_c, depth, u, Jp, sigma_c, conic, op, radii, clamp_flags, ranges,
     point_list, num_rendered) = _C.iekf_build_cache(
        means3D, scales, scale_modifier, rotations, opacity, cov3D_precomp,
        viewmatrix, projmatrix, tanfovx, tanfovy, image_height, image_width,
        antialiasing, debug)
    return IEKFLinearizationCache(
        mu_c=mu_c, depth=depth, u=u, Jp=Jp, sigma_c=sigma_c, conic=conic,
        opacity=op, radii=radii, clamp_flags=clamp_flags, ranges=ranges,
        point_list=point_list, num_rendered=num_rendered,
        means3D=means3D, viewmatrix=viewmatrix, projmatrix=projmatrix,
        tanfovx=tanfovx, tanfovy=tanfovy,
        image_height=image_height, image_width=image_width,
        colors=colors, background=background, eps_depth=eps_depth, debug=debug)


def iekf_render(cache: IEKFLinearizationCache):
    """h(x) at the frozen linearization.

    Returns (RGB, normalized depth, A, final T, n_contrib). ``n_contrib`` is
    the per-pixel count of contributors the front-to-back loop actually
    composited before transmittance saturated; the reverse passes must be
    handed it so they stop at the same place, or H^T transposes contributors H
    never applied and the adjoint identity fails on any dense scene.
    """
    return _C.iekf_render(
        *cache._args(), cache.colors, cache.background,
        cache.image_height, cache.image_width, cache.eps_depth, cache.debug)


def iekf_apply_H(cache: IEKFLinearizationCache, pose_delta: torch.Tensor):
    """H v for a [P,6] pose tangent (section 10). Returns (dRGB, dDepth)."""
    return _C.iekf_apply_H(
        *cache._args(), cache.colors, cache.background, cache.means3D,
        cache.viewmatrix, cache.projmatrix, cache.tanfovx, cache.tanfovy,
        cache.image_height, cache.image_width, cache.eps_depth,
        pose_delta.contiguous(), cache.debug)


def iekf_apply_HT(cache: IEKFLinearizationCache, z_color, z_depth,
                  final_T, final_A, final_D, n_contrib):
    """H^T z (section 11). Returns the [P,6] pose adjoint."""
    return _C.iekf_apply_HT(
        *cache._args(), cache.colors, cache.background, cache.means3D,
        cache.viewmatrix, cache.projmatrix, cache.tanfovx, cache.tanfovy,
        cache.image_height, cache.image_width, cache.eps_depth,
        z_color.contiguous(), z_depth.contiguous(),
        n_contrib.contiguous(),
        final_T.contiguous(), final_A.contiguous(), final_D.contiguous(),
        cache.debug)


def iekf_build_fisher(cache: IEKFLinearizationCache, precision_color,
                      precision_depth, final_T, final_A, final_D, n_contrib):
    """Per-Gaussian 6x6 Fisher blocks as [P,21] upper triangles (section 16)."""
    return _C.iekf_build_fisher(
        *cache._args(), cache.colors, cache.background, cache.means3D,
        cache.viewmatrix, cache.projmatrix, cache.tanfovx, cache.tanfovy,
        cache.image_height, cache.image_width, cache.eps_depth,
        precision_color.contiguous(), precision_depth.contiguous(),
        n_contrib.contiguous(),
        final_T.contiguous(), final_A.contiguous(), final_D.contiguous(),
        cache.debug)


def iekf_fisher_to_dense(fisher21: torch.Tensor) -> torch.Tensor:
    """[P,21] upper triangle -> [P,6,6] symmetric."""
    p = fisher21.shape[0]
    out = fisher21.new_zeros((p, 6, 6))
    t = 0
    for a in range(6):
        for b in range(a, 6):
            out[:, a, b] = fisher21[:, t]
            out[:, b, a] = fisher21[:, t]
            t += 1
    return out
