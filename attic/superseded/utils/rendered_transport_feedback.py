"""Rendered feedback for the closed-loop physical transport.

The functions here reuse the repository's projection and feature rasterizer:
per-Gaussian 3-D transport becomes dense image-space flow through the same
source-frame Gaussian ownership used by the optical-flow losses.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F

from utils.motion_correspondence import project_world_points


_EPS = 1.0e-8


@dataclass
class RenderedTransportFeedback:
    physical_flow: torch.Tensor | None
    physical_flow_weight: torch.Tensor | None
    fm_flow_loss: torch.Tensor
    observed_flow_loss: torch.Tensor
    target_render_loss: torch.Tensor
    per_gaussian_fm_energy: torch.Tensor
    per_gaussian_obs_energy: torch.Tensor
    per_gaussian_render_energy: torch.Tensor
    fm_reliability: torch.Tensor
    obs_reliability: torch.Tensor
    render_reliability: torch.Tensor
    valid_fraction: float
    diagnostics: dict


def _resize_chw(value: torch.Tensor, channels: int, height: int, width: int, device) -> torch.Tensor:
    tensor = torch.as_tensor(value, device=device, dtype=torch.float32)
    if tensor.ndim == 2:
        tensor = tensor[None]
    if tensor.ndim != 3 or int(tensor.shape[0]) != int(channels):
        raise ValueError(f"expected [{channels},H,W], got {tuple(tensor.shape)}")
    if tuple(tensor.shape[-2:]) != (height, width):
        tensor = F.interpolate(
            tensor[None], size=(height, width), mode="bilinear", align_corners=True
        )[0]
    return tensor.contiguous()


def _full_with_foreground(canonical_all: torch.Tensor, foreground_idx: torch.Tensor, foreground_xyz: torch.Tensor) -> torch.Tensor:
    full = torch.as_tensor(canonical_all).detach().clone()
    idx = torch.as_tensor(foreground_idx, device=full.device).long()
    full.index_copy_(0, idx, torch.as_tensor(foreground_xyz, device=full.device, dtype=full.dtype))
    return full


def _rasterized_projected_flow(
    *,
    source_camera,
    target_camera,
    gaussians,
    pipe,
    query_xyz: torch.Tensor,
    transported_xyz: torch.Tensor,
):
    from gaussian_renderer import render_query_feature_field

    source_px, _, source_valid = project_world_points(query_xyz, source_camera)
    target_px, _, target_valid = project_world_points(transported_xyz, target_camera)
    valid = source_valid & target_valid
    marker = valid[:, None].to(query_xyz)
    flow = (target_px - source_px).to(query_xyz) * marker
    features = torch.cat((flow, marker), dim=-1)
    package = render_query_feature_field(
        viewpoint_camera=source_camera,
        pc=gaussians,
        pipe=pipe,
        query_positions=query_xyz,
        features=features,
    )
    rendered = package["render"]
    opacity = rendered[2:3].clamp_min(0.0)
    dense_flow = rendered[:2] / opacity.clamp_min(1.0e-6)
    return dense_flow, opacity, valid


def _backproject_pixel_error(
    *,
    source_camera,
    gaussians,
    pipe,
    query_xyz: torch.Tensor,
    foreground_idx: torch.Tensor,
    residual: torch.Tensor,
    confidence: torch.Tensor,
):
    from gaussian_renderer import accumulate_pixel_features_to_gaussians

    gathered = accumulate_pixel_features_to_gaussians(
        source_camera,
        gaussians,
        pipe,
        residual[None],
        confidence,
        query_positions=query_xyz,
    )
    idx = torch.as_tensor(foreground_idx, device=query_xyz.device).long()
    support = gathered["weight_sum"].to(query_xyz)[idx]
    energy = gathered["feature_sum"].to(query_xyz)[idx, 0] / support.clamp_min(_EPS)
    return energy, support


@torch.no_grad()
def render_physical_transport_feedback(
    *,
    gaussians,
    pipe,
    canonical_all: torch.Tensor,
    foreground_idx: torch.Tensor,
    candidate_fg_positions: torch.Tensor,
    source_camera,
    target_camera,
    fm_displacement_fg: torch.Tensor,
    fm_confidence_fg: torch.Tensor,
    observed_flow: torch.Tensor | None = None,
    observed_confidence: torch.Tensor | None = None,
    background: torch.Tensor | None = None,
    min_support: float = 1.0e-4,
    target_render_weight: float = 0.0,
) -> RenderedTransportFeedback:
    """Render candidate physical transports and return per-label energies.

    ``candidate_fg_positions`` is ``[K+1,N_F,3]`` for one target frame.  Label
    0 should already be canonical/static.  The output energies are ``[N_F,K+1]``.
    """
    canonical = torch.as_tensor(canonical_all).detach()
    device, dtype = canonical.device, canonical.dtype
    idx = torch.as_tensor(foreground_idx, device=device).long()
    candidates = torch.as_tensor(candidate_fg_positions, device=device, dtype=dtype)
    if candidates.ndim != 3 or candidates.shape[1:] != (idx.numel(), 3):
        raise ValueError("candidate_fg_positions must be [K+1,N_F,3]")
    labels = int(candidates.shape[0])
    n = int(idx.numel())
    height, width = int(source_camera.image_height), int(source_camera.image_width)

    fm_full = canonical.clone()
    fm_full.index_copy_(
        0,
        idx,
        canonical.index_select(0, idx)
        + torch.as_tensor(fm_displacement_fg, device=device, dtype=dtype),
    )
    fm_flow, fm_weight, fm_project_valid = _rasterized_projected_flow(
        source_camera=source_camera,
        target_camera=target_camera,
        gaussians=gaussians,
        pipe=pipe,
        query_xyz=canonical,
        transported_xyz=fm_full,
    )
    fm_conf = torch.as_tensor(fm_confidence_fg, device=device, dtype=dtype).reshape(-1)
    if fm_conf.numel() != n:
        raise ValueError("fm_confidence_fg must match foreground count")

    observed_flow_t = None
    observed_conf_t = None
    if observed_flow is not None and observed_confidence is not None:
        observed_flow_t = _resize_chw(observed_flow, 2, height, width, device)
        observed_conf_t = _resize_chw(observed_confidence, 1, height, width, device)

    fm_energy = torch.zeros((n, labels), device=device, dtype=dtype)
    obs_energy = torch.zeros_like(fm_energy)
    render_energy = torch.zeros_like(fm_energy)
    fm_reliability = torch.zeros_like(fm_energy)
    obs_reliability = torch.zeros_like(fm_energy)
    render_reliability = torch.zeros_like(fm_energy)
    scalar_fm, scalar_obs, scalar_render = [], [], []
    first_flow = first_weight = None

    for label in range(labels):
        physical_full = _full_with_foreground(canonical, idx, candidates[label])
        physical_flow, physical_weight, physical_valid = _rasterized_projected_flow(
            source_camera=source_camera,
            target_camera=target_camera,
            gaussians=gaussians,
            pipe=pipe,
            query_xyz=canonical,
            transported_xyz=physical_full,
        )
        if first_flow is None:
            first_flow, first_weight = physical_flow.detach().cpu(), physical_weight.detach().cpu()

        support = torch.minimum(physical_weight, fm_weight)
        residual = (physical_flow - fm_flow).norm(dim=0)
        energy, mass = _backproject_pixel_error(
            source_camera=source_camera,
            gaussians=gaussians,
            pipe=pipe,
            query_xyz=canonical,
            foreground_idx=idx,
            residual=residual,
            confidence=support,
        )
        fm_energy[:, label] = fm_conf * energy
        fm_reliability[:, label] = fm_conf * (mass > float(min_support)).to(dtype)
        valid = support > float(min_support)
        if bool(valid.any()):
            scalar_fm.append(float(residual[valid[0]].mean()))

        if observed_flow_t is not None and observed_conf_t is not None:
            support_obs = torch.minimum(physical_weight, observed_conf_t.clamp(0.0, 1.0))
            residual_obs = (physical_flow - observed_flow_t.to(dtype)).norm(dim=0)
            energy, mass = _backproject_pixel_error(
                source_camera=source_camera,
                gaussians=gaussians,
                pipe=pipe,
                query_xyz=canonical,
                foreground_idx=idx,
                residual=residual_obs,
                confidence=support_obs,
            )
            obs_energy[:, label] = energy
            obs_reliability[:, label] = (mass > float(min_support)).to(dtype)
            valid_obs = support_obs > float(min_support)
            if bool(valid_obs.any()):
                scalar_obs.append(float(residual_obs[valid_obs[0]].mean()))

        if target_render_weight > 0.0:
            from gaussian_renderer import accumulate_pixel_features_to_gaussians, render

            bg = (
                torch.zeros(3, device=device, dtype=dtype)
                if background is None
                else torch.as_tensor(background, device=device, dtype=dtype)
            )
            rendered = render(target_camera, gaussians, pipe, bg, means3D_override=physical_full)["render"]
            target = target_camera.original_image.to(device=device, dtype=dtype).clamp(0.0, 1.0)
            residual_rgb = (rendered - target).abs().mean(0, keepdim=True)
            gathered = accumulate_pixel_features_to_gaussians(
                target_camera,
                gaussians,
                pipe,
                residual_rgb,
                torch.ones_like(residual_rgb),
                query_positions=physical_full,
            )
            mass = gathered["weight_sum"].to(dtype)[idx]
            render_energy[:, label] = gathered["feature_sum"].to(dtype)[idx, 0] / mass.clamp_min(_EPS)
            render_reliability[:, label] = (mass > float(min_support)).to(dtype)
            if bool((mass > float(min_support)).any()):
                scalar_render.append(float(render_energy[mass > float(min_support), label].mean()))

    valid_fraction = float((fm_reliability > 0).any(-1).float().mean()) if n else 0.0
    zero = torch.zeros((), device=device, dtype=dtype)
    return RenderedTransportFeedback(
        physical_flow=first_flow,
        physical_flow_weight=first_weight,
        fm_flow_loss=torch.tensor(sum(scalar_fm) / max(len(scalar_fm), 1), device=device, dtype=dtype) if scalar_fm else zero,
        observed_flow_loss=torch.tensor(sum(scalar_obs) / max(len(scalar_obs), 1), device=device, dtype=dtype) if scalar_obs else zero,
        target_render_loss=torch.tensor(sum(scalar_render) / max(len(scalar_render), 1), device=device, dtype=dtype) if scalar_render else zero,
        per_gaussian_fm_energy=fm_energy,
        per_gaussian_obs_energy=obs_energy,
        per_gaussian_render_energy=render_energy,
        fm_reliability=fm_reliability,
        obs_reliability=obs_reliability,
        render_reliability=render_reliability,
        valid_fraction=valid_fraction,
        diagnostics={
            "labels": labels,
            "num_foreground": n,
            "valid_fraction": valid_fraction,
            "fm_flow_loss": float(scalar_fm[-1]) if scalar_fm else None,
            "observed_flow_loss": float(scalar_obs[-1]) if scalar_obs else None,
            "target_render_loss": float(scalar_render[-1]) if scalar_render else None,
            "fm_project_valid_fraction": float(fm_project_valid.index_select(0, idx).float().mean()),
        },
    )
