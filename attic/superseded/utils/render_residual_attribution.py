"""Render residuals attributed back to Gaussians through the CUDA rasterizer."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch


@dataclass
class RenderAttributionConfig:
    cameras_per_frame: int = 2
    min_support: float = 1.0e-5
    use_depth: bool = True


@torch.no_grad()
def attribute_render_residuals(
    *,
    gaussians,
    pipe,
    camera_groups,
    white_background: bool,
    foreground_idx: torch.Tensor,
    full_positions: torch.Tensor,
    cfg: RenderAttributionConfig,
) -> dict[str, torch.Tensor | dict[str, Any]]:
    """Render a physical scene, then gather RGB/depth residuals per Gaussian."""
    from gaussian_renderer import accumulate_pixel_features_to_gaussians, render

    device = full_positions.device
    dtype = full_positions.dtype
    idx = foreground_idx.to(device=device).long()
    background = torch.tensor([1.0, 1.0, 1.0] if white_background else [0.0, 0.0, 0.0], device=device, dtype=dtype)
    render_residual = torch.zeros(idx.numel(), device=device, dtype=dtype)
    depth_residual = torch.zeros_like(render_residual)
    visibility = torch.zeros_like(render_residual)
    count = 0
    diagnostics = []
    frame_index = max(0, len(camera_groups) - 1)
    for camera in list(camera_groups[frame_index])[: max(1, int(cfg.cameras_per_frame))]:
        output = render(camera, gaussians, pipe, background, means3D_override=full_positions)
        image = output["render"].clamp(0.0, 1.0)
        target = camera.original_image.to(image.device).clamp(0.0, 1.0)
        rgb = (image - target).abs().mean(0, keepdim=True)
        depth = torch.zeros_like(rgb)
        if bool(cfg.use_depth) and hasattr(camera, "invdepthmap") and hasattr(camera, "depth_mask"):
            target_depth = camera.invdepthmap.to(image.device)
            depth_mask = camera.depth_mask.to(image.device).float()
            if target_depth.ndim == 2:
                target_depth = target_depth[None]
            if depth_mask.ndim == 2:
                depth_mask = depth_mask[None]
            rendered_depth = output.get("depth")
            if rendered_depth is not None:
                if rendered_depth.ndim == 2:
                    rendered_depth = rendered_depth[None]
                depth = (rendered_depth - target_depth).abs() * depth_mask
        pixel = torch.cat((rgb, depth), dim=0)
        gathered = accumulate_pixel_features_to_gaussians(
            camera, gaussians, pipe, pixel, torch.ones_like(rgb),
            query_positions=full_positions,
        )
        weight = gathered["weight_sum"].to(device=device, dtype=dtype)
        fg_weight = weight.index_select(0, idx)
        mean = gathered["feature_mean"].to(device=device, dtype=dtype).index_select(0, idx)
        valid = fg_weight > float(cfg.min_support)
        render_residual += torch.where(valid, mean[:, 0], torch.zeros_like(fg_weight))
        depth_residual += torch.where(valid, mean[:, 1], torch.zeros_like(fg_weight))
        visibility += fg_weight
        diagnostics.append({
            "image_name": str(getattr(camera, "image_name", "")),
            "visible_fg": int(valid.sum()),
            "mean_rgb_residual": float(rgb.mean().detach().cpu()),
            "mean_depth_residual": float(depth.mean().detach().cpu()),
        })
        count += 1
    denom = max(count, 1)
    return {
        "render_residual": render_residual / denom,
        "depth_residual": depth_residual / denom,
        "visibility_mass": visibility / denom,
        "observable": visibility > float(cfg.min_support),
        "diagnostics": {"cameras": count, "per_camera": diagnostics},
    }
