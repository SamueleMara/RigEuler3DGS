"""Local affected-pixel support for component hypothesis tests."""

from __future__ import annotations

import torch

from utils.motion_correspondence import project_world_points


def gaussian_pixel_support_mask(
    camera,
    positions: torch.Tensor,
    gaussian_ids: torch.Tensor,
    *,
    radius_px: int = 4,
    min_points: int = 1,
) -> torch.Tensor:
    """A cheap conservative footprint mask for a set of Gaussians."""
    height, width = int(camera.image_height), int(camera.image_width)
    mask = torch.zeros((height, width), device=positions.device, dtype=torch.bool)
    ids = torch.as_tensor(gaussian_ids, device=positions.device).long().reshape(-1)
    if int(ids.numel()) == 0:
        return mask
    pixels, _depth, visible = project_world_points(positions.index_select(0, ids), camera)
    keep = visible.bool()
    if int(keep.sum()) < int(min_points):
        return mask
    pix = pixels[keep].round().long()
    r = max(1, int(radius_px))
    for x, y in pix.tolist():
        x0, x1 = max(0, x - r), min(width, x + r + 1)
        y0, y1 = max(0, y - r), min(height, y + r + 1)
        if x0 < x1 and y0 < y1:
            mask[y0:y1, x0:x1] = True
    return mask


def affected_union_mask(
    camera,
    static_positions: torch.Tensor,
    moving_positions: torch.Tensor,
    gaussian_ids: torch.Tensor,
    *,
    radius_px: int = 4,
    min_points: int = 1,
) -> torch.Tensor:
    return gaussian_pixel_support_mask(
        camera, static_positions, gaussian_ids, radius_px=radius_px, min_points=min_points
    ) | gaussian_pixel_support_mask(
        camera, moving_positions, gaussian_ids, radius_px=radius_px, min_points=min_points
    )
