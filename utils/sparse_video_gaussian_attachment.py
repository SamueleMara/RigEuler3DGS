"""Sparse RGB-D video tracks -> endpoint Gaussian support.

The CUDA rasterizer does not expose a dense pixel-by-Gaussian ownership matrix.
This module therefore builds only the sparse rows needed by propagated video
tracks: project visible Gaussians into the endpoint camera, attach each tracked
pixel to nearby projected Gaussians, and form sparse source/target support.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from utils.motion_correspondence import project_world_points
from utils.video_endpoint_transport import SparseVideoEndpointSupport


@dataclass
class PixelGaussianRows:
    ids: torch.Tensor      # [P, K]
    weights: torch.Tensor  # [P, K]
    valid: torch.Tensor    # [P, K]


def _sample_flat(image: torch.Tensor | None, pixels: torch.Tensor) -> torch.Tensor | None:
    if image is None:
        return None
    value = torch.as_tensor(image).detach().float()
    if value.ndim == 3:
        value = value[0]
    flat = value.reshape(-1)
    p = torch.as_tensor(pixels).detach().cpu().long().reshape(-1).clamp(0, flat.numel() - 1)
    return flat[p]


@torch.no_grad()
def attach_pixels_to_projected_gaussians(
    xyz: torch.Tensor,
    camera,
    pixel_indices: torch.Tensor,
    *,
    topk: int = 8,
    max_pixel_radius: float = 6.0,
    depth_map: torch.Tensor | None = None,
    depth_sigma: float = 0.03,
    depth_is_inverse: bool = True,
    chunk_size: int = 512,
) -> PixelGaussianRows:
    points = torch.as_tensor(xyz).detach().cpu().float()
    pixels = torch.as_tensor(pixel_indices).detach().cpu().long().reshape(-1)
    k = max(1, int(topk))
    ids = torch.full((pixels.numel(), k), -1, dtype=torch.long)
    weights = torch.zeros((pixels.numel(), k), dtype=torch.float32)
    valid_out = torch.zeros((pixels.numel(), k), dtype=torch.bool)
    if pixels.numel() == 0 or points.numel() == 0:
        return PixelGaussianRows(ids, weights, valid_out)

    projected, depth, visible = project_world_points(points, camera)
    projected = projected.detach().cpu().float()
    depth = depth.detach().cpu().float()
    visible = visible.detach().cpu().bool()
    visible_ids = torch.nonzero(visible, as_tuple=False).flatten()
    if visible_ids.numel() == 0:
        return PixelGaussianRows(ids, weights, valid_out)
    proj = projected.index_select(0, visible_ids)
    z = depth.index_select(0, visible_ids)
    width = int(camera.image_width)
    pix_xy = torch.stack((pixels % width, pixels // width), dim=-1).float()
    observed_depth = _sample_flat(depth_map, pixels)

    for start in range(0, int(pixels.numel()), max(1, int(chunk_size))):
        end = min(start + int(chunk_size), int(pixels.numel()))
        d = torch.cdist(pix_xy[start:end], proj)
        kk = min(k, int(proj.shape[0]))
        best = torch.topk(d, k=kk, largest=False, dim=1)
        local_ids = visible_ids.index_select(0, best.indices.reshape(-1)).reshape(best.indices.shape)
        score = best.values / max(float(max_pixel_radius), 1.0e-6)
        keep = best.values <= float(max_pixel_radius)
        if observed_depth is not None:
            obs = observed_depth[start:end, None]
            cand_z = z.index_select(0, best.indices.reshape(-1)).reshape(best.indices.shape)
            valid_obs = torch.isfinite(obs) & (obs > 0)
            if bool(depth_is_inverse):
                # The camera stores `invdepthmap`, i.e. 1/z, while the
                # projection returns metric z. Comparing them directly is not
                # a small calibration error: measured on USB_100109 the map
                # read 0.241 where the geometry read 4.26, giving a score of
                # 556 -- exp(-556) is exactly 0 in float32, so every one of
                # 7966 support weights became 0 and the whole video branch
                # silently contributed nothing while still reporting pairs.
                obs = torch.where(valid_obs, 1.0 / obs.clamp_min(1.0e-6), torch.zeros_like(obs))
            scale = obs[valid_obs].abs().median().clamp_min(1.0e-6) * float(depth_sigma) if bool(valid_obs.any()) else torch.tensor(1.0)
            depth_score = (cand_z - obs).abs() / scale.clamp_min(1.0e-6)
            # Bound the score so a single bad depth sample cannot underflow
            # the whole row to zero; the relative ordering inside a row is
            # what the weights are for, and that survives the bound.
            score = score + depth_score.clamp_max(60.0)
            keep &= valid_obs
        # Softmax-style normalisation in log space: subtracting the row
        # minimum is mathematically a no-op after the normalisation below but
        # keeps exp() away from the underflow that produced the zeros.
        score = torch.where(keep, score, torch.full_like(score, float("inf")))
        row_min = score.min(dim=1, keepdim=True).values
        row_min = torch.where(torch.isfinite(row_min), row_min, torch.zeros_like(row_min))
        w = torch.exp(-(score - row_min)).float() * keep.float()
        if kk < k:
            ids[start:end, :kk] = local_ids
            weights[start:end, :kk] = w
            valid_out[start:end, :kk] = keep
        else:
            ids[start:end] = local_ids
            weights[start:end] = w
            valid_out[start:end] = keep
    norm = weights.sum(dim=1, keepdim=True).clamp_min(1.0e-8)
    weights = torch.where(valid_out, weights / norm, torch.zeros_like(weights))
    return PixelGaussianRows(ids, weights, valid_out)


@torch.no_grad()
def build_projected_video_endpoint_support(
    source_xyz: torch.Tensor,
    target_xyz: torch.Tensor,
    source_camera,
    target_camera,
    source_pixels: torch.Tensor,
    target_pixels: torch.Tensor,
    track_confidence: torch.Tensor,
    *,
    source_depth: torch.Tensor | None = None,
    target_depth: torch.Tensor | None = None,
    topk_gaussians: int = 8,
    max_pixel_radius: float = 6.0,
    depth_sigma: float = 0.03,
    depth_is_inverse: bool = True,
) -> SparseVideoEndpointSupport:
    source_rows = attach_pixels_to_projected_gaussians(
        source_xyz,
        source_camera,
        source_pixels,
        topk=topk_gaussians,
        max_pixel_radius=max_pixel_radius,
        depth_is_inverse=depth_is_inverse,
        depth_map=source_depth,
        depth_sigma=depth_sigma,
    )
    target_rows = attach_pixels_to_projected_gaussians(
        target_xyz,
        target_camera,
        target_pixels,
        topk=topk_gaussians,
        max_pixel_radius=max_pixel_radius,
        depth_is_inverse=depth_is_inverse,
        depth_map=target_depth,
        depth_sigma=depth_sigma,
    )
    conf = torch.as_tensor(track_confidence).detach().cpu().float().reshape(-1)
    sids: list[torch.Tensor] = []
    tids: list[torch.Tensor] = []
    vals: list[torch.Tensor] = []
    rows = min(source_rows.ids.shape[0], target_rows.ids.shape[0], conf.numel())
    for row in range(rows):
        if float(conf[row]) <= 0.0:
            continue
        sv = source_rows.valid[row]
        tv = target_rows.valid[row]
        if not bool(sv.any()) or not bool(tv.any()):
            continue
        si = source_rows.ids[row, sv]
        tj = target_rows.ids[row, tv]
        sw = source_rows.weights[row, sv]
        tw = target_rows.weights[row, tv]
        sids.append(si[:, None].expand(-1, int(tj.numel())).reshape(-1))
        tids.append(tj[None, :].expand(int(si.numel()), -1).reshape(-1))
        vals.append((sw[:, None] * tw[None, :] * conf[row]).reshape(-1))
    if not vals:
        empty = torch.empty(0, dtype=torch.long)
        return SparseVideoEndpointSupport(empty, empty, torch.empty(0), "forward", {"tracks": int(rows)})
    source_ids = torch.cat(sids)
    target_ids = torch.cat(tids)
    values = torch.cat(vals)
    keep = values > 0.0
    return SparseVideoEndpointSupport(
        source_ids[keep],
        target_ids[keep],
        values[keep],
        "forward",
        {"tracks": int(rows), "topk_gaussians": int(topk_gaussians), "max_pixel_radius": float(max_pixel_radius)},
    ).coalesce()
