"""Adjacent-frame semantic-foreground pixel flow matching probes.

This module builds empirical foreground-only pixel transports.  It does not
pretend to be optical flow; it is the small, testable core the closed-loop
pipeline can use for diagnostics and sparse RGB-D tracks.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Sequence

import torch

from utils.semantic_foreground_transport import build_foreground_pixel_indices, mask_sparse_pixel_transport

EPS = 1.0e-8


@dataclass
class PixelFMField:
    source_pixel_indices: torch.Tensor
    target_pixel_indices: torch.Tensor
    velocity: torch.Tensor
    confidence: torch.Tensor
    delta_t: float
    image_shape: tuple[int, int]
    frame_pair: tuple[int, int] = (0, 1)
    camera_id: str | int = 0
    direction: str = "forward"
    stats: dict[str, Any] = field(default_factory=dict)


def _image_chw(image: torch.Tensor) -> torch.Tensor:
    img = torch.as_tensor(image).detach().float()
    if img.ndim == 2:
        img = img[None]
    if img.ndim == 3 and img.shape[-1] in (1, 3, 4):
        img = img.permute(2, 0, 1)
    return img[:3]


def _xy(index: torch.Tensor, width: int) -> torch.Tensor:
    return torch.stack((index % width, index // width), dim=-1).float()


def build_foreground_adjacent_frame_pairs(
    source_rgb: torch.Tensor,
    target_rgb: torch.Tensor,
    source_depth: torch.Tensor,
    target_depth: torch.Tensor,
    source_foreground_mask: torch.Tensor,
    target_foreground_mask: torch.Tensor,
    *,
    delta_t: float,
    frame_pair: tuple[int, int] = (0, 1),
    camera_id: str | int = 0,
    max_source_pixels: int = 4096,
    candidates: int = 16,
    search_radius: int = 12,
    generator: torch.Generator | None = None,
    direction: str = "forward",
) -> PixelFMField:
    return match_foreground_pixels_to_next(
        source_rgb,
        target_rgb,
        source_depth,
        target_depth,
        source_foreground_mask,
        target_foreground_mask,
        source_pixel_indices=None,
        delta_t=delta_t,
        frame_pair=frame_pair,
        camera_id=camera_id,
        max_source_pixels=max_source_pixels,
        candidates=candidates,
        search_radius=search_radius,
        generator=generator,
        direction=direction,
    )


def match_foreground_pixels_to_next(
    source_rgb: torch.Tensor,
    target_rgb: torch.Tensor,
    source_depth: torch.Tensor,
    target_depth: torch.Tensor,
    source_foreground_mask: torch.Tensor,
    target_foreground_mask: torch.Tensor,
    *,
    source_pixel_indices: torch.Tensor | None,
    delta_t: float,
    frame_pair: tuple[int, int] = (0, 1),
    camera_id: str | int = 0,
    max_source_pixels: int = 4096,
    candidates: int = 16,
    search_radius: int = 12,
    generator: torch.Generator | None = None,
    direction: str = "forward",
) -> PixelFMField:
    if float(delta_t) <= 0.0:
        raise ValueError("real positive delta_t is mandatory")
    src_img = _image_chw(source_rgb)
    tgt_img = _image_chw(target_rgb).to(src_img)
    src_depth = torch.as_tensor(source_depth).detach().float()
    tgt_depth = torch.as_tensor(target_depth).detach().float().to(src_depth)
    height, width = int(src_img.shape[-2]), int(src_img.shape[-1])
    if source_pixel_indices is None:
        src = build_foreground_pixel_indices(source_foreground_mask, max_pixels=max_source_pixels, generator=generator)
    else:
        src = torch.as_tensor(source_pixel_indices).detach().cpu().long().reshape(-1)
        fg = torch.as_tensor(source_foreground_mask).detach().bool().reshape(-1)
        src = src[(src >= 0) & (src < fg.numel())]
        src = src[fg[src]]
        if src.numel() > int(max_source_pixels):
            src = src[: int(max_source_pixels)]
    target_fg = torch.as_tensor(target_foreground_mask).detach().bool().reshape(-1)
    if int(src.numel()) == 0 or not bool(target_fg.any()):
        empty = torch.empty(0, dtype=torch.long)
        return PixelFMField(empty, empty, torch.empty(0, 2), torch.empty(0), float(delta_t), (height, width), frame_pair, camera_id, direction, {"reason": "empty foreground"})
    radius = max(1, int(search_radius))
    offsets = torch.stack(
        torch.meshgrid(
            torch.arange(-radius, radius + 1),
            torch.arange(-radius, radius + 1),
            indexing="xy",
        ),
        dim=-1,
    ).reshape(-1, 2)
    offset_distance = torch.linalg.norm(offsets.float(), dim=-1)
    order = torch.argsort(offset_distance)
    offsets = offsets.index_select(0, order)
    offset_distance = offset_distance.index_select(0, order)
    max_offsets = max(int(candidates), 1)
    if max_offsets < int(offsets.shape[0]):
        offsets = offsets[:max_offsets]
        offset_distance = offset_distance[:max_offsets]
    src_x = (src % width).long()
    src_y = (src // width).long()
    cand_x = src_x[:, None] + offsets[None, :, 0]
    cand_y = src_y[:, None] + offsets[None, :, 1]
    in_bounds = (cand_x >= 0) & (cand_x < width) & (cand_y >= 0) & (cand_y < height)
    cand = (cand_y.clamp(0, height - 1) * width + cand_x.clamp(0, width - 1)).long()
    keep = in_bounds & target_fg.index_select(0, cand.reshape(-1)).reshape_as(cand)
    if not bool(keep.any()):
        empty = torch.empty(0, dtype=torch.long)
        return PixelFMField(empty, empty, torch.empty(0, 2), torch.empty(0), float(delta_t), (height, width), frame_pair, camera_id, direction, {"reason": "no local foreground candidates", "search_radius": radius})
    src_rgb = src_img.reshape(src_img.shape[0], -1).T[src]
    tgt_rgb = tgt_img.reshape(tgt_img.shape[0], -1).T[cand.reshape(-1)].reshape(cand.shape[0], cand.shape[1], -1)
    rgb_cost = torch.linalg.norm(tgt_rgb - src_rgb[:, None, :], dim=-1)
    src_d = src_depth.reshape(-1)[src]
    tgt_d = tgt_depth.reshape(-1)[cand]
    depth_cost = (tgt_d - src_d[:, None]).abs() / src_d.abs().median().clamp_min(1.0e-6)
    appearance_score = rgb_cost + depth_cost
    score = offset_distance[None, :].to(rgb_cost) / max(height, width) + appearance_score
    score = torch.where(keep.to(score.device), score, torch.full_like(score, float("inf")))
    finite = torch.isfinite(score)
    candidate_count = finite.sum(dim=1)
    topk = torch.topk(
        torch.where(finite, score, torch.full_like(score, float("inf"))),
        k=min(2, int(score.shape[1])),
        largest=False,
        dim=1,
    )
    best = topk.indices[:, 0]
    row = torch.arange(cand.shape[0])
    best_score = topk.values[:, 0]
    appearance_score = torch.where(keep.to(appearance_score.device), appearance_score, torch.full_like(appearance_score, float("inf")))
    best_appearance = appearance_score[row, best]
    alternative_appearance = appearance_score.clone()
    alternative_appearance[row, best] = float("inf")
    second_appearance = alternative_appearance.min(dim=1).values
    mean_appearance = torch.where(torch.isfinite(appearance_score), appearance_score, torch.zeros_like(appearance_score)).sum(dim=1) / candidate_count.clamp_min(1).to(score)
    distinctiveness = ((second_appearance - best_appearance).clamp_min(0.0) / (mean_appearance - best_appearance).clamp_min(EPS)).clamp(0.0, 1.0)
    row_keep = torch.isfinite(best_score)
    src = src[row_keep]
    matched = cand[row[row_keep], best[row_keep]]
    best_score = best_score[row_keep]
    confidence_value = torch.exp(-best_appearance[row_keep].clamp_min(0.0)) * distinctiveness[row_keep]
    src, matched, confidence = mask_sparse_pixel_transport(src, matched, confidence_value, source_foreground_mask, target_foreground_mask)
    pixel_delta = _xy(matched, width) - _xy(src, width)
    pixel_displacement = torch.linalg.norm(pixel_delta, dim=-1)
    velocity = pixel_delta / float(delta_t)
    return PixelFMField(
        source_pixel_indices=src,
        target_pixel_indices=matched,
        velocity=velocity,
        confidence=confidence.float(),
        delta_t=float(delta_t),
        image_shape=(height, width),
        frame_pair=frame_pair,
        camera_id=camera_id,
        direction=direction,
        stats={
            "foreground_sources": int(src.numel()),
            "foreground_targets": int(target_fg.sum()),
            "median_confidence": float(confidence.median()) if confidence.numel() else 0.0,
            "median_pixel_displacement": float(pixel_displacement.median()) if pixel_displacement.numel() else 0.0,
            "p90_pixel_displacement": float(torch.quantile(pixel_displacement.float(), 0.9)) if pixel_displacement.numel() else 0.0,
            "search_radius": int(radius),
            "candidate_offsets": int(offsets.shape[0]),
        },
    )


def _train_sequence(frames: Sequence[dict[str, Any]], *, backward: bool = False, **kwargs) -> list[PixelFMField]:
    fields: list[PixelFMField] = []
    order = range(len(frames) - 1)
    for t in order:
        a, b = (frames[t + 1], frames[t]) if backward else (frames[t], frames[t + 1])
        dt = abs(float(b.get("timestamp", t + 1)) - float(a.get("timestamp", t)))
        fields.append(build_foreground_adjacent_frame_pairs(
            a["rgb"], b["rgb"], a["depth"], b["depth"], a["mask"], b["mask"],
            delta_t=dt, frame_pair=(int(a.get("frame_id", t)), int(b.get("frame_id", t + 1))),
            camera_id=a.get("camera_id", 0), direction="backward" if backward else "forward", **kwargs,
        ))
    return fields


def train_forward_pixel_fm(frames: Sequence[dict[str, Any]], **kwargs) -> list[PixelFMField]:
    return _train_sequence(frames, backward=False, **kwargs)


def train_backward_pixel_fm(frames: Sequence[dict[str, Any]], **kwargs) -> list[PixelFMField]:
    return _train_sequence(frames, backward=True, **kwargs)


def integrate_forward_pixel_fm(field: PixelFMField, source_pixel_indices: torch.Tensor | None = None) -> torch.Tensor:
    source = field.source_pixel_indices if source_pixel_indices is None else torch.as_tensor(source_pixel_indices).long()
    width = int(field.image_shape[1])
    return _xy(source, width) + field.velocity[: int(source.numel())] * float(field.delta_t)


def integrate_backward_pixel_fm(field: PixelFMField, target_pixel_indices: torch.Tensor | None = None) -> torch.Tensor:
    target = field.source_pixel_indices if target_pixel_indices is None else torch.as_tensor(target_pixel_indices).long()
    width = int(field.image_shape[1])
    return _xy(target, width) + field.velocity[: int(target.numel())] * float(field.delta_t)


def compute_forward_backward_pixel_consistency(forward: PixelFMField, backward: PixelFMField) -> dict[str, torch.Tensor | float]:
    f_pairs = {(int(s), int(t)): i for i, (s, t) in enumerate(zip(forward.source_pixel_indices.tolist(), forward.target_pixel_indices.tolist()))}
    errors = []
    for j, (s_b, t_b) in enumerate(zip(backward.source_pixel_indices.tolist(), backward.target_pixel_indices.tolist())):
        idx = f_pairs.get((int(t_b), int(s_b)))
        if idx is not None:
            errors.append(torch.linalg.norm(forward.velocity[idx] + backward.velocity[j]))
    err = torch.stack(errors) if errors else torch.empty(0)
    return {"cycle_error": err, "median_cycle_error": float(err.median()) if err.numel() else float("nan"), "matched_cycles": int(err.numel())}
