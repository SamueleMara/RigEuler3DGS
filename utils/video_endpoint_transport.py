"""Sparse video-derived endpoint Gaussian support.

No dense ``N0 x NT`` coupling is materialised.  Tracks carry sparse source and
target ownership rows, and this module aggregates only observed pairs.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch


@dataclass
class SparseVideoEndpointSupport:
    source_ids: torch.Tensor
    target_ids: torch.Tensor
    support: torch.Tensor
    direction: str = "forward"
    stats: dict[str, Any] = field(default_factory=dict)

    def coalesce(self) -> "SparseVideoEndpointSupport":
        if int(self.source_ids.numel()) == 0:
            return self
        key = torch.stack((self.source_ids.long(), self.target_ids.long()), dim=1)
        unique, inverse = torch.unique(key, dim=0, return_inverse=True)
        values = torch.zeros(unique.shape[0], dtype=self.support.dtype, device=self.support.device)
        values.scatter_add_(0, inverse, self.support)
        return SparseVideoEndpointSupport(unique[:, 0], unique[:, 1], values, self.direction, {**self.stats, "coalesced_pairs": int(unique.shape[0])})


def _top_sparse(row: torch.Tensor, k: int) -> tuple[torch.Tensor, torch.Tensor]:
    row = torch.as_tensor(row).detach().float().reshape(-1)
    if int(row.numel()) == 0:
        return torch.empty(0, dtype=torch.long), torch.empty(0)
    kk = min(int(k), int((row > 0).sum()) if bool((row > 0).any()) else int(row.numel()))
    val, idx = torch.topk(row, k=max(1, kk), largest=True)
    keep = val > 0
    return idx[keep].long(), val[keep]


def propagate_endpoint_source_ownership_through_video(
    source_ownership: torch.Tensor,
    track_source_pixels: torch.Tensor,
    track_confidence: torch.Tensor,
    *,
    topk_gaussians: int = 8,
) -> list[tuple[torch.Tensor, torch.Tensor]]:
    ownership = torch.as_tensor(source_ownership).detach().float()
    pixels = torch.as_tensor(track_source_pixels).long()
    conf = torch.as_tensor(track_confidence).detach().float().reshape(-1)
    out = []
    for p, c in zip(pixels.tolist(), conf.tolist()):
        ids, vals = _top_sparse(ownership[int(p)], int(topk_gaussians))
        out.append((ids, vals * float(c)))
    return out


def attach_propagated_tracks_to_target_ownership(
    target_ownership: torch.Tensor,
    track_target_pixels: torch.Tensor,
    *,
    topk_gaussians: int = 8,
) -> list[tuple[torch.Tensor, torch.Tensor]]:
    ownership = torch.as_tensor(target_ownership).detach().float()
    pixels = torch.as_tensor(track_target_pixels).long()
    return [_top_sparse(ownership[int(p)], int(topk_gaussians)) for p in pixels.tolist()]


def build_sparse_video_endpoint_gaussian_support(
    source_ownership: torch.Tensor,
    target_ownership: torch.Tensor,
    track_source_pixels: torch.Tensor,
    track_target_pixels: torch.Tensor,
    track_confidence: torch.Tensor,
    *,
    topk_gaussians: int = 8,
    direction: str = "forward",
) -> SparseVideoEndpointSupport:
    src_rows = propagate_endpoint_source_ownership_through_video(source_ownership, track_source_pixels, track_confidence, topk_gaussians=topk_gaussians)
    tgt_rows = attach_propagated_tracks_to_target_ownership(target_ownership, track_target_pixels, topk_gaussians=topk_gaussians)
    sids: list[torch.Tensor] = []
    tids: list[torch.Tensor] = []
    vals: list[torch.Tensor] = []
    for (si, sv), (tj, tv) in zip(src_rows, tgt_rows):
        if int(si.numel()) == 0 or int(tj.numel()) == 0:
            continue
        sids.append(si[:, None].expand(-1, int(tj.numel())).reshape(-1))
        tids.append(tj[None, :].expand(int(si.numel()), -1).reshape(-1))
        vals.append((sv[:, None] * tv[None, :]).reshape(-1))
    if not vals:
        empty = torch.empty(0, dtype=torch.long)
        return SparseVideoEndpointSupport(empty, empty, torch.empty(0), direction, {"tracks": int(torch.as_tensor(track_source_pixels).numel())})
    return SparseVideoEndpointSupport(torch.cat(sids), torch.cat(tids), torch.cat(vals), direction, {"tracks": int(torch.as_tensor(track_source_pixels).numel())}).coalesce()


def build_sparse_reverse_video_endpoint_gaussian_support(*args, **kwargs) -> SparseVideoEndpointSupport:
    kwargs["direction"] = "backward"
    return build_sparse_video_endpoint_gaussian_support(*args, **kwargs)
