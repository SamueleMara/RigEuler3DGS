"""Rank endpoint candidates from sparse video support; physics decides later."""

from __future__ import annotations

import torch

from utils.video_endpoint_transport import SparseVideoEndpointSupport


def rank_target_gaussians_from_video_support(
    support: SparseVideoEndpointSupport,
    *,
    num_sources: int,
    k: int = 16,
) -> tuple[torch.Tensor, torch.Tensor]:
    rank = torch.full((int(num_sources), int(k)), -1, dtype=torch.long)
    score = torch.zeros((int(num_sources), int(k)), dtype=torch.float32)
    if int(support.source_ids.numel()) == 0:
        return rank, score
    for src in torch.unique(support.source_ids.long()).tolist():
        mask = support.source_ids.long() == int(src)
        val, order = torch.sort(support.support[mask].float(), descending=True)
        tgt = support.target_ids[mask].long()[order]
        kk = min(int(k), int(tgt.numel()))
        rank[int(src), :kk] = tgt[:kk]
        score[int(src), :kk] = val[:kk]
    return rank, score


def rank_target_patches_from_video_support(
    support: SparseVideoEndpointSupport,
    source_patch: torch.Tensor,
    target_patch: torch.Tensor,
    *,
    k: int = 16,
) -> tuple[torch.Tensor, torch.Tensor]:
    sp = torch.as_tensor(source_patch).long()
    tp = torch.as_tensor(target_patch).long()
    src_patch_ids = sp[support.source_ids.long()]
    tgt_patch_ids = tp[support.target_ids.long()]
    patch_support = SparseVideoEndpointSupport(src_patch_ids, tgt_patch_ids, support.support, support.direction).coalesce()
    return rank_target_gaussians_from_video_support(patch_support, num_sources=int(sp.max()) + 1, k=k)


def measure_candidate_recall_on_exhaustive_subset(ranked_targets: torch.Tensor, true_targets: torch.Tensor, *, depths: tuple[int, ...] = (1, 4, 8, 16)) -> dict[str, float]:
    ranked = torch.as_tensor(ranked_targets).long()
    truth = torch.as_tensor(true_targets).long().reshape(-1)
    n = min(int(ranked.shape[0]), int(truth.numel()))
    ranked = ranked[:n]
    truth = truth[:n]
    return {f"recall_at_{d}": float(((ranked[:, : min(d, ranked.shape[1])] == truth[:, None]).any(1)).float().mean()) for d in depths}
