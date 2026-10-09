"""Strict reciprocal one-to-one matching over a flat candidate graph."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch


@dataclass
class ReciprocalMatches:
    source_ids: torch.Tensor
    target_ids: torch.Tensor
    candidate_indices: torch.Tensor
    weights: torch.Tensor
    stats: dict[str, Any]


def strict_reciprocal_matches(
    *,
    source_ids: torch.Tensor,
    target_ids: torch.Tensor,
    energy: torch.Tensor,
    source_weight: torch.Tensor,
    target_weight: torch.Tensor,
    num_sources: int,
    num_targets: int,
) -> ReciprocalMatches:
    source_ids_cpu = torch.as_tensor(source_ids).detach().cpu().long()
    target_ids_cpu = torch.as_tensor(target_ids).detach().cpu().long()
    energy_cpu = torch.as_tensor(energy).detach().cpu().float()
    if source_ids_cpu.numel() != target_ids_cpu.numel() or source_ids_cpu.numel() != energy_cpu.numel():
        raise ValueError("source_ids, target_ids and energy must have matching length")

    best_by_source: dict[int, tuple[float, int]] = {}
    best_by_target: dict[int, tuple[float, int]] = {}
    for idx, (src, tgt, val) in enumerate(zip(source_ids_cpu.tolist(), target_ids_cpu.tolist(), energy_cpu.tolist())):
        if not torch.isfinite(torch.tensor(val)):
            continue
        current = best_by_source.get(src)
        if current is None or val < current[0]:
            best_by_source[src] = (val, idx)
        current = best_by_target.get(tgt)
        if current is None or val < current[0]:
            best_by_target[tgt] = (val, idx)

    keep: list[int] = []
    for src, (_, idx) in best_by_source.items():
        tgt = int(target_ids_cpu[idx])
        if best_by_target.get(tgt, (None, None))[1] == idx:
            keep.append(idx)
    keep_t = torch.tensor(sorted(keep), dtype=torch.long)
    if keep_t.numel() == 0:
        empty_f = torch.empty(0, dtype=torch.float32)
        stats = {
            "forward_best_matches": int(len(best_by_source)),
            "reciprocal_matches": 0,
            "reciprocal_fraction": 0.0,
            "unique_target_ratio": 0.0,
            "maximum_target_multiplicity": 0,
            "p90_target_multiplicity": 0.0,
        }
        return ReciprocalMatches(keep_t, keep_t, keep_t, empty_f, stats)

    matched_src = source_ids_cpu[keep_t]
    matched_tgt = target_ids_cpu[keep_t]
    source_weight_cpu = torch.as_tensor(source_weight).detach().cpu().float()
    target_weight_cpu = torch.as_tensor(target_weight).detach().cpu().float()
    selected_energy = energy_cpu[keep_t].clamp_min(0.0)
    match_conf = (1.0 / (1.0 + selected_energy)).clamp(0.0, 1.0)
    weights = torch.sqrt(source_weight_cpu[matched_src].clamp_min(0.0) * target_weight_cpu[matched_tgt].clamp_min(0.0)) * match_conf
    unique, counts = torch.unique(matched_tgt, return_counts=True)
    max_mult = int(counts.max()) if counts.numel() else 0
    stats = {
        "forward_best_matches": int(len(best_by_source)),
        "reciprocal_matches": int(keep_t.numel()),
        "reciprocal_fraction": float(keep_t.numel() / max(1, len(best_by_source))),
        "unique_target_ratio": float(unique.numel() / max(1, int(keep_t.numel()))),
        "maximum_target_multiplicity": max_mult,
        "p90_target_multiplicity": float(torch.quantile(counts.float(), 0.9)) if counts.numel() > 1 else float(max_mult),
        "num_sources_total": int(num_sources),
        "num_targets_total": int(num_targets),
    }
    if max_mult > 1:
        raise RuntimeError("strict reciprocal matching produced target multiplicity > 1")
    return ReciprocalMatches(matched_src, matched_tgt, keep_t, weights, stats)


def save_reciprocal_matches(matches: ReciprocalMatches, output_dir: str | Path) -> None:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    torch.save(matches.source_ids.cpu(), output / "source_ids.pt")
    torch.save(matches.target_ids.cpu(), output / "target_ids.pt")
    torch.save(matches.weights.cpu(), output / "weights.pt")
