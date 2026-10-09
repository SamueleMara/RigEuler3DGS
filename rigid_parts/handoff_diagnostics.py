"""Soft-P handoff and polarization diagnostics.

CODEX_DEBUG_SOFTP_GRADIENTS_STORAGE_DIAGNOSIS, Parts I-II (§3-§12).

The question these answer is narrow and load-bearing: does
``build_robust_inner_support`` actually receive the persistent SOFT probability
tensor P, or a one-hot projection of it? If it receives one-hot, then every
confidence gate in the support module (core threshold, attachment margin,
continuous fit weights) is structurally inert -- not because the data is
saturated, but because the information was destroyed at the boundary.

§5's ``mean |P - onehot(argmax P)|_1`` is the decisive statistic: it is exactly
0 for a one-hot input and strictly positive for anything genuinely soft.

Everything here is measurement only. No thresholds are tuned and no outer
behaviour is changed (§13).
"""

from __future__ import annotations

from typing import Any

import torch

EPS = 1.0e-12


def _percentiles(values: torch.Tensor, keys=(0.10, 0.25, 0.50, 0.75, 0.90)) -> dict[str, float]:
    if int(values.numel()) == 0:
        return {f"p{int(k * 100)}": float("nan") for k in keys}
    q = torch.quantile(values.double().flatten(),
                       torch.tensor(keys, dtype=torch.float64, device=values.device))
    return {f"p{int(k * 100)}": float(v) for k, v in zip(keys, q)}


@torch.no_grad()
def soft_probability_statistics(
    probs: torch.Tensor,
    hard_labels: torch.Tensor | None = None,
    *,
    eps: float = EPS,
) -> dict[str, Any]:
    """§4/§5/§6 statistics of the tensor ACTUALLY passed as ``soft_part_probs``."""
    p = torch.as_tensor(probs).detach().double()
    n, k1 = int(p.shape[0]), int(p.shape[1])
    z = (p.argmax(dim=-1) if hard_labels is None
         else torch.as_tensor(hard_labels).detach().reshape(-1).long())

    # §4: normalisation and entropy
    row_sum_error = float((p.sum(dim=1) - 1.0).abs().mean())
    entropy = -(p * (p + eps).log()).sum(dim=1)
    near_binary = (p < 1e-6) | (p > 1 - 1e-6)
    row_one_hot = near_binary.all(dim=1)
    max_p = p.max(dim=1).values

    # §5: THE decisive diagnostic. Exactly 0 <=> the input is one-hot.
    onehot = torch.zeros_like(p).scatter_(1, z[:, None], 1.0)
    l1_to_hard = (p - onehot).abs().sum(dim=1)

    stats: dict[str, Any] = {
        "num_gaussians": n,
        "num_classes": k1,
        "row_sum_error": row_sum_error,
        "entropy_mean": float(entropy.mean()),
        "entropy_median": float(entropy.median()),
        **{f"entropy_{k}": v for k, v in _percentiles(entropy).items()},
        "fraction_entries_near_binary": float(near_binary.double().mean()),
        "fraction_rows_near_one_hot": float(row_one_hot.double().mean()),
        "fraction_max_above_0.90": float((max_p > 0.90).double().mean()),
        "fraction_max_above_0.99": float((max_p > 0.99).double().mean()),
        "fraction_max_above_0.999": float((max_p > 0.999).double().mean()),
        # §5
        "l1_to_onehot_mean": float(l1_to_hard.mean()),
        "l1_to_onehot_median": float(l1_to_hard.median()),
        **{f"l1_to_onehot_{k}": v for k, v in _percentiles(l1_to_hard).items()},
        "l1_to_onehot_max": float(l1_to_hard.max()) if n else float("nan"),
        "fraction_l1_below_1e-6": float((l1_to_hard < 1e-6).double().mean()),
        "fraction_l1_below_1e-3": float((l1_to_hard < 1e-3).double().mean()),
        "fraction_l1_below_1e-2": float((l1_to_hard < 1e-2).double().mean()),
    }

    # §6: moving-candidate confidence
    candidate = z > 0
    c_dyn = 1.0 - p[:, 0]
    max_part = p[:, 1:].max(dim=1).values if k1 > 1 else torch.zeros_like(c_dyn)
    cand_dyn, cand_part = c_dyn[candidate], max_part[candidate]
    stats["candidate_count"] = int(candidate.sum())
    if int(candidate.sum()) > 0:
        stats["candidate_dynamic"] = {
            "min": float(cand_dyn.min()), "max": float(cand_dyn.max()),
            "mean": float(cand_dyn.mean()), "median": float(cand_dyn.median()),
            **_percentiles(cand_dyn)}
        stats["candidate_max_part"] = {
            "min": float(cand_part.min()), "max": float(cand_part.max()),
            "mean": float(cand_part.mean()), "median": float(cand_part.median()),
            **_percentiles(cand_part)}
        for level in (0.6, 0.7, 0.8, 0.9, 0.99, 0.999):
            stats[f"candidate_fraction_dyn_above_{level}"] = float(
                (cand_dyn > level).double().mean())
    # The headline verdict, so no one has to read the numbers to get it.
    stats["input_is_effectively_hard"] = bool(stats["fraction_l1_below_1e-6"] > 0.999)
    return stats


def format_softp_handoff(name: str, probs: torch.Tensor, stats: dict[str, Any]) -> str:
    """§8's console block, naming the tensor that was actually passed."""
    p = torch.as_tensor(probs)
    lines = [
        "[SoftPHandoff]",
        f"  tensor source: {name}",
        f"  shape: {tuple(p.shape)}   dtype: {p.dtype}   requires_grad: {bool(p.requires_grad)}",
        f"  min: {float(p.min()):.6g}   max: {float(p.max()):.6g}   "
        f"row-sum error: {stats['row_sum_error']:.3e}",
        f"  entropy mean: {stats['entropy_mean']:.6f}   median: {stats['entropy_median']:.6f}",
        f"  distance-to-onehot mean: {stats['l1_to_onehot_mean']:.6e}   "
        f"median: {stats['l1_to_onehot_median']:.6e}   max: {stats['l1_to_onehot_max']:.6e}",
        f"  rows near one-hot: {stats['fraction_rows_near_one_hot'] * 100:.2f}%   "
        f"max(P)>0.99: {stats['fraction_max_above_0.99'] * 100:.2f}%   "
        f"|P-onehot|<1e-6: {stats['fraction_l1_below_1e-6'] * 100:.2f}%",
    ]
    if "candidate_dynamic" in stats:
        d, m = stats["candidate_dynamic"], stats["candidate_max_part"]
        lines.append(f"  candidates {stats['candidate_count']}: P_dyn min {d['min']:.4f} "
                     f"median {d['median']:.4f} max {d['max']:.4f}   "
                     f"maxpart min {m['min']:.4f} median {m['median']:.4f}")
    if stats["input_is_effectively_hard"]:
        # §10: report, never crash -- genuine saturation is possible.
        lines.append("  WARNING: INNER SUPPORT RECEIVED NEAR-HARD PROBABILITIES "
                     "(|P - onehot(argmax P)|_1 < 1e-6 on >99.9% of rows). Every "
                     "confidence gate in the support module is inert on this input.")
    return "\n".join(lines)


@torch.no_grad()
def polarization_row(outer: int, probs: torch.Tensor,
                     previous: torch.Tensor | None = None) -> dict[str, Any]:
    """§11/§12: one row per outer tracking how polarization evolves.

    Whether saturation arrives from the outer update or is manufactured at the
    inner boundary is exactly what this separates.
    """
    p = torch.as_tensor(probs).detach().double()
    entropy = -(p * (p + EPS).log()).sum(dim=1)
    max_p = p.max(dim=1).values
    z = p.argmax(dim=-1)
    onehot = torch.zeros_like(p).scatter_(1, z[:, None], 1.0)
    row = {
        "outer": int(outer),
        "entropy_mean": float(entropy.mean()),
        "entropy_median": float(entropy.median()),
        "fraction_max_above_0.90": float((max_p > 0.90).double().mean()),
        "fraction_max_above_0.99": float((max_p > 0.99).double().mean()),
        "fraction_one_hot_like": float((((p < 1e-6) | (p > 1 - 1e-6)).all(dim=1)).double().mean()),
        "mean_l1_to_onehot": float((p - onehot).abs().sum(dim=1).mean()),
    }
    if previous is not None:
        q = torch.as_tensor(previous).detach().double()
        if q.shape == p.shape:
            row["mean_delta_P"] = float((p - q).abs().sum(dim=1).mean())
            row["argmax_label_changes"] = int((q.argmax(dim=-1) != z).sum())
    return row
