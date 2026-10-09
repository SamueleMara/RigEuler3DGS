"""Inner-solve acceptance (FIRM v3 §19, §20, §35).

A bad motion solution must not corrupt memberships.  v2 failed exactly here: the
recovered transform made the assigned support fit the final cloud *worse* than
static identity, and the outer loop then bled the moving support away one outer
iteration at a time.  Acceptance is checked before any embedding update.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch


@dataclass
class InnerAcceptance:
    accepted: bool
    reasons: list[str] = field(default_factory=list)
    metrics: dict[str, float] = field(default_factory=dict)

    def as_row(self) -> dict[str, float | str | bool]:
        row: dict[str, float | str | bool] = {"inner_motion_accepted": bool(self.accepted)}
        row["inner_reject_reasons"] = ";".join(self.reasons)
        row.update({f"accept_{k}": v for k, v in self.metrics.items()})
        return row


def endpoint_usefulness(
    candidate_nn: torch.Tensor,
    part_prob: torch.Tensor,
    *,
    eps: float = 1.0e-8,
) -> torch.Tensor:
    """§19C:  dE_k = sum_i p_ik (E_ik - E_i0) / (sum_i p_ik + eps), for k >= 1.

    Negative means the moving hypothesis explains the final cloud better than
    leaving that support static.
    """
    energies = torch.as_tensor(candidate_nn)
    prob = torch.as_tensor(part_prob, dtype=energies.dtype, device=energies.device)
    static = energies[:, 0:1]
    gain = energies - static                       # [N, K+1]
    num = (prob * gain).sum(dim=0)
    den = prob.sum(dim=0).clamp_min(float(eps))
    return (num / den)[1:]                          # moving parts only


def evaluate_inner_solve(
    *,
    acceptance_metrics: dict[str, float],
    candidate_nn: torch.Tensor,
    part_prob: torch.Tensor,
    previous_strain: float | None = None,
    strain_threshold: float = float("inf"),
    require_render_improvement: bool = True,
    render_tolerance: float = 0.0,
    endpoint_tolerance: float = 0.0,
    require_endpoint_gain: bool = True,
) -> InnerAcceptance:
    """Apply the three §19 criteria and return a decision with its evidence."""
    reasons: list[str] = []
    metrics = dict(acceptance_metrics)

    delta_e = endpoint_usefulness(candidate_nn, part_prob)
    for k, value in enumerate(delta_e.tolist()):
        metrics[f"delta_endpoint_part_{k + 1}"] = float(value)
    worst = float(delta_e.max()) if delta_e.numel() else 0.0
    metrics["delta_endpoint_worst"] = worst

    # A. Render quality.
    improvement = float(metrics.get("render_improvement", 0.0))
    if require_render_improvement and improvement < -abs(float(render_tolerance)):
        reasons.append(
            f"render worse than static (rgb motion {metrics.get('render_rgb_motion', float('nan')):.5f} "
            f"vs static {metrics.get('render_rgb_static', float('nan')):.5f})"
        )

    # B. Local rigidity.
    strain = float(metrics.get("mean_strain_norm", 0.0))
    if strain > float(strain_threshold):
        improved = previous_strain is not None and strain < float(previous_strain)
        if not improved:
            reasons.append(f"mean strain {strain:.5f} above threshold {float(strain_threshold):.5f} and not improving")

    # C. Endpoint usefulness.
    if require_endpoint_gain and worst > abs(float(endpoint_tolerance)):
        reasons.append(f"moving support fits final cloud worse than static (dE={worst:+.6f})")

    return InnerAcceptance(accepted=not reasons, reasons=reasons, metrics=metrics)
