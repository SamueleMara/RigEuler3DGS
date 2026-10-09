"""Assignment updates for progressive rigid discovery.

There is no graph smoothing here.  The only question this module answers is:
given immutable FM endpoint evidence and current rigid transforms, which
explicit hypothesis is best supported, and is it trusted enough to commit?
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F

from utils.progressive_rigid_state import (
    PROVENANCE_COMPONENT_ACCEPT,
    PROVENANCE_FM_CORE,
    STATUS_TRUSTED_STATIC,
    STATUS_UNCERTAIN,
    STATUS_UNOBSERVABLE,
    ProgressiveRigidState,
    status_for_part,
)
from utils.rigid_endpoint_fit import transform_points


@dataclass
class AssignmentThresholds:
    static_probability: float = 0.25
    dynamic_probability: float = 0.65
    confidence: float = 0.25
    moving_residual: float = 0.05
    static_residual: float = 0.03
    probability_margin: float = 0.15
    temperature: float = 0.05


def compute_fm_rigid_residuals(state: ProgressiveRigidState) -> torch.Tensor:
    """Endpoint distance for static and each rigid part against immutable FM."""
    residuals = []
    residuals.append(torch.linalg.norm(state.canonical_fg - state.fm_endpoint_xyz, dim=-1))
    for part in range(1, state.num_labels):
        residuals.append(torch.linalg.norm(state.part_positions(part) - state.fm_endpoint_xyz, dim=-1))
    state.fm_rigid_residual = torch.stack(residuals, dim=0)
    return state.fm_rigid_residual


def initialize_from_fm_probability(state: ProgressiveRigidState, thresholds: AssignmentThresholds) -> ProgressiveRigidState:
    """Seed static/moving cores from FM probability only; no GT, no smoothing."""
    p = state.fm_probability.clamp(0.0, 1.0)
    c = state.fm_confidence_current.clamp(0.0, 1.0)
    valid = state.fm_valid
    k = state.num_moving_parts
    state.responsibilities[:, 0] = (1.0 - p).clamp_min(1.0e-6)
    if k > 0:
        moving = (p / float(k)).clamp_min(1.0e-6)
        state.responsibilities[:, 1:] = moving[:, None]
    state.responsibilities = state.responsibilities / state.responsibilities.sum(-1, keepdim=True).clamp_min(1.0e-8)
    state.hard_labels = state.responsibilities.argmax(-1)
    state.assignment_confidence = (2.0 * (p - 0.5).abs() * c).clamp(0.0, 1.0)
    state.status[:] = STATUS_UNCERTAIN
    state.status[~valid] = STATUS_UNOBSERVABLE
    static = valid & (c >= thresholds.confidence) & (p <= thresholds.static_probability)
    moving = valid & (c >= thresholds.confidence) & (p >= thresholds.dynamic_probability)
    state.status[static] = STATUS_TRUSTED_STATIC
    if k > 0:
        state.status[moving] = status_for_part(1)
        state.hard_labels[moving] = 1
    state.provenance[static | moving] = PROVENANCE_FM_CORE
    return state.validate()


def update_responsibilities_from_residuals(
    state: ProgressiveRigidState,
    *,
    temperature: float,
    move_prior_bias: float = 0.0,
) -> ProgressiveRigidState:
    residual = compute_fm_rigid_residuals(state)
    temp = max(float(temperature), 1.0e-8)
    energy = residual / temp
    p = state.fm_probability.clamp(1.0e-4, 1.0 - 1.0e-4)
    prior = torch.empty_like(energy)
    prior[0] = -torch.log1p(-p)
    if state.num_moving_parts > 0:
        moving_prior = -torch.log(p / float(state.num_moving_parts)) + float(move_prior_bias)
        prior[1:] = moving_prior[None].expand_as(prior[1:])
    logits = -(energy + prior).transpose(0, 1)
    logits = torch.where(state.fm_valid[:, None], logits, torch.full_like(logits, -20.0))
    state.responsibilities = F.softmax(logits, dim=-1)
    state.hard_labels = state.responsibilities.argmax(-1)
    confidence, _ = state.responsibilities.max(-1)
    runner_up = torch.topk(state.responsibilities, k=min(2, state.num_labels), dim=-1).values
    if runner_up.shape[1] == 2:
        confidence = (runner_up[:, 0] - runner_up[:, 1]).clamp(0.0, 1.0)
    state.assignment_confidence = confidence
    return state.validate()


def hypothesis_energy_for_component(
    state: ProgressiveRigidState,
    component_mask: torch.Tensor,
    *,
    label: int,
    endpoint_target_xyz: torch.Tensor | None = None,
    pcd_weight: float = 1.0,
    fm_weight: float = 1.0,
    render_weight: float = 0.0,
    depth_weight: float = 0.0,
) -> dict[str, float]:
    """Score one component under one explicit label hypothesis."""
    mask = component_mask.to(device=state.device).bool()
    if not bool(mask.any()):
        return {"total": float("inf"), "fm": float("inf"), "pcd": 0.0, "render": 0.0, "depth": 0.0}
    if int(label) == 0:
        proposed = state.canonical_fg[mask]
    else:
        proposed = state.part_positions(int(label))[mask]
    fm = torch.linalg.norm(proposed - state.fm_endpoint_xyz[mask], dim=-1)
    conf = state.fm_confidence_current[mask] * state.fm_valid[mask].to(state.fm_confidence_current.dtype)
    fm_loss = (conf * fm).sum() / conf.sum().clamp_min(1.0e-8)
    pcd_loss = torch.zeros((), device=state.device, dtype=state.canonical_fg.dtype)
    if endpoint_target_xyz is not None and int(endpoint_target_xyz.numel()) > 0:
        target = endpoint_target_xyz.to(device=state.device, dtype=state.canonical_fg.dtype)
        chunk = min(int(proposed.shape[0]), 2048)
        vals = []
        for start in range(0, int(proposed.shape[0]), chunk):
            vals.append(torch.cdist(proposed[start:start + chunk], target).min(dim=1).values)
        pcd_loss = torch.cat(vals).mean()
    render_loss = state.render_residual_per_gaussian[mask].mean() if state.render_residual_per_gaussian.numel() else pcd_loss * 0.0
    depth_loss = state.depth_residual_per_gaussian[mask].mean() if state.depth_residual_per_gaussian.numel() else pcd_loss * 0.0
    total = (
        float(fm_weight) * fm_loss
        + float(pcd_weight) * pcd_loss
        + float(render_weight) * render_loss
        + float(depth_weight) * depth_loss
    )
    return {
        "total": float(total.detach().cpu()),
        "fm": float(fm_loss.detach().cpu()),
        "pcd": float(pcd_loss.detach().cpu()),
        "render": float(render_loss.detach().cpu()),
        "depth": float(depth_loss.detach().cpu()),
        "count": int(mask.sum()),
    }


#: Free parameters "spent" by explaining a component with a fixed rigid part
#: transform (H1) instead of leaving it static (H0).  Static is the
#: zero-parameter null (identity, nothing fit); a rigid SE(3) transform is
#: three rotational and three translational degrees of freedom -- a full
#: transform, not a direction, so there is no unit-gauge ambiguity to remove
#: the way there is in kpart_field_merge.py's screw-generator comparison.
#:
#: Unlike that comparison, the transform tested here was *not* fit to this
#: component's own data -- it was already fit from other, previously
#: committed support -- so there is no automatic better-fit bias from the
#: more complex model to correct for.  The log(n) charge instead guards
#: against the real risk at this decision point: dozens of candidate
#: components are tested every outer iteration, and a small, noisy one can
#: look decisively "moving" or "static" by chance alone.  Requiring the
#: evidence to clear a sample-size-scaled bar -- the same BIC form -- makes
#: small components harder to convince than well-observed ones, which a
#: fixed probability cutoff cannot do.
EXTRA_MOTION_DOF = 6

#: scores[label][cost_key] -> scores[label][count_key] holds how many
#: independent scalar observations that cost is averaged over.  fm/pcd are
#: per-Gaussian residuals (one scalar per Gaussian in the component); render
#: /depth are per-pixel residuals over the counterfactual renderer's shared
#: affected-pixel mask.
_MODALITY_COUNT_KEYS = (
    ("fm", "count"),
    ("pcd", "count"),
    ("render", "render_count"),
    ("depth", "render_count"),
)


def _modality_log_likelihood_ratio(static_cost: float, moving_cost: float, count: int) -> float:
    """``n * log(static_cost / moving_cost)``; positive favours moving.

    Same construction as ``compare_rigid_and_translation_evidence``: under
    i.i.d. Gaussian residuals this is the log-likelihood ratio of the two
    fits, with static standing in for the simpler (zero-parameter) side.
    """
    n = max(int(count), 0)
    if n <= 0:
        return 0.0
    static_cost = max(float(static_cost), 0.0)
    moving_cost = max(float(moving_cost), 0.0)
    if static_cost <= 0.0 and moving_cost <= 0.0:
        return 0.0
    if moving_cost <= 0.0:
        return math.inf
    if static_cost <= 0.0:
        return -math.inf
    return n * math.log(static_cost / moving_cost)


def analytic_component_evidence(scores: dict[int, dict[str, float]], *, label: int) -> dict[str, Any]:
    """BIC-corrected static-vs-``label`` log-likelihood ratio for one component.

    Mirrors ``utils/kpart_field_merge.py::compare_rigid_and_translation_evidence``:
    every term comes from the data (the per-modality costs ``hypothesis_energy_
    for_component``/the counterfactual renderer already computed, and how many
    independent observations back each one) or from the two models' own
    structure (``EXTRA_MOTION_DOF``).  Nothing is tuned and nothing is clamped
    except the returned magnitude, which is capped so a perfect-fit modality
    can't produce a non-JSON-serialisable infinity.

    As in that function, ``n`` counts correlated observations (neighbouring
    pixels under one Gaussian's splat footprint, say), so it overstates the
    independent sample size; the magnitude is an ordering, not a calibrated
    probability.
    """
    static_scores = scores.get(0, {})
    moving_scores = scores.get(int(label), {})
    total_llr = 0.0
    total_n = 0
    per_modality: dict[str, dict[str, float]] = {}
    for cost_key, count_key in _MODALITY_COUNT_KEYS:
        if cost_key not in static_scores or cost_key not in moving_scores:
            continue
        count = moving_scores.get(count_key, static_scores.get(count_key, 0))
        if not count:
            continue
        llr = _modality_log_likelihood_ratio(static_scores[cost_key], moving_scores[cost_key], count)
        per_modality[cost_key] = {
            "log_likelihood_ratio": float(max(min(llr, 1.0e18), -1.0e18)),
            "count": int(count),
        }
        total_n += int(count)
        if math.isinf(llr):
            total_llr = llr
            break
        total_llr += llr
    if total_n <= 0:
        return {
            "label": int(label),
            "log_likelihood_ratio": None,
            "observation_count": 0,
            "bic_penalty": None,
            "log_evidence_difference": None,
            "probability_moving": None,
            "per_modality": per_modality,
            "reason": "no_valid_observations",
        }
    penalty = EXTRA_MOTION_DOF * math.log(float(total_n))
    difference = total_llr - penalty
    # torch.sigmoid is exact at both infinities, so the odds need no clamp.
    probability = float(torch.sigmoid(torch.tensor(difference / 2.0, dtype=torch.float64)))
    clamped_llr = float(max(min(total_llr, 1.0e18), -1.0e18))
    clamped_difference = float(max(min(difference, 1.0e18), -1.0e18))
    return {
        "label": int(label),
        "log_likelihood_ratio": clamped_llr,
        "observation_count": total_n,
        "bic_penalty": penalty,
        "log_evidence_difference": clamped_difference,
        "probability_moving": probability,
        "per_modality": per_modality,
        "reason": None,
    }


@dataclass
class ComponentDecisionConfig:
    min_reliability: float = 0.05
    min_observable_fraction: float = 0.1


def decide_component(
    scores: dict[int, dict[str, float]],
    *,
    observable_fraction: float,
    reliability: float,
    cfg: ComponentDecisionConfig,
) -> dict[str, Any]:
    """Static-vs-moving verdict from the analytic evidence, not a fixed cutoff.

    Once there is any valid measurement, the BIC-corrected log-likelihood
    ratio in ``analytic_component_evidence`` always resolves to a sign --
    there is no "too close to call" zone to defer on the evidence alone (see
    that function's docstring).  "defer" here means the measurements
    themselves aren't trustworthy yet (too little of the component is
    observed, or the backing confidence is too low), never that the two
    hypotheses were merely close.
    """
    reason: list[str] = []
    if observable_fraction < cfg.min_observable_fraction:
        reason.append("low_observability")
    if reliability < cfg.min_reliability:
        reason.append("low_reliability")
    evidence = {int(label): analytic_component_evidence(scores, label=label) for label in scores if int(label) != 0}
    decisive = {label: ev for label, ev in evidence.items() if ev["log_evidence_difference"] is not None}
    probabilities = {label: float(ev["probability_moving"] or 0.0) for label, ev in evidence.items()}
    if not decisive:
        reason.append("no_valid_evidence")
        probabilities[0] = 1.0
        return {
            "action": "defer",
            "label": 0,
            "probability": 0.0,
            "log_evidence_difference": None,
            "evidence": evidence,
            "observable_fraction": float(observable_fraction),
            "reliability": float(reliability),
            "reason": reason,
            "probabilities": probabilities,
        }
    best_label, best_evidence = max(decisive.items(), key=lambda item: item[1]["log_evidence_difference"])
    difference = float(best_evidence["log_evidence_difference"])
    moving_wins = difference > 0.0
    label = int(best_label) if moving_wins else 0
    probability_moving = float(best_evidence["probability_moving"])
    probabilities[0] = 1.0 - probability_moving
    action = "defer" if reason else "accept"
    return {
        "action": action,
        "label": label,
        "probability": probability_moving if moving_wins else 1.0 - probability_moving,
        "log_evidence_difference": difference,
        "evidence": evidence,
        "observable_fraction": float(observable_fraction),
        "reliability": float(reliability),
        "reason": reason,
        "probabilities": probabilities,
    }


def apply_component_acceptance(
    state: ProgressiveRigidState,
    component_mask: torch.Tensor,
    label: int,
    *,
    probability: float,
) -> None:
    mask = component_mask.to(device=state.device).bool()
    if not bool(mask.any()):
        return
    label = int(label)
    state.hard_labels[mask] = label
    state.responsibilities[mask] = 1.0e-4
    state.responsibilities[mask, label] = 1.0
    state.responsibilities[mask] = state.responsibilities[mask] / state.responsibilities[mask].sum(-1, keepdim=True)
    state.assignment_confidence[mask] = float(probability)
    state.status[mask] = STATUS_TRUSTED_STATIC if label == 0 else status_for_part(label)
    state.provenance[mask] = PROVENANCE_COMPONENT_ACCEPT
    if state.resolved_once is not None:
        state.resolved_once[mask] = True
    if label > 0 and state.outer_accepted_moving is not None:
        state.outer_accepted_moving[label - 1, mask] = True


def loss_noise_tolerances(history: list[dict[str, float]], *, keys: tuple[str, ...] = ("fm", "pcd", "rgb", "depth", "total")) -> dict[str, float]:
    """Measured rollback epsilons from the last converged inner iterations.

    The tolerance is only optimizer/noise scale, never a hand-tuned percentage.
    """
    tail = [row for row in history[-10:] if isinstance(row, dict)]
    result: dict[str, float] = {}
    float_eps = torch.finfo(torch.float32).eps
    for key in keys:
        values = torch.tensor([float(row.get(key, 0.0)) for row in tail], dtype=torch.float32)
        if values.numel() <= 1:
            sigma = torch.tensor(0.0)
            ref = values[-1] if values.numel() else torch.tensor(0.0)
        else:
            sigma = values.std(unbiased=False)
            ref = values[-1]
        result[key] = float(torch.maximum(3.0 * sigma, 10.0 * float_eps * torch.maximum(torch.ones_like(ref), ref.abs())))
    return result


def transaction_decision(
    before: dict[str, float],
    after: dict[str, float],
    tolerance: dict[str, float],
) -> dict[str, Any]:
    """Commit/rollback decision for moving-support expansion."""
    failures: list[str] = []
    if float(after.get("total", 0.0)) > float(before.get("total", 0.0)) + float(tolerance.get("total", 0.0)):
        failures.append("weighted_total_worsened")
    for key in ("pcd", "rgb", "depth"):
        if float(after.get(key, 0.0)) > float(before.get(key, 0.0)) + float(tolerance.get(key, 0.0)):
            failures.append(f"{key}_worsened")
    if (
        float(after.get("fm", 0.0)) < float(before.get("fm", 0.0))
        and float(after.get("pcd", 0.0)) > float(before.get("pcd", 0.0)) + float(tolerance.get("pcd", 0.0))
    ):
        failures.append("fm_improved_but_pcd_worsened")
    return {
        "decision": "ROLLBACK" if failures else "COMMIT",
        "reason": ";".join(failures) if failures else "physical_losses_within_noise",
        "failures": failures,
    }
