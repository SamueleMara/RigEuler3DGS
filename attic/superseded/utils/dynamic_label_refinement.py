"""Conservative frame-0 dynamic/static probability refinement.

The only output this stage is responsible for is a better ``p_dynamic`` on
the frame-0 Gaussians.  It deliberately uses a transparent deterministic
update rather than a learned label-correction head, so every individual
signal can be switched off and attributed.

Three invariants hold the update together:

* The proposal is always anchored to the IMMUTABLE flow-matching prior, never
  to the running refined probability.  Re-deriving the base logit from the
  previous posterior would re-add the same physical evidence on every outer
  iteration; with this anchoring, unchanged evidence gives an unchanged
  proposal and the only iteration effect left is EMA convergence.
* Confidence gates evidence, it never adds evidence.  A well-supported
  neighbourhood is equally well supported whether it is static or moving.
* Missing evidence is neutral.  A signal that could not be computed
  contributes zero, and never turns a confidently-classified point ambiguous.

The central physical statement being encoded:

    coherent NONZERO rigid motion -> dynamic
    low motion                    -> static
    non-rigid / unsupported       -> reduce confidence, or mark ambiguous
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch

LABEL_STATIC = 0
LABEL_DYNAMIC = 1
LABEL_AMBIGUOUS = 2


@dataclass
class RefinementWeights:
    """Logit contributions. Each term is normalised into [-1,1] first, so a
    weight reads directly as "at most this many logits of evidence".

    The reverse-consistency weights default to zero: they are only
    trustworthy once the backward flow-matching branch passes its own
    diagnostics, and tuning them upward to compensate for a bad backward
    field would be fitting the label rule to an upstream defect.
    """

    motion: float = 2.0
    rigid: float = 2.0
    neighbor: float = 1.0
    #: Off by default: empirically larger in high-motion regions (distorted FM
    #: field, near-planar surface neighbourhoods, weakly-constrained normal
    #: direction of a full 3x3 gradient), so as signed evidence it suppresses
    #: exactly the moving points it is supposed to validate.  Still computed
    #: and saved for field quality and future part-boundary use.
    strain: float = 0.0
    cycle: float = 0.0
    rotation_cycle: float = 0.0
    jacobian_cycle: float = 0.0
    uncertainty: float = 0.0


@dataclass
class DynamicRefinementState:
    """Immutable prior plus the running EMA state."""

    fm_prior_prob: torch.Tensor
    fm_prior_logit: torch.Tensor
    refined_prob: torch.Tensor


@dataclass
class RefinementReport:
    probability: torch.Tensor
    proposal_probability: torch.Tensor
    label: torch.Tensor
    ambiguous: torch.Tensor
    evidence: torch.Tensor
    logit_terms: dict[str, torch.Tensor] = field(default_factory=dict)
    stats: dict[str, Any] = field(default_factory=dict)


def robust_unit_score(value: torch.Tensor, valid: torch.Tensor | None = None) -> torch.Tensor:
    """Map a non-negative signal into [0,1) using its own median as the unit.

    ``v / (v + median(v))`` is monotone, bounded, and carries no hand-tuned
    length/energy scale: the same weight means the same thing whether the
    scene is reconstructed in millimetres or metres.
    """
    value = torch.as_tensor(value).detach()
    finite = torch.isfinite(value)
    if valid is not None:
        finite = finite & valid.to(device=value.device, dtype=torch.bool)
    positive = value[finite & (value > 0)]
    if positive.numel() == 0:
        return torch.zeros_like(value)
    median = positive.median().clamp_min(1.0e-12)
    score = value.clamp_min(0.0) / (value.clamp_min(0.0) + median)
    return torch.where(torch.isfinite(score), score, torch.zeros_like(score))


def centered_unit_score(value: torch.Tensor, valid: torch.Tensor | None = None) -> torch.Tensor:
    """Same, but zero at the median so an unremarkable point is neutral.

    Without centring, every term reads ~0.5 for a typical point, so a sum of
    penalties structurally outweighs the bonuses and drags the whole scene
    toward static regardless of what the evidence says.
    """
    return 2.0 * robust_unit_score(value, valid) - 1.0


def compute_prior_logit(probability: torch.Tensor, *, cap: float = 4.0, eps: float = 1.0e-6) -> torch.Tensor:
    """Logit of the FM prior, clipped so geometry can still contradict it.

    A prior of exactly 1.0 is an infinite logit that no finite amount of
    physical evidence can move, which would make every confident
    flow-matching false positive permanent.  ``cap = 4`` corresponds to
    p ~= 0.982: strong confidence survives, contradiction stays possible.
    Pass ``cap <= 0`` to disable.
    """
    p = torch.as_tensor(probability).detach().clamp(float(eps), 1.0 - float(eps))
    prior = torch.log(p) - torch.log1p(-p)
    if float(cap) > 0.0:
        prior = prior.clamp(-float(cap), float(cap))
    return prior


def logit(probability: torch.Tensor, *, eps: float = 1.0e-6) -> torch.Tensor:
    return compute_prior_logit(probability, cap=0.0, eps=eps)


def unexplained_motion_fraction(
    rigid_fit_error: torch.Tensor,
    static_fit_error: torch.Tensor,
) -> torch.Tensor:
    """Share of the local motion the best rigid model fails to explain.

    ``E_R / (E_R + E_S)`` rather than ``E_R`` alone: an absolute residual
    grows with the size of the motion it is explaining, so penalising it
    directly penalises points for moving at all.  Dimensionless, no epsilon:
    0 when the rigid model explains what static cannot, 0.5 when the two are
    indistinguishable (including a genuinely static neighbourhood, where
    both are ~0 and no information exists), 1 when rigid is the worse fit.
    """
    rigid = torch.as_tensor(rigid_fit_error).detach().clamp_min(0.0)
    static = torch.as_tensor(static_fit_error).detach().to(rigid).clamp_min(0.0)
    total = rigid + static
    fraction = torch.where(total > 0, rigid / total.clamp_min(1.0e-30), torch.full_like(total, 0.5))
    return torch.where(torch.isfinite(fraction), fraction, torch.full_like(fraction, 0.5))


def fit_confidence_weights(
    dynamic_probability: torch.Tensor,
    fm_confidence: torch.Tensor,
    valid: torch.Tensor,
    *,
    alpha: float = 0.25,
) -> torch.Tensor:
    """Per-point weight for supporting a *moving* local transform.

    Mixed with a geometry-only floor ``alpha`` so that an early wrong label
    cannot delete the neighbourhood support that would have corrected it.
    """
    p = torch.as_tensor(dynamic_probability).detach().clamp(0.0, 1.0)
    c = torch.as_tensor(fm_confidence).detach().to(p).clamp(0.0, 1.0)
    v = torch.as_tensor(valid).detach().to(device=p.device).to(p.dtype)
    move = p * c * v
    a = float(alpha)
    return (a + (1.0 - a) * move).clamp(0.0, 1.0)


def _neutral_where_invalid(term: torch.Tensor, valid: torch.Tensor | None) -> torch.Tensor:
    if valid is None:
        return term
    return torch.where(valid.to(device=term.device).bool(), term, torch.zeros_like(term))


@torch.no_grad()
def compose_physical_evidence(
    forward_local,
    *,
    weights: RefinementWeights,
    motion_gate_threshold: float = 1.0,
    motion_gate_temperature: float = 0.5,
    center_displacement: torch.Tensor | None = None,
    inverse_consistency=None,
    diffusion_uncertainty: torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Sum the signed physical evidence, in logits, with per-term validity."""
    from utils.local_rigid_field import compute_motion_gated_rigid_evidence

    device = forward_local.motion_magnitude.device
    dtype = forward_local.motion_magnitude.dtype
    local_valid = forward_local.valid.to(device).bool()
    zeros = torch.zeros_like(forward_local.motion_magnitude)

    motion = centered_unit_score(forward_local.motion_magnitude, local_valid)
    rigid = compute_motion_gated_rigid_evidence(
        forward_local.motion_magnitude,
        forward_local.rigid_failure_fraction,
        threshold=float(motion_gate_threshold),
        temperature=float(motion_gate_temperature),
    )
    # Leave-one-centre-out recovery: the neighbours predict motion this point's
    # own flow-matching displacement does not show.  That disagreement, not the
    # motion magnitude itself, is what rescues a false-static Gaussian.
    if center_displacement is None:
        neighbor = zeros.clone()
    else:
        own = torch.linalg.norm(
            torch.as_tensor(center_displacement).detach().to(device=device, dtype=dtype), dim=-1
        ) / forward_local.effective_scale.clamp_min(1.0e-8)
        disagreement = (forward_local.motion_magnitude - own).clamp_min(0.0)
        neighbor = centered_unit_score(disagreement, local_valid)

    strain = centered_unit_score(forward_local.strain_error, local_valid)

    # Confidence gates evidence; it is never added as evidence of its own.
    support = forward_local.inlier_fraction.to(device=device, dtype=dtype).clamp(0.0, 1.0)

    if inverse_consistency is not None:
        reverse_valid = inverse_consistency.reverse_valid.to(device).bool()
        cycle = _neutral_where_invalid(centered_unit_score(inverse_consistency.cycle_error, reverse_valid), reverse_valid)
        rot = _neutral_where_invalid(centered_unit_score(inverse_consistency.rotation_cycle_error, reverse_valid), reverse_valid)
        jac = _neutral_where_invalid(centered_unit_score(inverse_consistency.jacobian_cycle_error, reverse_valid), reverse_valid)
    else:
        cycle = rot = jac = zeros.clone()
    uncertainty = zeros.clone() if diffusion_uncertainty is None else centered_unit_score(
        torch.as_tensor(diffusion_uncertainty).detach().to(device=device, dtype=dtype).reshape(-1), local_valid
    )

    terms = {
        "motion": weights.motion * motion,
        "rigid": weights.rigid * rigid * support,
        "neighbor": weights.neighbor * neighbor * support,
        "strain": -weights.strain * strain,
        "cycle": -weights.cycle * cycle,
        "rotation_cycle": -weights.rotation_cycle * rot,
        "jacobian_cycle": -weights.jacobian_cycle * jac,
        "uncertainty": -weights.uncertainty * uncertainty,
    }
    evidence = torch.stack(list(terms.values()), dim=0).sum(dim=0)
    evidence = _neutral_where_invalid(evidence, local_valid)
    evidence = torch.where(torch.isfinite(evidence), evidence, torch.zeros_like(evidence))
    diagnostics = dict(terms)
    diagnostics.update({
        "support_gate": support,
        "motion_gate": compute_motion_gate_value(
            forward_local.motion_magnitude, motion_gate_threshold, motion_gate_temperature
        ),
        "rigid_failure_fraction": forward_local.rigid_failure_fraction,
    })
    return evidence, diagnostics


def compute_motion_gate_value(motion_magnitude: torch.Tensor, threshold: float, temperature: float) -> torch.Tensor:
    from utils.local_rigid_field import compute_motion_gate

    return compute_motion_gate(motion_magnitude, threshold=threshold, temperature=temperature)


def compute_fixed_proposal(prior_logit: torch.Tensor, evidence: torch.Tensor) -> torch.Tensor:
    """p* = sigmoid(L0 + E). Depends only on the prior and current evidence."""
    return torch.sigmoid(prior_logit + evidence.to(prior_logit))


def ema_update(
    previous: torch.Tensor,
    proposal: torch.Tensor,
    alpha: float,
    *,
    valid: torch.Tensor | None = None,
) -> torch.Tensor:
    updated = (1.0 - float(alpha)) * previous + float(alpha) * proposal
    keep = torch.isfinite(updated)
    if valid is not None:
        keep = keep & valid.to(device=updated.device).bool()
    return torch.where(keep, updated, previous).clamp(0.0, 1.0)


def assign_discrete_labels(
    probability: torch.Tensor,
    *,
    ambiguous_low: float = 0.30,
    ambiguous_high: float = 0.70,
    geometry_valid: torch.Tensor | None = None,
    fm_confidence: torch.Tensor | None = None,
    weak_confidence: float = 0.25,
) -> tuple[torch.Tensor, torch.Tensor]:
    """STATIC / DYNAMIC / AMBIGUOUS from genuine uncertainty only.

    Ambiguity means "we cannot confidently decide", not "some auxiliary
    signal was unavailable".  Reverse-query availability deliberately plays
    no part here: a point with strong forward and local evidence stays
    classified even when the backward branch could not be evaluated.
    """
    probability = torch.as_tensor(probability).detach()
    ambiguous = (probability > float(ambiguous_low)) & (probability < float(ambiguous_high))
    if geometry_valid is not None:
        degenerate = ~geometry_valid.to(device=probability.device).bool()
        if fm_confidence is not None:
            # Degenerate geometry only forces ambiguity where the prior is also
            # weak; a confident prior with no usable geometry keeps the prior.
            degenerate = degenerate & (
                torch.as_tensor(fm_confidence).detach().to(probability) < float(weak_confidence)
            )
        ambiguous = ambiguous | degenerate
    label = torch.where(
        probability >= float(ambiguous_high),
        torch.full_like(probability, LABEL_DYNAMIC, dtype=torch.long),
        torch.where(
            probability <= float(ambiguous_low),
            torch.full_like(probability, LABEL_STATIC, dtype=torch.long),
            torch.full_like(probability, LABEL_AMBIGUOUS, dtype=torch.long),
        ),
    )
    label = torch.where(ambiguous, torch.full_like(label, LABEL_AMBIGUOUS), label)
    return label, ambiguous


def _binary_scores(predicted: torch.Tensor, gt_moving: torch.Tensor) -> dict[str, float]:
    tp = int((predicted & gt_moving).sum())
    fp = int((predicted & ~gt_moving).sum())
    fn = int((~predicted & gt_moving).sum())
    tn = int((~predicted & ~gt_moving).sum())
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    return {
        "precision": precision,
        "recall": recall,
        "f1": 2 * precision * recall / max(precision + recall, 1.0e-9),
        "accuracy": (tp + tn) / max(tp + tn + fp + fn, 1),
        "true_positive": tp,
        "false_positive": fp,
        "false_negative": fn,
        "predicted_dynamic_fraction": float(predicted.float().mean()),
    }


def compute_classification_metrics(
    probability: torch.Tensor,
    label: torch.Tensor,
    gt_moving: torch.Tensor,
    *,
    threshold: float = 0.5,
) -> dict[str, Any]:
    """Every metric family, each tied to an explicit decision rule.

    Reporting one number computed from a rule that is not the rule used to
    write the saved labels is how a probe ends up claiming a score the
    downstream consumer will never see.
    """
    probability = probability.detach().cpu().reshape(-1)
    label = label.detach().cpu().reshape(-1)
    gt = gt_moving.detach().cpu().reshape(-1).bool()
    ambiguous = label == LABEL_AMBIGUOUS

    by_probability = _binary_scores(probability >= float(threshold), gt)
    by_probability["rule"] = f"p_refined >= {float(threshold)}"

    keep = ~ambiguous
    if bool(keep.any()):
        selective = _binary_scores(label[keep] == LABEL_DYNAMIC, gt[keep])
    else:
        selective = {k: 0.0 for k in ("precision", "recall", "f1", "accuracy")}
    selective["coverage"] = float(keep.float().mean())
    selective["rule"] = "label == DYNAMIC, ambiguous excluded"

    as_static = _binary_scores(label == LABEL_DYNAMIC, gt)
    as_static["rule"] = "label == DYNAMIC, ambiguous treated as static"
    as_dynamic = _binary_scores((label == LABEL_DYNAMIC) | ambiguous, gt)
    as_dynamic["rule"] = "label in {DYNAMIC, AMBIGUOUS}, ambiguous treated as dynamic"

    ambiguity = {
        "ambiguous_fraction": float(ambiguous.float().mean()),
        "ambiguous_count": int(ambiguous.sum()),
        "ambiguous_gt_dynamic_fraction": float(gt[ambiguous].float().mean()) if bool(ambiguous.any()) else 0.0,
        "ambiguous_gt_static_fraction": float((~gt[ambiguous]).float().mean()) if bool(ambiguous.any()) else 0.0,
        "static_count": int((label == LABEL_STATIC).sum()),
        "dynamic_count": int((label == LABEL_DYNAMIC).sum()),
    }
    return {
        "ground_truth_used_as_algorithm_input": False,
        "gt_moving": int(gt.sum()),
        "probability_metrics": by_probability,
        "selective_metrics": selective,
        "f1_ambiguous_as_static": as_static,
        "f1_ambiguous_as_dynamic": as_dynamic,
        "ambiguity_metrics": ambiguity,
    }


@torch.no_grad()
def refine_dynamic_probabilities(
    state: DynamicRefinementState,
    forward_local,
    inverse_consistency=None,
    diffusion_uncertainty: torch.Tensor | None = None,
    *,
    center_displacement: torch.Tensor | None = None,
    fm_confidence: torch.Tensor | None = None,
    ema: float = 0.15,
    weights: RefinementWeights | None = None,
    motion_gate_threshold: float = 1.0,
    motion_gate_temperature: float = 0.5,
    ambiguous_low: float = 0.30,
    ambiguous_high: float = 0.70,
) -> RefinementReport:
    """One conservative refinement step on the frame-0 probabilities."""
    weights = weights or RefinementWeights()
    evidence, terms = compose_physical_evidence(
        forward_local,
        weights=weights,
        motion_gate_threshold=motion_gate_threshold,
        motion_gate_temperature=motion_gate_temperature,
        center_displacement=center_displacement,
        inverse_consistency=inverse_consistency,
        diffusion_uncertainty=diffusion_uncertainty,
    )
    prior_logit = state.fm_prior_logit.to(evidence.device)
    proposal = compute_fixed_proposal(prior_logit, evidence)
    local_valid = forward_local.valid.to(evidence.device).bool()
    updated = ema_update(state.refined_prob.to(evidence.device), proposal, float(ema), valid=local_valid)
    label, ambiguous = assign_discrete_labels(
        updated,
        ambiguous_low=float(ambiguous_low),
        ambiguous_high=float(ambiguous_high),
        geometry_valid=local_valid,
        fm_confidence=fm_confidence,
    )

    dtype = updated.dtype
    stats = {
        "mean_p_dynamic": float(updated.mean()),
        "dynamic_fraction": float((label == LABEL_DYNAMIC).to(dtype).mean()),
        "static_fraction": float((label == LABEL_STATIC).to(dtype).mean()),
        "ambiguous_fraction": float(ambiguous.to(dtype).mean()),
        "mean_abs_refined_prob_change": float((updated - state.refined_prob.to(updated)).abs().mean()),
        "mean_abs_evidence": float(evidence.abs().mean()),
        "valid_fraction": float(local_valid.to(dtype).mean()),
        "mean_motion_magnitude": float(forward_local.motion_magnitude[local_valid].mean()) if bool(local_valid.any()) else 0.0,
        "mean_rigid_gain": float(forward_local.rigid_gain[local_valid].mean()) if bool(local_valid.any()) else 0.0,
        "mean_rigid_failure_fraction": float(forward_local.rigid_failure_fraction[local_valid].mean()) if bool(local_valid.any()) else 0.0,
        "mean_strain": float(forward_local.strain_error[local_valid].mean()) if bool(local_valid.any()) else 0.0,
        "mean_inlier_fraction": float(forward_local.inlier_fraction.mean()),
        "mean_robust_delta": float(forward_local.robust_delta.mean()),
    }
    if inverse_consistency is not None:
        reverse_valid = inverse_consistency.reverse_valid.bool()
        stats["reverse_valid_fraction"] = float(reverse_valid.to(dtype).mean())
        if bool(reverse_valid.any()):
            stats.update({
                "mean_cycle_error": float(inverse_consistency.cycle_error[reverse_valid].mean()),
                "mean_rotation_cycle_error": float(inverse_consistency.rotation_cycle_error[reverse_valid].mean()),
                "mean_jacobian_cycle_error": float(inverse_consistency.jacobian_cycle_error[reverse_valid].mean()),
                "mean_reverse_magnitude_ratio": float(inverse_consistency.reverse_magnitude_ratio[reverse_valid].mean()),
                "mean_reverse_direction_cosine": float(inverse_consistency.reverse_direction_cosine[reverse_valid].mean()),
            })

    return RefinementReport(
        probability=updated,
        proposal_probability=proposal,
        label=label,
        ambiguous=ambiguous,
        evidence=evidence,
        logit_terms=terms,
        stats=stats,
    )
