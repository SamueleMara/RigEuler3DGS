"""Convergence certificate V2.

CODEX_STRICT_CORRECTIONS_IEKF_JOINT_DISCOVERY_V2.md section 17.

Section 17's first line is the whole point: "Do not mark a frame/state
CERTIFIED from monotone objective alone." A monotone objective says the step
control is behaving; it says nothing about whether the quantity you wanted was
observable, whether the linear solve met its own tolerance, or whether the
result sits at a stationary point of the true objective.

So the outcome is a STATUS, not a boolean, drawn from section 17's list:

    CERTIFIED_LOCAL_MAP            all ten conditions hold
    CONVERGED_BUT_UNOBSERVABLE     converged, but the state is not measured
    TRUST_REGION_STALLED           the radius collapsed; no step helps
    LINEAR_SOLVE_INEXACT           the subproblem never met its forcing rule
    MAX_NONLINEAR_ITERS            ran out of iterations still making progress
    REJECTED_MODEL                 nothing was accepted at all

and section 17 is explicit that these must not be collapsed back into one
boolean. The normalized first-order optimality it defines,

    eta_g = ||S^{-1/2} g|| / (1 + ||S^{1/2} delta||)

is scale-aware on both sides: the gradient in the dual metric, the state in the
primal one. Its threshold is persisted alongside the verdict so a reader can
see what "small" meant.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

CERTIFIED_LOCAL_MAP = "CERTIFIED_LOCAL_MAP"
CONVERGED_BUT_UNOBSERVABLE = "CONVERGED_BUT_UNOBSERVABLE"
TRUST_REGION_STALLED = "TRUST_REGION_STALLED"
LINEAR_SOLVE_INEXACT = "LINEAR_SOLVE_INEXACT"
MAX_NONLINEAR_ITERS = "MAX_NONLINEAR_ITERS"
REJECTED_MODEL = "REJECTED_MODEL"

ALL_STATUSES = (CERTIFIED_LOCAL_MAP, CONVERGED_BUT_UNOBSERVABLE,
                TRUST_REGION_STALLED, LINEAR_SOLVE_INEXACT,
                MAX_NONLINEAR_ITERS, REJECTED_MODEL)

# Statuses that mean "this frame's state may be used as a converged estimate".
USABLE = (CERTIFIED_LOCAL_MAP, CONVERGED_BUT_UNOBSERVABLE)


@dataclass
class CertificateThresholds:
    """Persisted alongside the verdict, so "small" is never implicit."""

    eta_g_max: float = 1e-3
    step_norm_scaled_max: float = 1e-2
    gain_ratio_min: float = 0.1
    gain_ratio_max: float = 10.0
    radius_stall: float = 1e-7
    max_weak_claimed: int = 0

    def to_dict(self) -> dict:
        return {k: getattr(self, k) for k in self.__dataclass_fields__}


def normalized_optimality(scaled_gradient_norm: float,
                          scaled_state_norm: float) -> float:
    """Section 17's eta_g = ||S^{-1/2}g|| / (1 + ||S^{1/2}delta||).

    The denominator is what makes this a RELATIVE statement: a gradient of 10
    is small if the state moved by 10000 and large if it moved by 0.001.
    """
    if not math.isfinite(scaled_gradient_norm):
        return float("inf")
    return scaled_gradient_norm / (1.0 + max(scaled_state_norm, 0.0))


@dataclass
class CertificateV2:
    """Section 17's ten conditions, each recorded with its evidence."""

    status: str = REJECTED_MODEL
    conditions: dict = field(default_factory=dict)
    evidence: dict = field(default_factory=dict)
    thresholds: CertificateThresholds = field(default_factory=CertificateThresholds)
    failed: list[str] = field(default_factory=list)

    @property
    def is_certified(self) -> bool:
        """True ONLY for the full certificate. Deliberately not a synonym for
        'the frame is usable' -- see :data:`USABLE`."""
        return self.status == CERTIFIED_LOCAL_MAP

    def to_dict(self) -> dict:
        return {"status": self.status, "conditions": self.conditions,
                "evidence": self.evidence, "failed_conditions": self.failed,
                "thresholds": self.thresholds.to_dict()}

    def format(self) -> str:
        lines = [f"convergence certificate V2: {self.status}"]
        for name, ok in self.conditions.items():
            lines.append(f"  [{'x' if ok else ' '}] {name}")
        if self.failed:
            lines.append(f"  failing: {', '.join(self.failed)}")
        ev = self.evidence
        lines.append(
            f"  eta_g {ev.get('eta_g', float('nan')):.3e} "
            f"(limit {self.thresholds.eta_g_max:.1e})   "
            f"scaled step {ev.get('final_step_norm_scaled', float('nan')):.3e}   "
            f"rho {ev.get('last_significant_rho', float('nan')):.4f}")
        return "\n".join(lines)


def assess(trust_history, observability=None, *, certificate=None,
           innovation_stats: dict | None = None,
           observable_error_decreased: bool | None = None,
           weak_directions_claimed: int = 0,
           max_iters_reached: bool = False,
           thresholds: CertificateThresholds | None = None) -> CertificateV2:
    """Evaluate section 17's ten conditions and pick a status.

    ``trust_history`` is a :class:`~estimator.trust_region.TrustRegionHistory`.
    ``observability`` is an :class:`~estimator.observability.ObservabilityReport`.
    ``observable_error_decreased`` is condition 10 and applies only to synthetic
    tests where a ground-truth state exists; ``None`` means "not applicable",
    which is recorded as such rather than silently passing.
    """
    th = thresholds or CertificateThresholds()
    recs = list(getattr(trust_history, "records", []) or [])
    accepted = [r for r in recs if r.accepted]
    cond: dict = {}
    ev: dict = {}

    # 1. monotone accepted objective sequence
    objs = [r.objective_after for r in accepted]
    befores = [r.objective_before for r in accepted]
    cond["1_monotone_objective"] = bool(
        accepted and all(a <= b + 1e-12 for a, b in zip(objs, befores)))
    ev["num_accepted"] = len(accepted)
    ev["num_iterations"] = len(recs)
    if accepted:
        ev["objective_initial"] = befores[0]
        ev["objective_final"] = objs[-1]
        ev["energy_reduction"] = (
            1.0 - objs[-1] / befores[0] if abs(befores[0]) > 0 else 0.0)

    # 2. last significant gain ratio finite and in bounds
    sig = [r for r in accepted if math.isfinite(r.rho)]
    last_rho = sig[-1].rho if sig else float("nan")
    ev["last_significant_rho"] = last_rho
    ev["num_significant"] = len(sig)
    cond["2_gain_ratio_in_bounds"] = bool(
        sig and th.gain_ratio_min <= last_rho <= th.gain_ratio_max)

    # 3. a trust-region step was accepted
    cond["3_step_accepted"] = bool(accepted)

    # 4/5. scaled first-order optimality and final scaled step
    last = recs[-1] if recs else None
    g_scaled = getattr(last, "gradient_norm_scaled", float("nan")) if last else float("nan")
    step_scaled = getattr(last, "step_norm_scaled", float("nan")) if last else float("nan")
    state_scaled = ev.get("state_norm_scaled", 0.0)
    eta_g = normalized_optimality(g_scaled, state_scaled)
    ev["scaled_gradient_norm"] = g_scaled
    ev["eta_g"] = eta_g
    ev["final_step_norm_scaled"] = step_scaled
    cond["4_first_order_optimality_small"] = bool(eta_g <= th.eta_g_max)
    cond["5_final_step_small"] = bool(
        math.isfinite(step_scaled) and step_scaled <= th.step_norm_scaled_max)

    # 6. linear solve met its forcing criterion, or is LABELLED boundary-truncated
    from .trust_region import STOP_BOUNDARY, STOP_FORCING, STOP_NEGATIVE_CURVATURE
    last_reason = getattr(last, "pcg_stop_reason", None) if last else None
    ev["final_pcg_stop_reason"] = last_reason
    ev["boundary_truncated"] = last_reason in (STOP_BOUNDARY, STOP_NEGATIVE_CURVATURE)
    cond["6_linear_solve_labelled"] = last_reason in (
        STOP_FORCING, STOP_BOUNDARY, STOP_NEGATIVE_CURVATURE)
    ev["linear_solve_met_forcing"] = last_reason == STOP_FORCING

    # 7. innovation / NIS statistics finite
    nis = innovation_stats or {}
    finite_nis = all(math.isfinite(float(v)) for v in nis.values()
                     if isinstance(v, (int, float)))
    cond["7_innovation_finite"] = bool(finite_nis)
    ev["innovation"] = nis

    # 8. observable-rank diagnostics reported
    cond["8_observability_reported"] = observability is not None
    if observability is not None:
        o = observability.to_dict()
        ev["observability"] = o
        ev["fraction_weak"] = o.get("fraction_weak")
        ev["active_dimension"] = o.get("active_dimension")

    # 9. no unsupported weak direction claimed as recovered
    cond["9_no_weak_direction_claimed"] = int(weak_directions_claimed) <= th.max_weak_claimed
    ev["weak_directions_claimed"] = int(weak_directions_claimed)

    # 10. synthetic only: observable-subspace error decreased
    if observable_error_decreased is None:
        cond["10_observable_error_decreased"] = True
        ev["observable_error_check"] = "not_applicable_no_ground_truth"
    else:
        cond["10_observable_error_decreased"] = bool(observable_error_decreased)
        ev["observable_error_check"] = bool(observable_error_decreased)

    failed = [k for k, v in cond.items() if not v]

    # --- status selection, in priority order ---------------------------
    final_radius = getattr(last, "trust_radius_after", float("nan")) if last else float("nan")
    ev["final_trust_radius"] = final_radius
    if not accepted:
        status = REJECTED_MODEL
    elif math.isfinite(final_radius) and final_radius <= th.radius_stall:
        status = TRUST_REGION_STALLED
    elif not cond["6_linear_solve_labelled"]:
        status = LINEAR_SOLVE_INEXACT
    elif (observability is not None
          and float(observability.fraction_weak) > 0.5
          and cond["1_monotone_objective"] and cond["3_step_accepted"]):
        # Checked BEFORE the all-conditions branch on purpose. A run can
        # satisfy every numerical condition -- monotone, small gradient, tiny
        # step -- while most of the state was never measured at all. Those
        # conditions are then describing the prior, not the data. Section 17
        # keeps this a separate status precisely so it cannot be reported as
        # CERTIFIED.
        status = CONVERGED_BUT_UNOBSERVABLE
    elif all(cond.values()):
        status = CERTIFIED_LOCAL_MAP
    elif max_iters_reached:
        status = MAX_NONLINEAR_ITERS
    elif not cond["4_first_order_optimality_small"] or not cond["5_final_step_small"]:
        status = MAX_NONLINEAR_ITERS
    else:
        status = CONVERGED_BUT_UNOBSERVABLE

    return CertificateV2(status=status, conditions=cond, evidence=ev,
                         thresholds=th, failed=failed)
