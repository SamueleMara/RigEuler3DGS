"""Convergence certification for the fixed-prior IEKF.

CODEX_IMPLEMENT_SPARSE_ALPHA_AWARE_IEKF.md sections 14, 15, 34.

An important honesty point about what can and cannot be certified.

Monotone decrease of the accepted MAP energy is TRUE BY CONSTRUCTION: section
14 only accepts a step when J(delta + alpha Delta) < J(delta). Reporting "the
energy decreased monotonically" therefore certifies almost nothing on its own --
it would still hold if every step were rejected down to alpha_min and the
estimator crawled.

A real certificate needs four things, which is what this module computes:

  1. MONOTONICITY       accepted energies are non-increasing (the construction
                        holds -- a violation means a bug, not slow progress).
  2. GAIN RATIO         rho = (actual decrease) / (predicted decrease), where
                        the prediction comes from the SAME quadratic model the
                        PCG solved. rho ~ 1 means the linearization is
                        trustworthy; rho << 1 means H is wrong or the step left
                        the trust region. This is what actually validates the
                        analytic Jacobian inside the optimizer.
  3. STATIONARITY       ||b|| = ||H^T R^-1 r - Lambda^- delta|| must decrease.
                        This is the gradient of the true objective at the
                        current iterate; driving it to zero is what convergence
                        MEANS. Energy can plateau while the gradient does not.
  4. ACCEPTANCE         the fraction of accepted steps and the line-search
                        alphas, which say whether progress is real or stalled.

No autograd: ``b`` and the quadratic model come from the analytic operators.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch


@dataclass
class IterationRecord:
    """One nonlinear iteration, with everything needed to judge it."""

    iteration: int
    accepted: bool
    energy_before: float
    energy_after: float
    predicted_decrease: float
    actual_decrease: float
    gradient_norm: float
    step_norm: float
    alpha: float
    lm: float
    pcg_iterations: int
    pcg_rel_residual: float

    @property
    def gain_ratio(self) -> float:
        """rho. Undefined when the model predicts no decrease at all."""
        if abs(self.predicted_decrease) < 1e-30:
            return float("nan")
        return self.actual_decrease / self.predicted_decrease

    def significant(self, rel_floor: float = 1e-6) -> bool:
        """Is this iteration's decrease above the numerical noise floor?

        Once the solve has converged, both the predicted and the actual decrease
        sit at float round-off and their RATIO becomes meaningless -- it is a
        ratio of two noise terms and can be arbitrarily large. Measured on a
        converged run: rho was 0.81, 1.13, 1.77, 1.01 while the decrease was
        real, then 4.9, 13, 207, 2211 once the energy had stopped moving at the
        6th significant figure. Judging the linearization on those last values
        would condemn a perfectly good Jacobian, so they are excluded from the
        statistic and counted separately.
        """
        scale = max(abs(self.energy_before), 1.0)
        return abs(self.predicted_decrease) > rel_floor * scale


@dataclass
class ConvergenceCertificate:
    """The verdict, with the evidence that produced it."""

    records: list[IterationRecord] = field(default_factory=list)
    # A gain ratio this close to 1 means the quadratic model built from the
    # analytic Jacobian predicted the true nonlinear decrease. Below this the
    # linearization is suspect, which is a Jacobian problem, not a tuning one.
    gain_ratio_tolerance: float = 0.25
    stationarity_drop: float = 0.5
    # Decreases below this fraction of the energy are numerical noise; see
    # IterationRecord.significant.
    significance_floor: float = 1e-6

    def add(self, record: IterationRecord) -> None:
        self.records.append(record)

    # -- the four criteria ---------------------------------------------------
    @property
    def accepted(self) -> list[IterationRecord]:
        return [r for r in self.records if r.accepted]

    @property
    def monotone(self) -> bool:
        acc = self.accepted
        return all(r.energy_after <= r.energy_before + 1e-9 for r in acc)

    @property
    def monotone_violations(self) -> list[int]:
        return [r.iteration for r in self.accepted
                if r.energy_after > r.energy_before + 1e-9]

    @property
    def significant(self) -> list[IterationRecord]:
        return [r for r in self.accepted if r.significant(self.significance_floor)]

    @property
    def n_at_noise_floor(self) -> int:
        return len(self.accepted) - len(self.significant)

    @property
    def median_gain_ratio(self) -> float:
        """Median rho over iterations whose decrease was above the noise floor."""
        rhos = [r.gain_ratio for r in self.significant
                if r.gain_ratio == r.gain_ratio]   # drop NaN
        if not rhos:
            return float("nan")
        rhos.sort()
        return rhos[len(rhos) // 2]

    @property
    def model_trustworthy(self) -> bool:
        """rho near 1 on the SIGNIFICANT accepted steps: the analytic H
        predicted the true nonlinear decrease. If every step was at the noise
        floor there is nothing to judge, and that is reported rather than
        silently passed."""
        rho = self.median_gain_ratio
        if rho != rho:
            return False
        return abs(rho - 1.0) <= (1.0 - self.gain_ratio_tolerance)

    @property
    def stationarity_improved(self) -> bool:
        if len(self.records) < 2:
            return False
        g0 = self.records[0].gradient_norm
        gN = min(r.gradient_norm for r in self.records)
        return gN <= self.stationarity_drop * g0 if g0 > 0 else True

    @property
    def acceptance_rate(self) -> float:
        return len(self.accepted) / max(len(self.records), 1)

    @property
    def energy_reduction(self) -> float:
        if not self.records:
            return 0.0
        first = self.records[0].energy_before
        last = min(r.energy_after for r in self.records)
        return 1.0 - last / first if first > 0 else 0.0

    @property
    def certified(self) -> bool:
        return (self.monotone and self.model_trustworthy
                and self.stationarity_improved and self.acceptance_rate > 0.0)

    def reasons(self) -> list[str]:
        out = []
        if not self.monotone:
            out.append(f"energy INCREASED on accepted iterations "
                       f"{self.monotone_violations} -- this is a bug, the "
                       f"acceptance test should have rejected them")
        if not self.model_trustworthy:
            rho = self.median_gain_ratio
            if rho != rho:
                out.append("every accepted decrease was at the numerical noise "
                           "floor, so the gain ratio could not be evaluated: "
                           "the linearization is untested by this run")
            else:
                out.append(f"median gain ratio {rho:.3f} is far from 1 over "
                           f"{len(self.significant)} significant iterations: the "
                           f"quadratic model built from the analytic Jacobian does "
                           f"not predict the true decrease, so H or the "
                           f"linearization is suspect")
        if not self.stationarity_improved:
            g0 = self.records[0].gradient_norm if self.records else float("nan")
            gN = min((r.gradient_norm for r in self.records), default=float("nan"))
            out.append(f"gradient norm did not fall by {self.stationarity_drop:.0%} "
                       f"({g0:.4g} -> {gN:.4g}): the iterate is not approaching a "
                       f"stationary point, even if the energy moved")
        if self.acceptance_rate == 0.0:
            out.append("no step was ever accepted")
        return out

    def format(self) -> str:
        rows = [f"  {'it':>3} {'acc':>4} {'energy':>13} {'actual':>11} "
                f"{'predicted':>11} {'rho':>7} {'||b||':>11} {'alpha':>6} "
                f"{'lm':>9} {'pcg':>4}"]
        rows.append("  " + "-" * 92)
        for r in self.records:
            rho = r.gain_ratio
            tag = "" if r.significant(self.significance_floor) else " (noise floor)"
            rows.append(
                f"  {r.iteration:>3} {'Y' if r.accepted else 'n':>4} "
                f"{r.energy_after:>13.5f} {r.actual_decrease:>11.4e} "
                f"{r.predicted_decrease:>11.4e} "
                f"{rho:>7.3f} {r.gradient_norm:>11.4e} {r.alpha:>6.3f} "
                f"{r.lm:>9.2e} {r.pcg_iterations:>4}{tag}")
        rows.append("")
        rows.append(f"  monotone (accepted)      : {self.monotone}"
                    + ("" if self.monotone else f"  violations {self.monotone_violations}"))
        rows.append(f"  median gain ratio rho    : {self.median_gain_ratio:.4f}"
                    f"   over {len(self.significant)} significant iteration(s), "
                    f"{self.n_at_noise_floor} at the noise floor")
        rows.append(f"  gradient ||b||           : "
                    f"{self.records[0].gradient_norm:.4e} -> "
                    f"{min(r.gradient_norm for r in self.records):.4e}"
                    if self.records else "  gradient: n/a")
        rows.append(f"  acceptance rate          : {self.acceptance_rate*100:.0f}%")
        rows.append(f"  energy reduction         : {self.energy_reduction*100:.2f}%")
        verdict = "CERTIFIED" if self.certified else "NOT CERTIFIED"
        rows.append(f"  => CONVERGENCE {verdict}")
        for reason in self.reasons():
            rows.append(f"     - {reason}")
        return "\n".join(rows)


def quadratic_model_decrease(step: torch.Tensor, b: torch.Tensor, A_mv) -> float:
    """Decrease the SOLVED quadratic model predicts for this step.

    The local model is  m(d) = J(0) - d^T b + 1/2 d^T A d, so the predicted
    decrease is  d^T b - 1/2 d^T A d.  Using the same A_mv the PCG used means
    rho compares like with like: a rho far from 1 then genuinely indicts the
    linearization rather than a mismatched model.
    """
    Ad = A_mv(step)
    return float((step * b).sum() - 0.5 * (step * Ad).sum())
