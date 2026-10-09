"""Explicit scaled trust region with Steihaug truncated PCG.

CODEX_STRICT_CORRECTIONS_IEKF_JOINT_DISCOVERY_V2.md section 4.

Why this replaces the previous LM + line search. At full scale the old code
only made progress with a 25-iteration PCG budget: at 100 or 400 iterations
*zero* steps were accepted. The linear-solver iteration count was acting as the
step-size controller -- early termination was an uncontrolled Tikhonov-like
regularizer (section 1.3). That is forbidden. The step length must be set by an
explicit trust radius, and the linear solve must be allowed to be as accurate as
the forcing criterion asks for without changing the physical step.

Three pieces, all from section 4:

  * a SCALING METRIC S (section 4.1). ||s||_2 is meaningless across mixed
    translation / rotation / velocity units -- a radius that is generous for
    metres is absurd for radians. The radius is measured in ||.||_S.
  * STEIHAUG truncated PCG (section 4.2), which stops at the trust boundary or
    on non-positive curvature and returns the boundary point, instead of
    running on to the unconstrained normal-equation solution.
  * an INEXACT-NEWTON forcing rule (section 4.3), so the linear tolerance is
    loose far from the solution and tight near it -- by design rather than by
    iteration budget.

No autograd anywhere: A is applied through the analytic matrix-free operator.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import torch

EPS = 1e-30

# section 4.2's stop reasons, reported rather than inferred
STOP_FORCING = "forcing_criterion_met"
STOP_BOUNDARY = "trust_boundary_hit"
STOP_NEGATIVE_CURVATURE = "non_positive_curvature"
STOP_MAX_ITERS = "max_iterations"
STOP_ZERO_GRADIENT = "zero_gradient"
STOP_NONFINITE = "non_finite"


@dataclass
class ScalingMetric:
    """Block-diagonal S for ``||s||_S^2 = s^T S s`` (section 4.1).

    Stored as a per-element positive weight of the same shape as the state, so
    the quadratic form is a weighted sum. For the 12D Gaussian error state the
    blocks are [dmu(3), dphi(3), dv(3), domega(3)]; for a 6D supernode pose
    correction only the first two are used.
    """

    weight: torch.Tensor          # [..., D], strictly positive

    # Why the weights are divided by the element count (``rms=True``).
    #
    # A plain ``sqrt(sum_i w_i s_i^2)`` is a sum over all 6N or 12N components,
    # so the SAME physical per-Gaussian motion has a scaled norm that grows
    # like sqrt(N). Measured at full scene scale: 11320 Gaussians each moving
    # 0.01 with sigma_mu = 0.05 gives ||s||_S = sqrt(11320 * 0.04) = 21.3,
    # against a default radius of 1.0. The true step was then 21x outside the
    # region, Steihaug exited at the boundary on its FIRST CG iteration at
    # every budget, and the radius halved each outer iteration because the
    # resulting gain ratio sat right on eta_weak. The method was correct and
    # useless -- it was taking 1/21 of the step it should have.
    #
    # Normalizing by the element count makes the radius mean "the RMS state
    # component moves this many prior sigmas", which is independent of scene
    # size and therefore portable: radius 1.0 means the same physical thing at
    # 6 Gaussians and at 11320. This is also what makes section 20's T-TR3
    # (world-unit rescaling) a meaningful test rather than a tautology.

    @staticmethod
    def from_physical_scales(shape, device, dtype, *, sigma_mu=0.05, sigma_phi=0.05,
                             sigma_v=0.5, sigma_omega=0.5,
                             rms: bool = True) -> "ScalingMetric":
        """S = diag(1/sigma^2) per component, optionally RMS-normalized.

        With ``rms=True`` a step whose every component is one prior sigma has
        ||s||_S = 1 regardless of how many Gaussians there are.
        """
        d = int(shape[-1])
        if d == 6:
            sig = [sigma_mu] * 3 + [sigma_phi] * 3
        elif d == 12:
            sig = [sigma_mu] * 3 + [sigma_phi] * 3 + [sigma_v] * 3 + [sigma_omega] * 3
        else:
            raise ValueError(f"state dimension {d} is neither 6 nor 12")
        w = torch.tensor([1.0 / (s * s) for s in sig], device=device, dtype=dtype)
        w = w.expand(shape).contiguous()
        if rms:
            w = w / float(w.numel())
        return ScalingMetric(weight=w)

    @staticmethod
    def from_prior_information(Lambda: torch.Tensor, *, floor: float = 1e-8,
                               rms: bool = True) -> "ScalingMetric":
        """S from the prior precision diagonal (section 4.1's first option)."""
        diag = Lambda.diagonal(dim1=-2, dim2=-1).clamp_min(floor).contiguous()
        if rms:
            diag = diag / float(diag.numel())
        return ScalingMetric(weight=diag)

    def norm(self, s: torch.Tensor) -> float:
        return float(torch.sqrt((self.weight * s * s).sum().clamp_min(0.0)))

    def inner(self, a: torch.Tensor, b: torch.Tensor) -> float:
        return float((self.weight * a * b).sum())

    def scaled_gradient_norm(self, g: torch.Tensor) -> float:
        """||S^{-1/2} g||, the gradient measured in the dual metric (section 17)."""
        return float(torch.sqrt((g * g / self.weight).sum().clamp_min(0.0)))

    def to_dict(self) -> dict:
        w = self.weight.reshape(-1, self.weight.shape[-1])[0]
        # Report the sigma the user set, not the RMS-scaled internal weight:
        # undo the 1/numel normalization so "implied_sigma" is readable.
        w_phys = w * float(self.weight.numel())
        return {"per_component_weight": w.detach().cpu().tolist(),
                "implied_sigma": (1.0 / w_phys.clamp_min(EPS)).sqrt().detach().cpu().tolist(),
                "num_elements": int(self.weight.numel()),
                "rms_normalized": True}


def forcing_eta(g_norm: float, g0_norm: float, *, c: float = 0.5,
                eta_min: float = 1e-6, eta_max: float = 0.5) -> float:
    """Section 4.3's inexact-Newton forcing rule.

    eta = clip(c * sqrt(||g_k|| / ||g_0||), eta_min, eta_max): loose far from the
    solution, tightening superlinearly as the gradient falls. The point is that
    the ACCURACY of the linear solve is controlled explicitly, so the iteration
    count is free to grow without changing the step.
    """
    if g0_norm <= 0.0:
        return eta_max
    return float(min(max(c * math.sqrt(max(g_norm, 0.0) / g0_norm), eta_min), eta_max))


def boundary_tau(s: torch.Tensor, p: torch.Tensor, radius: float,
                 metric: ScalingMetric) -> float:
    """tau >= 0 with ||s + tau p||_S = radius (section 4.2).

    Positive root of  tau^2 (p,p)_S + 2 tau (s,p)_S + (s,s)_S - radius^2 = 0.
    """
    pp = metric.inner(p, p)
    if pp <= EPS:
        return 0.0
    sp = metric.inner(s, p)
    ss = metric.inner(s, s)
    disc = sp * sp - pp * (ss - radius * radius)
    if disc < 0.0:
        disc = 0.0
    return float((-sp + math.sqrt(disc)) / pp)


@dataclass
class SteihaugResult:
    step: torch.Tensor
    iterations: int
    stop_reason: str
    rel_residual: float
    boundary_hit: bool
    step_norm_scaled: float
    eta: float
    curvature_min: float = float("nan")


def steihaug_cg(A_mv, b: torch.Tensor, radius: float, metric: ScalingMetric,
                *, M_inv=None, max_iters: int = 100, eta: float = 1e-2
                ) -> SteihaugResult:
    """Trust-region truncated preconditioned CG (section 4.2).

    Minimizes  -b^T s + 1/2 s^T A s  subject to ||s||_S <= radius.

    Returns the boundary point as soon as the next CG step would leave the
    region, or as soon as non-positive curvature is detected -- it never
    continues on to the unconstrained solution, which is precisely the failure
    the old early-stopped PCG was hiding.
    """
    s = torch.zeros_like(b)
    r = b.clone()                      # residual of A s = b, at s = 0
    b_norm = float(r.norm())
    if b_norm <= EPS or not math.isfinite(b_norm):
        return SteihaugResult(s, 0, STOP_ZERO_GRADIENT if b_norm <= EPS else STOP_NONFINITE,
                              0.0, False, 0.0, eta)
    z = M_inv(r) if M_inv is not None else r.clone()
    p = z.clone()
    rz = float((r * z).sum())
    curvature_min = float("inf")

    for k in range(1, int(max_iters) + 1):
        Ap = A_mv(p)
        pAp = float((p * Ap).sum())
        curvature_min = min(curvature_min, pAp)
        if not math.isfinite(pAp):
            return SteihaugResult(s, k, STOP_NONFINITE, float(r.norm()) / b_norm,
                                  False, metric.norm(s), eta, curvature_min)
        if pAp <= 0.0:
            # Section 4.2 case 3: the model is unbounded along p, so go to the
            # boundary in that direction and stop.
            tau = boundary_tau(s, p, radius, metric)
            s = s + tau * p
            return SteihaugResult(s, k, STOP_NEGATIVE_CURVATURE,
                                  float((b - A_mv(s)).norm()) / b_norm,
                                  True, metric.norm(s), eta, curvature_min)
        alpha = rz / pAp
        s_next = s + alpha * p
        if metric.norm(s_next) >= radius:
            # Section 4.2 case 2: crossing the boundary -- return the exact
            # intersection, NOT the unconstrained continuation.
            tau = boundary_tau(s, p, radius, metric)
            s = s + tau * p
            return SteihaugResult(s, k, STOP_BOUNDARY,
                                  float((b - A_mv(s)).norm()) / b_norm,
                                  True, metric.norm(s), eta, curvature_min)
        s = s_next
        r = r - alpha * Ap
        rel = float(r.norm()) / b_norm
        if rel <= eta:
            return SteihaugResult(s, k, STOP_FORCING, rel, False,
                                  metric.norm(s), eta, curvature_min)
        z = M_inv(r) if M_inv is not None else r
        rz_new = float((r * z).sum())
        if abs(rz) <= EPS:
            return SteihaugResult(s, k, STOP_FORCING, rel, False,
                                  metric.norm(s), eta, curvature_min)
        p = z + (rz_new / rz) * p
        rz = rz_new

    return SteihaugResult(s, int(max_iters), STOP_MAX_ITERS,
                          float((b - A_mv(s)).norm()) / b_norm, False,
                          metric.norm(s), eta, curvature_min)


@dataclass
class TrustRegionConfig:
    """Section 4.4's radius policy."""

    radius_init: float = 1.0
    radius_max: float = 100.0
    radius_min: float = 1e-8
    eta_accept: float = 0.1          # rho below this: reject
    eta_weak: float = 0.25           # below this: accept only if J decreases, shrink
    eta_good: float = 0.75           # above this with boundary active: enlarge
    shrink_strong: float = 0.25
    shrink: float = 0.5
    enlarge: float = 2.0
    forcing_c: float = 0.5
    forcing_min: float = 1e-6
    forcing_max: float = 0.01
    max_pcg_iters: int = 100


def adapt_radius(radius: float, rho: float, boundary_hit: bool,
                 actual_reduction: float, cfg: TrustRegionConfig) -> tuple[float, bool]:
    """Section 4.4's policy. Returns (new_radius, accept)."""
    if not math.isfinite(rho):
        return max(radius * cfg.shrink_strong, cfg.radius_min), False
    if rho < cfg.eta_accept:
        return max(radius * cfg.shrink_strong, cfg.radius_min), False
    if rho < cfg.eta_weak:
        # Accept only on a genuine decrease of the TRUE objective, and shrink.
        return max(radius * cfg.shrink, cfg.radius_min), actual_reduction > 0.0
    if rho <= cfg.eta_good:
        return radius, True
    # rho > eta_good: only grow when the step was actually limited by the radius
    return (min(radius * cfg.enlarge, cfg.radius_max) if boundary_hit else radius), True


@dataclass
class TrustRegionRecord:
    """Section 4.5's mandatory per-iteration diagnostics."""

    iteration: int
    objective_before: float
    objective_after: float
    predicted_reduction: float
    actual_reduction: float
    rho: float
    trust_radius_before: float
    trust_radius_after: float
    step_norm_scaled: float
    step_norm_translation: float
    step_norm_rotation: float
    pcg_stop_reason: str
    pcg_iterations: int
    pcg_rel_residual: float
    forcing_eta: float
    boundary_hit: bool
    lm_lambda: float
    accepted: bool
    gradient_norm_scaled: float = float("nan")
    optimality_eta_g: float = float("nan")

    def to_dict(self) -> dict:
        return {k: getattr(self, k) for k in self.__dataclass_fields__}


@dataclass
class TrustRegionHistory:
    records: list[TrustRegionRecord] = field(default_factory=list)
    scaling: dict = field(default_factory=dict)

    def add(self, r: TrustRegionRecord) -> None:
        self.records.append(r)

    @property
    def accepted(self) -> list[TrustRegionRecord]:
        return [r for r in self.records if r.accepted]

    def to_dict(self) -> dict:
        return {"scaling_metric": self.scaling,
                "records": [r.to_dict() for r in self.records]}

    def format(self) -> str:
        rows = [f"  {'it':>3} {'acc':>4} {'objective':>13} {'pred':>11} {'actual':>11} "
                f"{'rho':>8} {'radius':>10} {'|s|_S':>9} {'pcg':>4} {'rel':>9} "
                f"{'eta':>8} {'bnd':>4}  stop"]
        rows.append("  " + "-" * 118)
        for r in self.records:
            rows.append(
                f"  {r.iteration:>3} {'Y' if r.accepted else 'n':>4} "
                f"{r.objective_after:>13.5f} {r.predicted_reduction:>11.3e} "
                f"{r.actual_reduction:>11.3e} {r.rho:>8.3f} "
                f"{r.trust_radius_after:>10.3e} {r.step_norm_scaled:>9.3e} "
                f"{r.pcg_iterations:>4} {r.pcg_rel_residual:>9.2e} "
                f"{r.forcing_eta:>8.2e} {'Y' if r.boundary_hit else 'n':>4}  "
                f"{r.pcg_stop_reason}")
        return "\n".join(rows)


def predicted_reduction(step: torch.Tensor, b: torch.Tensor, A_mv) -> float:
    """Section 4.4: Delta_m = b^T s - 1/2 s^T A s."""
    return float((b * step).sum() - 0.5 * (step * A_mv(step)).sum())
