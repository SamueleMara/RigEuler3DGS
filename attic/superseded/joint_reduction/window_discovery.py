"""Joint discovery over the whole RGB-D window.

CODEX_STRICT_CORRECTIONS_IEKF_JOINT_DISCOVERY_V2.md section 11.

Section 11's opening line is the correction: "The rigid twist estimate is an
initializer only." Choosing between a prismatic and a revolute joint from one
frame's twist is choosing from a noisy sample of a quantity that is still
converging. Worse, the previous selection compared a 'rotation share' for the
prismatic hypothesis against a 'pitch share' for the revolute one -- two
different numbers, so the comparison never meant anything.

So both candidates are fitted to the SAME robust whitened RGB-D objective over
the whole bootstrap window,

    J = 1/2 sum_{t,c,p} || y_GT - h(G(eta, q_t, z)) ||^2_{R^-1} + J_time + J_z

differing only in their kinematic basis G:

    prismatic   mu(q) = mu0 + q e                        unknowns e, {q_t}
    revolute    mu(q) = a + R(e,q)(mu0 - a),  a = e x m   unknowns e, m, {q_t}

and compared on that same number (section 11.4), with an absolute quality
threshold AND a relative margin. If neither passes, the answer is UNRESOLVED --
section 11.4 forbids forcing a P/R label on data that supports neither.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import torch

from .line_geometry import PluckerLine

EPS = 1e-12
UNRESOLVED = "UNRESOLVED"


# ---------------------------------------------------------------------------
# 11.3 multi-frame kinematic initializer
# ---------------------------------------------------------------------------
@torch.no_grad()
def initialize_prismatic(mu: torch.Tensor, velocities: list[torch.Tensor],
                         weights: torch.Tensor | None = None) -> dict:
    """Section 11.3: v_{i,t} ~ qdot_t e, omega_{i,t} ~ 0.

    Over several frames at once, so a single noisy frame cannot set the axis.
    The direction is the leading singular vector of the stacked velocity field,
    which is the covariance-weighted least-squares answer for a common
    direction with per-frame speeds.
    """
    rows = []
    for v in velocities:
        w = (torch.ones(v.shape[0], device=v.device, dtype=v.dtype)
             if weights is None else weights.to(v.dtype))
        rows.append(v * w.sqrt().reshape(-1, 1))
    V = torch.cat(rows, dim=0)
    if V.numel() == 0 or float(V.norm()) < EPS:
        return {"observable": False, "reason": "no motion in the window"}
    U, S, Vh = torch.linalg.svd(V, full_matrices=False)
    e = Vh[0]
    e = e / e.norm().clamp_min(EPS)
    qdot = [float((v.mean(dim=0) @ e)) for v in velocities]
    # Sign gauge: a prismatic axis and its negation are the same joint, so fix
    # the sign by the dominant direction of travel rather than leaving it to
    # the SVD's arbitrary choice.
    if sum(qdot) < 0:
        e, qdot = -e, [-q for q in qdot]
    explained = float((S[0] ** 2) / (S ** 2).sum().clamp_min(EPS))
    return {"observable": True, "axis": e, "qdot": qdot,
            "direction_share": explained,
            "singular_values": S.tolist()}


@torch.no_grad()
def initialize_revolute(mu: torch.Tensor, velocities: list[torch.Tensor],
                        omegas: list[torch.Tensor] | None = None,
                        min_omega: float = 1e-4) -> dict:
    """Section 11.3's boxed revolute initializer:

        v_{i,t} ~ qdot_t (e x mu_i + m),   omega_{i,t} ~ qdot_t e

    Solved over the WHOLE window at once as a linear system in (omega_t, u_t)
    per frame, then the per-frame axes are fused into one line. Fusing after
    fitting -- rather than fitting one frame and trusting it -- is what section
    11.3 means by 'not a single twist'.
    """
    dirs, moments, qdots, infos = [], [], [], []
    for t, v in enumerate(velocities):
        line, info = _fit_line_one_frame(mu, v, min_omega=min_omega)
        infos.append(info)
        if line is None:
            continue
        dirs.append(line.direction)
        moments.append(line.moment)
        qdots.append(info.get("qdot", 0.0))
    if not dirs:
        return {"observable": False,
                "reason": "no frame in the window has an observable rotation",
                "per_frame": infos}
    D = torch.stack(dirs)
    # Resolve the +-e sign ambiguity against the first frame before averaging,
    # or opposite-signed frames would cancel into a meaningless mean.
    sign = torch.sign((D @ D[0]).clamp(min=-1.0, max=1.0))
    sign[sign == 0] = 1.0
    D = D * sign.reshape(-1, 1)
    M = torch.stack(moments) * sign.reshape(-1, 1)
    e = D.mean(dim=0)
    e = e / e.norm().clamp_min(EPS)
    m = M.mean(dim=0)
    line = PluckerLine(direction=e, moment=m)
    spread = [float(torch.rad2deg(torch.arccos(torch.clamp(d @ e, -1.0, 1.0))))
              for d in D]
    return {"observable": True, "line": line,
            "qdot": [q * float(s) for q, s in zip(qdots, sign.tolist())],
            "axis_spread_deg": max(spread) if spread else float("nan"),
            "frames_used": len(dirs), "per_frame": infos}


def _fit_line_one_frame(mu, v, min_omega=1e-4):
    from .line_geometry import fit_plucker_from_motion
    return fit_plucker_from_motion(mu, v, min_omega=min_omega)


# ---------------------------------------------------------------------------
# 11.1 / 11.2 the shared window objective
# ---------------------------------------------------------------------------
@dataclass
class WindowFit:
    """One candidate model's fit to the whole window."""

    name: str
    data_term: float                     # robust whitened RGB-D NLL
    time_term: float = 0.0
    support_term: float = 0.0
    prior_term: float = 0.0
    num_parameters: int = 0
    num_residuals: int = 0
    q: list[float] = field(default_factory=list)
    geometry: dict = field(default_factory=dict)
    converged: bool = True
    iterations: int = 0

    @property
    def objective(self) -> float:
        """J, the number section 11.4 requires both models be compared on."""
        return self.data_term + self.time_term + self.support_term + self.prior_term

    @property
    def bic(self) -> float:
        """Section 11.4's optional complexity correction. A revolute joint has
        more parameters than a prismatic one, so a raw objective comparison
        would always favour revolute; BIC charges for that."""
        n = max(self.num_residuals, 1)
        return 2.0 * self.objective + self.num_parameters * math.log(n)

    def normalized_data_term(self) -> float:
        """Per-residual whitened NLL: the absolute quality figure, in units of
        'sigma^2 per measurement', so a threshold on it is interpretable."""
        return self.data_term / max(self.num_residuals, 1)

    def to_dict(self) -> dict:
        return {"model": self.name, "objective": self.objective,
                "data_term": self.data_term, "time_term": self.time_term,
                "support_term": self.support_term, "prior_term": self.prior_term,
                "normalized_data_term": self.normalized_data_term(),
                "bic": self.bic, "num_parameters": self.num_parameters,
                "num_residuals": self.num_residuals, "q": self.q,
                "geometry": self.geometry, "converged": self.converged,
                "iterations": self.iterations}


@dataclass
class SelectionConfig:
    """Section 11.4 requires BOTH of these, not either."""

    max_normalized_data_term: float = 1.0     # absolute quality threshold
    min_relative_margin: float = 0.05         # relative evidence margin
    use_bic: bool = True


def select_model(fits: list[WindowFit], cfg: SelectionConfig | None = None) -> dict:
    """Section 11.4's comparable selection, or UNRESOLVED.

    Both requirements must hold:
      1. the winner must be good enough in ABSOLUTE terms;
      2. it must beat the runner-up by a RELATIVE margin.

    Either failing yields UNRESOLVED rather than a forced label.
    """
    cfg = cfg or SelectionConfig()
    usable = [f for f in fits if f.converged and math.isfinite(f.objective)]
    if not usable:
        return {"selected": UNRESOLVED,
                "reason": "no candidate model produced a finite objective",
                "candidates": [f.to_dict() for f in fits]}

    key = (lambda f: f.bic) if cfg.use_bic else (lambda f: f.objective)
    ranked = sorted(usable, key=key)
    best = ranked[0]
    reasons = []

    quality_ok = best.normalized_data_term() <= cfg.max_normalized_data_term
    if not quality_ok:
        reasons.append(
            f"the best model ({best.name}) does not explain the window well "
            f"enough in absolute terms: normalized data term "
            f"{best.normalized_data_term():.4f} > "
            f"{cfg.max_normalized_data_term:.4f}")

    if len(ranked) >= 2:
        runner = ranked[1]
        a, b = key(best), key(runner)
        margin = (b - a) / max(abs(b), EPS)
        margin_ok = margin >= cfg.min_relative_margin
        if not margin_ok:
            reasons.append(
                f"{best.name} and {runner.name} are not distinguishable on the "
                f"same criterion (relative margin {margin:.4f} < "
                f"{cfg.min_relative_margin:.4f})")
    else:
        margin, margin_ok = float("inf"), True

    selected = best.name if (quality_ok and margin_ok) else UNRESOLVED
    return {"selected": selected, "reason": reasons,
            "relative_margin": margin,
            "criterion": "bic" if cfg.use_bic else "objective",
            "best": best.to_dict(),
            "candidates": [f.to_dict() for f in ranked],
            "comparable": True,
            "note": ("both candidates are scored by the same robust whitened "
                     "RGB-D window objective (section 11.4); no model-specific "
                     "score such as 'rotation share' or 'pitch share' is used")}


# ---------------------------------------------------------------------------
# the window fitter
# ---------------------------------------------------------------------------
@torch.no_grad()
def fit_window(model_name: str, transport_fn, geometry0, frames, energy_fn,
               *, q_init=None, max_outer: int = 8, refine_geometry=None,
               num_parameters: int = 0, num_residuals: int = 0,
               time_weight: float = 0.0, tol: float = 1e-10) -> WindowFit:
    """Fit one candidate to the whole window (sections 11.1 / 11.2).

    Alternates between the q trajectory (one 1-D search per frame on the true
    objective) and, if ``refine_geometry`` is supplied, the geometry itself --
    which is how section 11.2's "the axis location must be optimized jointly
    with direction and q trajectory" is satisfied rather than fixing the axis
    from the initializer.
    """
    geom = geometry0
    q = list(q_init) if q_init is not None else [0.0] * len(frames)
    phi = 0.5 * (5 ** 0.5 - 1)
    best = float("inf")
    iterations = 0

    for outer in range(int(max_outer)):
        # --- q trajectory, with the geometry fixed ---------------------
        for t, frame in enumerate(frames):
            lo, hi = q[t] - 0.5, q[t] + 0.5

            def e_at(x, _f=frame, _g=geom):
                return float(energy_fn(_f, *transport_fn(_g, float(x))))

            a, b = lo, hi
            cc, dd = b - phi * (b - a), a + phi * (b - a)
            fc, fd = e_at(cc), e_at(dd)
            for _ in range(40):
                if abs(b - a) <= tol:
                    break
                if fc < fd:
                    b, dd, fd = dd, cc, fc
                    cc = b - phi * (b - a)
                    fc = e_at(cc)
                else:
                    a, cc, fc = cc, dd, fd
                    dd = a + phi * (b - a)
                    fd = e_at(dd)
            q[t] = 0.5 * (a + b)

        # --- geometry, with the q trajectory fixed ---------------------
        if refine_geometry is not None:
            geom = refine_geometry(geom, q, frames, energy_fn, transport_fn)

        total = sum(float(energy_fn(f, *transport_fn(geom, q[t])))
                    for t, f in enumerate(frames))
        iterations = outer + 1
        if abs(best - total) <= 1e-10 * max(abs(best), 1.0):
            best = total
            break
        best = total

    time_term = 0.0
    if time_weight > 0.0 and len(q) >= 3:
        # J_time: a smoothness prior on the joint trajectory, identical for
        # both candidates so it cannot bias the comparison.
        acc = [q[i + 1] - 2 * q[i] + q[i - 1] for i in range(1, len(q) - 1)]
        time_term = float(time_weight * sum(a * a for a in acc))

    return WindowFit(name=model_name, data_term=best, time_term=time_term,
                     num_parameters=num_parameters, num_residuals=num_residuals,
                     q=[float(x) for x in q],
                     geometry=_geometry_dict(geom), iterations=iterations)


def _geometry_dict(geom) -> dict:
    if isinstance(geom, PluckerLine):
        return geom.to_dict()
    if torch.is_tensor(geom):
        return {"axis": geom.detach().cpu().tolist()}
    if isinstance(geom, dict):
        return {k: (v.detach().cpu().tolist() if torch.is_tensor(v) else v)
                for k, v in geom.items()}
    return {"value": str(geom)}


@torch.no_grad()
def refine_line_geometry(line: PluckerLine, q, frames, energy_fn, transport_fn,
                         *, step_dir: float = 0.02, step_mom: float = 0.02,
                         rounds: int = 2) -> PluckerLine:
    """Section 11.2: optimize the axis LOCATION jointly, not just its direction.

    A coordinate search over the 4 real degrees of freedom of a line (2 for the
    direction on S^2, 2 for the moment in the plane orthogonal to it), always
    re-projecting onto the e.m = 0 gauge so the iterate stays a line.
    """
    def total(l_):
        return sum(float(energy_fn(f, *transport_fn(l_, q[t])))
                   for t, f in enumerate(frames))

    best_line, best = line, total(line)
    e = line.direction
    # a basis of the plane orthogonal to e, for both direction and moment steps
    tmp = torch.tensor([1.0, 0.0, 0.0], device=e.device, dtype=e.dtype)
    if abs(float(e @ tmp)) > 0.9:
        tmp = torch.tensor([0.0, 1.0, 0.0], device=e.device, dtype=e.dtype)
    u1 = torch.linalg.cross(e, tmp)
    u1 = u1 / u1.norm().clamp_min(EPS)
    u2 = torch.linalg.cross(e, u1)

    sd, sm = step_dir, step_mom
    for _ in range(int(rounds)):
        improved = False
        for basis, scale, is_dir in ((u1, sd, True), (u2, sd, True),
                                     (u1, sm, False), (u2, sm, False)):
            for sign in (+1.0, -1.0):
                if is_dir:
                    cand = PluckerLine(
                        direction=best_line.direction + sign * scale * basis,
                        moment=best_line.moment)
                else:
                    cand = PluckerLine(
                        direction=best_line.direction,
                        moment=best_line.moment + sign * scale * basis)
                val = total(cand)
                if val < best:
                    best, best_line, improved = val, cand, True
        if not improved:
            sd, sm = sd * 0.5, sm * 0.5
    return best_line
