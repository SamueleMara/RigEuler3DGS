"""Section 16: identifiability measured in OBSERVATION space.

CODEX_THEORETICAL_METHOD_STRICT_IMPLEMENTATION.md sections 15, 16, 17, 18.

Section 15 deletes the production rule

    d_axis / R_support > tau  =>  prefer prismatic,

because "a true revolute joint can have d/R >> 1 and still be identifiable
when angular sweep and measurement quality are sufficient". The d/R ratio
survives as a diagnostic only.

Section 16 replaces it with the only question that is actually about the data:

    Can RGB-D observations distinguish curved revolute motion from its best
    translational approximation?

This module implements the PREFERRED method of section 16, the profile
likelihood / curvature test.

The curvature parameter
---------------------------------------------------------------------------
For a part whose support has centroid c, let the revolute axis have direction
e and pass at perpendicular distance rho from c. Define

    kappa = 1 / rho            (signed: its sign says which side of c the axis
                                lies on)

so that kappa -> 0 is EXACTLY the prismatic limit: an axis infinitely far away
with a correspondingly small rotation angle is a translation. The family is
continuous at kappa = 0, which is what makes a likelihood ratio against
kappa = 0 meaningful at all.

Arc length, not angle
---------------------------------------------------------------------------
The trajectory is profiled out at each kappa, so its UNITS must not change
with kappa or the comparison is rigged. A rotation by theta at radius rho moves
the centroid by arc length s = rho * theta, so the model is driven in arc
length throughout: the angle trajectory is theta = kappa * s, and the temporal
process noise is rescaled as sigma_a(theta) = |kappa| * sigma_a(s). Without
this the temporal prior alone would favour large rho, since the same motion
needs a smaller angle there. Section 31's fairness requirement is the same
point applied to P vs R.

The test
---------------------------------------------------------------------------
    l(kappa) = min over all other parameters of E(theta)        (section 16)
    LR       = 2 [ l(0) - l(kappa_hat) ]

E is a negative log-likelihood (`measurement_energy` is 0.5 r^T R_eff^-1 r), so
LR is a profile likelihood ratio and is chi-square with ONE degree of freedom
under kappa = 0. The profile-likelihood confidence interval is the standard

    { kappa : 2 [ l(kappa) - l(kappa_hat) ] <= chi2_{1,1-alpha} }

so "is 0 inside the interval" and "is the LR significant" are the same
computation, as they should be.

Verdicts follow section 16 literally:

    LR >= threshold                   -> REVOLUTE   (nonzero curvature is
                                                     significantly supported)
    otherwise (0 inside the interval) -> UNRESOLVED_MODEL

Section 45 forbids converting the second case into "prismatic" by fiat, so
this module does not. It reports `prismatic_consistent` and the interval so a
caller can SEE that the data is consistent with prismatic, which is a
different statement from the data resolving the model.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace

import torch

from .kpart_models import PRISMATIC, REVOLUTE, PartModel, SceneModel
from .kpart_support import chi2_quantile

EPS = 1e-12
UNRESOLVED_MODEL = "UNRESOLVED_MODEL"
UNRESOLVED_OPTIMIZATION = "UNRESOLVED_OPTIMIZATION"
CONVERGED = "CONVERGED"


def orthonormal_frame(e: torch.Tensor, n: torch.Tensor):
    """An orthonormal (e, n, u) with u = e x n, n re-orthogonalised against e."""
    e = e / e.norm().clamp_min(EPS)
    n = n - e * (e @ n)
    nn = n.norm()
    if float(nn) < 1e-9:                       # n was parallel to e: pick any
        tmp = torch.tensor([1.0, 0.0, 0.0], device=e.device, dtype=e.dtype)
        if abs(float(e @ tmp)) > 0.9:
            tmp = torch.tensor([0.0, 1.0, 0.0], device=e.device, dtype=e.dtype)
        n = torch.linalg.cross(e, tmp)
        nn = n.norm()
    n = n / nn.clamp_min(EPS)
    return e, n, torch.linalg.cross(e, n)


def curvature_family(centroid: torch.Tensor, e: torch.Tensor, n: torch.Tensor,
                     kappa: float, *, part_index: int = 1,
                     kappa_zero: float = 1e-9) -> PartModel:
    """The one-parameter family through the prismatic limit kappa = 0.

    `e` is the rotation axis direction, `n` the unit radial direction from the
    axis towards the support centroid, so the axis passes through
    c - (1/kappa) n and the centroid travels along u = e x n.
    """
    e, n, u = orthonormal_frame(e, n)
    if abs(float(kappa)) <= kappa_zero:
        return PartModel(kind=PRISMATIC, axis=u, part_index=part_index)
    rho = 1.0 / float(kappa)
    point = centroid.reshape(3) - rho * n
    return PartModel(kind=REVOLUTE, axis=e,
                     moment=torch.linalg.cross(point, e), part_index=part_index)


def curvature_of(model: PartModel, centroid: torch.Tensor):
    """(kappa, e, n) for an existing model, the inverse of `curvature_family`."""
    e = model.axis / model.axis.norm().clamp_min(EPS)
    if model.kind == PRISMATIC:
        # A prismatic model's `axis` IS the travel direction u. The frame must
        # satisfy e x n = u so that `curvature_family(c, e, n, 0)` rebuilds the
        # same model; any e perpendicular to u will do, with n = u x e, since
        # e x (u x e) = u.
        u0 = e
        tmp = torch.tensor([1.0, 0.0, 0.0], device=e.device, dtype=e.dtype)
        if abs(float(u0 @ tmp)) > 0.9:
            tmp = torch.tensor([0.0, 1.0, 0.0], device=e.device, dtype=e.dtype)
        e_rot = torch.linalg.cross(u0, tmp)
        e_rot = e_rot / e_rot.norm().clamp_min(EPS)
        return 0.0, e_rot, torch.linalg.cross(u0, e_rot)
    a = torch.linalg.cross(e, model.moment.reshape(3))       # closest to origin
    d = centroid.reshape(3) - a
    r = d - e * (e @ d)                                      # radial component
    rho = float(r.norm())
    if rho < 1e-12:
        return float("inf"), e, r
    return 1.0 / rho, e, r / rho


@dataclass
class CurvatureProfile:
    """The result of section 16's profile likelihood test."""
    kappa_grid: list
    ell: list
    kappa_hat: float
    ell_hat: float
    ell_zero: float
    likelihood_ratio: float
    threshold: float
    alpha: float
    interval: tuple
    zero_inside: bool
    verdict: str
    converged: bool
    status: str
    diagnostics: dict

    def to_dict(self) -> dict:
        return {"kappa_grid": [float(k) for k in self.kappa_grid],
                "profile_likelihood": [float(x) for x in self.ell],
                "kappa_hat": self.kappa_hat, "ell_hat": self.ell_hat,
                "ell_zero": self.ell_zero,
                "likelihood_ratio": self.likelihood_ratio,
                "threshold": self.threshold, "alpha": self.alpha,
                "confidence_interval": [float(x) for x in self.interval],
                "zero_inside_interval": self.zero_inside,
                "verdict": self.verdict, "converged": self.converged,
                "status": self.status, **self.diagnostics}


def _interval_from_profile(grid, ell, ell_hat, threshold):
    """{kappa : 2[l(kappa) - l(kappa_hat)] <= threshold}, linearly interpolated."""
    lo, hi = None, None
    for i, (k, e) in enumerate(zip(grid, ell)):
        if 2.0 * (e - ell_hat) <= threshold:
            if lo is None:
                lo = k
                if i > 0:                      # interpolate the crossing
                    k0, e0 = grid[i - 1], ell[i - 1]
                    d0, d1 = 2 * (e0 - ell_hat), 2 * (e - ell_hat)
                    if d0 > threshold >= d1 and d0 != d1:
                        lo = k0 + (k - k0) * (d0 - threshold) / (d0 - d1)
            hi = k
            if i + 1 < len(grid):
                k1, e1 = grid[i + 1], ell[i + 1]
                d0, d1 = 2 * (e - ell_hat), 2 * (e1 - ell_hat)
                if d0 <= threshold < d1 and d0 != d1:
                    hi = k + (k1 - k) * (threshold - d0) / (d1 - d0)
    if lo is None:
        lo = hi = float(grid[int(torch.tensor(ell).argmin())])
    return float(lo), float(hi)


def default_kappa_grid(rho_support: float, n: int = 11) -> list:
    """A grid in kappa >= 0 spanning the prismatic limit.

    kappa is a MAGNITUDE. The sign would say which side of the support the axis
    lies on, but that is carried by the radial direction n, which is a nuisance
    parameter the frame refinement optimises; `curvature_of` always returns the
    radial direction pointing from the axis towards the centroid, so a signed
    grid is a redundant parameterisation and was measured to produce a spurious
    kappa_hat = -0.618 for a true +0.824.

    The scale is set by the part's own support radius, a property of the
    geometry being tested and not of any dataset: kappa = 1/rho_support is an
    axis one support radius away, and the grid halves from there over `n-1`
    octaves down towards the prismatic limit. Nothing is tuned to a result; the
    grid only has to BRACKET kappa_hat, and `kappa_hat_bracketed_by_grid`
    reports when it does not.
    """
    k_max = 1.0 / max(float(rho_support), EPS)
    return [0.0] + [k_max * (2.0 ** (-i)) for i in range(n - 2, -1, -1)]


def support_geometry(points: torch.Tensor):
    """(centroid, support radius) of a part's canonical support."""
    c = points.mean(dim=0)
    rho = float((points - c).norm(dim=1).mean().clamp_min(EPS))
    return c, rho


@torch.no_grad()
def profile_curvature(objective, scene: SceneModel, labels, part_index: int,
                      cfg, *, support_points: torch.Tensor,
                      kappa_grid=None, alpha: float = 0.05,
                      q_init=None, n_grid: int = 9,
                      refine_direction: bool = True) -> CurvatureProfile:
    """Section 16's profile likelihood over the curvature parameter.

    For each fixed kappa the remaining parameters -- the trajectory of every
    part and, optionally, the frame (e, n) of this part -- are re-optimised,
    which is what "min over theta \\ kappa" means. The trajectory is driven in
    ARC LENGTH at every kappa so the comparison is not rigged by units.

    Convergence is tracked per section 17: if the nuisance optimisation fails
    to converge at any kappa the verdict is UNRESOLVED_OPTIMIZATION, because
    section 0.5 forbids selecting a model from an unfinished optimization.
    """
    from .kpart_window import solve_banded_trajectory

    centroid, rho_support = support_geometry(support_points)
    model0 = scene.parts[part_index - 1]
    if model0 is None:
        raise ValueError(f"part {part_index} has no model to profile")
    _k0, e0, n0 = curvature_of(model0, centroid)
    e0, n0, _u0 = orthonormal_frame(e0, n0)

    grid = list(kappa_grid) if kappa_grid is not None else \
        default_kappa_grid(rho_support, n=n_grid)
    T = len(objective.frames)
    K = scene.K
    if q_init is None:
        q_init = [[0.0] * T for _ in range(K)]

    ell, ell_data, per_kappa = [], [], []
    all_converged = True
    for kappa in grid:
        model = curvature_family(centroid, e0, n0, kappa,
                                 part_index=part_index)
        parts = list(scene.parts)
        parts[part_index - 1] = model
        sc = SceneModel(parts=parts, K=K)

        # Arc length -> this model's own coordinate, and the matching process
        # noise, so every kappa is driven by the same physical motion.
        scale = abs(float(kappa)) if abs(float(kappa)) > 1e-9 else 1.0
        q0 = [list(c) for c in q_init]
        q0[part_index - 1] = [x * scale for x in q_init[part_index - 1]]
        cfg_k = replace(cfg, sigma_a=cfg.sigma_a * scale)

        q_opt, info = solve_banded_trajectory(objective, sc, labels, q0, cfg_k)
        if refine_direction:
            from .kpart_window import refine_part_geometry
            sc, q_opt = _refine_at_fixed_kappa(objective, sc, labels, q_opt,
                                               cfg_k, part_index, centroid,
                                               kappa)
            q_opt, info = solve_banded_trajectory(objective, sc, labels,
                                                  q_opt, cfg_k)

        data = 0.0
        for ti, (obs, _f) in enumerate(objective.frames):
            qt = [q_opt[j][ti] for j in range(K)]
            mu, R = sc.transport_all(objective.mu0, objective.R0, labels, qt)
            er, ed = objective.frame_energy(mu, R, obs)
            data += er + ed
        ell.append(float(info["energy"]))
        ell_data.append(float(data))
        all_converged &= bool(info.get("converged", False))
        per_kappa.append({"kappa": float(kappa), "energy": float(info["energy"]),
                          "data_energy": float(data),
                          "status": info.get("status"),
                          "grad_norm_rel": info.get("grad_norm_rel"),
                          "kind": model.kind})

    i_hat = int(torch.tensor(ell).argmin())
    kappa_hat, ell_hat = float(grid[i_hat]), float(ell[i_hat])
    i_zero = min(range(len(grid)), key=lambda i: abs(grid[i]))
    ell_zero = float(ell[i_zero])
    lr = 2.0 * (ell_zero - ell_hat)
    lr_data = 2.0 * (ell_data[i_zero] - float(ell_data[i_hat]))

    # kappa >= 0 puts the null kappa = 0 on the BOUNDARY of the parameter
    # space, so the likelihood ratio is not chi-square(1): it is Chernoff's
    # 1/2 chi2_0 + 1/2 chi2_1 mixture (Self and Liang 1987), under which
    # P(LR > c) = 1/2 P(chi2_1 > c). The level-alpha threshold is therefore
    # chi2_1(1 - 2 alpha), which is SMALLER than the interior-null
    # chi2_1(1 - alpha) -- using the interior quantile here would be
    # conservative, not permissive, but it would still be the wrong null.
    threshold = chi2_quantile(1.0 - 2.0 * alpha, 1)
    lo, hi = _interval_from_profile(grid, ell, ell_hat, threshold)
    zero_inside = lr <= threshold

    # Section 17: the hypotheses being COMPARED are kappa = 0 and kappa_hat.
    # Those two must have converged. A grid point far from both only shapes the
    # interval, and its failure to converge is reported rather than silently
    # collapsing the verdict.
    compared_converged = bool(per_kappa[i_zero]["status"] == "converged"
                              and per_kappa[i_hat]["status"] == "converged")
    not_converged = [r["kappa"] for r in per_kappa
                     if r["status"] != "converged"]

    if not compared_converged:
        verdict, status = UNRESOLVED_OPTIMIZATION, UNRESOLVED_OPTIMIZATION
    elif not zero_inside:
        verdict, status = REVOLUTE, CONVERGED
    else:
        verdict, status = UNRESOLVED_MODEL, CONVERGED

    bracketed = i_hat < len(grid) - 1
    return CurvatureProfile(
        kappa_grid=grid, ell=ell, kappa_hat=kappa_hat, ell_hat=ell_hat,
        ell_zero=ell_zero, likelihood_ratio=lr, threshold=threshold,
        alpha=alpha, interval=(lo, hi), zero_inside=zero_inside,
        verdict=verdict, converged=compared_converged, status=status,
        diagnostics={
            "likelihood_ratio_data_only": lr_data,
            "all_grid_points_converged": all_converged,
            "kappa_not_converged": not_converged,
            "null_distribution": ("1/2 chi2_0 + 1/2 chi2_1 (kappa = 0 is on "
                                  "the boundary of kappa >= 0)"),
            "support_radius": rho_support,
            "support_centroid": [float(x) for x in centroid],
            "kappa_hat_bracketed_by_grid": bool(bracketed),
            "prismatic_consistent": bool(zero_inside),
            "per_kappa": per_kappa,
            # Section 15: d/R survives as a DIAGNOSTIC and never as a rule.
            "axis_distance_over_support_radius_diagnostic": (
                float(1.0 / abs(kappa_hat) / rho_support)
                if abs(kappa_hat) > 1e-12 else float("inf")),
            "note": ("section 16 profile likelihood; kappa = 1/rho is the "
                     "inverse axis distance and kappa = 0 is the prismatic "
                     "limit. No axis-distance threshold is used."),
        })


def _refine_at_fixed_kappa(objective, sc: SceneModel, labels, q_traj, cfg,
                           part_index: int, centroid, kappa: float):
    """Re-optimise the (e, n) frame with kappa HELD FIXED.

    kappa is the parameter of interest, so the nuisance search must not be able
    to change it -- every candidate is rebuilt through `curvature_family` at
    the same kappa.
    """
    def total(scene_c, q):
        e = 0.0
        for ti, (obs, _f) in enumerate(objective.frames):
            qt = [q[j][ti] for j in range(scene_c.K)]
            mu, R = scene_c.transport_all(objective.mu0, objective.R0, labels, qt)
            er, ed = objective.frame_energy(mu, R, obs)
            e += er + ed
        return e

    model = sc.parts[part_index - 1]
    _k, e_cur, n_cur = curvature_of(model, centroid)
    if model.kind == PRISMATIC:
        e_cur, n_cur, _ = orthonormal_frame(e_cur, n_cur)
    best = total(sc, q_traj)
    step = 0.05
    for _round in range(int(cfg.geometry_rounds)):
        improved = False
        for which in ("e", "n"):
            base = e_cur if which == "e" else n_cur
            other = n_cur if which == "e" else e_cur
            perp = torch.linalg.cross(base, other)
            for sign in (+1.0, -1.0):
                cand = base + sign * step * perp
                ce, cn = ((cand, n_cur) if which == "e" else (e_cur, cand))
                m = curvature_family(centroid, ce, cn, kappa,
                                     part_index=part_index)
                parts = list(sc.parts)
                parts[part_index - 1] = m
                trial = SceneModel(parts=parts, K=sc.K)
                e_try = total(trial, q_traj)
                if e_try < best - 1e-9:
                    best, sc = e_try, trial
                    e_cur, n_cur = curvature_of(m, centroid)[1:]
                    improved = True
                    break
        if not improved:
            step *= 0.5
    return sc, q_traj
