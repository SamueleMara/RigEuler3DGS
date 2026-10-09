"""Sections 19-22: the joint type decision, from geometry alone.

CODEX_IMPLEMENT_GEOMETRIC_RIGID_MOTION_JOINT_DISCOVERY.md sections 14, 19-22, 28.

Section 14 fixes the ordering: a joint is inferred only AFTER the K rigid
trajectories {T_{k,t}} exist. Section 1 removes the previous classifier from
production -- not BIC, not a profile likelihood, not a curvature significance
test, not an axis-distance gate. The decision here reads nothing but the
recovered transforms:

    M_k = argmin_{M in {P,R}} Ebar_M^{(k)}                       (section 19)

on section 20's normalized error, with both fits converged (section 28) and
measured against the SAME trajectory observations.

`UNRESOLVED_GEOMETRY`
---------------------------------------------------------------------------
Section 19 requires this answer when the two normalized errors are
"numerically indistinguishable", and forbids inventing a new heuristic to break
the tie. The tolerance is therefore NOT a model-selection margin chosen by
anyone. It is the precision to which the two numbers are known, and that has
two parts:

  * a numerical floor, `tau_indistinguishable`, relative to the larger error;
  * the NOISE FLOOR of the rigid transforms being fitted.

The second is the one that matters on real data and it is measured, not
chosen. Both fits consume {T_{k,t}}, which were themselves recovered to a
finite accuracy -- the part's rigid residual. Expressed on section 20's scale
that accuracy is (rigid_rmse / L)^2, so a difference in E^geom smaller than it
is a difference between two numbers inside their own error bars. Comparing
them would be reading noise.

A short revolute arc is the case this exists for: over a small sweep a circle
and its tangent line agree to within the accuracy of the recovered transforms,
and the honest answer is that the geometry does not decide. With exactly
noiseless input the noise floor is ~1e-32 and the test correctly keeps
resolving arcs down to 1e-4 rad; it is real measurement error, not the arc
alone, that makes a joint unresolvable.
"""

from __future__ import annotations

import math

import torch

from estimator.manifold import project_to_so3, so3_log

from .joint_geometry import (fit_rigid_trajectory, normalized_geometric_error,
                             pairwise_distance_drift, rms_radius)
from .prismatic_fit import fit_prismatic
from .revolute_fit import fit_revolute

PRISMATIC = "prismatic"
REVOLUTE = "revolute"
UNRESOLVED_GEOMETRY = "UNRESOLVED_GEOMETRY"
UNRESOLVED_OPTIMIZATION = "UNRESOLVED_OPTIMIZATION"

EPS = 1e-12


def relative_trajectory(R: torch.Tensor, t: torch.Tensor):
    """Refer every transform to frame 0: T_t <- T_t o T_0^{-1}.

    Section 22's gauge is q_0 = 0, and both models must see the SAME
    observations (section 19). Doing this once here, rather than inside each
    fit, is what guarantees the two fits cannot be handed different data.
    """
    R = project_to_so3(R.double())
    t = t.double()
    R0t = R[0].T
    R_rel = torch.einsum("tij,jk->tik", R, R0t)
    t_rel = t - torch.einsum("tij,j->ti", R_rel, t[0])
    return project_to_so3(R_rel), t_rel


def decide_joint_type(R_obs: torch.Tensor, t_obs: torch.Tensor, length: float,
                      *, weights: torch.Tensor | None = None,
                      tau_indistinguishable: float = 1e-9,
                      noise_floor: float = 0.0,
                      max_iters: int = 200) -> dict:
    """Section 19's geometric P/R decision on one part's rigid trajectory.

    `length` is section 20's L_k, the part's RMS radius. `noise_floor` is the
    precision of the supplied transforms on section 20's scale, normally
    (rigid_rmse / L)^2 -- see the module docstring.
    """
    R_rel, t_rel = relative_trajectory(R_obs, t_obs)

    fp = fit_prismatic(R_rel, t_rel, length, weights=weights)
    fr = fit_revolute(R_rel, t_rel, length, weights=weights,
                      max_iters=max_iters)

    # Section 26: eligibility for the comparison is PHYSICAL, not parametric.
    # A revolute candidate whose induced trajectory has settled is eligible
    # even when its axis coordinates keep drifting, because near the
    # translation limit (r -> inf, q -> 0, r q -> d) those coordinates are not
    # identifiable and waiting for them to settle would mean waiting forever.
    # Sections 24/25: that state is DEGENERATE_FLAT_DIRECTION -- eligible to be
    # compared, but never by itself a reason to call the joint revolute.
    p_ok = bool(fp.get("eligible", fp["converged"]))
    r_ok = bool(fr.get("eligible", fr["converged"]))
    both_converged = p_ok and r_ok
    e_p, e_r = fp["error"], fr["error"]
    scale = max(abs(e_p), abs(e_r), 1.0)
    gap = abs(e_p - e_r) / scale
    gap_abs = abs(e_p - e_r)
    floor = max(tau_indistinguishable * scale, float(noise_floor))
    indistinguishable = gap_abs <= floor

    if not both_converged:
        # Section 28: "Do not compare unconverged P/R fits."
        selected, status = None, UNRESOLVED_OPTIMIZATION
    elif indistinguishable:
        selected, status = None, UNRESOLVED_GEOMETRY
    else:
        selected = PRISMATIC if e_p < e_r else REVOLUTE
        status = "resolved"

    fit = fp if selected == PRISMATIC else (fr if selected == REVOLUTE else None)
    out = {
        "selected": selected,
        "status": status,
        "prismatic": _public(fp),
        "revolute": _public(fr),
        "normalized_error": {"prismatic": e_p, "revolute": e_r,
                             "relative_gap": gap, "absolute_gap": gap_abs,
                             "indistinguishable_floor": floor,
                             "noise_floor": float(noise_floor),
                             "tau_indistinguishable": tau_indistinguishable},
        "both_converged": both_converged,
        "eligibility": {
            "prismatic": {"eligible": p_ok,
                          "state": (fp.get("convergence") or {}).get("status")},
            "revolute": {"eligible": r_ok,
                         "state": (fr.get("convergence") or {}).get("status"),
                         "trajectory_converged": fr.get("physical_motion_converged"),
                         "joint_axis_identifiable": fr.get("joint_parameters_identifiable")},
            "rule": ("section 26: CONVERGED or DEGENERATE_FLAT_DIRECTION with "
                     "physical_motion_converged are eligible; "
                     "MAX_ITERS_UNSTABLE and NUMERICAL_FAILURE are not")},
        "q": (fit["q"] if fit is not None else None),
        "axis": ([float(x) for x in fit["axis"]] if fit is not None else None),
        "axis_point": ([float(x) for x in fr["axis_point"]]
                       if selected == REVOLUTE else None),
        "decision_inputs": ("only the recovered rigid transforms {T_k,t} and "
                            "section 20's normalized geometric error; no BIC, "
                            "no likelihood, no curvature test, no axis-distance "
                            "gate (sections 1 and 19)"),
    }
    return out


def _public(f: dict) -> dict:
    """The JSON-able view of a fit (section 30's required joint outputs)."""
    d = {"kind": f["kind"], "error": f["error"],
         "error_terms": {k: v for k, v in f["error_terms"].items()
                         if k != "per_frame"},
         "q": f["q"], "converged": f["converged"],
         "eligible": f.get("eligible", f["converged"]),
         "physical_motion_converged": f.get("physical_motion_converged"),
         "joint_parameters_identifiable": f.get("joint_parameters_identifiable"),
         "convergence": {k: v for k, v in f["convergence"].items()
                         if k != "history"},
         "num_parameters": f["num_parameters"],
         "axis": [float(x) for x in f["axis"]],
         "diagnostics": f["diagnostics"]}
    if "axis_point" in f:
        d["axis_point"] = [float(x) for x in f["axis_point"]]
    if "axis_iterates" in f:
        # Sections 26/27 of the debug addendum: per-iteration axis history.
        d["axis_iterates"] = f["axis_iterates"]
    # Section 25/26: the per-frame residuals, written whether or not this
    # model was selected.
    d["per_frame_error"] = f["error_terms"].get("per_frame")
    return d


def discover_joint_from_points(x0: torch.Tensor, frames: list, *,
                               weights: torch.Tensor | None = None,
                               tau_indistinguishable: float = 1e-9,
                               max_iters: int = 200) -> dict:
    """The whole geometric chain for one part, from point tracks.

        points -> {T_{k,t}} (section 13) -> P/R decision (section 19)

    `frames` are the part's point positions at each frame, in the order of
    `x0`. This is the entry point the renderer stage feeds once it has decided
    WHO moves and HOW MUCH (section 48's ordering).
    """
    from .gaussian_transport_metric import canonical_moments, gain_attribution

    traj = fit_rigid_trajectory(x0, frames, weights)
    L = rms_radius(x0, weights)
    # Section 19's "numerically indistinguishable", measured: the transforms
    # handed to the two fits are themselves only accurate to the part's rigid
    # residual, so on section 20's scale they are known to (rmse / L)^2.
    rmse = (max(traj["rigid_rmse"]) if traj["rigid_rmse"] else 0.0)
    noise_floor = (rmse / max(L, EPS)) ** 2

    # Section 3: the decision runs on the continuous chart, scored by the
    # coordinate-invariant transport error of section 2. The legacy path below
    # could not select prismatic on exactly-prismatic data: prismatic is a
    # boundary point there (||a|| -> infinity), so its revolute solver stalled
    # far from the nested solution and returned E_R > E_P, which is impossible
    # when P is contained in R. In the chart prismatic is eta = 0, an ordinary
    # interior point, and the revolute fit reaches it.
    moments = canonical_moments(x0, weights)
    cont = decide_joint_type_continuous(
        traj["R"], traj["t"], moments["c"], moments["L"],
        epsilon_geom=noise_floor, max_iters=max_iters, moments=moments)
    out = dict(cont)
    out["normalized_error"] = {
        "prismatic": cont["E_P"], "revolute": cont["E_R"],
        "absolute_gap": cont["curvature_gain"],
        "relative_gap": cont["curvature_gain"],
        "noise_floor": noise_floor,
        "indistinguishable_floor": cont["floor"],
        "tau_indistinguishable": tau_indistinguishable}
    out["gain_attribution"] = cont.get("gain_attribution")
    # The superseded comparison is kept as a recorded diagnostic (section
    # 2.5), never as a selector.
    out["legacy_decision"] = decide_joint_type(
        traj["R"], traj["t"], L, weights=None,
        tau_indistinguishable=tau_indistinguishable,
        noise_floor=noise_floor, max_iters=max_iters)
    out["rigid"] = {
        "length_scale": L,
        "rigid_rmse": traj["rigid_rmse"],
        "max_rigid_rmse": max(traj["rigid_rmse"]) if traj["rigid_rmse"] else 0.0,
        "normalized_rigid_rmse": [r / max(L, EPS) for r in traj["rigid_rmse"]],
        "pairwise_distance_drift": pairwise_distance_drift(x0, frames),
    }
    out["transforms"] = {"R": traj["R"], "t": traj["t"]}
    return out


# ---------------------------------------------------------------------------
# sections 17, 18: the nesting-aware P/R decision
# ---------------------------------------------------------------------------
UNRESOLVED_IDENTIFIABILITY = "UNRESOLVED_IDENTIFIABILITY"


def _gauge_dot(fr) -> float:
    """e . a for the reported axis point, which the gauge pins to zero."""
    fa = fr.get("finite_axis")
    if not fa:
        return 0.0
    e = torch.tensor(fa["axis"], dtype=torch.float64)
    a = torch.tensor(fa["axis_point_origin"], dtype=torch.float64)
    return float(e @ a)


def curvature_observability(c, d, s, R_obs, p_obs, L, *, eps: float = 1e-6,
                            moments=None):
    """Section 18: can curvature be seen at all, from geometry alone?

    At the prismatic solution, profile the curvature Jacobian against the
    nuisance directions (d and the trajectory s):

        J_curve^profile = (I - J_nu J_nu^+) J_eta

    and report its singular values in the two-dimensional curvature tangent
    plane. A rank-deficient profiled Jacobian means curvature was not
    observable -- which section 18 says is a reason for UNRESOLVED_GEOMETRY,
    never evidence for prismatic.

    No likelihood, no chi-square, no BIC, no ground truth: these are
    derivatives of the geometric residual.
    """
    from .continuous_joint_geometry import (_geometric_residual,
                                            continuous_joint_transform)
    from .gaussian_transport_metric import (gaussian_transport_residual,
                                            psd_sqrt)
    _Sh = None if moments is None else psd_sqrt(moments["S"])
    c = c.reshape(3).double()
    d = d.reshape(3).double()
    s = s.reshape(-1).double()
    # curvature tangent plane: the two directions perpendicular to d
    tmp = torch.tensor([0.0, 0.0, 1.0], dtype=torch.float64, device=c.device)
    if abs(float(d @ tmp)) > 0.9:
        tmp = torch.tensor([0.0, 1.0, 0.0], dtype=torch.float64, device=c.device)
    u1 = torch.linalg.cross(d, tmp)
    u1 = u1 / u1.norm()
    u2 = torch.linalg.cross(d, u1)

    def res(eta_vec, d_vec, s_vec):
        R, p = continuous_joint_transform(c, d_vec, eta_vec, s_vec)
        if moments is None:
            return _geometric_residual(R_obs, p_obs, R, p, L)
        # Observability must be measured on the SAME objective the decision
        # is made with, or it answers a different question than it is asked.
        return gaussian_transport_residual(R_obs, p_obs, R, p, moments["c"],
                                           moments["S"], moments["L"],
                                           S_half=_Sh)

    zero = torch.zeros(3, dtype=torch.float64, device=c.device)
    base = res(zero, d, s)
    cols_eta = []
    for u in (u1, u2):
        cols_eta.append((res(eps * u, d, s) - res(-eps * u, d, s)) / (2 * eps))
    J_eta = torch.stack(cols_eta, dim=1)                       # [m, 2]

    cols_nu = []
    for k in range(3):                                          # d directions
        e = torch.zeros(3, dtype=torch.float64, device=c.device)
        e[k] = 1.0
        dp = d + eps * e
        dm = d - eps * e
        cols_nu.append((res(zero, dp / dp.norm(), s)
                        - res(zero, dm / dm.norm(), s)) / (2 * eps))
    for k in range(len(s)):                                     # trajectory
        sp = s.clone(); sp[k] += eps
        sm = s.clone(); sm[k] -= eps
        cols_nu.append((res(zero, d, sp) - res(zero, d, sm)) / (2 * eps))
    J_nu = torch.stack(cols_nu, dim=1)                          # [m, 3+T]

    # profile out the nuisance directions
    P = J_nu @ torch.linalg.pinv(J_nu)
    J_prof = J_eta - P @ J_eta
    sv = torch.linalg.svdvals(J_prof)
    smax = float(sv.max()) if sv.numel() else 0.0
    smin = float(sv.min()) if sv.numel() else 0.0
    # scale-aware numerical rank, from matrix precision (section 18)
    tol = max(J_prof.shape) * torch.finfo(torch.float64).eps * max(smax, 1e-300)
    rank = int((sv > tol).sum())
    return {"singular_values": [float(x) for x in sv],
            "sigma_max": smax, "sigma_min": smin,
            "condition": (smax / smin) if smin > 0 else float("inf"),
            "effective_rank": rank, "rank_tolerance": float(tol),
            "observable": rank == 2,
            "base_residual_norm": float(base.norm()),
            "note": ("section 18: profiled curvature Jacobian singular values "
                     "in the 2-D curvature tangent plane; rank deficiency "
                     "means curvature was not observable, which is not "
                     "evidence for prismatic")}


@torch.no_grad()
def decide_joint_type_continuous(R_obs, t_obs, c, L, *, epsilon_geom: float,
                                 epsilon_opt: float = 1e-12,
                                 max_iters: int = 120, moments=None,
                                 verbose: bool = False) -> dict:
    """Section 17: decide P vs R respecting that P is nested inside R.

    P is eta = 0 INSIDE the extended model, so at exact optimisation
    E_* <= E_P always. A raw argmin(E_P, E_R) is therefore not a basis for
    choosing P -- it only ever reported P because R had been stopped early.
    The quantity that carries information is the finite-curvature GAIN

        Delta_curve = E_P - E_best,R  >= 0,

    judged against a geometric resolution floor applied unchanged to both.
    """
    from .continuous_joint_geometry import fit_continuous_joint
    R_rel, t_rel = relative_trajectory(R_obs, t_obs)

    fp = fit_continuous_joint(c, R_rel, t_rel, L, max_iters=max_iters,
                              fix_eta_zero=True, moments=moments,
                              verbose=verbose)
    fr = fit_continuous_joint(c, R_rel, t_rel, L, max_iters=max_iters,
                              fix_eta_zero=False, moments=moments,
                              verbose=verbose)
    obs = curvature_observability(c, fp["d"], fp["s"], R_rel, t_rel, L,
                                  moments=moments)

    # A prismatic joint moves the part's centre along ONE direction, so the
    # matrix of centre displacements has rank 1. The ratio of its leading
    # singular value to the total says how nearly that holds, independently of
    # how well either model fitted -- a direct check on the data rather than
    # on the optimiser.
    c3 = c.reshape(3).double()
    disp = (torch.einsum("tij,j->ti", R_rel.double(), c3)
            + t_rel.double() - c3)
    sv = torch.linalg.svdvals(disp)
    sv2 = float((sv ** 2).sum())
    rank1 = (float(sv[0] ** 2) / sv2) if sv2 > 0 else 1.0

    E_P, E_R = fp["energy"], fr["energy"]
    gain = E_P - E_R
    floor = float(epsilon_geom) + float(epsilon_opt)
    eligible = fp["eligible"] and fr["eligible"]

    # Section 18, made quantitative. The profiled curvature Jacobian says how
    # much energy a curvature delta_kappa buys: about (sigma delta_kappa)^2.
    # Setting that equal to the floor gives the SMALLEST CURVATURE THIS
    # MEASUREMENT COULD HAVE SEEN,
    #
    #     kappa_min = sqrt(floor) / sigma_min.
    #
    # This is what separates "no curvature is present" from "no curvature
    # could have been detected". A part has its own scale -- an axis one
    # part-radius away is kappa = 1/L, about as curved as an articulated part
    # ever gets -- so when kappa_min exceeds 1/L the data could not have
    # revealed even that. Reporting prismatic there would assert eta = 0 from
    # a measurement that could not have refuted any eta at all, which section
    # 8 forbids: "if the evidence cannot resolve P/R, return
    # UNRESOLVED_GEOMETRY, not a fabricated positive P or R".
    sigma_min = float(obs.get("sigma_min", 0.0) or 0.0)
    kappa_min = (math.sqrt(max(floor, 0.0)) / sigma_min
                 if sigma_min > 0 else float("inf"))
    kappa_reference = 1.0 / max(float(L), EPS)
    curvature_resolvable = kappa_min <= kappa_reference
    obs = {**obs, "kappa_min_detectable": kappa_min,
           "kappa_reference_one_part_radius": kappa_reference,
           "curvature_resolvable": bool(curvature_resolvable),
           "resolvability_note": (
               "kappa_min = sqrt(floor)/sigma_min is the smallest curvature "
               "this measurement could have distinguished from zero; it is "
               "compared with 1/L, the curvature of an axis one part-radius "
               "away")}

    if not eligible:
        selected, status = None, UNRESOLVED_OPTIMIZATION
    elif not obs["observable"]:
        # Section 18: curvature could not be seen; that is ambiguity, not P.
        selected, status = None, UNRESOLVED_GEOMETRY
    elif gain <= floor and not curvature_resolvable:
        # No curvature was found, but none would have been found either way.
        selected, status = None, UNRESOLVED_GEOMETRY
    elif gain > floor and fr["kappa"] > 0.0:
        selected, status = REVOLUTE, "resolved"
    elif gain <= floor:
        # Section 17: eta = 0 is a verified stationary solution and no
        # finite-curvature solution improves on it beyond the floor.
        selected, status = PRISMATIC, "resolved"
    else:
        selected, status = None, UNRESOLVED_IDENTIFIABILITY

    out = {"selected": selected, "status": status,
           "E_P": E_P, "E_R": E_R,
           "curvature_gain": gain, "floor": floor,
           "epsilon_geom": float(epsilon_geom),
           "epsilon_opt": float(epsilon_opt),
           "kappa": fr["kappa"],
           "prismatic": {**{k: v for k, v in fp.items()
                            if k not in ("R", "p", "d", "eta", "s",
                                         "history")},
                         "diagnostics": {
                             "translation_rank1_ratio": rank1,
                             "centre_displacement_singular_values":
                                 sv.tolist(),
                             "meaning": "fraction of the centre-displacement "
                                        "energy along a single direction; 1 "
                                        "means the centre travelled a "
                                        "straight line"}},
           "revolute": {**{k: v for k, v in fr.items()
                           if k not in ("R", "p", "d", "eta", "s",
                                        "history")},
                        "diagnostics": {
                            "gauge_e_dot_a": _gauge_dot(fr),
                            "meaning": "e . a for the reported axis point; "
                                       "zero by the closest-point-to-origin "
                                       "gauge"}},
           "curvature_observability": obs,
           "decision_inputs": ("only the recovered rigid transforms {T_k,t} "
                               "and the coordinate-invariant Gaussian "
                               "transport error of section 2; no BIC, no "
                               "likelihood, no curvature test, no "
                               "axis-distance gate (sections 1 and 19)"),
           "decision_rule": (
               "section 17: P is eta = 0 nested in R, so E_* <= E_P always and "
               "argmin(E_P, E_R) cannot select P. The decision is the "
               "finite-curvature gain E_P - E_R against epsilon_geom + "
               "epsilon_opt, with curvature observability (section 18) "
               "required before either label.")}
    # Every caller gets the same keys whatever the verdict, so an unresolved
    # part is an explicit None rather than a missing field that raises.
    out.setdefault("axis", None)
    out.setdefault("axis_point", None)
    out.setdefault("axis_point_centroid", None)
    out.setdefault("q", None)
    if selected == REVOLUTE and fr.get("finite_axis"):
        out["axis"] = fr["finite_axis"]["axis"]
        # Section 8 of the chart spec: an axis point is only defined up to
        # sliding along the axis, so one gauge has to be named. The rest of
        # the codebase uses the point CLOSEST TO THE WORLD ORIGIN (e . a = 0);
        # `axis_point_centroid` is the same line anchored at the part centre
        # and stays available under `finite_axis` for display.
        out["axis_point"] = fr["finite_axis"]["axis_point_origin"]
        out["axis_point_centroid"] = fr["finite_axis"]["axis_point_centroid"]
        out["q"] = fr["finite_axis"]["q"]
    elif selected == PRISMATIC:
        out["axis"] = fp["d"].tolist()
        out["q"] = fp["s"].tolist()
    return out
