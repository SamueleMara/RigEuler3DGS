"""Section 16: the geometric revolute fit, with the axis LOCATION unknown.

CODEX_IMPLEMENT_GEOMETRIC_RIGID_MOTION_JOINT_DISCOVERY.md sections 16, 20, 22.

A revolute joint means there is ONE fixed line L = {a + lambda e} such that

    R_{k,t} = Exp(q_{k,t} [e]_x),      t_{k,t} = (I - R_{k,t}) a,

so a point moves as x_{i,t} = a + R(e, q_t)(x_i^0 - a). The fit is

    E_R = min_{e,a,q} sum_t [ w^R ||log(R(e,q_t)^T R_t)||^2
                            + w^p ||t_t - (I - R(e,q_t)) a||^2 ].

Section 16 is explicit that `a` is an unknown and that a = 0 must never be
assumed. The weights are taken from section 20, w^R = 1 and w^p = 1/L^2, so
E_R is exactly T times the normalized geometric error the section 19 decision
compares -- the fit minimises the quantity it is later judged by, which is what
"same normalization" requires.

Why `a` is solvable but only up to a gauge
---------------------------------------------------------------------------
For fixed (e, q) the translation term is LINEAR in a:

    min_a sum_t || t_t - M_t a ||^2,    M_t = I - R(e, q_t).

Every M_t annihilates e (a rotation does not move its own axis), so the normal
matrix is singular along e and the component of a parallel to e is genuinely
unobservable -- sliding the axis point along the axis describes the same line.
The minimum-norm solution therefore lands exactly in section 16's closest-point
gauge e^T a = 0. That is not a numerical convenience; it is the correct
statement that a line has four degrees of freedom, not six.

Section 23's numerical tolerances
---------------------------------------------------------------------------
tau_motion bounds the per-iteration change of the INDUCED motion, in units of
the part radius, and tau_error the relative change of the objective. Both are
set by the precision of the comparison they feed, which is section 26's
framing: a candidate is eligible when one more iteration cannot change the P/R
verdict. Concretely:

  * the smallest indistinguishability floor measured on real data is ~2.3e-06
    (storage), and every geometric decision compares E_P and E_R against such
    a floor;
  * tau_error = 1e-6 therefore means one iteration cannot move E by a
    decision-relevant amount;
  * E^geom is quadratic in displacement, so an induced-motion change of
    tau_motion = 1e-4 part-radii perturbs E^geom by roughly 1e-8, two orders
    below that floor.

A pure translation is the case that forces this reasoning. It is the r -> inf
limit of a revolute, so the fit approaches its asymptote without ever reaching
it: measured, the objective still falls 7.5e-08 per iteration at iteration 200
and the induced motion still moves 4.5e-05, neither decaying. Demanding they
reach zero would mark every translation-like motion MAX_ITERS_UNSTABLE forever
and, by section 26, keep it out of the comparison permanently -- which is the
bug this section exists to prevent.

A fit meeting these tolerances therefore cannot move either error by enough to
change the comparison. The earlier values (1e-7 and 1e-10) were unreachable
rather than strict: measured on a pure translation the induced-motion change
plateaus at 4.5e-05 while the objective, at 3.8e-04, changes by 7.6e-08 per
iteration, so both floors sat below the arithmetic the fit can deliver and
every run ended MAX_ITERS_UNSTABLE with no candidate ever eligible.

"""

from __future__ import annotations

import math

import torch

from estimator.manifold import project_to_so3, so3_exp, so3_log

from .joint_geometry import normalized_geometric_error, unwrap_angles

EPS = 1e-12



# ---------------------------------------------------------------------------
# sections 19-27: physical convergence of a revolute fit
# ---------------------------------------------------------------------------
CONVERGED = "CONVERGED"
DEGENERATE_FLAT_DIRECTION = "DEGENERATE_FLAT_DIRECTION"
MAX_ITERS_UNSTABLE = "MAX_ITERS_UNSTABLE"
NUMERICAL_FAILURE = "NUMERICAL_FAILURE"

#: Section 26: which termination states may enter the P/R comparison.
ELIGIBLE_STATES = (CONVERGED, DEGENERATE_FLAT_DIRECTION)


def physical_change(R_a, t_a, R_b, t_b, probes, length):
    """Sections 19 and 20: how much the INDUCED MOTION moved between iterates.

        Delta_point = sqrt( sum_{t,i} || T_t^{j+1} x_i - T_t^{j} x_i ||^2 / (T N) )
        tilde Delta = Delta_point / (L + eps)

    plus section 20's per-frame transform-space changes.

    This is the primary convergence metric because the revolute
    parameterisation is not identifiable near the translation limit: as
    r -> inf, q -> 0 with r q -> d, the raw axis point and angle can drift
    without bound while every T_t stays put. Section 23 therefore defines
    convergence on the motion these parameters INDUCE, not on the parameters.
    """
    pa = torch.einsum("tij,nj->tni", R_a, probes) + t_a.unsqueeze(1)
    pb = torch.einsum("tij,nj->tni", R_b, probes) + t_b.unsqueeze(1)
    d2 = ((pb - pa) ** 2).sum(-1)
    delta_point = float(d2.mean().clamp_min(0.0).sqrt())
    dR = project_to_so3(torch.einsum("tji,tjk->tik", R_a, R_b))
    return {"delta_point": delta_point,
            "delta_point_normalised": delta_point / max(float(length), EPS),
            "delta_R_max": float(so3_log(dR).norm(dim=1).max()),
            "delta_t_max": float((t_b - t_a).norm(dim=1).max()),
            "delta_t_max_normalised": float((t_b - t_a).norm(dim=1).max())
            / max(float(length), EPS)}


def _rot(e: torch.Tensor, q) -> torch.Tensor:
    """R(e, q) = Exp(q [e]_x) for a batch of angles."""
    if not torch.is_tensor(q):
        q = torch.tensor([float(q)], dtype=torch.float64, device=e.device)
    return so3_exp(q.reshape(-1, 1).double() * e.reshape(1, 3).double())


def solve_axis_point(e: torch.Tensor, q: torch.Tensor, t_obs: torch.Tensor,
                     weights: torch.Tensor | None = None) -> torch.Tensor:
    """min-norm argmin_a sum_t w_t || t_t - (I - R(e,q_t)) a ||^2.

    Returns `a` in the closest-point gauge e^T a = 0 (see the module docstring:
    the component along e is unobservable, so the minimum-norm solution IS the
    gauge).
    """
    dev = e.device
    T = int(q.numel())
    w = (torch.ones(T, dtype=torch.float64, device=dev)
         if weights is None else weights.double().clamp_min(0.0))
    eye = torch.eye(3, dtype=torch.float64, device=dev)
    rots = _rot(e, q)
    M = eye.unsqueeze(0) - rots                                # [T,3,3]
    lhs = torch.einsum("t,tij,tik->jk", w, M, M)
    rhs = torch.einsum("t,tij,ti->j", w, M, t_obs.double())
    # Minimum-norm solve: drop the directions the data cannot see, which is
    # exactly span(e) plus anything else that happens to be unobserved.
    evals, evecs = torch.linalg.eigh(lhs)
    tol = float(evals.max().clamp_min(0.0)) * 1e-10
    inv = torch.where(evals > tol, 1.0 / evals.clamp_min(EPS),
                      torch.zeros_like(evals))
    a = evecs @ (inv * (evecs.T @ rhs))
    return a - e * (e @ a)                                     # enforce e^T a = 0


def _objective(e, a, q, R_obs, t_obs, length, w_t):
    """Section 16's E_R with section 20's weights."""
    rots = _rot(e, q)
    dR = project_to_so3(torch.einsum("tji,tjk->tik", rots, R_obs))
    rot_term = (so3_log(dR) ** 2).sum(dim=1)
    eye = torch.eye(3, dtype=torch.float64, device=e.device)
    pred_t = torch.einsum("tij,j->ti", eye.unsqueeze(0) - rots, a)
    pos_term = ((t_obs - pred_t) ** 2).sum(dim=1) / max(length ** 2, EPS)
    return float((w_t * (rot_term + pos_term)).sum()), rots, pred_t


def fit_revolute(R_obs: torch.Tensor, t_obs: torch.Tensor, length: float, *,
                 weights: torch.Tensor | None = None, max_iters: int = 200,
                 tau_error: float = 1e-6, tau_eta: float = 1e-9,
                 tau_q: float = 1e-9, flat_patience: int = 3,
                 tau_motion: float = 1e-4, tau_R: float = 1e-4,
                 tau_t: float = 1e-4, probes: torch.Tensor | None = None
                 ) -> dict:
    """Section 16's E_R, minimised over (e, a, q) jointly.

    Initialization uses the fact that for a true revolute every rotation log is
    parallel to the axis: log(R_t) = q_t e, so e is the leading eigenvector of
    sum_t log(R_t) log(R_t)^T. Refinement then alternates

        q_t  (one scalar per frame, Newton on the true objective)
        a    (linear, minimum norm -> closest-point gauge)
        e    (2-DoF Gauss-Newton in the tangent of S^2)

    until section 28's joint-fit criteria hold on all three of the error, the
    geometry and the trajectory.
    """
    R_obs = project_to_so3(R_obs.double())
    t_obs = t_obs.double()
    T = int(R_obs.shape[0])
    dev = R_obs.device
    if T == 0:
        raise ValueError("a revolute fit needs at least one frame")
    w_t = (torch.ones(T, dtype=torch.float64, device=dev)
           if weights is None else weights.double().clamp_min(0.0))
    L = max(float(length), EPS)

    # ---- initialization from the rotation logs ---------------------------
    phi = so3_log(R_obs)                                       # [T,3]
    scatter = (w_t.unsqueeze(1) * phi).T @ phi
    evals, evecs = torch.linalg.eigh(scatter)
    e = evecs[:, -1]
    e = e / e.norm().clamp_min(EPS)
    q = phi @ e
    if float(q.sum()) < 0:                                     # sign convention
        e, q = -e, -q
    q = torch.tensor(unwrap_angles([float(x) for x in q]),
                     dtype=torch.float64, device=dev)

    # Section 18's translation limit. When the observed rotations are
    # negligible there is no axis to read off log(R): q comes out 0, the
    # min-norm axis point comes out 0, and with a = 0 the translation term
    # t - (I - R(e,q)) a loses all dependence on q. The fit then sits on a
    # trivial stationary point at the FIRST iteration and reports CONVERGED
    # with a large residual, which is not the asymptotic revolute description
    # of a translation -- it is a dead start.
    #
    # Seed the limit explicitly instead. For small q,
    #     t ~= -q (e x a),
    # so choosing e perpendicular to the travel direction u and
    #     a = r (e x u)   gives   e x a = -r u   and   t ~= q r u,
    # i.e. a far axis at radius r turning by q = |t| / r. The subsequent
    # optimisation is free to move r; this only provides a point where the
    # gradient exists.
    if float(q.abs().max()) < 1e-9 and T > 1:
        tt = t_obs - t_obs[0:1]
        mags = tt.norm(dim=1)
        if float(mags.max()) > EPS:
            u = tt[int(mags.argmax())]
            u = u / u.norm().clamp_min(EPS)
            tmp = torch.tensor([0.0, 0.0, 1.0], dtype=torch.float64,
                               device=dev)
            if abs(float(u @ tmp)) > 0.9:
                tmp = torch.tensor([0.0, 1.0, 0.0], dtype=torch.float64,
                                   device=dev)
            e = torch.linalg.cross(u, tmp)
            e = e / e.norm().clamp_min(EPS)
            r = 100.0 * max(float(L), EPS)          # deep in the limit
            a_seed = r * torch.linalg.cross(e, u)
            q = (tt @ u) / r
            a = a_seed - e * (e @ a_seed)
        else:
            a = solve_axis_point(e, q, t_obs, w_t)
    else:
        a = solve_axis_point(e, q, t_obs, w_t)

    # Section 19 needs points to push through the transforms. Without a part
    # supplied, a unit shell around the origin scaled by L is used: it probes
    # rotation and translation at the length scale the error is normalised by.
    if probes is None:
        ang = torch.linspace(0.0, 2.0 * math.pi, 9, dtype=torch.float64,
                             device=dev)[:8]
        probes = torch.stack([torch.cos(ang), torch.sin(ang),
                              torch.zeros_like(ang)], dim=1) * float(L)
        probes = torch.cat([probes, probes.roll(1, dims=1)], dim=0)
    probes = probes.double()

    E, _r, _p = _objective(e, a, q, R_obs, t_obs, L, w_t)
    hist = [E]
    status, iters, flat = "max_iters", 0, 0
    d_eta = d_q = float("inf")
    # Sections 26/27 of the debug addendum: the axis direction, the axis point
    # and the q step are needed PER OPTIMIZER ITERATION so axis stability can
    # be plotted. Recording is unconditional and cheap (a few floats per
    # iteration) and changes no decision.
    iterates = [{"iteration": 0, "axis": [float(x) for x in e],
                 "axis_point": [float(x) for x in a], "energy": E,
                 "max_abs_q": float(q.abs().max())}]

    for it in range(int(max_iters)):
        iters = it + 1
        e_prev, a_prev, q_prev, E_prev = e.clone(), a.clone(), q.clone(), E

        # --- q_t: one scalar per frame, on the TRUE objective --------------
        for t in range(T):
            qt = float(q[t])
            for _ in range(12):
                h = 1e-6
                def f(x, _t=t):
                    qq = q.clone()
                    qq[_t] = x
                    return _objective(e, a, qq, R_obs[_t:_t + 1],
                                      t_obs[_t:_t + 1], L, w_t[_t:_t + 1])[0]
                f0 = f(qt)
                d1 = (f(qt + h) - f(qt - h)) / (2 * h)
                d2 = (f(qt + h) - 2 * f0 + f(qt - h)) / (h * h)
                step = -d1 / d2 if d2 > EPS else -d1 * 1e-3
                step = max(min(step, 0.5), -0.5)
                if f(qt + step) < f0:
                    qt += step
                else:
                    break
                if abs(step) < tau_q:
                    break
            q[t] = qt

        # --- a: linear, minimum norm -> closest-point gauge ---------------
        a = solve_axis_point(e, q, t_obs, w_t)

        # --- e: 2-DoF Gauss-Newton in the tangent of S^2 -------------------
        tmp = torch.tensor([1.0, 0.0, 0.0], dtype=torch.float64, device=dev)
        if abs(float(e @ tmp)) > 0.9:
            tmp = torch.tensor([0.0, 1.0, 0.0], dtype=torch.float64, device=dev)
        u1 = torch.linalg.cross(e, tmp)
        u1 = u1 / u1.norm().clamp_min(EPS)
        u2 = torch.linalg.cross(e, u1)
        h = 1e-6
        grad = torch.zeros(2, dtype=torch.float64, device=dev)
        for bi, b in enumerate((u1, u2)):
            ep = (e + h * b)
            em = (e - h * b)
            fp = _objective(ep / ep.norm(),
                            solve_axis_point(ep / ep.norm(), q, t_obs, w_t),
                            q, R_obs, t_obs, L, w_t)[0]
            fm = _objective(em / em.norm(),
                            solve_axis_point(em / em.norm(), q, t_obs, w_t),
                            q, R_obs, t_obs, L, w_t)[0]
            grad[bi] = (fp - fm) / (2 * h)
        step = 1.0
        for _ in range(24):
            cand = e - step * (grad[0] * u1 + grad[1] * u2)
            cand = cand / cand.norm().clamp_min(EPS)
            a_c = solve_axis_point(cand, q, t_obs, w_t)
            E_c, _, _ = _objective(cand, a_c, q, R_obs, t_obs, L, w_t)
            if E_c < E:
                e, a, E = cand, a_c, E_c
                break
            step *= 0.5
        else:
            E, _, _ = _objective(e, a, q, R_obs, t_obs, L, w_t)

        E, _, _ = _objective(e, a, q, R_obs, t_obs, L, w_t)
        hist.append(E)
        d_eta = float((e - e_prev).norm() + (a - a_prev).norm())
        d_q = float((q - q_prev).abs().max())
        rel = abs(E_prev - E) / max(1.0, abs(E_prev))
        iterates.append({"iteration": it + 1,
                         "axis": [float(x) for x in e],
                         "axis_point": [float(x) for x in a],
                         "energy": E, "d_eta": d_eta, "d_q": d_q,
                         "relative_energy_change": rel,
                         "max_abs_q": float(q.abs().max())})
        # ---- sections 19-24: PHYSICAL convergence -----------------------
        rots_prev = _rot(e_prev, q_prev)
        t_prev_hat = torch.einsum(
            "tij,j->ti",
            torch.eye(3, dtype=torch.float64, device=dev).unsqueeze(0)
            - rots_prev, a_prev)
        rots_now = _rot(e, q)
        t_now_hat = torch.einsum(
            "tij,j->ti",
            torch.eye(3, dtype=torch.float64, device=dev).unsqueeze(0)
            - rots_now, a)
        phys = physical_change(rots_prev, t_prev_hat, rots_now, t_now_hat,
                               probes, L)
        iterates[-1].update(phys)

        obj_ok = rel < tau_error
        motion_ok = (phys["delta_point_normalised"] < tau_motion
                     and phys["delta_R_max"] < tau_R
                     and phys["delta_t_max_normalised"] < tau_t)
        params_ok = d_eta < tau_eta and d_q < tau_q

        # Section 23: physically converged = objective stable AND the induced
        # motion stable. Section 24: if that holds but the raw parameters are
        # still moving, the trajectory is settled while the revolute
        # parameterisation is not identifiable -- a degeneracy, not a failure.
        if obj_ok and motion_ok:
            status = CONVERGED if params_ok else DEGENERATE_FLAT_DIRECTION
            break
    # Section 22: gauge q_0 = 0. The line itself is unchanged; only the
    # trajectory's origin moves, so the observed transforms must be referred to
    # the same origin when the error is reported.
    q = q - q[0]
    a = solve_axis_point(e, q, t_obs, w_t)
    rots = _rot(e, q)
    eye = torch.eye(3, dtype=torch.float64, device=dev)
    t_hat = torch.einsum("tij,j->ti", eye.unsqueeze(0) - rots, a)
    err = normalized_geometric_error(R_obs, t_obs, rots, t_hat, L)

    if status == "max_iters":
        # Section 27: reaching the cap while the INDUCED MOTION is still
        # changing is MAX_ITERS_UNSTABLE -- distinct from a flat direction,
        # where the motion has settled and only the coordinates drift.
        status = MAX_ITERS_UNSTABLE
    if not (torch.isfinite(e).all() and torch.isfinite(a).all()
            and torch.isfinite(q).all() and math.isfinite(E)):
        status = NUMERICAL_FAILURE
    last = iterates[-1] if iterates else {}
    physically_converged = status in ELIGIBLE_STATES
    converged = status == CONVERGED
    return {"kind": "revolute", "axis": e, "axis_point": a,
            "axis_iterates": iterates,
            "q": [float(x) for x in q], "R_hat": rots, "t_hat": t_hat,
            "error": err["error"], "error_terms": err,
            "num_parameters": 4 + T,        # 2 for e on S^2, 2 for a, one q/frame
            "converged": converged,
            # Section 26: eligibility for the P/R comparison is PHYSICAL.
            "eligible": physically_converged,
            "physical_motion_converged": physically_converged,
            "joint_parameters_identifiable": status == CONVERGED,
            "convergence": {"status": status, "iterations": iters,
                            "history": hist,
                            "eligible": physically_converged,
                            # Section 49: say plainly that the trajectory
                            # converged while the axis did not, rather than
                            # presenting a drifting axis as a result.
                            "trajectory_converged": physically_converged,
                            "joint_axis_identifiable": status == CONVERGED,
                            "delta_point_normalised":
                                last.get("delta_point_normalised"),
                            "delta_R_max": last.get("delta_R_max"),
                            "delta_t_max_normalised":
                                last.get("delta_t_max_normalised"),
                            "tolerances": {"tau_error": tau_error,
                                           "tau_motion": tau_motion,
                                           "tau_R": tau_R, "tau_t": tau_t,
                                           "tau_eta": tau_eta,
                                           "tau_q": tau_q},
                            "degenerate_flat_direction":
                                status == DEGENERATE_FLAT_DIRECTION,
                            "final_d_eta": d_eta, "final_d_q": d_q,
                            "relative_error_change": (
                                abs(hist[-1] - hist[-2]) / max(1.0, abs(hist[-2]))
                                if len(hist) > 1 else 0.0)},
            "diagnostics": {
                "gauge_e_dot_a": float(e @ a),
                "axis_distance_from_origin": float(a.norm()),
                "total_sweep_rad": float(q.max() - q.min()),
                "rotation_log_rank1_ratio": float(
                    evals[-1] / evals.sum().clamp_min(EPS))}}
