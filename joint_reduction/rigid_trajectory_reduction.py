"""Sections 24-35: infer the joint AFTER the rigid trajectory is known.

CODEX_IMPLEMENT_RIGID_SE3_PHASE2_AND_POSTHOC_JOINT_REDUCTION.md, Part II.

Input is `{T_t, P_{T,t}}` from rigid Phase 2 and the canonical geometry of the
moving part. Output is a one-DOF consistency statistic and the two physical
models. Section 24: no renderer is imported here, and `test_rigid_se3_phase2`
checks that by inspecting this module's imports.

WHY THIS IS A SEPARATE STAGE. A far axis with a small angle produces almost
the same pixels as a translation, so an image optimiser handed an axis as a
free parameter has a flat valley to slide down (section 36). A finite rotation
sequence does not: `Log(R_t)^vee = theta_t e_R` pins the direction, and
`p_t = (I - R_t) a` pins the point, both without a single pixel.

CONVENTIONS, stated once (sections 10, 25):

  * poses compose on the LEFT, matching Phase 1's `boxplus` and the rigid
    Phase-2 error state: `T = Exp(delta_tau^) That`;
  * relative increments are `Delta T_t = T_{t+1} T_t^{-1}` -- world/spatial,
    never `T_t^{-1} T_{t+1}`;
  * `eta = [rho; phi]`, translation first, as in `joint_manifold.se3_exp`.

ONE DELIBERATE DEVIATION FROM SECTION 32. The spec writes the pose residual as
`Log(T_M^{-1} That)`, a RIGHT perturbation, and weights it by `P_T^{-1}`. But
`P_T` is the covariance of a LEFT perturbation, because that is the rigid
Phase-2 error state. The two charts differ by `Ad_{T_M}`, whose translation
block grows with `|p|` -- so on a part that travels, the spec's pairing
silently mis-weights the fit by exactly the quantity being estimated. The
primary residual here is therefore the LEFT one, `Log(That T_M^{-1})`, which
is in the same chart as the covariance. The right-chart residual is computed
too and written to the CSV so the two can be compared.
"""

from __future__ import annotations

import math

import torch
from torch import Tensor

from estimator.joint_manifold import se3_adjoint, se3_exp, se3_log
from estimator.manifold import (so3_exp, so3_hat, so3_left_jacobian_inverse,
                                so3_log)

PRISMATIC = "prismatic"
REVOLUTE = "revolute"

# Below this the SVD of the finite-rotation stack is reporting noise, not an
# axis: it is sqrt(machine epsilon) on a quantity that is already a ratio.
AXIS_NOISE_FLOOR = float(torch.finfo(torch.float64).eps) ** 0.5


# ---------------------------------------------------------------------------
# Section 25: relative rigid increments
# ---------------------------------------------------------------------------

def relative_increments(R: Tensor, p: Tensor, times=None) -> dict:
    """`Delta T_t = T_{t+1} T_t^{-1}`, then `zeta_t = Log(Delta T_t)/dt`."""
    R = torch.as_tensor(R).double().reshape(-1, 3, 3)
    p = torch.as_tensor(p).double().reshape(-1, 3)
    T = R.shape[0]
    t_arr = (torch.arange(T, dtype=torch.float64) if times is None
             else torch.as_tensor(times).double().reshape(-1))
    dR = torch.einsum("tij,tkj->tik", R[1:], R[:-1])
    dp = p[1:] - torch.einsum("tij,tj->ti", dR, p[:-1])
    dt = (t_arr[1:] - t_arr[:-1]).clamp_min(1e-300)
    eta = se3_log(dR, dp)
    return {"delta_R": dR, "delta_p": dp, "dt": dt,
            "eta": eta, "zeta": eta / dt[:, None],
            "convention": "Delta T_t = T_{t+1} T_t^{-1} (world/spatial)"}


def part_scale(mu0: Tensor, members: Tensor) -> float:
    """`L^2 = (1/N) sum_i ||mu_i^0 - c_0||^2` (section 26)."""
    X = torch.as_tensor(mu0).double().reshape(-1, 3)[
        torch.as_tensor(members).long().reshape(-1)]
    c = X.mean(0)
    return float(((X - c) ** 2).sum(1).mean().sqrt())


# ---------------------------------------------------------------------------
# Section 27: propagate pose uncertainty into the relative twist
# ---------------------------------------------------------------------------

def relative_twist_covariance(P_tau: Tensor, dR: Tensor, dp: Tensor,
                              dt: Tensor, eta: Tensor) -> dict:
    """First-order `P_zeta` from ADJACENT POSE MARGINALS.

    With `T_t = Exp(delta_tau_t^) That_t`,

        Log(Delta T) = Log(Delta That)
                     + J_l^-1 ( delta_tau_{t+1} - Ad_{Delta That} delta_tau_t )

    so `P_eta = Jinv (P_{t+1} + Ad P_t Ad^T - X - X^T) Jinv^T`.

    THE APPROXIMATION, named (section 27): the cross-covariance `X` between
    consecutive frames is NOT available -- the filter stores per-frame
    marginals, not the full trajectory covariance -- so it is dropped. The
    two frames are positively correlated in practice, which means the
    neighbouring errors partly cancel in the difference and the dropped term
    would REDUCE `P_zeta`. The weights computed here are therefore
    conservative: too little confidence, never too much.
    """
    P = torch.as_tensor(P_tau).double().reshape(-1, 6, 6)
    n = dR.shape[0]
    Ad = se3_adjoint(dR, dp)
    out = torch.zeros((n, 6, 6), dtype=torch.float64)
    for t in range(n):
        Jl = _se3_left_jacobian_inverse(eta[t])
        S = P[t + 1] + Ad[t] @ P[t] @ Ad[t].T
        out[t] = Jl @ S @ Jl.T / float(dt[t]) ** 2
        out[t] = 0.5 * (out[t] + out[t].T)
    return {"P_zeta": out,
            "cross_covariance_used": False,
            "approximation": ("adjacent pose marginals only; the dropped "
                              "inter-frame cross-covariance would reduce "
                              "P_zeta, so these weights are conservative")}


def _se3_left_jacobian_inverse(eta: Tensor) -> Tensor:
    """`J_l^-1` for SE(3), by inverting the 6x6 left Jacobian.

    The closed form for the inverse exists but its `Q` block is long and easy
    to get subtly wrong; this is a 6x6 solve on a matrix that is well
    conditioned for any increment a real trajectory produces.
    """
    from estimator.joint_manifold import se3_left_jacobian
    J = se3_left_jacobian(torch.as_tensor(eta).double().reshape(1, 6))[0]
    return torch.linalg.inv(J)


def _scalar_weights(P_tau: Tensor | None, n: int, block: slice) -> Tensor:
    """One scalar weight per frame from a pose covariance block.

    Sections 28-30 are written with scalar `w_t`; the natural scalar from a
    6x6 covariance is the inverse mean variance of the relevant block, which
    is what a Mahalanobis fit would reduce to if the block were isotropic.
    Returns uniform weights when no covariance is available.
    """
    if P_tau is None:
        return torch.ones(n, dtype=torch.float64)
    P = torch.as_tensor(P_tau).double().reshape(-1, 6, 6)[:n]
    var = torch.stack([torch.diagonal(P[t][block, block]).mean()
                       for t in range(n)])
    w = 1.0 / var.clamp_min(1e-300)
    bad = ~torch.isfinite(w)
    if bool(bad.all()):
        return torch.ones(n, dtype=torch.float64)
    w[bad] = 0.0
    return w / w.max().clamp_min(1e-300)


# ---------------------------------------------------------------------------
# Section 26: the dimensionless one-DOF rank test
# ---------------------------------------------------------------------------

def one_dof_diagnostics(zeta: Tensor, L: float, weights=None) -> dict:
    """Weighted SVD of the dimensionless generator stack.

    A fixed one-DOF joint implies `zeta_t = qdot_t xi_J`, so the stack has
    rank 1. `gamma_1DOF = sigma_1^2 / sum sigma_k^2`.

    `v` is divided by `L` because a translation and a rotation do not share
    units: without it the statistic would change if the scene were measured
    in centimetres, and the "dominant" direction would be whichever block
    happened to carry the larger numbers.

    Section 26 is explicit that this is a STRUCTURAL DIAGNOSTIC and not a
    classifier, so no threshold is applied and no label is returned.
    """
    Z = torch.as_tensor(zeta).double().reshape(-1, 6).clone()
    n = Z.shape[0]
    L = max(float(L), 1e-300)
    Z[:, :3] = Z[:, :3] / L
    w = (torch.ones(n, dtype=torch.float64) if weights is None
         else torch.as_tensor(weights).double().reshape(-1)[:n])
    Zw = Z * w.clamp_min(0.0).sqrt()[:, None]
    U, S, Vh = torch.linalg.svd(Zw, full_matrices=False)
    s2 = S ** 2
    tot = float(s2.sum())
    gamma = float(s2[0] / tot) if tot > AXIS_NOISE_FLOOR else float("nan")
    gen = Vh[0]
    # sign gauge: the generator points the way the motion actually went
    if float((Zw @ gen).sum()) < 0:
        gen = -gen
    return {"singular_values": S.tolist(),
            "explained_fraction": (s2 / max(tot, 1e-300)).tolist(),
            "gamma_1dof": gamma,
            "generator": gen.tolist(),
            "generator_translation_norm": float(gen[:3].norm()),
            "generator_rotation_norm": float(gen[3:].norm()),
            "part_scale_L": L,
            "num_increments": n,
            "total_energy": tot,
            "note": ("section 26: structural diagnostic only. No threshold "
                     "is applied and no joint type is decided here.")}


# ---------------------------------------------------------------------------
# Section 28: the prismatic model
# ---------------------------------------------------------------------------

def fit_prismatic(R: Tensor, p: Tensor, weights=None) -> dict:
    """`e_P` by weighted PCA of the translations, `s_t = e_P^T p_t`.

    The scatter is NOT mean-centred: under the model `p_t = s_t e_P` the
    trajectory passes through the origin because `T_0 = I` is the gauge, so
    centring would throw away the one point the gauge actually fixes.
    """
    R = torch.as_tensor(R).double().reshape(-1, 3, 3)
    p = torch.as_tensor(p).double().reshape(-1, 3)
    T = p.shape[0]
    w = (torch.ones(T, dtype=torch.float64) if weights is None
         else torch.as_tensor(weights).double().reshape(-1)[:T])
    M = (w[:, None] * p).T @ p
    evals, evecs = torch.linalg.eigh(M)
    e = evecs[:, -1]
    s = p @ e
    if float((w * s).sum()) < 0:         # one continuous sign gauge
        e, s = -e, -s
    scatter = float(evals.sum())
    straight = (float(evals[-1]) / scatter) if scatter > AXIS_NOISE_FLOOR \
        else float("nan")
    # the rotation the data shows, which a pure P model cannot explain
    phi = so3_log(R)
    return {"model": PRISMATIC,
            "direction": e.tolist(), "s": s.tolist(),
            "scatter_eigenvalues": evals.flip(0).tolist(),
            "translation_rank1_fraction": straight,
            "max_abs_s": float(s.abs().max()),
            "residual_rotation_angle_max": float(phi.norm(dim=1).max()),
            "residual_rotation_angle_mean": float(phi.norm(dim=1).mean()),
            "unit_defect": float(e.norm()) - 1.0}


def prismatic_poses(direction: Tensor, s: Tensor):
    """`T_P(s_t) = (I, s_t e_P)` (section 33)."""
    e = torch.as_tensor(direction).double().reshape(3)
    s = torch.as_tensor(s).double().reshape(-1)
    T = s.numel()
    R = torch.eye(3, dtype=torch.float64).expand(T, 3, 3).clone()
    return R, s[:, None] * e[None, :]


# ---------------------------------------------------------------------------
# Section 29: the revolute axis direction, from FINITE rotations
# ---------------------------------------------------------------------------

def fit_revolute_axis(R: Tensor, weights=None) -> dict:
    """`e_R = argmax_{||e||=1} sum_t w_t (e^T phi_t)^2`, `phi_t = Log(R_t)`.

    This is the principal eigenvector of `sum w_t phi_t phi_t^T`. It uses the
    FINITE rotations, not instantaneous angular velocities: a finite rotation
    sequence about a fixed axis is exactly rank 1, while differentiating first
    throws away the signal on any frame where the part is momentarily still.
    """
    R = torch.as_tensor(R).double().reshape(-1, 3, 3)
    T = R.shape[0]
    w = (torch.ones(T, dtype=torch.float64) if weights is None
         else torch.as_tensor(weights).double().reshape(-1)[:T])
    phi = so3_log(R)
    M = (w[:, None] * phi).T @ phi
    evals, evecs = torch.linalg.eigh(M)
    e = evecs[:, -1]
    if float((w * (phi @ e)).sum()) < 0:     # section 31's sign gauge
        e = -e
    tot = float(evals.sum())
    frac = (float(evals[-1]) / tot) if tot > AXIS_NOISE_FLOOR else float("nan")
    ang = phi.norm(dim=1)
    # per-frame alignment: how parallel is each finite rotation to the axis?
    align = torch.full((T,), float("nan"), dtype=torch.float64)
    good = ang > AXIS_NOISE_FLOOR
    align[good] = ((phi[good] @ e) / ang[good]).abs().clamp(-1.0, 1.0)
    return {"axis": e.tolist(),
            "scatter_eigenvalues": evals.flip(0).tolist(),
            "rank1_fraction": frac,
            "per_frame_alignment": align.tolist(),
            "min_alignment": float(align[good].min()) if bool(good.any())
            else float("nan"),
            "max_rotation_angle": float(ang.max()),
            "frames_with_rotation": int(good.sum()),
            "unit_defect": float(e.norm()) - 1.0}


# ---------------------------------------------------------------------------
# Section 30: the axis POINT, from finite transforms
# ---------------------------------------------------------------------------

def axis_plane_basis(e: Tensor) -> Tensor:
    """An orthonormal 3x2 basis of the plane perpendicular to `e`."""
    e = torch.as_tensor(e).double().reshape(3)
    e = e / e.norm().clamp_min(1e-300)
    a = torch.tensor([1.0, 0.0, 0.0], dtype=torch.float64)
    if float(e.abs()[0]) > 0.9:
        a = torch.tensor([0.0, 1.0, 0.0], dtype=torch.float64)
    u = a - (a @ e) * e
    u = u / u.norm().clamp_min(1e-300)
    return torch.stack([u, torch.linalg.cross(e, u)], dim=1)


def fit_axis_point(R: Tensor, p: Tensor, axis: Tensor, c0: Tensor,
                   weights=None) -> dict:
    """`a* = argmin_a sum_t w_t ||(I - R_t) a - p_t||^2`, gauged on the axis.

    Section 30 forbids `a = omega x v / ||omega||^2` as the primary estimator:
    it is the instantaneous screw centre, it divides by an angular speed that
    is near zero exactly when the part is slow, and it is the route by which
    the old pipeline produced axes at infinity.

    The gauge `e_R^T (a - c_0) = 0` is imposed by construction, by solving for
    `a = c_0 + E y` with `E` spanning the plane perpendicular to the axis --
    so the returned point is the one ON the recovered axis closest to the
    canonical part centroid, and the solve is 2x2 rather than a rank-deficient
    3x3.
    """
    R = torch.as_tensor(R).double().reshape(-1, 3, 3)
    p = torch.as_tensor(p).double().reshape(-1, 3)
    e = torch.as_tensor(axis).double().reshape(3)
    e = e / e.norm().clamp_min(1e-300)
    c0 = torch.as_tensor(c0).double().reshape(3)
    T = R.shape[0]
    w = (torch.ones(T, dtype=torch.float64) if weights is None
         else torch.as_tensor(weights).double().reshape(-1)[:T])
    E = axis_plane_basis(e)
    eye = torch.eye(3, dtype=torch.float64)
    IR = eye[None] - R                                   # [T,3,3]
    Mt = IR @ E                                          # [T,3,2]
    rt = p - torch.einsum("tij,j->ti", IR, c0)           # [T,3]
    sw = w.clamp_min(0.0).sqrt()
    A = (sw[:, None, None] * Mt).reshape(-1, 2)
    b = (sw[:, None] * rt).reshape(-1)
    # SVD rather than normal equations: section 30 wants the conditioning
    # reported, and A^T A squares the condition number before we can see it.
    U, S, Vh = torch.linalg.svd(A, full_matrices=False)
    smin = float(S.min())
    cond = float(S.max() / S.min()) if smin > 0 else float("inf")
    tol = float(S.max()) * max(A.shape) * torch.finfo(torch.float64).eps
    Sinv = torch.where(S > tol, 1.0 / S, torch.zeros_like(S))
    y = Vh.T @ (Sinv * (U.T @ b))
    a = c0 + E @ y
    pred = torch.einsum("tij,j->ti", IR, a)
    res = (pred - p).norm(dim=1)
    return {"axis_point": a.tolist(),
            "singular_values": S.tolist(),
            "condition_number": cond,
            "rank_deficient": bool(smin <= tol),
            "finite_axis_residual_per_frame": res.tolist(),
            "finite_axis_residual_rms": float((res ** 2).mean().sqrt()),
            "radius_from_centroid": float((a - c0).norm()),
            "gauge": "e_R^T (a - c_0) = 0",
            "estimator": "finite (I - R_t) a = p_t, never omega x v / |omega|^2"}


# ---------------------------------------------------------------------------
# Section 31: the angular trajectory
# ---------------------------------------------------------------------------

def revolute_angles(R: Tensor, axis: Tensor) -> dict:
    """`theta_t = e_R^T phi_t`, made continuous in time.

    `Log` returns an angle in `[0, pi]`, so a part that turns past `pi` folds
    back and `e_R^T phi_t` reverses sign. The continuous series is therefore
    accumulated from the RELATIVE rotations, each of which is small; the raw
    per-frame projection is returned alongside it and the largest gap between
    the two is reported, so a wrap is visible rather than silent.
    """
    R = torch.as_tensor(R).double().reshape(-1, 3, 3)
    e = torch.as_tensor(axis).double().reshape(3)
    e = e / e.norm().clamp_min(1e-300)
    T = R.shape[0]
    phi = so3_log(R)
    raw = phi @ e
    acc = torch.zeros(T, dtype=torch.float64)
    for t in range(1, T):
        d = so3_log((R[t] @ R[t - 1].T).reshape(1, 3, 3))[0]
        acc[t] = acc[t - 1] + float(d @ e)
    return {"theta": acc.tolist(), "theta_raw_projection": raw.tolist(),
            "max_wrap_gap": float((acc - raw).abs().max()),
            "unwrapped": bool(float((acc - raw).abs().max()) > 1e-9),
            "sign_convention": "(e_R, theta) == (-e_R, -theta); "
                               "sign chosen so the net rotation is positive"}


def revolute_poses(axis: Tensor, axis_point: Tensor, theta: Tensor):
    """`T_R(theta) = (R(e,theta), (I - R(e,theta)) a)` (section 33)."""
    e = torch.as_tensor(axis).double().reshape(3)
    e = e / e.norm().clamp_min(1e-300)
    a = torch.as_tensor(axis_point).double().reshape(3)
    th = torch.as_tensor(theta).double().reshape(-1)
    R = so3_exp(th[:, None] * e[None, :]).double()
    p = a[None, :] - torch.einsum("tij,j->ti", R, a)
    return R, p


# ---------------------------------------------------------------------------
# Section 32: pose-space residuals and refinement
# ---------------------------------------------------------------------------

def pose_residuals(R_hat: Tensor, p_hat: Tensor, R_m: Tensor, p_m: Tensor,
                   chart: str = "left") -> Tensor:
    """`r_t` between the recovered pose and a model pose.

    `chart="left"`  -> `Log(That T_M^-1)`, matching the LEFT error state whose
                       covariance rigid Phase 2 reports. This is the default.
    `chart="right"` -> `Log(T_M^-1 That)`, section 32's literal formula, kept
                       for comparison. The two differ by `Ad_{T_M}`.
    """
    R_hat = torch.as_tensor(R_hat).double().reshape(-1, 3, 3)
    p_hat = torch.as_tensor(p_hat).double().reshape(-1, 3)
    R_m = torch.as_tensor(R_m).double().reshape(-1, 3, 3)
    p_m = torch.as_tensor(p_m).double().reshape(-1, 3)
    if chart == "left":
        dR = torch.einsum("tij,tkj->tik", R_hat, R_m)
        dp = p_hat - torch.einsum("tij,tj->ti", dR, p_m)
    elif chart == "right":
        dR = torch.einsum("tji,tjk->tik", R_m, R_hat)
        dp = torch.einsum("tji,tj->ti", R_m, p_hat - p_m)
    else:
        raise ValueError(f"unknown chart {chart!r}")
    return se3_log(dR, dp)


def _weighted_energy(r: Tensor, W) -> float:
    if W is None:
        return float((r * r).sum())
    W = torch.as_tensor(W).double()
    if W.dim() == 1:                                   # scalar per frame
        return float((W[:, None] * r * r).sum())
    return float(torch.einsum("ti,tij,tj->", r, W, r))


def _information_weights(P_tau: Tensor | None, T: int) -> Tensor | None:
    """`W_t = P_{T,t}^-1`, floored onto the PSD cone (section 32)."""
    if P_tau is None:
        return None
    P = torch.as_tensor(P_tau).double().reshape(-1, 6, 6)[:T]
    out = torch.zeros_like(P)
    for t in range(P.shape[0]):
        S = 0.5 * (P[t] + P[t].T)
        ev = torch.linalg.eigvalsh(S)
        lo = float(ev.min())
        if lo <= 1e-14:
            S = S + (1e-14 - lo) * torch.eye(6, dtype=torch.float64)
        out[t] = torch.linalg.inv(S)
        out[t] = 0.5 * (out[t] + out[t].T)
    return out


def _gauss_newton(residual_fn, x0: Tensor, W, iters: int = 40,
                  lam0: float = 1e-6, step_eps: float = 1e-7) -> dict:
    """Levenberg-damped Gauss-Newton with finite-difference Jacobians.

    FD rather than analytic: this problem has at most `T + 3` parameters and
    `6T` residuals, it never touches the renderer, and everything is float64
    and well scaled. The repository's no-autograd rule is about the estimator
    and the image Jacobians, both of which are elsewhere; hand-deriving four
    more Jacobians here would buy nothing and risk a sign.
    """
    x = torch.as_tensor(x0).double().reshape(-1).clone()
    n = x.numel()
    r = residual_fn(x)
    E = _weighted_energy(r, W)
    E0 = E
    lam = float(lam0)
    hist = []
    for _ in range(int(iters)):
        J = torch.zeros((r.numel(), n), dtype=torch.float64)
        for k in range(n):
            d = torch.zeros(n, dtype=torch.float64)
            d[k] = step_eps * max(1.0, float(x[k].abs()))
            J[:, k] = ((residual_fn(x + d) - residual_fn(x - d))
                       / (2.0 * float(d[k]))).reshape(-1)
        if W is None:
            JtW = J.T
        else:
            Wt = torch.as_tensor(W).double()
            if Wt.dim() == 1:
                JtW = (J * Wt.repeat_interleave(6)[:, None]).T
            else:
                blk = torch.block_diag(*[Wt[t] for t in range(Wt.shape[0])])
                JtW = J.T @ blk
        A = JtW @ J
        g = JtW @ r.reshape(-1)
        ok = False
        for _try in range(12):
            try:
                step = torch.linalg.solve(
                    A + lam * torch.diag(torch.diagonal(A).clamp_min(1e-12)),
                    -g)
            except Exception:                              # noqa: BLE001
                lam *= 10.0
                continue
            xn = x + step
            rn = residual_fn(xn)
            En = _weighted_energy(rn, W)
            if math.isfinite(En) and En < E:
                x, r, E = xn, rn, En
                lam = max(lam * 0.3, 1e-12)
                ok = True
                break
            lam *= 10.0
        hist.append(E)
        if not ok or lam > 1e10:
            break
    return {"x": x, "energy": E, "energy_initial": E0, "history": hist,
             "iterations": len(hist),
             "reduction": (E0 - E) / max(abs(E0), 1e-300)}


def refine_prismatic(R_hat, p_hat, direction, s, W=None,
                     chart: str = "left") -> dict:
    """Refine `(e_P, {s_t})` against the rigid poses. No renderer (section 32).

    `e_P` moves on the sphere through its tangent plane, so it stays a unit
    vector exactly instead of being renormalised after each step.
    """
    R_hat = torch.as_tensor(R_hat).double().reshape(-1, 3, 3)
    p_hat = torch.as_tensor(p_hat).double().reshape(-1, 3)
    T = R_hat.shape[0]
    e0 = torch.as_tensor(direction).double().reshape(3)
    e0 = e0 / e0.norm().clamp_min(1e-300)
    E = axis_plane_basis(e0)
    s0 = torch.as_tensor(s).double().reshape(-1)[:T].clone()

    def unpack(x):
        e = so3_exp((E @ x[:2]).reshape(1, 3))[0] @ e0
        sv = torch.cat([torch.zeros(1, dtype=torch.float64), x[2:]])
        return e, sv

    def res(x):
        e, sv = unpack(x)
        Rm, pm = prismatic_poses(e, sv)
        return pose_residuals(R_hat, p_hat, Rm, pm, chart)

    x0 = torch.cat([torch.zeros(2, dtype=torch.float64), s0[1:]])
    out = _gauss_newton(res, x0, W)
    e, sv = unpack(out["x"])
    return {"direction": e.tolist(), "s": sv.tolist(),
            "energy": out["energy"], "energy_initial": out["energy_initial"],
            "iterations": out["iterations"], "reduction": out["reduction"],
            "chart": chart, "renderer_called": False}


def refine_revolute(R_hat, p_hat, axis, axis_point, theta, c0, W=None,
                    chart: str = "left") -> dict:
    """Refine `(e_R, a_R, {theta_t})` against the rigid poses (section 32).

    `a_R` moves only in the plane perpendicular to the axis, which is the same
    gauge section 30 imposes -- sliding it ALONG the axis changes nothing
    about the motion and would leave the problem rank deficient.
    """
    R_hat = torch.as_tensor(R_hat).double().reshape(-1, 3, 3)
    p_hat = torch.as_tensor(p_hat).double().reshape(-1, 3)
    T = R_hat.shape[0]
    e0 = torch.as_tensor(axis).double().reshape(3)
    e0 = e0 / e0.norm().clamp_min(1e-300)
    a0 = torch.as_tensor(axis_point).double().reshape(3)
    c0 = torch.as_tensor(c0).double().reshape(3)
    th0 = torch.as_tensor(theta).double().reshape(-1)[:T].clone()
    E0 = axis_plane_basis(e0)

    def unpack(x):
        W_rot = so3_exp((E0 @ x[:2]).reshape(1, 3))[0]
        e = W_rot @ e0
        a = a0 + (W_rot @ E0) @ x[2:4]
        th = torch.cat([torch.zeros(1, dtype=torch.float64), x[4:]])
        return e, a, th

    def res(x):
        e, a, th = unpack(x)
        Rm, pm = revolute_poses(e, a, th)
        return pose_residuals(R_hat, p_hat, Rm, pm, chart)

    x0 = torch.cat([torch.zeros(4, dtype=torch.float64), th0[1:]])
    out = _gauss_newton(res, x0, W)
    e, a, th = unpack(out["x"])
    # re-apply section 30's gauge so the reported point stays the one nearest
    # the canonical centroid rather than drifting along the axis
    a = a - ((a - c0) @ e) * e
    return {"axis": e.tolist(), "axis_point": a.tolist(), "theta": th.tolist(),
            "energy": out["energy"], "energy_initial": out["energy_initial"],
            "iterations": out["iterations"], "reduction": out["reduction"],
            "radius_from_centroid": float((a - c0).norm()),
            "chart": chart, "renderer_called": False}


# ---------------------------------------------------------------------------
# The whole reduction (sections 24, 35)
# ---------------------------------------------------------------------------

def reduce_trajectory(R: Tensor, p: Tensor, mu0: Tensor, members: Tensor,
                      P_tau=None, times=None, refine: bool = True) -> dict:
    """Rigid trajectory -> one-DOF diagnostic + P and R models.

    Section 34: `R` and `p` are READ. Nothing here writes back to them, and
    neither model is allowed to adjust the rigid trajectory to fit itself.
    """
    R = torch.as_tensor(R).double().reshape(-1, 3, 3)
    p = torch.as_tensor(p).double().reshape(-1, 3)
    T = R.shape[0]
    members = torch.as_tensor(members).long().reshape(-1)
    X = torch.as_tensor(mu0).double().reshape(-1, 3)[members]
    c0 = X.mean(0)
    L = part_scale(mu0, members)

    inc = relative_increments(R, p, times)
    if P_tau is not None:
        cov = relative_twist_covariance(P_tau, inc["delta_R"], inc["delta_p"],
                                        inc["dt"], inc["eta"])
        zw = _scalar_weights(cov["P_zeta"], inc["zeta"].shape[0],
                             slice(0, 6))
    else:
        cov = {"P_zeta": None, "cross_covariance_used": False,
               "approximation": "no pose covariance supplied; uniform weights"}
        zw = torch.ones(inc["zeta"].shape[0], dtype=torch.float64)
    one_dof = one_dof_diagnostics(inc["zeta"], L, zw)

    w_t = _scalar_weights(P_tau, T, slice(0, 3))
    w_r = _scalar_weights(P_tau, T, slice(3, 6))
    W = _information_weights(P_tau, T)

    # ---- prismatic -----------------------------------------------------
    P_fit = fit_prismatic(R, p, w_t)
    Rp, pp = prismatic_poses(P_fit["direction"], P_fit["s"])
    P_fit["pose_residual_initial"] = _residual_report(R, p, Rp, pp, W)
    if refine:
        ref = refine_prismatic(R, p, P_fit["direction"], P_fit["s"], W)
        P_fit["refined"] = ref
        Rp, pp = prismatic_poses(ref["direction"], ref["s"])
        P_fit["direction"], P_fit["s"] = ref["direction"], ref["s"]
    P_fit["pose_residual"] = _residual_report(R, p, Rp, pp, W)
    P_fit["poses_R"], P_fit["poses_p"] = Rp, pp

    # ---- revolute ------------------------------------------------------
    ax = fit_revolute_axis(R, w_r)
    pt = fit_axis_point(R, p, ax["axis"], c0, w_t)
    ang = revolute_angles(R, ax["axis"])
    R_fit = {"model": REVOLUTE, **ax, "finite_axis_solver": pt,
             "axis_point": pt["axis_point"], **{"angles": ang},
             "theta": ang["theta"]}
    Rr, pr = revolute_poses(R_fit["axis"], R_fit["axis_point"], R_fit["theta"])
    R_fit["pose_residual_initial"] = _residual_report(R, p, Rr, pr, W)
    if refine:
        ref = refine_revolute(R, p, R_fit["axis"], R_fit["axis_point"],
                              R_fit["theta"], c0, W)
        R_fit["refined"] = ref
        Rr, pr = revolute_poses(ref["axis"], ref["axis_point"], ref["theta"])
        R_fit["axis"] = ref["axis"]
        R_fit["axis_point"] = ref["axis_point"]
        R_fit["theta"] = ref["theta"]
        R_fit["radius_from_centroid"] = ref["radius_from_centroid"]
    R_fit["pose_residual"] = _residual_report(R, p, Rr, pr, W)
    R_fit["poses_R"], R_fit["poses_p"] = Rr, pr

    # The far-axis tell, in units of the part itself. A revolute joint puts its
    # axis ON the object, so this is order 1. A revolute fit to a TRANSLATION
    # has to push the axis away to make the arc straight, so this blows up --
    # and it does so even when the pose energy prefers R, which is exactly the
    # case the pose energy alone cannot call. Reported, not thresholded:
    # section 26 forbids classifying from a single statistic.
    R_fit["axis_radius_over_part_scale"] = (
        float(R_fit.get("radius_from_centroid", float("nan"))) / max(L, 1e-300))
    R_fit["far_axis_note"] = (
        "radius_from_centroid / part_scale_L. Order 1 means the axis lies on "
        "the part. A large value means the revolute fit is approximating a "
        "translation with a distant axis, which section 36 identifies as the "
        "degenerate direction the OLD image optimiser could slide down.")

    return {"one_dof": one_dof,
            "increments": {k: (v.tolist() if torch.is_tensor(v) else v)
                           for k, v in inc.items()},
            "relative_twist_covariance": {
                k: v for k, v in cov.items() if k != "P_zeta"},
            "prismatic": P_fit, "revolute": R_fit,
            "part_scale_L": L, "canonical_centroid": c0.tolist(),
            "num_frames": T, "num_members": int(members.numel()),
            "weighting": ("Mahalanobis with W_t = P_tau_t^-1 in the LEFT "
                          "chart; see the module docstring for why not the "
                          "right chart section 32 writes"),
            "renderer_called": False,
            "rigid_trajectory_modified": False}


def _residual_report(R_hat, p_hat, R_m, p_m, W) -> dict:
    """Section 35: the comparison numbers, in both charts."""
    left = pose_residuals(R_hat, p_hat, R_m, p_m, "left")
    right = pose_residuals(R_hat, p_hat, R_m, p_m, "right")
    n = left.shape[0]
    return {"weighted_energy": _weighted_energy(left, W),
            "unweighted_energy": float((left * left).sum()),
            "rms_translation": float((left[:, :3] ** 2).sum(1).mean().sqrt()),
            "rms_rotation": float((left[:, 3:] ** 2).sum(1).mean().sqrt()),
            "max_translation": float(left[:, :3].norm(dim=1).max()),
            "max_rotation": float(left[:, 3:].norm(dim=1).max()),
            "per_frame_left": left.tolist(),
            "per_frame_right": right.tolist(),
            "right_chart_energy": float((right * right).sum()),
            "num_frames": n}
