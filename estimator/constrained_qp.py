"""Convex QP with linear inequality constraints for the constrained IEKF step.

    minimize    q(x) = 1/2 x^T A x - b^T x
    subject to  C x <= u                         (one row per constrained ray)

A is the IEKF information operator (symmetric positive definite: prior +
measurement + damping), C the visible-alpha constraint Jacobian lifted into the
solved state (C_s = J_A J_g,s). Both are matrix-free; a dense path builds them
explicitly when the state is small (rigid / joint states).

Method: augmented Lagrangian (method of multipliers) for inequality
constraints, a standard convergent method for convex QPs (Rockafellar 1973;
Nocedal & Wright, ch. 17), NOT a fixed penalty:
    L_rho(x, nu) = q(x) + 1/(2 rho) sum_i [max(0, nu_i + rho (C x - u)_i)^2 - nu_i^2]
inner: L_rho is convex and C^1 with a semismooth gradient
    grad = A x - b + C^T max(0, nu + rho (C x - u)),
so semismooth Newton with the generalized Hessian A + rho C_I^T C_I (I the
currently positive set), solved by CG, plus an Armijo line search on L_rho.
outer: nu <- max(0, nu + rho (C x - u)); rho grows only while feasibility
stalls. Stops on primal feasibility AND complementarity, never on a budget
alone; a budget exit is reported as such.

Also provided:
  * zero_rows(): rows the state cannot move (C_i = 0 to round-off), found from
    two random JVPs -- a violated zero row is reported INDISTINGUISHABLE, not
    "fixed";
  * data_conflict_test(): q(x_c) - q(x_u) against 1/2 chi2_{0.99}(k), k the
    number of active constraints (capped at the state dimension). Under a
    correct model whose truth satisfies the constraints this excess is
    approximately 1/2 chi2(k); a much larger excess means the constraint can only
    be met by contradicting the RGB-D (e.g. a whole rigid part moved to hide one
    misowned Gaussian) and is reported as a model / ownership infeasibility.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import torch


def _dot(a, b):
    return float((a.double() * b.double()).sum())


@dataclass
class QPResult:
    x: torch.Tensor
    nu: torch.Tensor
    status: str                       # OPTIMAL | BUDGET_EXIT | INFEASIBLE_SUSPECTED
    outer_iterations: int
    inner_iterations: int
    cg_iterations: int
    max_violation: float              # max_i (C x - u)_+ at the returned x (linear model)
    complementarity: float
    active: int
    objective: float
    history: list = field(default_factory=list)
    seconds_in_A_solves: float = 0.0


def cg(apply, rhs, x0=None, tol=1e-8, max_iters=200, M_inv=None):
    """(Preconditioned) CG for an SPD operator; returns (x, iterations, relative residual)."""
    x = torch.zeros_like(rhs) if x0 is None else x0.clone()
    r = rhs - apply(x) if x0 is not None else rhs.clone()
    z = M_inv(r) if M_inv is not None else r
    p = z.clone()
    rz = _dot(r, z)
    r0 = math.sqrt(max(_dot(rhs, rhs), 1e-300))
    it = 0
    while it < max_iters and math.sqrt(max(_dot(r, r), 0.0)) > tol * r0:
        Ap = apply(p)
        pAp = _dot(p, Ap)
        if pAp <= 0:
            break
        a = rz / pAp
        x = x + a * p
        r = r - a * Ap
        z = M_inv(r) if M_inv is not None else r
        rz_new = _dot(r, z)
        p = z + (rz_new / rz) * p
        rz = rz_new
        it += 1
    return x, it, math.sqrt(max(_dot(r, r), 0.0)) / r0


def solve_qp_alm(A_mv, b, C_mv, CT_mv, u, x0=None, *, tol_feas=1e-4, tol_comp=1e-6,
                 max_outer=40, max_newton=8, cg_tol=1e-6, cg_iters=200, rho0=None, M_inv=None,
                 newton_cg_tol=1e-2):
    """See the module docstring. u, C_mv(x) and CT_mv(nu) live on the row vector."""
    x = torch.zeros_like(b) if x0 is None else x0.clone()
    m = int(u.numel())
    nu = torch.zeros_like(u)
    if m == 0:
        x, it, _ = cg(A_mv, b, x0=x, tol=cg_tol, max_iters=cg_iters, M_inv=M_inv)
        return QPResult(x, nu, "OPTIMAL", 0, 0, it, 0.0, 0.0, 0, 0.5 * _dot(x, A_mv(x)) - _dot(b, x))
    if rho0 is None:      # balance the constraint curvature against A
        g = torch.Generator(device=b.device).manual_seed(0)
        v = torch.randn(b.shape, generator=g, device=b.device, dtype=b.dtype)
        Cv = C_mv(v)
        rho0 = _dot(v, A_mv(v)) / max(_dot(Cv, Cv), 1e-30)
    rho = float(rho0)
    tot_in = tot_cg = 0
    hist = []
    viol_prev = float("inf")
    status = "BUDGET_EXIT"
    for k in range(max_outer):
        # ---- inner: minimize L_rho(., nu) by semismooth Newton-CG ----------
        for _n in range(max_newton):
            s = nu + rho * (C_mv(x) - u)
            sp = s.clamp_min(0.0)
            grad = A_mv(x) - b + CT_mv(sp)
            gn = math.sqrt(_dot(grad, grad))
            if gn <= cg_tol * max(math.sqrt(_dot(b, b)), 1.0):
                break
            act = (s > 0).to(x.dtype)

            def G(v, act=act):
                return A_mv(v) + rho * CT_mv(act * C_mv(v))
            # inexact Newton: the direction only has to be a descent direction of
            # the convex L_rho; the line search and the outer loop carry the rest.
            # A's own preconditioner is used (the constraint block is low rank).
            d, it, _ = cg(G, -grad, tol=newton_cg_tol, max_iters=cg_iters, M_inv=M_inv)
            tot_cg += it
            tot_in += 1

            def Lval(z):
                sz = (nu + rho * (C_mv(z) - u)).clamp_min(0.0)
                return 0.5 * _dot(z, A_mv(z)) - _dot(b, z) + (_dot(sz, sz) - _dot(nu, nu)) / (2 * rho)
            L0, slope, t = Lval(x), _dot(grad, d), 1.0
            while t > 1e-8 and Lval(x + t * d) > L0 + 1e-4 * t * slope:
                t *= 0.5
            x = x + t * d
            if t <= 1e-8:
                break
        r = C_mv(x) - u
        viol = float(r.clamp_min(0.0).max())
        nu = (nu + rho * r).clamp_min(0.0)
        comp = float((nu * r).abs().max())
        hist.append({"outer": k, "rho": rho, "max_violation": viol, "complementarity": comp,
                     "active": int((nu > 0).sum())})
        if viol <= tol_feas and comp <= tol_comp * max(1.0, float(nu.abs().max())):
            status = "OPTIMAL"
            break
        if viol > 0.25 * viol_prev:
            rho *= 10.0
        viol_prev = viol
        if rho > 1e12 * float(rho0):
            status = "INFEASIBLE_SUSPECTED"
            break
    r = C_mv(x) - u
    return QPResult(x, nu, status, len(hist), tot_in, tot_cg, float(r.clamp_min(0.0).max()),
                    float((nu * r).abs().max()), int((nu > 0).sum()),
                    0.5 * _dot(x, A_mv(x)) - _dot(b, x), hist)


def solve_qp_projection(A_mv, d0, C_mv, CT_mv, u, *, M_inv=None, keys=None, tail_key=None, cache=None,
                        tol_feas=1e-6, max_rounds=40, add_per_round=8, max_rows=256,
                        z_cg_tol=1e-6, z_cg_iters=150):
    """A-norm projection onto the linearized constraints, by constraint generation
    (a dual active-set method):

        minimize 1/2 (x - d0)^T A (x - d0)   subject to   C x <= u.

    With d0 = A^-1 b this is exactly min q(x) s.t. C x <= u, so classify-style
    callers pass the unconstrained minimizer.

    The algorithm keeps a working set W of rows:
      * each row i in W carries c_i = C^T e_i and z_i = A^-1 c_i (one PCG solve
        with A's preconditioner);
      * on W the problem is solved exactly through its dual,
            min_{lam >= 0} 1/2 lam^T S lam - g^T lam,
        with S_ij = c_i^T z_j and g = C_W d0 - u_W (NNLS on the Cholesky factor
        of S, plus a few refinement passes that absorb the PCG asymmetry of S);
      * the primal is x = d0 - Z_W lam;
      * EVERY row is then checked with one C x product, and the most-violated
        rows outside W are added.
    Termination: W only grows, so the method is finite.
    - Every row satisfied gives OPTIMAL.
    - Running out of rows or rounds gives BUDGET_EXIT.
    - A violated row with c_i = 0 gives INFEASIBLE_SUSPECTED.

    Why not the ALM: on the IEKF the active set is tiny (1-5 rays of millions),
    and this costs one A-solve per row added. solve_qp_alm's Newton systems
    A + rho C_I^T C_I grow ill-conditioned as rho rises; on the 8-camera fridge it
    needed 400-3700 CG iterations per QP.

    `cache` (dict key -> (c_i, z_i)) reuses rows across re-solves at the same
    linearization and the same A (second-order corrections, the end-of-frame
    classification). `keys[i]` names row i in it (rows past the end of a key
    tensor take `tail_key`); a None key is never cached.

    z_cg_tol / z_cg_iters: accuracy of each A^-1 c_i. Feasibility on W does not
    depend on it, because S is built from the same z_i. Only the metric of the
    projection does, so the IEKF passes its own trust-region PCG accuracy."""
    import time
    import numpy as np
    from scipy.optimize import nnls
    t_cg = 0.0
    m = int(u.numel())
    nu = torch.zeros_like(u)
    x = d0.clone()
    if m == 0:
        return QPResult(x, nu, "OPTIMAL", 0, 0, 0, 0.0, 0.0, 0, 0.0)
    W, cs, zs, hist = [], [], [], []
    tot_cg = 0
    lam = np.zeros(0)
    W_lam = []                                         # the working set lam belongs to
    status = "BUDGET_EXIT"
    in_W = torch.zeros(m, dtype=torch.bool, device=u.device)
    for rnd in range(max_rounds):
        r = C_mv(x) - u
        viol = float(r.max().clamp_min(0.0))
        hist.append({"round": rnd, "rows": len(W), "max_violation": viol})
        if viol <= tol_feas:
            status = "OPTIMAL"
            break
        cand = torch.where(in_W, torch.full_like(r, -float("inf")), r)
        k = min(add_per_round, int((cand > tol_feas).sum()), max_rows - len(W))
        if k <= 0:
            break
        new = torch.topk(cand, k).indices.tolist()
        for i in new:
            if keys is None:
                key = None
            elif torch.is_tensor(keys):                # e.g. the global ray index of each row
                key = int(keys[i]) if i < keys.numel() else tail_key
            else:
                key = keys[i]
            if cache is not None and key is not None and key in cache:
                c, z = cache[key]
            else:
                e = torch.zeros_like(u)
                e[i] = 1.0
                c = CT_mv(e).to(d0.dtype)
                if float(c.abs().max()) == 0.0:
                    status = "INFEASIBLE_SUSPECTED"     # a violated row the state cannot move
                    break
                _t = time.perf_counter()
                z, it, _ = cg(A_mv, c, tol=z_cg_tol, max_iters=z_cg_iters, M_inv=M_inv)
                t_cg += time.perf_counter() - _t
                tot_cg += it
                if cache is not None and key is not None:
                    cache[key] = (c, z)
            W.append(i)
            cs.append(c)
            zs.append(z)
            in_W[i] = True
        if status == "INFEASIBLE_SUSPECTED":
            break
        Cm = torch.stack([c.reshape(-1) for c in cs]).double()      # state vectors may be [N,12]
        Zm = torch.stack([z.reshape(-1) for z in zs]).double()
        S = (Cm @ Zm.T).cpu().numpy()
        g = ((Cm @ d0.reshape(-1).double()).cpu().numpy() - u[W].double().cpu().numpy())
        Ss = 0.5 * (S + S.T)
        ridge = 1e-12 * max(float(np.trace(Ss)) / len(W), 1e-300)
        while True:
            try:
                L = np.linalg.cholesky(Ss + ridge * np.eye(len(W)))
                break
            except np.linalg.LinAlgError:              # (near-)parallel rows: S is rank deficient
                ridge *= 100.0
        tgt = g.copy()
        for _ref in range(4):
            lam, _ = nnls(L.T, np.linalg.solve(L, tgt), maxiter=50 * len(W))
            res = g - S @ lam                          # = C_W x - u_W
            err = np.where(lam > 0, res, np.maximum(res, 0.0))
            if float(np.abs(err).max()) <= tol_feas:
                break
            tgt = tgt + err
        x = d0 - (Zm.T @ torch.from_numpy(lam).to(Zm.device)).reshape(d0.shape).to(d0.dtype)
        W_lam = list(W)
    if W_lam:
        nu[torch.tensor(W_lam, device=u.device)] = torch.from_numpy(lam).to(u.device, u.dtype)
    r = C_mv(x) - u
    q = 0.5 * _dot(x - d0, A_mv(x - d0))
    return QPResult(x, nu, status, len(hist), len(W), tot_cg, float(r.clamp_min(0.0).max()),
                    float((nu * r).abs().max()), int((nu > 0).sum()), q, hist, t_cg)


def zero_rows(C_mv, like, n_probe: int = 2, seed: int = 0):
    """Rows of C that are zero to round-off (the state cannot move that ray)."""
    g = torch.Generator(device=like.device).manual_seed(seed)
    z = None
    for _ in range(n_probe):
        v = torch.randn(like.shape, generator=g, device=like.device, dtype=like.dtype)
        a = C_mv(v).abs()
        z = a if z is None else torch.maximum(z, a)
    return z == 0


def chi2_ppf_99(k: int) -> float:
    """0.99 quantile of chi2(k) (Wilson-Hilferty; exact enough for a test bound)."""
    if k <= 0:
        return 0.0
    z = 2.3263478740408408
    return k * (1 - 2 / (9 * k) + z * math.sqrt(2 / (9 * k))) ** 3


def data_conflict_test(q_constrained: float, q_unconstrained: float, active: int, dim: int) -> dict:
    k = max(1, min(int(active), int(dim)))
    excess = q_constrained - q_unconstrained
    bound = 0.5 * chi2_ppf_99(k)
    return {"objective_excess": excess, "bound_half_chi2_99": bound, "dof": k,
            "conflict": bool(excess > bound)}
