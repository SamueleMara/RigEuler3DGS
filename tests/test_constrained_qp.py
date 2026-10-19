#!/usr/bin/env python
"""estimator/constrained_qp.py on problems with a known answer (CPU).

  [1] random convex QPs: the ALM solution equals an independent solver
      (scipy SLSQP) and satisfies KKT (stationarity, primal/dual feasibility,
      complementarity)
  [2] many rows (more constraints than variables, most inactive)
  [3] an inactive constraint set returns the unconstrained minimizer
  [4] zero_rows finds exactly the rows the state cannot move
  [5] an infeasible system (a zero row with u < 0) is reported, not "solved"
  [6] the data-conflict test: a constraint consistent with the data passes,
      one that forces a large move fails
  [7] solve_qp_projection (the dual active-set solver the IEKF uses): projecting
      the unconstrained minimizer equals SLSQP on the same QP and satisfies KKT,
      also with duplicated (parallel) rows; the row cache reproduces the result
      with no new A-solves; an infeasible zero row is not reported OPTIMAL

    python tests/test_constrained_qp.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
CHECKS, FAILURES = 0, []


def check(label, cond, detail=""):
    global CHECKS
    CHECKS += 1
    print(f"  {'ok  ' if cond else 'FAIL'}  {label}  {detail}")
    if not cond:
        FAILURES.append(label)


def problem(n, m, seed):
    g = np.random.default_rng(seed)
    M = g.normal(size=(n, n))
    A = M @ M.T + n * np.eye(n)
    b = g.normal(size=n) * 5
    C = g.normal(size=(m, n))
    # feasible by construction: an interior point x_f with slack, so the many-row
    # cases test the solver, not an empty feasible set
    x_f = g.normal(size=n) * 0.1
    u = C @ x_f + np.abs(g.normal(size=m)) * 0.5
    return A, b, C, u


def ops(A, b, C, u):
    At, bt, Ct, ut = (torch.tensor(x, dtype=torch.float64) for x in (A, b, C, u))
    return (lambda v: At @ v), bt, (lambda v: Ct @ v), (lambda w: Ct.T @ w), ut


def scipy_qp(A, b, C, u):
    from scipy.optimize import minimize
    n = A.shape[0]
    r = minimize(lambda x: 0.5 * x @ A @ x - b @ x, np.zeros(n), jac=lambda x: A @ x - b, method="SLSQP",
                 constraints=[{"type": "ineq", "fun": lambda x: u - C @ x, "jac": lambda x: -C}],
                 options={"ftol": 1e-14, "maxiter": 1000})
    return r.x


def main() -> int:
    from estimator.constrained_qp import data_conflict_test, solve_qp_alm, solve_qp_projection, zero_rows
    print("\n[1-2] against scipy SLSQP, with KKT")
    for (n, m, seed) in ((6, 10, 0), (12, 30, 1), (8, 200, 2)):
        A, b, C, u = problem(n, m, seed)
        Am, bt, Cm, CTm, ut = ops(A, b, C, u)
        res = solve_qp_alm(Am, bt, Cm, CTm, ut, tol_feas=1e-9, tol_comp=1e-9)
        xs = scipy_qp(A, b, C, u)
        x = res.x.numpy()
        stat = np.linalg.norm(A @ x - b + C.T @ res.nu.numpy()) / np.linalg.norm(b)
        prim = float(np.maximum(C @ x - u, 0).max())
        comp = float(np.abs(res.nu.numpy() * (C @ x - u)).max())
        check(f"[1] n={n} m={m}: ALM = SLSQP", np.linalg.norm(x - xs) / max(np.linalg.norm(xs), 1e-9) < 1e-5,
              f"rel {np.linalg.norm(x - xs) / max(np.linalg.norm(xs), 1e-9):.1e}, status {res.status}, active {res.active}")
        check(f"[1] n={n} m={m}: KKT", stat < 1e-6 and prim < 1e-8 and comp < 1e-7 and bool((res.nu >= 0).all()),
              f"stationarity {stat:.1e}, primal {prim:.1e}, compl {comp:.1e}")

    print("\n[3] inactive constraints")
    A, b, C, u = problem(6, 10, 3)
    u = u + 100.0
    Am, bt, Cm, CTm, ut = ops(A, b, C, u)
    res = solve_qp_alm(Am, bt, Cm, CTm, ut)
    check("[3] returns A^-1 b", np.allclose(res.x.numpy(), np.linalg.solve(A, b), atol=1e-6) and res.active == 0)

    print("\n[4-5] zero rows and infeasibility")
    A, b, C, u = problem(6, 10, 4)
    C[[2, 7]] = 0.0
    Am, bt, Cm, CTm, ut = ops(A, b, C, u)
    zr = zero_rows(Cm, bt)
    check("[4] zero rows found exactly", torch.equal(torch.nonzero(zr).reshape(-1), torch.tensor([2, 7])))
    u2 = u.copy(); u2[2] = -0.5                           # a row the state cannot satisfy
    Am, bt, Cm, CTm, ut = ops(A, b, C, u2)
    res = solve_qp_alm(Am, bt, Cm, CTm, ut, max_outer=15)
    check("[5] an infeasible system is not reported OPTIMAL", res.status != "OPTIMAL" and res.max_violation > 0.4,
          f"status {res.status}, violation {res.max_violation:.2f}")

    print("\n[6] data-conflict test")
    n = 6
    A = np.diag([1e4] * n); b = A @ np.ones(n)            # data says x = 1 with sigma 1e-2
    q = lambda x: 0.5 * x @ A @ x - b @ x
    xu = np.ones(n)
    C = np.zeros((1, n)); C[0, 0] = 1.0
    for lim, expect in ((1.0 - 1e-3, False), (0.5, True)):    # a 0.1-sigma vs a 50-sigma move
        Am, bt, Cm, CTm, ut = ops(A, b, C, np.array([lim]))
        res = solve_qp_alm(Am, bt, Cm, CTm, ut, tol_feas=1e-9)
        t = data_conflict_test(q(res.x.numpy()), q(xu), res.active, n)
        check(f"[6] x0 <= {lim}: conflict {expect}", t["conflict"] == expect,
              f"excess {t['objective_excess']:.3g} vs bound {t['bound_half_chi2_99']:.3g}")

    print("\n[7] dual active-set projection (the IEKF's solver)")
    for (n, m, seed, dup) in ((6, 10, 0, False), (12, 30, 1, False), (8, 200, 2, False), (10, 40, 5, True)):
        A, b, C, u = problem(n, m, seed)
        if dup:                                           # parallel / duplicated rows, as on neighbouring rays
            C = np.vstack([C, 2.0 * C[:10]]); u = np.concatenate([u, 2.0 * u[:10]])
        Am, bt, Cm, CTm, ut = ops(A, b, C, u)
        d0 = torch.linalg.solve(torch.tensor(A), bt)
        cache = {}
        res = solve_qp_projection(Am, d0, Cm, CTm, ut, tol_feas=1e-10, keys=list(range(len(u))), cache=cache)
        xs = scipy_qp(A, b, C, u)
        x = res.x.numpy()
        stat = np.linalg.norm(A @ x - b + C.T @ res.nu.numpy()) / np.linalg.norm(b)
        prim = float(np.maximum(C @ x - u, 0).max())
        comp = float(np.abs(res.nu.numpy() * (C @ x - u)).max())
        tag = f"n={n} m={len(u)}{' (duplicated rows)' if dup else ''}"
        check(f"[7] {tag}: projection = SLSQP", np.linalg.norm(x - xs) / max(np.linalg.norm(xs), 1e-9) < 1e-5,
              f"rel {np.linalg.norm(x - xs) / max(np.linalg.norm(xs), 1e-9):.1e}, status {res.status}, "
              f"rows {res.inner_iterations}, active {res.active}")
        check(f"[7] {tag}: KKT", res.status == "OPTIMAL" and stat < 1e-6 and prim < 1e-8 and comp < 1e-7
              and bool((res.nu >= 0).all()), f"stationarity {stat:.1e}, primal {prim:.1e}, compl {comp:.1e}")
        if seed == 1:
            res2 = solve_qp_projection(Am, d0, Cm, CTm, ut, tol_feas=1e-10, keys=list(range(len(u))), cache=cache)
            check("[7] the row cache reproduces the solution with no new A-solves",
                  res2.cg_iterations == 0 and torch.allclose(res2.x, res.x), f"cg {res2.cg_iterations}")
    A, b, C, u = problem(6, 10, 4)
    C[2] = 0.0; u[2] = -0.5
    Am, bt, Cm, CTm, ut = ops(A, b, C, u)
    res = solve_qp_projection(Am, torch.linalg.solve(torch.tensor(A), bt), Cm, CTm, ut)
    check("[7] an infeasible zero row is not reported OPTIMAL", res.status != "OPTIMAL", f"status {res.status}")
    print()
    if FAILURES:
        print(f"{len(FAILURES)} of {CHECKS} checks FAILED")
        return 1
    print(f"{CHECKS} checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
