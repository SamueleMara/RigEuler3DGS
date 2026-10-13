"""T7 — the full-window operator against an exact linear-Gaussian reference.

CODEX_IMPLEMENT_EXACT_UNIFIED_PHASE1_IEKF_IEKS.md section 7 and test T7.

The operator claims to be the information matrix of ONE full-window Markov
posterior. Three independent constructions must agree on a linear-Gaussian
problem where the answer is computable in closed form:

  1. the matrix-free `information_matvec` / `rhs`,
  2. a dense normal-equation matrix assembled in the TEST from the factor
     Jacobians, by a different route,
  3. a covariance-form Rauch-Tung-Striebel smoother.

Agreement of (1) and (2) says the operator applies the factors it claims.
Agreement with (3) says those factors really are the Markov posterior and not
a per-frame filter wearing its name -- an implementation that dropped the
temporal off-diagonals would still pass (1) vs (2) and fail against RTS.

Also tested: that double-counting an observation CHANGES the answer, so the
"use each image once" rule is enforced by a test that fails when broken
rather than by a comment.

    python tests/test_unified_full_window_map.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from bidirectional_iekf.full_window_free import (  # noqa: E402
    DenseLinearMeasurement,
    FullWindowFreeOperator,
    ProcessFactor,
    block_jacobi_preconditioner,
    pcg,
    schur_forward_backward,
)

ERROR_DIM = 12
CHECKS = 0
FAILURES: list[str] = []


def check(label: str, condition: bool, detail: str = "") -> None:
    global CHECKS
    CHECKS += 1
    if not condition:
        FAILURES.append(f"{label}: {detail or 'failed'}")
        print(f"  FAIL  {label}  {detail}")


def build_linear_problem(T=5, N=2, m=4, seed=0, duplicate_frame=None):
    """A linear-Gaussian chain with explicit F, Q, H, R.

    Deliberately NOT the SO(3) process: T7 is about the window algebra, and a
    linear chain is the only case with a closed-form reference to compare to.
    """
    g = torch.Generator().manual_seed(seed)
    size = N * ERROR_DIM

    F = torch.eye(ERROR_DIM, dtype=torch.float64).expand(N, ERROR_DIM, ERROR_DIM).clone()
    F[:, 0:3, 6:9] = torch.eye(3, dtype=torch.float64) * 0.1       # mu += dt v
    F[:, 3:6, 9:12] = torch.eye(3, dtype=torch.float64) * 0.1      # phi += dt omega

    Q = torch.eye(ERROR_DIM, dtype=torch.float64).expand(N, ERROR_DIM, ERROR_DIM).clone() * 0.05
    Q_inv = torch.linalg.inv(Q)
    P0_inv = torch.eye(ERROR_DIM, dtype=torch.float64).expand(
        N, ERROR_DIM, ERROR_DIM).clone() * 2.0

    process = []
    for t in range(T - 1):
        process.append(ProcessFactor(
            frame=t, A=-F.clone(), B=torch.eye(ERROR_DIM, dtype=torch.float64)
            .expand(N, ERROR_DIM, ERROR_DIM).clone(),
            Q_inv=Q_inv.clone(),
            residual=torch.randn(N, ERROR_DIM, generator=g, dtype=torch.float64) * 0.02))

    measurements = []
    for t in range(T):
        H = torch.randn(m, size, generator=g, dtype=torch.float64)
        R_inv = torch.eye(m, dtype=torch.float64) * 4.0
        r = torch.randn(m, generator=g, dtype=torch.float64) * 0.1
        measurements.append(DenseLinearMeasurement(t, H, R_inv, r))
        if duplicate_frame is not None and t == duplicate_frame:
            measurements.append(DenseLinearMeasurement(t, H.clone(),
                                                       R_inv.clone(), r.clone()))

    prior_residual = torch.randn(N, ERROR_DIM, generator=g, dtype=torch.float64) * 0.05
    operator = FullWindowFreeOperator(
        num_frames=T, num_gaussians=N, process_factors=process,
        measurement_factors=measurements,
        prior_information=P0_inv, prior_residual=prior_residual)
    return operator, F, Q_inv, P0_inv, prior_residual


def dense_reference(operator, F, Q_inv, P0_inv, prior_residual):
    """Assemble A and b in the TEST, by a different route than the operator.

    Built from the stacked factor Jacobian J and weight W as J^T W J, rather
    than by accumulating blocks, so a shared bug is unlikely to cancel.
    """
    T, N = operator.T, operator.N
    size = N * ERROR_DIM
    total = T * size

    rows, weights, resid = [], [], []

    prior_row = torch.zeros(size, total, dtype=torch.float64)
    prior_row[:, :size] = torch.eye(size, dtype=torch.float64)
    rows.append(prior_row)
    weights.append(_block_diag(P0_inv))
    resid.append(-prior_residual.reshape(-1))

    for factor in operator.process:
        t = factor.frame
        row = torch.zeros(size, total, dtype=torch.float64)
        row[:, t * size:(t + 1) * size] = _block_diag(factor.A)
        row[:, (t + 1) * size:(t + 2) * size] = _block_diag(factor.B)
        rows.append(row)
        weights.append(_block_diag(factor.Q_inv))
        resid.append(-factor.residual.reshape(-1))

    for factor in operator.measurements:
        t = factor.frame
        row = torch.zeros(factor.H.shape[0], total, dtype=torch.float64)
        row[:, t * size:(t + 1) * size] = factor.H
        rows.append(row)
        weights.append(factor.R_inv)
        resid.append(factor.residual())

    J = torch.cat(rows, dim=0)
    W = torch.block_diag(*weights)
    e = torch.cat(resid, dim=0)
    return J.transpose(0, 1) @ W @ J, J.transpose(0, 1) @ W @ e


def _block_diag(per_gauss: torch.Tensor) -> torch.Tensor:
    return torch.block_diag(*[per_gauss[i] for i in range(per_gauss.shape[0])])


def rts_reference(operator, F, Q_inv, P0_inv, prior_residual):
    """Covariance-form RTS on the same linear chain, as an independent check.

    Information-form Kalman forward pass then the standard RTS backward
    recursion. Returns the smoothed tangent per frame.
    """
    T, N = operator.T, operator.N
    size = N * ERROR_DIM
    Fb = _block_diag(F)
    Qb = torch.linalg.inv(_block_diag(Q_inv))
    P0 = torch.linalg.inv(_block_diag(P0_inv))

    by_frame = {}
    for factor in operator.measurements:
        by_frame.setdefault(factor.frame, []).append(factor)

    # Forward filter. The nominal is zero, so the "measurement" of frame t is
    # its residual and the process factor's residual is a known offset.
    x_pred, P_pred, x_filt, P_filt = [], [], [], []
    x, P = -prior_residual.reshape(-1).clone(), P0.clone()
    for t in range(T):
        x_pred.append(x.clone())
        P_pred.append(P.clone())
        info = torch.linalg.inv(P)
        rhs = info @ x
        for factor in by_frame.get(t, []):
            info = info + factor.H.transpose(0, 1) @ factor.R_inv @ factor.H
            rhs = rhs + factor.H.transpose(0, 1) @ factor.R_inv @ factor.residual()
        P = torch.linalg.inv(info)
        x = P @ rhs
        x_filt.append(x.clone())
        P_filt.append(P.clone())
        if t < T - 1:
            offset = -operator.process[t].residual.reshape(-1)
            x = Fb @ x + offset
            P = Fb @ P @ Fb.transpose(0, 1) + Qb

    smoothed = [None] * T
    smoothed[T - 1] = x_filt[T - 1]
    P_s = P_filt[T - 1]
    for t in range(T - 2, -1, -1):
        C = P_filt[t] @ Fb.transpose(0, 1) @ torch.linalg.inv(P_pred[t + 1])
        smoothed[t] = x_filt[t] + C @ (smoothed[t + 1] - x_pred[t + 1])
        P_s = P_filt[t] + C @ (P_s - P_pred[t + 1]) @ C.transpose(0, 1)
    return torch.stack(smoothed)


# --- 1. operator vs independently assembled dense reference ---------------
def test_operator_matches_dense() -> None:
    print("[1] matrix-free operator vs dense normal equations")
    op, F, Q_inv, P0_inv, pr = build_linear_problem()
    A_ref, b_ref = dense_reference(op, F, Q_inv, P0_inv, pr)
    A_op = op.dense_information()
    b_op = op.rhs().reshape(-1)

    check("information matrices agree",
          float((A_op - A_ref).abs().max()) < 1e-9,
          f"max dev {float((A_op - A_ref).abs().max()):.3e}")
    check("right-hand sides agree",
          float((b_op - b_ref).abs().max()) < 1e-9,
          f"max dev {float((b_op - b_ref).abs().max()):.3e}")
    check("operator is symmetric",
          float((A_op - A_op.transpose(0, 1)).abs().max()) < 1e-10)
    check("operator is positive definite",
          bool((torch.linalg.eigvalsh(A_op) > 0).all()))

    # The off-diagonals must actually be there.
    size = op.N * ERROR_DIM
    off = A_op[:size, size:2 * size]
    check("temporal off-diagonal block is nonzero",
          float(off.abs().max()) > 1e-6,
          "a per-frame filter would leave this zero and still look like a trajectory")

    # And they must equal the K^T Q^-1 K reduction when K = [-F, I].
    expected = -_block_diag(F).transpose(0, 1) @ _block_diag(Q_inv)
    check("off-diagonal equals -F^T Q^-1",
          float((off - expected).abs().max()) < 1e-9,
          f"max dev {float((off - expected).abs().max()):.3e}")


# --- 2. exact solve three ways --------------------------------------------
def test_three_solvers_agree() -> None:
    print("[2] dense solve vs Schur forward/backward vs PCG vs RTS")
    op, F, Q_inv, P0_inv, pr = build_linear_problem()
    A_ref, b_ref = dense_reference(op, F, Q_inv, P0_inv, pr)
    x_dense = torch.linalg.solve(A_ref, b_ref)

    diag, upper = op.block_tridiagonal()
    size = op.N * ERROR_DIM
    rhs_blocks = [op.rhs().reshape(op.T, size)[t] for t in range(op.T)]
    x_schur = torch.cat(schur_forward_backward(diag, upper, rhs_blocks))
    check("Schur elimination matches the dense solve",
          float((x_schur - x_dense).abs().max()) < 1e-8,
          f"max dev {float((x_schur - x_dense).abs().max()):.3e}")

    x_pcg, info = pcg(op, op.rhs(), preconditioner=block_jacobi_preconditioner(op),
                      max_iters=500, tol=1e-12)
    check("PCG converged", info["status"] == "CONVERGED", str(info["status"]))
    check("PCG matches the dense solve",
          float((x_pcg.reshape(-1) - x_dense).abs().max()) < 1e-6,
          f"max dev {float((x_pcg.reshape(-1) - x_dense).abs().max()):.3e}")

    x_rts = rts_reference(op, F, Q_inv, P0_inv, pr).reshape(-1)
    check("RTS smoother matches the dense MAP",
          float((x_rts - x_dense).abs().max()) < 1e-7,
          f"max dev {float((x_rts - x_dense).abs().max()):.3e}")


# --- 3. each observation once ---------------------------------------------
def test_double_counting_is_detectable() -> None:
    print("[3] counting an image twice changes the posterior")
    once, F, Q_inv, P0_inv, pr = build_linear_problem(seed=3)
    twice, *_ = build_linear_problem(seed=3, duplicate_frame=2)

    A1, b1 = dense_reference(once, F, Q_inv, P0_inv, pr)
    A2, b2 = dense_reference(twice, F, Q_inv, P0_inv, pr)
    x1 = torch.linalg.solve(A1, b1)
    x2 = torch.linalg.solve(A2, b2)
    gap = float((x1 - x2).abs().max())
    check("a duplicated factor moves the solution", gap > 1e-6,
          f"max dev {gap:.3e} -- if this were zero the 'use each image once' "
          "rule could be violated without any test noticing")
    check("the duplicate raises information at that frame",
          float(A2.diagonal().sum()) > float(A1.diagonal().sum()))


# --- 4. the preconditioner is not the posterior ---------------------------
def test_preconditioner_is_labelled() -> None:
    print("[4] block-Jacobi is a preconditioner, not the covariance")
    op, F, Q_inv, P0_inv, pr = build_linear_problem(T=4, N=2, seed=7)
    A = op.dense_information()
    exact_cov = torch.linalg.inv(A)
    apply = block_jacobi_preconditioner(op)

    probe = op.zeros()
    probe[0, 0, 0] = 1.0
    approx = apply(probe)[0, 0, 0]
    exact = exact_cov[0, 0]
    check("block-Jacobi differs from the exact posterior covariance",
          abs(float(approx) - float(exact)) > 1e-6,
          f"approx {float(approx):.6f} vs exact {float(exact):.6f} -- they "
          "must not be confused, and a test that let them coincide would "
          "licence reporting one as the other")
    check("it still accelerates PCG",
          pcg(op, op.rhs(), preconditioner=apply, max_iters=500, tol=1e-12
              )[1]["iterations"]
          <= pcg(op, op.rhs(), max_iters=500, tol=1e-12)[1]["iterations"])


# --- 5. shape and frame-range contracts -----------------------------------
def test_contracts() -> None:
    print("[5] refusals")
    op, *_ = build_linear_problem(T=3, N=2)
    for bad, why in ((torch.zeros(2, 2, 12), "wrong frame count"),
                     (torch.zeros(3, 1, 12), "wrong gaussian count"),
                     (torch.zeros(3, 2, 6), "wrong state dim")):
        global CHECKS
        CHECKS += 1
        try:
            op.information_matvec(bad)
            FAILURES.append(f"{why} not refused")
            print(f"  FAIL  {why} not refused")
        except ValueError:
            pass

    CHECKS += 1
    try:
        FullWindowFreeOperator(num_frames=3, num_gaussians=1,
                               process_factors=[ProcessFactor(
                                   frame=5, A=torch.zeros(1, 12, 12),
                                   B=torch.zeros(1, 12, 12),
                                   Q_inv=torch.zeros(1, 12, 12),
                                   residual=torch.zeros(1, 12))],
                               measurement_factors=[])
        FAILURES.append("out-of-range process factor not refused")
        print("  FAIL  out-of-range process factor not refused")
    except ValueError:
        pass


def main() -> int:
    test_operator_matches_dense()
    test_three_solvers_agree()
    test_double_counting_is_detectable()
    test_preconditioner_is_labelled()
    test_contracts()
    print()
    if FAILURES:
        print(f"{len(FAILURES)} of {CHECKS} checks FAILED")
        for failure in FAILURES:
            print(f"  - {failure}")
        return 1
    print(f"{CHECKS} checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
