"""One full-window posterior: matrix-free block-tridiagonal normal equations.

Section 7 of CODEX_IMPLEMENT_EXACT_UNIFIED_PHASE1_IEKF_IEKS.md.

The objective is ONE physical-time Markov posterior over the whole window,

    F(X) = 1/2 ||X_0 boxminus Xbar_0||^2_{P0^-1}
         + sum_t 1/2 ||e_t(X_t, X_{t+1})||^2_{Q_t^-1}
         + sum_{t,c} 1/2 ||y_{t,c} - h_c(X_t)||^2_{R^-1},

linearised at the current nominal into

    A u = I_prior u
        + sum_{t,c} H^T R^-1 (H u_t)
        + sum_t K_t^T Q_t^-1 (K_t u_{t:t+1})
        + lambda D u.

Two things this file exists to prevent.

**Solving frames independently and calling it smoothing.** The process factor
couples t and t+1, so K_t^T Q^-1 K_t writes into four blocks, two of them
off-diagonal. Dropping the off-diagonals leaves a per-frame filter whose
output looks like a trajectory. `information_matvec` always applies all four,
and `dense_information` (TEST ONLY) exposes them so a test can assert they
are nonzero.

**Calling a preconditioner the posterior.** A block-diagonal approximation is
offered explicitly as `block_jacobi_preconditioner`, named as such. It is
never substituted for the operator.

No measurement model is hard-coded here. A factor supplies its own matrix-free
`jvp`/`vjp` and inverse noise, so the CUDA RGB-D adapter and a synthetic
linear factor satisfy the same contract and the exact-MAP test can run on CPU.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

ERROR_DIM = 12


@dataclass
class ProcessFactor:
    """One temporal factor coupling frames t and t+1."""

    frame: int                 # t; couples t -> t+1
    A: torch.Tensor            # [N,12,12] d e_t / d delta x_t
    B: torch.Tensor            # [N,12,12] d e_t / d delta x_{t+1}
    Q_inv: torch.Tensor        # [N,12,12] inverse process covariance
    residual: torch.Tensor     # [N,12] e_t at the nominal


class MeasurementFactor:
    """Matrix-free image factor at one (frame, camera).

    `jvp(u_t) -> z` applies H to a [N,12] tangent and returns the stacked
    residual-space vector; `vjp(z) -> [N,12]` applies H^T. `R_inv_apply(z)`
    applies the inverse measurement covariance. The adjoint identity
    <H p, z> = <p, H^T z> is a required test, not an assumption.
    """

    frame: int

    def jvp(self, u_t: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def vjp(self, z: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def R_inv_apply(self, z: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def residual(self) -> torch.Tensor:
        raise NotImplementedError


class DenseLinearMeasurement(MeasurementFactor):
    """A fully explicit linear factor, for the exact-MAP reference tests."""

    def __init__(self, frame: int, H: torch.Tensor, R_inv: torch.Tensor,
                 r: torch.Tensor):
        self.frame = int(frame)
        self.H = H.double()            # [m, N*12]
        self.R_inv = R_inv.double()    # [m, m]
        self.r = r.double()            # [m]

    def jvp(self, u_t):
        return self.H @ u_t.reshape(-1).double()

    def vjp(self, z):
        n = self.H.shape[1] // ERROR_DIM
        return (self.H.transpose(0, 1) @ z.double()).reshape(n, ERROR_DIM)

    def R_inv_apply(self, z):
        return self.R_inv @ z.double()

    def residual(self):
        return self.r


class FullWindowFreeOperator:
    """The information operator of the single full-window posterior."""

    def __init__(self, num_frames: int, num_gaussians: int,
                 process_factors, measurement_factors,
                 prior_information: torch.Tensor | None = None,
                 prior_residual: torch.Tensor | None = None,
                 damping: float = 0.0):
        self.T = int(num_frames)
        self.N = int(num_gaussians)
        self.process = list(process_factors)
        self.measurements = list(measurement_factors)
        self.prior_information = prior_information  # [N,12,12] on frame 0
        self.prior_residual = prior_residual        # [N,12]
        self.damping = float(damping)
        for factor in self.process:
            if not 0 <= factor.frame < self.T - 1:
                raise ValueError(
                    f"process factor at frame {factor.frame} outside "
                    f"[0,{self.T - 1}); it couples t to t+1")
        for factor in self.measurements:
            if not 0 <= factor.frame < self.T:
                raise ValueError(f"measurement at frame {factor.frame} outside [0,{self.T})")

    # -- shape helpers ----------------------------------------------------
    def zeros(self) -> torch.Tensor:
        return torch.zeros(self.T, self.N, ERROR_DIM, dtype=torch.float64)

    def _check(self, u: torch.Tensor) -> torch.Tensor:
        u = torch.as_tensor(u).double()
        if u.shape != (self.T, self.N, ERROR_DIM):
            raise ValueError(
                f"trajectory tangent must be [{self.T},{self.N},{ERROR_DIM}], "
                f"got {tuple(u.shape)}")
        return u

    # -- the operator -----------------------------------------------------
    def information_matvec(self, u: torch.Tensor) -> torch.Tensor:
        """A u, with every temporal off-diagonal applied."""
        u = self._check(u)
        out = torch.zeros_like(u)

        if self.prior_information is not None:
            out[0] += torch.einsum("nij,nj->ni", self.prior_information, u[0])

        for factor in self.measurements:
            t = factor.frame
            out[t] += factor.vjp(factor.R_inv_apply(factor.jvp(u[t])))

        for factor in self.process:
            t = factor.frame
            # K u = A u_t + B u_{t+1}; then K^T Q^-1 (K u) lands in BOTH blocks.
            Ku = (torch.einsum("nij,nj->ni", factor.A, u[t])
                  + torch.einsum("nij,nj->ni", factor.B, u[t + 1]))
            QKu = torch.einsum("nij,nj->ni", factor.Q_inv, Ku)
            out[t] += torch.einsum("nji,nj->ni", factor.A, QKu)
            out[t + 1] += torch.einsum("nji,nj->ni", factor.B, QKu)

        if self.damping:
            out += self.damping * u
        return out

    def rhs(self) -> torch.Tensor:
        """-gradient: measurement innovations and process residual gradients."""
        b = self.zeros()
        if self.prior_information is not None and self.prior_residual is not None:
            b[0] -= torch.einsum("nij,nj->ni", self.prior_information,
                                 self.prior_residual.double())
        for factor in self.measurements:
            t = factor.frame
            b[t] += factor.vjp(factor.R_inv_apply(factor.residual()))
        for factor in self.process:
            t = factor.frame
            Qe = torch.einsum("nij,nj->ni", factor.Q_inv, factor.residual.double())
            b[t] -= torch.einsum("nji,nj->ni", factor.A, Qe)
            b[t + 1] -= torch.einsum("nji,nj->ni", factor.B, Qe)
        return b

    # -- reference constructions, TEST ONLY -------------------------------
    def dense_information(self) -> torch.Tensor:
        """The full matrix, built column by column through the operator.

        For tests and small problems only. Section 7 forbids materialising
        this at scale; it exists so the matrix-free path can be checked
        against an independently assembled reference.
        """
        size = self.T * self.N * ERROR_DIM
        if size > 4096:
            raise ValueError(
                f"dense_information would be {size}x{size}; it is a test "
                "reference, not a solver path")
        columns = []
        for k in range(size):
            basis = self.zeros().reshape(-1)
            basis[k] = 1.0
            columns.append(self.information_matvec(basis.reshape(
                self.T, self.N, ERROR_DIM)).reshape(-1))
        return torch.stack(columns, dim=1)

    def block_tridiagonal(self):
        """(diag, upper) blocks per frame. Used by the exact Schur solve."""
        size = self.N * ERROR_DIM
        diag = [torch.zeros(size, size, dtype=torch.float64) for _ in range(self.T)]
        upper = [torch.zeros(size, size, dtype=torch.float64)
                 for _ in range(max(self.T - 1, 0))]

        def block_diag_from(per_gauss):
            out = torch.zeros(size, size, dtype=torch.float64)
            for i in range(self.N):
                s = slice(i * ERROR_DIM, (i + 1) * ERROR_DIM)
                out[s, s] = per_gauss[i]
            return out

        if self.prior_information is not None:
            diag[0] += block_diag_from(self.prior_information)
        for factor in self.measurements:
            H = factor.H if isinstance(factor, DenseLinearMeasurement) else None
            if H is None:
                raise TypeError("block_tridiagonal needs explicit measurement H")
            diag[factor.frame] += H.transpose(0, 1) @ factor.R_inv @ H
        for factor in self.process:
            t = factor.frame
            A = block_diag_from(factor.A)
            B = block_diag_from(factor.B)
            Qi = block_diag_from(factor.Q_inv)
            diag[t] += A.transpose(0, 1) @ Qi @ A
            diag[t + 1] += B.transpose(0, 1) @ Qi @ B
            upper[t] += A.transpose(0, 1) @ Qi @ B
        if self.damping:
            for t in range(self.T):
                diag[t] += self.damping * torch.eye(size, dtype=torch.float64)
        return diag, upper


def schur_forward_backward(diag, upper, rhs_blocks):
    """Exact block-tridiagonal solve: forward elimination, back substitution.

        D_0 = A_00,  d_0 = b_0
        D_{t+1} = A_{t+1,t+1} - A_{t+1,t} D_t^-1 A_{t,t+1}
        d_{t+1} = b_{t+1}     - A_{t+1,t} D_t^-1 d_t
        x_T = D_T^-1 d_T,  x_t = D_t^-1 (d_t - A_{t,t+1} x_{t+1})
    """
    T = len(diag)
    D = [None] * T
    d = [None] * T
    D[0] = diag[0].clone()
    d[0] = rhs_blocks[0].clone()
    for t in range(T - 1):
        lower = upper[t].transpose(0, 1)
        solved = torch.linalg.solve(D[t], torch.cat(
            [upper[t], d[t].unsqueeze(-1)], dim=1))
        Dinv_U = solved[:, :-1]
        Dinv_d = solved[:, -1]
        D[t + 1] = diag[t + 1] - lower @ Dinv_U
        d[t + 1] = rhs_blocks[t + 1] - lower @ Dinv_d
    x = [None] * T
    x[T - 1] = torch.linalg.solve(D[T - 1], d[T - 1])
    for t in range(T - 2, -1, -1):
        x[t] = torch.linalg.solve(D[t], d[t] - upper[t] @ x[t + 1])
    return x


def block_jacobi_preconditioner(operator: FullWindowFreeOperator):
    """Per-frame, per-Gaussian inverse of the diagonal blocks.

    A PRECONDITIONER. It discards the temporal off-diagonals and the
    cross-Gaussian pixel coupling, so it is not the posterior covariance and
    must never be reported as one.
    """
    size = ERROR_DIM
    blocks = torch.zeros(operator.T, operator.N, size, size, dtype=torch.float64)
    eye = torch.eye(size, dtype=torch.float64)
    for t in range(operator.T):
        blocks[t] = eye
    if operator.prior_information is not None:
        blocks[0] += operator.prior_information
    for factor in operator.process:
        t = factor.frame
        blocks[t] += torch.einsum("nji,njk,nkl->nil", factor.A, factor.Q_inv, factor.A)
        blocks[t + 1] += torch.einsum("nji,njk,nkl->nil", factor.B, factor.Q_inv, factor.B)
    inverse = torch.linalg.inv(blocks)

    def apply(u):
        return torch.einsum("tnij,tnj->tni", inverse, u)

    return apply


def pcg(operator: FullWindowFreeOperator, b, preconditioner=None,
        max_iters: int = 200, tol: float = 1e-10):
    """Matrix-free preconditioned conjugate gradients on the full window."""
    x = operator.zeros()
    r = b.clone()
    z = preconditioner(r) if preconditioner else r.clone()
    p = z.clone()
    rz = float((r * z).sum())
    b_norm = float(b.norm()) or 1.0
    history = []
    for iteration in range(max_iters):
        Ap = operator.information_matvec(p)
        pAp = float((p * Ap).sum())
        if pAp <= 0:
            return x, {"iterations": iteration, "status": "NON_POSITIVE_CURVATURE",
                       "residual": float(r.norm()) / b_norm, "history": history}
        alpha = rz / pAp
        x = x + alpha * p
        r = r - alpha * Ap
        rel = float(r.norm()) / b_norm
        history.append(rel)
        if rel < tol:
            return x, {"iterations": iteration + 1, "status": "CONVERGED",
                       "residual": rel, "history": history}
        z = preconditioner(r) if preconditioner else r
        rz_new = float((r * z).sum())
        p = z + (rz_new / rz) * p
        rz = rz_new
    return x, {"iterations": max_iters, "status": "MAX_ITERS",
               "residual": float(r.norm()) / b_norm, "history": history}
