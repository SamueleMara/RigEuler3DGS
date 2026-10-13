"""Exact nonlinear process residual and BOTH of its Jacobians.

Section 6 of CODEX_IMPLEMENT_EXACT_UNIFIED_PHASE1_IEKF_IEKS.md.

The shortcut `K_t = [-F_t, I]` is correct only where the process residual is
zero and the current and next tangent bases coincide. With a nonzero
orientation innovation it is wrong, and wrong in a way that still converges to
something -- which is why it has to be derived and finite-difference tested
rather than assumed.

With the repository's left/spatial convention `R+ = Exp(hat(dphi)) R`:

    R_p = Exp(hat(dt * omega_t)) R_t          predicted rotation
    E   = R_{t+1} R_p^-1                      relative rotation
    e_R = Log(E)^v                            orientation residual

A left perturbation at the next state and at the prediction gives

    E' = Exp(hat(dphi_{t+1})) E Exp(-hat(dphi_p))

so to first order

    de_R = J_l^-1(e_R) dphi_{t+1} - J_r^-1(e_R) dphi_p

and from section 3, dphi_p = A_t dphi_t + J_l(dt omega_t) dt domega_t.

Therefore the rotation row of A_proc holds -J_r^-1(e_R) A_t and
-J_r^-1(e_R) J_l(dt omega_t) dt, and the next-state rotation block of B_proc
is J_l^-1(e_R). At e_R = 0 both inverse Jacobians are the identity and the
familiar [-F, I] limit is recovered -- which this module asserts rather than
assumes.

The translation and velocity rows are linear, so they equal -F and I exactly
in every case.
"""

from __future__ import annotations

import torch

from estimator.manifold import (
    so3_exp,
    so3_left_jacobian,
    so3_left_jacobian_inverse,
    so3_log,
)
from estimator.process import error_state_transition
from estimator.state import (
    DELTA_SLICE_MU,
    DELTA_SLICE_OMEGA,
    DELTA_SLICE_PHI,
    DELTA_SLICE_V,
    GaussianMotionState,
)

ERROR_DIM = 12


def so3_right_jacobian_inverse(phi: torch.Tensor) -> torch.Tensor:
    """J_r^-1(phi) = J_l^-1(-phi). Derived from the repository's J_l^-1 so the
    two share one small-angle series and cannot disagree near zero."""
    return so3_left_jacobian_inverse(-torch.as_tensor(phi).double())


def predicted_rotation(state: GaussianMotionState, dt) -> tuple:
    """R_p = Exp(hat(dt omega)) R, and the increment dt*omega it used."""
    omega = state.omega.double()
    step = _broadcast_dt(dt, omega.shape[0], omega.device)
    increment = omega * step.unsqueeze(-1)
    return so3_exp(increment) @ state.R.double(), increment


def _broadcast_dt(dt, n: int, device) -> torch.Tensor:
    tensor = torch.as_tensor(dt, dtype=torch.float64, device=device)
    if tensor.ndim == 0:
        tensor = tensor.expand(n)
    if tensor.shape[0] != n:
        raise ValueError(f"dt has {tensor.shape[0]} entries for {n} Gaussians")
    if bool((tensor <= 0).any()):
        raise ValueError("dt must be positive in physical chronological time")
    return tensor


def process_residual(current: GaussianMotionState, nxt: GaussianMotionState,
                     dt) -> torch.Tensor:
    """e_t = X_{t+1} boxminus f_t(X_t), as [N,12].

    The orientation block is Log(R_{t+1} R_p^-1), NOT a matrix subtraction.
    """
    n = current.mu.shape[0]
    if nxt.mu.shape[0] != n:
        raise ValueError(f"{n} Gaussians at t but {nxt.mu.shape[0]} at t+1")
    step = _broadcast_dt(dt, n, current.mu.device)

    R_p, _ = predicted_rotation(current, dt)
    mu_p = current.mu.double() + current.v.double() * step.unsqueeze(-1)

    residual = torch.zeros(n, ERROR_DIM, dtype=torch.float64,
                           device=current.mu.device)
    residual[:, DELTA_SLICE_MU] = nxt.mu.double() - mu_p
    residual[:, DELTA_SLICE_PHI] = so3_log(
        nxt.R.double() @ R_p.transpose(-1, -2))
    residual[:, DELTA_SLICE_V] = nxt.v.double() - current.v.double()
    residual[:, DELTA_SLICE_OMEGA] = nxt.omega.double() - current.omega.double()
    return residual


def process_residual_jacobians(current: GaussianMotionState,
                               nxt: GaussianMotionState, dt,
                               *, mode: str = "so3") -> tuple:
    """(A_proc, B_proc), each [N,12,12], of the residual above.

    A_proc = d e_t / d delta x_t,  B_proc = d e_t / d delta x_{t+1}.
    """
    n = current.mu.shape[0]
    step = _broadcast_dt(dt, n, current.mu.device)
    F = error_state_transition(current, dt, mode=mode).double()
    e = process_residual(current, nxt, dt)
    e_R = e[:, DELTA_SLICE_PHI]

    device = current.mu.device
    A = torch.zeros(n, ERROR_DIM, ERROR_DIM, dtype=torch.float64, device=device)
    B = torch.zeros(n, ERROR_DIM, ERROR_DIM, dtype=torch.float64, device=device)
    eye3 = torch.eye(3, dtype=torch.float64, device=device).expand(n, 3, 3)

    # Euclidean rows: the residual is linear, so these are exactly -F and I.
    for block in (DELTA_SLICE_MU, DELTA_SLICE_V, DELTA_SLICE_OMEGA):
        A[:, block, :] = -F[:, block, :]
        B[:, block, block] = eye3

    # Rotation row: the two inverse Jacobians of the relative rotation.
    Jl_inv = so3_left_jacobian_inverse(e_R)
    Jr_inv = so3_right_jacobian_inverse(e_R)
    A_t = F[:, DELTA_SLICE_PHI, DELTA_SLICE_PHI]                 # Exp(dt omega)
    Jl_omega = so3_left_jacobian(current.omega.double() * step.unsqueeze(-1))

    A[:, DELTA_SLICE_PHI, DELTA_SLICE_PHI] = -Jr_inv @ A_t
    A[:, DELTA_SLICE_PHI, DELTA_SLICE_OMEGA] = (
        -Jr_inv @ Jl_omega * step.view(n, 1, 1))
    B[:, DELTA_SLICE_PHI, DELTA_SLICE_PHI] = Jl_inv
    return A, B


def zero_innovation_reduces_to_minus_F(current: GaussianMotionState, dt,
                                       *, mode: str = "so3",
                                       tol: float = 1e-10) -> dict:
    """At a zero-residual nominal, A_proc = -F and B_proc = I. Asserted, not assumed."""
    from estimator.process import process_predict

    predicted = process_predict(current, dt)
    A, B = process_residual_jacobians(current, predicted, dt, mode=mode)
    F = error_state_transition(current, dt, mode=mode).double()
    eye = torch.eye(ERROR_DIM, dtype=torch.float64,
                    device=B.device).expand_as(B)
    dA = float((A + F).abs().max())
    dB = float((B - eye).abs().max())
    return {"max_abs_A_plus_F": dA, "max_abs_B_minus_I": dB,
            "passes": dA < tol and dB < tol}


def process_jacobians_finite_difference(current: GaussianMotionState,
                                        nxt: GaussianMotionState, dt,
                                        *, eps: float = 1e-6) -> dict:
    """Central differences of the real residual against the analytic Jacobians.

    The point of the test is the NONZERO-innovation case: at zero innovation
    any implementation that returns [-F, I] passes, including a wrong one.
    """
    from estimator.state import boxplus

    A, B = process_residual_jacobians(current, nxt, dt)
    n = current.mu.shape[0]
    errA = torch.zeros_like(A)
    errB = torch.zeros_like(B)

    for j in range(ERROR_DIM):
        d = torch.zeros(n, ERROR_DIM, dtype=current.mu.dtype, device=current.mu.device)
        d[:, j] = eps
        colA = (process_residual(boxplus(current, d), nxt, dt)
                - process_residual(boxplus(current, -d), nxt, dt)) / (2 * eps)
        colB = (process_residual(current, boxplus(nxt, d), dt)
                - process_residual(current, boxplus(nxt, -d), dt)) / (2 * eps)
        errA[:, :, j] = colA - A[:, :, j]
        errB[:, :, j] = colB - B[:, :, j]

    return {"max_abs_error_A": float(errA.abs().max()),
            "max_abs_error_B": float(errB.abs().max()),
            "rotation_block_error_A": float(
                errA[:, DELTA_SLICE_PHI, :].abs().max()),
            "rotation_block_error_B": float(
                errB[:, DELTA_SLICE_PHI, :].abs().max())}
