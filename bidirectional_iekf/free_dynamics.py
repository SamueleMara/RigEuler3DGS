"""Reverse-time state conversion and verification of the audited transition.

Section 3.1 of CODEX_IMPLEMENT_EXACT_UNIFIED_PHASE1_IEKF_IEKS.md.

The backward free run processes T, T-1, ..., 0 with a POSITIVE dt, so its
stored rates are reverse-time quantities. Reversing the array puts the frames
back in order and leaves the rates pointing the wrong way; negating the array
without reversing leaves them in the wrong place. Both are needed, and they
are different operations on different objects:

    poses, covariances   reorder   (per-frame quantities)
    v, omega             reorder AND negate   (per-step rates)

For a same-frame world-coordinate representation the mapping is

    v_phys = -v_reverse,  omega_phys = -omega_reverse

and, in the perturbation order (mu, phi, v, omega),

    E = diag(I, I, -I, -I),   P_phys = E P_reverse E^T

which holds ONLY when both covariances refer to the corresponding same-time
tangent state. Any re-gauging of the pose additionally needs its adjoint and
an error-state covariance reset; this module does not silently perform one.

This module also verifies rather than replaces the repository's transition.
`estimator/process.py::error_state_transition(mode='so3')` already builds

    F = [[I, 0, dt I, 0],
         [0, Exp(Omega), 0, dt J_l(Omega)],
         [0, 0, I, 0],
         [0, 0, 0, I]]

which is the specification's section 3 F_t exactly, with J_l beside the left
perturbation. The specification warns not to change working code on the
strength of a letter name in the PDF, so the convention is settled here by
central finite differences of the actual retraction instead.
"""

from __future__ import annotations

import numpy as np
import torch

from estimator.manifold import so3_exp, so3_log
from estimator.process import error_state_transition, process_predict
from estimator.state import (
    DELTA_SLICE_MU,
    DELTA_SLICE_OMEGA,
    DELTA_SLICE_PHI,
    DELTA_SLICE_V,
    GaussianMotionState,
    boxplus,
)

# Perturbation-order sign flip taking a reverse-time tangent state to physical
# time. Position and attitude errors are unchanged; the two rate blocks flip.
REVERSE_TO_PHYSICAL_SIGNS = torch.tensor(
    [1.0] * 3 + [1.0] * 3 + [-1.0] * 3 + [-1.0] * 3, dtype=torch.float64)


def reverse_to_physical_state(mu, R, v, omega):
    """Backward-run arrays in PROCESSING order -> physical chronological order.

    Poses are reordered. Rates are reordered AND negated, because a reverse
    walk's velocity is the rate at which the state moves as the processing
    index advances, and the processing index advances backwards through
    physical time.
    """
    mu = torch.as_tensor(mu).double()
    R = torch.as_tensor(R).double()
    v = torch.as_tensor(v).double()
    omega = torch.as_tensor(omega).double()
    if not (mu.shape[0] == R.shape[0] == v.shape[0] == omega.shape[0]):
        raise ValueError(
            f"frame counts disagree: mu {mu.shape[0]}, R {R.shape[0]}, "
            f"v {v.shape[0]}, omega {omega.shape[0]}")
    return (mu.flip(0).contiguous(), R.flip(0).contiguous(),
            (-v).flip(0).contiguous(), (-omega).flip(0).contiguous())


def reverse_to_physical_covariance(P):
    """P_phys = E P_rev E^T with E = diag(I, I, -I, -I), then reordered.

    Valid only when both covariances refer to the corresponding same-time
    tangent state. A pose re-gauge needs its own adjoint transport and is not
    applied here; doing it silently is how a covariance comes to describe a
    different chart than the state it belongs to.
    """
    P = torch.as_tensor(P).double()
    if P.shape[-1] != 12 or P.shape[-2] != 12:
        raise ValueError(f"expected [...,12,12] error covariance, got {tuple(P.shape)}")
    E = torch.diag(REVERSE_TO_PHYSICAL_SIGNS.to(P.dtype)).to(P.device)
    transformed = E @ P @ E.transpose(-1, -2)
    return transformed.flip(0).contiguous() if P.ndim == 3 else transformed


def verify_transition_against_finite_differences(
        state: GaussianMotionState, dt: float, *, eps: float = 1e-6,
        mode: str = "so3") -> dict:
    """Every column of F against central differences of the real retraction.

    The transition is accepted or rejected on the measured derivative of the
    code's own boxplus/predict composition, not on agreement with a formula.
    Column j is

        d/d(eps) [ predict(state boxplus eps e_j) boxminus predict(state) ]

    evaluated by central differences, compared with F[:, :, j].
    """
    F = error_state_transition(state, dt, mode=mode).double()
    n = state.mu.shape[0]
    errors = torch.zeros(n, 12, 12, dtype=torch.float64)

    base = process_predict(state, dt)
    for j in range(12):
        direction = torch.zeros(n, 12, dtype=state.mu.dtype, device=state.mu.device)
        direction[:, j] = eps
        plus = process_predict(boxplus(state, direction), dt)
        minus = process_predict(boxplus(state, -direction), dt)
        column = _tangent_difference(plus, minus, base) / (2.0 * eps)
        errors[:, :, j] = (column.double() - F[:, :, j].double())

    worst = float(errors.abs().max())
    return {
        "mode": mode,
        "dt": float(dt),
        "max_abs_error": worst,
        "max_abs_error_per_block": {
            "mu": float(errors[:, DELTA_SLICE_MU, :].abs().max()),
            "phi": float(errors[:, DELTA_SLICE_PHI, :].abs().max()),
            "v": float(errors[:, DELTA_SLICE_V, :].abs().max()),
            "omega": float(errors[:, DELTA_SLICE_OMEGA, :].abs().max()),
        },
        "F": F,
    }


def _tangent_difference(plus: GaussianMotionState, minus: GaussianMotionState,
                        base: GaussianMotionState) -> torch.Tensor:
    """(plus boxminus base) - (minus boxminus base), in the base's chart.

    The rotational part uses Log of the LEFT-relative rotation, matching the
    repository's boxplus convention R+ = Exp(hat(dphi)) R.
    """
    n = base.mu.shape[0]
    out = torch.zeros(n, 12, dtype=torch.float64)
    out[:, DELTA_SLICE_MU] = (plus.mu - minus.mu).double()
    out[:, DELTA_SLICE_V] = (plus.v - minus.v).double()
    out[:, DELTA_SLICE_OMEGA] = (plus.omega - minus.omega).double()
    base_T = base.R.double().transpose(-1, -2)
    out[:, DELTA_SLICE_PHI] = (
        so3_log(plus.R.double() @ base_T) - so3_log(minus.R.double() @ base_T))
    return out


def physical_timestamps(total_frames: int, dt: float) -> np.ndarray:
    """Strictly increasing chronological timestamps. dt must be positive."""
    if dt <= 0.0:
        raise ValueError(
            f"dt must be positive, got {dt}. The backward pass reverses the "
            "ORDER and keeps dt positive; it never negates time.")
    return np.arange(total_frames, dtype=np.float64) * float(dt)
