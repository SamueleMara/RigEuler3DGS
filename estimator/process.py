"""Discrete constant-twist process model for the per-Gaussian error-state IEKF.

Implements specification section 5:

    mu^-_{k+1}    = mu^+_k + v^+_k * dt
    R^-_{k+1}     = Exp([omega^+_k]_x * dt) R^+_k
    v^-_{k+1}     = v^+_k
    omega^-_{k+1} = omega^+_k

together with the 12x12 error-state transition matrix F in the two mandated
modes ('first_order' baseline and the 'so3' refinement).

The error state ordering is fixed by specification section 4:

    delta = [ delta_mu (0:3) | delta_phi (3:6) | delta_v (6:9) | delta_omega (9:12) ]

and the retraction is a LEFT perturbation of the rotation,
``R^+ = Exp(delta_phi) R^-``; every Jacobian below is derived under that
convention, so swapping to a right perturbation would silently invalidate F.

No autograd, no optimizer state: everything here is closed form (spec 0.4/0.5).
"""

from __future__ import annotations

import copy
import dataclasses
from typing import Tuple, Union

import torch

from .manifold import so3_exp, so3_hat, so3_left_jacobian
from .state import GaussianMotionState

FIRST_ORDER = "first_order"
SO3 = "so3"
TRANSITION_MODES = (FIRST_ORDER, SO3)

ERROR_DIM = 12
POSE_SLICE = slice(0, 6)
MU_SLICE = slice(0, 3)
PHI_SLICE = slice(3, 6)
V_SLICE = slice(6, 9)
OMEGA_SLICE = slice(9, 12)

DtLike = Union[float, int, torch.Tensor]


def _as_step(dt: DtLike, n: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    """Broadcast a real frame spacing to ``[n, 1]``.

    ``dt`` is seconds of real video time (spec rule 11): a scalar per frame, or
    one value per Gaussian when a caller integrates a ragged batch.  Unit frame
    spacing is never assumed, so no default is provided.
    """
    if isinstance(dt, torch.Tensor):
        step = dt.to(device=device, dtype=dtype).reshape(-1)
        if step.numel() == 1:
            step = step.expand(n)
        elif step.numel() != n:
            raise ValueError(f"dt must be scalar or have {n} entries, got {tuple(dt.shape)}")
    else:
        step = torch.full((n,), float(dt), device=device, dtype=dtype)
    return step.unsqueeze(-1)


def _rebuild_state(
    reference: GaussianMotionState,
    mu: torch.Tensor,
    R: torch.Tensor,
    v: torch.Tensor,
    omega: torch.Tensor,
) -> GaussianMotionState:
    """Produce a new state of the same concrete type as ``reference``.

    The state container is owned by ``estimator/state.py``; it may be a
    dataclass, a NamedTuple or a plain class carrying extra bookkeeping fields
    (ids, timestamps).  Rebuilding through the container's own replace/ctor
    protocol preserves those extra fields instead of dropping them.
    """
    if hasattr(reference, "_replace"):  # NamedTuple
        return reference._replace(mu=mu, R=R, v=v, omega=omega)
    if dataclasses.is_dataclass(reference) and not isinstance(reference, type):
        return dataclasses.replace(reference, mu=mu, R=R, v=v, omega=omega)
    try:
        return type(reference)(mu=mu, R=R, v=v, omega=omega)
    except TypeError:
        clone = copy.copy(reference)
        clone.mu = mu
        clone.R = R
        clone.v = v
        clone.omega = omega
        return clone


def angular_increment(
    omega: torch.Tensor, dt: DtLike
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return ``(Omega, [Omega]_x, Exp([Omega]_x))`` for ``Omega = omega * dt``.

    The generator ``[Omega]_x`` is returned alongside the group element because
    the downstream linearization cache (spec 9.3, ``dSigma = [dphi]_x Sigma -
    Sigma [dphi]_x``) and the rotational diagnostics both need the skew form,
    and recomputing it per call is pure waste at N ~ 1e6.
    """
    if omega.dim() != 2 or omega.shape[-1] != 3:
        raise ValueError(f"omega must be [N, 3], got {tuple(omega.shape)}")
    step = _as_step(dt, omega.shape[0], omega.device, omega.dtype)
    Omega = omega * step
    return Omega, so3_hat(Omega), so3_exp(Omega)


def process_predict(state: GaussianMotionState, dt: DtLike) -> GaussianMotionState:
    """Propagate the nominal state one frame forward under a constant twist.

    Exactly spec section 5: linear part is Euclidean integration, rotational
    part is a LEFT group composition so that it commutes with the left
    retraction used by ``boxplus``.  Velocities are carried through unchanged;
    unmodelled acceleration is absorbed by the process noise ``Q`` instead.
    """
    mu = state.mu
    if mu.dim() != 2 or mu.shape[-1] != 3:
        raise ValueError(f"state.mu must be [N, 3], got {tuple(mu.shape)}")
    step = _as_step(dt, mu.shape[0], mu.device, mu.dtype)

    mu_pred = mu + state.v * step
    _, _, exp_omega = angular_increment(state.omega, dt)
    R_pred = exp_omega @ state.R

    # Clone the carried-through velocities: the predicted state must not alias
    # the posterior it was built from, otherwise an in-place IEKF update on one
    # silently corrupts the other.
    return _rebuild_state(state, mu_pred, R_pred, state.v.clone(), state.omega.clone())


def error_state_transition(
    state: GaussianMotionState, dt: DtLike, mode: str = SO3
) -> torch.Tensor:
    """Per-Gaussian 12x12 error-state transition ``F`` with ``delta^- = F delta^+``.

    ``mode='first_order'`` is the spec section 5 baseline

        F = [[I, 0, dt I, 0], [0, I, 0, dt I], [0, 0, I, 0], [0, 0, 0, I]].

    ``mode='so3'`` replaces the rotational row by the exact first-order
    propagation of a left rotational error through the group composition
    ``R^- = Exp(Omega) R``:

        dphi_{k+1} = Ad_{Exp(Omega)} dphi_k + J_l(Omega) dt domega_k,
        Omega = omega dt.

    For SO(3) the adjoint of a group element IS the rotation matrix itself,
    which is why ``Exp(Omega)`` appears directly in the (dphi, dphi) block.
    As dt -> 0 both ``Exp(Omega) -> I`` and ``J_l(Omega) -> I``, so 'so3'
    degenerates to 'first_order'.  The two modes converge at different rates,
    which is worth knowing when choosing one: the velocity-coupling block
    agrees to O(dt^2) (``dt J_l(Omega) - dt I = O(dt^2)``) while the adjoint
    block only agrees to O(dt) (``Exp(Omega) - I = [Omega]_x + O(dt^2)``).
    The baseline therefore drops the frame rotation of the attitude error
    entirely; at the per-step rotations seen in articulated motion that term is
    the dominant error in F, which is exactly why 'so3' is the default here.

    The translational row is exact in both modes (the position recursion is
    linear in mu and v), so only the rotational row differs.
    """
    if mode not in TRANSITION_MODES:
        raise ValueError(f"mode must be one of {TRANSITION_MODES}, got {mode!r}")

    mu = state.mu
    n = mu.shape[0]
    device, dtype = mu.device, mu.dtype
    step = _as_step(dt, n, device, dtype).unsqueeze(-1)  # [N, 1, 1] for block scaling

    eye3 = torch.eye(3, device=device, dtype=dtype).expand(n, 3, 3)
    F = torch.zeros(n, ERROR_DIM, ERROR_DIM, device=device, dtype=dtype)

    F[:, MU_SLICE, MU_SLICE] = eye3
    F[:, MU_SLICE, V_SLICE] = step * eye3
    F[:, V_SLICE, V_SLICE] = eye3
    F[:, OMEGA_SLICE, OMEGA_SLICE] = eye3

    if mode == FIRST_ORDER:
        F[:, PHI_SLICE, PHI_SLICE] = eye3
        F[:, PHI_SLICE, OMEGA_SLICE] = step * eye3
    else:
        Omega, _, exp_omega = angular_increment(state.omega, dt)
        F[:, PHI_SLICE, PHI_SLICE] = exp_omega
        F[:, PHI_SLICE, OMEGA_SLICE] = step * so3_left_jacobian(Omega)

    return F
