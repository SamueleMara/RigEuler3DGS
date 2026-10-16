"""Sections 15, 16: the Phase-2 process model and manifold covariance transport.

CODEX_IMPLEMENT_PHASE2_RIGID_SCREW_IEKF.md.

    q_{t+1}^-    = q_t^+ + dt_t qdot_t^+
    qdot_{t+1}^- = qdot_t^+ + w
    zeta_{t+1}^- = zeta_t^+          (the geometry is static)

A kinematic Phase 2: no gravity, torque, mass, inertia or damping (section 15),
and the real `dt_t` rather than a unit frame step.

Section 16 is the part that is easy to get wrong. The geometry error state
lives in a LOCAL basis `B_zeta` that changes whenever the nominal geometry is
retracted. Updating `zeta`, recomputing `B_zeta` and keeping the old 4x4 block
unchanged silently rotates the uncertainty relative to the state it describes,
and destroys the cross-covariance that makes `qdot` observable at all --
because nothing in a single image measures `qdot` (section 15, `H_qdot = 0`),
it is corrected ONLY through that coupling.
"""

from __future__ import annotations

import torch
from torch import Tensor

from .joint_manifold import retract_joint_geometry
from .joint_state import GEOMETRY_DIM, JOINT_ERROR_DIM


def _stride(state) -> int:
    """The per-part error dimension.

    Six for the generic screw, but Stage B's physical models are narrower (four
    for M_P, six for M_R), so the stride is asked of the state rather than
    assumed.  The process model itself is identical for all three: the first
    two coordinates are always (displacement, velocity) and everything after
    them is static geometry.
    """
    return int(getattr(state, "error_dim", JOINT_ERROR_DIM))


def process_matrix(state, dt: float) -> Tensor:
    """F for one step, block diagonal over parts.

    Per part, in `[dq, dqdot, dtheta_J]` order:

        [[1, dt, 0],
         [0,  1, 0],
         [0,  0, I_4]]

    The `dt` in the corner is the entire coupling between position and
    velocity, and it is what carries image information about `q` back onto
    `qdot` over a window.
    """
    n = state.dim
    F = torch.eye(n, dtype=torch.float64)
    d = _stride(state)
    for k in range(state.num_parts):
        i = k * d
        F[i, i + 1] = float(dt)
    return F


def process_noise(state, dt: float, sigma_qddot: float = 1.0) -> Tensor:
    """Q for one step: a white-acceleration model on qdot only.

    The geometry is static by construction (section 15), so its block is
    exactly zero -- not a small number. Letting the geometry random-walk would
    quietly turn the one constant screw into a per-frame one and undo
    section 8.
    """
    n = state.dim
    Q = torch.zeros((n, n), dtype=torch.float64)
    s2 = float(sigma_qddot) ** 2
    dt = float(dt)
    d = _stride(state)
    for k in range(state.num_parts):
        i = k * d
        # continuous white-acceleration, integrated once
        Q[i, i] = s2 * dt ** 3 / 3.0
        Q[i, i + 1] = Q[i + 1, i] = s2 * dt ** 2 / 2.0
        Q[i + 1, i + 1] = s2 * dt
    return Q


def predict(state, frame: int, dt: float, sigma_qddot: float = 1.0):
    """Section 15: advance the nominal and its covariance by one frame."""
    F = process_matrix(state, dt)
    Q = process_noise(state, dt, sigma_qddot)
    for p in state.parts:
        if frame + 1 < p.num_frames:
            p.q[frame + 1] = p.q[frame] + dt * p.qdot[frame]
            p.qdot[frame + 1] = p.qdot[frame]
    state.covariance = F @ state.covariance @ F.T + Q
    return state


def reset_transport(state, deltas) -> Tensor:
    """Section 16: G_reset for an accepted retraction of every part.

    Only the geometry block moves: `q` and `qdot` are ordinary scalars whose
    coordinates do not depend on the basis. The returned matrix is applied as
    `P_new = G P G^T`, which is the repository's existing error-state reset
    convention.
    """
    n = state.dim
    G = torch.eye(n, dtype=torch.float64)
    for k, p in enumerate(state.parts):
        d = torch.as_tensor(deltas[k]).double().reshape(-1)
        _z, Gk = retract_joint_geometry(p.zeta, d[2:6])
        i = k * JOINT_ERROR_DIM + 2
        G[i:i + GEOMETRY_DIM, i:i + GEOMETRY_DIM] = Gk
    return G


def apply_correction(state, frame: int, delta: Tensor,
                     transport: bool = True):
    """Retract the whole joint state and carry the covariance with it.

    Returns `(state, G_reset)`. Cross-covariances between `q`, `qdot` and the
    geometry are preserved because `G` is applied to the WHOLE matrix, not
    block by block -- section 16 calls that out as essential.
    """
    d = torch.as_tensor(delta).double().reshape(-1)
    G = torch.eye(state.dim, dtype=torch.float64)
    if hasattr(state, "retract_in_place"):
        # Stage B: the physical model owns its own exact retraction and its
        # own basis-reset Jacobian.  Everything else here -- in particular the
        # covariance transport below -- is shared.
        # A reduced state (e.g. displacement-only over a fixed joint) owns its
        # reset; otherwise the part's full-state reset applies.
        G = (state.reset_jacobian(d) if hasattr(state, "reset_jacobian")
             else state.parts[0].reset_jacobian(d))
        state.retract_in_place(d, frame)
        if transport and state.covariance is not None:
            state.covariance = G @ state.covariance @ G.T
        return state, G
    for k, p in enumerate(state.parts):
        dk = d[state.slice_of(k)]
        p.q[frame] = p.q[frame] + dk[0]
        p.qdot[frame] = p.qdot[frame] + dk[1]
        z_new, Gk = retract_joint_geometry(p.zeta, dk[2:6])
        p.zeta = z_new
        i = k * JOINT_ERROR_DIM + 2
        G[i:i + GEOMETRY_DIM, i:i + GEOMETRY_DIM] = Gk
    if transport and state.covariance is not None:
        state.covariance = G @ state.covariance @ G.T
    return state, G
