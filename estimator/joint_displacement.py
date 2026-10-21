"""Phase 3 with the joint FIXED: estimate only the displacement q_k and its rate.

The joint type and its geometry (prismatic direction e_P; revolute axis e_R
and axis point a_R) come from joint identification and are held exactly
constant. The RGB-D IEKF refines, frame by frame, only

    x_k = (q_k, qdot_k)            q = s [m] (P) or theta [rad] (R)

with the same renderer, measurement model and iterated update as every other
stage. The moving Gaussians are still transported by the full rigid motion of
the fixed joint (mu = mu_bar + s e_P; mu = a + Q(theta)(mu_bar - a),
R = Q(theta) R_bar), so the image Jacobian is the q-column of the joint model's
H^G J_G, i.e. dmu/ds = e_P or dmu/dtheta = e_R x (mu - a), dphi/dtheta = e_R.

This wraps the existing exact P / R model state (phase2_model_comparison):
the geometry columns are zero-padded on the way in and dropped on the way out,
and the retraction touches only q[k], qdot[k], so the geometry objects are
never written.
"""
from __future__ import annotations

import torch
from torch import Tensor

from .joint_process import process_matrix, process_noise

DISPLACEMENT_DIM = 2


class DisplacementOnlyState:
    """(q, qdot) error state over a fixed P or R joint, drivable by run_window."""

    def __init__(self, full_state, prior_scale=(1.0, 1.0)):
        self._full = full_state
        self.parts = full_state.parts
        self.error_dim = DISPLACEMENT_DIM
        self.times = full_state.times
        self.covariance = torch.diag(torch.as_tensor(prior_scale, dtype=torch.float64).reshape(2))

    # -- shape ------------------------------------------------------------
    @property
    def num_parts(self) -> int:
        return 1

    @property
    def dim(self) -> int:
        return DISPLACEMENT_DIM

    @property
    def geometry_slice(self) -> slice:
        return slice(DISPLACEMENT_DIM, DISPLACEMENT_DIM)      # no geometry in the state

    def _pad(self, delta) -> Tensor:
        d = torch.zeros(self._full.error_dim, dtype=torch.float64)
        d[:DISPLACEMENT_DIM] = torch.as_tensor(delta).double().reshape(-1)[:DISPLACEMENT_DIM]
        return d

    # -- measurement map: the q-column of the fixed joint ------------------
    def jvp(self, frame, delta, mu0, R0=None):
        return self._full.jvp(frame, self._pad(delta), mu0, R0)

    def vjp(self, frame, pose_adjoint, mu0, R0=None):
        return self._full.vjp(frame, pose_adjoint, mu0, R0)[:DISPLACEMENT_DIM]

    def gaussians(self, frame, mu0, R0=None):
        return self._full.gaussians(frame, mu0, R0)

    # -- process: constant rate, white acceleration on qdot ----------------
    def predict(self, frame: int, dt: float, sigma_qddot: float = 1.0):
        F = process_matrix(self, dt)
        Q = process_noise(self, dt, sigma_qddot)
        p = self.parts[0]
        if frame + 1 < p.num_frames:
            q, qd = p.q.clone(), p.qdot.clone()
            q[frame + 1] = q[frame] + float(dt) * qd[frame]
            qd[frame + 1] = qd[frame]
            p.q, p.qdot = q, qd
        self.covariance = F @ self.covariance @ F.T + Q

    # -- retraction: displacement only, geometry untouched -----------------
    def retract_in_place(self, delta, frame):
        d = torch.as_tensor(delta).double().reshape(-1)
        p = self.parts[0]
        q, qd = p.q.clone(), p.qdot.clone()
        if frame is None:
            q, qd = q + float(d[0]), qd + float(d[1])
        else:
            q[frame] = q[frame] + float(d[0])
            qd[frame] = qd[frame] + float(d[1])
        p.q, p.qdot = q, qd

    def reset_jacobian(self, delta) -> Tensor:
        """q and qdot are plain scalars: no basis moves, G = I."""
        return torch.eye(DISPLACEMENT_DIM, dtype=torch.float64)

    def prior_error(self, frame: int, prior_state) -> Tensor:
        """e = (q - q_minus, qdot - qdot_minus) at `frame`: exact, the retraction is additive."""
        p, m = self.parts[0], prior_state.parts[0]
        return torch.stack([p.q[frame] - m.q[frame], p.qdot[frame] - m.qdot[frame]]).double()

    def prior_jacobian(self, frame: int, prior_state) -> Tensor:
        """d e(x boxplus delta) / d delta = I for the additive (q, qdot) retraction."""
        return torch.eye(DISPLACEMENT_DIM, dtype=torch.float64)


def geometry_snapshot(part) -> dict:
    """Every geometry tensor of a P or R part, for an exact before/after check."""
    keys = ("direction", "plane") if hasattr(part, "direction") else ("axis", "plane", "rho", "centroid")
    return {k: getattr(part, k).clone() for k in keys}


def geometry_unchanged(before: dict, part) -> bool:
    return all(torch.equal(v, getattr(part, k)) for k, v in before.items())
