"""Sections 4, 7, 10-12: the exact rigid SE(3) Phase-2 state.

CODEX_IMPLEMENT_RIGID_SE3_PHASE2_AND_POSTHOC_JOINT_REDUCTION.md.

The whole moving part shares ONE transform:

    mu_{i,t} = R_t mu_i^0 + p_t          R_{i,t} = R_t R_i^0

so rigidity is exact by construction (section 5) and no pairwise penalty
exists anywhere in this file. Static Gaussians are untouched and still
rendered -- they carry visibility, alpha compositing and occlusion.

WHY THERE IS NO AXIS HERE. The old Phase 2 handed the renderer an axis
direction, an axis point and an angle, and let the image optimiser move them.
A far axis with a small angle looks like a translation in pixels, so that
parameterisation has a degenerate valley the optimiser can slide down --
section 36, and the stapler result. This state has six pose numbers and no
joint geometry at all, so the valley does not exist. The joint is inferred
afterwards, from the recovered finite rotations (sections 29-31).

CONVENTION (section 10). Phase 1's `boxplus` is `R+ = Exp(hat(dphi)) @ R`, a
LEFT/spatial perturbation, and everything here uses the same one:

    T^+ = Exp(delta_tau^) T,      delta_tau = [delta_v; delta_omega].

Error state, 12 numbers (section 7):

    delta_x = [ delta_tau (6) | delta_xi (6) ]

with `delta_xi` the perturbation of the spatial rigid velocity. The image has
NO derivative with respect to velocity (section 11, `H_xi = 0`); velocity is
observable only through the process cross-covariance, which is why the
process model's `dt` coupling block is not optional.
"""

from __future__ import annotations

import torch
from torch import Tensor

from .joint_manifold import se3_exp, se3_left_jacobian
from .manifold import project_to_so3

RIGID_ERROR_DIM = 12
POSE_DIM = 6
TWIST_DIM = 6

# `delta_tau` occupies the first six coordinates, `delta_xi` the last six.
TAU = slice(0, 6)
XI = slice(6, 12)


class RigidPart:
    """One rigid moving part: a pose and a spatial velocity per frame.

    Frame 0 is the canonical gauge (section 8): `R[0] = I`, `p[0] = 0`. That
    is a definition, not an estimate, and `run_rigid_window` never updates it.
    """

    def __init__(self, members, R=None, p=None, v=None, omega=None,
                 num_frames: int | None = None):
        self.members = torch.as_tensor(members).long().reshape(-1).cpu()
        if R is not None:
            R = torch.as_tensor(R).double().reshape(-1, 3, 3).cpu()
            T = R.shape[0]
        elif p is not None:
            p = torch.as_tensor(p).double().reshape(-1, 3).cpu()
            T = p.shape[0]
        else:
            T = int(num_frames or 0)
        if T <= 0:
            raise ValueError("RigidPart needs at least one frame")
        eye = torch.eye(3, dtype=torch.float64).expand(T, 3, 3)
        self.R = eye.clone() if R is None else R
        self.p = (torch.zeros((T, 3), dtype=torch.float64) if p is None
                  else torch.as_tensor(p).double().reshape(T, 3).cpu())
        self.v = (torch.zeros((T, 3), dtype=torch.float64) if v is None
                  else torch.as_tensor(v).double().reshape(T, 3).cpu())
        self.omega = (torch.zeros((T, 3), dtype=torch.float64) if omega is None
                      else torch.as_tensor(omega).double().reshape(T, 3).cpu())

    # -- shape ------------------------------------------------------------
    @property
    def num_frames(self) -> int:
        return int(self.R.shape[0])

    # -- the shared loop's diagnostic scalars ------------------------------
    #
    # `FrameRecord` carries a `q`/`qdot` pair for every Phase-2 model. This
    # model has NO joint coordinate -- that is the point of it -- so these are
    # the translation and linear-speed magnitudes, exposed read-only so that
    # nothing downstream can mistake them for one. The real output is
    # `poses.npy` (section 18).
    @property
    def q(self) -> Tensor:
        return self.p.norm(dim=1)

    @property
    def qdot(self) -> Tensor:
        return self.v.norm(dim=1)

    def transform(self, frame: int):
        return self.R[frame], self.p[frame]

    def twist(self, frame: int) -> Tensor:
        return torch.cat([self.v[frame], self.omega[frame]])

    def clone(self) -> "RigidPart":
        return RigidPart(self.members, self.R.clone(), self.p.clone(),
                         self.v.clone(), self.omega.clone())

    # -- section 10: the retraction ---------------------------------------
    def retract(self, delta: Tensor, frame: int | None = None) -> "RigidPart":
        """`T^+ = Exp(delta_tau^) T` and `xi^+ = xi + delta_xi`.

        `project_to_so3` is applied to the product because a long window of
        accepted steps otherwise accumulates drift off SO(3) at the 1e-15 per
        step level, and section 20's exact-rigidity gate measures precisely
        that.
        """
        d = torch.as_tensor(delta).double().reshape(-1).cpu()
        if d.numel() != RIGID_ERROR_DIM:
            raise ValueError(f"rigid retraction needs {RIGID_ERROR_DIM} "
                             f"coordinates, got {d.numel()}")
        new = self.clone()
        dR, dt = se3_exp(d[TAU].reshape(1, 6))
        dR, dt = dR[0], dt[0]
        frames = range(self.num_frames) if frame is None else [int(frame)]
        for t in frames:
            new.R[t] = project_to_so3((dR @ self.R[t]).reshape(1, 3, 3))[0]
            new.p[t] = dR @ self.p[t] + dt
            new.v[t] = self.v[t] + d[XI][:3]
            new.omega[t] = self.omega[t] + d[XI][3:]
        return new

    def reset_jacobian(self, delta: Tensor) -> Tensor:
        """G: old error coordinates -> new ones, after `retract(delta)`.

        For the left perturbation `T = Exp(delta_tau^) That`, injecting `d`
        into the nominal leaves the residual error at `Exp((J_l(d) eps)^)`,
        so the pose block transports by the SE(3) left Jacobian at `d`. The
        velocity block is an ordinary vector space and transports by the
        identity.

        Dropping this and using `I` would rotate the pose uncertainty away
        from the pose it describes, and the first thing to pay for it is the
        pose/velocity cross-covariance -- which is the only route by which
        image evidence reaches the velocity at all (section 11).
        """
        d = torch.as_tensor(delta).double().reshape(-1).cpu()
        G = torch.eye(RIGID_ERROR_DIM, dtype=torch.float64)
        G[TAU, TAU] = se3_left_jacobian(d[TAU].reshape(1, 6))[0]
        return G

    # -- section 18: what this stage is allowed to report -------------------
    def geometry(self) -> dict:
        """Poses and velocities. Deliberately NOT a joint (sections 18, 34)."""
        return {"frames": self.num_frames,
                "num_members": int(self.members.numel()),
                "R": self.R.tolist(), "p": self.p.tolist(),
                "v": self.v.tolist(), "omega": self.omega.tolist(),
                "frame0_is_identity": bool(
                    torch.equal(self.R[0], torch.eye(3, dtype=torch.float64))
                    and float(self.p[0].abs().max()) == 0.0),
                "note": ("section 18: rigid pose trajectory only. No "
                         "prismatic direction, revolute axis, axis point or "
                         "joint type is produced by this stage.")}


# ---------------------------------------------------------------------------
# Section 4: the exact state-to-Gaussian map
# ---------------------------------------------------------------------------

def current_points(part: RigidPart, frame: int, mu0: Tensor,
                   R0: Tensor | None = None):
    """The moving members' transformed centers and orientations."""
    idx = part.members.to(mu0.device)
    Rt = part.R[frame].to(mu0.device)
    pt = part.p[frame].to(mu0.device)
    mu = mu0.double()[idx] @ Rt.T + pt
    Rg = None if R0 is None else Rt @ R0.double()[idx]
    return mu, Rg, Rt, pt


def state_to_gaussians(part: RigidPart, frame: int, mu0: Tensor,
                       R0: Tensor | None = None):
    """Centers AND orientations, for every Gaussian in the scene.

    Section 4 is explicit that these move together: updating a center without
    its orientation leaves the covariance pointing the wrong way, and a
    rotation-only motion then becomes indistinguishable from a translation in
    the image -- the exact bug that made the old Stage-A audit meaningless.
    """
    mu = mu0.double().clone()
    Rg = None if R0 is None else R0.double().clone()
    idx = part.members.to(mu.device)
    Rt = part.R[frame].to(mu.device)
    pt = part.p[frame].to(mu.device)
    mu[idx] = mu[idx] @ Rt.T + pt
    if Rg is not None:
        Rg[idx] = Rt @ Rg[idx]
    return mu, Rg


def transported_covariance(part: RigidPart, frame: int, Sigma0: Tensor) -> Tensor:
    """`Sigma_{i,t} = R_t Sigma_i^0 R_t^T` for the moving members.

    The renderer is given `(scales, rotation)` rather than a covariance, and
    `R_{i,t} = R_t R_i^0` with unchanged scales IS this transport. The
    function exists so section 20 can check that claim rather than assert it.
    """
    S = Sigma0.double().clone()
    idx = part.members.to(S.device)
    Rt = part.R[frame].to(S.device)
    S[idx] = Rt @ S[idx] @ Rt.T
    return S


# ---------------------------------------------------------------------------
# Sections 11, 12: the analytic JVP and its exact transpose
# ---------------------------------------------------------------------------

def rigid_jvp(part: RigidPart, frame: int, delta: Tensor,
              mu_now: Tensor) -> Tensor:
    """Pose tangent per moving Gaussian, from the 12-D rigid error state.

        d mu_i = delta_v + delta_omega x mu_i = [I, -hat(mu_i)] delta_tau
        d phi_i = delta_omega

    `mu_now` is the CURRENT transformed center, because a left perturbation
    acts in the world frame on the pose that is already there.

    The velocity block is identically zero (section 11): a single image has no
    derivative with respect to rigid velocity, and pretending otherwise would
    count the process prior as an extra measurement.
    """
    d = torch.as_tensor(delta).double().reshape(-1)
    dev = mu_now.device
    M = mu_now.shape[0]
    out = torch.zeros((M, POSE_DIM), dtype=torch.float64, device=dev)
    dv = d[0:3].to(dev)
    dw = d[3:6].to(dev)
    out[:, :3] = dv[None, :] + torch.linalg.cross(
        dw[None, :].expand(M, 3), mu_now.double())
    out[:, 3:] = dw[None, :]
    return out


def rigid_vjp(part: RigidPart, frame: int, pose_adjoint: Tensor,
              mu_now: Tensor) -> Tensor:
    """The exact transpose of `rigid_jvp` (section 12).

        bar_v     = sum_i bar_mu_i
        bar_omega = sum_i ( mu_i x bar_mu_i + bar_phi_i )

    from `<bar_mu, delta_omega x mu> = <mu x bar_mu, delta_omega>`.
    """
    dev = mu_now.device
    a = torch.as_tensor(pose_adjoint).double().reshape(-1, POSE_DIM).to(dev)
    a_mu, a_phi = a[:, :3], a[:, 3:]
    out = torch.zeros(RIGID_ERROR_DIM, dtype=torch.float64, device=dev)
    out[0:3] = a_mu.sum(0)
    out[3:6] = torch.linalg.cross(mu_now.double(), a_mu).sum(0) + a_phi.sum(0)
    return out
