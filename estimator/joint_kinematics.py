"""Sections 19, 25, 26: the state-to-Gaussian map and its analytic J_{G,J}.

CODEX_IMPLEMENT_PHASE2_RIGID_SCREW_IEKF.md.

This module is the ONLY mathematical change the architecture makes
(section 0). Phase 1 maps a free per-Gaussian error state to a Gaussian pose
tangent; Phase 2 maps a six-number-per-part joint error state to the same
Gaussian pose tangent, and everything downstream -- the renderer JVP/VJP, the
RGB-D residual, the information-form solve -- is untouched:

    delta y = H_G J_{G,J} delta x_J.

The pose tangent is Phase-1's own layout, `[P, 6] = (dmu, dphi)` with `dphi`
a SPATIAL rotation error, because `boxplus` is `R+ = Exp(hat(dphi)) R`.

Section 38 forbids autograd for the production JVP/VJP, so both directions are
written out analytically and checked against finite differences and against
each other by `tests/test_joint_screw_iekf.py`.
"""

from __future__ import annotations

import torch
from torch import Tensor

from .joint_manifold import (se3_exp, se3_left_jacobian, tangent_basis,
                             zeta_to_xi)
from .manifold import so3_hat

POSE_DIM = 6


def part_transform(q: float | Tensor, zeta: Tensor, length: float):
    """T(q) = Exp_SE3(q xi^) -> (R, t), the one transform the part may have."""
    xi = zeta_to_xi(zeta.double().reshape(6), length)
    qv = torch.as_tensor(q, dtype=torch.float64).reshape(1)
    R, t = se3_exp((qv[:, None] * xi[None, :]))
    return R[0], t[0]


def transport_gaussians(mu0: Tensor, R0: Tensor | None, R: Tensor, t: Tensor):
    # The joint state is six numbers per part and is kept on the CPU; the
    # Gaussians live wherever the renderer wants them. Moving the tiny
    # transform to the data rather than the data to the transform is what
    # keeps this map free of host/device copies of anything large.
    R, t = R.to(mu0.device), t.to(mu0.device)
    """Section 19: the exact rigid map that IS the rigidity constraint.

        mu_i = R mu_i^0 + t
        R_i  = R R_i^0
        Sigma_i = R Sigma_i^0 R^T   (implied by R_i, since Sigma = R S S^T R^T)

    Intrinsic scale, opacity and radiance are untouched -- frozen appearance.
    """
    mu = mu0.double() @ R.T + t
    Rg = None if R0 is None else (R @ R0.double())
    return mu, Rg


def _geometry_tau(q: float, zeta: Tensor, length: float,
                  basis: Tensor | None = None):
    """d(tau) / d(theta_J): the left SE(3) tangent of a geometry perturbation.

    Section 26, left/spatial convention:

        delta_xi  = S_L B_zeta delta_theta_J
        delta_tau = J_l(q xi) q delta_xi

    Returns the 6x4 matrix mapping `delta_theta_J` to `delta_tau`.
    """
    z = zeta.double().reshape(6)
    B = tangent_basis(z) if basis is None else basis.double()
    S = torch.ones(6, dtype=torch.float64, device=z.device)
    S[:3] = float(length)
    dxi = S[:, None] * B                       # S_L B, [6,4]
    xi = zeta_to_xi(z, length)
    Jl = se3_left_jacobian((float(q) * xi).reshape(1, 6))[0]
    return float(q) * (Jl @ dxi)               # [6,4]


def joint_jvp(part, frame: int, delta: Tensor, mu_now: Tensor | None = None,
              basis: Tensor | None = None) -> Tensor:
    """J_{G,J} delta: joint error state -> Gaussian pose tangent [M, 6].

    `delta` is `[dq, dqdot, dtheta_J(4)]` for THIS part at THIS frame.

    Section 25 gives the q column in closed form,

        d mu_i / dq = v + omega x mu_i(q) = v - [mu_i]_x omega
        d phi_i / dq = omega,

    where `mu_i(q)` is the CURRENT point, not the canonical one. At a
    prismatic state `omega = 0`, so the rotation column vanishes exactly and
    the translation column is `v` -- which after the section 10 conversion to
    physical travel is the theory note's `d mu / ds = e_P`.

    `dqdot` contributes nothing: section 15's `H_qdot = 0`. A single frame
    carries no direct residual on velocity, exactly as Phase 1's
    `embed_pose_to_full_state` zeroes its own velocity block. qdot is
    corrected through the process cross-covariance, never by inventing an
    image term for it.
    """
    d = torch.as_tensor(delta).double().reshape(-1)
    z = part.zeta
    xi = zeta_to_xi(z, part.length)
    v, w = xi[:3], xi[3:]
    q = float(part.q[frame])
    if mu_now is None:
        R, t = part_transform(q, z, part.length)
        mu_now = None                       # caller must supply for speed
        raise ValueError("joint_jvp needs mu_now, the current member points")
    mu = mu_now.double()
    M = mu.shape[0]
    dev = mu.device
    v, w = v.to(dev), w.to(dev)
    out = torch.zeros((M, POSE_DIM), dtype=torch.float64, device=dev)

    # ---- dq column (section 25) -----------------------------------------
    dq = float(d[0])
    if dq != 0.0:
        out[:, :3] += dq * (v[None, :] + torch.linalg.cross(
            w[None, :].expand(M, 3), mu))
        out[:, 3:] += dq * w[None, :]

    # ---- dtheta_J columns (section 26) ----------------------------------
    dth = d[2:6]
    if float(dth.abs().sum()) != 0.0:
        Gt = _geometry_tau(q, z, part.length, basis).to(dev)   # [6,4]
        tau = Gt @ dth.to(dev)                                 # [6]
        out[:, :3] += tau[None, :3] - torch.linalg.cross(
            mu, tau[None, 3:].expand(M, 3))
        out[:, 3:] += tau[None, 3:]
    return out


def joint_vjp(part, frame: int, pose_adjoint: Tensor, mu_now: Tensor,
              basis: Tensor | None = None) -> Tensor:
    """J_{G,J}^T a: Gaussian pose adjoint [M, 6] -> joint error state [6].

    The exact transpose of `joint_jvp`, written out rather than obtained by
    autograd (section 38). The composed identity section 24 requires,

        <H_G J v, z> == <v, J^T H_G^T z>,

    only holds if this is the true adjoint, so it is tested directly.
    """
    mu = mu_now.double()
    # The adjoint may arrive from the renderer (device) or from a caller's own
    # tensor (host); the Gaussians decide where the reduction happens.
    a = torch.as_tensor(pose_adjoint).double().reshape(-1, POSE_DIM).to(mu.device)
    a_mu, a_phi = a[:, :3], a[:, 3:]
    z = part.zeta
    xi = zeta_to_xi(z, part.length)
    v, w = xi[:3], xi[3:]
    q = float(part.q[frame])
    dev = mu.device
    v, w = v.to(dev), w.to(dev)
    out = torch.zeros(6, dtype=torch.float64, device=dev)

    # ---- dq ---------------------------------------------------------------
    dmu_dq = v[None, :] + torch.linalg.cross(w[None, :].expand(mu.shape[0], 3),
                                             mu)
    out[0] = (a_mu * dmu_dq).sum() + (a_phi * w[None, :]).sum()
    # out[1] stays exactly zero: section 15, H_qdot = 0.

    # ---- dtheta_J ---------------------------------------------------------
    # dmu_i = tau_v - [mu_i]_x tau_w  =>  adjoint wrt tau is
    #   g_v = sum_i a_mu,i ;  g_w = sum_i mu_i x a_mu,i + a_phi,i
    g_v = a_mu.sum(0)
    g_w = torch.linalg.cross(mu, a_mu).sum(0) + a_phi.sum(0)
    g_tau = torch.cat([g_v, g_w])
    Gt = _geometry_tau(q, z, part.length, basis).to(dev)
    out[2:6] = Gt.T @ g_tau
    return out


# ---------------------------------------------------------------------------
# whole-state helpers
# ---------------------------------------------------------------------------
def current_points(part, frame: int, mu0: Tensor, R0: Tensor | None = None):
    """The member Gaussians' current pose under this part's screw."""
    R, t = part_transform(float(part.q[frame]), part.zeta, part.length)
    idx = part.members.to(mu0.device)
    sub0 = mu0.double()[idx]
    subR = None if R0 is None else R0.double()[idx]
    mu, Rg = transport_gaussians(sub0, subR, R, t)
    return mu, Rg, R, t


def state_to_gaussians(state, frame: int, mu0: Tensor, R0: Tensor | None = None):
    """Section 19 over every part: the full Gaussian geometry at one frame.

    Non-member Gaussians keep their canonical pose exactly, so their
    derivative with respect to every joint coordinate is identically zero --
    section 47's acceptance gate M, satisfied by construction rather than by a
    mask applied afterwards.
    """
    mu = mu0.double().clone()
    Rg = None if R0 is None else R0.double().clone()
    for part in state.parts:
        sub, subR, _R, _t = current_points(part, frame, mu0, R0)
        idx = part.members.to(mu.device)
        mu[idx] = sub
        if Rg is not None and subR is not None:
            Rg[idx] = subR
    return mu, Rg


def full_jvp(state, frame: int, delta: Tensor, mu0: Tensor,
             R0: Tensor | None = None, bases=None) -> Tensor:
    """J_{G,J} delta for the whole state -> pose tangent [P, 6]."""
    d = torch.as_tensor(delta).double().reshape(-1)
    P = mu0.shape[0]
    out = torch.zeros((P, POSE_DIM), dtype=torch.float64, device=mu0.device)
    for k, part in enumerate(state.parts):
        mu, _Rg, _R, _t = current_points(part, frame, mu0, R0)
        b = None if bases is None else bases[k]
        out[part.members.to(out.device)] = joint_jvp(
            part, frame, d[state.slice_of(k)], mu_now=mu, basis=b)
    return out


def full_vjp(state, frame: int, pose_adjoint: Tensor, mu0: Tensor,
             R0: Tensor | None = None, bases=None) -> Tensor:
    """J_{G,J}^T a for the whole state -> joint error state [6K]."""
    a = torch.as_tensor(pose_adjoint).double().reshape(-1, POSE_DIM)
    out = torch.zeros(state.dim, dtype=torch.float64, device=a.device)
    for k, part in enumerate(state.parts):
        mu, _Rg, _R, _t = current_points(part, frame, mu0, R0)
        b = None if bases is None else bases[k]
        out[state.slice_of(k)] = joint_vjp(
            part, frame, a[part.members.to(a.device)], mu_now=mu,
            basis=b).to(out.device)
    return out
