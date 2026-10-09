"""Stage B (sections 21-26): the exact revolute Phase-2 model M_R.

CODEX_VALIDATE_GENERIC_TWIST_THEN_PARALLEL_PRISMATIC_REVOLUTE_PHASE2.md.

    mu_i(t) = a_R + R(e_R, theta_t) (mu_i^0 - a_R)
    R_i(t)  = R(e_R, theta_t) R_i^0

so the motion is a rotation about a FIXED finite axis, by construction: there
is no translation degree of freedom at all, and no way for the model to drift
into a screw.

Section 22's minimal geometry, which is what keeps the axis gauge fixed:

    e_R           unit, 2 DOF       the axis direction
    a_R - c_0 = E_R rho, 2 DOF      the offset, in the plane orthogonal to e_R

so `e_R^T (a_R - c_0) = 0` holds exactly rather than being imposed afterwards,
and the component of the offset ALONG the axis -- which moves no Gaussian and
is pure gauge -- is simply not representable.

State (section 23): `(theta_t, thetadot_t, e_R, rho)`, with the 6-dimensional
local error state `[dtheta, dthetadot, deta_1, deta_2, drho_1, drho_2]` and a
full dense 6x6 covariance.
"""

from __future__ import annotations

import math

import torch
from torch import Tensor

from .manifold import so3_exp, so3_hat, so3_left_jacobian, so3_log

REVOLUTE_ERROR_DIM = 6
POSE_DIM = 6


def axis_basis(e: Tensor) -> Tensor:
    """E_R in R^{3x2}, orthonormal, spanning the plane orthogonal to e."""
    e = e.double().reshape(3)
    e = e / e.norm().clamp_min(1e-300)
    tmp = torch.tensor([0.0, 0.0, 1.0], dtype=torch.float64, device=e.device)
    if abs(float(e @ tmp)) > 0.9:
        tmp = torch.tensor([0.0, 1.0, 0.0], dtype=torch.float64, device=e.device)
    e1 = torch.linalg.cross(e, tmp)
    e1 = e1 / e1.norm().clamp_min(1e-300)
    e2 = torch.linalg.cross(e, e1)
    return torch.stack([e1, e2], dim=1)


class RevolutePart:
    """One moving part under M_R."""

    def __init__(self, members, centroid, axis, rho, theta, thetadot=None,
                 plane=None):
        self.members = torch.as_tensor(members).long().reshape(-1).cpu()
        self.centroid = torch.as_tensor(centroid).double().reshape(3).cpu()
        e = torch.as_tensor(axis).double().reshape(3).cpu()
        self.axis = e / e.norm().clamp_min(1e-300)
        # E_R is CARRIED in the state, not recomputed from the axis.  `rho` is
        # a coordinate pair in this frame, so if the frame were rebuilt from
        # scratch after every update it would spin freely about the axis and
        # `rho` would silently mean something different on each iteration.
        self.plane = (axis_basis(self.axis) if plane is None
                      else torch.as_tensor(plane).double().reshape(3, 2).cpu())
        self.rho = torch.as_tensor(rho).double().reshape(2).cpu()
        self.theta = torch.as_tensor(theta).double().reshape(-1).cpu()
        self.thetadot = (torch.zeros_like(self.theta) if thetadot is None
                         else torch.as_tensor(thetadot).double().reshape(-1).cpu())

    @property
    def num_frames(self) -> int:
        return int(self.theta.numel())

    def basis(self) -> Tensor:
        return self.plane

    def retract(self, delta: Tensor, frame: int | None = None) -> "RevolutePart":
        """Section 24: the boxplus that keeps every constraint exact.

        `deta` rotates the WHOLE frame (e, e_1, e_2) rigidly by
        `w = e x E deta`, so `e` stays unit and `e^T (a - c_0) = 0` survives
        exactly -- not to first order -- because both sides are carried by the
        same rotation.  `drho` then moves the offset inside that rotated plane.
        """
        d = torch.as_tensor(delta).double().reshape(-1).cpu()
        new = RevolutePart(self.members, self.centroid, self.axis, self.rho,
                           self.theta, self.thetadot, plane=self.plane)
        if frame is None:
            new.theta = self.theta + float(d[0])
            new.thetadot = self.thetadot + float(d[1])
        else:
            new.theta = self.theta.clone()
            new.thetadot = self.thetadot.clone()
            new.theta[frame] = new.theta[frame] + float(d[0])
            new.thetadot[frame] = new.thetadot[frame] + float(d[1])
        de = self.plane @ d[2:4]
        w = torch.linalg.cross(self.axis, de)
        W = so3_exp(w.reshape(1, 3))[0].double()
        new.axis = W @ self.axis
        new.axis = new.axis / new.axis.norm()
        new.plane = W @ self.plane
        new.rho = self.rho + d[4:6]
        return new

    @property
    def axis_point(self) -> Tensor:
        """a_R = c_0 + E_R rho, so e_R . (a_R - c_0) = 0 by construction."""
        return self.centroid + self.basis() @ self.rho

    def transform(self, frame: int):
        """(R_t, p_t) of the rotation about the finite axis."""
        th = float(self.theta[frame])
        R = so3_exp((th * self.axis).reshape(1, 3))[0].double()
        a = self.axis_point
        return R, a - R @ a


    def reset_jacobian(self, delta: Tensor) -> Tensor:
        """G: old error coordinates -> new ones, after `retract(delta)`.

        The displacement pair is unaffected -- `theta` and its rate mean the
        same thing before and after.  The geometry coordinates do move: they
        are components in the carried plane `E_R`, and the retraction
        rotates that plane by `W`, so a direction held as `x` in the old basis
        is `(E_R')^T E_R x` in the new one.  Dropping this is what
        silently rotates the geometry covariance away from the geometry it
        describes, and it is the cross-covariance to the displacement -- the
        only route by which image evidence reaches the velocity -- that pays
        for it first.
        """
        d = torch.as_tensor(delta).double().reshape(-1).cpu()
        de = self.plane @ d[2:4]
        W = so3_exp(torch.linalg.cross(self.axis, de).reshape(1, 3))[0].double()
        M = (W @ self.plane).T @ self.plane
        G = torch.eye(REVOLUTE_ERROR_DIM, dtype=torch.float64)
        G[2:4, 2:4] = M
        G[4:6, 4:6] = M
        return G

    def geometry(self) -> dict:
        return {"axis": self.axis.tolist(), "rho": self.rho.tolist(),
                "axis_point": self.axis_point.tolist(),
                "centroid": self.centroid.tolist(),
                "gauge_e_dot_offset": float(
                    self.axis @ (self.axis_point - self.centroid)),
                "theta": self.theta.tolist(),
                "thetadot": self.thetadot.tolist()}


def current_points(part: RevolutePart, frame: int, mu0: Tensor,
                   R0: Tensor | None = None):
    R, p = part.transform(frame)
    idx = part.members.to(mu0.device)
    Rd, pd = R.to(mu0.device), p.to(mu0.device)
    mu = mu0.double()[idx] @ Rd.T + pd
    Rg = None if R0 is None else (Rd @ R0.double()[idx])
    return mu, Rg, Rd, pd


def state_to_gaussians(part: RevolutePart, frame: int, mu0: Tensor,
                       R0: Tensor | None = None):
    mu = mu0.double().clone()
    Rg = None if R0 is None else R0.double().clone()
    sub, subR, _R, _p = current_points(part, frame, mu0, R0)
    idx = part.members.to(mu.device)
    mu[idx] = sub
    if Rg is not None and subR is not None:
        Rg[idx] = subR
    return mu, Rg


def revolute_jvp(part: RevolutePart, frame: int, delta: Tensor,
                 mu_now: Tensor) -> Tensor:
    """Section 25/26: the analytic pose tangent for M_R's six coordinates.

    `delta = [dtheta, dthetadot, deta(2), drho(2)]`.

    theta (section 25):
        d mu_i / dtheta  = e_R x (mu_i(t) - a_R)
        d phi_i / dtheta = e_R

    axis direction (section 26): the direction is perturbed in its own plane,
    `de = E_R deta`, and the rotation responds through the SO(3) left Jacobian,
    `dphi = J_l(theta e) theta de`. The axis POINT moves with it: `a` is tied
    to `e` through `a = c_0 + E_R(e) rho`, and the constraint
    `e . (a - c_0) = 0` must survive the perturbation, which costs a
    compensating component along `e`:

        da = -e (deta . rho).

    Dropping that term leaves the offset no longer perpendicular to the axis,
    so the gauge drifts and the recovered axis point wanders along the axis
    from frame to frame.

    axis offset: `da = E_R drho`, and since the axis direction is unchanged
    `dphi = 0` while `dmu = (I - R) da`.

    `dthetadot` contributes nothing: a single frame has no angular-velocity
    derivative, exactly as in the generic model.
    """
    d = torch.as_tensor(delta).double().reshape(-1)
    mu = mu_now.double()
    dev = mu.device
    M = mu.shape[0]
    e = part.axis.to(dev)
    E = part.basis().to(dev)
    a = part.axis_point.to(dev)
    th = float(part.theta[frame])
    R, _p = part.transform(frame)
    R = R.to(dev)
    out = torch.zeros((M, POSE_DIM), dtype=torch.float64, device=dev)

    # ---- dtheta (section 25) --------------------------------------------
    dth = float(d[0])
    if dth != 0.0:
        out[:, :3] += dth * torch.linalg.cross(e.expand(M, 3), mu - a[None, :])
        out[:, 3:] += dth * e[None, :]

    # ---- deta: the axis direction ---------------------------------------
    deta = d[2:4]
    if float(deta.abs().sum()) != 0.0:
        de = E @ deta.to(dev)
        Jl = so3_left_jacobian((th * e).reshape(1, 3))[0].double().to(dev)
        dphi = Jl @ (th * de)
        # The frame rotates rigidly (see `retract`), so the axis point is
        # carried along: da = (e x de) x (E rho) = -e (deta . rho), which is
        # exactly the component that keeps e . (a - c_0) = 0.
        da = -e * float(deta.to(dev) @ part.rho.to(dev))
        # mu = a + R (mu0 - a): perturbing R about the axis point
        out[:, :3] += (da[None, :] - torch.linalg.cross(
            mu - a[None, :], dphi.expand(M, 3)) - R @ da)
        out[:, 3:] += dphi[None, :]

    # ---- drho: the axis offset ------------------------------------------
    drho = d[4:6]
    if float(drho.abs().sum()) != 0.0:
        da = E @ drho.to(dev)
        out[:, :3] += (da - R @ da)[None, :]
    return out


def revolute_vjp(part: RevolutePart, frame: int, pose_adjoint: Tensor,
                 mu_now: Tensor) -> Tensor:
    """The exact transpose of `revolute_jvp` (no autograd)."""
    mu = mu_now.double()
    dev = mu.device
    a_adj = torch.as_tensor(pose_adjoint).double().reshape(-1, POSE_DIM).to(dev)
    a_mu, a_phi = a_adj[:, :3], a_adj[:, 3:]
    e = part.axis.to(dev)
    E = part.basis().to(dev)
    a = part.axis_point.to(dev)
    th = float(part.theta[frame])
    R, _p = part.transform(frame)
    R = R.to(dev)
    M = mu.shape[0]
    out = torch.zeros(REVOLUTE_ERROR_DIM, dtype=torch.float64, device=dev)

    # dtheta
    dmu = torch.linalg.cross(e.expand(M, 3), mu - a[None, :])
    out[0] = (a_mu * dmu).sum() + (a_phi * e[None, :]).sum()
    # out[1] stays zero: no instantaneous thetadot derivative.

    # deta: adjoint through dphi and through da
    Jl = so3_left_jacobian((th * e).reshape(1, 3))[0].double().to(dev)
    g_phi = torch.linalg.cross(mu - a[None, :], a_mu).sum(0) + a_phi.sum(0)
    g_de = th * (Jl.T @ g_phi)
    g_da = a_mu.sum(0) - R.T @ a_mu.sum(0)
    out[2:4] = E.T @ g_de - part.rho.to(dev) * float(e @ g_da)

    # drho
    out[4:6] = E.T @ g_da
    return out
