"""Stage B (sections 17-20): the exact prismatic Phase-2 model M_P.

CODEX_VALIDATE_GENERIC_TWIST_THEN_PARALLEL_PRISMATIC_REVOLUTE_PHASE2.md.

    mu_i(t) = mu_i^0 + s_t e_P
    R_i(t)  = R_i^0

A pure translation along a fixed unit direction, by construction: the model
has no rotational degree of freedom to spend, so a translating part cannot be
explained by a far-away axis and the `eta -> 0` limit of the generic chart is
never approached from a direction where the geometry is ill-conditioned.

State (section 19): `(s_t, sdot_t, e_P)` with the 4-dimensional local error
state `[ds, dsdot, dgamma_1, dgamma_2]` and a dense 4x4 covariance.  As in
M_R, the orthonormal plane `G_P` spanning the directions transverse to `e_P`
is carried in the state and rotated with it, never rebuilt from scratch.
"""

from __future__ import annotations

import torch
from torch import Tensor

from .manifold import so3_exp
from .phase2_revolute import axis_basis

PRISMATIC_ERROR_DIM = 4
POSE_DIM = 6


class PrismaticPart:
    """One moving part under M_P."""

    def __init__(self, members, direction, s, sdot=None, plane=None):
        self.members = torch.as_tensor(members).long().reshape(-1).cpu()
        e = torch.as_tensor(direction).double().reshape(3).cpu()
        self.direction = e / e.norm().clamp_min(1e-300)
        self.plane = (axis_basis(self.direction) if plane is None
                      else torch.as_tensor(plane).double().reshape(3, 2).cpu())
        self.s = torch.as_tensor(s).double().reshape(-1).cpu()
        self.sdot = (torch.zeros_like(self.s) if sdot is None
                     else torch.as_tensor(sdot).double().reshape(-1).cpu())

    @property
    def num_frames(self) -> int:
        return int(self.s.numel())

    def basis(self) -> Tensor:
        return self.plane

    def transform(self, frame: int):
        R = torch.eye(3, dtype=torch.float64)
        return R, float(self.s[frame]) * self.direction

    def retract(self, delta: Tensor, frame: int | None = None) -> "PrismaticPart":
        d = torch.as_tensor(delta).double().reshape(-1).cpu()
        new = PrismaticPart(self.members, self.direction, self.s, self.sdot,
                            plane=self.plane)
        if frame is None:
            new.s = self.s + float(d[0])
            new.sdot = self.sdot + float(d[1])
        else:
            new.s = self.s.clone()
            new.sdot = self.sdot.clone()
            new.s[frame] = new.s[frame] + float(d[0])
            new.sdot[frame] = new.sdot[frame] + float(d[1])
        de = self.plane @ d[2:4]
        W = so3_exp(torch.linalg.cross(self.direction, de).reshape(1, 3))[0].double()
        new.direction = W @ self.direction
        new.direction = new.direction / new.direction.norm()
        new.plane = W @ self.plane
        return new


    def reset_jacobian(self, delta: Tensor) -> Tensor:
        """G: old error coordinates -> new ones, after `retract(delta)`.

        The displacement pair is unaffected -- `s` and its rate mean the
        same thing before and after.  The geometry coordinates do move: they
        are components in the carried plane `G_P`, and the retraction
        rotates that plane by `W`, so a direction held as `x` in the old basis
        is `(G_P')^T G_P x` in the new one.  Dropping this is what
        silently rotates the geometry covariance away from the geometry it
        describes, and it is the cross-covariance to the displacement -- the
        only route by which image evidence reaches the velocity -- that pays
        for it first.
        """
        d = torch.as_tensor(delta).double().reshape(-1).cpu()
        de = self.plane @ d[2:4]
        W = so3_exp(torch.linalg.cross(self.direction, de).reshape(1, 3))[0].double()
        M = (W @ self.plane).T @ self.plane
        G = torch.eye(PRISMATIC_ERROR_DIM, dtype=torch.float64)
        G[2:4, 2:4] = M
        return G

    def geometry(self) -> dict:
        return {"direction": self.direction.tolist(),
                "s": self.s.tolist(), "sdot": self.sdot.tolist(),
                "unit_defect": float(self.direction.norm()) - 1.0}


def current_points(part: PrismaticPart, frame: int, mu0: Tensor,
                   R0: Tensor | None = None):
    idx = part.members.to(mu0.device)
    R, p = part.transform(frame)
    pd = p.to(mu0.device)
    mu = mu0.double()[idx] + pd
    Rg = None if R0 is None else R0.double()[idx]
    return mu, Rg, R.to(mu0.device), pd


def state_to_gaussians(part: PrismaticPart, frame: int, mu0: Tensor,
                       R0: Tensor | None = None):
    mu = mu0.double().clone()
    Rg = None if R0 is None else R0.double().clone()
    idx = part.members.to(mu.device)
    mu[idx] = mu[idx] + (float(part.s[frame]) * part.direction).to(mu.device)
    return mu, Rg


def prismatic_jvp(part: PrismaticPart, frame: int, delta: Tensor,
                  mu_now: Tensor) -> Tensor:
    """Analytic pose tangent for M_P's four coordinates.

        d mu / ds      = e_P            d phi / ds      = 0
        d mu / dgamma  = s_t G_P        d phi / dgamma  = 0

    The rotational block is identically zero: that is the whole point of the
    model, and it is why `s` and `e_P` are the only things the data can move.
    """
    d = torch.as_tensor(delta).double().reshape(-1)
    dev = mu_now.device
    M = mu_now.shape[0]
    out = torch.zeros((M, POSE_DIM), dtype=torch.float64, device=dev)
    e = part.direction.to(dev)
    G = part.plane.to(dev)
    out[:, :3] += float(d[0]) * e[None, :]
    out[:, :3] += float(part.s[frame]) * (G @ d[2:4].to(dev))[None, :]
    return out


def prismatic_vjp(part: PrismaticPart, frame: int, pose_adjoint: Tensor,
                  mu_now: Tensor) -> Tensor:
    """The exact transpose of `prismatic_jvp`."""
    dev = mu_now.device
    a_mu = torch.as_tensor(pose_adjoint).double().reshape(-1, POSE_DIM)[:, :3].to(dev)
    g = a_mu.sum(0)
    out = torch.zeros(PRISMATIC_ERROR_DIM, dtype=torch.float64, device=dev)
    out[0] = part.direction.to(dev) @ g
    out[2:4] = float(part.s[frame]) * (part.plane.to(dev).T @ g)
    return out
