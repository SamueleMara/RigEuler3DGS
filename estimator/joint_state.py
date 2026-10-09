"""Section 14: the per-part Phase-2 joint state.

CODEX_IMPLEMENT_PHASE2_RIGID_SCREW_IEKF.md.

    nominal      x_k = (q_{k,t}, qdot_{k,t}, zeta_k)
    error state  dx_k = [dq, dqdot, dtheta_J] in R^6

The geometry `zeta_k` is ONE screw per part, persistent across every frame
(section 8): there is deliberately no `xi_t`, no per-frame `T_t` and no
per-Gaussian twist in the production state. That absence is what makes the
parameterisation rigid by construction, so no rigidity penalty is needed on
top of it (section 19).

`q` is the internal NORMALISED coordinate of the gauge in section 10, not a
distance and not an angle. Physical travel is `s = q |v|` and physical angle is
`theta = q |omega|`; the conversions live in `physical_coordinate` and must not
be skipped when comparing against the older P/R variables.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import torch
from torch import Tensor

from .joint_manifold import (project_to_manifold, prismatic_direction,
                             revolute_axis, tangent_basis, zeta_to_xi)

JOINT_ERROR_DIM = 6          # [dq, dqdot, dtheta_J (4)]
GEOMETRY_DIM = 4             # dtheta_J


@dataclass
class JointPart:
    """One moving part: its members, its canonical frame and its screw."""
    members: Tensor                      # [M] long, Gaussian indices
    centroid: Tensor                     # [3] canonical COM c^0
    length: float                        # L, the part's canonical radius
    zeta: Tensor                         # [6] = [u, omega], constrained
    q: Tensor                            # [T] internal coordinate per frame
    qdot: Tensor                         # [T]

    def __post_init__(self):
        # The joint state is six numbers per part and is deliberately kept on
        # the HOST: it is solved in float64, it is far too small to benefit
        # from the device, and keeping it there avoids a host/device round
        # trip on every tangent-basis SVD. The member indices are moved to the
        # Gaussians' device at each use site instead.
        self.zeta = project_to_manifold(torch.as_tensor(self.zeta).double()
                                        .reshape(6).cpu())
        self.centroid = torch.as_tensor(self.centroid).double().reshape(3).cpu()
        self.q = torch.as_tensor(self.q).double().reshape(-1).cpu()
        self.qdot = torch.as_tensor(self.qdot).double().reshape(-1).cpu()
        self.members = torch.as_tensor(self.members).long().reshape(-1).cpu()
        self.length = float(self.length)

    @property
    def num_frames(self) -> int:
        return int(self.q.numel())

    @property
    def xi(self) -> Tensor:
        """The physical twist [v; omega] = S_L zeta."""
        return zeta_to_xi(self.zeta, self.length)

    def basis(self) -> Tensor:
        return tangent_basis(self.zeta)

    def physical_coordinate(self, kind: str) -> Tensor:
        """Section 10: internal q -> physical travel or angle.

        Returning `q` itself would silently equate a normalised coordinate
        with a distance or an angle, which the spec calls out as mandatory to
        avoid.
        """
        xi = self.xi
        if kind == "prismatic":
            return self.q * float(xi[:3].norm())
        if kind == "revolute":
            return self.q * float(xi[3:].norm())
        raise ValueError("kind must be 'prismatic' or 'revolute'")

    def geometry(self) -> dict:
        """Section 13: derived physical geometry, never part of the state."""
        out = {"zeta": self.zeta.tolist(), "xi": self.xi.tolist(),
               "length_scale": self.length,
               "omega_norm": float(self.xi[3:].norm()),
               "v_norm": float(self.xi[:3].norm()),
               "prismatic_direction": prismatic_direction(
                   self.zeta, self.length).tolist()}
        ax = revolute_axis(self.zeta, self.length, self.centroid)
        out["revolute"] = (None if ax is None else
                           {k: (v.tolist() if torch.is_tensor(v) else v)
                            for k, v in ax.items()})
        return out


@dataclass
class JointState:
    """K parts, their shared frame times, and the joint covariance."""
    parts: list
    times: Tensor
    covariance: Tensor | None = None     # [6K, 6K] in the CURRENT bases

    def __post_init__(self):
        self.times = torch.as_tensor(self.times).double().reshape(-1)
        if self.covariance is None:
            n = JOINT_ERROR_DIM * len(self.parts)
            # A deliberately loose, finite prior: the geometry block is
            # dimensionless by construction (section 11), so one scale is
            # meaningful across every scene.
            self.covariance = torch.eye(n, dtype=torch.float64)

    @property
    def num_parts(self) -> int:
        return len(self.parts)

    @property
    def dim(self) -> int:
        return JOINT_ERROR_DIM * len(self.parts)

    def slice_of(self, k: int) -> slice:
        return slice(k * JOINT_ERROR_DIM, (k + 1) * JOINT_ERROR_DIM)

    def state_dict(self) -> dict:
        return {"times": self.times.tolist(),
                "parts": [{"members": p.members.tolist(),
                           "centroid": p.centroid.tolist(),
                           "length": p.length, "zeta": p.zeta.tolist(),
                           "q": p.q.tolist(), "qdot": p.qdot.tolist()}
                          for p in self.parts],
                "covariance": self.covariance.tolist()}


def initialise_from_twist(members, mu0, xi_guess, q_guess, times,
                          qdot_guess=None) -> JointPart:
    """Section 33: build a constrained part from an unconstrained twist guess.

    The guess is projected onto the manifold and `q` is rescaled by the same
    factor, so the product `q * xi` -- the only thing the transform sees -- is
    preserved exactly. Normalising the generator without compensating `q`
    would silently rescale the motion.
    """
    members = torch.as_tensor(members).long().reshape(-1)
    pts = torch.as_tensor(mu0).double()[members]
    c0 = pts.mean(0)
    L = float((pts - c0).norm(dim=1).square().mean().sqrt().clamp_min(1e-9))
    xi = torch.as_tensor(xi_guess).double().reshape(6)
    zeta_raw = torch.cat([xi[:3] / L, xi[3:]])
    scale = float(zeta_raw.norm())
    zeta = project_to_manifold(zeta_raw)
    q = torch.as_tensor(q_guess).double().reshape(-1) * scale
    qd = (torch.zeros_like(q) if qdot_guess is None
          else torch.as_tensor(qdot_guess).double().reshape(-1) * scale)
    return JointPart(members=members, centroid=c0, length=L, zeta=zeta,
                     q=q, qdot=qd)
