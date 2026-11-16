"""OPTIONAL, separate, disabled-by-default final joint-geometry refinement (spec section 8). NOT RUN.

Nothing in Phase 4, Phase 5 or run_bidirectional_all.sh imports or calls this module. It exists so a future
experiment can refine the joint AFTER Phase 5 with every Gaussian tensor, label, camera and the topology
frozen. Its CLI (tools/refine_joint_geometry_final.py) is a dry run unless --enable is given, and --enable
requires a completed Phase-5 joint_unchanged_proof.json whose hashes match the Phase-3 files.

Minimal, type-preserving parameterisation (the joint type never changes):
  prismatic  e = normalize(e0 + B d_e), d_e in R^2 (tangent plane of S^2 at e0); no origin DOF
  revolute   e = normalize(e0 + B d_e); a = a0 + B d_a, d_a in R^2 (perpendicular to e0; no slide along e)
  optional   q_k = q3_k + dq_k with dq_anchor = 0 fixed, Gaussian prior sum (dq_k / sigma_k)^2 and a
             time-scaled smoothness term (only with allow_q_correction)
B = an orthonormal basis of e0^perp. Objective: full-sequence RGB-D-mask loss + d^T Lambda d, with a declared
trust region (|angle(e, e0)| <= trust_axis_deg, |d_a| <= trust_point_m). Corrections that exceed the caps or
do not lower the declared full-sequence validation loss are rejected; the output goes to a separate
phase5_joint_final/ and never overwrites Phase 3, 4 or 5.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import torch

from .articulated_scene import joint_transform_torch


@dataclass
class RefineConfig:
    steps: int = 10
    lr: float = 1e-3
    prior_axis: float = 1e4          # Lambda for d_e (rad^-2)
    prior_point: float = 1e4         # Lambda for d_a (m^-2)
    trust_axis_deg: float = 2.0
    trust_point_m: float = 0.01
    allow_q_correction: bool = False
    q_prior_sd: float = 0.01
    q_smooth: float = 1.0

    def to_dict(self):
        return asdict(self)


def tangent_basis(e: np.ndarray) -> np.ndarray:
    e = np.asarray(e, float) / np.linalg.norm(e)
    a = np.array([1.0, 0, 0]) if abs(e[0]) < 0.9 else np.array([0, 1.0, 0])
    u = np.cross(e, a); u /= np.linalg.norm(u)
    w = np.cross(e, u)
    return np.stack([u, w], 1)                          # [3,2]


class MinimalJoint(torch.nn.Module):
    def __init__(self, joint, allow_q_correction=False, device="cpu", dtype=torch.float64):
        super().__init__()
        self.joint = joint
        t = lambda x: torch.as_tensor(np.asarray(x, float), device=device, dtype=dtype)
        self.register_buffer("e0", t(joint.axis))
        self.register_buffer("B", t(tangent_basis(joint.axis)))
        self.register_buffer("q0", t(joint.q))
        self.d_e = torch.nn.Parameter(torch.zeros(2, device=device, dtype=dtype))
        if joint.type == "revolute":
            self.register_buffer("a0", t(joint.point))
            self.d_a = torch.nn.Parameter(torch.zeros(2, device=device, dtype=dtype))
        else:
            self.a0, self.d_a = None, None
        self.dq = torch.nn.Parameter(torch.zeros(len(joint.q) - 1, device=device, dtype=dtype),
                                     requires_grad=allow_q_correction)    # anchor row excluded: fixed 0

    def axis(self):
        e = self.e0 + self.B @ self.d_e
        return e / e.norm()

    def point(self):
        return None if self.a0 is None else self.a0 + self.B @ self.d_a

    def q(self, k: int):
        return self.q0[k] + (self.dq[k - 1] if k > 0 else 0.0)

    def transform(self, frame: int):
        k = self.joint.frames.index(int(frame))
        return joint_transform_torch(self.joint.type, self.axis(), self.point(), self.q(k))

    def dof(self) -> int:
        return 2 + (2 if self.a0 is not None else 0) + (self.dq.numel() if self.dq.requires_grad else 0)

    def prior(self, cfg: RefineConfig):
        p = cfg.prior_axis * (self.d_e ** 2).sum()
        if self.d_a is not None:
            p = p + cfg.prior_point * (self.d_a ** 2).sum()
        if self.dq.requires_grad:
            p = p + ((self.dq / cfg.q_prior_sd) ** 2).sum() + cfg.q_smooth * (torch.diff(self.dq) ** 2).sum()
        return p

    def within_trust(self, cfg: RefineConfig) -> bool:
        ang = float(torch.rad2deg(torch.arccos(torch.clamp(self.axis() @ self.e0, -1, 1))))
        ok = ang <= cfg.trust_axis_deg
        if self.d_a is not None:
            ok = ok and float(self.d_a.norm()) <= cfg.trust_point_m
        return bool(ok)
