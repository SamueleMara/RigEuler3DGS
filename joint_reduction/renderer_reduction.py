"""Reduced joint observation operators through the real renderer.

CODEX_IMPLEMENT_SPARSE_ALPHA_AWARE_IEKF.md sections 29 and 30.

Once a joint is identified, the per-Gaussian 12N error state is replaced by the
joint coordinate q. The renderer Jacobian machinery is REUSED rather than
rebuilt (section 2's closing requirement): the pose tangent basis G maps a joint
tangent to the per-Gaussian 6D pose tangent, so

    H_q v_q = H (G v_q)          (section 29.1 / 29.2's pixel-joint Jacobian)
    H_q^T z = G^T (H^T z)

and the dense RGB-D information collapses to the tiny normal system of
section 30:

    I_q = P_q^-1 + sum_p H_pq^T R_p^-1 H_pq,   b_q = sum_p H_pq^T R_p^-1 r_p.

Note what this does NOT do: it never forms H, and it never lets the Gaussians
move independently again (section 30's "all moving Gaussian centers and
rotations are generated rigidly from the joint").
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch

from .prismatic import prismatic_pose_tangent, prismatic_transport
from .revolute import revolute_pose_tangent, revolute_transport


@dataclass
class RenderJointModel:
    """A prismatic or revolute joint bound to the RENDERER's pose tangents.

    Distinct from ``joint_filter.JointModel``, which is the kinematic model:
    this one knows how to turn a joint tangent into the per-Gaussian 6D pose
    tangent the CUDA operators consume.
    """

    kind: str                     # "prismatic" | "revolute"
    axis: torch.Tensor            # [3] unit
    point: torch.Tensor | None = None   # [3], revolute only
    member: torch.Tensor | None = None  # [N] bool; None means every Gaussian

    def __post_init__(self) -> None:
        if self.kind not in ("prismatic", "revolute"):
            # Section 0 rule 13: screw/helical is deliberately out of scope for
            # the first estimator, so an unknown kind is an error, not a
            # silently-accepted third case.
            raise ValueError(f"joint kind must be prismatic or revolute, got {self.kind!r}")
        if self.kind == "revolute" and self.point is None:
            raise ValueError("a revolute joint needs a point on its axis")

    def pose_tangent(self, mu: torch.Tensor) -> torch.Tensor:
        """G_i, [N,6,1]: how each Gaussian's pose moves per unit dq."""
        n = int(mu.shape[0])
        if self.kind == "prismatic":
            g = prismatic_pose_tangent(self.axis, n)
        else:
            g = revolute_pose_tangent(self.axis, self.point, mu)
        g = g.reshape(n, 6, -1)
        if self.member is not None:
            # A Gaussian outside the part does not move with the joint, so its
            # tangent is exactly zero -- not a small number.
            g = g * self.member.reshape(n, 1, 1).to(g.dtype)
        return g

    def transport(self, mu0: torch.Tensor, R0: torch.Tensor, q: float):
        """Generate the Gaussian poses from the joint coordinate (section 30)."""
        if self.kind == "prismatic":
            mu, R = prismatic_transport(mu0, R0, self.axis, q)
        else:
            mu, R = revolute_transport(mu0, R0, self.axis, self.point, q)
        if self.member is not None:
            m = self.member.reshape(-1, 1).to(mu.dtype)
            mu = mu0 + (mu - mu0) * m
            keep = self.member.reshape(-1, 1, 1).to(R.dtype)
            R = R0 + (R - R0) * keep
        return mu, R


def joint_apply_H(linearization, model: RenderJointModel, mu: torch.Tensor,
                  dq: torch.Tensor) -> list[tuple[torch.Tensor, torch.Tensor]]:
    """H_q dq, via the pose tangent basis (section 29.1/29.2)."""
    g = model.pose_tangent(mu)                       # [N,6,m]
    pose = (g @ dq.reshape(1, -1, 1).expand(g.shape[0], -1, 1)).squeeze(-1)
    return linearization.apply_H(pose.contiguous().float())


def joint_apply_HT(linearization, model: RenderJointModel, mu: torch.Tensor,
                   pixel_adjoints) -> torch.Tensor:
    """H_q^T z = G^T (H^T z) -> [m]."""
    pose_adj = linearization.apply_HT(pixel_adjoints)   # [N,6]
    g = model.pose_tangent(mu).to(pose_adj.dtype)       # [N,6,m]
    return torch.einsum("nkm,nk->m", g, pose_adj)


def joint_normal_system(linearization, model: RenderJointModel, mu: torch.Tensor,
                        residuals, precisions, prior_information=None):
    """Section 30's reduced normal system.

    Returns (I_q, b_q). The Fisher is built by applying H_q to each basis
    direction, which is m extra JVPs -- affordable precisely because m is 1 or 2
    after reduction, and still matrix-free in the Gaussian state.
    """
    g = model.pose_tangent(mu)
    m = int(g.shape[-1])
    weighted = [(pc * rc, pd * rd) for (pc, pd), (rc, rd) in zip(precisions, residuals)]
    b_q = joint_apply_HT(linearization, model, mu, weighted)

    cols = []
    for k in range(m):
        e = torch.zeros(m, device=g.device, dtype=g.dtype)
        e[k] = 1.0
        hv = joint_apply_H(linearization, model, mu, e)
        cols.append([(pc * a, pd * d) for (pc, pd), (a, d) in zip(precisions, hv)])
    I_q = torch.zeros((m, m), device=g.device, dtype=torch.float64)
    for a in range(m):
        for b in range(m):
            e = torch.zeros(m, device=g.device, dtype=g.dtype)
            e[b] = 1.0
            hv_b = joint_apply_H(linearization, model, mu, e)
            acc = 0.0
            for (wc, wd), (hc, hd) in zip(cols[a], hv_b):
                acc += float((wc * hc).sum() + (wd * hd).sum())
            I_q[a, b] = acc
    I_q = 0.5 * (I_q + I_q.T)
    if prior_information is not None:
        I_q = I_q + prior_information.to(I_q.dtype)
    return I_q, b_q.to(torch.float64)


def joint_update(linearization, model: RenderJointModel, mu: torch.Tensor,
                 residuals, precisions, prior_information=None,
                 damping: float = 1e-9) -> tuple[torch.Tensor, dict]:
    """delta_q = I_q^-1 b_q (section 30). Tiny dense solve, by construction."""
    I_q, b_q = joint_normal_system(linearization, model, mu, residuals,
                                   precisions, prior_information)
    m = I_q.shape[0]
    A = I_q + damping * torch.eye(m, device=I_q.device, dtype=I_q.dtype)
    try:
        dq = torch.linalg.solve(A, b_q.reshape(m, 1)).reshape(m)
        ok = True
    except Exception:
        # Report rather than return a confident number from a singular system.
        dq = torch.zeros(m, device=I_q.device, dtype=I_q.dtype)
        ok = False
    ev = torch.linalg.eigvalsh(A)
    return dq, {"information": I_q, "rhs": b_q, "solved": ok,
                "eigenvalues": ev.tolist(),
                "condition": float(ev.max() / ev.min().clamp_min(1e-30))}
