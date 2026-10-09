"""Per-part physical models and the K-part pixel Jacobian.

CODEX_REQUIRED_MODIFICATIONS_MULTIVIEW_MULTIFRAME_KPART_IEKF.md sections 8, 9, 19.

Section 8: each moving part k carries its OWN model M_k in {P, R}, and the
scene at time t is

    G_t = G_0^static  U  union_k T_k(q_k,t; eta_k) G_0,k          (section 8)

"Do not fit one global twist over all moving labels."

Section 9: the pixel Jacobian is assembled from the EXISTING analytic
Gaussian-level one, with no conceptual rewrite of the renderer:

    delta y_p = sum_k H_{p,q_k} delta q_k,
    H_{p,q_k} = sum_{i: z_i = k} H_{p,i} G_{i,k}                  (section 9)

so a K-part JVP is one call to the Gaussian JVP on a tangent that is zero
outside part k, and a K-part VJP is one Gaussian VJP contracted with each
part's tangent.

Section 19: once a revolute hypothesis exists, that part is propagated in JOINT
coordinates, x_dot = q_dot e x (x - a), not by constant Cartesian twist.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch

from .line_geometry import PluckerLine

EPS = 1e-12
PRISMATIC = "prismatic"
REVOLUTE = "revolute"
UNRESOLVED = "UNRESOLVED"


@dataclass
class PartModel:
    """One moving part's kinematic model (section 8)."""

    kind: str                                  # "prismatic" | "revolute"
    axis: torch.Tensor                         # [3] unit, e_k
    moment: torch.Tensor | None = None         # [3], revolute only (Plucker m_k)
    part_index: int = 1
    info: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.kind not in (PRISMATIC, REVOLUTE):
            raise ValueError(f"part model must be prismatic or revolute, "
                             f"got {self.kind!r}")
        e = torch.as_tensor(self.axis).reshape(3)
        n = float(e.norm())
        if not (n > EPS):
            raise ValueError("a part model needs a non-degenerate axis")
        object.__setattr__(self, "axis", e / n)
        if self.kind == REVOLUTE:
            if self.moment is None:
                raise ValueError("a revolute part needs a Plucker moment; "
                                 "section 1.5 forbids an origin-centred axis")
            m = torch.as_tensor(self.moment).reshape(3)
            # enforce the e^T m = 0 gauge
            object.__setattr__(self, "moment", m - (m @ self.axis) * self.axis)

    @property
    def num_parameters(self) -> int:
        """Free parameters on the manifold: 2 for a direction on S^2, plus 2
        more for the line's position within the plane orthogonal to it."""
        return 2 if self.kind == PRISMATIC else 4

    @property
    def line(self) -> PluckerLine | None:
        if self.kind != REVOLUTE:
            return None
        return PluckerLine(direction=self.axis, moment=self.moment)

    def transport(self, mu0: torch.Tensor, R0: torch.Tensor | None, q):
        """Section 8's T_k(q; eta_k) applied to canonical geometry."""
        qq = float(q) if not torch.is_tensor(q) else float(q.reshape(-1)[0])
        if self.kind == PRISMATIC:
            return mu0 + self.axis.reshape(1, 3) * qq, R0
        from .line_geometry import revolute_transport_line
        return revolute_transport_line(mu0, R0, self.line, qq)

    def tangent(self, mu: torch.Tensor) -> torch.Tensor:
        """G_{i,k}: the [N,6] per-Gaussian pose tangent per unit dq."""
        n = int(mu.shape[0])
        t = torch.zeros((n, 6), device=mu.device, dtype=mu.dtype)
        if self.kind == PRISMATIC:
            t[:, :3] = self.axis.reshape(1, 3)
            return t
        # d mu/dq = e x mu + m ; d phi/dq = e   -- written in Plucker form so it
        # never needs the axis point and cannot be corrupted by a bad gauge.
        e = self.axis.reshape(1, 3).expand(n, 3)
        t[:, :3] = torch.linalg.cross(e, mu) + self.moment.reshape(1, 3)
        t[:, 3:] = e
        return t

    def velocity(self, mu: torch.Tensor, qdot: float) -> torch.Tensor:
        """Section 19: x_dot = q_dot e x (x - a) for revolute, q_dot e for
        prismatic. This is the propagation the reduced stage must use once a
        hypothesis exists -- constant Cartesian twist is a bootstrap-only
        approximation."""
        if self.kind == PRISMATIC:
            return self.axis.reshape(1, 3).expand_as(mu) * float(qdot)
        return self.line.velocity(mu, qdot=float(qdot))

    def to_dict(self) -> dict:
        d = {"kind": self.kind, "part_index": self.part_index,
             "axis": self.axis.detach().cpu().tolist(),
             "num_parameters": self.num_parameters, **self.info}
        if self.kind == REVOLUTE:
            l = self.line
            d.update({"plucker_moment": self.moment.detach().cpu().tolist(),
                      "closest_point_to_origin": l.closest_point.detach().cpu().tolist(),
                      "gauge_residual_e_dot_m": float(self.axis @ self.moment),
                      "passes_through_origin": bool(float(self.moment.norm()) < 1e-9)})
        return d


@dataclass
class SceneModel:
    """The full K-part scene model of section 8."""

    parts: list                                 # [K] PartModel (or None if unresolved)
    K: int

    def transport_all(self, mu0: torch.Tensor, R0: torch.Tensor | None,
                      labels: torch.Tensor, q: list):
        """G_t: static Gaussians stay at canonical geometry, each moving part is
        transported by its own model (section 8's boxed equation)."""
        mu = mu0.clone()
        R = None if R0 is None else R0.clone()
        for k in range(1, self.K + 1):
            model = self.parts[k - 1]
            if model is None:
                continue
            mask = labels == k
            if not bool(mask.any()):
                continue
            m_k, R_k = model.transport(mu0[mask],
                                       None if R0 is None else R0[mask], q[k - 1])
            mu[mask] = m_k
            if R is not None and R_k is not None:
                R[mask] = R_k
        return mu, R

    def part_tangent(self, mu: torch.Tensor, labels: torch.Tensor,
                     k: int) -> torch.Tensor:
        """Section 9's G_{i,k}, zero outside part k so one Gaussian-level JVP
        call yields H_{p,q_k} without touching the renderer."""
        model = self.parts[k - 1]
        t = torch.zeros((mu.shape[0], 6), device=mu.device, dtype=mu.dtype)
        if model is None:
            return t
        mask = labels == k
        if bool(mask.any()):
            t[mask] = model.tangent(mu[mask])
        return t

    @property
    def num_parameters(self) -> int:
        return sum(p.num_parameters for p in self.parts if p is not None)

    def to_dict(self) -> dict:
        return {"K": self.K,
                "parts": [(p.to_dict() if p is not None
                           else {"kind": UNRESOLVED}) for p in self.parts],
                "total_parameters": self.num_parameters}


# ---------------------------------------------------------------------------
# section 9: the K-part pixel Jacobian, matrix-free
# ---------------------------------------------------------------------------
class KPartObservation:
    """H_{p,q} for the stacked joint coordinate q = (q_1..q_K).

    Everything is expressed through the existing Gaussian-level operators, as
    section 9 requires ("The CUDA renderer does not need a conceptual rewrite").
    """

    def __init__(self, linearization, scene: SceneModel, mu: torch.Tensor,
                 labels: torch.Tensor):
        self.lin = linearization
        self.scene = scene
        self.mu = mu
        self.labels = labels
        self.K = scene.K
        self._tangents = [scene.part_tangent(mu, labels, k)
                          for k in range(1, self.K + 1)]

    def tangent(self, k: int) -> torch.Tensor:
        return self._tangents[k - 1]

    def apply_H(self, dq) -> list:
        """sum_k H_{p,q_k} dq_k -- one Gaussian JVP on the combined tangent."""
        total = None
        for k in range(1, self.K + 1):
            s = float(dq[k - 1])
            if s == 0.0:
                continue
            t = self._tangents[k - 1] * s
            total = t if total is None else total + t
        if total is None:
            total = torch.zeros_like(self.mu).repeat(1, 2)
        return self.lin.apply_H(total.contiguous())

    def apply_HT(self, pixel_adjoints) -> torch.Tensor:
        """[K] vector whose k-th entry is <H_{p,q_k}, z>: one Gaussian VJP,
        contracted with each part's tangent."""
        g = self.lin.apply_HT(pixel_adjoints)
        return torch.stack([(g * self._tangents[k - 1]).sum()
                            for k in range(1, self.K + 1)])

    def normal_system(self, precisions, residuals):
        """(I_q, b_q) for the K-dimensional joint coordinate.

        I_q is [K,K] and dense -- but K is the number of moving PARTS, so this
        is a 2x2 or 3x3 solve, not a scene-scale one. Cross terms are kept
        because two parts can share pixels.
        """
        weighted = [(p * r, q * d) for (p, q), (r, d) in zip(precisions, residuals)]
        b = self.apply_HT(weighted).double()
        cols = []
        for k in range(1, self.K + 1):
            hv = self.lin.apply_H(self._tangents[k - 1])
            w = [(p * a, q * d) for (p, q), (a, d) in zip(precisions, hv)]
            cols.append(self.apply_HT(w).double())
        I = torch.stack(cols, dim=1)
        I = 0.5 * (I + I.T)
        return I, b
