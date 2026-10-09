"""Plücker line geometry for revolute joints.

CODEX_STRICT_CORRECTIONS_IEKF_JOINT_DISCOVERY_V2.md sections 7, 8, 9.

A revolute joint is a LINE in space, not a direction. Representing it by a
direction alone silently places it through the origin, which section 7 calls out
as a P0 bug. The axis is therefore carried as a Plücker pair

    (e, m),   m = a x e,   ||e|| = 1,   e . m = 0                 (section 9)

from which the closest point on the line is recovered by

    a = e x m                                                      (section 9)

and the instantaneous velocity field is

    v(x) = qdot (e x x + m).

m = 0 means "axis through the origin". That is a legitimate special case and
must never be the default assumption (section 9).

The equivariance that makes this representation trustworthy, and which section 9
requires as a test: under a world translation x' = x + t the SAME physical line
has m' = m + t x e.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

EPS = 1e-12


def _unit(v: torch.Tensor, eps: float = EPS) -> torch.Tensor:
    return v / v.norm(dim=-1, keepdim=True).clamp_min(eps)


@dataclass
class PluckerLine:
    """A 3D line (e, m) with the gauge e.m = 0."""

    direction: torch.Tensor           # [3] unit
    moment: torch.Tensor              # [3], = a x e

    def __post_init__(self) -> None:
        e = torch.as_tensor(self.direction).reshape(3)
        m = torch.as_tensor(self.moment).reshape(3)
        n = float(e.norm())
        if not (n > EPS):
            raise ValueError("a Plucker line needs a non-degenerate direction")
        e = e / n
        # Project the moment onto the gauge e.m = 0. A component of m along e is
        # not a line at all -- it is the pitch of a screw, which section 0 rule 9
        # keeps out of the joint family.
        m = m - (m @ e) * e
        object.__setattr__(self, "direction", e)
        object.__setattr__(self, "moment", m)

    @property
    def closest_point(self) -> torch.Tensor:
        """a = e x m (section 9): the point on the line closest to the origin."""
        return torch.linalg.cross(self.direction, self.moment)

    @staticmethod
    def from_point_direction(a: torch.Tensor, e: torch.Tensor) -> "PluckerLine":
        """m = a x e (section 9). ``a`` may be ANY point on the line."""
        e = _unit(torch.as_tensor(e).reshape(3))
        a = torch.as_tensor(a).reshape(3)
        return PluckerLine(direction=e, moment=torch.linalg.cross(a, e))

    @staticmethod
    def from_twist(omega: torch.Tensor, u: torch.Tensor,
                   *, min_omega: float = 1e-6) -> tuple["PluckerLine | None", dict]:
        """Section 8: the axis of an ORIGIN-referenced twist v(x) = u + omega x x.

        For a pure revolute joint u = -omega x a, so a = (omega x u)/||omega||^2
        and m = a x e.

        When ||omega|| is small the axis LOCATION is unobservable -- the line
        runs off to infinity and any finite point is arbitrary. Section 8 is
        explicit: never return a huge axis point and call it meaningful. The
        caller gets None and a reason.
        """
        omega = torch.as_tensor(omega).reshape(3)
        u = torch.as_tensor(u).reshape(3)
        w2 = float(omega @ omega)
        w = w2 ** 0.5
        info = {"omega_norm": w, "u_norm": float(u.norm()),
                "pitch": float((omega @ u) / max(w2, EPS)) if w2 > EPS else float("nan")}
        if w < min_omega:
            info["observable"] = False
            info["reason"] = (f"||omega|| = {w:.3e} < {min_omega:.1e}: the rotation "
                              f"axis location is unobservable, so no finite axis "
                              f"point is reported")
            return None, info
        a = torch.linalg.cross(omega, u) / w2
        info["observable"] = True
        info["axis_point"] = a.tolist()
        return PluckerLine.from_point_direction(a, omega / w), info

    def velocity(self, x: torch.Tensor, qdot: float = 1.0) -> torch.Tensor:
        """v(x) = qdot (e x x + m) (section 9)."""
        e = self.direction.reshape(1, 3).expand_as(x)
        return qdot * (torch.linalg.cross(e, x) + self.moment.reshape(1, 3))

    def translate(self, t: torch.Tensor) -> "PluckerLine":
        """The same physical line in coordinates x' = x + t: m' = m + t x e."""
        t = torch.as_tensor(t).reshape(3)
        return PluckerLine(direction=self.direction,
                           moment=self.moment + torch.linalg.cross(t, self.direction))

    def distance_to(self, other: "PluckerLine") -> dict:
        """Line-to-line geometry, which is how axis error must be measured.

        Section 19's T-J1 is explicit that a raw point difference is wrong: two
        identical lines can have arbitrarily different 'axis points' under a
        different gauge. What is physical is the angle between directions and
        the perpendicular distance between the lines.
        """
        e1, e2 = self.direction, other.direction
        cos = float(torch.clamp((e1 @ e2).abs(), -1.0, 1.0))
        angle = float(torch.rad2deg(torch.arccos(torch.tensor(cos))))
        a1, a2 = self.closest_point, other.closest_point
        cross = torch.linalg.cross(e1, e2)
        cn = float(cross.norm())
        if cn < 1e-7:
            # parallel: distance is the perpendicular offset between them
            delta = a2 - a1
            perp = delta - (delta @ e1) * e1
            dist = float(perp.norm())
        else:
            dist = float(torch.abs((a2 - a1) @ (cross / cn)))
        return {"angle_deg": angle, "line_distance": dist,
                "point_distance": float((a1 - a2).norm())}

    def to_dict(self) -> dict:
        return {"direction": self.direction.detach().cpu().tolist(),
                "moment": self.moment.detach().cpu().tolist(),
                "closest_point": self.closest_point.detach().cpu().tolist(),
                "gauge_residual": float(self.direction @ self.moment)}


def centroid_twist_to_origin(v_c: torch.Tensor, omega: torch.Tensor,
                             c: torch.Tensor) -> torch.Tensor:
    """Section 8: u = v_c - omega x c.

    A rigid fit solving v(x) = v_c + omega x (x - c) returns v_c AT THE
    CENTROID. Feeding v_c straight into the axis formula puts the axis in the
    wrong place -- this exact bug was found and fixed once already.
    """
    return torch.as_tensor(v_c).reshape(3) - torch.linalg.cross(
        torch.as_tensor(omega).reshape(3), torch.as_tensor(c).reshape(3))


def revolute_transport_line(mu0: torch.Tensor, R0: torch.Tensor | None,
                            line: PluckerLine, q):
    """Finite revolute motion about an arbitrary line (section 10.2).

        mu(q) = a + R(e,q)(mu0 - a),   R(q) = R(e,q) R0,   a = e x m
    """
    from estimator.manifold import so3_exp

    e = line.direction
    a = line.closest_point
    qq = float(q) if not torch.is_tensor(q) else float(q.reshape(-1)[0])
    Rq = so3_exp((e * qq).reshape(1, 3)).reshape(3, 3)
    mu = (mu0 - a.reshape(1, 3)) @ Rq.T + a.reshape(1, 3)
    R = None if R0 is None else Rq.unsqueeze(0) @ R0
    return mu, R


def revolute_joint_tangent(mu: torch.Tensor, line: PluckerLine):
    """Section 10.2: d mu/dq = e x mu + m,  d phi/dq = e.

    Written in the Plücker form deliberately: `e x (mu - a)` and `e x mu + m`
    are algebraically identical, but the second never needs the axis point and
    so cannot be corrupted by a bad gauge.
    """
    e = line.direction.reshape(1, 3).expand_as(mu)
    dmu = torch.linalg.cross(e, mu) + line.moment.reshape(1, 3)
    dphi = line.direction.reshape(1, 3).expand_as(mu)
    return dmu, dphi


def fit_plucker_from_motion(mu: torch.Tensor, v: torch.Tensor,
                            weights: torch.Tensor | None = None,
                            *, min_omega: float = 1e-6) -> tuple["PluckerLine | None", dict]:
    """Section 11.3's revolute initializer: fit v_i ~ qdot (e x mu_i + m).

    Solved as a linear problem in the 6-vector (omega, u) with omega = qdot e
    and u = qdot m, which is exactly the origin-referenced twist, then converted
    to a line. The scale qdot and the line are separable because the line only
    needs the DIRECTION of omega and the normalized moment.
    """
    x = torch.as_tensor(mu)
    vv = torch.as_tensor(v)
    n = int(x.shape[0])
    if weights is None:
        weights = torch.ones(n, device=x.device, dtype=x.dtype)
    w = weights.reshape(-1).clamp_min(0.0)

    # v_i = u + omega x x_i = [I , -[x_i]_x] [u; omega]
    zero = torch.zeros(n, device=x.device, dtype=x.dtype)
    hat = torch.stack([
        torch.stack([zero, -x[:, 2], x[:, 1]], dim=1),
        torch.stack([x[:, 2], zero, -x[:, 0]], dim=1),
        torch.stack([-x[:, 1], x[:, 0], zero], dim=1)], dim=1)
    eye = torch.eye(3, device=x.device, dtype=x.dtype).expand(n, 3, 3)
    A = torch.cat([eye, -hat], dim=2)                   # [n,3,6]
    wA = w[:, None, None] * A
    H = torch.einsum("nij,nik->jk", A, wA)
    g = torch.einsum("nij,ni->j", wA, vv)
    try:
        xi = torch.linalg.solve(H + 1e-10 * torch.eye(6, device=x.device, dtype=x.dtype),
                                g.unsqueeze(-1)).squeeze(-1)
    except Exception:
        return None, {"observable": False, "reason": "normal matrix is singular"}
    u, omega = xi[:3], xi[3:]
    line, info = PluckerLine.from_twist(omega, u, min_omega=min_omega)
    ev = torch.linalg.eigvalsh(H.double())
    info.update({"eigenvalues": ev.tolist(),
                 "condition_number": float(ev.max() / ev.min().clamp_min(1e-30)),
                 "qdot": float(omega.norm())})
    if line is not None:
        pred = line.velocity(x, qdot=float(omega.norm()))
        num = float((w * (vv - pred).norm(dim=1)).sum() / w.sum().clamp_min(EPS))
        den = float((w * vv.norm(dim=1)).sum() / w.sum().clamp_min(EPS))
        info["residual_share"] = num / max(den, EPS)
    return line, info
