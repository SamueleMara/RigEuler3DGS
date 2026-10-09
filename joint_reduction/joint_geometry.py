"""SE(3) trajectory primitives for geometric joint discovery.

CODEX_IMPLEMENT_GEOMETRIC_RIGID_MOTION_JOINT_DISCOVERY.md sections 13, 17, 20.

Section 13 makes the FINITE transform the primary representation of part
motion:

    x_{i,t} = R_{k,t} x_i^0 + t_{k,t},      R^G_{i,t} = R_{k,t} R_i^0

and demotes the instantaneous twist to an initializer. Section 48 is explicit
about the ordering this file sits in: the renderer says who moves and how much,
and only then does rigid geometry say which joint. Nothing here looks at an
image -- it consumes the recovered {T_{k,t}} and nothing else.

Section 20 fixes the error that the P/R decision is allowed to use. It must be
dimensionless, so a position residual is divided by the part's own RMS radius:

    E^geom_M = (1/T) sum_t [ ||dt_t||^2 / L_k^2 + ||dtheta_t||^2 ]

which makes one radian of orientation error worth one RMS-radius displacement.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import torch

from estimator.manifold import so3_exp, so3_hat, so3_log, project_to_so3

EPS = 1e-12


# ---------------------------------------------------------------------------
# section 13: finite rigid transforms
# ---------------------------------------------------------------------------
def fit_rigid_transform(x0: torch.Tensor, xt: torch.Tensor,
                        weights: torch.Tensor | None = None) -> tuple:
    """The weighted orthogonal Procrustes (Kabsch) solution.

        argmin_{R in SO(3), t} sum_i w_i || x_{i,t} - (R x_i^0 + t) ||^2

    The reflection branch is removed explicitly: a raw SVD can return a matrix
    with det = -1, which fits a mirrored point set better than any rotation and
    would silently leave SO(3).

    Returns ``(R, t)`` with ``R`` in ``[3,3]`` and ``t`` in ``[3]``.
    """
    x0 = x0.double()
    xt = xt.double()
    n = x0.shape[0]
    if n == 0:
        eye = torch.eye(3, dtype=torch.float64, device=x0.device)
        return eye, torch.zeros(3, dtype=torch.float64, device=x0.device)
    w = (torch.ones(n, dtype=torch.float64, device=x0.device)
         if weights is None else weights.double().clamp_min(0.0))
    sw = w.sum().clamp_min(EPS)
    c0 = (w.unsqueeze(1) * x0).sum(0) / sw
    ct = (w.unsqueeze(1) * xt).sum(0) / sw
    a = x0 - c0
    b = xt - ct
    h = (w.unsqueeze(1) * a).T @ b                      # [3,3]
    u, _s, vh = torch.linalg.svd(h)
    d = torch.det(vh.T @ u.T)
    s = torch.diag(torch.tensor([1.0, 1.0, float(torch.sign(d).clamp_min(-1.0))],
                                dtype=torch.float64, device=x0.device))
    if float(d) == 0.0:
        s = torch.eye(3, dtype=torch.float64, device=x0.device)
    rot = vh.T @ s @ u.T
    return rot, ct - rot @ c0


def fit_rigid_trajectory(x0: torch.Tensor, frames: list,
                         weights: torch.Tensor | None = None) -> dict:
    """{T_{k,t}} for one part over a whole window (section 13).

    `frames` is a list of ``[N,3]`` positions of the SAME points at each frame,
    in the same order as ``x0``.
    """
    rots, trans, resid = [], [], []
    for xt in frames:
        rot, tr = fit_rigid_transform(x0, xt, weights)
        rots.append(rot)
        trans.append(tr)
        pred = x0.double() @ rot.T + tr
        resid.append(float(((xt.double() - pred) ** 2).sum(dim=1).mean().sqrt()))
    return {"R": torch.stack(rots), "t": torch.stack(trans),
            "rigid_rmse": resid}


def rms_radius(x0: torch.Tensor, weights: torch.Tensor | None = None) -> float:
    """L_k of section 20: the RMS radius of the part about its own centroid."""
    x0 = x0.double()
    if x0.shape[0] == 0:
        return 1.0
    w = (torch.ones(x0.shape[0], dtype=torch.float64, device=x0.device)
         if weights is None else weights.double().clamp_min(0.0))
    sw = w.sum().clamp_min(EPS)
    c = (w.unsqueeze(1) * x0).sum(0) / sw
    return float(((w * ((x0 - c) ** 2).sum(1)).sum() / sw).clamp_min(EPS).sqrt())


def se3_distance(ra: torch.Tensor, ta: torch.Tensor,
                 rb: torch.Tensor, tb: torch.Tensor, length: float = 1.0) -> float:
    """d_SE(3) of section 28, on the same scale as section 20's error."""
    dth = float(so3_log(project_to_so3(ra.T @ rb).unsqueeze(0)).norm())
    dt = float((ta - tb).norm()) / max(float(length), EPS)
    return math.sqrt(dth * dth + dt * dt)


# ---------------------------------------------------------------------------
# section 20: the normalized geometric error
# ---------------------------------------------------------------------------
def normalized_geometric_error(R_obs: torch.Tensor, t_obs: torch.Tensor,
                               R_hat: torch.Tensor, t_hat: torch.Tensor,
                               length: float) -> dict:
    """Section 20's E^geom.

        E^geom = (1/T) sum_t [ ||dt_t||^2 / L^2 + ||dtheta_t||^2 ]

    Both halves are dimensionless, so prismatic and revolute fits are compared
    on one scale without any weight to choose (section 19 requires "same
    normalization").
    """
    L2 = max(float(length) ** 2, EPS)
    dt = ((t_obs.double() - t_hat.double()) ** 2).sum(dim=1) / L2
    dR = torch.stack([project_to_so3((R_hat[i].double().T @ R_obs[i].double()))
                      for i in range(R_obs.shape[0])])
    dth = (so3_log(dR) ** 2).sum(dim=1)
    total = float((dt + dth).mean())
    return {"error": total,
            "position_term": float(dt.mean()),
            "rotation_term": float(dth.mean()),
            "per_frame": [float(x) for x in (dt + dth)],
            "length_scale": float(length)}


# ---------------------------------------------------------------------------
# section 17: geometric invariants, used to validate rigid support
# ---------------------------------------------------------------------------
def pairwise_distance_drift(x0: torch.Tensor, frames: list,
                            max_pairs: int = 4096, seed: int = 0) -> dict:
    """Section 17.1: ||x_i(t) - x_j(t)|| must equal ||x_i(0) - x_j(0)||.

    A sampled estimate -- the full set is O(N^2) and the statistic only needs
    to be unbiased, not exhaustive.
    """
    n = x0.shape[0]
    if n < 2 or not frames:
        return {"relative_drift": 0.0, "pairs": 0}
    g = torch.Generator(device="cpu").manual_seed(seed)
    m = min(max_pairs, n * (n - 1) // 2)
    i = torch.randint(0, n, (m,), generator=g).to(x0.device)
    j = torch.randint(0, n, (m,), generator=g).to(x0.device)
    keep = i != j
    i, j = i[keep], j[keep]
    d0 = (x0.double()[i] - x0.double()[j]).norm(dim=1)
    rel = []
    for xt in frames:
        dt = (xt.double()[i] - xt.double()[j]).norm(dim=1)
        rel.append(float(((dt - d0).abs() / d0.clamp_min(EPS)).mean()))
    return {"relative_drift": float(sum(rel) / max(len(rel), 1)),
            "per_frame": rel, "pairs": int(i.numel())}


def center_of_mass(x: torch.Tensor, weights: torch.Tensor | None = None):
    """Section 17.2."""
    x = x.double()
    w = (torch.ones(x.shape[0], dtype=torch.float64, device=x.device)
         if weights is None else weights.double().clamp_min(0.0))
    return (w.unsqueeze(1) * x).sum(0) / w.sum().clamp_min(EPS)


def second_moment(x: torch.Tensor, weights: torch.Tensor | None = None):
    """Section 17.3: the inertia-like tensor about the weighted centroid."""
    x = x.double()
    w = (torch.ones(x.shape[0], dtype=torch.float64, device=x.device)
         if weights is None else weights.double().clamp_min(0.0))
    c = center_of_mass(x, w)
    d = x - c
    return (w.unsqueeze(1) * d).T @ d / w.sum().clamp_min(EPS)


def unwrap_angles(q: list) -> list:
    """Section 22: a consistent temporal unwrap of the recovered angle.

    An axis-angle read frame by frame is only defined modulo 2*pi and flips
    sign with the axis; without unwrapping, a joint that sweeps past pi would
    read as a reversal.
    """
    if not q:
        return []
    out = [float(q[0])]
    for x in q[1:]:
        prev = out[-1]
        v = float(x)
        while v - prev > math.pi:
            v -= 2.0 * math.pi
        while prev - v > math.pi:
            v += 2.0 * math.pi
        out.append(v)
    return out
