"""Section 29.2 -- revolute reduction.

A revolute part has a single scalar coordinate ``q`` (an angle about the line
``{a + s e}``):

    mu_i(q) = a + R(e,q) (mu_i^0 - a),      R_i(q) = R(e,q) R_i^0,

so the pose tangent is

    G_i^R = [e x (mu_i - a); e]  in R^{6x1},

ordered ``[d mu / dq; d phi / dq]``.  The lower block is ``e`` precisely because
section 4 retracts rotations on the *left* (``R^+ = Exp([delta phi]_x) R^-``):

    R(q+h) R(q)^{-1} = R(e, q+h) R_0 R_0^T R(e, q)^T = R(e, h)  =>  d phi/dq = e,

independently of ``R_i^0``.  ``tests/test_joint_reduction.py`` checks exactly this
against a central difference of :func:`revolute_transport`.

Everything is closed form; nothing here differentiates anything (section 36).
"""

from __future__ import annotations

import math
from typing import Any

import torch

from .prismatic import _as_scalar, normalize_axis
from .rigid_fit import _sym, hat

EPS = 1.0e-12
#: Below this angle Rodrigues' coefficients are evaluated by Taylor series.
SMALL_ANGLE = 1.0e-6


def rotation_about_axis(e: torch.Tensor, q: torch.Tensor | float) -> torch.Tensor:
    """``R(e,q) = I + sin q [e]_x + (1 - cos q) [e]_x^2`` for a unit ``e``.

    Rodrigues in this form has no removable singularity at ``q = 0`` -- the usual
    ``sin(theta)/theta`` blow-up only appears when the angle is folded into the
    axis, which is exactly why the joint coordinate is kept separate from ``e``
    everywhere in this package.
    """
    e = normalize_axis(e)
    qq = _as_scalar(q, e.device, e.dtype)
    k = hat(e)
    eye = torch.eye(3, device=e.device, dtype=e.dtype)
    return eye + torch.sin(qq) * k + (1.0 - torch.cos(qq)) * (k @ k)


def so3_log_matrix(rot: torch.Tensor) -> torch.Tensor:
    """Batched ``SO(3)`` logarithm, ``[...,3,3] -> [...,3]``.

    Local copy on purpose: `joint_reduction` must stay importable without
    `estimator/` (section 4's ``manifold.py`` owns the canonical version used for
    state retraction).  Three branches, because the one-line formula
    ``theta/(2 sin theta) * vee(R - R^T)`` is 0/0 at ``theta = 0`` and blows up at
    ``theta = pi``:

    * ``theta -> 0``: Taylor of ``theta/(2 sin theta)``;
    * generic: the closed form;
    * ``theta -> pi``: ``(R + I)/2 -> a a^T``, so the axis is read off the column
      with the largest diagonal entry.  The sign is genuinely ambiguous there
      (``+a`` and ``-a`` describe the same rotation), so it is canonicalised to
      make the dominant component positive.
    """
    rot = torch.as_tensor(rot)
    if rot.shape[-2:] != (3, 3):
        raise ValueError(f"expected [...,3,3], got {tuple(rot.shape)}")
    batch = rot.reshape(-1, 3, 3)
    trace = batch[:, 0, 0] + batch[:, 1, 1] + batch[:, 2, 2]
    cos_theta = (0.5 * (trace - 1.0)).clamp(-1.0, 1.0)
    theta = torch.acos(cos_theta)
    vee = 0.5 * torch.stack(
        (
            batch[:, 2, 1] - batch[:, 1, 2],
            batch[:, 0, 2] - batch[:, 2, 0],
            batch[:, 1, 0] - batch[:, 0, 1],
        ),
        dim=-1,
    )  # = sin(theta) * axis

    sin_theta = torch.sin(theta)
    small = theta < SMALL_ANGLE
    safe_sin = torch.where(small, torch.ones_like(sin_theta), sin_theta)
    t2 = torch.where(small, theta * theta, torch.zeros_like(theta))
    factor = torch.where(
        small,
        1.0 + t2 / 6.0 + 7.0 * t2 * t2 / 360.0,
        theta / torch.where(safe_sin.abs() < EPS, torch.ones_like(safe_sin), safe_sin),
    )
    out = vee * factor[:, None]

    near_pi = theta > (math.pi - 1.0e-4)
    if bool(near_pi.any().item()):
        sub = batch[near_pi]
        sym = _sym(sub) + torch.eye(3, device=sub.device, dtype=sub.dtype)
        sym = 0.5 * sym                                  # -> a a^T as theta -> pi
        diag = torch.diagonal(sym, dim1=-2, dim2=-1).clamp_min(0.0)
        k = torch.argmax(diag, dim=-1)
        col = torch.gather(sym, 2, k.view(-1, 1, 1).expand(-1, 3, 1)).squeeze(-1)
        axis = col / col.norm(dim=-1, keepdim=True).clamp_min(EPS)
        sign = torch.sign(torch.gather(axis, 1, k.view(-1, 1)).squeeze(1))
        axis = axis * torch.where(sign == 0, torch.ones_like(sign), sign)[:, None]
        out = out.clone()
        out[near_pi] = theta[near_pi, None] * axis
    return out.reshape(rot.shape[:-2] + (3,))


def revolute_pose_tangent(
    e: torch.Tensor, a: torch.Tensor, mu: torch.Tensor
) -> torch.Tensor:
    """``G^R in R^{N x 6 x 1}`` with ``G_i = [e x (mu_i - a); e]`` (section 29.2).

    ``mu`` must be the centres at the *current* ``q`` (the tangent is evaluated at
    the current pose, not at ``q = 0``).
    """
    mu = torch.as_tensor(mu)
    if mu.dim() != 2 or mu.shape[1] != 3:
        raise ValueError(f"mu must be [N,3], got {tuple(mu.shape)}")
    e = normalize_axis(torch.as_tensor(e, device=mu.device, dtype=mu.dtype))
    a = torch.as_tensor(a, device=mu.device, dtype=mu.dtype).reshape(3)
    g = torch.zeros(mu.shape[0], 6, 1, device=mu.device, dtype=mu.dtype)
    g[:, :3, 0] = torch.linalg.cross(e.expand_as(mu), mu - a, dim=-1)
    g[:, 3:, 0] = e
    return g


def revolute_transport(
    mu0: torch.Tensor,
    R0: torch.Tensor | None,
    e: torch.Tensor,
    a: torch.Tensor,
    q: torch.Tensor | float,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """``mu = a + R(e,q)(mu0 - a)``, ``R = R(e,q) R0`` (section 29.2).

    ``R0=None`` is accepted for the centres-only paths (axis fitting, residual
    evaluation) and propagates as ``None``.
    """
    mu0 = torch.as_tensor(mu0)
    if mu0.dim() != 2 or mu0.shape[1] != 3:
        raise ValueError(f"mu0 must be [N,3], got {tuple(mu0.shape)}")
    e = torch.as_tensor(e, device=mu0.device, dtype=mu0.dtype)
    a = torch.as_tensor(a, device=mu0.device, dtype=mu0.dtype).reshape(3)
    rot = rotation_about_axis(e, _as_scalar(q, mu0.device, mu0.dtype))
    mu = a + (mu0 - a) @ rot.transpose(0, 1)
    if R0 is None:
        return mu, None
    R0 = torch.as_tensor(R0, device=mu0.device, dtype=mu0.dtype)
    if R0.shape != (mu0.shape[0], 3, 3):
        raise ValueError(f"R0 must be [N,3,3], got {tuple(R0.shape)}")
    return mu, rot @ R0


@torch.no_grad()
def fit_revolute_axis(
    mu0: torch.Tensor,
    mu_t: torch.Tensor,
    weights: torch.Tensor | None = None,
    *,
    rank_tol: float = 1.0e-10,
) -> dict[str, Any]:
    """Recover ``(e, a, q)`` of a revolute joint from two poses of a part (29.2).

    Method -- two exact linear facts about a rotation, used in sequence:

    1. **Direction.**  ``e^T (R - I) x = (R^T e)^T x - e^T x = 0`` because
       ``R^T e = e``, so *every* displacement ``d_i = mu_t_i - mu0_i`` is
       orthogonal to the axis.  ``e`` is therefore the null direction of
       ``sum_i w_i d_i d_i^T``, i.e. the eigenvector of its smallest eigenvalue.
    2. **Point.**  A rotation preserves the distance to any point of its own
       axis, ``|a - mu0_i| = |a - mu_t_i|``, which linearises to the
       perpendicular-bisector condition ``d_i^T a = d_i^T m_i`` with
       ``m_i = (mu0_i + mu_t_i)/2``.  Stacking those gives the normal system
       ``(sum w d d^T) a = sum w d (d . m)``, whose matrix is rank 2 *by
       construction* (all ``d`` lie in the plane orthogonal to ``e``).  That null
       direction is the slide along the axis, which no two-pose observation can
       ever fix, so the minimum-norm solution is taken and then projected to
       ``a - (a . e) e``: the point of the axis closest to the origin.  This is
       the usual screw-axis canonicalisation -- compare fitted axis points up to
       a slide along ``e``.
    3. **Angle.**  With ``e`` and ``a`` known, project both poses into the plane
       orthogonal to ``e`` and solve the 2D Procrustes problem in closed form:
       ``q = atan2(sum w e . (p0 x pt), sum w p0 . pt)``.

    Sign convention: ``(e, q) -> (-e, -q)`` describes the same motion, resolved by
    forcing ``q >= 0``.  ``a`` is invariant under that flip.
    """
    mu0 = torch.as_tensor(mu0)
    mu_t = torch.as_tensor(mu_t, device=mu0.device, dtype=mu0.dtype)
    if mu0.shape != mu_t.shape or mu0.dim() != 2 or mu0.shape[1] != 3:
        raise ValueError(f"expected matching [N,3], got {tuple(mu0.shape)} / {tuple(mu_t.shape)}")
    n = mu0.shape[0]
    if n < 2:
        raise ValueError("fit_revolute_axis needs at least two Gaussians")
    device, dtype = mu0.device, mu0.dtype

    if weights is None:
        w = torch.ones(n, device=device, dtype=dtype)
    else:
        w = torch.as_tensor(weights, device=device, dtype=dtype).reshape(n).clamp_min(0.0)
    w_sum = w.sum()
    if float(w_sum.item()) <= 0.0:
        raise ValueError("all weights are zero")

    disp = mu_t - mu0
    mid = 0.5 * (mu0 + mu_t)
    scatter = _sym(torch.einsum("n,na,nb->ab", w, disp, disp))

    evals, evecs = torch.linalg.eigh(scatter)             # ascending
    e = evecs[:, 0]

    rhs = torch.einsum("n,na,n->a", w, disp, (disp * mid).sum(dim=-1))
    lam_max = evals.max().clamp_min(0.0)
    tol = rank_tol * lam_max
    keep = evals > tol
    inv = torch.where(keep, 1.0 / torch.where(keep, evals, torch.ones_like(evals)),
                      torch.zeros_like(evals))
    a = evecs @ (inv * (evecs.transpose(0, 1) @ rhs))
    a = a - (a @ e) * e                                   # canonical slide

    p0 = (mu0 - a) - ((mu0 - a) @ e)[:, None] * e
    pt = (mu_t - a) - ((mu_t - a) @ e)[:, None] * e
    sin_acc = (w * (torch.linalg.cross(p0, pt, dim=-1) @ e)).sum()
    cos_acc = (w * (p0 * pt).sum(dim=-1)).sum()
    q = torch.atan2(sin_acc, cos_acc)
    if float(q.item()) < 0.0:                             # canonical (e, q >= 0)
        e, q = -e, -q

    pred, _ = revolute_transport(mu0, None, e, a, q)
    per_point = (pred - mu_t).norm(dim=-1)
    residual = float(((w * per_point.square()).sum() / w_sum).clamp_min(0.0).sqrt().item())

    radius_sq = float(((w * (p0 * p0).sum(dim=-1)).sum() / w_sum).item())
    return {
        "axis": e,
        "point": a,
        "q": q,
        "residual": residual,
        "per_point_residual": per_point,
        # evals[0] ~ 0 and evals[1:] >> 0 is the signature of a clean revolute
        # fit; a second small eigenvalue means the displacements are collinear
        # and only the axis *plane* is observable.
        "scatter_eigenvalues": evals,
        "axis_rank": int(keep.sum().item()),
        "axis_gap": float((evals[0] / lam_max.clamp_min(EPS)).item()),
        "mean_radius": math.sqrt(max(radius_sq, 0.0)),
        "degenerate": bool(int(keep.sum().item()) < 2),
        "num_gaussians": n,
    }
