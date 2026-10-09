"""Section 29.1 -- prismatic reduction.

A prismatic part has a single scalar coordinate ``q`` (a translation along a unit
axis ``e``):

    mu_i(q) = mu_i^0 + q e,        R_i(q) = R_i^0,

so the pose tangent of every Gaussian is the constant

    G_i^P = [e; 0]  in R^{6x1},

ordered ``[d mu / dq ; d phi / dq]`` to match the ``[delta mu; delta phi]`` pose
error state of section 4.  Contracting the per-Gaussian pixel Jacobians with
``G`` gives the boxed reduced Jacobian of 29.1,
``H^P_{p,q} = sum_{i in S_p} H^mu_{p,i} e`` -- see
:func:`joint_reduction.joint_filter.reduce_pixel_jacobian`.

Everything is closed form; nothing here differentiates anything (section 36).
"""

from __future__ import annotations

from typing import Any

import torch

EPS = 1.0e-12


def normalize_axis(e: torch.Tensor, *, eps: float = EPS) -> torch.Tensor:
    """Unit axis, with the zero vector rejected rather than silently rescaled.

    A degenerate axis would make every downstream tangent zero and the reduced
    information singular, which is far harder to diagnose than an exception here.
    """
    e = torch.as_tensor(e).reshape(3)
    norm = e.norm()
    if float(norm.item()) <= eps:
        raise ValueError("prismatic axis has (near) zero length")
    return e / norm


def _as_scalar(q: Any, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    t = torch.as_tensor(q, device=device, dtype=dtype)
    if t.numel() != 1:
        raise ValueError(f"joint coordinate must be scalar, got {tuple(t.shape)}")
    return t.reshape(())


def prismatic_pose_tangent(e: torch.Tensor, n: int) -> torch.Tensor:
    """``G^P in R^{N x 6 x 1}`` with ``G_i = [e; 0]`` (section 29.1).

    Materialised rather than returned as a broadcast view: ``expand`` would alias
    one 6-vector across all ``N`` rows, and the consumers of ``G`` (reduced
    information, pixel-Jacobian contraction) are free to write into it.
    """
    e = normalize_axis(e)
    if n <= 0:
        raise ValueError("n must be positive")
    g = torch.zeros(n, 6, 1, device=e.device, dtype=e.dtype)
    g[:, :3, 0] = e
    return g


def prismatic_transport(
    mu0: torch.Tensor,
    R0: torch.Tensor | None,
    e: torch.Tensor,
    q: torch.Tensor | float,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """``mu = mu0 + q e``, ``R = R0`` (section 29.1).

    ``R0`` is returned unchanged (and un-copied): a prismatic joint does not
    rotate anything, and cloning an ``[N,3,3]`` buffer every frame for a million
    Gaussians is pure waste.  ``R0=None`` is accepted for the centres-only paths
    (axis fitting, residual evaluation) and propagates as ``None``.
    """
    mu0 = torch.as_tensor(mu0)
    if mu0.dim() != 2 or mu0.shape[1] != 3:
        raise ValueError(f"mu0 must be [N,3], got {tuple(mu0.shape)}")
    e = normalize_axis(torch.as_tensor(e, device=mu0.device, dtype=mu0.dtype))
    qq = _as_scalar(q, mu0.device, mu0.dtype)
    return mu0 + qq * e, R0


@torch.no_grad()
def fit_prismatic_axis(
    mu0: torch.Tensor,
    mu_t: torch.Tensor,
    weights: torch.Tensor | None = None,
    *,
    rank_tol: float = 1.0e-10,
) -> dict[str, Any]:
    """Recover ``(e, q)`` from the displacements of a prismatic part (29.1).

    Method: for an exact prismatic motion every displacement obeys
    ``d_i = mu_t_i - mu0_i = q e``, so the weighted displacement matrix
    ``sqrt(w) D`` is rank one and its leading right singular vector *is* ``e`` --
    to machine precision, with no centring and no normal-equation squaring of the
    conditioning.  The magnitude follows as the weighted mean of ``d_i . e``.

    Sign convention: ``e`` is only defined up to sign by the SVD, so it is flipped
    to make ``q >= 0``, i.e. ``e`` points along the observed motion.  The second
    singular value is reported because it is the honest measure of how prismatic
    the data actually was.
    """
    mu0 = torch.as_tensor(mu0)
    mu_t = torch.as_tensor(mu_t, device=mu0.device, dtype=mu0.dtype)
    if mu0.shape != mu_t.shape or mu0.dim() != 2 or mu0.shape[1] != 3:
        raise ValueError(f"expected matching [N,3], got {tuple(mu0.shape)} / {tuple(mu_t.shape)}")
    n = mu0.shape[0]
    if n == 0:
        raise ValueError("fit_prismatic_axis needs at least one Gaussian")
    device, dtype = mu0.device, mu0.dtype

    if weights is None:
        w = torch.ones(n, device=device, dtype=dtype)
    else:
        w = torch.as_tensor(weights, device=device, dtype=dtype).reshape(n).clamp_min(0.0)
    w_sum = w.sum()
    if float(w_sum.item()) <= 0.0:
        raise ValueError("all weights are zero")

    disp = mu_t - mu0
    scaled = w.sqrt()[:, None] * disp
    _, svals, vh = torch.linalg.svd(scaled, full_matrices=False)
    e = vh[0]

    proj = disp @ e                                   # [N] signed slide per point
    q = (w * proj).sum() / w_sum
    if float(q.item()) < 0.0:                         # keep e along the motion
        e, proj, q = -e, -proj, -q

    res = disp - q * e
    per_point = res.norm(dim=-1)
    residual = float(((w * per_point.square()).sum() / w_sum).clamp_min(0.0).sqrt().item())

    lead = float(svals[0].item())
    degenerate = lead <= rank_tol * float(scaled.abs().max().clamp_min(EPS).item()) * max(n, 1)
    return {
        "axis": e,
        "q": q,
        "per_point_q": proj,
        "residual": residual,
        "per_point_residual": per_point,
        "singular_values": svals,
        # 0 for an exactly prismatic motion; grows with any rotation or noise.
        "offaxis_ratio": float((svals[1] / max(lead, EPS)).item()) if svals.numel() > 1 else 0.0,
        "degenerate": bool(degenerate),
        "num_gaussians": n,
    }
