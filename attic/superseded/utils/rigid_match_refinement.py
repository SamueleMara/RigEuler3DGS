"""Standalone rigid SE(3) fitting for bidirectional matching.

This module deliberately does not import the progressive-rigid matcher.  The
bidirectional probe needs a clean correspondence -> SE(3) block whose inputs are
explicit reciprocal pairs, not FM endpoints pretending to be point matches.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any

import torch


def transform_points(T: torch.Tensor, points: torch.Tensor) -> torch.Tensor:
    T = torch.as_tensor(T, device=points.device, dtype=points.dtype)
    return points @ T[:3, :3].T + T[:3, 3]


def _skew(vector: torch.Tensor) -> torch.Tensor:
    x, y, z = vector.unbind(-1)
    zero = torch.zeros_like(x)
    return torch.stack(
        (
            torch.stack((zero, -z, y), dim=-1),
            torch.stack((z, zero, -x), dim=-1),
            torch.stack((-y, x, zero), dim=-1),
        ),
        dim=-2,
    )


def _so3_log(R: torch.Tensor) -> torch.Tensor:
    trace = R.diagonal(offset=0, dim1=-2, dim2=-1).sum(-1)
    cos = ((trace - 1.0) * 0.5).clamp(-1.0, 1.0)
    theta = torch.acos(cos)
    vee = torch.stack(
        (R[..., 2, 1] - R[..., 1, 2], R[..., 0, 2] - R[..., 2, 0], R[..., 1, 0] - R[..., 0, 1]),
        dim=-1,
    )
    small = theta < 1.0e-5
    scale = theta / (2.0 * torch.sin(theta).clamp_min(1.0e-8))
    return torch.where(small[..., None], 0.5 * vee, scale[..., None] * vee)


def se3_log(T: torch.Tensor) -> torch.Tensor:
    T = torch.as_tensor(T)
    R = T[..., :3, :3]
    t = T[..., :3, 3]
    w = _so3_log(R)
    theta = torch.linalg.norm(w, dim=-1)
    W = _skew(w)
    I = torch.eye(3, device=T.device, dtype=T.dtype).expand(T.shape[:-2] + (3, 3))
    theta2 = theta.square()
    coef = (
        (1.0 / theta2.clamp_min(1.0e-12))
        - (1.0 + torch.cos(theta)) / (2.0 * theta * torch.sin(theta).clamp_min(1.0e-8))
    )
    small = theta < 1.0e-5
    Vinv = I - 0.5 * W + coef[..., None, None] * (W @ W)
    Vinv_small = I - 0.5 * W + (1.0 / 12.0) * (W @ W)
    v = torch.einsum("...ij,...j->...i", torch.where(small[..., None, None], Vinv_small, Vinv), t)
    return torch.cat((v, w), dim=-1)


def se3_delta_norm(previous: torch.Tensor, current: torch.Tensor) -> float:
    delta = torch.linalg.inv(torch.as_tensor(previous).to(current)) @ current
    return float(torch.linalg.norm(se3_log(delta)).detach().cpu())


def transform_summary(T: torch.Tensor) -> dict[str, Any]:
    T = torch.as_tensor(T).detach().cpu().float()
    xi = se3_log(T)
    angle = float(torch.linalg.norm(xi[3:]))
    axis = torch.zeros(3)
    if angle > 1.0e-8:
        axis = xi[3:] / angle
    return {
        "rotation_angle_deg": math.degrees(angle),
        "rotation_axis": [float(v) for v in axis],
        "translation": [float(v) for v in T[:3, 3]],
        "det_R": float(torch.det(T[:3, :3])),
    }


def huber_weight(residual: torch.Tensor, delta: float) -> torch.Tensor:
    delta = max(float(delta), 1.0e-8)
    residual = torch.as_tensor(residual).clamp_min(0.0)
    return torch.where(residual <= delta, torch.ones_like(residual), delta / residual.clamp_min(1.0e-8))


def weighted_kabsch(source: torch.Tensor, target: torch.Tensor, weight: torch.Tensor | None = None) -> dict[str, torch.Tensor]:
    source = torch.as_tensor(source).float()
    target = torch.as_tensor(target, device=source.device, dtype=source.dtype)
    if source.ndim != 2 or source.shape != target.shape or source.shape[1] != 3:
        raise ValueError("source and target must be matching [N,3]")
    if int(source.shape[0]) < 3:
        raise ValueError("weighted_kabsch needs at least three pairs")
    if weight is None:
        weight = torch.ones(source.shape[0], device=source.device, dtype=source.dtype)
    weight = torch.as_tensor(weight, device=source.device, dtype=source.dtype).reshape(-1).clamp_min(0.0)
    valid = torch.isfinite(source).all(-1) & torch.isfinite(target).all(-1) & torch.isfinite(weight) & (weight > 0)
    if int(valid.sum()) < 3:
        raise ValueError("weighted_kabsch has fewer than three valid weighted pairs")
    x = source[valid]
    y = target[valid]
    w = weight[valid]
    w = w / w.sum().clamp_min(1.0e-8)
    xbar = (w[:, None] * x).sum(0)
    ybar = (w[:, None] * y).sum(0)
    H = (x - xbar).T @ (w[:, None] * (y - ybar))
    U, _, Vh = torch.linalg.svd(H)
    R = Vh.T @ U.T
    if torch.det(R) < 0:
        Vh = Vh.clone()
        Vh[-1] *= -1
        R = Vh.T @ U.T
    t = ybar - R @ xbar
    T = torch.eye(4, device=source.device, dtype=source.dtype)
    T[:3, :3] = R
    T[:3, 3] = t
    residual = torch.linalg.norm(transform_points(T, source) - target, dim=-1)
    return {"T": T, "residual": residual, "valid": valid}


@dataclass
class RigidFitConfig:
    irls_iters: int = 8
    huber_delta: float = 0.02
    min_extent_ratio: float = 2.0


def _spatial_extent(points: torch.Tensor) -> torch.Tensor:
    if points.numel() == 0:
        return points.new_tensor(0.0)
    center = points.mean(0, keepdim=True)
    return torch.linalg.norm(points - center, dim=-1).median()


def check_rigid_degeneracy(source: torch.Tensor, target: torch.Tensor, scale: float) -> dict[str, Any]:
    if int(source.shape[0]) < 3:
        return {"ok": False, "reason": "fewer_than_three_pairs"}
    scale_t = torch.as_tensor(max(float(scale), 1.0e-8), device=source.device, dtype=source.dtype)
    src_extent = _spatial_extent(source)
    tgt_extent = _spatial_extent(target)
    src_rank = int((torch.linalg.svdvals((source - source.mean(0)).T @ (source - source.mean(0))) > scale_t.square()).sum())
    tgt_rank = int((torch.linalg.svdvals((target - target.mean(0)).T @ (target - target.mean(0))) > scale_t.square()).sum())
    ok = bool(src_extent > scale_t * 2.0 and tgt_extent > scale_t * 2.0 and src_rank >= 2 and tgt_rank >= 2)
    return {
        "ok": ok,
        "reason": "ok" if ok else "degenerate_extent_or_rank",
        "source_extent": float(src_extent.detach().cpu()),
        "target_extent": float(tgt_extent.detach().cpu()),
        "source_rank": src_rank,
        "target_rank": tgt_rank,
    }


def robust_rigid_fit(
    source: torch.Tensor,
    target: torch.Tensor,
    weight: torch.Tensor,
    *,
    scale: float,
    cfg: RigidFitConfig | None = None,
    init_T: torch.Tensor | None = None,
) -> dict[str, Any]:
    cfg = cfg or RigidFitConfig()
    source = torch.as_tensor(source).float()
    target = torch.as_tensor(target, device=source.device, dtype=source.dtype)
    weight = torch.as_tensor(weight, device=source.device, dtype=source.dtype).reshape(-1).clamp_min(0.0)
    deg = check_rigid_degeneracy(source[weight > 0], target[weight > 0], scale)
    if not deg["ok"]:
        raise RuntimeError(f"degenerate rigid pair set: {deg['reason']}")
    base_weight = weight.clone()
    if init_T is not None:
        residual = torch.linalg.norm(transform_points(init_T.to(source), source) - target, dim=-1)
        weight = base_weight * huber_weight(residual, cfg.huber_delta)
    history = []
    fit = None
    for iteration in range(max(1, int(cfg.irls_iters))):
        fit = weighted_kabsch(source, target, weight)
        residual = fit["residual"]
        active = weight > 0
        history.append({
            "iteration": int(iteration),
            "active_pairs": int(active.sum()),
            "residual_mean": float(residual[active].mean().detach().cpu()) if bool(active.any()) else float("inf"),
            "residual_median": float(residual[active].median().detach().cpu()) if bool(active.any()) else float("inf"),
            "residual_p90": float(torch.quantile(residual[active], 0.9).detach().cpu()) if int(active.sum()) > 1 else float("inf"),
            "weight_sum": float(weight.sum().detach().cpu()),
        })
        weight = base_weight * huber_weight(residual, cfg.huber_delta)
    assert fit is not None
    residual = fit["residual"]
    active = weight > 0
    rms = torch.sqrt((weight * residual.square()).sum() / weight.sum().clamp_min(1.0e-8))
    return {
        "T": fit["T"],
        "residual": residual.detach(),
        "robust_weight": weight.detach(),
        "weighted_rms": float(rms.detach().cpu()),
        "robust_inliers": int(active.sum()),
        "history": history,
        "degeneracy": deg,
        "summary": transform_summary(fit["T"]),
    }
