"""Uncertainty-aware fusion of RGB-D world tracks."""

from __future__ import annotations

import torch

from utils.rgbd_track_lifting import WorldTrack

EPS = 1.0e-9


def _inv(cov: torch.Tensor) -> torch.Tensor:
    eye = torch.eye(3, device=cov.device, dtype=cov.dtype)
    return torch.linalg.pinv(cov + eye * EPS)


def associate_multiview_track_observations(tracks: list[WorldTrack]) -> list[list[int]]:
    """Associate by covariance-normalised agreement, not by metric threshold."""
    if not tracks:
        return []
    centers = torch.stack([t.positions[t.visible].mean(0) if bool(t.visible.any()) else t.positions.mean(0) for t in tracks])
    covs = torch.stack([t.covariance[t.visible].mean(0) if bool(t.visible.any()) else t.covariance.mean(0) for t in tracks])
    score = torch.empty((len(tracks), len(tracks)))
    for i in range(len(tracks)):
        for j in range(len(tracks)):
            d = centers[i] - centers[j]
            score[i, j] = d @ _inv(covs[i] + covs[j]) @ d
    order = torch.argsort(score + torch.eye(len(tracks)) * 1.0e9, dim=1)[:, 0]
    groups: list[list[int]] = []
    used: set[int] = set()
    for i, j in enumerate(order.tolist()):
        if i in used:
            continue
        group = {i}
        if int(order[j]) == i:
            group.add(int(j))
        used |= group
        groups.append(sorted(group))
    return groups


def fuse_world_track_measurements(positions: torch.Tensor, covariance: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    pos = torch.as_tensor(positions).detach().float()
    cov = torch.as_tensor(covariance).detach().float()
    precision = torch.linalg.pinv(cov + torch.eye(3, device=pos.device)[None] * EPS)
    fused_cov = torch.linalg.pinv(precision.sum(0))
    fused_pos = fused_cov @ (precision @ pos[..., None]).sum(0).reshape(3)
    return fused_pos, fused_cov


def compute_multiview_track_consistency(tracks: list[WorldTrack]) -> dict[str, torch.Tensor | float]:
    if len(tracks) < 2:
        return {"pairwise_mahalanobis": torch.empty(0), "median_mahalanobis": float("nan")}
    values = []
    centers = [t.positions[t.visible].mean(0) if bool(t.visible.any()) else t.positions.mean(0) for t in tracks]
    covs = [t.covariance[t.visible].mean(0) if bool(t.visible.any()) else t.covariance.mean(0) for t in tracks]
    for i in range(len(tracks)):
        for j in range(i + 1, len(tracks)):
            d = centers[i] - centers[j]
            values.append(d @ _inv(covs[i] + covs[j]) @ d)
    out = torch.stack(values)
    return {"pairwise_mahalanobis": out, "median_mahalanobis": float(out.median())}
