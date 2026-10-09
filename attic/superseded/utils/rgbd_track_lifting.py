"""Depth lifting for semantic foreground pixel tracks."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch


@dataclass
class WorldTrack:
    positions: torch.Tensor
    covariance: torch.Tensor
    visible: torch.Tensor
    frame_ids: torch.Tensor
    timestamps: torch.Tensor
    pixel_indices: torch.Tensor | None = None
    stats: dict[str, Any] = field(default_factory=dict)


def backproject_pixel_with_depth(pixel_uv: torch.Tensor, depth: torch.Tensor, intrinsics: torch.Tensor) -> torch.Tensor:
    uv = torch.as_tensor(pixel_uv).detach().float()
    d = torch.as_tensor(depth).detach().float().reshape(-1)
    K = torch.as_tensor(intrinsics).detach().float()
    ones = torch.ones((uv.shape[0], 1), dtype=uv.dtype, device=uv.device)
    homog = torch.cat((uv, ones), dim=1)
    rays = torch.linalg.solve(K.to(uv), homog.T).T
    return rays * d[:, None]


def transform_camera_to_world(points_cam: torch.Tensor, camera_to_world: torch.Tensor) -> torch.Tensor:
    pts = torch.as_tensor(points_cam).detach().float()
    T = torch.as_tensor(camera_to_world).detach().float().to(pts)
    return pts @ T[:3, :3].T + T[:3, 3]


def estimate_track_covariance(world_positions: torch.Tensor, visible: torch.Tensor, *, depth_sigma: float = 0.005) -> torch.Tensor:
    pos = torch.as_tensor(world_positions).detach().float()
    vis = torch.as_tensor(visible).detach().bool()
    base = max(float(depth_sigma), 1.0e-6) ** 2
    cov = torch.eye(3, dtype=pos.dtype, device=pos.device).expand(pos.shape[0], 3, 3).clone() * base
    if int(vis.sum()) > 1:
        speed = torch.linalg.norm(pos[1:] - pos[:-1], dim=-1)
        jitter = torch.zeros(pos.shape[0], device=pos.device, dtype=pos.dtype)
        jitter[1:] = speed
        cov = cov + torch.eye(3, device=pos.device, dtype=pos.dtype)[None] * jitter[:, None, None].square()
    cov[~vis] = torch.eye(3, dtype=pos.dtype, device=pos.device) * 1.0e6
    return cov


def propagate_pixel_track_to_world_track(
    pixel_uv: torch.Tensor,
    depth_values: torch.Tensor,
    intrinsics: torch.Tensor,
    camera_to_world: torch.Tensor,
    *,
    frame_ids: torch.Tensor | None = None,
    timestamps: torch.Tensor | None = None,
    depth_sigma: float = 0.005,
) -> WorldTrack:
    pix = torch.as_tensor(pixel_uv).detach().float()
    depth = torch.as_tensor(depth_values).detach().float().reshape(-1)
    Tcw = torch.as_tensor(camera_to_world).detach().float()
    if Tcw.ndim == 2:
        Tcw = Tcw.expand(pix.shape[0], 4, 4)
    visible = torch.isfinite(depth) & (depth > 0)
    cam = backproject_pixel_with_depth(pix, torch.where(visible, depth, torch.ones_like(depth)), intrinsics)
    world = torch.stack([transform_camera_to_world(cam[i:i + 1], Tcw[i])[0] for i in range(int(pix.shape[0]))])
    cov = estimate_track_covariance(world, visible, depth_sigma=depth_sigma)
    ids = torch.arange(pix.shape[0]) if frame_ids is None else torch.as_tensor(frame_ids).long()
    ts = ids.float() if timestamps is None else torch.as_tensor(timestamps).float()
    return WorldTrack(world, cov, visible, ids, ts, stats={"visible_frames": int(visible.sum()), "track_length": int(pix.shape[0])})
