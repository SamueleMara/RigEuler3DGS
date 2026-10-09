"""DINOv3 dense feature-map utilities for adaptive Eulerian 3DGS.

The renderer is intentionally not changed. DINOv3 features are loaded from
precomputed per-image tensors and aggregated into frozen per-Gaussian feature
vectors used by DINO-weighted rigidity and motion-grid refinement.

Expected per-image tensor layout:
    ROOT/dinov3/cam_000/frame_000000.pt

Each .pt file may contain either:
    Tensor [C, H, W]
    Tensor [H, W, C]
    dict with one of keys: features, dino, feat, feature_map
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Optional, Tuple

import torch
import torch.nn.functional as F


def _as_chw_feature_map(data, device="cuda") -> torch.Tensor:
    if isinstance(data, dict):
        for key in ["features", "dino", "feat", "feature_map", "x_norm_patchtokens"]:
            if key in data:
                data = data[key]
                break
    feat = torch.as_tensor(data, dtype=torch.float32, device=device)
    if feat.ndim == 4 and feat.shape[0] == 1:
        feat = feat[0]
    if feat.ndim != 3:
        raise ValueError(f"DINO feature map must be [C,H,W] or [H,W,C], got {tuple(feat.shape)}")
    # Heuristic: DINO channels are normally much larger than spatial dimensions.
    if feat.shape[0] <= 8 and feat.shape[-1] > 8:
        feat = feat.permute(2, 0, 1).contiguous()
    return feat.contiguous()


def load_feature_map(path: str, device="cuda") -> torch.Tensor:
    data = torch.load(path, map_location=device)
    feat = _as_chw_feature_map(data, device=device)
    return F.normalize(feat, dim=0)


def _camera_full_proj_transform(cam, device="cuda") -> torch.Tensor:
    if hasattr(cam, "full_proj_transform"):
        return cam.full_proj_transform.to(device=device, dtype=torch.float32)
    if hasattr(cam, "world_view_transform") and hasattr(cam, "projection_matrix"):
        return (cam.world_view_transform @ cam.projection_matrix).to(device=device, dtype=torch.float32)
    raise AttributeError("Camera must expose full_proj_transform or world_view_transform/projection_matrix")


def project_points_to_feature_grid(xyz: torch.Tensor, cam, eps: float = 1e-7) -> Tuple[torch.Tensor, torch.Tensor]:
    """Project world-space points to grid_sample coordinates.

    Returns:
        grid:  [1, N, 1, 2], normalized [-1,1] coordinates.
        valid: [N] mask inside the view frustum.
    """
    device = xyz.device
    ones = torch.ones((xyz.shape[0], 1), dtype=xyz.dtype, device=device)
    pts_h = torch.cat([xyz, ones], dim=-1)
    proj = _camera_full_proj_transform(cam, device=device)
    clip = pts_h @ proj
    w = clip[:, 3:4]
    ndc = clip[:, :3] / (w + eps)
    valid = (
        torch.isfinite(ndc).all(dim=-1)
        & (w.squeeze(-1) > eps)
        & (ndc[:, 0] >= -1.0) & (ndc[:, 0] <= 1.0)
        & (ndc[:, 1] >= -1.0) & (ndc[:, 1] <= 1.0)
        & (ndc[:, 2] >= 0.0) & (ndc[:, 2] <= 1.0)
    )
    grid = ndc[:, :2].view(1, xyz.shape[0], 1, 2)
    return grid, valid


@torch.no_grad()
def build_gaussian_dino_features_from_cameras(
    xyz: torch.Tensor,
    cameras: Iterable,
    frame_id: int = 0,
    max_cameras: int = 0,
    device: str = "cuda",
) -> Optional[torch.Tensor]:
    """Aggregate dense DINO maps into one frozen descriptor per Gaussian.

    The default uses only frame_id=0 because the canonical 3DGS lives at t0.
    This gives stable descriptors for rigidity without using DINO as a direct
    motion target. Set frame_id=-1 to aggregate all available feature maps.
    """
    xyz = xyz.detach().to(device=device, dtype=torch.float32)
    accum = None
    counts = torch.zeros((xyz.shape[0], 1), dtype=torch.float32, device=device)
    used = 0

    for cam in cameras:
        path = getattr(cam, "dino_feature_path", "")
        if not path or not Path(path).exists():
            continue
        if frame_id >= 0 and int(getattr(cam, "frame_id", -1)) != int(frame_id):
            continue
        feat_map = load_feature_map(path, device=device)  # [C,H,W]
        C = feat_map.shape[0]
        if accum is None:
            accum = torch.zeros((xyz.shape[0], C), dtype=torch.float32, device=device)
        elif accum.shape[1] != C:
            raise ValueError(f"Inconsistent DINO feature dimensions: {accum.shape[1]} vs {C} for {path}")

        grid, valid = project_points_to_feature_grid(xyz, cam)
        if valid.any():
            sampled = F.grid_sample(
                feat_map.unsqueeze(0),
                grid,
                mode="bilinear",
                padding_mode="zeros",
                align_corners=True,
            )[0, :, :, 0].transpose(0, 1)  # [N,C]
            sampled = sampled[valid]
            accum[valid] += sampled
            counts[valid] += 1.0
            used += 1
            if max_cameras > 0 and used >= max_cameras:
                break

    if accum is None or used == 0:
        return None
    feat = accum / counts.clamp_min(1.0)
    feat = F.normalize(feat, dim=-1)
    feat[counts.squeeze(-1) <= 0] = 0.0
    return feat.detach()
