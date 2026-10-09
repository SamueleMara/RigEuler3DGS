"""Compare latent endpoint-FM trajectories with RGB-D world tracks."""

from __future__ import annotations

import torch

from utils.bidirectional_fm import FlowTrainConfig
from utils.rgbd_track_lifting import WorldTrack

EPS = 1.0e-9


def attach_video_tracks_to_source_gaussians(source_ownership: torch.Tensor, track_source_pixels: torch.Tensor) -> torch.Tensor:
    ownership = torch.as_tensor(source_ownership).detach().float()
    pixels = torch.as_tensor(track_source_pixels).long()
    return ownership[pixels]


@torch.no_grad()
def sample_fm_trajectory_at_video_times(model, source_xyz: torch.Tensor, times: torch.Tensor, cfg: FlowTrainConfig, *, device: str | torch.device = "cuda") -> torch.Tensor:
    dev = torch.device(device if torch.cuda.is_available() or str(device) == "cpu" else "cpu")
    model = model.to(dev).eval()
    x0 = torch.as_tensor(source_xyz).detach().float().to(dev)
    ts = torch.as_tensor(times).detach().float().to(dev).clamp(0.0, 1.0)
    out = []
    steps = max(1, int(cfg.integration_steps))
    for tau_end in ts.tolist():
        x = x0.clone()
        dt = float(tau_end) / float(steps)
        for step in range(steps):
            tau = torch.full((x.shape[0],), float(step) * dt, device=dev).clamp(0.0, 1.0)
            x = x + dt * model(x, tau)
        out.append(x.detach().cpu())
    return torch.stack(out, dim=0)


def compare_latent_trajectory_to_rgbd_track(predicted: torch.Tensor, track: WorldTrack) -> dict[str, torch.Tensor | float]:
    pred = torch.as_tensor(predicted).detach().float()
    obs = track.positions.to(pred)
    cov = track.covariance.to(pred)
    n = min(pred.shape[0], obs.shape[0])
    diff = pred[:n] - obs[:n]
    inv = torch.linalg.pinv(cov[:n] + torch.eye(3, device=pred.device)[None] * EPS)
    maha = torch.einsum("ti,tij,tj->t", diff, inv, diff)
    visible = track.visible[:n].to(maha.device).bool()
    return {"mahalanobis": maha[visible], "median_mahalanobis": float(maha[visible].median()) if bool(visible.any()) else float("nan")}


def project_fm_trajectory_to_all_cameras(trajectory: torch.Tensor, intrinsics: torch.Tensor, world_to_camera: torch.Tensor) -> torch.Tensor:
    traj = torch.as_tensor(trajectory).detach().float()
    K = torch.as_tensor(intrinsics).detach().float().to(traj)
    W2C = torch.as_tensor(world_to_camera).detach().float().to(traj)
    if W2C.ndim == 2:
        W2C = W2C[None]
    cams = []
    for T in W2C:
        cam = traj @ T[:3, :3].T + T[:3, 3]
        pix_h = cam @ K.T
        cams.append(pix_h[..., :2] / pix_h[..., 2:3].clamp_min(EPS))
    return torch.stack(cams, dim=0)
