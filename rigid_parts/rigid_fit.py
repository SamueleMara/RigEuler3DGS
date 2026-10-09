from __future__ import annotations

import torch


def skew_matrix(vector: torch.Tensor) -> torch.Tensor:
    v = torch.as_tensor(vector)
    z = torch.zeros_like(v[..., 0])
    return torch.stack(
        (
            torch.stack((z, -v[..., 2], v[..., 1]), dim=-1),
            torch.stack((v[..., 2], z, -v[..., 0]), dim=-1),
            torch.stack((-v[..., 1], v[..., 0], z), dim=-1),
        ),
        dim=-2,
    )


def weighted_kabsch(
    xyz0: torch.Tensor,
    xyzt: torch.Tensor,
    weights: torch.Tensor,
    eps: float = 1.0e-8,
) -> tuple[torch.Tensor, torch.Tensor]:
    source = torch.as_tensor(xyz0)
    target = torch.as_tensor(xyzt, dtype=source.dtype, device=source.device)
    w = torch.as_tensor(weights, dtype=source.dtype, device=source.device).reshape(-1).clamp_min(0.0)
    mass = w.sum().clamp_min(float(eps))
    c0 = (source * w[:, None]).sum(dim=0) / mass
    ct = (target * w[:, None]).sum(dim=0) / mass
    x = source - c0[None]
    y = target - ct[None]
    h = x.transpose(0, 1) @ (y * w[:, None])
    u, _s, vh = torch.linalg.svd(h)
    v = vh.transpose(-1, -2)
    det = torch.det(v @ u.transpose(-1, -2))
    fix = torch.ones(3, dtype=source.dtype, device=source.device)
    fix[-1] = torch.where(det < 0.0, -1.0, 1.0)
    r = v @ torch.diag(fix) @ u.transpose(-1, -2)
    t = ct - r @ c0
    return r, t


def fit_weighted_twist(
    xyz: torch.Tensor,
    velocity: torch.Tensor,
    weights: torch.Tensor,
    eps: float = 1.0e-6,
) -> tuple[torch.Tensor, torch.Tensor]:
    points = torch.as_tensor(xyz)
    vel = torch.as_tensor(velocity, dtype=points.dtype, device=points.device)
    w = torch.as_tensor(weights, dtype=points.dtype, device=points.device).reshape(-1).clamp_min(0.0)
    a = torch.zeros((points.shape[0], 3, 6), dtype=points.dtype, device=points.device)
    a[:, :, :3] = -skew_matrix(points)
    a[:, :, 3:] = torch.eye(3, dtype=points.dtype, device=points.device)[None]
    aw = a.reshape(-1, 6) * w[:, None].sqrt().repeat_interleave(3, dim=0)
    bw = vel.reshape(-1) * w.sqrt().repeat_interleave(3)
    normal = aw.transpose(0, 1) @ aw + float(eps) * torch.eye(6, dtype=points.dtype, device=points.device)
    rhs = aw.transpose(0, 1) @ bw
    solution = torch.linalg.solve(normal, rhs)
    return solution[:3], solution[3:]
