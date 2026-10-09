from __future__ import annotations

import torch


def _eye(dtype: torch.dtype, device: torch.device) -> torch.Tensor:
    return torch.eye(3, dtype=dtype, device=device)


def compute_soft_mass(part_prob: torch.Tensor) -> torch.Tensor:
    return torch.as_tensor(part_prob).sum(dim=0)


def compute_soft_com(
    xyz: torch.Tensor,
    part_prob: torch.Tensor,
    eps: float = 1.0e-8,
) -> torch.Tensor:
    points = torch.as_tensor(xyz)
    prob = torch.as_tensor(part_prob, dtype=points.dtype, device=points.device)
    mass = prob.sum(dim=0).clamp_min(float(eps))
    return prob.transpose(0, 1) @ points / mass[:, None]


def compute_soft_inertia(
    xyz: torch.Tensor,
    part_prob: torch.Tensor,
    com: torch.Tensor | None = None,
    normalize: bool = True,
    eps: float = 1.0e-8,
) -> tuple[torch.Tensor, torch.Tensor]:
    points = torch.as_tensor(xyz)
    prob = torch.as_tensor(part_prob, dtype=points.dtype, device=points.device)
    center = compute_soft_com(points, prob, eps=eps) if com is None else torch.as_tensor(com, dtype=points.dtype, device=points.device)
    rel = points[:, None, :] - center[None, :, :]
    rr = rel[..., :, None] * rel[..., None, :]
    r2 = rel.square().sum(dim=-1)
    term = r2[..., None, None] * _eye(points.dtype, points.device) - rr
    inertia = (prob[:, :, None, None] * term).sum(dim=0)
    if bool(normalize):
        inertia = inertia / prob.sum(dim=0).clamp_min(float(eps))[:, None, None]
    inertia = 0.5 * (inertia + inertia.transpose(-1, -2))
    return inertia, center


def compute_global_com(xyz: torch.Tensor) -> torch.Tensor:
    return torch.as_tensor(xyz).mean(dim=0)


def compute_global_inertia(
    xyz: torch.Tensor,
    com: torch.Tensor | None = None,
    normalize: bool = True,
    eps: float = 1.0e-8,
) -> torch.Tensor:
    points = torch.as_tensor(xyz)
    center = compute_global_com(points) if com is None else torch.as_tensor(com, dtype=points.dtype, device=points.device)
    rel = points - center[None]
    term = rel.square().sum(dim=-1)[:, None, None] * _eye(points.dtype, points.device) - rel[:, :, None] * rel[:, None, :]
    inertia = term.sum(dim=0)
    if bool(normalize):
        inertia = inertia / max(int(points.shape[0]), 1)
    return 0.5 * (inertia + inertia.T)


def inertia_invariants(inertia: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    j = torch.as_tensor(inertia)
    trace = torch.diagonal(j, dim1=-2, dim2=-1).sum(dim=-1)
    trace_j2 = torch.diagonal(j @ j, dim1=-2, dim2=-1).sum(dim=-1)
    return trace, 0.5 * (trace.square() - trace_j2)


def parallel_axis_global_inertia(
    part_inertia: torch.Tensor,
    mass: torch.Tensor,
    part_com: torch.Tensor,
    global_com: torch.Tensor,
    *,
    normalized_part_inertia: bool = True,
    normalize: bool = True,
    eps: float = 1.0e-8,
) -> torch.Tensor:
    inertia = torch.as_tensor(part_inertia)
    m = torch.as_tensor(mass, dtype=inertia.dtype, device=inertia.device).clamp_min(0.0)
    com = torch.as_tensor(part_com, dtype=inertia.dtype, device=inertia.device)
    cg = torch.as_tensor(global_com, dtype=inertia.dtype, device=inertia.device)
    unnormalized = inertia * m[:, None, None] if bool(normalized_part_inertia) else inertia
    rel = com - cg[None]
    shift = rel.square().sum(dim=-1)[:, None, None] * _eye(inertia.dtype, inertia.device) - rel[:, :, None] * rel[:, None, :]
    total = (unnormalized + m[:, None, None] * shift).sum(dim=0)
    if bool(normalize):
        total = total / m.sum().clamp_min(float(eps))
    return 0.5 * (total + total.T)
