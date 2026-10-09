from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from plyfile import PlyData

from .moments import compute_global_com, compute_global_inertia, compute_soft_inertia


def load_ply_xyz(path: str | Path) -> torch.Tensor:
    data = PlyData.read(str(Path(path).expanduser()))["vertex"].data
    return torch.from_numpy(np.stack((data["x"], data["y"], data["z"]), axis=1).astype("float32"))


def _nearest_squared(query: torch.Tensor, reference: torch.Tensor, chunk_size: int) -> torch.Tensor:
    q = torch.as_tensor(query)
    r = torch.as_tensor(reference, dtype=q.dtype, device=q.device)
    out = []
    chunk = max(1, int(chunk_size))
    for start in range(0, int(q.shape[0]), chunk):
        q_chunk = q[start:start + chunk]
        best = torch.full((q_chunk.shape[0],), float("inf"), dtype=q.dtype, device=q.device)
        for ref_start in range(0, int(r.shape[0]), chunk):
            dist = torch.cdist(q_chunk, r[ref_start:ref_start + chunk]).square().min(dim=1).values
            best = torch.minimum(best, dist)
        out.append(best)
    return torch.cat(out, dim=0)


def compute_counterfactual_endpoint_positions(
    canonical_xyz: torch.Tensor,
    final_rotation: torch.Tensor,
    final_translation: torch.Tensor,
) -> torch.Tensor:
    """Candidate final positions [N,K+1,3] for static identity + moving parts."""
    xyz = torch.as_tensor(canonical_xyz)
    rotation = torch.as_tensor(final_rotation, dtype=xyz.dtype, device=xyz.device)
    translation = torch.as_tensor(final_translation, dtype=xyz.dtype, device=xyz.device)
    candidates = [xyz]
    for part in range(1, int(rotation.shape[0])):
        candidates.append(xyz @ rotation[part].transpose(0, 1) + translation[part])
    return torch.stack(candidates, dim=1)


def compute_counterfactual_endpoint_unary(
    canonical_xyz: torch.Tensor,
    final_rotation: torch.Tensor,
    final_translation: torch.Tensor,
    final_xyz: torch.Tensor,
    *,
    chunk_size: int = 2048,
    scale: torch.Tensor | float | None = None,
) -> torch.Tensor:
    """Per-Gaussian endpoint energy [N,K+1] for label discovery.

    Each class asks where the Gaussian would land at the final frame if it
    belonged to that class, then measures nearest final-3DGS distance.
    """
    candidates = compute_counterfactual_endpoint_positions(canonical_xyz, final_rotation, final_translation)
    final = torch.as_tensor(final_xyz, dtype=candidates.dtype, device=candidates.device)
    energies = []
    for part in range(int(candidates.shape[1])):
        energies.append(_nearest_squared(candidates[:, part], final, chunk_size))
    out = torch.stack(energies, dim=-1)
    if scale is None:
        diag = torch.linalg.norm(final.max(dim=0).values - final.min(dim=0).values).clamp_min(1.0e-8)
    else:
        diag = torch.as_tensor(scale, dtype=out.dtype, device=out.device).clamp_min(1.0e-8)
    return out / diag.square()


def endpoint_chamfer_loss(
    predicted_xyz: torch.Tensor,
    final_xyz: torch.Tensor,
    *,
    chunk_size: int = 2048,
) -> torch.Tensor:
    pred = torch.as_tensor(predicted_xyz)
    final = torch.as_tensor(final_xyz, dtype=pred.dtype, device=pred.device)
    return _nearest_squared(pred, final, chunk_size).mean() + _nearest_squared(final, pred, chunk_size).mean()


def assign_final_gaussians_to_parts(
    final_xyz: torch.Tensor,
    predicted_xyz: torch.Tensor,
    part_prob: torch.Tensor,
    *,
    temperature: float = 0.01,
    strong_threshold: float = 0.5,
    chunk_size: int = 2048,
) -> torch.Tensor:
    final = torch.as_tensor(final_xyz)
    pred = torch.as_tensor(predicted_xyz, dtype=final.dtype, device=final.device)
    prob = torch.as_tensor(part_prob, dtype=final.dtype, device=final.device)
    k1 = int(prob.shape[1])
    distances = []
    hard = prob.argmax(dim=-1)
    confidence = prob.max(dim=-1).values
    for part in range(k1):
        mask = (hard == part) & (confidence >= float(strong_threshold))
        if not bool(mask.any()):
            mask = hard == part
        if not bool(mask.any()):
            distances.append(torch.full((final.shape[0],), float("inf"), dtype=final.dtype, device=final.device))
            continue
        distances.append(_nearest_squared(final, pred[mask], chunk_size))
    dist = torch.stack(distances, dim=-1)
    finite = torch.isfinite(dist).any(dim=-1, keepdim=True)
    fallback = torch.zeros_like(dist)
    fallback[:, 0] = 1.0
    q = torch.softmax(-dist / max(float(temperature), 1.0e-8), dim=-1)
    return torch.where(finite, q, fallback)


def final_part_moment_losses(
    predicted_final_xyz: torch.Tensor,
    final_xyz: torch.Tensor,
    predicted_part_prob: torch.Tensor,
    final_part_prob: torch.Tensor,
    *,
    min_mass: float = 3.0,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, float]]:
    pred = torch.as_tensor(predicted_final_xyz)
    final = torch.as_tensor(final_xyz, dtype=pred.dtype, device=pred.device)
    p = torch.as_tensor(predicted_part_prob, dtype=pred.dtype, device=pred.device)
    q = torch.as_tensor(final_part_prob, dtype=pred.dtype, device=pred.device)
    pred_j, pred_c = compute_soft_inertia(pred, p, normalize=True)
    final_j, final_c = compute_soft_inertia(final, q, normalize=True)
    mass = p.sum(dim=0).minimum(q.sum(dim=0))
    valid = mass > float(min_mass)
    if not bool(valid.any()):
        zero = pred.new_tensor(0.0)
        return zero, zero, {"valid_parts": 0}
    com_loss = (pred_c[valid] - final_c[valid]).square().sum(dim=-1).mean()
    inertia_loss = (pred_j[valid] - final_j[valid]).square().sum(dim=(-1, -2)).mean()
    return com_loss, inertia_loss, {"valid_parts": int(valid.sum())}


def global_endpoint_moment_losses(
    predicted_final_xyz: torch.Tensor,
    final_xyz: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    pred = torch.as_tensor(predicted_final_xyz)
    final = torch.as_tensor(final_xyz, dtype=pred.dtype, device=pred.device)
    c_pred = compute_global_com(pred)
    c_final = compute_global_com(final)
    j_pred = compute_global_inertia(pred, c_pred, normalize=True)
    j_final = compute_global_inertia(final, c_final, normalize=True)
    return (c_pred - c_final).square().sum(), (j_pred - j_final).square().sum()
