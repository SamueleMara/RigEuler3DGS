"""§6 -- extremely confident static anchors, from initial/final PCD only.

§5 forbids every other initialisation source: no legacy K-part, no
dynamic_prior.pt, no previous responsibilities, no DINO/SAM labels, no old
velocity clustering.  The only information allowed is G0 <-> GT.

The set optimises PRECISION, not recall (§6), and the asymmetry is
deliberate: an anchor is pinned to class 0 and can be hard-frozen, so a wrong
anchor is uncorrectable, while an omitted true-static point is exactly what
§36's candidate competition exists to recover.  Being strict here is what
makes "fixed static motion != fixed static membership" (§4)实 true.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch

from rigid_parts.endpoint import load_ply_xyz


@dataclass
class StaticAnchorResult:
    """Which Gaussians are confidently static, and what made them qualify."""

    mask: torch.Tensor            # [N] bool
    target_index: torch.Tensor    # [N] nearest final-PCD index
    distance: torch.Tensor        # [N] normalised NN distance
    threshold: torch.Tensor       # scalar, the accepted normalised distance
    bbox_diag: torch.Tensor       # scalar, initial-cloud bounding box diagonal
    mutual: torch.Tensor          # [N] bool, reciprocal NN held
    neighborhood_ok: torch.Tensor  # [N] bool, local structure preserved
    summary: dict[str, Any] = field(default_factory=dict)


@torch.no_grad()
def _chunked_argmin(query: torch.Tensor, reference: torch.Tensor, chunk_size: int = 2048):
    """Nearest reference index and distance, chunked.

    Never an ``N x M`` matrix: on the larger scenes that is billions of
    entries, and the spec's own §29 makes the same demand for the outer NN.
    """
    idx, dist = [], []
    for start in range(0, int(query.shape[0]), max(1, int(chunk_size))):
        block = torch.cdist(query[start:start + int(chunk_size)], reference)
        best = block.min(dim=1)
        idx.append(best.indices)
        dist.append(best.values)
    return torch.cat(idx, 0), torch.cat(dist, 0)


@torch.no_grad()
def _knn_signature(points: torch.Tensor, k: int, chunk_size: int = 2048):
    """``(neighbour indices, neighbour distances)`` excluding self."""
    idx, dist = [], []
    kk = min(int(k) + 1, int(points.shape[0]))
    for start in range(0, int(points.shape[0]), max(1, int(chunk_size))):
        block = torch.cdist(points[start:start + int(chunk_size)], points)
        top = torch.topk(block, k=kk, dim=1, largest=False)
        idx.append(top.indices[:, 1:])
        dist.append(top.values[:, 1:])
    return torch.cat(idx, 0), torch.cat(dist, 0)


@torch.no_grad()
def extract_confident_static_anchors(
    xyz_initial: torch.Tensor,
    xyz_final: torch.Tensor,
    *,
    distance_ratio: float = 0.002,
    mutual_nn: bool = True,
    use_neighborhood_check: bool = False,
    neighborhood_k: int = 8,
    neighborhood_tolerance: float = 0.25,
    chunk_size: int = 2048,
) -> StaticAnchorResult:
    """§6's tests: reciprocal NN, strict normalised distance, local structure.

    ``distance_ratio`` multiplies the bounding-box diagonal rather than being
    a metric constant, so the same strictness transfers between scenes of
    different physical size without retuning.
    """
    x0 = torch.as_tensor(xyz_initial).detach().float()
    yT = torch.as_tensor(xyz_final).detach().float().to(x0.device)
    n = int(x0.shape[0])

    # Centred before cdist: a COLMAP cloud far from the origin loses exactly
    # the float32 precision this strict threshold depends on.
    origin = x0.mean(dim=0, keepdim=True)
    a, b = x0 - origin, yT - origin
    bbox = torch.linalg.norm(a.max(dim=0).values - a.min(dim=0).values)

    forward_idx, forward_dist = _chunked_argmin(a, b, chunk_size)
    normalised = forward_dist / bbox.clamp_min(1.0e-8)
    close = normalised < float(distance_ratio)

    if bool(mutual_nn):
        backward_idx, _ = _chunked_argmin(b, a, chunk_size)
        mutual = backward_idx[forward_idx] == torch.arange(n, device=x0.device)
    else:
        mutual = torch.ones(n, dtype=torch.bool, device=x0.device)

    if bool(use_neighborhood_check):
        # A point can land in the same place while its neighbourhood deformed
        # around it. Requiring the local edge lengths to survive makes the
        # anchor a statement about the BODY being static rather than about one
        # coordinate happening to coincide.
        knn_idx, source_edges = _knn_signature(a, int(neighborhood_k), chunk_size)
        matched = b[forward_idx]
        target_edges = torch.linalg.norm(matched[knn_idx] - matched[:, None, :], dim=-1)
        relative = (target_edges - source_edges).abs() / source_edges.clamp_min(1.0e-8)
        neighborhood_ok = relative.median(dim=1).values <= float(neighborhood_tolerance)
    else:
        neighborhood_ok = torch.ones(n, dtype=torch.bool, device=x0.device)

    mask = close & mutual & neighborhood_ok
    return StaticAnchorResult(
        mask=mask,
        target_index=forward_idx.detach().cpu(),
        distance=normalised.detach().cpu(),
        threshold=torch.as_tensor(float(distance_ratio)),
        bbox_diag=bbox.detach(),
        mutual=mutual,
        neighborhood_ok=neighborhood_ok,
        summary={
            "num_anchors": int(mask.sum()),
            "anchor_fraction": float(mask.float().mean()),
            "passed_distance": int(close.sum()),
            "passed_mutual": int(mutual.sum()),
            "passed_neighborhood": int(neighborhood_ok.sum()),
            "distance_ratio": float(distance_ratio),
            "bbox_diagonal": float(bbox),
            "median_normalised_distance": float(normalised.median()),
            "mutual_nn": bool(mutual_nn),
            "neighborhood_check": bool(use_neighborhood_check),
        },
    )


@torch.no_grad()
def extract_static_anchors_from_ply(
    xyz_initial: torch.Tensor,
    final_ply: str | Path,
    **kwargs,
) -> StaticAnchorResult:
    """Same test, with the final endpoint read straight from its PLY."""
    final_xyz = load_ply_xyz(final_ply)
    return extract_confident_static_anchors(xyz_initial, final_xyz, **kwargs)
