"""Raw spatial mismatch, computed independently at both endpoints.

One question, asked separately of each cloud:

    forward   does compatible geometry still exist where this frame-0
              Gaussian currently is?
    backward  does compatible geometry already exist where this frame-T
              Gaussian currently is?

Both directions are needed because they answer different halves of the same
question and neither implies the other.  A frame-0 Gaussian whose region was
vacated has a large forward mismatch; a frame-T Gaussian that appeared in a
region nothing used to occupy has a large backward mismatch.  A part that
moved produces both, at two different places, and it is the CONJUNCTION that
localises a motion rather than merely detecting one.

The counts are deliberately not assumed equal and no Gaussian identity is
assumed: these are two independently trained models.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class BidirectionalSpatialMismatch:
    """Scale-normalised nearest-neighbour mismatch at both endpoints."""

    forward: torch.Tensor        # [N0] E_{0->T}
    backward: torch.Tensor       # [NT] E_{T->0}
    source_scale: torch.Tensor   # [N0] local length unit in frame 0
    target_scale: torch.Tensor   # [NT] local length unit in frame T
    origin: torch.Tensor         # [1, 3] shared centring origin


@torch.no_grad()
def _nearest_squared_distance(
    query: torch.Tensor,
    reference: torch.Tensor,
    *,
    chunk_size: int = 1024,
) -> torch.Tensor:
    """Squared distance to the nearest reference point, chunked over queries."""
    out: list[torch.Tensor] = []
    for start in range(0, int(query.shape[0]), max(1, int(chunk_size))):
        d2 = torch.cdist(query[start:start + int(chunk_size)], reference).square()
        out.append(d2.min(dim=1).values)
    return torch.cat(out, 0).clamp_min(0.0)


def shared_origin(source_xyz: torch.Tensor, target_xyz: torch.Tensor | None = None) -> torch.Tensor:
    """The common centring origin, ``[1, 3]``.

    Both clouds must be shifted by the SAME vector or the distances between
    them change.  The shift itself is not cosmetic: ``cdist`` evaluates
    ``|a|^2 + |b|^2 - 2a.b``, which cancels catastrophically in float32 for a
    cloud far from the origin.  A COLMAP frame centred 500 units out reported
    a mismatch of ~0.96 local scales for points that had not moved at all.
    """
    source_xyz = torch.as_tensor(source_xyz).detach()
    return source_xyz.mean(dim=0, keepdim=True)


@torch.no_grad()
def compute_forward_spatial_mismatch(
    source_xyz: torch.Tensor,
    target_xyz: torch.Tensor,
    source_scale: torch.Tensor,
    *,
    origin: torch.Tensor | None = None,
    chunk_size: int = 1024,
) -> torch.Tensor:
    """``E_i^{0->T}``: nearest frame-T geometry, in units of frame-0 local scale."""
    source_xyz = torch.as_tensor(source_xyz).detach()
    target_xyz = torch.as_tensor(target_xyz).detach().to(source_xyz)
    origin = shared_origin(source_xyz) if origin is None else torch.as_tensor(origin).detach().to(source_xyz)
    squared = _nearest_squared_distance(source_xyz - origin, target_xyz - origin, chunk_size=int(chunk_size))
    scale = torch.as_tensor(source_scale).detach().to(squared).reshape(-1).clamp_min(1.0e-8)
    return squared / scale.square()


@torch.no_grad()
def compute_backward_spatial_mismatch(
    source_xyz: torch.Tensor,
    target_xyz: torch.Tensor,
    target_scale: torch.Tensor,
    *,
    origin: torch.Tensor | None = None,
    chunk_size: int = 1024,
) -> torch.Tensor:
    """``E_j^{T->0}``: nearest frame-0 geometry, in units of frame-T local scale.

    Normalised by the TARGET's own scale, not the source's.  The two models
    densify independently, so a frame-T Gaussian's neighbourhood spacing is
    its own property and borrowing the source's would make the two directions
    incomparable in exactly the regions where they differ most.
    """
    source_xyz = torch.as_tensor(source_xyz).detach()
    target_xyz = torch.as_tensor(target_xyz).detach().to(source_xyz)
    origin = shared_origin(source_xyz) if origin is None else torch.as_tensor(origin).detach().to(source_xyz)
    squared = _nearest_squared_distance(target_xyz - origin, source_xyz - origin, chunk_size=int(chunk_size))
    scale = torch.as_tensor(target_scale).detach().to(squared).reshape(-1).clamp_min(1.0e-8)
    return squared / scale.square()


@torch.no_grad()
def compute_bidirectional_spatial_mismatch(
    source_xyz: torch.Tensor,
    target_xyz: torch.Tensor,
    source_scale: torch.Tensor,
    target_scale: torch.Tensor,
    *,
    chunk_size: int = 1024,
) -> BidirectionalSpatialMismatch:
    """Both directions against one shared origin."""
    source_xyz = torch.as_tensor(source_xyz).detach()
    target_xyz = torch.as_tensor(target_xyz).detach().to(source_xyz)
    origin = shared_origin(source_xyz)
    return BidirectionalSpatialMismatch(
        forward=compute_forward_spatial_mismatch(
            source_xyz, target_xyz, source_scale, origin=origin, chunk_size=int(chunk_size)
        ),
        backward=compute_backward_spatial_mismatch(
            source_xyz, target_xyz, target_scale, origin=origin, chunk_size=int(chunk_size)
        ),
        source_scale=torch.as_tensor(source_scale).detach().reshape(-1),
        target_scale=torch.as_tensor(target_scale).detach().reshape(-1),
        origin=origin,
    )
