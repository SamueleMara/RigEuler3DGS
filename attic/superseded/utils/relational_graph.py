"""Rigid-invariant graph descriptors for an endpoint 3DGS model.

A rigid motion ``y = Rx + t`` changes every absolute position but leaves the
distances *inside* a rigid component untouched:

    ||x_i - x_k|| == ||y_pi(i) - y_pi(k)||

so absolute XYZ proximity is the wrong primary matching criterion between two
independently trained endpoint models -- it asks "which target Gaussian is
spatially near this one", when the question is "which target region has
compatible intrinsic geometry".  Everything in this module is therefore
built to be invariant to ``R`` and ``t``:

* the distance spectrum is a SORTED set of scale-normalised neighbour
  distances, so it needs no neighbour identity and no common frame;
* covariance eigenvalues are invariant because ``Sigma' = R Sigma R^T``.

Graph topology here is geometry-only and built once.  It must never be
derived from the dynamic labels it is later used to refine.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from utils.local_rigid_field import (
    build_knn_graph,
    compute_effective_local_scale,
    local_scale_from_knn,
)


@dataclass
class RelationalGraph:
    """One endpoint cloud, described in a way a rigid motion cannot change."""

    xyz: torch.Tensor                   # [N, 3]
    knn_rig: torch.Tensor               # [N, K_rig] local rigidity graph
    knn_ctx: torch.Tensor               # [N, K_ctx] wider context graph
    edge_len_rig: torch.Tensor          # [N, K_rig]
    edge_len_ctx: torch.Tensor          # [N, K_ctx]
    effective_scale: torch.Tensor       # [N] robust local length unit
    distance_descriptor: torch.Tensor   # [N, K_rig] sorted, scale-normalised
    covariance_descriptor: torch.Tensor # [N, 3] sorted, trace-normalised
    geometry_valid: torch.Tensor        # [N] bool
    pca_degeneracy: torch.Tensor        # [N] lambda3 / lambda1 of the local PCA
    appearance: torch.Tensor | None = None  # [N, C] view-independent colour cue


def compute_edge_lengths(xyz: torch.Tensor, knn_indices: torch.Tensor) -> torch.Tensor:
    """Euclidean length of every graph edge, as ``[N, K]``."""
    xyz = torch.as_tensor(xyz).detach()
    neighbours = xyz[torch.as_tensor(knn_indices).detach().to(device=xyz.device).long()]
    return torch.linalg.norm(neighbours - xyz[:, None, :], dim=-1)


def compute_distance_spectrum_descriptor(
    edge_len: torch.Tensor,
    knn_indices: torch.Tensor,
    effective_scale: torch.Tensor,
    *,
    eps: float = 1.0e-8,
) -> torch.Tensor:
    """Sorted, pairwise-scale-normalised neighbour distances.

    Sorting is what removes the need for neighbour identity: two
    corresponding regions in independently trained models will not have their
    neighbours in the same order, or even the same neighbours, but a rigid
    region's *set* of normalised distances is preserved.  Normalising by
    ``sqrt(s_i s_k)`` rather than by ``s_i`` alone keeps the descriptor
    symmetric in the two endpoints of the edge.
    """
    effective_scale = torch.as_tensor(effective_scale).detach().reshape(-1)
    neighbour_scale = effective_scale[torch.as_tensor(knn_indices).detach().to(device=effective_scale.device).long()]
    pair_scale = torch.sqrt(effective_scale[:, None] * neighbour_scale).clamp_min(eps)
    normalised = edge_len / (pair_scale + eps)
    return torch.sort(normalised, dim=1).values


def compute_covariance_invariants(
    scaling: torch.Tensor | None = None,
    *,
    covariance: torch.Tensor | None = None,
    eps: float = 1.0e-8,
) -> torch.Tensor:
    """Sorted, trace-normalised covariance eigenvalues -- a rotation invariant.

    For a 3DGS Gaussian ``Sigma = R diag(s^2) R^T``, so the eigenvalues are
    exactly the squared scales and can be read straight off ``get_scaling``
    without ever forming ``Sigma``.  A full covariance is accepted too, for
    representations where the scales are not available separately.
    """
    if covariance is not None:
        eigenvalues = torch.linalg.eigvalsh(torch.as_tensor(covariance).detach().double()).clamp_min(0.0).float()
    elif scaling is not None:
        eigenvalues = torch.as_tensor(scaling).detach().float().clamp_min(0.0).square()
    else:
        raise ValueError("compute_covariance_invariants needs scaling or covariance")
    eigenvalues = torch.sort(eigenvalues, dim=-1).values
    trace = eigenvalues.sum(dim=-1, keepdim=True).clamp_min(eps)
    return eigenvalues / trace


def compute_local_pca_degeneracy(
    xyz: torch.Tensor,
    knn_indices: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """``(lambda3/lambda1, eigenvalues)`` of each local neighbourhood's PCA.

    A near-planar Gaussian surface patch is rank-2, which makes any 3D
    quantity estimated on it weakly constrained along the normal.  Reporting
    it lets a degenerate neighbourhood be discounted rather than silently
    trusted.
    """
    xyz = torch.as_tensor(xyz).detach()
    neighbours = xyz[torch.as_tensor(knn_indices).detach().to(device=xyz.device).long()]
    centred = neighbours - neighbours.mean(dim=1, keepdim=True)
    covariance = torch.einsum("nka,nkb->nab", centred, centred) / max(1, int(neighbours.shape[1]))
    from utils.metric_patch_graph import batched_eigvalsh

    eigenvalues = batched_eigvalsh(covariance.double()).clamp_min(0.0).float()  # ascending
    ratio = eigenvalues[:, 0] / eigenvalues[:, 2].clamp_min(1.0e-30)
    return ratio, eigenvalues


@torch.no_grad()
def build_relational_graph(
    xyz: torch.Tensor,
    *,
    scaling: torch.Tensor | None = None,
    covariance: torch.Tensor | None = None,
    appearance: torch.Tensor | None = None,
    knn_rig: int = 16,
    knn_ctx: int = 64,
    scale_mode: str = "neighborhood_median",
    scale_clip_low_q: float = 0.10,
    scale_clip_high_q: float = 0.90,
    degeneracy_ratio: float = 1.0e-3,
    chunk_size: int = 2048,
) -> RelationalGraph:
    """Build both graph scales and every rigid-invariant descriptor.

    The two scales answer different questions: the rigidity graph asks
    whether a neighbourhood preserved its own internal metric, the context
    graph asks whether that neighbourhood moved relative to the rest of the
    scene.  Neither on its own separates static from rigidly moving.
    """
    xyz = torch.as_tensor(xyz).detach()
    n = int(xyz.shape[0])
    if n < 2:
        raise ValueError("a relational graph needs at least two Gaussians")

    # Distances are computed on a centred copy.  Mathematically this changes
    # nothing, but in float32 it matters: a cloud sitting far from the origin
    # (as a COLMAP frame routinely is) loses relative precision in cdist, and
    # the rigid invariance these descriptors depend on degrades with it.
    centred = xyz - xyz.mean(dim=0, keepdim=True)
    idx_rig, dist_rig = build_knn_graph(centred, int(knn_rig), chunk_size=int(chunk_size))
    idx_ctx, dist_ctx = build_knn_graph(centred, int(knn_ctx), chunk_size=int(chunk_size))

    raw_scale = local_scale_from_knn(dist_rig)
    effective_scale = compute_effective_local_scale(
        raw_scale, idx_rig, mode=str(scale_mode),
        clip_low_quantile=float(scale_clip_low_q),
        clip_high_quantile=float(scale_clip_high_q),
    )

    distance_descriptor = compute_distance_spectrum_descriptor(dist_rig, idx_rig, effective_scale)
    covariance_descriptor = compute_covariance_invariants(scaling, covariance=covariance)
    if int(covariance_descriptor.shape[0]) != n:
        raise ValueError("covariance descriptor must have one row per Gaussian")

    degeneracy, _eigenvalues = compute_local_pca_degeneracy(centred, idx_rig)
    geometry_valid = (
        torch.isfinite(distance_descriptor).all(dim=-1)
        & torch.isfinite(covariance_descriptor).all(dim=-1)
        & (degeneracy >= float(degeneracy_ratio))
    )

    if appearance is not None:
        appearance = torch.as_tensor(appearance).detach().float().reshape(n, -1)

    return RelationalGraph(
        xyz=xyz,
        knn_rig=idx_rig,
        knn_ctx=idx_ctx,
        edge_len_rig=dist_rig,
        edge_len_ctx=dist_ctx,
        effective_scale=effective_scale,
        distance_descriptor=distance_descriptor,
        covariance_descriptor=covariance_descriptor,
        geometry_valid=geometry_valid,
        pca_degeneracy=degeneracy,
        appearance=appearance,
    )
