"""P1/P2 -- local patches whose size is decided by the geometry, not by a knob.

A patch is the unit of matching in this architecture (§15): the candidate is
``P_i^0 <-> P_j^T``, never ``x_i -> y_j``.  A single point carries no metric
structure, so it cannot be tested for isometry at all; only a patch can.

Patch size is derived from local rank.  A neighbourhood that is still
effectively planar or collinear cannot constrain a 3-D rigid transform -- the
rotation about the degenerate direction is unobservable -- so the smallest
usable patch is the smallest one whose local covariance is full rank.  That
is a property of the surface at that point, which is why it is measured
rather than chosen, and why the same code works on a dense flat panel and a
sparse curved edge.

The graph is built once from the full endpoint geometry and never rebuilt
from dynamic points (§12): the shape context of a moving Gaussian includes
its static neighbours, and re-deriving the neighbourhood from labels would
change the descriptor rather than only the candidate set.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from utils import progress

EPS = 1.0e-8


def batched_eigvalsh(matrix: torch.Tensor, *, chunk_size: int = 8192) -> torch.Tensor:
    """Ascending eigenvalues of many small symmetric matrices, chunked.

    The matrices are 3x3, so the arithmetic is trivial, but cuSOLVER sizes a
    batched workspace from the batch length and on a 40k-Gaussian cloud it
    asked for 4.58 GiB and then failed with INVALID_VALUE on a GPU that could
    not give it.  Chunking keeps that workspace bounded, and falling back to
    CPU means a large scene degrades in speed rather than dying -- these
    scenes differ in size by 4x, so the code has to survive the big ones.
    """
    matrix = torch.as_tensor(matrix)
    out: list[torch.Tensor] = []
    for start in range(0, int(matrix.shape[0]), max(1, int(chunk_size))):
        block = matrix[start:start + int(chunk_size)]
        try:
            out.append(torch.linalg.eigvalsh(block))
        except (RuntimeError, torch._C._LinAlgError):
            out.append(torch.linalg.eigvalsh(block.cpu()).to(block.device))
    return torch.cat(out, 0)


@dataclass
class NestedPatches:
    """A deterministic nested patch family per anchor."""

    indices: torch.Tensor       # [A, K_max] neighbour indices, anchor first
    sizes: torch.Tensor         # [A, L] the L nested patch sizes
    rank_size: torch.Tensor     # [A] K1, the smallest full-rank size
    full_rank: torch.Tensor     # [A] bool -- did the anchor ever reach rank 3?
    condition: torch.Tensor     # [A] lambda_min / lambda_max at K1


@torch.no_grad()
def build_full_geometry_knn(
    xyz: torch.Tensor,
    k: int,
    *,
    chunk_size: int = 2048,
    desc: str = "",
) -> tuple[torch.Tensor, torch.Tensor]:
    """``(indices, distances)`` over the FULL cloud, self included and first.

    Distances are computed on a centred copy: mathematically identical, but
    in float32 a cloud far from the origin loses the relative precision that
    every metric quantity downstream depends on.
    """
    xyz = torch.as_tensor(xyz).detach()
    centred = xyz - xyz.mean(dim=0, keepdim=True)
    k = min(int(k), int(xyz.shape[0]))
    idx: list[torch.Tensor] = []
    dist: list[torch.Tensor] = []
    starts = list(range(0, int(xyz.shape[0]), max(1, int(chunk_size))))
    for start in progress.bar(starts, desc=desc or "knn", unit="chunk"):
        block = torch.cdist(centred[start:start + int(chunk_size)], centred)
        value, index = torch.topk(block, k=k, dim=1, largest=False, sorted=True)
        idx.append(index)
        dist.append(value)
    return torch.cat(idx, 0), torch.cat(dist, 0)


@torch.no_grad()
def find_minimum_full_rank_patch(
    xyz: torch.Tensor,
    knn_indices: torch.Tensor,
    *,
    min_size: int = 6,
    rank_tolerance: float = 0.02,
    desc: str = "",
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Smallest neighbourhood size whose local covariance is full rank.

    ``rank_tolerance`` is the ratio ``lambda_min / lambda_max`` at which a
    neighbourhood stops being degenerate.  It is a numerical rank test -- the
    same role as the tolerance in ``matrix_rank`` -- not a physical
    threshold, and the recovered transform is not a function of it beyond
    which anchors are admitted at all.

    Returns ``(size, full_rank, condition)``.
    """
    xyz = torch.as_tensor(xyz).detach()
    knn = torch.as_tensor(knn_indices).detach().to(device=xyz.device).long()
    anchors, k_max = int(knn.shape[0]), int(knn.shape[1])

    size = torch.full((anchors,), k_max, dtype=torch.long, device=xyz.device)
    found = torch.zeros(anchors, dtype=torch.bool, device=xyz.device)
    condition = torch.zeros(anchors, dtype=torch.float32, device=xyz.device)

    # Incremental sweep over sizes, keeping the first size that is full rank.
    sweep = progress.bar(range(max(4, int(min_size)), k_max + 1), desc=desc or "rank sweep", unit="K")
    for k in sweep:
        pending = ~found
        sweep.set_postfix({"resolved": f"{float(found.float().mean()):.3f}", "K": k})
        if not bool(pending.any()):
            break
        points = xyz[knn[pending, :k]]                                    # [P, k, 3]
        centred = points - points.mean(dim=1, keepdim=True)
        covariance = torch.einsum("pka,pkb->pab", centred, centred) / float(k)
        eigenvalues = batched_eigvalsh(covariance.double()).clamp_min(0.0)   # ascending
        ratio = (eigenvalues[:, 0] / eigenvalues[:, 2].clamp_min(1.0e-30)).float()
        ok = ratio >= float(rank_tolerance)
        where = pending.nonzero(as_tuple=True)[0]
        size[where[ok]] = k
        condition[where[ok]] = ratio[ok]
        found[where[ok]] = True
    condition[~found] = 0.0
    return size, found, condition


@torch.no_grad()
def build_nested_metric_patches(
    xyz: torch.Tensor,
    knn_indices: torch.Tensor,
    *,
    min_size: int = 6,
    rank_tolerance: float = 0.02,
    levels: int = 3,
    desc: str = "",
) -> NestedPatches:
    """§13's K1 / 2*K1 / 4*K1 family, bounded by the available neighbours.

    The expansion is a fixed rule applied to a measured K1, so nothing here
    is dataset-specific: a scene whose surfaces are locally flatter gets
    larger patches automatically, because that is what it took to reach full
    rank there.
    """
    knn = torch.as_tensor(knn_indices).detach().long()
    k_max = int(knn.shape[1])
    size, full_rank, condition = find_minimum_full_rank_patch(
        xyz, knn, min_size=int(min_size), rank_tolerance=float(rank_tolerance), desc=desc,
    )
    sizes = torch.stack([(size * (2 ** level)).clamp(max=k_max) for level in range(int(levels))], dim=1)
    return NestedPatches(
        indices=knn,
        sizes=sizes,
        rank_size=size,
        full_rank=full_rank,
        condition=condition,
    )


@torch.no_grad()
def compute_pairwise_metric_matrix(
    xyz: torch.Tensor,
    patch_indices: torch.Tensor,
) -> torch.Tensor:
    """``D(a, b) = ||x_a - x_b||`` for a batch of patches, ``[B, m, m]``."""
    xyz = torch.as_tensor(xyz).detach()
    points = xyz[torch.as_tensor(patch_indices).detach().to(device=xyz.device).long()]
    # Centre each patch before cdist for the same float32 reason as above.
    points = points - points.mean(dim=1, keepdim=True)
    return torch.cdist(points, points)


@torch.no_grad()
def normalise_metric_matrix(
    distance: torch.Tensor,
    *,
    valid: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Divide each patch by its own median off-diagonal distance.

    §16.  This is what makes the comparison intrinsic: the two endpoint
    models sample the surface at different densities, so two patches of the
    same physical region have systematically different absolute spacing.
    Normalising by a per-patch robust scale removes that without introducing
    a free scale parameter, and the median is used rather than the mean
    because a patch that straddles a depth discontinuity has a few enormous
    pair distances that would otherwise set the unit.

    Returns ``(normalised, scale)``.
    """
    distance = torch.as_tensor(distance).detach()
    b, m, _ = distance.shape
    off = ~torch.eye(m, dtype=torch.bool, device=distance.device)[None, :, :].expand(b, m, m)
    if valid is not None:
        pair = valid[:, :, None] & valid[:, None, :]
        off = off & pair
    big = torch.where(off, distance, torch.full_like(distance, float("nan")))
    scale = big.flatten(1).nanmedian(dim=1).values
    scale = torch.where(torch.isfinite(scale) & (scale > 0), scale, torch.ones_like(scale))
    return distance / (scale[:, None, None] + EPS), scale


@torch.no_grad()
def gather_patches(
    patches: NestedPatches,
    anchors: torch.Tensor,
    *,
    level: int = 0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """``(indices, valid)`` of one nested level for the given anchors.

    Patches of different sizes are returned padded to a common width with a
    validity mask rather than truncated to the smallest: truncating would
    silently give every anchor the degenerate patch of its worst neighbour.
    """
    anchors = torch.as_tensor(anchors).detach().long()
    sizes = patches.sizes[anchors, int(level)]
    width = int(sizes.max())
    indices = patches.indices[anchors, :width]
    valid = torch.arange(width, device=indices.device)[None, :] < sizes[:, None]
    # Padding slots point at the anchor itself so every gathered index is
    # legal; the mask is what keeps them out of the computation.
    indices = torch.where(valid, indices, indices[:, :1].expand_as(indices))
    return indices, valid
