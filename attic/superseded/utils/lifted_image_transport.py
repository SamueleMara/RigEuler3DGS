"""R4/R5 -- lift pixel transport back to Gaussian/patch candidate support.

    P_img = (Omega^0)^T Gamma Omega^T

Read plainly: source pixel p is explained by source Gaussian i, target pixel q
by target Gaussian j, and if p transports to q then the pair (i, j) receives
image support.

This is evaluated at PATCH granularity, which is not an approximation of what
§16 asks for -- it is exactly what §16 asks for.  The Gaussian-level P_img is
an N x N object (1.6 billion entries on storage_45135) that nobody can store,
while the quantity the candidate ordering actually consumes is

    u_ij = sum_{a in P_i} sum_{b in P_j} P_img_ab

and splatting patch indicators computes that directly: the indicator's splat
IS sum_{a in P_i} Omega_cpa, so the double sum collapses to a bilinear form in
two footprint images.

§15: this is search support and nothing else.  It orders which candidates get
evaluated and rejects ones with no image support at all.  It never accepts a
match -- metric isometry, SE(3) validity, uncertainty consistency and render
validation remain the only acceptance mechanisms.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch

EPS = 1.0e-12


@dataclass
class LiftedSupport:
    """Patch-level image support, fused over the cameras that can see it."""

    support: torch.Tensor          # [S, T] u_ij
    view_count: torch.Tensor       # [S, T] cameras contributing
    stats: dict[str, Any] = field(default_factory=dict)

    def ordering(self, k: int | None = None) -> torch.Tensor:
        """Target patches per source patch, best-supported first."""
        order = torch.argsort(self.support, dim=1, descending=True)
        return order if k is None else order[:, : int(k)]


@torch.no_grad()
def lift_pixel_transport_to_gaussian_support(
    transport,
    source_footprints: torch.Tensor,
    target_footprints: torch.Tensor,
) -> torch.Tensor:
    """``u = F0^T Gamma FT`` for one camera, as ``[S, T]``.

    ``source_footprints`` is ``[S, H, W]`` -- each source patch's splatted
    ownership -- and likewise for the target.  The contraction is over the
    transported pixel pairs only, so cost scales with the sparse coupling
    rather than with the image.
    """
    if int(transport.source_pixel_indices.numel()) == 0:
        return torch.zeros(
            (int(source_footprints.shape[0]), int(target_footprints.shape[0])),
            device=source_footprints.device,
        )
    source_flat = source_footprints.reshape(int(source_footprints.shape[0]), -1)
    target_flat = target_footprints.reshape(int(target_footprints.shape[0]), -1)

    # [S_patches, S_pixels] and [T_patches, S_pixels, K]
    source_weight = source_flat[:, transport.source_pixel_indices]
    target_weight = target_flat[:, transport.target_pixel_indices.reshape(-1)]
    target_weight = target_weight.reshape(
        int(target_footprints.shape[0]), *transport.target_pixel_indices.shape
    )
    # u_ij = sum_{p,k} F0_i(p) Gamma(p,k) FT_j(target(p,k))
    weighted = transport.coupling_values[None] * target_weight        # [T, S_pix, K]
    per_source = weighted.sum(dim=2)                                  # [T, S_pix]
    return source_weight @ per_source.T                               # [S, T]


@torch.no_grad()
def fuse_multiview_gaussian_support(
    per_camera: list[torch.Tensor],
    *,
    visibility: list[torch.Tensor] | None = None,
) -> LiftedSupport:
    """§14: fuse as a sum of log evidence, with no per-camera weights.

    A camera that cannot see a pair contributes nothing rather than a zero:
    scoring an invisible pair as disagreement would punish a correct
    hypothesis for being occluded, which is the opposite of what occlusion
    should do.
    """
    if not per_camera:
        empty = torch.zeros((0, 0))
        return LiftedSupport(empty, empty, {"cameras": 0})

    stacked = torch.stack(per_camera, dim=0)                          # [C, S, T]
    contributing = (stacked > 0).float()
    log_support = torch.log(stacked.clamp_min(EPS)) * contributing
    count = contributing.sum(dim=0)
    fused = torch.where(count > 0, (log_support.sum(dim=0) / count.clamp_min(1.0)).exp(), torch.zeros_like(count))
    return LiftedSupport(
        support=fused, view_count=count,
        stats={
            "cameras": len(per_camera),
            "pairs_with_support": int((fused > 0).sum()),
            "support_density": float((fused > 0).float().mean()),
            "median_views_per_supported_pair": float(count[fused > 0].median()) if bool((fused > 0).any()) else 0.0,
        },
    )


@torch.no_grad()
def score_patch_pairs_from_gaussian_support(
    lifted: LiftedSupport,
    *,
    candidates: int,
    fallback_generator: torch.Generator | None = None,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Ordered target candidates per source patch, with honest fallback.

    Where image support is entirely absent -- a patch no camera sees moving --
    the ordering falls back to random draws rather than silently returning
    the same low-support targets for every anchor. The report says how many
    anchors that was, because an "image-supported" search that was mostly
    random is a different experiment from one that was not.
    """
    support = lifted.support
    sources, targets = int(support.shape[0]), int(support.shape[1])
    k = min(int(candidates), targets)
    order = torch.argsort(support, dim=1, descending=True)[:, :k]

    supported = support.gather(1, order) > 0
    without_support = (~supported.any(dim=1))
    if bool(without_support.any()):
        random_order = torch.stack([
            torch.randperm(targets, generator=fallback_generator)[:k]
            for _ in range(int(without_support.sum()))
        ]).to(order.device)
        order = order.clone()
        order[without_support] = random_order

    return order, {
        "sources": sources, "targets": targets, "candidates": k,
        "anchors_with_image_support": int((~without_support).sum()),
        "anchors_without_image_support": int(without_support.sum()),
        "image_supported_fraction": float((~without_support).float().mean()),
        "median_supported_candidates_per_anchor": float(supported.sum(dim=1).float().median()),
    }


@torch.no_grad()
def candidate_recall_audit(
    order: torch.Tensor,
    true_target: torch.Tensor,
    *,
    depths: tuple[int, ...] = (1, 4, 8, 16, 32),
) -> dict[str, float]:
    """§44: where does the true counterpart rank under image-supported order?

    This is the measurement that decides whether the rasterizer coupling
    solved candidate starvation, and it is the only way to tell an ordering
    that helps from one that merely reorders noise.
    """
    matches = order == true_target[:, None]
    rank = torch.where(
        matches.any(dim=1), matches.float().argmax(dim=1), torch.full((int(order.shape[0]),), -1)
    )
    found = rank >= 0
    report = {f"recall_at_{d}": float(((rank >= 0) & (rank < d)).float().mean()) for d in depths}
    report["median_rank_when_found"] = float(rank[found].float().median()) if bool(found.any()) else float("nan")
    report["found_fraction"] = float(found.float().mean())
    return report
