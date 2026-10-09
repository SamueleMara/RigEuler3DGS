"""P7 -- appearance ranks physically valid candidates, and nothing else.

The structural rule of this stage (§27): appearance cannot make a
geometrically invalid candidate valid.  It is given only candidates that
already passed the metric and SE(3) tests, and its job is to break ties
between them.

That is enforced by the interface rather than by discipline -- the ranking
function takes a validity mask and refuses to score anything outside it -- so
violating the rule requires changing this file, not just passing a larger
weight.  There is no appearance weight to pass: with physics deciding
admissibility first, there is nothing for a weight to trade against.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

SH_C0 = 0.28209479177387814
EPS = 1.0e-12


@dataclass
class AppearanceRanking:
    """Which physically valid candidate looks like the same surface."""

    best: torch.Tensor           # [A] index into the candidate axis, -1 if none
    distance: torch.Tensor       # [A, C] appearance distance per candidate
    ambiguous: torch.Tensor      # [A] bool -- top candidates indistinguishable
    valid_count: torch.Tensor    # [A] how many candidates were physically valid


def sh_dc_to_rgb(features: torch.Tensor) -> torch.Tensor:
    """View-independent colour from the DC band of the spherical harmonics.

    Only the DC band. The higher bands encode view-dependent appearance, so
    they change with the camera rather than with the Gaussian, and two
    independently trained endpoint models have no reason to agree on them.
    """
    features = torch.as_tensor(features).detach().float()
    dc = features[:, 0, :] if features.dim() == 3 else features
    return (SH_C0 * dc + 0.5).clamp(0.0, 1.0)


@torch.no_grad()
def build_patch_sh_dc_descriptor(
    rgb: torch.Tensor,
    patch_indices: torch.Tensor,
    patch_valid: torch.Tensor | None = None,
    *,
    bins: int = 8,
) -> torch.Tensor:
    """A patch colour histogram, ``[B, 3 * bins]``.

    A histogram rather than a mean: the two endpoint models sample the
    surface differently, so a patch's mean colour shifts with which parts of
    it happened to get Gaussians, while the distribution of colours present
    is far more stable.  It is also permutation-invariant, which it has to be
    -- there is no correspondence between the two patches at this point.
    """
    rgb = torch.as_tensor(rgb).detach().float()
    indices = torch.as_tensor(patch_indices).detach().to(device=rgb.device).long()
    values = rgb[indices]                                            # [B, m, 3]
    b, m, channels = values.shape
    mask = (torch.ones((b, m), dtype=torch.bool, device=rgb.device)
            if patch_valid is None else patch_valid.to(device=rgb.device).bool())

    edges = torch.linspace(0.0, 1.0, int(bins) + 1, device=rgb.device)
    slot = torch.bucketize(values.clamp(0.0, 1.0 - 1.0e-6), edges) - 1
    slot = slot.clamp(0, int(bins) - 1)                              # [B, m, 3]

    histogram = torch.zeros((b, channels, int(bins)), device=rgb.device)
    weight = mask.float()[:, :, None].expand(b, m, channels)
    histogram.scatter_add_(2, slot.transpose(1, 2), weight.transpose(1, 2))
    histogram = histogram / histogram.sum(dim=2, keepdim=True).clamp_min(EPS)
    return histogram.reshape(b, channels * int(bins))


@torch.no_grad()
def build_renderer_observed_patch_descriptor(
    observed_rgb: torch.Tensor,
    patch_indices: torch.Tensor,
    patch_valid: torch.Tensor | None = None,
    *,
    bins: int = 8,
) -> torch.Tensor:
    """§28's preferred descriptor, from colours a renderer actually observed.

    Same histogram, different source: per-Gaussian colour aggregated from the
    pixels each Gaussian influenced. Supplying it is the caller's job because
    it needs cameras; the shape and semantics match the SH-DC version so the
    two are interchangeable here.
    """
    return build_patch_sh_dc_descriptor(observed_rgb, patch_indices, patch_valid, bins=int(bins))


@torch.no_grad()
def rank_physically_valid_candidates(
    source_descriptor: torch.Tensor,
    candidate_descriptor: torch.Tensor,
    physically_valid: torch.Tensor,
    *,
    indistinguishable_ratio: float = 0.9,
) -> AppearanceRanking:
    """Rank ONLY physically valid candidates; never resurrect a rejected one.

    ``indistinguishable_ratio`` decides when the top two are too close to
    separate.  §29 requires retained ambiguity rather than a forced choice,
    because picking arbitrarily between two equally good candidates produces
    a confident wrong answer where the honest output is "cannot tell".
    """
    source = torch.as_tensor(source_descriptor).detach().float()     # [A, F]
    candidates = torch.as_tensor(candidate_descriptor).detach().float()  # [A, C, F]
    valid = torch.as_tensor(physically_valid).detach().bool()        # [A, C]

    distance = torch.linalg.norm(candidates - source[:, None, :], dim=-1)
    masked = torch.where(valid, distance, torch.full_like(distance, float("inf")))

    valid_count = valid.sum(dim=1)
    best = masked.argmin(dim=1)
    best = torch.where(valid_count > 0, best, torch.full_like(best, -1))

    if int(masked.shape[1]) >= 2:
        top2 = torch.topk(masked, k=2, dim=1, largest=False).values
        close = top2[:, 0] >= float(indistinguishable_ratio) * top2[:, 1]
        ambiguous = (valid_count >= 2) & close & torch.isfinite(top2[:, 1])
    else:
        ambiguous = torch.zeros_like(valid_count, dtype=torch.bool)
    return AppearanceRanking(
        best=best,
        distance=torch.where(valid, distance, torch.full_like(distance, float("nan"))),
        ambiguous=ambiguous | (valid_count == 0),
        valid_count=valid_count,
    )
