"""Relative-distance feedback that separates rigid motion from bad matching.

A large static mismatch says only that a Gaussian is not where it was.  It
cannot say whether it moved rigidly or whether the dynamic branch simply
matched it to the wrong place, and those two produce identical ``E_S``.  The
distinguishing observable is relational, and it needs BOTH halves:

    internal distance change  -- did the neighbourhood keep its own metric?
    context distance change   -- did it move relative to what stayed put?

    STATIC          internal low   context low
    RIGID DYNAMIC   internal low   context HIGH
    BAD MATCH       internal HIGH  (context uninformative)

Internal change alone cannot see motion at all -- a rigid part preserves its
internal distances exactly whether or not it moved -- and context change
alone cannot tell motion from a scrambled match.  Only the pair does.

Everything is measured against transported positions supplied by the caller
and normalised by the local scale, so nothing here carries a metric constant
that would fail to transfer between scenes.
"""

from __future__ import annotations

import torch


def saturating_score(value: torch.Tensor, *, knee: float = 1.0) -> torch.Tensor:
    """Bounded ``g``: ``x / (x + knee)``, zero at zero and monotone.

    Deliberately NOT the quantile-based score used elsewhere.  A rank score
    puts a typical point at 0.5 by construction, so a scene where nothing
    moved would still report half its points as having large context change.
    The claims here ("context change is approximately zero for a static
    point") are absolute, so the mapping must be absolute too -- with the
    input already in units of the local scale, ``knee`` is dimensionless.
    """
    value = torch.as_tensor(value).clamp_min(0.0)
    return value / (value + max(float(knee), 1.0e-8))


def robust_mean(value: torch.Tensor, weight: torch.Tensor, dim: int = 1) -> torch.Tensor:
    """Weighted mean that ignores empty rows instead of producing NaN."""
    total = weight.sum(dim=dim)
    return (value * weight).sum(dim=dim) / total.clamp_min(1.0e-12)


@torch.no_grad()
def compute_internal_distance_change(
    source_xyz: torch.Tensor,
    transported_xyz: torch.Tensor,
    knn_indices: torch.Tensor,
    effective_scale: torch.Tensor,
    *,
    edge_tau: float = 0.5,
    trust: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """``(R_int, edge rigidity confidence)`` for the local rigidity graph.

    ``R_int`` is the robust mean absolute change in edge length under the
    proposed transport, in units of the local scale.  Edges are weighted by
    the per-point trust (typically the coupling confidence), because a KNN
    edge that crosses a static/dynamic boundary or lands on a badly matched
    neighbour is not evidence about this point's rigidity -- §19's "do not
    assume all KNN edges belong to the same rigid part".
    """
    source_xyz = torch.as_tensor(source_xyz).detach()
    transported_xyz = torch.as_tensor(transported_xyz).detach().to(source_xyz)
    knn = torch.as_tensor(knn_indices).detach().to(device=source_xyz.device).long()
    scale = torch.as_tensor(effective_scale).detach().to(source_xyz).reshape(-1).clamp_min(1.0e-8)

    d0 = torch.linalg.norm(source_xyz[knn] - source_xyz[:, None, :], dim=-1)
    dT = torch.linalg.norm(transported_xyz[knn] - transported_xyz[:, None, :], dim=-1)
    pair_scale = torch.sqrt(scale[:, None] * scale[knn]).clamp_min(1.0e-8)
    delta = (dT - d0) / pair_scale

    edge_confidence = torch.exp(-0.5 * delta.square() / max(float(edge_tau), 1.0e-6) ** 2)
    weight = torch.ones_like(delta)
    if trust is not None:
        trust = torch.as_tensor(trust).detach().to(delta).reshape(-1).clamp(0.0, 1.0)
        weight = weight * torch.sqrt(trust[:, None] * trust[knn])
    return robust_mean(delta.abs(), weight), edge_confidence


@torch.no_grad()
def compute_static_context_anchors(
    static_mismatch: torch.Tensor,
    *,
    displacement: torch.Tensor | None = None,
    visibility: torch.Tensor | None = None,
    mismatch_quantile: float = 0.25,
    displacement_quantile: float = 0.25,
    min_visibility: float = 0.0,
) -> torch.Tensor:
    """A CONSERVATIVE static anchor set, not "every context neighbour".

    The anchors define the frame against which motion is measured, so a
    moving point admitted here does not merely add noise -- it cancels the
    very contrast the measurement exists to detect.  Hence quantiles on the
    strictest available evidence rather than a threshold near the median.
    """
    mismatch = torch.as_tensor(static_mismatch).detach().reshape(-1)
    anchors = mismatch <= torch.quantile(mismatch.float(), float(mismatch_quantile))
    if displacement is not None:
        value = torch.as_tensor(displacement).detach().to(mismatch).reshape(-1)
        anchors = anchors & (value <= torch.quantile(value.float(), float(displacement_quantile)))
    if visibility is not None and float(min_visibility) > 0.0:
        anchors = anchors & (torch.as_tensor(visibility).detach().to(mismatch).reshape(-1) >= float(min_visibility))
    return anchors


@torch.no_grad()
def compute_context_distance_change(
    source_xyz: torch.Tensor,
    transported_xyz: torch.Tensor,
    anchor_mask: torch.Tensor,
    effective_scale: torch.Tensor,
    *,
    max_anchors: int = 64,
    chunk_size: int = 2048,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """``C_ctx``: robust mean ``| ||F(x_i) - x_a|| - ||x_i - x_a|| |`` over anchors.

    Anchors are subsampled uniformly rather than by proximity: the nearest
    static points to a moving part sit on the boundary and move least
    relative to it, so a nearest-anchor set systematically under-reports the
    contrast it is meant to measure.
    """
    source_xyz = torch.as_tensor(source_xyz).detach()
    transported_xyz = torch.as_tensor(transported_xyz).detach().to(source_xyz)
    scale = torch.as_tensor(effective_scale).detach().to(source_xyz).reshape(-1).clamp_min(1.0e-8)
    anchor_idx = torch.as_tensor(anchor_mask).detach().to(device=source_xyz.device).reshape(-1).nonzero(as_tuple=True)[0]
    if int(anchor_idx.numel()) == 0:
        return torch.zeros(int(source_xyz.shape[0]), device=source_xyz.device, dtype=source_xyz.dtype)
    if int(anchor_idx.numel()) > int(max_anchors):
        pick = torch.randperm(int(anchor_idx.numel()), device=anchor_idx.device, generator=generator)[: int(max_anchors)]
        anchor_idx = anchor_idx[pick]
    anchors = source_xyz[anchor_idx]

    out = torch.zeros(int(source_xyz.shape[0]), device=source_xyz.device, dtype=source_xyz.dtype)
    for start in range(0, int(source_xyz.shape[0]), max(1, int(chunk_size))):
        stop = min(int(source_xyz.shape[0]), start + int(chunk_size))
        before = torch.cdist(source_xyz[start:stop], anchors)
        after = torch.cdist(transported_xyz[start:stop], anchors)
        out[start:stop] = ((after - before).abs() / scale[start:stop, None]).median(dim=1).values
    return out


@torch.no_grad()
def compute_relational_dynamic_evidence(
    context_change: torch.Tensor,
    internal_change: torch.Tensor,
    relational_confidence: torch.Tensor,
    *,
    lambda_internal: float = 1.0,
    context_knee: float = 1.0,
    internal_knee: float = 0.5,
) -> torch.Tensor:
    """``e_rel = c_rel * (g(C_ctx) - lambda * g(R_int))``, signed.

    Negative is a real and necessary outcome: a point whose neighbourhood
    was torn apart by the proposed transport is evidence AGAINST the dynamic
    hypothesis, not merely absent evidence for it.  ``internal_knee`` is
    tighter than ``context_knee`` because a genuine rigid part should
    preserve its internal metric to well under one local scale, while the
    context contrast it produces is expected to be of order the motion.
    """
    context = saturating_score(context_change, knee=float(context_knee))
    internal = saturating_score(internal_change, knee=float(internal_knee))
    confidence = torch.as_tensor(relational_confidence).detach().to(context).clamp(0.0, 1.0)
    return confidence * (context - float(lambda_internal) * internal)
