"""Relational transport restricted to the likely-dynamic subset of each cloud.

The ordering is the whole point.  Matching the full scene and suppressing
static pairs afterwards leaves every static Gaussian in the candidate pool
while the retrieval runs, so a moving Gaussian competes against ~11k mostly
static targets and the winner is decided by whichever of them happens to be
closest in descriptor space.  Restricting the pool BEFORE retrieval removes
those competitors instead of down-weighting them after they have already won:

    detect likely motion  ->  match likely-moving geometry

not the reverse.

Local descriptors are still built from the FULL endpoint geometry (§10): the
shape context of a moving Gaussian includes its static neighbours, and
rebuilding the KNN inside the subset would change the descriptor itself
rather than only the set of candidates it is compared against.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch

from utils.relational_coupling import (
    SparseRelationalCoupling,
    build_bidirectional_candidate_union,
    initialize_soft_partial_transport,
)


@dataclass
class DynamicCandidateConfidence:
    """Three ways of asking "is this row committed to anything?"."""

    cost_margin: torch.Tensor    # [N] (C_(2) - C_(1)) / (|C_(2)| + |C_(1)|)
    peak_ratio: torch.Tensor     # [N] P_(1) / (P_(1) + P_(2))
    entropy: torch.Tensor        # [N] normalised row entropy, diagnostic only
    peak_mass: torch.Tensor      # [N] P_(1)
    stats: dict[str, Any] = field(default_factory=dict)


def _descriptor_blocks(graph, *, lambda_distance: float, lambda_covariance: float, lambda_appearance: float):
    """Weighted descriptor matrix, so one Euclidean search realises the sum."""
    blocks = [graph.distance_descriptor * float(lambda_distance) ** 0.5]
    if float(lambda_covariance) > 0.0:
        blocks.append(graph.covariance_descriptor * float(lambda_covariance) ** 0.5)
    if float(lambda_appearance) > 0.0:
        appearance = getattr(graph, "appearance", None)
        if appearance is None:
            raise ValueError("lambda_appearance > 0 requires appearance on the graph")
        blocks.append(torch.as_tensor(appearance).detach().float() * float(lambda_appearance) ** 0.5)
    return torch.cat(blocks, dim=1).float()


@torch.no_grad()
def apply_target_dynamic_eligibility(
    cost: torch.Tensor,
    target_weight: torch.Tensor,
    *,
    mode: str = "soft",
    lambda_q: float = 1.0,
    eps: float = 1.0e-6,
) -> torch.Tensor:
    """Fold target-side dynamic eligibility into a ``[n, M_target]`` cost block.

    ``hard`` removes ineligible targets from the search entirely; ``soft``
    charges ``-lambda_q log q^T``, which lets a confident descriptor match
    outvote a middling gate.  §13 asks for both because they fail differently:
    the hard form cannot recover a moved Gaussian whose target-side gate was
    mis-set, and the soft form cannot stop a large static population from
    crowding the top-k.
    """
    if mode == "none":
        return cost
    q = torch.as_tensor(target_weight).detach().to(cost).reshape(1, -1)
    if mode == "hard":
        return torch.where(q > 0.5, cost, torch.full_like(cost, float("inf")))
    if mode == "soft":
        return cost - float(lambda_q) * torch.log(q.clamp_min(float(eps)))
    raise ValueError(f"unknown target eligibility mode {mode!r}")


@torch.no_grad()
def build_dynamic_descriptor_candidates(
    source_graph,
    target_graph,
    *,
    target_weight: torch.Tensor | None = None,
    source_weight: torch.Tensor | None = None,
    max_candidates: int = 32,
    lambda_distance: float = 1.0,
    lambda_covariance: float = 0.2,
    lambda_appearance: float = 0.0,
    target_mode: str = "soft",
    lambda_q: float = 1.0,
    source_threshold: float = 0.05,
    chunk_size: int = 512,
    allow_full_scene: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Top-``M`` dynamic-eligible targets per source. Returns ``(idx, cost, row_active)``.

    Rows whose source-side weight is negligible are still returned (shapes
    stay rectangular) but flagged inactive, so an obviously static Gaussian
    never contributes a fabricated dynamic endpoint.
    """
    if target_weight is None and not bool(allow_full_scene):
        raise ValueError("dynamic candidates need target_weight; pass allow_full_scene=True for the A0/A1 ablation")

    source = _descriptor_blocks(
        source_graph, lambda_distance=lambda_distance,
        lambda_covariance=lambda_covariance, lambda_appearance=lambda_appearance,
    )
    target = _descriptor_blocks(
        target_graph, lambda_distance=lambda_distance,
        lambda_covariance=lambda_covariance, lambda_appearance=lambda_appearance,
    ).to(source.device)
    if int(source.shape[1]) != int(target.shape[1]):
        raise ValueError("source and target descriptors must have matching width")

    n, num_targets = int(source.shape[0]), int(target.shape[0])
    mode = "none" if target_weight is None else str(target_mode)
    k = min(max(1, int(max_candidates)), num_targets)

    costs: list[torch.Tensor] = []
    indices: list[torch.Tensor] = []
    for start in range(0, n, max(1, int(chunk_size))):
        block = torch.cdist(source[start:start + int(chunk_size)], target).square()
        block = apply_target_dynamic_eligibility(block, target_weight, mode=mode, lambda_q=float(lambda_q)) \
            if target_weight is not None else block
        value, index = torch.topk(block, k=k, dim=1, largest=False, sorted=True)
        costs.append(value)
        indices.append(index)
    cost = torch.cat(costs, 0)
    idx = torch.cat(indices, 0)

    if source_weight is None:
        row_active = torch.ones(n, dtype=torch.bool, device=idx.device)
    else:
        row_active = torch.as_tensor(source_weight).detach().to(idx.device).reshape(-1) > float(source_threshold)
    return idx, cost, row_active


@torch.no_grad()
def build_bidirectional_dynamic_candidate_union(
    source_graph,
    target_graph,
    *,
    source_weight: torch.Tensor,
    target_weight: torch.Tensor,
    max_candidates: int = 32,
    backward_candidates: int | None = None,
    lambda_distance: float = 1.0,
    lambda_covariance: float = 0.2,
    lambda_appearance: float = 0.0,
    target_mode: str = "soft",
    lambda_q: float = 1.0,
    source_threshold: float = 0.05,
    chunk_size: int = 512,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """``C_D = C_{0->T} union (C_{T->0})^T``, both directions dynamic-restricted.

    The reverse search takes frame-0 eligibility as ITS target-side mask
    (§14), which is what makes the union a statement about dynamic geometry
    in both frames rather than a one-sided filter applied twice.
    """
    forward_idx, forward_cost, row_active = build_dynamic_descriptor_candidates(
        source_graph, target_graph, target_weight=target_weight, source_weight=source_weight,
        max_candidates=int(max_candidates), lambda_distance=lambda_distance,
        lambda_covariance=lambda_covariance, lambda_appearance=lambda_appearance,
        target_mode=target_mode, lambda_q=lambda_q, source_threshold=source_threshold,
        chunk_size=int(chunk_size),
    )
    backward_idx, backward_cost, _ = build_dynamic_descriptor_candidates(
        target_graph, source_graph, target_weight=source_weight, source_weight=target_weight,
        max_candidates=int(backward_candidates or max(4, int(max_candidates) // 4)),
        lambda_distance=lambda_distance, lambda_covariance=lambda_covariance,
        lambda_appearance=lambda_appearance, target_mode=target_mode, lambda_q=lambda_q,
        source_threshold=source_threshold, chunk_size=int(chunk_size),
    )
    idx, cost, valid = build_bidirectional_candidate_union(
        forward_idx, forward_cost, backward_idx, backward_cost,
        num_sources=int(source_graph.xyz.shape[0]), max_candidates=int(max_candidates),
    )
    return idx, cost, valid & row_active[:, None]


@torch.no_grad()
def compute_dynamic_relational_support(
    source_graph,
    target_graph,
    candidate_idx: torch.Tensor,
    candidate_valid: torch.Tensor,
    *,
    source_weight: torch.Tensor,
    target_weight: torch.Tensor,
    mass: torch.Tensor | None = None,
    bandwidth: float = 1.0,
    edge_chunk: int = 2048,
) -> torch.Tensor:
    """§16: relational support whose neighbour votes are themselves dynamic-weighted.

        S_ij = sum_kl W_kl psi_ij,kl / sum_kl W_kl,   W_kl = q_k^0 q_l^T P_kl

    Without the ``q`` factors a moving Gaussian's mostly-static KNN would
    supply most of the votes, and those neighbours corroborate the candidate
    that did NOT move -- so the support term would actively argue against the
    correct dynamic match.  Bounded compatibility (a reward, not an additive
    error) keeps inconsistent neighbours contributing ~0 instead of swamping
    the signal with residual.
    """
    source_xyz = source_graph.xyz
    target_xyz = target_graph.xyz.to(source_xyz)
    knn = source_graph.knn_rig
    n, m = int(candidate_idx.shape[0]), int(candidate_idx.shape[1])
    scale = source_graph.effective_scale
    pair_scale = torch.sqrt(scale[:, None] * scale[knn]).clamp_min(1.0e-8)
    d0 = source_graph.edge_len_rig

    q0 = torch.as_tensor(source_weight).detach().to(target_xyz).reshape(-1).clamp(0.0, 1.0)
    qT = torch.as_tensor(target_weight).detach().to(target_xyz).reshape(-1).clamp(0.0, 1.0)

    weights = candidate_valid.to(target_xyz.dtype) * qT[candidate_idx.clamp_min(0)]
    if mass is not None:
        weights = weights * mass.to(target_xyz.dtype)

    sigma2 = max(float(bandwidth), 1.0e-6) ** 2
    out = torch.zeros((n, m), device=source_xyz.device, dtype=source_xyz.dtype)
    for start in range(0, n, max(1, int(edge_chunk))):
        stop = min(n, start + int(edge_chunk))
        cand = target_xyz[candidate_idx[start:stop].clamp_min(0)]           # [c, M, 3]
        nbr = knn[start:stop]                                                # [c, K]
        nbr_cand = target_xyz[candidate_idx[nbr].clamp_min(0)]               # [c, K, L, 3]
        nbr_weight = weights[nbr] * q0[nbr][..., None]                       # [c, K, L]
        dT = torch.linalg.norm(cand[:, :, None, None, :] - nbr_cand[:, None, :, :, :], dim=-1)
        # Relative residual per §15, so a long edge is not penalised simply
        # for being long.
        residual = (d0[start:stop][:, None, :, None] - dT) / pair_scale[start:stop][:, None, :, None]
        compatibility = torch.exp(-0.5 * residual.square() / sigma2)         # [c, M, K, L]
        numerator = (compatibility * nbr_weight[:, None, :, :]).sum(dim=(-1, -2))
        denominator = nbr_weight.sum(dim=(-1, -2))[:, None].clamp_min(1.0e-12)
        out[start:stop] = numerator / denominator
    return torch.where(candidate_valid, out, torch.zeros_like(out))


@torch.no_grad()
def initialize_dynamic_partial_transport(
    cost: torch.Tensor,
    valid: torch.Tensor,
    target_idx: torch.Tensor,
    *,
    num_targets: int,
    source_weight: torch.Tensor,
    target_weight: torch.Tensor,
    unmatched_quantile: float = 0.90,
    column_capacity: float = 4.0,
    temperature: float | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Soft partial transport, then the §17 dynamic pair weight ``q_i^0 q_j^T``.

    The pair weight is applied AFTER normalisation and the lost mass is
    returned to the dustbin rather than renormalised away.  Renormalising
    would restore every suppressed row to full confidence, which is exactly
    the "suppress static afterwards" behaviour this stage exists to replace.
    """
    mass, unmatched = initialize_soft_partial_transport(
        cost, valid, num_targets=int(num_targets), target_idx=target_idx,
        temperature=temperature, unmatched_quantile=float(unmatched_quantile),
        column_capacity=float(column_capacity),
    )
    q0 = torch.as_tensor(source_weight).detach().to(mass).reshape(-1, 1).clamp(0.0, 1.0)
    qT = torch.as_tensor(target_weight).detach().to(mass).reshape(-1)[target_idx.clamp_min(0)].clamp(0.0, 1.0)
    weighted = mass * q0 * qT
    unmatched = (unmatched + (mass - weighted).sum(dim=1)).clamp(0.0, 1.0)
    return weighted, unmatched


@torch.no_grad()
def compute_dynamic_candidate_confidence(
    cost: torch.Tensor,
    mass: torch.Tensor,
    valid: torch.Tensor,
    *,
    eps: float = 1.0e-12,
) -> DynamicCandidateConfidence:
    """§19's three confidences, reported side by side rather than blended.

    They disagree, and which one to trust is the empirical question this
    stage has to answer.  Entropy is dominated by the candidate-pool size and
    the single global softmax temperature, so it moved by under 0.01 on a
    synthetic case where the ambiguity was resolved completely; the cost
    margin moved 0.0 -> 1.0 on the same case but was noise on real data.
    Blending them would hide that.
    """
    safe_cost = torch.where(valid, cost, torch.full_like(cost, float("inf")))
    safe_mass = torch.where(valid, mass, torch.zeros_like(mass))
    columns = int(cost.shape[1])

    if columns >= 2:
        top2_cost = torch.topk(safe_cost, k=2, dim=1, largest=False).values
        margin = (top2_cost[:, 1] - top2_cost[:, 0]) / (top2_cost[:, 1].abs() + top2_cost[:, 0].abs()).clamp_min(eps)
        top2_mass = torch.topk(safe_mass, k=2, dim=1).values
        peak_ratio = top2_mass[:, 0] / (top2_mass[:, 0] + top2_mass[:, 1]).clamp_min(eps)
        peak_mass = top2_mass[:, 0]
    else:
        margin = torch.ones(int(cost.shape[0]), device=cost.device, dtype=cost.dtype)
        peak_ratio = torch.ones_like(margin)
        peak_mass = safe_mass.reshape(-1)

    normalised = safe_mass / safe_mass.sum(dim=1, keepdim=True).clamp_min(eps)
    entropy = -(normalised * torch.log(normalised.clamp_min(eps))).sum(dim=1)
    available = valid.sum(dim=1).clamp_min(2).to(entropy.dtype)
    entropy = entropy / torch.log(available)

    margin = torch.where(torch.isfinite(margin), margin, torch.zeros_like(margin)).clamp(0.0, 1.0)
    return DynamicCandidateConfidence(
        cost_margin=margin,
        peak_ratio=torch.where(torch.isfinite(peak_ratio), peak_ratio, torch.full_like(peak_ratio, 0.5)),
        entropy=torch.where(torch.isfinite(entropy), entropy, torch.ones_like(entropy)),
        peak_mass=peak_mass,
        stats={
            "median_cost_margin": float(margin.median()),
            "median_peak_ratio": float(peak_ratio.median()),
            "median_entropy": float(entropy.median()),
            "median_peak_mass": float(peak_mass.median()),
        },
    )


@torch.no_grad()
def peak_target(coupling: SparseRelationalCoupling) -> tuple[torch.Tensor, torch.Tensor]:
    """``(global target index, matched)`` of each row's highest-mass candidate.

    §18: never the posterior mean.  A diffuse row's mean is a centroid, and a
    centroid is not a place anything went -- on an unmoved cloud whose top-1
    matches were all exactly correct, the mean still implied a displacement
    of most of a local scale.
    """
    masked = torch.where(coupling.valid, coupling.mass, torch.full_like(coupling.mass, -1.0))
    peak = masked.argmax(dim=1)
    matched = coupling.valid.gather(1, peak[:, None]).reshape(-1) & (coupling.row_mass > 0)
    return coupling.tgt_idx.gather(1, peak[:, None]).reshape(-1).clamp_min(0), matched


@torch.no_grad()
def compute_relational_rigidity_error(
    source_graph,
    target_xyz: torch.Tensor,
    coupling: SparseRelationalCoupling,
    *,
    evaluation_mask: torch.Tensor | None = None,
    inlier_tolerance: float = 0.25,
) -> dict[str, torch.Tensor]:
    """§23: does the peak transport preserve neighbour distances?

    Only edges whose BOTH endpoints were actually matched are measured.
    Scoring an unmatched endpoint against its own unchanged position would
    manufacture a perfect-rigidity reading out of the transport having
    declined to answer.

    ``evaluation_mask`` must be the SAME for every variant being compared.
    Deriving it from each variant's own dynamic weights instead measures each
    on a different population: a restricted variant evaluated only on its own
    subset scored 17.2 against an unrestricted one's 37.9 on this data, which
    reads as a large improvement and is mostly the population change.
    """
    source_xyz = source_graph.xyz
    target_xyz = torch.as_tensor(target_xyz).detach().to(source_xyz)
    knn = source_graph.knn_rig
    scale = source_graph.effective_scale.clamp_min(1.0e-8)

    tgt, matched = peak_target(coupling)
    moved = target_xyz[tgt]

    d0 = source_graph.edge_len_rig
    dT = torch.linalg.norm(moved[knn] - moved[:, None, :], dim=-1)
    pair_scale = torch.sqrt(scale[:, None] * scale[knn]).clamp_min(1.0e-8)
    error = (dT - d0).abs() / pair_scale

    usable = matched[:, None] & matched[knn]
    if evaluation_mask is not None:
        keep = torch.as_tensor(evaluation_mask).detach().to(device=error.device).reshape(-1).bool()
        usable = usable & keep[:, None] & keep[knn]
    count = usable.sum(dim=1)
    per_point = (error * usable).sum(dim=1) / count.clamp_min(1).to(error.dtype)
    return {
        "per_point_error": torch.where(count > 0, per_point, torch.full_like(per_point, float("nan"))),
        "edge_error": error,
        "edge_usable": usable,
        "inlier_fraction": ((error <= float(inlier_tolerance)) & usable).sum(dim=1) / count.clamp_min(1).to(error.dtype),
        "measured_fraction": float((count > 0).float().mean()),
    }


@torch.no_grad()
def compute_reverse_neighborhood_consistency(
    source_graph,
    coupling: SparseRelationalCoupling,
    reverse_idx: torch.Tensor,
    reverse_valid: torch.Tensor | None = None,
) -> torch.Tensor:
    """§24: does reverse retrieval come back to the same NEIGHBOURHOOD?

        E_i^rev-nbr = min_{k in C_j^B} ||x_i - x_k|| / s_i^eff

    Deliberately a neighbourhood test, not reciprocal identity.  The two
    endpoint models densify and prune independently, so demanding that the
    target's best reverse candidate be literally ``i`` would fail on correct
    matches for reasons that have nothing to do with the motion.
    """
    source_xyz = source_graph.xyz
    scale = source_graph.effective_scale.clamp_min(1.0e-8)
    tgt, matched = peak_target(coupling)

    back = torch.as_tensor(reverse_idx).detach().to(device=source_xyz.device).long()[tgt]  # [N, Kb]
    candidates = source_xyz[back.clamp_min(0)]
    distance = torch.linalg.norm(candidates - source_xyz[:, None, :], dim=-1) / scale[:, None]
    if reverse_valid is not None:
        ok = torch.as_tensor(reverse_valid).detach().to(device=source_xyz.device).bool()[tgt]
        distance = torch.where(ok, distance, torch.full_like(distance, float("inf")))
    best = distance.min(dim=1).values
    return torch.where(matched & torch.isfinite(best), best, torch.full_like(best, float("nan")))


@torch.no_grad()
def build_dynamic_subset_coupling(
    source_graph,
    target_graph,
    *,
    source_weight: torch.Tensor | None,
    target_weight: torch.Tensor | None,
    max_candidates: int = 32,
    lambda_distance: float = 1.0,
    lambda_covariance: float = 0.2,
    lambda_appearance: float = 0.0,
    lambda_relational: float = 1.0,
    relational_sweeps: int = 0,
    relational_bandwidth: float = 1.0,
    target_mode: str = "soft",
    lambda_q: float = 1.0,
    source_threshold: float = 0.05,
    bidirectional_union: bool = False,
    unmatched_quantile: float = 0.90,
    column_capacity: float = 4.0,
    chunk_size: int = 512,
    edge_chunk: int = 2048,
    allow_full_scene: bool = False,
) -> SparseRelationalCoupling:
    """One variant of the A0-A7 ladder, built in the corrected order."""
    n = int(source_graph.xyz.shape[0])
    num_targets = int(target_graph.xyz.shape[0])
    device = source_graph.xyz.device
    ones_src = torch.ones(n, device=device)
    ones_tgt = torch.ones(num_targets, device=device)

    if bool(bidirectional_union):
        if source_weight is None or target_weight is None:
            raise ValueError("the bidirectional union needs both dynamic weights")
        idx, unary, valid = build_bidirectional_dynamic_candidate_union(
            source_graph, target_graph, source_weight=source_weight, target_weight=target_weight,
            max_candidates=int(max_candidates), lambda_distance=lambda_distance,
            lambda_covariance=lambda_covariance, lambda_appearance=lambda_appearance,
            target_mode=target_mode, lambda_q=lambda_q, source_threshold=source_threshold,
            chunk_size=int(chunk_size),
        )
    else:
        idx, unary, row_active = build_dynamic_descriptor_candidates(
            source_graph, target_graph, target_weight=target_weight, source_weight=source_weight,
            max_candidates=int(max_candidates), lambda_distance=lambda_distance,
            lambda_covariance=lambda_covariance, lambda_appearance=lambda_appearance,
            target_mode=target_mode, lambda_q=lambda_q, source_threshold=source_threshold,
            chunk_size=int(chunk_size),
            # `target_weight` is a required keyword here, so a None reaching
            # this point was passed deliberately (the A0/A2 ablations) and
            # cannot be an omission -- which is the only thing the retrieval
            # guard exists to catch.
            allow_full_scene=bool(allow_full_scene) or target_weight is None,
        )
        valid = torch.isfinite(unary) & row_active[:, None]

    q0 = ones_src if source_weight is None else source_weight
    qT = ones_tgt if target_weight is None else target_weight

    total = torch.where(valid, unary, torch.full_like(unary, float("inf")))
    relational = torch.zeros_like(unary)
    mass, unmatched = initialize_dynamic_partial_transport(
        total, valid, idx, num_targets=num_targets, source_weight=q0, target_weight=qT,
        unmatched_quantile=float(unmatched_quantile), column_capacity=float(column_capacity),
    )

    for _ in range(max(0, int(relational_sweeps))):
        support = compute_dynamic_relational_support(
            source_graph, target_graph, idx, valid,
            source_weight=q0, target_weight=qT, mass=mass,
            bandwidth=float(relational_bandwidth), edge_chunk=int(edge_chunk),
        )
        relational = -support
        # Support is dimensionless in [0, 1]; charging it against one robust
        # unit of descriptor cost keeps lambda_relational meaningful without
        # importing a metric constant from another scene.
        finite = unary[valid & torch.isfinite(unary)]
        unit = float((finite - finite.median()).abs().median() * 1.4826) if int(finite.numel()) else 1.0
        total = unary - float(lambda_relational) * max(unit, 1.0e-12) * support
        total = torch.where(valid, total, torch.full_like(total, float("inf")))
        mass, unmatched = initialize_dynamic_partial_transport(
            total, valid, idx, num_targets=num_targets, source_weight=q0, target_weight=qT,
            unmatched_quantile=float(unmatched_quantile), column_capacity=float(column_capacity),
        )

    col_mass = torch.zeros(num_targets, device=mass.device, dtype=mass.dtype)
    if bool(valid.any()):
        col_mass.scatter_add_(0, idx.clamp_min(0)[valid].reshape(-1), mass[valid].reshape(-1))
    return SparseRelationalCoupling(
        src_idx=torch.arange(n, device=idx.device)[:, None].expand_as(idx),
        tgt_idx=idx,
        unary_cost=unary,
        relational_cost=relational,
        total_cost=total,
        mass=mass,
        valid=valid,
        unmatched_src=unmatched,
        row_mass=mass.sum(dim=1),
        col_mass=col_mass,
        num_targets=num_targets,
    )
