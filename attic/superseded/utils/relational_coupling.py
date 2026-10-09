"""Sparse soft coupling between two endpoint 3DGS graphs.

``P_ij`` is transport mass / compatibility between a frame-0 and a frame-T
Gaussian.  It is emphatically NOT an assertion that the two are the same
Gaussian: the endpoint models are trained independently, with different
sampling, pruning and densification, so exact identity does not exist and
must never be assumed.  A Gaussian is also allowed to stay unmatched --
forcing a permutation onto two clouds of different size invents
correspondences where the honest answer is "nothing here matches".

Candidate support is retrieved in DESCRIPTOR space, not XYZ space, so a
rigidly moved region can be found however far it travelled.  The relational
term then asks whether a candidate pair preserves the distances to the
neighbourhood's other candidate pairs, which is the part that actually
distinguishes a structurally compatible region from a coincidentally similar
one.

Nothing here is dense: costs live on an ``[N, M]`` candidate list, and the
relational term is evaluated over source-KNN x candidate neighbourhoods with
chunking, never as an ``N^4`` tensor.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class SparseRelationalCoupling:
    """Edge-list coupling. ``src_idx``/``tgt_idx`` are ``[N, M]``."""

    src_idx: torch.Tensor        # [N, M] source index per candidate slot
    tgt_idx: torch.Tensor        # [N, M] target index per candidate slot
    unary_cost: torch.Tensor     # [N, M] descriptor cost
    relational_cost: torch.Tensor  # [N, M] graph-consistency cost
    total_cost: torch.Tensor     # [N, M]
    mass: torch.Tensor           # [N, M] P_ij, rows sum to <= 1
    valid: torch.Tensor          # [N, M] bool -- real candidate slot
    unmatched_src: torch.Tensor  # [N] mass kept by the row dustbin
    row_mass: torch.Tensor       # [N] matched mass per source
    col_mass: torch.Tensor       # [M_target] accumulated mass per target
    num_targets: int

    def expected_target(
        self,
        target_xyz: torch.Tensor,
        *,
        source_xyz: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Posterior-mean target position per source, ``[N, 3]``.

        Rows whose mass all went to the dustbin fall back to no displacement
        rather than to the origin, so an unmatched Gaussian reads as "did not
        move" instead of "teleported to the world origin".  Dividing an
        all-zero numerator by a clamped denominator would put the point at
        the world origin with no indication anything was wrong, which reads
        downstream as an enormous confident displacement -- the same failure
        that made the EM correspondence a no-op before it was found.
        """
        target_xyz = torch.as_tensor(target_xyz).detach()
        gathered = target_xyz[self.tgt_idx.clamp_min(0)]
        weights = (self.mass * self.valid).to(gathered.dtype)
        total = weights.sum(dim=1, keepdim=True)
        empty = total <= torch.finfo(total.dtype).tiny * 10.0
        expected = (weights[..., None] * gathered).sum(dim=1) / total.clamp_min(torch.finfo(total.dtype).tiny)
        if source_xyz is None:
            # No anchor to fall back to: the mass-weighted mean over the row's
            # candidates is at least a location the geometry proposed.
            fallback = gathered.mean(dim=1)
        else:
            fallback = torch.as_tensor(source_xyz).detach().to(expected)
        return torch.where(empty, fallback, expected)


def _chunked_topk_cost(
    source: torch.Tensor,
    target: torch.Tensor,
    k: int,
    *,
    chunk_size: int = 512,
    weights: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Smallest-``k`` squared descriptor distances, chunked over sources."""
    k = min(max(1, int(k)), int(target.shape[0]))
    costs: list[torch.Tensor] = []
    indices: list[torch.Tensor] = []
    for start in range(0, int(source.shape[0]), max(1, int(chunk_size))):
        block = source[start:start + int(chunk_size)]
        if weights is None:
            d2 = torch.cdist(block, target).square()
        else:
            d2 = torch.cdist(block * weights, target * weights).square()
        value, index = torch.topk(d2, k=k, dim=1, largest=False, sorted=True)
        costs.append(value)
        indices.append(index)
    return torch.cat(costs, 0), torch.cat(indices, 0)


def build_descriptor_candidates(
    source_graph,
    target_graph,
    *,
    max_candidates: int = 32,
    lambda_distance: float = 1.0,
    lambda_covariance: float = 0.2,
    lambda_appearance: float = 0.0,
    source_appearance: torch.Tensor | None = None,
    target_appearance: torch.Tensor | None = None,
    chunk_size: int = 512,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Top-``M`` target candidates per source, ranked in descriptor space.

    Weighting each descriptor block by ``sqrt(lambda)`` before a single
    Euclidean search makes the combined cost exactly the weighted sum the
    spec defines, while still being one batched top-k rather than three.
    """
    blocks_src = [source_graph.distance_descriptor * float(lambda_distance) ** 0.5]
    blocks_tgt = [target_graph.distance_descriptor * float(lambda_distance) ** 0.5]
    if float(lambda_covariance) > 0.0:
        blocks_src.append(source_graph.covariance_descriptor * float(lambda_covariance) ** 0.5)
        blocks_tgt.append(target_graph.covariance_descriptor * float(lambda_covariance) ** 0.5)
    if float(lambda_appearance) > 0.0:
        # Default to the appearance the graphs carry, so a caller cannot get
        # the two clouds' colour blocks the wrong way round on the backward
        # pass.  A requested weight with nothing to apply it to is an error,
        # not a silent geometry-only fallback: the run would then report a
        # tuned appearance weight it never actually used.
        source_appearance = source_appearance if source_appearance is not None else getattr(source_graph, "appearance", None)
        target_appearance = target_appearance if target_appearance is not None else getattr(target_graph, "appearance", None)
        if source_appearance is None or target_appearance is None:
            raise ValueError("lambda_appearance > 0 requires appearance on both graphs")
        blocks_src.append(torch.as_tensor(source_appearance).detach().float() * float(lambda_appearance) ** 0.5)
        blocks_tgt.append(torch.as_tensor(target_appearance).detach().float() * float(lambda_appearance) ** 0.5)
    source = torch.cat(blocks_src, dim=1).float()
    target = torch.cat(blocks_tgt, dim=1).float().to(source.device)
    if int(source.shape[1]) != int(target.shape[1]):
        raise ValueError("source and target descriptors must have matching width")
    return _chunked_topk_cost(source, target, int(max_candidates), chunk_size=int(chunk_size))


def build_bidirectional_candidate_union(
    forward_idx: torch.Tensor,
    forward_cost: torch.Tensor,
    backward_idx: torch.Tensor,
    backward_cost: torch.Tensor,
    *,
    num_sources: int,
    max_candidates: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """``C = C_{0->T} union C_{T->0}^T``, kept as a padded ``[N, M]`` list.

    Retrieval is not symmetric: a source's best target need not return the
    favour, and insisting on reciprocity this early throws away exactly the
    asymmetric-but-correct matches that partial visibility produces.  Both
    directions are therefore unioned, and reciprocity is left to the
    relational term to reward rather than to a hard filter to enforce.
    """
    device = forward_idx.device
    n = int(num_sources)
    capacity = int(max_candidates)

    forward_src = torch.arange(n, device=device).repeat_interleave(int(forward_idx.shape[1]))
    forward_tgt = forward_idx.reshape(-1).to(device)
    # Transposed backward retrieval: target j proposed source i, which is the
    # candidate (i, j) exactly as if i had proposed j.
    backward_src = backward_idx.reshape(-1).to(device)
    backward_tgt = torch.arange(int(backward_idx.shape[0]), device=device).repeat_interleave(int(backward_idx.shape[1]))

    src = torch.cat((forward_src, backward_src))
    tgt = torch.cat((forward_tgt, backward_tgt))
    cost = torch.cat((forward_cost.reshape(-1).to(device), backward_cost.reshape(-1).to(device)))

    num_targets = int(backward_idx.shape[0])
    # Deduplicate (i, j) keeping the cheaper of the two retrieval directions:
    # sorting by cost first makes the first occurrence of each key the minimum.
    by_cost = torch.argsort(cost)
    src, tgt, cost = src[by_cost], tgt[by_cost], cost[by_cost]
    key = src * num_targets + tgt
    by_key = torch.argsort(key, stable=True)
    src, tgt, cost, key = src[by_key], tgt[by_key], cost[by_key], key[by_key]
    first = torch.ones_like(key, dtype=torch.bool)
    first[1:] = key[1:] != key[:-1]
    src, tgt, cost = src[first], tgt[first], cost[first]

    # Keep the cheapest `capacity` candidates per source: sort by cost, then
    # stably by source, so each group is already in ascending-cost order.
    by_cost = torch.argsort(cost)
    src, tgt, cost = src[by_cost], tgt[by_cost], cost[by_cost]
    by_src = torch.argsort(src, stable=True)
    src, tgt, cost = src[by_src], tgt[by_src], cost[by_src]
    counts = torch.bincount(src, minlength=n)
    offsets = torch.cumsum(counts, dim=0) - counts
    rank = torch.arange(int(src.numel()), device=device) - offsets[src]
    keep = rank < capacity

    idx = torch.full((n, capacity), -1, dtype=torch.long, device=device)
    out_cost = torch.full((n, capacity), float("inf"), dtype=forward_cost.dtype, device=device)
    idx[src[keep], rank[keep]] = tgt[keep]
    out_cost[src[keep], rank[keep]] = cost[keep]
    valid = idx >= 0
    return idx, out_cost, valid


@torch.no_grad()
def compute_sparse_relational_support(
    source_graph,
    target_graph,
    candidate_idx: torch.Tensor,
    candidate_valid: torch.Tensor,
    *,
    mass: torch.Tensor | None = None,
    bandwidth: float = 1.0,
    edge_chunk: int = 2048,
) -> torch.Tensor:
    """Relational COMPATIBILITY in ``[0, 1]``: how well a pair is corroborated.

    ``support_ij = mean_k  sum_l P_kl exp(-(d0_ik - ||y_j - y_l||)^2 / 2 sigma^2)``

    This is a reward, not a penalty, and that is load-bearing.  Written as an
    additive cost, every candidate ``l`` that the neighbour does NOT actually
    transport to still contributes a large residual, so with a diffuse
    posterior the noise from the wrong candidates swamps the signal from the
    right one and the term ends up preferring whichever ``j`` sits nearest the
    centroid of the neighbourhood's candidate cloud.  As a bounded reward,
    inconsistent candidates simply contribute ~0 and only genuine corroboration
    accumulates -- which is what lets this disambiguate a descriptor tie
    instead of destroying a descriptor match.
    """
    source_xyz = source_graph.xyz
    target_xyz = target_graph.xyz.to(source_xyz)
    knn = source_graph.knn_rig
    n, m = int(candidate_idx.shape[0]), int(candidate_idx.shape[1])
    scale = source_graph.effective_scale
    pair_scale = torch.sqrt(scale[:, None] * scale[knn]).clamp_min(1.0e-8)     # [N, K]
    d0 = source_graph.edge_len_rig                                             # [N, K]

    weights = candidate_valid.to(target_xyz.dtype)
    if mass is not None:
        weights = weights * mass.to(target_xyz.dtype)
    weights = weights / weights.sum(dim=1, keepdim=True).clamp_min(1.0e-12)    # [N, M]

    sigma2 = max(float(bandwidth), 1.0e-6) ** 2
    out = torch.zeros((n, m), device=source_xyz.device, dtype=source_xyz.dtype)
    for start in range(0, n, max(1, int(edge_chunk))):
        stop = min(n, start + int(edge_chunk))
        cand = target_xyz[candidate_idx[start:stop].clamp_min(0)]              # [c, M, 3]
        nbr = knn[start:stop]                                                  # [c, K]
        nbr_cand = target_xyz[candidate_idx[nbr].clamp_min(0)]                 # [c, K, L, 3]
        nbr_weight = weights[nbr]                                              # [c, K, L]
        dT = torch.linalg.norm(cand[:, :, None, None, :] - nbr_cand[:, None, :, :, :], dim=-1)
        residual = (d0[start:stop][:, None, :, None] - dT) / pair_scale[start:stop][:, None, :, None]
        compatibility = torch.exp(-0.5 * residual.square() / sigma2)           # [c, M, K, L]
        per_neighbour = (compatibility * nbr_weight[:, None, :, :]).sum(dim=-1)  # [c, M, K]
        out[start:stop] = per_neighbour.mean(dim=-1)
    return torch.where(candidate_valid, out, torch.zeros_like(out))


@torch.no_grad()
def compute_sparse_relational_cost(
    source_graph,
    target_graph,
    candidate_idx: torch.Tensor,
    candidate_valid: torch.Tensor,
    *,
    mass: torch.Tensor | None = None,
    huber_delta: float = 1.0,
    edge_chunk: int = 4096,
) -> torch.Tensor:
    """How badly a candidate pair distorts its neighbourhood's distances.

    For candidate ``(i, j)`` and each rigidity neighbour ``k`` of ``i``, the
    source distance ``d0_ik`` is compared against ``||y_j - y_l||`` for each
    of ``k``'s own candidates ``l``, weighted by that candidate's current
    mass ``P_kl``.

    The sum is over individual candidates rather than over a mass-weighted
    mean POSITION for ``k``.  That distinction matters: averaging positions
    first blurs a multi-modal posterior into a centroid that no candidate
    actually occupies, so the cost stops being able to say "j is at the right
    distance from where k plausibly went" -- which is exactly the
    disambiguation this term exists to provide when unary descriptors tie.

    Evaluated over source-KNN x candidate neighbourhoods in chunks -- never
    as a dense quadratic-assignment tensor.
    """
    source_xyz = source_graph.xyz
    target_xyz = target_graph.xyz.to(source_xyz)
    knn = source_graph.knn_rig
    n, m = int(candidate_idx.shape[0]), int(candidate_idx.shape[1])
    scale = source_graph.effective_scale
    neighbour_scale = scale[knn]
    pair_scale = torch.sqrt(scale[:, None] * neighbour_scale).clamp_min(1.0e-8)  # [N, K]
    d0 = source_graph.edge_len_rig                                               # [N, K]

    weights = candidate_valid.to(target_xyz.dtype)
    if mass is not None:
        weights = weights * mass.to(target_xyz.dtype)
    weights = weights / weights.sum(dim=1, keepdim=True).clamp_min(1.0e-12)       # [N, M]

    out = torch.zeros((n, m), device=source_xyz.device, dtype=source_xyz.dtype)
    delta = max(float(huber_delta), 1.0e-8)
    for start in range(0, n, max(1, int(edge_chunk))):
        stop = min(n, start + int(edge_chunk))
        cand = target_xyz[candidate_idx[start:stop].clamp_min(0)]                 # [c, M, 3]
        nbr = knn[start:stop]                                                     # [c, K]
        nbr_cand = target_xyz[candidate_idx[nbr].clamp_min(0)]                    # [c, K, L, 3]
        nbr_weight = weights[nbr]                                                 # [c, K, L]
        # [c, M, K, L] distances from each candidate j to each candidate l of k
        dT = torch.linalg.norm(
            cand[:, :, None, None, :] - nbr_cand[:, None, :, :, :], dim=-1
        )
        residual = (d0[start:stop][:, None, :, None] - dT) / pair_scale[start:stop][:, None, :, None]
        absolute = residual.abs()
        robust = torch.where(
            absolute <= delta, 0.5 * absolute.square(), delta * (absolute - 0.5 * delta)
        )
        # Expectation over l under P_kl, then robust mean over the neighbours k.
        per_neighbour = (robust * nbr_weight[:, None, :, :]).sum(dim=-1)          # [c, M, K]
        out[start:stop] = per_neighbour.mean(dim=-1)
    return torch.where(candidate_valid, out, torch.zeros_like(out))


def _row_softmax_with_dustbin(
    cost: torch.Tensor,
    valid: torch.Tensor,
    *,
    temperature: float,
    unmatched_cost: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Row-normalised soft assignment that may keep mass unmatched."""
    temperature = max(float(temperature), 1.0e-8)
    logits = torch.where(valid, -cost / temperature, torch.full_like(cost, -float("inf")))
    dustbin = torch.full((int(cost.shape[0]), 1), -float(unmatched_cost) / temperature, device=cost.device, dtype=cost.dtype)
    full = torch.cat((logits, dustbin), dim=1)
    probability = torch.softmax(full, dim=1)
    return probability[:, :-1], probability[:, -1]


@torch.no_grad()
def initialize_soft_partial_transport(
    cost: torch.Tensor,
    valid: torch.Tensor,
    *,
    num_targets: int,
    target_idx: torch.Tensor,
    temperature: float | None = None,
    unmatched_quantile: float = 0.90,
    column_capacity: float = 4.0,
    capacity_iters: int = 8,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Row-softmax plus column-capacity damping (the spec's first probe).

    Deliberately not Sinkhorn-to-a-permutation: the two clouds have different
    sizes and partial visibility, so columns are *capped* rather than forced
    to a fixed marginal.  Both the temperature and the unmatched cost are
    derived from the cost distribution itself, so nothing here carries a
    hard-coded metric threshold across datasets.
    """
    finite = cost[valid & torch.isfinite(cost)]
    if finite.numel() == 0:
        mass = torch.zeros_like(cost)
        return mass, torch.ones(int(cost.shape[0]), device=cost.device, dtype=cost.dtype)
    if temperature is None:
        median = finite.median()
        spread = (finite - median).abs().median().clamp_min(1.0e-12)
        temperature = float((1.4826 * spread).clamp_min(1.0e-6))
    unmatched_cost = float(torch.quantile(finite, float(unmatched_quantile)))

    mass, unmatched = _row_softmax_with_dustbin(
        cost, valid, temperature=float(temperature), unmatched_cost=unmatched_cost
    )
    # Cap any target that is being claimed by far too many sources; excess
    # claims are pushed back to the dustbin rather than redistributed, so a
    # popular-but-wrong target cannot absorb a whole neighbourhood.
    capacity = float(column_capacity) * max(1.0, float(cost.shape[0]) / max(1, int(num_targets)))
    for _ in range(max(0, int(capacity_iters))):
        col = torch.zeros(int(num_targets), device=cost.device, dtype=cost.dtype)
        col.scatter_add_(0, target_idx.clamp_min(0)[valid].reshape(-1), mass[valid].reshape(-1))
        overload = (col / capacity).clamp_min(1.0)
        scale = 1.0 / overload[target_idx.clamp_min(0)]
        new_mass = torch.where(valid, mass * scale, torch.zeros_like(mass))
        unmatched = unmatched + (mass - new_mass).sum(dim=1)
        mass = new_mass
        if float((overload > 1.0 + 1.0e-6).float().mean()) == 0.0:
            break
    return mass, unmatched.clamp(0.0, 1.0)


@torch.no_grad()
def update_soft_partial_transport(
    coupling: SparseRelationalCoupling,
    source_graph,
    target_graph,
    *,
    lambda_relational: float = 1.0,
    huber_delta: float = 1.0,
    temperature: float | None = None,
    unmatched_quantile: float = 0.90,
    column_capacity: float = 4.0,
    edge_chunk: int = 4096,
) -> SparseRelationalCoupling:
    """One relational refinement sweep: rescore against the current transport."""
    support = compute_sparse_relational_support(
        source_graph, target_graph, coupling.tgt_idx, coupling.valid,
        mass=coupling.mass, bandwidth=float(huber_delta), edge_chunk=int(edge_chunk),
    )
    # Reported separately as the raw distortion, which is the diagnostic the
    # probe needs; the transport itself is driven by the bounded support.
    relational = compute_sparse_relational_cost(
        source_graph, target_graph, coupling.tgt_idx, coupling.valid,
        mass=coupling.mass, huber_delta=float(huber_delta), edge_chunk=int(edge_chunk),
    )
    # Support is dimensionless in [0,1]; converting it into a discount of one
    # robust unit of descriptor cost keeps lambda_relational comparable across
    # scenes without hard-coding any metric threshold.
    finite = coupling.unary_cost[coupling.valid & torch.isfinite(coupling.unary_cost)]
    unary_unit = float((finite - finite.median()).abs().median() * 1.4826) if finite.numel() else 1.0
    unary_unit = max(unary_unit, 1.0e-12)
    total = coupling.unary_cost - float(lambda_relational) * unary_unit * support
    total = torch.where(coupling.valid, total, torch.full_like(total, float("inf")))
    mass, unmatched = initialize_soft_partial_transport(
        total, coupling.valid, num_targets=int(coupling.num_targets), target_idx=coupling.tgt_idx,
        temperature=temperature, unmatched_quantile=float(unmatched_quantile),
        column_capacity=float(column_capacity),
    )
    col_mass = torch.zeros(int(coupling.num_targets), device=mass.device, dtype=mass.dtype)
    col_mass.scatter_add_(0, coupling.tgt_idx.clamp_min(0)[coupling.valid].reshape(-1), mass[coupling.valid].reshape(-1))
    return SparseRelationalCoupling(
        src_idx=coupling.src_idx,
        tgt_idx=coupling.tgt_idx,
        unary_cost=coupling.unary_cost,
        relational_cost=relational,
        total_cost=total,
        mass=mass,
        valid=coupling.valid,
        unmatched_src=unmatched,
        row_mass=mass.sum(dim=1),
        col_mass=col_mass,
        num_targets=int(coupling.num_targets),
    )


@torch.no_grad()
def build_relational_coupling(
    source_graph,
    target_graph,
    *,
    max_candidates: int = 32,
    lambda_distance: float = 1.0,
    lambda_covariance: float = 0.2,
    lambda_appearance: float = 0.0,
    lambda_relational: float = 1.0,
    relational_sweeps: int = 2,
    huber_delta: float = 1.0,
    unmatched_quantile: float = 0.90,
    column_capacity: float = 4.0,
    bidirectional_union: bool = True,
    chunk_size: int = 512,
    edge_chunk: int = 4096,
) -> SparseRelationalCoupling:
    """Descriptor retrieval, optional union, then relational refinement."""
    forward_cost, forward_idx = build_descriptor_candidates(
        source_graph, target_graph, max_candidates=int(max_candidates),
        lambda_distance=float(lambda_distance), lambda_covariance=float(lambda_covariance),
        lambda_appearance=float(lambda_appearance), chunk_size=int(chunk_size),
    )
    n = int(source_graph.xyz.shape[0])
    num_targets = int(target_graph.xyz.shape[0])
    if bool(bidirectional_union):
        backward_cost, backward_idx = build_descriptor_candidates(
            target_graph, source_graph, max_candidates=max(4, int(max_candidates) // 4),
            lambda_distance=float(lambda_distance), lambda_covariance=float(lambda_covariance),
            lambda_appearance=float(lambda_appearance), chunk_size=int(chunk_size),
        )
        idx, unary, valid = build_bidirectional_candidate_union(
            forward_idx, forward_cost, backward_idx, backward_cost,
            num_sources=n, max_candidates=int(max_candidates),
        )
    else:
        idx, unary, valid = forward_idx, forward_cost, torch.ones_like(forward_idx, dtype=torch.bool)

    src_idx = torch.arange(n, device=idx.device)[:, None].expand_as(idx)
    mass, unmatched = initialize_soft_partial_transport(
        unary, valid, num_targets=num_targets, target_idx=idx,
        unmatched_quantile=float(unmatched_quantile), column_capacity=float(column_capacity),
    )
    col_mass = torch.zeros(num_targets, device=mass.device, dtype=mass.dtype)
    col_mass.scatter_add_(0, idx.clamp_min(0)[valid].reshape(-1), mass[valid].reshape(-1))
    coupling = SparseRelationalCoupling(
        src_idx=src_idx,
        tgt_idx=idx,
        unary_cost=unary,
        relational_cost=torch.zeros_like(unary),
        total_cost=unary,
        mass=mass,
        valid=valid,
        unmatched_src=unmatched,
        row_mass=mass.sum(dim=1),
        col_mass=col_mass,
        num_targets=num_targets,
    )
    for _ in range(max(0, int(relational_sweeps))):
        coupling = update_soft_partial_transport(
            coupling, source_graph, target_graph,
            lambda_relational=float(lambda_relational), huber_delta=float(huber_delta),
            unmatched_quantile=float(unmatched_quantile), column_capacity=float(column_capacity),
            edge_chunk=int(edge_chunk),
        )
    return coupling


def compute_transport_entropy(coupling: SparseRelationalCoupling) -> torch.Tensor:
    """Per-source entropy of the row distribution, dustbin included."""
    mass = torch.cat((coupling.mass * coupling.valid, coupling.unmatched_src[:, None]), dim=1)
    mass = mass / mass.sum(dim=1, keepdim=True).clamp_min(1.0e-12)
    return -(mass * torch.log(mass.clamp_min(1.0e-12))).sum(dim=1)


def compute_transport_coverage(
    coupling: SparseRelationalCoupling,
    *,
    mass_threshold: float = 0.05,
) -> dict[str, float]:
    """Coverage, symmetry and distortion diagnostics for one coupling."""
    valid_mass = coupling.mass * coupling.valid
    matched_src = (coupling.row_mass > float(mass_threshold))
    covered_tgt = (coupling.col_mass > float(mass_threshold))
    relational = coupling.relational_cost[coupling.valid]
    entropy = compute_transport_entropy(coupling)
    best = valid_mass.argmax(dim=1)
    best_mass = valid_mass.gather(1, best[:, None]).reshape(-1)
    return {
        "source_coverage": float(matched_src.float().mean()),
        "target_coverage": float(covered_tgt.float().mean()),
        "unmatched_source_fraction": float((~matched_src).float().mean()),
        "unmatched_target_fraction": float((~covered_tgt).float().mean()),
        "mean_row_mass": float(coupling.row_mass.mean()),
        "mean_col_mass": float(coupling.col_mass.mean()),
        "max_col_mass": float(coupling.col_mass.max()),
        "mean_unmatched_mass": float(coupling.unmatched_src.mean()),
        "transport_entropy_mean": float(entropy.mean()),
        "transport_entropy_median": float(entropy.median()),
        "peak_mass_mean": float(best_mass.mean()),
        "relational_distortion_mean": float(relational.mean()) if relational.numel() else 0.0,
        "relational_distortion_p50": float(relational.median()) if relational.numel() else 0.0,
        "relational_distortion_p90": float(torch.quantile(relational, 0.90)) if relational.numel() > 1 else 0.0,
        "relational_distortion_p99": float(torch.quantile(relational, 0.99)) if relational.numel() > 1 else 0.0,
        "candidates_per_source": float(coupling.valid.float().sum(dim=1).mean()),
    }
