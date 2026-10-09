"""A graph over HYPOTHESES, with consistency judged by their own uncertainty.

§17-§23 of the audit.  Two structural corrections, both of which the previous
implementation got wrong in ways that produced confident nonsense.

First, the nodes are hypotheses, not source anchors.  Collapsing each anchor
to its single best target before the graph is built destroys the ambiguity
that the graph, appearance and rendering exist to resolve -- by the time
those stages ran there was nothing left for them to choose between.  A source
anchor may legitimately carry several physically valid hypotheses, and two
hypotheses of the SAME anchor are never connected: they are alternatives, not
evidence for each other.

Second, pairwise consistency uses each hypothesis' own covariance, transported
into a common frame by the SE(3) adjoint.  The previous scene-wide covariance
was circular in the worst way: bad transforms inflate the empirical scatter,
the inflated scatter makes bad transforms mutually compatible, and the test
certifies the noise it was built to reject.  A per-hypothesis covariance
cannot be inflated by other hypotheses being bad.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch

from utils.se3_hypothesis_consistency import _chi2_quantile, se3_exp, se3_log
from utils.transform_uncertainty import compose_inverse_transform, effective_rank, se3_adjoint

EPS = 1.0e-12


@dataclass
class RigidPatchHypothesis:
    """§3: one source-target hypothesis, kept whole rather than collapsed."""

    source_anchor: int
    target_anchor: int
    metric_distortion: float
    metric_pvalue: float
    metric_fdr_pass: bool
    se3_residual: float
    se3_pvalue: float
    se3_fdr_pass: bool
    observability_rank: int
    matched_mass: float
    appearance_distance: float | None = None
    render_loglik_ratio: float | None = None
    accepted_physical: bool = False
    accepted_component: int | None = None


@dataclass
class HypothesisGraph:
    """Nodes, statistically consistent edges, and the components they form."""

    source_anchor: torch.Tensor       # [H]
    target_anchor: torch.Tensor       # [H]
    rotation: torch.Tensor            # [H, 3, 3]
    translation: torch.Tensor         # [H, 3]
    covariance: torch.Tensor          # [H, 6, 6]
    observability_rank: torch.Tensor  # [H]
    edges: torch.Tensor               # [E, 2]
    edge_distance: torch.Tensor       # [E] Mahalanobis d^2
    edge_pvalue: torch.Tensor         # [E]
    component: torch.Tensor           # [H] component id, -1 if unassigned
    component_transform: torch.Tensor # [C, 4, 3] rows 0-2 R, row 3 t
    component_covariance: torch.Tensor
    stats: dict[str, Any] = field(default_factory=dict)


@torch.no_grad()
def pairwise_transform_discrepancy(
    rotation: torch.Tensor,
    translation: torch.Tensor,
    covariance: torch.Tensor,
    pairs: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """§18: ``d^2 = dxi^T Sigma_rel^+ dxi`` with both covariances transported.

    ``Sigma_rel = Sigma_i + Ad Sigma_k Ad^T``.  The adjoint is not optional:
    the two covariances are expressed in their own tangent frames, and adding
    them directly compares numbers written in different bases -- which looks
    like a consistency test and is not one.

    Returns ``(d^2, effective rank, relative tangent)``.
    """
    left, right = pairs[:, 0], pairs[:, 1]
    relative_r, relative_t = compose_inverse_transform(
        rotation[left], translation[left], rotation[right], translation[right]
    )
    tangent = se3_log(relative_r, relative_t)                       # [E, 6]

    adjoint = se3_adjoint(relative_r, relative_t)
    transported = adjoint @ covariance[right] @ adjoint.transpose(1, 2)
    relative_covariance = covariance[left] + transported

    rank, _spectrum = effective_rank(relative_covariance)
    # Pseudoinverse, so an unobservable direction contributes nothing instead
    # of dominating through a near-zero eigenvalue (§7, §19).
    inverse = torch.linalg.pinv(relative_covariance.double())
    distance = torch.einsum("ei,eij,ej->e", tangent.double(), inverse, tangent.double()).float()
    return distance.clamp_min(0.0), rank, tangent


@torch.no_grad()
def build_hypothesis_graph(
    source_anchor: torch.Tensor,
    target_anchor: torch.Tensor,
    rotation: torch.Tensor,
    translation: torch.Tensor,
    covariance: torch.Tensor,
    anchor_xyz: torch.Tensor,
    *,
    accepted: torch.Tensor | None = None,
    neighbours: int = 8,
    alpha: float = 0.05,
    min_anchors_per_component: int = 3,
) -> HypothesisGraph:
    """Connect spatially neighbouring, statistically consistent hypotheses.

    Two conditions, both required (§20): the SOURCE anchors must be spatial
    neighbours, and the transforms must agree under their own uncertainty.
    Hypotheses of the same source anchor are excluded by construction.
    """
    from utils.metric_patch_graph import build_full_geometry_knn

    device = rotation.device
    hypotheses = int(rotation.shape[0])
    keep = torch.ones(hypotheses, dtype=torch.bool, device=device) if accepted is None else accepted.to(device).bool()

    if int(keep.sum()) < 2:
        return _empty_graph(source_anchor, target_anchor, rotation, translation, covariance, hypotheses)

    # Spatial neighbourhood over the distinct source anchors.
    unique_anchors, inverse = torch.unique(source_anchor, return_inverse=True)
    positions = anchor_xyz[unique_anchors]
    k = min(int(neighbours) + 1, int(positions.shape[0]))
    if k < 2:
        return _empty_graph(source_anchor, target_anchor, rotation, translation, covariance, hypotheses)
    knn, _dist = build_full_geometry_knn(positions.cpu(), k)
    neighbour_set = {int(a): set(int(b) for b in row[1:]) for a, row in enumerate(knn.tolist())}

    index = torch.arange(hypotheses, device=device)
    candidate_pairs: list[list[int]] = []
    kept = index[keep].tolist()
    anchor_of = inverse.tolist()
    for position, left in enumerate(kept):
        for right in kept[position + 1:]:
            if anchor_of[left] == anchor_of[right]:
                continue                      # alternatives, never evidence for each other
            if anchor_of[right] not in neighbour_set.get(anchor_of[left], ()):
                continue
            candidate_pairs.append([left, right])

    if not candidate_pairs:
        return _empty_graph(source_anchor, target_anchor, rotation, translation, covariance, hypotheses)

    pairs = torch.tensor(candidate_pairs, device=device, dtype=torch.long)
    distance, rank, _tangent = pairwise_transform_discrepancy(rotation, translation, covariance, pairs)

    # chi-square on the OBSERVABLE tangent dimensions, not a blanket six (§19).
    limits = torch.tensor(
        [_chi2_quantile(max(1, int(r)), 1.0 - float(alpha)) for r in rank.tolist()],
        device=device, dtype=distance.dtype,
    )
    consistent = distance <= limits
    pvalue = torch.clamp(1.0 - distance / limits.clamp_min(EPS), 0.0, 1.0)

    edges = pairs[consistent]
    component = _components_with_anchor_support(
        edges, hypotheses, inverse, int(min_anchors_per_component)
    ).to(device)

    consensus_transform, consensus_covariance = _component_consensus(
        rotation, translation, covariance, component
    )
    rank_all, _ = effective_rank(covariance)
    return HypothesisGraph(
        source_anchor=source_anchor,
        target_anchor=target_anchor,
        rotation=rotation,
        translation=translation,
        covariance=covariance,
        observability_rank=rank_all,
        edges=edges,
        edge_distance=distance[consistent],
        edge_pvalue=pvalue[consistent],
        component=component,
        component_transform=consensus_transform,
        component_covariance=consensus_covariance,
        stats={
            "nodes": int(keep.sum()),
            "candidate_edges": int(pairs.shape[0]),
            "consistent_edges": int(consistent.sum()),
            "edge_acceptance_rate": float(consistent.float().mean()),
            "components": int(consensus_transform.shape[0]),
            "assigned_hypotheses": int((component >= 0).sum()),
            "median_edge_distance": float(distance.median()) if int(distance.numel()) else float("nan"),
            "median_observable_rank": float(rank.float().median()) if int(rank.numel()) else float("nan"),
        },
    )


def _empty_graph(source_anchor, target_anchor, rotation, translation, covariance, hypotheses) -> HypothesisGraph:
    rank, _ = effective_rank(covariance) if int(covariance.shape[0]) else (
        torch.zeros(0, dtype=torch.long), None
    )
    return HypothesisGraph(
        source_anchor=source_anchor, target_anchor=target_anchor,
        rotation=rotation, translation=translation, covariance=covariance,
        observability_rank=rank,
        edges=torch.zeros((0, 2), dtype=torch.long, device=rotation.device),
        edge_distance=torch.zeros(0, device=rotation.device),
        edge_pvalue=torch.zeros(0, device=rotation.device),
        component=torch.full((hypotheses,), -1, dtype=torch.long, device=rotation.device),
        component_transform=torch.zeros((0, 4, 3), device=rotation.device),
        component_covariance=torch.zeros((0, 6, 6), device=rotation.device),
        stats={"nodes": 0, "candidate_edges": 0, "consistent_edges": 0,
               "edge_acceptance_rate": 0.0, "components": 0, "assigned_hypotheses": 0,
               "median_edge_distance": float("nan"), "median_observable_rank": float("nan")},
    )


def _components_with_anchor_support(
    edges: torch.Tensor, hypotheses: int, anchor_of: torch.Tensor, min_anchors: int
) -> torch.Tensor:
    """Union-find, then drop components that do not span enough ANCHORS.

    Anchors, not hypotheses (§21): several hypotheses of one anchor agreeing
    is not spatial support, and a component must cover more than a single
    local patch to be a motion rather than a coincidence.
    """
    parent = list(range(hypotheses))

    def find(a: int) -> int:
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    for u, v in edges.tolist():
        ru, rv = find(int(u)), find(int(v))
        if ru != rv:
            parent[ru] = rv

    groups: dict[int, list[int]] = {}
    for node in set(int(x) for x in edges.reshape(-1).tolist()):
        groups.setdefault(find(node), []).append(node)

    label = torch.full((hypotheses,), -1, dtype=torch.long)
    next_id = 0
    anchors = anchor_of.tolist()
    for members in groups.values():
        if len({anchors[m] for m in members}) < int(min_anchors):
            continue
        for member in members:
            label[member] = next_id
        next_id += 1
    return label


@torch.no_grad()
def _component_consensus(
    rotation: torch.Tensor, translation: torch.Tensor,
    covariance: torch.Tensor, component: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """§22: ``T* = argmin sum_h ||log(T^-1 T_h)||^2_{Sigma_h^-1}``.

    Each hypothesis is weighted by its own information, so a patch that could
    not determine a rotation contributes little to the component's rotation
    and still contributes what it does know about the translation.  No manual
    scalar weights anywhere.
    """
    count = int(component.max()) + 1 if int(component.numel()) and int(component.max()) >= 0 else 0
    transforms = torch.zeros((count, 4, 3), device=rotation.device)
    covariances = torch.zeros((count, 6, 6), device=rotation.device)

    for index in range(count):
        members = (component == index).nonzero(as_tuple=True)[0]
        if int(members.numel()) == 0:
            continue
        information = torch.linalg.pinv(covariance[members].double())
        centre_r, centre_t = rotation[members[0]][None], translation[members[0]][None]
        for _ in range(12):
            relative_r, relative_t = compose_inverse_transform(
                centre_r.expand(int(members.numel()), 3, 3), centre_t.expand(int(members.numel()), 3),
                rotation[members], translation[members],
            )
            tangent = se3_log(relative_r, relative_t).double()
            total_information = information.sum(dim=0)
            weighted = torch.einsum("hij,hj->i", information, tangent)
            step = torch.linalg.pinv(total_information) @ weighted
            if float(torch.linalg.norm(step)) < 1.0e-9:
                break
            step_r, step_t = se3_exp(step.float()[None])
            centre_t = centre_t + torch.einsum("bij,bj->bi", centre_r, step_t)
            centre_r = centre_r @ step_r
        transforms[index, :3, :] = centre_r[0]
        transforms[index, 3, :] = centre_t[0]
        covariances[index] = torch.linalg.pinv(information.sum(dim=0)).float()
    return transforms, covariances


@torch.no_grad()
def hypotheses_per_source(source_anchor: torch.Tensor, accepted: torch.Tensor) -> dict[str, float]:
    """§39: how much ambiguity actually survived, which P7 exists to resolve."""
    kept = source_anchor[accepted.bool()]
    if int(kept.numel()) == 0:
        return {"anchors_with_hypothesis": 0, "mean_per_anchor": 0.0, "max_per_anchor": 0,
                "anchors_ambiguous": 0}
    _unique, counts = torch.unique(kept, return_counts=True)
    return {
        "anchors_with_hypothesis": int(counts.numel()),
        "mean_per_anchor": float(counts.float().mean()),
        "max_per_anchor": int(counts.max()),
        "anchors_ambiguous": int((counts > 1).sum()),
    }
