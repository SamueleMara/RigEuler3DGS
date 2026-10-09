"""CODEX_endpoint_contrastive_initialization -- static/dynamic first, parts second.

The previous initialisation (``static_init.diffuse_static_anchors`` +
``initial_part_logits``) biased only the static logit from diffused trusted
anchors and left every moving logit random, so the *moving* support the first
Killing solve received was essentially noise.  This module implements the spec's
replacement: learn a single binary ``p^dyn`` per Gaussian from endpoint geometry
and local contrast (sections 2-16), then -- and only then -- split the predicted
dynamic support into the known K parts (sections 17-22).

Section 12 forbids render loss, Killing loss, COM, inertia, mass, motion
integration and outer-loop memory from participating here; the only data signal
is the initial-to-final nearest-neighbour distance (section 29).  Ground truth is
read exclusively by :func:`evaluate_binary_against_gt`, after optimisation
(section 26).
"""

from __future__ import annotations

import csv
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .neighborhood import NeighborGraph, build_neighbor_graph
from .static_anchor_init import _chunked_argmin

EPS = 1.0e-6


# ---------------------------------------------------------------------------
# Section 2 / 3 -- initial-to-final geometric mismatch
# ---------------------------------------------------------------------------


@dataclass
class EndpointMatching:
    """Per-initial-Gaussian nearest final Gaussian (section 2)."""

    distance: torch.Tensor            # [N]  d_i
    index: torch.Tensor               # [N]  j*_i
    normalized: torch.Tensor          # [N]  d_i / (D_bbox + eps)
    bbox_diag: torch.Tensor           # scalar D_bbox

    def as_state(self) -> dict[str, torch.Tensor]:
        return {
            "endpoint_nn_distance": self.distance.detach().cpu(),
            "endpoint_nn_index": self.index.detach().cpu(),
            "endpoint_nn_distance_normalized": self.normalized.detach().cpu(),
            "bbox_diagonal": self.bbox_diag.detach().cpu(),
        }


@torch.no_grad()
def compute_endpoint_matching(
    xyz_initial: torch.Tensor,
    xyz_final: torch.Tensor,
    *,
    chunk_size: int = 2048,
) -> EndpointMatching:
    """d_i = min_j ||x_i^0 - y_j^T||, normalised by the initial bbox diagonal.

    Section 3 forbids a dense ``N x M`` matrix; ``_chunked_argmin`` is the
    repository's existing chunked GPU nearest-neighbour, already used by the
    trusted static anchors, so both use identical numerics.  Both clouds are
    centred on the initial-cloud mean first: a COLMAP scene far from the origin
    otherwise loses the float32 precision these small distances depend on.
    """
    x0 = torch.as_tensor(xyz_initial).detach().float()
    yT = torch.as_tensor(xyz_final).detach().float().to(x0.device)

    origin = x0.mean(dim=0, keepdim=True)
    a, b = x0 - origin, yT - origin
    bbox = torch.linalg.norm(a.max(dim=0).values - a.min(dim=0).values)

    index, distance = _chunked_argmin(a, b, int(chunk_size))
    normalized = distance / (bbox + EPS)
    return EndpointMatching(distance=distance, index=index,
                            normalized=normalized, bbox_diag=bbox)


# ---------------------------------------------------------------------------
# Section 6 -- weak logit initialisation from the endpoint-distance ordering
# ---------------------------------------------------------------------------


def rank_percentile(values: torch.Tensor) -> torch.Tensor:
    """r_i = rankPercentile(v_i) in [0, 1], deterministic on ties."""
    v = torch.as_tensor(values).reshape(-1)
    n = int(v.numel())
    if n <= 1:
        return torch.full_like(v, 0.5)
    order = torch.argsort(v, stable=True)
    rank = torch.empty(n, dtype=v.dtype, device=v.device)
    rank[order] = torch.arange(n, dtype=v.dtype, device=v.device)
    return rank / float(n - 1)


def initial_binary_logits(normalized_distance: torch.Tensor, *, eps: float = EPS) -> torch.Tensor:
    """l_i^(0) = logit(clamp(rankPercentile(d_bar_i), eps, 1-eps)) (section 6).

    Deliberately weak: a pure ordering with no magnitude information, so the
    contrastive terms below are free to reshape it.  Never random (section 6).
    """
    r = rank_percentile(normalized_distance).clamp(float(eps), 1.0 - float(eps))
    return torch.log(r / (1.0 - r))


# ---------------------------------------------------------------------------
# Section 7 -- endpoint-matching similarity on the canonical KNN edges
# ---------------------------------------------------------------------------


def endpoint_edge_similarity(
    graph: NeighborGraph,
    normalized_distance: torch.Tensor,
    *,
    sigma_floor: float = 1.0e-6,
) -> tuple[torch.Tensor, torch.Tensor]:
    """s_ij = exp(-(d_bar_i - d_bar_j)^2 / (2 sigma_d^2)), sigma_d = median|delta|.

    Returns ``(s, sigma_d)``.  The median is taken over the KNN edges only, so
    sigma_d is a robust graph statistic rather than an absolute world unit.
    """
    d = torch.as_tensor(normalized_distance).reshape(-1).to(graph.weight)
    if graph.num_edges == 0:
        return graph.weight.new_zeros(0), graph.weight.new_tensor(float(sigma_floor))
    delta = d.index_select(0, graph.source) - d.index_select(0, graph.target)
    sigma = delta.abs().median().clamp_min(float(sigma_floor))
    s = torch.exp(-delta.square() / (2.0 * sigma.square()))
    return s, sigma


# ---------------------------------------------------------------------------
# Section 10 -- pseudo-anchors
# ---------------------------------------------------------------------------


@dataclass
class PseudoAnchors:
    """Which Gaussians weakly orient the binary field (section 10)."""

    trusted_static: torch.Tensor      # [N] bool, A_S^trusted
    geometric_static: torch.Tensor    # [N] bool, A_S^geom \ A_S^trusted
    static_union: torch.Tensor        # [N] bool, A_S
    dynamic: torch.Tensor             # [N] bool, A_D
    tau_static: torch.Tensor          # scalar Q_static_quantile(d_bar)
    tau_dynamic: torch.Tensor         # scalar Q_{1-dynamic_quantile}(d_bar)


@torch.no_grad()
def build_pseudo_anchors(
    normalized_distance: torch.Tensor,
    trusted_static_mask: torch.Tensor | None,
    *,
    static_quantile: float = 0.10,
    dynamic_quantile: float = 0.10,
) -> PseudoAnchors:
    """tau_S = Q_{static_quantile}(d_bar), tau_D = Q_{1-dynamic_quantile}(d_bar).

    ``dynamic_quantile`` is the *upper tail fraction* (spec: the flag is 0.10 and
    the threshold is ``Q_0.90``), so both flags read as "how much of each tail".
    """
    d = torch.as_tensor(normalized_distance).reshape(-1)
    n = int(d.numel())
    q_s = min(max(float(static_quantile), 0.0), 1.0)
    q_d = min(max(float(dynamic_quantile), 0.0), 1.0)
    tau_s = torch.quantile(d.float(), q_s)
    tau_d = torch.quantile(d.float(), 1.0 - q_d)

    geom = d <= tau_s
    dynamic = d >= tau_d

    if trusted_static_mask is None:
        trusted = torch.zeros(n, dtype=torch.bool, device=d.device)
    else:
        trusted = torch.as_tensor(trusted_static_mask, device=d.device).bool().reshape(-1)
        if trusted.numel() != n:
            raise ValueError("trusted static anchor mask does not match the Gaussian count")

    # A trusted anchor that also passes the geometric quantile receives ONLY the
    # trusted (strongest) supervision, never both -- section 11 orders the
    # weights lambda_trusted > lambda_geom, and double counting would invert it.
    geom_only = geom & ~trusted
    # A point cannot be a static and a dynamic pseudo-anchor at once; the trusted
    # test is strictly stronger evidence than the upper-decile heuristic.
    dynamic = dynamic & ~trusted
    return PseudoAnchors(
        trusted_static=trusted,
        geometric_static=geom_only,
        static_union=geom | trusted,
        dynamic=dynamic,
        tau_static=tau_s,
        tau_dynamic=tau_d,
    )


# ---------------------------------------------------------------------------
# Sections 8, 9, 11, 12 -- the binary objective
# ---------------------------------------------------------------------------


def coherence_loss(
    p_dynamic: torch.Tensor,
    graph: NeighborGraph,
    similarity: torch.Tensor,
    *,
    eps: float = EPS,
) -> torch.Tensor:
    """Section 8:  L_same = sum w^x s (p_i - p_j)^2 / (sum w^x s + eps)."""
    if graph.num_edges == 0:
        return p_dynamic.sum() * 0.0
    w = graph.weight * similarity
    diff = (p_dynamic.index_select(0, graph.source)
            - p_dynamic.index_select(0, graph.target)).square()
    return (w * diff).sum() / (w.sum() + float(eps))


def separation_loss(
    p_dynamic: torch.Tensor,
    graph: NeighborGraph,
    similarity: torch.Tensor,
    *,
    margin: float = 0.5,
    eps: float = EPS,
) -> torch.Tensor:
    """Section 9:  L_diff = sum w^x (1-s) [m_p - |p_i-p_j|]_+^2 / (sum w^x (1-s) + eps).

    A soft margin only: once a dissimilar pair is ``margin`` apart it stops
    contributing, so nothing forces dissimilar neighbours to opposite hard labels.
    """
    if graph.num_edges == 0:
        return p_dynamic.sum() * 0.0
    w = graph.weight * (1.0 - similarity)
    gap = (p_dynamic.index_select(0, graph.source)
           - p_dynamic.index_select(0, graph.target)).abs()
    hinge = (float(margin) - gap).clamp_min(0.0)
    return (w * hinge.square()).sum() / (w.sum() + float(eps))


def _negative_log_mean(values: torch.Tensor, mask: torch.Tensor, *, eps: float = EPS) -> torch.Tensor:
    if not bool(mask.any()):
        return values.sum() * 0.0
    return -torch.log(values[mask] + float(eps)).mean()


def anchor_losses(
    p_dynamic: torch.Tensor,
    anchors: PseudoAnchors,
    *,
    eps: float = EPS,
) -> dict[str, torch.Tensor]:
    """Section 11: L_S = -mean log(1-p), L_D = -mean log(p), split by anchor kind."""
    static = 1.0 - p_dynamic
    return {
        "trusted_static": _negative_log_mean(static, anchors.trusted_static, eps=eps),
        "geometric_static": _negative_log_mean(static, anchors.geometric_static, eps=eps),
        # The union value is what section 13 asks to log as ``loss_static_anchor``.
        "static": _negative_log_mean(static, anchors.static_union, eps=eps),
        "dynamic": _negative_log_mean(p_dynamic, anchors.dynamic, eps=eps),
    }


@dataclass
class BinaryInitResult:
    """Section 15/16 -- the learned binary static/dynamic field."""

    logits: torch.Tensor              # [N] l_i
    p_dynamic: torch.Tensor           # [N] sigma(l_i)
    binary_label: torch.Tensor        # [N] long, 0 static / 1 dynamic
    anchors: PseudoAnchors
    similarity: torch.Tensor          # [E] s_ij
    sigma_d: torch.Tensor             # scalar
    history: list[dict[str, float]] = field(default_factory=list)
    summary: dict[str, Any] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)


def optimize_binary_dynamic(
    graph: NeighborGraph,
    normalized_distance: torch.Tensor,
    anchors: PseudoAnchors,
    *,
    iterations: int = 300,
    lr: float = 1.0e-2,
    margin: float = 0.5,
    lambda_same: float = 1.0,
    lambda_diff: float = 0.5,
    lambda_static: float = 1.0,
    lambda_dynamic: float = 0.25,
    lambda_trusted_static: float = 2.0,
    log_every: int = 10,
    log_fn=None,
) -> BinaryInitResult:
    """Sections 12-15: Adam on one scalar logit per Gaussian, nothing else.

    No renderer, no Killing solver, no outer-loop memory participates; the only
    trainable tensor is ``logits`` (section 13).
    """
    d = torch.as_tensor(normalized_distance).reshape(-1).to(graph.weight)
    similarity, sigma_d = endpoint_edge_similarity(graph, d)

    logits = initial_binary_logits(d).clone().to(graph.weight).requires_grad_(True)
    optimizer = torch.optim.Adam([logits], lr=max(float(lr), 1.0e-8))

    history: list[dict[str, float]] = []
    steps = max(0, int(iterations))
    every = max(1, int(log_every))

    def _record(step: int, parts: dict[str, torch.Tensor], total: torch.Tensor,
                p: torch.Tensor) -> None:
        history.append({
            "iteration": int(step),
            "loss_total": float(total.detach()),
            "loss_same": float(parts["same"].detach()),
            "loss_diff": float(parts["diff"].detach()),
            "loss_static_anchor": float(parts["static"].detach()),
            "loss_trusted_static_anchor": float(parts["trusted_static"].detach()),
            "loss_geometric_static_anchor": float(parts["geometric_static"].detach()),
            "loss_dynamic_anchor": float(parts["dynamic"].detach()),
            "mean_p_dynamic": float(p.detach().mean()),
            "fraction_p_dynamic_gt_0_5": float((p.detach() > 0.5).float().mean()),
        })

    for step in range(steps + 1):
        p = torch.sigmoid(logits)
        parts = anchor_losses(p, anchors)
        parts["same"] = coherence_loss(p, graph, similarity)
        parts["diff"] = separation_loss(p, graph, similarity, margin=margin)
        total = (
            float(lambda_same) * parts["same"]
            + float(lambda_diff) * parts["diff"]
            + float(lambda_trusted_static) * parts["trusted_static"]
            + float(lambda_static) * parts["geometric_static"]
            + float(lambda_dynamic) * parts["dynamic"]
        )
        if step % every == 0 or step == steps:
            _record(step, parts, total, p)
            if log_fn is not None:
                row = history[-1]
                log_fn(f"[Init binary] it {step:4d}  L {row['loss_total']:.5f} "
                       f"(same {row['loss_same']:.5f} diff {row['loss_diff']:.5f} "
                       f"S {row['loss_static_anchor']:.5f} D {row['loss_dynamic_anchor']:.5f})  "
                       f"mean p {row['mean_p_dynamic']:.4f}  f_D {row['fraction_p_dynamic_gt_0_5']:.4f}")
        if step == steps:
            break
        optimizer.zero_grad(set_to_none=True)
        total.backward()
        optimizer.step()

    with torch.no_grad():
        final_logits = logits.detach().clone()
        p_dynamic = torch.sigmoid(final_logits)
        binary = (p_dynamic > 0.5).long()          # section 15: fixed 0.5 threshold

    # Section 14: report the collapse, never silently repair it.
    fraction = float(binary.float().mean())
    warnings: list[str] = []
    if fraction < 0.001 or fraction > 0.999:
        message = (f"binary initialization collapsed: f_D = {fraction:.6f} "
                   f"(outside [0.001, 0.999]); the result is NOT replaced by a heuristic")
        warnings.append(message)
        if log_fn is not None:
            log_fn(f"[Init binary] WARNING: {message}")

    quantiles = torch.tensor([0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0], device=p_dynamic.device)
    summary = {
        "iterations": steps,
        "lr": float(lr),
        "sigma_d": float(sigma_d),
        "fraction_dynamic": fraction,
        "mean_p_dynamic": float(p_dynamic.mean()),
        "median_p_dynamic": float(p_dynamic.median()),
        "p_dynamic_quantile_levels": quantiles.tolist(),
        "p_dynamic_quantiles": torch.quantile(p_dynamic.float(), quantiles).tolist(),
        "num_trusted_static_anchors": int(anchors.trusted_static.sum()),
        "num_geometric_static_anchors": int(anchors.geometric_static.sum()),
        "num_static_anchors_union": int(anchors.static_union.sum()),
        "num_dynamic_pseudo_anchors": int(anchors.dynamic.sum()),
        "tau_static": float(anchors.tau_static),
        "tau_dynamic": float(anchors.tau_dynamic),
        "weights": {
            "lambda_init_same": float(lambda_same),
            "lambda_init_diff": float(lambda_diff),
            "lambda_init_static": float(lambda_static),
            "lambda_init_dynamic": float(lambda_dynamic),
            "lambda_init_trusted_static": float(lambda_trusted_static),
            "init_contrastive_margin": float(margin),
        },
    }
    return BinaryInitResult(
        logits=final_logits,
        p_dynamic=p_dynamic,
        binary_label=binary,
        anchors=anchors,
        similarity=similarity.detach(),
        sigma_d=sigma_d.detach(),
        history=history,
        summary=summary,
        warnings=warnings,
    )


# ---------------------------------------------------------------------------
# Sections 17-20 -- split ONLY the predicted dynamic support into K parts
# ---------------------------------------------------------------------------


def _kmeanspp_centroids(points: torch.Tensor, weights: torch.Tensor, k: int,
                        generator: torch.Generator) -> torch.Tensor:
    """Deterministic weighted K-means++ seeding (section 18)."""
    n = int(points.shape[0])
    w = weights.clamp_min(0.0)
    if float(w.sum()) <= 0.0:
        w = torch.ones_like(w)
    first = int(torch.multinomial(w.float().cpu(), 1, generator=generator).item())
    centroids = [points[first]]
    for _ in range(1, int(k)):
        d2 = torch.cdist(points, torch.stack(centroids)).square().min(dim=1).values
        prob = (w * d2).clamp_min(0.0)
        if float(prob.sum()) <= 0.0:                 # all points already coincide
            prob = torch.ones(n, device=points.device, dtype=points.dtype)
        pick = int(torch.multinomial(prob.float().cpu(), 1, generator=generator).item())
        centroids.append(points[pick])
    return torch.stack(centroids)


@torch.no_grad()
def weighted_kmeans(
    points: torch.Tensor,
    weights: torch.Tensor,
    k: int,
    *,
    iterations: int = 100,
    tolerance: float = 1.0e-6,
    seed: int = 20260919,
) -> tuple[torch.Tensor, torch.Tensor]:
    """min_mu sum_i w_i min_k ||x_i - mu_k||^2.  Returns ``(centroids, assignment)``."""
    pts = torch.as_tensor(points).float()
    w = torch.as_tensor(weights, dtype=pts.dtype, device=pts.device).reshape(-1)
    k_eff = max(1, min(int(k), int(pts.shape[0])))

    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed))
    centroids = _kmeanspp_centroids(pts, w, k_eff, generator)

    assign = torch.zeros(int(pts.shape[0]), dtype=torch.long, device=pts.device)
    for _ in range(max(1, int(iterations))):
        assign = torch.cdist(pts, centroids).argmin(dim=1)
        mass = torch.zeros(k_eff, dtype=pts.dtype, device=pts.device).scatter_add_(0, assign, w)
        total = torch.zeros((k_eff, 3), dtype=pts.dtype, device=pts.device)
        total.scatter_add_(0, assign[:, None].expand(-1, 3), w[:, None] * pts)
        # An emptied cluster keeps its previous centroid rather than collapsing
        # to the origin, which would silently reduce the effective K.
        alive = mass > 0
        updated = centroids.clone()
        updated[alive] = total[alive] / mass[alive, None]
        shift = float((updated - centroids).norm(dim=-1).max())
        centroids = updated
        if shift <= float(tolerance):
            break
    return centroids, assign


@dataclass
class PartSplitResult:
    """Sections 19-21 -- the initial class labels and soft memberships."""

    class_label: torch.Tensor         # [N] long, z^(0) in 0..K
    membership: torch.Tensor          # [N, K+1] float, exactly normalised
    cluster_index: torch.Tensor       # [N] long, c_i (-1 where K == 1)
    centroids: torch.Tensor           # [K, 3] (empty for K == 1)
    kmeans_used: bool
    diagnostics: list[dict[str, Any]] = field(default_factory=list)


@torch.no_grad()
def split_dynamic_support(
    canonical_xyz: torch.Tensor,
    p_dynamic: torch.Tensor,
    binary_label: torch.Tensor,
    num_moving_parts: int,
    *,
    seed: int = 20260919,
    kmeans_iterations: int = 100,
) -> PartSplitResult:
    """Section 17-21: K-means is fit on the predicted dynamic support only.

    K = 1 bypasses K-means entirely (section 19).  For K > 1 the centroids are
    fit from ``D = {p > 0.5}`` alone; static Gaussians are never part of that
    fit.  They are still *assigned* a nearest centroid afterwards, purely so
    their residual moving mass ``p_i`` has a class to sit on and section 21's
    membership normalises exactly -- their hard label stays 0 either way.
    """
    xyz = torch.as_tensor(canonical_xyz).float()
    p = torch.as_tensor(p_dynamic, dtype=xyz.dtype, device=xyz.device).reshape(-1)
    binary = torch.as_tensor(binary_label, device=xyz.device).reshape(-1).long()
    n = int(xyz.shape[0])
    k = max(1, int(num_moving_parts))
    dynamic = binary == 1

    if k == 1:
        # Section 19: the binary result IS the answer; no clustering is run.
        class_label = binary.clone()
        membership = torch.stack((1.0 - p, p), dim=-1)
        cluster = torch.full((n,), -1, dtype=torch.long, device=xyz.device)
        centroids = xyz.new_zeros((0, 3))
        return PartSplitResult(class_label, membership, cluster, centroids, kmeans_used=False)

    if not bool(dynamic.any()):
        # Nothing to split. Reported, not papered over (section 14's spirit).
        membership = torch.zeros((n, k + 1), dtype=xyz.dtype, device=xyz.device)
        membership[:, 0] = 1.0
        return PartSplitResult(
            torch.zeros(n, dtype=torch.long, device=xyz.device), membership,
            torch.full((n,), -1, dtype=torch.long, device=xyz.device),
            xyz.new_zeros((0, 3)), kmeans_used=False)

    centroids, _ = weighted_kmeans(
        xyz[dynamic], p[dynamic], k, iterations=kmeans_iterations, seed=seed)
    cluster = torch.cdist(xyz, centroids).argmin(dim=1)          # c_i, 0-based

    class_label = torch.where(dynamic, cluster + 1, torch.zeros_like(cluster))  # section 20
    membership = torch.zeros((n, k + 1), dtype=xyz.dtype, device=xyz.device)
    membership[:, 0] = 1.0 - p
    membership.scatter_(1, (cluster + 1)[:, None], p[:, None])   # section 21
    return PartSplitResult(class_label, membership, cluster, centroids, kmeans_used=True)


# ---------------------------------------------------------------------------
# Section 22 -- spatial compactness diagnostics
# ---------------------------------------------------------------------------


@torch.no_grad()
def count_connected_components(graph: NeighborGraph, member: torch.Tensor) -> int:
    """Connected components of the canonical KNN subgraph induced by ``member``."""
    mask = torch.as_tensor(member, device=graph.weight.device).bool().reshape(-1)
    n = int(graph.num_nodes)
    if not bool(mask.any()):
        return 0
    if graph.num_edges == 0:
        return int(mask.sum())

    keep = mask.index_select(0, graph.source) & mask.index_select(0, graph.target)
    src = torch.cat((graph.source[keep], graph.target[keep]))
    dst = torch.cat((graph.target[keep], graph.source[keep]))

    label = torch.arange(n, device=mask.device)
    for _ in range(n):
        candidate = label.index_select(0, dst)
        updated = label.scatter_reduce(0, src, candidate, reduce="amin", include_self=True)
        if bool(torch.equal(updated, label)):
            break
        label = updated
    return int(torch.unique(label[mask]).numel())


@torch.no_grad()
def part_compactness(
    canonical_xyz: torch.Tensor,
    class_label: torch.Tensor,
    membership: torch.Tensor,
    graph: NeighborGraph,
    num_moving_parts: int,
) -> list[dict[str, Any]]:
    """Section 22, per initialised moving part."""
    xyz = torch.as_tensor(canonical_xyz).float()
    z = torch.as_tensor(class_label, device=xyz.device).reshape(-1)
    prob = torch.as_tensor(membership, dtype=xyz.dtype, device=xyz.device)
    out: list[dict[str, Any]] = []
    for part in range(1, int(num_moving_parts) + 1):
        member = z == part
        count = int(member.sum())
        if count == 0:
            out.append({"part": part, "num_gaussians": 0, "weighted_centroid": [0.0, 0.0, 0.0],
                        "bbox_extent": [0.0, 0.0, 0.0], "mean_radius": 0.0,
                        "num_connected_components": 0})
            continue
        pts = xyz[member]
        w = prob[member, part].clamp_min(0.0)
        mass = w.sum().clamp_min(EPS)
        centroid = (w[:, None] * pts).sum(dim=0) / mass
        extent = pts.max(dim=0).values - pts.min(dim=0).values
        radius = (pts - centroid).norm(dim=-1).mean()
        out.append({
            "part": part,
            "num_gaussians": count,
            "weighted_centroid": centroid.tolist(),
            "bbox_extent": extent.tolist(),
            "mean_radius": float(radius),
            "num_connected_components": count_connected_components(graph, member),
        })
    return out


# ---------------------------------------------------------------------------
# Section 26 -- evaluation-only diagnostics (never touches the objective)
# ---------------------------------------------------------------------------


@torch.no_grad()
def evaluate_binary_against_gt(
    binary_label: torch.Tensor,
    gt_part_labels: torch.Tensor,
    anchors: PseudoAnchors,
) -> dict[str, Any]:
    """Binary accuracy/IoU/precision/recall/F1 and anchor purities (section 26).

    Called only after optimisation.  Ground truth influences no loss, no
    quantile threshold, no K-means and not the 0.5 threshold (section 26).
    """
    pred = torch.as_tensor(binary_label).reshape(-1).long().cpu()
    gt = (torch.as_tensor(gt_part_labels).reshape(-1).long().cpu() > 0).long()
    if pred.numel() != gt.numel():
        return {"available": False,
                "reason": f"gt has {int(gt.numel())} entries for {int(pred.numel())} gaussians"}

    tp = int(((pred == 1) & (gt == 1)).sum())
    fp = int(((pred == 1) & (gt == 0)).sum())
    fn = int(((pred == 0) & (gt == 1)).sum())
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)

    def purity(mask: torch.Tensor, target: int) -> dict[str, float]:
        m = torch.as_tensor(mask).reshape(-1).cpu().bool()
        if not bool(m.any()):
            return {"n": 0, "purity": 0.0}
        return {"n": int(m.sum()), "purity": float((gt[m] == target).float().mean() * 100.0)}

    return {
        "available": True,
        "accuracy": float((pred == gt).float().mean() * 100.0),
        "moving_iou": float(tp / max(tp + fp + fn, 1) * 100.0),
        "precision": float(precision * 100.0),
        "recall": float(recall * 100.0),
        "f1": float(2 * precision * recall / max(precision + recall, 1.0e-9) * 100.0),
        "predicted_dynamic_fraction": float((pred == 1).float().mean() * 100.0),
        "gt_dynamic_fraction": float((gt == 1).float().mean() * 100.0),
        "trusted_static_anchor_purity": purity(anchors.trusted_static, 0),
        "geometric_static_anchor_purity": purity(anchors.geometric_static, 0),
        "dynamic_pseudo_anchor_purity": purity(anchors.dynamic, 1),
    }


@torch.no_grad()
def evaluate_parts_against_gt(class_label: torch.Tensor, gt_part_labels: torch.Tensor,
                             num_moving_parts: int) -> dict[str, Any]:
    """Optional K>1 cluster purity after optimal permutation matching (section 26)."""
    k = int(num_moving_parts)
    if k < 2:
        return {"available": False, "reason": "K == 1 has no part permutation"}
    pred = torch.as_tensor(class_label).reshape(-1).long().cpu().numpy()
    gt = torch.as_tensor(gt_part_labels).reshape(-1).long().cpu().numpy()
    if pred.shape != gt.shape:
        return {"available": False, "reason": "gt length mismatch"}

    moving = (pred > 0) & (gt > 0)
    if not moving.any():
        return {"available": False, "reason": "no gaussian is moving in both"}
    confusion = np.zeros((k, int(gt.max()) + 1), dtype=np.int64)
    np.add.at(confusion, (pred[moving] - 1, gt[moving]), 1)
    try:
        from scipy.optimize import linear_sum_assignment

        rows, cols = linear_sum_assignment(-confusion)
        matched = int(confusion[rows, cols].sum())
    except Exception:                                       # scipy is optional here
        matched = int(confusion.max(axis=1).sum())
        rows, cols = np.arange(k), confusion.argmax(axis=1)
    return {
        "available": True,
        "num_jointly_moving": int(moving.sum()),
        "matched_fraction": float(matched / max(int(moving.sum()), 1) * 100.0),
        "permutation": {int(r) + 1: int(c) for r, c in zip(rows, cols)},
    }


# ---------------------------------------------------------------------------
# Section 25 -- visualisation
# ---------------------------------------------------------------------------


_PART_PALETTE = np.array([
    [160, 160, 160],   # 0 static
    [220,  50,  40],
    [ 40, 120, 220],
    [ 40, 180,  90],
    [230, 170,  30],
    [160,  70, 200],
    [ 30, 190, 190],
    [230, 110, 170],
], dtype=np.uint8)


def _scalar_rgb(values: np.ndarray) -> np.ndarray:
    """Blue (low) -> red (high) ramp over the value range."""
    v = np.asarray(values, dtype=np.float64).reshape(-1)
    lo, hi = float(v.min()), float(v.max())
    t = np.zeros_like(v) if hi - lo < 1.0e-12 else (v - lo) / (hi - lo)
    rgb = np.stack((t, np.zeros_like(t), 1.0 - t), axis=1)
    return np.clip(np.rint(rgb * 255.0), 0, 255).astype(np.uint8)


def _label_rgb(labels: np.ndarray) -> np.ndarray:
    idx = np.clip(np.asarray(labels, dtype=np.int64).reshape(-1), 0, len(_PART_PALETTE) - 1)
    return _PART_PALETTE[idx]


def write_initialization_visualizations(
    directory: Path,
    canonical_xyz: torch.Tensor,
    normalized_distance: torch.Tensor,
    p_dynamic: torch.Tensor,
    binary_label: torch.Tensor,
    class_label: torch.Tensor,
) -> list[str]:
    """Section 25: four PLYs of the initial Gaussians, reusing the repo writer."""
    from utils.static_endpoint_scene import write_labeled_points_ply

    directory.mkdir(parents=True, exist_ok=True)
    xyz = torch.as_tensor(canonical_xyz).detach().cpu().numpy().astype(np.float32)
    d = torch.as_tensor(normalized_distance).detach().cpu().numpy()
    p = torch.as_tensor(p_dynamic).detach().cpu().numpy()
    binary = torch.as_tensor(binary_label).detach().cpu().numpy().astype(np.int64)
    z = torch.as_tensor(class_label).detach().cpu().numpy().astype(np.int64)

    written = []
    for name, rgb, parts in (
        ("init_endpoint_distance.ply", _scalar_rgb(d), binary),
        ("init_p_dynamic.ply", _scalar_rgb(p), binary),
        ("init_binary_label.ply", _label_rgb(binary), binary),
        ("init_class_label.ply", _label_rgb(z), z),
    ):
        path = directory / name
        write_labeled_points_ply(path, xyz, rgb, parts)
        written.append(str(path))
    return written


# ---------------------------------------------------------------------------
# Orchestration -- sections 16, 23, 24, 30
# ---------------------------------------------------------------------------


@dataclass
class EndpointContrastiveInit:
    """Everything the caller needs to start the first Killing solve."""

    matching: EndpointMatching
    binary: BinaryInitResult
    split: PartSplitResult
    graph: NeighborGraph
    summary: dict[str, Any]
    gt_evaluation: dict[str, Any] = field(default_factory=dict)

    @property
    def p_dynamic(self) -> torch.Tensor:
        return self.binary.p_dynamic

    @property
    def class_label(self) -> torch.Tensor:
        return self.split.class_label

    @property
    def membership(self) -> torch.Tensor:
        return self.split.membership

    def part_logits(self, temperature: float = 1.0) -> torch.Tensor:
        """Logits ``l`` with ``softmax(l / temperature) == membership`` (section 21).

        ``PartEmbeddingModel`` parameterises ``p = softmax(logits / tau)``, so the
        initial belief handed downstream is exactly the normalised membership
        this module produced -- not an approximation of it.
        """
        prob = self.split.membership.clamp_min(EPS)
        prob = prob / prob.sum(dim=-1, keepdim=True)
        return max(float(temperature), 1.0e-6) * torch.log(prob)


def _format_summary(summary: dict[str, Any]) -> str:
    """Section 24's required pre-Killing print."""
    q_levels = summary["p_dynamic_quantile_levels"]
    q_values = summary["p_dynamic_quantiles"]
    quantiles = "  ".join(f"q{l:.2f}={v:.4f}" for l, v in zip(q_levels, q_values))
    lines = [
        "[Init endpoint_contrastive] ===== initialization summary =====",
        f"  N total                        {summary['num_total']}",
        f"  N predicted static             {summary['num_predicted_static']}",
        f"  N predicted dynamic            {summary['num_predicted_dynamic']}",
        f"  predicted dynamic fraction     {summary['fraction_dynamic']:.4f}",
        f"  endpoint distance  static      mean {summary['endpoint_distance_static_mean']:.6f}"
        f"  median {summary['endpoint_distance_static_median']:.6f}",
        f"  endpoint distance  dynamic     mean {summary['endpoint_distance_dynamic_mean']:.6f}"
        f"  median {summary['endpoint_distance_dynamic_median']:.6f}",
        f"  trusted static anchors         {summary['num_trusted_static_anchors']}",
        f"  geometric static pseudo-anchors{summary['num_geometric_static_anchors']:>6d}",
        f"  dynamic pseudo-anchors         {summary['num_dynamic_pseudo_anchors']}",
        f"  mean p_dynamic                 {summary['mean_p_dynamic']:.4f}",
        f"  median p_dynamic               {summary['median_p_dynamic']:.4f}",
        f"  p_dynamic quantiles            {quantiles}",
        f"  K moving cluster sizes         {summary['moving_cluster_sizes']}",
    ]
    return "\n".join(lines)


def run_endpoint_contrastive_initialization(
    canonical_xyz: torch.Tensor,
    final_xyz: torch.Tensor,
    num_moving_parts: int,
    *,
    trusted_static_mask: torch.Tensor | None = None,
    args=None,
    result_dir: Path | None = None,
    gt_part_labels: torch.Tensor | None = None,
    log_fn=None,
) -> EndpointContrastiveInit:
    """Run sections 2-26 end to end and return the initial labels.

    ``args`` supplies the section 23 CLI values; every one has the spec default
    so the function is usable standalone (the tests call it that way).
    """
    def opt(name: str, default):
        return default if args is None else type(default)(getattr(args, name, default))

    printer = log_fn if log_fn is not None else print
    xyz = torch.as_tensor(canonical_xyz).detach().float()
    k = max(1, int(num_moving_parts))
    seed = int(getattr(args, "seed", 20260919)) if args is not None else 20260919

    # --- sections 2, 3 ----------------------------------------------------
    matching = compute_endpoint_matching(
        xyz, final_xyz, chunk_size=opt("rigid_part_endpoint_chunk", 2048))

    # --- section 4 --------------------------------------------------------
    graph = build_neighbor_graph(
        xyz,
        k=opt("init_graph_k", 16),
        chunk_size=opt("killing_neighbor_chunk", 4096),
    )

    # --- section 10 -------------------------------------------------------
    anchors = build_pseudo_anchors(
        matching.normalized,
        trusted_static_mask,
        static_quantile=opt("init_static_quantile", 0.10),
        dynamic_quantile=opt("init_dynamic_quantile", 0.10),
    )

    # --- sections 5-15 ----------------------------------------------------
    binary = optimize_binary_dynamic(
        graph,
        matching.normalized,
        anchors,
        iterations=opt("init_binary_iters", 300),
        lr=opt("init_binary_lr", 1.0e-2),
        margin=opt("init_contrastive_margin", 0.5),
        lambda_same=opt("lambda_init_same", 1.0),
        lambda_diff=opt("lambda_init_diff", 0.5),
        lambda_static=opt("lambda_init_static", 1.0),
        lambda_dynamic=opt("lambda_init_dynamic", 0.25),
        lambda_trusted_static=opt("lambda_init_trusted_static", 2.0),
        log_fn=printer,
    )

    # --- sections 17-21 ---------------------------------------------------
    split = split_dynamic_support(
        xyz, binary.p_dynamic, binary.binary_label, k, seed=seed)

    # --- section 22 -------------------------------------------------------
    split.diagnostics = part_compactness(xyz, split.class_label, split.membership, graph, k)

    # --- section 24 -------------------------------------------------------
    dynamic = binary.binary_label == 1
    d = matching.normalized
    def _stat(mask, fn):
        return float(fn(d[mask])) if bool(mask.any()) else float("nan")

    summary = dict(binary.summary)
    summary.update({
        "init_mode": "endpoint_contrastive",
        "num_total": int(xyz.shape[0]),
        "num_predicted_static": int((~dynamic).sum()),
        "num_predicted_dynamic": int(dynamic.sum()),
        "num_moving_parts": k,
        "kmeans_used": bool(split.kmeans_used),
        "init_graph_k": int(graph.k),
        "num_graph_edges": int(graph.num_edges),
        "bbox_diagonal": float(matching.bbox_diag),
        "endpoint_distance_static_mean": _stat(~dynamic, torch.mean),
        "endpoint_distance_static_median": _stat(~dynamic, torch.median),
        "endpoint_distance_dynamic_mean": _stat(dynamic, torch.mean),
        "endpoint_distance_dynamic_median": _stat(dynamic, torch.median),
        "moving_cluster_sizes": [int((split.class_label == c).sum()) for c in range(1, k + 1)],
        "part_compactness": split.diagnostics,
        "warnings": list(binary.warnings),
    })
    printer(_format_summary(summary))
    for part in split.diagnostics:
        printer(f"  part {part['part']}: n={part['num_gaussians']} "
                f"components={part['num_connected_components']} "
                f"mean_radius={part['mean_radius']:.5f} "
                f"bbox={['%.4f' % e for e in part['bbox_extent']]}")

    # --- section 26 (after everything above; never feeds back) ------------
    gt_eval: dict[str, Any] = {"available": False, "reason": "no gt part labels supplied"}
    if gt_part_labels is not None and int(torch.as_tensor(gt_part_labels).numel()) > 0:
        gt_eval = evaluate_binary_against_gt(binary.binary_label, gt_part_labels, anchors)
        if gt_eval.get("available") and k > 1:
            gt_eval["parts"] = evaluate_parts_against_gt(split.class_label, gt_part_labels, k)
        if gt_eval.get("available"):
            printer(f"[Init endpoint_contrastive] GT diagnostics (evaluation only): "
                    f"acc {gt_eval['accuracy']:.2f}  IoU {gt_eval['moving_iou']:.2f}  "
                    f"prec {gt_eval['precision']:.2f}  rec {gt_eval['recall']:.2f}  "
                    f"F1 {gt_eval['f1']:.2f}  "
                    f"pred dyn {gt_eval['predicted_dynamic_fraction']:.1f}% vs "
                    f"gt {gt_eval['gt_dynamic_fraction']:.1f}%")
            printer(f"    anchor purity: trusted static "
                    f"{gt_eval['trusted_static_anchor_purity']['purity']:.2f}% "
                    f"(n={gt_eval['trusted_static_anchor_purity']['n']})  geometric static "
                    f"{gt_eval['geometric_static_anchor_purity']['purity']:.2f}% "
                    f"(n={gt_eval['geometric_static_anchor_purity']['n']})  dynamic "
                    f"{gt_eval['dynamic_pseudo_anchor_purity']['purity']:.2f}% "
                    f"(n={gt_eval['dynamic_pseudo_anchor_purity']['n']})")

    # --- sections 16, 25 --------------------------------------------------
    if result_dir is not None:
        save_initialization_artifacts(Path(result_dir), matching, binary, split, graph,
                                      summary, gt_eval, xyz)

    return EndpointContrastiveInit(matching=matching, binary=binary, split=split,
                                   graph=graph, summary=summary, gt_evaluation=gt_eval)


@dataclass
class SoftInitResult:
    """CODEX_soft_labels §16: the persistent SOFT initial state P^(0)."""

    matching: EndpointMatching
    binary: Any                       # soft_labels.SoftBinaryResult
    part_probs: torch.Tensor          # [N, K+1], sums to 1
    pi: torch.Tensor | None           # [N, K] conditional moving-part probabilities
    centroids: torch.Tensor
    graph: NeighborGraph
    anchors: PseudoAnchors
    summary: dict[str, Any]
    gt_evaluation: dict[str, Any] = field(default_factory=dict)


def run_soft_endpoint_contrastive_initialization(
    canonical_xyz: torch.Tensor,
    final_xyz: torch.Tensor,
    num_moving_parts: int,
    *,
    trusted_static_mask: torch.Tensor | None = None,
    args=None,
    result_dir: Path | None = None,
    gt_part_labels: torch.Tensor | None = None,
    log_fn=None,
) -> SoftInitResult:
    """CODEX_soft_labels sections 3-16, 40, 52.

    Differences from the hard initializer in this module, all mandated:

    * logits start neutral at zero, not at the rank percentile of d_bar
      (section 12) -- the rank construction put half the cloud on each side of
      0.5 by definition;
    * dynamic anchors are the top 5% of endpoint distance, not the top 10%
      (section 5);
    * the objective is pairwise-contrastive + polarization + anchors (sections
      7-11) rather than same/diff coherence;
    * nothing is thresholded at 0.5 and no hard K-means label is stored
      (sections 13, 15, 43);
    * the output is the soft P^(0) that the pipeline keeps as persistent state.

    There is deliberately no ``memory`` parameter (sections 3, 44, 54).
    """
    from .soft_labels import (
        assert_no_memory_in_initialization,
        compose_soft_initial_state,
        initialization_diagnostics,
        soft_spatial_clustering,
        train_soft_binary_labels,
    )

    assert_no_memory_in_initialization()            # section 44, structural guard

    def opt(name: str, default):
        return default if args is None else type(default)(getattr(args, name, default))

    printer = log_fn if log_fn is not None else print
    xyz = torch.as_tensor(canonical_xyz).detach().float()
    k = max(1, int(num_moving_parts))
    seed = int(getattr(args, "seed", 20260919)) if args is not None else 20260919

    matching = compute_endpoint_matching(
        xyz, final_xyz, chunk_size=opt("rigid_part_endpoint_chunk", 2048))
    graph = build_neighbor_graph(
        xyz, k=opt("init_graph_k", 16),
        chunk_size=opt("killing_neighbor_chunk", 4096))

    # Section 5: the top 5% tail only -- deliberately high precision.
    anchors = build_pseudo_anchors(
        matching.normalized, trusted_static_mask,
        static_quantile=opt("init_static_quantile", 0.10),
        dynamic_quantile=1.0 - opt("init_dynamic_anchor_quantile", 0.95),
    )
    similarity, sigma_d = endpoint_edge_similarity(graph, matching.normalized)

    binary = train_soft_binary_labels(
        graph, similarity,
        anchors.trusted_static, anchors.geometric_static, anchors.dynamic,
        num_gaussians=int(xyz.shape[0]),
        # Corrections §8: the weak endpoint-informed logit asymmetry.
        normalized_distance=matching.normalized,
        iterations=opt("init_binary_iters", 300),
        lr=opt("init_binary_lr", 1.0e-2),
        gamma=opt("init_positive_similarity_gamma", 2.0),
        lambda_smooth=opt("lambda_init_smooth", 1.0),
        lambda_polar=opt("lambda_init_polar", 0.5),
        lambda_anchor=opt("lambda_init_anchor", 1.0),
        lambda_trusted_static=opt("lambda_init_trusted_static", 2.0),
        lambda_geometric_static=opt("lambda_init_static", 1.0),
        lambda_dynamic=opt("lambda_init_dynamic", 0.25),
        endpoint_logit_bias=opt("init_endpoint_logit_bias", 0.2),
        polarization_ramp_start=opt("polarization_ramp_start", 0.20),
        polarization_ramp_end=opt("polarization_ramp_end", 0.70),
        log_fn=printer,
        device=xyz.device,
    )

    # Sections 14-16: soft clustering, then compose the persistent soft state.
    pi, centroids = (None, xyz.new_zeros((0, 3)))
    if k > 1:
        pi, centroids = soft_spatial_clustering(
            xyz, binary.q_dynamic, k,
            iterations=opt("init_soft_cluster_iters", 20),
            temperature=opt("init_soft_cluster_temperature", 0.0),
            seed=seed)
    part_probs = compose_soft_initial_state(binary.q, pi, k)

    total = part_probs.sum(dim=1)
    if not bool(torch.allclose(total, torch.ones_like(total), atol=1.0e-5)):
        raise RuntimeError("section 16 violated: P^(0) rows do not sum to 1 "
                           f"(max error {float((total - 1).abs().max()):.3e})")

    summary = dict(binary.summary)
    summary.update({
        "init_mode": "soft_endpoint_contrastive",
        "num_total": int(xyz.shape[0]),
        "num_moving_parts": k,
        "sigma_d": float(sigma_d),
        "init_graph_k": int(graph.k),
        "bbox_diagonal": float(matching.bbox_diag),
        "tau_static": float(anchors.tau_static),
        "tau_dynamic": float(anchors.tau_dynamic),
        "soft_cluster_used": bool(k > 1),
        # Reported for evaluation only; the state itself is never thresholded.
        "argmax_class_counts": [int((part_probs.argmax(dim=1) == c).sum()) for c in range(k + 1)],
    })
    printer(initialization_diagnostics(binary.q, summary))
    printer(f"  argmax class counts (reporting only) {summary['argmax_class_counts']}")

    # Section 52: evaluation-only, after initialization, influencing nothing.
    gt_eval: dict[str, Any] = {"available": False, "reason": "no gt part labels supplied"}
    if gt_part_labels is not None and int(torch.as_tensor(gt_part_labels).numel()) > 0:
        gt_eval = evaluate_binary_against_gt(
            (part_probs.argmax(dim=1) > 0).long(), gt_part_labels, anchors)
        if gt_eval.get("available"):
            printer(f"[Soft init] GT diagnostics (evaluation only): "
                    f"acc {gt_eval['accuracy']:.2f}  IoU {gt_eval['moving_iou']:.2f}  "
                    f"prec {gt_eval['precision']:.2f}  rec {gt_eval['recall']:.2f}  "
                    f"F1 {gt_eval['f1']:.2f}  pred dyn "
                    f"{gt_eval['predicted_dynamic_fraction']:.1f}% vs gt "
                    f"{gt_eval['gt_dynamic_fraction']:.1f}%")
            printer(f"    dynamic pseudo-anchor purity "
                    f"{gt_eval['dynamic_pseudo_anchor_purity']['purity']:.2f}% "
                    f"(n={gt_eval['dynamic_pseudo_anchor_purity']['n']})")

    if result_dir is not None:
        from utils.temporal_visibility_pretraining import _atomic_torch_save

        directory = Path(result_dir) / "initialization"
        directory.mkdir(parents=True, exist_ok=True)
        _atomic_torch_save({"schema_version": 1, **matching.as_state()},
                           directory / "endpoint_matching.pt")
        from .soft_labels import binary_entropy

        _atomic_torch_save({
            "schema_version": 1,
            # Corrections §33.
            "endpoint_distance": matching.distance.detach().cpu(),
            "endpoint_rank": rank_percentile(matching.normalized).detach().cpu(),
            "initial_logits": (None if binary.initial_logits is None
                               else binary.initial_logits),
            "final_logits": binary.logits.detach().cpu(),
            "q_static": binary.q_static.detach().cpu(),
            "q_dynamic": binary.q_dynamic.detach().cpu(),
            "entropy": binary_entropy(binary.q).detach().cpu(),
            "dynamic_anchor_mask": anchors.dynamic.detach().cpu(),
            "static_anchor_mask": anchors.static_union.detach().cpu(),
            "binary_logits": binary.logits.detach().cpu(),
            "q": binary.q.detach().cpu(),
            "soft_part_probs": part_probs.detach().cpu(),
            "pi": None if pi is None else pi.detach().cpu(),
            "cluster_centroids": centroids.detach().cpu(),
            "trusted_static_anchor_mask": anchors.trusted_static.detach().cpu(),
            "geometric_static_anchor_mask": anchors.geometric_static.detach().cpu(),
            "dynamic_pseudo_anchor_mask": anchors.dynamic.detach().cpu(),
            "graph_knn_indices": torch.stack((graph.source, graph.target), dim=1).detach().cpu(),
            "graph_spatial_weights": graph.weight.detach().cpu(),
            "graph_endpoint_similarity": similarity.detach().cpu(),
            "endpoint_nn_distance_normalized": matching.normalized.detach().cpu(),
            "summary": summary,
            "gt_evaluation": gt_eval,
        }, directory / "soft_binary_init.pt")
        with (directory / "soft_init_history.csv").open("w", newline="", encoding="utf-8") as s:
            import csv as _csv

            keys = sorted({key for row in binary.history for key in row}) if binary.history else []
            if keys:
                writer = _csv.DictWriter(s, fieldnames=keys)
                writer.writeheader()
                writer.writerows(binary.history)
        write_initialization_visualizations(
            directory, xyz, matching.normalized, binary.q_dynamic,
            (part_probs.argmax(dim=1) > 0).long(), part_probs.argmax(dim=1))

    return SoftInitResult(matching=matching, binary=binary, part_probs=part_probs,
                          pi=pi, centroids=centroids, graph=graph, anchors=anchors,
                          summary=summary, gt_evaluation=gt_eval)


def save_initialization_artifacts(
    result_dir: Path,
    matching: EndpointMatching,
    binary: BinaryInitResult,
    split: PartSplitResult,
    graph: NeighborGraph,
    summary: dict[str, Any],
    gt_evaluation: dict[str, Any],
    canonical_xyz: torch.Tensor,
) -> Path:
    """Section 16: the two artifacts, plus section 25's visualisations."""
    from utils.temporal_visibility_pretraining import _atomic_torch_save

    directory = Path(result_dir) / "initialization"
    directory.mkdir(parents=True, exist_ok=True)

    _atomic_torch_save({"schema_version": 1, **matching.as_state()},
                       directory / "endpoint_matching.pt")

    knn_indices = torch.stack((graph.source, graph.target), dim=1).detach().cpu()
    _atomic_torch_save({
        "schema_version": 1,
        "endpoint_nn_distance": matching.distance.detach().cpu(),
        "endpoint_nn_distance_normalized": matching.normalized.detach().cpu(),
        "binary_logits": binary.logits.detach().cpu(),
        "p_dynamic": binary.p_dynamic.detach().cpu(),
        "binary_label": binary.binary_label.detach().cpu(),
        "static_anchor_mask": binary.anchors.static_union.detach().cpu(),
        "trusted_static_anchor_mask": binary.anchors.trusted_static.detach().cpu(),
        "geometric_static_anchor_mask": binary.anchors.geometric_static.detach().cpu(),
        "dynamic_pseudo_anchor_mask": binary.anchors.dynamic.detach().cpu(),
        "graph_knn_indices": knn_indices,
        "graph_spatial_weights": graph.weight.detach().cpu(),
        "graph_endpoint_similarity": binary.similarity.detach().cpu(),
        "class_label": split.class_label.detach().cpu(),
        "membership": split.membership.detach().cpu(),
        "cluster_index": split.cluster_index.detach().cpu(),
        "cluster_centroids": split.centroids.detach().cpu(),
        "summary": summary,
        "gt_evaluation": gt_evaluation,
    }, directory / "binary_dynamic_init.pt")

    rows = binary.history
    with (directory / "binary_init_history.csv").open("w", newline="", encoding="utf-8") as stream:
        keys = sorted({key for row in rows for key in row}) if rows else []
        if keys:
            writer = csv.DictWriter(stream, fieldnames=keys)
            writer.writeheader()
            writer.writerows(rows)

    write_initialization_visualizations(
        directory, canonical_xyz, matching.normalized, binary.p_dynamic,
        binary.binary_label, split.class_label)
    return directory
