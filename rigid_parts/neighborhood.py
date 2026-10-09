"""Spatial neighbourhood graph for the local Killing field (FIRM v3 §3, §4, §8).

The v3 contract makes rigidity *local*: strain, rotation and translation are
per-Gaussian quantities, and coherence is enforced across neighbour edges
weighted by

    w_ij^(k) = w_ij^spatial * p_ik * p_jk        (§3)

so the current memberships decide which neighbourhoods train which field, and
no global twist is ever imposed before the support is discovered (§8).
"""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class NeighborGraph:
    """Symmetric kNN edge list on the canonical Gaussian centres."""

    source: torch.Tensor  # [E] long
    target: torch.Tensor  # [E] long
    weight: torch.Tensor  # [E] float, w_ij^spatial
    num_nodes: int
    k: int

    @property
    def num_edges(self) -> int:
        return int(self.source.numel())

    def to(self, *args, **kwargs) -> "NeighborGraph":
        return NeighborGraph(
            source=self.source.to(*args, **kwargs),
            target=self.target.to(*args, **kwargs),
            weight=self.weight.to(*args, **kwargs),
            num_nodes=self.num_nodes,
            k=self.k,
        )


def _knn_chunked(points: torch.Tensor, k: int, chunk_size: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Return (idx, dist2) of the k nearest neighbours, excluding self."""
    n = int(points.shape[0])
    k_eff = min(max(1, int(k)), max(1, n - 1))
    idx_rows = []
    dist_rows = []
    for start in range(0, n, max(1, int(chunk_size))):
        stop = min(n, start + max(1, int(chunk_size)))
        block = points[start:stop]
        d2 = torch.cdist(block, points).square()
        # Exclude self by pushing the diagonal to +inf.
        rows = torch.arange(start, stop, device=points.device)
        d2[torch.arange(stop - start, device=points.device), rows] = float("inf")
        vals, idx = torch.topk(d2, k_eff, dim=1, largest=False)
        idx_rows.append(idx)
        dist_rows.append(vals)
    return torch.cat(idx_rows, dim=0), torch.cat(dist_rows, dim=0)


def build_neighbor_graph(
    points: torch.Tensor,
    *,
    k: int = 16,
    chunk_size: int = 4096,
    sigma_scale: float = 1.0,
) -> NeighborGraph:
    """Build a symmetric kNN graph with Gaussian spatial weights.

    ``w_ij^spatial = exp(-||x_i - x_j||^2 / (2 sigma^2))`` with ``sigma`` set from
    the median neighbour distance, so the weights are scale-free.
    """
    pts = torch.as_tensor(points).detach()
    n = int(pts.shape[0])
    if n < 2:
        empty = torch.zeros(0, dtype=torch.long, device=pts.device)
        return NeighborGraph(empty, empty, pts.new_zeros(0), n, 0)

    idx, dist2 = _knn_chunked(pts, k, chunk_size)
    k_eff = int(idx.shape[1])
    src = torch.arange(n, device=pts.device)[:, None].expand(-1, k_eff).reshape(-1)
    dst = idx.reshape(-1)
    d2 = dist2.reshape(-1)

    finite = torch.isfinite(d2)
    src, dst, d2 = src[finite], dst[finite], d2[finite]

    # Symmetrise: keep each undirected edge once (i < j), so a pair is not
    # double counted when only one of the two lists it.
    lo = torch.minimum(src, dst)
    hi = torch.maximum(src, dst)
    key = lo.to(torch.int64) * int(n) + hi.to(torch.int64)
    uniq, inverse = torch.unique(key, return_inverse=True)
    keep = torch.zeros(uniq.numel(), dtype=torch.long, device=pts.device)
    keep.scatter_(0, inverse, torch.arange(inverse.numel(), device=pts.device))
    src, dst, d2 = lo[keep], hi[keep], d2[keep]

    median = d2.sqrt().median().clamp_min(1.0e-8) if d2.numel() else pts.new_tensor(1.0)
    sigma = (median * float(sigma_scale)).clamp_min(1.0e-8)
    weight = torch.exp(-0.5 * d2 / sigma.square())
    return NeighborGraph(src, dst, weight, n, k_eff)


def edge_part_weights(
    graph: NeighborGraph,
    part_prob_moving: torch.Tensor,
    hard_label: torch.Tensor | None = None,
) -> torch.Tensor:
    """FINAL spec §13:

        w_ij^(k) = w_ij^spatial * 1[z_i=k] * 1[z_j=k] * p_ik * p_jk

    The hard indicators keep local rigidity inside the CURRENT physical support,
    so static and moving Gaussians never regularise each other through the same
    local Jacobian. ``part_prob_moving`` is [N, K] (moving classes only);
    ``hard_label`` is [N] over K+1 classes with 0 = static. Returns [K, E].
    """
    prob = torch.as_tensor(part_prob_moving)
    pi = prob.index_select(0, graph.source)  # [E,K]
    pj = prob.index_select(0, graph.target)
    weight = graph.weight[:, None] * pi * pj
    if hard_label is not None:
        z = torch.as_tensor(hard_label, device=prob.device).reshape(-1)
        k_moving = int(prob.shape[1])
        classes = torch.arange(1, k_moving + 1, device=prob.device)
        in_k = (z[:, None] == classes[None, :]).to(prob.dtype)  # [N,K]
        weight = weight * in_k.index_select(0, graph.source) * in_k.index_select(0, graph.target)
    return weight.transpose(0, 1).contiguous()


def local_coherence_loss(
    values_by_part: torch.Tensor,
    graph: NeighborGraph,
    edge_weight: torch.Tensor,
    *,
    eps: float = 1.0e-8,
) -> torch.Tensor:
    """Sum over parts of edge-weighted local disagreement (§8).

        L = sum_k  [ sum_(i,j) w_ij^(k) ||f_ik - f_jk||^2 ] / [ sum_(i,j) w_ij^(k) + eps ]

    ``values_by_part`` is [K, N, D]; ``edge_weight`` is [K, E].
    """
    if graph.num_edges == 0:
        return values_by_part.sum() * 0.0
    vi = values_by_part.index_select(1, graph.source)  # [K,E,D]
    vj = values_by_part.index_select(1, graph.target)
    diff = (vi - vj).square().sum(dim=-1)  # [K,E]
    num = (edge_weight * diff).sum(dim=1)
    den = edge_weight.sum(dim=1).clamp_min(float(eps))
    return (num / den).sum()


def fit_local_jacobian(
    points: torch.Tensor,
    velocity: torch.Tensor,
    graph: NeighborGraph,
    edge_weight: torch.Tensor,
    *,
    ridge: float = 1.0e-6,
) -> torch.Tensor:
    """Weighted least-squares local Jacobian (§4).

        G_ik = argmin_G sum_j w_ij^(k) || (v_jk - v_ik) - G (x_j - x_i) ||^2

    ``points``/``velocity`` are [K, N, 3]; ``edge_weight`` is [K, E].
    Returns [K, N, 3, 3] with ``G[..., m, d] = d v_m / d x_d``.
    """
    k_parts, n, _ = points.shape
    if graph.num_edges == 0:
        return points.new_zeros((k_parts, n, 3, 3))

    # Accumulate both directions of every undirected edge so each node sees all
    # of its neighbours.
    src = torch.cat((graph.source, graph.target))
    dst = torch.cat((graph.target, graph.source))
    w = torch.cat((edge_weight, edge_weight), dim=1)  # [K, 2E]

    dx = points.index_select(1, dst) - points.index_select(1, src)  # [K,2E,3]
    dv = velocity.index_select(1, dst) - velocity.index_select(1, src)

    wd = w[..., None] * dx  # [K,2E,3]
    # A_i = sum_j w dx dx^T   ;   B_i = sum_j w dv dx^T
    a_edge = wd[..., :, None] * dx[..., None, :]  # [K,2E,3,3]
    b_edge = dv[..., :, None] * wd[..., None, :]  # [K,2E,3,3]

    index = src[None, :, None, None].expand(k_parts, -1, 3, 3)
    a = torch.zeros((k_parts, n, 3, 3), dtype=points.dtype, device=points.device)
    b = torch.zeros_like(a)
    a.scatter_add_(1, index, a_edge)
    b.scatter_add_(1, index, b_edge)

    eye = torch.eye(3, dtype=points.dtype, device=points.device)
    trace = torch.diagonal(a, dim1=-2, dim2=-1).sum(-1).clamp_min(0.0)
    damping = (float(ridge) * (trace / 3.0 + 1.0))[..., None, None]
    return torch.linalg.solve(a + damping * eye, b.transpose(-1, -2)).transpose(-1, -2)
