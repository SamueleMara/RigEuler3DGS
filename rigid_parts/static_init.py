"""Spatially decaying, randomized initial belief (FINAL spec §7-§10).

Initialising every non-anchor near ``p = [0.5, 0.5, ...]`` gives no physical
first partition, which is what left the v4 runs with a uniform 48/52 mixture.
Instead the trusted static anchors are diffused over the canonical kNN graph to
build a static *affinity*, that affinity biases only the static logit, and every
other logit stays random so discovery is still possible.

Crucially the affinity is used ONLY to construct p^(0) (§10). It never becomes a
persistent static prior: after initialisation every non-anchor Gaussian is fully
trainable static <-> moving.
"""

from __future__ import annotations

import torch

from .neighborhood import NeighborGraph


def diffuse_static_anchors(
    graph: NeighborGraph,
    anchor_mask: torch.Tensor,
    *,
    alpha: float = 0.85,
    iterations: int = 32,
    eps: float = 1.0e-8,
) -> torch.Tensor:
    """§7: graph diffusion of trusted static anchors.

        s^(n+1) = (1-a) s^(0) + a * (sum_j w_ij s_j^(n)) / (sum_j w_ij + eps)

    Anchors stay clamped at 1 throughout. Returns s in [0, 1].
    """
    n = int(graph.num_nodes)
    device = graph.weight.device
    dtype = graph.weight.dtype
    mask = torch.as_tensor(anchor_mask, device=device).bool().reshape(-1)
    if mask.numel() != n:
        raise ValueError("anchor mask does not match the graph node count")

    s0 = mask.to(dtype)
    s = s0.clone()
    if graph.num_edges == 0:
        return s

    # Undirected: accumulate both orientations of every edge.
    src = torch.cat((graph.source, graph.target))
    dst = torch.cat((graph.target, graph.source))
    w = torch.cat((graph.weight, graph.weight))
    degree = torch.zeros(n, dtype=dtype, device=device).scatter_add_(0, src, w).clamp_min(float(eps))

    a = min(max(float(alpha), 0.0), 1.0)
    for _ in range(max(0, int(iterations))):
        neighbour = torch.zeros(n, dtype=dtype, device=device).scatter_add_(0, src, w * s.index_select(0, dst))
        s = (1.0 - a) * s0 + a * (neighbour / degree)
        s = s.clamp(0.0, 1.0)
        s = torch.where(mask, torch.ones_like(s), s)
    return s


def shortest_path_static_affinity(
    graph: NeighborGraph,
    anchor_mask: torch.Tensor,
    *,
    sigma: float = 1.0,
    max_hops: int = 64,
) -> torch.Tensor:
    """§7 alternative: s_i = exp(-d_graph(i, anchors)^2 / (2 sigma^2))."""
    n = int(graph.num_nodes)
    device = graph.weight.device
    mask = torch.as_tensor(anchor_mask, device=device).bool().reshape(-1)
    dist = torch.full((n,), float("inf"), dtype=graph.weight.dtype, device=device)
    dist[mask] = 0.0
    if graph.num_edges:
        src = torch.cat((graph.source, graph.target))
        dst = torch.cat((graph.target, graph.source))
        for _ in range(max(1, int(max_hops))):
            candidate = dist.index_select(0, dst) + 1.0
            updated = dist.scatter_reduce(0, src, candidate, reduce="amin", include_self=True)
            if bool(torch.equal(updated, dist)):
                break
            dist = updated
    finite = torch.isfinite(dist)
    out = torch.zeros_like(dist)
    s = max(float(sigma), 1.0e-8)
    out[finite] = torch.exp(-dist[finite].square() / (2.0 * s * s))
    return out


def initial_part_logits(
    static_affinity: torch.Tensor,
    anchor_mask: torch.Tensor,
    num_parts: int,
    *,
    static_gain: float = 4.0,
    init_std: float = 1.0,
    anchor_logit: float = 12.0,
    seed: int = 20260919,
) -> torch.Tensor:
    """§8/§9: spatially biased static logit, everything else random.

        l_i0 = beta_S * s_i + eps_i0
        l_ik = eps_ik                      (k = 1..K, independent)

    Independent random moving logits are exactly §9's symmetry break: the moving
    mass ``m_i = 1 - p_i0`` is split by ``softmax_{k>0}(r_ik)`` rather than all
    landing on part 1.  Far from anchors the static boost vanishes, which means
    *less evidence for static* -- not automatically moving (§8).
    """
    affinity = torch.as_tensor(static_affinity).reshape(-1)
    n = int(affinity.numel())
    k_plus_1 = int(num_parts)
    if k_plus_1 < 2:
        raise ValueError("need at least one static and one moving class")

    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed))
    noise = torch.randn((n, k_plus_1), generator=generator).to(affinity) * float(init_std)

    logits = noise.clone()
    logits[:, 0] = float(static_gain) * affinity + noise[:, 0]

    # Trusted anchors start near p_i0 ~= 0.999 but are NOT hard pinned (§11).
    mask = torch.as_tensor(anchor_mask, device=affinity.device).bool().reshape(-1)
    if bool(mask.any()):
        logits[mask] = 0.0
        logits[mask, 0] = float(anchor_logit)
    return logits
