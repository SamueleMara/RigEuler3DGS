"""Global Potts optimization for static plus K rigid Gaussian labels."""

from __future__ import annotations

import torch


@torch.no_grad()
def _robust_normalize(values, *, valid=None, eps=1.0e-6):
    value = torch.as_tensor(values).detach().cpu().float()
    mask = torch.isfinite(value) if valid is None else (torch.as_tensor(valid).cpu().bool() & torch.isfinite(value))
    finite = value[mask]
    if finite.numel() == 0:
        return torch.zeros_like(value), {"median": 0.0, "scale": 1.0}
    median = finite.median()
    scale = (finite - median).abs().median().clamp_min(float(eps))
    normalized = (value - median) / scale
    normalized = torch.where(mask, normalized, torch.zeros_like(normalized)).clamp(-20.0, 20.0)
    return normalized, {"median": float(median), "scale": float(scale)}


@torch.no_grad()
def normalize_unary_families(families):
    """Robustly normalize each named unary family independently."""
    normalized, statistics = {}, {}
    for name, item in families.items():
        if isinstance(item, dict):
            value, valid = item["values"], item.get("valid")
        else:
            value, valid = item, None
        normalized[name], statistics[name] = _robust_normalize(value, valid=valid)
    return {"families": normalized, "statistics": statistics}


@torch.no_grad()
def assemble_unary_matrix(families, weights):
    if not families:
        raise ValueError("At least one unary family is required")
    result = None
    contributions = {}
    for name, value in families.items():
        weighted = torch.as_tensor(value).detach().cpu().float() * float(weights.get(name, 0.0))
        contributions[name] = weighted
        result = weighted.clone() if result is None else result + weighted
    return {"unary": result, "contributions": contributions}


def _integer_capacity(values, scale):
    value = torch.as_tensor(values).detach().cpu().double().clamp_min(0)
    return torch.round(value * float(scale)).to(torch.int64).numpy()


@torch.no_grad()
def solve_binary_potts_min_cut(unary, pairs, affinity, *, pairwise_weight=1.0, capacity_scale=1.0e5):
    """Exact binary Potts solve using SciPy's maximum-flow implementation."""
    import numpy as np
    from scipy.sparse import csr_matrix
    from scipy.sparse.csgraph import maximum_flow

    cost = torch.as_tensor(unary).detach().cpu().double()
    edge = torch.as_tensor(pairs).detach().cpu().long()
    weight = torch.as_tensor(affinity).detach().cpu().double().clamp_min(0) * float(pairwise_weight)
    if cost.ndim != 2 or cost.shape[1] != 2:
        raise ValueError("binary unary must be [N,2]")
    cost = cost - cost.amin(dim=-1, keepdim=True)
    count, source, sink = cost.shape[0], cost.shape[0], cost.shape[0] + 1
    row = [source] * count + list(range(count))
    col = list(range(count)) + [sink] * count
    data = list(_integer_capacity(cost[:, 1], capacity_scale)) + list(_integer_capacity(cost[:, 0], capacity_scale))
    pair_capacity = _integer_capacity(weight, capacity_scale)
    for (first, second), capacity in zip(edge.tolist(), pair_capacity.tolist()):
        if capacity <= 0:
            continue
        row.extend((first, second)); col.extend((second, first)); data.extend((capacity, capacity))
    capacity = csr_matrix((np.asarray(data, dtype=np.int64), (row, col)), shape=(count + 2, count + 2))
    result = maximum_flow(capacity, source, sink)
    residual = capacity - result.flow
    reachable = np.zeros(count + 2, dtype=bool); reachable[source] = True
    stack = [source]
    while stack:
        node = stack.pop()
        start, end = residual.indptr[node], residual.indptr[node + 1]
        for neighbor, amount in zip(residual.indices[start:end], residual.data[start:end]):
            if amount > 0 and not reachable[neighbor]:
                reachable[neighbor] = True; stack.append(int(neighbor))
    labels = torch.from_numpy((~reachable[:count]).astype(np.int64))
    return labels


@torch.no_grad()
def solve_multilabel_potts(unary, pairs, affinity, *, pairwise_weight=1.0, initial_labels=None, max_iterations=8):
    """Alpha-expansion for a metric Potts model; binary remains exact."""
    cost = torch.as_tensor(unary).detach().cpu().double()
    edge = torch.as_tensor(pairs).detach().cpu().long()
    weight = torch.as_tensor(affinity).detach().cpu().double().clamp_min(0) * float(pairwise_weight)
    if cost.ndim != 2 or cost.shape[1] < 2:
        raise ValueError("unary must be [N,L] with L>=2")
    if cost.shape[1] == 2:
        return solve_binary_potts_min_cut(cost, edge, weight, pairwise_weight=1.0)
    labels = cost.argmin(-1) if initial_labels is None else torch.as_tensor(initial_labels).detach().cpu().long().clone()
    old_energy = float(compute_segmentation_energy(cost, edge, weight, labels, pairwise_weight=1.0)["total"])
    for _ in range(int(max_iterations)):
        changed = False
        for alpha in range(cost.shape[1]):
            keep = labels
            first, second = edge[:, 0], edge[:, 1]
            e00 = weight * (keep[first] != keep[second]).double()
            e01 = weight * (keep[first] != alpha).double()
            e10 = weight * (alpha != keep[second]).double()
            e11 = torch.zeros_like(weight)
            coupling = 0.5 * (e10 + e01 - e00 - e11).clamp_min(0)
            switch_add = torch.zeros(cost.shape[0], dtype=torch.float64)
            switch_add.index_add_(0, first, e10 - e00 - coupling)
            switch_add.index_add_(0, second, e01 - e00 - coupling)
            binary = torch.stack((cost[torch.arange(cost.shape[0]), keep], cost[:, alpha] + switch_add), -1)
            proposal_switch = solve_binary_potts_min_cut(binary, edge, coupling, pairwise_weight=1.0).bool()
            proposal = torch.where(proposal_switch, torch.full_like(labels, alpha), labels)
            proposal_energy = float(compute_segmentation_energy(cost, edge, weight, proposal, pairwise_weight=1.0)["total"])
            if proposal_energy + 1.0e-8 < old_energy:
                changed = bool((proposal != labels).any()); labels, old_energy = proposal, proposal_energy
        if not changed:
            break
    return labels


@torch.no_grad()
def labels_to_soft_responsibilities(labels, unary, *, temperature=0.25, graph_strength=0.75):
    cost = torch.as_tensor(unary).detach().cpu().float()
    label = torch.as_tensor(labels).detach().cpu().long()
    soft = torch.softmax(-cost / max(float(temperature), 1.0e-6), dim=-1)
    hard = torch.nn.functional.one_hot(label, cost.shape[1]).float()
    result = (1.0 - float(graph_strength)) * soft + float(graph_strength) * hard
    return result / result.sum(-1, keepdim=True).clamp_min(1.0e-8)


@torch.no_grad()
def compute_segmentation_energy(unary, pairs, affinity, labels, *, pairwise_weight=1.0):
    cost = torch.as_tensor(unary).detach().cpu().double()
    edge = torch.as_tensor(pairs).detach().cpu().long()
    weight = torch.as_tensor(affinity).detach().cpu().double()
    label = torch.as_tensor(labels).detach().cpu().long()
    unary_energy = cost[torch.arange(cost.shape[0]), label].sum()
    pairwise = float(pairwise_weight) * weight[label[edge[:, 0]] != label[edge[:, 1]]].sum()
    return {"unary": float(unary_energy), "pairwise": float(pairwise), "total": float(unary_energy + pairwise)}
