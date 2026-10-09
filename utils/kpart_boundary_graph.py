"""Canonical local DINO/motion graph for K-part segmentation."""

from __future__ import annotations

import torch

from utils.dynamic_loss_utils import build_lagrangian_knn_pair_graph


@torch.no_grad()
def build_canonical_gaussian_knn(points, *, neighbors=8, max_pairs=262144, query_chunk_size=512):
    graph = build_lagrangian_knn_pair_graph(
        torch.as_tensor(points).detach(),
        neighbors_per_point=int(neighbors),
        max_pairs=int(max_pairs),
        query_chunk_size=int(query_chunk_size),
    )
    return {"pairs": graph.pairs.detach().cpu(), "spatial_distance": graph.distance.detach().cpu()}


@torch.no_grad()
def compute_dino_cosine_distance(features, pairs):
    feature = torch.nn.functional.normalize(torch.as_tensor(features).detach().cpu().float(), dim=-1)
    edge = torch.as_tensor(pairs).detach().cpu().long()
    cosine = (feature[edge[:, 0]] * feature[edge[:, 1]]).sum(-1) if edge.numel() else feature.new_empty(0)
    return {"cosine": cosine, "distance": (1.0 - cosine).clamp(0.0, 2.0)}


@torch.no_grad()
def compute_local_dino_boundary_saliency(distance, pairs, *, num_nodes=None, mad_floor=1.0e-4):
    """Edge saliency relative to the union of its endpoint neighborhoods."""
    value = torch.as_tensor(distance).detach().cpu().float()
    edge = torch.as_tensor(pairs).detach().cpu().long()
    node_count = int(num_nodes) if num_nodes is not None else (int(edge.max()) + 1 if edge.numel() else 0)
    incident = [[] for _ in range(node_count)]
    for index, (first, second) in enumerate(edge.tolist()):
        incident[first].append(index)
        incident[second].append(index)
    saliency = torch.zeros_like(value)
    local_median = torch.zeros_like(value)
    local_mad = torch.zeros_like(value)
    for index, (first, second) in enumerate(edge.tolist()):
        neighborhood = list(set(incident[first] + incident[second]))
        local = value[neighborhood]
        median = local.median()
        mad = (local - median).abs().median().clamp_min(float(mad_floor))
        local_median[index], local_mad[index] = median, mad
        saliency[index] = (value[index] - median) / mad
    return {"saliency": saliency, "local_median": local_median, "local_mad": local_mad}


@torch.no_grad()
def compute_dino_affinity(boundary_saliency, *, gamma=1.0):
    boundary = torch.as_tensor(boundary_saliency).float()
    return torch.exp(-float(gamma) * boundary.clamp_min(0.0))


@torch.no_grad()
def combine_dino_motion_affinity(dino_affinity, motion_distance, *, motion_scale=None, eps=1.0e-8):
    dino = torch.as_tensor(dino_affinity).detach().cpu().float()
    motion = torch.as_tensor(motion_distance).detach().cpu().float()
    valid = torch.isfinite(motion) & (motion >= 0)
    if motion_scale is None:
        positive = motion[valid & (motion > 0)]
        scale = float(positive.median()) if positive.numel() else 1.0
    else:
        scale = float(motion_scale)
    scale = max(scale, float(eps))
    motion_affinity = torch.where(valid, torch.exp(-motion / scale), torch.ones_like(motion))
    return {"affinity": dino * motion_affinity, "motion_affinity": motion_affinity, "motion_scale": scale}


@torch.no_grad()
def extract_uncertain_components(
    pairs,
    affinity,
    uncertain_mask,
    *,
    min_affinity=0.25,
    max_components=None,
    priority=None,
):
    """Connected uncertain Gaussian components on sufficiently compatible edges."""
    edge = torch.as_tensor(pairs).detach().cpu().long()
    weight = torch.as_tensor(affinity).detach().cpu().float()
    uncertain = torch.as_tensor(uncertain_mask).detach().cpu().bool()
    adjacency = [[] for _ in range(uncertain.numel())]
    keep = (weight >= float(min_affinity)) & uncertain[edge[:, 0]] & uncertain[edge[:, 1]]
    for first, second in edge[keep].tolist():
        adjacency[first].append(second)
        adjacency[second].append(first)
    visited = torch.zeros_like(uncertain)
    components = []
    for seed in torch.nonzero(uncertain, as_tuple=False).flatten().tolist():
        if visited[seed]:
            continue
        stack, members = [seed], []
        visited[seed] = True
        while stack:
            node = stack.pop()
            members.append(node)
            for neighbor in adjacency[node]:
                if not visited[neighbor]:
                    visited[neighbor] = True
                    stack.append(neighbor)
        components.append(torch.tensor(members, dtype=torch.long))
    score = torch.as_tensor(priority).detach().cpu().float() if priority is not None else torch.ones_like(uncertain, dtype=torch.float32)
    components.sort(key=lambda item: (float(score[item].mean()), int(item.numel())), reverse=True)
    return components if max_components is None else components[: int(max_components)]
