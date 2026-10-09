"""Canonical 3-D SAM3 affinity graph for K-part responsibility smoothing."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from utils.kpart_boundary_graph import build_canonical_gaussian_knn


@torch.no_grad()
def build_canonical_sam3_graph(points, sam3_features, *, neighbors=8, max_pairs=262144, query_chunk_size=512):
    graph = build_canonical_gaussian_knn(
        points,
        neighbors=int(neighbors),
        max_pairs=int(max_pairs),
        query_chunk_size=int(query_chunk_size),
    )
    feature = F.normalize(torch.as_tensor(sam3_features).detach().cpu().float(), dim=-1)
    pair = graph["pairs"]
    cosine = (
        (feature[pair[:, 0]] * feature[pair[:, 1]]).sum(-1)
        if pair.numel()
        else feature.new_empty(0)
    )
    return {
        **graph,
        "sam3_cosine": cosine,
        "sam3_distance": (1.0 - cosine).clamp(0.0, 2.0),
    }


@torch.no_grad()
def compute_3d_sam3_same_material_affinity(points, pairs, sam3_distance, *, sigma=0.15, spatial_sigma=None):
    xyz = torch.as_tensor(points).detach().cpu().float()
    edge = torch.as_tensor(pairs).detach().cpu().long()
    distance = torch.as_tensor(sam3_distance).detach().cpu().float()
    spatial = (
        torch.linalg.norm(xyz[edge[:, 0]] - xyz[edge[:, 1]], dim=-1)
        if edge.numel()
        else distance.new_empty(0)
    )
    if spatial_sigma is None:
        positive = spatial[spatial > 0]
        spatial_sigma = float(positive.median()) if positive.numel() else 1.0
    sigma = max(float(sigma), 1.0e-8)
    spatial_sigma = max(float(spatial_sigma), 1.0e-8)
    affinity = torch.exp(-distance / sigma) * torch.exp(-0.5 * (spatial / spatial_sigma).square())
    return {
        "affinity": affinity,
        "spatial_distance": spatial,
        "spatial_sigma": float(spatial_sigma),
        "sam3_sigma": float(sigma),
    }
