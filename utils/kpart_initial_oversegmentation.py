"""Static-failure bootstrap and conservative spatial/material supernodes."""

from __future__ import annotations

import heapq
import math

import torch


def _robust_unit(value, valid=None):
    x = torch.as_tensor(value).detach().cpu().float().reshape(-1)
    mask = torch.isfinite(x)
    if valid is not None:
        mask &= torch.as_tensor(valid).detach().cpu().bool().reshape(-1)
    if not bool(mask.any()):
        return torch.zeros_like(x)
    finite = x[mask]
    low, high = torch.quantile(finite, torch.tensor([0.1, 0.9]))
    scaled = ((x - low) / (high - low).clamp_min(1.0e-6)).clamp(0.0, 1.0)
    return torch.where(mask, scaled, torch.zeros_like(scaled))


@torch.no_grad()
def compute_static_render_failure(modalities, *, weights=None, validity=None):
    """Robustly combine available RGB/depth/flow/DINO static failures."""
    if torch.is_tensor(modalities):
        modalities = {"flow": modalities}
    if not modalities:
        raise ValueError("at least one static-render failure modality is required")
    weights = {} if weights is None else dict(weights)
    validity = {} if validity is None else dict(validity)
    combined = None
    denominator = None
    normalized = {}
    for name, raw in modalities.items():
        value = _robust_unit(raw, validity.get(name))
        normalized[name] = value
        valid = torch.isfinite(torch.as_tensor(raw).detach().cpu().float().reshape(-1))
        if name in validity:
            valid &= torch.as_tensor(validity[name]).detach().cpu().bool().reshape(-1)
        weight = max(0.0, float(weights.get(name, 1.0)))
        term = valid.float() * weight
        combined = value * term if combined is None else combined + value * term
        denominator = term if denominator is None else denominator + term
    failure = combined / denominator.clamp_min(1.0e-8)
    return {
        "failure": failure,
        "dynamic_likelihood": failure,
        "normalized_modalities": normalized,
        "available_modalities": [name for name in modalities],
    }


@torch.no_grad()
def rasterize_failure_to_gaussians(
    pixel_failure=None,
    pixel_confidence=None,
    *,
    camera=None,
    gaussians=None,
    pipe=None,
    query_positions=None,
    query_rotations=None,
    rasterized=None,
):
    """Reverse-rasterize image evidence through the renderer's T*alpha ownership."""
    if rasterized is not None:
        feature_sum = torch.as_tensor(rasterized["feature_sum"]).detach().cpu().float()
        weight = torch.as_tensor(rasterized["weight_sum"]).detach().cpu().float()
    else:
        if any(item is None for item in (pixel_failure, pixel_confidence, camera, gaussians, pipe)):
            raise ValueError("pixel evidence and rasterization context are required")
        from gaussian_renderer import accumulate_pixel_features_to_gaussians

        result = accumulate_pixel_features_to_gaussians(
            camera,
            gaussians,
            pipe,
            torch.as_tensor(pixel_failure)[None] if torch.as_tensor(pixel_failure).ndim == 2 else pixel_failure,
            torch.as_tensor(pixel_confidence)[None] if torch.as_tensor(pixel_confidence).ndim == 2 else pixel_confidence,
            query_positions=query_positions,
            query_rotations=query_rotations,
        )
        feature_sum = result["feature_sum"].detach().cpu().float()
        weight = result["weight_sum"].detach().cpu().float()
    if feature_sum.ndim == 2:
        feature_sum = feature_sum[:, 0]
    return {"failure_sum": feature_sum, "weight_sum": weight, "failure": feature_sum / weight.clamp_min(1.0e-8)}


def _otsu_active_mask(failure, minimum):
    value = torch.as_tensor(failure).detach().cpu().float()
    if value.numel() == 0:
        return torch.zeros(0, dtype=torch.bool)
    bins = 64
    histogram = torch.histc(value.clamp(0, 1), bins=bins, min=0.0, max=1.0)
    probability = histogram / histogram.sum().clamp_min(1.0)
    centers = (torch.arange(bins).float() + 0.5) / bins
    cumulative = torch.cumsum(probability, 0)
    mean = torch.cumsum(probability * centers, 0)
    total = mean[-1]
    between = (total * cumulative - mean).square() / (cumulative * (1.0 - cumulative)).clamp_min(1.0e-8)
    threshold = centers[int(torch.argmax(between[:-1]))]
    active = value >= threshold
    if int(active.sum()) < int(minimum):
        keep = min(value.numel(), int(minimum))
        active = torch.zeros_like(active)
        active[torch.topk(value, keep).indices] = True
    return active


@torch.no_grad()
def build_conservative_dino_spatial_supernodes(
    points,
    dino_features,
    dynamic_likelihood,
    pairs,
    *,
    num_moving_parts,
    initial_oversegment=None,
    min_region_size=32,
    dino_affinity_weight=1.0,
    active_mask_override=None,
):
    """Oversegment the dynamic bootstrap on a canonical graph.

    DINO features are optional.  When omitted, the edge cost is purely spatial
    plus dynamic-likelihood contrast; this is the render-denoise static/dynamic
    discovery path.
    """
    xyz = torch.as_tensor(points).detach().cpu().float()
    feature = None
    if dino_features is not None:
        candidate = torch.as_tensor(dino_features).detach().cpu().float()
        if candidate.ndim == 2 and candidate.shape[0] == xyz.shape[0] and candidate.numel() > 0:
            feature = torch.nn.functional.normalize(candidate, dim=-1)
    dynamic = _robust_unit(dynamic_likelihood)
    edge = torch.as_tensor(pairs).detach().cpu().long()
    n, k = xyz.shape[0], int(num_moving_parts)
    if n == 0 or k < 1:
        raise ValueError("supernode construction requires points and K>=1")
    if initial_oversegment is None or int(initial_oversegment) <= k:
        data_driven = k + max(2, int(math.ceil(math.log2(1.0 + n / max(64, min_region_size)))))
        region_count = min(max(k + 1, data_driven), max(k + 1, n // max(1, int(min_region_size))))
    else:
        region_count = int(initial_oversegment)
    if active_mask_override is None:
        active = _otsu_active_mask(dynamic, max(region_count * int(min_region_size), region_count))
    else:
        active = torch.as_tensor(active_mask_override).detach().cpu().bool().reshape(-1)
        if active.numel() != n:
            raise ValueError("active_mask_override must match point count")
    active_ids = torch.nonzero(active, as_tuple=False).flatten()
    if active_ids.numel() < region_count and active_mask_override is not None:
        region_count = max(1, min(region_count, int(active_ids.numel())))
    if active_ids.numel() < region_count:
        raise RuntimeError("static-failure bootstrap has too few moving candidates for K' supernodes")

    span = (xyz.amax(0) - xyz.amin(0)).norm().clamp_min(1.0e-8)
    seeds = [int(active_ids[torch.argmax(dynamic[active_ids])])]
    minimum_distance = torch.linalg.norm(xyz - xyz[seeds[0]], dim=-1) / span
    for _ in range(1, region_count):
        score = minimum_distance * (0.25 + dynamic)
        score[~active] = -1
        score[torch.as_tensor(seeds)] = -1
        seed = int(torch.argmax(score))
        seeds.append(seed)
        minimum_distance = torch.minimum(minimum_distance, torch.linalg.norm(xyz - xyz[seed], dim=-1) / span)

    spatial = torch.linalg.norm(xyz[edge[:, 0]] - xyz[edge[:, 1]], dim=-1) if edge.numel() else torch.empty(0)
    spatial_scale = spatial[spatial > 0].median().clamp_min(1.0e-8) if bool((spatial > 0).any()) else torch.tensor(1.0)
    if feature is not None and edge.numel():
        dino_distance = 1.0 - (feature[edge[:, 0]] * feature[edge[:, 1]]).sum(-1)
        material_weight = float(dino_affinity_weight)
    else:
        dino_distance = torch.zeros(edge.shape[0], dtype=spatial.dtype)
        material_weight = 0.0
    edge_cost = spatial / spatial_scale * (1.0 + material_weight * dino_distance.clamp_min(0.0)) * (1.0 + (dynamic[edge[:, 0]] - dynamic[edge[:, 1]]).abs())
    adjacency = [[] for _ in range(n)]
    for index, (first, second) in enumerate(edge.tolist()):
        if not (active[first] and active[second]):
            continue
        cost = max(float(edge_cost[index]), 1.0e-6)
        adjacency[first].append((second, cost))
        adjacency[second].append((first, cost))
    labels = torch.full((n,), -1, dtype=torch.long)
    distance = torch.full((n,), float("inf"))
    queue = []
    for label, seed in enumerate(seeds):
        labels[seed], distance[seed] = label, 0.0
        heapq.heappush(queue, (0.0, seed, label))
    while queue:
        current, node, label = heapq.heappop(queue)
        if current > float(distance[node]) + 1.0e-7:
            continue
        for neighbor, cost in adjacency[node]:
            proposal = current + cost
            if proposal < float(distance[neighbor]):
                distance[neighbor], labels[neighbor] = proposal, label
                heapq.heappush(queue, (proposal, neighbor, label))
    unreachable = active & (labels < 0)
    if bool(unreachable.any()):
        nearest = torch.cdist(xyz[unreachable], xyz[torch.as_tensor(seeds)]).argmin(-1)
        labels[unreachable] = nearest
    sizes = torch.bincount(labels[active], minlength=region_count)
    if bool((sizes == 0).any()):
        raise RuntimeError("conservative supernode construction produced an empty provisional region")
    return {
        "labels": labels,
        "active_mask": active,
        "seeds": torch.as_tensor(seeds),
        "num_supernodes": region_count,
        "sizes": sizes,
        "dino_distance": dino_distance,
        "edge_cost": edge_cost,
    }


@torch.no_grad()
def initialize_provisional_regions(points, dino_features, static_failure, pairs, **kwargs):
    likelihood = static_failure["dynamic_likelihood"] if isinstance(static_failure, dict) else static_failure
    return build_conservative_dino_spatial_supernodes(
        points, dino_features, likelihood, pairs, **kwargs
    )
