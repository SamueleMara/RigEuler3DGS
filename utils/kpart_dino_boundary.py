"""Canonical 3D DINO affinity and rendered one-sided seam regularization."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from utils.kpart_boundary_graph import build_canonical_gaussian_knn


@torch.no_grad()
def build_canonical_dino_graph(points, dino_features, *, neighbors=8, max_pairs=262144, query_chunk_size=512):
    graph = build_canonical_gaussian_knn(
        points,
        neighbors=int(neighbors),
        max_pairs=int(max_pairs),
        query_chunk_size=int(query_chunk_size),
    )
    feature = F.normalize(torch.as_tensor(dino_features).detach().cpu().float(), dim=-1)
    pair = graph["pairs"]
    cosine = (feature[pair[:, 0]] * feature[pair[:, 1]]).sum(-1) if pair.numel() else feature.new_empty(0)
    return {**graph, "dino_cosine": cosine, "dino_distance": (1.0 - cosine).clamp(0.0, 2.0)}


@torch.no_grad()
def compute_3d_same_material_affinity(points, pairs, dino_distance, *, dino_beta=4.0, spatial_sigma=None):
    xyz = torch.as_tensor(points).detach().cpu().float()
    edge = torch.as_tensor(pairs).detach().cpu().long()
    distance = torch.as_tensor(dino_distance).detach().cpu().float()
    spatial = torch.linalg.norm(xyz[edge[:, 0]] - xyz[edge[:, 1]], dim=-1) if edge.numel() else distance.new_empty(0)
    if spatial_sigma is None:
        positive = spatial[spatial > 0]
        sigma = float(positive.median()) if positive.numel() else 1.0
    else:
        sigma = float(spatial_sigma)
    sigma = max(sigma, 1.0e-8)
    affinity = torch.exp(-float(dino_beta) * distance) * torch.exp(-0.5 * (spatial / sigma).square())
    return {"affinity": affinity, "spatial_distance": spatial, "spatial_sigma": sigma}


def render_soft_part_probability_maps(camera, gaussians, pipe, responsibilities, *, positions, orientations):
    """Render soft Gaussian responsibilities at field-integrated geometry."""
    from gaussian_renderer import render_feature_field

    xyz = torch.as_tensor(positions, device=gaussians.get_xyz.device, dtype=gaussians.get_xyz.dtype)
    rotation = F.normalize(torch.as_tensor(orientations, device=xyz.device, dtype=gaussians.get_rotation.dtype), dim=-1)
    probability = torch.as_tensor(responsibilities, device=xyz.device, dtype=torch.float32)
    gaussians.set_free_working_state(
        xyz,
        float(getattr(camera, "timestamp", 0.0)),
        velocity_override=torch.zeros_like(xyz),
        rotation_override=rotation,
    )
    try:
        rendered = render_feature_field(
            camera,
            gaussians,
            pipe,
            probability,
            bg_feature=torch.zeros(probability.shape[1], device=probability.device),
            detach_geometry=True,
        )
        return {"probability_maps": rendered["render"], "depth": rendered["depth"], "radii": rendered["radii"]}
    finally:
        gaussians.clear_time()


def _gradient_magnitude(image):
    value = torch.as_tensor(image)
    dx = value[..., :, 1:] - value[..., :, :-1]
    dy = value[..., 1:, :] - value[..., :-1, :]
    dx = F.pad(dx, (0, 1, 0, 0))
    dy = F.pad(dy, (0, 0, 0, 1))
    return torch.sqrt(dx.square() + dy.square() + 1.0e-12)


def compute_rendered_part_boundary(probability_maps):
    probability = torch.as_tensor(probability_maps)
    gradient = _gradient_magnitude(probability)
    return gradient.amax(dim=-3) if gradient.ndim >= 3 else gradient


@torch.no_grad()
def compute_image_dino_boundary(dino_map):
    feature = F.normalize(torch.as_tensor(dino_map).detach().float(), dim=-3)
    boundary = _gradient_magnitude(feature).square().sum(dim=-3).sqrt()
    finite = boundary[torch.isfinite(boundary)]
    if finite.numel():
        low, high = torch.quantile(finite, torch.tensor([0.1, 0.9], device=finite.device))
        boundary = ((boundary - low) / (high - low).clamp_min(1.0e-6)).clamp(0.0, 1.0)
    return boundary


def compute_one_sided_dino_seam_penalty(part_boundary, dino_boundary, *, beta=3.0, valid_mask=None):
    part = torch.as_tensor(part_boundary)
    dino = torch.as_tensor(dino_boundary, device=part.device, dtype=part.dtype).detach()
    valid = torch.isfinite(part) & torch.isfinite(dino)
    if valid_mask is not None:
        valid &= torch.as_tensor(valid_mask, device=part.device).bool()
    if not bool(valid.any()):
        return part.sum() * 0.0
    return (part[valid] * torch.exp(-float(beta) * dino[valid])).mean()
