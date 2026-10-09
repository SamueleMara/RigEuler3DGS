import torch

from utils.kpart_motion_signature import (
    compute_gaussian_part_speed_distance,
    compute_gaussian_speed_signature,
    compute_pairwise_motion_compatibility,
    compute_projected_rigid_scalar_speed,
)


def test_same_part_different_radii_share_normalized_signature():
    speed = torch.tensor([0.2, 0.6, -0.3], dtype=torch.float64)
    radius = torch.tensor([1.0, 4.0], dtype=torch.float64)
    phi = radius[:, None, None] * torch.tensor([1.0, 0.0], dtype=torch.float64)
    observed = phi * speed[None, :, None]
    result = compute_projected_rigid_scalar_speed(phi, observed)
    assert torch.allclose(result["speed"], speed.expand(2, -1), atol=1e-8)


def test_nonconstant_part_signatures_separate_without_constant_speed_prior():
    part_a = torch.tensor([0.1, 0.7, 0.2, 1.1], dtype=torch.float64)
    part_b = torch.tensor([-0.4, -0.1, 0.8, 0.3], dtype=torch.float64)
    gaussian = torch.stack((part_a, part_a, part_b, part_b))
    weight = torch.ones_like(gaussian)
    distance_a = compute_gaussian_part_speed_distance(gaussian, part_a, weights=weight)["distance"]
    distance_b = compute_gaussian_part_speed_distance(gaussian, part_b, weights=weight)["distance"]
    labels = torch.stack((distance_a, distance_b), -1).argmin(-1)
    assert labels.tolist() == [0, 0, 1, 1]


def test_speed_reversal_and_stop_are_preserved():
    trajectory = torch.tensor([0.2, 0.8, 0.0, -0.3, -0.9], dtype=torch.float64)
    samples = trajectory[None, None].repeat(3, 2, 1)
    signature = compute_gaussian_speed_signature(samples, torch.ones_like(samples))
    assert torch.allclose(signature["signature"], trajectory.expand(3, -1))
    assert float(compute_gaussian_part_speed_distance(signature["signature"], trajectory)["distance"].max()) < 1e-10


def test_same_dino_different_temporal_motion_creates_motion_boundary():
    signatures = torch.tensor([[0.0, 1.0, -1.0], [0.0, 1.0, -1.0], [1.0, -1.0, 0.5]])
    result = compute_pairwise_motion_compatibility(signatures, torch.tensor([[0, 1], [1, 2]]))
    assert result["distance"][0] < result["distance"][1]
