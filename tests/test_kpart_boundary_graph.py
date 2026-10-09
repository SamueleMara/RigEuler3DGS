import torch

from utils.kpart_boundary_graph import (
    combine_dino_motion_affinity,
    compute_dino_affinity,
    compute_dino_cosine_distance,
    compute_local_dino_boundary_saliency,
)


def test_local_dino_boundary_suppresses_affinity_for_similar_motion_parts():
    pairs = torch.tensor([[0, 1], [1, 2], [2, 3], [3, 4]])
    features = torch.tensor([[1.0, 0.0], [1.0, 0.0], [1.0, 0.0], [0.0, 1.0], [0.0, 1.0]])
    distance = compute_dino_cosine_distance(features, pairs)["distance"]
    saliency = compute_local_dino_boundary_saliency(distance, pairs, num_nodes=5)["saliency"]
    affinity = compute_dino_affinity(saliency)
    assert affinity[2] < affinity[0]


def test_combined_affinity_uses_motion_when_dino_is_uniform():
    combined = combine_dino_motion_affinity(torch.ones(2), torch.tensor([0.01, 3.0]), motion_scale=0.2)
    assert combined["affinity"][0] > 0.9
    assert combined["affinity"][1] < 1e-4
