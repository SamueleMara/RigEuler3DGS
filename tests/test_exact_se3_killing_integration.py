import math

import torch

from rigid_parts.pipeline import _graph_repulsive_loss, _integrate_local_killing_twist
from utils.rigid_transform_utils import quaternion_to_matrix, se3_exp_spatial


def test_spatial_se3_zero_angular_velocity_translation_limit():
    omega = torch.zeros(2, 3)
    b = torch.tensor([[1.0, -2.0, 0.5], [0.25, 0.5, -1.0]])
    rotation, translation, quaternion = se3_exp_spatial(omega, b, 0.25)
    eye = torch.eye(3).expand(2, 3, 3)
    assert torch.allclose(rotation, eye, atol=1.0e-6)
    assert torch.allclose(translation, b * 0.25, atol=1.0e-6)
    assert torch.allclose(quaternion, torch.tensor([[1.0, 0.0, 0.0, 0.0]]).expand(2, 4), atol=1.0e-6)


def test_spatial_se3_translation_uses_left_jacobian_not_euler_bdt():
    omega = torch.tensor([[0.0, 0.0, 1.0]])
    b = torch.tensor([[1.0, 0.0, 0.0]])
    _, translation, _ = se3_exp_spatial(omega, b, 1.0)
    expected = torch.tensor([[math.sin(1.0), 1.0 - math.cos(1.0), 0.0]])
    wrong_euler = torch.tensor([[1.0, 0.0, 0.0]])
    assert torch.allclose(translation, expected, atol=1.0e-6)
    assert not torch.allclose(translation, wrong_euler, atol=1.0e-3)


def test_local_killing_transport_moves_centers_and_orientations_exactly():
    xyz0 = torch.tensor([[1.0, 0.0, 0.0]])
    q0 = torch.tensor([[1.0, 0.0, 0.0, 0.0]])
    omega = torch.tensor([[[0.0, 0.0, math.pi]]])
    b = torch.zeros_like(omega)
    times = torch.tensor([0.0, 0.5])
    state = _integrate_local_killing_twist(xyz0, q0, omega, b, times)
    assert torch.allclose(state.tracks[1, 0], torch.tensor([0.0, 1.0, 0.0]), atol=1.0e-5)
    rotation = quaternion_to_matrix(state.rotations[1, 0])
    expected_rotation = torch.tensor(
        [[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]
    )
    assert torch.allclose(rotation, expected_rotation, atol=1.0e-5)


def test_graph_repulsive_penalizes_same_labels_across_boundary():
    same = torch.tensor([[0.95, 0.05], [0.90, 0.10]])
    split = torch.tensor([[0.95, 0.05], [0.05, 0.95]])
    edge_src = torch.tensor([0])
    edge_dst = torch.tensor([1])
    low_affinity = torch.tensor([0.0])
    assert _graph_repulsive_loss(same, edge_src, edge_dst, low_affinity) > _graph_repulsive_loss(
        split, edge_src, edge_dst, low_affinity
    )
