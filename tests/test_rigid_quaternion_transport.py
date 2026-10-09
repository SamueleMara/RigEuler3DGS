import torch

from utils.rigid_transform_utils import (
    axis_angle_to_quaternion,
    direct_joint_transform,
    integrate_joint_transforms,
    quaternion_multiply,
    quaternion_transport_metrics,
    quaternion_to_matrix,
    apply_rotation_to_gaussian_quaternions,
    transport_gaussian_state,
)


def test_identity_and_world_left_multiplication_match_matrices():
    dtype = torch.float64
    canonical = axis_angle_to_quaternion(
        torch.tensor([1.0, 0.0, 0.0], dtype=dtype), torch.tensor(0.37, dtype=dtype)
    )
    delta = axis_angle_to_quaternion(
        torch.tensor([0.0, 0.0, 1.0], dtype=dtype), torch.tensor(torch.pi / 2, dtype=dtype)
    )
    updated = quaternion_multiply(delta, canonical)
    assert torch.allclose(
        quaternion_to_matrix(updated),
        quaternion_to_matrix(delta) @ quaternion_to_matrix(canonical),
        atol=1.0e-12,
        rtol=1.0e-12,
    )


def test_two_45_degree_increments_equal_one_90_degree_rotation():
    axis = torch.tensor([0.0, 1.0, 0.0], dtype=torch.float64)
    incremental = integrate_joint_transforms(
        "revolute", axis, torch.tensor([torch.pi / 4, torch.pi / 4], dtype=torch.float64)
    )
    expected = axis_angle_to_quaternion(axis, torch.tensor(torch.pi / 2, dtype=torch.float64))
    assert torch.allclose(
        quaternion_to_matrix(incremental["rotation_quaternions"][-1]),
        quaternion_to_matrix(expected),
        atol=1.0e-12,
        rtol=1.0e-12,
    )


def test_long_quaternion_integration_stays_normalized():
    trajectory = integrate_joint_transforms(
        "screw",
        torch.tensor([0.2, -0.4, 0.7], dtype=torch.float64),
        torch.full((1000,), 0.003, dtype=torch.float64),
        pivot=torch.tensor([0.1, 0.2, -0.3], dtype=torch.float64),
        pitch=0.15,
    )
    assert float((trajectory["rotation_quaternions"].norm(dim=-1) - 1).abs().max()) < 1.0e-12


def test_static_gaussian_is_exactly_unchanged():
    points = torch.tensor([[1.0, 0.0, 0.0], [0.0, 2.0, 0.0]], dtype=torch.float64)
    orientations = torch.tensor([[1.0, 0.0, 0.0, 0.0]]).repeat(2, 1).double()
    trajectory = integrate_joint_transforms(
        "revolute", torch.tensor([0.0, 0.0, 1.0], dtype=torch.float64),
        torch.tensor([0.5], dtype=torch.float64),
    )
    moved, rotated = transport_gaussian_state(
        points, orientations, trajectory["rotations"][-1],
        trajectory["translations"][-1], trajectory["rotation_quaternions"][-1],
        mask=torch.tensor([True, False]),
    )
    assert torch.equal(moved[1], points[1])
    assert torch.equal(rotated[1], orientations[1])


def test_quaternion_norm_audit_measures_raw_drift():
    orientations = torch.tensor(
        [[[2.0, 0.0, 0.0, 0.0]], [[2.0, 0.0, 0.0, 0.0]]],
        dtype=torch.float64,
    )
    positions = torch.zeros((2, 1, 3), dtype=torch.float64)
    metrics = quaternion_transport_metrics(
        positions[0], orientations[0], positions, orientations, torch.tensor([0])
    )
    assert metrics["max_quaternion_norm_error"] == 1.0


def test_incremental_and_direct_replay_match_centers_and_orientations():
    dtype = torch.float64
    joint = {
        "joint_type": "screw",
        "axis": torch.tensor([0.2, -0.3, 0.7], dtype=dtype),
        "pivot": torch.tensor([0.4, -0.1, 0.2], dtype=dtype),
        "pitch": 0.17,
    }
    delta_q = torch.tensor([0.12, -0.03, 0.21, 0.08], dtype=dtype)
    incremental = integrate_joint_transforms(
        joint["joint_type"], joint["axis"], delta_q,
        pivot=joint["pivot"], pitch=joint["pitch"],
    )
    direct_rotation, direct_translation, direct_quaternion = direct_joint_transform(
        joint, delta_q.sum()
    )
    points = torch.tensor([[0.1, 0.5, -0.2], [-0.3, 0.2, 0.8]], dtype=dtype)
    orientations = torch.nn.functional.normalize(
        torch.tensor([[1.0, 0.2, -0.1, 0.3], [0.8, -0.1, 0.4, 0.2]], dtype=dtype),
        dim=-1,
    )
    incremental_state = transport_gaussian_state(
        points, orientations, incremental["rotations"][-1],
        incremental["translations"][-1], incremental["rotation_quaternions"][-1],
    )
    direct_state = transport_gaussian_state(
        points, orientations, direct_rotation, direct_translation, direct_quaternion,
    )
    assert torch.allclose(incremental["rotations"][-1], direct_rotation, atol=1.0e-12)
    assert torch.allclose(incremental["translations"][-1], direct_translation, atol=1.0e-12)
    assert torch.allclose(incremental_state[0], direct_state[0], atol=1.0e-12)
    assert torch.allclose(
        quaternion_to_matrix(incremental_state[1]),
        quaternion_to_matrix(direct_state[1]),
        atol=1.0e-12,
    )


def test_float32_roundoff_is_not_misreported_as_spatial_rotation_variance():
    generator = torch.Generator().manual_seed(31)
    canonical = torch.nn.functional.normalize(
        torch.randn((64, 4), generator=generator), dim=-1
    )
    delta = axis_angle_to_quaternion(
        torch.tensor([0.1, -0.2, 0.8]), torch.tensor(0.013)
    )
    transported = apply_rotation_to_gaussian_quaternions(canonical, delta)
    positions = torch.zeros((2, canonical.shape[0], 3))
    positions[1, :, 0] = 0.01
    orientations = torch.stack((canonical, transported))
    part_quaternions = torch.stack((
        torch.tensor([[1.0, 0.0, 0.0, 0.0]]),
        delta[None],
    ))
    labels = torch.ones(canonical.shape[0], dtype=torch.long)
    metrics = quaternion_transport_metrics(
        positions[0], canonical, positions, orientations, labels,
        part_rotation_quaternions=part_quaternions,
    )
    assert metrics["spatial_rigid_rotation_variance_max"] <= metrics["spatial_rotation_tolerance"]
    assert metrics["part_quaternion_agreement_error_max"] <= metrics["part_quaternion_agreement_tolerance"]

    corrupted = orientations.clone()
    corrupted[1, 0] = apply_rotation_to_gaussian_quaternions(
        corrupted[1, 0:1],
        axis_angle_to_quaternion(torch.tensor([1.0, 0.0, 0.0]), torch.tensor(0.01)),
    )[0]
    broken = quaternion_transport_metrics(
        positions[0], canonical, positions, corrupted, labels,
        part_rotation_quaternions=part_quaternions,
    )
    assert broken["spatial_rigid_rotation_variance_max"] > broken["spatial_rotation_tolerance"]
