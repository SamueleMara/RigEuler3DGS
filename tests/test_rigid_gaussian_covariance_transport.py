import torch

from utils.rigid_transform_utils import (
    apply_rotation_to_gaussian_quaternions,
    axis_angle_to_quaternion,
    covariance_from_quaternion_scale,
    quaternion_to_matrix,
)


def test_quaternion_transport_matches_rigid_covariance_transport():
    generator = torch.Generator().manual_seed(17)
    orientation = torch.nn.functional.normalize(
        torch.randn((32, 4), generator=generator, dtype=torch.float64), dim=-1
    )
    scale = torch.rand((32, 3), generator=generator, dtype=torch.float64) + 0.1
    delta = axis_angle_to_quaternion(
        torch.tensor([0.3, -0.4, 0.8], dtype=torch.float64),
        torch.tensor(0.73, dtype=torch.float64),
    )
    updated = apply_rotation_to_gaussian_quaternions(orientation, delta)
    actual = covariance_from_quaternion_scale(updated, scale)
    rigid = quaternion_to_matrix(delta)
    canonical = covariance_from_quaternion_scale(orientation, scale)
    expected = rigid @ canonical @ rigid.T
    assert torch.allclose(actual, expected, atol=2.0e-12, rtol=2.0e-12)

