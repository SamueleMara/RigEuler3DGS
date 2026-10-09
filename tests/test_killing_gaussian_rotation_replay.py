import torch

from utils.rigid_killing_motion import RigidKillingMode, integrate_rigid_mode_exact
from utils.rigid_transform_utils import covariance_from_quaternion_scale


def test_killing_replay_rotates_centers_orientations_and_anisotropic_covariances():
    mode = RigidKillingMode(
        reference_point=torch.tensor([0.2, -0.1, 0.3], dtype=torch.float64),
        reference_velocity=torch.zeros(3, dtype=torch.float64),
        angular_mode=torch.tensor([0.0, 0.0, 1.0], dtype=torch.float64),
        mode_type="rotational", normalization_scale=1.0, valid=True, diagnostics={},
    )
    points = torch.tensor([[1.0, 0.0, 0.2], [0.5, 0.8, -0.1]], dtype=torch.float64)
    orientations = torch.nn.functional.normalize(
        torch.tensor([[1.0, 0.2, 0.1, 0.0], [0.9, -0.1, 0.3, 0.1]], dtype=torch.float64), dim=-1
    )
    scales = torch.tensor([[0.4, 0.2, 0.1], [0.3, 0.15, 0.05]], dtype=torch.float64)
    result = integrate_rigid_mode_exact(points, mode, 0.8, orientations=orientations)
    assert not torch.allclose(result["positions"], points)
    assert not torch.allclose(result["orientations"], orientations)
    canonical_covariance = covariance_from_quaternion_scale(orientations, scales)
    transported_covariance = covariance_from_quaternion_scale(result["orientations"], scales)
    expected_covariance = result["rotation_matrix"] @ canonical_covariance @ result["rotation_matrix"].T
    assert torch.allclose(transported_covariance, expected_covariance, atol=2.0e-12, rtol=2.0e-12)
    relative = transported_covariance @ torch.linalg.inv(canonical_covariance)
    assert torch.isfinite(relative).all()
