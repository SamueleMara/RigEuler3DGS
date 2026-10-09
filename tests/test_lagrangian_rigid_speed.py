import torch

from utils.lagrangian_rigid_motion import compute_gaussian_speed_observations
from utils.rigid_killing_motion import normalize_rigid_mode, predict_rigid_cell_velocity


def test_per_gaussian_speed_observations_recover_shared_speed():
    generator = torch.Generator().manual_seed(41)
    points = torch.randn((24, 3), generator=generator, dtype=torch.float64)
    mode, _ = normalize_rigid_mode(
        points[0],
        torch.tensor((0.2, -0.1, 0.05), dtype=torch.float64),
        torch.tensor((0.1, 0.3, 0.9), dtype=torch.float64),
    )
    speeds = torch.tensor((0.2, -0.4, 0.7), dtype=torch.float64)
    positions = points[None].expand(speeds.numel(), -1, -1).clone()
    velocity = predict_rigid_cell_velocity(mode, positions, speeds)
    normal = torch.eye(3, dtype=torch.float64).reshape(1, 1, 3, 3).expand(
        speeds.numel(), points.shape[0], -1, -1
    )
    result = compute_gaussian_speed_observations(
        positions,
        velocity,
        normal,
        torch.ones(positions.shape[:2], dtype=torch.bool),
        mode,
        speeds,
    )
    assert torch.allclose(
        result["observed_gaussian_speed"], speeds[:, None].expand_as(result["observed_gaussian_speed"]),
        atol=1e-10,
    )
    assert float(result["perpendicular_velocity_residual"].max()) < 1e-10
    assert float(result["speed_rmse"].max()) < 1e-10

