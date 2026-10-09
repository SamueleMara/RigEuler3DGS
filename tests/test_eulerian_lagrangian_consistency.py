import torch

from utils.rigid_killing_motion import (
    aggregate_points_to_rigid_cells,
    normalize_rigid_mode,
    predict_rigid_cell_velocity,
    solve_rigid_killing_mode_all_intervals,
)


def test_shared_solver_agrees_for_affine_grouped_and_gaussian_supports():
    generator = torch.Generator().manual_seed(3)
    points = torch.randn((120, 3), generator=generator, dtype=torch.float64)
    groups = torch.arange(points.shape[0]) // 4
    count = int(groups.max()) + 1
    mode, _ = normalize_rigid_mode(
        points[0],
        torch.tensor((0.2, 0.1, -0.1), dtype=torch.float64),
        torch.tensor((0.1, -0.2, 0.8), dtype=torch.float64),
    )
    speeds = torch.tensor((0.2, 0.4, 0.7), dtype=torch.float64)
    velocity = predict_rigid_cell_velocity(mode, points, speeds)
    cell_points = aggregate_points_to_rigid_cells(points, groups, count)
    cell_velocity = torch.stack(
        [aggregate_points_to_rigid_cells(value, groups, count) for value in velocity]
    )
    point_fit, point_speed, _ = solve_rigid_killing_mode_all_intervals(
        points, velocity, min_valid_cells=8
    )
    cell_fit, cell_speed, _ = solve_rigid_killing_mode_all_intervals(
        cell_points, cell_velocity, min_valid_cells=8
    )
    assert torch.dot(point_fit.angular_mode, cell_fit.angular_mode) > 0.999999
    assert torch.allclose(point_speed, speeds, atol=1e-7)
    assert torch.allclose(cell_speed, speeds, atol=1e-7)

