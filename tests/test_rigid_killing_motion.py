import torch

from utils.rigid_killing_motion import (
    fit_local_angular_velocity_from_edges,
    normalize_rigid_mode,
    predict_rigid_cell_velocity,
    solve_rigid_field_single_interval,
    solve_rigid_killing_mode_all_intervals,
)


def _points(count=80):
    generator = torch.Generator().manual_seed(17)
    return torch.randn((count, 3), generator=generator, dtype=torch.float64)


def test_translation_has_zero_angular_velocity():
    points = _points()
    translation = torch.tensor([0.4, -0.2, 0.7], dtype=torch.float64)
    velocity = translation.expand_as(points)
    angular, diagnostics = fit_local_angular_velocity_from_edges(points, velocity)
    fit = solve_rigid_field_single_interval(points, velocity)
    generator = fit.diagnostics["generator"]
    assert diagnostics["valid"]
    assert torch.allclose(angular, torch.zeros(3, dtype=torch.float64), atol=1e-11)
    assert torch.allclose(generator[:3], translation, atol=1e-11)
    assert torch.allclose(generator[3:], angular, atol=1e-11)
    assert float(fit.residual_per_cell.max()) < 1e-11


def test_rotation_far_reference_and_screw_are_origin_independent():
    points = _points()
    physical_center = torch.tensor([1e6, -2e6, 5e5], dtype=torch.float64)
    angular_gt = torch.tensor([0.2, -0.3, 0.7], dtype=torch.float64)
    pitch_velocity = 0.13 * angular_gt
    velocity = torch.cross(
        angular_gt.expand_as(points), points - physical_center, dim=-1
    ) + pitch_velocity
    reference = points[11]
    fit = solve_rigid_field_single_interval(
        points, velocity, reference_point=reference, huber_delta=1.0
    )
    generator = fit.diagnostics["generator"]
    expected_reference_velocity = torch.cross(
        angular_gt, reference - physical_center, dim=-1
    ) + pitch_velocity
    assert torch.allclose(generator[3:], angular_gt, rtol=1e-9, atol=1e-9)
    assert torch.allclose(generator[:3], expected_reference_velocity, rtol=1e-10, atol=1e-8)
    assert float(fit.residual_per_cell.max()) < 1e-8


def test_one_dimensional_mode_recovers_signed_interval_speeds_with_outliers():
    points = _points(120)
    reference = points[7]
    linear = torch.tensor([0.2, -0.1, 0.05], dtype=torch.float64)
    angular = torch.tensor([0.0, 0.0, 2.0], dtype=torch.float64)
    mode_gt, _ = normalize_rigid_mode(reference, linear, angular)
    speeds_gt = torch.tensor([0.1, 0.2, 0.4, 0.8, -0.3], dtype=torch.float64)
    observed = predict_rigid_cell_velocity(mode_gt, points, speeds_gt)
    observed = observed.clone()
    observed[:, :12] += 0.3
    weights = torch.ones(observed.shape[:2], dtype=torch.float64)
    mode, speeds, diagnostics = solve_rigid_killing_mode_all_intervals(
        points,
        observed,
        weights=weights,
        reference_point=reference,
        max_iters=30,
        huber_delta=0.03,
        min_valid_cells=20,
    )
    prediction = predict_rigid_cell_velocity(mode, points, speeds)
    inlier_error = torch.linalg.norm(prediction[:, 12:] - observed[:, 12:], dim=-1)
    assert mode.valid
    assert diagnostics["history"]
    assert torch.dot(mode.angular_mode, mode_gt.angular_mode) > 0.999
    assert torch.allclose(speeds, speeds_gt, atol=0.015, rtol=0.03)
    assert float(inlier_error.median()) < 3e-3
