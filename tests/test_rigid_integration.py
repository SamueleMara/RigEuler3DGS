import torch

from utils.rigid_killing_motion import (
    integrate_rigid_mode_exact,
    normalize_rigid_mode,
    predict_rigid_cell_velocity,
    re_reference_rigid_mode,
)


def _mode():
    mode, _ = normalize_rigid_mode(
        torch.tensor([0.3, -0.5, 0.2], dtype=torch.float64),
        torch.tensor([0.2, 0.1, -0.05], dtype=torch.float64),
        torch.tensor([0.2, -0.4, 0.8], dtype=torch.float64),
    )
    return mode


def test_exact_transport_preserves_distances_and_round_trips():
    generator = torch.Generator().manual_seed(3)
    points = torch.randn((64, 3), generator=generator, dtype=torch.float64)
    orientation = torch.nn.functional.normalize(
        torch.randn((64, 4), generator=generator, dtype=torch.float64), dim=-1
    )
    before = torch.pdist(points)
    forward = integrate_rigid_mode_exact(
        points, _mode(), 1.7, orientations=orientation
    )
    after = torch.pdist(forward["positions"])
    backward = integrate_rigid_mode_exact(
        forward["positions"],
        _mode(),
        -1.7,
        orientations=forward["orientations"],
    )
    assert torch.allclose(after, before, atol=2e-12, rtol=2e-12)
    assert torch.allclose(backward["positions"], points, atol=2e-12, rtol=2e-12)
    assert torch.allclose(
        backward["orientations"], orientation, atol=2e-12, rtol=2e-12
    )


def test_reference_point_invariance_for_velocity_and_exact_transform():
    generator = torch.Generator().manual_seed(9)
    points = torch.randn((40, 3), generator=generator, dtype=torch.float64)
    first = _mode()
    second = re_reference_rigid_mode(
        first, torch.tensor([-3.0, 4.0, 2.0], dtype=torch.float64)
    )
    assert torch.allclose(
        predict_rigid_cell_velocity(first, points, 0.7),
        predict_rigid_cell_velocity(second, points, 0.7),
        atol=2e-12,
        rtol=2e-12,
    )
    first_result = integrate_rigid_mode_exact(points, first, 0.7)["positions"]
    second_result = integrate_rigid_mode_exact(points, second, 0.7)["positions"]
    assert torch.allclose(first_result, second_result, atol=2e-12, rtol=2e-12)


def test_effective_static_mask_is_exactly_stationary():
    points = torch.randn((20, 3), dtype=torch.float64)
    mask = torch.arange(20) % 2 == 0
    orientation = torch.zeros((20, 4), dtype=torch.float64)
    orientation[:, 0] = 3.0
    result = integrate_rigid_mode_exact(
        points,
        _mode(),
        0.3,
        orientations=orientation,
        effective_dynamic_mask=mask,
    )
    moved = result["positions"]
    assert torch.equal(moved[~mask], points[~mask])
    assert float(torch.linalg.norm(moved[mask] - points[mask], dim=-1).min()) > 0
    assert torch.allclose(
        result["orientations"].norm(dim=-1), torch.ones(20, dtype=torch.float64)
    )


def test_frame_zero_rigid_replay_normalizes_raw_gaussian_rotations():
    from utils.unified_dynamic_training import _rigid_replay_to_frame

    points = torch.randn((8, 3), dtype=torch.float32)
    raw_orientation = torch.randn((8, 4), dtype=torch.float32) * 4.0
    xyz, orientation, velocity = _rigid_replay_to_frame(
        points,
        raw_orientation,
        None,
        torch.zeros(1),
        [0.0, 1.0],
        0,
        torch.ones(8, dtype=torch.bool),
    )
    assert torch.equal(xyz, points)
    assert torch.equal(velocity, torch.zeros_like(points))
    assert torch.allclose(
        orientation.norm(dim=-1), torch.ones(8), atol=1.0e-6
    )


def test_gaussian_model_replays_the_saved_rigid_state_exactly():
    from scene.gaussian_model import GaussianModel

    points = torch.randn((12, 3), dtype=torch.float32)
    orientations = torch.zeros((12, 4), dtype=torch.float32)
    orientations[:, 0] = 1.0
    dynamic = torch.arange(12) % 3 != 0
    mode = _mode().to(dtype=torch.float32)
    expected = integrate_rigid_mode_exact(
        points,
        mode,
        0.4,
        orientations=orientations,
        effective_dynamic_mask=dynamic,
    )
    replay_by_representation = {}
    for representation in ("rigid_killing_eulerian", "rigid_killing_lagrangian"):
        model = GaussianModel(0)
        model._xyz = points
        model._rotation = orientations
        model.dynamic_enabled = True
        model.motion_representation = representation
        model.rigid_reference_point = mode.reference_point
        model.rigid_reference_velocity = mode.reference_velocity
        model.rigid_angular_mode = mode.angular_mode
        model.rigid_interval_speeds = torch.tensor([0.4, 0.4])
        model.rigid_frame_times = torch.tensor([0.0, 0.5, 1.0])
        model.rigid_effective_dynamic_mask = dynamic
        model.rigid_solver_diagnostics = {"mode_type": mode.mode_type}
        replay_xyz, replay_q, replay_velocity, _ = model.dynamic_state_at(1.0)
        assert torch.allclose(replay_xyz, expected["positions"], atol=2e-6)
        assert torch.allclose(replay_q, expected["orientations"], atol=2e-6)
        assert torch.equal(replay_xyz[~dynamic], points[~dynamic])
        assert torch.equal(
            replay_velocity[~dynamic], torch.zeros_like(replay_velocity[~dynamic])
        )
        replay_by_representation[representation] = (replay_xyz, replay_q, replay_velocity)
    for lhs, rhs in zip(*replay_by_representation.values()):
        assert torch.equal(lhs, rhs)
