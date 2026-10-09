import torch
from types import SimpleNamespace

from utils.eulerian_contrastive_support import (
    build_raw_support_state,
    split_mixed_supports_with_dino,
)
from utils.joint_model_fitting import fit_revolute_joint, joint_to_rigid_mode
from utils.joint_trajectory_optimization import (
    cubic_bspline_basis,
    compute_joint_projected_flow,
    derive_qdot_qddot,
    render_joint_constrained_motion,
)
from utils.kpart_rigid_discovery import (
    classify_kpart_responsibilities,
    deterministic_mode_seed_weights,
    initialize_uniform_responsibilities,
    update_multiclass_responsibilities,
)
from utils.rigid_killing_motion import RigidKillingMode, integrate_rigid_mode_exact


def _revolute_mode():
    return RigidKillingMode(
        reference_point=torch.tensor([0.2, -0.1, 0.3], dtype=torch.float64),
        reference_velocity=torch.zeros(3, dtype=torch.float64),
        angular_mode=torch.tensor([0.0, 0.0, 1.0], dtype=torch.float64),
        mode_type="rotational",
        normalization_scale=1.0,
        valid=True,
        diagnostics={},
    )


def test_kpart_uniform_initialization_and_softmax():
    responsibility = initialize_uniform_responsibilities(7, 2)
    assert responsibility.shape == (7, 3)
    assert torch.allclose(responsibility, torch.full_like(responsibility, 1.0 / 3.0))
    energy = torch.tensor([[0.0, 2.0, 1.0], [2.0, 0.0, 1.0]])
    updated = update_multiclass_responsibilities(energy, temperature=0.2)["responsibilities"]
    assert int(updated[0].argmax()) == 0
    assert int(updated[1].argmax()) == 1
    assert torch.allclose(updated.sum(dim=-1), torch.ones(2), atol=1.0e-6)


def test_kpart_deterministic_seeds_are_soft_and_distinct():
    points = torch.stack((torch.linspace(-1, 1, 20), torch.zeros(20), torch.zeros(20)), dim=-1)
    seed = deterministic_mode_seed_weights(points, 3)
    assert seed.shape == (20, 3)
    assert bool((seed > 0).all())
    assert not torch.allclose(seed[:, 0], seed[:, 1])


def test_eulerian_split_requires_responsibility_conflict():
    raw = torch.tensor([0, 0, 0, 0, 1, 1])
    state = build_raw_support_state(raw, 2, min_gaussians=1)
    responsibility = torch.tensor([
        [0.95, 0.05], [0.90, 0.10], [0.05, 0.95], [0.10, 0.90],
        [0.55, 0.45], [0.52, 0.48],
    ])
    cycle_zero = split_mixed_supports_with_dino(
        state, responsibility, None, cycle=0, enabled=True,
        confidence_threshold=0.8, min_class_fraction=0.2,
    )
    assert cycle_zero["num_supports"] == 2
    split = split_mixed_supports_with_dino(
        state, responsibility, None, cycle=1, enabled=True,
        confidence_threshold=0.8, min_class_fraction=0.2,
    )
    assert split["num_supports"] == 3
    assert split["mixed_diagnostic"]["mixed_support_count"] == 1


def test_revolute_joint_conversion_preserves_exact_motion():
    mode = _revolute_mode()
    q = torch.linspace(0.0, 1.0, 6, dtype=torch.float64)
    joint = fit_revolute_joint(mode, q)
    converted = joint_to_rigid_mode(joint)
    points = torch.tensor([[1.0, 0.0, 0.3], [0.4, 0.7, -0.2]], dtype=torch.float64)
    expected = integrate_rigid_mode_exact(points, mode, q[-1])["positions"]
    actual = integrate_rigid_mode_exact(points, converted, joint["coarse_q"][-1])["positions"]
    assert torch.allclose(actual, expected, atol=1.0e-10, rtol=1.0e-10)


def test_nonuniform_timestamp_derivatives_and_spline_partition():
    times = torch.tensor([0.0, 0.1, 0.35, 0.8, 1.4], dtype=torch.float64)
    q = torch.stack((2.0 * times, -0.5 * times))
    velocity, acceleration = derive_qdot_qddot(q, times)
    assert torch.allclose(velocity[0], torch.full_like(times, 2.0), atol=1.0e-10)
    assert torch.allclose(velocity[1], torch.full_like(times, -0.5), atol=1.0e-10)
    assert torch.allclose(acceleration, torch.zeros_like(acceleration), atol=1.0e-9)
    basis, derivative, second = cubic_bspline_basis(times, 4)
    assert torch.allclose(basis.sum(dim=-1), torch.ones(times.numel(), dtype=torch.float64), atol=1.0e-10)
    assert torch.isfinite(derivative).all() and torch.isfinite(second).all()


def test_final_replay_keeps_static_and_ambiguous_gaussians_canonical():
    mode = _revolute_mode()
    joint = fit_revolute_joint(mode, torch.tensor([0.0, 0.5], dtype=torch.float64))
    points = torch.tensor([[1.0, 0.0, 0.3], [0.0, 1.0, 0.0], [-1.0, 0.0, 0.2]], dtype=torch.float64)
    rotations = torch.tensor([[1.0, 0.0, 0.0, 0.0]]).repeat(3, 1).double()
    labels = torch.tensor([1, 0, -1])
    tracks = render_joint_constrained_motion(
        points, rotations, labels, [joint],
        torch.tensor([[0.0, 0.5]], dtype=torch.float64),
        torch.tensor([[0.5, 0.5]], dtype=torch.float64),
        [0.0, 1.0],
    )
    assert torch.equal(tracks["positions"][:, 1], points[1].expand(2, -1))
    assert torch.equal(tracks["positions"][:, 2], points[2].expand(2, -1))
    assert not torch.allclose(tracks["positions"][1, 0], points[0])


def test_kpart_confidence_preserves_ambiguous_state():
    classification = classify_kpart_responsibilities(
        torch.tensor([[0.9, 0.05, 0.05], [0.4, 0.35, 0.25]]),
        confidence_threshold=0.8,
    )
    assert classification["labels"].tolist() == [0, -1]


def test_joint_projected_flow_is_differentiable_in_absolute_coordinates():
    mode = _revolute_mode()
    joint = fit_revolute_joint(mode, torch.tensor([0.0, 0.2], dtype=torch.float64))
    camera = SimpleNamespace(
        full_proj_transform=torch.eye(4, dtype=torch.float64),
        world_view_transform=torch.eye(4, dtype=torch.float64),
        image_width=64,
        image_height=64,
    )
    points = torch.tensor([[0.3, 0.1, 0.5], [-0.2, 0.25, 0.6]], dtype=torch.float64)
    q0 = torch.tensor(0.0, dtype=torch.float64, requires_grad=True)
    q1 = torch.tensor(0.2, dtype=torch.float64, requires_grad=True)
    result = compute_joint_projected_flow(points, joint, q0, q1, camera, camera)
    result["projected_flow"].square().sum().backward()
    assert q0.grad is not None and q1.grad is not None
    assert torch.isfinite(q0.grad) and torch.isfinite(q1.grad)


def test_gaussian_model_replays_frozen_kpart_joint_state():
    from scene.gaussian_model import GaussianModel
    from scene.material_assignment_field import MaterialAssignmentField

    model = GaussianModel(0)
    model._xyz = torch.tensor([[0.8, 0.0, 0.3], [0.0, 0.5, 0.2], [-0.5, 0.0, 0.1]])
    model._rotation = torch.tensor([[1.0, 0.0, 0.0, 0.0]]).repeat(3, 1)
    model.material_assignment = MaterialAssignmentField(
        torch.tensor([True, False, False]), 1
    )
    model.motion_representation = "rigid_killing_eulerian"
    joint = fit_revolute_joint(_revolute_mode(), torch.tensor([0.0, 0.5], dtype=torch.float64))
    responsibility = torch.tensor([[0.01, 0.99], [0.99, 0.01], [0.5, 0.5]])
    model.set_kpart_joint_state(
        [joint],
        torch.tensor([[0.0, 0.5]]),
        torch.tensor([[0.5, 0.5]]),
        torch.tensor([0.0, 1.0]),
        responsibility,
        torch.tensor([1, 0, -1]),
    )
    xyz, _, velocity, labels = model.dynamic_state_at(1.0)
    assert labels.tolist() == [1, 0, -1]
    assert torch.equal(xyz[1:], model._xyz[1:])
    assert torch.equal(velocity[1:], torch.zeros_like(velocity[1:]))
    assert not torch.allclose(xyz[0], model._xyz[0])
