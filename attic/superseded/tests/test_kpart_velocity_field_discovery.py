from types import SimpleNamespace

import torch

from utils.kpart_counterfactual_rendering import (
    extract_uncertain_components,
    rank_components_for_rendering,
    update_persistent_render_evidence,
)
from utils.kpart_velocity_field_discovery import (
    _apply_stage1_domain_constraints,
    _mask_unary_with_stage1_domain,
    _prior_kpart_initialization,
)
from utils.kpart_dino_boundary import (
    compute_3d_same_material_affinity,
    compute_one_sided_dino_seam_penalty,
)
from utils.kpart_field_merge import (
    compute_shared_field_merge_penalty,
    fit_region_rigid_field,
    fit_shared_field_to_union,
    merge_until_k,
)
from utils.kpart_soft_responsibility import compute_velocity_field_unaries
from utils.rigid_field_integration import integrate_rigid_field
from utils.rigid_killing_motion import RigidKillingMode, predict_rigid_cell_velocity


def _mode(axis=(0.0, 0.0, 1.0), linear=(0.0, 0.0, 0.0)):
    return RigidKillingMode(
        reference_point=torch.zeros(3, dtype=torch.float64),
        reference_velocity=torch.tensor(linear, dtype=torch.float64),
        angular_mode=torch.tensor(axis, dtype=torch.float64),
        mode_type="rotational" if any(axis) else "translational",
        normalization_scale=1.0,
        valid=True,
        diagnostics={},
    )


def _observations(points, mode, speed, noise=0.0):
    xyz = torch.as_tensor(points).double()
    positions = xyz[None].expand(len(speed), -1, -1).clone()
    velocity = predict_rigid_cell_velocity(mode, positions, torch.as_tensor(speed).double())
    if noise:
        generator = torch.Generator().manual_seed(4)
        velocity += noise * torch.randn(velocity.shape, generator=generator, dtype=velocity.dtype)
    count = velocity.shape[:2]
    return {
        "positions": positions,
        "velocity": velocity,
        "confidence": torch.ones(count, dtype=torch.float64),
        "valid": torch.ones(count, dtype=torch.bool),
        "weight_sum": torch.ones(count, dtype=torch.float64),
        "normal_matrix": torch.eye(3, dtype=torch.float64)[None, None].expand(*count, -1, -1).clone(),
    }


def _solver_kwargs():
    return {"max_iters": 15, "min_valid_cells": 6, "huber_delta": 0.02}


def test_static_zero_field_beats_unrelated_motion_for_noisy_static_points():
    points = torch.randn(30, 3, generator=torch.Generator().manual_seed(1)).double()
    pack = _observations(points, _mode(axis=(0, 0, 0), linear=(1, 0, 0)), [0.0, 0.0, 0.0], noise=1.0e-3)
    field = {"mode": _mode(), "temporal_speed": torch.ones(3), "valid": True}
    unary = compute_velocity_field_unaries(pack, [field])["unaries"]
    assert float(unary[:, 0].mean()) < float(unary[:, 1].mean())


def test_full_field_unary_keeps_near_axis_moving_points_out_of_static():
    points = torch.tensor([[1.0e-2, 0.0, z] for z in torch.linspace(-1, 1, 20)], dtype=torch.float64)
    mode = _mode()
    speed = torch.tensor([0.4, -0.2, 0.8], dtype=torch.float64)
    pack = _observations(points, mode, speed)
    field = {"mode": mode, "temporal_speed": speed, "valid": True}
    unary = compute_velocity_field_unaries(pack, [field])["unaries"]
    assert bool((unary[:, 1] < unary[:, 0]).all())


def test_shared_field_merge_accepts_one_body_split_into_supernodes():
    points = torch.randn(50, 3, generator=torch.Generator().manual_seed(2)).double()
    pack = _observations(points, _mode(), [0.2, 0.7, -0.4, 0.0])
    first = torch.zeros(50); first[:25] = 1
    second = torch.zeros(50); second[25:] = 1
    a = fit_region_rigid_field(pack, first, solver_kwargs=_solver_kwargs())
    b = fit_region_rigid_field(pack, second, solver_kwargs=_solver_kwargs())
    shared = fit_shared_field_to_union(pack, first, second, solver_kwargs=_solver_kwargs())
    assert a["valid"] and b["valid"] and shared["valid"]
    assert a["active_indices"].numel() == 25
    assert b["active_indices"].numel() == 25
    assert abs(compute_shared_field_merge_penalty(a, b, shared)) < 1.0e-5


def test_shared_field_merge_rejects_distinct_fields_with_similar_speed():
    generator = torch.Generator().manual_seed(3)
    points = torch.randn(60, 3, generator=generator).double()
    first = torch.zeros(60); first[:30] = 1
    second = 1.0 - first
    speed = torch.tensor([0.5, -0.5, 0.8], dtype=torch.float64)
    positions = points[None].expand(3, -1, -1).clone()
    velocity = torch.zeros_like(positions)
    velocity[:, :30] = predict_rigid_cell_velocity(_mode(axis=(0, 0, 1)), positions[:, :30], speed)
    velocity[:, 30:] = predict_rigid_cell_velocity(_mode(axis=(0, 1, 0)), positions[:, 30:], speed)
    pack = {"positions": positions, "velocity": velocity, "confidence": torch.ones(3, 60), "valid": torch.ones(3, 60, dtype=torch.bool)}
    a = fit_region_rigid_field(pack, first, solver_kwargs=_solver_kwargs())
    b = fit_region_rigid_field(pack, second, solver_kwargs=_solver_kwargs())
    shared = fit_shared_field_to_union(pack, first, second, solver_kwargs=_solver_kwargs())
    assert a["valid"] and b["valid"]
    assert (not shared["valid"]) or compute_shared_field_merge_penalty(a, b, shared) > 0.1


def test_k_greater_than_one_merge_recovers_two_physical_fields():
    points = torch.randn(90, 3, generator=torch.Generator().manual_seed(8)).double()
    positions = points[None].expand(4, -1, -1).clone()
    speed = torch.tensor([0.3, 0.8, -0.4, 0.1], dtype=torch.float64)
    velocity = torch.zeros_like(positions)
    velocity[:, :60] = predict_rigid_cell_velocity(_mode(axis=(0, 0, 1)), positions[:, :60], speed)
    velocity[:, 60:] = predict_rigid_cell_velocity(_mode(axis=(0, 1, 0)), positions[:, 60:], speed)
    pack = {"positions": positions, "velocity": velocity, "confidence": torch.ones(4, 90), "valid": torch.ones(4, 90, dtype=torch.bool)}
    labels = torch.arange(90) // 30
    pairs = torch.stack((torch.arange(89), torch.arange(1, 90)), -1)
    merged = merge_until_k(
        labels, pack, pairs, points,
        target_k=2,
        dynamic_likelihood=torch.ones(90),
        dino_distance=torch.zeros(89),
        solver_kwargs=_solver_kwargs(),
    )
    assert len(merged["fields"]) == 2
    assert int(merged["labels"][0]) == int(merged["labels"][59])
    assert int(merged["labels"][0]) != int(merged["labels"][60])


def test_nonconstant_pause_and_reversal_remain_one_factorized_field():
    points = torch.randn(45, 3, generator=torch.Generator().manual_seed(5)).double()
    speed = torch.tensor([0.2, 0.9, 0.0, -0.6, -0.1], dtype=torch.float64)
    pack = _observations(points, _mode(), speed)
    fitted = fit_region_rigid_field(pack, torch.ones(45), solver_kwargs=_solver_kwargs())
    assert fitted["valid"]
    recovered = fitted["temporal_speed"]
    nonzero = speed.abs() > 1.0e-6
    assert bool((torch.sign(recovered[nonzero]) == torch.sign(speed[nonzero])).all())
    assert abs(float(recovered[2])) < 1.0e-8


def test_dino_is_soft_same_material_and_one_sided_seam_only():
    points = torch.tensor([[0.0, 0, 0], [0.1, 0, 0], [0.2, 0, 0]])
    pairs = torch.tensor([[0, 1], [1, 2]])
    affinity = compute_3d_same_material_affinity(points, pairs, torch.tensor([0.0, 2.0]))["affinity"]
    assert affinity[0] > affinity[1]
    part_seam = torch.tensor([[1.0, 1.0]])
    smooth = compute_one_sided_dino_seam_penalty(part_seam, torch.tensor([[0.0, 0.0]]))
    supported = compute_one_sided_dino_seam_penalty(part_seam, torch.tensor([[1.0, 1.0]]))
    assert smooth > supported


def test_counterfactual_regions_are_coherent_ranked_by_mass_and_persistent():
    pairs = torch.tensor([[0, 1], [1, 2], [3, 4]])
    uncertain = torch.tensor([True, True, True, True, True])
    components = extract_uncertain_components(pairs, torch.ones(3), uncertain)
    ranked = rank_components_for_rendering(components, torch.ones(5), max_components=1)
    assert ranked[0].numel() == 3
    state = {"evidence": torch.full((5, 2), float("nan")), "valid": torch.zeros(5, 2, dtype=torch.bool)}
    state = update_persistent_render_evidence(state, ranked[0], {1: 2.0}, cycle=1)
    retained = state["evidence"].clone()
    state = update_persistent_render_evidence(state, torch.tensor([3, 4]), {0: 1.0}, cycle=2)
    assert torch.equal(state["evidence"][:3, 1], retained[:3, 1])


def test_rigid_field_integration_is_exact_and_static_is_identity():
    timestamps = torch.tensor([0.0, 0.2, 0.7, 1.0], dtype=torch.float64)
    speed = torch.tensor([1.0, -0.5, 2.0], dtype=torch.float64)
    result = integrate_rigid_field(_mode(), speed, 0.0, 1.0, timestamps)
    expected_q = 1.0 * 0.2 - 0.5 * 0.5 + 2.0 * 0.3
    assert abs(float(result["integrated_coordinate"]) - expected_q) < 1.0e-12
    assert torch.allclose(result["R_ab"].T @ result["R_ab"], torch.eye(3, dtype=torch.float64), atol=1.0e-10)
    static = integrate_rigid_field(None, None, 0.0, 1.0, timestamps)
    assert torch.equal(static["T_ab"], torch.eye(4, dtype=torch.float64))


def test_prior_kpart_initialization_freezes_static_leftover_and_dynamic_anchors():
    points = torch.tensor(
        [
            [0.0, 0.0, 0.0],
            [0.1, 0.0, 0.0],
            [0.2, 0.0, 0.0],
            [0.3, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.1, 0.0, 0.0],
            [1.2, 0.0, 0.0],
            [1.3, 0.0, 0.0],
        ],
        dtype=torch.float32,
    )
    probability = torch.tensor([0.02, 0.05, 0.10, 0.20, 0.75, 0.80, 0.95, 0.98])
    bootstrap = {"dynamic_likelihood": probability}
    prior = {
        "method": "flow_matching",
        "probability": probability,
        "confidence": torch.ones_like(probability),
        "valid": torch.ones_like(probability, dtype=torch.bool),
    }
    anchors = {
        "static_anchor": probability <= 0.10,
        "dynamic_anchor": probability >= 0.95,
        "active_mask": probability >= 0.75,
    }
    args = SimpleNamespace(num_moving_parts=1, rigid_min_valid_cells=1)

    init = _prior_kpart_initialization(points, bootstrap, prior, anchors, args)
    assert init["cluster_sizes"] == [4]
    assert bool((init["responsibilities"][:4, 0] == 1.0).all())
    assert bool((init["responsibilities"][6:, 0] == 0.0).all())

    unary = torch.zeros(points.shape[0], 2)
    masked = _mask_unary_with_stage1_domain(unary, init)
    assert bool((masked[:4, 1] > 1000.0).all())
    assert bool((masked[6:, 0] > 1000.0).all())

    free = torch.full((points.shape[0], 2), 0.5)
    constrained = _apply_stage1_domain_constraints(free, init)
    assert bool((constrained[:4, 0] == 1.0).all())
    assert bool((constrained[6:, 0] == 0.0).all())
