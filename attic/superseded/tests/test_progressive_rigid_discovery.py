"""Synthetic checks for the progressive rigid discovery Stage-1 core.

Run:
    PYTHONPATH=. python tests/test_progressive_rigid_discovery.py
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from types import SimpleNamespace

import torch

from utils.affected_render_support import affected_union_mask
from utils.candidate_component_builder import CandidateComponentConfig, build_candidate_components
from utils.multiframe_twist_refinement import monotonic_coordinates_from_raw
from utils.progressive_assignment import (
    ComponentDecisionConfig,
    analytic_component_evidence,
    apply_component_acceptance,
    decide_component,
    transaction_decision,
)
from utils.progressive_counterfactual import render_difference_stats
from utils.progressive_rigid_state import STATUS_UNCERTAIN, allocate_progressive_state, status_for_part
from utils.rigid_endpoint_fit import (
    RansacConfig,
    build_source_neighbor_graph,
    correspondence_estep,
    fm_guided_correspondence_em,
    ransac_candidate_se3,
    se3_exp,
    se3_log,
    transform_points,
    weighted_kabsch,
    ransac_fit_se3,
)
from tools.train_progressive_rigid_discovery import _apply_strict_config_lock, _inner_rigid_fit, _load_gt_reference_axis, build_parser
from tools._closed_loop_scene import _endpoint_candidates, _find_model_ply


def _grid(n: int = 7) -> torch.Tensor:
    x = torch.linspace(-0.3, 0.3, n)
    y = torch.linspace(-0.2, 0.2, n)
    xx, yy = torch.meshgrid(x, y, indexing="ij")
    zz = 0.1 * torch.sin(xx * 3.0)
    return torch.stack((xx.reshape(-1), yy.reshape(-1), zz.reshape(-1)), dim=-1)


def _state(points: torch.Tensor) -> object:
    n = points.shape[0]
    disp = torch.zeros_like(points)
    return allocate_progressive_state(
        canonical_all=points.clone(),
        canonical_fg=points.clone(),
        foreground_idx=torch.arange(n),
        foreground_mask=torch.ones(n, dtype=torch.bool),
        frame_ids=[0, 1],
        frame_times=[0.0, 1.0],
        num_moving_parts=1,
        fm_endpoint_xyz=points + disp,
        fm_displacement=disp,
        fm_probability=torch.full((n,), 0.5),
        fm_confidence=torch.ones(n),
        fm_valid=torch.ones(n, dtype=torch.bool),
    )


def test_weighted_kabsch_exact_transform():
    source = _grid()
    xi = torch.tensor([0.1, -0.04, 0.03, 0.0, 0.0, 0.35])
    T = se3_exp(xi)
    target = transform_points(T, source)
    fit = weighted_kabsch(source, target, torch.ones(source.shape[0]))
    assert torch.allclose(fit["T"], T, atol=1.0e-4), fit["T"]
    assert fit["weighted_rms"] < 1.0e-5


def test_ransac_se3_ignores_endpoint_outliers():
    source = _grid()
    xi = torch.tensor([0.07, -0.02, 0.01, 0.0, 0.0, -0.25])
    T = se3_exp(xi)
    target = transform_points(T, source)
    target[:8] = torch.randn_like(target[:8]) * 3.0
    confidence = torch.ones(source.shape[0])
    fit = ransac_fit_se3(
        source, target, confidence,
        RansacConfig(iters=120, inlier_threshold=0.03, irls_iters=4, huber_delta=0.02, seed=2),
    )
    recovered = fit["T"]
    good = torch.arange(source.shape[0]) >= 8
    err = torch.linalg.norm(transform_points(recovered, source[good]) - transform_points(T, source[good]), dim=-1).median()
    assert float(err) < 0.01, float(err)


def test_se3_log_exp_round_trip():
    xi = torch.tensor([0.03, -0.01, 0.02, 0.04, -0.03, 0.2])
    recovered = se3_log(se3_exp(xi))
    assert torch.allclose(recovered, xi, atol=1.0e-4), recovered


def test_source_neighbor_graph_excludes_self_and_matches_true_distances():
    points = torch.tensor([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 3.0, 0.0]])
    graph = build_source_neighbor_graph(points, k=2)
    assert int(graph["neighbor_idx"][0, 0]) == 1  # nearest to point 0 is point 1, not itself
    assert torch.allclose(graph["neighbor_dist0"][0, 0], torch.tensor(1.0), atol=1.0e-5)
    assert torch.allclose(graph["neighbor_dist0"][0, 1], torch.tensor(3.0), atol=1.0e-5)


def test_correspondence_estep_does_not_collapse_to_origin_when_evidence_underflows():
    # If the current transform sits far enough from EVERY candidate that all
    # of a point's likelihoods underflow together, expected_target must NOT
    # silently collapse toward the coordinate origin with spurious full
    # confidence (the old `.clamp_min(1e-12)` denominator did exactly this:
    # a too-small-to-matter posterior sum got floored up to 1e-12, which
    # crushed the already-tiny posterior instead of renormalising it,
    # producing an expected_target near [0,0,0] with corr_confidence ~1.0 --
    # verified on real data to corrupt the rigid fit on the very first EM
    # iteration). It must fall back to a uniform posterior over the valid
    # candidates instead, reporting low/zero confidence.
    far_away_candidates = torch.tensor([[[50.0, 0.0, 0.0], [50.1, 0.0, 0.0]]])  # one point, 2 candidates
    source = torch.tensor([[0.0, 0.0, 0.0]])
    T = torch.eye(4)  # pred = [0,0,0], ~50 units from every candidate
    candidate_prior = torch.full((1, 2), 0.5)
    candidate_valid = torch.ones((1, 2), dtype=torch.bool)
    fm_endpoint_xyz = torch.tensor([[0.0, 0.0, 0.0]])

    estep = correspondence_estep(
        source, T, far_away_candidates, candidate_prior, candidate_valid, fm_endpoint_xyz,
        spcd=0.01,
    )
    assert torch.allclose(estep["posterior"].sum(), torch.tensor(1.0), atol=1.0e-4)
    assert torch.allclose(estep["posterior"], torch.full((1, 2), 0.5), atol=1.0e-4)
    expected_displacement = torch.linalg.norm(estep["expected_target"][0] - source[0])
    assert expected_displacement > 40.0  # near the candidates, not collapsed to the origin
    assert float(estep["corr_confidence"][0]) < 0.5  # "don't know", not spuriously full confidence


def test_correspondence_isometry_favors_neighbour_consistent_candidate():
    # Point 0's t0 neighbour is point 1, one unit away.  Point 0 has two
    # candidate frame-T targets: c0 exactly matches the rigid-transform
    # prediction but sits far (distance 3) from where point 1's t0 neighbour
    # is currently believed to be -- as if c0 were actually nearby static
    # structure the transform happens to pass through; c1 fits the rigid
    # prediction poorly (distance 2) but exactly preserves the true t0
    # neighbour distance (1.0) to that same reference point.
    source = torch.tensor([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    T = torch.eye(4)
    T[:3, 3] = torch.tensor([3.0, 0.0, 0.0])
    c0 = torch.tensor([3.0, 0.0, 0.0])
    c1 = torch.tensor([1.0, 0.0, 0.0])
    candidate_xyz = torch.stack([torch.stack([c0, c1]), torch.stack([c0, c1])])
    candidate_prior = torch.full((2, 2), 0.5)
    candidate_valid = torch.ones((2, 2), dtype=torch.bool)
    fm_endpoint_xyz = torch.stack([c0, c0])
    neighbor_idx = torch.tensor([[1], [0]])
    neighbor_dist0 = torch.tensor([[1.0], [1.0]])
    neighbor_target_xyz = torch.tensor([[[0.0, 0.0, 0.0]], [[0.0, 0.0, 0.0]]])

    without = correspondence_estep(
        source, T, candidate_xyz, candidate_prior, candidate_valid, fm_endpoint_xyz,
        spcd=0.5, neighbor_idx=neighbor_idx, neighbor_dist0=neighbor_dist0,
        neighbor_target_xyz=neighbor_target_xyz, isometry_weight=0.0,
    )
    with_isometry = correspondence_estep(
        source, T, candidate_xyz, candidate_prior, candidate_valid, fm_endpoint_xyz,
        spcd=0.5, neighbor_idx=neighbor_idx, neighbor_dist0=neighbor_dist0,
        neighbor_target_xyz=neighbor_target_xyz, isometry_weight=3.0,
    )
    assert int(without["selected"][0]) == 0
    assert int(with_isometry["selected"][0]) == 1


def test_component_acceptance_is_isolated():
    points = _grid()
    state = _state(points)
    old_labels = state.hard_labels.clone()
    mask = torch.zeros(points.shape[0], dtype=torch.bool)
    mask[3:9] = True
    apply_component_acceptance(state, mask, 1, probability=0.9)
    assert bool((state.hard_labels[mask] == 1).all())
    assert torch.equal(state.hard_labels[~mask], old_labels[~mask])


def test_component_decision_defers_on_low_observability():
    # Evidence looks decisive, but too little of the component is actually
    # observed to trust it -- this is the only thing "defer" should mean now
    # that the analytic verdict itself always commits to a sign.
    scores = {
        0: {"fm": 0.02, "pcd": 0.05, "render": 0.5, "depth": 0.1, "count": 50, "render_count": 500},
        1: {"fm": 0.08, "pcd": 0.03, "render": 0.3, "depth": 0.05, "count": 50, "render_count": 500},
    }
    decision = decide_component(
        scores,
        observable_fraction=0.01,
        reliability=1.0,
        cfg=ComponentDecisionConfig(),
    )
    assert decision["action"] == "defer"
    assert "low_observability" in decision["reason"]


def test_component_decision_commits_given_valid_evidence():
    # Moving fits better on every modality with a large, well-observed
    # sample: the analytic evidence should commit, not sit on a fence the
    # way a fixed probability threshold could.
    scores = {
        0: {"fm": 0.05, "pcd": 0.05, "render": 0.05, "depth": 0.05, "count": 200, "render_count": 2000},
        1: {"fm": 0.01, "pcd": 0.01, "render": 0.01, "depth": 0.01, "count": 200, "render_count": 2000},
    }
    decision = decide_component(
        scores,
        observable_fraction=1.0,
        reliability=1.0,
        cfg=ComponentDecisionConfig(),
    )
    assert decision["action"] == "accept"
    assert decision["label"] == 1


def test_analytic_evidence_requires_more_signal_from_smaller_components():
    # Same relative cost gap for both; the BIC penalty (scaled by how many
    # independent observations back it) should let the well-observed
    # component commit to moving while the tiny one can't clear the bar --
    # a fixed probability cutoff is blind to this sample-size difference.
    small = {0: {"fm": 0.011, "count": 3}, 1: {"fm": 0.010, "count": 3}}
    large = {0: {"fm": 0.011, "count": 3000}, 1: {"fm": 0.010, "count": 3000}}
    small_evidence = analytic_component_evidence(small, label=1)
    large_evidence = analytic_component_evidence(large, label=1)
    assert small_evidence["log_evidence_difference"] < 0.0 < large_evidence["log_evidence_difference"]


def test_affected_support_is_local():
    camera = SimpleNamespace(
        image_height=32,
        image_width=32,
        full_proj_transform=torch.eye(4),
        world_view_transform=torch.eye(4),
    )
    positions = torch.tensor([[-0.8, -0.8, 1.0], [0.8, 0.8, 1.0]])
    moved = positions + torch.tensor([[0.2, 0.0, 0.0], [0.0, 0.0, 0.0]])
    mask = affected_union_mask(camera, positions, moved, torch.tensor([0]), radius_px=2)
    assert int(mask.sum()) > 0
    assert int(mask[:8, :8].sum()) > 0
    assert int(mask[-8:, -8:].sum()) == 0


def test_uncertain_status_is_not_a_fake_label():
    points = _grid()
    state = _state(points)
    state.status[:] = STATUS_UNCERTAIN
    state.hard_labels[:] = 0
    state.validate()
    assert int(state.uncertain_mask().sum()) == points.shape[0]
    assert bool((state.hard_labels == 0).all())


def test_monotonic_coordinate_parameterization():
    raw = torch.tensor([-10.0, 0.0, 1.0, -1.0])
    q = monotonic_coordinates_from_raw(raw)
    assert bool((q[1:] - q[:-1] > 0).all())


def test_fm_endpoint_is_not_exact_correspondence():
    source = _grid(6)
    true_T = se3_exp(torch.tensor([0.05, -0.03, 0.02, 0.02, -0.01, 0.25]))
    correct = transform_points(true_T, source)
    gen = torch.Generator().manual_seed(123)
    wrong = correct + 0.35 * torch.randn(correct.shape, generator=gen)
    fm_endpoint = wrong.clone()
    direct = weighted_kabsch(source, fm_endpoint, torch.ones(source.shape[0]))["T"]
    direct_err = torch.linalg.norm(transform_points(direct, source) - correct, dim=-1).median()
    candidate_xyz = torch.stack((wrong, correct), dim=1)
    candidate_idx = torch.stack(
        (torch.arange(source.shape[0]), torch.arange(source.shape[0]) + source.shape[0]),
        dim=1,
    )
    candidate_prior = torch.tensor([0.6, 0.4]).expand(source.shape[0], 2).clone()
    candidate_valid = torch.ones_like(candidate_prior, dtype=torch.bool)
    support = torch.ones(source.shape[0], dtype=torch.bool)
    ransac = ransac_candidate_se3(
        source,
        candidate_xyz,
        candidate_idx,
        candidate_prior,
        candidate_valid,
        fm_endpoint,
        support,
        torch.ones(source.shape[0]),
        threshold=0.03,
        spcd=0.01,
        iters=400,
        seed=4,
        print_fn=lambda *_args, **_kwargs: None,
    )
    em = fm_guided_correspondence_em(
        source,
        candidate_xyz,
        candidate_idx,
        candidate_prior,
        candidate_valid,
        fm_endpoint,
        support,
        torch.ones(source.shape[0]),
        init_T=ransac["T"],
        spcd=0.01,
        irls_iters=4,
        huber_delta=0.02,
        em_iters=5,
        print_fn=lambda *_args, **_kwargs: None,
    )
    em_err = torch.linalg.norm(transform_points(em["T"], source) - correct, dim=-1).median()
    assert float(direct_err) > 0.1
    assert float(em_err) < 0.01, float(em_err)


def test_static_points_cannot_support_moving_transform():
    moving = _grid(4)
    static = _grid(4) + torch.tensor([2.0, 0.0, 0.0])
    source = torch.cat((moving, static), dim=0)
    true_T = se3_exp(torch.tensor([0.1, 0.0, 0.0, 0.0, 0.0, 0.1]))
    target = transform_points(true_T, source)
    candidate_xyz = target[:, None]
    candidate_idx = torch.arange(source.shape[0])[:, None]
    candidate_prior = torch.ones(source.shape[0], 1)
    candidate_valid = torch.ones(source.shape[0], 1, dtype=torch.bool)
    seed_mask = torch.zeros(source.shape[0], dtype=torch.bool)
    seed_mask[: moving.shape[0]] = True
    ransac = ransac_candidate_se3(
        source,
        candidate_xyz,
        candidate_idx,
        candidate_prior,
        candidate_valid,
        target,
        seed_mask,
        torch.ones(source.shape[0]),
        threshold=0.02,
        spcd=0.01,
        iters=80,
        seed=7,
        print_fn=lambda *_args, **_kwargs: None,
    )
    assert ransac["summary"]["seed_source_count"] == moving.shape[0]
    assert ransac["summary"]["inliers"] <= moving.shape[0]


def test_counterfactual_render_difference_stats():
    image0 = torch.zeros(3, 8, 8)
    image1 = image0.clone()
    image1[:, 3:5, 3:5] = 1.0
    mask = torch.zeros(8, 8, dtype=torch.bool)
    mask[3:5, 3:5] = True
    stats = render_difference_stats(image0, image1, None, None, mask)
    assert stats["affected_pixels"] == 4
    assert stats["rgb_diff"] > 0.0


def _clustered_state() -> object:
    clusters = []
    for offset in (0.0, 1.0, 2.0):
        base = torch.tensor([offset, 0.0, 0.0])
        clusters.append(base + 0.01 * torch.randn(5, 3, generator=torch.Generator().manual_seed(int(offset * 10 + 1))))
    points = torch.cat(clusters, dim=0)
    state = _state(points)
    state.status[:] = STATUS_UNCERTAIN
    state.observable[:] = True
    state.visibility_mass[:] = 1.0
    for cid, start in enumerate((0, 5, 10)):
        state.fm_displacement[start:start + 5] = torch.tensor([float(cid), 0.0, 0.0])
    return state


def test_resolved_candidate_never_reselected():
    state = _clustered_state()
    state.resolved_once[:5] = True
    result = build_candidate_components(
        state,
        CandidateComponentConfig(knn=4, max_components=3, min_size=3, distance_scale=0.05, fm_scale=0.05),
    )
    selected = torch.cat(result["components"]) if result["components"] else torch.empty(0, dtype=torch.long)
    assert not bool((selected < 5).any())


def test_uncertainty_is_current_not_stale():
    state = _clustered_state()
    state.render_residual_per_gaussian[:] = 1.0
    state.responsibilities[:5] = torch.tensor([0.999, 0.001])
    state.responsibilities[5:10] = torch.tensor([0.5, 0.5])
    state.responsibilities[10:] = torch.tensor([0.999, 0.001])
    result = build_candidate_components(
        state,
        CandidateComponentConfig(knn=4, max_components=3, min_size=3, distance_scale=0.05, fm_scale=0.05),
    )
    assert result["leaderboard"][0]["uncertainty"] > 0.4


def test_residual_first_ranking():
    state = _clustered_state()
    state.responsibilities[:] = torch.tensor([0.5, 0.5])
    state.render_residual_per_gaussian[:5] = 0.2
    state.render_residual_per_gaussian[5:10] = 0.001
    state.fm_probability[:5] = 0.55
    state.fm_probability[5:10] = 0.99
    result = build_candidate_components(
        state,
        CandidateComponentConfig(knn=4, max_components=3, min_size=3, distance_scale=0.05, fm_scale=0.05),
    )
    assert result["leaderboard"][0]["mean_residual"] > 0.1


def test_static_acceptance_skips_transform_refit():
    state = _state(_grid())
    before = state.part_transforms_T.clone()
    mask = torch.zeros(state.num_foreground, dtype=torch.bool)
    mask[:4] = True
    apply_component_acceptance(state, mask, 0, probability=0.99)
    assert torch.equal(state.part_transforms_T, before)
    assert bool(state.resolved_once[mask].all())


def test_moving_acceptance_transaction_rolls_back_pcd_conflict():
    before = {"fm": 10.0, "pcd": 1.0, "rgb": 0.1, "depth": 0.1, "total": 10.27}
    after = {"fm": 9.0, "pcd": 2.0, "rgb": 0.1, "depth": 0.1, "total": 9.52}
    tolerance = {"fm": 1.0e-6, "pcd": 1.0e-6, "rgb": 1.0e-6, "depth": 1.0e-6, "total": 1.0e-6}
    decision = transaction_decision(before, after, tolerance)
    assert decision["decision"] == "ROLLBACK"
    assert "fm_improved_but_pcd_worsened" in decision["failures"]


def test_inner_rigid_fit_rolls_back_fm_only_drift():
    points = _grid(4)
    state = _state(points)
    state.fm_endpoint_xyz = points + torch.tensor([0.2, 0.0, 0.0])
    state.fm_displacement = state.fm_endpoint_xyz - points
    state.fm_probability[:] = 1.0
    state.fm_confidence_base[:] = 1.0
    state.fm_confidence_current[:] = 1.0
    state.fm_valid[:] = True
    state.status[:] = status_for_part(1)
    state.hard_labels[:] = 1
    state.part_support_weight[0] = 1.0
    before = state.part_transforms_T.clone()
    args = SimpleNamespace(
        inner_rigid_lr=0.05,
        inner_rigid_iters=8,
        inner_trust_region=1.0,
        rigid_huber_delta=0.02,
        lambda_fm=1.0,
        lambda_pcd=0.25,
        lambda_rgb=0.0,
        lambda_depth=0.0,
        pcd_fit_samples=0,
        pcd_target_samples=0,
        pcd_chunk_size=64,
        inner_render_cameras=0,
        log_interval=9999,
    )
    result = _inner_rigid_fit(state, {}, points.clone(), args, desc="test inner rollback")
    assert result["last"]["rolled_back"] is True
    assert torch.allclose(state.part_transforms_T, before)


def test_strict_config_lock_rejects_loss_tuning():
    args = build_parser().parse_args([
        "--root-scene", ".",
        "--static-model-dir", ".",
        "--output-dir", ".",
        "--strict-stage1-bugfix-validation",
        "--lambda-pcd", "1.0",
    ])
    with tempfile.TemporaryDirectory() as tmp:
        try:
            _apply_strict_config_lock(args, Path(tmp))
        except SystemExit as error:
            assert "Strict bug-fix validation" in str(error)
        else:
            raise AssertionError("strict config lock accepted tuned lambda-pcd")


def test_gt_reference_axis_loads_from_scene_metadata():
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        (root / "pointcloud").mkdir()
        (root / "pointcloud" / "joint_metadata.json").write_text(
            '{"name":"j","type":"continuous","axis_vector":[0,0,2],"coordinate_frame":"render_world"}',
            encoding="utf-8",
        )
        info = _load_gt_reference_axis(root)
    assert info is not None
    assert info["axis_vector"] == [0.0, 0.0, 1.0]
    assert info["ground_truth_used_as_algorithm_input"] is False


def test_endpoint_dir_alias_frame59_finds_frame_t_static():
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        shared = root / "shared"
        ply = shared / "frame_t_static" / "point_cloud" / "iteration_14000" / "point_cloud.ply"
        ply.parent.mkdir(parents=True)
        ply.write_text("ply\n", encoding="utf-8")
        args = SimpleNamespace(
            endpoint_static_model_dir=str(shared / "frame59_static"),
            static_model_dir=str(shared / "frame0_static"),
        )
        found = None
        for candidate in _endpoint_candidates(args):
            found = _find_model_ply(candidate, 14000)
            if found is not None:
                break
    assert found == ply.resolve()


def main() -> None:
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"PASS {name}")


if __name__ == "__main__":
    main()
