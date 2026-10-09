"""Synthetic acceptance tests from the pure-depth geometry specification.

Run directly in the project environment (pytest is not required):
    conda run -n rigeuler3dgs python tests/test_depth_geometry_relative_hungarian.py
"""

import math
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from scipy.optimize import linear_sum_assignment

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from depth_geometry_relative_hungarian import (  # noqa: E402
    FrameCloud,
    ResidualCloud,
    Track,
    adaptive_geometric_coreset,
    apply_static_anchor_only_labels,
    annotate_skip_consistency,
    build_full_resolution_residuals,
    build_residual_graph,
    build_residual_tracks,
    dynamic_labels_for_frame,
    build_rigidity_graph,
    build_static_anchor_ghost_priors,
    build_tracks,
    final_temporal_hungarian_coherence,
    match_frame_pair,
    optimize_contrastive_groups,
    optimize_binary_dynamic_probabilities,
    recover_static_fragments_from_anchors,
    recover_static_from_temporal_reappearance,
    match_residual_graph_pair,
    solve_short_fragments_with_certain_knn,
    validate_dynamic_anchors,
    validate_se3_models,
    voxel_reduce,
    write_ply,
)


def rotation(axis, angle):
    axis = np.asarray(axis, np.float64)
    axis /= np.linalg.norm(axis)
    x, y, z = axis
    skew = np.asarray([[0, -z, y], [z, 0, -x], [-y, x, 0]])
    return np.eye(3) + math.sin(angle) * skew + (1 - math.cos(angle)) * (skew @ skew)


def matching_accuracy(source, target, target_identity):
    accepted, diagnostics, _stats = match_frame_pair(source, target)
    correct = sum(target_identity[row["j"]] == row["i"] for row in accepted)
    return correct / max(len(accepted), 1), accepted, diagnostics


def test_a_one_rigid_body_se3():
    cases = [
        (np.eye(3), np.asarray([0.8, -0.2, 0.4])),
        (rotation([0.2, 0.7, 0.4], 1.1), np.zeros(3)),
        (rotation([0.8, -0.1, 0.5], 0.8), np.asarray([-0.4, 0.9, 0.2])),
    ]
    for case, (rigid_rotation, translation) in enumerate(cases):
        rng = np.random.default_rng(100 + case)
        source = rng.normal(size=(28, 3))
        transformed = source @ rigid_rotation.T + translation
        order = rng.permutation(len(source))
        accuracy, accepted, _ = matching_accuracy(source, transformed[order], order)
        assert len(accepted) == len(source)
        assert accuracy == 1.0


def test_a2_preliminary_icp_anchors_only_stationary_points():
    rng = np.random.default_rng(177)
    static = rng.normal(scale=0.2, size=(20, 3))
    moving = rng.normal(scale=0.2, size=(20, 3)) + np.asarray([2.0, 0.0, 0.0])
    source = np.concatenate((static, moving))
    target = source.copy()
    target[20:] += np.asarray([0.05, 0.02, 0.0])
    order = rng.permutation(len(target))
    accepted, _diagnostics, stats = match_frame_pair(source, target[order])
    anchors = [row for row in accepted if row["preliminary_static_anchor"]]
    assert stats["preliminary_static_anchors"] == 20
    assert len(anchors) == 20
    assert all(row["i"] < 20 and order[row["j"]] == row["i"] for row in anchors)


def synthetic_tracks(num_moving_parts):
    rng = np.random.default_rng(200 + num_moving_parts)
    frames = list(range(7))
    tracks, truth = {}, {}
    track_id = 0
    for group in range(num_moving_parts + 1):
        base = rng.normal(scale=0.25, size=(10, 3)) + np.asarray([1.5 * group, 0.2 * group, 0.0])
        for point in base:
            track = Track(track_id)
            truth[track_id] = group
            for frame in frames:
                if group == 0:
                    transformed = point
                else:
                    rigid_rotation = rotation([0.15 * group, 0.25, 1.0], 0.11 * group * frame)
                    translation = np.asarray([0.05 * group * frame, -0.025 * frame, 0.02 * group * frame])
                    transformed = point @ rigid_rotation.T + translation
                track.points[frame] = transformed.copy()
                track.point_ids[frame] = track_id
            track.match_confidences = [0.85] * (len(frames) - 1)
            tracks[track_id] = track
            track_id += 1
    return tracks, truth, frames


def cluster_accuracy(assignments, truth, groups):
    confusion = np.zeros((groups, groups), np.int32)
    for track_id, value in assignments.items():
        confusion[truth[track_id], value[0]] += 1
    rows, columns = linear_sum_assignment(-confusion)
    return float(confusion[rows, columns].sum() / confusion.sum())


def solve_groups(num_moving_parts):
    tracks, truth, frames = synthetic_tracks(num_moving_parts)
    valid, edges, _scale, _modes = build_rigidity_graph(tracks, frames)
    assignments, _history = optimize_contrastive_groups(
        valid, edges, num_moving_parts + 1, iterations=250, learning_rate=0.05,
    )
    return tracks, truth, edges, assignments


def test_b_static_plus_moving_rigid_body():
    _tracks, truth, edges, assignments = solve_groups(1)
    same = [edge["rigidity_error"] for edge in edges if truth[edge["track_a"]] == truth[edge["track_b"]]]
    cross = [edge["rigidity_error"] for edge in edges if truth[edge["track_a"]] != truth[edge["track_b"]]]
    assert np.median(same) < np.median(cross)
    assert cluster_accuracy(assignments, truth, 2) == 1.0
    assert all(assignments[track_id][0] == 0 for track_id in assignments if truth[track_id] == 0)


def test_c_two_moving_parts_plus_static():
    _tracks, truth, _edges, assignments = solve_groups(2)
    assert len({value[0] for value in assignments.values()}) == 3
    assert cluster_accuracy(assignments, truth, 3) == 1.0


def test_d_point_dropout_leaves_unmatched_points():
    rng = np.random.default_rng(301)
    source = rng.normal(size=(36, 3))
    retained = np.sort(rng.choice(len(source), size=26, replace=False))
    rigid_rotation = rotation([0.3, 0.4, 0.7], 0.6)
    target = source[retained] @ rigid_rotation.T + np.asarray([0.4, -0.2, 0.1])
    order = rng.permutation(len(target))
    target_identity = retained[order]
    accuracy, accepted, diagnostics = matching_accuracy(source, target[order], target_identity)
    assert len(accepted) >= int(0.9 * len(retained))
    assert accuracy >= 0.98
    assert sum(row["i"] >= 0 and row["j"] < 0 for row in diagnostics) >= 8


def test_e_resampling_noise_remains_stable():
    rng = np.random.default_rng(401)
    source = rng.normal(size=(32, 3))
    rigid_rotation = rotation([0.6, -0.2, 0.3], 0.75)
    # Independent local surface samples are approximated by small metric
    # perturbations around every material point, plus 20% dropout.
    retained = np.sort(rng.choice(len(source), size=26, replace=False))
    target = source[retained] + rng.normal(scale=0.004, size=(len(retained), 3))
    target = target @ rigid_rotation.T + np.asarray([-0.3, 0.2, 0.15])
    order = rng.permutation(len(target))
    accuracy, accepted, _ = matching_accuracy(source, target[order], retained[order])
    assert len(accepted) >= int(0.75 * len(retained))
    assert accuracy >= 0.85


def test_f_static_ghosts_protect_reappearances_and_no_ghost_births_stay_uncertain():
    frames = list(range(11))
    tracks = {
        0: Track(0),  # Initial static point, then occluded.
        1: Track(1),  # Same point after a two-frame occlusion.
        2: Track(2),  # Same point after another occlusion.
        3: Track(3),  # Genuinely new point with no static ghost.
        4: Track(4),  # First-frame ambiguity must not be called dynamic.
        5: Track(5),  # Later-born but independently trusted STATIC anchor.
    }
    samples = {
        0: {0: [0.0, 0.0, 0.0], 1: [0.0, 0.0, 0.0], 2: [0.0, 0.0, 0.0], 3: [0.0, 0.0, 0.0]},
        1: {5: [0.002, 0.0, 0.0]},
        2: {7: [0.003, 0.0, 0.0]},
        3: {5: [1.0, 0.0, 0.0], 6: [1.1, 0.0, 0.0]},
        4: {0: [2.0, 0.0, 0.0]},
        5: {5: [3.0, 0.0, 0.0], 6: [3.0, 0.0, 0.0],
            7: [3.0, 0.0, 0.0], 8: [3.0, 0.0, 0.0]},
    }
    for track_id, observations in samples.items():
        for frame, point in observations.items():
            tracks[track_id].points[frame] = np.asarray(point)
            tracks[track_id].point_ids[frame] = track_id
    for track_id in (0, 5):
        tracks[track_id].temporal_matches = 8
        tracks[track_id].static_anchor_matches = 8
    rows, stats = build_static_anchor_ghost_priors(
        tracks, frames, spacing=0.1, radius_scale=0.125,
    )
    assert tracks[1].static_prior == 0.995
    assert tracks[2].static_prior == 0.995
    assert tracks[1].seed_type == "STATIC_GHOST"
    assert tracks[2].seed_type == "STATIC_GHOST"
    assert tracks[1].ghost_source_track_id == 0
    assert tracks[2].ghost_source_track_id == 1
    assert tracks[3].static_prior == 0.5
    assert tracks[3].static_prior_weight == 0.0
    apply_static_anchor_only_labels(tracks)
    assert tracks[1].label == "STATIC"
    assert tracks[2].label == "STATIC"
    assert tracks[3].label == "AMBIGUOUS"
    assert tracks[4].label == "AMBIGUOUS"
    assert tracks[5].label == "STATIC"
    assert [row["event"] for row in rows].count("ghost_prior_reactivated") == 2
    assert stats["ghost_reactivations"] == 2
    assert stats["births_without_ghost_left_unlabelled"] >= 1


def test_g_skip_frame_graph_bridges_one_missing_observation():
    rng = np.random.default_rng(501)
    canonical = rng.normal(size=(24, 3))
    transforms = [
        (rotation([0.2, 0.5, 0.8], 0.08 * frame), np.asarray([0.03 * frame, 0.0, 0.0]))
        for frame in range(3)
    ]
    clouds = {}
    for frame, (rigid_rotation, translation) in enumerate(transforms):
        xyz = canonical @ rigid_rotation.T + translation
        if frame == 1:
            xyz = xyz[1:]
        clouds[frame] = SimpleNamespace(xyz=xyz)
    tracks, rows, pair_stats = build_tracks(
        clouds, [0, 1, 2], {
            "num_rel_neighbors": 8,
            "hungarian_candidates": 16,
            "refine_iters": 5,
            "min_relational_support": 3,
            "unmatched_cost_scale": 2.5,
        },
        max_gap=1,
    )
    occluded = next(track for track in tracks.values() if track.point_ids.get(0) == 0)
    assert set(occluded.points) == {0, 2}
    assert occluded.point_ids[2] == 0
    assert pair_stats[(0, 2)]["bridged_tracks"] >= 1
    assert any(row["temporal_role"] == "skip_bridge" for row in rows)


def test_f2_short_stationary_occupancy_is_uncertain_not_a_static_seed():
    frames = list(range(60))
    track = Track(0)
    for frame in range(10):
        track.points[frame] = np.asarray([0.0, 0.0, 0.0])
        track.point_ids[frame] = 0
    track.temporal_matches = 9
    track.static_anchor_matches = 9
    tracks = {0: track}
    _rows, stats = build_static_anchor_ghost_priors(tracks, frames, spacing=0.1)
    assert track.stationary_occupancy_ambiguous
    assert track.seed_type == "none"
    assert track.static_prior == 0.5
    assert track.static_prior_weight == 0.0
    assert stats["minimum_trusted_static_track_length"] == 20


def test_h_uncertain_single_frame_tracks_are_learned_inside_optimizer():
    tracks, truth, frames = synthetic_tracks(1)
    static_core = next(track_id for track_id, group in truth.items() if group == 0)
    moving_core = next(track_id for track_id, group in truth.items() if group == 1)
    frame = 3
    uncertain_truth = {}
    for source_id, group in ((static_core, 0), (moving_core, 1)):
        track_id = max(tracks) + 1
        track = Track(track_id)
        track.points[frame] = tracks[source_id].points[frame] + np.asarray([0.002, 0.0, 0.0])
        track.point_ids[frame] = track_id
        tracks[track_id] = track
        uncertain_truth[track_id] = group
    valid, edges, _spacing, modes = build_rigidity_graph(tracks, frames)
    assert modes["uncertain_optimized_tracks"] == 2
    assert any(edge["edge_type"] == "uncertain_core_completion" for edge in edges)
    assignments, _history = optimize_contrastive_groups(
        valid, edges, 2, iterations=200, learning_rate=0.05,
        spatial_loss_weight=1.0, tracks=tracks,
    )
    assert all(assignments[track_id][0] == group for track_id, group in uncertain_truth.items())


def test_i_initial_point_cloud_labels_preserve_uncertainty():
    cloud = FrameCloud(
        frame=0,
        raw_xyz=np.empty((0, 3)),
        raw_camera_ids=np.empty(0, np.int32),
        xyz=np.asarray([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]]),
        num_raw_members=np.ones(3, np.int32),
        num_support_cameras=np.ones(3, np.int16),
        support_camera_ids=[(0,), (0,), (0,)],
        voxel_indices=np.zeros((3, 3), np.int64),
        spatial_spread=np.zeros(3),
        quality={},
    )
    tracks = {index: Track(index) for index in range(3)}
    for index, label in enumerate(("STATIC", "MOVING_1", "AMBIGUOUS")):
        tracks[index].point_ids[0] = index
        tracks[index].label = label
    dynamic = dynamic_labels_for_frame(cloud, tracks)
    assert dynamic.tolist() == [0.0, 1.0, 0.5]
    with tempfile.TemporaryDirectory() as folder:
        path = Path(folder) / "labels.ply"
        write_ply(path, cloud.xyz, {"dynamic": dynamic})
        header = path.read_bytes().split(b"end_header\n", 1)[0]
        assert b"property float dynamic" in header


def test_j_uncertain_knn_targets_follow_temporal_pair_rigidity():
    tracks, truth, frames = synthetic_tracks(1)
    uncertain = next(track_id for track_id, group in truth.items() if group == 1)
    tracks[uncertain].stationary_occupancy_ambiguous = True
    _valid, edges, _spacing, modes = build_rigidity_graph(
        tracks, frames, uncertain_knn_neighbors=16,
    )
    neighbors = [
        edge for edge in edges
        if edge["edge_type"] == "temporal_rigidity"
        and uncertain in (edge["track_a"], edge["track_b"])
        and edge["uncertain_neighbor_weight"] > 0
    ]
    same, different = [], []
    for edge in neighbors:
        other = edge["track_b"] if edge["track_a"] == uncertain else edge["track_a"]
        (same if truth[other] == truth[uncertain] else different).append(
            edge["uncertain_neighbor_target"]
        )
    assert same and different
    assert np.median(same) > np.median(different)
    assert modes["uncertain_neighbor_edges"] >= len(neighbors)


def test_k_singleton_learns_only_from_certain_local_teachers():
    tracks, truth, frames = synthetic_tracks(1)
    assignments = {
        track_id: (group, 0.99, np.asarray([0.99, 0.01]) if group == 0 else np.asarray([0.01, 0.99]))
        for track_id, group in truth.items()
    }
    moving_source = next(track_id for track_id, group in truth.items() if group == 1)
    singleton_id = max(tracks) + 1
    singleton = Track(singleton_id)
    singleton.points[3] = tracks[moving_source].points[3] + np.asarray([0.001, 0.0, 0.0])
    singleton.point_ids[3] = singleton_id
    tracks[singleton_id] = singleton
    stats = solve_short_fragments_with_certain_knn(
        tracks, assignments, frames, spacing=0.1, neighbors=3,
    )
    assert assignments[singleton_id][0] == 1
    assert assignments[singleton_id][1] > 0.9
    assert tracks[singleton_id].knn_teacher_count >= 3
    assert stats["solved_from_certain_knn"] == 1
    assert stats["previously_unoptimized_initialized_uncertain"] == 1


def test_l_final_label_hungarian_recovers_fragmented_trajectories():
    rng = np.random.default_rng(707)
    source = rng.normal(size=(24, 3))
    target = source @ rotation([0.2, 0.3, 0.8], 0.1).T + np.asarray([0.03, 0.0, 0.0])
    clouds = {0: SimpleNamespace(xyz=source), 1: SimpleNamespace(xyz=target)}
    tracks, assignments = {}, {}
    for point_id in range(24):
        first = Track(point_id)
        first.points[0] = source[point_id]
        first.point_ids[0] = point_id
        second = Track(24 + point_id)
        second.points[1] = target[point_id]
        second.point_ids[1] = point_id
        tracks[point_id], tracks[24 + point_id] = first, second
        probability = np.asarray([0.01, 0.99])
        assignments[point_id] = (1, 0.99, probability.copy())
        assignments[24 + point_id] = (1, 0.99, probability.copy())
    rows, stats = final_temporal_hungarian_coherence(
        clouds, [0, 1], tracks, assignments,
        {
            "num_rel_neighbors": 8,
            "hungarian_candidates": 16,
            "refine_iters": 5,
            "min_relational_support": 3,
            "unmatched_cost_scale": 2.5,
            "compatibility_relative_threshold": 0.03,
        },
        max_gap=0, label_cost_weight=1.0, teacher_confidence=0.9,
    )
    assert stats["merged_fragment_links"] == 24
    assert stats["recovered_trajectories"] == 24
    assert all(row["merged"] for row in rows)
    assert tracks[0].temporal_trajectory_id == tracks[24].temporal_trajectory_id


def test_k2_short_static_ghosts_are_strong_local_teachers():
    frames = list(range(7))
    tracks, assignments = {}, {}
    for track_id, x in enumerate((-0.01, 0.0, 0.01)):
        track = Track(track_id)
        track.points[3] = np.asarray([x, 0.0, 0.0])
        track.point_ids[3] = track_id
        track.seed_type = "STATIC_GHOST"
        track.seed_strength = 1.0
        tracks[track_id] = track
        assignments[track_id] = (0, 0.99, np.asarray([0.99, 0.01]))
    student_id = 3
    student = Track(student_id)
    student.points[3] = np.asarray([0.002, 0.0, 0.0])
    student.point_ids[3] = student_id
    tracks[student_id] = student
    stats = solve_short_fragments_with_certain_knn(
        tracks, assignments, frames, spacing=0.1, neighbors=3,
    )
    assert assignments[student_id][0] == 0
    assert assignments[student_id][1] > 0.9
    assert stats["ghost_teacher_observations"] >= 3


def residual_cloud(frame, xyz):
    xyz = np.asarray(xyz, np.float64)
    return ResidualCloud(
        frame, xyz, np.ones(len(xyz), np.int32),
        np.arange(len(xyz), dtype=np.int32), {"median_spacing": 0.01},
    )


def test_m_static_is_removed_before_full_resolution_residual_matching():
    raw = np.asarray([
        [0.001, 0.001, 0.001], [0.002, 0.001, 0.001],
        [0.051, 0.001, 0.001], [0.052, 0.001, 0.001],
    ])
    camera_ids = np.asarray([0, 1, 0, 1], np.int16)
    xyz, members, supports, support_ids, voxels, spreads = voxel_reduce(raw, camera_ids, 0.03)
    cloud = FrameCloud(
        0, raw, camera_ids, xyz, members, supports, support_ids, voxels, spreads,
        {"voxel_size": 0.03},
    )
    tracks = {0: Track(0), 1: Track(1)}
    for point_id, track in tracks.items():
        track.points[0] = xyz[point_id]
        track.point_ids[0] = point_id
    tracks[0].seed_type = "STATIC_ANCHOR"
    tracks[0].seed_strength = 1.0
    with tempfile.TemporaryDirectory() as folder:
        residuals = build_full_resolution_residuals(
            {0: cloud}, [0], tracks, Path(folder), fine_tolerance=0.005,
        )
    residual = residuals[0]
    assert residual.quality["removed_static_points"] == 2
    assert residual.quality["raw_residual_points"] == 2
    assert np.all(residual.coarse_point_ids == 1)


def test_n_residual_graph_matching_handles_large_rigid_rotation_and_translation():
    rng = np.random.default_rng(811)
    source = rng.normal(size=(40, 3))
    transformed = source @ rotation([0.3, 0.7, -0.2], 1.2).T + np.asarray([2.0, -1.0, 0.4])
    graph_a = build_residual_graph(residual_cloud(0, source), 8, 8, 8)
    graph_b = build_residual_graph(residual_cloud(1, transformed), 8, 8, 8)
    rows, stats = match_residual_graph_pair(graph_a, graph_b)
    assert stats["accepted_matches"] == len(source)
    assert all(row["source_id"] == row["target_id"] for row in rows)


def test_o_residual_dropout_is_partial_and_not_forced():
    rng = np.random.default_rng(812)
    source = rng.normal(size=(48, 3))
    retained = np.sort(rng.choice(len(source), size=34, replace=False))
    target = source[retained] + rng.normal(scale=0.002, size=(len(retained), 3))
    rows, stats = match_residual_graph_pair(
        build_residual_graph(residual_cloud(0, source), 8, 8, 8),
        build_residual_graph(residual_cloud(1, target), 8, 8, 8),
    )
    correct = sum(retained[row["target_id"]] == row["source_id"] for row in rows)
    assert stats["unmatched_source"] >= 10
    assert correct >= int(0.55 * len(retained))


def test_p_dynamic_anchor_requires_persistent_nonzero_motion_and_se3():
    rng = np.random.default_rng(813)
    canonical = rng.normal(size=(32, 3))
    residuals = {}
    for frame in range(3):
        xyz = canonical @ rotation([0.2, 0.4, 0.8], 0.12 * frame).T
        xyz += np.asarray([0.04 * frame, -0.01 * frame, 0.0])
        residuals[frame] = residual_cloud(frame, xyz)
    graphs = {frame: build_residual_graph(cloud, 8, 8, 8) for frame, cloud in residuals.items()}
    pair_rows = {}
    for pair in ((0, 1), (1, 2), (0, 2)):
        pair_rows[pair] = match_residual_graph_pair(graphs[pair[0]], graphs[pair[1]])[0]
    annotate_skip_consistency(pair_rows, [0, 1, 2], 1)
    tracks = build_residual_tracks(residuals, [0, 1, 2], pair_rows)
    for pair, rows in pair_rows.items():
        validate_se3_models(rows, residuals[pair[0]], residuals[pair[1]], 1e-5, seed=17)
    dynamic = validate_dynamic_anchors(tracks, pair_rows, 1e-3, 3, 0.08)
    assert len(dynamic) >= int(0.9 * len(canonical))


def test_q_no_motion_residual_is_not_promoted_dynamic():
    track = Track(0)
    for frame in range(3):
        track.points[frame] = np.asarray([0.1, 0.2, 0.3])
        track.point_ids[frame] = 0
    rows = [{
        "track_id": 0, "compatibility_support": 12,
        "edge_deformation_median": 0.0, "edge_deformation_p90": 0.0,
        "motion_magnitude": 0.0, "skip_consistency": "confirmed",
        "se3_inlier": True, "dynamic_anchor_accepted": False,
    } for _ in range(2)]
    dynamic = validate_dynamic_anchors({0: track}, {(0, 1): rows}, 1e-3, 3, 0.08)
    assert not dynamic
    assert track.dynamic_probability == 0.5


def test_r_binary_optimizer_uses_temporal_pair_targets_and_keeps_uncertainty_trainable():
    static = Track(10)
    missed_static = Track(1)
    moving = Track(2)
    nodes = {-1: static, 1: missed_static, 2: moving}
    edges = [
        {"track_a": -1, "track_b": 1, "q_same": 1.0, "weight": 1.0},
        {"track_a": -1, "track_b": 2, "q_same": 0.0, "weight": 1.0},
    ]
    probabilities, history = optimize_binary_dynamic_probabilities(
        nodes, edges, set(), outer_iteration=0, iterations=150,
        learning_rate=0.08, static_threshold=0.25, dynamic_threshold=0.75,
        ignored_margin_edges=3,
    )
    assert probabilities[-1] == 0.0
    assert probabilities[1] < 0.25
    assert probabilities[2] > 0.75
    assert history[-1]["mean_positive_same_probability"] > 0.9
    assert history[-1]["mean_negative_same_probability"] < 0.1


def test_s_adaptive_coreset_caps_graph_without_losing_scene_coverage():
    x, y = np.meshgrid(np.linspace(-1.0, 1.0, 120), np.linspace(-0.5, 0.5, 80))
    xyz = np.stack((x.ravel(), y.ravel(), 0.03 * np.sin(5.0 * x).ravel()), axis=1)
    members = np.ones(len(xyz), np.int32)
    coarse_ids = np.arange(len(xyz), dtype=np.int32)
    reduced, reduced_members, _provenance, cell_size = adaptive_geometric_coreset(
        xyz, members, coarse_ids, target_points=512,
    )
    assert 400 <= len(reduced) <= 512
    assert reduced_members.sum() == len(xyz)
    assert np.ptp(reduced[:, 0]) > 1.9
    assert np.ptp(reduced[:, 1]) > 0.9
    assert cell_size > 0.0


def test_t_no_ghost_or_stale_dynamic_seed_ever_emits_dynamic():
    tracks = {index: Track(index) for index in range(4)}
    tracks[0].seed_type = "STATIC_ANCHOR"
    tracks[0].seed_strength = 1.0
    tracks[1].seed_type = "STATIC_GHOST"
    tracks[1].seed_strength = 0.9
    tracks[2].seed_type = "DYNAMIC_GRAPH_SE3"
    tracks[2].seed_strength = 1.0
    tracks[3].seed_type = "DYNAMIC_BIRTH_NO_GHOST"
    tracks[3].seed_strength = 1.0
    apply_static_anchor_only_labels(tracks)
    assert tracks[0].label == "STATIC" and tracks[0].dynamic_probability == 0.0
    assert tracks[1].label == "STATIC" and tracks[1].dynamic_probability == 0.0
    assert tracks[2].label == "AMBIGUOUS" and tracks[2].dynamic_probability == 0.5
    assert tracks[3].label == "AMBIGUOUS" and tracks[3].dynamic_probability == 0.5
    assert not any(track.label.startswith("MOVING_") for track in tracks.values())


def test_u_static_recovery_is_bidirectional_and_never_creates_dynamic():
    tracks = {index: Track(index) for index in range(4)}
    tracks[0].seed_type = "STATIC_ANCHOR"
    tracks[0].seed_strength = 1.0
    for frame in range(10, 30):
        tracks[0].points[frame] = np.asarray([0.0, 0.0, 0.0])
    for track_id, position, frames in (
        (1, [0.004, 0.0, 0.0], range(0, 5)),
        (2, [0.004, 0.0, 0.0], range(12, 17)),
        (3, [0.03, 0.0, 0.0], range(0, 5)),
    ):
        tracks[track_id].stationary_occupancy_ambiguous = True
        tracks[track_id].temporal_matches = 4
        tracks[track_id].static_anchor_matches = 4
        for frame in frames:
            tracks[track_id].points[frame] = np.asarray(position)
    rows, stats = recover_static_fragments_from_anchors(tracks, spacing=0.02)
    apply_static_anchor_only_labels(tracks)
    assert tracks[1].label == "STATIC"  # Evidence propagates backward in time.
    assert tracks[2].label == "AMBIGUOUS"  # Overlapping tracks are not fragments.
    assert tracks[3].label == "AMBIGUOUS"  # Spatially incompatible remains uncertain.
    assert stats["recovered_static_tracks"] == 1
    assert stats["backward_recoveries"] == 1
    assert len(rows) == 1
    assert not any(track.label.startswith("MOVING_") for track in tracks.values())


def test_v_temporal_reappearance_recovers_static_without_a_dynamic_complement():
    frames = list(range(12))
    tracks = {track_id: Track(track_id) for track_id in range(1, 21)}
    neighbor_xyz = np.asarray([
        [0.02 * math.cos(angle), 0.02 * math.sin(angle), 0.0]
        for angle in np.linspace(0.0, 2.0 * math.pi, 8, endpoint=False)
    ])
    for neighbor_id in range(1, 9):
        tracks[neighbor_id].seed_type = "STATIC_ANCHOR"
        tracks[neighbor_id].seed_strength = 1.0

    clouds = {}
    for frame in frames:
        reappearance_id = 9 + frame
        xyz = np.concatenate((np.zeros((1, 3)), neighbor_xyz), axis=0)
        clouds[frame] = FrameCloud(
            frame, np.empty((0, 3)), np.empty(0, np.int32), xyz,
            np.ones(9, np.int32), np.ones(9, np.int16), [(0,)] * 9,
            np.zeros((9, 3), np.int64), np.zeros(9), {},
        )
        tracks[reappearance_id].points[frame] = xyz[0]
        tracks[reappearance_id].point_ids[frame] = 0
        for point_id, neighbor_id in enumerate(range(1, 9), start=1):
            tracks[neighbor_id].points[frame] = xyz[point_id]
            tracks[neighbor_id].point_ids[frame] = point_id

    rows, stats = recover_static_from_temporal_reappearance(
        clouds, frames, tracks, spacing=0.02,
    )
    apply_static_anchor_only_labels(tracks)
    assert tracks[9].label == "STATIC"
    assert stats["recovered_static_tracks"] == len(frames)
    assert len(rows) == len(frames)
    assert not any(track.label.startswith("MOVING_") for track in tracks.values())


if __name__ == "__main__":
    tests = [
        test_a_one_rigid_body_se3,
        test_a2_preliminary_icp_anchors_only_stationary_points,
        test_b_static_plus_moving_rigid_body,
        test_c_two_moving_parts_plus_static,
        test_d_point_dropout_leaves_unmatched_points,
        test_e_resampling_noise_remains_stable,
        test_f_static_ghosts_protect_reappearances_and_no_ghost_births_stay_uncertain,
        test_f2_short_stationary_occupancy_is_uncertain_not_a_static_seed,
        test_g_skip_frame_graph_bridges_one_missing_observation,
        test_h_uncertain_single_frame_tracks_are_learned_inside_optimizer,
        test_i_initial_point_cloud_labels_preserve_uncertainty,
        test_j_uncertain_knn_targets_follow_temporal_pair_rigidity,
        test_k_singleton_learns_only_from_certain_local_teachers,
        test_k2_short_static_ghosts_are_strong_local_teachers,
        test_l_final_label_hungarian_recovers_fragmented_trajectories,
        test_m_static_is_removed_before_full_resolution_residual_matching,
        test_n_residual_graph_matching_handles_large_rigid_rotation_and_translation,
        test_o_residual_dropout_is_partial_and_not_forced,
        test_p_dynamic_anchor_requires_persistent_nonzero_motion_and_se3,
        test_q_no_motion_residual_is_not_promoted_dynamic,
        test_r_binary_optimizer_uses_temporal_pair_targets_and_keeps_uncertainty_trainable,
        test_s_adaptive_coreset_caps_graph_without_losing_scene_coverage,
        test_t_no_ghost_or_stale_dynamic_seed_ever_emits_dynamic,
        test_u_static_recovery_is_bidirectional_and_never_creates_dynamic,
        test_v_temporal_reappearance_recovers_static_without_a_dynamic_complement,
    ]
    for test in tests:
        test()
        print(f"[PASS] {test.__name__}")
