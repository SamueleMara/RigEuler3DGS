import math
import sys
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment

sys.path.insert(0, str(Path(__file__).parent))

from rigid_distance_multiframe_colmap import (
    MVPoint,
    PointObservation,
    TemporalTrack3D,
    build_rigidity_edges,
    label_tracks_rigid_parts,
    temporal_match_frame_pair,
)


def rotation(axis, angle):
    axis = np.asarray(axis, np.float64)
    axis /= np.linalg.norm(axis)
    x, y, z = axis
    K = np.array([[0, -z, y], [z, 0, -x], [-y, x, 0]])
    return np.eye(3) + math.sin(angle) * K + (1 - math.cos(angle)) * (K @ K)


def make_points(xyz, frame, descriptors, identities):
    observations = [
        PointObservation(0, 0, "cam_000", frame, np.zeros(2)),
        PointObservation(1, 0, "cam_001", frame, np.zeros(2)),
    ]
    return [
        MVPoint(int(identity), frame, np.asarray(point), 0.1, observations, descriptor)
        for point, descriptor, identity in zip(xyz, descriptors, identities)
    ]


def match_accuracy(source, target):
    matches, diagnostics, _ = temporal_match_frame_pair(source, target)
    correct = sum(source[m["i"]].point3d_id == target[m["j"]].point3d_id for m in matches)
    return correct / max(len(matches), 1), matches, diagnostics


def test_rigid_transform_invariance():
    transforms = [
        (np.eye(3), np.array([1.2, -0.7, 0.3])),
        (rotation([0.2, 0.7, 0.4], 1.1), np.zeros(3)),
        (rotation([0.8, -0.1, 0.5], 0.8), np.array([-0.4, 0.9, 0.2])),
    ]
    for transform_index, (R, t) in enumerate(transforms):
        rng = np.random.default_rng(4 + transform_index)
        xyz = rng.normal(size=(24, 3))
        descriptors = np.eye(24, dtype=np.float32)
        identities = np.arange(24)
        target_xyz = xyz @ R.T + t + rng.normal(scale=1e-5, size=xyz.shape)
        order = rng.permutation(len(xyz))
        source = make_points(xyz, 0, descriptors, identities)
        target = make_points(target_xyz[order], 1, descriptors[order], identities[order])
        accuracy, matches, _ = match_accuracy(source, target)
        assert len(matches) == len(source)
        assert accuracy == 1.0


def synthetic_tracks(num_moving_parts):
    rng = np.random.default_rng(11 + num_moving_parts)
    frames = list(range(6))
    tracks = {}
    truth = {}
    tid = 0
    for group in range(num_moving_parts + 1):
        base = rng.normal(scale=0.25, size=(9, 3)) + np.array([group * 1.7, group * 0.2, 0.0])
        for local_index, point in enumerate(base):
            track = TemporalTrack3D(tid)
            for frame in frames:
                if group == 0:
                    transformed = point
                else:
                    R = rotation([0.2 * group, 0.3, 1.0], 0.12 * group * frame)
                    translation = np.array([0.08 * group * frame, -0.035 * frame, 0.025 * group * frame])
                    transformed = point @ R.T + translation
                track.points[frame] = MVPoint(tid, frame, transformed.copy(), 0.1)
            tracks[tid] = track
            truth[tid] = group
            tid += 1
    return tracks, truth, frames


def cluster_accuracy(assignments, truth, num_parts):
    matrix = np.zeros((num_parts, num_parts), np.int32)
    for tid, (predicted, _) in assignments.items():
        matrix[truth[tid], predicted] += 1
    rows, cols = linear_sum_assignment(-matrix)
    return float(matrix[rows, cols].sum() / matrix.sum())


def solve_synthetic_parts(num_moving_parts):
    tracks, truth, frames = synthetic_tracks(num_moving_parts)
    valid, edges, scale, modes = build_rigidity_edges(tracks, frames)
    result = label_tracks_rigid_parts(
        tracks, valid, edges, num_moving_parts, frames, scale, modes
    )
    assignments = result[3]
    return tracks, truth, edges, assignments


def test_two_rigid_bodies_match_and_cluster():
    tracks, truth, edges, assignments = solve_synthetic_parts(1)
    same = [e["rigidity_error"] for e in edges if truth[e["track_a"]] == truth[e["track_b"]]]
    cross = [e["rigidity_error"] for e in edges if truth[e["track_a"]] != truth[e["track_b"]]]
    assert np.median(same) < np.median(cross)
    assert cluster_accuracy(assignments, truth, 2) == 1.0
    assert all(tracks[tid].label == "STATIC" for tid, group in truth.items() if group == 0)


def test_three_rigid_groups_are_recovered_before_static_naming():
    tracks, truth, _, assignments = solve_synthetic_parts(2)
    assert len({cluster for cluster, _ in assignments.values()}) == 3
    assert cluster_accuracy(assignments, truth, 3) == 1.0
    assert all(tracks[tid].label == "STATIC" for tid, group in truth.items() if group == 0)


def test_point_dropout_uses_unmatched_assignments():
    rng = np.random.default_rng(31)
    xyz = rng.normal(size=(28, 3))
    descriptors = np.eye(28, dtype=np.float32)
    retained = np.sort(rng.choice(28, size=20, replace=False))
    order = rng.permutation(len(retained))
    R = rotation([0.3, 0.4, 0.7], 0.6)
    source = make_points(xyz, 0, descriptors, np.arange(28))
    target = make_points(
        (xyz[retained] @ R.T + np.array([0.4, -0.2, 0.1]))[order],
        1, descriptors[retained][order], retained[order],
    )
    accuracy, matches, diagnostics = match_accuracy(source, target)
    assert accuracy == 1.0
    assert len(matches) >= int(0.9 * len(retained))
    assert any(d["i"] >= 0 and d["j"] < 0 for d in diagnostics)


def test_descriptor_corruption_does_not_override_geometry():
    rng = np.random.default_rng(44)
    xyz = rng.normal(size=(30, 3))
    descriptors = np.eye(30, dtype=np.float32)
    R = rotation([0.5, -0.2, 0.3], 0.9)
    target_descriptors = descriptors.copy()
    corrupt = rng.choice(30, size=12, replace=False)
    target_descriptors[corrupt] = target_descriptors[rng.permutation(corrupt)]
    order = rng.permutation(30)
    source = make_points(xyz, 0, descriptors, np.arange(30))
    target = make_points(
        (xyz @ R.T + np.array([0.6, 0.2, -0.4]))[order],
        1, target_descriptors[order], np.arange(30)[order],
    )
    accuracy, matches, _ = match_accuracy(source, target)
    assert len(matches) >= 27
    assert accuracy >= 0.90


if __name__ == "__main__":
    tests = [
        test_rigid_transform_invariance,
        test_two_rigid_bodies_match_and_cluster,
        test_three_rigid_groups_are_recovered_before_static_naming,
        test_point_dropout_uses_unmatched_assignments,
        test_descriptor_corruption_does_not_override_geometry,
    ]
    for test in tests:
        test()
        print(f"[PASS] {test.__name__}")
