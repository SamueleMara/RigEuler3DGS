from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
import sys
import tempfile

import torch

REPO = Path(__file__).resolve().parent.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from rigid_parts.endpoint import compute_counterfactual_endpoint_unary
from rigid_parts.losses import build_sequence_state, inertia_transport_loss
from rigid_parts.pipeline import run_corrected_rigid_part_discovery_from_tracks


def _rz(angle: float) -> torch.Tensor:
    c, s = torch.cos(torch.tensor(angle)), torch.sin(torch.tensor(angle))
    return torch.tensor([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def _args() -> SimpleNamespace:
    return SimpleNamespace(
        rigid_part_max_frames=6,
        rigid_part_label_temperature=1.2,
        rigid_part_label_temperature_final=0.7,
        rigid_part_knn=6,
        rigid_part_refine_iters=80,
        rigid_part_lr=0.08,
        rigid_part_min_fraction=0.02,
        rigid_part_endpoint_chunk=128,
        transported_html_max_points=200,
        wandb_log_interval=1000,
        lambda_rigid_trajectory=5.0,
        lambda_rigid_twist=1.0,
        lambda_rigid_graph=0.2,
        lambda_rigid_pair=0.2,
        lambda_rigid_distance_rate=0.1,
        lambda_rigid_com=0.05,
        lambda_rigid_inertia=0.05,
        lambda_rigid_inertia_invariants=0.01,
        lambda_endpoint_geometry=2.0,
        lambda_rigid_occupancy=0.0,
        lambda_rigid_entropy=0.0,
    )


def _scene():
    torch.manual_seed(3)
    static = torch.randn(24, 3) * 0.05 + torch.tensor([-0.8, 0.0, 0.0])
    moving = torch.randn(24, 3) * 0.05 + torch.tensor([0.8, 0.0, 0.0])
    xyz = torch.cat((static, moving), dim=0)
    times = torch.linspace(0.0, 1.0, 6)
    frames = []
    for t in times:
        r = _rz(float(0.8 * t))
        trans = torch.tensor([0.0, 0.35 * float(t), 0.0])
        frames.append(torch.cat((static, moving @ r.T + trans), dim=0))
    positions = torch.stack(frames)
    velocities = torch.zeros_like(positions)
    velocities[:-1] = (positions[1:] - positions[:-1]) / torch.diff(times)[:, None, None]
    velocities[-1] = velocities[-2]
    final_xyz = positions[-1].clone()
    true = torch.tensor([0] * static.shape[0] + [1] * moving.shape[0])
    return xyz, positions, velocities, times, final_xyz, true


def _wrong_init(true: torch.Tensor) -> torch.Tensor:
    prob = torch.zeros((true.numel(), 2))
    prob[:, 0] = 0.85
    prob[:, 1] = 0.15
    prob[true == 1, 0] = 0.15
    prob[true == 1, 1] = 0.85
    prob[24:30] = torch.tensor([0.85, 0.15])  # moving wrongly static
    prob[0:6] = torch.tensor([0.15, 0.85])    # static wrongly moving
    return prob


def test_wrong_static_and_moving_labels_are_corrected_from_raw_tracks():
    xyz, positions, velocities, times, final_xyz, true = _scene()
    rotations = torch.zeros((xyz.shape[0], 4))
    rotations[:, 0] = 1.0
    with tempfile.TemporaryDirectory() as tmp:
        result = run_corrected_rigid_part_discovery_from_tracks(
            canonical_xyz=xyz,
            canonical_rotation=rotations,
            raw_tracks=positions,
            raw_velocities=velocities,
            raw_times=times,
            init_prob=_wrong_init(true),
            args=_args(),
            output=Path(tmp),
            final_xyz=final_xyz,
        )
    labels = result["labels"].long()
    assert int(((labels == 1) & (true == 1)).sum()) >= 20
    assert int(((labels == 0) & (true == 0)).sum()) >= 20
    assert int((labels != _wrong_init(true).argmax(-1)).sum()) >= 6


def test_inertia_raw_partition_beats_mixed_partition_and_replay_is_tautological():
    xyz, positions, velocities, times, _final_xyz, true = _scene()
    correct = torch.nn.functional.one_hot(true, 2).float()
    mixed_labels = true.clone()
    mixed_labels[:12] = 1
    mixed_labels[24:36] = 0
    mixed = torch.nn.functional.one_hot(mixed_labels, 2).float()
    correct_state = build_sequence_state(xyz, positions, velocities, correct, times)
    mixed_state = build_sequence_state(xyz, positions, velocities, mixed, times)
    assert float(inertia_transport_loss(mixed_state)) > float(inertia_transport_loss(correct_state)) + 1.0e-4

    # A replay generated from wrong labels can still look rigid under those same
    # wrong labels. This documents why replay cannot be discovery evidence.
    replay_state = build_sequence_state(xyz, positions, velocities, mixed, times)
    replay = []
    for frame in range(positions.shape[0]):
        pred = []
        for part in range(2):
            if part == 0:
                pred.append(xyz)
            else:
                pred.append(xyz @ replay_state.rotation[frame, part].T + replay_state.translation[frame, part])
        replay.append(torch.stack(pred, dim=1)[torch.arange(xyz.shape[0]), mixed_labels])
    tautology_state = build_sequence_state(xyz, torch.stack(replay), velocities * 0.0, mixed, times)
    assert float(inertia_transport_loss(tautology_state)) < float(inertia_transport_loss(mixed_state))


def test_counterfactual_endpoint_unary_prefers_correct_class():
    xyz, positions, _velocities, _times, final_xyz, true = _scene()
    r = _rz(0.8)
    t = torch.tensor([[0.0, 0.0, 0.0], [0.0, 0.35, 0.0]])
    rotations = torch.stack((torch.eye(3), r))
    energy = compute_counterfactual_endpoint_unary(xyz, rotations, t, final_xyz, chunk_size=128)
    assert float((energy[true == 1, 1] < energy[true == 1, 0]).float().mean()) > 0.8
    assert float((energy[true == 0, 0] <= energy[true == 0, 1]).float().mean()) > 0.8


def main():
    test_wrong_static_and_moving_labels_are_corrected_from_raw_tracks()
    test_inertia_raw_partition_beats_mixed_partition_and_replay_is_tautological()
    test_counterfactual_endpoint_unary_prefers_correct_class()
    print("PASS corrected_rigid_part_discovery")


if __name__ == "__main__":
    main()
