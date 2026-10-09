from __future__ import annotations

from pathlib import Path
import sys

import torch

REPO = Path(__file__).resolve().parent.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from rigid_parts.losses import build_sequence_state, pairwise_distance_loss, twist_assignment_loss
from rigid_parts.moments import (
    compute_global_com,
    compute_global_inertia,
    compute_soft_com,
    compute_soft_inertia,
    inertia_invariants,
    parallel_axis_global_inertia,
)


def _rz(angle: float) -> torch.Tensor:
    c, s = torch.cos(torch.tensor(angle)), torch.sin(torch.tensor(angle))
    return torch.tensor([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def _hard(labels: torch.Tensor, k1: int) -> torch.Tensor:
    return torch.nn.functional.one_hot(labels.long(), num_classes=k1).float()


def test_translation_preserves_inertia_and_distances():
    xyz = torch.randn(32, 3)
    p = torch.ones(32, 1)
    t = torch.tensor([0.4, -0.2, 0.7])
    j0, c0 = compute_soft_inertia(xyz, p)
    j1, c1 = compute_soft_inertia(xyz + t, p)
    assert torch.allclose(c1[0] - c0[0], t, atol=1.0e-5)
    assert torch.allclose(j0, j1, atol=1.0e-5)
    d0 = torch.cdist(xyz, xyz)
    d1 = torch.cdist(xyz + t, xyz + t)
    assert torch.allclose(d0, d1, atol=2.0e-3)


def test_rotation_transports_com_and_inertia_invariants():
    xyz = torch.randn(40, 3)
    p = torch.ones(40, 1)
    r = _rz(0.6)
    moved = xyz @ r.T
    j0, c0 = compute_soft_inertia(xyz, p)
    j1, c1 = compute_soft_inertia(moved, p)
    assert torch.allclose(c1[0], r @ c0[0], atol=1.0e-5)
    assert torch.allclose(j1[0], r @ j0[0] @ r.T, atol=1.0e-5)
    i10, i20 = inertia_invariants(j0)
    i11, i21 = inertia_invariants(j1)
    assert torch.allclose(i10, i11, atol=1.0e-5)
    assert torch.allclose(i20, i21, atol=1.0e-5)


def test_two_rigid_bodies_are_worse_when_merged():
    a = torch.randn(12, 3) * 0.1 + torch.tensor([-1.0, 0.0, 0.0])
    b = torch.randn(12, 3) * 0.1 + torch.tensor([1.0, 0.0, 0.0])
    xyz = torch.cat((a, b))
    moved = torch.cat((a + torch.tensor([0.1, 0.0, 0.0]), b @ _rz(0.8).T))
    correct = _hard(torch.tensor([1] * 12 + [2] * 12), 3)
    merged = _hard(torch.tensor([1] * 24), 3)
    src, dst = torch.triu_indices(24, 24, offset=1)
    d0 = torch.linalg.norm(xyz[src] - xyz[dst], dim=-1)
    correct_loss = pairwise_distance_loss(torch.stack((xyz, moved)), src, dst, d0, correct)
    merged_loss = pairwise_distance_loss(torch.stack((xyz, moved)), src, dst, d0, merged)
    assert float(merged_loss) > float(correct_loss) + 1.0e-4


def test_static_component_is_identity_velocity_model():
    xyz = torch.randn(20, 3)
    positions = torch.stack((xyz, xyz.clone()))
    velocities = torch.zeros_like(positions)
    static_prob = _hard(torch.zeros(20, dtype=torch.long), 2)
    moving_prob = _hard(torch.ones(20, dtype=torch.long), 2)
    times = torch.tensor([0.0, 1.0])
    static_state = build_sequence_state(xyz, positions, velocities, static_prob, times)
    moving_state = build_sequence_state(xyz, positions, velocities, moving_prob, times)
    assert float(twist_assignment_loss(positions, velocities, static_state)) <= float(twist_assignment_loss(positions, velocities, moving_state)) + 1.0e-6
    assert torch.allclose(static_state.rotation[:, 0], torch.eye(3).expand(2, 3, 3))
    assert torch.allclose(static_state.translation[:, 0], torch.zeros(2, 3))


def test_parallel_axis_matches_direct_global_inertia():
    xyz = torch.randn(30, 3)
    labels = torch.tensor([0] * 10 + [1] * 20)
    p = _hard(labels, 2)
    j, c = compute_soft_inertia(xyz, p, normalize=True)
    mass = p.sum(dim=0)
    global_c = compute_global_com(xyz)
    direct = compute_global_inertia(xyz, global_c, normalize=True)
    reconstructed = parallel_axis_global_inertia(j, mass, c, global_c, normalized_part_inertia=True, normalize=True)
    assert torch.allclose(direct, reconstructed, atol=1.0e-5)


def main():
    test_translation_preserves_inertia_and_distances()
    test_rotation_transports_com_and_inertia_invariants()
    test_two_rigid_bodies_are_worse_when_merged()
    test_static_component_is_identity_velocity_model()
    test_parallel_axis_matches_direct_global_inertia()
    print("PASS rigid_parts_geometry")


if __name__ == "__main__":
    main()
