"""Synthetic checks for relational graph transport (Stage A).

The property under test throughout is rigid invariance: an arbitrary R,t
applied to a cloud must leave every descriptor and every relational cost
unchanged, because that invariance is the entire reason this coupling can
find a region that moved a long way.

Run:
    PYTHONPATH=. python tests/test_relational_graph_transport.py
"""

from __future__ import annotations

import math

import torch

from utils.relational_coupling import (
    build_bidirectional_candidate_union,
    build_descriptor_candidates,
    build_relational_coupling,
    compute_sparse_relational_cost,
    compute_transport_coverage,
    compute_transport_entropy,
    initialize_soft_partial_transport,
)
from utils.relational_graph import (
    build_relational_graph,
    compute_covariance_invariants,
    compute_distance_spectrum_descriptor,
    compute_local_pca_degeneracy,
)


def _block(n: int = 5, spacing: float = 0.02, origin: tuple[float, float, float] = (0.0, 0.0, 0.0),
           jitter: float = 0.25, seed: int = 0) -> torch.Tensor:
    """An irregular point cloud.

    Deliberately NOT a perfect lattice: a regular grid has genuinely
    degenerate distance spectra (every corner is indistinguishable from every
    other corner), so matching it would be testing an impossibility rather
    than the method.  Real Gaussian clouds are irregular.
    """
    axis = torch.arange(n, dtype=torch.float32) * spacing
    gx, gy, gz = torch.meshgrid(axis, axis, axis, indexing="ij")
    points = torch.stack((gx.reshape(-1), gy.reshape(-1), gz.reshape(-1)), dim=-1)
    generator = torch.Generator().manual_seed(int(seed))
    points = points + jitter * spacing * torch.randn(points.shape, generator=generator)
    return points + torch.tensor(origin)


def _rotation(axis: torch.Tensor, angle_deg: float) -> torch.Tensor:
    axis = axis / axis.norm()
    angle = math.radians(angle_deg)
    K = torch.tensor([
        [0.0, -float(axis[2]), float(axis[1])],
        [float(axis[2]), 0.0, -float(axis[0])],
        [-float(axis[1]), float(axis[0]), 0.0],
    ])
    return torch.eye(3) + math.sin(angle) * K + (1.0 - math.cos(angle)) * (K @ K)


def _graph(xyz: torch.Tensor, scaling: torch.Tensor | None = None, knn_rig: int = 12, knn_ctx: int = 24):
    if scaling is None:
        scaling = torch.full((xyz.shape[0], 3), 0.01)
    return build_relational_graph(xyz, scaling=scaling, knn_rig=knn_rig, knn_ctx=knn_ctx)


def test_descriptor_is_invariant_to_an_arbitrary_rigid_transform():
    # Section 41 test 1. This is the property the whole coupling rests on.
    xyz = _block()
    R = _rotation(torch.tensor([0.3, -0.7, 0.5]), 37.0)
    t = torch.tensor([1.7, -2.3, 0.9])
    moved = xyz @ R.transpose(0, 1) + t

    a = _graph(xyz)
    b = _graph(moved)
    assert torch.allclose(a.distance_descriptor, b.distance_descriptor, atol=1.0e-4)
    assert torch.allclose(a.edge_len_rig.sort(dim=1).values, b.edge_len_rig.sort(dim=1).values, atol=1.0e-5)
    assert torch.allclose(a.effective_scale, b.effective_scale, atol=1.0e-6)


def test_covariance_invariants_are_rotation_invariant():
    # Sigma' = R Sigma R^T leaves the eigenvalues alone.
    scales = torch.tensor([[0.03, 0.01, 0.005], [0.02, 0.02, 0.001]])
    direct = compute_covariance_invariants(scales)
    R = _rotation(torch.tensor([0.2, 0.5, -0.9]), 51.0)
    sigma = torch.stack([R @ torch.diag(s.square()) @ R.transpose(0, 1) for s in scales])
    rotated = compute_covariance_invariants(covariance=sigma)
    assert torch.allclose(direct, rotated, atol=1.0e-5)
    assert torch.allclose(direct.sum(dim=-1), torch.ones(2), atol=1.0e-5)


def test_descriptor_retrieval_survives_a_large_translation():
    # Section 41 test 2. A translation far larger than the object must not
    # stop descriptor retrieval finding the structurally matching region --
    # this is precisely where XYZ-nearest-neighbour matching fails.
    xyz = _block(n=4, spacing=0.02)
    moved = xyz + torch.tensor([5.0, -3.0, 2.0])
    source, target = _graph(xyz, knn_rig=8, knn_ctx=12), _graph(moved, knn_rig=8, knn_ctx=12)
    cost, idx = build_descriptor_candidates(source, target, max_candidates=4)
    # Identity is the true correspondence here; it must be retrieved at rank 0
    # for the overwhelming majority of points despite the 6-metre offset.
    hit = (idx[:, 0] == torch.arange(xyz.shape[0])).float().mean()
    assert float(hit) > 0.5, float(hit)
    assert float(cost[:, 0].max()) < 1.0e-3


def test_relational_cost_is_zero_for_a_consistent_rigid_pairing():
    xyz = _block(n=4)
    R = _rotation(torch.tensor([0.0, 0.0, 1.0]), 25.0)
    moved = xyz @ R.transpose(0, 1) + torch.tensor([0.4, 0.1, -0.2])
    source, target = _graph(xyz, knn_rig=8, knn_ctx=12), _graph(moved, knn_rig=8, knn_ctx=12)

    n = xyz.shape[0]
    identity = torch.arange(n)[:, None]
    valid = torch.ones_like(identity, dtype=torch.bool)
    mass = torch.ones_like(identity, dtype=torch.float32)
    cost = compute_sparse_relational_cost(source, target, identity, valid, mass=mass)
    assert float(cost.max()) < 1.0e-6, float(cost.max())

    # A deliberately scrambled pairing must be penalised instead.
    scrambled = torch.roll(torch.arange(n), 7)[:, None]
    bad = compute_sparse_relational_cost(source, target, scrambled, valid, mass=mass)
    assert float(bad.mean()) > float(cost.mean())


def test_bidirectional_union_is_transpose_compatible():
    # Section 41 test 3. A target's proposal must appear in its source's row
    # even when that source did not propose it.
    forward_idx = torch.tensor([[0], [1]])
    forward_cost = torch.tensor([[0.1], [0.1]])
    # Target 2 proposes source 0; source 0 never proposed target 2.
    backward_idx = torch.tensor([[1], [1], [0]])
    backward_cost = torch.tensor([[0.9], [0.9], [0.05]])
    idx, cost, valid = build_bidirectional_candidate_union(
        forward_idx, forward_cost, backward_idx, backward_cost, num_sources=2, max_candidates=4
    )
    row0 = set(int(v) for v in idx[0][valid[0]])
    assert 0 in row0 and 2 in row0, row0
    # The cheaper of the two directions wins for a duplicated pair.
    assert float(cost[valid].max()) <= 0.9


def test_partial_transport_leaves_hopeless_points_unmatched():
    # Section 41 test 9. Unmatched mass must absorb a region with no
    # plausible counterpart instead of forcing a bad pair.
    cost = torch.tensor([[0.01, 5.0], [9.0, 9.5]])
    valid = torch.ones_like(cost, dtype=torch.bool)
    target_idx = torch.tensor([[0, 1], [0, 1]])
    mass, unmatched = initialize_soft_partial_transport(
        cost, valid, num_targets=2, target_idx=target_idx, unmatched_quantile=0.5
    )
    assert float(mass[0].sum()) > float(mass[1].sum())
    assert float(unmatched[1]) > float(unmatched[0])
    assert float((mass.sum(dim=1) + unmatched).max()) <= 1.0 + 1.0e-5


def test_coupling_on_a_rigidly_moved_cloud_is_confident_and_covers_both_sides():
    xyz = _block(n=4)
    R = _rotation(torch.tensor([0.1, 0.2, 0.97]), 18.0)
    moved = xyz @ R.transpose(0, 1) + torch.tensor([0.6, -0.4, 0.25])
    source, target = _graph(xyz, knn_rig=8, knn_ctx=12), _graph(moved, knn_rig=8, knn_ctx=12)
    unary_only = build_relational_coupling(
        source, target, max_candidates=8, relational_sweeps=0, column_capacity=8.0
    )
    coupling = build_relational_coupling(
        source, target, max_candidates=8, relational_sweeps=2, column_capacity=8.0
    )
    stats = compute_transport_coverage(coupling)
    assert stats["source_coverage"] > 0.9, stats
    assert stats["target_coverage"] > 0.5, stats

    def _picked(c):
        return c.tgt_idx.gather(1, (c.mass * c.valid).argmax(dim=1)[:, None]).reshape(-1)

    identity = torch.arange(xyz.shape[0])
    assert float((_picked(coupling) == identity).float().mean()) > 0.9
    # On clean synthetic geometry the descriptor already resolves the match,
    # so what the relational sweeps must do is sharpen it -- and above all
    # never degrade a correct descriptor match, which is the failure mode a
    # cost-shaped (rather than reward-shaped) relational term produces.
    assert float((_picked(coupling) == identity).float().mean()) >= float(
        (_picked(unary_only) == identity).float().mean()
    )
    assert coupling.mass.max(dim=1).values.mean() > unary_only.mass.max(dim=1).values.mean()


def test_two_moving_parts_keep_their_own_relational_support():
    # Section 41 test 8. Two parts moving differently must each couple to
    # their own counterpart without any part label being supplied.
    # Distinct jitter seeds: two *different* physical parts have different
    # local geometry.  Two exact translates of one another would be genuinely
    # indistinguishable to any rigid-invariant descriptor -- a real limitation
    # of this approach on repeated structure, not something to test around.
    part_a = _block(n=3, origin=(0.0, 0.0, 0.0), seed=1)
    part_b = _block(n=3, origin=(0.6, 0.0, 0.0), seed=2)
    xyz = torch.cat((part_a, part_b), dim=0)
    R_a = _rotation(torch.tensor([0.0, 0.0, 1.0]), 12.0)
    R_b = _rotation(torch.tensor([1.0, 0.0, 0.0]), -20.0)
    moved = torch.cat((
        part_a @ R_a.transpose(0, 1) + torch.tensor([0.3, 0.0, 0.0]),
        part_b @ R_b.transpose(0, 1) + torch.tensor([0.0, 0.5, -0.1]),
    ), dim=0)
    source, target = _graph(xyz, knn_rig=6, knn_ctx=10), _graph(moved, knn_rig=6, knn_ctx=10)
    coupling = build_relational_coupling(
        source, target, max_candidates=8, relational_sweeps=2, column_capacity=8.0
    )
    best = (coupling.mass * coupling.valid).argmax(dim=1)
    chosen = coupling.tgt_idx.gather(1, best[:, None]).reshape(-1)
    n_a = part_a.shape[0]
    # Points must land in their OWN part's index range, which is the thing a
    # single global transform would get wrong for at least one of the parts.
    a_correct = float((chosen[:n_a] < n_a).float().mean())
    b_correct = float((chosen[n_a:] >= n_a).float().mean())
    assert a_correct > 0.7 and b_correct > 0.7, (a_correct, b_correct)


def test_transport_entropy_is_low_for_a_confident_row():
    cost = torch.tensor([[0.0, 10.0, 10.0], [1.0, 1.0, 1.0]])
    valid = torch.ones_like(cost, dtype=torch.bool)
    target_idx = torch.tensor([[0, 1, 2], [0, 1, 2]])
    mass, unmatched = initialize_soft_partial_transport(
        cost, valid, num_targets=3, target_idx=target_idx, temperature=1.0, column_capacity=100.0
    )
    from utils.relational_coupling import SparseRelationalCoupling

    coupling = SparseRelationalCoupling(
        src_idx=torch.arange(2)[:, None].expand_as(target_idx), tgt_idx=target_idx,
        unary_cost=cost, relational_cost=torch.zeros_like(cost), total_cost=cost,
        mass=mass, valid=valid, unmatched_src=unmatched,
        row_mass=mass.sum(dim=1), col_mass=torch.zeros(3), num_targets=3,
    )
    entropy = compute_transport_entropy(coupling)
    assert float(entropy[0]) < float(entropy[1])


def test_local_pca_degeneracy_flags_a_planar_patch():
    plane = torch.rand(200, 3)
    plane[:, 2] = 0.0  # perfectly flat
    volume = torch.rand(200, 3)
    from utils.local_rigid_field import build_knn_graph

    flat_idx, _ = build_knn_graph(plane, k=12)
    full_idx, _ = build_knn_graph(volume, k=12)
    flat_ratio, _ = compute_local_pca_degeneracy(plane, flat_idx)
    full_ratio, _ = compute_local_pca_degeneracy(volume, full_idx)
    assert float(flat_ratio.median()) < 1.0e-6
    assert float(full_ratio.median()) > 1.0e-3


def main() -> None:
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"PASS {name}")


if __name__ == "__main__":
    main()
