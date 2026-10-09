"""Checks for the standalone bidirectional FM/diffusion matcher.

Run:
    PYTHONPATH=. python tests/test_bidirectional_fm_diffusion_matching.py
"""

from __future__ import annotations

import torch

from utils.bidirectional_match_candidates import FROM_BACKWARD_FM, FROM_DIFFUSION, FROM_FORWARD_FM, FROM_RIGID, build_motion_candidates
from utils.reciprocal_matching import strict_reciprocal_matches
from utils.rigid_match_refinement import RigidFitConfig, robust_rigid_fit, transform_points


def test_motion_candidates_have_no_static_source_fallback_and_dedup_rows():
    source = torch.tensor([[0.0, 0.0, 0.0], [0.1, 0.0, 0.0], [0.2, 0.0, 0.0]])
    target = torch.tensor([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.1, 0.0, 0.0], [1.2, 0.0, 0.0]])
    forward_endpoint = source + torch.tensor([1.0, 0.0, 0.0])
    backward_endpoint = target - torch.tensor([1.0, 0.0, 0.0])
    active_source = torch.ones(3, dtype=torch.bool)
    active_target = torch.tensor([False, True, True, True])
    T = torch.eye(4)
    T[:3, 3] = torch.tensor([1.0, 0.0, 0.0])
    candidates = build_motion_candidates(
        source_xyz=source,
        target_xyz=target,
        forward_endpoint=forward_endpoint,
        backward_endpoint=backward_endpoint,
        forward_displacement=forward_endpoint - source,
        backward_displacement=backward_endpoint - target,
        source_scale=torch.ones(3),
        target_scale=torch.ones(4),
        active_source_mask=active_source,
        active_target_mask=active_target,
        T=T,
        k=3,
        include_rigid_branch=True,
    )
    assert int(candidates.duplicate_stats["duplicates_after_dedup"]) == 0
    assert bool((candidates.target_ids != 0).all())  # target 0 is static-domain and forbidden here
    allowed = FROM_FORWARD_FM | FROM_RIGID | FROM_DIFFUSION | FROM_BACKWARD_FM
    assert bool(((candidates.provenance & ~allowed) == 0).all())
    for src in torch.unique(candidates.source_ids):
        row = candidates.target_ids[candidates.source_ids == src]
        assert row.numel() == torch.unique(row).numel()


def test_reverse_fm_branch_adds_missing_bidirectional_edges():
    source = torch.tensor([[0.0, 0.0, 0.0], [10.0, 0.0, 0.0]])
    target = torch.tensor([[0.5, 0.0, 0.0], [10.5, 0.0, 0.0]])
    forward_endpoint = torch.tensor([[0.5, 0.0, 0.0], [0.5, 0.0, 0.0]])
    backward_endpoint = torch.tensor([[0.0, 0.0, 0.0], [10.0, 0.0, 0.0]])
    candidates = build_motion_candidates(
        source_xyz=source,
        target_xyz=target,
        forward_endpoint=forward_endpoint,
        backward_endpoint=backward_endpoint,
        forward_displacement=forward_endpoint - source,
        backward_displacement=backward_endpoint - target,
        source_scale=torch.ones(2),
        target_scale=torch.ones(2),
        active_source_mask=torch.ones(2, dtype=torch.bool),
        active_target_mask=torch.ones(2, dtype=torch.bool),
        T=torch.eye(4),
        k=1,
        include_rigid_branch=False,
        include_reverse_branch=True,
    )
    pair_mask = (candidates.source_ids == 1) & (candidates.target_ids == 1)
    assert bool(pair_mask.any())
    assert bool((candidates.provenance[pair_mask] & FROM_BACKWARD_FM).any())


def test_strict_reciprocal_is_one_to_one():
    source_ids = torch.tensor([0, 0, 1, 1, 2])
    target_ids = torch.tensor([0, 1, 0, 2, 2])
    energy = torch.tensor([0.1, 0.3, 0.2, 0.1, 0.4])
    matches = strict_reciprocal_matches(
        source_ids=source_ids,
        target_ids=target_ids,
        energy=energy,
        source_weight=torch.ones(3),
        target_weight=torch.ones(3),
        num_sources=3,
        num_targets=3,
    )
    assert matches.stats["maximum_target_multiplicity"] == 1
    assert matches.source_ids.tolist() == [0, 1]
    assert matches.target_ids.tolist() == [0, 2]


def test_rigid_fit_from_reciprocal_pairs_recovers_transform():
    source = torch.tensor(
        [
            [-0.2, -0.1, 0.0],
            [0.2, -0.1, 0.0],
            [-0.2, 0.2, 0.1],
            [0.2, 0.2, -0.1],
            [0.0, 0.0, 0.3],
        ],
        dtype=torch.float32,
    )
    angle = torch.tensor(0.25)
    c, s = torch.cos(angle), torch.sin(angle)
    T = torch.eye(4)
    T[:3, :3] = torch.tensor([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
    T[:3, 3] = torch.tensor([0.05, -0.03, 0.02])
    target = transform_points(T, source)
    matches = strict_reciprocal_matches(
        source_ids=torch.arange(source.shape[0]),
        target_ids=torch.arange(source.shape[0]),
        energy=torch.zeros(source.shape[0]),
        source_weight=torch.ones(source.shape[0]),
        target_weight=torch.ones(source.shape[0]),
        num_sources=source.shape[0],
        num_targets=target.shape[0],
    )
    fit = robust_rigid_fit(
        source[matches.source_ids],
        target[matches.target_ids],
        matches.weights,
        scale=0.01,
        cfg=RigidFitConfig(irls_iters=3),
    )
    assert torch.allclose(fit["T"], T, atol=1.0e-4), fit["T"]
    assert fit["weighted_rms"] < 1.0e-5


if __name__ == "__main__":
    test_motion_candidates_have_no_static_source_fallback_and_dedup_rows()
    test_reverse_fm_branch_adds_missing_bidirectional_edges()
    test_strict_reciprocal_is_one_to_one()
    test_rigid_fit_from_reciprocal_pairs_recovers_transform()
    print("PASS bidirectional_fm_diffusion_matching")
