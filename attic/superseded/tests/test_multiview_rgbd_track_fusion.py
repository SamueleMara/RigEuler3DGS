from __future__ import annotations

import torch

from utils.multiview_track_fusion import associate_multiview_track_observations, compute_multiview_track_consistency, fuse_world_track_measurements
from utils.rgbd_track_lifting import WorldTrack


def _track(x):
    pos = torch.tensor([[x, 0.0, 0.0], [x + 0.01, 0.0, 0.0]])
    cov = torch.eye(3).expand(2, 3, 3).clone() * 0.01
    return WorldTrack(pos, cov, torch.ones(2, dtype=torch.bool), torch.arange(2), torch.arange(2).float())


def test_multiview_fusion_precision_weighted():
    pos, cov = fuse_world_track_measurements(torch.tensor([[0.0, 0.0, 0.0], [0.1, 0.0, 0.0]]), torch.eye(3).expand(2, 3, 3) * 0.01)
    assert abs(float(pos[0]) - 0.05) < 1.0e-5
    assert cov[0, 0] < 0.01


def test_track_association_and_consistency_are_covariance_based():
    groups = associate_multiview_track_observations([_track(0.0), _track(0.02), _track(5.0)])
    assert any(set(g) == {0, 1} for g in groups)
    stats = compute_multiview_track_consistency([_track(0.0), _track(0.02)])
    assert float(stats["median_mahalanobis"]) < 1.0


if __name__ == "__main__":
    test_multiview_fusion_precision_weighted()
    test_track_association_and_consistency_are_covariance_based()
    print("PASS multiview_rgbd_track_fusion")
