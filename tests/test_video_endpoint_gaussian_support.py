from __future__ import annotations

import torch

from utils.endpoint_video_candidate_search import measure_candidate_recall_on_exhaustive_subset, rank_target_gaussians_from_video_support
from utils.video_endpoint_transport import build_sparse_video_endpoint_gaussian_support


def test_sparse_video_endpoint_support_ranks_correct_target():
    source_own = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    target_own = torch.tensor([[0.0, 1.0], [1.0, 0.0]])
    support = build_sparse_video_endpoint_gaussian_support(
        source_own,
        target_own,
        torch.tensor([0, 1]),
        torch.tensor([1, 0]),
        torch.tensor([1.0, 1.0]),
        topk_gaussians=1,
    )
    ranked, score = rank_target_gaussians_from_video_support(support, num_sources=2, k=1)
    assert ranked[:, 0].tolist() == [0, 1]
    recall = measure_candidate_recall_on_exhaustive_subset(ranked, torch.tensor([0, 1]), depths=(1,))
    assert recall["recall_at_1"] == 1.0


if __name__ == "__main__":
    test_sparse_video_endpoint_support_ranks_correct_target()
    print("PASS video_endpoint_gaussian_support")
