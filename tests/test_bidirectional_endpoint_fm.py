from __future__ import annotations

import torch

from utils.bidirectional_endpoint_fm import compute_endpoint_cycle_consistency, train_backward_endpoint_fm, train_forward_endpoint_fm
from utils.bidirectional_fm import FlowTrainConfig


def test_forward_backward_endpoint_fm_are_independent_and_cycle_runs():
    source = torch.tensor([[0.0, 0.0, 0.0], [0.2, 0.0, 0.0], [0.0, 0.2, 0.0], [0.2, 0.2, 0.0]])
    target = source + torch.tensor([0.1, 0.0, 0.0])
    cfg = FlowTrainConfig(iters=2, batch_size=4, knn=2, integration_steps=2, hidden_dim=16, layers=1, seed=7)
    fwd = train_forward_endpoint_fm(source, target, cfg, device="cpu")
    bwd = train_backward_endpoint_fm(source, target, FlowTrainConfig(**{**cfg.__dict__, "seed": 8}), device="cpu")
    assert fwd.model is not bwd.model
    cycle = compute_endpoint_cycle_consistency(source, target, fwd, bwd, cfg, cfg, device="cpu")
    assert cycle["forward_cycle_error"].shape[0] == source.shape[0]
    assert cycle["backward_cycle_error"].shape[0] == target.shape[0]


if __name__ == "__main__":
    test_forward_backward_endpoint_fm_are_independent_and_cycle_runs()
    print("PASS bidirectional_endpoint_fm")
