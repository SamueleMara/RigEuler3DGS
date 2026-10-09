import inspect
from pathlib import Path
import sys

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from utils.dynamic_prior import make_all_foreground_prior, make_gt_prior, prior_metrics
from utils.endpoint_transport_prior import EndpointTransportConfig, EndpointTransportPrior
from utils.endpoint_flow_matching import (
    FlowMatchingConfig,
    build_dynamic_prior_from_flow,
    integrate_flow,
    train_endpoint_flow_matching,
)
from utils.killing_prior_adapter import dynamic_prior_to_motion_state


def test_dynamic_prior_container_and_metrics():
    ids = torch.tensor([0, 2, 3])
    gt = torch.tensor([False, True, True, False])
    prior = make_gt_prior(ids, gt)
    metrics = prior_metrics(prior, gt)
    assert metrics["all_f1"] == 1.0
    all_fg = make_all_foreground_prior(ids)
    assert torch.equal(all_fg.gaussian_ids, ids)
    assert torch.allclose(all_fg.probability, torch.ones(3))


def test_prior_adapter_keeps_low_confidence_unknown():
    ids = torch.tensor([1, 3])
    prior = make_all_foreground_prior(ids)
    prior.confidence[1] = 0.0
    state = dynamic_prior_to_motion_state(prior, torch.zeros(5, 3))
    assert state["unknown_score"][3] == 1.0
    assert state["initial_dynamic_mask"][1]
    assert not state["initial_dynamic_mask"][3]


def test_endpoint_transport_identical_is_static():
    xyz = torch.randn(32, 3) * 0.05
    cfg = EndpointTransportConfig(knn=1, temperature=1.0e-3, prior_prob_bias=0.5, prior_prob_temperature=0.1)
    result = EndpointTransportPrior(xyz, {}, cfg).compute(xyz)
    assert float(result.displacement.norm(dim=1).max()) < 1.0e-6
    assert float(result.prior.probability.max()) < 0.05


def test_endpoint_transport_half_translated_marks_moved_cluster():
    a = torch.randn(32, 3) * 0.02
    b = torch.randn(32, 3) * 0.02 + torch.tensor([2.0, 0.0, 0.0])
    source = torch.cat((a, b), 0)
    target = torch.cat((a, b + torch.tensor([0.4, 0.0, 0.0])), 0)
    cfg = EndpointTransportConfig(knn=1, temperature=1.0e-3, prior_prob_bias=2.0, prior_prob_temperature=0.5)
    result = EndpointTransportPrior(target, {}, cfg).compute(source)
    assert float(result.prior.probability[:32].mean()) < 0.05
    assert float(result.prior.probability[32:].mean()) > 0.8


def test_missing_endpoint_region_reduces_confidence():
    a = torch.randn(32, 3) * 0.02
    b = torch.randn(32, 3) * 0.02 + torch.tensor([3.0, 0.0, 0.0])
    source = torch.cat((a, b), 0)
    cfg = EndpointTransportConfig(knn=4, temperature=1.0e-3, prior_prob_bias=1.0, prior_prob_temperature=0.3)
    result = EndpointTransportPrior(a, {}, cfg).compute(source)
    assert float(result.prior.confidence[32:].mean()) < float(result.prior.confidence[:32].mean())
    assert float(result.prior.confidence[32:].mean()) < 0.2


def test_endpoint_prior_apis_are_not_killing_conditioned():
    assert "killing" not in inspect.signature(EndpointTransportPrior.compute).parameters
    assert "rigid_field" not in inspect.signature(train_endpoint_flow_matching).parameters


def test_flow_matching_prior_recovers_from_zero_endpoint_confidence():
    source = torch.randn(32, 3) * 0.02
    integrated = source.clone()
    integrated[16:] = integrated[16:] + torch.tensor([0.4, 0.0, 0.0])
    prior = build_dynamic_prior_from_flow(
        source,
        integrated,
        torch.arange(32),
        FlowMatchingConfig(prior_prob_bias=2.0, prior_prob_temperature=0.5),
        confidence=torch.zeros(32),
        valid=torch.ones(32, dtype=torch.bool),
    )
    assert float(prior.confidence[:16].mean()) > 0.8
    assert float(prior.confidence[16:].mean()) > 0.8


def test_flow_matching_training_smoke_cpu():
    source = torch.randn(16, 3) * 0.02
    target = source + torch.tensor([0.1, 0.0, 0.0])
    cfg_transport = EndpointTransportConfig(knn=1, temperature=1.0e-3, prior_prob_bias=0.5, prior_prob_temperature=0.2)
    transport = EndpointTransportPrior(target, {}, cfg_transport).compute(source)
    cfg = FlowMatchingConfig(hidden_dim=16, layers=1, iters=2, batch_size=8, integration_steps=2, seed=7)
    model, logs = train_endpoint_flow_matching(source, target, transport, cfg, device="cpu")
    integrated = integrate_flow(model, source, cfg)
    assert integrated.shape == source.shape
    assert len(logs) >= 1


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
    print("dynamic_prior_comparison tests passed")
