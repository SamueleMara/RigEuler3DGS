import torch

from utils.static_flow_calibration import (
    evaluate_static_flow_residual,
    fit_robust_flow_null,
)


def test_systematic_static_bias_is_removed():
    generator = torch.Generator().manual_seed(7)
    static = torch.tensor([0.4, -0.2]) + 0.02 * torch.randn(1000, 2, generator=generator)
    calibration = fit_robust_flow_null(static, mad_floor=0.01, min_samples=64)
    evaluated = evaluate_static_flow_residual(static, calibration)
    assert torch.allclose(calibration["median"], torch.tensor([0.4, -0.2]), atol=0.01)
    assert float(evaluated["corrected_flow"].norm(dim=-1).median()) < 0.04


def test_true_additional_motion_remains_detectable():
    static = torch.tensor([0.4, -0.2]).repeat(100, 1)
    calibration = fit_robust_flow_null(static, mad_floor=0.05, min_samples=64)
    biased_foreground = torch.tensor([[0.4, -0.2], [1.4, -0.2]])
    residual = evaluate_static_flow_residual(biased_foreground, calibration)["normalized_residual"]
    assert float(residual[0]) < 0.1
    assert float(residual[1]) > 10.0

