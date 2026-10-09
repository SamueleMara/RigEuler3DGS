import math

import pytest
import torch

from utils.gaussian_sequence_assignment import build_initial_cores


def test_initial_cores_are_precision_first_and_occlusion_is_unknown():
    evidence = torch.zeros((4, 3, 5, 6))
    # Reliable static, reliable dynamic, occluded static, weak moving.
    evidence[0, ..., 0] = 1.0
    evidence[0, ..., 3] = 1.0
    evidence[0, ..., 5] = 1.0
    # Confidence-weighted amplitudes need not approach one. Model selection
    # uses their dynamic-vs-static ratio, while persistence remains mandatory.
    evidence[1, ..., 2] = 0.25
    evidence[1, ..., 4] = 0.25
    evidence[1, ..., 5] = 1.0
    evidence[2, ..., 1] = 1.0
    state = build_initial_cores({
        "feature_mean": evidence,
        "weight_sum": torch.ones((4, 3, 5)),
        "observation_count": torch.full((4, 3, 5), 10),
        "channel_names": ("present", "occluded", "vacated", "static_flow", "dynamic_flow", "informative"),
    })
    assert state["static_core_mask"].tolist() == [True, False, False, False]
    assert state["dynamic_core_mask"].tolist() == [False, True, False, False]
    assert state["unknown_mask"].tolist() == [False, False, True, True]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA rasterizer test")
def test_constant_multichannel_features_equal_constant_times_ownership():
    from diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer
    from utils.graphics_utils import getProjectionMatrix

    device = "cuda"
    means = torch.tensor([[0.0, 0.0, 2.0], [0.0, 0.0, 3.0]], device=device)
    opacity = torch.tensor([[0.8], [0.8]], device=device)
    scales = torch.full((2, 3), 0.25, device=device)
    rotations = torch.tensor([[1.0, 0.0, 0.0, 0.0]] * 2, device=device)
    identity = torch.eye(4, device=device)
    projection = getProjectionMatrix(0.01, 100.0, 1.0, 1.0).T.cuda()
    settings = GaussianRasterizationSettings(
        image_height=16, image_width=16, tanfovx=math.tan(0.5), tanfovy=math.tan(0.5),
        bg=torch.zeros(1, device=device), scale_modifier=1.0,
        viewmatrix=identity, projmatrix=identity @ projection, sh_degree=0,
        campos=torch.zeros(3, device=device), prefiltered=False, debug=False,
        antialiasing=False,
    )
    rasterizer = GaussianRasterizer(settings)
    constants = torch.tensor([2.0, -3.0, 0.5], device=device)[:, None, None]
    features = constants.expand(3, 16, 16).contiguous()
    confidence = torch.ones((1, 16, 16), device=device)
    _, feature_sum, weight, count, _ = rasterizer.accumulate_pixel_features_to_gaussians(
        means, features, confidence, opacity, scales=scales, rotations=rotations
    )
    assert torch.allclose(feature_sum, weight[:, None] * constants[:, 0, 0], atol=2e-5, rtol=2e-5)
    assert torch.all(count[weight > 0] > 0)
    assert weight[0] > weight[1]  # front splat receives ownership before the rear splat

    _, grouped_sum, grouped_weight, grouped_count, _ = rasterizer.accumulate_pixel_features_to_gaussians(
        means, features, confidence, opacity, scales=scales, rotations=rotations,
        group_ids=torch.zeros(2, device=device, dtype=torch.long), num_groups=1,
    )
    assert torch.allclose(grouped_sum[0], feature_sum.sum(dim=0), atol=2e-5, rtol=2e-5)
    assert torch.allclose(grouped_weight[0], weight.sum(), atol=2e-5, rtol=2e-5)
    assert grouped_count[0] == count.sum()

    _, zero_sum, zero_weight, zero_count, _ = rasterizer.accumulate_pixel_features_to_gaussians(
        means, features, torch.zeros_like(confidence), opacity,
        scales=scales, rotations=rotations
    )
    assert not bool(zero_sum.any())
    assert not bool(zero_weight.any())
    assert not bool(zero_count.any())
