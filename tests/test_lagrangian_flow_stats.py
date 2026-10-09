import torch

from utils.motion_correspondence import accumulate_flow_stats_cpu_reference


def test_cpu_grouped_flow_stats_equal_summed_per_gaussian_stats():
    generator = torch.Generator().manual_seed(7)
    jacobian = torch.randn((12, 2, 3), generator=generator, dtype=torch.float64)
    flow = torch.randn((12, 2), generator=generator, dtype=torch.float64)
    weight = torch.rand(12, generator=generator, dtype=torch.float64) + 0.1
    identity = torch.arange(12)
    group = torch.arange(12) // 3
    direct = accumulate_flow_stats_cpu_reference(
        jacobian, flow, weight, identity, 12, delta_time=0.1
    )
    grouped = accumulate_flow_stats_cpu_reference(
        jacobian, flow, weight, group, 4, delta_time=0.1
    )
    for key in ("normal_matrix", "rhs", "target_y2", "weight_sum", "observation_count"):
        manual = torch.stack(
            [direct[key][group == index].sum(dim=0) for index in range(4)]
        )
        assert torch.allclose(grouped[key], manual, atol=1e-10, rtol=1e-10)


def test_cuda_grouped_flow_stats_equal_summed_ungrouped_stats():
    if not torch.cuda.is_available():
        return
    try:
        from diff_gaussian_rasterization import (
            GaussianRasterizationSettings,
            GaussianRasterizer,
        )
        from utils.graphics_utils import getProjectionMatrix
    except ImportError:
        return
    device = "cuda"
    fov = 1.0
    view = torch.eye(4, device=device)
    projection = getProjectionMatrix(0.01, 100.0, fov, fov).T.cuda()
    settings = GaussianRasterizationSettings(
        image_height=32,
        image_width=32,
        tanfovx=float(torch.tan(torch.tensor(0.5 * fov))),
        tanfovy=float(torch.tan(torch.tensor(0.5 * fov))),
        bg=torch.zeros(3, device=device),
        scale_modifier=1.0,
        viewmatrix=view,
        projmatrix=view @ projection,
        sh_degree=0,
        campos=torch.zeros(3, device=device),
        prefiltered=False,
        debug=False,
        antialiasing=False,
    )
    rasterizer = GaussianRasterizer(settings)
    common = {
        "means3D": torch.tensor(((-0.15, 0.0, 2.0), (0.15, 0.0, 2.0)), device=device),
        "observed_flow": torch.ones((2, 32, 32), device=device) * 0.02,
        "observed_confidence": torch.ones((1, 32, 32), device=device),
        "opacities": torch.full((2, 1), 0.9, device=device),
        "scales": torch.full((2, 3), 0.12, device=device),
        "rotations": torch.tensor(((1.0, 0.0, 0.0, 0.0),) * 2, device=device),
        "delta_time": 1.0 / 30.0,
    }
    ungrouped = rasterizer.accumulate_flow_stats(**common)
    grouped = rasterizer.accumulate_flow_stats(
        **common,
        group_ids=torch.zeros(2, device=device, dtype=torch.long),
        num_groups=1,
    )
    for index in (1, 2, 3, 4, 5):
        expected = ungrouped[index].sum(dim=0)
        if grouped[index].dtype.is_floating_point:
            assert torch.allclose(
                grouped[index][0], expected, atol=2e-4, rtol=2e-4
            )
        else:
            assert int(grouped[index][0]) == int(expected)
