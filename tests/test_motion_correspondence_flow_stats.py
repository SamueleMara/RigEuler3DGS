import torch

from utils.motion_correspondence import (
    accumulate_flow_stats_cpu_reference,
    moving_model_static_improvement_mask,
)


def test_cpu_flow_stats_match_direct_normal_equations():
    jacobian = torch.randn(
        (8, 2, 3), generator=torch.Generator().manual_seed(12), dtype=torch.float64
    )
    velocity = torch.tensor(
        [[0.4, -0.2, 0.1], [-0.3, 0.2, 0.5]], dtype=torch.float64
    )
    groups = torch.tensor([0, 0, 0, 0, 1, 1, 1, 1])
    weights = torch.tensor(
        [0.5, 0.8, 0.4, 0.6, 0.7, 0.9, 0.3, 0.8], dtype=torch.float64
    )
    dt = 0.25
    flow = torch.einsum("mij,mj->mi", jacobian, velocity[groups]) * dt
    stats = accumulate_flow_stats_cpu_reference(
        jacobian, flow, weights, groups, 2, delta_time=dt
    )
    solved = torch.linalg.solve(stats["normal_matrix"], stats["rhs"])
    assert torch.allclose(solved, velocity.double(), atol=1e-10, rtol=1e-10)


def test_static_improvement_is_mandatory_when_enabled():
    moving = torch.tensor([0.10, 0.50, 0.01])
    static = torch.tensor([0.12, 1.00, 0.50])
    accepted, absolute, relative = moving_model_static_improvement_mask(
        moving,
        static,
        absolute_margin_px=0.05,
        relative_margin=0.15,
        require_improvement=True,
    )
    assert accepted.tolist() == [False, True, True]
    assert torch.allclose(absolute, static - moving)
    assert torch.allclose(relative, (static - moving) / static)


def test_zero_flow_reference_has_zero_rhs_and_cannot_be_moving():
    jacobian = torch.randn((20, 2, 3), generator=torch.Generator().manual_seed(4))
    stats = accumulate_flow_stats_cpu_reference(
        jacobian,
        torch.zeros((20, 2)),
        torch.ones(20),
        torch.arange(20) % 2,
        2,
        delta_time=1.0 / 30.0,
    )
    accepted, _, _ = moving_model_static_improvement_mask(
        torch.zeros(2),
        torch.zeros(2),
        absolute_margin_px=0.05,
        relative_margin=0.15,
        require_improvement=True,
    )
    assert torch.equal(stats["rhs"], torch.zeros_like(stats["rhs"]))
    assert not bool(accepted.any())


def test_cuda_flow_stats_zero_flow_smoke():
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
    result = GaussianRasterizer(settings).accumulate_flow_stats(
        means3D=torch.tensor([[0.0, 0.0, 2.0]], device=device),
        observed_flow=torch.zeros((2, 32, 32), device=device),
        observed_confidence=torch.ones((1, 32, 32), device=device),
        group_ids=torch.zeros(1, device=device, dtype=torch.long),
        num_groups=1,
        opacities=torch.full((1, 1), 0.9, device=device),
        scales=torch.full((1, 3), 0.2, device=device),
        rotations=torch.tensor([[1.0, 0.0, 0.0, 0.0]], device=device),
        delta_time=1.0 / 30.0,
    )
    rendered, normal6, rhs, y2, weight, count, _ = result
    assert rendered > 0
    assert torch.isfinite(normal6).all() and float(weight[0]) > 0
    assert torch.equal(rhs, torch.zeros_like(rhs))
    assert torch.equal(y2, torch.zeros_like(y2))
    assert int(count[0]) > 0


def test_cuda_flow_stats_matches_projected_velocity():
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
    fov, height, width = 1.0, 32, 32
    view = torch.eye(4, device=device)
    projection = getProjectionMatrix(0.01, 100.0, fov, fov).T.cuda()
    full_projection = view @ projection
    settings = GaussianRasterizationSettings(
        image_height=height,
        image_width=width,
        tanfovx=float(torch.tan(torch.tensor(0.5 * fov))),
        tanfovy=float(torch.tan(torch.tensor(0.5 * fov))),
        bg=torch.zeros(3, device=device),
        scale_modifier=1.0,
        viewmatrix=view,
        projmatrix=full_projection,
        sh_degree=0,
        campos=torch.zeros(3, device=device),
        prefiltered=False,
        debug=False,
        antialiasing=False,
    )
    point = torch.tensor([0.0, 0.0, 2.0], dtype=torch.float64, requires_grad=True)
    matrix = full_projection.detach().cpu().double()

    def project(value):
        clip = torch.cat((value, value.new_ones(1))) @ matrix
        ndc = clip[:2] / clip[3]
        return torch.stack(
            (
                0.5 * (ndc[0] + 1.0) * (width - 1),
                0.5 * (ndc[1] + 1.0) * (height - 1),
            )
        )

    jacobian = torch.autograd.functional.jacobian(project, point)
    velocity = torch.tensor([0.2, -0.1, 0.05], dtype=torch.float64)
    dt = 1.0 / 30.0
    displacement = (jacobian @ velocity * dt).float().cuda()
    flow = displacement[:, None, None].expand(2, height, width).contiguous()
    result = GaussianRasterizer(settings).accumulate_flow_stats(
        means3D=point.detach().float().cuda()[None],
        observed_flow=flow,
        observed_confidence=torch.ones((1, height, width), device=device),
        group_ids=torch.zeros(1, device=device, dtype=torch.long),
        num_groups=1,
        opacities=torch.full((1, 1), 0.9, device=device),
        scales=torch.full((1, 3), 0.2, device=device),
        rotations=torch.tensor([[1.0, 0.0, 0.0, 0.0]], device=device),
        delta_time=dt,
    )
    _, normal6, rhs, y2, weight, count, _ = result
    normal = torch.stack(
        (
            torch.stack((normal6[0, 0], normal6[0, 1], normal6[0, 2])),
            torch.stack((normal6[0, 1], normal6[0, 3], normal6[0, 4])),
            torch.stack((normal6[0, 2], normal6[0, 4], normal6[0, 5])),
        )
    )
    expected_rhs = normal @ velocity.float().cuda()
    expected_y2 = weight[0] * torch.linalg.norm(displacement / dt).square()
    assert int(count[0]) > 0
    assert torch.allclose(rhs[0], expected_rhs, rtol=3e-4, atol=3e-4)
    assert torch.allclose(y2[0], expected_y2, rtol=3e-4, atol=3e-4)
