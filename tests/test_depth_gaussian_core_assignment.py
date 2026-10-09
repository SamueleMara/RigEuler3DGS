import torch

from utils.gaussian_sequence_assignment import (
    DEPTH_EVIDENCE_CHANNELS,
    build_depth_only_initial_cores,
    compute_gaussian_camera_votes,
)


def test_multiview_core_assignment_rejects_one_camera_and_conflicts():
    features = torch.zeros((4, 3, len(DEPTH_EVIDENCE_CHANNELS)))
    observable = DEPTH_EVIDENCE_CHANNELS.index("observable")
    static = DEPTH_EVIDENCE_CHANNELS.index("static")
    dynamic = DEPTH_EVIDENCE_CHANNELS.index("dynamic")
    features[0, :2, observable] = features[0, :2, dynamic] = 1.0
    features[1, :, observable] = features[1, :, static] = 1.0
    features[2, 0, observable] = features[2, 0, dynamic] = 1.0
    features[3, :2, observable] = features[3, :2, dynamic] = 1.0
    features[3, 2, observable] = features[3, 2, static] = 1.0
    evidence = {
        "feature_sum": features,
        "raster_weight_sum": torch.ones((4, 3)),
        "channel_names": DEPTH_EVIDENCE_CHANNELS,
    }
    votes = compute_gaussian_camera_votes(
        evidence,
        min_camera_weight=0.5,
        static_threshold=0.75,
        dynamic_threshold=0.75,
    )
    cores = build_depth_only_initial_cores(
        votes,
        min_static_cameras=2,
        min_dynamic_cameras=2,
        min_static_camera_fraction=0.5,
        min_dynamic_camera_fraction=0.5,
        max_conflicting_cameras=0,
    )
    assert cores["dynamic_core_mask"].tolist() == [True, False, False, False]
    assert cores["static_core_mask"].tolist() == [False, True, False, False]
    assert cores["unknown_mask"].tolist() == [False, False, True, True]


def test_cuda_constant_pixel_features_preserve_talpha_weighting():
    if not torch.cuda.is_available():
        return
    from diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer
    from utils.graphics_utils import getProjectionMatrix

    device, fov = "cuda", 1.0
    view = torch.eye(4, device=device)
    projection = getProjectionMatrix(0.01, 100.0, fov, fov).T.cuda()
    settings = GaussianRasterizationSettings(
        image_height=24,
        image_width=24,
        tanfovx=float(torch.tan(torch.tensor(0.5 * fov))),
        tanfovy=float(torch.tan(torch.tensor(0.5 * fov))),
        bg=torch.zeros(2, device=device),
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
    constant = torch.tensor((0.25, 0.75), device=device)[:, None, None]
    _, feature_sum, weight, _, _ = rasterizer.accumulate_pixel_features_to_gaussians(
        means3D=torch.tensor(((-0.1, 0.0, 2.0), (0.1, 0.0, 2.0)), device=device),
        pixel_features=constant.expand(2, 24, 24).contiguous(),
        pixel_confidence=torch.ones((1, 24, 24), device=device),
        opacities=torch.full((2, 1), 0.9, device=device),
        scales=torch.full((2, 3), 0.12, device=device),
        rotations=torch.tensor(((1.0, 0.0, 0.0, 0.0),) * 2, device=device),
    )
    assert torch.allclose(feature_sum, weight[:, None] * constant[:, 0, 0], atol=2e-4, rtol=2e-4)

