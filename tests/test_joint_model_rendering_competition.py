import torch

from utils.joint_model_fitting import (
    PrismaticJointModel,
    RevoluteJointModel,
    ScrewJointModel,
    _select_candidate,
)
from utils.rigid_transform_utils import integrate_joint_transforms, transport_gaussian_state


def _soft_render(points, rgb, dino, camera_rotation):
    camera = points @ camera_rotation.T + torch.tensor([0.0, 0.0, 2.5], dtype=points.dtype)
    uv = camera[:, :2] / camera[:, 2:].clamp_min(0.2)
    coordinate = torch.linspace(-0.7, 0.7, 9, dtype=points.dtype)
    gy, gx = torch.meshgrid(coordinate, coordinate, indexing="ij")
    grid = torch.stack((gx, gy), dim=-1)
    weight = torch.exp(-((uv[:, None, None] - grid[None]).square().sum(-1)) / 0.025)
    opacity = weight.sum(0).clamp_min(1.0e-8)
    rendered_rgb = torch.einsum("nhw,nc->chw", weight, rgb) / opacity
    rendered_depth = torch.einsum("nhw,n->hw", weight, camera[:, 2]) / opacity
    rendered_dino = torch.einsum("nhw,nc->chw", weight, dino) / opacity
    return torch.cat((rendered_rgb.reshape(-1), rendered_depth.reshape(-1), rendered_dino.reshape(-1)))


def _trajectory_render(model, points, orientations, rgb, dino, cameras):
    trajectory = model.integrate()
    rendered = []
    for frame in range(trajectory["rotations"].shape[0]):
        moved, _ = transport_gaussian_state(
            points, orientations, trajectory["rotations"][frame],
            trajectory["translations"][frame], trajectory["rotation_quaternions"][frame],
        )
        rendered.extend(_soft_render(moved, rgb, dino, camera) for camera in cameras)
    return torch.stack(rendered)


def _fit_candidates(target_type, pitch=0.0):
    dtype = torch.float64
    points = torch.tensor([
        [-0.7, -0.2, 0.1], [-0.2, 0.5, 0.3], [0.4, -0.4, -0.2],
        [0.8, 0.3, 0.2], [0.1, 0.8, -0.1], [-0.5, -0.7, 0.4],
    ], dtype=dtype)
    orientations = torch.tensor([[1.0, 0.0, 0.0, 0.0]], dtype=dtype).repeat(points.shape[0], 1)
    rgb = torch.tensor([
        [1.0, 0.1, 0.2], [0.2, 0.9, 0.1], [0.1, 0.2, 1.0],
        [0.8, 0.7, 0.1], [0.6, 0.1, 0.8], [0.1, 0.8, 0.8],
    ], dtype=dtype)
    dino = torch.nn.functional.normalize(torch.arange(24, dtype=dtype).reshape(6, 4) + 1, dim=-1)
    cameras = [
        torch.eye(3, dtype=dtype),
        torch.tensor([[0.866, 0.0, 0.5], [0.0, 1.0, 0.0], [-0.5, 0.0, 0.866]], dtype=dtype),
        torch.tensor([[0.94, -0.342, 0.0], [0.342, 0.94, 0.0], [0.0, 0.0, 1.0]], dtype=dtype),
    ]
    axis = torch.tensor([0.0, 0.0, 1.0], dtype=dtype)
    pivot = torch.tensor([0.15, -0.2, 0.0], dtype=dtype)
    q = torch.tensor([0.0, 0.18, 0.42, 0.7], dtype=dtype)
    truth = {"joint_type": target_type, "axis": axis, "pivot": None if target_type == "prismatic" else pivot, "pitch": pitch, "coarse_q": q}
    truth_class = {"revolute": RevoluteJointModel, "prismatic": PrismaticJointModel, "screw": ScrewJointModel}[target_type]
    target_model = truth_class(truth, device="cpu", dtype=dtype)
    target_train = _trajectory_render(
        target_model, points, orientations, rgb, dino, cameras[:2]
    ).detach()
    target_validation = _trajectory_render(
        target_model, points, orientations, rgb, dino, cameras[2:]
    ).detach()

    payloads = []
    for kind, cls in (("revolute", RevoluteJointModel), ("prismatic", PrismaticJointModel), ("screw", ScrewJointModel)):
        initial = {
            "joint_type": kind, "axis": axis,
            "pivot": None if kind == "prismatic" else pivot,
            "pitch": pitch if kind == "screw" else 0.0,
            "coarse_q": 0.7 * q,
        }
        model = cls(initial, device="cpu", dtype=dtype)
        model.set_trainable("trajectory")
        optimizer = torch.optim.Adam([model.delta_q], lr=0.03)
        for _ in range(50):
            optimizer.zero_grad(set_to_none=True)
            prediction = _trajectory_render(model, points, orientations, rgb, dino, cameras[:2])
            loss = (prediction - target_train).square().mean()
            loss.backward()
            optimizer.step()
        final_loss = float((
            _trajectory_render(model, points, orientations, rgb, dino, cameras[2:])
            - target_validation
        ).square().mean().detach())
        payloads.append({"joint_parameters": {"joint_type": kind}, "validation": {"total": final_loss}})
    return payloads


def test_synthetic_revolute_prefers_revolute_over_nested_zero_pitch_screw():
    payloads = _fit_candidates("revolute")
    selected, _, _ = _select_candidate(payloads, screw_margin=0.01)
    scores = {item["joint_parameters"]["joint_type"]: item["validation"]["total"] for item in payloads}
    assert abs(scores["revolute"] - scores["screw"]) < 1.0e-12
    assert selected["joint_parameters"]["joint_type"] == "revolute"


def test_synthetic_prismatic_prefers_prismatic():
    payloads = _fit_candidates("prismatic")
    selected, _, _ = _select_candidate(payloads, screw_margin=0.01)
    assert selected["joint_parameters"]["joint_type"] == "prismatic"


def test_synthetic_screw_with_pitch_prefers_screw():
    payloads = _fit_candidates("screw", pitch=0.35)
    selected, _, _ = _select_candidate(payloads, screw_margin=0.01)
    scores = {item["joint_parameters"]["joint_type"]: item["validation"]["total"] for item in payloads}
    assert scores["screw"] < 0.99 * scores["revolute"]
    assert selected["joint_parameters"]["joint_type"] == "screw"
