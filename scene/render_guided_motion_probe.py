"""Render-guided noising/denoising static/dynamic discovery.

This module deliberately optimizes only temporary per-cell variables.  It never
writes probe centers into the canonical Gaussian tensors and never uses DINO for
static/dynamic evidence.
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from gaussian_renderer import (
    accumulate_pixel_features_to_gaussians,
    render,
    render_feature_field,
)
from scene.material_assignment_field import MaterialAssignmentField
from utils.dynamic_loss_utils import build_occupied_cell_knn_pair_graph
from utils.eulerian_contrastive_support import assign_gaussians_to_cells
from utils.loss_utils import ssim
from utils.rigid_killing_motion import integrate_rigid_mode_exact
from utils.temporal_visibility_pretraining import _atomic_torch_save


STATIC, DYNAMIC, UNKNOWN = 0, 1, 2
STATE_NAMES = {STATIC: "static", DYNAMIC: "dynamic", UNKNOWN: "unknown"}


@dataclass
class RenderDenoiseConfig:
    probe_frames: list[int]
    cameras_per_probe: int
    min_support_cameras: int
    min_informative_frames: int
    num_seeds: int
    denoise_steps: int
    # Stage A is parameterized in physical world-units / second.
    velocity_noise_sigma: float
    velocity_lr: float
    max_speed: float
    max_probe_displacement: float
    static_speed_threshold: float
    dynamic_speed_threshold: float
    speed_temperature: float
    lambda_rgb: float
    lambda_ssim: float
    lambda_depth: float
    lambda_zero: float
    lambda_neighbor: float
    support_threshold: float
    seed_consistency_threshold: float
    view_consistency_threshold: float
    dynamic_threshold: float
    static_threshold: float
    confidence_threshold: float
    gate_num_seeds: int
    gate_steps: int
    gate_lr: float
    gate_temperature: float
    gate_noise_sigma: float
    lambda_gate_prior: float
    lambda_gate_neighbor: float
    lambda_gate_entropy: float


@dataclass
class RenderMotionEvidence:
    per_cell_dynamic_score: torch.Tensor
    per_cell_static_score: torch.Tensor
    per_cell_confidence: torch.Tensor
    per_cell_state: torch.Tensor
    # Velocity-native evidence used by the following Eulerian stage.
    per_cell_mean_velocity: torch.Tensor
    per_cell_mean_speed: torch.Tensor
    per_cell_speed_std: torch.Tensor
    per_cell_dynamic_persistence: torch.Tensor
    per_cell_static_persistence: torch.Tensor
    # Compatibility/debug displacement fields: never used for classification.
    per_cell_mean_delta: torch.Tensor
    per_cell_delta_std: torch.Tensor
    per_cell_seed_consistency: torch.Tensor
    per_cell_view_consistency: torch.Tensor
    per_cell_support_count: torch.Tensor
    per_cell_occluded_fraction: torch.Tensor
    per_cell_projected_fraction: torch.Tensor
    per_cell_visibility_ratio: torch.Tensor
    per_cell_final_dynamic_probability: torch.Tensor
    per_gaussian_dynamic_score: torch.Tensor
    per_gaussian_static_score: torch.Tensor
    per_gaussian_unknown_score: torch.Tensor
    per_gaussian_confidence: torch.Tensor
    per_gaussian_state: torch.Tensor
    per_gaussian_final_dynamic_probability: torch.Tensor
    cell_ids: torch.Tensor
    cell_centers: torch.Tensor
    point_counts: torch.Tensor
    per_frame: dict


def _safe_json(value):
    if torch.is_tensor(value):
        return value.detach().cpu().tolist()
    if isinstance(value, dict):
        return {str(key): _safe_json(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_safe_json(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _pseudo_huber(value: torch.Tensor, delta: float) -> torch.Tensor:
    scale = max(float(delta), 1.0e-8)
    return scale * scale * (torch.sqrt(1.0 + (value / scale).square()) - 1.0)


def _robust_unit(value: torch.Tensor, valid: torch.Tensor | None = None) -> torch.Tensor:
    x = torch.as_tensor(value).detach().float()
    mask = torch.isfinite(x)
    if valid is not None:
        mask &= torch.as_tensor(valid, device=x.device).bool()
    if not bool(mask.any()):
        return torch.zeros_like(x)
    finite = x[mask]
    low, high = torch.quantile(finite, torch.tensor([0.1, 0.9], device=x.device))
    return torch.where(mask, ((x - low) / (high - low).clamp_min(1.0e-8)).clamp(0.0, 1.0), torch.zeros_like(x))


def _parse_probe_frames(text: str, groups: list[tuple[int, float, list]], num_probe_frames: int) -> list[int]:
    frame_ids = [int(group[0]) for group in groups]
    if text.strip():
        requested = [int(item) for item in text.replace(" ", "").split(",") if item != ""]
        selected = [frame for frame in requested if frame in frame_ids and frame != frame_ids[0]]
        if selected:
            return sorted(dict.fromkeys(selected))
    num = max(1, int(num_probe_frames))
    candidates = np.rint(np.linspace(1, len(frame_ids) - 1, min(num, len(frame_ids) - 1))).astype(int)
    return sorted(dict.fromkeys(frame_ids[int(index)] for index in candidates))


def _selected_probe_frames(args, groups):
    return _parse_probe_frames(
        str(args.render_denoise_probe_frames),
        groups,
        int(args.render_denoise_num_probe_frames),
    )


def _camera_map(cameras):
    return {str(getattr(camera, "dynamic_camera_id", "")): camera for camera in cameras}


def _select_camera_keys(camera_map: dict, count: int) -> list[str]:
    keys = sorted(camera_map)
    if count <= 0 or count >= len(keys):
        return keys
    indices = np.rint(np.linspace(0, len(keys) - 1, count)).astype(int)
    return [keys[int(index)] for index in sorted(set(indices))]


def _make_config(args, probe_frames: list[int]) -> RenderDenoiseConfig:
    # New velocity-native arguments are preferred.  The old displacement names
    # are accepted only as compatibility fallbacks so existing configs still run.
    velocity_noise_sigma = float(getattr(
        args, "render_denoise_velocity_noise_sigma",
        getattr(args, "render_denoise_noise_sigma", 0.015),
    ))
    velocity_lr = float(getattr(
        args, "render_denoise_velocity_lr",
        getattr(args, "render_denoise_lr", 0.01),
    ))
    max_speed = float(getattr(
        args, "render_denoise_max_speed",
        getattr(args, "render_denoise_max_displacement", 0.12),
    ))
    max_probe_displacement = float(getattr(args, "render_denoise_max_probe_displacement", 0.12))
    static_speed_threshold = float(getattr(
        args, "render_denoise_static_speed_threshold", 0.40 * velocity_noise_sigma
    ))
    dynamic_speed_threshold = float(getattr(
        args, "render_denoise_dynamic_speed_threshold", 1.00 * velocity_noise_sigma
    ))
    speed_temperature = float(getattr(
        args, "render_denoise_speed_temperature", 0.30 * velocity_noise_sigma
    ))
    if static_speed_threshold >= dynamic_speed_threshold:
        raise ValueError("render-denoise static speed threshold must be < dynamic speed threshold")
    return RenderDenoiseConfig(
        probe_frames=probe_frames,
        cameras_per_probe=int(args.render_denoise_cameras_per_probe),
        min_support_cameras=int(args.render_denoise_min_support_cameras),
        min_informative_frames=int(args.render_denoise_min_informative_frames),
        num_seeds=int(args.render_denoise_num_seeds),
        denoise_steps=int(args.render_denoise_steps),
        velocity_noise_sigma=max(velocity_noise_sigma, 1.0e-8),
        velocity_lr=max(velocity_lr, 1.0e-8),
        max_speed=max(max_speed, 1.0e-8),
        max_probe_displacement=max(max_probe_displacement, 1.0e-8),
        static_speed_threshold=max(static_speed_threshold, 0.0),
        dynamic_speed_threshold=max(dynamic_speed_threshold, 1.0e-8),
        speed_temperature=max(speed_temperature, 1.0e-8),
        lambda_rgb=float(args.lambda_render_denoise_rgb),
        lambda_ssim=float(args.lambda_render_denoise_ssim),
        lambda_depth=float(args.lambda_render_denoise_depth),
        lambda_zero=float(args.lambda_render_denoise_zero),
        lambda_neighbor=float(args.lambda_render_denoise_neighbor),
        support_threshold=float(args.render_denoise_support_threshold),
        seed_consistency_threshold=float(args.render_denoise_seed_consistency_threshold),
        view_consistency_threshold=float(args.render_denoise_view_consistency_threshold),
        dynamic_threshold=float(args.render_denoise_dynamic_threshold),
        static_threshold=float(args.render_denoise_static_threshold),
        confidence_threshold=float(args.render_denoise_confidence_threshold),
        gate_num_seeds=int(args.render_gate_num_seeds),
        gate_steps=int(args.render_gate_steps),
        gate_lr=float(args.render_gate_lr),
        gate_temperature=float(args.render_gate_temperature),
        gate_noise_sigma=float(args.render_gate_noise_sigma),
        lambda_gate_prior=float(args.lambda_render_gate_prior),
        lambda_gate_neighbor=float(args.lambda_render_gate_neighbor),
        lambda_gate_entropy=float(args.lambda_render_gate_entropy),
    )


def build_probe_xyz(points: torch.Tensor, cell_ids: torch.Tensor, cell_delta: torch.Tensor) -> torch.Tensor:
    return points.detach() + cell_delta[cell_ids]


def _clamp_vector_norm_(value: torch.Tensor, max_norm: float) -> None:
    limit = max(float(max_norm), 1.0e-12)
    with torch.no_grad():
        norm = torch.linalg.norm(value, dim=-1, keepdim=True).clamp_min(1.0e-12)
        value.mul_(torch.clamp(limit / norm, max=1.0))


def _rgb_depth_loss(camera, package, args, opt_depth_weight=1.0):
    image = package["render"]
    target = camera.original_image.to(device=image.device, dtype=image.dtype)
    mask = torch.ones_like(image[:1])
    if getattr(camera, "alpha_mask", None) is not None:
        alpha = camera.alpha_mask.to(device=image.device, dtype=image.dtype)
        mask = alpha if alpha.ndim == 3 else alpha[None]
    rgb = (torch.abs(image - target) * mask).sum() / mask.expand_as(image).sum().clamp_min(1.0)
    dssim = 1.0 - ssim(image * mask, target * mask)
    depth = image.new_zeros(())
    if bool(getattr(camera, "depth_reliable", False)):
        target_depth = camera.invdepthmap.to(device=image.device, dtype=image.dtype)
        depth_mask = camera.depth_mask.to(device=image.device, dtype=image.dtype)
        if depth_mask.ndim == 2:
            depth_mask = depth_mask[None]
        valid = depth_mask * mask
        depth = (torch.abs(package["depth"] - target_depth) * valid).sum() / valid.sum().clamp_min(1.0)
    total = (
        float(args.lambda_render_denoise_rgb) * rgb
        + float(args.lambda_render_denoise_ssim) * dssim
        + float(args.lambda_render_denoise_depth) * float(opt_depth_weight) * depth
    )
    return total, {"rgb": float(rgb.detach()), "ssim": float(dssim.detach()), "depth": float(depth.detach())}


def _render_loss(camera, gaussians, pipe, background, xyz, rotation, args):
    with gaussians.render_probe_state(
        xyz=xyz,
        t=float(getattr(camera, "timestamp", 0.0)),
        rotation=rotation,
    ):
        package = render(
            camera,
            gaussians,
            pipe,
            background,
            use_trained_exp=False,
        )
    return _rgb_depth_loss(camera, package, args), package


@torch.no_grad()
def _cell_support(camera, gaussians, pipe, xyz, rotation, cell_ids, num_cells) -> dict[str, torch.Tensor]:
    """Occlusion-aware support, computed per Gaussian before cell aggregation.

    CUDA returns raw alpha support and exact front-to-back visible T*alpha mass
    per Gaussian.  We first form each Gaussian visibility ratio, then aggregate
    masses to cells.  This avoids interpreting within-cell splat overlap as a
    literal cell occlusion probability.
    """
    device = xyz.device
    height, width = int(camera.image_height), int(camera.image_width)
    ones = torch.ones((1, height, width), device=device, dtype=torch.float32)
    confidence = ones
    if getattr(camera, "alpha_mask", None) is not None:
        confidence = camera.alpha_mask.to(device=device, dtype=torch.float32)
        if confidence.ndim == 2:
            confidence = confidence[None]
        confidence = confidence[:1].clamp(0.0, 1.0)
    with gaussians.render_probe_state(
        xyz=xyz.detach(),
        t=float(getattr(camera, "timestamp", 0.0)),
        rotation=rotation.detach(),
    ):
        stats = accumulate_pixel_features_to_gaussians(
            camera, gaussians, pipe, ones, confidence,
            query_positions=xyz.detach(),
            query_rotations=rotation.detach(),
            group_ids=None, num_groups=None,
        )
    raw_i = stats["raw_support"].detach().float().reshape(-1)
    visible_i = stats["visible_mass"].detach().float().reshape(-1)
    valid_i = stats["valid_visible_mass"].detach().float().reshape(-1)
    ids = cell_ids.detach().long().reshape(-1)
    if ids.numel() != raw_i.numel():
        raise ValueError("cell_ids must match Gaussian count")
    valid_id = ids >= 0

    def scatter_sum(value):
        out = torch.zeros(int(num_cells), device=device, dtype=torch.float32)
        if bool(valid_id.any()):
            out.scatter_add_(0, ids[valid_id], value[valid_id])
        return out

    raw = scatter_sum(raw_i)
    visible = scatter_sum(visible_i)
    valid = scatter_sum(valid_i)
    ratio_i = (visible_i / raw_i.clamp_min(1.0e-8)).clamp(0.0, 1.0)
    projected_i = raw_i > 1.0e-8
    # Alpha-weighted fraction of projected Gaussian support that is strongly
    # transmittance-suppressed. This is a diagnostic, not a classification vote.
    low_vis_alpha = raw_i * (projected_i & (ratio_i < 0.10)).to(raw_i.dtype)
    low_visibility_fraction = scatter_sum(low_vis_alpha) / raw.clamp_min(1.0e-8)

    def robust_mass_score(value):
        positive = value[value > 0]
        scale = torch.quantile(positive, 0.75).clamp_min(1.0e-8) if positive.numel() else value.new_tensor(1.0)
        return (value / scale).clamp(0.0, 1.0)

    projected_score = robust_mass_score(raw)
    visible_score = robust_mass_score(visible)
    valid_score = robust_mass_score(valid)
    visibility_ratio = (visible / raw.clamp_min(1.0e-8)).clamp(0.0, 1.0)
    valid_fraction = (valid / visible.clamp_min(1.0e-8)).clamp(0.0, 1.0)
    return {
        "evidence_score": valid_score,
        "projected_score": projected_score,
        "visible_score": visible_score,
        "visibility_ratio": visibility_ratio,
        "valid_fraction": valid_fraction,
        "occluded_score": low_visibility_fraction.clamp(0.0, 1.0),
        "raw_support": raw,
        "visible_mass": visible,
        "valid_visible_mass": valid,
    }


def _neighbor_loss(cell_delta, cell_edges, delta_scale):
    if cell_edges.numel() == 0:
        return cell_delta.sum() * 0.0
    diff = torch.linalg.norm(cell_delta[cell_edges[:, 0]] - cell_delta[cell_edges[:, 1]], dim=-1)
    return _pseudo_huber(diff, delta_scale).mean()


def _seed_consistency(delta: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    norm = torch.linalg.norm(delta, dim=-1)
    w = weight.clamp_min(0.0)
    numerator = torch.linalg.norm((w[..., None] * delta).sum(0), dim=-1)
    denominator = (w * norm).sum(0).clamp_min(1.0e-8)
    return (numerator / denominator).clamp(0.0, 1.0)


def _view_consistency(corrections: torch.Tensor, support: torch.Tensor, min_cameras: int, threshold: float):
    supported = support >= float(threshold)
    count = supported.sum(0)
    norm = torch.linalg.norm(corrections, dim=-1).clamp_min(1.0e-8)
    unit = corrections / norm[..., None]
    summed = (unit * supported[..., None]).sum(0)
    consistency = torch.linalg.norm(summed, dim=-1) / count.clamp_min(1).to(corrections.dtype)
    consistency = torch.where(count >= int(min_cameras), consistency, torch.zeros_like(consistency))
    return consistency.clamp(0.0, 1.0), count


def _cell_state(dynamic_score, static_score, confidence, support_count, config):
    p_dynamic = torch.where(
        (dynamic_score + static_score) > 1.0e-8,
        dynamic_score / (dynamic_score + static_score).clamp_min(1.0e-8),
        torch.full_like(dynamic_score, 0.5),
    )
    state = torch.full_like(support_count.long(), UNKNOWN)
    supported = support_count >= int(config.min_informative_frames)
    state[(supported) & (p_dynamic >= config.dynamic_threshold) & (confidence >= config.confidence_threshold)] = DYNAMIC
    state[(supported) & (p_dynamic <= config.static_threshold) & (confidence >= config.confidence_threshold)] = STATIC
    return p_dynamic.clamp(0.0, 1.0), state


def _cell_to_gaussian(cell_value: torch.Tensor, cell_ids: torch.Tensor) -> torch.Tensor:
    return cell_value.to(device=cell_ids.device)[cell_ids]


def _frustum_lines(camera, scale):
    width, height = int(camera.image_width), int(camera.image_height)
    fx = width / (2.0 * math.tan(0.5 * float(camera.FoVx)))
    fy = height / (2.0 * math.tan(0.5 * float(camera.FoVy)))
    camera_r = camera.R.detach().cpu().numpy() if torch.is_tensor(camera.R) else np.asarray(camera.R)
    camera_t = camera.T.detach().cpu().numpy() if torch.is_tensor(camera.T) else np.asarray(camera.T)
    world_to_camera = np.eye(4, dtype=np.float64)
    world_to_camera[:3, :3] = camera_r.T
    world_to_camera[:3, 3] = camera_t
    camera_to_world = np.linalg.inv(world_to_camera)
    center = camera_to_world[:3, 3]
    rcw = camera_to_world[:3, :3]
    corners = []
    for u, v in ((0, 0), (width, 0), (width, height), (0, height)):
        ray = np.asarray([(u - 0.5 * width) / fx, (v - 0.5 * height) / fy, 1.0]) * scale
        corners.append(center + rcw @ ray)
    edges = [(center, corner) for corner in corners]
    edges += [(corners[i], corners[(i + 1) % 4]) for i in range(4)]
    return edges


def _write_3d_html(path, xyz, state, confidence, vectors, cameras, title, max_points):
    xyz = torch.as_tensor(xyz).detach().cpu().float()
    state = torch.as_tensor(state).detach().cpu().long()
    confidence = torch.as_tensor(confidence).detach().cpu().float()
    vectors = torch.as_tensor(vectors).detach().cpu().float()
    if xyz.shape[0] > int(max_points):
        ids = torch.linspace(0, xyz.shape[0] - 1, int(max_points)).round().long().unique()
        xyz, state, confidence, vectors = xyz[ids], state[ids], confidence[ids], vectors[ids]
    colors = ["#3b6ee8" if int(s) == STATIC else "#e84a3b" if int(s) == DYNAMIC else "#8a8a8a" for s in state]
    traces = [{
        "type": "scatter3d",
        "mode": "markers",
        "x": xyz[:, 0].tolist(),
        "y": xyz[:, 1].tolist(),
        "z": xyz[:, 2].tolist(),
        "marker": {"size": 3, "color": colors, "opacity": 0.75},
        "text": [f"{STATE_NAMES.get(int(s), 'unknown')} conf={float(c):.3f}" for s, c in zip(state, confidence)],
        "name": "points",
    }]
    vector_ids = torch.nonzero(torch.linalg.norm(vectors, dim=-1) > 1.0e-8, as_tuple=False).flatten()
    if vector_ids.numel() > 0:
        vector_ids = vector_ids[: min(300, int(vector_ids.numel()))]
        xs, ys, zs = [], [], []
        for point, vector in zip(xyz[vector_ids], vectors[vector_ids]):
            end = point + vector
            xs += [float(point[0]), float(end[0]), None]
            ys += [float(point[1]), float(end[1]), None]
            zs += [float(point[2]), float(end[2]), None]
        traces.append({"type": "scatter3d", "mode": "lines", "x": xs, "y": ys, "z": zs, "line": {"color": "#111111", "width": 3}, "name": "mean motion"})
    if cameras:
        diagonal = max(float(torch.linalg.norm(xyz.amax(0) - xyz.amin(0))), 1.0e-3)
        xs, ys, zs = [], [], []
        for camera in cameras:
            for start, end in _frustum_lines(camera, 0.08 * diagonal):
                xs += [float(start[0]), float(end[0]), None]
                ys += [float(start[1]), float(end[1]), None]
                zs += [float(start[2]), float(end[2]), None]
        traces.append({"type": "scatter3d", "mode": "lines", "x": xs, "y": ys, "z": zs, "line": {"color": "#000000", "width": 2}, "name": "cameras"})
    html = f"""<!doctype html><html><head><meta charset='utf-8'><script src='https://cdn.plot.ly/plotly-2.35.2.min.js'></script></head><body><h2>{title}</h2><div id='plot' style='width:100%;height:90vh'></div><script>Plotly.newPlot('plot',{json.dumps(traces)},{{scene:{{aspectmode:'data'}},legend:{{orientation:'h'}}}});</script></body></html>"""
    Path(path).write_text(html, encoding="utf-8")


def _to_u8_rgb(tensor):
    value = torch.as_tensor(tensor).detach().float().clamp(0, 1)
    if value.ndim == 3 and value.shape[0] in (1, 3):
        value = value.permute(1, 2, 0)
    if value.shape[-1] == 1:
        value = value.expand(-1, -1, 3)
    return (255.0 * value.cpu().numpy()).round().astype(np.uint8)


@torch.no_grad()
def _render_scalar_map(camera, gaussians, pipe, xyz, rotation, values):
    with gaussians.render_probe_state(
        xyz=xyz.detach(),
        t=float(getattr(camera, "timestamp", 0.0)),
        rotation=rotation.detach(),
    ):
        rendered = render_feature_field(
            camera,
            gaussians,
            pipe,
            values[:, None].to(device=xyz.device, dtype=torch.float32),
            bg_feature=torch.zeros(1, device=xyz.device),
            detach_geometry=True,
        )["render"][0]
    return rendered.clamp(0, 1)


@torch.no_grad()
def _write_stage_a_panels(out, frame_maps, probe_frames, camera_keys_by_frame, gaussians, pipe, background, points, rotations, cell_ids, frame_mean_delta, gaussian_dynamic, gaussian_unknown):
    panel_dir = out / "image_panels"
    panel_dir.mkdir(parents=True, exist_ok=True)
    for frame_pos, frame_id in enumerate(probe_frames):
        camera_map = frame_maps[frame_id]
        delta = frame_mean_delta[frame_pos].to(points.device, dtype=points.dtype)
        probe_xyz = points + delta[cell_ids]
        for key in camera_keys_by_frame[frame_id]:
            camera = camera_map[key]
            with gaussians.render_probe_state(xyz=points, t=float(getattr(camera, "timestamp", 0.0)), rotation=rotations):
                static_render = render(camera, gaussians, pipe, background)["render"]
            with gaussians.render_probe_state(xyz=probe_xyz, t=float(getattr(camera, "timestamp", 0.0)), rotation=rotations):
                probe_render = render(camera, gaussians, pipe, background)["render"]
            target = camera.original_image.to(device=points.device, dtype=points.dtype)
            before = (static_render - target).abs().mean(0, keepdim=True).expand(3, -1, -1)
            after = (probe_render - target).abs().mean(0, keepdim=True).expand(3, -1, -1)
            dynamic = _render_scalar_map(camera, gaussians, pipe, probe_xyz, rotations, gaussian_dynamic).expand(3, -1, -1)
            unknown = _render_scalar_map(camera, gaussians, pipe, probe_xyz, rotations, gaussian_unknown).expand(3, -1, -1)
            panel = np.concatenate([
                _to_u8_rgb(target),
                _to_u8_rgb(static_render),
                _to_u8_rgb(probe_render),
                _to_u8_rgb(before),
                _to_u8_rgb(after),
                _to_u8_rgb(dynamic),
                _to_u8_rgb(unknown),
            ], axis=1)
            cv2.imwrite(str(panel_dir / f"stageA_{frame_id:06d}_{key}.png"), cv2.cvtColor(panel, cv2.COLOR_RGB2BGR))


def run_render_guided_displacement_probe(scene, gaussians, pipe, dataset, args, *, log_fn=None):
    """Stage A: noisy per-cell *velocity* denoising and tri-state evidence.

    The optimized variable is velocity [world units / s]. For probe frame t,
    rendering uses x(t) = x0 + v_bar(t) * (t - t0).  v_bar is an average
    velocity over the canonical-to-probe horizon; it is not assumed to have a
    common direction across probe times.  Static/dynamic classification is
    therefore based on speed magnitude/persistence plus within-frame seed/view
    agreement, which is compatible with revolute motion and with the following
    Eulerian velocity optimization.
    """
    from utils.unified_dynamic_training import _complete_frame_groups, _ordered_dynamic_cameras

    cameras = _ordered_dynamic_cameras(scene.getTrainCameras())
    groups = _complete_frame_groups(cameras, args)
    if len(groups) < 2:
        raise RuntimeError("render-denoise discovery requires at least two complete dynamic frames")
    t0 = float(groups[0][1])
    group_by_frame = {
        int(frame): (rank, float(time_value), _camera_map(group_cameras))
        for rank, (frame, time_value, group_cameras) in enumerate(groups)
    }
    probe_frames = _selected_probe_frames(args, groups)
    config = _make_config(args, probe_frames)
    output = Path(scene.model_path) / "render_denoise_discovery"
    output.mkdir(parents=True, exist_ok=True)
    (output / "config.json").write_text(json.dumps(asdict(config), indent=2), encoding="utf-8")
    (output / "probe_frames.json").write_text(json.dumps({"frames": probe_frames}, indent=2), encoding="utf-8")

    device = gaussians.get_canonical_xyz.device
    dtype = gaussians.get_canonical_xyz.dtype
    points0 = gaussians.get_canonical_xyz.detach().clone()
    points = points0.detach()
    rotations = gaussians.get_rotation.detach().clone()
    grid = assign_gaussians_to_cells(
        points,
        int(args.curl_grid_target_cells),
        padding_fraction=float(args.curl_grid_padding),
        min_cells_per_axis=int(args.curl_grid_min_cells_per_axis),
    )
    cell_ids = grid.point_group_ids.to(device=device, dtype=torch.long)
    cell_centers = grid.cell_centers.to(device=device, dtype=dtype)
    point_counts = grid.point_counts.to(device=device)
    occupied = torch.ones(grid.num_cells, device=device, dtype=torch.bool)
    cell_graph = build_occupied_cell_knn_pair_graph(
        cell_centers, occupied,
        neighbors_per_cell=int(args.curl_material_neighbors),
        max_pairs=int(args.phase1_pair_max_pairs),
    )
    cell_edges = cell_graph.pairs.to(device)
    background = torch.tensor(
        [1.0, 1.0, 1.0] if bool(dataset.white_background) else [0.0, 0.0, 0.0],
        device=device, dtype=dtype,
    )

    frame_maps = {frame: item[2] for frame, item in group_by_frame.items()}
    camera_keys_by_frame = {
        frame: _select_camera_keys(frame_maps[frame], config.cameras_per_probe)
        for frame in probe_frames
    }
    per_frame = {}
    frame_dynamic, frame_static, frame_support = [], [], []
    frame_seed, frame_view = [], []
    frame_velocity, frame_speed, frame_speed_stability, frame_delta = [], [], [], []
    seed_rows = []

    for frame_id in tqdm(probe_frames, desc="Render-denoise velocity Stage A", unit="frame"):
        _, time_value, camera_map = group_by_frame[frame_id]
        dt = float(time_value - t0)
        if not math.isfinite(dt) or dt <= 1.0e-8:
            raise RuntimeError(f"probe frame {frame_id} has invalid dt={dt}")
        camera_keys = camera_keys_by_frame[frame_id]
        final_velocity_seeds, work_seeds, grad_seeds = [], [], []
        initial_speed_seeds = []
        support_count_seeds, view_consistency_seeds = [], []
        low_visibility_seeds, projected_fraction_seeds, visibility_ratio_seeds = [], [], []
        loss_start, loss_end = [], []
        # Geometry safety: speed is primary, but do not let a long probe horizon
        # teleport a cell by more than max_probe_displacement.
        effective_max_speed = min(config.max_speed, config.max_probe_displacement / dt)

        for seed in range(config.num_seeds):
            generator = torch.Generator(device=device).manual_seed(20_260_831 + 1000 * int(frame_id) + seed)
            velocity = torch.nn.Parameter(
                config.velocity_noise_sigma
                * torch.randn((grid.num_cells, 3), device=device, dtype=dtype, generator=generator)
            )
            _clamp_vector_norm_(velocity, effective_max_speed)
            optimizer = torch.optim.Adam([velocity], lr=config.velocity_lr)
            work = torch.zeros(grid.num_cells, device=device, dtype=dtype)
            gradient_energy = torch.zeros_like(work)
            initial_speed = torch.linalg.norm(velocity.detach(), dim=-1)
            initial_speed_seeds.append(initial_speed)
            first_loss = None
            last_loss = None

            for _step in range(config.denoise_steps):
                optimizer.zero_grad(set_to_none=True)
                delta = velocity * dt
                xyz = build_probe_xyz(points, cell_ids, delta)
                render_loss = velocity.sum() * 0.0
                for key in camera_keys:
                    (loss_view, _view_stats), _package = _render_loss(
                        camera_map[key], gaussians, pipe, background, xyz, rotations, args
                    )
                    render_loss = render_loss + loss_view / max(1, len(camera_keys))
                zero = _pseudo_huber(
                    torch.linalg.norm(velocity, dim=-1), config.velocity_noise_sigma
                ).mean()
                neighbor = _neighbor_loss(velocity, cell_edges, config.velocity_noise_sigma)
                loss = render_loss + config.lambda_zero * zero + config.lambda_neighbor * neighbor
                if first_loss is None:
                    first_loss = float(loss.detach())
                previous = velocity.detach().clone()
                loss.backward()
                grad = velocity.grad.detach().clone()
                optimizer.step()
                _clamp_vector_norm_(velocity, effective_max_speed)
                step_velocity = velocity.detach() - previous
                work += (-grad * step_velocity).sum(-1)
                gradient_energy += torch.linalg.norm(grad, dim=-1)
                last_loss = float(loss.detach())

            corrections, support_values = [], []
            low_visibility_values, projected_values, visibility_values = [], [], []
            for key in camera_keys:
                optimizer.zero_grad(set_to_none=True)
                delta = velocity * dt
                xyz = build_probe_xyz(points, cell_ids, delta)
                (loss_view, _view_stats), _package = _render_loss(
                    camera_map[key], gaussians, pipe, background, xyz, rotations, args
                )
                loss_view.backward()
                corrections.append((-velocity.grad.detach()).clone())
                support_stats = _cell_support(
                    camera_map[key], gaussians, pipe, xyz, rotations, cell_ids, grid.num_cells
                )
                support_values.append(support_stats["evidence_score"])
                low_visibility_values.append(support_stats["occluded_score"])
                projected_values.append(support_stats["projected_score"])
                visibility_values.append(support_stats["visibility_ratio"])

            corrections_t = torch.stack(corrections)
            support_t = torch.stack(support_values)
            low_visibility_t = torch.stack(low_visibility_values)
            projected_t = torch.stack(projected_values)
            visibility_t = torch.stack(visibility_values)
            view_consistency, support_count = _view_consistency(
                corrections_t, support_t,
                config.min_support_cameras, config.support_threshold,
            )
            final_velocity_seeds.append(velocity.detach().clone())
            work_seeds.append(work.detach().clone())
            grad_seeds.append(gradient_energy.detach().clone())
            support_count_seeds.append(support_count.detach().clone())
            view_consistency_seeds.append(view_consistency.detach().clone())
            low_visibility_seeds.append(low_visibility_t.mean(0).detach().clone())
            projected_fraction_seeds.append(projected_t.mean(0).detach().clone())
            visibility_ratio_seeds.append(visibility_t.mean(0).detach().clone())
            loss_start.append(float(first_loss if first_loss is not None else 0.0))
            loss_end.append(float(last_loss if last_loss is not None else 0.0))
            seed_rows.append({
                "frame": int(frame_id), "seed": int(seed), "dt": dt,
                "loss_start": loss_start[-1], "loss_end": loss_end[-1],
                "mean_initial_speed": float(initial_speed.mean()),
                "mean_final_speed": float(torch.linalg.norm(velocity.detach(), dim=-1).mean()),
                "mean_final_displacement": float(torch.linalg.norm(velocity.detach() * dt, dim=-1).mean()),
                "mean_positive_work": float(work.clamp_min(0).mean()),
                "mean_support_cameras": float(support_count.float().mean()),
                "mean_view_consistency": float(view_consistency.mean()),
                "mean_low_visibility_fraction": float(low_visibility_t.mean()),
                "mean_projected_fraction": float(projected_t.mean()),
                "mean_visibility_ratio": float(visibility_t.mean()),
            })

        final_velocity = torch.stack(final_velocity_seeds)       # [S,C,3]
        initial_speed = torch.stack(initial_speed_seeds)         # [S,C]
        work = torch.stack(work_seeds).clamp_min(0)
        gradient_energy = torch.stack(grad_seeds)
        support_count = torch.stack(support_count_seeds).float()
        view_consistency = torch.stack(view_consistency_seeds)
        low_visibility_fraction = torch.stack(low_visibility_seeds).mean(0).clamp(0.0, 1.0)
        projected_fraction = torch.stack(projected_fraction_seeds).mean(0).clamp(0.0, 1.0)
        visibility_ratio_frame = torch.stack(visibility_ratio_seeds).mean(0).clamp(0.0, 1.0)
        support_ok = support_count >= config.min_support_cameras
        support_fraction = (support_count / max(1, len(camera_keys))).mean(0).clamp(0.0, 1.0)
        seed_weight = support_fraction[None] * (0.50 + 0.50 * _robust_unit(work))
        seed_consistency = _seed_consistency(final_velocity, seed_weight)

        seed_speed = torch.linalg.norm(final_velocity, dim=-1)
        mean_velocity = final_velocity.mean(0)
        mean_speed = seed_speed.mean(0)  # mean norm; cannot cancel opposite seed directions
        speed_std = seed_speed.std(0, unbiased=False)
        velocity_std = final_velocity.std(0, unbiased=False).norm(dim=-1)
        mean_delta = mean_velocity * dt
        delta_std = velocity_std * dt
        initial_mean_speed = initial_speed.mean(0)
        shrink_score = ((initial_mean_speed - mean_speed) / config.velocity_noise_sigma).clamp(0.0, 1.0)
        speed_stability = torch.exp(-speed_std / max(config.static_speed_threshold, 1.0e-8)).clamp(0.0, 1.0)
        work_score = _robust_unit(work.mean(0))
        dynamic_speed_score = torch.sigmoid(
            (mean_speed - config.dynamic_speed_threshold) / config.speed_temperature
        )
        static_speed_score = torch.sigmoid(
            (config.static_speed_threshold - mean_speed) / config.speed_temperature
        )
        view_mean = view_consistency.mean(0).clamp(0.0, 1.0)
        supported_frame = support_ok.any(0)

        dynamic_score = (
            support_fraction
            * seed_consistency
            * view_mean
            * dynamic_speed_score
            * (0.50 + 0.50 * work_score)
        )
        # Static evidence is convergence of independently-noised speed toward
        # zero while the cell is actually observed. It intentionally does NOT
        # require cross-frame velocity direction consistency.
        static_score = (
            support_fraction
            * static_speed_score
            * speed_stability
            * (0.50 + 0.50 * shrink_score)
        )
        dynamic_score = torch.where(supported_frame, dynamic_score, torch.zeros_like(dynamic_score))
        static_score = torch.where(supported_frame, static_score, torch.zeros_like(static_score))

        frame_dynamic.append(dynamic_score)
        frame_static.append(static_score)
        frame_support.append(supported_frame.float())
        frame_seed.append(seed_consistency)
        frame_view.append(view_mean)
        frame_velocity.append(mean_velocity)
        frame_speed.append(mean_speed)
        frame_speed_stability.append(speed_stability)
        frame_delta.append(mean_delta)
        per_frame[int(frame_id)] = {
            "dt": dt,
            "dynamic_score": dynamic_score.detach().cpu(),
            "static_score": static_score.detach().cpu(),
            "support_fraction": support_fraction.detach().cpu(),
            "support_count": support_count.mean(0).detach().cpu(),
            "occluded_fraction": low_visibility_fraction.detach().cpu(),
            "projected_fraction": projected_fraction.detach().cpu(),
            "visibility_ratio": visibility_ratio_frame.detach().cpu(),
            "seed_consistency": seed_consistency.detach().cpu(),
            "view_consistency": view_mean.detach().cpu(),
            "mean_velocity": mean_velocity.detach().cpu(),
            "mean_speed": mean_speed.detach().cpu(),
            "speed_std": speed_std.detach().cpu(),
            "speed_stability": speed_stability.detach().cpu(),
            "mean_delta": mean_delta.detach().cpu(),
            "delta_std": delta_std.detach().cpu(),
            "gradient_energy": gradient_energy.mean(0).detach().cpu(),
            "positive_work": work.mean(0).detach().cpu(),
            "loss_start_mean": float(np.mean(loss_start)),
            "loss_end_mean": float(np.mean(loss_end)),
        }

    dynamic_stack = torch.stack(frame_dynamic)
    static_stack = torch.stack(frame_static)
    informative = torch.stack(frame_support) > 0
    informative_f = informative.to(dtype)
    informative_count = informative.sum(0)
    denominator = informative_count.to(dtype).clamp_min(1.0)

    # Correct masked temporal averaging: unsupported frames contribute neither
    # numerator nor denominator.
    cell_dynamic = (dynamic_stack * informative_f).sum(0) / denominator
    cell_static = (static_stack * informative_f).sum(0) / denominator
    seed_stack = torch.stack(frame_seed)
    view_stack = torch.stack(frame_view)
    seed_consistency = ((seed_stack * informative_f).sum(0) / denominator).clamp(0.0, 1.0)
    view_consistency = ((view_stack * informative_f).sum(0) / denominator).clamp(0.0, 1.0)

    speed_stack = torch.stack(frame_speed)
    stability_stack = torch.stack(frame_speed_stability)
    mean_speed = (speed_stack * informative_f).sum(0) / denominator
    speed_std = torch.sqrt(
        (((speed_stack - mean_speed[None]) ** 2) * informative_f).sum(0) / denominator
    )
    speed_stability = ((stability_stack * informative_f).sum(0) / denominator).clamp(0.0, 1.0)
    dynamic_persistence = (
        ((speed_stack >= config.dynamic_speed_threshold).to(dtype) * informative_f).sum(0) / denominator
    ).clamp(0.0, 1.0)
    static_persistence = (
        ((speed_stack <= config.static_speed_threshold).to(dtype) * informative_f).sum(0) / denominator
    ).clamp(0.0, 1.0)

    velocity_stack = torch.stack(frame_velocity)
    mean_velocity = (velocity_stack * informative_f[..., None]).sum(0) / denominator[:, None]
    # Compatibility only: average rendered displacement is not a classifier.
    delta_stack = torch.stack(frame_delta)
    mean_delta = (delta_stack * informative_f[..., None]).sum(0) / denominator[:, None]
    delta_std = torch.sqrt(
        (((delta_stack - mean_delta[None]) ** 2).sum(-1) * informative_f).sum(0) / denominator
    )

    support_fraction = informative_count.to(dtype) / max(1, len(probe_frames))
    occluded_fraction = torch.stack([
        per_frame[int(frame)]["occluded_fraction"].to(device) for frame in probe_frames
    ]).mean(0)
    projected_fraction = torch.stack([
        per_frame[int(frame)]["projected_fraction"].to(device) for frame in probe_frames
    ]).mean(0)
    visibility_ratio = torch.stack([
        per_frame[int(frame)]["visibility_ratio"].to(device) for frame in probe_frames
    ]).mean(0)

    # Temporal persistence strengthens, but does not hard-gate, the two modes.
    cell_dynamic = cell_dynamic * (0.50 + 0.50 * dynamic_persistence)
    cell_static = cell_static * (0.50 + 0.50 * static_persistence)
    p_dynamic = torch.where(
        (cell_dynamic + cell_static) > 1.0e-8,
        cell_dynamic / (cell_dynamic + cell_static).clamp_min(1.0e-8),
        torch.full_like(cell_dynamic, 0.5),
    )

    dynamic_ok = (
        (seed_consistency >= float(config.seed_consistency_threshold))
        & (view_consistency >= float(config.view_consistency_threshold))
    )
    dynamic_consistency = (
        seed_consistency * view_consistency * (0.50 + 0.50 * dynamic_persistence)
    ).clamp(0.0, 1.0)
    dynamic_consistency = torch.where(dynamic_ok, dynamic_consistency, torch.zeros_like(dynamic_consistency))
    static_consistency = (speed_stability * (0.50 + 0.50 * static_persistence)).clamp(0.0, 1.0)
    margin_confidence = (2.0 * torch.abs(p_dynamic - 0.5)).clamp(0.0, 1.0)
    mode_consistency = torch.where(p_dynamic >= 0.5, dynamic_consistency, static_consistency)
    confidence = (margin_confidence * support_fraction * mode_consistency).clamp(0.0, 1.0)
    p_dynamic, state = _cell_state(cell_dynamic, cell_static, confidence, informative_count, config)
    unknown_score = torch.where(state == UNKNOWN, torch.ones_like(p_dynamic), 1.0 - confidence).clamp(0.0, 1.0)

    evidence = RenderMotionEvidence(
        per_cell_dynamic_score=cell_dynamic.detach().cpu(),
        per_cell_static_score=cell_static.detach().cpu(),
        per_cell_confidence=confidence.detach().cpu(),
        per_cell_state=state.detach().cpu(),
        per_cell_mean_velocity=mean_velocity.detach().cpu(),
        per_cell_mean_speed=mean_speed.detach().cpu(),
        per_cell_speed_std=speed_std.detach().cpu(),
        per_cell_dynamic_persistence=dynamic_persistence.detach().cpu(),
        per_cell_static_persistence=static_persistence.detach().cpu(),
        per_cell_mean_delta=mean_delta.detach().cpu(),
        per_cell_delta_std=delta_std.detach().cpu(),
        per_cell_seed_consistency=seed_consistency.detach().cpu(),
        per_cell_view_consistency=view_consistency.detach().cpu(),
        per_cell_support_count=informative_count.detach().cpu(),
        per_cell_occluded_fraction=occluded_fraction.detach().cpu(),
        per_cell_projected_fraction=projected_fraction.detach().cpu(),
        per_cell_visibility_ratio=visibility_ratio.detach().cpu(),
        per_cell_final_dynamic_probability=p_dynamic.detach().cpu(),
        per_gaussian_dynamic_score=_cell_to_gaussian(cell_dynamic, cell_ids).detach().cpu(),
        per_gaussian_static_score=_cell_to_gaussian(cell_static, cell_ids).detach().cpu(),
        per_gaussian_unknown_score=_cell_to_gaussian(unknown_score, cell_ids).detach().cpu(),
        per_gaussian_confidence=_cell_to_gaussian(confidence, cell_ids).detach().cpu(),
        per_gaussian_state=_cell_to_gaussian(state, cell_ids).detach().cpu(),
        per_gaussian_final_dynamic_probability=_cell_to_gaussian(p_dynamic, cell_ids).detach().cpu(),
        cell_ids=cell_ids.detach().cpu(),
        cell_centers=cell_centers.detach().cpu(),
        point_counts=point_counts.detach().cpu(),
        per_frame=per_frame,
    )
    payload = {
        "schema_version": 2,
        "stage": "A_velocity_denoise",
        "units": {"velocity": "world_units_per_second", "time": "seconds"},
        "ground_truth_used_as_algorithm_input": False,
        "dino_used_for_static_dynamic_discovery": False,
        **{key: value for key, value in asdict(evidence).items() if key != "per_frame"},
        "per_frame": per_frame,
        "seed_rows": seed_rows,
    }
    _atomic_torch_save(payload, output / "per_cell_evidence.pt")
    _atomic_torch_save({
        "schema_version": 2,
        "dynamic_score": evidence.per_gaussian_dynamic_score,
        "static_score": evidence.per_gaussian_static_score,
        "unknown_score": evidence.per_gaussian_unknown_score,
        "confidence": evidence.per_gaussian_confidence,
        "state": evidence.per_gaussian_state,
        "final_dynamic_probability": evidence.per_gaussian_final_dynamic_probability,
        "cell_ids": evidence.cell_ids,
        "per_cell_mean_velocity": evidence.per_cell_mean_velocity,
        "per_cell_mean_speed": evidence.per_cell_mean_speed,
    }, output / "per_gaussian_evidence.pt")

    state_counts = {name: int((state == code).sum()) for code, name in STATE_NAMES.items()}
    summary = {
        "schema_version": 2,
        "representation": "velocity",
        "velocity_units": "world_units_per_second",
        "num_probe_frames": len(probe_frames),
        "probe_frames": probe_frames,
        "num_cells": int(grid.num_cells),
        "num_gaussians": int(points.shape[0]),
        "state_counts": state_counts,
        "mean_seed_consistency": float(seed_consistency.mean()),
        "mean_view_consistency": float(view_consistency.mean()),
        "mean_support_fraction": float(support_fraction.mean()),
        "mean_low_visibility_fraction": float(occluded_fraction.mean()),
        "mean_projected_fraction": float(projected_fraction.mean()),
        "mean_visibility_ratio": float(visibility_ratio.mean()),
        "mean_speed": float(mean_speed.mean()),
        "median_speed": float(mean_speed.median()),
        "mean_dynamic_persistence": float(dynamic_persistence.mean()),
        "mean_static_persistence": float(static_persistence.mean()),
        "mean_positive_work": float(torch.stack([
            per_frame[int(frame)]["positive_work"].to(device) for frame in probe_frames
        ]).mean()),
        "canonical_max_abs_change": float((gaussians.get_canonical_xyz.detach() - points0).abs().max()),
        "seed_rows": seed_rows,
        "ground_truth_used_as_algorithm_input": False,
    }
    (output / "summary.json").write_text(json.dumps(_safe_json(summary), indent=2), encoding="utf-8")

    _write_3d_html(
        output / "render_denoise_stageA_cells.html",
        cell_centers.cpu(), state.cpu(), confidence.cpu(), mean_velocity.cpu(),
        [frame_maps[probe_frames[0]][key] for key in camera_keys_by_frame[probe_frames[0]]],
        "Render-denoise Stage A cell velocity evidence",
        int(args.transported_html_max_points),
    )
    gaussian_state = evidence.per_gaussian_state
    _write_3d_html(
        output / "render_denoise_final_gaussians.html",
        points.cpu(), gaussian_state, evidence.per_gaussian_confidence,
        _cell_to_gaussian(mean_velocity.detach(), cell_ids).detach().cpu(),
        [frame_maps[probe_frames[0]][key] for key in camera_keys_by_frame[probe_frames[0]]],
        "Render-denoise Stage A Gaussian velocity evidence",
        int(args.transported_html_max_points),
    )
    _write_stage_a_panels(
        output, frame_maps, probe_frames, camera_keys_by_frame,
        gaussians, pipe, background, points, rotations, cell_ids,
        torch.stack(frame_delta).detach().cpu(),
        evidence.per_gaussian_dynamic_score.to(device),
        evidence.per_gaussian_unknown_score.to(device),
    )
    if log_fn is not None:
        log_fn("render_denoise", 0, {
            "discovery/render_denoise/num_probe_frames": len(probe_frames),
            "discovery/render_denoise/num_cells": int(grid.num_cells),
            "discovery/render_denoise/dynamic_cells": state_counts["dynamic"],
            "discovery/render_denoise/static_cells": state_counts["static"],
            "discovery/render_denoise/unknown_cells": state_counts["unknown"],
            "discovery/render_denoise/mean_seed_consistency": summary["mean_seed_consistency"],
            "discovery/render_denoise/mean_view_consistency": summary["mean_view_consistency"],
            "discovery/render_denoise/mean_support_fraction": summary["mean_support_fraction"],
            "discovery/render_denoise/mean_low_visibility_fraction": summary["mean_low_visibility_fraction"],
            "discovery/render_denoise/mean_speed": summary["mean_speed"],
            "discovery/render_denoise/mean_dynamic_persistence": summary["mean_dynamic_persistence"],
        })
    return {"config": config, "evidence": evidence, "output": output, "summary": summary}


# Explicit name for new callers; keep the old name for compatibility.
run_render_guided_velocity_probe = run_render_guided_displacement_probe


def _coordinates(field, frame_times, device, dtype):
    speed = torch.as_tensor(field["temporal_speed"], device=device, dtype=dtype).reshape(-1)
    time = torch.as_tensor(frame_times, device=device, dtype=dtype).reshape(-1)
    return torch.cat((torch.zeros(1, device=device, dtype=dtype), torch.cumsum(speed * torch.diff(time), 0)))


def _slerp(q0, q1, amount):
    q0 = F.normalize(q0, dim=-1)
    q1 = F.normalize(q1, dim=-1)
    dot = (q0 * q1).sum(-1, keepdim=True)
    q1 = torch.where(dot < 0, -q1, q1)
    dot = dot.abs().clamp(-1.0, 1.0)
    theta = torch.acos(dot)
    sin_theta = torch.sin(theta)
    amount = amount.reshape(-1, 1).to(q0)
    lerp = F.normalize((1.0 - amount) * q0 + amount * q1, dim=-1)
    s0 = torch.sin((1.0 - amount) * theta) / sin_theta.clamp_min(1.0e-6)
    s1 = torch.sin(amount * theta) / sin_theta.clamp_min(1.0e-6)
    return torch.where(sin_theta > 1.0e-5, F.normalize(s0 * q0 + s1 * q1, dim=-1), lerp)


def _prior_from_stage_a(render_state, cell_ids, num_cells, device, dtype):
    if render_state is None:
        probability = torch.full((num_cells,), 0.5, device=device, dtype=dtype)
        confidence = torch.zeros(num_cells, device=device, dtype=dtype)
        return probability, confidence
    cell_probability = torch.as_tensor(
        render_state.get("per_cell_final_dynamic_probability", torch.full((num_cells,), 0.5)),
        device=device,
        dtype=dtype,
    ).reshape(-1)
    cell_confidence = torch.as_tensor(
        render_state.get("per_cell_confidence", torch.zeros(num_cells)),
        device=device,
        dtype=dtype,
    ).reshape(-1)
    if cell_probability.numel() != num_cells:
        gaussian_probability = torch.as_tensor(render_state["final_dynamic_probability"], device=device, dtype=dtype)
        cell_probability = torch.zeros(num_cells, device=device, dtype=dtype)
        cell_confidence = torch.zeros(num_cells, device=device, dtype=dtype)
        count = torch.zeros(num_cells, device=device, dtype=dtype)
        cell_probability.index_add_(0, cell_ids, gaussian_probability)
        cell_confidence.index_add_(0, cell_ids, torch.as_tensor(render_state["confidence"], device=device, dtype=dtype))
        count.index_add_(0, cell_ids, torch.ones_like(gaussian_probability))
        cell_probability /= count.clamp_min(1.0)
        cell_confidence /= count.clamp_min(1.0)
    return cell_probability.clamp(0.0, 1.0), cell_confidence.clamp(0.0, 1.0)


def _gate_render_state(points, rotations, field, coordinate, cell_ids, logits, temperature):
    rigid = integrate_rigid_mode_exact(
        points,
        field["mode"].to(device=points.device, dtype=points.dtype),
        coordinate.to(points),
        orientations=rotations,
    )
    amount = torch.sigmoid(logits / max(float(temperature), 1.0e-6))
    per_point = amount[cell_ids]
    xyz = points + per_point[:, None] * (rigid["positions"] - points)
    orientation = _slerp(rotations, rigid["orientations"], per_point)
    return xyz, orientation, amount


def run_static_rigid_gate_probe(scene, gaussians, pipe, dataset, args, fields, frame_times, *, log_fn=None):
    """Stage B: noisy static-vs-rigid gate denoising after rigid fields exist."""
    from utils.unified_dynamic_training import _complete_frame_groups, _ordered_dynamic_cameras

    if not fields:
        return None
    cameras = _ordered_dynamic_cameras(scene.getTrainCameras())
    groups = _complete_frame_groups(cameras, args)
    group_by_frame = {int(frame): (rank, time_value, _camera_map(group_cameras)) for rank, (frame, time_value, group_cameras) in enumerate(groups)}
    probe_frames = _selected_probe_frames(args, groups)
    output = Path(scene.model_path) / "render_denoise_discovery"
    output.mkdir(parents=True, exist_ok=True)
    stage_a = getattr(gaussians, "render_discovery_state", None)
    device = gaussians.get_canonical_xyz.device
    dtype = gaussians.get_canonical_xyz.dtype
    points = gaussians.get_canonical_xyz.detach()
    rotations = gaussians.get_rotation.detach().clone()
    if stage_a is not None and "cell_ids" in stage_a:
        cell_ids = torch.as_tensor(stage_a["cell_ids"], device=device, dtype=torch.long)
        cell_centers = torch.as_tensor(stage_a["cell_centers"], device=device, dtype=dtype)
        point_counts = torch.as_tensor(stage_a["point_counts"], device=device)
        num_cells = int(cell_centers.shape[0])
    else:
        grid = assign_gaussians_to_cells(
            points,
            int(args.curl_grid_target_cells),
            padding_fraction=float(args.curl_grid_padding),
            min_cells_per_axis=int(args.curl_grid_min_cells_per_axis),
        )
        cell_ids = grid.point_group_ids.to(device=device, dtype=torch.long)
        cell_centers = grid.cell_centers.to(device=device, dtype=dtype)
        point_counts = grid.point_counts.to(device=device)
        num_cells = grid.num_cells
    occupied = torch.ones(num_cells, device=device, dtype=torch.bool)
    cell_edges = build_occupied_cell_knn_pair_graph(
        cell_centers,
        occupied,
        neighbors_per_cell=int(args.curl_material_neighbors),
        max_pairs=int(args.phase1_pair_max_pairs),
    ).pairs.to(device)
    prior_probability, prior_confidence = _prior_from_stage_a(stage_a, cell_ids, num_cells, device, dtype)
    prior_logit = torch.logit(prior_probability.clamp(0.05, 0.95))
    background = torch.tensor(
        [1.0, 1.0, 1.0] if bool(dataset.white_background) else [0.0, 0.0, 0.0],
        device=device,
        dtype=dtype,
    )
    camera_keys_by_frame = {
        frame: _select_camera_keys(group_by_frame[frame][2], int(args.render_denoise_cameras_per_probe))
        for frame in probe_frames
    }
    field_probabilities = []
    gate_rows = []
    for field_index, field in enumerate(fields):
        coordinates = _coordinates(field, frame_times, device, dtype)
        seed_probs = []
        for seed in range(int(args.render_gate_num_seeds)):
            generator = torch.Generator(device=device).manual_seed(20_260_901 + 1000 * field_index + seed)
            logits = torch.nn.Parameter(prior_logit + float(args.render_gate_noise_sigma) * torch.randn(num_cells, device=device, dtype=dtype, generator=generator))
            optimizer = torch.optim.Adam([logits], lr=float(args.render_gate_lr))
            first_loss = None
            last_loss = None
            progress = range(int(args.render_gate_steps))
            for step in progress:
                optimizer.zero_grad(set_to_none=True)
                total = logits.sum() * 0.0
                for frame in probe_frames:
                    frame_rank = int(group_by_frame[frame][0])
                    xyz, orientation, gate = _gate_render_state(
                        points, rotations, field, coordinates[frame_rank], cell_ids,
                        logits, float(args.render_gate_temperature),
                    )
                    camera_map = group_by_frame[frame][2]
                    for key in camera_keys_by_frame[frame]:
                        (loss_view, _stats), _package = _render_loss(
                            camera_map[key], gaussians, pipe, background, xyz, orientation, args
                        )
                        total = total + loss_view / max(1, len(probe_frames) * len(camera_keys_by_frame[frame]))
                gate = torch.sigmoid(logits / max(float(args.render_gate_temperature), 1.0e-6))
                prior_bce = F.binary_cross_entropy(gate.clamp(1.0e-6, 1.0 - 1.0e-6), prior_probability, reduction="none")
                prior = (prior_confidence * prior_bce).sum() / prior_confidence.sum().clamp_min(1.0e-8)
                neighbor = _neighbor_loss(gate[:, None], cell_edges, 0.25)
                entropy = -(gate * torch.log(gate.clamp_min(1.0e-8)) + (1.0 - gate) * torch.log((1.0 - gate).clamp_min(1.0e-8))).mean()
                ramp = float(step + 1) / max(1, int(args.render_gate_steps))
                loss = (
                    total
                    + float(args.lambda_render_gate_prior) * prior
                    + float(args.lambda_render_gate_neighbor) * neighbor
                    + float(args.lambda_render_gate_entropy) * ramp * entropy
                )
                if first_loss is None:
                    first_loss = float(loss.detach())
                if not bool(torch.isfinite(loss)):
                    raise RuntimeError("render static-vs-rigid gate produced non-finite loss")
                loss.backward()
                optimizer.step()
                last_loss = float(loss.detach())
            probability = torch.sigmoid(logits.detach() / max(float(args.render_gate_temperature), 1.0e-6))
            seed_probs.append(probability)
            gate_rows.append({
                "field": int(field_index),
                "seed": int(seed),
                "loss_start": float(first_loss if first_loss is not None else 0.0),
                "loss_end": float(last_loss if last_loss is not None else 0.0),
                "mean_probability": float(probability.mean()),
            })
        field_probabilities.append(torch.stack(seed_probs))
    gate_by_field = torch.stack(field_probabilities)
    gate_mean_by_field = gate_by_field.mean(1)
    gate_mean = gate_mean_by_field.max(0).values
    gate_std = gate_by_field.reshape(-1, num_cells).std(0, unbiased=False)
    stage_a_dynamic = torch.as_tensor(stage_a.get("per_cell_dynamic_score", torch.zeros(num_cells)), device=device, dtype=dtype) if stage_a is not None else torch.zeros(num_cells, device=device, dtype=dtype)
    stage_a_static = torch.as_tensor(stage_a.get("per_cell_static_score", torch.zeros(num_cells)), device=device, dtype=dtype) if stage_a is not None else torch.zeros(num_cells, device=device, dtype=dtype)
    support_count = torch.as_tensor(stage_a.get("per_cell_support_count", torch.zeros(num_cells)), device=device) if stage_a is not None else torch.zeros(num_cells, device=device)
    confidence = ((gate_mean - 0.5).abs() * 2.0 * (1.0 - gate_std.clamp(0, 1))).clamp(0.0, 1.0)
    p_dynamic, state = _cell_state(stage_a_dynamic + gate_mean, stage_a_static + (1.0 - gate_mean), confidence, support_count, _make_config(args, probe_frames))
    unknown_score = torch.where(state == UNKNOWN, torch.ones_like(p_dynamic), 1.0 - confidence).clamp(0.0, 1.0)
    per_gaussian = {
        "dynamic_score": _cell_to_gaussian(stage_a_dynamic + gate_mean, cell_ids).detach().cpu(),
        "static_score": _cell_to_gaussian(stage_a_static + (1.0 - gate_mean), cell_ids).detach().cpu(),
        "unknown_score": _cell_to_gaussian(unknown_score, cell_ids).detach().cpu(),
        "confidence": _cell_to_gaussian(confidence, cell_ids).detach().cpu(),
        "state": _cell_to_gaussian(state, cell_ids).detach().cpu(),
        "final_dynamic_probability": _cell_to_gaussian(p_dynamic, cell_ids).detach().cpu(),
    }
    payload = {
        "schema_version": 1,
        "stage": "B_static_vs_rigid_gate",
        "gate_mean": gate_mean.detach().cpu(),
        "gate_std": gate_std.detach().cpu(),
        "final_dynamic_probability": p_dynamic.detach().cpu(),
        "confidence": confidence.detach().cpu(),
        "state": state.detach().cpu(),
        "cell_ids": cell_ids.detach().cpu(),
        "cell_centers": cell_centers.detach().cpu(),
        "point_counts": point_counts.detach().cpu(),
        "per_gaussian": per_gaussian,
        "gate_rows": gate_rows,
        "ground_truth_used_as_algorithm_input": False,
        "dino_used_for_static_dynamic_discovery": False,
    }
    _atomic_torch_save(payload, output / "stageB_static_rigid_gate.pt")
    _atomic_torch_save({"schema_version": 1, **per_gaussian, "canonical_xyz": points.detach().cpu()}, output / "per_gaussian_evidence.pt")
    _write_3d_html(
        output / "render_denoise_stageB_static_rigid.html",
        cell_centers.detach().cpu(),
        state.detach().cpu(),
        confidence.detach().cpu(),
        torch.zeros_like(cell_centers).detach().cpu(),
        [group_by_frame[probe_frames[0]][2][key] for key in camera_keys_by_frame[probe_frames[0]]],
        "Render-denoise Stage B static-vs-rigid gate",
        int(args.transported_html_max_points),
    )
    state_counts = {name: int((state == code).sum()) for code, name in STATE_NAMES.items()}
    summary_path = output / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8")) if summary_path.exists() else {}
    summary.update({
        "stageB_dynamic_cells": state_counts["dynamic"],
        "stageB_static_cells": state_counts["static"],
        "stageB_unknown_cells": state_counts["unknown"],
        "render_gate_mean_probability": float(gate_mean.mean()),
        "render_gate_mean_confidence": float(confidence.mean()),
        "render_gate_unknown_fraction": float((state == UNKNOWN).float().mean()),
        "stageB_gate_rows": gate_rows,
    })
    summary_path.write_text(json.dumps(_safe_json(summary), indent=2), encoding="utf-8")
    if log_fn is not None:
        log_fn("render_gate", 0, {
            "discovery/render_gate/mean_probability": float(gate_mean.mean()),
            "discovery/render_gate/mean_confidence": float(confidence.mean()),
            "discovery/render_gate/unknown_fraction": float((state == UNKNOWN).float().mean()),
        })
    return payload


def material_assignment_from_render_evidence(gaussians, evidence: dict, args):
    device = gaussians.get_canonical_xyz.device
    dynamic = torch.as_tensor(evidence["dynamic_score"], device=device, dtype=torch.float32).reshape(-1)
    static = torch.as_tensor(evidence["static_score"], device=device, dtype=torch.float32).reshape(-1)
    unknown = torch.as_tensor(evidence["unknown_score"], device=device, dtype=torch.float32).reshape(-1)
    probability = torch.as_tensor(evidence.get("final_dynamic_probability", dynamic), device=device, dtype=torch.float32).reshape(-1)
    state = torch.as_tensor(evidence["state"], device=device).long().reshape(-1)
    if dynamic.numel() != gaussians.get_canonical_xyz.shape[0]:
        raise ValueError("render evidence must match compact Gaussian count")
    hard_dynamic = (state == DYNAMIC) | ((probability >= float(args.render_denoise_dynamic_threshold)) & (unknown < 0.5))
    assignment = MaterialAssignmentField(
        hard_dynamic,
        num_moving_parts=int(args.num_moving_parts),
        initial_dynamic_score=probability.clamp(0.0, 1.0),
        initial_static_score=static.clamp(0.0, 1.0),
        initial_unknown_score=unknown.clamp(0.0, 1.0),
        evidence_probability_floor=float(args.eulerian_assignment_init_floor),
        evidence_probability_ceiling=float(args.eulerian_assignment_init_ceiling),
    )
    return assignment
