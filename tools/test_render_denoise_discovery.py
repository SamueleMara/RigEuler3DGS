#!/usr/bin/env python3
"""Standalone smoke test for render-guided 3DGS static/dynamic discovery.

This script intentionally does NOT call train.py and does NOT depend on the
training CLI. It tests two things in order:

1) the new native CUDA reverse-raster visibility/responsibility statistics
   raw_support = sum(alpha)
   visible_mass = sum(T * alpha)
   valid_visible_mass = sum(confidence * T * alpha)

2) a small Stage-A noisy per-cell VELOCITY denoising experiment driven
   only by real multiview RGB/depth rendering loss.

It loads an already-trained canonical 3DGS iteration from --static-model-dir,
expands the fixed COLMAP cameras over --root-scene/rgb, keeps canonical Gaussian
parameters frozen, and optimizes only temporary per-cell velocity tensors in world-units/second.

No DINO or optical flow is used.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch


def _find_repo_root() -> Path:
    here = Path(__file__).resolve().parent
    candidates = [Path.cwd().resolve(), here, here.parent]
    for root in candidates:
        if (root / "scene").is_dir() and (root / "gaussian_renderer").is_dir():
            return root
    raise RuntimeError(
        "Run this script from the RigEuler3DGS repository root, or place it in "
        "RigEuler3DGS/tools/."
    )


REPO_ROOT = _find_repo_root()
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from diff_gaussian_rasterization import GaussianRasterizer, _C  # noqa: E402
from gaussian_renderer import render, _make_rasterizer, _geometry_inputs  # noqa: E402
from scene import Scene  # noqa: E402
from scene.gaussian_model import GaussianModel  # noqa: E402
from utils.eulerian_contrastive_support import assign_gaussians_to_cells  # noqa: E402
from utils.loss_utils import ssim  # noqa: E402


STATIC = 0
DYNAMIC = 1
UNKNOWN = 2
STATE_NAMES = {STATIC: "static", DYNAMIC: "dynamic", UNKNOWN: "unknown"}


def _load_cfg_namespace(model_dir: Path):
    cfg_path = model_dir / "cfg_args"
    if not cfg_path.is_file():
        return None
    text = cfg_path.read_text(encoding="utf-8").strip()
    try:
        # GraphDeco writes cfg_args as Namespace(...). This is a local trusted file.
        return eval(text, {"__builtins__": {}, "Namespace": argparse.Namespace}, {})
    except Exception as exc:
        print(f"[cfg] WARNING: could not parse {cfg_path}: {exc}")
        return None


def _cfg_value(cfg, name, default):
    return getattr(cfg, name, default) if cfg is not None else default


@contextmanager
def temporary_geometry(gaussians: GaussianModel, xyz: torch.Tensor, rotation: torch.Tensor):
    """Render temporary positions without requiring dynamic motion initialization."""
    previous = (
        gaussians._current_xyz,
        gaussians._current_rotation,
        gaussians._current_velocity,
        gaussians._current_cell_id,
        gaussians._current_time,
    )
    gaussians._current_xyz = xyz
    # GaussianModel.get_rotation applies normalization; normalized quaternions are fine here.
    gaussians._current_rotation = rotation
    gaussians._current_velocity = None
    gaussians._current_cell_id = torch.empty(0, device=xyz.device, dtype=torch.long)
    gaussians._current_time = None
    try:
        yield
    finally:
        (
            gaussians._current_xyz,
            gaussians._current_rotation,
            gaussians._current_velocity,
            gaussians._current_cell_id,
            gaussians._current_time,
        ) = previous


def _parse_frames(text: str) -> list[int]:
    return [int(x) for x in text.replace(" ", "").split(",") if x != ""]


def _ordered_complete_frame_groups(cameras):
    camera_ids = sorted({str(getattr(c, "dynamic_camera_id", "")) for c in cameras})
    grouped: dict[int, list] = {}
    for camera in cameras:
        grouped.setdefault(int(getattr(camera, "frame_id", 0)), []).append(camera)
    result = []
    for frame_id in sorted(grouped):
        cams = sorted(grouped[frame_id], key=lambda c: str(getattr(c, "dynamic_camera_id", "")))
        present = [str(getattr(c, "dynamic_camera_id", "")) for c in cams]
        if present != camera_ids:
            continue
        result.append((frame_id, cams))
    return result


def _select_cameras(cameras: list, count: int):
    if count <= 0 or count >= len(cameras):
        return cameras
    ids = np.rint(np.linspace(0, len(cameras) - 1, count)).astype(int)
    return [cameras[i] for i in sorted(set(int(x) for x in ids))]


def _pseudo_huber(x: torch.Tensor, delta: float):
    d = max(float(delta), 1e-8)
    return d * d * (torch.sqrt(1.0 + (x / d).square()) - 1.0)


def _robust_unit(x: torch.Tensor):
    x = x.detach().float()
    finite = torch.isfinite(x)
    if not bool(finite.any()):
        return torch.zeros_like(x)
    vals = x[finite]
    lo = torch.quantile(vals, 0.10)
    hi = torch.quantile(vals, 0.90)
    return torch.where(finite, ((x - lo) / (hi - lo).clamp_min(1e-8)).clamp(0, 1), 0.0)


def _render_loss(camera, gaussians, pipe, background, xyz, rotation, args):
    with temporary_geometry(gaussians, xyz, rotation):
        package = render(camera, gaussians, pipe, background, use_trained_exp=False)
    image = package["render"]
    target = camera.original_image.to(device=image.device, dtype=image.dtype)

    mask = torch.ones_like(image[:1])
    if getattr(camera, "alpha_mask", None) is not None:
        alpha = camera.alpha_mask.to(device=image.device, dtype=image.dtype)
        mask = alpha if alpha.ndim == 3 else alpha[None]
        if mask.shape[0] != 1:
            mask = mask[:1]

    rgb = (torch.abs(image - target) * mask).sum() / mask.expand_as(image).sum().clamp_min(1.0)
    dssim = 1.0 - ssim(image * mask, target * mask)

    depth = image.new_zeros(())
    if args.use_depth and bool(getattr(camera, "depth_reliable", False)):
        target_depth = camera.invdepthmap.to(device=image.device, dtype=image.dtype)
        depth_mask = camera.depth_mask.to(device=image.device, dtype=image.dtype)
        if depth_mask.ndim == 2:
            depth_mask = depth_mask[None]
        valid = depth_mask * mask
        depth = (torch.abs(package["depth"] - target_depth) * valid).sum() / valid.sum().clamp_min(1.0)

    total = args.lambda_rgb * rgb + args.lambda_ssim * dssim + args.lambda_depth * depth
    return total, package, (float(rgb.detach()), float(dssim.detach()), float(depth.detach()))


@torch.no_grad()
def _reverse_stats(
    camera,
    gaussians,
    pipe,
    xyz,
    rotation,
    *,
    group_ids: torch.Tensor | None = None,
    num_groups: int | None = None,
):
    """Call the NEW native reverse rasterizer directly, bypassing train.py."""
    device = xyz.device
    height, width = int(camera.image_height), int(camera.image_width)
    features = torch.ones((1, height, width), device=device, dtype=torch.float32)
    confidence = torch.ones_like(features)
    if getattr(camera, "alpha_mask", None) is not None:
        confidence = camera.alpha_mask.to(device=device, dtype=torch.float32)
        if confidence.ndim == 2:
            confidence = confidence[None]
        confidence = confidence[:1].clamp(0.0, 1.0)

    with temporary_geometry(gaussians, xyz.detach(), rotation.detach()):
        rasterizer = _make_rasterizer(
            camera,
            pipe,
            torch.zeros(1, device=device, dtype=torch.float32),
            gaussians,
            scaling_modifier=1.0,
        )
        scales, rotations, cov3d = _geometry_inputs(
            gaussians, pipe, scaling_modifier=1.0, detach_geometry=True
        )
        result = rasterizer.accumulate_pixel_features_to_gaussians(
            means3D=gaussians.get_xyz.detach().contiguous(),
            pixel_features=features.contiguous(),
            pixel_confidence=confidence.contiguous(),
            opacities=gaussians.get_opacity.detach(),
            scales=scales,
            rotations=rotations,
            cov3D_precomp=cov3d,
            group_ids=group_ids,
            num_groups=num_groups,
            min_observation_weight=0.0,
        )

    if not isinstance(result, (tuple, list)) or len(result) != 7:
        raise RuntimeError(
            "Expected the updated rasterizer to return 7 tensors: "
            "(rendered, feature_sum, valid_weight, count, raw_support, visible_mass, radii). "
            f"Got type={type(result)} len={len(result) if isinstance(result, (tuple, list)) else 'n/a'}. "
            "The old rasterizer Python/CUDA extension is probably still being imported."
        )

    rendered, feature_sum, valid_weight, count, raw_support, visible_mass, radii = result
    eps = 1e-8
    return {
        "rendered": rendered,
        "feature_sum": feature_sum,
        "valid_visible_mass": valid_weight.float(),
        "count": count.long(),
        "raw_support": raw_support.float(),
        "visible_mass": visible_mass.float(),
        "visibility_ratio": (visible_mass.float() / raw_support.float().clamp_min(eps)).clamp(0, 1),
        "valid_fraction": (valid_weight.float() / visible_mass.float().clamp_min(eps)).clamp(0, 1),
        "radii": radii,
    }


def _mass_score(value: torch.Tensor):
    positive = value[value > 0]
    if positive.numel() == 0:
        return torch.zeros_like(value)
    scale = torch.quantile(positive, 0.75).clamp_min(1e-8)
    return (value / scale).clamp(0, 1)


def _cell_support(camera, gaussians, pipe, xyz, rotation, full_group_ids, num_cells):
    # Request per-Gaussian support first; aggregate only after each Gaussian has
    # its own visibility ratio. This avoids turning normal within-cell overlap
    # into a fake cell-occlusion probability.
    stats = _reverse_stats(
        camera, gaussians, pipe, xyz, rotation, group_ids=None, num_groups=None
    )
    raw_i = stats["raw_support"].reshape(-1)
    visible_i = stats["visible_mass"].reshape(-1)
    valid_i = stats["valid_visible_mass"].reshape(-1)
    ids = full_group_ids.reshape(-1).long()
    valid_id = ids >= 0

    def scatter_sum(value):
        out = torch.zeros(int(num_cells), device=value.device, dtype=torch.float32)
        if bool(valid_id.any()):
            out.scatter_add_(0, ids[valid_id], value[valid_id].float())
        return out

    raw = scatter_sum(raw_i)
    visible = scatter_sum(visible_i)
    valid = scatter_sum(valid_i)
    ratio_i = (visible_i / raw_i.clamp_min(1e-8)).clamp(0, 1)
    projected_i = raw_i > 1e-8
    low_vis_alpha = raw_i * (projected_i & (ratio_i < 0.10)).to(raw_i.dtype)
    low_visibility_fraction = scatter_sum(low_vis_alpha) / raw.clamp_min(1e-8)
    projected = _mass_score(raw)
    visible_score = _mass_score(visible)
    valid_score = _mass_score(valid)
    visibility_ratio = (visible / raw.clamp_min(1e-8)).clamp(0, 1)
    valid_fraction = (valid / visible.clamp_min(1e-8)).clamp(0, 1)
    return {
        "raw_support": raw, "visible_mass": visible, "valid_visible_mass": valid,
        "visibility_ratio": visibility_ratio, "valid_fraction": valid_fraction,
        "projected_score": projected, "visible_score": visible_score,
        "evidence_score": valid_score,
        "occluded_score": low_visibility_fraction.clamp(0, 1),
    }


def _quantiles(x: torch.Tensor, mask: torch.Tensor | None = None):
    x = x.detach().float()
    if mask is not None:
        x = x[mask]
    x = x[torch.isfinite(x)]
    if x.numel() == 0:
        return {"q10": None, "q50": None, "q90": None}
    q = torch.quantile(x, torch.tensor([0.1, 0.5, 0.9], device=x.device))
    return {"q10": float(q[0]), "q50": float(q[1]), "q90": float(q[2])}


def native_sanity(camera, gaussians, pipe, points, rotations, output_dir: Path):
    print("\n=== Native CUDA responsibility sanity ===")
    print("_C:", _C.__file__)
    print("pixel stats binding:", hasattr(_C, "rasterize_gaussian_pixel_feature_stats"))
    print(
        "facade method:",
        hasattr(GaussianRasterizer, "accumulate_pixel_features_to_gaussians"),
    )
    if not hasattr(_C, "rasterize_gaussian_pixel_feature_stats"):
        raise RuntimeError("Compiled extension does not expose rasterize_gaussian_pixel_feature_stats")
    if not hasattr(GaussianRasterizer, "accumulate_pixel_features_to_gaussians"):
        raise RuntimeError("GaussianRasterizer lacks accumulate_pixel_features_to_gaussians")

    stats = _reverse_stats(camera, gaussians, pipe, points, rotations)
    raw = stats["raw_support"]
    visible = stats["visible_mass"]
    valid = stats["valid_visible_mass"]
    ratio = stats["visibility_ratio"]

    for name, tensor in (("raw_support", raw), ("visible_mass", visible), ("valid_visible_mass", valid)):
        if not bool(torch.isfinite(tensor).all()):
            raise RuntimeError(f"{name} contains NaN/Inf")
        if float(tensor.min()) < -1e-6:
            raise RuntimeError(f"{name} contains negative values")

    tolerance = 1e-4
    vis_violation = visible > raw + tolerance * (1.0 + raw.abs())
    valid_violation = valid > visible + tolerance * (1.0 + visible.abs())
    if bool(vis_violation.any()):
        raise RuntimeError(f"visible_mass > raw_support for {int(vis_violation.sum())} Gaussians")
    if bool(valid_violation.any()):
        raise RuntimeError(f"valid_visible_mass > visible_mass for {int(valid_violation.sum())} Gaussians")

    projected = raw > 1e-8
    if not bool(projected.any()):
        raise RuntimeError("No Gaussian has raw projected support; camera/rasterizer setup is wrong")
    occluded_like = projected & (ratio < 0.25)

    report = {
        "camera": str(getattr(camera, "dynamic_camera_id", getattr(camera, "image_name", "camera"))),
        "frame_id": int(getattr(camera, "frame_id", 0)),
        "num_gaussians": int(raw.numel()),
        "projected_gaussians": int(projected.sum()),
        "alpha_visible_gaussians": int((visible > 1e-8).sum()),
        "valid_visible_gaussians": int((valid > 1e-8).sum()),
        "low_visibility_ratio_gaussians": int(occluded_like.sum()),
        "visibility_ratio_quantiles_projected": _quantiles(ratio, projected),
        "raw_support_quantiles_projected": _quantiles(raw, projected),
        "visible_mass_quantiles_projected": _quantiles(visible, projected),
    }
    print(json.dumps(report, indent=2))
    (output_dir / "native_responsibility_sanity.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print("[PASS] native responsibility inequalities and finiteness checks")
    return report


def _seed_consistency(vector: torch.Tensor, weight: torch.Tensor):
    norm = torch.linalg.norm(vector, dim=-1)
    w = weight.clamp_min(0)
    num = torch.linalg.norm((w[..., None] * vector).sum(0), dim=-1)
    den = (w * norm).sum(0).clamp_min(1e-8)
    return (num / den).clamp(0, 1)


def _clamp_vector_norm_(value: torch.Tensor, max_norm: float):
    limit = max(float(max_norm), 1e-12)
    with torch.no_grad():
        norm = torch.linalg.norm(value, dim=-1, keepdim=True).clamp_min(1e-12)
        value.mul_(torch.clamp(limit / norm, max=1.0))


def _median_cell_spacing(centers: torch.Tensor) -> float:
    if centers.shape[0] < 2:
        return 1.0
    distance = torch.cdist(centers.float(), centers.float())
    distance.fill_diagonal_(float("inf"))
    return float(distance.min(1).values.median())


def _view_consistency(corrections: torch.Tensor, support: torch.Tensor, min_cameras: int, threshold: float):
    supported = support >= threshold
    count = supported.sum(0)
    unit = corrections / torch.linalg.norm(corrections, dim=-1, keepdim=True).clamp_min(1e-8)
    summed = (unit * supported[..., None]).sum(0)
    consistency = torch.linalg.norm(summed, dim=-1) / count.clamp_min(1).to(corrections.dtype)
    consistency = torch.where(count >= min_cameras, consistency, torch.zeros_like(consistency))
    return consistency.clamp(0, 1), count


def _html_points(path: Path, xyz, state, confidence, vectors, title):
    xyz = xyz.detach().cpu().float()
    state = state.detach().cpu().long()
    confidence = confidence.detach().cpu().float()
    vectors = vectors.detach().cpu().float()
    colors = ["#3b6ee8" if int(s) == STATIC else "#e84a3b" if int(s) == DYNAMIC else "#888888" for s in state]
    trace = {
        "type": "scatter3d", "mode": "markers",
        "x": xyz[:, 0].tolist(), "y": xyz[:, 1].tolist(), "z": xyz[:, 2].tolist(),
        "marker": {"size": 3, "color": colors, "opacity": 0.8},
        "text": [f"{STATE_NAMES[int(s)]} conf={float(c):.3f}" for s, c in zip(state, confidence)],
        "name": "cells",
    }
    lines = {"type": "scatter3d", "mode": "lines", "x": [], "y": [], "z": [], "line": {"color": "#111", "width": 3}, "name": "mean delta"}
    ids = torch.nonzero(torch.linalg.norm(vectors, dim=-1) > 1e-8, as_tuple=False).flatten()[:400]
    for idx in ids:
        p = xyz[idx]
        e = p + vectors[idx]
        lines["x"] += [float(p[0]), float(e[0]), None]
        lines["y"] += [float(p[1]), float(e[1]), None]
        lines["z"] += [float(p[2]), float(e[2]), None]
    traces = [trace] + ([lines] if ids.numel() else [])
    html = f"""<!doctype html><html><head><meta charset='utf-8'><script src='https://cdn.plot.ly/plotly-2.35.2.min.js'></script></head><body><h2>{title}</h2><div id='p' style='height:92vh'></div><script>Plotly.newPlot('p',{json.dumps(traces)},{{scene:{{aspectmode:'data'}}}});</script></body></html>"""
    path.write_text(html, encoding="utf-8")


def stage_a_smoke(scene, gaussians, pipe, args, output_dir: Path):
    print("\n=== Stage A render-guided VELOCITY denoising smoke ===")
    cameras = scene.getTrainCameras()
    groups = _ordered_complete_frame_groups(cameras)
    if len(groups) < 2:
        raise RuntimeError("Need at least two complete synchronized multiview frames")

    group_map = {fid: cams for fid, cams in groups}
    time_map = {
        fid: float(getattr(cams[0], "timestamp", fid / args.fps))
        for fid, cams in groups
    }
    available = sorted(group_map)
    t0 = time_map[available[0]]
    requested = _parse_frames(args.probe_frames)
    probe_frames = [f for f in requested if f in group_map and f != available[0]]
    if not probe_frames:
        candidates = np.rint(np.linspace(1, len(available) - 1, min(args.num_probe_frames, len(available) - 1))).astype(int)
        probe_frames = sorted({available[int(i)] for i in candidates})
    probe_dt = {f: time_map[f] - t0 for f in probe_frames}
    if any((not math.isfinite(dt)) or dt <= 0 for dt in probe_dt.values()):
        raise RuntimeError(f"Invalid probe dt values: {probe_dt}")
    print("probe frames/dt:", {f: round(probe_dt[f], 6) for f in probe_frames})

    points0 = gaussians.get_canonical_xyz.detach().clone()
    points = points0.detach()
    rotations = gaussians.get_rotation.detach().clone()
    device, dtype = points.device, points.dtype

    semantic = gaussians.get_semantic_probability
    if semantic is not None:
        semantic = semantic.detach().reshape(-1)
        foreground = semantic >= args.semantic_threshold
        if int(foreground.sum()) < 32:
            print("[semantic] too few foreground Gaussians; using all Gaussians")
            foreground = torch.ones(points.shape[0], device=device, dtype=torch.bool)
        else:
            print(f"[semantic] foreground: {int(foreground.sum())}/{points.shape[0]} at threshold {args.semantic_threshold}")
    else:
        print("[semantic] no semantic field in PLY; using all Gaussians")
        foreground = torch.ones(points.shape[0], device=device, dtype=torch.bool)

    fg_points = points[foreground]
    grid = assign_gaussians_to_cells(
        fg_points, args.grid_target_cells,
        padding_fraction=args.grid_padding,
        min_cells_per_axis=args.grid_min_cells_per_axis,
    )
    fg_cell_ids = grid.point_group_ids.to(device=device, dtype=torch.long)
    full_group_ids = torch.full((points.shape[0],), -1, device=device, dtype=torch.long)
    full_group_ids[foreground] = fg_cell_ids
    cell_centers = grid.cell_centers.to(device=device, dtype=dtype)
    num_cells = int(grid.num_cells)
    cell_spacing = _median_cell_spacing(cell_centers)
    median_dt = float(np.median(list(probe_dt.values())))

    # Scene-aware defaults: the noise produces about 12% of one cell of
    # displacement over the median probe horizon. Thresholds are in speed units.
    velocity_noise_sigma = args.velocity_noise_sigma
    if velocity_noise_sigma <= 0:
        velocity_noise_sigma = 0.12 * cell_spacing / max(median_dt, 1.0 / args.fps)
    velocity_lr = args.velocity_lr if args.velocity_lr > 0 else 0.50 * velocity_noise_sigma
    static_speed_threshold = args.static_speed_threshold if args.static_speed_threshold > 0 else 0.50 * velocity_noise_sigma
    dynamic_speed_threshold = args.dynamic_speed_threshold if args.dynamic_speed_threshold > 0 else 1.50 * velocity_noise_sigma
    speed_temperature = args.speed_temperature if args.speed_temperature > 0 else 0.35 * velocity_noise_sigma
    max_speed = args.max_speed if args.max_speed > 0 else 6.0 * velocity_noise_sigma
    max_probe_displacement = args.max_probe_displacement if args.max_probe_displacement > 0 else 1.00 * cell_spacing
    if static_speed_threshold >= dynamic_speed_threshold:
        raise RuntimeError("static-speed-threshold must be smaller than dynamic-speed-threshold")

    print(f"motion cells: {num_cells}, foreground Gaussians: {int(foreground.sum())}")
    print(f"median cell spacing: {cell_spacing:.6f} world units")
    print("velocity parameters:")
    print(f"  noise sigma       = {velocity_noise_sigma:.6f} world/s")
    print(f"  optimizer lr      = {velocity_lr:.6f} world/s")
    print(f"  static threshold  = {static_speed_threshold:.6f} world/s")
    print(f"  dynamic threshold = {dynamic_speed_threshold:.6f} world/s")
    print(f"  max speed         = {max_speed:.6f} world/s")
    print(f"  max probe disp    = {max_probe_displacement:.6f} world")

    background = torch.tensor(
        [1.0, 1.0, 1.0] if args.white_background else [0.0, 0.0, 0.0],
        device=device, dtype=dtype,
    )

    frame_dynamic, frame_static, frame_support = [], [], []
    frame_seed_cons, frame_view_cons = [], []
    frame_velocity, frame_speed, frame_speed_stability, frame_delta = [], [], [], []
    frame_occlusion, frame_visibility = [], []
    per_frame = {}

    for frame_id in probe_frames:
        dt = float(probe_dt[frame_id])
        cams = _select_cameras(group_map[frame_id], args.cameras_per_probe)
        effective_max_speed = min(max_speed, max_probe_displacement / dt)
        print(f"\n[frame {frame_id}] dt={dt:.6f}s max_speed={effective_max_speed:.6f} cameras={[getattr(c, 'dynamic_camera_id', '') for c in cams]}")
        final_velocities, initial_speeds, works = [], [], []
        support_counts, view_consistencies = [], []
        occlusion_seeds, visibility_seeds = [], []
        seed_logs = []

        for seed in range(args.num_seeds):
            generator = torch.Generator(device=device).manual_seed(args.seed + 1000 * frame_id + seed)
            velocity = torch.nn.Parameter(
                velocity_noise_sigma * torch.randn((num_cells, 3), device=device, dtype=dtype, generator=generator)
            )
            _clamp_vector_norm_(velocity, effective_max_speed)
            optimizer = torch.optim.Adam([velocity], lr=velocity_lr)
            work = torch.zeros(num_cells, device=device, dtype=dtype)
            initial_speed = torch.linalg.norm(velocity.detach(), dim=-1)
            loss_start = None
            loss_end = None

            for step_idx in range(args.steps):
                optimizer.zero_grad(set_to_none=True)
                delta = velocity * dt
                probe_xyz = points.clone()
                probe_xyz[foreground] = fg_points + delta[fg_cell_ids]
                total = velocity.sum() * 0.0
                rgb_vals = []
                for camera in cams:
                    loss_view, _, components = _render_loss(camera, gaussians, pipe, background, probe_xyz, rotations, args)
                    total = total + loss_view / len(cams)
                    rgb_vals.append(components[0])
                zero_prior = _pseudo_huber(torch.linalg.norm(velocity, dim=-1), velocity_noise_sigma).mean()
                total = total + args.lambda_zero * zero_prior
                if loss_start is None:
                    loss_start = float(total.detach())
                previous = velocity.detach().clone()
                total.backward()
                grad = velocity.grad.detach().clone()
                optimizer.step()
                _clamp_vector_norm_(velocity, effective_max_speed)
                step_velocity = velocity.detach() - previous
                work += (-grad * step_velocity).sum(-1)
                loss_end = float(total.detach())
                if args.verbose:
                    print(
                        f"  seed={seed} step={step_idx+1}/{args.steps} loss={loss_end:.6f} "
                        f"rgb={np.mean(rgb_vals):.6f} speed={float(torch.linalg.norm(velocity.detach(), dim=-1).mean()):.6f}"
                    )

            corrections, supports, occlusions, visibilities = [], [], [], []
            for camera in cams:
                optimizer.zero_grad(set_to_none=True)
                delta = velocity * dt
                probe_xyz = points.clone()
                probe_xyz[foreground] = fg_points + delta[fg_cell_ids]
                loss_view, _, _ = _render_loss(camera, gaussians, pipe, background, probe_xyz, rotations, args)
                loss_view.backward()
                corrections.append((-velocity.grad.detach()).clone())
                sup = _cell_support(camera, gaussians, pipe, probe_xyz, rotations, full_group_ids, num_cells)
                supports.append(sup["evidence_score"])
                occlusions.append(sup["occluded_score"])
                visibilities.append(sup["visibility_ratio"])

            corrections = torch.stack(corrections)
            supports = torch.stack(supports)
            occlusions = torch.stack(occlusions)
            visibilities = torch.stack(visibilities)
            view_cons, support_count = _view_consistency(
                corrections, supports, args.min_support_cameras, args.support_threshold
            )

            final_velocities.append(velocity.detach().clone())
            initial_speeds.append(initial_speed)
            works.append(work.clamp_min(0))
            support_counts.append(support_count)
            view_consistencies.append(view_cons)
            occlusion_seeds.append(occlusions.mean(0))
            visibility_seeds.append(visibilities.mean(0))
            seed_logs.append({
                "seed": seed, "dt": dt, "loss_start": loss_start, "loss_end": loss_end,
                "mean_initial_speed": float(initial_speed.mean()),
                "mean_final_speed": float(torch.linalg.norm(velocity.detach(), dim=-1).mean()),
                "mean_final_displacement": float(torch.linalg.norm(velocity.detach() * dt, dim=-1).mean()),
                "mean_supported_cameras": float(support_count.float().mean()),
                "mean_low_visibility_fraction": float(occlusions.mean()),
                "mean_visibility_ratio": float(visibilities.mean()),
            })
            print("  ", seed_logs[-1])

        final_velocity = torch.stack(final_velocities)
        initial_speed = torch.stack(initial_speeds)
        work = torch.stack(works)
        support_count = torch.stack(support_counts).float()
        view_cons = torch.stack(view_consistencies)
        support_fraction = (support_count / max(1, len(cams))).mean(0).clamp(0, 1)
        seed_weight = support_fraction[None] * (0.50 + 0.50 * _robust_unit(work))
        seed_cons = _seed_consistency(final_velocity, seed_weight)
        seed_speed = torch.linalg.norm(final_velocity, dim=-1)
        mean_velocity = final_velocity.mean(0)
        mean_speed = seed_speed.mean(0)
        speed_std = seed_speed.std(0, unbiased=False)
        speed_stability = torch.exp(-speed_std / max(static_speed_threshold, 1e-8)).clamp(0, 1)
        shrink_score = ((initial_speed.mean(0) - mean_speed) / velocity_noise_sigma).clamp(0, 1)
        work_score = _robust_unit(work.mean(0))
        dynamic_speed_score = torch.sigmoid((mean_speed - dynamic_speed_threshold) / speed_temperature)
        static_speed_score = torch.sigmoid((static_speed_threshold - mean_speed) / speed_temperature)
        view_mean = view_cons.mean(0).clamp(0, 1)
        supported_frame = (support_count >= args.min_support_cameras).any(0)
        dynamic_score = support_fraction * seed_cons * view_mean * dynamic_speed_score * (0.50 + 0.50 * work_score)
        static_score = support_fraction * static_speed_score * speed_stability * (0.50 + 0.50 * shrink_score)
        dynamic_score = torch.where(supported_frame, dynamic_score, torch.zeros_like(dynamic_score))
        static_score = torch.where(supported_frame, static_score, torch.zeros_like(static_score))

        frame_dynamic.append(dynamic_score)
        frame_static.append(static_score)
        frame_support.append(supported_frame.float())
        frame_seed_cons.append(seed_cons)
        frame_view_cons.append(view_mean)
        frame_velocity.append(mean_velocity)
        frame_speed.append(mean_speed)
        frame_speed_stability.append(speed_stability)
        frame_delta.append(mean_velocity * dt)
        frame_occlusion.append(torch.stack(occlusion_seeds).mean(0))
        frame_visibility.append(torch.stack(visibility_seeds).mean(0))
        per_frame[frame_id] = {
            "dt": dt, "seed_logs": seed_logs,
            "mean_dynamic_score": float(dynamic_score.mean()),
            "mean_static_score": float(static_score.mean()),
            "mean_seed_consistency": float(seed_cons.mean()),
            "mean_view_consistency": float(view_mean.mean()),
            "mean_support_fraction": float(support_fraction.mean()),
            "mean_low_visibility_fraction": float(frame_occlusion[-1].mean()),
            "mean_visibility_ratio": float(frame_visibility[-1].mean()),
            "mean_speed": float(mean_speed.mean()),
            "median_speed": float(mean_speed.median()),
            "mean_displacement": float(torch.linalg.norm(mean_velocity * dt, dim=-1).mean()),
        }

    dyn_stack = torch.stack(frame_dynamic)
    sta_stack = torch.stack(frame_static)
    info = torch.stack(frame_support) > 0
    info_f = info.to(dtype)
    info_count = info.sum(0)
    denom = info_count.to(dtype).clamp_min(1)
    dyn_score = (dyn_stack * info_f).sum(0) / denom
    sta_score = (sta_stack * info_f).sum(0) / denom
    seed_stack = torch.stack(frame_seed_cons)
    view_stack = torch.stack(frame_view_cons)
    seed_cons = ((seed_stack * info_f).sum(0) / denom).clamp(0, 1)
    view_cons = ((view_stack * info_f).sum(0) / denom).clamp(0, 1)

    speed_stack = torch.stack(frame_speed)
    stability_stack = torch.stack(frame_speed_stability)
    mean_speed = (speed_stack * info_f).sum(0) / denom
    speed_std = torch.sqrt((((speed_stack - mean_speed[None]) ** 2) * info_f).sum(0) / denom)
    speed_stability = ((stability_stack * info_f).sum(0) / denom).clamp(0, 1)
    dynamic_persistence = (((speed_stack >= dynamic_speed_threshold).to(dtype) * info_f).sum(0) / denom).clamp(0, 1)
    static_persistence = (((speed_stack <= static_speed_threshold).to(dtype) * info_f).sum(0) / denom).clamp(0, 1)

    velocity_stack = torch.stack(frame_velocity)
    mean_velocity = (velocity_stack * info_f[..., None]).sum(0) / denom[:, None]
    delta_stack = torch.stack(frame_delta)
    mean_delta = (delta_stack * info_f[..., None]).sum(0) / denom[:, None]
    support_fraction = info_count.to(dtype) / len(probe_frames)
    low_visibility_fraction = torch.stack(frame_occlusion).mean(0)
    visibility_ratio = torch.stack(frame_visibility).mean(0)

    dyn_score = dyn_score * (0.50 + 0.50 * dynamic_persistence)
    sta_score = sta_score * (0.50 + 0.50 * static_persistence)
    p_dynamic = torch.where(
        dyn_score + sta_score > 1e-8,
        dyn_score / (dyn_score + sta_score).clamp_min(1e-8),
        torch.full_like(dyn_score, 0.5),
    )
    margin = (2 * (p_dynamic - 0.5).abs()).clamp(0, 1)
    dynamic_consistency = torch.where(
        (seed_cons >= args.seed_consistency_threshold) & (view_cons >= args.view_consistency_threshold),
        seed_cons * view_cons * (0.50 + 0.50 * dynamic_persistence),
        torch.zeros_like(seed_cons),
    )
    static_consistency = (speed_stability * (0.50 + 0.50 * static_persistence)).clamp(0, 1)
    mode_consistency = torch.where(p_dynamic >= 0.5, dynamic_consistency, static_consistency)
    confidence = (margin * support_fraction * mode_consistency).clamp(0, 1)

    state = torch.full((num_cells,), UNKNOWN, device=device, dtype=torch.long)
    enough_frames = info_count >= args.min_informative_frames
    state[enough_frames & (p_dynamic >= args.dynamic_threshold) & (confidence >= args.confidence_threshold)] = DYNAMIC
    state[enough_frames & (p_dynamic <= args.static_threshold) & (confidence >= args.confidence_threshold)] = STATIC

    full_p = torch.full((points.shape[0],), 0.5, device=device)
    full_conf = torch.zeros(points.shape[0], device=device)
    full_state = torch.full((points.shape[0],), UNKNOWN, device=device, dtype=torch.long)
    full_p[foreground] = p_dynamic[fg_cell_ids]
    full_conf[foreground] = confidence[fg_cell_ids]
    full_state[foreground] = state[fg_cell_ids]

    canonical_change = float((gaussians.get_canonical_xyz.detach() - points0).abs().max())
    summary = {
        "schema_version": 2,
        "representation": "velocity",
        "velocity_units": "world_units_per_second",
        "num_gaussians": int(points.shape[0]),
        "foreground_gaussians": int(foreground.sum()),
        "num_cells": num_cells,
        "probe_frames": probe_frames,
        "probe_dt": probe_dt,
        "median_cell_spacing": cell_spacing,
        "effective_parameters": {
            "velocity_noise_sigma": velocity_noise_sigma,
            "velocity_lr": velocity_lr,
            "static_speed_threshold": static_speed_threshold,
            "dynamic_speed_threshold": dynamic_speed_threshold,
            "speed_temperature": speed_temperature,
            "max_speed": max_speed,
            "max_probe_displacement": max_probe_displacement,
        },
        "dynamic_cells": int((state == DYNAMIC).sum()),
        "static_cells": int((state == STATIC).sum()),
        "unknown_cells": int((state == UNKNOWN).sum()),
        "mean_dynamic_probability": float(p_dynamic.mean()),
        "mean_confidence": float(confidence.mean()),
        "mean_seed_consistency": float(seed_cons.mean()),
        "mean_view_consistency": float(view_cons.mean()),
        "mean_support_fraction": float(support_fraction.mean()),
        "mean_low_visibility_fraction": float(low_visibility_fraction.mean()),
        "mean_visibility_ratio": float(visibility_ratio.mean()),
        "mean_speed": float(mean_speed.mean()),
        "median_speed": float(mean_speed.median()),
        "mean_speed_std": float(speed_std.mean()),
        "mean_dynamic_persistence": float(dynamic_persistence.mean()),
        "mean_static_persistence": float(static_persistence.mean()),
        "canonical_max_abs_change": canonical_change,
        "per_frame": per_frame,
    }
    print("\n=== Stage A velocity summary ===")
    print(json.dumps(summary, indent=2))
    if canonical_change != 0.0:
        raise RuntimeError(f"Canonical XYZ changed during probe: max |delta|={canonical_change}")
    if float(seed_cons.max()) > 1.000001 or float(view_cons.max()) > 1.000001:
        raise RuntimeError("Consistency aggregation escaped [0,1]")

    torch.save({
        "schema_version": 2,
        "cell_centers": cell_centers.cpu(),
        "dynamic_score": dyn_score.cpu(), "static_score": sta_score.cpu(),
        "dynamic_probability": p_dynamic.cpu(), "confidence": confidence.cpu(), "state": state.cpu(),
        "mean_velocity": mean_velocity.cpu(), "mean_speed": mean_speed.cpu(), "speed_std": speed_std.cpu(),
        "dynamic_persistence": dynamic_persistence.cpu(), "static_persistence": static_persistence.cpu(),
        "mean_delta": mean_delta.cpu(), "seed_consistency": seed_cons.cpu(), "view_consistency": view_cons.cpu(),
        "support_fraction": support_fraction.cpu(), "low_visibility_fraction": low_visibility_fraction.cpu(),
        "visibility_ratio": visibility_ratio.cpu(), "probe_frames": probe_frames, "probe_dt": probe_dt,
    }, output_dir / "per_cell_scores.pt")
    torch.save({
        "schema_version": 2, "canonical_xyz": points.cpu(), "foreground_mask": foreground.cpu(),
        "dynamic_probability": full_p.cpu(), "confidence": full_conf.cpu(), "state": full_state.cpu(),
        "cell_ids_foreground": fg_cell_ids.cpu(), "per_cell_mean_velocity": mean_velocity.cpu(),
        "per_cell_mean_speed": mean_speed.cpu(),
    }, output_dir / "per_gaussian_scores.pt")
    (output_dir / "stageA_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    _html_points(output_dir / "stageA_cells.html", cell_centers, state, confidence, mean_velocity, "Render-guided velocity Stage A")
    print(f"[PASS] Stage A velocity probe completed; outputs: {output_dir}")
    return summary


def build_scene(args):
    static_model_dir = Path(args.static_model_dir).expanduser().resolve()
    root_scene = Path(args.root_scene).expanduser().resolve()
    cfg = _load_cfg_namespace(static_model_dir)

    sh_degree = int(args.sh_degree if args.sh_degree is not None else _cfg_value(cfg, "sh_degree", 3))
    resolution = int(args.resolution if args.resolution is not None else _cfg_value(cfg, "resolution", -1))
    white_background = bool(
        args.white_background
        if args.white_background is not None
        else _cfg_value(cfg, "white_background", False)
    )
    data_device = str(getattr(args, "camera_data_device", "cpu") or "cpu")

    source_path = root_scene / "colmap" if (root_scene / "colmap").is_dir() else root_scene
    scene_args = SimpleNamespace(
        model_path=str(static_model_dir),
        source_path=str(source_path),
        images=_cfg_value(cfg, "images", "images"),
        depths="",
        eval=False,
        train_test_exp=False,
        white_background=white_background,
        resolution=resolution,
        data_device=data_device,
        sapien_depth_is_metric=False,
    )
    dyn = SimpleNamespace(
        dynamic_motion=True,
        dynamic_sequence_root=str(root_scene),
        dynamic_rgb_dir="rgb",
        dynamic_depth_npy_dir="depth_npy",
        dynamic_depth_png_dir="depth_png",
        dynamic_use_depth=bool(args.use_depth),
        dynamic_use_depth_npy=True,
        dynamic_depth_is_metric=True,
        dynamic_cameras=args.cameras,
        dynamic_frame_start=0,
        dynamic_frame_end=-1,
        dynamic_frame_stride=1,
        dynamic_test_stride=0,
        dynamic_timestamps_file="",
        fps=float(args.fps),
        time_scale=1.0,
        motion_time_mode="seconds",
        use_dinov3=False,
        dinov3_root="dinov3",
        dinov3_feature_ext=".pt",
    )

    gaussians = GaussianModel(sh_degree, optimizer_type="default")
    scene = Scene(
        scene_args,
        gaussians,
        load_iteration=int(args.iteration),
        shuffle=False,
        dynamic_args=dyn,
    )
    pipe = SimpleNamespace(
        debug=False,
        antialiasing=bool(_cfg_value(cfg, "antialiasing", False)),
        compute_cov3D_python=False,
        convert_SHs_python=False,
    )
    return scene, gaussians, pipe, white_background


def parse_args():
    p = argparse.ArgumentParser(
        description="Standalone CUDA + render-guided 3DGS static/dynamic smoke test"
    )
    p.add_argument("--root-scene", required=True, help="Dataset ROOT containing colmap/ and rgb/")
    p.add_argument("--static-model-dir", required=True, help="Existing static 3DGS model directory")
    p.add_argument("--iteration", type=int, default=30000)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--mode", choices=["native", "stagea", "all"], default="all")

    p.add_argument("--probe-frames", default="15,30,45,59")
    p.add_argument("--num-probe-frames", type=int, default=4)
    p.add_argument("--cameras-per-probe", type=int, default=4)
    p.add_argument("--cameras", default="", help="Optional comma-separated camera IDs; empty=all")
    p.add_argument(
        "--camera-data-device",
        default="cpu",
        help="Device for cached camera RGB/depth tensors; use cpu to avoid preloading every dynamic frame on CUDA.",
    )
    p.add_argument("--fps", type=float, default=30.0)

    p.add_argument("--num-seeds", type=int, default=3)
    p.add_argument("--steps", type=int, default=10)
    # <=0 means scene-adaptive automatic value derived after grid construction.
    p.add_argument("--velocity-noise-sigma", type=float, default=-1.0, help="World units/s; <=0 auto")
    p.add_argument("--velocity-lr", type=float, default=-1.0, help="Adam step in world units/s; <=0 auto")
    p.add_argument("--static-speed-threshold", type=float, default=-1.0, help="World units/s; <=0 auto")
    p.add_argument("--dynamic-speed-threshold", type=float, default=-1.0, help="World units/s; <=0 auto")
    p.add_argument("--speed-temperature", type=float, default=-1.0, help="World units/s; <=0 auto")
    p.add_argument("--max-speed", type=float, default=-1.0, help="World units/s; <=0 auto")
    p.add_argument("--max-probe-displacement", type=float, default=-1.0, help="World units; <=0 = one median cell")
    p.add_argument("--seed", type=int, default=20260831)

    p.add_argument("--grid-target-cells", type=int, default=512)
    p.add_argument("--grid-padding", type=float, default=0.05)
    p.add_argument("--grid-min-cells-per-axis", type=int, default=2)
    p.add_argument("--semantic-threshold", type=float, default=0.2)

    p.add_argument("--lambda-rgb", type=float, default=1.0)
    p.add_argument("--lambda-ssim", type=float, default=0.2)
    p.add_argument("--lambda-depth", type=float, default=0.2)
    p.add_argument("--lambda-zero", type=float, default=0.05)
    p.add_argument("--use-depth", action="store_true")

    p.add_argument("--support-threshold", type=float, default=0.03)
    p.add_argument("--min-support-cameras", type=int, default=2)
    p.add_argument("--min-informative-frames", type=int, default=2)
    p.add_argument("--seed-consistency-threshold", type=float, default=0.50)
    p.add_argument("--view-consistency-threshold", type=float, default=0.50)
    p.add_argument("--dynamic-threshold", type=float, default=0.60)
    p.add_argument("--static-threshold", type=float, default=0.40)
    p.add_argument("--confidence-threshold", type=float, default=0.05)

    p.add_argument("--resolution", type=int, default=None, help="Override cfg_args resolution")
    p.add_argument("--sh-degree", type=int, default=None, help="Override cfg_args sh_degree")
    bg = p.add_mutually_exclusive_group()
    bg.add_argument("--white-background", dest="white_background", action="store_true")
    bg.add_argument("--black-background", dest="white_background", action="store_false")
    p.set_defaults(white_background=None)
    p.add_argument("--verbose", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    print("repo root:", REPO_ROOT)
    print("torch:", torch.__version__, "CUDA build:", torch.version.cuda)
    print("GPU:", torch.cuda.get_device_name(0))
    print("CUDA extension:", _C.__file__)

    scene, gaussians, pipe, white_background = build_scene(args)
    args.white_background = white_background
    groups = _ordered_complete_frame_groups(scene.getTrainCameras())
    if not groups:
        raise RuntimeError("No complete dynamic multiview frame groups were constructed")
    print(f"dynamic complete frames: {len(groups)}")
    print(f"cameras/frame: {len(groups[0][1])}")
    print(f"Gaussians: {gaussians.get_canonical_xyz.shape[0]}")

    points = gaussians.get_canonical_xyz.detach()
    rotations = gaussians.get_rotation.detach()
    requested = _parse_frames(args.probe_frames)
    group_map = {fid: cams for fid, cams in groups}
    native_frame = next((f for f in requested if f in group_map), groups[min(1, len(groups)-1)][0])
    native_camera = _select_cameras(group_map[native_frame], args.cameras_per_probe)[0]

    if args.mode in ("native", "all"):
        native_sanity(native_camera, gaussians, pipe, points, rotations, output_dir)
    if args.mode in ("stagea", "all"):
        stage_a_smoke(scene, gaussians, pipe, args, output_dir)


if __name__ == "__main__":
    main()
