#!/usr/bin/env python3
"""Generic frame-0 -> target evaluator with paired target-0 null correction.

This keeps the v1 renderer-guided denoiser frozen.  The hard label is always:
    motion_ratio = ||mean_displacement|| / sigma_max
    dynamic_label = motion_ratio > 1.0
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from argparse import BooleanOptionalAction
from pathlib import Path

import cv2
import numpy as np
import torch
from tqdm import tqdm


def _repo_root() -> Path:
    root = Path(__file__).resolve().parents[1]
    if not (root / "scene").is_dir():
        raise RuntimeError("Run from RigEuler3DGS, or keep this script in RigEuler3DGS/tools")
    return root


REPO_ROOT = _repo_root()
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from gaussian_renderer import render  # noqa: E402
from scene.gaussian_motion_denoiser import build_canonical_knn  # noqa: E402
from tools.evaluate_frame01_counterfactual_v3 import (  # noqa: E402
    _camera_id,
    _frame_maps,
    _frame_time,
    _image_mask,
    _load_v1_checkpoint,
    _nearest_gt_labels,
    _quantiles,
    _render_loss_and_evidence,
    _run_v1_denoising_pass,
    _signed_log_ratio,
    _to_png,
)
from tools.frame01_motion_classifier import (  # noqa: E402
    _group_stats,
    _json_ready,
    _metric_counts,
    _metrics,
    _threshold_sweep,
)
from tools.test_render_denoise_discovery import build_scene  # noqa: E402


SCHEMA_VERSION = 3


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_tensor(tensor: torch.Tensor) -> str:
    array = tensor.detach().cpu().contiguous().numpy()
    return hashlib.sha256(array.tobytes()).hexdigest()


def trial_seeds(base_seed: int, count: int) -> torch.Tensor:
    return torch.arange(int(count), dtype=torch.long) + int(base_seed) + 100_000


def compute_motion_fields(trial_displacements: torch.Tensor, sigma_max: float) -> dict:
    if trial_displacements.ndim != 3 or trial_displacements.shape[-1] != 3:
        raise ValueError("trial_displacements must be [S,N,3]")
    mean = trial_displacements.mean(0)
    norm = torch.linalg.norm(mean, dim=-1)
    ratio = norm / max(float(sigma_max), 1.0e-8)
    label = ratio > 1.0
    trial_norm = torch.linalg.norm(trial_displacements, dim=-1)
    consistency = (norm / trial_norm.mean(0).clamp_min(1.0e-8)).clamp(0.0, 1.0)
    for name, value in (("mean", mean), ("norm", norm), ("ratio", ratio), ("consistency", consistency)):
        if not bool(torch.isfinite(value).all()):
            raise RuntimeError(f"{name} contains non-finite values")
    return {
        "mean": mean,
        "norm": norm,
        "ratio": ratio,
        "label": label,
        "consistency": consistency,
    }


def null_correct(raw_trials: torch.Tensor, null_trials: torch.Tensor) -> torch.Tensor:
    if raw_trials.shape != null_trials.shape:
        raise ValueError(f"raw/null trial shape mismatch: {tuple(raw_trials.shape)} vs {tuple(null_trials.shape)}")
    corrected = raw_trials - null_trials
    if not torch.allclose(corrected, raw_trials - null_trials):
        raise RuntimeError("null correction invariant failed")
    return corrected


def _checkpoint_train_target(payload: dict, checkpoint_path: Path) -> int:
    for key in ("target_frame", "checkpoint_train_target"):
        if key in payload:
            return int(payload[key])
    cfg = payload.get("training_config", {}) if isinstance(payload, dict) else {}
    if isinstance(cfg, dict) and "target_frame" in cfg:
        return int(cfg["target_frame"])
    stem = checkpoint_path.stem
    if "_to_" in stem:
        return int(stem.rsplit("_to_", 1)[1])
    raise RuntimeError(f"cannot infer checkpoint train target from {checkpoint_path}")


def _loss_dict(result) -> dict:
    return {
        "total": float(result["loss"]),
        "rgb": float(result["rgb"]),
        "ssim": float(result["ssim"]),
        "depth": float(result["depth"]),
    }


def _weighted_mean(stack: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    return (stack * weights).sum(0) / weights.sum(0).clamp_min(1.0e-8)


def _render_eval(cameras, gaussians, pipe, background, xyz, rotations, args, collect_evidence=False):
    return _render_loss_and_evidence(
        cameras,
        gaussians,
        pipe,
        background,
        xyz,
        rotations,
        args,
        collect_evidence=collect_evidence,
    )


def run_trials(args, target_cameras, gaussians, pipe, background, canonical, rotations, denoiser, knn_idx, schedule, median_nn, seeds):
    trial_displacements = []
    rgb_stack = []
    depth_stack = []
    combined_stack = []
    valid_stack = []
    raw_stack = []
    visible_stack = []
    trajectory = None
    for trial, seed in enumerate(tqdm(seeds.tolist(), desc=f"frame0->{args.target_frame} frozen-v1 trials", unit="trial")):
        generator = torch.Generator(device=canonical.device).manual_seed(int(seed))
        noise = schedule[0] * torch.randn(canonical.shape, device=canonical.device, dtype=canonical.dtype, generator=generator)
        out = _run_v1_denoising_pass(
            denoiser,
            target_cameras,
            gaussians,
            pipe,
            background,
            canonical,
            rotations,
            knn_idx,
            noise,
            schedule,
            median_nn,
            args,
            keep_trajectory=(trial == 0),
        )
        xyz = out["xyz"].detach()
        if xyz.shape != canonical.shape:
            raise RuntimeError("bad denoiser output shape")
        trial_displacements.append(xyz - canonical)
        evidence = out["final"]["evidence"]
        rgb_stack.append(evidence["rgb_evidence"].detach())
        depth_stack.append(evidence["depth_evidence"].detach())
        combined_stack.append(evidence["combined_residual"].detach())
        valid_stack.append(evidence["valid_visible_mass"].detach())
        raw_stack.append(evidence["raw_support"].detach())
        visible_stack.append(evidence["visible_mass"].detach())
        if trajectory is None:
            trajectory = out["trajectory"]
    return {
        "trial_displacements_raw": torch.stack(trial_displacements),
        "rgb_stack": torch.stack(rgb_stack),
        "depth_stack": torch.stack(depth_stack),
        "combined_stack": torch.stack(combined_stack),
        "valid_stack": torch.stack(valid_stack),
        "raw_stack": torch.stack(raw_stack),
        "visible_stack": torch.stack(visible_stack),
        "trajectory": trajectory,
    }


def _html(path: Path, traces, title: str, xyz: torch.Tensor):
    points = xyz.detach().cpu().float()
    lo = points.min(0).values.tolist()
    hi = points.max(0).values.tolist()
    layout = {
        "title": title,
        "scene": {
            "aspectmode": "data",
            "xaxis": {"range": [lo[0], hi[0]]},
            "yaxis": {"range": [lo[1], hi[1]]},
            "zaxis": {"range": [lo[2], hi[2]]},
        },
    }
    path.write_text(
        "<!doctype html><html><head><meta charset='utf-8'>"
        "<script src='https://cdn.plot.ly/plotly-2.35.2.min.js'></script>"
        "</head><body><div id='p' style='height:95vh'></div><script>"
        f"Plotly.newPlot('p',{json.dumps(traces)},{json.dumps(layout)});"
        "</script></body></html>",
        encoding="utf-8",
    )


def _subset(n: int, max_points: int) -> torch.Tensor:
    if n <= max_points:
        return torch.arange(n, dtype=torch.long)
    return torch.linspace(0, n - 1, max_points).round().long().unique()


def _scatter(xyz, ids, color, name, text=None, size=3, opacity=0.8):
    pts = xyz[ids.to(xyz.device)].detach().cpu().float()
    return {
        "type": "scatter3d",
        "mode": "markers",
        "name": name,
        "x": pts[:, 0].tolist(),
        "y": pts[:, 1].tolist(),
        "z": pts[:, 2].tolist(),
        "marker": {"size": size, "color": color, "opacity": opacity},
        "text": text if text is not None else [str(int(i)) for i in ids],
        "hoverinfo": "text",
    }


def _point_text(ids, raw, corrected, null, visibility, gt=None, gt_dist=None):
    out = []
    for i in ids:
        idx = int(i)
        parts = [
            f"idx={idx}",
            f"Mraw={float(raw['ratio'][idx]):.4f}",
            f"Craw={float(raw['consistency'][idx]):.4f}",
            f"visibility={float(visibility[idx]):.4f}",
        ]
        if corrected is not None:
            parts += [f"Mcorr={float(corrected['ratio'][idx]):.4f}", f"Ccorr={float(corrected['consistency'][idx]):.4f}"]
        if null is not None:
            parts += [f"Mnull={float(null['ratio'][idx]):.4f}", f"Cnull={float(null['consistency'][idx]):.4f}"]
        if gt is not None:
            parts += [f"GT={'dynamic' if bool(gt[idx]) else 'static'}", f"GTdist={float(gt_dist[idx]):.6f}"]
        out.append("<br>".join(parts))
    return out


def write_label_html(path, xyz, fields, mode_name, raw, corrected, null, visibility, max_points):
    ids = _subset(xyz.shape[0], max_points)
    label = fields["label"]
    traces = []
    for value, name, color in ((False, "static", "#3267d6"), (True, "dynamic", "#d33f2f")):
        part = ids[label[ids.to(label.device)].detach().cpu() == value]
        traces.append(_scatter(xyz, part, color, name, _point_text(part, raw, corrected, null, visibility)))
    _html(path, traces, f"{mode_name}: dynamic iff motion_ratio > 1.0", xyz)


def write_score_html(path, xyz, score, title, colorbar, raw, corrected, null, visibility, max_points, clamp=None):
    ids = _subset(xyz.shape[0], max_points)
    pts = xyz[ids.to(xyz.device)].detach().cpu().float()
    values = score[ids.to(score.device)].detach().cpu().float()
    shown = values.clamp(*clamp) if clamp is not None else values
    trace = {
        "type": "scatter3d",
        "mode": "markers",
        "x": pts[:, 0].tolist(),
        "y": pts[:, 1].tolist(),
        "z": pts[:, 2].tolist(),
        "marker": {"size": 3, "color": shown.tolist(), "colorscale": "Turbo", "opacity": 0.8, "colorbar": {"title": colorbar}},
        "text": _point_text(ids, raw, corrected, null, visibility),
        "hoverinfo": "text",
    }
    _html(path, [trace], title, xyz)


def write_vectors_html(path, xyz, displacement, label, title, raw, corrected, null, visibility, max_points):
    ids = _subset(xyz.shape[0], min(max_points, 1000))
    pts = xyz[ids.to(xyz.device)].detach().cpu().float()
    disp = displacement[ids.to(displacement.device)].detach().cpu().float()
    labels = label[ids.to(label.device)].detach().cpu().bool()
    traces = []
    for value, name, color in ((False, "static vectors", "#3267d6"), (True, "dynamic vectors", "#d33f2f")):
        line = {"type": "scatter3d", "mode": "lines", "name": name, "x": [], "y": [], "z": [], "line": {"color": color, "width": 3}}
        for j in torch.nonzero(labels == value, as_tuple=False).flatten():
            p = pts[j]
            q = p + disp[j]
            line["x"] += [float(p[0]), float(q[0]), None]
            line["y"] += [float(p[1]), float(q[1]), None]
            line["z"] += [float(p[2]), float(q[2]), None]
        traces.append(line)
    traces.append(_scatter(xyz, ids, "#999999", "points", _point_text(ids, raw, corrected, null, visibility), size=2, opacity=0.18))
    _html(path, traces, title, xyz)


def write_gt_htmls(prefix: str, output_dir: Path, xyz, gt, gt_dist, reliable, fields, raw, corrected, null, visibility, max_points):
    pred = fields["label"]
    classes = [
        ("TP", pred & gt & reliable, "#d33f2f"),
        ("TN", (~pred) & (~gt) & reliable, "#3267d6"),
        ("FP", pred & (~gt) & reliable, "#ff9f1c"),
        ("FN", (~pred) & gt & reliable, "#7b2cbf"),
    ]
    traces = []
    for name, mask, color in classes:
        ids = _subset(int(mask.sum()), max_points)
        all_ids = torch.nonzero(mask.detach().cpu(), as_tuple=False).flatten()
        ids = all_ids[ids] if all_ids.numel() else all_ids
        traces.append(_scatter(xyz, ids, color, name, _point_text(ids, raw, corrected, null, visibility, gt, gt_dist)))
    _html(output_dir / f"{prefix}_gt_confusion.html", traces, f"{prefix} GT confusion", xyz)

    context = torch.nonzero(reliable.detach().cpu(), as_tuple=False).flatten()
    context = context[_subset(context.numel(), max_points)] if context.numel() else context
    for kind, mask in (("false_positives", pred & (~gt) & reliable), ("false_negatives", (~pred) & gt & reliable)):
        errors = torch.nonzero(mask.detach().cpu(), as_tuple=False).flatten()
        errors = errors[_subset(errors.numel(), max_points)] if errors.numel() else errors
        traces = [
            _scatter(xyz, context, "#bbbbbb", "reliable GT context", size=2, opacity=0.18),
            _scatter(xyz, errors, "#ff006e", kind, _point_text(errors, raw, corrected, null, visibility, gt, gt_dist), size=6, opacity=0.95),
        ]
        _html(output_dir / f"{prefix}_{kind}.html", traces, f"{prefix} {kind}", xyz)


def _scale_gray(value: torch.Tensor, quantile=0.99):
    x = torch.as_tensor(value).detach().float()
    if x.ndim == 3:
        x = x.mean(0)
    finite = x[torch.isfinite(x)]
    scale = torch.quantile(finite.abs(), quantile).clamp_min(1.0e-8) if finite.numel() else x.new_tensor(1.0)
    vis = (x / scale).clamp(0.0, 1.0)
    return _to_png(vis)


def _write_histograms(path: Path, series: dict[str, torch.Tensor]):
    width, height = 1100, 520
    image = np.full((height, width, 3), 255, np.uint8)
    colors = [(220, 70, 40), (40, 120, 220), (45, 170, 75)]
    cv2.putText(image, path.stem, (35, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (20, 20, 20), 2, cv2.LINE_AA)
    for idx, (name, value) in enumerate(series.items()):
        x = value.detach().cpu().float().numpy()
        x = x[np.isfinite(x)]
        if x.size == 0:
            continue
        hi = max(float(np.percentile(x, 99.0)), 1.0e-8)
        hist, bins = np.histogram(np.clip(x, 0.0, hi), bins=80, range=(0.0, hi))
        hist = hist.astype(np.float32) / max(float(hist.max()), 1.0)
        x0 = 60
        y0 = 70 + idx * 135
        w = width - 120
        h = 95
        pts = []
        for b, v in enumerate(hist):
            px = int(x0 + b / max(len(hist) - 1, 1) * w)
            py = int(y0 + h * (1.0 - float(v)))
            pts.append((px, py))
        for a, b in zip(pts[:-1], pts[1:]):
            cv2.line(image, a, b, colors[idx % len(colors)], 2, cv2.LINE_AA)
        cv2.rectangle(image, (x0, y0), (x0 + w, y0 + h), (0, 0, 0), 1)
        cv2.putText(image, f"{name}  p99={hi:.5g}", (x0, y0 - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, colors[idx % len(colors)], 1, cv2.LINE_AA)
    cv2.imwrite(str(path), image)


def _label(img: np.ndarray, text: str) -> np.ndarray:
    out = img.copy()
    cv2.rectangle(out, (0, 0), (out.shape[1], 25), (0, 0, 0), -1)
    cv2.putText(out, text[:90], (5, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 255, 255), 1, cv2.LINE_AA)
    return out


def write_debug_2d(output_dir, cameras, gaussians, pipe, background, canonical, rotations, raw_xyz, corrected_xyz, args):
    debug_dir = output_dir / "debug_2d"
    debug_dir.mkdir(parents=True, exist_ok=True)
    for camera in cameras:
        target = camera.original_image.to(canonical)
        mask = _image_mask(camera, canonical.device, canonical.dtype)
        packages = {
            "canonical": render(camera, gaussians, pipe, background, means3D_override=canonical, rotations_override=rotations),
            "raw": render(camera, gaussians, pipe, background, means3D_override=raw_xyz, rotations_override=rotations),
        }
        if corrected_xyz is not None:
            packages["corrected"] = render(camera, gaussians, pipe, background, means3D_override=corrected_xyz, rotations_override=rotations)
        else:
            packages["corrected"] = packages["raw"]
        images = {k: v["render"] for k, v in packages.items()}
        rgb_l1 = {
            key: float((torch.abs(value - target) * mask).sum() / mask.expand_as(value).sum().clamp_min(1.0))
            for key, value in images.items()
        }
        row1 = np.concatenate(
            [
                _label(_to_png(target), "target RGB"),
                _label(_to_png(images["canonical"]), f"canonical RGB L1={rgb_l1['canonical']:.6f}"),
                _label(_to_png(images["raw"]), f"raw RGB L1={rgb_l1['raw']:.6f}"),
                _label(_to_png(images["corrected"]), f"corrected RGB L1={rgb_l1['corrected']:.6f}"),
            ],
            axis=1,
        )
        row2 = np.concatenate(
            [
                _label(_scale_gray(torch.abs(target - images["canonical"]).mean(0)), "|target-canonical|"),
                _label(_scale_gray(torch.abs(target - images["raw"]).mean(0)), "|target-raw|"),
                _label(_scale_gray(torch.abs(target - images["corrected"]).mean(0)), "|target-corrected|"),
                _label(_to_png(mask), "alpha/valid mask"),
            ],
            axis=1,
        )
        rows = [row1, row2]
        if bool(args.use_depth) and bool(getattr(camera, "depth_reliable", False)):
            target_depth = camera.invdepthmap.to(canonical)
            depth_mask = camera.depth_mask.to(canonical)
            if depth_mask.ndim == 2:
                depth_mask = depth_mask[None]
            depth_l1 = {
                key: float((torch.abs(pkg["depth"] - target_depth) * depth_mask[:1]).sum() / depth_mask[:1].sum().clamp_min(1.0))
                for key, pkg in packages.items()
            }
            row3 = np.concatenate(
                [
                    _label(_scale_gray(target_depth), "target depth"),
                    _label(_scale_gray(packages["canonical"]["depth"]), f"canonical depth L1={depth_l1['canonical']:.6f}"),
                    _label(_scale_gray(packages["raw"]["depth"]), f"raw depth L1={depth_l1['raw']:.6f}"),
                    _label(_scale_gray(packages["corrected"]["depth"]), f"corrected depth L1={depth_l1['corrected']:.6f}"),
                ],
                axis=1,
            )
            row4 = np.concatenate(
                [
                    _label(_scale_gray(torch.abs(packages["canonical"]["depth"] - target_depth)), "depth residual canonical"),
                    _label(_scale_gray(torch.abs(packages["raw"]["depth"] - target_depth)), "depth residual raw"),
                    _label(_scale_gray(torch.abs(packages["corrected"]["depth"] - target_depth)), "depth residual corrected"),
                    _label(_to_png(depth_mask[:1]), "depth valid mask"),
                ],
                axis=1,
            )
            rows.extend([row3, row4])
        cv2.imwrite(str(debug_dir / f"{_camera_id(camera)}.png"), cv2.cvtColor(np.concatenate(rows, axis=0), cv2.COLOR_RGB2BGR))


def _resolve_gt(root_scene: str, explicit: str) -> Path | None:
    if explicit:
        path = Path(explicit).expanduser().resolve()
        return path if path.is_file() else None
    for path in (Path(root_scene) / "pointcloud" / "points3D_parts.ply", Path(root_scene) / "points3D_parts.ply"):
        if path.is_file():
            return path
    return None


def _check_null_compatible(null_data: dict, *, checkpoint_sha, canonical_hash, canonical, schedule, sigma_max, seeds, target_frame):
    checks = {
        "checkpoint_sha256": checkpoint_sha,
        "canonical_xyz_sha256": canonical_hash,
        "source_frame": 0,
        "target_frame": 0,
        "sigma_max": sigma_max,
    }
    for key, expected in checks.items():
        actual = null_data.get(key)
        if key == "sigma_max":
            if abs(float(actual) - float(expected)) > 1.0e-10:
                raise RuntimeError("null sigma_max mismatch")
        elif actual != expected:
            raise RuntimeError(f"null compatibility mismatch: {key} {actual!r} != {expected!r}")
    if int(target_frame) <= 0:
        raise RuntimeError("null correction requires real target_frame > 0")
    if not torch.equal(torch.as_tensor(null_data["trial_seeds"]).long(), seeds.cpu().long()):
        raise RuntimeError("null trial seeds do not match target trial seeds")
    if not torch.allclose(torch.as_tensor(null_data["canonical_xyz"]), canonical.cpu(), atol=1.0e-10, rtol=0.0):
        raise RuntimeError("null canonical xyz/order mismatch")
    if not torch.allclose(torch.as_tensor(null_data["noise_schedule"]), schedule.detach().cpu(), atol=0.0, rtol=0.0):
        raise RuntimeError("null noise schedule mismatch")


def _summary_block(fields, gt_metrics=None):
    return {
        "num_dynamic": int(fields["label"].sum()),
        "dynamic_fraction": float(fields["label"].float().mean()),
        "displacement_quantiles": _quantiles(fields["norm"]),
        "motion_ratio_quantiles": _quantiles(fields["ratio"]),
        "seed_consistency_quantiles": _quantiles(fields["consistency"]),
        "gt_metrics": gt_metrics,
    }


def _visibility_conditioned_null_stats(ratio: torch.Tensor, visibility: torch.Tensor):
    visible = visibility > 0
    high = visible & (visibility >= torch.quantile(visibility[visible], 0.75)) if bool(visible.any()) else visible
    low = visible & (~high)
    return {
        "all": _quantiles(ratio),
        "visible": _quantiles(ratio[visible]),
        "low_visibility": _quantiles(ratio[low]),
        "high_visibility": _quantiles(ratio[high]),
    }


def run(args):
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    scene, gaussians, pipe, white_background = build_scene(args)
    args.white_background = white_background
    frame_map = _frame_maps(scene)
    source_frame = int(args.source_frame)
    target_frame = int(args.target_frame)
    if source_frame != 0:
        raise RuntimeError("this evaluator is explicitly frame-0 canonical")
    if source_frame not in frame_map or target_frame not in frame_map:
        raise RuntimeError(f"missing source/target frame: source={source_frame} target={target_frame}")

    target_cameras = sorted(frame_map[target_frame], key=_camera_id)
    source_time = _frame_time(frame_map[source_frame], source_frame, args.fps)
    target_time = _frame_time(target_cameras, target_frame, args.fps)
    delta_t = 0.0 if target_frame == source_frame else target_time - source_time
    if target_frame != source_frame and delta_t <= 0:
        delta_t = 1.0 / max(float(args.fps), 1.0e-8)

    device = gaussians.get_canonical_xyz.device
    dtype = gaussians.get_canonical_xyz.dtype
    canonical = gaussians.get_canonical_xyz.detach().clone()
    rotations = gaussians.get_rotation.detach().clone()
    canonical_guard = canonical.clone()
    canonical_hash = sha256_tensor(canonical)
    for tensor in (gaussians._xyz, gaussians._features_dc, gaussians._features_rest, gaussians._opacity, gaussians._scaling, gaussians._rotation):
        tensor.requires_grad_(False)

    checkpoint_path = Path(args.denoiser_checkpoint).expanduser().resolve()
    checkpoint_sha = sha256_file(checkpoint_path)
    denoiser, payload = _load_v1_checkpoint(str(checkpoint_path), device)
    schedule = torch.as_tensor(payload["noise_schedule"], device=device, dtype=dtype).flatten()
    if schedule.numel() != 6:
        raise RuntimeError(f"expected six-step v1 schedule, got {schedule.numel()}")
    checkpoint_train_target = _checkpoint_train_target(payload, checkpoint_path)
    knn_idx, median_nn = build_canonical_knn(canonical, int(args.knn))
    knn_idx = knn_idx.to(device)
    median_nn = median_nn.to(device=device, dtype=dtype)
    if "median_nn_distance" in payload and abs(float(payload["median_nn_distance"]) - float(median_nn)) > 1.0e-8:
        raise RuntimeError("checkpoint median_nn_distance does not match loaded canonical geometry")
    sigma_max = float(schedule[0].detach().cpu())
    seeds = trial_seeds(int(args.seed), int(args.inference_trials))

    background = torch.tensor([1.0, 1.0, 1.0] if white_background else [0.0, 0.0, 0.0], device=device, dtype=dtype)
    static_eval = _render_eval(target_cameras, gaussians, pipe, background, canonical, rotations, args, collect_evidence=True)
    trials = run_trials(args, target_cameras, gaussians, pipe, background, canonical, rotations, denoiser, knn_idx, schedule, median_nn, seeds)
    raw_trials = trials["trial_displacements_raw"]
    raw = compute_motion_fields(raw_trials, sigma_max)
    raw_xyz = canonical + raw["mean"]

    raw_support = trials["raw_stack"].sum(0)
    visible_mass = trials["visible_stack"].sum(0)
    valid_visible_mass = trials["valid_stack"].sum(0)
    visibility_ratio = (visible_mass / raw_support.clamp_min(1.0e-8)).clamp(0.0, 1.0)
    valid_fraction = (valid_visible_mass / visible_mass.clamp_min(1.0e-8)).clamp(0.0, 1.0)
    observed = valid_visible_mass > 1.0e-8
    if float(visibility_ratio.min()) < -1.0e-6 or float(visibility_ratio.max()) > 1.0 + 1.0e-6:
        raise RuntimeError("visibility_ratio outside [0,1]")
    if float(valid_fraction.min()) < -1.0e-6 or float(valid_fraction.max()) > 1.0 + 1.0e-6:
        raise RuntimeError("valid_fraction outside [0,1]")

    weights = trials["valid_stack"]
    moved_rgb = _weighted_mean(trials["rgb_stack"], weights)
    moved_depth = _weighted_mean(trials["depth_stack"], weights)
    moved_combined = _weighted_mean(trials["combined_stack"], weights)
    static_e = static_eval["evidence"]
    rgb_counterfactual = _signed_log_ratio(static_e["rgb_evidence"], moved_rgb)
    depth_counterfactual = _signed_log_ratio(static_e["depth_evidence"], moved_depth)
    combined_counterfactual = _signed_log_ratio(static_e["combined_residual"], moved_combined)

    raw_eval = _render_eval(target_cameras, gaussians, pipe, background, raw_xyz.detach(), rotations, args)
    null_trials = corrected_trials = corrected = null = corrected_eval = corrected_xyz = None
    render_block = {"canonical": _loss_dict(static_eval), "raw_moved": _loss_dict(raw_eval)}

    if target_frame == 0:
        null_trials = raw_trials
        null = raw
        render_block["null_moved"] = _loss_dict(raw_eval)
        render_block["render_degradation"] = float(raw_eval["loss"] - static_eval["loss"])
    else:
        if not args.null_motion_file:
            raise RuntimeError("real target evaluation requires --null-motion-file")
        null_data = torch.load(Path(args.null_motion_file).expanduser().resolve(), map_location="cpu")
        _check_null_compatible(null_data, checkpoint_sha=checkpoint_sha, canonical_hash=canonical_hash, canonical=canonical, schedule=schedule, sigma_max=sigma_max, seeds=seeds, target_frame=target_frame)
        null_trials = torch.as_tensor(null_data["trial_displacements_raw"], device=device, dtype=dtype)
        null = compute_motion_fields(null_trials, sigma_max)
        corrected_trials = null_correct(raw_trials, null_trials)
        corrected = compute_motion_fields(corrected_trials, sigma_max)
        if not torch.allclose(corrected_trials, raw_trials - null_trials):
            raise RuntimeError("corrected trial tensor is not raw-null")
        corrected_xyz = canonical + corrected["mean"]
        corrected_eval = _render_eval(target_cameras, gaussians, pipe, background, corrected_xyz.detach(), rotations, args)
        render_block["corrected_moved"] = _loss_dict(corrected_eval)

    predictions_finalized = True
    gt_path = _resolve_gt(args.root_scene, args.gt_ply) if target_frame > 0 else None
    gt_dynamic = gt_part = gt_distance = None
    gt_metrics_raw = gt_metrics_corrected = threshold_sweep = fp_stats = fn_stats = None
    if gt_path is not None and gt_path.is_file():
        assert predictions_finalized
        gt_dynamic, gt_part, gt_distance = _nearest_gt_labels(canonical, gt_path)
        gt_dynamic = gt_dynamic.to(device).bool()
        gt_part = gt_part.to(device)
        gt_distance = gt_distance.to(device)
        reliable = gt_distance <= float(args.max_gt_distance)
        reliable_observed = reliable & observed
        gt_metrics_raw = {
            "reliable_all": _metrics(gt_dynamic, raw["label"], raw["ratio"], reliable),
            "reliable_observed": _metrics(gt_dynamic, raw["label"], raw["ratio"], reliable_observed),
        }
        threshold_sweep = {"raw": _threshold_sweep(gt_dynamic, raw["ratio"], {"reliable_all": reliable, "reliable_observed": reliable_observed})}
        if corrected is not None:
            gt_metrics_corrected = {
                "reliable_all": _metrics(gt_dynamic, corrected["label"], corrected["ratio"], reliable),
                "reliable_observed": _metrics(gt_dynamic, corrected["label"], corrected["ratio"], reliable_observed),
            }
            threshold_sweep["corrected"] = _threshold_sweep(gt_dynamic, corrected["ratio"], {"reliable_all": reliable, "reliable_observed": reliable_observed})
            fields = {
                "displacement_norm": corrected["norm"],
                "motion_ratio": corrected["ratio"],
                "seed_consistency": corrected["consistency"],
                "visibility_ratio": visibility_ratio,
                "valid_fraction": valid_fraction,
            }
            fp_stats = _group_stats(corrected["label"] & (~gt_dynamic) & reliable_observed, fields)
            fn_stats = _group_stats((~corrected["label"]) & gt_dynamic & reliable_observed, fields)
        primary = reliable_observed
        for prefix, fields in (("raw", raw), ("corrected", corrected)):
            if fields is None:
                continue
            pred = fields["label"]
            torch.save(torch.nonzero(pred & gt_dynamic & primary, as_tuple=False).flatten().cpu(), output_dir / f"{prefix}_true_positive_indices.pt")
            torch.save(torch.nonzero((~pred) & (~gt_dynamic) & primary, as_tuple=False).flatten().cpu(), output_dir / f"{prefix}_true_negative_indices.pt")
            torch.save(torch.nonzero(pred & (~gt_dynamic) & primary, as_tuple=False).flatten().cpu(), output_dir / f"{prefix}_false_positive_indices.pt")
            torch.save(torch.nonzero((~pred) & gt_dynamic & primary, as_tuple=False).flatten().cpu(), output_dir / f"{prefix}_false_negative_indices.pt")
    elif target_frame == 0:
        print("[GT] target 0 is null diagnostics; part-label GT metrics intentionally skipped")
    else:
        print("[GT] points3D_parts.ply not found; skipping GT metrics")

    canonical_change = float((gaussians.get_canonical_xyz.detach() - canonical_guard).abs().max())
    if canonical_change >= 1.0e-10:
        raise RuntimeError(f"canonical Gaussian geometry changed: {canonical_change}")

    per_gaussian = {
        "schema_version": SCHEMA_VERSION,
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_sha256": checkpoint_sha,
        "checkpoint_train_target": int(checkpoint_train_target),
        "source_frame": source_frame,
        "target_frame": target_frame,
        "timestamp": float(target_time),
        "delta_t": float(delta_t),
        "trial_seeds": seeds.cpu(),
        "canonical_xyz": canonical.cpu(),
        "canonical_xyz_sha256": canonical_hash,
        "sigma_max": sigma_max,
        "median_nn_distance": float(median_nn),
        "noise_schedule": schedule.detach().cpu(),
        "trial_displacements_raw": raw_trials.cpu(),
        "mean_displacement_raw": raw["mean"].cpu(),
        "displacement_norm_raw": raw["norm"].cpu(),
        "motion_ratio_raw": raw["ratio"].cpu(),
        "dynamic_label_raw": raw["label"].cpu(),
        "seed_consistency_raw": raw["consistency"].cpu(),
        "trial_null_displacements": null_trials.cpu() if null_trials is not None else None,
        "mean_null_displacement": null["mean"].cpu() if null is not None else None,
        "null_displacement_norm": null["norm"].cpu() if null is not None else None,
        "null_motion_ratio": null["ratio"].cpu() if null is not None else None,
        "trial_displacements_corrected": corrected_trials.cpu() if corrected_trials is not None else None,
        "mean_displacement_corrected": corrected["mean"].cpu() if corrected is not None else None,
        "displacement_norm_corrected": corrected["norm"].cpu() if corrected is not None else None,
        "motion_ratio_corrected": corrected["ratio"].cpu() if corrected is not None else None,
        "dynamic_label_corrected": corrected["label"].cpu() if corrected is not None else None,
        "seed_consistency_corrected": corrected["consistency"].cpu() if corrected is not None else None,
        "raw_support": raw_support.cpu(),
        "visible_mass": visible_mass.cpu(),
        "valid_visible_mass": valid_visible_mass.cpu(),
        "visibility_ratio": visibility_ratio.cpu(),
        "valid_fraction": valid_fraction.cpu(),
        "observed": observed.cpu(),
        "rgb_counterfactual": rgb_counterfactual.cpu(),
        "depth_counterfactual": depth_counterfactual.cpu(),
        "combined_counterfactual": combined_counterfactual.cpu(),
        "gt_dynamic": gt_dynamic.cpu() if gt_dynamic is not None else None,
        "gt_part": gt_part.cpu() if gt_part is not None else None,
        "gt_distance": gt_distance.cpu() if gt_distance is not None else None,
    }
    torch.save(per_gaussian, output_dir / "per_gaussian_motion.pt")
    if trials["trajectory"] is not None:
        ids = _subset(canonical.shape[0], int(args.max_html_points))
        torch.save({"indices": ids, "trajectory": torch.stack([x[ids] for x in trials["trajectory"]])}, output_dir / "denoising_trajectory_subset.pt")
    if threshold_sweep is not None:
        (output_dir / "threshold_sweep.json").write_text(
            json.dumps({"warning": "GT diagnostic only; do not use to set production threshold", **_json_ready(threshold_sweep)}, indent=2),
            encoding="utf-8",
        )

    if bool(args.debug_html):
        write_label_html(output_dir / "raw_static_dynamic.html", canonical, raw, "raw", raw, corrected, null, visibility_ratio, int(args.max_html_points))
        write_score_html(output_dir / "raw_motion_ratio.html", canonical, raw["ratio"], "raw motion ratio, M>1 dynamic", "Mraw", raw, corrected, null, visibility_ratio, int(args.max_html_points), clamp=(0.0, 3.0))
        write_score_html(output_dir / "raw_seed_consistency.html", canonical, raw["consistency"], "raw seed consistency", "Craw", raw, corrected, null, visibility_ratio, int(args.max_html_points), clamp=(0.0, 1.0))
        write_vectors_html(output_dir / "raw_displacement_vectors.html", canonical, raw["mean"], raw["label"], "raw displacement vectors", raw, corrected, null, visibility_ratio, int(args.max_html_points))
        write_score_html(output_dir / "visibility_ratio.html", canonical, visibility_ratio, "visibility ratio", "visibility", raw, corrected, null, visibility_ratio, int(args.max_html_points), clamp=(0.0, 1.0))
        write_score_html(output_dir / "valid_fraction.html", canonical, valid_fraction, "valid fraction", "valid_fraction", raw, corrected, null, visibility_ratio, int(args.max_html_points), clamp=(0.0, 1.0))
        if target_frame == 0:
            write_score_html(output_dir / "null_motion_ratio.html", canonical, raw["ratio"], "null motion ratio", "Mnull", raw, None, raw, visibility_ratio, int(args.max_html_points), clamp=(0.0, 3.0))
            write_score_html(output_dir / "null_seed_consistency.html", canonical, raw["consistency"], "null seed consistency", "Cnull", raw, None, raw, visibility_ratio, int(args.max_html_points), clamp=(0.0, 1.0))
            write_vectors_html(output_dir / "null_bias_vectors.html", canonical, raw["mean"], raw["label"], "null bias vectors", raw, None, raw, visibility_ratio, int(args.max_html_points))
        if corrected is not None:
            write_label_html(output_dir / "corrected_static_dynamic.html", canonical, corrected, "corrected", raw, corrected, null, visibility_ratio, int(args.max_html_points))
            write_score_html(output_dir / "corrected_motion_ratio.html", canonical, corrected["ratio"], "corrected motion ratio, M>1 dynamic", "Mcorr", raw, corrected, null, visibility_ratio, int(args.max_html_points), clamp=(0.0, 3.0))
            write_score_html(output_dir / "corrected_seed_consistency.html", canonical, corrected["consistency"], "corrected seed consistency", "Ccorr", raw, corrected, null, visibility_ratio, int(args.max_html_points), clamp=(0.0, 1.0))
            write_vectors_html(output_dir / "corrected_displacement_vectors.html", canonical, corrected["mean"], corrected["label"], "corrected displacement vectors", raw, corrected, null, visibility_ratio, int(args.max_html_points))
        if gt_dynamic is not None:
            reliable = gt_distance <= float(args.max_gt_distance)
            write_gt_htmls("raw", output_dir, canonical, gt_dynamic, gt_distance, reliable, raw, raw, corrected, null, visibility_ratio, int(args.max_html_points))
            if corrected is not None:
                write_gt_htmls("corrected", output_dir, canonical, gt_dynamic, gt_distance, reliable, corrected, raw, corrected, null, visibility_ratio, int(args.max_html_points))

    if bool(args.debug_2d):
        write_debug_2d(output_dir, target_cameras, gaussians, pipe, background, canonical, rotations, raw_xyz, corrected_xyz, args)

    if target_frame == 0:
        summary = {
            "schema_version": SCHEMA_VERSION,
            "mode": "null",
            "source_frame": source_frame,
            "target_frame": target_frame,
            "checkpoint_train_target": int(checkpoint_train_target),
            "num_gaussians": int(canonical.shape[0]),
            "sigma_max": sigma_max,
            "observed_fraction": float(observed.float().mean()),
            "null_displacement_quantiles": _quantiles(raw["norm"]),
            "null_motion_ratio_quantiles": _quantiles(raw["ratio"]),
            "null_seed_consistency_quantiles": _quantiles(raw["consistency"]),
            "fraction_null_motion_ratio_gt_1": float(raw["label"].float().mean()),
            "visibility_conditioned_null_ratio_statistics": _visibility_conditioned_null_stats(raw["ratio"], visibility_ratio),
            "canonical_render": _loss_dict(static_eval),
            "null_moved_render": _loss_dict(raw_eval),
            "render_degradation": float(raw_eval["loss"] - static_eval["loss"]),
            "canonical_geometry_max_change": canonical_change,
        }
        (output_dir / "null_bias_summary.json").write_text(json.dumps(_json_ready(summary), indent=2), encoding="utf-8")
        _write_histograms(
            output_dir / "null_bias_histograms.png",
            {
                "null_displacement_norm": raw["norm"],
                "null_motion_ratio": raw["ratio"],
                "null_seed_consistency": raw["consistency"],
            },
        )
    else:
        summary = {
            "schema_version": SCHEMA_VERSION,
            "source_frame": source_frame,
            "target_frame": target_frame,
            "checkpoint_train_target": int(checkpoint_train_target),
            "num_gaussians": int(canonical.shape[0]),
            "sigma_max": sigma_max,
            "observed_fraction": float(observed.float().mean()),
            "raw": _summary_block(raw, gt_metrics_raw),
            "corrected": _summary_block(corrected, gt_metrics_corrected) if corrected is not None else None,
            "render": render_block,
            "canonical_geometry_max_change": canonical_change,
            "gt_ply": str(gt_path) if gt_path is not None else None,
            "fp_stats": fp_stats,
            "fn_stats": fn_stats,
        }
    (output_dir / "motion_summary.json").write_text(json.dumps(_json_ready(summary), indent=2), encoding="utf-8")
    print(json.dumps(_json_ready(summary), indent=2))
    print("[COMPLETE]", output_dir)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root-scene", required=True)
    p.add_argument("--static-model-dir", required=True)
    p.add_argument("--iteration", type=int, default=30000)
    p.add_argument("--denoiser-checkpoint", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--null-motion-file", default="")
    p.add_argument("--source-frame", type=int, default=0)
    p.add_argument("--target-frame", type=int, required=True)
    p.add_argument("--inference-trials", type=int, default=16)
    p.add_argument("--knn", type=int, default=16)
    p.add_argument("--use-depth", action=BooleanOptionalAction, default=True)
    p.add_argument("--lambda-rgb", type=float, default=1.0)
    p.add_argument("--lambda-ssim", type=float, default=0.2)
    p.add_argument("--lambda-depth", type=float, default=0.1)
    p.add_argument("--gt-ply", default="")
    p.add_argument("--max-gt-distance", type=float, default=0.02)
    p.add_argument("--max-html-points", type=int, default=5000)
    p.add_argument("--seed", type=int, default=20260901)
    p.add_argument("--fps", type=float, default=30.0)
    p.add_argument("--resolution", type=int, default=None)
    p.add_argument("--sh-degree", type=int, default=None)
    p.add_argument("--cameras", default="")
    p.add_argument(
        "--camera-data-device",
        default="cpu",
        help="Device for cached camera RGB/depth tensors; cpu keeps only the selected render batch on CUDA.",
    )
    p.add_argument("--debug-2d", action=BooleanOptionalAction, default=True)
    p.add_argument("--debug-html", action=BooleanOptionalAction, default=True)
    bg = p.add_mutually_exclusive_group()
    bg.add_argument("--white-background", dest="white_background", action="store_true")
    bg.add_argument("--black-background", dest="white_background", action="store_false")
    p.set_defaults(white_background=None)
    args = p.parse_args()
    args.inference_trials = max(1, int(args.inference_trials))
    args.knn = max(1, int(args.knn))
    args.max_html_points = max(10, int(args.max_html_points))
    args.lambda_rgb = max(0.0, float(args.lambda_rgb))
    args.lambda_ssim = max(0.0, float(args.lambda_ssim))
    args.lambda_depth = max(0.0, float(args.lambda_depth))
    return args


if __name__ == "__main__":
    run(parse_args())
