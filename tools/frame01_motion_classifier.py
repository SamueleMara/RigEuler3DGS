#!/usr/bin/env python3
"""Frame-0 -> frame-1 static/dynamic classifier from frozen v1 denoiser motion.

Hard classifier:
    dynamic_i = ||mean_trial_displacement_i|| / sigma_max > 1.0

No probability head. No classifier training. No GT in the algorithm.
"""

from __future__ import annotations

import argparse
import json
import sys
from argparse import BooleanOptionalAction
from pathlib import Path

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

from scene.gaussian_motion_denoiser import build_canonical_knn  # noqa: E402
from tools.evaluate_frame01_counterfactual_v3 import (  # noqa: E402
    _camera_id,
    _frame_maps,
    _frame_time,
    _load_v1_checkpoint,
    _nearest_gt_labels,
    _quantiles,
    _render_loss_and_evidence,
    _run_v1_denoising_pass,
    _signed_log_ratio,
)
from tools.test_render_denoise_discovery import build_scene  # noqa: E402


def _json_ready(value):
    if isinstance(value, torch.Tensor):
        return _json_ready(value.detach().cpu().tolist())
    if isinstance(value, np.ndarray):
        return _json_ready(value.tolist())
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(k): _json_ready(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(v) for v in value]
    return value


def _limit_indices(indices: torch.Tensor, max_count: int) -> torch.Tensor:
    indices = indices.detach().cpu().long().flatten()
    if indices.numel() <= max_count:
        return indices
    keep = torch.linspace(0, indices.numel() - 1, max_count).round().long().unique()
    return indices[keep]


def _all_indices(n: int, max_count: int) -> torch.Tensor:
    return _limit_indices(torch.arange(n), max_count)


def _html(path: Path, traces, title: str, extra_layout: dict | None = None):
    layout = {"scene": {"aspectmode": "data"}, "title": title}
    if extra_layout:
        layout.update(extra_layout)
    path.write_text(
        "<!doctype html><html><head><meta charset='utf-8'>"
        "<script src='https://cdn.plot.ly/plotly-2.35.2.min.js'></script>"
        "</head><body><div id='p' style='height:95vh'></div><script>"
        f"Plotly.newPlot('p',{json.dumps(traces)},{json.dumps(layout)});"
        "</script></body></html>",
        encoding="utf-8",
    )


def _trace_points(name, xyz, indices, color, size=3, opacity=0.8, text=None):
    pts = xyz[indices].detach().cpu().float()
    return {
        "type": "scatter3d",
        "mode": "markers",
        "name": name,
        "x": pts[:, 0].tolist(),
        "y": pts[:, 1].tolist(),
        "z": pts[:, 2].tolist(),
        "marker": {"size": size, "color": color, "opacity": opacity},
        "text": text if text is not None else [str(int(i)) for i in indices],
        "hoverinfo": "text",
    }


def _metric_counts(gt_dynamic, pred_dynamic, mask):
    gt = gt_dynamic[mask].bool()
    pred = pred_dynamic[mask].bool()
    tp = int((pred & gt).sum())
    fp = int((pred & ~gt).sum())
    fn = int((~pred & gt).sum())
    tn = int((~pred & ~gt).sum())
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2.0 * precision * recall / max(precision + recall, 1.0e-12)
    return {
        "n": int(mask.sum()),
        "n_dynamic": int(gt.sum()),
        "n_static": int((~gt).sum()),
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "iou": tp / max(tp + fp + fn, 1),
        "accuracy": (tp + tn) / max(tp + fp + fn + tn, 1),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
    }


def _ranking_metrics(gt_dynamic, score, mask):
    y = gt_dynamic[mask].detach().cpu().numpy().astype(np.int64)
    s = score[mask].detach().cpu().numpy().astype(np.float64)
    valid = np.isfinite(s)
    y, s = y[valid], s[valid]
    pos = int(y.sum())
    neg = int(y.size - pos)
    if y.size == 0 or pos == 0 or neg == 0:
        return {"roc_auc": None, "pr_auc": None}

    order = np.argsort(-s, kind="mergesort")
    ys = y[order]
    tp = np.cumsum(ys == 1)
    fp = np.cumsum(ys == 0)
    precision = tp / np.maximum(tp + fp, 1)
    recall = tp / max(pos, 1)
    fpr = fp / max(neg, 1)
    roc_x = np.concatenate([[0.0], fpr, [1.0]]).astype(np.float64)
    roc_y = np.concatenate([[0.0], recall, [1.0]]).astype(np.float64)
    roc_auc = float(np.sum((roc_x[1:] - roc_x[:-1]) * 0.5 * (roc_y[1:] + roc_y[:-1])))
    recall_prev = np.concatenate([[0.0], recall[:-1]])
    pr_auc = float(np.sum((recall - recall_prev) * precision))
    return {"roc_auc": roc_auc, "pr_auc": pr_auc}


def _metrics(gt_dynamic, pred_dynamic, motion_ratio, mask):
    out = _metric_counts(gt_dynamic, pred_dynamic, mask)
    out.update(_ranking_metrics(gt_dynamic, motion_ratio, mask))
    return out


def _group_stats(mask, fields):
    result = {"count": int(mask.sum())}
    for name, value in fields.items():
        x = value[mask].detach().float()
        if x.numel() == 0:
            result[name] = {"mean": None, "median": None, "p10": None, "p90": None}
        else:
            result[name] = {
                "mean": float(x.mean()),
                "median": float(x.median()),
                "p10": float(torch.quantile(x, 0.10)),
                "p90": float(torch.quantile(x, 0.90)),
            }
    return result


def _print_group_stats(name, stats):
    print(f"[{name}] count={stats['count']}")
    for key, value in stats.items():
        if key == "count":
            continue
        print(
            f"  {key}: mean={value['mean']} median={value['median']} "
            f"p10={value['p10']} p90={value['p90']}"
        )


def _threshold_sweep(gt_dynamic, motion_ratio, masks):
    thresholds = [0.50, 0.75, 1.00, 1.25, 1.50]
    thresholds += [0.10 / 0.15, 0.12 / 0.15, 0.18 / 0.15, 0.20 / 0.15]
    thresholds = sorted({round(float(t), 6) for t in thresholds})
    return {
        subset: [
            {"motion_ratio_threshold": t, **_metric_counts(gt_dynamic, motion_ratio > t, mask)}
            for t in thresholds
        ]
        for subset, mask in masks.items()
    }


def _write_static_dynamic_html(path, xyz, label, disp, ratio, consistency, visibility, valid_fraction, max_points):
    ids = _all_indices(xyz.shape[0], max_points)
    traces = []
    for value, name, color in ((0, "static", "#3267d6"), (1, "dynamic", "#d33f2f")):
        part = ids[label[ids].cpu() == value]
        text = [
            f"idx={int(i)}<br>label={name}<br>displacement_norm={float(disp[i]):.8f}"
            f"<br>motion_ratio={float(ratio[i]):.4f}<br>seed_consistency={float(consistency[i]):.4f}"
            f"<br>visibility_ratio={float(visibility[i]):.4f}<br>valid_fraction={float(valid_fraction[i]):.4f}"
            for i in part
        ]
        traces.append(_trace_points(name, xyz, part, color, text=text))
    _html(path, traces, "Frame01 hard classification: dynamic iff motion_ratio > 1.0")


def _write_continuous_html(path, xyz, value, title, colorbar, max_points, clamp=None):
    ids = _all_indices(xyz.shape[0], max_points)
    pts = xyz[ids].detach().cpu().float()
    raw = value[ids].detach().cpu().float()
    shown = raw.clamp(*clamp) if clamp is not None else raw
    trace = {
        "type": "scatter3d",
        "mode": "markers",
        "x": pts[:, 0].tolist(),
        "y": pts[:, 1].tolist(),
        "z": pts[:, 2].tolist(),
        "marker": {
            "size": 3,
            "color": shown.tolist(),
            "colorscale": "Turbo",
            "opacity": 0.8,
            "colorbar": {"title": colorbar},
        },
        "text": [f"idx={int(i)}<br>{colorbar}={float(value[i]):.6f}" for i in ids],
        "hoverinfo": "text",
    }
    _html(path, [trace], title)


def _write_confusion_html(path, xyz, gt, pred, reliable, disp, ratio, consistency, gt_dist, visibility, max_points):
    classes = [
        ("TP", pred & gt & reliable, "#d33f2f"),
        ("TN", (~pred) & (~gt) & reliable, "#3267d6"),
        ("FP", pred & (~gt) & reliable, "#ff9f1c"),
        ("FN", (~pred) & gt & reliable, "#7b2cbf"),
    ]
    traces = []
    for name, mask, color in classes:
        ids = _limit_indices(torch.nonzero(mask, as_tuple=False).flatten(), max_points)
        text = [
            f"idx={int(i)}<br>confusion={name}<br>gt={'dynamic' if bool(gt[i]) else 'static'}"
            f"<br>pred={'dynamic' if bool(pred[i]) else 'static'}"
            f"<br>displacement_norm={float(disp[i]):.8f}<br>motion_ratio={float(ratio[i]):.4f}"
            f"<br>seed_consistency={float(consistency[i]):.4f}<br>gt_distance={float(gt_dist[i]):.6f}"
            f"<br>visibility_ratio={float(visibility[i]):.4f}"
            for i in ids
        ]
        traces.append(_trace_points(name, xyz, ids, color, text=text))
    _html(path, traces, "Frame01 GT confusion at motion_ratio > 1.0")


def _write_error_html(path, title, xyz, error_mask, reliable, ratio, disp, consistency, visibility, gt_dist, max_points):
    context_ids = _limit_indices(torch.nonzero(reliable, as_tuple=False).flatten(), max_points)
    error_ids = _limit_indices(torch.nonzero(error_mask, as_tuple=False).flatten(), max_points)
    error_text = [
        f"idx={int(i)}<br>motion_ratio={float(ratio[i]):.4f}"
        f"<br>displacement_norm={float(disp[i]):.8f}<br>seed_consistency={float(consistency[i]):.4f}"
        f"<br>visibility_ratio={float(visibility[i]):.4f}<br>gt_distance={float(gt_dist[i]):.6f}"
        for i in error_ids
    ]
    traces = [
        _trace_points("reliable GT context", xyz, context_ids, "#bbbbbb", size=2, opacity=0.18),
        _trace_points(title, xyz, error_ids, "#ff006e", size=6, opacity=0.95, text=error_text),
    ]
    _html(path, traces, title)


def _write_vector_html(path, xyz, displacement, label, max_points):
    ids = _all_indices(xyz.shape[0], min(max_points, 1000))
    pts = xyz[ids].detach().cpu().float()
    disp = displacement[ids].detach().cpu().float()
    lab = label[ids].detach().cpu().long()
    traces = []
    for value, name, color in ((0, "static vectors", "#3267d6"), (1, "dynamic vectors", "#d33f2f")):
        mask = lab == value
        part_ids = torch.nonzero(mask, as_tuple=False).flatten()
        line = {"type": "scatter3d", "mode": "lines", "name": name, "x": [], "y": [], "z": [], "line": {"color": color, "width": 3}}
        for j in part_ids:
            p = pts[j]
            q = pts[j] + disp[j]
            line["x"] += [float(p[0]), float(q[0]), None]
            line["y"] += [float(p[1]), float(q[1]), None]
            line["z"] += [float(p[2]), float(q[2]), None]
        traces.append(line)
    _html(path, traces, "Frame01 mean displacement vectors")


def _run_trials(args, target_cameras, gaussians, pipe, background, canonical, rotations, denoiser, knn_idx, schedule, median_nn):
    trial_displacements = []
    rgb_stack = []
    depth_stack = []
    combined_stack = []
    valid_stack = []
    raw_stack = []
    visible_stack = []
    trajectory = None

    for trial in tqdm(range(int(args.inference_trials)), desc="Frame01 frozen-v1 trials", unit="trial"):
        generator = torch.Generator(device=canonical.device)
        generator.manual_seed(int(args.seed) + 100_000 + trial)
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
        assert xyz.shape == canonical.shape
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
        "trial_displacements": torch.stack(trial_displacements),
        "rgb_stack": torch.stack(rgb_stack),
        "depth_stack": torch.stack(depth_stack),
        "combined_stack": torch.stack(combined_stack),
        "valid_stack": torch.stack(valid_stack),
        "raw_stack": torch.stack(raw_stack),
        "visible_stack": torch.stack(visible_stack),
        "trajectory": trajectory,
    }


def run(args):
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")

    torch.manual_seed(int(args.seed))
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    scene, gaussians, pipe, white_background = build_scene(args)
    args.white_background = white_background
    frame_map = _frame_maps(scene)
    if int(args.source_frame) not in frame_map:
        raise RuntimeError(f"source frame {args.source_frame} not available")
    if int(args.target_frame) not in frame_map:
        raise RuntimeError(f"target frame {args.target_frame} not available")

    source_cameras = frame_map[int(args.source_frame)]
    target_cameras = sorted(frame_map[int(args.target_frame)], key=_camera_id)
    source_time = _frame_time(source_cameras, int(args.source_frame), float(args.fps))
    target_time = _frame_time(target_cameras, int(args.target_frame), float(args.fps))
    delta_t = target_time - source_time
    if delta_t <= 0:
        delta_t = 1.0 / max(float(args.fps), 1.0e-8)

    device = gaussians.get_canonical_xyz.device
    dtype = gaussians.get_canonical_xyz.dtype
    canonical = gaussians.get_canonical_xyz.detach().clone()
    rotations = gaussians.get_rotation.detach().clone()
    canonical_guard = canonical.clone()
    for tensor in (gaussians._xyz, gaussians._features_dc, gaussians._features_rest, gaussians._opacity, gaussians._scaling, gaussians._rotation):
        tensor.requires_grad_(False)

    background = torch.tensor([1.0, 1.0, 1.0] if white_background else [0.0, 0.0, 0.0], device=device, dtype=dtype)
    denoiser, payload = _load_v1_checkpoint(args.denoiser_checkpoint, device)
    if "noise_schedule" not in payload:
        raise RuntimeError("v1 denoiser checkpoint does not contain noise_schedule")
    schedule = torch.as_tensor(payload["noise_schedule"], device=device, dtype=dtype).flatten()
    if schedule.numel() != 6:
        raise RuntimeError(f"expected the v1 six-step checkpoint schedule, got {schedule.numel()} steps")

    knn_idx, median_nn = build_canonical_knn(canonical, int(args.knn))
    knn_idx = knn_idx.to(device)
    median_nn = median_nn.to(device=device, dtype=dtype)
    sigma_max = float(schedule[0].detach().cpu())

    print("[Frame01]")
    print(f"  num_gaussians={canonical.shape[0]} num_cameras={len(target_cameras)}")
    print(f"  median_nn={float(median_nn):.12f} sigma_max={sigma_max:.12f}")
    print(f"  noise_schedule={[float(x) for x in schedule.detach().cpu()]}")
    print(f"  inference_trials={int(args.inference_trials)}")

    static_eval = _render_loss_and_evidence(target_cameras, gaussians, pipe, background, canonical, rotations, args, collect_evidence=True)
    trials = _run_trials(args, target_cameras, gaussians, pipe, background, canonical, rotations, denoiser, knn_idx, schedule, median_nn)

    trial_displacements = trials["trial_displacements"]
    assert trial_displacements.ndim == 3 and trial_displacements.shape[1:] == canonical.shape
    mean_displacement = trial_displacements.mean(0)
    mean_predicted_xyz = canonical + mean_displacement
    displacement_norm = torch.linalg.norm(mean_displacement, dim=-1)
    motion_ratio = displacement_norm / max(sigma_max, 1.0e-8)
    dynamic_label = motion_ratio > 1.0
    assert dynamic_label.dtype == torch.bool
    part_label = torch.where(dynamic_label, torch.ones_like(motion_ratio, dtype=torch.long), torch.zeros_like(motion_ratio, dtype=torch.long))
    trial_norm = torch.linalg.norm(trial_displacements, dim=-1)
    seed_consistency = (displacement_norm / trial_norm.mean(0).clamp_min(1.0e-8)).clamp(0.0, 1.0)
    motion_confidence = seed_consistency

    for name, tensor in (
        ("mean_displacement", mean_displacement),
        ("displacement_norm", displacement_norm),
        ("motion_ratio", motion_ratio),
        ("seed_consistency", seed_consistency),
    ):
        if not bool(torch.isfinite(tensor).all()):
            raise RuntimeError(f"{name} contains non-finite values")

    raw_support = trials["raw_stack"].sum(0)
    visible_mass = trials["visible_stack"].sum(0)
    valid_visible_mass = trials["valid_stack"].sum(0)
    visibility_ratio = (visible_mass / raw_support.clamp_min(1.0e-8)).clamp(0.0, 1.0)
    valid_fraction = (valid_visible_mass / visible_mass.clamp_min(1.0e-8)).clamp(0.0, 1.0)
    observed = valid_visible_mass > 1.0e-8

    weights = trials["valid_stack"]
    moved_rgb = (trials["rgb_stack"] * weights).sum(0) / weights.sum(0).clamp_min(1.0e-8)
    moved_depth = (trials["depth_stack"] * weights).sum(0) / weights.sum(0).clamp_min(1.0e-8)
    moved_combined = (trials["combined_stack"] * weights).sum(0) / weights.sum(0).clamp_min(1.0e-8)
    static_e = static_eval["evidence"]
    rgb_counterfactual = _signed_log_ratio(static_e["rgb_evidence"], moved_rgb)
    depth_counterfactual = _signed_log_ratio(static_e["depth_evidence"], moved_depth)
    combined_counterfactual = _signed_log_ratio(static_e["combined_residual"], moved_combined)

    moved_eval = _render_loss_and_evidence(target_cameras, gaussians, pipe, background, mean_predicted_xyz.detach(), rotations, args, collect_evidence=False)
    print("[Render sanity]")
    print(f"  static_total={float(static_eval['loss']):.9f} moved_total={float(moved_eval['loss']):.9f}")
    print(f"  static_rgb={float(static_eval['rgb']):.9f} moved_rgb={float(moved_eval['rgb']):.9f}")

    print("[Motion]")
    print(f"  displacement={_quantiles(displacement_norm)}")
    print(f"  motion_ratio={_quantiles(motion_ratio)}")
    print(f"  seed_consistency={_quantiles(seed_consistency)}")
    print(f"  static={int((~dynamic_label).sum())} dynamic={int(dynamic_label.sum())} dynamic_fraction={float(dynamic_label.float().mean()):.6f}")

    gt_path = Path(args.gt_ply).expanduser().resolve() if args.gt_ply else None
    if gt_path is None or not gt_path.is_file():
        candidates = [Path(args.root_scene) / "pointcloud" / "points3D_parts.ply", Path(args.root_scene) / "points3D_parts.ply"]
        gt_path = next((p for p in candidates if p.is_file()), None)

    gt_dynamic = gt_part = gt_distance = None
    gt_metrics = None
    fp = fn = tp = tn = torch.empty(0, device=device, dtype=torch.long)
    threshold_sweep = None
    fp_stats = fn_stats = None
    if gt_path is not None and gt_path.is_file():
        gt_dynamic, gt_part, gt_distance = _nearest_gt_labels(canonical, gt_path)
        gt_dynamic = gt_dynamic.to(device).bool()
        gt_part = gt_part.to(device)
        gt_distance = gt_distance.to(device)
        reliable = gt_distance <= float(args.max_gt_distance)
        reliable_observed = reliable & observed
        gt_metrics = {
            "reliable_all": _metrics(gt_dynamic, dynamic_label, motion_ratio, reliable),
            "reliable_observed": _metrics(gt_dynamic, dynamic_label, motion_ratio, reliable_observed),
        }
        threshold_sweep = _threshold_sweep(gt_dynamic, motion_ratio, {"reliable_all": reliable, "reliable_observed": reliable_observed})

        primary = reliable_observed
        tp = torch.nonzero(dynamic_label & gt_dynamic & primary, as_tuple=False).flatten()
        tn = torch.nonzero((~dynamic_label) & (~gt_dynamic) & primary, as_tuple=False).flatten()
        fp = torch.nonzero(dynamic_label & (~gt_dynamic) & primary, as_tuple=False).flatten()
        fn = torch.nonzero((~dynamic_label) & gt_dynamic & primary, as_tuple=False).flatten()
        fields = {
            "displacement_norm": displacement_norm,
            "motion_ratio": motion_ratio,
            "seed_consistency": seed_consistency,
            "visibility_ratio": visibility_ratio,
            "valid_fraction": valid_fraction,
        }
        fp_stats = _group_stats(dynamic_label & (~gt_dynamic) & primary, fields)
        fn_stats = _group_stats((~dynamic_label) & gt_dynamic & primary, fields)

        print("[GT]")
        for name, metrics in gt_metrics.items():
            print(
                f"  {name}: n={metrics['n']} precision={metrics['precision']:.4f} "
                f"recall={metrics['recall']:.4f} f1={metrics['f1']:.4f} "
                f"iou={metrics['iou']:.4f} acc={metrics['accuracy']:.4f} "
                f"TP={metrics['tp']} FP={metrics['fp']} FN={metrics['fn']} TN={metrics['tn']} "
                f"ROC={metrics['roc_auc']} PR={metrics['pr_auc']}"
            )
        _print_group_stats("False positives", fp_stats)
        _print_group_stats("False negatives", fn_stats)

        torch.save(tp.cpu(), output_dir / "true_positive_indices.pt")
        torch.save(tn.cpu(), output_dir / "true_negative_indices.pt")
        torch.save(fp.cpu(), output_dir / "false_positive_indices.pt")
        torch.save(fn.cpu(), output_dir / "false_negative_indices.pt")
        torch.save(
            {
                "subset": "reliable_observed",
                "tp_indices": tp.cpu(),
                "tn_indices": tn.cpu(),
                "fp_indices": fp.cpu(),
                "fn_indices": fn.cpu(),
                "gt_dynamic": gt_dynamic.cpu(),
                "gt_part": gt_part.cpu(),
                "gt_distance": gt_distance.cpu(),
            },
            output_dir / "frame01_gt_confusion.pt",
        )
        (output_dir / "frame01_threshold_sweep.json").write_text(json.dumps(_json_ready(threshold_sweep), indent=2), encoding="utf-8")
        _write_confusion_html(output_dir / "frame01_gt_confusion.html", canonical, gt_dynamic, dynamic_label, reliable, displacement_norm, motion_ratio, seed_consistency, gt_distance, visibility_ratio, int(args.max_html_points))
        _write_error_html(output_dir / "frame01_false_positives.html", "Frame01 false positives", canonical, dynamic_label & (~gt_dynamic) & reliable, reliable, motion_ratio, displacement_norm, seed_consistency, visibility_ratio, gt_distance, int(args.max_html_points))
        _write_error_html(output_dir / "frame01_false_negatives.html", "Frame01 false negatives", canonical, (~dynamic_label) & gt_dynamic & reliable, reliable, motion_ratio, displacement_norm, seed_consistency, visibility_ratio, gt_distance, int(args.max_html_points))
    else:
        print("[GT] points3D_parts.ply not found; skipping GT metrics/visualizations")

    canonical_change = float((gaussians.get_canonical_xyz.detach() - canonical_guard).abs().max())
    if canonical_change >= 1.0e-10:
        raise RuntimeError(f"canonical Gaussian geometry changed: {canonical_change}")

    per_gaussian = {
        "canonical_xyz": canonical.cpu(),
        "mean_predicted_xyz_frame1": mean_predicted_xyz.cpu(),
        "mean_displacement": mean_displacement.cpu(),
        "displacement_norm": displacement_norm.cpu(),
        "motion_ratio": motion_ratio.cpu(),
        "dynamic_label": dynamic_label.cpu(),
        "part_label": part_label.cpu(),
        "seed_consistency": seed_consistency.cpu(),
        "motion_confidence": motion_confidence.cpu(),
        "trial_displacements": trial_displacements.cpu(),
        "sigma_max": sigma_max,
        "median_nn_distance": float(median_nn),
        "raw_support": raw_support.cpu(),
        "visible_mass": visible_mass.cpu(),
        "valid_visible_mass": valid_visible_mass.cpu(),
        "visibility_ratio": visibility_ratio.cpu(),
        "valid_fraction": valid_fraction.cpu(),
        "observed": observed.cpu(),
        "delta_t": float(delta_t),
        "rgb_counterfactual": rgb_counterfactual.cpu(),
        "depth_counterfactual": depth_counterfactual.cpu(),
        "combined_counterfactual": combined_counterfactual.cpu(),
    }
    if gt_dynamic is not None:
        per_gaussian.update({"gt_dynamic": gt_dynamic.cpu(), "gt_part": gt_part.cpu(), "gt_distance": gt_distance.cpu()})
    torch.save(per_gaussian, output_dir / "per_gaussian_frame01_motion.pt")

    if trials["trajectory"] is not None:
        ids = _all_indices(canonical.shape[0], int(args.max_html_points))
        torch.save({"indices": ids, "trajectory": torch.stack([x[ids] for x in trials["trajectory"]])}, output_dir / "denoising_trajectory_subset.pt")

    _write_static_dynamic_html(output_dir / "frame01_static_dynamic.html", canonical, part_label, displacement_norm, motion_ratio, seed_consistency, visibility_ratio, valid_fraction, int(args.max_html_points))
    _write_continuous_html(output_dir / "frame01_motion_ratio.html", canonical, motion_ratio, "Frame01 motion_ratio, dynamic threshold = 1.0", "motion_ratio", int(args.max_html_points), clamp=(0.0, 3.0))
    _write_continuous_html(output_dir / "frame01_seed_consistency.html", canonical, seed_consistency, "Frame01 seed consistency", "seed_consistency", int(args.max_html_points), clamp=(0.0, 1.0))
    _write_vector_html(output_dir / "frame01_displacement_vectors.html", canonical, mean_displacement, part_label, int(args.max_html_points))

    summary = {
        "num_gaussians": int(canonical.shape[0]),
        "source_frame": int(args.source_frame),
        "target_frame": int(args.target_frame),
        "inference_trials": int(args.inference_trials),
        "median_nn_distance": float(median_nn),
        "sigma_max": sigma_max,
        "classification_threshold_motion_ratio": 1.0,
        "num_static": int((~dynamic_label).sum()),
        "num_dynamic": int(dynamic_label.sum()),
        "dynamic_fraction": float(dynamic_label.float().mean()),
        "motion_ratio_quantiles": _quantiles(motion_ratio),
        "displacement_norm_quantiles": _quantiles(displacement_norm),
        "seed_consistency_quantiles": _quantiles(seed_consistency),
        "observed_fraction": float(observed.float().mean()),
        "static_render_loss": float(static_eval["loss"]),
        "moved_render_loss": float(moved_eval["loss"]),
        "static_rgb_loss": float(static_eval["rgb"]),
        "moved_rgb_loss": float(moved_eval["rgb"]),
        "canonical_geometry_max_change": canonical_change,
        "gt_ply": str(gt_path) if gt_path is not None else None,
        "gt_metrics": gt_metrics,
        "fp_stats": fp_stats,
        "fn_stats": fn_stats,
    }
    (output_dir / "frame01_motion_summary.json").write_text(json.dumps(_json_ready(summary), indent=2), encoding="utf-8")
    print("[COMPLETE]", output_dir)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root-scene", required=True)
    p.add_argument("--static-model-dir", required=True)
    p.add_argument("--iteration", type=int, default=30000)
    p.add_argument("--denoiser-checkpoint", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--source-frame", type=int, default=0)
    p.add_argument("--target-frame", type=int, default=1)
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
