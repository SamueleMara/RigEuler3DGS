#!/usr/bin/env python3
"""Orchestrate frame-0 multitarget denoiser training/evaluation with null correction."""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from argparse import BooleanOptionalAction
from pathlib import Path

import cv2
import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.frame0_motion_evaluator import _html, sha256_file  # noqa: E402


def _parse_ints(text: str) -> list[int]:
    if not text:
        return []
    return [int(x) for x in text.replace(" ", "").split(",") if x]


def _parse_checkpoints(items: list[str]) -> dict[int, Path]:
    result = {}
    for item in items:
        if "=" not in item:
            raise ValueError(f"--checkpoint must be target=path, got {item!r}")
        key, value = item.split("=", 1)
        result[int(key)] = Path(value).expanduser().resolve()
    return result


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
    if isinstance(value, Path):
        return str(value)
    return value


def _run(cmd: list[str]):
    print("[RUN]", " ".join(cmd))
    subprocess.run(cmd, cwd=str(REPO_ROOT), check=True)


def _discover_last_frame(root_scene: Path) -> int:
    frames = set()
    for path in root_scene.rglob("*"):
        if path.is_file() and path.stem.isdigit():
            frames.add(int(path.stem))
    return max(frames) if frames else 59


def _eval_targets(args) -> list[int]:
    explicit = _parse_ints(args.eval_targets)
    if explicit:
        return sorted(set(explicit))
    n = int(args.num_eval_targets)
    if n <= 1:
        return [1]
    last = _discover_last_frame(Path(args.root_scene).expanduser().resolve())
    return sorted(set(int(x) for x in np.rint(np.linspace(1, last, n)).astype(np.int64)))


def _checkpoint_output_path(output_root: Path, target: int) -> Path:
    return output_root / "checkpoints" / f"target_{target:06d}" / f"denoiser_frame0_to_{target}.pt"


def _train_or_reuse(args, output_root: Path, target: int, explicit: dict[int, Path]) -> Path:
    if target in explicit:
        path = explicit[target]
        if not path.is_file():
            raise FileNotFoundError(path)
        return path

    checkpoint = _checkpoint_output_path(output_root, target)
    if checkpoint.is_file():
        if args.skip_existing and not args.force:
            print(f"[reuse checkpoint] {checkpoint}")
            return checkpoint
        if not args.force:
            raise RuntimeError(f"checkpoint exists; use --skip-existing or --force: {checkpoint}")
    elif not args.train:
        raise RuntimeError(f"checkpoint missing and --no-train was supplied: {checkpoint}")

    out_dir = checkpoint.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable,
        "tools/train_frame0_target_denoiser.py",
        "--root-scene",
        args.root_scene,
        "--static-model-dir",
        args.static_model_dir,
        "--iteration",
        str(args.iteration),
        "--output-dir",
        str(out_dir),
        "--source-frame",
        "0",
        "--target-frame",
        str(target),
        "--training-iterations",
        str(args.training_iterations),
        "--denoise-iterations",
        "6",
        "--knn",
        str(args.knn),
        "--hidden-dim",
        "96",
        "--message-passing-layers",
        "2",
        "--noise-max-ratio",
        "0.15",
        "--noise-min-ratio",
        "0.02",
        "--lr",
        "0.001",
        "--weight-decay",
        "0.00001",
        "--lambda-rgb",
        "1.0",
        "--lambda-ssim",
        "0.2",
        "--lambda-depth",
        "0.1",
        "--lambda-iter",
        "0.25",
        "--lambda-spatial",
        "0.005",
        "--lambda-motion",
        "0.002",
        "--train-cameras-per-iter",
        str(args.train_cameras_per_iter),
        "--seed",
        str(args.seed),
    ]
    if str(args.cameras).strip():
        cmd.extend(["--cameras", str(args.cameras).strip()])
    if str(args.camera_data_device).strip():
        cmd.extend(["--camera-data-device", str(args.camera_data_device).strip()])
    cmd.append("--use-depth" if args.use_depth else "--no-use-depth")
    _run(cmd)
    if not checkpoint.is_file():
        raise RuntimeError(f"trainer did not create checkpoint: {checkpoint}")
    return checkpoint


def _evaluate(args, checkpoint: Path, checkpoint_target: int, target: int, output_dir: Path, null_file: Path | None):
    result_file = output_dir / "per_gaussian_motion.pt"
    if result_file.is_file() and args.skip_existing and not args.force:
        print(f"[reuse evaluation] {result_file}")
        return result_file
    if result_file.is_file() and not args.force:
        raise RuntimeError(f"evaluation exists; use --skip-existing or --force: {result_file}")
    output_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable,
        "tools/frame0_motion_evaluator.py",
        "--root-scene",
        args.root_scene,
        "--static-model-dir",
        args.static_model_dir,
        "--iteration",
        str(args.iteration),
        "--denoiser-checkpoint",
        str(checkpoint),
        "--output-dir",
        str(output_dir),
        "--source-frame",
        "0",
        "--target-frame",
        str(target),
        "--inference-trials",
        str(args.inference_trials),
        "--knn",
        str(args.knn),
        "--seed",
        str(args.seed),
        "--max-gt-distance",
        str(args.max_gt_distance),
        "--max-html-points",
        str(args.max_html_points),
    ]
    if str(args.cameras).strip():
        cmd.extend(["--cameras", str(args.cameras).strip()])
    if str(args.camera_data_device).strip():
        cmd.extend(["--camera-data-device", str(args.camera_data_device).strip()])
    if args.gt_ply:
        cmd += ["--gt-ply", args.gt_ply]
    if null_file is not None:
        cmd += ["--null-motion-file", str(null_file)]
    cmd.append("--use-depth" if args.use_depth else "--no-use-depth")
    cmd.append("--debug-2d" if args.debug_2d else "--no-debug-2d")
    cmd.append("--debug-html" if args.debug_html else "--no-debug-html")
    _run(cmd)
    if not result_file.is_file():
        raise RuntimeError(f"evaluator did not create {result_file}")
    return result_file


def _scores(motion_ratio: torch.Tensor, target_frames: torch.Tensor) -> dict:
    eps = 1.0e-8
    label = motion_ratio > 1.0
    count = label.sum(0)
    first = torch.full((motion_ratio.shape[1],), -1, dtype=torch.long)
    for t_idx, frame in enumerate(target_frames.tolist()):
        hit = label[t_idx] & (first < 0)
        first[hit] = int(frame)
    return {
        "max": motion_ratio.max(0).values,
        "mean": motion_ratio.mean(0),
        "geom": torch.exp(torch.log(motion_ratio.clamp_min(eps)).mean(0)),
        "dynamic_confirmation_count": count,
        "dynamic_confirmation_fraction": count.float() / max(1, motion_ratio.shape[0]),
        "first_dynamic_frame": first,
    }


def _write_slider(path: Path, xyz: torch.Tensor, frames: list[int], values: torch.Tensor, title: str, max_points: int, cmax=3.0):
    ids = torch.linspace(0, xyz.shape[0] - 1, min(max_points, xyz.shape[0])).round().long().unique()
    pts = xyz[ids].float()
    frame_payload = []
    for idx, frame in enumerate(frames):
        v = values[idx, ids].float().clamp(0.0, cmax)
        frame_payload.append(
            {
                "name": str(frame),
                "data": [
                    {
                        "marker": {"color": v.tolist()},
                        "text": [f"idx={int(i)}<br>frame={frame}<br>value={float(values[idx, i]):.5f}" for i in ids],
                    }
                ],
            }
        )
    trace = {
        "type": "scatter3d",
        "mode": "markers",
        "x": pts[:, 0].tolist(),
        "y": pts[:, 1].tolist(),
        "z": pts[:, 2].tolist(),
        "marker": {"size": 3, "color": values[0, ids].clamp(0, cmax).tolist(), "colorscale": "Turbo", "cmin": 0, "cmax": cmax, "opacity": 0.8},
        "text": [f"idx={int(i)}<br>frame={frames[0]}<br>value={float(values[0, i]):.5f}" for i in ids],
        "hoverinfo": "text",
    }
    controls = {
        "updatemenus": [{"type": "buttons", "buttons": [{"label": "play", "method": "animate", "args": [None]}]}],
        "sliders": [{"steps": [{"label": str(f), "method": "animate", "args": [[str(f)], {"mode": "immediate"}]} for f in frames]}],
    }
    html = "<!doctype html><html><head><meta charset='utf-8'><script src='https://cdn.plot.ly/plotly-2.35.2.min.js'></script></head><body><div id='p' style='height:95vh'></div><script>"
    html += f"Plotly.newPlot('p',{json.dumps([trace])},{{scene:{{aspectmode:'data'}},title:{json.dumps(title)},...{json.dumps(controls)}}}).then(()=>Plotly.addFrames('p',{json.dumps(frame_payload)}));"
    html += "</script></body></html>"
    path.write_text(html, encoding="utf-8")


def _write_vector_slider(path: Path, xyz: torch.Tensor, frames: list[int], displacement: torch.Tensor, label: torch.Tensor, title: str, max_points: int):
    count = min(max_points, 600, xyz.shape[0])
    ids = torch.linspace(0, xyz.shape[0] - 1, count).round().long().unique()
    pts = xyz[ids].float()

    def frame_lines(frame_index: int):
        disp = displacement[frame_index, ids].float()
        lab = label[frame_index, ids].bool()
        traces = []
        for value, name, color in ((False, "static", "#3267d6"), (True, "dynamic", "#d33f2f")):
            line = {"type": "scatter3d", "mode": "lines", "name": name, "x": [], "y": [], "z": [], "line": {"color": color, "width": 3}}
            for j in torch.nonzero(lab == value, as_tuple=False).flatten():
                p = pts[j]
                q = p + disp[j]
                line["x"] += [float(p[0]), float(q[0]), None]
                line["y"] += [float(p[1]), float(q[1]), None]
                line["z"] += [float(p[2]), float(q[2]), None]
            traces.append(line)
        return traces

    first = frame_lines(0)
    frame_payload = [{"name": str(frame), "data": frame_lines(idx)} for idx, frame in enumerate(frames)]
    controls = {
        "updatemenus": [{"type": "buttons", "buttons": [{"label": "play", "method": "animate", "args": [None]}]}],
        "sliders": [{"steps": [{"label": str(f), "method": "animate", "args": [[str(f)], {"mode": "immediate"}]} for f in frames]}],
    }
    html = "<!doctype html><html><head><meta charset='utf-8'><script src='https://cdn.plot.ly/plotly-2.35.2.min.js'></script></head><body><div id='p' style='height:95vh'></div><script>"
    html += f"Plotly.newPlot('p',{json.dumps(first)},{{scene:{{aspectmode:'data'}},title:{json.dumps(title)},...{json.dumps(controls)}}}).then(()=>Plotly.addFrames('p',{json.dumps(frame_payload)}));"
    html += "</script></body></html>"
    path.write_text(html, encoding="utf-8")


def _write_score_html(path: Path, xyz: torch.Tensor, score: torch.Tensor, title: str, max_points: int, cmax=3.0):
    ids = torch.linspace(0, xyz.shape[0] - 1, min(max_points, xyz.shape[0])).round().long().unique()
    pts = xyz[ids].float()
    trace = {
        "type": "scatter3d",
        "mode": "markers",
        "x": pts[:, 0].tolist(),
        "y": pts[:, 1].tolist(),
        "z": pts[:, 2].tolist(),
        "marker": {"size": 3, "color": score[ids].float().clamp(0, cmax).tolist(), "colorscale": "Turbo", "cmin": 0, "cmax": cmax, "opacity": 0.8},
        "text": [f"idx={int(i)}<br>score={float(score[i]):.5f}" for i in ids],
        "hoverinfo": "text",
    }
    _html(path, [trace], title, xyz)


def _aggregate_temporal(eval_root: Path, checkpoint_target: int, targets: list[int], args):
    files = [eval_root / f"frame_{t:06d}" / "per_gaussian_motion.pt" for t in targets]
    data = [torch.load(path, map_location="cpu") for path in files]
    null = torch.load(eval_root / "null_frame_000000" / "per_gaussian_motion.pt", map_location="cpu")
    target_frames = torch.tensor(targets, dtype=torch.long)
    temporal_dir = eval_root / "temporal"
    temporal_dir.mkdir(parents=True, exist_ok=True)
    raw_m = torch.stack([d["motion_ratio_raw"] for d in data])
    corrected_m = torch.stack([d["motion_ratio_corrected"] for d in data])
    raw_scores = _scores(raw_m, target_frames)
    corrected_scores = _scores(corrected_m, target_frames)
    payload = {
        "schema_version": 3,
        "checkpoint_path": data[0]["checkpoint_path"],
        "checkpoint_sha256": data[0]["checkpoint_sha256"],
        "checkpoint_train_target": int(checkpoint_target),
        "source_frame": 0,
        "target_frames": target_frames,
        "timestamps": torch.tensor([d["timestamp"] for d in data], dtype=torch.float32),
        "delta_t": torch.tensor([d["delta_t"] for d in data], dtype=torch.float32),
        "trial_seeds": data[0]["trial_seeds"],
        "canonical_xyz": data[0]["canonical_xyz"],
        "sigma_max": float(data[0]["sigma_max"]),
        "mean_displacement_raw": torch.stack([d["mean_displacement_raw"] for d in data]),
        "displacement_norm_raw": torch.stack([d["displacement_norm_raw"] for d in data]),
        "motion_ratio_raw": raw_m,
        "dynamic_label_raw": torch.stack([d["dynamic_label_raw"] for d in data]),
        "seed_consistency_raw": torch.stack([d["seed_consistency_raw"] for d in data]),
        "mean_displacement_corrected": torch.stack([d["mean_displacement_corrected"] for d in data]),
        "displacement_norm_corrected": torch.stack([d["displacement_norm_corrected"] for d in data]),
        "motion_ratio_corrected": corrected_m,
        "dynamic_label_corrected": torch.stack([d["dynamic_label_corrected"] for d in data]),
        "seed_consistency_corrected": torch.stack([d["seed_consistency_corrected"] for d in data]),
        "visibility_ratio": torch.stack([d["visibility_ratio"] for d in data]),
        "valid_fraction": torch.stack([d["valid_fraction"] for d in data]),
        "observed": torch.stack([d["observed"] for d in data]),
        "null_mean_displacement": null["mean_displacement_raw"],
        "null_motion_ratio": null["motion_ratio_raw"],
        "null_seed_consistency": null["seed_consistency_raw"],
        "gt_dynamic": data[0].get("gt_dynamic"),
        "gt_distance": data[0].get("gt_distance"),
    }
    torch.save(payload, temporal_dir / "frame0_temporal_motion.pt")
    summary = {
        "checkpoint_train_target": int(checkpoint_target),
        "target_frames": targets,
        "raw": {k: _summary_tensor(v) for k, v in raw_scores.items()},
        "corrected": {k: _summary_tensor(v) for k, v in corrected_scores.items()},
    }
    (temporal_dir / "temporal_summary.json").write_text(json.dumps(_json_ready(summary), indent=2), encoding="utf-8")
    if args.debug_html:
        xyz = payload["canonical_xyz"]
        _write_slider(temporal_dir / "raw_motion_ratio_slider.html", xyz, targets, raw_m, "raw motion ratio slider, M>1 dynamic", args.max_html_points)
        _write_slider(temporal_dir / "corrected_motion_ratio_slider.html", xyz, targets, corrected_m, "corrected motion ratio slider, M>1 dynamic", args.max_html_points)
        _write_slider(temporal_dir / "visibility_slider.html", xyz, targets, payload["visibility_ratio"], "visibility slider", args.max_html_points, cmax=1.0)
        _write_slider(temporal_dir / "raw_static_dynamic_slider.html", xyz, targets, payload["dynamic_label_raw"].float(), "raw static/dynamic slider", args.max_html_points, cmax=1.0)
        _write_slider(temporal_dir / "corrected_static_dynamic_slider.html", xyz, targets, payload["dynamic_label_corrected"].float(), "corrected static/dynamic slider", args.max_html_points, cmax=1.0)
        _write_score_html(temporal_dir / "dynamic_confirmation_count.html", xyz, corrected_scores["dynamic_confirmation_count"].float(), "corrected dynamic confirmation count", args.max_html_points, cmax=max(1.0, float(len(targets))))
        _write_score_html(temporal_dir / "first_dynamic_frame.html", xyz, corrected_scores["first_dynamic_frame"].float().clamp_min(0), "first corrected dynamic frame (-1 shown as 0)", args.max_html_points, cmax=max(targets))
        _write_vector_slider(temporal_dir / "corrected_displacement_vectors_slider.html", xyz, targets, payload["mean_displacement_corrected"], payload["dynamic_label_corrected"], "corrected displacement vectors slider", args.max_html_points)
        if payload["gt_dynamic"] is not None:
            gt = payload["gt_dynamic"].bool()
            errors = []
            for pred in payload["dynamic_label_corrected"]:
                # 0 TN, 1 TP, 2 FP, 3 FN
                errors.append((pred & gt).long() + 2 * (pred & ~gt).long() + 3 * ((~pred) & gt).long())
            _write_slider(temporal_dir / "temporal_gt_error_slider.html", xyz, targets, torch.stack(errors).float(), "GT error classes: 0 TN, 1 TP, 2 FP, 3 FN", args.max_html_points, cmax=3.0)
    return temporal_dir / "frame0_temporal_motion.pt", summary


def _summary_tensor(tensor):
    t = torch.as_tensor(tensor).float()
    if t.numel() == 0:
        return {}
    return {"p10": float(torch.quantile(t, 0.10)), "p50": float(torch.quantile(t, 0.50)), "p90": float(torch.quantile(t, 0.90)), "max": float(t.max())}


def _metric_rows(output_root: Path, checkpoint_targets: list[int], eval_targets: list[int]):
    rows = []
    for ckpt_target in checkpoint_targets:
        ckpt_eval = output_root / "evaluation" / f"checkpoint_target_{ckpt_target:06d}"
        for target in eval_targets:
            summary_path = ckpt_eval / f"frame_{target:06d}" / "motion_summary.json"
            if not summary_path.is_file():
                continue
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            render = summary.get("render", {})
            for mode in ("raw", "corrected"):
                metrics = ((summary.get(mode) or {}).get("gt_metrics") or {}).get("reliable_observed") or {}
                rows.append(
                    {
                        "checkpoint_train_target": ckpt_target,
                        "eval_target": target,
                        "mode": mode,
                        "precision": metrics.get("precision"),
                        "recall": metrics.get("recall"),
                        "f1": metrics.get("f1"),
                        "iou": metrics.get("iou"),
                        "accuracy": metrics.get("accuracy"),
                        "roc_auc": metrics.get("roc_auc"),
                        "pr_auc": metrics.get("pr_auc"),
                        "tp": metrics.get("tp"),
                        "fp": metrics.get("fp"),
                        "fn": metrics.get("fn"),
                        "tn": metrics.get("tn"),
                        "static_render_loss": (render.get("canonical") or {}).get("total"),
                        "raw_moved_render_loss": (render.get("raw_moved") or {}).get("total"),
                        "corrected_moved_render_loss": (render.get("corrected_moved") or {}).get("total"),
                    }
                )
    return rows


def _write_scatter_png(path: Path, x: torch.Tensor, y: torch.Tensor, title: str):
    x = x.float().numpy()
    y = y.float().numpy()
    w = h = 720
    image = np.full((h, w, 3), 255, np.uint8)
    finite = np.isfinite(x) & np.isfinite(y)
    if finite.any():
        m = max(float(np.percentile(x[finite], 99)), float(np.percentile(y[finite], 99)), 1.0)
        px = np.clip(60 + x[finite] / m * (w - 100), 0, w - 1).astype(np.int32)
        py = np.clip(h - 60 - y[finite] / m * (h - 100), 0, h - 1).astype(np.int32)
        for a, b in zip(px, py):
            cv2.circle(image, (int(a), int(b)), 1, (40, 90, 220), -1)
        cv2.line(image, (60, h - 60), (w - 40, 60), (200, 200, 200), 1)
    cv2.putText(image, title[:95], (25, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (20, 20, 20), 1, cv2.LINE_AA)
    cv2.imwrite(str(path), image)


def _compare(output_root: Path, checkpoint_targets: list[int], eval_targets: list[int], args):
    comparison = output_root / "comparison"
    comparison.mkdir(parents=True, exist_ok=True)
    rows = _metric_rows(output_root, checkpoint_targets, eval_targets)
    with (comparison / "per_target_metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        columns = ["checkpoint_train_target", "eval_target", "mode", "precision", "recall", "f1", "iou", "accuracy", "roc_auc", "pr_auc", "tp", "fp", "fn", "tn", "static_render_loss", "raw_moved_render_loss", "corrected_moved_render_loss"]
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
    (comparison / "per_target_metrics.json").write_text(json.dumps(_json_ready(rows), indent=2), encoding="utf-8")

    if len(checkpoint_targets) >= 2:
        a, b = checkpoint_targets[:2]
        null_a = torch.load(output_root / "evaluation" / f"checkpoint_target_{a:06d}" / "null_frame_000000" / "per_gaussian_motion.pt", map_location="cpu")
        null_b = torch.load(output_root / "evaluation" / f"checkpoint_target_{b:06d}" / "null_frame_000000" / "per_gaussian_motion.pt", map_location="cpu")
        va = null_a["mean_displacement_raw"]
        vb = null_b["mean_displacement_raw"]
        cos = (va * vb).sum(-1) / (torch.linalg.norm(va, dim=-1) * torch.linalg.norm(vb, dim=-1)).clamp_min(1.0e-8)
        ratio = torch.linalg.norm(vb, dim=-1) / torch.linalg.norm(va, dim=-1).clamp_min(1.0e-8)
        payload = {"a": a, "b": b, "cosine": _summary_tensor(cos), "magnitude_ratio": _summary_tensor(ratio)}
        (comparison / "null_bias_comparison.json").write_text(json.dumps(_json_ready(payload), indent=2), encoding="utf-8")
        (comparison / f"null_bias_D{a:02d}_vs_D{b:03d}.json").write_text(json.dumps(_json_ready(payload), indent=2), encoding="utf-8")
        _write_score_html(comparison / "null_bias_direction_cosine.html", null_a["canonical_xyz"], cos.clamp(-1, 1) + 1.0, "null bias direction cosine (+1 shift for color)", args.max_html_points, cmax=2.0)
        _write_score_html(comparison / "null_bias_magnitude_ratio.html", null_a["canonical_xyz"], ratio, "null bias magnitude ratio", args.max_html_points)
        for target in eval_targets:
            pa = torch.load(output_root / "evaluation" / f"checkpoint_target_{a:06d}" / f"frame_{target:06d}" / "per_gaussian_motion.pt", map_location="cpu")
            pb = torch.load(output_root / "evaluation" / f"checkpoint_target_{b:06d}" / f"frame_{target:06d}" / "per_gaussian_motion.pt", map_location="cpu")
            _write_scatter_png(comparison / f"checkpoint_score_scatter_frame_{target:06d}_raw.png", pa["motion_ratio_raw"], pb["motion_ratio_raw"], f"raw motion ratio D{a} vs D{b}, frame {target}")
            _write_scatter_png(comparison / f"checkpoint_score_scatter_frame_{target:06d}_corrected.png", pa["motion_ratio_corrected"], pb["motion_ratio_corrected"], f"corrected motion ratio D{a} vs D{b}, frame {target}")
            _write_scatter_png(comparison / f"checkpoint_score_scatter_frame_{target:06d}.png", pa["motion_ratio_corrected"], pb["motion_ratio_corrected"], f"corrected motion ratio D{a} vs D{b}, frame {target}")

    lines = ["# Frame-0 checkpoint comparison", "", f"Rows: {len(rows)}", "", "See `per_target_metrics.csv` and generated scatter/null-bias debug files."]
    (comparison / "checkpoint_comparison_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return rows


def _write_root_report(output_root: Path, config: dict, checkpoints: dict, temporal: dict, rows: list[dict]):
    summary = {
        "config": config,
        "checkpoints": checkpoints,
        "temporal": temporal,
        "per_target_metrics": rows,
        "comparison_dir": str(output_root / "comparison"),
    }
    (output_root / "pipeline_summary.json").write_text(json.dumps(_json_ready(summary), indent=2), encoding="utf-8")
    lines = [
        "# Frame-0 multitarget null-corrected motion pipeline",
        "",
        "## Configuration",
        "",
        "```json",
        json.dumps(_json_ready(config), indent=2),
        "```",
        "",
        "## Checkpoints",
        "",
    ]
    for target, item in checkpoints.items():
        lines += [f"- D0{target}: `{item['path']}`", f"  - sha256: `{item['sha256']}`"]
    lines += ["", "## Per-target reliable-observed GT metrics", ""]
    for row in rows:
        lines.append(
            f"- D0{row['checkpoint_train_target']} -> frame {row['eval_target']} {row['mode']}: "
            f"F1={row['f1']} precision={row['precision']} recall={row['recall']} "
            f"TP/FP/FN/TN={row['tp']}/{row['fp']}/{row['fn']}/{row['tn']}"
        )
    lines += ["", "## Debug roots", "", f"- evaluation: `{output_root / 'evaluation'}`", f"- comparison: `{output_root / 'comparison'}`"]
    (output_root / "pipeline_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(args):
    output_root = Path(args.output_root).expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    checkpoints_arg = _parse_checkpoints(args.checkpoint)
    train_targets = _parse_ints(args.train_targets)
    eval_targets = _eval_targets(args)
    config = vars(args).copy()
    config["eval_targets_resolved"] = eval_targets
    (output_root / "config.json").write_text(json.dumps(_json_ready(config), indent=2), encoding="utf-8")

    checkpoints = {}
    for target in train_targets:
        checkpoint = _train_or_reuse(args, output_root, target, checkpoints_arg)
        checkpoints[target] = {"path": str(checkpoint), "sha256": sha256_file(checkpoint)}

    if not args.evaluate:
        _write_root_report(output_root, config, checkpoints, {}, [])
        return

    temporal = {}
    for checkpoint_target in train_targets:
        checkpoint = Path(checkpoints[checkpoint_target]["path"])
        eval_root = output_root / "evaluation" / f"checkpoint_target_{checkpoint_target:06d}"
        null_dir = eval_root / "null_frame_000000"
        null_file = _evaluate(args, checkpoint, checkpoint_target, 0, null_dir, None)
        for target in eval_targets:
            _evaluate(args, checkpoint, checkpoint_target, target, eval_root / f"frame_{target:06d}", null_file)
        temporal_path, temporal_summary = _aggregate_temporal(eval_root, checkpoint_target, eval_targets, args)
        temporal[checkpoint_target] = {"path": str(temporal_path), "summary": temporal_summary}

    rows = _compare(output_root, train_targets, eval_targets, args)
    _write_root_report(output_root, config, checkpoints, temporal, rows)
    print("[COMPLETE]", output_root)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root-scene", required=True)
    p.add_argument("--static-model-dir", required=True)
    p.add_argument("--iteration", type=int, default=30000)
    p.add_argument("--output-root", required=True)
    p.add_argument("--train-targets", default="1,59")
    p.add_argument("--eval-targets", default="1,30,59")
    p.add_argument("--num-eval-targets", type=int, default=0)
    p.add_argument("--inference-trials", type=int, default=16)
    p.add_argument("--knn", type=int, default=16)
    p.add_argument("--use-depth", action=BooleanOptionalAction, default=True)
    p.add_argument("--seed", type=int, default=20260901)
    p.add_argument("--skip-existing", action="store_true")
    p.add_argument("--force", action="store_true")
    p.add_argument("--train", action=BooleanOptionalAction, default=True)
    p.add_argument("--evaluate", action=BooleanOptionalAction, default=True)
    p.add_argument("--debug-2d", action=BooleanOptionalAction, default=True)
    p.add_argument("--debug-html", action=BooleanOptionalAction, default=True)
    p.add_argument("--gt-ply", default="")
    p.add_argument("--max-gt-distance", type=float, default=0.02)
    p.add_argument("--checkpoint", action="append", default=[])
    p.add_argument("--training-iterations", type=int, default=1000)
    p.add_argument("--train-cameras-per-iter", type=int, default=0)
    p.add_argument(
        "--cameras",
        default="",
        help="Optional comma-separated dynamic camera IDs forwarded to denoiser train/eval; empty=all.",
    )
    p.add_argument(
        "--camera-data-device",
        default="cpu",
        help="Device for cached camera RGB/depth tensors in denoiser train/eval.",
    )
    p.add_argument("--max-html-points", type=int, default=5000)
    args = p.parse_args()
    args.inference_trials = max(1, int(args.inference_trials))
    args.knn = max(1, int(args.knn))
    args.training_iterations = max(1, int(args.training_iterations))
    args.train_cameras_per_iter = max(0, int(args.train_cameras_per_iter))
    args.max_html_points = max(10, int(args.max_html_points))
    return args


if __name__ == "__main__":
    run(parse_args())
