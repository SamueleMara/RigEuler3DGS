#!/usr/bin/env python3
"""Run dynamic pipelines for every scene in input/.

Default mode runs the exact closed-loop SE(3) rigid-part Stage-1 from existing
frame-0 / frame-T static endpoint models.
"""

from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
from pathlib import Path

from tqdm import tqdm


REPO = Path(__file__).resolve().parents[1]
FRAME_RE = re.compile(r"frame_(\d+)\.")


def _scene_last_frame(scene: Path) -> int | None:
    frames: list[int] = []
    for path in (scene / "rgb").rglob("frame_*.*"):
        match = FRAME_RE.search(path.name)
        if match:
            frames.append(int(match.group(1)))
    return max(frames) if frames else None


def _iter_scenes(input_root: Path, only: set[str] | None) -> list[Path]:
    scenes = []
    for scene in sorted(input_root.iterdir()):
        if not scene.is_dir():
            continue
        if only is not None and scene.name not in only:
            continue
        if not (scene / "colmap").is_dir() or not (scene / "rgb").is_dir():
            continue
        scenes.append(scene)
    return scenes


def _model_ply(model_dir: Path, iteration: int) -> Path:
    return model_dir / "point_cloud" / f"iteration_{int(iteration)}" / "point_cloud.ply"


def _default_prior_path(args, scene: Path) -> Path:
    template = str(args.dynamic_prior_template)
    return Path(template.format(scene=scene.name, output_root=str(Path(args.output_root).expanduser().resolve()))).expanduser().resolve()


def _comparison_command(args, scene: Path, last_frame: int, output: Path) -> list[str]:
    mid_frame = last_frame // 2
    cmd = [
        sys.executable,
        str(REPO / "tools" / "run_dynamic_prior_comparison.py"),
        "--root-scene",
        str(scene),
        "--static-model-dir",
        str(output / "shared" / "frame0_static"),
        "--static-iteration",
        str(args.static_iteration),
        "--output",
        str(output),
        "--methods",
        args.methods,
        "--denoise-train-targets",
        f"1,{last_frame}",
        "--denoise-eval-targets",
        f"1,{mid_frame},{last_frame}",
        "--denoise-camera-data-device",
        args.denoise_camera_data_device,
        "--endpoint-frame",
        str(last_frame),
        "--endpoint-iteration",
        str(args.static_iteration),
        "--fm-iters",
        str(args.fm_iters),
        "--fm-endpoint-knn",
        str(args.fm_endpoint_knn),
        "--fm-transport-temperature",
        str(args.fm_transport_temperature),
        "--fm-transport-robust-delta",
        str(args.fm_transport_robust_delta),
        "--fm-motion-scale",
        args.fm_motion_scale,
        "--fm-confidence-entropy-scale",
        str(args.fm_confidence_entropy_scale),
        "--fm-prior-prob-bias",
        str(args.fm_prior_prob_bias),
        "--fm-prior-prob-temperature",
        str(args.fm_prior_prob_temperature),
        "--phase1-motion-representation",
        args.phase1_motion_representation,
        "--run-killing",
        "--run-kpart",
        "--kpart-k",
        str(args.kpart_k),
        "--kpart-anneal-cycles",
        str(args.kpart_anneal_cycles),
        "--kpart-plateau-cycles",
        str(args.kpart_plateau_cycles),
        "--kpart-dynamic-prior-weight",
        str(args.kpart_dynamic_prior_weight),
        "--kpart-prior-boundary-expansion",
        str(args.kpart_prior_boundary_expansion),
        "--kpart-prior-anchor-threshold",
        str(args.kpart_prior_anchor_threshold),
        "--kpart-prior-cleanup-min-size",
        str(args.kpart_prior_cleanup_min_size),
        "--prior-smooth-iters",
        str(args.prior_smooth_iters),
        "--prior-smooth-knn",
        str(args.prior_smooth_knn),
    ]
    if args.force_killing:
        cmd.append("--force-killing")
    if args.fit_joints:
        cmd.extend(
            [
                "--fit-joints",
                "--joint-types",
                args.joint_types,
                "--joint-topology",
                args.joint_topology,
                "--joint-compare-iters",
                str(args.joint_compare_iters),
                "--joint-compare-batch-size",
                str(args.joint_compare_batch_size),
                "--joint-compare-validation-cameras",
                str(args.joint_compare_validation_cameras),
            ]
        )
    if args.optimize_joint_trajectory:
        cmd.extend(
            [
                "--optimize-joint-trajectory",
                "--joint-trajectory-mode",
                args.joint_trajectory_mode,
                "--joint-trajectory-iters",
                str(args.joint_trajectory_iters),
                "--joint-trajectory-batch-size",
                str(args.joint_trajectory_batch_size),
                "--joint-trajectory-acceleration-weight",
                str(args.joint_trajectory_acceleration_weight),
                "--joint-trajectory-jerk-weight",
                str(args.joint_trajectory_jerk_weight),
                "--joint-trajectory-reverse-weight",
                str(args.joint_trajectory_reverse_weight),
            ]
        )
    return cmd


def _exact_closed_loop_command(args, scene: Path, last_frame: int, output: Path) -> list[str]:
    endpoint_root = Path(args.output_root).expanduser().resolve() / f"{scene.name}_{args.static_endpoints_tag}" / "shared"
    frame0_dir = endpoint_root / "frame0_static"
    frame_t_dir = endpoint_root / "frame_t_static"
    start_ply = _model_ply(frame0_dir, int(args.static_iteration))
    if not start_ply.is_file():
        raise FileNotFoundError(f"missing frame0 static model: {start_ply}")
    if not _model_ply(frame_t_dir, int(args.static_iteration)).is_file():
        raise FileNotFoundError(f"missing frame-T static model: {_model_ply(frame_t_dir, int(args.static_iteration))}")
    cmd = [
        sys.executable,
        str(REPO / "train.py"),
        "-s",
        str(scene),
        "-m",
        str(output),
        "--start_ply",
        str(start_ply),
        "--start_ply_iteration",
        str(args.static_iteration),
        "--iterations",
        str(args.static_iteration),
        "--static_train_iters",
        str(args.static_iteration),
        "--dynamic_motion",
        "--phase1_motion_representation",
        args.phase1_motion_representation,
        "--phase1_only",
        "--kpart_rigid_discovery",
        "--kpart_velocity_field_segmentation",
        "--num_moving_parts",
        str(args.kpart_k),
        "--kpart_dynamic_prior_weight",
        str(args.kpart_dynamic_prior_weight),
        "--kpart_prior_anchor_threshold",
        str(args.kpart_prior_anchor_threshold),
        "--kpart_confidence_threshold",
        str(args.kpart_confidence_threshold),
        "--kpart_prior_boundary_expansion",
        str(args.kpart_prior_boundary_expansion),
        "--kpart_prior_cleanup_min_size",
        str(args.kpart_prior_cleanup_min_size),
        "--kpart_anneal_cycles",
        str(args.kpart_anneal_cycles),
        "--kpart_plateau_cycles",
        str(args.kpart_plateau_cycles),
        "--enable_rigid_part_discovery",
        "--rigid_part_outer_iters",
        str(args.rigid_part_outer_iters),
        "--rigid_part_local_motion_steps",
        str(args.rigid_part_local_motion_steps),
        "--rigid_part_label_steps",
        str(args.rigid_part_label_steps),
        "--rigid_part_render_frames",
        str(args.rigid_part_render_frames),
        "--rigid_part_render_cameras",
        str(args.rigid_part_render_cameras),
        "--rigid_part_max_frames",
        str(args.rigid_part_max_frames),
        "--rigid_part_raw_lr",
        str(args.rigid_part_raw_lr),
        "--lambda_raw_observed_velocity",
        str(args.lambda_raw_observed_velocity),
        "--lambda_raw_killing",
        str(args.lambda_raw_killing),
        "--lambda_raw_temporal",
        str(args.lambda_raw_temporal),
        "--lambda_raw_rigid_feedback",
        str(args.lambda_raw_rigid_feedback),
        "--lambda_rigid_prior",
        str(args.lambda_rigid_prior),
        "--lambda_rigid_moving_complexity",
        str(args.lambda_rigid_moving_complexity),
        "--lambda_rigid_graph_repulsive",
        str(args.lambda_rigid_graph_repulsive),
        "--final_3dgs_path",
        str(frame_t_dir),
        "--use_endpoint_static_model",
        "--endpoint_frame",
        str(last_frame),
        "--endpoint_model_path",
        str(frame_t_dir),
        "--endpoint_static_iters",
        str(args.static_iteration),
        "--kpart_endpoint_transport",
        "--endpoint_transport_mode",
        "local_soft",
        "--dynamic_use_depth",
        "--use_semantic_feature_field",
        "--curl_grid_target_cells",
        str(args.curl_grid_target_cells),
        "--no_wandb",
        "--disable_viewer",
    ]
    prior = _default_prior_path(args, scene)
    if prior.is_file():
        cmd.extend(
            [
                "--dynamic_prior_path",
                str(prior),
                "--prior_use_soft_weights",
                "--prior_hard_threshold",
                str(args.prior_hard_threshold),
            ]
        )
    return cmd


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", default=str(REPO / "input"))
    parser.add_argument("--output-root", default=str(REPO / "output"))
    parser.add_argument("--log-dir", default=str(REPO / "output" / "night_logs"))
    parser.add_argument("--only", default="", help="Comma-separated scene names to run.")
    parser.add_argument("--tag", default="exact_closed_loop_se3_14k")
    parser.add_argument(
        "--pipeline",
        choices=("exact_closed_loop_se3", "legacy_comparison"),
        default="exact_closed_loop_se3",
    )
    parser.add_argument("--static-endpoints-tag", default="static_endpoints_14k")
    parser.add_argument("--skip-missing-static", action="store_true")
    parser.add_argument("--static-iteration", type=int, default=14000)
    parser.add_argument(
        "--dynamic-prior-template",
        default="{output_root}/{scene}_strict_static_dynamic_labelling/dynamic_prior.pt",
        help="Optional formatted path. Available fields: {output_root}, {scene}. Used only when the file exists.",
    )
    parser.add_argument("--methods", default="baseline,denoise,flow_matching")
    parser.add_argument(
        "--denoise-camera-data-device",
        default="cpu",
        help="Device for cached denoise RGB/depth tensors.",
    )
    parser.add_argument(
        "--denoise-cameras",
        default="",
        help="Optional comma-separated camera IDs for denoise discovery only.",
    )
    parser.add_argument(
        "--phase1-motion-representation",
        choices=("rigid_killing_eulerian", "rigid_killing_lagrangian", "rigid_killing_1d"),
        default="rigid_killing_eulerian",
    )
    parser.add_argument("--fm-iters", type=int, default=1000)
    parser.add_argument("--fm-endpoint-knn", type=int, default=16)
    parser.add_argument("--fm-transport-temperature", type=float, default=2e-5)
    parser.add_argument("--fm-transport-robust-delta", type=float, default=0.02)
    parser.add_argument("--fm-motion-scale", default="local_knn")
    parser.add_argument("--fm-confidence-entropy-scale", type=float, default=1.0)
    parser.add_argument("--fm-prior-prob-bias", type=float, default=1.3)
    parser.add_argument("--fm-prior-prob-temperature", type=float, default=0.35)
    parser.add_argument("--kpart-k", type=int, default=1)
    parser.add_argument("--kpart-anneal-cycles", type=int, default=8)
    parser.add_argument("--kpart-plateau-cycles", type=int, default=5)
    parser.add_argument("--kpart-dynamic-prior-weight", type=float, default=1.0)
    parser.add_argument("--kpart-prior-boundary-expansion", type=int, default=0)
    parser.add_argument("--kpart-prior-anchor-threshold", type=float, default=0.75)
    parser.add_argument("--kpart-confidence-threshold", type=float, default=0.20)
    parser.add_argument("--kpart-prior-cleanup-min-size", type=int, default=16)
    parser.add_argument("--prior-hard-threshold", type=float, default=0.75)
    parser.add_argument("--prior-smooth-iters", type=int, default=3)
    parser.add_argument("--prior-smooth-knn", type=int, default=16)
    parser.add_argument("--curl-grid-target-cells", type=int, default=4096)
    parser.add_argument("--rigid-part-outer-iters", type=int, default=4)
    parser.add_argument("--rigid-part-local-motion-steps", type=int, default=4)
    parser.add_argument("--rigid-part-label-steps", type=int, default=8)
    parser.add_argument("--rigid-part-render-frames", type=int, default=4)
    parser.add_argument("--rigid-part-render-cameras", type=int, default=2)
    parser.add_argument("--rigid-part-max-frames", type=int, default=12)
    parser.add_argument("--rigid-part-raw-lr", type=float, default=5.0e-3)
    parser.add_argument("--lambda-raw-observed-velocity", type=float, default=1.0)
    parser.add_argument("--lambda-raw-killing", type=float, default=0.1)
    parser.add_argument("--lambda-raw-temporal", type=float, default=0.01)
    parser.add_argument("--lambda-raw-rigid-feedback", type=float, default=0.1)
    parser.add_argument("--lambda-rigid-prior", type=float, default=0.1)
    parser.add_argument("--lambda-rigid-moving-complexity", type=float, default=0.01)
    parser.add_argument("--lambda-rigid-graph-repulsive", type=float, default=0.5)
    parser.add_argument("--force-killing", action="store_true")
    parser.add_argument("--fit-joints", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--joint-types", default="auto")
    parser.add_argument("--joint-topology", default="known")
    parser.add_argument("--joint-compare-iters", type=int, default=40)
    parser.add_argument("--joint-compare-batch-size", type=int, default=8)
    parser.add_argument("--joint-compare-validation-cameras", type=int, default=2)
    parser.add_argument("--optimize-joint-trajectory", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--joint-trajectory-mode", choices=("monotonic", "profile_scale", "per_frame", "spline"), default="monotonic")
    parser.add_argument("--joint-trajectory-iters", type=int, default=300)
    parser.add_argument("--joint-trajectory-batch-size", type=int, default=8)
    parser.add_argument("--joint-trajectory-acceleration-weight", type=float, default=1.0e-5)
    parser.add_argument("--joint-trajectory-jerk-weight", type=float, default=1.0e-8)
    parser.add_argument("--joint-trajectory-reverse-weight", type=float, default=1.0e-2)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    input_root = Path(args.input_root).expanduser().resolve()
    output_root = Path(args.output_root).expanduser().resolve()
    log_dir = Path(args.log_dir).expanduser().resolve()
    log_dir.mkdir(parents=True, exist_ok=True)
    only = {x.strip() for x in args.only.replace(",", " ").split() if x.strip()} or None
    args.joint_trajectory_acceleration_weight = max(0.0, float(args.joint_trajectory_acceleration_weight))
    args.joint_trajectory_jerk_weight = max(0.0, float(args.joint_trajectory_jerk_weight))
    args.joint_trajectory_reverse_weight = max(0.0, float(args.joint_trajectory_reverse_weight))

    env = os.environ.copy()
    env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    env.setdefault("HF_HUB_OFFLINE", "1")

    failures: list[str] = []
    scenes = _iter_scenes(input_root, only)
    for scene in tqdm(scenes, desc="datasets", unit="scene"):
        tqdm.write(f"QUEUE {scene.name}")
        last_frame = _scene_last_frame(scene)
        if last_frame is None:
            tqdm.write(f"{scene.name} FAIL no_frames")
            failures.append(scene.name)
            continue
        output = output_root / f"{scene.name}_{args.tag}"
        log = log_dir / f"{scene.name}_{args.tag}.log"
        try:
            if args.pipeline == "legacy_comparison":
                cmd = _comparison_command(args, scene, last_frame, output)
                if args.denoise_cameras.strip():
                    cmd.extend(["--denoise-cameras", args.denoise_cameras.strip()])
            else:
                cmd = _exact_closed_loop_command(args, scene, last_frame, output)
        except Exception as exc:
            if args.skip_missing_static and "missing frame" in str(exc):
                tqdm.write(f"{scene.name} SKIP command: {exc}")
            else:
                tqdm.write(f"{scene.name} FAIL command: {exc}")
                failures.append(scene.name)
            continue
        tqdm.write(f"START {scene.name} endpoint={last_frame} output={output}")
        if args.dry_run:
            tqdm.write(" ".join(cmd))
            continue
        with log.open("w", encoding="utf-8") as stream:
            result = subprocess.run(cmd, cwd=REPO, env=env, stdout=stream, stderr=subprocess.STDOUT)
        tqdm.write(f"{scene.name} exit={result.returncode} log={log}")
        if result.returncode != 0:
            failures.append(scene.name)
    if failures:
        print("FAILED " + ",".join(failures), flush=True)
        return 1
    print("COMPLETE", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
