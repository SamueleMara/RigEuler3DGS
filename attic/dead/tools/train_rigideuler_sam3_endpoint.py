#!/usr/bin/env python3
"""Endpoint-aware RigEuler training wrapper.

This is a thin launcher over train.py. The implementation remains in the normal
training path so Eulerian/Lagrangian behavior stays shared and testable.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Launch endpoint-aware RigEuler K-part training")
    parser.add_argument("-s", "--source", required=True)
    parser.add_argument("-m", "--model-path", required=True)
    parser.add_argument("--start-ply", required=True)
    parser.add_argument("--support", choices=("eulerian", "lagrangian"), default="eulerian")
    parser.add_argument("--endpoint-frame", default="-1")
    parser.add_argument("--endpoint-model-path", default="")
    parser.add_argument("--endpoint-static-iters", type=int, default=1000)
    parser.add_argument("--endpoint-init", choices=("depth_fusion", "colmap", "canonical"), default="depth_fusion")
    parser.add_argument("--endpoint-build-if-missing", action="store_true")
    parser.add_argument("--sam3-feature-root", default="")
    parser.add_argument("--sam3-weight", type=float, default=0.0)
    parser.add_argument("--sam3-graph-weight", type=float, default=0.0)
    parser.add_argument("--endpoint-unary-weight", type=float, default=0.25)
    parser.add_argument("--endpoint-velocity-weight", type=float, default=0.0)
    parser.add_argument("--cycles", type=int, default=8)
    parser.add_argument("--plateau-cycles", type=int, default=4)
    parser.add_argument("--num-moving-parts", type=int, default=1)
    parser.add_argument("--wandb-name", default="")
    parser.add_argument("--no-wandb", action="store_true")
    args, passthrough = parser.parse_known_args()

    representation = "rigid_killing_lagrangian" if args.support == "lagrangian" else "rigid_killing_eulerian"
    endpoint_frame = "-1" if str(args.endpoint_frame).strip().upper() in {"LAST", "END", "FINAL"} else str(args.endpoint_frame)
    cmd = [
        sys.executable,
        str(Path(__file__).resolve().parents[1] / "train.py"),
        "-s",
        args.source,
        "-m",
        args.model_path,
        "--start_ply",
        args.start_ply,
        "--dynamic_motion",
        "--phase1_motion_representation",
        representation,
        "--kpart_rigid_discovery",
        "--num_moving_parts",
        str(args.num_moving_parts),
        "--dynamic_use_depth",
        "--use_semantic_feature_field",
        "--dynamic_discovery_mode",
        "render_denoise",
        "--use_endpoint_static_model",
        "--endpoint_frame",
        endpoint_frame,
        "--endpoint_static_iters",
        str(args.endpoint_static_iters),
        "--endpoint_init",
        args.endpoint_init,
        "--kpart_endpoint_transport",
        "--endpoint_transport_mode",
        "local_soft",
        "--endpoint_unary_weight",
        str(args.endpoint_unary_weight),
        "--endpoint_velocity_weight",
        str(args.endpoint_velocity_weight),
        "--endpoint_sam3_weight",
        str(args.sam3_weight),
        "--kpart_anneal_cycles",
        str(args.cycles),
        "--kpart_plateau_cycles",
        str(args.plateau_cycles),
        "--disable_viewer",
    ]
    if args.endpoint_model_path:
        cmd += ["--endpoint_model_path", args.endpoint_model_path]
    if args.endpoint_build_if_missing:
        cmd += ["--endpoint_build_if_missing"]
    if args.sam3_feature_root:
        cmd += [
            "--use_sam3_feature_field",
            "--sam3_feature_root",
            args.sam3_feature_root,
            "--kpart_sam3",
            "--kpart_sam3_3d_weight",
            str(args.sam3_graph_weight),
            "--no_kpart_dino",
        ]
    if args.no_wandb:
        cmd += ["--no_wandb"]
    elif args.wandb_name:
        cmd += ["--wandb", "--wandb_name", args.wandb_name]
    cmd += passthrough
    print(" ".join(cmd))
    subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
