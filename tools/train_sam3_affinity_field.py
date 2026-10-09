#!/usr/bin/env python3
"""Thin wrapper to train only the cached SAM3 Gaussian affinity field."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Train SAM3 affinity sidecar on an existing static 3DGS PLY")
    parser.add_argument("--root-scene", required=True)
    parser.add_argument("--static-model-dir", required=True)
    parser.add_argument("--output-root", default="")
    parser.add_argument("--iteration", type=int, default=30000)
    parser.add_argument("--sam3-mask-root", default="")
    parser.add_argument("--sam3-affinity-iters", type=int, default=500)
    parser.add_argument("--sam3-affinity-dim", type=int, default=16)
    parser.add_argument("--device", default="cuda", help="Accepted for CLI symmetry; training uses repo CUDA path")
    parser.add_argument("extra", nargs=argparse.REMAINDER)
    args = parser.parse_args()

    repo = Path(__file__).resolve().parents[1]
    output = Path(args.output_root).expanduser().resolve() if args.output_root else Path(args.static_model_dir).expanduser().resolve()
    ply = (
        Path(args.static_model_dir).expanduser().resolve()
        / "point_cloud"
        / f"iteration_{int(args.iteration)}"
        / "point_cloud.ply"
    )
    command = [
        sys.executable,
        str(repo / "train.py"),
        "-s",
        str(Path(args.root_scene).expanduser().resolve()),
        "-m",
        str(output),
        "--iterations",
        str(int(args.iteration)),
        "--start_ply",
        str(ply),
        "--start_ply_iteration",
        str(int(args.iteration)),
        "--static_train_iters",
        str(int(args.iteration)),
        "--use_semantic_feature_field",
        "--use_sam3_affinity_field",
        "--sam3_affinity_iters",
        str(int(args.sam3_affinity_iters)),
        "--sam3_affinity_dim",
        str(int(args.sam3_affinity_dim)),
        "--disable_viewer",
    ]
    if args.sam3_mask_root:
        mask_root = Path(args.sam3_mask_root).expanduser()
        if mask_root.name == "sam3":
            mask_root = mask_root.with_name("sam3_masks")
        command.extend(["--sam3_mask_root", str(mask_root)])
    if args.extra and args.extra[0] == "--":
        command.extend(args.extra[1:])
    elif args.extra:
        command.extend(args.extra)
    subprocess.run(command, cwd=repo, check=True)


if __name__ == "__main__":
    main()
