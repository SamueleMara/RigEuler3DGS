#!/usr/bin/env python3
"""Standalone final prismatic/revolute/screw estimation from gaussian_tracks.pt."""
from pathlib import Path
import argparse
import json
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils.final_joint_estimation import (
    JointEstimationConfig,
    estimate_and_write_final_joint_from_file,
)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--tracks', required=True, help='Path to gaussian_tracks.pt')
    p.add_argument('--output_dir', default='', help='Default: directory containing --tracks')
    p.add_argument('--dynamic_threshold', type=float, default=0.80)
    p.add_argument('--min_moving_points', type=int, default=64)
    p.add_argument('--max_fit_points', type=int, default=4096)
    p.add_argument('--max_html_points', type=int, default=20000)
    p.add_argument('--axis_length_scale', type=float, default=1.25)
    p.add_argument('--omega_activity_threshold', type=float, default=1e-3)
    p.add_argument('--prismatic_rotation_ratio', type=float, default=0.10)
    p.add_argument('--screw_min_pitch', type=float, default=1e-3)
    p.add_argument('--screw_min_axial_ratio', type=float, default=0.03)
    p.add_argument('--screw_min_relative_improvement', type=float, default=0.05)
    args = p.parse_args()

    tracks = Path(args.tracks)
    out = Path(args.output_dir) if args.output_dir else tracks.parent
    cfg = JointEstimationConfig(
        dynamic_threshold=args.dynamic_threshold,
        min_moving_points=args.min_moving_points,
        max_fit_points=args.max_fit_points,
        max_html_points=args.max_html_points,
        axis_length_scale=args.axis_length_scale,
        omega_activity_threshold=args.omega_activity_threshold,
        prismatic_rotation_ratio=args.prismatic_rotation_ratio,
        screw_min_pitch=args.screw_min_pitch,
        screw_min_axial_ratio=args.screw_min_axial_ratio,
        screw_min_relative_improvement=args.screw_min_relative_improvement,
    )
    result = estimate_and_write_final_joint_from_file(tracks, out, cfg)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
