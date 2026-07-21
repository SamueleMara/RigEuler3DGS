#!/usr/bin/env python3
"""CLI wrapper for adaptive motion-grid octomap-style HTML visualization."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from utils.visualize_motion_grid_octomap_html import write_motion_grid_octomap_html_from_file


def main() -> None:
    parser = argparse.ArgumentParser(description="Create octomap-style HTML from motion_grid_state.pt")
    parser.add_argument("--state", required=True, help="Path to motion_grid_state.pt")
    parser.add_argument("--output", required=True, help="Output HTML path")
    parser.add_argument("--title", default="Adaptive Motion Grid")
    parser.add_argument(
        "--color_mode",
        default="feature_pca",
        choices=[
            "feature_pca",
            "feature_norm",
            "dynamic_prob",
            "semantic_prob",
            "mean_speed",
            "max_speed",
            "max_displacement",
            "score_ema",
            "level",
            "count",
        ],
        help="Cell color source. feature_pca displays learned feature field as PCA RGB.",
    )
    parser.add_argument("--max_cells", type=int, default=5000)
    parser.add_argument("--min_count", type=int, default=1)
    parser.add_argument("--opacity", type=float, default=0.24)
    parser.add_argument("--include_inactive", action="store_true", help="Show inactive cells too.")
    args = parser.parse_args()

    summary = write_motion_grid_octomap_html_from_file(
        state_path=args.state,
        out_html=args.output,
        title=args.title,
        color_mode=args.color_mode,
        max_cells=args.max_cells,
        active_only=not args.include_inactive,
        min_count=args.min_count,
        opacity=args.opacity,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
