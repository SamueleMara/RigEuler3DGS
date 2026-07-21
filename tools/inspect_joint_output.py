#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import torch


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect semantic/joint contamination in gaussian_tracks.pt")
    parser.add_argument("tracks", type=Path)
    args = parser.parse_args()
    tracks = torch.load(args.tracks, map_location="cpu")
    xyz = tracks["canonical_positions"]
    n = int(xyz.shape[0])
    sem = tracks.get("semantic_prob")
    threshold = float(tracks.get("dynamic_foreground_threshold", 0.2))
    if sem is None:
        raise SystemExit("semantic_prob is missing")
    semantic_fg = sem.reshape(-1) >= threshold
    exported_fg = tracks.get("dynamic_foreground_mask")
    joint_ids = tracks.get("joint_part_field_foreground_ids", tracks.get("joint_part_field_gaussian_ids"))
    cluster = tracks.get("joint_cluster_id")
    prob = tracks.get("dynamic_prob")

    print(f"full Gaussians: {n}")
    print(f"semantic threshold: {threshold}")
    print(f"semantic foreground: {int(semantic_fg.sum())}")
    print(f"semantic background: {int((~semantic_fg).sum())}")
    if exported_fg is not None:
        exported_fg = exported_fg.reshape(-1).bool()
        print(f"exported dynamic_foreground_mask true: {int(exported_fg.sum())}")
        print(f"mask disagreement: {int((exported_fg != semantic_fg).sum())}")
    if joint_ids is not None:
        joint_ids = joint_ids.reshape(-1).long()
        background_ids = joint_ids[~semantic_fg[joint_ids]]
        missing_ids = torch.where(semantic_fg)[0][~torch.isin(torch.where(semantic_fg)[0], joint_ids)]
        print(f"joint field IDs: {joint_ids.numel()}")
        print(f"background IDs inside joint field: {background_ids.numel()}")
        print(f"foreground IDs missing from joint field: {missing_ids.numel()}")
    if cluster is not None:
        moving = cluster.reshape(-1).long() == 1
        print(f"hard moving total: {int(moving.sum())}")
        print(f"hard moving semantic foreground: {int((moving & semantic_fg).sum())}")
        print(f"hard moving semantic background: {int((moving & ~semantic_fg).sum())}")
    if prob is not None:
        p = prob.reshape(-1)
        print(f"p>=0.8 total: {int((p>=0.8).sum())}")
        print(f"p>=0.8 semantic foreground: {int(((p>=0.8)&semantic_fg).sum())}")
        print(f"mean p foreground: {float(p[semantic_fg].mean()):.6f}")
        if (~semantic_fg).any():
            print(f"mean p background: {float(p[~semantic_fg].mean()):.6f}")


if __name__ == "__main__":
    main()
