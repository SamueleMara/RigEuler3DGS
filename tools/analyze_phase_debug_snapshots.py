#!/usr/bin/env python3
"""Compare deep phase diagnostics exported by the debug snapshot patch.

Usage:
  python tools/analyze_phase_debug_snapshots.py \
    --debug_dir output/USB_100109_dynamic_filter_sem_dino_debug/debug_phase_snapshots

It compares joint_early and joint_final snapshots by default. Legacy end_dynamic_labeling/end_rigid_motion tags are still accepted.
loads per_point_diagnostics.pt from both, and writes:
  phase_comparison_summary.json
  phase_comparison_points.csv
"""

import argparse
import csv
import json
from pathlib import Path
import torch


def stats(x):
    if x is None or not torch.is_tensor(x) or x.numel() == 0:
        return {"numel": 0, "finite_fraction": 0.0}
    v = x.detach().float().reshape(-1)
    finite = torch.isfinite(v)
    out = {"numel": int(v.numel()), "finite_fraction": float(finite.float().mean().item()) if v.numel() else 0.0}
    if finite.any():
        vf = v[finite]
        out.update({
            "min": float(vf.min().item()),
            "max": float(vf.max().item()),
            "mean": float(vf.mean().item()),
            "median": float(torch.quantile(vf, 0.50).item()),
            "p25": float(torch.quantile(vf, 0.25).item()),
            "p75": float(torch.quantile(vf, 0.75).item()),
            "p95": float(torch.quantile(vf, 0.95).item()),
            "p99": float(torch.quantile(vf, 0.99).item()),
        })
    return out


def group(name, mask, early, final):
    mask = mask.bool().reshape(-1)
    row = {"name": name, "count": int(mask.sum().item())}
    if not mask.any():
        return row
    for prefix, data in [("early", early), ("final", final)]:
        for key in [
            "dynamic_prob", "joint_part_confidence", "mean_speed", "mean_proposal_speed", "max_proposal_speed",
            "explicit_rigid_mean_speed", "explicit_residual_mean_speed", "max_speed", "max_displacement",
            "nearest_high_motion_seed_distance", "nearest_high_motion_seed_dino_similarity",
            "e_static_mean_speed", "e_rigid_mean_residual", "q_model_identity_vs_rigid",
        ]:
            if key in data and torch.is_tensor(data[key]) and data[key].reshape(-1).numel() == mask.numel():
                row[f"{prefix}_{key}"] = stats(data[key].reshape(-1)[mask])
    return row


def find_snapshot(debug_dir, tag):
    debug_dir = Path(debug_dir)
    hits = sorted([p for p in debug_dir.iterdir() if p.is_dir() and tag in p.name])
    if not hits:
        raise FileNotFoundError(f"No snapshot containing '{tag}' under {debug_dir}")
    return hits[-1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--debug_dir", required=True, help="debug_phase_snapshots directory")
    ap.add_argument("--early_tag", default="joint_early")
    ap.add_argument("--final_tag", default="joint_final")
    ap.add_argument("--out_dir", default="", help="default: debug_dir")
    ap.add_argument("--dynamic_threshold", type=float, default=0.5)
    ap.add_argument("--dead_speed", type=float, default=0.05)
    ap.add_argument("--dead_disp", type=float, default=0.02)
    args = ap.parse_args()

    debug_dir = Path(args.debug_dir)
    early_dir = find_snapshot(debug_dir, args.early_tag)
    final_dir = find_snapshot(debug_dir, args.final_tag)
    out_dir = Path(args.out_dir) if args.out_dir else debug_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    early = torch.load(early_dir / "per_point_diagnostics.pt", map_location="cpu")
    final = torch.load(final_dir / "per_point_diagnostics.pt", map_location="cpu")
    N = int(final["dynamic_prob"].numel())
    if int(early["dynamic_prob"].numel()) != N:
        raise ValueError("Early/final per-point diagnostics have different number of Gaussians")

    thr = float(args.dynamic_threshold)
    fg = final["dynamic_foreground_mask"].bool().reshape(-1)
    early_dyn = early["dynamic_prob"].reshape(-1) >= thr
    final_dyn = final["dynamic_prob"].reshape(-1) >= thr
    early_dead = fg & (early["max_speed"].reshape(-1) < args.dead_speed) & (early["max_displacement"].reshape(-1) < args.dead_disp)
    final_dead = fg & (final["max_speed"].reshape(-1) < args.dead_speed) & (final["max_displacement"].reshape(-1) < args.dead_disp)
    activated = fg & early_dead & final_dyn
    remained_dead_static = fg & early_dead & (~final_dyn) & final_dead
    rescued_motion_but_static = fg & early_dead & (~final_dyn) & (~final_dead)
    became_dead = fg & (~early_dead) & final_dead
    initial_cluster = final.get("joint_initial_cluster_id", final.get("joint_cluster_id", torch.zeros(N, dtype=torch.long))).reshape(-1).long()
    early_cluster = early.get("joint_cluster_id", initial_cluster).reshape(-1).long()
    final_cluster = final.get("joint_cluster_id", initial_cluster).reshape(-1).long()
    changed_early = fg & (early_cluster != initial_cluster)
    changed_final = fg & (final_cluster != initial_cluster)
    changed_between = fg & (final_cluster != early_cluster)

    summary = {
        "early_snapshot": str(early_dir),
        "final_snapshot": str(final_dir),
        "num_gaussians": N,
        "dynamic_threshold": thr,
        "groups": [
            group("foreground", fg, early, final),
            group("early_dead_foreground", early_dead, early, final),
            group("final_dead_foreground", final_dead, early, final),
            group("activated_from_early_dead", activated, early, final),
            group("remained_dead_static", remained_dead_static, early, final),
            group("rescued_motion_but_static", rescued_motion_but_static, early, final),
            group("became_dead", became_dead, early, final),
            group("final_dynamic", fg & final_dyn, early, final),
            group("final_static", fg & (~final_dyn), early, final),
            group("changed_from_initial_early", changed_early, early, final),
            group("changed_from_initial_final", changed_final, early, final),
            group("changed_between_early_final", changed_between, early, final),
            group("initial_static_slot", fg & (initial_cluster == 0), early, final),
            group("initial_moving_slot", fg & (initial_cluster > 0), early, final),
            group("final_static_slot", fg & (final_cluster == 0), early, final),
            group("final_moving_slot", fg & (final_cluster > 0), early, final),
        ],
        "interpretation": {
            "remained_dead_static": "already dead at dynamic-label snapshot and still dead/static after rigid phase; early seeding/activation failure.",
            "rescued_motion_but_static": "motion appeared but p_dyn stayed below threshold; label/threshold/model-target problem.",
            "became_dead": "was not dead early but dead final; joint optimization later suppressed or failed to maintain its motion.",
            "activated_from_early_dead": "joint label+rigid optimization successfully rescued initially dead foreground points.",
            "changed_from_initial_final": "optimized cosine-field assignment differs from the initial full per-Gaussian XYZ+DINO k-medoids clustering.",
            "changed_between_early_final": "cosine-field clustering continued to optimize after the early snapshot.",
        },
    }
    with open(out_dir / "phase_comparison_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    with open(out_dir / "phase_comparison_points.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "idx", "group", "x", "y", "z", "cell_id",
            "initial_cluster", "early_cluster", "final_cluster", "early_part_confidence", "final_part_confidence",
            "early_p", "final_p", "early_mean_proposal_speed", "final_mean_proposal_speed",
            "early_max_speed", "final_max_speed", "early_max_disp", "final_max_disp",
            "early_nearest_seed_dist", "final_nearest_seed_dist", "early_seed_dino", "final_seed_dino",
            "early_q_model", "final_q_model",
        ])
        xyz = final["canonical_xyz"]
        for i in range(N):
            labels = []
            if bool(remained_dead_static[i]): labels.append("remained_dead_static")
            if bool(rescued_motion_but_static[i]): labels.append("rescued_motion_but_static")
            if bool(became_dead[i]): labels.append("became_dead")
            if bool(activated[i]): labels.append("activated")
            if not labels and not bool(fg[i]):
                labels.append("background")
            elif not labels:
                labels.append("other")
            writer.writerow([
                i, ";".join(labels), float(xyz[i,0]), float(xyz[i,1]), float(xyz[i,2]), int(final["cell_id"][i]),
                int(initial_cluster[i]), int(early_cluster[i]), int(final_cluster[i]),
                float(early.get("joint_part_confidence", torch.ones(N))[i]),
                float(final.get("joint_part_confidence", torch.ones(N))[i]),
                float(early["dynamic_prob"][i]), float(final["dynamic_prob"][i]),
                float(early.get("mean_proposal_speed", early["mean_speed"])[i]),
                float(final.get("mean_proposal_speed", final["mean_speed"])[i]),
                float(early["max_speed"][i]), float(final["max_speed"][i]),
                float(early["max_displacement"][i]), float(final["max_displacement"][i]),
                float(early["nearest_high_motion_seed_distance"][i]), float(final["nearest_high_motion_seed_distance"][i]),
                float(early["nearest_high_motion_seed_dino_similarity"][i]) if torch.isfinite(early["nearest_high_motion_seed_dino_similarity"][i]) else "nan",
                float(final["nearest_high_motion_seed_dino_similarity"][i]) if torch.isfinite(final["nearest_high_motion_seed_dino_similarity"][i]) else "nan",
                float(early["q_model_identity_vs_rigid"][i]) if torch.isfinite(early["q_model_identity_vs_rigid"][i]) else "nan",
                float(final["q_model_identity_vs_rigid"][i]) if torch.isfinite(final["q_model_identity_vs_rigid"][i]) else "nan",
            ])
    print(f"Wrote {out_dir / 'phase_comparison_summary.json'}")
    print(f"Wrote {out_dir / 'phase_comparison_points.csv'}")


if __name__ == "__main__":
    main()
