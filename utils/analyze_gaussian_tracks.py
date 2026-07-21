"""Diagnostics for exported dynamic Gaussian trajectories.

This is intentionally independent of the renderer.  It reads gaussian_tracks.pt
and reports whether the learned transport is temporally coherent or whether a
subset of points starts moving late / jumps between frames.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch


def _to_np(x):
    return x.detach().cpu().numpy() if isinstance(x, torch.Tensor) else np.asarray(x)


def analyze_tracks(tracks_path: str, out_json: str, out_prefix: str | None = None, dynamic_threshold: float = 0.5) -> dict:
    tracks = torch.load(str(tracks_path), map_location="cpu")
    pos = _to_np(tracks["positions"]).astype(np.float32)
    vel = _to_np(tracks["velocities"]).astype(np.float32) if "velocities" in tracks and tracks["velocities"] is not None else None
    prob = _to_np(tracks.get("dynamic_prob", np.zeros((pos.shape[1], 1), dtype=np.float32))).reshape(-1).astype(np.float32)
    frame_ids = _to_np(tracks.get("frame_ids", np.arange(pos.shape[0]))).reshape(-1).astype(np.int64)

    T, N, _ = pos.shape
    labels = prob >= float(dynamic_threshold)
    disp = np.linalg.norm(pos - pos[0:1], axis=2)
    max_disp = disp.max(axis=0)
    step = np.linalg.norm(np.diff(pos, axis=0), axis=2) if T > 1 else np.zeros((0, N), dtype=np.float32)
    accel = np.linalg.norm(np.diff(pos, n=2, axis=0), axis=2) if T > 2 else np.zeros((0, N), dtype=np.float32)
    speed = np.linalg.norm(vel, axis=2) if vel is not None else step

    dyn = labels if labels.any() else max_disp >= np.percentile(max_disp, 80)
    sig = dyn & (max_disp > max(0.01, float(np.percentile(max_disp, 75))))

    onset_stats = {}
    onset_arrays = {}
    for frac in (0.05, 0.10, 0.20, 0.50):
        onset = np.full((N,), np.nan, dtype=np.float32)
        for i in np.flatnonzero(sig):
            thr = max(0.002, float(frac) * float(max_disp[i]))
            hit = np.flatnonzero(disp[:, i] >= thr)
            if hit.size:
                onset[i] = float(frame_ids[int(hit[0])])
        vals = onset[sig]
        vals = vals[np.isfinite(vals)]
        onset_arrays[str(frac)] = onset
        if vals.size:
            onset_stats[str(frac)] = {
                "count": int(vals.size),
                "min": float(vals.min()),
                "p10": float(np.percentile(vals, 10)),
                "p25": float(np.percentile(vals, 25)),
                "median": float(np.percentile(vals, 50)),
                "p75": float(np.percentile(vals, 75)),
                "p90": float(np.percentile(vals, 90)),
                "max": float(vals.max()),
                "late_gt_30": int(np.count_nonzero(vals > 30)),
            }
        else:
            onset_stats[str(frac)] = {"count": 0}

    mean_step_dyn = step[:, dyn].mean(axis=1) if step.size and dyn.any() else np.zeros((max(T-1,0),), dtype=np.float32)
    p95_step_dyn = np.percentile(step[:, dyn], 95, axis=1) if step.size and dyn.any() else np.zeros_like(mean_step_dyn)
    top_jumps = sorted(
        [
            {
                "to_frame": int(frame_ids[i + 1]),
                "mean_step_dynamic": float(mean_step_dyn[i]),
                "p95_step_dynamic": float(p95_step_dyn[i]),
            }
            for i in range(mean_step_dyn.shape[0])
        ],
        key=lambda x: -x["mean_step_dynamic"],
    )[:20]

    bbox_min = pos.reshape(-1, 3).min(axis=0)
    bbox_max = pos.reshape(-1, 3).max(axis=0)
    out = {
        "tracks_path": str(tracks_path),
        "num_frames": int(T),
        "num_gaussians": int(N),
        "position_bbox_min": bbox_min.astype(float).tolist(),
        "position_bbox_max": bbox_max.astype(float).tolist(),
        "position_range_xyz": (bbox_max - bbox_min).astype(float).tolist(),
        "dynamic_prob": {
            "threshold": float(dynamic_threshold),
            "min": float(np.nanmin(prob)),
            "max": float(np.nanmax(prob)),
            "mean": float(np.nanmean(prob)),
            "num_dynamic": int(np.count_nonzero(labels)),
        },
        "max_displacement_percentiles_all": {str(p): float(np.percentile(max_disp, p)) for p in (0, 25, 50, 75, 90, 95, 99, 100)},
        "max_displacement_percentiles_dynamic": {str(p): float(np.percentile(max_disp[dyn], p)) for p in (0, 25, 50, 75, 90, 95, 99, 100)} if dyn.any() else {},
        "onset_frame_stats_by_fraction_of_own_max_displacement": onset_stats,
        "top_dynamic_jump_frames": top_jumps,
        "position_second_difference_percentiles": {str(p): float(np.percentile(accel, p)) for p in (50, 75, 90, 95, 99, 100)} if accel.size else {},
        "speed_dynamic_percentiles": {str(p): float(np.percentile(speed[:, dyn], p)) for p in (0, 25, 50, 75, 90, 95, 98, 99, 100)} if dyn.any() else {},
    }

    Path(out_json).parent.mkdir(parents=True, exist_ok=True)
    Path(out_json).write_text(json.dumps(out, indent=2), encoding="utf-8")

    if out_prefix:
        try:
            import matplotlib.pyplot as plt
            pref = Path(out_prefix)
            pref.parent.mkdir(parents=True, exist_ok=True)
            vals = onset_arrays["0.5"][sig]
            vals = vals[np.isfinite(vals)]
            if vals.size:
                plt.figure(figsize=(8, 4))
                plt.hist(vals, bins=np.arange(frame_ids.min(), frame_ids.max() + 2) - 0.5)
                plt.xlabel("Onset frame for 50% of own max displacement")
                plt.ylabel("Number of dynamic Gaussians")
                plt.title("Motion lag / asynchronous displacement onset")
                plt.tight_layout()
                plt.savefig(str(pref) + "_onset_hist.png", dpi=160)
                plt.close()

            if dyn.any():
                frac = disp[:, dyn] / np.maximum(max_disp[dyn][None, :], 1e-8)
                plt.figure(figsize=(8, 4))
                for q in (5, 25, 50, 75, 95):
                    plt.plot(frame_ids, np.percentile(frac, q, axis=1), label=f"p{q}")
                plt.xlabel("Frame")
                plt.ylabel("Displacement / own max displacement")
                plt.title("Dynamic Gaussian displacement progress percentiles")
                plt.legend()
                plt.tight_layout()
                plt.savefig(str(pref) + "_displacement_progress.png", dpi=160)
                plt.close()

            if mean_step_dyn.size:
                plt.figure(figsize=(8, 4))
                plt.plot(frame_ids[1:], mean_step_dyn, label="mean dynamic step")
                plt.plot(frame_ids[1:], p95_step_dyn, label="p95 dynamic step")
                plt.xlabel("Frame")
                plt.ylabel("Frame-to-frame displacement [world units]")
                plt.title("Dynamic Gaussian frame-to-frame jumps")
                plt.legend()
                plt.tight_layout()
                plt.savefig(str(pref) + "_step_jumps.png", dpi=160)
                plt.close()
        except Exception as exc:
            out["plot_error"] = str(exc)
            Path(out_json).write_text(json.dumps(out, indent=2), encoding="utf-8")
    return out


def main():
    parser = argparse.ArgumentParser(description="Analyze gaussian_tracks.pt temporal coherence and lag.")
    parser.add_argument("--tracks_path", required=True)
    parser.add_argument("--out_json", required=True)
    parser.add_argument("--out_prefix", default="")
    parser.add_argument("--dynamic_threshold", type=float, default=0.5)
    args = parser.parse_args()
    analyze_tracks(args.tracks_path, args.out_json, args.out_prefix or None, args.dynamic_threshold)


if __name__ == "__main__":
    main()
