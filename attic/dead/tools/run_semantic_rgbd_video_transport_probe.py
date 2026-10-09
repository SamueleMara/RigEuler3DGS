#!/usr/bin/env python3
"""Run strict semantic-foreground adjacent RGB-D transport over a video root."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
from PIL import Image
import torch

REPO = Path(__file__).resolve().parent.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from utils.rgbd_track_lifting import propagate_pixel_track_to_world_track
from utils.temporal_pixel_diffusion import sample_forward_pixel_transport, score_samples_with_rgbd
from utils.temporal_pixel_fm import build_foreground_adjacent_frame_pairs, compute_forward_backward_pixel_consistency


def _frame_id(path: Path) -> int:
    digits = "".join(ch for ch in path.stem if ch.isdigit())
    return int(digits[-6:] if len(digits) >= 6 else digits or 0)


def _load_rgb(path: Path) -> torch.Tensor:
    arr = np.asarray(Image.open(path).convert("RGB")).astype("float32") / 255.0
    return torch.from_numpy(arr).permute(2, 0, 1)


def _load_depth(root: Path, cam: str, frame: int, shape: tuple[int, int]) -> torch.Tensor:
    stem = f"frame_{frame:06d}"
    for rel in (f"depth_npy/{cam}/{stem}.npy", f"depth/{cam}/{stem}.npy"):
        path = root / rel
        if path.is_file():
            arr = np.load(path).astype("float32")
            return torch.from_numpy(arr[..., 0] if arr.ndim == 3 else arr)
    for rel in (f"depth_png/{cam}/{stem}.png", f"depth/{cam}/{stem}.png"):
        path = root / rel
        if path.is_file():
            arr = np.asarray(Image.open(path)).astype("float32")
            return torch.from_numpy(arr / (65535.0 if arr.max() > 255 else 255.0))
    return torch.ones(shape, dtype=torch.float32)


def _load_mask(root: Path, cam: str, frame: int, shape: tuple[int, int], allow_missing: bool) -> torch.Tensor:
    stem = f"frame_{frame:06d}"
    for folder in ("mask", "masks", "semantic_mask", "semantic_masks"):
        for suffix in (".png", ".jpg", ".jpeg", ".npy"):
            path = root / folder / cam / f"{stem}{suffix}"
            if path.is_file():
                arr = np.load(path) if suffix == ".npy" else np.asarray(Image.open(path))
                return torch.from_numpy(arr > 0)
    if allow_missing:
        return torch.ones(shape, dtype=torch.bool)
    raise FileNotFoundError(f"missing semantic foreground mask for {cam} {stem}")


def _write_json(path: Path, data) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def _load_timestamps(root: Path):
    for rel in ("timestamps.json", "metadata/timestamps.json"):
        path = root / rel
        if path.is_file():
            return json.loads(path.read_text(encoding="utf-8"))
    return None


def _timestamp(table, cam: str, frame: int, fps: float) -> float:
    if table is None:
        return float(frame) / float(fps)
    if isinstance(table, list):
        return float(table[frame])
    nested = table.get(cam, table) if isinstance(table, dict) else table
    if isinstance(nested, list):
        value = nested[frame]
        return float(value.get("timestamp", value.get("time", value.get("timestamp_seconds"))) if isinstance(value, dict) else value)
    if isinstance(nested, dict):
        for key, value in nested.items():
            digits = "".join(ch for ch in str(key) if ch.isdigit())
            if digits and int(digits) == int(frame):
                return float(value.get("timestamp", value.get("time", value.get("timestamp_seconds"))) if isinstance(value, dict) else value)
    raise KeyError(f"missing timestamp for {cam} frame {frame}")


def run(args) -> dict:
    root = Path(args.root_scene).expanduser().resolve()
    output = Path(args.output_dir).expanduser().resolve()
    cams = sorted(p.name for p in (root / "rgb").glob("cam_*") if p.is_dir())[: int(args.max_cameras)]
    if not cams:
        raise FileNotFoundError(f"no rgb/cam_* folders under {root}")
    timestamps = _load_timestamps(root)
    reports = []
    saved_tracks = []
    for cam in cams:
        rgb_files = sorted((root / "rgb" / cam).glob("frame_*.*"), key=_frame_id)[: int(args.max_frames)]
        frames = [_frame_id(p) for p in rgb_files]
        for a, b, pa, pb in zip(frames[:-1], frames[1:], rgb_files[:-1], rgb_files[1:]):
            rgb0, rgb1 = _load_rgb(pa), _load_rgb(pb)
            shape = tuple(rgb0.shape[-2:])
            depth0, depth1 = _load_depth(root, cam, a, shape), _load_depth(root, cam, b, shape)
            mask0 = _load_mask(root, cam, a, shape, bool(args.allow_missing_masks))
            mask1 = _load_mask(root, cam, b, shape, bool(args.allow_missing_masks))
            dt = _timestamp(timestamps, cam, b, float(args.fps)) - _timestamp(timestamps, cam, a, float(args.fps))
            fwd = build_foreground_adjacent_frame_pairs(rgb0, rgb1, depth0, depth1, mask0, mask1, delta_t=dt, frame_pair=(a, b), camera_id=cam, max_source_pixels=int(args.max_pixels))
            bwd = build_foreground_adjacent_frame_pairs(rgb1, rgb0, depth1, depth0, mask1, mask0, delta_t=dt, frame_pair=(b, a), camera_id=cam, max_source_pixels=int(args.max_pixels), direction="backward")
            cyc = compute_forward_backward_pixel_consistency(fwd, bwd)
            samples = sample_forward_pixel_transport(fwd.source_pixel_indices[:128], mask1, image_shape=shape, samples=4)
            sample_score = score_samples_with_rgbd(rgb0, rgb1, depth0, depth1, fwd.source_pixel_indices[:128], samples, mask1) if samples.numel() else torch.empty(0)
            if int(fwd.source_pixel_indices.numel()) > 0:
                width = shape[1]
                uv = torch.stack((fwd.source_pixel_indices[:64] % width, fwd.source_pixel_indices[:64] // width), dim=1).float()
                d = depth0.reshape(-1)[fwd.source_pixel_indices[:64]]
                track = propagate_pixel_track_to_world_track(uv, d, torch.eye(3), torch.eye(4), frame_ids=torch.full((uv.shape[0],), a))
                saved_tracks.append(track.positions)
            reports.append({
                "camera": cam,
                "pair": [a, b],
                "foreground_source": int(mask0.sum()),
                "foreground_target": int(mask1.sum()),
                "matches": int(fwd.source_pixel_indices.numel()),
                "median_cycle_error": cyc["median_cycle_error"],
                "diffusion_samples": int(samples.numel()),
                "diffusion_score_median": float(sample_score.median()) if sample_score.numel() else 0.0,
            })
    summary = {
        "root_scene": str(root),
        "pairs": len(reports),
        "foreground_pixels_total": int(sum(r["foreground_source"] for r in reports)),
        "matched_pixels_total": int(sum(r["matches"] for r in reports)),
        "reports": reports,
    }
    output.mkdir(parents=True, exist_ok=True)
    _write_json(output / "semantic_rgbd_video_transport_summary.json", summary)
    if saved_tracks:
        torch.save(torch.cat(saved_tracks, 0), output / "lifted_world_track_samples.pt")
    print(json.dumps(summary, indent=2))
    return summary


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--root-scene", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--max-frames", type=int, default=8)
    p.add_argument("--max-cameras", type=int, default=2)
    p.add_argument("--max-pixels", type=int, default=2048)
    p.add_argument("--fps", type=float, default=30.0)
    p.add_argument("--allow-missing-masks", action="store_true")
    run(p.parse_args())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
