#!/usr/bin/env python3
"""
Plot / visualize ALL dynamic seeds from temporal_track_observations.csv.

Why this script
---------------
The previous K+1 mask script uses dynamic seeds as sparse support. Before doing
anything more sophisticated, we need to SEE all the seed evidence directly.

This script visualizes:
1) per-frame seed overlays for every camera
2) per-camera videos showing all dynamic seeds through time
3) a synchronized 2x4 all-camera panel video
4) per-camera seed-density heatmaps over the whole sequence

Dynamic seed types
------------------
- GLOBAL dynamic seed: label == DYNAMIC and global_track_id >= 0
  (multiview-confirmed dynamic track)
- LOCAL dynamic seed:  label == DYNAMIC and global_track_id < 0
  (same-view dynamic only, not confirmed cross-camera)

By default the overlay shows BOTH, with different colors:
- GLOBAL = green
- LOCAL  = red

Optionally:
- --only-global : show only multiview-confirmed dynamic seeds
- --only-local  : show only local-only dynamic seeds

Expected input
--------------
ROOT/
  rgb/cam_000/frame_000000.png    or video_rgb/...
  mask/cam_000/frame_000000.png   (optional, for drawing only on foreground)
  dynamic_sift_multicam/temporal_track_observations.csv

Outputs
-------
ROOT/dynamic_seed_visualization/
  cam_000/
    overlay/frame_000000.png
    cam_000_dynamic_seeds.mp4
    cam_000_seed_density_all.png
    cam_000_seed_density_global.png
    cam_000_seed_density_local.png
    cam_000_seed_counts.csv
  ...
  all_cameras_dynamic_seeds.mp4
  dynamic_seed_summary.csv
"""

import argparse
import csv
import math
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


# -----------------------------------------------------------------------------
# Data structures
# -----------------------------------------------------------------------------

@dataclass
class SeedObs:
    camera: str
    frame: int
    x: float
    y: float
    track_id: int
    global_track_id: int
    label: str
    is_global: bool


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------

GLOBAL_COLOR = (60, 220, 60)    # BGR, green
LOCAL_COLOR  = (40, 40, 255)    # BGR, red-ish
TEXT_COLOR   = (255, 255, 255)
BG_COLOR     = (0, 0, 0)


def ensure_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)


def frame_id(path: Path):
    return int(path.stem.split("_")[-1])


def find_rgb_parent(root: Path):
    for parent in ("rgb", "video_rgb"):
        p = root / parent
        if p.is_dir():
            cams = sorted(
                x.name for x in p.iterdir()
                if x.is_dir() and x.name.startswith("cam_")
            )
            if cams:
                return parent, cams
    raise FileNotFoundError("Could not find rgb/cam_* or video_rgb/cam_*")


def list_frame_paths(folder: Path):
    for ext in ("png", "jpg", "jpeg"):
        files = sorted(folder.glob(f"frame_*.{ext}"))
        if files:
            return files
    return []


def read_image(path: Path):
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise RuntimeError(f"Cannot read image: {path}")
    return img


def read_mask(path: Path, target_hw=None):
    m = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if m is None:
        return None
    if target_hw is not None and m.shape != target_hw:
        m = cv2.resize(m, (target_hw[1], target_hw[0]), interpolation=cv2.INTER_NEAREST)
    return m > 0


def write_video(frames, path, fps):
    if not frames:
        return
    h, w = frames[0].shape[:2]
    writer = cv2.VideoWriter(
        str(path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (w, h),
    )
    if not writer.isOpened():
        raise RuntimeError(f"Could not create video: {path}")
    for f in frames:
        writer.write(f)
    writer.release()


# -----------------------------------------------------------------------------
# CSV loading
# -----------------------------------------------------------------------------

def load_dynamic_seeds(csv_path: Path):
    required = {"camera", "track_id", "global_track_id", "label", "frame", "x", "y"}
    by_cam_frame = defaultdict(list)

    with open(csv_path, "r", newline="") as f:
        reader = csv.DictReader(f)
        fields = set(reader.fieldnames or [])
        missing = required - fields
        if missing:
            raise RuntimeError(
                f"CSV {csv_path} missing required columns: {sorted(missing)}"
            )

        for row in reader:
            label = str(row["label"]).strip()
            if label != "DYNAMIC":
                continue

            gid = int(float(row["global_track_id"]))
            obs = SeedObs(
                camera=str(row["camera"]).strip(),
                frame=int(float(row["frame"])),
                x=float(row["x"]),
                y=float(row["y"]),
                track_id=int(float(row["track_id"])),
                global_track_id=gid,
                label=label,
                is_global=(gid >= 0),
            )
            by_cam_frame[(obs.camera, obs.frame)].append(obs)

    return by_cam_frame


def filter_seeds(seeds, only_global=False, only_local=False):
    if only_global and only_local:
        raise ValueError("Cannot use both --only-global and --only-local")

    if only_global:
        return [s for s in seeds if s.is_global]
    if only_local:
        return [s for s in seeds if not s.is_global]
    return list(seeds)


# -----------------------------------------------------------------------------
# Drawing
# -----------------------------------------------------------------------------

def draw_seeds(img_bgr, seeds, mask=None, radius=4, draw_ids=False):
    out = img_bgr.copy()

    # Slightly dim background outside mask to focus on foreground if available.
    if mask is not None:
        dim = (0.25 * out).astype(np.uint8)
        out[~mask] = dim[~mask]

    n_global = 0
    n_local = 0

    for s in seeds:
        x = int(round(s.x))
        y = int(round(s.y))

        if x < 0 or x >= out.shape[1] or y < 0 or y >= out.shape[0]:
            continue

        color = GLOBAL_COLOR if s.is_global else LOCAL_COLOR
        if s.is_global:
            n_global += 1
        else:
            n_local += 1

        cv2.circle(out, (x, y), radius, color, -1, cv2.LINE_AA)
        cv2.circle(out, (x, y), radius + 2, (255, 255, 255), 1, cv2.LINE_AA)

        if draw_ids:
            tid = s.global_track_id if s.is_global else s.track_id
            prefix = "G" if s.is_global else "L"
            cv2.putText(
                out,
                f"{prefix}{tid}",
                (x + 5, y - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.35,
                color,
                1,
                cv2.LINE_AA,
            )

    # Header bar
    cv2.rectangle(out, (0, 0), (out.shape[1] - 1, 32), BG_COLOR, -1)
    txt = f"global={n_global}  local={n_local}  total={n_global + n_local}"
    cv2.putText(
        out,
        txt,
        (8, 22),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        TEXT_COLOR,
        1,
        cv2.LINE_AA,
    )
    return out, n_global, n_local


def density_heatmap(h, w, seeds, sigma=7.0):
    acc = np.zeros((h, w), dtype=np.float32)
    for s in seeds:
        x = int(round(s.x))
        y = int(round(s.y))
        if 0 <= x < w and 0 <= y < h:
            acc[y, x] += 1.0

    if acc.max() <= 0:
        vis = np.zeros((h, w, 3), dtype=np.uint8)
        return vis

    blur = cv2.GaussianBlur(acc, (0, 0), sigmaX=sigma, sigmaY=sigma)
    norm = blur / max(float(blur.max()), 1e-8)
    heat = (255.0 * norm).clip(0, 255).astype(np.uint8)
    heat = cv2.applyColorMap(heat, cv2.COLORMAP_TURBO)
    return heat


def make_panel_frame(images, titles, columns=4, footer_text=None):
    H = max(im.shape[0] for im in images)
    W = max(im.shape[1] for im in images)
    rows = int(math.ceil(len(images) / columns))
    footer_h = 34 if footer_text else 0

    canvas = np.zeros((rows * H + footer_h, columns * W, 3), dtype=np.uint8)

    for idx, (im, title) in enumerate(zip(images, titles)):
        r, c = divmod(idx, columns)
        y0, x0 = r * H, c * W

        if im.shape[:2] != (H, W):
            im = cv2.resize(im, (W, H), interpolation=cv2.INTER_LINEAR)

        tile = im.copy()
        cv2.rectangle(tile, (0, 0), (W - 1, 25), BG_COLOR, -1)
        cv2.putText(
            tile,
            title,
            (8, 18),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.48,
            TEXT_COLOR,
            1,
            cv2.LINE_AA,
        )
        canvas[y0:y0+H, x0:x0+W] = tile

    if footer_text:
        cv2.putText(
            canvas,
            footer_text,
            (10, rows * H + 22),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            TEXT_COLOR,
            1,
            cv2.LINE_AA,
        )

    return canvas


# -----------------------------------------------------------------------------
# Main processing
# -----------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("root", type=str)
    ap.add_argument("--tracks-csv", type=str, default=None,
                    help="Path to temporal_track_observations.csv. "
                         "Default: ROOT/dynamic_sift_multicam/temporal_track_observations.csv")
    ap.add_argument("--output", type=str, default=None)
    ap.add_argument("--only-global", action="store_true")
    ap.add_argument("--only-local", action="store_true")
    ap.add_argument("--use-mask", action="store_true",
                    help="If set, dim background outside the provided foreground mask.")
    ap.add_argument("--radius", type=int, default=4)
    ap.add_argument("--draw-ids", action="store_true")
    ap.add_argument("--fps", type=float, default=10.0)
    ap.add_argument("--panel-columns", type=int, default=4)
    ap.add_argument("--heatmap-sigma", type=float, default=7.0)
    ap.add_argument("--no-video", action="store_true")
    args = ap.parse_args()

    root = Path(args.root)
    tracks_csv = Path(args.tracks_csv) if args.tracks_csv else (root / "dynamic_sift_multicam" / "temporal_track_observations.csv")
    if not tracks_csv.exists():
        raise FileNotFoundError(f"Track CSV not found: {tracks_csv}")

    rgb_parent, cameras = find_rgb_parent(root)
    mask_root = root / "mask"

    out_root = Path(args.output) if args.output else (root / "dynamic_seed_visualization")
    ensure_dir(out_root)

    seeds_by_cam_frame = load_dynamic_seeds(tracks_csv)

    all_summary_rows = []
    panel_frames = []
    common_frames = None

    for cam in cameras:
        rgb_dir = root / rgb_parent / cam
        cam_out = out_root / cam
        overlay_dir = cam_out / "overlay"
        ensure_dir(overlay_dir)

        frame_paths = list_frame_paths(rgb_dir)
        if not frame_paths:
            print(f"[{cam}] no RGB frames, skipped")
            continue

        if common_frames is None:
            common_frames = [frame_id(p) for p in frame_paths]

        video_frames = []
        rows = []

        # Collect all seeds for heatmaps.
        seeds_all = []
        seeds_global = []
        seeds_local = []

        for img_path in frame_paths:
            fid = frame_id(img_path)
            img = read_image(img_path)

            mask = None
            if args.use_mask:
                mask_path = mask_root / cam / f"frame_{fid:06d}.png"
                if mask_path.exists():
                    mask = read_mask(mask_path, target_hw=img.shape[:2])

            raw_seeds = seeds_by_cam_frame.get((cam, fid), [])
            seeds = filter_seeds(raw_seeds, only_global=args.only_global, only_local=args.only_local)

            for s in seeds:
                seeds_all.append(s)
                if s.is_global:
                    seeds_global.append(s)
                else:
                    seeds_local.append(s)

            overlay, n_global, n_local = draw_seeds(
                img_bgr=img,
                seeds=seeds,
                mask=mask,
                radius=args.radius,
                draw_ids=args.draw_ids,
            )

            cv2.imwrite(str(overlay_dir / f"frame_{fid:06d}.png"), overlay)
            video_frames.append(overlay)

            row = {
                "camera": cam,
                "frame": fid,
                "global_seed_count": n_global,
                "local_seed_count": n_local,
                "total_seed_count": n_global + n_local,
            }
            rows.append(row)
            all_summary_rows.append(row)

        # Save per-camera counts CSV
        counts_csv = cam_out / "cam_seed_counts.csv"
        with open(counts_csv, "w", newline="") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=["camera", "frame", "global_seed_count", "local_seed_count", "total_seed_count"]
            )
            writer.writeheader()
            for row in rows:
                writer.writerow(row)

        # Save heatmaps
        first_img = read_image(frame_paths[0])
        h, w = first_img.shape[:2]

        hm_all = density_heatmap(h, w, seeds_all, sigma=args.heatmap_sigma)
        hm_global = density_heatmap(h, w, seeds_global, sigma=args.heatmap_sigma)
        hm_local = density_heatmap(h, w, seeds_local, sigma=args.heatmap_sigma)

        cv2.imwrite(str(cam_out / f"{cam}_seed_density_all.png"), hm_all)
        cv2.imwrite(str(cam_out / f"{cam}_seed_density_global.png"), hm_global)
        cv2.imwrite(str(cam_out / f"{cam}_seed_density_local.png"), hm_local)

        # Per-camera video
        if not args.no_video:
            write_video(video_frames, cam_out / f"{cam}_dynamic_seeds.mp4", args.fps)

        print(
            f"[{cam}] frames={len(frame_paths)} "
            f"| mean seeds/frame={np.mean([r['total_seed_count'] for r in rows]):.2f} "
            f"| mean global/frame={np.mean([r['global_seed_count'] for r in rows]):.2f} "
            f"| mean local/frame={np.mean([r['local_seed_count'] for r in rows]):.2f}"
        )

    # Summary CSV
    summary_csv = out_root / "dynamic_seed_summary.csv"
    with open(summary_csv, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["camera", "frame", "global_seed_count", "local_seed_count", "total_seed_count"]
        )
        writer.writeheader()
        for row in all_summary_rows:
            writer.writerow(row)

    # All-camera synchronized panel video
    if not args.no_video:
        # Determine common frame list from the RGB folders.
        if common_frames is None:
            common_frames = []

        for fid in common_frames:
            images = []
            titles = []
            for cam in cameras:
                p = out_root / cam / "overlay" / f"frame_{fid:06d}.png"
                if p.exists():
                    im = read_image(p)
                else:
                    # blank tile if a frame is missing
                    ref_dir = root / rgb_parent / cam
                    fps = list_frame_paths(ref_dir)
                    if fps:
                        ref_im = read_image(fps[0])
                        im = np.zeros_like(ref_im)
                    else:
                        im = np.zeros((256, 256, 3), dtype=np.uint8)

                images.append(im)
                titles.append(cam)

            footer = f"frame {fid:06d} | GLOBAL dynamic seeds = green | LOCAL dynamic seeds = red"
            panel = make_panel_frame(
                images=images,
                titles=titles,
                columns=args.panel_columns,
                footer_text=footer,
            )
            panel_frames.append(panel)

        write_video(panel_frames, out_root / "all_cameras_dynamic_seeds.mp4", args.fps)

    print("\n[DONE]")
    print(f"Output root: {out_root}")
    print(f"Summary CSV: {summary_csv}")


if __name__ == "__main__":
    main()
