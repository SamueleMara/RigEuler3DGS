#!/usr/bin/env python3
"""
Foreground-only per-frame K-means segmentation for each camera.

Goal
----
Cluster ONLY the foreground pixels selected by the binary masks, producing
a per-frame segmentation of the object into K parts.

This is NOT a motion segmentation algorithm. It is a pure image-space
appearance+position clustering cue that can be used as a weak prior for
part boundaries.

Features used for clustering
----------------------------
For each foreground pixel p = (x, y), we build:

    f(p) = [ w_color * color(p),  w_xy * x_norm,  w_xy * y_norm ]

Default color is CIELab because it is more stable than raw RGB.
Spatial coordinates are included so that K-means prefers compact regions
instead of scattering identical colors across the object.

Temporal consistency
--------------------
Cluster identities are kept more stable over time by initializing each frame
from the previous frame's cluster centers. This avoids random color/label
swaps between consecutive frames.

Expected input
--------------
ROOT/
  rgb/cam_000/frame_000000.png
  ...
  mask/cam_000/frame_000000.png
  ...

or:
  video_rgb/cam_000/frame_000000.png

Outputs
-------
ROOT/foreground_kmeans_k{K}/
  cam_000/
    labels/frame_000000.png
    overlay/frame_000000.png
    masked_clusters/frame_000000.png
    cluster_stats.csv
    cam_000_kmeans_overlay.mp4
  ...
  summary.csv

Example
-------
python foreground_kmeans_segmentation.py \
    /home/samuelemara/Joint3DGS/RigEuler3DGS/input/USB_100109 \
    --k 2

Useful variants
---------------
# More spatially compact parts
python foreground_kmeans_segmentation.py ROOT --k 2 --xy-weight 0.9

# More color-driven segmentation
python foreground_kmeans_segmentation.py ROOT --k 2 --xy-weight 0.25

# Use raw RGB instead of Lab
python foreground_kmeans_segmentation.py ROOT --k 2 --color-space rgb
"""

import argparse
import csv
import math
from pathlib import Path

import cv2
import numpy as np


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------

PALETTE = np.array(
    [
        [255, 64, 64],
        [64, 200, 255],
        [80, 255, 120],
        [255, 220, 80],
        [200, 100, 255],
        [255, 150, 60],
        [120, 255, 255],
        [255, 120, 180],
        [180, 255, 100],
        [100, 180, 255],
        [255, 255, 255],
        [160, 160, 160],
    ],
    dtype=np.uint8,
)


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
    raise FileNotFoundError("Could not find rgb/cam_* or video_rgb/cam_*.")


def list_frame_paths(folder: Path):
    for ext in ("png", "jpg", "jpeg"):
        files = sorted(folder.glob(f"frame_*.{ext}"))
        if files:
            return files
    return []


def frame_id(path: Path):
    return int(path.stem.split("_")[-1])


def read_image(path: Path):
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise RuntimeError(f"Cannot read image: {path}")
    return img


def read_mask(path: Path, target_hw=None):
    mask = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise RuntimeError(f"Cannot read mask: {path}")
    if target_hw is not None and mask.shape != target_hw:
        mask = cv2.resize(mask, (target_hw[1], target_hw[0]), interpolation=cv2.INTER_NEAREST)
    return mask > 0


def ensure_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)


def build_palette(k: int):
    if k <= len(PALETTE):
        return PALETTE[:k].copy()

    out = []
    for i in range(k):
        hue = int((i * 137.508) % 180)
        hsv = np.uint8([[[hue, 220, 255]]])
        bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0, 0]
        out.append(bgr)
    return np.asarray(out, dtype=np.uint8)


def nearest_center_labels(features: np.ndarray, centers: np.ndarray):
    # features: [N, D], centers: [K, D]
    d2 = ((features[:, None, :] - centers[None, :, :]) ** 2).sum(axis=2)
    return np.argmin(d2, axis=1).astype(np.int32)


def remap_by_overlap(prev_labels_img, curr_labels_img, mask, k):
    """
    Optional extra label stabilization:
    remap current cluster IDs to maximize overlap with previous frame labels.
    Greedy is enough for small K.
    """
    if prev_labels_img is None:
        return curr_labels_img

    fg = mask > 0
    prev = prev_labels_img[fg]
    curr = curr_labels_img[fg]

    overlap = np.zeros((k, k), dtype=np.int64)
    for a in range(k):
        for b in range(k):
            overlap[a, b] = np.sum((prev == a) & (curr == b))

    # greedy matching
    used_prev = set()
    used_curr = set()
    mapping = {}
    flat = []
    for a in range(k):
        for b in range(k):
            flat.append((overlap[a, b], a, b))
    flat.sort(reverse=True)

    for score, a, b in flat:
        if a in used_prev or b in used_curr:
            continue
        mapping[b] = a
        used_prev.add(a)
        used_curr.add(b)

    for b in range(k):
        if b not in mapping:
            for a in range(k):
                if a not in used_prev:
                    mapping[b] = a
                    used_prev.add(a)
                    break

    out = curr_labels_img.copy()
    for old_b, new_a in mapping.items():
        out[curr_labels_img == old_b] = new_a
    return out


# -----------------------------------------------------------------------------
# Feature construction
# -----------------------------------------------------------------------------

def image_to_color_features(img_bgr: np.ndarray, color_space: str):
    if color_space == "rgb":
        out = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        return out
    if color_space == "lab":
        out = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
        # Put all channels roughly on similar scale.
        out[..., 0] /= 255.0
        out[..., 1] = (out[..., 1] - 128.0) / 127.0
        out[..., 2] = (out[..., 2] - 128.0) / 127.0
        return out
    if color_space == "hsv":
        out = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV).astype(np.float32)
        out[..., 0] /= 180.0
        out[..., 1] /= 255.0
        out[..., 2] /= 255.0
        return out
    raise ValueError(f"Unsupported color space: {color_space}")


def build_foreground_features(
    img_bgr: np.ndarray,
    mask: np.ndarray,
    color_space: str,
    color_weight: float,
    xy_weight: float,
):
    h, w = mask.shape
    color = image_to_color_features(img_bgr, color_space)

    yy, xx = np.meshgrid(
        np.linspace(0.0, 1.0, h, dtype=np.float32),
        np.linspace(0.0, 1.0, w, dtype=np.float32),
        indexing="ij",
    )
    xy = np.stack([xx, yy], axis=-1)

    feats = np.concatenate(
        [
            color_weight * color,
            xy_weight * xy,
        ],
        axis=-1,
    )

    fg = mask > 0
    pts = feats[fg].astype(np.float32)
    return pts, fg


# -----------------------------------------------------------------------------
# K-means
# -----------------------------------------------------------------------------

def run_kmeans(
    features: np.ndarray,
    k: int,
    rng_seed: int,
    attempts: int,
    initial_centers: np.ndarray = None,
):
    if len(features) < k:
        raise RuntimeError(
            f"Foreground pixel count {len(features)} is smaller than K={k}."
        )

    criteria = (
        cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
        100,
        1e-4,
    )

    if initial_centers is None:
        compactness, labels, centers = cv2.kmeans(
            data=features,
            K=k,
            bestLabels=None,
            criteria=criteria,
            attempts=attempts,
            flags=cv2.KMEANS_PP_CENTERS,
        )
    else:
        init_labels = nearest_center_labels(features, initial_centers).reshape(-1, 1)
        compactness, labels, centers = cv2.kmeans(
            data=features,
            K=k,
            bestLabels=init_labels,
            criteria=criteria,
            attempts=1,
            flags=cv2.KMEANS_USE_INITIAL_LABELS,
        )

    labels = labels.reshape(-1).astype(np.int32)
    return labels, centers.astype(np.float32), float(compactness)


# -----------------------------------------------------------------------------
# Rendering
# -----------------------------------------------------------------------------

def colorize_labels(labels_img: np.ndarray, mask: np.ndarray, palette: np.ndarray):
    out = np.zeros((labels_img.shape[0], labels_img.shape[1], 3), dtype=np.uint8)
    fg = mask > 0
    out[fg] = palette[labels_img[fg] % len(palette)]
    return out


def make_overlay(img_bgr: np.ndarray, color_labels: np.ndarray, mask: np.ndarray, alpha: float):
    out = img_bgr.copy()
    fg = mask > 0
    blend = (
        alpha * color_labels.astype(np.float32)
        + (1.0 - alpha) * img_bgr.astype(np.float32)
    ).astype(np.uint8)
    out[fg] = blend[fg]
    return out


def make_masked_clusters(color_labels: np.ndarray, mask: np.ndarray):
    out = np.zeros_like(color_labels)
    out[mask > 0] = color_labels[mask > 0]
    return out


def draw_cluster_boundaries(img_bgr: np.ndarray, labels_img: np.ndarray, mask: np.ndarray):
    out = img_bgr.copy()
    fg = mask > 0

    edges = np.zeros(mask.shape, dtype=np.uint8)

    # Horizontal boundaries
    diff_x = np.zeros_like(edges)
    diff_x[:, 1:] = (labels_img[:, 1:] != labels_img[:, :-1]).astype(np.uint8) * 255

    # Vertical boundaries
    diff_y = np.zeros_like(edges)
    diff_y[1:, :] = (labels_img[1:, :] != labels_img[:-1, :]).astype(np.uint8) * 255

    edges = np.maximum(diff_x, diff_y)
    edges[~fg] = 0

    out[edges > 0] = (255, 255, 255)
    return out


# -----------------------------------------------------------------------------
# Per-camera processing
# -----------------------------------------------------------------------------

def process_camera(
    root: Path,
    rgb_parent: str,
    camera: str,
    frame_paths,
    mask_paths_map,
    out_root: Path,
    k: int,
    color_space: str,
    color_weight: float,
    xy_weight: float,
    overlay_alpha: float,
    attempts: int,
    rng_seed: int,
    video_fps: float,
    make_videos: bool,
):
    cam_out = out_root / camera
    labels_dir = cam_out / "labels"
    overlay_dir = cam_out / "overlay"
    masked_dir = cam_out / "masked_clusters"
    boundary_dir = cam_out / "boundaries"

    for d in (labels_dir, overlay_dir, masked_dir, boundary_dir):
        ensure_dir(d)

    palette = build_palette(k)

    summary_rows = []
    prev_centers = None
    prev_labels_img = None

    overlay_frames = []
    masked_frames = []
    boundary_frames = []

    print(f"[{camera}] {len(frame_paths)} frames")

    for i, img_path in enumerate(frame_paths):
        fid = frame_id(img_path)
        mask_path = mask_paths_map.get(fid, None)
        if mask_path is None:
            print(f"  frame {fid:06d}: missing mask -> skipped")
            continue

        img = read_image(img_path)
        mask = read_mask(mask_path, target_hw=img.shape[:2])

        fg_count = int(mask.sum())
        if fg_count < k:
            print(f"  frame {fid:06d}: foreground too small ({fg_count} px) -> skipped")
            continue

        feats, fg = build_foreground_features(
            img_bgr=img,
            mask=mask,
            color_space=color_space,
            color_weight=color_weight,
            xy_weight=xy_weight,
        )

        labels_fg, centers, compactness = run_kmeans(
            features=feats,
            k=k,
            rng_seed=rng_seed,
            attempts=attempts,
            initial_centers=prev_centers,
        )

        labels_img = np.full(mask.shape, 255, dtype=np.uint8)
        labels_img[fg] = labels_fg.astype(np.uint8)

        # Extra label stabilization via overlap with previous frame.
        labels_img = remap_by_overlap(prev_labels_img, labels_img, mask, k)

        # Recompute centers consistently with remapped labels for next frame init.
        remapped_fg = labels_img[fg].astype(np.int32)
        new_centers = []
        for c in range(k):
            sel = remapped_fg == c
            if np.any(sel):
                new_centers.append(feats[sel].mean(axis=0))
            else:
                # fallback if a cluster vanished
                if prev_centers is not None:
                    new_centers.append(prev_centers[c])
                else:
                    new_centers.append(np.zeros((feats.shape[1],), dtype=np.float32))
        prev_centers = np.asarray(new_centers, dtype=np.float32)
        prev_labels_img = labels_img.copy()

        color_labels = colorize_labels(labels_img, mask, palette)
        overlay = make_overlay(img, color_labels, mask, overlay_alpha)
        masked = make_masked_clusters(color_labels, mask)
        boundaries = draw_cluster_boundaries(img, labels_img, mask)

        labels_path = labels_dir / f"frame_{fid:06d}.png"
        overlay_path = overlay_dir / f"frame_{fid:06d}.png"
        masked_path = masked_dir / f"frame_{fid:06d}.png"
        boundary_path = boundary_dir / f"frame_{fid:06d}.png"

        cv2.imwrite(str(labels_path), labels_img)
        cv2.imwrite(str(overlay_path), overlay)
        cv2.imwrite(str(masked_path), masked)
        cv2.imwrite(str(boundary_path), boundaries)

        overlay_frames.append(overlay)
        masked_frames.append(masked)
        boundary_frames.append(boundaries)

        cluster_sizes = [(labels_img[fg] == c).sum() for c in range(k)]
        row = {
            "frame": fid,
            "foreground_pixels": fg_count,
            "compactness": compactness,
        }
        for c in range(k):
            row[f"cluster_{c}_pixels"] = int(cluster_sizes[c])
            row[f"cluster_{c}_fraction"] = float(cluster_sizes[c] / max(fg_count, 1))
        summary_rows.append(row)

        if i % 10 == 0 or i == len(frame_paths) - 1:
            print(
                f"  frame {fid:06d}: fg={fg_count:6d} | "
                + " ".join([f"c{c}={cluster_sizes[c]:5d}" for c in range(k)])
            )

    # Save per-camera CSV
    stats_csv = cam_out / "cluster_stats.csv"
    fields = ["frame", "foreground_pixels", "compactness"]
    for c in range(k):
        fields.extend([f"cluster_{c}_pixels", f"cluster_{c}_fraction"])

    with open(stats_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in summary_rows:
            writer.writerow(row)

    if make_videos and overlay_frames:
        h, w = overlay_frames[0].shape[:2]

        def write_video(frames, path):
            writer = cv2.VideoWriter(
                str(path),
                cv2.VideoWriter_fourcc(*"mp4v"),
                video_fps,
                (w, h),
            )
            if not writer.isOpened():
                raise RuntimeError(f"Could not create video {path}")
            for frame in frames:
                writer.write(frame)
            writer.release()

        write_video(overlay_frames, cam_out / f"{camera}_kmeans_overlay.mp4")
        write_video(masked_frames, cam_out / f"{camera}_kmeans_masked_clusters.mp4")
        write_video(boundary_frames, cam_out / f"{camera}_kmeans_boundaries.mp4")

    return summary_rows


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("root", type=str)
    ap.add_argument("--k", type=int, required=True,
                    help="Number of foreground clusters / candidate moving parts.")
    ap.add_argument("--color-space", type=str, default="lab",
                    choices=["lab", "rgb", "hsv"])
    ap.add_argument("--color-weight", type=float, default=1.0)
    ap.add_argument("--xy-weight", type=float, default=0.50,
                    help="Weight of normalized image coordinates in K-means features.")
    ap.add_argument("--overlay-alpha", type=float, default=0.60)
    ap.add_argument("--attempts", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--fps", type=float, default=10.0)
    ap.add_argument("--output", type=str, default=None)
    ap.add_argument("--no-video", action="store_true")
    args = ap.parse_args()

    if args.k < 2:
        raise ValueError("--k must be >= 2")

    root = Path(args.root)
    rgb_parent, cameras = find_rgb_parent(root)

    if not (root / "mask").is_dir():
        raise FileNotFoundError(f"Missing mask folder: {root / 'mask'}")

    out_root = (
        Path(args.output)
        if args.output is not None
        else root / f"foreground_kmeans_k{args.k}"
    )
    ensure_dir(out_root)

    all_summary = []

    for camera in cameras:
        rgb_dir = root / rgb_parent / camera
        mask_dir = root / "mask" / camera

        frame_paths = list_frame_paths(rgb_dir)
        mask_paths = list_frame_paths(mask_dir)

        if not frame_paths:
            print(f"[{camera}] no RGB frames -> skipped")
            continue
        if not mask_paths:
            print(f"[{camera}] no mask frames -> skipped")
            continue

        mask_map = {frame_id(p): p for p in mask_paths}

        summary_rows = process_camera(
            root=root,
            rgb_parent=rgb_parent,
            camera=camera,
            frame_paths=frame_paths,
            mask_paths_map=mask_map,
            out_root=out_root,
            k=args.k,
            color_space=args.color_space,
            color_weight=args.color_weight,
            xy_weight=args.xy_weight,
            overlay_alpha=args.overlay_alpha,
            attempts=args.attempts,
            rng_seed=args.seed,
            video_fps=args.fps,
            make_videos=(not args.no_video),
        )

        for row in summary_rows:
            row2 = {"camera": camera}
            row2.update(row)
            all_summary.append(row2)

    summary_csv = out_root / "summary.csv"
    fields = ["camera", "frame", "foreground_pixels", "compactness"]
    for c in range(args.k):
        fields.extend([f"cluster_{c}_pixels", f"cluster_{c}_fraction"])

    with open(summary_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in all_summary:
            writer.writerow(row)

    print("\n[DONE]")
    print(f"Output root: {out_root}")
    print(f"Summary:     {summary_csv}")


if __name__ == "__main__":
    main()
