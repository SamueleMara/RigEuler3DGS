#!/usr/bin/env python3
"""
Create K+1 per-frame masks from dynamic pixels:
  class 0 = static / non-dynamic foreground
  class 1..K = K moving-part masks

Key idea
--------
We do NOT K-means all foreground pixels.

Instead:
1) read dynamic pixel trajectories from temporal_track_observations.csv
2) keep only DYNAMIC tracks (by default only multiview-confirmed global tracks)
3) for each frame and camera, cluster the dynamic seed pixels into K groups
4) assign only pixels sufficiently close to dynamic seeds to one of those K groups
5) all remaining foreground pixels become class 0 (static)

So the output is exactly K+1 masks, built from dynamic pixels.

Inputs
------
ROOT/
  rgb/cam_000/frame_000000.png    or video_rgb/...
  mask/cam_000/frame_000000.png
  ...
and a track CSV produced by the dynamic_sift_multicam pipeline:
  temporal_track_observations.csv

Expected CSV columns
--------------------
camera,track_id,global_track_id,label,frame,x,y, ...

Outputs
-------
ROOT/dynamic_kplus1_masks_k{K}/
  cam_000/
    labels/frame_000000.png
    overlay/frame_000000.png
    masks/class_0/frame_000000.png
    masks/class_1/frame_000000.png
    ...
    stats.csv
    cam_000_overlay.mp4
  ...
  summary.csv

Usage
-----
python dynamic_kplus1_masks_from_tracks.py \
    /home/samuelemara/Joint3DGS/RigEuler3DGS/input/USB_100109 \
    --tracks-csv /path/to/temporal_track_observations.csv \
    --k 1

For 2 moving parts:
python dynamic_kplus1_masks_from_tracks.py ROOT --tracks-csv temporal_track_observations.csv --k 2

Notes
-----
- By default ONLY multiview-confirmed dynamic tracks are used:
      global_track_id >= 0 and label == DYNAMIC
  This is intentional because local-only dynamic tracks are less reliable.
- Pixels farther than --dynamic-radius-px from any dynamic seed are forced to class 0.
- Dynamic pixels inside that radius are assigned to one of K dynamic clusters based on
  appearance + image position features.
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
# Helpers
# -----------------------------------------------------------------------------

PALETTE = np.array(
    [
        [40, 40, 40],      # class 0 static  (dark gray)
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
    ],
    dtype=np.uint8,
)


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
    raise FileNotFoundError("Could not find rgb/cam_* or video_rgb/cam_*.")


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
        raise RuntimeError(f"Cannot read mask: {path}")
    if target_hw is not None and m.shape != target_hw:
        m = cv2.resize(m, (target_hw[1], target_hw[0]), interpolation=cv2.INTER_NEAREST)
    return m > 0


def build_palette(nclasses: int):
    if nclasses <= len(PALETTE):
        return PALETTE[:nclasses].copy()
    out = [PALETTE[0]]
    for i in range(1, nclasses):
        hue = int((i * 137.508) % 180)
        hsv = np.uint8([[[hue, 220, 255]]])
        bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0, 0]
        out.append(bgr)
    return np.asarray(out, dtype=np.uint8)


# -----------------------------------------------------------------------------
# Track CSV
# -----------------------------------------------------------------------------

@dataclass
class SeedObs:
    camera: str
    frame: int
    x: float
    y: float
    global_track_id: int
    track_id: int
    label: str


def load_dynamic_seeds(csv_path: Path, only_global: bool):
    by_cam_frame = defaultdict(list)

    with open(csv_path, "r", newline="") as f:
        reader = csv.DictReader(f)

        required = {"camera", "track_id", "global_track_id", "label", "frame", "x", "y"}
        missing = required - set(reader.fieldnames)
        if missing:
            raise RuntimeError(
                f"CSV {csv_path} is missing columns: {sorted(missing)}"
            )

        for row in reader:
            label = row["label"].strip()
            if label != "DYNAMIC":
                continue

            gid = int(float(row["global_track_id"]))
            if only_global and gid < 0:
                continue

            obs = SeedObs(
                camera=row["camera"].strip(),
                frame=int(float(row["frame"])),
                x=float(row["x"]),
                y=float(row["y"]),
                global_track_id=gid,
                track_id=int(float(row["track_id"])),
                label=label,
            )
            by_cam_frame[(obs.camera, obs.frame)].append(obs)

    return by_cam_frame


# -----------------------------------------------------------------------------
# Features
# -----------------------------------------------------------------------------

def image_to_color_features(img_bgr: np.ndarray, color_space: str):
    if color_space == "lab":
        feat = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
        feat[..., 0] /= 255.0
        feat[..., 1] = (feat[..., 1] - 128.0) / 127.0
        feat[..., 2] = (feat[..., 2] - 128.0) / 127.0
        return feat
    if color_space == "rgb":
        feat = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        return feat
    if color_space == "hsv":
        feat = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV).astype(np.float32)
        feat[..., 0] /= 180.0
        feat[..., 1] /= 255.0
        feat[..., 2] /= 255.0
        return feat
    raise ValueError(f"Unsupported color space: {color_space}")


def build_dense_features(img_bgr, mask, color_space, color_weight, xy_weight):
    h, w = mask.shape
    col = image_to_color_features(img_bgr, color_space)

    yy, xx = np.meshgrid(
        np.linspace(0.0, 1.0, h, dtype=np.float32),
        np.linspace(0.0, 1.0, w, dtype=np.float32),
        indexing="ij"
    )
    xy = np.stack([xx, yy], axis=-1)

    feat = np.concatenate(
        [color_weight * col, xy_weight * xy],
        axis=-1
    ).astype(np.float32)

    fg = mask > 0
    return feat, fg


def sample_patch_feature(feature_map, x, y, patch_radius):
    h, w = feature_map.shape[:2]
    xi = int(round(x))
    yi = int(round(y))
    x0 = max(0, xi - patch_radius)
    x1 = min(w, xi + patch_radius + 1)
    y0 = max(0, yi - patch_radius)
    y1 = min(h, yi + patch_radius + 1)

    patch = feature_map[y0:y1, x0:x1]
    if patch.size == 0:
        return feature_map[min(max(yi, 0), h - 1), min(max(xi, 0), w - 1)]
    return patch.reshape(-1, feature_map.shape[-1]).mean(axis=0)


# -----------------------------------------------------------------------------
# Small KMeans
# -----------------------------------------------------------------------------

def nearest_center_labels(features: np.ndarray, centers: np.ndarray):
    d2 = ((features[:, None, :] - centers[None, :, :]) ** 2).sum(axis=2)
    return np.argmin(d2, axis=1).astype(np.int32)


def kmeans(features, k, attempts, initial_centers=None):
    if len(features) < k:
        raise RuntimeError(f"Need at least K={k} seed samples, got {len(features)}")

    criteria = (
        cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
        100,
        1e-4,
    )

    if initial_centers is None:
        compactness, labels, centers = cv2.kmeans(
            features.astype(np.float32),
            k,
            None,
            criteria,
            attempts,
            cv2.KMEANS_PP_CENTERS,
        )
    else:
        init_labels = nearest_center_labels(features, initial_centers).reshape(-1, 1)
        compactness, labels, centers = cv2.kmeans(
            features.astype(np.float32),
            k,
            init_labels.astype(np.int32),
            criteria,
            1,
            cv2.KMEANS_USE_INITIAL_LABELS,
        )

    return labels.reshape(-1).astype(np.int32), centers.astype(np.float32), float(compactness)


def remap_labels_by_center_distance(current_centers, previous_centers):
    """
    Greedy remapping so cluster IDs are temporally more stable.
    Returns mapping old_current_id -> new_id_in_previous_order.
    """
    k = len(current_centers)
    d = ((current_centers[:, None, :] - previous_centers[None, :, :]) ** 2).sum(axis=2)

    mapping = {}
    used_cur = set()
    used_prev = set()

    flat = []
    for i in range(k):
        for j in range(k):
            flat.append((float(d[i, j]), i, j))
    flat.sort()

    for _, i, j in flat:
        if i in used_cur or j in used_prev:
            continue
        mapping[i] = j
        used_cur.add(i)
        used_prev.add(j)

    for i in range(k):
        if i not in mapping:
            for j in range(k):
                if j not in used_prev:
                    mapping[i] = j
                    used_prev.add(j)
                    break

    return mapping


# -----------------------------------------------------------------------------
# Dynamic radius and dense assignment
# -----------------------------------------------------------------------------

def dynamic_seed_distance_map(mask, seeds_xy):
    """
    Distance from every pixel to nearest dynamic seed.
    """
    h, w = mask.shape
    if len(seeds_xy) == 0:
        return np.full((h, w), np.inf, dtype=np.float32)

    seed_img = np.zeros((h, w), dtype=np.uint8)
    for x, y in seeds_xy:
        xi = int(round(x))
        yi = int(round(y))
        if 0 <= xi < w and 0 <= yi < h:
            seed_img[yi, xi] = 255

    # distanceTransform measures distance to nearest zero pixel,
    # so invert seed image.
    inv = 255 - seed_img
    dist = cv2.distanceTransform(inv, cv2.DIST_L2, 5)
    return dist.astype(np.float32)


def assign_dense_kplus1(
    dense_features,
    fg_mask,
    dynamic_dist_map,
    dynamic_radius_px,
    dynamic_centers,
):
    """
    class 0 = foreground pixels farther than dynamic_radius_px from any seed
    class 1..K = nearby dynamic pixels assigned to nearest dynamic center
    """
    h, w = fg_mask.shape
    labels = np.zeros((h, w), dtype=np.uint8)  # 0 = static by default

    if dynamic_centers is None or len(dynamic_centers) == 0:
        return labels

    fg_idx = np.argwhere(fg_mask)
    fg_feat = dense_features[fg_mask]
    fg_dist = dynamic_dist_map[fg_mask]

    dynamic_fg = fg_dist <= dynamic_radius_px
    if not np.any(dynamic_fg):
        return labels

    dyn_feat = fg_feat[dynamic_fg]
    d2 = ((dyn_feat[:, None, :] - dynamic_centers[None, :, :]) ** 2).sum(axis=2)
    dyn_labels = np.argmin(d2, axis=1).astype(np.uint8) + 1  # shift by 1 because 0=static

    out_fg = np.zeros(len(fg_idx), dtype=np.uint8)
    out_fg[dynamic_fg] = dyn_labels

    labels[fg_mask] = out_fg
    return labels


def postprocess_masks(labels, fg_mask, min_component_px):
    """
    Remove tiny connected components inside dynamic classes only.
    Reassign removed pixels to static class 0.
    """
    labels = labels.copy()
    k = int(labels.max())

    for cls in range(1, k + 1):
        binary = ((labels == cls) & fg_mask).astype(np.uint8)
        if binary.sum() == 0:
            continue

        num_cc, cc_labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)

        keep = np.zeros_like(binary)
        for cc in range(1, num_cc):
            area = stats[cc, cv2.CC_STAT_AREA]
            if area >= min_component_px:
                keep[cc_labels == cc] = 1

        labels[(labels == cls) & (keep == 0)] = 0

    labels[~fg_mask] = 0
    return labels


# -----------------------------------------------------------------------------
# Rendering
# -----------------------------------------------------------------------------

def colorize_labels(labels, palette):
    out = np.zeros((labels.shape[0], labels.shape[1], 3), dtype=np.uint8)
    for cls in range(len(palette)):
        out[labels == cls] = palette[cls]
    return out


def overlay_labels(img_bgr, labels, fg_mask, palette, alpha):
    colored = colorize_labels(labels, palette)
    out = img_bgr.copy()
    blend = (
        alpha * colored.astype(np.float32)
        + (1.0 - alpha) * img_bgr.astype(np.float32)
    ).astype(np.uint8)
    out[fg_mask] = blend[fg_mask]
    return out


def draw_seed_points(img_bgr, seeds, cluster_ids=None):
    out = img_bgr.copy()

    for i, s in enumerate(seeds):
        x = int(round(s.x))
        y = int(round(s.y))

        if cluster_ids is None:
            color = (255, 255, 255)
            text = "d"
        else:
            cid = int(cluster_ids[i]) + 1
            color = tuple(int(v) for v in PALETTE[min(cid, len(PALETTE) - 1)])
            text = str(cid)

        cv2.circle(out, (x, y), 4, color, -1, cv2.LINE_AA)
        cv2.circle(out, (x, y), 6, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(
            out, text, (x + 5, y - 5),
            cv2.FONT_HERSHEY_SIMPLEX, 0.35, color, 1, cv2.LINE_AA
        )
    return out


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
# Main per-camera loop
# -----------------------------------------------------------------------------

def process_camera(
    root: Path,
    rgb_parent: str,
    camera: str,
    frame_paths,
    mask_map,
    seeds_by_cam_frame,
    out_root: Path,
    k: int,
    color_space: str,
    color_weight: float,
    xy_weight: float,
    patch_radius: int,
    dynamic_radius_px: float,
    min_seed_points_per_frame: int,
    min_component_px: int,
    overlay_alpha: float,
    attempts: int,
    fps: float,
    no_video: bool,
):
    cam_out = out_root / camera
    labels_dir = cam_out / "labels"
    overlay_dir = cam_out / "overlay"
    seeds_dir = cam_out / "seed_overlay"
    masks_root = cam_out / "masks"

    ensure_dir(labels_dir)
    ensure_dir(overlay_dir)
    ensure_dir(seeds_dir)
    ensure_dir(masks_root)
    for cls in range(k + 1):
        ensure_dir(masks_root / f"class_{cls}")

    palette = build_palette(k + 1)

    prev_centers = None
    rows = []

    overlay_frames = []
    seed_frames = []

    for img_path in frame_paths:
        fid = frame_id(img_path)
        if fid not in mask_map:
            continue

        img = read_image(img_path)
        fg_mask = read_mask(mask_map[fid], target_hw=img.shape[:2])
        h, w = fg_mask.shape

        dense_features, fg = build_dense_features(
            img, fg_mask, color_space, color_weight, xy_weight
        )

        seeds = seeds_by_cam_frame.get((camera, fid), [])
        valid_seeds = []
        seed_features = []

        for s in seeds:
            xi = int(round(s.x))
            yi = int(round(s.y))
            if not (0 <= xi < w and 0 <= yi < h):
                continue
            if not fg_mask[yi, xi]:
                continue
            feat = sample_patch_feature(dense_features, s.x, s.y, patch_radius)
            valid_seeds.append(s)
            seed_features.append(feat)

        seed_features = np.asarray(seed_features, dtype=np.float32) if seed_features else np.zeros((0, dense_features.shape[-1]), dtype=np.float32)

        have_clustering = len(valid_seeds) >= max(k, min_seed_points_per_frame)

        if have_clustering:
            labels_seed, centers, compactness = kmeans(
                seed_features,
                k=k,
                attempts=attempts,
                initial_centers=prev_centers,
            )

            if prev_centers is not None and len(prev_centers) == k:
                mapping = remap_labels_by_center_distance(centers, prev_centers)

                remapped_seed = labels_seed.copy()
                remapped_centers = centers.copy()

                for old_i, new_i in mapping.items():
                    remapped_seed[labels_seed == old_i] = new_i
                    remapped_centers[new_i] = centers[old_i]

                labels_seed = remapped_seed
                centers = remapped_centers

            prev_centers = centers.copy()

            seed_xy = [(s.x, s.y) for s in valid_seeds]
            dist_map = dynamic_seed_distance_map(fg_mask, seed_xy)

            labels = assign_dense_kplus1(
                dense_features=dense_features,
                fg_mask=fg_mask,
                dynamic_dist_map=dist_map,
                dynamic_radius_px=dynamic_radius_px,
                dynamic_centers=centers,
            )

            labels = postprocess_masks(
                labels,
                fg_mask=fg_mask,
                min_component_px=min_component_px,
            )
        else:
            compactness = float("nan")
            labels_seed = None
            labels = np.zeros((h, w), dtype=np.uint8)
            labels[~fg_mask] = 0

        # Save images
        color_overlay = overlay_labels(img, labels, fg_mask, palette, overlay_alpha)
        seed_overlay = draw_seed_points(img, valid_seeds, labels_seed)

        cv2.imwrite(str(labels_dir / f"frame_{fid:06d}.png"), labels)
        cv2.imwrite(str(overlay_dir / f"frame_{fid:06d}.png"), color_overlay)
        cv2.imwrite(str(seeds_dir / f"frame_{fid:06d}.png"), seed_overlay)

        for cls in range(k + 1):
            m = np.zeros((h, w), dtype=np.uint8)
            m[(labels == cls) & fg_mask] = 255
            cv2.imwrite(str((masks_root / f"class_{cls}" / f"frame_{fid:06d}.png")), m)

        overlay_frames.append(color_overlay)
        seed_frames.append(seed_overlay)

        fg_count = int(fg_mask.sum())
        dynamic_count = int(np.sum((labels > 0) & fg_mask))
        static_count = int(np.sum((labels == 0) & fg_mask))

        row = {
            "frame": fid,
            "foreground_pixels": fg_count,
            "num_seed_points": len(valid_seeds),
            "used_for_clustering": int(have_clustering),
            "dynamic_pixels_total": dynamic_count,
            "static_pixels_total": static_count,
            "compactness": compactness,
        }
        for cls in range(k + 1):
            count = int(np.sum((labels == cls) & fg_mask))
            row[f"class_{cls}_pixels"] = count
            row[f"class_{cls}_fraction"] = float(count / max(fg_count, 1))
        rows.append(row)

        print(
            f"[{camera}] frame {fid:06d} | seeds={len(valid_seeds):3d} "
            f"| static={static_count:6d} | dynamic={dynamic_count:6d}"
        )

    # CSV
    fields = [
        "frame",
        "foreground_pixels",
        "num_seed_points",
        "used_for_clustering",
        "dynamic_pixels_total",
        "static_pixels_total",
        "compactness",
    ]
    for cls in range(k + 1):
        fields.extend([f"class_{cls}_pixels", f"class_{cls}_fraction"])

    with open(cam_out / "stats.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for r in rows:
            writer.writerow(r)

    if not no_video:
        write_video(overlay_frames, cam_out / f"{camera}_overlay.mp4", fps)
        write_video(seed_frames, cam_out / f"{camera}_seed_overlay.mp4", fps)

    return rows


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("root", type=str)
    ap.add_argument("--tracks-csv", type=str, required=True,
                    help="temporal_track_observations.csv from the dynamic tracking pipeline.")
    ap.add_argument("--k", type=int, required=True,
                    help="Number of moving parts. Output classes are 0..K (K+1 masks total).")
    ap.add_argument("--only-global", action="store_true",
                    help="Use only multiview-confirmed dynamic tracks (global_track_id >= 0).")
    ap.add_argument("--color-space", type=str, default="lab", choices=["lab", "rgb", "hsv"])
    ap.add_argument("--color-weight", type=float, default=1.0)
    ap.add_argument("--xy-weight", type=float, default=0.50)
    ap.add_argument("--patch-radius", type=int, default=2,
                    help="Average feature over a small patch around each dynamic seed.")
    ap.add_argument("--dynamic-radius-px", type=float, default=20.0,
                    help="Foreground pixels farther than this from any dynamic seed stay static (class 0).")
    ap.add_argument("--min-seed-points-per-frame", type=int, default=2,
                    help="Minimum number of dynamic seed observations required before dense dynamic assignment.")
    ap.add_argument("--min-component-px", type=int, default=25,
                    help="Remove tiny dynamic connected components smaller than this.")
    ap.add_argument("--overlay-alpha", type=float, default=0.60)
    ap.add_argument("--attempts", type=int, default=5)
    ap.add_argument("--fps", type=float, default=10.0)
    ap.add_argument("--output", type=str, default=None)
    ap.add_argument("--no-video", action="store_true")
    args = ap.parse_args()

    if args.k < 1:
        raise ValueError("--k must be >= 1")

    root = Path(args.root)
    tracks_csv = Path(args.tracks_csv)

    rgb_parent, cameras = find_rgb_parent(root)

    mask_root = root / "mask"
    if not mask_root.is_dir():
        raise FileNotFoundError(f"Missing mask folder: {mask_root}")

    seeds_by_cam_frame = load_dynamic_seeds(tracks_csv, only_global=args.only_global)

    out_root = (
        Path(args.output)
        if args.output is not None
        else root / f"dynamic_kplus1_masks_k{args.k}"
    )
    ensure_dir(out_root)

    all_rows = []

    for camera in cameras:
        rgb_dir = root / rgb_parent / camera
        mask_dir = root / "mask" / camera

        frame_paths = list_frame_paths(rgb_dir)
        mask_paths = list_frame_paths(mask_dir)
        mask_map = {frame_id(p): p for p in mask_paths}

        rows = process_camera(
            root=root,
            rgb_parent=rgb_parent,
            camera=camera,
            frame_paths=frame_paths,
            mask_map=mask_map,
            seeds_by_cam_frame=seeds_by_cam_frame,
            out_root=out_root,
            k=args.k,
            color_space=args.color_space,
            color_weight=args.color_weight,
            xy_weight=args.xy_weight,
            patch_radius=args.patch_radius,
            dynamic_radius_px=args.dynamic_radius_px,
            min_seed_points_per_frame=args.min_seed_points_per_frame,
            min_component_px=args.min_component_px,
            overlay_alpha=args.overlay_alpha,
            attempts=args.attempts,
            fps=args.fps,
            no_video=args.no_video,
        )

        for r in rows:
            rr = {"camera": camera}
            rr.update(r)
            all_rows.append(rr)

    fields = ["camera",
              "frame",
              "foreground_pixels",
              "num_seed_points",
              "used_for_clustering",
              "dynamic_pixels_total",
              "static_pixels_total",
              "compactness"]
    for cls in range(args.k + 1):
        fields.extend([f"class_{cls}_pixels", f"class_{cls}_fraction"])

    with open(out_root / "summary.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for r in all_rows:
            writer.writerow(r)

    print("\n[DONE]")
    print(f"Output root: {out_root}")
    print(f"Summary:     {out_root / 'summary.csv'}")


if __name__ == "__main__":
    main()
