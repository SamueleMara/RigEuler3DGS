#!/usr/bin/env python3
"""
Temporal DINO consistency evaluated ALONG SIFT/RootSIFT trajectories.

This is the correct temporal test:

    cos( F_DINO(t0, p_t0), F_DINO(t, p_t) )

where p_t is the pixel reached by a classical SIFT track, rather than comparing
DINO at the same image coordinate over time.

Expected dataset:
ROOT/
  rgb/cam_000/frame_000000.png
  dinov3/cam_000/frame_000000.pt
  mask/cam_000/frame_000000.png       # optional

Example:
python test_dino_along_sift_tracks.py \
    /home/samuelemara/Joint3DGS/RigEuler3DGS/input/USB_100109 \
    --camera cam_000 \
    --num-tracks 30 \
    --min-track-length 15

Outputs:
ROOT/sift_dino_diagnostics/
  cam_000_dino_along_sift_tracks.csv
  cam_000_dino_along_sift_tracks.png
  cam_000_dino_along_sift_tracks.mp4
  cam_000_sift_tracks_for_dino_test.csv
"""

import argparse
import csv
import math
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F


# -----------------------------------------------------------------------------
# Dataset helpers
# -----------------------------------------------------------------------------

def frame_id(path):
    return int(path.stem.split("_")[-1])


def find_rgb_dir(root, camera):
    for parent in ("rgb", "video_rgb"):
        p = root / parent / camera
        if p.is_dir():
            return p
    raise FileNotFoundError(f"Could not find rgb/{camera} or video_rgb/{camera}")


def list_rgb_frames(folder):
    for ext in ("png", "jpg", "jpeg"):
        files = sorted(folder.glob(f"frame_*.{ext}"))
        if files:
            return files
    return []


def read_mask(mask_dir, fid, image_hw):
    if mask_dir is None:
        return None
    for ext in ("png", "jpg", "jpeg"):
        p = mask_dir / f"frame_{fid:06d}.{ext}"
        if p.exists():
            m = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
            if m is None:
                return None
            if m.shape != image_hw:
                m = cv2.resize(m, (image_hw[1], image_hw[0]), interpolation=cv2.INTER_NEAREST)
            return (m > 0).astype(np.uint8) * 255
    return None


# -----------------------------------------------------------------------------
# SIFT / RootSIFT tracks
# -----------------------------------------------------------------------------

def rootsift(desc):
    if desc is None or len(desc) == 0:
        return None
    desc = desc.astype(np.float32)
    desc /= np.sum(desc, axis=1, keepdims=True) + 1e-12
    return np.sqrt(desc)


def create_sift(nfeatures):
    if not hasattr(cv2, "SIFT_create"):
        raise RuntimeError("This OpenCV build does not provide cv2.SIFT_create().")
    return cv2.SIFT_create(
        nfeatures=nfeatures,
        contrastThreshold=0.01,
        edgeThreshold=20,
        sigma=1.6,
    )


def extract_sift(sift, image, mask):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    kp, desc = sift.detectAndCompute(gray, mask)
    return kp, rootsift(desc)


def ratio_matches(desc_a, desc_b, ratio):
    if desc_a is None or desc_b is None or len(desc_a) < 1 or len(desc_b) < 2:
        return {}
    matcher = cv2.BFMatcher(cv2.NORM_L2)
    out = {}
    for pair in matcher.knnMatch(desc_a, desc_b, k=2):
        if len(pair) != 2:
            continue
        m, n = pair
        if m.distance < ratio * n.distance:
            out[m.queryIdx] = (m.trainIdx, float(m.distance))
    return out


def mutual_ratio_matches(desc_a, desc_b, ratio):
    ab = ratio_matches(desc_a, desc_b, ratio)
    ba = ratio_matches(desc_b, desc_a, ratio)
    out = []
    for ia, (ib, dab) in ab.items():
        rev = ba.get(ib)
        if rev is not None and rev[0] == ia:
            out.append((ia, ib, 0.5 * (dab + rev[1])))
    return out


def build_tracks(frames, masks, nfeatures, ratio, min_length, max_tracks):
    sift = create_sift(nfeatures)
    all_kp, all_desc = [], []

    print("[SIFT] Extracting features...")
    for t, frame in enumerate(frames):
        kp, desc = extract_sift(sift, frame, masks[t])
        all_kp.append(kp)
        all_desc.append(desc)
        print(f"  frame {t:03d}: {len(kp)} keypoints")

    tracks = {}
    next_tid = 0
    prev_map = {}

    # Start frame-0 tracks.
    for i, kp in enumerate(all_kp[0]):
        tracks[next_tid] = [{
            "frame_index": 0,
            "x": float(kp.pt[0]),
            "y": float(kp.pt[1]),
            "rootsift_distance": np.nan,
        }]
        prev_map[i] = next_tid
        next_tid += 1

    for t in range(len(frames) - 1):
        matches = mutual_ratio_matches(all_desc[t], all_desc[t + 1], ratio)
        curr_map = {}

        for ia, ib, dist in matches:
            if ia not in prev_map:
                continue

            tid = prev_map[ia]
            tracks[tid][-1]["rootsift_distance"] = dist
            tracks[tid].append({
                "frame_index": t + 1,
                "x": float(all_kp[t + 1][ib].pt[0]),
                "y": float(all_kp[t + 1][ib].pt[1]),
                "rootsift_distance": np.nan,
            })
            curr_map[ib] = tid

        prev_map = curr_map
        print(f"[SIFT] {t:03d}->{t+1:03d}: {len(matches)} mutual matches, "
              f"{len(curr_map)} continuing frame-0 tracks")

    # Strict test: only trajectories beginning at frame 0.
    candidates = [
        (tid, tr) for tid, tr in tracks.items()
        if tr[0]["frame_index"] == 0 and len(tr) >= min_length
    ]

    def quality(item):
        _, tr = item
        distances = [p["rootsift_distance"] for p in tr if np.isfinite(p["rootsift_distance"])]
        mean_d = float(np.mean(distances)) if distances else 1e9
        return (-len(tr), mean_d)

    candidates.sort(key=quality)
    candidates = candidates[:max_tracks]

    result = {new_tid: tr for new_tid, (_, tr) in enumerate(candidates)}
    print(f"[SIFT] Keeping {len(result)} frame-0 tracks with length >= {min_length}.")
    return result


# -----------------------------------------------------------------------------
# DINO loading
# -----------------------------------------------------------------------------

def find_tensor(obj):
    if torch.is_tensor(obj):
        return obj

    if isinstance(obj, dict):
        preferred = (
            "features", "feature", "feat", "feats",
            "patch_tokens", "x_norm_patchtokens",
            "x_prenorm", "dense_features",
        )
        for k in preferred:
            if k in obj:
                t = find_tensor(obj[k])
                if t is not None:
                    return t

        candidates = []
        for v in obj.values():
            t = find_tensor(v)
            if t is not None and t.is_floating_point():
                candidates.append(t)
        if candidates:
            candidates.sort(key=lambda x: x.numel(), reverse=True)
            return candidates[0]

    if isinstance(obj, (list, tuple)):
        candidates = []
        for v in obj:
            t = find_tensor(v)
            if t is not None and t.is_floating_point():
                candidates.append(t)
        if candidates:
            candidates.sort(key=lambda x: x.numel(), reverse=True)
            return candidates[0]

    return None


def best_factor_pair(n, target_aspect):
    best = None
    for h in range(1, int(math.sqrt(n)) + 1):
        if n % h != 0:
            continue
        w = n // h
        for hh, ww in ((h, w), (w, h)):
            err = abs(ww / max(hh, 1) - target_aspect)
            if best is None or err < best[0]:
                best = (err, hh, ww)
    return None if best is None else (best[1], best[2])


def arrange_dense_feature_tensor(feat, image_hw):
    feat = feat.detach().float().cpu()

    while feat.ndim > 3 and feat.shape[0] == 1:
        feat = feat.squeeze(0)

    Himg, Wimg = image_hw
    aspect = Wimg / Himg

    if feat.ndim == 3:
        shape = tuple(feat.shape)
        candidates = []
        for caxis in range(3):
            spatial = [a for a in range(3) if a != caxis]
            for haxis, waxis in (spatial, spatial[::-1]):
                Hf, Wf = shape[haxis], shape[waxis]
                err = abs(Wf / max(Hf, 1) - aspect)
                candidates.append((err, caxis, haxis, waxis))
        _, caxis, haxis, waxis = min(candidates, key=lambda x: x[0])
        return feat.permute(caxis, haxis, waxis).contiguous()

    if feat.ndim == 2:
        candidates = []
        for transposed in (False, True):
            x = feat.t().contiguous() if transposed else feat
            N, C = x.shape
            if C < 4:
                continue

            for n_tokens, drop_cls in ((N, False), (N - 1, True)):
                if n_tokens <= 0:
                    continue
                pair = best_factor_pair(n_tokens, aspect)
                if pair is None:
                    continue
                Hf, Wf = pair
                err = abs(Wf / Hf - aspect)
                candidates.append((err, transposed, Hf, Wf, drop_cls))

        if not candidates:
            raise RuntimeError(f"Cannot infer DINO spatial layout from {tuple(feat.shape)}.")

        _, transposed, Hf, Wf, drop_cls = min(candidates, key=lambda x: x[0])
        x = feat.t().contiguous() if transposed else feat
        if drop_cls:
            x = x[1:]
        C = x.shape[1]
        return x.reshape(Hf, Wf, C).permute(2, 0, 1).contiguous()

    raise RuntimeError(f"Unsupported DINO tensor shape {tuple(feat.shape)}.")


def load_dino(path, image_hw):
    obj = torch.load(path, map_location="cpu")
    feat = find_tensor(obj)
    if feat is None:
        raise RuntimeError(f"No DINO tensor found in {path}")
    fmap = arrange_dense_feature_tensor(feat, image_hw)
    return F.normalize(fmap, dim=0, eps=1e-8)


def image_to_feature_xy(x, y, image_hw, feature_hw):
    Himg, Wimg = image_hw
    Hf, Wf = feature_hw
    xf = (x + 0.5) * Wf / Wimg - 0.5
    yf = (y + 0.5) * Hf / Himg - 0.5
    return xf, yf


def sample_dino(fmap, x, y, image_hw):
    _, Hf, Wf = fmap.shape
    xf, yf = image_to_feature_xy(x, y, image_hw, (Hf, Wf))

    gx = 2.0 * xf / max(Wf - 1, 1) - 1.0
    gy = 2.0 * yf / max(Hf - 1, 1) - 1.0

    grid = torch.tensor([[[[gx, gy]]]], dtype=fmap.dtype)
    d = F.grid_sample(
        fmap.unsqueeze(0),
        grid,
        mode="bilinear",
        padding_mode="border",
        align_corners=True,
    )[0, :, 0, 0]

    return F.normalize(d, dim=0, eps=1e-8)


# -----------------------------------------------------------------------------
# DINO along trajectories
# -----------------------------------------------------------------------------

def evaluate_dino_along_tracks(tracks, dino_paths, image_hw):
    maps = {}
    rows = []

    def get_map(t):
        if t not in maps:
            maps[t] = load_dino(dino_paths[t], image_hw)
        return maps[t]

    for tid, tr in tracks.items():
        p0 = tr[0]
        d0 = sample_dino(get_map(0), p0["x"], p0["y"], image_hw)
        prev_d = d0

        for j, p in enumerate(tr):
            t = p["frame_index"]
            dt = sample_dino(get_map(t), p["x"], p["y"], image_hw)

            cos_start = float(torch.dot(d0, dt).item())
            cos_prev = float(torch.dot(prev_d, dt).item()) if j > 0 else 1.0

            rows.append({
                "track_id": tid,
                "frame_index": t,
                "x": p["x"],
                "y": p["y"],
                "dino_cos_to_track_start": cos_start,
                "dino_cos_to_previous_track_point": cos_prev,
                "rootsift_distance_to_next": p["rootsift_distance"],
            })

            prev_d = dt

    return rows


def save_rows(rows, path):
    fields = [
        "track_id", "frame_index", "x", "y",
        "dino_cos_to_track_start",
        "dino_cos_to_previous_track_point",
        "rootsift_distance_to_next",
    ]
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def save_sift_tracks_csv(tracks, frame_ids, path):
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["track_id", "frame_index", "frame_id", "x", "y", "rootsift_distance_to_next"])
        for tid, tr in tracks.items():
            for p in tr:
                d = p["rootsift_distance"]
                w.writerow([
                    tid,
                    p["frame_index"],
                    frame_ids[p["frame_index"]],
                    f"{p['x']:.4f}",
                    f"{p['y']:.4f}",
                    "" if not np.isfinite(d) else f"{d:.6f}",
                ])


def plot_dino_track_consistency(rows, output_path):
    import matplotlib.pyplot as plt

    by_track = {}
    for r in rows:
        by_track.setdefault(r["track_id"], []).append(r)

    max_frame = max(r["frame_index"] for r in rows)
    aggregate = {t: [] for t in range(max_frame + 1)}

    plt.figure(figsize=(11, 6))

    # Light individual trajectories.
    for tid, tr in sorted(by_track.items()):
        tr = sorted(tr, key=lambda x: x["frame_index"])
        x = [r["frame_index"] for r in tr]
        y = [r["dino_cos_to_track_start"] for r in tr]
        plt.plot(x, y, linewidth=0.8, alpha=0.28)
        for r in tr:
            aggregate[r["frame_index"]].append(r["dino_cos_to_track_start"])

    frames = []
    med = []
    q25 = []
    q75 = []

    for t in sorted(aggregate):
        vals = aggregate[t]
        if not vals:
            continue
        frames.append(t)
        med.append(float(np.median(vals)))
        q25.append(float(np.percentile(vals, 25)))
        q75.append(float(np.percentile(vals, 75)))

    plt.plot(frames, med, linewidth=2.8, label="Median DINO cosine along SIFT tracks")
    plt.fill_between(frames, q25, q75, alpha=0.18, label="25–75 percentile")
    plt.xlabel("Frame")
    plt.ylabel("DINO cosine to same SIFT-tracked point at frame 0")
    plt.ylim(-0.05, 1.05)
    plt.grid(alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=170)
    plt.close()


def color_from_similarity(sim):
    """
    BGR color: low similarity -> red, high -> green.
    """
    s = float(np.clip((sim - 0.5) / 0.5, 0.0, 1.0))
    return (0, int(255 * s), int(255 * (1.0 - s)))


def render_video(frames, frame_ids, tracks, rows, output_path, fps):
    H, W = frames[0].shape[:2]
    writer = cv2.VideoWriter(
        str(output_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (W, H),
    )
    if not writer.isOpened():
        raise RuntimeError(f"Could not open video writer: {output_path}")

    row_map = {(r["track_id"], r["frame_index"]): r for r in rows}

    for t, frame in enumerate(frames):
        out = frame.copy()

        for tid, tr in tracks.items():
            visible = [p for p in tr if p["frame_index"] <= t]
            if not visible:
                continue

            pts = np.array(
                [[int(round(p["x"])), int(round(p["y"]))] for p in visible],
                dtype=np.int32,
            )
            if len(pts) >= 2:
                cv2.polylines(out, [pts], False, (230, 230, 230), 1, cv2.LINE_AA)

            current = next((p for p in tr if p["frame_index"] == t), None)
            if current is None:
                continue

            r = row_map[(tid, t)]
            sim = r["dino_cos_to_track_start"]
            color = color_from_similarity(sim)

            x = int(round(current["x"]))
            y = int(round(current["y"]))
            cv2.circle(out, (x, y), 6, color, -1, cv2.LINE_AA)
            cv2.putText(
                out,
                f"{tid}:{sim:.2f}",
                (x + 7, y - 7),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.42,
                color,
                1,
                cv2.LINE_AA,
            )

        cv2.putText(
            out,
            f"DINO along SIFT tracks | frame {frame_ids[t]:06d}",
            (12, 27),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

        writer.write(out)

    writer.release()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("root")
    parser.add_argument("--camera", default="cam_000")
    parser.add_argument("--num-tracks", type=int, default=30)
    parser.add_argument("--sift-nfeatures", type=int, default=5000)
    parser.add_argument("--sift-ratio", type=float, default=0.72)
    parser.add_argument("--min-track-length", type=int, default=15)
    parser.add_argument("--fps", type=float, default=10.0)
    parser.add_argument("--no-mask", action="store_true")
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    root = Path(args.root)
    rgb_dir = find_rgb_dir(root, args.camera)
    dino_dir = root / "dinov3" / args.camera

    rgb_files = list_rgb_frames(rgb_dir)
    pairs = []

    for rgb_path in rgb_files:
        fid = frame_id(rgb_path)
        dp = dino_dir / f"frame_{fid:06d}.pt"
        if dp.exists():
            pairs.append((fid, rgb_path, dp))

    if not pairs:
        raise RuntimeError("No matched RGB/DINO frames.")

    frame_ids = [x[0] for x in pairs]
    dino_paths = [x[2] for x in pairs]
    frames = [cv2.imread(str(x[1]), cv2.IMREAD_COLOR) for x in pairs]

    if any(f is None for f in frames):
        raise RuntimeError("Could not read one or more RGB frames.")

    H, W = frames[0].shape[:2]
    image_hw = (H, W)

    for i, f in enumerate(frames):
        if f.shape[:2] != image_hw:
            frames[i] = cv2.resize(f, (W, H), interpolation=cv2.INTER_LINEAR)

    mask_dir = root / "mask" / args.camera
    if args.no_mask or not mask_dir.is_dir():
        mask_dir = None

    masks = [read_mask(mask_dir, fid, image_hw) for fid in frame_ids]

    output = Path(args.output) if args.output else root / "sift_dino_diagnostics"
    output.mkdir(parents=True, exist_ok=True)

    tracks = build_tracks(
        frames,
        masks,
        args.sift_nfeatures,
        args.sift_ratio,
        args.min_track_length,
        args.num_tracks,
    )

    if not tracks:
        raise RuntimeError(
            "No sufficiently long frame-0 SIFT tracks. "
            "Try --min-track-length 8 or --sift-ratio 0.78."
        )

    rows = evaluate_dino_along_tracks(tracks, dino_paths, image_hw)

    csv_path = output / f"{args.camera}_dino_along_sift_tracks.csv"
    plot_path = output / f"{args.camera}_dino_along_sift_tracks.png"
    video_path = output / f"{args.camera}_dino_along_sift_tracks.mp4"
    sift_csv = output / f"{args.camera}_sift_tracks_for_dino_test.csv"

    save_rows(rows, csv_path)
    save_sift_tracks_csv(tracks, frame_ids, sift_csv)
    plot_dino_track_consistency(rows, plot_path)
    render_video(frames, frame_ids, tracks, rows, video_path, args.fps)

    print("\n[DONE]")
    print(f"  DINO/SIFT CSV:   {csv_path}")
    print(f"  DINO/SIFT plot:  {plot_path}")
    print(f"  DINO/SIFT video: {video_path}")
    print(f"  SIFT track CSV:  {sift_csv}")


if __name__ == "__main__":
    main()
