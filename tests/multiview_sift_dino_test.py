#!/usr/bin/env python3
"""
Multiview SIFT/RootSIFT correspondence test at ONE video frame, with optional
DINO consistency evaluated at the classical SIFT correspondences.

For each target camera:
  reference camera -> target camera
  RootSIFT
  Lowe ratio in both directions
  mutual nearest match
  robust fundamental-matrix filtering (USAC_MAGSAC when available)
  keep a spatially distributed subset
  sample DINO at the matched pixels
  report DINO cosine for the SAME candidate physical correspondence

The output is ONE montage panel containing all cameras and correspondence lines.

Example:
python multiview_sift_dino_test.py \
    /home/samuelemara/Joint3DGS/RigEuler3DGS/input/USB_100109 \
    --frame 20 \
    --reference-camera cam_000 \
    --matches-per-camera 8

Outputs:
ROOT/sift_dino_diagnostics/
  frame_000020_multiview_sift_dino.png
  frame_000020_multiview_sift_dino.csv
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
# Basic I/O
# -----------------------------------------------------------------------------

def list_cameras(root):
    for parent in ("rgb", "video_rgb"):
        p = root / parent
        if p.is_dir():
            cams = sorted([x.name for x in p.iterdir() if x.is_dir() and x.name.startswith("cam_")])
            if cams:
                return cams, parent
    raise FileNotFoundError("No rgb/cam_* or video_rgb/cam_* directories found.")


def find_frame(root, rgb_parent, camera, frame):
    folder = root / rgb_parent / camera
    for ext in ("png", "jpg", "jpeg"):
        p = folder / f"frame_{frame:06d}.{ext}"
        if p.exists():
            return p
    raise FileNotFoundError(f"Missing RGB frame {frame:06d} for {camera}")


def read_mask(root, camera, frame, image_hw):
    folder = root / "mask" / camera
    if not folder.is_dir():
        return None
    for ext in ("png", "jpg", "jpeg"):
        p = folder / f"frame_{frame:06d}.{ext}"
        if p.exists():
            m = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
            if m is None:
                return None
            if m.shape != image_hw:
                m = cv2.resize(m, (image_hw[1], image_hw[0]), interpolation=cv2.INTER_NEAREST)
            return (m > 0).astype(np.uint8) * 255
    return None


# -----------------------------------------------------------------------------
# RootSIFT
# -----------------------------------------------------------------------------

def rootsift(desc):
    if desc is None or len(desc) == 0:
        return None
    desc = desc.astype(np.float32)
    desc /= np.sum(desc, axis=1, keepdims=True) + 1e-12
    return np.sqrt(desc)


def create_sift(nfeatures):
    if not hasattr(cv2, "SIFT_create"):
        raise RuntimeError("This OpenCV build does not provide SIFT.")
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


def ratio_dict(desc_a, desc_b, ratio):
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


def mutual_matches(desc_a, desc_b, ratio):
    ab = ratio_dict(desc_a, desc_b, ratio)
    ba = ratio_dict(desc_b, desc_a, ratio)
    out = []
    for ia, (ib, da) in ab.items():
        rev = ba.get(ib)
        if rev is not None and rev[0] == ia:
            out.append((ia, ib, 0.5 * (da + rev[1])))
    return out


def geometric_filter(kp_a, kp_b, matches, ransac_px):
    if len(matches) < 8:
        return matches, None

    pts_a = np.float32([kp_a[a].pt for a, _, _ in matches])
    pts_b = np.float32([kp_b[b].pt for _, b, _ in matches])

    method = getattr(cv2, "USAC_MAGSAC", cv2.FM_RANSAC)

    Fmat, inlier = cv2.findFundamentalMat(
        pts_a,
        pts_b,
        method,
        ransac_px,
        0.999,
        10000,
    )

    if Fmat is None or inlier is None:
        return matches, None

    keep = inlier.reshape(-1).astype(bool)
    filtered = [m for m, k in zip(matches, keep) if k]
    return filtered, Fmat


def choose_spread_matches(kp_ref, matches, n_keep, image_hw):
    if len(matches) <= n_keep:
        return sorted(matches, key=lambda x: x[2])

    # Prefer descriptor quality but ensure spatial coverage in reference image.
    candidates = sorted(matches, key=lambda x: x[2])
    H, W = image_hw
    min_dist = 0.08 * math.sqrt(H * H + W * W)

    chosen = []
    for m in candidates:
        p = np.array(kp_ref[m[0]].pt, dtype=np.float32)
        if all(np.linalg.norm(p - np.array(kp_ref[c[0]].pt, dtype=np.float32)) >= min_dist for c in chosen):
            chosen.append(m)
            if len(chosen) >= n_keep:
                break

    if len(chosen) < n_keep:
        used = {(m[0], m[1]) for m in chosen}
        for m in candidates:
            if (m[0], m[1]) not in used:
                chosen.append(m)
                if len(chosen) >= n_keep:
                    break

    return chosen


# -----------------------------------------------------------------------------
# DINO sampling
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


def best_factor_pair(n, aspect):
    best = None
    for h in range(1, int(math.sqrt(n)) + 1):
        if n % h != 0:
            continue
        w = n // h
        for hh, ww in ((h, w), (w, h)):
            err = abs(ww / max(hh, 1) - aspect)
            if best is None or err < best[0]:
                best = (err, hh, ww)
    return None if best is None else (best[1], best[2])


def arrange_dino(feat, image_hw):
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
                candidates.append((abs(Wf / max(Hf, 1) - aspect), caxis, haxis, waxis))
        _, caxis, haxis, waxis = min(candidates, key=lambda x: x[0])
        return feat.permute(caxis, haxis, waxis).contiguous()

    if feat.ndim == 2:
        candidates = []
        for transposed in (False, True):
            x = feat.t().contiguous() if transposed else feat
            N, C = x.shape
            if C < 4:
                continue
            for nt, drop_cls in ((N, False), (N - 1, True)):
                if nt <= 0:
                    continue
                pair = best_factor_pair(nt, aspect)
                if pair is None:
                    continue
                Hf, Wf = pair
                candidates.append((abs(Wf / Hf - aspect), transposed, Hf, Wf, drop_cls))
        if not candidates:
            raise RuntimeError(f"Cannot infer DINO spatial shape from {tuple(feat.shape)}")
        _, transposed, Hf, Wf, drop_cls = min(candidates, key=lambda x: x[0])
        x = feat.t().contiguous() if transposed else feat
        if drop_cls:
            x = x[1:]
        C = x.shape[1]
        return x.reshape(Hf, Wf, C).permute(2, 0, 1).contiguous()

    raise RuntimeError(f"Unsupported DINO tensor shape {tuple(feat.shape)}")


def load_dino(path, image_hw):
    obj = torch.load(path, map_location="cpu")
    feat = find_tensor(obj)
    if feat is None:
        raise RuntimeError(f"No DINO tensor found: {path}")
    return F.normalize(arrange_dino(feat, image_hw), dim=0, eps=1e-8)


def sample_dino(fmap, x, y, image_hw):
    _, Hf, Wf = fmap.shape
    Himg, Wimg = image_hw

    xf = (x + 0.5) * Wf / Wimg - 0.5
    yf = (y + 0.5) * Hf / Himg - 0.5

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
# Panel visualization
# -----------------------------------------------------------------------------

def make_grid(images, cameras, columns):
    n = len(images)
    rows = int(math.ceil(n / columns))

    H = max(im.shape[0] for im in images)
    W = max(im.shape[1] for im in images)

    canvas = np.zeros((rows * H, columns * W, 3), dtype=np.uint8)
    offsets = {}

    for idx, (cam, im) in enumerate(zip(cameras, images)):
        r = idx // columns
        c = idx % columns
        y0 = r * H
        x0 = c * W

        if im.shape[:2] != (H, W):
            im = cv2.resize(im, (W, H), interpolation=cv2.INTER_LINEAR)

        canvas[y0:y0 + H, x0:x0 + W] = im
        offsets[cam] = (x0, y0)

        cv2.rectangle(canvas, (x0, y0), (x0 + W - 1, y0 + 30), (0, 0, 0), -1)
        cv2.putText(
            canvas, cam, (x0 + 10, y0 + 21),
            cv2.FONT_HERSHEY_SIMPLEX, 0.55,
            (255, 255, 255), 1, cv2.LINE_AA
        )

    return canvas, offsets, (H, W)


def match_color(pair_idx, local_idx):
    hue = int((pair_idx * 43 + local_idx * 137.5) % 180)
    hsv = np.uint8([[[hue, 220, 255]]])
    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0, 0]
    return tuple(int(x) for x in bgr)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("root")
    parser.add_argument("--frame", type=int, required=True)
    parser.add_argument("--reference-camera", default="cam_000")
    parser.add_argument("--matches-per-camera", type=int, default=8)
    parser.add_argument("--sift-nfeatures", type=int, default=6000)
    parser.add_argument("--sift-ratio", type=float, default=0.72)
    parser.add_argument("--ransac-px", type=float, default=1.5)
    parser.add_argument("--columns", type=int, default=4)
    parser.add_argument("--no-mask", action="store_true")
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    root = Path(args.root)
    cameras, rgb_parent = list_cameras(root)

    if args.reference_camera not in cameras:
        raise RuntimeError(f"Reference {args.reference_camera} not found in {cameras}")

    # Put reference first.
    cameras = [args.reference_camera] + [c for c in cameras if c != args.reference_camera]

    images = {}
    masks = {}
    dino = {}

    for cam in cameras:
        p = find_frame(root, rgb_parent, cam, args.frame)
        im = cv2.imread(str(p), cv2.IMREAD_COLOR)
        if im is None:
            raise RuntimeError(f"Could not read {p}")
        images[cam] = im

    H, W = images[args.reference_camera].shape[:2]
    image_hw = (H, W)

    for cam in cameras:
        if images[cam].shape[:2] != image_hw:
            images[cam] = cv2.resize(images[cam], (W, H), interpolation=cv2.INTER_LINEAR)

        masks[cam] = None if args.no_mask else read_mask(root, cam, args.frame, image_hw)

        dp = root / "dinov3" / cam / f"frame_{args.frame:06d}.pt"
        if dp.exists():
            dino[cam] = load_dino(dp, image_hw)
        else:
            dino[cam] = None

    sift = create_sift(args.sift_nfeatures)

    kps = {}
    descs = {}

    for cam in cameras:
        kps[cam], descs[cam] = extract_sift(sift, images[cam], masks[cam])
        print(f"[{cam}] {len(kps[cam])} SIFT keypoints")

    ref = args.reference_camera
    pair_results = []

    for target in cameras[1:]:
        raw = mutual_matches(descs[ref], descs[target], args.sift_ratio)
        geometric, Fmat = geometric_filter(
            kps[ref], kps[target], raw, args.ransac_px
        )
        selected = choose_spread_matches(
            kps[ref], geometric, args.matches_per_camera, image_hw
        )

        enriched = []
        for ia, ib, dist in selected:
            p_ref = kps[ref][ia].pt
            p_tgt = kps[target][ib].pt

            dino_cos = np.nan
            if dino[ref] is not None and dino[target] is not None:
                dr = sample_dino(dino[ref], p_ref[0], p_ref[1], image_hw)
                dt = sample_dino(dino[target], p_tgt[0], p_tgt[1], image_hw)
                dino_cos = float(torch.dot(dr, dt).item())

            enriched.append({
                "ia": ia,
                "ib": ib,
                "rootsift_distance": dist,
                "ref_x": float(p_ref[0]),
                "ref_y": float(p_ref[1]),
                "tgt_x": float(p_tgt[0]),
                "tgt_y": float(p_tgt[1]),
                "dino_cosine": dino_cos,
            })

        pair_results.append((target, len(raw), len(geometric), enriched))

        print(
            f"[{ref}->{target}] "
            f"mutual={len(raw)}, geometric={len(geometric)}, shown={len(enriched)}"
        )

    # Montage
    ordered_images = [images[c].copy() for c in cameras]
    canvas, offsets, tile_hw = make_grid(ordered_images, cameras, args.columns)
    tile_h, tile_w = tile_hw

    rows_out = []

    for pair_idx, (target, n_raw, n_geo, matches) in enumerate(pair_results):
        xref0, yref0 = offsets[ref]
        xtgt0, ytgt0 = offsets[target]

        for local_idx, m in enumerate(matches):
            color = match_color(pair_idx, local_idx)

            a = (
                int(round(xref0 + m["ref_x"])),
                int(round(yref0 + m["ref_y"])),
            )
            b = (
                int(round(xtgt0 + m["tgt_x"])),
                int(round(ytgt0 + m["tgt_y"])),
            )

            cv2.line(canvas, a, b, color, 1, cv2.LINE_AA)
            cv2.circle(canvas, a, 5, color, -1, cv2.LINE_AA)
            cv2.circle(canvas, b, 5, color, -1, cv2.LINE_AA)

            if np.isfinite(m["dino_cosine"]):
                cv2.putText(
                    canvas,
                    f"{m['dino_cosine']:.2f}",
                    (b[0] + 6, b[1] - 5),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.38,
                    color,
                    1,
                    cv2.LINE_AA,
                )

            rows_out.append({
                "frame": args.frame,
                "reference_camera": ref,
                "target_camera": target,
                "ref_x": m["ref_x"],
                "ref_y": m["ref_y"],
                "target_x": m["tgt_x"],
                "target_y": m["tgt_y"],
                "rootsift_distance": m["rootsift_distance"],
                "dino_cosine_at_sift_correspondence": m["dino_cosine"],
                "pair_mutual_match_count": n_raw,
                "pair_geometric_inlier_count": n_geo,
            })

    # Footer text.
    footer = 42
    out = np.zeros((canvas.shape[0] + footer, canvas.shape[1], 3), dtype=np.uint8)
    out[:canvas.shape[0]] = canvas

    valid_dino = [
        r["dino_cosine_at_sift_correspondence"]
        for r in rows_out
        if np.isfinite(r["dino_cosine_at_sift_correspondence"])
    ]

    summary = f"frame {args.frame:06d} | ref {ref} | {len(rows_out)} shown SIFT correspondences"
    if valid_dino:
        summary += (
            f" | DINO cosine on SIFT matches: "
            f"median={np.median(valid_dino):.3f}, "
            f"mean={np.mean(valid_dino):.3f}"
        )

    cv2.putText(
        out, summary, (12, canvas.shape[0] + 27),
        cv2.FONT_HERSHEY_SIMPLEX, 0.58,
        (255, 255, 255), 1, cv2.LINE_AA
    )

    output_dir = Path(args.output) if args.output else root / "sift_dino_diagnostics"
    output_dir.mkdir(parents=True, exist_ok=True)

    png_path = output_dir / f"frame_{args.frame:06d}_multiview_sift_dino.png"
    csv_path = output_dir / f"frame_{args.frame:06d}_multiview_sift_dino.csv"

    cv2.imwrite(str(png_path), out)

    fields = [
        "frame", "reference_camera", "target_camera",
        "ref_x", "ref_y", "target_x", "target_y",
        "rootsift_distance", "dino_cosine_at_sift_correspondence",
        "pair_mutual_match_count", "pair_geometric_inlier_count",
    ]

    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows_out)

    print("\n[DONE]")
    print(f"  Panel: {png_path}")
    print(f"  CSV:   {csv_path}")


if __name__ == "__main__":
    main()
