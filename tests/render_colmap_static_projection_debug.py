#!/usr/bin/env python3
"""
Render visual debug for an EXISTING colmap_static_projection_verify run.

No COLMAP rerun is needed.

Example:
python render_colmap_static_projection_debug.py \
    /home/samuelemara/Joint3DGS/RigEuler3DGS/input/USB_100109

Default result directory:
ROOT/colmap_static_projection_verify

Outputs:
ROOT/colmap_static_projection_verify/visual_debug/
    verification_timeline.png
    all_cameras_verification.mp4
    cam_000_verification.mp4 ... cam_007_verification.mp4
    frame_XXXXXX_panel.png

Visual convention:
  white X       = projection of initial triangulated anchor X0 (STATIC hypothesis)
  filled circle = current SAME-CAMERA temporal source observation
  square        = current SAME-FRAME multiview SIFT observation in a target camera
  vector        = static projection -> observed feature
  white ring    = dynamic candidate survived COLMAP triangulation/BA/filtering

This renderer also applies the corrected interpretation when visualizing old runs:
cross-view disagreement CANNOT create dynamics if the source temporal track
itself is static.
"""

import argparse
import csv
import math
import shutil
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np


COLORS = {
    "STATIC": (70, 220, 70),
    "DYNAMIC_CANDIDATE": (0, 165, 255),
    "DYNAMIC_VERIFIED": (40, 40, 255),
    "MISTRACK": (255, 0, 255),
    "CROSSVIEW_MISMATCH": (255, 255, 0),
    "UNCERTAIN_IDENTITY": (0, 215, 255),
    "UNCERTAIN": (0, 215, 255),
    "UNSUPPORTED": (150, 150, 150),
}

FINAL_COLORS = {
    "STATIC": (70, 220, 70),
    "DYNAMIC": (40, 40, 255),
    "MISTRACK": (255, 0, 255),
    "UNCERTAIN": (0, 215, 255),
}


def read_csv(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def as_int(v, default=0):
    try:
        return int(float(v))
    except Exception:
        return default


def as_float(v, default=float("nan")):
    try:
        return float(v)
    except Exception:
        return default


def find_rgb_parent(root):
    for name in ("rgb", "video_rgb"):
        p = root / name
        if p.is_dir() and any(x.is_dir() and x.name.startswith("cam_") for x in p.iterdir()):
            return p
    raise FileNotFoundError("Could not find ROOT/rgb/cam_* or ROOT/video_rgb/cam_*")


def find_frame(image_path, cam, frame):
    for ext in (".png", ".jpg", ".jpeg", ".PNG", ".JPG", ".JPEG"):
        p = image_path / cam / f"frame_{frame:06d}{ext}"
        if p.exists():
            return p
    return None


def draw_cross(img, xy, color, size=6, thickness=1):
    x, y = int(round(xy[0])), int(round(xy[1]))
    cv2.line(img, (x - size, y), (x + size, y), color, thickness, cv2.LINE_AA)
    cv2.line(img, (x, y - size), (x, y + size), color, thickness, cv2.LINE_AA)


def put_text_box(img, text, origin, color=(255, 255, 255), scale=0.42):
    x, y = int(origin[0]), int(origin[1])
    (tw, th), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, 1)
    x = max(1, min(x, max(1, img.shape[1] - tw - 3)))
    y = max(th + 3, min(y, max(th + 3, img.shape[0] - baseline - 2)))
    cv2.rectangle(
        img,
        (x - 2, y - th - 3),
        (x + tw + 2, y + baseline + 2),
        (0, 0, 0),
        -1,
    )
    cv2.putText(
        img, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX,
        scale, color, 1, cv2.LINE_AA
    )


def corrected_label(r, static_px=1.0, dynamic_px=2.0, min_dynamic_cameras=2):
    n_targets = as_int(r["num_crossview_targets"])
    n_static_targets = as_int(r["num_static_targets"])
    n_dynamic_targets = as_int(r["num_dynamic_targets"])
    source_err = as_float(r["source_static_error_px"])
    direct_identity = bool(as_int(r["direct_reference_sift_match"]))

    source_static = source_err <= static_px
    source_dynamic = source_err >= dynamic_px
    required_dynamic_targets = max(1, min_dynamic_cameras - 1)

    if n_targets < 1:
        return "UNSUPPORTED"
    if source_static and n_static_targets == n_targets:
        return "STATIC"
    if source_dynamic and n_dynamic_targets >= required_dynamic_targets and direct_identity:
        return "DYNAMIC_CANDIDATE"
    if source_dynamic and n_dynamic_targets >= required_dynamic_targets and not direct_identity:
        return "UNCERTAIN_IDENTITY"
    if source_dynamic and n_static_targets == n_targets:
        return "MISTRACK"
    if source_static and n_dynamic_targets > 0:
        return "CROSSVIEW_MISMATCH"
    return "UNCERTAIN"


def color_for(r):
    if as_int(r.get("geometry_ok", 0)) and r["_debug_label"] == "DYNAMIC_CANDIDATE":
        return COLORS["DYNAMIC_VERIFIED"]
    return COLORS.get(r["_debug_label"], (220, 220, 220))


def short_label(r):
    if as_int(r.get("geometry_ok", 0)) and r["_debug_label"] == "DYNAMIC_CANDIDATE":
        return "DYN-G"
    return {
        "STATIC": "S",
        "DYNAMIC_CANDIDATE": "D?",
        "MISTRACK": "MIS",
        "CROSSVIEW_MISMATCH": "XMV",
        "UNCERTAIN_IDENTITY": "UID",
        "UNCERTAIN": "U",
        "UNSUPPORTED": "-",
    }.get(r["_debug_label"], r["_debug_label"][:4])


def collect_static_uv(frame_rows, target_rows):
    static_uv = {}
    for r in frame_rows:
        static_uv[(as_int(r["anchor_id"]), r["source_cam"])] = (
            as_float(r["static_source_x"]),
            as_float(r["static_source_y"]),
        )
    for r in target_rows:
        static_uv[(as_int(r["anchor_id"]), r["target_cam"])] = (
            as_float(r["static_target_x"]),
            as_float(r["static_target_y"]),
        )
    return static_uv


def load_run(result_dir, static_px, dynamic_px, min_dynamic_cameras):
    frame_rows = read_csv(result_dir / "static_projection_frame_evidence.csv")
    target_rows = read_csv(result_dir / "static_projection_target_evidence.csv")
    track_rows = read_csv(result_dir / "verified_temporal_tracks.csv")
    global_rows = read_csv(result_dir / "verified_global_anchors.csv")

    for r in frame_rows:
        r["_debug_label"] = corrected_label(
            r, static_px, dynamic_px, min_dynamic_cameras
        )

    source_lookup = {
        (
            as_int(r["anchor_id"]),
            r["source_cam"],
            as_int(r["source_track"]),
            as_int(r["frame"]),
        ): r
        for r in frame_rows
    }

    for tr in target_rows:
        p = source_lookup.get(
            (
                as_int(tr["anchor_id"]),
                tr["source_cam"],
                as_int(tr["source_track"]),
                as_int(tr["frame"]),
            )
        )
        tr["_parent"] = p

    return frame_rows, target_rows, track_rows, global_rows


def summarize_corrected_tracks(frame_rows, min_dynamic_frames=3, min_static_frames=5, static_fraction=0.8):
    grouped = defaultdict(list)
    for r in frame_rows:
        grouped[(as_int(r["anchor_id"]), r["source_cam"], as_int(r["source_track"]))].append(r)

    out = {}
    for key, rows in grouped.items():
        supported = [r for r in rows if r["_debug_label"] != "UNSUPPORTED"]
        n_static = sum(r["_debug_label"] == "STATIC" for r in rows)
        n_dyn_geom = sum(
            r["_debug_label"] == "DYNAMIC_CANDIDATE" and as_int(r["geometry_ok"]) == 1
            for r in rows
        )
        n_mis = sum(r["_debug_label"] == "MISTRACK" for r in rows)
        frac = n_static / max(len(supported), 1)
        if n_dyn_geom >= min_dynamic_frames:
            label = "DYNAMIC"
        elif n_static >= min_static_frames and frac >= static_fraction:
            label = "STATIC"
        elif n_mis >= min_dynamic_frames and n_dyn_geom == 0:
            label = "MISTRACK"
        else:
            label = "UNCERTAIN"
        out[key] = label
    return out


def render_camera_frame(
    image_path, cam, frame, frame_source, frame_target,
    source_history, static_uv, corrected_track_labels, trail_frames
):
    p = find_frame(image_path, cam, frame)
    if p is None:
        img = np.zeros((480, 640, 3), dtype=np.uint8)
        put_text_box(img, f"MISSING {cam} frame {frame}", (10, 25), (0, 0, 255), 0.55)
        return img

    img = cv2.imread(str(p), cv2.IMREAD_COLOR)
    if img is None:
        return np.zeros((480, 640, 3), dtype=np.uint8)

    # All static hypotheses visible in this camera.
    for (aid, c), uv in static_uv.items():
        if c == cam and np.isfinite(uv[0]) and np.isfinite(uv[1]):
            draw_cross(img, uv, (180, 180, 180), size=4, thickness=1)

    src_rows = frame_source.get((cam, frame), [])
    for r in src_rows:
        aid = as_int(r["anchor_id"])
        tid = as_int(r["source_track"])
        color = color_for(r)

        sxy = (as_float(r["static_source_x"]), as_float(r["static_source_y"]))
        cxy = (as_float(r["source_x"]), as_float(r["source_y"]))

        hist = source_history.get((aid, cam, tid), [])
        pts = []
        for h in hist:
            hf = as_int(h["frame"])
            if frame - trail_frames <= hf <= frame:
                pts.append(
                    (int(round(as_float(h["source_x"]))), int(round(as_float(h["source_y"]))))
                )
        if len(pts) >= 2:
            cv2.polylines(img, [np.asarray(pts, np.int32)], False, color, 1, cv2.LINE_AA)

        draw_cross(img, sxy, (255, 255, 255), 7, 1)
        sx, sy = int(round(sxy[0])), int(round(sxy[1]))
        cx, cy = int(round(cxy[0])), int(round(cxy[1]))
        cv2.line(img, (sx, sy), (cx, cy), color, 2, cv2.LINE_AA)
        cv2.circle(img, (cx, cy), 5, color, -1, cv2.LINE_AA)

        if as_int(r["geometry_ok"]) and r["_debug_label"] == "DYNAMIC_CANDIDATE":
            cv2.circle(img, (cx, cy), 9, (255, 255, 255), 2, cv2.LINE_AA)
        if not as_int(r["direct_reference_sift_match"]):
            cv2.circle(img, (cx, cy), 12, (0, 0, 0), 2, cv2.LINE_AA)

        tlabel = corrected_track_labels.get((aid, cam, tid), "?")
        txt = (
            f"A{aid}/T{tid} {short_label(r)} [{tlabel}] "
            f"eS={as_float(r['source_static_error_px']):.1f}px"
        )
        put_text_box(img, txt, (cx + 7, cy - 7), color, 0.36)

    # Target-camera same-frame SIFT matches.
    for tr in frame_target.get((cam, frame), []):
        parent = tr["_parent"]
        if parent is None:
            continue
        color = color_for(parent)
        sxy = (as_float(tr["static_target_x"]), as_float(tr["static_target_y"]))
        cxy = (as_float(tr["target_x"]), as_float(tr["target_y"]))
        draw_cross(img, sxy, (255, 255, 255), 6, 1)
        sx, sy = int(round(sxy[0])), int(round(sxy[1]))
        cx, cy = int(round(cxy[0])), int(round(cxy[1]))
        cv2.line(img, (sx, sy), (cx, cy), color, 1, cv2.LINE_AA)
        cv2.rectangle(img, (cx - 4, cy - 4), (cx + 4, cy + 4), color, 2)

        err = as_float(tr["target_static_error_px"])
        if err >= 2.0:
            put_text_box(
                img,
                f"A{as_int(tr['anchor_id'])} MV={err:.1f}px",
                (cx + 5, cy + 13),
                color,
                0.30,
            )

    counts = defaultdict(int)
    for r in src_rows:
        key = (
            "DYNAMIC_VERIFIED"
            if as_int(r["geometry_ok"]) and r["_debug_label"] == "DYNAMIC_CANDIDATE"
            else r["_debug_label"]
        )
        counts[key] += 1

    header = (
        f"{cam} frame {frame:06d} "
        f"S:{counts['STATIC']} Dg:{counts['DYNAMIC_VERIFIED']} "
        f"D?:{counts['DYNAMIC_CANDIDATE']} Xmv:{counts['CROSSVIEW_MISMATCH']} "
        f"Mis:{counts['MISTRACK']} U:{counts['UNCERTAIN'] + counts['UNCERTAIN_IDENTITY']}"
    )
    put_text_box(img, header, (8, 22), (255, 255, 255), 0.45)
    return img


def render_timeline(debug_dir, frame_rows, corrected_track_labels, max_frame):
    grouped = defaultdict(dict)
    for r in frame_rows:
        key = (as_int(r["anchor_id"]), r["source_cam"], as_int(r["source_track"]))
        grouped[key][as_int(r["frame"])] = r

    keys = sorted(grouped)
    left, cell_w, row_h, top = 290, 12, 22, 45
    width = left + (max_frame + 1) * cell_w + 20
    height = top + len(keys) * row_h + 60
    img = np.zeros((height, width, 3), dtype=np.uint8)

    cv2.putText(
        img,
        "Corrected verification timeline: white dot = COLMAP dynamic geometry survived",
        (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 255, 255), 1, cv2.LINE_AA
    )

    for f in range(max_frame + 1):
        if f % 5 == 0:
            x = left + f * cell_w
            cv2.putText(
                img, str(f), (x, 40), cv2.FONT_HERSHEY_SIMPLEX,
                0.32, (190, 190, 190), 1, cv2.LINE_AA
            )

    for ri, key in enumerate(keys):
        aid, cam, tid = key
        y0 = top + ri * row_h
        final = corrected_track_labels.get(key, "?")
        cv2.putText(
            img, f"A{aid:02d} {cam} T{tid:02d} {final}",
            (6, y0 + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.38,
            FINAL_COLORS.get(final, (210, 210, 210)), 1, cv2.LINE_AA
        )
        for f in range(max_frame + 1):
            x0 = left + f * cell_w
            r = grouped[key].get(f)
            if r is None:
                c, geom = (25, 25, 25), False
            else:
                c = color_for(r)
                geom = (
                    as_int(r["geometry_ok"])
                    and r["_debug_label"] == "DYNAMIC_CANDIDATE"
                )
            cv2.rectangle(
                img, (x0, y0 + 2), (x0 + cell_w - 2, y0 + row_h - 3), c, -1
            )
            if geom:
                cv2.circle(
                    img, (x0 + 5, y0 + row_h // 2), 2,
                    (255, 255, 255), -1, cv2.LINE_AA
                )

    legend_y = height - 35
    x = 8
    for label in (
        "STATIC", "DYNAMIC_CANDIDATE", "DYNAMIC_VERIFIED",
        "MISTRACK", "CROSSVIEW_MISMATCH", "UNCERTAIN", "UNSUPPORTED"
    ):
        c = COLORS[label]
        cv2.rectangle(img, (x, legend_y), (x + 14, legend_y + 14), c, -1)
        cv2.putText(
            img, label, (x + 19, legend_y + 12),
            cv2.FONT_HERSHEY_SIMPLEX, 0.33, (230, 230, 230), 1, cv2.LINE_AA
        )
        x += 24 + 7 * len(label)

    p = debug_dir / "verification_timeline.png"
    cv2.imwrite(str(p), img)
    return p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("root")
    ap.add_argument("--result-dir", default=None)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--fps", type=float, default=10.0)
    ap.add_argument("--trail-frames", type=int, default=10)
    ap.add_argument("--panel-tile-width", type=int, default=320)
    ap.add_argument("--snapshots", default="0,10,20,30,40,50,59")
    ap.add_argument("--no-per-camera-videos", action="store_true")

    # These MUST match the verification run for faithful re-interpretation.
    ap.add_argument("--static-px", type=float, default=1.0)
    ap.add_argument("--dynamic-px", type=float, default=2.0)
    ap.add_argument("--min-dynamic-cameras", type=int, default=2)
    ap.add_argument("--min-dynamic-frames", type=int, default=3)
    ap.add_argument("--min-static-frames", type=int, default=5)
    ap.add_argument("--static-frame-fraction", type=float, default=0.8)
    args = ap.parse_args()

    root = Path(args.root)
    image_path = find_rgb_parent(root)
    result_dir = (
        Path(args.result_dir)
        if args.result_dir
        else root / "colmap_static_projection_verify"
    )

    required = [
        "static_projection_frame_evidence.csv",
        "static_projection_target_evidence.csv",
        "verified_temporal_tracks.csv",
        "verified_global_anchors.csv",
    ]
    for name in required:
        p = result_dir / name
        if not p.exists():
            raise FileNotFoundError(p)

    frame_rows, target_rows, _, _ = load_run(
        result_dir, args.static_px, args.dynamic_px, args.min_dynamic_cameras
    )

    corrected_track_labels = summarize_corrected_tracks(
        frame_rows,
        min_dynamic_frames=args.min_dynamic_frames,
        min_static_frames=args.min_static_frames,
        static_fraction=args.static_frame_fraction,
    )

    original_counts = defaultdict(int)
    for r in frame_rows:
        original_counts[r["frame_label"]] += 1
    corrected_counts = defaultdict(int)
    for r in frame_rows:
        corrected_counts[r["_debug_label"]] += 1

    print("[DEBUG REINTERPRETATION] original frame labels:")
    for k in sorted(original_counts):
        print(f"  {k}: {original_counts[k]}")
    print("[DEBUG REINTERPRETATION] corrected temporal-first labels:")
    for k in sorted(corrected_counts):
        print(f"  {k}: {corrected_counts[k]}")

    tc = defaultdict(int)
    for label in corrected_track_labels.values():
        tc[label] += 1
    print("[DEBUG REINTERPRETATION] corrected source-track labels:")
    for k in ("STATIC", "DYNAMIC", "MISTRACK", "UNCERTAIN"):
        print(f"  {k}: {tc[k]}")

    frame_source = defaultdict(list)
    frame_target = defaultdict(list)
    source_history = defaultdict(list)

    for r in frame_rows:
        cam = r["source_cam"]
        f = as_int(r["frame"])
        frame_source[(cam, f)].append(r)
        source_history[
            (as_int(r["anchor_id"]), cam, as_int(r["source_track"]))
        ].append(r)

    for key in source_history:
        source_history[key].sort(key=lambda r: as_int(r["frame"]))

    for tr in target_rows:
        frame_target[(tr["target_cam"], as_int(tr["frame"]))].append(tr)

    static_uv = collect_static_uv(frame_rows, target_rows)
    all_frames = sorted({as_int(r["frame"]) for r in frame_rows})
    if not all_frames:
        raise RuntimeError("No frame evidence.")

    cameras = sorted(
        p.name for p in image_path.iterdir()
        if p.is_dir() and p.name.startswith("cam_")
    )
    if not cameras:
        raise RuntimeError("No cam_* image folders.")

    debug_dir = result_dir / "visual_debug"
    if args.force and debug_dir.exists():
        shutil.rmtree(debug_dir)
    debug_dir.mkdir(parents=True, exist_ok=True)

    timeline = render_timeline(
        debug_dir, frame_rows, corrected_track_labels, max(all_frames)
    )
    print(timeline)

    sample = None
    for cam in cameras:
        p = find_frame(image_path, cam, all_frames[0])
        if p:
            sample = cv2.imread(str(p))
            if sample is not None:
                break
    if sample is None:
        raise RuntimeError("Could not open RGB frames.")

    h, w = sample.shape[:2]
    tile_w = args.panel_tile_width
    tile_h = int(round(tile_w * h / w))
    cols = 4
    rows = int(math.ceil(len(cameras) / cols))

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    panel_path = debug_dir / "all_cameras_verification.mp4"
    panel_writer = cv2.VideoWriter(
        str(panel_path), fourcc, args.fps, (cols * tile_w, rows * tile_h)
    )
    if not panel_writer.isOpened():
        raise RuntimeError(f"Could not create {panel_path}")

    cam_writers = {}
    if not args.no_per_camera_videos:
        for cam in cameras:
            p = debug_dir / f"{cam}_verification.mp4"
            vw = cv2.VideoWriter(str(p), fourcc, args.fps, (w, h))
            if not vw.isOpened():
                raise RuntimeError(f"Could not create {p}")
            cam_writers[cam] = (vw, p)

    snapshots = {
        int(x.strip()) for x in args.snapshots.split(",")
        if x.strip()
    }

    for frame in range(min(all_frames), max(all_frames) + 1):
        tiles = []
        for cam in cameras:
            dbg = render_camera_frame(
                image_path, cam, frame,
                frame_source, frame_target, source_history,
                static_uv, corrected_track_labels, args.trail_frames
            )
            if cam in cam_writers:
                cam_writers[cam][0].write(dbg)
            tiles.append(cv2.resize(dbg, (tile_w, tile_h), interpolation=cv2.INTER_AREA))

        while len(tiles) < rows * cols:
            tiles.append(np.zeros((tile_h, tile_w, 3), np.uint8))

        panel_rows = []
        for rr in range(rows):
            panel_rows.append(
                np.concatenate(tiles[rr * cols:(rr + 1) * cols], axis=1)
            )
        panel = np.concatenate(panel_rows, axis=0)
        panel_writer.write(panel)

        if frame in snapshots:
            p = debug_dir / f"frame_{frame:06d}_panel.png"
            cv2.imwrite(str(p), panel)
            print(p)

    panel_writer.release()
    for vw, p in cam_writers.values():
        vw.release()
        print(p)

    print(panel_path)
    print(
        "\nLegend: white X=initial X0 static projection; "
        "circle=same-camera temporal observation; "
        "square=same-frame multiview SIFT observation; "
        "white ring=COLMAP-verified dynamic geometry."
    )


if __name__ == "__main__":
    main()
