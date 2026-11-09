#!/usr/bin/env python
"""Suggest the static and motion frame intervals of a real capture (DIAGNOSTIC, not inference).

The red_fridge table of ~/run_calibrated_bootstrap.sh used fixed thresholds against frame 0:
mask IoU < 0.98 or median in-mask |dz| > 2 mm in any camera, for 3 frames. Other captures sit
at a different steady level from the first frame (a camera at IoU 0.92, a constant 3 mm depth
offset), which trips those thresholds at frame 1. So each camera is measured against its own
first frames:

    |IoU - b_iou| > max(0.02, 5 * robust sd)   or   dz > b_dz + 2 mm     for 3 frames

with b_iou, b_dz the medians over frames 1..20, on the selected cameras. --lag shifts the
result earlier, because this test fires later than the fixed one; 19 covers every red_fridge
episode (static end = onset - lag - 5, motion start = onset - lag). The same test with time
reversed gives the last changing frame: a backward run's endpoint and its Phase-0 evidence must
lie after it. Reads masks and depth only.

    python tools/estimate_motion_interval.py --source S --cameras cam_001,cam_002 --lag 19
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from utils.camera_selection import available_cameras, resolve  # noqa: E402


def _mask(source, cam, t):
    m = np.asarray(Image.open(source / "mask" / cam / f"frame_{t:06d}.png")) > 0
    return m[..., 0] if m.ndim == 3 else m


def series(source, cam, frames, ref=0):
    """Mask IoU and median in-mask |dz| of every frame against frame `ref`."""
    m0, z0 = _mask(source, cam, ref), np.load(source / "depth_npy" / cam / f"frame_{ref:06d}.npy")
    iou, dz = [], []
    for t in frames:
        m, z = _mask(source, cam, t), np.load(source / "depth_npy" / cam / f"frame_{t:06d}.npy")
        iou.append((m & m0).sum() / max((m | m0).sum(), 1))
        ok = m & m0 & (z > 0) & (z0 > 0)
        dz.append(float(np.median(np.abs(z[ok] - z0[ok]))) if ok.any() else np.inf)
    return np.array(iou), np.array(dz)


def changed(iou, dz, baseline=20, d_iou=0.02, d_z=0.002, k=5.0):
    b = np.median(iou[1:baseline + 1])
    noise = 1.4826 * np.median(np.abs(iou[1:baseline + 1] - b))
    out = (np.abs(iou - b) > max(d_iou, k * noise)) | (dz > np.median(dz[1:baseline + 1]) + d_z)
    out[0] = False
    return out


def onset(curves, run=3):
    n = len(curves[0][0])
    bad = np.any([changed(i, z) for i, z in curves], axis=0)
    return next((t for t in range(n - run + 1) if bad[t:t + run].all()), None)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--source", required=True)
    ap.add_argument("--cameras", default="all")
    ap.add_argument("--lag", type=int, default=19)
    a = ap.parse_args(argv)
    src = Path(a.source)
    cams = resolve(a.cameras, available_cameras(src))
    n = len(list((src / "rgb" / cams[0]).iterdir()))
    t0 = onset([series(src, c, range(n)) for c in cams])
    back = onset([series(src, c, range(n - 1, -1, -1), ref=n - 1) for c in cams])
    rec = {"source": str(src), "cameras": cams, "frames": n, "onset": t0,
           "static": None if t0 is None else [0, t0 - a.lag - 5],
           "motion": None if t0 is None else [t0 - a.lag, n - 1],
           "last_changing_frame": None if back is None else n - 1 - back}
    print(json.dumps(rec))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
