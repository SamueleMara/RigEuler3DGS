#!/usr/bin/env python
"""Preflight for the masked observation modes: are the whole-object masks there?

Checks, for the first N cameras (rgb/cam_* sorted, the loader's order) and the
first F frames (or all), that mask/<cam>/frame_NNNNNN.{png,jpg,jpeg} exists
exactly once, has the RGB image's size, and is binary (0/255 or 0/1). Reads
only images; no GPU, no writes. Exit 0 = ready, 1 = not ready (each problem is
listed). run_bidirectional_all.sh calls it before any training or output
deletion when the observation mode needs masks.

    python tools/check_masks.py --source <scene> [--mask-source DIR] [--cameras 8] [--frames 60]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from PIL import Image

EXT = (".png", ".jpg", ".jpeg")


def check(source: Path, mask_root: Path, cameras: int | None, frames: int | None, max_report: int = 12):
    cams = sorted(p.name for p in (source / "rgb").glob("cam_*") if p.is_dir())
    if not cams:
        return [f"{source / 'rgb'}: no cam_* folders"], {}
    if cameras:
        if cameras > len(cams):
            return [f"{cameras} cameras requested but {source / 'rgb'} has {len(cams)}"], {}
        cams = cams[:cameras]
    problems, checked = [], 0
    for c in cams:
        rgbs = sorted(q for q in (source / "rgb" / c).iterdir() if q.suffix.lower() in EXT)
        if frames:
            rgbs = rgbs[:frames]
        size = Image.open(rgbs[0]).size if rgbs else None
        for r in rgbs:
            hits = [mask_root / c / (r.stem + e) for e in EXT if (mask_root / c / (r.stem + e)).is_file()]
            if not hits:
                problems.append(f"missing mask {mask_root / c / r.stem}.png")
                continue
            if len(hits) > 1:
                problems.append(f"ambiguous masks {[h.name for h in hits]} in {mask_root / c}")
                continue
            im = Image.open(hits[0])
            if im.size != size:
                problems.append(f"{hits[0]}: size {im.size}, RGB is {size}")
                continue
            a = np.asarray(im)
            if a.ndim == 3:
                if not all(np.array_equal(a[..., 0], a[..., k]) for k in range(1, min(a.shape[2], 3))):
                    problems.append(f"{hits[0]}: colour mask with differing channels")
                    continue
                a = a[..., 0]
            vals = set(np.unique(a).tolist())
            if not (vals <= {0, 255} or vals <= {0, 1}):
                problems.append(f"{hits[0]}: non-binary values {sorted(vals)[:6]}")
                continue
            checked += 1
    return problems, {"cameras": len(cams), "masks_ok": checked}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--source", required=True)
    ap.add_argument("--mask-source", default=None, help="mask root (default <source>/mask)")
    ap.add_argument("--cameras", type=int, default=None)
    ap.add_argument("--frames", type=int, default=None)
    a = ap.parse_args(argv)
    src = Path(a.source)
    root = Path(a.mask_source) if a.mask_source else src / "mask"
    if not root.is_dir():
        print(f"MASKS NOT READY: {root} does not exist (the default observation mode "
              "masked-rgb-expected-quad needs whole-object masks; pass --observation-mode original "
              "explicitly to run without them)")
        return 1
    problems, info = check(src, root, a.cameras, a.frames)
    if problems:
        print(f"MASKS NOT READY under {root}: {len(problems)} problem(s)")
        for p in problems[:12]:
            print("   ", p)
        if len(problems) > 12:
            print(f"    ... and {len(problems) - 12} more")
        return 1
    print(f"masks OK: {info['masks_ok']} files, {info['cameras']} cameras, {root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
