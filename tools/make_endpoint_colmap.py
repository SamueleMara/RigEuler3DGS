#!/usr/bin/env python
"""Make an ENDPOINT COLMAP model: one instant, one image per camera.

Some conversions (e.g. the real-capture converter) write every frame of every
camera into colmap/sparse/0. A static endpoint model must be trained on one
instant only -- otherwise it is fitted to a moving object, and loading every
frame's image is what runs the GPU out of memory. This keeps, from an existing
model, only the images of frame --frame (one per camera), with the same
cameras and the same 3D points, and writes a text model:

    <out>/sparse/0/{cameras.txt, images.txt, points3D.txt or .bin, *.ply}

The 3D points are copied as they are (the 3DGS loader reads only their xyz and
colour); their track entries may name removed images, which only COLMAP's own
tools would read. Nothing in the source model is modified. --replace moves the
source folder aside to <model>_allframes and puts the new one in its place.

    python tools/make_endpoint_colmap.py --scene <scene folder> --frame 0
    python tools/make_endpoint_colmap.py --scene <scene folder> --frame 0 --replace
"""
from __future__ import annotations

import argparse
import json
import re
import shutil
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))


def read_model(sparse: Path):
    from scene.colmap_loader import (read_extrinsics_binary, read_extrinsics_text,
                                     read_intrinsics_binary, read_intrinsics_text)
    if (sparse / "images.bin").is_file() and (sparse / "cameras.bin").is_file():
        return read_intrinsics_binary(str(sparse / "cameras.bin")), read_extrinsics_binary(str(sparse / "images.bin"))
    return read_intrinsics_text(str(sparse / "cameras.txt")), read_extrinsics_text(str(sparse / "images.txt"))


def frame_of(name: str):
    m = re.search(r"frame_(\d+)\.[A-Za-z]+$", name)
    return int(m.group(1)) if m else None


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--scene", required=True, help="scene folder (contains rgb/ and the COLMAP model)")
    ap.add_argument("--model", default="colmap", help="source model folder inside the scene (default colmap)")
    ap.add_argument("--frame", type=int, default=0, help="the endpoint frame to keep (default 0)")
    ap.add_argument("--out", default=None, help="output model folder (default <scene>/<model>_frameNNNNNN)")
    ap.add_argument("--replace", action="store_true",
                    help="move <model> to <model>_allframes and put the new model at <model>")
    args = ap.parse_args(argv)

    scene = Path(args.scene).expanduser().resolve()
    src = scene / args.model
    sparse = src / "sparse" / "0"
    if not sparse.is_dir():
        raise SystemExit(f"no COLMAP model at {sparse}")
    out = Path(args.out).expanduser().resolve() if args.out else scene / f"{args.model}_frame{args.frame:06d}"
    if out.exists():
        raise SystemExit(f"{out} exists; refusing to overwrite")
    if args.replace and (scene / f"{args.model}_allframes").exists():
        raise SystemExit(f"{scene / (args.model + '_allframes')} exists; refusing to overwrite")

    cams, imgs = read_model(sparse)
    keep = {i: im for i, im in imgs.items() if frame_of(im.name) == args.frame}
    rgb_cams = sorted(p.name for p in (scene / "rgb").glob("cam_*") if p.is_dir()) if (scene / "rgb").is_dir() else []
    per_cam = sorted(im.name.split("/")[0] for im in keep.values())
    if not keep:
        raise SystemExit(f"{sparse}: no image named ...frame_{args.frame:06d}.* among {len(imgs)}")
    if len(set(per_cam)) != len(per_cam):
        raise SystemExit(f"more than one image per camera at frame {args.frame}: {per_cam}")
    if rgb_cams and set(per_cam) != set(rgb_cams):
        raise SystemExit(f"frame {args.frame} has images for {per_cam}, rgb/ has {rgb_cams}")
    used = sorted({im.camera_id for im in keep.values()})

    dst = out / "sparse" / "0"
    dst.mkdir(parents=True)
    with open(dst / "cameras.txt", "w") as fh:
        fh.write("# Camera list with one line of data per camera:\n"
                 "#   CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]\n")
        for cid in used:
            c = cams[cid]
            fh.write(f"{c.id} {c.model} {c.width} {c.height} " + " ".join(repr(float(x)) for x in c.params) + "\n")
    with open(dst / "images.txt", "w") as fh:
        fh.write("# Image list with two lines of data per image:\n"
                 "#   IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME\n"
                 "#   POINTS2D[] as (X, Y, POINT3D_ID)  -- left empty\n")
        for i in sorted(keep):
            im = keep[i]
            q, t = im.qvec, im.tvec
            fh.write(f"{im.id} {q[0]!r} {q[1]!r} {q[2]!r} {q[3]!r} {t[0]!r} {t[1]!r} {t[2]!r} "
                     f"{im.camera_id} {im.name}\n\n")
    copied = []
    for f in sorted(sparse.iterdir()):
        if f.is_file() and (f.name.startswith("points3D") or f.suffix == ".ply"):
            shutil.copy2(f, dst / f.name)
            copied.append(f.name)
    (out / "endpoint_model.json").write_text(json.dumps({
        "source_model": str(sparse), "source_images": len(imgs), "endpoint_frame": args.frame,
        "kept_images": [keep[i].name for i in sorted(keep)], "cameras": used, "copied": copied,
        "note": "one image per camera at the endpoint frame; cameras and 3D points unchanged"}, indent=1))
    print(f"{sparse}: {len(imgs)} images -> {len(keep)} at frame {args.frame} ({', '.join(per_cam)}); "
          f"copied {copied}")
    if args.replace:
        aside = scene / f"{args.model}_allframes"
        src.rename(aside)
        out.rename(src)
        print(f"moved {src} -> {aside}; new endpoint model at {src}")
    else:
        print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
