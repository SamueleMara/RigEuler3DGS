#!/usr/bin/env python3
"""Create a COLMAP scene subset containing only selected camera/image prefixes."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from utils.read_write_model import Image, Point3D, read_model, write_model


def _link(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    dst.symlink_to(os.path.relpath(src.resolve(), dst.parent.resolve()))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-colmap", required=True)
    parser.add_argument("--output-colmap", required=True)
    parser.add_argument("--camera-prefix", required=True, help="Example: cam_000")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    source = Path(args.source_colmap).expanduser().resolve()
    output = Path(args.output_colmap).expanduser().resolve()
    prefix = str(args.camera_prefix).strip().rstrip("/") + "/"
    if output.exists() and not args.force:
        raise FileExistsError(f"{output} already exists; pass --force to rebuild")

    cameras, images, points3d = read_model(str(source / "sparse" / "0"), ext=".bin")
    kept_images = {
        image_id: image
        for image_id, image in images.items()
        if str(image.name).startswith(prefix)
    }
    if not kept_images:
        raise RuntimeError(f"No COLMAP images found with prefix {prefix!r}")

    kept_camera_ids = {int(image.camera_id) for image in kept_images.values()}
    kept_cameras = {
        camera_id: camera
        for camera_id, camera in cameras.items()
        if int(camera_id) in kept_camera_ids
    }
    kept_image_ids = set(kept_images)
    referenced_point_ids = {
        int(pid)
        for image in kept_images.values()
        for pid in np.asarray(image.point3D_ids).reshape(-1)
        if int(pid) >= 0
    }

    kept_points: dict[int, Point3D] = {}
    for point_id, point in points3d.items():
        if int(point_id) not in referenced_point_ids:
            continue
        image_ids = np.asarray(point.image_ids)
        point2d_idxs = np.asarray(point.point2D_idxs)
        mask = np.asarray([int(image_id) in kept_image_ids for image_id in image_ids])
        kept_points[int(point_id)] = Point3D(
            id=int(point_id),
            xyz=np.asarray(point.xyz),
            rgb=np.asarray(point.rgb),
            error=float(point.error),
            image_ids=image_ids[mask],
            point2D_idxs=point2d_idxs[mask],
        )

    output.mkdir(parents=True, exist_ok=True)
    sparse = output / "sparse" / "0"
    sparse.mkdir(parents=True, exist_ok=True)
    write_model(kept_cameras, kept_images, kept_points, str(sparse), ext=".bin")
    write_model(kept_cameras, kept_images, kept_points, str(sparse), ext=".txt")

    for image in kept_images.values():
        src = source / "images" / image.name
        if not src.exists():
            raise FileNotFoundError(f"Image referenced by COLMAP is missing: {src}")
        _link(src, output / "images" / image.name)

    summary = {
        "source_colmap": str(source),
        "output_colmap": str(output),
        "camera_prefix": prefix.rstrip("/"),
        "num_cameras": len(kept_cameras),
        "num_images": len(kept_images),
        "num_points3D": len(kept_points),
        "images": [image.name for image in kept_images.values()],
    }
    (output / "camera_subset_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
