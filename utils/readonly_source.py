"""Load an endpoint scene without writing into the dataset.

CODEX_BIDIRECTIONAL_IEKF_AND_ARTICULATED_3DGS_MERGE.md section S.

The 3DGS COLMAP reader converts `points3D.bin` to `points3D.ply` on first
open and writes it NEXT TO THE SOURCE. Pointed at a dataset under a backup
tree, that mutates data nobody meant to modify -- observed when loading
`colmap_final` for the first time.

This does not change the loader. It gives the loader a staging directory
under the RUN's own output instead: the small COLMAP metadata is copied
there, the image and depth trees are left where they are and reached through
a relative `--images ../..`-style path, and any derived cache the loader
writes lands in the staging copy.

Only the bidirectional/endpoint path uses this. The ordinary forward run is
untouched, per rule D7.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

# Everything the COLMAP reader needs from sparse/0. Images and depth are NOT
# copied -- they are the bulk of the dataset and are read through the source.
SPARSE_FILES = ("cameras.bin", "cameras.txt", "images.bin", "images.txt",
                "points3D.bin", "points3D.txt", "depth_params.json",
                "test.txt", "points3D.ply")


def stage_colmap(source_colmap, cache_root, *, name: str) -> dict:
    """Copy a COLMAP model into the run's cache and report where it landed.

    `source_colmap` is the directory holding `sparse/0` -- <scene>/colmap or
    <scene>/colmap_final. Returns a manifest the caller persists, so a reader
    of the run can see exactly which source it came from and that nothing was
    written back.
    """
    source_colmap = Path(source_colmap).expanduser().resolve()
    sparse = source_colmap / "sparse" / "0"
    if not sparse.is_dir():
        raise FileNotFoundError(
            f"{source_colmap} has no sparse/0; expected a COLMAP model "
            "directory such as <scene>/colmap or <scene>/colmap_final")

    staged = Path(cache_root).expanduser().resolve() / name
    staged_sparse = staged / "sparse" / "0"
    staged_sparse.mkdir(parents=True, exist_ok=True)

    copied, missing = [], []
    for filename in SPARSE_FILES:
        src = sparse / filename
        if src.is_file():
            shutil.copy2(src, staged_sparse / filename)
            copied.append(filename)
        else:
            missing.append(filename)
    if not any(f.startswith("cameras") for f in copied):
        raise FileNotFoundError(f"{sparse} holds no cameras.bin or cameras.txt")
    if not any(f.startswith("images") for f in copied):
        raise FileNotFoundError(f"{sparse} holds no images.bin or images.txt")

    manifest = {
        "source_colmap": str(source_colmap),
        "source_scene": str(source_colmap.parent),
        "staged_colmap": str(staged),
        "copied": copied,
        "absent_in_source": missing,
        "images_read_from_source": True,
        "note": ("The COLMAP reader writes points3D.ply beside the model it "
                 "opens. It opens this staged copy, so any derived cache "
                 "lands here and the dataset tree is never written to."),
    }
    (staged / "stage_manifest.json").write_text(json.dumps(manifest, indent=1))
    return manifest


def images_argument(staged_colmap, source_scene, images_dirname: str = "rgb") -> str:
    """The --images value that reaches the real frames from the staged model.

    The reader joins this onto the staged COLMAP root, so it has to climb out
    of the cache and back to the dataset. Returned as a relative path when
    one exists and an absolute path otherwise, because os.path.join accepts
    either and an absolute one is correct across filesystems.
    """
    staged_colmap = Path(staged_colmap).expanduser().resolve()
    target = Path(source_scene).expanduser().resolve() / images_dirname
    if not target.is_dir():
        raise FileNotFoundError(f"no image directory at {target}")
    try:
        import os
        return os.path.relpath(target, staged_colmap)
    except ValueError:
        return str(target)


def source_untouched(source_colmap, before: set[str] | None = None) -> dict:
    """Evidence that the dataset tree still holds what it held.

    Call once before staging to capture the listing, once after the run to
    compare. Cheap, and it turns "we did not write to the source" from a
    claim into a check.
    """
    sparse = Path(source_colmap).expanduser().resolve() / "sparse" / "0"
    now = {p.name for p in sparse.iterdir()} if sparse.is_dir() else set()
    if before is None:
        return {"listing": sorted(now)}
    added = sorted(now - set(before))
    removed = sorted(set(before) - now)
    return {
        "listing": sorted(now),
        "added": added,
        "removed": removed,
        "untouched": not added and not removed,
    }
