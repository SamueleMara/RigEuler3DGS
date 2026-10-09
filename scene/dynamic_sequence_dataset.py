#
# Fixed-camera video expansion for unified dynamic 3DGS.
#
# This module deliberately keeps the canonical 3DGS initialization vanilla:
# Scene first loads the standard COLMAP dataset from source_path/sparse and
# source_path/images. Only after those COLMAP cameras have been instantiated do
# we clone each fixed camera template over all dynamic RGB/depth frames.
#
# Expected user layout:
#   ROOT/
#     colmap/
#       images/
#       sparse/0/{cameras,images,points3D}.{bin,txt}
#     rgb/cam_000/frame_000000.png ...
#     depth_npy/cam_000/frame_000000.npy ...   optional
#     depth_png/cam_000/frame_000000.png ...   optional
#     dinov3/cam_000/frame_000000.pt ...      optional dense DINOv3 tensors
#     metadata/...                             optional
#
# Recommended invocation:
#   python train.py -s ROOT/colmap --dynamic_motion --dynamic_sequence_root ROOT
#

from __future__ import annotations

import csv
import json
import re
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np
from PIL import Image

from scene.cameras import Camera


_TIMESTAMP_TABLE_CACHE: Dict[str, object] = {}


def _get_arg(args, name: str, default=None):
    return getattr(args, name, default)


def _timestamp_file(root: Path, dyn) -> Optional[Path]:
    """Resolve an optional real acquisition-timestamp table.

    ``--dynamic_timestamps_file`` accepts JSON or CSV.  When it is omitted,
    ``ROOT/timestamps.json`` and ``ROOT/metadata/timestamps.json`` are detected
    automatically.  Falling back to ``frame_id / fps`` remains exact for a
    constant-rate rendered sequence.
    """
    requested = str(_get_arg(dyn, "dynamic_timestamps_file", "") or "").strip()
    candidates = []
    if requested:
        path = Path(requested).expanduser()
        candidates.append(path if path.is_absolute() else root / path)
    else:
        candidates.extend((root / "timestamps.json", root / "metadata" / "timestamps.json"))
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    if requested:
        raise FileNotFoundError(f"Dynamic timestamp file does not exist: {candidates[0]}")
    return None


def _load_timestamp_table(path: Path):
    cache_key = str(path)
    if cache_key in _TIMESTAMP_TABLE_CACHE:
        return _TIMESTAMP_TABLE_CACHE[cache_key]
    if path.suffix.lower() == ".json":
        with path.open("r", encoding="utf-8") as stream:
            table = json.load(stream)
    elif path.suffix.lower() in (".csv", ".tsv"):
        delimiter = "\t" if path.suffix.lower() == ".tsv" else ","
        rows = []
        with path.open("r", encoding="utf-8", newline="") as stream:
            for row in csv.DictReader(stream, delimiter=delimiter):
                rows.append(row)
        table = rows
    else:
        raise ValueError("Dynamic timestamps must be JSON, CSV or TSV")
    _TIMESTAMP_TABLE_CACHE[cache_key] = table
    return table


def _frame_key_to_int(value) -> Optional[int]:
    if isinstance(value, int):
        return int(value)
    match = re.search(r"(\d+)$", str(value))
    return int(match.group(1)) if match else None


def _timestamp_from_table(table, camera_id: str, frame_id: int) -> float:
    """Read common global/per-camera JSON and tabular timestamp layouts."""
    if isinstance(table, dict):
        nested = table.get(camera_id, table)
        if isinstance(nested, list):
            if frame_id >= len(nested):
                raise KeyError(frame_id)
            value = nested[frame_id]
            if isinstance(value, dict):
                value = next(
                    value[key] for key in ("timestamp_seconds", "timestamp", "time")
                    if key in value
                )
            return float(value)
        if isinstance(nested, dict):
            for key, value in nested.items():
                if _frame_key_to_int(key) == frame_id:
                    if isinstance(value, dict):
                        value = next(
                            value[name]
                            for name in ("timestamp_seconds", "timestamp", "time")
                            if name in value
                        )
                    return float(value)
    if isinstance(table, list):
        if table and not isinstance(table[0], dict):
            return float(table[frame_id])
        for row in table:
            row_camera = str(row.get("camera", row.get("camera_id", camera_id)))
            row_frame = _frame_key_to_int(row.get("frame", row.get("frame_id", "")))
            if row_camera == camera_id and row_frame == frame_id:
                for name in ("timestamp_seconds", "timestamp", "time"):
                    if name in row and str(row[name]).strip():
                        return float(row[name])
    raise KeyError(f"No timestamp for {camera_id} frame {frame_id}")


def _physical_timestamp(root: Path, camera_id: str, frame_id: int, dyn) -> float:
    path = _timestamp_file(root, dyn)
    if path is None:
        fps = float(_get_arg(dyn, "fps", 30.0))
        if fps <= 0.0:
            raise ValueError("--fps must be positive when no timestamp file is supplied")
        return float(frame_id) / fps
    try:
        return _timestamp_from_table(_load_timestamp_table(path), camera_id, frame_id)
    except (KeyError, StopIteration) as exc:
        raise RuntimeError(
            f"Missing physical timestamp for {camera_id} frame {frame_id} in {path}"
        ) from exc


def infer_dynamic_sequence_root(source_path: str, dyn=None) -> Optional[Path]:
    """Infer the ROOT that contains rgb/cam_xxx/frame_xxxxxx.png.

    If the user trains vanilla 3DGS with -s ROOT/colmap, the sequence root is
    ROOT. If the user passes -s ROOT and ROOT/colmap exists, the sequence root is
    ROOT. Explicit --dynamic_sequence_root always wins.
    """
    explicit = _get_arg(dyn, "dynamic_sequence_root", "") if dyn is not None else ""
    if explicit:
        root = Path(explicit).expanduser().resolve()
        return root if root.exists() else root

    src = Path(source_path).expanduser().resolve()
    rgb_dir = _get_arg(dyn, "dynamic_rgb_dir", "rgb") if dyn is not None else "rgb"
    if (src / rgb_dir).is_dir():
        return src
    if src.name == "colmap" and (src.parent / rgb_dir).is_dir():
        return src.parent
    if (src / "colmap").is_dir() and (src / rgb_dir).is_dir():
        return src
    return None


def infer_colmap_source_path(source_path: str) -> str:
    """Return the path that should be passed to the vanilla COLMAP reader."""
    src = Path(source_path).expanduser().resolve()
    if (src / "sparse").exists():
        return str(src)
    if (src / "colmap" / "sparse").exists():
        return str(src / "colmap")
    return str(src)


def dynamic_sequence_available(source_path: str, dyn=None) -> bool:
    root = infer_dynamic_sequence_root(source_path, dyn)
    if root is None:
        return False
    rgb_dir = _get_arg(dyn, "dynamic_rgb_dir", "rgb") if dyn is not None else "rgb"
    return (root / rgb_dir).is_dir() and any((root / rgb_dir).glob("cam_*"))


def _extract_cam_id(text: str) -> Optional[str]:
    m = re.search(r"cam_\d+", text or "")
    return m.group(0) if m else None


def _extract_frame_id(path_or_name: str) -> int:
    name = Path(path_or_name).stem
    m = re.search(r"frame[_-]?(\d+)", name)
    if m:
        return int(m.group(1))
    nums = re.findall(r"\d+", name)
    return int(nums[-1]) if nums else 0


def _camera_template_name(cam) -> str:
    parts = [
        str(getattr(cam, "image_name", "")),
        str(getattr(cam, "image_path", "")),
        str(getattr(cam, "colmap_id", "")),
        str(getattr(cam, "uid", "")),
    ]
    return " ".join(parts)


def map_colmap_templates_to_camera_ids(templates: Sequence, cam_ids: Sequence[str]) -> Dict[str, object]:
    """Map COLMAP frame-0 Camera objects to dynamic folders cam_000, cam_001, ...

    Preferred mapping uses the cam_XXX token in the COLMAP image name. Fallback
    is sorted-order mapping when the number of templates equals the number of
    dynamic camera folders.
    """
    by_id: Dict[str, object] = {}
    unmatched = []
    for cam in templates:
        cam_id = _extract_cam_id(_camera_template_name(cam))
        if cam_id is not None and cam_id in cam_ids and cam_id not in by_id:
            by_id[cam_id] = cam
        else:
            unmatched.append(cam)

    missing = [c for c in cam_ids if c not in by_id]
    if missing:
        # Stable fallback: sort by image_name/uid and assign to sorted cam folders.
        unmatched_sorted = sorted(unmatched, key=lambda c: (str(getattr(c, "image_name", "")), int(getattr(c, "uid", 0))))
        if len(unmatched_sorted) >= len(missing):
            for cam_id, cam in zip(missing, unmatched_sorted):
                by_id[cam_id] = cam

    return by_id


def _compute_resolution_from_args(image: Image.Image, args, resolution_scale: float) -> Tuple[int, int]:
    orig_w, orig_h = image.size
    resolution_arg = int(getattr(args, "resolution", -1))
    if resolution_arg in [1, 2, 4, 8]:
        return round(orig_w / (resolution_scale * resolution_arg)), round(orig_h / (resolution_scale * resolution_arg))
    if resolution_arg == -1:
        global_down = orig_w / 1600 if orig_w > 1600 else 1
    else:
        global_down = orig_w / float(resolution_arg)
    scale = float(global_down) * float(resolution_scale)
    return int(orig_w / scale), int(orig_h / scale)


def _load_depth_for_frame(root: Path, cam_id: str, frame_id: int, resolution: Tuple[int, int], dyn) -> Optional[np.ndarray]:
    use_depth = bool(_get_arg(dyn, "dynamic_use_depth", False))
    if not use_depth:
        return None

    use_npy = bool(_get_arg(dyn, "dynamic_use_depth_npy", True))
    depth_npy_dir = _get_arg(dyn, "dynamic_depth_npy_dir", "depth_npy")
    depth_png_dir = _get_arg(dyn, "dynamic_depth_png_dir", "depth_png")

    npy_path = root / depth_npy_dir / cam_id / f"frame_{frame_id:06d}.npy"
    png_path = root / depth_png_dir / cam_id / f"frame_{frame_id:06d}.png"

    depth = None
    if use_npy and npy_path.exists():
        depth = np.load(str(npy_path)).astype(np.float32)
    elif png_path.exists():
        raw = cv2.imread(str(png_path), cv2.IMREAD_UNCHANGED)
        if raw is not None:
            depth = raw.astype(np.float32)
            # Scale 16-bit PNGs only when they encode normalized inverse depth.
            # Metric depth PNGs commonly store millimetres and must retain that
            # physical scale before inversion.
            if (
                not bool(_get_arg(dyn, "dynamic_depth_is_metric", True))
                and np.issubdtype(raw.dtype, np.integer)
                and raw.dtype.itemsize > 1
            ):
                depth = depth / float(2 ** 16)
    elif (not use_npy) and npy_path.exists():
        depth = np.load(str(npy_path)).astype(np.float32)

    if depth is None:
        return None
    if depth.ndim == 3:
        depth = depth[..., 0]

    metric_depth = bool(_get_arg(dyn, "dynamic_depth_is_metric", True))
    if metric_depth:
        valid = depth > 1e-8
        invdepth = np.zeros_like(depth, dtype=np.float32)
        invdepth[valid] = 1.0 / depth[valid]
    else:
        invdepth = depth.astype(np.float32)

    # Camera expects an inverse-depth map aligned to the resized image.
    target_w, target_h = resolution
    if invdepth.shape[1] != target_w or invdepth.shape[0] != target_h:
        invdepth = cv2.resize(invdepth, (target_w, target_h), interpolation=cv2.INTER_NEAREST)
    return invdepth.astype(np.float32)



def _dinov3_path_for_frame(root: Path, cam_id: str, frame_id: int, dyn) -> str:
    use_dino = bool(_get_arg(dyn, "use_dinov3", False))
    if not use_dino:
        return ""
    dino_root_name = _get_arg(dyn, "dinov3_root", "dinov3")
    dino_ext = _get_arg(dyn, "dinov3_feature_ext", ".pt")
    if not str(dino_ext).startswith("."):
        dino_ext = "." + str(dino_ext)
    path = root / dino_root_name / cam_id / f"frame_{frame_id:06d}{dino_ext}"
    return str(path) if path.exists() else ""

def _iter_frame_paths(root: Path, rgb_dir: str, cam_id: str, dyn) -> List[Path]:
    cam_dir = root / rgb_dir / cam_id
    paths = sorted(list(cam_dir.glob("*.png")) + list(cam_dir.glob("*.jpg")) + list(cam_dir.glob("*.jpeg")))
    frame_start = int(_get_arg(dyn, "dynamic_frame_start", 0))
    frame_end = int(_get_arg(dyn, "dynamic_frame_end", -1))
    frame_stride = max(1, int(_get_arg(dyn, "dynamic_frame_stride", 1)))
    out = []
    for p in paths:
        fid = _extract_frame_id(p.name)
        if fid < frame_start:
            continue
        if frame_end >= 0 and fid > frame_end:
            continue
        if (fid - frame_start) % frame_stride != 0:
            continue
        out.append(p)
    return out


def clone_fixed_camera_for_frame(template_cam, image_path: Path, root: Path, cam_id: str, uid: int, args, dyn, resolution_scale: float, is_test_dataset: bool):
    image = Image.open(str(image_path))
    resolution = _compute_resolution_from_args(image, args, resolution_scale)
    frame_id = _extract_frame_id(image_path.name)
    invdepthmap = _load_depth_for_frame(root, cam_id, frame_id, resolution, dyn)

    image_name = f"{cam_id}_{image_path.stem}"
    cam = Camera(
        resolution,
        colmap_id=getattr(template_cam, "colmap_id", uid),
        R=getattr(template_cam, "R"),
        T=getattr(template_cam, "T"),
        FoVx=getattr(template_cam, "FoVx"),
        FoVy=getattr(template_cam, "FoVy"),
        depth_params=getattr(template_cam, "depth_params", None),
        image=image,
        invdepthmap=invdepthmap,
        image_name=image_name,
        uid=uid,
        data_device=getattr(args, "data_device", "cuda"),
        train_test_exp=getattr(args, "train_test_exp", False),
        is_test_dataset=is_test_dataset,
        is_test_view=getattr(template_cam, "is_test_view", False),
    )
    cam.dynamic_camera_id = cam_id
    cam.frame_id = frame_id
    raw_seconds = _physical_timestamp(root, cam_id, frame_id, dyn)
    # Store the physical acquisition time. Scene-level normalization is applied
    # once, jointly across train and test cameras, according to motion_time_mode.
    cam.raw_timestamp_seconds = raw_seconds
    cam.raw_timestamp = raw_seconds
    cam.timestamp = raw_seconds
    timestamp_path = _timestamp_file(root, dyn)
    cam.timestamp_source = (
        str(timestamp_path)
        if timestamp_path is not None
        else f"frame_id/{float(_get_arg(dyn, 'fps', 30.0)):.12g}"
    )
    cam.original_colmap_image_name = getattr(template_cam, "image_name", "")
    cam.dynamic_image_path = str(image_path)
    cam.dino_feature_path = _dinov3_path_for_frame(root, cam_id, frame_id, dyn)
    return cam


def expand_fixed_colmap_cameras_to_dynamic_sequence(train_templates: Sequence, test_templates: Sequence, source_path: str, args, dyn, resolution_scale: float):
    """Return dynamic train/test camera lists by cloning fixed COLMAP cameras.

    This is the function that wires your layout into the original 3DGS scene:
    COLMAP supplies camera intrinsics/extrinsics, while ROOT/rgb supplies all
    per-frame observations.
    """
    root = infer_dynamic_sequence_root(source_path, dyn)
    if root is None:
        return list(train_templates), list(test_templates), False

    rgb_dir = _get_arg(dyn, "dynamic_rgb_dir", "rgb")
    rgb_root = root / rgb_dir
    if not rgb_root.is_dir():
        return list(train_templates), list(test_templates), False

    requested = _get_arg(dyn, "dynamic_cameras", "")
    if requested:
        cam_ids = [c.strip() for c in requested.split(",") if c.strip()]
    else:
        cam_ids = sorted([p.name for p in rgb_root.iterdir() if p.is_dir() and p.name.startswith("cam_")])

    templates = list(train_templates) + list(test_templates)
    template_by_cam = map_colmap_templates_to_camera_ids(templates, cam_ids)
    missing = [c for c in cam_ids if c not in template_by_cam]
    if missing:
        print(f"[Dynamic sequence] WARNING: no COLMAP camera template found for: {missing}. These cameras will be skipped.")

    test_stride = int(_get_arg(dyn, "dynamic_test_stride", 0))
    dyn_train = []
    dyn_test = []
    uid = 0
    for cam_id in cam_ids:
        if cam_id not in template_by_cam:
            continue
        template = template_by_cam[cam_id]
        frame_paths = _iter_frame_paths(root, rgb_dir, cam_id, dyn)
        for p in frame_paths:
            fid = _extract_frame_id(p.name)
            is_test = test_stride > 0 and (fid % test_stride == 0)
            cam = clone_fixed_camera_for_frame(template, p, root, cam_id, uid, args, dyn, resolution_scale, is_test)
            uid += 1
            if is_test:
                dyn_test.append(cam)
            else:
                dyn_train.append(cam)

    print(
        f"[Dynamic sequence] Expanded fixed COLMAP cameras from {source_path} with root={root}: "
        f"{len(dyn_train)} train, {len(dyn_test)} test, cameras={cam_ids}"
    )
    return dyn_train, dyn_test, True
