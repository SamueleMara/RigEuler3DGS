#
# SAPIEN / PartNet dynamic multiview sequence reader for the adaptive
# Eulerian-motion 3DGS extension.
#
# Expected root layout, matching the user's renderer output:
#   root/
#     colmap/images, colmap/sparse[/0]
#     rgb/cam_000/frame_000000.png ...
#     depth_npy/cam_000/frame_000000.npy ...      optional
#     depth_png/cam_000/frame_000000.png ...      optional
#     metadata/cameras.json, frame_values.csv ... optional
#     pointcloud/points3D_parts.ply               preferred canonical PLY
#
# The returned object mimics scene.dataset_readers.SceneInfo and CameraInfo
# through SimpleNamespace attributes so it remains compatible with
# utils.camera_utils.cameraList_from_camInfos.
#

from __future__ import annotations

import json
import math
import os
import re
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
from PIL import Image
from plyfile import PlyData, PlyElement

from utils.graphics_utils import BasicPointCloud, focal2fov


def is_sapien_dynamic_scene(source_path: str) -> bool:
    root = Path(source_path)
    return (
        (root / "rgb").is_dir()
        and any((root / "rgb").glob("cam_*"))
        and ((root / "metadata").is_dir() or (root / "colmap").is_dir())
    )


def _safe_float(x, default=0.0):
    try:
        return float(x)
    except Exception:
        return default


def _matrix_from_any(value) -> Optional[np.ndarray]:
    if value is None:
        return None
    arr = np.asarray(value, dtype=np.float64)
    if arr.size == 16:
        return arr.reshape(4, 4)
    if arr.shape == (4, 4):
        return arr
    if arr.size == 9:
        out = np.eye(4, dtype=np.float64)
        out[:3, :3] = arr.reshape(3, 3)
        return out
    if arr.shape == (3, 4):
        out = np.eye(4, dtype=np.float64)
        out[:3, :] = arr
        return out
    return None


def _k_from_entry(entry: dict) -> Optional[np.ndarray]:
    for key in ["K", "k", "intrinsic", "intrinsics", "camera_matrix", "cameraMatrix"]:
        if key in entry:
            arr = np.asarray(entry[key], dtype=np.float64)
            if arr.size == 9:
                return arr.reshape(3, 3)
            if arr.shape == (3, 3):
                return arr
            if arr.size >= 4:
                fx, fy, cx, cy = arr.flatten()[:4]
                return np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float64)
    if "fx" in entry and "fy" in entry:
        fx = _safe_float(entry.get("fx"))
        fy = _safe_float(entry.get("fy"))
        cx = _safe_float(entry.get("cx"), _safe_float(entry.get("width", 0)) * 0.5)
        cy = _safe_float(entry.get("cy"), _safe_float(entry.get("height", 0)) * 0.5)
        return np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float64)
    return None


def _pose_from_entry(entry: dict) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """Return GraphDeco CameraInfo convention: R=Rcw.T, T=tcw."""
    c2w_keys = [
        "T_wc", "Twc", "c2w", "camera_to_world", "cam2world",
        "world_from_camera", "pose", "transform_matrix",
    ]
    w2c_keys = [
        "T_cw", "Tcw", "w2c", "world_to_camera", "world2cam",
        "extrinsic", "extrinsics", "view_matrix",
    ]
    for key in w2c_keys:
        if key in entry:
            w2c = _matrix_from_any(entry[key])
            if w2c is not None:
                Rcw = w2c[:3, :3]
                tcw = w2c[:3, 3]
                return Rcw.T.astype(np.float32), tcw.astype(np.float32)
    for key in c2w_keys:
        if key in entry:
            c2w = _matrix_from_any(entry[key])
            if c2w is not None:
                w2c = np.linalg.inv(c2w)
                Rcw = w2c[:3, :3]
                tcw = w2c[:3, 3]
                return Rcw.T.astype(np.float32), tcw.astype(np.float32)
    # Fallback fields occasionally used by synthetic renderers.
    if "rotation" in entry and "position" in entry:
        R = np.asarray(entry["rotation"], dtype=np.float64)
        p = np.asarray(entry["position"], dtype=np.float64).reshape(3)
        if R.size == 9:
            R = R.reshape(3, 3)
            c2w = np.eye(4, dtype=np.float64)
            c2w[:3, :3] = R
            c2w[:3, 3] = p
            w2c = np.linalg.inv(c2w)
            return w2c[:3, :3].T.astype(np.float32), w2c[:3, 3].astype(np.float32)
    return None, None


def _iter_metadata_camera_entries(data) -> Iterable[Tuple[str, dict]]:
    if isinstance(data, list):
        for idx, entry in enumerate(data):
            if isinstance(entry, dict):
                name = str(entry.get("name", entry.get("camera_id", entry.get("id", f"cam_{idx:03d}"))))
                yield name, entry
        return
    if not isinstance(data, dict):
        return
    if isinstance(data.get("cameras"), list):
        for idx, entry in enumerate(data["cameras"]):
            if isinstance(entry, dict):
                name = str(entry.get("name", entry.get("camera_id", entry.get("id", f"cam_{idx:03d}"))))
                yield name, entry
        return
    for key, value in data.items():
        if isinstance(value, dict) and (key.startswith("cam_") or "K" in value or "fx" in value or "intrinsics" in value):
            name = str(value.get("name", value.get("camera_id", key)))
            yield name, value


def _load_metadata_cameras(root: Path, metadata_dir: str) -> Dict[str, SimpleNamespace]:
    out: Dict[str, SimpleNamespace] = {}
    path = root / metadata_dir / "cameras.json"
    if not path.exists():
        return out
    with open(path, "r") as f:
        data = json.load(f)
    for raw_name, entry in _iter_metadata_camera_entries(data):
        name_match = re.search(r"cam_\d+", raw_name)
        name = name_match.group(0) if name_match else str(raw_name)
        K = _k_from_entry(entry)
        R, T = _pose_from_entry(entry)
        width = int(entry.get("width", entry.get("w", 0)) or 0)
        height = int(entry.get("height", entry.get("h", 0)) or 0)
        if K is None or R is None or T is None:
            continue
        if width <= 0:
            width = int(round(K[0, 2] * 2)) if K[0, 2] > 0 else 0
        if height <= 0:
            height = int(round(K[1, 2] * 2)) if K[1, 2] > 0 else 0
        fx = float(K[0, 0])
        fy = float(K[1, 1])
        out[name] = SimpleNamespace(
            camera_id=name,
            width=width,
            height=height,
            fx=fx,
            fy=fy,
            FovX=focal2fov(fx, width) if width > 0 and fx > 0 else float(entry.get("FovX", entry.get("fovx", math.pi / 3))),
            FovY=focal2fov(fy, height) if height > 0 and fy > 0 else float(entry.get("FovY", entry.get("fovy", math.pi / 3))),
            R=R,
            T=T,
        )
    return out


def _load_colmap_camera_templates(root: Path, args) -> Dict[str, SimpleNamespace]:
    """Fallback: reuse the original COLMAP reader on root/colmap."""
    out: Dict[str, SimpleNamespace] = {}
    colmap_root = root / "colmap"
    if not (colmap_root / "sparse").exists():
        return out
    try:
        from scene.dataset_readers import sceneLoadTypeCallbacks
        scene_info = sceneLoadTypeCallbacks["Colmap"](
            str(colmap_root), "images", getattr(args, "depths", ""), False, getattr(args, "train_test_exp", False)
        )
    except Exception as exc:
        print(f"[SAPIEN reader] Could not reuse COLMAP reader for camera templates: {exc}")
        return out
    for cam in list(scene_info.train_cameras) + list(scene_info.test_cameras):
        candidates = [str(getattr(cam, "image_name", "")), str(getattr(cam, "image_path", ""))]
        cam_id = None
        for c in candidates:
            m = re.search(r"cam_\d+", c)
            if m:
                cam_id = m.group(0)
                break
        if cam_id is None:
            cam_id = f"cam_{len(out):03d}"
        if cam_id not in out:
            out[cam_id] = SimpleNamespace(
                camera_id=cam_id,
                width=int(getattr(cam, "width", 0)),
                height=int(getattr(cam, "height", 0)),
                FovX=float(getattr(cam, "FovX")),
                FovY=float(getattr(cam, "FovY")),
                R=getattr(cam, "R"),
                T=getattr(cam, "T"),
            )
    return out


def _read_ply_as_basic_point_cloud(path: Path) -> BasicPointCloud:
    ply = PlyData.read(str(path))
    v = ply.elements[0]
    xyz = np.stack([np.asarray(v["x"]), np.asarray(v["y"]), np.asarray(v["z"])], axis=1).astype(np.float32)
    names = {p.name for p in v.properties}
    if {"red", "green", "blue"}.issubset(names):
        colors = np.stack([np.asarray(v["red"]), np.asarray(v["green"]), np.asarray(v["blue"])], axis=1).astype(np.float32)
        if colors.max() > 1.0:
            colors /= 255.0
    elif {"r", "g", "b"}.issubset(names):
        colors = np.stack([np.asarray(v["r"]), np.asarray(v["g"]), np.asarray(v["b"])], axis=1).astype(np.float32)
        if colors.max() > 1.0:
            colors /= 255.0
    else:
        colors = np.full_like(xyz, 0.5, dtype=np.float32)
    normals = np.zeros_like(xyz, dtype=np.float32)
    return BasicPointCloud(points=xyz, colors=colors, normals=normals)


def _write_basic_point_cloud_ply(pcd: BasicPointCloud, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    xyz = np.asarray(pcd.points, dtype=np.float32)
    colors = np.asarray(pcd.colors, dtype=np.float32)
    colors_u8 = np.clip(colors * 255.0, 0, 255).astype(np.uint8)
    dtype = [("x", "f4"), ("y", "f4"), ("z", "f4"), ("red", "u1"), ("green", "u1"), ("blue", "u1")]
    arr = np.empty(xyz.shape[0], dtype=dtype)
    arr["x"], arr["y"], arr["z"] = xyz[:, 0], xyz[:, 1], xyz[:, 2]
    arr["red"], arr["green"], arr["blue"] = colors_u8[:, 0], colors_u8[:, 1], colors_u8[:, 2]
    PlyData([PlyElement.describe(arr, "vertex")]).write(str(path))


def _normalization_from_cameras(cameras: List[SimpleNamespace]) -> dict:
    centers = []
    for cam in cameras:
        # GraphDeco CameraInfo convention: R is Rcw.T and T is tcw.
        Rcw = np.asarray(cam.R, dtype=np.float64).T
        tcw = np.asarray(cam.T, dtype=np.float64).reshape(3)
        center = -Rcw.T @ tcw
        centers.append(center)
    if not centers:
        return {"translate": np.zeros(3, dtype=np.float32), "radius": 1.0}
    centers = np.stack(centers, axis=0)
    center = centers.mean(axis=0)
    radius = float(np.linalg.norm(centers - center, axis=1).max() * 1.1)
    if radius <= 1e-6:
        radius = 1.0
    return {"translate": -center.astype(np.float32), "radius": radius}


def _selected_camera_dirs(rgb_root: Path, camera_filter: str) -> List[Path]:
    dirs = [p for p in sorted(rgb_root.glob("cam_*")) if p.is_dir()]
    if camera_filter.strip():
        wanted = {x.strip() for x in camera_filter.split(",") if x.strip()}
        dirs = [p for p in dirs if p.name in wanted]
    return dirs


def _frame_id_from_path(path: Path) -> int:
    m = re.search(r"frame_(\d+)", path.stem)
    if m:
        return int(m.group(1))
    nums = re.findall(r"\d+", path.stem)
    return int(nums[-1]) if nums else 0


def _depth_path_for(root: Path, args, cam_id: str, frame_stem: str) -> str:
    if not getattr(args, "sapien_use_depth", False):
        return ""
    if getattr(args, "sapien_use_depth_npy", True):
        for suffix in [".npy", ".npz"]:
            p = root / getattr(args, "sapien_depth_npy_dir", "depth_npy") / cam_id / f"{frame_stem}{suffix}"
            if p.exists():
                return str(p)
    p = root / getattr(args, "sapien_depth_png_dir", "depth_png") / cam_id / f"{frame_stem}.png"
    return str(p) if p.exists() else ""


def load_sapien_dynamic_scene(source_path: str, args) -> SimpleNamespace:
    root = Path(source_path)
    rgb_root = root / getattr(args, "sapien_rgb_dir", "rgb")
    if not rgb_root.is_dir():
        raise FileNotFoundError(f"SAPIEN reader expected RGB folder at {rgb_root}")

    templates = _load_metadata_cameras(root, getattr(args, "sapien_metadata_dir", "metadata"))
    if not templates:
        templates = _load_colmap_camera_templates(root, args)
    if not templates:
        raise RuntimeError(
            "SAPIEN reader could not load camera intrinsics/poses. Expected metadata/cameras.json "
            "or a valid colmap/sparse folder."
        )

    cam_dirs = _selected_camera_dirs(rgb_root, getattr(args, "sapien_cameras", ""))
    if not cam_dirs:
        raise RuntimeError(f"No cam_* RGB folders found in {rgb_root}")

    train_cams: List[SimpleNamespace] = []
    test_cams: List[SimpleNamespace] = []
    uid = 0
    f_start = int(getattr(args, "sapien_frame_start", 0))
    f_end = int(getattr(args, "sapien_frame_end", -1))
    f_stride = max(int(getattr(args, "sapien_frame_stride", 1)), 1)
    test_stride = int(getattr(args, "sapien_test_stride", 0))
    fps = float(getattr(args, "fps", 30.0))

    for cam_dir in cam_dirs:
        cam_id = cam_dir.name
        if cam_id not in templates:
            print(f"[SAPIEN reader] Skipping {cam_id}: no camera metadata/pose found")
            continue
        tmpl = templates[cam_id]
        frames = sorted(cam_dir.glob("*.png")) + sorted(cam_dir.glob("*.jpg")) + sorted(cam_dir.glob("*.jpeg"))
        frames = sorted(frames, key=_frame_id_from_path)
        for img_path in frames:
            frame_id = _frame_id_from_path(img_path)
            if frame_id < f_start:
                continue
            if f_end >= 0 and frame_id > f_end:
                continue
            if (frame_id - f_start) % f_stride != 0:
                continue
            stem = img_path.stem
            width = int(tmpl.width)
            height = int(tmpl.height)
            if width <= 0 or height <= 0:
                with Image.open(img_path) as im:
                    width, height = im.size
            info = SimpleNamespace(
                uid=uid,
                R=tmpl.R,
                T=tmpl.T,
                FovY=float(tmpl.FovY),
                FovX=float(tmpl.FovX),
                image=None,
                image_path=str(img_path),
                depth_path=_depth_path_for(root, args, cam_id, stem),
                depth_params=None,
                image_name=f"{cam_id}_{stem}",
                width=width,
                height=height,
                is_test=False,
                camera_id=cam_id,
                frame_id=frame_id,
                timestamp=frame_id / max(fps, 1e-6),
                raw_timestamp=frame_id / max(fps, 1e-6),
                mask_path=str(root / "mask" / cam_id / f"{stem}.png") if (root / "mask" / cam_id / f"{stem}.png").exists() else "",
            )
            uid += 1
            if test_stride > 0 and ((frame_id - f_start) // f_stride) % test_stride == 0:
                info.is_test = True
                test_cams.append(info)
            else:
                train_cams.append(info)

    if not train_cams and test_cams:
        # Keep training usable even if test_stride accidentally selected all frames.
        train_cams, test_cams = test_cams, []

    pcd_path = Path(getattr(args, "sapien_pointcloud_path", "") or "")
    if not pcd_path:
        pcd_path = root / "pointcloud" / "points3D_parts.ply"
    if pcd_path.exists():
        point_cloud = _read_ply_as_basic_point_cloud(pcd_path)
        ply_path = str(pcd_path)
    else:
        colmap_info = None
        try:
            from scene.dataset_readers import sceneLoadTypeCallbacks
            colmap_info = sceneLoadTypeCallbacks["Colmap"](
                str(root / "colmap"), "images", getattr(args, "depths", ""), False, getattr(args, "train_test_exp", False)
            )
        except Exception as exc:
            raise FileNotFoundError(
                f"Could not find canonical point cloud at {pcd_path} and COLMAP fallback failed: {exc}"
            )
        point_cloud = colmap_info.point_cloud
        ply_path = colmap_info.ply_path

    out_ply = root / "metadata" / "sapien_dynamic_input.ply"
    try:
        if not out_ply.exists():
            _write_basic_point_cloud_ply(point_cloud, out_ply)
        ply_path_for_scene = str(out_ply)
    except Exception:
        ply_path_for_scene = ply_path

    nerf_norm = _normalization_from_cameras(train_cams + test_cams)
    print(
        f"[SAPIEN reader] Loaded {len(train_cams)} train views, {len(test_cams)} test views, "
        f"{len(cam_dirs)} camera folders, canonical points={len(point_cloud.points)}"
    )

    return SimpleNamespace(
        point_cloud=point_cloud,
        train_cameras=train_cams,
        test_cameras=test_cams,
        nerf_normalization=nerf_norm,
        ply_path=ply_path_for_scene,
        is_nerf_synthetic=False,
    )
