"""Build reusable static 3DGS endpoint scenes with GT part lineage.

This keeps GT labels on the initialization PLY.  Gaussian densification then
inherits labels in ``GaussianModel``; no dense post-hoc nearest-neighbor port is
needed.
"""

from __future__ import annotations

import csv
import json
import os
import re
from pathlib import Path

import numpy as np
import torch
from PIL import Image as PILImage
from plyfile import PlyData, PlyElement

from utils.read_write_model import Image, Point3D, qvec2rotmat, read_model, write_model


FRAME_RE = re.compile(r"frame_(\d+)\.")
GT_LINEAGE_SOURCE = "gaussian_model_inherited_sparse_parts"


def model_ply_path(model_dir: str | Path, iteration: int) -> Path:
    return Path(model_dir).expanduser().resolve() / "point_cloud" / f"iteration_{int(iteration)}" / "point_cloud.ply"


def scene_last_frame(scene: str | Path) -> int | None:
    frames: list[int] = []
    for path in (Path(scene) / "rgb").rglob("frame_*.*"):
        if path.suffix.lower() not in {".png", ".jpg", ".jpeg"}:
            continue
        match = FRAME_RE.search(path.name)
        if match:
            frames.append(int(match.group(1)))
    return max(frames) if frames else None


def has_sparse_gt_parts(scene: str | Path) -> bool:
    return (Path(scene).expanduser().resolve() / "pointcloud" / "points3D_parts.ply").is_file()


def has_inherited_gt_labels(model_or_ply: str | Path, iteration: int | None = None) -> bool:
    ply = (
        model_ply_path(model_or_ply, int(iteration))
        if iteration is not None
        else Path(model_or_ply).expanduser().resolve()
    )
    if not ply.is_file():
        return False
    sidecar = ply.parent / "gt_parts.pt"
    if sidecar.is_file():
        try:
            obj = torch.load(sidecar, map_location="cpu", weights_only=False)
            if isinstance(obj, dict) and obj.get("source") == GT_LINEAGE_SOURCE:
                labels = torch.as_tensor(obj.get("gt_part_labels", []))
                return labels.ndim == 1 and int(labels.numel()) > 0
        except Exception:
            return False
    return False


def _find_rgb_frame(rgb_root: Path, cam_name: str, frame_idx: int) -> Path | None:
    frame_str = f"frame_{int(frame_idx):06d}"
    cam_dir = rgb_root / cam_name
    for ext in (".png", ".jpg", ".jpeg", ".PNG", ".JPG", ".JPEG"):
        candidate = cam_dir / f"{frame_str}{ext}"
        if candidate.is_file():
            return candidate
    matches = sorted(
        path
        for path in cam_dir.glob(f"{frame_str}.*")
        if path.suffix.lower() in {".png", ".jpg", ".jpeg"}
    )
    return matches[0] if matches else None


def _camera_name(image_name: str) -> str:
    parts = str(image_name).split("/")
    if len(parts) > 1 and parts[0]:
        return parts[0]
    match = re.search(r"cam_\d+", str(image_name))
    if match is None:
        raise ValueError(f"Cannot infer camera name from COLMAP image name {image_name!r}")
    return match.group(0)


def _load_sparse_parts(scene: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    path = scene / "pointcloud" / "points3D_parts.ply"
    if not path.is_file():
        raise FileNotFoundError(f"Missing sparse GT part PLY: {path}")
    vertex = PlyData.read(str(path))["vertex"].data
    names = set(vertex.dtype.names or ())
    required = {"x", "y", "z", "red", "green", "blue", "part"}
    missing = sorted(required - names)
    if missing:
        raise RuntimeError(f"{path} is missing fields: {missing}")
    xyz = np.stack((vertex["x"], vertex["y"], vertex["z"]), axis=1).astype(np.float64)
    rgb = np.stack((vertex["red"], vertex["green"], vertex["blue"]), axis=1).astype(np.uint8)
    parts = np.asarray(vertex["part"], dtype=np.int32)
    return xyz, rgb, parts


def _joint_values(scene: Path, frame_idx: int) -> tuple[dict, float, float]:
    metadata_path = scene / "pointcloud" / "joint_metadata.json"
    values_path = scene / "metadata" / "frame_values.csv"
    if not metadata_path.is_file() or not values_path.is_file():
        raise FileNotFoundError(
            f"Missing joint metadata/frame_values for endpoint GT transport in {scene}"
        )
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    joint_name = str(metadata["name"])
    values: dict[int, float] = {}
    with values_path.open("r", encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames is None or joint_name not in reader.fieldnames:
            raise RuntimeError(f"{values_path} has no column for joint {joint_name!r}")
        for row in reader:
            values[int(row["frame_index"])] = float(row[joint_name])
    if 0 not in values or int(frame_idx) not in values:
        raise RuntimeError(f"Missing frame 0 or frame {int(frame_idx)} in {values_path}")
    return metadata, values[0], values[int(frame_idx)]


def _rotate(points: np.ndarray, origin: np.ndarray, axis: np.ndarray, angle: float) -> np.ndarray:
    axis = axis / max(float(np.linalg.norm(axis)), 1.0e-12)
    x, y, z = axis.tolist()
    skew = np.array([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]], dtype=np.float64)
    rot = np.eye(3, dtype=np.float64) + np.sin(angle) * skew + (1.0 - np.cos(angle)) * (skew @ skew)
    return origin[None] + (points - origin[None]) @ rot.T


def transport_sparse_gt_parts(scene: str | Path, frame_idx: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    scene = Path(scene).expanduser().resolve()
    xyz0, rgb0, parts = _load_sparse_parts(scene)
    xyz = xyz0.copy()
    moving = parts != 0
    summary = {
        "source_sparse_ply": str(scene / "pointcloud" / "points3D_parts.ply"),
        "frame": int(frame_idx),
        "num_points": int(xyz.shape[0]),
        "part_counts": {str(int(p)): int((parts == int(p)).sum()) for p in np.unique(parts)},
        "moving_rule": "part != 0",
    }
    if bool(moving.any()):
        metadata, q0, qt = _joint_values(scene, int(frame_idx))
        delta = float(qt - q0)
        axis = np.asarray(metadata["axis_vector"], dtype=np.float64)
        axis = axis / max(float(np.linalg.norm(axis)), 1.0e-12)
        origin = np.asarray(metadata["point_of_application"], dtype=np.float64)
        joint_type = str(metadata.get("type", "")).lower()
        if joint_type in {"revolute", "continuous"}:
            xyz[moving] = _rotate(xyz[moving], origin, axis, delta)
        elif joint_type == "prismatic":
            xyz[moving] = xyz[moving] + delta * axis[None]
        else:
            raise RuntimeError(f"Unsupported joint type for endpoint GT transport: {joint_type!r}")
        summary.update(
            {
                "joint_name": str(metadata["name"]),
                "joint_type": joint_type,
                "frame0_value": q0,
                "frame_value": qt,
                "joint_delta": delta,
                "axis_vector": axis.tolist(),
                "point_of_application": origin.tolist(),
            }
        )
    return xyz.astype(np.float32), rgb0, parts, summary


def _camera_params(camera) -> tuple[float, float, float, float]:
    params = np.asarray(camera.params, dtype=np.float64)
    if str(camera.model) == "SIMPLE_PINHOLE":
        return float(params[0]), float(params[0]), float(params[1]), float(params[2])
    if str(camera.model) == "PINHOLE":
        return float(params[0]), float(params[1]), float(params[2]), float(params[3])
    raise RuntimeError(f"Unsupported camera model for endpoint color projection: {camera.model}")


def _read_rgb(path: Path, width: int, height: int) -> np.ndarray:
    image = PILImage.open(path).convert("RGB")
    if image.size != (int(width), int(height)):
        image = image.resize((int(width), int(height)), PILImage.BILINEAR)
    return np.asarray(image, dtype=np.uint8)


def _read_depth(path: Path, width: int, height: int) -> np.ndarray | None:
    if not path.is_file():
        return None
    depth = np.load(str(path)).astype(np.float32)
    if depth.shape != (int(height), int(width)):
        return None
    return depth


def colorize_points_from_frame(
    scene: str | Path,
    xyz: np.ndarray,
    fallback_rgb: np.ndarray,
    frame_idx: int,
    *,
    depth_abs_tol: float = 0.04,
    depth_rel_tol: float = 0.03,
) -> tuple[np.ndarray, np.ndarray]:
    scene = Path(scene).expanduser().resolve()
    cameras, images, _ = read_model(str(scene / "colmap" / "sparse" / "0"), ext=".bin")
    rgb_root = scene / "rgb"
    depth_root = scene / "depth_npy"
    rgb_sum = np.zeros((xyz.shape[0], 3), dtype=np.float64)
    rgb_count = np.zeros((xyz.shape[0],), dtype=np.int32)
    fallback_sum = np.zeros_like(rgb_sum)
    fallback_count = np.zeros_like(rgb_count)

    for image in images.values():
        cam_name = _camera_name(image.name)
        rgb_path = _find_rgb_frame(rgb_root, cam_name, int(frame_idx))
        if rgb_path is None:
            continue
        camera = cameras[image.camera_id]
        width, height = int(camera.width), int(camera.height)
        rgb = _read_rgb(rgb_path, width, height)
        depth = _read_depth(depth_root / cam_name / f"frame_{int(frame_idx):06d}.npy", width, height)
        fx, fy, cx, cy = _camera_params(camera)
        rot = qvec2rotmat(image.qvec)
        cam_xyz = xyz.astype(np.float64) @ rot.T + np.asarray(image.tvec, dtype=np.float64)[None]
        z = cam_xyz[:, 2]
        valid_z = z > 1.0e-8
        u = np.rint(fx * cam_xyz[:, 0] / np.maximum(z, 1.0e-8) + cx).astype(np.int64)
        v = np.rint(fy * cam_xyz[:, 1] / np.maximum(z, 1.0e-8) + cy).astype(np.int64)
        in_bounds = valid_z & (u >= 0) & (u < width) & (v >= 0) & (v < height)
        if not bool(in_bounds.any()):
            continue
        idx = np.nonzero(in_bounds)[0]
        sampled = rgb[v[idx], u[idx], :3].astype(np.float64)
        fallback_sum[idx] += sampled
        fallback_count[idx] += 1
        if depth is not None:
            observed = depth[v[idx], u[idx]].astype(np.float64)
            good = np.isfinite(observed) & (observed > 0.0)
            good &= np.abs(observed - z[idx]) <= np.maximum(float(depth_abs_tol), float(depth_rel_tol) * np.maximum(z[idx], 1.0))
            if bool(good.any()):
                good_idx = idx[good]
                rgb_sum[good_idx] += sampled[good]
                rgb_count[good_idx] += 1

    out = np.asarray(fallback_rgb, dtype=np.float64).copy()
    use_fallback = rgb_count == 0
    if bool((~use_fallback).any()):
        out[~use_fallback] = rgb_sum[~use_fallback] / rgb_count[~use_fallback, None]
    use_projected = use_fallback & (fallback_count > 0)
    if bool(use_projected.any()):
        out[use_projected] = fallback_sum[use_projected] / fallback_count[use_projected, None]
    return np.clip(np.rint(out), 0, 255).astype(np.uint8), rgb_count


def write_labeled_points_ply(path: str | Path, xyz: np.ndarray, rgb: np.ndarray, parts: np.ndarray) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    dtype = [
        ("x", "f4"), ("y", "f4"), ("z", "f4"),
        ("nx", "f4"), ("ny", "f4"), ("nz", "f4"),
        ("red", "u1"), ("green", "u1"), ("blue", "u1"),
        ("part", "i4"),
    ]
    elements = np.empty(xyz.shape[0], dtype=dtype)
    normals = np.zeros_like(xyz, dtype=np.float32)
    elements["x"], elements["y"], elements["z"] = xyz[:, 0], xyz[:, 1], xyz[:, 2]
    elements["nx"], elements["ny"], elements["nz"] = normals[:, 0], normals[:, 1], normals[:, 2]
    elements["red"], elements["green"], elements["blue"] = rgb[:, 0], rgb[:, 1], rgb[:, 2]
    elements["part"] = parts.astype(np.int32)
    PlyData([PlyElement.describe(elements, "vertex")]).write(str(path))


def link_endpoint_depth(scene, output_scene, frame_idx) -> int:
    """Link frame `frame_idx`'s depth into a frame_t scene, beside colmap/.

    Idempotent and separate from the scene build so that an ALREADY BUILT
    endpoint scene picks the depth up too. Without that, every frame_t scene
    made before depth supervision worked would keep training on RGB alone
    while frame 0 got depth -- the two ends of the sequence fitted under
    different supervision, which is the asymmetry the endpoint pair exists to
    avoid.
    """
    scene = Path(scene).expanduser().resolve()
    output_scene = Path(output_scene).expanduser().resolve()
    images_dir = output_scene / "colmap" / "images"
    depth_dir = output_scene / "depth_npy"
    if not images_dir.is_dir():
        return 0
    n = 0
    for cam_dir in sorted(p for p in images_dir.iterdir() if p.is_dir()):
        for img in sorted(cam_dir.iterdir()):
            src = scene / "depth_npy" / cam_dir.name / f"{img.stem}.npy"
            if not src.is_file():
                continue
            dst = depth_dir / cam_dir.name / f"{img.stem}.npy"
            dst.parent.mkdir(parents=True, exist_ok=True)
            if dst.exists() or dst.is_symlink():
                dst.unlink()
            dst.symlink_to(os.path.relpath(src.resolve(), dst.parent.resolve()))
            n += 1
    return n


def build_labeled_frame_static_scene(
    scene: str | Path,
    output_scene: str | Path,
    frame_idx: int,
    *,
    force: bool = False,
) -> Path:
    scene = Path(scene).expanduser().resolve()
    output_scene = Path(output_scene).expanduser().resolve()
    colmap_root = output_scene / "colmap"
    sparse = colmap_root / "sparse" / "0"
    summary_path = output_scene / "frame_static_scene_summary.json"
    ply = sparse / "points3D.ply"
    if ply.is_file() and summary_path.is_file() and not force:
        # built already, but still make sure its depth is linked
        link_endpoint_depth(scene, output_scene, int(frame_idx))
        return output_scene

    source_sparse = scene / "colmap" / "sparse" / "0"
    cameras, images, _ = read_model(str(source_sparse), ext=".bin")
    xyz, fallback_rgb, parts, summary = transport_sparse_gt_parts(scene, int(frame_idx))
    rgb, color_counts = colorize_points_from_frame(scene, xyz, fallback_rgb, int(frame_idx))

    sparse.mkdir(parents=True, exist_ok=True)
    images_dir = colmap_root / "images"
    images_dir.mkdir(parents=True, exist_ok=True)
    rgb_root = scene / "rgb"
    new_images: dict[int, Image] = {}
    used_camera_ids: set[int] = set()
    for image_id, image in sorted(images.items()):
        cam_name = _camera_name(image.name)
        rgb_path = _find_rgb_frame(rgb_root, cam_name, int(frame_idx))
        if rgb_path is None:
            continue
        rel_name = f"{cam_name}/{rgb_path.name}"
        dst = images_dir / rel_name
        dst.parent.mkdir(parents=True, exist_ok=True)
        if dst.exists() or dst.is_symlink():
            dst.unlink()
        dst.symlink_to(os.path.relpath(rgb_path.resolve(), dst.parent.resolve()))

        used_camera_ids.add(int(image.camera_id))
        new_images[int(image_id)] = Image(
            id=int(image_id),
            qvec=np.asarray(image.qvec).copy(),
            tvec=np.asarray(image.tvec).copy(),
            camera_id=int(image.camera_id),
            name=rel_name,
            xys=np.empty((0, 2), dtype=np.float64),
            point3D_ids=np.empty((0,), dtype=np.int64),
        )
    if not new_images:
        raise RuntimeError(f"No RGB cameras found for frame {int(frame_idx)} in {scene / 'rgb'}")
    kept_cameras = {cid: camera for cid, camera in cameras.items() if int(cid) in used_camera_ids}
    points3d = {
        int(i + 1): Point3D(
            id=int(i + 1),
            xyz=xyz[i].astype(np.float64),
            rgb=rgb[i].astype(np.uint8),
            error=0.0,
            image_ids=np.empty((0,), dtype=np.int32),
            point2D_idxs=np.empty((0,), dtype=np.int32),
        )
        for i in range(xyz.shape[0])
    }
    write_model(kept_cameras, new_images, points3d, str(sparse), ext=".bin")
    write_model(kept_cameras, new_images, points3d, str(sparse), ext=".txt")
    link_endpoint_depth(scene, output_scene, int(frame_idx))
    write_labeled_points_ply(ply, xyz, rgb, parts)
    write_labeled_points_ply(sparse / "points3D_parts.ply", xyz, rgb, parts)

    summary.update(
        {
            "schema_version": 1,
            "output_scene": str(output_scene),
            "colmap_root": str(colmap_root),
            "points3D_ply": str(ply),
            "num_cameras": len(new_images),
            "color_depth_consistent_points": int((color_counts > 0).sum()),
            "gt_label_source": "transported_sparse_points3D_parts_ply",
            "gt_lineage_ready": True,
        }
    )
    summary_path.write_text(json.dumps(summary, indent=2, allow_nan=False), encoding="utf-8")
    return output_scene
