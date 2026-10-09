#!/usr/bin/env python3
"""Pure-depth rigid-part discovery using relative-distance Hungarian matching.

The estimator in this file deliberately has no image-feature path.  RGB is
opened only by the debug renderer after all matches and labels have been
computed. During estimation the calibrated COLMAP model supplies only fixed
K, Rcw, tcw. COLMAP is never executed; its seed PLY is opened only after the
labels are frozen, to write the labelled 3DGS handoff artifact.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath

import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.sparse import coo_matrix, csr_matrix
from scipy.sparse.csgraph import min_weight_full_bipartite_matching
from scipy.sparse.linalg import ArpackNoConvergence, eigsh
from scipy.spatial.distance import cdist
from scipy.spatial import cKDTree

try:
    import torch
except Exception as exc:  # pragma: no cover - environment diagnostic
    raise RuntimeError(
        "PyTorch is required only for the small contrastive optimization; "
        "run this script in the rigeuler3dgs environment."
    ) from exc
from tqdm import tqdm

from utils.read_write_model import (
    read_cameras_binary,
    read_cameras_text,
    read_images_binary,
    read_images_text,
)


EPS = 1e-8
DEFAULT_OUTPUT_SUFFIX = "_pure_depth_geometry"
CONFIDENT_LABEL_THRESHOLD = 0.75
LABEL_COLORS = {
    "STATIC": (40, 190, 40),
    "MOVING_1": (30, 30, 235),
    "MOVING_2": (220, 80, 30),
    "MOVING_3": (210, 40, 210),
    "AMBIGUOUS": (90, 190, 220),
}


@dataclass(frozen=True)
class Calibration:
    name: str
    width: int
    height: int
    fx: float
    fy: float
    cx: float
    cy: float
    Rcw: np.ndarray
    tcw: np.ndarray

    @property
    def center_world(self) -> np.ndarray:
        return -self.Rcw.T @ self.tcw


@dataclass
class FrameCloud:
    frame: int
    raw_xyz: np.ndarray
    raw_camera_ids: np.ndarray
    xyz: np.ndarray
    num_raw_members: np.ndarray
    num_support_cameras: np.ndarray
    support_camera_ids: list[tuple[int, ...]]
    voxel_indices: np.ndarray
    spatial_spread: np.ndarray
    quality: dict


@dataclass
class Track:
    track_id: int
    points: dict[int, np.ndarray] = field(default_factory=dict)
    point_ids: dict[int, int] = field(default_factory=dict)
    match_confidences: list[float] = field(default_factory=list)
    cluster_id: int = -1
    confidence: float = 0.0
    probabilities: np.ndarray | None = None
    label: str = "AMBIGUOUS"
    initial_cluster_id: int = -1
    initial_confidence: float = 0.0
    initial_label: str = "AMBIGUOUS"
    label_source: str = "unresolved"
    ghost_source_track_id: int = -1
    ghost_distance: float = math.nan
    ghost_age_frames: int = 0
    temporal_matches: int = 0
    static_anchor_matches: int = 0
    static_prior: float = 0.5
    static_prior_weight: float = 0.0
    stationary_occupancy_ambiguous: bool = False
    knn_teacher_count: int = 0
    temporal_trajectory_id: int = -1
    temporal_coherence_links: int = 0
    seed_type: str = "none"
    seed_strength: float = 0.0
    dynamic_static_relative_score: float = 0.5
    dynamic_static_relative_support: int = 0
    dynamic_probability: float = 0.5
    dynamic_anchor: bool = False


@dataclass
class ResidualCloud:
    """Fine geometric residual after conservative STATIC removal."""

    frame: int
    xyz: np.ndarray
    num_raw_members: np.ndarray
    coarse_point_ids: np.ndarray
    quality: dict


@dataclass
class ResidualGraph:
    """One multi-scale, purely geometric KNN graph for a residual frame."""

    frame: int
    xyz: np.ndarray
    neighbors: np.ndarray
    edge_lengths: np.ndarray
    edge_scales: np.ndarray
    signatures: np.ndarray
    spacing: float


def frame_number(path: Path) -> int:
    return int(path.stem.rsplit("_", 1)[-1])


def robust_mad(values: np.ndarray | list[float]) -> float:
    a = np.asarray(values, np.float64)
    if not a.size:
        return 0.0
    med = np.median(a)
    return float(1.4826 * np.median(np.abs(a - med)))


def camera_name(image_name: str) -> str:
    for part in PurePosixPath(image_name).parts:
        if part.startswith("cam_"):
            return part
    raise RuntimeError(f"No cam_XXX component in calibrated image name: {image_name}")


def find_calibration_model(root: Path) -> Path:
    candidates = (
        root / "colmap" / "sparse" / "0",
        root / "colmap" / "sparse",
        root / "sparse" / "0",
        root / "sparse",
    )
    for path in candidates:
        if ((path / "cameras.bin").is_file() and (path / "images.bin").is_file()) or (
            (path / "cameras.txt").is_file() and (path / "images.txt").is_file()
        ):
            return path
    raise FileNotFoundError("Could not find the trusted calibrated model under ROOT/colmap/sparse/0")


def load_calibrations(root: Path) -> dict[str, Calibration]:
    model_path = find_calibration_model(root)
    # Do not even parse points3D: this pipeline uses the existing model solely
    # as a container for fixed camera calibration.
    if (model_path / "cameras.bin").is_file():
        cameras = read_cameras_binary(str(model_path / "cameras.bin"))
        images = read_images_binary(str(model_path / "images.bin"))
    else:
        cameras = read_cameras_text(str(model_path / "cameras.txt"))
        images = read_images_text(str(model_path / "images.txt"))
    result = {}
    for image in images.values():
        name = camera_name(image.name)
        if name in result:
            raise RuntimeError(f"Calibration model contains more than one pose for physical camera {name}")
        camera = cameras[image.camera_id]
        if camera.model == "PINHOLE":
            fx, fy, cx, cy = map(float, camera.params[:4])
        elif camera.model == "SIMPLE_PINHOLE":
            fx, cx, cy = map(float, camera.params[:3])
            fy = fx
        else:
            raise RuntimeError(
                f"Depth backprojection currently requires an undistorted PINHOLE model; "
                f"{name} uses {camera.model}"
            )
        result[name] = Calibration(
            name=name,
            width=int(camera.width),
            height=int(camera.height),
            fx=fx,
            fy=fy,
            cx=cx,
            cy=cy,
            Rcw=image.qvec2rotmat().astype(np.float64),
            tcw=np.asarray(image.tvec, np.float64),
        )
    if not result:
        raise RuntimeError(f"No calibrated cameras in {model_path}")
    print("[CALIBRATION] Xc = Rcw Xw + tcw; Xw = Rcw^T (Xc - tcw)")
    for name, calibration in sorted(result.items()):
        center = calibration.center_world
        print(
            f"  {name} fx={calibration.fx:.6f} fy={calibration.fy:.6f} "
            f"cx={calibration.cx:.3f} cy={calibration.cy:.3f} "
            f"center_world=({center[0]:.6f}, {center[1]:.6f}, {center[2]:.6f})"
        )
    return dict(sorted(result.items()))


def depth_extension(depth_root: Path) -> str:
    if any(depth_root.glob("cam_*/frame_*.npy")):
        return ".npy"
    if any(depth_root.glob("cam_*/frame_*.png")):
        return ".png"
    raise FileNotFoundError(f"No frame_*.npy or frame_*.png depth files under {depth_root}")


def select_frames(
    depth_root: Path,
    camera_names: list[str],
    frame_start: int,
    frame_stride: int,
    num_frames: int | None,
) -> tuple[list[int], str]:
    extension = depth_extension(depth_root)
    common = None
    for name in camera_names:
        folder = depth_root / name
        if not folder.is_dir():
            raise FileNotFoundError(f"Depth folder missing for calibrated camera: {folder}")
        frames = {frame_number(p) for p in folder.glob(f"frame_*{extension}")}
        common = frames if common is None else common & frames
    chosen = [f for f in sorted(common or ()) if f >= frame_start][::frame_stride]
    if num_frames is not None:
        chosen = chosen[:num_frames]
    if len(chosen) < 2:
        raise RuntimeError("At least two common selected depth frames are required")
    print(
        f"[DATASET] depth={depth_root} format={extension} cameras={len(camera_names)} "
        f"common_frames={len(common or ())} selected={len(chosen)} "
        f"range={chosen[0]:06d}..{chosen[-1]:06d}"
    )
    return chosen, extension


def read_depth(path: Path, scale: float) -> np.ndarray:
    if path.suffix == ".npy":
        depth = np.load(path)
    else:
        depth = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        if depth is None:
            raise RuntimeError(f"Could not read {path}")
    if depth.ndim != 2:
        raise RuntimeError(f"Expected one-channel depth at {path}, got {depth.shape}")
    return np.asarray(depth, np.float64) * scale


def mask_path(root: Path, camera_name_: str, frame: int) -> Path | None:
    folder = root / "mask" / camera_name_
    for suffix in (".png", ".npy"):
        path = folder / f"frame_{frame:06d}{suffix}"
        if path.is_file():
            return path
    return None


def read_mask(path: Path, shape: tuple[int, int]) -> np.ndarray:
    if path.suffix == ".npy":
        mask = np.load(path)
    else:
        mask = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if mask is None:
        raise RuntimeError(f"Could not read mask {path}")
    if mask.ndim == 3:
        mask = np.any(mask != 0, axis=2)
    if tuple(mask.shape) != tuple(shape):
        raise RuntimeError(f"Mask/depth shape mismatch at {path}: {mask.shape} vs {shape}")
    return mask != 0


def backproject_depth_world(
    depth: np.ndarray,
    calibration: Calibration,
    validity: np.ndarray,
) -> np.ndarray:
    """Backproject camera-space Z depth through trusted world-to-camera pose."""
    v, u = np.nonzero(validity)
    z = depth[v, u]
    camera_xyz = np.column_stack(
        ((u - calibration.cx) * z / calibration.fx,
         (v - calibration.cy) * z / calibration.fy,
         z)
    )
    # Row-vector form of Rcw.T @ (Xc - tcw).
    return (camera_xyz - calibration.tcw) @ calibration.Rcw


def occupied_voxel_count(xyz: np.ndarray, voxel_size: float) -> int:
    return len(np.unique(np.floor(xyz / voxel_size).astype(np.int64), axis=0))


def choose_global_voxel_size(xyz: np.ndarray, minimum_size: float, target_points: int) -> float:
    """Choose one coarse world lattice once; never select points per frame.

    The target is a complexity budget measured on the first selected frame,
    not a hard per-frame truncation.
    """
    if target_points <= 0:
        return minimum_size
    candidate = minimum_size
    for _ in range(96):
        if occupied_voxel_count(xyz, candidate) <= target_points:
            return candidate
        candidate *= 1.08
    raise RuntimeError(
        f"Could not reduce the first frame below {target_points} occupied voxels; "
        "increase --target-points or --voxel-size"
    )


def voxel_reduce(
    xyz: np.ndarray,
    camera_ids: np.ndarray,
    voxel_size: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[tuple[int, ...]], np.ndarray, np.ndarray]:
    # The origin is intentionally fixed at world (0,0,0) for every frame.
    voxels = np.floor(xyz / voxel_size).astype(np.int64)
    unique_voxels, inverse = np.unique(voxels, axis=0, return_inverse=True)
    order = np.argsort(inverse, kind="stable")
    counts = np.bincount(inverse, minlength=len(unique_voxels)).astype(np.int32)
    offsets = np.concatenate(([0], np.cumsum(counts)))
    representatives = np.empty((len(unique_voxels), 3), np.float64)
    spreads = np.empty(len(unique_voxels), np.float64)
    support_ids: list[tuple[int, ...]] = []
    for voxel_id in range(len(unique_voxels)):
        members = order[offsets[voxel_id]:offsets[voxel_id + 1]]
        representative = np.median(xyz[members], axis=0)
        representatives[voxel_id] = representative
        spreads[voxel_id] = float(np.sqrt(np.mean(np.sum((xyz[members] - representative) ** 2, axis=1))))
        support_ids.append(tuple(int(x) for x in np.unique(camera_ids[members])))
    return (
        representatives,
        counts,
        np.asarray([len(ids) for ids in support_ids], np.int16),
        support_ids,
        unique_voxels,
        spreads,
    )


def write_ply(path: Path, xyz: np.ndarray, properties: dict[str, np.ndarray] | None = None) -> None:
    properties = properties or {}
    dtype_fields = [("x", "<f4"), ("y", "<f4"), ("z", "<f4")]
    for name, values in properties.items():
        values = np.asarray(values)
        if values.dtype.kind == "f":
            dtype = "<f4"
        elif values.dtype.itemsize == 1:
            dtype = "i1" if values.dtype.kind == "i" else "u1"
        elif values.dtype.itemsize <= 2:
            dtype = "<i2" if values.dtype.kind == "i" else "<u2"
        else:
            dtype = "<i4" if values.dtype.kind == "i" else "<u4"
        dtype_fields.append((name, dtype))
    data = np.empty(len(xyz), dtype=np.dtype(dtype_fields))
    data["x"], data["y"], data["z"] = np.asarray(xyz, np.float32).T
    for name, values in properties.items():
        data[name] = values
    lines = ["ply", "format binary_little_endian 1.0", f"element vertex {len(data)}"]
    ply_types = {
        ("f", 4): "float", ("i", 1): "char", ("u", 1): "uchar",
        ("i", 2): "short", ("u", 2): "ushort",
        ("i", 4): "int", ("u", 4): "uint",
    }
    for name in data.dtype.names:
        dtype = data.dtype.fields[name][0]
        lines.append(f"property {ply_types[(dtype.kind, dtype.itemsize)]} {name}")
    lines.append("end_header")
    with path.open("wb") as stream:
        stream.write(("\n".join(lines) + "\n").encode("ascii"))
        data.tofile(stream)


def export_frame_cloud(out: Path, cloud: FrameCloud, save_raw_ply: bool = False) -> None:
    folder = out / "depth_models" / f"frame_{cloud.frame:06d}"
    folder.mkdir(parents=True, exist_ok=True)
    if save_raw_ply:
        write_ply(folder / "points_raw_world.ply", cloud.raw_xyz, {"camera_id": cloud.raw_camera_ids})
    write_ply(
        folder / "points_world.ply",
        cloud.xyz,
        {
            "point_id": np.arange(len(cloud.xyz), dtype=np.int32),
            "num_raw_members": cloud.num_raw_members,
            "num_support_cameras": cloud.num_support_cameras,
            "spatial_spread": cloud.spatial_spread,
            "voxel_x": cloud.voxel_indices[:, 0],
            "voxel_y": cloud.voxel_indices[:, 1],
            "voxel_z": cloud.voxel_indices[:, 2],
        },
    )
    with (folder / "points_world.txt").open("w") as stream:
        stream.write(
            "point_id X Y Z num_raw_members num_support_cameras spatial_spread "
            "voxel_x voxel_y voxel_z support_camera_ids\n"
        )
        for point_id, point in enumerate(cloud.xyz):
            voxel = cloud.voxel_indices[point_id]
            support = ",".join(map(str, cloud.support_camera_ids[point_id]))
            stream.write(
                f"{point_id} {point[0]:.9g} {point[1]:.9g} {point[2]:.9g} "
                f"{int(cloud.num_raw_members[point_id])} {int(cloud.num_support_cameras[point_id])} "
                f"{cloud.spatial_spread[point_id]:.9g} {voxel[0]} {voxel[1]} {voxel[2]} {support}\n"
            )
    (folder / "quality.json").write_text(json.dumps(cloud.quality, indent=2) + "\n")


def dynamic_labels_for_frame(cloud: FrameCloud, tracks: dict[int, Track]) -> np.ndarray:
    """Return 0=confident static, 0.5=uncertain, 1=confident dynamic."""
    dynamic = np.full(len(cloud.xyz), 0.5, dtype=np.float32)
    claimed = np.zeros(len(cloud.xyz), dtype=bool)
    for track in tracks.values():
        point_id = track.point_ids.get(cloud.frame)
        if point_id is None:
            continue
        if point_id < 0 or point_id >= len(dynamic) or claimed[point_id]:
            raise RuntimeError(f"Invalid or duplicate point ownership in frame {cloud.frame}: {point_id}")
        claimed[point_id] = True
        if track.label == "STATIC":
            dynamic[point_id] = 0.0
        elif track.label.startswith("MOVING_"):
            dynamic[point_id] = 1.0
    return dynamic


def dynamic_probabilities_for_frame(cloud: FrameCloud, tracks: dict[int, Track]) -> np.ndarray:
    """Return the continuous optimized dynamic/static embedding before label gating."""
    probability = np.full(len(cloud.xyz), 0.5, dtype=np.float32)
    claimed = np.zeros(len(cloud.xyz), dtype=bool)
    for track in tracks.values():
        point_id = track.point_ids.get(cloud.frame)
        if point_id is None:
            continue
        if point_id < 0 or point_id >= len(probability) or claimed[point_id]:
            raise RuntimeError(f"Invalid or duplicate point ownership in frame {cloud.frame}: {point_id}")
        claimed[point_id] = True
        probability[point_id] = float(track.dynamic_probability)
    return probability


def export_initial_point_labels(
    root: Path,
    out: Path,
    cloud: FrameCloud,
    tracks: dict[int, Track],
) -> dict:
    """Export learned labels on depth anchors and on the canonical 3DGS seed cloud.

    The COLMAP PLY is opened only after optimization has finished. Only its
    geometry, normals, and RGB are copied; no semantic/part property is read.
    """
    dynamic = dynamic_labels_for_frame(cloud, tracks)
    dynamic_probability = dynamic_probabilities_for_frame(cloud, tracks)
    depth_path = out / "initial_depth_anchors_with_dynamic.ply"
    write_ply(
        depth_path,
        cloud.xyz,
        {
            "point_id": np.arange(len(cloud.xyz), dtype=np.int32),
            "num_raw_members": cloud.num_raw_members,
            "num_support_cameras": cloud.num_support_cameras,
            "dynamic": dynamic,
            "dynamic_probability_optimized": dynamic_probability,
        },
    )

    model_path = find_calibration_model(root)
    seed_path = model_path / "points3D.ply"
    if not seed_path.is_file():
        raise FileNotFoundError(
            f"The initial 3DGS seed PLY is required for labelled export: {seed_path}"
        )
    from plyfile import PlyData

    vertex = PlyData.read(str(seed_path))["vertex"].data
    names = set(vertex.dtype.names or ())
    seed_xyz = np.stack([np.asarray(vertex[name], np.float64) for name in ("x", "y", "z")], axis=1)
    distances, nearest = cKDTree(cloud.xyz).query(seed_xyz, k=1)
    nearest = np.asarray(nearest, np.int64)
    seed_dynamic = dynamic[nearest]
    seed_dynamic_probability = dynamic_probability[nearest]
    copied = {}
    for name in ("nx", "ny", "nz", "red", "green", "blue"):
        if name in names:
            copied[name] = np.asarray(vertex[name])
    copied["dynamic"] = seed_dynamic
    copied["dynamic_probability_optimized"] = seed_dynamic_probability
    labelled_seed_path = out / "initial_point_cloud_with_dynamic.ply"
    write_ply(labelled_seed_path, seed_xyz, copied)

    result = {
        "label_semantics": (
            "dynamic=0.0 confident static, 0.5 uncertain, 1.0 confident dynamic; "
            "dynamic_probability_optimized is the continuous pre-gating dynamic/static embedding"
        ),
        "depth_anchor_path": str(depth_path),
        "depth_anchor_points": int(len(dynamic)),
        "depth_static_anchors": int(np.count_nonzero(dynamic == 0.0)),
        "depth_uncertain": int(np.count_nonzero(dynamic == 0.5)),
        "depth_dynamic": int(np.count_nonzero(dynamic == 1.0)),
        "depth_dynamic_probability_mean": float(np.mean(dynamic_probability)),
        "depth_dynamic_probability_p10": float(np.percentile(dynamic_probability, 10)),
        "depth_dynamic_probability_p50": float(np.percentile(dynamic_probability, 50)),
        "depth_dynamic_probability_p90": float(np.percentile(dynamic_probability, 90)),
        "initial_3dgs_source": str(seed_path),
        "initial_3dgs_labelled_path": str(labelled_seed_path),
        "initial_3dgs_points": int(len(seed_dynamic)),
        "initial_3dgs_static_anchors": int(np.count_nonzero(seed_dynamic == 0.0)),
        "initial_3dgs_uncertain": int(np.count_nonzero(seed_dynamic == 0.5)),
        "initial_3dgs_dynamic": int(np.count_nonzero(seed_dynamic == 1.0)),
        "initial_3dgs_dynamic_probability_mean": float(np.mean(seed_dynamic_probability)),
        "initial_3dgs_dynamic_probability_p10": float(np.percentile(seed_dynamic_probability, 10)),
        "initial_3dgs_dynamic_probability_p50": float(np.percentile(seed_dynamic_probability, 50)),
        "initial_3dgs_dynamic_probability_p90": float(np.percentile(seed_dynamic_probability, 90)),
        "nearest_depth_anchor_distance_median": float(np.median(distances)),
        "nearest_depth_anchor_distance_max": float(np.max(distances)),
    }
    (out / "initial_point_cloud_label_summary.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def load_raw_frame(
    root: Path,
    depth_root: Path,
    extension: str,
    calibrations: dict[str, Calibration],
    frame: int,
    depth_scale: float,
    min_depth: float | None,
    max_depth: float | None,
    use_mask: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    camera_clouds = []
    camera_ids = []
    depth_values = []
    per_camera = {}
    for camera_id, (name, calibration) in enumerate(calibrations.items()):
        path = depth_root / name / f"frame_{frame:06d}{extension}"
        depth = read_depth(path, depth_scale)
        if depth.shape != (calibration.height, calibration.width):
            raise RuntimeError(f"Depth/calibration shape mismatch at {path}: {depth.shape}")
        raw_valid = np.isfinite(depth) & (depth > 0)
        if min_depth is not None:
            raw_valid &= depth >= min_depth
        if max_depth is not None:
            raw_valid &= depth <= max_depth
        before_mask = int(raw_valid.sum())
        valid = raw_valid
        used_mask_path = None
        if use_mask:
            used_mask_path = mask_path(root, name, frame)
            if used_mask_path is None:
                raise FileNotFoundError(f"--use-mask requested but no mask exists for {name} frame {frame}")
            valid = raw_valid & read_mask(used_mask_path, depth.shape)
        world = backproject_depth_world(depth, calibration, valid)
        camera_clouds.append(world)
        camera_ids.append(np.full(len(world), camera_id, np.int16))
        depth_values.append(depth[valid])
        per_camera[name] = {
            "total_pixels": int(depth.size),
            "valid_depth_pixels": before_mask,
            "masked_valid_pixels": int(valid.sum()),
            "mask_path": str(used_mask_path) if used_mask_path else None,
        }
    return (
        np.concatenate(camera_clouds, axis=0),
        np.concatenate(camera_ids),
        np.concatenate(depth_values),
        per_camera,
    )


def build_frame_clouds(
    root: Path,
    out: Path,
    depth_root: Path,
    extension: str,
    calibrations: dict[str, Calibration],
    frames: list[int],
    depth_scale: float,
    min_depth: float | None,
    max_depth: float | None,
    use_mask: bool,
    voxel_size: float,
    target_points: int,
    save_raw_ply: bool = False,
) -> dict[int, FrameCloud]:
    clouds = {}
    quality_rows = []
    first_raw = load_raw_frame(
        root, depth_root, extension, calibrations, frames[0], depth_scale,
        min_depth, max_depth, use_mask,
    )
    effective_voxel_size = choose_global_voxel_size(first_raw[0], voxel_size, target_points)
    first_count = occupied_voxel_count(first_raw[0], effective_voxel_size)
    print(
        f"[SAMPLING] fixed world lattice origin=(0,0,0) minimum_voxel={voxel_size:.6g} "
        f"effective_voxel={effective_voxel_size:.6g} first_frame_points={first_count} "
        f"target={target_points if target_points > 0 else 'disabled'}; per-frame FPS disabled"
    )
    for frame in frames:
        raw_xyz, raw_camera_ids, all_depth, per_camera = (
            first_raw if frame == frames[0] else load_raw_frame(
                root, depth_root, extension, calibrations, frame, depth_scale,
                min_depth, max_depth, use_mask,
            )
        )
        reduced = voxel_reduce(raw_xyz, raw_camera_ids, effective_voxel_size)
        xyz, members, supports, support_ids, voxels, spreads = reduced
        quality = {
            "frame": frame,
            "raw_valid_points": int(len(raw_xyz)),
            "downsampled_points": int(len(xyz)),
            "median_depth": float(np.median(all_depth)),
            "requested_minimum_voxel_size": voxel_size,
            "voxel_size": effective_voxel_size,
            "target_points": target_points,
            "sampling": "all occupied cells in one fixed world lattice; no FPS",
            "voxel_origin_world": [0.0, 0.0, 0.0],
            "median_voxel_members": float(np.median(members)),
            "median_spatial_spread": float(np.median(spreads)),
            "p90_spatial_spread": float(np.percentile(spreads, 90)),
            "support_1cam": int(np.sum(supports == 1)),
            "support_2cam": int(np.sum(supports == 2)),
            "support_3plus": int(np.sum(supports >= 3)),
            "per_camera": per_camera,
        }
        cloud = FrameCloud(
            frame, raw_xyz, raw_camera_ids, xyz, members, supports,
            support_ids, voxels, spreads, quality,
        )
        clouds[frame] = cloud
        export_frame_cloud(out, cloud, save_raw_ply)
        quality_rows.append({key: quality[key] for key in (
            "frame", "raw_valid_points", "downsampled_points", "median_depth",
            "voxel_size", "median_voxel_members", "median_spatial_spread", "p90_spatial_spread",
            "support_1cam", "support_2cam", "support_3plus",
        )})
        print(
            f"[DEPTH FRAME] {frame:06d}\n"
            f"  raw_valid_points={len(raw_xyz)}\n"
            f"  reduced_points={len(xyz)}\n"
            f"  median_depth={quality['median_depth']:.6f}\n"
            f"  median_spatial_spread={quality['median_spatial_spread']:.6g}\n"
            f"  support: 1cam={quality['support_1cam']} 2cam={quality['support_2cam']} "
            f"3+={quality['support_3plus']}"
        )
    write_csv(out / "depth_frame_quality.csv", quality_rows)
    return clouds


def write_csv(path: Path, rows: list[dict], fieldnames: list[str] | None = None) -> None:
    if fieldnames is None:
        fieldnames = list(rows[0]) if rows else []
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


# =============================================================================
# Pure relative-distance Hungarian matching
# =============================================================================

def sparse_graph_signatures(
    xyz: np.ndarray,
    neighbors: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """KNN signatures without an NxN distance matrix."""
    count = min(neighbors, max(1, len(xyz) - 1))
    radial, indices = cKDTree(xyz).query(xyz, k=count + 1)
    radial = np.asarray(radial[:, 1:], np.float64)
    neighbor_indices = np.asarray(indices[:, 1:], np.int32)
    upper = np.triu_indices(count + 1, 1)
    patch = np.empty((len(xyz), len(upper[0])), np.float64)
    for index, local_neighbors in enumerate(neighbor_indices):
        local = xyz[np.concatenate(([index], local_neighbors))]
        patch[index] = np.sort(np.linalg.norm(local[:, None] - local[None, :], axis=2)[upper])
    return radial, patch, neighbor_indices, max(float(np.median(radial[:, 0])), EPS)


def sparse_signature_candidates(
    signature_a: tuple[np.ndarray, np.ndarray, np.ndarray, float],
    signature_b: tuple[np.ndarray, np.ndarray, np.ndarray, float],
    candidates: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Use invariant local signatures only to propose a sparse edge set."""
    radial_a, patch_a, _neighbors_a, spacing_a = signature_a
    radial_b, patch_b, _neighbors_b, spacing_b = signature_b
    d_scale = max(0.125 * (spacing_a + spacing_b), EPS)

    def features(radial: np.ndarray, patch: np.ndarray) -> np.ndarray:
        radial_features = np.log1p(radial / d_scale) * math.sqrt(0.4 / radial.shape[1])
        patch_features = np.log1p(patch / d_scale) * math.sqrt(0.6 / patch.shape[1])
        return np.concatenate((radial_features, patch_features), axis=1)

    features_a, features_b = features(radial_a, patch_a), features(radial_b, patch_b)
    forward_count = min(candidates, len(features_b))
    reverse_count = min(candidates, len(features_a))
    forward = cKDTree(features_b).query(features_a, k=forward_count)[1]
    reverse = cKDTree(features_a).query(features_b, k=reverse_count)[1]
    if forward_count == 1:
        forward = forward[:, None]
    if reverse_count == 1:
        reverse = reverse[:, None]
    candidate_sets = [set(map(int, row)) for row in forward]
    for destination, sources in enumerate(reverse):
        for source in sources:
            candidate_sets[int(source)].add(destination)

    edge_i = np.concatenate([
        np.full(len(destinations), source, np.int32)
        for source, destinations in enumerate(candidate_sets)
    ])
    edge_j = np.concatenate([
        np.asarray(sorted(destinations), np.int32) for destinations in candidate_sets
    ])
    radial_left, radial_right = radial_a[edge_i], radial_b[edge_j]
    patch_left, patch_right = patch_a[edge_i], patch_b[edge_j]
    radial_cost = np.median(
        np.abs(radial_left - radial_right) /
        np.maximum(np.maximum(radial_left, radial_right), d_scale),
        axis=1,
    )
    patch_cost = np.median(
        np.abs(patch_left - patch_right) /
        np.maximum(np.maximum(patch_left, patch_right), d_scale),
        axis=1,
    )
    return edge_i, edge_j, 0.4 * radial_cost + 0.6 * patch_cost, np.asarray(
        [len(destinations) for destinations in candidate_sets], np.int32,
    )


def candidate_compatibility_graph(
    xyz_a: np.ndarray,
    xyz_b: np.ndarray,
    neighbors_a: np.ndarray,
    edge_i: np.ndarray,
    edge_j: np.ndarray,
    counts: np.ndarray,
    d_scale: float,
    threshold: float = 0.03,
) -> csr_matrix:
    """Connect candidate matches that preserve a local relative distance."""
    offsets = np.concatenate(([0], np.cumsum(counts)))
    source_pairs = {
        tuple(sorted((source, int(neighbor))))
        for source, local in enumerate(neighbors_a)
        for neighbor in local
        if source != int(neighbor)
    }
    graph_rows, graph_columns, graph_weights = [], [], []
    for source, neighbor in source_pairs:
        left_ids = np.arange(offsets[source], offsets[source + 1])
        right_ids = np.arange(offsets[neighbor], offsets[neighbor + 1])
        if not len(left_ids) or not len(right_ids):
            continue
        destinations_left = edge_j[left_ids]
        destinations_right = edge_j[right_ids]
        destination_distances = np.linalg.norm(
            xyz_b[destinations_left, None] - xyz_b[destinations_right][None, :], axis=2,
        )
        source_distance = float(np.linalg.norm(xyz_a[source] - xyz_a[neighbor]))
        residual = np.abs(source_distance - destination_distances) / np.maximum(
            np.maximum(source_distance, destination_distances), d_scale,
        )
        compatible = (residual <= threshold) & (
            destinations_left[:, None] != destinations_right[None, :]
        )
        left_index, right_index = np.nonzero(compatible)
        if not len(left_index):
            continue
        graph_rows.append(left_ids[left_index])
        graph_columns.append(right_ids[right_index])
        graph_weights.append(np.exp(-0.5 * (residual[left_index, right_index] / threshold) ** 2))
    if not graph_rows:
        return csr_matrix((len(edge_i), len(edge_i)), dtype=np.float64)
    rows = np.concatenate(graph_rows)
    columns = np.concatenate(graph_columns)
    weights = np.concatenate(graph_weights)
    return coo_matrix(
        (np.concatenate((weights, weights)),
         (np.concatenate((rows, columns)), np.concatenate((columns, rows)))),
        shape=(len(edge_i), len(edge_i)),
    ).tocsr()


def graph_matching_scores(
    signature_cost: np.ndarray,
    compatibility: csr_matrix,
    iterations: int,
) -> tuple[np.ndarray, np.ndarray]:
    cost_scale = max(float(np.median(signature_cost)) + robust_mad(signature_cost), 0.01)
    unary = np.exp(-signature_cost / cost_scale)
    degree = np.asarray((compatibility != 0).sum(axis=1)).ravel().astype(np.int32)
    scores = unary.copy()
    for _ in range(iterations):
        scores = unary * (0.01 + np.asarray(compatibility @ scores).ravel())
        scores /= max(float(np.linalg.norm(scores)), EPS)
    return scores, degree


def sparse_one_to_one_projection(
    edge_i: np.ndarray,
    edge_j: np.ndarray,
    candidate_cost: np.ndarray,
    counts: np.ndarray,
    num_source: int,
    num_destination: int,
    unmatched_scale: float,
) -> tuple[dict[int, int], float]:
    offsets = np.concatenate(([0], np.cumsum(counts)))
    row_best = np.asarray([
        np.min(candidate_cost[offsets[index]:offsets[index + 1]])
        for index in range(num_source)
    ])
    median = float(np.median(row_best))
    spread = max(robust_mad(row_best), 0.05 * max(median, EPS), EPS)
    unmatched = max(
        float(np.max(row_best) + spread),
        median + unmatched_scale * spread,
    )
    rows = np.concatenate((edge_i, np.arange(num_source, dtype=np.int32)))
    columns = np.concatenate((edge_j, num_destination + np.arange(num_source, dtype=np.int32)))
    costs = np.maximum(np.concatenate((candidate_cost, np.full(num_source, unmatched))), EPS)
    matrix = coo_matrix(
        (costs, (rows, columns)), shape=(num_source, num_destination + num_source),
    ).tocsr()
    assigned_rows, assigned_columns = min_weight_full_bipartite_matching(matrix)
    return {
        int(source): int(destination)
        for source, destination in zip(assigned_rows, assigned_columns)
        if destination < num_destination
    }, float(unmatched)


def sparse_metric_hungarian(
    source: np.ndarray,
    target: np.ndarray,
    radius: float,
    candidates: int = 4,
) -> dict[int, int]:
    """One-to-one metric correspondences without constructing a dense cost matrix."""
    if not len(source) or not len(target):
        return {}
    count = min(candidates, len(target))
    distances, indices = cKDTree(target).query(
        source, k=count, distance_upper_bound=radius,
    )
    if count == 1:
        distances, indices = distances[:, None], indices[:, None]
    valid = np.isfinite(distances) & (indices < len(target))
    source_rows, candidate_columns = np.nonzero(valid)
    destination_columns = indices[source_rows, candidate_columns].astype(np.int32)
    metric_costs = distances[source_rows, candidate_columns] / max(radius, EPS)
    rows = np.concatenate((source_rows.astype(np.int32), np.arange(len(source), dtype=np.int32)))
    columns = np.concatenate((destination_columns, len(target) + np.arange(len(source), dtype=np.int32)))
    costs = np.concatenate((np.maximum(metric_costs, EPS), np.full(len(source), 1.0)))
    matrix = coo_matrix(
        (costs, (rows, columns)), shape=(len(source), len(target) + len(source)),
    ).tocsr()
    assigned_rows, assigned_columns = min_weight_full_bipartite_matching(matrix)
    return {
        int(source_id): int(destination_id)
        for source_id, destination_id in zip(assigned_rows, assigned_columns)
        if destination_id < len(target)
    }


def preliminary_static_anchor_icp(
    xyz_a: np.ndarray,
    xyz_b: np.ndarray,
    spacing_a: float,
    spacing_b: float,
    iterations: int = 3,
) -> tuple[np.ndarray, dict[int, int], dict]:
    """Align residual frame drift from conservative, nearly stationary anchors.

    The sparse metric Hungarian solve is deliberately only a pre-alignment.
    Relative-distance graph matching still decides the temporal identity. A
    pair becomes static-anchor evidence only when both solvers select it.
    """
    spacing = max(min(spacing_a, spacing_b), EPS)
    # A global moving body must not become "static" merely because ICP can
    # align it. Seeds therefore have an absolute motion ceiling as well as a
    # spacing-relative one; fixed-lattice static cells normally coincide.
    icp_radius = min(0.25 * spacing, 0.01)
    strong_radius = min(0.125 * spacing, 0.003)
    rotation = np.eye(3)
    translation = np.zeros(3)
    aligned = xyz_a.copy()
    completed = 0
    for _ in range(iterations):
        mapping = sparse_metric_hungarian(aligned, xyz_b, icp_radius)
        if len(mapping) < 3:
            break
        source_ids = np.fromiter(mapping, np.int32)
        destination_ids = np.asarray([mapping[index] for index in source_ids], np.int32)
        delta_rotation, delta_translation, _residual = fit_rigid_transform(
            aligned[source_ids], xyz_b[destination_ids],
        )
        residuals = np.linalg.norm(
            aligned[source_ids] @ delta_rotation.T + delta_translation - xyz_b[destination_ids],
            axis=1,
        )
        cutoff = min(icp_radius, float(np.median(residuals)) + 2.5 * max(robust_mad(residuals), EPS))
        inliers = residuals <= cutoff
        if np.sum(inliers) >= 3:
            delta_rotation, delta_translation, _residual = fit_rigid_transform(
                aligned[source_ids[inliers]], xyz_b[destination_ids[inliers]],
            )
        rotation = delta_rotation @ rotation
        translation = delta_rotation @ translation + delta_translation
        aligned = xyz_a @ rotation.T + translation
        completed += 1

    aligned_anchors = sparse_metric_hungarian(aligned, xyz_b, strong_radius)
    anchors = {
        source_id: destination_id
        for source_id, destination_id in aligned_anchors.items()
        if np.linalg.norm(xyz_a[source_id] - xyz_b[destination_id]) <= strong_radius
    }
    if anchors:
        source_ids = np.fromiter(anchors, np.int32)
        destination_ids = np.asarray([anchors[index] for index in source_ids], np.int32)
        anchor_residual = float(np.median(np.linalg.norm(
            aligned[source_ids] - xyz_b[destination_ids], axis=1,
        )))
    else:
        anchor_residual = math.nan
    return aligned, anchors, {
        "prealignment_iterations": completed,
        "preliminary_static_anchors": len(anchors),
        "prealignment_anchor_radius": strong_radius,
        "prealignment_icp_radius": icp_radius,
        "prealignment_median_anchor_residual": anchor_residual,
        "prealignment_rotation": rotation.tolist(),
        "prealignment_translation": translation.tolist(),
    }


def match_frame_pair(
    xyz_a: np.ndarray,
    xyz_b: np.ndarray,
    num_rel_neighbors: int = 8,
    hungarian_candidates: int = 16,
    refine_iters: int = 5,
    min_relational_support: int = 3,
    unmatched_cost_scale: float = 2.5,
    compatibility_relative_threshold: float = 0.03,
    signature_a: tuple[np.ndarray, np.ndarray, np.ndarray, float] | None = None,
    signature_b: tuple[np.ndarray, np.ndarray, np.ndarray, float] | None = None,
    label_probabilities_a: np.ndarray | None = None,
    label_probabilities_b: np.ndarray | None = None,
    label_cost_weight: float = 0.0,
) -> tuple[list[dict], list[dict], dict]:
    """Static-anchor ICP, then relative graph matching and one-to-one projection."""
    if len(xyz_a) < min_relational_support + 2 or len(xyz_b) < min_relational_support + 2:
        raise RuntimeError("Too few reduced depth points for relative graph matching")
    signature_a = signature_a or sparse_graph_signatures(xyz_a, num_rel_neighbors)
    signature_b = signature_b or sparse_graph_signatures(xyz_b, num_rel_neighbors)
    aligned_a, preliminary_anchors, prealignment_stats = preliminary_static_anchor_icp(
        xyz_a, xyz_b, signature_a[3], signature_b[3],
    )
    edge_i, edge_j, signature_cost, counts = sparse_signature_candidates(
        signature_a, signature_b, hungarian_candidates,
    )
    d_scale = max(0.125 * (signature_a[3] + signature_b[3]), EPS)
    compatibility = candidate_compatibility_graph(
        aligned_a, xyz_b, signature_a[2], edge_i, edge_j, counts, d_scale,
        compatibility_relative_threshold,
    )
    graph_scores, compatibility_degree = graph_matching_scores(
        signature_cost, compatibility, refine_iters,
    )
    graph_cost = -np.log(np.maximum(graph_scores, EPS))
    label_similarity = np.full(len(edge_i), math.nan, np.float64)
    if label_probabilities_a is not None or label_probabilities_b is not None:
        if label_probabilities_a is None or label_probabilities_b is None:
            raise ValueError("Both frame label-probability arrays are required")
        if len(label_probabilities_a) != len(xyz_a) or len(label_probabilities_b) != len(xyz_b):
            raise ValueError("Label-probability arrays must align with their frame clouds")
        label_similarity = np.sum(
            label_probabilities_a[edge_i] * label_probabilities_b[edge_j], axis=1,
        )
        graph_cost += label_cost_weight * -np.log(np.maximum(label_similarity, EPS))
    mapping, unmatched = sparse_one_to_one_projection(
        edge_i, edge_j, graph_cost, counts, len(xyz_a), len(xyz_b), unmatched_cost_scale,
    )

    offsets = np.concatenate(([0], np.cumsum(counts)))
    candidate_lookup = {
        (int(source), int(destination)): edge_id
        for edge_id, (source, destination) in enumerate(zip(edge_i, edge_j))
    }
    selected_nodes = np.zeros(len(edge_i), np.float64)
    for source, destination in mapping.items():
        selected_nodes[candidate_lookup[source, destination]] = 1.0
    selected_support = np.asarray((compatibility != 0) @ selected_nodes).ravel()
    selected_weight = np.asarray(compatibility @ selected_nodes).ravel()

    cost_threshold = unmatched
    margins, margin_ratios = {}, {}
    for source, destination in mapping.items():
        edge_id = candidate_lookup[source, destination]
        alternatives = graph_cost[offsets[source]:offsets[source + 1]]
        other = alternatives[np.arange(len(alternatives)) != edge_id - offsets[source]]
        best_other = float(min(np.min(other) if other.size else unmatched, unmatched))
        margin = best_other - float(graph_cost[edge_id])
        margins[source, destination] = margin
        margin_ratios[source, destination] = margin / max(
            abs(best_other), abs(float(graph_cost[edge_id])), EPS,
        )
    ratio_values = np.asarray(list(margin_ratios.values()))
    margin_threshold = min(0.0, float(np.percentile(ratio_values, 5))) if ratio_values.size else 0.0

    accepted, diagnostics, accepted_destinations = [], [], set()
    for source in range(len(xyz_a)):
        destination = mapping.get(source, -1)
        reasons = []
        if destination < 0:
            record = {
                "i": source, "j": -1, "total_cost": unmatched,
                "relative_cost": math.nan, "signature_cost": math.nan,
                "relational_support": 0, "compatibility_degree": 0,
                "graph_score": 0.0, "assignment_margin": 0.0,
                "label_similarity": math.nan,
                "preliminary_static_anchor": False,
                "accepted": False, "reject_reason": "dummy_assignment",
            }
        else:
            edge_id = candidate_lookup[source, destination]
            support = int(selected_support[edge_id])
            relative_cost = 1.0 - float(selected_weight[edge_id]) / max(support, 1)
            if support < min_relational_support:
                reasons.append("low_compatibility_support")
            if graph_cost[edge_id] > cost_threshold:
                reasons.append("robust_graph_cost_outlier")
            if margin_ratios[source, destination] <= margin_threshold:
                reasons.append("weak_assignment_margin")
            record = {
                "i": source, "j": destination,
                "total_cost": float(graph_cost[edge_id]),
                "relative_cost": relative_cost,
                "signature_cost": float(signature_cost[edge_id]),
                "relational_support": support,
                "compatibility_degree": int(compatibility_degree[edge_id]),
                "graph_score": float(graph_scores[edge_id]),
                "assignment_margin": float(margins[source, destination]),
                "label_similarity": float(label_similarity[edge_id]),
                "preliminary_static_anchor": preliminary_anchors.get(source) == destination,
                "accepted": not reasons, "reject_reason": ";".join(reasons),
            }
            if not reasons:
                accepted.append(record)
                accepted_destinations.add(destination)
        diagnostics.append(record)
    selected_destinations = set(mapping.values())
    for destination in range(len(xyz_b)):
        if destination not in accepted_destinations:
            diagnostics.append({
                "i": -1, "j": destination, "total_cost": unmatched,
                "relative_cost": math.nan, "signature_cost": math.nan,
                "relational_support": 0, "compatibility_degree": 0,
                "graph_score": 0.0, "assignment_margin": 0.0,
                "label_similarity": math.nan,
                "preliminary_static_anchor": False,
                "accepted": False,
                "reject_reason": "rejected_source" if destination in selected_destinations else "dummy_assignment",
            })
    stats = {
        "initial_assignments": len(mapping),
        "initial_dummy_assignments": len(xyz_a) - len(mapping),
        "refinement_iterations": refine_iters,
        "changed_assignments": [],
        "convergence_status": "compatibility_graph_projected",
        "converged": True,
        "final_real_assignments": len(mapping),
        "final_dummy_assignments": len(xyz_a) - len(mapping),
        "stable_assignments": len(mapping),
        "accepted_assignments": len(accepted),
        "candidate_edges": len(edge_i),
        "compatibility_edges": compatibility.nnz // 2,
        "unmatched_cost": unmatched,
        "cost_threshold": cost_threshold,
        "margin_threshold": margin_threshold,
        "d_scale": d_scale,
        "label_cost_weight": label_cost_weight,
        **prealignment_stats,
    }
    return accepted, diagnostics, stats


def build_tracks(
    clouds: dict[int, FrameCloud],
    frames: list[int],
    matcher_kwargs: dict,
    max_gap: int = 1,
) -> tuple[dict[int, Track], list[dict], dict[tuple[int, int], dict]]:
    tracks: dict[int, Track] = {}
    point_to_track: dict[tuple[int, int], int] = {}
    frame_rank = {frame: rank for rank, frame in enumerate(frames)}
    signatures = {
        frame: sparse_graph_signatures(clouds[frame].xyz, matcher_kwargs["num_rel_neighbors"])
        for frame in frames
    }
    next_track_id = 0
    first = frames[0]
    for point_id, point in enumerate(clouds[first].xyz):
        tracks[next_track_id] = Track(next_track_id, {first: point.copy()}, {first: point_id})
        point_to_track[(first, point_id)] = next_track_id
        next_track_id += 1
    rows: list[dict] = []
    pair_stats: dict[tuple[int, int], dict] = {}

    def solve_pair(frame_a: int, frame_b: int, temporal_role: str):
        accepted, diagnostics, stats = match_frame_pair(
            clouds[frame_a].xyz, clouds[frame_b].xyz, **matcher_kwargs,
            signature_a=signatures[frame_a], signature_b=signatures[frame_b],
        )
        stats.update({
            "frame_a": frame_a,
            "frame_b": frame_b,
            "temporal_role": temporal_role,
            "bridged_tracks": 0,
            "points_a": len(clouds[frame_a].xyz),
            "points_b": len(clouds[frame_b].xyz),
            "rejected_source_assignments": sum(
                record["i"] >= 0 and not record["accepted"] for record in diagnostics
            ),
        })
        pair_stats[(frame_a, frame_b)] = stats
        return accepted, diagnostics, stats

    def attach(track_id: int, frame: int, point_id: int, record: dict, unmatched: float) -> None:
        track = tracks[track_id]
        track.points[frame] = clouds[frame].xyz[point_id].copy()
        track.point_ids[frame] = point_id
        normalized_margin = record["assignment_margin"] / max(unmatched, EPS)
        confidence = float(
            np.clip(0.5 + 0.5 * normalized_margin, 0.0, 1.0) *
            math.exp(-record["relative_cost"])
        )
        track.match_confidences.append(confidence)
        track.temporal_matches += 1
        track.static_anchor_matches += int(record.get("preliminary_static_anchor", False))
        point_to_track[(frame, point_id)] = track_id

    def append_rows(frame_a: int, frame_b: int, diagnostics: list[dict], temporal_role: str) -> None:
        for record in diagnostics:
            track_id = -1
            if record["i"] >= 0:
                track_id = point_to_track.get((frame_a, record["i"]), -1)
            elif record["j"] >= 0:
                track_id = point_to_track.get((frame_b, record["j"]), -1)
            rows.append({
                "frame_a": frame_a,
                "frame_b": frame_b,
                "temporal_role": record.get("temporal_role", temporal_role),
                "multi_frame_consistency": record.get("multi_frame_consistency", "not_applicable"),
                "track_id": track_id,
                "point_a": record["i"],
                "point_b": record["j"],
                "total_cost": record["total_cost"],
                "relative_cost": record["relative_cost"],
                "signature_cost": record["signature_cost"],
                "relational_support": record["relational_support"],
                "compatibility_degree": record["compatibility_degree"],
                "graph_score": record["graph_score"],
                "assignment_margin": record["assignment_margin"],
                "label_similarity": record.get("label_similarity", math.nan),
                "preliminary_static_anchor": record.get("preliminary_static_anchor", False),
                "accepted": record["accepted"],
                "reject_reason": record["reject_reason"],
            })

    for frame_index, (frame_a, frame_b) in enumerate(zip(frames[:-1], frames[1:]), start=1):
        accepted, diagnostics, stats = solve_pair(frame_a, frame_b, "adjacent")
        skip_accepted, skip_diagnostics, skip_stats = [], [], None
        skip_frame = None
        if max_gap > 0 and frame_index >= 2:
            skip_frame = frames[frame_index - 2]
            skip_accepted, skip_diagnostics, skip_stats = solve_pair(skip_frame, frame_b, "skip")
            skip_mapping = {record["i"]: record["j"] for record in skip_accepted}
            for record in accepted:
                track_id = point_to_track.get((frame_a, record["i"]))
                track = tracks.get(track_id) if track_id is not None else None
                if track is None or skip_frame not in track.point_ids:
                    record["multi_frame_consistency"] = "unavailable"
                    continue
                skip_destination = skip_mapping.get(track.point_ids[skip_frame])
                if skip_destination == record["j"]:
                    record["multi_frame_consistency"] = "confirmed"
                elif skip_destination is None:
                    record["multi_frame_consistency"] = "unavailable"
                else:
                    record["multi_frame_consistency"] = "conflict"
                    record["accepted"] = False
                    record["reject_reason"] = ";".join(filter(None, (
                        record["reject_reason"], "skip_frame_conflict",
                    )))
            accepted = [record for record in accepted if record["accepted"]]

        used_b = set()
        for record in accepted:
            point_a, point_b = record["i"], record["j"]
            track_id = point_to_track.get((frame_a, point_a))
            if track_id is None:
                track_id = next_track_id
                tracks[track_id] = Track(
                    track_id,
                    {frame_a: clouds[frame_a].xyz[point_a].copy()},
                    {frame_a: point_a},
                )
                point_to_track[(frame_a, point_a)] = track_id
                next_track_id += 1
            attach(track_id, frame_b, point_b, record, stats["unmatched_cost"])
            used_b.add(point_b)

        bridged = 0
        if skip_frame is not None:
            for record in skip_accepted:
                track_id = point_to_track.get((skip_frame, record["i"]))
                track = tracks.get(track_id) if track_id is not None else None
                if (
                    track is None or record["j"] in used_b or frame_a in track.points or
                    max(track.points, key=frame_rank.get) != skip_frame
                ):
                    record["temporal_role"] = "skip_consistency_only"
                    continue
                attach(track_id, frame_b, record["j"], record, skip_stats["unmatched_cost"])
                record["temporal_role"] = "skip_bridge"
                record["multi_frame_consistency"] = "gap_bridged"
                used_b.add(record["j"])
                bridged += 1
            skip_stats["bridged_tracks"] = bridged

        for point_b, point in enumerate(clouds[frame_b].xyz):
            if point_b in used_b:
                continue
            tracks[next_track_id] = Track(next_track_id, {frame_b: point.copy()}, {frame_b: point_b})
            point_to_track[(frame_b, point_b)] = next_track_id
            next_track_id += 1
        append_rows(frame_a, frame_b, diagnostics, "adjacent")
        if skip_frame is not None:
            append_rows(skip_frame, frame_b, skip_diagnostics, "skip")

        relative = [row["relative_cost"] for row in accepted]
        signature = [row["signature_cost"] for row in accepted]
        support = [row["relational_support"] for row in accepted]
        unmatched_count = sum(not row["accepted"] for row in diagnostics)
        print(
            f"[PAIR GRAPH] {frame_a:06d} -> {frame_b:06d}\n"
            f"  points: {len(clouds[frame_a].xyz)} -> {len(clouds[frame_b].xyz)}\n"
            f"  candidates={stats['candidate_edges']} compatibilities={stats['compatibility_edges']}\n"
            f"  preliminary static anchors={stats['preliminary_static_anchors']} "
            f"ICP residual={stats['prealignment_median_anchor_residual']:.6g}\n"
            f"  accepted={len(accepted)}\n"
            f"  unmatched/rejected={unmatched_count}\n"
            f"  median signature cost={np.median(signature) if signature else math.nan:.6g}\n"
            f"  median relative cost={np.median(relative) if relative else math.nan:.6g}\n"
            f"  median relational support={np.median(support) if support else 0:.1f}"
        )
        if skip_frame is not None:
            print(
                f"[SKIP GRAPH] {skip_frame:06d} -> {frame_b:06d} "
                f"accepted={len(skip_accepted)} bridged={bridged}"
            )
    return tracks, rows, pair_stats


# =============================================================================
# Relative-distance rigidity graph and contrastive K+1 grouping
# =============================================================================

def track_rows(tracks: dict[int, Track]) -> list[dict]:
    rows = []
    for track_id, track in sorted(tracks.items()):
        for frame, point in sorted(track.points.items()):
            rows.append({
                "track_id": track_id,
                "frame": frame,
                "point_id": track.point_ids[frame],
                "X": float(point[0]),
                "Y": float(point[1]),
                "Z": float(point[2]),
                "track_length": len(track.points),
                "mean_match_confidence": float(np.mean(track.match_confidences)) if track.match_confidences else 0.0,
                "temporal_matches": track.temporal_matches,
                "static_anchor_matches": track.static_anchor_matches,
                "stationary_occupancy_ambiguous": track.stationary_occupancy_ambiguous,
                "knn_teacher_count": track.knn_teacher_count,
                "temporal_trajectory_id": track.temporal_trajectory_id,
                "temporal_coherence_links": track.temporal_coherence_links,
                "static_prior": track.static_prior,
                "static_prior_weight": track.static_prior_weight,
                "seed_type": track.seed_type,
                "seed_strength": track.seed_strength,
                "dynamic_static_relative_score": track.dynamic_static_relative_score,
                "dynamic_static_relative_support": track.dynamic_static_relative_support,
            })
    return rows


def scene_spacing(tracks: dict[int, Track], valid_ids: list[int], frames: list[int]) -> float:
    nearest = []
    for frame in frames:
        xyz = [tracks[track_id].points[frame] for track_id in valid_ids if frame in tracks[track_id].points]
        if len(xyz) > 1:
            distances = cKDTree(np.stack(xyz)).query(np.stack(xyz), k=2)[0]
            nearest.extend(distances[:, 1])
    return max(float(np.median(nearest)) if nearest else 1.0, EPS)


def fit_two_log_modes(values: np.ndarray) -> dict:
    z = np.log(np.asarray(values, np.float64) + EPS)
    # The exact-static spike can exceed 75% of fine-lattice tracks, so quartile
    # initialization collapses both modes onto zero motion.
    low, high = np.percentile(z, [10, 90])
    for _ in range(100):
        choose_low = np.abs(z - low) <= np.abs(z - high)
        next_low = float(np.median(z[choose_low])) if np.any(choose_low) else float(low)
        next_high = float(np.median(z[~choose_low])) if np.any(~choose_low) else float(high)
        if abs(next_low - low) + abs(next_high - high) < 1e-7:
            break
        low, high = next_low, next_high
    low, high = sorted((float(low), float(high)))
    low_group = z[np.abs(z - low) <= np.abs(z - high)]
    high_group = z[np.abs(z - high) < np.abs(z - low)]
    low_scale = max(robust_mad(low_group), 0.05)
    high_scale = max(robust_mad(high_group), 0.05)
    return {
        "low_mode": low,
        "high_mode": high,
        "low_scale": low_scale,
        "high_scale": high_scale,
    }


def same_probability(log_value: float, modes: dict) -> float:
    low_likelihood = -0.5 * ((log_value - modes["low_mode"]) / modes["low_scale"]) ** 2 - math.log(modes["low_scale"])
    high_likelihood = -0.5 * ((log_value - modes["high_mode"]) / modes["high_scale"]) ** 2 - math.log(modes["high_scale"])
    logit = float(np.clip(low_likelihood - high_likelihood, -60.0, 60.0))
    return 1.0 / (1.0 + math.exp(-logit))



def build_static_anchor_ghost_priors(
    tracks: dict[int, Track],
    frames: list[int],
    spacing: float,
    radius_scale: float = 0.125,
) -> tuple[list[dict], dict]:
    """Build trusted STATIC seeds and soft occlusion memory before optimization.

    A strong preliminary static-anchor track is now treated as positive STATIC
    evidence, because these anchors are the reliable cue in the current depth
    run. A compatible ghost protects reappearing static material. The absence
    of a compatible ghost is not motion evidence, so such births stay uncertain.
    """
    frame_rank = {frame: rank for rank, frame in enumerate(frames)}
    trusted_static = 0
    occupancy_candidates = 0
    minimum_trusted_length = max(4, math.ceil(len(frames) / 3))
    for track in tracks.values():
        anchor_fraction = track.static_anchor_matches / max(track.temporal_matches, 1)
        occupancy_candidate = (
            track.static_anchor_matches >= 3
            and anchor_fraction >= 0.8
            and len(track.points) >= 4
        )
        # Repeated occupancy of one fixed world cell is not enough to prove
        # material identity: moving interior samples can repeatedly enter that
        # cell.  Keep short occupancy tracks uncertain and let the relative-
        # distance graph solve them.  Only sequence-persistent tracks can teach
        # STATIC or leave static ghosts.
        track.stationary_occupancy_ambiguous = occupancy_candidate
        occupancy_candidates += int(occupancy_candidate)
        trusted = occupancy_candidate and len(track.points) >= minimum_trusted_length
        if trusted:
            track.seed_type = "STATIC_ANCHOR"
            track.seed_strength = min(
                1.0,
                0.5 * anchor_fraction
                + 0.5 * min(track.static_anchor_matches / 8.0, 1.0),
            )
            # Near-hard unary. The optimizer will also clamp these rows.
            track.static_prior = 0.995
            track.static_prior_weight = track.seed_strength
            trusted_static += 1
        else:
            track.static_prior = 0.5
            track.static_prior_weight = 0.0

    births: dict[int, list[int]] = {frame: [] for frame in frames}
    deaths: dict[int, list[int]] = {frame: [] for frame in frames}
    for track_id, track in tracks.items():
        track_frames = sorted(track.points, key=frame_rank.get)
        births[track_frames[0]].append(track_id)
        deaths[track_frames[-1]].append(track_id)

    radius = radius_scale * spacing
    rows: list[dict] = []
    ghosts: list[dict] = []
    next_ghost_id = 0
    reactivated = unmatched_births = peak_ghosts = 0

    for rank, frame in enumerate(frames[1:], start=1):
        previous_frame = frames[rank - 1]

        # Trusted static anchors and their still-confident reactivations create
        # ghosts, so repeated occlusions preserve static identity without an
        # immortal unconstrained memory chain.
        for track_id in deaths[previous_frame]:
            track = tracks[track_id]
            if not trusted_static_track(track):
                continue
            ghosts.append({
                "ghost_id": next_ghost_id,
                "source_track_id": track_id,
                "position": track.points[previous_frame].copy(),
                "disappeared_frame": previous_frame,
                "static_prior": track.static_prior,
                "static_prior_weight": track.static_prior_weight,
            })
            rows.append({
                "event": "ghost_prior_created",
                "frame": previous_frame,
                "track_id": track_id,
                "ghost_id": next_ghost_id,
                "source_track_id": track_id,
                "distance": math.nan,
                "ghost_age_frames": 0,
                "assigned_cluster_id": -1,
                "static_prior": track.static_prior,
                "static_prior_weight": track.static_prior_weight,
            })
            next_ghost_id += 1

        frame_births = births[frame]
        matched_births: set[int] = set()
        matched_ghosts: set[int] = set()

        if frame_births and ghosts:
            birth_xyz = np.stack([tracks[track_id].points[frame] for track_id in frame_births])
            ghost_xyz = np.stack([ghost["position"] for ghost in ghosts])
            distances = cdist(birth_xyz, ghost_xyz)
            costs = np.full(
                (len(frame_births), len(ghosts) + len(frame_births)),
                2.0 * radius,
            )
            costs[:, :len(ghosts)] = np.where(
                distances <= radius, distances, 4.0 * radius
            )
            costs[
                np.arange(len(frame_births)),
                len(ghosts) + np.arange(len(frame_births)),
            ] = radius

            birth_indices, ghost_indices = linear_sum_assignment(costs)
            for birth_index, ghost_index in zip(birth_indices, ghost_indices):
                if ghost_index >= len(ghosts):
                    continue
                distance = float(distances[birth_index, ghost_index])
                if distance > radius:
                    continue

                track_id = frame_births[birth_index]
                track = tracks[track_id]
                ghost = ghosts[ghost_index]
                distance_confidence = 1.0 - 0.5 * distance / max(radius, EPS)
                transferred_weight = ghost["static_prior_weight"] * distance_confidence

                if transferred_weight > track.static_prior_weight:
                    track.static_prior = max(track.static_prior, ghost["static_prior"])
                    track.static_prior_weight = transferred_weight
                if transferred_weight >= 0.75:
                    track.seed_type = "STATIC_GHOST"
                    track.seed_strength = transferred_weight

                track.ghost_source_track_id = ghost["source_track_id"]
                track.ghost_distance = distance
                track.ghost_age_frames = (
                    frame_rank[frame] - frame_rank[ghost["disappeared_frame"]]
                )
                rows.append({
                    "event": "ghost_prior_reactivated",
                    "frame": frame,
                    "track_id": track_id,
                    "ghost_id": ghost["ghost_id"],
                    "source_track_id": ghost["source_track_id"],
                    "distance": distance,
                    "ghost_age_frames": track.ghost_age_frames,
                    "assigned_cluster_id": -1,
                    "static_prior": track.static_prior,
                    "static_prior_weight": track.static_prior_weight,
                })
                matched_births.add(birth_index)
                matched_ghosts.add(ghost_index)
                reactivated += 1

            ghosts = [
                ghost for index, ghost in enumerate(ghosts)
                if index not in matched_ghosts
            ]

        # A later-frame birth with no compatible trusted-static ghost carries
        # no class evidence. Leave it uncertain instead of inventing motion.
        for birth_index, track_id in enumerate(frame_births):
            if birth_index in matched_births:
                continue
            track = tracks[track_id]
            if trusted_static_track(track):
                rows.append({
                    "event": "trusted_static_birth_without_ghost",
                    "frame": frame,
                    "track_id": track_id,
                    "ghost_id": -1,
                    "source_track_id": -1,
                    "distance": math.nan,
                    "ghost_age_frames": 0,
                    "assigned_cluster_id": 0,
                    "static_prior": track.static_prior,
                    "static_prior_weight": track.static_prior_weight,
                })
                continue
            rows.append({
                "event": "birth_without_ghost_unlabelled",
                "frame": frame,
                "track_id": track_id,
                "ghost_id": -1,
                "source_track_id": -1,
                "distance": math.nan,
                "ghost_age_frames": 0,
                "assigned_cluster_id": -1,
                "static_prior": track.static_prior,
                "static_prior_weight": track.static_prior_weight,
            })
            unmatched_births += 1

        peak_ghosts = max(peak_ghosts, len(ghosts))

    return rows, {
        "enabled": True,
        "method": (
            "trusted static-anchor seeds + ghost reactivation; absence of a "
            "compatible ghost never creates a dynamic label"
        ),
        "unmatched_birth_policy": "neutral/unlabelled",
        "radius": radius,
        "radius_scale": radius_scale,
        "trusted_static_seed_tracks": trusted_static,
        "stationary_occupancy_candidates": occupancy_candidates,
        "minimum_trusted_static_track_length": minimum_trusted_length,
        "ghosts_created": next_ghost_id,
        "ghost_reactivations": reactivated,
        "births_without_ghost_left_unlabelled": unmatched_births,
        "peak_pending_ghosts": peak_ghosts,
        "pending_ghosts_at_end": len(ghosts),
        "static_seed_contract": (
            ">=3 preliminary static-anchor matches, >=80% anchor fraction, "
            f"and persistence for >=1/3 of the sequence ({minimum_trusted_length} frames)"
        ),
    }

def build_rigidity_graph(
    tracks: dict[int, Track],
    frames: list[int],
    mad_weight: float = 1.0,
    uncertain_knn_neighbors: int = 8,
) -> tuple[list[int], list[dict], float, dict]:
    """Build one joint graph for certain trajectories and uncertain fragments."""
    core_ids = [track_id for track_id, track in tracks.items() if len(track.points) >= 4]
    if not core_ids:
        return [], [], 1.0, {"relative_rigidity": {}, "track_motion": {}}
    spacing = scene_spacing(tracks, core_ids, frames)
    frame_rank = {frame: index for index, frame in enumerate(frames)}
    motion_errors = {}
    for track_id in core_ids:
        track_frames = sorted(tracks[track_id].points, key=frame_rank.get)
        speeds = [
            np.linalg.norm(
                tracks[track_id].points[frame_b] - tracks[track_id].points[frame_a]
            ) / max(frame_rank[frame_b] - frame_rank[frame_a], 1)
            for frame_a, frame_b in zip(track_frames[:-1], track_frames[1:])
        ]
        motion_errors[track_id] = float(np.median(speeds))
    motion_modes = fit_two_log_modes(np.asarray(list(motion_errors.values())))

    static_probabilities: dict[int, float] = {}
    for track_id, track in tracks.items():
        # Trusted seed evidence takes precedence over the noisy per-track motion
        # mode. The latter remains a weak unary only for unseeded core tracks.
        if track.static_prior_weight > 0:
            static_probabilities[track_id] = track.static_prior
        elif track_id in motion_errors:
            static_probabilities[track_id] = same_probability(
                math.log(motion_errors[track_id] + EPS), motion_modes,
            )
        else:
            static_probabilities[track_id] = 0.5

    def spatial_target(distance: float) -> float:
        logit = np.clip(
            (distance - 3.0 * spacing) / max(0.5 * spacing, EPS), -60, 60,
        )
        return float(1.0 / (1.0 + math.exp(float(logit))))

    # Temporal rigidity edges join tracks that coexist for at least two frames.
    candidate_pairs: set[tuple[int, int]] = set()
    uncertain_neighbor_pairs: set[tuple[int, int]] = set()
    for frame in frames:
        present = [track_id for track_id in core_ids if frame in tracks[track_id].points]
        if len(present) < 2:
            continue
        xyz = np.stack([tracks[track_id].points[frame] for track_id in present])
        neighbor_count = min(max(17, uncertain_knn_neighbors + 1), len(present))
        neighbors = cKDTree(xyz).query(xyz, k=neighbor_count)[1]
        for source, local in enumerate(neighbors[:, 1:]):
            for neighbor in local:
                candidate_pairs.add(tuple(sorted((present[source], present[int(neighbor)]))))
            if tracks[present[source]].stationary_occupancy_ambiguous:
                for neighbor in local[:uncertain_knn_neighbors]:
                    uncertain_neighbor_pairs.add(
                        tuple(sorted((present[source], present[int(neighbor)])))
                    )

    temporal_edges = []
    for track_a, track_b in sorted(candidate_pairs):
        common = sorted(
            set(tracks[track_a].points) & set(tracks[track_b].points),
            key=frame_rank.get,
        )
        if len(common) < 2:
            continue
        distances = np.asarray([
            np.linalg.norm(tracks[track_a].points[frame] - tracks[track_b].points[frame])
            for frame in common
        ])
        median_distance = float(np.median(distances))
        denominator = max(median_distance, 0.25 * spacing, EPS)
        relative_changes = np.abs(distances - median_distance) / denominator
        median_change = float(np.median(relative_changes))
        mad_change = robust_mad(relative_changes)
        rigidity = median_change + mad_weight * mad_change
        velocity_differences = []
        neighbor_temporal_changes = []
        for frame_a, frame_b, distance_a, distance_b in zip(
            common[:-1], common[1:], distances[:-1], distances[1:],
        ):
            gap = max(frame_rank[frame_b] - frame_rank[frame_a], 1)
            velocity_a = (
                tracks[track_a].points[frame_b] - tracks[track_a].points[frame_a]
            ) / gap
            velocity_b = (
                tracks[track_b].points[frame_b] - tracks[track_b].points[frame_a]
            ) / gap
            velocity_differences.append(float(np.linalg.norm(velocity_a - velocity_b)))
            if gap == 1:
                denominator = max(0.5 * (distance_a + distance_b), 0.25 * spacing, EPS)
                neighbor_temporal_changes.append(
                    float(abs(distance_b - distance_a) / denominator)
                )
        temporal_velocity_difference = float(np.median(velocity_differences))
        track_confidence = 0.5 * (
            (float(np.mean(tracks[track_a].match_confidences))
             if tracks[track_a].match_confidences else 0.0) +
            (float(np.mean(tracks[track_b].match_confidences))
             if tracks[track_b].match_confidences else 0.0)
        )
        temporal_edges.append({
            "edge_type": "temporal_rigidity",
            "track_a": track_a,
            "track_b": track_b,
            "common_frames": len(common),
            "median_pair_distance": median_distance,
            "median_relative_change": median_change,
            "MAD_relative_change": mad_change,
            "p90_relative_change": float(np.percentile(relative_changes, 90)),
            "max_robust_relative_change": float(min(
                np.max(relative_changes), median_change + 3.0 * max(mad_change, EPS)
            )),
            "rigidity_error": rigidity,
            "track_match_confidence": track_confidence,
            "track_motion_error_a": motion_errors[track_a],
            "track_motion_error_b": motion_errors[track_b],
            "track_static_probability_a": static_probabilities[track_a],
            "track_static_probability_b": static_probabilities[track_b],
            "track_static_prior_weight_a": tracks[track_a].static_prior_weight,
            "track_static_prior_weight_b": tracks[track_b].static_prior_weight,
            "temporal_velocity_difference": temporal_velocity_difference,
            "neighbor_temporal_relative_change": (
                float(np.median(neighbor_temporal_changes))
                if neighbor_temporal_changes else math.nan
            ),
            "neighbor_temporal_transitions": len(neighbor_temporal_changes),
            "spatial_same_target": spatial_target(median_distance),
            "temporal_same_target": 0.5,
        })
    if not temporal_edges:
        return core_ids, [], spacing, {
            "relative_rigidity": {}, "track_motion": motion_modes,
        }

    rigidity_modes = fit_two_log_modes(np.asarray([
        edge["rigidity_error"] for edge in temporal_edges
    ]))
    neighbor_changes = np.asarray([
        edge["neighbor_temporal_relative_change"] for edge in temporal_edges
        if np.isfinite(edge["neighbor_temporal_relative_change"])
    ])
    neighbor_temporal_modes = (
        fit_two_log_modes(neighbor_changes) if len(neighbor_changes) >= 2
        else rigidity_modes
    )

    # ---------------------------------------------------------------------
    # Positive DYNAMIC seeds: a track is dynamic only when its distances to
    # several trusted STATIC seeds change beyond the same-rigid noise mode.
    # This is positive physical evidence; "birth without ghost" is never used.
    # ---------------------------------------------------------------------
    static_seed_ids = {
        track_id for track_id, track in tracks.items()
        if track.seed_type in ("STATIC_ANCHOR", "STATIC_GHOST")
        and track.seed_strength >= 0.75
    }
    dynamic_votes: dict[int, list[float]] = {}
    for edge in temporal_edges:
        track_a, track_b = edge["track_a"], edge["track_b"]
        if track_a in static_seed_ids and track_b not in static_seed_ids:
            candidate = track_b
        elif track_b in static_seed_ids and track_a not in static_seed_ids:
            candidate = track_a
        else:
            continue
        different_probability = 1.0 - same_probability(
            math.log(edge["rigidity_error"] + EPS), rigidity_modes,
        )
        dynamic_votes.setdefault(candidate, []).append(float(different_probability))

    dynamic_seed_count = 0
    for track_id, votes in dynamic_votes.items():
        if len(tracks[track_id].points) < 2 or len(votes) < 2:
            continue
        score = float(np.median(votes))
        tracks[track_id].dynamic_static_relative_score = score
        tracks[track_id].dynamic_static_relative_support = len(votes)
        if score >= 0.90:
            tracks[track_id].seed_type = "DYNAMIC_RELATIVE_STATIC"
            tracks[track_id].seed_strength = min(
                1.0, score * min(len(votes) / 4.0, 1.0)
            )
            tracks[track_id].static_prior = 0.005
            tracks[track_id].static_prior_weight = tracks[track_id].seed_strength
            static_probabilities[track_id] = tracks[track_id].static_prior
            dynamic_seed_count += 1

    max_frames = max(len(tracks[track_id].points) for track_id in core_ids)
    for edge in temporal_edges:
        moving_rigid = same_probability(
            math.log(edge["rigidity_error"] + EPS), rigidity_modes,
        )
        both_static = (
            edge["track_static_probability_a"] *
            edge["track_static_probability_b"]
        )
        both_moving = (
            (1.0 - edge["track_static_probability_a"]) *
            (1.0 - edge["track_static_probability_b"])
        )
        # Same-rigid membership is determined ONLY by pairwise-distance
        # preservation. Individual static/dynamic probabilities are unary
        # evidence and must not contaminate the pairwise rigidity target.
        q_same = moving_rigid
        overlap = edge["common_frames"] / max(max_frames, 1)
        length = min(
            len(tracks[edge["track_a"]].points),
            len(tracks[edge["track_b"]].points),
        ) / max(max_frames, 1)
        confidence = max(edge["track_match_confidence"], 0.05)
        edge["both_static_probability"] = both_static
        edge["same_moving_rigid_probability"] = moving_rigid
        neighbor_change = edge["neighbor_temporal_relative_change"]
        neighbor_target = (
            same_probability(math.log(neighbor_change + EPS), neighbor_temporal_modes)
            if np.isfinite(neighbor_change) else 0.5
        )
        occupancy_uncertain = (
            tracks[edge["track_a"]].stationary_occupancy_ambiguous or
            tracks[edge["track_b"]].stationary_occupancy_ambiguous
        ) and (edge["track_a"], edge["track_b"]) in uncertain_neighbor_pairs
        edge["uncertain_neighbor_target"] = neighbor_target
        edge["uncertain_neighbor_weight"] = (
            max(0.01, abs(2.0 * neighbor_target - 1.0)) *
            math.sqrt(edge["neighbor_temporal_transitions"] / max(max_frames - 1, 1)) *
            confidence
            if occupancy_uncertain else 0.0
        )
        # Pairwise-distance preservation is invariant to rigid rotation;
        # equality of velocity vectors is not.
        edge["temporal_same_target"] = moving_rigid
        edge["q_same"] = q_same
        edge["weight"] = max(
            0.01, abs(2.0 * q_same - 1.0) *
            math.sqrt(overlap * length) * confidence,
        )
        edge["spatial_weight"] = max(
            0.01, abs(2.0 * edge["spatial_same_target"] - 1.0) * confidence,
        )
        edge["temporal_weight"] = max(
            0.01, abs(2.0 * edge["temporal_same_target"] - 1.0) *
            math.sqrt(overlap * length) * confidence,
        )

    # Short fragments are optimized against the temporal core. Their edges
    # are consumed with a detached core endpoint, so uncertain evidence cannot
    # corrupt the reliable trajectories that are meant to explain it.
    completion_pairs: set[tuple[int, int]] = set()
    uncertain_ids = {
        track_id for track_id, track in tracks.items() if len(track.points) < 4
    }
    for frame in frames:
        present_core = [
            track_id for track_id in core_ids if frame in tracks[track_id].points
        ]
        present_uncertain = [
            track_id for track_id in uncertain_ids if frame in tracks[track_id].points
        ]
        if not present_core or not present_uncertain:
            continue
        core_xyz = np.stack([tracks[track_id].points[frame] for track_id in present_core])
        uncertain_xyz = np.stack([
            tracks[track_id].points[frame] for track_id in present_uncertain
        ])
        neighbor_count = min(8, len(present_core))
        distances, neighbors = cKDTree(core_xyz).query(
            uncertain_xyz, k=neighbor_count, distance_upper_bound=6.0 * spacing,
        )
        if neighbor_count == 1:
            distances, neighbors = distances[:, None], neighbors[:, None]
        for uncertain_index, track_id in enumerate(present_uncertain):
            for distance, neighbor in zip(distances[uncertain_index], neighbors[uncertain_index]):
                if np.isfinite(distance) and neighbor < len(present_core):
                    completion_pairs.add((present_core[int(neighbor)], track_id))

    completion_edges = []
    for core_id, track_id in sorted(completion_pairs):
        common = sorted(
            set(tracks[core_id].points) & set(tracks[track_id].points),
            key=frame_rank.get,
        )
        distances = np.asarray([
            np.linalg.norm(tracks[core_id].points[frame] - tracks[track_id].points[frame])
            for frame in common
        ])
        median_distance = float(np.median(distances))
        target = spatial_target(median_distance)
        if len(common) >= 2:
            denominator = max(median_distance, 0.25 * spacing, EPS)
            relative_changes = np.abs(distances - median_distance) / denominator
            median_change = float(np.median(relative_changes))
            mad_change = robust_mad(relative_changes)
            rigidity_error = median_change + mad_weight * mad_change
            temporal_target = same_probability(
                math.log(rigidity_error + EPS), rigidity_modes,
            )
            velocity_differences = []
            for frame_a, frame_b in zip(common[:-1], common[1:]):
                gap = max(frame_rank[frame_b] - frame_rank[frame_a], 1)
                velocity_core = (
                    tracks[core_id].points[frame_b] - tracks[core_id].points[frame_a]
                ) / gap
                velocity_fragment = (
                    tracks[track_id].points[frame_b] - tracks[track_id].points[frame_a]
                ) / gap
                velocity_differences.append(
                    float(np.linalg.norm(velocity_core - velocity_fragment))
                )
            temporal_velocity_difference = float(np.median(velocity_differences))
            p90_change = float(np.percentile(relative_changes, 90))
            max_change = float(min(
                np.max(relative_changes), median_change + 3.0 * max(mad_change, EPS)
            ))
        else:
            median_change = mad_change = rigidity_error = math.nan
            temporal_target = 0.5
            temporal_velocity_difference = p90_change = max_change = math.nan

        core_certainty = abs(2.0 * static_probabilities[core_id] - 1.0)
        completion_edges.append({
            "edge_type": "uncertain_core_completion",
            "track_a": core_id,
            "track_b": track_id,
            "common_frames": len(common),
            "median_pair_distance": median_distance,
            "median_relative_change": median_change,
            "MAD_relative_change": mad_change,
            "p90_relative_change": p90_change,
            "max_robust_relative_change": max_change,
            "rigidity_error": rigidity_error,
            "track_match_confidence": 0.0,
            "track_motion_error_a": motion_errors[core_id],
            "track_motion_error_b": math.nan,
            "track_static_probability_a": static_probabilities[core_id],
            "track_static_probability_b": static_probabilities[track_id],
            "track_static_prior_weight_a": tracks[core_id].static_prior_weight,
            "track_static_prior_weight_b": tracks[track_id].static_prior_weight,
            "temporal_velocity_difference": temporal_velocity_difference,
            "neighbor_temporal_relative_change": math.nan,
            "neighbor_temporal_transitions": 0,
            "spatial_same_target": target,
            "temporal_same_target": temporal_target,
            "uncertain_neighbor_target": 0.5,
            "uncertain_neighbor_weight": 0.0,
            "both_static_probability": (
                static_probabilities[core_id] * static_probabilities[track_id]
            ),
            "same_moving_rigid_probability": temporal_target,
            "q_same": 0.5,
            "weight": 0.0,
            "spatial_weight": max(
                0.01, abs(2.0 * target - 1.0) * max(core_certainty, 0.1),
            ),
            "temporal_weight": (
                max(0.01, abs(2.0 * temporal_target - 1.0) * max(core_certainty, 0.1))
                if len(common) >= 2 else 0.0
            ),
        })
    optimized_ids = sorted(
        set(core_ids) |
        {edge["track_b"] for edge in completion_edges} |
        {
            track_id for track_id in uncertain_ids
            if tracks[track_id].static_prior_weight > 0
        }
    )
    return optimized_ids, temporal_edges + completion_edges, spacing, {
        "relative_rigidity": rigidity_modes,
        "track_motion": motion_modes,
        "uncertain_neighbor_temporal": neighbor_temporal_modes,
        "temporal_core_tracks": len(core_ids),
        "uncertain_optimized_tracks": len(optimized_ids) - len(core_ids),
        "temporal_rigidity_edges": len(temporal_edges),
        "uncertain_completion_edges": len(completion_edges),
        "uncertain_neighbor_edges": sum(
            edge["uncertain_neighbor_weight"] > 0 for edge in temporal_edges
        ),
        "uncertain_neighbor_k": uncertain_knn_neighbors,
        "trusted_static_seed_tracks": len(static_seed_ids),
        "dynamic_relative_static_seed_tracks": dynamic_seed_count,
    }


def numpy_kmeans(features: np.ndarray, clusters: int, iterations: int = 100) -> np.ndarray:
    if len(features) < clusters:
        raise RuntimeError(f"Cannot cluster {len(features)} tracks into {clusters} groups")
    centers = [0]
    while len(centers) < clusters:
        distances = np.min(cdist(features, features[centers]), axis=1)
        distances[centers] = -1.0
        centers.append(int(np.argmax(distances)))
    means = features[centers].copy()
    labels = np.full(len(features), -1, np.int32)
    for _ in range(iterations):
        new_labels = np.argmin(cdist(features, means), axis=1)
        if np.array_equal(new_labels, labels):
            break
        labels = new_labels
        for cluster in range(clusters):
            if np.any(labels == cluster):
                means[cluster] = np.mean(features[labels == cluster], axis=0)
    return labels


def spectral_initialization(valid_ids: list[int], edges: list[dict], clusters: int) -> np.ndarray:
    lookup = {track_id: index for index, track_id in enumerate(valid_ids)}
    static_sum = np.zeros(len(valid_ids), np.float64)
    static_count = np.zeros(len(valid_ids), np.float64)
    for edge in edges:
        for suffix in ("a", "b"):
            index = lookup[edge[f"track_{suffix}"]]
            static_sum[index] += edge[f"track_static_probability_{suffix}"]
            static_count[index] += 1
    static_evidence = static_sum / np.maximum(static_count, 1)
    if clusters == 2:
        labels = numpy_kmeans(static_evidence[:, None], clusters)
        static_label = max(
            range(clusters),
            key=lambda cluster: float(np.mean(static_evidence[labels == cluster])),
        )
        if static_label != 0:
            labels = np.where(labels == 0, static_label, np.where(labels == static_label, 0, labels))
        return labels

    rows, columns, values = [], [], []
    for edge in edges:
        i, j = lookup[edge["track_a"]], lookup[edge["track_b"]]
        value = (
            edge["q_same"] * edge["weight"] +
            edge["spatial_same_target"] * edge["spatial_weight"]
        )
        rows.extend((i, j))
        columns.extend((j, i))
        values.extend((value, value))
    rows.extend(range(len(valid_ids)))
    columns.extend(range(len(valid_ids)))
    values.extend([1.0] * len(valid_ids))
    affinity = coo_matrix(
        (values, (rows, columns)), shape=(len(valid_ids), len(valid_ids)),
    ).tocsr()
    degree = np.asarray(affinity.sum(axis=1)).ravel()
    inverse = 1.0 / np.sqrt(np.maximum(degree, EPS))
    normalized = affinity.multiply(inverse[:, None]).multiply(inverse[None, :]).tocsr()
    if clusters >= len(valid_ids) - 1:
        _eigenvalues, eigenvectors = np.linalg.eigh(normalized.toarray())
        embedding = eigenvectors[:, -clusters:]
    else:
        try:
            _eigenvalues, embedding = eigsh(
                normalized, k=clusters, which="LA", v0=np.ones(len(valid_ids)),
                maxiter=1000, tol=1e-4,
            )
        except ArpackNoConvergence:
            positive_degree = np.asarray(affinity.sum(axis=1)).ravel()
            labels = numpy_kmeans(
                np.column_stack((static_evidence, positive_degree)), clusters,
            )
            static_label = max(
                range(clusters),
                key=lambda cluster: float(np.mean(static_evidence[labels == cluster])),
            )
            return np.where(labels == 0, static_label, np.where(labels == static_label, 0, labels))
    embedding /= np.maximum(np.linalg.norm(embedding, axis=1, keepdims=True), EPS)
    labels = numpy_kmeans(embedding, clusters)
    static_label = max(
        range(clusters),
        key=lambda cluster: float(np.mean(static_evidence[labels == cluster])),
    )
    return np.where(labels == 0, static_label, np.where(labels == static_label, 0, labels))


def optimize_contrastive_groups(
    valid_ids: list[int],
    edges: list[dict],
    num_groups: int,
    iterations: int,
    learning_rate: float,
    spatial_loss_weight: float = 0.001,
    temporal_loss_weight: float = 0.01,
    static_loss_weight: float = 0.01,
    completion_loss_weight: float = 1.0,
    tracks: dict[int, Track] | None = None,
    uncertain_neighbor_loss_weight: float = 1.0,
) -> tuple[dict[int, tuple[int, float, np.ndarray]], list[dict]]:
    if len(valid_ids) < num_groups:
        raise RuntimeError(f"Only {len(valid_ids)} temporally supported tracks are available for {num_groups} rigid groups")
    if not edges:
        raise RuntimeError("No temporally overlapping track pairs for contrastive rigid grouping")
    lookup = {track_id: index for index, track_id in enumerate(valid_ids)}
    core_ids = [
        track_id for track_id in valid_ids
        if tracks is None or len(tracks[track_id].points) >= 4
    ]
    core_edges = [edge for edge in edges if edge["edge_type"] == "temporal_rigidity"]
    initial = spectral_initialization(core_ids, core_edges, num_groups)
    logits = torch.zeros((len(valid_ids), num_groups), dtype=torch.float32)
    core_rows = torch.tensor([lookup[track_id] for track_id in core_ids], dtype=torch.long)
    logits[core_rows] = -2.0
    logits[core_rows, torch.from_numpy(initial).long()] = 2.0

    static_seed_rows_list = []
    dynamic_seed_rows_list = []
    if tracks is not None:
        for index, track_id in enumerate(valid_ids):
            seed_type = tracks[track_id].seed_type
            strength = tracks[track_id].seed_strength
            if seed_type in ("STATIC_ANCHOR", "STATIC_GHOST") and strength >= 0.75:
                static_seed_rows_list.append(index)
                logits[index] = -8.0
                logits[index, 0] = 8.0
            elif seed_type == "DYNAMIC_RELATIVE_STATIC" and strength >= 0.75:
                dynamic_seed_rows_list.append(index)
                # Positive evidence says "not STATIC". For K=1 this fixes the
                # moving cluster exactly; for K>1 it leaves the moving-part
                # identity to the relative-rigidity graph.
                logits[index, 0] = -8.0
                if num_groups == 2:
                    logits[index, 1] = 8.0

    static_seed_rows = torch.tensor(static_seed_rows_list, dtype=torch.long)
    dynamic_seed_rows = torch.tensor(dynamic_seed_rows_list, dtype=torch.long)
    logits = torch.nn.Parameter(logits)
    edge_i = torch.tensor([lookup[edge["track_a"]] for edge in edges], dtype=torch.long)
    edge_j = torch.tensor([lookup[edge["track_b"]] for edge in edges], dtype=torch.long)
    rigidity_targets = torch.tensor([edge["q_same"] for edge in edges], dtype=torch.float32)
    spatial_targets = torch.tensor([edge["spatial_same_target"] for edge in edges], dtype=torch.float32)
    temporal_targets = torch.tensor([edge["temporal_same_target"] for edge in edges], dtype=torch.float32)
    uncertain_neighbor_targets = torch.tensor(
        [edge.get("uncertain_neighbor_target", 0.5) for edge in edges], dtype=torch.float32,
    )
    completion_mask = torch.tensor(
        [edge["edge_type"] == "uncertain_core_completion" for edge in edges],
        dtype=torch.bool,
    )
    core_indices = torch.tensor([
        index for index, track_id in enumerate(valid_ids)
        if tracks is None or len(tracks[track_id].points) >= 4
    ], dtype=torch.long)

    def normalized_weights(key: str) -> torch.Tensor:
        values = torch.tensor([edge[key] for edge in edges], dtype=torch.float32)
        return values / values.mean().clamp_min(1e-8)

    rigidity_weights = normalized_weights("weight")
    spatial_weights = normalized_weights("spatial_weight")
    temporal_weights = normalized_weights("temporal_weight")
    uncertain_neighbor_weights = normalized_weights("uncertain_neighbor_weight")
    uncertain_a = torch.tensor([
        tracks is not None and tracks[edge["track_a"]].stationary_occupancy_ambiguous
        for edge in edges
    ], dtype=torch.bool)
    uncertain_b = torch.tensor([
        tracks is not None and tracks[edge["track_b"]].stationary_occupancy_ambiguous
        for edge in edges
    ], dtype=torch.bool)
    static_sum = np.zeros(len(valid_ids), np.float64)
    static_count = np.zeros(len(valid_ids), np.float64)
    for edge in edges:
        for suffix in ("a", "b"):
            index = lookup[edge[f"track_{suffix}"]]
            static_sum[index] += edge[f"track_static_probability_{suffix}"]
            static_count[index] += 1.0
    static_values = static_sum / np.maximum(static_count, 1.0)
    static_strength = np.abs(2.0 * static_values - 1.0)
    if tracks is not None:
        for index, track_id in enumerate(valid_ids):
            prior_weight = tracks[track_id].static_prior_weight
            static_values[index] = tracks[track_id].static_prior
            static_strength[index] = prior_weight * abs(
                2.0 * tracks[track_id].static_prior - 1.0
            )
    static_targets = torch.tensor(static_values, dtype=torch.float32)
    static_certainty = torch.tensor(static_strength, dtype=torch.float32)
    optimizer = torch.optim.Adam([logits], lr=learning_rate)
    history = []
    minimum_use = 0.02
    for iteration in range(iterations):
        optimizer.zero_grad(set_to_none=True)
        probabilities = torch.softmax(logits, dim=1)
        symmetric_same = torch.sum(
            probabilities[edge_i] * probabilities[edge_j], dim=1,
        )
        teacher_same = torch.sum(
            probabilities[edge_i].detach() * probabilities[edge_j], dim=1,
        )
        predicted_same = torch.where(
            completion_mask, teacher_same, symmetric_same,
        ).clamp(1e-6, 1 - 1e-6)
        uncertain_neighbor_same = torch.where(
            uncertain_a & ~uncertain_b,
            torch.sum(probabilities[edge_i] * probabilities[edge_j].detach(), dim=1),
            torch.where(
                uncertain_b & ~uncertain_a,
                torch.sum(probabilities[edge_i].detach() * probabilities[edge_j], dim=1),
                symmetric_same,
            ),
        ).clamp(1e-6, 1 - 1e-6)

        def weighted_binary_loss(
            predictions: torch.Tensor,
            targets: torch.Tensor,
            weights: torch.Tensor,
        ) -> torch.Tensor:
            losses = -targets * torch.log(predictions) - (1.0 - targets) * torch.log(1.0 - predictions)
            return torch.sum(weights * losses) / torch.sum(weights).clamp_min(1e-8)

        core_mask = ~completion_mask
        completion_weights = completion_mask.float()
        rigidity_loss = weighted_binary_loss(predicted_same, rigidity_targets, rigidity_weights)
        spatial_loss = weighted_binary_loss(
            predicted_same, spatial_targets, spatial_weights * core_mask,
        )
        temporal_loss = weighted_binary_loss(
            predicted_same, temporal_targets, temporal_weights * core_mask,
        )
        completion_spatial_loss = weighted_binary_loss(
            predicted_same, spatial_targets, spatial_weights * completion_weights,
        )
        completion_temporal_loss = weighted_binary_loss(
            predicted_same, temporal_targets, temporal_weights * completion_weights,
        )
        completion_loss = completion_spatial_loss + 0.5 * completion_temporal_loss
        uncertain_neighbor_loss = weighted_binary_loss(
            uncertain_neighbor_same,
            uncertain_neighbor_targets,
            uncertain_neighbor_weights,
        )
        static_probability = probabilities[:, 0].clamp(1e-6, 1 - 1e-6)
        static_losses = (
            -static_targets * torch.log(static_probability) -
            (1.0 - static_targets) * torch.log(1.0 - static_probability)
        )
        core_node_mask = torch.zeros(len(valid_ids), dtype=torch.float32)
        core_node_mask[core_indices] = 1.0
        uncertain_node_mask = 1.0 - core_node_mask
        core_static_weights = static_certainty * core_node_mask
        uncertain_static_weights = static_certainty * uncertain_node_mask
        static_loss = torch.sum(core_static_weights * static_losses) / core_static_weights.sum().clamp_min(1e-8)
        uncertain_unary_loss = (
            torch.sum(uncertain_static_weights * static_losses) /
            uncertain_static_weights.sum().clamp_min(1e-8)
        )
        completion_loss = completion_loss + uncertain_unary_loss
        core_probabilities = probabilities[core_indices]
        entropy = -(
            core_probabilities * core_probabilities.clamp_min(1e-8).log()
        ).sum(dim=1).mean()
        occupancy = core_probabilities.mean(dim=0)
        occupancy_penalty = torch.relu(minimum_use - occupancy).square().sum()
        entropy_weight = 0.02 * (1.0 - iteration / max(iterations - 1, 1)) + 0.002
        loss = (
            rigidity_loss + spatial_loss_weight * spatial_loss +
            temporal_loss_weight * temporal_loss + static_loss_weight * static_loss +
            completion_loss_weight * completion_loss +
            uncertain_neighbor_loss_weight * uncertain_neighbor_loss +
            entropy_weight * entropy + 0.25 * occupancy_penalty
        )
        loss.backward()
        optimizer.step()

        # Preserve trusted positive seeds. Static anchors are the reliable
        # evidence in this run; dynamic seeds require measured change of
        # relative distance to multiple trusted static anchors.
        with torch.no_grad():
            if static_seed_rows.numel():
                logits[static_seed_rows] = -8.0
                logits[static_seed_rows, 0] = 8.0
            if dynamic_seed_rows.numel():
                logits[dynamic_seed_rows, 0] = -8.0
                if num_groups == 2:
                    logits[dynamic_seed_rows, 1] = 8.0

        if iteration % 50 == 0 or iteration == iterations - 1:
            rigidity_active = rigidity_weights > 0
            positive = predicted_same[rigidity_active & (rigidity_targets >= 0.5)]
            negative = predicted_same[rigidity_active & (rigidity_targets < 0.5)]
            row = {
                "iteration": iteration,
                "loss": float(loss.detach()),
                "rigidity_loss": float(rigidity_loss.detach()),
                "spatial_consistency_loss": float(spatial_loss.detach()),
                "temporal_consistency_loss": float(temporal_loss.detach()),
                "uncertain_neighbor_loss": float(uncertain_neighbor_loss.detach()),
                "static_anchor_loss": float(static_loss.detach()),
                "uncertain_completion_loss": float(completion_loss.detach()),
                "mean_positive_similarity": float(positive.mean().detach()) if positive.numel() else math.nan,
                "mean_negative_similarity": float(negative.mean().detach()) if negative.numel() else math.nan,
                "assignment_entropy": float(entropy.detach()),
                "cluster_occupancy": json.dumps([round(float(x), 6) for x in occupancy.detach()]),
            }
            history.append(row)
            print(
                f"[CONTRASTIVE] iter={iteration:04d} loss={row['loss']:.6f} "
                f"rigid={row['rigidity_loss']:.4f} spatial={row['spatial_consistency_loss']:.4f} "
                f"temporal={row['temporal_consistency_loss']:.4f} static={row['static_anchor_loss']:.4f} "
                f"completion={row['uncertain_completion_loss']:.4f} "
                f"uncertain_knn={row['uncertain_neighbor_loss']:.4f} "
                f"pos={row['mean_positive_similarity']:.4f} neg={row['mean_negative_similarity']:.4f} "
                f"entropy={row['assignment_entropy']:.4f} occupancy={row['cluster_occupancy']}"
            )
    probabilities = torch.softmax(logits.detach(), dim=1).cpu().numpy()
    hard = np.argmax(probabilities, axis=1)
    # The regularizer is deliberately mild, so enforce only the actual
    # non-empty contract if soft optimization still leaves an unused argmax.
    # This moves at most one donor from a cluster containing >1 tracks and
    # does not impose equal cluster sizes.
    for missing in sorted(set(range(num_groups)) - set(hard.tolist())):
        donor_candidates = [
            index for index in range(len(hard)) if np.sum(hard == hard[index]) > 1
        ]
        if not donor_candidates:
            raise RuntimeError("Contrastive optimizer produced an empty rigid group with no valid donor")
        donor = max(donor_candidates, key=lambda index: probabilities[index, missing])
        hard[donor] = missing
    assignments = {
        track_id: (int(cluster), float(row[cluster]), row.copy())
        for track_id, cluster, row in zip(valid_ids, hard, probabilities)
    }
    return assignments, history



def solve_short_fragments_with_certain_knn(
    tracks: dict[int, Track],
    assignments: dict[int, tuple[int, float, np.ndarray]],
    frames: list[int],
    spacing: float,
    neighbors: int = 8,
    teacher_confidence: float = 0.9,
    edges: list[dict] | None = None,
) -> dict:
    """Distill short fragments from certain kNN teachers and temporal rigidity."""
    groups = len(next(iter(assignments.values()))[2])
    uniform = np.full(groups, 1.0 / groups, np.float64)
    initialized_uniform = 0
    for track_id in tracks:
        if track_id in assignments:
            continue
        assignments[track_id] = (0, float(uniform[0]), uniform.copy())
        initialized_uniform += 1
    short_ids = {
        track_id for track_id in tracks
        if len(tracks[track_id].points) < 4
    }
    votes = {track_id: np.zeros(groups, np.float64) for track_id in short_ids}
    weights = {track_id: 0.0 for track_id in short_ids}
    evidence_count = {track_id: 0 for track_id in short_ids}

    if edges is None:
        edges = []

    for edge in edges:
        if edge.get("edge_type") != "uncertain_core_completion":
            continue
        core_id = int(edge["track_a"])
        fragment_id = int(edge["track_b"])
        if fragment_id not in short_ids or core_id not in assignments:
            continue
        if edge.get("common_frames", 0) < 2:
            continue
        core_cluster, core_confidence, core_probabilities = assignments[core_id]
        if core_confidence < teacher_confidence:
            continue
        target = float(edge.get("temporal_same_target", 0.5))
        certainty = abs(2.0 * target - 1.0)
        if certainty < 0.60:  # equivalent to target in (0.2, 0.8)
            continue
        base_weight = float(edge.get("temporal_weight", 0.0))
        weight = max(base_weight, 0.05) * certainty * core_confidence

        if target >= 0.80:
            proposal = np.asarray(core_probabilities, np.float64)
        elif groups == 2:
            # With STATIC + one moving part, strong "different rigid" evidence
            # against a confident teacher uniquely determines the other group.
            proposal = np.asarray(core_probabilities[::-1], np.float64)
        else:
            # For K>1, "different from teacher" does not identify which moving
            # part. Distribute evidence over all alternatives and let multiple
            # teachers resolve the identity.
            proposal = 1.0 - np.asarray(core_probabilities, np.float64)
            proposal /= max(float(np.sum(proposal)), EPS)

        votes[fragment_id] += weight * proposal
        weights[fragment_id] += weight
        evidence_count[fragment_id] += 1
        tracks[fragment_id].knn_teacher_count += 1

    # A singleton has no own temporal derivative. It can still learn from
    # nearby teachers whose labels are temporally established. Occupancy-only
    # static tracks are teachers only when they persist for at least a third
    # of the sequence, preventing short-lived moving interiors from teaching
    # the STATIC label merely because they refill one world cell.
    minimum_static_teacher_length = max(4, math.ceil(len(frames) / 3))
    radius = 6.0 * spacing
    scale = 3.0 * spacing
    teacher_observations = 0
    ghost_teacher_observations = 0
    for frame in frames:
        teachers = []
        for track_id, (cluster, confidence, _probabilities) in assignments.items():
            track = tracks[track_id]
            trusted_ghost = track.seed_type == "STATIC_GHOST" and track.seed_strength >= 0.75
            if (
                (len(track.points) < 4 and not trusted_ghost)
                or confidence < teacher_confidence
                or frame not in track.points
            ):
                continue
            if (
                cluster == 0 and track.stationary_occupancy_ambiguous and
                len(track.points) < minimum_static_teacher_length
            ):
                continue
            teachers.append(track_id)
        students = [track_id for track_id in short_ids if frame in tracks[track_id].points]
        if not teachers or not students:
            continue
        teacher_xyz = np.stack([tracks[track_id].points[frame] for track_id in teachers])
        student_xyz = np.stack([tracks[track_id].points[frame] for track_id in students])
        count = min(neighbors, len(teachers))
        distances, indices = cKDTree(teacher_xyz).query(
            student_xyz, k=count, distance_upper_bound=radius,
        )
        if count == 1:
            distances, indices = distances[:, None], indices[:, None]
        for student_index, fragment_id in enumerate(students):
            for distance, teacher_index in zip(distances[student_index], indices[student_index]):
                if not np.isfinite(distance) or teacher_index >= len(teachers):
                    continue
                teacher_id = teachers[int(teacher_index)]
                confidence = assignments[teacher_id][1]
                ghost_multiplier = 2.0 if tracks[teacher_id].seed_type == "STATIC_GHOST" else 1.0
                weight = 0.5 * ghost_multiplier * confidence * math.exp(
                    -0.5 * (float(distance) / max(scale, EPS)) ** 2
                )
                votes[fragment_id] += weight * assignments[teacher_id][2]
                weights[fragment_id] += weight
                evidence_count[fragment_id] += 1
                tracks[fragment_id].knn_teacher_count += 1
                teacher_observations += 1
                ghost_teacher_observations += int(ghost_multiplier > 1.0)

    solved = unsupported = changed = dynamic_from_static = 0
    for track_id in short_ids:
        old_cluster = assignments[track_id][0]

        minimum_evidence = min(3, neighbors) if len(tracks[track_id].points) == 1 else 2
        if evidence_count[track_id] < minimum_evidence or weights[track_id] <= EPS:
            probabilities = uniform.copy()
            unsupported += 1
        else:
            probabilities = votes[track_id] / weights[track_id]
            probabilities /= max(float(np.sum(probabilities)), EPS)
            solved += 1
            if groups == 2 and probabilities[1] > 0.75:
                dynamic_from_static += 1

        cluster = int(np.argmax(probabilities))
        changed += cluster != old_cluster
        assignments[track_id] = (
            cluster,
            float(probabilities[cluster]),
            probabilities,
        )

    return {
        "short_fragments": len(short_ids),
        "previously_unoptimized_initialized_uncertain": initialized_uniform,
        "solved_from_certain_knn": solved,
        "left_uniform_uncertain": unsupported,
        "changed_clusters": changed,
        "teacher_confidence": teacher_confidence,
        "neighbors": neighbors,
        "radius": radius,
        "method": "certain same-frame kNN distillation plus temporal relative-distance evidence",
        "teacher_observations": teacher_observations,
        "ghost_teacher_observations": ghost_teacher_observations,
        "minimum_static_teacher_length": minimum_static_teacher_length,
        "dynamic_from_static_relative_evidence": dynamic_from_static,
    }

def final_temporal_hungarian_coherence(
    clouds: dict[int, FrameCloud],
    frames: list[int],
    tracks: dict[int, Track],
    assignments: dict[int, tuple[int, float, np.ndarray]],
    matcher_kwargs: dict,
    max_gap: int,
    label_cost_weight: float,
    teacher_confidence: float,
) -> tuple[list[dict], dict]:
    """Recover fragmented trajectories with label-aware relative Hungarian matching."""
    groups = len(next(iter(assignments.values()))[2])
    uniform_similarity = 1.0 / groups
    point_to_track = {
        (frame, point_id): track_id
        for track_id, track in tracks.items()
        for frame, point_id in track.point_ids.items()
    }
    frame_probabilities = {}
    for frame in frames:
        probabilities = np.full((len(clouds[frame].xyz), groups), 1.0 / groups, np.float64)
        for point_id in range(len(probabilities)):
            track_id = point_to_track.get((frame, point_id))
            if track_id in assignments:
                probabilities[point_id] = assignments[track_id][2]
        frame_probabilities[frame] = probabilities
    signatures = {
        frame: sparse_graph_signatures(clouds[frame].xyz, matcher_kwargs["num_rel_neighbors"])
        for frame in frames
    }

    parent = {track_id: track_id for track_id in assignments}
    component_frames = {track_id: set(tracks[track_id].points) for track_id in assignments}

    def find(track_id: int) -> int:
        while parent[track_id] != track_id:
            parent[track_id] = parent[parent[track_id]]
            track_id = parent[track_id]
        return track_id

    def merge(track_a: int, track_b: int) -> bool:
        root_a, root_b = find(track_a), find(track_b)
        if root_a == root_b:
            return False
        if component_frames[root_a] & component_frames[root_b]:
            return False
        if root_b < root_a:
            root_a, root_b = root_b, root_a
        parent[root_b] = root_a
        component_frames[root_a] |= component_frames.pop(root_b)
        return True

    frame_pairs = list(zip(frames[:-1], frames[1:]))
    if max_gap > 0:
        frame_pairs += list(zip(frames[:-2], frames[2:]))
    rows = []
    merged_links = label_rejected = overlap_rejected = relaxed_geometry_links = 0
    for frame_a, frame_b in frame_pairs:
        _accepted, diagnostics, _stats = match_frame_pair(
            clouds[frame_a].xyz,
            clouds[frame_b].xyz,
            **matcher_kwargs,
            signature_a=signatures[frame_a],
            signature_b=signatures[frame_b],
            label_probabilities_a=frame_probabilities[frame_a],
            label_probabilities_b=frame_probabilities[frame_b],
            label_cost_weight=label_cost_weight,
        )
        for record in diagnostics:
            if record["i"] < 0 or record["j"] < 0:
                continue
            # The first pass deliberately rejected weak-margin assignments.
            # After label discovery, a geometrically supported one-to-one match
            # may be recovered when both endpoints strongly agree. Never relax
            # low compatibility support or a skip-frame conflict.
            relaxed_geometry = (
                not record["accepted"]
                and record["reject_reason"] == "weak_assignment_margin"
                and record["relational_support"] >= matcher_kwargs["min_relational_support"]
                and record["relative_cost"] <= 0.12
                and record["label_similarity"] >= 0.80
            )
            if not record["accepted"] and not relaxed_geometry:
                continue
            track_a = point_to_track[(frame_a, record["i"])]
            track_b = point_to_track[(frame_b, record["j"])]
            if track_a == track_b or track_a not in assignments or track_b not in assignments:
                continue
            similarity = float(record["label_similarity"])
            reason = ""
            merged = False
            if similarity + EPS < uniform_similarity:
                reason = "label_conflict"
                label_rejected += 1
            elif component_frames[find(track_a)] & component_frames[find(track_b)]:
                reason = "trajectory_frame_overlap"
                overlap_rejected += 1
            else:
                merged = merge(track_a, track_b)
                if merged:
                    tracks[track_a].temporal_coherence_links += 1
                    tracks[track_b].temporal_coherence_links += 1
                    merged_links += 1
                    relaxed_geometry_links += int(relaxed_geometry)
            rows.append({
                "frame_a": frame_a,
                "frame_b": frame_b,
                "track_a": track_a,
                "track_b": track_b,
                "label_similarity": similarity,
                "relative_cost": record["relative_cost"],
                "relational_support": record["relational_support"],
                "assignment_margin": record["assignment_margin"],
                "merged": merged,
                "relaxed_geometry_recovery": relaxed_geometry,
                "reject_reason": reason,
            })

    components: dict[int, list[int]] = {}
    for track_id in assignments:
        components.setdefault(find(track_id), []).append(track_id)
    recovered = [members for members in components.values() if len(members) > 1]
    changed = 0
    for trajectory_id, members in enumerate(sorted(recovered, key=lambda values: min(values))):
        certain_core = [
            track_id for track_id in members
            if len(tracks[track_id].points) >= 4 and assignments[track_id][1] >= teacher_confidence
        ]
        teachers = certain_core or [
            track_id for track_id in members if assignments[track_id][1] >= teacher_confidence
        ] or members
        teacher_weights = np.asarray([max(assignments[track_id][1], EPS) for track_id in teachers])
        target = np.average(
            np.stack([assignments[track_id][2] for track_id in teachers]),
            axis=0,
            weights=teacher_weights,
        )
        target /= max(float(np.sum(target)), EPS)
        for track_id in members:
            tracks[track_id].temporal_trajectory_id = trajectory_id
            tracks[track_id].temporal_coherence_links = max(
                tracks[track_id].temporal_coherence_links, len(members) - 1,
            )
            if len(tracks[track_id].points) >= 4 and assignments[track_id][1] >= teacher_confidence:
                continue
            old_cluster = assignments[track_id][0]
            cluster = int(np.argmax(target))
            changed += cluster != old_cluster
            assignments[track_id] = (cluster, float(target[cluster]), target.copy())
    return rows, {
        "hungarian_frame_pairs": len(frame_pairs),
        "candidate_cross_track_links": len(rows),
        "merged_fragment_links": merged_links,
        "relaxed_weak_margin_links": relaxed_geometry_links,
        "label_conflict_rejections": label_rejected,
        "frame_overlap_rejections": overlap_rejected,
        "recovered_trajectories": len(recovered),
        "tracks_in_recovered_trajectories": sum(map(len, recovered)),
        "changed_fragment_clusters": changed,
        "label_cost_weight": label_cost_weight,
    }


def fit_rigid_transform(source: np.ndarray, target: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    if len(source) < 3:
        raise ValueError("At least three correspondences are needed for SE(3)")
    source_center = np.mean(source, axis=0)
    target_center = np.mean(target, axis=0)
    u, _s, vt = np.linalg.svd((source - source_center).T @ (target - target_center))
    rotation = vt.T @ u.T
    if np.linalg.det(rotation) < 0:
        vt[-1] *= -1
        rotation = vt.T @ u.T
    translation = target_center - rotation @ source_center
    residuals = np.linalg.norm(source @ rotation.T + translation - target, axis=1)
    return rotation, translation, float(np.median(residuals))


def rotation_angle(rotation: np.ndarray) -> float:
    return float(math.acos(float(np.clip((np.trace(rotation) - 1.0) * 0.5, -1.0, 1.0))))


def fit_group_transforms(
    tracks: dict[int, Track],
    assignments: dict[int, tuple[int, float, np.ndarray]],
    num_groups: int,
    frames: list[int],
    scale: float,
) -> tuple[list[dict], dict[int, float], int]:
    rows = []
    scores = {}
    for cluster in range(num_groups):
        ids = [track_id for track_id, value in assignments.items() if value[0] == cluster]
        reference = max(frames, key=lambda frame: sum(frame in tracks[track_id].points for track_id in ids))
        identity_scores = []
        for frame in frames:
            common = [track_id for track_id in ids if reference in tracks[track_id].points and frame in tracks[track_id].points]
            if len(common) < 3:
                continue
            source = np.stack([tracks[track_id].points[reference] for track_id in common])
            target = np.stack([tracks[track_id].points[frame] for track_id in common])
            rotation, translation, residual = fit_rigid_transform(source, target)
            angle = rotation_angle(rotation)
            identity = angle + np.linalg.norm(translation) / max(scale, EPS)
            identity_scores.append(float(identity))
            rows.append({
                "cluster_id": cluster,
                "reference_frame": reference,
                "frame": frame,
                "num_correspondences": len(common),
                "rotation_angle_rad": angle,
                "translation_norm": float(np.linalg.norm(translation)),
                "median_residual": residual,
                "identity_motion_score": float(identity),
                **{f"R{r}{c}": float(rotation[r, c]) for r in range(3) for c in range(3)},
                "tx": float(translation[0]), "ty": float(translation[1]), "tz": float(translation[2]),
            })
        scores[cluster] = float(np.median(identity_scores)) if identity_scores else math.inf
    static_cluster = min(scores, key=scores.get)
    return rows, scores, static_cluster


def apply_final_labels(
    tracks: dict[int, Track],
    assignments: dict[int, tuple[int, float, np.ndarray]],
    scores: dict[int, float],
    static_cluster: int,
) -> dict[int, str]:
    moving_clusters = sorted((cluster for cluster in scores if cluster != static_cluster), key=lambda c: (-scores[c], c))
    names = {static_cluster: "STATIC"}
    names.update({cluster: f"MOVING_{index + 1}" for index, cluster in enumerate(moving_clusters)})
    for track_id, track in tracks.items():
        if track_id not in assignments:
            continue
        cluster, confidence, probabilities = assignments[track_id]
        track.cluster_id = cluster
        track.confidence = confidence
        track.probabilities = probabilities
        track.label = (
            names[cluster] if confidence >= CONFIDENT_LABEL_THRESHOLD else "AMBIGUOUS"
        )
        if track.initial_cluster_id < 0:
            track.initial_cluster_id = cluster
            track.initial_confidence = confidence
            track.initial_label = track.label
        track.label_source = (
            "final_temporal_hungarian_coherence"
            if track.temporal_coherence_links > 0
            else "certain_knn_fragment_distillation"
            if track.knn_teacher_count > 0 and len(track.points) < 4
            else "contrastive_uncertain_temporal_knn"
            if track.stationary_occupancy_ambiguous
            else "contrastive_uncertain_completion"
            if len(track.points) < 4
            else "contrastive_temporal"
        )
    return names


# =============================================================================
# Static-first full-resolution residual graph matching
# =============================================================================

def trusted_static_track(track: Track) -> bool:
    return (
        track.seed_type in (
            "STATIC_ANCHOR", "STATIC_GHOST", "STATIC_RECOVERED",
            "STATIC_TEMPORAL_REAPPEARANCE",
        )
        and track.seed_strength >= 0.75
    )


def recover_static_fragments_from_anchors(
    tracks: dict[int, Track],
    spacing: float,
) -> tuple[list[dict], dict]:
    """Recover interrupted static fragments without creating dynamic labels."""
    trusted_ids = [track_id for track_id, track in tracks.items() if trusted_static_track(track)]
    candidate_ids = [
        track_id for track_id, track in tracks.items()
        if track.stationary_occupancy_ambiguous and not trusted_static_track(track)
    ]
    radius = min(0.5 * spacing, 0.01)
    if not trusted_ids or not candidate_ids or radius <= EPS:
        return [], {
            "enabled": True,
            "method": "bidirectional static-fragment recovery from trusted anchors",
            "radius": radius,
            "candidate_tracks": len(candidate_ids),
            "recovered_static_tracks": 0,
            "backward_recoveries": 0,
        }

    relevant_ids = set(trusted_ids) | set(candidate_ids)
    representatives = {
        track_id: np.median(np.stack(list(track.points.values())), axis=0)
        for track_id, track in tracks.items()
        if track_id in relevant_ids
    }

    trusted_xyz = np.stack([representatives[track_id] for track_id in trusted_ids])
    tree = cKDTree(trusted_xyz)
    rows = []
    backward = 0
    neighbor_count = min(8, len(trusted_ids))
    for track_id in candidate_ids:
        distances, indices = tree.query(
            representatives[track_id], k=neighbor_count, distance_upper_bound=radius,
        )
        distances = np.atleast_1d(distances)
        indices = np.atleast_1d(indices)
        source_id = -1
        distance = math.inf
        for candidate_distance, candidate_index in zip(distances, indices):
            if not np.isfinite(candidate_distance) or candidate_index >= len(trusted_ids):
                continue
            candidate_source = trusted_ids[int(candidate_index)]
            if set(tracks[track_id].points).isdisjoint(tracks[candidate_source].points):
                source_id = candidate_source
                distance = float(candidate_distance)
                break
        if source_id < 0:
            continue

        track = tracks[track_id]
        source = tracks[source_id]
        anchor_fraction = track.static_anchor_matches / max(track.temporal_matches, 1)
        track.seed_type = "STATIC_RECOVERED"
        track.seed_strength = min(source.seed_strength, 0.5 + 0.5 * anchor_fraction)
        track.static_prior = 0.995
        track.static_prior_weight = track.seed_strength
        track.ghost_source_track_id = source_id
        track.ghost_distance = distance
        frame_gap = min(
            abs(frame_a - frame_b)
            for frame_a in track.points
            for frame_b in source.points
        )
        track.ghost_age_frames = frame_gap
        is_backward = min(track.points) < min(source.points)
        backward += int(is_backward)
        rows.append({
            "event": "static_fragment_recovered",
            "frame": min(track.points),
            "track_id": track_id,
            "ghost_id": -1,
            "source_track_id": source_id,
            "distance": distance,
            "ghost_age_frames": frame_gap,
            "assigned_cluster_id": 0,
            "static_prior": track.static_prior,
            "static_prior_weight": track.static_prior_weight,
        })
    return rows, {
        "enabled": True,
        "method": (
            "bidirectional recovery of strong stationary fragments from a "
            "spatially coincident trusted static/ghost trajectory"
        ),
        "radius": radius,
        "candidate_tracks": len(candidate_ids),
        "recovered_static_tracks": len(rows),
        "backward_recoveries": backward,
        "unresolved_candidate_tracks": len(candidate_ids) - len(rows),
        "dynamic_labels_created": 0,
    }


def recover_static_from_temporal_reappearance(
    clouds: dict[int, FrameCloud],
    frames: list[int],
    tracks: dict[int, Track],
    spacing: float,
) -> tuple[list[dict], dict]:
    """Recover static tracks whose world location reappears across the sequence."""
    candidate_ids = [track_id for track_id, track in tracks.items() if not trusted_static_track(track)]
    radius = min(0.5 * spacing, 0.01)
    minimum_occurrences = max(6, math.ceil(0.15 * len(frames)))
    if not candidate_ids or radius <= EPS:
        return [], {
            "enabled": True,
            "radius": radius,
            "minimum_occurrences": minimum_occurrences,
            "candidate_tracks": len(candidate_ids),
            "recovered_static_tracks": 0,
            "dynamic_labels_created": 0,
        }

    representatives = np.stack([
        np.median(np.stack(list(tracks[track_id].points.values())), axis=0)
        for track_id in candidate_ids
    ])
    support_samples = np.full((len(candidate_ids), len(frames)), np.nan, np.float32)
    distance_sum = np.zeros(len(candidate_ids), np.float64)
    occurrence_count = np.zeros(len(candidate_ids), np.int32)
    trusted_ids = {track_id for track_id, track in tracks.items() if trusted_static_track(track)}

    for frame_index, frame in enumerate(frames):
        cloud = clouds[frame]
        owner = np.full(len(cloud.xyz), -1, np.int64)
        for track_id, track in tracks.items():
            point_id = track.point_ids.get(frame)
            if point_id is not None:
                owner[point_id] = track_id
        if np.any(owner < 0):
            raise RuntimeError(f"Temporal static recovery found unowned points in frame {frame}")

        tree = cKDTree(cloud.xyz)
        neighbor_count = min(9, len(cloud.xyz))
        _neighbor_distances, neighbor_ids = tree.query(cloud.xyz, k=neighbor_count)
        if neighbor_count == 1:
            local_static_support = np.zeros(len(cloud.xyz), np.float64)
        else:
            local_static_support = np.isin(
                owner[neighbor_ids[:, 1:]], list(trusted_ids),
            ).mean(axis=1)

        distances, nearest_ids = tree.query(representatives, k=1)
        occupied = distances <= radius
        occurrence_count += occupied
        distance_sum += np.where(occupied, distances, 0.0)
        support_samples[occupied, frame_index] = local_static_support[nearest_ids[occupied]]

    support_p25 = np.zeros(len(candidate_ids), np.float64)
    observed = occurrence_count > 0
    support_p25[observed] = np.nanpercentile(
        support_samples[observed], 25, axis=1,
    )
    rows = []
    for index, track_id in enumerate(candidate_ids):
        track = tracks[track_id]
        reappearance_observations = int(occurrence_count[index]) - len(track.points)
        if (
            occurrence_count[index] < minimum_occurrences
            or reappearance_observations < 3
            or support_p25[index] < 0.375
        ):
            continue
        recurrence = float(occurrence_count[index]) / len(frames)
        track.seed_type = "STATIC_TEMPORAL_REAPPEARANCE"
        track.seed_strength = min(1.0, 0.75 + 0.125 * recurrence + 0.125 * support_p25[index])
        track.static_prior = 0.995
        track.static_prior_weight = track.seed_strength
        rows.append({
            "track_id": track_id,
            "first_frame": min(track.points),
            "last_frame": max(track.points),
            "track_length": len(track.points),
            "temporal_occurrences": int(occurrence_count[index]),
            "reappearance_observations": reappearance_observations,
            "local_static_support_p25": float(support_p25[index]),
            "mean_reappearance_distance": float(
                distance_sum[index] / max(occurrence_count[index], 1)
            ),
            "radius": radius,
            "seed_strength": track.seed_strength,
        })
    return rows, {
        "enabled": True,
        "method": (
            "whole-sequence world-location recurrence with persistent local "
            "trusted-static neighborhood support"
        ),
        "radius": radius,
        "minimum_occurrences": minimum_occurrences,
        "minimum_reappearance_observations": 3,
        "minimum_local_static_support_p25": 0.375,
        "candidate_tracks": len(candidate_ids),
        "recovered_static_tracks": len(rows),
        "unresolved_tracks": len(candidate_ids) - len(rows),
        "dynamic_labels_created": 0,
    }


def raw_to_coarse_point_ids(cloud: FrameCloud, voxel_size: float) -> np.ndarray:
    """Recover the fixed-lattice cell id for every original depth sample."""
    raw_voxels = np.floor(cloud.raw_xyz / voxel_size).astype(np.int64)
    combined = np.concatenate((cloud.voxel_indices, raw_voxels), axis=0)
    _unique, inverse = np.unique(combined, axis=0, return_inverse=True)
    lookup = np.full(int(inverse.max()) + 1, -1, np.int32)
    lookup[inverse[:len(cloud.voxel_indices)]] = np.arange(
        len(cloud.voxel_indices), dtype=np.int32,
    )
    result = lookup[inverse[len(cloud.voxel_indices):]]
    if np.any(result < 0):
        raise RuntimeError(f"Raw/coarse voxel provenance failed in frame {cloud.frame}")
    return result


def fine_geometric_deduplication(
    xyz: np.ndarray,
    coarse_ids: np.ndarray,
    tolerance: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Fuse only near-identical multi-camera samples; never apply FPS."""
    if not len(xyz):
        return (
            np.empty((0, 3), np.float64),
            np.empty(0, np.int32),
            np.empty(0, np.int32),
        )
    cells = np.floor(xyz / tolerance).astype(np.int64)
    _unique, inverse = np.unique(cells, axis=0, return_inverse=True)
    counts = np.bincount(inverse).astype(np.int32)
    fused = np.zeros((len(counts), 3), np.float64)
    np.add.at(fused, inverse, xyz)
    fused /= counts[:, None]
    order = np.argsort(inverse, kind="stable")
    offsets = np.concatenate(([0], np.cumsum(counts)))
    provenance = np.empty(len(counts), np.int32)
    for cell_id in range(len(counts)):
        members = coarse_ids[order[offsets[cell_id]:offsets[cell_id + 1]]]
        values, value_counts = np.unique(members, return_counts=True)
        provenance[cell_id] = int(values[np.argmax(value_counts)])
    return fused, counts, provenance


def adaptive_geometric_coreset(
    xyz: np.ndarray,
    members: np.ndarray,
    coarse_ids: np.ndarray,
    target_points: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """Build a spatially balanced per-frame coreset with no temporal cell identity.

    The cell origin and size are recomputed independently per frame. Cell ids
    therefore cannot become correspondences or static evidence; they only cap
    graph size. Each retained node is the observed sample nearest the weighted
    cell centroid, while its weight retains all represented raw samples.
    """
    if target_points <= 0 or len(xyz) <= target_points:
        return xyz, members, coarse_ids, 0.0
    nearest = cKDTree(xyz).query(xyz, k=2)[0][:, 1]
    low = max(float(np.median(nearest)), EPS)
    origin = np.min(xyz, axis=0) - 0.5 * low

    def partition(cell_size: float) -> tuple[np.ndarray, int]:
        cells = np.floor((xyz - origin) / cell_size).astype(np.int64)
        _unique, inverse = np.unique(cells, axis=0, return_inverse=True)
        return inverse, int(inverse.max()) + 1

    high = low
    inverse, count = partition(high)
    while count > target_points:
        high *= 1.35
        inverse, count = partition(high)
    for _ in range(10):
        middle = 0.5 * (low + high)
        trial_inverse, trial_count = partition(middle)
        if trial_count > target_points:
            low = middle
        else:
            high, inverse, count = middle, trial_inverse, trial_count

    raw_weights = np.asarray(members, np.float64)
    weight_sum = np.bincount(inverse, weights=raw_weights, minlength=count)
    centroid = np.zeros((count, 3), np.float64)
    np.add.at(centroid, inverse, xyz * raw_weights[:, None])
    centroid /= np.maximum(weight_sum[:, None], EPS)
    order = np.argsort(inverse, kind="stable")
    counts = np.bincount(inverse, minlength=count)
    offsets = np.concatenate(([0], np.cumsum(counts)))
    selected = np.empty(count, np.int32)
    provenance = np.empty(count, np.int32)
    for cell_id in range(count):
        local = order[offsets[cell_id]:offsets[cell_id + 1]]
        selected[cell_id] = int(local[np.argmin(np.linalg.norm(
            xyz[local] - centroid[cell_id], axis=1,
        ))])
        values, value_inverse = np.unique(coarse_ids[local], return_inverse=True)
        votes = np.bincount(value_inverse, weights=raw_weights[local])
        provenance[cell_id] = int(values[np.argmax(votes)])
    return (
        xyz[selected],
        np.rint(weight_sum).astype(np.int32),
        provenance,
        float(high),
    )


def build_full_resolution_residuals(
    clouds: dict[int, FrameCloud],
    frames: list[int],
    coarse_tracks: dict[int, Track],
    out: Path,
    fine_tolerance: float,
    target_points: int = 4096,
    save_fine_full_resolution: bool = False,
) -> dict[int, ResidualCloud]:
    """Remove STATIC, fuse raw depth finely, then build a bounded graph coreset."""
    owner = {
        (frame, point_id): track
        for track in coarse_tracks.values()
        for frame, point_id in track.point_ids.items()
    }
    residuals: dict[int, ResidualCloud] = {}
    root = out / "residual_fullres"
    for frame in frames:
        cloud = clouds[frame]
        raw_coarse_ids = raw_to_coarse_point_ids(cloud, cloud.quality["voxel_size"])
        static_coarse = np.asarray([
            trusted_static_track(owner[(frame, point_id)])
            if (frame, point_id) in owner else False
            for point_id in range(len(cloud.xyz))
        ], dtype=bool)
        keep = ~static_coarse[raw_coarse_ids]
        residual_raw = cloud.raw_xyz[keep]
        fine_xyz, fine_members, fine_coarse_ids = fine_geometric_deduplication(
            residual_raw, raw_coarse_ids[keep], fine_tolerance,
        )
        if len(fine_xyz) > 1:
            fine_nearest = cKDTree(fine_xyz).query(fine_xyz, k=2)[0][:, 1]
            fine_spacing = float(np.median(fine_nearest))
        else:
            fine_spacing = 0.0
        fused, members, coarse_ids, coreset_cell_size = adaptive_geometric_coreset(
            fine_xyz, fine_members, fine_coarse_ids, target_points,
        )
        if len(fused) > 1:
            nearest = cKDTree(fused).query(fused, k=2)[0][:, 1]
            spacing = float(np.median(nearest))
        else:
            spacing = 0.0
        quality = {
            "frame": int(frame),
            "raw_residual_points": int(len(residual_raw)),
            "fine_fused_residual_points": int(len(fine_xyz)),
            "fused_residual_points": int(len(fused)),
            "graph_coreset_points": int(len(fused)),
            "removed_static_points": int(np.count_nonzero(~keep)),
            "fraction_retained": float(np.mean(keep)) if len(keep) else 0.0,
            "graph_fraction_of_fine_residual": float(len(fused) / max(len(fine_xyz), 1)),
            "fine_median_spacing": fine_spacing,
            "median_spacing": spacing,
            "fine_geometric_deduplication_tolerance": float(fine_tolerance),
            "graph_target_points": int(target_points),
            "adaptive_coreset_cell_size": coreset_cell_size,
            "sampling": (
                "fine raw-depth fusion followed by a per-frame adaptive geometric coreset; "
                "coreset cells have no temporal identity and are never matching evidence"
            ),
        }
        residual = ResidualCloud(frame, fused, members, coarse_ids, quality)
        residuals[frame] = residual
        folder = root / f"frame_{frame:06d}"
        folder.mkdir(parents=True, exist_ok=True)
        write_ply(folder / "points_world.ply", fused, {
            "point_id": np.arange(len(fused), dtype=np.int32),
            "num_raw_members": members,
            "coarse_point_id": coarse_ids,
        })
        if save_fine_full_resolution and len(fine_xyz) != len(fused):
            write_ply(folder / "points_fine_full_resolution.ply", fine_xyz, {
                "num_raw_members": fine_members,
                "coarse_point_id": fine_coarse_ids,
            })
        (folder / "quality.json").write_text(json.dumps(quality, indent=2) + "\n")
        print(
            f"[RESIDUAL CORESET] frame={frame:06d} raw={len(residual_raw)} "
            f"fine={len(fine_xyz)} graph={len(fused)} removed_static={np.count_nonzero(~keep)} "
            f"cell={coreset_cell_size:.6g} spacing={spacing:.6g}"
        )
    return residuals


def build_residual_graph(
    cloud: ResidualCloud,
    near_neighbors: int,
    medium_neighbors: int,
    far_neighbors: int,
) -> ResidualGraph:
    total_requested = near_neighbors + medium_neighbors + far_neighbors
    if len(cloud.xyz) < 2:
        return ResidualGraph(
            cloud.frame, cloud.xyz, np.empty((len(cloud.xyz), 0), np.int32),
            np.empty((len(cloud.xyz), 0)), np.empty((len(cloud.xyz), 0), np.int8),
            np.empty((len(cloud.xyz), 0)), max(cloud.quality["median_spacing"], EPS),
        )
    count = min(total_requested, len(cloud.xyz) - 1)
    distances, neighbors = cKDTree(cloud.xyz).query(cloud.xyz, k=count + 1)
    distances = np.asarray(distances[:, 1:], np.float64)
    neighbors = np.asarray(neighbors[:, 1:], np.int32)
    scales = np.full(count, 2, np.int8)
    scales[:min(near_neighbors, count)] = 0
    scales[min(near_neighbors, count):min(near_neighbors + medium_neighbors, count)] = 1
    edge_scales = np.broadcast_to(scales[None], distances.shape).copy()
    spacing = max(float(np.median(distances[:, 0])), EPS)
    signatures = np.log1p(distances / spacing)
    return ResidualGraph(
        cloud.frame, cloud.xyz, neighbors, distances, edge_scales, signatures, spacing,
    )


def build_residual_graphs(
    residuals: dict[int, ResidualCloud],
    near_neighbors: int,
    medium_neighbors: int,
    far_neighbors: int,
) -> dict[int, ResidualGraph]:
    return {
        frame: build_residual_graph(
            cloud, near_neighbors, medium_neighbors, far_neighbors,
        )
        for frame, cloud in residuals.items()
    }


def residual_signature_candidates(
    source: ResidualGraph,
    target: ResidualGraph,
    candidates: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Permissive candidates from sorted local distances, never world XYZ."""
    if not len(source.xyz) or not len(target.xyz):
        return (np.empty(0, np.int32), np.empty(0, np.int32),
                np.empty(0), np.zeros(len(source.xyz), np.int32))
    dimensions = min(source.signatures.shape[1], target.signatures.shape[1])
    features_a = source.signatures[:, :dimensions]
    features_b = target.signatures[:, :dimensions]
    forward_count = min(candidates, len(target.xyz))
    reverse_count = min(candidates, len(source.xyz))
    forward = cKDTree(features_b).query(features_a, k=forward_count)[1]
    reverse = cKDTree(features_a).query(features_b, k=reverse_count)[1]
    if forward_count == 1:
        forward = forward[:, None]
    if reverse_count == 1:
        reverse = reverse[:, None]
    sets = [set(map(int, row)) for row in np.asarray(forward)]
    for destination, source_ids in enumerate(np.asarray(reverse)):
        for source_id in np.atleast_1d(source_ids):
            sets[int(source_id)].add(int(destination))
    maximum = min(candidates, len(target.xyz))
    edge_i_parts, edge_j_parts, costs, counts = [], [], [], []
    for source_id, destinations in enumerate(sets):
        ids = np.asarray(sorted(destinations), np.int32)
        diff = features_b[ids] - features_a[source_id]
        local_cost = np.median(np.abs(diff), axis=1)
        if len(ids) > maximum:
            retained = np.argpartition(local_cost, maximum - 1)[:maximum]
            ids, local_cost = ids[retained], local_cost[retained]
            order = np.argsort(ids)
            ids, local_cost = ids[order], local_cost[order]
        edge_i_parts.append(np.full(len(ids), source_id, np.int32))
        edge_j_parts.append(ids)
        costs.append(local_cost)
        counts.append(len(ids))
    return (
        np.concatenate(edge_i_parts), np.concatenate(edge_j_parts),
        np.concatenate(costs), np.asarray(counts, np.int32),
    )


def residual_candidate_compatibility(
    source: ResidualGraph,
    target: ResidualGraph,
    edge_i: np.ndarray,
    edge_j: np.ndarray,
    counts: np.ndarray,
    relative_threshold: float,
    source_p_dynamic: np.ndarray | None = None,
    target_p_dynamic: np.ndarray | None = None,
) -> csr_matrix:
    """Sparse candidate-pair graph based solely on edge-length preservation."""
    if not len(edge_i):
        return csr_matrix((0, 0), dtype=np.float64)

    def representative_ranks(graph: ResidualGraph) -> np.ndarray:
        ranks = []
        for scale_id in range(3):
            available = np.flatnonzero(graph.edge_scales[0] == scale_id)
            if len(available):
                ranks.extend((
                    int(available[0]),
                    int(available[len(available) // 2]),
                    int(available[-1]),
                ))
        return np.asarray(sorted(set(ranks)), np.int32)

    source_ranks = representative_ranks(source)
    target_ranks = representative_ranks(target)
    candidate_keys = edge_i.astype(np.int64) * len(target.xyz) + edge_j
    row_parts, column_parts, weight_parts = [], [], []
    batch_size = 4096
    for start in range(0, len(edge_i), batch_size):
        stop = min(start + batch_size, len(edge_i))
        candidate_ids = np.arange(start, stop, dtype=np.int64)
        source_ids, target_ids = edge_i[start:stop], edge_j[start:stop]
        source_neighbors = source.neighbors[source_ids][:, source_ranks]
        target_neighbors = target.neighbors[target_ids][:, target_ranks]
        keys = (
            source_neighbors[:, :, None].astype(np.int64) * len(target.xyz)
            + target_neighbors[:, None, :]
        )
        flat_keys = keys.reshape(-1)
        positions = np.searchsorted(candidate_keys, flat_keys)
        in_range = positions < len(candidate_keys)
        found = np.zeros(len(flat_keys), dtype=bool)
        found[in_range] = candidate_keys[positions[in_range]] == flat_keys[in_range]
        found = found.reshape(keys.shape)
        if not np.any(found):
            continue
        source_lengths = source.edge_lengths[source_ids][:, source_ranks, None]
        target_lengths = target.edge_lengths[target_ids][:, None, target_ranks]
        error = np.abs(source_lengths - target_lengths) / np.maximum(
            np.maximum(source_lengths, target_lengths),
            max(source.spacing, target.spacing, EPS),
        )
        compatible = found & (error <= relative_threshold)
        batch_index, source_rank, target_rank = np.nonzero(compatible)
        if not len(batch_index):
            continue
        other_ids = positions.reshape(keys.shape)[batch_index, source_rank, target_rank]
        current_ids = candidate_ids[batch_index]
        nonself = current_ids != other_ids
        if not np.any(nonself):
            continue
        current_ids, other_ids = current_ids[nonself], other_ids[nonself]
        selected_error = error[batch_index, source_rank, target_rank][nonself]
        local_weight = np.exp(-0.5 * (selected_error / relative_threshold) ** 2)
        if source_p_dynamic is not None:
            k = source_neighbors[batch_index, source_rank][nonself]
            i = source_ids[batch_index][nonself]
            local_weight *= 0.25 + 0.75 * np.sqrt(np.maximum(
                source_p_dynamic[i] * source_p_dynamic[k], EPS,
            ))
        if target_p_dynamic is not None:
            l = target_neighbors[batch_index, target_rank][nonself]
            j = target_ids[batch_index][nonself]
            local_weight *= 0.25 + 0.75 * np.sqrt(np.maximum(
                target_p_dynamic[j] * target_p_dynamic[l], EPS,
            ))
        row_parts.append(current_ids)
        column_parts.append(other_ids)
        weight_parts.append(local_weight)
    if not row_parts:
        return csr_matrix((len(edge_i), len(edge_i)), dtype=np.float64)
    rows = np.concatenate(row_parts)
    columns = np.concatenate(column_parts)
    weights = np.concatenate(weight_parts)
    return coo_matrix(
        (np.concatenate((weights, weights)),
         (np.concatenate((rows, columns)), np.concatenate((columns, rows)))),
        shape=(len(edge_i), len(edge_i)),
    ).tocsr()


def match_residual_graph_pair(
    source: ResidualGraph,
    target: ResidualGraph,
    candidates: int = 48,
    iterations: int = 5,
    min_support: int = 3,
    relative_threshold: float = 0.08,
    unmatched_scale: float = 2.5,
    source_p_dynamic: np.ndarray | None = None,
    target_p_dynamic: np.ndarray | None = None,
) -> tuple[list[dict], dict]:
    """Iterative relational graph alignment followed by sparse Hungarian."""
    if len(source.xyz) < 2 or len(target.xyz) < 2:
        quality = {
            "frame_a": source.frame, "frame_b": target.frame,
            "source_residual_points": len(source.xyz), "target_residual_points": len(target.xyz),
            "candidate_nodes": 0, "candidate_compatibility_edges": 0,
            "projected_matches": 0, "accepted_matches": 0,
            "unmatched_source": len(source.xyz), "unmatched_target": len(target.xyz),
            "median_edge_deformation": math.nan, "p90_edge_deformation": math.nan,
            "median_graph_support": 0.0, "median_assignment_margin": math.nan,
            "skip_confirmed": 0, "skip_conflict": 0, "new_dynamic_anchors": 0,
        }
        return [], quality
    edge_i, edge_j, signature_cost, counts = residual_signature_candidates(
        source, target, candidates,
    )
    compatibility = residual_candidate_compatibility(
        source, target, edge_i, edge_j, counts, relative_threshold,
        source_p_dynamic, target_p_dynamic,
    )
    # Relational message passing is the optimizer. Hungarian appears once,
    # afterward, solely to enforce a partial one-to-one projection.
    scores, degree = graph_matching_scores(signature_cost, compatibility, iterations)
    scale = max(float(np.median(signature_cost)) + robust_mad(signature_cost), 0.01)
    candidate_cost = -np.log(np.maximum(scores, EPS))
    projected, _unmatched = sparse_one_to_one_projection(
        edge_i, edge_j, candidate_cost, counts,
        len(source.xyz), len(target.xyz), unmatched_scale,
    )

    offsets = np.concatenate(([0], np.cumsum(counts)))
    accepted: list[dict] = []
    for i, j in projected.items():
        deformation, scales = [], set()
        for rank, k_value in enumerate(source.neighbors[i]):
            k = int(k_value)
            l = projected.get(k)
            if l is None:
                continue
            left = float(source.edge_lengths[i, rank])
            right = float(np.linalg.norm(target.xyz[j] - target.xyz[l]))
            relative_error = abs(left - right) / max(
                left, right, source.spacing, target.spacing, EPS,
            )
            if relative_error > 2.0 * relative_threshold:
                continue
            deformation.append(relative_error)
            scales.add(int(source.edge_scales[i, rank]))
        if not deformation:
            continue
        deformation_array = np.asarray(deformation)
        edge_ids = np.arange(offsets[i], offsets[i + 1])
        selected_ids = edge_ids[edge_j[edge_ids] == j]
        if not len(selected_ids):
            continue
        selected_id = int(selected_ids[0])
        row_costs = np.sort(candidate_cost[edge_ids])
        selected_cost = float(candidate_cost[selected_id])
        second = float(row_costs[1]) if len(row_costs) > 1 else selected_cost
        margin = (second - selected_cost) / max(abs(second), scale, EPS)
        median_deformation = float(np.median(deformation_array))
        p90_deformation = float(np.percentile(deformation_array, 90))
        support = len(deformation)
        if (
            support < min_support
            or len(scales) < min(2, len(np.unique(source.edge_scales)))
            or median_deformation > relative_threshold
            or p90_deformation > 2.0 * relative_threshold
        ):
            continue
        accepted.append({
            "frame_a": source.frame, "frame_b": target.frame,
            "source_id": int(i), "target_id": int(j),
            "graph_score": float(scores[selected_id]),
            "edge_deformation_median": median_deformation,
            "edge_deformation_p90": p90_deformation,
            "compatibility_support": int(support),
            "support_scales": int(len(scales)),
            "assignment_margin": float(margin),
            "skip_consistency": "not_available",
            "se3_residual": math.nan,
            "se3_inlier": False,
            "motion_magnitude": float(np.linalg.norm(source.xyz[i] - target.xyz[j])),
            "dynamic_anchor_accepted": False,
            "track_id": -1,
        })
    deformation_values = [row["edge_deformation_median"] for row in accepted]
    supports = [row["compatibility_support"] for row in accepted]
    margins = [row["assignment_margin"] for row in accepted]
    quality = {
        "frame_a": source.frame, "frame_b": target.frame,
        "source_residual_points": len(source.xyz), "target_residual_points": len(target.xyz),
        "candidate_nodes": len(edge_i),
        "candidate_compatibility_edges": int(compatibility.nnz // 2),
        "projected_matches": len(projected), "accepted_matches": len(accepted),
        "unmatched_source": len(source.xyz) - len(accepted),
        "unmatched_target": len(target.xyz) - len({row['target_id'] for row in accepted}),
        "median_edge_deformation": float(np.median(deformation_values)) if deformation_values else math.nan,
        "p90_edge_deformation": float(np.percentile(deformation_values, 90)) if deformation_values else math.nan,
        "median_graph_support": float(np.median(supports)) if supports else 0.0,
        "median_assignment_margin": float(np.median(margins)) if margins else math.nan,
        "skip_confirmed": 0, "skip_conflict": 0, "new_dynamic_anchors": 0,
    }
    return accepted, quality


def annotate_skip_consistency(
    pair_rows: dict[tuple[int, int], list[dict]],
    frames: list[int],
    maximum_skip: int,
) -> None:
    """Compare adjacent compositions with direct t->t+2/t+3 graph matches."""
    for rows in pair_rows.values():
        for row in rows:
            row["skip_consistency"] = "not_available"
    for rank in range(len(frames) - 1):
        pair = (frames[rank], frames[rank + 1])
        rows = pair_rows.get(pair, [])
        by_source = {row["source_id"]: row for row in rows}
        for source_id, row in by_source.items():
            current = row["target_id"]
            evidence = []
            for delta in range(2, min(maximum_skip + 2, len(frames) - rank)):
                composed = current
                available = True
                for step in range(1, delta):
                    step_rows = pair_rows.get((frames[rank + step], frames[rank + step + 1]), [])
                    step_map = {item["source_id"]: item["target_id"] for item in step_rows}
                    if composed not in step_map:
                        available = False
                        break
                    composed = step_map[composed]
                direct_rows = pair_rows.get((frames[rank], frames[rank + delta]), [])
                direct_map = {item["source_id"]: item["target_id"] for item in direct_rows}
                if available and source_id in direct_map:
                    evidence.append(direct_map[source_id] == composed)
            if any(evidence):
                row["skip_consistency"] = "confirmed"
            elif evidence:
                row["skip_consistency"] = "conflict"


def build_residual_tracks(
    residuals: dict[int, ResidualCloud],
    frames: list[int],
    pair_rows: dict[tuple[int, int], list[dict]],
) -> dict[int, Track]:
    """Union accepted partial matches without allowing two nodes from one frame."""
    nodes = [(frame, point_id) for frame in frames for point_id in range(len(residuals[frame].xyz))]
    index = {node: node_id for node_id, node in enumerate(nodes)}
    parent = np.arange(len(nodes), dtype=np.int64)
    component_frames = [{frame} for frame, _point_id in nodes]

    def find(value: int) -> int:
        while parent[value] != value:
            parent[value] = parent[parent[value]]
            value = int(parent[value])
        return value

    def union(left: int, right: int) -> bool:
        left_root, right_root = find(left), find(right)
        if left_root == right_root:
            return True
        if component_frames[left_root] & component_frames[right_root]:
            return False
        if len(component_frames[left_root]) < len(component_frames[right_root]):
            left_root, right_root = right_root, left_root
        parent[right_root] = left_root
        component_frames[left_root].update(component_frames[right_root])
        component_frames[right_root].clear()
        return True

    for pair in sorted(pair_rows, key=lambda value: (value[1] - value[0], value)):
        frame_a, frame_b = pair
        for row in pair_rows[pair]:
            union(index[(frame_a, row["source_id"])], index[(frame_b, row["target_id"])])

    grouped: dict[int, list[tuple[int, int]]] = {}
    for node_id, node in enumerate(nodes):
        grouped.setdefault(find(node_id), []).append(node)
    tracks: dict[int, Track] = {}
    node_owner: dict[tuple[int, int], int] = {}
    for track_id, component in enumerate(sorted(grouped.values(), key=lambda values: values[0])):
        track = Track(track_id)
        for frame, point_id in component:
            track.points[frame] = residuals[frame].xyz[point_id]
            track.point_ids[frame] = point_id
            node_owner[(frame, point_id)] = track_id
        tracks[track_id] = track
    for rows in pair_rows.values():
        for row in rows:
            track_id = node_owner[(row["frame_a"], row["source_id"])]
            if node_owner[(row["frame_b"], row["target_id"])] != track_id:
                continue
            row["track_id"] = track_id
            track = tracks[track_id]
            track.temporal_matches += 1
            track.match_confidences.append(float(row["graph_score"]))
    return tracks


def validate_se3_models(
    rows: list[dict],
    source: ResidualCloud,
    target: ResidualCloud,
    threshold: float,
    seed: int,
) -> None:
    """Fit several generic rigid consensus models; this validates, never proposes matches."""
    if len(rows) < 3:
        return
    source_xyz = np.stack([source.xyz[row["source_id"]] for row in rows])
    target_xyz = np.stack([target.xyz[row["target_id"]] for row in rows])
    remaining = np.arange(len(rows), dtype=np.int32)
    best_residual = np.full(len(rows), np.inf)
    rng = np.random.default_rng(seed)
    for _model in range(3):
        if len(remaining) < 3:
            break
        best_inliers = np.empty(0, np.int32)
        for _ in range(min(96, max(24, len(remaining)))):
            sample = rng.choice(remaining, size=3, replace=False)
            try:
                rotation, translation, _ = fit_rigid_transform(source_xyz[sample], target_xyz[sample])
            except np.linalg.LinAlgError:
                continue
            residual = np.linalg.norm(source_xyz @ rotation.T + translation - target_xyz, axis=1)
            inliers = remaining[residual[remaining] <= threshold]
            if len(inliers) > len(best_inliers):
                best_inliers = inliers
        if len(best_inliers) < 3:
            break
        rotation, translation, _ = fit_rigid_transform(
            source_xyz[best_inliers], target_xyz[best_inliers],
        )
        residual = np.linalg.norm(source_xyz @ rotation.T + translation - target_xyz, axis=1)
        best_residual = np.minimum(best_residual, residual)
        remaining = remaining[residual[remaining] > threshold]
    for row, residual in zip(rows, best_residual):
        row["se3_residual"] = float(residual)
        row["se3_inlier"] = bool(residual <= threshold)


def estimate_static_motion_threshold(
    tracks: dict[int, Track],
    clouds: dict[int, FrameCloud],
    frames: list[int],
) -> tuple[float, dict]:
    rank = {frame: index for index, frame in enumerate(frames)}
    displacements, measurement_uncertainty = [], []
    for track in tracks.values():
        if not trusted_static_track(track):
            continue
        ordered = sorted(track.points, key=rank.get)
        for left, right in zip(ordered[:-1], ordered[1:]):
            if rank[right] == rank[left] + 1:
                displacements.append(float(np.linalg.norm(track.points[right] - track.points[left])))
        for frame, point_id in track.point_ids.items():
            cloud = clouds[frame]
            measurement_uncertainty.append(float(
                cloud.spatial_spread[point_id] / math.sqrt(max(int(cloud.num_raw_members[point_id]), 1))
            ))
    displacement_array = np.asarray(displacements, np.float64)
    uncertainty_array = np.asarray(measurement_uncertainty, np.float64)
    temporal_noise = (
        float(np.percentile(displacement_array, 99)) + 2.0 * robust_mad(displacement_array)
        if len(displacement_array) else 0.0
    )
    measurement_noise = 3.0 * float(np.median(uncertainty_array)) if len(uncertainty_array) else 0.0
    threshold = max(temporal_noise, measurement_noise, EPS)
    return threshold, {
        "threshold": threshold,
        "trusted_static_transitions": int(len(displacement_array)),
        "temporal_noise_p99_plus_2mad": temporal_noise,
        "measurement_uncertainty_3x_median": measurement_noise,
        "source": "trusted STATIC observations only",
    }


def validate_dynamic_anchors(
    residual_tracks: dict[int, Track],
    pair_rows: dict[tuple[int, int], list[dict]],
    motion_threshold: float,
    min_support: int,
    relative_threshold: float,
) -> set[int]:
    rows_by_track: dict[int, list[dict]] = {}
    for rows in pair_rows.values():
        for row in rows:
            if row["track_id"] >= 0:
                rows_by_track.setdefault(row["track_id"], []).append(row)
    dynamic: set[int] = set()
    for track_id, track in residual_tracks.items():
        rows = rows_by_track.get(track_id, [])
        if len(track.points) < 3 or len(rows) < 2:
            continue
        deformations = np.asarray([row["edge_deformation_median"] for row in rows])
        p90 = np.asarray([row["edge_deformation_p90"] for row in rows])
        supports = np.asarray([row["compatibility_support"] for row in rows])
        motions = np.asarray([row["motion_magnitude"] for row in rows])
        skip_conflicts = sum(row["skip_consistency"] == "conflict" for row in rows)
        skip_confirmed = sum(row["skip_consistency"] == "confirmed" for row in rows)
        skip_observed = skip_conflicts + skip_confirmed
        se3_fraction = float(np.mean([row["se3_inlier"] for row in rows]))
        accepted = (
            float(np.median(supports)) >= min_support
            and float(np.median(deformations)) <= relative_threshold
            and float(np.percentile(p90, 90)) <= 2.0 * relative_threshold
            and se3_fraction >= 0.6
            and (skip_observed == 0 or skip_conflicts == 0)
            and float(np.median(motions)) > motion_threshold
        )
        if accepted:
            dynamic.add(track_id)
            track.dynamic_anchor = True
            track.dynamic_probability = 1.0
            track.seed_type = "DYNAMIC_GRAPH_SE3"
            track.seed_strength = min(1.0, 0.5 * se3_fraction + 0.5 * min(float(np.median(supports)) / 8.0, 1.0))
            for row in rows:
                row["dynamic_anchor_accepted"] = True
    return dynamic


def run_residual_graph_matching(
    residuals: dict[int, ResidualCloud],
    graphs: dict[int, ResidualGraph],
    frames: list[int],
    args: argparse.Namespace,
    node_probabilities: dict[int, np.ndarray] | None,
) -> tuple[dict[int, Track], dict[tuple[int, int], list[dict]], list[dict]]:
    pair_rows: dict[tuple[int, int], list[dict]] = {}
    quality_rows: list[dict] = []
    maximum_delta = min(args.residual_max_skip + 1, len(frames) - 1)
    total_pairs = sum(len(frames) - delta for delta in range(1, maximum_delta + 1))
    completed_pairs = 0
    started = time.perf_counter()
    for delta in range(1, maximum_delta + 1):
        for rank in range(len(frames) - delta):
            frame_a, frame_b = frames[rank], frames[rank + delta]
            rows, quality = match_residual_graph_pair(
                graphs[frame_a], graphs[frame_b],
                candidates=args.residual_candidates,
                iterations=args.residual_graph_iters,
                min_support=args.residual_min_support,
                relative_threshold=args.residual_relative_threshold,
                unmatched_scale=args.unmatched_cost_scale,
                source_p_dynamic=None if node_probabilities is None else node_probabilities[frame_a],
                target_p_dynamic=None if node_probabilities is None else node_probabilities[frame_b],
            )
            pair_rows[(frame_a, frame_b)] = rows
            quality_rows.append(quality)
            completed_pairs += 1
            if completed_pairs % 10 == 0 or completed_pairs == total_pairs:
                elapsed = time.perf_counter() - started
                print(
                    f"[RESIDUAL MATCH PROGRESS] pairs={completed_pairs}/{total_pairs} "
                    f"elapsed={elapsed:.1f}s average={elapsed / completed_pairs:.2f}s/pair"
                )
    annotate_skip_consistency(pair_rows, frames, args.residual_max_skip)
    quality_lookup = {(row["frame_a"], row["frame_b"]): row for row in quality_rows}
    for pair, rows in pair_rows.items():
        quality_lookup[pair]["skip_confirmed"] = sum(row["skip_consistency"] == "confirmed" for row in rows)
        quality_lookup[pair]["skip_conflict"] = sum(row["skip_consistency"] == "conflict" for row in rows)
    tracks = build_residual_tracks(residuals, frames, pair_rows)
    return tracks, pair_rows, quality_rows


def temporal_relative_change(track_a: Track, track_b: Track, spacing: float) -> tuple[float, int] | None:
    common = sorted(set(track_a.points) & set(track_b.points))
    if len(common) < 2:
        return None
    values = []
    for left, right in zip(common[:-1], common[1:]):
        distance_left = float(np.linalg.norm(track_a.points[left] - track_b.points[left]))
        distance_right = float(np.linalg.norm(track_a.points[right] - track_b.points[right]))
        values.append(abs(distance_right - distance_left) / max(
            0.5 * (distance_left + distance_right), spacing, EPS,
        ))
    if not values:
        return None
    array = np.asarray(values, np.float64)
    return float(np.median(array) + robust_mad(array)), len(common)


def build_binary_rigidity_relations(
    residual_tracks: dict[int, Track],
    static_tracks: dict[int, Track],
    frames: list[int],
    spacing: float,
    neighbors: int = 12,
) -> tuple[dict[int, Track], list[dict], dict]:
    """Build binary pair targets only from temporal relative-distance evolution."""
    nodes: dict[int, Track] = dict(residual_tracks)
    static_ids: dict[int, int] = {}
    for track_id, track in static_tracks.items():
        if trusted_static_track(track):
            node_id = -(track_id + 1)
            nodes[node_id] = track
            static_ids[track_id] = node_id

    candidate_pairs: set[tuple[int, int]] = set()
    for frame in frames:
        residual_observations = [
            (track_id, track.points[frame])
            for track_id, track in residual_tracks.items() if frame in track.points
        ]
        if len(residual_observations) > 1:
            ids = np.asarray([value[0] for value in residual_observations], np.int64)
            xyz = np.stack([value[1] for value in residual_observations])
            count = min(neighbors + 1, len(ids))
            local = cKDTree(xyz).query(xyz, k=count)[1]
            if count == 1:
                local = local[:, None]
            for row, local_ids in enumerate(local):
                for column in np.atleast_1d(local_ids)[1:]:
                    left, right = sorted((int(ids[row]), int(ids[int(column)])))
                    candidate_pairs.add((left, right))
        static_observations = [
            (static_ids[track_id], track.points[frame])
            for track_id, track in static_tracks.items()
            if track_id in static_ids and frame in track.points
        ]
        if residual_observations and static_observations:
            static_node_ids = np.asarray([value[0] for value in static_observations], np.int64)
            static_xyz = np.stack([value[1] for value in static_observations])
            count = min(neighbors, len(static_xyz))
            nearest = cKDTree(static_xyz).query(
                np.stack([value[1] for value in residual_observations]), k=count,
            )[1]
            if count == 1:
                nearest = nearest[:, None]
            for (residual_id, _point), local in zip(residual_observations, nearest):
                for column in np.atleast_1d(local):
                    candidate_pairs.add((int(static_node_ids[int(column)]), int(residual_id)))
        if len(static_observations) > 1:
            static_node_ids = np.asarray([value[0] for value in static_observations], np.int64)
            static_xyz = np.stack([value[1] for value in static_observations])
            count = min(5, len(static_xyz))
            local = cKDTree(static_xyz).query(static_xyz, k=count)[1]
            for row, local_ids in enumerate(np.asarray(local)):
                for column in np.atleast_1d(local_ids)[1:]:
                    candidate_pairs.add(tuple(sorted((
                        int(static_node_ids[row]), int(static_node_ids[int(column)]),
                    ))))

    raw_relations = []
    for track_a, track_b in candidate_pairs:
        relation = temporal_relative_change(nodes[track_a], nodes[track_b], spacing)
        if relation is None:
            continue
        rigidity, common_frames = relation
        confidence_a = float(np.mean(nodes[track_a].match_confidences)) if nodes[track_a].match_confidences else 1.0
        confidence_b = float(np.mean(nodes[track_b].match_confidences)) if nodes[track_b].match_confidences else 1.0
        raw_relations.append({
            "track_a": track_a, "track_b": track_b,
            "common_frames": common_frames,
            "rigidity_error": rigidity,
            "temporal_match_confidence": math.sqrt(confidence_a * confidence_b),
            "both_trusted_static": track_a < 0 and track_b < 0,
        })
    static_values = np.asarray([
        row["rigidity_error"] for row in raw_relations if row["both_trusted_static"]
    ], np.float64)
    all_values = np.asarray([row["rigidity_error"] for row in raw_relations], np.float64)
    if len(static_values):
        positive_threshold = max(
            float(np.percentile(static_values, 95)),
            float(np.median(static_values)) + 3.0 * robust_mad(static_values),
            EPS,
        )
    elif len(all_values):
        positive_threshold = max(float(np.percentile(all_values, 15)), EPS)
    else:
        positive_threshold = EPS
    high = all_values[all_values > positive_threshold + max(robust_mad(all_values), EPS)]
    negative_threshold = (
        max(2.0 * positive_threshold, float(np.percentile(high, 25)))
        if len(high) else math.inf
    )
    finite_high = all_values[all_values >= negative_threshold] if math.isfinite(negative_threshold) else np.empty(0)
    high_scale = max(
        float(np.percentile(finite_high, 90) - negative_threshold) if len(finite_high) else 0.0,
        positive_threshold,
        EPS,
    )
    edges, ignored = [], 0
    for row in raw_relations:
        rigidity = row["rigidity_error"]
        if rigidity <= positive_threshold:
            q_same = 1.0
            margin_weight = 0.25 + 0.75 * (positive_threshold - rigidity) / positive_threshold
            evidence = "positive_same_label"
        elif rigidity >= negative_threshold:
            q_same = 0.0
            margin_weight = 0.25 + 0.75 * min((rigidity - negative_threshold) / high_scale, 1.0)
            evidence = "negative_different_label"
        else:
            ignored += 1
            continue
        weight = (
            math.log1p(row["common_frames"])
            * row["temporal_match_confidence"]
            * margin_weight
        )
        edges.append({
            **row,
            "q_same": q_same,
            "weight": float(weight),
            "evidence": evidence,
        })
    return nodes, edges, {
        "candidate_pairs": len(candidate_pairs),
        "measured_relations": len(raw_relations),
        "positive_threshold": positive_threshold,
        "negative_threshold": negative_threshold,
        "positive_edges": sum(row["q_same"] == 1.0 for row in edges),
        "negative_edges": sum(row["q_same"] == 0.0 for row in edges),
        "ignored_margin_edges": ignored,
        "trusted_static_reference_nodes": len(static_ids),
        "pair_target_source": "temporal relative-distance evolution only",
    }


def optimize_binary_dynamic_probabilities(
    nodes: dict[int, Track],
    edges: list[dict],
    dynamic_anchor_ids: set[int],
    outer_iteration: int,
    iterations: int,
    learning_rate: float,
    static_threshold: float,
    dynamic_threshold: float,
    ignored_margin_edges: int,
) -> tuple[dict[int, float], list[dict]]:
    """Optimize scalar p_dynamic; anchor values are exactly clamped."""
    ordered = sorted(nodes)
    lookup = {track_id: index for index, track_id in enumerate(ordered)}
    initial = np.asarray([
        0.0 if track_id < 0 else 1.0 if track_id in dynamic_anchor_ids
        else float(np.clip(nodes[track_id].dynamic_probability, 1e-4, 1.0 - 1e-4))
        for track_id in ordered
    ], np.float64)
    logits = torch.nn.Parameter(torch.as_tensor(
        np.log(np.maximum(initial, 1e-4) / np.maximum(1.0 - initial, 1e-4))
        + np.asarray([0.01 * math.sin(track_id + 1) for track_id in ordered]),
        dtype=torch.float64,
    ))
    static_mask = torch.as_tensor([track_id < 0 for track_id in ordered], dtype=torch.bool)
    dynamic_mask = torch.as_tensor([track_id in dynamic_anchor_ids for track_id in ordered], dtype=torch.bool)
    trainable_mask = ~(static_mask | dynamic_mask)
    if edges:
        left = torch.as_tensor([lookup[row["track_a"]] for row in edges], dtype=torch.long)
        right = torch.as_tensor([lookup[row["track_b"]] for row in edges], dtype=torch.long)
        target = torch.as_tensor([row["q_same"] for row in edges], dtype=torch.float64)
        weight = torch.as_tensor([row["weight"] for row in edges], dtype=torch.float64)
    else:
        left = right = torch.empty(0, dtype=torch.long)
        target = weight = torch.empty(0, dtype=torch.float64)
    optimizer = torch.optim.Adam([logits], lr=learning_rate)
    history: list[dict] = []
    for inner_iteration in range(iterations):
        optimizer.zero_grad()
        free_probability = torch.sigmoid(logits)
        probability = torch.where(
            static_mask, torch.zeros_like(free_probability),
            torch.where(dynamic_mask, torch.ones_like(free_probability), free_probability),
        )
        if len(edges):
            same = probability[left] * probability[right] + (
                1.0 - probability[left]
            ) * (1.0 - probability[right])
            binary = -(target * torch.log(same.clamp_min(EPS)) +
                       (1.0 - target) * torch.log((1.0 - same).clamp_min(EPS)))
            positive = target > 0.5
            negative = ~positive
            positive_loss = (
                (weight[positive] * binary[positive]).sum() / weight[positive].sum().clamp_min(EPS)
                if bool(positive.any()) else logits.sum() * 0.0
            )
            negative_loss = (
                (weight[negative] * binary[negative]).sum() / weight[negative].sum().clamp_min(EPS)
                if bool(negative.any()) else logits.sum() * 0.0
            )
            loss = positive_loss + negative_loss
        else:
            same = torch.empty(0, dtype=torch.float64)
            positive = negative = torch.empty(0, dtype=torch.bool)
            positive_loss = negative_loss = loss = logits.sum() * 0.0
        if bool(trainable_mask.any()) and loss.requires_grad:
            loss.backward()
            optimizer.step()
        detached = probability.detach()
        mean_positive = float(same[positive].mean().detach()) if bool(positive.any()) else math.nan
        mean_negative = float(same[negative].mean().detach()) if bool(negative.any()) else math.nan
        history.append({
            "outer_iteration": outer_iteration,
            "inner_iteration": inner_iteration,
            "loss": float(loss.detach()),
            "positive_pair_loss": float(positive_loss.detach()),
            "negative_pair_loss": float(negative_loss.detach()),
            "static_anchor_loss": 0.0,
            "dynamic_anchor_loss": 0.0,
            "mean_positive_same_probability": mean_positive,
            "mean_negative_same_probability": mean_negative,
            "num_positive_edges": int(positive.sum()),
            "num_negative_edges": int(negative.sum()),
            "num_ignored_margin_edges": int(ignored_margin_edges),
            "mean_p_dynamic": float(detached.mean()) if len(detached) else math.nan,
            "num_static": int((detached <= static_threshold).sum()),
            "num_dynamic": int((detached >= dynamic_threshold).sum()),
            "num_uncertain": int(((detached > static_threshold) & (detached < dynamic_threshold)).sum()),
        })
    with torch.no_grad():
        probability = torch.where(
            static_mask, torch.zeros_like(logits),
            torch.where(dynamic_mask, torch.ones_like(logits), torch.sigmoid(logits)),
        ).cpu().numpy()
    return {track_id: float(probability[index]) for index, track_id in enumerate(ordered)}, history


def probabilities_to_residual_nodes(
    residuals: dict[int, ResidualCloud],
    tracks: dict[int, Track],
    probabilities: dict[int, float],
) -> dict[int, np.ndarray]:
    result = {frame: np.full(len(cloud.xyz), 0.5, np.float64) for frame, cloud in residuals.items()}
    for track_id, track in tracks.items():
        probability = probabilities.get(track_id, 0.5)
        track.dynamic_probability = probability
        for frame, point_id in track.point_ids.items():
            result[frame][point_id] = probability
    return result


def transfer_residual_probabilities_to_coarse_tracks(
    coarse_tracks: dict[int, Track],
    residuals: dict[int, ResidualCloud],
    node_probabilities: dict[int, np.ndarray],
    static_threshold: float,
    dynamic_threshold: float,
) -> dict[int, str]:
    owner = {
        (frame, point_id): track_id
        for track_id, track in coarse_tracks.items()
        for frame, point_id in track.point_ids.items()
    }
    evidence: dict[int, list[tuple[float, int]]] = {}
    for frame, residual in residuals.items():
        for point_id, coarse_point_id in enumerate(residual.coarse_point_ids):
            track_id = owner.get((frame, int(coarse_point_id)))
            if track_id is not None:
                evidence.setdefault(track_id, []).append((
                    float(node_probabilities[frame][point_id]),
                    int(residual.num_raw_members[point_id]),
                ))
    for track_id, track in coarse_tracks.items():
        if trusted_static_track(track):
            probability = 0.0
            source = "frozen_trusted_static"
        elif track_id in evidence:
            values = np.asarray([value for value, _weight in evidence[track_id]])
            weights = np.asarray([weight for _value, weight in evidence[track_id]], np.float64)
            if np.any(values >= 1.0 - EPS):
                # A trusted dynamic graph+SE(3) anchor is clamped in the
                # optimizer and must survive the fine-to-coarse handoff.
                probability = 1.0
                source = "trusted_dynamic_graph_se3_anchor"
            else:
                probability = float(np.average(values, weights=weights))
                source = "alternating_residual_graph_binary_optimization"
        else:
            probability = 0.5
            source = "insufficient_temporal_evidence"
        track.dynamic_probability = probability
        track.probabilities = np.asarray([1.0 - probability, probability])
        track.cluster_id = int(probability >= 0.5)
        track.confidence = max(probability, 1.0 - probability)
        if probability <= static_threshold:
            track.label = "STATIC"
        elif probability >= dynamic_threshold:
            track.label = "MOVING_1"
        else:
            track.label = "AMBIGUOUS"
        track.initial_cluster_id = track.cluster_id
        track.initial_confidence = track.confidence
        track.initial_label = track.label
        track.label_source = source
    return {0: "STATIC", 1: "MOVING_1"}


def apply_static_anchor_only_labels(tracks: dict[int, Track]) -> None:
    """Trusted static/ghost tracks=0; every other track remains uncertain=0.5."""
    for track in tracks.values():
        is_static = trusted_static_track(track)
        track.dynamic_probability = 0.0 if is_static else 0.5
        track.probabilities = np.asarray(
            [1.0, 0.0] if is_static else [0.5, 0.5],
            np.float64,
        )
        track.cluster_id = 0 if is_static else -1
        track.confidence = 1.0 if is_static else 0.5
        track.label = "STATIC" if is_static else "AMBIGUOUS"
        track.initial_cluster_id = track.cluster_id
        track.initial_confidence = track.confidence
        track.initial_label = track.label
        track.label_source = (
            "temporal_static_reappearance"
            if track.seed_type == "STATIC_TEMPORAL_REAPPEARANCE"
            else "recovered_static_fragment" if track.seed_type == "STATIC_RECOVERED"
            else "frozen_trusted_static" if is_static
            else "static_ghost_unresolved"
        )


def sample_keyframes(frames: list[int], count: int) -> list[int]:
    if count < 2:
        raise ValueError("--keyframe-count must be at least 2")
    if count > len(frames):
        raise ValueError("--keyframe-count cannot exceed the selected frame count")
    indices = np.rint(np.linspace(0, len(frames) - 1, count)).astype(np.int32)
    chosen: list[int] = []
    used: set[int] = set()
    for target in indices:
        for offset in range(len(frames)):
            for candidate in (int(target) - offset, int(target) + offset):
                if 0 <= candidate < len(frames) and candidate not in used:
                    used.add(candidate)
                    chosen.append(frames[candidate])
                    break
            else:
                continue
            break
    return sorted(chosen)


def undirected_knn_edges(neighbors: np.ndarray) -> np.ndarray:
    pairs = {
        tuple(sorted((source, int(neighbor))))
        for source, local in enumerate(neighbors)
        for neighbor in local
        if 0 <= int(neighbor) < len(neighbors) and source != int(neighbor)
    }
    if not pairs:
        return np.empty((0, 2), np.int32)
    return np.asarray(sorted(pairs), np.int32)


def owner_by_point(tracks: dict[int, Track], frame: int) -> dict[int, int]:
    result = {}
    for track_id, track in tracks.items():
        point_id = track.point_ids.get(frame)
        if point_id is not None:
            result[int(point_id)] = track_id
    return result


def frame_static_priors(
    tracks: dict[int, Track],
    frame: int,
    point_count: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    prior = np.full(point_count, 0.5, np.float32)
    weight = np.zeros(point_count, np.float32)
    hard_static = np.zeros(point_count, bool)
    for track in tracks.values():
        point_id = track.point_ids.get(frame)
        if point_id is None or point_id < 0 or point_id >= point_count:
            continue
        if trusted_static_track(track):
            prior[point_id] = 0.0
            weight[point_id] = max(float(track.seed_strength), 0.75)
            hard_static[point_id] = True
    return prior, weight, hard_static


def mapping_arrays(mapping: dict[int, int]) -> tuple[np.ndarray, np.ndarray]:
    if not mapping:
        return np.empty(0, np.int32), np.empty(0, np.int32)
    source = np.asarray(sorted(mapping), np.int32)
    target = np.asarray([mapping[int(index)] for index in source], np.int32)
    return source, target


def robust_keyframe_se3(
    source: np.ndarray,
    target: np.ndarray,
    mapping: dict[int, int],
    p_dynamic: np.ndarray,
    static_anchor: np.ndarray,
    spacing: float,
) -> tuple[np.ndarray, np.ndarray, dict]:
    source_ids, target_ids = mapping_arrays(mapping)
    if len(source_ids) < 3:
        return np.eye(3), np.zeros(3), {"num_correspondences": 0, "median_residual": math.nan}
    displacement = np.linalg.norm(target[target_ids] - source[source_ids], axis=1)
    moving = (
        (~static_anchor[source_ids]) &
        (p_dynamic[source_ids] >= 0.55) &
        (displacement >= 0.25 * spacing)
    )
    if np.count_nonzero(moving) < 3:
        cutoff = max(float(np.percentile(displacement, 70)), 0.25 * spacing)
        moving = (~static_anchor[source_ids]) & (displacement >= cutoff)
    if np.count_nonzero(moving) < 3:
        return np.eye(3), np.zeros(3), {"num_correspondences": 0, "median_residual": math.nan}

    fit_source = source[source_ids[moving]]
    fit_target = target[target_ids[moving]]
    if len(fit_source) > 2048:
        score = p_dynamic[source_ids[moving]] * displacement[moving]
        keep = np.argpartition(score, -2048)[-2048:]
        fit_source, fit_target = fit_source[keep], fit_target[keep]
    rotation, translation, _residual = fit_rigid_transform(fit_source, fit_target)
    residuals = np.linalg.norm(fit_source @ rotation.T + translation - fit_target, axis=1)
    cutoff = float(np.median(residuals)) + 2.5 * max(robust_mad(residuals), EPS)
    inliers = residuals <= cutoff
    if np.count_nonzero(inliers) >= 3:
        rotation, translation, _residual = fit_rigid_transform(
            fit_source[inliers], fit_target[inliers],
        )
        residuals = np.linalg.norm(fit_source[inliers] @ rotation.T + translation - fit_target[inliers], axis=1)
    return rotation, translation, {
        "num_correspondences": int(len(residuals)),
        "median_residual": float(np.median(residuals)) if len(residuals) else math.nan,
    }


def project_keyframe_mapping(
    edge_i: np.ndarray,
    edge_j: np.ndarray,
    counts: np.ndarray,
    base_cost: np.ndarray,
    source_xyz: np.ndarray,
    target_xyz: np.ndarray,
    transformed_source: np.ndarray,
    spacing: float,
    position_weight: float,
    unmatched_scale: float,
) -> tuple[dict[int, int], float, np.ndarray]:
    position_cost = (
        np.linalg.norm(transformed_source[edge_i] - target_xyz[edge_j], axis=1) /
        max(3.0 * spacing, EPS)
    )
    identity_cost = (
        np.linalg.norm(source_xyz[edge_i] - target_xyz[edge_j], axis=1) /
        max(spacing, EPS)
    )
    candidate_cost = base_cost + position_weight * np.minimum(position_cost, identity_cost)
    mapping, unmatched = sparse_one_to_one_projection(
        edge_i, edge_j, candidate_cost, counts,
        len(source_xyz), len(target_xyz), unmatched_scale,
    )
    return mapping, unmatched, candidate_cost


def keyframe_training_tensors(
    mapping: dict[int, int],
    source_edges: np.ndarray,
    source_xyz: np.ndarray,
    target_xyz: np.ndarray,
    spacing: float,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    match_i, match_j = mapping_arrays(mapping)
    destination = np.full(len(source_xyz), -1, np.int32)
    destination[match_i] = match_j
    if len(source_edges):
        left, right = source_edges[:, 0], source_edges[:, 1]
        valid = (destination[left] >= 0) & (destination[right] >= 0)
        valid &= destination[left] != destination[right]
        left, right = left[valid], right[valid]
        target_left, target_right = destination[left], destination[right]
        source_distance = np.linalg.norm(source_xyz[left] - source_xyz[right], axis=1)
        target_distance = np.linalg.norm(target_xyz[target_left] - target_xyz[target_right], axis=1)
        sigma_r = max(0.5 * spacing, EPS)
        rigidity_weight = np.exp(-0.5 * ((source_distance - target_distance) / sigma_r) ** 2)
    else:
        left = right = target_left = target_right = np.empty(0, np.int32)
        source_distance = target_distance = np.empty(0, np.float32)
        rigidity_weight = np.empty(0, np.float32)
    return {
        "match_i": torch.as_tensor(match_i, dtype=torch.long, device=device),
        "match_j": torch.as_tensor(match_j, dtype=torch.long, device=device),
        "edge_i": torch.as_tensor(left, dtype=torch.long, device=device),
        "edge_k": torch.as_tensor(right, dtype=torch.long, device=device),
        "edge_j": torch.as_tensor(target_left, dtype=torch.long, device=device),
        "edge_l": torch.as_tensor(target_right, dtype=torch.long, device=device),
        "source_distance": torch.as_tensor(source_distance, dtype=torch.float32, device=device),
        "target_distance": torch.as_tensor(target_distance, dtype=torch.float32, device=device),
        "rigidity_weight": torch.as_tensor(rigidity_weight, dtype=torch.float32, device=device),
    }


def huber01(values: torch.Tensor) -> torch.Tensor:
    absolute = values.abs()
    return torch.where(absolute < 1.0, 0.5 * absolute.square(), absolute - 0.5)


def weighted_mean(values: torch.Tensor, weights: torch.Tensor | None = None) -> torch.Tensor:
    if values.numel() == 0:
        return torch.zeros((), dtype=torch.float32, device=values.device)
    if weights is None:
        return values.mean()
    return torch.sum(values * weights) / torch.sum(weights).clamp_min(1e-8)


def skew_np(vector: np.ndarray) -> np.ndarray:
    x, y, z = vector
    return np.asarray([
        [0.0, -z, y],
        [z, 0.0, -x],
        [-y, x, 0.0],
    ], np.float64)


def rotation_matrix_to_rotvec_np(rotation: np.ndarray) -> np.ndarray:
    trace = float(np.trace(rotation))
    cosine = np.clip((trace - 1.0) * 0.5, -1.0, 1.0)
    angle = float(math.acos(cosine))
    if angle < 1e-8:
        return np.zeros(3, np.float64)
    sine = max(float(math.sin(angle)), 1e-8)
    return (
        angle / (2.0 * sine) *
        np.asarray([
            rotation[2, 1] - rotation[1, 2],
            rotation[0, 2] - rotation[2, 0],
            rotation[1, 0] - rotation[0, 1],
        ], np.float64)
    )


def se3_log_np(rotation: np.ndarray, translation: np.ndarray) -> np.ndarray:
    omega = rotation_matrix_to_rotvec_np(rotation)
    theta = float(np.linalg.norm(omega))
    if theta < 1e-8:
        v = np.asarray(translation, np.float64)
    else:
        omega_hat = skew_np(omega)
        theta2 = theta * theta
        V = (
            np.eye(3, dtype=np.float64) +
            ((1.0 - math.cos(theta)) / theta2) * omega_hat +
            ((theta - math.sin(theta)) / (theta2 * theta)) * (omega_hat @ omega_hat)
        )
        try:
            v = np.linalg.solve(V, np.asarray(translation, np.float64))
        except np.linalg.LinAlgError:
            v = np.asarray(translation, np.float64)
    return np.concatenate([v, omega]).astype(np.float32)


def fit_weighted_rigid_transform(
    source: np.ndarray,
    target: np.ndarray,
    weights: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    weights = np.asarray(weights, np.float64)
    valid = np.isfinite(weights) & (weights > 1e-6)
    if np.count_nonzero(valid) < 3:
        return np.eye(3, dtype=np.float64), np.zeros(3, np.float64)
    source = np.asarray(source[valid], np.float64)
    target = np.asarray(target[valid], np.float64)
    weights = weights[valid]
    weights = weights / max(float(weights.sum()), EPS)
    source_center = np.sum(source * weights[:, None], axis=0)
    target_center = np.sum(target * weights[:, None], axis=0)
    source_centered = source - source_center
    target_centered = target - target_center
    covariance = (source_centered * weights[:, None]).T @ target_centered
    try:
        u, _s, vt = np.linalg.svd(covariance)
    except np.linalg.LinAlgError:
        return np.eye(3, dtype=np.float64), np.zeros(3, np.float64)
    rotation = vt.T @ u.T
    if np.linalg.det(rotation) < 0:
        vt[-1, :] *= -1.0
        rotation = vt.T @ u.T
    translation = target_center - rotation @ source_center
    return rotation, translation


def local_kabsch_twist_initialization(
    source: np.ndarray,
    target: np.ndarray,
    mapping: dict[int, int],
    neighbors: np.ndarray,
    static_anchor: np.ndarray,
    spacing: float,
) -> np.ndarray:
    destination = np.full(len(source), -1, np.int32)
    match_i, match_j = mapping_arrays(mapping)
    destination[match_i] = match_j
    sigma_r = max(0.5 * spacing, EPS)
    twists = np.zeros((len(source), 6), np.float32)
    for point_id in range(len(source)):
        if static_anchor[point_id]:
            continue
        local_ids: list[int] = []
        if destination[point_id] >= 0:
            local_ids.append(point_id)
        for neighbor in neighbors[point_id]:
            neighbor = int(neighbor)
            if 0 <= neighbor < len(source) and destination[neighbor] >= 0:
                local_ids.append(neighbor)
        if not local_ids:
            continue
        local_ids = list(dict.fromkeys(local_ids))
        target_ids = destination[local_ids]
        if len(local_ids) >= 3:
            weights = np.ones(len(local_ids), np.float64)
            if destination[point_id] >= 0:
                center_source = source[point_id]
                center_target = target[destination[point_id]]
                for local_index, neighbor in enumerate(local_ids):
                    if neighbor == point_id:
                        weights[local_index] = 1.0
                        continue
                    source_distance = float(np.linalg.norm(center_source - source[neighbor]))
                    target_distance = float(np.linalg.norm(center_target - target[destination[neighbor]]))
                    weights[local_index] = math.exp(-0.5 * ((source_distance - target_distance) / sigma_r) ** 2)
            rotation, translation = fit_weighted_rigid_transform(
                source[local_ids], target[target_ids], weights,
            )
        elif destination[point_id] >= 0:
            rotation = np.eye(3, dtype=np.float64)
            translation = target[destination[point_id]].astype(np.float64) - source[point_id].astype(np.float64)
        else:
            rotation = np.eye(3, dtype=np.float64)
            translation = np.zeros(3, np.float64)
        twists[point_id] = se3_log_np(rotation, translation)
    return twists


def skew_torch(vector: torch.Tensor) -> torch.Tensor:
    zero = torch.zeros_like(vector[..., 0])
    x, y, z = vector[..., 0], vector[..., 1], vector[..., 2]
    rows = (
        torch.stack((zero, -z, y), dim=-1),
        torch.stack((z, zero, -x), dim=-1),
        torch.stack((-y, x, zero), dim=-1),
    )
    return torch.stack(rows, dim=-2)


def se3_exp_torch(twist: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    v = twist[:, :3]
    omega = twist[:, 3:]
    theta = torch.linalg.norm(omega, dim=1, keepdim=True)
    theta2 = theta.square()
    theta3 = theta2 * theta
    omega_hat = skew_torch(omega)
    omega_hat2 = omega_hat @ omega_hat
    small = theta < 1e-4
    A = torch.where(
        small,
        1.0 - theta2 / 6.0 + theta2.square() / 120.0,
        torch.sin(theta) / theta.clamp_min(EPS),
    )
    B = torch.where(
        small,
        0.5 - theta2 / 24.0 + theta2.square() / 720.0,
        (1.0 - torch.cos(theta)) / theta2.clamp_min(EPS),
    )
    C = torch.where(
        small,
        (1.0 / 6.0) - theta2 / 120.0 + theta2.square() / 5040.0,
        (theta - torch.sin(theta)) / theta3.clamp_min(EPS),
    )
    eye = torch.eye(3, dtype=twist.dtype, device=twist.device).unsqueeze(0).expand(len(twist), -1, -1)
    rotation = eye + A[:, None] * omega_hat + B[:, None] * omega_hat2
    V = eye + B[:, None] * omega_hat + C[:, None] * omega_hat2
    translation = torch.bmm(V, v[:, :, None]).squeeze(-1)
    return rotation, translation


def apply_local_se3_torch(
    source: torch.Tensor,
    rotation: torch.Tensor,
    translation: torch.Tensor,
) -> torch.Tensor:
    return torch.bmm(rotation, source[:, :, None]).squeeze(-1) + translation


def se3_log_norm_torch(
    rotation_a: torch.Tensor,
    translation_a: torch.Tensor,
    rotation_b: torch.Tensor,
    translation_b: torch.Tensor,
    spacing: float,
) -> torch.Tensor:
    relative_rotation = rotation_a.transpose(-1, -2) @ rotation_b
    relative_translation = torch.bmm(
        rotation_a.transpose(-1, -2),
        (translation_b - translation_a)[:, :, None],
    ).squeeze(-1)
    trace = relative_rotation.diagonal(dim1=-2, dim2=-1).sum(dim=-1)
    cosine = ((trace - 1.0) * 0.5).clamp(-1.0 + 1e-6, 1.0 - 1e-6)
    angle = torch.acos(cosine)
    translation_norm = torch.linalg.norm(relative_translation, dim=1) / max(spacing, EPS)
    return torch.sqrt(angle.square() + translation_norm.square() + EPS)


def se3_log_norm_np(
    rotation_a: np.ndarray,
    translation_a: np.ndarray,
    rotation_b: np.ndarray,
    translation_b: np.ndarray,
    spacing: float,
) -> float:
    relative_rotation = rotation_a.T @ rotation_b
    relative_translation = rotation_a.T @ (translation_b - translation_a)
    angle = float(np.linalg.norm(rotation_matrix_to_rotvec_np(relative_rotation)))
    translation_norm = float(np.linalg.norm(relative_translation) / max(spacing, EPS))
    return float(math.sqrt(angle * angle + translation_norm * translation_norm))


def compose_se3_np(
    first_rotation: np.ndarray,
    first_translation: np.ndarray,
    second_rotation: np.ndarray,
    second_translation: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    return (
        second_rotation @ first_rotation,
        second_rotation @ first_translation + second_translation,
    )


def keyframe_composition_training_tensors(
    frame_a: int,
    frame_b: int,
    mapping: dict[int, int],
    previous_results: dict[tuple[int, int], dict],
    device: torch.device,
) -> dict[str, torch.Tensor]:
    source_ids: list[int] = []
    rotations: list[np.ndarray] = []
    translations: list[np.ndarray] = []
    weights: list[float] = []
    for (left_frame, mid_frame), result_ab in previous_results.items():
        if left_frame != frame_a:
            continue
        result_bc = previous_results.get((mid_frame, frame_b))
        if result_bc is None:
            continue
        for source_id, mid_id in result_ab["mapping"].items():
            source_id = int(source_id)
            mid_id = int(mid_id)
            composed_target = result_bc["mapping"].get(mid_id)
            if composed_target is None or mapping.get(source_id) != composed_target:
                continue
            composed_rotation, composed_translation = compose_se3_np(
                result_ab["local_rotation"][source_id],
                result_ab["local_translation"][source_id],
                result_bc["local_rotation"][mid_id],
                result_bc["local_translation"][mid_id],
            )
            source_ids.append(source_id)
            rotations.append(composed_rotation.astype(np.float32))
            translations.append(composed_translation.astype(np.float32))
            confidence = float(result_ab.get("composition_confidence", np.ones_like(result_ab["p_dynamic"]))[source_id])
            confidence *= float(result_bc.get("composition_confidence", np.ones_like(result_bc["p_dynamic"]))[mid_id])
            weights.append(max(confidence, 0.05))
    if not source_ids:
        return {
            "source": torch.empty(0, dtype=torch.long, device=device),
            "rotation": torch.empty((0, 3, 3), dtype=torch.float32, device=device),
            "translation": torch.empty((0, 3), dtype=torch.float32, device=device),
            "weight": torch.empty(0, dtype=torch.float32, device=device),
        }
    return {
        "source": torch.as_tensor(source_ids, dtype=torch.long, device=device),
        "rotation": torch.as_tensor(np.stack(rotations), dtype=torch.float32, device=device),
        "translation": torch.as_tensor(np.stack(translations), dtype=torch.float32, device=device),
        "weight": torch.as_tensor(weights, dtype=torch.float32, device=device),
    }


def optimize_keyframe_pair(
    cloud_a: FrameCloud,
    cloud_b: FrameCloud,
    tracks: dict[int, Track],
    args: argparse.Namespace,
    pair_index: int,
    previous_results: dict[tuple[int, int], dict] | None = None,
) -> dict:
    previous_results = previous_results or {}
    source_np = np.asarray(cloud_a.xyz, np.float32)
    target_np = np.asarray(cloud_b.xyz, np.float32)
    signature_a = sparse_graph_signatures(source_np, args.num_rel_neighbors)
    signature_b = sparse_graph_signatures(target_np, args.num_rel_neighbors)
    aligned_a, preliminary_anchors, prealignment_stats = preliminary_static_anchor_icp(
        source_np, target_np, signature_a[3], signature_b[3],
    )
    edge_i, edge_j, signature_cost, counts = sparse_signature_candidates(
        signature_a, signature_b, args.hungarian_candidates,
    )
    spacing = max(0.5 * (signature_a[3] + signature_b[3]), EPS)
    compatibility = candidate_compatibility_graph(
        aligned_a, target_np, signature_a[2], edge_i, edge_j, counts,
        max(0.125 * (signature_a[3] + signature_b[3]), EPS),
        args.compatibility_relative_threshold,
    )
    graph_scores, _degree = graph_matching_scores(
        signature_cost, compatibility, args.hungarian_refine_iters,
    )
    base_cost = -np.log(np.maximum(graph_scores, EPS))
    prior_dynamic, prior_weight, static_anchor = frame_static_priors(
        tracks, cloud_a.frame, len(source_np),
    )
    transformed_np = source_np.copy()
    mapping, unmatched, _candidate_cost = project_keyframe_mapping(
        edge_i, edge_j, counts, base_cost, source_np, target_np, transformed_np,
        spacing, args.keyframe_match_position_weight, args.unmatched_cost_scale,
    )
    for source_id, target_id in mapping.items():
        if preliminary_anchors.get(source_id) == target_id:
            static_anchor[source_id] = True
            prior_dynamic[source_id] = 0.0
            prior_weight[source_id] = max(prior_weight[source_id], 1.0)

    match_i, match_j = mapping_arrays(mapping)
    p_initial = np.full(len(source_np), 0.5, np.float32)
    if len(match_i):
        identity_error = np.linalg.norm(target_np[match_j] - source_np[match_i], axis=1)
        p_initial[match_i] = 1.0 / (
            1.0 + np.exp(-(identity_error - 0.25 * spacing) / max(0.25 * spacing, EPS))
        )
    p_initial = prior_weight * prior_dynamic + (1.0 - prior_weight) * p_initial
    p_initial[static_anchor] = 0.01
    p_initial = np.clip(p_initial, 1e-4, 1.0 - 1e-4)

    rotation, translation_se3, se3_stats = robust_keyframe_se3(
        source_np, target_np, mapping, p_initial, static_anchor, spacing,
    )
    dynamic_pred_np = (source_np @ rotation.T + translation_se3).astype(np.float32)
    twist_initial = local_kabsch_twist_initialization(
        source_np, target_np, mapping, signature_a[2], static_anchor, spacing,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    source = torch.as_tensor(source_np, dtype=torch.float32, device=device)
    target = torch.as_tensor(target_np, dtype=torch.float32, device=device)
    prior = torch.as_tensor(prior_dynamic, dtype=torch.float32, device=device)
    prior_w = torch.as_tensor(prior_weight, dtype=torch.float32, device=device)
    anchor_rows = torch.as_tensor(np.nonzero(static_anchor)[0], dtype=torch.long, device=device)
    logits = torch.nn.Parameter(torch.as_tensor(
        np.log(p_initial / (1.0 - p_initial)), dtype=torch.float32, device=device,
    ))
    twist = torch.nn.Parameter(torch.as_tensor(twist_initial, dtype=torch.float32, device=device))
    dynamic_pred = torch.as_tensor(dynamic_pred_np, dtype=torch.float32, device=device)
    source_edges = undirected_knn_edges(signature_a[2])
    tensors = keyframe_training_tensors(mapping, source_edges, source_np, target_np, spacing, device)
    composition_tensors = keyframe_composition_training_tensors(
        cloud_a.frame, cloud_b.frame, mapping, previous_results, device,
    )
    optimizer = torch.optim.Adam([logits, twist], lr=args.keyframe_lr)
    history = []
    last_losses: dict[str, float] = {}

    progress = tqdm(
        range(args.keyframe_iters),
        desc=f"Keyframe joint {cloud_a.frame:06d}->{cloud_b.frame:06d}",
        unit="iter",
        file=sys.stdout,
    )
    for iteration in progress:
        if iteration > 0 and iteration % args.keyframe_match_refresh == 0:
            with torch.no_grad():
                probability_np = torch.sigmoid(logits).detach().cpu().numpy()
                local_rotation, local_translation = se3_exp_torch(twist)
                transformed_np = apply_local_se3_torch(
                    source, local_rotation, local_translation,
                ).detach().cpu().numpy()
            mapping, unmatched, _candidate_cost = project_keyframe_mapping(
                edge_i, edge_j, counts, base_cost, source_np, target_np, transformed_np,
                spacing, args.keyframe_match_position_weight, args.unmatched_cost_scale,
            )
            rotation, translation_se3, se3_stats = robust_keyframe_se3(
                source_np, target_np, mapping, probability_np, static_anchor, spacing,
            )
            dynamic_pred_np = (source_np @ rotation.T + translation_se3).astype(np.float32)
            dynamic_pred = torch.as_tensor(dynamic_pred_np, dtype=torch.float32, device=device)
            tensors = keyframe_training_tensors(mapping, source_edges, source_np, target_np, spacing, device)
            composition_tensors = keyframe_composition_training_tensors(
                cloud_a.frame, cloud_b.frame, mapping, previous_results, device,
            )

        optimizer.zero_grad(set_to_none=True)
        probability = torch.sigmoid(logits)
        local_rotation, local_translation = se3_exp_torch(twist)
        transformed = apply_local_se3_torch(source, local_rotation, local_translation)
        match_i_t, match_j_t = tensors["match_i"], tensors["match_j"]
        if match_i_t.numel() == 0:
            raise RuntimeError(f"Keyframe pair {cloud_a.frame}->{cloud_b.frame} has no temporal matches")

        match_error = torch.linalg.norm(transformed[match_i_t] - target[match_j_t], dim=1) / spacing
        match_loss = huber01(match_error).mean()
        identity_error = torch.linalg.norm(target[match_j_t] - source[match_i_t], dim=1) / spacing
        local_dynamic_error = torch.linalg.norm(target[match_j_t] - transformed[match_i_t], dim=1) / spacing
        model_margin = (identity_error - local_dynamic_error).detach()
        dynamic_target = torch.sigmoid(model_margin / 0.25).detach()
        evidence_weight = torch.clamp(model_margin.abs() / 2.0, 0.0, 1.0).detach()
        p_match = probability[match_i_t].clamp(1e-6, 1.0 - 1e-6)
        evidence_loss = weighted_mean(
            -(dynamic_target * torch.log(p_match) + (1.0 - dynamic_target) * torch.log(1.0 - p_match)),
            evidence_weight,
        )

        edge_i_t, edge_k_t = tensors["edge_i"], tensors["edge_k"]
        if edge_i_t.numel():
            source_distance = tensors["source_distance"]
            target_distance = tensors["target_distance"]
            rigidity_weight = tensors["rigidity_weight"].detach()
            transformed_distance = torch.linalg.norm(
                transformed[edge_i_t] - transformed[edge_k_t], dim=1,
            )
            relative_error = (transformed_distance - target_distance).abs() / spacing
            temporal_same = rigidity_weight
            same_label = (
                probability[edge_i_t] * probability[edge_k_t] +
                (1.0 - probability[edge_i_t]) * (1.0 - probability[edge_k_t])
            ).clamp(1e-6, 1.0 - 1e-6)
            relative_loss = weighted_mean(huber01(relative_error), rigidity_weight)
            label_loss = -(
                temporal_same * torch.log(same_label) +
                (1.0 - temporal_same) * torch.log(1.0 - same_label)
            ).mean()
            transform_smoothness = weighted_mean(
                huber01(se3_log_norm_torch(
                    local_rotation[edge_i_t],
                    local_translation[edge_i_t],
                    local_rotation[edge_k_t],
                    local_translation[edge_k_t],
                    spacing,
                )),
                rigidity_weight,
            )
        else:
            relative_loss = label_loss = transform_smoothness = torch.zeros((), dtype=torch.float32, device=device)

        identity_loss = weighted_mean(
            huber01(torch.sqrt(
                (torch.linalg.norm(twist[:, :3], dim=1) / spacing).square() +
                torch.linalg.norm(twist[:, 3:], dim=1).square() +
                EPS
            )),
            (1.0 - probability).detach(),
        )
        dynamic_loss = weighted_mean(
            huber01(torch.linalg.norm(transformed - dynamic_pred, dim=1) / spacing),
            probability.detach(),
        )
        composition_source = composition_tensors["source"]
        if composition_source.numel():
            composition_error = se3_log_norm_torch(
                local_rotation[composition_source],
                local_translation[composition_source],
                composition_tensors["rotation"],
                composition_tensors["translation"],
                spacing,
            )
            composition_loss = weighted_mean(
                huber01(composition_error),
                composition_tensors["weight"],
            )
        else:
            composition_loss = torch.zeros((), dtype=torch.float32, device=device)
        prior_probability = probability.clamp(1e-6, 1.0 - 1e-6)
        prior_loss = weighted_mean(
            -(prior * torch.log(prior_probability) + (1.0 - prior) * torch.log(1.0 - prior_probability)),
            prior_w,
        )
        ramp = min(1.0, iteration / max(1, args.keyframe_iters // 2))
        polarization_loss = (probability * (1.0 - probability)).mean() * ramp
        with torch.no_grad():
            binary_target = (probability > 0.5).float()
            binary_weight = torch.zeros_like(probability)
            evidence_target_sum = torch.zeros_like(probability)
            evidence_weight_sum = torch.zeros_like(probability)
            evidence_target_sum.index_add_(0, match_i_t, (dynamic_target >= 0.5).float() * evidence_weight)
            evidence_weight_sum.index_add_(0, match_i_t, evidence_weight)
            has_evidence = evidence_weight_sum > 1e-6
            evidence_target = evidence_target_sum / evidence_weight_sum.clamp_min(1e-6)
            binary_target = torch.where(has_evidence, (evidence_target >= 0.5).float(), binary_target)
            binary_weight = torch.where(has_evidence, evidence_weight_sum.clamp(0.25, 1.0), binary_weight)
            uncertain_focus = 1.0 - (2.0 * (probability - 0.5).abs()).clamp(0.0, 1.0)
            binary_weight = binary_weight * (0.25 + 0.75 * uncertain_focus)
            if anchor_rows.numel():
                binary_weight[anchor_rows] = 0.0
            binary_ramp = torch.as_tensor(
                min(1.0, max(0.0, (iteration - 0.6 * args.keyframe_iters) / max(1.0, 0.4 * args.keyframe_iters))),
                dtype=torch.float32,
                device=device,
            )
        binarization_loss = weighted_mean(
            -(binary_target * torch.log(prior_probability) + (1.0 - binary_target) * torch.log(1.0 - prior_probability)),
            binary_weight,
        ) * binary_ramp
        loss = (
            args.keyframe_match_weight * match_loss +
            args.keyframe_relative_weight * relative_loss +
            args.keyframe_static_identity_weight * identity_loss +
            args.keyframe_dynamic_rigid_weight * dynamic_loss +
            args.keyframe_transform_smoothness_weight * transform_smoothness +
            args.keyframe_label_smoothness_weight * label_loss +
            args.keyframe_static_prior_weight * prior_loss +
            args.keyframe_polarization_weight * polarization_loss +
            args.keyframe_binarization_weight * binarization_loss +
            args.keyframe_composition_weight * composition_loss +
            args.keyframe_evidence_weight * evidence_loss
        )
        loss.backward()
        optimizer.step()
        with torch.no_grad():
            if anchor_rows.numel():
                logits[anchor_rows] = -8.0
                twist[anchor_rows] = 0.0
            p_now = torch.sigmoid(logits)
            num_static = int(torch.count_nonzero(p_now <= args.static_probability_threshold).detach().cpu())
            num_dynamic = int(torch.count_nonzero(p_now >= args.dynamic_probability_threshold).detach().cpu())
            num_uncertain = int(torch.count_nonzero(
                (p_now > args.static_probability_threshold) &
                (p_now < args.dynamic_probability_threshold)
            ).detach().cpu())
            mean_p_dynamic = float(p_now.mean().detach().cpu())
            margin_cpu = model_margin.detach().cpu().numpy()
            margin_finite = margin_cpu[np.isfinite(margin_cpu)]
            if len(margin_finite):
                margin_p10, margin_p50, margin_p90 = np.percentile(margin_finite, [10, 50, 90])
            else:
                margin_p10 = margin_p50 = margin_p90 = math.nan
        progress.set_postfix({
            "loss": f"{float(loss.detach().cpu()):.4f}",
            "match": f"{float(match_loss.detach().cpu()):.3f}",
            "rel": f"{float(relative_loss.detach().cpu()):.3f}",
            "id": f"{float(identity_loss.detach().cpu()):.3f}",
            "se3": f"{float(dynamic_loss.detach().cpu()):.3f}",
            "comp": f"{float(composition_loss.detach().cpu()):.3f}",
            "bin": f"{float(binarization_loss.detach().cpu()):.3f}",
            "m": f"{float(margin_p50):.3f}",
            "p": f"{mean_p_dynamic:.3f}",
            "S/D/U": f"{num_static}/{num_dynamic}/{num_uncertain}",
        })
        if iteration % 25 == 0 or iteration == args.keyframe_iters - 1:
            row = {
                "pair_index": pair_index,
                "frame_a": cloud_a.frame,
                "frame_b": cloud_b.frame,
                "iteration": iteration,
                "loss": float(loss.detach().cpu()),
                "match_loss": float(match_loss.detach().cpu()),
                "relative_graph_loss": float(relative_loss.detach().cpu()),
                "static_identity_loss": float(identity_loss.detach().cpu()),
                "dynamic_se3_loss": float(dynamic_loss.detach().cpu()),
                "composition_loss": float(composition_loss.detach().cpu()),
                "transform_smoothness_loss": float(transform_smoothness.detach().cpu()),
                "label_smoothness_loss": float(label_loss.detach().cpu()),
                "static_prior_loss": float(prior_loss.detach().cpu()),
                "evidence_loss": float(evidence_loss.detach().cpu()),
                "polarization_loss": float(polarization_loss.detach().cpu()),
                "binarization_loss": float(binarization_loss.detach().cpu()),
                "model_margin_p10": float(margin_p10),
                "model_margin_p50": float(margin_p50),
                "model_margin_p90": float(margin_p90),
                "mean_p_dynamic": mean_p_dynamic,
                "num_static": num_static,
                "num_dynamic": num_dynamic,
                "num_uncertain": num_uncertain,
                "matches": int(match_i_t.numel()),
                "knn_edges": int(edge_i_t.numel()),
                "composition_edges": int(composition_source.numel()),
                "dynamic_se3_correspondences": int(se3_stats.get("num_correspondences", 0)),
                "dynamic_se3_median_residual": float(se3_stats.get("median_residual", math.nan)),
            }
            history.append(row)
            last_losses = row

    with torch.no_grad():
        probability_np = torch.sigmoid(logits).detach().cpu().numpy().astype(np.float32)
        local_rotation, local_translation = se3_exp_torch(twist)
        transformed_np = apply_local_se3_torch(
            source, local_rotation, local_translation,
        ).detach().cpu().numpy().astype(np.float32)
        local_rotation_np = local_rotation.detach().cpu().numpy().astype(np.float32)
        local_translation_np = local_translation.detach().cpu().numpy().astype(np.float32)
        twist_np = twist.detach().cpu().numpy().astype(np.float32)
        delta_np = (transformed_np - source_np).astype(np.float32)
    mapping, unmatched, candidate_cost = project_keyframe_mapping(
        edge_i, edge_j, counts, base_cost, source_np, target_np, transformed_np,
        spacing, args.keyframe_match_position_weight, args.unmatched_cost_scale,
    )
    rotation, translation_se3, se3_stats = robust_keyframe_se3(
        source_np, target_np, mapping, probability_np, static_anchor, spacing,
    )
    match_i, match_j = mapping_arrays(mapping)
    rows = []
    destination = dict(mapping)
    e_static_np = np.full(len(source_np), np.nan, np.float32)
    e_dynamic_np = np.full(len(source_np), np.nan, np.float32)
    evidence_margin_np = np.full(len(source_np), np.nan, np.float32)
    if len(match_i):
        e_static_np[match_i] = (
            np.linalg.norm(target_np[match_j] - source_np[match_i], axis=1) /
            max(spacing, EPS)
        )
        e_dynamic_np[match_i] = (
            np.linalg.norm(target_np[match_j] - transformed_np[match_i], axis=1) /
            max(spacing, EPS)
        )
        evidence_margin_np[match_i] = e_static_np[match_i] - e_dynamic_np[match_i]
    for point_id, probability in enumerate(probability_np):
        label = (
            "STATIC" if probability <= args.static_probability_threshold
            else "MOVING_1" if probability >= args.dynamic_probability_threshold
            else "AMBIGUOUS"
        )
        target_id = destination.get(point_id, -1)
        rows.append({
            "pair_index": pair_index,
            "frame_a": cloud_a.frame,
            "frame_b": cloud_b.frame,
            "source_point": point_id,
            "target_point": target_id,
            "p_dynamic": float(probability),
            "label": label,
            "static_prior_weight": float(prior_weight[point_id]),
            "static_anchor": bool(static_anchor[point_id]),
            "dx": float(delta_np[point_id, 0]),
            "dy": float(delta_np[point_id, 1]),
            "dz": float(delta_np[point_id, 2]),
            "twist_vx": float(twist_np[point_id, 0]),
            "twist_vy": float(twist_np[point_id, 1]),
            "twist_vz": float(twist_np[point_id, 2]),
            "twist_wx": float(twist_np[point_id, 3]),
            "twist_wy": float(twist_np[point_id, 4]),
            "twist_wz": float(twist_np[point_id, 5]),
            "E_static": float(e_static_np[point_id]),
            "E_dynamic": float(e_dynamic_np[point_id]),
            "evidence_margin": float(evidence_margin_np[point_id]),
        })
    finite_margin = evidence_margin_np[np.isfinite(evidence_margin_np)]
    finite_static = e_static_np[np.isfinite(e_static_np)]
    finite_dynamic = e_dynamic_np[np.isfinite(e_dynamic_np)]
    stats = {
        "pair_index": pair_index,
        "frame_a": cloud_a.frame,
        "frame_b": cloud_b.frame,
        "source_points": len(source_np),
        "target_points": len(target_np),
        "candidate_edges": len(edge_i),
        "compatibility_edges": compatibility.nnz // 2,
        "final_matches": len(mapping),
        "unmatched_cost": unmatched,
        "spacing": spacing,
        "static_anchors": int(np.count_nonzero(static_anchor)),
        "dynamic_se3_correspondences": int(se3_stats.get("num_correspondences", 0)),
        "dynamic_se3_median_residual": float(se3_stats.get("median_residual", math.nan)),
        "E_static_median": float(np.median(finite_static)) if len(finite_static) else math.nan,
        "E_dynamic_median": float(np.median(finite_dynamic)) if len(finite_dynamic) else math.nan,
        "evidence_margin_p10": float(np.percentile(finite_margin, 10)) if len(finite_margin) else math.nan,
        "evidence_margin_p50": float(np.percentile(finite_margin, 50)) if len(finite_margin) else math.nan,
        "evidence_margin_p90": float(np.percentile(finite_margin, 90)) if len(finite_margin) else math.nan,
        **{f"R{r}{c}": float(rotation[r, c]) for r in range(3) for c in range(3)},
        "tx": float(translation_se3[0]),
        "ty": float(translation_se3[1]),
        "tz": float(translation_se3[2]),
        **prealignment_stats,
        **{f"final_{key}": value for key, value in last_losses.items() if key not in ("pair_index", "frame_a", "frame_b")},
    }
    return {
        "frame_a": cloud_a.frame,
        "frame_b": cloud_b.frame,
        "p_dynamic": probability_np,
        "delta": delta_np,
        "twist": twist_np,
        "local_rotation": local_rotation_np,
        "local_translation": local_translation_np,
        "E_static": e_static_np,
        "E_dynamic": e_dynamic_np,
        "evidence_margin": evidence_margin_np,
        "mapping": mapping,
        "history": history,
        "rows": rows,
        "stats": stats,
        "source_xyz": source_np,
        "target_xyz": target_np,
    }


def apply_keyframe_joint_labels_to_tracks(
    tracks: dict[int, Track],
    results: list[dict],
    static_threshold: float,
    dynamic_threshold: float,
) -> dict:
    evidence: dict[int, list[tuple[float, float, str]]] = {}
    owners = {
        frame: owner_by_point(tracks, frame)
        for result in results
        for frame in (result["frame_a"], result["frame_b"])
    }
    for result in results:
        source_owner = owners[result["frame_a"]]
        target_owner = owners[result["frame_b"]]
        p_dynamic = result["p_dynamic"]
        composition_confidence = result.get("composition_confidence")
        if composition_confidence is None:
            composition_confidence = np.ones_like(p_dynamic, dtype=np.float32)
        for point_id, probability in enumerate(p_dynamic):
            track_id = source_owner.get(point_id)
            if track_id is not None:
                confidence = abs(float(probability) - 0.5) * 2.0
                weight = max(confidence, 0.25) * max(float(composition_confidence[point_id]), 0.05)
                evidence.setdefault(track_id, []).append((float(probability), weight, "keyframe_source"))
        for source_id, target_id in result["mapping"].items():
            track_id = target_owner.get(int(target_id))
            if track_id is not None:
                probability = float(p_dynamic[int(source_id)])
                confidence = abs(probability - 0.5) * 2.0
                weight = max(0.75 * confidence, 0.10) * max(float(composition_confidence[int(source_id)]), 0.05)
                evidence.setdefault(track_id, []).append((probability, weight, "keyframe_target_match"))

    counts = {"STATIC": 0, "MOVING_1": 0, "AMBIGUOUS": 0}
    for track_id, track in tracks.items():
        if track_id in evidence:
            values = np.asarray([value for value, _weight, _source in evidence[track_id]], np.float64)
            weights = np.asarray([weight for _value, weight, _source in evidence[track_id]], np.float64)
            probability = float(np.average(values, weights=weights))
            source = "keyframe_joint_transform_label"
        elif trusted_static_track(track):
            probability = 0.0
            source = "keyframe_static_prior_only"
        else:
            probability = 0.5
            source = "keyframe_unobserved_uncertain"
        track.dynamic_probability = probability
        track.probabilities = np.asarray([1.0 - probability, probability], np.float64)
        track.cluster_id = int(probability >= 0.5)
        track.confidence = max(probability, 1.0 - probability)
        if probability <= static_threshold:
            track.label = "STATIC"
        elif probability >= dynamic_threshold:
            track.label = "MOVING_1"
        else:
            track.label = "AMBIGUOUS"
        track.initial_cluster_id = track.cluster_id
        track.initial_confidence = track.confidence
        track.initial_label = track.label
        track.label_source = source
        counts[track.label] += 1
    return counts


def write_keyframe_evidence_histograms(out: Path, rows: list[dict], bins: int = 40) -> dict:
    metrics = ("E_static", "E_dynamic", "evidence_margin", "p_dynamic")
    groups = {
        "trusted_static_anchor": [row for row in rows if row.get("static_anchor")],
        "pred_static": [row for row in rows if row.get("label") == "STATIC"],
        "pred_dynamic": [row for row in rows if row.get("label") == "MOVING_1"],
        "pred_uncertain": [row for row in rows if row.get("label") == "AMBIGUOUS"],
    }
    histogram_rows: list[dict] = []
    summary_rows: list[dict] = []
    for metric in metrics:
        all_values = np.asarray([
            float(row[metric]) for row in rows
            if metric in row and np.isfinite(float(row[metric]))
        ], np.float64)
        if len(all_values) == 0:
            continue
        if metric == "p_dynamic":
            edges = np.linspace(0.0, 1.0, bins + 1)
        else:
            low, high = np.percentile(all_values, [1, 99])
            if not np.isfinite(low) or not np.isfinite(high) or abs(high - low) < EPS:
                low, high = float(np.min(all_values)), float(np.max(all_values) + 1.0)
            padding = 0.05 * max(high - low, EPS)
            edges = np.linspace(low - padding, high + padding, bins + 1)
        for group_name, group_rows in groups.items():
            values = np.asarray([
                float(row[metric]) for row in group_rows
                if metric in row and np.isfinite(float(row[metric]))
            ], np.float64)
            if len(values) == 0:
                summary_rows.append({
                    "group": group_name,
                    "metric": metric,
                    "count": 0,
                    "mean": math.nan,
                    "p10": math.nan,
                    "p50": math.nan,
                    "p90": math.nan,
                })
                continue
            counts, metric_edges = np.histogram(values, bins=edges)
            for bin_index, count in enumerate(counts):
                histogram_rows.append({
                    "group": group_name,
                    "metric": metric,
                    "bin_left": float(metric_edges[bin_index]),
                    "bin_right": float(metric_edges[bin_index + 1]),
                    "count": int(count),
                })
            p10, p50, p90 = np.percentile(values, [10, 50, 90])
            summary_rows.append({
                "group": group_name,
                "metric": metric,
                "count": int(len(values)),
                "mean": float(np.mean(values)),
                "p10": float(p10),
                "p50": float(p50),
                "p90": float(p90),
            })
    write_csv(out / "keyframe_joint_evidence_histograms.csv", histogram_rows)
    write_csv(out / "keyframe_joint_evidence_summary.csv", summary_rows)
    return {
        "histogram_rows": len(histogram_rows),
        "summary_rows": len(summary_rows),
        "histogram_file": str(out / "keyframe_joint_evidence_histograms.csv"),
        "summary_file": str(out / "keyframe_joint_evidence_summary.csv"),
    }


def save_keyframe_joint_debug(out: Path, results: list[dict]) -> tuple[list[dict], list[dict], list[dict]]:
    history = [row for result in results for row in result["history"]]
    rows = [row for result in results for row in result["rows"]]
    stats = [result["stats"] for result in results]
    write_csv(out / "keyframe_joint_history.csv", history)
    write_csv(out / "keyframe_joint_point_labels.csv", rows)
    write_csv(out / "keyframe_joint_pair_stats.csv", stats)
    write_keyframe_evidence_histograms(out, rows)
    for result in results:
        labels = np.asarray([
            0.0 if row["label"] == "STATIC" else 1.0 if row["label"] == "MOVING_1" else 0.5
            for row in result["rows"]
        ], np.float32)
        composition_confidence = result.get("composition_confidence")
        if composition_confidence is None:
            composition_confidence = np.ones_like(result["p_dynamic"], dtype=np.float32)
        write_ply(
            out / f"keyframe_joint_{result['frame_a']:06d}_to_{result['frame_b']:06d}.ply",
            result["source_xyz"],
            {
                "dynamic": labels,
                "p_dynamic": result["p_dynamic"],
                "E_static": result["E_static"],
                "E_dynamic": result["E_dynamic"],
                "evidence_margin": result["evidence_margin"],
                "composition_confidence": composition_confidence,
                "dx": result["delta"][:, 0],
                "dy": result["delta"][:, 1],
                "dz": result["delta"][:, 2],
                "twist_vx": result["twist"][:, 0],
                "twist_vy": result["twist"][:, 1],
                "twist_vz": result["twist"][:, 2],
                "twist_wx": result["twist"][:, 3],
                "twist_wy": result["twist"][:, 4],
                "twist_wz": result["twist"][:, 5],
            },
        )
    return history, rows, stats


def keyframe_pair_schedule(keyframes: list[int]) -> list[tuple[int, int]]:
    pairs: list[tuple[int, int]] = []
    for index in range(len(keyframes) - 1):
        pairs.append((keyframes[index], keyframes[index + 1]))
    for span in range(2, len(keyframes)):
        for index in range(len(keyframes) - span):
            pairs.append((keyframes[index], keyframes[index + span]))
    return pairs


def keyframe_composition_consistency(
    results: list[dict],
    keyframes: list[int],
    out: Path,
) -> dict:
    for result in results:
        result["composition_confidence"] = np.ones(len(result["p_dynamic"]), np.float32)
    result_by_pair = {
        (int(result["frame_a"]), int(result["frame_b"])): result
        for result in results
    }
    frame_to_rank = {frame: rank for rank, frame in enumerate(keyframes)}
    rows: list[dict] = []
    confidences: dict[tuple[int, int], list[tuple[np.ndarray, np.ndarray]]] = {}
    for frame_a in keyframes:
        for frame_b in keyframes:
            for frame_c in keyframes:
                if not (frame_to_rank[frame_a] < frame_to_rank[frame_b] < frame_to_rank[frame_c]):
                    continue
                result_ab = result_by_pair.get((frame_a, frame_b))
                result_bc = result_by_pair.get((frame_b, frame_c))
                result_ac = result_by_pair.get((frame_a, frame_c))
                if result_ab is None or result_bc is None or result_ac is None:
                    continue
                mapping_ab = result_ab["mapping"]
                mapping_bc = result_bc["mapping"]
                mapping_ac = result_ac["mapping"]
                for source_id, mid_id in mapping_ab.items():
                    source_id = int(source_id)
                    mid_id = int(mid_id)
                    direct_target = mapping_ac.get(source_id)
                    composed_target = mapping_bc.get(mid_id)
                    if direct_target is None or composed_target is None:
                        continue
                    direct_rotation = result_ac["local_rotation"][source_id]
                    direct_translation = result_ac["local_translation"][source_id]
                    composed_rotation, composed_translation = compose_se3_np(
                        result_ab["local_rotation"][source_id],
                        result_ab["local_translation"][source_id],
                        result_bc["local_rotation"][mid_id],
                        result_bc["local_translation"][mid_id],
                    )
                    spacing = max(float(result_ac["stats"].get("spacing", 1.0)), EPS)
                    transform_error = se3_log_norm_np(
                        direct_rotation, direct_translation,
                        composed_rotation, composed_translation,
                        spacing,
                    )
                    target_disagreement = float(direct_target != composed_target)
                    confidence = math.exp(-0.5 * transform_error * transform_error)
                    confidence *= 0.25 if target_disagreement else 1.0
                    rows.append({
                        "frame_a": frame_a,
                        "frame_b": frame_b,
                        "frame_c": frame_c,
                        "source_point": source_id,
                        "mid_point": mid_id,
                        "direct_target_point": int(direct_target),
                        "composed_target_point": int(composed_target),
                        "target_disagreement": int(target_disagreement),
                        "composition_error": transform_error,
                        "composition_confidence": confidence,
                    })
                    for pair, point in (
                        ((frame_a, frame_b), source_id),
                        ((frame_b, frame_c), mid_id),
                        ((frame_a, frame_c), source_id),
                    ):
                        key = pair
                        if key not in confidences:
                            result = result_by_pair[key]
                            confidences[key] = (
                                np.zeros(len(result["p_dynamic"]), np.float64),
                                np.zeros(len(result["p_dynamic"]), np.float64),
                            )
                        sums, counts = confidences[key]
                        sums[point] += confidence
                        counts[point] += 1.0
    for pair, (sums, counts) in confidences.items():
        mask = counts > 0
        result_by_pair[pair]["composition_confidence"][mask] = (sums[mask] / counts[mask]).astype(np.float32)
    for result in results:
        composition_confidence = result["composition_confidence"]
        for row in result["rows"]:
            row["composition_confidence"] = float(composition_confidence[int(row["source_point"])])
        values = composition_confidence[np.isfinite(composition_confidence)]
        result["stats"]["composition_confidence_mean"] = float(np.mean(values)) if len(values) else math.nan
        result["stats"]["composition_confidence_min"] = float(np.min(values)) if len(values) else math.nan
    write_csv(out / "keyframe_joint_composition.csv", rows)
    errors = np.asarray([row["composition_error"] for row in rows], np.float64)
    confidences_values = np.asarray([row["composition_confidence"] for row in rows], np.float64)
    return {
        "triplet_rows": int(len(rows)),
        "composition_error_p50": float(np.percentile(errors, 50)) if len(errors) else math.nan,
        "composition_error_p90": float(np.percentile(errors, 90)) if len(errors) else math.nan,
        "composition_confidence_mean": float(np.mean(confidences_values)) if len(confidences_values) else math.nan,
        "composition_file": str(out / "keyframe_joint_composition.csv"),
    }


def keyframe_joint_loss_recap() -> dict:
    return {
        "optimized_by_adam": {
            "logits": "one scalar per source keyframe point; sigmoid(logit)=dynamic_probability_optimized",
            "twist": "one local SE(3) twist [v, omega] per source keyframe point; T_i=exp(xi_i)",
        },
        "recomputed_not_learned": {
            "temporal_mapping": "sparse relative-distance graph matching projected by Hungarian; refreshed by --keyframe-match-refresh",
            "dynamic_se3": "robust one-part rigid transform fit from current dynamic probabilities",
            "local_kabsch_initialization": "per-point twist initialized by weighted Kabsch over graph-matched kNN neighborhoods",
            "static_anchors": "fixed confident identity correspondences; logits pinned static and twist pinned identity",
            "composition_consistency": "all keyframe triplets are audited; low composition confidence down-weights final track evidence",
        },
        "not_algorithm_inputs": [
            "ground truth labels",
            "RGB/appearance",
            "DINO features",
        ],
        "losses": {
            "match_loss": {
                "weight": "--keyframe-match-weight",
                "updates": ["twist"],
                "meaning": "fits exp(xi_i) transformed source points to current temporal matches",
            },
            "relative_graph_loss": {
                "weight": "--keyframe-relative-weight",
                "updates": ["twist"],
                "meaning": "preserves matched kNN relative distances across the keyframe pair",
            },
            "static_identity_loss": {
                "weight": "--keyframe-static-identity-weight",
                "updates": ["twist"],
                "meaning": "keeps low-dynamic-probability points near identity motion for every optimized keyframe pair",
            },
            "dynamic_se3_loss": {
                "weight": "--keyframe-dynamic-rigid-weight",
                "updates": ["twist"],
                "meaning": "keeps high-dynamic-probability points near the current shared moving-part SE(3)",
            },
            "composition_loss": {
                "weight": "--keyframe-composition-weight",
                "updates": ["twist"],
                "meaning": "for skip keyframe pairs, penalizes ||log(T_direct^-1 T_composed)|| when shorter-pair mappings agree",
            },
            "transform_smoothness_loss": {
                "weight": "--keyframe-transform-smoothness-weight",
                "updates": ["twist"],
                "meaning": "uses rigidity-weighted ||log(T_i^-1 T_j)|| so neighbors share transforms only when relative distances are preserved",
            },
            "label_smoothness_loss": {
                "weight": "--keyframe-label-smoothness-weight",
                "updates": ["logits"],
                "meaning": "encourages neighbours with preserved temporal distance to share static/dynamic labels",
            },
            "static_prior_loss": {
                "weight": "--keyframe-static-prior-weight",
                "updates": ["logits"],
                "meaning": "keeps ghost/static-anchor priors static when such priors exist",
            },
            "evidence_loss": {
                "weight": "--keyframe-evidence-weight",
                "updates": ["logits"],
                "meaning": "directly trains dynamic logits from the model margin E_static - E_dynamic",
            },
            "polarization_loss": {
                "weight": "--keyframe-polarization-weight",
                "updates": ["logits"],
                "meaning": "weak symmetric pressure away from p=0.5",
            },
            "binarization_loss": {
                "weight": "--keyframe-binarization-weight",
                "updates": ["logits"],
                "meaning": "late-annealed confidence-gated hardening toward the better model-evidence side",
            },
        },
        "debug_outputs": {
            "keyframe_joint_evidence_histograms.csv": "histograms of E_static, E_dynamic, E_static-E_dynamic, p_dynamic by trusted-static/static/dynamic/uncertain groups",
            "keyframe_joint_composition.csv": "triplet transform composition errors and confidence",
            "keyframe_joint_*.ply": "per-pair source points with p_dynamic, E_static, E_dynamic, margin, composition confidence, displacement, and twist fields",
        },
    }


def run_keyframe_joint_motion_labeling(
    clouds: dict[int, FrameCloud],
    frames: list[int],
    tracks: dict[int, Track],
    args: argparse.Namespace,
    out: Path,
) -> tuple[list[dict], list[dict], list[dict], list[int], dict, dict]:
    keyframes = sample_keyframes(frames, args.keyframe_count)
    print(f"[KEYFRAME JOINT] selected keyframes={keyframes}")
    results = []
    previous_results: dict[tuple[int, int], dict] = {}
    keyframe_pairs = keyframe_pair_schedule(keyframes)
    for pair_index, (frame_a, frame_b) in enumerate(
        tqdm(keyframe_pairs, desc="Keyframe joint pairs", unit="pair", file=sys.stdout)
    ):
        result = optimize_keyframe_pair(
            clouds[frame_a], clouds[frame_b], tracks, args, pair_index, previous_results,
        )
        results.append(result)
        previous_results[(frame_a, frame_b)] = result
    composition_stats = keyframe_composition_consistency(results, keyframes, out)
    counts = apply_keyframe_joint_labels_to_tracks(
        tracks, results, args.static_probability_threshold, args.dynamic_probability_threshold,
    )
    history, rows, stats = save_keyframe_joint_debug(out, results)
    return history, rows, stats, keyframes, counts, composition_stats


# =============================================================================
# Visualization (RGB is first opened here and never feeds the estimator)
# =============================================================================

def camera_frustum(calibration: Calibration, scale: float) -> list[tuple[np.ndarray, np.ndarray]]:
    center = calibration.center_world
    corners = []
    for u, v in ((0, 0), (calibration.width, 0),
                 (calibration.width, calibration.height), (0, calibration.height)):
        ray = np.asarray([
            (u - calibration.cx) / calibration.fx,
            (v - calibration.cy) / calibration.fy,
            1.0,
        ]) * scale
        corners.append(center + calibration.Rcw.T @ ray)
    segments = [(center, corner) for corner in corners]
    segments += [(corners[index], corners[(index + 1) % 4]) for index in range(4)]
    return segments


def fixed_world_projection(clouds: dict[int, FrameCloud], calibrations: dict[str, Calibration]):
    points = np.concatenate([cloud.xyz for cloud in clouds.values()], axis=0)
    diagonal = max(float(np.linalg.norm(np.ptp(points, axis=0))), 1e-3)
    frustums = [segment for calibration in calibrations.values()
                for segment in camera_frustum(calibration, 0.08 * diagonal)]
    all_points = np.concatenate([
        points,
        np.stack([point for segment in frustums for point in segment]),
        np.zeros((1, 3)),
    ])
    view_z = np.asarray([1.0, -1.0, 0.8])
    view_z /= np.linalg.norm(view_z)
    view_x = np.cross(np.asarray([0.0, 0.0, 1.0]), view_z)
    view_x /= np.linalg.norm(view_x)
    view_y = np.cross(view_z, view_x)
    view = np.stack((view_x, view_y))
    projected = all_points @ view.T
    low, high = projected.min(axis=0), projected.max(axis=0)
    padding = np.maximum(0.08 * (high - low), 1e-3)
    return view, low - padding, high + padding, frustums, diagonal


def world_pixel(point: np.ndarray, projection, size: tuple[int, int]) -> tuple[int, int]:
    width, height = size
    view, low, high, _frustums, _diagonal = projection
    uv = np.asarray(point) @ view.T
    x = int(round(30 + (uv[0] - low[0]) / max(high[0] - low[0], EPS) * (width - 60)))
    y = int(round(30 + (1.0 - (uv[1] - low[1]) / max(high[1] - low[1], EPS)) * (height - 60)))
    return x, y


def render_world(
    points: list[np.ndarray] | np.ndarray,
    colors: list[tuple[int, int, int]],
    lines: list[tuple[np.ndarray, np.ndarray, tuple[int, int, int], int]],
    projection,
    header: str,
    size: tuple[int, int] = (960, 720),
    radii: list[int] | None = None,
) -> np.ndarray:
    width, height = size
    image = np.full((height, width, 3), 248, np.uint8)
    for start, end in projection[3]:
        cv2.line(image, world_pixel(start, projection, size), world_pixel(end, projection, size),
                 (190, 190, 190), 1, cv2.LINE_AA)
    diagonal = projection[4]
    for axis, color in zip(np.eye(3), ((0, 0, 230), (0, 170, 0), (220, 40, 20))):
        cv2.arrowedLine(
            image, world_pixel(np.zeros(3), projection, size),
            world_pixel(axis * 0.1 * diagonal, projection, size), color, 2, cv2.LINE_AA,
            tipLength=0.15,
        )
    for start, end, color, thickness in lines:
        cv2.line(image, world_pixel(start, projection, size), world_pixel(end, projection, size),
                 color, thickness, cv2.LINE_AA)
    radii = radii or [3] * len(points)
    for point, color, radius in zip(points, colors, radii):
        cv2.circle(image, world_pixel(point, projection, size), radius, color, -1, cv2.LINE_AA)
    cv2.rectangle(image, (0, 0), (width, 30), (0, 0, 0), -1)
    cv2.putText(image, header, (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.48,
                (255, 255, 255), 1, cv2.LINE_AA)
    return image


def video_writer(path: Path, fps: float, size: tuple[int, int]) -> cv2.VideoWriter:
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, size)
    if not writer.isOpened():
        raise RuntimeError(f"Could not create debug video {path}")
    return writer


def render_depth_geometry_debug(
    clouds: dict[int, FrameCloud],
    frames: list[int],
    calibrations: dict[str, Calibration],
    debug: Path,
    fps: float,
    projection,
) -> None:
    geometry_dir = debug / "depth_geometry"
    geometry_dir.mkdir(parents=True, exist_ok=True)
    writer = video_writer(debug / "depth_models_3d.mp4", fps, (960, 720))
    representative = set(frames[::max(1, len(frames) // 5)]) | {frames[-1]}
    camera_palette = [(40, 70, 220), (40, 150, 230), (40, 190, 80), (180, 180, 30),
                      (220, 100, 30), (200, 50, 150), (130, 60, 220), (80, 150, 150)]
    for frame in frames:
        cloud = clouds[frame]
        raw_stride = max(1, len(cloud.raw_xyz) // 4000)
        raw_points = cloud.raw_xyz[::raw_stride]
        raw_ids = cloud.raw_camera_ids[::raw_stride]
        raw_colors = [camera_palette[int(index) % len(camera_palette)] for index in raw_ids]
        image = render_world(
            list(raw_points) + list(cloud.xyz),
            raw_colors + [(20, 20, 20)] * len(cloud.xyz),
            [], projection,
            f"Depth frame={frame:06d} raw={len(cloud.raw_xyz)} reduced={len(cloud.xyz)} | fixed world view",
            radii=[1] * len(raw_points) + [3] * len(cloud.xyz),
        )
        writer.write(image)
        if frame in representative:
            folder = geometry_dir / f"frame_{frame:06d}"
            folder.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(folder / "raw_depth_world.png"), render_world(
                raw_points, raw_colors, [], projection,
                f"Raw depth frame={frame:06d} points={len(cloud.raw_xyz)} (display stride={raw_stride})",
                radii=[1] * len(raw_points),
            ))
            cv2.imwrite(str(folder / "reduced_depth_world.png"), render_world(
                cloud.xyz, [(30, 30, 30)] * len(cloud.xyz), [], projection,
                f"Reduced depth frame={frame:06d} Hungarian points={len(cloud.xyz)}",
            ))
    writer.release()


def track_lookup(tracks: dict[int, Track]) -> dict[tuple[int, int], int]:
    return {(frame, point_id): track_id for track_id, track in tracks.items()
            for frame, point_id in track.point_ids.items()}


def render_hungarian_3d(
    clouds: dict[int, FrameCloud],
    frames: list[int],
    tracks: dict[int, Track],
    match_rows: list[dict],
    pair_stats: dict[tuple[int, int], dict],
    debug: Path,
    fps: float,
    projection,
) -> None:
    writer = video_writer(debug / "hungarian_3d.mp4", fps, (960, 720))
    lookup = track_lookup(tracks)
    for frame_a, frame_b in zip(frames[:-1], frames[1:]):
        rows = [row for row in match_rows if row["frame_a"] == frame_a and row["frame_b"] == frame_b]
        accepted = [row for row in rows if row["accepted"]]
        rejected_sources = {row["point_a"] for row in rows if not row["accepted"] and row["point_a"] >= 0}
        points = list(clouds[frame_a].xyz) + list(clouds[frame_b].xyz)
        colors = [((90, 90, 90) if i in rejected_sources else (30, 140, 230))
                  for i in range(len(clouds[frame_a].xyz))]
        colors += [(210, 80, 40)] * len(clouds[frame_b].xyz)
        radii = [2] * len(points)
        lines = [(clouds[frame_a].xyz[row["point_a"]], clouds[frame_b].xyz[row["point_b"]],
                  (30, 180, 40), 1) for row in accepted]
        relative = [row["relative_cost"] for row in accepted]
        support = [row["relational_support"] for row in accepted]
        stats = pair_stats[(frame_a, frame_b)]
        image = render_world(
            points, colors, lines, projection,
            f"Hungarian {frame_a:06d}->{frame_b:06d} accepted={len(accepted)} "
            f"rejected/unmatched={sum(not row['accepted'] for row in rows)} "
            f"rel={np.median(relative) if relative else math.nan:.4g} "
            f"support={np.median(support) if support else 0:.1f} "
            f"status={stats['convergence_status']} stable={stats['stable_assignments']} "
            f"dummy={stats['final_dummy_assignments']}",
            radii=radii,
        )
        for point_id in range(0, len(clouds[frame_a].xyz), max(1, len(clouds[frame_a].xyz) // 12)):
            track_id = lookup.get((frame_a, point_id))
            if track_id is not None:
                x, y = world_pixel(clouds[frame_a].xyz[point_id], projection, (960, 720))
                cv2.putText(image, f"T{track_id}", (x + 3, y - 3), cv2.FONT_HERSHEY_SIMPLEX,
                            0.3, (20, 20, 20), 1, cv2.LINE_AA)
        writer.write(image)
    writer.release()


def find_rgb_root(root: Path) -> Path:
    rgb = root / "rgb"
    if rgb.is_dir():
        return rgb
    raise FileNotFoundError("RGB visualization requested but ROOT/rgb is missing")


def rgb_frame_path(rgb_root: Path, camera: str, frame: int) -> Path:
    for suffix in (".png", ".jpg", ".jpeg"):
        path = rgb_root / camera / f"frame_{frame:06d}{suffix}"
        if path.is_file():
            return path
    raise FileNotFoundError(f"No RGB frame for {camera} frame {frame}")


def project_world(point: np.ndarray, calibration: Calibration) -> tuple[float, float] | None:
    camera = calibration.Rcw @ point + calibration.tcw
    if camera[2] <= EPS:
        return None
    return (
        calibration.fx * camera[0] / camera[2] + calibration.cx,
        calibration.fy * camera[1] / camera[2] + calibration.cy,
    )


def track_color(track: Track, labelled: bool) -> tuple[int, int, int]:
    if labelled:
        return LABEL_COLORS.get(track.label, LABEL_COLORS["AMBIGUOUS"])
    hue = (track.track_id * 47) % 180
    hsv = np.uint8([[[hue, 210, 230]]])
    return tuple(int(x) for x in cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0, 0])


def overlay_tracks(
    image: np.ndarray,
    calibration: Calibration,
    frame: int,
    tracks: dict[int, Track],
    labelled: bool,
) -> int:
    count = 0
    for track_id, track in tracks.items():
        if frame not in track.points:
            continue
        projected = project_world(track.points[frame], calibration)
        if projected is None:
            continue
        x, y = map(lambda value: int(round(value)), projected)
        if not (0 <= x < image.shape[1] and 0 <= y < image.shape[0]):
            continue
        color = track_color(track, labelled)
        history = [previous for previous in sorted(track.points) if previous <= frame][-8:]
        trail = []
        for previous in history:
            uv = project_world(track.points[previous], calibration)
            if uv is not None:
                trail.append(tuple(map(lambda value: int(round(value)), uv)))
        for start, end in zip(trail[:-1], trail[1:]):
            cv2.line(image, start, end, color, 1, cv2.LINE_AA)
        cv2.circle(image, (x, y), 3, color, -1, cv2.LINE_AA)
        if track_id % 12 == 0:
            cv2.putText(image, f"T{track_id}", (x + 3, y - 3), cv2.FONT_HERSHEY_SIMPLEX,
                        0.28, color, 1, cv2.LINE_AA)
        count += 1
    return count


def render_rgb_track_videos(
    root: Path,
    frames: list[int],
    calibrations: dict[str, Calibration],
    tracks: dict[int, Track],
    debug: Path,
    fps: float,
    labelled: bool,
) -> None:
    rgb_root = find_rgb_root(root)
    suffix = "rigid_labels" if labelled else "hungarian_tracks"
    camera_names = list(calibrations)
    sample = cv2.imread(str(rgb_frame_path(rgb_root, camera_names[0], frames[0])))
    height, width = sample.shape[:2]
    tile_size = (width // 2, height // 2)
    columns = min(4, len(camera_names))
    rows = math.ceil(len(camera_names) / columns)
    montage_size = (tile_size[0] * columns, tile_size[1] * rows)
    montage = video_writer(debug / f"all_cameras_{suffix}.mp4", fps, montage_size)
    per_camera = {}
    if not labelled:
        per_camera = {
            name: video_writer(debug / f"{name}_hungarian_tracks.mp4", fps, (width, height))
            for name in camera_names
        }
    for frame in frames:
        panels = []
        for name, calibration in calibrations.items():
            image = cv2.imread(str(rgb_frame_path(rgb_root, name, frame)))
            count = overlay_tracks(image, calibration, frame, tracks, labelled)
            cv2.rectangle(image, (0, 0), (width, 24), (0, 0, 0), -1)
            cv2.putText(image, f"{name} frame={frame:06d} projected_tracks={count}", (5, 16),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA)
            if not labelled:
                per_camera[name].write(image)
            panels.append(cv2.resize(image, tile_size, interpolation=cv2.INTER_AREA))
        while len(panels) < rows * columns:
            panels.append(np.zeros((tile_size[1], tile_size[0], 3), np.uint8))
        grid = np.concatenate([
            np.concatenate(panels[row * columns:(row + 1) * columns], axis=1)
            for row in range(rows)
        ], axis=0)
        montage.write(grid)
    montage.release()
    for writer in per_camera.values():
        writer.release()


def render_rigid_tracks_3d(
    frames: list[int],
    tracks: dict[int, Track],
    debug: Path,
    fps: float,
    projection,
) -> None:
    writer = video_writer(debug / "rigid_tracks_3d.mp4", fps, (960, 720))
    for frame in frames:
        points, colors, lines = [], [], []
        for track in tracks.values():
            if frame not in track.points:
                continue
            color = track_color(track, True)
            points.append(track.points[frame])
            colors.append(color)
            history = [previous for previous in sorted(track.points) if previous <= frame][-8:]
            lines.extend((track.points[a], track.points[b], color, 1) for a, b in zip(history[:-1], history[1:]))
        writer.write(render_world(
            points, colors, lines, projection,
            f"Rigid labels frame={frame:06d} tracks={len(points)} | STATIC remains visible",
        ))
    writer.release()


def save_matrix_debug(
    valid_ids: list[int],
    edges: list[dict],
    assignments: dict[int, tuple[int, float, np.ndarray]],
    debug: Path,
) -> None:
    ordered = sorted(valid_ids, key=lambda track_id: (assignments[track_id][0], track_id))
    if len(ordered) > 800:
        ordered = ordered[::math.ceil(len(ordered) / 800)]
    lookup = {track_id: index for index, track_id in enumerate(ordered)}
    size = len(ordered)
    rigidity = np.full((size, size), np.nan, np.float64)
    target = np.full((size, size), np.nan, np.float64)
    np.fill_diagonal(rigidity, 0.0)
    np.fill_diagonal(target, 1.0)
    for edge in edges:
        if edge["track_a"] not in lookup or edge["track_b"] not in lookup:
            continue
        i, j = lookup[edge["track_a"]], lookup[edge["track_b"]]
        rigidity[i, j] = rigidity[j, i] = edge["rigidity_error"]
        target[i, j] = target[j, i] = edge["q_same"]
    probabilities = np.stack([assignments[track_id][2] for track_id in ordered])
    learned = probabilities @ probabilities.T

    def save(path: Path, matrix: np.ndarray, invert: bool = False) -> None:
        finite = matrix[np.isfinite(matrix)]
        low, high = (float(np.min(finite)), float(np.percentile(finite, 98))) if finite.size else (0.0, 1.0)
        normalized = np.nan_to_num((matrix - low) / max(high - low, EPS), nan=0.0)
        if invert:
            normalized = 1.0 - normalized
        colored = cv2.applyColorMap(np.uint8(np.clip(normalized, 0, 1) * 255), cv2.COLORMAP_VIRIDIS)
        scale = max(1, min(8, 800 // max(size, 1)))
        cv2.imwrite(str(path), cv2.resize(colored, (size * scale, size * scale), interpolation=cv2.INTER_NEAREST))

    save(debug / "relative_distance_rigidity_matrix.png", rigidity, invert=True)
    save(debug / "relative_distance_samepart_target_matrix.png", target)
    save(debug / "learned_samepart_similarity_matrix.png", learned)


# =============================================================================
# Output and command-line orchestration
# =============================================================================

def run_static_dynamic_residual_pipeline(
    clouds: dict[int, FrameCloud],
    coarse_tracks: dict[int, Track],
    residuals: dict[int, ResidualCloud],
    frames: list[int],
    args: argparse.Namespace,
) -> tuple[
    dict[int, Track], dict[int, np.ndarray], list[dict], list[dict], list[dict], dict,
]:
    graphs = build_residual_graphs(
        residuals,
        args.residual_near_neighbors,
        args.residual_medium_neighbors,
        args.residual_far_neighbors,
    )
    motion_threshold, motion_noise = estimate_static_motion_threshold(
        coarse_tracks, clouds, frames,
    )
    node_probabilities: dict[int, np.ndarray] | None = None
    previous_probabilities: dict[int, np.ndarray] | None = None
    all_match_rows, all_quality_rows, all_history = [], [], []
    final_tracks: dict[int, Track] = {}
    final_edges: list[dict] = []
    final_pair_rows: dict[tuple[int, int], list[dict]] = {}
    relation_stats: dict = {}
    for outer_iteration in range(args.outer_iterations):
        outer_started = time.perf_counter()
        residual_tracks, pair_rows, quality_rows = run_residual_graph_matching(
            residuals, graphs, frames, args, node_probabilities,
        )
        se3_threshold = max(
            0.25 * float(np.median([graph.spacing for graph in graphs.values()])),
            2.0 * motion_threshold,
            EPS,
        )
        for pair, rows in pair_rows.items():
            validate_se3_models(
                rows, residuals[pair[0]], residuals[pair[1]],
                se3_threshold, seed=outer_iteration * 1000003 + pair[0] * 1009 + pair[1],
            )
        # A generic rigid consensus is validation, not a proposal mechanism.
        # Retain all graph matches when there is too little evidence to fit a
        # model; otherwise discard geometrically incoherent projections.
        for pair, rows in pair_rows.items():
            inliers = [row for row in rows if row["se3_inlier"]]
            if len(inliers) >= 3:
                pair_rows[pair] = inliers
            for row in pair_rows[pair]:
                row["track_id"] = -1
        annotate_skip_consistency(pair_rows, frames, args.residual_max_skip)
        residual_tracks = build_residual_tracks(residuals, frames, pair_rows)
        dynamic_anchors = validate_dynamic_anchors(
            residual_tracks, pair_rows, motion_threshold,
            args.residual_min_support, args.residual_relative_threshold,
        )
        quality_lookup = {(row["frame_a"], row["frame_b"]): row for row in quality_rows}
        for pair, rows in pair_rows.items():
            quality_lookup[pair]["accepted_matches"] = len(rows)
            quality_lookup[pair]["unmatched_source"] = (
                quality_lookup[pair]["source_residual_points"] - len(rows)
            )
            quality_lookup[pair]["unmatched_target"] = (
                quality_lookup[pair]["target_residual_points"]
                - len({row["target_id"] for row in rows})
            )
            quality_lookup[pair]["new_dynamic_anchors"] = len({
                row["track_id"] for row in rows if row["dynamic_anchor_accepted"]
            })
        if node_probabilities is not None:
            for track in residual_tracks.values():
                values = [node_probabilities[frame][point_id]
                          for frame, point_id in track.point_ids.items()]
                track.dynamic_probability = float(np.mean(values)) if values else 0.5
        spacing_values = [graph.spacing for graph in graphs.values() if len(graph.xyz) > 1]
        residual_spacing = max(float(np.median(spacing_values)) if spacing_values else 1.0, EPS)
        nodes, edges, relation_stats = build_binary_rigidity_relations(
            residual_tracks, coarse_tracks, frames, residual_spacing,
            neighbors=args.binary_relation_neighbors,
        )
        probabilities, history = optimize_binary_dynamic_probabilities(
            nodes, edges, dynamic_anchors,
            outer_iteration=outer_iteration,
            iterations=args.num_contrastive_iters,
            learning_rate=args.contrastive_lr,
            static_threshold=args.static_probability_threshold,
            dynamic_threshold=args.dynamic_probability_threshold,
            ignored_margin_edges=relation_stats["ignored_margin_edges"],
        )
        node_probabilities = probabilities_to_residual_nodes(
            residuals, residual_tracks, probabilities,
        )
        for rows in pair_rows.values():
            for row in rows:
                all_match_rows.append({"outer_iteration": outer_iteration, **row})
        all_quality_rows.extend({"outer_iteration": outer_iteration, **row} for row in quality_rows)
        all_history.extend(history)
        if previous_probabilities is None:
            maximum_change = math.inf
        else:
            maximum_change = max(
                float(np.max(np.abs(node_probabilities[frame] - previous_probabilities[frame])))
                if len(node_probabilities[frame]) else 0.0
                for frame in frames
            )
        positive_same = history[-1]["mean_positive_same_probability"] if history else math.nan
        negative_same = history[-1]["mean_negative_same_probability"] if history else math.nan
        print(
            f"[BINARY ALTERNATION] outer={outer_iteration + 1}/{args.outer_iterations} "
            f"tracks={len(residual_tracks)} matches={sum(len(rows) for rows in pair_rows.values())} "
            f"dynamic_anchors={len(dynamic_anchors)} positive_same={positive_same:.4f} "
            f"negative_same={negative_same:.4f} max_dp={maximum_change:.4g} "
            f"elapsed={time.perf_counter() - outer_started:.1f}s"
        )
        final_tracks, final_edges, final_pair_rows = residual_tracks, edges, pair_rows
        if (
            outer_iteration >= 2
            and maximum_change < args.outer_convergence_tolerance
        ):
            break
        previous_probabilities = {
            frame: values.copy() for frame, values in node_probabilities.items()
        }
    if node_probabilities is None:
        node_probabilities = {
            frame: np.full(len(residual.xyz), 0.5)
            for frame, residual in residuals.items()
        }
    return (
        final_tracks, node_probabilities, all_match_rows, all_quality_rows,
        all_history, {
            "motion_noise": motion_noise,
            "se3_validation_threshold": se3_threshold if graphs else math.nan,
            "binary_relations": relation_stats,
            "effective_outer_iterations": 1 + max(
                (row["outer_iteration"] for row in all_history), default=-1,
            ),
            "final_edges": final_edges,
            "final_pair_rows": final_pair_rows,
        },
    )


def render_residual_graph_matching_3d(
    residuals: dict[int, ResidualCloud],
    pair_rows: dict[tuple[int, int], list[dict]],
    quality_rows: list[dict],
    debug: Path,
    fps: float,
    projection,
) -> None:
    writer = video_writer(debug / "residual_graph_matching_3d.mp4", fps, (960, 720))
    quality_lookup = {(row["frame_a"], row["frame_b"]): row for row in quality_rows}
    minimum_delta = min((pair[1] - pair[0] for pair in pair_rows), default=1)
    adjacent = sorted(pair for pair in pair_rows if pair[1] - pair[0] == minimum_delta)
    for pair in adjacent:
        source, target = residuals[pair[0]], residuals[pair[1]]
        source_step = max(1, math.ceil(len(source.xyz) / 2500))
        target_step = max(1, math.ceil(len(target.xyz) / 2500))
        points = list(source.xyz[::source_step]) + list(target.xyz[::target_step])
        colors = [(100, 100, 100)] * len(source.xyz[::source_step]) + [
            (90, 150, 190)
        ] * len(target.xyz[::target_step])
        lines = []
        for row in pair_rows[pair][::max(1, math.ceil(len(pair_rows[pair]) / 1200))]:
            color = (30, 30, 240) if row["dynamic_anchor_accepted"] else (220, 210, 30)
            lines.append((
                source.xyz[row["source_id"]], target.xyz[row["target_id"]], color, 1,
            ))
        quality = quality_lookup[pair]
        header = (
            f"residual graph {pair[0]:06d}->{pair[1]:06d} accepted={quality['accepted_matches']} "
            f"unmatched={quality['unmatched_source']} support={quality['median_graph_support']:.1f} "
            f"deformation med/p90={quality['median_edge_deformation']:.3g}/"
            f"{quality['p90_edge_deformation']:.3g} skip={quality['skip_confirmed']}"
        )
        writer.write(render_world(points, colors, lines, projection, header))
    writer.release()


def render_residual_graph_matching_all_cameras(
    root: Path,
    residuals: dict[int, ResidualCloud],
    pair_rows: dict[tuple[int, int], list[dict]],
    calibrations: dict[str, Calibration],
    debug: Path,
    fps: float,
) -> None:
    rgb_root = find_rgb_root(root)
    camera_names = list(calibrations)
    minimum_delta = min((pair[1] - pair[0] for pair in pair_rows), default=1)
    adjacent = sorted(pair for pair in pair_rows if pair[1] - pair[0] == minimum_delta)
    if not adjacent:
        return
    sample = cv2.imread(str(rgb_frame_path(rgb_root, camera_names[0], adjacent[0][0])))
    height, width = sample.shape[:2]
    tile_size = (width // 2, height // 2)
    columns = min(4, len(camera_names))
    rows_count = math.ceil(len(camera_names) / columns)
    writer = video_writer(
        debug / "residual_graph_matching_all_cameras.mp4", fps,
        (tile_size[0] * columns, tile_size[1] * rows_count),
    )
    for pair in adjacent:
        accepted = {row["source_id"]: row for row in pair_rows[pair]}
        cloud = residuals[pair[0]]
        step = max(1, math.ceil(len(cloud.xyz) / 3500))
        panels = []
        for name, calibration in calibrations.items():
            image = cv2.imread(str(rgb_frame_path(rgb_root, name, pair[0])))
            projected = 0
            for point_id in range(0, len(cloud.xyz), step):
                pixel = project_world(cloud.xyz[point_id], calibration)
                if pixel is None:
                    continue
                x, y = map(int, map(round, pixel))
                if not (0 <= x < width and 0 <= y < height):
                    continue
                row = accepted.get(point_id)
                color = (
                    (30, 30, 240) if row and row["dynamic_anchor_accepted"]
                    else (220, 210, 30) if row else (110, 110, 110)
                )
                cv2.circle(image, (x, y), 1, color, -1, cv2.LINE_AA)
                projected += 1
            cv2.rectangle(image, (0, 0), (width, 24), (0, 0, 0), -1)
            cv2.putText(
                image,
                f"{name} residual {pair[0]:06d}->{pair[1]:06d} projected={projected}",
                (5, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA,
            )
            panels.append(cv2.resize(image, tile_size, interpolation=cv2.INTER_AREA))
        while len(panels) < rows_count * columns:
            panels.append(np.zeros((tile_size[1], tile_size[0], 3), np.uint8))
        writer.write(np.concatenate([
            np.concatenate(panels[row * columns:(row + 1) * columns], axis=1)
            for row in range(rows_count)
        ], axis=0))
    writer.release()


def save_binary_outputs(
    out: Path,
    tracks: dict[int, Track],
    edges: list[dict],
    history: list[dict],
    ghost_rows: list[dict],
) -> None:
    edge_fields = [
        "track_a", "track_b", "common_frames", "rigidity_error",
        "temporal_match_confidence", "both_trusted_static", "q_same", "weight", "evidence",
    ]
    write_csv(out / "relative_distance_temporal_edges.csv", edges, edge_fields)
    write_csv(out / "contrastive_history.csv", history, [
        "outer_iteration", "inner_iteration", "loss", "positive_pair_loss",
        "negative_pair_loss", "static_anchor_loss", "dynamic_anchor_loss",
        "mean_positive_same_probability", "mean_negative_same_probability",
        "num_positive_edges", "num_negative_edges", "num_ignored_margin_edges",
        "mean_p_dynamic", "num_static", "num_dynamic", "num_uncertain",
    ])
    write_csv(out / "static_ghost_events.csv", ghost_rows, [
        "event", "frame", "track_id", "ghost_id", "source_track_id",
        "distance", "ghost_age_frames", "assigned_cluster_id",
        "static_prior", "static_prior_weight",
    ])
    rows = []
    for track_id, track in sorted(tracks.items()):
        rows.append({
            "track_id": track_id,
            "track_length": len(track.points),
            "mean_match_confidence": float(np.mean(track.match_confidences)) if track.match_confidences else 0.0,
            "cluster_id": track.cluster_id,
            "rigid_label": track.label,
            "assignment_confidence": track.confidence,
            "p_static": 1.0 - track.dynamic_probability,
            "p_dynamic": track.dynamic_probability,
            "label_source": track.label_source,
            "seed_type": track.seed_type,
            "seed_strength": track.seed_strength,
            "ghost_source_track_id": track.ghost_source_track_id,
            "ghost_distance": track.ghost_distance,
            "ghost_age_frames": track.ghost_age_frames,
        })
    write_csv(out / "rigid_part_assignments.csv", rows)
    write_csv(out / "rigid_part_transforms.csv", [], [
        "cluster_id", "reference_frame", "frame", "num_correspondences",
        "rotation_angle_rad", "translation_norm", "median_residual", "identity_motion_score",
    ])
    names = {"0": "STATIC"}
    if any(track.label == "MOVING_1" for track in tracks.values()):
        names["1"] = "MOVING_1"
    (out / "rigid_group_names.json").write_text(json.dumps(names, indent=2) + "\n")


def offline_gt_label_audit(
    root: Path,
    out: Path,
    cloud: FrameCloud,
    tracks: dict[int, Track],
    moving_part: int,
) -> dict:
    """Evaluate frozen labels only; this function is deliberately called last."""
    from utils.rigid_killing_validation import load_rigid_dataset_ground_truth

    truth = load_rigid_dataset_ground_truth(
        root, torch.as_tensor(cloud.xyz), moving_part=moving_part,
    )
    gt_dynamic = truth.moving_mask.cpu().numpy().astype(bool)
    prediction = dynamic_labels_for_frame(cloud, tracks)
    predicted_static = prediction == 0.0
    predicted_dynamic = prediction == 1.0
    report = {
        "ground_truth_used_as_algorithm_input": False,
        "audit_stage": "after frozen labels and exports",
        "frame": cloud.frame,
        "moving_part": moving_part,
        "num_points": len(prediction),
        "static_precision": float(np.mean(~gt_dynamic[predicted_static])) if np.any(predicted_static) else math.nan,
        "dynamic_precision": float(np.mean(gt_dynamic[predicted_dynamic])) if np.any(predicted_dynamic) else math.nan,
        "dynamic_recall": float(np.mean(prediction[gt_dynamic] == 1.0)) if np.any(gt_dynamic) else math.nan,
        "moving_uncertain_fraction": float(np.mean(prediction[gt_dynamic] == 0.5)) if np.any(gt_dynamic) else math.nan,
        "moving_false_static_fraction": float(np.mean(prediction[gt_dynamic] == 0.0)) if np.any(gt_dynamic) else math.nan,
    }
    (out / "offline_gt_label_audit.json").write_text(json.dumps(report, indent=2) + "\n")
    return report

def save_final_outputs(
    out: Path,
    tracks: dict[int, Track],
    edges: list[dict],
    history: list[dict],
    transform_rows: list[dict],
    group_names: dict[int, str],
    ghost_rows: list[dict],
) -> None:
    write_csv(out / "relative_distance_temporal_edges.csv", edges, [
        "edge_type", "track_a", "track_b", "common_frames", "median_pair_distance",
        "median_relative_change", "MAD_relative_change", "p90_relative_change",
        "max_robust_relative_change", "rigidity_error", "track_match_confidence",
        "track_motion_error_a", "track_motion_error_b",
        "track_static_probability_a", "track_static_probability_b",
        "track_static_prior_weight_a", "track_static_prior_weight_b",
        "temporal_velocity_difference", "neighbor_temporal_relative_change",
        "neighbor_temporal_transitions", "spatial_same_target", "temporal_same_target",
        "uncertain_neighbor_target", "uncertain_neighbor_weight",
        "both_static_probability", "same_moving_rigid_probability",
        "q_same", "weight", "spatial_weight", "temporal_weight",
    ])
    write_csv(out / "contrastive_history.csv", history, [
        "iteration", "loss", "rigidity_loss", "spatial_consistency_loss",
        "temporal_consistency_loss", "uncertain_neighbor_loss",
        "static_anchor_loss", "uncertain_completion_loss",
        "mean_positive_similarity",
        "mean_negative_similarity", "assignment_entropy", "cluster_occupancy",
    ])
    write_csv(out / "static_ghost_events.csv", ghost_rows, [
        "event", "frame", "track_id", "ghost_id", "source_track_id",
        "distance", "ghost_age_frames", "assigned_cluster_id",
        "static_prior", "static_prior_weight",
    ])
    assignment_rows = []
    max_groups = max((len(track.probabilities) for track in tracks.values()
                      if track.probabilities is not None), default=0)
    for track_id, track in sorted(tracks.items()):
        row = {
            "track_id": track_id,
            "track_length": len(track.points),
            "mean_match_confidence": float(np.mean(track.match_confidences)) if track.match_confidences else 0.0,
            "cluster_id": track.cluster_id,
            "rigid_label": track.label,
            "assignment_confidence": track.confidence,
            "initial_cluster_id": track.initial_cluster_id,
            "initial_rigid_label": track.initial_label,
            "initial_assignment_confidence": track.initial_confidence,
            "label_source": track.label_source,
            "ghost_source_track_id": track.ghost_source_track_id,
            "ghost_distance": track.ghost_distance,
            "ghost_age_frames": track.ghost_age_frames,
            "temporal_matches": track.temporal_matches,
            "static_anchor_matches": track.static_anchor_matches,
            "stationary_occupancy_ambiguous": track.stationary_occupancy_ambiguous,
            "knn_teacher_count": track.knn_teacher_count,
            "temporal_trajectory_id": track.temporal_trajectory_id,
            "temporal_coherence_links": track.temporal_coherence_links,
            "static_prior": track.static_prior,
            "static_prior_weight": track.static_prior_weight,
            "seed_type": track.seed_type,
            "seed_strength": track.seed_strength,
            "dynamic_static_relative_score": track.dynamic_static_relative_score,
            "dynamic_static_relative_support": track.dynamic_static_relative_support,
        }
        for cluster in range(max_groups):
            row[f"probability_cluster_{cluster}"] = (
                float(track.probabilities[cluster]) if track.probabilities is not None else math.nan
            )
        assignment_rows.append(row)
    write_csv(out / "rigid_part_assignments.csv", assignment_rows)
    write_csv(out / "rigid_part_transforms.csv", transform_rows)
    (out / "rigid_group_names.json").write_text(
        json.dumps({str(cluster): name for cluster, name in group_names.items()}, indent=2) + "\n"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Pure depth + calibrated geometry rigid-part discovery (no appearance matching)."
    )
    parser.add_argument("root", help="Dataset root, e.g. input/USB_100109")
    parser.add_argument("--num-frames", type=int, default=None)
    parser.add_argument("--frame-start", type=int, default=0)
    parser.add_argument("--frame-stride", type=int, default=1)
    parser.add_argument("--num-moving-parts", type=int, default=1)
    parser.add_argument("--depth-dir", default="depth_npy")
    parser.add_argument("--depth-scale", type=float, default=1.0,
                        help="Multiplicative conversion from stored depth to world units")
    parser.add_argument("--min-depth", type=float, default=None)
    parser.add_argument("--max-depth", type=float, default=None)
    parser.add_argument("--use-mask", action="store_true")
    parser.add_argument("--voxel-size", type=float, default=0.03)
    parser.add_argument("--target-points", type=int, default=0,
                        help="Optional first-frame cell cap; 0 keeps the requested motion-resolving voxel size")
    parser.add_argument("--num-rel-neighbors", type=int, default=8)
    parser.add_argument("--hungarian-candidates", type=int, default=16,
                        help="Sparse local-signature candidates retained per point and direction")
    parser.add_argument("--hungarian-refine-iters", type=int, default=5,
                        help="Compatibility-graph message-passing iterations")
    parser.add_argument("--min-relational-support", type=int, default=3)
    parser.add_argument("--compatibility-relative-threshold", type=float, default=0.03,
                        help="Maximum normalized distance error between compatible candidate matches")
    parser.add_argument("--unmatched-cost-scale", type=float, default=2.5)
    parser.add_argument("--track-max-gap", type=int, default=0,
                        help="Missing selected frames bridged by skip-frame relative matching")
    parser.add_argument("--residual-fine-tolerance", type=float, default=0.005,
                        help="Tiny geometric deduplication tolerance for raw residual depth samples")
    parser.add_argument("--residual-target-points", type=int, default=2048,
                        help="Per-frame adaptive geometric graph budget; 0 disables the coreset")
    parser.add_argument("--residual-near-neighbors", type=int, default=8)
    parser.add_argument("--residual-medium-neighbors", type=int, default=8)
    parser.add_argument("--residual-far-neighbors", type=int, default=8)
    parser.add_argument("--residual-candidates", type=int, default=16,
                        help="Permissive local-signature candidates per residual point")
    parser.add_argument("--residual-graph-iters", type=int, default=5)
    parser.add_argument("--residual-min-support", type=int, default=3)
    parser.add_argument("--residual-relative-threshold", type=float, default=0.08)
    parser.add_argument("--residual-max-skip", type=int, default=0,
                        help="Validate adjacent matches directly through t+2 and t+3")
    parser.add_argument("--outer-iterations", type=int, default=1,
                        help="Full graph-matching/binary cycles; use 3 only for a slower refinement run")
    parser.add_argument("--outer-convergence-tolerance", type=float, default=0.01)
    parser.add_argument("--binary-relation-neighbors", type=int, default=12)
    parser.add_argument("--static-probability-threshold", type=float, default=0.25)
    parser.add_argument("--dynamic-probability-threshold", type=float, default=0.75)
    parser.add_argument("--num-contrastive-iters", type=int, default=200)
    parser.add_argument("--contrastive-lr", type=float, default=0.05)
    parser.add_argument("--label-spatial-weight", type=float, default=0.001,
                        help="Weight of local 3D same/part-apart consistency")
    parser.add_argument("--label-temporal-weight", type=float, default=0.01,
                        help="Weight of multi-frame velocity consistency")
    parser.add_argument("--label-static-weight", type=float, default=0.01,
                        help="Weight of static-anchor and occlusion unary evidence")
    parser.add_argument("--label-completion-weight", type=float, default=1.0,
                        help="Weight for uncertain fragments learning from the detached temporal core")
    parser.add_argument("--label-uncertain-knn-weight", type=float, default=1.0,
                        help="Weight for uncertain labels learning from temporally rigid kNNs")
    parser.add_argument("--label-uncertain-knn-k", type=int, default=8,
                        help="Spatial neighbours per occupancy-uncertain track")
    parser.add_argument("--label-teacher-confidence", type=float, default=0.9,
                        help="Minimum temporal-core confidence for fragment teaching")
    parser.add_argument("--final-temporal-label-weight", type=float, default=1.0,
                        help="Label cost in the final trajectory-recovery Hungarian pass")
    parser.add_argument("--keyframe-joint-optimization", action="store_true",
                        help="Jointly optimize keyframe matching, per-point transforms, and static/dynamic labels")
    parser.add_argument("--keyframe-count", type=int, default=2,
                        help="Evenly spaced keyframes; default uses first and last selected frames")
    parser.add_argument("--keyframe-iters", type=int, default=200,
                        help="Iterations per keyframe pair for joint transform/label optimization")
    parser.add_argument("--keyframe-lr", type=float, default=0.03)
    parser.add_argument("--keyframe-match-refresh", type=int, default=25,
                        help="Refresh sparse Hungarian matches every N iterations")
    parser.add_argument("--keyframe-match-position-weight", type=float, default=0.25)
    parser.add_argument("--keyframe-match-weight", type=float, default=1.0)
    parser.add_argument("--keyframe-relative-weight", type=float, default=0.5)
    parser.add_argument("--keyframe-static-identity-weight", type=float, default=0.5)
    parser.add_argument("--keyframe-dynamic-rigid-weight", type=float, default=0.5)
    parser.add_argument("--keyframe-transform-smoothness-weight", type=float, default=0.05)
    parser.add_argument("--keyframe-label-smoothness-weight", type=float, default=0.25)
    parser.add_argument("--keyframe-static-prior-weight", type=float, default=2.0)
    parser.add_argument("--keyframe-evidence-weight", type=float, default=0.5)
    parser.add_argument("--keyframe-polarization-weight", type=float, default=0.02)
    parser.add_argument("--keyframe-binarization-weight", type=float, default=0.05,
                        help="Hardens non-anchor keyframe dynamic probabilities toward 0 or 1")
    parser.add_argument("--keyframe-composition-weight", type=float, default=0.25,
                        help="For skip keyframe pairs, penalize disagreement with composed shorter-pair SE(3) transforms")
    parser.add_argument(
        "--disable-static-anchor-priors", "--disable-static-ghost-memory",
        "--disable-occlusion-label-correction",
        dest="disable_static_ghost_memory", action="store_true",
    )
    parser.add_argument(
        "--static-ghost-radius-scale", "--occlusion-radius-scale",
        dest="static_ghost_radius_scale", type=float, default=0.125,
        help="Static reappearance radius in units of scene point spacing",
    )
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--no-visual-debug", action="store_true")
    parser.add_argument("--save-raw-depth-ply", action="store_true",
                        help="Write large per-camera raw depth PLYs (off by default)")
    parser.add_argument("--save-full-resolution-residual", action="store_true",
                        help="Also write the large pre-coreset residual PLYs")
    parser.add_argument("--debug-fps", type=float, default=10.0)
    parser.add_argument("--gt-audit", action="store_true",
                        help="Run an optional post-solve label audit; GT never enters estimation")
    parser.add_argument("--gt-moving-part", type=int, default=1)
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if args.num_frames is not None and args.num_frames < 2:
        raise ValueError("--num-frames must be at least 2")
    if args.frame_stride < 1:
        raise ValueError("--frame-stride must be positive")
    if args.num_moving_parts < 0:
        raise ValueError("--num-moving-parts cannot be negative")
    if args.depth_scale <= 0 or args.voxel_size <= 0:
        raise ValueError("--depth-scale and --voxel-size must be positive")
    if args.target_points < 0:
        raise ValueError("--target-points cannot be negative")
    if args.target_points and args.target_points < args.min_relational_support + 2:
        raise ValueError("--target-points is too small for --min-relational-support")
    if args.num_rel_neighbors < args.min_relational_support or args.hungarian_refine_iters < 1:
        raise ValueError("neighbor count and compatibility message iterations must be positive")
    if args.hungarian_candidates < 1:
        raise ValueError("--hungarian-candidates must be positive")
    if args.min_relational_support < 1:
        raise ValueError("--min-relational-support must be positive")
    if args.compatibility_relative_threshold <= 0:
        raise ValueError("--compatibility-relative-threshold must be positive")
    if args.track_max_gap not in (0, 1):
        raise ValueError("--track-max-gap currently supports 0 or 1 missing selected frame")
    if args.residual_fine_tolerance <= 0 or args.residual_fine_tolerance >= args.voxel_size:
        raise ValueError("--residual-fine-tolerance must be positive and finer than --voxel-size")
    if args.residual_target_points < 0:
        raise ValueError("--residual-target-points cannot be negative")
    if args.residual_target_points and args.residual_target_points < (
        args.residual_near_neighbors + args.residual_medium_neighbors + args.residual_far_neighbors + 1
    ):
        raise ValueError("--residual-target-points is too small for the requested multiscale graph")
    if min(
        args.residual_near_neighbors,
        args.residual_medium_neighbors,
        args.residual_far_neighbors,
        args.residual_candidates,
        args.residual_graph_iters,
        args.residual_min_support,
        args.binary_relation_neighbors,
    ) < 1:
        raise ValueError("residual graph counts must be positive")
    if args.residual_relative_threshold <= 0:
        raise ValueError("--residual-relative-threshold must be positive")
    if args.residual_max_skip not in (0, 1, 2):
        raise ValueError("--residual-max-skip supports 0 (adjacent), 1 (t+2), or 2 (t+3)")
    if not 1 <= args.outer_iterations <= 5:
        raise ValueError("--outer-iterations must be between 1 and 5")
    if args.outer_convergence_tolerance <= 0:
        raise ValueError("--outer-convergence-tolerance must be positive")
    if not 0.0 < args.static_probability_threshold < args.dynamic_probability_threshold < 1.0:
        raise ValueError("probability thresholds must satisfy 0 < static < dynamic < 1")
    if args.unmatched_cost_scale <= 0 or args.num_contrastive_iters < 1 or args.contrastive_lr <= 0:
        raise ValueError("unmatched scale and contrastive optimization arguments must be positive")
    if args.static_ghost_radius_scale <= 0:
        raise ValueError("--static-ghost-radius-scale must be positive")
    if args.label_uncertain_knn_k < 1:
        raise ValueError("--label-uncertain-knn-k must be positive")
    if not 0.5 < args.label_teacher_confidence <= 1.0:
        raise ValueError("--label-teacher-confidence must be in (0.5, 1]")
    if args.final_temporal_label_weight < 0:
        raise ValueError("--final-temporal-label-weight cannot be negative")
    if args.keyframe_count < 2:
        raise ValueError("--keyframe-count must be at least 2")
    if args.keyframe_iters < 1:
        raise ValueError("--keyframe-iters must be positive")
    if args.keyframe_lr <= 0:
        raise ValueError("--keyframe-lr must be positive")
    if args.keyframe_match_refresh < 1:
        raise ValueError("--keyframe-match-refresh must be positive")
    if min(
        args.keyframe_match_position_weight, args.keyframe_match_weight,
        args.keyframe_relative_weight, args.keyframe_static_identity_weight,
        args.keyframe_dynamic_rigid_weight, args.keyframe_transform_smoothness_weight,
        args.keyframe_label_smoothness_weight, args.keyframe_static_prior_weight,
        args.keyframe_evidence_weight, args.keyframe_polarization_weight,
        args.keyframe_binarization_weight, args.keyframe_composition_weight,
    ) < 0:
        raise ValueError("keyframe joint loss weights cannot be negative")
    if min(
        args.label_spatial_weight, args.label_temporal_weight,
        args.label_static_weight, args.label_completion_weight,
        args.label_uncertain_knn_weight,
    ) < 0:
        raise ValueError("label-consistency loss weights cannot be negative")


def prepare_output(root: Path, explicit: str | None, force: bool) -> Path:
    out = Path(explicit).expanduser().resolve() if explicit else root / f"{root.name}{DEFAULT_OUTPUT_SUFFIX}"
    out = out.resolve()
    if out == root or out == Path("/"):
        raise RuntimeError(f"Refusing to use unsafe output directory: {out}")
    if out.exists():
        if not force:
            raise FileExistsError(f"Output already exists: {out}; pass --force to replace it")
        shutil.rmtree(out)
    out.mkdir(parents=True)
    return out


def _legacy_kpart_main() -> None:
    args = parse_args()
    validate_args(args)
    root = Path(args.root).expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(root)
    depth_root = root / args.depth_dir
    calibrations = load_calibrations(root)
    frames, extension = select_frames(
        depth_root, list(calibrations), args.frame_start, args.frame_stride, args.num_frames,
    )
    if args.keyframe_joint_optimization:
        all_frames = frames
        frames = sample_keyframes(all_frames, args.keyframe_count)
        print(
            f"[KEYFRAME JOINT] using {len(frames)}/{len(all_frames)} selected frames "
            f"for depth loading and matching: {frames}"
        )
    out = prepare_output(root, args.output_dir, args.force)
    parameters = vars(args).copy()
    parameters["root"] = str(root)
    parameters["output_dir"] = str(out)

    clouds = build_frame_clouds(
        root, out, depth_root, extension, calibrations, frames,
        args.depth_scale, args.min_depth, args.max_depth, args.use_mask,
        args.voxel_size, args.target_points, args.save_raw_depth_ply,
    )
    matcher_kwargs = {
        "num_rel_neighbors": args.num_rel_neighbors,
        "hungarian_candidates": args.hungarian_candidates,
        "refine_iters": args.hungarian_refine_iters,
        "min_relational_support": args.min_relational_support,
        "unmatched_cost_scale": args.unmatched_cost_scale,
        "compatibility_relative_threshold": args.compatibility_relative_threshold,
    }
    tracks, match_rows, pair_stats = build_tracks(
        clouds, frames, matcher_kwargs, args.track_max_gap,
    )
    write_csv(out / "temporal_hungarian_matches.csv", match_rows)
    pair_quality_rows = []
    for stats in pair_stats.values():
        row = stats.copy()
        row["changed_assignments"] = json.dumps(row["changed_assignments"])
        pair_quality_rows.append(row)
    write_csv(out / "hungarian_pair_quality.csv", pair_quality_rows)

    debug = out / "visual_debug"
    projection = fixed_world_projection(clouds, calibrations)
    if not args.no_visual_debug:
        debug.mkdir(parents=True, exist_ok=True)
        render_depth_geometry_debug(clouds, frames, calibrations, debug, args.debug_fps, projection)
        # Matching is rendered before contrastive labels, as required by the
        # implementation order and to expose bad correspondences directly.
        render_hungarian_3d(clouds, frames, tracks, match_rows, pair_stats, debug, args.debug_fps, projection)
        render_rgb_track_videos(root, frames, calibrations, tracks, debug, args.debug_fps, labelled=False)

    prior_track_ids = [track_id for track_id, track in tracks.items() if len(track.points) >= 3]
    prior_spacing = scene_spacing(tracks, prior_track_ids, frames)
    ghost_rows = []
    ghost_stats = {
        "enabled": False,
        "unmatched_birth_policy": "no birth prior when static-anchor priors are disabled",
    }
    if not args.disable_static_ghost_memory:
        ghost_rows, ghost_stats = build_static_anchor_ghost_priors(
            tracks, frames, prior_spacing, args.static_ghost_radius_scale,
        )
        print(
            "[STATIC PRIORS BEFORE OPTIMIZATION] "
            f"created={ghost_stats['ghosts_created']} "
            f"reactivated={ghost_stats['ghost_reactivations']} "
            f"unmatched_births_left_unlabelled={ghost_stats['births_without_ghost_left_unlabelled']}"
        )
    write_csv(out / "temporal_3d_tracks.csv", track_rows(tracks))
    valid_ids, edges, spacing, modes = build_rigidity_graph(
        tracks, frames, uncertain_knn_neighbors=args.label_uncertain_knn_k,
    )
    minimum_long_tracks = 3 * (args.num_moving_parts + 1)
    median_track_length = float(np.median([len(tracks[track_id].points) for track_id in valid_ids])) if valid_ids else 0.0
    print(
        f"[TRACK QUALITY] total={len(tracks)} optimized={len(valid_ids)} "
        f"temporal_core={modes.get('temporal_core_tracks', len(valid_ids))} "
        f"uncertain_completion={modes.get('uncertain_optimized_tracks', 0)} "
        f"uncertain_knn_edges={modes.get('uncertain_neighbor_edges', 0)} "
        f"median_supported_length={median_track_length:.1f} edges={len(edges)}"
    )
    if len(valid_ids) < minimum_long_tracks or not edges:
        raise RuntimeError(
            "Temporal Hungarian tracks are not sufficient/plausible for contrastive grouping; "
            f"inspect {debug / 'hungarian_3d.mp4'} and temporal_hungarian_matches.csv"
        )
    assignments, history = optimize_contrastive_groups(
        valid_ids, edges, args.num_moving_parts + 1,
        args.num_contrastive_iters, args.contrastive_lr,
        args.label_spatial_weight, args.label_temporal_weight, args.label_static_weight,
        args.label_completion_weight, tracks,
        uncertain_neighbor_loss_weight=args.label_uncertain_knn_weight,
    )
    for track_id, (cluster, confidence, _probabilities) in assignments.items():
        tracks[track_id].initial_cluster_id = cluster
        tracks[track_id].initial_confidence = confidence
        tracks[track_id].initial_label = (
            "STATIC" if cluster == 0 and confidence >= CONFIDENT_LABEL_THRESHOLD
            else f"MOVING_{cluster}" if confidence >= CONFIDENT_LABEL_THRESHOLD
            else "AMBIGUOUS"
        )
    fragment_knn_stats = solve_short_fragments_with_certain_knn(
        tracks,
        assignments,
        frames,
        spacing,
        neighbors=args.label_uncertain_knn_k,
        teacher_confidence=args.label_teacher_confidence,
        edges=edges,
    )
    temporal_coherence_rows, temporal_coherence_stats = final_temporal_hungarian_coherence(
        clouds,
        frames,
        tracks,
        assignments,
        matcher_kwargs,
        args.track_max_gap,
        args.final_temporal_label_weight,
        args.label_teacher_confidence,
    )
    write_csv(out / "final_temporal_hungarian_coherence.csv", temporal_coherence_rows, [
        "frame_a", "frame_b", "track_a", "track_b", "label_similarity",
        "relative_cost", "relational_support", "assignment_margin", "merged",
        "relaxed_geometry_recovery", "reject_reason",
    ])
    print(
        "[FRAGMENT LABEL SOLVE] "
        f"short={fragment_knn_stats['short_fragments']} "
        f"initialized_uncertain={fragment_knn_stats['previously_unoptimized_initialized_uncertain']} "
        f"knn_solved={fragment_knn_stats['solved_from_certain_knn']} "
        f"unsupported={fragment_knn_stats['left_uniform_uncertain']} "
        f"changed={fragment_knn_stats['changed_clusters']}"
    )
    print(
        "[FINAL TEMPORAL HUNGARIAN] "
        f"links={temporal_coherence_stats['merged_fragment_links']} "
        f"relaxed={temporal_coherence_stats['relaxed_weak_margin_links']} "
        f"trajectories={temporal_coherence_stats['recovered_trajectories']} "
        f"tracks={temporal_coherence_stats['tracks_in_recovered_trajectories']} "
        f"changed={temporal_coherence_stats['changed_fragment_clusters']}"
    )
    contrastive_assignments = assignments
    transform_rows, scores, empirical_static_cluster = fit_group_transforms(
        tracks, assignments, args.num_moving_parts + 1, frames, spacing,
    )
    static_cluster = 0
    if empirical_static_cluster != static_cluster:
        print(
            "[STATIC CONTRACT WARNING] static-prior cluster 0 is not the lowest-motion cluster; "
            f"empirical candidate={empirical_static_cluster}. Keeping cluster 0 so the unary "
            "static objective and output semantics cannot silently swap."
        )
    group_names = apply_final_labels(tracks, assignments, scores, static_cluster)
    initial_point_labels = export_initial_point_labels(
        root, out, clouds[frames[0]], tracks,
    )
    save_final_outputs(
        out, tracks, edges, history, transform_rows, group_names, ghost_rows,
    )

    print(
        f"[RIGID CONTRAST]\n"
        f"  optimized tracks={len(valid_ids)}\n"
        f"  temporal rigidity edges={modes.get('temporal_rigidity_edges', len(edges))}\n"
        f"  uncertain completion edges={modes.get('uncertain_completion_edges', 0)}\n"
        f"  uncertain temporal-kNN edges={modes.get('uncertain_neighbor_edges', 0)}\n"
        f"  static-motion mode={math.exp(modes['track_motion']['low_mode']):.6g}\n"
        f"  moving-motion mode={math.exp(modes['track_motion']['high_mode']):.6g}\n"
        f"  same-rigid mode={math.exp(modes['relative_rigidity']['low_mode']):.6g}\n"
        f"  different-rigid mode={math.exp(modes['relative_rigidity']['high_mode']):.6g}\n"
        f"  final loss={history[-1]['loss']:.6g}"
    )
    print("[PARTS]")
    for cluster in sorted(scores):
        count = sum(value[0] == cluster for value in assignments.values())
        print(f"  cluster {cluster} tracks={count} identity_motion_score={scores[cluster]:.6g}")
    print(f"  STATIC = cluster {static_cluster}")

    if not args.no_visual_debug:
        render_rgb_track_videos(root, frames, calibrations, tracks, debug, args.debug_fps, labelled=True)
        render_rigid_tracks_3d(frames, tracks, debug, args.debug_fps, projection)
        save_matrix_debug(valid_ids, edges, contrastive_assignments, debug)

    summary = {
        "status": "complete",
        "estimator_inputs": [
            "depth", "fixed_camera_intrinsics", "fixed_camera_extrinsics",
            "stationary_anchor_proximity", "relative_3d_distances",
            "local_3d_neighborhoods", "temporal_velocity_consistency",
        ],
        "appearance_used_by_estimator": False,
        "colmap_usage": (
            "calibration only during estimation; seed points opened after frozen labels only "
            "to write the labelled 3DGS handoff; executable not called"
        ),
        "rgb_usage": "visualization only" if not args.no_visual_debug else "not opened",
        "parameters": parameters,
        "selected_frames": frames,
        "num_cameras": len(calibrations),
        "effective_voxel_size": clouds[frames[0]].quality["voxel_size"],
        "sampling": clouds[frames[0]].quality["sampling"],
        "hungarian_converged_pairs": sum(stats["converged"] for stats in pair_stats.values()),
        "hungarian_total_pairs": len(pair_stats),
        "hungarian_dummy_assignments": sum(
            stats["final_dummy_assignments"] for stats in pair_stats.values()
        ),
        "sparse_candidate_edges": sum(stats["candidate_edges"] for stats in pair_stats.values()),
        "compatibility_graph_edges": sum(stats["compatibility_edges"] for stats in pair_stats.values()),
        "skip_frame_bridges": sum(stats["bridged_tracks"] for stats in pair_stats.values()),
        "num_tracks": len(tracks),
        "num_optimized_tracks": len(valid_ids),
        "num_final_labelled_or_uncertain_tracks": len(assignments),
        "num_temporally_supported_tracks": modes.get("temporal_core_tracks", len(valid_ids)),
        "num_uncertain_tracks_inside_optimizer": modes.get("uncertain_optimized_tracks", 0),
        "num_label_edges": len(edges),
        "num_temporal_rigidity_edges": modes.get("temporal_rigidity_edges", len(edges)),
        "num_uncertain_completion_edges": modes.get("uncertain_completion_edges", 0),
        "num_uncertain_neighbor_edges": modes.get("uncertain_neighbor_edges", 0),
        "confident_label_threshold": CONFIDENT_LABEL_THRESHOLD,
        "scene_spacing": spacing,
        "rigidity_modes": modes,
        "static_cluster": static_cluster,
        "group_names": {str(key): value for key, value in group_names.items()},
        "identity_motion_scores": {str(key): value for key, value in scores.items()},
        "static_anchor_ghost_priors": ghost_stats,
        "short_fragment_certain_knn": fragment_knn_stats,
        "final_temporal_hungarian_coherence": temporal_coherence_stats,
        "initial_point_cloud_labels": initial_point_labels,
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(f"[COMPLETE] {out}")


def main() -> None:
    args = parse_args()
    validate_args(args)
    root = Path(args.root).expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(root)
    depth_root = root / args.depth_dir
    calibrations = load_calibrations(root)
    frames, extension = select_frames(
        depth_root, list(calibrations), args.frame_start, args.frame_stride, args.num_frames,
    )
    if args.keyframe_joint_optimization:
        all_frames = frames
        frames = sample_keyframes(all_frames, args.keyframe_count)
        print(
            f"[KEYFRAME JOINT] using {len(frames)}/{len(all_frames)} selected frames "
            f"for depth loading and matching: {frames}"
        )
    out = prepare_output(root, args.output_dir, args.force)
    parameters = vars(args).copy()
    parameters["root"] = str(root)
    parameters["output_dir"] = str(out)

    # Phase A: retain the existing conservative coarse static/ghost discovery.
    phase_a_started = time.perf_counter()
    clouds = build_frame_clouds(
        root, out, depth_root, extension, calibrations, frames,
        args.depth_scale, args.min_depth, args.max_depth, args.use_mask,
        args.voxel_size, args.target_points, args.save_raw_depth_ply,
    )
    coarse_matcher = {
        "num_rel_neighbors": args.num_rel_neighbors,
        "hungarian_candidates": args.hungarian_candidates,
        "refine_iters": args.hungarian_refine_iters,
        "min_relational_support": args.min_relational_support,
        "unmatched_cost_scale": args.unmatched_cost_scale,
        "compatibility_relative_threshold": args.compatibility_relative_threshold,
    }
    coarse_tracks, coarse_match_rows, coarse_pair_stats = build_tracks(
        clouds, frames, coarse_matcher, args.track_max_gap,
    )
    write_csv(out / "temporal_hungarian_matches.csv", coarse_match_rows)
    coarse_quality = []
    for stats in coarse_pair_stats.values():
        row = stats.copy()
        row["changed_assignments"] = json.dumps(row["changed_assignments"])
        coarse_quality.append(row)
    write_csv(out / "hungarian_pair_quality.csv", coarse_quality)

    prior_ids = [track_id for track_id, track in coarse_tracks.items() if len(track.points) >= 3]
    prior_spacing = scene_spacing(coarse_tracks, prior_ids, frames)
    ghost_rows: list[dict] = []
    ghost_stats = {
        "enabled": False,
        "unmatched_birth_policy": "no birth prior when static-anchor priors are disabled",
    }
    if not args.disable_static_ghost_memory:
        ghost_rows, ghost_stats = build_static_anchor_ghost_priors(
            coarse_tracks, frames, prior_spacing, args.static_ghost_radius_scale,
        )
    if args.keyframe_joint_optimization:
        keyframe_started = time.perf_counter()
        history, keyframe_rows, keyframe_stats, keyframes, label_counts, composition_stats = run_keyframe_joint_motion_labeling(
            clouds, frames, coarse_tracks, args, out,
        )
        write_csv(out / "temporal_3d_tracks.csv", track_rows(coarse_tracks))
        save_binary_outputs(out, coarse_tracks, [], [], ghost_rows)
        initial_point_labels = export_initial_point_labels(
            root, out, clouds[frames[0]], coarse_tracks,
        )

        debug = out / "visual_debug"
        projection = fixed_world_projection(clouds, calibrations)
        if not args.no_visual_debug:
            debug.mkdir(parents=True, exist_ok=True)
            render_depth_geometry_debug(clouds, frames, calibrations, debug, args.debug_fps, projection)
            render_hungarian_3d(
                clouds, frames, coarse_tracks, coarse_match_rows, coarse_pair_stats,
                debug, args.debug_fps, projection,
            )
            render_rgb_track_videos(
                root, frames, calibrations, coarse_tracks, debug, args.debug_fps, labelled=True,
            )
            render_rigid_tracks_3d(frames, coarse_tracks, debug, args.debug_fps, projection)

        gt_audit = None
        if args.gt_audit:
            gt_audit = offline_gt_label_audit(
                root, out, clouds[frames[0]], coarse_tracks, args.gt_moving_part,
            )
        summary = {
            "status": "complete",
            "stage": "keyframe joint matching + transform-field + static/dynamic optimization",
            "ground_truth_used_as_algorithm_input": False,
            "appearance_used_by_estimator": False,
            "dynamic_discovery_enabled": True,
            "dynamic_label_contract": (
                "0.0=confident static, 0.5=uncertain, 1.0=confident dynamic; "
                "labels come from optimized transform evidence, not hard temporal recurrence"
            ),
            "parameters": parameters,
            "selected_frames": frames,
            "keyframes": keyframes,
            "keyframe_pairs": [[row["frame_a"], row["frame_b"]] for row in keyframe_stats],
            "num_cameras": len(calibrations),
            "coarse_static_lattice_voxel_size": clouds[frames[0]].quality["voxel_size"],
            "final_static_tracks": label_counts["STATIC"],
            "final_dynamic_tracks": label_counts["MOVING_1"],
            "final_uncertain_tracks": label_counts["AMBIGUOUS"],
            "static_anchor_ghost_priors": ghost_stats,
            "keyframe_joint_optimization_recap": keyframe_joint_loss_recap(),
            "keyframe_joint_pair_stats": keyframe_stats,
            "keyframe_joint_composition": composition_stats,
            "keyframe_joint_history_rows": len(history),
            "keyframe_joint_point_rows": len(keyframe_rows),
            "initial_point_cloud_labels": initial_point_labels,
            "offline_gt_audit": gt_audit,
            "disabled_stages": [
                "hard static fragment recurrence recovery",
                "full-resolution residual graph matching",
                "post-contrastive hard static overwrite",
            ],
            "elapsed_seconds": time.perf_counter() - keyframe_started,
        }
        (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
        print(
            "[KEYFRAME JOINT LABELS] "
            f"STATIC={summary['final_static_tracks']} "
            f"DYNAMIC={summary['final_dynamic_tracks']} "
            f"UNCERTAIN={summary['final_uncertain_tracks']}"
        )
        print(f"[COMPLETE] {out}")
        return

    recovery_rows, recovery_stats = recover_static_fragments_from_anchors(
        coarse_tracks, prior_spacing,
    )
    ghost_rows.extend(recovery_rows)
    temporal_recovery_rows, temporal_recovery_stats = recover_static_from_temporal_reappearance(
        clouds, frames, coarse_tracks, prior_spacing,
    )
    write_csv(out / "static_temporal_recovery.csv", temporal_recovery_rows, [
        "track_id", "first_frame", "last_frame", "track_length",
        "temporal_occurrences", "reappearance_observations",
        "local_static_support_p25", "mean_reappearance_distance",
        "radius", "seed_strength",
    ])
    write_csv(out / "temporal_3d_tracks.csv", track_rows(coarse_tracks))
    print(
        "[STATIC PHASE FROZEN] "
        f"trusted={sum(trusted_static_track(track) for track in coarse_tracks.values())} "
        f"ghost_reactivations={ghost_stats.get('ghost_reactivations', 0)}; "
        f"recovered_fragments={recovery_stats['recovered_static_tracks']}; "
        f"temporal_reappearances={temporal_recovery_stats['recovered_static_tracks']}; "
        f"elapsed={time.perf_counter() - phase_a_started:.1f}s; "
        "all remaining samples start UNCERTAIN"
    )

    # Terminal contract: ghosts protect reappearing STATIC material; all
    # incompatible, unmatched, or otherwise unresolved tracks stay UNCERTAIN.
    apply_static_anchor_only_labels(coarse_tracks)
    write_csv(out / "temporal_3d_tracks.csv", track_rows(coarse_tracks))
    save_binary_outputs(out, coarse_tracks, [], [], ghost_rows)
    initial_point_labels = export_initial_point_labels(
        root, out, clouds[frames[0]], coarse_tracks,
    )

    debug = out / "visual_debug"
    projection = fixed_world_projection(clouds, calibrations)
    if not args.no_visual_debug:
        debug.mkdir(parents=True, exist_ok=True)
        render_depth_geometry_debug(clouds, frames, calibrations, debug, args.debug_fps, projection)
        render_hungarian_3d(
            clouds, frames, coarse_tracks, coarse_match_rows, coarse_pair_stats,
            debug, args.debug_fps, projection,
        )
        render_rgb_track_videos(
            root, frames, calibrations, coarse_tracks, debug, args.debug_fps, labelled=True,
        )
        render_rigid_tracks_3d(frames, coarse_tracks, debug, args.debug_fps, projection)

    gt_audit = None
    if args.gt_audit:
        gt_audit = offline_gt_label_audit(
            root, out, clouds[frames[0]], coarse_tracks, args.gt_moving_part,
        )
    labels = [track.label for track in coarse_tracks.values()]
    summary = {
        "status": "complete",
        "stage": "trusted STATIC anchors plus spatial and whole-sequence temporal recovery",
        "ground_truth_used_as_algorithm_input": False,
        "appearance_used_by_estimator": False,
        "dynamic_discovery_enabled": False,
        "dynamic_label_contract": (
            "no point is labelled dynamic; no-ghost/incompatible observations remain uncertain"
        ),
        "parameters": parameters,
        "selected_frames": frames,
        "num_cameras": len(calibrations),
        "coarse_static_lattice_voxel_size": clouds[frames[0]].quality["voxel_size"],
        "trusted_static_tracks": labels.count("STATIC"),
        "uncertain_tracks": labels.count("AMBIGUOUS"),
        "dynamic_tracks": labels.count("MOVING_1"),
        "static_anchor_ghost_priors": ghost_stats,
        "static_fragment_recovery": recovery_stats,
        "temporal_static_recovery": temporal_recovery_stats,
        "initial_point_cloud_labels": initial_point_labels,
        "offline_gt_audit": gt_audit,
        "disabled_stages": [
            "full-resolution residual graph matching",
            "dynamic-anchor promotion",
            "binary contrastive optimization",
            "post-optimization relabeling",
        ],
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(
        "[STATIC/GHOST/UNCERTAIN LABELS] "
        f"STATIC={summary['trusted_static_tracks']} "
        f"UNCERTAIN={summary['uncertain_tracks']} DYNAMIC={summary['dynamic_tracks']}"
    )
    print(f"[COMPLETE] {out}")
    return

    # Phases B-E: fine residual graph matching and binary uncertainty discovery.
    residual_started = time.perf_counter()
    residuals = build_full_resolution_residuals(
        clouds, frames, coarse_tracks, out, args.residual_fine_tolerance,
        args.residual_target_points, args.save_full_resolution_residual,
    )
    (
        residual_tracks, node_probabilities, residual_match_rows,
        residual_quality_rows, contrastive_history, pipeline_metadata,
    ) = run_static_dynamic_residual_pipeline(
        clouds, coarse_tracks, residuals, frames, args,
    )
    final_edges = pipeline_metadata.pop("final_edges")
    final_pair_rows = pipeline_metadata.pop("final_pair_rows")
    group_names = transfer_residual_probabilities_to_coarse_tracks(
        coarse_tracks, residuals, node_probabilities,
        args.static_probability_threshold, args.dynamic_probability_threshold,
    )

    write_csv(out / "residual_graph_match_quality.csv", residual_quality_rows, [
        "outer_iteration", "frame_a", "frame_b", "source_residual_points",
        "target_residual_points", "candidate_nodes", "candidate_compatibility_edges",
        "projected_matches", "accepted_matches", "unmatched_source", "unmatched_target",
        "median_edge_deformation", "p90_edge_deformation", "median_graph_support",
        "median_assignment_margin", "skip_confirmed", "skip_conflict", "new_dynamic_anchors",
    ])
    write_csv(out / "residual_graph_matches.csv", residual_match_rows, [
        "outer_iteration", "frame_a", "frame_b", "source_id", "target_id",
        "graph_score", "edge_deformation_median", "edge_deformation_p90",
        "compatibility_support", "support_scales", "assignment_margin",
        "skip_consistency", "se3_residual", "se3_inlier", "motion_magnitude",
        "dynamic_anchor_accepted", "track_id",
    ])
    save_binary_outputs(
        out, coarse_tracks, final_edges, contrastive_history, ghost_rows,
    )
    initial_point_labels = export_initial_point_labels(
        root, out, clouds[frames[0]], coarse_tracks,
    )

    final_outer = max((row["outer_iteration"] for row in residual_quality_rows), default=0)
    final_quality = [
        {key: value for key, value in row.items() if key != "outer_iteration"}
        for row in residual_quality_rows if row["outer_iteration"] == final_outer
    ]
    debug = out / "visual_debug"
    projection = fixed_world_projection(clouds, calibrations)
    if not args.no_visual_debug:
        debug.mkdir(parents=True, exist_ok=True)
        render_depth_geometry_debug(clouds, frames, calibrations, debug, args.debug_fps, projection)
        render_hungarian_3d(
            clouds, frames, coarse_tracks, coarse_match_rows, coarse_pair_stats,
            debug, args.debug_fps, projection,
        )
        render_residual_graph_matching_3d(
            residuals, final_pair_rows, final_quality, debug, args.debug_fps, projection,
        )
        render_residual_graph_matching_all_cameras(
            root, residuals, final_pair_rows, calibrations, debug, args.debug_fps,
        )
        render_rgb_track_videos(
            root, frames, calibrations, coarse_tracks, debug, args.debug_fps, labelled=True,
        )
        render_rigid_tracks_3d(frames, coarse_tracks, debug, args.debug_fps, projection)

    # This optional read happens only after estimation, labels, and handoff are frozen.
    gt_audit = None
    if args.gt_audit:
        gt_audit = offline_gt_label_audit(
            root, out, clouds[frames[0]], coarse_tracks, args.gt_moving_part,
        )

    final_labels = [track.label for track in coarse_tracks.values()]
    final_history = contrastive_history[-1] if contrastive_history else {}
    summary = {
        "status": "complete",
        "stage": "binary STATIC/DYNAMIC/UNCERTAIN discovery only",
        "ground_truth_used_as_algorithm_input": False,
        "appearance_used_by_estimator": False,
        "rgb_usage": "visualization only" if not args.no_visual_debug else "not opened",
        "estimator_inputs": [
            "depth", "fixed_camera_intrinsics", "fixed_camera_extrinsics",
            "trusted_static_ghosts", "multi_scale_relative_3d_distances",
        ],
        "parameters": parameters,
        "selected_frames": frames,
        "num_cameras": len(calibrations),
        "coarse_static_lattice_voxel_size": clouds[frames[0]].quality["voxel_size"],
        "residual_fine_tolerance": args.residual_fine_tolerance,
        "trusted_static_tracks": sum(trusted_static_track(track) for track in coarse_tracks.values()),
        "residual_points_per_frame": {
            str(frame): len(residuals[frame].xyz) for frame in frames
        },
        "residual_tracks_final": len(residual_tracks),
        "residual_accepted_matches_final": sum(
            row["accepted_matches"] for row in final_quality
        ),
        "final_static_tracks": final_labels.count("STATIC"),
        "final_dynamic_tracks": final_labels.count("MOVING_1"),
        "final_uncertain_tracks": final_labels.count("AMBIGUOUS"),
        "static_probability_threshold": args.static_probability_threshold,
        "dynamic_probability_threshold": args.dynamic_probability_threshold,
        "final_contrastive_state": final_history,
        "static_anchor_ghost_priors": ghost_stats,
        "residual_pipeline": pipeline_metadata,
        "initial_point_cloud_labels": initial_point_labels,
        "offline_gt_audit": gt_audit,
        "removed_effective_postprocessing": [
            "hard short-fragment kNN label copying",
            "post-contrastive final temporal label overwrite",
            "K-part clustering and rigid-group relabeling",
        ],
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(
        "[FINAL BINARY LABELS] "
        f"STATIC={summary['final_static_tracks']} DYNAMIC={summary['final_dynamic_tracks']} "
        f"UNCERTAIN={summary['final_uncertain_tracks']} "
        f"residual_elapsed={time.perf_counter() - residual_started:.1f}s"
    )
    print(f"[COMPLETE] {out}")


if __name__ == "__main__":
    main()
