#!/usr/bin/env python3
"""
Rigid-part discovery from per-frame multiview COLMAP triangulations.

Pipeline
========
1. Select N video frames (default: all available frames).
2. Run official COLMAP SIFT feature extraction once on those images.
3. Match ONLY cameras belonging to the SAME frame.
4. Triangulate each frame's matches into independent 3-D points using the
   trusted fixed camera poses from ROOT/colmap/sparse/0.
5. Match the resulting 3-D points only between consecutive selected frames.
   The temporal matcher is geometry-first: RootSIFT appearance is used as a
   candidate cue, but the decisive compatibility term is preservation of
   relative distances to neighboring matched 3-D points.
6. Build 3-D point tracks through time.
7. Learn K+1 rigid-part labels (STATIC + K moving parts) from a contrastive
   objective defined on the temporal evolution of pairwise 3-D distances:

       d_ij(t) = ||X_i(t) - X_j(t)||

   Points belonging to the same rigid part should keep d_ij(t) constant.
   Large temporal variation of d_ij(t) is negative evidence that the two
   tracks belong to the same rigid part.
8. Fit one generic SE(3) transform per anonymous cluster and frame. The cluster
   with the smallest rotation-plus-translation identity score is named STATIC.
9. Render synchronized 8-camera debug videos with temporal 3-D matches and
   all static/moving triangulated points.
10. Optional --depth-gt: sample ROOT/depth_npy at every feature observation,
    backproject depth with the trusted camera calibration, fuse across views,
    and compare the triangulated motion / relative-distance evolution with
    the depth-derived 3-D motion.

No pycolmap. No learned matcher. No depth is used for the actual labels.
Depth is evaluation-only when --depth-gt is supplied.

Example
=======
python rigid_hungarian_relative_distance.py \
    /home/samuelemara/Joint3DGS/RigEuler3DGS/input/USB_100109 \
    --colmap /usr/local/bin/colmap \
    --num-moving-parts 1 \
    --force

Use only the first 30 frames:
python rigid_hungarian_relative_distance.py ROOT --num-frames 30 --force

Evaluate the discovered motion against depth:
python rigid_hungarian_relative_distance.py ROOT --depth-gt --force
"""

import argparse
import csv
import json
import math
import os
import shutil
import sqlite3
import subprocess
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist

try:
    import torch
except Exception as exc:
    raise RuntimeError(
        "This script needs PyTorch only for the small contrastive label optimization. "
        "Run it from the rigeuler3dgs environment."
    ) from exc


# =============================================================================
# Compact algorithm constants. These are deliberately kept here instead of
# exposing dozens of CLI flags.
# =============================================================================

MAX_IMAGE_ID = 2147483647

# 3-D temporal matching.
TEMPORAL_SIGNATURE_K = 8
TEMPORAL_REFINEMENT_ROUNDS = 5
TEMPORAL_MIN_REL_SUPPORT = 3
TEMPORAL_RELATIONAL_FRACTION = 0.50
TEMPORAL_INITIAL_WEIGHTS = (0.75, 0.20, 0.05)  # signature, descriptor, quality
TEMPORAL_REFINED_WEIGHTS = (0.70, 0.20, 0.10)  # relational, signature, descriptor

# Relative-distance graph for rigid-part discovery.
RIGID_GRAPH_K = 12
RIGID_MIN_COMMON_FRAMES = 4
RIGID_MAD_WEIGHT = 1.0
CONTRASTIVE_ITERS = 600
CONTRASTIVE_LR = 0.05
CONTRASTIVE_CONFIDENCE = 0.60

# COLMAP geometry filtering.
MAX_REPROJ_ERROR_PX = 2.0
MIN_TRI_ANGLE_DEG = 1.0

# Video debug.
DEFAULT_DEBUG_FPS = 10.0

CAMERA_MODEL_NAME_TO_ID = {
    "SIMPLE_PINHOLE": 0,
    "PINHOLE": 1,
    "SIMPLE_RADIAL": 2,
    "RADIAL": 3,
    "OPENCV": 4,
    "OPENCV_FISHEYE": 5,
    "FULL_OPENCV": 6,
    "FOV": 7,
    "SIMPLE_RADIAL_FISHEYE": 8,
    "RADIAL_FISHEYE": 9,
    "THIN_PRISM_FISHEYE": 10,
}


# =============================================================================
# COLMAP CLI
# =============================================================================

class ColmapCLI:
    def __init__(self, executable: str, log_dir: Path):
        exe = shutil.which(executable) if os.path.sep not in executable else executable
        if exe is None or not Path(exe).exists():
            raise RuntimeError(f"COLMAP executable not found: {executable}")
        self.exe = str(exe)
        self.log_dir = log_dir
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self._help = {}

        p = subprocess.run(
            [self.exe, "-h"],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
        )
        first = (p.stdout or "COLMAP").splitlines()[0]
        print(f"[COLMAP] executable: {self.exe}")
        print(f"[COLMAP] {first}")

    def help(self, command: str) -> str:
        if command not in self._help:
            p = subprocess.run(
                [self.exe, command, "-h"],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                check=False,
            )
            self._help[command] = p.stdout or ""
        return self._help[command]

    def run(self, command: str, args, log_name: str):
        cmd = [self.exe, command] + [str(x) for x in args]
        log_path = self.log_dir / log_name
        print("\n[COLMAP]", " ".join(cmd))
        with open(log_path, "w") as log:
            p = subprocess.run(
                cmd,
                stdout=log,
                stderr=subprocess.STDOUT,
                text=True,
                check=False,
            )
        if p.returncode != 0:
            tail = ""
            try:
                tail = "\n".join(log_path.read_text(errors="replace").splitlines()[-80:])
            except Exception:
                pass
            raise RuntimeError(
                f"COLMAP command failed ({p.returncode}): {' '.join(cmd)}\n"
                f"Log: {log_path}\n\n{tail}"
            )
        return log_path


# =============================================================================
# Dataset / calibrated model helpers
# =============================================================================

def frame_number(name: str) -> int:
    return int(Path(name).stem.split("_")[-1])


def camera_name_from_image_name(name: str) -> str:
    for part in Path(name).parts:
        if part.startswith("cam_"):
            return part
    raise RuntimeError(f"Cannot find cam_XXX in image name: {name}")


def find_rgb_parent(root: Path) -> Path:
    for name in ("rgb", "video_rgb"):
        p = root / name
        if p.is_dir() and any(x.is_dir() and x.name.startswith("cam_") for x in p.iterdir()):
            return p
    raise FileNotFoundError("Could not find ROOT/rgb/cam_* or ROOT/video_rgb/cam_*")


def find_colmap_model(root: Path, explicit: str | None) -> Path:
    if explicit:
        p = Path(explicit)
        if not p.exists():
            raise FileNotFoundError(p)
        return p
    candidates = [
        root / "colmap" / "sparse" / "0",
        root / "colmap" / "sparse",
        root / "colmap",
        root / "sparse" / "0",
        root / "sparse",
    ]
    for p in candidates:
        if (
            ((p / "cameras.bin").exists() and (p / "images.bin").exists())
            or ((p / "cameras.txt").exists() and (p / "images.txt").exists())
        ):
            return p
    raise FileNotFoundError("Could not locate calibrated COLMAP sparse model")


def available_frames(image_path: Path):
    cams = sorted(p for p in image_path.iterdir() if p.is_dir() and p.name.startswith("cam_"))
    if not cams:
        raise RuntimeError(f"No cam_* directories under {image_path}")
    per_cam = []
    for cam in cams:
        fs = {
            frame_number(p.name)
            for p in cam.iterdir()
            if p.is_file() and p.stem.startswith("frame_")
        }
        per_cam.append(fs)
    common = sorted(set.intersection(*per_cam))
    if not common:
        raise RuntimeError("No frame index is present in all cameras")
    return [c.name for c in cams], common


@dataclass
class TextCamera:
    camera_id: int
    model: str
    width: int
    height: int
    params: list[float]


@dataclass
class TextImagePose:
    image_id: int
    qvec: tuple[float, float, float, float]
    tvec: tuple[float, float, float]
    camera_id: int
    name: str


def parse_cameras_txt(path: Path):
    out = {}
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        p = line.split()
        cid = int(p[0])
        out[cid] = TextCamera(
            cid, p[1], int(p[2]), int(p[3]), [float(v) for v in p[4:]]
        )
    return out


def parse_images_txt_poses(path: Path):
    lines = path.read_text().splitlines()
    out = {}
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        i += 1
        if not line or line.startswith("#"):
            continue
        p = line.split()
        if len(p) < 10:
            continue
        iid = int(p[0])
        out[iid] = TextImagePose(
            iid,
            tuple(float(v) for v in p[1:5]),
            tuple(float(v) for v in p[5:8]),
            int(p[8]),
            " ".join(p[9:]),
        )
        if i < len(lines):
            i += 1
    return out


def choose_original_pose_per_camera(images, cameras):
    grouped = defaultdict(list)
    for im in images.values():
        try:
            cam = camera_name_from_image_name(im.name)
        except RuntimeError:
            continue
        score = (0 if "frame_000000" in im.name else 1000) + len(im.name)
        grouped[cam].append((score, im))
    out = {}
    for cam, vals in grouped.items():
        vals.sort(key=lambda x: x[0])
        im = vals[0][1]
        out[cam] = (im, cameras[im.camera_id])
    return out


def convert_model_to_txt(cli: ColmapCLI, src: Path, dst: Path, log_name: str):
    if dst.exists():
        shutil.rmtree(dst)
    dst.mkdir(parents=True)
    cli.run(
        "model_converter",
        ["--input_path", src, "--output_path", dst, "--output_type", "TXT"],
        log_name,
    )


def convert_model_to_bin(cli: ColmapCLI, src: Path, dst: Path, log_name: str):
    if dst.exists():
        shutil.rmtree(dst)
    dst.mkdir(parents=True)
    cli.run(
        "model_converter",
        ["--input_path", src, "--output_path", dst, "--output_type", "BIN"],
        log_name,
    )


def qvec_to_rotmat(qvec):
    q = np.asarray(qvec, dtype=np.float64)
    q /= np.linalg.norm(q) + 1e-15
    w, x, y, z = q
    return np.array([
        [1 - 2*y*y - 2*z*z, 2*x*y - 2*z*w,     2*x*z + 2*y*w],
        [2*x*y + 2*z*w,     1 - 2*x*x - 2*z*z, 2*y*z - 2*x*w],
        [2*x*z - 2*y*w,     2*y*z + 2*x*w,     1 - 2*x*x - 2*y*y],
    ], dtype=np.float64)


def project_world(pose: TextImagePose, camera: TextCamera, X):
    X = np.asarray(X, dtype=np.float64)
    Xc = qvec_to_rotmat(pose.qvec) @ X + np.asarray(pose.tvec, dtype=np.float64)
    if Xc[2] <= 1e-10:
        return None
    xn, yn = Xc[0] / Xc[2], Xc[1] / Xc[2]
    if camera.model == "PINHOLE":
        fx, fy, cx, cy = camera.params[:4]
        return np.array([fx * xn + cx, fy * yn + cy], dtype=np.float64)
    if camera.model == "SIMPLE_PINHOLE":
        f, cx, cy = camera.params[:3]
        return np.array([f * xn + cx, f * yn + cy], dtype=np.float64)
    raise RuntimeError(f"Projection implemented only for PINHOLE/SIMPLE_PINHOLE, got {camera.model}")


def backproject_depth_world(pose: TextImagePose, camera: TextCamera, u, v, depth_z):
    if camera.model == "PINHOLE":
        fx, fy, cx, cy = camera.params[:4]
    elif camera.model == "SIMPLE_PINHOLE":
        f, cx, cy = camera.params[:3]
        fx = fy = f
    else:
        raise RuntimeError(f"Depth backprojection supports PINHOLE/SIMPLE_PINHOLE, got {camera.model}")
    Xc = np.array([
        (float(u) - cx) / fx * depth_z,
        (float(v) - cy) / fy * depth_z,
        depth_z,
    ], dtype=np.float64)
    Rcw = qvec_to_rotmat(pose.qvec)
    tcw = np.asarray(pose.tvec, dtype=np.float64)
    return Rcw.T @ (Xc - tcw)


# =============================================================================
# COLMAP SQLite helpers
# =============================================================================

def pair_id_to_image_ids(pair_id: int):
    image_id2 = pair_id % MAX_IMAGE_ID
    image_id1 = (pair_id - image_id2) // MAX_IMAGE_ID
    return int(image_id1), int(image_id2)


def blob_array(blob, dtype, shape):
    if blob is None:
        return np.empty(shape, dtype=dtype)
    return np.frombuffer(blob, dtype=dtype).reshape(shape)


def db_images(db_path: Path):
    con = sqlite3.connect(str(db_path))
    rows = con.execute("SELECT image_id, name, camera_id FROM images").fetchall()
    con.close()
    return {
        int(iid): {"name": name, "camera_id": int(cid)}
        for iid, name, cid in rows
    }


def db_keypoints(db_path: Path):
    con = sqlite3.connect(str(db_path))
    out = {}
    for iid, rows, cols, data in con.execute(
        "SELECT image_id, rows, cols, data FROM keypoints"
    ):
        a = blob_array(data, np.float32, (int(rows), int(cols)))
        out[int(iid)] = a[:, :2].copy()
    con.close()
    return out


def db_descriptors(db_path: Path):
    con = sqlite3.connect(str(db_path))
    out = {}
    for iid, rows, cols, data in con.execute(
        "SELECT image_id, rows, cols, data FROM descriptors"
    ):
        a = blob_array(data, np.uint8, (int(rows), int(cols))).astype(np.float32)
        n = np.linalg.norm(a, axis=1, keepdims=True)
        a /= np.maximum(n, 1e-12)
        out[int(iid)] = a
    con.close()
    return out


def db_matches(db_path: Path, table="matches"):
    con = sqlite3.connect(str(db_path))
    out = []
    for pair_id, rows, cols, data in con.execute(
        f"SELECT pair_id, rows, cols, data FROM {table}"
    ):
        if int(rows) <= 0 or data is None:
            continue
        m = blob_array(data, np.uint32, (int(rows), int(cols)))[:, :2].copy()
        ia, ib = pair_id_to_image_ids(int(pair_id))
        out.append((ia, ib, m))
    con.close()
    return out


def count_matches(db_path: Path, table="matches"):
    con = sqlite3.connect(str(db_path))
    pairs, matches = con.execute(
        f"SELECT COUNT(*), COALESCE(SUM(rows),0) FROM {table} WHERE rows>0"
    ).fetchone()
    con.close()
    return int(pairs), int(matches)


def update_database_cameras_from_original(db_path: Path, pose_source_by_cam):
    con = sqlite3.connect(str(db_path))
    rows = con.execute("SELECT name, camera_id FROM images").fetchall()
    folder_to_ids = defaultdict(set)
    for name, cid in rows:
        folder_to_ids[camera_name_from_image_name(name)].add(int(cid))

    for cam_name, ids in folder_to_ids.items():
        if len(ids) != 1:
            raise RuntimeError(f"Expected one DB camera for {cam_name}, got {ids}")
        if cam_name not in pose_source_by_cam:
            raise RuntimeError(f"No trusted calibration for {cam_name}")
        db_cid = next(iter(ids))
        _, c = pose_source_by_cam[cam_name]
        if c.model not in CAMERA_MODEL_NAME_TO_ID:
            raise RuntimeError(f"Unsupported COLMAP camera model mapping: {c.model}")
        con.execute(
            "UPDATE cameras SET model=?, width=?, height=?, params=?, prior_focal_length=1 WHERE camera_id=?",
            (
                CAMERA_MODEL_NAME_TO_ID[c.model],
                c.width,
                c.height,
                np.asarray(c.params, dtype=np.float64).tobytes(),
                db_cid,
            ),
        )
    con.commit()
    con.close()


def build_expanded_known_pose_model(master_db: Path, pose_source_by_cam, dst: Path):
    if dst.exists():
        shutil.rmtree(dst)
    dst.mkdir(parents=True)

    images = db_images(master_db)
    db_cam_to_original = {}
    for rec in images.values():
        cam = camera_name_from_image_name(rec["name"])
        _, original = pose_source_by_cam[cam]
        db_cam_to_original.setdefault(rec["camera_id"], original)

    with open(dst / "cameras.txt", "w") as f:
        f.write("# CAMERA_ID MODEL WIDTH HEIGHT PARAMS[]\n")
        for cid in sorted(db_cam_to_original):
            c = db_cam_to_original[cid]
            f.write(
                f"{cid} {c.model} {c.width} {c.height} "
                + " ".join(f"{v:.17g}" for v in c.params)
                + "\n"
            )

    with open(dst / "images.txt", "w") as f:
        for iid in sorted(images):
            rec = images[iid]
            cam = camera_name_from_image_name(rec["name"])
            pose, _ = pose_source_by_cam[cam]
            f.write(
                f"{iid} "
                + " ".join(f"{v:.17g}" for v in pose.qvec)
                + " "
                + " ".join(f"{v:.17g}" for v in pose.tvec)
                + f" {rec['camera_id']} {rec['name']}\n\n"
            )
    (dst / "points3D.txt").write_text("# empty\n")


# =============================================================================
# Pair lists and raw-match import
# =============================================================================

def write_image_list(image_path: Path, cameras, selected_frames, path: Path):
    names = []
    for frame in selected_frames:
        for cam in cameras:
            found = None
            for ext in (".png", ".jpg", ".jpeg", ".PNG", ".JPG", ".JPEG"):
                p = image_path / cam / f"frame_{frame:06d}{ext}"
                if p.exists():
                    found = p.relative_to(image_path).as_posix()
                    break
            if found is None:
                raise FileNotFoundError(f"Missing {cam}/frame_{frame:06d}.*")
            names.append(found)
    path.write_text("\n".join(names) + "\n")
    return names


def write_same_frame_multiview_pairs(images, selected_frames, path: Path):
    selected = set(selected_frames)
    by_frame = defaultdict(list)
    for iid, rec in images.items():
        f = frame_number(rec["name"])
        if f in selected:
            by_frame[f].append((camera_name_from_image_name(rec["name"]), iid, rec["name"]))
    n = 0
    with open(path, "w") as fp:
        for frame in selected_frames:
            views = sorted(by_frame[frame])
            for i in range(len(views)):
                for j in range(i + 1, len(views)):
                    fp.write(f"{views[i][2]} {views[j][2]}\n")
                    n += 1
    print(f"[PAIRS] same-frame multiview: {n} -> {path}")


def read_mask_value(root: Path, cam: str, frame: int, x: float, y: float):
    mdir = root / "mask" / cam
    if not mdir.is_dir():
        return 1.0
    p = None
    for ext in (".png", ".jpg", ".jpeg"):
        q = mdir / f"frame_{frame:06d}{ext}"
        if q.exists():
            p = q
            break
    if p is None:
        return 1.0
    im = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
    if im is None:
        return 1.0
    xi = int(np.clip(round(x), 0, im.shape[1] - 1))
    yi = int(np.clip(round(y), 0, im.shape[0] - 1))
    return float(im[yi, xi]) / 255.0


def write_raw_matches_as_inliers(
    root: Path,
    raw_db: Path,
    master_db: Path,
    out_path: Path,
    use_mask: bool,
):
    images = db_images(raw_db)
    keypoints = db_keypoints(master_db)
    kept_pairs = 0
    kept_matches = 0
    raw_matches = 0

    with open(out_path, "w") as f:
        for ia, ib, matches in db_matches(raw_db, "matches"):
            raw_matches += len(matches)
            ra, rb = images[ia], images[ib]
            frame_a = frame_number(ra["name"])
            frame_b = frame_number(rb["name"])
            if frame_a != frame_b:
                continue
            cam_a = camera_name_from_image_name(ra["name"])
            cam_b = camera_name_from_image_name(rb["name"])
            keep = []
            for ka, kb in matches:
                ka, kb = int(ka), int(kb)
                if use_mask:
                    ua = keypoints[ia][ka]
                    ub = keypoints[ib][kb]
                    if read_mask_value(root, cam_a, frame_a, ua[0], ua[1]) < 0.5:
                        continue
                    if read_mask_value(root, cam_b, frame_b, ub[0], ub[1]) < 0.5:
                        continue
                keep.append((ka, kb))
            if not keep:
                continue
            f.write(f"{ra['name']} {rb['name']}\n")
            for ka, kb in keep:
                f.write(f"{ka} {kb}\n")
            f.write("\n")
            kept_pairs += 1
            kept_matches += len(keep)

    print(
        f"[RAW MATCHES] {raw_matches} COLMAP descriptor matches -> "
        f"{kept_matches} foreground candidates over {kept_pairs} pairs"
    )
    return kept_pairs, kept_matches


# =============================================================================
# Official known-pose triangulation
# =============================================================================

def run_known_pose_pipeline(
    cli: ColmapCLI,
    database_path: Path,
    image_path: Path,
    expanded_model: Path,
    output_root: Path,
):
    if output_root.exists():
        shutil.rmtree(output_root)
    output_root.mkdir(parents=True)

    triangulated = output_root / "triangulated"
    adjusted = output_root / "bundle_adjusted"
    filtered = output_root / "filtered"
    filtered_txt = output_root / "filtered_txt"
    triangulated.mkdir()

    h = cli.help("point_triangulator")
    args = [
        "--database_path", database_path,
        "--image_path", image_path,
        "--input_path", expanded_model,
        "--output_path", triangulated,
    ]
    if "--clear_points" in h:
        args += ["--clear_points", "1"]
    if "--refine_intrinsics" in h:
        args += ["--refine_intrinsics", "0"]
    if "Mapper.fix_existing_frames" in h:
        args += ["--Mapper.fix_existing_frames", "1"]
    if "Mapper.ba_refine_focal_length" in h:
        args += ["--Mapper.ba_refine_focal_length", "0"]
    if "Mapper.ba_refine_principal_point" in h:
        args += ["--Mapper.ba_refine_principal_point", "0"]
    if "Mapper.ba_refine_extra_params" in h:
        args += ["--Mapper.ba_refine_extra_params", "0"]
    if "Mapper.ba_refine_sensor_from_rig" in h:
        args += ["--Mapper.ba_refine_sensor_from_rig", "0"]
    if "Mapper.min_num_matches" in h:
        args += ["--Mapper.min_num_matches", "1"]
    if "Mapper.tri_ignore_two_view_tracks" in h:
        args += ["--Mapper.tri_ignore_two_view_tracks", "0"]
    if "Mapper.tri_min_angle" in h:
        args += ["--Mapper.tri_min_angle", MIN_TRI_ANGLE_DEG]
    if "Mapper.filter_max_reproj_error" in h:
        args += ["--Mapper.filter_max_reproj_error", MAX_REPROJ_ERROR_PX]
    if "Mapper.filter_min_tri_angle" in h:
        args += ["--Mapper.filter_min_tri_angle", MIN_TRI_ANGLE_DEG]
    cli.run("point_triangulator", args, "04_point_triangulator.log")

    adjusted.mkdir()
    h = cli.help("bundle_adjuster")
    args = ["--input_path", triangulated, "--output_path", adjusted]
    for opt in (
        "BundleAdjustment.refine_focal_length",
        "BundleAdjustment.refine_principal_point",
        "BundleAdjustment.refine_extra_params",
    ):
        if opt in h:
            args += [f"--{opt}", "0"]
    if "BundleAdjustment.refine_extrinsics" in h:
        args += ["--BundleAdjustment.refine_extrinsics", "0"]
    if "BundleAdjustment.refine_rig_from_world" in h:
        args += ["--BundleAdjustment.refine_rig_from_world", "0"]
    if "BundleAdjustment.refine_sensor_from_rig" in h:
        args += ["--BundleAdjustment.refine_sensor_from_rig", "0"]
    cli.run("bundle_adjuster", args, "05_bundle_adjuster.log")

    filtered.mkdir()
    h = cli.help("point_filtering")
    args = ["--input_path", adjusted, "--output_path", filtered]
    if "--min_track_len" in h:
        args += ["--min_track_len", "2"]
    if "--max_reproj_error" in h:
        args += ["--max_reproj_error", MAX_REPROJ_ERROR_PX]
    if "--min_tri_angle" in h:
        args += ["--min_tri_angle", MIN_TRI_ANGLE_DEG]
    cli.run("point_filtering", args, "06_point_filtering.log")

    convert_model_to_txt(cli, filtered, filtered_txt, "07_filtered_to_txt.log")
    return filtered_txt


# =============================================================================
# Read per-frame COLMAP 3-D points and aggregate their RootSIFT descriptors
# =============================================================================

@dataclass
class PointObservation:
    image_id: int
    kp_idx: int
    cam: str
    frame: int
    xy: np.ndarray


@dataclass
class MVPoint:
    point3d_id: int
    frame: int
    xyz: np.ndarray
    error: float
    observations: list[PointObservation] = field(default_factory=list)
    descriptor: np.ndarray | None = None


@dataclass
class TemporalTrack3D:
    track_id: int
    points: dict[int, MVPoint] = field(default_factory=dict)
    label_id: int = -1
    label: str = "AMBIGUOUS"
    confidence: float = 0.0


def parse_filtered_points(
    points3d_txt: Path,
    db_path: Path,
):
    images = db_images(db_path)
    keypoints = db_keypoints(db_path)
    descriptors = db_descriptors(db_path)
    by_frame = defaultdict(list)

    for line in points3d_txt.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        p = line.split()
        if len(p) < 10:
            continue
        pid = int(p[0])
        xyz = np.asarray([float(p[1]), float(p[2]), float(p[3])], np.float64)
        error = float(p[7])
        track_tokens = p[8:]
        if len(track_tokens) % 2 != 0:
            continue

        obs = []
        frame_set = set()
        descs = []
        for k in range(0, len(track_tokens), 2):
            iid = int(track_tokens[k])
            kp = int(track_tokens[k + 1])
            if iid not in images or iid not in keypoints:
                continue
            rec = images[iid]
            fr = frame_number(rec["name"])
            cam = camera_name_from_image_name(rec["name"])
            if kp >= len(keypoints[iid]):
                continue
            xy = keypoints[iid][kp].astype(np.float64)
            obs.append(PointObservation(iid, kp, cam, fr, xy))
            frame_set.add(fr)
            if iid in descriptors and kp < len(descriptors[iid]):
                descs.append(descriptors[iid][kp])

        # Because we match only cameras at the same frame, every reconstructed
        # point must belong to exactly one video frame.
        if len(frame_set) != 1 or len(obs) < 2:
            continue
        fr = next(iter(frame_set))

        desc = None
        if descs:
            desc = np.mean(np.stack(descs), axis=0)
            desc /= np.linalg.norm(desc) + 1e-12

        by_frame[fr].append(MVPoint(pid, fr, xyz, error, obs, desc))

    for fr in by_frame:
        by_frame[fr].sort(key=lambda x: x.point3d_id)
    return by_frame


def gate_reconstruction_points(points_by_frame, selected_frames, pose_source_by_cam, num_cameras):
    accepted = defaultdict(list)
    rejected = defaultdict(list)
    quality = {}
    for frame in selected_frames:
        points = list(points_by_frame.get(frame, []))
        finite_errors = np.asarray([p.error for p in points if np.isfinite(p.error)], np.float64)
        if finite_errors.size:
            reproj_limit = max(
                float(np.median(finite_errors)) + 3.0 * robust_mad(finite_errors),
                float(np.percentile(finite_errors, 90)),
            )
        else:
            reproj_limit = -np.inf

        for p in points:
            reasons = []
            support = len({o.cam for o in p.observations})
            if support < 2:
                reasons.append("support_cameras<2")
            if not np.all(np.isfinite(p.xyz)):
                reasons.append("nonfinite_xyz")
            if not np.isfinite(p.error):
                reasons.append("nonfinite_reprojection_error")
            elif p.error > reproj_limit:
                reasons.append("robust_reprojection_outlier")
            for o in p.observations:
                pose = pose_source_by_cam.get(o.cam)
                if pose is None:
                    reasons.append(f"missing_pose:{o.cam}")
                    continue
                Xc = qvec_to_rotmat(pose[0].qvec) @ p.xyz + np.asarray(pose[0].tvec)
                if not np.isfinite(Xc[2]) or Xc[2] <= 0:
                    reasons.append(f"nonpositive_depth:{o.cam}")
            if reasons:
                rejected[frame].append({
                    "point3d_id": p.point3d_id,
                    "xyz": " ".join(f"{v:.17g}" for v in p.xyz),
                    "support_cameras": support,
                    "reprojection_error": p.error,
                    "rejection_reason": ";".join(sorted(set(reasons))),
                })
            else:
                accepted[frame].append(p)

        lengths = np.asarray([len({o.cam for o in p.observations}) for p in points], np.float64)
        errors = np.asarray([p.error for p in points if np.isfinite(p.error)], np.float64)
        quality[frame] = {
            "frame": frame,
            "num_registered_images": num_cameras,
            "num_points3D": len(points),
            "num_observations": int(sum(len(p.observations) for p in points)),
            "mean_track_length": float(np.mean(lengths)) if lengths.size else 0.0,
            "median_track_length": float(np.median(lengths)) if lengths.size else 0.0,
            "mean_reprojection_error_px": float(np.mean(errors)) if errors.size else 0.0,
            "median_reprojection_error_px": float(np.median(errors)) if errors.size else 0.0,
            "p90_reprojection_error_px": float(np.percentile(errors, 90)) if errors.size else 0.0,
            "max_reprojection_error_px": float(np.max(errors)) if errors.size else 0.0,
            "num_points_support_2cams": int(np.sum(lengths == 2)),
            "num_points_support_3cams": int(np.sum(lengths == 3)),
            "num_points_support_4plus_cams": int(np.sum(lengths >= 4)),
            "num_points_rejected": len(rejected[frame]),
            "num_points_used_for_temporal_matching": len(accepted[frame]),
            "robust_reprojection_threshold_px": float(reproj_limit),
        }
        print(
            f"[MULTIVIEW] frame {frame:06d}: cameras={num_cameras} "
            f"triangulated_points={len(points)} "
            f"median_reproj={quality[frame]['median_reprojection_error_px']:.3f}px "
            f"accepted={len(accepted[frame])}"
        )
    return accepted, rejected, quality


def _read_colmap_image_blocks(path):
    lines = path.read_text().splitlines()
    blocks = {}
    i = 0
    while i < len(lines):
        header = lines[i]
        i += 1
        stripped = header.strip()
        if not stripped or stripped.startswith("#"):
            continue
        tokens = stripped.split()
        if len(tokens) < 10:
            continue
        points_line = lines[i] if i < len(lines) else ""
        i += 1
        blocks[int(tokens[0])] = (header, points_line, int(tokens[8]), " ".join(tokens[9:]))
    return blocks


def export_per_frame_colmap_models(cli, filtered_txt, selected_frames, out, quality, rejected):
    cameras_lines = [
        line for line in (filtered_txt / "cameras.txt").read_text().splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]
    image_blocks = _read_colmap_image_blocks(filtered_txt / "images.txt")
    point_lines = [
        line for line in (filtered_txt / "points3D.txt").read_text().splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]

    models_root = out / "colmap_models"
    models_root.mkdir(parents=True, exist_ok=True)
    for frame in selected_frames:
        frame_dir = models_root / f"frame_{frame:06d}"
        staging = frame_dir / "_input_txt"
        staging.mkdir(parents=True, exist_ok=True)
        selected = {
            iid: block for iid, block in image_blocks.items()
            if frame_number(block[3]) == frame
        }
        image_ids = set(selected)
        camera_ids = {block[2] for block in selected.values()}
        (staging / "cameras.txt").write_text(
            "# CAMERA_ID MODEL WIDTH HEIGHT PARAMS[]\n" +
            "\n".join(line for line in cameras_lines if int(line.split()[0]) in camera_ids) + "\n"
        )
        with open(staging / "images.txt", "w") as fp:
            fp.write("# IMAGE_ID QW QX QY QZ TX TY TZ CAMERA_ID NAME\n")
            for iid in sorted(selected):
                fp.write(selected[iid][0] + "\n" + selected[iid][1] + "\n")
        with open(staging / "points3D.txt", "w") as fp:
            fp.write("# POINT3D_ID X Y Z R G B ERROR TRACK[]\n")
            for line in point_lines:
                tokens = line.split()
                obs_ids = {int(tokens[k]) for k in range(8, len(tokens), 2)}
                if obs_ids and obs_ids.issubset(image_ids):
                    fp.write(line + "\n")

        bin_dir = frame_dir / "bin"
        txt_dir = frame_dir / "txt"
        convert_model_to_bin(cli, staging, bin_dir, f"frame_{frame:06d}_to_bin.log")
        convert_model_to_txt(cli, bin_dir, txt_dir, f"frame_{frame:06d}_to_txt.log")
        cli.run(
            "model_converter",
            [
                "--input_path", bin_dir,
                "--output_path", frame_dir / f"frame_{frame:06d}.ply",
                "--output_type", "PLY",
            ],
            f"frame_{frame:06d}_to_ply.log",
        )
        analyzer_path = frame_dir / "model_analyzer.txt"
        with open(analyzer_path, "w") as fp:
            subprocess.run(
                [cli.exe, "model_analyzer", "--path", str(bin_dir)],
                stdout=fp,
                stderr=subprocess.STDOUT,
                text=True,
                check=False,
            )
        (frame_dir / "quality.json").write_text(json.dumps(finite_json(quality[frame]), indent=2) + "\n")
        write_dict_csv(frame_dir / "rejected_points.csv", rejected[frame])
        shutil.rmtree(staging)

    rows = []
    for frame in selected_frames:
        q = quality[frame]
        rows.append({
            "frame": frame,
            "num_registered_images": q["num_registered_images"],
            "num_points3D": q["num_points3D"],
            "num_observations": q["num_observations"],
            "mean_track_length": q["mean_track_length"],
            "median_track_length": q["median_track_length"],
            "mean_reproj_error": q["mean_reprojection_error_px"],
            "median_reproj_error": q["median_reprojection_error_px"],
            "p90_reproj_error": q["p90_reprojection_error_px"],
            "max_reproj_error": q["max_reprojection_error_px"],
            "num_support_2": q["num_points_support_2cams"],
            "num_support_3": q["num_points_support_3cams"],
            "num_support_4plus": q["num_points_support_4plus_cams"],
            "num_points_rejected": q["num_points_rejected"],
            "num_points_used_for_temporal_matching": q["num_points_used_for_temporal_matching"],
        })
    write_dict_csv(out / "colmap_frame_quality.csv", rows)


# =============================================================================
# Geometry-first consecutive-frame matching
# =============================================================================

def local_distance_signatures(points: list[MVPoint], k: int):
    if not points:
        return np.zeros((0, k), np.float64), 1.0, np.zeros((0, 0), np.float64)
    X = np.stack([p.xyz for p in points])
    D = cdist(X, X)
    np.fill_diagonal(D, np.inf)
    finite = D[np.isfinite(D)]
    spacing = float(np.median(np.min(D, axis=1))) if len(points) > 1 else 1.0
    spacing = max(spacing, 1e-8)
    kk = min(k, max(0, len(points) - 1))
    sig = np.full((len(points), k), np.nan, np.float64)
    if kk > 0:
        vals = np.sort(D, axis=1)[:, :kk]
        sig[:, :kk] = vals
        if kk < k:
            sig[:, kk:] = vals[:, -1:]
    else:
        sig[:] = 0.0
    return sig, spacing, D


def descriptor_cost_matrix(a: list[MVPoint], b: list[MVPoint]):
    if not a or not b:
        return np.zeros((len(a), len(b)), np.float64)
    da = []
    db = []
    valid_a = []
    valid_b = []
    dim = None
    for p in a + b:
        if p.descriptor is not None:
            dim = len(p.descriptor)
            break
    if dim is None:
        return np.full((len(a), len(b)), 0.5, np.float64)
    for p in a:
        if p.descriptor is None:
            da.append(np.zeros(dim, np.float32)); valid_a.append(False)
        else:
            da.append(p.descriptor); valid_a.append(True)
    for p in b:
        if p.descriptor is None:
            db.append(np.zeros(dim, np.float32)); valid_b.append(False)
        else:
            db.append(p.descriptor); valid_b.append(True)
    da = np.stack(da); db = np.stack(db)
    C = 1.0 - np.clip(da @ db.T, -1.0, 1.0)
    for i, va in enumerate(valid_a):
        if not va:
            C[i, :] = 0.5
    for j, vb in enumerate(valid_b):
        if not vb:
            C[:, j] = 0.5
    return C.astype(np.float64)


def signature_cost_matrix(sa, sb):
    if len(sa) == 0 or len(sb) == 0:
        return np.zeros((len(sa), len(sb)), np.float64)
    A = sa[:, None, :]
    B = sb[None, :, :]
    den = np.maximum(np.maximum(np.abs(A), np.abs(B)), 1e-8)
    return np.nanmedian(np.abs(A - B) / den, axis=2)


def robust_mad(x):
    x = np.asarray(x, np.float64)
    if x.size == 0:
        return 0.0
    med = np.median(x)
    return float(1.4826 * np.median(np.abs(x - med)))


def finite_json(value):
    if isinstance(value, dict):
        return {key: finite_json(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [finite_json(item) for item in value]
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def point_quality_cost_matrix(a: list[MVPoint], b: list[MVPoint]):
    points = a + b
    errors = np.asarray([p.error for p in points if np.isfinite(p.error)], np.float64)
    scale = max(float(np.median(errors)) if errors.size else 1.0, 1e-8)

    def quality(p):
        reproj = min(float(p.error) / (3.0 * scale), 1.0) if np.isfinite(p.error) else 1.0
        support = len({o.cam for o in p.observations})
        return 0.75 * reproj + 0.25 * (1.0 - min(support / 4.0, 1.0))

    qa = np.asarray([quality(p) for p in a], np.float64)
    qb = np.asarray([quality(p) for p in b], np.float64)
    return 0.5 * (qa[:, None] + qb[None, :])


def adaptive_unmatched_cost(cost):
    if cost.size == 0:
        return 0.0
    best = np.concatenate([np.min(cost, axis=1), np.min(cost, axis=0)])
    med = float(np.median(best))
    mad = robust_mad(best)
    return max(med + 3.0 * mad, float(np.percentile(best, 90)) * 1.25, 1e-8)


def padded_assignment(cost, unmatched_cost):
    """Hungarian assignment with independent source/destination dummy slots."""
    na, nb = cost.shape
    size = na + nb
    large = max(float(np.max(cost)) if cost.size else 1.0, unmatched_cost, 1.0) * 1e6
    padded = np.full((size, size), large, np.float64)
    padded[:na, :nb] = cost
    padded[np.arange(na), nb + np.arange(na)] = unmatched_cost
    padded[na + np.arange(nb), np.arange(nb)] = unmatched_cost
    padded[na:, nb:] = 0.0
    rows, cols = linear_sum_assignment(padded)
    return {int(r): int(c) for r, c in zip(rows, cols)}


def relational_cost_matrix(Da, Db, mapping, d_scale):
    rel = np.ones((Da.shape[0], Db.shape[0]), np.float64)
    support = np.zeros(rel.shape, np.int32)
    for i in range(Da.shape[0]):
        for j in range(Db.shape[0]):
            residuals = []
            for k, l in mapping.items():
                if k == i or l == j:
                    continue
                den = max(float(Da[i, k]), float(Db[j, l]), d_scale, 1e-12)
                residuals.append(abs(float(Da[i, k]) - float(Db[j, l])) / den)
            if len(residuals) < TEMPORAL_MIN_REL_SUPPORT:
                continue
            residuals.sort()
            keep = max(
                TEMPORAL_MIN_REL_SUPPORT,
                int(math.ceil(TEMPORAL_RELATIONAL_FRACTION * len(residuals))),
            )
            kept = residuals[:keep]
            rel[i, j] = float(np.median(kept))
            support[i, j] = len(kept)
    return rel, support


def temporal_match_frame_pair(pa: list[MVPoint], pb: list[MVPoint]):
    if len(pa) < 2 or len(pb) < 2:
        return [], [], {"refinement_iterations": 0, "changed_assignments": []}

    sa, spacing_a, Da = local_distance_signatures(pa, TEMPORAL_SIGNATURE_K)
    sb, spacing_b, Db = local_distance_signatures(pb, TEMPORAL_SIGNATURE_K)
    d_scale = max(0.25 * 0.5 * (spacing_a + spacing_b), 1e-8)

    Cdesc = descriptor_cost_matrix(pa, pb)
    Csig = signature_cost_matrix(sa, sb)
    Cquality = point_quality_cost_matrix(pa, pb)
    wsig, wdesc, wquality = TEMPORAL_INITIAL_WEIGHTS
    C = wsig * Csig + wdesc * Cdesc + wquality * Cquality
    unmatched = adaptive_unmatched_cost(C)
    assignment = padded_assignment(C, unmatched)
    mapping = {i: j for i, j in assignment.items() if i < len(pa) and j < len(pb)}

    relational = np.ones_like(C)
    support = np.zeros_like(C, dtype=np.int32)
    changed = []
    for _ in range(TEMPORAL_REFINEMENT_ROUNDS):
        relational, support = relational_cost_matrix(Da, Db, mapping, d_scale)
        wrel, wsig, wdesc = TEMPORAL_REFINED_WEIGHTS
        C = wrel * relational + wsig * Csig + wdesc * Cdesc
        unmatched = adaptive_unmatched_cost(C)
        assignment = padded_assignment(C, unmatched)
        new_mapping = {
            i: j for i, j in assignment.items()
            if i < len(pa) and j < len(pb)
        }
        changed.append(sum(mapping.get(i) != new_mapping.get(i) for i in range(len(pa))))
        mapping = new_mapping
        if changed[-1] == 0:
            break

    # Report each selected assignment against the final support mapping.
    relational, support = relational_cost_matrix(Da, Db, mapping, d_scale)
    wrel, wsig, wdesc = TEMPORAL_REFINED_WEIGHTS
    C = wrel * relational + wsig * Csig + wdesc * Cdesc
    unmatched = adaptive_unmatched_cost(C)

    accepted = []
    diagnostics = []
    matched_destinations = set()
    for i in range(len(pa)):
        j = assignment.get(i, len(pb))
        if j < len(pb):
            row = C[i]
            second = float(np.partition(row, 1)[1]) if len(row) > 1 else unmatched
            margin = second - float(C[i, j])
            ok = support[i, j] >= TEMPORAL_MIN_REL_SUPPORT
            rec = {
                "i": i,
                "j": j,
                "total_cost": float(C[i, j]),
                "relative_cost": float(relational[i, j]),
                "signature_cost": float(Csig[i, j]),
                "descriptor_cost": float(Cdesc[i, j]),
                "relational_support": int(support[i, j]),
                "assignment_margin": margin,
                "accepted": bool(ok),
            }
            diagnostics.append(rec)
            if ok:
                accepted.append(rec)
                matched_destinations.add(j)
        else:
            diagnostics.append({
                "i": i,
                "j": -1,
                "total_cost": float(unmatched),
                "relative_cost": math.nan,
                "signature_cost": math.nan,
                "descriptor_cost": math.nan,
                "relational_support": 0,
                "assignment_margin": 0.0,
                "accepted": False,
            })
    for j in range(len(pb)):
        if j not in matched_destinations:
            diagnostics.append({
                "i": -1,
                "j": j,
                "total_cost": float(unmatched),
                "relative_cost": math.nan,
                "signature_cost": math.nan,
                "descriptor_cost": math.nan,
                "relational_support": 0,
                "assignment_margin": 0.0,
                "accepted": False,
            })
    return accepted, diagnostics, {
        "refinement_iterations": len(changed),
        "changed_assignments": changed,
        "unmatched_cost": float(unmatched),
    }


def print_colmap_frame_quality(frame, quality):
    q = quality[frame]
    print(
        f"[COLMAP FRAME] {frame:06d}\n"
        f"  registered_images={q['num_registered_images']}\n"
        f"  points3D={q['num_points3D']}\n"
        f"  observations={q['num_observations']}\n"
        f"  median_track_length={q['median_track_length']:.2f}\n"
        f"  median_reproj={q['median_reprojection_error_px']:.3f} px\n"
        f"  p90_reproj={q['p90_reprojection_error_px']:.3f} px\n"
        f"  support: 2cam={q['num_points_support_2cams']} "
        f"3cam={q['num_points_support_3cams']} "
        f"4+={q['num_points_support_4plus_cams']}\n"
        f"  accepted_for_temporal={q['num_points_used_for_temporal_matching']}"
    )


def build_temporal_3d_tracks(points_by_frame, selected_frames, frame_quality=None):
    tracks: dict[int, TemporalTrack3D] = {}
    point_to_track = {}
    next_tid = 0
    match_rows = []

    first = selected_frames[0]
    for p in points_by_frame.get(first, []):
        t = TemporalTrack3D(next_tid, {first: p})
        tracks[next_tid] = t
        point_to_track[(first, p.point3d_id)] = next_tid
        next_tid += 1

    for fa, fb in zip(selected_frames[:-1], selected_frames[1:]):
        pa = points_by_frame.get(fa, [])
        pb = points_by_frame.get(fb, [])
        if frame_quality is not None:
            print_colmap_frame_quality(fa, frame_quality)
            print_colmap_frame_quality(fb, frame_quality)
        matches, diagnostics, stats = temporal_match_frame_pair(pa, pb)
        used_b = set()

        for m in matches:
            a = pa[m["i"]]
            b = pb[m["j"]]
            tid = point_to_track.get((fa, a.point3d_id))
            if tid is None:
                tid = next_tid
                tracks[tid] = TemporalTrack3D(tid, {fa: a})
                point_to_track[(fa, a.point3d_id)] = tid
                next_tid += 1
            tracks[tid].points[fb] = b
            point_to_track[(fb, b.point3d_id)] = tid
            used_b.add(b.point3d_id)
            match_rows.append({
                "frame_a": fa,
                "frame_b": fb,
                "track_id": tid,
                "point3d_a": a.point3d_id,
                "point3d_b": b.point3d_id,
                "total_cost": m["total_cost"],
                "relative_cost": m["relative_cost"],
                "signature_cost": m["signature_cost"],
                "descriptor_cost": m["descriptor_cost"],
                "relational_support": m["relational_support"],
                "assignment_margin": m["assignment_margin"],
                "accepted": True,
            })

        for b in pb:
            if b.point3d_id in used_b:
                continue
            if (fb, b.point3d_id) not in point_to_track:
                tracks[next_tid] = TemporalTrack3D(next_tid, {fb: b})
                point_to_track[(fb, b.point3d_id)] = next_tid
                next_tid += 1

        for d in diagnostics:
            if d["accepted"]:
                continue
            a = pa[d["i"]] if d["i"] >= 0 else None
            b = pb[d["j"]] if d["j"] >= 0 else None
            tid = -1
            if a is not None:
                tid = point_to_track.get((fa, a.point3d_id), -1)
            elif b is not None:
                tid = point_to_track.get((fb, b.point3d_id), -1)
            match_rows.append({
                "frame_a": fa,
                "frame_b": fb,
                "track_id": tid,
                "point3d_a": a.point3d_id if a is not None else -1,
                "point3d_b": b.point3d_id if b is not None else -1,
                "total_cost": d["total_cost"],
                "relative_cost": d["relative_cost"],
                "signature_cost": d["signature_cost"],
                "descriptor_cost": d["descriptor_cost"],
                "relational_support": d["relational_support"],
                "assignment_margin": d["assignment_margin"],
                "accepted": False,
            })

        rel = [m["relative_cost"] for m in matches]
        sig = [m["signature_cost"] for m in matches]
        sup = [m["relational_support"] for m in matches]
        print(
            f"[HUNGARIAN] {fa:06d} -> {fb:06d}\n"
            f"  points: {len(pa)} -> {len(pb)}\n"
            f"  initial assignments={min(len(pa), len(pb))}\n"
            f"  accepted={len(matches)}\n"
            f"  rejected/unmatched={len(diagnostics) - len(matches)}\n"
            f"  median signature cost={np.median(sig) if sig else math.nan:.6g}\n"
            f"  median relative cost={np.median(rel) if rel else math.nan:.6g}\n"
            f"  median relational support={np.median(sup) if sup else 0:.1f}\n"
            f"  refinement iterations={stats['refinement_iterations']}\n"
            f"  changed assignments per iteration={stats['changed_assignments']}"
        )

    return tracks, match_rows


# =============================================================================
# Relative-distance rigidity graph + contrastive part labels
# =============================================================================

def build_rigidity_edges(tracks: dict[int, TemporalTrack3D], selected_frames):
    # Only tracks with enough temporal support can define a distance trajectory.
    valid_ids = [
        tid for tid, tr in tracks.items()
        if len(tr.points) >= RIGID_MIN_COMMON_FRAMES
    ]
    # For articulated segmentation the NEGATIVE evidence is exactly the
    # evolution of distances between points on DIFFERENT rigid parts. A pure
    # local-KNN graph can miss those cross-part pairs and can split one rigid
    # body arbitrarily. Therefore use the complete pair graph over temporally
    # supported tracks. The USB sequences are sparse enough for this to be
    # inexpensive and it is the most faithful implementation of the proposed
    # relative-distance contrast.
    pair_candidates = {
        (valid_ids[i], valid_ids[j])
        for i in range(len(valid_ids))
        for j in range(i + 1, len(valid_ids))
    }

    edges = []
    all_nn = []
    # Characteristic scene spacing from local tracked geometry.
    for frame in selected_frames:
        xyz = [tracks[tid].points[frame].xyz for tid in valid_ids if frame in tracks[tid].points]
        if len(xyz) > 1:
            D = cdist(np.stack(xyz), np.stack(xyz))
            np.fill_diagonal(D, np.inf)
            all_nn.extend(np.min(D, axis=1).tolist())
    scene_spacing = float(np.median(all_nn)) if all_nn else 1.0
    scene_spacing = max(scene_spacing, 1e-8)

    for a, b in sorted(pair_candidates):
        ta, tb = tracks[a], tracks[b]
        common = sorted(set(ta.points) & set(tb.points))
        if len(common) < RIGID_MIN_COMMON_FRAMES:
            continue
        ds = np.asarray([
            np.linalg.norm(ta.points[f].xyz - tb.points[f].xyz)
            for f in common
        ], np.float64)
        med = float(np.median(ds))
        denom = max(med, 0.25 * scene_spacing, 1e-8)
        frame_rank = {f: i for i, f in enumerate(selected_frames)}
        jumps = np.asarray([
            abs(ds[k + 1] - ds[k]) / denom
            for k in range(len(common) - 1)
            if frame_rank[common[k + 1]] == frame_rank[common[k]] + 1
        ], np.float64)
        if not jumps.size:
            continue
        rigidity_error = float(np.median(jumps) + RIGID_MAD_WEIGHT * robust_mad(jumps))
        rel_range = float(np.percentile(ds, 90) - np.percentile(ds, 10)) / denom
        edges.append({
            "track_a": a,
            "track_b": b,
            "common_frames": len(common),
            "median_distance": med,
            "p90_p10_variation": rel_range,
            "max_relative_distance_jump": float(np.max(jumps)),
            "rigidity_error": rigidity_error,
        })

    if not edges:
        return valid_ids, edges, scene_spacing, {"low_mode": math.nan, "high_mode": math.nan}

    # Two-mode fit on log rigidity error: low-error pairs are same-rigid positive
    # evidence; high-error pairs are contrastive negative evidence.
    z = np.log(np.asarray([e["rigidity_error"] for e in edges]) + 1e-8)
    c0, c1 = np.percentile(z, [25, 75])
    for _ in range(50):
        g0 = z[np.abs(z - c0) <= np.abs(z - c1)]
        g1 = z[np.abs(z - c1) < np.abs(z - c0)]
        nc0 = float(np.median(g0)) if len(g0) else c0
        nc1 = float(np.median(g1)) if len(g1) else c1
        if abs(nc0 - c0) + abs(nc1 - c1) < 1e-6:
            break
        c0, c1 = nc0, nc1
    low, high = sorted((c0, c1))
    low_group = z[np.abs(z - low) <= np.abs(z - high)]
    high_group = z[np.abs(z - high) < np.abs(z - low)]
    low_scale = max(robust_mad(low_group), 0.05)
    high_scale = max(robust_mad(high_group), 0.05)

    for e, zi in zip(edges, z):
        ll_low = -0.5 * ((float(zi) - low) / low_scale) ** 2 - math.log(low_scale)
        ll_high = -0.5 * ((float(zi) - high) / high_scale) ** 2 - math.log(high_scale)
        logit = float(np.clip(ll_low - ll_high, -60.0, 60.0))
        q = 1.0 / (1.0 + math.exp(-logit))
        temporal_weight = min(1.0, e["common_frames"] / 10.0)
        e["q_same"] = q
        e["weight"] = max(0.01, temporal_weight * abs(2.0 * q - 1.0))

    print(
        f"[RIGID GRAPH] {len(valid_ids)} temporally supported tracks, {len(edges)} edges | "
        f"log rigidity modes {low:.3f}, {high:.3f}"
    )
    return valid_ids, edges, scene_spacing, {
        "low_mode": float(low),
        "high_mode": float(high),
        "low_scale": float(low_scale),
        "high_scale": float(high_scale),
    }


def numpy_kmeans(X, k, labels0=None, iters=100):
    n = len(X)
    if n < k:
        raise RuntimeError(f"Cannot cluster {n} tracks into {k} parts")
    if labels0 is None:
        # deterministic farthest-point initialization
        centers = [0]
        while len(centers) < k:
            d = np.min(cdist(X, X[centers]), axis=1)
            centers.append(int(np.argmax(d)))
        C = X[centers].copy()
    else:
        C = np.stack([
            X[labels0 == j].mean(axis=0) if np.any(labels0 == j) else X[j % n]
            for j in range(k)
        ])
    labels = np.zeros(n, np.int32)
    for _ in range(iters):
        new = np.argmin(cdist(X, C), axis=1)
        if np.array_equal(new, labels):
            break
        labels = new
        for j in range(k):
            if np.any(labels == j):
                C[j] = X[labels == j].mean(axis=0)
    return labels


def spectral_initialization(valid_ids, edges, num_parts):
    n = len(valid_ids)
    idx = {tid: i for i, tid in enumerate(valid_ids)}
    A = np.zeros((n, n), np.float64)
    for e in edges:
        i, j = idx[e["track_a"]], idx[e["track_b"]]
        val = e["q_same"] * e["weight"]
        A[i, j] = max(A[i, j], val)
        A[j, i] = max(A[j, i], val)
    # self affinity avoids isolated numerical singularities
    np.fill_diagonal(A, 1.0)
    d = np.sum(A, axis=1)
    Dm = 1.0 / np.sqrt(np.maximum(d, 1e-12))
    N = Dm[:, None] * A * Dm[None, :]
    vals, vecs = np.linalg.eigh(N)
    U = vecs[:, np.argsort(vals)[-num_parts:]]
    U /= np.maximum(np.linalg.norm(U, axis=1, keepdims=True), 1e-12)
    return numpy_kmeans(U, num_parts)


def optimize_contrastive_labels(valid_ids, edges, num_parts):
    if len(valid_ids) < num_parts:
        raise RuntimeError("Not enough long 3-D tracks for requested number of rigid parts")
    if not edges:
        return {tid: (0, 1.0) for tid in valid_ids}, []

    idx = {tid: i for i, tid in enumerate(valid_ids)}
    init = spectral_initialization(valid_ids, edges, num_parts)

    device = torch.device("cpu")
    logits = torch.full((len(valid_ids), num_parts), -2.0, dtype=torch.float32, device=device)
    logits[torch.arange(len(valid_ids)), torch.from_numpy(init).long()] = 2.0
    logits = torch.nn.Parameter(logits)

    ei = torch.tensor([idx[e["track_a"]] for e in edges], dtype=torch.long)
    ej = torch.tensor([idx[e["track_b"]] for e in edges], dtype=torch.long)
    q = torch.tensor([e["q_same"] for e in edges], dtype=torch.float32)
    w = torch.tensor([e["weight"] for e in edges], dtype=torch.float32)
    w = w / (w.mean() + 1e-12)

    opt = torch.optim.Adam([logits], lr=CONTRASTIVE_LR)
    history = []
    for it in range(CONTRASTIVE_ITERS):
        opt.zero_grad(set_to_none=True)
        p = torch.softmax(logits, dim=1)
        same = torch.sum(p[ei] * p[ej], dim=1).clamp(1e-6, 1 - 1e-6)

        # Soft contrastive BCE: low relative-distance variation pulls two tracks
        # into the same rigid part; high variation pushes them apart.
        pair_loss = -q * torch.log(same) - (1 - q) * torch.log(1 - same)
        pair_loss = torch.sum(w * pair_loss) / torch.sum(w)

        # Small confidence term; do not force equal part sizes.
        entropy = -(p * torch.log(p.clamp_min(1e-8))).sum(dim=1).mean()
        loss = pair_loss + 0.01 * entropy
        loss.backward()
        opt.step()
        if it % 50 == 0 or it == CONTRASTIVE_ITERS - 1:
            history.append((it, float(loss.detach()), float(pair_loss.detach())))

    p = torch.softmax(logits.detach(), dim=1).cpu().numpy()
    result = {}
    for tid, row in zip(valid_ids, p):
        result[tid] = (int(np.argmax(row)), float(np.max(row)))
    return result, history


def fit_rigid_transform(source, target):
    source = np.asarray(source, np.float64)
    target = np.asarray(target, np.float64)
    if len(source) < 3:
        raise ValueError("At least three points are required for an SE(3) fit")
    cs = source.mean(axis=0)
    ct = target.mean(axis=0)
    U, _, Vt = np.linalg.svd((source - cs).T @ (target - ct))
    R = Vt.T @ U.T
    if np.linalg.det(R) < 0:
        Vt[-1] *= -1
        R = Vt.T @ U.T
    t = ct - R @ cs
    residual = np.linalg.norm((source @ R.T + t) - target, axis=1)
    return R, t, float(np.median(residual))


def rotation_angle(R):
    return float(math.acos(float(np.clip((np.trace(R) - 1.0) * 0.5, -1.0, 1.0))))


def fit_cluster_transforms(tracks, assignments, num_parts, selected_frames, scene_scale):
    rows = []
    scores = {}
    for cid in range(num_parts):
        tids = [tid for tid, (cluster, _) in assignments.items() if cluster == cid]
        if not tids:
            scores[cid] = math.inf
            continue
        reference = max(
            selected_frames,
            key=lambda f: sum(f in tracks[tid].points for tid in tids),
        )
        identities = []
        valid_transform_frames = 0
        for frame in selected_frames:
            common = [
                tid for tid in tids
                if reference in tracks[tid].points and frame in tracks[tid].points
            ]
            if len(common) < 3:
                continue
            source = np.stack([tracks[tid].points[reference].xyz for tid in common])
            target = np.stack([tracks[tid].points[frame].xyz for tid in common])
            R, t, residual = fit_rigid_transform(source, target)
            angle = rotation_angle(R)
            identity = angle + float(np.linalg.norm(t)) / max(scene_scale, 1e-8)
            identities.append(identity)
            valid_transform_frames += 1
            row = {
                "cluster_id": cid,
                "reference_frame": reference,
                "frame": frame,
                "num_correspondences": len(common),
                "rotation_angle_rad": angle,
                "translation_norm": float(np.linalg.norm(t)),
                "rigid_residual": residual,
                "identity_motion_score": identity,
                "tx": t[0], "ty": t[1], "tz": t[2],
            }
            for i in range(3):
                for j in range(3):
                    row[f"R{i}{j}"] = R[i, j]
            rows.append(row)
        scores[cid] = float(np.median(identities)) if valid_transform_frames >= 2 else math.inf
    return rows, scores


def label_tracks_rigid_parts(
    tracks, valid_ids, edges, num_moving_parts, selected_frames, scene_scale, contrast_stats
):
    num_parts = num_moving_parts + 1
    assignments, history = optimize_contrastive_labels(valid_ids, edges, num_parts)

    transform_rows, cluster_score = fit_cluster_transforms(
        tracks, assignments, num_parts, selected_frames, scene_scale
    )
    static_cluster = min(cluster_score, key=cluster_score.get)

    moving_clusters = [c for c in range(num_parts) if c != static_cluster]
    moving_clusters.sort()
    cluster_name = {static_cluster: "STATIC"}
    for k, c in enumerate(moving_clusters, 1):
        cluster_name[c] = f"MOVING_{k}"

    for tid, tr in tracks.items():
        if tid in assignments:
            cid, conf = assignments[tid]
            tr.label_id = cid
            tr.confidence = conf
            tr.label = cluster_name[cid] if conf >= CONTRASTIVE_CONFIDENCE else "AMBIGUOUS"

    # Propagate to short tracks only from rigidity-neighbor votes. No image-space
    # fill or forced hard completion.
    incident = defaultdict(list)
    for e in edges:
        incident[e["track_a"]].append((e["track_b"], e["q_same"] * e["weight"]))
        incident[e["track_b"]].append((e["track_a"], e["q_same"] * e["weight"]))
    for tid, tr in tracks.items():
        if tid in assignments:
            continue
        votes = defaultdict(float)
        total = 0.0
        for nb, ww in incident.get(tid, []):
            if nb in assignments:
                cid, conf = assignments[nb]
                votes[cid] += ww * conf
                total += ww * conf
        if total > 0 and votes:
            cid, val = max(votes.items(), key=lambda kv: kv[1])
            conf = val / total
            tr.label_id = cid
            tr.confidence = conf
            tr.label = cluster_name[cid] if conf >= CONTRASTIVE_CONFIDENCE else "AMBIGUOUS"

    final_loss = history[-1][1] if history else math.nan
    print(
        "[RIGID CONTRAST]\n"
        f"  supported tracks={len(valid_ids)}\n"
        f"  pair edges={len(edges)}\n"
        f"  low-rigidity mode={contrast_stats['low_mode']:.6g}\n"
        f"  high-rigidity mode={contrast_stats['high_mode']:.6g}\n"
        f"  contrastive final loss={final_loss:.6g}"
    )
    print("[PARTS]")
    for c in range(num_parts):
        ntracks = sum(cid == c for cid, _ in assignments.values())
        print(f"  cluster {c}: tracks={ntracks}, identity_motion_score={cluster_score[c]:.6g}")
    print(f"  STATIC = cluster {static_cluster}")
    counts = defaultdict(int)
    for tr in tracks.values():
        counts[tr.label] += 1
    print("[RIGID LABELS]", dict(sorted(counts.items())))
    return cluster_name, static_cluster, history, assignments, transform_rows, cluster_score


# =============================================================================
# Optional depth-derived GT motion evaluation
# =============================================================================

def find_depth_file(root: Path, cam: str, frame: int):
    for directory, extensions in (("depth_npy", (".npy",)), ("depth_png", (".png",))):
        d = root / directory / cam
        for extension in extensions:
            for name in (f"frame_{frame:06d}{extension}", f"{frame:06d}{extension}"):
                p = d / name
                if p.exists():
                    return p
    return None


def load_depth(path):
    if path.suffix.lower() == ".npy":
        return np.load(path)
    depth = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if depth is None:
        raise RuntimeError(f"Could not load depth image: {path}")
    return depth


def sample_depth_nearest(arr, x, y):
    a = np.asarray(arr)
    if a.ndim == 3:
        a = a[..., 0]
    xi = int(np.clip(round(float(x)), 0, a.shape[1] - 1))
    yi = int(np.clip(round(float(y)), 0, a.shape[0] - 1))
    z = float(a[yi, xi])
    return z if np.isfinite(z) and z > 0 else None


def estimate_depth_scale(root, points_by_frame, pose_source_by_cam):
    ratios = []
    cache = {}
    for frame, pts in points_by_frame.items():
        for p in pts:
            for o in p.observations:
                dp = find_depth_file(root, o.cam, frame)
                if dp is None:
                    continue
                key = str(dp)
                if key not in cache:
                    cache[key] = load_depth(dp)
                zraw = sample_depth_nearest(cache[key], o.xy[0], o.xy[1])
                if zraw is None:
                    continue
                pose, _ = pose_source_by_cam[o.cam]
                Xc = qvec_to_rotmat(pose.qvec) @ p.xyz + np.asarray(pose.tvec)
                if Xc[2] > 0:
                    ratios.append(float(Xc[2] / zraw))
                if len(ratios) >= 5000:
                    break
            if len(ratios) >= 5000:
                break
        if len(ratios) >= 5000:
            break
    if not ratios:
        raise RuntimeError("--depth-gt requested but no valid depth samples were found")
    scale = float(np.median(ratios))
    print(f"[DEPTH GT] automatic z-depth scale = {scale:.9g} from {len(ratios)} samples")
    return scale


def depth_gt_evaluation(root, tracks, pose_source_by_cam, rigidity_edges, out):
    points_by_frame = defaultdict(list)
    seen = set()
    for tr in tracks.values():
        for f, p in tr.points.items():
            key = (f, p.point3d_id)
            if key not in seen:
                points_by_frame[f].append(p)
                seen.add(key)

    scale = estimate_depth_scale(root, points_by_frame, pose_source_by_cam)
    cache = {}
    depth_xyz = {}
    point_rows = []

    for tid, tr in tracks.items():
        for frame, p in sorted(tr.points.items()):
            worlds = []
            for o in p.observations:
                dp = find_depth_file(root, o.cam, frame)
                if dp is None:
                    continue
                key = str(dp)
                if key not in cache:
                    cache[key] = load_depth(dp)
                zraw = sample_depth_nearest(cache[key], o.xy[0], o.xy[1])
                if zraw is None:
                    continue
                z = zraw * scale
                pose, cam = pose_source_by_cam[o.cam]
                worlds.append(backproject_depth_world(pose, cam, o.xy[0], o.xy[1], z))
            if not worlds:
                continue
            Xgt = np.median(np.stack(worlds), axis=0)
            depth_xyz[(tid, frame)] = Xgt
            point_rows.append({
                "track_id": tid,
                "frame": frame,
                "label": tr.label,
                "X_tri": p.xyz[0], "Y_tri": p.xyz[1], "Z_tri": p.xyz[2],
                "X_depth": Xgt[0], "Y_depth": Xgt[1], "Z_depth": Xgt[2],
                "tri_depth_error": float(np.linalg.norm(p.xyz - Xgt)),
                "depth_views": len(worlds),
            })

    motion_rows = []
    for tid, tr in tracks.items():
        frames = sorted(f for f in tr.points if (tid, f) in depth_xyz)
        for fa, fb in zip(frames[:-1], frames[1:]):
            tri_da = tr.points[fb].xyz - tr.points[fa].xyz
            gt_da = depth_xyz[(tid, fb)] - depth_xyz[(tid, fa)]
            motion_rows.append({
                "track_id": tid,
                "label": tr.label,
                "frame_a": fa,
                "frame_b": fb,
                "tri_motion": float(np.linalg.norm(tri_da)),
                "depth_motion": float(np.linalg.norm(gt_da)),
                "motion_vector_error": float(np.linalg.norm(tri_da - gt_da)),
            })

    pair_rows = []
    for e in rigidity_edges:
        a, b = e["track_a"], e["track_b"]
        common = sorted(
            f for f in set(tracks[a].points) & set(tracks[b].points)
            if (a, f) in depth_xyz and (b, f) in depth_xyz
        )
        if len(common) < 2:
            continue
        dtri = np.asarray([
            np.linalg.norm(tracks[a].points[f].xyz - tracks[b].points[f].xyz)
            for f in common
        ])
        dgt = np.asarray([
            np.linalg.norm(depth_xyz[(a, f)] - depth_xyz[(b, f)])
            for f in common
        ])
        pair_rows.append({
            "track_a": a,
            "track_b": b,
            "label_a": tracks[a].label,
            "label_b": tracks[b].label,
            "common_frames": len(common),
            "tri_relative_mad": robust_mad(dtri),
            "depth_relative_mad": robust_mad(dgt),
            "relative_distance_evolution_error": float(np.median(np.abs((dtri - np.median(dtri)) - (dgt - np.median(dgt))))),
            "median_absolute_relative_distance_error": float(np.median(np.abs(dtri - dgt))),
        })

    track_motion_rows = []
    by_track = defaultdict(list)
    for row in motion_rows:
        by_track[row["track_id"]].append(row)
    for tid, rows in sorted(by_track.items()):
        track_motion_rows.append({
            "track_id": tid,
            "label": tracks[tid].label,
            "num_frame_pairs": len(rows),
            "median_colmap_motion": float(np.median([r["tri_motion"] for r in rows])),
            "median_depth_motion": float(np.median([r["depth_motion"] for r in rows])),
            "median_motion_vector_error": float(np.median([r["motion_vector_error"] for r in rows])),
        })

    write_dict_csv(out / "depth_gt_points.csv", point_rows)
    write_dict_csv(out / "depth_gt_temporal_matches.csv", motion_rows)
    write_dict_csv(out / "depth_gt_track_motion.csv", track_motion_rows)
    write_dict_csv(out / "depth_gt_relative_distance.csv", pair_rows)
    metrics = {
        "num_depth_points": len(point_rows),
        "num_temporal_motion_pairs": len(motion_rows),
        "num_relative_distance_pairs": len(pair_rows),
        "median_point_geometry_error": (
            float(np.median([r["tri_depth_error"] for r in point_rows])) if point_rows else math.nan
        ),
        "median_frame_to_frame_motion_error": (
            float(np.median([r["motion_vector_error"] for r in motion_rows])) if motion_rows else math.nan
        ),
        "median_relative_distance_error": (
            float(np.median([r["median_absolute_relative_distance_error"] for r in pair_rows]))
            if pair_rows else math.nan
        ),
        "median_rigidity_evolution_error": (
            float(np.median([r["relative_distance_evolution_error"] for r in pair_rows]))
            if pair_rows else math.nan
        ),
        "estimator_used_depth": False,
    }
    (out / "depth_gt_metrics.json").write_text(json.dumps(finite_json(metrics), indent=2) + "\n")

    if point_rows:
        print(
            f"[DEPTH GT] median triangulation-vs-depth point error = "
            f"{np.median([r['tri_depth_error'] for r in point_rows]):.6g}"
        )
    if motion_rows:
        print(
            f"[DEPTH GT] median 3-D motion-vector error = "
            f"{np.median([r['motion_vector_error'] for r in motion_rows]):.6g}"
        )


# =============================================================================
# CSV outputs
# =============================================================================

def write_dict_csv(path: Path, rows):
    rows = list(rows)
    if not rows:
        path.write_text("")
        return path
    fields = []
    seen = set()
    for r in rows:
        for k in r:
            if k not in seen:
                fields.append(k); seen.add(k)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader(); w.writerows(rows)
    return path


def save_outputs(
    out,
    points_by_frame,
    all_points_by_frame,
    tracks,
    match_rows,
    rigidity_edges,
    contrastive_history,
    assignments,
    transform_rows,
    selected_frames,
    static_cluster,
    cluster_scores,
):
    point_rows = []
    pid_to_track = {}
    for tid, tr in tracks.items():
        for frame, p in tr.points.items():
            pid_to_track[(frame, p.point3d_id)] = tid

    accepted_point_ids = {
        (frame, p.point3d_id)
        for frame, points in points_by_frame.items()
        for p in points
    }
    for frame in sorted(all_points_by_frame):
        for p in all_points_by_frame[frame]:
            tid = pid_to_track.get((frame, p.point3d_id), -1)
            tr = tracks.get(tid)
            point_rows.append({
                "frame": frame,
                "point3d_id": p.point3d_id,
                "track_id": tid,
                "label": tr.label if tr else "UNTRACKED",
                "confidence": tr.confidence if tr else 0.0,
                "X": p.xyz[0], "Y": p.xyz[1], "Z": p.xyz[2],
                "reproj_error": p.error,
                "support_cameras": len(p.observations),
                "cameras": ";".join(sorted(o.cam for o in p.observations)),
                "accepted_for_temporal_matching": (frame, p.point3d_id) in accepted_point_ids,
            })

    track_rows = []
    for tid, tr in sorted(tracks.items()):
        fs = sorted(tr.points)
        if len(fs) >= 2:
            X0 = tr.points[fs[0]].xyz
            disp = np.asarray([np.linalg.norm(tr.points[f].xyz - X0) for f in fs])
            med_disp = float(np.median(disp))
            max_disp = float(np.max(disp))
        else:
            med_disp = max_disp = 0.0
        track_rows.append({
            "track_id": tid,
            "label": tr.label,
            "label_id": tr.label_id,
            "confidence": tr.confidence,
            "num_frames": len(fs),
            "start_frame": fs[0] if fs else -1,
            "end_frame": fs[-1] if fs else -1,
            "median_absolute_displacement": med_disp,
            "max_absolute_displacement": max_disp,
        })

    edge_rows = list(rigidity_edges)
    hist_rows = [
        {"iteration": it, "loss": loss, "pair_loss": pair_loss}
        for it, loss, pair_loss in contrastive_history
    ]
    assignment_rows = [{
        "track_id": tid,
        "anonymous_cluster_id": tr.label_id,
        "rigid_label": tr.label,
        "confidence": tr.confidence,
        "num_frames": len(tr.points),
    } for tid, tr in sorted(tracks.items())]

    files = [
        write_dict_csv(out / "multiview_3d_points.csv", point_rows),
        write_dict_csv(out / "temporal_hungarian_matches.csv", match_rows),
        write_dict_csv(out / "temporal_3d_tracks.csv", track_rows),
        write_dict_csv(out / "relative_distance_temporal_edges.csv", edge_rows),
        write_dict_csv(out / "contrastive_loss.csv", hist_rows),
        write_dict_csv(out / "rigid_part_assignments.csv", assignment_rows),
        write_dict_csv(out / "rigid_part_transforms.csv", transform_rows),
    ]

    source_rows = [r for r in match_rows if r["point3d_a"] >= 0]
    accepted = [r for r in source_rows if r["accepted"]]
    internal = defaultdict(list)
    for e in rigidity_edges:
        ca = tracks[e["track_a"]].label_id
        cb = tracks[e["track_b"]].label_id
        if ca >= 0 and ca == cb:
            internal[str(ca)].append(e["rigidity_error"])
    summary = {
        "num_frames": len(selected_frames),
        "num_multiview_points_per_frame": {
            str(f): len(all_points_by_frame.get(f, [])) for f in selected_frames
        },
        "num_temporal_tracks": len(tracks),
        "mean_track_length": float(np.mean([len(t.points) for t in tracks.values()])),
        "median_track_length": float(np.median([len(t.points) for t in tracks.values()])),
        "num_rigidity_edges": len(rigidity_edges),
        "num_static_tracks": sum(t.label == "STATIC" for t in tracks.values()),
        "num_moving_tracks_per_part": {
            label: sum(t.label == label for t in tracks.values())
            for label in sorted({t.label for t in tracks.values() if t.label.startswith("MOVING_")})
        },
        "num_ambiguous_tracks": sum(t.label == "AMBIGUOUS" for t in tracks.values()),
        "hungarian_mean_relative_cost": (
            float(np.mean([r["relative_cost"] for r in accepted])) if accepted else math.nan
        ),
        "hungarian_rejection_fraction": (
            1.0 - len(accepted) / len(source_rows) if source_rows else 0.0
        ),
        "cluster_internal_rigidity_error": {
            cid: float(np.median(values)) for cid, values in internal.items()
        },
        "static_identity_motion_score": cluster_scores[static_cluster],
    }
    (out / "summary.json").write_text(json.dumps(finite_json(summary), indent=2) + "\n")
    files.append(out / "summary.json")
    return files


# =============================================================================
# Visual debug
# =============================================================================

LABEL_COLORS = {
    "STATIC": (40, 220, 40),
    "MOVING_1": (40, 40, 240),
    "MOVING_2": (0, 170, 255),
    "MOVING_3": (255, 80, 80),
    "MOVING_4": (200, 80, 220),
    "AMBIGUOUS": (160, 160, 160),
    "UNTRACKED": (100, 100, 100),
}


def label_color(label):
    if label in LABEL_COLORS:
        return LABEL_COLORS[label]
    if label.startswith("MOVING_"):
        k = int(label.split("_")[-1])
        palette = [(40,40,240),(0,170,255),(255,80,80),(200,80,220),(220,180,40),(80,220,220)]
        return palette[(k - 1) % len(palette)]
    return (160,160,160)


def find_rgb_frame(image_path, cam, frame):
    for ext in (".png", ".jpg", ".jpeg", ".PNG", ".JPG", ".JPEG"):
        p = image_path / cam / f"frame_{frame:06d}{ext}"
        if p.exists():
            return p
    return None


def render_colmap_reprojection_debug(
    image_path, cameras, selected_frames, raw_points_by_frame,
    pose_source_by_cam, quality, out, fps
):
    debug = out / "visual_debug"
    reproj_root = debug / "colmap_reprojection"
    reproj_root.mkdir(parents=True, exist_ok=True)
    sample = cv2.imread(str(find_rgb_frame(image_path, cameras[0], selected_frames[0])))
    if sample is None:
        raise RuntimeError("Could not open RGB frames for COLMAP quality debug")
    h, w = sample.shape[:2]
    tile_w = 320
    tile_h = int(round(tile_w * h / w))
    cols, rows = 4, int(math.ceil(len(cameras) / 4))
    video_path = debug / "all_cameras_colmap_quality.mp4"
    writer = cv2.VideoWriter(
        str(video_path), cv2.VideoWriter_fourcc(*"mp4v"), fps,
        (cols * tile_w, rows * tile_h),
    )
    if not writer.isOpened():
        raise RuntimeError(f"Could not create {video_path}")

    for frame in selected_frames:
        frame_dir = reproj_root / f"frame_{frame:06d}"
        frame_dir.mkdir(parents=True, exist_ok=True)
        panels = []
        q = quality[frame]
        color_scale = max(q["p90_reprojection_error_px"], 1e-8)
        for cam in cameras:
            img = cv2.imread(str(find_rgb_frame(image_path, cam, frame)), cv2.IMREAD_COLOR)
            if img is None:
                img = np.zeros((h, w, 3), np.uint8)
            pose, camera = pose_source_by_cam[cam]
            sparse_counter = 0
            for point in raw_points_by_frame.get(frame, []):
                obs = next((o for o in point.observations if o.cam == cam), None)
                if obs is None:
                    continue
                projected = project_world(pose, camera, point.xyz)
                if projected is None:
                    continue
                observed_xy = tuple(np.rint(obs.xy).astype(int))
                projected_xy = tuple(np.rint(projected).astype(int))
                residual = float(np.linalg.norm(obs.xy - projected))
                a = float(np.clip(residual / color_scale, 0.0, 1.0))
                color = (0, int(round(255 * (1.0 - a))), int(round(255 * a)))
                cv2.circle(img, observed_xy, 3, color, -1, cv2.LINE_AA)
                cv2.drawMarker(img, projected_xy, color, cv2.MARKER_CROSS, 7, 1, cv2.LINE_AA)
                cv2.line(img, observed_xy, projected_xy, color, 1, cv2.LINE_AA)
                if sparse_counter % 20 == 0:
                    cv2.putText(
                        img, f"P{point.point3d_id} {residual:.2f}px S{len(point.observations)}",
                        (observed_xy[0] + 4, observed_xy[1] - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.28, color, 1, cv2.LINE_AA,
                    )
                sparse_counter += 1
            header = (
                f"{cam} frame={frame:06d} points3D={q['num_points3D']} "
                f"med={q['median_reprojection_error_px']:.2f}px "
                f"p90={q['p90_reprojection_error_px']:.2f}px "
                f"2/3/4+={q['num_points_support_2cams']}/"
                f"{q['num_points_support_3cams']}/{q['num_points_support_4plus_cams']}"
            )
            cv2.rectangle(img, (0, 0), (min(w, 720), 24), (0, 0, 0), -1)
            cv2.putText(img, header, (6, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (255, 255, 255), 1, cv2.LINE_AA)
            cv2.imwrite(str(frame_dir / f"{cam}.png"), img)
            panels.append(cv2.resize(img, (tile_w, tile_h), interpolation=cv2.INTER_AREA))
        while len(panels) < rows * cols:
            panels.append(np.zeros((tile_h, tile_w, 3), np.uint8))
        grid = np.concatenate([
            np.concatenate(panels[r * cols:(r + 1) * cols], axis=1)
            for r in range(rows)
        ], axis=0)
        writer.write(grid)
    writer.release()
    print(f"[COLMAP QUALITY DEBUG] {video_path}")
    return video_path


def camera_frustum_segments(pose, camera, scale):
    R = qvec_to_rotmat(pose.qvec)
    center = -R.T @ np.asarray(pose.tvec, np.float64)
    if camera.model == "PINHOLE":
        fx, fy, cx, cy = camera.params[:4]
    elif camera.model == "SIMPLE_PINHOLE":
        fx, cx, cy = camera.params[:3]
        fy = fx
    else:
        return center, []
    corners = []
    for u, v in ((0, 0), (camera.width, 0), (camera.width, camera.height), (0, camera.height)):
        ray = np.asarray([(u - cx) / fx, (v - cy) / fy, 1.0]) * scale
        corners.append(center + R.T @ ray)
    segments = [(center, corner) for corner in corners]
    segments += [(corners[i], corners[(i + 1) % 4]) for i in range(4)]
    return center, segments


def fixed_world_projection(points, pose_source_by_cam):
    xyz = [np.asarray(p, np.float64) for p in points]
    if not xyz:
        xyz = [np.zeros(3), np.ones(3)]
    cloud = np.stack(xyz)
    diagonal = max(float(np.linalg.norm(np.ptp(cloud, axis=0))), 1e-3)
    segments = []
    centers = []
    for pose, camera in pose_source_by_cam.values():
        center, frustum = camera_frustum_segments(pose, camera, 0.08 * diagonal)
        centers.append(center)
        segments.extend(frustum)
    all_xyz = xyz + centers + [p for segment in segments for p in segment]
    view_z = np.asarray([1.0, -1.0, 0.8]); view_z /= np.linalg.norm(view_z)
    view_x = np.cross(np.asarray([0.0, 0.0, 1.0]), view_z); view_x /= np.linalg.norm(view_x)
    view_y = np.cross(view_z, view_x)
    view = np.stack([view_x, view_y])
    projected = np.stack(all_xyz) @ view.T
    lo = projected.min(axis=0); hi = projected.max(axis=0)
    pad = np.maximum((hi - lo) * 0.08, 1e-3)
    return view, lo - pad, hi + pad, segments, diagonal


def world_to_pixel(point, view, lo, hi, width, height):
    uv = np.asarray(point) @ view.T
    x = int(round((uv[0] - lo[0]) / max(hi[0] - lo[0], 1e-8) * (width - 60) + 30))
    y = int(round((1.0 - (uv[1] - lo[1]) / max(hi[1] - lo[1], 1e-8)) * (height - 60) + 30))
    return x, y


def render_world_canvas(points, colors, lines, projection, header, size=(960, 720)):
    width, height = size
    view, lo, hi, frustum_segments, diagonal = projection
    canvas = np.full((height, width, 3), 250, np.uint8)
    for a, b in frustum_segments:
        cv2.line(canvas, world_to_pixel(a, view, lo, hi, width, height),
                 world_to_pixel(b, view, lo, hi, width, height), (180, 180, 180), 1, cv2.LINE_AA)
    origin = np.zeros(3)
    for axis, color in zip(np.eye(3), ((0, 0, 255), (0, 170, 0), (255, 0, 0))):
        cv2.arrowedLine(
            canvas,
            world_to_pixel(origin, view, lo, hi, width, height),
            world_to_pixel(origin + axis * 0.1 * diagonal, view, lo, hi, width, height),
            color, 2, cv2.LINE_AA, tipLength=0.15,
        )
    for a, b, color in lines:
        cv2.line(canvas, world_to_pixel(a, view, lo, hi, width, height),
                 world_to_pixel(b, view, lo, hi, width, height), color, 1, cv2.LINE_AA)
    for point, color in zip(points, colors):
        cv2.circle(canvas, world_to_pixel(point, view, lo, hi, width, height), 3, color, -1, cv2.LINE_AA)
    cv2.rectangle(canvas, (0, 0), (width, 30), (0, 0, 0), -1)
    cv2.putText(canvas, header, (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    return canvas


def render_3d_debug_videos(
    selected_frames, raw_points_by_frame, points_by_frame, tracks,
    pose_source_by_cam, out, fps
):
    debug = out / "visual_debug"
    all_xyz = [p.xyz for points in raw_points_by_frame.values() for p in points]
    projection = fixed_world_projection(all_xyz, pose_source_by_cam)
    size = (960, 720)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    model_path = debug / "colmap_models_3d.mp4"
    track_path = debug / "rigid_tracks_3d.mp4"
    model_writer = cv2.VideoWriter(str(model_path), fourcc, fps, size)
    track_writer = cv2.VideoWriter(str(track_path), fourcc, fps, size)
    if not model_writer.isOpened() or not track_writer.isOpened():
        raise RuntimeError("Could not create 3-D debug videos")
    still_dir = debug / "colmap_models_3d"
    still_dir.mkdir(parents=True, exist_ok=True)
    snapshots = set(selected_frames[::max(1, len(selected_frames) // 6)]) | {selected_frames[-1]}

    for frame in selected_frames:
        raw = raw_points_by_frame.get(frame, [])
        model_colors = []
        for p in raw:
            support = len(p.observations)
            model_colors.append((255, max(30, 220 - support * 35), min(255, 60 + support * 45)))
        model_canvas = render_world_canvas(
            [p.xyz for p in raw], model_colors, [], projection,
            f"COLMAP frame={frame:06d} points3D={len(raw)} | fixed world view", size,
        )
        model_writer.write(model_canvas)
        if frame in snapshots:
            cv2.imwrite(str(still_dir / f"frame_{frame:06d}.png"), model_canvas)

        current_points = []
        current_colors = []
        trails = []
        for track in tracks.values():
            if frame not in track.points:
                continue
            current_points.append(track.points[frame].xyz)
            color = label_color(track.label)
            current_colors.append(color)
            history = [f for f in sorted(track.points) if f <= frame][-8:]
            for fa, fb in zip(history[:-1], history[1:]):
                trails.append((track.points[fa].xyz, track.points[fb].xyz, color))
        track_canvas = render_world_canvas(
            current_points, current_colors, trails, projection,
            f"Rigid tracks frame={frame:06d} tracked={len(current_points)} | fixed world view", size,
        )
        view, lo, hi, _, _ = projection
        for tid, track in tracks.items():
            if tid % 20 == 0 and frame in track.points:
                x, y = world_to_pixel(track.points[frame].xyz, view, lo, hi, size[0], size[1])
                cv2.putText(
                    track_canvas, f"T{tid}", (x + 3, y - 3),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.32, label_color(track.label), 1, cv2.LINE_AA,
                )
        track_writer.write(track_canvas)
    model_writer.release(); track_writer.release()
    print(f"[3D DEBUG] {model_path}")
    print(f"[3D DEBUG] {track_path}")
    return model_path, track_path


def render_debug_video(
    image_path, cameras, selected_frames, points_by_frame, tracks,
    pose_source_by_cam, match_rows, out, fps
):
    debug = out / "visual_debug"
    debug.mkdir(parents=True, exist_ok=True)

    pid_to_track = {}
    for tid, tr in tracks.items():
        for f, p in tr.points.items():
            pid_to_track[(f, p.point3d_id)] = tid

    sample = cv2.imread(str(find_rgb_frame(image_path, cameras[0], selected_frames[0])))
    if sample is None:
        raise RuntimeError("Could not open debug RGB frames")
    h, w = sample.shape[:2]
    tile_w = 320
    tile_h = int(round(tile_w * h / w))
    cols = 4
    rows = int(math.ceil(len(cameras) / cols))
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    panel_path = debug / "all_cameras_hungarian_tracks.mp4"
    panel_writer = cv2.VideoWriter(str(panel_path), fourcc, fps, (cols*tile_w, rows*tile_h))
    if not panel_writer.isOpened():
        raise RuntimeError(f"Could not create {panel_path}")

    per_cam = {}
    for cam in cameras:
        p = debug / f"{cam}_hungarian_tracks.mp4"
        vw = cv2.VideoWriter(str(p), fourcc, fps, (w, h))
        if not vw.isOpened():
            raise RuntimeError(f"Could not create {p}")
        per_cam[cam] = (vw, p)

    snapshots = set(selected_frames[:: max(1, len(selected_frames)//6)])
    snapshots.add(selected_frames[-1])

    prev_selected = {selected_frames[i]: selected_frames[i-1] for i in range(1, len(selected_frames))}
    rejected = defaultdict(set)
    for row in match_rows:
        if row["accepted"]:
            continue
        if row["point3d_a"] >= 0:
            rejected[row["frame_a"]].add(row["point3d_a"])
        if row["point3d_b"] >= 0:
            rejected[row["frame_b"]].add(row["point3d_b"])

    for frame in selected_frames:
        panels = []
        for cam in cameras:
            img = cv2.imread(str(find_rgb_frame(image_path, cam, frame)), cv2.IMREAD_COLOR)
            if img is None:
                img = np.zeros((h,w,3), np.uint8)

            # Draw every triangulated point visible in this camera.
            visible = 0
            static_count = 0
            moving_count = 0
            for p in points_by_frame.get(frame, []):
                obs = next((o for o in p.observations if o.cam == cam), None)
                if obs is None:
                    continue
                tid = pid_to_track.get((frame, p.point3d_id), -1)
                tr = tracks.get(tid)
                label = tr.label if tr else "UNTRACKED"
                c = label_color(label)
                xy = obs.xy
                x, y = int(round(xy[0])), int(round(xy[1]))

                # Temporal rigid-distance match arrow: project the previous 3-D
                # point of the same track into this fixed camera, then connect it
                # to the current actual observation.
                if tr is not None and frame in prev_selected:
                    pf = prev_selected[frame]
                    pp = tr.points.get(pf)
                    if pp is not None:
                        pose, cam_model = pose_source_by_cam[cam]
                        uv_prev = project_world(pose, cam_model, pp.xyz)
                        if uv_prev is not None:
                            xp, yp = int(round(uv_prev[0])), int(round(uv_prev[1]))
                            cv2.arrowedLine(img, (xp,yp), (x,y), c, 1, cv2.LINE_AA, tipLength=0.25)

                if label == "STATIC":
                    cv2.drawMarker(img, (x,y), c, cv2.MARKER_CROSS, 8, 1, cv2.LINE_AA)
                    static_count += 1
                else:
                    cv2.circle(img, (x,y), 4, c, -1, cv2.LINE_AA)
                    if label.startswith("MOVING_"):
                        moving_count += 1
                visible += 1

                if p.point3d_id in rejected[frame]:
                    cv2.drawMarker(img, (x, y), (0, 0, 255), cv2.MARKER_TILTED_CROSS, 10, 2, cv2.LINE_AA)

                if tr is not None and label.startswith("MOVING_") and tr.confidence >= 0.75:
                    cv2.putText(img, f"T{tid}", (x+4,y-4), cv2.FONT_HERSHEY_SIMPLEX, 0.30, c, 1, cv2.LINE_AA)

            header = f"{cam} f={frame:06d} pts={visible} static={static_count} moving={moving_count}"
            cv2.rectangle(img, (0,0), (min(w, 430), 25), (0,0,0), -1)
            cv2.putText(img, header, (7,17), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255,255,255), 1, cv2.LINE_AA)

            per_cam[cam][0].write(img)
            panels.append(cv2.resize(img, (tile_w,tile_h), interpolation=cv2.INTER_AREA))

        while len(panels) < rows*cols:
            panels.append(np.zeros((tile_h,tile_w,3), np.uint8))
        grid = np.concatenate([
            np.concatenate(panels[r*cols:(r+1)*cols], axis=1)
            for r in range(rows)
        ], axis=0)
        panel_writer.write(grid)
        if frame in snapshots:
            cv2.imwrite(str(debug / f"frame_{frame:06d}_panel.png"), grid)

    panel_writer.release()
    for vw, _ in per_cam.values():
        vw.release()

    print(f"[VISUAL DEBUG] {panel_path}")
    print("[VISUAL DEBUG] crosses=STATIC, filled circles=MOVING/AMBIGUOUS, arrows=consecutive 3-D track matches")
    return panel_path


def render_rigidity_matrix(tracks, rigidity_edges, selected_frames, out, fps):
    def label_order(item):
        tid, track = item
        if track.label == "STATIC":
            group = 0
        elif track.label.startswith("MOVING_"):
            group = int(track.label.split("_")[-1])
        else:
            group = 10_000
        return group, tid

    ids = [
        tid for tid, tr in sorted(tracks.items(), key=label_order)
        if len(tr.points) >= RIGID_MIN_COMMON_FRAMES
    ]
    if not ids:
        return None
    idx = {tid:i for i,tid in enumerate(ids)}
    M = np.full((len(ids),len(ids)), np.nan, np.float64)
    np.fill_diagonal(M, 0.0)
    for e in rigidity_edges:
        if e["track_a"] in idx and e["track_b"] in idx:
            i,j=idx[e["track_a"]],idx[e["track_b"]]
            M[i,j]=M[j,i]=e["rigidity_error"]
    finite = M[np.isfinite(M)]
    if not finite.size:
        return None
    vmax = max(float(np.percentile(finite,95)),1e-8)
    vis = np.nan_to_num(M, nan=vmax)
    vis = np.clip(vis / vmax * 255, 0, 255).astype(np.uint8)
    vis = cv2.applyColorMap(255-vis, cv2.COLORMAP_TURBO)
    vis = cv2.resize(vis, (700, 700), interpolation=cv2.INTER_NEAREST)
    canvas = np.full((768, 768, 3), 245, np.uint8)
    canvas[50:750, 34:734] = vis
    cv2.putText(
        canvas, "Relative-distance rigidity R_ij | ordered by final label, then track ID",
        (18, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.47, (20, 20, 20), 1, cv2.LINE_AA,
    )
    png = out / "visual_debug" / "relative_distance_matrix.png"
    cv2.imwrite(str(png), canvas)
    video = out / "visual_debug" / "relative_distance_matrix.mp4"
    writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*"mp4v"), fps, (768, 768))
    if not writer.isOpened():
        raise RuntimeError(f"Could not create {video}")
    for frame in selected_frames:
        frame_canvas = canvas.copy()
        cv2.putText(
            frame_canvas, f"frame {frame:06d}", (610, 28),
            cv2.FONT_HERSHEY_SIMPLEX, 0.42, (20, 20, 20), 1, cv2.LINE_AA,
        )
        writer.write(frame_canvas)
    writer.release()
    return video


# =============================================================================
# Main
# =============================================================================

def main():
    ap = argparse.ArgumentParser(
        description="Independent multiview COLMAP reconstruction and relative-distance Hungarian tracking"
    )
    ap.add_argument("root", help="Dataset root, e.g. input/USB_100109")
    ap.add_argument("--num-frames", type=int, default=None,
                    help="Number of selected common frames. Default: all.")
    ap.add_argument("--frame-start", type=int, default=0)
    ap.add_argument("--frame-stride", type=int, default=1)
    ap.add_argument("--num-moving-parts", type=int, default=1,
                    help="Known number K of moving rigid parts. Labels are STATIC + K moving parts.")
    ap.add_argument("--depth-gt", action="store_true",
                    help="Evaluation only: compare triangulated motion with backprojected depth_npy motion.")
    ap.add_argument("--colmap", default="colmap")
    ap.add_argument("--colmap-model", default=None)
    ap.add_argument("--output-dir", default=None)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--debug-fps", type=float, default=DEFAULT_DEBUG_FPS)
    ap.add_argument("--use-mask", action="store_true",
                    help="Filter multiview features with ROOT/mask when available.")
    ap.add_argument("--no-visual-debug", action="store_true")
    args = ap.parse_args()

    root = Path(args.root).resolve()
    image_path = find_rgb_parent(root)
    cameras, frames_all = available_frames(image_path)
    if args.frame_stride <= 0:
        raise ValueError("--frame-stride must be positive")
    selected_frames = [f for f in frames_all if f >= args.frame_start][::args.frame_stride]
    if args.num_frames is not None:
        if args.num_frames <= 0:
            raise ValueError("--num-frames must be positive")
        selected_frames = selected_frames[:args.num_frames]
    if len(selected_frames) < 2:
        raise RuntimeError("Need at least 2 frames")

    out = Path(args.output_dir).resolve() if args.output_dir else root / "rigid_hungarian_relative_distance"
    logs = out / "logs"
    if args.force and out.exists():
        shutil.rmtree(out)
    elif out.exists() and any(out.iterdir()):
        raise FileExistsError(f"Output exists: {out}. Pass --force to recompute it.")
    out.mkdir(parents=True, exist_ok=True)
    logs.mkdir(parents=True, exist_ok=True)

    print(f"[DATA] cameras={len(cameras)} frames={len(selected_frames)}/{len(frames_all)}")
    print(f"[DATA] selected frame range {selected_frames[0]}..{selected_frames[-1]}")
    print(f"[DATA] mask={'ON' if args.use_mask else 'OFF'} depth_gt={args.depth_gt}")

    cli = ColmapCLI(args.colmap, logs)

    # Trusted original calibration.
    original_model = find_colmap_model(root, args.colmap_model)
    original_txt = out / "original_model_txt"
    convert_model_to_txt(cli, original_model, original_txt, "00_original_model_to_txt.log")
    orig_cams = parse_cameras_txt(original_txt / "cameras.txt")
    orig_images = parse_images_txt_poses(original_txt / "images.txt")
    pose_source_by_cam = choose_original_pose_per_camera(orig_images, orig_cams)
    missing = [c for c in cameras if c not in pose_source_by_cam]
    if missing:
        raise RuntimeError(f"Original COLMAP model lacks camera poses for {missing}")

    # Official SIFT extraction on only the selected images.
    image_list = out / "selected_images.txt"
    write_image_list(image_path, cameras, selected_frames, image_list)
    master_db = out / "master.db"
    if master_db.exists():
        master_db.unlink()
    h = cli.help("feature_extractor")
    feat = [
        "--database_path", master_db,
        "--image_path", image_path,
        "--camera_mode", "2",
    ]
    if "--image_list_path" in h:
        feat += ["--image_list_path", image_list]
    else:
        raise RuntimeError("Installed COLMAP feature_extractor has no --image_list_path; cannot restrict selected frames safely")
    if "FeatureExtraction.use_gpu" in h:
        feat += ["--FeatureExtraction.use_gpu", "1"]
    if "SiftExtraction.max_num_orientations" in h:
        feat += ["--SiftExtraction.max_num_orientations", "1"]
    cli.run("feature_extractor", feat, "01_feature_extractor.log")
    update_database_cameras_from_original(master_db, pose_source_by_cam)

    images = db_images(master_db)
    print(f"[FEATURES] {len(images)} images in selected-frame COLMAP database")

    expanded_txt = out / "expanded_known_pose_model_txt"
    expanded_bin = out / "expanded_known_pose_model"
    build_expanded_known_pose_model(master_db, pose_source_by_cam, expanded_txt)
    convert_model_to_bin(cli, expanded_txt, expanded_bin, "02_expanded_model_to_bin.log")

    pair_file = out / "multiview_pairs.txt"
    write_same_frame_multiview_pairs(images, selected_frames, pair_file)

    raw_db = out / "multiview_raw.db"
    shutil.copy2(master_db, raw_db)
    h = cli.help("matches_importer")
    match_args = [
        "--database_path", raw_db,
        "--match_list_path", pair_file,
        "--match_type", "pairs",
    ]
    if "SiftMatching.max_ratio" in h:
        match_args += ["--SiftMatching.max_ratio", "0.8"]
    if "SiftMatching.cross_check" in h:
        match_args += ["--SiftMatching.cross_check", "1"]
    if "TwoViewGeometry.min_num_inliers" in h:
        # Preserve sparse raw descriptor matches. Geometry is handled later by
        # known-pose COLMAP triangulation + BA + filtering.
        match_args += ["--TwoViewGeometry.min_num_inliers", "1"]
    if "FeatureMatching.use_gpu" in h:
        match_args += ["--FeatureMatching.use_gpu", "1"]
    elif "SiftMatching.use_gpu" in h:
        match_args += ["--SiftMatching.use_gpu", "1"]
    cli.run("matches_importer", match_args, "03_multiview_rootSIFT_matching.log")
    rp, rm = count_matches(raw_db, "matches")
    print(f"[MATCHES] raw COLMAP RootSIFT: {rm} matches over {rp} image pairs")
    if rm == 0:
        raise RuntimeError("COLMAP returned zero same-frame multiview SIFT matches")

    explicit_matches = out / "multiview_matches_for_triangulation.txt"
    kpairs, kmatches = write_raw_matches_as_inliers(
        root, raw_db, master_db, explicit_matches,
        use_mask=(args.use_mask and (root / "mask").is_dir()),
    )
    if kmatches == 0:
        raise RuntimeError("No multiview feature matches remain after foreground filtering")

    tri_db = out / "triangulation.db"
    shutil.copy2(master_db, tri_db)
    cli.run(
        "matches_importer",
        [
            "--database_path", tri_db,
            "--match_list_path", explicit_matches,
            "--match_type", "inliers",
        ],
        "03b_import_triangulation_graph.log",
    )
    gp, gm = count_matches(tri_db, "two_view_geometries")
    print(f"[TRI GRAPH] {gm} reconstruction correspondences over {gp} pairs")
    if gm == 0:
        raise RuntimeError("No two_view_geometries after importing the multiview graph")

    filtered_txt = run_known_pose_pipeline(cli, tri_db, image_path, expanded_bin, out / "colmap_geometry")
    raw_points_by_frame = parse_filtered_points(filtered_txt / "points3D.txt", tri_db)
    points_by_frame, rejected_points, frame_quality = gate_reconstruction_points(
        raw_points_by_frame, selected_frames, pose_source_by_cam, len(cameras)
    )
    export_per_frame_colmap_models(
        cli, filtered_txt, selected_frames, out, frame_quality, rejected_points
    )
    counts = {f: len(points_by_frame.get(f, [])) for f in selected_frames}
    print(f"[3D] triangulated points/frame: min={min(counts.values())} median={np.median(list(counts.values())):.1f} max={max(counts.values())}")
    print(f"[3D] total independent frame points: {sum(counts.values())}")

    if not args.no_visual_debug:
        render_colmap_reprojection_debug(
            image_path, cameras, selected_frames, raw_points_by_frame,
            pose_source_by_cam, frame_quality, out, args.debug_fps,
        )

    tracks, temporal_match_rows = build_temporal_3d_tracks(
        points_by_frame, selected_frames, frame_quality
    )
    valid_ids, rigidity_edges, scene_spacing, contrast_stats = build_rigidity_edges(
        tracks, selected_frames
    )
    if len(valid_ids) < args.num_moving_parts + 1:
        raise RuntimeError(
            f"Only {len(valid_ids)} temporal 3-D tracks have >= {RIGID_MIN_COMMON_FRAMES} frames; "
            f"need at least {args.num_moving_parts + 1} for STATIC + K clusters"
        )
    (
        cluster_name, static_cluster, contrastive_history, assignments,
        transform_rows, cluster_scores,
    ) = label_tracks_rigid_parts(
        tracks, valid_ids, rigidity_edges, args.num_moving_parts,
        selected_frames, scene_spacing, contrast_stats,
    )

    output_files = save_outputs(
        out, points_by_frame, raw_points_by_frame, tracks, temporal_match_rows, rigidity_edges,
        contrastive_history, assignments, transform_rows, selected_frames,
        static_cluster, cluster_scores,
    )

    visual_files = []
    if not args.no_visual_debug:
        visual_files.append(render_debug_video(
            image_path, cameras, selected_frames, points_by_frame, tracks,
            pose_source_by_cam, temporal_match_rows, out, args.debug_fps
        ))
        visual_files.extend(render_3d_debug_videos(
            selected_frames, raw_points_by_frame, points_by_frame, tracks,
            pose_source_by_cam, out, args.debug_fps,
        ))
        matrix_video = render_rigidity_matrix(
            tracks, rigidity_edges, selected_frames, out, args.debug_fps
        )
        if matrix_video:
            visual_files.append(matrix_video)

    if args.depth_gt:
        depth_gt_evaluation(root, tracks, pose_source_by_cam, rigidity_edges, out)

    print("\n[DONE]")
    for p in output_files:
        print(p)
    for path in visual_files:
        print(path)
    if args.depth_gt:
        print(out / "depth_gt_points.csv")
        print(out / "depth_gt_temporal_matches.csv")
        print(out / "depth_gt_track_motion.csv")
        print(out / "depth_gt_relative_distance.csv")
        print(out / "depth_gt_metrics.json")
    print(f"Logs: {logs}")


if __name__ == "__main__":
    main()
