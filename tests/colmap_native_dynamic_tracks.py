#!/usr/bin/env python3
"""
COLMAP-native dynamic track discovery and multiview confirmation.

IMPORTANT
=========
This script deliberately does NOT reimplement COLMAP triangulation, geometric
verification, point filtering, or bundle adjustment.

It calls the installed official `colmap` executable for:
  - SIFT feature extraction
  - SIFT matching
  - guided/standard geometric verification
  - point_triangulator
  - bundle_adjuster
  - point_filtering
  - model_converter

Python is used only to:
  - generate image-pair lists,
  - read the SQLite database,
  - build same-camera temporal SIFT tracks from COLMAP matches,
  - classify STATIC / DYNAMIC from pixel-space motion,
  - build an expanded known-pose model by repeating the calibrated pose of each
    physical camera for every video frame,
  - read back COLMAP's final triangulated point associations.

No depth is used.
No pycolmap is used.
No Python triangulator is used.

Pipeline
--------
1) COLMAP extracts RootSIFT features for all 8 x T images.

2) TEMPORAL database:
   COLMAP matches same-camera frame pairs.
   Python links COLMAP feature identities over time.
   Static/dynamic is decided ONLY from image-space trajectory motion.

3) MULTIVIEW database:
   COLMAP matches all camera pairs at the SAME frame.
   COLMAP guided geometric verifier (when available) uses the known camera
   poses. Otherwise COLMAP's normal geometric_verifier is used.

4) COLMAP point_triangulator:
   known camera poses + same-frame multiview matches.
   There are NO temporal matches in this database, therefore dynamic points
   are triangulated independently at every frame rather than incorrectly
   forced to be static.

5) COLMAP bundle_adjuster:
   camera intrinsics and extrinsics are FIXED; only 3D points are refined.

6) COLMAP point_filtering:
   official reprojection-error / triangulation-angle filtering.

7) A cross-camera link is called GLOBAL DYNAMIC only when:
   - both endpoints belong to long-lived pixel-space DYNAMIC temporal tracks;
   - COLMAP's geometric verification accepted the image pair;
   - after triangulation + fixed-camera BA + filtering, both feature
     observations belong to the SAME surviving COLMAP point3D.

This means triangulation is exactly what you requested: confirmation of
multiview coherence of already-dynamic tracks, never the source of motion.

Example
-------
python colmap_native_dynamic_tracks.py \
    /home/samuelemara/Joint3DGS/RigEuler3DGS/input/USB_100109

Outputs
-------
ROOT/colmap_native_dynamic/
    master.db
    temporal.db
    multiview.db
    temporal_tracks.csv
    temporal_track_observations.csv
    colmap_dynamic_multiview_evidence.csv
    global_dynamic_links.csv
    expanded_known_pose_model/
    triangulated_model/
    bundle_adjusted_model/
    filtered_model/
    logs/
"""

import argparse
import csv
import math
import os
import shutil
import sqlite3
import subprocess
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


MAX_IMAGE_ID = 2147483647

# COLMAP sensor/models.h legacy/current IDs for common models.
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
            raise RuntimeError(
                f"COLMAP executable '{executable}' was not found. "
                "Run `which colmap` and pass --colmap /path/to/colmap if needed."
            )
        self.exe = str(exe)
        self.log_dir = log_dir
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self._help_cache = {}

        version = subprocess.run(
            [self.exe, "-h"],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
        ).stdout
        print(f"[COLMAP] executable: {self.exe}")
        first = version.splitlines()[0] if version else "COLMAP"
        print(f"[COLMAP] {first}")

    def help(self, command: str) -> str:
        if command not in self._help_cache:
            p = subprocess.run(
                [self.exe, command, "-h"],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                check=False,
            )
            self._help_cache[command] = p.stdout or ""
        return self._help_cache[command]

    def has_option(self, command: str, option: str) -> bool:
        return option in self.help(command)

    def has_command(self, command: str) -> bool:
        p = subprocess.run(
            [self.exe, command, "-h"],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
        )
        txt = p.stdout or ""
        return p.returncode == 0 or "Options" in txt or "Usage" in txt

    def run(self, command: str, args, log_name: str):
        cmd = [self.exe, command] + [str(x) for x in args]
        print("\n[COLMAP]", " ".join(cmd))
        log_path = self.log_dir / log_name

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
                lines = log_path.read_text(errors="replace").splitlines()
                tail = "\n".join(lines[-50:])
            except Exception:
                pass
            raise RuntimeError(
                f"COLMAP command failed ({p.returncode}): {' '.join(cmd)}\n"
                f"Log: {log_path}\n\n{tail}"
            )

        return log_path


# =============================================================================
# Dataset/model helpers
# =============================================================================

def frame_number(name: str) -> int:
    return int(Path(name).stem.split("_")[-1])


def camera_name_from_image_name(name: str) -> str:
    parts = Path(name).parts
    for p in parts:
        if p.startswith("cam_"):
            return p
    raise RuntimeError(f"Cannot find cam_XXX in image name: {name}")


def find_rgb_parent(root: Path) -> Path:
    for name in ("rgb", "video_rgb"):
        p = root / name
        if p.is_dir():
            cams = [x for x in p.iterdir() if x.is_dir() and x.name.startswith("cam_")]
            if cams:
                return p
    raise FileNotFoundError("Could not find ROOT/rgb/cam_* or ROOT/video_rgb/cam_*")


def find_colmap_model(root: Path, explicit: str | None) -> Path:
    if explicit:
        p = Path(explicit)
        if not p.exists():
            raise FileNotFoundError(p)
        return p

    candidates = [
        root / "colmap",
        root / "colmap" / "sparse",
        root / "colmap" / "sparse" / "0",
        root / "sparse",
        root / "sparse" / "0",
    ]

    def valid(p: Path):
        return (
            ((p / "cameras.bin").exists() and (p / "images.bin").exists())
            or ((p / "cameras.txt").exists() and (p / "images.txt").exists())
        )

    for p in candidates:
        if valid(p):
            return p

    for base in (root / "colmap", root / "sparse"):
        if base.exists():
            for p in base.rglob("*"):
                if p.is_dir() and valid(p):
                    return p

    raise FileNotFoundError(
        "Could not locate the calibrated COLMAP sparse model. "
        "Pass --colmap-model /path/to/model."
    )


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
            camera_id=cid,
            model=p[1],
            width=int(p[2]),
            height=int(p[3]),
            params=[float(v) for v in p[4:]],
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

        image_id = int(p[0])
        out[image_id] = TextImagePose(
            image_id=image_id,
            qvec=tuple(float(v) for v in p[1:5]),
            tvec=tuple(float(v) for v in p[5:8]),
            camera_id=int(p[8]),
            name=" ".join(p[9:]),
        )

        # Points2D line.
        if i < len(lines):
            i += 1

    return out


def convert_model_to_txt(cli: ColmapCLI, input_model: Path, output_dir: Path, log_prefix: str):
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True)
    cli.run(
        "model_converter",
        [
            "--input_path", input_model,
            "--output_path", output_dir,
            "--output_type", "TXT",
        ],
        f"{log_prefix}_model_converter_txt.log",
    )


def convert_model_to_bin(cli: ColmapCLI, input_model: Path, output_dir: Path, log_prefix: str):
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True)
    cli.run(
        "model_converter",
        [
            "--input_path", input_model,
            "--output_path", output_dir,
            "--output_type", "BIN",
        ],
        f"{log_prefix}_model_converter_bin.log",
    )


def choose_original_pose_per_camera(images, cameras):
    by_cam = defaultdict(list)
    for im in images.values():
        try:
            cam = camera_name_from_image_name(im.name)
        except RuntimeError:
            continue
        score = 0
        if "frame_000000" in im.name:
            score -= 1000
        score += len(im.name)
        by_cam[cam].append((score, im))

    result = {}
    for cam, values in by_cam.items():
        values.sort(key=lambda x: x[0])
        im = values[0][1]
        result[cam] = (im, cameras[im.camera_id])

    return result


# =============================================================================
# SQLite helpers
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
        int(image_id): {"name": name, "camera_id": int(camera_id)}
        for image_id, name, camera_id in rows
    }


def db_keypoints(db_path: Path):
    con = sqlite3.connect(str(db_path))
    out = {}
    for image_id, rows, cols, data in con.execute(
        "SELECT image_id, rows, cols, data FROM keypoints"
    ):
        arr = blob_array(data, np.float32, (int(rows), int(cols)))
        out[int(image_id)] = arr[:, :2].copy()
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
        matches = blob_array(data, np.uint32, (int(rows), int(cols)))
        a, b = pair_id_to_image_ids(int(pair_id))
        out.append((a, b, matches[:, :2].copy()))
    con.close()
    return out


def update_database_cameras_from_original(
    db_path: Path,
    pose_source_by_cam,
):
    """
    feature_extractor creates one camera per folder (camera_mode=PER_FOLDER).
    Replace those estimated intrinsics with the trusted intrinsics from the
    original calibrated model, while preserving DB camera IDs.
    """
    con = sqlite3.connect(str(db_path))

    images = con.execute(
        "SELECT image_id, name, camera_id FROM images ORDER BY image_id"
    ).fetchall()

    folder_to_db_cam_ids = defaultdict(set)
    for _, name, camera_id in images:
        folder_to_db_cam_ids[camera_name_from_image_name(name)].add(int(camera_id))

    for cam_name, ids in folder_to_db_cam_ids.items():
        if len(ids) != 1:
            raise RuntimeError(
                f"Expected one COLMAP DB camera per folder for {cam_name}, got {ids}. "
                "Feature extraction did not use PER_FOLDER camera mode."
            )

        if cam_name not in pose_source_by_cam:
            raise RuntimeError(f"No original calibration found for {cam_name}")

        db_camera_id = next(iter(ids))
        _, original_cam = pose_source_by_cam[cam_name]

        if original_cam.model not in CAMERA_MODEL_NAME_TO_ID:
            raise RuntimeError(
                f"Camera model {original_cam.model} is not in the supported ID map."
            )

        model_id = CAMERA_MODEL_NAME_TO_ID[original_cam.model]
        params = np.asarray(original_cam.params, dtype=np.float64).tobytes()

        con.execute(
            """
            UPDATE cameras
               SET model=?, width=?, height=?, params=?, prior_focal_length=1
             WHERE camera_id=?
            """,
            (
                model_id,
                original_cam.width,
                original_cam.height,
                params,
                db_camera_id,
            ),
        )

    con.commit()
    con.close()


# =============================================================================
# Expanded known-pose model
# =============================================================================

def build_expanded_known_pose_model(
    master_db: Path,
    pose_source_by_cam,
    output_txt: Path,
):
    """
    Every video frame is a separate COLMAP image, but every frame from the same
    physical camera gets the same trusted cam_from_world pose.
    """
    if output_txt.exists():
        shutil.rmtree(output_txt)
    output_txt.mkdir(parents=True)

    images = db_images(master_db)

    # DB camera ID -> original calibrated camera, inferred from image folder.
    db_cam_to_original = {}
    for rec in images.values():
        cam_name = camera_name_from_image_name(rec["name"])
        _, original_cam = pose_source_by_cam[cam_name]
        cid = rec["camera_id"]
        if cid in db_cam_to_original:
            prev = db_cam_to_original[cid]
            if prev.model != original_cam.model or prev.params != original_cam.params:
                raise RuntimeError(
                    f"DB camera {cid} mixes incompatible physical cameras."
                )
        else:
            db_cam_to_original[cid] = original_cam

    with open(output_txt / "cameras.txt", "w") as f:
        f.write("# Camera list generated from original calibrated COLMAP model.\n")
        f.write("# CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]\n")
        for cid in sorted(db_cam_to_original):
            c = db_cam_to_original[cid]
            params = " ".join(f"{v:.17g}" for v in c.params)
            f.write(f"{cid} {c.model} {c.width} {c.height} {params}\n")

    with open(output_txt / "images.txt", "w") as f:
        f.write("# Each video frame repeats the trusted fixed camera pose.\n")
        for image_id in sorted(images):
            rec = images[image_id]
            cam_name = camera_name_from_image_name(rec["name"])
            pose, _ = pose_source_by_cam[cam_name]
            q = " ".join(f"{v:.17g}" for v in pose.qvec)
            t = " ".join(f"{v:.17g}" for v in pose.tvec)
            f.write(
                f"{image_id} {q} {t} {rec['camera_id']} {rec['name']}\n"
            )
            f.write("\n")

    (output_txt / "points3D.txt").write_text(
        "# Empty. COLMAP point_triangulator will create points.\n"
    )


# =============================================================================
# Pair files
# =============================================================================

def image_index_by_camera_frame(images):
    out = {}
    for iid, rec in images.items():
        cam = camera_name_from_image_name(rec["name"])
        frame = frame_number(rec["name"])
        out[(cam, frame)] = (iid, rec["name"])
    return out


def write_temporal_pairs(images, path: Path, window: int, all_pairs: bool):
    by_cam = defaultdict(list)
    for iid, rec in images.items():
        cam = camera_name_from_image_name(rec["name"])
        by_cam[cam].append((frame_number(rec["name"]), iid, rec["name"]))

    count = 0
    with open(path, "w") as f:
        for cam in sorted(by_cam):
            seq = sorted(by_cam[cam])
            for i in range(len(seq)):
                if all_pairs:
                    js = range(i + 1, len(seq))
                else:
                    js = range(i + 1, min(len(seq), i + 1 + window))
                for j in js:
                    f.write(f"{seq[i][2]} {seq[j][2]}\n")
                    count += 1

    print(f"[PAIRS] temporal: {count} image pairs -> {path}")


def write_multiview_pairs(images, path: Path):
    by_frame = defaultdict(list)
    for iid, rec in images.items():
        cam = camera_name_from_image_name(rec["name"])
        fr = frame_number(rec["name"])
        by_frame[fr].append((cam, iid, rec["name"]))

    count = 0
    with open(path, "w") as f:
        for fr in sorted(by_frame):
            views = sorted(by_frame[fr])
            for i in range(len(views)):
                for j in range(i + 1, len(views)):
                    f.write(f"{views[i][2]} {views[j][2]}\n")
                    count += 1

    print(f"[PAIRS] multiview same-frame: {count} image pairs -> {path}")


# =============================================================================
# Temporal COLMAP SIFT tracks
# =============================================================================

class ConflictDSU:
    def __init__(self):
        self.parent = {}
        self.rank = {}
        self.image_ids = {}

    def add(self, node, image_id):
        if node in self.parent:
            return
        self.parent[node] = node
        self.rank[node] = 0
        self.image_ids[node] = {image_id}

    def find(self, x):
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])
        return self.parent[x]

    def can_union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return True
        return self.image_ids[ra].isdisjoint(self.image_ids[rb])

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return True
        if not self.can_union(a, b):
            return False

        if self.rank[ra] < self.rank[rb]:
            ra, rb = rb, ra

        self.parent[rb] = ra
        self.image_ids[ra] |= self.image_ids[rb]
        if self.rank[ra] == self.rank[rb]:
            self.rank[ra] += 1
        return True

    def groups(self):
        g = defaultdict(list)
        for node in self.parent:
            g[self.find(node)].append(node)
        return list(g.values())


@dataclass
class Observation:
    image_id: int
    frame: int
    kp_idx: int
    x: float
    y: float


@dataclass
class TemporalTrack:
    camera: str
    track_id: int
    observations: list[Observation]
    frame_span: int = 0
    static_rms_px: float = 0.0
    coherent_range_px: float = 0.0
    jitter_px: float = 0.0
    noise_sigma_px: float = 0.0
    motion_score: float = 0.0
    label: str = "UNCERTAIN"


def read_mask_value(root: Path, cam: str, frame: int, x: float, y: float):
    p = root / "mask" / cam / f"frame_{frame:06d}.png"
    if not p.exists():
        return True
    m = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
    if m is None:
        return True
    xi = int(round(x))
    yi = int(round(y))
    return 0 <= xi < m.shape[1] and 0 <= yi < m.shape[0] and m[yi, xi] > 0


def temporal_track_metrics(track: TemporalTrack, num_bins=6):
    obs = sorted(track.observations, key=lambda o: o.frame)
    pts = np.asarray([[o.x, o.y] for o in obs], dtype=np.float64)
    frames = np.asarray([o.frame for o in obs], dtype=np.float64)

    track.frame_span = int(obs[-1].frame - obs[0].frame + 1)

    center = np.median(pts, axis=0)
    track.static_rms_px = float(
        np.sqrt(np.mean(np.sum((pts - center[None]) ** 2, axis=1)))
    )

    # Robust low-frequency trajectory from temporal-bin medians.
    edges = np.linspace(frames.min(), frames.max() + 1e-6, num_bins + 1)
    medians = []
    residuals = []

    for b in range(num_bins):
        sel = (frames >= edges[b]) & (frames < edges[b + 1])
        if not np.any(sel):
            continue
        med = np.median(pts[sel], axis=0)
        medians.append(med)
        residuals.extend(np.linalg.norm(pts[sel] - med[None], axis=1).tolist())

    if len(medians) >= 2:
        medians = np.asarray(medians)
        d = np.linalg.norm(
            medians[:, None, :] - medians[None, :, :],
            axis=2,
        )
        track.coherent_range_px = float(np.max(d))
    else:
        track.coherent_range_px = 0.0

    track.jitter_px = float(np.median(residuals)) if residuals else 0.0


def estimate_static_noise(tracks, floor_px):
    candidates = [
        t for t in tracks
        if t.frame_span >= 10 and len(t.observations) >= 5
    ]
    if not candidates:
        return float(floor_px)

    values = np.asarray([t.jitter_px for t in candidates], dtype=np.float64)
    q = np.quantile(values, 0.35)
    low = values[values <= q]
    med = float(np.median(low)) if len(low) else float(q)
    mad = float(np.median(np.abs(low - med))) if len(low) else 0.0
    return max(float(floor_px), med + 1.4826 * mad)


def build_temporal_tracks(
    root: Path,
    temporal_db: Path,
    min_track_obs: int,
    min_foreground_fraction: float,
    min_dynamic_span: int,
    static_sigma: float,
    dynamic_sigma: float,
    min_dynamic_range_px: float,
    noise_floor_px: float,
):
    images = db_images(temporal_db)
    keypoints = db_keypoints(temporal_db)
    raw_matches = db_matches(temporal_db, "matches")

    image_frame = {
        iid: frame_number(rec["name"])
        for iid, rec in images.items()
    }
    image_cam = {
        iid: camera_name_from_image_name(rec["name"])
        for iid, rec in images.items()
    }

    dsu_by_cam = defaultdict(ConflictDSU)

    # Add only nodes participating in temporal matches.
    edges = []
    for ia, ib, matches in raw_matches:
        if image_cam[ia] != image_cam[ib]:
            continue
        gap = abs(image_frame[ia] - image_frame[ib])
        cam = image_cam[ia]
        for a, b in matches:
            na = (ia, int(a))
            nb = (ib, int(b))
            dsu_by_cam[cam].add(na, ia)
            dsu_by_cam[cam].add(nb, ib)
            edges.append((gap, cam, na, nb))

    # Short temporal baselines first.
    edges.sort(key=lambda x: x[0])
    for _, cam, a, b in edges:
        dsu = dsu_by_cam[cam]
        if dsu.can_union(a, b):
            dsu.union(a, b)

    tracks_by_cam = defaultdict(list)
    obs_lookup = {}

    for cam in sorted(dsu_by_cam):
        groups = dsu_by_cam[cam].groups()
        temp = []

        for g in groups:
            observations = []
            fg_count = 0

            for iid, kp_idx in g:
                if iid not in keypoints or kp_idx >= len(keypoints[iid]):
                    continue
                x, y = keypoints[iid][kp_idx]
                fr = image_frame[iid]
                observations.append(
                    Observation(
                        image_id=iid,
                        frame=fr,
                        kp_idx=kp_idx,
                        x=float(x),
                        y=float(y),
                    )
                )
                if read_mask_value(root, cam, fr, x, y):
                    fg_count += 1

            observations.sort(key=lambda o: o.frame)

            if len(observations) < min_track_obs:
                continue

            fg_fraction = fg_count / max(len(observations), 1)
            if fg_fraction < min_foreground_fraction:
                continue

            # Remove rare off-mask observations.
            observations = [
                o for o in observations
                if read_mask_value(root, cam, o.frame, o.x, o.y)
            ]
            if len(observations) < min_track_obs:
                continue

            t = TemporalTrack(
                camera=cam,
                track_id=-1,
                observations=observations,
            )
            temporal_track_metrics(t)
            temp.append(t)

        temp.sort(
            key=lambda t: (-t.frame_span, -len(t.observations), t.observations[0].frame)
        )

        sigma = estimate_static_noise(temp, noise_floor_px)

        for tid, t in enumerate(temp):
            t.track_id = tid
            t.noise_sigma_px = sigma
            t.motion_score = t.coherent_range_px / max(sigma, 1e-8)

            if t.coherent_range_px <= static_sigma * sigma:
                t.label = "STATIC"
            elif (
                t.frame_span >= min_dynamic_span
                and t.coherent_range_px >= min_dynamic_range_px
                and t.motion_score >= dynamic_sigma
            ):
                t.label = "DYNAMIC"
            else:
                t.label = "UNCERTAIN"

            tracks_by_cam[cam].append(t)
            for o in t.observations:
                obs_lookup[(o.image_id, o.kp_idx)] = t

        counts = defaultdict(int)
        for t in temp:
            counts[t.label] += 1
        print(
            f"[TEMPORAL] {cam}: sigma={sigma:.3f}px | "
            f"STATIC={counts['STATIC']} UNCERTAIN={counts['UNCERTAIN']} "
            f"DYNAMIC={counts['DYNAMIC']}"
        )

    return tracks_by_cam, obs_lookup


# =============================================================================
# Parse final COLMAP triangulation assignments
# =============================================================================

def parse_images_txt_point3d_assignments(path: Path):
    """
    Returns {(image_id, point2D_idx): point3D_id}.
    The order of POINTS2D in images.txt is the keypoint index used by COLMAP.
    """
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
        image_id = int(p[0])

        if i >= len(lines):
            break

        points_line = lines[i].strip()
        i += 1
        if not points_line:
            continue

        vals = points_line.split()
        if len(vals) % 3 != 0:
            raise RuntimeError(
                f"Malformed POINTS2D line for image {image_id}"
            )

        num = len(vals) // 3
        for kp_idx in range(num):
            point3d_id = int(vals[3 * kp_idx + 2])
            if point3d_id >= 0:
                out[(image_id, kp_idx)] = point3d_id

    return out


# =============================================================================
# Global dynamic evidence
# =============================================================================

@dataclass
class GlobalEvidence:
    frame: int
    cam_a: str
    track_a: int
    cam_b: str
    track_b: int
    image_a: int
    kp_a: int
    image_b: int
    kp_b: int
    colmap_point3d_id: int


def collect_colmap_dynamic_evidence(
    multiview_db: Path,
    temporal_obs_lookup,
    final_point_assignments,
):
    images = db_images(multiview_db)

    # Use COLMAP geometrically verified inlier matches, not raw matches.
    inlier_pairs = db_matches(multiview_db, "two_view_geometries")

    out = []
    seen = set()

    for ia, ib, matches in inlier_pairs:
        rec_a = images[ia]
        rec_b = images[ib]

        cam_a = camera_name_from_image_name(rec_a["name"])
        cam_b = camera_name_from_image_name(rec_b["name"])
        frame_a = frame_number(rec_a["name"])
        frame_b = frame_number(rec_b["name"])

        if frame_a != frame_b:
            continue

        for ka, kb in matches:
            ka = int(ka)
            kb = int(kb)

            ta = temporal_obs_lookup.get((ia, ka))
            tb = temporal_obs_lookup.get((ib, kb))

            if ta is None or tb is None:
                continue
            if ta.label != "DYNAMIC" or tb.label != "DYNAMIC":
                continue

            pa = final_point_assignments.get((ia, ka))
            pb = final_point_assignments.get((ib, kb))

            # This is the official COLMAP triangulator + BA + filtering gate.
            if pa is None or pb is None or pa != pb:
                continue

            key = (
                frame_a,
                cam_a, ta.track_id,
                cam_b, tb.track_id,
                pa,
            )
            if key in seen:
                continue
            seen.add(key)

            out.append(
                GlobalEvidence(
                    frame=frame_a,
                    cam_a=cam_a,
                    track_a=ta.track_id,
                    cam_b=cam_b,
                    track_b=tb.track_id,
                    image_a=ia,
                    kp_a=ka,
                    image_b=ib,
                    kp_b=kb,
                    colmap_point3d_id=pa,
                )
            )

    return out


def canonical_track_link(cam_a, ta, cam_b, tb):
    if cam_a < cam_b:
        return cam_a, ta, cam_b, tb
    return cam_b, tb, cam_a, ta


def aggregate_global_links(evidence, min_support_frames):
    grouped = defaultdict(list)
    for e in evidence:
        grouped[
            canonical_track_link(
                e.cam_a, e.track_a,
                e.cam_b, e.track_b,
            )
        ].append(e)

    links = []
    for key, values in grouped.items():
        frames = len(set(v.frame for v in values))
        if frames >= min_support_frames:
            links.append((*key, frames))

    links.sort(key=lambda x: (-x[4], x[0], x[2]))
    return links


# =============================================================================
# Output CSV
# =============================================================================

def save_temporal_tracks(tracks_by_cam, out_dir):
    summary_path = out_dir / "temporal_tracks.csv"
    obs_path = out_dir / "temporal_track_observations.csv"

    with open(summary_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "camera", "track_id", "label",
            "num_observations", "start_frame", "end_frame", "frame_span",
            "static_rms_px", "coherent_range_px", "jitter_px",
            "noise_sigma_px", "motion_score"
        ])
        for cam in sorted(tracks_by_cam):
            for t in tracks_by_cam[cam]:
                w.writerow([
                    cam, t.track_id, t.label,
                    len(t.observations),
                    t.observations[0].frame,
                    t.observations[-1].frame,
                    t.frame_span,
                    t.static_rms_px,
                    t.coherent_range_px,
                    t.jitter_px,
                    t.noise_sigma_px,
                    t.motion_score,
                ])

    with open(obs_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "camera", "track_id", "label",
            "image_id", "frame", "kp_idx", "x", "y"
        ])
        for cam in sorted(tracks_by_cam):
            for t in tracks_by_cam[cam]:
                for o in t.observations:
                    w.writerow([
                        cam, t.track_id, t.label,
                        o.image_id, o.frame, o.kp_idx, o.x, o.y,
                    ])

    return summary_path, obs_path


def save_global_evidence(evidence, path):
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "frame",
            "cam_a", "track_a", "image_a", "kp_a",
            "cam_b", "track_b", "image_b", "kp_b",
            "colmap_point3d_id",
        ])
        for e in evidence:
            w.writerow([
                e.frame,
                e.cam_a, e.track_a, e.image_a, e.kp_a,
                e.cam_b, e.track_b, e.image_b, e.kp_b,
                e.colmap_point3d_id,
            ])


def save_global_links(links, path):
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "cam_a", "track_a", "cam_b", "track_b", "support_frames"
        ])
        w.writerows(links)


# =============================================================================
# Main
# =============================================================================

def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("root")
    ap.add_argument("--colmap", default="colmap")
    ap.add_argument("--colmap-model", default=None)
    ap.add_argument("--output", default=None)

    ap.add_argument("--force", action="store_true")
    ap.add_argument("--use-gpu", type=int, default=1)

    # Temporal matching.
    ap.add_argument("--temporal-window", type=int, default=15)
    ap.add_argument("--temporal-all-pairs", action="store_true")
    ap.add_argument("--sift-max-ratio", type=float, default=0.8)

    # Pixel-motion test.
    ap.add_argument("--min-track-observations", type=int, default=6)
    ap.add_argument("--min-foreground-fraction", type=float, default=0.8)
    ap.add_argument("--min-dynamic-span", type=int, default=15)
    ap.add_argument("--noise-floor-px", type=float, default=0.25)
    ap.add_argument("--static-sigma", type=float, default=2.5)
    ap.add_argument("--dynamic-sigma", type=float, default=5.0)
    ap.add_argument("--min-dynamic-range-px", type=float, default=3.0)

    # COLMAP geometry.
    ap.add_argument("--max-reproj-error", type=float, default=2.0)
    ap.add_argument("--min-tri-angle", type=float, default=1.0)
    ap.add_argument("--min-global-support-frames", type=int, default=3)

    args = ap.parse_args()

    root = Path(args.root)
    image_path = find_rgb_parent(root)
    original_model = find_colmap_model(root, args.colmap_model)

    out = (
        Path(args.output)
        if args.output
        else root / "colmap_native_dynamic"
    )

    if args.force and out.exists():
        shutil.rmtree(out)

    out.mkdir(parents=True, exist_ok=True)
    logs = out / "logs"
    cli = ColmapCLI(args.colmap, logs)

    # -------------------------------------------------------------------------
    # Original trusted calibration -> TXT.
    # -------------------------------------------------------------------------
    original_txt = out / "original_model_txt"
    if not original_txt.exists() or args.force:
        convert_model_to_txt(
            cli, original_model, original_txt, "00_original"
        )

    original_cameras = parse_cameras_txt(original_txt / "cameras.txt")
    original_images = parse_images_txt_poses(original_txt / "images.txt")
    pose_source_by_cam = choose_original_pose_per_camera(
        original_images, original_cameras
    )

    if not pose_source_by_cam:
        raise RuntimeError(
            "Could not map original registered images to cam_XXX folders."
        )

    print("[CALIBRATION] physical cameras:")
    for cam in sorted(pose_source_by_cam):
        pose, camera = pose_source_by_cam[cam]
        print(
            f"  {cam}: pose from '{pose.name}', "
            f"{camera.model} {camera.width}x{camera.height}"
        )

    # -------------------------------------------------------------------------
    # MASTER COLMAP feature database.
    # -------------------------------------------------------------------------
    master_db = out / "master.db"

    if not master_db.exists() or args.force:
        if master_db.exists():
            master_db.unlink()

        feat_help = cli.help("feature_extractor")
        feat_args = [
            "--database_path", master_db,
            "--image_path", image_path,
        ]

        if "--camera_mode" in feat_help:
            feat_args += ["--camera_mode", "2"]  # PER_FOLDER
        elif "ImageReader.single_camera_per_folder" in feat_help:
            feat_args += ["--ImageReader.single_camera_per_folder", "1"]

        if "FeatureExtraction.use_gpu" in feat_help:
            feat_args += ["--FeatureExtraction.use_gpu", args.use_gpu]
        elif "SiftExtraction.use_gpu" in feat_help:
            feat_args += ["--SiftExtraction.use_gpu", args.use_gpu]

        cli.run(
            "feature_extractor",
            feat_args,
            "01_feature_extractor.log",
        )

        # Replace inferred intrinsics with trusted original calibration.
        update_database_cameras_from_original(
            master_db,
            pose_source_by_cam,
        )

    images = db_images(master_db)
    print(f"[MASTER] {len(images)} video images in COLMAP database")

    # -------------------------------------------------------------------------
    # Expand known fixed camera poses to every video frame.
    # -------------------------------------------------------------------------
    expanded_txt = out / "expanded_known_pose_model_txt"
    expanded_bin = out / "expanded_known_pose_model"

    build_expanded_known_pose_model(
        master_db,
        pose_source_by_cam,
        expanded_txt,
    )

    # Let COLMAP itself validate/read/write the generated model.
    convert_model_to_bin(
        cli,
        expanded_txt,
        expanded_bin,
        "02_expanded_model",
    )

    # -------------------------------------------------------------------------
    # TEMPORAL COLMAP SIFT matching.
    # -------------------------------------------------------------------------
    temporal_pairs = out / "temporal_pairs.txt"
    write_temporal_pairs(
        images,
        temporal_pairs,
        args.temporal_window,
        args.temporal_all_pairs,
    )

    temporal_db = out / "temporal.db"
    if not temporal_db.exists() or args.force:
        shutil.copy2(master_db, temporal_db)

        matcher_help = cli.help("matches_importer")
        match_args = [
            "--database_path", temporal_db,
            "--match_list_path", temporal_pairs,
            "--match_type", "pairs",
        ]

        if "SiftMatching.max_ratio" in matcher_help:
            match_args += ["--SiftMatching.max_ratio", args.sift_max_ratio]
        if "SiftMatching.cross_check" in matcher_help:
            match_args += ["--SiftMatching.cross_check", "1"]

        # We want COLMAP's raw SIFT matches for temporal tracking. A static
        # camera + moving object is not a normal two-view epipolar problem.
        if "FeatureMatching.skip_geometric_verification" in matcher_help:
            match_args += ["--FeatureMatching.skip_geometric_verification", "1"]

        if "FeatureMatching.use_gpu" in matcher_help:
            match_args += ["--FeatureMatching.use_gpu", args.use_gpu]
        elif "SiftMatching.use_gpu" in matcher_help:
            match_args += ["--SiftMatching.use_gpu", args.use_gpu]

        cli.run(
            "matches_importer",
            match_args,
            "03_temporal_matches.log",
        )

    tracks_by_cam, temporal_obs_lookup = build_temporal_tracks(
        root=root,
        temporal_db=temporal_db,
        min_track_obs=args.min_track_observations,
        min_foreground_fraction=args.min_foreground_fraction,
        min_dynamic_span=args.min_dynamic_span,
        static_sigma=args.static_sigma,
        dynamic_sigma=args.dynamic_sigma,
        min_dynamic_range_px=args.min_dynamic_range_px,
        noise_floor_px=args.noise_floor_px,
    )

    temporal_summary, temporal_obs_csv = save_temporal_tracks(
        tracks_by_cam, out
    )

    # -------------------------------------------------------------------------
    # MULTIVIEW same-frame COLMAP matching only.
    # -------------------------------------------------------------------------
    multiview_pairs = out / "multiview_pairs.txt"
    write_multiview_pairs(images, multiview_pairs)

    multiview_db = out / "multiview.db"
    if not multiview_db.exists() or args.force:
        shutil.copy2(master_db, multiview_db)

        matcher_help = cli.help("matches_importer")
        match_args = [
            "--database_path", multiview_db,
            "--match_list_path", multiview_pairs,
            "--match_type", "pairs",
        ]

        if "SiftMatching.max_ratio" in matcher_help:
            match_args += ["--SiftMatching.max_ratio", args.sift_max_ratio]
        if "SiftMatching.cross_check" in matcher_help:
            match_args += ["--SiftMatching.cross_check", "1"]

        # First store raw COLMAP SIFT matches.
        if "FeatureMatching.skip_geometric_verification" in matcher_help:
            match_args += ["--FeatureMatching.skip_geometric_verification", "1"]

        if "FeatureMatching.use_gpu" in matcher_help:
            match_args += ["--FeatureMatching.use_gpu", args.use_gpu]
        elif "SiftMatching.use_gpu" in matcher_help:
            match_args += ["--SiftMatching.use_gpu", args.use_gpu]

        cli.run(
            "matches_importer",
            match_args,
            "04_multiview_raw_matches.log",
        )

        # Trusted COLMAP geometric verification.
        if cli.has_command("guided_geometric_verifier"):
            gh = cli.help("guided_geometric_verifier")
            gargs = ["--database_path", multiview_db]
            if "--input_path" in gh:
                gargs += ["--input_path", expanded_bin]
            elif "--input_model_path" in gh:
                gargs += ["--input_model_path", expanded_bin]
            else:
                gargs = None

            if gargs is not None:
                if "TwoViewGeometry.max_error" in gh:
                    gargs += ["--TwoViewGeometry.max_error", args.max_reproj_error]

                cli.run(
                    "guided_geometric_verifier",
                    gargs,
                    "05_guided_geometric_verifier.log",
                )
            else:
                cli.run(
                    "geometric_verifier",
                    [
                        "--database_path", multiview_db,
                        "--TwoViewGeometry.max_error", args.max_reproj_error,
                    ],
                    "05_geometric_verifier.log",
                )
        else:
            cli.run(
                "geometric_verifier",
                [
                    "--database_path", multiview_db,
                    "--TwoViewGeometry.max_error", args.max_reproj_error,
                ],
                "05_geometric_verifier.log",
            )

    # -------------------------------------------------------------------------
    # Official COLMAP point triangulation with trusted known poses.
    # -------------------------------------------------------------------------
    triangulated = out / "triangulated_model"

    if not triangulated.exists() or args.force:
        if triangulated.exists():
            shutil.rmtree(triangulated)
        triangulated.mkdir()

        th = cli.help("point_triangulator")
        targs = [
            "--database_path", multiview_db,
            "--image_path", image_path,
            "--input_path", expanded_bin,
            "--output_path", triangulated,
        ]

        if "Mapper.tri_ignore_two_view_tracks" in th:
            targs += ["--Mapper.tri_ignore_two_view_tracks", "0"]
        if "Mapper.tri_min_angle" in th:
            targs += ["--Mapper.tri_min_angle", args.min_tri_angle]
        if "Mapper.filter_max_reproj_error" in th:
            targs += ["--Mapper.filter_max_reproj_error", args.max_reproj_error]
        if "Mapper.filter_min_tri_angle" in th:
            targs += ["--Mapper.filter_min_tri_angle", args.min_tri_angle]

        cli.run(
            "point_triangulator",
            targs,
            "06_point_triangulator.log",
        )

    # -------------------------------------------------------------------------
    # Official COLMAP BA. FIX cameras, refine only points.
    # -------------------------------------------------------------------------
    ba_model = out / "bundle_adjusted_model"

    if not ba_model.exists() or args.force:
        if ba_model.exists():
            shutil.rmtree(ba_model)
        ba_model.mkdir()

        bh = cli.help("bundle_adjuster")
        bargs = [
            "--input_path", triangulated,
            "--output_path", ba_model,
        ]

        for opt in (
            "BundleAdjustment.refine_focal_length",
            "BundleAdjustment.refine_principal_point",
            "BundleAdjustment.refine_extra_params",
        ):
            if opt in bh:
                bargs += [f"--{opt}", "0"]

        # Old COLMAP:
        if "BundleAdjustment.refine_extrinsics" in bh:
            bargs += ["--BundleAdjustment.refine_extrinsics", "0"]

        # New rig-aware COLMAP:
        if "BundleAdjustment.refine_rig_from_world" in bh:
            bargs += ["--BundleAdjustment.refine_rig_from_world", "0"]
        if "BundleAdjustment.refine_sensor_from_rig" in bh:
            bargs += ["--BundleAdjustment.refine_sensor_from_rig", "0"]

        if "BundleAdjustment.refine_points3D" in bh:
            bargs += ["--BundleAdjustment.refine_points3D", "1"]

        cli.run(
            "bundle_adjuster",
            bargs,
            "07_bundle_adjuster_fixed_cameras.log",
        )

    # -------------------------------------------------------------------------
    # Official COLMAP point filtering.
    # -------------------------------------------------------------------------
    filtered = out / "filtered_model"

    if not filtered.exists() or args.force:
        if filtered.exists():
            shutil.rmtree(filtered)
        filtered.mkdir()

        fh = cli.help("point_filtering")
        fargs = [
            "--input_path", ba_model,
            "--output_path", filtered,
        ]
        if "--min_track_len" in fh:
            fargs += ["--min_track_len", "2"]
        if "--max_reproj_error" in fh:
            fargs += ["--max_reproj_error", args.max_reproj_error]
        if "--min_tri_angle" in fh:
            fargs += ["--min_tri_angle", args.min_tri_angle]

        cli.run(
            "point_filtering",
            fargs,
            "08_point_filtering.log",
        )

    # -------------------------------------------------------------------------
    # Read the final COLMAP associations back.
    # -------------------------------------------------------------------------
    filtered_txt = out / "filtered_model_txt"
    convert_model_to_txt(
        cli,
        filtered,
        filtered_txt,
        "09_filtered",
    )

    assignments = parse_images_txt_point3d_assignments(
        filtered_txt / "images.txt"
    )

    evidence = collect_colmap_dynamic_evidence(
        multiview_db,
        temporal_obs_lookup,
        assignments,
    )

    evidence_csv = out / "colmap_dynamic_multiview_evidence.csv"
    save_global_evidence(evidence, evidence_csv)

    links = aggregate_global_links(
        evidence,
        args.min_global_support_frames,
    )
    links_csv = out / "global_dynamic_links.csv"
    save_global_links(links, links_csv)

    print("\n" + "=" * 80)
    print("COLMAP-NATIVE DYNAMIC RESULT")
    print("=" * 80)

    n_dyn = sum(
        t.label == "DYNAMIC"
        for ts in tracks_by_cam.values()
        for t in ts
    )
    n_static = sum(
        t.label == "STATIC"
        for ts in tracks_by_cam.values()
        for t in ts
    )
    n_unc = sum(
        t.label == "UNCERTAIN"
        for ts in tracks_by_cam.values()
        for t in ts
    )

    print(f"Pixel-space temporal STATIC tracks:    {n_static}")
    print(f"Pixel-space temporal UNCERTAIN tracks: {n_unc}")
    print(f"Pixel-space temporal DYNAMIC tracks:   {n_dyn}")
    print(f"COLMAP-triangulated dynamic evidence:  {len(evidence)}")
    print(f"Repeated global dynamic track links:   {len(links)}")
    print()
    print(f"Temporal summary: {temporal_summary}")
    print(f"Temporal samples: {temporal_obs_csv}")
    print(f"COLMAP evidence:  {evidence_csv}")
    print(f"Global links:     {links_csv}")
    print(f"COLMAP logs:      {logs}")


if __name__ == "__main__":
    main()
