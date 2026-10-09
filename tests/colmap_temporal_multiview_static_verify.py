#!/usr/bin/env python3
"""
COLMAP temporal/multiview static-dynamic verification for fixed cameras.

Core idea
=========
The two evidence directions are kept separate:

1) SAME CAMERA, MULTI-FRAME
   COLMAP RootSIFT matches build temporal 2D feature tracks.
   Because the camera is fixed, a static world point must remain at one pixel.

2) SAME FRAME, MULTI-CAMERA
   COLMAP RootSIFT gives cross-camera correspondences. At a reference frame,
   official COLMAP point_triangulator + fixed-camera bundle adjustment + point
   filtering produce trusted 3D anchors X0.

For every later temporal observation u_a(t) of an anchored point:
  - compare u_a(t) with the source-camera static pixel from the reference frame;
  - use the SAME-FRAME multiview SIFT matches of u_a(t) to find the point in
    the other cameras;
  - compare those observed pixels with the static projections of X0;
  - if only the source camera leaves the static prediction while the other
    cameras remain static -> MISTRACK;
  - if multiple cameras leave the static prediction coherently -> dynamic
    candidate;
  - triangulate those candidate observations independently at each frame with
    official COLMAP using the known fixed camera poses. Only candidates that
    survive triangulation + fixed-camera BA + point filtering are DYNAMIC.

No depth. No pycolmap. No Python triangulator.

The only projection implemented in Python is the standard COLMAP PINHOLE /
SIMPLE_PINHOLE projection of an already-COLMAP-triangulated X0. USB_100109 uses
PINHOLE cameras.

Example
=======
python colmap_temporal_multiview_static_verify.py \
    /home/samuelemara/Joint3DGS/RigEuler3DGS/input/USB_100109 \
    --colmap /usr/local/bin/colmap \
    --force
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
            try:
                tail = "\n".join(log_path.read_text(errors="replace").splitlines()[-60:])
            except Exception:
                tail = ""
            raise RuntimeError(
                f"COLMAP command failed ({p.returncode}): {' '.join(cmd)}\n"
                f"Log: {log_path}\n\n{tail}"
            )
        return log_path


# =============================================================================
# Dataset / COLMAP text-model helpers
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

    def valid(p: Path):
        return (
            ((p / "cameras.bin").exists() and (p / "images.bin").exists())
            or ((p / "cameras.txt").exists() and (p / "images.txt").exists())
        )

    for p in candidates:
        if valid(p):
            return p
    raise FileNotFoundError("Could not locate calibrated COLMAP sparse model")


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
            i += 1  # POINTS2D line
    return out


def choose_original_pose_per_camera(images, cameras):
    grouped = defaultdict(list)
    for im in images.values():
        try:
            cam = camera_name_from_image_name(im.name)
        except RuntimeError:
            continue
        score = 0 if "frame_000000" in im.name else 1000
        score += len(im.name)
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
            raise RuntimeError(f"Unsupported COLMAP camera model ID mapping: {c.model}")
        con.execute(
            "UPDATE cameras SET model=?, width=?, height=?, params=?, prior_focal_length=1 "
            "WHERE camera_id=?",
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


# =============================================================================
# Known fixed-pose model expanded to all video frames
# =============================================================================

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
# Pair files
# =============================================================================

def write_all_temporal_pairs(images, path: Path):
    by_cam = defaultdict(list)
    for iid, rec in images.items():
        cam = camera_name_from_image_name(rec["name"])
        by_cam[cam].append((frame_number(rec["name"]), iid, rec["name"]))

    n = 0
    with open(path, "w") as f:
        for cam in sorted(by_cam):
            seq = sorted(by_cam[cam])
            for i in range(len(seq)):
                for j in range(i + 1, len(seq)):
                    f.write(f"{seq[i][2]} {seq[j][2]}\n")
                    n += 1
    print(f"[PAIRS] temporal all-pairs: {n} -> {path}")


def write_same_frame_multiview_pairs(images, path: Path):
    by_frame = defaultdict(list)
    for iid, rec in images.items():
        by_frame[frame_number(rec["name"])].append(
            (camera_name_from_image_name(rec["name"]), iid, rec["name"])
        )

    n = 0
    with open(path, "w") as f:
        for frame in sorted(by_frame):
            views = sorted(by_frame[frame])
            for i in range(len(views)):
                for j in range(i + 1, len(views)):
                    f.write(f"{views[i][2]} {views[j][2]}\n")
                    n += 1
    print(f"[PAIRS] same-frame multiview: {n} -> {path}")


# =============================================================================
# Foreground mask
# =============================================================================

def read_mask_value(root: Path, cam: str, frame: int, x: float, y: float):
    p = root / "mask" / cam / f"frame_{frame:06d}.png"
    if not p.exists():
        return True
    m = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
    if m is None:
        return True
    xi, yi = int(round(x)), int(round(y))
    return 0 <= xi < m.shape[1] and 0 <= yi < m.shape[0] and m[yi, xi] > 0


# =============================================================================
# Same-camera temporal tracks
# =============================================================================

class ConflictDSU:
    def __init__(self):
        self.parent = {}
        self.rank = {}
        self.frames = {}

    def add(self, node, frame):
        if node in self.parent:
            return
        self.parent[node] = node
        self.rank[node] = 0
        self.frames[node] = {frame}

    def find(self, x):
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])
        return self.parent[x]

    def component_size(self, x):
        return len(self.frames[self.find(x)])

    def can_union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        return ra == rb or self.frames[ra].isdisjoint(self.frames[rb])

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return True
        if not self.can_union(a, b):
            return False
        if self.rank[ra] < self.rank[rb]:
            ra, rb = rb, ra
        self.parent[rb] = ra
        self.frames[ra] |= self.frames[rb]
        if self.rank[ra] == self.rank[rb]:
            self.rank[ra] += 1
        return True

    def groups(self):
        out = defaultdict(list)
        for x in self.parent:
            out[self.find(x)].append(x)
        return list(out.values())


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
    frame_span: int
    static_rms_px: float
    coherent_range_px: float
    max_consecutive_speed_px_per_frame: float


def temporal_metrics(obs):
    obs = sorted(obs, key=lambda o: o.frame)
    pts = np.asarray([[o.x, o.y] for o in obs], dtype=np.float64)
    center = np.median(pts, axis=0)
    rms = float(np.sqrt(np.mean(np.sum((pts - center) ** 2, axis=1))))
    if len(pts) > 1:
        D = np.linalg.norm(pts[:, None, :] - pts[None, :, :], axis=2)
        coherent_range = float(np.max(D))
        speeds = []
        for a, b in zip(obs[:-1], obs[1:]):
            dt = max(1, b.frame - a.frame)
            speeds.append(math.hypot(b.x - a.x, b.y - a.y) / dt)
        max_speed = float(max(speeds)) if speeds else 0.0
    else:
        coherent_range = 0.0
        max_speed = 0.0
    span = obs[-1].frame - obs[0].frame + 1
    return span, rms, coherent_range, max_speed


def build_temporal_tracks(
    root: Path,
    temporal_db: Path,
    min_track_obs: int,
    min_foreground_fraction: float,
    max_link_gap: int,
):
    images = db_images(temporal_db)
    keypoints = db_keypoints(temporal_db)
    image_frame = {iid: frame_number(r["name"]) for iid, r in images.items()}
    image_cam = {iid: camera_name_from_image_name(r["name"]) for iid, r in images.items()}

    dsus = defaultdict(ConflictDSU)
    edges = []

    for ia, ib, matches in db_matches(temporal_db, "matches"):
        if image_cam[ia] != image_cam[ib]:
            continue
        fa, fb = image_frame[ia], image_frame[ib]
        gap = abs(fa - fb)
        if gap <= 0 or gap > max_link_gap:
            continue
        cam = image_cam[ia]
        for ka, kb in matches:
            a, b = (ia, int(ka)), (ib, int(kb))
            dsus[cam].add(a, fa)
            dsus[cam].add(b, fb)
            edges.append((gap, cam, a, b))

    # Very important: longer-gap matches can bridge a missed detection but are
    # not allowed to merge two already-established trajectories. This avoids a
    # single long-baseline SIFT error merging two physical material points.
    edges.sort(key=lambda e: e[0])
    for gap, cam, a, b in edges:
        dsu = dsus[cam]
        if not dsu.can_union(a, b):
            continue
        if gap > 1 and dsu.component_size(a) > 1 and dsu.component_size(b) > 1:
            continue
        dsu.union(a, b)

    tracks_by_cam = defaultdict(list)
    obs_lookup = {}

    for cam in sorted(dsus):
        groups = []
        for g in dsus[cam].groups():
            obs = []
            fg = 0
            for iid, kp in g:
                x, y = keypoints[iid][kp]
                fr = image_frame[iid]
                inside = read_mask_value(root, cam, fr, x, y)
                fg += int(inside)
                if inside:
                    obs.append(Observation(iid, fr, kp, float(x), float(y)))
            if len(g) == 0 or fg / len(g) < min_foreground_fraction:
                continue
            if len(obs) < min_track_obs:
                continue
            obs.sort(key=lambda o: o.frame)
            groups.append(obs)

        groups.sort(key=lambda g: (-len(g), g[0].frame))
        for tid, obs in enumerate(groups):
            span, rms, rng, vmax = temporal_metrics(obs)
            tr = TemporalTrack(cam, tid, obs, span, rms, rng, vmax)
            tracks_by_cam[cam].append(tr)
            for o in obs:
                obs_lookup[(o.image_id, o.kp_idx)] = tr

        print(f"[TEMPORAL] {cam}: {len(tracks_by_cam[cam])} tracks")

    return tracks_by_cam, obs_lookup


def save_temporal_tracks(tracks_by_cam, out: Path):
    summary = out / "temporal_tracks.csv"
    observations = out / "temporal_track_observations.csv"

    with open(summary, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "camera", "track_id", "num_observations", "start_frame", "end_frame",
            "frame_span", "static_rms_px", "coherent_range_px",
            "max_consecutive_speed_px_per_frame",
        ])
        for cam in sorted(tracks_by_cam):
            for t in tracks_by_cam[cam]:
                w.writerow([
                    cam, t.track_id, len(t.observations), t.observations[0].frame,
                    t.observations[-1].frame, t.frame_span, t.static_rms_px,
                    t.coherent_range_px, t.max_consecutive_speed_px_per_frame,
                ])

    with open(observations, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["camera", "track_id", "image_id", "frame", "kp_idx", "x", "y"])
        for cam in sorted(tracks_by_cam):
            for t in tracks_by_cam[cam]:
                for o in t.observations:
                    w.writerow([cam, t.track_id, o.image_id, o.frame, o.kp_idx, o.x, o.y])

    return summary, observations


# =============================================================================
# COLMAP model readback / projection
# =============================================================================

def parse_images_txt_point3d_assignments(path: Path):
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
        if i >= len(lines):
            break
        pts = lines[i].strip().split()
        i += 1
        if not pts:
            continue
        if len(pts) % 3 != 0:
            raise RuntimeError(f"Malformed POINTS2D line for image {iid}")
        for kp in range(len(pts) // 3):
            pid = int(pts[3 * kp + 2])
            if pid >= 0:
                out[(iid, kp)] = pid
    return out


def parse_points3D_txt(path: Path):
    out = {}
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        p = line.split()
        if len(p) < 8:
            continue
        pid = int(p[0])
        out[pid] = {
            "xyz": np.asarray([float(p[1]), float(p[2]), float(p[3])], dtype=np.float64),
            "error": float(p[7]),
        }
    return out


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
    raise RuntimeError(
        f"Static projection supports PINHOLE/SIMPLE_PINHOLE; got {camera.model}"
    )


# =============================================================================
# Official COLMAP known-pose triangulation + fixed-camera BA + filtering
# =============================================================================

def run_known_pose_pipeline(
    cli: ColmapCLI,
    database_path: Path,
    image_path: Path,
    expanded_model: Path,
    output_root: Path,
    log_prefix: str,
    max_reproj_error: float,
    min_tri_angle: float,
    min_track_len: int,
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
        args += ["--Mapper.tri_min_angle", min_tri_angle]
    if "Mapper.filter_max_reproj_error" in h:
        args += ["--Mapper.filter_max_reproj_error", max_reproj_error]
    if "Mapper.filter_min_tri_angle" in h:
        args += ["--Mapper.filter_min_tri_angle", min_tri_angle]
    cli.run("point_triangulator", args, f"{log_prefix}_triangulator.log")

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
    cli.run("bundle_adjuster", args, f"{log_prefix}_bundle_adjuster.log")

    filtered.mkdir()
    h = cli.help("point_filtering")
    args = ["--input_path", adjusted, "--output_path", filtered]
    if "--min_track_len" in h:
        args += ["--min_track_len", min_track_len]
    if "--max_reproj_error" in h:
        args += ["--max_reproj_error", max_reproj_error]
    if "--min_tri_angle" in h:
        args += ["--min_tri_angle", min_tri_angle]
    cli.run("point_filtering", args, f"{log_prefix}_point_filtering.log")

    convert_model_to_txt(
        cli, filtered, filtered_txt, f"{log_prefix}_model_converter.log"
    )
    assignments = parse_images_txt_point3d_assignments(filtered_txt / "images.txt")
    points = parse_points3D_txt(filtered_txt / "points3D.txt")
    return assignments, points, filtered_txt


# =============================================================================
# Multiview match graph / reference anchors
# =============================================================================

def build_match_adjacency(db_path: Path, table="matches"):
    adj = defaultdict(list)
    for ia, ib, matches in db_matches(db_path, table):
        for ka, kb in matches:
            a = (ia, int(ka))
            b = (ib, int(kb))
            adj[a].append(b)
            adj[b].append(a)
    return adj


def write_reference_match_file(
    root: Path,
    raw_multiview_db: Path,
    reference_frame: int,
    path: Path,
):
    images = db_images(raw_multiview_db)
    keypoints = db_keypoints(raw_multiview_db)
    n_pairs = 0
    n_matches = 0
    with open(path, "w") as f:
        for ia, ib, matches in db_matches(raw_multiview_db, "matches"):
            ra, rb = images[ia], images[ib]
            if frame_number(ra["name"]) != reference_frame:
                continue
            if frame_number(rb["name"]) != reference_frame:
                continue
            ca = camera_name_from_image_name(ra["name"])
            cb = camera_name_from_image_name(rb["name"])
            keep = []
            for ka, kb in matches:
                ka, kb = int(ka), int(kb)
                xa, ya = keypoints[ia][ka]
                xb, yb = keypoints[ib][kb]
                if not read_mask_value(root, ca, reference_frame, xa, ya):
                    continue
                if not read_mask_value(root, cb, reference_frame, xb, yb):
                    continue
                keep.append((ka, kb))
            if not keep:
                continue
            f.write(f"{ra['name']} {rb['name']}\n")
            for ka, kb in keep:
                f.write(f"{ka} {kb}\n")
            f.write("\n")
            n_pairs += 1
            n_matches += len(keep)
    print(
        f"[REFERENCE] frame={reference_frame}: {n_matches} foreground SIFT matches "
        f"across {n_pairs} camera pairs"
    )
    return n_pairs, n_matches


@dataclass
class ReferenceAnchor:
    anchor_id: int
    reference_point3d_id: int
    X0: np.ndarray
    reference_frame: int
    initial_observations: dict   # cam -> Observation
    source_tracks: dict          # cam -> TemporalTrack
    static_uv: dict              # cam -> np.ndarray(2)


def build_reference_anchors(
    reference_frame: int,
    assignments,
    points3d,
    images,
    keypoints,
    temporal_obs_lookup,
    pose_source_by_cam,
):
    by_pid = defaultdict(list)
    for (iid, kp), pid in assignments.items():
        if iid in images and frame_number(images[iid]["name"]) == reference_frame:
            by_pid[pid].append((iid, kp))

    anchors = []
    for pid, nodes in sorted(by_pid.items()):
        if pid not in points3d:
            continue
        cams = {camera_name_from_image_name(images[iid]["name"]) for iid, _ in nodes}
        if len(cams) < 2:
            continue

        initial = {}
        sources = {}
        for iid, kp in nodes:
            cam = camera_name_from_image_name(images[iid]["name"])
            x, y = keypoints[iid][kp]
            initial.setdefault(cam, Observation(iid, reference_frame, kp, float(x), float(y)))
            tr = temporal_obs_lookup.get((iid, kp))
            if tr is not None:
                sources.setdefault(cam, tr)
        if not sources:
            continue

        X0 = points3d[pid]["xyz"].copy()
        static_uv = {}
        for cam, (pose, camera) in pose_source_by_cam.items():
            # For a camera that actually observed the anchor at t0, its measured
            # reference keypoint is the exact static pixel hypothesis. For a
            # camera that did not observe it, project the trusted COLMAP X0.
            if cam in initial:
                static_uv[cam] = np.array([initial[cam].x, initial[cam].y], dtype=np.float64)
            else:
                uv = project_world(pose, camera, X0)
                if uv is not None:
                    static_uv[cam] = uv

        anchors.append(
            ReferenceAnchor(
                len(anchors), pid, X0, reference_frame, initial, sources, static_uv
            )
        )

    print(f"[REFERENCE] usable anchors with temporal tracks: {len(anchors)}")
    return anchors


# =============================================================================
# Static-hypothesis verification
# =============================================================================

def verify_static_hypothesis(
    root: Path,
    anchors,
    temporal_db: Path,
    raw_multiview_db: Path,
    static_px: float,
    dynamic_px: float,
    min_crossview_targets: int,
    min_dynamic_cameras: int,
):
    images = db_images(raw_multiview_db)
    keypoints = db_keypoints(raw_multiview_db)
    multiview_adj = build_match_adjacency(raw_multiview_db, "matches")
    temporal_adj = build_match_adjacency(temporal_db, "matches")

    frame_rows = []
    target_rows = []
    dynamic_candidates = []

    for a in anchors:
        for source_cam, track in a.source_tracks.items():
            if source_cam not in a.initial_observations or source_cam not in a.static_uv:
                continue

            ref_ob = a.initial_observations[source_cam]
            ref_node = (ref_ob.image_id, ref_ob.kp_idx)
            direct_from_reference = set(temporal_adj.get(ref_node, []))
            static_source = a.static_uv[source_cam]

            for o in track.observations:
                source_node = (o.image_id, o.kp_idx)
                direct_identity = int(
                    source_node == ref_node or source_node in direct_from_reference
                )
                source_err = float(
                    np.linalg.norm(np.array([o.x, o.y]) - static_source)
                )

                # Exact same-frame COLMAP SIFT correspondences of this source
                # feature. Cross-check + one SIFT orientation should give at most
                # one feature per target camera. If not, that camera is ambiguous
                # and is ignored for this frame.
                candidates_by_cam = defaultdict(list)
                for ti, tk in multiview_adj.get(source_node, []):
                    rec = images.get(ti)
                    if rec is None or frame_number(rec["name"]) != o.frame:
                        continue
                    tc = camera_name_from_image_name(rec["name"])
                    if tc == source_cam or tc not in a.static_uv:
                        continue
                    x, y = keypoints[ti][tk]
                    if not read_mask_value(root, tc, o.frame, x, y):
                        continue
                    candidates_by_cam[tc].append((ti, int(tk), float(x), float(y)))

                per_target = {}
                for tc, vals in candidates_by_cam.items():
                    if len(vals) != 1:
                        continue
                    ti, tk, x, y = vals[0]
                    err = float(np.linalg.norm(np.array([x, y]) - a.static_uv[tc]))
                    per_target[tc] = (ti, tk, x, y, err)

                errs = [v[4] for v in per_target.values()]
                n_targets = len(errs)
                n_static_targets = sum(e <= static_px for e in errs)
                n_dynamic_targets = sum(e >= dynamic_px for e in errs)
                source_static = source_err <= static_px
                source_dynamic = source_err >= dynamic_px
                n_dynamic_cams = int(source_dynamic) + n_dynamic_targets

                if n_targets < min_crossview_targets:
                    label = "UNSUPPORTED"
                elif source_static and n_static_targets == n_targets:
                    label = "STATIC"
                elif n_dynamic_cams >= min_dynamic_cameras and direct_identity:
                    label = "DYNAMIC_CANDIDATE"
                elif n_dynamic_cams >= min_dynamic_cameras and not direct_identity:
                    # The apparent motion is multiview-consistent, but the source
                    # observation is not directly RootSIFT-matched to its initial
                    # feature. Do not confuse an identity switch with dynamics.
                    label = "UNCERTAIN_IDENTITY"
                elif source_dynamic and n_static_targets == n_targets:
                    # Exactly the failure mode we wanted to detect: the temporal
                    # source track moved, while other cameras still observe the
                    # feature at the original static projection.
                    label = "MISTRACK"
                else:
                    label = "UNCERTAIN"

                row = {
                    "anchor_id": a.anchor_id,
                    "reference_point3d_id": a.reference_point3d_id,
                    "source_cam": source_cam,
                    "source_track": track.track_id,
                    "frame": o.frame,
                    "source_image_id": o.image_id,
                    "source_kp_idx": o.kp_idx,
                    "source_x": o.x,
                    "source_y": o.y,
                    "static_source_x": float(static_source[0]),
                    "static_source_y": float(static_source[1]),
                    "source_static_error_px": source_err,
                    "direct_reference_sift_match": direct_identity,
                    "num_crossview_targets": n_targets,
                    "num_static_targets": n_static_targets,
                    "num_dynamic_targets": n_dynamic_targets,
                    "median_target_static_error_px": float(np.median(errs)) if errs else math.nan,
                    "max_target_static_error_px": max(errs) if errs else math.nan,
                    "frame_label": label,
                    "geometry_ok": 0,
                    "geometry_support_cameras": 0,
                    "dynamic_point3d_id": -1,
                    "dynamic_X": None,
                }
                frame_rows.append(row)

                obs_nodes = [(o.image_id, o.kp_idx, source_cam)]
                for tc, (ti, tk, x, y, err) in per_target.items():
                    target_rows.append({
                        "anchor_id": a.anchor_id,
                        "source_cam": source_cam,
                        "source_track": track.track_id,
                        "frame": o.frame,
                        "target_cam": tc,
                        "target_image_id": ti,
                        "target_kp_idx": tk,
                        "target_x": x,
                        "target_y": y,
                        "static_target_x": float(a.static_uv[tc][0]),
                        "static_target_y": float(a.static_uv[tc][1]),
                        "target_static_error_px": err,
                        "target_static": int(err <= static_px),
                        "target_dynamic": int(err >= dynamic_px),
                    })
                    obs_nodes.append((ti, tk, tc))

                if label == "DYNAMIC_CANDIDATE":
                    dynamic_candidates.append((row, obs_nodes))

    return frame_rows, target_rows, dynamic_candidates


def write_dynamic_candidate_match_file(raw_multiview_db: Path, candidates, path: Path):
    images = db_images(raw_multiview_db)
    pair_matches = defaultdict(set)

    for row, obs_nodes in candidates:
        si, sk = row["source_image_id"], row["source_kp_idx"]
        for ti, tk, _ in obs_nodes[1:]:
            if si < ti:
                key, match = (si, ti), (sk, tk)
            else:
                key, match = (ti, si), (tk, sk)
            pair_matches[key].add(match)

    with open(path, "w") as f:
        for (ia, ib), ms in sorted(pair_matches.items()):
            f.write(f"{images[ia]['name']} {images[ib]['name']}\n")
            for ka, kb in sorted(ms):
                f.write(f"{ka} {kb}\n")
            f.write("\n")

    n = sum(len(v) for v in pair_matches.values())
    print(f"[DYNAMIC] candidate graph: {n} edges across {len(pair_matches)} image pairs")
    return len(pair_matches), n


def apply_dynamic_geometry(candidates, assignments, points3d, min_cameras: int):
    for row, obs_nodes in candidates:
        source_pid = assignments.get((row["source_image_id"], row["source_kp_idx"]))
        if source_pid is None or source_pid not in points3d:
            continue
        cams = {
            cam
            for iid, kp, cam in obs_nodes
            if assignments.get((iid, kp)) == source_pid
        }
        if len(cams) < min_cameras:
            continue
        row["geometry_ok"] = 1
        row["geometry_support_cameras"] = len(cams)
        row["dynamic_point3d_id"] = int(source_pid)
        row["dynamic_X"] = points3d[source_pid]["xyz"].copy()


def summarize_tracks(
    anchors,
    frame_rows,
    min_dynamic_frames: int,
    min_static_frames: int,
    static_frame_fraction: float,
):
    grouped = defaultdict(list)
    for r in frame_rows:
        grouped[(r["anchor_id"], r["source_cam"], r["source_track"])].append(r)

    tracks = []
    for (aid, cam, tid), rows in sorted(grouped.items()):
        supported = [r for r in rows if r["frame_label"] != "UNSUPPORTED"]
        n_static = sum(r["frame_label"] == "STATIC" for r in rows)
        n_dyn_candidate = sum(r["frame_label"] == "DYNAMIC_CANDIDATE" for r in rows)
        n_dyn = sum(r["geometry_ok"] == 1 for r in rows)
        n_mistrack = sum(r["frame_label"] == "MISTRACK" for r in rows)
        n_uncertain = sum(
            r["frame_label"] in ("UNCERTAIN", "UNCERTAIN_IDENTITY") for r in rows
        )
        static_frac = n_static / max(len(supported), 1)

        if n_dyn >= min_dynamic_frames:
            label = "DYNAMIC"
        elif n_static >= min_static_frames and static_frac >= static_frame_fraction:
            label = "STATIC"
        elif n_mistrack >= max(2, min_dynamic_frames) and n_dyn == 0:
            label = "MISTRACK"
        else:
            label = "UNCERTAIN"

        tracks.append({
            "anchor_id": aid,
            "source_cam": cam,
            "source_track": tid,
            "final_label": label,
            "num_frames": len(rows),
            "supported_frames": len(supported),
            "static_frames": n_static,
            "dynamic_candidate_frames": n_dyn_candidate,
            "dynamic_geometry_frames": n_dyn,
            "mistrack_frames": n_mistrack,
            "uncertain_frames": n_uncertain,
            "static_frame_fraction": static_frac,
            "median_source_static_error_px": float(np.median([r["source_static_error_px"] for r in rows])),
            "max_source_static_error_px": float(np.max([r["source_static_error_px"] for r in rows])),
        })

    by_anchor = defaultdict(list)
    for t in tracks:
        by_anchor[t["anchor_id"]].append(t)
    amap = {a.anchor_id: a for a in anchors}
    globals_ = []
    for aid, vals in sorted(by_anchor.items()):
        labels = [v["final_label"] for v in vals]
        if "DYNAMIC" in labels:
            label = "DYNAMIC"
        elif labels and all(x == "STATIC" for x in labels):
            label = "STATIC"
        else:
            label = "UNCERTAIN"
        a = amap[aid]
        globals_.append({
            "anchor_id": aid,
            "reference_point3d_id": a.reference_point3d_id,
            "X0_x": float(a.X0[0]),
            "X0_y": float(a.X0[1]),
            "X0_z": float(a.X0[2]),
            "reference_cameras": ";".join(sorted(a.initial_observations)),
            "source_tracks": ";".join(f"{c}:{t.track_id}" for c, t in sorted(a.source_tracks.items())),
            "global_label": label,
            "dynamic_source_tracks": sum(x == "DYNAMIC" for x in labels),
            "static_source_tracks": sum(x == "STATIC" for x in labels),
            "uncertain_source_tracks": sum(x not in ("DYNAMIC", "STATIC") for x in labels),
        })
    return tracks, globals_


# =============================================================================
# CSV output
# =============================================================================

def write_dict_csv(path: Path, rows, fieldnames):
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        for row in rows:
            r = dict(row)
            r.pop("dynamic_X", None)
            w.writerow(r)


def save_results(out: Path, anchors, frame_rows, target_rows, tracks, globals_):
    anchors_csv = out / "reference_multiview_anchors.csv"
    with open(anchors_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "anchor_id", "reference_point3d_id", "reference_frame",
            "X0_x", "X0_y", "X0_z", "reference_cameras", "source_tracks",
        ])
        for a in anchors:
            w.writerow([
                a.anchor_id, a.reference_point3d_id, a.reference_frame,
                a.X0[0], a.X0[1], a.X0[2],
                ";".join(sorted(a.initial_observations)),
                ";".join(f"{c}:{t.track_id}" for c, t in sorted(a.source_tracks.items())),
            ])

    frame_fields = [
        "anchor_id", "reference_point3d_id", "source_cam", "source_track", "frame",
        "source_image_id", "source_kp_idx", "source_x", "source_y",
        "static_source_x", "static_source_y", "source_static_error_px",
        "direct_reference_sift_match", "num_crossview_targets", "num_static_targets",
        "num_dynamic_targets", "median_target_static_error_px",
        "max_target_static_error_px", "frame_label", "geometry_ok",
        "geometry_support_cameras", "dynamic_point3d_id",
    ]
    frame_csv = out / "static_projection_frame_evidence.csv"
    write_dict_csv(frame_csv, frame_rows, frame_fields)

    target_fields = [
        "anchor_id", "source_cam", "source_track", "frame", "target_cam",
        "target_image_id", "target_kp_idx", "target_x", "target_y",
        "static_target_x", "static_target_y", "target_static_error_px",
        "target_static", "target_dynamic",
    ]
    target_csv = out / "static_projection_target_evidence.csv"
    write_dict_csv(target_csv, target_rows, target_fields)

    tracks_csv = out / "verified_temporal_tracks.csv"
    write_dict_csv(tracks_csv, tracks, list(tracks[0].keys()) if tracks else ["anchor_id"])

    globals_csv = out / "verified_global_anchors.csv"
    write_dict_csv(globals_csv, globals_, list(globals_[0].keys()) if globals_ else ["anchor_id"])

    # One row per source verification.
    traj_csv = out / "verified_dynamic_3d_trajectory.csv"
    with open(traj_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "anchor_id", "source_cam", "source_track", "frame",
            "colmap_point3d_id", "support_cameras", "X", "Y", "Z",
        ])
        for r in frame_rows:
            if not r["geometry_ok"] or r["dynamic_X"] is None:
                continue
            X = r["dynamic_X"]
            w.writerow([
                r["anchor_id"], r["source_cam"], r["source_track"], r["frame"],
                r["dynamic_point3d_id"], r["geometry_support_cameras"], X[0], X[1], X[2],
            ])

    # Deduplicated material trajectory: one COLMAP point per anchor/frame when
    # multiple source cameras verify the same dynamic observation.
    global_traj_csv = out / "verified_global_dynamic_3d_trajectory.csv"
    seen = set()
    with open(global_traj_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "anchor_id", "frame", "colmap_point3d_id", "support_cameras", "X", "Y", "Z"
        ])
        for r in frame_rows:
            if not r["geometry_ok"] or r["dynamic_X"] is None:
                continue
            key = (r["anchor_id"], r["frame"], r["dynamic_point3d_id"])
            if key in seen:
                continue
            seen.add(key)
            X = r["dynamic_X"]
            w.writerow([
                r["anchor_id"], r["frame"], r["dynamic_point3d_id"],
                r["geometry_support_cameras"], X[0], X[1], X[2],
            ])

    return [anchors_csv, frame_csv, target_csv, tracks_csv, globals_csv, traj_csv, global_traj_csv]


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

    ap.add_argument("--sift-max-ratio", type=float, default=0.8)
    ap.add_argument("--temporal-link-max-gap", type=int, default=5)
    ap.add_argument("--min-track-observations", type=int, default=6)
    ap.add_argument("--min-foreground-fraction", type=float, default=0.8)

    ap.add_argument("--reference-frame", type=int, default=0)
    ap.add_argument("--static-px", type=float, default=1.0)
    ap.add_argument("--dynamic-px", type=float, default=2.0)
    ap.add_argument("--min-crossview-targets", type=int, default=1)
    ap.add_argument("--min-dynamic-cameras", type=int, default=2)
    ap.add_argument("--min-dynamic-frames", type=int, default=3)
    ap.add_argument("--min-static-frames", type=int, default=5)
    ap.add_argument("--static-frame-fraction", type=float, default=0.8)

    ap.add_argument("--max-reproj-error", type=float, default=2.0)
    ap.add_argument("--min-tri-angle", type=float, default=1.0)
    args = ap.parse_args()

    if args.dynamic_px <= args.static_px:
        raise ValueError("--dynamic-px must be > --static-px")

    root = Path(args.root)
    image_path = find_rgb_parent(root)
    original_model = find_colmap_model(root, args.colmap_model)
    out = Path(args.output) if args.output else root / "colmap_static_projection_verify"

    if args.force and out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True, exist_ok=True)
    logs = out / "logs"
    cli = ColmapCLI(args.colmap, logs)

    # Trusted original calibration.
    original_txt = out / "original_model_txt"
    convert_model_to_txt(cli, original_model, original_txt, "00_original_to_txt.log")
    original_cameras = parse_cameras_txt(original_txt / "cameras.txt")
    original_images = parse_images_txt_poses(original_txt / "images.txt")
    pose_source_by_cam = choose_original_pose_per_camera(original_images, original_cameras)
    if not pose_source_by_cam:
        raise RuntimeError("Could not map original COLMAP images to cam_XXX cameras")

    print("[CALIBRATION]")
    for cam in sorted(pose_source_by_cam):
        pose, c = pose_source_by_cam[cam]
        print(f"  {cam}: {c.model} {c.width}x{c.height} <- {pose.name}")

    # ---------------------------------------------------------------------
    # Master RootSIFT database.
    # ---------------------------------------------------------------------
    master_db = out / "master.db"
    if master_db.exists():
        master_db.unlink()
    h = cli.help("feature_extractor")
    feat = ["--database_path", master_db, "--image_path", image_path]
    if "--camera_mode" in h:
        feat += ["--camera_mode", "2"]  # PER_FOLDER
    elif "ImageReader.single_camera_per_folder" in h:
        feat += ["--ImageReader.single_camera_per_folder", "1"]
    # One physical image location -> one SIFT feature orientation. This removes
    # the exact duplicate temporal tracks found in the previous run.
    if "SiftExtraction.max_num_orientations" in h:
        feat += ["--SiftExtraction.max_num_orientations", "1"]
    if "FeatureExtraction.use_gpu" in h:
        feat += ["--FeatureExtraction.use_gpu", args.use_gpu]
    cli.run("feature_extractor", feat, "01_feature_extractor.log")
    update_database_cameras_from_original(master_db, pose_source_by_cam)

    images = db_images(master_db)
    print(f"[MASTER] {len(images)} images")

    expanded_txt = out / "expanded_known_pose_model_txt"
    expanded_bin = out / "expanded_known_pose_model"
    build_expanded_known_pose_model(master_db, pose_source_by_cam, expanded_txt)
    convert_model_to_bin(cli, expanded_txt, expanded_bin, "02_expanded_model_to_bin.log")

    # ---------------------------------------------------------------------
    # Temporal: all same-camera frame pairs are matched so every anchored
    # observation can be directly checked against its reference descriptor.
    # Only short gaps are allowed to CREATE temporal-track identity links.
    # ---------------------------------------------------------------------
    temporal_pairs = out / "temporal_pairs.txt"
    write_all_temporal_pairs(images, temporal_pairs)
    temporal_db = out / "temporal.db"
    if temporal_db.exists():
        temporal_db.unlink()
    shutil.copy2(master_db, temporal_db)

    h = cli.help("matches_importer")
    margs = [
        "--database_path", temporal_db,
        "--match_list_path", temporal_pairs,
        "--match_type", "pairs",
    ]
    if "SiftMatching.max_ratio" in h:
        margs += ["--SiftMatching.max_ratio", args.sift_max_ratio]
    if "SiftMatching.cross_check" in h:
        margs += ["--SiftMatching.cross_check", "1"]
    if "TwoViewGeometry.min_num_inliers" in h:
        # COLMAP 3.13 otherwise clears raw match rows containing <15 matches.
        margs += ["--TwoViewGeometry.min_num_inliers", "1"]
    if "FeatureMatching.use_gpu" in h:
        margs += ["--FeatureMatching.use_gpu", args.use_gpu]
    cli.run("matches_importer", margs, "03_temporal_matches.log")

    tracks_by_cam, temporal_obs_lookup = build_temporal_tracks(
        root,
        temporal_db,
        min_track_obs=args.min_track_observations,
        min_foreground_fraction=args.min_foreground_fraction,
        max_link_gap=args.temporal_link_max_gap,
    )
    temporal_outputs = save_temporal_tracks(tracks_by_cam, out)

    # ---------------------------------------------------------------------
    # Multiview: every camera pair at each SAME frame.
    # ---------------------------------------------------------------------
    multiview_pairs = out / "multiview_pairs.txt"
    write_same_frame_multiview_pairs(images, multiview_pairs)
    raw_multiview_db = out / "multiview_raw.db"
    if raw_multiview_db.exists():
        raw_multiview_db.unlink()
    shutil.copy2(master_db, raw_multiview_db)

    h = cli.help("matches_importer")
    margs = [
        "--database_path", raw_multiview_db,
        "--match_list_path", multiview_pairs,
        "--match_type", "pairs",
    ]
    if "SiftMatching.max_ratio" in h:
        margs += ["--SiftMatching.max_ratio", args.sift_max_ratio]
    if "SiftMatching.cross_check" in h:
        margs += ["--SiftMatching.cross_check", "1"]
    if "TwoViewGeometry.min_num_inliers" in h:
        margs += ["--TwoViewGeometry.min_num_inliers", "1"]
    if "FeatureMatching.use_gpu" in h:
        margs += ["--FeatureMatching.use_gpu", args.use_gpu]
    cli.run("matches_importer", margs, "04_multiview_matches.log")

    mv_pairs, mv_matches = count_matches(raw_multiview_db, "matches")
    print(f"[MULTIVIEW] raw RootSIFT: {mv_matches} matches / {mv_pairs} image pairs")
    if mv_matches == 0:
        raise RuntimeError("COLMAP produced zero same-frame multiview SIFT matches")

    # ---------------------------------------------------------------------
    # Initial multicamera correspondences -> trusted X0 anchors.
    # ---------------------------------------------------------------------
    ref_match_file = out / "reference_multiview_matches.txt"
    _, n_ref = write_reference_match_file(
        root, raw_multiview_db, args.reference_frame, ref_match_file
    )
    if n_ref == 0:
        raise RuntimeError("No foreground multiview matches at reference frame")

    ref_db = out / "reference_multiview.db"
    if ref_db.exists():
        ref_db.unlink()
    shutil.copy2(master_db, ref_db)
    cli.run(
        "matches_importer",
        [
            "--database_path", ref_db,
            "--match_list_path", ref_match_file,
            "--match_type", "inliers",
        ],
        "05_reference_import.log",
    )

    ref_assignments, ref_points, _ = run_known_pose_pipeline(
        cli,
        ref_db,
        image_path,
        expanded_bin,
        out / "reference_colmap",
        "06_reference",
        args.max_reproj_error,
        args.min_tri_angle,
        min_track_len=2,
    )

    keypoints = db_keypoints(master_db)
    anchors = build_reference_anchors(
        args.reference_frame,
        ref_assignments,
        ref_points,
        images,
        keypoints,
        temporal_obs_lookup,
        pose_source_by_cam,
    )
    if not anchors:
        raise RuntimeError("No reference 3D anchor is connected to a temporal track")

    # ---------------------------------------------------------------------
    # Compare temporal source tracks + same-frame multiview matches against
    # the static projections of the initial X0 anchors.
    # ---------------------------------------------------------------------
    frame_rows, target_rows, dynamic_candidates = verify_static_hypothesis(
        root,
        anchors,
        temporal_db,
        raw_multiview_db,
        static_px=args.static_px,
        dynamic_px=args.dynamic_px,
        min_crossview_targets=args.min_crossview_targets,
        min_dynamic_cameras=args.min_dynamic_cameras,
    )
    print(
        f"[VERIFY] {len(frame_rows)} anchored source-frame tests; "
        f"{len(dynamic_candidates)} multiview dynamic candidates"
    )

    # ---------------------------------------------------------------------
    # Dynamic candidates are triangulated SEPARATELY at each frame. There are
    # no temporal edges in this DB. This is the official COLMAP geometry gate.
    # ---------------------------------------------------------------------
    if dynamic_candidates:
        dyn_match_file = out / "dynamic_verification_candidates.txt"
        _, n_edges = write_dynamic_candidate_match_file(
            raw_multiview_db, dynamic_candidates, dyn_match_file
        )
        if n_edges > 0:
            dyn_db = out / "dynamic_verification.db"
            if dyn_db.exists():
                dyn_db.unlink()
            shutil.copy2(master_db, dyn_db)
            cli.run(
                "matches_importer",
                [
                    "--database_path", dyn_db,
                    "--match_list_path", dyn_match_file,
                    "--match_type", "inliers",
                ],
                "07_dynamic_import.log",
            )

            dyn_assignments, dyn_points, _ = run_known_pose_pipeline(
                cli,
                dyn_db,
                image_path,
                expanded_bin,
                out / "dynamic_colmap",
                "08_dynamic",
                args.max_reproj_error,
                args.min_tri_angle,
                min_track_len=2,
            )
            apply_dynamic_geometry(
                dynamic_candidates,
                dyn_assignments,
                dyn_points,
                min_cameras=args.min_dynamic_cameras,
            )

    track_summaries, global_summaries = summarize_tracks(
        anchors,
        frame_rows,
        min_dynamic_frames=args.min_dynamic_frames,
        min_static_frames=args.min_static_frames,
        static_frame_fraction=args.static_frame_fraction,
    )
    result_files = save_results(
        out, anchors, frame_rows, target_rows, track_summaries, global_summaries
    )

    track_counts = defaultdict(int)
    for r in track_summaries:
        track_counts[r["final_label"]] += 1
    global_counts = defaultdict(int)
    for r in global_summaries:
        global_counts[r["global_label"]] += 1

    print("\n" + "=" * 80)
    print("RESULT")
    print("=" * 80)
    print(f"Reference anchors: {len(anchors)}")
    print(
        "Verified source tracks: "
        f"STATIC={track_counts['STATIC']} DYNAMIC={track_counts['DYNAMIC']} "
        f"MISTRACK={track_counts['MISTRACK']} UNCERTAIN={track_counts['UNCERTAIN']}"
    )
    print(
        "Global anchored material points: "
        f"STATIC={global_counts['STATIC']} DYNAMIC={global_counts['DYNAMIC']} "
        f"UNCERTAIN={global_counts['UNCERTAIN']}"
    )
    print(
        "COLMAP-confirmed dynamic source-frame observations: "
        f"{sum(r['geometry_ok'] for r in frame_rows)}"
    )
    print()
    for p in temporal_outputs + tuple(result_files):
        print(p)
    print(f"Logs: {logs}")


if __name__ == "__main__":
    main()
