#!/usr/bin/env python3
"""
Long-lived SIFT-seeded KLT + calibrated multi-camera correspondence test (v5).

NO DEPTH. NO PYCOLMAP. NO 3D POSITIONS ARE SAVED.

Pipeline:
  1) build same-camera RootSIFT tracks through time;
  2) classify each temporal track STATIC / UNCERTAIN / DYNAMIC from pixel motion;
  3) only DYNAMIC tracks enter cross-camera matching;
  4) for each same-frame cross-camera candidate, use known COLMAP calibration
     and COLMAP-style two-view DLT triangulation only as a PASS/FAIL check
     (cheirality + reprojection + triangulation angle); transient XYZ is discarded;
  5) accumulate repeated cross-camera agreement over the video and form final
     multi-camera dynamic-track groups;
  6) save one multi-camera panel for --viz-frame.

Expected structure:
ROOT/
  rgb/cam_000/frame_000000.png
  ...
  colmap/.../cameras.bin + images.bin   OR cameras.txt + images.txt
  mask/cam_000/frame_000000.png        # optional
"""

import argparse
import csv
import math
import struct
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


# -----------------------------------------------------------------------------
# Dataset helpers
# -----------------------------------------------------------------------------

def frame_id(path: Path) -> int:
    return int(path.stem.split("_")[-1])


def find_rgb_parent(root: Path):
    for parent in ("rgb", "video_rgb"):
        p = root / parent
        if p.is_dir():
            cams = sorted(x.name for x in p.iterdir() if x.is_dir() and x.name.startswith("cam_"))
            if cams:
                return parent, cams
    raise FileNotFoundError("Could not find rgb/cam_* or video_rgb/cam_*.")


def list_rgb_frames(root: Path, parent: str, camera: str):
    folder = root / parent / camera
    for ext in ("png", "jpg", "jpeg"):
        files = sorted(folder.glob(f"frame_*.{ext}"))
        if files:
            return files
    return []


def read_mask(root: Path, camera: str, fid: int, image_hw, disabled: bool):
    if disabled:
        return None
    folder = root / "mask" / camera
    if not folder.is_dir():
        return None
    for ext in ("png", "jpg", "jpeg"):
        p = folder / f"frame_{fid:06d}.{ext}"
        if p.exists():
            m = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
            if m is None:
                return None
            if m.shape != image_hw:
                m = cv2.resize(m, (image_hw[1], image_hw[0]), interpolation=cv2.INTER_NEAREST)
            return ((m > 0).astype(np.uint8) * 255)
    return None


# -----------------------------------------------------------------------------
# Minimal COLMAP model reader: cameras/images only, text or binary
# -----------------------------------------------------------------------------

CAMERA_MODELS = {
    0: ("SIMPLE_PINHOLE", 3),
    1: ("PINHOLE", 4),
    2: ("SIMPLE_RADIAL", 4),
    3: ("RADIAL", 5),
    4: ("OPENCV", 8),
    5: ("OPENCV_FISHEYE", 8),
    6: ("FULL_OPENCV", 12),
    7: ("FOV", 5),
    8: ("SIMPLE_RADIAL_FISHEYE", 4),
    9: ("RADIAL_FISHEYE", 5),
    10: ("THIN_PRISM_FISHEYE", 12),
}


@dataclass
class CCamera:
    camera_id: int
    model: str
    width: int
    height: int
    params: np.ndarray


@dataclass
class CImage:
    image_id: int
    qvec: np.ndarray
    tvec: np.ndarray
    camera_id: int
    name: str


def _is_model_dir(p: Path):
    return (((p / "cameras.bin").exists() and (p / "images.bin").exists()) or
            ((p / "cameras.txt").exists() and (p / "images.txt").exists()))


def find_colmap_model(root: Path, explicit=None):
    if explicit:
        p = Path(explicit)
        if not _is_model_dir(p):
            raise FileNotFoundError(f"Not a COLMAP model directory: {p}")
        return p
    for p in (
        root / "colmap",
        root / "colmap" / "sparse",
        root / "colmap" / "sparse" / "0",
        root / "sparse",
        root / "sparse" / "0",
    ):
        if _is_model_dir(p):
            return p
    base = root / "colmap"
    if base.exists():
        for p in base.rglob("*"):
            if p.is_dir() and _is_model_dir(p):
                return p
    raise FileNotFoundError("Could not locate COLMAP cameras/images model. Use --colmap-model.")


def _read_cstr(f):
    chunks = []
    while True:
        c = f.read(1)
        if c == b"":
            raise EOFError("Unexpected EOF reading COLMAP string")
        if c == b"\x00":
            return b"".join(chunks).decode("utf-8")
        chunks.append(c)


def read_cameras_bin(path):
    out = {}
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        for _ in range(n):
            cid, mid = struct.unpack("<ii", f.read(8))
            w, h = struct.unpack("<QQ", f.read(16))
            if mid not in CAMERA_MODELS:
                raise RuntimeError(f"Unsupported COLMAP camera model id {mid}")
            name, np_ = CAMERA_MODELS[mid]
            params = np.array(struct.unpack("<" + "d" * np_, f.read(8 * np_)), np.float64)
            out[cid] = CCamera(cid, name, int(w), int(h), params)
    return out


def read_images_bin(path):
    out = {}
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        for _ in range(n):
            iid = struct.unpack("<i", f.read(4))[0]
            qvec = np.array(struct.unpack("<dddd", f.read(32)), np.float64)
            tvec = np.array(struct.unpack("<ddd", f.read(24)), np.float64)
            cid = struct.unpack("<i", f.read(4))[0]
            name = _read_cstr(f)
            n2d = struct.unpack("<Q", f.read(8))[0]
            f.seek(24 * n2d, 1)
            out[iid] = CImage(iid, qvec, tvec, cid, name)
    return out


def read_cameras_txt(path):
    out = {}
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        p = line.split()
        cid, model, w, h = int(p[0]), p[1], int(p[2]), int(p[3])
        out[cid] = CCamera(cid, model, w, h, np.array([float(x) for x in p[4:]], np.float64))
    return out


def read_images_txt(path):
    out = {}
    lines = path.read_text().splitlines()
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
        qvec = np.array([float(x) for x in p[1:5]], np.float64)
        tvec = np.array([float(x) for x in p[5:8]], np.float64)
        cid = int(p[8])
        name = " ".join(p[9:])
        out[iid] = CImage(iid, qvec, tvec, cid, name)
        if i < len(lines):
            i += 1  # POINTS2D line
    return out


def read_colmap_model(model_dir):
    model_dir = Path(model_dir)
    if (model_dir / "cameras.bin").exists():
        return read_cameras_bin(model_dir / "cameras.bin"), read_images_bin(model_dir / "images.bin")
    return read_cameras_txt(model_dir / "cameras.txt"), read_images_txt(model_dir / "images.txt")


def qvec_to_rotmat(q):
    qw, qx, qy, qz = np.asarray(q, np.float64)
    return np.array([
        [1 - 2*qy*qy - 2*qz*qz, 2*qx*qy - 2*qz*qw,     2*qx*qz + 2*qy*qw],
        [2*qx*qy + 2*qz*qw,     1 - 2*qx*qx - 2*qz*qz, 2*qy*qz - 2*qx*qw],
        [2*qx*qz - 2*qy*qw,     2*qy*qz + 2*qx*qw,     1 - 2*qx*qx - 2*qy*qy],
    ], np.float64)


def choose_registered_image(images, camera_name):
    cands = []
    for im in images.values():
        name = im.name.replace("\\", "/")
        if camera_name not in name:
            continue
        score = 0
        if "frame_000000" in name:
            score -= 100
        if f"/{camera_name}/" in f"/{name}/":
            score -= 20
        cands.append((score, len(name), im))
    if not cands:
        raise RuntimeError(f"No registered COLMAP image found for {camera_name}")
    cands.sort(key=lambda x: (x[0], x[1]))
    return cands[0][2]


def rescale_camera(cam, target_w, target_h):
    if cam.width == target_w and cam.height == target_h:
        return cam
    sx, sy = target_w / cam.width, target_h / cam.height
    if abs(sx - sy) > 1e-5:
        raise RuntimeError(f"Non-uniform camera scaling {cam.width}x{cam.height} -> {target_w}x{target_h}")
    p = cam.params.copy()
    s = sx
    if cam.model in ("SIMPLE_PINHOLE", "SIMPLE_RADIAL", "RADIAL", "FOV", "SIMPLE_RADIAL_FISHEYE", "RADIAL_FISHEYE"):
        p[0:3] *= s
    elif cam.model in ("PINHOLE", "OPENCV", "OPENCV_FISHEYE", "FULL_OPENCV", "THIN_PRISM_FISHEYE"):
        p[0] *= sx; p[1] *= sy; p[2] *= sx; p[3] *= sy
    else:
        raise RuntimeError(f"Unsupported camera model for scaling: {cam.model}")
    return CCamera(cam.camera_id, cam.model, target_w, target_h, p)


@dataclass
class Calibration:
    camera: CCamera
    P: np.ndarray        # 3x4 world -> camera, normalized projection
    C: np.ndarray        # camera center in world


def load_calibration(root, cameras, image_hw, explicit=None):
    model_dir = find_colmap_model(root, explicit)
    cam_db, images = read_colmap_model(model_dir)
    H, W = image_hw
    out = {}
    print(f"[COLMAP] {model_dir}")
    for name in cameras:
        im = choose_registered_image(images, name)
        cam = rescale_camera(cam_db[im.camera_id], W, H)
        R = qvec_to_rotmat(im.qvec)
        t = im.tvec.reshape(3)
        P = np.c_[R, t]
        C = -R.T @ t
        out[name] = Calibration(cam, P, C)
        print(f"  {name} <- {im.name} | {cam.model}")
    return out


# -----------------------------------------------------------------------------
# Camera models: enough for common COLMAP pinhole/OpenCV models
# -----------------------------------------------------------------------------

def intrinsics(cam):
    p = cam.params
    if cam.model in ("SIMPLE_PINHOLE", "SIMPLE_RADIAL", "RADIAL"):
        f, cx, cy = p[:3]
        return float(f), float(f), float(cx), float(cy)
    if cam.model in ("PINHOLE", "OPENCV", "FULL_OPENCV", "OPENCV_FISHEYE"):
        fx, fy, cx, cy = p[:4]
        return float(fx), float(fy), float(cx), float(cy)
    raise RuntimeError(f"Camera model not implemented in this test: {cam.model}")


def K_matrix(cam):
    fx, fy, cx, cy = intrinsics(cam)
    return np.array([[fx,0,cx],[0,fy,cy],[0,0,1]], np.float64)


def distortion(cam):
    p = cam.params
    if cam.model in ("SIMPLE_PINHOLE", "PINHOLE"):
        return "opencv", np.zeros(5, np.float64)
    if cam.model == "SIMPLE_RADIAL":
        return "opencv", np.array([p[3],0,0,0,0], np.float64)
    if cam.model == "RADIAL":
        return "opencv", np.array([p[3],p[4],0,0,0], np.float64)
    if cam.model == "OPENCV":
        return "opencv", np.array([p[4],p[5],p[6],p[7],0], np.float64)
    if cam.model == "FULL_OPENCV":
        return "opencv", np.array([p[4],p[5],p[6],p[7],p[8],p[9],p[10],p[11]], np.float64)
    if cam.model == "OPENCV_FISHEYE":
        return "fisheye", np.array([p[4],p[5],p[6],p[7]], np.float64)
    raise RuntimeError(f"Distortion model not implemented: {cam.model}")


def pixel_to_norm(cam, xy):
    pts = np.asarray(xy, np.float64).reshape(1,1,2)
    K = K_matrix(cam)
    kind, d = distortion(cam)
    if kind == "fisheye":
        u = cv2.fisheye.undistortPoints(pts, K, d)
    else:
        u = cv2.undistortPoints(pts, K, d)
    return u.reshape(2)


def project_world(calib, X):
    Xc = calib.P[:,:3] @ np.asarray(X, np.float64) + calib.P[:,3]
    if Xc[2] <= 1e-9:
        return None
    x = Xc[:2] / Xc[2]
    K = K_matrix(calib.camera)
    kind, d = distortion(calib.camera)
    obj = np.array([[x[0], x[1], 1.0]], np.float64)
    if kind == "fisheye":
        img, _ = cv2.fisheye.projectPoints(obj.reshape(1,1,3), np.zeros(3), np.zeros(3), K, d)
    else:
        img, _ = cv2.projectPoints(obj, np.zeros(3), np.zeros(3), K, d)
    return img.reshape(2)


# -----------------------------------------------------------------------------
# COLMAP-style two-view triangulation: transient consistency check only
# -----------------------------------------------------------------------------

def triangulate_colmap_dlt(P1, P2, x1, x2):
    x1 = np.asarray(x1, np.float64); x2 = np.asarray(x2, np.float64)
    A = np.stack([
        x1[0] * P1[2] - P1[0],
        x1[1] * P1[2] - P1[1],
        x2[0] * P2[2] - P2[0],
        x2[1] * P2[2] - P2[1],
    ])
    try:
        _, _, Vt = np.linalg.svd(A)
    except np.linalg.LinAlgError:
        return None
    Xh = Vt[-1]
    if abs(Xh[3]) < 1e-12:
        return None
    return Xh[:3] / Xh[3]


def triangulation_angle_deg(C1, C2, X):
    r1 = np.asarray(X) - np.asarray(C1)
    r2 = np.asarray(X) - np.asarray(C2)
    n1, n2 = np.linalg.norm(r1), np.linalg.norm(r2)
    if n1 < 1e-12 or n2 < 1e-12:
        return 0.0
    a = math.acos(float(np.clip(np.dot(r1,r2)/(n1*n2), -1, 1)))
    a = min(a, math.pi - a)
    return math.degrees(a)


def pair_geometry_ok(cal1, cal2, uv1, uv2, max_reproj_px, min_angle_deg):
    x1 = pixel_to_norm(cal1.camera, uv1)
    x2 = pixel_to_norm(cal2.camera, uv2)
    X = triangulate_colmap_dlt(cal1.P, cal2.P, x1, x2)
    if X is None:
        return None
    z1 = (cal1.P[:,:3] @ X + cal1.P[:,3])[2]
    z2 = (cal2.P[:,:3] @ X + cal2.P[:,3])[2]
    if z1 <= 0 or z2 <= 0:
        return None
    p1 = project_world(cal1, X); p2 = project_world(cal2, X)
    if p1 is None or p2 is None:
        return None
    e1 = np.linalg.norm(p1 - np.asarray(uv1)); e2 = np.linalg.norm(p2 - np.asarray(uv2))
    rms = float(math.sqrt(0.5*(e1*e1 + e2*e2)))
    angle = triangulation_angle_deg(cal1.C, cal2.C, X)
    if rms > max_reproj_px or angle < min_angle_deg:
        return None
    return rms, angle  # XYZ intentionally discarded


# -----------------------------------------------------------------------------
# RootSIFT + cache
# -----------------------------------------------------------------------------

def rootsift(desc):
    if desc is None or len(desc) == 0:
        return None
    d = desc.astype(np.float32)
    d /= np.sum(d, axis=1, keepdims=True) + 1e-12
    return np.sqrt(d)


def create_sift(nfeatures):
    return cv2.SIFT_create(nfeatures=nfeatures, contrastThreshold=0.01, edgeThreshold=20, sigma=1.6)


def extract_sift(sift, image, mask):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    kp, desc = sift.detectAndCompute(gray, mask)
    return kp, rootsift(desc)


def mutual_ratio_matches(da, db, ratio):
    if da is None or db is None or len(da) < 2 or len(db) < 2:
        return []
    bf = cv2.BFMatcher(cv2.NORM_L2)
    ab = {}
    for pair in bf.knnMatch(da, db, k=2):
        if len(pair) == 2 and pair[0].distance < ratio * pair[1].distance:
            ab[pair[0].queryIdx] = (pair[0].trainIdx, pair[0].distance)
    ba = {}
    for pair in bf.knnMatch(db, da, k=2):
        if len(pair) == 2 and pair[0].distance < ratio * pair[1].distance:
            ba[pair[0].queryIdx] = (pair[0].trainIdx, pair[0].distance)
    out = []
    for ia, (ib, dab) in ab.items():
        if ib in ba and ba[ib][0] == ia:
            out.append((ia, ib, 0.5*(dab + ba[ib][1])))
    return out


@dataclass
class Features:
    xy: np.ndarray
    desc: np.ndarray


def cache_path(cache_dir, cam, fid):
    return cache_dir / cam / f"frame_{fid:06d}.npz"


def prepare_cache(root, rgb_parent, cameras, frame_paths, image_hw, cache_dir, nfeatures, no_mask, force):
    sift = create_sift(nfeatures)
    for cam in cameras:
        for p in frame_paths[cam]:
            fid = frame_id(p)
            cp = cache_path(cache_dir, cam, fid)
            if cp.exists() and not force:
                continue
            im = cv2.imread(str(p), cv2.IMREAD_COLOR)
            if im is None:
                raise RuntimeError(f"Cannot read {p}")
            if im.shape[:2] != image_hw:
                im = cv2.resize(im, (image_hw[1], image_hw[0]))
            mask = read_mask(root, cam, fid, image_hw, no_mask)
            kp, desc = extract_sift(sift, im, mask)
            cp.parent.mkdir(parents=True, exist_ok=True)
            xy = np.array([k.pt for k in kp], np.float32).reshape(-1,2)
            if desc is None:
                desc = np.empty((0,128), np.float32)
            np.savez_compressed(cp, xy=xy, desc=desc.astype(np.float32))
        print(f"[SIFT] {cam}: cached {len(frame_paths[cam])} frames")


def load_features(path):
    d = np.load(path)
    return Features(d["xy"].astype(np.float32), d["desc"].astype(np.float32))


# -----------------------------------------------------------------------------
# Same-camera temporal tracks: simple, auditable adjacent-frame chaining
# -----------------------------------------------------------------------------

@dataclass
class Obs:
    frame: int
    kp_idx: int
    x: float
    y: float


@dataclass
class Track:
    camera: str
    track_id: int
    obs: list
    static_rms_px: float = 0.0
    excursion_px: float = 0.0
    coherent_excursion_px: float = 0.0
    jitter_rms_px: float = 0.0
    frame_span: int = 0
    motion_score: float = 0.0
    label: str = "UNCERTAIN"


def _find_rgb_frame(root, rgb_parent, cam, fid):
    folder = root / rgb_parent / cam
    for ext in ("png", "jpg", "jpeg"):
        p = folder / f"frame_{fid:06d}.{ext}"
        if p.exists():
            return p
    raise FileNotFoundError(f"Missing RGB frame {fid:06d} for {cam}")


def _read_gray_and_mask(root, rgb_parent, cam, fid, image_hw, no_mask):
    p = _find_rgb_frame(root, rgb_parent, cam, fid)
    bgr = cv2.imread(str(p), cv2.IMREAD_COLOR)
    if bgr is None:
        raise RuntimeError(f"Cannot read {p}")
    if bgr.shape[:2] != image_hw:
        bgr = cv2.resize(
            bgr,
            (image_hw[1], image_hw[0]),
            interpolation=cv2.INTER_LINEAR,
        )
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)

    mask = None
    if not no_mask:
        mp = root / "mask" / cam / f"frame_{fid:06d}.png"
        if mp.exists():
            mask = cv2.imread(str(mp), cv2.IMREAD_GRAYSCALE)
            if mask is not None and mask.shape != image_hw:
                mask = cv2.resize(
                    mask,
                    (image_hw[1], image_hw[0]),
                    interpolation=cv2.INTER_NEAREST,
                )
            if mask is not None:
                mask = (mask > 0).astype(np.uint8) * 255

    return gray, mask


def _sift_seed_points(gray, mask, nfeatures, min_distance, existing_xy=None):
    """
    Detect SIFT points only for track initialization/reseeding.
    Temporal propagation itself is KLT, so detector repeatability is no longer
    required every frame.
    """
    sift = create_sift(nfeatures)
    kp = sift.detect(gray, mask)
    kp = sorted(kp, key=lambda k: -float(k.response))

    accepted = []
    existing = (
        np.asarray(existing_xy, dtype=np.float32).reshape(-1, 2)
        if existing_xy is not None and len(existing_xy)
        else np.zeros((0, 2), dtype=np.float32)
    )

    for k in kp:
        p = np.array(k.pt, dtype=np.float32)

        if len(existing):
            if np.min(np.linalg.norm(existing - p[None], axis=1)) < min_distance:
                continue

        if accepted:
            a = np.asarray(accepted, dtype=np.float32)
            if np.min(np.linalg.norm(a - p[None], axis=1)) < min_distance:
                continue

        accepted.append((float(p[0]), float(p[1])))

    return accepted


def build_tracks(
    cam,
    frame_ids,
    root,
    rgb_parent,
    image_hw,
    no_mask,
    sift_nfeatures,
    min_len,
    reseed_interval,
    reseed_min_distance,
    target_active_tracks,
    lk_win_size,
    lk_max_level,
    fb_max_error,
    lk_max_error,
):
    """
    Long-lived same-camera tracking:

      SIFT = initialize/reseed sparse salient points.
      pyramidal Lucas-Kanade = propagate the SAME pixel continuously.
      forward/backward KLT = reject drift.

    This directly fixes the previous failure mode where SIFT was re-detected in
    every frame and track identity died whenever the detector changed scale,
    orientation, or missed a feature for one frame.
    """
    if not frame_ids:
        return []

    tracks = {}
    active = {}  # tid -> [x,y]
    next_tid = 0

    first_fid = frame_ids[0]
    prev_gray, prev_mask = _read_gray_and_mask(
        root, rgb_parent, cam, first_fid, image_hw, no_mask
    )

    initial = _sift_seed_points(
        prev_gray,
        prev_mask,
        sift_nfeatures,
        reseed_min_distance,
        existing_xy=None,
    )[:target_active_tracks]

    for x, y in initial:
        tid = next_tid
        next_tid += 1
        tracks[tid] = Track(
            camera=cam,
            track_id=tid,
            obs=[Obs(first_fid, -1, x, y)],
        )
        active[tid] = np.array([x, y], dtype=np.float32)

    lk_params = dict(
        winSize=(lk_win_size, lk_win_size),
        maxLevel=lk_max_level,
        criteria=(
            cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
            30,
            0.01,
        ),
        minEigThreshold=1e-4,
    )

    for pos in range(1, len(frame_ids)):
        fid = frame_ids[pos]
        curr_gray, curr_mask = _read_gray_and_mask(
            root, rgb_parent, cam, fid, image_hw, no_mask
        )

        tids = list(active.keys())

        if tids:
            p0 = np.asarray([active[t] for t in tids], dtype=np.float32).reshape(-1, 1, 2)

            p1, st1, err1 = cv2.calcOpticalFlowPyrLK(
                prev_gray, curr_gray, p0, None, **lk_params
            )
            p0_back, st2, _ = cv2.calcOpticalFlowPyrLK(
                curr_gray, prev_gray, p1, None, **lk_params
            )

            p0f = p0.reshape(-1, 2)
            p1f = p1.reshape(-1, 2)
            p0bf = p0_back.reshape(-1, 2)
            st1 = st1.reshape(-1).astype(bool)
            st2 = st2.reshape(-1).astype(bool)
            err1 = err1.reshape(-1)

            fb = np.linalg.norm(p0f - p0bf, axis=1)

            h, w = curr_gray.shape
            new_active = {}

            for i, tid in enumerate(tids):
                x, y = p1f[i]

                good = (
                    st1[i]
                    and st2[i]
                    and np.isfinite(x)
                    and np.isfinite(y)
                    and fb[i] <= fb_max_error
                    and err1[i] <= lk_max_error
                    and 0 <= x < w
                    and 0 <= y < h
                )

                if good and curr_mask is not None:
                    xi = min(max(int(round(x)), 0), w - 1)
                    yi = min(max(int(round(y)), 0), h - 1)
                    good = curr_mask[yi, xi] > 0

                if not good:
                    continue

                tracks[tid].obs.append(
                    Obs(fid, -1, float(x), float(y))
                )
                new_active[tid] = np.array([x, y], dtype=np.float32)

            active = new_active

        # Reseed without killing the surviving long tracks.
        need_reseed = (
            pos % max(reseed_interval, 1) == 0
            or len(active) < target_active_tracks
        )

        if need_reseed:
            existing = list(active.values())
            candidates = _sift_seed_points(
                curr_gray,
                curr_mask,
                sift_nfeatures,
                reseed_min_distance,
                existing_xy=existing,
            )

            slots = max(0, target_active_tracks - len(active))

            for x, y in candidates[:slots]:
                tid = next_tid
                next_tid += 1
                tracks[tid] = Track(
                    camera=cam,
                    track_id=tid,
                    obs=[Obs(fid, -1, x, y)],
                )
                active[tid] = np.array([x, y], dtype=np.float32)

        prev_gray = curr_gray
        prev_mask = curr_mask

    out = [
        t for t in tracks.values()
        if len(t.obs) >= min_len
    ]
    out.sort(
        key=lambda t: (
            -len(t.obs),
            t.obs[0].frame,
            t.track_id,
        )
    )

    # Renumber after filtering.
    for new_tid, t in enumerate(out):
        t.track_id = new_tid

    return out


def _moving_average_trajectory(pts, window=7):
    if len(pts) < 3:
        return pts.copy()

    window = min(window, len(pts))
    if window % 2 == 0:
        window -= 1
    window = max(window, 1)

    if window == 1:
        return pts.copy()

    radius = window // 2
    padded = np.pad(pts, ((radius, radius), (0, 0)), mode="edge")
    kernel = np.ones(window, dtype=np.float64) / window

    sx = np.convolve(padded[:, 0], kernel, mode="valid")
    sy = np.convolve(padded[:, 1], kernel, mode="valid")
    return np.stack([sx, sy], axis=1)


def characterize_track(t: Track):
    pts = np.asarray([[o.x, o.y] for o in t.obs], dtype=np.float64)
    frames = np.asarray([o.frame for o in t.obs], dtype=np.int64)

    center = np.median(pts, axis=0)
    residual = np.linalg.norm(pts - center[None], axis=1)

    t.static_rms_px = float(np.sqrt(np.mean(residual ** 2)))
    t.excursion_px = float(
        np.linalg.norm(pts.max(axis=0) - pts.min(axis=0))
    )
    t.frame_span = int(frames[-1] - frames[0] + 1)

    smooth = _moving_average_trajectory(pts, window=7)
    t.coherent_excursion_px = float(
        np.linalg.norm(smooth.max(axis=0) - smooth.min(axis=0))
    )
    t.jitter_rms_px = float(
        np.sqrt(
            np.mean(
                np.sum((pts - smooth) ** 2, axis=1)
            )
        )
    )


def estimate_noise(tracks, floor_px):
    """
    Estimate image-space localization noise from long, lowest-motion tracks.
    Short tracks are deliberately excluded because they caused false dynamics.
    """
    long_tracks = [
        t for t in tracks
        if t.frame_span >= 12
    ]

    source = long_tracks if long_tracks else tracks
    vals = np.asarray(
        [t.jitter_rms_px for t in source],
        dtype=np.float64,
    )

    if len(vals) == 0:
        return float(floor_px)

    q = np.quantile(vals, 0.35)
    low = vals[vals <= q]

    med = float(np.median(low)) if len(low) else float(q)
    mad = (
        float(np.median(np.abs(low - med)))
        if len(low)
        else 0.0
    )

    return max(
        float(floor_px),
        med + 1.4826 * mad,
    )


def classify_tracks(
    tracks,
    sigma,
    static_sigma,
    dynamic_sigma,
    min_excursion,
    min_dynamic_span,
):
    """
    Dynamic = coherent low-frequency image motion over a sufficiently long track.
    Random subpixel jitter is measured separately and does not become motion.
    """
    for t in tracks:
        t.motion_score = (
            t.coherent_excursion_px
            / max(sigma, 1e-8)
        )

        if (
            t.coherent_excursion_px <= static_sigma * sigma
            and t.excursion_px <= max(min_excursion, 2.0 * sigma)
        ):
            t.label = "STATIC"

        elif (
            t.frame_span >= min_dynamic_span
            and t.coherent_excursion_px >= min_excursion
            and t.motion_score >= dynamic_sigma
        ):
            t.label = "DYNAMIC"

        else:
            t.label = "UNCERTAIN"


def dynamic_tracks_by_frame(tracks):
    out = defaultdict(list)

    for t in tracks:
        if t.label != "DYNAMIC":
            continue
        for o in t.obs:
            out[o.frame].append((t, np.array([o.x, o.y], dtype=np.float32)))

    return out


def nearest_dynamic_track(frame_tracks, uv, max_dist):
    if not frame_tracks:
        return None

    uv = np.asarray(uv, dtype=np.float32)
    pts = np.asarray([xy for _, xy in frame_tracks], dtype=np.float32)

    d = np.linalg.norm(pts - uv[None], axis=1)
    i = int(np.argmin(d))

    if d[i] > max_dist:
        return None

    return frame_tracks[i][0]


# -----------------------------------------------------------------------------
# Dynamic-only cross-camera evidence
# -----------------------------------------------------------------------------

@dataclass
class Evidence:
    frame: int
    cam_a: str
    track_a: int
    cam_b: str
    track_b: int
    x_a: float
    y_a: float
    x_b: float
    y_b: float
    rootsift_distance: float
    reproj_rms_px: float
    angle_deg: float


def dynamic_crossview(
    cameras,
    frame_ids,
    cache_dir,
    calibs,
    frame_dynamic_tracks,
    ratio,
    max_reproj,
    min_angle,
    track_attach_radius,
):
    """
    Cross-view SIFT still provides viewpoint-robust same-frame appearance
    matching. A SIFT match is accepted only if each endpoint lies near an
    already-established long-lived DYNAMIC KLT trajectory.
    """
    best = {}

    for fid in frame_ids:
        feats = {
            c: load_features(cache_path(cache_dir, c, fid))
            for c in cameras
        }

        accepted = 0

        for i in range(len(cameras)):
            for j in range(i + 1, len(cameras)):
                ca, cb = cameras[i], cameras[j]
                fa, fb = feats[ca], feats[cb]

                matches = mutual_ratio_matches(
                    fa.desc,
                    fb.desc,
                    ratio,
                )

                for ia, ib, dist in matches:
                    uv1 = fa.xy[ia]
                    uv2 = fb.xy[ib]

                    ta = nearest_dynamic_track(
                        frame_dynamic_tracks[ca].get(fid, []),
                        uv1,
                        track_attach_radius,
                    )
                    tb = nearest_dynamic_track(
                        frame_dynamic_tracks[cb].get(fid, []),
                        uv2,
                        track_attach_radius,
                    )

                    if ta is None or tb is None:
                        continue

                    q = pair_geometry_ok(
                        calibs[ca],
                        calibs[cb],
                        uv1,
                        uv2,
                        max_reproj,
                        min_angle,
                    )

                    if q is None:
                        continue

                    rms, angle = q

                    e = Evidence(
                        fid,
                        ca,
                        ta.track_id,
                        cb,
                        tb.track_id,
                        float(uv1[0]),
                        float(uv1[1]),
                        float(uv2[0]),
                        float(uv2[1]),
                        float(dist),
                        rms,
                        angle,
                    )

                    key = canon(
                        ca,
                        ta.track_id,
                        cb,
                        tb.track_id,
                    ) + (fid,)

                    score = (
                        e.reproj_rms_px,
                        e.rootsift_distance,
                    )

                    if key not in best or score < best[key][0]:
                        best[key] = (score, e)

        # Print after deduplication for this frame.
        accepted = sum(
            1 for k in best
            if k[-1] == fid
        )
        print(
            f"[MULTIVIEW] frame {fid:06d}: "
            f"{accepted} long-track dynamic matches confirmed"
        )

    return [value[1] for value in best.values()]


# -----------------------------------------------------------------------------
# Repeated whole-video multi-camera association
# -----------------------------------------------------------------------------

@dataclass
class Link:
    cam_a: str
    track_a: int
    cam_b: str
    track_b: int
    support_frames: int
    median_reproj: float
    median_sift: float


def canon(a,ta,b,tb):
    return (a,ta,b,tb) if a < b else (b,tb,a,ta)


def aggregate_links(evidence, min_support):
    g = defaultdict(list)
    for e in evidence:
        g[canon(e.cam_a,e.track_a,e.cam_b,e.track_b)].append(e)
    out = []
    for k, es in g.items():
        support = len(set(e.frame for e in es))
        if support < min_support:
            continue
        out.append(Link(k[0],k[1],k[2],k[3],support,
                        float(np.median([e.reproj_rms_px for e in es])),
                        float(np.median([e.rootsift_distance for e in es]))))
    out.sort(key=lambda l: (-l.support_frames, l.median_reproj, l.median_sift))
    return out


class CameraDSU:
    def __init__(self):
        self.p={}; self.cams={}
    def add(self,x): self.p[x]=x; self.cams[x]={x[0]}
    def find(self,x):
        if self.p[x]!=x: self.p[x]=self.find(self.p[x])
        return self.p[x]
    def union(self,a,b):
        ra,rb=self.find(a),self.find(b)
        if ra==rb: return True
        if not self.cams[ra].isdisjoint(self.cams[rb]): return False
        self.p[rb]=ra; self.cams[ra]|=self.cams[rb]; return True
    def groups(self):
        g=defaultdict(list)
        for x in self.p: g[self.find(x)].append(x)
        return list(g.values())


def build_groups(all_tracks, links):
    d=CameraDSU()
    for cam, tracks in all_tracks.items():
        for t in tracks:
            if t.label=="DYNAMIC": d.add((cam,t.track_id))
    accepted=[]
    for l in links:
        a=(l.cam_a,l.track_a); b=(l.cam_b,l.track_b)
        if a in d.p and b in d.p and d.union(a,b): accepted.append(l)
    groups=[x for x in d.groups() if len(x)>=2]
    groups.sort(key=lambda x:(-len(x),x))
    node_to_gid={node:gid for gid,g in enumerate(groups) for node in g}
    return groups,node_to_gid,accepted


# -----------------------------------------------------------------------------
# Outputs
# -----------------------------------------------------------------------------

def save_motion(all_tracks, noise, path):
    with open(path,"w",newline="") as f:
        w=csv.writer(f); w.writerow(["camera","track_id","label","num_frames","start_frame","end_frame","static_rms_px","excursion_px","pixel_noise_px","motion_score_sigma"])
        for cam,ts in all_tracks.items():
            for t in ts:
                w.writerow([cam,t.track_id,t.label,len(t.obs),t.obs[0].frame,t.obs[-1].frame,t.static_rms_px,t.excursion_px,noise[cam],t.motion_score])



def save_track_observations(all_tracks, node_to_gid, path):
    """Save actual per-frame 2D trajectory samples for later diagnostics."""
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "camera", "track_id", "global_track_id", "label",
            "frame", "x", "y",
            "static_rms_px", "excursion_px", "motion_score_sigma"
        ])
        for cam, tracks in all_tracks.items():
            for t in tracks:
                gid = node_to_gid.get((cam, t.track_id), -1)
                for o in t.obs:
                    w.writerow([
                        cam, t.track_id, gid, t.label,
                        o.frame, o.x, o.y,
                        t.static_rms_px, t.excursion_px, t.motion_score
                    ])


def save_evidence(es,path):
    with open(path,"w",newline="") as f:
        w=csv.writer(f); w.writerow(["frame","cam_a","track_a","x_a","y_a","cam_b","track_b","x_b","y_b","rootsift_distance","triangulation_reproj_rms_px","triangulation_angle_deg"])
        for e in es:
            w.writerow([e.frame,e.cam_a,e.track_a,e.x_a,e.y_a,e.cam_b,e.track_b,e.x_b,e.y_b,e.rootsift_distance,e.reproj_rms_px,e.angle_deg])


def save_links(links,accepted,path):
    acc={canon(l.cam_a,l.track_a,l.cam_b,l.track_b) for l in accepted}
    with open(path,"w",newline="") as f:
        w=csv.writer(f); w.writerow(["cam_a","track_a","cam_b","track_b","support_frames","median_reproj_rms_px","median_rootsift_distance","accepted_global_link"])
        for l in links:
            w.writerow([l.cam_a,l.track_a,l.cam_b,l.track_b,l.support_frames,l.median_reproj,l.median_sift,int(canon(l.cam_a,l.track_a,l.cam_b,l.track_b) in acc)])


def color(gid):
    hsv=np.uint8([[[int((gid*137.508)%180),230,255]]]); bgr=cv2.cvtColor(hsv,cv2.COLOR_HSV2BGR)[0,0]
    return tuple(int(v) for v in bgr)


def render_panel(root,rgb_parent,cameras,fid,all_tracks,node_to_gid,path,cols,max_tracks):
    ims=[]
    for cam in cameras:
        folder=root/rgb_parent/cam; p=None
        for ext in ("png","jpg","jpeg"):
            q=folder/f"frame_{fid:06d}.{ext}"
            if q.exists(): p=q; break
        if p is None: raise FileNotFoundError(f"Missing {cam} frame {fid}")
        ims.append(cv2.imread(str(p)))
    H=max(im.shape[0] for im in ims); W=max(im.shape[1] for im in ims); rows=math.ceil(len(ims)/cols)
    canvas=np.zeros((rows*H,cols*W,3),np.uint8); offsets={}
    for idx,(cam,im) in enumerate(zip(cameras,ims)):
        r,c=divmod(idx,cols); x0,y0=c*W,r*H
        if im.shape[:2]!=(H,W): im=cv2.resize(im,(W,H))
        canvas[y0:y0+H,x0:x0+W]=im; offsets[cam]=(x0,y0)
        cv2.putText(canvas,cam,(x0+8,y0+20),cv2.FONT_HERSHEY_SIMPLEX,.5,(255,255,255),1,cv2.LINE_AA)
    bygid=defaultdict(list)
    for cam,ts in all_tracks.items():
        for t in ts:
            gid=node_to_gid.get((cam,t.track_id))
            if gid is None: continue
            for o in t.obs:
                if o.frame==fid: bygid[gid].append((cam,o))
    gids=sorted(bygid,key=lambda g:(-len(bygid[g]),g))[:max_tracks]
    for gid in gids:
        col=color(gid); pts=[]
        for cam,o in bygid[gid]:
            ox,oy=offsets[cam]; p=(int(round(ox+o.x)),int(round(oy+o.y))); pts.append(p)
            cv2.circle(canvas,p,6,col,-1,cv2.LINE_AA); cv2.circle(canvas,p,8,(255,255,255),1,cv2.LINE_AA)
            cv2.putText(canvas,str(gid),(p[0]+7,p[1]-6),cv2.FONT_HERSHEY_SIMPLEX,.4,col,1,cv2.LINE_AA)
        if len(pts)>=2: cv2.polylines(canvas,[np.array(pts,np.int32)],False,col,1,cv2.LINE_AA)
    cv2.imwrite(str(path),canvas)



def local_track_color(camera_index, track_id):
    """Color for local-only dynamic tracks."""
    hue = int(((camera_index + 1) * 31 + track_id * 137.508) % 180)
    hsv = np.uint8([[[hue, 170, 235]]])
    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0, 0]
    return tuple(int(v) for v in bgr)


def choose_tracks_for_video(all_tracks, node_to_gid, cameras, tracks_per_camera):
    """
    Prefer multiview-confirmed dynamic tracks, then fill with the strongest
    local dynamic tracks in each camera.
    """
    chosen = {}

    for cam in cameras:
        dynamic = [t for t in all_tracks[cam] if t.label == "DYNAMIC"]

        confirmed = [
            t for t in dynamic
            if (cam, t.track_id) in node_to_gid
        ]
        confirmed.sort(
            key=lambda t: (-len(t.obs), -t.excursion_px, -t.motion_score)
        )

        local = [
            t for t in dynamic
            if (cam, t.track_id) not in node_to_gid
        ]
        local.sort(
            key=lambda t: (-len(t.obs), -t.excursion_px, -t.motion_score)
        )

        chosen[cam] = (confirmed + local)[:tracks_per_camera]

    return chosen


def render_track_evolution_video(
    root,
    rgb_parent,
    cameras,
    frame_ids,
    all_tracks,
    node_to_gid,
    output_path,
    columns,
    tracks_per_camera,
    fps,
    tail,
):
    """
    Synchronized multi-camera panel video.

    Each camera tile shows a few dynamic trajectories evolving through time.
    G# tracks are multiview-confirmed and keep the SAME color/global ID across
    cameras. L# tracks are strong local dynamic tracks used only to fill a tile.
    """
    selected = choose_tracks_for_video(
        all_tracks, node_to_gid, cameras, tracks_per_camera
    )

    obs_lookup = {}
    for cam in cameras:
        for t in selected[cam]:
            obs_lookup[(cam, t.track_id)] = {o.frame: o for o in t.obs}

    first_images = []
    for cam in cameras:
        folder = root / rgb_parent / cam
        p = None
        for ext in ("png", "jpg", "jpeg"):
            q = folder / f"frame_{frame_ids[0]:06d}.{ext}"
            if q.exists():
                p = q
                break
        if p is None:
            raise FileNotFoundError(f"Missing first frame for {cam}")
        im = cv2.imread(str(p), cv2.IMREAD_COLOR)
        if im is None:
            raise RuntimeError(f"Cannot read {p}")
        first_images.append(im)

    H = max(im.shape[0] for im in first_images)
    W = max(im.shape[1] for im in first_images)
    rows = int(math.ceil(len(cameras) / columns))
    footer_h = 38

    writer = cv2.VideoWriter(
        str(output_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (columns * W, rows * H + footer_h),
    )
    if not writer.isOpened():
        raise RuntimeError(f"Could not create {output_path}")

    cam_index = {cam: i for i, cam in enumerate(cameras)}

    for fid in frame_ids:
        canvas = np.zeros(
            (rows * H + footer_h, columns * W, 3),
            dtype=np.uint8,
        )

        for idx, cam in enumerate(cameras):
            r, c = divmod(idx, columns)
            x0, y0 = c * W, r * H

            folder = root / rgb_parent / cam
            p = None
            for ext in ("png", "jpg", "jpeg"):
                q = folder / f"frame_{fid:06d}.{ext}"
                if q.exists():
                    p = q
                    break
            if p is None:
                continue

            im = cv2.imread(str(p), cv2.IMREAD_COLOR)
            if im is None:
                continue
            if im.shape[:2] != (H, W):
                im = cv2.resize(im, (W, H), interpolation=cv2.INTER_LINEAR)

            tile = im.copy()

            cv2.rectangle(tile, (0, 0), (W - 1, 29), (0, 0, 0), -1)
            cv2.putText(
                tile,
                f"{cam} | {len(selected[cam])} dynamic tracks",
                (8, 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.48,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )

            for t in selected[cam]:
                gid = node_to_gid.get((cam, t.track_id))

                if gid is not None:
                    col = color(gid)
                    label = f"G{gid}"
                    thickness = 2
                else:
                    col = local_track_color(cam_index[cam], t.track_id)
                    label = f"L{t.track_id}"
                    thickness = 1

                history = [
                    o for o in t.obs
                    if o.frame <= fid
                    and (tail <= 0 or o.frame >= fid - tail + 1)
                ]
                current = obs_lookup[(cam, t.track_id)].get(fid)

                if len(history) >= 2:
                    pts = np.array(
                        [[int(round(o.x)), int(round(o.y))] for o in history],
                        dtype=np.int32,
                    )
                    cv2.polylines(
                        tile, [pts], False, col, thickness, cv2.LINE_AA
                    )

                if current is not None:
                    pxy = (int(round(current.x)), int(round(current.y)))
                    cv2.circle(tile, pxy, 5, col, -1, cv2.LINE_AA)
                    cv2.circle(tile, pxy, 7, (255, 255, 255), 1, cv2.LINE_AA)
                    cv2.putText(
                        tile,
                        label,
                        (pxy[0] + 6, pxy[1] - 6),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.38,
                        col,
                        1,
                        cv2.LINE_AA,
                    )

            canvas[y0:y0 + H, x0:x0 + W] = tile

        cv2.putText(
            canvas,
            (
                f"frame {fid:06d} | dynamic pixel trajectories | "
                f"G# = multiview-confirmed, L# = local dynamic"
            ),
            (10, rows * H + 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )

        writer.write(canvas)

    writer.release()


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("root")
    ap.add_argument("--colmap-model",default=None)
    ap.add_argument("--sift-nfeatures",type=int,default=5000)
    ap.add_argument("--temporal-ratio",type=float,default=.75)
    ap.add_argument("--multiview-ratio",type=float,default=.75)
    ap.add_argument("--min-temporal-track-length",type=int,default=5)
    # Same-camera temporal identity is now KLT, not detector-to-detector SIFT.
    ap.add_argument("--reseed-interval",type=int,default=5)
    ap.add_argument("--reseed-min-distance-px",type=float,default=5.0)
    ap.add_argument("--target-active-tracks",type=int,default=120)
    ap.add_argument("--lk-win-size",type=int,default=21)
    ap.add_argument("--lk-max-level",type=int,default=3)
    ap.add_argument("--fb-max-error-px",type=float,default=0.75)
    ap.add_argument("--lk-max-error",type=float,default=25.0)
    ap.add_argument("--track-attach-radius-px",type=float,default=3.0,
                    help="Max distance between a cross-view SIFT keypoint and a dynamic KLT track.")
    ap.add_argument("--min-dynamic-span",type=int,default=12,
                    help="A DYNAMIC core track must span at least this many frames.")
    ap.add_argument("--pixel-noise-floor",type=float,default=.25)
    ap.add_argument("--static-sigma",type=float,default=2.0)
    ap.add_argument("--dynamic-sigma",type=float,default=4.0)
    ap.add_argument("--min-dynamic-excursion-px",type=float,default=1.5)
    ap.add_argument("--max-reproj-px",type=float,default=2.0)
    ap.add_argument("--min-triangulation-angle-deg",type=float,default=1.0)
    ap.add_argument("--min-crossview-frame-support",type=int,default=2)
    ap.add_argument("--no-mask",action="store_true")
    ap.add_argument("--force-reextract",action="store_true")
    ap.add_argument("--output",default=None)
    ap.add_argument("--viz-frame",type=int,default=20)
    ap.add_argument("--viz-columns",type=int,default=4)
    ap.add_argument("--viz-max-tracks",type=int,default=30)
    ap.add_argument("--tracks-per-camera",type=int,default=5)
    ap.add_argument("--track-video-tail",type=int,default=20,
                    help="Trajectory tail in frames; 0 draws full history.")
    ap.add_argument("--track-video-fps",type=float,default=10.0)
    ap.add_argument("--no-track-video",action="store_true")
    args=ap.parse_args()

    root=Path(args.root); rgb_parent,cameras=find_rgb_parent(root)
    frame_paths={c:list_rgb_frames(root,rgb_parent,c) for c in cameras}
    common=sorted(set.intersection(*[{frame_id(p) for p in frame_paths[c]} for c in cameras]))
    if not common: raise RuntimeError("No common frames across cameras")
    common_set=set(common)
    frame_paths={c:[p for p in frame_paths[c] if frame_id(p) in common_set] for c in cameras}
    first=cv2.imread(str(frame_paths[cameras[0]][0])); image_hw=first.shape[:2]

    outdir=Path(args.output) if args.output else root/"dynamic_sift_klt_multicam"
    outdir.mkdir(parents=True,exist_ok=True); cache=outdir/"feature_cache"
    calibs=load_calibration(root,cameras,image_hw,args.colmap_model)
    prepare_cache(root,rgb_parent,cameras,frame_paths,image_hw,cache,args.sift_nfeatures,args.no_mask,args.force_reextract)

    all_tracks={}; noise={}; lookups={}
    for cam in cameras:
        ts = build_tracks(
            cam=cam,
            frame_ids=common,
            root=root,
            rgb_parent=rgb_parent,
            image_hw=image_hw,
            no_mask=args.no_mask,
            sift_nfeatures=args.sift_nfeatures,
            min_len=args.min_temporal_track_length,
            reseed_interval=args.reseed_interval,
            reseed_min_distance=args.reseed_min_distance_px,
            target_active_tracks=args.target_active_tracks,
            lk_win_size=args.lk_win_size,
            lk_max_level=args.lk_max_level,
            fb_max_error=args.fb_max_error_px,
            lk_max_error=args.lk_max_error,
        )
        for t in ts:
            characterize_track(t)
        sig = estimate_noise(ts, args.pixel_noise_floor)
        classify_tracks(
            ts,
            sig,
            args.static_sigma,
            args.dynamic_sigma,
            args.min_dynamic_excursion_px,
            args.min_dynamic_span,
        )
        all_tracks[cam] = ts
        noise[cam] = sig
        lookups[cam] = dynamic_tracks_by_frame(ts)
        cnt=defaultdict(int)
        for t in ts: cnt[t.label]+=1
        print(f"[PIXEL] {cam}: noise={sig:.3f}px STATIC={cnt['STATIC']} UNCERTAIN={cnt['UNCERTAIN']} DYNAMIC={cnt['DYNAMIC']}")

    save_motion(all_tracks,noise,outdir/"temporal_track_motion.csv")
    evidence = dynamic_crossview(
        cameras,
        common,
        cache,
        calibs,
        lookups,
        args.multiview_ratio,
        args.max_reproj_px,
        args.min_triangulation_angle_deg,
        args.track_attach_radius_px,
    )
    save_evidence(evidence,outdir/"dynamic_multiview_evidence.csv")
    links=aggregate_links(evidence,args.min_crossview_frame_support)
    groups,node_to_gid,accepted=build_groups(all_tracks,links)
    save_links(links,accepted,outdir/"dynamic_multicam_links.csv")
    save_track_observations(
        all_tracks,
        node_to_gid,
        outdir/"temporal_track_observations.csv"
    )

    if args.viz_frame in common_set:
        render_panel(
            root,rgb_parent,cameras,args.viz_frame,
            all_tracks,node_to_gid,
            outdir/f"frame_{args.viz_frame:06d}_dynamic_multicam.png",
            args.viz_columns,args.viz_max_tracks
        )

    track_video_path = outdir/"dynamic_tracks_all_cameras.mp4"
    if not args.no_track_video:
        render_track_evolution_video(
            root=root,
            rgb_parent=rgb_parent,
            cameras=cameras,
            frame_ids=common,
            all_tracks=all_tracks,
            node_to_gid=node_to_gid,
            output_path=track_video_path,
            columns=args.viz_columns,
            tracks_per_camera=args.tracks_per_camera,
            fps=args.track_video_fps,
            tail=args.track_video_tail,
        )

    print("\n[DONE]")
    print(f"  dynamic triangulation-confirmed correspondences: {len(evidence)}")
    print(f"  repeated cross-camera track links: {len(links)}")
    print(f"  final multi-camera dynamic groups: {len(groups)}")
    print(f"  {outdir/'temporal_track_motion.csv'}")
    print(f"  {outdir/'dynamic_multiview_evidence.csv'}")
    print(f"  {outdir/'dynamic_multicam_links.csv'}")
    print(f"  {outdir/'temporal_track_observations.csv'}")
    if not args.no_track_video:
        print(f"  {track_video_path}")

if __name__ == "__main__":
    main()
