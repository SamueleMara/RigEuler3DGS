"""Calibrated cameras and (camera, original frame) RGB-D-mask views (spec section 2.3).

Camera geometry is fixed and shared across the sequence. The rasterizer's symmetric frustum needs a centred
principal point; the datasets declare an undistorted, centred PINHOLE K (README.txt), and a camera whose K is
not centred is refused rather than resampled.

Pixel convention: the kernel's pixel (u, v) has its centre at K-coordinates (u + 0.5, v + 0.5) when
cx = W / 2, i.e. ndc 0 lands on u = (W - 1) / 2. Back-projection uses K^-1 [u + 0.5, v + 0.5, 1].

A view is (rgb [3,H,W] in [0,1], depth [H,W] metric camera-Z, mask [H,W] bool, depth_valid [H,W] bool)
with depth_valid = mask & finite(depth) & depth > 0. Views are read from disk on demand and kept on the CPU.
"""
from __future__ import annotations

import json
import math
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from PIL import Image


@dataclass
class RenderCamera:
    name: str
    W: int
    H: int
    V: torch.Tensor          # [4,4] row-vector world->view (p_view = [p,1] @ V), float32
    P: torch.Tensor          # [4,4] row-vector full projection
    campos: torch.Tensor     # [3]
    tanfovx: float
    tanfovy: float

    @property
    def K(self) -> np.ndarray:
        fx, fy = self.W / (2 * self.tanfovx), self.H / (2 * self.tanfovy)
        return np.array([[fx, 0, self.W / 2.0], [0, fy, self.H / 2.0], [0, 0, 1.0]])

    @property
    def c2w(self) -> np.ndarray:
        return np.linalg.inv(self.V.double().cpu().numpy().T)

    def to(self, device):
        return RenderCamera(self.name, self.W, self.H, self.V.to(device), self.P.to(device), self.campos.to(device),
                            self.tanfovx, self.tanfovy)


def projection_matrix(tanfovx, tanfovy, znear=0.01, zfar=100.0) -> torch.Tensor:
    """utils.graphics_utils.getProjectionMatrix with a symmetric frustum, transposed (row-vector)."""
    P = torch.zeros(4, 4, dtype=torch.float64)
    P[0, 0], P[1, 1] = 1 / tanfovx, 1 / tanfovy
    P[3, 2] = 1.0
    P[2, 2] = zfar / (zfar - znear)
    P[2, 3] = -(zfar * znear) / (zfar - znear)
    return P.T


def camera_from_K(name, K, c2w, W, H, znear=0.01, zfar=100.0) -> RenderCamera:
    K, c2w = np.asarray(K, float), np.asarray(c2w, float)
    if abs(K[0, 2] - W / 2) > 0.5 or abs(K[1, 2] - H / 2) > 0.5 or abs(K[0, 1]) > 1e-9:
        raise ValueError(f"{name}: principal point ({K[0, 2]}, {K[1, 2]}) is not centred in {W}x{H}; "
                         "the rasterizer's symmetric frustum cannot represent it")
    tx, ty = W / (2 * K[0, 0]), H / (2 * K[1, 1])
    w2c = torch.as_tensor(np.linalg.inv(c2w), dtype=torch.float64)
    V = w2c.T
    P = V @ projection_matrix(tx, ty, znear, zfar)
    return RenderCamera(name, int(W), int(H), V.float(), P.float(), torch.as_tensor(c2w[:3, 3]).float(), tx, ty)


def camera_from_3dgs(cam) -> RenderCamera:
    """From scene.cameras.Camera (tools/run_iekf_scene.load_scene): the matrices the upstream stages render with."""
    from utils.camera_selection import camera_name
    return RenderCamera(camera_name(cam.image_name), int(cam.image_width), int(cam.image_height),
                        cam.world_view_transform.detach().float().cpu(), cam.full_proj_transform.detach().float().cpu(),
                        cam.camera_center.detach().float().cpu(), math.tan(cam.FoVx * 0.5), math.tan(cam.FoVy * 0.5))


def check_against_metadata(cam: RenderCamera, source: Path, tol_px=0.6, tol_m=1e-4) -> dict:
    """The rendering camera must equal metadata/cameras.json -- no silent recalibration. Two formats exist:
    real captures (fx, fy, cx, cy, pose_matrix_4x4 = OpenCV camera-to-world) and synthetic renders
    (fov_y_deg, position, target, up). Each field present is checked; absent ones are reported."""
    meta = {c["name"]: c for c in json.loads((Path(source) / "metadata" / "cameras.json").read_text())["cameras"]}
    if cam.name not in meta:
        raise ValueError(f"{cam.name} not in {source}/metadata/cameras.json")
    m = meta[cam.name]
    rep = {"camera": cam.name, "checked": []}
    if (m.get("width"), m.get("height")) != (cam.W, cam.H):
        raise ValueError(f"{cam.name}: size {m.get('width')}x{m.get('height')} vs renderer {cam.W}x{cam.H}")
    K = cam.K
    if m.get("fx") is not None:
        d = max(abs(K[0, 0] - m["fx"]), abs(K[1, 1] - m["fy"]), abs(K[0, 2] - m["cx"]), abs(K[1, 2] - m["cy"]))
        rep["intrinsics_max_abs_px"] = d
        rep["checked"].append("K")
        if d > tol_px:
            raise ValueError(f"{cam.name}: intrinsics differ from cameras.json by {d:.3g} px")
    elif m.get("fov_y_deg") is not None:
        d = abs(math.degrees(2 * math.atan(cam.tanfovy)) - m["fov_y_deg"])
        rep["fov_y_abs_deg"] = d
        rep["checked"].append("fov_y")
        if d > 0.05:
            raise ValueError(f"{cam.name}: vertical FOV differs from cameras.json by {d:.3g} deg")
    c2w = cam.c2w
    if m.get("pose_matrix_4x4") is not None:
        M = np.asarray(m["pose_matrix_4x4"], float)
        dp, dr = float(np.abs(c2w[:3, 3] - M[:3, 3]).max()), float(np.abs(c2w[:3, :3] - M[:3, :3]).max())
        rep.update(position_max_abs_m=dp, rotation_max_abs=dr)
        rep["checked"].append("pose")
        if dp > tol_m or dr > 1e-4:
            raise ValueError(f"{cam.name}: pose differs from cameras.json (position {dp:.3g} m, rotation {dr:.3g})")
    elif m.get("position") is not None:
        dp = float(np.abs(c2w[:3, 3] - np.asarray(m["position"], float)).max())
        rep["position_max_abs_m"] = dp
        rep["checked"].append("position")
        if dp > max(tol_m, 1e-3):
            raise ValueError(f"{cam.name}: camera centre differs from cameras.json by {dp:.3g} m")
    return rep


@dataclass
class View:
    camera: str
    frame: int
    rgb: torch.Tensor
    depth: torch.Tensor
    mask: torch.Tensor
    depth_valid: torch.Tensor

    def to(self, device):
        return View(self.camera, self.frame, self.rgb.to(device, non_blocking=True), self.depth.to(device),
                    self.mask.to(device), self.depth_valid.to(device))


def _load_mask(path: Path, W: int, H: int) -> torch.Tensor:
    from estimator.masked_observation import load_object_mask
    return load_object_mask(path, H, W)


class ViewStore:
    """Lazy RGB-D-mask views from <source>/{rgb,depth_npy,mask}/<cam>/frame_NNNNNN.*; small CPU LRU cache.

    Phase 5 only. Phase 4 uses recovery/view_cache.PreparedViews (one decode per view): with cache=64 a
    sequential sweep over more than 64 views misses on every access (measured 2026-11-11: 3 TB of cached
    reads in 13 h). `counters` are diagnostics only."""

    def __init__(self, source: Path, cameras: dict, cache: int = 64):
        self.source = Path(source)
        self.cameras = cameras             # name -> RenderCamera
        self._cache: OrderedDict = OrderedDict()
        self._cap = cache
        self.counters = {"get": 0, "hits": 0, "misses": 0, "decodes": 0}

    def _path(self, sub, cam, frame, exts):
        hits = [self.source / sub / cam / f"frame_{int(frame):06d}{e}" for e in exts]
        hits = [h for h in hits if h.is_file()]
        if len(hits) != 1:
            raise FileNotFoundError(f"{sub}/{cam}/frame_{int(frame):06d}: {len(hits)} candidates")
        return hits[0]

    def get(self, cam: str, frame: int) -> View:
        key = (cam, int(frame))
        self.counters["get"] += 1
        if key in self._cache:
            self.counters["hits"] += 1
            self._cache.move_to_end(key)
            return self._cache[key]
        self.counters["misses"] += 1
        self.counters["decodes"] += 1
        c = self.cameras[cam]
        im = Image.open(self._path("rgb", cam, frame, (".png", ".jpg", ".jpeg"))).convert("RGB")
        if im.size != (c.W, c.H):
            raise ValueError(f"{cam} frame {frame}: RGB {im.size} vs camera {c.W}x{c.H}")
        rgb = torch.from_numpy(np.asarray(im).copy()).permute(2, 0, 1).float() / 255.0
        d = np.load(self._path("depth_npy", cam, frame, (".npy",))).astype(np.float32)
        if d.shape != (c.H, c.W):
            raise ValueError(f"{cam} frame {frame}: depth {d.shape} vs camera {c.H}x{c.W}")
        depth = torch.from_numpy(d)
        mask = _load_mask(self._path("mask", cam, frame, (".png", ".jpg", ".jpeg")), c.W, c.H)
        valid = mask & torch.isfinite(depth) & (depth > 0)
        v = View(cam, int(frame), rgb, torch.nan_to_num(depth, nan=0.0), mask, valid)
        self._cache[key] = v
        if len(self._cache) > self._cap:
            self._cache.popitem(last=False)
        return v
