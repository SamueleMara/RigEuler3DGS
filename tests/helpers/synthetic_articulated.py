"""Procedural articulated objects for the Phase 4/5 gates, rendered with the native RGB-D-alpha path.

Object = static body (an open-front box: back, left, right, top, bottom walls) + one moving part:
  revolute  -- a door covering the front, hinged on the left vertical edge (axis +y through the edge);
               its back face (dark) is a separate surface, seen only once the door opens
  prismatic -- a drawer (front panel + inner floor + inner walls) sliding along -z (towards the cameras);
               the inner floor is visible only when the drawer is out

Every surface is sampled on a grid with flat Gaussians (tangential scale ~ spacing, thin normal scale) and a
per-surface colour. Observations (RGB over white, metric depth where alpha > 0.5, mask = alpha > 0.5) are
rendered from the GROUND-TRUTH labelled scene at every (camera, frame) with the same renderer the stages
use; `drop` removes named surfaces from the MODEL handed to the stage (missing geometry), not from the
observations. GT labels / surfaces are returned for the tests only.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import torch

import recovery  # noqa: F401
from recovery.articulated_scene import JointTransport, render_articulated
from recovery.contracts import Joint
from recovery.gaussians import MOVING, PROV_ORIGINAL, PROV_ORIGINAL_CORE, STATIC, GaussianTensors
from recovery.material_completion import ViewRef
from recovery.views import View, camera_from_K

C_BODY = {"back": (0.75, 0.55, 0.35), "left": (0.70, 0.50, 0.30), "right": (0.65, 0.45, 0.28),
          "top": (0.80, 0.60, 0.40), "bottom": (0.60, 0.42, 0.25)}
C_PART = {"door_front": (0.15, 0.35, 0.80), "door_back": (0.10, 0.10, 0.12),
          "drawer_front": (0.15, 0.55, 0.30), "drawer_floor": (0.85, 0.20, 0.20),
          "drawer_left": (0.80, 0.80, 0.20), "drawer_right": (0.80, 0.80, 0.20)}


def _rect(origin, u, v, nu, nv, normal_offset=0.0, normal=None):
    origin, u, v = (np.asarray(x, float) for x in (origin, u, v))
    a, b = np.meshgrid((np.arange(nu) + 0.5) / nu, (np.arange(nv) + 0.5) / nv, indexing="ij")
    P = origin + a.reshape(-1, 1) * u + b.reshape(-1, 1) * v
    if normal is not None:
        P = P + normal_offset * np.asarray(normal, float)
    return P, u / np.linalg.norm(u), v / np.linalg.norm(v), np.linalg.norm(u) / nu, np.linalg.norm(v) / nv


def _quat_from_axes(t1, t2):
    n = np.cross(t1, t2)
    R = np.stack([t1, t2, n], 1)                   # columns: local x, y, z (normal)
    from recovery.articulated_scene import rotmat_to_quat
    return rotmat_to_quat(R)


@dataclass
class SynthScene:
    model: GaussianTensors          # what the stage receives
    full: GaussianTensors           # the ground-truth scene the observations come from
    gt_moving_model: np.ndarray     # GT material of the model rows
    surface_model: np.ndarray       # surface name per model row
    surface_full: np.ndarray
    joint: Joint
    cameras: list
    views: list                     # ViewRef list, every (camera, frame)
    core: np.ndarray                # Phase-2-like core handed to the stage


def make_scene(jtype="revolute", direction="forward", T=8, q_max=None, drop=(), withhold_frac=0.4, seed=0,
               W=96, H=72, density=1.0, observe_q=None, device="cuda", extra=(), q_profile=None):
    """extra: (name, origin, u, v, nu, nv, moving, rgb) surfaces added to the GROUND TRUTH (drop them from the model with
    `drop` to make them missing geometry). q_profile: per processing row fraction of q_max (default linear)."""
    rng = np.random.default_rng(seed)
    w, h, d = 0.40, 0.30, 0.30              # body: x in [-w/2, w/2], y in [-h/2, h/2], z in [2.0, 2.0 + d]
    x0, y0, z0 = -w / 2, -h / 2, 2.0
    nb = lambda L: max(3, int(round(L / 0.025 * density)))
    surf = []
    surf.append(("back", *_rect((x0, y0, z0 + d), (w, 0, 0), (0, h, 0), nb(w), nb(h))))
    surf.append(("left", *_rect((x0, y0, z0), (0, 0, d), (0, h, 0), nb(d), nb(h))))
    surf.append(("right", *_rect((x0 + w, y0, z0), (0, h, 0), (0, 0, d), nb(h), nb(d))))
    surf.append(("top", *_rect((x0, y0, z0), (0, 0, d), (w, 0, 0), nb(d), nb(w))))
    surf.append(("bottom", *_rect((x0, y0 + h, z0), (w, 0, 0), (0, 0, d), nb(w), nb(d))))
    moving_names = set()
    if jtype == "revolute":
        gap = 0.004
        surf.append(("door_front", *_rect((x0, y0, z0 - gap), (w, 0, 0), (0, h, 0), nb(w), nb(h))))
        surf.append(("door_back", *_rect((x0, y0, z0 - gap + 0.012), (0, h, 0), (w, 0, 0), nb(h), nb(w))))
        moving_names = {"door_front", "door_back"}
        axis, point = (0.0, 1.0, 0.0), (x0, 0.0, z0)           # opens towards the cameras, hinge on the left
        q_max = 1.1 if q_max is None else q_max
    else:
        dw, dh, dd = w - 0.04, 0.08, d - 0.04                 # a shallow tray near the top
        dx, dy = x0 + 0.02, y0 + 0.02
        surf.append(("drawer_front", *_rect((dx, dy, z0 - 0.004), (dw, 0, 0), (0, dh, 0), nb(dw), nb(dh))))
        surf.append(("drawer_floor", *_rect((dx, dy + dh, z0), (dw, 0, 0), (0, 0, dd), nb(dw), nb(dd))))
        surf.append(("drawer_left", *_rect((dx, dy, z0), (0, 0, dd), (0, dh, 0), nb(dd), nb(dh))))
        surf.append(("drawer_right", *_rect((dx + dw, dy, z0), (0, dh, 0), (0, 0, dd), nb(dh), nb(dd))))
        moving_names = {"drawer_front", "drawer_floor", "drawer_left", "drawer_right"}
        axis, point = (0.0, 0.0, -1.0), None
        q_max = 0.18 if q_max is None else q_max
    for nm, o, u, v, nu, nv, mv, rgb in extra:
        surf.append((nm, *_rect(o, u, v, nu, nv)))
        C_PART[nm] = tuple(rgb)
        if mv:
            moving_names.add(nm)
    xyz, rot, scl, col, name = [], [], [], [], []
    for nm, P, t1, t2, s1, s2 in surf:
        q = _quat_from_axes(t1, t2)
        xyz.append(P)
        rot.append(np.repeat(q[None], len(P), 0))
        scl.append(np.repeat([[0.7 * s1, 0.7 * s2, 0.0015]], len(P), 0))
        c = np.asarray({**C_BODY, **C_PART}[nm]) + rng.normal(0, 0.03, (len(P), 3))
        col.append(np.clip(c, 0.02, 0.98))
        name += [nm] * len(P)
    xyz, rot, scl, col = (np.concatenate(a) for a in (xyz, rot, scl, col))
    name = np.asarray(name)
    gt_moving = np.isin(name, list(moving_names))
    N = len(xyz)
    t = lambda a: torch.as_tensor(np.asarray(a, float), dtype=torch.float32, device=device)
    dc = (col - 0.5) / 0.28209479177387814
    full = GaussianTensors(t(xyz), t(rot), t(np.log(scl)), t(np.full((N, 1), math.log(0.92 / 0.08))),
                           t(dc[:, None, :]), torch.zeros(N, 15, 3, device=device), 3,
                           {"material": gt_moving.astype(np.int8), "gaussian_uid": np.arange(N, dtype=np.int64),
                            "parent_uid": np.full(N, -1, np.int64), "original_id": np.arange(N, dtype=np.int64),
                            "provenance": np.full(N, PROV_ORIGINAL, np.int8)})
    full.active_sh_degree = 0
    frames = list(range(T)) if direction == "forward" else list(range(T - 1, -1, -1))
    qs = [q_max * k / (T - 1) for k in range(T)] if q_profile is None else [q_max * float(f) for f in q_profile]
    e = np.asarray(axis, float)
    joint = Joint(type=jtype, axis=tuple(e / np.linalg.norm(e)), point=point, frames=tuple(frames), q=tuple(qs),
                  posterior_sd_q=tuple([1e-5] * T), direction=direction, geometry_file="synthetic",
                  trajectory_file="synthetic", geometry_sha256="", trajectory_sha256="")
    obs_joint = joint if observe_q is None else Joint(**{**joint.__dict__, "q": tuple(observe_q)})
    cams = []
    for k, eye in enumerate(((0.0, -0.7, 1.2), (0.55, -0.6, 1.3), (-0.5, -0.65, 1.25))):
        eye = np.asarray(eye, float)
        tgt = np.array([0.0, 0.0, z0 + d / 2])
        zc = tgt - eye; zc /= np.linalg.norm(zc)
        xc = np.cross([0.0, -1.0, 0.0], zc); xc /= np.linalg.norm(xc)
        yc = np.cross(zc, xc)
        c2w = np.eye(4); c2w[:3, :3] = np.stack([xc, yc, zc], 1); c2w[:3, 3] = eye
        f = W / (2 * math.tan(math.radians(28)))
        cams.append(camera_from_K(f"cam_{k:03d}", [[f, 0, W / 2], [0, f, H / 2], [0, 0, 1]], c2w, W, H))
    tr = JointTransport(obs_joint, device)
    bg = torch.ones(3, device=device)
    mv_full = torch.as_tensor(gt_moving, device=device)
    store = {}
    for cam in cams:
        for f in frames:
            with torch.no_grad():
                o = render_articulated(cam, tr.at(f), full, mv_full, bg, "world_SH", 0)
            A = o["alpha"][0]
            mask = A > 0.5
            depth = torch.where(mask, o["depth"][0], torch.zeros_like(A))
            v = View(cam.name, f, o["color"].clamp(0, 1).cpu(), depth.cpu(), mask.cpu(), (mask & (depth > 0)).cpu())
            store[(cam.name, f)] = v
    views = [ViewRef(c, f, (lambda key=(c.name, f): store[key])) for c in cams for f in frames]
    keep = ~np.isin(name, list(drop))
    model = full.subset(np.nonzero(keep)[0])
    model.lineage["original_id"] = np.arange(len(model), dtype=np.int64)
    model.lineage["gaussian_uid"] = np.arange(len(model), dtype=np.int64)
    gt_m = gt_moving[keep]
    mv_idx = np.nonzero(gt_m)[0]
    withheld = rng.choice(mv_idx, int(round(withhold_frac * len(mv_idx))), replace=False) if withhold_frac else []
    core = np.setdiff1d(mv_idx, withheld)
    model.lineage["material"] = np.zeros(len(model), np.int8)
    model.lineage["material"][core] = MOVING
    model.lineage["provenance"] = np.full(len(model), PROV_ORIGINAL, np.int8)
    model.lineage["provenance"][core] = PROV_ORIGINAL_CORE
    model.active_sh_degree = 0
    return SynthScene(model=model, full=full, gt_moving_model=gt_m, surface_model=name[keep], surface_full=name,
                      joint=joint, cameras=cams, views=views, core=core)


def reobserve(s: SynthScene, full: GaussianTensors, gt_moving: np.ndarray, device="cuda"):
    """Re-render every (camera, frame) observation from a modified ground-truth scene (same cameras, same
    joint, same conventions as make_scene). Returns a new ViewRef list."""
    tr = JointTransport(s.joint, device)
    bg = torch.ones(3, device=device)
    mv = torch.as_tensor(gt_moving, device=device)
    store = {}
    for cam in s.cameras:
        for f in s.joint.frames:
            with torch.no_grad():
                o = render_articulated(cam, tr.at(f), full, mv, bg, "world_SH", 0)
            A = o["alpha"][0]
            mask = A > 0.5
            depth = torch.where(mask, o["depth"][0], torch.zeros_like(A))
            store[(cam.name, f)] = View(cam.name, f, o["color"].clamp(0, 1).cpu(), depth.cpu(), mask.cpu(),
                                        (mask & (depth > 0)).cpu())
    return [ViewRef(c, f, (lambda key=(c.name, f): store[key])) for c in s.cameras for f in s.joint.frames]
