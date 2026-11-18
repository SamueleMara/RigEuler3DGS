"""A completed synthetic Phase-5 run directory (CPU, no training), laid out exactly as tools/train_phase5_rgbd_gaussians.py
writes it: final/point_cloud.ply (articulated, via recovery.articulated_ply.write_articulated_ply), final lineage NPZs,
final/joint/{geometry.json, trajectory.csv} byte copies, joint_unchanged_proof.json, unresolved_originals.npz,
appearance_settings.json, manifest.json (upstream.setup) and complete.json (joint fingerprint of the frozen joint).

Geometry: a static box (body) and a moving box (door / drawer); Gaussians are sampled inside the boxes, the physics
fixture writes the same boxes as OBJ collision meshes, so every plausibility gate has a known answer.
"""
from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np
from plyfile import PlyData, PlyElement

REPO = Path(__file__).resolve().parents[2]
import sys  # noqa: E402

sys.path.insert(0, str(REPO))
from recovery.articulated_ply import write_articulated_ply  # noqa: E402
from recovery.contracts import load_joint  # noqa: E402
from recovery.phase4_artifacts import joint_fingerprint  # noqa: E402

STATIC_BOX = (np.array([-0.20, -0.15, 2.00]), np.array([0.20, 0.15, 2.30]))
MOVING_BOX = {"revolute": (np.array([-0.20, -0.15, 1.97]), np.array([0.20, 0.15, 1.99])),
              "prismatic": (np.array([-0.17, -0.12, 2.01]), np.array([0.17, -0.03, 2.27]))}


def unit(v):
    v = np.asarray(v, float)
    return v / np.linalg.norm(v)


def box_vertices(lo, hi):
    return np.array([[x, y, z] for x in (lo[0], hi[0]) for y in (lo[1], hi[1]) for z in (lo[2], hi[2])], float)


def box_obj_text(lo, hi, T=np.eye(4), scale=1.0, header="box"):
    v = box_vertices(lo, hi)
    v = v @ T[:3, :3].T + T[:3, 3]
    v = v / scale
    # outward, counter-clockwise faces of the 8-corner box (index = 4 ix + 2 iy + iz, 1-based in OBJ)
    F = [(1, 2, 4), (1, 4, 3), (5, 7, 8), (5, 8, 6), (1, 5, 6), (1, 6, 2), (3, 4, 8), (3, 8, 7), (1, 3, 7), (1, 7, 5),
         (2, 6, 8), (2, 8, 4)]
    return f"# {header}\n" + "".join(f"v {a!r} {b!r} {c!r}\n" for a, b, c in v) + "".join(
        f"f {a} {b} {c}\n" for a, b, c in F)


def joint_params(jtype, axis=None, point=None):
    if jtype == "revolute":
        return unit(axis if axis is not None else (0.15, 0.95, -0.25)), np.asarray(
            point if point is not None else (-0.20, 0.01, 1.98), float)
    return unit(axis if axis is not None else (0.10, -0.20, -0.97)), None


def make_phase5_run(root, jtype="revolute", direction="forward", sh_degree=3, n_static=400, n_moving=250, T=15,
                    q_scale=None, seed=0, axis=None, point=None, scene=None, not_for_results=False) -> dict:
    root = Path(root)
    rng = np.random.default_rng(seed)
    scene = scene or f"final_synthetic_{jtype}_{direction}_episode_000000"
    p3 = root / "phase3"
    p5 = root / "phase5" / "run"
    (p3 / jtype).mkdir(parents=True, exist_ok=True)
    (p5 / "final").mkdir(parents=True, exist_ok=True)
    a, o = joint_params(jtype, axis, point)
    geo = {"joint_type": jtype, "status": "FITTED", "geometry_unchanged_bitwise": True,
           "geometry_fixed": ({"axis": a.tolist(), "axis_point": o.tolist()} if jtype == "revolute"
                              else {"direction": a.tolist()})}
    (p3 / jtype / "geometry.json").write_text(json.dumps(geo, indent=1))
    frames = list(range(100, 100 + T)) if direction == "forward" else list(range(100 + T - 1, 99, -1))
    qs = q_scale if q_scale is not None else (0.9 if jtype == "revolute" else 0.12)
    # starts at 0 (anchor), dips negative, then opens: exercises negative q and a non-monotone trajectory
    s = np.linspace(0, 1, T)
    q = qs * (s - 0.35 * np.sin(np.pi * s) * (s < 0.5)) * 1.0
    q[0] = 0.0
    sd = np.concatenate([[1.0], rng.uniform(1e-4, 1e-3, T - 1)])
    (p3 / jtype / "trajectory.csv").write_text("processing_index,original_frame,q,posterior_sd_q\n" + "".join(
        f"{k},{f},{float(x)!r},{float(y)!r}\n" for k, (f, x, y) in enumerate(zip(frames, q, sd))))
    joint = load_joint(p3, jtype, direction)

    # ---- Gaussians with every 3DGS field present and unique per-row stored values
    n = n_static + n_moving
    lo_s, hi_s = STATIC_BOX
    lo_m, hi_m = MOVING_BOX[jtype]
    xyz = np.concatenate([rng.uniform(lo_s, hi_s, (n_static, 3)), rng.uniform(lo_m, hi_m, (n_moving, 3))])
    part = np.concatenate([np.zeros(n_static, np.uint8), np.ones(n_moving, np.uint8)])
    perm = rng.permutation(n)                                   # interleave parts in the row order
    xyz, part = xyz[perm], part[perm]
    K = (sh_degree + 1) ** 2
    names = ["x", "y", "z", "nx", "ny", "nz", "f_dc_0", "f_dc_1", "f_dc_2"] + \
            [f"f_rest_{i}" for i in range(3 * (K - 1))] + ["opacity", "scale_0", "scale_1", "scale_2",
                                                           "rot_0", "rot_1", "rot_2", "rot_3"]
    v = np.empty(n, dtype=[(nm, "<f4") for nm in names])
    for i, nm in enumerate(names):
        v[nm] = rng.normal(size=n).astype(np.float32) * (i + 1) + np.arange(n, dtype=np.float32) * 1e-3
    v["x"], v["y"], v["z"] = xyz[:, 0], xyz[:, 1], xyz[:, 2]
    v["nx"] = v["ny"] = v["nz"] = 0.0
    v["scale_0"], v["scale_1"], v["scale_2"] = (rng.uniform(-6, -3, (3, n))).astype(np.float32)
    rot = rng.normal(size=(n, 4)) * rng.uniform(0.5, 2.0, (n, 1))               # non-unit stored quaternions
    for i in range(4):
        v[f"rot_{i}"] = rot[:, i]
    plain = root / "plain.ply"
    PlyData([PlyElement.describe(v, "vertex")], text=False).write(str(plain))
    uid = rng.permutation(np.arange(1000, 1000 + n)).astype(np.int64)
    orig = np.where(rng.random(n) < 0.8, np.arange(n), -1).astype(np.int64)
    parent = -np.ones(n, np.int64)
    born = np.nonzero(orig < 0)[0]
    parent[born] = uid[rng.choice(np.nonzero(orig >= 0)[0], len(born))]
    prov = np.where(orig >= 0, 1, 3).astype(np.uint8)
    lineage = {"material": part.astype(np.int8), "gaussian_uid": uid, "original_id": orig, "parent_uid": parent,
               "provenance": prov}
    unres_uid = uid[orig >= 0][: max(1, n // 20)]
    ply = p5 / "final" / "point_cloud.ply"
    write_articulated_ply(plain, lineage, joint, ply, unresolved_uids=unres_uid,
                          extra_comments=["phase5 synthetic fixture"])
    np.savez(p5 / "final" / "material_ids.npz", material=lineage["material"])
    np.savez(p5 / "final" / "gaussian_uid.npz", gaussian_uid=uid, original_id=orig)
    np.savez(p5 / "final" / "parent_uid.npz", parent_uid=parent)
    np.savez(p5 / "final" / "provenance.npz", provenance=prov)
    np.savez(p5 / "unresolved_originals.npz", gaussian_uid=unres_uid, row=np.arange(len(unres_uid)),
             status=np.array(["undecided"] * len(unres_uid)), frozen_render_label=np.zeros(len(unres_uid), bool))
    jd = p5 / "final" / "joint"
    jd.mkdir(exist_ok=True)
    for f in ("geometry.json", "trajectory.csv"):
        shutil.copy2(p3 / jtype / f, jd / f)
    fp = joint_fingerprint(joint)
    (p5 / "final" / "joint_unchanged_proof.json").write_text(json.dumps(
        {"joint_optimized": False, "identical": True, "fingerprint_before": fp, "fingerprint_after": fp}))
    (p5 / "final" / "appearance_settings.json").write_text(json.dumps(
        {"sh_mode": "local_SH", "active_sh_degree": sh_degree, "background": [0.5, 0.5, 0.5], "antialiasing": False}))
    setup = {"source": f"/nonexistent/datasets/{scene}", "joint_type": jtype, "direction": direction,
             "phase3_dir": str(p3)}
    (p5 / "manifest.json").write_text(json.dumps({"mode": "gaussian_rgbd", "upstream": {"stage": "phase5",
                                                                                        "setup": setup}}))
    completion = {"stage": "phase5", "train_mode": "gaussian_rgbd", "joint_fingerprint": fp, "final_gaussians": n,
                  "joint_unchanged": True, "NOT_FOR_RESULTS": bool(not_for_results)}
    (p5 / "complete.json").write_text(json.dumps(completion, indent=1))
    plain.unlink()
    return {"phase5_dir": p5, "joint": joint, "scene": scene, "setup": setup, "completion": completion,
            "axis": a, "point": o, "part": part, "xyz": xyz}


# ------------------------------------------------------------------------------------------------ physics fixture
def box_inertia(lo, hi, mass):
    e = hi - lo
    return mass / 12.0 * np.diag([e[1] ** 2 + e[2] ** 2, e[0] ** 2 + e[2] ** 2, e[0] ** 2 + e[1] ** 2])


def make_physics_spec(run: dict, root, moving_frame=None, units="m", registration=True, lower=None, upper=None,
                      moving_inertia="explicit", extra=None) -> Path:
    """Box collision meshes matching the fixture, the moving box posed at `moving_frame` (default: the last frame)
    in capture coordinates; static inertia explicit (box analytic), moving explicit (posed at the same frame) or
    density; T_world_capture a declared rigid transform."""
    import yaml
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    joint = run["joint"]
    jt = joint.type
    fr = joint.chronological()[0]
    f = int(fr[-1]) if moving_frame is None else int(moving_frame)
    Q, p = joint.transform(f)
    Tf = np.eye(4)
    Tf[:3, :3], Tf[:3, 3] = Q, p
    scale = {"m": 1.0, "cm": 0.01, "mm": 0.001}[units]
    # the static mesh is supplied in its own mesh coordinates: T_capture_mesh is a non-trivial rigid transform
    ang = 0.3
    Tcm = np.eye(4)
    Tcm[:3, :3] = np.array([[np.cos(ang), -np.sin(ang), 0], [np.sin(ang), np.cos(ang), 0], [0, 0, 1]])
    Tcm[:3, 3] = [0.05, -0.02, 0.10]
    lo_s, hi_s = STATIC_BOX
    lo_m, hi_m = MOVING_BOX[jt]
    (root / "static_mesh.obj").write_text(box_obj_text(lo_s, hi_s, np.linalg.inv(Tcm), scale, "static box"))
    (root / "moving_mesh.obj").write_text(box_obj_text(lo_m, hi_m, Tf, scale, f"moving box posed at frame {f}"))
    ms, mm = 12.0, 1.5
    Is = box_inertia(lo_s, hi_s, ms)
    com_m_f = Q @ (0.5 * (lo_m + hi_m)) + p
    Im_f = Q @ box_inertia(lo_m, hi_m, mm) @ Q.T
    q = np.asarray(joint.q)
    lo = float(q.min() - 0.2 * (q.max() - q.min())) if lower is None else lower
    hi = float(q.max() + 0.2 * (q.max() - q.min())) if upper is None else upper
    lo, hi = min(lo, 0.0), max(hi, 0.0)
    Tw = np.eye(4)
    Tw[:3, :3] = np.array([[1, 0, 0], [0, 0, 1], [0, -1, 0]], float)          # capture y-down camera -> z-up world
    Tw[:3, 3] = [0.3, -0.1, 2.5]
    d = {"schema_version": 1,
         "registration": ({"T_world_capture": Tw.tolist(), "gravity_alignment": "declared",
                           "provenance": "synthetic fixture: declared"} if registration else None),
         "collision": {"static": {"source": "static_mesh.obj", "units": units, "T_capture_mesh": Tcm.tolist(),
                                  "provenance": "measured"},
                       "moving": {"source": "moving_mesh.obj", "units": units, "T_capture_mesh": np.eye(4).tolist(),
                                  "at_original_frame": f, "provenance": "measured"}},
         "links": {"static_link": {"mass_kg": ms, "com_capture_m": (0.5 * (lo_s + hi_s)).tolist(),
                                   "inertia_com_capture_kg_m2": Is.tolist(), "provenance": "measured"},
                   "moving_link": ({"mass_kg": mm, "com_capture_m": com_m_f.tolist(),
                                    "inertia_com_capture_kg_m2": Im_f.tolist(), "at_original_frame": f,
                                    "provenance": "measured"} if moving_inertia == "explicit" else
                                   {"density_kg_m3": 600.0, "provenance": "density_assumption"})},
         "joint": {"lower": lo, "upper": hi, "velocity_limit": 2.0, "effort_limit": 5.0, "damping": 0.3,
                   "coulomb_friction": 0.05, "provenance": "measured"},
         "contact": {"static_friction": 0.8, "dynamic_friction": 0.6, "restitution": 0.0, "provenance": "estimated"}}
    if extra:
        for k, val in extra.items():
            cur = d
            ks = k.split(".")
            for kk in ks[:-1]:
                cur = cur[kk]
            if val is _DELETE:
                cur.pop(ks[-1], None)
            else:
                cur[ks[-1]] = val
    path = root / "physics_spec.yaml"
    path.write_text(yaml.safe_dump(d, sort_keys=False))
    return path


class _Delete:
    pass


_DELETE = _Delete()
DELETE = _DELETE
