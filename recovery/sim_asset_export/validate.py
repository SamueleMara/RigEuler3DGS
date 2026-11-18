"""Validation of an exported asset against the frozen Phase-3 joint (spec 3, 8). Every check reports PASS / FAIL /
NOT_RUN with its measured error and declared tolerance; NOT_RUN is never PASS.

Reference: recovery.contracts.Joint.transform(f) (float64), applied to the stored PLY means, for EVERY recorded
Phase-3 frame. Engines are executed in their own interpreters (subprocess); their readback is compared here.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

from .frames import M_z, R_from_quat_wxyz, T_B_P_moving, apply, inv_se3, quat_mul, quat_wxyz_from_R, se3
from .schema import JOINT_NAME

# declared tolerances: float64 analytic / text round trip vs float32 engine storage
TOL = {
    "analytic": {"pos_m": 1e-9, "rot": 1e-9},
    "urdf": {"pos_m": 1e-9, "rot": 1e-9, "value_rel": 1e-12},
    "mujoco": {"pos_m": 1e-8, "rot": 1e-8, "inertia_rel": 1e-6, "mesh_vertex_m": 2e-6},
    "usd": {"pos_m": 2e-6, "rot": 2e-6, "inertia_rel": 2e-6, "value_rel": 2e-6},
    "sapien": {"pos_m": 2e-5, "rot": 2e-5, "inertia_rel": 2e-5},
    "parttwin_adapter": {"pos_m": 5e-5, "quat": 5e-5},
}
PASS, FAIL, NOT_RUN = "PASS", "FAIL", "NOT_RUN"
HERE = Path(__file__).parent


class ValidationError(RuntimeError):
    pass


# ------------------------------------------------------------------------------------------------ runtimes
def _probe(py, modules) -> str | None:
    try:
        code = "import importlib;" + ";".join(f"importlib.import_module({m!r})" for m in modules)
        r = subprocess.run([str(py), "-c", code], capture_output=True, timeout=180)
        return str(py) if r.returncode == 0 else None
    except (OSError, subprocess.TimeoutExpired):
        return None


def find_runtime(kind: str, explicit=None) -> dict:
    """{"python": path | None, "source": ...}. Order: explicit > RIGEULER_<KIND>_PYTHON > RIGEULER_SIM_PYTHON (usd,
    mujoco) > the current interpreter. Never a hard-coded workstation path."""
    mods = {"usd": ["pxr.Usd", "pxr.UsdPhysics"], "mujoco": ["mujoco", "numpy"], "sapien": ["sapien", "numpy"]}[kind]
    cands = []
    if explicit:
        cands.append(("explicit", explicit))
    env = os.environ.get(f"RIGEULER_{kind.upper()}_PYTHON")
    if env:
        cands.append((f"RIGEULER_{kind.upper()}_PYTHON", env))
    if kind in ("usd", "mujoco") and os.environ.get("RIGEULER_SIM_PYTHON"):
        cands.append(("RIGEULER_SIM_PYTHON", os.environ["RIGEULER_SIM_PYTHON"]))
    cands.append(("current_interpreter", sys.executable))
    tried = []
    for src, py in cands:
        if _probe(py, mods):
            return {"python": str(py), "source": src}
        tried.append(f"{src}={py}")
        if src == "explicit":
            raise ValidationError(f"{kind} runtime {py} cannot import {mods}")
    return {"python": None, "source": None, "tried": tried}


# ------------------------------------------------------------------------------------------------ helpers
def _rot_err(A, B):
    return float(np.abs(np.asarray(A)[:3, :3] - np.asarray(B)[:3, :3]).max())


def _pos_err(A, B):
    return float(np.abs(np.asarray(A)[:3, 3] - np.asarray(B)[:3, 3]).max())


def _status(ok):
    return PASS if ok else FAIL


def _rel(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    return float(np.abs(a - b).max() / max(np.abs(b).max(), 1e-300))


def joint_T(joint, f):
    Q, p = joint.transform(int(f))
    return se3(Q, p)


def expected_link_world(asset, q):
    """T_W_static, T_W_moving(q)."""
    return asset.T_W_B, asset.T_W_B @ asset.T_B_J0 @ M_z(asset.joint_type, q)


def test_qs(asset) -> list:
    q = list(map(float, asset.trajectory["q"]))
    extra = [0.0]
    if asset.physics is not None:
        lo, hi = asset.physics.joint.lower, asset.physics.joint.upper
        extra += [lo, hi, 0.5 * lo, 0.5 * hi]
    return q + extra


# ------------------------------------------------------------------------------------------------ checks
def check_analytic(asset, joint, moving_xyz, seed=0) -> dict:
    """T_B_Lmoving(q_f) T_L_P x == Q_f x + p_f for every Phase-3 frame, random points + the moving Gaussians;
    q0 identity; covariance / quaternion composition; SH direction invariance when camera and link move together."""
    rng = np.random.default_rng(seed)
    lo, hi = moving_xyz.min(0), moving_xyz.max(0)
    pts = np.concatenate([rng.uniform(lo - 0.5, hi + 0.5, (256, 3)),
                          moving_xyz[rng.choice(len(moving_xyz), min(2000, len(moving_xyz)), replace=False)]])
    qP = rng.normal(size=(64, 4))
    qP /= np.linalg.norm(qP, axis=1, keepdims=True)
    S = rng.normal(size=(64, 3, 3))
    Sig = S @ S.transpose(0, 2, 1)
    cam = rng.uniform(lo - 2, hi + 2, (16, 3))
    e = {"pos_m": 0.0, "rot": 0.0, "quat_compose": 0.0, "cov": 0.0, "sh_dir": 0.0}
    for f in joint.frames:
        q = joint.q_of(f)
        T = T_B_P_moving(asset.joint_type, asset.T_B_J0, q)
        Tj = joint_T(joint, f)
        e["rot"] = max(e["rot"], _rot_err(T, Tj))
        e["pos_m"] = max(e["pos_m"], float(np.abs(apply(T, pts) - apply(Tj, pts)).max()))
        R = T[:3, :3]
        qw = quat_mul(np.broadcast_to(quat_wxyz_from_R(R), qP.shape), qP)
        Rw = np.stack([R_from_quat_wxyz(x) for x in qw])
        Rp = np.stack([R_from_quat_wxyz(x) for x in qP])
        e["quat_compose"] = max(e["quat_compose"], float(np.abs(Rw - R @ Rp).max()))
        e["cov"] = max(e["cov"], float(np.abs(R @ Sig @ R.T - Tj[:3, :3] @ Sig @ Tj[:3, :3].T).max()))
        mu_w, c_w = apply(T, pts[:16]), apply(T, cam)
        d_p = (mu_w[:, None] - c_w[None]) @ R                       # R^T (mu_W - c_W), row form
        e["sh_dir"] = max(e["sh_dir"], float(np.abs(d_p - (pts[:16, None] - cam[None])).max()))
    T0 = T_B_P_moving(asset.joint_type, asset.T_B_J0, 0.0)
    e0 = float(np.abs(T0 - np.eye(4)).max())
    t = TOL["analytic"]
    ok = e["pos_m"] <= t["pos_m"] and max(e["rot"], e["quat_compose"], e["sh_dir"]) <= t["rot"] and e0 <= 1e-12
    Rbj = asset.T_B_J0[:3, :3]
    axis_ok = float(np.abs(Rbj[:, 2] - asset.axis).max()) <= 1e-12 and abs(np.linalg.det(Rbj) - 1) <= 1e-12
    return {"status": _status(ok and axis_ok), "frames": len(joint.frames), "points": len(pts), "max_error": e,
            "q0_identity_error": e0, "local_z_equals_axis": axis_ok, "tolerance": t,
            "revolute_point_invariance": (float(np.abs(apply(T_B_P_moving("revolute", asset.T_B_J0, 0.7),
                                                             asset.origin[None]) - asset.origin).max())
                                          if asset.joint_type == "revolute" else None)}


def check_urdf(path: Path, asset, joint) -> dict:
    from scipy.spatial.transform import Rotation
    root = ET.parse(path).getroot()
    links = [l.get("name") for l in root.findall("link")]
    joints = root.findall("joint")
    errs = []
    if root.tag != "robot" or links != ["static_link", "moving_link"]:
        errs.append(f"links {links}")
    if len(joints) != 1:
        errs.append(f"{len(joints)} joints")
    j = joints[0]
    if (j.get("name"), j.get("type"), j.find("parent").get("link"), j.find("child").get("link")) != \
            (JOINT_NAME, asset.joint_type, "static_link", "moving_link"):
        errs.append("joint name / type / parent / child")
    o = j.find("origin")
    xyz = np.array(o.get("xyz").split(), float)
    rpy = np.array(o.get("rpy").split(), float)
    T_origin = se3(Rotation.from_euler("xyz", rpy).as_matrix(), xyz)          # independent of frames.py
    if j.find("axis").get("xyz").split() != ["0", "0", "1"]:
        errs.append("axis is not 0 0 1")
    lim = j.find("limit")
    jl = asset.physics.joint
    got = {k: float(lim.get(k)) for k in ("lower", "upper", "effort", "velocity")}
    want = {"lower": jl.lower, "upper": jl.upper, "effort": jl.effort, "velocity": jl.velocity}
    if got != want:
        errs.append(f"limits {got} != {want}")
    dyn = j.find("dynamics")
    if (float(dyn.get("damping")), float(dyn.get("friction"))) != (jl.damping, jl.coulomb_friction):
        errs.append("dynamics")
    inert, meshes = {}, []
    for L in root.findall("link"):
        n = L.get("name")
        ie = L.find("inertial")
        I = {k: float(ie.find("inertia").get(k)) for k in ("ixx", "ixy", "ixz", "iyy", "iyz", "izz")}
        Im = np.array([[I["ixx"], I["ixy"], I["ixz"]], [I["ixy"], I["iyy"], I["iyz"]], [I["ixz"], I["iyz"], I["izz"]]])
        com = np.array(ie.find("origin").get("xyz").split(), float)
        ref = asset.physics.inertial[n]
        inert[n] = {"mass": float(ie.find("mass").get("value")) == ref.mass,
                    "com": bool(np.array_equal(com, ref.com)), "inertia": bool(np.array_equal(Im, ref.inertia)),
                    "rpy_zero": ie.find("origin").get("rpy") == "0 0 0"}
        if not all(inert[n].values()):
            errs.append(f"inertial {n} {inert[n]}")
        for c in L.findall("collision"):
            fn = c.find("geometry/mesh").get("filename")
            meshes.append(fn)
            if Path(fn).is_absolute() or not (path.parent / fn).is_file():
                errs.append(f"collision mesh {fn} is absolute or missing")
    e_o = max(_rot_err(T_origin, asset.T_B_J0), _pos_err(T_origin, asset.T_B_J0))
    e_fk = 0.0
    for f in joint.frames:
        T = T_origin @ M_z(asset.joint_type, joint.q_of(f)) @ inv_se3(T_origin)
        e_fk = max(e_fk, _rot_err(T, joint_T(joint, f)), _pos_err(T, joint_T(joint, f)))
    t = TOL["urdf"]
    if e_o > t["rot"] or e_fk > t["pos_m"]:
        errs.append(f"origin error {e_o:.3g}, FK error {e_fk:.3g}")
    return {"status": _status(not errs), "errors": errs, "links": links, "collision_meshes": meshes,
            "origin_vs_T_B_J0": e_o, "fk_max_error_all_frames": e_fk, "inertial": inert, "tolerance": t,
            "rpy_convention": "URDF fixed-axis xyz (R = Rz Ry Rx), read back with scipy from_euler('xyz')"}


def _run(py, script, args, timeout=900):
    r = subprocess.run([py, str(HERE / script), *map(str, args)], capture_output=True, text=True, timeout=timeout)
    if r.returncode != 0:
        raise ValidationError(f"{script} failed in {py}: {r.stderr.strip()[-3000:]}")


def _contact_request(asset):
    from .collision import Mesh  # noqa: F401
    pts = np.concatenate([apply(asset.T_W_B, p.vertices) for p in asset.physics.collision["static_link"].parts])
    ext = float(np.linalg.norm(pts.max(0) - pts.min(0)))
    r = max(0.005, 0.03 * ext)
    ztop = pts[:, 2].max()
    top = pts[pts[:, 2] >= ztop - 0.01 * ext]
    xy = top[:, :2].mean(0)
    return {"drop_world": [float(xy[0]), float(xy[1]), float(ztop + 3 * r)], "radius": r, "seconds": 2.0,
            "z_top": float(ztop)}


def _contact_report(c, req):
    if c is None:
        return {"status": NOT_RUN}
    z = c["final_world"][2] - req["z_top"]
    rest = 0.5 * req["radius"] <= z <= 1.5 * req["radius"]
    return {"status": "RESTING_ON_STATIC" if rest else "DEVIATION", "final_height_above_static_top_m": z,
            "radius_m": req["radius"], "penetration_m": req["radius"] - z, "engine": c,
            "note": "smoke test of collision geometry (articulation held still by gravity compensation, probe sphere "
                    "dropped on the static top); dynamic responses of different solvers are reported, not "
                    "claimed equal; informational only (gravity is -Z of W"
                    + (", whose alignment is UNVERIFIED)" if asset_gravity_unverified(req) else ")")}


def asset_gravity_unverified(req):
    return bool(req.get("gravity_unverified"))


def check_mujoco(xml: Path, asset, joint, python) -> dict:
    if not python:
        return {"status": NOT_RUN, "reason": "no interpreter with mujoco (set RIGEULER_MUJOCO_PYTHON)"}
    qs = test_qs(asset)
    creq = _contact_request(asset)
    creq["gravity_unverified"] = asset.registration.gravity_alignment == "unverified"
    with tempfile.TemporaryDirectory() as tmp:
        rq, oj, on = Path(tmp) / "req.json", Path(tmp) / "out.json", Path(tmp) / "out.npz"
        rq.write_text(json.dumps({"q": qs, "contact": creq}))
        _run(python, "_mujoco_check.py", [xml, rq, oj, on])
        out = json.loads(oj.read_text())
        verts = dict(np.load(on))
    errs = []
    t = TOL["mujoco"]
    if (out["nq"], out["nv"], out["njnt"]) != (1, 1, 1) or out["bodies"] != ["world", "static_link", "moving_link"]:
        errs.append(f"structure nq {out['nq']} nv {out['nv']} njnt {out['njnt']} bodies {out['bodies']}")
    jo = out["joint"]
    jl = asset.physics.joint
    want = "hinge" if asset.joint_type == "revolute" else "slide"
    if jo["type_name"] != want or jo["axis"] != [0.0, 0.0, 1.0] or jo["pos"] != [0.0, 0.0, 0.0] or \
            jo["body"] != "moving_link" or not jo["limited"]:
        errs.append(f"joint {jo}")
    if _rel(jo["range"], [jl.lower, jl.upper]) > 1e-12 or jo["damping"] != jl.damping or \
            jo["frictionloss"] != jl.coulomb_friction:
        errs.append(f"joint parameters {jo}")
    e_pose = 0.0
    for q, pose in zip(qs, out["poses"]):
        Ts, Tm = expected_link_world(asset, q)
        for nm, Te in (("static_link", Ts), ("moving_link", Tm)):
            Tg = se3(pose[nm]["xmat"], pose[nm]["xpos"])
            e_pose = max(e_pose, _pos_err(Tg, Te), _rot_err(Tg, Te))
    if e_pose > t["pos_m"]:
        errs.append(f"pose error {e_pose:.3g}")
    e_in = {}
    for lk in ("static_link", "moving_link"):
        r = out["inertial"][lk]
        ref = asset.physics.inertial[lk]
        R = R_from_quat_wxyz(r["iquat"])
        I = R @ np.diag(r["inertia"]) @ R.T
        e_in[lk] = {"mass_rel": abs(r["mass"] - ref.mass) / ref.mass, "com_m": float(np.abs(np.array(r["ipos"]) -
                                                                                          ref.com).max()),
                    "inertia_rel": _rel(I, ref.inertia)}
        if e_in[lk]["mass_rel"] > 1e-12 or e_in[lk]["com_m"] > t["pos_m"] or e_in[lk]["inertia_rel"] > t["inertia_rel"]:
            errs.append(f"inertial {lk} {e_in[lk]}")
    from scipy.spatial import cKDTree
    e_mesh = 0.0
    for lk in ("static_link", "moving_link"):
        T = expected_link_world(asset, 0.0)[0 if lk == "static_link" else 1]
        for k, part in enumerate(asset.physics.collision[lk].parts):
            nm = f"{lk}_collision_{k:03d}"
            if nm not in verts:
                errs.append(f"mesh geom {nm} missing")
                continue
            a, b = apply(T, part.vertices), verts[nm]
            e_mesh = max(e_mesh, float(cKDTree(b).query(a)[0].max()), float(cKDTree(a).query(b)[0].max()))
    if e_mesh > t["mesh_vertex_m"]:
        errs.append(f"mesh vertices displaced by {e_mesh:.3g} m")
    return {"status": _status(not errs), "errors": errs, "mujoco_version": out["mujoco_version"],
            "q_evaluated": len(qs), "pose_max_error": e_pose, "inertial_error": e_in, "mesh_vertex_max_error_m": e_mesh,
            "joint_readback": jo, "contact": _contact_report(out.get("contact"), creq), "tolerance": t}


def _T_from_pq(p, q_wxyz):
    return se3(R_from_quat_wxyz(q_wxyz), p)


def check_usd(readback: dict | None, asset, joint, asset_dir: Path) -> dict:
    if readback is None:
        return {"status": NOT_RUN, "reason": "no interpreter with pxr (set RIGEULER_USD_PYTHON)"}
    rb = readback
    errs = []
    t = TOL["usd"]
    if rb["meters_per_unit"] != 1.0 or rb["kilograms_per_unit"] != 1.0 or rb["up_axis"] != "Z":
        errs.append("stage units / up axis")
    usd_file = asset_dir / "isaac" / "model.usd"
    magic = usd_file.read_bytes()[:8] if usd_file.is_file() else b""
    if magic != b"PXR-USDC":
        errs.append(f"model.usd is not a binary crate file (magic {magic!r}; format id {rb['layer_format']})")
    root = "/" + asset.name
    if rb["default_prim"] != root or rb["articulation_roots"] != [root]:
        errs.append(f"default prim {rb['default_prim']} articulation roots {rb['articulation_roots']}")
    if sorted(rb["links"]) != ["moving_link", "static_link"]:
        errs.append(f"rigid bodies {sorted(rb['links'])}")
    Ts, Tm0 = expected_link_world(asset, 0.0)
    e_body = max(_pos_err(rb["links"]["static_link"]["T_world"], Ts), _rot_err(rb["links"]["static_link"]["T_world"], Ts),
                 _pos_err(rb["links"]["moving_link"]["T_world"], Tm0),
                 _rot_err(rb["links"]["moving_link"]["T_world"], Tm0))
    if e_body > t["pos_m"]:
        errs.append(f"body poses error {e_body:.3g}")
    e_in = {}
    for lk in ("static_link", "moving_link"):
        L, ref = rb["links"][lk], asset.physics.inertial[lk]
        R = R_from_quat_wxyz(L["principal_axes_wxyz"])
        e_in[lk] = {"mass_rel": abs(L["mass"] - ref.mass) / ref.mass,
                    "com_m": float(np.abs(np.array(L["com"]) - ref.com).max()),
                    "inertia_rel": _rel(R @ np.diag(L["diagonal_inertia"]) @ R.T, ref.inertia)}
        if max(e_in[lk]["mass_rel"], e_in[lk]["inertia_rel"]) > t["inertia_rel"] or e_in[lk]["com_m"] > t["pos_m"]:
            errs.append(f"mass {lk} {e_in[lk]}")
        ply = (asset_dir / "isaac" / L["visual_ply"]).resolve()
        if Path(L["visual_ply"]).is_absolute() or not ply.is_file() or \
                np.abs(np.reshape(L["T_link_ply"], (4, 4)) - asset.T_link_ply[lk]).max() > 0:
            errs.append(f"visual reference / T_link_ply of {lk}")
    J = rb["joints"].get(JOINT_NAME)
    jl = asset.physics.joint
    rev = asset.joint_type == "revolute"
    k = 180.0 / np.pi if rev else 1.0
    if J is None:
        errs.append("no articulation joint")
        e_fk = float("inf")
    else:
        if J["type"] != ("PhysicsRevoluteJoint" if rev else "PhysicsPrismaticJoint") or J["axis"] != "Z" or \
                J["body0"] != [root + "/static_link"] or J["body1"] != [root + "/moving_link"]:
            errs.append(f"joint schema {J['type']} axis {J['axis']} bodies {J['body0']} {J['body1']}")
        if J.get("collision_enabled") is not False:
            errs.append("articulation joint must author physics:collisionEnabled = false")
        lim_err = max(abs(J["lower"] - jl.lower * k), abs(J["upper"] - jl.upper * k)) / max(abs(jl.lower * k),
                                                                                            abs(jl.upper * k), 1e-30)
        if lim_err > t["value_rel"]:
            errs.append(f"limits {J['lower']} {J['upper']} vs {jl.lower * k} {jl.upper * k} "
                        f"({'deg' if rev else 'm'})")
        if abs(J["drive"]["damping"] - jl.damping / k) > t["value_rel"] * max(jl.damping / k, 1e-30) or \
                J["drive"]["stiffness"] != 0.0:
            errs.append(f"drive {J['drive']}")
        T0 = _T_from_pq(J["localPos0"], J["localRot0_wxyz"])
        T1 = _T_from_pq(J["localPos1"], J["localRot1_wxyz"])
        Tb0 = np.asarray(rb["links"]["static_link"]["T_world"])
        e_fk = 0.0
        for q in test_qs(asset):
            Tl = Tb0 @ T0 @ M_z(asset.joint_type, q) @ inv_se3(T1)
            Te = expected_link_world(asset, q)[1]
            e_fk = max(e_fk, _pos_err(Tl, Te), _rot_err(Tl, Te))
        if e_fk > t["pos_m"]:
            errs.append(f"joint-frame FK error {e_fk:.3g}")
        lim_err = float(lim_err)
    fx = rb["joints"].get("root_fixed")
    if fx is None or fx["body0"] != [] or fx["body1"] != [root + "/static_link"]:
        errs.append("fixed base joint")
    e_col = 0.0
    for path, c in rb["collisions"].items():
        lk = c["parent"]
        kk = int(path.rsplit("_", 1)[1])
        want = apply(expected_link_world(asset, 0.0)[0 if lk == "static_link" else 1],
                     asset.physics.collision[lk].parts[kk].vertices)
        got = apply(np.asarray(c["T_world"]), np.asarray(c["points"]))
        e_col = max(e_col, float(np.abs(got - want).max()))
        if c["approximation"] != "convexHull":
            errs.append(f"{path} approximation {c['approximation']}")
    n_parts = sum(len(asset.physics.collision[lk].parts) for lk in ("static_link", "moving_link"))
    if len(rb["collisions"]) != n_parts or e_col > t["pos_m"] * 10:
        errs.append(f"collision meshes: {len(rb['collisions'])} of {n_parts}, max vertex error {e_col:.3g}")
    return {"status": _status(not errs), "errors": errs, "usd_version": rb["usd_version"],
            "body_pose_error": e_body, "joint_frame_fk_error": e_fk, "inertial_error": e_in,
            "collision_vertex_error_m": e_col, "physx_schema_plugin": rb["physx_schema_plugin"],
            "angle_units": "deg (USD revolute limits / drive)" if rev else "m", "tolerance": t,
            "isaac_runtime": NOT_RUN}


def check_sapien(urdf: Path, object_yaml: Path, asset, joint, python, visual: dict) -> dict:
    if not python:
        return {"status": NOT_RUN, "reason": "no interpreter with sapien (set RIGEULER_SAPIEN_PYTHON)"}
    qs = test_qs(asset)
    fr = list(joint.chronological()[0])
    pick = sorted({fr[0], fr[len(fr) // 2], fr[-1], fr[int(np.argmin(asset.trajectory["q"]))],
                   fr[int(np.argmax(asset.trajectory["q"]))]})
    aq = [joint.q_of(f) for f in pick]
    creq = _contact_request(asset)
    creq["gravity_unverified"] = asset.registration.gravity_alignment == "unverified"
    with tempfile.TemporaryDirectory() as tmp:
        rq, oj, on = Path(tmp) / "req.json", Path(tmp) / "out.json", Path(tmp) / "out.npz"
        rq.write_text(json.dumps({"q": qs, "T_world_root": asset.T_W_B.tolist(), "contact": creq,
                                  "object_yaml": str(object_yaml), "adapter_q": aq}))
        _run(python, "_sapien_check.py", [urdf, rq, oj, on])
        out = json.loads(oj.read_text())
        arr = dict(np.load(on))
    t = TOL["sapien"]
    errs = []
    if out["dof"] != 1 or out["links"] != ["static_link", "moving_link"] or out["joints"] != [JOINT_NAME]:
        errs.append(f"structure dof {out['dof']} links {out['links']} joints {out['joints']}")
    jl = asset.physics.joint
    lim = out["joint"]["limits"][0]
    if abs(lim[0] - jl.lower) > 1e-6 or abs(lim[1] - jl.upper) > 1e-6:
        errs.append(f"limits {lim}")
    e_pose = 0.0
    for q, pose in zip(qs, out["poses"]):
        Ts, Tm = expected_link_world(asset, q)
        e_pose = max(e_pose, _pos_err(pose["static_link"], Ts), _rot_err(pose["static_link"], Ts),
                     _pos_err(pose["moving_link"], Tm), _rot_err(pose["moving_link"], Tm))
    if e_pose > t["pos_m"]:
        errs.append(f"pose error {e_pose:.3g}")
    e_in = {}
    for lk in ("static_link", "moving_link"):
        r, ref = out["inertial"][lk], asset.physics.inertial[lk]
        C = np.asarray(r["cmass_local_pose"])
        I = C[:3, :3] @ np.diag(r["inertia"]) @ C[:3, :3].T
        e_in[lk] = {"mass_rel": abs(r["mass"] - ref.mass) / ref.mass, "com_m": float(np.abs(C[:3, 3] - ref.com).max()),
                    "inertia_rel": _rel(I, ref.inertia), "collision_shapes": r["collision_shapes"]}
        if e_in[lk]["mass_rel"] > t["inertia_rel"] or e_in[lk]["com_m"] > t["pos_m"] or \
                e_in[lk]["inertia_rel"] > t["inertia_rel"]:
            errs.append(f"inertial {lk} {e_in[lk]}")
    deviations = {"joint_friction": {"urdf": jl.coulomb_friction, "sapien": out["joint"]["friction"],
                                     "note": "SAPIEN maps URDF <dynamics friction> to a PhysX joint friction "
                                             "coefficient; values are reported, not equated"},
                  "joint_damping": {"urdf": jl.damping, "sapien": out["joint"]["damping"]},
                  "max_velocity": {"urdf": jl.velocity, "sapien": out["joint"]["max_velocity"]}}
    pt = {"status": NOT_RUN}
    if "parttwin" in out and "means_0" in arr:
        ta = TOL["parttwin_adapter"]
        from plyfile import PlyData
        means, quats = [], []
        for lk in out["parttwin"]["links"]:
            v = PlyData.read(str(visual[lk]))["vertex"].data
            means.append(np.stack([v["x"], v["y"], v["z"]], 1).astype(np.float64))
            qq = np.stack([v[f"rot_{i}"] for i in range(4)], 1).astype(np.float64)
            quats.append(qq / np.linalg.norm(qq, axis=1, keepdims=True))
        e_m = e_q = 0.0
        for k, f in enumerate(pick):
            exp_m, exp_q = [], []
            for lk, mu, qq in zip(out["parttwin"]["links"], means, quats):
                if lk == "static_link":
                    T = asset.T_W_B
                else:
                    T = asset.T_W_B @ joint_T(joint, f)
                exp_m.append(apply(T, mu))
                exp_q.append(quat_mul(np.broadcast_to(quat_wxyz_from_R(T[:3, :3]), qq.shape), qq))
            exp_m, exp_q = np.concatenate(exp_m), np.concatenate(exp_q)
            e_m = max(e_m, float(np.abs(arr[f"means_{k}"] - exp_m).max()))
            gq = arr[f"quats_{k}"]
            e_q = max(e_q, float(np.minimum(np.abs(gq - exp_q).max(1), np.abs(gq + exp_q).max(1)).max()))
        ok = e_m <= ta["pos_m"] and e_q <= ta["quat"] and out["parttwin"].get("T_world_root_equal", False)
        pt = {"status": _status(ok), "frames": pick, "means_max_error_m": e_m, "quat_max_error": e_q,
              "tolerance": ta, "counts": out["parttwin"]["counts"],
              "reference": "Joint.transform(f) applied to the stored PLY means / quaternions (float64)"}
        if not ok:
            errs.append(f"PArtTwin1D adapter transport error means {e_m:.3g} quats {e_q:.3g}")
    elif "parttwin" in out:
        pt = out["parttwin"]
    return {"status": _status(not errs), "errors": errs, "sapien_version": out["sapien_version"],
            "q_evaluated": len(qs), "pose_max_error": e_pose, "inertial_error": e_in, "engine_deviations": deviations,
            "parttwin1d_consumer": pt, "contact": _contact_report(out.get("contact"), creq), "tolerance": t}


def check_object_yaml(path: Path, asset) -> dict:
    import yaml
    d = yaml.safe_load(path.read_text())
    allowed = {"schema_version", "object_name", "urdf", "fixed_base", "T_world_root", "links", "cameras", "initial_q",
               "render", "viewer", "sim", "robot", "notes"}
    errs = [f"unknown keys {sorted(set(d) - allowed)}"] if set(d) - allowed else []
    for lk, b in d["links"].items():
        if set(b) - {"ply", "T_link_ply", "notes"}:
            errs.append(f"links.{lk} keys")
        if Path(b["ply"]).is_absolute() or not (path.parent / b["ply"]).is_file():
            errs.append(f"links.{lk}.ply {b['ply']}")
        if np.abs(np.asarray(b["T_link_ply"]) - asset.T_link_ply[lk]).max() > 0:
            errs.append(f"links.{lk}.T_link_ply")
    if np.abs(np.asarray(d["T_world_root"]) - asset.T_W_B).max() > 0:
        errs.append("T_world_root")
    if Path(d["urdf"]).is_absolute():
        errs.append("urdf path absolute")
    return {"status": _status(not errs), "errors": errs, "urdf_present": (path.parent / d["urdf"]).is_file()}


def check_portable(asset_dir: Path, forbidden: list) -> dict:
    """No absolute workstation path in any runtime file (export_manifest.json keeps provenance paths on purpose)."""
    bad = []
    for p in sorted(asset_dir.rglob("*")):
        if not p.is_file() or p.name in ("export_manifest.json", "export_complete.json") or \
                p.suffix in (".ply", ".npz", ".usd"):
            continue
        txt = p.read_text(errors="ignore")
        for s in forbidden:
            if s and s in txt:
                bad.append(f"{p.relative_to(asset_dir)} contains {s}")
    return {"status": _status(not bad), "errors": bad, "forbidden_substrings": forbidden}
