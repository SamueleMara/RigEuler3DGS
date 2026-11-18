"""USD for Isaac Sim / Isaac Lab (spec 6B), generated from the CanonicalAsset only, written by a dedicated OpenUSD
interpreter (`_usd_writer.py` in a subprocess) -- Isaac Sim is never imported into the training environment.

Stage: metersPerUnit 1, kilogramsPerUnit 1, upAxis Z, binary crate (.usd), /PhysicsScene gravity -Z of W.
    /<name>                  Xform = T_W_B, ArticulationRootAPI, rigeuler:* metadata (asset.yaml, PLYs, units)
      static_link            Xform = I,       RigidBodyAPI + MassAPI, collision_k Meshes (convexHull, purpose guide)
      moving_link            Xform = T_B_J0,  (q = 0 pose)
      joints/articulation_joint  Revolute|PrismaticJoint, body0 static_link, body1 moving_link,
                                 localPos0/localRot0 = T_B_J0 (static frame), localPos1/localRot1 = identity, axis Z
      joints/root_fixed      FixedJoint world -> static_link (fixed base)
Unit conversions (explicit): revolute limits rad -> deg; revolute drive damping N m s/rad -> N m s/deg (x pi/180);
maxJointVelocity rad/s -> deg/s. Prismatic: metres, unchanged. Passive damping is a DriveAPI with stiffness 0 and
target velocity 0 (PhysX has no separate passive joint damping); Coulomb joint friction is NOT mapped (PhysX
jointFriction is a dimensionless coefficient, not a torque). Each link carries rigeuler:visualPly and
rigeuler:T_link_ply; stock USD renderers cannot draw Graphdeco 3DGS without a Gaussian adapter.
"""
from __future__ import annotations

import json
import math
import subprocess
import tempfile
from pathlib import Path

import numpy as np

from .frames import quat_wxyz_from_R
from .model import GRAVITY_W, CanonicalAsset
from .schema import JOINT_NAME

WRITER = Path(__file__).with_name("_usd_writer.py")
NOT_REPRESENTED = {"coulomb_friction": "PhysX jointFriction is a dimensionless coefficient; not mapped",
                   "damping": "mapped as DriveAPI damping with stiffness 0, target velocity 0"}


def usd_units(a: CanonicalAsset) -> dict:
    jl = a.physics.joint
    rev = a.joint_type == "revolute"
    k = 180.0 / math.pi if rev else 1.0
    return {"lower_usd": jl.lower * k, "upper_usd": jl.upper * k,
            "damping_usd": jl.damping / k,                     # N m s/rad -> N m s/deg  (torque = d * omega)
            "max_velocity_usd": jl.velocity * k, "drive_token": "angular" if rev else "linear",
            "limit_units": "deg" if rev else "m"}


def payload(a: CanonicalAsset) -> dict:
    p = a.physics
    links = []
    T_root = {"static_link": np.eye(4), "moving_link": a.T_B_J0}
    for lk in ("static_link", "moving_link"):
        i = p.inertial[lk]
        cols = []
        for rel, part in zip(a.collision_files[lk]["parts"], p.collision[lk].parts):
            cols.append({"obj": f"../{rel}", "vertices": part.vertices.tolist(), "faces": part.faces.tolist()})
        links.append({"name": lk, "T_root_link": T_root[lk].tolist(), "mass": i.mass, "com": i.com.tolist(),
                      "diagonal_inertia": i.principal_moments.tolist(),
                      "principal_axes_wxyz": quat_wxyz_from_R(i.principal_axes).tolist(),
                      "ply": f"../{a.visual[lk]['ply']}", "T_link_ply": a.T_link_ply[lk].tolist(),
                      "collision": cols})
    u = usd_units(a)
    jl = p.joint
    joint = {"name": JOINT_NAME, "type": a.joint_type, "localPos0": a.joint_origin_xyz.tolist(),
             "localRot0_wxyz": a.joint_origin_quat_wxyz.tolist(), "localPos1": [0.0, 0.0, 0.0],
             "localRot1_wxyz": [1.0, 0.0, 0.0, 0.0], "effort": jl.effort, "armature": jl.armature, **u}
    return {"name": a.name, "T_world_root": a.T_W_B.tolist(),
            "T_world_root_quat_wxyz": quat_wxyz_from_R(a.T_W_B[:3, :3]).tolist(), "gravity": list(GRAVITY_W),
            "contact": p.contact, "links": links, "joint": joint,
            "doc": f"RigEuler3DGS sim asset {a.scene}: Graphdeco 3DGS visuals in ../visual/*.ply, rendered by a "
                   f"Gaussian adapter from live link poses with T_W_P = T_W_L @ rigeuler:T_link_ply",
            "root_metadata": {"assetYaml": {"type": "asset", "value": "../asset.yaml"},
                              "objectYaml": {"type": "asset", "value": "../object.yaml"},
                              "jointTrajectory": {"type": "asset", "value": "../joint_trajectory.csv"},
                              "requiresGaussianRenderer": {"type": "bool", "value": True},
                              "qUnits": {"type": "string", "value": a.q_units},
                              "limitUnits": {"type": "string", "value": u["limit_units"]},
                              "observedRange": {"type": "double2", "value": list(a.observed_range)},
                              "jointLimitsProvenance": {"type": "string", "value": jl.limits_provenance}}}


def run_usd(python: str, args: list, timeout=600):
    r = subprocess.run([python, str(WRITER), *args], capture_output=True, text=True, timeout=timeout)
    if r.returncode != 0:
        raise RuntimeError(f"USD writer failed ({python}): {r.stderr.strip()[-2000:]}")
    return r


def write_usd(a: CanonicalAsset, path, python: str) -> dict:
    """Write path (binary .usd) and return the reader's JSON of the written stage."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        pj, rj = Path(tmp) / "payload.json", Path(tmp) / "readback.json"
        pj.write_text(json.dumps(payload(a)))
        run_usd(python, ["write", str(pj), str(path), str(rj)])
        return json.loads(rj.read_text())


def read_usd(path, python: str) -> dict:
    with tempfile.TemporaryDirectory() as tmp:
        rj = Path(tmp) / "readback.json"
        run_usd(python, ["read", str(path), str(rj)])
        return json.loads(rj.read_text())
