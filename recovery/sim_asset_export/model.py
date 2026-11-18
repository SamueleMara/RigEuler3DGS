"""The single canonical asset object (spec 5, 6D). URDF, USD and MJCF are generated from THIS object only; asset.yaml,
object.yaml and physics/dynamics.yaml are its serializations. No exporter reads another exporter's output.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

import numpy as np

from .frames import inv_se3, joint_frame, quat_wxyz_from_R, urdf_rpy_from_R
from .schema import JOINT_NAME, LINK_NAMES, SCHEMA_VERSION, Registration, UNVERIFIED_GAUGE

GRAVITY_W = (0.0, 0.0, -9.81)        # simulator world convention: z up, gravity along -Z of W (see registration)
FORMAT_VERSION = "rigeuler3dgs-sim-asset/1"


def model_name(scene: str) -> str:
    s = re.sub(r"[^A-Za-z0-9_]", "_", scene)
    return s if s and not s[0].isdigit() else f"asset_{s}"


@dataclass
class CanonicalAsset:
    scene: str
    name: str
    direction: str
    joint_type: str
    axis: np.ndarray                 # a, Wc
    origin: np.ndarray               # o, Wc
    origin_kind: str
    T_B_J0: np.ndarray
    registration: Registration
    anchor_frame: int
    trajectory: dict                 # chronological frame / q / posterior_sd_q
    q_units: str
    observed_range: tuple
    visual: dict                     # link -> {"ply": rel path, "n": int, "sha256": str}
    sh_degree: int
    appearance: dict | None
    physics: object | None           # physics.ResolvedPhysics
    collision_files: dict = field(default_factory=dict)   # link -> {"surface": rel, "parts": [rel...]} (asset-relative)
    not_for_results: bool = False
    source: dict = field(default_factory=dict)

    @property
    def T_W_B(self) -> np.ndarray:
        return self.registration.T_world_capture

    @property
    def T_link_ply(self) -> dict:
        return {"static_link": np.eye(4), "moving_link": inv_se3(self.T_B_J0)}

    @property
    def joint_origin_xyz(self):
        return self.T_B_J0[:3, 3]

    @property
    def joint_origin_rpy(self):
        return urdf_rpy_from_R(self.T_B_J0[:3, :3])

    @property
    def joint_origin_quat_wxyz(self):
        return quat_wxyz_from_R(self.T_B_J0[:3, :3])


def build_asset(inp, split, physics, registration: Registration | None) -> CanonicalAsset:
    j = inp.joint
    if j.type == "revolute":
        T, o, kind = joint_frame("revolute", j.axis, j.point)
    else:
        # gauge = the moving centroid stored in the articulated PLY's joint element (same float64 mean of the stored
        # float32 means); recomputed and compared so the gauge is not an unverified number
        T, o, kind = joint_frame("prismatic", j.axis, moving_xyz=inp.xyz[inp.part_id == 1])
        stored = np.array([inp.ply_joint["origin_x"], inp.ply_joint["origin_y"], inp.ply_joint["origin_z"]])
        if np.abs(stored - o).max() > 1e-9:
            raise ValueError(f"prismatic gauge {o} differs from the articulated PLY origin {stored}")
    reg = registration or (physics.registration if physics is not None else None) or UNVERIFIED_GAUGE
    q = inp.trajectory["q"]
    return CanonicalAsset(
        scene=inp.scene, name=model_name(inp.scene), direction=inp.direction, joint_type=j.type,
        axis=np.asarray(j.axis, float), origin=o, origin_kind=kind, T_B_J0=T, registration=reg,
        anchor_frame=j.anchor_frame, trajectory=inp.trajectory, q_units="rad" if j.type == "revolute" else "m",
        observed_range=(float(q.min()), float(q.max())),
        visual={lk: {"ply": f"visual/{lk.split('_')[0]}.ply", "n": split.counts[lk], "sha256": split.sha256[lk]}
                for lk in LINK_NAMES},
        sh_degree=split.sh_degree, appearance=inp.appearance, physics=physics, not_for_results=inp.not_for_results)


def _m(T):
    return [[float(x) for x in r] for r in np.asarray(T)]


def asset_dict(a: CanonicalAsset, status: str, readiness: dict, hashes: dict, missing: list) -> dict:
    """asset.yaml content (portable: every path is relative to the asset directory)."""
    p = a.physics
    lim = None
    if p is not None:
        lim = {"lower": p.joint.lower, "upper": p.joint.upper, "velocity": p.joint.velocity,
               "effort": p.joint.effort, "provenance": p.joint.limits_provenance}
    return {
        "schema_version": SCHEMA_VERSION, "format": FORMAT_VERSION, "asset_name": a.name, "scene": a.scene,
        "direction": a.direction, "status": status, "readiness": readiness, "missing_inputs": missing,
        "NOT_FOR_RESULTS": a.not_for_results,
        "frames": {
            "convention": "column vectors; T_A_B maps coordinates in frame B to frame A",
            "Wc": "Phase-5 canonical capture world = coordinates stored in both visual PLYs (anchor q = 0)",
            "B": "static_link frame = Wc", "J0": "moving_link frame at q = 0; local +Z = joint axis",
            "W": "simulator world", "world_gauge": a.registration.gauge,
            "gravity_alignment": a.registration.gravity_alignment, "registration_provenance":
                a.registration.provenance, "gravity_world": list(GRAVITY_W),
            "T_world_root": _m(a.T_W_B)},
        "links": {lk: {"visual_ply": a.visual[lk]["ply"], "T_link_ply": _m(a.T_link_ply[lk]),
                       "n_gaussians": a.visual[lk]["n"], "ply_sha256": a.visual[lk]["sha256"],
                       "collision": a.collision_files.get(lk)} for lk in LINK_NAMES},
        "visual": {"format": "Graphdeco 3DGS binary PLY (x y z [nx ny nz] f_dc f_rest opacity scale rot), original "
                             "stored values", "sh_degree": a.sh_degree, "quaternion": "rot_0..3 = (w, x, y, z), "
                             "Hamilton, unnormalized as stored", "sh_frame": "evaluate SH in the PLY frame P",
                   "appearance_settings": a.appearance,
                   "render_rule": "T_W_P = T_W_L(live physics pose) @ T_link_ply; never re-animate with q"},
        "joint": {"name": JOINT_NAME, "type": a.joint_type, "parent": "static_link", "child": "moving_link",
                  "axis_capture": a.axis.tolist(), "origin_capture": a.origin.tolist(), "origin_kind": a.origin_kind,
                  "T_parent_joint": _m(a.T_B_J0), "axis_local": [0.0, 0.0, 1.0], "q_units": a.q_units,
                  "q_canonical": 0.0, "anchor_original_frame": a.anchor_frame,
                  "observed_demonstration_range": list(a.observed_range),
                  "observed_range_is_not_a_mechanical_limit": True,
                  "physical_limits": lim, "trajectory": "joint_trajectory.csv"},
        "exports": {"urdf": "maniskill/model.urdf", "usd": "isaac/model.usd", "mjcf": "mujoco/model.xml",
                    "object_yaml": "object.yaml", "dynamics": "physics/dynamics.yaml" if p is not None else None,
                    "lineage": "lineage.npz"},
        "source_phase5": a.source, "hashes": hashes,
    }


def dynamics_dict(a: CanonicalAsset) -> dict:
    p = a.physics
    links = {}
    for lk, i in p.inertial.items():
        links[lk] = {"mass_kg": i.mass, "com_link_m": i.com.tolist(), "inertia_com_link_kg_m2": _m(i.inertia),
                     "principal_moments_kg_m2": i.principal_moments.tolist(),
                     "principal_axes_link": _m(i.principal_axes), "provenance": i.provenance, "audit": i.report}
    col = {lk: {"provenance": c.provenance, "source_kind": c.source_kind, "approximation": c.approximation,
                "files": a.collision_files.get(lk), "audit": c.report} for lk, c in p.collision.items()}
    j = p.joint
    return {"schema_version": SCHEMA_VERSION, "units": {"length": "m", "mass": "kg", "angle": a.q_units,
                                                        "time": "s"},
            "registration": a.registration.to_dict(), "links": links, "collision": col,
            "joint": {"name": JOINT_NAME, "type": a.joint_type, "lower": j.lower, "upper": j.upper,
                      "limits_provenance": j.limits_provenance, "velocity_limit": j.velocity, "effort_limit": j.effort,
                      "damping": j.damping, "coulomb_friction": j.coulomb_friction, "armature": j.armature,
                      "actuation": None, "provenance": j.provenance, "audit": j.report},
            "contact": p.contact or {"status": "NOT_SUPPLIED: each engine uses its own default contact material"},
            "assumptions": p.assumptions, "provenance_summary": p.provenance_summary, "warnings": p.warnings,
            "source_spec_file": p.spec_path.rsplit("/", 1)[-1]}


def object_yaml_text(a: CanonicalAsset, status: str) -> str:
    """PArtTwin1D schema_version-1 consumer config (keys exactly as its loader accepts)."""
    import yaml
    d = {"schema_version": 1, "object_name": a.name, "urdf": "maniskill/model.urdf", "fixed_base": True,
         "T_world_root": _m(a.T_W_B),
         "links": {lk: {"ply": a.visual[lk]["ply"], "T_link_ply": _m(a.T_link_ply[lk])} for lk in LINK_NAMES},
         "initial_q": {JOINT_NAME: 0.0},
         "notes": f"GENERATED by RigEuler3DGS tools/export_phase5_sim_asset.py; status {status}; "
                  f"world gauge {a.registration.gauge}, gravity alignment {a.registration.gravity_alignment}"}
    if a.appearance:
        r = {}
        bg = a.appearance.get("background")
        if isinstance(bg, (list, tuple)) and len(bg) == 3:
            r["background_rgb"] = [float(x) for x in bg]
        deg = a.appearance.get("active_sh_degree")
        if isinstance(deg, int) and 0 <= deg <= 3:
            r["sh_degree"] = deg
        if r:
            d["render"] = r
    head = ["# GENERATED -- do not edit; re-run tools/export_phase5_sim_asset.py",
            f"# status: {status}"]
    if status not in ("FULL_PHYSICS_READY", "PHYSICS_PREVIEW_ASSUMPTIONS", "PHYSICS_INCOMPLETE_RUNTIME"):
        head.append("# maniskill/model.urdf was NOT generated (no validated physics): this config parses but cannot be "
                    "loaded into a simulator until the asset is re-exported with --asset-physics-spec")
    return "\n".join(head) + "\n" + yaml.safe_dump(d, sort_keys=False, default_flow_style=None, width=1000)
