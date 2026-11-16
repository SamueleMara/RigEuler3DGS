"""Articulated PLY: the 3DGS vertex fields (unchanged, first element) + part / lineage fields + the frozen joint.

vertex  (one row per Gaussian, 3DGS fields first and byte-identical; loaders that read 3DGS fields by name ignore the rest)
    part_id        uint8   0 = base (static, parent link), 1 = moving part (child link)
    gaussian_uid   int32   stable id across Phases 4/5
    original_id    int32   row in the Phase-0 model, -1 for Gaussians born later
    parent_uid     int32   uid this Gaussian was cloned / split from, -1 if none
    provenance     uint8   0 original_core, 1 original, 2 phase4_patch, 3 densify_clone, 4 densify_split
    unresolved     uint8   1 = 4A left it undecided (part_id is its frozen render label), or descends from such a row
joint  (one row)
    type           uint8   0 = prismatic, 1 = revolute
    parent_part, child_part  uint8   0, 1
    axis_x/y/z     float64 unit axis (revolute) / translation direction (prismatic)
    origin_x/y/z   float64 a point on the revolute axis; prismatic: centroid of the moving part (origin_defined = 0)
    origin_defined uint8
    q_min, q_max   float64 motion range over the sequence (rad for revolute, m for prismatic)
    q_canonical    float64 q of the stored geometry (0: the canonical model is the anchor frame)
    anchor_frame   int32   original frame id at which the stored geometry holds
joint_state  (one row per frame, chronological)
    frame int32, q float64, posterior_sd_q float64
Pose of the moving part at frame f:  x(f) = R(q_f) (x - o) + o  (revolute about axis through o),  x(f) = x + q_f * axis
(prismatic); base Gaussians never move. The same content is repeated as human-readable header comments.
"""
import json
import os
from pathlib import Path

import numpy as np
from plyfile import PlyData, PlyElement

EXTRA = (("part_id", "u1"), ("gaussian_uid", "i4"), ("original_id", "i4"), ("parent_uid", "i4"), ("provenance", "u1"),
         ("unresolved", "u1"))


def _i4(a, name):
    a = np.asarray(a, np.int64)
    if a.size and (a.max() > np.iinfo(np.int32).max or a.min() < np.iinfo(np.int32).min):
        raise ValueError(f"{name} does not fit int32")
    return a.astype(np.int32)


def descendants_of(uid, parent_uid, seed_uids) -> np.ndarray:
    """Boolean mask: uid in seed_uids, or reachable from one through parent_uid links."""
    flag = np.isin(uid, np.asarray(seed_uids, np.int64))
    while True:
        new = flag | np.isin(parent_uid, uid[flag])
        if (new == flag).all():
            return flag
        flag = new


def write_articulated_ply(src_ply, lineage: dict, joint, out_ply, unresolved_uids=(), extra_comments=()) -> dict:
    """Append part / lineage vertex fields and the joint elements to the plain 3DGS PLY `src_ply`; write `out_ply`
    atomically (src == out is allowed). Refuses if the source already carries the extra fields."""
    src = PlyData.read(str(src_ply))
    v = src["vertex"].data
    n = len(v)
    names = v.dtype.names
    if any(k in names for k, _ in EXTRA):
        raise ValueError(f"{src_ply} already has articulated fields")
    for k in ("material", "gaussian_uid", "original_id", "parent_uid", "provenance"):
        if len(lineage[k]) != n:
            raise ValueError(f"lineage[{k!r}] has {len(lineage[k])} rows, the PLY {n}")
    uid = np.asarray(lineage["gaussian_uid"], np.int64)
    par = np.asarray(lineage["parent_uid"], np.int64)
    part = np.asarray(lineage["material"]).astype(np.uint8)
    unres = descendants_of(uid, par, unresolved_uids)
    vd = np.empty(n, dtype=list(v.dtype.descr) + list(EXTRA))
    for k in names:
        vd[k] = v[k]
    vd["part_id"] = part
    vd["gaussian_uid"] = _i4(uid, "gaussian_uid")
    vd["original_id"] = _i4(lineage["original_id"], "original_id")
    vd["parent_uid"] = _i4(par, "parent_uid")
    vd["provenance"] = np.asarray(lineage["provenance"]).astype(np.uint8)
    vd["unresolved"] = unres.astype(np.uint8)

    rev = joint.type == "revolute"
    axis = np.asarray(joint.axis, float)
    if rev:
        origin, defined = np.asarray(joint.point, float), 1
    else:
        xyz = np.stack([v["x"], v["y"], v["z"]], 1).astype(float)
        origin, defined = (xyz[part == 1].mean(0) if (part == 1).any() else np.zeros(3)), 0
    fr, q = joint.chronological()
    sd_map = dict(zip(joint.frames, joint.posterior_sd_q))
    sd = np.array([sd_map[int(f)] for f in fr], float)
    q = np.asarray(q, float)
    jd = np.empty(1, dtype=[("type", "u1"), ("parent_part", "u1"), ("child_part", "u1"),
                            ("axis_x", "f8"), ("axis_y", "f8"), ("axis_z", "f8"),
                            ("origin_x", "f8"), ("origin_y", "f8"), ("origin_z", "f8"), ("origin_defined", "u1"),
                            ("q_min", "f8"), ("q_max", "f8"), ("q_canonical", "f8"), ("anchor_frame", "i4")])
    jd[0] = (int(rev), 0, 1, *axis, *origin, defined, q.min(), q.max(), 0.0, joint.anchor_frame)
    st = np.empty(len(fr), dtype=[("frame", "i4"), ("q", "f8"), ("posterior_sd_q", "f8")])
    st["frame"], st["q"], st["posterior_sd_q"] = np.asarray(fr, np.int32), q, sd

    unit = "rad" if rev else "m"
    info = {"joint_type": joint.type, "parent_part": 0, "child_part": 1, "axis": axis.tolist(),
            "origin": origin.tolist(), "origin_defined": bool(defined), "q_range": [float(q.min()), float(q.max())],
            "q_units": unit, "anchor_frame": joint.anchor_frame, "direction": joint.direction,
            "parts": {"0": "base (static)", "1": "moving"}, "n_gaussians": n, "n_moving": int(part.sum()),
            "n_unresolved": int(unres.sum()), "geometry_sha256": joint.geometry_sha256,
            "trajectory_sha256": joint.trajectory_sha256}
    comments = [f"articulated_3dgs part_id 0=base(static) 1=moving; joint {joint.type}",
                f"joint_type {joint.type}", f"joint_axis {axis[0]:.9g} {axis[1]:.9g} {axis[2]:.9g}",
                f"joint_origin {origin[0]:.9g} {origin[1]:.9g} {origin[2]:.9g}"
                + ("" if defined else " (prismatic: moving-part centroid, not a joint parameter)"),
                f"joint_range {q.min():.9g} {q.max():.9g} {unit}", f"anchor_frame {joint.anchor_frame} (q = 0 here)",
                "articulated_json " + json.dumps(info, separators=(",", ":"))] + list(extra_comments)
    out = PlyData([PlyElement.describe(vd, "vertex"), PlyElement.describe(jd, "joint"),
                   PlyElement.describe(st, "joint_state")], comments=comments)
    out_ply = Path(out_ply)
    out_ply.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_ply.with_name(out_ply.name + ".tmp")
    out.write(str(tmp))
    os.replace(tmp, out_ply)
    return info


def read_articulated_ply(path) -> dict:
    p = PlyData.read(str(path))
    v = p["vertex"].data
    j = p["joint"].data[0]
    s = p["joint_state"].data
    return {"part_id": np.asarray(v["part_id"]), "gaussian_uid": np.asarray(v["gaussian_uid"]),
            "unresolved": np.asarray(v["unresolved"]),
            "joint_type": "revolute" if int(j["type"]) else "prismatic",
            "axis": np.array([j["axis_x"], j["axis_y"], j["axis_z"]]),
            "origin": np.array([j["origin_x"], j["origin_y"], j["origin_z"]]),
            "q_range": (float(j["q_min"]), float(j["q_max"])), "anchor_frame": int(j["anchor_frame"]),
            "frames": np.asarray(s["frame"]), "q": np.asarray(s["q"]), "comments": list(p.comments)}
