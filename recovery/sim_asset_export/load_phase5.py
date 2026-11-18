"""Phase-5 run preflight for the asset exporter (spec 1.1-1.3). Read only.

Post-hoc: <P5>/complete.json must exist (stage phase5, joint_unchanged). In the training driver the completion payload
that is ABOUT to be written is passed instead (complete.json is written only after the export).

The joint is rebuilt from the byte copies in <P5>/final/joint/ with recovery.contracts.load_joint (the same parser
Phase 4/5 used) and must match, exactly:
  * the file hashes and the per-frame transform hash of the completion's joint_fingerprint,
  * joint_unchanged_proof.json (identical, not optimized),
  * the joint / joint_state elements of final/point_cloud.ply (type, axis, hinge point, anchor, chronological q, sd).
"""
from __future__ import annotations

import json
import os
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from plyfile import PlyData

from ..contracts import ContractError, load_joint
from .ply_split import sha256_file


class Phase5InputError(ValueError):
    pass


@dataclass
class Phase5Inputs:
    phase5_dir: Path
    scene: str
    direction: str
    completion: dict
    completion_source: str          # "complete.json" | "driver"
    complete_sha256: str | None
    setup: dict
    joint: object                   # recovery.contracts.Joint
    ply_path: Path
    ply_sha256: str
    part_id: np.ndarray
    xyz: np.ndarray                 # float64 canonical means (Wc)
    lineage: dict                   # per PLY row: gaussian_uid, original_id, parent_uid, provenance, unresolved
    ply_joint: dict                 # the PLY joint element as stored
    trajectory: dict                # chronological frame / q / posterior_sd_q from the PLY joint_state
    appearance: dict | None
    unresolved_originals: dict | None
    not_for_results: bool
    warnings: list = field(default_factory=list)


def _json(p: Path) -> dict:
    try:
        return json.loads(p.read_text())
    except FileNotFoundError:
        raise Phase5InputError(f"{p} missing") from None
    except json.JSONDecodeError as e:
        raise Phase5InputError(f"{p}: invalid JSON ({e})") from None


def _setup_from_manifest(p5: Path) -> dict:
    m = _json(p5 / "manifest.json")
    for path in (("upstream", "setup"), ("setup",)):
        d = m
        for k in path:
            d = d.get(k) if isinstance(d, dict) else None
        if isinstance(d, dict):
            return d
    raise Phase5InputError(f"{p5}/manifest.json has no upstream.setup")


def _joint_from_copies(p5: Path, joint_type: str, direction: str):
    jd = p5 / "final" / "joint"
    for n in ("geometry.json", "trajectory.csv"):
        if not (jd / n).is_file():
            raise Phase5InputError(f"{jd / n} missing (Phase 5 copies the frozen Phase-3 joint there)")
    with tempfile.TemporaryDirectory() as tmp:                 # load_joint reads <dir>/<joint_type>/{geometry,trajectory}
        os.symlink(jd.resolve(), Path(tmp) / joint_type)
        try:
            j = load_joint(Path(tmp), joint_type, direction)
        except ContractError as e:
            raise Phase5InputError(f"frozen joint copy invalid: {e}") from None
    from dataclasses import replace
    return replace(j, geometry_file=str(jd / "geometry.json"), trajectory_file=str(jd / "trajectory.csv"))


def load_phase5(phase5_dir, completion: dict | None = None, setup: dict | None = None, scene: str | None = None,
                ply_path=None) -> Phase5Inputs:
    from ..phase4_artifacts import joint_fingerprint
    p5 = Path(phase5_dir)
    warn = []
    if completion is None:
        cp = p5 / "complete.json"
        if not cp.is_file():
            why = " (incomplete Phase-5 run)" if p5.is_dir() else ""
            raise Phase5InputError(f"{cp} missing{why}; the exporter only reads completed Phase-5 runs")
        completion, src, csha = _json(cp), "complete.json", sha256_file(cp)
    else:
        src, csha = "driver", None
    if completion.get("stage") != "phase5":
        raise Phase5InputError(f"completion stage is {completion.get('stage')!r}, not 'phase5'")
    if completion.get("joint_unchanged") is not True:
        raise Phase5InputError("completion does not certify joint_unchanged = true")
    nfr = bool(completion.get("NOT_FOR_RESULTS"))
    if nfr:
        warn.append("Phase-5 run is NOT_FOR_RESULTS (smoke / debug run); the asset inherits that flag")
    setup = dict(setup) if setup is not None else _setup_from_manifest(p5)
    for k in ("source", "joint_type", "direction"):
        if not setup.get(k):
            raise Phase5InputError(f"Phase-5 setup has no {k!r}")
    jt, direction = setup["joint_type"], setup["direction"]
    sc = scene or Path(str(setup["source"])).name
    if not sc or "/" in sc or sc in (".", ".."):
        raise Phase5InputError(f"invalid scene name {sc!r}")

    # ---- frozen joint, three independent records
    joint = _joint_from_copies(p5, jt, direction)
    fp = completion.get("joint_fingerprint") or {}
    have = joint_fingerprint(joint)
    if fp.get("files") is None or have["files"] != fp["files"]:
        raise Phase5InputError(f"frozen joint copy hashes {have['files']} differ from the completion fingerprint "
                               f"{fp.get('files')} (corrupted or replaced joint files)")
    if have["transforms_sha256"] != fp.get("transforms_sha256") or have["frames"] != fp.get("frames"):
        raise Phase5InputError("per-frame joint transforms differ from the completion fingerprint")
    proof = _json(p5 / "final" / "joint_unchanged_proof.json")
    if not proof.get("identical") or proof.get("joint_optimized"):
        raise Phase5InputError("joint_unchanged_proof.json does not certify an identical, non-optimized joint")

    ply_path = Path(ply_path) if ply_path is not None else p5 / "final" / "point_cloud.ply"
    if not ply_path.is_file():
        raise Phase5InputError(f"{ply_path} missing")
    psha = sha256_file(ply_path)
    ply = PlyData.read(str(ply_path))
    els = [e.name for e in ply.elements]
    for need in ("vertex", "joint", "joint_state"):
        if need not in els:
            raise Phase5InputError(f"{ply_path}: no {need!r} element -- not the Phase-5 articulated PLY")
    v = ply["vertex"].data
    for k in ("part_id", "gaussian_uid", "original_id", "parent_uid", "provenance", "unresolved"):
        if k not in v.dtype.names:
            raise Phase5InputError(f"{ply_path}: vertex has no {k!r}")
    part = np.asarray(v["part_id"])
    if not np.isin(part, [0, 1]).all():
        raise Phase5InputError(f"part_id values {np.unique(part).tolist()} outside {{0, 1}}")
    for pid, nm in ((0, "static"), (1, "moving")):
        if not (part == pid).any():
            raise Phase5InputError(f"zero {nm} Gaussians (part_id {pid})")
    xyz = np.stack([np.asarray(v[k], np.float64) for k in ("x", "y", "z")], 1)
    if not np.isfinite(xyz).all():
        raise Phase5InputError("non-finite Gaussian means")
    uid = np.asarray(v["gaussian_uid"], np.int64)
    if len(np.unique(uid)) != len(uid):
        raise Phase5InputError("duplicate gaussian_uid in the articulated PLY")
    lin = {k: np.asarray(v[k]).copy() for k in ("gaussian_uid", "original_id", "parent_uid", "provenance", "unresolved")}

    j = ply["joint"].data
    if len(j) != 1:
        raise Phase5InputError(f"{len(j)} joint rows; expected 1")
    j = j[0]
    pj = {k: (float(j[k]) if j.dtype[k].kind == "f" else int(j[k])) for k in j.dtype.names}
    rev = jt == "revolute"
    if pj["type"] != int(rev) or pj["parent_part"] != 0 or pj["child_part"] != 1:
        raise Phase5InputError(f"PLY joint element {pj} disagrees with joint type {jt} / parts 0 -> 1")
    if not np.array_equal(np.array([pj["axis_x"], pj["axis_y"], pj["axis_z"]]), np.asarray(joint.axis, float)):
        raise Phase5InputError("PLY joint axis differs from the frozen Phase-3 axis")
    org = np.array([pj["origin_x"], pj["origin_y"], pj["origin_z"]])
    if rev and (pj["origin_defined"] != 1 or not np.array_equal(org, np.asarray(joint.point, float))):
        raise Phase5InputError("PLY revolute origin differs from the frozen Phase-3 hinge point")
    if pj["q_canonical"] != 0.0:
        raise Phase5InputError(f"PLY q_canonical = {pj['q_canonical']}; the stored geometry must be the anchor (q = 0)")
    if pj["anchor_frame"] != joint.anchor_frame:
        raise Phase5InputError(f"PLY anchor frame {pj['anchor_frame']} != Phase-3 anchor {joint.anchor_frame}")
    s = ply["joint_state"].data
    fr, q = joint.chronological()
    sdm = dict(zip(joint.frames, joint.posterior_sd_q))
    sd = np.array([sdm[int(f)] for f in fr], float)
    if not (np.array_equal(np.asarray(s["frame"], np.int64), np.asarray(fr, np.int64)) and
            np.array_equal(np.asarray(s["q"], float), np.asarray(q, float)) and
            np.array_equal(np.asarray(s["posterior_sd_q"], float), sd, equal_nan=True)):
        raise Phase5InputError("PLY joint_state (frame, q, posterior_sd_q) differs from the frozen Phase-3 trajectory")
    if pj["q_min"] != float(np.min(q)) or pj["q_max"] != float(np.max(q)):
        raise Phase5InputError("PLY q_min / q_max differ from the trajectory range")

    app = p5 / "final" / "appearance_settings.json"
    uo = p5 / "unresolved_originals.npz"
    unres = None
    if uo.is_file():
        z = np.load(uo)
        unres = {k: z[k] for k in z.files}
    n_total = completion.get("final_gaussians")
    if n_total is not None and int(n_total) != len(v):
        raise Phase5InputError(f"articulated PLY has {len(v)} rows, the completion says {n_total}")
    return Phase5Inputs(phase5_dir=p5, scene=sc, direction=direction, completion=completion, completion_source=src,
                        complete_sha256=csha, setup=setup, joint=joint, ply_path=ply_path, ply_sha256=psha,
                        part_id=part.astype(np.uint8), xyz=xyz, lineage=lin, ply_joint=pj,
                        trajectory={"frame": np.asarray(fr, np.int64), "q": np.asarray(q, float), "posterior_sd_q": sd},
                        appearance=_json(app) if app.is_file() else None, unresolved_originals=unres,
                        not_for_results=nfr, warnings=warn)
