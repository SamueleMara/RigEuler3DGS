"""Phase 3 ingestion and preflight for Phase 4 / Phase 5 (spec sections 0, 2.1, 2.2).

Everything here is read only. A mismatch raises ContractError: no silent conversion, no fallback to another
model, direction, camera set or joint.

  * ONE direction (forward | backward); `auto` resolves metadata/direction/summary.json:recommended_direction
  * the endpoint model is the one Phase 3 used (summary.json canonical_model), static_bidi/initial for
    forward, static_bidi/final for backward
  * the joint type is explicit (Phase 3 fits both P and R; the selection is not machine-readable upstream)
  * q comes from <type>/trajectory.csv, indexed by ORIGINAL frame; row 0 is the anchor with q = 0 exactly
  * the moving core is support_coherence/support.json:members, row indices into the endpoint PLY
"""
from __future__ import annotations

import csv
import hashlib
import json
import math
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

JOINT_TYPES = ("prismatic", "revolute")
ENDPOINT = {"forward": "initial", "backward": "final"}


class ContractError(RuntimeError):
    pass


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def ply_vertex_count(path: Path) -> int:
    with open(path, "rb") as f:
        for _ in range(64):
            line = f.readline().decode("ascii", "replace").strip()
            if line.startswith("element vertex"):
                return int(line.split()[-1])
            if line == "end_header":
                break
    raise ContractError(f"{path}: no 'element vertex' in the PLY header")


def resolve_direction(source: Path, direction: str) -> str:
    d = direction.lower()
    if d == "auto":
        p = Path(source) / "metadata" / "direction" / "summary.json"
        if not p.is_file():
            raise ContractError(f"--direction auto: {p} missing")
        d = str(json.loads(p.read_text()).get("recommended_direction", "")).lower()
    if d not in ENDPOINT:
        raise ContractError(f"direction must resolve to forward or backward, got {direction!r} -> {d!r}")
    return d


def rot_exp(axis: np.ndarray, theta: float) -> np.ndarray:
    a = np.asarray(axis, float)
    K = np.array([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]])
    return np.eye(3) + math.sin(theta) * K + (1 - math.cos(theta)) * K @ K


@dataclass(frozen=True)
class Joint:
    """The selected Phase-3 joint, immutable. q in metres (P) or radians (R), relative to the anchor."""
    type: str
    axis: tuple                       # unit direction e_P or axis e_R
    point: tuple | None               # a_R (revolute) or None
    frames: tuple                     # ORIGINAL frame ids in Phase-3 processing order
    q: tuple                          # q per processing row, q[0] == 0
    posterior_sd_q: tuple
    direction: str
    geometry_file: str
    trajectory_file: str
    geometry_sha256: str
    trajectory_sha256: str

    @property
    def anchor_frame(self) -> int:
        return int(self.frames[0])

    def q_of(self, original_frame: int) -> float:
        try:
            return float(self.q[self.frames.index(int(original_frame))])
        except ValueError:
            raise ContractError(f"original frame {original_frame} is outside the Phase-3 window "
                                f"[{min(self.frames)}, {max(self.frames)}]") from None

    def transform(self, original_frame: int):
        """(Q_f, p_f) relative to the anchor: x(f) = Q_f x0 + p_f (float64)."""
        q = self.q_of(original_frame)
        e = np.asarray(self.axis, float)
        if self.type == "prismatic":
            return np.eye(3), q * e
        Q = rot_exp(e, q)
        a = np.asarray(self.point, float)
        return Q, (np.eye(3) - Q) @ a

    def chronological(self):
        """(original frames ascending, q) — display / sampling order; inference order is `frames`."""
        o = np.argsort(self.frames)
        return np.asarray(self.frames)[o], np.asarray(self.q)[o]

    def to_dict(self) -> dict:
        return {"type": self.type, "axis": list(self.axis), "point": None if self.point is None else list(self.point),
                "direction": self.direction, "anchor_original_frame": self.anchor_frame,
                "frames_processing_order": list(self.frames), "q": list(self.q),
                "units": "m" if self.type == "prismatic" else "rad",
                "geometry_file": self.geometry_file, "geometry_sha256": self.geometry_sha256,
                "trajectory_file": self.trajectory_file, "trajectory_sha256": self.trajectory_sha256}


@dataclass
class Upstream:
    source: Path
    direction: str
    model_dir: Path
    ply: Path
    num_gaussians: int
    members: np.ndarray
    joint: Joint
    cameras: list
    phase2_dir: Path
    phase3_dir: Path
    sensor_model: Path | None
    hashes: dict = field(default_factory=dict)

    def manifest(self) -> dict:
        return {"source": str(self.source), "direction": self.direction, "model_dir": str(self.model_dir),
                "num_gaussians": self.num_gaussians, "core_members": int(len(self.members)),
                "cameras": list(self.cameras), "phase2_dir": str(self.phase2_dir), "phase3_dir": str(self.phase3_dir),
                "sensor_model": None if self.sensor_model is None else str(self.sensor_model),
                "joint": self.joint.to_dict(), "sha256": dict(self.hashes)}


def load_joint(phase3_dir: Path, joint_type: str, direction: str) -> Joint:
    if joint_type not in JOINT_TYPES:
        raise ContractError(f"--joint-type must be one of {JOINT_TYPES}, got {joint_type!r}")
    g_path = Path(phase3_dir) / joint_type / "geometry.json"
    t_path = Path(phase3_dir) / joint_type / "trajectory.csv"
    for p in (g_path, t_path):
        if not p.is_file():
            raise ContractError(f"Phase 3 output missing: {p}")
    geo = json.loads(g_path.read_text())
    if geo.get("joint_type") != joint_type or geo.get("status") != "FITTED":
        raise ContractError(f"{g_path}: joint_type={geo.get('joint_type')} status={geo.get('status')}")
    if not geo.get("geometry_unchanged_bitwise", False):
        raise ContractError(f"{g_path}: Phase 3 did not certify its geometry as unchanged")
    gf = geo.get("geometry_fixed") or {}
    if joint_type == "prismatic":
        axis, point = np.asarray(gf.get("direction"), float), None
    else:
        axis, point = np.asarray(gf.get("axis"), float), gf.get("axis_point")
        if point is None or not np.isfinite(point).all():
            raise ContractError(f"{g_path}: revolute geometry has no finite axis_point")
        point = tuple(float(x) for x in point)
    if axis.shape != (3,) or not np.isfinite(axis).all() or abs(np.linalg.norm(axis) - 1.0) > 1e-6:
        raise ContractError(f"{g_path}: axis {axis} is not a finite unit vector (|e| = {np.linalg.norm(axis)})")
    rows = list(csv.DictReader(open(t_path)))
    if not rows:
        raise ContractError(f"{t_path}: empty")
    pidx = [int(r["processing_index"]) for r in rows]
    frames = [int(r["original_frame"]) for r in rows]
    q = [float(r["q"]) for r in rows]
    sd = [float(r.get("posterior_sd_q", "nan")) for r in rows]
    if pidx != list(range(len(rows))):
        raise ContractError(f"{t_path}: processing_index is not 0..T-1")
    if q[0] != 0.0:
        raise ContractError(f"{t_path}: q at the anchor row is {q[0]}, must be exactly 0")
    if not np.isfinite(q).all():
        raise ContractError(f"{t_path}: non-finite q")
    step = np.diff(frames)
    want = 1 if direction == "forward" else -1
    if len(frames) > 1 and not np.all(step == want):
        raise ContractError(f"{t_path}: original frames are not consecutive in {direction} processing order")
    return Joint(type=joint_type, axis=tuple(float(x) for x in axis / np.linalg.norm(axis)), point=point,
                 frames=tuple(frames), q=tuple(q), posterior_sd_q=tuple(sd), direction=direction,
                 geometry_file=str(g_path), trajectory_file=str(t_path),
                 geometry_sha256=sha256(g_path), trajectory_sha256=sha256(t_path))


def load_upstream(source, model, phase2_dir, phase3_dir, joint_type, direction="auto", cameras=None,
                  frame_window=None, sensor_model=None) -> Upstream:
    """Preflight (spec 2.1). `cameras`: None | 'all' | 'cam_a,cam_b' (must equal Phase 3's set)."""
    source, model = Path(source), Path(model)
    phase2_dir, phase3_dir = Path(phase2_dir), Path(phase3_dir)
    d = resolve_direction(source, direction)
    if phase3_dir.name != f"{d}_joint" or phase2_dir.name != f"{d}_rigid":
        raise ContractError(f"direction {d}: expected <..>/{d}_rigid and <..>/{d}_joint, got "
                            f"{phase2_dir.name} and {phase3_dir.name}")
    if model.name != ENDPOINT[d]:
        raise ContractError(f"direction {d} needs the static_bidi/{ENDPOINT[d]} endpoint, got {model}")
    summ_p = phase3_dir / "summary.json"
    if not summ_p.is_file():
        raise ContractError(f"{summ_p} missing")
    summ = json.loads(summ_p.read_text())
    if str(summ.get("direction", "")).lower() != d:
        raise ContractError(f"{summ_p}: direction {summ.get('direction')} != {d}")
    if Path(summ.get("canonical_model", "")).resolve() != model.resolve():
        raise ContractError(f"Phase 3 used {summ.get('canonical_model')}, not {model}")
    joint = load_joint(phase3_dir, joint_type, d)
    if joint_type not in summ.get("joint_types_run", []):
        raise ContractError(f"{summ_p}: {joint_type} not in joint_types_run {summ.get('joint_types_run')}")
    if list(summ.get("original_frames_processing_order", [])) != list(joint.frames):
        raise ContractError("trajectory frames differ from summary.json original_frames_processing_order")
    if int(summ.get("canonical_original_frame", -1)) != joint.anchor_frame:
        raise ContractError("anchor frame differs from summary.json canonical_original_frame")
    if frame_window is not None:
        lo, hi = (int(x) for x in str(frame_window).split(":"))
        if (lo, hi) != (min(joint.frames), max(joint.frames)):
            raise ContractError(f"--frame-window {lo}:{hi} != Phase-3 window {min(joint.frames)}:{max(joint.frames)}")
    sig = json.loads((phase3_dir / "stage_manifest.json").read_text()).get("signature", {})
    p3_cams = list(sig.get("source", {}).get("cameras", []))
    if not p3_cams:
        raise ContractError(f"{phase3_dir}/stage_manifest.json records no camera set")
    if cameras not in (None, "all") and [c.strip() for c in str(cameras).split(",")] != p3_cams:
        raise ContractError(f"--cameras {cameras} != Phase 3's {p3_cams}")
    ply = model / "point_cloud.ply"
    if not ply.is_file():
        cands = sorted(model.glob("point_cloud/iteration_*/point_cloud.ply"))
        if len(cands) != 1:
            raise ContractError(f"{model}: no unique point_cloud.ply")
        ply = cands[0]
    n = ply_vertex_count(ply)
    sup_p = phase2_dir / "support_coherence" / "support.json"
    sup = json.loads(sup_p.read_text())
    src_dir = Path(sup.get("source", "")).parent.name
    if src_dir and src_dir != d:
        raise ContractError(f"{sup_p}: the support came from '{src_dir}', not {d}")
    members = np.asarray(sup["members"], dtype=np.int64)
    if members.ndim != 1 or len(members) == 0 or members.min() < 0 or members.max() >= n:
        raise ContractError(f"{sup_p}: members out of range for {n} Gaussians")
    if len(np.unique(members)) != len(members):
        raise ContractError(f"{sup_p}: duplicated members")
    sens = None if sensor_model is None else Path(sensor_model)
    if sens is not None and not sens.is_file():
        raise ContractError(f"sensor model {sens} missing")
    hashes = {"ply": sha256(ply), "support": sha256(sup_p), "phase3_summary": sha256(summ_p),
              "geometry": joint.geometry_sha256, "trajectory": joint.trajectory_sha256}
    if sens is not None:
        hashes["sensor_model"] = sha256(sens)
    return Upstream(source=source, direction=d, model_dir=model, ply=ply, num_gaussians=n, members=np.sort(members),
                    joint=joint, cameras=p3_cams, phase2_dir=phase2_dir, phase3_dir=phase3_dir, sensor_model=sens,
                    hashes=hashes)
