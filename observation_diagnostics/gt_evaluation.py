"""EVALUATION ONLY: dataset ground truth scored against FROZEN endpoint-diagnosis
predictions.

This module is imported only by tools/evaluate_fb_endpoint_correspondence_gt.py,
a separate process started after a diagnosis run is sealed
(run_state.json complete: true). Nothing in the diagnosis (tools/
analyze_fb_endpoint_correspondence.py, observation_diagnostics/endpoint_*,
joint_evidence, pair/pixel/gaussian evidence) imports it, and the diagnosis
process keeps its GT firewall. Its results are written only under
<diagnosis>/evaluation_gt/ and never feed back into any selection, ranking,
threshold or label.

Ground truth used, when present (dataset layout verified per scene):
  metadata/joint_info.json        selected joint: name, type, dof, limits
  pointcloud/joint_metadata.json  joint axis_vector, point_of_application, coordinate_frame
  metadata/frame_values.csv       joint coordinate q(t) per original frame (rad | m)
  pointcloud/points3D_parts.ply   frame-0 surface samples with part labels
  pointcloud/points3D_parts_final.ply  frame-T surface samples with part labels
  mask/<cam>/frame_%06d.png       binary OBJECT foreground mask (not a part mask)
  metadata/cameras.json           camera positions (gauge check against COLMAP)
"""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import numpy as np

TOLS_M = (0.01, 0.02, 0.05)
PRIMARY_TOL = 0.02


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for b in iter(lambda: fh.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


# ---------------------------------------------------------------------------
# geometry
# ---------------------------------------------------------------------------

def unit(v):
    v = np.asarray(v, float)
    n = np.linalg.norm(v)
    return v / n if n > 0 else v


def line_angle_deg(a, b) -> float:
    """Unoriented line angle: acos(clamp(|a.b|, 0, 1)), degrees."""
    return float(np.degrees(np.arccos(np.clip(abs(unit(a) @ unit(b)), 0.0, 1.0))))


def line_distance(p1, a1, p2, a2, parallel_tol=1e-9) -> float:
    """Minimum distance between two infinite lines (robust when parallel)."""
    a1, a2 = unit(a1), unit(a2)
    d = np.asarray(p2, float) - np.asarray(p1, float)
    c = np.cross(a1, a2)
    nc = np.linalg.norm(c)
    if nc < parallel_tol:
        return float(np.linalg.norm(d - (d @ a1) * a1))
    return float(abs(d @ c) / nc)


def rodrigues(axis, theta):
    a = unit(axis)
    K = np.array([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]])
    return np.eye(3) + np.sin(theta) * K + (1 - np.cos(theta)) * K @ K


def revolute_T(axis, point, theta):
    R = rodrigues(axis, theta)
    T = np.eye(4)
    T[:3, :3] = R
    p = np.asarray(point, float)
    T[:3, 3] = p - R @ p
    return T


def prismatic_T(axis, s):
    T = np.eye(4)
    T[:3, 3] = unit(axis) * s
    return T


def apply(T, X):
    return np.asarray(X) @ T[:3, :3].T + T[:3, 3]


def geodesic_deg(R1, R2) -> float:
    c = (np.trace(R1.T @ R2) - 1) / 2
    return float(np.degrees(np.arccos(np.clip(c, -1, 1))))


# ---------------------------------------------------------------------------
# ground truth
# ---------------------------------------------------------------------------

def read_ply_xyz_part(path: Path):
    from plyfile import PlyData
    v = PlyData.read(str(path))["vertex"]
    xyz = np.stack([v["x"], v["y"], v["z"]], 1).astype(np.float64)
    part = np.asarray(v["part"]).astype(np.int64) if "part" in v.data.dtype.names else None
    return xyz, part


class GroundTruth:
    """What the dataset actually provides for one scene, with provenance."""

    FILES = {"joint_info": "metadata/joint_info.json", "joint_metadata": "pointcloud/joint_metadata.json",
             "frame_values": "metadata/frame_values.csv", "parts_frame0": "pointcloud/points3D_parts.ply",
             "parts_frameT": "pointcloud/points3D_parts_final.ply", "cameras": "metadata/cameras.json"}
    MEANING = {"joint_info": "selected joint name / type / dof / limits (SAPIEN articulation)",
               "joint_metadata": "joint axis_vector and point_of_application in coordinate_frame",
               "frame_values": "joint coordinate q per ORIGINAL frame index (rad revolute, m prismatic)",
               "parts_frame0": "surface samples of the object at frame 0, part id per point (0 static link, 1 moving link)",
               "parts_frameT": "surface samples at the final frame, part id per point",
               "cameras": "camera positions/targets used to render (gauge check only)"}

    def __init__(self, dataset_dir: Path):
        self.dir = Path(dataset_dir)
        self.files, self.missing, self.notes = {}, [], []
        for k, rel in self.FILES.items():
            p = self.dir / rel
            if p.is_file():
                self.files[k] = {"path": str(p), "sha256": sha256(p), "meaning": self.MEANING[k]}
            else:
                self.missing.append(k)
        mdir = self.dir / "mask"
        self.mask_dir = mdir if mdir.is_dir() else None
        self.joint = None
        if "joint_info" in self.files and "joint_metadata" in self.files:
            ji = json.loads(Path(self.files["joint_info"]["path"]).read_text())
            jm = json.loads(Path(self.files["joint_metadata"]["path"]).read_text())
            sel = ji.get("selected_joint") or {}
            typ = {"continuous": "revolute", "revolute": "revolute", "prismatic": "prismatic"}.get(
                sel.get("type") or jm.get("type"))
            self.joint = {"name": sel.get("name") or jm.get("name"), "type": typ,
                          "axis": unit(jm["axis_vector"]), "point": np.asarray(jm["point_of_application"], float),
                          "frame": jm.get("coordinate_frame"), "limits": sel.get("limits")}
            if sel.get("name") and jm.get("name") and sel["name"] != jm["name"]:
                self.notes.append(f"joint name mismatch {sel['name']} vs {jm['name']}")
        self.q = None
        if "frame_values" in self.files and self.joint:
            rows = list(csv.reader(open(self.files["frame_values"]["path"])))
            hdr = rows[0]
            col = hdr.index(self.joint["name"]) if self.joint["name"] in hdr else 1
            fr = [int(r[0]) for r in rows[1:]]
            if fr != list(range(len(fr))):
                self.notes.append("frame_values frame_index is not 0..T-1")
            self.q = np.array([float(r[col]) for r in rows[1:]])
        self.cloud = {}
        for k, key in (("parts_frame0", 0), ("parts_frameT", "T")):
            if k in self.files:
                self.cloud[key] = read_ply_xyz_part(Path(self.files[k]["path"]))
        self.sign = None
        self.motion_check = None

    @property
    def status(self):
        need_joint = all(k in self.files for k in ("joint_info", "joint_metadata", "frame_values"))
        need_geom = all(k in self.files for k in ("parts_frame0", "parts_frameT"))
        if not self.files:
            return "GT_UNAVAILABLE"
        return "GT_AVAILABLE" if need_joint and need_geom else "GT_INCOMPLETE"

    def gauge_check(self, colmap_sparse: Path) -> dict:
        """COLMAP camera centres (the 3DGS world) vs the renderer's camera positions."""
        if "cameras" not in self.files:
            return {"status": "NOT_AVAILABLE"}
        import sys
        sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
        from scene.colmap_loader import qvec2rotmat, read_extrinsics_binary
        ex = read_extrinsics_binary(str(colmap_sparse / "images.bin"))
        cams = {c["name"]: np.asarray(c["position"], float) for c in
                json.loads(Path(self.files["cameras"]["path"]).read_text())["cameras"]}
        err = 0.0
        for e in ex.values():
            R = qvec2rotmat(e.qvec)
            C = -R.T @ e.tvec
            n = e.name.split("/")[0]
            if n in cams:
                err = max(err, float(np.abs(C - cams[n]).max()))
        return {"status": "IDENTITY" if err < 1e-6 else "DIFFERENT", "max_abs_centre_difference_m": err,
                "meaning": "render_world (GT joint frame) == COLMAP/3DGS world when IDENTITY"}

    def transform(self, t_from: int, t_to: int) -> np.ndarray:
        """GT rigid motion of the MOVING link from frame t_from to t_to."""
        dq = (self.q[t_to] - self.q[t_from]) * (self.sign or 1.0)
        if self.joint["type"] == "revolute":
            return revolute_T(self.joint["axis"], self.joint["point"], dq)
        return prismatic_T(self.joint["axis"], dq)

    # The exporter's convention, verified from GT alone on the scenes whose frame-0
    # export contains moving-link samples (median landing distance of the frame-T
    # moving samples moved back by +dq: oven 3.5 cm, storage 2.3 cm, laptop 1.8 cm;
    # by -dq: 25.5 cm, 99 cm, 30 cm). Used only where a scene cannot verify it itself.
    EXPORTER_SIGN = 1.0
    EXPORTER_SIGN_EVIDENCE = {"oven_101917": [0.0349, 0.2550], "storage_45135": [0.0227, 0.9921],
                              "laptop_10211": [0.0175, 0.3003]}

    def verify_motion_sign(self, T_last: int) -> dict:
        """X_t = M(sigma (q(t) - q(t0))) X_t0 about point_of_application along axis.
        sigma is VERIFIED from this scene's GT when its frame-0 export has moving-link
        samples (the frame-T moving samples moved to frame 0 must land on them);
        otherwise the exporter convention is used and the check says so."""
        from scipy.spatial import cKDTree
        if self.q is None or 0 not in self.cloud or "T" not in self.cloud:
            return {"status": "NOT_AVAILABLE"}
        xT, pT = self.cloud["T"]
        x0, p0 = self.cloud[0]
        mov = xT[pT == 1] if pT is not None else np.empty((0, 3))
        mov0 = x0[p0 == 1] if p0 is not None else np.empty((0, 3))
        if mov.shape[0] < 50 or mov0.shape[0] < 50:
            self.sign = self.EXPORTER_SIGN
            return {"status": "NOT_VERIFIABLE_IN_THIS_SCENE", "sign": self.sign,
                    "reason": f"moving-link samples: {mov0.shape[0]} at frame 0, {mov.shape[0]} at frame T",
                    "used": "exporter convention, verified on other scenes (landing medians [+dq, -dq] m)",
                    "evidence": self.EXPORTER_SIGN_EVIDENCE}
        tree = cKDTree(mov0)
        res = {}
        for sgn in (1.0, -1.0):
            self.sign = sgn
            res[sgn] = float(np.median(tree.query(apply(self.transform(T_last, 0), mov))[0]))
        self.sign = min(res, key=res.get)
        other = res[-self.sign]
        return {"status": "VERIFIED" if res[self.sign] < 0.05 and other > 2 * res[self.sign] else "UNCERTAIN",
                "sign": self.sign, "median_landing_on_frame0_moving_samples_m": res,
                "agrees_with_exporter_convention": self.sign == self.EXPORTER_SIGN}

    def labelled_cloud(self, frame_key, T_last):
        """(xyz, part) at an endpoint. Frame T: as given. Frame 0: two mappings --
        'as_given' (points3D_parts.ply) and 'kinematic' (the frame-T samples with
        the moving link moved to frame 0 by the GT motion)."""
        if frame_key == "T":
            return {"as_given": self.cloud.get("T")}
        out = {"as_given": self.cloud.get(0)}
        if "T" in self.cloud and self.q is not None and self.cloud["T"][1] is not None:
            xT, pT = self.cloud["T"]
            x0 = xT.copy()
            m = pT == 1
            x0[m] = apply(self.transform(T_last, 0), xT[m])
            out["kinematic"] = (x0, pT.copy())
        return out


def map_labels(xyz_rows: np.ndarray, cloud, tol: float) -> np.ndarray:
    """Part label of each row from its nearest GT sample within tol; -1 unlabelled."""
    from scipy.spatial import cKDTree
    xyz_gt, part = cloud
    d, i = cKDTree(xyz_gt).query(xyz_rows, k=1)
    lab = part[i].copy()
    lab[d > tol] = -1
    return lab


def confusion(pred_pos: np.ndarray, gt_lab: np.ndarray) -> dict:
    """Binary scores with unlabelled rows (gt -1) EXCLUDED from every denominator."""
    m = gt_lab >= 0
    p, g = pred_pos[m], gt_lab[m] == 1
    tp, fp = int((p & g).sum()), int((p & ~g).sum())
    fn, tn = int((~p & g).sum()), int((~p & ~g).sum())

    def div(a, b):
        return a / b if b else None
    prec, rec = div(tp, tp + fp), div(tp, tp + fn)
    f1 = (2 * prec * rec / (prec + rec)) if prec and rec else (0.0 if prec == 0 or rec == 0 else None)
    return {"tp": tp, "fp": fp, "fn": fn, "tn": tn, "labelled_rows": int(m.sum()),
            "unlabelled_rows_excluded": int((~m).sum()), "precision": prec, "recall": rec, "f1": f1,
            "iou": div(tp, tp + fp + fn), "selection_fraction": div(int(p.sum()), int(m.sum())),
            "prevalence": div(int(g.sum()), int(m.sum()))}


def surface_scores(model_xyz: np.ndarray, gt_xyz: np.ndarray, tol: float) -> dict:
    """Centre-to-sample distances. accuracy: model rows -> GT samples (one-sided);
    completeness: GT samples -> model rows (one-sided); F-score of the two at tol."""
    from scipy.spatial import cKDTree
    if len(model_xyz) == 0 or len(gt_xyz) == 0:
        return {"status": "EMPTY"}
    d_m = cKDTree(gt_xyz).query(model_xyz)[0]
    d_g = cKDTree(model_xyz).query(gt_xyz)[0]
    acc, comp = float(np.mean(d_m <= tol)), float(np.mean(d_g <= tol))
    return {"status": "OK", "model_rows": len(model_xyz), "gt_samples": len(gt_xyz),
            "accuracy_median_m": float(np.median(d_m)), "precision_at_tol": acc,
            "completeness_median_m": float(np.median(d_g)), "recall_at_tol": comp,
            "f_score_at_tol": (2 * acc * comp / (acc + comp)) if acc + comp > 0 else 0.0}
