"""Read-only ingestion of the ESTIMATED rigid motion and joint hypotheses.

Sources, per direction d in {forward, backward}:
  bidirectional/<d>_rigid/phase2_rigid/poses_original_order.npy   T_part(t), [T,4,4]
  bidirectional/<d>_rigid/phase2_rigid/pose_covariances.npy       [T,6,6]
  bidirectional/<d>_rigid/support_coherence/support.json          owned rows (members)
  bidirectional/<d>_rigid/joint_reduction/one_dof_diagnostics.json
  bidirectional/<d>_rigid/joint_reduction/relative_increments.json
  bidirectional/<d>_rigid/joint_reduction/{revolute,prismatic}/geometry.json
NEVER read: joint_reduction/gt_evaluation.json, <d>_rigid/summary.json (it embeds
the GT evaluation). The GT firewall (gt_firewall.py) enforces this.

Conventions, verified at load time and refused when they fail:
  * poses_original_order[t] maps the source's ANCHOR-frame geometry to original
    frame t: X_t = T_part(t) X_anchor, with T_part(anchor) = I (checked to 1e-9).
  * relative_increments.json must declare "Delta T_t = T_{t+1} T_t^{-1} (world/spatial)".
  * revolute: axis direction a, point p, angle theta(t); R(a, theta) = R(-a, -theta),
    so axes are compared up to sign as LINES (angle and point-to-line distance).
  * prismatic: direction d, displacement s(t); (d, s) = (-d, -s).

No joint type is chosen here unless the estimator recorded one. The existing
one_dof_diagnostics.json says "no joint type is decided", so BOTH hypotheses
are carried with their own fit residuals as evidence, never as a decision.

Usability (declared, not tuned on GT): an estimate is
  NO_MEASURABLE_MOTION  if the owned part's largest rigid displacement over the
                        sequence is < MIN_MOTION_M at the part scale and its
                        rotation < MIN_ROTATION_RAD -- it then cannot inform
                        transport;
  DEGENERATE            if the pose array is not a proper rigid track;
  USABLE                otherwise, with confidence notes (pose covariance trace,
                        revolute far-axis ratio, residual ratio).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .gt_firewall import assert_allowed

MIN_MOTION_M = 0.01
MIN_ROTATION_RAD = 0.02
FAR_AXIS_RATIO = 10.0


def _read_json(p: Path):
    assert_allowed(p)
    return json.loads(Path(p).read_text())


def _load(p: Path):
    assert_allowed(p)
    return np.load(p)


def rodrigues(axis, theta):
    a = np.asarray(axis, float)
    a = a / np.linalg.norm(a)
    K = np.array([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]])
    return np.eye(3) + np.sin(theta) * K + (1 - np.cos(theta)) * K @ K


def revolute_transform(axis, point, theta):
    """X -> R (X - p) + p about the line (p, axis)."""
    R = rodrigues(axis, theta)
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = np.asarray(point, float) - R @ np.asarray(point, float)
    return T


def prismatic_transform(direction, s):
    d = np.asarray(direction, float)
    T = np.eye(4)
    T[:3, 3] = d / np.linalg.norm(d) * s
    return T


def line_compare(a1, p1, a2, p2) -> dict:
    """Two revolute axes as lines, sign-invariant."""
    a1 = np.asarray(a1, float) / np.linalg.norm(a1)
    a2 = np.asarray(a2, float) / np.linalg.norm(a2)
    ang = float(np.degrees(np.arccos(np.clip(abs(a1 @ a2), 0, 1))))
    d = np.asarray(p2, float) - np.asarray(p1, float)
    perp = d - (d @ a1) * a1
    return {"axis_angle_deg_up_to_sign": ang, "point2_to_line1_distance_m": float(np.linalg.norm(perp))}


def direction_compare(d1, d2) -> dict:
    d1 = np.asarray(d1, float) / np.linalg.norm(d1)
    d2 = np.asarray(d2, float) / np.linalg.norm(d2)
    return {"direction_angle_deg_up_to_sign": float(np.degrees(np.arccos(np.clip(abs(d1 @ d2), 0, 1))))}


def apply(T, X):
    return X @ T[:3, :3].T + T[:3, 3]


class DirectionEstimate:
    """One direction's estimated part motion and joint hypotheses."""

    def __init__(self, bidi: Path, direction: str, anchor_frame: int, n_rows: int,
                 anchor_xyz: np.ndarray | None = None):
        self.direction, self.anchor = direction, int(anchor_frame)
        root = Path(bidi) / f"{direction}_rigid"
        self.root = root
        self.status, self.notes, self.files = "UNAVAILABLE", [], {}
        need = {"poses": root / "phase2_rigid" / "poses_original_order.npy",
                "covariances": root / "phase2_rigid" / "pose_covariances.npy",
                "support": root / "support_coherence" / "support.json",
                "one_dof": root / "joint_reduction" / "one_dof_diagnostics.json",
                "increments": root / "joint_reduction" / "relative_increments.json",
                "revolute": root / "joint_reduction" / "revolute" / "geometry.json",
                "prismatic": root / "joint_reduction" / "prismatic" / "geometry.json"}
        missing = [k for k, p in need.items() if not p.is_file()]
        if missing:
            self.notes.append(f"missing {missing}")
            return
        self.files = {k: str(p) for k, p in need.items()}
        P = _load(need["poses"]).astype(np.float64)
        C = _load(need["covariances"]).astype(np.float64)
        sup = _read_json(need["support"])
        od = _read_json(need["one_dof"])
        inc = _read_json(need["increments"])
        rev = _read_json(need["revolute"])
        pri = _read_json(need["prismatic"])
        self.poses, self.cov = P, C
        self.members = np.asarray(sup.get("members", []), dtype=np.int64)
        self.support = {k: sup.get(k) for k in ("method", "class_used", "kept_fraction",
                                                "num_members", "frozen_before_phase2")}
        self.support["identification_status"] = (sup.get("identification") or {}).get("status")
        self.support["all_rows_owned"] = bool(self.members.size == n_rows)
        # --- convention checks ------------------------------------------------
        checks = {}
        checks["poses_shape"] = list(P.shape)
        checks["anchor_is_identity"] = float(np.abs(P[self.anchor] - np.eye(4)).max())
        Rs = P[:, :3, :3]
        checks["max_orthonormality_defect"] = float(np.abs(np.einsum("tij,tkj->tik", Rs, Rs)
                                                           - np.eye(3)).max())
        checks["min_det"] = float(np.linalg.det(Rs).min())
        checks["increment_convention"] = inc.get("convention")
        checks["increment_convention_ok"] = "T_{t+1} T_t^{-1}" in str(inc.get("convention")) \
            and "world" in str(inc.get("convention"))
        if inc.get("zeta") is not None:
            checks["num_increments"] = len(inc["zeta"])
        self.checks = checks
        if (P.ndim != 3 or P.shape[1:] != (4, 4) or checks["anchor_is_identity"] > 1e-9
                or checks["max_orthonormality_defect"] > 1e-6 or checks["min_det"] < 0.99):
            self.status = "DEGENERATE"
            self.notes.append(f"pose track fails the convention checks {checks}")
            return
        if not checks["increment_convention_ok"]:
            self.notes.append("relative_increments convention differs from the declared "
                              "world/spatial T_{t+1} T_t^{-1}; increments not used")
        # --- hypotheses (no decision) ----------------------------------------
        self.decided_type = od.get("joint_type") or od.get("decided_type")
        self.one_dof = {k: od.get(k) for k in ("gamma_1dof", "generator_translation_norm",
                                               "generator_rotation_norm", "part_scale_L", "note")}
        rr = rev.get("refined", rev)
        pr = pri.get("refined", pri)
        self.revolute = {"axis": rr.get("axis", rev.get("axis")),
                         "axis_point": rr.get("axis_point", rev.get("axis_point")),
                         "theta": rr.get("theta", rev.get("theta")),
                         "rank1_fraction": rev.get("rank1_fraction"),
                         "axis_radius_over_part_scale": rev.get("axis_radius_over_part_scale"),
                         "residual": rev.get("pose_residual"),
                         "max_rotation_angle": rev.get("max_rotation_angle")}
        self.prismatic = {"direction": pr.get("direction", pri.get("direction")),
                          "s": pr.get("s", pri.get("s")),
                          "rank1_fraction": pri.get("translation_rank1_fraction"),
                          "residual": pri.get("pose_residual"),
                          "residual_rotation_angle_max": pri.get("residual_rotation_angle_max")}
        # --- motion magnitude of the OWNED rows, from the pose track only ----
        if anchor_xyz is not None and self.members.size:
            X = anchor_xyz[self.members]
            disp = max(float(np.linalg.norm(apply(P[t], X) - X, axis=1).max()) for t in range(P.shape[0]))
        else:
            disp = float(np.linalg.norm(P[:, :3, 3], axis=1).max())
        rot = float(max(np.arccos(np.clip((np.trace(P[t, :3, :3]) - 1) / 2, -1, 1))
                        for t in range(P.shape[0])))
        self.motion = {"max_owned_displacement_m": disp, "max_rotation_rad": rot}
        er = (self.revolute["residual"] or {}).get("unweighted_energy")
        ep = (self.prismatic["residual"] or {}).get("unweighted_energy")
        self.residual_preference = None
        if er is not None and ep is not None and min(er, ep) > 0:
            ratio = max(er, ep) / min(er, ep)
            self.residual_preference = {
                "lower_residual": "revolute" if er < ep else "prismatic",
                "residual_ratio": ratio, "revolute_energy": er, "prismatic_energy": ep,
                "meaning": "evidence only: both fits were made to the same rigid track; "
                           "no type was decided by the estimator"}
        far = self.revolute["axis_radius_over_part_scale"]
        if far is not None and far > FAR_AXIS_RATIO:
            self.notes.append(f"revolute axis {far:.1f} part radii from the part: the revolute "
                              "hypothesis approximates a translation (far axis)")
        tr = np.trace(C, axis1=1, axis2=2)
        self.uncertainty = {"pose_cov_trace_median": float(np.median(tr)),
                            "pose_cov_trace_max": float(tr.max())}
        if disp < MIN_MOTION_M and rot < MIN_ROTATION_RAD:
            self.status = "NO_MEASURABLE_MOTION"
            self.notes.append(f"owned part moves at most {disp * 1000:.2f} mm and {rot:.4f} rad: "
                              "this estimate cannot inform transport (J1 unavailable for it)")
        else:
            self.status = "USABLE"
        if self.support["all_rows_owned"]:
            self.notes.append("support owns every row of the model: ownership is not informative")

    def transform(self, t_from: int, t_to: int) -> np.ndarray:
        """T_part(t_to) T_part(t_from)^{-1}: moves owned geometry between frames."""
        return self.poses[t_to] @ np.linalg.inv(self.poses[t_from])

    def describe(self) -> dict:
        out = {"direction": self.direction, "status": self.status, "notes": self.notes,
               "files": self.files, "anchor_frame": self.anchor}
        if self.status in ("USABLE", "NO_MEASURABLE_MOTION"):
            out |= {"convention_checks": self.checks, "support": self.support,
                    "decided_type": self.decided_type or "NONE (estimator decided no type; "
                                                         "both hypotheses reported)",
                    "one_dof": self.one_dof, "motion": self.motion,
                    "uncertainty": self.uncertainty,
                    "residual_preference": self.residual_preference,
                    "revolute": {k: v for k, v in self.revolute.items() if k != "theta"}
                    | {"theta_range": [float(np.min(self.revolute["theta"])),
                                       float(np.max(self.revolute["theta"]))]},
                    "prismatic": {k: v for k, v in self.prismatic.items() if k != "s"}
                    | {"s_range": [float(np.min(self.prismatic["s"])),
                                   float(np.max(self.prismatic["s"]))]}}
        return out


def cross_direction(fw: DirectionEstimate, bw: DirectionEstimate) -> dict:
    """Forward vs backward estimates of the same joint, sign-invariant."""
    if fw.status != "USABLE" or bw.status != "USABLE":
        return {"comparable": False,
                "reason": f"forward {fw.status}, backward {bw.status}"}
    return {"comparable": True,
            "revolute": line_compare(fw.revolute["axis"], fw.revolute["axis_point"],
                                     bw.revolute["axis"], bw.revolute["axis_point"]),
            "prismatic": direction_compare(fw.prismatic["direction"], bw.prismatic["direction"])}
