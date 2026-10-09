"""Section 39: score the REDUCED joint against the dataset's own geometry.

CODEX_IMPLEMENT_RIGID_SE3_PHASE2_AND_POSTHOC_JOINT_REDUCTION.md section 39.

Ground truth is read here and nowhere else. It never enters rigid Phase 2 and
it never enters the reduction: both are finished before this module is
imported, and the pipeline asserts that ordering.

Two files carry it:

    <source>/pointcloud/joint_metadata.json   axis_vector, point_of_application
                                              in `render_world`, the same frame
                                              the 3DGS model is trained in
    <source>/metadata/joint_info.json         joint type
    <source>/metadata/frame_values.csv        q(t)

`q(t)` is an ABSOLUTE joint coordinate with a dataset-specific offset, while
the reduction's `s_t`/`theta_t` start at zero at the canonical frame. Only
increments are comparable, so that is what is compared (section 39's "after
sign/gauge alignment").
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import torch
from torch import Tensor

NOT_AVAILABLE = "GT_NOT_AVAILABLE"


def load_joint_gt(source) -> dict:
    """Axis, point, type and q(t), or a stated reason for their absence."""
    source = Path(source)
    geo = source / "pointcloud" / "joint_metadata.json"
    info = source / "metadata" / "joint_info.json"
    vals = source / "metadata" / "frame_values.csv"
    out: dict = {"available": False, "source": str(geo)}
    if not geo.is_file():
        out["reason"] = f"{NOT_AVAILABLE}: no {geo}"
        return out
    d = json.loads(geo.read_text())
    axis = torch.tensor(d["axis_vector"], dtype=torch.float64).reshape(3)
    out.update({
        "available": True,
        "axis": (axis / axis.norm().clamp_min(1e-300)).tolist(),
        "axis_point": torch.tensor(d["point_of_application"],
                                   dtype=torch.float64).reshape(3).tolist(),
        "coordinate_frame": d.get("coordinate_frame"),
        "joint_name": d.get("name"),
        "joint_type_raw": d.get("type"),
        # SAPIEN calls an unlimited 1-DoF rotational joint "continuous"
        "joint_type": {"continuous": "revolute", "revolute": "revolute",
                       "prismatic": "prismatic"}.get(d.get("type"))})
    if info.is_file():
        sel = (json.loads(info.read_text()).get("selected_joint") or {})
        out["limits"] = sel.get("limits")
        name = sel.get("name")
        if vals.is_file() and name:
            rows = list(csv.reader(vals.open()))
            if rows:
                header, body = rows[0], rows[1:]
                col = header.index(name) if name in header else 1
                out["gt_q"] = [float(r[col]) for r in body if len(r) > col]
    return out


def axis_angle_error(e: Tensor, e_gt: Tensor) -> float:
    """`alpha = arccos(|e . e_gt|)` in degrees. The sign gauge is immaterial."""
    e = torch.as_tensor(e).double().reshape(3)
    g = torch.as_tensor(e_gt).double().reshape(3)
    c = (e @ g) / (e.norm() * g.norm()).clamp_min(1e-300)
    return float(torch.rad2deg(torch.arccos(c.abs().clamp(max=1.0))))


def line_to_line_distance(a1, e1, a2, e2) -> dict:
    """Shortest distance between two infinite lines, and how it was computed.

    Parallel lines need the point-to-line formula; the skew formula divides by
    `|e1 x e2|`, which is exactly zero when the two axes agree -- the case we
    most want a number for.
    """
    a1 = torch.as_tensor(a1).double().reshape(3)
    a2 = torch.as_tensor(a2).double().reshape(3)
    e1 = torch.as_tensor(e1).double().reshape(3)
    e2 = torch.as_tensor(e2).double().reshape(3)
    e1 = e1 / e1.norm().clamp_min(1e-300)
    e2 = e2 / e2.norm().clamp_min(1e-300)
    w = a2 - a1
    n = torch.linalg.cross(e1, e2)
    nn = float(n.norm())
    if nn < 1e-9:
        d = float((w - (w @ e1) * e1).norm())
        return {"distance": d, "case": "parallel",
                "formula": "||w - (w.e1) e1||"}
    return {"distance": float(abs(w @ n) / nn), "case": "skew",
            "formula": "|w . (e1 x e2)| / |e1 x e2|"}


def _increment_match(q_est, q_gt) -> dict:
    """Compare increments, with the sign gauge resolved by correlation."""
    a = torch.as_tensor(q_est, dtype=torch.float64).reshape(-1)
    b = torch.as_tensor(q_gt, dtype=torch.float64).reshape(-1)
    n = min(a.numel(), b.numel())
    if n < 2:
        return {"comparable": False,
                "reason": "fewer than two overlapping frames"}
    da, db = a[:n].diff(), b[:n].diff()
    sign = 1.0 if float(da @ db) >= 0 else -1.0
    da = sign * da
    den = float((db ** 2).sum())
    scale = float((da * db).sum() / den) if den > 1e-300 else float("nan")
    return {"comparable": True, "frames_compared": n, "sign": sign,
            "mean_increment_estimated": float(da.mean()),
            "mean_increment_gt": float(db.mean()),
            "increment_scale_ratio": scale,
            "increment_rms_error": float(((da - db) ** 2).mean().sqrt()),
            "increment_correlation": float(
                (da - da.mean()) @ (db - db.mean())
                / (da.std(unbiased=False) * db.std(unbiased=False)
                   * n).clamp_min(1e-300))
            if float(da.std()) > 0 and float(db.std()) > 0 else float("nan"),
            "note": ("the dataset's q is absolute with a joint offset while "
                     "the reduction's starts at zero, so only increments are "
                     "comparable")}


def evaluate(reduction: dict, gt: dict, frame_ids=None) -> dict:
    """Section 39, for both reduced models. Returns evidence, not a verdict."""
    if not gt.get("available"):
        return {"available": False, "reason": gt.get("reason", NOT_AVAILABLE)}
    e_gt, a_gt = gt["axis"], gt["axis_point"]
    P, R = reduction["prismatic"], reduction["revolute"]

    out = {"available": True,
           "gt_joint_type": gt.get("joint_type"),
           "gt_axis": e_gt, "gt_axis_point": a_gt,
           "gt_coordinate_frame": gt.get("coordinate_frame"),
           "prismatic": {"axis_angle_deg": axis_angle_error(P["direction"],
                                                            e_gt)},
           "revolute": {"axis_angle_deg": axis_angle_error(R["axis"], e_gt),
                        **{"axis_line_error": line_to_line_distance(
                            R["axis_point"], R["axis"], a_gt, e_gt)}}}
    # section 39: the axis POINT only means something for a revolute joint --
    # a prismatic axis is a direction, and any point on it is as good as any
    # other, so no line error is reported for P.
    out["prismatic"]["axis_line_error"] = {
        "distance": None,
        "case": "not applicable",
        "formula": "a prismatic joint has no distinguished axis point"}
    if gt.get("gt_q"):
        q_gt = gt["gt_q"]
        if frame_ids is not None:
            q_gt = [q_gt[i] for i in frame_ids if i < len(q_gt)]
        out["prismatic"]["q_match"] = _increment_match(P["s"], q_gt)
        out["revolute"]["q_match"] = _increment_match(R["theta"], q_gt)
    out["note"] = ("section 39: GT was read AFTER rigid Phase 2 and the "
                   "reduction finished. It never entered either.")
    return out
