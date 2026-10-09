"""Sections 43-48: the numeric artifacts of the dual differential/finite layer.

CODEX_IMPLEMENT_DUAL_DIFFERENTIAL_FINITE_JOINT_DISCOVERY.md.

Every quantity the hierarchy of section 35 consulted is written out, in the
column order the spec fixes, so a verdict can be traced back to the frame and
the statistic that produced it. Section 39's question -- does the angular
motion exist in Phase 1, or does it appear only afterwards -- is answerable
from `differential_finite_consistency.csv` alone.

Nothing here computes anything. If a quantity was not measured it is written
as an empty field with the reason recorded, never as a fabricated zero.
"""

from __future__ import annotations

import math

import torch

VELOCITY_GRADIENT_HEADER = (
    ["frame", "vc_x", "vc_y", "vc_z"]
    + [f"G_{i}{j}" for i in range(3) for j in range(3)]
    + [f"D_{i}{j}" for i in range(3) for j in range(3)]
    + [f"Omega_{i}{j}" for i in range(3) for j in range(3)]
    + ["omega_x", "omega_y", "omega_z",
       "strain_stat", "affine_rank", "affine_condition",
       "chi2_rigid_vs_affine", "velocity_source"])

RIGID_TWIST_HEADER = (
    ["frame", "v0_x", "v0_y", "v0_z", "omega_x", "omega_y", "omega_z",
     "twist_rank", "twist_condition", "rigid_fit_error",
     "chi2_omega_zero", "omega_rank"]
    + [f"omega_cov_{i}{j}" for i in range(3) for j in range(3)])

FINITE_ROTATION_HEADER = (
    ["frame"] + [f"R_{i}{j}" for i in range(3) for j in range(3)]
    + ["phi_x", "phi_y", "phi_z", "angle", "kabsch_error",
       "shape_rank", "rotation_observability"])

CONSISTENCY_HEADER = [
    "frame", "omega_diff_x", "omega_diff_y", "omega_diff_z",
    "omega_finite_x", "omega_finite_y", "omega_finite_z",
    "difference_norm", "mahalanobis_difference", "axis_cosine"]

EDGE_HEADER = ["frame", "i", "j", "separation", "rho_diff",
               "rho_diff_variance", "rho_diff_normalized"]


def _f(v):
    """A float, or an empty field when the quantity does not exist."""
    if v is None:
        return ""
    try:
        x = float(v)
    except (TypeError, ValueError):
        return ""
    return "" if math.isnan(x) else x


def _flat(m):
    return [float(x) for x in torch.as_tensor(m).double().reshape(-1)]


def velocity_gradient_rows(evidence) -> list:
    """Section 44."""
    rows = []
    src = evidence["velocity"]["velocity_source"]
    for t, a in enumerate(evidence["affine"]):
        rows.append([t] + _flat(a["v_c"]) + _flat(a["G"]) + _flat(a["D"])
                    + _flat(a["Omega"]) + _flat(a["omega"])
                    + [_f(evidence["strain"][t]["Q_D"]), int(a["rank"]),
                       _f(a["condition"]),
                       _f(evidence["rigidity_chi2"][t]),
                       src[t] if t < len(src) else ""])
    return rows


def rigid_twist_rows(evidence) -> list:
    """Section 45."""
    rows = []
    xi = evidence["twist"]["xi_spatial"]
    for t, f in enumerate(evidence["twist"]["fits"]):
        rows.append([t] + _flat(xi[t][:3]) + _flat(f["omega"])
                    + [int(f["rank"]), _f(f["condition"]),
                       _f(f["rigid_fit_error"]), _f(f["chi2_omega_zero"]),
                       int(f["omega_rank"])]
                    + _flat(f["omega_covariance"]))
    return rows


def finite_rotation_rows(evidence) -> list:
    """Section 46."""
    fin = evidence["finite"]
    rows = []
    for t in range(fin["R"].shape[0]):
        per = fin["per_frame"][t]
        rows.append([t] + _flat(fin["R"][t]) + _flat(fin["phi"][t])
                    + [_f(fin["angle"][t]), _f(per["kabsch_error"]),
                       int(per["observability_shape_rank"]),
                       _f(per["observability_least_observable_axis"])])
    return rows


def consistency_rows(evidence) -> list:
    """Section 47. Stamped at interval midpoints, as the comparison is."""
    c = evidence["angular_comparison"]
    rows = []
    for t in range(c["omega_diff"].shape[0]):
        rows.append([t] + _flat(c["omega_diff"][t]) + _flat(c["omega_finite"][t])
                    + [_f(c["difference_norm"][t]),
                       _f(c["mahalanobis_difference"][t]),
                       _f(c["axis_cosine"][t])])
    return rows


def edge_rows(evidence, max_rows: int = 200000) -> list:
    """Section 31's per-edge terms, when differential edges were requested."""
    er = evidence.get("edge_rigidity")
    if not er:
        return []
    rows = []
    for t, e in enumerate(er):
        ed = e["edges"]
        for k in range(ed.shape[0]):
            rows.append([t, int(ed[k, 0]), int(ed[k, 1]),
                         _f(e["separation"][k]), _f(e["rho_diff"][k]),
                         _f(e["rho_diff_variance"][k]),
                         _f(e["rho_diff_normalized"][k])])
            if len(rows) >= max_rows:
                return rows
    return rows


def common_screw_payload(evidence, candidate=None) -> dict:
    """Section 48."""
    s = evidence["screw"]
    rev, pri, full, ang = s["revolute"], s["prismatic"], s["twist_rank"], s["angular_rank"]
    out = {
        "xi_J": _flat(full["xi_J"]),
        "axis_direction": (None if rev.get("axis_direction") is None
                           else _flat(rev["axis_direction"])),
        "axis_point": (None if rev.get("axis_point") is None
                       else _flat(rev["axis_point"])),
        "pitch": _f(rev.get("pitch")),
        "qdot": _flat(full["rates"]),
        "q": _flat(torch.cumsum(full["rates"], 0)),
        "rank1_residual": _f(full["chi2_rank1"]),
        "rank1_dof": int(full["dof_rank1"]),
        "angular_rank": {"chi2_rank0": _f(ang["chi2_rank0"]),
                         "dof_rank0": int(ang["dof_rank0"]),
                         "chi2_rank1": _f(ang["chi2_rank1"]),
                         "dof_rank1": int(ang["dof_rank1"]),
                         "axis": _flat(ang["axis"])},
        "twist_rank": {"chi2_rank1": _f(full["chi2_rank1"]),
                       "dof": int(full["dof_rank1"]),
                       "per_dof": _f(full["rank1_per_dof"])},
        "prismatic_candidate": {
            "direction_of_travel": _flat(pri["direction_of_travel"]),
            "chi2_rank1": _f(pri["chi2_rank1"])},
        "revolute_candidate": {
            "identifiable": bool(rev.get("identifiable", False)),
            "zero_pitch_residual": _f(rev.get("zero_pitch_residual"))},
        "geometry_status": (None if candidate is None
                            else candidate.get("status")),
        "note": ("section 36: the common screw is a parameter estimator and "
                 "consistency test, not the classifier"),
    }
    return out


def joint_type_evidence_payload(evidence, candidate, rgbd=None) -> dict:
    """Section 43's `joint_type_evidence.json` and section 64's report row."""
    att = evidence["angular_attribution"]
    cmp = evidence["angular_comparison"]
    payload = {
        "final_status": (candidate.get("status") if rgbd is None
                         else rgbd.get("status")),
        "geometric_candidate": candidate.get("candidate"),
        "reason": candidate.get("reason"),
        "confidence": candidate.get("confidence"),
        "stage_a_rigid": {k: (_f(v) if isinstance(v, float) else v)
                          for k, v in candidate["stage_a_rigid"].items()},
        "stage_b_one_dof": {k: (_f(v) if isinstance(v, float) else v)
                            for k, v in candidate["stage_b_one_dof"].items()},
        "stage_c_angular": {k: (_f(v) if isinstance(v, float) else v)
                            for k, v in candidate["stage_c_angular"].items()},
        "median_strain_stat": _f(evidence["strain_statistic"].median()),
        "median_chi2_rigid_vs_affine": _f(evidence["rigidity_chi2"].median()),
        "median_omega_diff": _f(att["median_omega_diff"]),
        "median_omega_finite": _f(att["median_omega_finite"]),
        "differential_finite_disagreement": _f(cmp["median_difference_norm"]),
        "median_axis_cosine": _f(cmp["median_axis_cosine"]),
        "angular_attribution": att["attribution"],
        "angular_attribution_reason": att["reason"],
        "decision_inputs": candidate.get("decision_inputs"),
    }
    if rgbd is not None:
        payload["rgbd_confirmation"] = {
            "status": rgbd.get("status"),
            "confirmed": rgbd.get("confirmed"),
            "reason": rgbd.get("reason"),
            "train_gain_total": _f(rgbd.get("train_gain_total")),
            "heldout_gain_total": _f(rgbd.get("heldout_gain_total")),
            "chi2_cut_curvature": _f(rgbd.get("chi2_cut_curvature"))}
    else:
        payload["rgbd_confirmation"] = {
            "status": None,
            "reason": ("section 34: the renderer confirmation was not run; "
                       "no RGB-D evidence exists for this part")}
    return payload


def write_dual_joint_artifacts(writer, evidence, candidate, rgbd=None, *,
                               prefix: str = "") -> list:
    """Sections 43-48: emit every artifact. Returns the names written."""
    p = prefix
    writer.csv(f"{p}velocity_gradient_by_frame", VELOCITY_GRADIENT_HEADER,
               velocity_gradient_rows(evidence))
    writer.csv(f"{p}rigid_twist_by_frame", RIGID_TWIST_HEADER,
               rigid_twist_rows(evidence))
    writer.csv(f"{p}finite_rotation_by_frame", FINITE_ROTATION_HEADER,
               finite_rotation_rows(evidence))
    writer.csv(f"{p}differential_finite_consistency", CONSISTENCY_HEADER,
               consistency_rows(evidence))
    rows = edge_rows(evidence)
    if rows:
        writer.csv(f"{p}differential_rigidity_edges", EDGE_HEADER, rows)
    writer.json(f"{p}common_screw_fit",
                common_screw_payload(evidence, candidate))
    writer.json(f"{p}joint_type_evidence",
                joint_type_evidence_payload(evidence, candidate, rgbd))
    return [f"{p}velocity_gradient_by_frame", f"{p}rigid_twist_by_frame",
            f"{p}finite_rotation_by_frame",
            f"{p}differential_finite_consistency",
            f"{p}common_screw_fit", f"{p}joint_type_evidence"]
