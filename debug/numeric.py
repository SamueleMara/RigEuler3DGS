"""The numerical debug directory.

CODEX_ADD_VISUAL_AND_NUMERICAL_DEBUG_GEOMETRIC_PIPELINE.md sections 13-38.

Section 0 rule 5 is the purpose of this file: persist enough numerical state
that every visual feature can be checked independently. Section 13 lists the
files; sections 14-37 list what goes in them; section 38 turns the numbers into
an automatic, non-speculative failure summary.

Section 13 also requires these to be written even when the run is unresolved,
so every emitter degrades to an explicit "not available at this stage" record
rather than being skipped.
"""

from __future__ import annotations

import math

import torch

from estimator.manifold import project_to_so3, so3_log

from .recorder import GeometricDebug, part_color

EPS = 1e-12


def _q(rot: torch.Tensor) -> list:
    """Rotation matrix -> quaternion (w, x, y, z), for section 21."""
    m = rot.double()
    tr = float(m[0, 0] + m[1, 1] + m[2, 2])
    if tr > 0:
        s = math.sqrt(tr + 1.0) * 2
        w = 0.25 * s
        x = float(m[2, 1] - m[1, 2]) / s
        y = float(m[0, 2] - m[2, 0]) / s
        z = float(m[1, 0] - m[0, 1]) / s
    elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
        s = math.sqrt(1.0 + float(m[0, 0] - m[1, 1] - m[2, 2])) * 2
        w = float(m[2, 1] - m[1, 2]) / s
        x, y, z = 0.25 * s, float(m[0, 1] + m[1, 0]) / s, float(m[0, 2] + m[2, 0]) / s
    elif m[1, 1] > m[2, 2]:
        s = math.sqrt(1.0 + float(m[1, 1] - m[0, 0] - m[2, 2])) * 2
        w = float(m[0, 2] - m[2, 0]) / s
        x, y, z = float(m[0, 1] + m[1, 0]) / s, 0.25 * s, float(m[1, 2] + m[2, 1]) / s
    else:
        s = math.sqrt(1.0 + float(m[2, 2] - m[0, 0] - m[1, 1])) * 2
        w = float(m[1, 0] - m[0, 1]) / s
        x, y, z = float(m[0, 2] + m[2, 0]) / s, float(m[1, 2] + m[2, 1]) / s, 0.25 * s
    return [w, x, y, z]


def _stats(v: torch.Tensor) -> dict:
    """Section 22's required aggregation."""
    if v.numel() == 0:
        return {"mean": 0.0, "median": 0.0, "p90": 0.0, "p95": 0.0, "max": 0.0}
    d = v.double().reshape(-1)
    return {"mean": float(d.mean()), "median": float(d.median()),
            "p90": float(d.quantile(0.90)), "p95": float(d.quantile(0.95)),
            "max": float(d.max())}


# ===========================================================================
def write_numeric_debug(dbg: GeometricDebug, objective, writer) -> dict:
    """Emit section 13's `debug_numeric/` directory. Returns the index."""
    if not dbg.enabled:
        return {}
    root = writer.dir / "debug_numeric"
    root.mkdir(parents=True, exist_ok=True)

    class _W:
        """A writer rooted in debug_numeric/, reusing the artifact writer."""

        def __init__(self, d):
            self.dir = d
            self.written = []

        json = writer.__class__.json
        tensor = writer.__class__.tensor
        csv = writer.__class__.csv
        text = writer.__class__.text

    w = _W(root)

    energies = _stage_energies(dbg, objective)
    _run_summary(dbg, w, energies)
    _frame_metrics(dbg, w, energies)
    _camera_metrics(dbg, w)
    _bootstrap_motion(dbg, w)
    _supernode_debug(dbg, w)
    _support_iterations(dbg, w)
    _rigid_transform_history(dbg, w)
    _rigid_residuals(dbg, w)
    _rigid_invariants(dbg, w)
    _geometric_fit_summary(dbg, w)
    _q_trajectories(dbg, w)
    _axis_history(dbg, w)
    _renderer_refinement(dbg, w)
    _reduced_filter(dbg, w)
    _projection_losses(dbg, w, energies)
    _handover(dbg, w)
    _convergence_summary(dbg, w)
    failures = _failure_summary(dbg, w, energies)

    index = {"directory": "debug_numeric", "files": sorted(w.written),
             "failures": failures}
    writer.json("debug_numeric_index", index)
    return index


# ---------------------------------------------------------------------------
# sections 35/36: the projection losses, and the per-frame energies they need
# ---------------------------------------------------------------------------
@torch.no_grad()
def _stage_energies(dbg: GeometricDebug, objective) -> dict:
    """RGB-D energy per frame at each estimator stage.

    Sections 35 and 36 are differences of these, and section 36 calls the
    rigid -> joint loss "one of the most important diagnostics": it is exactly
    the explanatory power given up by constraining a free rigid trajectory to a
    one-parameter joint.
    """
    out = {s: [] for s in ("canonical", "bootstrap", "rigid", "refined")}
    if objective is None or dbg.mu0 is None:
        return out
    for t, (obs, _fid) in enumerate(objective.frames):
        for stage in list(out):
            mu = dbg.positions_at(stage, t)
            if mu is None:
                out[stage].append(None)
                continue
            R = dbg.orientations_at(stage, t)
            er, ed = objective.frame_energy(mu, R, obs)
            out[stage].append({"rgb": float(er), "depth": float(ed),
                               "total": float(er + ed)})
    return out


def _projection_losses(dbg, w, energies) -> None:
    rows = []
    tot = {"bootstrap_to_rigid": 0.0, "rigid_to_joint": 0.0}
    n = dbg.num_frames
    for t in range(n):
        b = (energies["bootstrap"][t] or {}).get("total") if t < len(energies["bootstrap"]) else None
        r = (energies["rigid"][t] or {}).get("total") if t < len(energies["rigid"]) else None
        j = (energies["refined"][t] or {}).get("total") if t < len(energies["refined"]) else None
        d1 = (r - b) if (b is not None and r is not None) else None
        d2 = (j - r) if (r is not None and j is not None) else None
        if d1 is not None:
            tot["bootstrap_to_rigid"] += d1
        if d2 is not None:
            tot["rigid_to_joint"] += d2
        rows.append([t, _f(b), _f(r), _f(j), _f(d1), _f(d2)])
    w.csv("projection_losses", ["frame", "E_bootstrap", "E_rigid", "E_joint",
                                "delta_bootstrap_to_rigid",
                                "delta_rigid_to_joint"], rows)
    w.json("projection_losses_summary", {
        "section_35_bootstrap_to_rigid": tot["bootstrap_to_rigid"],
        "section_36_rigid_to_joint": tot["rigid_to_joint"],
        "meaning": {
            "section_35": ("E_rigid - E_bootstrap. Large means the K-part "
                           "rigid decomposition is already wrong before any "
                           "joint is fitted."),
            "section_36": ("E_joint - E_rigid. The explanatory power lost by "
                           "constraining a free rigid trajectory to P or R.")}})


def _f(x):
    return "" if x is None else float(x)


# ---------------------------------------------------------------------------
# section 14
# ---------------------------------------------------------------------------
def _run_summary(dbg, w, energies) -> None:
    import time
    per_part = []
    for k in range(1, dbg.K + 1):
        j = dbg.joints[k - 1] if k - 1 < len(dbg.joints) else None
        per_part.append({
            "part": k,
            "selected": (j or {}).get("selected"),
            "status": (j or {}).get("status", "no fit"),
            "num_gaussians": int((dbg.labels == k).sum()) if dbg.labels is not None else 0,
            "normalized_error": (j or {}).get("normalized_error"),
        })
    w.json("run_summary", {
        "model": dbg.model, "dataset": dbg.dataset,
        "num_gaussians": dbg.num_gaussians,
        "num_supernodes": dbg.num_supernodes,
        "K": dbg.K, "frames": dbg.num_frames,
        "cameras": len(dbg.camera_meta),
        "image_size": list(dbg.image_size),
        "valid_rgb_px": sum(r.get("valid_rgb_px", 0) for r in dbg.camera_rows),
        "valid_depth_px": sum(r.get("valid_depth_px", 0) for r in dbg.camera_rows),
        "runtime_s": time.perf_counter() - dbg.started,
        "peak_cuda_gib": (torch.cuda.max_memory_allocated() / 2 ** 30
                          if torch.cuda.is_available() else 0.0),
        "status": dbg.status,
        "phases": {
            "phase1_motion_evidence": bool(dbg.positions),
            "phase2_rigid_parts": {"ran": dbg.transforms is not None,
                                   "converged": dbg.rigid_converged,
                                   "stop_reason": dbg.rigid_stop_reason},
            "phase3_geometric_joint": {"ran": bool(dbg.joints)},
            "phase4_renderer_refinement": {"ran": bool(dbg.refine_info),
                                           **(dbg.refine_info or {})},
            "phase5_reduced_filter": {"ran": bool(dbg.reduced)},
        },
        "per_part": per_part,
        "part_colors": {str(k): part_color(k) for k in range(dbg.K + 1)},
        "notes": dbg.notes,
    })


# ---------------------------------------------------------------------------
# section 15
# ---------------------------------------------------------------------------
def _frame_metrics(dbg, w, energies) -> None:
    K = dbg.K
    header = ["frame", "timestamp", "dt", "visible_gaussians",
              "active_supernodes", "dynamic_gaussians", "dynamic_supernodes",
              "rgb_energy", "depth_energy", "total_observation_energy",
              "bootstrap_step_norm", "rigid_stage_energy", "joint_stage_energy"]
    header += [f"q_{k}" for k in range(1, K + 1)]
    header += [f"qdot_{k}" for k in range(1, K + 1)]
    rows = []
    dyn_g = int((dbg.labels > 0).sum()) if dbg.labels is not None else 0
    dyn_c = int((dbg.node_labels > 0).sum()) if dbg.node_labels is not None else 0
    for t in range(dbg.num_frames):
        fid = dbg.frame_ids[t] if t < len(dbg.frame_ids) else t
        fr = next((r for r in dbg.frame_rows if r.get("frame") == fid), {})
        e = energies["refined"][t] if t < len(energies["refined"]) else None
        e = e or (energies["rigid"][t] if t < len(energies["rigid"]) else None) or {}
        step = 0.0
        if dbg.positions and t < len(dbg.positions) and dbg.mu0 is not None:
            step = float((dbg.positions[t] - dbg.mu0).norm(dim=1).mean())
        row = [fid, t * dbg.dt, dbg.dt,
               fr.get("visible_gaussians", ""),
               dbg.num_supernodes, dyn_g, dyn_c,
               e.get("rgb", ""), e.get("depth", ""), e.get("total", ""),
               step,
               (energies["rigid"][t] or {}).get("total", "")
               if t < len(energies["rigid"]) and energies["rigid"][t] else "",
               (energies["refined"][t] or {}).get("total", "")
               if t < len(energies["refined"]) and energies["refined"][t] else ""]
        for k in range(K):
            row.append(dbg.q_traj[k][t] if k < len(dbg.q_traj)
                       and t < len(dbg.q_traj[k]) else "")
        for k in range(K):
            if k < len(dbg.q_traj) and 0 < t < len(dbg.q_traj[k]):
                row.append((dbg.q_traj[k][t] - dbg.q_traj[k][t - 1]) / max(dbg.dt, EPS))
            else:
                row.append("")
        rows.append(row)
    w.csv("frame_metrics", header, rows)


# ---------------------------------------------------------------------------
# section 16
# ---------------------------------------------------------------------------
def _camera_metrics(dbg, w) -> None:
    header = ["frame", "camera_id", "valid_rgb_px", "valid_depth_px",
              "visible_gaussians", "rgb_rmse", "rgb_robust_energy",
              "depth_rmse", "depth_robust_energy", "motion_sensitive_px",
              "fisher_trace"]
    rows = [[r.get(h, "") for h in header] for r in dbg.camera_rows]
    w.csv("camera_metrics", header, rows)
    if not any(r.get("fisher_trace") not in ("", None) for r in dbg.camera_rows):
        w.json("camera_metrics_note", {
            "fisher_trace": ("not defined for this run: the joint Fisher "
                             "H_q^T R^-1 H_q needs a joint model, and this run "
                             "did not reach one. The column is left empty "
                             "rather than zero, which would read as 'the "
                             "cameras carry no information'.")})
    if dbg.multiview:
        w.json("multiview_consistency", {
            "method": ("section 32: per camera dq_{t,c} = g_{t,c} / "
                       "(h_{t,c} + eps), the joint step that camera alone "
                       "would ask for"),
            "rows": dbg.multiview})


# ---------------------------------------------------------------------------
# section 17
# ---------------------------------------------------------------------------
def _bootstrap_motion(dbg, w) -> None:
    header = ["frame", "supernode", "dx", "dy", "dz", "drot_x", "drot_y",
              "drot_z", "vx", "vy", "vz", "omega_x", "omega_y", "omega_z",
              "motion_evidence", "member_count"]
    rows = []
    if dbg.positions and dbg.supernode_id is not None and dbg.mu0 is not None:
        sid = dbg.supernode_id.long().clamp_min(0)
        C = dbg.num_supernodes
        cnt = torch.zeros(C, device=sid.device).index_add_(
            0, sid, torch.ones(sid.shape[0], device=sid.device))
        ev = dbg.motion_statistic
        prev = dbg.mu0
        for t, pos in enumerate(dbg.positions):
            d = pos - dbg.mu0
            dc = torch.zeros((C, 3), device=d.device).index_add_(0, sid, d) \
                / cnt.clamp_min(1).unsqueeze(1)
            v = (pos - prev) / max(dbg.dt, EPS)
            vc = torch.zeros((C, 3), device=d.device).index_add_(0, sid, v) \
                / cnt.clamp_min(1).unsqueeze(1)
            fid = dbg.frame_ids[t] if t < len(dbg.frame_ids) else t
            for c in range(C):
                rows.append([
                    fid, c,
                    float(dc[c, 0]), float(dc[c, 1]), float(dc[c, 2]),
                    "", "", "",                      # per-supernode rotation
                    float(vc[c, 0]), float(vc[c, 1]), float(vc[c, 2]),
                    "", "", "",                      # omega: see section 12
                    float(ev[c]) if ev is not None and c < ev.numel() else "",
                    int(cnt[c])])
            prev = pos
    w.csv("bootstrap_motion", header, rows)
    w.json("bootstrap_motion_meta", {
        "note": ("section 17: the per-supernode displacement the multiview "
                 "gradients requested BEFORE any rigid projection. Rotation "
                 "and omega columns are blank because the bootstrap estimates "
                 "a per-Gaussian position field, not a per-supernode "
                 "orientation; the rigid stage is where orientation appears "
                 "(section 13's T_{k,t})."),
        **(dbg.motion_info or {})})


# ---------------------------------------------------------------------------
# sections 18/19
# ---------------------------------------------------------------------------
def _supernode_debug(dbg, w) -> None:
    if dbg.supernode_id is None or dbg.mu0 is None:
        w.csv("supernode_motion", ["supernode"], [])
        return
    sid = dbg.supernode_id.long().clamp_min(0)
    C = dbg.num_supernodes
    cnt = torch.zeros(C, device=sid.device).index_add_(
        0, sid, torch.ones(sid.shape[0], device=sid.device))
    c0 = torch.zeros((C, 3), device=sid.device).index_add_(
        0, sid, dbg.mu0) / cnt.clamp_min(1).unsqueeze(1)
    last = dbg.positions_at("refined", dbg.num_frames - 1)
    if last is None:
        last = dbg.positions_at("rigid", dbg.num_frames - 1)
    if last is None:
        last = dbg.positions[-1] if dbg.positions else dbg.mu0
    cT = torch.zeros((C, 3), device=sid.device).index_add_(
        0, sid, last) / cnt.clamp_min(1).unsqueeze(1)

    costs = dbg.assignment_costs
    K = dbg.K
    header = ["supernode", "cx0", "cy0", "cz0", "cxT", "cyT", "czT", "label",
              "member_count", "motion_evidence", "static_cost"]
    header += [f"part{k}_cost" for k in range(1, K + 1)]
    header += ["best_cost", "second_best_cost", "assignment_margin"]
    rows = []
    for c in range(C):
        lab = int(dbg.node_labels[c]) if dbg.node_labels is not None else ""
        ev = (float(dbg.motion_statistic[c])
              if dbg.motion_statistic is not None
              and c < dbg.motion_statistic.numel() else "")
        row = [c, float(c0[c, 0]), float(c0[c, 1]), float(c0[c, 2]),
               float(cT[c, 0]), float(cT[c, 1]), float(cT[c, 2]),
               lab, int(cnt[c]), ev]
        if costs is not None and c < costs.shape[0]:
            vals = [float(x) for x in costs[c]]
            row += vals
            s = sorted(vals)
            row += [s[0], s[1] if len(s) > 1 else "",
                    (s[1] - s[0]) if len(s) > 1 else ""]
        else:
            row += [""] * (K + 1 + 3)
        rows.append(row)
    w.csv("supernode_motion", header, rows)

    if costs is not None:
        w.tensor("part_assignment_costs", costs.cpu())
        chosen = costs.argmin(dim=1)
        srt = costs.sort(dim=1).values
        w.csv("part_assignment_costs",
              ["supernode", "static_cost"]
              + [f"part{k}_cost" for k in range(1, K + 1)]
              + ["chosen_label", "best_cost", "second_best_cost", "margin"],
              [[c] + [float(x) for x in costs[c]] + [int(chosen[c]),
               float(srt[c, 0]),
               float(srt[c, 1]) if costs.shape[1] > 1 else "",
               float(srt[c, 1] - srt[c, 0]) if costs.shape[1] > 1 else ""]
               for c in range(costs.shape[0])])


# ---------------------------------------------------------------------------
# section 20
# ---------------------------------------------------------------------------
def _support_iterations(dbg, w) -> None:
    header = ["iteration", "total_energy", "relative_change", "max_se3_change",
              "label_jaccard", "label_move_accepted", "num_static"]
    header += [f"count_part{k}" for k in range(1, dbg.K + 1)]
    rows = []
    for h in dbg.rigid_history:
        counts = h.get("counts", [])
        rows.append([h.get("outer", ""), h.get("energy", ""),
                     h.get("relative_change", ""), h.get("max_se3_step", ""),
                     h.get("label_jaccard", ""),
                     h.get("label_move_accepted", ""),
                     counts[0] if counts else ""]
                    + [counts[k] if k < len(counts) else ""
                       for k in range(1, dbg.K + 1)])
    w.csv("support_iterations", header, rows)


# ---------------------------------------------------------------------------
# section 21
# ---------------------------------------------------------------------------
def _rigid_transform_history(dbg, w) -> None:
    if dbg.transforms is None:
        w.json("rigid_transform_history", {"available": False,
                                           "stage": "phase 2 did not run"})
        return
    T, K = dbg.transforms.num_frames, dbg.transforms.K
    per = []
    for k in range(K):
        prev = None
        for t in range(T):
            R = dbg.transforms.R[t, k]
            tr = dbg.transforms.t[t, k]
            d = None
            if prev is not None:
                dth = float(so3_log(project_to_so3(
                    (prev[0].T @ R).unsqueeze(0))).norm())
                d = {"d_theta_rad": dth,
                     "d_translation": float((tr - prev[1]).norm())}
            per.append({"part": k + 1,
                        "frame": dbg.frame_ids[t] if t < len(dbg.frame_ids) else t,
                        "R": R.detach().cpu().tolist(),
                        "quaternion_wxyz": _q(R),
                        "t": tr.detach().cpu().tolist(),
                        "se3_change_from_previous_frame": d})
            prev = (R, tr)
    w.json("rigid_transform_history", {
        "stage": "section 25 Phase 2 (rigid parts), final iterate",
        "shape": [K, T, 4, 4],
        "converged": dbg.rigid_converged,
        "stop_reason": dbg.rigid_stop_reason,
        "outer_iterations": dbg.rigid_history,
        "transforms": per})
    M = torch.zeros((K, T, 4, 4), dtype=torch.float64)
    M[:, :, 3, 3] = 1.0
    for k in range(K):
        for t in range(T):
            M[k, t, :3, :3] = dbg.transforms.R[t, k].cpu()
            M[k, t, :3, 3] = dbg.transforms.t[t, k].cpu()
    w.tensor("rigid_transforms", M)


# ---------------------------------------------------------------------------
# section 22
# ---------------------------------------------------------------------------
def _rigid_residuals(dbg, w) -> None:
    if dbg.rigid_residual is None:
        w.csv("rigid_residuals", ["frame"], [])
        return
    w.tensor("rigid_residuals", dbg.rigid_residual.cpu())
    rows = []
    header = ["scope", "key", "mean", "median", "p90", "p95", "max"]
    s = _stats(dbg.rigid_residual)
    rows.append(["all", "", s["mean"], s["median"], s["p90"], s["p95"], s["max"]])
    if dbg.labels is not None:
        for k in range(0, dbg.K + 1):
            m = dbg.labels == k
            if bool(m.any()):
                s = _stats(dbg.rigid_residual[m])
                rows.append(["part", k, s["mean"], s["median"], s["p90"],
                             s["p95"], s["max"]])
    # per frame
    if dbg.positions and dbg.transforms is not None and dbg.labels is not None:
        for t in range(min(dbg.num_frames, len(dbg.positions))):
            pred = dbg.positions_at("rigid", t)
            if pred is None:
                break
            r = (dbg.positions[t].double() - pred.double()).norm(dim=1)
            s = _stats(r[dbg.labels > 0])
            rows.append(["frame",
                         dbg.frame_ids[t] if t < len(dbg.frame_ids) else t,
                         s["mean"], s["median"], s["p90"], s["p95"], s["max"]])
    w.csv("rigid_residuals", header, rows)


# ---------------------------------------------------------------------------
# sections 23/24
# ---------------------------------------------------------------------------
def _rigid_invariants(dbg, w) -> None:
    header = ["frame", "part", "pairwise_mean_abs", "pairwise_rms",
              "pairwise_p95", "pairwise_max", "inertia_eig0", "inertia_eig1",
              "inertia_eig2", "inertia_rel_drift", "inertia_frobenius_vs_rotated"]
    rows = []
    if dbg.labels is None or not dbg.positions or dbg.mu0 is None:
        w.csv("rigid_invariants", header, rows)
        return
    g = torch.Generator(device="cpu").manual_seed(0)
    for k in range(1, dbg.K + 1):
        m = dbg.labels == k
        n = int(m.sum())
        if n < 3:
            continue
        x0 = dbg.mu0[m].double()
        npair = min(4096, n * (n - 1) // 2)
        i = torch.randint(0, n, (npair,), generator=g).to(x0.device)
        j = torch.randint(0, n, (npair,), generator=g).to(x0.device)
        keep = i != j
        i, j = i[keep], j[keep]
        d0 = (x0[i] - x0[j]).norm(dim=1)
        c0 = x0.mean(0)
        J0 = ((x0 - c0).T @ (x0 - c0))
        e0 = torch.linalg.eigvalsh(J0)
        for t in range(min(dbg.num_frames, len(dbg.positions))):
            xt = dbg.positions[t][m].double()
            dt_ = (xt[i] - xt[j]).norm(dim=1)
            dd = (dt_ - d0).abs()
            ct = xt.mean(0)
            Jt = ((xt - ct).T @ (xt - ct))
            et = torch.linalg.eigvalsh(Jt)
            fro = ""
            if dbg.transforms is not None:
                R = dbg.transforms.R[t, k - 1].double()
                fro = float((Jt - R @ J0 @ R.T).norm())
            rows.append([
                dbg.frame_ids[t] if t < len(dbg.frame_ids) else t, k,
                float(dd.mean()), float((dd ** 2).mean().sqrt()),
                float(dd.quantile(0.95)), float(dd.max()),
                float(et[0]), float(et[1]), float(et[2]),
                float(((et - e0).abs() / e0.clamp_min(EPS)).max()), fro])
    w.csv("rigid_invariants", header, rows)


# ---------------------------------------------------------------------------
# sections 25/26/27/28
# ---------------------------------------------------------------------------
def _geometric_fit_summary(dbg, w) -> None:
    summary, pris, revo, flat = [], [], [], []
    for k in range(1, dbg.K + 1):
        j = dbg.joints[k - 1] if k - 1 < len(dbg.joints) else None
        if j is None:
            summary.append({"part": k, "available": False,
                            "stage": "section 25 Phase 3 produced no fit"})
            continue
        summary.append({
            "part": k, "selected": j.get("selected"), "status": j.get("status"),
            "normalized_error": j.get("normalized_error"),
            "both_converged": j.get("both_converged"),
            "length_scale": j.get("length_scale"),
            "rigid_rmse_median": j.get("rigid_rmse_median"),
            "transform_precision": j.get("transform_precision"),
            "decision_inputs": j.get("decision_inputs")})
        # Sections 25 and 26 both say "write this even when it is not selected".
        p = j.get("prismatic") or {}
        pris.append({"part": k, "selected": j.get("selected") == "prismatic",
                     **p})
        r = j.get("revolute") or {}
        revo.append({"part": k, "selected": j.get("selected") == "revolute",
                     **r})
        conv = r.get("convergence") or {}
        if conv.get("degenerate_flat_direction"):
            flat.append({
                "part": k,
                "final_objective_change": conv.get("relative_error_change"),
                "final_axis_and_point_change": conv.get("final_d_eta"),
                "final_q_change": conv.get("final_d_q"),
                "interpretation": (
                    "section 28: the objective is stationary while the axis "
                    "and q still move. Moving the axis out and shrinking q by "
                    "the matching factor describes nearly the same motion, so "
                    "this is a physical ambiguity (the prismatic limit), not "
                    "an optimizer that stopped early.")})
    w.json("geometric_fit_summary", {"per_part": summary,
                                     "stage": "section 25 Phase 3"})
    w.json("prismatic_fit_history", {"note": "section 25: written whether or "
                                             "not P was selected",
                                     "per_part": pris})
    w.json("revolute_fit_history", {"note": "section 26: written whether or "
                                            "not R was selected",
                                    "per_part": revo})
    w.json("degenerate_flat_direction", {
        "note": ("section 28: only populated when the revolute fit converged "
                 "on a stationary objective with unsettled parameters"),
        "cases": flat})


def _axis_history(dbg, w) -> None:
    """Section 27: axis stability across the revolute fit's iterations."""
    per = []
    for k in range(1, dbg.K + 1):
        j = dbg.joints[k - 1] if k - 1 < len(dbg.joints) else None
        if j is None:
            continue
        r = j.get("revolute") or {}
        hist = (r.get("axis_iterates") or [])
        steps = []
        for i in range(1, len(hist)):
            a = torch.tensor(hist[i - 1]["axis"], dtype=torch.float64)
            b = torch.tensor(hist[i]["axis"], dtype=torch.float64)
            cos = float((a @ b).clamp(-1.0, 1.0))
            steps.append({"iteration": i,
                          "delta_angle_rad": math.acos(abs(cos)),
                          "axis_point_shift": float(
                              (torch.tensor(hist[i]["axis_point"],
                                            dtype=torch.float64)
                               - torch.tensor(hist[i - 1]["axis_point"],
                                              dtype=torch.float64)).norm())})
        per.append({"part": k, "axis": r.get("axis"),
                    "axis_point": r.get("axis_point"),
                    "iterates": hist, "stability": steps,
                    "gauge_e_dot_a": (r.get("diagnostics") or {}).get(
                        "gauge_e_dot_a")})
    w.json("joint_axis_history", {"note": "section 27", "per_part": per})


# ---------------------------------------------------------------------------
# section 29
# ---------------------------------------------------------------------------
def _q_trajectories(dbg, w) -> None:
    header = ["frame", "timestamp", "part", "q_geometric",
              "q_renderer_refined", "q_filter_predicted", "q_filter_posterior",
              "qdot_filter_predicted", "qdot_filter_posterior",
              "visual_delta_q"]
    rows = []
    # `run_reduced_filter` names its per-frame list "frames"; accept both so
    # the mapping survives either shape.
    red = ((dbg.reduced or {}).get("per_frame")
           or (dbg.reduced or {}).get("frames") or [])
    for k in range(1, dbg.K + 1):
        j = dbg.joints[k - 1] if k - 1 < len(dbg.joints) else None
        q_geo = (j or {}).get("q") or []
        q_ref = dbg.q_traj[k - 1] if k - 1 < len(dbg.q_traj) else []
        for t in range(dbg.num_frames):
            fid = dbg.frame_ids[t] if t < len(dbg.frame_ids) else t
            rf = red[t] if t < len(red) else {}
            rows.append([
                fid, t * dbg.dt, k,
                q_geo[t] if t < len(q_geo) else "",
                q_ref[t] if t < len(q_ref) else "",
                _get(rf, "q_pred", k), _get(rf, "q_post", k),
                _get(rf, "qdot_pred", k), _get(rf, "qdot_post", k),
                _get(rf, "q_visual_delta", k)])
    w.csv("q_trajectories", header, rows)


def _get(row, key, k):
    v = row.get(key)
    if v is None:
        return ""
    if isinstance(v, (list, tuple)):
        return v[k - 1] if k - 1 < len(v) else ""
    return v


# ---------------------------------------------------------------------------
# sections 30/31
# ---------------------------------------------------------------------------
def _renderer_refinement(dbg, w) -> None:
    info = dbg.refine_info or {}
    rows = []
    for it in (info.get("trajectory_iterations") or []):
        rows.append([it.get("iter", ""), it.get("energy", ""),
                     it.get("decrease", ""), it.get("lambda", ""),
                     it.get("step_norm", ""), it.get("grad_norm", ""),
                     it.get("accepted", "")])
    w.csv("renderer_refinement",
          ["iteration", "total_energy", "decrease", "lm_lambda", "q_step_norm",
           "normal_equation_residual", "accepted"], rows)
    w.json("renderer_refinement_summary", {
        "stage": "section 25 Phase 4 / section 23",
        "energy_before": info.get("energy_before"),
        "energy_after": info.get("energy_after"),
        "per_frame_delta_q": info.get("per_frame_delta_q"),
        "status": info.get("status"),
        "meaning": ("section 30: whether image supervision really corrects "
                    "the geometric estimate")})


def _reduced_filter(dbg, w) -> None:
    red = dbg.reduced or {}
    header = ["frame", "part", "q_pred", "q_post", "qdot_pred", "qdot_post",
              "innovation", "innovation_variance", "joint_information",
              "visual_delta_q", "posterior_variance"]
    rows = []
    for t, r in enumerate(red.get("per_frame") or red.get("frames") or []):
        for k in range(1, dbg.K + 1):
            rows.append([r.get("frame", t), k,
                         _get(r, "q_pred", k), _get(r, "q_post", k),
                         _get(r, "qdot_pred", k), _get(r, "qdot_post", k),
                         _get(r, "q_visual_delta", k),
                         _get(r, "measurement_energy", k),
                         _get(r, "joint_information", k),
                         _get(r, "q_visual_delta", k),
                         _get(r, "posterior_variance_q", k)])
    w.csv("reduced_filter", header, rows)
    w.json("reduced_filter_summary", {
        "stage": "section 25 Phase 5",
        **{k: v for k, v in red.items() if k not in ("per_frame", "frames")},
        "meaning": ("section 31: whether the filter is measurement-driven or "
                    "process-prior-driven")})


def _handover(dbg, w) -> None:
    ho = dbg.handover or {}
    payload = {"stage": "section 24 / section 37", **ho,
               "requirement": "the canonical hash must remain identical"}
    if not ho.get("canonical_hash_before"):
        # Explicit rather than an empty record: the handover runs only after a
        # joint has been selected, so on an unresolved run there is nothing to
        # hand over and no hash to compare. Saying so is different from saying
        # the hash check passed.
        payload["performed"] = False
        payload["why_not"] = (
            f"the run ended as {dbg.status!r}; section 24's handover runs only "
            f"after a joint type has been selected, so no canonical hash was "
            f"taken")
    w.json("canonical_handover", payload)


def _convergence_summary(dbg, w) -> None:
    per = []
    for k in range(1, dbg.K + 1):
        j = dbg.joints[k - 1] if k - 1 < len(dbg.joints) else None
        if j is None:
            continue
        per.append({
            "part": k,
            "prismatic": (j.get("prismatic") or {}).get("convergence"),
            "revolute": (j.get("revolute") or {}).get("convergence"),
            "both_converged": j.get("both_converged")})
    w.json("convergence_summary", {
        "rigid_stage": {"converged": dbg.rigid_converged,
                        "stop_reason": dbg.rigid_stop_reason,
                        "criteria": "section 28 rigid-part stage"},
        "joint_stage": {"per_part": per,
                        "criteria": "section 28 joint-fit stage"},
        "renderer_refinement": (dbg.refine_info or {}).get("status"),
        "observability": dbg.observability})


# ---------------------------------------------------------------------------
# section 38
# ---------------------------------------------------------------------------
def _failure_summary(dbg, w, energies) -> list:
    """Section 38: identify conditions. Never state speculation as fact."""
    f = []

    def add(observed, evidence, interpretation):
        f.append({"Observed": observed, "Evidence": evidence,
                  "Possible interpretation": interpretation})

    if dbg.transforms is not None and not dbg.rigid_converged:
        add("The rigid-part stage did not meet section 28's criteria.",
            f"stop_reason = {dbg.rigid_stop_reason!r}",
            "The outer-iteration budget may be too small, or labels and "
            "transforms may be cycling. Compare support_iterations.csv.")

    jac = [h.get("label_jaccard") for h in dbg.rigid_history
           if h.get("label_jaccard") is not None]
    if jac and min(jac) < 0.9:
        add("Large label churn between outer iterations.",
            f"minimum support Jaccard {min(jac):.4f} over "
            f"{len(jac)} iterations",
            "Supernodes are moving between parts; the assignment margin in "
            "part_assignment_costs.csv shows which ones are near-ties.")

    tot_br = sum((energies["rigid"][t] or {}).get("total", 0.0)
                 - (energies["bootstrap"][t] or {}).get("total", 0.0)
                 for t in range(min(len(energies["rigid"]),
                                    len(energies["bootstrap"])))
                 if energies["rigid"][t] and energies["bootstrap"][t])
    base = sum((energies["bootstrap"][t] or {}).get("total", 0.0)
               for t in range(len(energies["bootstrap"]))
               if energies["bootstrap"][t]) or 1.0
    if tot_br > 0.25 * abs(base):
        add("Constraining the free motion to K rigid parts cost a large "
            "fraction of the explained energy.",
            f"section 35 bootstrap->rigid loss {tot_br:.1f} against a "
            f"bootstrap energy of {base:.1f}",
            "The K-part decomposition may be wrong before any joint is "
            "fitted, or K may not match the scene.")

    tot_rj = sum((energies["refined"][t] or {}).get("total", 0.0)
                 - (energies["rigid"][t] or {}).get("total", 0.0)
                 for t in range(min(len(energies["refined"]),
                                    len(energies["rigid"])))
                 if energies["refined"][t] and energies["rigid"][t])
    if tot_rj > 0.25 * abs(base):
        add("Constraining the rigid trajectory to a one-parameter joint cost "
            "a large fraction of the explained energy.",
            f"section 36 rigid->joint loss {tot_rj:.1f}",
            "The recovered rigid motion may not lie on either the prismatic "
            "or the revolute manifold.")

    if dbg.rigid_residual is not None and dbg.labels is not None:
        mv = dbg.labels > 0
        if bool(mv.any()):
            s = _stats(dbg.rigid_residual[mv])
            if s["p95"] > 0.5 * s["median"] * 4:
                add("The rigid residual has a heavy tail on the moving set.",
                    f"median {s['median']:.5f}, p95 {s['p95']:.5f}, "
                    f"max {s['max']:.5f}",
                    "Some members may not follow their part's rigid motion; "
                    "section 27 splitting is the mechanism for that.")

    for k in range(1, dbg.K + 1):
        j = dbg.joints[k - 1] if k - 1 < len(dbg.joints) else None
        if j is None:
            continue
        ne = j.get("normalized_error") or {}
        if j.get("status") == "UNRESOLVED_GEOMETRY":
            add(f"Part {k}: the prismatic and revolute fits are "
                f"geometrically indistinguishable.",
                f"E_P {ne.get('prismatic'):.3e}, E_R {ne.get('revolute'):.3e}, "
                f"gap {ne.get('absolute_gap'):.3e} below the floor "
                f"{ne.get('indistinguishable_floor'):.3e}",
                "The motion may genuinely lie in the shared limit of both "
                "models, or the transforms may be too imprecise to separate "
                "them. The floor is set by the measured transform precision.")
        if not j.get("both_converged", True):
            add(f"Part {k}: a joint fit did not converge.",
                f"prismatic {(j.get('prismatic') or {}).get('converged')}, "
                f"revolute {(j.get('revolute') or {}).get('converged')}",
                "Section 28 forbids comparing unconverged fits, so no type is "
                "reported even if one error is smaller.")
        conv = (j.get("revolute") or {}).get("convergence") or {}
        if conv.get("degenerate_flat_direction"):
            add(f"Part {k}: the revolute fit converged along a flat direction.",
                f"final axis/point change {conv.get('final_d_eta')}, "
                f"final q change {conv.get('final_d_q')}",
                "Axis distance and q trade off against each other here, which "
                "is the prismatic limit; see degenerate_flat_direction.json.")

    info = dbg.refine_info or {}
    if (info.get("energy_after") is not None
            and info.get("energy_before") is not None
            and info["energy_after"] >= info["energy_before"] - 1e-9):
        add("Renderer refinement did not decrease the objective.",
            f"E {info['energy_before']:.2f} -> {info['energy_after']:.2f}",
            "The geometric estimate may already be at a local optimum of the "
            "image objective, or the joint model may not be able to express "
            "the remaining residual.")

    red = dbg.reduced or {}
    if red.get("mean_visual_correction") is not None and \
            float(red["mean_visual_correction"]) < 1e-3:
        add("The reduced filter's visual correction is nearly zero.",
            f"mean visual correction {float(red['mean_visual_correction']):.2e}",
            "The prior rather than the image may be producing the trajectory; "
            "reduced_filter.csv separates predicted from posterior.")

    if dbg.multiview:
        dqs = [r.get("delta_q") for r in dbg.multiview
               if r.get("delta_q") is not None]
        if dqs and max(dqs) * min(dqs) < 0:
            add("Cameras disagree on the sign of the joint step.",
                f"per-camera dq ranges from {min(dqs):.3e} to {max(dqs):.3e}",
                "One or more cameras may see a different part, be "
                "mis-calibrated, or carry almost no information; "
                "camera_metrics.csv has their Fisher traces.")

    if not f:
        f.append({"Observed": "No configured failure condition was triggered.",
                  "Evidence": "section 38's checklist was evaluated against "
                              "this run's artifacts",
                  "Possible interpretation": "This is not a statement that the "
                                             "result is correct, only that "
                                             "none of the listed conditions "
                                             "was observed."})

    w.json("failure_summary", {"note": ("section 38: observations and "
                                        "evidence; interpretations are "
                                        "explicitly marked as possible, not "
                                        "asserted"),
                               "conditions": f})
    md = ["# Automatic failure summary", "",
          "Section 38 of the debug addendum. Interpretations are **possible**",
          "explanations, not findings.", ""]
    for i, c in enumerate(f, 1):
        md += [f"## {i}. {c['Observed']}", "",
               f"**Evidence:** {c['Evidence']}", "",
               f"**Possible interpretation:** {c['Possible interpretation']}",
               ""]
    w.text("failure_summary.md", "\n".join(md))
    return f
