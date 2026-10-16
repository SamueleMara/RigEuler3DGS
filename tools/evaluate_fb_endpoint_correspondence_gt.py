#!/usr/bin/env python
"""EVALUATION ONLY: score a sealed endpoint-correspondence diagnosis against the
dataset's ground truth, in a separate process, after the fact.

    python tools/evaluate_fb_endpoint_correspondence_gt.py \
        --diagnosis-dir '<results>/<scene>/endpoint_correspondence/<tag>' \
        --dataset-dir   '<dataset>/<scene>' [--dry-run] [--resume]

One-way data flow. The diagnosis (tools/analyze_fb_endpoint_correspondence.py)
runs GT-free behind its firewall and seals its outputs (run_state.json
complete: true). This tool reads those frozen outputs plus the dataset GT and
writes ONLY under <diagnosis>/evaluation_gt/. It hashes every primary file
before and after and fails if any changed. No GT number is ever written into a
primary table, and nothing here can change a J0/J1 label, patch, rank or
threshold: the diagnosis never reads evaluation_gt/.

Exit: 0 evaluated, 3 GT unavailable/incomplete (statuses written), 2 refused
(diagnosis not sealed, unsafe output path, fingerprint conflict), 1 failed.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
import time
import traceback
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from observation_diagnostics import gt_evaluation as G  # noqa: E402

EVAL_DIR = "evaluation_gt"
OWN = {"B_ONLY_AT_T": ("B", "T", "BT", "FT"), "F_ONLY_AT_0": ("F", 0, "F0", "B0"),
       "B_ONLY_AT_0": ("B", 0, "B0", "F0"), "F_ONLY_AT_T": ("F", "T", "FT", "BT")}
DATA_DICTIONARY = {
    "sensor_observation_metric": "residuals and categories against the RECORDED RGB-D frames (rgb/, "
                                 "depth_npy/); legal for inference; produced by the diagnosis. In older "
                                 "outputs a valid recorded depth pixel was called 'gt_valid' -- it is "
                                 "observed_depth_valid, not ground truth.",
    "inferred_joint_metric": "quantities of the ESTIMATED joints/poses saved by the rigid phase; legal as "
                             "evidence; produced by the diagnosis",
    "gt_evaluation_metric": "dataset ground truth (joint annotations, part-labelled surface samples, "
                            "object masks) scored against frozen predictions; EVALUATION ONLY",
    "oracle_metric": "a GT metric that uses GT to resolve a convention (here only the sign of an "
                     "unoriented axis); EVALUATION ONLY, never a selection criterion"}


def code_hash() -> str:
    h = hashlib.sha256()
    for p in (Path(__file__), REPO / "observation_diagnostics" / "gt_evaluation.py"):
        h.update(p.read_bytes())
    return h.hexdigest()


def primary_manifest(diag: Path) -> dict:
    """sha256 of every primary output (everything except evaluation_gt/)."""
    out = {}
    for p in sorted(diag.rglob("*")):
        if p.is_file() and EVAL_DIR not in p.relative_to(diag).parts:
            out[str(p.relative_to(diag))] = G.sha256(p)
    return out


def write_csv(path: Path, rows: list, header: list):
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=header, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow({k: (json.dumps(v) if isinstance(v, (list, dict)) else ("" if v is None else v))
                        for k, v in r.items()})


def fmt(x, d=4):
    return "n/a" if x is None or (isinstance(x, float) and x != x) else (f"{x:.{d}f}" if isinstance(x, float) else str(x))


# ---------------------------------------------------------------------------

class Frozen:
    """The sealed diagnosis outputs plus the inputs they name (read only)."""

    def __init__(self, diag: Path, results_dir: Path):
        self.diag, self.res = diag, results_dir
        self.state = json.loads((diag / "run_state.json").read_text())
        self.prov = json.loads((diag / "provenance.json").read_text())
        self.summary = json.loads((diag / "endpoint_summary.json").read_text())
        self.jprov = json.loads((diag / "joint_estimate_provenance.json").read_text())
        self.T = int(self.summary["endpoints"][1])
        from plyfile import PlyData
        self.xyz = {}
        for k in ("F", "B"):
            v = PlyData.read(self.prov["sources"][k]["ply"])["vertex"]
            self.xyz[k] = np.stack([v["x"], v["y"], v["z"]], 1).astype(np.float64)
        bidi = results_dir / "bidirectional"

        def centres(d, f):
            return np.load(bidi / d / "phase1_centers" / f"centers_frame_{f:04d}.npy").astype(np.float64)
        self.pos = {"F0": self.xyz["F"], "BT": self.xyz["B"], "B0": centres("backward", 0),
                    "FT": centres("forward", self.T)}
        self.bidi = bidi

    def patches(self):
        out = []
        for name in ("backward_only_observed_patches.csv", "forward_only_observed_patches.csv"):
            p = self.diag / name
            if not p.is_file() or p.stat().st_size == 0:
                continue
            for r in csv.DictReader(open(p)):
                cj = json.loads((self.diag / "candidates" / f"{r['patch_id']}.json").read_text())
                rows = cj.get("rows_list") if isinstance(cj.get("rows_list"), list) else (
                    cj.get("rows") if isinstance(cj.get("rows"), list) else None)
                out.append({"patch_id": r["patch_id"], "type": r["type"],
                            "J0_label": r.get("J0_label") or "", "screen_J0_label": r.get("screen_J0_label") or "",
                            "J1": {d: x.get("label") for d, x in (cj.get("J1") or {}).items()},
                            "rows": None if rows is None else np.asarray(rows, dtype=np.int64),
                            "detailed": bool(r.get("J0_label"))})
        return out

    def statics(self):
        out = []
        p = self.diag / "static_common_candidates.csv"
        if p.is_file() and p.stat().st_size:
            for r in csv.DictReader(open(p)):
                cj = json.loads((self.diag / "candidates" / f"{r['patch_id']}.json").read_text())
                out.append({"patch_id": r["patch_id"], "tier": r.get("tier") or r.get("J0"),
                            "F_rows": np.asarray(cj["F_rows"]), "B_rows": np.asarray(cj["B_rows"]),
                            "best_B_for_F": np.asarray(cj["best_B_for_F"])})
        return out


class Projector:
    """Pinhole projection from the scene's COLMAP calibration (sensor calibration)."""

    def __init__(self, sparse: Path):
        from scene.colmap_loader import qvec2rotmat, read_extrinsics_binary, read_intrinsics_binary
        ex, intr = read_extrinsics_binary(str(sparse / "images.bin")), read_intrinsics_binary(str(sparse / "cameras.bin"))
        self.cams = {}
        for e in ex.values():
            c = intr[e.camera_id]
            fx, fy, cx, cy = (c.params[0], c.params[1], c.params[2], c.params[3]) if len(c.params) >= 4 else \
                (c.params[0], c.params[0], c.params[1], c.params[2])
            self.cams[e.name.split("/")[0]] = (qvec2rotmat(e.qvec), np.asarray(e.tvec), fx, fy, cx, cy, c.width, c.height)

    def inside_mask(self, xyz, mask_dir: Path, frame: int):
        """Per row: inside the GT object mask of at least one camera where it is in view."""
        from PIL import Image
        hit = np.zeros(len(xyz), bool)
        inview = np.zeros(len(xyz), bool)
        for name, (R, t, fx, fy, cx, cy, W, H) in self.cams.items():
            p = mask_dir / name / f"frame_{frame:06d}.png"
            if not p.is_file():
                continue
            m = np.asarray(Image.open(p)) > 127
            Xc = xyz @ R.T + t
            z = Xc[:, 2]
            u = fx * Xc[:, 0] / np.maximum(z, 1e-9) + cx
            v = fy * Xc[:, 1] / np.maximum(z, 1e-9) + cy
            ok = (z > 0.01) & (u >= 0) & (u < W) & (v >= 0) & (v < H)
            inview |= ok
            ui, vi = np.clip(u.astype(int), 0, W - 1), np.clip(v.astype(int), 0, H - 1)
            hit |= ok & m[vi, ui]
        return hit, inview


# ---------------------------------------------------------------------------

def joint_metrics(F: Frozen, gt: G.GroundTruth) -> list:
    rows = []
    T = F.T
    gt_ok = gt.joint is not None and gt.q is not None
    for d, est in F.jprov.get("estimates", {}).items():
        anchor = 0 if d == "forward" else T
        status = est.get("status")
        decided = est.get("decided_type") or ""
        base = {"direction": d, "estimate_status": status,
                "decided_type": "UNDECIDED" if decided.startswith("NONE") else decided,
                "gt_type": gt.joint["type"] if gt.joint else None,
                "gt_axis": gt.joint["axis"].round(6).tolist() if gt.joint else None,
                "metric_class": "gt_evaluation_metric",
                "degenerate_estimate": status != "USABLE"}
        files = est.get("files") or {}
        for hyp in ("revolute", "prismatic"):
            r = dict(base, hypothesis=hyp)
            if not gt_ok:
                r["status"] = "GT_UNAVAILABLE"
                rows.append(r)
                continue
            if hyp not in files:
                r["status"] = "ESTIMATE_UNAVAILABLE"
                rows.append(r)
                continue
            g = json.loads(Path(files[hyp]).read_text())
            gr = g.get("refined", g)
            r["hypothesis_matches_gt_type"] = (hyp == gt.joint["type"])
            if hyp == "revolute":
                a, p = np.asarray(gr.get("axis", g.get("axis")), float), np.asarray(gr.get("axis_point", g.get("axis_point")), float)
                coord = np.asarray(gr.get("theta", g.get("theta")), float)
                r["axis_line_angle_deg"] = G.line_angle_deg(a, gt.joint["axis"]) if gt.joint["type"] == "revolute" else None
                r["axis_direction_angle_deg_vs_gt_axis"] = G.line_angle_deg(a, gt.joint["axis"])
                r["axis_line_distance_m"] = (G.line_distance(p, a, gt.joint["point"], gt.joint["axis"])
                                             if gt.joint["type"] == "revolute" else None)
                units = "rad"
            else:
                a = np.asarray(gr.get("direction", g.get("direction")), float)
                coord = np.asarray(gr.get("s", g.get("s")), float)
                r["axis_direction_angle_deg_vs_gt_axis"] = G.line_angle_deg(a, gt.joint["axis"])
                r["axis_line_angle_deg"] = r["axis_direction_angle_deg_vs_gt_axis"] if gt.joint["type"] == "prismatic" else None
                r["axis_line_distance_m"] = None
                units = "m"
            # the estimator's own sign convention, from ESTIMATED data only: which of
            # (+/-) d coord about/along its axis reproduces its own Phase-2 pose track
            est_sign, conv_res = 1.0, None
            pf_ = files.get("poses")
            if pf_ and Path(pf_).is_file() and coord.size:
                Pp = np.load(pf_)
                res_ = {}
                for sg in (1.0, -1.0):
                    if hyp == "revolute":
                        res_[sg] = max(G.geodesic_deg(Pp[t][:3, :3], G.rodrigues(a, sg * (coord[t] - coord[anchor])))
                                       for t in range(min(len(coord), Pp.shape[0])))
                    else:
                        res_[sg] = max(float(np.linalg.norm(Pp[t][:3, 3] - G.unit(a) * sg * (coord[t] - coord[anchor])))
                                       for t in range(min(len(coord), Pp.shape[0])))
                est_sign = min(res_, key=res_.get)
                conv_res = res_[est_sign]
            r["estimator_sign_convention"] = est_sign
            r["estimator_convention_residual"] = conv_res
            r["estimator_convention_residual_units"] = "deg (max rotation)" if hyp == "revolute" else "m (max translation)"
            if hyp == gt.joint["type"] and coord.size == gt.q.size:
                sgn = float(np.sign(G.unit(a) @ gt.joint["axis"]) or 1.0)
                est_rel = (coord - coord[anchor]) * est_sign * sgn
                gt_rel = (gt.q - gt.q[anchor]) * gt.sign
                err = est_rel - gt_rel
                r |= {"coord_units": units, "coord_mae": float(np.abs(err).mean()),
                      "coord_rmse": float(np.sqrt((err ** 2).mean())), "coord_max_abs": float(np.abs(err).max()),
                      "gt_coord_range": float(np.ptp(gt.q)), "est_coord_range": float(np.ptp(coord)),
                      "coord_alignment": ("both referenced to the direction's anchor frame (known reference, "
                                          "no fitted offset or scale); the estimator's own sign convention is "
                                          "taken from its pose track; the axis sign from the estimated axis "
                                          "orientation relative to the GT axis"),
                      "coord_metric_class": "oracle_metric (GT axis resolves the unoriented sign)"}
                if units == "rad":
                    r["coord_mae_deg"] = float(np.degrees(r["coord_mae"]))
                    r["coord_rmse_deg"] = float(np.degrees(r["coord_rmse"]))
            else:
                r["coord_note"] = ("hypothesis type differs from the GT type: coordinate trajectory not "
                                   "comparable; scored geometry only")
            r["status"] = "OK"
            rows.append(r)
        # the Phase-2 rigid track itself (type-agnostic)
        r = dict(base, hypothesis="phase2_rigid_track")
        pf = files.get("poses")
        if gt_ok and pf and Path(pf).is_file():
            P = np.load(pf)
            xT, pT = gt.cloud.get("T", (None, None))
            if pT is not None and (pT == 1).any():
                mov_T = xT[pT == 1]
                mov_anchor = mov_T if anchor == T else G.apply(gt.transform(T, anchor), mov_T)
                rot, pte = [], []
                for t in range(P.shape[0]):
                    Gt = gt.transform(anchor, t)
                    rot.append(G.geodesic_deg(P[t][:3, :3], Gt[:3, :3]))
                    pte.append(float(np.linalg.norm(G.apply(P[t], mov_anchor) - G.apply(Gt, mov_anchor), axis=1).mean()))
                gt_disp = float(np.linalg.norm(G.apply(gt.transform(anchor, 0 if anchor == T else T), mov_anchor)
                                               - mov_anchor, axis=1).mean())
                r |= {"pose_rotation_err_deg_mean": float(np.mean(rot)), "pose_rotation_err_deg_max": float(np.max(rot)),
                      "point_transfer_err_m_mean": float(np.mean(pte)), "point_transfer_err_m_max": float(np.max(pte)),
                      "gt_moving_part_mean_displacement_m": gt_disp,
                      "pose_metric_definition": ("anchor-referenced: estimated T_part(t) vs GT moving-link motion "
                                                 "from the same anchor; rotation = SO(3) geodesic; translation = "
                                                 "mean displacement error of the GT moving-link samples (no "
                                                 "gauge alignment: both are in render_world)"),
                      "status": "OK"}
            else:
                r["status"] = "GT_NO_MOVING_SAMPLES"
        else:
            r["status"] = "GT_UNAVAILABLE" if not gt_ok else "ESTIMATE_UNAVAILABLE"
        rows.append(r)
    return rows


def support_metrics(F: Frozen, gt: G.GroundTruth, tols) -> tuple[list, dict]:
    rows, mapping_info = [], {}
    T = F.T
    labs = {}
    for s, key, frame in (("F", "F0", 0), ("B", "BT", "T")):
        clouds = gt.labelled_cloud(frame, T)
        for mname, cl in clouds.items():
            if cl is None or cl[1] is None:
                continue
            for tol in tols:
                lab = G.map_labels(F.pos[key], cl, tol)
                labs[(s, mname, tol)] = lab
                mapping_info[f"{s}/{mname}/{tol}"] = {
                    "rows": int(lab.size), "labelled_static": int((lab == 0).sum()),
                    "labelled_moving": int((lab == 1).sum()), "unlabelled": int((lab == -1).sum()),
                    "coverage": float((lab >= 0).mean()),
                    "rule": f"nearest GT sample within {tol} m of the row's ANCHOR position ({key}); "
                            f"GT cloud '{mname}' at frame {frame}"}
    preds = {}
    for s, d in (("F", "forward"), ("B", "backward")):
        n = F.pos["F0" if s == "F" else "BT"].shape[0]
        ps = F.bidi / d / "predicted_support.json"
        if ps.is_file():
            j = json.loads(ps.read_text())
            p = np.zeros(n, bool)
            p[np.asarray(j.get("dynamic_gaussians", []), dtype=np.int64)] = True
            preds[(s, "predicted_support_dynamic")] = (p, "moving")
        sp = F.bidi / f"{d}_rigid" / "support_coherence" / "support.json"
        if sp.is_file():
            j = json.loads(sp.read_text())
            p = np.zeros(n, bool)
            p[np.asarray(j.get("members", []), dtype=np.int64)] = True
            preds[(s, "rigid_phase2_support_members")] = (p, "moving")
        jo = F.diag / "j1_ownership.npz"
        if jo.is_file():        # J1 ownership tested by the diagnosis on the recorded RGB-D
            z = np.load(jo)
            for jd in ("forward", "backward"):
                if f"{jd}__{s}" in z.files:
                    p = np.zeros(n, bool)
                    p[z[f"{jd}__{s}"]] = True
                    preds[(s, f"j1_tested_ownership_{jd}_joint")] = (p, "moving")
        p = np.zeros(n, bool)
        for st in F.statics():
            rr = st["F_rows"] if s == "F" else st["B_rows"]
            p[rr] = True
        preds[(s, "endpoint_static_world_matched")] = (p, "static")
    for (s, pname), (p, cls) in preds.items():
        for (s2, mname, tol), lab in labs.items():
            if s2 != s:
                continue
            gtpos = lab.copy()
            if cls == "static":       # score the STATIC prediction as positive = static
                gtpos = np.where(lab >= 0, (lab == 0).astype(int), -1)
            c = G.confusion(p, gtpos)
            rows.append({"source": s, "direction": "forward" if s == "F" else "backward",
                         "prediction": pname, "positive_class": cls, "gt_mapping": mname,
                         "tolerance_m": tol, **c, "metric_class": "gt_evaluation_metric",
                         "note": "different predictions are scored separately; none is chosen by GT"})
    return rows, mapping_info, labs


def geometry_metrics(F: Frozen, gt: G.GroundTruth, tols) -> list:
    rows = []
    for key, frame in (("F0", 0), ("B0", 0), ("FT", "T"), ("BT", "T")):
        cl = gt.cloud.get(frame)
        if cl is None:
            rows.append({"state": key, "status": "GT_UNAVAILABLE"})
            continue
        xyz, part = cl
        for subset in ("all", "static_link", "moving_link"):
            if subset == "all":
                g = xyz
            else:
                if part is None:
                    continue
                g = xyz[part == (0 if subset == "static_link" else 1)]
            for tol in tols:
                sc = G.surface_scores(F.pos[key], g, tol) if len(g) else {"status": "NO_GT_SAMPLES"}
                rows.append({"state": key, "frame": 0 if frame == 0 else F.T, "gt_subset": subset,
                             "tolerance_m": tol, **sc,
                             "definition": ("Gaussian CENTRES vs GT surface SAMPLES (camera-visible, sparse); "
                                            "precision = rows within tol of a sample (one-sided), recall = "
                                            "samples within tol of a row (one-sided); recall on a link uses "
                                            "that link's samples only"),
                             "metric_class": "gt_evaluation_metric"})
    return rows


def patch_metrics(F: Frozen, gt: G.GroundTruth, proj: Projector | None, tols) -> list:
    from scipy.spatial import cKDTree
    rows = []
    T = F.T
    tol = G.PRIMARY_TOL
    cloud = {0: gt.labelled_cloud(0, T).get("kinematic") or gt.cloud.get(0), "T": gt.cloud.get("T")}
    # the opposite source's ANCHORED model moved by the GT motion (evaluation only)
    gt_moved = {}
    for s, key, frame in (("F", "F0", 0), ("B", "BT", "T")):
        cl = cloud[frame]
        if cl is None or cl[1] is None or gt.q is None:
            continue
        lab = G.map_labels(F.pos[key], cl, tol)
        mv = lab == 1
        for t in (0, T):
            X = F.pos[key].copy()
            X[mv] = G.apply(gt.transform(0 if frame == 0 else T, t), X[mv])
            gt_moved[(s, t)] = X
    for pt in F.patches():
        s, fr, own, opp = OWN[pt["type"]]
        t = 0 if fr == 0 else T
        cl = cloud[fr]
        rec = {"patch_id": pt["patch_id"], "type": pt["type"], "detailed": pt["detailed"],
               "frozen_J0_label": pt["J0_label"] or None, "frozen_screen_J0_label": pt["screen_J0_label"],
               "frozen_J1_labels": pt["J1"], "n_rows": None if pt["rows"] is None else int(pt["rows"].size),
               "metric_class": "gt_evaluation_metric (post hoc)"}
        if pt["rows"] is None:
            rec["status"] = ("ROWS_NOT_SAVED: this diagnosis output stored no row list for the patch "
                             "(screened-only patches before the rows_list fix); not scored")
            rows.append(rec)
            continue
        if cl is None:
            rec["status"] = "GT_UNAVAILABLE"
            rows.append(rec)
            continue
        X = F.pos[own][pt["rows"]]
        xyz, part = cl
        tree = cKDTree(xyz)
        d, i = tree.query(X)
        rec["patch_on_gt_surface_frac"] = float(np.mean(d <= tol))
        rec["patch_median_dist_to_gt_m"] = float(np.median(d))
        if part is not None:
            near = d <= tol
            rec["patch_gt_moving_frac"] = float(np.mean(part[i][near] == 1)) if near.any() else None
            rec["patch_gt_static_frac"] = float(np.mean(part[i][near] == 0)) if near.any() else None
        if proj is not None and gt.mask_dir is not None:
            hit, inview = proj.inside_mask(X, gt.mask_dir, t)
            rec["patch_inside_object_mask_frac"] = float(hit.mean())
            rec["patch_in_view_frac"] = float(inview.mean())
        # GT samples near the patch, and who covers them
        gi = np.unique(np.concatenate(tree.query_ball_point(X, tol))) if len(X) else np.empty(0, int)
        gi = gi.astype(np.int64)
        rec["gt_samples_near_patch"] = int(gi.size)
        if gi.size:
            P = xyz[gi]
            rec["opposite_render_state"] = opp
            rec["opposite_coverage_frac"] = float(np.mean(cKDTree(F.pos[opp]).query(P)[0] <= tol))
            o = "F" if s == "B" else "B"
            if (o, t) in gt_moved:
                rec["opposite_anchor_gt_transported_coverage_frac"] = float(
                    np.mean(cKDTree(gt_moved[(o, t)]).query(P)[0] <= tol))
            own_cov = rec["opposite_coverage_frac"]
            gtt = rec.get("opposite_anchor_gt_transported_coverage_frac")
            if rec["patch_on_gt_surface_frac"] < 0.5:
                reading = "PATCH_NOT_ON_GT_SURFACE"
            elif own_cov >= 0.5:
                reading = "GT_OPPOSITE_COVERS_IT (common surface in GT space)"
            elif gtt is not None and gtt >= 0.5:
                reading = "GT_OPPOSITE_HAS_IT_WHEN_MOVED_BY_GT (transport)"
            else:
                reading = "GT_OPPOSITE_LACKS_IT (complementary in GT space)"
            rec["posthoc_gt_reading"] = reading
        else:
            rec["posthoc_gt_reading"] = "NO_GT_SAMPLES_NEAR_PATCH"
        # tolerance sweep of the same reading (evaluation only)
        for tl in tols:
            dd = tree.query(X)[0]
            gi_ = np.unique(np.concatenate(tree.query_ball_point(X, tl))).astype(np.int64) if len(X) else np.empty(0, int)
            tag = f"{int(round(tl * 100))}cm"
            if gi_.size == 0:
                rec[f"reading_{tag}"] = "NO_GT_SAMPLES_NEAR_PATCH"
                continue
            P_ = xyz[gi_]
            oc = float(np.mean(cKDTree(F.pos[opp]).query(P_)[0] <= tl))
            o_ = "F" if s == "B" else "B"
            gc = float(np.mean(cKDTree(gt_moved[(o_, t)]).query(P_)[0] <= tl)) if (o_, t) in gt_moved else None
            rec[f"opposite_coverage_{tag}"] = oc
            rec[f"gt_transported_coverage_{tag}"] = gc
            rec[f"reading_{tag}"] = ("PATCH_NOT_ON_GT_SURFACE" if np.mean(dd <= tl) < 0.5 else
                                     "GT_OPPOSITE_COVERS_IT" if oc >= 0.5 else
                                     "GT_OPPOSITE_HAS_IT_WHEN_MOVED_BY_GT" if gc is not None and gc >= 0.5 else
                                     "GT_OPPOSITE_LACKS_IT")
        rec["status"] = "OK"
        rec["tolerance_m"] = tol
        rows.append(rec)
    for st in F.statics():
        rec = {"patch_id": st["patch_id"], "type": "STATIC_" + str(st["tier"]), "detailed": True,
               "frozen_J0_label": st["tier"], "n_rows": int(st["F_rows"].size),
               "metric_class": "gt_evaluation_metric (post hoc)", "tolerance_m": tol}
        c0, cT = cloud[0], cloud["T"]
        if c0 is None or cT is None or c0[1] is None or cT[1] is None:
            rec["status"] = "GT_UNAVAILABLE"
            rows.append(rec)
            continue
        lf = G.map_labels(F.pos["F0"][st["F_rows"]], c0, tol)
        lb = G.map_labels(F.pos["BT"][st["B_rows"]], cT, tol)
        rec |= {"F_rows_gt_static_frac": float(np.mean(lf[lf >= 0] == 0)) if (lf >= 0).any() else None,
                "B_rows_gt_static_frac": float(np.mean(lb[lb >= 0] == 0)) if (lb >= 0).any() else None,
                "F_rows_unlabelled": int((lf < 0).sum()), "B_rows_unlabelled": int((lb < 0).sum()),
                "status": "OK"}
        rows.append(rec)
    return rows


def write_report(out: Path, S: dict, joint_rows, support_rows, geom_rows, patch_rows, gtp):
    L = [f"# {S['scene']}: GT evaluation of the endpoint diagnosis — EVALUATION ONLY\n",
         "> These numbers are computed AFTER the diagnosis was sealed, in a separate process, from the "
         "dataset ground truth. They were not available to, and cannot change, any J0/J1 label, patch, "
         "rank, threshold or support decision. Primary files were hashed before and after: "
         f"**{S['primary_outputs_unchanged']}**.\n",
         f"Status: **{S['status']}**. Diagnosis: `{S['diagnosis_dir']}` (tag {S['tag']}, fingerprint "
         f"{S['diagnosis_fingerprint'][:16]}).\n", "## Ground truth used\n"]
    for k, v in gtp["files"].items():
        L.append(f"* `{v['path']}` — {v['meaning']}")
    for k in gtp["missing"]:
        L.append(f"* MISSING: {k}")
    L.append(f"* gauge: {json.dumps(gtp['gauge'])}")
    L.append(f"* GT motion convention: {json.dumps(gtp['motion_sign_check'])}\n")
    L.append("## Estimated joints vs GT\n")
    L.append("| direction | hypothesis | estimate | decided | GT type | line angle (deg) | line dist (m) | coord MAE | coord RMSE | rot err mean (deg) | point-transfer err mean (m) | status |")
    L.append("|---|---|---|---|---|---|---|---|---|---|---|---|")
    for r in joint_rows:
        mae = r.get("coord_mae")
        L.append(f"| {r['direction']} | {r['hypothesis']} | {r.get('estimate_status')} | {r.get('decided_type')} | "
                 f"{r.get('gt_type')} | {fmt(r.get('axis_line_angle_deg'), 2)} | {fmt(r.get('axis_line_distance_m'))} | "
                 f"{fmt(mae)} {r.get('coord_units', '')} | {fmt(r.get('coord_rmse'))} | "
                 f"{fmt(r.get('pose_rotation_err_deg_mean'), 2)} | {fmt(r.get('point_transfer_err_m_mean'))} | {r.get('status')} |")
    L.append("\nThe estimator decided no type; both hypotheses are scored individually and neither is "
             "promoted by GT. Estimates flagged NO_MEASURABLE_MOTION are reported as degenerate, not hidden.\n")
    L.append(f"## Support (tolerance {G.PRIMARY_TOL} m, unlabelled rows excluded)\n")
    L.append("| source | prediction | positive | GT mapping | P | R | F1 | IoU | labelled / unlabelled |")
    L.append("|---|---|---|---|---|---|---|---|---|")
    for r in support_rows:
        if r["tolerance_m"] != G.PRIMARY_TOL:
            continue
        L.append(f"| {r['source']} | {r['prediction']} | {r['positive_class']} | {r['gt_mapping']} | {fmt(r['precision'], 3)} | "
                 f"{fmt(r['recall'], 3)} | {fmt(r['f1'], 3)} | {fmt(r['iou'], 3)} | {r['labelled_rows']} / {r['unlabelled_rows_excluded']} |")
    L.append("\n## Model geometry vs GT surface samples (tolerance 0.02 m)\n")
    L.append("| state | GT subset | precision | recall | F | median acc (m) |")
    L.append("|---|---|---|---|---|---|")
    for r in geom_rows:
        if r.get("tolerance_m") == G.PRIMARY_TOL and r.get("status") == "OK":
            L.append(f"| {r['state']} | {r['gt_subset']} | {fmt(r['precision_at_tol'], 3)} | {fmt(r['recall_at_tol'], 3)} | "
                     f"{fmt(r['f_score_at_tol'], 3)} | {fmt(r['accuracy_median_m'])} |")
    L.append("\n## Frozen patch labels vs a post-hoc GT reading\n")
    L.append("Rows: frozen label (detailed patches' J0, else the screened J0). Columns: the post-hoc GT "
             "reading of the same patch. This is an evaluation cross-tab, not a re-labelling.\n")
    tab = {}
    for r in patch_rows:
        if r["type"].startswith("STATIC"):
            continue
        k = (r["type"], r.get("frozen_J0_label") or r.get("frozen_screen_J0_label"))
        tab.setdefault(k, {})
        g = r.get("posthoc_gt_reading", r.get("status"))
        tab[k][g] = tab[k].get(g, 0) + 1
    readings = sorted({g for v in tab.values() for g in v})
    L.append("| type | frozen label | " + " | ".join(readings) + " |")
    L.append("|---|---|" + "---|" * len(readings))
    for (ty, lab), v in sorted(tab.items()):
        L.append(f"| {ty} | {lab} | " + " | ".join(str(v.get(g, 0)) for g in readings) + " |")
    L.append("\nTolerance sensitivity (detailed patches only): the same reading at 1 / 2 / 5 cm.\n")
    L.append("| patch | frozen J0 | 1 cm | 2 cm | 5 cm | GT-moved opposite coverage 1/2/5 cm |")
    L.append("|---|---|---|---|---|---|")
    for r in patch_rows:
        if r.get("detailed") and not r["type"].startswith("STATIC") and r.get("status") == "OK":
            L.append(f"| {r['patch_id']} | {r.get('frozen_J0_label')} | {r.get('reading_1cm')} | {r.get('reading_2cm')} | "
                     f"{r.get('reading_5cm')} | {fmt(r.get('gt_transported_coverage_1cm'), 2)} / "
                     f"{fmt(r.get('gt_transported_coverage_2cm'), 2)} / {fmt(r.get('gt_transported_coverage_5cm'), 2)} |")
    j1 = {}
    for r in patch_rows:
        for d, lab in (r.get("frozen_J1_labels") or {}).items():
            k = (lab, r.get("posthoc_gt_reading"))
            j1[k] = j1.get(k, 0) + 1
    if j1:
        L.append("\nJ1 labels (detailed patches) vs the GT reading:\n")
        for (lab, g), n in sorted(j1.items(), key=lambda x: (str(x[0][0]), str(x[0][1]))):
            L.append(f"* {lab} / {g}: {n}")
    st = [r for r in patch_rows if r["type"].startswith("STATIC") and r.get("status") == "OK"]
    if st:
        fs = [r["F_rows_gt_static_frac"] for r in st if r.get("F_rows_gt_static_frac") is not None]
        bs = [r["B_rows_gt_static_frac"] for r in st if r.get("B_rows_gt_static_frac") is not None]
        L.append(f"\nStatic world-matched groups: {len(st)}; median fraction of their F rows on the GT static "
                 f"link {fmt(float(np.median(fs)) if fs else None, 3)}, B rows {fmt(float(np.median(bs)) if bs else None, 3)}.")
    L.append("\n## Caveats\n")
    L += ["* GT surface samples are sparse and camera-visible only; Gaussian centres are not surfaces. "
          "Recall/precision are about those samples, not the full mesh.",
          "* Frame-0 labels: 'as_given' (points3D_parts.ply) and 'kinematic' (frame-T samples with the "
          "moving link moved back by the GT motion) are reported separately; they can disagree when "
          "the frame-0 export labels the moving link poorly.",
          "* Object masks are silhouettes, not visibility: 'inside mask' does not mean unoccluded.",
          "* The post-hoc GT reading uses GT ownership of the opposite anchor to move it by the GT motion; "
          "that is an evaluation construct, never an input.",
          "* Coordinate errors use the estimated axis orientation vs the GT axis to resolve the sign "
          "(oracle_metric); no offset, scale or time shift is fitted."]
    (out / "gt_evaluation_report.md").write_text("\n".join(L) + "\n")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--diagnosis-dir", required=True)
    ap.add_argument("--dataset-dir", required=True)
    ap.add_argument("--results-dir", default=None,
                    help="the scene's results dir (default: two levels above the diagnosis dir)")
    ap.add_argument("--dry-run", action="store_true", help="report GT availability; write nothing")
    ap.add_argument("--resume", action="store_true",
                    help="reuse a complete evaluation with the same fingerprint, or replace an incomplete one")
    ap.add_argument("--replace-evaluation", action="store_true",
                    help="supersede a complete evaluation with a different fingerprint (kept under "
                         "evaluation_gt/superseded/)")
    args = ap.parse_args()
    diag = Path(args.diagnosis_dir)
    data = Path(args.dataset_dir)
    res = Path(args.results_dir) if args.results_dir else diag.parents[1]
    rs = diag / "run_state.json"
    if not rs.is_file() or not json.loads(rs.read_text()).get("complete"):
        print(f"REFUSED: {diag} is not a sealed diagnosis (run_state.json complete: true required)")
        return 2
    out = diag / EVAL_DIR
    if out.is_symlink() or (out.exists() and diag.resolve() not in out.resolve().parents):
        print(f"REFUSED: {out} is a symlink or resolves outside the diagnosis directory")
        return 2
    gt = G.GroundTruth(data)
    if args.dry_run:
        rep = {"diagnosis_dir": str(diag), "dataset_dir": str(data), "gt_status": gt.status,
               "files": {k: v["path"] for k, v in gt.files.items()}, "missing": gt.missing,
               "masks": str(gt.mask_dir) if gt.mask_dir else None,
               "joint": ({k: (v.tolist() if hasattr(v, "tolist") else v) for k, v in gt.joint.items()}
                         if gt.joint else None),
               "metrics_available": {
                   "joint_axis_and_coordinate": gt.joint is not None and gt.q is not None,
                   "rigid_pose": gt.q is not None and "parts_frameT" in gt.files,
                   "support_confusion": "parts_frame0" in gt.files or "parts_frameT" in gt.files,
                   "geometry_and_patch_posthoc": "parts_frame0" in gt.files and "parts_frameT" in gt.files,
                   "object_mask": gt.mask_dir is not None},
               "would_write_to": str(out), "writes": "NONE (dry run)"}
        print(json.dumps(rep, indent=1))
        return 0
    t0 = time.time()
    before = primary_manifest(diag)
    man_hash = hashlib.sha256(json.dumps(before, sort_keys=True).encode()).hexdigest()
    fp = hashlib.sha256(json.dumps({"primary": man_hash, "gt": {k: v["sha256"] for k, v in gt.files.items()},
                                    "code": code_hash(), "tols": G.TOLS_M}, sort_keys=True).encode()).hexdigest()
    st_path = out / "evaluation_state.json"
    if st_path.is_file():
        old = json.loads(st_path.read_text())
        if old.get("fingerprint") != fp:
            if args.replace_evaluation or (args.resume and not old.get("complete")):
                # keep the old evaluation, moved INSIDE evaluation_gt/ (nothing deleted)
                arch = out / "superseded" / f"{str(old.get('fingerprint', 'unknown'))[:16]}_{int(time.time())}"
                arch.mkdir(parents=True)
                for q in list(out.iterdir()):
                    if q.name != "superseded":
                        q.rename(arch / q.name)
                print(f"previous evaluation ({'complete' if old.get('complete') else 'incomplete'}) moved to {arch}")
            else:
                print(f"REFUSED: {out} holds evaluation {old.get('fingerprint', '')[:16]}, this is {fp[:16]} "
                      "(primary outputs, GT or evaluator changed); nothing deleted. Use --resume to replace an "
                      "incomplete evaluation or --replace-evaluation to supersede a complete one (both keep it "
                      "under evaluation_gt/superseded/)")
                return 2
        elif old.get("complete") and args.resume:
            print(f"evaluation {fp[:16]} already complete: {out}")
            return 0
        elif not args.resume:
            print(f"REFUSED: {out} exists; pass --resume")
            return 2
    out.mkdir(exist_ok=True)
    st_path.write_text(json.dumps({"fingerprint": fp, "complete": False}))
    summary = {"scene": json.loads((diag / "endpoint_summary.json").read_text()).get("scene"),
               "tag": json.loads((diag / "provenance.json").read_text()).get("tag"),
               "diagnosis_dir": str(diag), "diagnosis_fingerprint": json.loads(rs.read_text()).get("fingerprint", ""),
               "gt_status": gt.status, "evaluation_fingerprint": fp}
    status, err = "EVAL_OK", None
    joint_rows = support_rows = geom_rows = patch_rows = []
    mapping_info = {}
    gtp = {"files": gt.files, "missing": gt.missing, "notes": gt.notes,
           "masks": str(gt.mask_dir) if gt.mask_dir else None}
    try:
        F = Frozen(diag, res)
        gtp["gauge"] = gt.gauge_check(Path(data) / "colmap" / "sparse" / "0")
        gtp["motion_sign_check"] = gt.verify_motion_sign(F.T) if gt.q is not None else {"status": "NOT_AVAILABLE"}
        if gt.joint:
            gtp["joint"] = {k: (v.tolist() if hasattr(v, "tolist") else v) for k, v in gt.joint.items()}
        if gt.status == "GT_UNAVAILABLE":
            status = "GT_UNAVAILABLE"
        else:
            joint_rows = joint_metrics(F, gt)
            support_rows, mapping_info, _labs = support_metrics(F, gt, G.TOLS_M)
            geom_rows = geometry_metrics(F, gt, G.TOLS_M)
            proj = Projector(Path(data) / "colmap" / "sparse" / "0") if (Path(data) / "colmap").is_dir() else None
            patch_rows = patch_metrics(F, gt, proj, G.TOLS_M)
            if gt.status == "GT_INCOMPLETE":
                status = "GT_INCOMPLETE"
    except Exception as e:  # a failed evaluation never touches the diagnosis
        status, err = "EVAL_FAILED", "".join(traceback.format_exception(e))[-2000:]
    jh = ["direction", "hypothesis", "estimate_status", "decided_type", "gt_type", "hypothesis_matches_gt_type",
          "degenerate_estimate", "axis_line_angle_deg", "axis_direction_angle_deg_vs_gt_axis", "axis_line_distance_m",
          "estimator_sign_convention", "estimator_convention_residual", "estimator_convention_residual_units",
          "coord_units", "coord_mae", "coord_rmse", "coord_max_abs", "coord_mae_deg", "coord_rmse_deg",
          "gt_coord_range", "est_coord_range", "coord_alignment", "coord_metric_class", "coord_note",
          "pose_rotation_err_deg_mean", "pose_rotation_err_deg_max", "point_transfer_err_m_mean",
          "point_transfer_err_m_max", "gt_moving_part_mean_displacement_m", "pose_metric_definition",
          "gt_axis", "metric_class", "status"]
    sh = ["source", "direction", "prediction", "positive_class", "gt_mapping", "tolerance_m", "tp", "fp", "fn", "tn",
          "labelled_rows", "unlabelled_rows_excluded", "precision", "recall", "f1", "iou", "selection_fraction",
          "prevalence", "metric_class", "note"]
    gh = ["state", "frame", "gt_subset", "tolerance_m", "status", "model_rows", "gt_samples", "accuracy_median_m",
          "precision_at_tol", "completeness_median_m", "recall_at_tol", "f_score_at_tol", "definition", "metric_class"]
    ph = ["patch_id", "type", "detailed", "frozen_J0_label", "frozen_screen_J0_label", "frozen_J1_labels", "n_rows",
          "patch_on_gt_surface_frac", "patch_median_dist_to_gt_m", "patch_gt_moving_frac", "patch_gt_static_frac",
          "patch_inside_object_mask_frac", "patch_in_view_frac", "gt_samples_near_patch", "opposite_render_state",
          "opposite_coverage_frac", "opposite_anchor_gt_transported_coverage_frac", "posthoc_gt_reading",
          "F_rows_gt_static_frac", "B_rows_gt_static_frac", "F_rows_unlabelled", "B_rows_unlabelled",
          "tolerance_m", "metric_class", "status"] + [f"{k}_{int(round(t * 100))}cm" for t in G.TOLS_M
                                                      for k in ("opposite_coverage", "gt_transported_coverage",
                                                                "reading")]
    write_csv(out / "metrics_joint.csv", joint_rows, jh)
    write_csv(out / "metrics_support.csv", support_rows, sh)
    write_csv(out / "metrics_geometry.csv", geom_rows, gh)
    write_csv(out / "metrics_patches.csv", patch_rows, ph)
    gtp["support_mapping"] = mapping_info
    gtp["data_dictionary"] = DATA_DICTIONARY
    gtp["tolerances_m"] = list(G.TOLS_M)
    gtp["evaluator_code_sha256"] = code_hash()
    (out / "gt_evaluation_provenance.json").write_text(json.dumps(gtp, indent=1, default=str))
    after = primary_manifest(diag)
    changed = sorted(k for k in set(before) | set(after) if before.get(k) != after.get(k))
    summary |= {"status": status, "error": err, "primary_outputs_unchanged": not changed,
                "primary_files_hashed": len(before), "changed": changed,
                "counts": {"joint_rows": len(joint_rows), "support_rows": len(support_rows),
                           "geometry_rows": len(geom_rows), "patch_rows": len(patch_rows)},
                "seconds": time.time() - t0}
    (out / "gt_evaluation_summary.json").write_text(json.dumps(summary, indent=1, default=str))
    writes = sorted(str(p.relative_to(diag)) for p in out.rglob("*") if p.is_file())
    checks = {"primary_manifest_sha256_before": man_hash,
              "primary_manifest_sha256_after": hashlib.sha256(json.dumps(after, sort_keys=True).encode()).hexdigest(),
              "primary_outputs_unchanged": not changed, "changed": changed,
              "all_writes_under_evaluation_gt": all(w.startswith(EVAL_DIR + "/") for w in writes),
              "evaluation_dir_is_symlink": out.is_symlink(),
              "diagnosis_sealed_before_evaluation": True,
              "diagnosis_never_imports_evaluator": "checked by tests/test_fb_endpoint_gt_evaluation.py",
              "gauge": gtp.get("gauge"), "gt_motion_sign": gtp.get("motion_sign_check"),
              "gt_status": gt.status, "status": status}
    (out / "evaluation_checks.json").write_text(json.dumps(checks, indent=1, default=str))
    if status in ("EVAL_OK", "GT_INCOMPLETE"):
        write_report(out, summary, joint_rows, support_rows, geom_rows, patch_rows, gtp)
    else:
        (out / "gt_evaluation_report.md").write_text(
            f"# GT evaluation — {status}\n\nNo scores were produced. {err or ''}\n")
    st_path.write_text(json.dumps({"fingerprint": fp, "complete": status == "EVAL_OK" and not changed,
                                   "status": status}))
    print(f"GT evaluation {status}: {out}  (primary unchanged: {not changed})")
    if changed:
        return 1
    return {"EVAL_OK": 0, "GT_UNAVAILABLE": 3, "GT_INCOMPLETE": 3}.get(status, 1)


if __name__ == "__main__":
    raise SystemExit(main())
