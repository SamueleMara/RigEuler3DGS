#!/usr/bin/env python
"""Endpoint correspondence, missing geometry and visibility diagnosis, with the
existing IEKF renderer and the ESTIMATED joints (read only).

CODEX_DIAGNOSE_ENDPOINT_COMPLEMENTARITY_WITH_IEKF_AND_ESTIMATED_JOINTS.md.

  A  frame 0:  F0 (initial PLY, anchor) vs B0 (backward Phase-1 transport) against Y_0
  B  frame T:  FT (forward Phase-1 transport) vs BT (final PLY, anchor) against Y_T
  C  static:   F0 vs BT in world coordinates, on pixels whose MEASURED depth is
               unchanged between 0 and T, for rows whose own tracks are static
  J0 joint agnostic, J1 joint informed (estimated rigid part motion from
  bidirectional/<d>_rigid, never GT), reported side by side.

Every state is rendered alone; nothing is merged, trained or optimised. A GT
firewall (observation_diagnostics.gt_firewall) refuses GT files process-wide.

    python tools/analyze_fb_endpoint_correspondence.py --scene washer_103776 \
        --frames endpoints --cameras all --estimated-joints auto \
        --tag endpoint_evidence_v1 --execute
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import resource
import shutil
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
# less fragmentation on a shared GPU (set before torch is imported)
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

DEFAULT_RESULTS = "/media/samuelemara/Extreme SSD/Samuele/results_11_10_26/ours"
DEFAULT_DATASET = ("/media/samuelemara/Extreme SSD/Samuele/Backup_09_10_26/Sapien/output/"
                   "preparation_8cams/partnet_subset_fixed_render_linear")
OUT_SUBDIR = "endpoint_correspondence"
CODE = ("observation_diagnostics/endpoint_compare.py", "observation_diagnostics/joint_evidence.py",
        "observation_diagnostics/gt_firewall.py", "observation_diagnostics/renderer_access.py",
        "observation_diagnostics/pixel_evidence.py", "observation_diagnostics/gaussian_evidence.py",
        "observation_diagnostics/pair_evidence.py", "tools/analyze_fb_endpoint_correspondence.py")

# declared rules (reported in provenance; swept where stated)
RULES = {"U_MIN": 1.0, "S_MIN": 1.0, "MIN_WITNESS_PIXELS": 30, "MIN_WITNESS_CAMERAS": 2,
         "COMMON_FRACTION": 0.5, "COUNTERPART_FRACTION": 0.3, "EXCLUSIVE_FRINGE": 0.25, "WORLD_NEAR_FRACTION": 0.5, "APPEARANCE_RGB_RATIO": 0.5,
         "STATIC_NET_M": 0.02, "STATIC_PATH_M": 0.05, "STATIC_RHO": 0.3,
         "UNSEEN_FRACTION": 0.6, "J1_RECOVERS": 0.5, "J1_FAILS": 0.1,
         "MAX_DETAILED_PATCHES_PER_TYPE": 12, "MAX_OCCLUSION_PATCHES_PER_TYPE": 5,
         "MAX_PROBE_PATCHES_PER_TYPE": 3, "STATIC_VOXEL_M": 0.05, "PRUNE_TAU": 1e-4}
SWEEP_DEPTH_M = (0.02, 0.04, 0.06)
PAIR_COLS = ("O", "O_surf_1", "O_incompat", "O_surf_rgb", "dist3d", "dist3d_surf")

TYPES = {  # (endpoint, explaining source) -> how to read it
    "B_ONLY_AT_T": {"t": "T", "s": "B", "contrast": "B", "W": "W_Bonly", "own": "BT",
                    "fail": "FT", "fail_transported": True},
    "F_ONLY_AT_0": {"t": "0", "s": "F", "contrast": "A", "W": "W_Fonly", "own": "F0",
                    "fail": "B0", "fail_transported": True},
    "B_ONLY_AT_0": {"t": "0", "s": "B", "contrast": "A", "W": "W_Bonly", "own": "B0",
                    "fail": "F0", "fail_transported": False},
    "F_ONLY_AT_T": {"t": "T", "s": "F", "contrast": "B", "W": "W_Fonly", "own": "FT",
                    "fail": "BT", "fail_transported": False},
}


class Log:
    def __init__(self, out: Path):
        self.path = out / "log.txt"

    def __call__(self, *a):
        msg = " ".join(str(x) for x in a)
        print(msg, flush=True)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.path, "a") as fh:
            fh.write(time.strftime("%H:%M:%S ") + msg + "\n")


def code_hash() -> str:
    h = hashlib.sha256()
    for rel in CODE:
        h.update(rel.encode())
        h.update((REPO / rel).read_bytes())
    return h.hexdigest()


def peak():
    import torch
    return {"peak_gpu_mib": round(torch.cuda.max_memory_allocated() / 2 ** 20, 1)
            if torch.cuda.is_available() else 0,
            "peak_host_rss_mib": round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024, 1)}


def write_csv(path: Path, rows: list):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("")
        return
    keys = []
    for r in rows:
        for k in r:
            if k not in keys:
                keys.append(k)
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow({k: (json.dumps(v) if isinstance(v, (list, dict)) else v) for k, v in r.items()})


def observed_names(row: dict) -> dict:
    """Output naming only: a valid RECORDED depth pixel is 'observed_depth_valid'
    (older outputs said 'gt_valid'; it was never ground truth)."""
    out = {}
    for k, v in row.items():
        k2 = (k.replace("gt_valid", "observed_depth_valid").replace("gt_invalid", "observed_depth_invalid")
               .replace("GT_INVALID", "OBSERVED_DEPTH_INVALID"))
        out[k2] = observed_names(v) if isinstance(v, dict) else v
    return out


DATA_DICTIONARY = {
    "sensor_observation_metric": "residuals/categories against the RECORDED RGB-D frames (rgb/, depth_npy/); "
                                 "legal for inference. observed_depth_valid = the recorded depth has a return",
    "inferred_joint_metric": "quantities of the ESTIMATED joints/poses from <d>_rigid; legal as evidence",
    "gt_evaluation_metric": "dataset ground truth vs frozen predictions; EVALUATION ONLY, produced by "
                            "tools/evaluate_fb_endpoint_correspondence_gt.py under evaluation_gt/",
    "oracle_metric": "a GT metric that uses GT to resolve a convention; EVALUATION ONLY"}


def jsonable(x):
    if isinstance(x, dict):
        return {str(k): jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [jsonable(v) for v in x]
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (np.floating,)):
        return float(x)
    if isinstance(x, (np.bool_,)):
        return bool(x)
    return x


# ---------------------------------------------------------------------------
# discovery, preflight, plan
# ---------------------------------------------------------------------------

def scene_paths(args):
    res = Path(args.results_root) / args.scene
    data = Path(args.dataset_root) / args.scene
    bidi = res / "bidirectional"
    return {"res": res, "data": data, "bidi": bidi,
            "F": {"model": res / "static_bidi" / "initial", "centers": bidi / "forward" / "phase1_centers",
                  "endpoint": "initial", "direction": "forward",
                  "support": bidi / "forward" / "predicted_support.json"},
            "B": {"model": res / "static_bidi" / "final", "centers": bidi / "backward" / "phase1_centers",
                  "endpoint": "final", "direction": "backward",
                  "support": bidi / "backward" / "predicted_support.json"}}


def ply_rows(ply: Path) -> int:
    with open(ply, "rb") as fh:
        for line in fh:
            line = line.decode(errors="replace").strip()
            if line.startswith("element vertex"):
                return int(line.split()[-1])
            if line == "end_header":
                break
    return -1


def find_ply(model: Path):
    its = sorted((int(p.name.split("_")[-1]), p) for p in (model / "point_cloud").glob("iteration_*")
                 if (p / "point_cloud.ply").is_file()) if (model / "point_cloud").is_dir() else []
    return its[-1][1] / "point_cloud.ply" if its else None


def preflight(args, P) -> dict:
    from observation_diagnostics.joint_evidence import DirectionEstimate
    rep = {"scene": args.scene, "results_dir": str(P["res"]), "dataset_dir": str(P["data"]),
           "blockers": [], "warnings": [], "gpu_work": False}
    for k in ("F", "B"):
        s = P[k]
        ply = find_ply(s["model"])
        e = {"model": str(s["model"]), "ply": str(ply) if ply else None,
             "rows": ply_rows(ply) if ply else None}
        man = s["centers"] / "trajectory_manifest.json"
        if not man.is_file():
            rep["blockers"].append(f"{k}: no {man}")
        else:
            m = json.loads(man.read_text())
            e["manifest"] = {x: m.get(x) for x in ("direction", "canonical_endpoint",
                                                   "canonical_original_frame", "trajectory_order",
                                                   "num_frames", "num_gaussians")}
            e["center_files"] = len(list(s["centers"].glob("centers_frame_*.npy")))
            if m.get("trajectory_order") != "original_chronological":
                rep["blockers"].append(f"{k}: trajectory_order {m.get('trajectory_order')}")
            if e["rows"] is not None and m.get("num_gaussians") != e["rows"]:
                rep["blockers"].append(f"{k}: manifest rows {m.get('num_gaussians')} vs PLY {e['rows']}")
        e["full_state_export"] = (s["centers"].parent / "full_state" / "full_state_manifest.json").is_file()
        e["state_mode"] = "RECORDED_FULL_12D_STATE" if e["full_state_export"] else "CENTER_ONLY_FIXED_ORIENTATION"
        if not ply:
            rep["blockers"].append(f"{k}: no trained PLY under {s['model']}")
        rep[f"source_{k}"] = e
    T = None
    if not rep["blockers"]:
        T = rep["source_B"]["manifest"]["canonical_original_frame"]
        rep["endpoints"] = {"0": 0, "T": T}
        if rep["source_F"]["manifest"]["canonical_original_frame"] != 0:
            rep["blockers"].append("forward anchor is not frame 0")
    cams = sorted(p.name for p in (P["data"] / "rgb").glob("cam_*")) if (P["data"] / "rgb").is_dir() else []
    rep["dataset_cameras"] = cams
    if not cams:
        rep["blockers"].append("no dataset rgb/cam_* directories")
    if T is not None:
        for f in (0, T):
            for c in cams:
                if not (P["data"] / "depth_npy" / c / f"frame_{f:06d}.npy").is_file():
                    rep["blockers"].append(f"missing depth {c} frame {f}")
    # estimated joints (read only; GT files never opened)
    est = {}
    if T is not None and args.estimated_joints != "off":
        for d, anchor, k in (("forward", 0, "F"), ("backward", T, "B")):
            e = DirectionEstimate(P["bidi"], d, anchor, rep[f"source_{k}"]["rows"] or 0)
            est[d] = e.describe()
    rep["estimated_joints"] = est
    usable = [d for d, e in est.items() if e["status"] == "USABLE"]
    rep["j1_usable_directions_pre_motion_check"] = usable
    if args.estimated_joints == "require" and not est:
        rep["blockers"].append("--estimated-joints require: no estimate found")
    try:
        import torch
        rep["cuda_available"] = torch.cuda.is_available()
        if torch.cuda.is_available():
            free, total = torch.cuda.mem_get_info()
            rep["gpu"] = {"free_mib": free // 2 ** 20, "total_mib": total // 2 ** 20}
    except Exception as e:  # pragma: no cover
        rep["cuda_available"] = False
        rep["warnings"].append(str(e))
    rep["disk_free_gib"] = round(shutil.disk_usage(P["res"]).free / 2 ** 30, 1)
    return rep


def plan(args, pf, cams) -> dict:
    n_est = 2 if args.estimated_joints != "off" else 0
    return {"base_renders": 4 * len(cams),
            "base_render_breakdown": "F0, B0 at frame 0 and FT, BT at frame T, per camera",
            "j1_renders_max": 2 * len(cams) * n_est,
            "j1_render_breakdown": "FT_J and B0_J per camera per USABLE estimate (decided after "
                                   "loading; NO_MEASURABLE_MOTION estimates render nothing)",
            "intermediate_renders_max": (RULES["MAX_OCCLUSION_PATCHES_PER_TYPE"] * 4 * 4 * 2),
            "intermediate_breakdown": "<= 5 patches x 4 types x <= 4 frames x 2 cameras, own source only",
            "case_study_and_probe_renders_max": 4 * RULES["MAX_DETAILED_PATCHES_PER_TYPE"] * 4,
            "pair_enumeration": "exact sparse products on the 3 x len(cams) endpoint views only "
                                "(no 60-frame pass), stored in RAM, summarised per patch"}


# ---------------------------------------------------------------------------
# the run
# ---------------------------------------------------------------------------

def to_np_contrib(c):
    return {"gid": c.gid.cpu().numpy(), "pix": c.pixel.cpu().numpy(), "q": c.q.cpu().numpy()}


def patch_mass(cn, rows_mask, z, depth, valid, tol, W, HW):
    """Per-pixel mass of the patch's depth-compatible contributions on W pixels."""
    sel = rows_mask[cn["gid"]]
    if not sel.any():
        return np.zeros(HW)
    pix, g = cn["pix"][sel], cn["gid"][sel]
    ok = valid.reshape(-1)[pix] & (np.abs(z[g] - depth.reshape(-1)[pix]) <= tol)
    if W is not None:
        ok &= W.reshape(-1)[pix]
    return np.bincount(pix[ok], weights=cn["q"][sel][ok].astype(np.float64), minlength=HW)


def execute(args, P, pf, out: Path, log: Log) -> int:
    import torch
    from observation_diagnostics import endpoint_compare as ec
    from observation_diagnostics import endpoint_visuals as ev
    ev.set_transport_mode(args.transport_orientation)
    from observation_diagnostics import gaussian_evidence as ge
    from observation_diagnostics import pair_evidence as pv
    from observation_diagnostics import pixel_evidence as pe
    from observation_diagnostics import visualize as vz
    from observation_diagnostics.gt_firewall import Firewall
    from observation_diagnostics.joint_evidence import DirectionEstimate, apply, cross_direction
    from observation_diagnostics.renderer_access import KERNEL_SHA256
    from observation_diagnostics.sources import check_gauge, load_gt
    from tools.run_iekf_scene import camera_specs

    fw = Firewall.install([str(Path(args.results_root)), str(Path(args.dataset_root))])
    t_start = time.time()
    if args.gpu_memory_fraction:
        torch.cuda.set_per_process_memory_fraction(args.gpu_memory_fraction)
    checks = {}
    T = pf["endpoints"]["T"]
    # ---- protected inputs, hashed before --------------------------------
    prot = []
    for k in ("F", "B"):
        prot += [find_ply(P[k]["model"]), P[k]["centers"] / "trajectory_manifest.json"]
        prot += sorted(P[k]["centers"].glob("*.npy"))
        if P[k]["support"].is_file():
            prot.append(P[k]["support"])
    for d in ("forward_rigid", "backward_rigid"):
        r = P["bidi"] / d
        for rel in ("phase2_rigid/poses_original_order.npy", "phase2_rigid/pose_covariances.npy",
                    "support_coherence/support.json", "joint_reduction/one_dof_diagnostics.json",
                    "joint_reduction/relative_increments.json",
                    "joint_reduction/revolute/geometry.json", "joint_reduction/prismatic/geometry.json",
                    "joint_reduction/revolute/trajectory.csv", "joint_reduction/prismatic/trajectory.csv"):
            if (r / rel).is_file():
                prot.append(r / rel)
    for rel in ("submodules/diff-gaussian-rasterization/cuda_rasterizer/iekf.cu",
                "submodules/diff-gaussian-rasterization/diff_gaussian_rasterization/__init__.py",
                "estimator/renderer_bridge.py", "tools/run_iekf_scene.py"):
        prot.append(REPO / rel)
    before = {str(p): ec.sha256_file(p) for p in prot}
    log(f"=== {args.scene}: {len(prot)} protected inputs hashed; GT firewall active")
    # ---- sources ----------------------------------------------------------
    F = ec.EndpointSource("F", P["F"]["model"], P["F"]["centers"], P["data"], "initial", "forward")
    B = ec.EndpointSource("B", P["B"]["model"], P["B"]["centers"], P["data"], "final", "backward")
    gauge = check_gauge(F, B)
    specs = camera_specs(F.cams_all, len(F.cams_all))
    all_cams = gauge["F_cameras"]
    cams = all_cams if args.cameras == "all" else [c.strip() for c in args.cameras.split(",")]
    if any(c not in all_cams for c in cams):
        raise SystemExit(f"cameras {cams} not among {all_cams}")
    checks["RENDER_AND_INDEXING"] = {
        "F0_is_initial_ply": F.anchor == 0 and F.anchor_error <= 1e-5,
        "BT_is_final_ply": B.anchor == T and B.anchor_error <= 1e-5,
        "B_frame0_is_transported_not_endpoint": float(np.abs(B.mu[0] - B.mu[T]).max()) > 1e-4,
        "rows": {"F": F.N, "B": B.N}, "row_mapping": {"F": F.row_mapping, "B": B.row_mapping},
        "camera_gauge_max_abs": gauge["max_abs_extrinsic_or_projection_difference"],
        "chronology": {"F": F.describe()["processing_order"], "B": B.describe()["processing_order"]},
        "state_mode": F.state_mode}
    th = pe.Thresholds()
    tol = th.contrib_depth_tol_m
    # ---- estimated joints -------------------------------------------------
    est = {}
    if args.estimated_joints != "off":
        est["forward"] = DirectionEstimate(P["bidi"], "forward", 0, F.N, anchor_xyz=F.mu[0])
        est["backward"] = DirectionEstimate(P["bidi"], "backward", T, B.N, anchor_xyz=B.mu[T])
    usable = {d: e for d, e in est.items() if e.status == "USABLE"}
    if args.estimated_joints == "require" and not usable:
        raise SystemExit("--estimated-joints require: no USABLE estimate: " + json.dumps(
            {d: [e.status] + e.notes for d, e in est.items()}))
    # ---- covariance-consistent free transport (J0) --------------------------
    rot_mode = args.transport_orientation
    Rtrack, track_stats = {}, {}

    def rot_track(src, f):
        """Local rigid rotation of src's free track from its anchor to frame f."""
        k = (src.key, f)
        if k not in Rtrack:
            if rot_mode == "fixed" or f == src.anchor:
                Rtrack[k] = None
            else:
                R_, res_, deg_, nonr_ = ec.track_rotations(src.mu[src.anchor], src.mu[f])
                ang = np.degrees(np.arccos(np.clip((np.trace(R_, axis1=1, axis2=2) - 1) / 2, -1, 1)))
                turned = ~(deg_ | nonr_)
                Rtrack[k] = R_
                track_stats[f"{src.key}:{src.anchor}->{f}"] = {
                    "turned_fraction": float(turned.mean()),
                    "track_not_locally_rigid_fraction": float(nonr_.mean()),
                    "degenerate_fraction": float(deg_.mean()),
                    "median_rotation_deg_turned": float(np.median(ang[turned])) if turned.any() else None,
                    "p90_rotation_deg_turned": float(np.percentile(ang[turned], 90)) if turned.any() else None,
                    "median_fit_residual_m": float(np.median(res_))}
                log(f"  J0 orientation {src.key} {src.anchor}->{f}: {turned.mean():.1%} of rows turned by a "
                    f"locally rigid track; {nonr_.mean():.1%} torn (trained orientation kept)")
        return Rtrack[k]

    def state_at(src, f):
        """src at original frame f: its anchor as trained, otherwise the free
        Phase-1 transport with orientations turned by the track (unless fixed)."""
        return ec.transported_state(src, f, rot_track(src, f))
    gt_cache = {}

    def gt_at(cam, f):
        k = (cam, f)
        if k not in gt_cache:
            spec_ = specs[all_cams.index(cam)]
            gt_cache[k] = dict(zip(("rgb", "depth", "valid"),
                                   load_gt(P["data"], cam, f, spec_.image_height, spec_.image_width)[:3]))
        return gt_cache[k]

    # ---- J1: estimated part motion, ownership tested against the RGB-D -------
    J1 = {}
    for d, e in usable.items():
        own_src, oth_src = (B, F) if d == "backward" else (F, B)
        ta, tx = own_src.anchor, oth_src.anchor          # the two endpoints
        M_own = e.transform(ta, tx)                      # moves owned geometry ta -> tx
        M_oth = e.transform(tx, ta)                      # moves the other source's part tx -> ta
        members = e.members
        sc_own = own_src.base.scales.max(1).values.cpu().numpy()
        r_own = max(2.0 * float(np.median(sc_own[members])), 0.01)
        # owner source: every row tested at tx; rows near the support also by rendering
        cand_own = np.union1d(members, ec.infer_ownership(own_src.mu[ta], own_src.mu[ta][members], r_own))
        own_res = ec.tested_ownership(own_src, ta, tx, M_own, cams, specs, all_cams,
                                      lambda cam: gt_at(cam, tx), th, cand_own, prior_owned=members,
                                      min_cameras=RULES["MIN_WITNESS_CAMERAS"])
        # other source: every row tested at ta; rows near the estimated part also by rendering
        part_at_tx = apply(M_own, own_src.mu[ta][members])
        cand_oth = ec.infer_ownership(oth_src.mu[tx], part_at_tx, 3.0 * r_own)
        oth_res = ec.tested_ownership(oth_src, tx, ta, M_oth, cams, specs, all_cams,
                                      lambda cam: gt_at(cam, ta), th, cand_oth, prior_owned=None,
                                      track_xyz=oth_src.mu[ta], min_cameras=RULES["MIN_WITNESS_CAMERAS"])
        # wall-clock timings go to the log only, so the provenance stays reproducible
        t_fs = own_res["reasons"].pop("seconds_freespace") + oth_res["reasons"].pop("seconds_freespace")
        t_rt = own_res["reasons"].pop("seconds_render_test") + oth_res["reasons"].pop("seconds_render_test")
        prox_oth = ec.infer_ownership(oth_src.mu[tx], part_at_tx, r_own)
        owned = {own_src.key: own_res["owned"], oth_src.key: oth_res["owned"]}
        J1[d] = {"owned_B": owned["B"], "owned_F": owned["F"], "own_source": own_src.key,
                 "B0_J": (T, owned["B"], e.transform(T, 0)), "FT_J": (0, owned["F"], e.transform(0, T)),
                 "ownership_radius_m": r_own,
                 "ownership": {own_src.key: {"source_role": "estimator's own source", "estimator_support": int(members.size),
                                             "candidates": own_res["candidates"], **own_res["reasons"],
                                             "rule": own_res.get("rule")},
                               oth_src.key: {"source_role": "other source (inferred)", "candidates": oth_res["candidates"],
                                             "previous_proximity_rule_would_own": int(prox_oth.size), **oth_res["reasons"],
                                             "rule": oth_res.get("rule")}}}
        n_r = 4 * len(cams)
        ro, rt = own_res["reasons"], oth_res["reasons"]
        log(f"  J1[{d}] ownership tested on the RGB-D ({n_r} renders + free-space): {own_src.key} owns "
            f"{ro['final_owned']} (support {members.size}: +{ro['added_beyond_estimator_support']} "
            f"-{ro['estimator_support_dropped']}); {oth_src.key} owns {rt['final_owned']} "
            f"(free-space {rt['freespace_moved']}, proximity alone would own {prox_oth.size}); "
            f"{t_fs:.0f} s free-space, {t_rt:.0f} s render test")
    if J1:      # the tested ownership sets themselves, for the separate GT evaluator
        np.savez_compressed(out / "j1_ownership.npz",
                            **{f"{d}__{k}": j[f"owned_{k}"].astype(np.int64) for d, j in J1.items() for k in ("F", "B")})
    jprov = {"estimates": {d: e.describe() for d, e in est.items()},
             "cross_direction": (cross_direction(est["forward"], est["backward"])
                                 if len(est) == 2 else None),
             "j1_runs": {d: {"own_source": j["own_source"],
                             "owned_rows_estimator": int((j["owned_B"] if j["own_source"] == "B"
                                                          else j["owned_F"]).size),
                             "owned_rows_inferred_in_other_source": int((j["owned_F"] if j["own_source"] == "B"
                                                                         else j["owned_B"]).size),
                             "ownership": j["ownership"],
                             "inferred_ownership_rule": "both sources: every row's centre tested against the "
                                                        "recorded depth at the other endpoint, static vs moved "
                                                        "(free-space test, net >= MIN_WITNESS_CAMERAS); rows "
                                                        "it leaves open by the render test near the part, "
                                                        "then the estimator support (owner) or the free "
                                                        "track (other source)",
                             "transport": "covariance-consistent: X <- R X + t and R_gauss <- R_part R_gauss "
                                          "(T_part(t) T_part(t_ref)^-1); scales/opacity/colour unchanged; "
                                          "unowned rows static"}
                         for d, j in J1.items()},
             "j1_status": ("AVAILABLE" if J1 else
                           "JOINT_ESTIMATE_UNAVAILABLE" if not est else
                           "NO_USABLE_ESTIMATE: " + "; ".join(f"{d}: {e.status}" for d, e in est.items())),
             "gt_used": False}
    (out / "joint_estimate_provenance.json").write_text(json.dumps(jsonable(jprov), indent=1))
    log(f"  sources F {F.N}, B {B.N} rows, endpoints 0 and {T}, cameras {cams}; J1: {jprov['j1_status']}"
        + (f" ({list(J1)})" if J1 else ""))
    # ---- per-camera endpoint renders ---------------------------------------
    from estimator.renderer_bridge import build_linearization
    stF0, stB0, stFT, stBT = F.state(0), state_at(B, 0), state_at(F, T), B.state(T)
    stB0_fixed, stFT_fixed = (B.state(0), F.state(T)) if rot_mode != "fixed" else (None, None)
    jprov["transport_orientation"] = {"mode": rot_mode, "track_rotation_stats": track_stats,
                                      "method": ("free Phase-1 centres; orientation turned by a local rigid "
                                                 "(Kabsch) fit of each row's 16 nearest neighbours' track "
                                                 "(R_gauss <- R_track R_gauss) where that neighbourhood moved "
                                                 "rigidly (rms residual <= 0.25 x its rms extent); torn or "
                                                 "collinear neighbourhoods keep the trained orientation") if rot_mode != "fixed" else
                                      "trained orientation kept (centres only)"}
    (out / "joint_estimate_provenance.json").write_text(json.dumps(jsonable(jprov), indent=1))
    stJ = {}
    for d, j in J1.items():
        stJ[d] = {"FT_J": ec.j1_state(F, j["FT_J"][0], j["FT_J"][1], j["FT_J"][2]),
                  "B0_J": ec.j1_state(B, j["B0_J"][0], j["B0_J"][1], j["B0_J"][2])}
    acc = {k: ge.new_frame_accumulator(F.N if k[1] == "F" else B.N, "cuda")
           for k in ("AF", "AB", "BF", "BB", "CF", "CB")}
    pair_views = {"A": [], "B": [], "C": []}
    per_cam, view_rows, cons_all, sweep, orient_rows = {}, [], {}, [], []
    drange = None
    budget_free = None
    if args.gpu_memory_fraction:
        budget_free = args.gpu_memory_fraction * torch.cuda.get_device_properties(0).total_memory
    n_renders = 0
    t0 = time.time()
    for cam in cams:
        torch.cuda.empty_cache()
        ci = all_cams.index(cam)
        spec = specs[ci]
        H, W = spec.image_height, spec.image_width
        g0 = dict(zip(("rgb", "depth", "valid"), load_gt(P["data"], cam, 0, H, W)[:3]))
        gT = dict(zip(("rgb", "depth", "valid"), load_gt(P["data"], cam, T, H, W)[:3]))
        if drange is None:
            v = torch.cat([g0["depth"][g0["valid"]], gT["depth"][gT["valid"]]])
            drange = (float(torch.quantile(v.float()[:500000], 0.005)),
                      float(torch.quantile(v.float()[:500000], 0.995)))
        V = {"F0": ec.render("F0", stF0, spec, 0, cam), "B0": ec.render("B0", stB0, spec, 0, cam),
             "FT": ec.render("FT", stFT, spec, T, cam), "BT": ec.render("BT", stBT, spec, T, cam)}
        n_renders += 4
        for k, v in V.items():
            cons_all[f"{k}/{cam}"] = ec.conservation(v)
        if stFT_fixed is not None:      # what turning the splats changes, against the recorded RGB-D
            for nm, st_t, st_f, g_, f_ in (("FT", stFT, stFT_fixed, gT, T), ("B0", stB0, stB0_fixed, g0, 0)):
                for mode_, d_ in (("track_kabsch", V[nm].d),
                                  ("fixed", build_linearization(st_f, [spec]).views[0])):
                    d_ = d_ if isinstance(d_, dict) else {"rgb": d_.rgb, "depth": d_.depth, "alpha": d_.alpha}
                    m_ = g_["valid"] & (d_["alpha"] >= th.alpha_cover)
                    dz = (d_["depth"] - g_["depth"]).abs()[m_]
                    orient_rows.append({"state": nm, "frame": f_, "camera": cam, "orientation": mode_,
                                        "covered_valid_px": int(m_.sum()),
                                        "rgb_rmse": float(((d_["rgb"] - g_["rgb"]) ** 2)[..., m_].mean().sqrt())
                                        if m_.any() else float("nan"),
                                        "depth_mae_m": float(dz.mean()) if m_.any() else float("nan"),
                                        "depth_within_tol_frac": float((dz <= th.depth_tol_m).float().mean())
                                        if m_.any() else float("nan")})
                n_renders += 1
        gS = ec.static_observation(g0, gT, th.depth_tol_m)
        fields = {"A": pe.classify(V["F0"].d, V["B0"].d, g0, th),
                  "B": pe.classify(V["FT"].d, V["BT"].d, gT, th),
                  "C": pe.classify(V["F0"].d, V["BT"].d, gS, th)}
        gts = {"A": g0, "B": gT, "C": gS}
        pairs_src = {"A": ("F0", "B0"), "B": ("FT", "BT"), "C": ("F0", "BT")}
        free = None
        for ct, (kf, kb) in pairs_src.items():
            f_ = fields[ct]
            m = pe.view_metrics(f_, V[kf].d, V[kb].d, th)
            view_rows.append({"contrast": ct, "frame": {"A": 0, "B": T, "C": f"0|{T}"}[ct], "camera": cam,
                              "F_state": kf, "B_state": kb,
                              **{k: v for k, v in m.items() if not isinstance(v, list)},
                              "F_depth_err_q10_q50_q90": m["F_depth_err_q10_q50_q90__covered"],
                              "B_depth_err_q10_q50_q90": m["B_depth_err_q10_q50_q90__covered"]})
            ge.accumulate(acc[ct + "F"], V[kf].contrib, V[kf].lin.cache.depth, f_, gts[ct]["depth"],
                          "F", ci, tol)
            ge.accumulate(acc[ct + "B"], V[kb].contrib, V[kb].lin.cache.depth, f_, gts[ct]["depth"],
                          "B", ci, tol)
            torch.cuda.empty_cache()
            if budget_free is not None:
                free = max(budget_free - torch.cuda.memory_allocated(), 2 ** 27)
            else:
                free = torch.cuda.mem_get_info()[0]
            # only the overlap sums this analysis reads (2 sparse products per view);
            # the other SUM_COLS are filled with zeros and never used
            vp = pv.view_pairs(V[kf].contrib, V[kb].contrib, V[kf].lin.cache.depth, V[kb].lin.cache.depth,
                               V[kf].state.colors.float(), V[kb].state.colors.float(),
                               V[kf].state.mu.float(), V[kb].state.mu.float(),
                               gts[ct]["depth"], gts[ct]["valid"], F.N, B.N, th,
                               prune_tau=RULES["PRUNE_TAU"], only_cols=PAIR_COLS,
                               block_budget=0.15 * free / 120.0)
            for c_ in pv.SUM_COLS:
                if c_ not in vp["cols"]:
                    vp["cols"][c_] = np.zeros(vp["keys"].size, np.float32)
            pair_views[ct].append((ci, vp))
            # sensitivity sweep of the depth tolerance on the explanation categories
            for dt in SWEEP_DEPTH_M:
                ff = pe.classify(V[kf].d, V[kb].d, gts[ct], pe.Thresholds(depth_tol_m=dt))
                sweep.append({"contrast": ct, "camera": cam, "depth_tol_m": dt,
                              **{n: int((ff["support"] == c).sum()) for c, n in pe.SUPPORT_CODES.items()},
                              "same_surface": int((ff["layer"] == 1).sum()),
                              "different_depth": int((ff["layer"] == 2).sum())})
        # J1 views
        jS = {}
        for d, sj in stJ.items():
            vFJ = ec.render("FT_J", sj["FT_J"], spec, T, cam)
            vBJ = ec.render("B0_J", sj["B0_J"], spec, 0, cam)
            n_renders += 2
            fBJ = pe.classify(vFJ.d, V["BT"].d, gT, th)
            fAJ = pe.classify(V["F0"].d, vBJ.d, g0, th)
            for ct, ff, kf, kb, a, b in (("B_J1_" + d, fBJ, "FT_J", "BT", vFJ, V["BT"]),
                                          ("A_J1_" + d, fAJ, "F0", "B0_J", V["F0"], vBJ)):
                m = pe.view_metrics(ff, a.d, b.d, th)
                view_rows.append({"contrast": ct, "frame": T if ct.startswith("B") else 0, "camera": cam,
                                  "F_state": kf, "B_state": kb,
                                  **{k: v for k, v in m.items() if not isinstance(v, list)}})
            jS[d] = {"FT_J_Sd": fBJ["SdF"].cpu().numpy(), "B0_J_Sd": fAJ["SdB"].cpu().numpy(),
                     "FT_J_depth": vFJ.lin.depth.cpu().numpy(), "FT_J_rgb": vFJ.lin.rgb.cpu().numpy(),
                     "B0_J_depth": vBJ.lin.depth.cpu().numpy(), "B0_J_rgb": vBJ.lin.rgb.cpu().numpy(),
                     "FT_J_alpha": vFJ.lin.alpha.cpu().numpy(), "B0_J_alpha": vBJ.lin.alpha.cpu().numpy()}
            del vFJ, vBJ
        # pixel maps (A, B, C layers) and CPU copies for the patch analysis
        for ct, (kf, kb), frame_dir in (("A", ("F0", "B0"), "frame_0000"),
                                        ("B", ("FT", "BT"), f"frame_{T:04d}"),
                                        ("C", ("F0", "BT"), "static_F0_vs_BT")):
            f_ = fields[ct]
            fn = {k: f_[k].cpu().numpy() for k in ("cover", "support", "layer", "eF", "eB", "dFB",
                                                    "covF", "covB", "W_Bonly", "W_Fonly")}
            Fn = {k: v.cpu().numpy() for k, v in V[kf].d.items()}
            Bn = {k: v.cpu().numpy() for k, v in V[kb].d.items()}
            gn = {k: v.cpu().numpy() for k, v in gts[ct].items()}
            imgs = vz.view_maps(fn, Fn, Bn, gn, drange)
            imgs["GT_RGB"] = vz._to_u8(gn["rgb"].transpose(1, 2, 0))
            imgs["F_RGB"] = vz._to_u8(Fn["rgb"].transpose(1, 2, 0))
            imgs["B_RGB"] = vz._to_u8(Bn["rgb"].transpose(1, 2, 0))
            mdir = out / "pixel_maps" / frame_dir / cam
            for name, img in imgs.items():
                safe_name = ("observed_" + name[3:] if name.upper().startswith("GT_") else name)
                vz.save_png(mdir / f"{safe_name}.png", img)
            (mdir / "states.json").write_text(json.dumps({
                "F": kf, "B": kb, "observation": {"A": "frame 0", "B": f"frame {T}",
                                                  "C": f"frame {T} depth on pixels unchanged since frame 0"}[ct],
                "legend": "see observation_diagnostics/visualize.LEGENDS; F_* files are the F state, B_* the B state"}))
            np.savez_compressed(mdir / "numeric.npz", cover=fn["cover"], support=fn["support"],
                                layer=fn["layer"], F_depth=Fn["depth"], B_depth=Bn["depth"],
                                F_alpha=Fn["alpha"], B_alpha=Bn["alpha"],
                                note=f"F = {kf}, B = {kb}; codes as in observation_diagnostics.pixel_evidence")
        per_cam[cam] = {
            "HW": (H, W),
            "g0": {k: v.cpu().numpy() for k, v in g0.items()},
            "gT": {k: v.cpu().numpy() for k, v in gT.items()},
            "gS_valid": gS["valid"].cpu().numpy(),
            "W": {ct: {w: fields[ct][w].cpu().numpy() for w in ("W_Bonly", "W_Fonly", "SdF", "SdB")}
                  for ct in ("A", "B")},
            "contrib": {k: to_np_contrib(v.contrib) for k, v in V.items()},
            "z": {k: v.lin.cache.depth.cpu().numpy() for k, v in V.items()},
            "img": {k: (v.lin.rgb.cpu().numpy(), v.lin.depth.cpu().numpy(), v.lin.alpha.cpu().numpy())
                    for k, v in V.items()},
            "J1": jS}
        del V, fields
        torch.cuda.empty_cache()
        log(f"  {cam}: 4 endpoint renders" + (f" + {2 * len(stJ)} J1" if stJ else "")
            + f", {time.time() - t0:.0f} s, GPU peak {peak()['peak_gpu_mib']} MiB")
    # ---- totals -----------------------------------------------------------
    A_ = {k: ge.to_numpy(v) for k, v in acc.items()}
    tables = {ct: pv.frame_table(v) for ct, v in pair_views.items()}
    pair_stats = {ct: {"pair_records": sum(v["stats"]["pairs"] for _c, v in vs),
                       "stored_pairs": int(tables[ct]["keys"].size),
                       "omitted_mass": sum(v["stats"]["omitted_mass"] for _c, v in vs),
                       "pair_mass": sum(v["stats"]["pair_mass"] for _c, v in vs)}
                  for ct, vs in pair_views.items()}
    omitted = {ct: {s: sum(v[f"omitted_O_surf_{s}"] for _c, v in vs) for s in ("F", "B")}
               for ct, vs in pair_views.items()}
    del pair_views
    log(f"  endpoint pass: {n_renders} renders in {time.time() - t0:.0f} s; pairs "
        + ", ".join(f"{ct} {s['stored_pairs']:,} stored of {s['pair_records']:,} records"
                    for ct, s in pair_stats.items()))
    write_csv(out / "endpoint_views.csv", [observed_names(r) for r in view_rows])
    write_csv(out / "depth_tolerance_sweep.csv", [observed_names(r) for r in sweep])
    if orient_rows:
        write_csv(out / "transport_orientation_effect.csv", orient_rows)
    (out / "data_dictionary.json").write_text(json.dumps(DATA_DICTIONARY, indent=1))
    endpoint = {"A": 0, "B": T}
    pos = {"F0": F.mu[0], "FT": F.mu[T], "B0": B.mu[0], "BT": B.mu[T]}
    scl = {"F": F.base.scales.max(1).values.cpu().numpy(), "B": B.base.scales.max(1).values.cpu().numpy()}
    col = {"F": F.base.colors.cpu().numpy(), "B": B.base.colors.cpu().numpy()}
    opa = {"F": F.base.opacity.reshape(-1).cpu().numpy(), "B": B.base.opacity.reshape(-1).cpu().numpy()}
    smin = {"F": F.base.scales.min(1).values.cpu().numpy(), "B": B.base.scales.min(1).values.cpu().numpy()}
    track = {"F": {"net": np.linalg.norm(F.mu[-1] - F.mu[0], axis=1),
                   "path": np.linalg.norm(np.diff(F.mu, axis=0), axis=2).sum(0)},
             "B": {"net": np.linalg.norm(B.mu[-1] - B.mu[0], axis=1),
                   "path": np.linalg.norm(np.diff(B.mu, axis=0), axis=2).sum(0)}}
    NB = B.N

    def pair_rows(ct, side, rows):
        """Rows of the contrast table whose `side` row is in `rows` (bool mask)."""
        t = tables[ct]
        i, j = t["keys"] // NB, t["keys"] % NB
        own, oth = (i, j) if side == "F" else (j, i)
        m = rows[own]
        return {c: t[c][m] for c in pv.SUM_COLS} | {"own": own[m], "other": oth[m]}

    # ---- static common candidates (contrast C) -----------------------------
    log("--- static common candidates (F0 vs BT, world coordinates)")
    tC = tables["C"]
    iC, jC = tC["keys"] // NB, tC["keys"] % NB
    q2F, q2B = A_["CF"]["q2_kappa_1"].astype(np.float64), A_["CB"]["q2_kappa_1"].astype(np.float64)
    r_match = max(2.0 * float(np.median(np.r_[scl["F"], scl["B"]])), 0.02)
    staticF = (track["F"]["net"] <= RULES["STATIC_NET_M"]) & (track["F"]["path"] <= RULES["STATIC_PATH_M"])
    staticB = (track["B"]["net"] <= RULES["STATIC_NET_M"]) & (track["B"]["path"] <= RULES["STATIC_PATH_M"])
    supC = A_["CF"]["mass_supported"] >= RULES["S_MIN"]
    sc_ = ec.static_candidates(tC["keys"], tC["O_surf_1"], tC["dist3d_surf"], q2F, q2B, NB, F.N,
                               supC, staticF, staticB, RULES, r_match)
    bestB, best_rho, best_dist, tier = sc_["best_B"], sc_["rho"], sc_["dist"], sc_["tier"]
    cand = sc_["world"]
    j1_moving_F = np.zeros(F.N, bool)
    for d, j in J1.items():
        j1_moving_F[j["owned_F"]] = True
    vox = np.floor(F.mu[0] / RULES["STATIC_VOXEL_M"]).astype(np.int64)
    rows_c = np.flatnonzero(cand)
    static_rows = []
    cdir = out / "candidates"
    cdir.mkdir(exist_ok=True)
    if rows_c.size:
        keys = [tuple(v) for v in vox[rows_c]]
        groups = {}
        for r, k in zip(rows_c, keys):
            groups.setdefault(k, []).append(r)
        for gi, (k, rs) in enumerate(sorted(groups.items(), key=lambda kv: -len(kv[1]))):
            rs = np.asarray(rs)
            bs = np.unique(bestB[rs])
            pid = f"static_{gi:04d}"
            tiers = dict(zip(*[x.tolist() for x in np.unique(tier[rs].astype(str), return_counts=True)]))
            rec = {"patch_id": pid, "voxel": list(map(int, k)), "n_F_rows": int(rs.size),
                   "tier_counts": tiers,
                   "tier": max(tiers, key=tiers.get),
                   "n_B_rows": int(bs.size), "centre_F0": F.mu[0][rs].mean(0).round(4).tolist(),
                   "median_rho_surf": float(np.median(best_rho[rs])),
                   "median_world_distance_m": float(np.median(best_dist[rs])),
                   "F_track_net_median_m": float(np.median(track["F"]["net"][rs])),
                   "B_track_net_median_m": float(np.median(track["B"]["net"][bs])),
                   "observed_static_pixels_supported_mass_F": float(A_["CF"]["mass_supported"][rs].sum()),
                   "rgb_compatible_fraction": float(np.median(np.divide(
                       tC["O_surf_rgb"][np.isin(iC, rs)], np.maximum(tC["O_surf_1"][np.isin(iC, rs)], 1e-30)))),
                   "J0": max(tiers, key=tiers.get),
                   "J1": ("UNAVAILABLE" if not J1 else
                          "CONFLICT: rows owned by an estimated moving part" if j1_moving_F[rs].mean() > 0.5
                          else "CONSISTENT: not owned by an estimated moving part"),
                   "F_rows": f"candidates/{pid}.json", "B_rows": f"candidates/{pid}.json"}
            static_rows.append(rec)
            (cdir / f"{pid}.json").write_text(json.dumps({"F_rows": rs.tolist(), "B_rows": bs.tolist(),
                                                          "best_B_for_F": bestB[rs].tolist(),
                                                          "rho_surf": best_rho[rs].round(4).tolist(),
                                                          "world_distance_m": best_dist[rs].round(5).tolist()}))
    write_csv(out / "static_common_candidates.csv", static_rows)
    log(f"  {int(cand.sum())} F rows world-matched to BT ({int(sc_['full'].sum())} with both free tracks "
        f"static) in {len(static_rows)} voxel patches; match radius {r_match:.3f} m")
    # ---- complementary patches -------------------------------------------
    log("--- complementary patches (J0 and J1 side by side)")
    depth_cache = {}

    def gt_depth(cam, f):
        k = (cam, f)
        if k not in depth_cache:
            if f == 0:
                depth_cache[k] = (per_cam[cam]["g0"]["depth"], per_cam[cam]["g0"]["valid"])
            elif f == T:
                depth_cache[k] = (per_cam[cam]["gT"]["depth"], per_cam[cam]["gT"]["valid"])
            else:
                H, W = per_cam[cam]["HW"]
                _r, d, v, _ = load_gt(P["data"], cam, f, H, W, device="cpu")
                depth_cache[k] = (d.numpy(), v.numpy())
        return depth_cache[k]

    coh = {}
    for k_, src_ in (("F", F), ("B", B)):
        sp = P[k_]["support"]
        if sp.is_file():
            d_ = json.loads(sp.read_text())
            lab_ = np.full(src_.N, "absent", dtype=object)
            lab_[np.asarray(d_.get("static_gaussians", []), dtype=np.int64)] = "static"
            lab_[np.asarray(d_.get("dynamic_gaussians", []), dtype=np.int64)] = "dynamic"
            coh[k_] = lab_
    patch_out = {"B": [], "F": []}
    ambiguous, jcmp, all_patch_rows = [], [], {"F": set(), "B": set()}
    counterpart_rows = {"F": set(), "B": set()}
    # per-row sums of the endpoint pair tables, for screening every patch cheaply
    rowsum = {}
    for ct_, t_ in tables.items():
        i_, j_ = t_["keys"] // NB, t_["keys"] % NB
        rowsum[ct_] = {side: {c: np.bincount(own_, weights=t_[c].astype(np.float64),
                                             minlength=(F.N if side == "F" else B.N))
                              for c in ("O", "O_surf_1", "O_incompat", "O_surf_rgb")}
                       for side, own_ in (("F", i_), ("B", j_))}
    from scipy.spatial import cKDTree
    trees = {"F0": cKDTree(pos["F0"]), "BT": cKDTree(pos["BT"])}
    for tname, ty in TYPES.items():
        s, ct = ty["s"], ty["contrast"]
        o = "F" if s == "B" else "B"
        src = B if s == "B" else F
        t_end = endpoint[ct]
        t_oth = T if t_end == 0 else 0
        ct_oth = "B" if ct == "A" else "A"
        accs = A_[ct + s]
        accs_oth = A_[ct_oth + s]
        U = accs["witness"].astype(np.float64)
        rows = np.flatnonzero(U >= RULES["U_MIN"])
        xyz = pos[ty["own"]]
        groups, r_link = ec.group_rows(xyz, rows, scl[s])
        groups.sort(key=lambda g: -U[g].sum())
        # ---- stage 1: screen EVERY patch with per-row aggregates ----------
        own_oth_key_ = ("B0" if s == "B" else "F0") if t_oth == 0 else ("BT" if s == "B" else "FT")
        screen = []
        for g in groups:
            rs_t, rs_o, rs_c = rowsum[ct][s], rowsum[ct_oth][s], rowsum["C"][s]
            mk_ = float(accs["mass_kappa"][g].sum())
            mko_ = float(accs_oth["mass_kappa"][g].sum())
            mkc_ = float(A_["C" + s]["mass_kappa"][g].sum())
            vis_ = {}
            for cam in cams:
                d_, v_ = gt_depth(cam, t_oth)
                lab_ = ec.visibility_by_depth(specs[all_cams.index(cam)], pos[own_oth_key_][g], d_, v_, tol)
                for k_ in sorted(set(lab_)):
                    vis_[k_] = vis_.get(k_, 0) + int((lab_ == k_).sum())
            nv_ = sum(vis_.values())
            dd_, _ = trees["F0" if o == "F" else "BT"].query(pos["F0" if s == "F" else "BT"][g], k=1)
            m_ = dict(n_pix=int(accs["witness_records"][g].sum()),
                      n_cam=int(pv.popcount(np.array([np.bitwise_or.reduce(accs["cams_witness"][g])]))[0]),
                      frac_same=float(rs_t["O_surf_1"][g].sum()) / mk_ if mk_ > 0 else 0.0,
                      rgb_ratio=float(rs_t["O_surf_rgb"][g].sum()) / max(float(rs_t["O_surf_1"][g].sum()), 1e-30),
                      frac_other=float(rs_o["O_surf_1"][g].sum()) / mko_ if mko_ > 0 else 0.0,
                      is_static=bool(np.median(track[s]["net"][g]) <= RULES["STATIC_NET_M"]
                                     and np.median(track[s]["path"][g]) <= RULES["STATIC_PATH_M"]),
                      frac_static=float(rs_c["O_surf_1"][g].sum()) / mkc_ if mkc_ > 0 else 0.0,
                      excl=float(U[g].sum()) / max(float(accs["mass_supported"][g].sum()), 1e-30),
                      unseen=(vis_.get("OCCLUDED", 0) + vis_.get("OUT_OF_VIEW", 0) + vis_.get("GT_INVALID", 0)) / max(nv_, 1),
                      sup_o=float(accs_oth["mass_supported"][g].sum()),
                      world_near=float(np.mean(dd_ <= r_match)))
            lab_s, _w = ec.j0_label(RULES, s=s, o=o, fail=ty["fail"], fail_transported=ty["fail_transported"],
                                    t_end=t_end, t_oth=t_oth, **m_)
            screen.append((lab_s, m_))
        nmax = RULES["MAX_DETAILED_PATCHES_PER_TYPE"]
        detailed = set(range(min(nmax, len(groups))))
        detailed |= set([k for k, (lab_s, _m) in enumerate(screen) if lab_s.endswith("PROPOSAL")][:nmax])
        screen_counts = {}
        for lab_s, _m in screen:
            screen_counts[lab_s] = screen_counts.get(lab_s, 0) + 1
        log(f"  {tname}: {rows.size} rows with witness U >= {RULES['U_MIN']} in {len(groups)} patches "
            f"(link radius {r_link:.3f} m); screened labels {json.dumps(screen_counts)}; "
            f"detailed {len(detailed)} (top {nmax} by U + top {nmax} proposals)")
        for gi, g in enumerate(groups):
            pid = f"{tname}_{gi:03d}"
            all_patch_rows[s].update(g.tolist())
            mask = np.zeros(src.N, bool)
            mask[g] = True
            base = {"patch_id": pid, "type": tname, "endpoint_frame": t_end, "source": s,
                    "n_rows": int(g.size), "witness_U": float(U[g].sum()),
                    "centre": xyz[g].mean(0).round(4).tolist(),
                    "rows": f"candidates/{pid}.json"}
            base["screen_J0_label"] = screen[gi][0]
            base["screen_metrics"] = {k: (round(v, 4) if isinstance(v, float) else v)
                                      for k, v in screen[gi][1].items()}
            base["screen_note"] = ("per-row aggregates; n_pix is the witness-record count, an upper "
                                   "bound of distinct witness pixels")
            if gi not in detailed:
                base["detail"] = "SCREENED_ONLY (not in the top-N by U or top-N proposals)"
                (cdir / f"{pid}.json").write_text(json.dumps(jsonable(base | {"rows_list": g.tolist()})))
                patch_out[s].append(base)
                continue
            # witness pixels per camera
            wit, mass_maps = {}, {}
            for cam in cams:
                pc = per_cam[cam]
                g_ = pc["g0"] if t_end == 0 else pc["gT"]
                mm = patch_mass(pc["contrib"][ty["own"]], mask, pc["z"][ty["own"]], g_["depth"],
                                g_["valid"], tol, pc["W"][ct][ty["W"]], pc["HW"][0] * pc["HW"][1])
                nz = np.flatnonzero(mm > 0)
                if nz.size:
                    W_ = pc["HW"][1]
                    top = nz[np.argsort(-mm[nz])][:50]
                    wit[cam] = {"pixels": int(nz.size), "mass": float(mm[nz].sum()),
                                "bbox": [int((nz % W_).min()), int((nz // W_).min()),
                                         int((nz % W_).max()), int((nz // W_).max())],
                                "top_witnesses_frame_cam_u_v_mass": [[t_end, cam, int(p % W_), int(p // W_),
                                                                      round(float(mm[p]), 5)] for p in top]}
                    mass_maps[cam] = mm
            n_pix = sum(w["pixels"] for w in wit.values())
            n_cam = len(wit)
            best_cam = max(wit, key=lambda c: wit[c]["mass"]) if wit else None
            # same-endpoint counterpart (J0)
            pr = pair_rows(ct, s, mask)
            mk = float(accs["mass_kappa"][g].sum())
            frac_same = float(pr["O_surf_1"].sum()) / mk if mk > 0 else 0.0
            frac_incompat = float(pr["O_incompat"].sum()) / max(float(pr["O"].sum()), 1e-30)
            rgb_ratio = float(pr["O_surf_rgb"].sum()) / max(float(pr["O_surf_1"].sum()), 1e-30)
            partners = {}
            if pr["other"].size:
                agg = np.bincount(pr["other"], weights=pr["O_surf_1"], minlength=(B.N if o == "B" else F.N))
                topp = np.argsort(-agg)[:20]
                partners = {int(r): float(agg[r]) for r in topp if agg[r] > 0}
            counterpart_rows[o].update(partners.keys())
            # other endpoint under the own Phase-1 track (J0 transport)
            mk_o = float(accs_oth["mass_kappa"][g].sum())
            sup_o = float(accs_oth["mass_supported"][g].sum())
            pr_o = pair_rows(ct_oth, s, mask)
            frac_other = float(pr_o["O_surf_1"].sum()) / mk_o if mk_o > 0 else 0.0
            own_oth_key = ("B0" if s == "B" else "F0") if t_oth == 0 else ("BT" if s == "B" else "FT")
            vis = {}
            for cam in cams:
                d_, v_ = gt_depth(cam, t_oth)
                lab = ec.visibility_by_depth(specs[all_cams.index(cam)], pos[own_oth_key][g], d_, v_, tol)
                for k_ in sorted(set(lab)):
                    vis[k_] = vis.get(k_, 0) + int((lab == k_).sum())
            nv = sum(vis.values())
            unseen = (vis.get("OCCLUDED", 0) + vis.get("OUT_OF_VIEW", 0) + vis.get("GT_INVALID", 0)) / max(nv, 1)
            # static bypass (contrast C)
            is_static = bool(np.median(track[s]["net"][g]) <= RULES["STATIC_NET_M"]
                             and np.median(track[s]["path"][g]) <= RULES["STATIC_PATH_M"])
            prC = pair_rows("C", s, mask)
            mkC = float(A_["C" + s]["mass_kappa"][g].sum())
            frac_static = float(prC["O_surf_1"].sum()) / mkC if mkC > 0 else 0.0
            anchored_other = pos["F0"] if o == "F" else pos["BT"]
            anchored_own = pos["F0"] if s == "F" else pos["BT"]
            from scipy.spatial import cKDTree
            dd, nn = cKDTree(anchored_other).query(anchored_own[g], k=1)
            world_near = float(np.mean(dd <= r_match))
            # appearance / shape against same-endpoint partners (else world neighbours)
            cp = np.array(list(partners.keys())) if partners else np.unique(nn[dd <= r_match])
            app = {"patch_colour": col[s][g].mean(0).round(3).tolist(),
                   "patch_opacity_mean": float(opa[s][g].mean()),
                   "patch_scale_max_median_m": float(np.median(scl[s][g])),
                   "patch_anisotropy_median": float(np.median(scl[s][g] / np.maximum(smin[s][g], 1e-9)))}
            if cp.size:
                app |= {"counterpart_colour": col[o][cp].mean(0).round(3).tolist(),
                        "counterpart_opacity_mean": float(opa[o][cp].mean()),
                        "counterpart_scale_max_median_m": float(np.median(scl[o][cp])),
                        "colour_rms_difference": float(np.sqrt(((col[s][g].mean(0) - col[o][cp].mean(0)) ** 2).mean()))}
            # J1 per usable estimate
            j1 = {}
            for d, j in J1.items():
                owned = j["owned_B"] if s == "B" else j["owned_F"]
                direct = (j["own_source"] == s)
                frac_owned = float(np.isin(g, owned).mean())
                rec = {"estimate": d, "ownership": "estimator support" if direct else "inferred by proximity",
                       "patch_rows_owned_fraction": frac_owned}
                if ty["fail_transported"]:
                    key = "FT_J_Sd" if ty["fail"] == "FT" else "B0_J_Sd"
                    num = den = 0.0
                    for cam, mm in mass_maps.items():
                        S = per_cam[cam]["J1"][d][key]
                        num += float((mm * S.reshape(-1)).sum())
                        den += float(mm.sum())
                    rec["witness_explained_by_J1_transport_of_the_other_source"] = num / den if den > 0 else None
                    rec["witness_explained_by_J0_transport"] = 0.0
                    rec["note"] = ("J0 is 0 by construction: these are pixels the J0 transported "
                                   f"{ty['fail']} does not explain")
                else:
                    rec["witness_explained_by_J1_transport_of_the_other_source"] = None
                    rec["note"] = f"the failing render {ty['fail']} is an anchor; J1 does not move it"
                if frac_owned > 0 and t_end == T and s == "B":
                    pred = apply(est[d].transform(T, 0), pos["BT"][g[np.isin(g, owned)]])
                    dd1, _ = cKDTree(pos["F0"]).query(pred, k=1)
                    rec["J1_predicted_frame0_positions_with_F0_row_within_r"] = float(np.mean(dd1 <= r_match))
                if frac_owned > 0 and t_end == 0 and s == "F":
                    pred = apply(est[d].transform(0, T), pos["F0"][g[np.isin(g, owned)]])
                    dd1, _ = cKDTree(pos["BT"]).query(pred, k=1)
                    rec["J1_predicted_frameT_positions_with_BT_row_within_r"] = float(np.mean(dd1 <= r_match))
                rec["label"] = ec.j1_label(RULES, rec["witness_explained_by_J1_transport_of_the_other_source"])
                if not direct or frac_owned < 0.5:
                    rec["confidence"] = "LOW (ownership inferred or partial)"
                j1[d] = rec
            # J0 label (declared order; see RULES)
            sup_t = float(accs["mass_supported"][g].sum())
            excl = float(U[g].sum()) / sup_t if sup_t > 0 else 1.0
            lab0, why = ec.j0_label(RULES, s=s, o=o, fail=ty["fail"], fail_transported=ty["fail_transported"],
                                    t_end=t_end, t_oth=t_oth, n_pix=n_pix, n_cam=n_cam,
                                    frac_same=frac_same, rgb_ratio=rgb_ratio, frac_other=frac_other,
                                    is_static=is_static, frac_static=frac_static, excl=excl,
                                    unseen=unseen, sup_o=sup_o, world_near=world_near)
            why = [why]
            rec_full = base | {
                "witness_pixels": n_pix, "witness_cameras": n_cam, "best_camera": best_cam,
                "exclusive_fraction": excl, "supported_mass_at_endpoint": sup_t,
                "same_endpoint_counterpart_fraction": frac_same,
                "same_endpoint_incompatible_depth_fraction": frac_incompat,
                "same_endpoint_rgb_compatible_ratio": rgb_ratio,
                "other_endpoint_frame": t_oth, "other_endpoint_own_supported_mass": sup_o,
                "other_endpoint_counterpart_fraction": frac_other,
                "other_endpoint_visibility_counts": vis, "other_endpoint_unseen_fraction": unseen,
                "static_by_own_track": is_static, "static_common_fraction_C": frac_static,
                "world_neighbour_fraction_anchored_other": world_near,
                "coherence_annotation": ({k: int(v) for k, v in zip(*np.unique(coh[s][g].astype(str), return_counts=True))}
                                         if s in coh else None),
                "coherence_note": "predicted_support.json: an uncertain annotation, never a gate",
                "J0_label": lab0, "why": "; ".join(why),
                "J1": j1, **app}
            patch_out[s].append(rec_full)
            for d, r in j1.items():
                jcmp.append({"patch_id": pid, "type": tname, "J0_label": lab0, "estimate": d,
                             "J1_label": r["label"], "ownership": r["ownership"],
                             "owned_fraction": r["patch_rows_owned_fraction"],
                             "witness_explained_J0": r.get("witness_explained_by_J0_transport"),
                             "witness_explained_J1": r["witness_explained_by_J1_transport_of_the_other_source"],
                             "confidence": r.get("confidence", "as estimated")})
            if not j1:
                jcmp.append({"patch_id": pid, "type": tname, "J0_label": lab0, "estimate": None,
                             "J1_label": jprov["j1_status"]})
            if lab0 in ("CONFLICT_AMBIGUOUS", "INSUFFICIENT_EVIDENCE", "APPEARANCE_SHAPE_DISAGREEMENT") or \
                    any(r["label"] == "J1_TRANSPORT_EXPLAINS" for r in j1.values()) and lab0.endswith("PROPOSAL"):
                ambiguous.append({"patch_id": pid, "type": tname, "J0_label": lab0,
                                  "J1_labels": {d: r["label"] for d, r in j1.items()}, "why": "; ".join(why)})
            (cdir / f"{pid}.json").write_text(json.dumps(jsonable(rec_full | {
                "rows_list": g.tolist(), "witnesses": wit,
                "same_endpoint_partners_by_O_surf": partners})))
            rec_full["_mass_maps"] = mass_maps
            rec_full["_rows"] = g
    # ---- occlusion / intermediate frames for the strongest proposals ------
    log("--- intermediate-frame visibility for the strongest proposals")
    from estimator.renderer_bridge import build_linearization
    n_inter = 0
    for s in ("B", "F"):
        src = B if s == "B" else F
        done_by_type = {}
        for rec in patch_out[s]:
            if "J0_label" not in rec or not rec["J0_label"].endswith("PROPOSAL"):
                continue
            if done_by_type.get(rec["type"], 0) >= RULES["MAX_OCCLUSION_PATCHES_PER_TYPE"]:
                continue
            done_by_type[rec["type"]] = done_by_type.get(rec["type"], 0) + 1
            g = rec["_rows"]
            frac_vis = []
            for f in range(len(src.frames)):
                at = tot = 0
                for cam in cams:
                    d_, v_ = gt_depth(cam, f)
                    lab = ec.visibility_by_depth(specs[all_cams.index(cam)], src.mu[f][g], d_, v_, tol)
                    at += int((lab == "AT_OBSERVED_DEPTH").sum())
                    tot += int((lab != "OUT_OF_VIEW").sum())
                frac_vis.append(at / max(tot, 1))
            fv = np.asarray(frac_vis)
            # chosen by support: where the observed depth starts to agree with the patch
            t_end = rec["endpoint_frame"]
            order = range(len(fv)) if t_end == T else range(len(fv) - 1, -1, -1)
            reveal = next((f for f in order if fv[f] >= 0.5), None)
            chosen = []
            if reveal is None:
                # no frame reaches 50%: measure the frames where the observation
                # supports the patch best (still chosen from measured depth only)
                cand_f = [f for f in np.argsort(-fv) if f not in (0, T) and fv[f] > 0][:3]
                chosen = sorted(int(f) for f in cand_f)
            else:
                far_end = 0 if t_end == T else T
                prev = reveal - 1 if t_end == T else reveal + 1
                mid = (far_end + reveal) // 2
                chosen = sorted({f for f in (mid, prev, reveal, (reveal + t_end) // 2)
                                 if 0 <= f < len(fv) and f not in (0, T)})[:4]
            measured = []
            for f in chosen:
                best = sorted(cams, key=lambda c: -int((ec.visibility_by_depth(
                    specs[all_cams.index(c)], src.mu[f][g], *gt_depth(c, f), tol) == "AT_OBSERVED_DEPTH").sum()))[:2]
                for cam in best:
                    spec = specs[all_cams.index(cam)]
                    v = build_linearization(state_at(src, f), [spec]).views[0]
                    from observation_diagnostics.renderer_access import pixel_contributions
                    cn = to_np_contrib(pixel_contributions(v.cache))
                    d_, v_ = gt_depth(cam, f)
                    Sd = v_ & (v.alpha.cpu().numpy() >= th.alpha_cover) & \
                        (np.abs(v.depth.cpu().numpy() - d_) <= th.depth_tol_m)
                    mask = np.zeros(src.N, bool)
                    mask[g] = True
                    mm = patch_mass(cn, mask, v.lin.cache.depth.cpu().numpy() if hasattr(v, "lin") else
                                    v.cache.depth.cpu().numpy(), d_, v_, tol, Sd,
                                    d_.size)
                    measured.append({"frame": f, "camera": cam, "patch_supported_mass": float(mm.sum()),
                                     "patch_supported_pixels": int((mm > 0).sum())})
                    n_inter += 1
            rec["occlusion_test"] = {
                "visible_at_observed_depth_fraction_per_frame": fv.round(3).tolist(),
                "reveal_frame": reveal, "chosen_frames": chosen,
                "chosen_how": ("projected own-track positions vs MEASURED depth at every frame; "
                               "the frame where >= 50% of in-view projections first sit at the observed "
                               "depth (scanning from the unseen endpoint), the frame before it, the "
                               "midpoint to the unseen endpoint and to the witness endpoint; if no frame "
                               "reaches 50%, the (<= 3) frames with the highest fraction"),
                "measured": measured,
                "status": ((f"NO_REVEAL: never at the observed depth for half its in-view projections "
                            f"(best {fv.max():.2f}); measured at the best-supported frames")
                           if reveal is None else
                           "OCCLUSION_HYPOTHESIS: unseen at the other endpoint, measurable from frame "
                           f"{reveal}" if rec["other_endpoint_unseen_fraction"] >= RULES["UNSEEN_FRACTION"]
                           else "VISIBLE_AT_BOTH_ENDPOINTS")}
            pj = cdir / f"{rec['patch_id']}.json"
            d0 = json.loads(pj.read_text())
            d0["occlusion_test"] = rec["occlusion_test"]
            pj.write_text(json.dumps(jsonable(d0)))
    log(f"  {n_inter} intermediate renders")
    # ---- selective sensitivity probes and case studies ---------------------
    log("--- sensitivity probes and case studies")
    probes = 0
    for s in ("B", "F"):
        src, oth = (B, F) if s == "B" else (F, B)
        chosen_ids = []
        for tname_ in TYPES:
            recs = [r for r in patch_out[s] if r["type"] == tname_ and "J0_label" in r and r.get("best_camera")]
            k = RULES["MAX_PROBE_PATCHES_PER_TYPE"]
            top = recs[:k]
            props = [r for r in recs if r["J0_label"].endswith("PROPOSAL")][:k]
            conflict = [r for r in recs if r["J0_label"].endswith("PROPOSAL") and any(
                x["label"] == "J1_TRANSPORT_EXPLAINS" for x in r["J1"].values())][:2]
            for r in top + props + conflict:
                if r["patch_id"] not in chosen_ids:
                    chosen_ids.append(r["patch_id"])
        for rec in patch_out[s]:
            if rec["patch_id"] not in chosen_ids:
                continue
            ty = TYPES[rec["type"]]
            cam = rec["best_camera"]
            spec = specs[all_cams.index(cam)]
            t_end = rec["endpoint_frame"]
            pj = cdir / f"{rec['patch_id']}.json"
            dj = json.loads(pj.read_text())
            partners = [int(x) for x in dj.get("same_endpoint_partners_by_O_surf", {})]
            if not partners:
                from scipy.spatial import cKDTree
                anch_o = pos["F0"] if s == "B" else pos["BT"]
                anch_s = pos["BT"] if s == "B" else pos["F0"]
                dd, nn = cKDTree(anch_o).query(anch_s[rec["_rows"]], k=1)
                partners = np.unique(nn[dd <= r_match]).tolist()
            v_s = ec.render(ty["own"], state_at(src, t_end), spec, t_end, cam)
            v_o = ec.render(ty["fail"], state_at(oth, t_end), spec, t_end, cam)
            region = torch.as_tensor(rec["_mass_maps"][cam] > 0, device="cuda")
            g_ = per_cam[cam]["g0"] if t_end == 0 else per_cam[cam]["gT"]
            probe = ec.sensitivity_alignment(v_s, rec["_rows"], v_o, partners, region,
                                             torch.as_tensor(g_["valid"], device="cuda"))
            probe["counterpart_rows_source"] = ("same-endpoint O_surf partners" if dj.get(
                "same_endpoint_partners_by_O_surf") else f"anchored {('F0' if s == 'B' else 'BT')} rows within {r_match:.3f} m")
            dj["sensitivity_probe"] = probe
            rec["sensitivity_probe"] = probe
            probes += 1
            # case study: crops on the real sensor RGB
            pc = per_cam[cam]
            H, W = pc["HW"]
            box = dj["witnesses"][cam]["bbox"]
            gr = (pc["g0"] if t_end == 0 else pc["gT"])
            gt_rgb = vz._to_u8(gr["rgb"].transpose(1, 2, 0))
            mm = rec["_mass_maps"][cam].reshape(H, W)
            rgb_s, dep_s, al_s = pc["img"][ty["own"]]
            rgb_o, dep_o, al_o = pc["img"][ty["fail"]]
            panels = [("observed RGB", gt_rgb), ("observed depth", vz.depth_rgb(gr["depth"], gr["valid"], drange)),
                      (f"{ty['own']} RGB", vz._to_u8(rgb_s.transpose(1, 2, 0))),
                      (f"{ty['own']} depth", vz.depth_rgb(dep_s, al_s > 0.05, drange)),
                      (f"{ty['fail']} RGB", vz._to_u8(rgb_o.transpose(1, 2, 0))),
                      (f"{ty['fail']} depth", vz.depth_rgb(dep_o, al_o > 0.05, drange)),
                      ("witness (patch q on witness px)", ev.overlay(gt_rgb, mm, (255, 160, 0)))]
            for d in pc["J1"]:
                if ty["fail"] == "FT":
                    panels.append((f"FT_J1[{d}] depth", vz.depth_rgb(pc["J1"][d]["FT_J_depth"],
                                                                    pc["J1"][d]["FT_J_alpha"] > 0.05, drange)))
                elif ty["fail"] == "B0":
                    panels.append((f"B0_J1[{d}] depth", vz.depth_rgb(pc["J1"][d]["B0_J_depth"],
                                                                    pc["J1"][d]["B0_J_alpha"] > 0.05, drange)))
            rows_img = [(f"frame {t_end} {cam}", [(t, ev.crop(im, box)) for t, im in panels])]
            occ = rec.get("occlusion_test")
            if occ and occ.get("chosen_frames"):
                inter = []
                for f in occ["chosen_frames"][:4]:
                    v = build_linearization(state_at(src, f), [spec]).views[0]
                    d_, v_ = gt_depth(cam, f)
                    inter.append((f"*{s} track, frame {f}", ev.crop(vz.depth_rgb(v.depth.cpu().numpy(),
                                                                               v.alpha.cpu().numpy() > 0.05, drange), box)))
                    inter.append((f"*observed depth, frame {f}", ev.crop(vz.depth_rgb(d_, v_, drange), box)))
                rows_img.append(("intermediate frames", inter))
            pth = out / "visuals" / "case_studies" / f"{rec['patch_id']}.png"
            ev.case_sheet(pth, f"{args.scene} {rec['patch_id']}: J0 {rec['J0_label']}; "
                               + "; ".join(f"J1[{d}] {r['label']}" for d, r in rec["J1"].items()),
                          rows_img, note=rec["why"][:220])
            dj["case_study_png"] = str(pth.relative_to(out))
            rec["case_study_png"] = str(pth.relative_to(out))
            pj.write_text(json.dumps(jsonable(dj)))
            del v_s, v_o
            torch.cuda.empty_cache()
    log(f"  {probes} sensitivity probes")
    # ---- tables ----------------------------------------------------------
    def flat(rec):
        r = {k: v for k, v in rec.items() if not k.startswith("_") and k not in ("J1", "other_endpoint_visibility_counts",
                                                                                   "occlusion_test", "sensitivity_probe")}
        r["other_endpoint_visibility_counts"] = rec.get("other_endpoint_visibility_counts")
        r["J1_labels"] = {d: x["label"] for d, x in rec.get("J1", {}).items()} or jprov["j1_status"]
        r["occlusion_status"] = (rec.get("occlusion_test") or {}).get("status")
        r["sensitivity_diag_mean"] = (rec.get("sensitivity_probe") or {}).get("diag_mean")
        r["F_rows" if rec["source"] == "F" else "B_rows"] = rec["rows"]
        return r
    write_csv(out / "backward_only_observed_patches.csv", [flat(r) for r in patch_out["B"]])
    write_csv(out / "forward_only_observed_patches.csv", [flat(r) for r in patch_out["F"]])
    write_csv(out / "ambiguous_patches.csv", ambiguous)
    write_csv(out / "joint_agnostic_vs_joint_informed.csv", jcmp)
    # ---- accounting --------------------------------------------------------
    acct = {}
    for s, src in (("F", F), ("B", B)):
        cat = np.full(src.N, "NOT_VISIBLE_AT_EITHER_ENDPOINT", dtype=object)
        vis_any = (A_["A" + s]["mass_total"] + A_["B" + s]["mass_total"]) > 0
        cat[vis_any] = "VISIBLE_UNCLASSIFIED"
        if s == "F":
            cat[cand] = "STATIC_WORLD_MATCHED"
        else:
            cat[np.unique(bestB[cand])] = "STATIC_WORLD_MATCHED"
        cr = np.fromiter(counterpart_rows[s], np.int64) if counterpart_rows[s] else np.empty(0, np.int64)
        cat[cr] = "PATCH_COUNTERPART"
        pr_ = np.fromiter(all_patch_rows[s], np.int64) if all_patch_rows[s] else np.empty(0, np.int64)
        cat[pr_] = "COMPLEMENTARY_PATCH_MEMBER"
        u, c = np.unique(cat.astype(str), return_counts=True)
        acct[s] = {"rows": src.N, **dict(zip(u.tolist(), c.tolist()))}
        acct[s]["sum_check"] = int(sum(c)) == src.N
    # ---- 3-D pages and overview ---------------------------------------------
    rng = np.random.default_rng(0)

    def sub(xyz, n=12000):
        return xyz[rng.choice(len(xyz), size=min(n, len(xyz)), replace=False)]
    arrows = []
    for d, e in usable.items():
        ax = np.asarray(e.revolute["axis"], float)
        arrows.append((f"{d} revolute axis (estimated; residual {(e.revolute['residual'] or {}).get('unweighted_energy', float('nan')):.2e})",
                       e.revolute["axis_point"], ax, 0.6, "red"))
        arrows.append((f"{d} prismatic direction (estimated; residual {(e.prismatic['residual'] or {}).get('unweighted_energy', float('nan')):.2e})",
                       np.asarray(e.revolute["axis_point"], float), e.prismatic["direction"], 0.4, "purple"))
    layers = [(f"F0 (display subsample)", sub(pos["F0"]), "rgba(40,110,230,0.25)", 1.2),
              (f"BT (display subsample)", sub(pos["BT"]), "rgba(245,140,20,0.25)", 1.2),
              ("F0 rows: static common candidates", pos["F0"][cand], "rgb(0,150,0)", 2),
              ("their best BT partners", pos["BT"][np.unique(bestB[cand])] if cand.any() else np.empty((0, 3)),
               "rgb(120,0,160)", 2)]
    ev.plot3d(out / "visuals" / "static_correspondence_3d.html", f"{args.scene}: static F0-BT correspondence",
              layers, arrows, note="World coordinates, no transport: F0 and BT are the two trained anchors.")
    layers = [("F0 (display subsample)", sub(pos["F0"]), "rgba(40,110,230,0.2)", 1),
              ("BT (display subsample)", sub(pos["BT"]), "rgba(245,140,20,0.2)", 1)]
    for s in ("B", "F"):
        for rec in patch_out[s][:24]:
            if "J0_label" not in rec:
                continue
            g = rec["_rows"]
            layers.append((f"{rec['patch_id']} {rec['J0_label']}", pos[TYPES[rec['type']]['own']][g],
                           "rgb(255,60,0)" if s == "B" else "rgb(0,60,255)", 3))
    for d, j in J1.items():
        layers.append((f"J1[{d}] owned B rows at T", pos["BT"][j["owned_B"]], "rgba(150,0,0,0.5)", 1.5))
    ev.plot3d(out / "visuals" / "complementary_patches_3d.html", f"{args.scene}: complementary patches",
              layers, arrows, note="Patches at their witness endpoint; red B-sourced, blue F-sourced. "
                                   "Joint arrows are ESTIMATES (sign is conventional).")
    # endpoint comparison grid: two cameras with most GT-valid pixels
    best2 = sorted(cams, key=lambda c: -int(per_cam[c]["gT"]["valid"].sum()))[:2]
    grid = []
    for cam in best2:
        pc = per_cam[cam]
        for ct, (kf, kb), gk in (("A", ("F0", "B0"), "g0"), ("B", ("FT", "BT"), "gT")):
            gr = pc[gk]
            fdir = "frame_0000" if ct == "A" else f"frame_{T:04d}"
            from PIL import Image
            cat_img = np.asarray(Image.open(out / "pixel_maps" / fdir / cam / "support_categories.png"))
            lay_img = np.asarray(Image.open(out / "pixel_maps" / fdir / cam / "same_surface_candidates.png"))
            grid.append((f"{ct}: frame {endpoint[ct]} {cam}", [
                ("observed RGB", vz._to_u8(gr["rgb"].transpose(1, 2, 0))),
                ("observed depth", vz.depth_rgb(gr["depth"], gr["valid"], drange)),
                (f"{kf} depth", vz.depth_rgb(pc["img"][kf][1], pc["img"][kf][2] > 0.05, drange)),
                (f"{kb} depth", vz.depth_rgb(pc["img"][kb][1], pc["img"][kb][2] > 0.05, drange)),
                ("explanation (green both, blue F only, orange B only, red neither)", cat_img),
                ("depth layers (green same surface, magenta different)", lay_img)]))
    ev.endpoint_grid(out / "visuals" / "endpoint_comparisons.png", args.scene, grid, drange)
    # ---- summary ---------------------------------------------------------
    def pct(rows, key):
        n = sum(r["gt_valid"] for r in rows)
        return round(100 * sum(r[key] for r in rows) / max(n, 1), 2)
    contrast_summary = {}
    for ct in sorted({r["contrast"] for r in view_rows}):
        rs = [r for r in view_rows if r["contrast"] == ct]
        contrast_summary[ct] = {k: pct(rs, k) for k in ("BOTH_VALID_SUPPORT", "F_SUPPORTED_B_UNEXPLAINED",
                                                         "B_SUPPORTED_F_UNEXPLAINED", "NEITHER_VALID_SUPPORT",
                                                         "BOTH_SAME_OBSERVED_SURFACE_CANDIDATE",
                                                         "BOTH_COVER_DIFFERENT_DEPTH",
                                                         "BOTH_COVER_AGREE_OFF_OBSERVED_SURFACE",
                                                         "NEITHER_COVERS__on_gt_valid")} | {
            "observed_depth_valid_pixels": sum(r["gt_valid"] for r in rs)}
    labels, screened = {}, {}
    for s in ("B", "F"):
        for rec in patch_out[s]:
            if "J0_label" in rec:
                k = f"{rec['type']}:{rec['J0_label']}"
                labels[k] = labels.get(k, 0) + 1
            k = f"{rec['type']}:{rec.get('screen_J0_label')}"
            screened[k] = screened.get(k, 0) + 1
    worst_cons = {"max_abs_sum_q_minus_A": max(c["max_abs_sum_q_minus_A"] for c in cons_all.values()),
                  "max_abs_depth_recon_m": max(c["max_abs_depth_recon_m"] for c in cons_all.values()),
                  "final_T_bitwise_all": all(c["final_T_bitwise"] for c in cons_all.values()),
                  "early_terminated_pixels": sum(c["early_terminated_pixels"] for c in cons_all.values())}
    # in-run checks: adjoint / FD on a real view, repeatability
    spec0 = specs[all_cams.index(cams[0])]
    vA = ec.render("F0", stF0, spec0, 0, cams[0])
    vA2 = ec.render("F0", stF0, spec0, 0, cams[0])
    checks["FROZEN_LINEARIZATION"] = ec.adjoint_check(vA)
    checks["REPEATABILITY"] = {"same_view_rendered_twice_max_abs_rgb": float((vA.lin.rgb - vA2.lin.rgb).abs().max()),
                               "contributor_lists_identical": bool(torch.equal(vA.contrib.gid, vA2.contrib.gid)
                                                                  and torch.equal(vA.contrib.offsets, vA2.contrib.offsets))}
    del vA, vA2
    after = {str(p): ec.sha256_file(p) for p in prot}
    changed = [p for p in before if before[p] != after[p]]
    fwr = fw.report()
    flags = {
        "RENDER_AND_INDEXING": "PASS" if (all(v is True or not isinstance(v, bool) for v in
                                              checks["RENDER_AND_INDEXING"].values())
                                          and worst_cons["final_T_bitwise_all"]
                                          and worst_cons["max_abs_sum_q_minus_A"] < 5e-6) else "FAIL",
        "ESTIMATED_JOINT_PROVENANCE": jprov["j1_status"],
        "STATIC_ENDPOINT_CORRESPONDENCE": (f"{int(sc_['full'].sum())} F rows STATIC_COMMON_CANDIDATE, "
                                           f"{int(cand.sum() - sc_['full'].sum())} world-matched with a disagreeing "
                                           f"free track, in {len(static_rows)} voxel patches (candidates, not proof)"),
        "COMPLEMENTARY_GEOMETRY_EVIDENCE": "detailed " + json.dumps(labels) + "; screened (all patches) "
                                           + json.dumps(screened),
        "OCCLUSION_INFERENCE": f"{sum(1 for s in patch_out for r in patch_out[s] if r.get('occlusion_test'))} proposals tested at intermediate frames",
        "JOINT_AIDED_DIAGNOSIS": ("J1 compared for every detailed patch" if J1 else jprov["j1_status"]),
        "GT_FIREWALL": "PASS: no forbidden file opened" if not fwr["refused"] else f"REFUSED {fwr['refused']}",
        "SOURCE_IMMUTABILITY": "PASS" if not changed else f"FAIL {changed}",
        "STATE_MODE": ("CENTRES_FROM_PHASE1_TRACK + ORIENTATION_FROM_LOCAL_TRACK_ROTATION"
                       if rot_mode != "fixed" else F.state_mode)}
    summary = {"scene": args.scene, "endpoints": [0, T], "cameras": cams, "rows": {"F": F.N, "B": B.N},
               "flags": flags,
               "contrasts_pct_of_observed_depth_valid": {k: observed_names(v) for k, v in contrast_summary.items()},
               "patch_labels": labels,
               "screened_patch_labels_all_patches": screened,
               "static_patches": len(static_rows), "static_F_rows": int(cand.sum()),
               "accounting": acct, "pairs": pair_stats, "renderer_checks": worst_cons,
               "rules": RULES, "depth_tolerance_sweep": "depth_tolerance_sweep.csv",
               "renders": {"endpoint_and_J1": n_renders, "intermediate": n_inter, "probes": probes},
               "runtime_s": time.time() - t_start, **peak()}
    (out / "endpoint_summary.json").write_text(json.dumps(jsonable(summary), indent=1))
    tests = {"checks": jsonable(checks), "renderer_conservation_per_view": cons_all,
             "renderer_conservation_worst": worst_cons, "gt_firewall": fwr,
             "immutability": {"files": len(before), "changed": changed},
             "accounting": acct}
    (out / "tests_and_checks.json").write_text(json.dumps(jsonable(tests), indent=1))
    prov = {"scene": args.scene, "tag": args.tag, "code_sha256": code_hash(),
            "kernel_sha256": KERNEL_SHA256, "sources": {"F": F.describe(), "B": B.describe()},
            "camera_gauge": gauge, "thresholds": th.as_dict(), "rules": RULES,
            "protected_hashes_before": before, "protected_hashes_after": after,
            "gt_firewall": fwr, "gt_access": "DISABLED (audit hook); estimated joints only",
            "finished": time.strftime("%Y-%m-%d %H:%M:%S")}
    (out / "provenance.json").write_text(json.dumps(jsonable(prov), indent=1))
    imgs = [("endpoint comparisons", "endpoint_comparisons.png")] + [
        (r["patch_id"], r["case_study_png"].replace("visuals/", "")) for s in patch_out for r in patch_out[s]
        if r.get("case_study_png")]
    tabs = {"contrasts (% of pixels with a valid RECORDED depth)": [{"contrast": k, **observed_names(v)}
                                                                   for k, v in contrast_summary.items()],
            "estimated joints": [{"direction": d, "status": e.status, "notes": "; ".join(e.notes)[:300],
                                  **({"max_owned_displacement_m": round(e.motion["max_owned_displacement_m"], 4),
                                      "max_rotation_rad": round(e.motion["max_rotation_rad"], 4),
                                      "lower_residual": (e.residual_preference or {}).get("lower_residual"),
                                      "residual_ratio": round((e.residual_preference or {}).get("residual_ratio", float("nan")), 2)}
                                     if hasattr(e, "motion") else {})} for d, e in est.items()],
            "complementary patches (detailed)": [{k: (flat(r).get(k)) for k in (
                "patch_id", "type", "n_rows", "witness_pixels", "witness_cameras", "J0_label", "J1_labels",
                "occlusion_status", "why")} for s in patch_out for r in patch_out[s] if "J0_label" in r],
            "static common patches (top 20)": static_rows[:20],
            "row accounting": [{"source": s, **v} for s, v in acct.items()]}
    ev.overview(out / "visuals" / "overview.html", args.scene, summary, tabs, imgs)
    write_summary_md(out, summary, patch_out, est, jprov)
    log(f"=== done in {summary['runtime_s'] / 60:.1f} min: flags {json.dumps(flags)}")
    return 0 if not changed and not fwr["refused"] else 1


def write_summary_md(out, S, patch_out, est, jprov):
    L = [f"# {S['scene']}: endpoint correspondence and complementarity (read only)\n",
         ("> CENTER_ONLY_FIXED_ORIENTATION: transported states use trained orientations at every frame.\n"
          if jprov.get("transport_orientation", {}).get("mode") == "fixed" else
          "> TRACK_ROTATION_ORIENTATION: free-transport states turn each splat by its local track rotation; "
          "J1 states by the estimated part rotation (covariance-consistent).\n"),
         "## Report flags\n"]
    L += [f"* **{k}**: {v}" for k, v in S["flags"].items()]
    L.append("\n## Endpoint contrasts (% of pixels with a valid RECORDED depth, all cameras; sensor data, not GT)\n")
    keys = ["BOTH_VALID_SUPPORT", "F_SUPPORTED_B_UNEXPLAINED", "B_SUPPORTED_F_UNEXPLAINED",
            "NEITHER_VALID_SUPPORT", "BOTH_SAME_OBSERVED_SURFACE_CANDIDATE", "BOTH_COVER_DIFFERENT_DEPTH",
            "NEITHER_COVERS__on_gt_valid"]
    keys = [observed_names({k: 0}).popitem()[0] for k in keys]
    L.append("| contrast | " + " | ".join(keys) + " |")
    L.append("|---|" + "---|" * len(keys))
    for ct, v in S["contrasts_pct_of_observed_depth_valid"].items():
        L.append(f"| {ct} | " + " | ".join(str(v[k]) for k in keys) + " |")
    L.append("\nA = F0 vs B0 at frame 0; B = FT vs BT at frame T; C = F0 vs BT on pixels whose measured "
             "depth is unchanged between 0 and T; *_J1_<d> = the same with the transported source "
             "replaced by its joint-informed transport from estimate <d>.\n")
    L.append("## Estimated joints\n")
    for d, e in est.items():
        L.append(f"* {d}: **{e.status}**. " + "; ".join(e.notes))
        if getattr(e, "motion", None):
            L.append(f"  motion of the owned part: {e.motion['max_owned_displacement_m']:.4f} m, "
                     f"{e.motion['max_rotation_rad']:.4f} rad; residual evidence: "
                     f"{json.dumps(e.residual_preference)}")
    L.append(f"* J1: {jprov['j1_status']}\n")
    L.append("## Complementary patches (J0 vs J1)\n")
    L.append("| patch | type | rows | witness px / cams | J0 | J1 | occlusion | why |")
    L.append("|---|---|---|---|---|---|---|---|")
    for s in patch_out:
        for r in patch_out[s]:
            if "J0_label" not in r:
                continue
            j1 = ", ".join(f"{d}: {x['label']}" + (f" ({x['witness_explained_by_J1_transport_of_the_other_source']:.2f})"
                                                   if x.get('witness_explained_by_J1_transport_of_the_other_source') is not None else "")
                           for d, x in r["J1"].items()) or jprov["j1_status"]
            L.append(f"| {r['patch_id']} | {r['type']} | {r['n_rows']} | {r['witness_pixels']} / {r['witness_cameras']} "
                     f"| {r['J0_label']} | {j1} | {(r.get('occlusion_test') or {}).get('status', '')} | {r['why'][:160]} |")
    L.append("\n## Row accounting\n")
    for s, v in S["accounting"].items():
        L.append(f"* {s}: " + ", ".join(f"{k} {x}" for k, x in v.items()))
    L.append("\n## Limitations\n")
    L += ["* Fixed (trained) orientations for transported states; anisotropic footprints of moving parts are wrong away from their anchor.",
          "* Estimated joints, Phase-1 tracks and the endpoint models come from the same RGB-D: they are correlated, not independent evidence.",
          "* J1 ownership in the non-estimating source is inferred by proximity (low confidence).",
          "* Labels are declared-rule hypotheses (rules in endpoint_summary.json), not physical truth; a label is never a decision to add or remove geometry.",
          "* Contributor depth is the Gaussian centre's view depth, not a ray-surface intersection."]
    (out / "endpoint_summary.md").write_text("\n".join(L) + "\n")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scene", required=True)
    ap.add_argument("--results-root", default=os.environ.get("RESULTS_ROOT", DEFAULT_RESULTS))
    ap.add_argument("--dataset-root", default=os.environ.get("DATASET_ROOT", DEFAULT_DATASET))
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--frames", default="endpoints", choices=("endpoints",),
                    help="the endpoint frames 0 and T (the only supported mode)")
    ap.add_argument("--cameras", default="all")
    ap.add_argument("--estimated-joints", choices=("auto", "require", "off"), default="auto")
    mode = ap.add_mutually_exclusive_group(required=True)
    mode.add_argument("--preflight", action="store_true")
    mode.add_argument("--dry-run", action="store_true")
    mode.add_argument("--execute", action="store_true")
    ap.add_argument("--resume", action="store_true",
                    help="reuse a COMPLETE experiment with the same fingerprint; refuse a different one")
    ap.add_argument("--transport-orientation", choices=("track_kabsch", "fixed"), default="track_kabsch",
                    help="J0 free transport: turn each splat with the local rigid rotation of its "
                         "track (covariance-consistent, default) or keep the trained orientation")
    ap.add_argument("--gpu-memory-fraction", type=float, default=None,
                    help="cap this process's GPU memory (shared GPU)")
    args = ap.parse_args()
    from observation_diagnostics.gt_firewall import Firewall
    P = scene_paths(args)
    Firewall.install([str(Path(args.results_root)), str(Path(args.dataset_root))])
    out = Path(args.out_dir) if args.out_dir else P["res"] / OUT_SUBDIR / args.tag
    pf = preflight(args, P)
    cams = pf["dataset_cameras"] if args.cameras == "all" else [c.strip() for c in args.cameras.split(",")]
    pf["plan"] = plan(args, pf, cams)
    pf["output"] = str(out)
    if args.preflight or args.dry_run:
        print(json.dumps(jsonable(pf), indent=1))
        if args.dry_run:
            print(f"\nplan: {args.scene} endpoints 0 and {pf.get('endpoints', {}).get('T')}, cameras {cams}, "
                  f"{pf['plan']['base_renders']} endpoint renders + up to {pf['plan']['j1_renders_max']} J1 "
                  f"+ <= {pf['plan']['intermediate_renders_max']} intermediate + <= "
                  f"{pf['plan']['case_study_and_probe_renders_max']} probe renders -> {out}")
        else:
            out.mkdir(parents=True, exist_ok=True)
            (out / "preflight.json").write_text(json.dumps(jsonable(pf), indent=1))
        return 2 if pf["blockers"] else 0
    if pf["blockers"]:
        print("BLOCKED:", "; ".join(pf["blockers"]))
        return 2
    # fingerprint and resume
    fp_payload = {"scene": args.scene, "cameras": cams, "estimated_joints": args.estimated_joints,
                  "transport_orientation": args.transport_orientation,
                  "code": code_hash(), "rules": RULES,
                  "inputs": {k: [pf[f"source_{k}"]["ply"], pf[f"source_{k}"]["rows"]] for k in ("F", "B")},
                  "joint_files": {d: e.get("files") for d, e in pf["estimated_joints"].items()}}
    fp = hashlib.sha256(json.dumps(fp_payload, sort_keys=True, default=str).encode()).hexdigest()
    state = out / "run_state.json"
    if state.is_file():
        old = json.loads(state.read_text())
        if old.get("fingerprint") != fp:
            raise SystemExit(f"{out} holds experiment {old.get('fingerprint', '')[:16]}, this is {fp[:16]}: "
                             "use a new --tag")
        if old.get("complete") and args.resume:
            print(f"complete experiment {fp[:16]} reused: {out}")
            return 0
        if not args.resume:
            raise SystemExit(f"{out} already holds this experiment; pass --resume")
    out.mkdir(parents=True, exist_ok=True)
    (out / "preflight.json").write_text(json.dumps(jsonable(pf), indent=1))
    state.write_text(json.dumps({"fingerprint": fp, "payload": fp_payload, "complete": False}, default=str))
    log = Log(out)
    rc = execute(args, P, pf, out, log)
    state.write_text(json.dumps({"fingerprint": fp, "payload": fp_payload, "complete": rc == 0}, default=str))
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
