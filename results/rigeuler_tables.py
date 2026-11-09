"""RigEuler3DGS result tables, built read-only from the pipeline output folders.

  synthetic_results.csv          one row per object x run (v5 canonical, washer from the v5 gates, v4 bootstrap):
                                 GT type / axis / coordinate errors for BOTH hypotheses + the geometric type decision
  synthetic_trajectories.csv     per-frame Phase-3 q of the GT-type hypothesis next to GT q, primary runs only,
                                 in the evaluator's native alignment (q relative to the canonical frame, axis sign
                                 aligned) -- validated against the saved coord_mae / coord_max_abs
  real_results.csv               one row per real scene x arm: v6 (velocity decay + free-space refinement) next to
                                 v5, support refinement, Phase 2/3 motion, type decision, Phase-1 drift attribution
  columns.csv                    column dictionary

GT is read only here, after the runs; it never entered inference. Nothing is written into the result folders.
Called by build_inputs.py; paths are set with configure().
"""
import csv, glob, json
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation as Rot

R = Path("/media/samuelemara/Extreme SSD/Samuele/results_11_10_26")          # results root
REAL = Path("/media/samuelemara/Extreme SSD/Samuele/Backup_09_10_26/Sapien/output/real")
SYN = Path("/media/samuelemara/Extreme SSD/Samuele/Backup_09_10_26/Sapien/output/preparation_8cams/"
           "partnet_subset_fixed_render_linear")
SYNTHETIC_RUNS = [("v5_canonical", "ours_v5_canonical_synthetic", None),
                  ("v5_gates_canonical", "ours_v5_phase1_gates", "washer_103776_canonical"),
                  ("v4_bootstrap", "ours_v4_bootstrap_synthetic", None)]
REAL_RUNS = {"v6": "ours_v6_two_cams", "v5": "ours_v5_two_cams"}
CAMS = ["cam_001", "cam_002"]
RL_PRISMATIC = 10.0           # axis_radius_over_part_scale above this -> prismatic (revolute fit degenerate)
DEG = 57.29577951308232


def configure(results_root=None, real_root=None, synthetic_root=None):
    global R, REAL, SYN
    R = Path(results_root) if results_root else R
    REAL = Path(real_root) if real_root else REAL
    SYN = Path(synthetic_root) if synthetic_root else SYN


def cameras(src):
    d = json.load(open(src / "metadata" / "cameras.json"))["cameras"]
    return {c["name"]: c for c in d if c["name"] in CAMS}


def project(cam, X):
    M = np.array(cam["pose_matrix_4x4"])
    Xc = (X - M[:3, 3]) @ M[:3, :3]
    z = Xc[:, 2]
    return cam["fx"] * Xc[:, 0] / z + cam["cx"], cam["fy"] * Xc[:, 1] / z + cam["cy"], z


def classify(cam, D, X, tau):
    """Free-space code per point: 0 out/invalid, 1 consistent, 2 violated (in front of depth), 3 hidden."""
    u, v, z = project(cam, X)
    H, W = D.shape
    ui, vi = np.round(u).astype(int), np.round(v).astype(int)
    ok = (z > 0.05) & (ui >= 0) & (ui < W) & (vi >= 0) & (vi < H)
    out = np.zeros(len(X), np.int8)
    d = np.zeros(len(X))
    d[ok] = D[vi[ok], ui[ok]]
    ok &= d > 0
    r = z - d
    out[ok & (np.abs(r) <= tau)] = 1
    out[ok & (r < -tau)] = 2
    out[ok & (r > tau)] = 3
    return out


def ang(T):
    return float(np.degrees(np.arccos(np.clip((np.trace(T[:3, :3]) - 1) / 2, -1, 1))))


def rnd(x, k=4):
    if x is None:
        return ""
    if isinstance(x, (float, np.floating)):
        return "" if not np.isfinite(x) else round(float(x), k)
    return x


def joint_geometry(rigid):
    """Type-decision quantities from the rigid summary + both geometry files."""
    s = json.load(open(rigid / "summary.json"))
    rv, pr = s["revolute"], s["prismatic"]
    th = np.array(rv["refined"]["theta"])
    sv = np.array(json.load(open(rigid / "joint_reduction/prismatic/geometry.json"))["s"])
    P = np.load(rigid / "phase2_rigid/poses.npy")
    rl = rv.get("axis_radius_over_part_scale") or rv["radius_from_centroid"] / s["one_dof"]["part_scale_L"]
    eP, eR = pr["pose_residual"]["right_chart_energy"], rv["pose_residual"]["right_chart_energy"]
    return dict(
        support=s["support"]["num_members"], support_fraction=s["support"]["fraction"], support_method=s["support"]["method"],
        gaussians=s["gaussians"], frames=s["frames"], cameras=s["cameras"],
        p2_max_rotation_deg=max(ang(T) for T in P), p2_max_translation_m=float(max(np.linalg.norm(T[:3, 3]) for T in P)),
        p2_end_rotation_deg=ang(P[-1]), p2_tail15_rotation_deg=ang(np.linalg.inv(P[-16]) @ P[-1]),
        rev_axis="%.4f %.4f %.4f" % tuple(rv["refined"]["axis"]), rev_axis_point="%.4f %.4f %.4f" % tuple(rv["refined"]["axis_point"]),
        rev_sweep_deg=float(np.degrees(th.max() - th.min())), rev_axis_radius_over_part_scale=rl,
        part_scale_m=rv["radius_from_centroid"] / rl,
        pri_direction="%.4f %.4f %.4f" % tuple(pr["direction"]), pri_travel_m=float(sv.max() - sv.min()),
        pose_res_prismatic=eP, pose_res_revolute=eR, pose_res_ratio_P_over_R=eP / max(eR, 1e-300),
        type_lowest_residual="prismatic" if eP < eR else "revolute",
        type_decision="prismatic" if rl > RL_PRISMATIC else "revolute")


def p3(joint):
    out = {}
    for h in ("prismatic", "revolute"):
        tr = list(csv.DictReader(open(joint / h / "trajectory.csv")))
        q = np.array([float(r["q"]) for r in tr])
        u = 100.0 if h == "prismatic" else DEG
        out[h] = dict(end=u * q[-1], range=u * (q.max() - q.min()), tail15=u * abs(q[-1] - q[-16]))
    return out


# ------------------------------------------------------------------------------------------- synthetic
def synthetic():
    runs = []
    for ver, folder, only in SYNTHETIC_RUNS:
        runs += [(ver, R / folder / only)] if only else [(ver, d) for d in sorted((R / folder).iterdir()) if d.is_dir()]
    rows = []
    for ver, d in runs:
        B = glob.glob(f"{d}/**/bidirectional", recursive=True)
        if not B:
            continue
        B = Path(B[0]); rigid = sorted(B.glob("*_rigid"))[0]; joint = sorted(B.glob("*_joint"))[0]
        g = joint_geometry(rigid)
        e3 = json.load(open(joint / "evaluation_gt/phase3_gt_evaluation.json"))
        e2 = json.load(open(rigid / "joint_reduction/gt_evaluation.json"))
        gj = e3["gt_joint"]; gt = "revolute" if gj["type"] in ("revolute", "continuous") else gj["type"]
        traj = p3(joint)
        row = dict(object=d.name.replace("_canonical", ""), run=ver, result_dir=str(d), direction=joint.name.split("_")[0],
                   gt_type=gt, gt_axis="%.4f %.4f %.4f" % tuple(gj["axis"]), gt_axis_point="%.4f %.4f %.4f" % tuple(gj["point"]))
        row.update({k: g[k] for k in ("frames", "cameras", "gaussians", "support", "support_fraction")})
        row.update(type_decision=g["type_decision"], type_decision_correct=g["type_decision"] == gt,
                   rev_axis_radius_over_part_scale=g["rev_axis_radius_over_part_scale"], rev_sweep_deg=g["rev_sweep_deg"],
                   pri_travel_m=g["pri_travel_m"], part_scale_m=g["part_scale_m"], p2_max_rotation_deg=g["p2_max_rotation_deg"],
                   p2_max_translation_m=g["p2_max_translation_m"],
                   type_lowest_residual=g["type_lowest_residual"], type_lowest_residual_correct=g["type_lowest_residual"] == gt,
                   pose_res_ratio_P_over_R=g["pose_res_ratio_P_over_R"])
        for h in ("prismatic", "revolute"):
            f3, f2 = e3["fits"][h], e2[h]
            unit = "deg" if h == "revolute" else "cm"; sc = DEG if h == "revolute" else 100.0
            row[f"{h}_axis_err_deg_p3"] = f3["axis_angle_deg"]
            row[f"{h}_axis_err_deg_p2"] = f2["axis_angle_deg"]
            row[f"{h}_line_dist_cm"] = 100 * f3["line_distance_m"] if f3.get("line_distance_m") is not None else ""
            if f3["matches_gt_type"]:
                row[f"{h}_q_mae_{unit}_p3"] = sc * f3["coord_mae"]
                row[f"{h}_q_max_abs_{unit}_p3"] = sc * f3["coord_max_abs"]
                row[f"{h}_q_mae_{unit}_p2"] = sc * f3["coord_mae_phase2_init"]
                row[f"{h}_range_est_{unit}"] = sc * f3["coord_range_est"]
                row[f"{h}_range_gt_{unit}"] = sc * f3["coord_range_gt"]
                row[f"{h}_range_rel_err"] = f3["coord_range_est"] / f3["coord_range_gt"] - 1
            row[f"{h}_increment_scale_ratio_p2"] = f2["q_match"]["increment_scale_ratio"]
            row[f"{h}_p3_end_{unit}"] = traj[h]["end"]
        # convenience: errors of the GT-type hypothesis
        row["gt_hyp_axis_err_deg"] = e3["fits"][gt]["axis_angle_deg"]
        row["gt_hyp_line_dist_cm"] = row.get(f"{gt}_line_dist_cm", "")
        rows.append(row)
    return rows


# ------------------------------------------------------------------------------------------- real
SCENES = {  # what the images and depth-change maps show (visual, not GT)
    "black_box": ("lid opens slightly", "revolute"),
    "black_laptop": ("lid closes ~100 deg (glossy black, depth mostly invalid)", "revolute"),
    "light_blue_box": ("drawer closes ~21 cm; whole case pushed back a few cm", "prismatic"),
    "matchbox": ("matchbox drawer pushed in, small", "prismatic"),
    "orange_suitecase": ("telescopic handle pushed in ~22 cm", "prismatic"),
    "red_fridge": ("door closes ~40-44 deg (depth sweep); arm retracts after", "revolute"),
    "robot_dog": ("battery box on the back pushed in, small", "prismatic"),
    "wooden_box": ("lid opens", "revolute"),
}
TABLE = {  # direction and motion interval used by ~/run_v5_two_cams.sh and ~/run_v6_two_cams.sh
    f"final_{k}_episode_000000": dict(direction=d, motion_start=a, motion_end=b) for k, d, a, b in (
        ("black_box", "backward", 52, 228), ("black_laptop", "forward", 55, 228),
        ("light_blue_box", "forward", 35, 200), ("matchbox", "forward", 61, 207),
        ("orange_suitecase", "forward", 51, 162), ("red_fridge", "forward", 100, 288),
        ("robot_dog", "forward", 36, 186), ("wooden_box", "forward", 38, 127))}


def pipe(ver, scene, arm):
    B = Path(glob.glob(f"{R}/{REAL_RUNS[ver]}/{scene}/{arm}/pipeline/*/bidirectional")[0])
    return B, sorted(B.glob("*_rigid"))[0], sorted(B.glob("*_joint"))[0], sorted(B.glob("*/phase1_centers"))[0]


def depth_sweep(scene, X0, P, fr, cams, ks):
    """Scale the end rotation about its own screw axis; fraction of door Gaussians consistent with depth."""
    D = {(k, c): np.load(REAL / scene / "depth_npy" / c / f"frame_{fr[k]:06d}.npy") for k in ks for c in CAMS}
    res = {}
    for a in np.round(np.arange(0.70, 1.36, 0.05), 2):
        C = V = 0
        for k in ks:
            Rk, pk = P[k][:3, :3], P[k][:3, 3]; rv = Rot.from_matrix(Rk).as_rotvec()
            ctr = np.linalg.lstsq(np.eye(3) - Rk, pk, rcond=None)[0]
            Ra = Rot.from_rotvec(a * rv).as_matrix(); pa = (np.eye(3) - Ra) @ ctr + (pk - (np.eye(3) - Rk) @ ctr)
            X = X0 @ Ra.T + pa
            for c in CAMS:
                cl = classify(cams[c], D[(k, c)], X, 0.02); C += (cl == 1).sum(); V += (cl == 2).sum()
        res[a] = C / max(C + V, 1)
    th = np.degrees(np.linalg.norm(Rot.from_matrix(P[ks[-1]][:3, :3]).as_rotvec())); best = max(res, key=res.get)
    return res[1.0], best * th, res[best]


def real_row(args):
    """One real scene x arm. Uses the module paths, so call configure() before forking workers."""
    sc, arm = args
    scene = f"final_{sc}_episode_000000"; src = REAL / scene; cams = cameras(src)
    row = dict(scene=sc, arm=arm, motion_observed=SCENES[sc][0], expected_type=SCENES[sc][1], **TABLE[scene],
               cameras="cam_001,cam_002")
    B6, rg6, jt6, cd6 = pipe("v6", scene, arm); B5, rg5, jt5, cd5 = pipe("v5", scene, arm)
    fr = json.load(open(jt6 / "summary.json"))["original_frames_processing_order"]
    row.update(canonical_frame=fr[0], processing_frames=len(fr))
    # support + refinement (v6)
    rf = json.load(open(rg6 / "support_refine/round_0.json"))
    coh = int(np.load(rg6 / "support_refine/round_0.npz")["previous"].sum())   # refused refinements keep no before-file
    g6, g5 = joint_geometry(rg6), joint_geometry(rg5)
    row.update(gaussians=g6["gaussians"], v6_coherence_support=coh, v6_refine_accepted=rf["accepted"], v6_refine_added=rf["added"],
               v6_refine_removed=rf["removed"], v6_final_support=g6["support"], v6_final_support_fraction=g6["support_fraction"],
               v6_label_moving=rf["labels"]["moving"]["total"], v6_label_static=rf["labels"]["static"]["total"],
               v6_label_small_motion=rf["labels"]["small_motion"]["total"], v6_label_unobserved=rf["labels"]["unobserved"]["total"],
               v6_coherence_support_unobserved=rf["labels"]["unobserved"]["in_previous_support"],
               v5_support=g5["support"], v5_support_fraction=g5["support_fraction"])
    z = np.load(rg6 / "support_refine/round_0.npz"); c = z["counts"]; prev, mem = z["previous"], z["members"]
    add = mem & ~prev
    row["v6_added_mean_moving_consistent"] = float(c[add, 1, 0].mean()) if add.any() else ""
    row["v6_added_mean_static_violated"] = float(c[add, 0, 1].mean()) if add.any() else ""
    conf = (c[:, 1, 0] >= 6) & (c[:, 1, 0] >= c[:, 1, 1]) & (2 * (c[:, 0, 1] - c[:, 1, 1]) + (c[:, 1, 0] - c[:, 0, 0]) > 4) & (z["displacement_max"] >= 0.04)
    row["v6_moving_confirmed_rule"] = int(conf.sum())
    row["v6_added_unconfirmed"] = int((add & ~conf).sum())
    # Phase 2 / joint reduction / Phase 3, both versions
    for ver, g, jt in (("v6", g6, jt6), ("v5", g5, jt5)):
        for k in ("p2_max_rotation_deg", "p2_max_translation_m", "p2_end_rotation_deg", "p2_tail15_rotation_deg", "rev_sweep_deg",
                  "rev_axis_radius_over_part_scale", "part_scale_m", "pri_travel_m", "rev_axis", "rev_axis_point", "pri_direction",
                  "pose_res_ratio_P_over_R", "type_decision"):
            row[f"{ver}_{k}"] = g[k]
        row[f"{ver}_type_decision_matches_expected"] = g["type_decision"] == SCENES[sc][1]
        t = p3(jt)
        for h, u in (("revolute", "deg"), ("prismatic", "cm")):
            for k in ("end", "range", "tail15"):
                row[f"{ver}_p3_{h}_{k}_{u}"] = t[h][k]
        row[f"{ver}_p3_selected_{'deg_or_cm'}"] = t[g["type_decision"]]["end"]
    # fridge only: depth-implied end angle
    if sc == "red_fridge":
        ref = Path(glob.glob(f"{R}/{REAL_RUNS['v6']}/{scene}/unanimous/pipeline/*/bidirectional/*_rigid")[0])
        zr = np.load(ref / "support_refine/round_0.npz"); cr = zr["counts"]
        door = (cr[:, 1, 0] >= 20) & (cr[:, 1, 0] >= 4 * cr[:, 1, 1]) & (cr[:, 0, 1] >= 10)
        Xd = np.load(sorted(ref.parent.glob("*/phase1_centers"))[0] / "canonical_xyz.npy")[door]; ks = list(range(len(fr) - 30, len(fr), 3))   # world-frame door points of the reference arm
        for ver, rg in (("v6", rg6), ("v5", rg5)):
            c0, best, cb = depth_sweep(scene, Xd, np.load(rg / "phase2_rigid/poses.npy"), fr, cams, ks)
            row[f"{ver}_depth_consistency_at_estimate"] = c0; row[f"{ver}_depth_best_end_angle_deg"] = best
            row[f"{ver}_depth_consistency_at_best"] = cb
    # Phase-1 drift on the SAME Gaussians (shared endpoint)
    X0 = np.load(cd6 / "canonical_xyz.npy"); n = len(X0)
    st_conf = (c[:, 0, 0] >= 20) & (c[:, 0, 1] <= 1)
    for ver, cd in (("v6", cd6), ("v5", cd5)):
        Ce = np.load(cd / f"centers_frame_{fr[-1]:04d}.npy"); Ct = np.load(cd / f"centers_frame_{fr[-16]:04d}.npy")
        Cm = np.load(cd / f"centers_frame_{fr[len(fr) // 2]:04d}.npy")
        dmed = lambda A, Bb, m: 100 * float(np.median(np.linalg.norm(A[m] - Bb[m], axis=1))) if m.sum() > 20 else float("nan")
        row[f"{ver}_p1_confirmed_static_drift_end_cm"] = dmed(Ce, X0, st_conf)
        row[f"{ver}_p1_confirmed_static_drift_mid_cm"] = dmed(Cm, X0, st_conf)
        row[f"{ver}_p1_confirmed_static_tail15_cm"] = dmed(Ce, Ct, st_conf)
        row[f"{ver}_p1_all_tail15_cm"] = dmed(Ce, Ct, np.ones(n, bool))
    row["p1_confirmed_static_n"] = int(st_conf.sum())
    # v6 drift attribution: static set = outside both supports, never refuted at its canonical place
    mem6 = np.zeros(n, bool); mem6[json.load(open(rg6 / "support_coherence/support.json"))["members"]] = True
    S = (~mem6) & (~prev) & (c[:, 0, 1] <= 1) & (z["label"] != 0)
    row["drift_static_set_n"] = int(S.sum())
    if S.sum() >= 100 and mem6.any():
        Ce = np.load(cd6 / f"centers_frame_{fr[-1]:04d}.npy"); d = Ce - X0; dm = 100 * np.linalg.norm(d, axis=1)
        near = cKDTree(X0[mem6]).query(X0)[0] < 0.02
        inm = np.zeros(n); ks = np.linspace(0, len(fr) - 1, 12).astype(int)
        for k in ks:
            a = np.zeros(n, bool)
            for cm in CAMS:
                D = np.load(src / "depth_npy" / cm / f"frame_{fr[k]:06d}.npy")
                M = np.asarray(Image.open(src / "mask" / cm / f"frame_{fr[k]:06d}.png").convert("L")) > 127
                cl = classify(cams[cm], D, X0, 0.02); u, v, _ = project(cams[cm], X0); ui, vi = np.round(u).astype(int), np.round(v).astype(int)
                ok = (ui >= 0) & (ui < M.shape[1]) & (vi >= 0) & (vi < M.shape[0]); m = np.zeros(n, bool); m[ok] = M[vi[ok], ui[ok]]
                a |= m & (cl == 1)
            inm += a
        inm /= len(ks)
        _, nb = cKDTree(X0).query(X0, 16); Q = X0[nb] - X0[nb].mean(1, keepdims=True)
        nrm = np.linalg.eigh(np.einsum("nki,nkj->nij", Q, Q))[1][:, :, 0]
        dn = 100 * np.abs((d * nrm).sum(1)); dt = 100 * np.linalg.norm(d - (d * nrm).sum(1, keepdims=True) * nrm, axis=1)
        tot = dm[S].sum()
        row.update(drift_all_median_cm=float(np.median(dm[S])), drift_all_q90_cm=float(np.quantile(dm[S], .9)),
                   drift_normal_median_cm=float(np.median(dn[S])), drift_tangent_median_cm=float(np.median(dt[S])))
        for name, m in (("near_part", S & near), ("unseen", S & ~near & (inm < 0.5)), ("seen_away", S & ~near & (inm >= 0.5))):
            row[f"drift_{name}_n"] = int(m.sum()); row[f"drift_{name}_pct_of_set"] = 100 * m.sum() / S.sum()
            row[f"drift_{name}_median_cm"] = float(np.median(dm[m])) if m.sum() else ""
            row[f"drift_{name}_share_pct"] = 100 * dm[m].sum() / tot
        prof = [100 * float(np.median(np.linalg.norm(np.load(cd6 / f"centers_frame_{fr[k]:04d}.npy")[S] - X0[S], axis=1)))
                for k in np.linspace(0, len(fr) - 1, 6).astype(int)]
        row["drift_profile_0_20_40_60_80_100pct_cm"] = " ".join(f"{p:.2f}" for p in prof)
    print(sc, arm, "done", flush=True)
    return row


DOC = [
    ("synthetic: object/run", "object id; run = v5_canonical (ours_v5_canonical_synthetic), v5_gates_canonical (washer, ours_v5_phase1_gates), v4_bootstrap"),
    ("gt_type, gt_axis, gt_axis_point", "ground truth from the dataset (evaluation only)"),
    ("type_decision", f"prismatic if rev_axis_radius_over_part_scale > {RL_PRISMATIC} (revolute fit degenerate: far axis), else revolute"),
    ("type_lowest_residual", "argmin of joint-reduction right-chart pose residual (shown for comparison; not a good criterion)"),
    ("rev_axis_radius_over_part_scale", "revolute axis distance from the support centroid / part scale L"),
    ("rev_sweep_deg / pri_travel_m", "range of the revolute angle / prismatic coordinate in the joint-reduction fits"),
    ("p2_max_rotation_deg / p2_max_translation_m", "largest rotation / translation of the Phase-2 rigid poses relative to the canonical frame"),
    ("<hyp>_axis_err_deg_p3 / _p2", "angle between estimated and GT axis for each hypothesis, Phase 3 / joint reduction"),
    ("<hyp>_line_dist_cm", "distance between estimated and GT revolute axis lines"),
    ("<hyp>_q_mae_*", "joint coordinate MAE vs GT (only for the GT-type hypothesis); _p2 = Phase-2 reduction initialisation"),
    ("<hyp>_range_est/gt, range_rel_err", "coordinate range estimated vs GT"),
    ("real: motion_observed, expected_type", "visual reading of the RGB and depth-change maps (real scenes have no GT)"),
    ("v6_* / v5_*", "v6 = velocity decay (tau 0.5 s) + free-space support refinement, v5 = same endpoints without them"),
    ("v6_coherence_support / v6_final_support", "support from Phase-1 coherence / after the free-space refinement"),
    ("v6_label_*", "free-space labels of all Gaussians under the first rigid motion"),
    ("v6_added_mean_moving_consistent / _static_violated", "mean counts for Gaussians the refinement added; ~0 consistent = never confirmed by the motion"),
    ("v6_moving_confirmed_rule / v6_added_unconfirmed", "Gaussians passing the proposed confirmed rule (Cm>=6, Cm>=Vm, L>4) / added Gaussians that fail it"),
    ("vX_p3_<hyp>_end/range/tail15", "Phase-3 coordinate at the last frame / range / change over the last 15 frames"),
    ("vX_p3_selected", "Phase-3 end value of the hypothesis chosen by type_decision (deg for revolute, cm for prismatic)"),
    ("vX_depth_*", "fridge only: door Gaussians moved by the end pose scaled about its own axis; consistency with depth at the estimate and the best angle"),
    ("vX_p1_confirmed_static_*", "Phase-1 drift of Gaussians seen >=20 times at their canonical place and never refuted (same set in v5 and v6)"),
    ("drift_*", "v6 Phase-1 drift attribution on the static set (outside both supports, never refuted): near_part = within 2 cm of the support; unseen = in mask and depth-consistent in <50% of 12 sampled frames; seen_away = rest. share = fraction of the summed drift"),
    ("drift_normal / tangent", "drift component along / across the local surface normal (kNN PCA)"),
]


def write_csv(path, rows):
    keys = []
    for r in rows:
        keys += [k for k in r if k not in keys]
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, keys); w.writeheader()
        for r in rows:
            w.writerow({k: rnd(r.get(k, "")) for k in keys})


def primary_synthetic_dirs():
    """{object: (run, result_dir)} -- v5_canonical, or v5_gates_canonical where no v5_canonical run exists."""
    out = {}
    for ver, folder, only in SYNTHETIC_RUNS[:2]:
        dirs = [R / folder / only] if only else [d for d in sorted((R / folder).iterdir()) if d.is_dir()]
        for d in dirs:
            obj = d.name.replace("_canonical", "")
            out.setdefault(obj, (ver, d))
    return out


def synthetic_trajectories(tol=1e-9):
    """Per-frame Phase-3 q (GT-type hypothesis) and GT q, exactly as tools/run_joint_phase3.evaluate_gt aligns them:
    q_gt = q(frame) - q(canonical frame), estimate times sign(axis . gt_axis). Raises if the recomputed MAE / max
    differ from the saved evaluation."""
    rows = []
    for obj, (ver, d) in sorted(primary_synthetic_dirs().items()):
        B = Path(glob.glob(f"{d}/**/bidirectional", recursive=True)[0]); joint = sorted(B.glob("*_joint"))[0]
        e3 = json.load(open(joint / "evaluation_gt/phase3_gt_evaluation.json")); gj = e3["gt_joint"]
        gt = "revolute" if gj["type"] in ("revolute", "continuous") else gj["type"]
        geo = json.load(open(joint / gt / "geometry.json"))
        e = np.asarray(geo.get("direction") or geo.get("axis"), float)
        sgn = float(np.sign(e @ np.asarray(gj["axis"]))) or 1.0
        tr = list(csv.DictReader(open(joint / gt / "trajectory.csv")))
        frames = np.array([int(r["original_frame"]) for r in tr]); q = sgn * np.array([float(r["q"]) for r in tr])
        fv = list(csv.reader(open(SYN / obj / "metadata" / "frame_values.csv")))
        qall = np.array([float(r[1]) for r in fv[1:]])
        assert [int(r[0]) for r in fv[1:]] == list(range(len(qall))), f"{obj}: frame_values is not 0..T-1"
        q_gt = qall[frames] - qall[frames[0]]
        mae, mx = float(np.abs(q - q_gt).mean()), float(np.abs(q - q_gt).max())
        ref = e3["fits"][gt]
        if abs(mae - ref["coord_mae"]) > tol * max(1.0, ref["coord_mae"]) or abs(mx - ref["coord_max_abs"]) > tol * max(1.0, ref["coord_max_abs"]):
            raise ValueError(f"{obj}: recomputed MAE/max {mae}/{mx} != saved {ref['coord_mae']}/{ref['coord_max_abs']}")
        u, unit = (DEG, "deg") if gt == "revolute" else (1000.0, "mm")
        for k, (f, a, b) in enumerate(zip(frames, q, q_gt)):
            rows.append(dict(scene=obj, run=ver, joint_type=gt, unit=unit, processing_index=k, source_frame_index=int(f),
                             canonical_frame=int(frames[0]), q_est=u * a, q_gt=u * b, abs_error=u * abs(a - b)))
    return rows
