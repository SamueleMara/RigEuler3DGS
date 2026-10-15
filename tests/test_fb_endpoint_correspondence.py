#!/usr/bin/env python
"""Tests of the endpoint correspondence diagnosis
(CODEX_DIAGNOSE_ENDPOINT_COMPLEMENTARITY_WITH_IEKF_AND_ESTIMATED_JOINTS.md section 6).

Synthetic scenes go through the REAL CUDA renderer, the diagnostic contributor
kernel and the SAME module functions the CLI uses (classify, accumulate,
view_pairs, group_rows, visibility_by_depth, j1_state, static_candidates,
j0_label, j1_label). The real part runs the CLI on washer_103776 (two
cameras) under a GPU-memory cap.

  [2]  renderer equivalence            [3]  frozen linearization (adjoint, FD)
  [4]  two-layer overlap               [5]  static shared despite a bad track
  [6]  missing surface                 [7]  pure transport failure + J1
  [8]  initially occluded moving       [9]  joint type/axis conventions
  [10] untrusted joint                 [11] GT firewall
  [1, 12, 13, 14, 15] real run: identities, immutability, repeatability/resume,
       row accounting, washer smoke

    python tests/test_fb_endpoint_correspondence.py [--no-real]
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

CHECKS, FAILURES = 0, []


def check(label, cond, detail=""):
    global CHECKS
    CHECKS += 1
    print(f"  {'ok  ' if cond else 'FAIL'}  {label}  {detail}")
    if not cond:
        FAILURES.append(label)


from tests.test_fb_pixel_overlap_core import cams_pair, gt_of, plate  # noqa: E402


def make(*parts):
    from tests.test_fb_pixel_overlap_core import make_state
    return make_state(*parts)


class FakeSource:
    """What j1_state needs from an EndpointSource: per-frame centres + base."""

    def __init__(self, states):
        self.mu = np.stack([s.mu.cpu().numpy() for s in states])
        self.base = states[0]
        self.N = self.mu.shape[1]
        self.frames = list(range(len(states)))

    def state(self, frame, mu=None, R=None):
        import dataclasses
        mu_t = torch.as_tensor(self.mu[frame] if mu is None else mu, dtype=torch.float32, device="cuda")
        R_t = self.base.R if R is None else torch.as_tensor(R, dtype=torch.float32, device="cuda")
        return dataclasses.replace(self.base, mu=mu_t.contiguous(), R=R_t.contiguous())


def b_only_at_T_metrics(F_states, B_states, true_states, cams, patch_rows=None, FT_J=None):
    """The CLI's B_ONLY_AT_T analysis on a synthetic sequence (frames 0..T)."""
    from observation_diagnostics import endpoint_compare as ec
    from observation_diagnostics import gaussian_evidence as ge
    from observation_diagnostics import pair_evidence as pv
    from observation_diagnostics import pixel_evidence as pe
    from tools.analyze_fb_endpoint_correspondence import RULES, patch_mass, to_np_contrib
    th = pe.Thresholds()
    tol = th.contrib_depth_tol_m
    T = len(true_states) - 1
    NF, NB = F_states[0].mu.shape[0], B_states[0].mu.shape[0]
    gts = {t: [gt_of(true_states[t], c) for c in cams] for t in (0, T)}
    acc = {k: ge.new_frame_accumulator(NF if k[1] == "F" else NB, "cuda") for k in ("AF", "AB", "BF", "BB", "CF", "CB")}
    pviews = {"A": [], "B": [], "C": []}
    keep = {}
    for ci, cam in enumerate(cams):
        V = {"F0": ec.render("F0", F_states[0], cam, 0, ci), "B0": ec.render("B0", B_states[0], cam, 0, ci),
             "FT": ec.render("FT", F_states[T], cam, T, ci), "BT": ec.render("BT", B_states[T], cam, T, ci)}
        g0, gT = gts[0][ci], gts[T][ci]
        gS = ec.static_observation(g0, gT, th.depth_tol_m)
        for ct, kf, kb, g in (("A", "F0", "B0", g0), ("B", "FT", "BT", gT), ("C", "F0", "BT", gS)):
            f = pe.classify(V[kf].d, V[kb].d, g, th)
            ge.accumulate(acc[ct + "F"], V[kf].contrib, V[kf].lin.cache.depth, f, g["depth"], "F", ci, tol)
            ge.accumulate(acc[ct + "B"], V[kb].contrib, V[kb].lin.cache.depth, f, g["depth"], "B", ci, tol)
            pviews[ct].append((ci, pv.view_pairs(V[kf].contrib, V[kb].contrib, V[kf].lin.cache.depth,
                                                 V[kb].lin.cache.depth, V[kf].state.colors, V[kb].state.colors,
                                                 V[kf].state.mu, V[kb].state.mu, g["depth"], g["valid"],
                                                 NF, NB, th, prune_tau=0.0)))
            if ct == "B":
                keep[ci] = {"W": f["W_Bonly"].cpu().numpy(), "BT": to_np_contrib(V["BT"].contrib),
                            "zBT": V["BT"].lin.cache.depth.cpu().numpy(),
                            "gT": {k: v.cpu().numpy() for k, v in gT.items()},
                            "g0": {k: v.cpu().numpy() for k, v in g0.items()}}
        if FT_J is not None:
            vJ = ec.render("FT_J", FT_J, cam, T, ci)
            keep[ci]["SJ"] = pe.classify(vJ.d, V["BT"].d, gT, th)["SdF"].cpu().numpy()
    A_ = {k: ge.to_numpy(v) for k, v in acc.items()}
    tabs = {ct: pv.frame_table(v) for ct, v in pviews.items()}
    U = A_["BB"]["witness"].astype(np.float64)
    rows = np.flatnonzero(U >= RULES["U_MIN"])
    BT_xyz = B_states[T].mu.cpu().numpy()
    scl = B_states[T].scales.max(1).values.cpu().numpy()
    groups, _ = ec.group_rows(BT_xyz, rows, scl)
    if patch_rows is not None:
        groups.sort(key=lambda g: -np.isin(g, patch_rows).sum())
    else:
        groups.sort(key=lambda g: -U[g].sum())
    if not groups:
        return {"groups": 0}
    g = groups[0]
    mask = np.zeros(NB, bool)
    mask[g] = True
    n_pix = n_cam = 0
    num = den = 0.0
    for ci, k in keep.items():
        H, W = k["W"].shape
        mm = patch_mass(k["BT"], mask, k["zBT"], k["gT"]["depth"], k["gT"]["valid"], tol, k["W"], H * W)
        if (mm > 0).any():
            n_cam += 1
            n_pix += int((mm > 0).sum())
        if "SJ" in k:
            num += float((mm * k["SJ"].reshape(-1)).sum())
            den += float(mm.sum())

    def side(ct):
        t = tabs[ct]
        own = t["keys"] % NB
        m = mask[own]
        return {c: t[c][m] for c in pv.SUM_COLS}
    s_t, s_o, s_c = side("B"), side("A"), side("C")
    mk = float(A_["BB"]["mass_kappa"][g].sum())
    mk_o = float(A_["AB"]["mass_kappa"][g].sum())
    mk_c = float(A_["CB"]["mass_kappa"][g].sum())
    frac_same = float(s_t["O_surf_1"].sum()) / mk if mk > 0 else 0.0
    vis = {}
    B0_xyz = B_states[0].mu.cpu().numpy()
    for ci, cam in enumerate(cams):
        lab = ec.visibility_by_depth(cam, B0_xyz[g], keep[ci]["g0"]["depth"], keep[ci]["g0"]["valid"], tol)
        for x in set(lab):
            vis[x] = vis.get(x, 0) + int((lab == x).sum())
    nv = sum(vis.values())
    unseen = (vis.get("OCCLUDED", 0) + vis.get("OUT_OF_VIEW", 0) + vis.get("GT_INVALID", 0)) / max(nv, 1)
    Bmu = np.stack([s.mu.cpu().numpy() for s in B_states])
    is_static = bool(np.median(np.linalg.norm(Bmu[-1][g] - Bmu[0][g], axis=1)) <= RULES["STATIC_NET_M"])
    from scipy.spatial import cKDTree
    F0_xyz = F_states[0].mu.cpu().numpy()
    r_match = max(2.0 * float(np.median(np.r_[scl, F_states[0].scales.max(1).values.cpu().numpy()])), 0.02)
    dd, _ = cKDTree(F0_xyz).query(BT_xyz[g], k=1)
    m = dict(n_pix=n_pix, n_cam=n_cam, frac_same=frac_same,
             rgb_ratio=float(s_t["O_surf_rgb"].sum()) / max(float(s_t["O_surf_1"].sum()), 1e-30),
             frac_other=float(s_o["O_surf_1"].sum()) / mk_o if mk_o > 0 else 0.0,
             is_static=is_static, frac_static=float(s_c["O_surf_1"].sum()) / mk_c if mk_c > 0 else 0.0,
             excl=float(U[g].sum()) / max(float(A_["BB"]["mass_supported"][g].sum()), 1e-30),
             unseen=unseen, sup_o=float(A_["AB"]["mass_supported"][g].sum()),
             world_near=float(np.mean(dd <= r_match)))
    lab, why = ec.j0_label(RULES, s="B", o="F", fail="FT", fail_transported=True, t_end=T, t_oth=0, **m)
    return {"groups": len(groups), "patch": g, "metrics": m, "J0": lab, "why": why,
            "J1_explained": (num / den if den > 0 else None),
            "J1": ec.j1_label(RULES, num / den if den > 0 else None) if FT_J is not None else None,
            "acc": A_, "tabs": tabs, "incompat": float(s_t["O_incompat"].sum()) / max(float(s_t["O"].sum()), 1e-30)}


# ---------------------------------------------------------------------------

def t2_t3():
    from observation_diagnostics import endpoint_compare as ec
    print("\n[2] renderer equivalence  [3] frozen linearization")
    cams = cams_pair()
    st = make(plate(0.0, 0.0, 0.0, 0.6, 0.45, 9, (0.8, 0.5, 0.2)),
              plate(0.0, 0.0, -0.2, 0.9, 0.7, 11, (0.2, 0.5, 0.8)))
    v = ec.render("X", st, cams[0], 0, "c0")
    c = ec.conservation(v)
    check("[2] sum q = A, depth rebuilt, final T bitwise", c["max_abs_sum_q_minus_A"] < 2e-6
          and c["max_abs_depth_recon_m"] < 1e-5 and c["final_T_bitwise"], json.dumps(c))
    a = ec.adjoint_check(v, n_rows=20)
    check("[3] adjoint identity <Hv,z> = <v,H^T z> on the frozen sort", a["adjoint_relative_gap"] < 1e-4,
          f"{a['adjoint_relative_gap']:.1e}")
    check("[3] JVP agrees with a re-render finite difference", a["jvp_vs_finite_difference_relative"] < 0.1,
          f"{a['jvp_vs_finite_difference_relative']:.3f}")


def t4_layers():
    print("\n[4] two depth-separated surfaces sharing pixels")
    cams = cams_pair()
    front = plate(0.0, 0.0, 0.0, 0.6, 0.45, 9, (0.8, 0.5, 0.2))
    back = plate(0.0, 0.0, -0.3, 0.6, 0.45, 9, (0.8, 0.5, 0.2))
    wall = plate(0.0, 0.0, -0.6, 1.6, 1.2, 11, (0.4, 0.4, 0.4))
    sF, sB, true = make(front, wall), make(back, wall), make(back, wall)
    r = b_only_at_T_metrics([sF, sF], [sB, sB], [true, true], cams)
    check("[4] the B patch shares pixels with F but at another depth",
          r["incompat"] > 0.9 and r["metrics"]["frac_same"] == 0.0,
          f"incompatible {r['incompat']:.2f}, same-surface {r['metrics']['frac_same']:.2f}")
    check("[4] never called a common surface", r["J0"] != "COMMON_SURFACE", r["J0"])


def t5_static():
    from observation_diagnostics import endpoint_compare as ec
    from observation_diagnostics import gaussian_evidence as ge  # noqa: F401
    from tools.analyze_fb_endpoint_correspondence import RULES
    print("\n[5] a static surface matched across anchors despite an erroneous free track")
    cams = cams_pair()
    wF = plate(0.0, 0.0, -0.2, 1.4, 1.0, 13, (0.6, 0.6, 0.6))
    wB = plate(0.0, 0.0, -0.2, 1.4, 1.0, 10, (0.6, 0.6, 0.6))          # other density, other rows
    F0 = make(wF)
    wF_bad = (wF[0] + torch.tensor([0.12, 0.0, 0.0]), wF[1], wF[2], wF[3])
    FT = make(wF_bad)                                                  # transport error
    BT = make(wB)
    r = b_only_at_T_metrics([F0, FT], [BT, BT], [BT, BT], cams)
    tC, A_ = r["tabs"]["C"], r["acc"]
    NF, NB = F0.mu.shape[0], BT.mu.shape[0]
    netF = np.full(NF, 0.12)
    sc = ec.static_candidates(tC["keys"], tC["O_surf_1"], tC["dist3d_surf"],
                              A_["CF"]["q2_kappa_1"].astype(np.float64), A_["CB"]["q2_kappa_1"].astype(np.float64),
                              NB, NF, A_["CF"]["mass_supported"] >= 1.0, netF <= RULES["STATIC_NET_M"],
                              np.ones(NB, bool), RULES, 0.1)
    sup = A_["CF"]["mass_supported"] >= 1.0
    frac = float(sc["world"][sup].mean())
    check("[5] supported F0 rows matched to BT in world coordinates", frac > 0.8, f"{frac:.2f} of {int(sup.sum())}")
    check("[5] and flagged: the free track disagrees (transport error, not motion)",
          set(sc["tier"][sc["world"]]) == {"STATIC_WORLD_MATCH_TRACK_DISAGREES"})
    check("[5] matched world distance small", float(np.median(sc["dist"][sc["world"]])) < 0.06,
          f"median {np.median(sc['dist'][sc['world']]):.3f} m")
    check("[5] contrast B sees the erroneous transport as unexplained by FT",
          r["groups"] > 0 and r["J0"] == "TRANSPORT_MISMATCH", f"{r.get('J0')}: {r.get('why', '')[:120]}")


def scene_transport(shift_ok=True):
    """H moves from x=-0.3 (frame 0) to x=+0.3 (frame T); F's track leaves it behind."""
    wall_F = plate(0.0, 0.0, -0.3, 1.6, 1.2, 11, (0.4, 0.4, 0.4))
    wall_B = plate(0.0, 0.0, -0.3, 1.6, 1.2, 9, (0.4, 0.4, 0.4))
    H0F = plate(-0.3, 0.0, 0.0, 0.3, 0.3, 7, (0.9, 0.3, 0.1))
    H0B = plate(-0.3, 0.0, 0.0, 0.3, 0.3, 6, (0.9, 0.3, 0.1))
    mv = lambda P, dx: (P[0] + torch.tensor([dx, 0.0, 0.0]), P[1], P[2], P[3])  # noqa: E731
    true0, trueT = make(wall_B, H0B), make(wall_B, mv(H0B, 0.6))
    F0 = make(wall_F, H0F)
    FT = make(wall_F, H0F)                                   # failed transport
    BT = make(wall_B, mv(H0B, 0.6))
    B0 = make(wall_B, H0B)                                   # correct transport
    nwB, nwF = wall_B[0].shape[0], wall_F[0].shape[0]
    return F0, FT, B0, BT, true0, trueT, nwF, nwB


def t7_transport(wrong_joint=False):
    from observation_diagnostics import endpoint_compare as ec
    from observation_diagnostics.joint_evidence import apply
    print("\n[7] pure transport failure" + (" with a WRONG estimated joint [10]" if wrong_joint else ""))
    cams = cams_pair()
    F0, FT, B0, BT, true0, trueT, nwF, nwB = scene_transport()
    NB = BT.mu.shape[0]
    Hrows_B = np.arange(nwB, NB)
    # estimated part motion: translation +0.6 in x from frame 0 to T (anchor T = I)
    dx = -0.6 if wrong_joint else 0.6
    poses = np.stack([np.eye(4), np.eye(4)])
    poses[0][:3, 3] = [-dx, 0, 0]
    Tm = poses[1] @ np.linalg.inv(poses[0])                  # 0 -> T
    part0 = apply(np.linalg.inv(Tm), BT.mu.cpu().numpy()[Hrows_B])
    owned_F = ec.infer_ownership(F0.mu.cpu().numpy(), part0, 0.05)
    FT_J = ec.j1_state(FakeSource([F0, FT]), 0, owned_F, Tm)
    r = b_only_at_T_metrics([F0, FT], [B0, BT], [true0, trueT], cams, patch_rows=Hrows_B, FT_J=FT_J)
    hp = np.isin(r["patch"], Hrows_B).mean()
    if not wrong_joint:
        check("[7] the witness patch is the moved surface H (B rows)", hp > 0.8, f"{hp:.2f}")
        check("[7] inferred F ownership finds F's copy of H at frame 0",
              owned_F.size > 0 and bool((owned_F >= nwF).all()), f"{owned_F.size} rows")
        check("[7] J0 says TRANSPORT_MISMATCH, not missing geometry", r["J0"] == "TRANSPORT_MISMATCH",
              f"{r['J0']}: {r['why'][:120]}")
        check("[7] J1 (the estimated motion applied to F) explains the witness",
              r["J1"] == "J1_TRANSPORT_EXPLAINS", f"{r['J1']} ({r['J1_explained']})")
    else:
        check("[10] a wrong joint does not explain the witness", r["J1"] in ("J1_DOES_NOT_RECOVER", "J1_PARTIAL"),
              f"{r['J1']} ({r['J1_explained']})")
        check("[10] and does not change, delete or force the J0 reading", r["J0"] == "TRANSPORT_MISMATCH", r["J0"])


def scene_missing(T=5):
    """Q exists only in B; an occluder hides it until frame 3."""
    wall_F = plate(0.0, 0.0, -0.3, 1.6, 1.2, 11, (0.4, 0.4, 0.4))
    wall_B = plate(0.0, 0.0, -0.3, 1.6, 1.2, 9, (0.4, 0.4, 0.4))
    Q = plate(0.3, 0.0, 0.0, 0.3, 0.3, 6, (0.1, 0.8, 0.2))

    def occ(t):
        return plate(0.3 if t < 3 else 1.9, 0.0, 0.3, 0.45, 0.45, 8, (0.8, 0.1, 0.1), opacity=0.99)
    F = [make(wall_F, occ(t)) for t in range(T + 1)]
    B = [make(wall_B, occ(t), Q) for t in range(T + 1)]
    true = [make(wall_B, occ(t), Q) for t in range(T + 1)]
    nB = wall_B[0].shape[0] + occ(0)[0].shape[0]
    return F, B, true, np.arange(nB, B[0].mu.shape[0])


def t6_t8_missing(wrong_joint=False):
    from observation_diagnostics import endpoint_compare as ec
    print("\n[6] missing surface  [8] initially occluded, measurable later")
    cams = cams_pair()
    F, B, true, Qrows = scene_missing()
    T = len(true) - 1
    r = b_only_at_T_metrics([F[0], F[T]], [B[0], B[T]], [true[0], true[T]], cams, patch_rows=Qrows)
    qf = np.isin(r["patch"], Qrows).mean()
    check("[6] the B-only patch is Q, found although F has other row counts", qf > 0.8, f"{qf:.2f}")
    check("[6] labelled a B-unique OBSERVED proposal", r["J0"] == "B_UNIQUE_OBSERVED_PROPOSAL",
          f"{r['J0']}: {r['why'][:140]}")
    check("[8] at frame 0 its projections sit behind the measured foreground",
          r["metrics"]["unseen"] >= 0.6, f"unseen {r['metrics']['unseen']:.2f}")
    # intermediate frames from measured depth only
    tol = 0.04
    fr = []
    for t in range(T + 1):
        at = tot = 0
        for cam in cams:
            g = gt_of(true[t], cam)
            lab = ec.visibility_by_depth(cam, B[t].mu.cpu().numpy()[Qrows], g["depth"].cpu().numpy(),
                                         g["valid"].cpu().numpy(), tol)
            at += int((lab == "AT_OBSERVED_DEPTH").sum())
            tot += int((lab != "OUT_OF_VIEW").sum())
        fr.append(at / max(tot, 1))
    reveal = next((t for t in range(T + 1) if fr[t] >= 0.5), None)
    check("[8] the surface becomes measurable at the reveal frame (3), not before",
          reveal == 3 and max(fr[:3]) < 0.1, f"fractions {np.round(fr, 2).tolist()}")


def t9_conventions(tmp: Path):
    from observation_diagnostics.joint_evidence import (DirectionEstimate, direction_compare, line_compare,
                                                        prismatic_transform, revolute_transform)
    print("\n[9] joint type / axis conventions")
    a, p = np.array([0.2, 0.9, 0.1]), np.array([0.1, -0.2, 0.3])
    check("[9] revolute (a, theta) = (-a, -theta)", np.allclose(revolute_transform(a, p, 0.7),
                                                               revolute_transform(-a, p, -0.7)))
    check("[9] revolute point slides along the axis without changing the motion",
          np.allclose(revolute_transform(a, p, 0.7), revolute_transform(a, p + 0.5 * a, 0.7)))
    check("[9] prismatic (d, s) = (-d, -s)", np.allclose(prismatic_transform([1, 2, 0], 0.3),
                                                         prismatic_transform([-1, -2, 0], -0.3)))
    lc = line_compare(a, p, -a, p + 0.3 * a)
    check("[9] axes compared as lines, up to sign", lc["axis_angle_deg_up_to_sign"] < 1e-6
          and lc["point2_to_line1_distance_m"] < 1e-9, json.dumps(lc))
    check("[9] prismatic directions up to sign", direction_compare([0, 0, 1], [0, 0, -1])["direction_angle_deg_up_to_sign"] < 1e-6)
    # a synthetic estimate in the estimator's file layout, anchored at T = 3
    T = 3
    root = tmp / "bidirectional" / "backward_rigid"
    (root / "phase2_rigid").mkdir(parents=True)
    (root / "support_coherence").mkdir(parents=True)
    (root / "joint_reduction" / "revolute").mkdir(parents=True)
    (root / "joint_reduction" / "prismatic").mkdir(parents=True)
    th = np.linspace(-0.9, 0.0, T + 1)
    P = np.stack([revolute_transform(a, p, x) for x in th])
    np.save(root / "phase2_rigid" / "poses_original_order.npy", P)
    np.save(root / "phase2_rigid" / "pose_covariances.npy", np.tile(np.eye(6) * 1e-6, (T + 1, 1, 1)))
    (root / "support_coherence" / "support.json").write_text(json.dumps({"members": [0, 1, 2], "num_members": 3}))
    (root / "joint_reduction" / "one_dof_diagnostics.json").write_text(json.dumps({"note": "no joint type is decided"}))
    (root / "joint_reduction" / "relative_increments.json").write_text(json.dumps(
        {"convention": "Delta T_t = T_{t+1} T_t^{-1} (world/spatial)", "zeta": [[0] * 6] * T}))
    (root / "joint_reduction" / "revolute" / "geometry.json").write_text(json.dumps(
        {"refined": {"axis": a.tolist(), "axis_point": p.tolist(), "theta": th.tolist()},
         "pose_residual": {"unweighted_energy": 1e-6}, "axis_radius_over_part_scale": 0.5}))
    (root / "joint_reduction" / "prismatic" / "geometry.json").write_text(json.dumps(
        {"refined": {"direction": [1, 0, 0], "s": [0.0] * (T + 1)}, "pose_residual": {"unweighted_energy": 1e-2}}))
    X = np.random.default_rng(0).normal(size=(5, 3)) * 0.2
    e = DirectionEstimate(tmp / "bidirectional", "backward", T, 5, anchor_xyz=X)
    check("[9] the estimate is read, anchor = identity, convention verified",
          e.status == "USABLE" and e.checks["anchor_is_identity"] == 0.0 and e.checks["increment_convention_ok"],
          e.status)
    check("[9] no type decided -> both hypotheses carried, residual is evidence only",
          "NONE" in e.describe()["decided_type"] and e.residual_preference["lower_residual"] == "revolute")
    check("[9] frame reversal: T(0<-T) is the inverse of T(T<-0)",
          np.allclose(e.transform(T, 0) @ e.transform(0, T), np.eye(4)))
    check("[9] transform = T_part(t) T_part(ref)^-1 composes through intermediate frames",
          np.allclose(e.transform(1, 3) @ e.transform(0, 1), e.transform(0, 3)))
    return root


def t10_untrusted(tmp: Path):
    from observation_diagnostics.joint_evidence import DirectionEstimate
    print("\n[10] untrusted joint estimates")
    root = tmp / "b2" / "bidirectional" / "forward_rigid"
    for d in ("phase2_rigid", "support_coherence", "joint_reduction/revolute", "joint_reduction/prismatic"):
        (root / d).mkdir(parents=True)
    P = np.tile(np.eye(4), (4, 1, 1))
    P[1:, 0, 3] = 1e-4
    np.save(root / "phase2_rigid" / "poses_original_order.npy", P)
    np.save(root / "phase2_rigid" / "pose_covariances.npy", np.tile(np.eye(6), (4, 1, 1)))
    (root / "support_coherence" / "support.json").write_text(json.dumps({"members": list(range(5))}))
    (root / "joint_reduction" / "one_dof_diagnostics.json").write_text("{}")
    (root / "joint_reduction" / "relative_increments.json").write_text(json.dumps(
        {"convention": "Delta T_t = T_{t+1} T_t^{-1} (world/spatial)"}))
    for k in ("revolute", "prismatic"):
        (root / "joint_reduction" / k / "geometry.json").write_text(json.dumps(
            {"refined": {"axis": [0, 0, 1], "axis_point": [0, 0, 0], "theta": [0] * 4,
                         "direction": [1, 0, 0], "s": [0] * 4}}))
    e = DirectionEstimate(tmp / "b2" / "bidirectional", "forward", 0, 5, anchor_xyz=np.zeros((5, 3)))
    check("[10] a motionless estimate is NO_MEASURABLE_MOTION (J1 not run on it)",
          e.status == "NO_MEASURABLE_MOTION", e.status)
    check("[10] ownership of every row is reported as uninformative",
          any("not informative" in n for n in e.notes))
    P[2, 0, 0] = 2.0                                          # not a rotation
    np.save(root / "phase2_rigid" / "poses_original_order.npy", P)
    e2 = DirectionEstimate(tmp / "b2" / "bidirectional", "forward", 0, 5)
    check("[10] a malformed pose track is DEGENERATE, never used", e2.status == "DEGENERATE", e2.status)
    t7_transport(wrong_joint=True)


def t11_firewall(tmp: Path):
    from observation_diagnostics.gt_firewall import Firewall, GTAccessError, violation
    print("\n[11] GT access firewall")
    bad = ["/r/scene/bidirectional/forward_rigid/joint_reduction/gt_evaluation.json",
           "/r/scene/bidirectional/backward_rigid/summary.json", "/r/SUMMARY.txt",
           "/d/scene/metadata/joint_info.json", "/d/scene/metadata/frame_values.csv",
           "/d/scene/pointcloud/points3D_parts.ply", "/d/scene/mask/cam_000/frame_000000.png",
           "/d/scene/mesh/part.obj", "/d/scene/robot.urdf", "/d/scene/depth_gt.npy"]
    good = ["/d/scene/rgb/cam_000/frame_000000.png", "/d/scene/depth_npy/cam_000/frame_000000.npy",
            "/d/scene/colmap/sparse/0/cameras.bin", "/r/scene/static_bidi/final/point_cloud/iteration_30000/point_cloud.ply",
            "/r/scene/bidirectional/backward_rigid/joint_reduction/revolute/geometry.json",
            "/r/scene/bidirectional/backward_rigid/phase2_rigid/poses_original_order.npy"]
    check("[11] every GT/evaluation/mesh/mask path is refused", all(violation(p) for p in bad),
          str([p for p in bad if not violation(p)]))
    check("[11] measured RGB, metric depth, calibration, models and ESTIMATED joints are allowed",
          not any(violation(p) for p in good), str([p for p in good if violation(p)]))
    root = tmp / "fw"
    (root / "joint_reduction").mkdir(parents=True)
    (root / "joint_reduction" / "gt_evaluation.json").write_text("{}")
    (root / "rgb").mkdir()
    (root / "rgb" / "ok.txt").write_text("x")
    fw = Firewall.install([str(root)])
    refused = False
    try:
        open(root / "joint_reduction" / "gt_evaluation.json").read()
    except GTAccessError:
        refused = True
    ok = (root / "rgb" / "ok.txt").read_text() == "x"
    check("[11] the live audit hook refuses the file before reading it", refused)
    check("[11] and lets sensor data through", ok)
    try:
        np.load(root / "joint_reduction" / "gt_evaluation.json")
        via_numpy = False
    except GTAccessError:
        via_numpy = True
    except Exception:
        via_numpy = False
    check("[11] it also catches opens through numpy", via_numpy)


def real_run(tmp: Path):
    print("\n[1, 12-15] real washer run (2 cameras, GPU-capped)")
    out = tmp / "real"
    cmd = [sys.executable, "-W", "ignore", str(REPO / "tools" / "analyze_fb_endpoint_correspondence.py"),
           "--scene", "washer_103776", "--tag", "t", "--cameras", "cam_000,cam_004", "--execute",
           "--gpu-memory-fraction", os.environ.get("FB_TEST_GPU_FRACTION", "0.25"), "--out-dir", str(out)]
    r = subprocess.run(cmd, capture_output=True, text=True)
    check("[15] the real run completes", r.returncode == 0, (r.stdout + r.stderr)[-400:])
    if r.returncode != 0:
        return
    S = json.loads((out / "endpoint_summary.json").read_text())
    C = json.loads((out / "tests_and_checks.json").read_text())
    ri = C["checks"]["RENDER_AND_INDEXING"]
    check("[1] F0 = initial PLY, BT = final PLY, B frame 0 is transported, rows preserved, one gauge",
          ri["F0_is_initial_ply"] and ri["BT_is_final_ply"] and ri["B_frame0_is_transported_not_endpoint"]
          and ri["camera_gauge_max_abs"] == 0.0, json.dumps(ri)[:200])
    w = C["renderer_conservation_worst"]
    check("[2] every rendered view: sum q = A, depth rebuilt, final T bitwise",
          w["final_T_bitwise_all"] and w["max_abs_sum_q_minus_A"] < 5e-6, json.dumps(w))
    fl = C["checks"]["FROZEN_LINEARIZATION"]
    check("[3] real view: adjoint identity and central FD on frozen-contributor pixels",
          fl["adjoint_relative_gap"] < 1e-4 and fl["jvp_vs_finite_difference_relative"] < 0.05
          and fl["of_which_frozen_contributor_lists"] > 100, json.dumps(fl))
    check("[11] firewall: nothing refused, report written", not C["gt_firewall"]["refused"]
          and C["gt_firewall"]["opened_files"] > 0)
    check("[12] protected inputs byte-identical", not C["immutability"]["changed"]
          and C["immutability"]["files"] > 100, f"{C['immutability']['files']} files")
    check("[12] no model checkpoint or merged PLY written", not list(out.rglob("*.ply")))
    rp = C["checks"]["REPEATABILITY"]
    check("[13] a view rendered twice is identical", rp["same_view_rendered_twice_max_abs_rgb"] == 0.0
          and rp["contributor_lists_identical"])
    r2 = subprocess.run(cmd, capture_output=True, text=True)
    check("[13] rerunning into the tag without --resume is refused", r2.returncode != 0 and "--resume" in r2.stdout + r2.stderr)
    r3 = subprocess.run(cmd + ["--resume"], capture_output=True, text=True)
    check("[13] --resume reuses the complete experiment", r3.returncode == 0 and "reused" in r3.stdout)
    ok = all(v["sum_check"] for v in S["accounting"].values())
    check("[14] every source row accounted for exactly once", ok, json.dumps(S["accounting"]))
    check("[15] J1 ran with the backward estimate; the forward one was not used",
          S["flags"]["ESTIMATED_JOINT_PROVENANCE"] == "AVAILABLE")
    labels = S["patch_labels"]
    check("[15] patches were analysed with J0 labels", sum(labels.values()) > 0, json.dumps(labels)[:200])


def main() -> int:
    if not torch.cuda.is_available():
        print("NOT_RUN: CUDA unavailable")
        return 0
    if os.environ.get("FB_TEST_GPU_FRACTION"):
        torch.cuda.set_per_process_memory_fraction(float(os.environ["FB_TEST_GPU_FRACTION"]))
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        t2_t3()
        t4_layers()
        t5_static()
        t7_transport()
        t6_t8_missing()
        t9_conventions(td)
        t10_untrusted(td)
        t11_firewall(td)
        if "--no-real" not in sys.argv:
            real_run(td)
    print()
    if FAILURES:
        print(f"{len(FAILURES)} of {CHECKS} checks FAILED")
        for f in FAILURES:
            print(f"  - {f}")
        return 1
    print(f"{CHECKS} checks passed (real CUDA)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
