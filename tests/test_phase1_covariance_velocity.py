#!/usr/bin/env python
"""RigEuler3DGS_Phase1_Covariance_Velocity_Corrections_Codex.md: correctness oracles.

CPU
  [E2] dense covariance oracle (2-10 Gaussians, two cameras, co-visible and occluded
       rows, 12D blocks): the production block posterior equals (Lambda_ii + F_ii)^-1,
       the exact marginal [(Lambda + H^T W H)^-1]_ii dominates it (Loewner), the
       overconfidence is QUANTIFIED; a fixed group map G gives G Cov(z) G^T; a shared
       bias makes errors correlated across Gaussians, which the blocks do not represent
  [E5] SO(3) tangent transport: Log(Exp(e) Exp(a)) = a + J_l(a)^-1 e + O(|e|^2)
  [F1] counterfactual: a biased 5 mm position correction under P0_vv = (1 m/s)^2 and
       dt = 0.04 s becomes ~0.125 m/s of velocity; the audit's velocity_gain recovers
       the linear conditional exactly (explained fraction 1)
  [C2] alpha precision arms: coupled kappa = trace(W)/12, declared kappa = 1/sd^2,
       foreground RGB support zeroes off-mask RGB precision, refusals
  [GT] labels: nearest labelled sample of the SAME endpoint, -1 beyond 4 x spacing,
       gt_dynamic -1 for unlabelled, bootstrap PLY fields, optimized-GT loader excludes
       unlabelled, backward sparse-NN diagnostic reads the final cloud
CUDA
  [B]  fixed canonical partition: one hash for the whole pass, recorded membership,
       group lever arms at the current state keep a rotated group rigid (canonical
       lever arms do not), G / G^T adjoint with the override, moving blob still tracked
  [A]  audit: per-camera Fisher incl. alpha == production Fisher; tracking identical
       with / without the audit; attribution counts sum to N; finite FD check
  [GTc] label lineage through densify clone / split / prune; save_ply keeps -1
  [E5c] transport on: PSD posterior; zero rotation -> identical to off

    python tests/test_phase1_covariance_velocity.py
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import math
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "tools"))
CHECKS, FAILURES, NOT_RUN = 0, [], []
CUDA = torch.cuda.is_available()


def check(label, cond, detail=""):
    global CHECKS
    CHECKS += 1
    print(f"  {'ok  ' if cond else 'FAIL'}  {label}  {detail}")
    if not cond:
        FAILURES.append(label)


# ---------------------------------------------------------------------------
# [E2] dense covariance oracle
# ---------------------------------------------------------------------------
def test_dense_oracle():
    print("\n[E2] dense covariance oracle")
    from estimator.covariance import approximate_posterior_information, invert_blocks
    g = torch.Generator().manual_seed(0)
    worst = []
    for n in (2, 5, 10):
        D = 12 * n
        # prior: per-Gaussian 12x12 blocks with pose-velocity cross terms (integrated walk)
        dt = 0.04
        P0 = torch.zeros((n, 12, 12), dtype=torch.float64)
        for i in range(n):
            A = torch.randn(12, 12, generator=g, dtype=torch.float64) * 0.05
            P0[i] = A @ A.T + torch.diag(torch.tensor([1e-6] * 6 + [1.0] * 3 + [0.1] * 3, dtype=torch.float64))
        F = torch.eye(12, dtype=torch.float64)
        F[:6, 6:] = dt * torch.eye(6, dtype=torch.float64)
        Ppred = F @ P0 @ F.T
        Lam = torch.linalg.inv(Ppred)
        # two cameras: pixels see the POSE of 1..3 Gaussians (co-visible) or exactly one (occluded)
        rows = []
        for cam in range(2):
            for p in range(40 * n):
                k = 1 if p % 4 == 0 else int(torch.randint(2, 4, (1,), generator=g))
                ids = torch.randperm(n, generator=g)[:min(k, n)]
                h = torch.zeros(D, dtype=torch.float64)
                for i in ids:
                    h[12 * int(i):12 * int(i) + 6] = torch.randn(6, generator=g, dtype=torch.float64)
                rows.append(h)
        H = torch.stack(rows)
        W = torch.diag(torch.rand(H.shape[0], generator=g, dtype=torch.float64) * 1e4 + 1e2)
        Lfull = torch.block_diag(*Lam) + H.T @ W @ H
        P_exact = torch.linalg.inv(Lfull)
        # production per-Gaussian approximation: Lambda_prior,i + F_ii
        Fii = torch.stack([(H.T @ W @ H)[12 * i:12 * i + 6, 12 * i:12 * i + 6] for i in range(n)])
        Lpost = approximate_posterior_information(Lam, Fii)
        P_approx, _ = invert_blocks(Lpost)
        mine = torch.stack([torch.linalg.inv(Lfull[12 * i:12 * i + 12, 12 * i:12 * i + 12]) for i in range(n)])
        check(f"[E2] n={n}: production block posterior == (Lambda_ii + F_ii)^-1",
              float((P_approx - mine).abs().max() / mine.abs().max()) < 1e-8)
        diffs = [torch.linalg.eigvalsh(P_exact[12 * i:12 * i + 12, 12 * i:12 * i + 12] - P_approx[i]).min()
                 for i in range(n)]
        scale = float(P_exact.diagonal().abs().max())
        check(f"[E2] n={n}: exact marginal dominates the block approximation (Loewner)",
              min(float(d) for d in diffs) >= -1e-9 * scale, f"min eig {min(float(d) for d in diffs):.2e}")
        ratio = torch.stack([P_exact[12 * i:12 * i + 3, 12 * i:12 * i + 3].diagonal() /
                             P_approx[i, :3, :3].diagonal() for i in range(n)])
        worst.append(float(ratio.max()))
        print(f"       n={n}: exact / approximate position variance: median {float(ratio.median()):.3f}, "
              f"max {float(ratio.max()):.3f}")
    check("[E2] the overconfidence is quantified (> 1 wherever pixels are shared)", max(worst) > 1.0,
          f"worst ratio {max(worst):.3f}")
    # fixed group map: z (6 per group) -> Gaussian poses; covariance G Cov(z) G^T
    n, C = 6, 2
    sid = torch.tensor([0, 0, 0, 1, 1, 1])
    mu = torch.randn(n, 3, generator=g, dtype=torch.float64)
    cent = torch.stack([mu[sid == c].mean(0) for c in range(C)])
    G = torch.zeros((6 * n, 6 * C), dtype=torch.float64)
    for i in range(n):
        c = int(sid[i]); r = mu[i] - cent[c]
        G[6 * i:6 * i + 3, 6 * c:6 * c + 3] = torch.eye(3, dtype=torch.float64)
        G[6 * i:6 * i + 3, 6 * c + 3:6 * c + 6] = -torch.tensor(
            [[0, -r[2], r[1]], [r[2], 0, -r[0]], [-r[1], r[0], 0]], dtype=torch.float64)
        G[6 * i + 3:6 * i + 6, 6 * c + 3:6 * c + 6] = torch.eye(3, dtype=torch.float64)
    Hg = torch.randn(80, 6 * n, generator=g, dtype=torch.float64)
    Lz = G.T @ Hg.T @ Hg @ G + torch.eye(6 * C, dtype=torch.float64) * 1e-2
    Pg = G @ torch.linalg.inv(Lz) @ G.T
    cross = float(Pg[0:3, 6:9].abs().max() / Pg[0:3, 0:3].abs().max())
    check("[E2] group covariance G Cov(z) G^T: members of one group are strongly cross-correlated",
          cross > 0.1, f"|P_01| / |P_00| = {cross:.2f} (the per-Gaussian blocks drop this)")
    # shared bias: every pixel row shifted by the same b -> correlated errors
    n = 5
    H1 = torch.zeros((60, 3 * n), dtype=torch.float64)
    for p in range(60):
        H1[p, 3 * (p % n):3 * (p % n) + 3] = torch.randn(3, generator=g, dtype=torch.float64)
    Lx = H1.T @ H1 + torch.eye(3 * n, dtype=torch.float64) * 1e-3
    errs = []
    for k in range(400):
        b = torch.randn(1, generator=g, dtype=torch.float64) * 0.01
        y = torch.randn(60, generator=g, dtype=torch.float64) * 0.001 + b
        errs.append(torch.linalg.solve(Lx, H1.T @ y))
    E = torch.stack(errs)
    corr = torch.corrcoef(E.T)
    off = float((corr - torch.diag(corr.diagonal())).abs().max())
    check("[E2] a shared bias produces cross-Gaussian error correlation the block covariance cannot carry",
          off > 0.3, f"max |corr| between Gaussians {off:.2f}")


# ---------------------------------------------------------------------------
# [E5] SO(3) transport Jacobian
# ---------------------------------------------------------------------------
def test_so3_transport():
    print("\n[E5] SO(3) tangent transport")
    from estimator.manifold import so3_exp, so3_left_jacobian_inverse, so3_log
    g = torch.Generator().manual_seed(1)
    a = torch.randn(64, 3, generator=g, dtype=torch.float64) * 0.6
    errs = {}
    for eps in (1e-2, 1e-3, 1e-4):
        e = torch.randn(64, 3, generator=g, dtype=torch.float64) * eps
        lhs = so3_log(so3_exp(e) @ so3_exp(a))
        rhs = a + (so3_left_jacobian_inverse(a) @ e.unsqueeze(-1)).squeeze(-1)
        errs[eps] = float((lhs - rhs).norm(dim=1).max())
    check("[E5] Log(Exp(e) Exp(a)) = a + J_l(a)^-1 e + O(|e|^2) (left / spatial retraction)",
          errs[1e-3] < 1e-5 and errs[1e-4] < errs[1e-3] / 50, str({k: f"{v:.1e}" for k, v in errs.items()}))


# ---------------------------------------------------------------------------
# [F1] position correction -> velocity
# ---------------------------------------------------------------------------
def test_velocity_counterfactual():
    print("\n[F1] a biased position correction becomes velocity")
    from audit_calibrated_phase1 import velocity_gain
    from estimator.covariance import covariance_predict, make_process_noise
    from estimator.process import error_state_transition, process_predict
    from estimator.state import GaussianMotionState
    n, dt = 4, 0.04
    z = torch.zeros(n, 3, dtype=torch.float64)
    st = GaussianMotionState(mu=torch.randn(n, 3, dtype=torch.float64), R=torch.eye(3, dtype=torch.float64).repeat(n, 1, 1),
                             v=z.clone(), omega=z.clone(), colors=torch.full((n, 3), 0.5, dtype=torch.float64),
                             opacity=torch.full((n, 1), 0.9, dtype=torch.float64),
                             scales=torch.full((n, 3), 0.01, dtype=torch.float64))
    P0 = torch.zeros((n, 12, 12), dtype=torch.float64)
    P0[:, :3, :3] = torch.eye(3, dtype=torch.float64) * 0.00019 ** 2
    P0[:, 3:6, 3:6] = torch.eye(3, dtype=torch.float64) * 1e-6
    P0[:, 6:9, 6:9] = torch.eye(3, dtype=torch.float64) * 1.0
    P0[:, 9:, 9:] = torch.eye(3, dtype=torch.float64) * 0.1
    F = error_state_transition(st, dt, mode="so3").double()
    Q = make_process_noise(n, dt, 1e-3, 1e-3, 0.1, 0.1, dtype=torch.float64)
    Pp = covariance_predict(P0, F, Q)
    check("[F1] predicted position sd ~ dt x 1 m/s = 40 mm (a STANDARD DEVIATION, not a movement)",
          abs(math.sqrt(float(Pp[0, 0, 0])) - 0.04) < 1e-3, f"{1000 * math.sqrt(float(Pp[0, 0, 0])):.2f} mm")
    # exact linear Kalman update with a precise but BIASED position observation of 5 mm
    Hm = torch.zeros(3, 12, dtype=torch.float64); Hm[:, :3] = torch.eye(3, dtype=torch.float64)
    R = torch.eye(3, dtype=torch.float64) * (0.0005 ** 2)
    y = torch.tensor([0.005, 0.0, 0.0], dtype=torch.float64)
    K = Pp @ Hm.T @ torch.linalg.inv(Hm @ Pp @ Hm.T + R)
    dx = (K @ y)
    pred = process_predict(st, dt)
    post = dataclasses.replace(pred, mu=pred.mu + dx[:, :3], v=pred.v + dx[:, 6:9])
    vg = velocity_gain(st, post, Pp, dt)
    dv = float(dx[0, 6])
    check("[F1] 5 mm biased correction -> ~0.125 m/s velocity (delta_v ~ delta_p / dt)",
          abs(dv - 0.125) < 0.01, f"{dv:.4f} m/s")
    check("[F1] the audit recovers the linear pose->twist conditional exactly",
          vg["explained_fraction_q"][0] > 1 - 1e-6, f"explained {vg['explained_fraction_q'][0]:.8f}")
    check("[F1] dv / (dmu / dt) ~ 1: the whole correction was booked as velocity",
          abs(vg["dv_over_dmu_per_dt_q"][0] - 1.0) < 0.05, f"{vg['dv_over_dmu_per_dt_q'][0]:.3f}")


# ---------------------------------------------------------------------------
# [C2] likelihood arms
# ---------------------------------------------------------------------------
def test_alpha_arms():
    print("\n[C2] alpha precision / RGB support arms")
    from estimator.masked_observation import MaskedObservationModel, model_from_args
    from estimator.observation_terms import camera_terms
    from estimator.sparse_iekf import IEKFConfig
    W = torch.full((3, 4, 4), 1.0 / 0.005 ** 2)
    m_c = MaskedObservationModel("masked-rgb-expected-quad", (0.5,) * 3, "uniform01", 0.1)
    m_d = MaskedObservationModel("masked-rgb-expected-quad", (0.5,) * 3, "uniform01", 0.1,
                                 alpha_precision="declared", alpha_sd=0.2)
    check("[C2] coupled kappa = trace(W)/12 = 1/(4 sigma^2) (10000 at sigma 0.005)",
          abs(float(m_c.kappa(W)[0, 0]) - 1e4) < 1e-3)
    check("[C2] declared kappa = 1/alpha_sd^2, independent of W", abs(float(m_d.kappa(W)[0, 0]) - 25.0) < 1e-9
          and abs(float(m_d.kappa(W * 7)[0, 0]) - 25.0) < 1e-9)
    refused = []
    for kw in ({"alpha_precision": "declared"}, {"alpha_sd": 0.1}, {"rgb_support": "foreground"},
               {"alpha_precision": "bogus"}):
        try:
            MaskedObservationModel("masked-rgb-expected-quad", (0.5,) * 3, "uniform01", 0.1, **kw)
            refused.append(False)
        except ValueError:
            refused.append(True)
    check("[C2] declared without sd / sd without declared / foreground with coupled / unknown: refused",
          all(refused), str(refused))
    try:
        model_from_args(argparse.Namespace(observation_mode="masked-rgb-constant", rgb_background_constant=(0.1,) * 3,
                                           rgb_background_distribution=None, alpha_precision="declared",
                                           alpha_sd=0.1, rgb_support="full"))
        check("[C2] the constant mode refuses the alpha arms", False)
    except SystemExit:
        check("[C2] the constant mode refuses the alpha arms", True)
    view = argparse.Namespace(rgb=torch.zeros(3, 4, 4), depth=torch.ones(4, 4), alpha=torch.full((4, 4), 0.5))
    M = torch.zeros(4, 4, dtype=torch.bool); M[:2] = True
    t_full = camera_terms(view, torch.rand(3, 4, 4), torch.ones(4, 4), M, IEKFConfig(), rgb_robust=False,
                          coverage=M.float(), kappa_fn=m_d.kappa)
    t_fg = camera_terms(view, torch.rand(3, 4, 4), torch.ones(4, 4), M, IEKFConfig(), rgb_robust=False,
                        coverage=M.float(), kappa_fn=m_d.kappa, rgb_valid=M)
    check("[C2] foreground support: zero RGB precision off-mask, unchanged on the mask",
          bool((t_fg.p_rgb[:, ~M] == 0).all()) and torch.equal(t_fg.p_rgb[:, M], t_full.p_rgb[:, M]))
    check("[C2] the declared silhouette term is present everywhere (off-mask too)",
          bool((t_fg.kappa == 25.0).all()) and t_fg.e_a > 0)


# ---------------------------------------------------------------------------
# [GT] labels
# ---------------------------------------------------------------------------
def _write_cloud(path, xyz, part):
    from plyfile import PlyData, PlyElement
    arr = np.empty(len(xyz), dtype=[("x", "f4"), ("y", "f4"), ("z", "f4"), ("part", "u1")])
    arr["x"], arr["y"], arr["z"], arr["part"] = xyz[:, 0], xyz[:, 1], xyz[:, 2], part
    path.parent.mkdir(parents=True, exist_ok=True)
    PlyData([PlyElement.describe(arr, "vertex")]).write(str(path))


def test_gt_labels():
    print("\n[GT] static / dynamic labels")
    import torch as _t
    from plyfile import PlyData
    from estimator.motion_coherence_eval import load_static_dynamic_gt
    from estimator.optimized_gt import load_optimized_gt
    from train_tracking_bootstrap import write_model
    from utils.gt_part_labels import dynamic_from_part, labelled_cloud, nn_labels
    rng = np.random.default_rng(0)
    with tempfile.TemporaryDirectory() as td:
        src = Path(td) / "scene"
        g = np.stack(np.meshgrid(np.linspace(0, 1, 21), np.linspace(0, 1, 21)), -1).reshape(-1, 2)
        xyz0 = np.c_[g, np.zeros(len(g))]
        part0 = (xyz0[:, 0] > 0.5).astype(np.uint8)                  # the right half moves
        xyzT = xyz0.copy(); xyzT[part0 == 1, 2] += 0.3                # final: moving half lifted
        _write_cloud(src / "pointcloud" / "points3D_parts.ply", xyz0, part0)
        _write_cloud(src / "pointcloud" / "points3D_parts_final.ply", xyzT, part0)
        check("[GT] real capture (no labelled cloud): nothing", labelled_cloud(Path(td) / "real", "initial") is None)
        c0, cT = labelled_cloud(src, "initial"), labelled_cloud(src, "final")
        seeds = np.r_[xyz0[:200] + rng.normal(0, 0.005, (200, 3)), [[5.0, 5.0, 5.0]]]
        lab = nn_labels(seeds, c0)
        check("[GT] seeds take the part of the nearest labelled sample of the SAME endpoint",
              np.array_equal(lab["part"][:200], part0[:200].astype(np.int64)))
        check("[GT] a seed beyond 4 x the cloud spacing is unlabelled (-1)", lab["part"][-1] == -1,
              f"max_dist {lab['max_dist']:.3f} m")
        movedT = xyzT[part0 == 1][:50]
        check("[GT] the final endpoint uses points3D_parts_final.ply (the lifted moving link is dynamic)",
              bool((nn_labels(movedT, cT)["part"] == 1).all()) and bool((nn_labels(movedT, c0)["part"] == -1).all()))
        check("[GT] gt_dynamic: 1 moving, 0 static, -1 unlabelled",
              dynamic_from_part(np.array([0, 1, 2, -1])).tolist() == [0, 1, 1, -1])
        # bootstrap PLY fields and the optimized-GT loader
        out = Path(td) / "boot"
        n = len(seeds)
        params = {"xyz": seeds, "rgb": np.full((n, 3), 0.5), "opacity": np.full(n, 0.9),
                  "scales": np.full((n, 3), 0.01), "quat_wxyz": np.tile([1.0, 0, 0, 0], (n, 1))}
        ply = write_model(out, 7, params, src, "colmap", lab["part"])
        v = PlyData.read(str(ply))["vertex"]
        check("[GT] bootstrap PLY carries gt_part / gt_dynamic, -1 kept", "gt_part" in v.data.dtype.names
              and float(v["gt_dynamic"][-1]) == -1.0 and float(v["gt_part"][-1]) == -1.0)
        (out / "bootstrap_manifest.json").write_text(json.dumps({"gt_labels": {"status": "LABELLED"}}))
        gt = load_optimized_gt(out, _t.tensor(seeds))
        check("[GT] optimized-GT loader: unlabelled excluded (label -1), coverage < 1, bootstrap provenance",
              gt.get("status") == "FOUND" and int(gt["labels"][-1]) == -1 and gt["num_unmapped"] == 1
              and gt["coverage"] < 1 and "Phase-0 bootstrap" in gt["provenance"], str(gt.get("num_unmapped")))
        sp_f = load_static_dynamic_gt(src, _t.tensor(movedT), endpoint="final")
        sp_i = load_static_dynamic_gt(src, _t.tensor(movedT), endpoint="initial")
        check("[GT] backward sparse-NN diagnostic reads the FINAL cloud (initial would leave them unmapped)",
              int((sp_f["labels"] == 1).sum()) == 50 and int((sp_i["labels"] >= 0).sum()) == 0)


# ---------------------------------------------------------------------------
# CUDA
# ---------------------------------------------------------------------------
def test_cuda():
    if not CUDA:
        NOT_RUN.append("[B]/[A]/[GTc]/[E5c] (CUDA)")
        return
    from diff_gaussian_rasterization import _C
    from estimator.supernodes import GroupedObservation, build_supernodes
    from estimator.manifold import so3_exp
    print("\n[B] fixed canonical partition")
    torch.manual_seed(0)
    mu = torch.randn(400, 3, device="cuda") * 0.2
    part = build_supernodes(mu, target_size=50, seed=0)
    # rotate ONE group rigidly by 1.2 rad about its centroid
    sid = part.supernode_id.long()
    k = 0
    m = sid == k
    Rg = so3_exp(torch.tensor([[0.0, 0.0, 1.2]], device="cuda"))[0]
    c0 = mu[m].mean(0)
    cur = mu.clone()
    cur[m] = (mu[m] - c0) @ Rg.T + c0 + torch.tensor([0.1, 0.0, 0.0], device="cuda")
    cnt = torch.bincount(sid, minlength=part.num_supernodes).clamp_min(1).float()
    cent = torch.zeros((part.num_supernodes, 3), device="cuda").index_add_(0, sid, cur) / cnt[:, None]
    G_cur = GroupedObservation(None, part, canonical_means=cur, centroids=cent)
    G_can = GroupedObservation(None, part)
    dz = torch.zeros((part.num_supernodes, 6), device="cuda")
    dz[k, 3:] = torch.tensor([0.3, -0.2, 0.5], device="cuda")        # pure group rotation increment

    def rigidity(G):
        d = G.expand(dz)[m][:, :3]
        x = cur[m]
        # d/de |x_i - x_j|^2 = 2 (x_i - x_j).(d_i - d_j): zero for a rigid increment
        dx = x[:, None] - x[None]
        dd = d[:, None] - d[None]
        return float(((dx * dd).sum(-1)).abs().max() / (dx.norm(dim=-1) * dd.norm(dim=-1)).max().clamp_min(1e-12))
    rc, rk = rigidity(G_cur), rigidity(G_can)
    check("[B] current-geometry lever arms keep a rotated group's increment rigid", rc < 1e-5, f"{rc:.2e}")
    check("[B] canonical lever arms on a group rotated 1.2 rad do NOT (why 'current' is used)", rk > 1e-2,
          f"{rk:.2e}")
    v = torch.randn(400, 6, device="cuda")
    lhs = float((G_cur.expand(dz) * v).sum())
    rhs = float((dz * G_cur.reduce(v)).sum())
    check("[B] <G z, v> = <z, G^T v> with the current-geometry override", abs(lhs - rhs) / max(abs(lhs), 1e-12) < 1e-5)

    # production Phase 1 on a moving blob, fixed vs per-frame partition
    from tests.test_iekf_end_to_end import camera, make_state, render_gt
    from tools.kpart_pipeline import bootstrap_positions
    from estimator.sparse_iekf import IEKFConfig
    from audit_calibrated_phase1 import Phase1Audit
    cams = [camera(), camera(yaw=0.6, radius=1.2)]
    truth = make_state(n=60, seed=4, spread=0.4)
    cfg = IEKFConfig(max_iters=4)
    frames = []
    for t in range(4):
        moved = dataclasses.replace(truth, mu=truth.mu + torch.tensor([0.004 * t, 0, 0], device="cuda"))
        frames.append((render_gt(moved, cams, cfg), {"original_frame_index": t}))

    def go(mode, audit=None):
        a = argparse.Namespace(supernode_size=12, seed=0, bootstrap_iters=4, dt=1 / 25.0, visibility_constraint=None,
                               masked_observation=None, phase1_audit=audit, phase1_partition=mode)
        pos, info = bootstrap_positions(truth, cams, frames, cfg, None, a, verbose=False)
        return torch.stack([p.double() for p in pos]), info
    p_can, i_can = go("canonical")
    p_leg, i_leg = go("per-frame")
    pinfo = i_can["partition"]
    check("[B] canonical mode: one partition built from the endpoint, hash recorded",
          pinfo["mode"] == "canonical" and len(pinfo["canonical_hash"]) == 64
          and i_can["partition_arrays"]["supernode_id"].shape[0] == 60
          and np.allclose(i_can["partition_arrays"]["canonical_means"], truth.mu.cpu().numpy()))
    check("[B] legacy per-frame mode is still runnable", i_leg["partition"]["mode"] == "per-frame"
          and i_leg["partition_arrays"] is None)
    check("[B] Gaussian ordering preserved (frame 0 == canonical, by index)",
          float((p_can[0] - truth.mu.double()).abs().max()) == 0.0)
    disp = float((p_can[-1] - p_can[0])[:, 0].median())
    check("[B] a moving blob is still tracked under the fixed partition (not frozen)", disp > 0.006,
          f"median x displacement {1000 * disp:.1f} mm of 12 mm")
    # CUDA atomics: identical runs differ; the floor is measured from 3 control runs
    reps = [p_can] + [go("canonical")[0] for _ in range(2)]
    floor = max(float((a_ - b_).abs().max()) for i_, a_ in enumerate(reps) for b_ in reps[i_ + 1:])
    au = Phase1Audit(fd=True, forces=True)
    p_aud, _ = go("canonical", au)
    d_aud = float((p_can - p_aud).abs().max())
    check("[A] tracking with the extended audit equals tracking without it (within 2 x the measured floor)",
          d_aud <= 2 * max(floor, 1e-9), f"audit {d_aud:.1e}, floor {floor:.1e}")
    r = au.rows[-1]
    check("[A] audit Fisher (sum of cameras) == production measurement_operators Fisher",
          r["fisher_check"]["rel_err_sum_cameras_vs_production"] < 1e-5,
          f"{r['fisher_check']['rel_err_sum_cameras_vs_production']:.1e}")
    att = r["attribution"]
    check("[A] attribution: own / group-coupled / prediction-only add up to N",
          att["own_measurement"] + att["no_own_information_moved_by_group"] + att["prediction_only"] == 60)
    fd = r["fd_check"]["cameras"]
    check("[A] FD check of the cached Jacobian is finite and small on RGB",
          all(math.isfinite(c["rgb_rel_err"]) for c in fd) and max(c["rgb_rel_err"] for c in fd) < 0.05,
          str([round(c["rgb_rel_err"], 4) for c in fd]))
    check("[A] solver status recorded; iteration_budget is never 'converged'",
          r["solver"]["posterior_status"] in ("UNCERTIFIED_NOT_CONVERGED", "LOCAL_HESSIAN_AT_CONVERGED_STATE")
          and (r["solver"]["stop_reason"] != "iteration_budget" or not r["solver"]["converged"]))
    check("[A] forces at the canonical and prior states recorded per camera",
          len(r["forces_at"]["prior"]["cameras"]) == 2 and len(r["forces_at"]["canonical"]["cameras"]) == 2)

    print("\n[A] alpha Fisher in the masked mode")
    from estimator.masked_observation import MaskedObservationModel, prepare
    from estimator.renderer_bridge import build_linearization
    from estimator.sparse_iekf import _frame_terms, measurement_operators
    from tests.test_unified_cuda_renderer_jvp_vjp import make_cameras
    from tests.test_visibility_constrained_iekf_phase1 import blob, observe
    mc = make_cameras(n_cam=3, H=64, W=64, radius=2.2)
    ob = blob()
    gt, masks = observe(ob, mc)
    model = MaskedObservationModel("masked-rgb-expected-quad", (0.5,) * 3, "uniform01", 0.1)
    prep = [prepare(g_[0], g_[1], g_[2], m_, f"cam_{i:03d}", 1) for i, (g_, m_) in enumerate(zip(gt, masks))]
    moved = dataclasses.replace(ob, mu=ob.mu + torch.tensor([0.01, 0.0, 0.0], device="cuda"))
    lin = build_linearization(moved, mc, background=model.background_tensor("cuda"))
    terms = _frame_terms(lin.views, prep, IEKFConfig(), observation_model=model)
    Fs = None
    for c, t in enumerate(terms):
        pc = [(torch.zeros_like(x.p_rgb), torch.zeros_like(x.p_d)) for x in terms]
        pc[c] = (t.p_rgb, t.p_d)
        ka = [torch.zeros_like(x.r_d) for x in terms]
        ka[c] = t.kappa
        Fc = lin.build_block_fisher(pc) + lin.build_alpha_fisher(ka)
        Fs = Fc if Fs is None else Fs + Fc
    Fp = measurement_operators(lin, [x.as_prec4() for x in terms], [(x.kappa, x.r_a) for x in terms])[2]()
    Fno = sum(lin.build_block_fisher([(x.p_rgb, x.p_d) if j == c else (torch.zeros_like(x.p_rgb), torch.zeros_like(x.p_d))
                                      for j, x in enumerate(terms)]) for c in range(len(terms)))
    e_with = float((Fs - Fp).norm() / Fp.norm())
    e_without = float((Fno - Fp).norm() / Fp.norm())
    check("[A] per-camera Fisher with the alpha block reproduces production", e_with < 1e-5, f"{e_with:.1e}")
    check("[A] ...and the old audit (no alpha block) did not", e_without > 1e-3, f"{e_without:.1e}")

    print("\n[E5c] covariance transport")
    from estimator.sparse_iekf import iekf_frame_step
    from tests.test_visibility_constrained_iekf_phase1 import cov
    st = make_state(n=6, seed=2)
    obs = render_gt(dataclasses.replace(st, mu=st.mu + 0.003), [camera()], cfg)
    outs = {}
    for flag in (False, True):
        c2 = dataclasses.replace(cfg, covariance_tangent_transport=flag)
        _x, P, d = iekf_frame_step(st, cov(6), obs, [camera()], dt=0.04, cfg=c2)
        outs[flag] = (P, d)
    ev = torch.linalg.eigvalsh(outs[True][0].double().cpu())
    post = next(r_ for r_ in outs[True][1].rows if r_.get("event") == "posterior")
    check("[E5c] transport on: PSD posterior, labelled as a block approximation",
          float(ev.min()) >= -1e-10 * float(ev.max()) and post["approximation"] == "BLOCK_INFORMATION_APPROXIMATION"
          and post["covariance_tangent_transport"] is True)
    rel = float((outs[True][0] - outs[False][0]).abs().max() / outs[False][0].abs().max())
    check("[E5c] zero-omega state with tiny rotation updates: transport changes P only at second order",
          rel < 1e-3, f"{rel:.1e}")

    print("\n[GTc] label lineage through clone / split / prune")
    from arguments import OptimizationParams
    from scene.gaussian_model import GaussianModel
    from utils.graphics_utils import BasicPointCloud
    ap = argparse.ArgumentParser()
    op = OptimizationParams(ap)
    opt = op.extract(ap.parse_args([]))
    npt = 200
    rng = np.random.default_rng(0)
    pts = rng.normal(0, 0.3, (npt, 3))
    part = (pts[:, 0] > 0).astype(np.int64)
    part[:7] = -1                                        # some unlabelled seeds
    gm = GaussianModel(0)
    cols = np.zeros((npt, 3)); cols[:, 0] = np.arange(npt) / npt  # parent id encoded in colour
    gm.create_from_pcd(BasicPointCloud(points=pts, colors=cols, normals=np.zeros_like(pts), part_indices=part), [], 1.0)
    gm.training_setup(opt)
    parent_code = gm._features_dc[:, 0, 0].detach().clone()
    gm.tmp_radii = torch.ones(npt, device="cuda")
    grads = torch.ones((npt, 1), device="cuda")
    gm.densify_and_clone(grads, 0.5, 100.0)
    gm.densify_and_split(torch.ones((gm.get_xyz.shape[0], 1), device="cuda"), 0.5, 1e-3)
    keep = torch.rand(gm.get_xyz.shape[0], device="cuda") > 0.3
    gm.prune_points(~keep)
    code = gm._features_dc[:, 0, 0].detach()
    pid = torch.cdist(code[:, None], parent_code[:, None]).argmin(1).cpu().numpy()
    labs = gm._gt_part_labels.cpu().numpy()
    check("[GTc] every clone / split child carries its parent's label (incl. -1) after prune",
          labs.shape[0] == gm.get_xyz.shape[0] and np.array_equal(labs, part[pid]),
          f"{gm.get_xyz.shape[0]} Gaussians from {npt}")
    with tempfile.TemporaryDirectory() as td:
        pth = Path(td) / "pc.ply"
        gm.save_ply(str(pth))
        from plyfile import PlyData
        vv = PlyData.read(str(pth))["vertex"]
        gd, gp = np.asarray(vv["gt_dynamic"]), np.asarray(vv["gt_part"])
        check("[GTc] save_ply: unlabelled stays -1 in gt_dynamic (never 'static')",
              bool((gd[gp < 0] == -1).all()) and bool((gd[gp == 0] == 0).all()) and bool((gd[gp > 0] == 1).all()))


def main() -> int:
    for t in (test_dense_oracle, test_so3_transport, test_velocity_counterfactual, test_alpha_arms,
              test_gt_labels, test_cuda):
        try:
            t()
        except Exception as ex:                       # noqa: BLE001
            import traceback
            traceback.print_exc()
            check(f"{t.__name__} raised", False, f"{type(ex).__name__}: {ex}"[:200])
    print()
    if NOT_RUN:
        print(f"NOT_RUN: {', '.join(NOT_RUN)}")
    if FAILURES:
        print(f"{len(FAILURES)} of {CHECKS} checks FAILED")
        return 1
    print(f"{CHECKS} checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
