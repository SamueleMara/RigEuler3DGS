#!/usr/bin/env python
"""Regression tests of CODEX_FIX_MASKED_IEKF_DEFAULT_AND_VALIDATION_2026-10-21.md.

  [A] default observation mode and the strict mask preflight (P0-1)
  [B] driver dry runs: default -> masked in every stage, explicit original,
      missing masks -> skipped before anything runs (needs the SSD)
  [C] Student-t NLL: gradient = IRLS score, nu = inf = Gaussian, invalid samples
      absent, a candidate the re-weighted surrogate misranks (P1-1)
  [D] depth validity: off-valid depth NaN / 0 / huge / random -> identical terms,
      positive finite Z only, alpha-gate counts; mask = 0 / 1 semantics (P1-3)
  [E] original-mode precisions equal the pre-patch `_precisions_legacy`
  [F] kappa = trace(W Sigma_b) = trace(W)/12 for the declared W; W fixed across states
  [G] convergence_decision and logged stop reasons (P0-4)
  [H] Phase 1 recursion: P_post of frame k is the prior of frame k+1; a
      zero-information frame returns F P F^T + Q, PSD with cross terms (P0-3)
  [I] Phase 2 exact SE(3) prior: e(x_minus boxplus d) = d for nonzero rotation and
      twist, FD Jacobian first-order exact, not the identity away from x_minus (P0-2)
  [J] Phase 3 strong prior: monotone total MAP, total = sum of parts, stationary
      total, an image-improving / total-worsening candidate is not the answer,
      geometry bitwise unchanged, qdot posterior = prior (no image information)
  [K] experiment signatures: absent / own incomplete / reuse / refuse (P0-5)
  [L] model-error audit on a stray-opacity toy (P1-2)
  [M] mask diagnostics re-render the tracked ORIENTATIONS (P1-4.5)

    python tests/test_masked_default_repair.py
"""
from __future__ import annotations

import argparse
import copy
import dataclasses
import json
import math
import subprocess
import sys
import tempfile
import types
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "tools"))
CHECKS, FAILURES, NOT_RUN = 0, [], []
SSD = Path("/media/samuelemara/Extreme SSD/Samuele")
DATA = SSD / "Backup_09_10_26/Sapien/output/preparation_8cams/partnet_subset_fixed_render_linear"
STATIC = SSD / "results_11_10_26/ours/fridge_10905/static_bidi"
CUDA = torch.cuda.is_available()


def check(label, cond, detail=""):
    global CHECKS
    CHECKS += 1
    print(f"  {'ok  ' if cond else 'FAIL'}  {label}  {detail}")
    if not cond:
        FAILURES.append(label)


def not_run(label, why):
    print(f"  NOT_RUN  {label}  ({why})")
    NOT_RUN.append(label)


def view(rgb, depth, alpha):
    return types.SimpleNamespace(rgb=rgb, depth=depth, alpha=alpha)


def write_png(path, arr):
    from PIL import Image
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(arr).save(path)


# ---------------------------------------------------------------------------
def test_A_default_and_preflight():
    print("\n[A] default mode and mask preflight")
    from estimator.masked_observation import DEFAULT_MODE, add_cli, preflight
    ap = argparse.ArgumentParser()
    add_cli(ap)
    a = ap.parse_args([])
    check("[A] the CLI default is masked-rgb-expected-quad",
          a.observation_mode == DEFAULT_MODE == "masked-rgb-expected-quad", a.observation_mode)
    for f in ("run_motion_coherence_support.py", "run_rigid_se3_pipeline.py", "run_joint_phase3.py"):
        src = (REPO / "tools" / f).read_text()
        check(f"[A] {f} registers the shared CLI and runs the preflight",
              "add_cli" in src and "_obs_preflight(args, source)" in src)
    with tempfile.TemporaryDirectory() as td:
        src = Path(td)
        rgb = np.zeros((6, 8, 3), np.uint8)
        for c in ("cam_000", "cam_001"):
            for k in range(2):
                write_png(src / "rgb" / c / f"frame_{k:06d}.png", rgb)

        def ns(**kw):
            d = dict(observation_mode=DEFAULT_MODE, object_mask_constraint="off", mask_source=None,
                     cameras=2, frames=2)
            d.update(kw)
            return argparse.Namespace(**d)

        def refused(a):
            try:
                preflight(a, src)
                return None
            except SystemExit as ex:
                return str(ex)
        check("[A] no mask directory: the default mode stops before any load", refused(ns()) is not None,
              (refused(ns()) or "")[:70])
        check("[A] explicit original needs no masks", refused(ns(observation_mode="original")) is None)
        m = np.zeros((6, 8), np.uint8)
        m[2:4, 2:5] = 255
        for c in ("cam_000", "cam_001"):
            for k in range(2):
                write_png(src / "mask" / c / f"frame_{k:06d}.png", m)
        check("[A] complete binary masks pass", refused(ns()) is None, refused(ns()) or "")
        check("[A] masked mode + hard QP is refused",
              refused(ns(object_mask_constraint="visible-negative-hard")) is not None)
        bad = m.copy()
        bad[0, 0] = 128
        write_png(src / "mask" / "cam_001" / "frame_000001.png", bad)
        check("[A] a non-binary mask is refused (no thresholding)", refused(ns()) is not None)
        (src / "mask" / "cam_001" / "frame_000001.png").unlink()
        check("[A] one missing mask is refused", refused(ns()) is not None)


def test_B_driver_dry_runs():
    print("\n[B] driver dry runs")
    if not (DATA / "fridge_10905").is_dir() or not (STATIC / "initial").is_dir():
        not_run("[B] driver dry runs", "SSD dataset / static models not mounted")
        return
    with tempfile.TemporaryDirectory() as td:
        out = Path(td) / "out"
        (out / "fridge_10905" / "static_bidi").mkdir(parents=True)
        for e in ("initial", "final"):
            (out / "fridge_10905" / "static_bidi" / e).symlink_to(STATIC / e)

        def dry(*extra):
            r = subprocess.run(["bash", "run_bidirectional_all.sh", "-i", str(DATA), "-o", str(out),
                                "--scenes", "fridge_10905", "--stages", "phase1 phase2 phase3",
                                "--frames", "2", "--dry-run", *extra],
                               cwd=REPO, capture_output=True, text=True, timeout=600)
            lines = [ln for ln in r.stdout.splitlines()
                     if any(t in ln for t in ("run_motion_coherence_support", "run_rigid_se3_pipeline",
                                              "run_joint_phase3"))]
            return r.stdout, lines
        txt, cmds = dry()
        check("[B] default: six stage commands", len(cmds) == 6, f"{len(cmds)} commands")
        check("[B] default: every stage gets --observation-mode masked-rgb-expected-quad",
              cmds and all("--observation-mode masked-rgb-expected-quad" in c for c in cmds))
        txt, cmds = dry("--observation-mode", "original")
        check("[B] original: every stage gets --observation-mode original, no mask source",
              len(cmds) == 6 and all("--observation-mode original" in c and "--mask-source" not in c
                                     for c in cmds), f"{len(cmds)} commands")
        txt, cmds = dry("--mask-subdir", "no_such_masks")
        check("[B] missing masks: the scene is skipped before any stage",
              "SKIPPED(masks not ready)" in txt and not cmds)
        check("[B] dry runs wrote nothing into the output root",
              sorted(p.name for p in (out / "fridge_10905").iterdir()) == ["static_bidi"])


# ---------------------------------------------------------------------------
def test_C_student_t():
    print("\n[C] Student-t objective")
    from estimator.measurement import robust_term, student_t_nll
    g = torch.Generator().manual_seed(0)
    r = 3.0 * torch.randn(400, generator=g, dtype=torch.float64)
    s = 0.2 + torch.rand(400, generator=g, dtype=torch.float64)
    v = torch.ones(400, dtype=torch.bool)
    nu = 4.0
    prec, E = robust_term(r, s, v, True, nu)
    h, worst = 1e-6, 0.0
    for i in range(0, 400, 37):
        rp, rm = r.clone(), r.clone()
        rp[i] += h
        rm[i] -= h
        fd = (robust_term(rp, s, v, True, nu)[1] - robust_term(rm, s, v, True, nu)[1]) / (2 * h)
        worst = max(worst, abs(fd - float(prec[i] * r[i])) / max(abs(fd), 1e-12))
    check("[C] dE/dr = IRLS precision x residual (w(z) r / sigma^2)", worst < 1e-5, f"worst rel {worst:.2e}")
    _, Einf = robust_term(r, s, v, True, float("inf"))
    _, Eg = robust_term(r, s, v, False, nu)
    ref = 0.5 * float(((r / s) ** 2).sum())
    check("[C] nu = inf is the Gaussian objective", abs(Einf - ref) < 1e-9 * ref and abs(Eg - ref) < 1e-9 * ref)
    bad = r.clone()
    inval = torch.zeros(400, dtype=torch.bool)
    inval[::5] = True
    for k, val in enumerate((float("nan"), float("inf"), 1e30, -7.0)):
        bad[k * 5::20] = val
    p1, E1 = robust_term(r, s, ~inval, True, nu)
    p2, E2 = robust_term(bad, s, ~inval, True, nu)
    check("[C] invalid samples are absent: garbage there changes nothing",
          E1 == E2 and torch.equal(p1, p2) and bool((p2[inval] == 0).all()) and math.isfinite(E2))
    # the per-trial re-weighted surrogate 1/2 sum w(z) z^2 misranks: it prefers an
    # outlier that EXPLODES (z: 10 -> 100) because w(z) z^2 saturates at nu + 1
    one = torch.ones(2, dtype=torch.float64)
    A, B = torch.tensor([1.0, 10.0], dtype=torch.float64), torch.tensor([0.0, 100.0], dtype=torch.float64)

    def surrogate(z):
        w = (nu + 1) / (nu + z * z)
        return 0.5 * float((w * z * z).sum())
    nllA = robust_term(A, one, one.bool(), True, nu)[1]
    nllB = robust_term(B, one, one.bool(), True, nu)[1]
    check("[C] surrogate prefers the exploding outlier, the NLL does not",
          surrogate(B) < surrogate(A) and nllA < nllB,
          f"surrogate A {surrogate(A):.3f} B {surrogate(B):.3f}; NLL A {nllA:.3f} B {nllB:.3f}")
    check("[C] student_t_nll is the summed objective",
          abs(float(student_t_nll(A, nu).sum()) - nllA) < 1e-12)


# ---------------------------------------------------------------------------
def _toy_camera(seed=0, H=8, W=10):
    g = torch.Generator().manual_seed(seed)
    rgb_r = torch.rand(3, H, W, generator=g)
    dep = 1.0 + 0.2 * torch.rand(H, W, generator=g)
    alpha = torch.rand(H, W, generator=g)
    obs_rgb = (rgb_r + 0.05 * torch.randn(3, H, W, generator=g)).clamp(0, 1)
    obs_d = dep + 0.01 * torch.randn(H, W, generator=g)
    valid = torch.rand(H, W, generator=g) > 0.3
    mask = torch.rand(H, W, generator=g) > 0.5
    return view(rgb_r, dep, alpha), obs_rgb, obs_d, valid, mask


def test_D_depth_validity_and_masks():
    print("\n[D] depth validity, alpha gate, mask semantics")
    from estimator.masked_observation import MaskedObservationModel, phase1_camera_terms, prepare
    from estimator.observation_terms import camera_terms
    from estimator.sparse_iekf import IEKFConfig
    cfg = IEKFConfig()
    v, rgb, d, valid, mask = _toy_camera()
    ref = camera_terms(v, rgb, d, valid, cfg, rgb_robust=True)
    same = True
    for val in (float("nan"), 0.0, 1e9, None):
        d2 = d.clone()
        d2[~valid] = (torch.rand(int((~valid).sum())) * 50 - 25) if val is None else val
        t = camera_terms(v, rgb, d2, valid, cfg, rgb_robust=True)
        same &= (t.e_d == ref.e_d and torch.equal(t.p_d, ref.p_d) and torch.equal(t.r_d, ref.r_d)
                 and t.depth_used == ref.depth_used)
    check("[D] off-valid depth NaN / 0 / 1e9 / random: identical depth term", same)
    n_obs = int((valid & (d > 0)).sum())
    n_used = int((valid & (d > 0) & (v.alpha > cfg.min_alpha_for_depth)).sum())
    check("[D] counts: observed / used / gated-by-alpha",
          (ref.depth_observed, ref.depth_used, ref.depth_gated) == (n_obs, n_used, n_obs - n_used),
          f"{ref.depth_observed}/{ref.depth_used}/{ref.depth_gated}")
    d3 = d.clone()
    i = torch.nonzero(valid & (v.alpha > cfg.min_alpha_for_depth))[0]
    d3[i[0], i[1]] = 0.0
    t3 = camera_terms(v, rgb, d3, valid, cfg, rgb_robust=True)
    check("[D] a zero depth flagged valid is NOT a measurement", t3.depth_observed == n_obs - 1
          and float(t3.p_d[i[0], i[1]]) == 0.0)
    model = MaskedObservationModel("masked-rgb-expected-quad", (0.5, 0.5, 0.5), "uniform01", cfg.rgb_sigma)
    b = model.background_tensor("cpu")
    p0 = prepare(rgb, d, valid, torch.zeros_like(mask))
    check("[D] mask = 0: target is the background, no depth",
          torch.allclose(p0.target(b), b[:, None, None].expand_as(rgb)) and not bool(p0.depth_valid.any()))
    t0 = phase1_camera_terms([v], [p0], cfg, model)[0]
    check("[D] mask = 0: depth precision exactly zero", bool((t0.p_d == 0).all()))
    p1 = prepare(rgb, d, valid, torch.ones_like(mask))
    check("[D] mask = 1: target is the original image, depth = V_orig",
          torch.equal(p1.target(b), rgb) and torch.equal(p1.depth_valid, valid & (d > 0)))
    pm = prepare(rgb, d, valid, mask)
    d4 = d.clone()
    d4[~mask] = torch.rand(int((~mask).sum())) * 1e3 - 500
    rgb4 = rgb.clone()
    rgb4[:, ~mask] = torch.rand(3, int((~mask).sum()))
    pm4 = prepare(rgb4, d4, valid, mask)
    ta, tb = (phase1_camera_terms([v], [p], cfg, model)[0] for p in (pm, pm4))
    check("[D] adversarial off-mask RGB and depth: identical masked objective and precisions",
          ta.energy == tb.energy and torch.equal(ta.p_d, tb.p_d) and torch.equal(ta.p_rgb, tb.p_rgb))


def test_E_legacy_parity():
    print("\n[E] original mode vs pre-patch precisions")
    from estimator.sparse_iekf import IEKFConfig, _precisions, _precisions_legacy
    for robust in (True, False):
        cfg = IEKFConfig(robust_measurement=robust)
        views, gts = [], []
        for s in range(3):
            v, rgb, d, valid, _m = _toy_camera(seed=s)
            views.append(v)
            gts.append((rgb, d, valid))
        new, old = _precisions(views, gts, cfg), _precisions_legacy(views, gts, cfg)
        ok = all(torch.allclose(a[0], b[0], rtol=1e-6, atol=0) and torch.allclose(a[1], b[1], rtol=1e-6, atol=0)
                 and torch.equal(a[2], b[2]) and torch.allclose(a[1] * a[3], b[1] * b[3], rtol=1e-6, atol=1e-12)
                 for a, b in zip(new, old))
        check(f"[E] robust={robust}: p_rgb, p_d, r_rgb and p_d r_d equal the legacy builder", ok)


def test_F_kappa():
    print("\n[F] kappa and fixed W")
    from estimator.masked_observation import MaskedObservationModel, phase1_camera_terms, prepare
    from estimator.sparse_iekf import IEKFConfig
    cfg = IEKFConfig()
    model = MaskedObservationModel("masked-rgb-expected-quad", (0.5, 0.5, 0.5), "uniform01", cfg.rgb_sigma)
    W = 1.0 + torch.rand(3, 5, 6)
    check("[F] kappa = trace(W)/12 (Uniform[0,1] background)", torch.allclose(model.kappa(W), W.sum(0) / 12))
    v, rgb, d, valid, mask = _toy_camera()
    p = prepare(rgb, d, valid, mask)
    v2 = view(v.rgb.flip(-1), v.depth + 0.01, v.alpha.flip(-1))
    ta, tb = (phase1_camera_terms([x], [p], cfg, model)[0] for x in (v, v2))
    check("[F] W is fixed: the RGB precision does not depend on the state", torch.equal(ta.p_rgb, tb.p_rgb))
    check("[F] W = 1/sigma^2 and kappa = 25 at sigma = 0.1",
          abs(float(ta.p_rgb.mean()) - 1 / cfg.rgb_sigma ** 2) < 1e-3
          and (cfg.rgb_sigma != 0.1 or abs(float(ta.kappa.mean()) - 25.0) < 1e-3))


# ---------------------------------------------------------------------------
def test_G_convergence():
    print("\n[G] convergence decision")
    from estimator.sparse_iekf import IEKFConfig, convergence_decision
    cfg = IEKFConfig()
    tol, stol = cfg.rel_energy_tol, cfg.step_norm_tol
    s1, i1 = convergence_decision(100.0, 99.0, 0.1 * stol, cfg)
    check("[G] a real decrease does not stop", not s1 and abs(i1["rel_decrease"] - 0.01) < 1e-12)
    s2, _ = convergence_decision(100.0, 100.0 * (1 - 0.1 * tol), 0.1 * stol, cfg)
    check("[G] tiny decrease and tiny step stop", s2)
    s3, _ = convergence_decision(100.0, 100.0 * (1 - 0.1 * tol), 10 * stol, cfg)
    check("[G] tiny decrease with a large step does not stop", not s3)
    s4, i4 = convergence_decision(1e-3, 0.5e-3, 0.1 * stol, cfg)
    check("[G] the scale is max(|E|, 1)", abs(i4["rel_decrease"] - 0.5e-3) < 1e-15 and s4 == (0.5e-3 < tol))
    if not CUDA:
        not_run("[G] logged stop reason", "CUDA unavailable")
        return
    from tests.test_iekf_end_to_end import camera, identity_cov, make_state, render_gt
    from estimator.sparse_iekf import iekf_frame_step
    from estimator.state import boxplus
    cams = [camera()]
    truth = make_state(n=4, seed=9)
    delta = torch.zeros((4, 12), device="cuda")
    delta[:, 0] = 0.05
    c = IEKFConfig(max_iters=6, pcg_max_iters=20, robust_measurement=False)
    _o, _P, diag = iekf_frame_step(boxplus(truth, delta), identity_cov(4), render_gt(truth, cams, c), cams,
                                   dt=1 / 30.0, cfg=c)
    stops = [r for r in diag.rows if r.get("event") == "stop"]
    acc = [r for r in diag.rows if r.get("event") == "accepted"]
    check("[G] one stop row with a declared reason",
          len(stops) == 1 and stops[0].get("reason") in ("converged", "trust_radius_floor", "lm_max",
                                                         "iteration_budget"), str(stops))
    check("[G] the filter did not stop after the first large accepted decrease", len(acc) >= 2,
          f"{len(acc)} accepted")


# ---------------------------------------------------------------------------
def test_H_phase1_recursion():
    print("\n[H] Phase 1 recursive covariance")
    if not CUDA:
        not_run("[H] Phase 1 recursion", "CUDA unavailable")
        return
    import estimator
    from tests.test_iekf_end_to_end import camera, make_state
    from estimator.sparse_iekf import IEKFConfig
    from tools.kpart_pipeline import bootstrap_positions
    truth = make_state(n=5, seed=3)
    H = W = 8
    obs = [(torch.rand(3, H, W, device="cuda"), torch.ones(H, W, device="cuda"),
            torch.ones(H, W, dtype=torch.bool, device="cuda"))]
    frames = [(obs, {"original_frame_index": k}) for k in range(4)]
    calls = []
    real = estimator.iekf_frame_step

    def fake(state, P, *a, **k):
        P_new = 0.9 * P + 1e-4 * torch.eye(12, device=P.device, dtype=P.dtype)
        calls.append((P, P_new))
        return state, P_new, types.SimpleNamespace(rows=[{"event": "stop", "reason": "converged"}])
    try:
        estimator.iekf_frame_step = fake
        for reset in (False, True):
            calls.clear()
            args = argparse.Namespace(supernode_size=8, seed=0, bootstrap_iters=2, dt=1 / 30.0,
                                      visibility_constraint=None, masked_observation=None,
                                      phase1_covariance_reset=reset)
            _pos, info = bootstrap_positions(truth, [camera()], frames, IEKFConfig(), None, args, verbose=False)
            P0 = calls[0][0]
            if not reset:
                check("[H] frame 2 starts from frame 1's returned posterior (same tensor)",
                      calls[1][0] is calls[0][1] and calls[2][0] is calls[1][1])
                check("[H] the per-frame record names the prior source",
                      info["per_frame"][2]["covariance"]["prior_source"] == "previous frame posterior"
                      and info["per_frame"][2]["covariance"]["psd"])
            else:
                check("[H] --phase1-covariance-reset restores P0 every frame (diagnostic only)",
                      all(torch.equal(c[0], P0) for c in calls))
    finally:
        estimator.iekf_frame_step = real
    # a frame with no information returns exactly the prediction F P F^T + Q
    from estimator.covariance import covariance_predict, make_process_noise
    from estimator.process import error_state_transition
    from estimator.sparse_iekf import iekf_frame_step
    cams = [camera()]
    st = make_state(n=3, seed=5, v=torch.tensor([[0.3, 0.0, 0.1]] * 3), omega=torch.tensor([[0.0, 0.5, 0.2]] * 3))
    g = torch.Generator().manual_seed(2)
    A = 0.05 * torch.randn(3, 12, 12, generator=g, dtype=torch.float64)
    P = (A @ A.transpose(1, 2) + 1e-3 * torch.eye(12, dtype=torch.float64)).to("cuda", st.mu.dtype)
    cfg = IEKFConfig(use_rgb=False, use_depth=False, max_iters=2)
    gt = [(torch.zeros(3, 64, 64, device="cuda"), torch.ones(64, 64, device="cuda"),
           torch.ones(64, 64, dtype=torch.bool, device="cuda"))]
    dt = 1 / 30.0
    try:
        _s, P_post, _d = iekf_frame_step(st, P, gt, cams, dt=dt, cfg=cfg)
        F = error_state_transition(st, dt, mode="so3")
        Q = make_process_noise(3, dt, cfg.process_noise_pos, cfg.process_noise_rot, cfg.process_noise_vel,
                               cfg.process_noise_omega, device="cuda", dtype=st.mu.dtype)
        Pp = covariance_predict(P, F, Q).double()
        rel = float((P_post.double() - Pp).norm() / Pp.norm())
        check("[H] zero-information frame: P_post = F P F^T + Q", rel < 1e-4, f"rel {rel:.2e}")
        ev = torch.linalg.eigvalsh(0.5 * (P_post.double() + P_post.double().transpose(1, 2)).cpu())
        check("[H] P_post is PSD with position-velocity cross terms",
              float(ev.min()) > -1e-10 and float(P_post[:, 0:3, 6:9].abs().max()) > 1e-6)
    except Exception as ex:                                   # noqa: BLE001
        check("[H] zero-information frame runs", False, f"{type(ex).__name__}: {ex}"[:120])


# ---------------------------------------------------------------------------
def test_I_phase2_prior():
    print("\n[I] Phase 2 exact SE(3) prior")
    from estimator.joint_iekf import _FramePrior
    from estimator.joint_process import apply_correction
    from estimator.manifold import so3_exp
    from estimator.rigid_part_runtime import RigidIEKFConfig, RigidState
    from estimator.rigid_part_se3 import RigidPart
    part = RigidPart(torch.arange(5), num_frames=2)
    part.R[1] = so3_exp(torch.tensor([[0.3, -0.4, 0.6]], dtype=torch.float64))[0]
    part.p[1] = torch.tensor([0.05, -0.02, 0.1], dtype=torch.float64)
    part.v[1] = torch.tensor([0.2, 0.1, -0.3], dtype=torch.float64)
    part.omega[1] = torch.tensor([0.4, -0.1, 0.25], dtype=torch.float64)
    xm = RigidState(part, RigidIEKFConfig())
    g = torch.Generator().manual_seed(7)
    Lam = torch.randn(12, 12, generator=g, dtype=torch.float64)
    Lam = Lam @ Lam.T + torch.eye(12, dtype=torch.float64)
    prior = _FramePrior(xm, 1, Lam, 1.0)
    check("[I] exact mode for the rigid state", prior.exact)
    check("[I] e(x_minus) = 0", float(prior.error(xm).abs().max()) < 1e-12,
          f"{float(prior.error(xm).abs().max()):.1e}")
    d = 0.3 * torch.randn(12, generator=g, dtype=torch.float64)
    x = apply_correction(copy.deepcopy(xm), 1, d, transport=False)[0]
    e = prior.error(x)
    check("[I] e(x_minus boxplus d) = d (exact inverse of the retraction, |d| = %.2f)" % float(d.norm()),
          float((e - d).abs().max()) < 1e-9, f"max err {float((e - d).abs().max()):.2e}")
    J = prior.jacobian(x)
    v = torch.randn(12, generator=g, dtype=torch.float64)
    res = []
    for h in (1e-3, 2e-3):
        xh = apply_correction(copy.deepcopy(x), 1, h * v, transport=False)[0]
        res.append(float((prior.error(xh) - e - h * (J @ v)).norm()))
    check("[I] FD Jacobian is first-order exact (remainder O(h^2))", 2.5 < res[1] / max(res[0], 1e-300) < 6.0,
          f"ratio {res[1] / max(res[0], 1e-300):.2f}")
    check("[I] J is not the identity away from x_minus (chart transport handled)",
          float((J - torch.eye(12, dtype=torch.float64)).abs().max()) > 1e-3)
    check("[I] E_prior = w/2 d^T Lam d", abs(prior.energy(x) - 0.5 * float(d @ Lam @ d)) < 1e-8)
    # chart mode (states without prior_error): after a basis reset e_new = G e_old, the
    # prior energy must be unchanged -> Lam_new = G^-T Lam G^-1 (the pre-repair
    # G^-1 Lam G^-T is kept only for the legacy prismatic-null diagnostic)
    chart = types.SimpleNamespace(dim=12)
    G = torch.eye(12, dtype=torch.float64) + 0.3 * torch.randn(12, 12, generator=g, dtype=torch.float64)
    st = 0.1 * torch.randn(12, generator=g, dtype=torch.float64)
    E = {}
    for legacy in (False, True):
        pc = _FramePrior(chart, 1, Lam, 1.0)
        e_old = pc.error(chart, st)
        E0 = pc.energy(chart, st)
        pc.accepted(st, G, legacy_transport=legacy)
        E[legacy] = (E0, pc.energy(chart), float((pc.delta - G @ e_old).abs().max()))
    check("[I] chart transport conserves the prior energy (G^-T Lam G^-1)",
          abs(E[False][0] - E[False][1]) < 1e-9 * E[False][0] and E[False][2] < 1e-12,
          f"{E[False][0]:.6e} -> {E[False][1]:.6e}")
    check("[I] the pre-repair transport does not (why it is legacy-only)",
          abs(E[True][0] - E[True][1]) > 1e-3 * E[True][0], f"{E[True][0]:.6e} -> {E[True][1]:.6e}")
    from estimator.joint_iekf import JointIEKFConfig
    check("[I] Phase 2/3 default acceptance is the MAP", JointIEKFConfig().acceptance == "map")


def test_J_phase3_strong_prior():
    print("\n[J] Phase 3 strong prior (real CUDA renderer)")
    if not CUDA:
        not_run("[J] Phase 3 strong prior", "CUDA unavailable")
        return
    from estimator import phase2_prismatic as mp
    from estimator.joint_displacement import DisplacementOnlyState, geometry_snapshot, geometry_unchanged
    from estimator.joint_iekf import JointIEKFConfig, _energy, frame_step
    from estimator.phase2_model_comparison import prismatic_state
    from estimator.phase2_shared import measurement_terms, render_state
    from estimator.renderer_bridge import build_linearization
    from estimator.sparse_iekf import IEKFConfig
    from tests.test_unified_cuda_renderer_jvp_vjp import make_cameras
    from tests.test_visibility_constrained_iekf_phase1 import observe
    from tests.test_visibility_constrained_iekf_phase23 import scene
    cams = make_cameras(n_cam=3, H=96, W=96, radius=2.2)
    st, members = scene()
    lid = members.cuda()
    sel = torch.isin(torch.arange(st.mu.shape[0], device="cuda"), lid)[:, None]
    truth = dataclasses.replace(st, mu=torch.where(sel, st.mu + torch.tensor([0.015, 0.0, 0.0], device="cuda"),
                                                   st.mu))
    tex = [torch.rand(3, 96, 96, device="cuda", generator=torch.Generator(device="cuda").manual_seed(4))
           for _ in cams]
    obs1, _m = observe(truth, cams, texture=tex)
    cfg = IEKFConfig()
    mu0, R0 = st.mu.double(), st.R.double()
    cam_ids = list(range(len(cams)))

    def lin(s_, t):
        mu, Rg = s_.gaussians(t, mu0, R0)
        return render_state(st, mu, Rg, cams, cfg, build_linearization)

    def prec(L):
        return measurement_terms(L, obs1, cam_ids, cfg)
    e = torch.tensor([1.0, 0.0, 0.0], dtype=torch.float64)
    ds = DisplacementOnlyState(prismatic_state(mp.PrismaticPart(members, e, torch.zeros(2, dtype=torch.float64))))
    snap = geometry_snapshot(ds.parts[0])
    jcfg = JointIEKFConfig(max_iters=10)
    # the image optimum (a negligible prior), then a prior strong enough to hold
    # the coordinate well short of it: its curvature ~ the image curvature
    weak = copy.deepcopy(ds)
    frame_step(weak, 1, mu0, R0, obs1, lin, prec,
               torch.diag(torch.tensor([1e-6, 1e-6], dtype=torch.float64)), jcfg)
    q_img = float(weak.parts[0].s[1])
    m_img = _energy(lin, weak, 1, mu0, R0, obs1, prec, None)[0]
    m_0 = _energy(lin, ds, 1, mu0, R0, obs1, prec, None)[0]
    curv = 2.0 * (m_0 - m_img) / max(q_img ** 2, 1e-12)
    sq, sqd = (1.0 / curv) ** 0.5, 1e-2
    Lam = torch.diag(torch.tensor([1 / sq ** 2, 1 / sqd ** 2], dtype=torch.float64))
    x_minus = copy.deepcopy(ds)
    rec = frame_step(ds, 1, mu0, R0, obs1, lin, prec, Lam, jcfg)
    hist = rec.observability.get("iterations") or []
    acc = [h for h in hist if h.get("accepted")]
    tot = [rec.energy_before] + [h["total_energy"] for h in acc]
    check("[J] the total MAP objective decreases monotonically over accepted steps",
          len(acc) > 0 and all(b < a for a, b in zip(tot, tot[1:])), f"{len(acc)} accepted")
    check("[J] every logged total = rgb + alpha + depth + prior",
          all(abs(h["total_energy"] - (h["measurement_rgb"] + h["measurement_alpha"] + h["measurement_depth"]
                                       + h["prior_energy"])) <= 1e-9 * max(1.0, abs(h["total_energy"]))
              for h in hist if "total_energy" in h))
    check("[J] the joint geometry is bitwise unchanged", geometry_unchanged(snap, ds.parts[0]))
    prior_e = lambda s: 0.5 * float((s.prior_error(1, x_minus) @ Lam @ s.prior_error(1, x_minus)))  # noqa: E731

    def E_tot(q):
        s = copy.deepcopy(ds)
        s.parts[0].s[1] = q
        m, _L = _energy(lin, s, 1, mu0, R0, obs1, prec, None)
        return m + prior_e(s), m
    q_f = float(ds.parts[0].s[1])
    h = 2e-4
    g_f = (E_tot(q_f + h)[0] - E_tot(q_f - h)[0]) / (2 * h)
    g_0 = (E_tot(h)[0] - E_tot(-h)[0]) / (2 * h)
    check("[J] the final state is stationary for the TOTAL objective", abs(g_f) < 0.05 * abs(g_0),
          f"dE/dq final {g_f:.3e} vs start {g_0:.3e}, q = {q_f:.4f} m")
    tc, mc = E_tot(q_img)
    tf, mf = E_tot(q_f)
    check("[J] the strong prior holds q between the prior mean and the image optimum",
          0.0 < q_f < q_img, f"q {q_f:.4f}, image optimum {q_img:.4f} m, prior sd {sq:.2e}")
    check("[J] the image optimum improves the image but worsens the total -> not the answer",
          mc < mf and tc > tf, f"meas {mc:.6e} < {mf:.6e}, total {tc:.6e} > {tf:.6e}")
    P = rec.posterior.double()
    check("[J] qdot has no image information: posterior var = prior var",
          abs(float(P[1, 1]) - sqd ** 2) < 1e-6 * sqd ** 2, f"{float(P[1, 1]):.4e} vs {sqd ** 2:.4e}")
    check("[J] the image adds information on q", float(P[0, 0]) < sq ** 2)


# ---------------------------------------------------------------------------
def test_K_signatures():
    print("\n[K] experiment signatures")
    import stage_signature as ss
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        src, model = td / "src", td / "model"
        write_png(src / "rgb" / "cam_000" / "frame_000000.png", np.zeros((4, 4, 3), np.uint8))
        write_png(src / "mask" / "cam_000" / "frame_000000.png", np.zeros((4, 4), np.uint8))
        (src / "depth_npy" / "cam_000").mkdir(parents=True)
        np.save(src / "depth_npy" / "cam_000" / "frame_000000.npy", np.ones((4, 4)))
        ply = model / "point_cloud" / "iteration_30000" / "point_cloud.ply"
        ply.parent.mkdir(parents=True)
        ply.write_bytes(b"ply-v1")
        (model / "cfg_args").write_text("Namespace()")

        def sig(mode="masked-rgb-expected-quad", **kw):
            a = argparse.Namespace(stage="phase3", direction="forward", scene="toy", source=str(src),
                                   model=str(model), frames=1, cameras=1, iterations="30000",
                                   observation_mode=mode, rgb_background_distribution="uniform01",
                                   rgb_background_constant=None, object_mask_constraint="off", mask_options=None,
                                   mask_source=None, upstream=None, config=["iters=8"])
            for k, v in kw.items():
                setattr(a, k, v)
            return ss.compute(a)
        d = td / "out"
        s0 = sig()
        check("[K] nothing there -> 10", ss.check(d, s0)[0] == 10)
        ss.write(d, s0, "running")
        check("[K] our own running attempt -> 11 (redo)", ss.check(d, s0)[0] == 11)
        rc = ss.write(d, s0, "complete")
        check("[K] 'complete' with invalid outputs is recorded as failed", rc == 1 and
              json.loads((d / ss.MANIFEST).read_text())["status"] == "failed")
        (d / "summary.json").write_text(json.dumps({"ok": 1}))
        ss.write(d, s0, "complete")
        check("[K] complete, same signature, valid outputs -> 0 (reuse)", ss.check(d, s0)[0] == 0)
        rc, msg = ss.check(d, sig(mode="original"))
        check("[K] a different observation mode -> 12 (refuse) naming the field",
              rc == 12 and "observation.mode" in msg, msg[:90])
        write_png(src / "mask" / "cam_000" / "frame_000000.png", np.full((4, 4), 255, np.uint8))
        rc, msg = ss.check(d, sig())
        check("[K] changed mask CONTENT -> 12", rc == 12 and "mask_content_sha256" in msg)
        ply.write_bytes(b"ply-v2")
        check("[K] a different static PLY -> 12", "endpoint.ply_sha256" in ss.check(d, sig())[1])
        check("[K] a missing upstream manifest is part of the signature",
              sig(upstream=str(td / "nope"))["upstream_signature_sha256"] == "MISSING_UPSTREAM_MANIFEST")
        leg = td / "legacy"
        leg.mkdir()
        (leg / "summary.json").write_text("{}")
        check("[K] a legacy output without manifest -> 12", ss.check(leg, s0)[0] == 12)


def test_L_model_error_audit():
    print("\n[L] model-error audit, stray opacity")
    from estimator.model_error import estimate_model_error, model_error_audit
    H, W, sigma = 12, 12, 0.1
    M = torch.zeros(H, W, dtype=torch.bool)
    M[3:9, 3:9] = True
    obj = torch.full((3, H, W), 0.8)
    bbar = 0.5
    A0 = torch.where(M, torch.ones(H, W), torch.zeros(H, W))
    leak = torch.zeros(H, W, dtype=torch.bool)
    leak[0:2, 0:4] = True                          # stray canonical opacity on the background
    A0[leak] = 0.4
    C0 = A0 * obj + (1 - A0) * bbar                # the canonical render on bbar
    Y = torch.where(M[None], obj, torch.full_like(obj, bbar))
    lin = types.SimpleNamespace(views=[view(C0, torch.ones(H, W), A0)])
    me = estimate_model_error(lin, [(Y, torch.ones(H, W), M)])
    rep = model_error_audit(lin, me, sigma, masks=[M])
    reg = rep["per_camera"][0]
    check("[L] foreground (clean) keeps the sensor weight", reg["foreground"]["W_over_sensor_mean"] > 0.98)
    check("[L] leaking background is DOWN-weighted by the canonical map (the confound, measured)",
          reg["background_leaking"]["W_over_sensor_mean"] < 0.5 and reg["background_leaking"]["pixels"] == 8,
          f"W/W_sensor {reg['background_leaking']['W_over_sensor_mean']:.3f}")
    check("[L] the audit says the policy is unchanged", "NOT changed" in rep["note"])


def test_M_mask_diagnostics_rotations():
    print("\n[M] mask diagnostics use the tracked orientations")
    from estimator.mask_diagnostics import MaskDiagnostics, camera_diagnostics
    from estimator.sparse_iekf import IEKFConfig
    cfg = IEKFConfig()
    v, rgb, d, valid, mask = _toy_camera()
    r = camera_diagnostics(v, mask, (rgb, d, valid), cfg)
    check("[M] alpha outside the mask = sum of A on M = 0",
          abs(r["alpha_outside_sum"] - float(v.alpha[~mask].double().sum())) < 1e-9)
    n_orig = int((valid & (d > 0)).sum())
    check("[M] depth discarded by the mask", abs(r["depth_discarded_by_mask"]
                                                 - (1 - int((valid & (d > 0) & mask).sum()) / n_orig)) < 1e-12)
    if not CUDA:
        not_run("[M] rotations reach the renderer", "CUDA unavailable")
        return
    from estimator.manifold import so3_exp
    from estimator.renderer_bridge import build_linearization
    from tests.test_unified_cuda_renderer_jvp_vjp import make_cameras
    from tests.test_visibility_constrained_iekf_phase1 import blob
    cams = make_cameras(n_cam=1, H=64, W=64, radius=2.2)
    st = blob(n=1, seed=0, centre=(0.0, 0.0, 0.0))
    st = dataclasses.replace(st, scales=torch.tensor([[0.25, 0.01, 0.01]], device="cuda", dtype=st.scales.dtype),
                             opacity=torch.ones_like(st.opacity) * 0.9)
    Rz = so3_exp(torch.tensor([[0.0, 0.0, 1.2]], dtype=torch.float64))[0].to("cuda", st.R.dtype)
    m0 = torch.zeros(64, 64, dtype=torch.bool)
    obs = [(torch.zeros(3, 64, 64, device="cuda"), torch.ones(64, 64, device="cuda"),
            torch.ones(64, 64, dtype=torch.bool, device="cuda"))]
    md = MaskDiagnostics("t", lambda k: [m0], "x", cams, cfg, st, build_linearization)
    a = md.frame(0, 0, st.mu, st.R, obs)
    b = md.frame(1, 1, st.mu, Rz @ st.R, obs)
    rel = abs(a["per_camera"][0]["leaking_fraction"] - b["per_camera"][0]["leaking_fraction"])
    check("[M] same centres, different orientation -> different leakage (rotation is rendered)", rel > 1e-3,
          f"leaking share {a['per_camera'][0]['leaking_fraction']:.4f} vs {b['per_camera'][0]['leaking_fraction']:.4f}")
    check("[M] report is OK and flagged as not used in inference",
          md.report()["status"] == "OK" and md.report()["used_in_inference"] is False)


def main() -> int:
    for t in (test_A_default_and_preflight, test_B_driver_dry_runs, test_C_student_t,
              test_D_depth_validity_and_masks, test_E_legacy_parity, test_F_kappa, test_G_convergence,
              test_H_phase1_recursion, test_I_phase2_prior, test_J_phase3_strong_prior, test_K_signatures,
              test_L_model_error_audit, test_M_mask_diagnostics_rotations):
        try:
            t()
        except Exception as ex:                       # noqa: BLE001
            import traceback
            traceback.print_exc()
            check(f"{t.__name__} raised", False, f"{type(ex).__name__}: {ex}"[:160])
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
