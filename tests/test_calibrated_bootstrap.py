#!/usr/bin/env python
"""Regression tests of RigEuler3DGS_Calibrated_Bootstrap_IEKF_Implementation_Spec.md.

  [A] static calibration: AR(1) oracle per camera (3 noise levels, never pooled),
      holes + outliers, heteroscedastic model selection on held-out frames,
      quantized registered depth, bias non-identifiability, interval selection,
      model id round trip / tamper detection, no physical model for synthetic
  [B] surface bootstrap: point-to-plane variance vs finite differences, three
      differently noisy cameras beat plain averaging, a sharp step edge is not
      averaged across, zero-noise determinism, P0 = prior where F = 0, exact
      source responsibility vs the renderer (CUDA)
  [C] wiring: Phase-1 config copy keeps depth_sigma_a/b and the sensor model,
      P0 from the bootstrap file + declared velocity prior, per-frame dt from
      timestamps, calibrated per-camera RGB/depth precisions (original and masked),
      refusal without camera identity, canonical-endpoint and window checks,
      backward elapsed time positive, orchestrator validation refusals

    python tests/test_calibrated_bootstrap.py
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import math
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
CUDA = torch.cuda.is_available()


def check(label, cond, detail=""):
    global CHECKS
    CHECKS += 1
    print(f"  {'ok  ' if cond else 'FAIL'}  {label}  {detail}")
    if not cond:
        FAILURES.append(label)


def ar1(T, P, sigma, rho, rng):
    e = np.zeros((T, P))
    e[0] = rng.normal(0, sigma, P)
    w = sigma * math.sqrt(1 - rho * rho)
    for t in range(1, T):
        e[t] = rho * e[t - 1] + rng.normal(0, w, P)
    return e


def synth_camera(T, sigma_fn, rho=0.5, P=(60, 80), zr=(0.6, 1.2), seed=0, holes=0.0, outliers=0.0,
                 bias_sd=0.0, quant=0.0):
    rng = np.random.default_rng(seed)
    H, W = P
    z = np.linspace(zr[0], zr[1], W)[None].repeat(H, 0)
    bias = rng.normal(0, bias_sd, (H, W)) if bias_sd else 0.0
    e = ar1(T, H * W, 1.0, rho, rng).reshape(T, H, W) * sigma_fn(z)[None]
    D = z[None] + bias + e
    if quant:
        D = np.round(D / quant) * quant
    V = rng.random((T, H, W)) >= holes
    if outliers:
        o = rng.random((T, H, W)) < outliers
        D = np.where(o, D + 0.2, D)
    rgb = 0.5 + 0.01 * rng.normal(size=(T, 3, H, W))
    fg = np.ones((T, H, W), bool)
    return D, V, rgb, fg


# ---------------------------------------------------------------------------
def test_A():
    print("\n[A] static RGB-D calibration")
    from calibrate_static_rgbd import calibrate_camera
    from utils.noise.sensor_model import SceneNoiseModel, declared_model
    cams = {}
    for name, sd in (("cam_a", 1e-3), ("cam_b", 2e-3), ("cam_c", 4e-3)):
        D, V, rgb, fg = synth_camera(60, lambda z, s=sd: np.full_like(z, s), rho=0.5, seed=hash(name) % 100)
        cam, diag, _ = calibrate_camera(name, D, V, rgb, fg, min_count=100)
        cams[name] = cam
        s = math.sqrt(float(cam.depth_variance(np.array([0.9]))[0]))
        check(f"[A] {name}: sigma_D recovered (true {1000 * sd:.0f} mm)", abs(s / sd - 1) < 0.12, f"{1000 * s:.3f} mm")
        check(f"[A] {name}: lag-1 autocorrelation (true 0.5)", abs(cam.lag1_autocorrelation - 0.5) < 0.1,
              f"{cam.lag1_autocorrelation:.3f}")
        ve = diag["variance_estimators_pooled"]
        check(f"[A] {name}: naive Var(dD)/2 under-estimates AR(1) noise, the corrected form does not",
              ve["naive_var_dD_over_2"] < 0.8 * sd * sd and abs(ve["corrected_var_dD_over_2(1-rho1)"] / (sd * sd) - 1) < 0.25,
              f"naive {1000 * ve['naive_var_dD_over_2'] ** 0.5:.2f} mm, corrected "
              f"{1000 * ve['corrected_var_dD_over_2(1-rho1)'] ** 0.5:.2f} mm")
    check("[A] three cameras keep three distinct models (never pooled)",
          len({round(c.depth_coeffs[0], 9) for c in cams.values()}) == 3)
    D, V, rgb, fg = synth_camera(60, lambda z: np.full_like(z, 2e-3), seed=5, holes=0.1, outliers=0.01)
    cam, _, _ = calibrate_camera("holes", D, V, rgb, fg, min_count=100)
    s = math.sqrt(float(cam.depth_variance(np.array([0.9]))[0]))
    check("[A] 10 % holes + 1 % 0.2 m outliers: sigma within 15 %", abs(s / 2e-3 - 1) < 0.15, f"{1000 * s:.3f} mm")
    check("[A] invalid fraction measured", abs(cam.invalid_fraction - 0.1) < 0.02, f"{cam.invalid_fraction:.3f}")
    D, V, rgb, fg = synth_camera(60, lambda z: np.sqrt(1e-7 + 4e-6 * z ** 2), zr=(0.5, 2.0), rho=0.0, seed=7)
    cam, diag, _ = calibrate_camera("hetero", D, V, rgb, fg, min_count=100)
    check("[A] heteroscedastic noise: a + b z^2 selected on held-out frames",
          cam.depth_model in ("a+bz2", "a+bz2+dz4"), f"{cam.depth_model} {cam.depth_coeffs}")
    s1 = math.sqrt(float(cam.depth_variance(np.array([1.8]))[0]))
    check("[A] heteroscedastic model reproduces sigma(1.8 m)", abs(s1 / math.sqrt(1e-7 + 4e-6 * 1.8 ** 2) - 1) < 0.15,
          f"{1000 * s1:.3f} mm")
    D, V, rgb, fg = synth_camera(60, lambda z: np.full_like(z, 7e-4), seed=9, quant=1e-3)
    cam, _, _ = calibrate_camera("quant", D, V, rgb, fg, min_count=100)
    s = math.sqrt(float(cam.depth_variance(np.array([0.9]))[0]))
    check("[A] quantized depth: step measured, sigma includes the quantization (Sheppard)",
          abs(cam.depth_quantization_m - 1e-3) < 1e-6 and 0.7e-3 < s < 1.0e-3,
          f"q {1000 * cam.depth_quantization_m:.3f} mm, sigma {1000 * s:.3f} mm")
    D0, V0, rgb, fg = synth_camera(60, lambda z: np.full_like(z, 1e-3), seed=11)
    Db, _, _, _ = synth_camera(60, lambda z: np.full_like(z, 1e-3), seed=11, bias_sd=5e-3)
    c0, _, _ = calibrate_camera("nb", D0, V0, rgb, fg, min_count=100)
    cb, _, _ = calibrate_camera("b", Db, V0, rgb, fg, min_count=100)
    check("[A] a persistent 5 mm per-pixel bias is invisible to the static estimate (non-identifiable, reported)",
          abs(cb.depth_coeffs[0] / c0.depth_coeffs[0] - 1) < 0.02
          and "do NOT identify the persistent" in SceneNoiseModel(policy="calibrated", cameras={"x": c0}).identifiability)
    m = SceneNoiseModel(policy="calibrated", cameras=cams)
    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "s.json"
        mid = m.to_json(p)
        m2 = SceneNoiseModel.from_json(p)
        check("[A] sensor model id round trip", m2.model_id == mid)
        j = json.loads(p.read_text())
        j["cameras"]["cam_a"]["depth_coeffs"][0] *= 2
        p.write_text(json.dumps(j))
        try:
            SceneNoiseModel.from_json(p)
            check("[A] an edited sensor model file is refused", False)
        except ValueError:
            check("[A] an edited sensor model file is refused", True)
    check("[A] synthetic (ideal) carries no physical per-camera noise", not declared_model("ideal").cameras
          and not declared_model("ideal").is_physical)
    try:
        SceneNoiseModel(policy="ideal", cameras=cams)
        check("[A] an ideal model with camera noise is refused", False)
    except ValueError:
        check("[A] an ideal model with camera noise is refused", True)
    # interval selection on files
    from calibrate_static_rgbd import load_interval
    from PIL import Image
    with tempfile.TemporaryDirectory() as td:
        src = Path(td)
        for t in range(10):
            for sub, arr in (("rgb", np.full((4, 5, 3), t, np.uint8)), ("mask", np.full((4, 5), 255, np.uint8))):
                (src / sub / "cam_000").mkdir(parents=True, exist_ok=True)
                Image.fromarray(arr).save(src / sub / "cam_000" / f"frame_{t:06d}.png")
            (src / "depth_npy" / "cam_000").mkdir(parents=True, exist_ok=True)
            np.save(src / "depth_npy" / "cam_000" / f"frame_{t:06d}.npy", np.full((4, 5), 1.0 + t, np.float32))
        D, V, R, F, files = load_interval(src, "cam_000", list(range(3, 7)))
        check("[A] interval [3, 6] loads exactly frames 3..6", D.shape[0] == 4 and float(D[0, 0, 0]) == 4.0
              and float(D[-1, 0, 0]) == 7.0)


# ---------------------------------------------------------------------------
def _plane_scene(sig, n_per=1500, seed=0, step=False):
    """Samples of z = 0 (and, with step, z = 0.05 for x > 0) seen from 3 cameras with
    different depth noise along their rays."""
    rng = np.random.default_rng(seed)
    Y, D, V, Ctr = [], [], [], []
    cams = [np.array([0.0, 0.0, 1.0]), np.array([0.6, 0.0, 0.8]), np.array([-0.4, 0.5, 0.9])]
    for c, s in zip(cams, sig):
        xy = rng.uniform(-0.1, 0.1, (n_per, 2))
        p = np.c_[xy, np.where((xy[:, 0] > 0) & step, 0.05, 0.0)]
        ray = p - c
        z = np.linalg.norm(ray, axis=1)
        u = ray / z[:, None]
        noise = rng.normal(0, s, n_per)
        Y.append(p + u * noise[:, None])
        D.append(u)                 # unit-depth ray (|d| = 1 here: z = range)
        V.append(np.full(n_per, s * s))
        Ctr.append(np.full(n_per, len(Ctr)))
    return np.concatenate(Y), np.concatenate(D), np.concatenate(V), np.concatenate(Ctr)


def test_B():
    print("\n[B] surface-centred bootstrap")
    from estimator.surface_bootstrap import fit_surface, geometry_covariance, gaussian_parameters
    dev = "cuda" if CUDA else "cpu"
    # finite-difference check of n^T C n = sigma^2 (n . d)^2
    rng = np.random.default_rng(1)
    n = rng.normal(size=3); n /= np.linalg.norm(n)
    d = rng.normal(size=3); s = 0.003
    C = s * s * np.outer(d, d)
    zs = rng.normal(0, s, 200000)
    emp = np.var((zs[:, None] * d[None]) @ n)
    check("[B] point-to-plane variance n^T C n = sigma^2 (n.d)^2 (Monte Carlo)", abs(emp / float(n @ C @ n) - 1) < 0.02,
          f"{emp:.3e} vs {float(n @ C @ n):.3e}")
    Y, D, V, cam = _plane_scene((5e-4, 3e-3, 8e-3))
    rgb = np.full((len(Y), 3), 0.5)
    seeds = np.c_[np.random.default_rng(2).uniform(-0.08, 0.08, (200, 2)), np.zeros(200)]
    fit = fit_surface(Y, D, V, rgb, seeds, k=60, device=dev)
    from scipy.spatial import cKDTree
    nn = cKDTree(Y).query(seeds, k=60)[1]
    plain = Y[nn].mean(1)
    e_fit = float(np.sqrt(np.mean(fit["mu"][:, 2] ** 2)))
    e_plain = float(np.sqrt(np.mean(plain[:, 2] ** 2)))
    check("[B] 3 differently noisy cameras: weighted surface fit beats plain averaging",
          e_fit < 0.6 * e_plain, f"normal RMS {1000 * e_fit:.3f} vs {1000 * e_plain:.3f} mm")
    check("[B] normals recovered", float(np.median(np.abs(fit["normal"][:, 2]))) > 0.99)
    Y2, D2, V2, _ = _plane_scene((5e-4, 5e-4, 5e-4), step=True, seed=4)
    seeds2 = np.c_[np.full(50, -0.006), np.linspace(-0.08, 0.08, 50), np.zeros(50)]       # 6 mm from the step
    fit2 = fit_surface(Y2, D2, V2, np.full((len(Y2), 3), 0.5), seeds2, k=40, device=dev)
    check("[B] a sharp 5 cm step is not averaged across (front/back layer kept apart)",
          float(np.median(np.abs(fit2["mu"][:, 2]))) < 2e-3, f"median |z| {1000 * float(np.median(np.abs(fit2['mu'][:, 2]))):.2f} mm")
    Y3, D3, V3, _ = _plane_scene((1e-9, 1e-9, 1e-9), seed=6)
    fa = fit_surface(Y3, D3, V3, np.full((len(Y3), 3), 0.5), seeds, k=30, device=dev)
    fb = fit_surface(Y3, D3, V3, np.full((len(Y3), 3), 0.5), seeds, k=30, device=dev)
    check("[B] zero-noise synthetic: exact plane and deterministic",
          float(np.abs(fa["mu"][:, 2]).max()) < 1e-6 and np.array_equal(fa["mu"], fb["mu"]))
    gp = gaussian_parameters(fit)
    R = gp["R"]
    check("[B] Gaussian frames are proper rotations, normal axis thinnest",
          np.allclose(np.linalg.det(R), 1, atol=1e-9) and bool((gp["scales"][:, 2] <= gp["scales"][:, :2].min(1) + 1e-12).all()))
    cov = geometry_covariance(torch.zeros(4, 6, 6), 0.05, 0.5)
    check("[B] F = 0 (unobserved): P0 equals the declared prior, no fake certainty",
          np.allclose(cov["P0_geometry"][0], np.diag([0.05 ** 2] * 3 + [0.25] * 3)))
    if not CUDA:
        NOT_RUN.append("[B] responsibility vs renderer")
        return
    from estimator.renderer_bridge import build_linearization
    from estimator.surface_bootstrap import source_responsibility, ownership
    from tests.test_unified_cuda_renderer_jvp_vjp import make_cameras
    from tests.test_visibility_constrained_iekf_phase23 import scene
    st, _ = scene()
    cams = make_cameras(n_cam=3, H=64, W=64, radius=2.2)
    lin = build_linearization(st, cams, eps_depth=1e-6)
    resp = source_responsibility(lin)
    own = ownership(resp, st.mu.shape[0], 3)
    check("[B] responsibility sum_i T*alpha equals the rendered alpha (every camera)",
          max(resp["max_abs_alpha_sum_error_per_camera"]) < 1e-5, str(resp["max_abs_alpha_sum_error_per_camera"]))
    check("[B] ownership shares are a partition per seen Gaussian",
          np.allclose(own["camera_share"][own["total_weight"] > 0].sum(1), 1.0))


# ---------------------------------------------------------------------------
def test_C():
    print("\n[C] sensor model and time through Phase 1/2/3")
    from estimator.observation_terms import camera_terms
    from estimator.sparse_iekf import IEKFConfig
    from utils.noise.sensor_model import CameraNoise, SceneNoiseModel
    cam_n = {"cam_000": CameraNoise("cam_000", depth_coeffs=(1e-6, 4e-6, 0.0), rgb_covariance=((1e-4, 0, 0), (0, 4e-4, 0), (0, 0, 9e-4))),
             "cam_001": CameraNoise("cam_001", depth_coeffs=(9e-6, 0.0, 0.0), rgb_covariance=((4e-4, 0, 0), (0, 4e-4, 0), (0, 0, 4e-4)))}
    sm = SceneNoiseModel(policy="calibrated", cameras=cam_n)
    H, W = 6, 7
    z = torch.full((H, W), 1.5, dtype=torch.float64)
    view = types.SimpleNamespace(rgb=torch.zeros(3, H, W, dtype=torch.float64), depth=z.clone(),
                                 alpha=torch.ones(H, W, dtype=torch.float64))
    obs_rgb = torch.full((3, H, W), 0.01, dtype=torch.float64)
    valid = torch.ones(H, W, dtype=torch.bool)
    cfg = dataclasses.replace(IEKFConfig(robust_measurement=False), sensor_model=sm, camera_names=("cam_000", "cam_001"))
    t0 = camera_terms(view, obs_rgb, z + 0.001, valid, cfg, rgb_robust=False, camera="cam_000")
    check("[C] calibrated RGB precision is the camera's own per channel",
          torch.allclose(t0.p_rgb[:, 0, 0], torch.tensor([1e4, 2500.0, 1111.111111], dtype=torch.float64), rtol=1e-6))
    check("[C] calibrated depth precision 1/(a + b z^2) at the observed depth",
          abs(float(t0.p_d[0, 0]) - 1 / (1e-6 + 4e-6 * 1.501 ** 2)) < 1e-6 * float(t0.p_d[0, 0]))
    t1 = camera_terms(view, obs_rgb, z + 0.001, valid, cfg, rgb_robust=False, camera="cam_001")
    check("[C] the other camera gets ITS model", abs(float(t1.p_d[0, 0]) - 1 / 9e-6) < 1e-3)
    try:
        camera_terms(view, obs_rgb, z, valid, cfg, rgb_robust=False, camera=None)
        check("[C] calibrated model without camera identity is refused", False)
    except ValueError:
        check("[C] calibrated model without camera identity is refused", True)
    cfg_m = dataclasses.replace(cfg, rgb_model_sigma=0.1)
    tm = camera_terms(view, obs_rgb, z, valid, cfg_m, rgb_robust=False, camera="cam_000")
    check("[C] declared model-mismatch sigma adds in variance", abs(float(tm.p_rgb[0, 0, 0]) - 1 / (1e-4 + 0.01)) < 1e-6)
    base = IEKFConfig(robust_measurement=False)
    tl = camera_terms(view, obs_rgb, z, valid, base, rgb_robust=False, camera="cam_000")
    check("[C] no sensor model (ideal/assumed): the configured sigmas, unchanged",
          abs(float(tl.p_rgb[0, 0, 0]) - 1 / base.rgb_sigma ** 2) < 1e-9)
    # masked mode uses the calibrated W and kappa = trace(W)/12
    from estimator.masked_observation import MaskedObservationModel, phase1_camera_terms, prepare
    mmod = MaskedObservationModel("masked-rgb-expected-quad", (0.5, 0.5, 0.5), "uniform01", base.rgb_sigma)
    pr = prepare(obs_rgb.float(), (z + 0.001).float(), valid, torch.ones(H, W, dtype=torch.bool))
    vf = types.SimpleNamespace(rgb=view.rgb.float(), depth=view.depth.float(), alpha=view.alpha.float())
    tmk = phase1_camera_terms([vf], [pr], cfg, mmod)[0]
    check("[C] masked mode: calibrated per-camera W, kappa = trace(W)/12",
          abs(float(tmk.p_rgb[1, 0, 0]) - 2500.0) < 1e-2 and abs(float(tmk.kappa[0, 0]) - (1e4 + 2500 + 1111.111) / 12) < 0.1)
    # Phase-1 bootstrap: config copy, P0 file, per-frame dt
    import estimator
    from tools.kpart_pipeline import bootstrap_positions
    if not CUDA:
        NOT_RUN.append("[C] bootstrap_positions wiring")
    else:
        from tests.test_iekf_end_to_end import camera, make_state
        truth = make_state(n=4, seed=1)
        ts = [10.0, 10.04, 10.10, 10.18]
        obs = [(torch.rand(3, 8, 8, device="cuda"), torch.ones(8, 8, device="cuda"), torch.ones(8, 8, dtype=torch.bool, device="cuda"))]
        frames = [(obs, {"original_frame_index": 110 + k, "timestamp_s": ts[k]}) for k in range(4)]
        seen = []
        real = estimator.iekf_frame_step

        def fake(state, P, *a_, dt=None, cfg=None, **k):
            seen.append((P.clone(), dt, cfg))
            return state, P, types.SimpleNamespace(rows=[])
        with tempfile.TemporaryDirectory() as td:
            Pg = np.tile(np.diag([1e-4] * 3 + [4e-4] * 3)[None], (4, 1, 1))
            np.savez(Path(td) / "g.npz", P0_geometry=Pg)
            try:
                estimator.iekf_frame_step = fake
                cfg1 = dataclasses.replace(IEKFConfig(), depth_sigma_a=0.007, depth_sigma_b=0.003, sensor_model=sm,
                                           camera_names=("cam_000",))
                args = argparse.Namespace(supernode_size=8, seed=0, bootstrap_iters=2, dt=1 / 30.0,
                                          visibility_constraint=None, masked_observation=None,
                                          p0_geometry=str(Path(td) / "g.npz"), p0_velocity_sigma=(0.2, 0.1))
                _p, info = bootstrap_positions(truth, [camera()], frames, cfg1, None, args, verbose=False)
            finally:
                estimator.iekf_frame_step = real
        P_first, _dt0, c_first = seen[0]
        check("[C] Phase-1 config keeps depth_sigma_a/b and the sensor model (no subset copy)",
              c_first.depth_sigma_a == 0.007 and c_first.depth_sigma_b == 0.003 and c_first.sensor_model is sm)
        check("[C] P0 pose block from the bootstrap file, velocity block declared",
              torch.allclose(P_first[0, :6, :6].double().cpu(), torch.as_tensor(Pg[0]))
              and abs(float(P_first[0, 6, 6]) - 0.04) < 1e-7 and abs(float(P_first[0, 9, 9]) - 0.01) < 1e-7)
        check("[C] per-frame dt from the capture timestamps", [round(s[1], 6) for s in seen] == [0.04, 0.06, 0.08],
              str([round(s[1], 6) for s in seen]))
        check("[C] P0 provenance recorded", "bootstrap geometry covariance" in info["P0_source"]["pose"])
    from utils.sequence_direction import processing_time_s, reverse_sequence
    tsx = [10.0, 10.04, 10.10, 10.18]
    ivl = [0.04, 0.04, 0.06, 0.08]                        # forward interval to the previous frame
    fr = [(None, None, ivl[k]) for k in range(4)]
    prov = {"frames": [{"frame": k, "dataset_frame_index": 110 + k, "timestamp_s": tsx[k],
                        "dt_used_s": ivl[k]} for k in range(4)]}
    bf, bp = reverse_sequence(fr, prov)
    tb = processing_time_s(bf, bp)
    check("[C] backward elapsed time from timestamps: positive, increasing, nonuniform",
          torch.allclose(tb, torch.tensor([0.0, 0.08, 0.14, 0.18], dtype=torch.float64)), str(tb.tolist()))
    from estimator import run_config as rc
    with tempfile.TemporaryDirectory() as td:
        (Path(td) / "bootstrap_manifest.json").write_text(json.dumps({"endpoint_frame": 110}))
        try:
            rc.check_canonical_endpoint(td, "backward", 110, 200, explicit=True)
            check("[C] a frame-110 model is refused as the backward (frame 200) endpoint", False)
        except SystemExit:
            check("[C] a frame-110 model is refused as the backward (frame 200) endpoint", True)
        ev = rc.check_canonical_endpoint(td, "forward", 110, 200, explicit=True)
        check("[C] and accepted as the forward endpoint", ev["consistent"])
    try:
        rc.window(argparse.Namespace(motion_start_frame=10, motion_end_frame=20), 5)
        check("[C] --frames contradicting the motion interval is refused", False)
    except SystemExit:
        check("[C] --frames contradicting the motion interval is refused", True)
    from run_noise_aware_rigeuler import validate
    syn = Path("/media/samuelemara/Extreme SSD/Samuele/Backup_09_10_26/Sapien/output/preparation_8cams/"
               "partnet_subset_fixed_render_linear/fridge_10905")
    if syn.is_dir():
        def args_(**k):
            d = dict(source=str(syn), dataset_kind="synthetic", rgbd_noise_calibration=False, calibration_start_frame=None,
                     calibration_end_frame=None, motion_start_frame=0, motion_end_frame=59, sensor_noise_model="ideal",
                     frame_timestamps="off", cameras=None)
            d.update(k)
            return argparse.Namespace(**d)
        for label, kw in (("synthetic + calibration flags", dict(rgbd_noise_calibration=True, calibration_start_frame=0,
                                                                 calibration_end_frame=9)),
                          ("synthetic with a calibrated sensor", dict(sensor_noise_model="calibrated")),
                          ("motion end beyond the data", dict(motion_end_frame=600))):
            try:
                validate(args_(**kw))
                check(f"[C] orchestrator refuses {label}", False)
            except SystemExit:
                check(f"[C] orchestrator refuses {label}", True)
        check("[C] orchestrator accepts the synthetic plan", bool(validate(args_())["cameras"]))
    else:
        NOT_RUN.append("[C] orchestrator validation (SSD not mounted)")


def main() -> int:
    for t in (test_A, test_B, test_C):
        try:
            t()
        except Exception as ex:                      # noqa: BLE001
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
