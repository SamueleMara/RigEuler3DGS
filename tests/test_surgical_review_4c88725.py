#!/usr/bin/env python
"""Acceptance tests of RigEuler3DGS_Codex_Surgical_Fixes_4c88725.md sections 1-4.

  [1] bootstrap silhouette: whole-image deterministic disjoint split; negative-ray and
      positive-ray oracles (CPU logic + CUDA renderer); RGB/depth never on background or
      invalid pixels, alpha on both; empty supports finite; held-out bytes do not move a
      training step
  [2] content-safe reuse: same -> reuse; each changed setting / same-size input byte /
      calibration interval -> refuse; corrupted outputs, missing P0 file, swapped checkpoint,
      legacy dir -> refuse; nothing deleted; interrupted same-signature -> redo
  [3] physical-time exports: 25 Hz constant twist, nonuniform intervals, scalar
      compatibility, forward / backward chronological agreement
  [4] audit: linear-Gaussian predict decomposition, velocity-prior pose variance,
      camera-identity refusal, tracking bitwise identical with / without the audit (CUDA)

    python tests/test_surgical_review_4c88725.py
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
def _obs(H=6, W=8):
    M = torch.zeros(H, W)
    M[:, :4] = 1.0
    V = torch.ones(H, W, dtype=torch.bool)
    V[0, 0] = False
    return {"rgb": torch.full((3, H, W), 0.5), "M": M, "V": V, "invz": torch.ones(H, W),
            "var_invz": torch.full((H, W), 1e-4), "w_rgb": torch.full((3, 1, 1), 100.0)}


def test_1():
    print("\n[1] bootstrap silhouette loss")
    from estimator.surface_bootstrap import bootstrap_loss_terms, pixel_partition
    tr, he = pixel_partition(40, 50, 0.1, seed=3, camera_index=1)
    tr2, he2 = pixel_partition(40, 50, 0.1, seed=3, camera_index=1)
    check("[1] split is deterministic, disjoint and covers the whole image",
          torch.equal(torch.as_tensor(tr), torch.as_tensor(tr2)) and not bool((tr & he).any()) and bool((tr | he).all()))
    Mbig = np.zeros((40, 50), bool)
    Mbig[:, :25] = True
    check("[1] both partitions contain background pixels", bool((he & ~Mbig).any()) and bool((tr & ~Mbig).any()))
    obs = _obs()
    S = torch.ones(6, 8, dtype=torch.bool)
    A = torch.where(obs["M"] > 0, torch.ones(6, 8), torch.zeros(6, 8)).clone()
    A[2, 6] = 0.6                                                     # spill onto M = 0
    A.requires_grad_(True)
    rgb = torch.full((3, 6, 8), 0.5)
    inv = A.detach().clone()                                          # inverse depth 1 where alpha 1
    t = bootstrap_loss_terms(rgb, A, inv, obs, S, silhouette_sigma=0.1)
    t["alpha"].backward()
    check("[1] negative-ray oracle: off-mask opacity is penalised, gradient pushes A down",
          float(t["alpha"]) > 0 and float(A.grad[2, 6]) > 0, f"E {float(t['alpha']):.4g}, dE/dA {float(A.grad[2, 6]):.4g}")
    old_W = S & (obs["M"] > 0)                                       # the pre-fix support
    old = 0.5 * (old_W * (A.detach() - obs["M"]) ** 2).sum() / (0.01 * old_W.sum())
    check("[1] ... and the old (mask-only) objective gave it zero", float(old) == 0.0)
    A2 = torch.where(obs["M"] > 0, torch.ones(6, 8), torch.zeros(6, 8)).clone()
    A2[1, 1] = 0.2
    A2.requires_grad_(True)
    bootstrap_loss_terms(rgb, A2, inv, obs, S, silhouette_sigma=0.1)["alpha"].backward()
    check("[1] positive-ray oracle: missing in-mask alpha is pulled up (sign unchanged)", float(A2.grad[1, 1]) < 0)
    rgb_r = torch.rand(3, 6, 8, requires_grad=True)
    inv_r = torch.rand(6, 8, requires_grad=True)
    Aa = torch.ones(6, 8, requires_grad=True)
    tt = bootstrap_loss_terms(rgb_r, Aa, inv_r, obs, S, silhouette_sigma=0.1)
    (tt["rgb"] + tt["depth"]).backward()
    bgm = obs["M"] == 0
    check("[1] RGB and depth never touch background pixels", float(rgb_r.grad[:, bgm].abs().max()) == 0.0
          and float(inv_r.grad[bgm].abs().max()) == 0.0)
    check("[1] depth never touches invalid observations", float(inv_r.grad[0, 0]) == 0.0)
    check("[1] alpha covers foreground and background", tt["stats"]["foreground"] > 0 and tt["stats"]["background"] > 0)
    e = bootstrap_loss_terms(rgb, A.detach(), inv, obs, torch.zeros(6, 8, dtype=torch.bool), silhouette_sigma=0.1)
    check("[1] an empty partition gives finite zero terms with zero counts",
          all(math.isfinite(float(e[k])) and float(e[k]) == 0.0 for k in ("rgb", "alpha", "depth")) and e["stats"]["pixels"] == 0)
    held = torch.zeros(6, 8, dtype=torch.bool)
    held[:, 7] = True
    train = ~held
    obs2 = {k: (v.clone() if torch.is_tensor(v) else v) for k, v in obs.items()}
    obs2["M"][:, 7] = 1.0
    obs2["rgb"][:, :, 7] = 0.0
    obs2["invz"][:, 7] = 9.0
    Ax = A.detach().clone().requires_grad_(True)
    Ay = A.detach().clone().requires_grad_(True)
    la = bootstrap_loss_terms(rgb, Ax, inv, obs, train, silhouette_sigma=0.1)
    lb = bootstrap_loss_terms(rgb, Ay, inv, obs2, train, silhouette_sigma=0.1)
    (la["rgb"] + la["alpha"] + la["depth"]).backward()
    (lb["rgb"] + lb["alpha"] + lb["depth"]).backward()
    check("[1] changing only held-out measurements leaves the training loss and gradient unchanged",
          float(la["alpha"]) == float(lb["alpha"]) and torch.equal(Ax.grad, Ay.grad))
    if not CUDA:
        NOT_RUN.append("[1] CUDA renderer oracle")
        return
    _renderer_oracle()


def _renderer_oracle():
    """Real rasterizer: Gaussians whose footprint lies on M = 0 get a nonzero opacity
    gradient from the new silhouette term and exactly zero from the old one."""
    from estimator.surface_bootstrap import bootstrap_loss_terms
    from gaussian_renderer import render
    from tools.run_iekf_scene import load_scene
    from train_tracking_bootstrap import write_model
    src = Path("/media/samuelemara/Extreme SSD/Samuele/Backup_09_10_26/Sapien/output/preparation_8cams/"
               "partnet_subset_fixed_render_linear/fridge_10905")
    if not (src / "colmap").is_dir():
        NOT_RUN.append("[1] CUDA renderer oracle (dataset not mounted)")
        return
    from plyfile import PlyData
    v = PlyData.read(str(src / "pointcloud" / "points3D_parts.ply"))["vertex"]
    pts = np.stack([v["x"], v["y"], v["z"]], 1)[::50][:300]
    n = len(pts)
    params = {"xyz": pts, "quat_wxyz": np.tile([1.0, 0, 0, 0], (n, 1)), "scales": np.full((n, 3), 0.01),
              "opacity": np.full(n, 0.5), "rgb": np.full((n, 3), 0.5)}
    with tempfile.TemporaryDirectory() as td:
        write_model(Path(td), 0, params, src, "colmap")
        _d, pipe, gs, sc = load_scene(td, str(src / "colmap"), camera_device="cpu", inference_only=True)
        cam = sorted(sc.getTrainCameras() + sc.getTestCameras(), key=lambda c: c.image_name)[0]
        A0 = render(cam, gs, pipe, torch.zeros(3, device="cuda"),
                    override_color=torch.ones((n, 3), device="cuda"))["render"][0].detach()
        H, W = A0.shape
        M = torch.zeros(H, W, device="cuda")
        M[:, : W // 2] = 1.0                                     # half the image is "object"
        obs = {"rgb": torch.full((3, H, W), 0.5, device="cuda"), "M": M, "V": torch.zeros(H, W, dtype=torch.bool, device="cuda"),
               "invz": torch.ones(H, W, device="cuda"), "var_invz": torch.ones(H, W, device="cuda"),
               "w_rgb": torch.ones(3, 1, 1, device="cuda")}
        r = render(cam, gs, pipe, torch.zeros(3, device="cuda"))
        A = render(cam, gs, pipe, torch.zeros(3, device="cuda"), override_color=torch.ones((n, 3), device="cuda"))["render"][0]
        S = torch.ones(H, W, dtype=torch.bool, device="cuda")
        t = bootstrap_loss_terms(r["render"], A, r["depth"].squeeze(0), obs, S, silhouette_sigma=0.1)
        gs._opacity.grad = None
        t["alpha"].backward()
        g_new = gs._opacity.grad.detach().clone().reshape(-1)
        gs._opacity.grad = None
        A = render(cam, gs, pipe, torch.zeros(3, device="cuda"), override_color=torch.ones((n, 3), device="cuda"))["render"][0]
        Wold = M
        old = 0.5 * (Wold * (A - M) ** 2).sum() / (0.01 * Wold.sum())
        old.backward()
        g_old = gs._opacity.grad.detach().clone().reshape(-1)
        # Gaussians whose rendered footprint is entirely on M = 0
        import math as _m
        proj = (torch.cat([gs.get_xyz, torch.ones(n, 1, device="cuda")], 1) @ cam.full_proj_transform)
        x = ((proj[:, 0] / proj[:, 3]) + 1) * 0.5 * W
        off = (x > W // 2 + 20) & (proj[:, 3] > 0)
        check("[1] renderer oracle: off-mask Gaussians get an opacity gradient now",
              bool(off.any()) and float(g_new[off].abs().max()) > 0, f"{int(off.sum())} Gaussians off-mask")
        check("[1] renderer oracle: the old objective gave them exactly zero", float(g_old[off].abs().max()) == 0.0)


# ---------------------------------------------------------------------------
def _fake_dataset(root: Path, frames=range(0, 6)):
    from PIL import Image
    for c in ("cam_000", "cam_001"):
        for t in frames:
            for sub, arr in (("rgb", np.full((4, 5, 3), 10 * t, np.uint8)), ("mask", np.full((4, 5), 255, np.uint8))):
                (root / sub / c).mkdir(parents=True, exist_ok=True)
                Image.fromarray(arr).save(root / sub / c / f"frame_{t:06d}.png")
            (root / "depth_npy" / c).mkdir(parents=True, exist_ok=True)
            np.save(root / "depth_npy" / c / f"frame_{t:06d}.npy", np.full((4, 5), 1.0 + t, np.float32))
    sp = root / "colmap" / "sparse" / "0"
    sp.mkdir(parents=True, exist_ok=True)
    (sp / "cameras.bin").write_bytes(b"cams")
    (sp / "images.bin").write_bytes(b"imgs")


def _fake_bootstrap(d: Path, n: int, it: int, sensor_id: str, frame: int, end: str):
    from plyfile import PlyData, PlyElement
    arr = np.zeros(n, dtype=[("x", "f4"), ("y", "f4"), ("z", "f4")])
    arr["x"] = np.arange(n)
    (d / "point_cloud" / f"iteration_{it}").mkdir(parents=True, exist_ok=True)
    PlyData([PlyElement.describe(arr, "vertex")]).write(str(d / "point_cloud" / f"iteration_{it}" / "point_cloud.ply"))
    PlyData([PlyElement.describe(arr, "vertex")]).write(str(d / "point_cloud.ply"))
    np.savez(d / "geometry_information.npz", P0_geometry=np.tile(np.eye(6)[None], (n, 1, 1)))
    (d / "bootstrap_manifest.json").write_text(json.dumps({"num_gaussians": n, "sensor_model_id": sensor_id,
                                                           "endpoint_frame": frame, "endpoint": end}))


def test_2():
    print("\n[2] content-safe stage reuse")
    import stage_cache as sc
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        src = td / "src"
        _fake_dataset(src)
        cams = ["cam_000", "cam_001"]
        sensor = {"policy": "calibrated", "sensor_model_id": "abc"}
        opt = {"num_gaussians": 7, "knn": 24, "refine_iters": 0, "checkpoints": "0", "heldout_fraction": 0.1,
               "silhouette_sigma": 0.1, "thickness_weight": 0.01, "rgb_model_sigma": 0.0, "depth_model_sigma": 0.0,
               "p0_prior_sigma_pos": 0.05, "p0_prior_sigma_rot": 0.5, "lr_scale": 1.0, "seed": 0}
        d = td / "boot"
        sig = sc.bootstrap_signature(src, cams, "initial", 2, sensor, opt)
        check("[2] empty directory -> run", sc.decide(d, sig)[0] == "run")
        sc.record(d, sig, "running")
        check("[2] own interrupted attempt, same signature -> redo", sc.decide(d, sig)[0] == "redo")
        _fake_bootstrap(d, 7, 0, "abc", 2, "initial")
        sc.record(d, sig, "complete")
        check("[2] same inputs + settings + complete outputs -> reuse", sc.decide(d, sig)[0] == "reuse",
              sc.decide(d, sig)[1][-60:])
        for k, v in (("rgb_model_sigma", 0.05), ("depth_model_sigma", 0.005), ("num_gaussians", 8),
                     ("refine_iters", 10), ("p0_prior_sigma_pos", 0.1), ("silhouette_sigma", 0.2)):
            s2 = sc.bootstrap_signature(src, cams, "initial", 2, sensor, dict(opt, **{k: v}))
            r, msg = sc.decide(d, s2)
            check(f"[2] change only {k} -> refuse", r == "refuse" and k in msg)
        s3 = sc.bootstrap_signature(src, cams, "initial", 2, {"policy": "calibrated", "sensor_model_id": "xyz"}, opt)
        check("[2] a different sensor model id -> refuse", sc.decide(d, s3)[0] == "refuse")
        f = src / "depth_npy" / "cam_001" / "frame_000002.npy"
        b0 = f.read_bytes()
        arr = np.full((4, 5), 3.0, np.float32)
        arr[1, 1] = 3.0001
        np.save(f, arr)
        r, msg = sc.decide(d, sc.bootstrap_signature(src, cams, "initial", 2, sensor, opt))
        check("[2] same-size endpoint depth bytes changed -> refuse", len(f.read_bytes()) == len(b0) and r == "refuse"
              and "inputs_sha256" in msg)
        f.write_bytes(b0)
        check("[2] restored input -> reuse again", sc.decide(d, sig)[0] == "reuse")
        before = sorted(p.relative_to(d).as_posix() for p in d.rglob("*"))
        (d / "geometry_information.npz").rename(d / "geo.bak")
        check("[2] removed geometry_information.npz -> refuse", sc.decide(d, sig)[0] == "refuse")
        (d / "geo.bak").rename(d / "geometry_information.npz")
        m = json.loads((d / "bootstrap_manifest.json").read_text())
        (d / "bootstrap_manifest.json").write_text(json.dumps(dict(m, sensor_model_id="zzz")))
        check("[2] corrupted sensor_model_id in the manifest -> refuse", sc.decide(d, sig)[0] == "refuse")
        (d / "bootstrap_manifest.json").write_text(json.dumps(m))
        import shutil
        shutil.copy(d / "point_cloud.ply", td / "keep.ply")
        from plyfile import PlyData, PlyElement
        arr2 = np.zeros(7, dtype=[("x", "f4"), ("y", "f4"), ("z", "f4")])
        PlyData([PlyElement.describe(arr2, "vertex")]).write(str(d / "point_cloud.ply"))
        check("[2] swapped top-level checkpoint -> refuse", sc.decide(d, sig)[0] == "refuse")
        shutil.copy(td / "keep.ply", d / "point_cloud.ply")
        check("[2] refusals deleted nothing", sorted(p.relative_to(d).as_posix() for p in d.rglob("*")) == before)
        leg = td / "legacy"
        leg.mkdir()
        (leg / "bootstrap_manifest.json").write_text("{}")
        check("[2] legacy output without a signature -> refuse", sc.decide(leg, sig)[0] == "refuse")
        csig = sc.calibration_signature(src, cams, 0, 3, {"bins": 8, "min_bin_samples": 200})
        cd = td / "cal"
        sc.record(cd, csig, "running")
        from utils.noise.sensor_model import CameraNoise, SceneNoiseModel
        SceneNoiseModel(policy="calibrated", cameras={c: CameraNoise(c, depth_coeffs=(1e-6, 0, 0)) for c in cams}).to_json(cd / "sensor_model.json")
        (cd / "validation.json").write_text("{}")
        np.savez(cd / "covariance.npz", x=np.zeros(1))
        sc.record(cd, csig, "complete")
        check("[2] calibration: same -> reuse", sc.decide(cd, csig)[0] == "reuse")
        check("[2] calibration interval changed -> refuse",
              sc.decide(cd, sc.calibration_signature(src, cams, 0, 4, {"bins": 8, "min_bin_samples": 200}))[0] == "refuse")
        fm = src / "mask" / "cam_000" / "frame_000001.png"
        from PIL import Image
        mm = np.full((4, 5), 255, np.uint8)
        mm[0, 0] = 0
        Image.fromarray(mm).save(fm)
        check("[2] calibration: one mask file changed -> refuse", sc.decide(cd, sc.calibration_signature(
            src, cams, 0, 3, {"bins": 8, "min_bin_samples": 200}))[0] == "refuse")
        j = json.loads((cd / "sensor_model.json").read_text())
        j["cameras"]["cam_000"]["depth_coeffs"][0] = 5e-6
        (cd / "sensor_model.json").write_text(json.dumps(j))
        check("[2] calibration: edited sensor model -> refuse", sc.decide(cd, csig)[0] == "refuse")


# ---------------------------------------------------------------------------
def _poses(times, v=(0.3, 0.0, 0.0), w=(0.0, 0.0, 0.6)):
    from estimator.manifold import so3_exp
    T = []
    for t in times:
        R = so3_exp(torch.tensor([[w[0] * t, w[1] * t, w[2] * t]], dtype=torch.float64))[0]
        # left (spatial) twist with zero coupling: rotation about the origin plus translation
        M = torch.eye(4, dtype=torch.float64)
        M[:3, :3] = R
        M[:3, 3] = torch.tensor(v, dtype=torch.float64) * t
        T.append(M)
    return torch.stack(T)


def test_3():
    print("\n[3] physical-time exports")
    from utils.bidirectional_time import chronological_twists
    t = torch.arange(6, dtype=torch.float64) * 0.04
    xi = chronological_twists(_poses(t), torch.full((5,), 0.04, dtype=torch.float64))
    check("[3] 25 Hz constant twist: angular rate 0.6 rad/s per step", torch.allclose(xi[:, 3:].norm(dim=1),
          torch.full((6,), 0.6, dtype=torch.float64), atol=1e-9), str(xi[0, 3:].tolist()))
    xs = chronological_twists(_poses(t), 0.04)
    check("[3] scalar dt still supported and identical", torch.allclose(xs, xi))
    tn = torch.tensor([0.0, 0.04, 0.10, 0.18, 0.20], dtype=torch.float64)
    xn = chronological_twists(_poses(tn), tn[1:] - tn[:-1])
    check("[3] nonuniform intervals: interval-by-interval rates (constant motion -> constant rate)",
          torch.allclose(xn[:, 3:].norm(dim=1), torch.full((5,), 0.6, dtype=torch.float64), atol=1e-9))
    wrong = chronological_twists(_poses(tn), 1 / 30)
    check("[3] ... which a constant 1/30 s would get wrong", not torch.allclose(wrong, xn))
    # backward: processing order is reversed; chronological intervals = processing intervals reversed
    proc_time = torch.cumsum(torch.cat([torch.zeros(1, dtype=torch.float64), (tn[1:] - tn[:-1]).flip(0)]), 0)
    proc_iv = proc_time[1:] - proc_time[:-1]
    xb = chronological_twists(_poses(tn), proc_iv.flip(0))
    check("[3] backward reconstruction agrees with forward in sign and magnitude", torch.allclose(xb, xn))
    for bad in (torch.tensor([0.04, 0.0, 0.04, 0.04]), torch.tensor([0.04, -0.04, 0.04, 0.04])):
        try:
            chronological_twists(_poses(tn), bad.double())
            check("[3] zero / negative durations refused", False)
        except ValueError:
            check("[3] zero / negative durations refused", True)


# ---------------------------------------------------------------------------
def test_4():
    print("\n[4] read-only audit")
    from estimator.covariance import covariance_predict, make_process_noise
    from estimator.process import error_state_transition
    if not CUDA:
        NOT_RUN.append("[4] audit (CUDA)")
        return
    from tests.test_iekf_end_to_end import camera, make_state, render_gt
    st = make_state(n=3, seed=2, v=torch.tensor([[0.2, 0.0, 0.0]] * 3), omega=torch.tensor([[0.0, 0.3, 0.0]] * 3))
    dt = 0.04
    F = error_state_transition(st, dt, mode="so3")
    P = torch.zeros((3, 12, 12), device="cuda", dtype=st.mu.dtype)
    P[:, :6, :6] = torch.eye(6, device="cuda") * 1e-6
    P[:, 6:9, 6:9] = torch.eye(3, device="cuda") * 1.0
    P[:, 9:, 9:] = torch.eye(3, device="cuda") * 0.1
    Q = make_process_noise(3, dt, 1e-3, 1e-3, 0.1, 0.1, device="cuda", dtype=st.mu.dtype)
    Pp = covariance_predict(P, F, Q)
    check("[4] P_pred = F P F^T + Q (decomposition used by the audit)",
          float((Pp - (F @ P @ F.transpose(1, 2) + Q)).abs().max()) < 1e-6)
    Fpv = F[:, :6, 6:]
    vo = Fpv @ P[:, 6:, 6:] @ Fpv.transpose(1, 2)
    check("[4] velocity prior 1 m/s over 0.04 s -> 0.04 m position sd per axis",
          abs(math.sqrt(float(vo[0, 0, 0])) - 0.04) < 1e-4, f"{math.sqrt(float(vo[0, 0, 0])):.5f} m")
    check("[4] omega prior 0.316 rad/s over 0.04 s -> 0.01265 rad rotation sd",
          abs(math.sqrt(float(vo[0, 3, 3])) - 0.04 * math.sqrt(0.1)) < 1e-4, f"{math.sqrt(float(vo[0, 3, 3])):.5f} rad")
    from estimator import run_config as rc
    from estimator.sparse_iekf import IEKFConfig
    from utils.noise.sensor_model import CameraNoise, SceneNoiseModel
    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "s.json"
        SceneNoiseModel(policy="calibrated", cameras={"cam_000": CameraNoise("cam_000"), "cam_001": CameraNoise("cam_001")}).to_json(p)
        a = argparse.Namespace(sensor_model=str(p), sensor_policy="ideal", rgb_model_sigma=0.0, depth_model_sigma=0.0)
        try:
            rc.configure(a, IEKFConfig(), ["cam_000", "cam_009"])
            check("[4] a camera without calibration (swap / missing name) is refused", False)
        except SystemExit:
            check("[4] a camera without calibration (swap / missing name) is refused", True)
    # invariance: production bootstrap_positions with and without the audit
    from audit_calibrated_phase1 import Phase1Audit
    from tools.kpart_pipeline import bootstrap_positions
    cams = [camera()]
    truth = make_state(n=6, seed=4)
    cfg = IEKFConfig(max_iters=3)
    frames = []
    for k in range(3):
        moved = dataclasses.replace(truth, mu=truth.mu + torch.tensor([0.01 * k, 0, 0], device="cuda"))
        frames.append((render_gt(moved, cams, cfg), {"original_frame_index": k}))

    def go(audit):
        args = argparse.Namespace(supernode_size=8, seed=0, bootstrap_iters=3, dt=1 / 30.0, visibility_constraint=None,
                                  masked_observation=None, phase1_audit=audit)
        pos, info = bootstrap_positions(truth, cams, frames, cfg, None, args, verbose=False)
        return torch.stack([p_.double() for p_ in pos]), info
    p1, _ = go(None)
    p2, _ = go(None)
    floor = float((p1 - p2).abs().max())
    au = Phase1Audit()
    p3, _ = go(au)
    check("[4] tracking with the audit equals tracking without it",
          float((p1 - p3).abs().max()) <= floor, f"floor {floor:.2e}, with audit {float((p1 - p3).abs().max()):.2e}")
    check("[4] audit produced finite per-frame records", len(au.rows) == 2 and all(
        r["covariance"]["P_post"]["min_eigenvalue"] == r["covariance"]["P_post"]["min_eigenvalue"] for r in au.rows))


def main() -> int:
    for t in (test_1, test_2, test_3, test_4):
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
