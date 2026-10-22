#!/usr/bin/env python
"""Regression tests of CODEX_SURGICAL_CONFIRMED_BUG_FIXES_RIGEULER3DGS_2026-10-22.md.

  [A] frame-time units: processing-order elapsed seconds forward AND backward,
      contradictory / non-positive intervals refused, nonuniform intervals;
      0.3 m/s and 0.6 rad/s recovered by the Phase 2 Kabsch twist, the Phase 2
      prediction and the Phase 3 finite-difference qdot; Q of q/qdot at dt = 1/30;
      the production call sites pass the loader's times and the window solvers
      predict with dt = 1/30 (real CUDA renderer, tiny scene)
  [B] stage signatures hash RGB / depth BYTES: same-size changes refuse reuse,
      identical inputs reuse, mask changes still refuse, schema-1 manifests refused
  [C] run_bidirectional_all.sh exit status with a stub stage runner: all success 0,
      a stage failure / a refusal / missing masks nonzero, a later success does not
      reset an earlier failure, dry run unchanged

    python tests/test_surgical_fixes_2026_10_22.py
"""
from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "tools"))
CHECKS, FAILURES, NOT_RUN = 0, [], []
DT = 1.0 / 30.0


def check(label, cond, detail=""):
    global CHECKS
    CHECKS += 1
    print(f"  {'ok  ' if cond else 'FAIL'}  {label}  {detail}")
    if not cond:
        FAILURES.append(label)


def refused(fn, *a):
    try:
        fn(*a)
        return False
    except (ValueError, SystemExit):
        return True


# ---------------------------------------------------------------------------
def test_A_time_base():
    print("\n[A] time base")
    from utils.sequence_direction import (annotate_forward, processing_time_s, reverse_sequence,
                                          times_from_intervals)
    frames = [(f"obs{k}", "specs", DT) for k in range(3)]
    prov = {"frames": [{"frame": k, "dataset_frame_index": k, "dt_used_s": DT} for k in range(3)]}
    fwd = processing_time_s(frames, annotate_forward(prov))
    bf, bp = reverse_sequence(frames, prov)
    bwd = processing_time_s(bf, bp)
    ref = torch.tensor([0.0, DT, 2 * DT], dtype=torch.float64)
    check("[A] forward time_s = [0, 1/30, 2/30]", torch.allclose(fwd, ref, atol=0, rtol=1e-15), str(fwd.tolist()))
    check("[A] backward time_s = [0, 1/30, 2/30] (elapsed time increases along processing)",
          torch.allclose(bwd, ref, atol=0, rtol=1e-15)
          and [r["dataset_frame_index"] for r in bp["frames"]] == [2, 1, 0])
    bad = {"frames": [dict(r, dt_used_s=0.04) if r["frame"] == 1 else r for r in prov["frames"]]}
    check("[A] contradictory loader / provenance interval is refused", refused(processing_time_s, frames, bad))
    for name, iv in (("zero", [DT, 0.0, DT]), ("negative", [DT, -DT, DT]), ("nan", [DT, float("nan"), DT]),
                     ("inf", [DT, float("inf"), DT])):
        check(f"[A] a {name} interval is refused", refused(times_from_intervals, iv))
    nu = times_from_intervals([0.0, 0.03, 0.05])
    check("[A] nonuniform positive intervals -> cumulative elapsed time",
          torch.allclose(nu, torch.tensor([0.0, 0.03, 0.08], dtype=torch.float64)), str(nu.tolist()))


def test_A_rates():
    print("\n[A] rates in physical units")
    from estimator.manifold import so3_exp
    from estimator.rigid_part_runtime import RigidIEKFConfig, RigidState, kabsch_initialize
    from run_joint_phase3 import _finite_rate
    g = torch.Generator().manual_seed(0)
    X0 = torch.randn(40, 3, generator=g, dtype=torch.float64)
    t = torch.tensor([0.0, DT, 2 * DT], dtype=torch.float64)
    v = torch.tensor([0.3, 0.0, 0.0], dtype=torch.float64)
    pos = [X0 + v * float(tk) for tk in t]
    members = torch.arange(40)
    k = kabsch_initialize(members, X0, pos, times=t)
    part = k["part"]
    check("[A] 0.3 m/s moves 0.01 m in one 30 Hz interval (Kabsch pose)",
          abs(float(part.p[1][0]) - 0.01) < 1e-12, f"{float(part.p[1][0]):.6f} m")
    check("[A] Phase 2 Kabsch initial twist is 0.3 m/s (not 0.01 m/frame)",
          torch.allclose(part.v[0], v, atol=1e-9), str([round(x, 6) for x in part.v[0].tolist()]))
    w = 0.6
    R = [so3_exp(torch.tensor([[0.0, 0.0, w * float(tk)]], dtype=torch.float64))[0] for tk in t]
    kr = kabsch_initialize(members, X0, [X0 @ Rk.T for Rk in R], times=t)
    check("[A] Phase 2 Kabsch initial angular rate is 0.6 rad/s",
          abs(float(kr["part"].omega[0][2]) - w) < 1e-9, f"{float(kr['part'].omega[0][2]):.6f}")
    # Phase 2 prediction with dt = 1/30 and v in m/s
    st = RigidState(k["part"].clone(), RigidIEKFConfig())
    st.parts[0].R[1] = torch.eye(3, dtype=torch.float64)
    st.parts[0].p[1] = torch.zeros(3, dtype=torch.float64)
    st.predict(1, DT)
    check("[A] Phase 2 prediction advances 0.3 m/s x 1/30 s = 0.01 m",
          abs(float(st.parts[0].p[2][0] - st.parts[0].p[1][0]) - 0.01) < 1e-12,
          f"p = {[round(float(x), 6) for x in st.parts[0].p[:, 0]]}")
    # Phase 3 finite-difference qdot, prismatic and revolute, forward and backward
    q = 0.3 * t.numpy()
    check("[A] Phase 3 initial qdot = 0.3 m/s from the same time vector",
          np.allclose(_finite_rate(q, t.numpy()), 0.3, atol=1e-12), str(_finite_rate(q, t.numpy())))
    th = w * t.numpy()
    check("[A] Phase 3 initial thetadot = 0.6 rad/s", np.allclose(_finite_rate(th, t.numpy()), w, atol=1e-12))
    qb = 0.3 * (2 * DT - t.numpy())          # backward: original frames 2,1,0 -> q decreases
    check("[A] backward processing: positive elapsed time, rate sign follows the motion",
          np.allclose(_finite_rate(qb, t.numpy()), -0.3, atol=1e-12))
    check("[A] a time vector of the wrong length is refused", refused(_finite_rate, q, t.numpy()[:2]))


def test_A_process_noise():
    print("\n[A] process noise at dt = 1/30")
    from estimator import phase2_prismatic as mp
    from estimator.joint_displacement import DisplacementOnlyState
    from estimator.joint_process import process_noise
    from estimator.phase2_model_comparison import prismatic_state
    part = mp.PrismaticPart(torch.arange(5), torch.tensor([1.0, 0.0, 0.0], dtype=torch.float64),
                            torch.zeros(3, dtype=torch.float64))
    ds = DisplacementOnlyState(prismatic_state(part))
    s = 1.0
    Q = process_noise(ds, DT, s)
    ok = (abs(float(Q[0, 0]) - s * s * DT ** 3 / 3) < 1e-18 and abs(float(Q[0, 1]) - s * s * DT ** 2 / 2) < 1e-18
          and abs(float(Q[1, 1]) - s * s * DT) < 1e-15)
    check("[A] Q_qq = s^2 dt^3/3, Q_q,qdot = s^2 dt^2/2, Q_qdot,qdot = s^2 dt at dt = 1/30", ok,
          f"{float(Q[0, 0]):.3e} {float(Q[0, 1]):.3e} {float(Q[1, 1]):.3e}")


def test_A_production_paths():
    print("\n[A] production call sites")
    p2 = (REPO / "tools" / "run_rigid_se3_pipeline.py").read_text()
    p3 = (REPO / "tools" / "run_joint_phase3.py").read_text()
    check("[A] Phase 2 builds time_s from the loader and passes it to Kabsch and the window",
          "processing_time_s(frames_raw, frame_ids)" in p2 and "for p in positions], times=time_s)" in p2
          and "times=time_s," in p2)
    check("[A] Phase 3 passes time_s to fit_model and to the initial qdot",
          "processing_time_s(frames_raw, frame_ids)" in p3 and "times=time_s," in p3
          and "load_initialisation(p2, members, mu0, T, time_s=time_s.numpy())" in p3)
    if not torch.cuda.is_available():
        NOT_RUN.append("[A] window dt on the real renderer")
        print("  NOT_RUN  [A] window dt on the real renderer (CUDA unavailable)")
        return
    import dataclasses
    from estimator import phase2_prismatic as mp
    from estimator.joint_displacement import DisplacementOnlyState
    from estimator.joint_iekf import JointIEKFConfig
    from estimator.phase2_model_comparison import fit_model, prismatic_state
    from estimator.phase2_shared import measurement_terms, render_state
    from estimator.renderer_bridge import build_linearization
    from estimator.rigid_part_runtime import RigidIEKFConfig, RigidState, run_rigid_window
    from estimator.rigid_part_se3 import RigidPart
    from estimator.sparse_iekf import IEKFConfig
    from tests.test_unified_cuda_renderer_jvp_vjp import make_cameras
    from tests.test_visibility_constrained_iekf_phase1 import observe
    from tests.test_visibility_constrained_iekf_phase23 import scene
    cams = make_cameras(n_cam=2, H=48, W=48, radius=2.2)
    st, members = scene()
    obs = [observe(st, cams)[0] for _ in range(3)]
    cfg = IEKFConfig()
    mu0, R0 = st.mu.double(), st.R.double()
    ctr = {"t": 0}

    def lin(s_, t_):
        ctr["t"] = t_
        mu, Rg = s_.gaussians(t_, mu0, R0)
        return render_state(st, mu, Rg, cams, cfg, build_linearization)

    def prec(L):
        return measurement_terms(L, obs[ctr["t"]], [0, 1], cfg)
    times = torch.tensor([0.0, DT, 2 * DT], dtype=torch.float64)
    seen = []
    rs = RigidState(RigidPart(members, num_frames=3), RigidIEKFConfig(max_iters=1))
    _p = rs.predict
    rs.predict = lambda f, dt, *a, **k: (seen.append(("rigid", dt)), _p(f, dt, *a, **k))[1]
    run_rigid_window(rs, mu0, R0, obs, lin, prec, times=times, cfg=RigidIEKFConfig(max_iters=1))
    part = mp.PrismaticPart(members, torch.tensor([1.0, 0.0, 0.0], dtype=torch.float64),
                            torch.zeros(3, dtype=torch.float64))
    ds = DisplacementOnlyState(prismatic_state(part))
    _q = ds.predict
    ds.predict = lambda f, dt, *a, **k: (seen.append(("joint", dt)), _q(f, dt, *a, **k))[1]
    fit_model(ds, mu0, obs, lin, prec, JointIEKFConfig(max_iters=1), label="t", times=times)
    dts = [d for _n, d in seen]
    check("[A] Phase 2 and Phase 3 windows predict with dt = 1/30 (never 1)",
          len(dts) == 4 and all(abs(d - DT) < 1e-15 for d in dts), str([(n, round(d, 6)) for n, d in seen]))


# ---------------------------------------------------------------------------
def _write_png(path, arr):
    from PIL import Image
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(arr).save(path)


def test_B_signatures():
    print("\n[B] content signatures")
    import stage_signature as ss
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        src, model = td / "src", td / "model"
        rgb = np.zeros((4, 4, 3), np.uint8)
        rgb[1, 1] = 200
        _write_png(src / "rgb" / "cam_000" / "frame_000000.png", rgb)
        _write_png(src / "mask" / "cam_000" / "frame_000000.png", np.zeros((4, 4), np.uint8))
        (src / "depth_npy" / "cam_000").mkdir(parents=True)
        np.save(src / "depth_npy" / "cam_000" / "frame_000000.npy", np.ones((4, 4), np.float32))
        ply = model / "point_cloud" / "iteration_30000" / "point_cloud.ply"
        ply.parent.mkdir(parents=True)
        ply.write_bytes(b"ply")
        (model / "cfg_args").write_text("Namespace()")

        def sig():
            a = argparse.Namespace(stage="phase3", direction="forward", scene="toy", source=str(src),
                                   model=str(model), frames=1, cameras=1, iterations="30000",
                                   observation_mode="masked-rgb-expected-quad",
                                   rgb_background_distribution="uniform01", rgb_background_constant=None,
                                   object_mask_constraint="off", mask_options=None, mask_source=None,
                                   upstream=None, config=["iters=8"])
            return ss.compute(a)
        d = td / "out"
        d.mkdir()
        (d / "summary.json").write_text(json.dumps({"ok": 1}))
        s0 = sig()
        ss.write(d, s0, "complete")
        check("[B] identical inputs -> reuse (0)", ss.check(d, sig())[0] == 0)
        check("[B] schema 2 with content fields",
              s0["schema"] == 2 and "rgb_content_sha256" in s0["source"] and "depth_content_sha256" in s0["source"])
        f = src / "rgb" / "cam_000" / "frame_000000.png"
        b0 = f.read_bytes()
        rgb2 = rgb.copy()
        rgb2[1, 1] = 201                       # one pixel; PNG size checked below
        _write_png(f, rgb2)
        same = f.stat().st_size == len(b0)
        if not same:                           # force an equal-size byte change
            b = bytearray(b0)
            b[-13] ^= 0xFF
            f.write_bytes(bytes(b))
        rc, msg = ss.check(d, sig())
        check("[B] RGB bytes changed, file size unchanged -> refuse (12)",
              f.stat().st_size == len(b0) and rc == 12 and "rgb_content_sha256" in msg, msg[-90:])
        f.write_bytes(b0)
        check("[B] RGB restored -> reuse again", ss.check(d, sig())[0] == 0)
        dp = src / "depth_npy" / "cam_000" / "frame_000000.npy"
        n0 = dp.stat().st_size
        arr = np.ones((4, 4), np.float32)
        arr[2, 2] = 1.0001
        np.save(dp, arr)
        rc, msg = ss.check(d, sig())
        check("[B] depth float changed, file size unchanged -> refuse (12)",
              dp.stat().st_size == n0 and rc == 12 and "depth_content_sha256" in msg, msg[-90:])
        np.save(dp, np.ones((4, 4), np.float32))
        _write_png(src / "mask" / "cam_000" / "frame_000000.png", np.full((4, 4), 255, np.uint8))
        rc, msg = ss.check(d, sig())
        check("[B] changed masks still refuse", rc == 12 and "mask_content_sha256" in msg)
        _write_png(src / "mask" / "cam_000" / "frame_000000.png", np.zeros((4, 4), np.uint8))
        old = json.loads((d / ss.MANIFEST).read_text())
        sch1 = dict(old["signature"], schema=1)
        sch1["source"] = {k: v for k, v in sch1["source"].items() if "content" not in k}
        sch1["source"].update(rgb_listing_sha256="x", depth_listing_sha256="y")
        (d / ss.MANIFEST).write_text(json.dumps({"status": "complete", "signature": sch1,
                                                 "signature_sha256": ss.sig_hash(sch1)}))
        check("[B] a schema-1 (size-only) manifest is never reused", ss.check(d, sig())[0] == 12)


# ---------------------------------------------------------------------------
STUB = r'''#!REALPY
import json, os, sys
from pathlib import Path
REAL = "REALPY"
stages = {"tools/run_motion_coherence_support.py": "phase1", "tools/run_rigid_se3_pipeline.py": "phase2",
          "tools/run_joint_phase3.py": "phase3"}
if len(sys.argv) < 2 or sys.argv[1] not in stages:
    os.execv(REAL, [REAL] + sys.argv[1:])
st = stages[sys.argv[1]]
a = sys.argv[2:]
scene, direction, out = a[a.index("--scene") + 1], a[a.index("--direction") + 1], Path(a[a.index("--out") + 1])
if f"{scene}:{st}:{direction}" in os.environ.get("STUB_FAIL", "").split():
    print("stub failure"); sys.exit(3)
import numpy as np
out.mkdir(parents=True, exist_ok=True)
if st == "phase1":
    (out / "predicted_support.json").write_text(json.dumps({"status": "OK", "dynamic_gaussians": [0]}))
    (out / "phase1_centers").mkdir(exist_ok=True)
    np.save(out / "phase1_centers" / "canonical_xyz.npy", np.zeros((2, 3)))
    np.save(out / "phase1_centers" / "centers_frame_0000.npy", np.zeros((2, 3)))
elif st == "phase2":
    (out / "phase2_rigid").mkdir(exist_ok=True)
    np.save(out / "phase2_rigid" / "poses.npy", np.tile(np.eye(4), (2, 1, 1)))
    (out / "support_coherence").mkdir(exist_ok=True)
    (out / "support_coherence" / "support.json").write_text("{}")
    (out / "joint_reduction").mkdir(exist_ok=True)
    (out / "summary.json").write_text("{}")
else:
    (out / "summary.json").write_text(json.dumps({"ok": 1}))
'''


def _fixture_scene(root: Path, out: Path, name: str, masks=True):
    d = root / name
    for k in ("colmap/sparse/0", "colmap_final/sparse/0"):
        (d / k).mkdir(parents=True)
    for f in range(2):
        _write_png(d / "rgb" / "cam_000" / f"frame_{f:06d}.png", np.zeros((4, 4, 3), np.uint8))
        (d / "depth_npy" / "cam_000").mkdir(parents=True, exist_ok=True)
        np.save(d / "depth_npy" / "cam_000" / f"frame_{f:06d}.npy", np.ones((4, 4), np.float32))
        if masks:
            _write_png(d / "mask" / "cam_000" / f"frame_{f:06d}.png", np.zeros((4, 4), np.uint8))
    for e in ("initial", "final"):
        m = out / name / "static_bidi" / e
        (m / "point_cloud" / "iteration_30000").mkdir(parents=True)
        (m / "point_cloud" / "iteration_30000" / "point_cloud.ply").write_bytes(b"ply\n" + b"0" * 120000)
        (m / "cfg_args").write_text("Namespace()")


def test_C_exit_status():
    print("\n[C] driver exit status (stub stages)")
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        stub = td / "stubpy"
        stub.write_text(STUB.replace("REALPY", sys.executable))
        stub.chmod(0o755)

        def run(case, scenes, fail="", extra=(), legacy=None, nomask=()):
            root, out = td / case / "in", td / case / "out"
            for s in scenes.split():
                _fixture_scene(root, out, s, masks=s not in nomask)
            if legacy:
                p = out / legacy / "bidirectional" / "forward"
                p.mkdir(parents=True)
                (p / "predicted_support.json").write_text("{}")
            env = dict(os.environ, PY=str(stub), STUB_FAIL=fail, STALL_MIN="0")
            r = subprocess.run(["bash", "run_bidirectional_all.sh", "-i", str(root), "-o", str(out),
                                "--scenes", scenes, "--stages", "phase1 phase2 phase3", "--cameras", "1",
                                *extra], cwd=REPO, env=env, capture_output=True, text=True, timeout=600)
            return r.returncode, r.stdout
        rc, out = run("ok", "sa sb")
        check("[C] all stages complete -> exit 0", rc == 0 and out.count(" done") == 2,
              f"rc {rc}; " + " | ".join(ln.strip() for ln in out.splitlines()[-4:]))
        rc, out = run("fail", "sa sb", fail="sa:phase2:backward")
        check("[C] one stage failure -> nonzero", rc != 0 and "FAILED(phase2/backward)" in out, f"rc {rc}")
        check("[C] a later successful scene does not reset the failure", rc != 0 and "sb done" in out)
        rc, out = run("refuse", "sa", legacy="sa")
        check("[C] a signature refusal -> nonzero", rc != 0 and "REFUSED(phase1/forward" in out, f"rc {rc}")
        rc, out = run("nomask", "sa sb", nomask=("sa",))
        check("[C] missing masks -> nonzero", rc != 0 and "sa SKIPPED(masks not ready)" in out, f"rc {rc}")
        rc, out = run("dry", "sa", extra=("--dry-run",))
        written = sorted(p.name for p in (td / "dry" / "out" / "sa").iterdir())
        check("[C] dry run unchanged: commands printed, nothing written, exit 0",
              rc == 0 and "run_motion_coherence_support" in out and written == ["static_bidi"], f"rc {rc}")


def main() -> int:
    for t in (test_A_time_base, test_A_rates, test_A_process_noise, test_A_production_paths,
              test_B_signatures, test_C_exit_status):
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
