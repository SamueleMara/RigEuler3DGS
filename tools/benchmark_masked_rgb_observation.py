#!/usr/bin/env python
"""Benchmark L of CODEX_IMPLEMENT_EXACT_MASKED_RGB_VALID_DEPTH_IEKF_2026-10-20.md:
one tracked Phase-1 frame, original vs masked observation mode, same scene,
cameras, static endpoint and IEKF settings.

For each mode it runs the production Phase-1 path (tools/kpart_pipeline.py
bootstrap_positions -> estimator.sparse_iekf.iekf_frame_step) on frames 0..1
(frame 0 is the canonical state, frame 1 is tracked) and records:
  * wall time and its breakdown: data loading, mask preparation, primal renders,
    RGB-D JVP / VJP, alpha JVP / VJP, Fisher builds, candidate evaluations;
  * peak GPU memory;
  * the energy terms (original: RGB + depth; masked: RGB, alpha, depth -- different
    objectives, never compared as numbers);
  * background leakage of the tracked state -- its centres AND orientations, never
    the canonical rotations -- with estimator/mask_diagnostics.py, the same
    GT-free definition the pipeline stages write (alpha on mask = 0 pixels on a
    fixed evaluation background, foreground coverage, valid-depth counts);
  * the state change: |mu_1 - mu_0| statistics.
Nothing is written outside --out (a NEW directory); GT is not read.

    python tools/benchmark_masked_rgb_observation.py \\
        --model <results>/fridge_10905/static_bidi/initial \\
        --source <dataset>/fridge_10905 --out <new dir> [--cameras 8] [--iters 12]
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "tools"))


class Timers:
    def __init__(self):
        self.t, self.n = {}, {}

    def wrap(self, owner, name, key):
        f = getattr(owner, name)

        def g(*a, **k):
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            out = f(*a, **k)
            torch.cuda.synchronize()
            self.t[key] = self.t.get(key, 0.0) + time.perf_counter() - t0
            self.n[key] = self.n.get(key, 0) + 1
            return out
        setattr(owner, name, g)
        return (owner, name, f)


def run_mode(mode, args, scene_cache):
    import estimator.renderer_bridge as rb
    import estimator.sparse_iekf as si
    import estimator.masked_observation as mo
    from tools.kpart_pipeline import bootstrap_positions
    truth, cams_all, frames_raw, frame_ids, cfg, t_load = scene_cache
    ns = argparse.Namespace(
        sequence_start=0, sequence_length=2, all_frames=False, observation_source="dataset_rgbd",
        cameras=len(frames_raw[0][1]), bootstrap_iters=args.iters, supernode_size=args.supernode_size,
        seed=0, frames_on_cpu=True, frames_on_gpu=False, depth_sigma=None, joint_direct_rgbd=False,
        joint_direct_gaussian=False, window_length=0, all_cameras=False, dt=1.0 / 30.0, max_gaussians=0,
        no_model_error=False, model_error_beta=1.0, model_error_blur=1, visibility_constraint=None,
        masked_observation=None, observation_mode=mode, rgb_background_constant=None,
        rgb_background_distribution=None, object_mask_constraint="off",
        mask_source=args.mask_source)
    T = Timers()
    t0 = time.perf_counter()
    if mode != "original":
        ns.masked_observation = mo.from_args(ns, Path(args.source), frames_raw, frame_ids, cfg)
    t_masks = time.perf_counter() - t0
    restore = [T.wrap(rb, "iekf_render", "primal render (per camera)"),
               T.wrap(rb.FrameLinearization, "apply_H", "RGB-D JVP"),
               T.wrap(rb.FrameLinearization, "apply_HT", "RGB-D VJP"),
               T.wrap(rb.FrameLinearization, "apply_H_alpha", "alpha JVP"),
               T.wrap(rb.FrameLinearization, "apply_HT_alpha", "alpha VJP"),
               T.wrap(rb.FrameLinearization, "build_block_fisher", "RGB-D Fisher"),
               T.wrap(rb.FrameLinearization, "build_alpha_fisher", "alpha Fisher"),
               T.wrap(si, "evaluate_true_map_energy", "nonlinear evaluations (incl. their renders)"),
               T.wrap(si, "steihaug_cg", "matrix-free CG solves (incl. their JVP/VJP)"),
               T.wrap(si, "approximate_posterior_information", "posterior information"),
               T.wrap(si, "invert_blocks", "PSD block inversions"),
               T.wrap(si, "project_psd", "PSD projections"),
               T.wrap(mo.MaskedSequence, "prepare_frame", "mask/precision preparation")]
    torch.cuda.reset_peak_memory_stats()
    window = [(ob, fid) for (ob, _s, _d), fid in zip(frames_raw, frame_ids["frames"])]
    t0 = time.perf_counter()
    rows = {}

    def observer(ti, state, *_):
        rows[ti] = (state.mu.detach().clone(), state.R.detach().clone())
    try:
        positions, _info = bootstrap_positions(truth, frames_raw[0][1], window, cfg, None, ns,
                                               verbose=False, observer=observer)
    finally:
        for owner, name, f in restore:
            setattr(owner, name, f)
    wall = time.perf_counter() - t0
    peak = torch.cuda.max_memory_allocated() / 2 ** 20
    mu1, R1 = (t.to(truth.mu.device) for t in rows[1])
    if not torch.equal(mu1, positions[1].to(mu1.device)):
        raise AssertionError("observer state and returned positions disagree for frame 1")
    dx = (mu1 - truth.mu).norm(dim=1)
    # background leakage of the tracked state (centres AND orientations); masks
    # are an input measurement, loaded here for the report only
    from estimator.mask_diagnostics import MaskDiagnostics, masks_for_run
    from estimator.renderer_bridge import build_linearization
    mfn, mroot, why = masks_for_run(ns, Path(args.source), frames_raw, frame_ids, ns.masked_observation)
    md = MaskDiagnostics("phase1 benchmark", mfn, mroot, frames_raw[0][1], cfg, truth, build_linearization,
                         reason=why)
    md.frame(1, frame_ids["frames"][1], mu1, R1, frames_raw[1][0])
    leak = md.report()
    rot_change = float((R1 - truth.R).flatten(1).norm(dim=1).max())
    report = None if ns.masked_observation is None else ns.masked_observation.report
    return {"mode": mode, "wall_time_s_tracked_frame": wall, "data_loading_s (shared)": t_load,
            "mask_loading_s": t_masks, "gpu_peak_MiB": peak,
            "breakdown_s": T.t, "calls": T.n,
            "energy_terms": report[-1] if report else "original mode: see the IEKF diag (RGB + depth only)",
            "background_leakage": leak,
            "max_rotation_change_frobenius": rot_change,
            "state_change_vs_canonical_m": {"mean": float(dx.mean()), "median": float(dx.median()),
                                            "p99": float(dx.quantile(0.99)) if dx.numel() < 2 ** 24 else None,
                                            "max": float(dx.max())}}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--model", required=True)
    ap.add_argument("--source", required=True)
    ap.add_argument("--out", required=True, help="a NEW directory")
    ap.add_argument("--cameras", type=int, default=8)
    ap.add_argument("--iters", type=int, default=12)
    ap.add_argument("--supernode-size", type=int, default=64)
    ap.add_argument("--mask-source", default=None)
    ap.add_argument("--modes", default="original,masked-rgb-expected-quad")
    args = ap.parse_args(argv)
    out = Path(args.out)
    if out.exists() and any(out.iterdir()):
        raise SystemExit(f"{out} exists and is not empty; give a new directory")
    out.mkdir(parents=True, exist_ok=True)
    from estimator.sparse_iekf import IEKFConfig
    from tools.run_iekf_scene import dataset_rgbd_frames, load_scene, state_from_gaussians
    t0 = time.perf_counter()
    dataset, pipe, gaussians, sc = load_scene(args.model, args.source, camera_device="cpu")
    cams_all = sc.getTrainCameras()
    ncam = min(args.cameras, len(cams_all))
    truth = state_from_gaussians(gaussians)
    cfg = IEKFConfig()
    ns = argparse.Namespace(sequence_start=0, sequence_length=2, all_frames=False, observation_source="dataset_rgbd",
                            cameras=ncam, frames_on_cpu=True, frames_on_gpu=False, window_length=0,
                            all_cameras=False, dt=1.0 / 30.0)
    frames_raw, frame_ids = dataset_rgbd_frames(cams_all[:ncam], cfg, ns, truth, Path(args.source), direction="forward")
    t_load = time.perf_counter() - t0
    results = []
    for mode in [m.strip() for m in args.modes.split(",") if m.strip()]:
        print(f"== {mode}")
        r = run_mode(mode, args, (truth, cams_all, frames_raw, frame_ids, cfg, t_load))
        print(json.dumps({**{k: r[k] for k in ("wall_time_s_tracked_frame", "gpu_peak_MiB",
                                               "state_change_vs_canonical_m")},
                          "leakage": r["background_leakage"].get("mean_over_frames")}, indent=1))
        results.append(r)
        torch.cuda.empty_cache()
    payload = {"scene": Path(args.source).name, "model": args.model, "cameras": ncam,
               "frames": "0 (canonical) and 1 (tracked)", "phase1_iekf_iterations": args.iters,
               "gaussians": int(truth.mu.shape[0]), "gpu": torch.cuda.get_device_name(0),
               "results": results,
               "comparability": ("the original and masked objectives are different functions; their energies "
                                 "are not compared. Wall times are this run's, on this GPU; the earlier "
                                 "14 s (original) and 146 s (hard-constraint) smokes were NOT re-measured here "
                                 "unless the 'original' entry above is that re-measurement."),
               "historical_reference_s": {"constraint_off_smoke": 14.0, "hard_constraint_smoke": 146.0}}
    (out / "masked_rgb_depth_benchmark.json").write_text(json.dumps(payload, indent=1, default=str))
    print(f"wrote {out / 'masked_rgb_depth_benchmark.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
