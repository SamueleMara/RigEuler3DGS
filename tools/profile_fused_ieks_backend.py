#!/usr/bin/env python
"""Bounded real benchmark: one FIXED Washer linearisation, every backend.

CODEX_EXACT_FUSED_IEKS_ACCELERATION_SPARSE_DATA_LAYOUT_NO_APPROXIMATIONS.md
sections 2.4 and 9.

Builds the real two-source union (all Gaussians of both sources, the derived
12-D initial guess from the historical tracks, all requested frames and
cameras) at ONE state, then for each configuration

    legacy_reference + diagonal          (the production path until now)
    exact_cached_structured + diagonal
    exact_cached_structured + block12
    exact_cached_structured + temporal_block_tridiagonal

measures: system build, single operator application (with its parts), the
preconditioner build, a FIXED number of PCG iterations, and the iterations
and wall time to reach the SAME true residual ||b - A d|| / ||b|| that the
legacy path reaches in --reference-iters iterations. No nonlinear step is
taken; nothing is written outside --out. GPU contention is recorded, not
removed: other processes are listed and the result is labelled CONTENDED
when anything else is on the GPU.

  python tools/profile_fused_ieks_backend.py --frames 0:60 --cameras cam_000,cam_004 \\
      --out <new dir>
"""

from __future__ import annotations

import argparse
import json
import platform
import resource
import subprocess
import sys
import time
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "tools"))

O_DEF = "/media/samuelemara/Extreme SSD/Samuele/results_11_10_26/ours/washer_103776"
S_DEF = ("/media/samuelemara/Extreme SSD/Samuele/Backup_09_10_26/Sapien/output/"
         "preparation_8cams/partnet_subset_fixed_render_linear/washer_103776")


def gpu_snapshot():
    try:
        apps = subprocess.run(["nvidia-smi", "--query-compute-apps=pid,used_memory",
                               "--format=csv,noheader"], capture_output=True,
                              text=True).stdout.strip().splitlines()
        util = subprocess.run(["nvidia-smi", "--query-gpu=utilization.gpu,memory.used",
                               "--format=csv,noheader"], capture_output=True,
                              text=True).stdout.strip()
        return {"apps": apps, "util_mem": util}
    except Exception as exc:
        return {"error": str(exc)}


def host_rss_mib():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out-root", default=O_DEF)
    ap.add_argument("--data-root", default=S_DEF)
    ap.add_argument("--frames", default="0:60")
    ap.add_argument("--cameras", default="cam_000,cam_004")
    ap.add_argument("--fixed-iters", type=int, default=5)
    ap.add_argument("--reference-iters", type=int, default=20)
    ap.add_argument("--max-iters", type=int, default=60)
    ap.add_argument("--damping", type=float, default=1e-3)
    ap.add_argument("--configs", default="legacy:diagonal,cached:diagonal,cached:block12,"
                                         "cached:temporal_block_tridiagonal")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    out = Path(args.out)
    if (out / "benchmark.json").exists():
        raise SystemExit(f"{out} already holds a benchmark")
    out.mkdir(parents=True, exist_ok=True)

    import argparse as _ap
    from run_unified_phase1_real_bidirectional import Source, concat, parse_frames
    from bidirectional_iekf.full_window_free import pcg
    from bidirectional_iekf.ieks import IEKSConfig, Trajectory, UnifiedFreeIEKS
    from bidirectional_iekf.observations import FreeRGBDObservation
    from estimator.covariance import invert_blocks, make_process_noise
    from estimator.sparse_iekf import IEKFConfig
    from tools.run_iekf_scene import dataset_rgbd_frames

    t0 = time.perf_counter()
    O, S = Path(args.out_root), Path(args.data_root)
    src = {}
    for e, d in (("initial", "forward"), ("final", "backward")):
        src[e] = Source(e, O / "static_bidi" / e, None,
                        O / "bidirectional" / d / "phase1_centers", True, 30000, None, S)
    window = parse_frames(args.frames)
    names = args.cameras.split(",")
    cams_all = src["initial"].cams_all
    by = {str(getattr(c, "image_name", "")).split("/")[0]: c for c in cams_all}
    cams = [by[n] for n in names]
    cfg = IEKFConfig()
    ns = _ap.Namespace(sequence_start=window[0], sequence_length=len(window),
                       all_frames=False, observation_source="dataset_rgbd",
                       cameras=len(cams), all_cameras=False, window_length=0,
                       dt=1 / 30, frames_on_cpu=True, frames_on_gpu=False,
                       max_gaussians=0, seed=0)
    raw, _ = dataset_rgbd_frames(cams, cfg, ns, None, S, direction="forward")
    obs_data = [pc for pc, _s, _d in raw]
    specs = raw[0][1]
    states = [concat(src["initial"].state(f), src["final"].state(f)) for f in window]
    traj = Trajectory(states=states, frame_ids=window, dt=1 / 30)
    N = traj.N
    nI = src["initial"].N
    groups = []
    if 0 in window:
        groups.append((window.index(0), torch.arange(0, nI)))
    if src["final"].anchor_frame in window:
        groups.append((window.index(src["final"].anchor_frame), torch.arange(nI, N)))
    Q = make_process_noise(N, 1 / 30, cfg.process_noise_pos, cfg.process_noise_rot,
                           cfg.process_noise_vel, cfg.process_noise_omega,
                           device="cuda", dtype=torch.float64)
    Qi, _ = invert_blocks(Q)
    del Q
    setup_s = time.perf_counter() - t0
    print(f"setup {setup_s:.1f} s: N={N}, frames {window[0]}..{window[-1]}, cameras {names}, "
          f"{len(groups)} anchor groups", flush=True)

    meta = {"argv": sys.argv, "N": N, "frames": window, "cameras": names,
            "factors": len(window) * len(names), "damping": args.damping,
            "python": platform.python_version(), "torch": torch.__version__,
            "cuda": torch.version.cuda, "gpu": torch.cuda.get_device_name(0),
            "commit": subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO,
                                     capture_output=True, text=True).stdout.strip(),
            "gpu_before": gpu_snapshot(), "setup_s": setup_s,
            "initialization": "DERIVED_12D_INITIAL_GUESS_FROM_EXISTING_TRACKS"}
    others = [a for a in meta["gpu_before"].get("apps", []) if str(__import__("os").getpid())
              not in a]
    meta["contention"] = "CONTENDED" if others else "UNCONTENDED"

    results = {}
    target = None
    for spec in args.configs.split(","):
        backend, pre = spec.split(":")
        label = f"{backend}+{pre}"
        print(f"\n=== {label} ===", flush=True)
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        obs = FreeRGBDObservation(specs, obs_data, frame_ids=window,
                                  sigma_rgb=cfg.rgb_sigma,
                                  depth_sigma_model={"kind": cfg.depth_sigma_model,
                                                     "a": cfg.depth_sigma_a, "b": 0.0},
                                  robust=False, min_alpha_for_depth=cfg.min_alpha_for_depth)
        ieks = UnifiedFreeIEKS(obs, Qi, config=IEKSConfig(
            source_pose_policy="fixed_conditional", anchor_groups=tuple(groups),
            linear_backend=("exact_cached_structured" if backend == "cached"
                            else "legacy_reference"),
            preconditioner=pre, stream_frames=True, process_factors_on_cpu=True,
            diagonal_preconditioner=(pre == "diagonal"), solve_on_cpu=True,
            verbose=False))
        r = {}
        tb = time.perf_counter()
        op, _ = ieks.build_system(traj, args.damping)
        b = op.rhs()
        op.curvature_scale()
        r["build_system_s"] = time.perf_counter() - tb
        if backend == "cached":
            st = op.store
            r["cache_host_gib"] = st.host_bytes / 2 ** 30
            r["cache_build_s"] = st.timing["build_s"]
            r["cache_offload_s"] = st.timing["offload_s"]
            r["process_coeff_mib"] = op.proc.coeff_bytes / 2 ** 20
            r["process_on_gpu"] = op.proc.on_gpu
        tp = time.perf_counter()
        M = ieks._preconditioner(op, args.damping)
        r["preconditioner_build_s"] = time.perf_counter() - tp
        u = torch.randn(len(window), N, 12, dtype=torch.float64,
                        generator=torch.Generator().manual_seed(0))
        torch.cuda.synchronize()
        tm = time.perf_counter()
        op.information_matvec(u)
        torch.cuda.synchronize()
        r["one_matvec_s"] = time.perf_counter() - tm
        if backend == "cached":
            r["matvec_parts_s"] = dict(op.timing)
            r["stage_s"] = op.store.timing["stage_s"]
        ta = time.perf_counter()
        M(b)
        r["preconditioner_apply_s"] = time.perf_counter() - ta
        tf = time.perf_counter()
        _x, info = pcg(op, b, preconditioner=M, max_iters=args.fixed_iters, tol=1e-30,
                       true_residual=True)
        r["fixed_iters"] = args.fixed_iters
        r["fixed_pcg_s"] = time.perf_counter() - tf
        r["fixed_true_residual"] = info["true_residual"]
        if target is None:
            tr = time.perf_counter()
            _x, info = pcg(op, b, preconditioner=M, max_iters=args.reference_iters,
                           tol=1e-30, true_residual=True)
            target = info["true_residual"]
            r["to_target"] = {"iterations": info["iterations"],
                              "wall_s": time.perf_counter() - tr,
                              "true_residual": target, "reached": True,
                              "note": "defines the target"}
            print(f"  target true residual {target:.3e} after {info['iterations']} "
                  f"iterations", flush=True)
        else:
            hist = []
            tr = time.perf_counter()

            def prog(k, rel):
                hist.append((k, rel, time.perf_counter() - tr))
            # stop on the recursive residual, then CHECK the true residual
            _x, info = pcg(op, b, preconditioner=M, max_iters=args.max_iters,
                           tol=target, true_residual=True, progress=prog)
            r["to_target"] = {"iterations": info["iterations"],
                              "wall_s": time.perf_counter() - tr,
                              "true_residual": info["true_residual"],
                              "reached": info["true_residual"] <= target * 1.05,
                              "history": hist}
        r["peak_gpu_mib"] = torch.cuda.max_memory_allocated() / 2 ** 20
        r["peak_host_rss_mib"] = host_rss_mib()
        results[label] = r
        print("  " + json.dumps({k: (round(v, 4) if isinstance(v, float) else v)
                                 for k, v in r.items() if k not in ("to_target",)}), flush=True)
        print(f"  to target: {r['to_target']['iterations']} iterations, "
              f"{r['to_target']['wall_s']:.1f} s, true residual "
              f"{r['to_target']['true_residual']:.3e}", flush=True)
        del op, M, ieks, obs
        import gc
        gc.collect()
    meta["gpu_after"] = gpu_snapshot()
    payload = {"meta": meta, "results": results, "target_true_residual": target}
    (out / "benchmark.json").write_text(json.dumps(payload, indent=1, default=str))
    print(f"\n{out / 'benchmark.json'}  [{meta['contention']}]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
