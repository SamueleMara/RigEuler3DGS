#!/usr/bin/env python
"""Real RGB-D `SINGLE_SOURCE` unified IEKS on one trained endpoint.

Section 4.2 of CODEX_COMPLETE_MISSING_UNIFIED_BIDI_PHASE1_AFTER_9B78EFA.md.

This is NOT unified bidirectional fusion. It takes ONE endpoint's trained
Gaussians, the free Phase-1 trajectory already computed for that endpoint, and
the scene's real 60-frame 8-camera RGB-D files, and runs the unchanged
`UnifiedFreeIEKS` on them. Every run is labelled `SINGLE_SOURCE` in its
manifest for exactly that reason.

What it does NOT do, by construction:

  * retrain anything, touch the static models, or write into
    `bidirectional/forward`, `bidirectional/backward`, either `*_rigid` tree or
    any previous output. The destination is a new subtree and the tool refuses
    to start if it already holds a finished run it did not write;
  * read a `predicted_support.json`, a label, a mask or any ground truth;
  * assume the whole 8-camera 60-frame window fits in 8 GiB. The window and
    the camera count are arguments, the ladder is meant to be walked, and an
    out-of-memory failure is reported as OOM with the rung it happened on
    rather than hidden by a smaller default.

The warm start is honest about what Phase 1 actually saved. `phase1_centers/`
holds centers only -- no rotations, no rates -- so:

    mu(t)    the stored chronological centers, verbatim
    R(t)     the endpoint model's own rotations, constant
    v(t)     central differences of mu in PHYSICAL time
    omega(t) zero

For a backward Phase 1 the centers on disk are already in original
chronological order (`trajectory_order: original_chronological`), and the
finite difference is therefore taken in physical time with a positive dt, so
no rate sign flip is needed or applied here. The manifest records which.

    # the smallest rung, for a first look
    python tools/run_unified_phase1_single_source.py \
      --endpoint final --frames 4 --cameras 1 --dry-run

    # then the ladder
    ... --frames 4 --cameras 1
    ... --frames 8 --cameras 2
    ... --frames 16 --cameras 4
    ... --frames 60 --cameras 8
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

SINGLE_SOURCE = "SINGLE_SOURCE"
ENDPOINT_DIRECTION = {"initial": "forward", "final": "backward"}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scene", default="washer_103776")
    ap.add_argument("--endpoint", choices=("initial", "final"), required=True)
    ap.add_argument("--model", default=None,
                    help="trained endpoint model dir; default "
                         "<out-root>/static_bidi/<endpoint>")
    ap.add_argument("--source", required=True, help="dataset root for the scene")
    ap.add_argument("--out-root", required=True,
                    help="the scene's result root, e.g. .../ours/washer_103776")
    ap.add_argument("--phase1-centers", default=None,
                    help="default <out-root>/bidirectional/<direction>/phase1_centers")
    ap.add_argument("--out", default=None,
                    help="default <out-root>/unified_phase1_tests/single_source_<endpoint>")
    ap.add_argument("--frame-start", type=int, default=0)
    ap.add_argument("--frames", type=int, default=4)
    ap.add_argument("--cameras", type=int, default=1)
    ap.add_argument("--max-gaussians", type=int, default=0,
                    help="0 keeps the whole trained model")
    ap.add_argument("--dt", type=float, default=1.0 / 30.0)
    ap.add_argument("--max-outer", type=int, default=8)
    ap.add_argument("--pcg-iters", type=int, default=40)
    ap.add_argument("--pcg-tol", type=float, default=1e-6)
    ap.add_argument("--damping", type=float, default=1e-3)
    ap.add_argument("--iteration", type=int, default=30000)
    ap.add_argument("--source-pose-policy",
                    choices=("fixed_conditional", "optimizable_reference"),
                    default="fixed_conditional")
    ap.add_argument("--frames-on-cpu", action="store_true")
    ap.add_argument("--diagonal-preconditioner", action="store_true",
                    help="precondition on the diagonal instead of the 12x12 "
                         "blocks: 142 MiB instead of 1.70 GiB held three times "
                         "over at 60 frames. PCG converges to the same point "
                         "under any SPD preconditioner; this one costs "
                         "iterations, not accuracy")
    ap.add_argument("--process-factors-on-cpu", action="store_true",
                    help="keep the dense [N,12,12] process Jacobians in host "
                         "memory and move one frame pair at a time; 3.3 GiB "
                         "of device memory at 60 frames and 24,582 Gaussians")
    ap.add_argument("--stream-frames", action="store_true",
                    help="keep ONE frame's rasterizer cache resident and "
                         "rebuild it on demand. The resident path needs about "
                         "85 MiB per (frame, camera) factor on this scene, so "
                         "the 480 factors of a 60-frame 8-camera window do not "
                         "fit in 8 GiB; streaming re-renders instead, with the "
                         "same arithmetic (tests/test_unified_streaming_window.py)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--dry-run", action="store_true",
                    help="load, seal the inventory, report sizes, do not solve")
    args = ap.parse_args()

    out_root = Path(args.out_root)
    direction = ENDPOINT_DIRECTION[args.endpoint]
    model = Path(args.model or out_root / "static_bidi" / args.endpoint)
    centers_dir = Path(args.phase1_centers
                       or out_root / "bidirectional" / direction / "phase1_centers")
    out = Path(args.out or out_root / "unified_phase1_tests"
               / f"single_source_{args.endpoint}")

    print(f"=== {SINGLE_SOURCE} unified IEKS: {args.scene} / {args.endpoint} ===")
    for label, path in (("model", model), ("source", Path(args.source)),
                        ("phase1 centers", centers_dir)):
        if not path.exists():
            raise SystemExit(f"{label} not found: {path}")
        print(f"  {label:<16} {path}")

    # ---- refuse to write over anything --------------------------------------
    from bidirectional_iekf.unified_coherence import reject_stale_support
    reject_stale_support(out, what="the output directory")
    if (out / "run_status.json").is_file():
        raise SystemExit(
            f"{out} already holds a finished run. Choose another --out rather "
            "than overwriting a result that may be someone's reference.")
    forbidden = [out_root / "bidirectional" / d for d in
                 ("forward", "backward", "forward_rigid", "backward_rigid")]
    forbidden += [out_root / "static_bidi" / e for e in ("initial", "final")]
    resolved = out.resolve()
    for path in forbidden:
        if path.exists() and (resolved == path.resolve()
                              or path.resolve() in resolved.parents):
            raise SystemExit(
                f"--out {out} lies inside {path}, which holds previous "
                "results. This tool never writes into an existing stage.")

    ply = model / "point_cloud" / f"iteration_{args.iteration}" / "point_cloud.ply"
    if not ply.is_file():
        raise SystemExit(f"no trained PLY at {ply}")

    # ---- the stored Phase-1 trajectory -------------------------------------
    manifest_path = centers_dir / "trajectory_manifest.json"
    if not manifest_path.is_file():
        raise SystemExit(
            f"{centers_dir} has no trajectory_manifest.json, so the direction "
            "and the ordering of its centers cannot be established. Guessing "
            "either is how a backward trajectory ends up consumed as a "
            "forward one with every array the right shape.")
    p1 = json.loads(manifest_path.read_text())
    if p1.get("trajectory_order") != "original_chronological":
        raise SystemExit(
            f"{centers_dir} records trajectory_order="
            f"{p1.get('trajectory_order')!r}; this tool requires original "
            "chronological order.")
    if p1.get("canonical_endpoint") != args.endpoint:
        raise SystemExit(
            f"--endpoint {args.endpoint} but {centers_dir} was produced from "
            f"the {p1.get('canonical_endpoint')!r} endpoint.")
    print(f"  phase1 manifest  direction={p1['direction']} "
          f"endpoint={p1['canonical_endpoint']} frames={p1['num_frames']} "
          f"gaussians={p1['num_gaussians']}")

    frame_ids = list(range(args.frame_start, args.frame_start + args.frames))
    center_files = [centers_dir / f"centers_frame_{f:04d}.npy" for f in frame_ids]
    missing = [f for f in center_files if not f.is_file()]
    if missing:
        raise SystemExit(f"missing Phase-1 centers: {[str(m) for m in missing]}")

    # ---- the scene ---------------------------------------------------------
    from estimator.covariance import invert_blocks, make_process_noise
    from estimator.sparse_iekf import IEKFConfig
    from estimator.state import GaussianMotionState
    from tools.run_iekf_scene import (dataset_rgbd_frames, load_scene,
                                      state_from_gaussians)

    cfg = IEKFConfig()
    _dataset, _pipe, gaussians, scene = load_scene(str(model), str(args.source))
    cams_all = scene.getTrainCameras()
    ncam = min(args.cameras, len(cams_all))
    truth = state_from_gaussians(gaussians)
    N_full = int(truth.mu.shape[0])
    keep = None
    if args.max_gaussians and N_full > args.max_gaussians:
        import dataclasses
        g = torch.Generator().manual_seed(args.seed)
        keep = torch.randperm(N_full, generator=g)[:args.max_gaussians].sort().values
        truth = dataclasses.replace(
            truth, mu=truth.mu[keep], R=truth.R[keep], v=truth.v[keep],
            omega=truth.omega[keep], colors=truth.colors[keep],
            opacity=truth.opacity[keep], scales=truth.scales[keep])
    N = int(truth.mu.shape[0])
    print(f"  {N} of {N_full} Gaussians, {ncam} of {len(cams_all)} cameras, "
          f"frames {frame_ids[0]}..{frame_ids[-1]}")

    ns = argparse.Namespace(
        sequence_start=args.frame_start, sequence_length=args.frames,
        all_frames=False, observation_source="dataset_rgbd", cameras=ncam,
        all_cameras=False, window_length=0, dt=args.dt,
        frames_on_cpu=bool(args.frames_on_cpu),
        frames_on_gpu=not args.frames_on_cpu, max_gaussians=args.max_gaussians,
        seed=args.seed)
    # ALWAYS chronological: the unified window carries one physical time
    # coordinate whichever endpoint supplied the geometry.
    frames_raw, provenance = dataset_rgbd_frames(
        cams_all[:ncam], cfg, ns, truth, Path(args.source), direction="forward")
    cameras = frames_raw[0][1]
    observations = [per_cam for per_cam, _specs, _dt in frames_raw]
    dataset_frames = [f["dataset_frame_index"] for f in provenance["frames"]]
    if dataset_frames != frame_ids:
        raise SystemExit(
            f"the loader returned dataset frames {dataset_frames} but the "
            f"Phase-1 centers were read for {frame_ids}; the two must be the "
            "same physical frames or the warm start belongs to another time.")

    # ---- the warm start ----------------------------------------------------
    centers = []
    for f in center_files:
        c = torch.from_numpy(np.load(f)).to(truth.mu.device, torch.float32)
        if c.shape[0] != N_full:
            raise SystemExit(f"{f} holds {c.shape[0]} rows but the model has "
                             f"{N_full}")
        centers.append(c if keep is None else c[keep.to(c.device)])
    dt = float(args.dt)
    T = len(centers)
    velocities = []
    for t in range(T):
        if T == 1:
            velocities.append(torch.zeros_like(centers[0]))
        elif t == 0:
            velocities.append((centers[1] - centers[0]) / dt)
        elif t == T - 1:
            velocities.append((centers[-1] - centers[-2]) / dt)
        else:
            velocities.append((centers[t + 1] - centers[t - 1]) / (2.0 * dt))
    warm_states = [GaussianMotionState(
        mu=centers[t].contiguous(), R=truth.R.clone(), v=velocities[t].contiguous(),
        omega=torch.zeros_like(centers[t]), colors=truth.colors,
        opacity=truth.opacity, scales=truth.scales) for t in range(T)]

    from bidirectional_iekf.ieks import (IEKSConfig, Trajectory,
                                         UnifiedFreeIEKS, save_run)
    from bidirectional_iekf.observations import FreeRGBDObservation

    obs = FreeRGBDObservation(
        cameras, observations, frame_ids=frame_ids,
        sigma_rgb=cfg.rgb_sigma,
        depth_sigma_model={"kind": cfg.depth_sigma_model,
                           "a": cfg.depth_sigma_a, "b": cfg.depth_sigma_b},
        robust=False, use_rgb=cfg.use_rgb, use_depth=cfg.use_depth,
        eps_depth=cfg.eps_depth, min_alpha_for_depth=cfg.min_alpha_for_depth)
    inventory = obs.build_inventory()
    print(f"  sealed inventory: {inventory['num_factors']} factors, unique by "
          f"{inventory['unique_by']}")

    Q = make_process_noise(N, dt, cfg.process_noise_pos, cfg.process_noise_rot,
                           cfg.process_noise_vel, cfg.process_noise_omega,
                           device=truth.mu.device, dtype=torch.float64)
    Q_inv, _ = invert_blocks(Q.double())

    # The source geometry's own frame: the initial model is anchored at
    # original frame 0 and the final model at the last one. Under
    # 'fixed_conditional' those pose columns are ELIMINATED, not given a large
    # precision, and every reported uncertainty is conditional on that.
    anchor_original = 0 if args.endpoint == "initial" else p1["num_frames"] - 1
    anchor_frames = tuple(i for i, f in enumerate(frame_ids) if f == anchor_original)
    if args.source_pose_policy == "fixed_conditional" and not anchor_frames:
        print(f"  NOTE: the {args.endpoint} anchor is original frame "
              f"{anchor_original}, which is outside this window "
              f"[{frame_ids[0]}, {frame_ids[-1]}]. There is no source-time "
              "pose to eliminate here, so the policy is recorded as "
              "'optimizable_reference' for this rung.")
    policy = (args.source_pose_policy if anchor_frames
              else "optimizable_reference")

    ieks = UnifiedFreeIEKS(obs, Q_inv, config=IEKSConfig(
        max_outer=args.max_outer, damping=args.damping,
        pcg_max_iters=args.pcg_iters, pcg_tol=args.pcg_tol,
        source_pose_policy=policy, anchor_frames=anchor_frames,
        stream_frames=bool(args.stream_frames),
        process_factors_on_cpu=bool(args.process_factors_on_cpu),
        diagonal_preconditioner=bool(args.diagonal_preconditioner),
        verbose=True))
    warm = Trajectory(states=warm_states, frame_ids=frame_ids, dt=dt)

    torch.cuda.reset_peak_memory_stats()
    t0 = time.perf_counter()
    start_energy = ieks.evaluate(warm, with_support=True)
    load_peak = torch.cuda.max_memory_allocated() / 2 ** 20
    print(f"  E at the warm start: {start_energy['E_total']:.6e} "
          f"(rgb {start_energy['E_rgb']:.4e}, depth {start_energy['E_depth']:.4e}, "
          f"proc {start_energy['E_process']:.4e}); "
          f"{start_energy['depth_support']['total']} covered depth pixels; "
          f"peak {load_peak:.0f} MiB after one full evaluation")

    out.mkdir(parents=True, exist_ok=True)
    base = {
        "label": SINGLE_SOURCE,
        "not_a_unified_fusion": ("one endpoint's geometry against the real "
                                 "images; the two-source merge is a separate "
                                 "gate and is not claimed here"),
        "scene": args.scene, "endpoint": args.endpoint,
        "phase1_direction": p1["direction"],
        "model_path": str(model), "ply_path": str(ply),
        "ply_sha256": sha256_file(ply),
        "source_path": str(args.source),
        "phase1_centers": str(centers_dir),
        "original_frame_ids": frame_ids,
        "num_cameras": ncam, "camera_names": provenance["cameras"],
        "num_gaussians": N, "num_gaussians_in_model": N_full,
        "max_gaussians": args.max_gaussians, "seed": args.seed,
        "dt_s": dt,
        "sensor_model": {"sigma_rgb": cfg.rgb_sigma,
                         "depth_sigma_model": cfg.depth_sigma_model,
                         "depth_sigma_a": cfg.depth_sigma_a,
                         "depth_sigma_b": cfg.depth_sigma_b,
                         "min_alpha_for_depth": cfg.min_alpha_for_depth,
                         "from": "estimator.sparse_iekf.IEKFConfig defaults"},
        "process_noise": {"pos": cfg.process_noise_pos,
                          "rot": cfg.process_noise_rot,
                          "vel": cfg.process_noise_vel,
                          "omega": cfg.process_noise_omega,
                          "built_by": "estimator.covariance.make_process_noise"},
        "stream_frames": bool(args.stream_frames),
        "process_factors_on_cpu": bool(args.process_factors_on_cpu),
        "diagonal_preconditioner": bool(args.diagonal_preconditioner),
        "source_pose_policy": policy,
        "anchor_original_frame": anchor_original,
        "anchor_window_indices": list(anchor_frames),
        "warm_start": {
            "mu": "phase1_centers, chronological, verbatim",
            "R": "the endpoint model's own rotations, constant",
            "v": "central differences of mu in physical time",
            "omega": "zero",
            "note": ("Phase 1 persisted centers only. No rate sign flip is "
                     "applied: the stored centers are already in original "
                     "chronological order for both directions, so the "
                     "difference quotient is already a physical-time rate."),
        },
        "factor_inventory": inventory,
        "frame_provenance": provenance,
        "warm_start_energy": {k: v for k, v in start_energy.items()
                              if k != "depth_support"},
        "depth_support_at_warm_start": start_energy["depth_support"]["total"],
        "peak_cuda_mib_after_one_evaluation": load_peak,
    }
    if args.dry_run:
        base["status"] = "DRY_RUN"
        (out / "single_source_manifest.json").write_text(
            json.dumps(base, indent=1, default=str))
        print(f"  DRY_RUN; manifest written to "
              f"{out / 'single_source_manifest.json'}")
        return 0

    try:
        result = ieks.run(warm)
        failure = None
    except torch.cuda.OutOfMemoryError as exc:
        failure = {"status": "OOM", "rung": {"frames": args.frames,
                                             "cameras": ncam,
                                             "gaussians": N},
                   "error": str(exc)[:400]}
        result = None
    wall = time.perf_counter() - t0
    peak = torch.cuda.max_memory_allocated() / 2 ** 20

    if failure is not None:
        base.update(failure)
        base["wall_time_s"] = wall
        base["peak_cuda_mib"] = peak
        (out / "single_source_manifest.json").write_text(
            json.dumps(base, indent=1, default=str))
        print(f"  OOM at frames={args.frames} cameras={ncam} gaussians={N}; "
              f"peak {peak:.0f} MiB. Reported, not retried smaller.")
        return 2

    save_run(out, result)
    moved = max(float((result["trajectory"].states[t].mu
                       - warm.states[t].mu).norm(dim=1).max())
                for t in range(T))
    base.update({
        "status": result["status"],
        "energy": result["energy"],
        "wall_time_s": wall,
        "peak_cuda_mib": peak,
        "outer_iterations": len(result["trace"]),
        "accepted_steps": sum(1 for r in result["trace"] if r["accepted"]),
        "final_pcg": {k: result["trace"][-1].get(k) for k in
                      ("pcg_iterations", "pcg_residual", "pcg_status",
                       "grad_norm", "step_norm", "damping")},
        "max_center_change_from_warm_start": moved,
    })
    (out / "single_source_manifest.json").write_text(
        json.dumps(base, indent=1, default=str))
    print(f"\n  status {result['status']}  "
          f"E {start_energy['E_total']:.6e} -> {result['energy']['E_total']:.6e}  "
          f"({wall:.1f} s, peak {peak:.0f} MiB)")
    print(f"  largest center change from the warm start: {moved:.4e}")
    print(f"  artifacts: {out}")
    return 0 if result["status"] == "CONVERGED" else 3


if __name__ == "__main__":
    raise SystemExit(main())
