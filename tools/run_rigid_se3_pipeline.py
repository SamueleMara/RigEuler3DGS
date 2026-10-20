"""The production pipeline: rigid SE(3) Phase 2, then post-hoc joint reduction.

CODEX_IMPLEMENT_RIGID_SE3_PHASE2_AND_POSTHOC_JOINT_REDUCTION.md sections
44 (the default path), 45 (the output tree), 19/21/22 (the debug) and 39 (GT).

    Phase 1 free motion
      -> coherence MOVING_RIGID support
      -> rigid SE(3) RGB-D Phase 2
      -> joint reduction P + R
      -> pose-space model comparison

No axis, direction, centre or joint type is a parameter of the RGB-D
optimiser anywhere in this file. The renderer is asked one question only:
"what rigid pose best explains the moving part at this frame?"

    python tools/run_rigid_se3_pipeline.py --scene stapler_103111 \
        --frames 60 --cameras 8 --max-gaussians 0 --frames-on-cpu
"""

from __future__ import annotations

import argparse
import csv
import dataclasses
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from utils.progress import pbar, stage  # noqa: E402

MOTION_MODELS = ("rigid-se3", "legacy-screw", "legacy-direct-p",
                 "legacy-direct-r")

STATE_TRACE = [
    "frame", "time",
    "tx", "ty", "tz",
    "rotvec_x", "rotvec_y", "rotvec_z", "rotation_angle",
    "vx", "vy", "vz", "wx", "wy", "wz",
    "rgb_energy", "depth_energy", "total_energy",
    "accepted_steps", "rejected_steps", "trust_radius", "pcg_iterations",
    "pose_information_eigs", "velocity_information_eigs",
]

KABSCH_CMP = ["frame", "d_rotation_rad", "d_rotation_deg", "d_translation",
              "kabsch_angle", "rgbd_angle", "kabsch_translation",
              "rgbd_translation", "kabsch_mean_residual"]


def _csv(path: Path, header, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(header)
        w.writerows(rows)


def _json(path: Path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=1, default=str))


def _rotvec(R):
    from estimator.manifold import so3_log
    return so3_log(torch.as_tensor(R).double().reshape(-1, 3, 3))


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scene", required=True)
    ap.add_argument("--model", default=None)
    ap.add_argument("--source", default=None)
    ap.add_argument("--frames", type=int, default=60)
    ap.add_argument("--cameras", type=int, default=8)
    ap.add_argument("--camera", type=int, default=0,
                    help="section 22: the camera rendered for EVERY frame")
    ap.add_argument("--max-gaussians", type=int, default=0,
                    help="0 (the default) uses EVERY Gaussian")
    ap.add_argument("--bootstrap-iters", type=int, default=12)
    ap.add_argument("--rigid-iters", type=int, default=8,
                    help="nonlinear IEKF iterations per frame in Phase 2")
    ap.add_argument("--rigid-cg-iters", type=int, default=40)
    ap.add_argument("--sigma-vdot", type=float, default=1.0)
    ap.add_argument("--sigma-wdot", type=float, default=1.0)
    ap.add_argument("--phase2-motion-model", choices=MOTION_MODELS,
                    default="rigid-se3",
                    help="section 43: 'rigid-se3' is production. The legacy "
                         "direct-axis estimators remain reachable for "
                         "ablation but no longer define joint geometry.")
    ap.add_argument("--phase2-support-source",
                    choices=("coherence", "current"), default="coherence",
                    help="section 1: coherence MOVING_RIGID is the production "
                         "default. 'current' is the Otsu ablation. There is "
                         "no silent fallback between them.")
    ap.add_argument("--coherence-support", default=None)
    ap.add_argument("--phase1-centers", default=None,
                    help="reuse an existing phase1/centers directory")
    ap.add_argument("--supernode-size", type=int, default=64)
    ap.add_argument("--max-debug-points", type=int, default=6000)
    ap.add_argument("--no-renders", action="store_true")
    ap.add_argument("--no-refine", action="store_true",
                    help="skip the section 32 pose-space refinement")
    ap.add_argument("--frames-on-cpu", action="store_true")
    from estimator.visibility_constraint import add_cli as _vis_cli
    _vis_cli(ap)
    from estimator.masked_observation import add_cli as _mask_obs_cli
    _mask_obs_cli(ap)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--direction", choices=("forward", "backward"),
                    default="forward",
                    help="backward solves against the FINAL endpoint model as "
                         "canonical geometry, warm starting from identity at "
                         "the last original frame")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    if args.phase2_motion_model != "rigid-se3":
        raise SystemExit(
            f"--phase2-motion-model {args.phase2_motion_model} is a legacy "
            "ablation and lives in tools/run_complete_joint_debug.py "
            "(--phase2-support-source selects the support there). This "
            "pipeline implements the section 44 production path only.")

    model, source = args.model, args.source
    if model is None or source is None:
        from tools.run_all_datasets import discover
        for rec in discover(REPO):
            if rec["scene"] == args.scene:
                model = model or str(rec["model"])
                source = source or str(rec["source"])
                break
    if model is None:
        raise SystemExit(f"could not resolve the model for {args.scene}")

    out = Path(args.out or f"output/rigid_se3_{args.scene}")
    out.mkdir(parents=True, exist_ok=True)
    t_start = time.perf_counter()
    print(f"=== RIGID SE(3) PIPELINE: {args.scene} ===")
    print(f"  model  {model}\n  source {source}\n  out    {out}")

    from estimator.model_error import estimate_model_error
    from estimator.phase2_shared import (effective_precision, render_state,
                                         weighted_energy)
    from estimator.renderer_bridge import build_linearization
    from estimator.rigid_part_runtime import (RigidIEKFConfig, RigidState,
                                              kabsch_initialize,
                                              run_rigid_window)
    from estimator.sparse_iekf import IEKFConfig
    from tools.run_iekf_scene import (camera_specs, dataset_rgbd_frames,
                                      load_scene, state_from_gaussians)

    # ---- scene ----------------------------------------------------------
    dataset, pipe, gaussians, sc = load_scene(model, source, camera_device="cpu")
    cams_all = sc.getTrainCameras()
    ncam = min(args.cameras, len(cams_all))
    cameras = camera_specs(cams_all, ncam)
    truth = state_from_gaussians(gaussians)
    if args.max_gaussians and truth.mu.shape[0] > args.max_gaussians:
        g = torch.Generator().manual_seed(args.seed)
        keep = torch.randperm(truth.mu.shape[0],
                              generator=g)[:args.max_gaussians].sort().values
        truth = dataclasses.replace(
            truth, mu=truth.mu[keep], R=truth.R[keep], v=truth.v[keep],
            omega=truth.omega[keep], colors=truth.colors[keep],
            opacity=truth.opacity[keep], scales=truth.scales[keep])
    cfg = IEKFConfig()
    ns = argparse.Namespace(
        sequence_start=0, sequence_length=args.frames, all_frames=False,
        observation_source="dataset_rgbd", cameras=ncam,
        bootstrap_iters=args.bootstrap_iters,
        supernode_size=args.supernode_size, seed=args.seed,
        frames_on_cpu=bool(args.frames_on_cpu),
        frames_on_gpu=not args.frames_on_cpu,
        depth_sigma=None, joint_direct_rgbd=False, joint_direct_gaussian=False,
        window_length=0, all_cameras=False, dt=1.0 / 30.0,
        max_gaussians=args.max_gaussians, no_model_error=False,
        model_error_beta=1.0, model_error_blur=1)
    frames_raw, frame_ids = dataset_rgbd_frames(
        cams_all[:ncam], cfg, ns, truth, Path(source),
        direction=args.direction)
    from utils.sequence_direction import (context_from_frame_ids,
                                          verify_endpoint_provenance)
    ctx = context_from_frame_ids(frame_ids)
    endpoint_evidence = verify_endpoint_provenance(
        args.direction, model, total_frames=len(frames_raw), strict=True)
    print(f"  direction {ctx.direction}, canonical endpoint "
          f"{endpoint_evidence['observed_endpoint'] or 'unknown'} "
          f"(frame {endpoint_evidence['endpoint_original_frame']})")
    obs_all = [[(o[0], o[1], o[2] if len(o) > 2 else None) if o is not None
                else None for o in ob] for (ob, _s, _d) in frames_raw]
    T = len(obs_all)
    # Canonical geometry comes from the loaded MODEL, which for a backward
    # run is the final endpoint. It is never taken from a trajectory frame.
    mu0 = truth.mu.double()
    R0 = truth.R.double()
    cam_ids = list(range(ncam))
    print(f"  {mu0.shape[0]} Gaussians, {ncam} cameras, {T} frames")

    dev = mu0.device
    # Masked observation mode (estimator/masked_observation.py): the RGB target is
    # the masked composite on the model's background, the scene is rendered on the
    # same background, depth counts on the object only, and the model-error map is
    # estimated from THAT residual at the canonical frame.
    from estimator.masked_observation import from_args as _masked_from_args
    from estimator.phase2_shared import (PreparedFrames, effective_precision_masked,
                                         weighted_energy_masked)
    mseq = _masked_from_args(args, source, frames_raw, frame_ids, cfg)
    bg = None if mseq is None else mseq.model.background_tensor(dev)
    prepared = None if mseq is None else PreparedFrames(mseq, obs_all)
    if mseq is None:
        ref_obs = [(o[0].to(dev), o[1].to(dev),
                    o[2].to(dev) if o[2] is not None else None)
                   for o in obs_all[0] if o is not None]
    else:
        ref_obs = [(o.target(bg), o.depth_observed, o.depth_valid)
                   for o in (ob.to(dev) for ob in prepared[0])]
    lin0 = build_linearization(truth, cameras, eps_depth=cfg.eps_depth, background=bg)
    model_error = estimate_model_error(lin0, ref_obs, reference_index=0,
                                       beta=1.0, blur=1)
    del lin0

    def prec_at(lin, t):
        if mseq is not None:
            return effective_precision_masked(lin, prepared[t], cam_ids, cfg, model_error)
        return effective_precision(lin, obs_all[t], cam_ids, cfg, model_error)

    def energies(mu, Rg, t):
        lin = render_state(truth, mu, Rg, cameras, cfg, build_linearization, background=bg)
        if mseq is not None:
            rows = weighted_energy_masked(lin, prepared[t], cam_ids, prec_at(lin, t), mseq.model)
        else:
            rows = weighted_energy(lin, obs_all[t], cam_ids, prec_at(lin, t))
        del lin
        return rows

    # =====================================================================
    # 1. PHASE 1 -- free per-Gaussian motion
    # =====================================================================
    print("\n  [1/7] Phase 1: free per-Gaussian motion")
    stage(f"[1/7] {args.scene}: Phase 1 over {T} frames x {ncam} cameras")
    p1dir = out / "phase1"
    if args.phase1_centers:
        d = Path(args.phase1_centers)
        files = sorted(d.glob("centers_frame_*.npy"))[:T]
        if len(files) < T:
            raise SystemExit(f"{d} has {len(files)} center frames, need {T}")
        positions = [torch.from_numpy(np.load(f)).double() for f in files]
        # The centers are stored by ORIGINAL frame index, so sorted() gives
        # chronological order. The rigid solve walks in PROCESSING order --
        # for a backward run that is 59..0, which is what lets it warm start
        # from identity at the canonical frame instead of meeting the whole
        # sequence displacement on its first step. Reorder to match the
        # observations, which came from the loader already in processing
        # order.
        if ctx.is_backward:
            positions = list(positions)[::-1]
        if positions[0].shape[0] != mu0.shape[0]:
            raise SystemExit(
                f"{d} holds {positions[0].shape[0]} Gaussians but this run "
                f"has {mu0.shape[0]}; re-run Phase 1 or match --max-gaussians")
        print(f"      reusing {T} Phase-1 center frames from {d}")
        p1_info = {"reused_from": str(d)}
    else:
        from tools.kpart_pipeline import bootstrap_positions
        window = [(ob, fid) for (ob, _s, _d), fid
                  in zip(frames_raw, frame_ids["frames"])]
        ns.masked_observation = mseq
        positions, p1_info = bootstrap_positions(truth, cameras, window, cfg,
                                                 None, ns, verbose=True)
    (p1dir / "centers").mkdir(parents=True, exist_ok=True)
    for t, p in pbar(list(enumerate(positions)), "phase1 centers", total=T):
        np.save(p1dir / "centers" / f"centers_frame_{t:04d}.npy",
                torch.as_tensor(p).double().cpu().numpy())
    _json(p1dir / "phase1_info.json",
          {k: v for k, v in (p1_info or {}).items() if not torch.is_tensor(v)})

    # =====================================================================
    # 2. COHERENCE MOVING_RIGID SUPPORT (sections 1, 2)
    # =====================================================================
    print("\n  [2/7] frozen static/dynamic support")
    members, ev = _load_support(args, mu0, positions, truth)
    frac = 100.0 * members.numel() / max(mu0.shape[0], 1)
    print(f"      [{ev['method']}] {int(members.numel())} of {mu0.shape[0]} "
          f"Gaussians ({frac:.1f}%)")
    stage(f"[2/7] {args.scene}: {int(members.numel())} moving Gaussians "
          f"({frac:.1f}%)")
    _json(out / "support_coherence" / "support.json", {
        **ev, "num_members": int(members.numel()),
        "members": members.tolist(),
        "frozen_before_phase2": True,
        "note": ("section 2: this index set is frozen for the whole of rigid "
                 "Phase 2. Static Gaussians are still rendered -- they carry "
                 "visibility, alpha compositing and occlusion -- but the "
                 "rigid state does not move them.")})

    # =====================================================================
    # 3. PHASE-1 KABSCH INITIALIZATION (sections 16, 17)
    # =====================================================================
    print("\n  [3/7] Phase-1 Kabsch rigid initialization")
    stage(f"[3/7] {args.scene}: Phase-1 Kabsch")
    init = kabsch_initialize(members, mu0.cpu(),
                             [torch.as_tensor(p).double().cpu()
                              for p in positions])
    kab = init["part"]
    kabsch_R, kabsch_p = kab.R.clone(), kab.p.clone()
    unexp = init["unexplained_fraction_final"]
    print(f"      max mean Kabsch residual {init['max_mean_residual']:.5f}"
          f"   ({init['weighting']})")
    print(f"      unexplained by ONE rigid motion, at the last frame: "
          f"{100 * unexp:.0f}% of the displacement")
    if unexp == unexp and unexp > 0.4:
        print(f"      NOTE: a single rigid motion leaves {100 * unexp:.0f}% of "
              f"this support's displacement unexplained. That is either a "
              f"support spanning several bodies or a noisy Phase 1 over it -- "
              f"gamma_1DOF below separates them. Phase 2 fits the best single "
              f"rigid trajectory either way.")

    # =====================================================================
    # 4. RIGID SE(3) RGB-D PHASE 2 (sections 13-15, 18)
    # =====================================================================
    print("\n  [4/7] rigid SE(3) RGB-D Phase 2")
    stage(f"[4/7] {args.scene}: rigid SE(3) Phase 2")
    rcfg = RigidIEKFConfig(max_iters=args.rigid_iters,
                           max_cg_iters=args.rigid_cg_iters,
                           sigma_vdot=args.sigma_vdot,
                           sigma_wdot=args.sigma_wdot)
    state = RigidState(init["part"], rcfg)

    def linearize(st, frame):
        mu, Rg = st.gaussians(frame, mu0, R0)
        return render_state(truth, mu, Rg, cameras, cfg, build_linearization, background=bg)

    ctr = {"t": 0}

    def _lin(st, frame):
        ctr["t"] = frame
        return linearize(st, frame)

    def precision_fn(lin):
        return prec_at(lin, ctr["t"])

    obs_pairs = [[(o[0], o[1]) if o is not None else None for o in ob]
                 for ob in obs_all]
    from estimator.visibility_constraint import from_args as _vis_from_args
    vis_seq = _vis_from_args(args, source, frames_raw, frame_ids)
    res = run_rigid_window(state, mu0, R0, obs_pairs if mseq is None else prepared, _lin, precision_fn,
                           cfg=rcfg, label="rigid", visibility=vis_seq,
                           observation_model=None if mseq is None else mseq.model)
    if mseq is not None:
        _json(out / "phase2_rigid" / "masked_observation.json", {
            **mseq.model.to_dict(), "mask_root": mseq.mask_root, "cameras": mseq.cameras,
            "rgb_precision": "1/(rgb_sigma^2 + model-error variance), fixed, no Student-t weight",
            "rgb_sigma": cfg.rgb_sigma, "coverage": mseq.counts(),
            "frames": [{"processing_index": k, "energy_before": r.energy_before,
                        "energy_after": r.energy_after, "iterations": r.iterations,
                        "accepted": r.accepted} for k, r in enumerate(res["records"])]})
    if vis_seq is not None:
        vrows = [{"processing_index": k, **{kk: vv for kk, vv in (r.observability.get("visibility") or {}).items()
                                            if kk != "iterations"}}
                 for k, r in enumerate(res["records"]) if r.observability.get("visibility")]
        _json(out / "phase2_rigid" / "visibility_constraint.json",
              {"policy": vis_seq.policy.to_dict(), "mask_root": getattr(vis_seq, "mask_root", None), "frames": vrows})
    part = res["part"]
    recs = res["records"]

    p2 = out / "phase2_rigid"
    p2.mkdir(parents=True, exist_ok=True)
    poses = np.stack([_se3_matrix(part.R[t], part.p[t]) for t in range(T)])
    # Section 13: the authoritative arrays are in ORIGINAL chronological
    # order, whatever order they were solved in. Everything downstream --
    # the reduction, the forward/backward comparison, the fusion -- indexes
    # by original frame, so a processing-order array here would silently run
    # one of the two directions backwards through every later stage.
    def _chrono(a):
        return a[::-1].copy() if ctx.is_backward else a

    if ctx.is_backward:
        np.save(p2 / "poses_processing_order.npy", poses)
    np.save(p2 / "poses.npy", _chrono(poses))
    np.save(p2 / "poses_original_order.npy", _chrono(poses))
    # Velocities are NOT reordered. They are per-step rates in processing
    # time, so a backward run's stored twist is a reverse-time quantity and
    # reversing the array would keep the wrong sign. The chronological twist
    # is RECOMPUTED from the restored poses (section E).
    if ctx.is_backward:
        from utils.bidirectional_time import chronological_twists
        chrono_xi = chronological_twists(
            torch.from_numpy(_chrono(poses)).double(), float(ns.dt))
        np.save(p2 / "chronological_twists.npy",
                chrono_xi.cpu().numpy())
    np.save(p2 / "velocities.npy",
            torch.cat([part.v, part.omega], dim=1).numpy())
    np.save(p2 / "pose_covariances.npy",
            _chrono(res["pose_covariances"].numpy()))
    np.save(p2 / "state_covariances.npy", res["covariances"].numpy())
    np.save(p2 / "phase1_kabsch_poses.npy",
            np.stack([_se3_matrix(kabsch_R[t], kabsch_p[t])
                      for t in range(T)]))

    # ---- section 19: the numerical debug --------------------------------
    rv = _rotvec(part.R)
    rows, fc = [], []
    for t in pbar(range(T), "rigid trace", total=T):
        mu, Rg = part_gaussians(part, t, mu0, R0)
        er = energies(mu, Rg, t)
        rec = recs[t]
        o = rec.observability or {}
        pe = o.get("geometry_information_eigenvalues") or [float("nan")]
        ve = o.get("measurement_information_eigenvalues") or [float("nan")]
        rows.append([
            t, float(t), *part.p[t].tolist(), *rv[t].tolist(),
            float(rv[t].norm()), *part.v[t].tolist(), *part.omega[t].tolist(),
            sum(r["rgb"] for r in er), sum(r["depth"] for r in er),
            sum(r["total"] for r in er),
            rec.accepted, max(rec.iterations - rec.accepted, 0), rec.radius,
            rec.iterations,
            " ".join(f"{x:.6e}" for x in pe),
            " ".join(f"{x:.6e}" for x in ve[6:12]) if len(ve) >= 12 else ""])
        for r in er:
            fc.append([t, r["camera"], r["rgb"], r["depth"], r["total"]])
    _csv(p2 / "state_trace.csv", STATE_TRACE, rows)
    _csv(p2 / "per_frame_camera_energy.csv",
         ["frame", "camera", "rgb_energy", "depth_energy", "total_energy"], fc)

    # ---- section 20: the exact-rigidity gate, on the real support -------
    gate = _rigidity_gate(part, mu0, R0, T)
    _json(p2 / "exact_rigidity_gate.json", gate)
    print(f"      exact rigidity: max |d(t)-d(0)| = "
          f"{gate['max_pairwise_distance_drift']:.3e}"
          f"   orientation defect {gate['max_orientation_defect']:.3e}")

    # ---- section 21: Kabsch vs RGB-D ------------------------------------
    cmp_rows, dR_all, dp_all = [], [], []
    for t in range(T):
        dR = float(_rotvec(part.R[t] @ kabsch_R[t].T)[0].norm())
        dp = float((part.p[t] - kabsch_p[t]).norm())
        dR_all.append(dR)
        dp_all.append(dp)
        cmp_rows.append([t, dR, float(np.degrees(dR)), dp,
                         float(_rotvec(kabsch_R[t])[0].norm()),
                         float(rv[t].norm()),
                         float(kabsch_p[t].norm()),
                         float(part.p[t].norm()),
                         init["mean_residual_per_frame"][t]])
    _csv(p2 / "phase1_kabsch_vs_rgbd.csv", KABSCH_CMP, cmp_rows)
    kcmp = {"max_d_rotation_rad": max(dR_all), "max_d_translation": max(dp_all),
            "mean_d_rotation_rad": sum(dR_all) / max(len(dR_all), 1),
            "mean_d_translation": sum(dp_all) / max(len(dp_all), 1),
            "note": ("section 21: a large divergence from a highly coherent "
                     "Phase-1 rigid trajectory is a reason to inspect the "
                     "RGB-D objective BEFORE trusting any joint reduction")}
    _json(p2 / "phase1_kabsch_vs_rgbd.json", kcmp)
    print(f"      vs Phase-1 Kabsch: max d_rot {max(dR_all):.4f} rad, "
          f"max d_p {max(dp_all):.4f}")

    # =====================================================================
    # 5. JOINT REDUCTION (sections 24-35)
    # =====================================================================
    print("\n  [5/7] joint reduction (no renderer)")
    stage(f"[5/7] {args.scene}: joint reduction")
    from joint_reduction import reduce_trajectory
    red = reduce_trajectory(part.R, part.p, mu0.cpu(), members.cpu(),
                            P_tau=res["pose_covariances"],
                            refine=not args.no_refine)
    _write_reduction(out / "joint_reduction", red, T)
    od = red["one_dof"]
    print(f"      gamma_1DOF {od['gamma_1dof']:.4f}"
          f"   singular values {[f'{s:.3e}' for s in od['singular_values'][:3]]}")
    print(f"      E_P {red['prismatic']['pose_residual']['weighted_energy']:.4e}"
          f"   E_R {red['revolute']['pose_residual']['weighted_energy']:.4e}")
    print(f"      R axis {[round(x, 4) for x in red['revolute']['axis']]}"
          f"  point {[round(x, 4) for x in red['revolute']['axis_point']]}"
          f"  cond {red['revolute']['finite_axis_solver']['condition_number']:.2e}")

    # =====================================================================
    # 6. GT EVALUATION (section 39) -- strictly after everything above
    # =====================================================================
    print("\n  [6/7] GT joint evaluation")
    from joint_reduction.gt_evaluation import evaluate, load_joint_gt
    gt = load_joint_gt(source)
    ev_gt = evaluate(red, gt, frame_ids=list(range(T)))
    _json(out / "joint_reduction" / "gt_evaluation.json", ev_gt)
    if ev_gt.get("available"):
        print(f"      GT type {ev_gt['gt_joint_type']}"
              f"   P axis err {ev_gt['prismatic']['axis_angle_deg']:.2f} deg"
              f"   R axis err {ev_gt['revolute']['axis_angle_deg']:.2f} deg"
              f"   R line err {ev_gt['revolute']['axis_line_error']['distance']:.4f}")
    else:
        print(f"      {ev_gt.get('reason')}")

    # =====================================================================
    # 7. DEBUG (sections 22, 23, 40, 41)
    # =====================================================================
    print("\n  [7/7] debug renders and HTML")
    stage(f"[7/7] {args.scene}: debug")
    from debug.rigid_debug import (write_render_html, write_rigid_phase2_3d,
                                   write_rigid_to_joint_3d, write_timelines)
    html = out / "html"
    html.mkdir(parents=True, exist_ok=True)
    renders = None
    if not args.no_renders:
        renders = _render_camera(args, truth, cameras, cfg, obs_all, mu0, R0,
                                 part, kabsch_R, kabsch_p, positions, members,
                                 p2 / "renders", build_linearization,
                                 render_state)
    a = write_rigid_phase2_3d(
        html / "rigid_phase2_debug.html", mu0.cpu(), members.cpu(), positions,
        part, kabsch_R, kabsch_p, cameras,
        max_points=args.max_debug_points,
        title=f"{args.scene} - rigid SE(3) Phase 2",
        unexplained=init["unexplained_fraction_final"])
    b = write_rigid_to_joint_3d(
        html / "rigid_to_joint_debug.html", mu0.cpu(), members.cpu(), positions,
        part, kabsch_R, kabsch_p, red, ev_gt,
        max_points=args.max_debug_points,
        unexplained=init["unexplained_fraction_final"],
        title=f"{args.scene} - rigid reconstruction and joint reduction")
    c = write_timelines(html / "rigid_timelines.html", part, red, recs,
                        cmp_rows, title=f"{args.scene} - timelines")
    d = None
    if renders:
        d = write_render_html(html / "rigid_renders.html", renders,
                              title=f"{args.scene} - camera {renders['camera']}")

    summary = {
        "scene": args.scene, "model": str(model), "source": str(source),
        "frames": T, "cameras": ncam, "gaussians": int(mu0.shape[0]),
        "motion_model": args.phase2_motion_model,
        "support": {"method": ev["method"],
                    "num_members": int(members.numel()),
                    "fraction": frac / 100.0},
        "initialization": {k: v for k, v in init.items() if k != "part"},
        "support_rigidity": {
            "max_mean_kabsch_residual": init["max_mean_residual"],
            "part_scale_L": init["part_scale_L"],
            "unexplained_fraction_final": init["unexplained_fraction_final"],
            "max_unexplained_fraction": init["max_unexplained_fraction"],
            "note": init["unexplained_fraction_note"]},
        "exact_rigidity_gate": gate,
        "phase1_kabsch_vs_rgbd": kcmp,
        "one_dof": od,
        "prismatic": _geometry_only(red["prismatic"]),
        "revolute": _geometry_only(red["revolute"]),
        "gt_evaluation": ev_gt,
        "renders": renders,
        "runtime_s": time.perf_counter() - t_start,
        "no_axis_in_rgbd_optimizer": True,
        "note": ("section 49: the renderer was asked only which rigid pose "
                 "explains the moving part. The joint was inferred afterwards "
                 "from the recovered finite rotations."),
    }
    _json(out / "summary.json", summary)

    print(f"\n  === {args.scene} DONE in {summary['runtime_s']:.1f}s ===")
    print(f"  {out}")
    print(f"  3D rigid     {a}")
    print(f"  3D reduction {b}")
    print(f"  timelines    {c}")
    if d:
        print(f"  renders      {d}")
    return 0


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def part_gaussians(part, t, mu0, R0):
    from estimator.rigid_part_se3 import state_to_gaussians
    return state_to_gaussians(part, t, mu0, R0)


def _se3_matrix(R, p):
    M = np.eye(4)
    M[:3, :3] = torch.as_tensor(R).double().cpu().numpy()
    M[:3, 3] = torch.as_tensor(p).double().cpu().numpy()
    return M


def _geometry_only(fit: dict) -> dict:
    """Everything but the bulky per-frame arrays, for summary.json."""
    drop = {"poses_R", "poses_p", "pose_residual", "pose_residual_initial",
            "angles"}
    out = {k: v for k, v in fit.items() if k not in drop}
    for k in ("pose_residual", "pose_residual_initial"):
        if k in fit:
            out[k] = {kk: vv for kk, vv in fit[k].items()
                      if not kk.startswith("per_frame")}
    return out


def _load_support(args, mu0, positions, truth):
    """Section 1: coherence MOVING_RIGID, with no silent fallback."""
    if args.phase2_support_source == "current":
        from tools.kpart_pipeline import (_frozen_moving_support,
                                          last_support_evidence)
        members = _frozen_moving_support(truth, None, positions,
                                         argparse.Namespace())
        ev = last_support_evidence()
        ev.setdefault("method", "otsu_excursion")
        ev["ablation"] = ("section 1: the Otsu selector is an explicit "
                          "ablation; the production default is coherence")
        return members, ev

    from estimator.coherence_extraction import NO_MOVING
    frozen = Path(args.coherence_support or
                  f"output/motion_coherence_{args.scene}/"
                  "predicted_support.json")
    if not frozen.is_file():
        raise SystemExit(
            f"section 1: --phase2-support-source coherence needs the frozen "
            f"support at {frozen}. Run tools/run_motion_coherence_support.py "
            f"for {args.scene} first. Falling back to the Otsu selector is "
            "explicitly forbidden.")
    rec = json.loads(frozen.read_text())
    if rec.get("status") == NO_MOVING:
        raise SystemExit(f"section 1: {frozen} reports {NO_MOVING}; no "
                         "fallback is permitted, so this run stops.")
    if int(rec.get("num_gaussians", 0)) != int(mu0.shape[0]):
        raise SystemExit(
            f"{frozen} was computed on {rec.get('num_gaussians')} Gaussians "
            f"but this run has {int(mu0.shape[0])}; the indices would not "
            "refer to the same Gaussians.")
    members = torch.tensor(rec["dynamic_gaussians"], dtype=torch.long)
    return members, {
        "method": rec.get("method", "trajectory_metric_coherence"),
        "source": str(frozen), "frozen_externally": True,
        "kept_fraction": float(members.numel()) / max(mu0.shape[0], 1),
        "class_used": "MOVING_RIGID",
        "ambiguous_excluded": True,
        "selection": rec.get("selection"),
        "identification": rec.get("identification"),
        "note": ("section 1: only the final high-confidence MOVING_RIGID "
                 "support reaches Phase 2. INCOHERENT_OR_AMBIGUOUS nodes are "
                 "excluded: a false positive forces a truly static Gaussian "
                 "through the estimated transform and biases it directly, "
                 "while a false negative only removes evidence.")}


def _rigidity_gate(part, mu0, R0, T, max_pairs=4000, seed=0):
    """Section 20, measured on the real support rather than asserted."""
    from estimator.rigid_part_se3 import state_to_gaussians
    idx = part.members
    g = torch.Generator().manual_seed(seed)
    m = idx.numel()
    k = min(max_pairs, m * (m - 1) // 2 if m > 1 else 0)
    if k == 0:
        return {"max_pairwise_distance_drift": 0.0,
                "max_orientation_defect": 0.0, "pairs": 0,
                "reason": "support has fewer than two members"}
    i = torch.randint(0, m, (k,), generator=g)
    j = torch.randint(0, m, (k,), generator=g)
    keep = i != j
    i, j = idx[i[keep]], idx[j[keep]]
    X0 = mu0.double().cpu()
    d0 = (X0[i] - X0[j]).norm(dim=1)
    worst_d = worst_R = 0.0
    for t in pbar(range(T), "rigidity gate", total=T):
        mu, Rg = state_to_gaussians(part, t, mu0, R0)
        mu = mu.cpu()
        dt = (mu[i] - mu[j]).norm(dim=1)
        worst_d = max(worst_d, float((dt - d0).abs().max()))
        want = part.R[t].to(Rg.device).double() @ R0.double()[idx]
        worst_R = max(worst_R, float((Rg[idx].double() - want).abs().max()))
    return {"max_pairwise_distance_drift": worst_d,
            "max_orientation_defect": worst_R,
            "pairs": int(i.numel()), "frames": T,
            "passes": bool(worst_d < 1e-9 and worst_R < 1e-9),
            "note": ("section 20: exact rigidity is a property of the shared "
                     "SE(3) parameterisation, not of a penalty term")}


def _write_reduction(root: Path, red: dict, T: int):
    """Section 45's joint_reduction/ subtree."""
    root.mkdir(parents=True, exist_ok=True)
    _json(root / "one_dof_diagnostics.json", red["one_dof"])
    _json(root / "relative_increments.json",
          {"convention": red["increments"]["convention"],
           "zeta": red["increments"]["zeta"],
           "dt": red["increments"]["dt"],
           **red["relative_twist_covariance"]})
    for name in ("prismatic", "revolute"):
        f = red[name]
        d = root / name
        d.mkdir(parents=True, exist_ok=True)
        _json(d / "geometry.json", _geometry_only(f))
        if name == "prismatic":
            _csv(d / "trajectory.csv", ["frame", "s"],
                 [[t, f["s"][t]] for t in range(T)])
        else:
            _json(d / "finite_axis_solver.json", f["finite_axis_solver"])
            _csv(d / "trajectory.csv",
                 ["frame", "theta", "theta_raw_projection"],
                 [[t, f["theta"][t], f["angles"]["theta_raw_projection"][t]]
                  for t in range(T)])
        pr = f["pose_residual"]
        _csv(d / "pose_residuals.csv",
             ["frame", "left_rho_x", "left_rho_y", "left_rho_z",
              "left_phi_x", "left_phi_y", "left_phi_z",
              "right_rho_x", "right_rho_y", "right_rho_z",
              "right_phi_x", "right_phi_y", "right_phi_z"],
             [[t, *pr["per_frame_left"][t], *pr["per_frame_right"][t]]
              for t in range(T)])


def _render_camera(args, truth, cameras, cfg, obs_all, mu0, R0, part,
                   kabsch_R, kabsch_p, positions, members, outdir,
                   build_linearization, render_state):
    """Section 22: GT / Phase1 / Kabsch / Phase2 on one camera, every frame."""
    import imageio.v2 as imageio
    from estimator.rigid_part_se3 import state_to_gaussians

    c = max(0, min(args.camera, len(cameras) - 1))
    outdir.mkdir(parents=True, exist_ok=True)
    T = len(obs_all)
    idx = members.to(mu0.device)
    written = []

    def _u8(x):
        x = torch.as_tensor(x).detach().float().cpu()
        if x.dim() == 3 and x.shape[0] in (1, 3):
            x = x.permute(1, 2, 0)
        x = x.clamp(0, 1) if float(x.max()) <= 1.5 else x / max(float(x.max()), 1e-6)
        return (x.numpy() * 255).astype(np.uint8)

    def _depth_u8(d):
        d = torch.as_tensor(d).detach().float().cpu().squeeze()
        lo, hi = float(d.min()), float(d.max())
        n = (d - lo) / max(hi - lo, 1e-6)
        return (n.numpy() * 255).astype(np.uint8)

    for t in pbar(range(T), "renders: write PNGs", total=T):
        layers = {}
        # Phase 1 free motion
        mu_p1 = torch.as_tensor(positions[t]).double().to(mu0.device)
        layers["phase1"] = (mu_p1, R0)
        # Phase-1 Kabsch rigid
        mu_k = mu0.clone()
        Rg_k = R0.clone()
        Rk = kabsch_R[t].to(mu0.device)
        mu_k[idx] = mu_k[idx] @ Rk.T + kabsch_p[t].to(mu0.device)
        Rg_k[idx] = Rk @ Rg_k[idx]
        layers["kabsch"] = (mu_k, Rg_k)
        # rigid Phase 2
        layers["phase2"] = state_to_gaussians(part, t, mu0, R0)
        ob = obs_all[t][c]
        if ob is not None:
            imageio.imwrite(outdir / f"gt_rgb_{t:04d}.png", _u8(ob[0]))
            imageio.imwrite(outdir / f"gt_depth_{t:04d}.png", _depth_u8(ob[1]))
        for name, (mu, Rg) in layers.items():
            lin = render_state(truth, mu, Rg, [cameras[c]], cfg,
                               build_linearization)
            v = lin.views[0]
            imageio.imwrite(outdir / f"{name}_rgb_{t:04d}.png", _u8(v.rgb))
            imageio.imwrite(outdir / f"{name}_depth_{t:04d}.png",
                            _depth_u8(v.depth))
            if ob is not None:
                r = (ob[0].to(v.rgb.device) - v.rgb).abs()
                imageio.imwrite(outdir / f"{name}_residual_{t:04d}.png",
                                _u8(r / max(float(r.max()), 1e-6)))
            del lin
        written.append(t)
    return {"camera": c, "frames": written, "dir": str(outdir),
            "layers": ["gt", "phase1", "kabsch", "phase2"]}


if __name__ == "__main__":
    raise SystemExit(main())
