#!/usr/bin/env python
"""Section 20: the ONE authoritative per-dataset debug entry point.

CODEX_COMPLETE_PHASE2_DEBUG_AND_PARALLEL_PR_MODELS.md.

Runs, in order and with no manual intermediate step (section 20):

    1. Phase 1                      free per-Gaussian motion
    2. freeze the Phase-1 moving support
    3. generic rigid-screw Phase 2
    4. exact prismatic Phase 2
    5. exact revolute Phase 2       (section 13's initialization hierarchy)
    6. all numerical audits         (sections 17-19, corrected 2-6)
    7. chosen-camera renders        every frame (section 23)
    8. the definitive 3D HTML       (sections 25-29)
    9. the comparison summary       (sections 30, 31)

All four models see the SAME canonical 3DGS, the same frozen moving support,
the same cameras, the same RGB-D observations and the same `R_eff` -- and
`shared_inputs_manifest.json` records hashes proving it (section 21).

Nothing here classifies a joint (section 32).
"""

from __future__ import annotations

import argparse
import csv
import dataclasses
import hashlib
import json
import sys
import time
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from utils.progress import pbar, stage  # noqa: E402


# ---------------------------------------------------------------------------
# small helpers
# ---------------------------------------------------------------------------

def _hash_tensor(t) -> str:
    a = torch.as_tensor(t).detach().cpu().contiguous()
    return hashlib.sha256(a.float().numpy().tobytes()).hexdigest()[:16]


def _csv(path: Path, header, rows) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(header)
        w.writerows(rows)


def _json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=1, default=str))


class CudaStage:
    """Section 40: peak CUDA per stage, reset before each one."""

    def __init__(self):
        self.ok = torch.cuda.is_available()
        self.name = torch.cuda.get_device_name(0) if self.ok else None
        self.stages: dict = {}

    def start(self, stage: str):
        if self.ok:
            torch.cuda.reset_peak_memory_stats()
        self._t0, self._s = time.perf_counter(), stage

    def stop(self):
        peak = (torch.cuda.max_memory_allocated() / 2 ** 30) if self.ok else 0.0
        self.stages[self._s] = {"peak_cuda_gib": peak,
                                "seconds": time.perf_counter() - self._t0}
        return self.stages[self._s]

    def to_dict(self) -> dict:
        return {"cuda_available": self.ok, "cuda_device_name": self.name,
                "per_stage": self.stages}


# ---------------------------------------------------------------------------
# state traces (sections 17, 18, 19)
# ---------------------------------------------------------------------------

SCREW_TRACE = ["frame", "time", "q", "qdot", "com_x", "com_y", "com_z",
               "com_dx", "com_dy", "com_dz", "rotvec_x", "rotvec_y",
               "rotvec_z", "theta", "rgb_energy", "depth_energy",
               "total_energy", "accepted", "trust_radius", "pcg_iters",
               "geometry_update_norm", "geometry_info_min_eig",
               "geometry_info_max_eig"]
P_TRACE = ["frame", "time", "s", "sdot", "ePx", "ePy", "ePz",
           "direction_update_norm", "rgb_energy", "depth_energy",
           "total_energy", "accepted", "trust_radius", "pcg_iters",
           "posterior_min_eig", "posterior_max_eig",
           "geometry_info_min_eig", "geometry_info_max_eig"]
R_TRACE = ["frame", "time", "theta", "thetadot", "eRx", "eRy", "eRz",
           "aRx", "aRy", "aRz", "axis_direction_update_norm",
           "axis_position_update_norm", "rgb_energy", "depth_energy",
           "total_energy", "accepted", "trust_radius", "pcg_iters",
           "posterior_min_eig", "posterior_max_eig",
           "geometry_info_min_eig", "geometry_info_max_eig", "R_fit_status"]
PER_FC = ["frame", "camera", "rgb_energy", "depth_energy", "total_energy"]


def _eig(rec, key):
    ev = (getattr(rec, "observability", None) or {}).get(key) or []
    return (min(ev), max(ev)) if ev else (float("nan"), float("nan"))


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scene", required=True)
    ap.add_argument("--model", default=None)
    ap.add_argument("--source", default=None)
    ap.add_argument("--camera", type=int, default=0,
                    help="section 23: the camera rendered for EVERY frame")
    ap.add_argument("--all-frames", action="store_true")
    ap.add_argument("--frames", type=int, default=30)
    ap.add_argument("--cameras", type=int, default=4)
    ap.add_argument("--max-gaussians", type=int, default=0,
                    help="0 (the default) uses EVERY Gaussian in the model; a "
                         "positive value takes a deterministic subsample")
    ap.add_argument("--bootstrap-iters", type=int, default=6)
    ap.add_argument("--screw-iters", type=int, default=8)
    ap.add_argument("--pr-iters", type=int, default=8)
    ap.add_argument("--pr-folds", type=int, default=3)
    ap.add_argument("--max-debug-points", type=int, default=5000)
    ap.add_argument("--phase2-motion-model",
                    choices=("legacy-screw", "legacy-direct-p",
                             "legacy-direct-r", "rigid-se3"),
                    default="legacy-screw",
                    help="section 43: this tool runs the LEGACY direct-axis "
                         "estimators. The production motion model is "
                         "'rigid-se3' and lives in "
                         "tools/run_rigid_se3_pipeline.py.")
    ap.add_argument("--phase2-support-source", choices=("current", "coherence"),
                    default="coherence",
                    help="which static/dynamic selector freezes the Phase-2 "
                         "support. 'coherence' is the PRODUCTION DEFAULT "
                         "(CODEX_IMPLEMENT_RIGID_SE3_PHASE2... section 1); "
                         "'current' is the Otsu excursion ablation. They are "
                         "never combined and there is no silent fallback.")
    ap.add_argument("--coherence-knn", type=int, default=12)
    ap.add_argument("--coherence-support", default=None,
                    help="path to a frozen predicted_support.json from "
                         "tools/run_motion_coherence_support.py. Defaults to "
                         "output/motion_coherence_<scene>/"
                         "predicted_support.json when it exists (section 41).")
    ap.add_argument("--supernode-size", type=int, default=64)
    ap.add_argument("--no-renders", action="store_true")
    ap.add_argument("--frames-on-cpu", action="store_true",
                    help="stage the window in host memory and move each frame "
                         "to the device on demand. 60 frames x 8 cameras of "
                         "640x480 RGB-D pin about 2.4 GiB on the GPU, which "
                         "on an 8 GiB card leaves too little for the "
                         "rasterizer at the full Gaussian budget")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    if args.phase2_motion_model == "rigid-se3":
        raise SystemExit(
            "--phase2-motion-model rigid-se3 is the PRODUCTION model and is "
            "implemented in tools/run_rigid_se3_pipeline.py (or "
            "./run_rigid_se3_all.sh). This tool runs the legacy direct-axis "
            "estimators only.")

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

    out = Path(args.out or f"output/complete_joint_debug_{args.scene}")
    out.mkdir(parents=True, exist_ok=True)
    cuda = CudaStage()
    print("=" * 70)
    print(" LEGACY / ABLATION BRANCH -- direct P/R axis estimation in RGB-D")
    print(" The production path is  ./run_rigid_se3_all.sh  (rigid SE(3)")
    print(" Phase 2 + post-hoc joint reduction). Measured on all five")
    print(" datasets, the axes this tool produces are worse, by up to 89")
    print(" degrees: see RIGID_SE3_PHASE2_AND_JOINT_REDUCTION_REPORT.md 21.")
    print("=" * 70)
    print(f"=== COMPLETE JOINT DEBUG: {args.scene} ===")
    print(f"  model  {model}\n  source {source}\n  out    {out}")

    # ---- imports (after sys.path) ---------------------------------------
    from estimator.debug_3d import (LAYER_COLORS, build_payload,
                                    select_indices, write_3d_html)
    from estimator.debug_renders import (render_chosen_camera,
                                         write_chosen_camera_html)
    from estimator.joint_iekf import JointIEKFConfig, run_window
    from estimator.joint_runtime import run_phase2_rigid_screw
    from estimator.model_error import estimate_model_error
    from estimator.phase2_model_comparison import (fit_model, init_prismatic,
                                                   prismatic_state,
                                                   revolute_state)
    from estimator.phase2_revolute_init import (FITTED, UNOBSERVABLE,
                                                fit_status,
                                                initialize_revolute)
    from estimator.phase2_shared import (effective_precision, render_state,
                                         weighted_energy)
    from estimator.renderer_bridge import build_linearization
    from estimator.rigid_motion_audit import (PER_FRAME_CAMERA_HEADER,
                                              audit_frame, build_state,
                                              geometry_score_headers,
                                              geometry_scores,
                                              per_frame_camera_rows,
                                              rigidity_check, summarise)
    from estimator.sparse_iekf import IEKFConfig
    from tools.kpart_pipeline import _frozen_moving_support, bootstrap_positions
    from tools.run_iekf_scene import (camera_specs, dataset_rgbd_frames,
                                      load_scene, state_from_gaussians)

    # ---- scene, cameras, observations -----------------------------------
    dataset, pipe, gaussians, sc = load_scene(model, source)
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

    nframes = 0 if args.all_frames else args.frames
    ns = argparse.Namespace(
        sequence_start=0, sequence_length=nframes,
        all_frames=bool(args.all_frames),
        observation_source="dataset_rgbd", cameras=ncam,
        bootstrap_iters=args.bootstrap_iters, supernode_size=64,
        seed=args.seed, frames_on_cpu=bool(args.frames_on_cpu),
        frames_on_gpu=not args.frames_on_cpu,
        depth_sigma=None, joint_direct_rgbd=False, joint_direct_gaussian=False,
        window_length=0, all_cameras=False, dt=1.0 / 30.0,
        max_gaussians=args.max_gaussians, no_model_error=False,
        model_error_beta=1.0, model_error_blur=1)
    frames_raw, frame_ids = dataset_rgbd_frames(cams_all[:ncam], cfg, ns,
                                               truth, Path(source))
    cameras = frames_raw[0][1]
    obs_all = [[(o[0], o[1], o[2] if len(o) > 2 else None) if o is not None
                else None for o in ob] for (ob, _s, _d) in frames_raw]
    T = len(obs_all)
    mu0 = truth.mu.double()
    print(f"  {truth.mu.shape[0]} Gaussians, {ncam} cameras, {T} frames")
    if not 0 <= args.camera < ncam:
        raise SystemExit(f"--camera {args.camera} is outside 0..{ncam - 1}")

    # ---- section 13 of the backend spec: R_eff = R_sensor + R_model ------
    # With --frames-on-cpu the window lives in host memory; the model-error
    # estimator compares against rendered views on the device, so the
    # reference frame is moved explicitly rather than relying on the caller.
    dev = truth.mu.device
    ref_obs = [(o[0].to(dev), o[1].to(dev),
                o[2].to(dev) if o[2] is not None else None)
               for o in obs_all[0] if o is not None]
    lin0 = build_linearization(truth, cameras, eps_depth=cfg.eps_depth)
    model_error = estimate_model_error(lin0, ref_obs, reference_index=0,
                                       beta=1.0, blur=1)
    del lin0
    cam_ids = list(range(ncam))

    def prec_at(lin, t):
        return effective_precision(lin, obs_all[t], cam_ids, cfg, model_error)

    def energies(mu, Rg, t):
        lin = render_state(truth, mu, Rg, cameras, cfg, build_linearization)
        rows = weighted_energy(lin, obs_all[t], cam_ids, prec_at(lin, t))
        del lin
        return rows

    # =====================================================================
    # 1. PHASE 1 -- free per-Gaussian motion
    # =====================================================================
    print("\n  [1/9] Phase 1: free per-Gaussian motion")
    stage(f"[1/9] {args.scene}: Phase 1 over {T} frames x {ncam} cameras")
    cuda.start("phase1")
    # kpart/joint runtimes consume `(observations, frame_id)` pairs, while
    # the dataset loader yields `(observations, camera_specs, dt)`. The window
    # is built once here and shared by Phase 1 and every Phase-2 model, so
    # there is no chance of two stages pairing frames differently.
    window = [(ob, fid) for (ob, _specs, _dt), fid
              in zip(frames_raw, frame_ids["frames"])]
    positions, p1_info = bootstrap_positions(truth, cameras, window, cfg,
                                             None, ns, verbose=True)
    cuda.stop()
    p1dir = out / "phase1"
    (p1dir / "centers").mkdir(parents=True, exist_ok=True)
    import numpy as np
    for t, p in pbar(list(enumerate(positions)), "phase1 centers", total=T):
        np.save(p1dir / "centers" / f"centers_frame_{t:04d}.npy",
                torch.as_tensor(p).double().cpu().numpy())
    p1_rows, p1_fc = [], []
    energy_cache = {}
    for t in pbar(range(T), "phase1 energies", total=T):
        mu = torch.as_tensor(positions[t]).double().to(mu0.device)
        rows = energies(mu, None, t)
        energy_cache[("phase1", t)] = rows
        tot = sum(r["total"] for r in rows)
        d = (mu - mu0).norm(dim=1)
        p1_rows.append([t, float(t), float(d.mean()), float(d.max()),
                        sum(r["rgb"] for r in rows),
                        sum(r["depth"] for r in rows), tot])
        for r in rows:
            p1_fc.append([t, r["camera"], r["rgb"], r["depth"], r["total"]])
    _csv(p1dir / "numerical" / "state_trace.csv",
         ["frame", "time", "mean_displacement", "max_displacement",
          "rgb_energy", "depth_energy", "total_energy"], p1_rows)
    _csv(p1dir / "numerical" / "per_frame_camera.csv", PER_FC, p1_fc)

    # ---- 2. freeze the moving support (section 21 pins it here) ---------
    sup_args = argparse.Namespace()
    if args.phase2_support_source == "coherence":
        # Sections 19/41: the coherence branch freezes the support, and screw,
        # P and R all receive exactly these indices. Section 20: nothing from
        # the Otsu branch is unioned, intersected or voted in here.
        #
        # The frozen `predicted_support.json` written by
        # tools/run_motion_coherence_support.py is preferred over recomputing,
        # because section 41 is explicit that Phase 2 plugs in AFTER freezing:
        # loading it guarantees Phase 2 is given the same index set that was
        # evaluated against ground truth, rather than a fresh extraction that
        # could differ.
        from estimator.coherence_extraction import (
            MOVING_RIGID, NO_MOVING, extract_moving_component)
        from estimator.motion_coherence_support import (
            canonical_knn_graph, edge_strain, propagate_to_gaussians,
            supernode_trajectories)
        from estimator.supernodes import build_supernodes

        frozen = Path(args.coherence_support or
                      f"output/motion_coherence_{args.scene}/"
                      "predicted_support.json")
        if frozen.is_file():
            rec = json.loads(frozen.read_text())
            if rec.get("status") == NO_MOVING:
                raise SystemExit(
                    f"section 45: {frozen} reports {NO_MOVING} for "
                    f"{args.scene}. Falling back to the Otsu selector is "
                    "explicitly forbidden, so this run stops.")
            members = torch.tensor(rec["dynamic_gaussians"], dtype=torch.long)
            if int(rec.get("num_gaussians", 0)) != int(mu0.shape[0]):
                raise SystemExit(
                    f"{frozen} was computed on {rec.get('num_gaussians')} "
                    f"Gaussians but this run has {int(mu0.shape[0])}. The "
                    "frozen support would not refer to the same Gaussians, so "
                    "it cannot be used; re-run the coherence branch with the "
                    "same --max-gaussians.")
            ev = {"method": rec.get("method", "trajectory_metric_coherence"),
                  "source": str(frozen), "frozen_externally": True,
                  "kept_fraction": float(members.numel()) / max(mu0.shape[0], 1),
                  "num_supernodes": rec.get("num_supernodes"),
                  "threshold": float("nan"), "separability": float("nan"),
                  "selection": rec.get("selection"),
                  "identification": rec.get("identification"),
                  "excursion": {},
                  "note": ("section 41: loaded the ALREADY-FROZEN coherence "
                           "support, so Phase 2 sees exactly the indices that "
                           "were evaluated against GT")}
        else:
            part_sn = build_supernodes(mu0.float().cpu(),
                                       target_size=args.supernode_size,
                                       seed=args.seed)
            sid_sn = part_sn.supernode_id.long().cpu()
            sn = supernode_trajectories(positions, sid_sn)
            nid = torch.nonzero(sn["valid"], as_tuple=False).reshape(-1)
            tr = sn["trajectories"][:, nid, :]
            g_sn = canonical_knn_graph(tr[0], k=args.coherence_knn)
            st_sn = edge_strain(tr, g_sn["edge_i"], g_sn["edge_j"])
            ex = extract_moving_component(tr, g_sn["edge_i"], g_sn["edge_j"],
                                          st_sn["C"], int(nid.numel()),
                                          verbose=verbose)
            if ex["status"] == NO_MOVING:
                # Section 45: no silent fallback to the other selector.
                raise SystemExit(
                    f"section 45: the coherence extractor returned "
                    f"{NO_MOVING} on {args.scene}. Falling back to the Otsu "
                    "selector is explicitly forbidden, so this run stops.")
            members = propagate_to_gaussians(nid[ex["moving_nodes"]], sid_sn)
            ev = {"method": ex["method"], "frozen_externally": False,
                  "kept_fraction": float(members.numel()) / max(mu0.shape[0], 1),
                  "num_supernodes": int(nid.numel()),
                  "threshold": float("nan"), "separability": float("nan"),
                  "coherence_knn": args.coherence_knn,
                  "selection": ex["selection"], "counts": ex["counts"],
                  "excursion": {},
                  "note": ("section 19: support frozen by the motion-coherence "
                           "v2 extractor; the Otsu selector was not called")}
    else:
        members = _frozen_moving_support(truth, None, positions, sup_args)
        from tools.kpart_pipeline import last_support_evidence
        ev = last_support_evidence()
    ev.setdefault("method", "unknown")
    print(f"  [2/9] frozen moving support [{ev['method']}]: "
          f"{int(members.numel())} of "
          f"{mu0.shape[0]} Gaussians ({100 * ev.get('kept_fraction', 0):.1f}%)"
          f"  threshold {ev.get('threshold', 0):.4f}"
          f"  separability {ev.get('separability', 0):.2f}"
          f"  [{ev.get('method')}]")
    exc = ev.get("excursion", {})
    print(f"      excursion: selected median {exc.get('selected_median', 0):.4f}"
          f"  rejected mean {exc.get('rejected_mean', 0):.5f}"
          f"  contrast {exc.get('contrast', 0):.1f}x")
    _json(p1dir / "support.json", {
        **ev,
        "num_members": int(members.numel()),
        "members": members.tolist(),
        "frozen_before_phase2": True,
        "note": ("section 21: every Phase-2 model below is given exactly this "
                 "index set; it is never recomputed per model")})
    _json(p1dir / "covariance_or_information_summary.json",
          {k: v for k, v in (p1_info or {}).items()
           if not torch.is_tensor(v)})

    # =====================================================================
    # 3. GENERIC RIGID-SCREW PHASE 2
    # =====================================================================
    print("\n  [3/9] Phase 2: generic rigid screw")
    stage(f"[3/9] {args.scene}: generic rigid screw "
          f"({int(members.numel())} moving Gaussians)")
    from tools.iekf_artifacts import ArtifactWriter
    writer = ArtifactWriter(out / "phase2_screw" / "_runtime")
    pargs = argparse.Namespace(phase2_screw_iters=args.screw_iters,
                               phase2_screw_cg_iters=40,
                               phase2_block_frames=1, phase2_null_passes=2)
    cuda.start("phase2_screw")
    screw = run_phase2_rigid_screw(truth, cameras, window, cfg, pargs,
                                   writer, members=members,
                                   positions=positions,
                                   model_error=model_error, verbose=True)
    cuda.stop()
    s_state, s_recs = screw["state"], screw["records"]
    s_part = s_state.parts[0]

    sdir = out / "phase2_screw"
    rows, fc = [], []
    for t in pbar(range(T), "screw trace", total=T):
        rec = s_recs[t]
        mu, Rg = build_state("full", s_part, t, mu0, truth.R.double())
        er = energies(mu, Rg, t)
        energy_cache[("screw", t)] = er
        dec = __import__("estimator.rigid_motion_audit", fromlist=["x"]) \
            .com_decomposition(s_part, t, mu0)
        gmin, gmax = _eig(rec, "geometry_information_eigenvalues")
        rows.append([t, float(t), float(s_part.q[t]), float(s_part.qdot[t]),
                     *dec["c_t"].tolist(), *dec["com_displacement"].tolist(),
                     *dec["phi"].tolist(), dec["theta"],
                     sum(r["rgb"] for r in er), sum(r["depth"] for r in er),
                     sum(r["total"] for r in er), rec.iterations and
                     rec.accepted, rec.radius, rec.iterations,
                     float(s_part.zeta.norm()), gmin, gmax])
        for r in er:
            fc.append([t, r["camera"], r["rgb"], r["depth"], r["total"]])
    _csv(sdir / "numerical" / "state_trace.csv", SCREW_TRACE, rows)
    _csv(sdir / "numerical" / "per_frame_camera.csv", PER_FC, fc)

    # ---- static / translation-only / rotation-only / full (sections 2-5)
    print("  [4/9] Stage-A audit with REAL Gaussian orientations")
    R0 = truth.R.double()

    def build_lin(mu, Rg):
        return render_state(truth, mu, Rg, cameras, cfg, build_linearization)

    cuda.start("stage_a_audit")
    audits = [audit_frame(s_part, t, mu0, R0, obs_all[t], truth, cameras,
                          build_lin, lambda lin, _t=t: prec_at(lin, _t),
                          cfg.min_alpha_for_depth)
              for t in pbar(range(T), "stage-A static/trans/rot/full", total=T)]
    cuda.stop()
    a_rows = per_frame_camera_rows(audits)
    summ = summarise(audits, a_rows)
    _csv(sdir / "static_trans_rot_full" / "per_frame_camera.csv",
         PER_FRAME_CAMERA_HEADER, a_rows)
    rig = [rigidity_check(s_part, t, mu0, R0, scales=truth.scales.double())
           for t in (0, T // 2, T - 1)]
    _json(sdir / "static_trans_rot_full" / "summary.json", {
        "contribution": summ, "rigidity": rig,
        "reference_precision_source": "full_rigid",
        "robust_weights_frozen": True,
        "r_eff_source": "estimator.phase2_shared.effective_precision",
        "orientation_transport_exercised": True,
        "note": ("sections 2-5: all four states carry the real canonical "
                 "Gaussian orientations through the rasterizer, and all four "
                 "are scored with the SAME R_eff evaluated once at the "
                 "full-rigid state")})

    # ---- geometry scores, per frame AND per camera (section 6) ---------
    def _scores(state, tag, outdir):
        ctr = {"t": 0}

        def lin_for(st, t):
            ctr["t"] = t
            mu, Rg = (st.gaussians(t, mu0, R0) if hasattr(st, "gaussians")
                      else build_state("full", st.parts[0], t, mu0, R0))
            return build_lin(mu, Rg)

        g = geometry_scores(state, mu0, R0, obs_all, lin_for,
                            lambda lin: prec_at(lin, ctr["t"]),
                            camera_ids=cam_ids)
        h = geometry_score_headers(g["g_dim"])
        _csv(outdir / "numerical" / "geometry_score_per_frame_camera.csv",
             h["frame_camera"], g["rows_frame_camera"])
        _csv(outdir / "numerical" / "geometry_score_per_frame.csv",
             h["frame"], g["rows_frame"])
        _csv(outdir / "numerical" / "geometry_score_per_camera.csv",
             h["camera"], g["rows_camera"])
        np.savez(outdir / "numerical"
                 / "geometry_information_per_frame_camera.npz",
                 rows=np.array(g["rows_frame_camera"], dtype=object),
                 F_total=g["F_total"].numpy())
        _json(outdir / "numerical" / "geometry_consistency.json",
              {k: v for k, v in g.items()
               if k not in ("rows_frame_camera", "rows_frame", "rows_camera",
                            "g_per_frame", "g_per_camera", "F_total")})
        print(f"      {tag}: across frames cos "
              f"{g['across_frames']['median_pairwise_cosine']:.4f} "
              f"opposed {100 * g['across_frames']['fraction_opposed']:.1f}%  |  "
              f"across cameras cos "
              f"{g['across_cameras']['median_pairwise_cosine']:.4f} "
              f"opposed {100 * g['across_cameras']['fraction_opposed']:.1f}%")
        return g

    g_screw = _scores(s_state, "screw", sdir)

    # =====================================================================
    # 5. EXACT P and EXACT R
    # =====================================================================
    jcfg = JointIEKFConfig(max_iters=args.pr_iters, max_cg_iters=40,
                           verbose=False)

    def bind(state, train):
        ctr = {"t": 0}
        sub = [cameras[c] for c in train]

        def lin(st, t):
            ctr["t"] = t
            mu, Rg = st.gaussians(t, mu0, R0)
            return render_state(truth, mu, Rg, sub, cfg, build_linearization)

        def prec(l):
            return effective_precision(l, obs_all[ctr["t"]], train, cfg,
                                       model_error)

        o = [[(ob[c][0], ob[c][1]) if ob[c] is not None else None
              for c in train] for ob in obs_all]
        return lin, prec, o

    print("\n  [5/9] Phase 2: exact prismatic")
    stage(f"[5/9] {args.scene}: exact prismatic")
    cuda.start("phase2_prismatic")
    p_part = init_prismatic(mu0.cpu(), members,
                            [torch.as_tensor(p).double().cpu()
                             for p in positions])
    p_part_init_direction = p_part.direction.clone()
    p_state = prismatic_state(p_part)
    lin_p, prec_p, obs_p = bind(p_state, cam_ids)
    p_fit = fit_model(p_state, mu0, obs_p, lin_p, prec_p, jcfg,
                      label="prismatic")
    cuda.stop()

    print("  [6/9] Phase 2: exact revolute (section 13 hierarchy)")
    stage(f"[6/9] {args.scene}: exact revolute")
    cuda.start("phase2_revolute")
    r_init = initialize_revolute(members, mu0.cpu(), positions,
                                 screw_state=s_state, screw_records=s_recs)
    r_state = r_fit = None
    if r_init["part"] is not None:
        before_axis = r_init["part"].axis.clone()
        before_rho = r_init["part"].rho.clone()
        r_state = revolute_state(r_init["part"])
        lin_r, prec_r, obs_r = bind(r_state, cam_ids)
        r_fit = fit_model(r_state, mu0, obs_r, lin_r, prec_r, jcfg,
                          label="revolute")
        r_status = fit_status(r_state.parts[0], r_fit["records"],
                              before_axis, before_rho)
    else:
        r_status = {"R_fit_status": UNOBSERVABLE, "R_fit_valid": False,
                    "max_abs_theta": 0.0, "accepted_iterations": 0,
                    "axis_update_norm": 0.0, "axis_offset_update_norm": 0.0,
                    "geometry_information_rank": 0}
    cuda.stop()
    print(f"      R status: {r_status['R_fit_status']}"
          f"  (init {r_init.get('initialization_source')})")

    # ---- traces for P and R (sections 18, 19) --------------------------
    pdir, rdir = out / "phase2_prismatic", out / "phase2_revolute"
    p_dir_update = float((p_state.parts[0].direction
                          - p_part_init_direction).norm())
    for tag, st, fit, hdr, d in (("prismatic", p_state, p_fit, P_TRACE, pdir),
                                 ("revolute", r_state, r_fit, R_TRACE, rdir)):
        if st is None:
            _json(d / "numerical" / "not_fitted.json",
                  {"reason": r_init.get("reason"), **r_status})
            continue
        part = st.parts[0]
        rows, fcs = [], []
        for t in pbar(range(T), f"{tag} trace", total=T):
            rec = fit["records"][t]
            mu, Rg = st.gaussians(t, mu0, R0)
            er = energies(mu, Rg, t)
            energy_cache[(tag, t)] = er
            pe = (getattr(rec, "observability", None) or {}).get(
                "posterior_eigenvalues") or [float("nan")]
            gmin, gmax = _eig(rec, "geometry_information_eigenvalues")
            base = [t, float(t), float(part.q[t]), float(part.qdot[t])]
            if tag == "prismatic":
                base += part.direction.tolist() + [p_dir_update]
            else:
                base += (part.axis.tolist() + part.axis_point.tolist()
                         + [r_status["axis_update_norm"],
                            r_status["axis_offset_update_norm"]])
            row = base + [sum(x["rgb"] for x in er),
                          sum(x["depth"] for x in er),
                          sum(x["total"] for x in er),
                          rec.accepted, rec.radius, rec.iterations,
                          min(pe), max(pe), gmin, gmax]
            if tag == "revolute":
                row.append(r_status["R_fit_status"])
            rows.append(row)
            for x in er:
                fcs.append([t, x["camera"], x["rgb"], x["depth"], x["total"]])
        _csv(d / "numerical" / "state_trace.csv", hdr, rows)
        _csv(d / "numerical" / "per_frame_camera.csv", PER_FC, fcs)
        _scores(st, tag, d)
    _json(rdir / "numerical" / "observability.json",
          {**r_status, "initialization": r_init})

    # =====================================================================
    # 7-8. RENDERS AND HTML
    # =====================================================================
    def centers(name):
        if name == "phase1":
            return lambda t: torch.as_tensor(positions[t]).double()
        if name == "screw":
            return lambda t: build_state("full", s_part, t, mu0, R0)[0]
        if name == "prismatic":
            return lambda t: p_state.gaussians(t, mu0, R0)[0]
        if name == "revolute" and r_state is not None:
            return lambda t: r_state.gaussians(t, mu0, R0)[0]
        return None

    state_fns = {}
    for name in ("phase1", "screw", "prismatic", "revolute"):
        f = centers(name)
        if f is None:
            continue
        if name == "phase1":
            state_fns[name] = lambda t: (torch.as_tensor(positions[t]).double()
                                         .to(mu0.device), None)
        elif name == "screw":
            state_fns[name] = lambda t: build_state("full", s_part, t, mu0, R0)
        elif name == "prismatic":
            state_fns[name] = lambda t: p_state.gaussians(t, mu0, R0)
        else:
            state_fns[name] = lambda t: r_state.gaussians(t, mu0, R0)

    html_dir = out / "html"
    html_dir.mkdir(parents=True, exist_ok=True)
    render_index = None
    if not args.no_renders:
        print(f"\n  [7/9] rendering camera {args.camera}, all {T} frames")
        stage(f"[7/9] {args.scene}: rendering camera {args.camera}, "
              f"{T} frames x 5 methods")
        cuda.start("renders")
        render_index = render_chosen_camera(
            out, args.camera,
            [[(o[0], o[1]) if o is not None else None for o in ob]
             for ob in obs_all],
            state_fns, truth, cameras, cfg, build_linearization,
            method_dirs={"gt": out / "gt" / "renders",
                         "phase1": p1dir / "renders",
                         "screw": sdir / "renders",
                         "prismatic": pdir / "renders",
                         "revolute": rdir / "renders"})
        cuda.stop()
        per_frame_nums = {}
        for t in range(T):
            d = {"screw q": f"{float(s_part.q[t]):+.5f}",
                 "screw qdot": f"{float(s_part.qdot[t]):+.5f}",
                 "P s": f"{float(p_state.parts[0].s[t]):+.5f}",
                 "Phase1 |dx| mean": f"{p1_rows[t][2]:.5f}",
                 "R status": r_status["R_fit_status"]}
            if r_state is not None:
                d["R theta"] = f"{float(r_state.parts[0].theta[t]):+.5f}"
            d["screw total energy"] = f"{rows[t][16]:.4e}"
            d["Phase1 total energy"] = f"{p1_rows[t][6]:.4e}"
            per_frame_nums[t] = d
        write_chosen_camera_html(html_dir / "chosen_camera_all_frames.html",
                                 render_index, per_frame_nums, "..")

    # section 22: each Phase-2 model's own centers, frame by frame, so the
    # 3D page is reproducible from files rather than only from this process.
    for tag, fn, d in (("screw", centers("screw"), sdir),
                       ("prismatic", centers("prismatic"), pdir),
                       ("revolute", centers("revolute"), rdir)):
        if fn is None:
            continue
        (d / "centers").mkdir(parents=True, exist_ok=True)
        for t in pbar(range(T), f"{tag} centers", total=T):
            np.save(d / "centers" / f"centers_frame_{t:04d}.npy",
                    fn(t).detach().cpu().numpy())

    print("  [8/9] definitive 3D HTML")
    idx = select_indices(members, args.max_debug_points, args.seed)
    np.save(html_dir / "debug_gaussian_indices.npy", idx.numpy())
    allidx = torch.arange(mu0.shape[0])
    mask = torch.ones(mu0.shape[0], dtype=torch.bool)
    mask[members] = False
    static_idx = allidx[mask]
    if static_idx.numel() > args.max_debug_points:
        gseed = torch.Generator().manual_seed(args.seed + 1)
        static_idx = static_idx[torch.randperm(
            static_idx.numel(), generator=gseed)[:args.max_debug_points]]

    layers = {}
    for name in ("phase1", "screw", "prismatic", "revolute"):
        f = centers(name)
        if f is not None:
            layers[name] = (lambda t, _f=f: _f(t) if t < T else None)
    ov = {}
    c0 = mu0.cpu()[members].mean(0)
    span = float((mu0.cpu()[members] - c0).norm(dim=1).max()) * 1.6
    pe = p_state.parts[0].direction
    ov["p_direction"] = {"type": "line", "label": "P direction e_P",
                         "color": LAYER_COLORS["prismatic"],
                         "a": (c0 - pe * span).tolist(),
                         "b": (c0 + pe * span).tolist()}
    ax = __import__("estimator.joint_manifold", fromlist=["x"]).revolute_axis(
        s_part.zeta, s_part.length, s_part.centroid)
    if ax is not None:
        e, a = ax["axis"], ax["axis_point"]
        ov["screw_axis"] = {"type": "line", "label": "generic screw axis",
                            "color": LAYER_COLORS["screw"],
                            "a": (a - e * span).tolist(),
                            "b": (a + e * span).tolist()}
    if r_state is not None:
        e, a = r_state.parts[0].axis, r_state.parts[0].axis_point
        ov["revolute_axis"] = {"type": "line", "label": "R axis e_R, a_R",
                               "color": LAYER_COLORS["revolute"],
                               "a": (a - e * span).tolist(),
                               "b": (a + e * span).tolist()}
    com_path = [torch.as_tensor(positions[t]).double().cpu()[members].mean(0)
                .tolist() for t in range(T)]
    ov["phase1_com"] = {"type": "path", "label": "Phase 1 COM trail",
                        "color": LAYER_COLORS["phase1"], "points": com_path}
    labels = []
    for t in range(T):
        L = {"frame": t, "time": f"{t}",
             "screw q": f"{float(s_part.q[t]):+.5f}",
             "P s": f"{float(p_state.parts[0].s[t]):+.5f}"}
        if r_state is not None:
            L["R theta"] = f"{float(r_state.parts[0].theta[t]):+.5f}"
        labels.append(L)
    payload = build_payload(idx, mu0, layers, static_idx=static_idx,
                            overlays=ov, cameras=cameras,
                            frame_labels=labels, max_frames=T)
    write_3d_html(html_dir / "definitive_joint_3d.html", payload,
                  f"{args.scene} — Phase 1 / screw / P / R")

    # The full debugger: same shell as the old geometric_motion_debug page
    # (stage selector, playback, dashboard) with the four motions this
    # pipeline produces, and with the moving support drawn against the static
    # remainder -- the support is upstream of every joint estimate here.
    from debug.joint_motion_debug import build_payload as build4
    from debug.joint_motion_debug import write_html as write4

    stage_pos = {}
    for name in ("phase1", "screw", "prismatic", "revolute"):
        fn = centers(name)
        if fn is not None:
            stage_pos[name] = [fn(t).detach().cpu() for t in range(T)]
    ov4 = dict(ov)
    for k, o in ov4.items():
        if k == "p_direction":
            o["info"] = ("e_P = " + str([round(x, 3) for x in pe.tolist()])
                         + f", travel {float(p_state.parts[0].s.abs().max()):.4f}")
        elif k == "screw_axis" and ax is not None:
            o["info"] = (f"|omega| {ax['omega_norm']:.4f}, "
                         f"pitch {ax['pitch']:.4f}")
        elif k == "revolute_axis" and r_state is not None:
            o["info"] = (f"sweep {float(r_state.parts[0].theta.abs().max()):.4f} rad"
                         f", {r_status['R_fit_status']}")
    payload4 = build4(mu0, members, stage_pos, static_idx=static_idx,
                      overlays=ov4, cameras=payload.get("cameras"),
                      frame_labels=labels, support_evidence=ev,
                      max_points=args.max_debug_points, seed=args.seed,
                      meta={"scene": args.scene, "frames": T,
                            "cameras": ncam,
                            "gaussians": int(mu0.shape[0]),
                            "R status": r_status["R_fit_status"]})
    write4(html_dir / "joint_motion_debug.html", payload4,
           f"{args.scene} — four-motion joint debug")

    # =====================================================================
    # 9. COMPARISON (sections 30, 31)
    # =====================================================================
    print("  [9/9] comparison and held-out folds")
    stage(f"[9/9] {args.scene}: comparison and held-out folds")
    cdir = out / "comparison"

    # Every per-frame, per-camera energy was already computed and written to
    # the state traces above, on exactly these states and with exactly this
    # R_eff. Re-rendering all of them a second time just to sum them would
    # roughly double the run for no new number.
    def totals(tag, camera=None):
        rgb = dep = 0.0
        for t in range(T):
            for r in energy_cache.get((tag, t), []):
                if camera is not None and r["camera"] != camera:
                    continue
                rgb += r["rgb"]
                dep += r["depth"]
        return {"rgb": rgb, "depth": dep, "total": rgb + dep}

    train = {k: totals(k) for k in ("phase1", "screw", "prismatic")
             if any((k, t) in energy_cache for t in range(T))}
    if r_state is not None and any(("revolute", t) in energy_cache
                                   for t in range(T)):
        train["revolute"] = totals("revolute")

    # held-out folds, P and R only (section 31)
    nf = max(1, min(args.pr_folds, ncam))
    folds = [([c for c in cam_ids if c % nf != f],
              [c for c in cam_ids if c % nf == f]) for f in range(nf)]
    fold_rows, interpretable = [], True
    for fi, (tr, ho) in enumerate(folds):
        if not tr or not ho:
            continue
        ps = prismatic_state(init_prismatic(
            mu0.cpu(), members,
            [torch.as_tensor(p).double().cpu() for p in positions]))
        l, pr, o = bind(ps, tr)
        fit_model(ps, mu0, o, l, pr, jcfg, label=f"fold {fi} P fit")

        def E(st, cams_, _desc="fold energy"):
            rgb = dep = 0.0
            sub = [cameras[c] for c in cams_]
            for t in pbar(range(T), _desc, total=T):
                mu, Rg = st.gaussians(t, mu0, R0)
                lin = render_state(truth, mu, Rg, sub, cfg,
                                   build_linearization)
                for r in weighted_energy(lin, obs_all[t], cams_,
                                         effective_precision(
                                             lin, obs_all[t], cams_, cfg,
                                             model_error)):
                    rgb += r["rgb"]
                    dep += r["depth"]
                del lin
            return rgb, dep

        p_tr_rgb, p_tr_d = E(ps, tr, f"fold {fi} P train")
        p_ho_rgb, p_ho_d = E(ps, ho, f"fold {fi} P heldout")
        r_tr = r_ho = (float("nan"), float("nan"))
        rstat = UNOBSERVABLE
        if r_init["part"] is not None:
            rs = revolute_state(initialize_revolute(
                members, mu0.cpu(), positions, screw_state=s_state,
                screw_records=s_recs)["part"])
            ba, brho = rs.parts[0].axis.clone(), rs.parts[0].rho.clone()
            l, pr, o = bind(rs, tr)
            rf = fit_model(rs, mu0, o, l, pr, jcfg,
                           label=f"fold {fi} R fit")
            rstat = fit_status(rs.parts[0], rf["records"], ba,
                               brho)["R_fit_status"]
            r_tr = E(rs, tr, f"fold {fi} R train")
            r_ho = E(rs, ho, f"fold {fi} R heldout")
        valid = rstat == FITTED
        interpretable = interpretable and valid
        fold_rows.append([fi, " ".join(map(str, tr)), " ".join(map(str, ho)),
                          p_tr_rgb + p_tr_d, r_tr[0] + r_tr[1],
                          p_ho_rgb + p_ho_d, r_ho[0] + r_ho[1],
                          rstat, valid])
        print(f"      fold {fi}: P heldout {p_ho_rgb + p_ho_d:.4e}  "
              f"R heldout {r_ho[0] + r_ho[1]:.4e}  [{rstat}]")
    _csv(cdir / "heldout_metrics.csv",
         ["fold", "train_cameras", "heldout_cameras", "P_train", "R_train",
          "P_heldout", "R_heldout", "R_fit_status", "comparison_interpretable"],
         fold_rows)
    _csv(cdir / "train_metrics.csv",
         ["model", "rgb", "depth", "total"],
         [[k, v["rgb"], v["depth"], v["total"]] for k, v in train.items()])

    # section 30: the chosen camera's own share, from the same cache
    chosen = {k: totals(k, camera=args.camera) for k in train}
    summary = {
        "scene": args.scene, "frames": T, "cameras": ncam,
        "chosen_camera": args.camera,
        "num_moving_gaussians": int(members.numel()),
        "train": train,
        "chosen_camera_energy": chosen,
        "screw": {"geometry": screw.get("geometry"),
                  "translation_only_share": summ["translation_only_share"],
                  "rotation_after_translation_share":
                      summ["rotation_share_of_full_gain"],
                  "camera_spread": summ["rotation_share_spread_across_cameras"],
                  "max_theta_rad": summ["max_theta_rad"],
                  "max_com_displacement": summ["max_com_displacement"],
                  "geometry_consistency":
                      {"across_frames": g_screw["across_frames"],
                       "across_cameras": g_screw["across_cameras"]}},
        "prismatic": {"direction": p_state.parts[0].direction.tolist(),
                      "total_travel": float(p_state.parts[0].s.abs().max())},
        "revolute": ({"axis": r_state.parts[0].axis.tolist(),
                      "axis_point": r_state.parts[0].axis_point.tolist(),
                      "total_angle_sweep":
                          float(r_state.parts[0].theta.abs().max()),
                      **r_status} if r_state is not None else r_status),
        "heldout_interpretable": bool(interpretable and fold_rows),
        "not_a_classifier": True,
        "cuda": cuda.to_dict(),
    }
    _json(cdir / "summary.json", summary)

    # ---- section 21: prove the inputs were shared -----------------------
    _json(out / "shared_inputs_manifest.json", {
        "canonical_3dgs": {"num_gaussians": int(mu0.shape[0]),
                           "mu_sha256_16": _hash_tensor(mu0),
                           "R_sha256_16": _hash_tensor(truth.R),
                           "scales_sha256_16": _hash_tensor(truth.scales),
                           "colors_sha256_16": _hash_tensor(truth.colors),
                           "opacity_sha256_16": _hash_tensor(truth.opacity)},
        "moving_support": {"num_members": int(members.numel()),
                           "members_sha256_16": _hash_tensor(members)},
        "cameras": {"count": ncam,
                    "viewmatrix_sha256_16": [
                        _hash_tensor(c.viewmatrix) for c in cameras],
                    "projmatrix_sha256_16": [
                        _hash_tensor(c.projmatrix) for c in cameras]},
        "observations": {"frames": T, "frame_ids": frame_ids,
                         "rgb_sha256_16": [
                             _hash_tensor(obs_all[t][args.camera][0])
                             for t in range(T)
                             if obs_all[t][args.camera] is not None],
                         "depth_sha256_16": [
                             _hash_tensor(obs_all[t][args.camera][1])
                             for t in range(T)
                             if obs_all[t][args.camera] is not None]},
        "r_eff": {"helper": "estimator.phase2_shared.effective_precision",
                  "model_error": model_error.to_dict()
                  if model_error is not None else None,
                  "used_by": ["phase1_energies", "stage_a_audit",
                              "generic_screw", "exact_P", "exact_R",
                              "heldout"]},
        "renderer": "estimator.renderer_bridge.build_linearization (CUDA)",
        "appearance": "frozen: colors, opacity, scales are never updated",
        "note": ("section 21: identical inputs are a precondition for the "
                 "energies below being comparable at all")})
    _json(out / "run_summary.json", {
        **summary,
        "html": {"joint_motion_debug":
                 str(html_dir / "joint_motion_debug.html"),
                 "definitive_joint_3d":
                 str(html_dir / "definitive_joint_3d.html"),
                 "chosen_camera_all_frames":
                 str(html_dir / "chosen_camera_all_frames.html")},
        "renders": (render_index or {}).get("frames"),
        "depth_scale": (render_index or {}).get("depth_scale")})

    print(f"\n  === {args.scene} DONE ===")
    print(f"  {out}")
    print(f"  3D       {html_dir / 'joint_motion_debug.html'}")
    print(f"  renders  {html_dir / 'chosen_camera_all_frames.html'}")
    for k, v in cuda.stages.items():
        print(f"    {k:<20} {v['peak_cuda_gib']:.2f} GiB  {v['seconds']:.1f} s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
