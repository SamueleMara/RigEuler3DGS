#!/usr/bin/env python
"""Phase 3: per-frame joint displacement on the RGB-D, joint FIXED, F or B.

Joint identification (inside Phase 2) gives the joint type and its geometry:
prismatic direction e_P, or revolute axis e_R with axis point a_R. Phase 3
keeps them exactly as given and estimates, frame by frame, only the joint
displacement and its rate

    x_k = (q_k, qdot_k)        q = s [m] (prismatic) or theta [rad] (revolute)

with an RGB-D IEKF on the original images and the same renderer: the moving
Gaussians are transported by the fixed joint (mu = mu_bar + s e_P, or
mu = a + Q(theta)(mu_bar - a), R = Q(theta) R_bar), the static ones stay in
the render. The update is estimator/joint_iekf.run_window on
estimator/joint_displacement.DisplacementOnlyState; the geometry is checked to
be bitwise unchanged before anything is written.

  * frames in PROCESSING order for --direction, as Phase 2 loads them, with
    that direction's endpoint model: run_window pins its first frame (q = 0),
    so a backward run is anchored at original frame T on the final model and
    q is endpoint-relative;
  * the moving set is the frozen Phase-2 support (support_coherence/support.json);
  * q_k starts from joint identification's track (joint_reduction/<type>/
    trajectory.csv), already in processing order and anchor-relative.

The joint type is an input (--hypotheses), never chosen or changed here. GT is
never read during inference; --evaluate-gt scores the result afterwards into
<out>/evaluation_gt/ only.

    python tools/run_joint_phase3.py --scene fridge_10905 --direction backward \\
        --model <results>/fridge_10905/static_bidi/final \\
        --source <dataset>/fridge_10905 \\
        --phase2-dir <results>/fridge_10905/bidirectional/backward_rigid \\
        --hypotheses revolute --frames 60 --cameras 8 --iters 5 --frames-on-cpu \\
        --out <results>/fridge_10905/bidirectional/backward_joint
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

HYPOTHESES = ("prismatic", "revolute")


def _json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=1, default=float))


def _read_csv_column(path: Path, col: str) -> np.ndarray:
    rows = list(csv.DictReader(open(path)))
    return np.array([float(r[col]) for r in rows])


def _finite_rate(q: np.ndarray, time_s=None) -> np.ndarray:
    """Rate of an initial coordinate track (processing order): d q / d t in q-units
    per SECOND when `time_s` (elapsed seconds, processing order) is given -- the
    production path; per frame index only for callers that pass no times."""
    if q.size < 2:
        return np.zeros_like(q)
    if time_s is None:
        return np.gradient(q)
    t = np.asarray(time_s, float).reshape(-1)
    if t.size != q.size:
        raise ValueError(f"{q.size} coordinates but {t.size} time stamps")
    return np.gradient(q, t)


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--scene", required=True)
    ap.add_argument("--direction", choices=("forward", "backward"), required=True)
    ap.add_argument("--model", required=True,
                    help="static endpoint model of this direction: static_bidi/initial "
                         "(forward) or static_bidi/final (backward)")
    ap.add_argument("--source", required=True, help="dataset scene directory")
    ap.add_argument("--phase2-dir", required=True,
                    help="bidirectional/<direction>_rigid (support + joint_reduction)")
    ap.add_argument("--frames", type=int, default=60)
    ap.add_argument("--cameras", type=int, default=8)
    ap.add_argument("--iters", type=int, default=5, help="IEKF iterations per frame")
    ap.add_argument("--cg-iters", type=int, default=40)
    ap.add_argument("--sigma-qddot", type=float, default=1.0,
                    help="white joint-acceleration noise density, (m or rad)/s^(3/2): "
                         "Q_qq = s^2 dt^3/3, Q_q,qdot = s^2 dt^2/2, Q_qdot,qdot = s^2 dt, dt in s")
    ap.add_argument("--hypotheses", default="prismatic,revolute",
                    help="joint type(s) to run, each with its FIXED joint-identification geometry")
    ap.add_argument("--prior-q", type=float, default=1.0, help="prior sd of q at the anchor (m or rad)")
    ap.add_argument("--prior-qdot", type=float, default=1.0, help="prior sd of qdot (m/s or rad/s)")
    ap.add_argument("--frames-on-cpu", action="store_true")
    from estimator.visibility_constraint import add_cli as _vis_cli
    _vis_cli(ap)
    from estimator.masked_observation import add_cli as _mask_obs_cli
    _mask_obs_cli(ap)
    ap.add_argument("--evaluate-gt", action="store_true",
                    help="after the fits are written, score them against the dataset GT "
                         "(evaluation only, into <out>/evaluation_gt/)")
    ap.add_argument("--out", required=True)
    return ap.parse_args(argv)


def load_initialisation(p2: Path, members, mu0, T: int, time_s=None) -> dict:
    """The joint identification Phase 2 wrote, as P / R parts in processing
    order. Missing or degenerate hypotheses are reported, not invented."""
    from estimator import phase2_prismatic as mp
    from estimator.phase2_model_comparison import init_revolute
    jr = p2 / "joint_reduction"
    parts, notes = {}, {}
    gp, tp = jr / "prismatic" / "geometry.json", jr / "prismatic" / "trajectory.csv"
    if gp.is_file() and tp.is_file():
        g = json.loads(gp.read_text())
        ref = g.get("refined") or {}
        e = np.asarray(ref.get("direction") or g["direction"], float)
        s = _read_csv_column(tp, "s")
        if s.size != T:
            raise SystemExit(f"{tp}: {s.size} frames, the RGB-D window has {T}")
        s = s - s[0]
        parts["prismatic"] = mp.PrismaticPart(members, torch.as_tensor(e), torch.as_tensor(s),
                                              torch.as_tensor(_finite_rate(s, time_s)))
        notes["prismatic"] = {"init": str(gp), "direction": e.tolist(),
                              "s_range_init": float(s.max() - s.min())}
    else:
        notes["prismatic"] = {"init": None, "reason": f"{gp} or {tp} missing"}
    gr, tr = jr / "revolute" / "geometry.json", jr / "revolute" / "trajectory.csv"
    if gr.is_file() and tr.is_file():
        g = json.loads(gr.read_text())
        ref = g.get("refined") or {}
        axis = np.asarray(ref.get("axis") or g["axis"], float)
        point = ref.get("axis_point") or g.get("axis_point")
        th = _read_csv_column(tr, "theta")
        if th.size != T:
            raise SystemExit(f"{tr}: {th.size} frames, the RGB-D window has {T}")
        th = th - th[0]
        if point is None:
            notes["revolute"] = {"init": None, "reason": "no axis point in the joint reduction"}
        else:
            part = init_revolute(mu0.cpu(), members, None, axis=torch.as_tensor(axis),
                                 angles=torch.as_tensor(th), axis_point=torch.as_tensor(point))
            part.thetadot = torch.as_tensor(_finite_rate(th, time_s)).double()
            parts["revolute"] = part
            notes["revolute"] = {"init": str(gr), "axis": axis.tolist(), "axis_point": list(point),
                                 "theta_range_init": float(th.max() - th.min()),
                                 "axis_radius_over_part_scale": g.get("axis_radius_over_part_scale")}
    else:
        notes["revolute"] = {"init": None, "reason": f"{gr} or {tr} missing"}
    return {"parts": parts, "notes": notes}


def geometry_of(name: str, part) -> dict:
    if name == "prismatic":
        return {"direction": part.direction.tolist()}
    return {"axis": part.axis.tolist(), "axis_point": part.axis_point.tolist()
            if hasattr(part, "axis_point") else None, "rho": part.rho.tolist()}


def evaluate_gt(out: Path, source: Path, frame_ids, fits: dict) -> dict:
    """Post-hoc only: line angle, line distance and coordinate error per fit.
    Runs after every inference output is on disk and writes nothing else."""
    from observation_diagnostics import gt_evaluation as G
    gt = G.GroundTruth(source)
    res = {"gt_status": gt.status, "gt_joint": None if gt.joint is None else {
        "type": gt.joint["type"], "axis": gt.joint["axis"].tolist(), "point": gt.joint["point"].tolist()},
        "metric_class": "gt_evaluation_metric", "fits": {}}
    if gt.joint is None or gt.q is None:
        _json(out / "evaluation_gt" / "phase3_gt_evaluation.json", res)
        return res
    orig = np.asarray(frame_ids, int)
    q_gt = gt.q[orig] - gt.q[orig[0]]        # endpoint-relative, processing order
    for name, f in fits.items():
        g = f["geometry"]
        e = np.asarray(g.get("direction") or g.get("axis"), float)
        sgn = float(np.sign(e @ gt.joint["axis"])) or 1.0
        q = np.asarray(f["q"], float) * sgn
        q0 = np.asarray(f["q_init"], float) * sgn
        r = {"hypothesis": name, "matches_gt_type": name == gt.joint["type"],
             "axis_angle_deg": G.line_angle_deg(e, gt.joint["axis"]),
             "coord_mae": float(np.abs(q - q_gt).mean()), "coord_max_abs": float(np.abs(q - q_gt).max()),
             "coord_mae_phase2_init": float(np.abs(q0 - q_gt).mean()),
             "coord_range_est": float(q.max() - q.min()), "coord_range_gt": float(q_gt.max() - q_gt.min()),
             "coord_units": "m" if name == "prismatic" else "rad",
             "note": "coordinate compared only when the hypothesis matches the GT type"}
        if name == "revolute" and g.get("axis_point") is not None:
            r["line_distance_m"] = G.line_distance(np.asarray(g["axis_point"]), e,
                                                   gt.joint["point"], gt.joint["axis"])
        res["fits"][name] = r
    _json(out / "evaluation_gt" / "phase3_gt_evaluation.json", res)
    return res


def main(argv=None) -> int:
    args = parse_args(argv)
    from estimator.joint_iekf import JointIEKFConfig
    from estimator.model_error import estimate_model_error
    from estimator.joint_displacement import DisplacementOnlyState, geometry_snapshot, geometry_unchanged
    from estimator.phase2_model_comparison import fit_model, prismatic_state, revolute_state
    from estimator.phase2_shared import effective_precision, render_state
    from estimator.renderer_bridge import build_linearization
    from estimator.sparse_iekf import IEKFConfig
    from tools.run_iekf_scene import camera_specs, dataset_rgbd_frames, load_scene, state_from_gaussians
    from utils.sequence_direction import context_from_frame_ids, verify_endpoint_provenance

    t_start = time.time()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    p2 = Path(args.phase2_dir)
    model, source = Path(args.model), Path(args.source)
    from estimator.masked_observation import preflight as _obs_preflight
    _obs_preflight(args, source)            # masks ready? before any model load / GPU work

    # ---- the frozen Phase-2 support, and that it is this direction's -----
    sup = json.loads((p2 / "support_coherence" / "support.json").read_text())
    src_dir = Path(sup.get("source", "")).parent.name
    if src_dir and src_dir != args.direction:
        raise SystemExit(f"{p2}: its support came from '{src_dir}', not --direction {args.direction}")
    members = torch.as_tensor(sup["members"], dtype=torch.long)

    # ---- scene and RGB-D in processing order (identical to Phase 2) ------
    # inference_only: the loader would otherwise attach GT part labels to the model
    dataset, pipe, gaussians, sc = load_scene(str(model), str(source), inference_only=True,
                                            camera_device="cpu")
    cams_all = sc.getTrainCameras()
    ncam = min(args.cameras, len(cams_all))
    cameras = camera_specs(cams_all, ncam)
    truth = state_from_gaussians(gaussians)
    cfg = IEKFConfig()
    ns = argparse.Namespace(
        sequence_start=0, sequence_length=args.frames, all_frames=False,
        observation_source="dataset_rgbd", cameras=ncam, bootstrap_iters=0, supernode_size=64,
        seed=0, frames_on_cpu=bool(args.frames_on_cpu), frames_on_gpu=not args.frames_on_cpu,
        depth_sigma=None, joint_direct_rgbd=False, joint_direct_gaussian=False, window_length=0,
        all_cameras=False, dt=1.0 / 30.0, max_gaussians=0, no_model_error=False,
        model_error_beta=1.0, model_error_blur=1)
    frames_raw, frame_ids = dataset_rgbd_frames(cams_all[:ncam], cfg, ns, truth, source,
                                                direction=args.direction)
    ctx = context_from_frame_ids(frame_ids)
    endpoint = verify_endpoint_provenance(args.direction, str(model), total_frames=len(frames_raw),
                                          strict=True)
    obs_all = [[(o[0], o[1], o[2] if len(o) > 2 else None) if o is not None else None for o in ob]
               for (ob, _s, _d) in frames_raw]
    T = len(obs_all)
    # elapsed seconds along the PROCESSING order from the loader's frame interval:
    # the window prediction and the initial qdot are both per second
    from utils.sequence_direction import processing_time_s
    time_s = processing_time_s(frames_raw, frame_ids)
    if time_s.numel() != T:
        raise SystemExit(f"{time_s.numel()} time stamps for {T} frames")
    mu0, R0 = truth.mu.double(), truth.R.double()
    if int(members.max()) >= mu0.shape[0]:
        raise SystemExit(f"support indexes row {int(members.max())}, the model has {mu0.shape[0]}")
    orig_frames = [int(f["original_frame_index"]) if isinstance(f, dict) else int(f)
                   for f in frame_ids["frames"]]
    print(f"  {args.scene} {ctx.direction}: canonical endpoint {endpoint.get('observed_endpoint')} "
          f"(original frame {orig_frames[0]}), {mu0.shape[0]} Gaussians, {members.numel()} moving, "
          f"{ncam} cameras, {T} frames")

    # ---- measurement model: Phase 2's, unchanged -------------------------
    dev = mu0.device
    # masked observation mode: same composite target / background / foreground depth
    # and model-error estimate as Phase 2 (estimator/masked_observation.py)
    from estimator.masked_observation import from_args as _masked_from_args
    from estimator.phase2_shared import PreparedFrames, effective_precision_masked, measurement_terms
    mseq = _masked_from_args(args, source, frames_raw, frame_ids, cfg)
    from estimator.masked_observation import observation_manifest
    _json(out / "observation_mode.json", observation_manifest(args, mseq, cfg))
    bg = None if mseq is None else mseq.model.background_tensor(dev)
    prepared = None if mseq is None else PreparedFrames(mseq, obs_all)
    if mseq is None:
        ref_obs = [(o[0].to(dev), o[1].to(dev), o[2].to(dev) if o[2] is not None else None)
                   for o in obs_all[0] if o is not None]
    else:
        ref_obs = [(o.target(bg), o.depth_observed, o.depth_valid)
                   for o in (ob.to(dev) for ob in prepared[0])]
    lin0 = build_linearization(truth, cameras, eps_depth=cfg.eps_depth, background=bg)
    model_error = estimate_model_error(lin0, ref_obs, reference_index=0, beta=1.0, blur=1)
    from estimator.model_error import model_error_audit
    _json(out / "model_error_audit.json", model_error_audit(
        lin0, model_error, cfg.rgb_sigma, masks=None if mseq is None else mseq.masks[0]))
    del lin0
    cam_ids = list(range(ncam))
    obs_pairs = [[(o[0], o[1]) if o is not None else None for o in ob] for ob in obs_all]

    init = load_initialisation(p2, members, mu0, T, time_s=time_s.numpy())
    wanted = [h.strip() for h in args.hypotheses.split(",") if h.strip()]
    jcfg = JointIEKFConfig(max_iters=args.iters, max_cg_iters=args.cg_iters,
                           sigma_qddot=args.sigma_qddot)
    fits, summary_h = {}, {}
    for name in wanted:
        part = init["parts"].get(name)
        if part is None:
            summary_h[name] = {"status": "NOT_FITTED", **init["notes"].get(name, {})}
            continue
        full = prismatic_state(part) if name == "prismatic" else revolute_state(part)
        # the joint type and geometry are FIXED: only (q_k, qdot_k) is estimated
        state = DisplacementOnlyState(full, prior_scale=(args.prior_q ** 2, args.prior_qdot ** 2))
        g_init = geometry_of(name, state.parts[0])
        g_snap = geometry_snapshot(state.parts[0])
        q_init = (part.s if name == "prismatic" else part.theta).clone().tolist()
        ctr = {"t": 0}

        def lin(st, t, _ctr=ctr):
            _ctr["t"] = t
            mu, Rg = st.gaussians(t, mu0, R0)
            return render_state(truth, mu, Rg, cameras, cfg, build_linearization, background=bg)

        def prec(l, _ctr=ctr):
            # CameraTerms (phase2_shared.measurement_terms): same policy as Phase 2;
            # the opt-in hard negative-ray mode keeps its legacy precision tuples
            if getattr(args, "object_mask_constraint", "off") != "off":
                return effective_precision(l, obs_all[_ctr["t"]], cam_ids, cfg, model_error)
            return measurement_terms(l, prepared[_ctr["t"]] if mseq is not None else obs_all[_ctr["t"]],
                                     cam_ids, cfg, model_error, None if mseq is None else mseq.model)

        print(f"\n  Phase 3 {name}: {args.iters} iterations per frame")
        t0 = time.time()
        from estimator.visibility_constraint import from_args as _vis_from_args
        vis_seq = _vis_from_args(args, source, frames_raw, frame_ids)
        fit = fit_model(state, mu0, obs_pairs if mseq is None else prepared, lin, prec, jcfg, label=name,
                        times=time_s,
                        visibility=vis_seq, observation_model=None if mseq is None else mseq.model)
        if mseq is not None:
            _json(out / name / "masked_observation.json", {
                **mseq.model.to_dict(), "mask_root": mseq.mask_root, "cameras": mseq.cameras,
                "rgb_precision": "1/(rgb_sigma^2 + model-error variance), fixed, no Student-t weight",
                "rgb_sigma": cfg.rgb_sigma, "coverage": mseq.counts(),
                "frames": [{"processing_index": k, "energy_before": r.energy_before,
                            "energy_after": r.energy_after, "iterations": r.iterations,
                            "accepted": r.accepted} for k, r in enumerate(fit["records"])]})
        if vis_seq is not None:
            vrows = [{"processing_index": k, **{kk: vv for kk, vv in (r.observability.get("visibility") or {}).items()
                                                if kk != "iterations"}}
                     for k, r in enumerate(fit["records"]) if r.observability.get("visibility")]
            _json(out / name / "visibility_constraint.json", {"policy": vis_seq.policy.to_dict(), "mask_root": getattr(vis_seq, "mask_root", None), "frames": vrows})
        recs = fit["records"]
        _json(out / name / "map_objective.json", {"frames": [
            {"processing_index": k, "frame": r.frame, "energy_before": r.energy_before,
             "energy_after": r.energy_after, "accepted": r.accepted, "iterations_run": r.iterations,
             "objective": r.observability.get("objective"), "iterations": r.observability.get("iterations")}
            for k, r in enumerate(recs)]})
        p = state.parts[0]
        q = (p.s if name == "prismatic" else p.theta).tolist()
        qd = (p.sdot if name == "prismatic" else p.thetadot).tolist()
        if not geometry_unchanged(g_snap, p):
            raise SystemExit(f"{name}: joint geometry changed during Phase 3 -- refusing to write")
        d = out / name
        d.mkdir(parents=True, exist_ok=True)
        with open(d / "trajectory.csv", "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["processing_index", "original_frame", "q", "qdot", "q_phase2_init",
                        "posterior_sd_q", "energy_before", "energy_after", "iterations", "accepted"])
            for k, r in enumerate(recs):
                w.writerow([k, orig_frames[k], q[k], qd[k], q_init[k], float(r.posterior[0, 0]) ** 0.5,
                            r.energy_before, r.energy_after, r.iterations, r.accepted])
        e_tot = float(sum(r.energy_after for r in recs))
        fits[name] = {"q": q, "q_init": q_init, "geometry": g_init}
        summary_h[name] = {
            "status": "FITTED", "joint_type": name, "geometry_fixed": g_init,
            "geometry_unchanged_bitwise": True,
            "coordinate_units": "m" if name == "prismatic" else "rad",
            "coordinate_range": float(max(q) - min(q)),
            "energy_total_after": e_tot,
            "energy_total_before": float(sum(r.energy_before for r in recs)),
            "frames_with_accepted_steps": int(sum(r.accepted > 0 for r in recs)),
            "coordinate_range_phase2_init": float(max(q_init) - min(q_init)),
            "max_abs_change_from_phase2_q": float(max(abs(a - b) for a, b in zip(q, q_init))),
            "posterior_sd_q_last": float(recs[-1].posterior[0, 0]) ** 0.5,
            "seconds": round(time.time() - t0, 1), **init["notes"].get(name, {})}
        _json(d / "geometry.json", summary_h[name])
        # GT-free mask diagnostics of the fitted displacement (evaluation only)
        from estimator.mask_diagnostics import MaskDiagnostics, masks_for_run
        _mfn, _mroot, _mwhy = masks_for_run(args, source, frames_raw, frame_ids, mseq)
        mdiag = MaskDiagnostics(f"phase3/{name}", _mfn, _mroot, cameras, cfg, truth, build_linearization,
                                reason=_mwhy)
        if mdiag.available:
            for t in range(T):
                mu_t, R_t = state.gaussians(t, mu0, R0)
                mdiag.frame(t, orig_frames[t], mu_t, R_t, obs_all[t])
        _json(d / "mask_diagnostics.json", mdiag.report())
        print(f"    {name}: energy {summary_h[name]['energy_total_before']:.4e} -> {e_tot:.4e}, "
              f"range {summary_h[name]['coordinate_range']:.4f}, "
              f"{summary_h[name]['frames_with_accepted_steps']}/{T} frames moved")

    fitted = {k: v for k, v in summary_h.items() if v.get("status") == "FITTED"}
    summary = {
        "scene": args.scene, "direction": args.direction, "stage": "phase3_joint_displacement_rgbd_iekf",
        "estimated": "q_k and qdot_k per frame only; joint type and geometry fixed from joint identification",
        "canonical_model": str(model), "canonical_original_frame": orig_frames[0],
        "frame_order": "processing (backward = T..0); q is relative to the canonical frame",
        "original_frames_processing_order": orig_frames,
        "support": {"file": str(p2 / "support_coherence" / "support.json"), "members": int(members.numel())},
        "config": {"iters": args.iters, "cg_iters": args.cg_iters, "sigma_qddot": args.sigma_qddot,
                   "sigma_qddot_units": "(m or rad)/s^(3/2), white-acceleration density",
                   "dt_s": float(time_s[1] - time_s[0]) if T > 1 else None,
                   "time_s_processing_order": time_s.tolist(),
                   "rate_units": "qdot in m/s (prismatic) or rad/s (revolute)",
                   "cameras": ncam, "frames": T,
                   "prior_sd": {"q": args.prior_q, "qdot": args.prior_qdot}},
        "hypotheses": summary_h,
        "energy_difference_P_minus_R": (fitted["prismatic"]["energy_total_after"]
                                        - fitted["revolute"]["energy_total_after"])
        if {"prismatic", "revolute"} <= set(fitted) else None,
        "joint_types_run": [k for k in fitted],
        "type_note": ("the type is an input (--hypotheses), never changed here; with both, each "
                      "hypothesis keeps its own fixed joint-identification geometry"),
        "gt_used_in_inference": False,
        "runtime_s": round(time.time() - t_start, 1)}
    _json(out / "summary.json", summary)
    from estimator.mask_diagnostics import resources
    _json(out / "stage_resources.json", {"stage": "phase3", **resources(time.time() - t_start)})
    print(f"\n  wrote {out / 'summary.json'}  ({summary['runtime_s'] / 60:.1f} min)")
    if args.evaluate_gt:
        r = evaluate_gt(out, source, orig_frames, fits)
        for name, x in r.get("fits", {}).items():
            print(f"  [GT eval] {name}: axis {x['axis_angle_deg']:.3f} deg"
                  + (f", line {x['line_distance_m']:.4f} m" if "line_distance_m" in x else "")
                  + (f", coord MAE {x['coord_mae']:.4f} {x['coord_units']} (Phase-2 track "
                     f"{x['coord_mae_phase2_init']:.4f})" if x["matches_gt_type"] else ""))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
