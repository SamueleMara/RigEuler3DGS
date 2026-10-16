#!/usr/bin/env python
"""Phase 3: joint-specific RGB-D IEKF, forward OR backward, from Phase 2.

The framework's Phase 3 (Bidirectional_Articulated_3DGS_IEKF_Final_Framework,
section 8): one filter per joint hypothesis, on the ORIGINAL RGB-D and the
same renderer, with state

    P:  (s_k, sdot_k, e_P)                     moving mu = mu_bar + s e_P
    R:  (theta_k, thetadot_k, e_R, a_R)        mu = a + Q(theta)(mu_bar - a),
                                               R = Q(theta) R_bar

and time-invariant geometry corrected directly by the image innovation.

The filters are the existing exact P / R models (estimator/phase2_prismatic,
phase2_revolute) driven by estimator/joint_iekf.run_window. What this runner
adds is the bidirectional plumbing the legacy debug tool
(tools/run_complete_joint_debug.py) never had:

  * the frames are loaded in PROCESSING order for --direction, exactly as
    Phase 2 loads them, with the endpoint model of that direction. run_window
    pins its first frame as the gauge (q = 0), so for a backward run the
    canonical geometry is the FINAL endpoint at original frame T and the joint
    coordinate is endpoint-relative (section 8.1) -- not frame 0 of a model
    that was trained at T;
  * the support is the frozen Phase-2 support (support_coherence/support.json),
    not a re-run of Phase 1;
  * both hypotheses are initialised from the joint identification that Phase 2
    wrote (joint_reduction/{prismatic,revolute}/{geometry,trajectory}), which
    is already in processing order and anchor-relative.

Both hypotheses are fitted and reported; NO type is chosen (section 9: the
fits are evidence, a lower residual alone is not a verdict). Inference never
reads ground truth. --evaluate-gt scores the finished fits afterwards, into
<out>/evaluation_gt/ only.

    python tools/run_joint_phase3.py --scene fridge_10905 --direction backward \
        --model <results>/fridge_10905/static_bidi/final \
        --source <dataset>/fridge_10905 \
        --phase2-dir <results>/fridge_10905/bidirectional/backward_rigid \
        --frames 60 --cameras 8 --iters 5 --frames-on-cpu \
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


def _finite_rate(q: np.ndarray) -> np.ndarray:
    """Per-frame rate of an initial coordinate track (processing order)."""
    if q.size < 2:
        return np.zeros_like(q)
    return np.gradient(q)


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
                    help="white-acceleration noise on the joint rate, per frame^2")
    ap.add_argument("--hypotheses", default="prismatic,revolute")
    ap.add_argument("--frames-on-cpu", action="store_true")
    ap.add_argument("--evaluate-gt", action="store_true",
                    help="after the fits are written, score them against the dataset GT "
                         "(evaluation only, into <out>/evaluation_gt/)")
    ap.add_argument("--out", required=True)
    return ap.parse_args(argv)


def load_initialisation(p2: Path, members, mu0, T: int) -> dict:
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
                                              torch.as_tensor(_finite_rate(s)))
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
            part.thetadot = torch.as_tensor(_finite_rate(th)).double()
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
        r = {"hypothesis": name, "matches_gt_type": name == gt.joint["type"],
             "axis_angle_deg": G.line_angle_deg(e, gt.joint["axis"]),
             "coord_mae": float(np.abs(q - q_gt).mean()), "coord_max_abs": float(np.abs(q - q_gt).max()),
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

    # ---- the frozen Phase-2 support, and that it is this direction's -----
    sup = json.loads((p2 / "support_coherence" / "support.json").read_text())
    src_dir = Path(sup.get("source", "")).parent.name
    if src_dir and src_dir != args.direction:
        raise SystemExit(f"{p2}: its support came from '{src_dir}', not --direction {args.direction}")
    members = torch.as_tensor(sup["members"], dtype=torch.long)

    # ---- scene and RGB-D in processing order (identical to Phase 2) ------
    # inference_only: the loader would otherwise attach GT part labels to the model
    dataset, pipe, gaussians, sc = load_scene(str(model), str(source), inference_only=True)
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
    ref_obs = [(o[0].to(dev), o[1].to(dev), o[2].to(dev) if o[2] is not None else None)
               for o in obs_all[0] if o is not None]
    lin0 = build_linearization(truth, cameras, eps_depth=cfg.eps_depth)
    model_error = estimate_model_error(lin0, ref_obs, reference_index=0, beta=1.0, blur=1)
    del lin0
    cam_ids = list(range(ncam))
    obs_pairs = [[(o[0], o[1]) if o is not None else None for o in ob] for ob in obs_all]

    init = load_initialisation(p2, members, mu0, T)
    wanted = [h.strip() for h in args.hypotheses.split(",") if h.strip()]
    jcfg = JointIEKFConfig(max_iters=args.iters, max_cg_iters=args.cg_iters,
                           sigma_qddot=args.sigma_qddot)
    fits, summary_h = {}, {}
    for name in wanted:
        part = init["parts"].get(name)
        if part is None:
            summary_h[name] = {"status": "NOT_FITTED", **init["notes"].get(name, {})}
            continue
        state = prismatic_state(part) if name == "prismatic" else revolute_state(part)
        g_init = geometry_of(name, state.parts[0])
        ctr = {"t": 0}

        def lin(st, t, _ctr=ctr):
            _ctr["t"] = t
            mu, Rg = st.gaussians(t, mu0, R0)
            return render_state(truth, mu, Rg, cameras, cfg, build_linearization)

        def prec(l, _ctr=ctr):
            return effective_precision(l, obs_all[_ctr["t"]], cam_ids, cfg, model_error)

        print(f"\n  Phase 3 {name}: {args.iters} iterations per frame")
        t0 = time.time()
        fit = fit_model(state, mu0, obs_pairs, lin, prec, jcfg, label=name)
        recs = fit["records"]
        p = state.parts[0]
        q = (p.s if name == "prismatic" else p.theta).tolist()
        qd = (p.sdot if name == "prismatic" else p.thetadot).tolist()
        g_fin = geometry_of(name, p)
        d = out / name
        d.mkdir(parents=True, exist_ok=True)
        with open(d / "trajectory.csv", "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["processing_index", "original_frame", "q", "qdot", "energy_before",
                        "energy_after", "iterations", "accepted"])
            for k, r in enumerate(recs):
                w.writerow([k, orig_frames[k], q[k], qd[k], r.energy_before, r.energy_after,
                            r.iterations, r.accepted])
        e_tot = float(sum(r.energy_after for r in recs))
        fits[name] = {"q": q, "geometry": g_fin}
        summary_h[name] = {
            "status": "FITTED", "geometry_init": g_init, "geometry_final": g_fin,
            "coordinate_units": "m" if name == "prismatic" else "rad",
            "coordinate_range": float(max(q) - min(q)),
            "energy_total_after": e_tot,
            "energy_total_before": float(sum(r.energy_before for r in recs)),
            "frames_with_accepted_steps": int(sum(r.accepted > 0 for r in recs)),
            "posterior_geometry_covariance_diag_last": torch.diag(recs[-1].posterior)[2:].tolist(),
            "seconds": round(time.time() - t0, 1), **init["notes"].get(name, {})}
        _json(d / "geometry.json", summary_h[name])
        print(f"    {name}: energy {summary_h[name]['energy_total_before']:.4e} -> {e_tot:.4e}, "
              f"range {summary_h[name]['coordinate_range']:.4f}, "
              f"{summary_h[name]['frames_with_accepted_steps']}/{T} frames moved")

    fitted = {k: v for k, v in summary_h.items() if v.get("status") == "FITTED"}
    summary = {
        "scene": args.scene, "direction": args.direction, "stage": "phase3_joint_specific_rgbd_iekf",
        "canonical_model": str(model), "canonical_original_frame": orig_frames[0],
        "frame_order": "processing (backward = T..0); q is relative to the canonical frame",
        "original_frames_processing_order": orig_frames,
        "support": {"file": str(p2 / "support_coherence" / "support.json"), "members": int(members.numel())},
        "config": {"iters": args.iters, "cg_iters": args.cg_iters, "sigma_qddot_per_frame2": args.sigma_qddot,
                   "dt": "1 frame", "cameras": ncam, "frames": T,
                   "prior": "diag(1) on every error coordinate (estimator/phase2_model_comparison.py)"},
        "hypotheses": summary_h,
        "energy_difference_P_minus_R": (fitted["prismatic"]["energy_total_after"]
                                        - fitted["revolute"]["energy_total_after"])
        if {"prismatic", "revolute"} <= set(fitted) else None,
        "decided_type": "UNDECIDED",
        "decision_note": ("both hypotheses are fitted on the same RGB-D; no type is chosen here. "
                          "Compare geometry plausibility (axis radius / part scale) and held-out "
                          "evidence, not the raw energy alone (framework sec. 7, 9)"),
        "gt_used_in_inference": False,
        "runtime_s": round(time.time() - t_start, 1)}
    _json(out / "summary.json", summary)
    print(f"\n  wrote {out / 'summary.json'}  ({summary['runtime_s'] / 60:.1f} min)")
    if args.evaluate_gt:
        r = evaluate_gt(out, source, orig_frames, fits)
        for name, x in r.get("fits", {}).items():
            print(f"  [GT eval] {name}: axis {x['axis_angle_deg']:.3f} deg"
                  + (f", line {x['line_distance_m']:.4f} m" if "line_distance_m" in x else "")
                  + (f", coord MAE {x['coord_mae']:.4f} {x['coord_units']}" if x["matches_gt_type"] else ""))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
