#!/usr/bin/env python
"""Stage A runner (sections 8, 12, 13): audit the fitted generic rigid screw.

CODEX_VALIDATE_GENERIC_TWIST_THEN_PARALLEL_PRISMATIC_REVOLUTE_PHASE2.md.

    python tools/phase2_rigid_audit.py --scene storage_45135

Runs the generic Phase-2 rigid-screw IEKF unchanged (section 2), then renders
four states from that SAME solution -- static, translation-only,
rotation-only, full rigid -- through the SAME measurement model, and reports
what each one buys.

Produces section 8's tree under `<out>/phase2_rigid_motion_audit/`.
"""

from __future__ import annotations

import argparse
import csv
import dataclasses
import json
import sys
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from estimator.rigid_motion_audit import (  # noqa: E402
    GEOMETRY_SCORE_HEADER, PER_FRAME_CAMERA_HEADER, STATES, audit_frame,
    build_state, com_decomposition, geometry_score_per_frame,
    per_frame_camera_rows, rigidity_check, summarise)


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scene", required=True)
    ap.add_argument("--model", default=None)
    ap.add_argument("--source", default=None)
    ap.add_argument("--frames", type=int, default=20)
    ap.add_argument("--cameras", type=int, default=4)
    ap.add_argument("--max-gaussians", type=int, default=6000)
    ap.add_argument("--screw-iters", type=int, default=8)
    ap.add_argument("--bootstrap-iters", type=int, default=6)
    ap.add_argument("--out", default=None)
    ap.add_argument("--render-sequences", action="store_true",
                    help="section 12: write the six fixed-camera RGB/depth "
                         "sequences as PNGs under renders/ and an HTML index")
    ap.add_argument("--render-stride", type=int, default=1,
                    help="render every Nth frame (section 12)")
    args = ap.parse_args()

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

    out = Path(args.out or f"output/phase2_audit_{args.scene}")
    adir = out / "phase2_rigid_motion_audit"
    adir.mkdir(parents=True, exist_ok=True)
    print(f"=== Stage A: {args.scene} ===")

    # ---- run the generic Phase 2, UNCHANGED (section 2) ------------------
    import subprocess
    cmd = [sys.executable, "tools/run_iekf_scene.py",
           "--model", model, "--source", source,
           "--observation-source", "dataset_rgbd",
           "--sequence-start", "0",
           "--sequence-length", str(args.frames),
           "--cameras", str(args.cameras),
           "--max-gaussians", str(args.max_gaussians),
           "--supernodes", "--window-discovery", "--num-moving-parts", "1",
           "--phase2-rigid-screw-iekf",
           "--phase2-screw-iters", str(args.screw_iters),
           "--bootstrap-iters", str(args.bootstrap_iters),
           "--discovery-outer", "1", "--no-visualizations",
           "--out", str(out)]
    print("  running the generic rigid-screw Phase 2 ...")
    r = subprocess.run(cmd, cwd=REPO, capture_output=True, text=True)
    if r.returncode != 0:
        print(r.stdout[-3000:])
        print(r.stderr[-3000:])
        raise SystemExit("the generic Phase 2 failed; Stage A cannot audit it")
    for line in r.stdout.split("\n"):
        if any(k in line for k in ("RIGID-SCREW", "efficient P-null",
                                   "peak CUDA", "S_curv")):
            print("   " + line.strip())

    # ---- rebuild the SAME solution and audit it --------------------------
    from estimator.joint_runtime import initial_twist_from_phase1
    from estimator.joint_state import JointState, initialise_from_twist
    from estimator.measurement import build_precision
    from estimator.renderer_bridge import build_linearization
    from estimator.sparse_iekf import IEKFConfig
    from tools.run_iekf_scene import (camera_specs, dataset_rgbd_frames,
                                      load_scene, state_from_gaussians)

    summary = json.loads((out / "phase2_rigid_screw" / "summary.json").read_text())
    cfgj = json.loads((out / "phase2_rigid_screw" / "config.json").read_text())
    sup = json.loads((out / "phase2_rigid_screw" / "frozen_support.json").read_text())
    geom = summary["geometry"]

    dataset, pipe, gaussians, sc = load_scene(model, source)
    cams_all = sc.getTrainCameras()
    ncam = min(args.cameras, len(cams_all))
    cameras = camera_specs(cams_all, ncam)
    truth = state_from_gaussians(gaussians)
    cfg = IEKFConfig()
    ns = argparse.Namespace(sequence_start=0, sequence_length=args.frames,
                            observation_source="dataset_rgbd",
                            depth_sigma="constant:0.05:0.0",
                            frames_on_cpu=True, no_rgb=False, no_depth=False,
                            all_frames=False, all_cameras=False,
                            cameras=ncam, window_length=0, dt=1.0 / 30.0)
    frames, _fid = dataset_rgbd_frames(cams_all[:ncam], cfg, ns, truth,
                                       Path(source))
    obs = [[(o[0], o[1]) if o is not None else None for o in ob]
           for (ob, _s, _d) in frames]
    mu0 = truth.mu.double()

    model_error = None
    me_path = out / "model_error.json"
    members = torch.tensor(sup["members"], dtype=torch.long)
    R0 = truth.R.double()
    scales = truth.scales.double()
    part = initialise_from_twist(members, mu0,
                                 torch.tensor(geom["xi"], dtype=torch.float64),
                                 torch.tensor(summary["q"],
                                              dtype=torch.float64), None)
    part.zeta = torch.tensor(geom["zeta"], dtype=torch.float64)
    part.q = torch.tensor(summary["q"], dtype=torch.float64)
    part.qdot = torch.tensor(summary["qdot"], dtype=torch.float64)
    state = JointState([part], torch.arange(len(obs), dtype=torch.float64))

    # Section 2: the Gaussian ORIENTATIONS must reach the rasterizer. This
    # helper previously accepted `Rg` and dropped it, so every state was
    # rendered with canonical covariances and a rotation-only state was
    # indistinguishable from a translation of the same centers.
    from estimator.phase2_shared import render_state

    def build_lin(mu, Rg):
        return render_state(truth, mu, Rg, cameras, cfg, build_linearization)

    def precision_fn(lin, frame=None):
        # Section 4: the SAME R_eff the production runtime uses. The local
        # rebuild below is kept only as the no-model-error fallback.
        from estimator.phase2_shared import effective_precision
        t = frame if frame is not None else 0
        if model_error is not None:
            return effective_precision(lin, obs[t], list(range(ncam)), cfg,
                                       model_error)
        o = obs[t]
        res = []
        for v, ob in zip(lin.views, o):
            if ob is None:
                res.append((torch.zeros_like(v.rgb), torch.zeros_like(v.depth)))
                continue
            pr = build_precision(ob[0].to(v.rgb.device) - v.rgb,
                                 valid_mask=None, sigma_rgb=cfg.rgb_sigma,
                                 robust=cfg.robust_measurement,
                                 nu=cfg.student_t_nu,
                                 layout=("rgb", "rgb", "rgb"), channel_dim=0)
            m = (v.alpha > cfg.min_alpha_for_depth)
            pd = build_precision((ob[1].to(v.depth.device)
                                  - v.depth).unsqueeze(0),
                                 valid_mask=m.unsqueeze(0), sigma_rgb=(),
                                 depth_sigma_model=cfg.depth_sigma_model,
                                 robust=cfg.robust_measurement,
                                 nu=cfg.student_t_nu, layout=("depth",),
                                 channel_dim=0, use_rgb=False).squeeze(0)
            res.append((pr, pd))
        return res

    print(f"  auditing {len(obs)} frames x {ncam} cameras, four states each")
    audits = []
    for t in range(len(obs)):
        audits.append(audit_frame(
            part, t, mu0, R0, obs[t], truth, cameras, build_lin,
            lambda lin, _t=t: precision_fn(lin, _t), cfg.min_alpha_for_depth))

    rows = per_frame_camera_rows(audits)
    summ = summarise(audits, rows)
    with (adir / "per_frame_camera.csv").open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(PER_FRAME_CAMERA_HEADER)
        w.writerows(rows)

    # per-frame and per-camera aggregates
    def agg(key_index):
        groups: dict = {}
        for r in rows:
            groups.setdefault(r[key_index], []).append(r)
        out_rows = []
        for k in sorted(groups):
            g = groups[k]
            out_rows.append([k] + [sum(r[i] for r in g)
                                   for i in range(2, len(PER_FRAME_CAMERA_HEADER))])
        return out_rows

    for name, ki in (("per_frame.csv", 0), ("per_camera.csv", 1)):
        with (adir / name).open("w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow([PER_FRAME_CAMERA_HEADER[ki]]
                       + PER_FRAME_CAMERA_HEADER[2:])
            w.writerows(agg(ki))

    # ---- sections 9, 10, 11 ---------------------------------------------
    phys = [{"frame": t,
             "com_displacement": a["decomposition"]["com_displacement"].tolist(),
             "com_distance": float(a["decomposition"]["com_displacement"].norm()),
             "phi": a["decomposition"]["phi"].tolist(),
             "theta_rad": a["decomposition"]["theta"],
             "R": a["decomposition"]["R"].tolist()}
            for t, a in enumerate(audits)]
    rig = [rigidity_check(part, t, mu0, R0, scales=scales)
           for t in (0, len(obs) // 2, len(obs) - 1)]

    counter = {"frame": 0}

    def lin_for_state(st, frame):
        counter["frame"] = frame
        mu, Rg = build_state("full", st.parts[0], frame, mu0, R0)
        return build_lin(mu, Rg)

    gsc = geometry_score_per_frame(
        state, mu0, R0, obs, lin_for_state,
        lambda lin: precision_fn(lin, counter["frame"]))
    with (adir / "geometry_score.csv").open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(GEOMETRY_SCORE_HEADER)
        w.writerows(gsc["rows"])

    (adir / "twist_geometry.json").write_text(json.dumps({
        "geometry": geom, "q": summary["q"], "qdot": summary["qdot"],
        "physical_per_frame": phys,
        "rigidity_check": rig,
        "geometry_score": {k: v for k, v in gsc.items()
                           if k not in ("rows", "g_per_frame", "F_total")},
    }, indent=1, default=str))
    (adir / "summary.json").write_text(json.dumps({
        "scene": args.scene, "frames": len(obs), "cameras": ncam,
        "num_moving_gaussians": int(members.numel()),
        "contribution": summ,
        "rigidity": rig,
        "geometry_support": {k: v for k, v in gsc.items()
                             if k not in ("rows", "g_per_frame", "F_total")},
        "branch": cfgj.get("branch"),
    }, indent=1, default=str))

    print(f"\n  full gain over static        {summ['delta_full']:.6e}")
    print(f"  translation-only gain        {summ['delta_trans']:.6e}"
          f"   ({100 * summ['translation_only_share']:.1f}% of full)")
    print(f"  rotation-only gain           {summ['delta_rot']:.6e}")
    print(f"  rotation AFTER translation   {summ['delta_rot_given_trans']:.6e}"
          f"   ({100 * summ['rotation_share_of_full_gain']:.1f}% of full)")
    print(f"  translation AFTER rotation   {summ['delta_trans_given_rot']:.6e}"
          f"   ({100 * summ['translation_share_of_full_gain']:.1f}% of full)")
    print(f"  rotation share spread across cameras "
          f"{summ['rotation_share_spread_across_cameras']:.4f}")
    print(f"  max |c_t - c_0| {summ['max_com_displacement']:.5f}, "
          f"max theta {summ['max_theta_rad']:.5f} rad")
    print(f"  rigidity: max relative pair drift "
          f"{max(r['max_rel_drift'] for r in rig):.3e}")
    print(f"  orientation transport error "
          f"{max(r['max_orientation_transport_error'] for r in rig):.3e}, "
          f"covariance transport error "
          f"{max(r['max_covariance_transport_error'] for r in rig):.3e}")
    print(f"  one constant geometry: median pairwise score cosine "
          f"{gsc['median_pairwise_cosine']:.4f}, "
          f"{100 * gsc['fraction_opposed']:.1f}% of frame pairs opposed")
    # ---- section 12: fixed-camera sequences + the HTML index -----------
    if args.render_sequences:
        from estimator.rigid_motion_renders import render_sequences, write_html
        print(f"\n  section 12: rendering six states x {ncam} cameras")
        # Phase 1's own per-Gaussian solution is a row in its own right: it is
        # what Phase 2 has to beat, and showing it next to the rigid states is
        # the only way to see whether the rigid constraint is what costs the
        # fit or whether Phase 1 was already wrong.
        p1_mu = None
        for cand in ("phase1_positions.pt", "positions.pt",
                     "iekf_positions.pt"):
            q = out / cand
            if q.exists():
                p1_mu = torch.load(q, map_location=mu0.device)
                break
        if p1_mu is None:
            # Phase 1's trajectory was not saved by this run; the row is left
            # out of the grid rather than silently filled with another state,
            # which would make the comparison a lie.
            print("  section 12: no Phase-1 positions on disk; "
                  "the Phase1 row will be marked 'not rendered'")
        idx = render_sequences(adir, part, mu0, R0, obs, truth, cameras,
                               build_lin, phase1_positions=p1_mu,
                               frame_stride=args.render_stride)
        html = write_html(adir, idx, {
            "scene": args.scene, "contribution": summ, "rigidity": rig,
            "geometry_support": {k: v for k, v in gsc.items()
                                 if k not in ("rows", "g_per_frame",
                                              "F_total")}},
            f"Stage A - {args.scene}")
        print(f"  section 12: {len(idx['files'])} PNGs, index at {html}")

    print(f"\n  -> {adir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
