#!/usr/bin/env python
"""Sections 3, 36, 50, 55, 57: the deep stagewise debug + theory audit runner.

CODEX_DEEP_STAGEWISE_DEBUG_AND_THEORY_AUDIT.md.

    python tools/deep_debug.py --run output/geometric_all/storage_45135

Produces `<run>/deep_debug/` as section 3 lays it out. The runner instruments;
it does not tune (section 1). Nothing it writes feeds back into a decision.

It works from cached artifacts (section 58). Quantities that genuinely need a
live renderer -- per-camera RGB/depth residuals, P-versus-R renders, Phase-4
refinement images -- are recorded as NOT_AVAILABLE with the reason and the
command that would produce them, never as a fabricated zero.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from debug.stagewise_probe import (  # noqa: E402
    CONSISTENT, INCONSISTENT, LOW_SIGNAL, NOT_AVAILABLE, THREE_WAY_HEADER,
    UNOBSERVABLE, probe_phase0, probe_phase1_free_motion, probe_phase2_graph,
    probe_temporal_covariance, probe_velocity_estimators, three_way_angular)
from debug.theory_audit import (  # noqa: E402
    build_matrix, matrix_markdown, production_wiring_audit)
from debug.dual_joint_artifacts import (  # noqa: E402
    common_screw_payload, consistency_rows, finite_rotation_rows,
    joint_type_evidence_payload, rigid_twist_rows, velocity_gradient_rows,
    CONSISTENCY_HEADER, FINITE_ROTATION_HEADER, RIGID_TWIST_HEADER,
    VELOCITY_GRADIENT_HEADER)
from debug.deep_debug_html import write_joint_debug_html  # noqa: E402
from estimator.manifold import project_to_so3  # noqa: E402
from joint_reduction.differential_finite_consistency import \
    incremental_spatial_omega  # noqa: E402
from joint_reduction.direct_gaussian_joint import (  # noqa: E402
    dual_differential_finite_evidence, geometric_joint_candidate)
from tools.dual_joint_discovery import (DEFAULT_DT,  # noqa: E402
                                        load_phase1_tracks)
from tools.iekf_artifacts import ArtifactWriter  # noqa: E402

STAGES = ("phase0_canonical", "phase1_observation", "phase1_free_motion",
          "phase1_differential", "phase2_graph", "phase2_rigid",
          "phase2_finite_rotation", "common_screw", "phase3_geometric_joint",
          "branch_b_direct_gaussian", "branch_c_rgbd",
          "phase4_renderer_refinement", "phase5_reduced_filter")

PROVENANCE_HEADER = ["stage", "quantity", "matrix_shape", "source",
                     "scale_calibrated", "correlations_included",
                     "temporal_cross_covariance_included",
                     "camera_cross_covariance_included", "rank", "condition"]

CAUSAL_HEADER = ["transition", "median_point_displacement_change",
                 "rotation_change_rad", "rgb_residual_change",
                 "depth_residual_change", "support_change",
                 "angular_velocity_change", "axis_direction_change_cos",
                 "note"]


def _w(path: Path, name: str, obj):
    path.mkdir(parents=True, exist_ok=True)
    (path / name).write_text(json.dumps(obj, indent=1, default=str))


def _csv(path: Path, name: str, header, rows):
    import csv as _c
    path.mkdir(parents=True, exist_ok=True)
    with (path / name).open("w", newline="") as fh:
        w = _c.writer(fh)
        w.writerow(header)
        w.writerows(rows)


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", required=True)
    ap.add_argument("--part", type=int, default=1)
    ap.add_argument("--dt", type=float, default=DEFAULT_DT)
    ap.add_argument("--confidence", type=float, default=0.95)
    ap.add_argument("--edges", type=int, default=4000)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    root = Path(args.run)
    scene = root.name
    out = Path(args.out) if args.out else (root / "deep_debug")
    num = out / "numeric"
    out.mkdir(parents=True, exist_ok=True)
    print(f"=== deep debug: {scene} ===")

    stages: dict = {}
    provenance: list = []
    notes: list = []

    # ---- 1. source-code theory audit (sections 4, 54) --------------------
    rows = build_matrix()
    wiring = production_wiring_audit()
    (out / "theory_implementation_audit.md").write_text(
        matrix_markdown(rows)
        + "\n## Production wiring audit (section 54)\n\n"
        + "| symbol | what | runs in a standard batch run | called by |\n"
        + "|---|---|---|---|\n"
        + "".join(f"| `{k}` | {v['what']} | "
                 f"{'yes' if v['runs_in_standard_batch'] else '**no**'} | "
                 f"{', '.join(c.split('/')[-1] for c in v['called_by']) or '—'} |\n"
                 for k, v in wiring.items()))
    _w(out, "theory_implementation_audit.json",
       {"matrix": rows, "production_wiring": wiring})
    from collections import Counter
    tally = Counter(r["status"] for r in rows)
    print("  theory matrix: " + ", ".join(f"{k}={v}" for k, v in tally.most_common()))

    # ---- 2. Phase 0 (section 6) -----------------------------------------
    p0 = probe_phase0(root)
    stages["phase0_canonical"] = {k: v for k, v in p0.items() if k != "artifacts"}
    _w(num / "phase0_canonical", "canonical_summary.json", p0)
    print(f"  phase0: {p0['status']}")

    # Phase-0 fixed-camera render sanity (section 7) needs the renderer.
    stages["phase1_observation"] = {
        "status": NOT_AVAILABLE,
        "reason": ("per-camera RGB/depth residuals and fixed-camera renders "
                   "are not persisted by a standard run and cannot be "
                   "reconstructed from cached tensors; they require the "
                   "rasterizer and the dataset images"),
        "command": (f"python tools/run_iekf_scene.py --model <model> "
                    f"--source <source> --observation-source dataset_rgbd "
                    f"--window-discovery --debug-all --debug-render-panels "
                    f"--out {root}")}
    _w(num / "phase1_observation", "not_available.json",
       stages["phase1_observation"])

    # ---- 3. Phase 1 free motion (sections 9-11) -------------------------
    ph1 = load_phase1_tracks(root, args.part)
    fm = probe_phase1_free_motion(ph1, args.dt)
    ve = probe_velocity_estimators(ph1, args.dt)
    tc = probe_temporal_covariance(ph1, args.dt)
    stages["phase1_free_motion"] = {k: v for k, v in fm.items()
                                    if k != "artifacts"}
    _w(num / "phase1_free_motion", "free_motion_summary.json", fm)
    _w(num / "phase1_free_motion", "velocity_estimator_comparison.json", ve)
    _w(num / "phase1_free_motion", "temporal_velocity_covariance_summary.json", tc)
    print(f"  phase1 free motion: {fm['status']}  "
          f"travel {fm['artifacts']['centre_travel_over_L']:.3f} L, "
          f"{fm['artifacts']['nodes_moving_over_5pct_L']}/"
          f"{fm['artifacts']['num_supernodes']} nodes moving")
    provenance.append(["phase1_free_motion", "velocity", "[T,N,3]",
                       "ANALYTIC_PROPAGATION" if ph1["stored_velocity"]
                       is not None else "IDENTITY_FALLBACK",
                       False, False, False, False, "", ""])

    # ---- 4. the differential layer (sections 12-16) ----------------------
    graph = probe_phase2_graph(ph1, num_edges=args.edges)
    ev = dual_differential_finite_evidence(ph1["x0"], ph1["tracks"],
                                           dt=args.dt, edges=graph["edges"])
    cand = geometric_joint_candidate(ev, confidence=args.confidence,
                                     x0=ph1["x0"], dt=args.dt)
    _csv(num / "phase1_differential", "velocity_gradient_by_frame.csv",
         VELOCITY_GRADIENT_HEADER, velocity_gradient_rows(ev))
    nested = [[t, r["chi2_rigid_vs_affine"], r["dof"], r["affine_residual"],
               r["rigid_residual"], r["excess"], r["sigma2_from_affine"]]
              for t, r in enumerate(ev["rigidity_nested"])]
    _csv(num / "phase1_differential", "nested_test_by_frame.csv",
         ["frame", "chi2_rigid_vs_affine", "dof", "RSS_affine", "RSS_rigid",
          "excess", "sigma2_from_affine"], nested)
    stages["phase1_differential"] = {
        "status": (CONSISTENT if cand["stage_a_rigid"]["passes"]
                   else INCONSISTENT),
        "median_chi2_rigid_vs_affine":
            cand["stage_a_rigid"]["median_chi2_rigid_vs_affine"],
        "cut": cand["stage_a_rigid"]["cut"],
        "median_Q_D": cand["stage_a_rigid"]["median_Q_D"],
        "answers": "whether the instantaneous field is one rigid body"}
    print(f"  phase1 differential: {stages['phase1_differential']['status']}  "
          f"chi2 {cand['stage_a_rigid']['median_chi2_rigid_vs_affine']:.3e} "
          f"vs cut {cand['stage_a_rigid']['cut']:.2f}")
    provenance.append(["phase1_differential", "affine covariance", "[12,12]",
                       "RESIDUAL_CALIBRATED", True, True, False, False,
                       ev["affine"][0]["rank"],
                       f"{ev['affine'][0]['condition']:.3e}"])
    provenance.append(["phase1_differential", "strain covariance Sigma_D",
                       "[6,6]", "RESIDUAL_CALIBRATED", True, True, False,
                       False, "", ""])

    # ---- 5. Phase 2 graph (section 17-18) -------------------------------
    stages["phase2_graph"] = {k: v for k, v in graph.items()
                              if k in ("status", "artifacts", "answers")}
    _w(num / "phase2_graph", "finite_rigidity_summary.json",
       {k: v for k, v in graph.items() if k in ("status", "artifacts")})
    _csv(num / "phase2_graph", "rigidity_edges.csv",
         ["i", "j", "rho_finite", "max_relative_rotation_deg", "length_drift"],
         [[int(graph["edges"][k, 0]), int(graph["edges"][k, 1]),
           float(graph["rho"][k]),
           float(graph["relative_rotation"]["max_angle"][k]) * 57.2957795,
           float(graph["relative_rotation"]["length_drift"][k])]
          for k in range(graph["edges"].shape[0])])
    print(f"  phase2 graph: {graph['status']}  rho_p90 "
          f"{graph['artifacts']['rho_finite_p90']:.3e}, relative-vector "
          f"rotation p90 {graph['artifacts']['relative_vector_rotation_p90_deg']:.3f} deg")

    # ---- 6. Phase 2 rigid + three-way angular (sections 19-23) ----------
    tf = root / "debug_numeric" / "rigid_transforms.pt"
    p2_om, p2_note = None, None
    if tf.is_file():
        T4 = torch.load(tf, map_location="cpu", weights_only=False).double()
        idx = min(max(args.part - 1, 0), T4.shape[0] - 1)
        R2 = project_to_so3(T4[idx, :, :3, :3].contiguous()).detach()
        p2_om = incremental_spatial_omega(R2, dt=args.dt)["omega"]
        _csv(num / "phase2_rigid", "phase2_transform_by_frame.csv",
             ["frame"] + [f"R_{i}{j}" for i in range(3) for j in range(3)]
             + ["t_x", "t_y", "t_z"],
             [[t] + [float(x) for x in T4[idx, t, :3, :3].reshape(-1)]
              + [float(x) for x in T4[idx, t, :3, 3]]
              for t in range(T4.shape[1])])
    else:
        p2_note = f"{tf} absent; Phase-2 transforms unavailable"

    tw = three_way_angular(ev, ph1, p2_om, args.dt, args.confidence)
    _csv(num / "phase2_finite_rotation", "angular_three_way.csv",
         THREE_WAY_HEADER, tw["rows"])
    _csv(num / "phase2_finite_rotation", "finite_rotation_by_frame.csv",
         FINITE_ROTATION_HEADER, finite_rotation_rows(ev))
    _csv(num / "phase2_finite_rotation", "differential_finite_consistency.csv",
         CONSISTENCY_HEADER, consistency_rows(ev))
    _w(num / "phase2_finite_rotation", "three_way_summary.json",
       {**tw["summary"], "phase2_note": p2_note})
    stages["phase2_finite_rotation"] = {"status": tw["status"],
                                        **tw["summary"],
                                        "answers": tw["answers"]}
    s = tw["summary"]
    print(f"  three-way angular: {tw['status']}")
    print(f"    |omega| diff-P1 {s['median_norm_diff_p1']}, "
          f"finite-P1 {s['median_norm_finite_p1']}, P2 {s['median_norm_p2']}")
    print(f"    cos(diff,finite)={s['median_cos_diff_vs_finite_p1']}  "
          f"cos(diff,P2)={s['median_cos_diff_p1_vs_p2']}  "
          f"cos(finite,P2)={s['median_cos_finite_p1_vs_p2']}")
    print(f"    {s['interpretation']}")
    provenance.append(["phase2_finite_rotation", "omega diff-P1 covariance",
                       "[T,3,3]", "RESIDUAL_CALIBRATED", True, True, False,
                       False, "", ""])
    provenance.append(["phase2_finite_rotation", "omega finite-P1 covariance",
                       "[T,3,3]", "RESIDUAL_CALIBRATED", True, True, False,
                       False, "", ""])
    provenance.append(["phase2_finite_rotation", "omega P2 covariance", "",
                       "UNKNOWN", False, False, False, False, "", ""])

    # ---- 7. common screw + geometric joint (sections 25-28) -------------
    _w(num / "common_screw", "common_screw_fit.json",
       common_screw_payload(ev, cand))
    _csv(num / "common_screw", "rigid_twist_by_frame.csv",
         RIGID_TWIST_HEADER, rigid_twist_rows(ev))
    sc = ev["screw"]
    stages["common_screw"] = {
        "status": (CONSISTENT if cand["stage_b_one_dof"]["passes"]
                   else INCONSISTENT),
        "twist_chi2_rank1": sc["twist_rank"]["chi2_rank1"],
        "twist_dof": sc["twist_rank"]["dof_rank1"],
        "twist_per_dof": sc["twist_rank"]["rank1_per_dof"],
        "angular_chi2_rank0": sc["angular_rank"]["chi2_rank0"],
        "angular_dof_rank0": sc["angular_rank"]["dof_rank0"],
        "answers": "whether one screw generates the whole window"}
    _w(num / "phase3_geometric_joint", "joint_type_evidence.json",
       joint_type_evidence_payload(ev, cand))
    stages["phase3_geometric_joint"] = {
        "status": cand["status"], "candidate": cand["candidate"],
        "reason": cand["reason"],
        "stage_a": cand["stage_a_rigid"], "stage_b": cand["stage_b_one_dof"],
        "stage_c": cand["stage_c_angular"],
        "answers": "the geometric P/R candidate, before any renderer evidence"}
    print(f"  common screw: {stages['common_screw']['status']}  "
          f"rank1 {sc['twist_rank']['rank1_per_dof']:.3e} per dof")
    print(f"  geometric joint: {cand['status']} (candidate={cand['candidate']})")

    # ---- 8. the legacy transform-only fit, for comparison (section 28) ---
    legacy = root / "debug_numeric" / "joint_direct" / "joint_fit_comparison.json"
    gj = root / "geometric_joints.json"
    legacy_payload = None
    for p in (gj, legacy):
        if p.is_file():
            try:
                legacy_payload = json.loads(p.read_text())
                break
            except Exception:
                pass
    _w(num / "phase3_geometric_joint", "legacy_transform_only.json",
       {"found": legacy_payload is not None,
        "payload": legacy_payload,
        "note": ("section 38: the gain-versus-floor comparison is retained as "
                 "a diagnostic; this records what it said for the same part")})

    # ---- 9. Branch B / C provenance (sections 29-31) ---------------------
    stages["branch_b_direct_gaussian"] = {
        "status": "DEBUG_ONLY",
        "note": ("the dual chain runs here, offline. The theory matrix shows "
                 "it is not reachable from a standard batch run."),
        "source_of_tracks": "Phase-1 bootstrap_motion.csv (pre-rigid-projection)",
        "velocity_source": "section 8 central difference on those positions"}
    stages["branch_c_rgbd"] = {
        "status": NOT_AVAILABLE,
        "reason": ("Branch C needs a live RGBDJointProblem: the rasterizer, "
                   "the cameras and the dataset images. None of that is in "
                   "the cached tensors."),
        "theory_gap": ("section 5.10 / 30: the implemented test compares "
                       "refitted P and R losses with a held-out fold, not the "
                       "efficient score of H_0: beta = 0 at the P optimum"),
        "command": (f"python tools/run_iekf_scene.py --model <model> "
                    f"--source <source> --window-discovery "
                    f"--joint-direct-rgbd --joint-heldout-validation "
                    f"--out {root}")}
    for k in ("phase4_renderer_refinement", "phase5_reduced_filter"):
        src = root / "debug_numeric" / (
            "renderer_refinement.csv" if "renderer" in k else "reduced_filter.csv")
        stages[k] = ({"status": "CACHED", "source": str(src)}
                     if src.is_file() else
                     {"status": NOT_AVAILABLE,
                      "reason": f"{src.name} absent from this run"})
        _w(num / k, "summary.json", stages[k])
    _w(num / "branch_b_direct_gaussian", "provenance.json",
       stages["branch_b_direct_gaussian"])
    _w(num / "branch_c_rgbd", "not_available.json", stages["branch_c_rgbd"])

    # ---- 10. causal deltas (section 50) ---------------------------------
    X = ph1["tracks"]
    ca = []
    ca.append(["Phase0 -> Phase1",
               _fmt(float((X[-1] - X[0]).norm(dim=1).median())), "", "", "",
               f"{X.shape[1]} supernodes", "", "",
               "canonical centres to Phase-1 tracks"])
    if p2_om is not None:
        d1 = ev["twist"]["omega"]
        d1m = 0.5 * (d1[1:] + d1[:-1])
        n = min(d1m.shape[0], p2_om.shape[0])
        cos = float(((d1m[:n] * p2_om[:n]).sum(1)
                     / (d1m[:n].norm(dim=1).clamp_min(1e-300)
                        * p2_om[:n].norm(dim=1).clamp_min(1e-300))).median())
        ca.append(["Phase1 -> Phase2", "", "", "", "", "",
                   _fmt(float((p2_om[:n].norm(dim=1)
                               - d1m[:n].norm(dim=1)).median())),
                   _fmt(cos),
                   "angular velocity, Phase-1 differential vs Phase-2 rigid"])
    ca.append(["Phase2 -> common screw", "", "", "", "", "", "",
               _fmt(sc["twist_rank"]["rank1_per_dof"]),
               "rank-1 residual per dof of the twist sequence"])
    ca.append(["common screw -> geometric joint", "", "", "", "", "", "", "",
               f"status {cand['status']}"])
    ca.append(["geometric joint -> renderer refinement", "", "", "", "", "",
               "", "", "NOT_AVAILABLE: needs a live renderer"])
    ca.append(["renderer refinement -> reduced filter", "", "", "", "", "",
               "", "", "NOT_AVAILABLE: needs a live renderer"])
    _csv(out, "causal_deltas.csv", CAUSAL_HEADER, ca)

    # ---- 11. provenance + summaries (sections 36, 55) -------------------
    provenance.append(["phase3_geometric_joint", "chi-square cuts", "scalar",
                       "ANALYTIC_PROPAGATION", True, False, False, False,
                       "", ""])
    _csv(out, "information_provenance.csv", PROVENANCE_HEADER, provenance)
    _w(out, "stage_summary.json",
       {"scene": scene, "part": args.part, "dt": args.dt,
        "confidence": args.confidence, "stages": stages,
        "theory_status_tally": dict(tally)})
    _w(out, "causal_trace.json",
       {"scene": scene,
        "chain": ["phase0_canonical", "phase1_free_motion",
                  "phase1_differential", "phase2_graph",
                  "phase2_finite_rotation", "common_screw",
                  "phase3_geometric_joint"],
        "status_by_stage": {k: v.get("status") for k, v in stages.items()},
        "earliest_divergence": _earliest(stages)})
    (out / "stage_summary.md").write_text(_summary_md(scene, stages, tally, s,
                                                      cand))

    # ---- 11b. section 32/33: the render comparison, if images exist -----
    rend = out / "renders"
    rrows = []
    for cam in sorted(rend.glob("camera_*")):
        man = cam / "sequence_manifest.json"
        if not man.is_file():
            continue
        m = json.loads(man.read_text())
        for fr in m.get("frames", []):
            sc_ = fr.get("images", {})
            for st in m.get("stages", []):
                if f"{st}_rgb_l1" not in sc_:
                    continue
                rrows.append([int(m.get("camera", 0)), fr["frame"], st,
                              sc_.get(f"{st}_rgb_l1"),
                              sc_.get(f"{st}_depth_l1"),
                              sc_.get(f"{st}_valid_depth_px")])
    if rrows:
        _csv(num / "phase4_renderer_refinement", "render_comparison.csv",
             ["camera", "frame", "stage", "rgb_l1", "depth_l1",
              "valid_depth_px"], rrows)
        stages["render_comparison"] = {
            "status": "CACHED", "rows": len(rrows),
            "stages_rendered": sorted({r[2] for r in rrows}),
            "cameras": sorted({r[0] for r in rrows}),
            "answers": ("sections 32/49: what each motion model renders from "
                        "the same camera, and how far it is from the image")}
        print(f"  renders: {len(rrows)} stage-frames from "
              f"{len(set(r[0] for r in rrows))} camera(s)")
    else:
        stages["render_comparison"] = {
            "status": NOT_AVAILABLE,
            "reason": "no renders/ on disk for this run",
            "command": f"python tools/deep_debug_render.py --run {root}"}
        print("  renders: none on disk "
              "(run tools/deep_debug_render.py to produce them)")

    # ---- 12. the single synchronized HTML (section 45) ------------------
    html = write_joint_debug_html(out / "visual", scene, ph1, ev, cand, tw,
                                  p2_om, stages, rows, args.dt)
    print(f"\n  stage summary  -> {out / 'stage_summary.md'}")
    print(f"  theory audit   -> {out / 'theory_implementation_audit.md'}")
    print(f"  visual         -> {html}")

    gt = root / "dataset_joint_ground_truth.json"
    if gt.is_file():
        try:
            t = json.loads(gt.read_text()).get("joint_type_raw")
            t = {"continuous": "revolute"}.get(t, t)
            print(f"\n  [post-inference only] ground truth {t}")
        except Exception:
            pass
    return 0


def _fmt(v):
    try:
        return f"{float(v):.6e}"
    except (TypeError, ValueError):
        return ""


def _earliest(stages: dict):
    """Section 55 F: the first stage whose invariant does not hold."""
    order = ["phase0_canonical", "phase1_free_motion", "phase1_differential",
             "phase2_graph", "phase2_finite_rotation", "common_screw",
             "phase3_geometric_joint"]
    for k in order:
        st = stages.get(k, {}).get("status")
        if st in (INCONSISTENT, UNOBSERVABLE, LOW_SIGNAL):
            return {"stage": k, "status": st}
    return {"stage": None, "status": None}


def _summary_md(scene, stages, tally, three_way, cand) -> str:
    L = [f"# Deep stagewise debug — {scene}", "",
         "Instrumentation only. No threshold was tuned and nothing here feeds "
         "back into a decision (section 1).", "",
         "## Stage status", "", "| stage | status | what it answers |",
         "|---|---|---|"]
    for k, v in stages.items():
        L.append(f"| {k} | **{v.get('status')}** | {v.get('answers', v.get('reason', ''))} |")
    L += ["", "## Three-way angular comparison (sections 21-23)", "",
          f"- |omega| differential-P1: `{three_way['median_norm_diff_p1']}`",
          f"- |omega| finite-P1 (consecutive Kabsch): `{three_way['median_norm_finite_p1']}`",
          f"- |omega| Phase-2 rigid: `{three_way['median_norm_p2']}`",
          f"- cos(diff-P1, finite-P1): `{three_way['median_cos_diff_vs_finite_p1']}`",
          f"- cos(diff-P1, P2): `{three_way['median_cos_diff_p1_vs_p2']}`",
          f"- cos(finite-P1, P2): `{three_way['median_cos_finite_p1_vs_p2']}`",
          f"- Mahalanobis(diff-P1, finite-P1): `{three_way['median_mahal_diff_vs_finite_p1']}` "
          f"against a chi2(3) cut of `{three_way['chi2_3dof_cut']:.2f}`",
          f"- fraction of frames where diff-P1 is observable: "
          f"`{three_way['fraction_diff_observable']:.3f}`",
          f"- fraction where finite-P1 is observable: "
          f"`{three_way['fraction_finite_p1_observable']:.3f}`", "",
          f"**Interpretation (reported, never used as production logic):** "
          f"{three_way['interpretation']}", ""]
    if not three_way.get("axis_cosine_interpretable", True):
        L += [f"> `AXIS_COSINE_NOT_INTERPRETABLE` — at least one angular "
              f"vector is not observable, so the cosines above carry no "
              f"physical meaning (section 24).", ""]
    L += ["## Geometric candidate", "",
          f"- status: **{cand['status']}** (candidate `{cand['candidate']}`)",
          f"- {cand['reason']}", "",
          "## Theory matrix tally", ""]
    for k, v in tally.most_common():
        L.append(f"- {k}: {v}")
    return "\n".join(L) + "\n"


if __name__ == "__main__":
    raise SystemExit(main())
