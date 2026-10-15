"""report/RESULTS.md, TOP_MISSING_GEOMETRY.md, MATCHING_AND_AMBIGUITY.md.

Generated from the run's own numbers. Sentences that would need a human to
look at the figures are phrased as questions to check, not as findings.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np


def _p(x):
    return "n/a" if x is None else f"{x:.2f}%"


def _f(x, d=4):
    return "n/a" if x is None else f"{x:.{d}f}"


def write(out: Path, S: dict, per_frame: list, cases: dict, pairs: list, tables: dict, ctx):
    rdir = out / "report"
    rdir.mkdir(parents=True, exist_ok=True)
    P = S["pixel_pct_of_gt_valid"]
    T = S["pixel_totals"]
    th = S["thresholds"]
    pst = S["pairs"]
    scene = S["scene"]
    pilot = S["pilot"]
    head = (f"# {scene}: forward/backward observation overlap, same-surface evidence and "
            f"complementary coverage\n\n"
            f"**Status: {S['status']}**{' (PILOT: not the full experiment)' if pilot else ''}. "
            f"Original frames {S['frames'][0]}..{S['frames'][1]} ({S['num_frames']}), cameras "
            f"{', '.join(S['cameras'])}; {S['views_per_source']} views per source. "
            f"F = static_bidi/initial + bidirectional/forward ({S['N_F']:,} Gaussians), "
            f"B = static_bidi/final + bidirectional/backward ({S['N_B']:,} Gaussians), each "
            f"rendered ALONE with the production IEKF renderer at "
            f"{S['state_mode']['F']} / {S['state_mode']['B']}.\n\n"
            "Read-only: no training, merging, pruning or optimisation; the raw F+B union was "
            "never rendered. Provenance, hashes and thresholds: `provenance.json`.\n")
    tr = []
    for s in ("F", "B"):
        c = S["contributors"][s]
        tr.append(f"| {s} | {c['records_total']:,} | {c['per_pixel_p50_median_over_views']:.0f} | "
                  f"{c['per_pixel_p99_max_over_views']:.0f} | {c['per_pixel_max']} | "
                  f"{c['early_terminated_pixels_total']:,} | {c['T1_max_abs_sum_q_minus_A']:.1e} | "
                  f"{c['T1_max_abs_depth_reconstruction_m']:.1e} | "
                  f"{'yes' if c['T1_final_T_bitwise_all_views'] else 'NO'} |")
    gF = {k: v for k, v in S.get("label_counts", {}).get("F", {}).items()}
    gB = {k: v for k, v in S.get("label_counts", {}).get("B", {}).items()}
    labs = sorted(set(gF) | set(gB))
    lab_rows = "\n".join(f"| {k} | {gF.get(k, 0):,} | {gB.get(k, 0):,} |" for k in labs)
    # direction-dependent degradation: residual trends
    fr = [r["frame"] for r in per_frame]

    def trend(key):
        v = [r.get(key) for r in per_frame]
        v = [(f, x) for f, x in zip(fr, v) if x is not None]
        if len(v) < 2:
            return "n/a"
        a, b = np.polyfit([f for f, _ in v], [x for _, x in v], 1)
        return f"{v[0][1]:.4f} at frame {v[0][0]} -> {v[-1][1]:.4f} at frame {v[-1][0]} (slope {a:+.2e}/frame)"

    def first_last_pct(key):
        a, b = per_frame[0], per_frame[-1]
        return (f"{100 * a[key] / max(a['gt_valid'], 1):.2f}% at frame {a['frame']}, "
                f"{100 * b[key] / max(b['gt_valid'], 1):.2f}% at frame {b['frame']}")

    gB_cases, gF_cases = cases.get("B", []), cases.get("F", [])
    occ = {}
    for c in gB_cases:
        occ[c["occlusion"]["status"]] = occ.get(c["occlusion"]["status"], 0) + 1
    occF = {}
    for c in gF_cases:
        occF[c["occlusion"]["status"]] = occF.get(c["occlusion"]["status"], 0) + 1
    mdF = tables["F"]["mass_front_of_observed"].sum()
    mdB = tables["B"]["mass_front_of_observed"].sum()
    excl_B = tables["B"]["witness_U"].sum() / max(tables["B"]["mass_supported"].sum(), 1e-9)
    excl_F = tables["F"]["witness_U"].sum() / max(tables["F"]["mass_supported"].sum(), 1e-9)
    nB_wit = int((tables["B"]["witness_U"] >= 1.0).sum())
    nF_wit = int((tables["F"]["witness_U"] >= 1.0).sum())
    cfs = [c.get("counterfactual") for c in gB_cases if c.get("counterfactual")]

    body = f"""
## Definitions used throughout

* **covers**: alpha >= {th['alpha_cover']}. **explains**: covers, GT depth valid and
  |D - D_gt| <= {th['depth_tol_m']} m (D = Z/(A+eps), the renderer's own depth).
  RGB+depth explanation additionally requires RGB RMS <= {th['rgb_tol']}.
  Scales are the historical Phase-1 sensor model ({th['provenance']}).
* Every percentage below has the GT-valid pixels of the processed views as its
  denominator ({T['gt_valid']:,} pixels) unless it says otherwise. GT-invalid
  pixels ({T['gt_invalid']:,}) are never "missing geometry".
* Per-pixel contributions q = T*alpha come from a diagnostic kernel that walks
  the production cache with the production loop. Verified on every processed view (T1):

| source | contributor records | per-pixel p50 | p99 (max over views) | max | early-terminated pixels | max abs(sum q - A) | max depth reconstruction error (m) | final T bitwise |
|---|---|---|---|---|---|---|---|---|
{chr(10).join(tr)}

## 1. How much do F and B cover, share, and leave uncovered?

| category (alpha >= {th['alpha_cover']}) | % of GT-valid |
|---|---|
| both cover | {_p(P['BOTH_COVER__on_gt_valid'])} |
| F only | {_p(P['F_ONLY_COVER__on_gt_valid'])} |
| B only | {_p(P['B_ONLY_COVER__on_gt_valid'])} |
| neither | {_p(P['NEITHER_COVERS__on_gt_valid'])} |
| low visibility (max alpha in [{th['low_visibility_alpha']}, {th['alpha_cover']})) | {_p(P['LOW_VISIBILITY'])} |

Coverage on GT-invalid pixels (model mass where the sensor saw nothing):
F {T['GT_INVALID_BUT_F_COVERS']:,} px, B {T['GT_INVALID_BUT_B_COVERS']:,} px.
Per frame: `metrics/per_frame.csv`, per view: `metrics/per_frame_camera.csv`,
per camera: `metrics/global_summary.json` -> per_camera_pct_of_gt_valid.
Timeline: `visuals/coverage_timeline.png`.

## 2. How much shared coverage is the same observed surface?

Of the GT-valid pixels BOTH sources cover:

| | % of both-cover |
|---|---|
| same observed surface candidate (both within {th['depth_tol_m']} m of GT and of each other) | {_p(S['of_both_cover_gt_valid_pct']['BOTH_SAME_OBSERVED_SURFACE_CANDIDATE'])} |
| different depth layers (|D_F - D_B| > {th['depth_tol_m']} m) | {_p(S['of_both_cover_gt_valid_pct']['BOTH_COVER_DIFFERENT_DEPTH'])} |
| agree with each other but not with the observation | {_p(S['of_both_cover_gt_valid_pct']['BOTH_COVER_AGREE_OFF_OBSERVED_SURFACE'])} |

Pixel overlap is not surface identity: the second row is overlap that must never
be read as a match. Sensitivity of all of these to alpha and depth tolerance:
`metrics/threshold_sensitivity.csv` (T11).

## 3. Which source explains the observation, where and when?

| | % of GT-valid |
|---|---|
| both explain the depth | {_p(P['BOTH_VALID_SUPPORT'])} |
| F explains, B does not | {_p(P['F_SUPPORTED_B_UNEXPLAINED'])} |
| B explains, F does not | {_p(P['B_SUPPORTED_F_UNEXPLAINED'])} |
| neither explains | {_p(P['NEITHER_VALID_SUPPORT'])} |
| F depth advantage (by > {th['advantage_depth_m']} m, both cover) | {_p(P['F_DEPTH_ADVANTAGE'])} |
| B depth advantage | {_p(P['B_DEPTH_ADVANTAGE'])} |
| F colour-only advantage (depth tied) | {_p(P['F_COLOR_ONLY_ADVANTAGE'])} |
| B colour-only advantage | {_p(P['B_COLOR_ONLY_ADVANTAGE'])} |
| F in front of the observation where B is right | {_p(P['F_IN_FRONT_OF_OBS_B_CORRECT'])} |
| B in front of the observation where F is right | {_p(P['B_IN_FRONT_OF_OBS_F_CORRECT'])} |

Over time (all cameras pooled):
* "F explains, B does not": {first_last_pct('F_SUPPORTED_B_UNEXPLAINED')}
* "B explains, F does not": {first_last_pct('B_SUPPORTED_F_UNEXPLAINED')}
* F depth MAE on its covered pixels: {trend('F_depth_mae_m__covered')}
* B depth MAE on its covered pixels: {trend('B_depth_mae_m__covered')}

A direction-dependent degradation would show as F's error growing with the
frame index and B's shrinking. Whether it does here is read from these
trends and `visuals/depth_error_timeline.png`; regions are in
`pixel_maps/frame_NNNN/cam_NNN/{{B_explains_F_misses,F_explains_B_misses}}.png`.

## 4. Credible F <-> B Gaussian matches

Pair evidence: **{pst['exactness'].split(':')[0]}** over {pst['frames_processed']} of
{pst['frames_requested']} frames ({pst['views_processed']} views).
{pst['pair_records_total']:,} co-contributor pair records on
{pst['pixels_with_both_contributors_total']:,} pixels were enumerated and summed exactly per view
({pst['view_level_distinct_pairs_total']:,} view-level (i, j) sums); pairs with per-view
O < {pst['prune_tau']} were not stored ({pst['view_level_stored_pairs_total']:,} stored, merged into
{pst['distinct_pairs_stored_sequence']:,} sequence pairs). Pruned mass, accounted per Gaussian:
{100 * pst.get('omitted_fraction_of_pair_mass', 0):.3f}% of all pair mass,
{100 * pst.get('omitted_fraction_of_same_surface_mass', 0):.3f}% of same-surface (4 cm) mass.
Best partner by stored same-surface overlap CERTIFIED against everything pruning could hide:
F {_p(100 * pst['F_best_by_O_surf_certified_fraction'] if pst.get('F_best_by_O_surf_certified_fraction') is not None else None)},
B {_p(100 * pst['B_best_by_O_surf_certified_fraction'] if pst.get('B_best_by_O_surf_certified_fraction') is not None else None)}
of the Gaussians that have a same-surface partner; among those with supported mass >= S_MIN:
F {_p(100 * pst['F_best_by_O_surf_certified_fraction__supported_ge_S_MIN'] if pst.get('F_best_by_O_surf_certified_fraction__supported_ge_S_MIN') is not None else None)},
B {_p(100 * pst['B_best_by_O_surf_certified_fraction__supported_ge_S_MIN'] if pst.get('B_best_by_O_surf_certified_fraction__supported_ge_S_MIN') is not None else None)}
(uncertified ones are flagged per Gaussian in the evidence tables). Exact unpruned reference (frame {(pst.get('exact_reference_frame') or {}).get('frame')}, cameras {(pst.get('exact_reference_frame') or {}).get('cameras')}):
same-surface mass dropped {100 * (pst.get('exact_reference_frame') or {}).get('fraction_O_surf_mass_dropped', 0):.3f}%,
frame-level best-partner agreement F {100 * (pst.get('exact_reference_frame') or {}).get('F_best_partner_agreement', 0):.2f}%,
B {100 * (pst.get('exact_reference_frame') or {}).get('B_best_partner_agreement', 0):.2f}% over every row with any
same-surface mass; F {100 * (pst.get('exact_reference_frame') or {}).get('F_best_partner_agreement_best_O_surf_ge_1e-2', 0):.2f}%,
B {100 * (pst.get('exact_reference_frame') or {}).get('B_best_partner_agreement_best_O_surf_ge_1e-2', 0):.2f}% where the
best same-surface overlap is at least 1e-2; weighted by O_surf F
{100 * (pst.get('exact_reference_frame') or {}).get('F_best_partner_agreement_weighted_by_O_surf', 0):.2f}%,
B {100 * (pst.get('exact_reference_frame') or {}).get('B_best_partner_agreement_weighted_by_O_surf', 0):.2f}%.

| label (hypothesis, thresholds in metrics/global_summary.json) | F Gaussians | B Gaussians |
|---|---|---|
{lab_rows}

Label sensitivity to rho thresholds: `metrics/threshold_sensitivity_labels.csv`.
Examples: `MATCHING_AND_AMBIGUITY.md`.

## 5. Backward-source geometry missing from the forward explanation

{nB_wit:,} B Gaussians carry witness mass U >= 1 (pixel-weight units) on pixels B explains
and F does not; overall {100 * excl_B:.2f}% of B's supported mass is exclusive (F: {100 * excl_F:.2f}%,
{nF_wit:,} Gaussians). Grouped into 3-D proximity patches (not parts), the top
{min(len(gB_cases), 30)} are in `TOP_MISSING_GEOMETRY.md` with crops at frames
0..59, two cameras each, and the patch's own contribution overlaid.
Can a human see the geometry in the observation and its absence or error in F? That
is answered per patch by its crop sheet; this report does not answer it for you.

## 6. Initially occluded?

Occlusion test over the B patches (GT depth at the members' projections before their
first witnessed frame; rule in each `gaussians/trajectory_witnesses/B_group_*.json`):
{json.dumps(occ)}.
OCCLUSION_HYPOTHESIS means the GT depth shows a surface in front of the members in
most early views where they are in the field of view: an observation-based
hypothesis, not proof of physical occlusion. WITNESSED_FROM_FIRST_FRAME patches
are not initially hidden.

## 7. Forward-only observed surfaces

F patches: {len(gF_cases)} (occlusion test: {json.dumps(occF)}). Their tracks, coherence
annotation and endpoint appearance are in `gaussians/forward_unique_candidates.csv`
and `gaussians/trajectory_witnesses/F_group_*.json`.

## 8. How much putative novelty is contradicted?

* Mass placed IN FRONT of the observed surface (q on GT-valid pixels with the
  contributor more than {th['contrib_depth_tol_m']} m nearer than GT): F {mdF:.4g}, B {mdB:.4g}.
* Witness mass requires the contributor's own depth to match GT, so a splat behind a
  correctly rendered foreground does not take credit for it.
* Candidates whose label is OVERLAPPING_DIFFERENT_DEPTH share pixels with an F partner
  at a different depth; these are listed in the unique-candidate tables with that label.
* CENTER_ONLY_FIXED_ORIENTATION: the historical runs saved centres only, so every
  frame uses the PLY orientation. A rotating part's splats keep their endpoint
  orientation, which distorts their footprints away from the endpoint frame and can
  create or hide apparent exclusivity. This is the largest systematic caveat.

Within-source counterfactuals (patch hidden in its own source, one view re-rendered,
NOT a third-model test):
{chr(10).join(f"* B patch at frame {c['frame']} {c['camera']}: depth error on its pixels {_f(c['mean_abs_depth_err_with_m'])} m with it, {_f(c['mean_abs_depth_err_hidden_m'])} m without (on the {_f(c['fraction_still_covered_when_hidden'], 2)} of pixels still covered)" for c in cfs[:8]) or '* none'}

## 9. Merging versus preserving unique geometry

Evidence for merging: the same-surface candidate pixels (question 2) and the
LIKELY_COMMON_SURFACE Gaussians (question 4). Evidence for preserving: the exclusive
witness mass and the patches with an observation witness (question 5). What makes an
automatic decision premature: AMBIGUOUS Gaussians (several comparable partners),
OVERLAPPING_DIFFERENT_DEPTH pairs, UNRESOLVED_LOW_EVIDENCE Gaussians, the threshold
dependence in the sensitivity tables, and the fixed-orientation renders.

## 10. Limitations and the experiment that would resolve each

| limitation | effect | follow-up |
|---|---|---|
| centres only, PLY orientation (CENTER_ONLY_FIXED_ORIENTATION) | footprints of moving splats wrong away from their endpoint | re-run Phase 1 with `--full-state-out`; this tool then uses the recorded R automatically |
| DC-only colour | view-dependent colour not modelled; RGB support pessimistic on specular surfaces | evaluate full SH at the view direction in the renderer |
| contributor depth = centre depth | thin or large splats may fail kappa although their surface is right | per-pixel depth of the Gaussian's ray intersection |
| thresholds | category sizes and labels move with them | sensitivity tables; choose from a labelled subset, not from these results |
| attribution is not causation | q is a share inside the source, not the effect of moving it to a third model | insertion counterfactuals once a third-model construction exists |
| views processed | {S['views_per_source']} per source{' (PILOT)' if pilot else ''} | full run: all frames, all cameras |
"""
    (rdir / "RESULTS.md").write_text(head + body)

    # ---- top missing geometry ---------------------------------------------
    lines = [f"# {scene}: top backward-exclusive (and forward-exclusive) patches\n",
             "A patch = Gaussians whose centres stay within the link radius over the whole "
             "track and that carry witness mass on pixels their source explains and the other "
             "does not. Not a part, not a confirmed surface; a place to look.\n"]
    for s, name in (("B", "backward"), ("F", "forward")):
        lines.append(f"\n## {name}-exclusive\n")
        for c in cases.get(s, [])[:30]:
            o = c["occlusion"]
            lines.append(
                f"### {s} patch {c['group']}: {c['size']} Gaussians, U = {c['witness_U']:.2f}\n\n"
                f"* first witnessed frame {c['first_witness_frame']}, peak frame {c['peak_frame']}; "
                f"occlusion test **{o['status']}**"
                + (f" (votes {json.dumps(o.get('votes_over_early_frame_cameras'))})" if o.get('votes_over_early_frame_cameras') else "")
                + f"\n* match labels of the members: {json.dumps(c['match_labels'])}\n"
                f"* coherence annotation (secondary, not a confidence): {json.dumps(c['coherence_annotation'])}\n"
                f"* rows: `gaussians/trajectory_witnesses/{s}_group_{c['group']:03d}.json`\n"
                f"* curves: ![](../visuals/missing_surface_case_studies/{s}_group_{c['group']:03d}_curves.png)\n")
            for pth in c.get("case_study_pngs", []):
                lines.append(f"![]({'../' + pth})\n")
            cf = c.get("counterfactual")
            if cf:
                lines.append(f"* within-source counterfactual at frame {cf['frame']} {cf['camera']}: "
                             f"depth error {_f(cf['mean_abs_depth_err_with_m'])} m with, "
                             f"{_f(cf['mean_abs_depth_err_hidden_m'])} m without; still covered "
                             f"{_f(cf['fraction_still_covered_when_hidden'], 2)}\n")
    (rdir / "TOP_MISSING_GEOMETRY.md").write_text("\n".join(lines))

    # ---- matching and ambiguity ------------------------------------------
    ml = [f"# {scene}: cross-source matching examples\n",
          "Each example shows GT, both sources' depth, the two Gaussians' own per-pixel "
          "contributions (F red, B green) at the view where they co-contribute most, and "
          "the per-frame same-pixel (O) and same-surface (O_surf) overlap.\n",
          f"Pair evidence: {pst['exactness']}\n"]
    for e in pairs:
        ml.append(f"## {e['kind']}: F{e['F_row']} - B{e['B_row']}\n\n"
                  f"rho_surf {e['rho_surf']:.3f}, rho_pix {e['rho_pix']:.3f}, incompatible fraction "
                  f"{e['incompat_fraction']:.2f}, same-surface frames {e['frames_same_surface']}, "
                  f"mean centre distance {e['mean_center_dist_m']:.4f} m; view frame {e['view'][0]} "
                  f"{e['view'][1]}\n\n![](../{e['png']})\n")
    if not pairs:
        ml.append("No pair examples (pair pass incomplete).\n")
    (rdir / "MATCHING_AND_AMBIGUITY.md").write_text("\n".join(ml))
