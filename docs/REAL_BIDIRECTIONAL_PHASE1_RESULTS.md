# Real bidirectional Phase 1 — results so far, and the runs that produce the rest

Required by section 9 of
`CODEX_PRIORITY_REAL_BIDIRECTIONAL_PHASE1_MERGE_AND_VISUALIZE.md`.

**Execution policy for this pass.** The user chose *commands only*: I ran
smoke tests and render-only baselines; every long run (the two 60-frame
Phase-1 re-runs and all IEKS solves on the full population) is one command,
`run_real_bidi_phase1.sh`, for the user to launch. Gates that depend on
those runs are `NOT_RUN` below, not inferred.

    OUT=/media/samuelemara/Extreme SSD/Samuele/results_11_10_26/ours/<scene>
    EXP=$OUT/unified_phase1_experiments/real_bidi_phase1_v1

Everything is written under `$EXP`. `static_bidi/` and `bidirectional/` are
read only.

## 1. Gates

| Flag | washer_103776 | storage_45135 | Evidence |
|---|---|---|---|
| `INPUTS_VERIFIED` | PASS | PASS | camera extrinsics identical across the two models; anchor poses equal their PLYs to 0.00e+00 |
| `FULL_STATE_FWD` | NOT_RUN | NOT_RUN | stage `phase1` of the driver |
| `FULL_STATE_BWD` | NOT_RUN | NOT_RUN | stage `phase1` |
| `FWD_RENDER_BASELINE` | PASS (centre-only) | PASS (centre-only) | `$EXP/render_baseline_center_only/` |
| `BWD_RENDER_BASELINE` | PASS (centre-only) | PASS (centre-only) | same |
| `RAW_UNION_RENDER_EXPERIMENT` | PASS | PASS | same; 45,920 / 70,039 candidates, original opacity |
| `SINGLE_SOURCE_FWD_IEKS` | NOT_RUN | NOT_RUN | stage `fwd` |
| `SINGLE_SOURCE_BWD_IEKS` | NOT_RUN | NOT_RUN | stage `bwd` |
| `RAW_UNION_IEKS_EXECUTED` | code path verified on a subsample; real run NOT_RUN | NOT_RUN | stage `union` |
| `RAW_UNION_IEKS_CONVERGED` | NOT_RUN | NOT_RUN | |
| `VALID_CANONICAL_REPRESENTATION` | UNRESOLVED | UNRESOLVED | depth agreement on doubly covered pixels |
| `REAL_MERGED_SCENE_VALIDATED` | NOT_ESTABLISHED | NOT_ESTABLISHED | |
| `VISUAL_REPORT_READY` | PASS for the baseline | PASS for the baseline | `visuals/` below |

## 2. What the real renders already show (render-only, no optimisation)

Warm starts here are `CENTER_ONLY_APPROXIMATE` — the historical Phase-1
centres with the models' own rotations held constant — because the 12-D
exports are the `phase1` stage. Mean over frames 0/15/30/45/59 and cameras
0/2/4/6, on the common domain (real depth valid AND alpha > 0.05 for every
variant):

| scene | variant | RGB rmse | depth MAE (m) | doubly covered, agreeing in depth |
|---|---|---|---|---|
| washer (revolute) | F-only track | 0.0453 | 0.0112 | |
| | B-only track | 0.0179 | 0.0079 | |
| | F+B raw union, before IEKS | 0.0173 | 0.0140 | 93.7% |
| | F+B raw union, after IEKS | NOT_RUN | NOT_RUN | |
| storage (prismatic) | F-only track | 0.0480 | 0.0248 | |
| | B-only track | 0.0405 | 0.0124 | |
| | F+B raw union, before IEKS | 0.0415 | 0.0232 | 84.0% |
| | F+B raw union, after IEKS | NOT_RUN | NOT_RUN | |

What to look at, and what it is:

* `$EXP/render_baseline_center_only/visuals/rgb_comparison_cam_002_0045.png`
  (washer): the forward-tracked initial model renders the opened door as a
  diffuse blob; the backward-tracked final model reproduces it. These are
  Phase-1 WARM-START tracks, not an optimised posterior.
* `.../alpha_sources_f_b_raw_union_before_ieks_cam_004_0059.png`: per-source
  alpha and a depth-agreement map. The body is green (both sources put a
  surface at the same depth — duplicated), the door is red (they disagree).
* `contact_sheet_<cam>_frame<NNNN>.png`: RGB, depth, |RGB error|,
  |depth error|, alpha in one sheet; `*_comparison_<cam>.mp4` steps through
  the five observed frames only, nothing interpolated.
* `gaussians_unified_timeline.html`: frame slider, initial source blue, final
  orange; points are a fixed spatial display sample, the optimiser uses every
  Gaussian. `motion_tracks_source0_sourceT.html`: a fixed stratified sample
  of tracks with source and original PLY row on hover.
* `comparison.csv`, `visual_manifest.json` (image SHA-256s, crops, scales).

## 3. Process innovations — the "176σ" claim is retracted

`tools/diagnose_process_innovations.py`, computing the PDF's innovation
`e = X_{t+1} ⊟ f(X_t)` with the tested `process_residual` and the
`make_process_noise` Q the solver assembles (cross terms included;
corr(mu, v) = 0.768). Centre-only rebuild of the historical tracks,
`$EXP/diagnostics/center_only/process_innovations_{forward,backward}.{json,html}`:

| | forward | backward |
|---|---|---|
| sqrt(chi2) median | **0.94** | **1.86** |
| sqrt(chi2) p99 | 57.7 | 43.5 |
| sqrt(chi2) max | 746 | 494 |
| share of E_process in the top 1% of terms | 69.0% | 44.6% |
| share in the top 5% | 92.6% | 77.3% |
| the invalid displacement / sqrt(Q_mu), max | 1023 | 600 |
| time reversal, rates negated / NOT negated | equal / 12x | equal / 30x |

The earlier statement compared a raw displacement with sqrt(Q[mu,mu]); that
is not a process statistic. With the correct innovation the median Gaussian
sits at about one sigma, and the process energy is a TAIL: a few percent of
(Gaussian, frame) terms carry most of it. That is "some tracks jumped", not
"Q is mis-specified". Q is unchanged.

Caveat measured in a 6-frame smoke of a real 12-D export: with the filter's
own rotations the phi block is no longer zero (chi2 median 5.5 there),
because the centre-only rebuild hides rotation innovations by construction.
The 60-frame 12-D numbers come from the `diag` stage.

`dt` is a declared convention: the dataset records `fps: null` and
`duration_sec: null`; frames are uniform in the joint coordinate q, not in a
measured clock. Rates are per declared second.

## 4. What was built and verified this pass

| file | what | verified |
|---|---|---|
| `tools/diagnose_process_innovations.py` | whitened innovations, tail share, time reversal | ran on real washer F/B |
| `tools/run_unified_phase1_real_bidirectional.py` | `render_baseline`, `forward_only`, `backward_only`, `raw_union_experiment` | real renders on 2 scenes; joint run on a subsample; 12-D path on a real export |
| `bidirectional_iekf/real_visuals.py` | rgb/depth/alpha comparisons, contact sheets, videos, CSV, three HTMLs | real outputs above |
| `bidirectional_iekf/ieks.py` | `anchor_groups` (per-source anchors), `on_accept` (best accepted saved as it goes), preconditioner kept inside the anchored subspace | `tests/test_unified_union_anchors.py`, 10 checks, plus 7 regression suites |
| `run_real_bidi_phase1.sh` | the whole chain, resumable, collision-safe | syntax + its refusal path |

The raw-union run's contracts, each checked on real CUDA
(`tests/test_unified_union_anchors.py`): the initial rows' pose is eliminated
at frame 0 and the final rows' at frame 59, each only for its own rows; rates
stay free; a solve leaves both anchors bit-identical while unanchored poses
move; a tangent on one source produces image information on the other's rows
(the shared composite couples them). One factor per `(original_frame,
camera)`; no F/B energy addition.

Smoke of the joint path on washer (1,500 Gaussians per source, 60 frames,
cam_000, 2 outer iterations): 60 sealed factors, both anchors, 2 accepted
steps, best-accepted state on disk, every visual written. Status MAX_ITERS by
construction of the budget — it is a code-path check, not a result.

## 5. The command

    cd /home/samuelemara/Joint3DGS/RigEuler3DGS
    SCENE=washer_103776 bash run_real_bidi_phase1.sh
    SCENE=storage_45135 bash run_real_bidi_phase1.sh

Stages (`STAGES=` selects a subset; finished stages are skipped on re-run):

1. `phase1` — the existing free Phase 1, forward and backward, with exactly
   the settings `run_bidirectional_all.sh` used plus `--full-state-out`, into
   `$EXP/free_{forward,backward}/`. Also writes how closely the re-run
   reproduces the historical centres. Expect ~15 min each.
2. `diag` — 12-D innovations of both exports.
3. `baseline` — real | F | B | raw union, from the 12-D states.
4. `fwd` — forward-only IEKS on frames 0:8, 0:16, 0:60 (2 cameras).
5. `bwd` — backward-only on 52:60, 44:60, 0:60.
6. `union` — the joint raw-union IEKS over all 60 frames (both anchors in
   the window), 1 then 2 then 4 cameras (`UNION_MAX_CAMS=8` adds the 8-camera
   rung).
7. `report` — one comparison over every best accepted state:
   real | F track | B track | union before | F IEKS | B IEKS | union IEKS.

Progress and one-line outcomes go to `$EXP/logs/driver.log`; each stage's
full log, manifest (status, true PCG residual, GPU/host peaks, inventory,
hashes) and visuals are in its own directory.
