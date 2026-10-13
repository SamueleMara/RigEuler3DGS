# Unified bidirectional pipeline — completion report after `f80e3e1`

Required by section 7 of
`CODEX_NEXT_AFTER_F80E3E1_COMPLETE_ARTICULATED_PIPELINE.md`. Every number
below was produced by a command in this document. Counts record what
executed; they are not acceptance.

## 0. Stage states

| Stage | State | Where the evidence is |
|---|---|---|
| Regression accounting | `IMPLEMENTED_AND_TESTED` | `tools/run_test_ledger.py`, `docs/test_ledger.json` |
| GT isolation (1.3) | `IMPLEMENTED_AND_TESTED` | `tests/test_unified_gt_isolation_and_gauge.py` |
| World gauge (1.4) | `IMPLEMENTED_AND_TESTED` | same, 8 extrinsics identical to 0.000e+00 |
| Physical time (1.5) | `IMPLEMENTED_AND_TESTED` | same; `fps: null`, `duration_sec: null` |
| Likelihood labelling (1.6) | `IMPLEMENTED_AND_TESTED` | `GAUSSIAN_RGBD` in every manifest |
| Full 12-DOF warm start (2) | `IMPLEMENTED_AND_TESTED` | `bidirectional_iekf/free_state.py`, 19 checks |
| Full-state on the real scene | `NOT_RUN` | needs a 60-frame Phase 1; command in §5 |
| `FULL_WINDOW_EXECUTES` | `PASS` | 60x8, 480 factors, 2265 MiB, 43.6 s |
| `REAL_SOURCE_ONLY_CONVERGED` | `PARTIAL_NUMERICAL_VALIDATION` | §3 below; MAX_ITERS with a measured cause |
| True PCG residual (3.1) | `IMPLEMENTED_AND_TESTED` | agrees with the recursive one to 5 figures |
| Anchor policy honesty (3.2) | `IMPLEMENTED_AND_TESTED` | a window without its anchor is refused |
| `T10_ENGINEERING` | `IMPLEMENTED_AND_TESTED` | 42 checks, accepted IEKS -> production extractor |
| `T10_SEMANTIC_SUPPORT` | `PARTIAL_NUMERICAL_VALIDATION` | §4 below; the limitation is located |
| Convergence certificate (4) | `IMPLEMENTED_AND_TESTED` | Newton decrement, initialization-independent |
| Overlap analysis (5) | `IMPLEMENTED_AND_TESTED` | depth character, 11 checks incl. a layered control |
| `T11-merged` | `UNRESOLVED_REPRESENTATION` | §5 below |
| `REAL_SOURCE_ONLY_COHERENCE` | `NOT_RUN` | blocked on `REAL_SOURCE_ONLY_CONVERGED` |
| P2 rigid SE(3) | `NOT_RUN` | §6 below |
| P3 revolute/prismatic | `NOT_RUN` | §6 |
| P4 fixed geometry | `NOT_RUN` | §6 |

## 1. The ledger

    python tools/run_test_ledger.py

**19 suites executed, 505 checks, 0 failures** — computed by the tool from
parsed output, with each suite's exit code, wall time and log sha256 in
`docs/test_ledger.json`. The earlier "394" was typed and wrong; the list it
referred to summed to 437.

## 2. Audit findings that changed what is believed

* **Ground truth cannot reach the estimator.** Three independent arguments:
  no production module names any GT symbol (12 modules, parsed identifiers,
  plus `getattr`/`setattr` string access); a model loaded from its trained
  PLY has `_gt_part_labels.numel() == 0`; and injecting a distinct label per
  Gaussian leaves all seven state tensors bitwise equal.
* **The two endpoint models are in ONE world gauge.** Their 8 camera
  extrinsics and intrinsics differ by `0.000e+00`. No world-to-world map is
  derived and none is guessed. This was an open risk and the measurement
  closed it favourably.
* **`dt` is not a measured fact.** `fps: null`, `duration_sec: null`. What
  the dataset spaces evenly is `q`: 0.0225608954 rad per frame, relative
  spread 8.0e-11, 0 to 1.331093 rad. Every manifest now says so.
* **The Washer FORWARD Phase 1 is not missing** (recorded at `f80e3e1`): 60
  frames, 21,338 Gaussians, and its coherence classifies all 21,338 as
  moving, so it provides no static reference.

## 3. `REAL_SOURCE_ONLY_CONVERGED` = NOT_YET, and why

    python tools/run_unified_phase1_single_source.py \
      --endpoint final --frame-start 56 --frames 4 --cameras 2 \
      --max-outer 12 --pcg-iters 80 --pcg-tol 1e-6 --pcg-true-residual \
      --source "$SRC" --out-root "$OUT" --out <new dir>

washer final, frames 56-59 (the source anchor is inside the window), all
24,582 Gaussians, 8 sealed factors, 305,528 covered depth pixels:

    E 2.609054e+06 -> 8.984207e+05, 12 accepted outer, 63.6 s, 1132 MiB
    every PCG solve capped at 80 iterations, true residual 2.1480e-02
    (recursive 2.1480e-02 -- they agree to five figures)
    |g| 1.27e+07 -> 6.29e+06 then flat; step norm ~44; lambda -> 5.3e-10
    status MAX_ITERS

`--diagnose-objective` on the same window:

    image energy    1.8326e+05
    process energy  2.4258e+06   93.0% of the total
    located in      mu 2.737e+06, v 1.716e+06, phi 0, omega 0
    the prior allows 3.960e-04 m of unexplained position per frame
    the Phase-1 centres move up to 6.971e-02 m per frame  -- 176 sigma

The full-window objective is 93% process term. `process_noise_pos = 1e-3` is
a continuous-time density chosen for the per-frame filter, where each step
starts from a fresh covariance; a full-window smoother applies it between
every consecutive pair at once. The optimiser's first job is therefore to
smooth the Phase-1 trajectory rather than to fit the images. Reported, not
retuned.

A guess the measurement refuted: the fabricated central-difference
velocities are not the cause. Setting `v = 0` instead gives `2.1377e+07`,
nine times worse.

**Scaling, all measured, peak CUDA at the system build:**

| frames x cameras | factors | resident | +stream | +host Q | +diag precond |
|---|---|---|---|---|---|
| 4 x 1 | 4 | 878 MiB | | | |
| 8 x 2 | 16 | 1901 MiB | | | |
| 16 x 4 | 64 | **OOM** 4147 | 3611 MiB | | |
| 30 x 8 | 240 | | **OOM** 3950 | 4504 MiB, 22.2 s | |
| 60 x 8 | 480 | | | **OOM** 4155 | **2265 MiB, 43.6 s** |

`FULL_WINDOW_EXECUTES = PASS`; `REAL_IEKS_CONVERGED = NOT_YET`.

## 4. `T10_SEMANTIC_SUPPORT` = PARTIAL

    python tests/test_unified_coherence_semantics.py

`moving_and_static` is clean: recall 14/14, precision 14/14, stationary
specificity 26/26, core spanning both sources, 11 members from the selected
candidate and 3 from refinement with none unattributable.

The limitation, located rather than asserted:

| scene | nodes | eta_cv | rel motion | world dRSS/nf | dBIC |
|---|---|---|---|---|---|
| static_only | 27 | 0.9840 | 9.585e-05 | -7.640e-13 | 5.773e+02 |
| moving_and_static | 11 | 1.0000 | 4.791e-02 | 8.706e-04 | 6.760e+02 |
| incoherent_and_static | 29 | 0.9886 | 8.960e-02 | 1.141e-07 | 7.274e+02 |
| weakly_observed | 33 | 0.9994 | 7.277e-01 | 9.526e-05 | 1.491e+03 |

The world-frame dRSS per node-frame separates a real body from a body-free
scene by 7.6e+03. The qualification statistic does not — `relative_motion_
ratio` is *larger* without a body. The separating statistic is used only to
tie-break among already-qualified candidates. Closing the gap needs a
size-neutral null test, which is a change of method and was not made on the
strength of labelled fixtures.

## 5. `T11-merged` = UNRESOLVED_REPRESENTATION, on stronger grounds

    python tools/run_unified_representation_gate.py \
      --source "$SRC" --out-root "$OUT" --frames 0 15 30 45 59 \
      --cameras 4 --with-observations --save-images

Written to `$OUT/unified_phase1_tests/representation_gate/`.

| frame | doubly covered | of coverage | agree in depth |
|---|---|---|---|
| 0 | 147,723 | 94.5% | 98.7% |
| 15 | 150,915 | 94.3% | 97.6% |
| 30 | 157,517 | 94.7% | 94.8% |
| 45 | 162,516 | 92.7% | 93.4% |
| 59 | 160,448 | 86.2% | 96.0% |

Projected overlap alone would not settle this — two depth-separated parts
share pixels too, and `tests/test_unified_representation_depth_character.py`
proves the gate can tell them apart with a 0.6 m layered control. Agreement
*in depth* at 93-99% is the two mixtures placing a surface at the same place.

RGB rmse / depth rmse against the real calibrated frames:

| frame | initial only | final only | union |
|---|---|---|---|
| 0 | 0.0042 / 0.0092 m | 0.0280 / 0.0360 m | 0.0278 / 0.0364 m |
| 30 | 0.0295 / 0.0406 m | 0.0403 / 0.0322 m | 0.0458 / 0.0466 m |
| 59 | 0.0446 / 0.0541 m | 0.0063 / 0.0241 m | 0.0404 / 0.0505 m |

The union fits the real images **worse** than the better single source at
every frame. Section 5.4 warns against promoting a low union residual to
proof; the measurement went the other way, which is stronger evidence for
the same verdict.

The gate is labelled `CENTER_ONLY_FIXED_ORIENTATION`: Phase 1 saved centres
only, and a splat's footprint depends on its covariance orientation too. The
full-state export removes that approximation once a 60-frame Phase 1 has
been re-run.

**Methodological blocker, unchanged and explicitly not worked around.** The
theory gives a measurement likelihood for a FIXED Gaussian representation. It
does not supply an identifiable association or selection model for two
independently trained mixtures. Exploiting their complementary visibility
without double-counting needs such a model, developed and validated as a new
extension. Nothing was rescaled, pruned, deduplicated or masked.

## 6. P2, P3, P4 — NOT_RUN

Not reached in this pass, and not claimed. What each needs, so the next pass
starts from the right place:

* **P2 rigid SE(3).** `g_i(t) = T_t T_{s_i}^{-1} g_i^{s_i}` with the source
  pose in the state, analytic current-time and source-time pose Jacobians
  checked against finite differences including the `t = s_i` cancellation,
  and a dedicated RGB-D IEKF/IEKS — not a Kabsch fit and not a rigidity
  penalty on the free solve. It can be built on a validated synthetic
  single-canonical scene today; the real two-source case is blocked by §5.
* **P3 revolute and prismatic.** Two independent hypotheses, each with one
  globally constant geometry (`Q_theta = 0`) and temporal `(q_t, qdot_t)`,
  source-relative transport `T_J(q_t) T_J(q_{s_i})^{-1}`, and
  `H_J = H_G J_{G,J}` validated by CUDA finite differences with a stable
  active set. The dataset's own answer (revolute, 0 to 1.331 rad) is in
  `metadata/joint_info.json` and is read only in tests.
* **P4 fixed geometry.** Reduced `(q_t, qdot_t)` state with the geometry
  frozen, reporting the conditional covariance separately from the total.

## 7. Commands that belong to the user

    REPO=/home/samuelemara/Joint3DGS/RigEuler3DGS
    PY=/home/samuelemara/miniconda3/envs/rigeuler3dgs/bin/python
    SRC='/media/samuelemara/Extreme SSD/Samuele/Backup_09_10_26/Sapien/output/preparation_8cams/partnet_subset_fixed_render_linear/washer_103776'
    OUT='/media/samuelemara/Extreme SSD/Samuele/results_11_10_26/ours/washer_103776'

**The one that unblocks the most.** A 60-frame Phase 1 that saves the full
12-DOF state, into a NEW directory so the existing forward/backward results
are untouched:

    cd $REPO && $PY tools/run_motion_coherence_support.py \
      --scene washer_103776 \
      --model "$OUT/static_bidi/final" --source "$SRC" \
      --direction backward --canonical-endpoint final \
      --frames 60 --cameras 8 \
      --out "$OUT/bidirectional/backward_fullstate" \
      --full-state-out "$OUT/bidirectional/backward/phase1_full_state"

and the same with `--model "$OUT/static_bidi/initial" --direction forward
--canonical-endpoint initial --out "$OUT/bidirectional/forward_fullstate"
--full-state-out "$OUT/bidirectional/forward/phase1_full_state"`.

Then the real single-source IEKS from a state that was not reconstructed:

    cd $REPO && $PY tools/run_unified_phase1_single_source.py \
      --endpoint final --frame-start 0 --frames 60 --cameras 8 \
      --warm-start full_12d \
      --stream-frames --frames-on-cpu --process-factors-on-cpu \
      --diagonal-preconditioner --pcg-true-residual --diagnose-objective \
      --max-outer 8 --pcg-iters 40 \
      --source "$SRC" --out-root "$OUT"

Measured on a synthetic rotating fixture, the centre-only warm start STALLS
at a 0.825 rad sweep and the real Washer sweep is 1.331 rad, so this is the
difference between a diagnostic and a result.
