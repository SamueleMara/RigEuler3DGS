# Phase-1 covariance, false velocity and real RGB-D divergence: implementation report

Spec: `RigEuler3DGS_Phase1_Covariance_Velocity_Corrections_Codex.md`.
Branch `phase1-covariance-velocity`, based on `90dbf58` (the spec inspected `7954660`; the only
intervening commit adds `--direction` and the orchestrator's `sys.path` fix, which don't touch
Phase-1 numerics). **No full or real-data run was made for this report.** The CPU and CUDA oracles
below are toy-sized. No claim is made that tracking improved.

Reproduction identifiers (from the saved run, not re-run):

| | |
|---|---|
| scene | `Sapien/output/real/final_red_fridge_episode_000000`, frames 100–288, 3 cameras, dt 0.04 s |
| run | `results_11_10_26/ours_v4_calibrated_bootstrap/final_red_fridge_episode_000000` (code `7954660`) |
| sensor_model_id | `4588849703872584d30585e70edf796004601ef868ce83120070d99e3ee03e0c` |

## 1. Diagnosis, classified

| Class | Finding | Evidence |
|---|---|---|
| MEASURED FAILURE | Forward Phase 1 diverges: median \|dx\| 2.7 mm at 101, 0.31 m at 140, 2.6 m at 288; max 291 m; no Gaussian stays within 5 cm. Backward diverges the same way. | `phase1_centers/*` of the saved run |
| MEASURED FAILURE | 99.8% of the divergent Gaussians start moving while still projecting into a mask (median onset frame 111). Mask exit and coasting come later (42% are off-mask or out of view at the end; step change about 4%/frame there). | projection of every centre into every camera, mask and depth |
| MEASURED FAILURE | All 188 transitions stop at `iteration_budget`. After about frame 180 the per-frame correction is under 1 mm while the step is about 40 mm, i.e. pure extrapolation. | `phase1_frames.json`, step differences |
| MEASURED FAILURE | At the canonical frame, in-mask depth MAE is 20 mm against a calibrated sensor sd of 0.7 mm, and RGB rms about 0.085 against about 0.0045. | `mask_diagnostics.json`, `masked_observation.json` |
| CONFIRMED CODE DEFECT | Phase 1 passed `partition=None`, so the IEKF regrouped supernodes every frame from the predicted, already-drifting means. | `tools/kpart_pipeline.py::bootstrap_positions`, `estimator/sparse_iekf.py` |
| CONFIRMED CODE DEFECT | The audit's per-camera Fisher omitted the alpha block that the production Fisher contains (relative error 0.17 on the masked toy scene). | test `[A]` |
| CONFIRMED CODE DEFECT | GT labels: the fixed-count Phase-0 bootstrap wrote no `gt_part`/`gt_dynamic`, so every synthetic Phase-1 GT evaluation was `OPTIMIZED_LABELED_3DGS_GT_NOT_FOUND`. Unlabelled Gaussians would have been saved as `gt_dynamic = 0` (static). The sparse-NN diagnostic always read the initial-endpoint cloud, which is wrong for backward runs. | section 4 |
| EXPLANATORY HYPOTHESIS (consistent, not proven on real data) | A biased position correction δp becomes velocity δv ≈ δp/dt through the predicted pose–velocity cross covariance (P0_vv = (1 m/s)², pose sd about 0.19 mm). The constant-twist prediction then carries it. The equations reproduce it exactly (test `[F1]`: 5 mm gives 0.1250 m/s). Whether the real first-frame correction follows this linear conditional is what the Gate-A audit measures (`velocity_gain.explained_fraction`). | test `[F1]`; audit |

## 2. What changed (by gate)

### Gate A: audit (`tools/audit_calibrated_phase1.py`; read-only, diagnostic)
- The per-camera Fisher now includes the alpha block with the production `kappa`. `fisher_check` compares Σ_c F_c with `measurement_operators(...)[2]()` on the same linearization (toy: 6e-8 relative).
- `forces_at.{canonical,prior}`: per-camera channel energies, sample counts and forces at the canonical (zero-displacement) state and at the propagated prior, i.e. a common pre-update state. Posterior forces are kept as before.
- `velocity_gain`: accepted twist change vs the linear conditional P_tp P_pp⁻¹ δpose, the speed change, and `dv_over_dmu_per_dt` (1 = the whole correction was booked as velocity).
- `attribution`: own measurement / moved only through its group / prediction-only (no information, no update), so unmeasured moves are never attributed to a camera.
- `ray_motion`: the accepted update along and across each camera ray (all Gaussians and the top 1%).
- `solver`: every inner iteration, the stop reason and `posterior_status` (`UNCERTIFIED_NOT_CONVERGED` unless `converged`).
- `fd_check`: J·v against central finite differences of the production render, over a step ladder 1e-3…1e-6 m (large steps cross sorting/early-termination discontinuities). The best step per camera is reported.
- `residuals`: signed depth/RGB residuals by region (mask interior, 3-px boundary band, observed-depth discontinuity > 2 cm), with `fraction_render_behind`. These are diagnostic categories, never gates.
- `canonical_control`: the canonical state against its own endpoint frame. This is a control, not an independent test.
- `--residual-report-only`: no tracking. The canonical render is compared with every frame of a window the caller asserts is stationary. Per pixel it reports the temporal mean (persistent part) and temporal sd against the sensor sd, by region. This is section C1's split; nothing estimates or sets a sigma.
- `--snapshot-dir`: per-frame npz with the predicted and posterior 12D means and the full P_prev, P_pred, P_post, F and Fisher blocks.
- Audit-on/off invariance: identical within the GPU floor measured from 3 control runs (toy: 7.2e-7 with audit vs 9.5e-7 between plain runs).

### Gate B: fixed canonical partition (CONFIRMED CODE DEFECT, fixed)
- `bootstrap_positions` builds the supernode partition once, from `truth.mu`, which is the endpoint model of the direction being processed (forward: initial, backward: final). It holds it for the whole pass and asserts the hash every frame. The hash, config and member counts go into `phase1_frames.json`; membership and canonical centres go into `phase1_partition.npz`.
- **Deviation from the spec's pseudocode:** membership is canonical, but the group tangent's lever arms are evaluated at each frame's predicted means and member centroids (`group_geometry="current"`). With canonical lever arms, a pure group-rotation increment on a group already rotated 1.2 rad is non-rigid (pairwise-distance derivative 0.73, relative). With current lever arms it's 2e-8 (test `[B]`). The `G`/`Gᵀ` adjoint holds with the override.
- `--phase1-partition per-frame` restores the legacy regrouping as arm A. A caller-supplied partition (`tools/kpart_pipeline.run`) keeps its old semantics exactly.

### Gate C: likelihood (EXPERIMENTAL arms, defaults unchanged)
- C2: `--alpha-precision coupled|declared`, `--alpha-sd`, `--rgb-support full|foreground` (masked expected-quad only).
  - `coupled` is today's kappa = trace(W)/12 (= 1e4 at σ_rgb 0.005).
  - `declared` uses kappa = 1/alpha_sd², independent of W. The objective is then no longer the exact expectation over b, as documented in `masked_observation.py`.
  - `foreground` zeroes the RGB precision off-mask, so the silhouette is the declared factor alone. It requires `declared`.
- C1/C3: declared mismatch `--rgb-model-sigma/--depth-model-sigma` already existed (R_eff = R_sensor + R_model, added in variance). Its value must come from the C1 report. No sigma is set automatically.
- Not implemented: a joint/correlated residual covariance, a nuisance-bias state, or a Bernoulli opacity likelihood.

### Gate E: covariance
- E1/E4: every posterior is labelled `approximation = BLOCK_INFORMATION_APPROXIMATION` and `posterior_status = UNCERTIFIED_NOT_CONVERGED | LOCAL_HESSIAN_AT_CONVERGED_STATE` (IEKF `posterior` row, Phase-1 per-frame covariance report). `converged_frames` is counted.
- E2: dense CPU oracle (2–10 Gaussians, 2 cameras, co-visible and occluded rows):
  - the production blocks equal (Λ_ii + F_ii)⁻¹;
  - the exact marginal dominates them in Loewner order;
  - measured overconfidence on position variance: up to 1.12× in the toy;
  - the group covariance G Cov(z) Gᵀ has member cross-correlation 0.33, which the blocks drop;
  - a shared bias yields fully correlated errors across Gaussians.
- E5: `--phase1-covariance-transport` (opt-in) uses Λ_post = JᵀΛ_prior J + F, with J = blockdiag(I, J_l(a)⁻¹, I, I) and a = Log(R R_priorᵀ). It is FD-verified: Log(Exp(e)Exp(a)) − a − J_l(a)⁻¹e is O(\|e\|²). The in-loop prior gradient keeps the usual first-order J ≈ I.
- E3 (probe/sandwich covariance): NOT IMPLEMENTED (needs E2-validated design first).

### Gate F: velocity
- F1: instrumentation only (audit `velocity_gain`, counterfactual test).
- F2 (initial twist from a short multiview window) and F3 (pose-only fixed-lag prototype): NOT IMPLEMENTED. No speed damping, clipping or static labels were added, per the spec.

### Gate D: reference-relative / nuisance-bias factor
NOT IMPLEMENTED. It needs visibility-aware residual transport and shared-reference covariance, i.e. a separate prototype.

## 3. Plumbing and signatures
- Phase-1 flags are passed through `tools/run_noise_aware_rigeuler.py` → `run_bidirectional_all.sh` (Phase 1 only) → `tools/run_motion_coherence_support.py`: `--phase1-partition`, `--phase1-covariance-transport`, `--phase1-alpha-precision`, `--phase1-alpha-sd`, `--phase1-rgb-support`.
- All of them are in the driver's stage-signature config, so arms never reuse each other's outputs.
- The audit tool accepts the same flags.

## 4. Static / dynamic GT labels (EVALUATION ONLY)
- Synthetic inputs ship `pointcloud/points3D_parts.ply` (initial) and `points3D_parts_final.ply` (final), each with `part` (0 static, ≠0 moving). Real captures have neither, so nothing is labelled.
- **Static training with cloning (`train.py`):** labels attach at initialization when the COLMAP points are the labelled cloud (all 10 synthetic scenes: identical order, error ≤ 6e-8). Clone, split and prune inherit them exactly (CUDA test `[GTc]`: 556 Gaussians from 200, every child carries its parent's label, including −1). `save_ply` now writes `gt_dynamic = −1` for unlabelled Gaussians instead of 0.
- **Fixed-count Phase-0 bootstrap (no cloning):** each seed takes the part of the nearest labelled sample of the **same endpoint**. Seeds farther than 4× the cloud's median spacing get −1. IDs are fixed through the refinement. Fields `gt_part`/`gt_dynamic` go into every checkpoint PLY, and `bootstrap_manifest.json` gains `gt_labels` (cloud path, sha, method, counts).
- `--gt-labels auto|off` is on the bootstrap. The orchestrator passes `off` for `--dataset-kind real` and records it in the bootstrap signature.
- `estimator/optimized_gt.py`: unlabelled is −1 and excluded from every denominator (coverage reported). Provenance names the lineage actually used.
- `load_static_dynamic_gt(..., endpoint)`: the backward sparse-NN diagnostic reads `points3D_parts_final.ply`.
- Historical outputs are untouched. Older `ours/*/static_bidi/final` models without labels (e.g. blade, foldchair, fridge) predate the final-cloud lookup.

## 5. Tests

| Test | Result |
|---|---|
| `tests/test_phase1_covariance_velocity.py` (new; CPU + CUDA) | PASS, 47 checks |
| `test_surgical_review_4c88725` | PASS, 46 |
| `test_calibrated_bootstrap` | PASS, 49 |
| `test_masked_rgb_observation` (CUDA) | PASS, 46 |
| `test_masked_default_repair` | PASS, 77 |
| `test_v2_group_cuda` | PASS, 26 |
| `test_v2_grouped_solve` | PASS, 16 |
| `test_motion_coherence_support` | PASS, 41 |
| `test_iekf_end_to_end` | PASS, 26 |
| `test_surgical_fixes_2026_10_22` | PASS, 40 |
| `tests/test_bootstrap_geometry_eval.py` | 2 known FAILs kept visible (normal 2.9° vs 2° tolerance; 71% merged thin-panel seeds); Gate G not addressed |
| real red_fridge 100–105 audit, arms A–D | NOT RUN (commands in the hand-over) |
| real stationary window residual report (80–99) | NOT RUN |
| synthetic SAPIEN A/B with GT evaluation | NOT RUN |

## 6. Recommendation
None of the arms is promoted. The one default change is the fixed canonical partition, which fixes a confirmed defect; `per-frame` remains for the A/B. Decide on C/D only after:
- the stationary-window report has quantified the persistent vs fluctuating residual;
- the 100–105 audit shows whether the first correction is booked as velocity through the pose–velocity conditional;
- the synthetic A/B shows that moving-point tracking and Phase-2/3 joint accuracy are unchanged.
