# Masked IEKF default and validation repair (2026-10-21)

Specification: `CODEX_FIX_MASKED_IEKF_DEFAULT_AND_VALIDATION_2026-10-21.md`.
Branch: `masked-default-repair`, from `4379c0f`.
Machine-readable evidence: `tools/repair_report.py --root <root> ... --out <json>` over any run (section 5).

**Scope.** This report covers code changes, unit and toy tests, and two-frame smoke runs on one scene. It is not
production validation. Every full-scale check is marked **NOT RUN** in section 4 and is the
user's run (commands in section 5).

Unchanged: the number of moving parts, support / joint identification, static training (`train.py`), the `_C`
ABI, the datasets, and the historical results.

## 1. What changed, item by item

### P0-1 Default observation mode

| before | after |
|---|---|
| Every motion entry point defaulted to `original`. The masked mode was opt-in. | `estimator/masked_observation.py` sets `DEFAULT_MODE = "masked-rgb-expected-quad"`. This is shared by the three runners through `add_cli`, and by `run_bidirectional_all.sh` (`OBS_MODE`). `original` must be passed explicitly; `masked-rgb-constant` is an ablation. |
| Missing masks were found late, or never. | `masked_observation.preflight` (all runners) and `tools/check_masks.py` (driver) refuse missing, ambiguous, wrong-size or non-binary masks before any model load, training or deletion. The driver reports `SKIPPED(masks not ready)`. There is no fallback to `original`. |
| The hard QP could be combined with any mode. | The hard negative-ray QP requires an explicit `--observation-mode original`. The masked modes refuse it. |
| — | Each stage writes `observation_mode.json` (the effective mode, background, masks and precision policy). The driver echoes the effective mode. |

### P0-2 Phase 2 / Phase 3 MAP acceptance (`estimator/joint_iekf.py`)

Before, the step was solved on `E_meas + prior` (b and A), but accepted on `E_meas` alone. A trial that improved the image but
worsened the total MAP was therefore accepted. Its gain ratio compared a MAP prediction against an image-only actual decrease.

After, one objective is used for the solve, the prediction, the acceptance and the posterior:

```
E_total(x) = E_meas(x) + w/2 e(x)^T Lambda_minus e(x),      e(x) = x boxminus x_minus
b = J_meas^T W r - w J^T Lambda_minus e,     A = J_meas^T W J_meas + w J^T Lambda_minus J
pred = b^T s - s^T A s / 2,    actual = E_total(x) - E_total(x boxplus s),    accept iff E_total decreases
posterior information = H_meas(x_accepted) + w J^T Lambda_minus J
```

`_FramePrior` takes one snapshot of `x_minus` and `Lambda_minus`, after the prediction and before any correction.

- **Exact manifold error.**
  - Rigid state, `RigidState.prior_error`: the left error `tau = Log(T T_minus^-1)` (the exact inverse of the retraction
    `T+ = Exp(tau) T`) plus the twist difference.
  - Displacement state, `DisplacementOnlyState.prior_error`: `(q - q_minus, qdot - qdot_minus)`, with `J = I`.
  - Elsewhere, J is a central finite difference through the state's own retraction. The state is small (n <= 12) and
    no render is involved.
  - E_prior is a function of the physical state only, so it does not depend on the solver's chart.
- **Chart mode.** This applies to states without `prior_error`; only the legacy full-joint `JointState` is one. The
  tracked offset is now used in acceptance too. After a basis reset `e_new = G e_old`, the information is transported as
  `G^-T Lambda G^-1`, which conserves the energy. The pre-repair code used `G^-1 Lambda G^-T`, which is wrong unless G is
  orthogonal.
- **Per-iteration history** (`FrameRecord.observability["iterations"]`, written as `map_objective.json` by both
  runners):
  - the RGB, alpha, depth, prior and total energies;
  - predicted and actual decrease, and rho;
  - accepted or not, and the trust radius;
  - depth used and gated;
  - the prior mode.
- **Legacy diagnostic (scoped).** The legacy joint-identification diagnostic `estimator/prismatic_null.py` is not in the
  active pipeline; `tests/test_rigid_se3_phase2.py` S17 forbids the rigid runtime from importing it. Its score test and
  section-19 stationarity audit are calibrated on the old frame step. With the repaired step, its null fit no longer
  converges in `tests/test_efficient_p_null.py` C2. This is deterministic over 3 runs per tree:
  - pre-repair: 3/3 `REVOLUTE_CANDIDATE`;
  - repaired step: 3/3 `UNRESOLVED_GEOMETRY`.

  The spec does not allow joint identification to change. So that diagnostic passes
  `JointIEKFConfig(acceptance="pre_repair_legacy")`, which reproduces the old step exactly: measurement-only acceptance and
  the old transport. C2 is again `REVOLUTE_CANDIDATE`, at main's objectives within the CUDA floor. Every Phase 2/3 runner
  uses the default `acceptance="map"`. This is a **known inconsistency left in a non-active diagnostic**, not a fix.

### P0-3 Phase 1 recursion (`tools/kpart_pipeline.py: bootstrap_positions`)

Before, P was re-initialised to P0 every frame. After:

- P0 is built once. Each frame passes the previous frame's returned posterior to `iekf_frame_step`, which predicts it
  itself (F, Q).
- `--phase1-covariance-reset` restores the old behaviour as an explicit, non-default diagnostic.
- Each frame records `covariance_report` in `phase1_frames.json`:
  - block count, non-finite blocks, minimum eigenvalue, PSD;
  - median and maximum condition number;
  - mean position, rotation, velocity and angular-velocity variance;
  - the prior source.
- The report is computed on the CPU. The batched CUDA eigensolver asked for a 15 GiB workspace at N = 27k; this was
  found by the end-to-end smoke run.

**Posterior defect found by the recursion test.** `estimator/observability.py: inflate_weak_covariance` rebuilt the
pose block as `U diag(raised) U^T`:

- That dropped every pose-pose correlation not aligned with the Fisher eigenbasis, while the pose-velocity cross terms
  were kept.
- The 12x12 block went indefinite (minimum correlation eigenvalue -0.23), and `project_psd` then distorted it.
- A zero-information frame came back 32% away from `F P F^T + Q`. This was the same on main. It was harmless while P
  was reset every frame, but it is carried forward under recursion.

Now the block is raised by the PSD update `P + U diag(max(floor - var, 0)) U^T`. A zero-information frame returns
`F P F^T + Q` to 2.3e-7, with no floored blocks.

### P0-4 Phase 1 convergence (`estimator/sparse_iekf.py: convergence_decision`)

```
rel_decrease = (E_before - E_after) / max(|E_before|, 1)
stop  iff  rel_decrease < rel_energy_tol  and  step_norm < step_norm_tol
```

- `E_before` and `E_after` are evaluated before the state is replaced. The old check could compare an energy with
  itself and see a fake zero decrease.
- On the trust-region path, `step_norm` is the scaled norm.
- Every frame logs one `stop` row: `converged`, `trust_radius_floor`, `lm_max` or `iteration_budget`.

### P0-5 Experiment signatures (`tools/stage_signature.py`, driver)

- Every stage writes `stage_manifest.json` (running, complete or failed), atomically. `complete` validates the outputs
  first.
- The signature hashes:
  - the code commit and uncommitted diff;
  - stage, direction and scene;
  - the static PLY and `cfg_args` sha256;
  - the observation mode, background, hard-QP options, mask root and mask content sha256;
  - the source rgb/depth listing sha256, frames and cameras;
  - the driver config;
  - the upstream stage's signature.
- What the driver does with a stage:
  - **reuse** it only if it is complete, has the identical signature and validates;
  - **redo** its own incomplete attempt;
  - **refuse** a legacy (no manifest) or differently configured output, printing the field diff. `FORCE=1`
    overwrites; nothing is deleted automatically.
- Phase 2/3 require a complete upstream with a matching signature.

### P1-1 Robust depth (`estimator/measurement.py`, `estimator/observation_terms.py`)

- **One objective for RGB and depth.** `CameraTerms` is the single definition of the per-camera RGB-D terms in all three
  phases. The objective is the Student-t NLL `(nu+1)/2 log1p(z^2/nu)`. Its derivative is exactly the IRLS score
  `w(z) r / sigma^2`, so the GN precision is its local model.
  - Before, the acceptance energy was `1/2 sum w(z) z^2` with weights re-evaluated at each trial. That is not a fixed
    function: it saturates at nu + 1 per outlier, and it prefers a candidate whose outlier explodes from z = 10 to 100
    (test [C]).
- **One sigma per sample.** `sigma_eff^2 = sigma_sensor^2 + sigma_model^2` is used in the weight, the precision and the
  objective.
- **Masked RGB is unchanged.** It stays fixed-quadratic (no Student-t).
- **Invalid depth.** It is excluded by indexing before any arithmetic. Validity requires finite and > 0 (positive
  camera Z): a zero depth flagged valid is no longer a measurement.
- **Depth counts.** The pre-existing predicted-alpha gate `alpha > min_alpha_for_depth` stays. Observed-valid, used and
  gated counts are logged per camera and iteration.

### P1-2 Model-error confound (`estimator/model_error.py: model_error_audit`)

This is reproduced, measured and reported, not fixed:

- Each Phase 2/3 run writes `model_error_audit.json`: `W / W_sensor` and `alpha0` per camera on foreground, background
  and leaking-background pixels.
- In a toy with stray canonical opacity 0.4 on the background, the canonical map lowers W to 0.41 x sensor on exactly
  those pixels. kappa = trace(W)/12 drops with it, which down-weights the evidence that would remove the leakage.
- Separating irreducible appearance error from correctable opacity error needs observations that the canonical frame
  alone does not provide. So the policy is unchanged, the ambiguity is reported, and no gamma was invented.
- W stays fixed across a frame (test [F]).

### P1-3 Data path

- `prepare`: `V_used = V_orig AND M AND finite AND depth > 0`.
- Off-mask RGB or depth of any value (NaN, 0, huge, random) leaves the objective and precisions bit-identical.
- `M = 0` gives target = b and zero depth precision; `M = 1` gives the original image and depth on `V_orig`.
- The Phase 2 debug renders compare equivalent composites: in the masked mode the reference image is `Y(b)` on the
  render background. The rendered RGB is never masked.
- The debug "Phase 1" layer uses the tracked Phase-1 orientations, which are now saved as
  `phase1_centers/rotations_frame_*.npy`, instead of canonical ones.

### P1-4 Fair A/B (`scripts/run_object_mask_ab.sh`, `tools/compare_object_masks.py`, benchmark)

- **The old defect.** The original arm reran only Phase 3 on copied Phase 1/2 predictions, while the masked arm reran
  everything.
- **Now:**
  - the same static models are copied into two new roots and verified sha256-identical (PLY, `cfg_args`,
    `cameras.json`);
  - both arms run fresh Phase 1/2/3, scene by scene, with explicit modes;
  - nothing seeds one arm from the other.
- **The compare tool checks before it scores:**
  - complete manifests in both arms;
  - signatures that differ only in the observation block, the model dir and the upstream hash;
  - identical static PLY hashes.

  A missing file is a listed problem and gives exit 1. A recorded absence of GT, such as the backward endpoint having no
  labelled-3DGS GT, is a note.
- **Hard-constraint columns removed.** They are replaced by the GT-free `estimator/mask_diagnostics.py`, one definition
  for both modes. Each stage's tracked state, centres and orientations, is re-rendered on a fixed evaluation background
  b = 0.5. It reports:
  - alpha outside the mask (mean, sum) and the leaking-pixel share;
  - foreground alpha coverage;
  - RGB MAE on foreground and background of the same composite;
  - depth MAE on valid in-mask depth;
  - valid-depth counts and the share discarded by the mask;
  - from `stage_resources.json`: wall time and peak GPU per stage.
- **No energy comparison.** Total energies are never compared across the two objectives.
- **Benchmark tool.** `tools/benchmark_masked_rgb_observation.py` now renders the tracked rotations; it used to pair
  tracked centres with `truth.R`. It also times CG solves, posterior information, PSD inversions and projections, with
  GPU synchronisation in this profiler only.

### P2-1 Memory and runtime

Measured in section 3.3. No scheduler change was made: there was no demonstrated bottleneck to optimise on the 8-camera
workload, and the 64-camera measurement is in section 3.3.

### P2-2 Docs, hygiene and regressions

- **Docs updated:** `README.md` 2.2/2.5/2.6/6, `docs/PIPELINE_CODE_MAP.md`, and an update note in
  `docs/MASKED_RGB_FOREGROUND_DEPTH_IMPLEMENTATION_2026_10_20.md`.
- **Driver:** watchdog and heartbeat semantics unchanged.
- **Static training:** untouched; its tests are in section 2.
- **`tests/test_joint_phase3_runner.py` [3]:** now passes `--observation-mode original`, because its direction check
  would otherwise stop first at the mask preflight.
- **Original mode:**
  - against the pre-patch precision builder `_precisions_legacy`, it is equal for p_rgb, p_d, r_rgb and p_d r_d on
    finite positive depth (test [E]);
  - **intended difference 1:** a zero depth flagged valid is now excluded;
  - **intended difference 2:** acceptance uses the NLL instead of the re-weighted surrogate.
- **`test_iekf_end_to_end` T12b:**
  - On main, inside the suite, a near-converged step was rejected. The repaired run accepts every step, so the
    "rejected step shrank the radius / applied alpha = 0" checks have nothing to inspect.
  - Standalone, neither tree rejects a step: the last steps change E by about 5e-6, at the CUDA floor. The test accepts
    both outcomes by design.

## 2. Tests

`tests/test_masked_default_repair.py` (new) runs 76 checks on the real CUDA renderer. Sections [A] to [M] are listed in
its docstring. Highlights:

- **[C]** dE/dr = IRLS score (rel 1.2e-7); a surrogate misranking example.
- **[H]** Phase 1:
  - the posterior tensor of frame k is the prior of frame k+1;
  - a zero-information frame returns F P F^T + Q (2.3e-7);
  - PSD with cross terms.
- **[I]** Phase 2 prior:
  - `e(x_minus boxplus d) = d` to 8e-17 for |d| = 0.96 with nonzero rotation and twist;
  - FD Jacobian remainder O(h^2) (ratio 4.00);
  - J is not I away from x_minus;
  - chart transport conserves E.
- **[J]** Phase 3, on a real-renderer prismatic toy:
  - total MAP monotone; total = sum of parts;
  - stationary for E_total;
  - the image optimum (lower image energy, higher total) is not returned;
  - geometry bitwise unchanged;
  - qdot posterior = prior.
- **[K]** Signatures: absent / own incomplete / reuse / refuse on mode, mask content and PLY; legacy refused.
- **[M]** The leakage metric changes with orientation alone.

Full kept suite: see section 4 (`T-suite`) and the JSON.

## 3. Smoke runs (scratch roots, fridge_10905, 8 cameras, 2 frames)

All of these are scratch roots, run on the 8 GB RTX 4070 next to other jobs.

- **Signatures, end to end** (driver, code before the final commit):
  - Pass 1 ran all six stages (P1/P2/P3 x F/B), and every manifest is `complete`.
  - Rerunning unchanged code reused all six stages.
  - An `--observation-mode original` run into the masked root was REFUSED, with the field diff (mode, mask root, mask
    content hash, background).
  - An edit to the code between runs was REFUSED (code diff hash).
- **Fair A/B, 2 frames** (code before the last two fixes):
  - The manifests of the two arms differ only in `observation.*`, `endpoint.model_dir` and `upstream_signature_sha256`.
  - The static PLY hashes are identical.
  - Backward Phase 1 has no labelled-3DGS GT at the final endpoint. This is recorded as a note, not a failure.
- **GT-free medians** (original / masked):

  | stage | alpha outside mask | wall time | peak GPU |
  |---|---|---|---|
  | P1 | 1.14e-3 / 1.15e-3 | 59 / 52 s | 1896 / 1978 MiB |
  | P3 | 1.10e-3 / 1.10e-3 | — | — |

  With 2 frames the arms are near-identical. This only shows that the machinery works, not any accuracy effect.
- **Bug found by the smoke run and fixed:** the Phase 1 covariance report needed a 15 GiB CUDA eigensolver workspace at
  N = 27k. It now runs on the CPU: 0.2 s and 53 MiB.
- **Final smoke on the committed code:** running at the time of this commit (2-frame A/B, reuse check, stage report,
  8-camera and 64-camera one-frame benchmarks). Its numbers are not in this document; until they are, the matching
  rows in section 4 are NOT RUN.

## 4. PASS / FAIL / NOT RUN

| check | status | evidence |
|---|---|---|
| Default mode = masked-rgb-expected-quad, driver + 3 runners | PASS | test [A], [B] dry runs |
| Missing / non-binary mask stops before any write / GPU | PASS | test [A], [B] |
| Y = M I + (1-M) b, rendered RGB never masked | PASS | test [D], test_masked_rgb_observation |
| Off-mask depth NaN / 0 / huge / random: identical objective | PASS | test [D] |
| Analytic expectation vs 8-point quadrature | PASS | test_masked_rgb_observation (46) |
| JVP / VJP / Fisher, phases 1/2/3 | PASS | test_masked_rgb_observation, visibility tests |
| Phase 1 MAP acceptance includes prior and alpha | PASS | evaluate_true_map_energy; test_iekf_end_to_end |
| Phase 2 exact SE(3) prior, FD Jacobian, chart transport | PASS | test [I] |
| Phase 3 MAP: monotone, stationary, image-only optimum rejected, geometry bitwise | PASS | test [J] |
| Phase 1 recursion, P_post -> next prior, PSD, F P F^T + Q | PASS | test [H] (after the inflation fix) |
| Student-t NLL gradient = IRLS score, surrogate misranking | PASS | test [C] |
| Convergence before state assignment, stop reasons | PASS | test [G] |
| Experiment signatures reuse / refuse | PASS | test [K], driver smoke |
| Fair A/B machinery (fresh P1/P2/P3 both arms, same PLY hash) | PASS (2-frame smoke) | section 3 |
| Leakage metric re-renders tracked orientations | PASS | test [M] |
| Kept test suite (63 files) | PASS | all pass; 8 files need `PYTHONPATH=.` (they do not add the repo to sys.path) |
| Static training / carving regressions | PASS | test_transparency_carving (63), test_object_mask_transparency_carving (45), test_carving_depth_object_only |
| Legacy prismatic-null diagnostic | PASS with scoped legacy step | test_efficient_p_null (20); see P0-2 |
| 8-camera full-frame benchmark (committed code) | NOT RUN | final smoke still running at commit time |
| 64-camera stress / memory | NOT RUN | final smoke still running at commit time |
| Full-scale A/B, all 10 scenes, full sequences, F and B | NOT RUN | user's run, section 5 |
| Support F1, joint geometry, displacement error, leakage at full scale | NOT RUN | user's run |

## 5. Commands for the user's runs (new output roots)

The code is on branch `masked-default-repair`. Either run from the worktree
`~/Joint3DGS/RigEuler3DGS-repair`, or fast-forward `main` first
(`cd ~/Joint3DGS/RigEuler3DGS && git merge --ff-only masked-default-repair`).

```bash
RES="/media/samuelemara/Extreme SSD/Samuele/results_11_10_26"
# fair A/B, all 10 scenes: static models from ours_v2 copied + verified, fresh P1/P2/P3 in both arms
nohup bash scripts/run_object_mask_ab.sh --static-root "$RES/ours_v2" \
  --original-root "$RES/ours_v3_original" --masked-root "$RES/ours_v3_masked" \
  > ~/ours_v3_ab.log 2>&1 &
# afterwards: machine-readable stage report
python tools/repair_report.py --root "$RES/ours_v3_original" --root "$RES/ours_v3_masked" \
  --out "$RES/ours_v3_masked_vs_original/stage_report.json"
```

Legacy roots (`ours`, `ours_v2`, `ours_v2_masked_rgb`) have no stage manifests. The driver refuses to reuse or overwrite
them, so always use new roots.
