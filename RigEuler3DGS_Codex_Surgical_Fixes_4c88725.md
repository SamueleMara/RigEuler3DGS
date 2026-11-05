# CODEX — RigEuler3DGS calibrated bootstrap: surgical corrections and evidence gates

**Target repository:** `SamueleMara/RigEuler3DGS`  
**Reviewed baseline:** `4c88725` (compare `481b2cf..4c88725`)  
**Purpose:** Fix confirmed code defects in the newly integrated calibrated RGB-D / endpoint-bootstrap pipeline, then diagnose—without pre-emptive redesign—whether the new initialization and covariance improve Phase-1 motion recovery.  
**Scope:** Only the issues below. Do not reinterpret this as permission to replace the IEKF, change the Phase-2/Phase-3 mathematical model, introduce new motion heuristics, change the motion-support selection policy, or implement post-Phase-3 point discovery.

## 0. Working instructions and invariants

1. Inspect the **actual checkout and current HEAD** before editing. Compare it with `4c88725`. If a named defect has already been fixed, show the concrete existing fix and its test; do not implement it twice. Preserve user changes and do **not** reset/force-push.
2. Work in small, reviewable changes; add or update targeted tests for each defect. Avoid broad refactors and silent changes to default behavior beyond the explicit correctness fixes.
3. Continue to use **the production renderer and its analytic IEKF Jacobians**. No surrogate renderer in the final evaluation, no replacement with Adam-based tracking, and no GT in inference.
4. Preserve: hard object-mask constraints in measurement handling; calibrated camera identity; canonical initial/final endpoints; fixed Gaussian count/ordering in the bootstrap; independent forward/backward endpoint models; recursive Phase-1 covariance; positive physical elapsed time in both processing directions; sensor noise separate from model mismatch.
5. Preserve the currently implemented full Phase 1 -> Phase 2 -> Phase 3 pipeline, and all existing regression behavior except where this document explicitly asks for a fix.
6. **No arbitrary gates, new heuristic thresholds, or automatically tuned motion priors.** If a numerical threshold is necessary for an oracle test, identify it as a test tolerance and justify it. Never present a hyperparameter change as a correction to a proven code defect.
7. Do not start the full 14-episode / ten-scene batch while making these fixes. Run focused unit tests and a small real/synthetic pilot; report full-scale tests as `NOT RUN` until actually executed.
8. Keep generated datasets, experiments, checkpoint directories and prior baselines intact. No destructive cleanup or overwriting mismatched cached results.

## 1. P0 — Correct the bootstrap silhouette loss (confirmed defect)

**Primary file:** `tools/train_tracking_bootstrap.py`, around the construction of `held`, `train`, and `losses(dd, keyset)` (reviewed lines ~223–256).  
**Possible tests:** `tests/test_calibrated_bootstrap.py` (extend with a dedicated alpha/mask test).

### Observed implementation

The current code creates the hold-out and training selections only inside the foreground object mask:

```python
held = (rng.random(msk.shape) < a.heldout_fraction) & msk
train = msk & ~held
# ...
W = dd[keyset]
ls = 0.5 * (W * (A - dd["M"])**2).sum() / (
    a.silhouette_sigma**2 * W.sum().clamp_min(1)
)
```

Since `W=0` whenever `M=0`, this only penalizes insufficient opacity *inside* the object. Gaussian opacity spilled into background pixels has **zero silhouette loss**.

### Required implementation

1. Construct a deterministic, disjoint train/held-out pixel partition **over the full camera image**, not just the foreground. Use the same seeded partition for every refinement step (do not resample per iteration).
2. Keep different effective supports:
   - `W_alpha_train`, `W_alpha_held`: **all** pixels in the respective partition, including `M=0`.
   - `W_rgb_train`, `W_rgb_held`: foreground mask `M` intersected with the respective partition (preserve the current RGB measurement policy).
   - `W_depth_train`, `W_depth_held`: valid observed depth AND foreground mask AND the partition; preserve the existing `A > 0.5` condition *unless* separately justified and tested.
3. The silhouette term must implement the full-image objective on the selected partition:

   \[
   E_{\alpha,S}=\frac{1}{2\sigma_\alpha^2}\;
      \frac{\sum_{p\in S}(A(p)-M(p))^2}{|S|},
   \quad S\in\{\text{train},\text{held}\}.
   \]

   This is a **soft optimization loss on a hard binary ground-truth object mask**; do not relabel the mask/background as valid object depth measurements.
4. Preserve the foreground-only RGB fit and observed-valid object depth fit (including robust inverse-depth NLL). Do **not** add a depth residual for background rays, invalid observations or occluded object surfaces.
5. Ensure training and held-out scores use the **same term definitions and normalization**, with no pixel leaked between sets. Record foreground/background counts and separate foreground-miss versus off-mask-opacity statistics, not only the combined silhouette loss.
6. All loss terms must remain finite when a selected partition contains zero pixels for a particular channel. Handle these cases explicitly rather than hiding a misconfigured partition.

### Acceptance tests

- **Negative-ray oracle:** a synthetic/rendered Gaussian footprint spills into pixels with `M=0`. Its alpha loss and gradient are strictly nonzero in the direction needed to reduce off-mask opacity; this fails on the old code.
- **Positive-ray oracle:** missing alpha in `M=1` also has nonzero correction, unchanged in sign from the old objective.
- **Split oracle:** no train/held overlap; both domains contain background pixels in a nondegenerate test; altering only held-out measurements does not change a training step.
- **Mask/validity oracle:** RGB and depth never include invalid/background pixels; alpha does include both foreground/background pixels.
- **Fixed-population oracle:** Gaussian count and index/order stay unchanged at all checkpoints.
- If a renderer-specific test requires CUDA, provide a small CPU logic-level test as well and mark CUDA checks explicitly.

## 2. P0 — Make calibration and bootstrap cache reuse content-safe (confirmed defect)

**Primary file:** `tools/run_noise_aware_rigeuler.py`, particularly `bootstrap_reusable()` and calibration reuse logic (~178–296).  
**Reference implementation:** `tools/stage_signature.py` (content hashing, explicit re-use/refusal contracts).  
**Other affected output:** `calibration/sensor_model.json`, `bootstrap/{initial,final}/bootstrap_manifest.json`, `run_manifest.json`.

### Observed implementation

- An existing `calibration/sensor_model.json` is reused just because it exists.
- The bootstrap reuse test compares only `endpoint_frame`, `refine_iters`, and `sensor_policy`, ignoring e.g. the number of Gaussians, full sensor-model ID, model-mismatch sigmas, source inputs, and bootstrapping parameters.

### Required implementation

1. Compute deterministic **stage-input signatures before deciding reuse**. Reuse the existing hashing utilities/pattern when appropriate, rather than inventing a second incompatible reuse policy.
2. **Calibration signature:** include dataset/source identity, ordered camera IDs, exact inclusive calibration interval, content hashes for its RGB/depth/mask files, relevant camera metadata if consumed, all estimator options (`bins`, `min-bin-samples`, etc.), and a code/implementation fingerprint. Include the actual content hashes, not filenames/mtimes only.
3. **Per-endpoint bootstrap signature:** include canonical endpoint and frame, camera geometry/intrinsics/extrinsics content, the exact RGB/depth/mask bytes for that frame, selected sensor-policy and `sensor_model_id` (or explicit uncalibrated fallback policy), Gaussian target count, kNN/seed/refinement settings, loss weights, all model-mismatch sigmas, P0 prior controls, numerical settings, and relevant implementation/code fingerprints.
4. Store signature and input provenance in each stage manifest, with a schema version. A cached result is reusable **only if** the signature matches **and** mandatory outputs exist and pass minimal structural checks:
   - calibration: sensor file hash/ID valid; required validation files present;
   - bootstrap: requested checkpoint and `point_cloud.ply`, `geometry_information.npz`, and `bootstrap_manifest.json` present; expected dimensions/count and sensor ID match.
5. Existing output **without a compatible signature** or with a different signature: fail safely with a precise mismatch message and instruction to choose a new output path. Do not silently overwrite/delete it.
6. A same-signature but interrupted/incomplete stage may be handled using the repository's existing explicit resume/retry policy; make the behavior deterministic and report it.
7. Keep the stage manifests and the driver signatures consistent. A change to calibration or endpoint parameters must invalidate downstream Phase-1/2/3 reuse through the changed upstream model/sensor fingerprint.
8. Verify `--dry-run` does not mutate artifacts, stage directories, or manifests.

### Acceptance tests

- Exact same inputs + settings + complete outputs -> reuses without recomputation.
- Change **only** `--rgb-model-sigma` -> bootstrap reuse refused.
- Change **only** `--depth-model-sigma`, `--num-gaussians`, `--refine-iters`, prior sigma, or relevant source bytes -> reuse refused.
- Change calibration interval (or one same-sized RGB/depth/mask file) -> calibration reuse refused.
- Corrupt `sensor_model_id` / remove `geometry_information.npz` / swap checkpoint -> reuse refused.
- Cached legacy manifest without signature -> explicit refusal, not false reuse.
- No destructive action when refusing; dry-run is read-only.

## 3. P1 — Correct remaining physical-time export inconsistencies

**Files:** `tools/run_rigid_se3_pipeline.py`, `utils/bidirectional_time.py`, `tools/run_motion_coherence_support.py` and any affected artifact writer / tests.  
**Do not modify:** the already timestamp-aware main Phase-2/Phase-3 fitting path unless a test proves a specific defect.

### 3A. Backward chronological twists

`tools/run_rigid_se3_pipeline.py` currently invokes:

```python
chrono_xi = chronological_twists(
    torch.from_numpy(_chrono(poses)).double(), float(ns.dt)
)
```

Yet for the real fridge episodes `ns.dt` remains `1/30 s` while the measured data interval is `0.04 s` (`25 Hz`). This can overstate exported twist magnitudes by approximately 20% in uniform segments.

1. Extend `utils/bidirectional_time.chronological_twists(poses, dt)` to support an **array of positive elapsed durations** `[T-1]`, while preserving existing scalar behavior and conventions. If there are existing scalar-dependent consumers, keep backward compatibility.
2. Derive chronological durations **from the real timestamp metadata** already validated by the loader and `processing_time_s`, not from `ns.dt` or the frame indices. Reordering backward processing times requires care: recover increasing chronological times before taking differences.
3. Compute `xi[t] = Log(T[t+1] @ inverse(T[t])) / (time[t+1]-time[t])`, preserving the left/spatial twist convention and the existing final-row padding convention.
4. Confirm the saved `chronological_twists.npy` is in original chronological order for **both** forward and backward comparisons. Do not reverse or simply negate the filter's reverse-time velocity state.
5. Extend the per-stage time-base artifact to record the **complete actual interval sequence** when sampling is nonuniform; do not describe a single `dt_s` as representative of all frames if it is not.

### 3B. Phase-1 full-state recorder metadata

`tools/run_motion_coherence_support.py` still calls its recorder with `dt_seconds=float(ns.dt), dt_is_measured=False`. For physical real-data timestamps this metadata is misleading.

- Use measured time intervals (or capture timestamps), reporting the correct measured provenance.
- If the recorder supports only scalar intervals, add a minimally invasive compatible representation for nonuniform timestamps or fail explicitly rather than inserting `1/30` and labelling it measured.
- Do not change the IEKF state transition or estimation just to repair metadata.

### Acceptance tests

- 25 Hz constant-twist synthetic trajectory: physical 0.04-s intervals and expected twist rates within numerical tolerance; forward and reconstructed backward agree in sign and magnitude.
- Nonuniform time intervals: correct interval-by-interval rates, not a constant-rate approximation.
- Existing scalar-`dt` tests still pass.
- Real-data metadata reports the exact timestamp-derived intervals and correct units.

## 4. P1 — Add a read-only Phase-1 covariance and camera-conflict audit (diagnosis, NOT a guessed fix)

**Primary files:** `tools/kpart_pipeline.py::bootstrap_positions`, `estimator/sparse_iekf.py`, existing diagnostic output classes or a narrowly scoped new `tools/audit_calibrated_phase1.py`.  
**Read-only:** It must not modify optimization, priors, gating, support membership, or accepted steps.

### Why

The new `P0_geometry` is computed as independent per-Gaussian blocks of `(F_ii + Lambda_prior)^-1` while the renderer has inter-Gaussian coupling and shared image measurements. This is a **documented approximation, not the true marginal**. Moreover Phase 1 still initializes velocity priors with `sigma_v = 1 m/s` and `sigma_omega = 0.316227766 rad/s`, and the calibrated camera RGB standard deviation is approximately `0.005` on the reported real scene versus legacy `0.1`.

For the first 0.04-s prediction with zero pose-velocity cross-covariance, the velocity prior alone contributes approximately `0.04 m` position SD per axis and `0.01265 rad` rotation SD per rotational component. These are potential mechanisms for inaccurate motion—not proof of a defect in any particular estimated trajectory.

### Required diagnostics

For every camera, frame and a representative set of Gaussian/supernode populations, record:

1. **Initial/prior/posterior covariance:** pose and velocity block eigenvalues or directional SDs of `P0`, `P_pred`, and `P_post`; pose-velocity cross blocks; PSD minima; separate propagation contributions `F P F^T` and `Q` (including velocity-origin pose uncertainty).
2. **Image information:** per-camera RGB/depth/alpha valid counts, residual distribution and energy; effective sensor SD, model-mismatch SD and resulting SD, without combining their labels; Fisher eigenvalues/trace; weak-direction counts. Use existing production measurement terms.
3. **Image-vs-prior balance:** per-camera `J_c^T W_c r_c`, its norm, and agreement/disagreement in displacement direction among cameras; optionally per-camera local Fisher. The per-camera gradient is **diagnostic only**, not a new exclusion or ownership rule.
4. **State behavior:** accepted/rejected IEKF iterations, convergence/stop reasons, change of canonical Gaussian centers, per-frame displacement quantiles (not just mean/max), mask IoU, observed-valid object depth residuals, and off-mask opacity.
5. **Static-control test:** evaluate 3+ consecutive frames in a verified static interval, using the same calibrated bootstrap (or a clearly described compatible canonical control). Report observed drift in millimeters and its distribution, rather than asserting the object was static based on the detection heuristic alone.
6. **Motion-control test:** use the first short meaningful motion interval and a synthetic articulated sequence with known point/joint GT. Report dynamic displacement recovery separately from static-point drift. Do not use GT inside the filter.
7. **Uncertainty comparison:** empirical standardized residual histograms, per-camera quantiles, and coverage. Do not claim a chi-square NIS calibration from robust Student-t objectives or correlated pixels without stating and checking its assumptions. If NIS is implemented, identify the exact valid degrees of freedom and covariance approximation.

### Important restrictions

- **Do not tune** `--p0-velocity-sigma`, model-mismatch sigma, process noise or trust region as an automatic fix. Run declared sensitivity/ablation experiments with every setting logged. Select no constants based on GT.
- **Do not** replace block covariance by a dense full-Gaussian covariance in this patch. If block covariance is demonstrably overconfident, quantify it and propose a separate mathematically justified change.
- **Do not** implement an AR(1) or Woodbury common-camera update merely because temporal correlation appears in calibration. The measured near-flat lag correlations do not themselves justify AR(1).
- **Do not** impose permanent camera owners. Use responsibility as an association diagnostic, preserving the current data flow.

### Acceptance tests

- Synthetic linear-Gaussian 12D predict/update oracle verifies covariance decomposition and diagnostic outputs to numerical tolerance.
- Known `P0` + known `dt` + velocity prior reproduces expected predicted pose-variance contribution.
- Running audit enabled/disabled yields **identical tracking states and covariances** (within bitwise equality where feasible); no accidental backprop or optimizer perturbation from reporting.
- Camera-identity swaps/missing calibrated camera name are detected; no pooling/fallback.
- Pilot output has finite statistics with clear treatment of missing depth and no false assumption that all cameras observed every Gaussian.

## 5. P2 — Bootstrap geometric and uncertainty verification (measure before changing fitting)

**Files:** `estimator/surface_bootstrap.py`, `tools/train_tracking_bootstrap.py`; add a small focused test module as needed.

The current voxelized, multiview kNN + Student-t tangent-plane fit can mix nearby but distinct layers and smooth edges before its robust weighting resolves the correct surface. The previously reported `1.33 mm` median distance to the dataset's own fused cloud is **not independent geometric ground truth**.

Required evaluations:

1. **Synthetic surfaces** with known planes, sharp edges, a narrow panel with distinct front/back faces, an occluder, and camera viewpoints with different ray angles. Record center-to-true-surface errors (median/p90/max), normal angular errors, and whether the front/back layers are merged.
2. **Held-out rendered view tests:** image RGB, valid object depth, foreground silhouette and background false-alpha evaluated on genuinely held-out pixels; ensure the training/held-out masks are disjoint after P0 fix.
3. **Uncertainty test:** compare predicted P0 normal/tangential/rotational confidence with actual estimation errors on synthetic tests, and report weakly observable eigenvectors. Label P0 as **block information approximation**, not exact marginal posterior.
4. Preserve `Sigma_render` versus `P0_geometry` separation and fixed Gaussian count. Do not replace the kNN surface reconstruction until a reproducible failure has been measured and reported.
5. Inspect calibration for depth-discontinuity and per-camera bias effects; persistent bias cannot be inferred from static temporal variance alone. Report this limitation without inventing an offset correction.

## 6. P2 — Controlled A/B evidence, only after the fixes pass

Use the **same original RGB-D files, camera set, frame timestamps, masks, canonical frame(s), evaluation protocol and random seed**. Do not compare a 4-frame smoke run with a long sequence and attribute the difference to calibration. Where the existing orchestration lacks a baseline endpoint mode, create a narrow, explicit compatibility pathway or execute the equivalent commands separately; **do not silently substitute a noncanonical model**.

| Arm | Endpoint model | Sensor noise | Phase-1 initial pose covariance | Purpose |
|---|---|---|---|---|
| A | Existing compatible 3DGS baseline | Legacy assumed | Legacy | Prior baseline |
| B | New endpoint bootstrap | Legacy assumed | Legacy | Isolate geometry |
| C | Existing compatible 3DGS baseline | Calibrated | Legacy | Isolate sensor weighting |
| D | New endpoint bootstrap | Calibrated | Bootstrap P0 | Combined method |
| E | New endpoint bootstrap | Calibrated + explicitly declared mismatch | Bootstrap P0 | Mismatch sensitivity |

Additional declared sensitivity runs may vary the velocity prior, but they must be labelled as **ablations**, not changes to the nominal defaults.

**Minimum evaluations:**

- One real red-fridge episode with an independently verified static subwindow and an initial dynamic subwindow; record forward and backward results separately.
- One synthetic articulated SAPIEN scene with GT for support, motion and joint geometry; report per-part motion error, static drift, support precision/recall, joint type, joint-axis angular error and axis location error (when the joint is identifiable).
- Hold-out pixels/cameras where feasible; report the actual split and avoid leakage into bootstrap fitting.
- Compare **tracking and geometry errors**, not only loss decrease, stage completion or rendering quality.
- Do **not** claim the new calibration improves tracking without numerical A/B evidence.

## 7. Execution order and evidence gates

**Gate 0 — Baseline:** record HEAD, dirty status, `4c88725` comparison, test environment and existing outputs. Do not overwrite running experiments.  
**Gate 1 — Correctness:** implement §§1–3, run focused unit tests plus existing regression tests affected by these changes. Must pass before other changes.  
**Gate 2 — Read-only audit:** implement §4 and validate its invariance. Capture the short static and moving pilots; report failures, not speculative solutions.  
**Gate 3 — Validation:** perform §5 and the controlled small A/B from §6 if input data and GPU are accessible. Explicitly distinguish `PASS`, `FAIL`, and `NOT RUN`.  
**Gate 4 — Report:** provide one `docs/CALIBRATED_BOOTSTRAP_SURGICAL_REVIEW.md` with:

- Files, functions and exact changed line ranges; focused explanation of each fix.
- Mathematical definition of the corrected alpha loss, covariance diagnostics, and timestamp conversion.
- Reproduction commands, dataset names, selected frame intervals, seed, Gaussian count, sensor-model ID, code SHA, and input signatures.
- `PASS / FAIL / NOT RUN` table for **each test**; do not use words like *validated*, *fixed*, or *improved* for checks that were not run.
- Compact comparison table for static drift, dynamic motion, support, and joint errors, clearly marking unavailable GT and untested outcomes.
- Remaining assumptions/limitations: persistent sensor bias, RGB full-covariance ignored, temporal/camera correlations, inter-Gaussian Fisher coupling, weak pose observability, and unvalidated full-scale accuracy.
- Exact Git commit(s) and working-tree status. Commit and push only after tests; do not conflate successful `git push` with model validation.

### Targeted commands (adapt environment only; do not fabricate successes)

```bash
# Inspect current state
git rev-parse HEAD
git status --short
git diff 4c88725 --stat

# Existing focused test file; add new tests next to the appropriate suites
python tests/test_calibrated_bootstrap.py

# Other existing tests relevant to regression (inspect their invocation first)
python -m pytest -q tests/test_bidirectional_sequence_direction.py \
    tests/test_surgical_fixes_2026_10_22.py

# Inspect only; do not launch unattended full-scale experiments
python tools/run_noise_aware_rigeuler.py --help
```

If a named test path, test runner or data scene differs in the actual checkout, correct the command and report the adjustment. Run real GPU E2E tests only with accessible files and record true outcomes.

## 8. Definition of done

The requested task is done only when:

- [ ] Off-mask bootstrap opacity receives the correct nonzero silhouette penalty and held-out pixel independence is verified.
- [ ] Calibration/bootstrap reuse rejects every tested stale or mismatched input/configuration without destroying artifacts.
- [ ] Backward twist export uses measured chronological frame intervals and phase-1 timing metadata is honest.
- [ ] Covariance and camera-conflict reporting is available and demonstrably does not change the filter's results.
- [ ] Small synthetic geometry/uncertainty tests and a real static/motion audit are reported, or marked `NOT RUN` with a specific blocker.
- [ ] The A/B arms are honestly reported; *no performance improvement is claimed from smoke-test success alone*.
- [ ] Existing Phase-1/2/3 behavior and baseline tests are preserved except for the explicitly corrected defects.
- [ ] One surgical report with exact changes, tests, limitations and commits has been produced.

**Stop at evidence.** If an audit shows a deeper problem (for example, correlated Fisher overconfidence, depth bias, camera association disagreement, or new point discovery), explain it with measured evidence and a narrowly scoped follow-up proposal. Do **not** silently implement an unrequested theoretical redesign in this patch.
