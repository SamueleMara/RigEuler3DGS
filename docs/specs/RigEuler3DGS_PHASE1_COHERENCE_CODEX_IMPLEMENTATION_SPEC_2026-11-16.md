# CODEX IMPLEMENTATION DIRECTIVE — RigEuler3DGS Phase-1 visibility and coherence reliability

**Repository:** `SamueleMara/RigEuler3DGS`  
**Inspected commit:** `phase45` at `3bacfe51771efccf37228ed2f27a8b30bb509710` (2026-11-16).  
**Target:** the user's existing **local** codebase/worktree, after comparing its HEAD and modifications with this inspected commit.  
**Scope:** (I) correct Phase-1 observation/optimization; (II) correct motion-coherence candidate selection. **These are two separate development and validation phases.**  
**Out of scope:** joint type/geometry refinement, Phase 4A/B/C geometry recovery, Phase 5 Gaussian training, retraining the canonical 3DGS, unrestricted segmentation, GT-based selection, and altering the base CUDA renderer's legacy behavior.

> **Non-negotiable:** Inspect the actual checkout before editing. Run `git rev-parse HEAD`, `git branch --show-current`, `git status --short`, inspect the relevant source and compare to `3bacfe517`. The specifications below are grounded in that pushed commit; file paths and call sites must be reconciled against the local worktree. Never overwrite uncommitted changes. Report any difference before proceeding. Implement in opt-in modes so the original v6 behavior remains reproducible.

## 0. Verified code inventory and exact original behavior

All paths are relative to repository root. Verified against the commit above:

| File / function | What it actually does | Required action |
|---|---|---|
| `estimator/masked_observation.py`: `PreparedRGBD`, `prepare`, `phase1_camera_terms`, `model_from_args` | Stores `P=M*I`, constructs `Y=P+(1-M)*b`, **restricts depth validity to `V_raw & M`**, and in expected-quad mode defines `kappa=trace(W)/12` | Preserve legacy; new likelihood must use raw `depth_observed` and raw depth validity, not only masked `depth_valid` |
| `estimator/observation_terms.py`: `camera_terms` | Constructs RGB quadratic precision/energy, robust Student-t depth NLL with `alpha>min_alpha_for_depth`, alpha residual `M-A` and energy | Add a separate physically consistent observation path, not a partial replacement of energy only |
| `estimator/sparse_iekf.py`: `_frame_terms`, `measurement_operators`, `evaluate_true_map_energy`, `_meas_ops`, `iekf_frame_step` | At each nonlinear render caches terms; trial acceptance uses `measurement + prior`, RHS is `H^TWr + H_A^T kappa r_A`; all camera terms feed normal/Fisher; `observation_model` and `visibility` modes are mutually exclusive | Wire all four: true energy, first derivative, GN operator, Fisher/posterior |
| `estimator/renderer_bridge.py`: `FrameLinearization` | `v.rgb`, normalized blended depth `v.depth=Z/(A+eps)`, full alpha `v.alpha`, `v.cache`, ordered contributors, JVP/VJP/Fisher; `apply_H_alpha`, `apply_HT_alpha`, `build_alpha_fisher` operate on **full-ray alpha** | Reuse, but **do not** use normalized blended depth to determine first surface |
| `estimator/visibility_constraint.py`: `FrameVisibility`, `iekf_visible_alpha` and related CUDA operators | Existing `known-nonobject-depth` computes **front-of-measured-depth** alpha, JVP, VJP and nonlinear audit, but as a hard constraint in a distinct observation mode | Reuse data/ordering/derivatives, NOT its hard QP simultaneously with masked RGB; introduce unified likelihood |
| `estimator/first_return_observation.py` and `estimator/first_return_operators.py` | Optional `first_return=depth` replaces blended depth only, keeps legacy RGB+alpha; `rgbd` joint association is **not implemented** | Reuse contributor primitives only where valid; do not claim first-return-depth alone fixes EE contamination |
| `estimator/sparse_iekf.py`: grouped solve around lines 746–813 | 12D `b`; `A_mv`; grouped map `G`; solve `G^T A G du = G^T b` using `grouped.expand/reduce`; the weak-direction projector is skipped when grouped | Implement a consistent reduced **observable update** map, never post-hoc pose reset |
| `estimator/sparse_iekf.py`: posterior near 1091–1155 | `P_post ≈ (Lambda_prior + F_ii)^-1`, block approximation, potentially overconfident with shared contributions/groups | Require covariance oracle and honest approximate-posterior status; zero-information member must not get false covariance contraction |
| `estimator/process.py`: `process_predict`, `error_state_transition` | Pose+velocity 12D process; OU if `velocity_tau>0`, constant twist otherwise | **Keep unchanged** initially; flag predicted-only displacement rather than calling it observed motion |
| `estimator/coherence_extraction.py`: `cross_validated_rigidity`, `select_moving_component`, `refine_from_core`, `extract_moving_component` | 5-fold **node** CV Kabsch, relative `eta>0`, then maximize world `delta_rss_cv_per_node_frame`, unrestricted recruit if `E_m<E_i` | Replace final decision with independent original-RGBD hypothesis validation + calibrated null; conservative recruitment; preserve original as control |
| `tools/run_motion_coherence_support.py` | Runs Phase 1, saves `phase1_centers`, then graph/supernode selection and `predicted_support.json` | Expose new flags and diagnostic provenance, preserve direction/endpoint and GT isolation |
| `tools/run_rigid_se3_pipeline.py` and `estimator/free_space_support.py` | Phase-2 free-space `replace` can completely exchange support; accepts old support if too few moving labels | Add a fail-closed core preflight when using new coherence mode; **do not expand Phase-2 recovery now** |
| `run_bidirectional_all.sh` | Wires Phase1/coherence/Phase2; `OBS_MODE=masked-rgb-expected-quad`, `P1DET=0`, `COH_ETA=0`, `COH_RECRUIT=1`, `P2RMODE=replace` defaults | Opt-in environment wiring; new flags in cache/run signatures; preserve old defaults |

### 0.1 Exact baseline equations (NOT optional approximation)

For the reported active expected-quad mode, with camera/pixel/channel indices suppressed,

\[
J_\text{legacy}(x) = \tfrac12\delta(x,x^-)^T\Lambda^-\delta(x,x^-)
+ \tfrac12\sum\|M I+(1-M)\bar b-C(x;\bar b)\|_W^2
+\tfrac12\sum\tfrac{\operatorname{tr}W}{12}(M-A(x))^2
+\sum_{V_{\rm raw}\wedge M\wedge(A>0.05)}\rho_\nu\big((d-D(x))/\sigma_D\big).
\]

`D(x)` is the renderer's normalized blended depth, **not a first return**. `rho_nu` is Student-t NLL, not IRLS-weighted quadratic energy. The normal/Fisher use its IRLS local model. The alpha term is derived from the uniform background expectation, not a free silhouette coefficient. See `estimator/masked_observation.py:1–45,110–118,135–174,283–322`, `estimator/observation_terms.py:91–155`, `estimator/measurement.py:280–335`, `estimator/sparse_iekf.py:395–469,595–630`.

At `M=0`: RGB targets `bar b`, alpha is pushed toward zero, while depth is discarded, **even if raw sensor depth belongs to an external occluder in front of the object**. This is the specific bad observation model to correct, not evidence that all Phase-1 drift has a single cause.

## I. PHASE 1 — Correct the observation likelihood and observable grouped updates

**Deliverable I:** new opt-in Phase-1 likelihood, associated renderer derivative operators, observation-support metadata, and consistent reduced/grouped solver with its own tests. **Do not change coherence selection in this phase.** Run an identical old/new Phase-1 comparison before starting Part II.

### I.A. Use one coherent ray likelihood — no legacy off-mask penalty left active

Define for each camera-frame pixel raw color `I`, raw depth `d`, raw depth validity `V_raw`, supplied binary mask `M`, and the complete ordered Gaussian contributor sequence from the **actual CUDA render**.

**Negative mask ray `M=0` with reliable raw non-object depth:** the physical observation says no *modeled object* is in front of the observed surface. It does not say there is no modeled object behind the surface. With front transmittance

\[
T_p(z;x)=\prod_{i\in\mathrm{contributors}(p),\ z_i(x)<z}(1-\alpha_{ip}(x)),
\]

define the **proposed** negative-ray likelihood

\[
L^-_p(x)=-\log \mathbb E_{z\sim q_p(z\mid d_p)}[T_p(z;x)].
\]

`q_p` is the **calibrated or explicitly documented** distribution of true occluder depth given the return, including the sensor's uncertainty. The zero-uncertainty limit is `-log T_p(d_p;x)`. Use a deterministic, documented quadrature/integration scheme or a mathematically equivalent exact evaluation if feasible. On each sample, respect the rasterizer's actual clipping, alpha clamps, contributor order, and early termination. Mask-edge mixed returns may be marked `UNKNOWN` if `q` is not calibrated there; do not fabricate a 20 mm or 2-pixel physical tolerance.

**Implementation detail:** the existing `iekf_visible_alpha`/`iekf_vis_apply_H`/`iekf_vis_apply_HT` calculate `A_front(z)=1-T_front(z)` for known-nonobject-depth and are an efficient building block. For the fixed-depth case, the new score follows the chain rule

\[
\frac{\partial(-\log(1-A_{\rm front}))}{\partial x}
=\frac{1}{1-A_{\rm front}}\frac{\partial A_{\rm front}}{\partial x}.
\]

The new full nonlinear objective evaluates `-log T_front` on **every trust-region trial**; the linearized GN/Fisher approximation must be derived from this *same* likelihood. For the integrated uncertain-depth case, differentiation must pass through the normalized depth marginal, not through an arbitrary point estimate. Implement CPU/dense or CUDA reference for FD validation. Guard `log` using a numerically justified positivity floor, log its use, and test saturation branches. The existing hard `visibility_constraint` mode is **not** the new objective; do not merely enable both models.

**Negative mask ray with invalid/unknown raw depth:** do **not** infer that the entire ray is empty. No off-mask RGB-to-grey penalty, no full-ray alpha penalty, no invented far depth. Only an independently documented trusted-empty-ray measurement may justify full-ray exclusion. Otherwise this pixel contributes no object-motion evidence.

**Negative-ray mask reliability:** if a known/possible false-negative mask could label a real visible object point `M=0`, the negative interpretation itself is uncertain. Add explicit `KNOWN_NONOBJECT` / `UNKNOWN` measurement status or a calibrated contamination mixture. Do not treat every `M=0` as certified external material. Provenance must be saved per frame/camera.

**Positive mask ray `M=1`:** mask membership alone is not proof the RGB-D return belongs to an object Gaussian; a robot EE may be included. Construct a *joint* positive RGB-D ownership likelihood (new feature — not present in the committed code), for example

\[
p(y_p\mid x)=\pi_p p_{\rm obj}(I_p,d_p\mid x)+(1-\pi_p)p_{\rm unknown}(I_p,d_p),\qquad
L^+_p=-\log p(y_p\mid x).
\]

- `p_obj` must use a **single physically coherent** RGB-D first-return/visibility association (not independently paired blended depth and full alpha when the observed return is in front of the model); use the real contributor list. Reuse `first_return_operators.py` only after auditing its probabilistic assumptions and missing-association issue.
- `p_unknown` must be a specified **normalized** outlier/other-surface distribution with declared RGB/depth support, not an invented constant residual threshold. Calibrate it on independent data or document an explicit bounded prior. If that calibration is unavailable, fail closed / mark ambiguous; do not silently treat contaminated positives as object.
- `pi_p` must be based on independent ownership evidence or a declared mask reliability prior. Do not compute `pi_p` from a fitted candidate and then present that as independent validation. The posterior responsibility is `gamma_p = pi_p p_obj / (pi_p p_obj + (1-pi_p) p_unknown)`.
- No new Gaussian is born in Phase 1. A true missing object surface and an unmodeled EE may both be `UNKNOWN` with current model. That ambiguity is expected and passed to Phase 4, not resolved by guessing.
- The current `first_return=depth` **does not** solve this: `estimator/first_return_observation.py` explicitly says it leaves legacy RGB and alpha unchanged and that mode `rgbd` is not implemented.

**Build a single, exhaustive observation-status table for every view:** `OBJECT_SUPPORTED`, `KNOWN_NONOBJECT_DEPTH`, `TRUSTED_EMPTY` (only if independently established), `UNKNOWN`, plus invalid/mixed-depth provenance. Do **not** let a per-pixel test use rendered *blended* depth to determine `OBJECT_SUPPORTED` versus occlusion. The object visibility is determined by ordered per-contributor depths and measured first return.

**Critical model consistency:** if the positive mixture is implemented with EM/IRLS, responsibility may be frozen only for the **local surrogate** at a linearization. The **nonlinear trial energy** must always recompute the real mixture NLL at the trial state; the local gradient must match the derivative of that NLL at the current point. Use a derived PSD approximate Fisher, not the asserted Hessian of a different objective. Report approximation status in diagnostics.

### I.B. Concrete file and API plan

1. Add `estimator/occlusion_likelihood.py` with an explicit `RayEvidence`/`LikelihoodResult` interface carrying: per-view energy and components, per-Gaussian and pixel coverage diagnostics, validity/ownership status, `score_pose()`, `normal_pose(v)`, and `fisher_pose_blocks()`; choose class names to match the current code after inspecting it. Keep legacy `CameraTerms` untouched when opt-in is off.
2. Expose front transmittance and its exact matrix-free `Jv`, `J^T w` and per-Gaussian PSD curvature from the *existing* CUDA contributor cache; extend `estimator/renderer_bridge.py` and `visibility_constraint.py` utilities or add one versioned JIT extension adjacent to `estimator/first_return_operators.py`. Do not change legacy renderer bindings or enable antialiasing. An arbitrary Gaussian only affects its own relevant prefix (up to measured depth); no post-occluder derivative in the sharp depth-order limit.
3. Extend masked observation preparation so the opt-in mode retains **raw depth validity** outside M. Currently `PreparedRGBD.depth_valid = V_raw & M` (`masked_observation.py:169–174`); add a separate raw validity field and preserve the legacy field. Never use the masked field as proof that off-mask depth is unavailable. Ensure all `to()`, frame loaders, and camera ordering remain consistent in forward and backward sequences.
4. Integrate new `LikelihoodResult` dispatch into `sparse_iekf.evaluate_true_map_energy()` and `_meas_ops()`, and the accepted-state posterior, with one coherent shared linearization cache. Legacy `_frame_terms()` and `measurement_operators()` remain unchanged in legacy mode. **No double-counting** legacy RGB/alpha/depth on pixels served by the new likelihood.
5. Implement opt-in CLI `--phase1-likelihood legacy-masked|occlusion-consistent` (default `legacy-masked`) in `tools/run_motion_coherence_support.py` / shared runner, and wire `run_bidirectional_all.sh` with a separate cache signature. Names can change if consistent with repo conventions, but document them. Reject conflicting `--object-mask-constraint` and `--first-return-mode depth` unless an expressly derived joint model replaces both, not by accidental stacking.
6. Write a manifest including source commit, kernel hash, camera calibration hashes, sensor and ownership prior IDs, mask hashes, observation model mode, number of each ray status, exact energy decomposition, sample counts, optimization stop reasons, and numerical overflow/clamp counts.

### I.C. Exact derivative and occlusion tests (first acceptance gate)

Extend `tests/test_masked_rgb_observation.py` and `tests/test_visibility_constrained_iekf_phase1.py`; add `tests/test_phase1_occlusion_likelihood.py`.

- For a constructed front occluder (M=0, `d_occluder < z_object`), use the REAL CUDA contributors and verify the object **behind** a sharp observed depth has zero negative-ray score and zero corresponding JVP/VJP, excluding sort boundary pixels. For uncertain depth, verify the effect matches the calibrated marginal and diminishes continuously as separation grows. Never claim exact zero for a Gaussian depth distribution with nonzero tail probability.
- Moving the same Gaussian **in front** of measured non-object depth increases negative log-likelihood (check actual sign) and the finite-difference score matches the new derivative.
- Invalidate depth on `M=0` and verify the ray contributes zero object absence pressure, without fallback to total alpha.
- Test RGB-dark, RGB-background-colored and transparent/low-opacity gaussians; negative-ray evidence must depend on geometry/opacity, not their apparent matching of artificial grey.
- For M=1 contaminated with a synthetic EE first-return, demonstrate the calibrated unknown component receives posterior responsibility rather than automatically updating object; compare against a genuine object return and a genuinely newly visible surface that remains `UNKNOWN` until supported.
- Check a 3-camera scene with camera-specific calibration, mixed/invalid depths, occlusion from one camera and valid object visibility in another; visible camera evidence must survive.
- For every term: actual nonlinear `E(x±εv)` versus derivative, forward JVP versus finite difference, VJP adjoint identity, and normal operator symmetry/PSD within tolerances tied to fp32 CUDA error. Freeze contributor branches for local FD tests; separately test branch changes through the nonlinear accept/reject evaluation.
- **Exact baseline parity:** with the new flag OFF, previous test objectives, render outputs, trajectory centers, and output signatures remain unchanged (except added optional metadata when explicitly enabled). Do not accept a test that silently skips CUDA or reuses a stale compiled extension.

**Gate I.C:** do not begin grouped correction changes until the new likelihood is mathematically coherent and the tests above pass. Record the baseline and corrected gradient magnitude and drift on representative occluded Gaussian IDs.

### I.D. Correct the grouped optimization without retroactive pose resets

**Verified implementation:** `sparse_iekf.py:746–813` uses `G` for a **group update parameterization**; `sparse_iekf.py:857–863` skips `weak_direction_policy` if `grouped is not None`; renderer gives zero direct 6D-velocity Jacobian (`renderer_bridge.py:224–231`). The grouping by ~64 Gaussians is a numerical approximation, not a verified rigid segmentation.

For Gaussian `i`, let the new likelihood generate local 6D pose information `F_i >= 0`; no direct measurement information should appear in 6D velocity. Let the predicted 12D covariance be partitioned into pose and velocity:

\[
P_i^- = \begin{bmatrix}P_{\eta\eta,i}&P_{\eta\nu,i}\\P_{\nu\eta,i}&P_{\nu\nu,i}\end{bmatrix},\qquad
K_i=P_{\nu\eta,i}(P_{\eta\eta,i})^{-1}.
\]

The `K_i` relation is the **conditional mean** velocity correction for a local linear-Gaussian prior given a pose correction; it must be used only with correct tangent transport and a nonsingular/regularized prior block. Keep `process_predict` unchanged initially; do not reset hidden Gaussians to previous poses or set velocity to zero.

Whiten pose information with a consistent physical/prior scale `S_i`, e.g. a factor of `P_etaeta^-`:

\[
\widetilde F_i=S_i^T F_i S_i=U_i \operatorname{diag}(\lambda_{ij}) U_i^T.
\]

Construct a **declared** observation-supported pose map `Q_i` in the same tangent basis. It must be exactly zero on the numerical nullspace (`F_i=0`), and retain supported directions. A useful smooth candidate (to test against a hard well-defined numerical rank projector) is `U diag(lambda/(1+lambda)) U^T` in whitened coordinates; **do not claim this map is the unconstrained Kalman posterior**. Verify symmetry/adjoint in the actual state metric. The prior scale and numerical-rank threshold must be documented; do not invent a universal information threshold to force an outcome.

Lift a grouped 6D pose twist using the current `GroupedObservation.expand()` (`G_pose`) and then the local conditional velocity relation:

\[
B_i\,\delta u :=
\begin{bmatrix}I\\K_i\end{bmatrix}Q_i G_i^{\rm pose}\delta u.
\]

If a full 12D group parameterization is retained, its constrained pose/velocity map **must** be derived consistently from the actual 12D prior; do not simply set both 12D halves to the 6D projector or suppress prior cross-covariances. Using a 6D grouped update with the conditional velocity lift is an acceptable controlled implementation if dense-oracle tests validate it. Explain the restriction on the solution space.

Then use **the same B and B-transpose everywhere**:

\[
A_B=B^T(\Lambda^-+H^TWH+\lambda I)B,\quad b_B=B^Tb,\quad A_B\delta u=b_B,\quad \delta x=B\delta u.
\]

In code, locate `_G`, `_GT`, `solve_b`, `A_solve`, `group_precond`, `solve_metric`, `expand_step`, and trust-region predicted reduction and replace together in the opt-in path. Check that `B^T` includes the **transpose** of the per-Gaussian map and the velocity lift. Reject post-hoc gradient/motion masking that leaves the solver and trial objective inconsistent. The nonlinear trial must re-render the **same expanded** `boxplus(x,Bdu)`. The map is constructed at the current linearization and frozen for that iteration; rebuilt only for a new accepted iterate.

**Strict zero-information property:** For an object Gaussian not directly observed by any camera, if it is not contradicting measured free space and its visibility/ownership residual provides zero local Fisher, enforce `B_i=0`, hence `delta x_i=0` **relative to its predicted state**. `x^-` itself can move under the unchanged OU process; mark such intervals `PREDICTION_ONLY`, never treat them as measured movement.

#### Covariance (do not fake this)

The committed posterior is `approximate_posterior_information(Lambda_prior, F_i)` — the block approximation documented at `sparse_iekf.py:1091–1155`. It is not the marginal of the grouped constrained solve; the tests in `tests/test_phase1_covariance_velocity.py` already demonstrate overconfidence under co-visibility.

- Build a dense CPU oracle with 2–10 Gaussians, genuine prior pose–velocity cross-covariance and a grouped B map; compare the mean solution and **Joseph-form covariance of the actual linear measurement update** (including induced cross-block correlations) to any proposed scalable approximation.
- A Gaussian with `B_i=0` and no relevant prior cross-block correlation must keep its predicted 12D covariance `P_i^-=P_i^+` (not gain fictitious Fisher information). If the representation cannot carry the exact cross-correlations introduced by a group solve, do **not** mark its block covariance as exact.
- Derive a conservative production approximation from the oracle and stress tests; explicitly report its approximation class and numerical coverage, PSD, and any inflation. Avoid relying on an arbitrary diagonal floor as proof of uncertainty correctness. If the approximation does not meet declared coverage requirements, leave the group fix as diagnostic and do not enable it in production.
- Keep tangent transports, cross-covariance blocks, and all unobservable directions consistent; reproduce existing PSD safeguards.

#### Tests (second acceptance gate)

Add `tests/test_phase1_observable_group.py`, extend `tests/test_phase1_covariance_velocity.py`, `tests/test_deterministic_ops.py`:

1. Exact matrix-free adjoint identity `<Bu,v>=<u,B^T v>` and dense `B^T A B` parity (including pose-velocity correlation).
2. 2-member group: one directly observed, one fully hidden => hidden member's **measurement update** exactly zero, observed member can update.
3. Partially observable member: appropriate null directions do not receive group-forced updates; full-rank measurement recovers expected behavior of the defined parameterization.
4. Hidden in camera A but visible in B => nonzero information, no accidental over-freezing.
5. No information in any camera => mean stays at `x_prior`, `P_post=P_prior` for the corresponding block under the declared representation, while process prediction remains intact.
6. Dense constrained mean and Joseph covariance oracle; compare to production approximate posterior, report coverage/PSD discrepancy and refuse uncertified outcome.
7. Nonlinear trust-region actually lowers the **new** full MAP objective on accepted steps; no mismatch between reduced quadratic prediction and expanded trial state.
8. Deterministic CUDA mode repeatability of new forward/JVP/VJP/reductions, including supernode centroid reduction; record build/extension hashes.

**Gate I.D:** report trajectories, occlusion-conditioned measurement updates, predicted-only intervals, and measured drift against v6 (forward AND backward). Do not claim success merely because static motion decreased: measure moving-part reacquisition too.

### I.E. Phase-1 observability export contract (prerequisite for coherence)

Save per canonical original Gaussian and original frame/camera or compact sufficient statistics:

- valid object return counts and effective joint RGB-D responsibility; known-nonobject contradiction counts; unknown/occluded counts;
- per-Gaussian pose information eigenvalues or calibrated summary and a direct-observation-versus-prediction-only indicator;
- predicted mean and posterior mean, innovation correction, process-propagated pose delta, velocity, covariance quality flag;
- convergence reason (`converged`, `iteration_budget`, etc.), trust-region accepted/rejected counts;
- camera and frame index/original-frame mapping; forward/backward endpoint provenance; deterministic mode and model/kernel/source hashes.

When aggregating to graph supernodes, **do not merely average the time series and throw away observability and cross-correlations**. The current `supernode_trajectories()` averages centers unweighted; extend the new mode to carry the evidence needed by Part II while preserving old mode unchanged.

**STOP:** Part II must not be enabled if Phase I likelihood or grouped-update gates fail.

---

## II. MOTION COHERENCE — Validate moving support, do not pick the best drift

**Deliverable II:** opt-in conservative extractor using Phase-I observability metadata and independently held-out **real nonlinear RGB-D rendering** to choose between a stationary scene and a candidate rigid moving component; no Gaussian births and no Phase-4 joint assumptions. Do not edit Phase-I mathematics during this part.

### II.A. Keep graph generation as candidate proposal, never as qualification proof

Retain `estimator/motion_coherence_support.py` graph building, edge log-strain `C_ij`, supernodes, and `estimator/coherence_extraction.py:candidate_hierarchy`/stratified candidate pool as **candidate generators**, subject to stability diagnostics. They are NOT final evidence that a part moves.

Currently `cross_validated_rigidity()` uses node-index folds from common tracks; `select_moving_component()` qualifies on positive relative `eta_cv`/`delta_rss_cv` and maximizes world `delta_rss_cv_per_node_frame`; `refine_from_core()` unconditionally recruits any other node if `E_m<E_s`, without graph adjacency. These are verified code facts and must remain available as baseline controls, not as the production criterion in the new mode.

Create `estimator/coherence_validation.py` (or a justified local equivalent) to build and validate, for each candidate `C`:

\[
H_0:\ \mathcal G_t=\mathcal G^0,
\qquad
H_C:\ \mathcal G_t=\mathcal G_{\bar C}^0\cup T_C(t)\mathcal G_C^0.
\]

`T_C(t)` is estimated from candidate tracks/rigid model using only **training** observations and actual timestamps; the complement remains static in the calibrated camera world frame. Its orientation and anisotropic covariance are rigidly transformed with means. Fit candidate pose on the training observations **without looking at validation RGB-D**.

### II.B. Held-out RGB-D likelihood and null decision

Produce a deterministic camera holdout design: for each split, estimate Phase-1 trajectories and candidate `T_C(t)` on training camera(s) ONLY, then render and score **unseen camera(s)** with the exact new nonlinear observation likelihood and no refit. Do not call previously full-multiview-extracted trajectories a held-out test. If a shared canonical 3DGS was trained with all cameras, state that the test is conditionally held out with respect to motion fitting, not fully independent of initialization. Optionally use temporal holdouts ONLY if candidate dynamics predict those times without optimizing poses there.

For candidate `C`, compute

\[
S(C)=\sum_{(c,t)\in\mathcal V}\Big[L(y_{ct};H_0)-L(y_{ct};H_C)\Big].
\]

The likelihood and ray statuses must be identical to the corrected Phase-1 model, with **no IEKF linearization in final validation**. Record RGB/first-return/negative-ray components, eligible pixel counts and per-camera scores. If no validation views see candidate motion, mark `UNIDENTIFIABLE` — do not divide by zero or issue a positive verdict. Control candidate complexity / multiple hypotheses: arbitrarily large components cannot be favored merely for explaining more pixels and the same candidate must be traceable across camera splits.

A positive `S(C)` alone does not qualify. Calibrate the **maximum candidate score** under a fixed, declared stationary-null experiment reproducing known sensor variances, calibration, mask errors, depth-edge contamination and *correlated* Phase-1 tracking drift:

\[
p_{\rm null}={1+\sum_{b=1}^B\mathbf1[\max_C S_b(C)\ge\max_C S_{\rm observed}(C)]\over B+1}.
\]

Use predetermined statistical error rate `alpha` and fixed random seeds; document choice, effective independent replicates and sensitivity. **No GT used to select parameters.** Do not fake a null distribution by assuming Gaussian independent node noise when grouped IEKF has shared error. When sufficient matched null controls are missing, produce `INSUFFICIENT_NULL_CALIBRATION`, not a forced positive result. If observations cannot distinguish the winner, return `NO_CONFIDENT_MOVING_SUPPORT`. Require a supported **stationary complement** and camera-split consistency, or `AMBIGUOUS_SIDE` rather than forcibly choosing the static body as moving.

Implement deterministic tie-breaking (stable node IDs / candidate ID), stable CUDA reductions for scoring, and store every candidate's reasoning in a structured JSON. The chosen objective must be evaluated after all data-dependent candidate selection inside the null-control simulation (family-wise calibration).

### II.C. Conservative recruitment only after a validated core

Disable legacy global `refine_from_core()` in the new mode. For each neighboring supernode with measured evidence, compare

\[
D_i=E_i^S-E_i^M,
\]

using the core trajectory **fit without that node**, observed-only motion evidence, calibration of correlated errors and a declared multiple-comparison rule. `D_i` values within uncertainty lead to `UNRESOLVED`, not moving or confidently static. Recruit only from graph neighbors or supported 3D surface frontier, preserving physical connectivity and separate layers. Recheck held-out RGB-D after recruitment, preserving the original trusted core. Do not tune recruitment to achieve GT F1 or object-size targets. Comprehensive recovery of original moving Gaussians belongs to Phase 4A, not the coherence core.

Maintain three labels: `TRUSTED_MOVING`, `STATIC_COMPATIBLE`, `UNRESOLVED`. Do not write all non-moving nodes as *confidently static*. Ensure `predicted_support.json` still has the fields Phase 2 expects (`status`, `dynamic_gaussians`, correct original Gaussian indices) plus new confidence/provenance fields, without breaking legacy readers. The explicit null state must stop Phase 2/3, not enter joint fitting with a dummy empty or near-whole support.

### II.D. Phase 2 protection (small dependent compatibility change, NOT another recovery stage)

`estimator/free_space_support.py:assign()` currently defaults to `mode='replace'`, can recruit most of the scene, and falls back to previous members if fewer than `min_members` meet its test. For **new coherence mode only**, add an explicit preflight in `tools/run_rigid_se3_pipeline.py`: reject unresolved or uncalibrated supports; reject/diagnose a refined set that replaces the trusted core or is unsupported by a rigid scene model. Keep original mode unchanged as a historical control. Do **not** treat `intersect` as a solution to missed-moving-point recovery. If supporting complete-scene counterfactual tests are needed for Phase 2, label this as a separate, opt-in core-consistency verification, not a new recruitment algorithm.

### II.E. Code changes and output contracts

- `estimator/coherence_extraction.py`: retain legacy functions; add new `select_*`/`refine_*` entrypoints or a clearly named new mode. Preserve the candidate hierarchy's reproducible run ID and numeric reasoning.
- `estimator/coherence_validation.py`: new training/validation split protocol, actual full-render hypothesis scoring, explicit null test, null controls and candidate ranking.
- `estimator/motion_coherence_support.py`: propagate Phase-I reliability alongside `supernode_trajectories`; do not require direct covariance from a block approximation to be treated as calibrated.
- `tools/run_motion_coherence_support.py`: optional `--coherence-selection legacy|heldout-null`, default legacy; new control dataset input/config, `alpha`, validation camera splits and diagnostic paths; fail with informative reason when evidence unavailable.
- `tools/run_rigid_se3_pipeline.py`, `estimator/free_space_support.py`: fail-closed new-mode core contract; original free-space `replace` behavior kept in legacy mode.
- `run_bidirectional_all.sh` + stage signatures: separate new Phase-I observation, group update, and coherence flags. Ensure a new experiment cannot mistakenly reuse a v6 cached Phase-1 trajectory or a support from another camera split.

### II.F. Tests (third and fourth acceptance gates)

Add `tests/test_coherence_null_validation.py` and `tests/test_coherence_support_recruitment.py`, extend `tests/test_motion_coherence_support.py` and the repository's rigid pipeline tests.

**Synthetic, deterministic:**

1. Truly static entire scene with independent depth noise => `NO_CONFIDENT_MOVING_SUPPORT` at declared false-positive rate.
2. Static entire scene with *correlated* global/occlusion-induced drift and a near-identity Kabsch core => must **not** recruit scene by `E_m<E_s` sign alone.
3. One true articulated rigid component and truly stationary complement => correct side selected, quality held-out across cameras; stability for 3DGS density changes.
4. Static side much larger than moving side => no preference solely due to size.
5. Moving part near hinge (low displacement), partial occlusion, two contrasting viewpoints => unresolved when unobservable, recovery of confident support where actually observed.
6. EE contam inside `M=1`, moves independently of object => must not become part of object moving core; if indistinguishable from object across all cameras/time, explicit `AMBIGUOUS_OBJECT_OWNERSHIP`, not false guaranteed exclusion.
7. Camera validation leakage test: fit trajectories using validation camera and verify test detects the violation (asserted split disjointness).
8. Null controls use the same candidate search and choose the **maximum** score per replicate, not a single preselected candidate.
9. Existing node-CV tests still pass in legacy mode; new mode cannot succeed using only node-CV scores.
10. Forward and backward processing reorder frames correctly; validation sees original chronological acquisition and the matching endpoint model.

**Real-scene verification (no success claims without running):**

- At minimum red fridge, light-blue box (previously stable cases), black laptop and robot dog (failure cases); include wooden box if time/resources permit.
- Run repeated identical full-input, same-GPU deterministic Phase 1 on each selected scene, **both directions** where endpoint models exist.
- Report `PREDICTION_ONLY` displacement separately from observed motion; per-frame nonconvergence and posterior status; camera-by-camera negative/positive ownership counts; frozen, grouped, and corrected update comparisons.
- Compare selected original-Gaussian IDs and Jaccard overlap of core supports between repeats, against v6/draw1/draw2. Do not use these metrics as the sole correctness criterion; include real RGB-D validation and whether a moving component should be declared unidentifiable.
- Report forward/backward mask contamination sensitivity and raw RGB-D fit without generating new Gaussians. No GT during selection; GT metrics may be reported only after saved predictions.
- Missing/invalid calibration or no null controls => **explicitly unvalidated**, not a successful run.

## III. Execution sequence, acceptance criteria, and required Codex report

1. **Baseline audit:** print `git` status and hashes; document exact local call graph and whether existing CUDA binaries match source; run `python tests/test_masked_rgb_observation.py`, `python tests/test_visibility_constrained_iekf_phase1.py`, `python tests/test_phase1_covariance_velocity.py`, `python tests/test_deterministic_ops.py`, and `python tests/test_motion_coherence_support.py` where supported. Record CUDA availability; a `NOT_RUN` is not a pass.
2. **Stage I.A–C:** implement **only** the observation likelihood; pass true-energy/JVP/VJP/Fisher/full-nonlinear tests. Save immutable test manifest and diff.
3. **Stage I.D–E:** implement **only** observable grouped corrections and observability export; prove dense oracle and zero-information properties; run actual Phase-1 A/B in real representative scenes, with deterministic reductions. Do not proceed if covariance is unverified.
4. **Stage II.A–B:** preserve original candidate generation; implement held-out full-render core qualification + calibrated null, explicitly separate it from Phase-I changes; test static/drift/moving controls.
5. **Stage II.C–D:** implement conservative recruitment and Phase-2 fail-closed contract; test small-scene and saved real-scene outputs; run exact held-out revalidation after recruitment.
6. **End-to-end:** run v6 and the new opt-in mode on identical scene data and camera calibration, with saved seeds, hashes and no training; compare accuracy, uncertainty, runtime, failure rate and support stability. Only after gates pass consider new mode production-ready. Leave Phases 4 and 5 untouched.

### Mandatory final report `PHASE1_COHERENCE_OCCLUSION_IMPLEMENTATION_REPORT.md`

Include: precise local commit/branch/diff; source and kernel hashes; list of modified files and flags; implemented equations vs proposed-but-not-implemented concepts; test commands/output including NOT_RUNs; dense-vs-CUDA FD and adjoint residuals; covariance oracle discrepancies and coverage; per-frame convergence and timing; per-camera observation ownership histograms; static-noise and correlated-drift controls; full-render held-out likelihood and null calibration; results for each real scene by direction with exact member counts and Jaccard; remaining limitations and next steps.

**Do not replace missing mathematical tests with tuned thresholds or optimistic prose.** The report must plainly state what remains unimplemented, approximate, statistically uncalibrated, or not executed. If a stage fails, do not fabricate an acceptable core or fall back to legacy semantics without saying so.

## IV. Hard constraints / non-goals

- Preserve existing camera intrinsics/extrinsics, timestamp conventions, canonical 3DGS, Gaussian indexing, appearance and renderer semantics. Never substitute unrelated RGB-D/SLAM approaches.
- No image-space erosion as proof of object ownership, no arbitrary 20 mm occlusion threshold, no post-IEKF state reset, no force-zero-velocity hack, no label propagation purely from 2D adjacency.
- Do not double-count pixels through simultaneous legacy masked RGB/alpha and new depth-bounded likelihood. **Every nonlinear trial uses the true selected objective.**
- Full actual rasterizer validation is mandatory; IEKF linearization is allowed only for local optimization and must pass adjoint and finite-difference checks.
- All camera frames must be considered for published real-scene validation. Small subsets may be used for development only and labeled as such.
- No Phase 4 Gaussian birth, no Phase 5 training, no joint changes, no GT-driven optimization or per-scene hand tuning.
- Physical identifiability cannot be guaranteed when an EE follows exactly the same observed motion as the target surface; `UNKNOWN` is a valid and sometimes necessary outcome.

**Acceptance standard:** One can reproduce the original v6 mode, demonstrate that occluded rays exert no spurious object-removal force, show that unobserved members cannot be dragged by computational supernodes, and identify a confidently moving core **only** when its physically rendered held-out evidence exceeds a calibrated stationary null. Anything weaker is an experiment, not a verified fix.
