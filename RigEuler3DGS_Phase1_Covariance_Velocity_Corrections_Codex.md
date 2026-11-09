# CODEX — RigEuler3DGS Phase-1 covariance, false velocity and real RGB-D divergence

**Target repo:** `https://github.com/SamueleMara/RigEuler3DGS`  
**Inspected baseline:** pushed `main` commit `7954660` (implementation in parent `3b4b527`, compared with `4c88725`).  
**Request:** correct the scientifically inconsistent measurement uncertainty, grouped motion/covariance and speed feedback that cause the real RGB-D Phase-1 IEKF to diverge. **Preserve Phase 2, Phase 3 and the confident-support algorithm.**  
**Status of this document:** an implementation and experiment specification. Confirmed code defects and measured symptoms are distinguished from hypotheses and optional research prototypes. Do not claim any of these proposals has already improved tracking.

## 0. Executive diagnosis and non-negotiable invariants

The saved full forward run of `final_red_fridge_episode_000000`, frames **100–288**, has **30,316 Gaussians**, **3 calibrated cameras**, a **0.04 s physical frame interval**, and the measured `sensor_model_id` `4588849703872584d30585e70edf796004601ef868ce83120070d99e3ee03e0c`. Backward Phase 1 also diverges according to the supplied Codex forensics. Use these as reproduction identifiers, not as universally valid input paths.

| Frame | Median displacement from canonical | p99 displacement | Maximum | Fraction more than 20 cm away |
|---|---:|---:|---:|---:|
| 101 | 2.7 mm | 35 mm | 0.56 m | 0.1% |
| 110 | 48 mm | 0.42 m | 4.6 m | 4% |
| 140 | 0.31 m | 3.6 m | 50 m | 78% |
| 288 | 2.6 m | 43 m | 291 m | 99.9% |

The source report found ~99.8% of eventually divergent Gaussians still within some foreground mask when they start moving. Mask exit and subsequent constant-velocity coasting are **late amplifiers**, not the first cause. The reference model already has ~20 mm depth disagreement at the canonical frame, versus calibrated depth sensor SD ~0.7 mm; RGB residual RMS ~0.085 versus sensor SD ~0.0045. A physical-noise-only likelihood is consequently grossly misspecified. Approximately **all 188 forward transitions terminate at the 12-iteration budget**, so covariance based on the last local Hessian is not certified by convergence. The initial velocity SD is declared **1 m/s**, while bootstrap pose SD may be ~0.19 mm; prediction at 0.04 s has ~40 mm position SD before measurements.

### Preserve all of the following

1. **Phase 0:** fixed-population, separate forward/backward endpoint 3DGS bootstraps; no combining observations from different articulation configurations into an endpoint.
2. **Phase 1:** every Gaussian's motion remains possible; do not impose GT static/moving labels, hard maximum displacement/speed, arbitrary gates, per-point stationary bias, or a hidden joint hypothesis.
3. **Phase 2/3:** existing high-precision motion-support extraction, rigid shared `SE(3)` estimation, revolute/prismatic comparison and reduced joint filtering **are untouched**. Existing synthetic joint accuracy must remain reproducible.
4. Use the **production nonlinear 3DGS renderer** for accepted updates and validation; preserve existing analytic JVP/VJP where used. No surrogate render as final evidence.
5. Object masks remain hard constraints on **which RGB-D pixels belong to the object**. Never use background depth as a foreground measurement. Mask occupancy/alpha may be separately probabilistic, but do not reinterpret a mask as GT motion.
6. Maintain explicit camera identity and calibrated timestamps. Differentiate **sensor noise**, **renderer/model discrepancy**, **mask uncertainty**, **process noise**, **bootstrap geometry uncertainty**, and **cross-Gaussian coupling**.
7. Do not start a 14-episode or 10-scene production run before evidence gates. Do not overwrite prior results or change global defaults without a demonstrated gain. Do not replace the IEKF or add a new point-discovery phase during the correctness gates.

### Execution discipline

- Begin with `git rev-parse HEAD`, `git status --short`, and `git diff 7954660 --stat`. Adapt line references to HEAD and show each deviation from this baseline. Preserve local/uncommitted changes.
- Work in a **separate diagnostic branch**. Keep all old algorithm modes runnable and results cache-isolated by complete signatures. Run with explicit random seeds and actual calibrated model IDs.
- Distinguish `CONFIRMED CODE DEFECT`, `MEASURED FAILURE`, `EXPLANATORY HYPOTHESIS`, and `EXPERIMENTAL ALTERNATIVE` in reports and commit messages. If a cause is refuted, stop implementing that causal fix.

## 1. Gate A — Make the first false correction causally observable (diagnostic; do not change the estimator)

**Files:** `tools/audit_calibrated_phase1.py`; `tools/kpart_pipeline.py::bootstrap_positions`; `estimator/observation_terms.py`; `estimator/sparse_iekf.py`. Extend the existing read-only audit, do not replace it.

For real frames **100, 101, 102, 103, 104, 105**, and a separately **verified stationary** portion where possible, capture for each camera and the common rendered state:

1. At **canonical zero displacement**, **propagated prior** and **accepted posterior**: real-render RGB/depth/alpha, masks, valid depth, per-pixel residual distributions, signed depth error, and foreground/off-mask alpha. The canonical frame itself is a required control: it is *not* an independent test of a bootstrap fitted using that same frame.
2. Before/after every accepted IEKF inner step: total and per-camera RGB, depth, alpha, prior energies; **objective decrease and gradient contribution with the exact weights/units**. Do not compare raw totals as if their likelihoods had equal physical dimensionality. Record number of eligible pixels, valid depth, and actually used depth (`A > min_alpha_for_depth`).
3. Per camera: `g_c = J_c^T W_c r_c`, full normal/Fisher including alpha, and information-direction spectra evaluated at **the same pre-update state**. Explicitly separate gradients at the posterior (whose sum may cancel as an optimum) from gradients at prior/zero-motion. Do not infer sensor noise domination from ~50% opposition of posterior camera gradients.
4. For identifiable Gaussian subsets: depth-pixel source/occlusion, canonical responsibility and visibility, rendered versus observed depth at the actual contributing pixels, gradient direction relative to camera rays, distance moved *along* each camera ray, and the other cameras' response at the identical state.
5. 12D prior/updated means `p, R, v, omega`, full `12 x 12` block prior/posterior (including pose–velocity cross blocks), `P_pred`, actual `F P F^T`, `Q`, and per-iteration state increments. Report the induced speed change and the fraction attributable to pose–velocity coupling.
6. Solver status: trusted-region acceptance/radius, PCG tolerance/residual, termination cause, scaled gradient norm, and whether the last local linearization passes a finite-difference consistency check. `iteration_budget` is *not* convergence.
7. Identify **which depth returns** produce the reported 20 mm canonical mismatch (camera, mask, depth edge, foreground occlusion, depth normalization, Gaussian centre versus rendered surface). Do not assume all 20 mm is noise or a fixed camera bias.

**Confirmed audit defect to fix first:** `tools/audit_calibrated_phase1.py` calls `lin.build_block_fisher(prec_c)` but omits the alpha block, while production `estimator/sparse_iekf.py::measurement_operators` includes `lin.build_alpha_fisher`. Add the per-camera alpha Fisher with the same `kappa` as production. Verify `F_all ≈ sum_c F_c` against the production Fisher on identical linearization and channel terms, with a finite-difference/JVP test. Preserve no-grad and audit-on/audit-off state invariance.

**Pass/fail:** The exact first-frame displacement, post-update velocity, and camera/channel terms must be reproducible from the recorded snapshots; the audit must not modify state. Any early move of a Gaussian with no eligible measurement must be classified as **prediction-only**, not falsely attributed to a camera.

## 2. Gate B — Fix the canonical supernode partition (CONFIRMED CODE DEFECT)

**Files:** `tools/kpart_pipeline.py::bootstrap_positions` (~1738–1907); `estimator/sparse_iekf.py` (~577–583); `estimator/supernodes.py`.

Current: Phase 1 invokes `iekf_frame_step(..., partition=partition)` with `partition=None`; inside the IEKF, if `cfg.use_supernodes`, it calls `build_supernodes(x_prior.mu, ...)`. Thus groups are rebuilt every frame from already moving/drifting geometry, although the canonical partition is supposed to remain fixed over the directional sequence.

**Required narrow correction:** construct an immutable `SupernodePartition` once, from `truth.mu` at the appropriate **direction-specific canonical endpoint**, before `bootstrap_positions`'s frame loop. Pass it through `iekf_frame_step` on **every** update, including backward processing. Keep `use_supernodes`, group expansion/reduction and the current per-Gaussian state intact. Record partition hash, Gaussian membership, canonical centres, and target group size in the Phase-1 manifest. No adaptive regrouping.

**Tests:** (i) one hash across every frame despite large synthetic motions; (ii) forward and backward each independently canonical to their own endpoint, never mixed; (iii) grouped `G`, `G^T`, and `G^T A G` adjoint/finite-difference oracles; (iv) conservation of Gaussian ordering; (v) synthetic rigid-moving support not locked into a stationary group by this change. Do not assume that a fixed group corrects the entire drift; measure the change in an A/B.

## 3. Gate C — Fix the interpretation and weight of the real-data likelihood

**Files:** `estimator/observation_terms.py::camera_terms`; `estimator/masked_observation.py::kappa`; `estimator/renderer_bridge.py`; `utils/noise/sensor_model.py`; Phase-0 manifests. **No unvalidated automatic sigma tuning.**

### C1. Explicit physical generative model

The current physical calibration estimates fluctuations in **raw static sensor data**, not a complete 3DGS rendering-error covariance. Start from

```
y_(c,t) = h_c(x_t, geometry, appearance) + b_(c,t) + epsilon_(c,t)
epsilon ~ sensor noise from static calibration
b       ~ model/geometry/registration/boundary discrepancy, possibly biased and correlated
```

Report separate `R_sensor`, residual mean/bias, `R_model`, inter-pixel/camera/time correlation, and any nuisance state. Do not set `R_model = 0` simply because calibration exists. Do **not** infer `R_model` solely from temporal variance at fixed pixels; persistent depth/render bias and edge errors are not identified that way.

Use reference images **not treated as independent endpoint-training validation**; held-out pixels, stationary repeat frames, and synthetic known geometry are the preferred evidence. Split residuals into inlier surface interiors, depth discontinuities/occlusions and other geometry cases; report their conditional distributions without introducing arbitrary acceptance gates into production.

### C2. Investigate RGB–alpha precision coupling, keep hard masks

The masked expected-background mode uses `kappa = trace(W_rgb Sigma_b)`, with `Sigma_b = I/12`. For three equal channels, `kappa = 1/(4 sigma_rgb^2)`. This becomes `25` for `sigma_rgb=0.1` but `10000` for calibrated `sigma_rgb=0.005`: **400x stronger alpha precision** without independent mask calibration. This is an actual coupling in the code, not a proof the alpha factor is incorrect.

**Controlled arms (not an immediate default change):**

- Existing expected-quadratic alpha at existing calibrated `W_rgb` (baseline).
- Foreground-only physical RGB-D with hard masks and an explicit, separately declared silhouette factor. Compare occupancy/silhouette likelihoods against observed masks; verify alpha gradients/Fisher use exactly the chosen factor in all update operators and covariance diagnostics.
- If appropriate, a calibrated Bernoulli opacity likelihood `-M log A - (1-M) log(1-A)` **only after demonstrating the rasterizer alpha can be interpreted as an occupancy probability** and handling model-completeness uncertainty. This is an experimental candidate, not required or automatically preferable.

No background depth or background RGB should be promoted to foreground measurements; preserve negative-ray silhouette evidence independently. Record normalizations and pixel counts so objective strength is interpretable across cameras and modes.

### C3. Sensor/model discrepancy covariance and residual correlation

For unexplained zero-mean residual, the first test model can be `R_eff = R_sensor + R_model`, with separately reported units/terms. For correlated or nuisance-variable residuals, implement the justified **joint covariance** or marginalization, not a diagonal approximation disguised as complete statistical calibration.

Do not tune an RGB or depth sigma against GT motion or change these values automatically per frame. Sensitivity experiments with declared parameters are allowed, and must distinguish static error, dynamic tracking and held-out likelihood.

## 4. Gate D — Prevent persistent bootstrap bias from being interpreted as motion (EXPERIMENTAL measurement correction)

**Separate opt-in mode only until validated**; primary candidate is a reference-relative RGB-D factor using the production nonlinear renderer and visibility-aware association. Do not use raw unaligned difference images.

For the canonical observation `y_c0` and subsequent observation `y_ct`, write

```
r_c,t(x_t)     = y_c,t - render_c(x_t)
r_c,0(x_0)     = y_c,0 - render_c(x_0)
r_rel_c,t     = r_c,t - W_c,0->t(x_t) r_c,0
```

`W` denotes physically valid transport of *reference residual evidence* using Gaussian-wise source responsibility, current projected geometry, camera visibility/depth ordering and potentially changed occlusion. It **must not** copy an image-space residual to the same pixel when the surface moved. Reference evidence that is newly occluded or not corresponding has no transport factor; newly visible observations need their own properly uncertain absolute factor. `W` is state dependent, so the Jacobian includes its derivative or an explicitly justified frozen linearization consistently in the GN step and nonlinear acceptance test.

**Do not double-count the reference:** it was used in the endpoint bootstrap; its error is correlated with P0 and with every future reference-relative factor. Covariance must account for this. For a linearized warp:

```
R_rel,t = R_t + W_t R_0 W_t^T - Cov(e_t, W_t e_0) - Cov(W_t e_0, e_t)
Cov(r_rel,t, r_rel,s) ≠ 0 because the same reference observation is reused
```

If this dependence cannot be handled reliably in a recursive filter, implement a short fixed-lag factor model (with a shared reference nuisance variable) **as a separate prototype** rather than claiming independent per-frame IEKF measurements.

**Alternative:** explicit persistent camera/surface nuisance bias `b` estimated from a verified static interval and marginalized in a joint pose/bias objective. Never apply a bias correction and independently add the same bias uncertainty into `R_model` (double counting).

**Evaluation oracles:**

1. Real verified-static interval: a model with a persistent (but temporally stable) depth/appearance offset should yield near-zero supported displacement, without a speed prior or GT labels.
2. Synthetic articulated sequence with known motion and deliberately injected static camera/model bias: recover *nonzero* motion and correct joint support while canceling persistent bias.
3. Negative controls: changing viewpoint, appearance, occlusion, newly appearing surfaces, reference mismatches and missing depth must not create false correspondences or make new observations disappear.
4. Exact/nonlinear rendering recheck at every accepted update, analytic-vs-finite-difference derivative oracle, and covariance consistency under reuse of the reference across multiple frames.

**Decision rule:** promote a reference-relative mode only if it decreases *held-out real static drift without destroying synthetic moving-point tracking*, not merely because its own training loss decreases.

## 5. Gate E — Repair uncertainty estimation and distinguish state covariance from estimator error

**Files:** `estimator/covariance.py` (~383 onward); `estimator/sparse_iekf.py` (~991–1045); `estimator/state.py` (~589–648); `estimator/process.py`; `estimator/supernodes.py`.

### E1. Exact meaning of the present approximation

The current approximation is per-Gaussian `P_i ≈ (Lambda_prior,i + F_ii)^(-1)` for a 12D pose/twist block. But the nonlinear renderer couples co-visible Gaussians; the true local Gaussian marginal is

```
P_ii = [(Lambda_prior + H^T R^-1 H)^-1]_ii,
```

**not** the inverse diagonal information block. With correlated measurements, `R^-1` must represent their actual joint precision. Grouped solving makes this more important: optimization is in a group tangent while the recursive uncertainty remains per-Gaussian blocks. Mark all output blocks `BLOCK_INFORMATION_APPROXIMATION`, not exact marginal uncertainty.

### E2. Dense tiny-case oracle before production changes

Build an independent CPU, two-camera, **2–10 Gaussian** linear-Gaussian test with occluded/co-visible rays and a fixed 12D canonical supernode map. Form the full matrix, invert it in float64, compare against the current diagonal-block inverse, and explicitly quantify overconfidence and group cross-correlations. Include a controlled shared bias perturbation. Verify analytic `J^T`, `Jv`, normal-operator and alpha Fisher. No hand-selected tolerance gates in inference.

### E3. Scalable covariance candidate (do not replace default before E2 passes)

Reuse the existing matrix-free normal operator and PCG for **inverse-operator probing** or a smaller **fixed-group latent covariance**. One unbiased probe construction (for a fixed SPD normal matrix `A`) is

```
A u_k = z_k ; E[z_k z_k^T] = I
P_hat_ii = (1/K) sum_k u_(k,i) z_(k,i)^T
```

The finite-K result is noisy, not necessarily PSD and cannot silently be projected and called exact. Quantify probe variance, convergence, group-to-group correlations and PCG error against E2 before scaling. If the state is actually parameterized as group latent `z` with mapping `G`, its Gaussian covariance must be propagated as `G Cov(z) G^T` plus any **explicitly modeled** independent residual-deformation covariance. Do not simultaneously add group and Gaussian uncertainty twice.

A robust/sandwich estimator `A^-1 B A^-T` is a separate candidate for **sampling uncertainty under model misspecification**. It requires a justified score-covariance `B` (spatial/temporal correlations matter). It is not the same as the Bayesian conditional covariance; do not call them interchangeable or claim reliable cluster statistics from only three cameras.

### E4. Non-convergence and consistency

The current solver updates posterior Fisher information after each `iteration_budget` stop. A local Hessian inverse at a nonconverged state is **not a validated physical posterior covariance**. Report convergence/optimality and posterior status separately. Do not invent a fixed inflation multiplier. First test whether increasing the real convergence budget, or reducing likelihood misspecification, actually yields convergence and stable coverage. If it still fails, derive a defensible covariance for the approximate algorithm or mark uncertainty uncalibrated; do not silently propagate a falsely precise posterior.

### E5. Proper SO(3) covariance coordinate transport (surgical mathematical audit)

The `state.py::boxplus` convention is `R+ = Exp(dphi^) R` (left/spatial). Derive and finite-difference test the tangent-coordinate reset/transport Jacobian for the accepted correction under this convention. Propagate `P_new = J_reset P_local J_reset^T` **including cross-blocks** if the computed covariance is in the old tangent, and do not apply this twice if the posterior linearization already expresses covariance at the accepted tangent. Compare this with how `Lambda_prior` and `boxminus(x, x_prior)` are represented. Existing PSD projection is a numerical safeguard, **not** a substitute for a covariance reset. Never reset P to P0 each frame.

**Tests:** group/infinite-confidence marginal oracle, covariance PSD and symmetry, positive `P_vp`, forward/backward time signs, finite-difference Lie reset, non-converged-status metadata, repeat-sequence empirical coverage on synthetic GT, and audit-on/off invariance. Keep exact residual quantities and units.

## 6. Gate F — Correct speed inference without suppressing genuine motion

### F1. Identify why an initial position discrepancy becomes velocity

The Phase-1 state is `[p, phi, v, omega]`. The current initial covariance has `P0_pp ≈ (0.00019 m)^2`, `P0_vv = (1 m/s)^2`, negligible initial cross-block, and `dt ≈ 0.04 s`:

```
P_pred,pp ≈ P0_pp + dt^2 P0_vv + Qpp
           ≈ (40 mm)^2                 (velocity contribution)
P_pred,vp ≈ dt P0_vv                 (ignoring Q and geometry couplings)
```

A sufficiently strong but *biased* position measurement can induce `delta_v ≈ delta_p / dt`. Thus an erroneous 5 mm position correction could create approximately `0.125 m/s` velocity, and the constant-twist prediction keeps transporting it. This is **consistent with the equations**, not necessarily an algebraic Kalman bug.

Instrument the conditional gain from pose/image to velocity by block, compare `delta_v` predicted by the linearized update with the actual accepted nonlinear delta, and inspect whether errors propagate into angular velocity too. Distinguish a **40 mm predicted standard deviation** from a **40 mm observed movement**.

### F2. Correct the initial velocity inference, not via speed damping

Keep the present 12D IEKF available. Experimental option: estimate initial pose/twist from a small time-aligned multiview sequence using the corrected likelihood and a physically defined integrated-white-acceleration model, rather than treating the first image discrepancy as a confident speed observation. Explicitly separate the uncertain starting velocity prior from measured temporal displacement. Do not use GT speed, maximum velocity, per-point static labels, arbitrary velocity shrinkage, or a constant-travel-distance cap. Where motion evidence is insufficient, report unobservability and uncertainty, not a forced zero-speed state.

The current `estimator/covariance.py::make_process_noise` implements the standard continuous-time integrated-acceleration terms `Q_pp = q dt^3/3`, `Q_pv=q dt^2/2`, `Q_vv=q dt`. Retain this physical convention unless an oracle proves a defect; compare process noise and initial velocity variance separately. **Do not simply reduce `sigma_v` until the real static sequence looks good**: that would also bias genuinely moving parts.

**Required tests:** stationary point/cloud with biased renderer, constant translational/revolute motion, nonconstant articulated motion, temporary occlusion and reappearance, and pure zero-measurement intervals. Verify near-zero motion when the information supports it; verify nonzero constant velocity is not incorrectly suppressed. Use real `dt`, not frame counts.

### F3. Optional pose-only fixed-lag Phase 1 (separate opt-in prototype, NOT replacement)

If the corrected 12D model remains unstable, implement a separate local-window estimator using 6D pose per Gaussian (or fixed-group latent) and an integrated-acceleration factor. Its velocities are *derived with uncertainty from the pose sequence*, not directly updated from one image frame:

```
v_t ≈ Log(T_(t+1) T_t^-1) / dt_t
E_acc = physically discretized covariance-weighted variation of v over time
```

**Important:** penalizing acceleration alone does NOT prevent a false constant speed (which has zero acceleration). This prototype requires the corrected likelihood and adequate temporal anchoring; otherwise it will drift too. Preserve temporal factors across occlusion without creating new optical/depth correspondences. Evaluate the **full nonlinear renderer** and visibility for accepted trajectories. The regularization must come from declared process covariance/densities, not a hard motion threshold.

Deliver side-by-side metrics for original 12D, corrected 12D, and pose-only fixed-lag modes. Phase 2 and Phase 3 must accept the same trajectory artifact contract; do not migrate them until equivalence and accuracy gates pass.

## 7. Gate G — Additional surface/occlusion limitation (tracked, not a velocity fix)

`tests/test_bootstrap_geometry_eval.py` still reports **normal median 2.9 deg against 2 deg test tolerance** and **71% merged seeds** for a 4 mm two-sided panel. Preserve these two `FAIL` results as genuine findings. Do not claim all tests passed or hide them by changing tolerances. The proposal to select samples by `n·(camera - seed)>0` is only safe after obtaining an **oriented and visibility-consistent normal**; an unoriented PCA normal from an already merged front/back layer cannot disambiguate sides.

A separate small study may construct physically visible neighborhoods from ray direction, depth ordering, and camera ownership, then rerun the known-surface synthetic oracle. Do not change surface fitting in the covariance/speed patch unless this study independently demonstrates a specific correction.

## 8. Controlled ablations — mandatory before promoting any change

**Common controls:** same raw RGB-D sequence, exact masks, camera set and calibration, original frame IDs/time intervals, canonical endpoint geometry, Gaussian order, deterministic seeds, stage signatures and held-out evaluation pixels. If endpoints differ, distinguish the effect of geometry from dynamics/likelihood. The previous baseline standard 3DGS is unavailable at motion frame 100, so *do not substitute frame-0 COLMAP model and pretend it is a comparable endpoint*.

| Arm | Fixed partition | Real-data measurement model | Covariance | Motion model | Purpose |
|---|---|---|---|---|---|
| A | no (legacy) | calibrated sensor only, existing alpha | current block approx | 12D CV | Reproduce explosion |
| B | yes | exactly as A | current block approx | 12D CV | Isolate group fix |
| C | yes | explicit model mismatch + separately inspected alpha factor | current block approx | 12D CV | Isolate likelihood |
| D | yes | same as C | tested corrected grouped/marginal estimate | 12D CV | Isolate covariance |
| E | yes | reference-relative or nuisance-bias factor, with shared-reference covariance | tested corrected estimate | 12D CV | Isolate persistent bias |
| F | yes | best **validated** likelihood from C/E | valid covariance for the prototype | 6D fixed-lag (opt-in) | Compare motion architecture |

Additional initial velocity SD ablations may be included **without promoting new defaults** and must be labeled as sensitivity, not fixes.

**Validation: two compulsory environments:**

1. **Real red_fridge**: a separately verified static subwindow plus the initial real motion window. Track 100–105 first, not 100–288. Only after first-step stability is established run a longer segment. Report per-Gaussian displacement p50/p90/p99/max; directional/ray-wise drift; fraction out of view; mask IoU; actual inlier depth residual distributions; image energies; predicted versus posterior position/velocity uncertainty; and solver convergence.
2. **Synthetic SAPIEN articulated** with **GT only in evaluation**: static-point drift, moving-point 3D trajectory error, velocity calibration, motion support precision/recall, and unchanged Phase 2/3 joint type, axis direction/angular error and axis position error. Include an artificial persistent render/camera mismatch as a controlled stress test.

**Posterior validity:** compare standardized errors and coverage to reported covariance; state the exact effective likelihood and conditioning. Do not apply a chi-square NIS/NEES interpretation to correlated Student-t/robust measurements without verifying the necessary assumptions. For empirical covariance, run independent noise/scene simulations and report whether uncertainty intervals cover GT at their stated levels.

**Missing information:** when a Gaussian is not observed, log predicted-only status. Do not call prediction ground truth and do not use arbitrary speed clipping to keep an invisible point in the object. Later confident Phase-3 joint constraints can recover missed movement **in a distinct future phase**, not here.

## 9. Explicit acceptance criteria and stop rules

### Gate A–B correctness

- [ ] First 100–105 audit is reproducible from exact model/data versions and identifies prior/posterior camera/channel forces.
- [ ] Audit Fisher includes alpha exactly as the production operator does; gradients are compared at a common linearization state.
- [ ] Supernode partition hash remains fixed for each direction across the frame window; fixed from the correct canonical endpoint.
- [ ] Existing first three phases' regression tests continue to pass; synthetic moving points are not frozen.

### Gate C–E statistical correctness

- [ ] Report distinctly measures sensor SD versus raw-render mismatch versus correlated/persistent bias.
- [ ] Alpha/mask confidence is separately tested, not silently derived as an independent calibrated sensor measurement.
- [ ] Tiny dense covariance oracle quantifies the current diagonal Fisher error; any new scalable covariance implementation matches its marginal/spectrum within justified numerical/Monte Carlo uncertainty.
- [ ] Lie-group covariance tangent transport is derived/tested and only applied where needed.
- [ ] Nonconverged local Hessian is clearly labeled as an **uncertified uncertainty approximation**.
- [ ] No sigma, Q or motion-prior constant is changed based only on a failed real trajectory; no implicit static labels or hard limits.

### Gate F dynamics and real/synthetic evidence

- [ ] Demonstrate with numerical counterfactual that initial false pose displacement maps into velocity, and that corrected likelihood/initialization changes the measured failure mode.
- [ ] Verified-static drift and divergent tails are reduced **relative to the unchanged A/B baseline** without losing true synthetic moving-part displacement; report actual magnitudes, not only an asserted pass.
- [ ] Convergence status is explicit; frame-budget exhaustion cannot be presented as successful optimization.
- [ ] No full dataset batch or Phase-2/3 redesign starts before the two small pilot evaluations pass.
- [ ] Optional pose-only estimator remains off by default and compared scientifically before adopting.

### Deliverables

1. Small reviewable commits, separated by `audit`, `fixed-partition`, `likelihood`, `covariance-oracle`, and any `reference-relative` or `fixed-lag` **experimental** branch. Keep experiments gated off by default.
2. New/extended CPU correctness tests and focused CUDA renderer-oracle tests, with real reproduction commands. Clearly record `PASS`, `FAIL`, `NOT RUN` for each.
3. `docs/PHASE1_COVARIANCE_VELOCITY_ROOT_CAUSE_AND_FIXES.md`, including code SHA, sensor ID, exact data paths/frames, source provenance, before/after quantitative comparisons and failures still open.
4. A concise recommendation of which ablation, if any, should replace default Phase 1, and why its **uncertainty and real motion accuracy** are defensible. Avoid blanket claims like “fixed divergence” on a 3-frame smoke test.

## 10. Pseudocode targets and surgical code map

```python
# tools/kpart_pipeline.py::bootstrap_positions
# Construct once, before the time loop, from the canonical endpoint.
bcfg = dataclasses.replace(cfg, use_supernodes=True, ...)
fixed_partition = build_supernodes(
    truth.mu,
    target_size=bcfg.supernode_target_size,
    num_supernodes=bcfg.supernode_count,
    knn=bcfg.supernode_knn,
    seed=bcfg.supernode_seed,
)
# Persist fixed_partition.canonical_hash(); DO NOT rebuild per frame.
for ti, (obs, fid) in enumerate(frames):
    if ti == 0:
        continue
    new_state, P_post, diag = iekf_frame_step(
        seed, P_previous, prepared_obs, cameras,
        dt=measured_dt[ti], cfg=bcfg,
        partition=fixed_partition,
        observation_model=existing_mask_model,
    )
```

**Other exact code locations at baseline `7954660`:**

| Concern | Existing implementation | Intended scoped action |
|---|---|---|
| Dynamic group rebuild | `estimator/sparse_iekf.py` ~577–583; `tools/kpart_pipeline.py` ~1765, 1845 | Supply one canonical group partition through the entire pass |
| Speed from position uncertainty | `tools/kpart_pipeline.py` ~1778–1800; `estimator/process.py` ~113–145 | Audit 12D coupling and infer initial speed from time, no manual damping |
| Post-optimization Fisher | `estimator/sparse_iekf.py` ~991–1045; `estimator/covariance.py` ~383–410 | Validate full marginal/cross terms and lie reset, report nonconvergence |
| RGB/depth weighting | `estimator/observation_terms.py` ~110–147 | Separate sensor, model and bias, inspect real residuals |
| Mask alpha weight | `estimator/masked_observation.py` ~75–83 | Evaluate independent silhouette model without violating hard object mask |
| Audit Fisher omission | `tools/audit_calibrated_phase1.py::Phase1Audit.__call__` | Add per-camera alpha Fisher to diagnostic, not production solver |
| Image-space transport | `estimator/renderer_bridge.py` and a **new experimental factor** | Visibility-aware reference factor; exact full-render acceptance |
| Covariance / grouping oracle | `estimator/supernodes.py`, `estimator/covariance.py`, tests | Tiny dense oracle; scalable probes only once validated |

### Command examples (adapt actual paths and CLI; do not fabricate results)

```bash
# Identify checkout and changes.
git rev-parse HEAD
git status --short
git diff 7954660 --stat

# Baseline audit: same real canonical model at frame 100, 6 frames only.
# Use the existing successfully recorded model/sensor paths.
python tools/audit_calibrated_phase1.py \
  --model "$ENDPOINT_INITIAL" --source "$RED_FRIDGE_EP000000" \
  --motion-start-frame 100 --motion-end-frame 105 \
  --frame-timestamps auto --sensor-model "$SENSOR_MODEL" \
  --p0-geometry "$ENDPOINT_INITIAL/geometry_information.npz" \
  --out "$SCRATCH/audit_baseline_100_105.json"

# Existing unit/regression tests (inspect their current invocation first).
python tests/test_surgical_review_4c88725.py
python tests/test_calibrated_bootstrap.py
python tests/test_bootstrap_geometry_eval.py  # 2 known FAILS must remain visible
```

**Final instruction to Codex:** Diagnose first, fix confirmed defects surgically, then perform controlled experiments. **Do not treat a numerical stabilizer as a physically valid solution.** The required result is an estimator whose real stationary evidence does not become fictitious speed, whose covariance agrees with the declared model and actual estimation error, and whose genuinely moving Gaussians remain recoverable for the unchanged Phase 2/3 joint pipeline.
