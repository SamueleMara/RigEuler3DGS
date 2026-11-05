# RigEuler3DGS — Calibrated RGB-D → Surface-Centered Bootstrap → Time-Consistent IEKF

**Status:** implementation specification, not implemented. Preserve existing Phase 1 (free Gaussian IEKF), support extraction, Phase 2 (shared rigid SE(3) and revolute/prismatic fit), and Phase 3 (joint displacement IEKF). No GT joint/type/part-label data may enter inference. Baseline checkpoint and executable outputs must be preserved.

**Verified against:** `SamueleMara/RigEuler3DGS` main commit `481b2cf2d5602932286948cf14114f9caee4de3c`, inspected 2026-11-04. Reusable noise-statistics reference: `SamueleMara/RGBDiekf/scene/noise_calibration.py` main commit `02d2ea7ca027a4d34145ab7be37cf74eacbdb4ad`. Exact implementation must be tested against working dependencies and actual datasets.

## 0. Scope and nonnegotiable invariants

* Input motion interval `[motion_start_frame, motion_end_frame]` is **inclusive original dataset frame IDs**. It defines RGB-D observations on which motion must be estimated. It is **not GT motion**; optional synthetic GT remains evaluation-only. Forward canonical endpoint is `motion_start_frame`, backward canonical endpoint is `motion_end_frame`.
* Real sensor calibration is optional, flag-controlled, and requires **both** inclusive `calibration_start_frame` and `calibration_end_frame` with at least two samples; synthetic never attempts calibration and uses a separate `ideal` physical sensor policy.
* Calibration interval estimates **sensor statistics**, not geometry to merge unconditionally into the motion-start frame. No appearance or depth observations from a different articulation state are silently fused into the endpoint model.
* All three IEKF phases share one frozen, provenance-tracked sensor measurement model. Never reinterpret sensor variability, cross-camera mismatch, Gaussian spatial extent, geometric-centre covariance, process noise, and renderer error as the same variance.
* Stable IDs and Gaussian count throughout Phase 1. No densification/pruning in bootstrap or during motion. Source camera/pixel provenance can be saved but must not restrict later visibility to those cameras/pixels.
* Preserve the existing Lie-group retractions, analytic renderer H/H^T, masked foreground constraints, true nonlinear acceptance objective, shared rigid Phase 2, both P/R hypotheses, and fixed-joint Phase 3.
* No camera synchronization term at first implementation: hardware synchronization is reported sufficiently precise. Treat extrinsic/intrinsic *calibration* uncertainty separately.
* No GT part labels, GT motion, or GT joint axes in training, bootstrap selection, stopping or model selection.

## 1. CLI and manifest contract

Propose a new orchestration entrypoint `tools/run_noise_aware_rigeuler.py` (or additive extension to existing batch runner):

```bash
# Real — examples; these flags do NOT exist before implementation.
python tools/run_noise_aware_rigeuler.py \
  --source DATASET --dataset-kind real \
  --rgbd-noise-calibration \
  --calibration-start-frame 0 --calibration-end-frame 39 \
  --motion-start-frame 40 --motion-end-frame 99 \
  --bootstrap-mode covariance_surface --sensor-noise-model calibrated \
  --out RESULTS

# Synthetic — no calibration interval, no physical sensor noise.
python tools/run_noise_aware_rigeuler.py \
  --source DATASET --dataset-kind synthetic \
  --motion-start-frame 0 --motion-end-frame 59 \
  --bootstrap-mode covariance_surface --sensor-noise-model ideal \
  --out RESULTS
```

Run schema validation **before** CUDA: valid IDs, monotone physical timestamps, known camera IDs, complete depth/RGB/mask data for selected endpoint frames, positive calibration interval size, no calibration flags for synthetic, no interval without calibration flag, no silent fallback from calibrated to default sensor noise. Allow nonadjacent intervals and never infer calibration by assuming motion-frame 0 is stationary. Store hashes of exact input files, camera calibration, source commit, config and selected indices. Ensure artifacts carry original frame indices and physical units.

Output schema:

```
RESULTS/
  run_manifest.json
  calibration/sensor_model.json; calibration/covariance.npz; calibration/validation.json
  bootstrap/initial/point_cloud.ply; bootstrap/initial/geometry_information.npz;
  bootstrap/initial/source_responsibility.npz; bootstrap/initial/manifest.json
  bootstrap/final/...   # independently initialized on last motion frame
  phase1_forward/...; phase1_backward/...
  phase2/...; phase3/...
```

Include explicit policy `sensor=calibrated | ideal | assumed` in all downstream manifests. A real scene with calibration disabled may run only with a declared `assumed` sensor model (report status UNCALIBRATED).

## 2. A — Static-frame per-camera calibration

### 2.1 Measurement and identifiability model

For valid pixel `p=(u,v)` and static time `t` in the user-supplied interval:

`D_c,t,p = d*_c,p + b_c,p + e_c,t,p`, where `d*` is unknown fixed true scene depth, `b` persistent spatial/sensor bias, and `e` zero-mean temporally fluctuating noise. Camera/scene static-frame data identify the distribution and time correlations of `e`; **they do not separately identify absolute b and true depth**. Cross-view overlap can diagnose relative camera/geometry disagreement but is a distinct calibration stage.

Estimate per-camera `sigma_D,c^2(z[,u,v])`, RGB channel covariance (or independent channel approximation with measured off-diagonal diagnostics), missing-return frequencies, temporally correlated residuals, and residual contamination near discontinuities. Use raw finite `depth>0` and the exact foreground masks. Do not use background measurements for object noise statistics unless the user explicitly supplies a separate sensor-calibration mask/policy.

At a fixed pixel, estimate robust temporal location `median_t D`, residuals `e~=D-median D`, autocovariance `gamma_c(k)=Cov(e_t,e_t+k)`. For stationary independent noise, the MAD of consecutive differences gives `sigma^2≈(1.4826 MAD(ΔD))²/2`; in general `Var(ΔD)=2(sigma²−gamma(1))`, so naive `/2` underestimates autocorrelated noise. Use robust per-pixel residuals and pool by range with uncertainty intervals; frame-to-frame global fluctuations should be reported separately. Only fit heteroscedastic models supported by available depth ranges. Candidate `sigma_D^2(z)=a+bz²` or `a+bz²+dz⁴`, constrained nonnegative, selected on held-out stationary samples. Quantization is a measured/instrument floor, not an invented uncertainty.

For depth-to-world 3D covariance, with `r=K_c^{-1}[u,v,1]^T`, `X_wc=R_wc (z r)+t_wc`: `C_X = R_wc r sigma_D,c²(z) r^T R_wc^T + J_uv Sigma_uv J_uv^T` if pixel location uncertainty is explicitly modeled. Intrinsic/extrinsic parameter uncertainty `P_kappa` is **separate** and generally shared across many pixels.

### 2.2 Code and artifacts

New `utils/noise/sensor_model.py`: dataclasses `CameraNoise` / `SceneNoiseModel` with `depth_variance(camera_id,z,u,v,context)`, `rgb_covariance(camera_id,...)`, `temporal_model(camera_id)`, `to_json/from_json` and provenance. New `tools/calibrate_static_rgbd.py`: use existing loader and borrow robust estimation ideas from `RGBDiekf/scene/noise_calibration.py` without importing its unrelated pipeline internals. Never pool different camera models by default. Save diagnostic per-camera depth-binned variance, lag autocorrelation confidence, RGB covariance, invalid fractions and split-half/generalization reports.

Tests: correct interval selection; stationary Gaussian simulator with known variance and AR(1) recovers parameters; corrupted pixels/holes; different noise levels in 3 cameras; inability to infer absolute bias is reported; synthetic mode creates no physical sensor-noise artifact.

## 3. B — Fast, surface-centred endpoint 3DGS at `motion_start_frame`

### 3.1 Observation source

Only **one synchronized multiview timestamp** (the first motion frame) supplies the forward geometry/appearance; last motion frame supplies an independent backward model. Use every available calibrated camera and strictly masked valid depths. DO NOT blend actual geometry from the preceding stationary calibration interval unless explicitly validated to be the same physical configuration. Cross-camera geometry must be estimated as one common surface rather than independent fused depth clouds.

### 3.2 Geometry seeds and GICP-like surface consistency

For each valid sample j from camera c, compute 3D point `y_j`, covariance `C_j^sensor`; keep `(camera,frame,pixel,depth,rgb,mask,source_id)`. Infer local compatible surface patches without assuming same pixel location in different cameras is same material point. Use probabilistic soft associations, visibility and occlusion ordering, and local surface geometry; avoid merging front/back layers or crossing hard mask edges by interpolation. A convenient point-to-tangent-plane residual is `r_ij=n_i^T(y_j-mu_i)` with conditional variance `n_i^T C_j^sensor n_i + sigma_surface,i²`, and robust normalized NLL. The surface spread term is a *geometric approximation*, not sensor noise; prevent unconstrained covariance inflation through normalized log determinant and shape constraints. Direct full 3D Gaussian NLL `0.5 r^T(C+Sigma_surface)^-1 r + 0.5 logdet(C+Sigma_surface)` is an alternative, but must define correctly normalized correspondences and avoid degeneracy.

Set `mu_i` to recovered patch locations, `R_i` from fitted surface tangent directions, `s_i` tangential footprints from measured coverage, normal-axis thickness from physical reconstruction support, and `color_i` robustly from contributing observations. `Sigma_i^render=R_i diag(s_i²) R_i^T` is rendering footprint; it is not `P_i^mu`, the uncertainty of the estimated centre. Initialize opacity consistently with actual coverage. Freeze a compact population; choose its density from visible surface coverage and runtime constraints, not GT labels.

### 3.3 Pixel-to-Gaussian responsibility/provenance

At endpoint frame with actual Gaussian alpha order, export sparse positive compositing contributions `w_i,c,p = T_before(i,c,p) * alpha_i,c,p` together with camera/pixel IDs, the source depth/colour, visibility, residual, geometry covariance. Store `[gaussian_id,camera_id,flat_pixel_id,weight]` COO/CSR, normalized *for association diagnostics* where appropriate. Distinguish this from the current renderer's JVP/VJP cache; identities are persistent but observed pixels vary over time. If no public rasterizer API exposes each contributor's weight, implement an output-only extraction in the existing compositing traversal and verify the recovered RGB/alpha/depth sums against ordinary rendering (do not approximate with nearest projected centre only).

Use a per-Gaussian source-ownership **diagnostic** and matched camera influences to identify which cameras push correct geometry incorrectly. Do not gate away all other cameras based solely on source camera.

### 3.4 Short refinement

New additive `tools/train_tracking_bootstrap.py` or `--tracking-bootstrap` mode inside `train.py` reusing the same renderer and mask/carving policy. Disable densification/pruning, opacity reset, SH-degree promotion and expensive semantics. Use SH0 / DC only, optimize Gaussian means, quats, scales, opacity, DC colour. Use physically consistent metric depth or inverse-depth variance via `Var(1/z)≈Var(z)/z⁴` in `_static_visual_loss`; choose a **normalized heteroscedastic robust depth likelihood** using the per-camera model. Retain masked RGB and alpha/silhouette coverage supervision without background depth contamination. Add a surface-thickness/geometric fidelity regularizer with independently declared strength (NOT claimed sensor precision). If GICP geometry fitting and rendered depth use the same depth samples, treat them as sequential fitting stages or as a non-probabilistic composite; never count them as independent measurements twice.

Checkpoints at 0/250/500/1000/2000 updates are initial benchmark points, not a hardcoded universal stopping rule. Compare held-out pixels and motion-stage performance. Do not select checkpoint based on GT motion errors.

### 3.5 Initial geometry covariance

Estimate `P_i^(mu,phi)` in canonical motion coordinates using local curvature (Hessian/Fisher) of the fitted objective plus appropriate calibration uncertainty. The marginal `P_i` generally differs from inverse diagonal Hessian if cameras/Gaussians are coupled; document and check approximations. Use a positive semidefinite approximation, with large variance in unobservable directions; no fake high certainty for flat textureless surfaces. Export `P0_geometry[N,6,6]`, plus confidence/observability and camera residual provenance. Keep `Sigma_render` separate.

Files to modify: `scene/gaussian_model.py::create_from_pcd` to accept optional means, orientations, scales and metadata without changing the original call path; `train.py::_static_visual_loss` add selectable calibrated depth likelihood and prevent SH escalation and population modifications in bootstrap; new sensor file reader and source-responsibility extractor.

Tests: point-to-plane covariance finite differences; planar surface with three known differently noisy cameras reconstructs closer to GT than simple point averaging; skew and multi-view bias fixture; sharp edge not averaged into background; exactly identical Gaussian ID/count through training; verify source responsibility sums against renderer; zero-noise synthetic deterministic model; data-loading original frame labels correct.

## 4. C — Time-consistent Phase-1 IEKF without replacing RigEuler Phase 1–3

### 4.1 Existing code corrections before new filtering

* `tools/run_motion_coherence_support.py`: internal Namespace currently `sequence_start=0, dt=1/30`, constructs `cfg=IEKFConfig()` ignoring calibration. Add motion start, end and measured time deltas; make backwards canonical anchor actual last motion frame. Similarly change `tools/run_rigid_se3_pipeline.py` and `tools/run_joint_phase3.py` (they also build fresh default configs and fixed frame conventions).
* `tools/kpart_pipeline.py::bootstrap_positions`: constructs `bcfg=IEKFConfig(...)` by copying model kind but **not** `depth_sigma_a` and `depth_sigma_b`; propagate all exact calibrated parameters (prefer typed sensor model rather than old global coefficients). Existing `P0[:,diag]=[1e-2]*3+[1e-2]*3+[1.0]*3+[1e-1]*3` must be replaced in opt-in mode: pose `P0` comes from bootstrap covariance; velocity process prior is declared separately, never inferred from rendering Gaussian scale.
* `estimator/observation_terms.py::depth_sensor_sigma` and `camera_terms`: accept `camera_id` and query calibrated per-camera depth/RGB noise at observed depth; all Phase 1/2/3 call paths pass their actual camera identity rather than local subset index. `estimator/masked_observation.py` and `estimator/phase2_shared.py` must use the **same** objective/precision and invalid-depth semantics.
* `estimator/model_error.py::estimate_model_error` currently squares a per-pixel reference residual and treats it as a variance. Retain for baseline ablation; do **not** call that sensor noise. Diagnose persistent signed bias and geometric-model mismatch separately. When moving geometry shifts, an original fixed per-pixel mismatch map does not automatically follow that material surface. Do not silently reweight motion away.
* Retain `estimator/sparse_iekf.py` nonlinear trust-region IEKF, analytic operator, existing `P^- = F P^+ F^T + Q`, PSD/observability repair, and cross-Gaussian coupling within the matrix-free frame solve.

### 4.2 Correct time covariance semantics

Three independent quantities: measurement noise `R_t`, physical state evolution noise `Q(dt)`, predicted state covariance `P_t^-`. `S_t = H_t P_t^- H_t^T + R_t`. Never add the predicted state uncertainty into `R_t` again. Use actual physical `dt` for forward/reverse runs; check sign/rate conventions and elapsed seconds. Noise `Q` is motion-model acceleration uncertainty, **not sensor calibration output**.

For real depth, a first improvement is `R_sensor,c(z)` from calibration; use the same model at every motion frame, evaluated at each frame's depths and visibility context. Add robust mixture/Student-t for real mixed-depth returns, and encode invalid returns with zero information. A model that increases `R_t` simply because `|Dobs-Drender|` becomes large must be refused: it can hide wrong motion.

Temporal autocorrelation `rho(k)` from stationary frames is a diagnostic, NOT automatically an AR(1) model of surface residuals during motion. Static per-pixel readout may be temporally correlated, but the physical surface seen by a pixel changes under motion; validate the origin of the correlation on moving held-out observations. If justified, implement a small **per-camera latent shared bias** with AR(1) time dynamics and low-rank observation operator (or equivalent exact conditional whitening, using past-state uncertainty), not independent per-pixel AR(1) everywhere. For residual process `e_t=rho e_(t-1)+w_t`, stationary white innovation variance is `(1-rho²)sigma²`. Naive whitening `r_t-rho r_(t-1)` is incorrect if the earlier rendered-pose error and cross-time covariance are ignored. Use a synthetic correlated-noise oracle to test NIS and the posterior.

Camera extrinsic/intrinsic uncertainty induces shared pixel correlations. First report cross-view mismatch and fit common geometry. If nuisance covariances are measured/available, an optional later low-rank matrix-free precision operator can apply `W=(R + B P_b B^T)^-1` via Woodbury. This cannot be wired into current per-pixel `(precision_RGB,precision_D)` arrays without changing the normal operator, `H^T W H`, Fisher approximation and objective together. Leave default independent-diagonal model intact for first validated ablation; do not pretend it supports exact shared-camera covariance.

Source atlas tells which views/pixels contributed to initial geometry; recompute visibility/projections every frame. Preserve current masked observation constraints and actual foreground depth.

### 4.3 Expected outputs

* `sensor_model_id` propagated in every stage manifest; exact parameter/unit equality asserted Phase 0/1/2/3.
* Reference biases separately reported from random sensor variance and geometry error.
* Per-frame per-camera residual histograms, standardized innovation statistics (NIS with effective DOF note), depth coverage, camera information contribution and rejection reasons.
* Full Phase-1 `mu,R,v,omega` and covariance/information approximate marginals in **original chronological indexing**; `P0` provenance; processing-order manifest preserved for backward.
* Explicit warning when per-Gaussian block posterior ignores inter-Gaussian and shared-camera covariance. No claim of exact joint uncertainty.

## 5. Regression gates and rollout

1. **Gate A — calibration:** real stationary synthetic oracle, per-camera depth residuals and AR checks, held-out calibration frames, sensor model versioning and correct data-range isolation.
2. **Gate B — fast bootstrap:** fixed Gaussian indices/count, held-out geometry, source responsibility compositing sum, calibrated depth residuals and no background contamination. Compare raw seed, current standard 3DGS, covariance-initialized no-refinement, covariance-initialized short-refinement.
3. **Gate C — Phase 1 unchanged core:** same exact dataset, per-frame residual+Fisher, no static drift, moving part not collapsed to zero, real/synthetic. Independently report camera attribution, initial `P0`, and maximum numerical deviations.
4. **Gate D — Phase 2/3 regression:** on prismatic/revolute synthetic scenes (and stationary real control) verify same or better joint type, pose, axis and displacement, with GT **only after final output**. No threshold tuned on these scene labels.
5. **Gate E — sanity of time covariance:** tests with zero correlation, known AR1 camera bias and independent noise; posterior covariance calibrated, no double counting, no invented motion, correct physical `dt` and reverse-time processing.
6. **Gate F — opt-in production:** old flags reproduce original outputs; new flags fully connected (fail if calibration coefficients lost or artifact mismatched); no silent fallback.

Make changes in this strict sequence: `A sensor-only → B initialization-only → B calibrated loss → C P0 only → C sensor likelihood Phase 1/2/3 → C temporal correlation optional`. Do not modify joint-discrimination or Phase 3 model math while testing the sensor/geometry upgrades. Post-joint geometry birth and FreeArtGS photometric reconstruction are separate subsequent milestones, not part of this implementation.

## 6. Source-code inspection anchors

* `train.py::_static_visual_loss` (around 473): current masked inverse-depth L1, no calibrated per-camera variance; `train_static` (582): camera sampling, SH-degree promotion, densification and opacity reset.
* `scene/gaussian_model.py::create_from_pcd` (595): nearest-neighbour isotropic initial scale; learnable quaternions.
* `estimator/measurement.py::DepthSigmaModel` (62): constant/linear/quadratic, global coefficients; Student-t + normalized innovation diagnostics.
* `estimator/observation_terms.py::camera_terms` (73): diagonal per-pixel sigma and optional model-error variance.
* `estimator/model_error.py::estimate_model_error` (79): squared reference residual as fixed pixel-indexed map.
* `tools/run_motion_coherence_support.py` (80, 257): mandatory existing motion pipeline wrapper with default `IEKFConfig`, frame 0 and `dt=1/30`; full-state observer optional.
* `tools/kpart_pipeline.py::bootstrap_positions` (1723): `P0` hardcoded, recursive covariance, subset-copy of config that loses `depth_sigma_a/b`.
* `estimator/sparse_iekf.py::iekf_frame_step` (424): existing free IEKF with analytic H/H^T, grouped update; `approximate_posterior_information` in `estimator/covariance.py` intentionally drops inter-Gaussian posterior cross terms.
* `estimator/renderer_bridge.py::FrameLinearization` (67): JVP, VJP, per-Gaussian Fisher, alpha operator, all camera observations fused per frame.
* `tools/run_rigid_se3_pipeline.py` and `tools/run_joint_phase3.py`: independent default configuration paths; Phase 2/3 model-error reference and timing must share sensor profile from calibration.

This is a planned design, not evidence that the covariance-aware bootstrap or time-correlated IEKF improves results until the listed ablations pass.
