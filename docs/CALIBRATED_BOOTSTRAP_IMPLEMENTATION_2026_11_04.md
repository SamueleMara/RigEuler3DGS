# Calibrated RGB-D → surface-centred bootstrap → time-consistent IEKF (2026-11-04)

Specification: `RigEuler3DGS_Calibrated_Bootstrap_IEKF_Implementation_Spec.md`. Branch `calibrated-bootstrap`, from `7977987`.

**Scope.** This document reports what was implemented and how it was verified: unit tests and end-to-end smoke runs on
4-frame (real) and 3-frame (synthetic) windows. **No full-scale run has been made.** Every claim about improved
tracking stays open until the ablations of spec §5 are run (§6 below).

## 1. What exists now

### A — sensor-only calibration

| file | content |
|---|---|
| `utils/noise/sensor_model.py` | `CameraNoise` and `SceneNoiseModel`, with the explicit policy `calibrated` / `ideal` / `assumed`. Per-camera queries `depth_variance(camera, z)`, `rgb_covariance(camera)` and `temporal_model(camera)` (diagnostic). `model_id` is the sha256 of the numerical content, and `from_json` refuses a tampered file. Cameras are never pooled or substituted. The `identifiability` statement is stored in the model. |
| `tools/calibrate_static_rgbd.py` | Per camera, on pixels inside the object mask in every calibration frame, using only raw finite depth > 0 (spec §2.1). |

How `tools/calibrate_static_rgbd.py` estimates:
- **Location:** the per-pixel temporal median.
- **Common mode:** each frame's common-mode offset is removed and reported separately (`global_offset_std_m`). It is a
  camera-shared correlated term that the diagonal model does not represent.
- **Variance:** an outlier-trimmed sample variance pooled per depth bin, with a T/(T−1) correction and a pixel-bootstrap
  95% CI.
- **Why not MAD:** the MAD scale collapses on quantized data. The real red_fridge depth is registered onto a per-pixel
  ~1 mm grid, and RGB is 8-bit, so more than half of all residuals are exactly 0. The trimming bound is therefore
  max(MAD, step), and the observed variance (which includes quantization, per Sheppard) is what the filter sees. The
  measured step q, taken per pixel from each pixel's own time series, gives a floor of q²/12.
- **Cross-check:** the corrected Var(ΔD)/(2(1−ρ₁)) is reported next to the naive Var(ΔD)/2, which under-estimates
  correlated noise.
- **Depth model form:** a, a+bz² or a+bz²+dz⁴, non-negative. Each supported form is fitted on the EVEN frames and the
  form is selected by the Gaussian NLL of the ODD frames. Forms the depth range cannot support are not fitted.
- **Also measured:**
  - lag-1…5 autocorrelation with CI (a diagnostic, never an AR(1) model);
  - the RGB 3×3 covariance with correlations;
  - the in-mask invalid-return fraction;
  - split-half consistency.

It writes `calibration/sensor_model.json`, `covariance.npz` and `validation.json`.

**red_fridge episode 0, frames 0–99:**

| camera | σ_D (constant model selected) | RGB σ | invalid inside mask | lag-1…5 correlation |
|---|---|---|---|---|
| cam_000 | 0.71 mm | 0.004–0.005 | 7% | ≈ 0.3, flat (not AR(1)-like) |
| cam_001 | 0.94 mm | 0.005–0.0065 | 29% | ≈ 0.3, flat |
| cam_002 | 0.87 mm | 0.004–0.005 | 9% | ≈ 0.3, flat |

The depth step is q = 1 mm in every camera. Split-half variance ratios are 0.8–1.25.

### B — surface-centred endpoint bootstrap

| file | content |
|---|---|
| `estimator/surface_bootstrap.py` | The bootstrap's computations, listed below. |
| `tools/train_tracking_bootstrap.py` | Builds a fixed-count model from ONE frame, listed below. |
| `scene/__init__.py` | The only change to shared code. `if load_iteration` / `if self.loaded_iter` became explicit `is None` tests, so iteration 0 (the unrefined bootstrap) is loaded instead of being treated as "train from scratch". Every existing caller loads None or an iteration ≥ 1, so their behaviour is unchanged. |

`estimator/surface_bootstrap.py` contains:
- **Back-projection** with a rank-1 sensor covariance along each ray, C = σ_D²(z) d dᵀ.
- **Seeds** from a voxel grid whose size is searched to reach the requested population; the density comes from
  coverage.
- **A robust point-to-tangent-plane IRLS** over the k nearest samples from all cameras (one common surface):
  - variance nᵀCn + σ_surface² = σ_D²(n·d)² + σ_surface²;
  - Student-t weights, so another layer gets almost no weight;
  - σ_surface² capped by the tangential footprint (the shape constraint).
- **Gaussian parameters:** R = [t1 t2 n], tangential scales from the local seed spacing, normal scale from the fitted
  thickness, colour from the weighted samples, opacity from the inlier fraction.
- **Source responsibility:** w = T·α per (Gaussian, camera, pixel), taken from the production renderer's own
  contributor walk (`observation_diagnostics.renderer_access.pixel_contributions`) and checked against the rendered
  alpha. Per-Gaussian camera **ownership** is a diagnostic, never a gate.
- **Geometry covariance:** P0 = (F + Λ_prior)⁻¹ per Gaussian (pose 6×6). F is the renderer Fisher under the shared
  sensor model; Λ_prior is a declared weak prior, so unobservable directions keep the prior variance.

`tools/train_tracking_bootstrap.py` builds the model from ONE frame (`--frame`) as follows:
- **Output format:** a 3DGS model directory every loader in the repo reads, with DC colour only (SH rest = 0).
- **Refinement:** Adam on means, quaternions, scales, opacity and DC. There is no densification, pruning, opacity reset
  or SH promotion, and the Gaussian count is asserted unchanged at every step.
- **Loss terms:**
  - masked RGB with the calibrated per-camera W;
  - silhouette (A − M)²;
  - a Student-t NLL of inverse depth with Var(1/z) = Var(z)/z⁴, on V ∧ M ∧ A > 0.5;
  - a declared thickness regularizer.
- **Held-out pixels:** 10% of masked pixels are held out and scored at checkpoints 0/250/500/1000/2000. These are
  benchmark points, not a stopping rule, and nothing selects on GT.
- **Outputs:**
  - `bootstrap_manifest.json` (endpoint, frame, sensor model id and policy, held-out scores, checks);
  - `geometry_information.npz` (P0_geometry, Fisher, eigenvalues, seed-fit normal variances, and `Sigma_render` kept
    separate from P0);
  - `source_responsibility.npz` (COO entries plus ownership).

**Verified on red_fridge (frame 110):**
- The back-projected seeds lie a median 1.33 mm (p90 2.6 mm) from the dataset's own fused frame-0 cloud.
- The training rasterizer's `depth` divided by alpha equals 1/z exactly (median ratio 1.0000), so the inverse-depth
  likelihood uses the right quantity.
- The responsibility sums match the rendered alpha to 4.8e-7.
- Every Gaussian is seen by at least one camera.

### C — one sensor model through Phase 1/2/3, physical time, the motion interval

| file | change |
|---|---|
| `estimator/sparse_iekf.py: IEKFConfig` | `sensor_model`, `camera_names`, and the declared `rgb_model_sigma` / `depth_model_sigma` (default 0; renderer/appearance error, never a sensor statistic). |
| `estimator/observation_terms.py: camera_terms` | Takes `camera=`. With a CALIBRATED model, the camera's own per-channel RGB variance and its depth variance at the observed depth replace `rgb_sigma` and the `depth_sigma_*` terms; declared mismatch is added in variance. A calibrated model without a camera identity is an error, never a fallback. With no model (ideal/assumed), the terms are byte-identical to before. |
| `estimator/masked_observation.py`, `estimator/phase2_shared.py`, `estimator/sparse_iekf.py: _frame_terms` | Pass the actual camera identity (`camera_name(cfg, index)`). The masked mode uses the calibrated per-camera W (still no RGB reweighting); κ = trace(W)/12 follows it. |
| `tools/kpart_pipeline.py: bootstrap_positions` | The config is now copied by `dataclasses.replace`: the old subset copy dropped `depth_sigma_a/b`, and it would have dropped the sensor model. Opt-in `--p0-geometry`: the pose block comes from the bootstrap and the velocity block from a declared `--p0-velocity-sigma`. The per-step dt comes from the capture timestamps. `P0_source` and `dt_s_per_step` are recorded in `phase1_frames.json`. |
| `tools/run_iekf_scene.py: dataset_rgbd_frames` | `--frame-timestamps auto` reads `metadata/frame_timestamps.csv`, requires strictly increasing values, records `timestamp_s` and `source_frame_index` per frame, and sets the per-step interval. |
| `utils/sequence_direction.py` | `processing_time_s` uses the recorded timestamps, so elapsed time is positive in both directions and exact for non-uniform sampling. `verify_endpoint_provenance` trusts a bootstrap model's `bootstrap_manifest.json`. |
| `estimator/run_config.py` (new) | One CLI and configuration contract for the three runners: `--motion-start-frame/--motion-end-frame`, `--frame-timestamps`, `--sensor-model` / `--sensor-policy`, the declared mismatch sds, and (Phase 1) P0. It also writes `sensor_manifest.json` per stage (policy, `sensor_model_id`, cameras, original frame indices, timestamps, covariance note) and runs `check_canonical_endpoint`, which refuses a model that is not the canonical frame of the requested window. |
| the 3 runners | `sequence_start` / `sequence_length` come from the window. The config is rebuilt with the shared sensor model and camera identities after loading; manifests are written. |
| `tools/check_masks.py`, `masked_observation.preflight` | Check the masks of the motion window, not of frames 0…N. |
| `tools/stage_signature.py` | Content hashes cover the window's files (`--frame-start`). |
| `run_bidirectional_all.sh` | New pass-through flags (`--motion-start-frame`, `--motion-end-frame`, `--frame-timestamps`, `--sensor-model`, `--sensor-policy`, `--rgb/depth-model-sigma`, `--bootstrap-p0`, `--p0-velocity-sigma`). Window-aware frame and mask checks. The stage signatures include the interval, the timestamp mode, the sensor-model sha, the declared sigmas and the P0 sha. |

### Orchestrator — `tools/run_noise_aware_rigeuler.py`

Spec §1 CLI. Schema validation runs before any CUDA work. It refuses:
- unknown or out-of-range frame IDs;
- non-contiguous frames;
- non-increasing timestamps;
- incomplete endpoint or calibration frames;
- missing masks over the window;
- calibration flags on synthetic data, or an interval without `--rgbd-noise-calibration` (and the reverse);
- a calibration interval that overlaps the motion window;
- a dataset kind, calibration choice and `--sensor-noise-model` that don't match (real without calibration ⇒ `assumed`,
  UNCALIBRATED).

It then runs, in order:
1. calibration, into `RESULTS/calibration/`;
2. two INDEPENDENT bootstraps, at the motion start and at the motion end, into `RESULTS/bootstrap/{initial,final}/`;
3. the driver on `RESULTS/pipeline/<scene>/`, with the bootstrap models as endpoints.

`run_manifest.json` records:
- the policy and status;
- the intervals and canonical frames;
- the validation facts;
- the config;
- the code commit and dirty flag;
- sha256 of the camera calibration and timestamp files;
- each stage's status.

## 2. Not implemented (and why)

- **Temporal correlation model (spec §4.2, last step, "optional"):** neither the per-camera latent AR(1) bias nor
  whitening is implemented. The calibration measures a flat lag-1…5 correlation of about 0.3. That is not AR(1)-like
  and was measured on static readout, so the spec forbids carrying it into motion residuals without validation. It is
  stored as a diagnostic.
- **Shared-camera / extrinsic covariance via Woodbury (§4.2):** not implemented. The independent diagonal model is kept,
  and its limitation is written into every `sensor_manifest.json`.
- **NIS / innovation statistics and per-frame per-camera residual histograms (§4.3):** not added. The existing per-frame
  energy splits and depth counts are still written.
- **Model-error map (§4.1):** left as the baseline ablation. With a calibrated model the Phase 2/3 model-error variance
  still adds to the sensor variance (σ_eff² = σ_sensor² + σ_model²). It is not relabelled as sensor noise, and its
  persistent-bias diagnosis is still the 2026-10-21 `model_error_audit.json`.
- **Bootstrap mode `existing` (reusing trained static models with an interval):** not offered. The canonical-endpoint
  check would refuse a frame-0 COLMAP model for a window that does not start at 0.

## 3. Known consequences to watch

- **RGB precision rises about 400×.** The calibrated RGB σ on red_fridge is about 0.005, against the default 0.1. With
  `--rgb-model-sigma 0` the RGB term is weighted as if the frozen-appearance renderer were exact, so rendering
  mismatch then dominates the residuals. Use the declared `--rgb-model-sigma` / `--depth-model-sigma` as an explicit
  ablation, recorded in every manifest and signature.
- **Old results are not directly comparable.** Process noise per step still uses the unchanged σ values with the real
  dt (0.04 s on red_fridge).
- **Most Gaussians observe only 4–5 of 6 pose directions.** On real data only 730 of 19,749 seeds observe all 6 (a
  surface Gaussian's tangential translation and its spin about the normal are weakly observed). Those directions keep
  the declared prior variance, which is the intended behaviour.

## 4. Verification

| check | status | evidence |
|---|---|---|
| A: AR(1) oracle, 3 cameras with σ = 1/2/4 mm, ρ = 0.5, recovered and never pooled | PASS | `tests/test_calibrated_bootstrap.py` [A] (σ within 2%, ρ within 0.03) |
| A: naive Var(ΔD)/2 under-estimates autocorrelated noise; the corrected form recovers it | PASS | [A] |
| A: holes and outliers, invalid fraction, heteroscedastic a+bz² selected on held-out frames, quantized depth | PASS | [A] |
| A: persistent bias invisible (non-identifiable, reported); interval selection; id round trip and tamper refusal; no physical model for synthetic | PASS | [A] |
| B: nᵀCn Monte Carlo; 3 noisy cameras beat plain averaging (0.13 vs 0.39 mm); sharp 5 cm step not averaged across; zero-noise exact and deterministic; P0 = prior where F = 0 | PASS | [B] |
| B: responsibility sums vs renderer alpha (real CUDA) | PASS | [B] (≤ 1.8e-7); red_fridge 4.8e-7; SAPIEN 4.2e-7 |
| B: fixed Gaussian count and IDs through the refinement | PASS | asserted at every step (smoke runs) |
| C: calibrated per-camera precisions (original and masked), identity required, declared mismatch additive, ideal path unchanged | PASS | [C] |
| C: config copy keeps depth_sigma_a/b and the sensor model; P0 from file plus declared velocity; per-frame dt from timestamps; backward elapsed time positive; canonical-endpoint and window refusals; orchestrator refusals | PASS | [C] |
| E2E real (red_fridge ep 0, calibration 0–99, motion 110–113): all 6 stages, one `sensor_model_id` in every stage, 0.04 s steps, backward canonical 113, signatures include sensor, interval and P0 | PASS (smoke) | scratch run |
| E2E synthetic (SAPIEN fridge, motion 0–2, ideal) | PASS (smoke) | scratch run |
| Existing suites (Gate F: default paths unchanged) | PASS | 19 suites: masked_default_repair 76, masked_rgb_observation 46, iekf_end_to_end, rigid_se3_phase2 47, joint_phase3_runner 12, bidirectional_sequence_direction 54, phase2_parallel_pr 37, visibility phase1 12 / phase23 8, v2_grouped_solve 16, kpart_window 74, motion_coherence_support 41, transparency_carving 63, object_mask_transparency_carving 45, carving_depth_object_only 4, efficient_p_null 20, joint_screw_iekf 37, iekf_pcg_measurement 146, v2_full_scale_trust_region 35 |
| Gate A–E at full scale; P0/sensor ablations; NIS calibration; held-out benchmark of raw seed vs standard 3DGS vs bootstrap | NOT RUN | user's runs (§5) |
| Temporal AR(1) / Woodbury shared covariance | NOT IMPLEMENTED | §2 |

## 5. Running it

`~/run_calibrated_bootstrap.sh` runs the 14 red_fridge episodes (calibrated) and then the 10 SAPIEN scenes (ideal).
Each scene's results go to its own folder: `results_11_10_26/ours_v4_calibrated_bootstrap/<episode>/` and
`ours_v4_bootstrap_synthetic/<scene>/`.

The per-episode intervals in the script come from a one-off diagnostic, not from the pipeline:
- the **onset** is the first frame where, in any camera, the mask IoU with frame 0 drops below 0.98 or the median
  in-mask |Δz| exceeds 2 mm, for 3 consecutive frames;
- **calibration** = [0, onset − 10];
- **motion** = [onset − 5, last frame].

Verify them before running.
