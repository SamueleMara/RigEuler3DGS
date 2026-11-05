# RigEuler3DGS — Implementation Memory and Technical Handoff

**Snapshot date:** 2026-11-05 (Europe/Rome)  
**Purpose:** Self-contained context for a new ChatGPT/Codex session; carefully distinguishes **implemented code**, **reported tests**, **unverified performance**, and **future designs**.  
**Main repository:** https://github.com/SamueleMara/RigEuler3DGS  
**Latest verified remote `main` HEAD:** [`4c887257ef81e1d0b882214ada6cf762530da83b`](https://github.com/SamueleMara/RigEuler3DGS/commit/4c887257ef81e1d0b882214ada6cf762530da83b), dated 2026-11-04 22:46:47 UTC.  
**Commit title:** `Calibrated RGB-D sensor model, surface-centred endpoint bootstrap, sensor + time through Phase 1/2/3`.  
**Baseline for comparison:** `481b2cf2d5602932286948cf14114f9caee4de3c` (2026-10-22). There are **three** intervening commits in `481b2cf...4c88725`, including `b194e169...` (joint reduction in physical time) and `79779876...` (mask table fix), not just the large noise/bootstrap commit. GitHub comparison lists **26 changed paths** in this range.  
**Primary implementation report committed in the repo:** [`docs/CALIBRATED_BOOTSTRAP_IMPLEMENTATION_2026_11_04.md`](https://github.com/SamueleMara/RigEuler3DGS/blob/4c887257ef81e1d0b882214ada6cf762530da83b/docs/CALIBRATED_BOOTSTRAP_IMPLEMENTATION_2026_11_04.md).  
**Source of test/performance claims:** That Codex report and recent Codex transcript (`Pasted text(20261104-234901).txt`). The implementation was inspected at the remote commit, but the full-scale experiments have **not** been independently executed in this handoff.

---

## 1. Current scientific goal and design constraints

Replace the unreliable **front end of FreeArtGS** for real multiview RGB-D articulated objects with the already successful RigEuler3DGS motion/kinematic stages, then exploit the estimated joint to recover surfaces missed by the initial tracker and finally perform full articulated 3DGS appearance reconstruction.

**Preserve these roles and their order:**

1. **Phase 0:** static, fixed-population, tracking-oriented 3DGS at each selected motion endpoint. The new version uses a noise-aware, surface-centred bootstrap.
2. **RigEuler Phase 1:** free per-Gaussian IEKF transport forward/backward over the motion interval. This is *not* the final static/moving segmentation.
3. **Confident support extraction:** graph/trajectory coherence selects a deliberately high-precision moving core. The rest are not all definitely static.
4. **RigEuler Phase 2:** shared moving-body rigid `SE(3)` IEKF, then comparison of **both** revolute and prismatic joint hypotheses.
5. **RigEuler Phase 3:** fix chosen joint geometry; refit joint displacement and velocity `q_t, qdot_t` from RGB-D.
6. **NEW FUTURE Phase 4 (NOT IMPLEMENTED):** joint-conditioned recovery of initially present-but-missed moving Gaussians, then newly exposed/untracked surface geometry, using the **real nonlinear full renderer over the entire sequence**.
7. **NEW FUTURE Phase 5 (NOT IMPLEMENTED):** full photometrically accurate sequence-wide articulated 3DGS, borrowing FreeArtGS reconstruction principles, with rigid joint transport and eventual densification.

Do not replace these phases with SVGD, arbitrary per-point motion, blind nearest-neighbor matching, or a separate unverified algorithm. Do not use ground-truth part labels or joint parameters in inference. Use true GT only for post hoc evaluation where supplied. A supplied **motion-rich frame interval** is an observation/estimation interval, **not** numerical ground-truth motion.

Scenes of particular interest: real `red_fridge` episodes and SAPIEN synthetics including `blade_103706`, `storage_45135`, `USB_100109`, `scissor_11100`, `stapler_103111`, `laptop_10211`, `foldchair_102255`, `oven_101917`, `fridge_10905`, and `washer_103776`. The initial RigEuler synthetic joint performance is considered a vital baseline to preserve.

---

## 2. What **is implemented and committed** in `4c88725`

### A. Optional real static-frame per-camera RGB-D noise calibration — IMPLEMENTED

**Code:**
- `utils/noise/sensor_model.py`: `CameraNoise`, `SceneNoiseModel`; `depth_variance(camera,z)`, `rgb_covariance(camera)`, `temporal_model(camera)`; explicit `calibrated / ideal / assumed` policies, camera-ID validation, content SHA/model-ID verification.
- `tools/calibrate_static_rgbd.py`: estimators from precisely the user-supplied stationary frame interval; output `calibration/sensor_model.json`, `covariance.npz`, `validation.json`.

**Actual algorithm:** For each camera, use valid foreground raw RGB-D at a known stationary interval. Compute temporal per-pixel median, estimate and separately report a frame-wide common offset, pool outlier-trimmed variance by depth bins, account for depth/RGB quantization, and select a nonnegative depth-variance model (`a`, `a+b z²`, `a+b z²+d z⁴`) based on held-out odd/even frames. Estimate RGB `3x3` covariance, invalid-depth fractions and lag-1..5 correlation with uncertainty. Camera models remain separate. The time correlation is **diagnostic only**—no temporal AR filter is implemented.

**Why the estimator was changed:** The median absolute deviation collapses to zero on the actual real capture because depth is quantized per pixel at roughly **1 mm** and color is **8-bit**. The estimator uses trimmed variance and a measured quantization floor `q²/12`. Static frames cannot separately determine absolute depth bias versus actual surface range.

**Reported real calibration:** `red_fridge` episode 0, calibration frames `0–99`:

| Camera | Depth standard deviation | RGB std. (0–1 channels) | Invalid depth within mask | Lag 1–5 correlation |
|---|---:|---:|---:|---|
| `cam_000` | 0.71 mm | 0.004–0.005 | 7% | ≈0.3, approximately flat |
| `cam_001` | 0.94 mm | 0.005–0.0065 | 29% | ≈0.3, approximately flat |
| `cam_002` | 0.87 mm | 0.004–0.005 | 9% | ≈0.3, approximately flat |

The selected model was constant depth sigma across the available sampled depth ranges, and the reported split-half variance ratio was 0.8–1.25. These are measured diagnostics in the implementation report, **not independently verified sensor manufacturer specifications**.

**Policies:**
- Real calibrated: requires `--rgbd-noise-calibration` and both `--calibration-start-frame`, `--calibration-end-frame`; no assumed stationary motion data.
- Synthetic ideal: no stationary interval and no physical sensor noise calibration.
- Real without calibration: explicit `assumed` policy and status `UNCALIBRATED`.
- Calibration and motion intervals must be separately declared and the new orchestrator rejects overlap.

### B. Fast, surface-centred 3DGS at **motion-start and motion-end endpoints** — IMPLEMENTED

**Code:**
- `estimator/surface_bootstrap.py`
- `tools/train_tracking_bootstrap.py`
- `scene/__init__.py` corrected iteration-0 checkpoint loading (`is None` instead of truthiness), preserving ordinary callers.

**Each endpoint is independent:** Forward anchored at original `motion_start_frame`, backward anchored at original `motion_end_frame`. Each uses that endpoint's synchronized multi-camera RGB-D; the static calibration frames do **not** automatically contribute endpoint geometry.

**Actual algorithm:**
1. Backproject in-mask valid depth into a common 3D coordinate frame, with camera-specific rank-one depth covariance along its ray, `C_j = sigma_D,c²(z_j) d_j d_j^T` (world-frame rotation handled in geometry).
2. Produce approximately target-count seed positions via searched voxel spacing, not a free density-growing 3DGS.
3. Robust GICP-inspired multi-camera local **point-to-tangent-plane** IRLS fitting with Student-t weights, using normal variance `nᵀ C_j n + sigma_surface²`; constrain geometric thickness to avoid blending unrelated layers.
4. Initialize Gaussian means, tangent basis/quaternion, anisotropic tangent/normal scales, opacity and DC color; distinguish **rendering footprint covariance** `Sigma_render` from **geometric pose-estimate covariance** `P0_geometry`.
5. Fixed-count short Adam refinement (mean, quaternion, scale, opacity, DC), **no** densification, pruning, opacity reset or SH promotion. Supervision uses masked calibrated RGB, silhouette and Student-t **inverse-depth** NLL with propagated variance `Var(1/z) ~= Var(z)/z⁴`, plus declared thickness regularization.
6. Hold out 10% of masked pixels and evaluate checkpoints 0/250/500/1000/2000; these are benchmark checkpoints, not universal optima.
7. Export `bootstrap_manifest.json`, `geometry_information.npz` (`P0_geometry`, renderer Fisher/eigenvalues, seed normal variances, separate `Sigma_render`), and `source_responsibility.npz`.

**Pixel-to-Gaussian relationship:** Source responsibilities `w = T * alpha` are extracted from the production renderer's own contributor traversal (`observation_diagnostics.renderer_access.pixel_contributions`). The Gaussian/camera/pixel compositing provenance is stored as a sparse array, and per-Gaussian camera ownership is only **diagnostic**, not a hard rule excluding other cameras. The Gaussian identities are stable through bootstrap and Phase 1; **pixel positions do not stay fixed** as Gaussians move.

**Initial pose covariance:** `P0_geometry = (F + Lambda_prior)^(-1)` per Gaussian, where `F` is renderer Fisher and `Lambda_prior` is a declared weak prior. This is a **per-Gaussian approximation**, not exact joint covariance across Gaussians/cameras/time. Weakly observed directions retain high uncertainty.

**Reported checks:** On `red_fridge` frame 110, seeds were median 1.33 mm (p90 2.6 mm) away from that dataset's **own fused cloud**, not from independent true geometry. Renderer depth / alpha matched inverse depth with median ratio 1.0000, and responsibility alpha closure had maximum error ~4.8e-7. All 19,749 real seeds were visible in ≥1 camera; only 730 had all 6 pose degrees well observed; most had 4–5 observable directions. These are diagnostic, **not proof of improved tracking**.

### C. Per-camera calibrated measurement model + real time + initial P0 through Phases 1/2/3 — IMPLEMENTED

**Code changes:**
- `estimator/sparse_iekf.py`: `IEKFConfig` now carries `sensor_model`, `camera_names`, and separate declared `rgb_model_sigma`, `depth_model_sigma` (model mismatch, **not physical sensor noise**).
- `estimator/observation_terms.py::camera_terms`: camera-aware RGB/depth variance at measured depth, failing when a required calibrated camera ID is unavailable.
- `estimator/masked_observation.py`, `estimator/phase2_shared.py`, and Phase-1 `_frame_terms`: forward actual camera identity and use matching calibrated precisions.
- `tools/kpart_pipeline.py::bootstrap_positions`: `dataclasses.replace` prevents the old config-copy bug that lost `depth_sigma_a/b` and new noise parameters; opt-in bootstrap `P0_geometry` and separately declared velocity prior; physical per-step `dt`.
- `tools/run_iekf_scene.py::dataset_rgbd_frames`: optional timestamps from `metadata/frame_timestamps.csv`, preserves original frame IDs.
- `utils/sequence_direction.py`: positive elapsed time for forward/backward with original timestamps; endpoint provenance checking.
- `estimator/run_config.py`: shared CLI flags/config/manifest and canonical endpoint validation.
- `tools/run_motion_coherence_support.py`, `tools/run_rigid_se3_pipeline.py`, `tools/run_joint_phase3.py`: motion interval and sensor model plumbing.
- `run_bidirectional_all.sh`: pass-through flags, signatures including sensor/model/P0/time/range, window-aware masks.
- `tools/stage_signature.py`, `tools/check_masks.py` and related tools: original-frame-range-aware checks/hashes.

**Measurement-model semantics:** Calibrated per-camera `R_sensor` replaces arbitrary global RGB and depth sensor sigmas; optional `sigma_model` enters separately in variance. Phases 2/3 still retain reference model-error variance: `sigma_eff² = sigma_sensor² + sigma_model_error²`, whose systematic-bias interpretation remains open. The carried IEKF state prediction covariance `P^-` and process noise `Q` are not camera sensor noise, and must not be counted inside `R` a second time.

**Time:** The selected motion interval is inclusive and uses real `dt` from timestamps where available. Example real red_fridge was reported at `0.04 s/frame`. Forward/backward endpoint frames must correspond exactly to the requested motion start/end; a frame-0 static COLMAP model cannot silently be used for a nonzero-start dynamic window.

**What did not change:** RigEuler Phase 1 free Gaussian IEKF mechanics, high-confidence motion-support extraction, Phase 2 shared rigid `SE(3)` estimation and both P/R joint fits, Phase 3 fixed-joint `q, qdot` estimation. Grouped corrections and block-marginal covariance approximation remain as before. Existing robust RGB-D and masked rendering are retained.

### D. Added opt-in orchestrator — IMPLEMENTED

`tools/run_noise_aware_rigeuler.py` performs input schema validation before CUDA, then stationary noise calibration (when required), independent endpoint bootstraps, and the original bidirectional driver over Phases 1/2/3. Records `run_manifest.json` including code commit/dirty flag, input calibration hashes, frame ranges, stage status and `sensor_model_id`. Existing drivers remain usable.

**Exact currently supported CLI options** include `--source`, `--dataset-kind real|synthetic`, `--rgbd-noise-calibration`, `--calibration-start-frame`, `--calibration-end-frame`, `--motion-start-frame`, `--motion-end-frame`, `--bootstrap-mode covariance_surface`, `--sensor-noise-model calibrated|ideal|assumed`, `--frame-timestamps auto|off|PATH`, `--num-gaussians`, `--refine-iters`, `--rgb-model-sigma`, `--depth-model-sigma`, `--p0-velocity-sigma`, `--observation-mode`, `--stages`, `--out`, and `--dry-run`.

Illustrative **actual supported** command syntax (adjust input path and verified intervals):

```bash
python tools/run_noise_aware_rigeuler.py \
  --source /path/to/real_scene --dataset-kind real \
  --rgbd-noise-calibration \
  --calibration-start-frame 0 --calibration-end-frame 99 \
  --motion-start-frame 110 --motion-end-frame 113 \
  --bootstrap-mode covariance_surface --sensor-noise-model calibrated \
  --frame-timestamps auto --num-gaussians 30000 --refine-iters 1000 \
  --out /path/to/new_output
```

For ideal synthetic scenes:

```bash
python tools/run_noise_aware_rigeuler.py \
  --source /path/to/synthetic_scene --dataset-kind synthetic \
  --motion-start-frame 0 --motion-end-frame 2 \
  --bootstrap-mode covariance_surface --sensor-noise-model ideal \
  --frame-timestamps off --out /path/to/new_synthetic_output
```

**Important caution:** Real calibration and model-mismatch assumptions are separate; `--rgb-model-sigma 0` means the fixed-appearance renderer is assumed exact, which may be very overconfident relative to the measured RGB sigma ~0.005. Validate alternatives by declared, reproducible ablation rather than hiding residuals with tuned variance.

---

## 3. Actual reported verification and **what remains untested**

The checked-in Codex report lists:

- Calibration simulator (3 cameras, known 1/2/4 mm noise and AR(1) rho=0.5): recovered sigma within 2% and rho within ~0.03; quantization, holes, heteroscedastic selection, corruptions and no absolute-bias identifiability tests reported PASS.
- Bootstrap planar synthetic test: robust method 0.13 mm vs naïve averaging 0.39 mm error in that test; sharp 5 cm step preserved; numerical covariance oracle and zero-noise/deterministic cases passed.
- CUDA responsibility closure within ~1.8e-7 in unit fixtures, ~4.8e-7 in red_fridge, ~4.2e-7 in SAPIEN; fixed Gaussian count asserted.
- Camera-specific measurement precision and config propagation; P0 copying; timestamp and frame-selection validations; backward time and endpoint provenance tests reported PASS.
- **Smoke E2E real:** `red_fridge` episode 0, calibration `0–99`, motion **`110–113`** (four motion frames): all six stage components completed, a consistent `sensor_model_id`, 0.04 s steps, backward canonical 113.
- **Smoke E2E synthetic:** SAPIEN fridge frames **`0–2`** (three frames), ideal policy completed.
- Reported old regression suites: 19 suites PASS (including masked default and RGB terms, bidirectional directions, Phase-2 rigid, Phase-3 joint, visibility, grouped solve, null-space, trust-region and masked carving).

**NOT tested / not claimed:** No full-length motion run or systematic comparison showing that the new bootstrap improves free motion, static drift, moving-support precision/recall, Phase-2 rigid fit or final joint-axis accuracy. No calibrated NIS tests on real dynamic streams and no full A/B benchmark against the original trained endpoint models. The observed 1.33 mm distance to an already fused cloud is **not ground-truth surface accuracy**.

**Performance risk:** Measured RGB sigma ~0.005 vs previous default ~0.1 ⇒ roughly 400× larger inverse-variance RGB precision if `rgb_model_sigma=0`. This may cause fixed-appearance renderer errors and multiview mismatch to dominate an IEKF. Validate this before treating low residuals as good motion. Process noise temporal behavior is not yet physically reidentified.

**Future benchmark:** Codex notes a local `~/run_calibrated_bootstrap.sh` script for 14 red_fridge episodes then 10 SAPIEN scenes, with proposed outputs under `results_11_10_26/ours_v4_calibrated_bootstrap/` and `ours_v4_bootstrap_synthetic/`. This script is **not verified as committed to the repository** and no results of those full batches are in the checked-in report. Its motion-onset heuristic (mask IoU <0.98 or median depth change >2 mm for 3 frames, calibration ending onset−10 and motion beginning onset−5) is *one-off interval selection*, not a scientific GT motion boundary. **Manually verify the intervals before use**.

---

## 4. Explicitly NOT implemented in `4c88725`

1. **Motion-time correlated sensor-error filtering:** no per-camera latent AR process, no temporal whitening. The static lag correlation (~0.3, flat across lags) is a recorded diagnostic only; it does not justify an AR(1) process on moving material observations.
2. **Shared calibration/extrinsic covariance:** no low-rank/Woodbury joint camera nuisance treatment; current RGB-D measurement covariance is independent per pixel, as manifests admit. The calibration's frame-global offset is reported separately and not yet tracked in the IEKF.
3. **Additional real dynamic noise fitting:** no learned motion-dependent or contour-specific covariance calibrated on held-out moving observations; no automatic calibration of representation bias as a stochastic sensor term.
4. **New comprehensive NIS/per-camera temporal-residual histograms** were not implemented, although preexisting depth/count/energy diagnostics remain.
5. **Full-scale verification** of improved tracking and joint geometry is not done.
6. **Post-joint Phase 4** (missed-moving support recovery, new geometry birth, sequence-wide validation) is **a research design**, not code committed at this SHA.
7. **Full FreeArtGS-style Phase 5** sequence-wide articulated photometric training/integration is **not implemented in the latest RigEuler commit**.
8. The alternative `bootstrap-mode existing` is not available for nonzero-start windows, because canonical endpoint manifests must agree with the chosen frame range.

---

## 5. Agreed **next** algorithm: Phase 4, joint-conditioned completion (**design only**)

**Core logic:** Phase 1 extracts a deliberately conservative, confident moving core to estimate the joint. Do **not** treat all remaining Gaussians as static. After Phase 3, use the only allowed motions:

- **Static:** `T_S(t) = I`.
- **Moving:** `T_M(t) = T_J(q_t)`, using the recovered prismatic/revolute joint; freeze joint geometry and `q_t` during the first recovery version.

### 5.1 First recover already-existing Gaussians missed by Phase 1

- Retain all original endpoint Gaussians and original stable identities.
- The `MOVING_RIGID` core anchors the recovered joint; `STATIC_COMPATIBLE` is a reversible static-leaning hypothesis and `INCOHERENT/AMBIGUOUS` stays unresolved (do not silently call it static).
- Estimate one persistent **moving-membership probability** per Gaussian `w_i=P(z_i=M | full-sequence RGBD, joint information)`, rather than binary assignment from a phase intended for high precision/low recall. Weights **do not vary arbitrarily per frame**; observed visibility and evidence vary by frame.
- Account for the trust in the joint using Phase-2 P/R model evidence, Phase-3 per-frame `q_t` posterior uncertainty (Phase-3 CSV has `posterior_sd_q`), possible joint-geometry uncertainty and held-out RGB-D. `posterior_sd_q` alone is not a global joint-validity probability; do not invent a constant joint-confidence scalar.
- Compare the two permitted hypotheses with **actual full nonlinear articulated 3DGS renders over all motion frames**, not IEKF linearizations or free 6-DoF motion per rejected Gaussian. Joint parameter uncertainty may be marginalized/approximated, but account for shared correlation.
- FreeArtGS uses soft duplicated opacity `alpha_S=(1-w)alpha`, `alpha_M=w alpha`. This is a useful differentiable *relaxation*, but two half-opacity ghost surfaces are **not equivalent** to a physical latent single-part mixture model. For final acceptance compare full renders of distinct assignments/conditional hypotheses; keep unresolved parts distinct when not identifiable.

### 5.2 Then discover surfaces missing from the initial endpoint model

- Rerender all cameras and all motion frames with the recovered original Gaussian membership, using one combined depth-sorted alpha compositing pass. Identify *unexplained observed foreground* while distinguishing new geometry from bad pose, camera mismatch, mixed depths, occlusion or invalid sensor returns.
- A candidate observation `y_{c,t,p}` yields **two canonical interpretations**: static `x_S=y`, moving `x_M=T_J(q_t)^(-1)y`. For a prismatic joint in the appropriate gauge, `x_M=y-q_t a`; for revolute, inverse rotation about the joint axis. Candidate points need not be visible at the motion-start frame.
- Associate/fit **local surface patches**, not blind pixel-to-pixel temporal matches or one Gaussian per raw noisy depth sample. Use calibrated depth geometry covariance, joint pose uncertainty, and multiple views/frames where available.
- **Every accepted moving Gaussian moves according to the estimated joint for every frame**, including frames *before birth/discovery* and frames where fully occluded. Each Gaussian has one canonical mean/orientation/scale/color/opacity; its world mean/orientation/covariance are rigidly transported.
- Validate every candidate/patch by **the real full nonlinear 3DGS renderer over the entire motion interval**, using RGB/depth/silhouette, depth ordering, alpha, visibility and held-out evidence. Do not use the IEKF local linear approximation for acceptance. Occluded frames impose no direct depth match on an invisible candidate, but the candidate must not incorrectly render in front of a measured surface. Pixels outside the view/with invalid depth provide no positive depth evidence.
- Acceptance means good observed support, joint-consistent canonical geometry and no unsupported sequence-wide visible contradictions *within calibrated uncertainty*, **not zero RGB-D error** and **not mandatory visibility in every frame**.
- New accepted geometry goes into a **separate reconstruction pool** with birth time, source-camera/frame/pixel history, uncertainty and static/moving/unresolved evidence. It must **not** change the dimensionality of original Phase-1 tracking retrospectively. Never claim to reconstruct surfaces never seen in any camera/frame without an explicit completion prior.
- Keep joint geometry and trajectory frozen for the initial Phase-4 implementation; optional later fixed-joint `q_t` refit must be a separate ablation with independent held-out evidence, not circular self-validation.

### 5.3 Proposed future modules (NONE currently in the repo)

`recovery/joint_adapter.py`, `recovery/articulated_full_renderer.py`, `recovery/sequence_visibility.py`, `recovery/part_membership.py`, `recovery/unexplained_observations.py`, `recovery/canonical_observations.py`, `recovery/candidate_optimizer.py`, `recovery/sequence_acceptance.py`, `recovery/surface_fusion.py`, `recovery/gaussian_completion.py`, `recovery/freeartgs_export.py`, and `tools/run_joint_surface_recovery.py`.

These are proposed names/interfaces; verify and adapt to the actual repository structure before coding. The full renderer must render static and moving Gaussians in a **single, correct depth-sorted pass**. Existing `gaussian_renderer.render` supports means/rotations overrides, but new appended Gaussian tensors and per-part contribution/visibility outputs will require a wrapper/extension and parity tests.

### 5.4 Why compare with FreeArtGS

- FreeArtGS's `reconstruction/generate_ply.py` collects depth pixels selected by per-part masks from **all frames**, backprojects them into each part's canonical coordinates and concatenates them before training.
- Per-part `splatfacto.py` trains Gaussians with RGB, silhouette, depth. `splatfacto_art.py` duplicates Gaussian static/moving contributions according to soft membership weights and uses a **single combined renderer** for correct self-occlusion. It can train articulated geometry/appearance and joint parameters.
- FreeArtGS does **not** explicitly enforce our proposed *candidate birth → render in every prior/future joint pose → visibility-aware sequence-wide acceptance*; its part masks and tracking can be wrong on real scenes, including previously shared diagnostics where moving part was invisible but many static pixels were mislabeled moving.
- Intended replacement: RigEuler's noise-aware tracking and confident joint estimates replace FreeArtGS's tracking/part inference; Phase 4 supplies vetted canonical geometry; Phase 5 may reuse FreeArtGS's mature articulated renderer/photometric optimization.

---

## 6. Next executable priorities and validation gates

**Immediate highest priority (before implementing new geometry birth):** Run genuine full motion sequences with **A/B ablations** to test whether `4c88725` improves tracking and joint accuracy; the current smoke tests prove integration only.

Suggested controlled arms, with same scenes, masks, camera sets, time windows, Gaussian count, code/runtime, and GT only for evaluation:

| Arm | Endpoint model | Phase-1/2/3 sensor / initialization | Purpose |
|---|---|---|---|
| A | Previous RigEuler standard static 3DGS | Previous default noise + original P0 | Original validated baseline |
| B | Surface-centred, **zero refinement** | Original IEKF noise/P0 | Effect of geometric fusion alone |
| C | Surface-centred + short refinement | Original IEKF noise/P0 | Effect of short bootstrap training |
| D | Same as C | Calibrated camera-specific `R`, declared model-mismatch ablations | Is calibrated noise too confident? |
| E | Same as D | Bootstrap geometric P0, physical dt, full original phases | Combined new pipeline |

Compare on synthetic prismatic/revolute and stationary controls and real red_fridge/blade/storage. Report per-Gaussian drift, moving support coverage/precision, Phase-2 rigid trajectory, both P/R fitted residuals and GT joint-axis/angle error when known, Phase-3 `q` error, held-out multiview RGB-D, memory, runtime and failure/ambiguity cases. Do not tune on GT.

**Highest-priority risk checks:**
1. Is the ~400× increase in RGB precision destabilizing a frozen-appearance IEKF or suppressing actual motion? Record residual components per camera, try declared mismatch ablations without claiming they represent physical sensor noise.
2. Do multiview depth/camera residuals reflect a persistent bias or geometric inconsistency which the per-pixel diagonal covariance fails to represent? Do not erase real motion via arbitrary `R_model` inflation.
3. Are hard foreground masks and occlusion maintained? Mixed foreground/background depth must not pull foreground Gaussians to a background surface.
4. Do physical `dt`, backward canonical endpoint, signed velocities and selected original frame indices behave on a long nonzero-start interval?
5. Does covariance initialization remain well-conditioned on planar or low-texture regions? `P0_geometry` is approximate; test uncertainty calibration before trusting numerical values.

**Only after validating these:** implement Phase 4 in sequence: (i) full renderer/parity and visibility; (ii) missed support reassignment with soft probabilities and joint trust; (iii) truly new observed-surface birth and canonical fusion; (iv) full nonlinear held-out sequence acceptance including occluded frames; (v) FreeArtGS exporter and final articulated 3DGS.

**Required tests for Phase 4:** synthetic prismatic drawer exposes later walls; revolute blade uncovers hidden surfaces; confident moving core remains anchored; initially static-compatible moving Gaussians recover without static false positives; ambiguous hinge-axis points remain unresolved; late-born moving Gaussians obey `T_J(q_t)` at *all* frames; correct occlusion renders them hidden earlier; false foreground occlusion is rejected; covariance transforms correctly under joint pose; no GT used for inference.

---

## 7. Reference map and compact glossary

**RigEuler repository:** https://github.com/SamueleMara/RigEuler3DGS  
**Implemented report:** `docs/CALIBRATED_BOOTSTRAP_IMPLEMENTATION_2026_11_04.md` at `4c88725` (source of code status and test numbers).  
**Codex-ready original spec:** `RigEuler3DGS_Calibrated_Bootstrap_IEKF_Implementation_Spec.md` (historical planning artifact; may be untracked).  
**Theory PDF (historical design):** `Noise_Aware_Multiview_3DGS_IEKF_Theory_2026-11-04.pdf` (theory, not proof of implementation).  
**FreeArtGS:** https://github.com/FreeArtGS/FreeArtGS ; its pinned Nerfstudio submodule contains `splatfacto.py` and `splatfacto_art.py`.  
**RGBDiekf:** https://github.com/SamueleMara/RGBDiekf — separate experimental point-cloud/SVGD work. Do **not** confuse its incomplete SVGD and temporal point-cloud filters with RigEuler3DGS's proven renderer-based IEKF phases.

Notation: `R_sensor` is camera measurement covariance; `P0_geometry` is estimated Gaussian pose prior covariance; `Sigma_render` is Gaussian spatial footprint; `P_t^-` is IEKF predicted motion covariance; `Q_t` is motion-process covariance; `R_model` is a separately declared rendering/geometry mismatch term; `w_i` is time-invariant probability that Gaussian `i` belongs to moving part (design only); `T_J(q_t)` is the Phase-3 joint-constrained rigid motion. These are **not interchangeable**.

**Handoff status in one sentence:** As of commit `4c88725`, calibration + fast surface bootstrap + sensor/timestamp plumbing are **implemented and smoke-tested**, but performance gains over original RigEuler remain **unproven**, while joint-trust-aware support completion and sequence-wide full-render missing-surface recovery are **next planned stages, not implemented**.
