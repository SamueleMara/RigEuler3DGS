# robot_dog silhouette-induced static drift: audit, correction, tests

Date: 2026-11-17. Worktree `/home/samuelemara/Joint3DGS/RigEuler3DGS-occ2`, branch `phase1-occlusion-contam`.
It was created from `main` 1f61c47, plus a copy of the `RigEuler3DGS-occ` working tree that produced `ours_v9_phase1_occlusion_real3`. Nothing is committed.

The reference commit is 3bacfe5. Between 3bacfe5 and 1f61c47 nothing changed in `estimator/`, `tools/kpart_pipeline.py`, `tools/run_motion_coherence_support.py` or `run_bidirectional_all.sh`. So the diff against 3bacfe5 is exactly the occlusion implementation plus this correction.

## 0. Status

| item | status |
|---|---|
| 1 audit | done (section 1) |
| 2 regression isolated by ablation on real robot_dog | **done** (section 2A). The cause is `--phase1-group-update observable`, not the negative-ray likelihood. |
| 3 contamination likelihood with calibrated q | implemented (section 3). On real robot_dog it lowers drift relative to the legacy off-mask terms (section 2A). |
| 4 consistency of energy / RHS / operator / trial / posterior | implemented. Tested against a dense reference (section 5 [2]). |
| 5 mandatory synthetic tests | 30/30 (`tests/test_phase1_negative_ray_reliability.py`). Earlier suites: section 5. |
| 6 real regression test | first run (calibrated q + observable group) **FAILS** the gate: end median 34.1 mm against 7.7 mm for v6. Rerun with the legacy group is pending (section 6). black_laptop not run. |
| 7 report | this file |

**Root cause (section 2A):** the observable group update. The negative-ray term is not the cause: under the legacy group, it reduces static drift.

## 1. Audit of the current implementation

### 1.1 Where P1LIK and P1GRP act

```
run_bidirectional_all.sh  --phase1-likelihood P1LIK --phase1-group-update P1GRP [--negative-ray-reliability P1REL]
 -> tools/run_motion_coherence_support.py main()
      estimator/occlusion_likelihood.from_args(args)        -> ns.occlusion_likelihood (None = legacy)
      [new] calibrate_reliability(...)                       -> ns.occlusion_likelihood.calibration
      cfg = replace(cfg, group_update=..., group_update_map=...)
 -> tools/kpart_pipeline.bootstrap_positions(..., args)     (per frame)
      estimator/sparse_iekf.iekf_frame_step(..., occlusion=args.occlusion_likelihood)
        _frame_terms(views, obs, cfg, ..., occlusion)
           occlusion is None  -> masked_observation.phase1_camera_terms      (legacy: RGB + alpha on ALL pixels)
           else               -> occlusion_likelihood.positive_camera_terms  (RGB + alpha on M = 1 only)
        evaluate_true_map_energy(...)                       (initial state and EVERY trial)
           measurement = positive terms + [occlusion] NegativeRayEvaluation.build(...).energy
           lin.negative_ray_evaluation = that evaluation
        _meas_ops(lin, prec, alpha)                          (RHS, normal operator, Fisher)
           rhs    = measurement_operators(...)[0] + nr.score_pose()          (-grad L)
           normal = measurement_operators(...)[1](v) + nr.normal_pose(v)     (GN, PSD)
           fisher = measurement_operators(...)[2]() + nr.fisher_pose()      (6x6 blocks)
        b = rhs - Lambda^- (x - x^-)
        group_update == "observable": B = [I; K] Q G (observable_group.build_map from P^- and fisher)
                                      solve B^T A B du = B^T b, trial boxplus(x, B du)
        else (legacy):                G (12-DoF group twist), solve G^T A G du = G^T b
        posterior: fisher of the final linearization (same _meas_ops) -> block (Lambda^- + F_ii)^-1
```

The negative-ray kernels are in `submodules/diff-gaussian-rasterization/cuda_rasterizer/iekf_negative_ray.cu`, a separate JIT extension that only reads the production `IEKFLinearizationCache`:

| term | kernel |
|---|---|
| energy, A | `nrEvalCUDA` |
| gradient | `nrVJPCUDA` with c1 = q/E |
| JVP | `nrJVPCUDA` |
| curvature | `nrVJPCUDA(c2 · nrJVPCUDA)`, c2 = (q/E)² |
| posterior and observable-map Fisher | `nrFisherCUDA` |

All of them chain through the shared `poseTangent` and `dAlphaOf` in `iekf_device.cuh`, and they share the production contributor walk: `evalAlpha`, skip < 1/255, clamp 0.99, the same float `T_STOP` test.

### 1.2 Which pixels were negative rays in the run that drifted (certified, q = 1)

`ray_status`: status 2 (KNOWN_NONOBJECT_DEPTH) means M = 0, finite raw depth > 0, and no M = 1 pixel in the 3×3 neighbourhood. Every such pixel had q = 1. There was no test for depth discontinuities, and nothing checked whether the measured depth matched the object's own surface.

### 1.3 Off-mask RGB and alpha are removed on those pixels

- **In the code:** `positive_camera_terms` passes `rgb_valid = M` and `kappa_fn = kappa · M`, and depth validity is `V_raw AND M`, as in v6.
- **Measured:** the diagnostic decomposes the production RHS per camera, per channel and per side of the mask. In the occlusion arms the off-mask RGB, depth and alpha forces are exactly 0.0. The sum of all components equals the production RHS to 1.2–1.6e-7.
- **Legacy, for comparison:** off-mask RGB targets the grey background and alpha uses ½κ(M − A)² on every pixel. Neither depends on depth.

### 1.4 What changed against 3bacfe5

| file | change |
|---|---|
| `estimator/masked_observation.py` | `depth_valid_raw` field (unused by legacy) |
| `estimator/sparse_iekf.py` | `occlusion=` threading, `group_update`, observability export; observer kwargs `solve_b`, `observable_map`, `expand_step`, `occlusion` (new here) |
| `tools/kpart_pipeline.py` | passes `occlusion`, writes the observability export |
| `tools/run_motion_coherence_support.py` | CLI; reliability calibration before Phase 1 (new here) |
| `run_bidirectional_all.sh` | P1LIK, P1GRP, **P1REL** (new here; in the stage signature as `P1OCC=.../q=P1REL`) |
| new: `iekf_negative_ray.cu`, `negative_ray_operators.py`, `occlusion_likelihood.py`, `observable_group.py` | first implementation; **per-pixel q** (new here) |
| new: `estimator/negative_ray_reliability.py` | calibration (new here) |
| new: `tools/diagnose_phase1_negative_ray.py`, `tools/compare_static_displacement.py`, `run_phase1_negative_ray_fix.sh` | step 2 and step 6 (new here) |
| new: `tests/test_phase1_negative_ray_reliability.py` | step 5 (new here) |

With the default flags (legacy / legacy) one synthetic frame is bitwise identical to `main`: mu, R, v, P and every energy.

## 2. Evidence on the regression (real3, read-only; not yet the ablation)

These come from the saved outputs of `ours_v9_phase1_occlusion_real3`, `ours_v6_two_cams` and `ours_v8_overnight_real/draw1, draw2`. They are not proof of cause.

**robot_dog, one fixed static set** (22,676 Gaussians outside the v6 ∪ certified supports):

| run | end median | end p90 | > 50 mm |
|---|---|---|---|
| v6 / draw1 / draw2 (same pipeline, three runs) | 7.8 / 6.7 / 7.7 mm | 76 / 82 / 64 mm | 3,944 / 4,244 / 3,296 |
| certified (q = 1, observable) | 35.2 mm | 113 mm | 8,044 |

On black_laptop the certified run was better: 43 mm median against 76–80 mm.

**What is known about the drifting static Gaussians** (robot_dog, certified):
- 96% of their frames had measurements; less than 4% were prediction-only. This is not the hidden-Gaussian case.
- Their depth grew by 39 mm in both cameras, against 6 mm for static Gaussians that stayed put.
- Their footprints carry about 130× more weight in front of the measured M = 0 depth (1.2 against 0.009 per frame).

**Calibration statistics** (CPU, every frame):
- For mask-0 pixels 1–12 px from the contour, 40–95% have a depth within 3σ of the nearest object pixel. Far from the object the rate is 0.4–2.4%.
- With q = 1, each such pixel pushes nearby object Gaussians along its ray with a force ∂A/∂z = −w φ(u)/σ. Here σ ≈ 3.3 mm (calibrated sensor σ combined with the declared 3 mm model σ).

**Hypothesis to be confirmed by part T:** the certified negative ray, acting on contaminated near-contour pixels, pushes silhouette Gaussians deeper along the viewing rays. The push builds up through the velocity state.

Part T is decisive. It runs certified/legacy against certified/observable and calibrated/*, and replays the reference arm's exact frozen states through every arm. The hypothesis predicts:
- the negative-ray force on exposed static Gaussians points along +z (away from the camera);
- that force dominates their RHS;
- it is absent in calibrated/*;
- the group update is not the cause, so certified/legacy drifts as well.

**Outcome (section 2A): REFUTED.**
- certified/legacy does not drift: 2.3 mm median, against 3.5 for legacy/legacy.
- The negative-ray force along +z is real but small: 29-59 against the legacy off-mask 360-2530. It does not dominate the RHS.
- The drift follows the group update.
- The 39 mm depth growth above is consistent with the observable group's velocity build-up. The ablation does not show what sets its direction along the rays.

## 2A. Ablation on real robot_dog (2026-11-17, `~/results_local/ours_v10_negative_ray_fix/ablation_robot_dog`)

**Setup:** the same V6 endpoint, P0, sensor model, cameras and deterministic reductions for every arm. Frames 1-40 (processing indices). 22,676 static Gaussians, outside the v6 ∪ certified supports.

### Sequential arms (static displacement at frame 40)

| likelihood / group | median mm | p90 | p99 | > 50 mm | static speed at end, mm/s |
|---|---|---|---|---|---|
| legacy / legacy | 3.5 | 31.8 | 107 | 1340 | 3.4 |
| legacy / observable | 22.8 | 87.5 | 211 | 5245 | 13.9 |
| certified q=1 / legacy | 2.3 | 19.7 | 83 | 785 | 2.2 |
| certified q=1 / observable | 16.2 | 63.2 | 173 | 3319 | 10.1 |
| calibrated q / legacy | **2.3** | **18.8** | **75** | **620** | 2.5 |
| calibrated q / observable | 15.1 | 65.2 | 178 | 3409 | 9.0 |

- **Group factor:** with the likelihood held fixed, switching the group from legacy to observable multiplies the median 6.5-7x and the > 50 mm count 4-6x, for every likelihood.
- **Likelihood factor:** with the group held fixed, replacing the legacy off-mask terms by the negative-ray term lowers every statistic. Calibrated q is the lowest on p90, p99 and > 50 mm.

### Frozen states

Every arm takes one step from the legacy/legacy arm's exact pre-update state (processing indices 5, 15, 30). The table gives the median component of the static velocity correction along the existing velocity (mm/s; negative = braking).

| index | legacy lik. / legacy grp | legacy / observable | calibrated / legacy | calibrated / observable |
|---|---|---|---|---|
| 5 | -3.1 | -1.9 | -4.5 | -2.8 |
| 15 | -1.2 | -0.7 | -1.4 | -0.7 |
| 30 | -1.7 | -0.7 | -1.6 | -0.8 |

From the same state, the observable map brakes static velocity at 40-60 % of the legacy rate.

### Sequential consequence

- Static speed under legacy groups decays from 42 to 3.4 mm/s. Under observable groups it stays at 14-18 mm/s.
- |v_21| / |v_1| (median) is 0.09 under legacy and 0.38 under observable. Pure process decay (tau = 0.5 s) would give 0.20. So the observable arm re-injects velocity; it does not merely fail to remove it.
- Per-frame predicted displacement stays at ~1.1 mm/frame (legacy: ~0.5). The position correction cancels most of it each frame, but the remainder integrates.

### Silhouette depth push along camera Z (frozen, exposed static Gaussians, iteration 0)

- The legacy off-mask RGB+alpha force along each camera's z is +360 to +2530 (objective units per metre).
- The negative-ray force that replaces it is +17 to +59 for calibrated q, and +29 to +59 for certified.
- The accepted z correction of exposed static Gaussians is below 1 mm per frame in every arm. Its sign follows the prior force, not the negative-ray force.

So the silhouette depth push exists, and comes mainly from the legacy off-mask terms. The new likelihood shrinks it by one to two orders of magnitude.

### Refuted hypothesis

"The observable map leaves velocity uncorrected only on directions without measurements, so drift sits on weakly measured Gaussians." The data contradict it: drift under the observable group increases with measurement force. By quintile of force magnitude at index 5:
- observable: median 13.7 / 18.7 / 22.9 / 25.7 / 42.1 mm;
- legacy: 3.4 / 3.0 / 3.1 / 3.2 / 7.3 mm.

Force magnitude is a gradient, not information, so this is a proxy; the Fisher eigenvalues were not saved.

### Not established

- **Why** the [I; K] Q lift brakes less: shrinkage q = lam/(1+lam) of the pose step feeding K, versus the legacy group's free velocity DoF, is a candidate, untested.
- Why the drift concentrates on high-force members.

### Full 151-frame run with calibrated q + observable group (`ours_v10_negative_ray_fix/occ1`)

Gate FAIL. End median 34.1 mm, p90 114, > 50 mm 7804, against v6 7.7 / 75.9 / 3904, draw1 6.7 / 82.5 / 4206, draw2 7.6 / 64.3 / 3256. It is the same as the certified run's 35.0 / 112 / 7984, which also used the observable group. That is consistent with the ablation.

## 3. The correction

### 3.1 Objective

Before (certified):

    L_p = -log(1 - A_p),   A_p = Σ_k w_k Φ((D_p − z_k)/σ_p),   on every status-2 pixel

After (contamination mixture, per pixel):

    p(ray p | x) = (1 − q_p) · 1 + q_p · E[T_front(x; D_p)]
    L_p = −log((1 − q_p) + q_p (1 − A_p)) = −log(1 − q_p A_p)

- q_p is the probability that the ray is a clean non-object measurement.
- With probability 1 − q_p it is contaminated (a mask false negative, or a surface the sensor cannot tell apart from the object) and carries no information.
- q_p depends only on the observation (mask, raw depth, calibrated σ), never on the state, so L remains a likelihood.
- q_p = 0 means UNKNOWN: no term.
- Positivity: 1 − qA ≥ T_STOP > 0, as before.

### 3.2 Calibration of q (`estimator/negative_ray_reliability.py`)

The calibration runs per camera, pooled over every frame of the run, before Phase 1, from observation data only (no GT).

**Candidate pixels:** M = 0, valid raw depth, not on the 3×3 mask edge, not a depth edge. For each one:
- **d_p:** distance (px) to the nearest valid M = 1 pixel, by exact EDT.
- **cons_p:** whether |D_p − D_m(p)| ≤ z_{1−a/2} · √(σ_p² + σ_m²), with a = 0.0027 (the two-sided 3σ level).

**Declared assumptions:**
- **A1:** a contaminated pixel is depth-consistent.
- **A2:** clean pixels are depth-consistent at a rate r_∞ that does not depend on d. r_∞ is measured on the far half of the candidates (d ≥ d_far, the median candidate distance).
- **A3:** contamination depends on d only.

**Estimate:** r(d) is fitted as a non-increasing step function by weighted pool-adjacent-violators. Then:

    c(d) = clip((r(d) − r_∞) / (1 − r_∞), 0, 1)            for d < d_far,  0 beyond
    q_p = (1 − c(d)) r_∞ / r(d)   if cons_p                (Bayes: P(clean | consistent, d))
    q_p = 1                         otherwise

**When A2 fails:** a floor under the object's feet breaks A2. It inflates c near the object, so q is under-estimated. That direction only removes negative evidence; it never adds free-space pressure.

**Depth edges (mixed returns):** if the 3×3 range of the valid depths exceeds the (1 − a) quantile of the range of n iid N(0, σ_p²) samples, the pixel is UNKNOWN (q = 0). No reliability can be calibrated there, so the task's fallback applies: UNKNOWN, with zero negative evidence.

**Real data** (CPU, read-only, 151 and 174 frames):

| camera | d_far | r_∞ | c(2..6) | q near the contour (former status 2, d ≤ 5 px) |
|---|---|---|---|---|
| robot_dog cam_001 | 99 px | 0.004 | 0.71 → 0.42 | mean 0.42, 59% below 0.1 |
| robot_dog cam_002 | 91 px | 0.007 | 0.90 → 0.67 | mean 0.26, 74% below 0.1 |
| black_laptop cam_001 | 161 px | 0.006 | 0.98 → 0.87 | mean 0.17 |
| black_laptop cam_002 | 127 px | 0.024 | 0.96 → 0.77 | mean 0.24 |

- Far field (d > 40 px): mean q is 0.92–0.97.
- Depth-edge pixels: 0.7–3.6% of the M = 0 valid pixels.
- Calibration time: about 40 s per camera.

### 3.3 What was not done (forbidden by the task, or not needed)

No tuned mask-band width; no global multiplier; no post-update reset; negative rays are not silenced everywhere, so far-field q stays ≈ 1; nothing in Phases 2, 3 or 4 changed.

The spec's 1-px (3×3) mask-edge UNKNOWN rule is unchanged. It came from the spec, not from this correction.

## 4. Mathematical consistency

- **Kernel:** `NRCamera.q` is a per-pixel map. E = 1 − q_p A_p is the only place q enters. The contributor walk and ordering are unchanged.
- **Same maps everywhere:** `OcclusionLikelihood.pixel_model(ob, camera, σ)` computes the status and q once per (camera, frame) and caches them. The same maps reach `nr_eval` (initial state and every trial), the RHS c1 = q/E, the GN operator and Fisher c2 = (q/E)², and the posterior Fisher of the final linearization.
- **Curvature:** Gauss–Newton, with the (q/E) ∂²A term dropped (PSD), as before.
- **Same reduction maps:** the grouped (G) and observable (B) maps apply the same B and Bᵀ to every term. Tested in [2] below.
- **No post-update reset** of pose or velocity.

## 5. Tests

`tests/test_phase1_negative_ray_reliability.py`: **30/30**, identical across two consecutive runs. Real CUDA renderer; 3 cameras at 96×96 unless stated.

**[0] Calibration estimator**
- The range quantile for n = 2 is exactly √2·z: 4.2426.
- An injected 60% ring contamination is recovered as 0.50, within the tolerance of 0.1.
- Far field: c = 0.
- Consistent ring pixels: q ≤ 0.05. Inconsistent candidates: q = 1.

**[1] Random per-pixel q** (mean 0.50, sd 0.29), at frozen contributor order:

| check | result |
|---|---|
| JVP against FD of A, 26,543 pixels | rel L2 1.6e-4 |
| energy FD of −log(1 − qA) against (q/E)·JVP | 390.900 vs 390.940 |
| score = −Jᵀ(q/E) | rel 1e-7 |
| adjoint | rel < 1e-4 |
| normal operator | symmetric, PSD |
| Fisher against diagonal blocks of the normal operator | 8.4e-8 |

**[2] Dense small-scene reference** (14 Gaussians, 2 cameras at 40×40):

| check | result |
|---|---|
| negative-ray normal against dense Jᵀ diag((q/E)²) J | 1.6e-7 |
| negative-ray score against dense −Jᵀ(q/E) | 2.1e-8 |
| production grouped RHS against dense Bᵀb, observable / legacy | 4.9e-8 / 2.9e-8 |
| reduced normal operator against dense BᵀAB, observable / legacy | 6.5e-7 / 2.8e-7, symmetric to 4e-7 |

**[3] Static object with an eroded (3 px) mask contour.** The false-negative ring carries the object's own depth. Mean |motion along each camera axis| after 8 frames:

| mask | q = 0 | certified | calibrated |
|---|---|---|---|
| perfect | 0.461 mm | 0.452 mm | 0.453 mm |
| eroded | 1.135 mm | 1.816 mm | 1.157 mm |

- Increment over q = 0: certified +0.681 mm, calibrated **+0.022 mm**.
- With the perfect mask, calibrated keeps the certified evidence: 0.453 against 0.452 mm.
- The 1.1 mm with q = 0 is positive evidence lost on the eroded ring, not the negative ray. See section 7.
- In this scene, legacy/legacy diverges by 128 mm: its grey-background RGB term on the false-negative ring.

**[4] Hidden behind a measured occluder:** the score and Fisher of the hidden Gaussians are exactly 0.

**[5] Isolated Gaussians over reliable background:**
- They touch no M = 1 pixel, and q = 1 on their 234 free-space pixels.
- They are corrected along a descent direction of the negative-ray energy: ⟨score, step⟩ > 0, cos 0.44, moved 1.27 mm.
- With q = 0 they don't move (0.0000 mm).
- Their energy is flat (275.319 → 275.331): with a far wall, Φ = 1 and only the footprint term remains.

**[6] Mixed-depth silhouette pixels:** certified gives all of them q = 1. Calibrated makes all 208 of them UNKNOWN (q = 0).

**[7] Visible in two cameras, hidden in the third:**
- A camera that never sees the object calibrates to far field (no contamination).
- The hidden camera's negative-ray force on the object is exactly 0.
- The motion is re-acquired from the visible cameras: 8.94 → 1.19 mm.

**Earlier suites in this worktree:**
- `test_phase1_occlusion_likelihood`: 25/25, unchanged, with explicit `"certified"`.
- `test_phase1_observable_group`: 26/26, unchanged.
- Remaining suites, all passing: `test_masked_rgb_observation` 46, `test_visibility_constrained_iekf_phase1` 12, `test_phase1_covariance_velocity` 47, `test_deterministic_ops` 8, `test_motion_coherence_support` 41.
- Legacy parity against `main`: bitwise.

**Diagnostic tool on a synthetic scene (all six arms):** RHS additivity 1.2–1.6e-7; off-mask forces in the occlusion arms exactly 0; negative-ray force on exposed static Gaussians 88.6 (certified) against 8.4 (calibrated); the frozen replay runs.

## 6. Real regression test (your launch)

### First launch, 2026-11-17

```
cd /home/samuelemara/Joint3DGS/RigEuler3DGS-occ2
bash run_phase1_negative_ray_fix.sh 2>&1 | tee -a ~/phase1_negative_ray_fix.log
```

- T, the ablation: section 2A.
- A, calibrated + observable: gate FAIL (section 2A).
- B was not run.

### Second launch, pending

Calibrated q with the **legacy** group (`P1GRP` now defaults to legacy). It writes to the SSD, `results_11_10_26/ours_v10b_negative_ray_legacy_group`; the first fix run (internal disk, `~/results_local/ours_v10_negative_ray_fix`) is shown as `prev` in the tables.

```
cd /home/samuelemara/Joint3DGS/RigEuler3DGS-occ2
PARTS="A G B" bash run_phase1_negative_ray_fix.sh 2>&1 | tee -a ~/phase1_negative_ray_fix_b.log
```

**Gate:** the run must be no worse than the worst of v6 / draw1 / draw2 on the end median, p90 and > 50 mm count of the fixed static set (outside the v6 ∪ certified ∪ fix supports). black_laptop runs only if robot_dog passes.

**Safeguards:** the same as before. Only one copy runs at a time, stale build locks are removed only when idle, and it needs at least 12 GB free.

## 7. Remaining ambiguity and unsupported claims

- **Root cause:**
  - Proven at the level of the two switches: the group switch causes the drift, the likelihood switch reduces it.
  - Not proven at the level of mechanism inside the observable lift (section 2A, "Not established").
  - Ablation over frames 1-40 only; the full window is covered by the gate runs.
- **Whether calibrated + legacy passes the 151-frame gate:** unknown until the second launch. The 40-frame ablation predicts it.
- **Spec item I.D (observable group):** its purpose was to stop hidden Gaussians being dragged by their visible neighbours. With the legacy group that dragging returns. Under the legacy group it is what holds static Gaussians still on robot_dog.
- **Calibration:**
  - A2 is violated by surfaces continuous with the object (floors, tables). The estimator then treats them as contaminated (q low): conservative, but it discards valid free-space evidence near contact regions.
  - The calibration pools all frames of the run, future frames included. It is treated like the sensor-model calibration (a nuisance parameter), not as a causal filter quantity.
  - The test level a = 0.0027 and d_far = the median candidate distance are declared choices, written to `phase1_observation_model.json`.
- **Positive model:** with an imperfect mask contour it still moves a static object (1.1 mm in [3] with q = 0). That is not addressed here. The positive RGB-D ownership mixture remains unimplemented, as before.
- **Posterior:** the observable-group posterior is still the uncertified block approximation.
- **Diagnostic tool:** tested on synthetic scenes only. Running it on real data was not allowed from this session.
- **Speed:** the calibration adds about 80 s per two-camera scene; per-frame cost is unchanged.
