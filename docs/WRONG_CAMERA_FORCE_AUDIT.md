# Phase-1 wrong-camera force audit: implementation

Spec: `CODEX_DEBUG_RigEuler3DGS_REAL_FRIDGE_WRONG_CAMERA_GAUSSIAN_DRIFT.md`. Branch `phase1-wrong-camera-audit` (worktree `RigEuler3DGS-wrongcam`), based on `phase1-first-return` `8273cd0`. This work is read-only: no likelihood, calibration, model, motion/process/covariance/partition, optimiser, support or Phase-2/3 change.

## Production touch points (all default-off)

| File | Change | Default behaviour |
|---|---|---|
| `estimator/sparse_iekf.py` | `iteration_observer` kwarg, called once per inner iteration under `no_grad` with the frozen assembly (x, x_prior, P_prior, Λ_prior, lin, per-camera terms, rhs, normal, b, G, lm, radius, M_blocks, observations). New `IEKFConfig.debug_ablation` field. | `None`, so no call is made |
| `tools/kpart_pipeline.py` | passes `args.phase1_iteration_observer` and calls `begin_frame(ti, fid)` on it | absent, so no call is made |
| `estimator/masked_observation.py` | `cfg.debug_ablation = {camera: [channels]}` removes those samples (RGB: no valid samples; depth: no valid depth; alpha: zero weight), so energy, RHS and curvature leave together | `None`, so unchanged |

## New code

- `estimator/wrong_camera_audit.py`
  - Per-view production operators (bit-identical to `FrameLinearization.apply_H` per camera).
  - Per camera × channel force b[c,ch] = H^T W r and normal A[c,ch](v), with additivity, symmetry and PSD checks against `measurement_operators`.
  - The grouped frozen system `G^T(Λ_prior + Σ A + lm I)G`, with float64 PCG and leave-one-camera / leave-one-channel solves, including the exact identity check `I_k = A⁻¹[b_k − A_k(δ_-k)]`.
  - Contributor traces from the production CSR (`pixel_contributions`: α, T, q, β, depths, d/σ, depth-edge distance, front contributors) and a reporting-only class.
- `tools/audit_phase1_wrong_camera.py`
  - Gate B: three replays (off, off, on) and parity against the measured GPU floor.
  - Gate A: forces, checks and leave-one-out at processing index 1, inner iteration 0, the pre-drift frozen state. Runaway and stable sets are chosen after the replay; traces and curvature along each selected Gaussian's own accepted Δμ.
  - Gate C (`--gate-c`): 13 labelled ablation arms plus the static reference, scored on all cameras with the unchanged observation model.
  - Refuses a model whose endpoint frame ≠ the start frame unless `--accept-endpoint-mismatch REASON`, which is recorded in the manifest.
  - Artefacts are `.npz` + `schemas.json` (pyarrow is not installed).

## Caveats built into the outputs

- The leave-one-out solves are the **unconstrained frozen quadratic**, not the production trust-region step. The float32 operators put a ~1e-5 floor on the true PCG residual.
- Removing RGB in the coupled expected-quad mode also removes alpha, because κ = trace(W)/12 of the zeroed RGB precision.
- Contributor classes (`MATCH_COMPATIBLE`, `BEHIND_FIRST_RETURN`, `IN_FRONT_CONTRADICTION`, `EDGE_OR_MIXED`, `INVALID_OR_OUTSIDE_OBJECT`) use 3σ and 3 px **for reporting only**; continuous d/σ is stored next to each label. Centre depth stands in for the splat's surface depth: an approximation.
- `source_camera_id` is `null`, because the bootstrap does not store seed provenance per Gaussian.
- `conclusion.md` is generated with the evidence tables. Hypothesis statuses are assigned by reading them, not computed.

## Tests (toy CUDA)

| Test | Result |
|---|---|
| `tests/test_wrong_camera_force_additivity.py` | PASS 12. Additivity: RHS 7e-8, normal 1e-7. Symmetry 2e-5; PSD; adjoint 4e-5; per-view == production; FD best-of-ladder; invalid-depth oracle. |
| `tests/test_wrong_camera_grouped_loo.py` | PASS 9. G adjoint; PCG == dense solve (5e-7); LOO arms == dense; exact identity 4.5e-6 (float32); near-singular oracle. |
| `tests/test_wrong_camera_contributor_parity.py` | PASS 5. Σq = A, Σβz = D̂; door/cavity classes; false front flagged; read-only. |
| `tests/test_wrong_camera_readonly_parity.py` | PASS 8. Debug-on == debug-off within 2× the floor of 3 control runs, same accept sequence. The tool runs end to end with every artefact and 13 ablation arms. Endpoint mismatch is refused. |

## Real data: NOT RUN

The only real fridge model is the frame-100 endpoint. The stationary test needs a frame-80 endpoint, so `~/run_phase1_gates.sh wcep80` builds it with the v4 settings and the v4 calibrated sensor model. Gate D (wooden box, synthetic stationary/moving controls) is also NOT RUN: the tool is forward-only and would need those endpoints.

## Follow-up: deeper analysis (2026-11-06)

All of it is opt-in or a separate tool. The production default is unchanged: the observer receives three extra keyword arguments (`background`, `visibility`, `observation_model`), and only when an observer is set.

### Deterministic reductions: `estimator/deterministic_ops.py`, `--deterministic`

- The 14 float `atomicAdd` sites are replaced by a two-pass int64 fixed-point accumulation:
  - `iekf.cu`: 5 sites (H^T, Fisher)
  - `iekf_group.cu`: 7 sites (group adjoint, group Fisher)
  - `iekf_visibility.cu`: 2 sites
- These are compiled as a separate JIT copy generated from the current production sources, and the site count is asserted.
- `enable()` swaps the six entry points in-process and turns on `torch.use_deterministic_algorithms`.
- Available as `--deterministic` in the wrong-camera audit, `audit_calibrated_phase1.py`, `phase0_geometry_consistency.py` and `run_motion_coherence_support.py`. In the driver it is `--phase1-deterministic`.
- Real fridge, 80–99 from the frame-80 endpoint: two replays are **bitwise identical** in every frame (μ, v, ω, P_post and the accept sequence), at 134 s per replay. The stationary drift is unchanged (frame-99 median 12.9 mm, p99 500 mm), so it is not a nondeterminism artefact.

### Deep first-update analyses: `estimator/wrong_camera_deep.py`, `--deep`

All three run at processing index 1 (80→81).

- `objective_conflict.json`
  - Each camera × channel force, and the prior force, projected on every selected Gaussian's accepted 6D pose change (and per supernode, against its least-squares rigid increment).
  - Sign convention: s > 0 pulls along the update, s < 0 opposes it.
  - Evaluated at inner iteration 0, where the prior exerts no force, and at the last inner iteration.
- `group_mediation.json`
  - The exact split I_c = A⁻¹b_c[own runaway rows] + A⁻¹b_c[their groupmates] + A⁻¹b_c[rest of scene] − A⁻¹A_c(δ_-c).
  - Computed in the canonical grouped system and in the per-Gaussian system, with the same frozen factors, prior and damping.
- `nonlinear_attribution.json`
  - Per camera × channel energy of the full production objective (true render) with all Gaussians, only the runaway set, only the stable set, their supernodes, or one supernode moved to x_post.
  - Includes the exact prior energy ½dᵀΛd of each move.
  - Self-checked against the IEKF's own initial and accepted energies.

### Phase-0 geometry consistency: `tools/phase0_geometry_consistency.py`

Every composited contributor is classified per camera as off-mask, in front, matched, behind, or no depth. The comparison uses σ_eff, which includes the sensor sd, the declared sd and the Gaussian's extent along the ray.

Repair rules (opt-in, written to a new model directory):

- `majority`: C > S.
- `unanimous`: contradicted in every camera that sees the Gaussian.

With START == END, the evidence comes from that single frame (a synthetic endpoint).

Results (CPU/GPU-light, evidence 60–80 on the frame-80 endpoint):

| Scene | Majority | Unanimous | Cross-view in front > 2 cm |
|---|---|---|---|
| fridge ep80 | 9952 / 29957 (33%) | 3012 (10%) | cam0→cam2 45%, cam2→cam0 38%, cam1→others 5–10% |
| fridge ep100 (80–100) | 10044 / 30316 (33%) | 3114 | the same pattern |
| USB (frame 59) | 29 / 30036 | 7 | 1–6% in every pair |
| washer (frame 59) | 14 / 29989 | 2 | median 2.6%, worst pairs 30–39% (geometry) |

Runaway Gaussians removed: 253/299 (majority) and 91/299 (unanimous). Stable Gaussians removed: 72/300 and 22/300.

### Cross-camera registration: `tools/cross_camera_registration.py`

- Robust point-to-plane ICP between cameras' back-projected observed depth (median over the stationary frames, object mask), starting from the calibrated cameras. No model is used.
- Each pair records its cycle consistency, T_BA·T_AB.
- On the fridge, only cam_000↔cam_002 gains clearly. The fraction within 1 cm goes 0.12→0.33 and 0.07→0.24, and within 2 cm 0.29→0.54 and 0.16→0.41.
- That pair's transform is 5.8° / 31 mm and 6.7° / 51 mm, with a cycle residual of 1.95° / 38 mm.
- The largest false gain over the 56 USB control pairs (perfect cameras) is +0.12 within 1 cm, with cycle residuals of 0.7–1.9° / 19–27 mm. Low-overlap pairs slide.
- So the evidence is that cam_000↔cam_002 is more inconsistent than any control pair, but **not** that the extrinsics are wrong by a single rigid transform.

### Tests

| Test | Result |
|---|---|
| `tests/test_deterministic_ops.py` | PASS 8: each reduction equals production (6e-8 to 1.5e-7 rel); bitwise repeat; 4-frame replay bitwise |
| `tests/test_wrong_camera_deep.py` | PASS 11: artefacts; x_post reproduced bitwise; the mediation split is exact (≤1.1e-5) grouped and ungrouped; the nonlinear energies equal the IEKF's initial and accepted energies; zero prior force at iteration 0 |
| `tests/test_cross_camera_registration.py` | PASS 4: back-projection inverts projection (3e-14); a 2° / 20 mm transform is recovered; identity |
