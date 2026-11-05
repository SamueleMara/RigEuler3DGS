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
