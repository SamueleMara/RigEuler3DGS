# Masked RGB and foreground-valid depth in the IEKFs: implementation (2026-10-20)

> **Update 2026-10-21.** `masked-rgb-expected-quad` is now the DEFAULT of every motion entry point
> (`original` must be passed explicitly), and Phase 2/3 MAP acceptance, the Phase 1 covariance
> recursion, the Phase 1 convergence test, depth validity and stage reuse were repaired. See
> [MASKED_IEKF_DEFAULT_REPAIR_REPORT_2026_10_21.md](MASKED_IEKF_DEFAULT_REPAIR_REPORT_2026_10_21.md).

Specification: `CODEX_IMPLEMENT_EXACT_MASKED_RGB_VALID_DEPTH_IEKF_2026-10-20.md`.
Audit: `docs/MASKED_RGB_FOREGROUND_DEPTH_AUDIT_2026_10_20.md`.
Parent commit: `d2343f7`.

## Objective implemented

All quantities are per camera c, frame t and native pixel p.
M is the whole-object mask; I and d are the recorded RGB and depth.

```
P = M I                     Y(b) = P + (1 - M) b                  (prepare / PreparedRGBD.target)
C(x;b) = F(x) + (1 - A(x)) b                                      (renderer, background b)
V_used = V_orig AND M AND finite(d)                               (prepare)
r_C = Y(b) - C(x;b)  over the whole image
r_D = d - D(x)       on V_used only (exactly 0 elsewhere; d never read there)
r_A = M - A(x)
```

### masked-rgb-expected-quad

b is uniform in [0,1]³ with independent channels, so b̄ = (½, ½, ½) and Σ_b = I/12. The
objective is:

```
E = 1/2 r_C(b_bar)^T W r_C(b_bar) + 1/2 kappa (A - M)^2 + depth term + prior
kappa = trace(W Sigma_b) = trace(W)/12        (Phase 1: W = I/sigma^2 -> kappa = 1/(4 sigma^2) = 25)
```

- **Linearisation.** Gauss-Newton, using the same single state and one normal operator:

  ```
  normal = H_C^T W H_C + H_A^T K H_A + H_D^T W_D H_D (+ prior + damping)
  rhs    = H_C^T W r_C + H_A^T K r_A + H_D^T W_D r_D
  H_C    = H_C(b_bar) = H_F - b_bar H_A
  ```

- **Lifts.** Each stage lifts this with its own J_g: Phase 1 supernodes, Phase 2 rigid SE(3),
  Phase 3 fixed-joint displacement.
- **No QP and no extra renders.** The accumulated alpha A is the primal render's `v.alpha`.
- **Acceptance.** It re-evaluates the full objective, including the alpha term, at every trial.

### masked-rgb-constant

A single declared background b and no alpha term. This is the cheap ablation.

### RGB precision

The RGB precision is fixed in both modes:

- Phase 1: 1/σ², with σ = `IEKFConfig.rgb_sigma` = 0.1, which is the repository's configured
  value.
- Phase 2/3: 1/(σ² + σ²_model). This is the existing sensor plus model-error variance, estimated
  once at the canonical frame from the masked composite residual. It is used **without** the
  Student-t weight.

The Student-t IRLS of the original mode is residual-dependent. It cannot be carried through the
exact expectation, so in these modes it is not applied to RGB. Depth keeps its existing robust
precision and its pre-existing coverage gate `alpha > 0.05`; both are documented and unchanged.

The total is therefore **not** one quadratic likelihood. RGB is the exact expected quadratic
cost. Depth is the existing robust term, re-evaluated at every trial, as before. The artificial
background is a designed contrast cost, not an independent sensor: the Fisher and posterior are
conditional on this objective, not a calibrated sensor posterior.

## Files

| file | change |
|---|---|
| `estimator/masked_observation.py` | NEW. Modes, `PreparedRGBD`, `prepare`, strict mask loader (binary, native size, unique file, matched by camera and ORIGINAL frame), coverage-correct `area_downsample`, per-camera terms, energy terms, CLI, `from_args` |
| `estimator/renderer_bridge.py` | `FrameLinearization.apply_H_alpha / apply_HT_alpha / build_alpha_fisher`: full-ray `_C_vis` operators on the existing cache; T and `n_contrib` come from the primal render |
| `submodules/diff-gaussian-rasterization/cuda_rasterizer/iekf_visibility.cu` (+ `.h`, `iekf_visibility_points.cu`, `ext_vis.cpp`, `__init__.py`) | NEW `visFisherCUDA` / `iekf_vis_build_fisher`: per-Gaussian Σ_p w_p ∂A/∂θ ∂A/∂θᵀ, packed like `iekf_build_fisher`. It lives in `_C_vis` only; `_C` is not rebuilt (md5 `a53f7040…` unchanged) |
| `estimator/sparse_iekf.py` | `measurement_operators` (rhs, normal, Fisher in one place; the original path is the same arithmetic, moved). `observation_model=` on `evaluate_true_map_energy` / `iekf_frame_step` |
| `estimator/joint_renderer_bridge.py` | `joint_apply_H(..., alpha=True)` → (dC, dD, dA); `joint_apply_HT` accepts (z_C, z_D, z_A) |
| `estimator/joint_iekf.py` | `_masked_terms`; energy, rhs, normal and the exact 6×6 information carry the alpha row. `observation_model=` on `frame_step` / `run_window` |
| `estimator/rigid_part_runtime.py`, `estimator/phase2_model_comparison.py` | pass `observation_model` through |
| `estimator/phase2_shared.py` | `effective_precision_masked`, `weighted_energy_masked`, lazy `PreparedFrames`, `render_state(background=)` |
| `tools/kpart_pipeline.py` | Phase 1 bootstrap prepares each frame and calls the masked IEKF; per-frame energy terms are reported |
| `tools/run_motion_coherence_support.py`, `tools/run_rigid_se3_pipeline.py`, `tools/run_joint_phase3.py` | CLI; masked targets, background and model error; `masked_observation.json` |
| `run_bidirectional_all.sh` | `--observation-mode`, `--rgb-background-constant`, `--rgb-background-distribution` → Phase 1/2/3 only, with the per-scene `--mask-source`. Combining with `--object-mask-constraint` exits 2. Static training gets none of these switches (checked with dry runs) |
| `scripts/run_object_mask_ab.sh`, `tools/compare_object_masks.py` | superseded 2026-10-21: fair A/B, `--mode masked-rgb-expected-quad` (default) / `masked-rgb-constant` vs an explicit `original` arm, both fresh Phase 1/2/3 |
| `tools/benchmark_masked_rgb_observation.py` | NEW: benchmark L |
| `tests/test_masked_rgb_observation.py` | NEW: 46 checks |

The hard-constraint QP is kept unchanged and is not called by the new modes.
`estimator/silhouette.py` is not used.

## Tests (real CUDA renderer, RTX 4070)

`tests/test_masked_rgb_observation.py` passes **46/46**:

| spec test | result |
|---|---|
| A. Prepared RGB | M=1 gives I and M=0 gives b exactly. Masks match by camera and original frame in reverse order. Non-binary, wrong-size, missing and ambiguous masks are refused, as is a count mismatch |
| B. Resampling 2×2→1×1 | Matches the area mean of the native composite to 6e-8. Resizing I and M separately is off by up to 0.131 (the dark halo) |
| C. Depth outside the mask set to the wall value, 0, NaN, 1e6 or noise | Energy terms bitwise identical. RHS within 6e-8 (repeat floor 2e-8). No NaN. Accepted state within 2.5e-7 (floor 1.8e-7) |
| D. Compositing | C(b) = F + (1−A)b to 1.8e-7. H_C(b) − H_C(0) = −b H_A to 3.8e-8. RGB JVP vs central FD with b ≠ 0, translation and rotation: 1.3e-4 |
| E. Exact expectation vs 8-point cubature | Energy to 2e-8 relative, for diagonal W and cross-channel W (κ = trace W/12). Gradient 1.7e-7. Normal operator 1.7e-7 |
| F. Alpha | A = 1 − T. Full-ray A equals `v.alpha` (2.4e-7) with the same T and n_contrib. FD 1.0e-4 (translation), 1.6e-4 (rotation). Adjoint 2.2e-7. New alpha Fisher equals explicit JᵀWJ to 1.9e-7 and is PSD |
| G. Stage lifts | RGB / depth / alpha JVP vs FD of the whole stage map: Phase 1 supernodes ≤ 2.4e-4, Phase 2 rigid ≤ 2.7e-4, Phase 3 displacement ≤ 2.4e-4. Adjoint with the alpha row ≤ 8.8e-7 |
| H. Dense check | normal(v) matches J_Cᵀ W J_C + J_Aᵀ K J_A + J_Dᵀ W_D J_D to 1.3e-6. RHS 3.6e-7. Fisher equals the diagonal blocks (6.8e-7). κ = 25 = 1/(4σ²) |
| I. Acceptance | A candidate that lowers RGB (545 → 350) but raises alpha (268 → 810) has total 813 → 1160 and is rejected. The objective is reproducible bitwise. Rejected trials leave it unchanged (4 accepted / 4 rejected) |
| K. Instrumented run | No QP, constraint step or visible-alpha forward is called. Exactly one render per camera per nonlinear evaluation (27 = 9 × 3). Silhouette not imported |
| M. Flags | Hard constraint + masked mode is refused, as are a missing or misplaced background option and `iekf_frame_step(observation_model, visibility)`. Driver: exit 2 |
| N. Identifiability | An isolated grey Gaussian in a uniform negative region has alpha cost 273 and zero RGB gradient. Its gradient is identical with the object mask removed (2.9e-6): no attraction toward the mask. What remains is a "shrink the footprint" push along the ray (63) and a perspective term in the image plane (7.7) |
| P. End to end, toy | Phase 1 with the object lagging over a textured background: centre error 0.0120 → **0.0102**, against **0.0884** in the original mode. Alpha on negative pixels 99.6 → 83.2, against 638.7 in the original mode. Phase 2 rigid window: 0.148 rad (true 0.15). Phase 3 fixed prismatic: 0.004 → 0.0136 m (true 0.015), geometry bitwise unchanged |

**J. Original mode unchanged.** `sparse_iekf.iekf_frame_step` was run from HEAD and from this
patch on the same frame, three times each. The maximum state/covariance difference was:

| comparison | maximum difference |
|---|---|
| HEAD vs patch | 1.16e-6 |
| HEAD vs HEAD | up to 1.33e-6 |
| patch vs patch | up to 1.06e-6 |

So the patch is inside the CUDA run-to-run floor.

Other suites, unchanged and passing:

| suite | result |
|---|---|
| process covariance | 24 |
| rigid SE(3) | 47 |
| parallel P/R | 37 |
| grouped solve | 16 |
| end to end | 25 (the count varies 25–27 between runs, all pass) |
| full-scale trust region | 35 |
| constrained QP | 21 |
| background audit | 11 |
| visible-alpha operators | 12 |
| hard-constraint Phase 1 | 12 |
| hard-constraint Phase 2/3 | 8 |

`test_joint_phase3_runner` reported **NOT_RUN** this time: it reads
`results_11_10_26/ours/fridge_10905` on the external SSD, which was not mounted.

**L. Full-scene benchmark: NOT RUN.** The external SSD holding the datasets and static models
was unmounted during this session, so no real-scene timing, memory or leakage number is claimed. `tools/benchmark_masked_rgb_observation.py` has only been syntax- and
CLI-checked; its first real run is that benchmark.
The command is below. Not measured as a result:

- the earlier 14 s (original) and 146 s (hard constraint);
- the cost of the new mode;
- the 64-camera run.

Expected cost per IEKF iteration over the original mode: one alpha JVP and one alpha VJP per
normal-operator application, plus one alpha Fisher per iteration. Measured on the hard-constraint
smoke: about 13 ms + 9 ms per 8-camera application, against about 60 ms for the RGB-D operator.

**M. Scope.** No change to:

- `train.py`, `gaussian_renderer/`, `scene/`, `utils/`;
- the training rasterizer sources and `_C`;
- the GT evaluators;
- any data, checkpoint, PLY or result directory.

## Remaining unobservable cases

- A Gaussian that no camera constrains keeps the constant-velocity prediction, for example one
  20 cm behind the visible surface, or one seen by a single camera along its own ray. On the real
  `fridge_plastic` run this, not the background, explains the half of the dispersion that remained
  with a mask-only likelihood.
- An isolated Gaussian fully inside a negative region has a background cost but no in-image
  direction toward its true place (test N). This mode detects nothing and invents nothing for it.
- The masks are whole-object observations. A wrong mask (SAM) is a wrong measurement.

## Commands

```bash
cd ~/Joint3DGS/RigEuler3DGS && conda activate rigeuler3dgs
# benchmark L (2 frames, 8 cameras, 12 iterations; original vs expected-quad)
python tools/benchmark_masked_rgb_observation.py \
  --model "/media/samuelemara/Extreme SSD/Samuele/results_11_10_26/ours/fridge_10905/static_bidi/initial" \
  --source "/media/samuelemara/Extreme SSD/Samuele/Backup_09_10_26/Sapien/output/preparation_8cams/partnet_subset_fixed_render_linear/fridge_10905" \
  --out "/media/samuelemara/Extreme SSD/Samuele/results_11_10_26/benchmark_masked_rgb_20261020"
# all synthetic scenes, A/B with the comparison table
bash scripts/run_object_mask_ab.sh
```
