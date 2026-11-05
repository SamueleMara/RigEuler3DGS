# Phase-1 first-return depth likelihood: implementation report

Contract: `RigEuler3DGS_CODEX_Phase1_FirstReturn_Implementation_2026-11-05.md` and its mathematical companion.
Branch `phase1-first-return`, based on `phase1-covariance-velocity` (`21fb1c7`, which is based on `90dbf58`).
Default behaviour is unchanged: `--first-return-mode off`.

## 1. What is implemented

| Gate | Status | Content |
|---|---|---|
| A | **PASS** (21 checks) | CPU oracle copied verbatim (`estimator/first_return_reference.py`); strict, fail-closed prior schema (`estimator/first_return_types.py`): content hash, units, provenance, per-camera support, constant hazard only (a piecewise hazard is refused, not approximated), no defaults. |
| B | **PASS** (toy) + real-data audit run | Full-ray contributor dump. Composited with the production T_STOP rule it reproduces the production RGB / A / Z/(A+eps) (toy 3e-7; real fridge 30k Gaussians 6e-7). The list continues past T_STOP. Read-only audit tool `observation_diagnostics/first_return_audit.py audit`. |
| C | **PASS** (14 checks) | CUDA kernels `submodules/diff-gaussian-rasterization/cuda_rasterizer/iekf_first_return.cu`, a separate JIT extension that only reads the production `IEKFLinearizationCache` (production `_C`/`_C_vis` never rebuilt). Forward log L, exact score (VJP), JVP, B_vis Fisher blocks, debug dump. CUDA log L = float64 oracle (toy 2.5e-7, real 3e-8). Score = FD (translation 5.8e-4, rotation 2.3e-4). Adjoint identities ≤ 4e-7. Fisher blocks = explicit Σ g gᵀ. B is PSD. Invalid pixels have exactly zero effect. |
| D | **PASS as a diagnostic/experimental mode** (9 + 8 checks) | `off` = legacy; `audit` = legacy update + per-frame first-return summary; `depth` = blended depth removed from objective, RHS, normal, Fisher and posterior, replaced by the first-return likelihood; legacy RGB + alpha kept for the A/B. Every accepted step lowers the full nonlinear objective at a fresh render. The posterior is labelled `POSTERIOR_UNCERTIFIED` (see section 3). Toy physics: hidden cavity gets no direct depth or visibility score (1e-13 vs 36); a false front lowers log L by exactly Σ log(1-α) (error < 1e-3); unmodelled surfaces give exactly zero score; with λ_U = 0 a deeper return is impossible (log L −2808 vs −5.4). |
| E | **NOT IMPLEMENTED** | Joint RGB-D association (same latent hit for colour) and removal of the duplicate legacy RGB/alpha evidence. `--first-return-mode rgbd` is refused. |
| F | **NOT RUN** | The A/B runs are long; the commands are in `~/run_phase1_gates.sh` (`frstatic`, `frmotion`, `frfull`, `frsynth`). |

## 2. Model as implemented (and deviations from the contract)

- **Contributors.** Production `evalAlpha` (skip α < 1/255, clamp 0.99), the tile's depth order, the FULL list (no T_STOP), centre depth z = `cache.depth`. Only z in (z_near, z_far] are first-hit candidates. A contributor outside the support is excluded from both the value and the derivatives.
- **σ_D.** The calibrated (or configured) sensor sd **at the observed depth D** plus the declared `--depth-model-sigma` in variance, fixed per pixel. The contract writes σ_D(z_i). Evaluating at D makes the declared model variance-free, so the score is exact (contract 3.1 allows this when declared). The unknown branch uses the same σ (`"sensor"`) or a declared constant.
- **Eligibility.** `PreparedRGBD.depth_valid` (V_orig AND M, finite, > 0) AND D ∈ [z_near, z_far]. No background depth is read anywhere.
- **Local curvature.** B = B_match + B_vis. B_match = Σ γ_i/σ² J_zᵀJ_z (block-diagonal translation). B_vis = Σ_p g_p g_pᵀ, applied as VJP(JVP(v)), so cross-Gaussian ray coupling is kept. The same B gives the preconditioner and posterior blocks; no blended-depth Fisher is used in depth mode.
- **Front-contradiction diagnostic.** Σ_i −log(1 − α_i Φ((D − z_i)/σ_D)) per pixel and per Gaussian, using the contract's smooth formula with s_vis = σ_D. A Gaussian lying on the observed surface contributes −log(1 − α/2), so the diagnostic is not purely "false front". It is logged only; it never enters the objective (the survival factor already carries the contradiction).
- **λ_U.** `first_return_audit.py calibrate` gives a per-camera MLE of the fixed canonical render on a window the caller asserts is stationary (status `calibrated`, the full likelihood curve as provenance). `assume` writes an explicitly assumed prior with a recorded reason. z_near = 0.2 m is the renderer's own near-plane cull; z_far is declared.
- **File layout.** The contract's `first_return_likelihood.py`, `first_return_association.py` and the `.h` / binding files are merged into `first_return_types.py`, `first_return_observation.py`, `first_return_operators.py` and one `.cu` with its own pybind module. The responsibilities are the same.

## 3. Open blockers (stated, not hidden)

1. **Posterior.** The missing-association correction (−Cov_γ of component scores) is NOT implemented. B_vis is an empirical-Fisher PSD approximation, and no coverage test exists. Every depth-mode posterior is labelled `POSTERIOR_UNCERTIFIED` in the IEKF `posterior` row and the Phase-1 per-frame records. Phase 2/3 do not read Phase-1 covariance; only the positions are passed on.
2. **Gate E** (joint RGB-D, duplicate-evidence removal) not implemented.
3. **Piecewise hazard, invalid-depth (V=0) branch, registration/exposure footprint**: not implemented; refused or ignored explicitly.
4. **Convergence**: unchanged. The 12-iteration budget still ends most frames, and these are reported as `UNCERTIFIED_NOT_CONVERGED` in non-depth modes.

## 4. Real-data evidence so far (read-only; no tracking)

Prior: `ours_v5_phase1_gates/final_red_fridge_episode_000000/first_return_prior_60_79.json` (model_id `26c95fa7cf63`), MLE on stationary frames 60–79, declared depth mismatch 3 mm. λ_U = 0.046 / 0.022 / 0.215 m⁻¹ (cam_000/001/002), none at the grid edge.

Canonical (frame-100) model against frame f:

| frame | unknown-dominant pixels cam_000 / 001 / 002 |
|---|---|
| 90 (static) | 0.03 / 0.01 / 0.12 |
| 119 | 0.03 / 0.02 / 0.12 |
| 133 | 0.19 / 0.49 / 0.31 |
| 145 | 0.26 / 0.68 / 0.39 |
| 160 | 0.30 / 0.73 / 0.48 |
| 190 | 0.22 / 0.78 / 0.54 |

The same Gaussian ids top the front-contradiction ranking at every frame, including the static one (29947, 20074, 29452, 17474, …). These are persistent canonical-geometry defects, i.e. Phase-0 repair candidates, not motion.

## 5. Tests

| Test | Result |
|---|---|
| `tests/test_first_return_probability.py` | PASS 21 |
| `tests/test_first_return_cuda_parity.py` | PASS 14 |
| `tests/test_first_return_iekf.py` | PASS 9 |
| `tests/test_first_return_occlusion.py` | PASS 8 |
| `test_phase1_covariance_velocity` | PASS 47 |
| `test_surgical_review_4c88725` · `test_calibrated_bootstrap` | PASS 46 · 49 |
| `test_masked_rgb_observation` | PASS 46 on re-run. One run failed `[G] Phase 2 rigid SE(3) adjoint` with a relative gap of 1.2e-4 (re-runs: 4e-6, 9e-7, 1e-7). Random float32 inputs; no file it exercises changed on this branch. Existing intermittent failure, reported. |
| `test_masked_default_repair` · `test_v2_group_cuda` · `test_v2_grouped_solve` | PASS 77 · 26 · 16 |
| `test_motion_coherence_support` · `test_iekf_end_to_end` · `test_surgical_fixes_2026_10_22` | PASS 41 · 26 · 40 |
| Gate F A/B (real 80–99, 100–105, 100–288; USB) | NOT RUN |
