# Phase-1 occlusion fix: implementation report (Part I)

Spec: `RigEuler3DGS_PHASE1_COHERENCE_CODEX_IMPLEMENTATION_SPEC_2026-11-16.md`. Date: 2026-11-16.

## 0. Status

| Spec stage | Status |
|---|---|
| III.1 baseline audit | done (section 1) |
| I.A–I.C occlusion-consistent likelihood | **implemented and tested on CUDA**. The positive-ray ownership mixture is **not implemented** (section 3). |
| I.D observable grouped update | **implemented and tested**. Posterior is the block approximation, **uncertified** (section 5). |
| I.E observability export | implemented (per Gaussian, per original frame) |
| I.D gate on real scenes (forward/backward, against v6) | **NOT RUN**. The command is `run_phase1_occlusion_real.sh` (section 7). |
| Part II (held-out validation and calibrated null for coherence) | **not started**. The spec forbids it until the Part I gates pass on real scenes. |

## 1. Code identity

- Worktree `/home/samuelemara/Joint3DGS/RigEuler3DGS-occ`, branch `phase1-occlusion`, from `main` at `1f61c47`. The spec inspected `3bacfe5`, which is the parent of `1f61c47`. `1f61c47` adds docs and real-GT tools only; no Phase-1 source differs.
- Nothing is committed yet. Modified tracked files:
  - `estimator/masked_observation.py`
  - `estimator/sparse_iekf.py`
  - `run_bidirectional_all.sh`
  - `tools/kpart_pipeline.py`
  - `tools/run_motion_coherence_support.py`
  - `.gitignore`
- New files:
  - `estimator/{occlusion_likelihood,observable_group,negative_ray_operators}.py`
  - `submodules/diff-gaussian-rasterization/cuda_rasterizer/iekf_negative_ray.cu`
  - `tests/test_phase1_{occlusion_likelihood,observable_group}.py`
  - `tools/compare_phase1_occlusion_drift.py`
  - `run_phase1_occlusion_real.sh`
- Negative-ray kernel source hash (with `iekf_device.cuh` and `auxiliary.h`): `278733c2798870c96da4eb8a949f0e302b35abdcd108042c28aa8d4c3c766548`.
  - It is a separate JIT extension (`.build_negative_ray/<sha>`) that only reads the production `IEKFLinearizationCache`.
  - The installed `diff_gaussian_rasterization` is not rebuilt.
- `run_phase1_occlusion_real.sh` writes `provenance_*.txt`: commit, tracked-diff hash, and the hash of every untracked source file. The stage signature hashes only the tracked diff, so the provenance file covers the new modules.

### Flags (all opt-in; defaults reproduce v6)

| Flag | Default | Effect |
|---|---|---|
| `--phase1-likelihood legacy-masked\|occlusion-consistent` | `legacy-masked` | section 2 |
| `--negative-ray-reliability π` | 1.0 | declared, not calibrated |
| `--negative-ray-mask-edge unknown\|nonobject` | `unknown` | M=0 pixels touching M=1 |
| `--phase1-group-update legacy\|observable` | `legacy` | section 4 |
| `--phase1-group-update-map smooth\|rank` | `smooth` | section 4 |

`run_bidirectional_all.sh` takes `--phase1-likelihood` and `--phase1-group-update`. It adds `P1OCC=<lik>/<grp>` to the stage signature only when either is non-default, so existing signatures still match.

Refused combinations: `occlusion-consistent` together with any of
- `--object-mask-constraint`,
- `--first-return-mode`,
- `--rgb-support foreground`,
- an observation mode other than expected-quad.

### Legacy parity, measured

One grouped masked frame (52 Gaussians, 3 cameras, deterministic reductions) was run in `main` and in the worktree with the default flags. `mu`, `R`, `v`, `P` and every iteration energy are **bitwise identical**.

Existing suites in the worktree:

| Suite | Result |
|---|---|
| `test_masked_rgb_observation` | 46/46 |
| `test_visibility_constrained_iekf_phase1` | 12/12 |
| `test_deterministic_ops` | 8/8 |
| `test_motion_coherence_support` | 41/41 |
| `test_phase1_covariance_velocity` | 47/47 in 3 of 4 runs; 1 check failed once and did not reproduce |

The unreproduced failure looks like float-atomic ordering. The failing check was not captured, so it is unidentified. `main` passed 47/47 in one run.

## 2. Implemented likelihood (`--phase1-likelihood occlusion-consistent`)

Every pixel of every camera has exactly one status. The status is computed from the raw observation, never from the rendered blended depth.

| Status | Pixels | Term |
|---|---|---|
| OBJECT_SUPPORTED | M = 1 | v6 positive model restricted to M = 1: RGB fixed quadratic `I − C(x; b̄)`, alpha `½κ(1−A)²` with `κ = tr W/12`, blended-depth Student-t on `V_raw ∧ M` with the old alpha gate |
| KNOWN_NONOBJECT_DEPTH | M = 0, `V_raw`, not on the mask edge | `L_p = −log(1 − π A_p)` |
| UNKNOWN | M = 0 with invalid depth, or with an M = 1 pixel in its 3×3 neighbourhood | none |
| TRUSTED_EMPTY | — | never assigned |

The negative-ray quantity, over the renderer's own contributors k of pixel p (tile depth order, production `evalAlpha`, 0.99 clamp, T_STOP termination):

`A_p = Σ_k w_k Φ((D_p − z_k)/σ_p) = 1 − E_{Z∼N(D_p, σ_p²)}[T_front(Z)]`

- This is exact for the Gaussian depth marginal; no quadrature is used.
- σ_p is the calibrated sensor sd at the **observed** depth, plus the declared `depth_model_sigma` in variance, fixed per pixel.
- Positivity comes from the renderer itself: `1 − πA_p ≥ T_end ≥ 10⁻⁴`. A 1e-12 guard is counted (`log_floor_hits`) and was never hit in the tests.

Solver quantities, all from the same `L`:
- **Trial objective:** recomputed on every trust-region trial with fresh contributors.
- **RHS:** `−Σ (π/E_p) ∂A_p/∂x`, the exact score.
- **Normal operator:** `Σ (π/E_p)² J_pᵀJ_p`, Gauss–Newton (drops the `(π/E)∂²A` term; PSD).
- **Fisher and preconditioner blocks:** the per-Gaussian diagonal blocks of the same normal operator.

No pixel carries both a legacy and a negative-ray term: RGB precision and κ are exactly 0 off M = 1 (tested).

`PreparedRGBD` gains `depth_valid_raw` (`V_raw ∧ finite ∧ > 0` over the whole image). `depth_valid` (`V_raw ∧ M`) is unchanged, and only the new mode reads the raw field.

## 3. Not implemented (stated plainly)

- **Positive-ray RGB-D ownership mixture** (`π p_obj + (1−π) p_unknown`).
  - No calibrated `p_unknown` or independent ownership prior exists.
  - An end-effector inside M = 1 is therefore still explained as object, exactly as in v6.
  - First-return association on M = 1 is not used either; the blended depth is kept there.
- **Mask-edge mixed returns** are handled only by the 1-pixel UNKNOWN band. Flying pixels further out still count as non-object depth.
- **Mask reliability:** π defaults to 1 (every KNOWN_NONOBJECT ray certified). No calibrated false-negative rate exists.
- **TRUSTED_EMPTY:** never assigned.

## 4. Implemented grouped update (`--phase1-group-update observable`)

Per Gaussian i, at each linearization (frozen for that iteration):
- `S_i = chol(P_ee⁻)`, `F̃_i = S_iᵀ F_i S_i = U diag(λ) Uᵀ`.
- `Q_i = S_i U diag(q(λ)) Uᵀ S_i⁻¹`, with `q = λ/(1+λ)` (smooth) or a numerical-rank indicator with `rtol` 1e-6 (rank).
- `K_i = P_ve P_ee⁻¹`.
- `B_i = [I; K_i] Q_i G_i^pose`.

The solve is `Bᵀ(Λ⁻ + HᵀWH + λI)B du = Bᵀb`. The same B and Bᵀ are used in the RHS, operator, preconditioner, trust-region model and trial `x ⊞ B du`.

- `F_i` is the 6×6 measurement Fisher block of the full GN operator (RGB + alpha + depth + negative ray).
- The scale (prior pose covariance) and the rank tolerance are declared here; nothing was tuned.
- Restriction of the solution space: velocity is corrected only through the prior's conditional mean `K_i · δη_i`. When everything is observed and there are no groups, this equals the exact Kalman MAP (test [4], 2.9e-15).
- **Consequence to watch on real scenes:** members with no information (back faces, interiors, parts behind an occluder) now stay at the prediction instead of following their group. The legacy coherence selector averages supernode trajectories and may see a moving part's hidden side as static. That is the reason Part II has to use the observability export.

## 5. Covariance (not faked)

The posterior stays the block approximation `(Λ⁻ + F_ii)⁻¹`. A member with `F_ii = 0` keeps `P⁻` exactly, which is tested on CPU and on CUDA (max relative difference 7.4e-8). The approximation is **not** the Joseph covariance of the restricted update.

Dense oracle, 5 Gaussians in 2 groups, genuine pose–velocity correlation:

| Member | ‖block − Joseph‖/‖Joseph‖ | pose-trace ratio, block/Joseph |
|---|---|---|
| 0 (observed) | 0.000 | 1.000 |
| 1 (hidden) | 0.000 | 1.000 (exact) |
| 2 (observed) | 0.738 | 0.013 |
| 3 (observed) | 0.111 | 0.913 |
| 4 (hidden) | 0.000 | 1.000 (exact) |

The block posterior is overconfident for observed members whose group restriction discards information (member 2: about 77× too small). The posterior is labelled uncertified in every frame's diagnostics (`posterior_note`). Making a scalable covariance certifiable is open.

## 6. Tests

### `tests/test_phase1_occlusion_likelihood.py`: 25/25 (real CUDA, sm_89)

| Check | Result |
|---|---|
| kernel `A_p` vs CPU reference over dumped contributor lists | max diff 0.0 |
| analytic ∂A/∂α, ∂A/∂z vs FD (frozen branches) | 3.1e-10 |
| JVP vs FD of `A_p` | rel L2 1.2e-4 over 26 646 pixels (13 pixels that changed contributor set excluded) |
| energy FD vs `(π/E)·J v` | 446.317 vs 446.237 (rel 1.8e-4) |
| adjoint `<Jv,w> = <v,Jᵀw>` | 8.390088 vs 8.390096 |
| normal operator | symmetric (rel < 1e-6) and PSD |
| Fisher blocks vs normal-operator diagonal | 4.9e-8 |
| occluder 0.30 m in front (15σ) | `A_p` < 1e-44 on every covered ray; score exactly 0 |
| legacy alpha term on the same hidden part | max force 2.66e3 (contrast) |
| front-occluder sweep 0.30 → 0.00 m | mean `A_p` 2e-46 → 2e-6 → 2e-3 → 0.02 → 0.09 → 0.21, monotone and continuous |
| Gaussian moved from inside the silhouette onto free-space rays | E 24.4 → 69.9 |
| M = 0 with invalid depth | all UNKNOWN, energy and score exactly 0; RGB precision and κ exactly 0 off M = 1 |
| black / bright / background-grey Gaussians | identical negative energy (206.758914); opacity 0.85 → 0.1 gives 206.8 → 39.5 |
| covered in camera 0, visible in cameras 1–2 | 40/40 Gaussians keep nonzero Fisher |
| deterministic VJP / Fisher | bitwise repeatable; 3.8e-5 relative from the float-atomic path |

### `tests/test_phase1_observable_group.py`: 26/26

| Check | Result |
|---|---|
| CPU: adjoint `<Bu,y> = <u,Bᵀy>`; matrix-free `BᵀAB` = dense | holds |
| CPU: hidden member of a 2-member group | update exactly 0 (pose and velocity); legacy moves it by 0.043 |
| CPU: rank-1 measurement | update along `P_ee a` only (\|cos\| = 1 to 1e-12) |
| CPU: ungrouped full rank | equals the Kalman MAP |
| CPU: zero information | mean = `x⁻`, block posterior = Joseph = `P⁻` |

CUDA scenario: a visible part moves 12 mm while a static part sits behind a front occluder in all 3 cameras, in one supernode.

| Mode | hidden part displacement | visible part error |
|---|---|---|
| legacy | 11.38 mm | 12.00 → 0.54 mm |
| new | 0.000 mm | 12.00 → 0.58 mm |

Also in the CUDA scenario:
- Accepted steps strictly lower the new objective (674.29 → 454.08 → 453.60 → 453.60).
- Hidden members have zero whitened information and keep `P⁻`.
- The real `GroupedObservation` adjoint holds.
- Deterministic mode gives bitwise-identical `mu`, `R`, `P` across two frame solves.

## 7. Real scenes: NOT RUN (to be launched by the user)

`run_phase1_occlusion_real.sh`, from the worktree, runs:
- **A:** Phases 1–3 with the new Phase 1 (deterministic) on
  - robot_dog, black_laptop, black_box and wooden_box (defective);
  - light_blue_box and red_fridge (controls).
  
  Same endpoints, cameras, sensor model and windows as the 2026-11-15 overnight run.
- **R:** a Phase-1-only repeat (bitwise repeatability).
- **D:** CPU comparisons:
  - occlusion drift of the same static Gaussians while covered, for v6 / draw1 / occ1 (`tools/compare_phase1_occlusion_drift.py`);
  - supports, for v6 / draw1 / draw2 / occ1;
  - repeatability.

Per-frame observability is exported to `phase1_observability/obs_<frame>.npz`:
- `lam_max`, `lam_sum`, `q_max`;
- innovation and process translation, velocity;
- supported / front / occluded / unknown weights and pixel counts.

`frames.jsonl` holds stop reasons, accepted/rejected counts, ray-status counts per camera, and prediction-only member counts.

Baseline for the comparison (v6, wooden_box, same static set): median step 0.43 mm when seen vs 3.67 mm when covered; static Gaussians that were ever covered ended 210 mm from their start, against 39 mm for never-covered ones.

**Do not read success into the result if only the static drift drops.** The spec's gate also asks that moving-part tracking be kept (Phase-3 joint and support against v6).
