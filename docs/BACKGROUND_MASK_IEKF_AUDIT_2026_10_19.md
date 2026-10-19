# Background compositing and object-mask visibility in the IEKFs — code audit (2026-10-19)

Scope: the `main` checkout at `6093f0b` plus the uncommitted patch this audit justifies.
Line numbers refer to the files as they stand in that patch. The audit was done
first, and its numerical tests were run on the **installed** renderer
(`_C.cpython-311-x86_64-linux-gnu.so`, md5 `a53f7040…`) before anything was
changed.

Theory: *Bidirectional_Articulated_3DGS_IEKF_Background_Visibility_Updated_2026-10-19*,
section "Exact background, masks and occlusion for the RGB-D IEKF".

## Summary

| question | finding | action |
|---|---|---|
| 1. RGB model | `C = Σ T_i α_i c_i + T_end B`. `B` is one constant RGB vector per render, black by default in every dynamic stage. | none |
| 2. B in JVP / VJP / Fisher | Present and correct in all three. Proven numerically on four backgrounds (FD 1.1–1.5e-4; `H_C(B) − H_C(0) = −B dA` to 7e-6). | **not rewritten** |
| 3. What the stages fit | Full-frame RGB plus depth where `z > 0`. No mask is used anywhere in Phase 1/2/3. Masks are whole-object silhouettes for all 8 cameras × all frames. | new opt-in constraint |
| 4. Camouflage | Yes. Off the object the synthetic RGB is exactly black, `B = 0` and depth is invalid, so a dark Gaussian outside the mask leaves no residual at all. | new opt-in constraint |
| 5. Shared linearization in Phase 2/3 | Yes. The same `build_linearization` is used and the lifts carry both the translation and rotation columns. | constraint lifted through them |
| 6. Model-dependent gates | Pre-existing: the depth gate `alpha > min_alpha_for_depth` and the Student-t weights are rebuilt at each linearization. The new mask rays are frozen per frame. | reported, off path untouched |
| 7. Acceptance / zips | A full nonlinear re-render decides acceptance (gain ratio). Feasibility was never checked before. Some non-strict `zip`s remain. | audit added, guards added |

## 1. RGB prediction and the background

- **Forward pass:** `submodules/diff-gaussian-rasterization/cuda_rasterizer/iekf.cu:384`
  computes `out_color = C[ch] + T * bg[ch]`, where `T` is the transmittance after
  the last contributor.
  - The front-to-back walk stops when `test_T < T_STOP` (`iekf.cu:369`, with
    `T_STOP = 1e-4` in `iekf_device.cuh:19`).
  - Alpha is clamped to `ALPHA_MAX = 0.99` and contributions below
    `ALPHA_MIN = 1/255` are dropped (`iekf_device.cuh:17-18,55-56`).
  - These are the same constants as the training rasterizer
    (`forward.cu:372,414`).
- **`B` is a constant RGB vector per render.** It is passed as a `(3,)` tensor
  (`estimator/renderer_bridge.py:129,146-153`), defaults to **black**
  (`renderer_bridge.py:147`), and is never per pixel or per camera.
- None of the dynamic runners pass a background:
  - `tools/run_motion_coherence_support.py`, `tools/kpart_pipeline.py`,
    `tools/run_rigid_se3_pipeline.py`, `tools/run_joint_phase3.py`,
    `estimator/joint_iekf.py`;
  - `estimator/phase2_shared.py:107` calls `build_linearization` without one.
- So every dynamic stage composites over `B = (0,0,0)`.
- Depth is normalised: `D = Z / (A + eps_depth)`, with `eps_depth = 1e-6`
  (`renderer_bridge.py`).

## 2. Is `B·δT_end` in the JVP, the VJP and the Fisher? Yes, in all three

Where it is in the code:

- **JVP** (`iekf.cu:399-495`):
  - the recurrence is `dw = dT·α + T·dα` and `dT ← dT(1−α) − T·dα`
    (`iekf.cu:477,483`);
  - the output is `dC + dT·bg` (`iekf.cu:492`).
  - This is exactly `dC = Σ (c_i − B) dw_i`, since `dT_end = −Σ dw_i`.
- **VJP** (`iekf.cu:510-660`): the suffix colour accumulator starts at the
  background, `float B[3] = {bg…}` (`iekf.cu:565`). Each contributor's adjoint
  therefore sees `c_i − (suffix including B)`. This is the transpose of the JVP
  recurrence.
- **Fisher** (`iekf.cu:662-…`, launched at `iekf.cu:936-955`) has the same suffix
  initialisation (`iekf.cu:713`). Its 4×6 per-contributor block is built from the
  same quantities.

Proof: `tests/test_background_compositing_audit.py` ran on the real CUDA renderer,
before any change. The scene has rotated anisotropic overlapping splats, arbitrary
colours, and rays that reach the transmittance cut-off.

- **[1] JVP vs central FD.** The contributor lists are frozen; the 81 pixels whose
  contributor set changes inside the FD stencil are excluded and reported.

  | background B | relative error | pixels |
  |---|---|---|
  | (0,0,0) | 1.13e-4 | 348 |
  | (1,1,1) | 1.54e-4 | 348 |
  | (0.3,0.7,0.1) | 1.16e-4 | 348 |
  | (0.9,0.2,0.55) | 1.38e-4 | 348 |

  The error does not depend on `B`. A missing `B·δT` term would grow with `|B|`.
- **[2] Background identity.** `H_C(B) − H_C(0) = −B·dA` holds to 7e-6 against
  `|H_C| = 18.3`.
- **[3] Splat of the background's own colour.** The RGB JVP is 4e-7 even where
  alpha reaches 0.83. This is the camouflage case, and it is correct physics,
  not a missing term.
- **[4] Adjoint identity.** `⟨Hv, z⟩ = ⟨v, Hᵀz⟩` to 2.0e-10 (normalised).
- **[5] Fisher.** Each block matches `J_iᵀ W J_i` from explicit JVP columns to
  4e-7 (relative).

The renderer's background derivative is therefore **correct and was not
rewritten**. The earlier black-on-white auxiliary experiment matched an alpha FD
because the term is there.

One part of `iekf.cu` did change: shared device helpers were moved, unchanged,
into `cuda_rasterizer/iekf_device.cuh`, so the new visibility kernels compile
the same code. The multiset line diff of the move contains only additions of the
`#include`.

Parity was checked with `test_background_compositing_audit.py
--save-reference` / `--check-reference`. `_C` was rebuilt from the patched
source (CUDA 12.8, sm_89) and compared with the installed binary (md5
`a53f7040…`):

| outputs | rebuilt vs installed | run-to-run spread of the installed binary |
|---|---|---|
| render, JVP (4 backgrounds) | bitwise identical | — |
| VJP, Fisher | at most 1.8e-7 (relative) | the same 1.8e-7 |

The VJP and Fisher use atomic accumulation, which is why they are not bitwise
reproducible even on one binary.

While doing this, two unseeded inputs in the reference test were fixed: the
VJP adjoint and the Fisher precisions had been unseeded random draws, so they
were never comparable. The installed binary was put back afterwards, because
the rebuild also recompiles the static-training kernels and nothing in this
patch needs it. The new operators live in the separate `_C_vis`.

## 3. What the dynamic stages fit, and what the masks are

**Observation loader:** `tools/run_iekf_scene.py:236`, `dataset_rgbd_frames`.

- It reads the full-frame RGB from `rgb/<cam>/frame_%06d.png` and the metric depth
  from `depth_npy/<cam>/frame_%06d.npy`.
- Depth validity is `valid = dz > 0` (`run_iekf_scene.py:302`).
- The likelihood is in `estimator/sparse_iekf.py:234`, `_precisions`:
  - RGB has a precision on **every pixel** (`valid_mask=None`);
  - depth has a precision only where `depth_valid & (alpha > min_alpha_for_depth)`.
- Phase 2/3 use the same construction (`tools/kpart_pipeline.py:149-155,
  251-259,999-1009`; the joint stages through `precision_fn`).
- **No stage reads `mask/`.** Training uses masks: masked static 3DGS, background
  carving (1189024). The motion stages did not.

**Masks on disk** (`<scene>/mask/<cam>/frame_%06d.png`):

- binary 0/255, at full resolution;
- one per camera and frame (8 cameras × 60 frames on `fridge_10905`);
- they are the silhouette of the **whole object**, not of the moving part.

On the SAPIEN exports, depth is positive exactly inside the mask and zero outside
it (`fridge_10905` cam_000 frame 0: 100 % / 0 %), and the RGB outside the mask
is exactly `(0,0,0)`, with zero standard deviation. These scenes have no external
occluder, so the full-empty-ray semantics (`trusted-full-ray`) is the declared,
correct one for them. For real captures the masks come from the user's SAM
pipeline; the semantics must then be declared, and nothing is inferred.

## 4. Can a Gaussian outside the mask camouflage itself? Yes

Outside the mask the synthetic observation is black, the model background `B` is
black, and depth precision is zero (`dz = 0`).

- A **dark** Gaussian there has a zero RGB residual and contributes nothing to the
  depth term. It is invisible to the likelihood, so the IEKF has no reason to
  move it.
- A **coloured** Gaussian there is pulled only by its RGB residual, which is
  fitted equally well by shrinking its colour contribution as by moving it. The
  colour is not in the state, so the filter can move it in any direction that
  lowers the residual locally, including farther out.

This is the "Gaussians stuck outside the mask" behaviour. Invalid depth pixels
correctly carry no depth information (zero precision, never a fake value,
`sparse_iekf.py:254-262`).

## 5. Do Phase 2/3 share the Phase 1 linearization? Yes

- **Rigid Phase 2** renders through `phase2_shared.render_state` →
  `build_linearization` (`phase2_shared.py:92-107`).
- **Joint Phase 3** (`estimator/joint_iekf.py:150-160`) uses the same `lin.views`.
- The stage lifts map the solved state onto the per-Gaussian 6-column pose
  tangent: translation and **rotation**, the same convention as the renderer.
  - `RigidRuntimeState.jvp/vjp` in `estimator/rigid_part_runtime.py:101-115`
    use `rigid_jvp`/`rigid_vjp` on the member Gaussians.
  - `phase2_model_comparison.py:72-81`.
  - `JointDisplacementState.jvp/vjp` in `estimator/joint_displacement.py:59-63`
    carry only the q column.
- Phase 1 is free per-Gaussian (`expand_step`/`extract_pose`).
- So the orientation derivative reaches each stage.
- The FD of the lifted visible-alpha operator through each lift is about 1e-4
  (`tests/test_visibility_constrained_iekf_phase23.py`).

## 6. Fixed vs model-dependent gates

- **Raw observations.** RGB, depth and `dz > 0` validity are immutable.
- **Pre-existing, model-dependent:**
  - the depth coverage gate `alpha > min_alpha_for_depth`
    (`sparse_iekf.py:259`; same in `kpart_pipeline.py:153,257,1007`);
  - the Student-t robust weights (`student_t_nu = 5`), recomputed from the
    residual at every linearization.

  Both change the precision under the optimiser. They are part of the existing,
  documented IRLS design and are **not changed** by this patch: changing them
  would alter the off path, which must stay equal. They are flagged here as
  findings.
- **New, frozen per frame.** The negative-ray sets — `ray_mode`, `z_obs`, and the
  boundary band from the mask and measured depth — are frozen per frame
  (`estimator/visibility_constraint.py`, `ray_modes`/`FrameVisibility`).
- **Model-dependent in the new code:** only the physical visibility of the
  splats, which is re-evaluated at every trial state.
- The withdrawn prototype's gate `allowed = mask OR (d_obs < normalized model depth)`
  is not used.

## 7. Acceptance, feasibility, `zip`

**Acceptance.**

- The trust-region gain ratio uses the **true nonlinear energy** of a fresh
  render (`evaluate_true_map_energy`, `sparse_iekf.py:311`; trial at
  `sparse_iekf.py:674,694`). The joint stages do the same with
  `_energy(linearize, …)` (`joint_iekf.py:150`).
- Before this patch no feasibility of any kind was checked.

**`zip` truncation.**

- Pre-existing non-strict zips:
  - `sparse_iekf.py:243` (views × observations);
  - `joint_iekf.py:153,226` (views × observations × precisions).
- These are guarded on the constraint path by:
  - `iekf_frame_step` asserts that cameras, observations and masks have the same
    count;
  - `FrameVisibility.check_views` asserts the view count and every view's
    dimensions against its mask;
  - `joint_iekf.frame_step` checks the observation count;
  - `_object_valid`/`precision_fn` zip with `strict=True`.
- Spec test 9 exercises the guards.
- The off path is left exactly as it was (see Proof of no change).

## What was implemented (justified by 1–7)

- **Native visible-alpha operators.** A new separate extension `_C_vis` is built
  by `setup_vis.py`; the ABI of `_C` is untouched.
  - Kernels: `iekf_visibility.cu`. Bindings: `iekf_visibility_points.cu`,
    `ext_vis.cpp`. Python wrappers: `iekf_visible_alpha`, `iekf_vis_apply_H`,
    `iekf_vis_apply_HT` in `diff_gaussian_rasterization/__init__.py`.
  - The operators compute
    `A_vis = 1 − Π(1 − g_i α_i)` over the renderer's own contributor walk
    (same `T_STOP` termination, clamp and `ALPHA_MIN`), with the gate:
    - mode 1 (full-empty ray): `g = 1`, so `A_vis = A`;
    - mode 2 (front of the measured non-object surface): `g_i = 1[d_i < z_obs − δz]`,
      where `d_i` is **the splat's own camera depth**. This is a hard gate,
      frozen within a branch.
  - The VJP is `∂A/∂α_i = g_i·T_f,i·Π_{j>i}(1 − β_j)`.
  - No extra views, no CPU distance transform. One scalar row per ray, with no
    colour dependence.
  - **No alpha Fisher is built.** The constraint is a hard inequality, so
    `H_A` appears only in the QP constraint matrix.
- **Constrained QP.** `estimator/constrained_qp.py`, `solve_qp_projection`: an
  exact A-norm projection by constraint generation (a dual active-set method).
  - Every row in the working set carries `A⁻¹Cᵢᵀ` (one PCG solve with A's
    preconditioner); the working-set dual is solved exactly (NNLS on a Cholesky
    factor).
  - Every row is checked each round and the most-violated rows are added.
  - It reports `OPTIMAL`, `BUDGET_EXIT` or `INFEASIBLE_SUSPECTED`.
  - Rows are cached per linearization, so second-order-correction re-solves and
    the end-of-frame classification reuse them.
  - The augmented-Lagrangian solver (`solve_qp_alm`) is kept and tested but not
    used: on the 8-camera fridge its Newton systems `A + ρ C_Iᵀ C_I` took 400–3700
    CG iterations per QP, up to 1010 s, for 1–5 active rays.
  - The module also provides zero-row detection and a χ² data-conflict test.
- **Constrained IEKF update.** `estimator/visibility_constraint.py:constrained_step`.
  - The step is the A-norm projection of the trust-region step onto
    `A_vis + C_s δ ≤ ε` for every ray that was feasible at the frame start.
  - Rays already violated at the frame start get one aggregate row:
    `Σ_p C_p δ ≤ 0` (the total of the pre-existing violations may not grow).
  - Each trial is re-rendered and **audited over all negative rays**: no new
    violation, and the pre-existing total does not grow. A failed audit gets up to
    two second-order-correction re-solves, then the step is rejected (`ρ = NaN`).
  - With the constraint on, the RGB-D likelihood is restricted to mask = 1.
  - **Release of active rows.** If the projection's objective excess exceeds
    `½χ²₀.₉₉(k)`, holding the rows contradicts the RGB-D (a misowned Gaussian
    holding a rigid part back). The active rows are then released for the rest of
    the frame and reported as `MODEL_OWNERSHIP_INFEASIBLE`.
    - In Phase 2/3 the aggregate row can be released too.
    - In Phase 1 the aggregate row is never released. There the unconstrained step
      the test compares against is the background fit itself, and releasing the
      row let the pre-existing violation grow from 9.4 to 15.7 in spec test 2.
- **End-of-frame status per frame:** `FEASIBLE`, `MODEL_OWNERSHIP_INFEASIBLE`,
  `UNRESOLVED_INFEASIBLE` or `INDISTINGUISHABLE`. It is written to
  `visibility_constraint.json` beside each stage's outputs.
- **Covariance.** The exported covariance is still the unconstrained
  normal-operator inverse at the accepted state, and is labelled so
  (`posterior_note`). It is *not* claimed to be the truncated posterior.
- **Stages.** One shared `FrameVisibility`; the stage lift supplies `J_g`:
  - Phase 1: `sparse_iekf.iekf_frame_step(visibility=…)`, called from the
    `kpart_pipeline` bootstrap;
  - Phase 2: `joint_iekf.frame_step` via `rigid_part_runtime.run_rigid_window`;
  - Phase 3: `phase2_model_comparison.fit_model`.
  - Phase 3 only filters the per-frame displacement q. The joint geometry and type
    are fixed and are not touched.
- **CLI** (every runner; all forwarded by `run_bidirectional_all.sh`, and only
  when the mode is not off). The resolved policy is logged in each
  `visibility_constraint.json`.
  - `--object-mask-constraint {off,visible-negative-hard}`: default `off`.
  - `--mask-ray-semantics {trusted-full-ray,known-nonobject-depth,conservative}`.
  - `--mask-source`: the mask directory itself; defaults to `<source>/mask`;
    never inferred from GT files. The driver takes `--mask-subdir NAME` for
    `<scene>/NAME`.
  - `--mask-depth-noise-m`: **no default**. `known-nonobject-depth` is refused
    without it, and `δz = 3σ`.
  - `--mask-boundary-px`: the boundary policy; mask-boundary rays within N px are
    unconstrained, for antialiasing and Gaussian tails. Default 2.
  - `--mask-alpha-epsilon`: ε, the largest object alpha accepted on a negative ray,
    covering rasterizer tails. Default 0.05.
  - `conservative` constrains nothing and counts the rays.
- **Withdrawn.** The `--object-masks` silhouette prototype (fb7e7a5) is unhooked
  from all runners and from `run_bidirectional_all.sh`. `estimator/silhouette.py`
  is kept, marked as a withdrawn diagnostic.

## Tests (real CUDA renderer)

| suite | result |
|---|---|
| `test_background_compositing_audit.py` | 11/11 (section 2 above) |
| `test_visible_alpha_operators.py` | 12/12 |
| `test_constrained_qp.py` | 21/21 |
| `test_visibility_constrained_iekf_phase1.py` | 12/12 |
| `test_visibility_constrained_iekf_phase23.py` | 8/8 |

`test_visible_alpha_operators.py`:

- `A_vis` = renderer alpha to 3e-7, with the same last contributor;
- translation FD 1.3e-4, anisotropic-rotation FD 1.2e-4, mode-2 FD 1.4e-4;
- the front plate is counted and the plate behind it is not; composited depth is
  not used;
- overlap composes as `1 − (1−A_a)(1−A_b)` to 6e-8, where the plain sum would
  reach 1.68;
- black splats on black give identical `A`/`dA`;
- adjoint gaps: 8e-10 for alpha alone, 3e-10 with RGB-D.

`test_constrained_qp.py`:

- the projection solver = SLSQP to 1e-6–1e-14, KKT, including duplicated
  (parallel) rows; the row cache gives the same result with 0 new solves;
- ALM = SLSQP to 1e-8, KKT to 1e-8;
- zero rows are found exactly;
- an infeasible system reports `INFEASIBLE_SUSPECTED`;
- the χ² conflict test separates the two cases.

`test_visibility_constrained_iekf_phase1.py` (spec tests 1, 2, 3, 4, 9, 10):

- **Spec 1.** Detection is identical for black, bright and background-coloured
  strays (307 rays each).
- **Spec 2.** For an object that lags its observation over a textured background:
  - off: 1295 violating rays, mean error 0.088 m;
  - on: 168 violating rays, no new violation, pre-existing total
    9.40 → 9.39, error 0.0120 → 0.0110 m;
  - every accepted step passed the full nonlinear audit.
- **Spec 3.** A stranded Gaussian is not reported feasible, moved 0.2 mm, and
  created no new violation.
- **Spec 4.** Behind a measured occluder there is no penalty; in front, `A_vis` is
  0.998 (individual splat depth).
- **Spec 9.** Count mismatches are refused.

`test_visibility_constrained_iekf_phase23.py`:

- Phase 2 and Phase 3 lift FD is about 1e-4.
- Constrained rigid estimate 0.151 rad (true 0.15; off 0.161).
- **Spec 6.** A misowned stationary Gaussian does not hold the rigid part back:
  its rays are released and reported `MODEL_OWNERSHIP_INFEASIBLE`.
- The Phase 3 displacement still moves under the constraint.

Spec 5 (overlap) is in the operator suite.

Honest caveats from the toy scenes:

- **The constrained update is conservative.**
  - Phase 3 displacement s = 0.009 m (true 0.015, off 0.0165).
  - Phase 1 lag: error 0.0120 → 0.0110 m in one frame.
- **The χ² data-conflict test also fires in the clean rigid case (P2-b),**
  whose status is `MODEL_OWNERSHIP_INFEASIBLE`. Its bound assumes `A` is the
  calibrated information. With robust weights and a mask-restricted likelihood
  this is only approximately true, so on real runs read that status as "the
  constraint and the RGB-D disagreed", not as proof of misownership.

## Full-scale smoke (spec 8, partial)

Setup:

- scene: `fridge_10905`, forward, from `ours/fridge_10905/static_bidi/initial`
  (read only);
- 8 cameras, 640×480, 27,342 Gaussians in 427 supernodes (state dimension
  5124), 2.26 M negative-ray rows;
- 2 frames (one filtered frame), 12 IEKF iterations, which is the driver
  default;
- outputs in a scratch directory only.

Phase 1 on vs off:

| | constraint off | constraint on |
|---|---|---|
| wall time | 14 s | 146 s |
| violating negative rays after the frame | — | 18 (325 at the start) |
| pre-existing total violation | — | 242.15 → 3.67 |
| new violations | — | 0 |
| frame status | — | `MODEL_OWNERSHIP_INFEASIBLE` (the 18 remaining rays) |
| coherence extraction | RESOLVED: 18 moving-rigid, 19 static, 155 ambiguous supernodes | RESOLVED: 66 moving-rigid, 44 static, 82 ambiguous |

**Cost.** With the constraint on, Phase 1 costs about 10× more on frames with
active constraints.

- The first implementation used the augmented Lagrangian and took 1677 s on this
  frame, about 28 h for a 60-frame direction. The dual active-set solver brought
  that down to 146 s.
- Most of the remaining time is iterations whose trial fails the nonlinear audit.
  Each one gets two second-order re-solves, and those add new rows.

**Not run here:**

- the 64-camera stress test;
- Phase 2 and Phase 3 on the full scene (toy-scene tests only);
- the off-vs-on A/B for one object at the standard settings. These are long runs;
  the commands are in the README.

## Proof of no change with the constraint off (spec 7 and 11)

- **Untouched files.** No change to `train.py`, `gaussian_renderer/`, `scene/`,
  `utils/`, the GT evaluators (`tools/evaluate*`,
  `estimator/motion_coherence_eval.py`), the training rasterizer sources
  (`forward.cu`, `backward.cu`, `rasterizer_impl.cu`), or any data or result
  directory.
- **Off path guarded.** Every new code path sits under `if visibility is not None`
  and is gated by the default `--object-mask-constraint off`.
  `run_bidirectional_all.sh` forwards no mask flag when the mode is off; this was
  checked with `--dry-run`.
- **HEAD vs patch.** `sparse_iekf.iekf_frame_step` from `HEAD` and from this patch
  were run on the same frame with the constraint off. The largest state or
  covariance difference was:

  | runs compared | largest difference |
  |---|---|
  | HEAD vs patch | 4.9e-7 |
  | HEAD vs HEAD | 6.7e-7 |
  | patch vs patch | 9.4e-7 |

  The patch therefore sits inside the CUDA run-to-run floor.
- **Baseline suites,** unchanged and passing:
  - `test_iekf_process_covariance` 24/24;
  - `test_rigid_se3_phase2` 47;
  - `test_phase2_parallel_pr` 37;
  - `test_joint_phase3_runner` 12;
  - `test_v2_grouped_solve` 16;
  - `test_iekf_end_to_end` 26;
  - `test_v2_full_scale_trust_region` 35.

## Remaining limitation (what masks alone cannot fix)

A mask says where the object is **not**. It does not say where a wrongly placed
Gaussian **should** be. For three cases the observations hold no restoring
direction, and the patch does not invent one:

- **Offscreen.** The Gaussian projects outside every camera. There is no ray, so
  there is no constraint row, and nothing is reported or moved.
- **Fully occluded.** The Gaussian sits behind measured non-object depth on every
  ray: `g_i = 0`, a zero row, reported `INDISTINGUISHABLE`. Only with
  `known-nonobject-depth`.
- **Stranded far outside the mask.** The rays it covers are already violated at
  the frame start. Its own Jacobian on them points nowhere in particular: any
  direction that leaves the ray satisfies them, and the RGB-D on object-valid rays
  does not see it.

In each case the frame is reported as not `FEASIBLE` (`UNRESOLVED_INFEASIBLE`,
`INDISTINGUISHABLE` or `MODEL_OWNERSHIP_INFEASIBLE`). The Gaussian keeps its
anchored predicted trajectory: in spec test 3 it moved 0.2 mm. The aggregate row
and the audit forbid any step that grows the existing violation, so a false
"correction" cannot be accepted silently: the step is rejected (`ρ = NaN`) and the
state rolls back to the last accepted state. These Gaussians must be dealt with
by the ownership and support logic of the later stages, or by the static model.
This patch deliberately does not change that logic.
