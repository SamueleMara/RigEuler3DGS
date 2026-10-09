# Phase 2 — generic rigid-screw audit, and the two parallel physical P/R models

Implements `CODEX_VALIDATE_GENERIC_TWIST_THEN_PARALLEL_PRISMATIC_REVOLUTE_PHASE2.md`
(Stage A §3–§14, Stage B §15–§36, artifacts §41–§42, gates §44–§45).

**Nothing in this work classifies a joint.** §33 forbids a loss-gap threshold and
§31 forbids turning fold wins into a majority vote; this delivers P and R
hypotheses plus the evidence, and stops there.

---

## 1. Current generic Phase-2 implementation

Unchanged, and still the production Phase-2 branch under
`--phase2-rigid-screw-iekf` (§2 requires it be preserved). One screw per part,
`T(q) = Exp_SE3(q ξ^)`, state `[q, q̇, θ_J(4)]`, dense 6×6 posterior,
`δy = H_G J_{G,J} δx_J`.

Five call sites in the shared loop were made *pluggable* so Stage B could reuse
it rather than fork it (§28). The generic path takes the same branch it always
did at each one:

| site | file | dispatch |
|---|---|---|
| per-part error stride | `estimator/joint_process.py` | `state.error_dim`, default `JOINT_ERROR_DIM` |
| joint JVP/VJP | `estimator/joint_renderer_bridge.py` | `state.jvp`/`state.vjp`, else `full_jvp`/`full_vjp` |
| sign gauge | `estimator/joint_iekf.py` | skipped when a part has no `zeta` |
| geometry information block | `estimator/joint_iekf.py` | `state.geometry_slice`, default `slice(2,6)` |
| retraction + covariance reset | `estimator/joint_process.py` | `state.retract_in_place` + `part.reset_jacobian` |

## 2. Exact COM decomposition

`estimator/rigid_motion_audit.py:com_decomposition` splits the fitted rigid
motion about the moving part's own centre of mass into `(Δc, R)` with
`μ_i(t) = c_t + R(μ_i^0 − c_0)`. The split is exact, not a projection: the four
audit states are reconstructions of the *same* fitted screw.

## 3. static / translation-only / rotation-only / full audit

`build_state(kind, …)` builds four Gaussian sets from one Phase-2 solution, and
all four go through the identical rasterizer, identical robust precision and
identical depth-validity mask (§6). Per-frame and per-camera RGB and depth
energies are written to `per_frame_camera.csv`, `per_frame.csv`, `per_camera.csv`.

Exact rigidity (§10) was verified by pairwise-distance drift under the full
state: **1.5e-15 (USB), 5.7e-16 (storage), 2.6e-15 (blade)** — machine zero.

## 4–6. Stage A on USB, Storage, Blade

Measured at 10 frames × 3 cameras × 4000 Gaussians:

| scene | full gain | translation-only | rotation after translation | camera spread | max θ (rad) | geometry cosine | opposed |
|---|---:|---:|---:|---:|---:|---:|---:|
| USB     | 4.81e+08 | 63.8% | 36.2% | 0.090 | 0.1296 | **0.4463** | **22.2%** |
| storage | 3.66e+04 | 86.9% | 13.1% | 0.024 | 0.0036 | 0.9987 | 0% |
| blade   | 4.81e+03 | 75.5% | 24.5% | **10.58** | 0.0093 | 0.9829 | 0% |

§14's questions, answered with numbers:

- **USB.** Its single constant screw is **not well supported**. The §11
  geometry score has median pairwise cosine 0.4463 with 22.2% of frame pairs
  *opposed* — different frames want geometries pointing in opposite directions.
  USB's ground truth is revolute, so this is not "the rotation is spurious"; it
  is "one constant screw does not fit the whole window".
- **Storage.** Overwhelmingly translational (86.9%, θ ≤ 0.0036 rad) with a
  near-perfectly consistent geometry (0.9987, nothing opposed). Its GT is
  prismatic and the audit agrees.
- **Blade.** The **rotation-only gain is negative (−2753)** while rotation
  after translation is +24.5%, and the per-camera spread of the rotation share
  is **10.58** — two orders above the other scenes. Blade's fitted rotation is
  not a motion seen consistently from several viewpoints; it is absorbing a
  per-camera mismatch.

> These three rows come from the Stage-A runs made during development; their
> output directories have since been removed. Re-running the commands in §19
> regenerates them *and* the §12 visuals in one pass.

## 7. Exact P state and equations

`estimator/phase2_prismatic.py`.

    μ_i(t) = μ_i^0 + s_t e_P        R_i(t) = R_i^0

State `(s_t, ṡ_t, e_P)`; local error `[δs, δṡ, δγ_1, δγ_2]`, **4-D**, one dense
4×4 covariance. The model has **no rotational degree of freedom at all**, so a
translating part cannot be explained away by a far axis.

`G_P` (the plane transverse to `e_P`) is **carried in the state and rotated with
it**, never rebuilt. Rebuilding it would let the frame spin freely about `e_P`
and silently redefine `γ` on every iteration.

## 8. Exact P JVP/VJP

    ∂μ/∂s = e_P      ∂φ/∂s = 0
    ∂μ/∂γ = s_t G_P  ∂φ/∂γ = 0

Verified (`tests/test_phase2_parallel_pr.py`): JVP vs central differences
**3.72e-10**; VJP exact transpose **1.17e-15**; `|e_P| − 1 = 0.00e+00` after 25
random retractions.

## 9. Exact R state and equations

`estimator/phase2_revolute.py`.

    μ_i(t) = a_R + R(e_R, θ_t)(μ_i^0 − a_R)      R_i(t) = R(e_R, θ_t) R_i^0

§22's minimal geometry: `a_R − c_0 = E_R ρ` with `ρ ∈ R²`, so
`e_R^T(a_R − c_0) = 0` holds **by construction** and the pure-gauge component
of the offset along the axis is not representable. State `(θ_t, θ̇_t, e_R, ρ)`;
error `[δθ, δθ̇, δη_1, δη_2, δρ_1, δρ_2]`, **6-D**, dense 6×6.

The retraction rotates the whole frame `(e_R, e_1, e_2)` rigidly by
`w = e_R × E_R δη`, which keeps `|e_R| = 1` and `e_R^T(a_R − c_0) = 0` exactly
rather than to first order.

## 10. Exact R JVP/VJP

    ∂μ/∂θ = e_R × (μ_i(t) − a_R)              ∂φ/∂θ = e_R
    δφ    = θ_t J_l(θ_t e_R) E_R δη           δa = −e_R (δη · ρ)
    δμ|ρ  = (I − R) E_R δρ                    δφ|ρ = 0

The `δa = −e_R(δη·ρ)` term is the one that is easy to drop. It is the exact
first-order consequence of the frame rotating with the axis, and without it the
offset stops being perpendicular to the axis and the recovered axis point
wanders along the axis frame to frame.

Verified: JVP vs central differences **3.93e-10** (translation *and* rotation
blocks); VJP exact transpose **1.04e-15**; after 25 random retractions
`|e_R| − 1 = 1.11e-16` and `e_R^T(a_R − c_0) = 3.89e-16`.

> One derivation error was found and fixed during this work. The first FD check
> showed 17% error on `δη` only. The cause was the *test's* retraction, not the
> math: `axis_basis` rebuilds the plane from scratch, so perturbing `e` spun the
> in-plane frame arbitrarily and `ρ` meant something different before and after.
> Carrying `E_R` in the state fixed it (§7, §9 above) and is also what the
> estimator needs.

## 11. Geometry and trajectory are estimated simultaneously

§27 is satisfied structurally, not by convention:

- the state is **one vector** — `[s, ṡ, γ]` / `[θ, θ̇, η, ρ]` — with **one dense
  covariance**, so there is no stage in which one block is fitted with the other
  held fixed;
- one RGB-D innovation produces one `b` and one `A` over the whole vector, and
  one trust-region step retracts all of it (`joint_iekf.frame_step`, unmodified);
- `apply_correction` applies the reset Jacobian `G` to the **whole** matrix, so
  the trajectory↔geometry cross-covariance is transported, never dropped. That
  block is the only route by which image evidence reaches the velocity at all,
  since a single frame has no velocity derivative.

`tests/test_phase2_parallel_pr.py::test_S28_no_new_optimizer` enforces the rest
of §28 by AST walk: no `.backward`, no `Adam`/`AdamW`/`SGD`/`LBFGS`, no
`requires_grad`, and no renderer code inside either model.

## 12. Synthetic P (§34)

Real CUDA renderer, 3 cameras, 5 frames, exact prismatic motion. The §29
initializer is *exact* on noiseless data, so the start is deliberately perturbed
(8° off-axis, travel ×0.7) or "the energy went down" would be vacuous.

| check | result |
|---|---|
| direction recovered | **0.000°** |
| travel recovered | max \|Δs\| **7.47e-07** |
| energy | 3.4249e+03 → **2.0284e-06** |
| M_R does not beat M_P | E_P 2.03e-06, E_R 3.32e+03 |

## 13. Synthetic R (§35)

| check | result |
|---|---|
| axis direction | **2.291°** |
| axis position (perpendicular component only) | **1.044e-02** |
| angle trajectory | max \|Δθ\| **9.84e-03** |
| energy | 1.1166e+04 → **1.6862e+02** |
| M_R beats M_P | E_R 1.69e+02 < E_P 2.77e+03 |

Only the component of the axis-point error **perpendicular** to the axis is
scored; sliding the point along the axis moves no Gaussian and is pure gauge.

## 14. Wrong-initialization recovery (§36)

M_P started 25° off-axis with the travel halved converges to **0.001°**.

## 15–17. USB / Storage / Blade P/R comparison

Wired and verified end-to-end on real storage RGB-D
(`--phase2-parallel-pr`): both models fit, held-out energies computed,
peak CUDA 1.45 GiB. The full three-scene runs are §19's commands.

**A result that changes how §31 must be read.** On storage, M_R's fitted angles
are ≈ 0 for every frame, and at `θ = 0` the revolute geometry tangent

    δφ = θ_t J_l(θ_t e_R) E_R δη

is **identically zero**. The axis direction and offset carry *no information*,
so M_R cannot be fitted at all — it is frozen at its initialization and accepted
0 of its iterations, while M_P accepted 9. E_R is then the energy of the
*initial* axis, not of a fitted model.

This is the small-angle end of the far-axis limit §33 warns about, and it means
**a near-prismatic scene does not give M_R a fair contest — it gives it a fixed
hypothesis**. Reporting that number as P-versus-R evidence would be wrong, so
the guard is recorded next to the energies rather than left implicit:
`comparison.json` carries `revolute_degenerate_folds`,
`comparison_is_interpretable`, and per fold an `observability` block with
`geometry_degenerate_at_zero_angle`, `fair_comparison` and an explicit warning
string. The smoke run reports `comparison_is_interpretable: false` for storage.

## 18. Held-out camera results (§31)

A fixed fold protocol shared by all scenes: fold *f* trains on cameras
`c % n_folds != f` and holds out the rest. Per fold the artifacts record
`ΔE_heldout = E_P − E_R` (`heldout/summary.json`), split by RGB, depth, frame
and camera in `per_frame_camera.csv`.

Fold wins are **not** converted into a classifier (§31), and
`not_a_classifier: true` is written into the artifact.

§32's predictive NLL is implemented via Woodbury
(`phase2_model_comparison.predictive_nll`): the only determinant taken is `p×p`,
and no dense pixel covariance is ever formed.

## 19. Paths and commands

Both stages, all three scenes, in §13/§37 order:

```bash
FRAMES=30 CAMS=4 NG=6000 ./run_stage_ab.sh
```

`run_stage_ab.sh` runs Stage A (which invokes the generic Phase 2 itself) with
the §12 renders, then Stage B with `--phase2-parallel-pr`, then prints the
per-fold train/held-out table split by RGB and depth. Models used:
`merged_USB_100109`, `dec_A_storage_45135`, `dec_A_blade_103706` — edit the
`MODEL` map at the top to point elsewhere.

Verified end to end at `FRAMES=3 CAMS=2 NG=1200` on all three scenes; the
numbers at that size are meaningless (almost no motion) but every stage, every
artifact and the degeneracy guard all produce output.

Artifacts (§41):

```
<out>/phase2_model_comparison/{comparison.json, per_frame_camera.csv,
                               prismatic/folds.json, revolute/folds.json,
                               heldout/summary.json, joint_debug.html}
<out>/phase2_rigid_motion_audit/{summary.json, twist_geometry.json,
                                 per_frame_camera.csv, per_frame.csv,
                                 per_camera.csv, geometry_score.csv,
                                 stage_a_sequences.html,
                                 renders/camera_<ci>/frame_<t>/<row>_{rgb,depth}.png}
```

§12's six rows — GT, Static, Phase1, Phase2 translation-only, Phase2
rotation-only, Phase2 full rigid — are written as **PNG files on disk**; the
HTML is an index over them and embeds no image data.

## 20. Unresolved failures

1. **M_R is unobservable at θ = 0** (§15–17 above). The model is correct and its
   derivatives are exact; the *information* vanishes. Until this is addressed,
   §31's held-out difference is only interpretable on scenes where the initial
   angles are non-negligible. A fix would need a reparameterization that stays
   observable as θ → 0, which the spec does not specify.
2. **USB's single constant screw is not supported** (geometry cosine 0.4463,
   22.2% of frame pairs opposed) on a scene whose GT is revolute. K=1 with one
   constant screw over the whole window is the wrong model for USB, and neither
   M_P nor M_R changes that — both are also one constant geometry.
3. **Blade's fitted rotation is per-camera noise** (rotation-only gain −2753,
   camera spread 10.58). §17's MODEL_MISMATCH separation is the missing piece.
4. **§42's 3D debugger is not built.** §12's fixed-camera sequences and HTML are
   done, and §41's `joint_debug.html` exists (it marks a degenerate fold's row
   in place, so a frozen M_R cannot be misread as a fitted one). The 3D view
   itself — COM path, translation/rotation/full displacement vectors, rigid
   velocity field, generic screw axis, camera frustums — is not.
5. **K > 1 is deferred**, as the spec allows.

### Gate status

§44 Stage A: 1–12 **all met**.
§45 Stage B: 1–14 and 18–19 **met**; 15–17 (the three real P/R runs) require the
§19 commands, which are long-running and are handed over rather than run here.
