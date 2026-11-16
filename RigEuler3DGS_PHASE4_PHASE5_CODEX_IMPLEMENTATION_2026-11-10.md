# RigEuler3DGS — Codex implementation specification: Phase 4 geometry recovery and Phase 5 photorealistic RGB-D-mask optimization

**Specification date:** 2026-11-10  
**Audited target:** [`SamueleMara/RigEuler3DGS`](https://github.com/SamueleMara/RigEuler3DGS), `main` tree SHA `6807d834c0e10ca6eff7cbb2d6255307ff3b1dca`.  
**Conceptual baseline:** [`SamueleMara/FreeArtGS`](https://github.com/SamueleMara/FreeArtGS), particularly `icp/generate_pcd.py`, `icp/color_icp_gpu.py`, and `nerfstudio/models/splatfacto.py` on branch `nerfstudio-local`.  
**Status:** Engineering specification, **not an implemented or tested change**. Codex must audit actual checkout/branch before changing code; fail if upstream formats differ.  
**Scope:** Implement Phase 4 and Phase 5 **in RigEuler3DGS** using its original CUDA rasterizer and Python bindings. Do not switch to gsplat, Nerfstudio, FreeArtGS's runtime, or a secondary renderer. Preserve Phases 0–3.

## 0. Hard decisions — do not reinterpret

1. **One selected direction.** The existing data prior chooses exactly `forward` **or** `backward`. Phase 0–3 have been run in that direction, with exactly one trained canonical endpoint: `initial` for forward or `final` for backward. Do **not** run the opposite direction, train an opposite endpoint, fuse two endpoint models, or add a bidirectional dependency. Later physical observations can still be used for recovery in the single chosen canonical coordinate system.
2. **Phase 3 is immutable upstream input.** It provides the qualified joint type (`prismatic` or `revolute`), physical joint geometry, per-frame coordinate `q`, full original-frame mapping, selected core membership, and provenance. Neither Phase 4 nor the default Phase 5 modifies `J_3` or `q_3`.
3. **Phase 4 is a geometry-completion stage.** First recover missed **existing** moving Gaussians by static-vs-joint-motion full-scene counterfactuals; second propose truly missing *observed* surfaces from multiview RGB-D and back-transport them with the Phase-3 joint; finally accept/reject candidates by nonlinear whole-sequence rendering. A surface appearing late is NOT a physical birth — it exists at every frame, including while occluded.
4. **No per-Gaussian motion** and no per-frame unconstrained 6-DoF deformation. A Gaussian is physically static or belongs to the one moving rigid body. It retains one canonical state. Its motion is exactly the Phase-3 joint transform. A third `UNRESOLVED` label is an inference status, **not** a third physical motion.
5. **Phase 5 trains the Gaussian scene, never the joint.** Implement complete FreeArtGS-style RGB, metric-depth and object-mask training in the **existing RigEuler CUDA rasterizer**. The normal full-photorealistic profile optimizes Gaussian parameters only (means, anisotropic scale/rotation, opacity, SH/DC) while holding `J_3,q_3` bitwise fixed. Call this `gaussian_rgbd` (the user’s “appearance” as opposed to *joint kinematics*). Also offer a **strict** `sh_only` profile that trains ONLY DC/SH appearance; with fixed geometry/opacity its depth and mask losses are necessarily monitoring terms with zero parameter gradient. Do not claim otherwise or silently enable geometry for `sh_only`.
6. **Joint-geometry refinement is a separate, optional final command** `tools/refine_joint_geometry_final.py`, **disabled by default** and not invoked by Phase 4, Phase 5, or the ordinary runner. When explicitly enabled in a future experiment, it freezes **all** Gaussian tensors, material labels, camera calibration and topology, optimizes only minimal joint geometry (plus optionally very small, strongly regularized trajectory corrections if configured) for a few whole-sequence sweeps, and outputs a separate artifact. Its existence must not cause any joint update in the current default workflow.
7. **Do not train camera poses** in the new stages by default: calibrated fixed extrinsics/intrinsics are shared across the sequence. Preserve source images, depth, masks and original frame indices. Do not use dataset GT for inference.
8. **Native CUDA rendering only.** Keep the existing `submodules/diff-gaussian-rasterization` extension, CUDA front-to-back sorting/alpha compositing and `GaussianModel`. Add differentiable alpha and *metric axial depth* channels to the training API, retaining legacy inverse-depth semantics and legacy callers.
9. **Keep masks as real observations.** RGB is foreground-masked as FreeArtGS does; silhouette uses **the entire binary mask**, including negative/background pixels; sensor depth contributes only when mask=1, finite and positive. Never use background depth as a foreground constraint. Treat uncertain/mixed depth boundaries separately from crisp silhouette evidence.
10. Do not invent measured surfaces in unseen regions; complete the **union of observed surfaces** and record never-observed regions as unmeasured.

### 0.1 Fundamental distinction: two Gaussian training profiles

| Mode | Joint J, q | Trainable Gaussian parameters | RGB gradients | Depth/alpha gradients | Intended use |
|---|---|---|---|---|---|
| `gaussian_rgbd` (**default complete FreeArt-style training**) | Frozen | canonical XYZ, orientation, scale, opacity, DC and SH | Yes | Yes | Full photorealistic reconstruction while **never changing the joint** |
| `sh_only` (**strict literal appearance-only**) | Frozen | DC and SH only | Yes | **No useful parameter gradient** if opacity and geometry frozen | Color/view-dependent appearance polish after geometry is accepted |

The user requested both “complete RGB-D-mask training” and “only optimize appearance, not the joint”. These requirements are reconciled by **`gaussian_rgbd` = optimize the entire 3DGS visual representation, but not the joint**, with a separate `sh_only` option if “appearance” means only radiance. Do **not** fake RGB-D/mask training by including constant depth/mask losses in `sh_only` and claiming that those losses optimize the scene.

## 1. Exact existing-code anchors (audited, not fabricated)

- `run_bidirectional_all.sh`: despite its name, `--direction auto` already uses the scene's `metadata/direction/summary.json:recommended_direction` and chooses **one** pass. It writes `static_bidi/initial` or `static_bidi/final`, and Phase 2/3 under `bidirectional/<direction>_rigid` and `bidirectional/<direction>_joint`. Extend through **opt-in post-stage hooks only**; preserve existing 0–3 CLI and cache logic.
- `tools/run_joint_phase3.py`: takes `--source`, `--model`, `--phase2-dir`, `--direction`, `--cameras`, etc. Outputs `<p3>/summary.json`, `<p3>/<type>/geometry.json`, `<p3>/<type>/trajectory.csv`, with CSV headers `processing_index,original_frame,q,qdot,q_phase2_init,posterior_sd_q,...`. `summary.json` includes `original_frames_processing_order`, `canonical_original_frame`, `hypotheses`, `joint_types_run`. **Important:** Phase 3 can fit both P/R; it does **not** by itself universally assert which fitted hypothesis is selected. New stages need an explicit `--joint-type` or a verified machine-readable upstream qualified-selection file. Never choose by GT or an undocumented threshold.
- `<p2>/support_coherence/support.json`: `members` are the Phase-2/3 confident core's **original Gaussian row IDs**; validate 1-to-1 alignment with the canonical PLY before any rewrite.
- `utils/sequence_direction.py`: processing order versus original chronological frame indexing. Use the Phase-3 CSV's `original_frame`, do not infer `t = processing_index` for backward scenes. Canonical anchor is CSV row zero in processing order.
- `scene/dynamic_sequence_dataset.py` and `scene/cameras.py`: calibrated cameras, loaded original RGB/D, frame IDs and timestamps. Review actual `invdepthmap` representation before comparing to metric depth. For Phase 4/5 read metric depth from the validated source (`depth_npy/<camera>/frame_NNNNNN.npy` when applicable); never treat `Camera.invdepthmap` as metric without conversion.
- `tools/run_iekf_scene.py:load_scene(...)`: existing inference-only loading path for endpoint `GaussianModel` and camera set; `tools/run_joint_phase3.py` demonstrates usage. `utils/camera_selection.py` already enforces camera selection.
- `estimator/masked_observation.py:prepare`, `load_object_masks` and `utils/noise/sensor_model.py`: strict binary masks, foreground-valid depth and calibrated per-camera variances. Reuse validation/policy and sensor provenance, but **do not** blindly reuse an IEKF quadratic likelihood as the FreeArt-style photorealistic training loss.
- `gaussian_renderer/__init__.py:render(...)`: already accepts `means3D_override`, `rotations_override`, and delegates to `GaussianRasterizer`; `rotations_override` only applies with `compute_cov3D_python=False`. Reuse this path or make an articulated wrapper. It currently returns `render`, `radii`, `viewspace_points`, `visibility_filter`, and an **inverse-depth-related** `depth` channel.
- `submodules/diff-gaussian-rasterization/diff_gaussian_rasterization/__init__.py`: `_RasterizeGaussians` PyTorch autograd currently returns `(color, radii, invdepth)` from `_C.rasterize_gaussians`. Extend **additively/backward-compatibly**; preserve all legacy entry points and return conventions.
- `submodules/diff-gaussian-rasterization/cuda_rasterizer/forward.cu`: ordinary rendering computes `expected_invdepth += (1 / depths[gid]) * alpha * T` and stores `final_T`. `rasterizer_impl.h` image state already includes `accum_alpha` (despite the variable name, inspect its actual semantics before exposing). `backward.cu` already reverses compositing for RGB/inverse depth.
- `submodules/diff-gaussian-rasterization/{ext.cpp,rasterize_points.cu,rasterize_points.h}`, `cuda_rasterizer/{rasterizer_impl.cu,rasterizer_impl.h,forward.cu,forward.h,backward.cu,backward.h}`: the extension interface changes must be propagated consistently through these files and `setup.py` if necessary.
- `scene/gaussian_model.py:training_setup`, `densification_postfix`, `densify_and_clone`, `densify_and_split`, `prune_points`, `capture_aux_state`, `load_aux_state`, `save_ply`, `load_ply`: reuse original Adam/SH/densification but extend topology-changing methods with stable *material IDs and provenance*.
- `estimator/surface_bootstrap.py`: useful `camera_geometry`, `backproject`, `voxel_seeds`, `gaussian_parameters`, `bootstrap_loss_terms` and measurement uncertainty. **Do not** directly use its unmodified all-neighbor `fit_surface` on mixed or opposite faces of a thin surface; Phase 4 needs a layer-aware association.
- `utils/gaussian_sequence_assignment.py`, `utils/kpart_counterfactual_rendering.py`, `rigid_parts/outer_render.py`: reusable ranking, counterfactual composition and affected-pixel code, after auditing their assumptions. Their optional DINO/flow/soft-label objectives are *not* evidence required by the new phases, and their existing `depth` outputs may be in inverse-depth convention.
- `tools/train_tracking_bootstrap.py`: an existing source for no-densification Gaussian refinement and separate held-out pixel partitions; currently derives alpha via an additional `render(... override_color=ones)` pass. Replace this **for the new path** with a native differentiable alpha output, without breaking the old trainer.
- FreeArtGS reference: `nerfstudio/models/splatfacto.py:get_loss_dict` uses masked image L1+SSIM, `L1(mask, accumulation)` and masked sensor depth L1. `icp/generate_pcd.py` produces per-frame masked RGB-D clouds; `icp/color_icp_gpu.py` registers and merges them; **our Phase 3 joint supplies motion, so DO NOT rerun ICP to estimate unconstrained motions**.

## 2. Shared input/output contracts and invariants

### 2.1 CLI inputs to both stages

```text
--source <scene-directory>                    # rgb/, depth_npy/, mask/, calibration
--results-root <existing-results-root>
--direction auto|forward|backward             # auto validates recommended_direction
--model <static_bidi/initial-or-final>         # SAME model as Phase 3
--phase2-dir <bidirectional/direction_rigid>
--phase3-dir <bidirectional/direction_joint>
--joint-type prismatic|revolute               # explicit unless verified selected_joint.json exists
--cameras all|N|cam1,cam2,...                 # exact upstream camera subset
--frame-window <original_start>:<original_end> # exactly Phase 3's window
--sensor-model <calibrated-model.json>|...     # upstream calibrated/ideal/assumed policy
--frames-on-cpu                               # avoid T*C images resident on GPU
--output <results/phase4_or_phase5>
--seed <integer>
--dry-run
```

**Preflight:** verify all input hashes and manifests, depth units, one-to-one camera/ROI alignment, nonempty masks, positive finite depth support, PLY count, original Gaussian index identities, joint type, normalized unit axis, Phase-3 original-frame order, q anchor zero, timestamp sign/time spacing, camera extrinsics, and exact canonical endpoint. Fail on mismatch: no silent conversions or fallback to another model.

### 2.2 Representation and material indexing

Canonical Gaussian `i` stores

\[
G_i^0=(\mu_i^0,q_i^0,\ell_i,\eta_i,f_{i,\mathrm{DC}},f_{i,\mathrm{SH}},z_i,\mathrm{id}_i,\mathrm{parent}_i).
\]

Here `q_i^0` is its quaternion (not the joint coordinate), `ell_i` are log-scales, `eta_i` is opacity logit, and `z_i in {STATIC,MOVING}` is the **physical** assignment. `UNRESOLVED` is an inference status stored separately; in candidate scoring the explicitly recorded provisional baseline material is used, never a third physical transform. The core is frozen as moving unless an audit explicitly invalidates upstream support.

For selected joint `J_3` and a given original frame `f`, define `Q_f,p_f` **relative to the canonical anchor**:

- P: `Q_f=I`, `p_f=(s_f-s_a)*e_P` (metres).
- R: `Q_f=Exp((theta_f-theta_a)[e_R]_x)`, `p_f=(I-Q_f)*a_R` (radians).

\[
\mu_i(f)=\begin{cases}\mu_i^0 & z_i=S\\Q_f\mu_i^0+p_f&z_i=M\end{cases},
\qquad
R_i(f)=\begin{cases}R_i^0 &S\\Q_fR_i^0&M.\end{cases}
\]

`Sigma_i(f)=Q_f Sigma_i^0 Q_f^T` for moving Gaussians. **Transport orientations, not only centers.** SH/material appearance should use canonical local-frame viewing direction `v_i^0 = Q_f^T (C_c - mu_i(f))/||...||` for moving and ordinary world direction for static; never rotate SH coefficient rows as if they were 3D vectors. Prefer differentiable `eval_sh` -> `colors_precomp` in `gaussian_renderer`, reusing the CUDA compositing backend.

**Gauge:** canonical Gaussian means are anchored at the selected Phase-0 endpoint. `q_a=0` exactly; do not learn a global rigid transform. `q` is indexed by *original frame* after explicit conversion from Phase-3 processing order. For backward direction, the array may be stored in decreasing original-frame order; do not silently reverse twice.

### 2.3 Views, measurements and masks

A view `(c,f)` is `(RGB[3,H,W], D_metric[H,W], M_bool[H,W], V_depth_bool[H,W], K_c, T_WC, time_seconds)`. Input masks are binary at source resolution; when downsampling use area averaging for coverage and pre-multiplied RGB `resize(M*I)` rather than `resize(M)*resize(I)` at boundaries. Keep full-resolution source for final acceptance.

`V_depth = M_bool & isfinite(D_metric) & (D_metric>0)`. Foreground RGB applies to `M` only. Mask alpha loss uses *all* pixels where mask is trusted, including negative ones. Depth does not use values on background even if nonzero. Mixed-edge pixels may be excluded/downweighted from depth via a separately declared edge-validity policy, never silently erode the silhouette mask for all losses.

Camera poses/intrinsics remain fixed. Declare the treatment of lens distortion exactly as the upstream loader does; no resampling camera geometry without changing calibration consistently.

### 2.4 Rendering objective shared by Phase 4 and full-RGBD Phase 5

For front-to-back contributions `a_ip=clamp(alpha_i*exp(-0.5*d_ip^T Sigma_2D^{-1} d_ip),...)`, `T_ip=product_{j<i}(1-a_jp)`, `w_ip=T_ip*a_ip`, `A_p=sum_i w_ip = 1-T_final,p`, and camera-space **axial** depth `z_i`:

\[
\widehat I_p=\sum_i w_{ip} c_{ip}+T_{\rm final,p}\,b,
\quad
Z_p=\sum_i w_{ip}\,z_i,
\quad
\widehat D_p=Z_p/(A_p+\epsilon_A).
\]

Keep legacy inverse depth as a separate named output `weighted_invdepth` with unchanged semantics. `D_hat` is a normalized **expected camera-Z depth**, not physical first-return depth. This FreeArt-style definition is a baseline; edge/mixed layer caveats must be measured, not ignored.

The **full FreeArt-style** loss for each view is

\[
L_{ct}=\lambda_{rgb}\big[(1-\lambda_s)L_1(M\odot \widehat I,M\odot I)
+\lambda_s\{1-\mathrm{SSIM}(M\odot\widehat I,M\odot I)\}\big]
+\lambda_D L_D+\lambda_M L_M,
\]

\[
L_D=\frac{\sum_{p\in V_{depth}} \rho(\widehat D_p-D_p)}{\max(1,|V_{depth}|)},
\qquad
L_M=\frac{\sum_{p\in V_{mask}}|A_p-M_p|}{\max(1,|V_{mask}|)}.
\]

- For FreeArt parity, start with `rho(x)=abs(x)`, masked-image global-mean L1/SSIM, and FreeArt-style alpha L1; separately report coverage-normalized RGB L1 to make changing foreground areas comparable. `lambda_*` are documented config variables, never guessed “same as FreeArt” without reading the actual scene config.
- `L_D` is computed only at valid observed **object** depth; if no valid depths, set 0 and report support=0 (not NaN). Do not compute D loss when `A_hat` is near zero without a declared policy. Recommended: use `A_hat` clamped denominator and cap or mask unstable predicted depths with a separately reported coverage penalty; missing occupancy must still receive mask gradient.
- Mask loss and masked RGB must not be double-counted as statistically independent likelihoods when using the **IEKF expected-background quadratic** model. The Phase-4/5 photoreal objective here is an explicitly weighted engineering loss, **not** the IEKF probabilistic likelihood.
- For real depth, additionally log depth performance by sensor-noise bin and proximity to mask/depth discontinuities. Calibrated variance may inform candidate ranking/robustness, not silently modify the stated FreeArt baseline.
- In candidate model selection normalize/aggregate per camera/time with declared weights, use a held-out/data-role manifest, and record candidate complexity. Never identify training loss decrease with posterior probability.

## 3. CUDA extension first: differentiable RGB + metric depth + alpha

This is a **prerequisite** of Phase 4C and Phase 5. Implement it before changing reconstruction code.

### 3.1 Forward changes

Add a new opt-in CUDA pathway (e.g. `rasterize_gaussians_rgbd`) that returns exactly `(RGB[3,H,W], radii[N], weighted_invdepth[1,H,W] [optional for compatibility], alpha[1,H,W], weighted_metric_depth[1,H,W])` while preserving `_C.rasterize_gaussians` and its existing tuple for old code. The Python wrapper forms `metric_depth = weighted_metric_depth / clamp_min(alpha,eps_A)` using ordinary torch autograd. This avoids embedding the quotient derivative inside CUDA while preserving exact gradients to `Z` and `A`.

In the current `forward.cu` splat loop, accumulate `Z += w * depths[gid]`, and compute `alpha = 1 - T_final`; retain the **identical** contributing splat order, antialiasing, alpha clipping, early termination, background and calibration. Pixels without contributors produce `alpha=0,Z=0` (depth validity is handled in loss). Do not replace front-to-back composition with additive opacity.

Do not confuse `rasterizer_impl.h:ImageState::accum_alpha` (whose content/layout must be verified) with a PyTorch-returnable alpha tensor; expose via an actual output buffer and test its identity `A=1-T`.

### 3.2 Backward changes

Pass `dL/dZ` and `dL/dA` into a new CUDA backward entry point (or equivalent backward-compatible extension). In the reverse compositing kernel, add direct z-gradient `w_i * dL/dZ` and alpha/weighted-depth contributions to `dL/da_i` **including their effect on later splats**. Then use the existing chain from splat opacity/covariance/mean to raw Gaussian parameters; camera-space `z_i` additionally contributes to mean gradient through the camera rotation. Do NOT just differentiate `Z` with respect to z while treating weights as constants.

For the torch-side quotient `D_hat=Z/(A+eps)`:

\[
\partial L/\partial Z=(\partial L/\partial D)/(A+eps),
\quad
\partial L/\partial A=\partial L/\partial A_{mask}
- (\partial L/\partial D)Z/(A+eps)^2.
\]

CUDA receives these as independent output adjoints and performs exact compositing differentiation. Preserve legacy inverse-depth gradient behavior and callable interfaces; `None` grads -> zeros.

### 3.3 Required CUDA parity tests

1. **Forward:** 1 Gaussian, 2 overlapping layers, 2 cameras, partly transparent splats and background: compare RGB/A/Z against a scalar CPU front-to-back oracle using the same Gaussian weights, clipping and sorted order.
2. **Gradient:** central finite differences of total `L_RGB+L_depth+L_alpha` with respect to Gaussian `xyz`, `quaternion`, `log_scale`, `opacity_logit`, DC, and SH. Exercise true overlap/occlusion and nonzero background; compare relative errors, logging CUDA nondifferentiability near discrete culling/tile boundaries separately. No finite-difference test at contribution-order ties.
3. **Compatibility:** old `render(...)`, old `rasterizer(...)` and all existing renderer tests return exactly previous values/shapes; dynamic IEKF and phase0–3 commands unchanged.
4. **Depth semantics:** synthetic two-layer scene where weighted expected depth, inverse depth and first return differ significantly; prove new `metric_depth` matches weighted camera-Z oracle, not `1/invdepth`.
5. **Articulation gradients:** finite difference through a tensor-only P/R joint transform into Gaussian positions **and quaternions**, with all static Gaussian gradients wrt joint identically zero.
6. No second rasterization pass to estimate alpha in the final trainer. Keep feature splatting a separate optional diagnostic, not a required alpha computation.

### 3.4 RGB/SH consistency

For moving splats, SH directional dependence should rotate with material. Implement an **optional** `local_SH` mode: derive colors in Python from canonical-frame directions and pass as `colors_precomp`, preserving PyTorch differentiability through camera-to-point viewing directions. Maintain legacy `world_SH` mode for exact phase0 checkpoint rendering parity; select `local_SH` for the final articulated model after comparison, and report initial before-training color discrepancy. For Phase 4 candidate geometry compare using the same chosen SH policy across S/M counterfactuals.

## 4. Phase 4A — classify the original Gaussian population under joint motion

**Inputs:** canonical Phase-0 PLY, selected-direction Phase-2 core, selected joint `J_3,q_3`, all views. **Output:** exact same number of original Gaussians with durable material inference and identity.

### 4.1 Candidate construction

- `C_core = p2.support_coherence.members` is guaranteed moving seed; do not call its complement static truth.
- Build deterministic candidate components from endpoint spatial neighborhoods and overlap of projected splat footprints, optionally existing motion/assignment scores; split at measured depth-layer discontinuities, don't smooth labels across hinges/part edges.
- Preserve source original ID, component ID, evidence provenance and current provisional baseline label. Small components can be merged only when projected influence and physical adjacency justify it.
- If `J_3` is clearly degenerate/wrong type, abort Phase 4; do not soften wrong joint by multiplying `q` with a trust factor.

### 4.2 Full-scene hypothesis test

For component `B` and the currently accepted labeling `z_-B`, form **one physical scene** under each counterfactual `h in {S,M}`. All other Gaussians remain in place; use **full** RGB-D-alpha compositing with the candidate moved by the complete physical joint or left static. Let

\[
E_h(B)=\sum_{c,f}w_{cf}L_{cf}[\mathcal R(\mathcal G_{-B}\cup B_h)]
\quad\text{and}\quad \Delta_B=E_S(B)-E_M(B).
\]

Also compare a flagged `OMIT` diagnostic when a bad original Gaussian fits neither hypothesis (not automatic deletion). For large groups, rank with affected-pixel crops/informative frames first; the **acceptance step always scores all cameras and all original frames** at native validation resolution. Recompute the baseline after each accepted component; re-evaluate interacting components. For one group with no visible changing-state evidence, return `UNRESOLVED`.

### 4.3 Joint uncertainty sensitivity

Phase-3 `posterior_sd_q` is about q and **not a calibrated probability that the joint type/axis is correct**. Report sensitivity scores `Delta_B(J_3+delta_j)` under declared axis/axis-point/trajectory perturbations consistent with available Phase-2/3 diagnostics. If `p(J)` is calibrated, marginalize over **one shared** joint sample across all affected Gaussians per draw; never independent jitter per Gaussian. Without a calibrated distribution write `JOINT_CONFIDENCE_UNCALIBRATED`, numeric sensitivity ranges, no `prob_moving`.

### 4.4 Decisions and output

Statuses: `ACCEPTED_MOVING`, `ACCEPTED_STATIC`, `UNRESOLVED`, `SUSPECT_ORIGINAL`, `UPSTREAM_JOINT_INVALID`. The status is inferred from consistent comparative evidence (including negative visible occlusion), a predeclared complexity/uncertainty policy, and validation improvement; uncertain examples do not become static by default. For a physical render, `UNRESOLVED` retains a recorded **provisional** baseline assignment and must remain explicitly marked unresolved in exported metadata. No duplicated S/M ghost splats.

Artifacts: `phase4/support_completion/{original_ids.npz,material_status.npz,material_provisional.npz,components.json,full_scene_scores.json,joint_sensitivity.json,visibility.npz,full_scene_audit.json}`.

## 5. Phase 4B — discover missing *observed* surfaces

**FreeArt-inspired idea:** accumulate RGB-D measurements through time into one part frame. **RigEuler difference:** motion is supplied by the selected Phase-3 P/R joint, not estimated again with free-frame ICP.

### 5.1 Unexplained foreground map

Render Phase-4A scene at **every** `(c,f)` to get `I_hat,D_hat,A_hat`, with masks/validity. Write maps and statistics for:

- `EXPLAINED`: current model accounts for observed foreground.
- `COVERAGE_GAP`: `M=1`, valid D, insufficient alpha/coverage.
- `FRONT_SURFACE`: observed depth is nearer than predicted, conditional on depth uncertainty; possible new surface **or** existing pose/calibration error.
- `BEHIND_PREDICTION`: measured surface lies behind current visible prediction; no automatic birth because a wrong occluder is a competing explanation.
- `DEPTH_INVALID`: no depth-derived candidate (RGB/mask still constrain).
- `MIXED_EDGE`: likely silhouette/depth discontinuity; not eligible for naïve 3D fusion.
- `OUT_OF_VIEW/OCCLUDED`: missing observation is not negative evidence.

No universal `5 mm` or fixed alpha threshold is claimed to be calibrated; thresholds live in a declared config and are evaluated in synthetic/real sensitivity tests. Use calibrated depth variance (if available) plus separately declared rendering-model discrepancy. Maintain mask negatives for contradiction checking.

### 5.2 Pixel-to-world geometry

For eligible pixel center `p=(u+0.5,v+0.5)` with metric axial depth `d`:

\[
r=K_c^{-1}[u+0.5,v+0.5,1]^T,\qquad y_W=R_{WC}(d\,r)+t_{WC}.
\]

For each observation from original frame `f`, canonical under `h=S` or `M`:

\[
x_{S,f}=y_W,\qquad x_{M,f}=Q_f^T(y_W-p_f).
\]

Both hypotheses use the **same selected anchor** and same `J_3,q_3`; no opposite endpoint. Each proposal retains `(original_frame,camera,pixel,RGB,depth,depth_noise,material_hypothesis,source_hash)`.

### 5.3 Measurement covariance (not rendering covariance)

For input uncertainty `Sigma_{uvd}` in `(u,v,d)` and pixel backprojection `J_back`, propagate

\[
P_{y}=R_{WC}J_{back}\Sigma_{uvd}J_{back}^T R_{WC}^T,
\quad P_{x_M}=Q_f^T P_y Q_f.
\]

For independent pixel and depth uncertainty:

\[
J_{back}=\begin{bmatrix}d/f_x&0&(u+0.5-c_x)/f_x\\0&d/f_y&(v+0.5-c_y)/f_y\\0&0&1\end{bmatrix}.
\]

The sensor model may only estimate depth variance, in which case use its anisotropic ray-direction covariance `sigma_D^2 * d_ray d_ray^T` and declare any added pixel calibration uncertainty explicitly. Shared error in camera calibration and joint geometry must **not** be reduced by treating repeated observations as independent. The Gaussian render covariance is fitted later from surface shape, not set equal to `P_x`.

### 5.4 Association / surface fusion

1. Group eligible observations by candidate canonical material hypothesis and frame/view provenance.
2. Associate across frames/cameras using canonical 3D proximity relative to sensor+model uncertainty, projected pixel consistency and *front-to-back visibility*. Do not accept spatial nearest-neighbor matches alone when distinct front/back surface layers overlap in 3D.
3. Use connected surface patches and robust weighted tangent-plane fits, not one Gaussian per pixel. Adapt `surface_bootstrap.py` initialization but add layer-consistency; **test synthetic thin panels before using its neighbor fitter**.
4. Fit `mu_B`, tangent basis/normal, two tangential footprint scales from observed sampling density and normal thickness from geometric fit; initialize quaternion from orthonormal axes, opacity conservatively and DC from observed RGB. Record center-position uncertainty separately.
5. Compare static and moving temporal coherence only as *proposal ranking*. If evidence is ambiguous, retain both hypotheses for full-render testing without putting both into the same physical scene.

Mathematical seed fit for hypothesis `h`:

\[
\min_{\mu,n:\|n\|=1}\sum_{j\in O_B}\rho\left(
\frac{n^T(x_{h,j}-\mu)}{\sqrt{n^TP_{x_{h,j}}n+\sigma_{surface}^2}}
\right)+R_{surface}(B),
\]

with layer-aware associations `O_B`, robust weights and bounded excess surface thickness. This robust planar approximation **is not** a substitute for nonlinear render acceptance.

### 5.5 No physical birth time

For a Gaussian observed first at frame `b`, if moving, back-transport it into the canonical anchor at `a` and render at **all** f via `Q_f,p_f`:

\[
\mu_B^0=Q_b^T(y_W(b)-p_b),\quad
\mu_B(f)=Q_f\mu_B^0+p_f.
\]

Missing from early views is neutral if occluded. A candidate that would be visibly in front of measured early-frame surfaces can be rejected. Do not use “first visible” as a physical creation time; first visibility is only source provenance.

Artifacts: `phase4/discovery/{residual_maps/,pixel_evidence.npz,proposed_patches.npz,source_provenance.json,layer_association.json,proposal_scores.json}`.

## 6. Phase 4C — nonlinear whole-sequence acceptance and local refinement

For each proposed patch/group `B`, compare `h in {OMIT,STATIC,MOVING}`. `OMIT` leaves the current model unchanged. For `STATIC` and `MOVING`, initialize canonical parameters and perform short local refinement on `(mu,quat,log_scale,opacity_logit,DC)` **with J and q fixed**, rendering the **entire scene** at every step. Freeze original Gaussians for the primary test to isolate the candidate's effect; an optional bounded joint *Gaussian-only* cleanup pass after committing accepted patches can follow, but joint kinematics must remain frozen.

Training set for proposal refinement and validation set for acceptance must be established **before** inspecting per-candidate residual maps. If Phase 3 already used all frames, report the validation as *conditional on upstream J* rather than independent scientific evidence. Good choices: spatially disjoint patches in informative frames or camera splits with enough local coverage; never hold out all views containing a rare-visible patch and then reject it for zero evidence.

For each hypothesis:

\[
\Theta^*_{B,h}=\arg\min_{\Theta_{B,h}}
\sum_{(c,f)\in\mathcal D_{fit}}L_{cf}(G\cup B_h(\Theta_{B,h}))
+R_{shape}(B)+\lambda_N |B|.
\]

Evaluate `L_val` plus complexity under exact full-scene CUDA RGB-D-mask renders, including **every original frame and camera** in the final all-sequence audit. Acceptance must account for variance/camera imbalance and visible negative evidence, and detect duplicate geometry (new patch re-explaining a surface already represented). Only accepted candidates become real Gaussians. In provisional optimization, hold the candidate's static/moving label hard, not a trainable soft blend.

**Scoring caution:** compare data-fit losses in affected pixels **and** full views. A global view average can swamp evidence from a surface visible in 3 of 60 frames; use a documented per-candidate informative-view score plus an all-view contradiction audit. Both are necessary. Full scene must be rerendered after multiple accepted groups to catch interactions.

Acceptance statuses: `ACCEPTED_STATIC`, `ACCEPTED_MOVING`, `REJECTED_DUPLICATE`, `REJECTED_OCCLUSION_CONTRADICTION`, `AMBIGUOUS_MATERIAL`, `INSUFFICIENT_OBSERVATION`, `JOINT_UNRESOLVED`, `INVALID_DEPTH_SOURCE`.

**Final Phase-4 artifact**: one canonical Gaussian model `phase4/final_canonical/point_cloud.ply`, sidecar `material_ids.npz`, `gaussian_uid.npz`, `parent_uid.npz`, `source_provenance.json`, `joint_reference.json`, counts, full-sequence render audit, and a training-compatible checkpoint. No temporary physical static/moving copies.

## 7. Phase 5 — full photorealistic RGB-D-mask Gaussian training, joint frozen

### 7.1 Default behavior (important)

By default **NO joint parameter is registered with an optimizer**, not merely set to an extremely small LR. Treat Phase-3 `(type,axis,axis_point,q[])` as immutable, hash them at startup and assert exact equality at every checkpoint and end. All Gaussians are transported under that one fixed joint at each training view. Run over every original frame/camera with balanced sampling; do not restrict to frames in which the confident Phase-2 core was visible.

`--train-mode gaussian_rgbd` (full FreeArt-style objective, **no joint updates**): optimize `_xyz`, `_rotation`, `_scaling`, `_opacity`, `_features_dc`, `_features_rest` in **canonical** coordinates, with existing `GaussianModel` optimizers; no scene-specific per-frame Gaussian states. `--train-mode sh_only`: freeze `_xyz`, `_rotation`, `_scaling`, `_opacity`, and train ONLY `_features_dc/_features_rest` (and exposure only if explicitly requested; default fixed). In `sh_only`, depth and mask are validated every view but mathematically cannot improve unless some trainable parameter affects geometry/opacity; log their values as metrics, not pretend their constants alter SH gradients.

### 7.2 Time and camera sampler

Use deterministic seeded frame/camera shuffling that covers all original frames and cameras once before repeating, with optional balanced oversampling of articulation extremes and rare-visible surfaces. Per-iteration `L` uses sampled views; full-scene validation periodically sweeps all `(c,f)` and reports per-camera/per-frame metrics including RGB/PSNR/SSIM, depth L1 on valid foreground, foreground alpha gap, false off-mask opacity, total counts, visible moving Gaussian fraction, and rare-surface coverage. Never claim every iteration evaluates all views, and never restrict validation to a random subset.

### 7.3 Loss for `gaussian_rgbd`

\[
L_{5}=\sum_{c,f} w_{cf}[\lambda_{rgb}L_{RGB}+\lambda_D L_D+\lambda_M L_M]
+R_{geometry}(G)+R_{opacity}(G)+R_{SH}(G).
\]

Start with FreeArt parity (`RGB masked L1+SSIM`, valid-foreground metric depth L1, entire-mask alpha L1). Explicitly document and normalize `lambda_*`; produce independent ablations for loss changes. Optional physical sensor-noise robust depth loss is separate `--depth-loss student_t`, not the baseline. Scale and opacity regularizers need to prevent large alpha sheets/depth cheating; don't overregularize rare-visible valid surfaces. Freeze camera calibration and joint.

**Densification lifecycle:** use existing `scene/gaussian_model.py` clone/split/prune but enforce that every child inherits a **hard** static/moving material ID, source provenance and parent lineage, with canonical-space splitting. Keep `original_source_id` for old Gaussians and a unique persistent `gaussian_uid` for every descendant/new Gaussian. Topology changes must update all row-aligned arrays atomically and optimizer Adam moments consistently. Densification must never be performed on transported `get_xyz` and then saved as canonical `_xyz`. Prune by multi-view/multi-frame contribution over a sweep, NOT merely because a rare-visible Gaussian is not visible in the current view. Do not prune Phase-4-supported geometry solely on an opacity reset without a coverage audit.

Train in stages **within Gaussian-only optimization**: (A) short geometry + opacity + DC warm-up at low SH; (B) full RGB-D-mask Gaussian refinement with controlled densification/pruning and SH progression; (C) freeze topology and optionally geometry then final SH color polish. The joint stays frozen throughout all stages. If the user selects `sh_only`, skip A/B completely and run only C.

### 7.4 Rendering direction and rigid material appearance

`render_articulated(camera, f, G, J3, labels)` produces one `[N,3]` world-space mean tensor and one `[N,4]` quaternion tensor, formed from trainable canonical tensors plus **fixed** `Q_f,p_f`. Do not use CPU `float(q)` in differentiable joints (future optional stage needs autograd), do not `detach()` canonical Gaussian tensors in `gaussian_rgbd`, and do not mutate/overwrite canonical parameters by `set_time` in a persistent manner across views. Prefer `gaussian_renderer.render(... means3D_override=..., rotations_override=...)`, checking `compute_cov3D_python` as noted above. If `compute_cov3D_python=True` must refuse or correctly transport precomputed covariances.

Any SH evaluation dependent on material orientation should use canonical view directions; provide a legacy parity mode as documented in §3.4. Every moving Gaussian, including Phase-4 recovered geometry, uses the same `Q_f,p_f`; all static Gaussian means/rotations are unchanged.

### 7.5 Final artifacts

`phase5/{config_resolved.json,manifest.json,loss_log.csv,frame_camera_metrics.csv,renderer_parity.json,checkpoints/...,final/point_cloud.ply,final/material_ids.npz,final/gaussian_uid.npz,final/parent_uid.npz,final/appearance_settings.json,final/joint_unchanged_proof.json,renders/}`. The joint artifact must be a **bytewise-identical copy/reference** of the selected Phase-3 geometry and trajectory; emit SHA-256 before/after and explicit `joint_optimized=false`. Include final Gaussian count and how many came from each Phase-4 provenance class.

## 8. Separately optional final joint-only refinement — DO NOT RUN NOW

**Implementation requirement:** create a standalone CLI entry point and unit-test its mathematical parameterization; do **not** schedule it, invoke it by default, or modify Phase-5 checkpoints in place. Example future invocation:

```bash
python tools/refine_joint_geometry_final.py \
  --phase5-dir RESULTS/SCENE/phase5 \
  --phase3-dir RESULTS/SCENE/bidirectional/DIRECTION_joint \
  --joint-type prismatic_or_revolute \
  --steps 10 --full-sequence-sweep --enable
```

`--enable` must be explicitly provided. In default `--dry-run` mode report only changes that **would** be made. `--enable` without a completed Phase-5 `joint_unchanged_proof.json` or provenance-match check fails.

Freeze **all** Gaussian means/quaternions/scales/opacity/SH, topology, labels, cameras and exposure. Optimize only minimal joint geometry, using **all accepted moving Gaussians** and a complete camera/frame sweep. Default frozen `q`; optionally an explicit `--allow-q-correction` switch with very small prior-limited `delta q` (anchor delta fixed 0). Joint type never changes.

- P: axis direction `e` on `S²`, 2 tangent DOF; **no axis origin DOF**.
- R: axis direction `e` on `S²`, 2 tangent DOF, plus perpendicular axis-line displacement `delta a` in `e^perp`, 2 DOF; **no axis translation along itself**.
- Preserve angle in radians, slide in metres and physical source/world frame. Use normalized retraction and tangent-plane basis; alternatively reuse validated `estimator/joint_manifold.py` only if it is exactly equivalent to the selected `phase2_model_comparison` model's type-specific gauge. No unconstrained 6D screw representation that changes joint type.

\[
\min_{\delta g} \sum_{c,f}L_{cf}(G_5,J_3\oplus\delta g,q_3)
+\delta g^T\Lambda_g\delta g,
\quad \delta g\text{ constrained to a declared small physical trust region}.
\]

For optional `delta q`, add `sum_f (delta q_f/sigma_f)^2`, nonuniform-time smoothness, and anchor `delta q_a=0`. Compute/inspect Schur complement of joint geometry after accounting for q flexibility; if rank-deficient, retain Phase-3 geometry. Never assert an improved joint from image loss alone. Reject proposed corrections if they exceed predeclared angular/line-distance caps or worsen declared full-sequence validation. Export **separate** `phase5_joint_final/` including before/after axis, line, q, loss, trust-region size, complete audit, and `refinement_accepted` with source hashes. Must not overwrite Phase 3, Phase 4, or Phase 5.

**User instruction for present run:** `enabled=false`, joint refinement status `NOT_RUN_BY_DESIGN`.

## 9. Exact file implementation plan

Create minimal new modules; prefer reuse over cloned frameworks:

```text
recovery/
    __init__.py
    contracts.py                 # Phase 3/selected-direction ingestion, IDs, manifests
    articulated_scene.py        # canonical static/moving transport and SH convention
    rgbd_loss.py                # verified FreeArt-equivalent RGB+depth+alpha objective
    material_completion.py      # Phase 4A component hypotheses, scores, sensitivity
    residual_discovery.py       # Phase 4B unexplained pixel maps
    canonical_fusion.py         # masked RGB-D backprojection, joint inverse, layer-aware fusion
    candidate_acceptance.py     # Phase 4C local fit + full-scene evidence
    topology.py                 # material ID/provenance transfer during split/prune
    training.py                 # Phase 5 Gaussian-only trainer, sh_only branch
    joint_refine_final.py       # future *separate* tiny joint-only optimizer

tools/
    run_phase4_recovery.py
    train_phase5_rgbd_gaussians.py
    refine_joint_geometry_final.py

tests/
    test_phase45_contracts.py
    test_phase45_direction_anchor.py
    test_rgbd_cuda_forward_backward.py
    test_phase4_material_counterfactual.py
    test_phase4_canonical_birth.py
    test_phase4_layer_aware_fusion.py
    test_phase4_occlusion_acceptance.py
    test_phase5_immutable_joint.py
    test_phase5_topology_alignment.py
    test_phase5_full_sequence.py
    test_optional_final_joint_disabled.py
```

Modify only where necessary:

```text
gaussian_renderer/__init__.py
scene/gaussian_model.py
submodules/diff-gaussian-rasterization/diff_gaussian_rasterization/__init__.py
submodules/diff-gaussian-rasterization/{ext.cpp,rasterize_points.cu,rasterize_points.h}
submodules/diff-gaussian-rasterization/cuda_rasterizer/{forward.cu,forward.h,backward.cu,backward.h,rasterizer_impl.cu,rasterizer_impl.h}
```

Do NOT modify Phase-1/2/3 inference mathematics. Add optional hooks to `run_bidirectional_all.sh` only **after** standalone stages work; preserve `--direction auto` and the current `--stages` defaults. If hooking into stage cache, reuse `tools/stage_signature.py` and `tools/stage_cache.py` patterns, with an input signature hashing upstream Phase-3 artifact, masks, depth, cameras, renderer extension revision and exact mode.

## 10. Codex implementation order and mandatory gates

### Gate 0 — audit and unit conventions

Produce `phase45_repo_audit.md`: checked checkout commit, exact input files, selected direction, camera/depth conventions, Phase-3 joint selection status, supported output formats, potential SH inconsistency, model/renderer dependency compilation. Cite exact paths and line spans. No editing upstream data. Refuse missing/ambiguous source joint.

### Gate 1 — renderer and PyTorch autograd

Implement new RGB/depth/alpha CUDA forward+backward, new Python entry point, legacy compatibility and numeric oracles. Must pass both CPU scalar/compositing oracle and CUDA finite differences before continuing. A code compiling is not a correctness proof.

### Gate 2 — articulated canonical rendering

Implement P and R transport with static/moving masks and rotating Gaussian orientation/SH direction. Verify identity at anchor, correct original-frame mapping forward/backward, static invariance, rigid pairwise distances, covariance rotation, exact real pose relative anchor, and gradients. No joint optimization registered.

### Gate 3 — Phase 4A

Use one synthetic P and one synthetic R scene with deliberately withheld moving-core members and static neighbors. Require correct material classification where observable, `UNRESOLVED` for axis-near/occluded evidence, no false moving static wall, and immutable original count/IDs. Full-scene wrong-label counterfactual visibly worsens RGB-D-mask objective. Add static no-motion control.

### Gate 4 — Phase 4B

Synthetic drawer hidden side/interior and revolute back surface, as well as thin two-sided panel. Require late-visible RGB-D surface back-transport into *one* canonical moving part (or static when appropriate), no front/back collapse, no duplicate proposals and explicit never-seen regions. For backward processing order, ensure correct original-frame provenance and canonical transform.

### Gate 5 — Phase 4C

Run actual native CUDA RGB-D-mask render across every full-resolution camera/frame for finalists; test both an acceptably occluded-at-anchor patch and an incorrect early-front-occluding patch, ensure latter rejected. No accepted patch degrades all-view validation beyond declared tolerance. Check `N_final=N_original+N_accepted` before Phase-5 densification and all stable IDs/provenance.

### Gate 6 — Phase 5 Gaussian-only

Train `gaussian_rgbd` on synthetic P/R, then one real scene with correct direction. Report initial/final RGB, valid depth, alpha metrics for every view, plus all-frame peak memory/runtime, Gaussian count and material lineage. **Joint bytes and numerical parameters remain exactly unchanged.** Verify that disabling RGB affects only RGB gradients, disabling depth removes all depth loss gradients, and disabling mask removes alpha loss gradients. Test SH-only profile separately; assert geometry, opacity, joint are unchanged even after many updates, and depth/mask are treated as monitors.

### Gate 7 — optional final joint command, disabled

Verify default CLI and integrated runner never call joint optimizer and never register joint trainable parameters. Unit-test future P/R minimal geometry retraction, trust-region and all-Gaussian gradient. Do not execute it as part of normal Phase-5 training, real-scene evaluation, or acceptance of this task.

### Final mandatory report

Write `PHASE45_IMPLEMENTATION_REPORT.md` containing: modified files, compilation command, exact tests/pass/fail/skips with evidence, run commands, upstream commit hash, sensor/direction/joint manifests, per-scene before/after metrics, known caveats (especially mixed-edge depth, SH convention, joint ambiguity), GPU memory/runtime, and *whether the optional joint stage ran (`NO`)*. Never claim tests passed if they were not actually executed; clearly identify missing hardware/data.

## 11. End-to-end example commands (new CLIs to implement)

Paths are illustrative. These CLI flags and output names are **targets for the new implementation**, not existing working commands:

```bash
# Phase 0–3 already completed with --direction auto choosing ONE direction.
DIR=$(python -c 'import json,sys; print(json.load(open(sys.argv[1]))["recommended_direction"])' \
  "$SRC/metadata/direction/summary.json")
END=$([ "$DIR" = forward ] && echo initial || echo final)

# PHASE 4: recovery using the one chosen anchor and fixed Phase-3 J/q.
python tools/run_phase4_recovery.py \
  --source "$SRC" --direction "$DIR" \
  --model "$RESULTS/static_bidi/$END" \
  --phase2-dir "$RESULTS/bidirectional/${DIR}_rigid" \
  --phase3-dir "$RESULTS/bidirectional/${DIR}_joint" \
  --joint-type "$JOINT_TYPE" --cameras all \
  --frames-on-cpu --output "$RESULTS/phase4"

# PHASE 5: all-frame/camera Gaussian RGB-D-mask training; joint immutable.
python tools/train_phase5_rgbd_gaussians.py \
  --source "$SRC" --direction "$DIR" \
  --phase4-dir "$RESULTS/phase4" \
  --phase3-dir "$RESULTS/bidirectional/${DIR}_joint" \
  --joint-type "$JOINT_TYPE" --train-mode gaussian_rgbd \
  --frames-on-cpu --output "$RESULTS/phase5"

# Optional literal SH-only color polish (still NO joint refinement):
python tools/train_phase5_rgbd_gaussians.py \
  --source "$SRC" --direction "$DIR" \
  --phase4-dir "$RESULTS/phase4" \
  --phase3-dir "$RESULTS/bidirectional/${DIR}_joint" \
  --joint-type "$JOINT_TYPE" --train-mode sh_only \
  --frames-on-cpu --output "$RESULTS/phase5_sh_only"

# NOT NOW: separate future joint-only command, never chained into Phase 5.
# python tools/refine_joint_geometry_final.py ... --enable
```

## 12. Acceptance requirements / non-negotiable exclusions

**Required:** native CUDA RGB, differentiable alpha and metric-depth; exact selected-direction anchor; Phase-4 original-Gaussian reassignment; static/moving canonical fusion and only-when-observed births; full-sequence render acceptance including occlusion; phase-5 full Gaussian RGB-D-mask objective with immutable joint; lineage-safe topology; source-controlled outputs and all-frame audits; standalone disabled final joint optimizer.

**Forbidden:** rerunning both directions; opposite endpoint bootstrap/fusion; using GT to select a joint; FreeArt/Nerfstudio runtime replacement; per-Gaussian free motion; per-frame arbitrary 6D deformation; physically created/removed surfaces at first visibility; background depth fitted to foreground; mixed-depth edge smoothing that merges opposite faces; converting legacy inverse-depth to metric by blind inversion; treating source `q` as chronological index for backward; optimizing joint during ordinary Phase 5; claiming `sh_only` learns depth/alpha; silently falling back to software/renderer that changes physical semantics.

## 13. Self-contained Codex execution instruction

> Read **this entire specification** and audit the exact code first. Implement in the prescribed Gate 0 -> Gate 7 order, preserving all existing Phase 0–3 behavior. The current user requires a SINGLE selected reconstruction direction, FreeArtGS-style full RGB-D-mask loss implemented natively in the existing CUDA renderer, and Phase-4 recovery of missing moving/original and later-observed geometry with the fixed Phase-3 joint. Phase 5 optimizes only the canonical 3DGS visual model, **not joint geometry**. Implement the separate final joint-only refinement entry point but **DO NOT EXECUTE IT**, leave it disabled by default, and do not integrate it into the normal Phase-5 loop. If source structures/conventions differ from this audited baseline, explicitly reconcile them and update the audit before editing. Add tests, run every test supported by your environment, preserve upstream manifests, and finish with `PHASE45_IMPLEMENTATION_REPORT.md` stating exact results and unrun checks. Never report implementation or test success without evidence.
