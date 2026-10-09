# Codex Implementation Task — Native CUDA/Python Render-Guided Noising/Denoising for Static/Dynamic Discovery in RigEuler3DGS

## 0. Goal

Implement a **clean, repo-native static/dynamic discovery stage** inside the existing `RigEuler3DGS` repository.

The current rigid-motion/twist machinery works well and must be preserved. The failure to fix is the **initial per-Gaussian static/dynamic understanding**, especially false static / false dynamic labels caused by occlusions and disocclusions.

The new discovery must use the existing differentiable 3D Gaussian Splatting renderer itself:

1. start from the already-trained static/canonical frame-0 3DGS;
2. add small random 3D motion perturbations to the existing motion-grid cells;
3. render real later frames;
4. denoise the cell displacements using RGB/depth rendering gradients;
5. run several independent noise seeds;
6. accumulate **exact per-Gaussian alpha-compositing responsibility in CUDA**;
7. distinguish **VISIBLE / OCCLUDED / NO-EVIDENCE** from motion;
8. output only high-confidence moving seeds, high-confidence static evidence, and UNKNOWN;
9. fit the existing rigid twist from the moving seeds;
10. run a second render-guided static-vs-rigid gate optimization to expand the moving rigid link without falsely classifying hinge-near / temporarily occluded Gaussians.

This is **not a separate diffusion neural network** and **not DINO-based motion discovery**.

The useful diffusion idea is only:

```text
random perturbation -> render -> gradient denoising -> repeat
                    -> repeat from multiple random seeds
                    -> consistent nonzero attractor = motion evidence
```

The implementation should be directly integrated into the existing CUDA rasterizer and Python training code.

---

# 1. Non-negotiable constraints

## 1.1 Preserve the working rigid branch

Do not rewrite or replace:

- the explicit rigid twist `omega(t), u(t)`;
- the weighted twist fit;
- current rigid temporal smoothing;
- the final prismatic/revolute/screw estimation;
- the existing semantic-foreground isolation;
- the current dynamic transport once a reliable static/moving assignment exists.

The new code replaces/improves **motion discovery and label supervision**, not rigid kinematics.

## 1.2 No DINO for dynamic/static discovery

Do not use:

- DINO distances;
- DINO clustering;
- DINO temporal matching;
- DINO as a boundary score;

for the new static/dynamic decision.

The semantic foreground mask may remain and should be used to prevent background Gaussians from entering joint fitting.

## 1.3 Occlusion is not static and not dynamic

The implementation must explicitly maintain three discovery states:

```text
STATIC
DYNAMIC
UNKNOWN / OCCLUDED / INSUFFICIENT EVIDENCE
```

Never convert low gradient, low visibility, or absence from a view directly into STATIC.

Core invariant:

```text
no reliable observation != static
```

## 1.4 Do not implement an N x H x W responsibility tensor

That is too expensive.

Accumulate per-Gaussian scalar statistics directly in CUDA with atomics during the standard front-to-back splat traversal.

## 1.5 Extra CUDA passes must be probe-only

Normal RGB training should not pay this cost every iteration.

The new responsibility CUDA pass should run only during:

- initial render-guided motion discovery;
- periodic label refresh if enabled;
- explicit debug/validation.

---

# 2. Current repository integration points

Before editing, inspect the exact working tree and preserve all existing modifications.

Relevant paths currently include:

```text
RigEuler3DGS/
├── train.py
├── arguments/__init__.py
├── gaussian_renderer/__init__.py
├── scene/
│   ├── gaussian_model.py
│   ├── adaptive_motion_grid.py
│   └── joint_part_field.py
├── utils/
│   └── dynamic_loss_utils.py
└── submodules/diff-gaussian-rasterization/
    ├── ext.cpp
    ├── rasterize_points.h
    ├── rasterize_points.cu
    ├── diff_gaussian_rasterization/__init__.py
    └── cuda_rasterizer/
        ├── forward.h
        ├── forward.cu
        ├── backward.h
        ├── backward.cu
        ├── rasterizer.h
        ├── rasterizer_impl.h
        └── rasterizer_impl.cu
```

The nested rasterizer is already modified and currently contains:

- normal RGB/depth rendering;
- arbitrary per-Gaussian feature rendering;
- optical-flow statistics accumulation.

The new implementation should follow the same architectural pattern as the existing **flow-stat accumulation path**, not create an unrelated extension.

Do not overwrite the submodule with upstream GraphDeco files.

---

# 3. Exact alpha-compositing quantities to expose

For pixel `u`, Gaussians are traversed front-to-back in the same sorted order as normal 3DGS rendering.

For Gaussian `i`:

```text
alpha_i(u) = opacity_i * Gaussian2D_i(u)
```

using exactly the same:

- conic evaluation;
- alpha clamp;
- minimum-alpha cutoff;
- tile/bin ordering;

as the existing RGB kernel.

Before Gaussian `i`:

```text
T_i(u) = product_{j < i} (1 - alpha_j(u))
```

The exact rendered responsibility is:

```text
w_i(u) = T_i(u) * alpha_i(u)
```

This is the same coefficient multiplying RGB/features in standard alpha blending.

For every Gaussian and view accumulate four scalars:

```text
raw_support[i]          = sum_u alpha_i(u)
visible_mass[i]         = sum_u w_i(u)
valid_visible_mass[i]   = sum_u valid(u) * w_i(u)
weighted_residual[i]    = sum_u valid(u) * w_i(u) * residual(u)
```

Also optionally accumulate:

```text
weighted_residual_sq[i] = sum_u valid(u) * w_i(u) * residual(u)^2
```

The mean render residual assigned to Gaussian `i` is:

```text
mean_residual[i] = weighted_residual[i] /
                   max(valid_visible_mass[i], eps)
```

The alpha visibility ratio is:

```text
visibility_ratio[i] = visible_mass[i] /
                      max(raw_support[i], eps)
```

The observation-valid fraction is:

```text
valid_fraction[i] = valid_visible_mass[i] /
                    max(visible_mass[i], eps)
```

---

# 4. IMPORTANT CUDA detail: raw support must survive full occlusion

Do not compute `raw_support` only until standard transmittance early termination.

That would make a fully occluded Gaussian appear identical to an out-of-frustum Gaussian.

In the dedicated responsibility-statistics kernel:

1. continue iterating over the complete tile Gaussian range even after `T` becomes tiny;
2. keep accumulating:

```cpp
raw_support[id] += alpha;
```

for every overlapping splat passing the normal Gaussian/alpha cutoff;
3. once `T` is essentially zero, visible contributions can be skipped for efficiency, but the loop must continue for raw support.

This enables:

```text
raw_support ~ 0
    -> outside image / no projected support

raw_support > 0 and visible_mass/raw_support ~ 0
    -> geometrically projects but is alpha-occluded
```

This distinction is required.

---

# 5. Native CUDA API to add

Use names consistent with the current rasterizer naming.

Recommended public Python facade:

```python
GaussianRasterizer.accumulate_responsibility_stats(...)
```

Recommended C++/CUDA binding:

```text
_C.accumulate_gaussian_responsibility_stats
```

Exact names may be adapted to current style, but use one clean path.

## 5.1 Inputs

The native call should accept the same geometry needed by the existing flow-stat path:

```text
means3D            [N,3]
opacities          [N,1]
scales             [N,3] or empty
rotations          [N,4] or empty
cov3D_precomp      [...] or empty
viewmatrix
projmatrix
tan_fovx
tan_fovy
image_height
image_width
campos
scale_modifier
prefiltered
antialiasing
```

and the new per-pixel inputs:

```text
pixel_residual     [H,W] float32 CUDA
pixel_valid        [H,W] float32/bool CUDA
```

`pixel_valid` must default to all ones when omitted at the Python level.

## 5.2 Outputs

Return contiguous CUDA float tensors:

```text
raw_support          [N]
visible_mass         [N]
valid_visible_mass   [N]
weighted_residual    [N]
weighted_residual_sq [N]   # optional but recommended
radii                 [N]   # optional if already naturally available
```

These statistics are **detached/non-differentiable**. No backward kernel is needed for this statistics API.

The noising/denoising displacement itself remains differentiable through the existing RGB/depth renderer.

---

# 6. CUDA implementation details

## 6.1 Add a dedicated kernel in `cuda_rasterizer/forward.cu`

Mirror the existing flow-stat accumulation kernel and reuse the same preprocessing/binning infrastructure.

Pseudo-code per pixel:

```cpp
float T = 1.0f;

for every Gaussian id in the complete sorted tile range:
{
    // exact same 2D Gaussian/conic computation as RGB renderer
    float power = ...;
    if (power > 0.0f)
        continue;

    float alpha = min(0.99f, conic_opacity.w * exp(power));

    // use exactly the same alpha cutoff as current renderer
    if (alpha < ALPHA_MIN)
        continue;

    // IMPORTANT: projected support is independent of transmittance.
    atomicAdd(&raw_support[id], alpha);

    float w = T * alpha;

    if (w > RESPONSIBILITY_EPS)
    {
        atomicAdd(&visible_mass[id], w);

        float valid = pixel_valid[pix_id];
        if (valid > 0.0f)
        {
            float vw = valid * w;
            float r = pixel_residual[pix_id];

            atomicAdd(&valid_visible_mass[id], vw);
            atomicAdd(&weighted_residual[id], vw * r);
            atomicAdd(&weighted_residual_sq[id], vw * r * r);
        }
    }

    T *= (1.0f - alpha);

    // DO NOT mark the thread done merely because T is tiny.
    // We still need raw_support for deeper fully occluded Gaussians.
    // We may stop visible/residual accumulation, but continue alpha evaluation.
}
```

Use current renderer constants rather than inventing inconsistent ones.

## 6.2 Avoid shared-memory explosion

Outputs are one scalar per Gaussian, so use global `atomicAdd` initially.

Do not allocate per-pixel x per-Gaussian buffers.

If performance later becomes a problem, optimize with tile/block reductions, but correctness comes first.

## 6.3 Keep responsibility statistics float32

Do not use fp16 for these accumulators.

Small alpha masses and long visibility sums need stable accumulation.

---

# 7. C++ / PyBind files to modify

Wire the API all the way through the current extension.

At minimum inspect/update:

```text
submodules/diff-gaussian-rasterization/ext.cpp
submodules/diff-gaussian-rasterization/rasterize_points.h
submodules/diff-gaussian-rasterization/rasterize_points.cu
submodules/diff-gaussian-rasterization/cuda_rasterizer/forward.h
submodules/diff-gaussian-rasterization/cuda_rasterizer/forward.cu
submodules/diff-gaussian-rasterization/cuda_rasterizer/rasterizer.h
submodules/diff-gaussian-rasterization/cuda_rasterizer/rasterizer_impl.h
submodules/diff-gaussian-rasterization/cuda_rasterizer/rasterizer_impl.cu
submodules/diff-gaussian-rasterization/diff_gaussian_rasterization/__init__.py
```

Follow the existing `accumulate_flow_stats` implementation structure wherever possible.

Do not modify normal RGB backward just to compute these detached statistics.

---

# 8. Python rasterizer facade

In:

```text
submodules/diff-gaussian-rasterization/diff_gaussian_rasterization/__init__.py
```

add something equivalent to:

```python
@torch.no_grad()
def accumulate_responsibility_stats(
    self,
    means3D,
    opacities,
    pixel_residual,
    pixel_valid=None,
    scales=None,
    rotations=None,
    cov3D_precomp=None,
):
    ...
```

Validation requirements:

- CUDA only;
- contiguous tensors;
- residual shape exactly H x W or 1 x H x W;
- finite residual values;
- valid mask converted to float32 0/1;
- exactly one of `(scales, rotations)` or `cov3D_precomp` according to existing rasterizer convention;
- return a dictionary or named tuple with explicit names.

Do not hide shape errors.

---

# 9. `gaussian_renderer` wrapper

In:

```text
gaussian_renderer/__init__.py
```

add a high-level helper:

```python
@torch.no_grad()
def accumulate_gaussian_responsibility(
    viewpoint_camera,
    pc,
    pipe,
    pixel_residual,
    pixel_valid=None,
    scaling_modifier=1.0,
    means3D_override=None,
):
    ...
```

It must use:

- the same camera matrices;
- same scales/covariance;
- same rotations;
- same opacity;
- same current/overridden Gaussian positions;

as the RGB render corresponding to the hypothesis being scored.

The wrapper returns:

```python
{
    "raw_support": ...,
    "visible_mass": ...,
    "valid_visible_mass": ...,
    "weighted_residual": ...,
    "weighted_residual_sq": ...,
    "mean_residual": ...,
    "visibility_ratio": ...,
    "valid_fraction": ...,
    "projected": ...,
    "alpha_visible": ...,
    "evidence_valid": ...,
}
```

Thresholding should remain configurable at the higher discovery layer, not hard-coded into CUDA.

---

# 10. Add clean render-position override support

Do not mutate `pc._xyz` or abuse persistent `pc._current_xyz` inside the probe optimizer.

Extend the renderer cleanly so temporary hypotheses can be rendered with an explicit position tensor.

Recommended API:

```python
render(
    ...,
    means3D_override=None,
    rotations_override=None,
)
```

When `means3D_override` is supplied:

```text
means3D = means3D_override
```

instead of `pc.get_xyz`.

It must preserve gradients to `means3D_override`.

Normal training behavior with `None` must remain bitwise/functionally unchanged.

The responsibility wrapper must accept the same position override so the statistics exactly correspond to the rendered hypothesis.

---

# 11. Pixel residual definition

Do not use SSIM as the per-pixel responsibility map because the local window makes attribution less direct.

Use a robust local scalar map:

```python
rgb_residual = abs(render_rgb - gt_rgb).mean(dim=0)
```

Optionally robustify:

```python
rho(x) = sqrt(x*x + eps*eps)
```

Depth can be added when reliable:

```python
depth_residual = abs(render_depth_metric - gt_depth_metric)
```

Normalize RGB and depth terms robustly before combination:

```text
pixel_residual = lambda_rgb * rgb_residual_norm
               + lambda_depth * depth_residual_norm
```

Do not accidentally compare metric depth to inverse depth.

The main differentiable probe optimization may still use the normal RGB + SSIM + depth training loss. The scalar responsibility map is only for per-Gaussian evidence attribution.

---

# 12. Observation-valid mask and occlusion handling

There are two different visibility questions:

1. **Does this Gaussian project and survive alpha compositing under the current hypothesis?**
2. **Does the current hypothesis correspond to an observed surface that can provide reliable label evidence?**

The CUDA quantities answer the first.

For the second, build `pixel_valid` in Python.

## 12.1 Base valid mask

Start from:

```text
image bounds
AND finite RGB
AND available alpha/foreground acquisition validity
AND reliable depth mask when depth is used
```

## 12.2 Depth front-occlusion indicator

When reliable metric depth exists, compute a detached diagnostic:

```text
observed_front_occluder(u) =
    gt_depth(u) + depth_margin < rendered_depth(u)
```

Meaning: the real observation contains a surface clearly in front of the surface rendered by this hypothesis.

Do **not** use this to globally remove the pixel from the differentiable noising/denoising render loss, because a moving object appearing at a new location is itself a useful residual.

Use it only to qualify per-hypothesis/per-Gaussian evidence.

Recommended:

```python
pixel_valid_for_responsibility = base_valid & ~observed_front_occluder
```

This means a deeper currently rendered Gaussian does not receive a false STATIC/DYNAMIC vote from a pixel where the real scene says another surface is in front.

Important: evaluate this separately for every hypothesis. A correctly moved hypothesis may become depth-consistent and therefore valid even when the static hypothesis was invalid.

## 12.3 If depth is unavailable

Do not pretend alpha compositing alone proves observation visibility.

Fall back to:

- alpha `visibility_ratio`;
- multiview agreement;
- multiple noise-seed agreement;
- temporal recurrence;

and keep uncertain cases UNKNOWN more aggressively.

---

# 13. Stage A — render-guided noisy cell-displacement discovery

Implement a dedicated module, suggested path:

```text
utils/render_guided_motion_discovery.py
```

Do not put a giant block directly inside `train.py`.

Recommended entry point:

```python
def discover_motion_from_render_denoising(
    gaussians,
    scene,
    pipe,
    dyn,
    output_dir,
):
    ...
```

## 13.1 Operate on semantic foreground only

The canonical full scene remains rendered, but only semantic-foreground Gaussians receive probe displacements and labels.

Background remains at canonical position and never enters joint fitting.

## 13.2 Use the existing motion-grid cells

Prefer the existing `motion_grid.gaussian_cell_id`.

If the regular motion grid is not initialized yet at the point discovery runs, factor its initialization cleanly so it can be created before joint labels.

Do not create an unrelated second spatial partition unless absolutely necessary.

For active cell `c`, create one displacement parameter:

```text
delta_c in R^3
```

Every foreground Gaussian `i` in the cell uses:

```text
x_i = x_i0 + delta_cell[cell_id(i)]
```

## 13.3 Noise initialization

For each selected probe frame and independent seed:

```python
delta = sigma * torch.randn(num_cells, 3, device="cuda")
delta.requires_grad_(True)
```

Scale `sigma` relative to scene/object scale or median foreground Gaussian spacing, not a hard-coded meter value.

Recommended initial default:

```text
sigma = 0.25 - 0.5 * median foreground KNN spacing
```

Expose it as a CLI multiplier.

## 13.4 Denoising objective

Optimize only temporary `delta` parameters.

Canonical geometry/appearance remains frozen.

For probe frame `t`, render all or a subset of synchronized cameras:

```text
x_hypothesis = x0 + delta[cell_id]
```

Use:

```text
L_probe = L_rgb
        + lambda_ssim * L_ssim
        + lambda_depth * L_depth
        + lambda_zero * mean(||delta_c||^2)
        + lambda_neighbor * optional spatial smoothness
```

The zero-motion prior is important:

```text
move only if motion meaningfully improves the real observations
```

Do not use current static/dynamic probabilities to gate this probe.

Do not use the rigid twist in Stage A.

This stage must discover motion independently enough to bootstrap the rigid fit.

## 13.5 Optimization schedule

Keep it small.

Initial defaults:

```text
probe frames: 4 temporal positions across the sequence
noise seeds per frame: 4
denoising steps: 8-12
Adam or SGD on delta only
```

The full pipeline target is still approximately 40-45 minutes, so this should remain a short initialization pass.

---

# 14. Per-seed/per-view evidence extraction

After each noisy denoising run, compute responsibility stats for:

1. the static hypothesis `delta = 0`;
2. the final denoised hypothesis `delta = delta*`.

For every Gaussian, store:

```text
static_raw_support
static_visible_mass
static_valid_visible_mass
static_mean_residual

moved_raw_support
moved_visible_mass
moved_valid_visible_mass
moved_mean_residual

final_displacement
```

Do not decide static/dynamic from a single view.

---

# 15. View-state logic

For each Gaussian/view/hypothesis define:

```text
projected = raw_support > tau_raw
alpha_visible = visibility_ratio > tau_alpha_vis
observation_valid = valid_visible_mass > tau_valid_mass
                    AND valid_fraction > tau_valid_fraction
```

Then:

```text
not projected
    -> NO VIEW EVIDENCE

projected but not alpha_visible
    -> ALPHA OCCLUDED

alpha_visible but not observation_valid
    -> OBSERVATION OCCLUDED / DEPTH INCONSISTENT / UNKNOWN

observation_valid
    -> may vote
```

Thresholds must be CLI/configurable and logged.

Use robust scene-relative normalization for masses. Do not assume one universal absolute visible-mass threshold across datasets.

---

# 16. Multi-noise consistency

For Gaussian `i`, each seed yields final displacement:

```text
d_i^r
```

Compute mean displacement:

```text
mean_d_i = mean_r d_i^r
```

Directional agreement:

```text
C_dir_i = ||sum_r d_i^r|| /
          (sum_r ||d_i^r|| + eps)
```

Also compute magnitude agreement, for example coefficient of variation or robust MAD:

```text
C_mag_i = 1 - normalized_MAD(||d_i^r||)
```

Combine only over seeds/views with observation-valid support.

High-confidence motion requires repeated convergence toward a consistent nonzero displacement, not merely one good random perturbation.

---

# 17. Render-improvement evidence

For a valid view/hypothesis pair:

```text
DeltaE_i = E_static_i - E_moved_i
```

Positive means the moved hypothesis locally explains the observed pixels better.

Do not require both hypotheses to be valid in every case.

Use the following logic:

## 17.1 Both static and moved valid

```text
DeltaE >> 0 + consistent nonzero displacement
    -> dynamic evidence

DeltaE <= small + displacement returns near zero
    -> static evidence
```

## 17.2 Static invalid, moved valid

This may happen when the static hypothesis is hidden/inconsistent but the moved hypothesis lands on the observed surface.

Allow DYNAMIC evidence only if:

- multiple noise seeds agree strongly;
- multiple cameras or times support it;
- moved hypothesis has good valid responsibility;
- render residual improves substantially.

Otherwise UNKNOWN.

## 17.3 Static valid, moved invalid

This is strong STATIC evidence if the static residual is already low and the noisy optimization repeatedly returns toward zero.

## 17.4 Both invalid

Always UNKNOWN.

This rule is essential to avoid false static labels from occlusion.

---

# 18. Stage-A output probabilities

Do not force every Gaussian into a binary class.

Create detached evidence values:

```text
moving_seed_score[i] in [0,1]
static_seed_score[i] in [0,1]
unknown_score[i]
```

Suggested high-level ingredients:

```text
moving_seed_score ~
    positive_render_improvement
  * displacement_magnitude
  * seed_direction_consistency
  * seed_magnitude_consistency
  * multiview_support
  * temporal_support
```

Static evidence requires:

```text
valid visible support
AND repeated return near zero
AND no meaningful moved-hypothesis improvement
AND preferably low static render residual
```

UNKNOWN dominates when observational support is insufficient.

Initial `JointPartField` probabilities should be seeded approximately as:

```text
strong static  -> p_dynamic = 0.05 - 0.15
strong moving  -> p_dynamic = 0.85 - 0.95
unknown        -> p_dynamic = 0.50
```

Do not initialize unknown Gaussians as static.

---

# 19. Fit the existing rigid twist from high-confidence moving seeds

After Stage A:

```text
moving_seed_mask = moving_seed_score >= tau_seed
```

Use only these high-confidence moving seeds to bootstrap the existing rigid twist.

Do not modify the core weighted twist solver unless a small interface change is needed to accept an explicit seed weight/mask.

Suggested weight:

```text
twist_weight_i = moving_seed_score_i * observation_confidence_i
```

The existing rigid fit should then produce:

```text
omega(t), u(t)
```

from real render-discovered motion rather than appearance clustering.

---

# 20. Stage B — noisy static-vs-rigid gate denoising

Stage A intentionally finds only high-confidence motion seeds. It will miss:

- hinge-near Gaussians with almost zero displacement;
- temporarily occluded members of the moving link;
- weakly textured moving areas.

Use the already-good rigid model to expand the entire moving link.

## 20.1 Temporary gate variable

For each motion-grid cell create a temporary logit:

```text
z_c
p_c = sigmoid(z_c)
```

The two physical hypotheses are:

```text
x_static_i(t) = x_i0
x_rigid_i(t)  = rigidly integrated position from existing omega(t), u(t)
```

Blend only for this discovery optimization:

```text
x_gate_i(t) = x_static_i(t)
            + p_cell[c(i)] * (x_rigid_i(t) - x_static_i(t))
```

This is not the final production transport rule. It is a differentiable counterfactual used to classify membership.

## 20.2 Noise the gate

For each gate seed initialize:

```text
z_c = sigma_gate * randn(...)
```

so the optimization starts around uncertain `p ~= 0.5` with different perturbations.

Optimize through the real renderer for a small number of steps.

## 20.3 Gate objective

```text
L_gate = real RGB/SSIM/depth render loss
       + lambda_binary * p*(1-p)
       + lambda_spatial * local gate coherence
       + optional weak prior from Stage-A seeds
```

Stage-A high-confidence moving/static seeds may softly anchor the corresponding cells, but UNKNOWN cells must remain free.

Do not detach `p_c` in this separate discovery optimization.

## 20.4 Occlusion handling

After each gate run, use the native responsibility-statistics pass again.

A cell/Gaussian receives a confident final gate vote only where it has observation-valid responsibility in enough views/times.

Occluded intervals do not vote static.

The final classification can be propagated through temporal recurrence and rigid consistency.

---

# 21. Replace the current weak initialization, not the rigid training

The current XYZ/DINO k-medoids static/moving initialization should no longer be the default for joint motion discovery.

Implement a new mode, suggested flag:

```text
--joint_init_mode render_denoise
```

Possible choices for backwards compatibility:

```text
render_denoise   # new default for joint mode
legacy_xyz_dino  # old behavior, explicit only
```

The new render-denoise mode should initialize `JointPartField` from render-derived probabilities/evidence.

Do not delete old code immediately if it is useful for regression comparison, but ensure it is not silently used by the new mode.

---

# 22. Update dynamic label supervision

The existing label target compares proposal velocity against zero vs rigid velocity:

```text
e_static = ||v_proposal||
e_rigid  = ||v_proposal - v_rigid||
```

Keep this only as a **secondary physics/velocity regularizer**.

Add render-derived label evidence as the primary external observation target.

Suggested new loss:

```text
L_render_label = weighted BCE(p_dynamic, q_render)
```

where:

```text
q_render = render-derived probability
weight   = observation/evidence confidence
```

For UNKNOWN points:

```text
weight = 0
```

They must not be pushed toward static simply to satisfy BCE.

The existing velocity-space target can remain:

```text
L_velocity_label
```

with a smaller weight.

Final concept:

```text
real RGB/depth render evidence -> primary label evidence
rigid velocity consistency     -> propagation / regularization
```

---

# 23. Adaptive-grid interaction

The current adaptive grid should use the new evidence.

Add per-cell diagnostics/score terms such as:

```text
seed displacement disagreement
seed direction disagreement
render-improvement variance
dynamic-probability variance
visibility/evidence disagreement
static-vs-rigid gate disagreement
```

Split a cell when different Gaussians inside it have incompatible motion/evidence.

Do not use DINO variance for this new discovery mode.

Cells with insufficient visibility should not split just because labels are uncertain.

---

# 24. CLI arguments

Add a coherent group in `arguments/__init__.py`.

Suggested names/defaults:

```text
--joint_init_mode render_denoise

--render_discovery_num_frames 4
--render_discovery_num_seeds 4
--render_discovery_steps 10
--render_discovery_noise_scale 0.35
--render_discovery_lr 1e-2

--render_discovery_lambda_rgb 1.0
--render_discovery_lambda_ssim 0.2
--render_discovery_lambda_depth 0.2
--render_discovery_lambda_zero 1e-3
--render_discovery_lambda_neighbor 0.0

--render_resp_min_raw_support ...
--render_resp_min_visible_mass ...
--render_resp_min_visibility_ratio ...
--render_resp_min_valid_fraction ...
--render_resp_depth_margin ...

--render_motion_min_displacement ...
--render_motion_min_seed_consistency 0.7
--render_motion_min_view_support 2
--render_motion_min_improvement ...

--render_gate_num_seeds 3
--render_gate_steps 8
--render_gate_noise_scale 0.5
--render_gate_lr 1e-2
--render_gate_lambda_binary ...

--lambda_render_dynamic_label ...
--lambda_velocity_dynamic_label ...

--render_discovery_debug
--render_discovery_refresh_interval 0
```

Scene-scale-dependent thresholds should be derived from robust statistics where possible. Log the effective derived values.

---

# 25. Runtime requirements

The complete pipeline target remains about 40-45 minutes.

Do not run the full discovery every training iteration.

Recommended default schedule:

```text
once at dynamic-stage initialization
+ optional one refresh after rigid handover has partially stabilized
```

Use synchronized multiview frame groups efficiently:

- choose 4 informative timestamps across the interaction;
- use 2-4 cameras per optimization step if all cameras are too expensive;
- rotate camera subsets between steps;
- use all cameras when extracting final detached responsibility statistics if affordable.

Print exact timings:

```text
[RenderDiscovery] Stage A ... seconds
[RenderDiscovery] responsibility stats ... seconds
[RenderDiscovery] rigid seed fit ... seconds
[RenderDiscovery] Stage B ... seconds
[RenderDiscovery] total ... seconds
```

---

# 26. Required debug outputs

Create:

```text
<model_dir>/render_motion_discovery/
```

Save:

```text
config.json
summary.json
per_gaussian_scores.pt
per_cell_scores.pt
probe_frames.json
```

For every probe camera/frame save images:

```text
static_render.png
gt.png
static_rgb_residual.png
moved_render_seedXX.png
moved_rgb_residual_seedXX.png
pixel_valid_static.png
pixel_valid_moved.png
```

Create 3D HTML/debug outputs showing Gaussians colored by:

```text
raw projected support
alpha visibility ratio
observation-valid fraction
static mean residual
moved mean residual
render improvement
displacement magnitude
seed consistency
moving seed score
static seed score
UNKNOWN mask
final p_dynamic
```

Occluded/unknown should have a distinct neutral color, never be merged visually with static.

---

# 27. Validation tests — mandatory

## 27.1 CUDA alpha-responsibility unit test

Create a tiny synthetic scene with two overlapping Gaussians along one camera ray.

Verify:

```text
front Gaussian:
    visible_mass/raw_support high

back Gaussian:
    raw_support > 0
    visible_mass/raw_support much lower
```

Then reduce front opacity and verify back visibility increases monotonically.

## 27.2 Out-of-frustum test

Move a Gaussian outside the image.

Verify:

```text
raw_support ~= 0
visible_mass ~= 0
```

This must be distinguishable from a fully occluded in-frustum Gaussian.

## 27.3 Residual attribution test

Create a residual image with one nonzero pixel region.

Verify only Gaussians with nonzero alpha-composited responsibility there receive substantial `weighted_residual`.

## 27.4 Occluded static Gaussian test

Synthetic static object A is hidden temporarily by object B.

The hidden Gaussian must become:

```text
UNKNOWN / OCCLUDED
```

not DYNAMIC and not confidently STATIC during the hidden interval.

When it reappears at the same canonical position, its accumulated temporal evidence should support STATIC.

## 27.5 Moving Gaussian test

A rigid part moves between frame 0 and later frames.

Across noisy displacement seeds:

```text
delta* converges consistently to a nonzero direction
render residual decreases
moving_seed_score becomes high
```

## 27.6 Hinge-near moving Gaussian test

A Gaussian close to a revolute joint axis moves very little.

Stage A may leave it UNKNOWN.

After fitting the rigid twist from other moving seeds, Stage B must classify it as part of the moving rigid link from the static-vs-rigid render gate / rigid consistency, rather than permanently static because speed is small.

## 27.7 No-regression rasterizer test

With responsibility statistics disabled:

- RGB render must match previous output;
- depth render must match;
- arbitrary feature render must match;
- existing optical-flow statistics must still work;
- gradients for normal training must be unchanged.

---

# 28. Smoke test on `USB_100109`

After unit tests, run a short real-data smoke test.

Use the existing trained static frame-0 PLY and semantic foreground.

Stage A only first:

```text
4 probe frames
4 seeds
8 denoising steps
```

Do not immediately launch full dynamic training.

Inspect whether:

1. moving-link cells converge to coherent nonzero displacement;
2. static foreground cells return near zero;
3. alpha-occluded Gaussians are marked UNKNOWN;
4. multiview-visible Gaussians remain well constrained;
5. raw-support and visible-mass maps make geometric sense.

Then enable the rigid-seed fit and Stage B.

Compare against the existing initialization with diagnostics:

```text
number confident static
number confident dynamic
number unknown
mean visibility ratio per state
mean valid fraction per state
mean render improvement per state
seed consistency per state
```

If GT dynamic/static labels exist in SAPIEN metadata, use them **only for evaluation**, never as training supervision.

Report precision/recall specifically for:

```text
false dynamic due to occlusion
false static due to occlusion
moving-link recall
static precision
unknown fraction
```

---

# 29. Rebuild procedure

Because this modifies the nested CUDA extension, rebuild after implementation.

For the current RTX 4070 Laptop / Ada environment, use the current environment's CUDA/PyTorch-compatible build setup. A known working form in this repository has been:

```bash
cd ~/Joint3DGS/RigEuler3DGS
conda activate rigeuler3dgs

rm -rf submodules/diff-gaussian-rasterization/build
rm -rf submodules/diff-gaussian-rasterization/*.egg-info

export CUDA_HOME=/usr/local/cuda-12.8
export PATH="$CUDA_HOME/bin:$PATH"
export TORCH_CUDA_ARCH_LIST="8.9"
export FORCE_CUDA=1
export MAX_JOBS=4

pip install --no-build-isolation -v ./submodules/diff-gaussian-rasterization
```

Do not assume `/usr/local/cuda` points to the intended toolkit without checking.

After rebuild verify:

```bash
python - <<'PY'
import torch
import diff_gaussian_rasterization as dgr
print(torch.__version__)
print(torch.version.cuda)
print(torch.cuda.is_available())
print(hasattr(dgr.GaussianRasterizer, "accumulate_responsibility_stats"))
PY
```

---

# 30. Implementation order

Implement in this exact order and test after each block.

## Phase 1 — native visibility/responsibility statistics

1. CUDA accumulation kernel.
2. rasterizer_impl dispatch.
3. C++ host wrapper.
4. PyBind binding.
5. Python rasterizer facade.
6. `gaussian_renderer` wrapper.
7. synthetic visibility tests.

Do not proceed if visibility tests fail.

## Phase 2 — renderer override support

1. clean `means3D_override` path;
2. ensure gradients reach override positions;
3. regression-test normal render.

## Phase 3 — Stage-A noising/denoising

1. select probe times/cameras;
2. initialize per-cell noise;
3. optimize temporary displacement;
4. extract responsibility stats;
5. aggregate seed/view/time evidence;
6. output STATIC/DYNAMIC/UNKNOWN seeds.

## Phase 4 — rigid bootstrap

Use existing rigid twist fitting from high-confidence moving seeds.

## Phase 5 — Stage-B static-vs-rigid gate denoising

1. noisy gate initialization;
2. differentiable render-based gate optimization;
3. responsibility-qualified evidence;
4. update `JointPartField` initialization/targets.

## Phase 6 — dynamic training integration

1. make render evidence primary label supervision;
2. retain velocity expert comparison as secondary regularizer;
3. preserve existing rigid transport and final joint estimation.

---

# 31. Code-quality requirements

- Do not create patch scripts that rewrite source with regex.
- Edit the actual repository files cleanly.
- Do not duplicate rasterizer geometry preprocessing.
- Reuse the existing flow-stat path and existing rendering helpers.
- Do not add dead compatibility aliases unless required.
- Add clear docstrings/comments explaining the distinction between:
  - projected support;
  - alpha visibility;
  - observation-valid evidence;
  - static/dynamic/unknown.
- Fail loudly on tensor/device/shape mismatch.
- Keep all new outputs checkpoint-safe if any persistent state is added.
- Do not save giant dense responsibility tensors.
- Ensure background never enters rigid fitting.
- Ensure DINO is not silently used by `render_denoise` mode.

---

# 32. Core acceptance criterion

The implementation is successful only if the label logic becomes:

```text
A Gaussian is STATIC
    because it was reliably visible and repeated noisy motion hypotheses
    failed to improve the real observations / denoised back to zero,

NOT because it had low gradient or disappeared.

A Gaussian is DYNAMIC
    because multiple noisy 3D hypotheses consistently converge to a
    nonzero motion that improves real multiview rendering, or because
    the recovered rigid-body hypothesis consistently explains it better,

NOT because it had a high raw RGB residual.

A Gaussian is UNKNOWN/OCCLUDED
    when the renderer/observation cannot provide reliable evidence.
```

The desired pipeline is therefore:

```text
STATIC FRAME-0 3DGS
        |
        v
semantic foreground only
        |
        v
existing motion-grid cells
        |
        v
random 3D cell displacement noise
        |
        v
render real later multiview frames
        |
        v
RGB/SSIM/depth gradient denoising
        |
        +-------------------------------+
        | multiple seeds / views / times|
        +-------------------------------+
        |
        v
native CUDA alpha responsibility
raw alpha / T*alpha / valid T*alpha / residual*T*alpha
        |
        v
STATIC / MOVING-SEED / UNKNOWN
        |
        v
EXISTING RIGID TWIST FIT
        |
        v
noisy static-vs-rigid gate denoising
        |
        v
full moving rigid link + static + unknown
        |
        v
existing rigid dynamic optimization
        |
        v
existing joint type / axis estimation
```

Do not stop at merely exposing visibility. Complete the full Stage-A + Stage-B integration and validate the occlusion failure modes.
