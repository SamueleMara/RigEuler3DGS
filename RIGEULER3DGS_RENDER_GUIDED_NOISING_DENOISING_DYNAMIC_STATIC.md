# RigEuler3DGS: Render-Guided 3DGS Noising/Denoising for Occlusion-Robust Static/Dynamic Discovery

## Implementation context for Codex

**Target repository:** `~/Joint3DGS/RigEuler3DGS`  
**Primary target:** replace the weak initial static/dynamic discovery while preserving the rigid-motion machinery that is already working well.  
**Representation:** canonical frame-0 3D Gaussian Splatting model + differentiable renderer + multiview dynamic RGB/depth sequence.  
**Do not use DINO for static/dynamic discovery.** Semantic foreground gating may remain because it is already useful for isolating the articulated object from unrelated background.

---

# 1. Goal

Implement a self-supervised static/dynamic discovery stage that uses the useful idea behind diffusion — **perturb a latent state, render it, then repeatedly denoise it using observation likelihood** — directly in **3D Gaussian motion space**.

The method must answer:

> Does a Gaussian/cell need a coherent nonzero 3D motion to explain the real multiview video, or is the apparent change caused by occlusion/disocclusion, weak visibility, reconstruction error, or photometric noise?

The central requirements are:

1. **No separate RGB/video diffusion network.**
2. **No DINO-based static/dynamic initialization or boundary reasoning.**
3. Preserve the trained canonical frame-0 3DGS geometry and appearance.
4. Perturb only temporary motion variables, never the canonical Gaussian centers themselves.
5. Use the differentiable 3DGS renderer as the denoising/score oracle.
6. Use multiple random perturbation seeds so true motion must be reproducible.
7. Use multiview agreement so a single occlusion boundary cannot create a dynamic label.
8. Treat insufficient visibility as **unknown**, never as static.
9. Treat inconsistent render gradients as **ambiguous/unknown**, never as dynamic.
10. Fit the existing rigid twist only from high-confidence moving seeds.
11. After the rigid twist is known, classify the full moving link with a second render-guided **static-vs-rigid hypothesis test**. This is required to recover slow/hinge-near points that have little translational displacement.
12. Keep the existing rigid-motion fitting, temporal integration, joint-type estimation, and downstream pipeline unless a change is explicitly required below.

The desired flow is:

```text
canonical frame-0 3DGS
        |
        v
semantic foreground only
        |
        v
coarse motion-grid cells
        |
        v
random 3D displacement noise per cell
        |
        v
multiview render of real later frame
        |
        v
RGB + optional depth loss
        |
        v
backprop only to temporary cell displacement
        |
        v
repeat short denoising trajectory
        |
        +-------------------------------+
        |                               |
        v                               v
reproducible nonzero motion        weak/inconsistent evidence
        |                               |
        v                               v
high-confidence moving seed        static candidate / unknown
        |
        v
EXISTING rigid twist fit omega(t), u(t)
        |
        v
render-guided noisy static-vs-rigid gate optimization
        |
        v
static / moving / unknown probabilities
        |
        v
existing rigid integration and later joint estimation
```

---

# 2. Why the current assignment must be replaced

In the current joint implementation, the rigid part is much stronger than the discovery part.

The current assignment path is approximately:

```text
canonical XYZ + DINO
        |
        v
K+1 k-medoids initialization
        |
        v
largest cluster called static
other cluster called moving
        |
        v
trainable cosine part field
```

Later the label target compares the unrestricted proposal velocity to two velocity-space experts:

\[
e_i^S = \|v_i^{prop}\|,
\]

\[
e_i^R = \|v_i^{prop} - v_i^{rigid}\|,
\]

\[
q_i = \sigma\left(\frac{e_i^S-e_i^R}{T}\right).
\]

This asks:

> Which velocity expert is closer to the current proposal?

It does **not** directly ask:

> Which physical hypothesis explains the observed RGB/depth images better?

Also, the assignment probability is detached inside the transport rule, so rendering loss trains motion but does not directly provide the clean static/dynamic evidence needed for initialization.

The new method must therefore introduce an explicit render-driven discovery step before relying on the rigid expert.

---

# 3. Core principle: diffuse motion, not Gaussian geometry

Never noise the canonical Gaussian center itself as a new trainable scene parameter.

For Gaussian `i`, keep canonical state:

\[
\mu_i^0,\quad \Sigma_i^0,\quad q_i^0,\quad \alpha_i,\quad SH_i.
\]

For a motion-grid cell `c`, introduce only a temporary probe displacement:

\[
\delta_c \in \mathbb{R}^3.
\]

For every Gaussian in that cell:

\[
\mu_i^{probe} = \mu_i^0 + \delta_{c(i)}.
\]

The canonical 3DGS must remain frozen.

Noise only the temporary displacement:

\[
\delta_c^{(0,r)} = \sigma_0 \epsilon_c^r,
\qquad
\epsilon_c^r \sim \mathcal{N}(0,I),
\]

where `r` is the random probe seed.

Then optimize/denoise `delta` using real render loss.

This is diffusion-inspired inverse rendering / annealed Langevin optimization. It is intentionally **not** a conventional DDPM and must not introduce a separate denoiser network in the first implementation.

---

# 4. Use cells first, not independent per-Gaussian motion

Do not give every Gaussian an independent temporary displacement during the probe.

Use the existing persistent motion-grid cell mapping:

```python
gaussians.motion_grid.gaussian_cell_id
```

For all foreground Gaussians in cell `c`:

\[
\delta_i = \delta_c.
\]

Reasons:

- reduces the optimization dimension drastically;
- prevents individual splats from cheating photometric loss;
- makes random perturbation spatially meaningful;
- naturally integrates with the existing adaptive-grid split/merge logic;
- produces a clean per-cell motion evidence field that can later be lifted back to Gaussians.

Start from the current coarse grid. Do **not** run aggressive split/merge while the initial probe is still determining where motion exists. Refinement may be enabled after the first evidence pass.

---

# 5. Stage A — render-guided displacement noising/denoising

## 5.1 Probe frames

Do not initially process all 60 frames at full cost.

Select temporally separated probe frames that expose significant object motion, for example for a 60-frame sequence:

```text
[10, 25, 40, 59]
```

Make this configurable:

```text
--render_denoise_probe_frames 10,25,40,59
```

If not specified, derive evenly spaced nonzero frames from the dynamic sequence.

Frame 0 remains the canonical reference and must not be used as dynamic evidence against itself.

---

## 5.2 Camera selection

The evidence is fundamentally multiview.

For each probe frame `t`, use multiple distinct cameras at the **same physical timestamp**.

Default smoke-test target:

```text
4 cameras per probe frame
```

Full run may use all available cameras if runtime allows.

Never infer a dynamic label from one camera only unless a debug flag explicitly enables it.

Add:

```text
--render_denoise_cameras_per_probe 4
--render_denoise_min_support_cameras 2
```

Prefer camera subsets distributed around the object rather than adjacent camera indices if geometry is available.

---

## 5.3 Temporary probe transform

Create a small helper/module, recommended:

```text
scene/render_guided_motion_probe.py
```

Core state:

```python
class RenderGuidedMotionProbe:
    cell_delta              # [C,3], temporary differentiable parameter
    initial_noise           # [R,C,3] or regenerated deterministically
    per_seed_results
    per_frame_results
    per_camera_support
```

Do not register `cell_delta` inside the main long-lived Gaussian optimizer.

Use a short local optimizer, preferably Adam or AdamW:

```python
probe_optimizer = torch.optim.Adam([cell_delta], lr=...)
```

Canonical Gaussian parameters, motion MLP, rigid twist, joint part field, exposure, SH, opacity, scaling, and rotation are all frozen during this probe.

---

## 5.4 Applying the temporary displacement

Add a temporary rendering path that does not mutate canonical `_xyz`.

Preferred API:

```python
def build_probe_xyz(gaussians, cell_delta, foreground_only=True):
    xyz = gaussians.get_canonical_xyz.detach()
    probe_xyz = xyz.clone()
    probe_xyz[fg] = xyz[fg] + cell_delta[cell_id[fg]]
    return probe_xyz
```

Then render with an `override_xyz`-style path.

If the current renderer does not support overriding centers, add a clean optional argument rather than modifying `_xyz.data`.

Preferred:

```python
render(..., override_xyz=probe_xyz)
```

Do not permanently write the probe centers into the Gaussian model.

---

## 5.5 Probe loss

For one probe seed and frame:

\[
L_{probe}
=
\lambda_{rgb} L_{RGB}
+
\lambda_{ssim} L_{SSIM}
+
\lambda_d L_{depth}
+
\lambda_0 L_{zero}
+
\lambda_n L_{neighbor}.
\]

### RGB

Reuse the existing base 3DGS photometric loss.

Do not create a new image-space network.

### Depth

If reliable depth exists for the sampled camera, reuse the current depth term.

Depth is especially valuable because a wrong 3D displacement may improve RGB by moving a splat along an epipolar ambiguity but should not improve correct depth.

### Zero-motion prior

\[
L_{zero}
=
\frac{1}{C}
\sum_c \rho(\|\delta_c\|).
\]

Use a robust penalty, e.g. pseudo-Huber, not a hard threshold.

This encodes:

> motion must earn its existence by improving the real render.

### Weak neighboring-cell regularization

Use only a weak prior during the probe:

\[
L_{neighbor}
=
\sum_{(c,j)} w_{cj}\rho(\delta_c-\delta_j).
\]

This should suppress isolated one-cell photometric cheating but must not blur actual static/dynamic boundaries.

No DINO weighting.

---

# 6. Denoising dynamics

## 6.1 Basic deterministic version

For each seed:

```text
noise delta
  -> render all selected cameras
  -> backprop
  -> update delta
  -> smaller step/noise
  -> repeat
```

A simple optimizer is sufficient for v1.

Recommended smoke defaults:

```text
3 random seeds
6-8 denoising steps per seed
4 probe frames
4 cameras per frame
```

This is deliberately small.

---

## 6.2 Optional annealed stochastic update

A closer diffusion-inspired update may be enabled later:

\[
\delta^{k+1}
=
\delta^k
-
\eta_k \nabla_{\delta}L
+
\sigma_k \xi_k,
\]

with

\[
\sigma_k \downarrow 0.
\]

Do not make this mandatory for the first implementation. The core scientific test is whether **independent noisy initializations converge to the same render-supported displacement**.

---

# 7. Occlusion handling — mandatory tri-state evidence

This is the most important correctness requirement.

The probe must never force every Gaussian/cell into a binary label when there is insufficient visual evidence.

Use three states:

```text
STATIC
DYNAMIC
UNKNOWN / OCCLUDED / AMBIGUOUS
```

The final trainable part probability may remain binary later, but the discovery stage itself must retain a separate evidence/confidence mask.

## 7.1 Why `UNKNOWN` is necessary

A Gaussian hidden behind another surface produces little trustworthy image evidence.

If it is forced to binary:

- low gradient can become a **false static**;
- an occlusion-boundary residual can become a **false dynamic**.

Instead:

```text
insufficient visible evidence -> UNKNOWN
```

and do not update static/dynamic supervision from that observation.

---

## 7.2 Visibility/support measure

The current `visibility_filter` / positive radius indicates projection/rasterization participation, but this alone is not necessarily sufficient to distinguish heavy occlusion from strong image contribution.

Implement a per-cell/per-Gaussian **render support score**.

Preferred order:

1. Reuse a renderer-provided per-Gaussian contribution/responsibility statistic if already available.
2. Otherwise use accumulated gradient magnitude from the actual render loss as an observation-support proxy.
3. If necessary, extend the rasterizer to accumulate a detached per-Gaussian alpha-compositing contribution:

\[
S_i^{view}
=
\sum_p T_{i,p}\alpha_{i,p}
\]

or an equivalent normalized contribution.

Aggregate to cells.

Important:

- frustum-visible but fully occluded is **not** strong evidence;
- a cell must have enough actual render support before it may receive static evidence;
- dynamic evidence additionally requires coherent motion improvement.

Add a clean output:

```python
support_per_cell[frame, camera, cell]
```

---

# 8. Per-seed motion evidence

For each cell `c`, frame `t`, and seed `r`, store:

```text
initial_delta
final_delta
final_delta_norm
loss_start
loss_end
gradient_energy
integrated_gradient_work
support_cameras
```

## 8.1 Endpoint displacement

\[
m_{c,t,r}=\|\delta_{c,t,r}^{*}\|.
\]

A large displacement alone is **not enough** to declare motion.

---

## 8.2 Seed consistency

Let `R` be the number of noise seeds.

Compute mean displacement:

\[
\bar\delta_{c,t}
=
\frac{1}{R}\sum_r \delta_{c,t,r}^{*}.
\]

Directional/resultant consistency:

\[
C^{seed}_{c,t}
=
\frac{
\left\|\sum_r w_r\delta_{c,t,r}^{*}\right\|
}{
\sum_r w_r\|\delta_{c,t,r}^{*}\|+\epsilon
}.
\]

Interpretation:

```text
~1.0  -> seeds agree on a motion direction
~0.0  -> different seeds found incompatible explanations
```

False dynamic evidence from texture ambiguity or occlusion boundaries should have low seed consistency.

---

## 8.3 Render-supported gradient work

Global render loss cannot be naively assigned to one cell without one render per cell.

Do **not** perform a separate ablation render for every Gaussian/cell.

Instead accumulate first-order optimization work:

\[
W_{c,t,r}
=
\sum_k
-
\nabla_{\delta_c}L_k^T
(\delta_c^{k+1}-\delta_c^k).
\]

Positive large `W` means the renderer repeatedly supported moving this cell along the optimizer trajectory.

Also accumulate:

\[
G_{c,t,r}
=
\sum_k\|\nabla_{\delta_c}L_k\|.
\]

Use robust normalization across cells.

This gives a per-cell render-evidence signal without `C` additional renders.

---

# 9. Multiview consistency

For a true 3D motion, distinct visible cameras should generate compatible 3D correction directions after backpropagation through projection.

For every probe frame, compute at least one per-camera gradient observation near the start or final denoised state:

\[
g_{c,t,v}
=
\nabla_{\delta_c}L_{t,v}.
\]

For cameras whose support exceeds the visibility threshold, calculate agreement, for example:

\[
C^{view}_{c,t}
=
\frac{
\left\|\sum_v \hat g_{c,t,v}\right\|
}{N_{support}+\epsilon}
\]

or pairwise cosine agreement.

Use the sign consistently: the proposed correction is `-gradient`.

Requirements:

- fewer than `min_support_cameras` -> observation is `UNKNOWN`;
- strongly contradictory camera gradients -> `UNKNOWN/AMBIGUOUS`;
- one strong camera plus several unsupported/occluded cameras must not be treated as multiview-confirmed dynamic.

---

# 10. Stage-A classification: moving seeds, static candidates, unknown

Do **not** immediately force a final static/dynamic label after displacement denoising.

Compute robust normalized scores:

```text
M = nonzero denoised displacement magnitude
C_seed = agreement across random seeds
C_view = agreement across supported cameras
W = positive render-supported gradient work
V = actual visibility/support
```

Suggested conceptual dynamic evidence:

\[
D_{c,t}
=
V_{c,t}
\cdot C^{seed}_{c,t}
\cdot C^{view}_{c,t}
\cdot f(M_{c,t})
\cdot f(W_{c,t}).
\]

Suggested static evidence:

\[
S_{c,t}
=
V_{c,t}
\cdot f_{zero}(M_{c,t})
\cdot f_{lowgain}(W_{c,t}).
\]

But static evidence must be more conservative than dynamic evidence.

### High-confidence moving seed

Require all of:

```text
sufficient multiview support
nontrivial final displacement
positive render-supported work
high seed consistency
high view consistency
same behavior in >= N probe frames
```

### Static candidate

Require:

```text
strong visibility/support
multiple probe frames
repeated return close to zero motion
moving the cell does not yield sustained positive render gain
no strong dynamic evidence in another probe
```

Do not yet call hinge-near low-motion cells definitively static.

### Unknown

Any of:

```text
occluded / insufficient support
single-view-only evidence
seed disagreement
view disagreement
large gradient but inconsistent displacement
weak motion and weak static evidence
transient evidence in only one frame
```

Store all three states explicitly.

---

# 11. Temporal evidence accumulation

Occlusion is temporary, so labels must not be decided independently per frame.

For each cell, accumulate only **informative** observations:

```python
if support(frame, cell) < support_threshold:
    # no positive or negative label evidence from this frame
    continue
```

Then aggregate dynamic/static evidence across separated probe times.

Example:

```text
frame 10: visible -> strong dynamic
frame 25: occluded -> unknown, contributes nothing
frame 40: visible -> strong dynamic
frame 59: visible -> dynamic

FINAL: strong dynamic
```

Static example:

```text
frame 10: visible -> static candidate
frame 25: visible -> static candidate
frame 40: occluded -> unknown
frame 59: visible -> static candidate

FINAL: strong static candidate
```

Never convert missing evidence into static evidence.

---

# 12. Fit the existing rigid twist from high-confidence moving seeds

After Stage A, build a seed mask from only high-confidence moving cells/Gaussians.

Map cell evidence to Gaussians:

```python
moving_seed_gaussian = moving_seed_cell[cell_id]
```

Use these points to bootstrap the existing explicit rigid model:

\[
v(x,t)=\omega(t)\times(x-c)+u(t).
\]

Preserve the current rigid machinery and its gradient routing.

Do not fit the first rigid twist from arbitrary k-medoids moving clusters.

Initial twist weights should be based on render motion evidence:

\[
w_i
=
D_i\cdot V_i\cdot C_i.
\]

No DINO term.

If fewer than the required number of reliable moving seeds exist, fail the rigid bootstrap gracefully and report insufficient motion evidence rather than fabricating a moving cluster.

---

# 13. Stage B — noisy static-vs-rigid gate denoising

Stage A deliberately finds only reliable moving seeds. It will miss some moving Gaussians, especially:

- near a revolute hinge;
- points with small translational displacement;
- points temporarily occluded in the probe frames.

After the rigid motion is known, use the rigid model to classify the whole link.

This stage is essential.

---

## 13.1 Per-cell latent gate

For each foreground cell define a temporary gate:

\[
a_c \in [0,1],
\]

where:

```text
a_c = 0 -> static hypothesis
a_c = 1 -> rigid-moving hypothesis
```

Parameterize with a logit:

\[
a_c=\sigma(\ell_c/T_g).
\]

Do not initialize all cells hard static or hard moving.

Use Stage-A evidence:

```text
strong static candidate -> low but not saturated logit
moving seed -> high but not saturated logit
unknown -> logit near 0 => a ~ 0.5
```

---

## 13.2 Static and rigid hypotheses

For probe time `t`:

Static center:

\[
x_i^S(t)=x_i^0.
\]

Rigid center from the existing twist/integrator:

\[
x_i^R(t)=T_{rigid}(t)x_i^0.
\]

Interpolate using the cell gate:

\[
x_i(t)
=
(1-a_{c(i)})x_i^S(t)
+
a_{c(i)}x_i^R(t).
\]

For a revolute/screw rigid hypothesis, also interpolate orientation continuously.

Do **not** ignore Gaussian orientation if the current rigid pipeline already rotates covariance/orientation.

Preferred orientation interpolation:

\[
q_i(t,a)=
\exp\left(a\,\log(q_i^R(t)(q_i^0)^{-1})\right)q_i^0.
\]

Equivalent normalized quaternion slerp is acceptable.

This is important because hinge-axis points may have nearly zero center displacement while their local orientation still belongs to the rotating rigid body.

---

## 13.3 Noise and denoise the gate

Run multiple initial noisy gate logits:

\[
\ell_c^{(0,r)}=\ell_c^{prior}+\sigma_g\epsilon_c^r.
\]

For each seed, optimize only the gate logits by multiview RGB/depth rendering.

Optional weak priors:

- weak binary entropy near the end, not at the start;
- weak spatial consistency between neighboring cells;
- prior toward Stage-A evidence only where that evidence is high confidence.

Critical behavior:

```text
occluded cell -> little gate gradient -> remains uncertain
visible static cell -> render pulls gate toward 0
visible moving cell -> render pulls gate toward 1
ambiguous boundary -> seeds disagree -> retain uncertainty
```

This is the direct render-space replacement for the current velocity-only identity-vs-rigid label target.

---

# 14. Final static/dynamic/unknown probability

For each cell aggregate gate endpoints across seeds, cameras, and probe frames.

Store:

```text
p_dynamic_mean
p_dynamic_std
seed_consistency
view_consistency
support_count
support_fraction
occluded_fraction
render_motion_score
render_static_score
confidence
state = static | dynamic | unknown
```

A Gaussian inherits its cell probability initially.

Later refinement may allow neighboring child cells to separate.

Final label rules must include a confidence gate:

```python
if support < min_support:
    state = UNKNOWN
elif p_dynamic >= dyn_threshold and confidence >= conf_threshold:
    state = DYNAMIC
elif p_dynamic <= static_threshold and confidence >= conf_threshold:
    state = STATIC
else:
    state = UNKNOWN
```

Recommended initial thresholds:

```text
static_threshold = 0.2
dynamic_threshold = 0.8
```

but confidence/support thresholds matter more than the exact probability thresholds.

---

# 15. What to do with UNKNOWN points during real training

Unknown is not a third physical part. It is a state of insufficient discovery evidence.

During the first joint optimization stage:

- strong static -> zero-motion expert may be applied;
- strong dynamic -> rigid expert may be applied;
- unknown -> keep the free proposal available and do **not** apply strong static suppression.

This matches the existing useful idea that uncertain points should remain free to receive rendering evidence.

However, the unknown mask must be explicit and should prevent the system from converting occlusion-induced lack of gradient into confident static.

Once later frames provide evidence or the static-vs-rigid gate becomes confident, unknown points may transition to static/dynamic.

---

# 16. Replace current DINO/k-medoids initialization

For this new branch:

### Remove from static/dynamic discovery

```text
XYZ + DINO pairwise distance
K+1 k-medoids
largest cluster -> static
DINO boundary loss for dynamic assignment
DINO-based static/moving propagation
```

### Keep only if needed elsewhere

DINO code may remain in the repository for unrelated experiments, but the new render-guided branch must run with it disabled.

Add a new mode, recommended:

```text
--dynamic_discovery_mode render_denoise
```

Possible values:

```text
legacy
render_denoise
```

For the new mode, no DINO file should be required.

---

# 17. Keep semantic foreground gating

Semantic foreground is not the static/dynamic signal.

Use it only to define which Gaussians belong to the articulated object and are eligible for static/moving classification and rigid fitting.

The render-guided probe should operate on this semantic foreground.

Background Gaussians should not enter rigid motion estimation.

Do not use semantics to decide which foreground portion moves.

---

# 18. Integration into current `GaussianModel`

Recommended additions to `scene/gaussian_model.py`:

```python
def get_probe_foreground_state(...)

def build_probe_xyz_from_cell_delta(cell_delta, foreground_mask=None)

def build_static_rigid_gated_state(gate, t, ...)

def initialize_joint_part_field_from_render_evidence(evidence, args)

def set_render_discovery_state(...)

def clear_render_discovery_state(...)
```

Do not overload the normal `_current_xyz` state in ways that can leak probe motion into real training.

Prefer an explicit renderer override or a temporary context manager.

Example:

```python
with gaussians.render_probe_state(xyz=probe_xyz, rotation=probe_rot):
    render(...)
```

or pass overrides directly to the renderer.

---

# 19. New module structure

Recommended new file:

```text
scene/render_guided_motion_probe.py
```

Suggested API:

```python
@dataclass
class RenderDenoiseConfig:
    probe_frames: list[int]
    cameras_per_probe: int
    num_seeds: int
    denoise_steps: int
    noise_sigma: float
    lr: float
    lambda_zero: float
    lambda_neighbor: float
    min_support_cameras: int
    min_informative_frames: int
    seed_consistency_threshold: float
    view_consistency_threshold: float


@dataclass
class RenderMotionEvidence:
    per_cell_dynamic_score: torch.Tensor
    per_cell_static_score: torch.Tensor
    per_cell_confidence: torch.Tensor
    per_cell_state: torch.Tensor
    per_cell_mean_delta: torch.Tensor
    per_cell_delta_std: torch.Tensor
    per_cell_seed_consistency: torch.Tensor
    per_cell_view_consistency: torch.Tensor
    per_cell_support_count: torch.Tensor
    per_cell_occluded_fraction: torch.Tensor
    per_frame: dict


def run_render_guided_displacement_probe(...):
    ...


def bootstrap_rigid_twist_from_render_seeds(...):
    ...


def run_static_rigid_gate_probe(...):
    ...


def combine_render_discovery_evidence(...):
    ...
```

Keep all tensors GPU-resident during one probe where practical, but release probe graphs immediately after each optimizer step.

---

# 20. Changes to `train.py`

In `initialize_dynamic_stage_once()` change the order for `render_denoise` mode.

Current conceptual order:

```text
init dynamic motion
init part field from XYZ/DINO
expand dynamic sequence
freeze canonical
joint training
```

New order:

```text
init motion grid / proposal infrastructure
expand dynamic sequence
freeze canonical geometry + appearance
build semantic foreground mask
run Stage-A render displacement denoising
fit/bootstrap rigid twist from moving seeds
run Stage-B static-vs-rigid gate denoising
initialize JointPartField from render-derived probabilities
save initialization diagnostics
enter normal joint training
```

Important:

- the new probe must happen only once unless explicitly requested again;
- do not accidentally count probe optimizer steps as normal 3DGS global iterations;
- probe camera images may remain CPU-backed and be transferred one camera at a time as the normal dynamic sequence already does;
- clear CUDA graphs/tensors between probe seeds to avoid memory accumulation.

---

# 21. Initialize `JointPartField` from probabilities, not k-medoids

Add a constructor/path accepting soft render-derived priors.

Recommended:

```python
JointPartField.from_dynamic_probability_prior(
    gaussian_ids=fg_ids,
    p_dynamic=p_render,
    confidence=confidence,
    unknown_mask=unknown,
    ...
)
```

Do not initialize unknown points as hard static.

For two prototypes, construct point-field vectors so the softmax approximately matches the desired prior while remaining trainable.

Example conceptual mapping:

```text
p <= 0.1 and confident -> close to static prototype
p >= 0.9 and confident -> close to moving prototype
unknown -> approximately equidistant from both prototypes
```

Store the render-derived prior separately from the trainable field so debugging can distinguish:

```text
initial render evidence
current trainable assignment
```

---

# 22. Modify joint label supervision

The existing velocity-space label target may remain as a secondary physical regularizer after rigid fitting.

But add a primary render-derived target:

\[
L_{render-label}
=
\frac{
\sum_i w_i\,BCE(p_i,q_i^{render})
}{
\sum_i w_i+\epsilon
},
\]

where

\[
w_i
=
confidence_i\times support_i.
\]

For unknown/occluded points:

\[
w_i=0.
\]

Therefore they receive no false static or false dynamic target.

Use the current velocity expert target only as a secondary term:

```text
L_label_total = lambda_render_label * L_render_label
              + lambda_velocity_label * L_velocity_label
```

Recommended initial relation:

```text
lambda_render_label >> lambda_velocity_label
```

The render-derived evidence is the observation likelihood; velocity consistency is the physical regularizer.

---

# 23. Adaptive grid integration

The existing adaptive motion grid already supports cell-level split/merge and render-gradient ambiguity scores.

Use the new evidence to improve topology.

Add split score terms:

```text
render_denoise_seed_variance
render_denoise_view_disagreement
render_denoise_gate_entropy
mixed static/dynamic evidence inside cell
```

A cell should split when:

- it contains both strong static and strong dynamic Gaussian evidence;
- different views support incompatible denoised motion;
- gate optimization is consistently bimodal/ambiguous;
- neighboring child hypotheses would clearly reduce render loss.

Do not use DINO variance in `render_denoise` mode.

Merge sibling cells when:

- both have same confident state;
- denoised displacement/twist evidence is consistent;
- gate probabilities and motion directions agree;
- render evidence does not improve by keeping them separate.

---

# 24. Prevent false dynamic from occlusion boundaries

Explicit safeguards:

1. **Minimum real support:** no dynamic evidence without enough render contribution.
2. **Multiview requirement:** one camera cannot create a confident dynamic seed.
3. **Seed consistency:** independent noisy starts must converge similarly.
4. **Temporal persistence:** require evidence in multiple informative probe frames.
5. **Depth consistency:** when reliable depth is present, large RGB-only motion that worsens depth is rejected/downweighted.
6. **Magnitude cap:** constrain temporary probe displacement to a physically plausible range.
7. **Robust loss:** use pseudo-Huber/Charbonnier for RGB/depth residual sensitivity if needed.
8. **No forced binary assignment:** inconsistent evidence becomes `UNKNOWN`.

---

# 25. Prevent false static from occlusion

Explicit safeguards:

1. A cell may receive static evidence only while sufficiently visible.
2. A frame with low support contributes **zero** static evidence.
3. Missing gradient must never be interpreted as proof of zero motion.
4. Static requires repeated visible evidence across multiple timestamps.
5. After rigid fitting, low-motion/hinge-near unknown cells must be tested under the rigid hypothesis before becoming static.
6. A cell that is dynamic in one strong visible interval and occluded elsewhere remains a dynamic candidate; occluded intervals cannot vote it back to static.

---

# 26. Hinge-near moving points

Do not classify motion only from displacement magnitude.

For revolute motion:

\[
\|v_i\| = \|\omega\times(x_i-c)\|
\]

approaches zero near the axis even if the point belongs to the moving rigid body.

Therefore:

- Stage A displacement denoising finds reliable moving seeds away from the hinge;
- the existing rigid twist estimates the body motion;
- Stage B static-vs-rigid gate expands membership using the full rigid hypothesis;
- if available, rigid orientation change should contribute to the render hypothesis as well as center motion.

A hinge-near point must not be finalized as static merely because its Stage-A displacement magnitude is low.

---

# 27. Proposed command-line arguments

Add a grouped set of arguments in `arguments/__init__.py`.

```text
--dynamic_discovery_mode {legacy,render_denoise}

--render_denoise_probe_frames "10,25,40,59"
--render_denoise_num_probe_frames 4
--render_denoise_cameras_per_probe 4
--render_denoise_min_support_cameras 2
--render_denoise_min_informative_frames 2

--render_denoise_num_seeds 3
--render_denoise_steps 8
--render_denoise_noise_sigma 0.01
--render_denoise_lr 1e-2
--render_denoise_max_displacement 0.10

--lambda_render_denoise_rgb 1.0
--lambda_render_denoise_ssim 0.2
--lambda_render_denoise_depth 0.1
--lambda_render_denoise_zero 0.01
--lambda_render_denoise_neighbor 0.001

--render_denoise_seed_consistency_threshold 0.65
--render_denoise_view_consistency_threshold 0.55
--render_denoise_dynamic_threshold 0.75
--render_denoise_static_threshold 0.25
--render_denoise_confidence_threshold 0.60

--render_gate_num_seeds 3
--render_gate_steps 8
--render_gate_lr 1e-2
--render_gate_temperature 1.0

--lambda_render_label 1.0
--lambda_velocity_label 0.1
```

The numerical defaults above are starting points, not sacred values. Keep them centralized and logged.

Do not add a hard wall-clock cutoff.

---

# 28. Runtime target

The complete reconstruction pipeline should remain practical.

The discovery stage should be designed around a small fixed render/backward budget, not thousands of neural-network iterations.

Smoke configuration:

```text
4 probe frames
4 cameras/probe
3 displacement seeds
6 denoise steps
3 gate seeds
6 gate steps
```

Use this to measure actual RTX 4070 runtime.

Only increase the budget when diagnostics show the small probe is insufficient.

Avoid any `O(number_of_gaussians * number_of_renders)` loop.

All cells must be optimized simultaneously in each render/backward.

---

# 29. Required diagnostics

Create:

```text
<model_path>/render_denoise_discovery/
```

Save:

```text
config.json
probe_frames.json
summary.json
per_cell_evidence.pt
per_gaussian_evidence.pt
per_frame_evidence.pt
```

Per cell save at least:

```text
cell_id
count
mean_delta_xyz
mean_delta_norm
delta_std
seed_consistency
view_consistency
gradient_energy
integrated_gradient_work
support_count
support_fraction
occluded_fraction
stageA_dynamic_score
stageA_static_score
stageA_state
rigid_gate_mean
rigid_gate_std
final_dynamic_probability
final_confidence
final_state
```

Per Gaussian save mapped values plus canonical XYZ.

---

# 30. Required visual debugging

Generate both 3D and image-space diagnostics.

## 30.1 3D HTML

Show canonical Gaussians/cell centers colored as:

```text
blue/neutral  = confident static
red/warm      = confident dynamic
gray          = unknown / occluded / ambiguous
```

Also draw:

- mean denoised displacement vector;
- vector opacity/strength proportional to confidence;
- optional rigid predicted velocity vector after Stage B;
- camera frustums fixed in world coordinates.

Generate:

```text
render_denoise_stageA_cells.html
render_denoise_stageB_static_rigid.html
render_denoise_final_gaussians.html
```

## 30.2 Per-camera image diagnostics

For each probe frame/camera, save a panel containing:

```text
GT RGB
canonical/static render
final displacement-probe render
absolute residual before
absolute residual after
support mask
projected dynamic evidence
projected unknown/occlusion evidence
```

After rigid fitting also save:

```text
static-hypothesis render
rigid-hypothesis/gated render
final gated render
```

This is required to verify that the method is not merely moving cells around an occlusion contour.

---

# 31. W&B logging

Log compact scalar summaries:

```text
discovery/render_denoise/num_probe_frames
discovery/render_denoise/num_cells
discovery/render_denoise/dynamic_cells
discovery/render_denoise/static_cells
discovery/render_denoise/unknown_cells
discovery/render_denoise/mean_seed_consistency
discovery/render_denoise/mean_view_consistency
discovery/render_denoise/mean_support_fraction
discovery/render_denoise/mean_occluded_fraction
discovery/render_denoise/mean_displacement
discovery/render_denoise/mean_positive_work
discovery/render_gate/mean_probability
discovery/render_gate/mean_confidence
discovery/render_gate/unknown_fraction
```

Also log per-probe loss before/after denoising.

---

# 32. Unit tests — mandatory before a full scene run

Add:

```text
tests/test_render_guided_motion_probe.py
```

and, where CUDA rasterization is needed:

```text
tests/test_render_guided_motion_probe_cuda.py
```

## Test 1 — all-static scene

Create a synthetic static object rendered from multiple cameras at two timestamps.

Expected:

```text
mean denoised displacement ~ 0
no confident moving seeds
visible cells -> static candidate
```

---

## Test 2 — static object temporarily occluded

Keep target Gaussians static but introduce an occluder in one/subset of cameras.

Expected:

```text
occluded cells -> UNKNOWN for occluded views
visible cameras still provide static evidence
no false dynamic caused by occlusion contour
```

Crucially, a fully occluded cell must **not** become confident static only because gradient is zero.

---

## Test 3 — translating rigid part

Translate one connected subset.

Expected:

```text
Stage A finds high-confidence moving seeds
seed directions agree with GT translation
static region remains static/unknown
rigid bootstrap recovers translation
Stage B expands moving membership
```

---

## Test 4 — rotating rigid part

Rotate one subset about a known axis.

Expected:

```text
moving seeds away from hinge are found
rigid omega direction matches GT
hinge-near low-displacement points are not permanently labeled static
Stage B assigns them to moving body when render evidence supports rigid hypothesis
```

---

## Test 5 — moving part with self-occlusion

The moving link becomes hidden in some views/times and reappears later.

Expected:

```text
occluded observations do not vote static
visible observations recover same motion
final body membership survives occlusion interval
```

---

## Test 6 — inconsistent image residual

Introduce a local lighting/color perturbation without geometry motion.

Expected:

```text
RGB residual may be large
3D denoising directions disagree across seeds/views or depth does not support them
cell remains static candidate / unknown, not confident dynamic
```

---

## Test 7 — single-camera false solution

Construct an ambiguity where one camera can reduce RGB loss with an incorrect 3D displacement but other cameras reject it.

Expected:

```text
single camera proposes motion
view consistency low
no confident dynamic seed
```

---

# 33. Dataset-level validation on `USB_100109`

For the first real smoke test:

1. use the existing trained frame-0 PLY/canonical checkpoint;
2. semantic foreground gating ON;
3. DINO OFF for dynamic/static discovery;
4. 4 probe frames;
5. 4 cameras per frame;
6. 3 random displacement seeds;
7. 6-8 denoising steps;
8. save all diagnostics before launching long joint optimization.

First evaluate **only discovery**.

Do not hide discovery quality behind later rigid fitting.

Required questions:

```text
Are high-confidence moving seeds spatially on the real moving link?
Are static object regions actually static?
Do occluded areas become gray/unknown rather than false static/dynamic?
Do repeated noise seeds converge to the same moving displacement?
Do different cameras agree?
Does Stage B correctly add hinge-near points to the rigid link?
```

Only after these are visually correct should normal joint training continue.

---

# 34. Acceptance criteria

The implementation is not complete merely because losses decrease.

For the synthetic tests with GT labels:

### Occlusion robustness

- fully occluded cells must not receive confident static/dynamic evidence from the occluded frame;
- temporary occlusion must not flip a correct dynamic cell to static;
- static occlusion boundaries must not form persistent moving clusters.

### Motion recovery

- true visible rigid motion must produce reproducible nonzero denoised displacement;
- independent noise seeds must converge to compatible directions;
- multiview gradients must be compatible for confident dynamic cells.

### Hinge correctness

- hinge-near moving points must not be declared static only because displacement is small;
- after rigid fit, Stage B should assign them according to static-vs-rigid render evidence.

### Static correctness

- visible true static cells should repeatedly denoise toward zero/no-motion;
- large photometric residual alone is insufficient for a dynamic label.

### Pipeline integrity

- canonical Gaussian parameters remain bitwise/near-bitwise unchanged by discovery;
- normal rigid twist path still works;
- no DINO input is required in `render_denoise` mode;
- probe temporary state is fully cleared before normal training;
- checkpoints include render-derived discovery evidence and can resume normal joint training.

---

# 35. Do not do these things

Do **not**:

- train a separate video diffusion model;
- use DINO to decide static vs dynamic;
- classify high RGB residual directly as dynamic;
- classify zero gradient directly as static;
- use only one camera for confident dynamic discovery;
- optimize every Gaussian displacement independently;
- permanently modify canonical Gaussian centers during probing;
- render once per Gaussian/cell to estimate counterfactual loss;
- force all cells into static/dynamic before visibility confidence exists;
- initialize the moving link as simply the smaller/larger geometry cluster;
- suppress uncertain cells with the static zero-velocity prior;
- replace the existing rigid twist solver that is already working well.

---

# 36. Recommended implementation sequence

Implement incrementally.

## Step 1

Add the temporary per-cell displacement renderer override and verify canonical state is unchanged after a forward/backward.

## Step 2

Implement one frame + multiple cameras + one noise seed + 8 denoising steps.

Visualize displacement vectors.

## Step 3

Add multiple seeds and calculate seed consistency.

## Step 4

Add per-camera support and view-consistency calculations.

At this point implement the tri-state `static candidate / moving seed / unknown` output.

## Step 5

Aggregate evidence over multiple probe frames while ignoring unsupported/occluded observations.

## Step 6

Bootstrap the existing rigid twist from high-confidence moving seeds only.

## Step 7

Implement the noisy static-vs-rigid gate optimization, including rigid orientation interpolation when applicable.

## Step 8

Initialize `JointPartField` from render-derived probabilities/confidences.

## Step 9

Add `L_render_label` and keep the current velocity-space target as a weaker secondary regularizer.

## Step 10

Connect the new uncertainty/evidence scores to adaptive-grid split/merge.

## Step 11

Run all synthetic occlusion tests.

## Step 12

Run `USB_100109` discovery-only smoke test and inspect diagnostics before long training.

---

# 37. Final intended formulation

The initial motion discovery should optimize:

\[
\boxed{
\delta^{*}
=
\arg\min_{\delta}
\left[
L_{render}(R(\mathcal G_0,\delta),I_t)
+
\lambda_0\|\delta\|_{robust}
+
\lambda_n L_{neighbor}
\right]
}
\]

from multiple random noisy initializations.

But a cell is not dynamic just because `delta* != 0` once.

It is a moving seed only when:

\[
\boxed{
\text{visible}
\land
\text{render-supported}
\land
\text{seed-consistent}
\land
\text{multiview-consistent}
\land
\text{temporally persistent}
}
\]

Occlusion produces:

\[
\boxed{
\text{insufficient evidence} \Rightarrow \text{UNKNOWN},
}
\]

not static and not dynamic.

Then, after fitting the already-working rigid model, solve a second render-guided latent assignment:

\[
\boxed{
q_c^{render}
\approx
\arg\min_{a_c\in[0,1]}
L_{render}
\left(
R((1-a_c)\mathcal G^{static}+a_c\mathcal G^{rigid}),
I_t
\right)
}
\]

using multiple noisy gate initializations and the same visibility/consistency logic.

This converts the static/dynamic question from:

> Is the current proposal velocity closer to zero or to the rigid expert?

into the physically relevant question:

> Does the real multiview observation prefer this material to remain canonical, or to follow the recovered rigid body motion?

That render-driven counterfactual is the primary static/dynamic evidence. The rigid velocity field remains the physical regularizer and motion model.

