# Codex Implementation Spec — One-Step Self-Supervised Gaussian Motion Denoising
## Frame 0 → Frame 1 only, multiview RGB-D supervision, per-Gaussian dynamic probability

## 0. Scope — implement ONLY this experiment

Implement a standalone, testable one-step motion-discovery experiment in the existing `RigEuler3DGS` repository.

The experiment must use:

- the trained static 3DGS as the state at **frame 0**;
- synchronized multiview RGB images at **frame 1** as ground truth;
- depth at frame 1 when available;
- the existing differentiable CUDA Gaussian rasterizer;
- the already-added CUDA reverse-responsibility statistics:
  - `raw_support`
  - `visible_mass`
  - `valid_visible_mass`
- a small learned denoising network operating **per Gaussian with local spatial message passing**;
- repeated stochastic noising/denoising trials;
- self-supervision only from how well the denoised Gaussians render the frame-1 multiview observations.

The final output is:

\[
P_i^{dyn}\in[0,1]
\]

for every Gaussian \(i\), representing the probability that Gaussian \(i\) must move from frame 0 to frame 1.

Also output the estimated displacement and velocity:

\[
\hat{\mathbf d}_i^{0\rightarrow1}
\]

\[
\hat{\mathbf v}_i^{0\rightarrow1}
=
\frac{\hat{\mathbf d}_i^{0\rightarrow1}}{\Delta t_{01}}.
\]

### Explicitly OUT OF SCOPE

Do **not** implement any of the following in this task:

- frame 1 → frame 2 or longer sequences;
- recursive propagation across the video;
- rigid-motion fitting;
- SE(3) exponential-map integration;
- revolute/prismatic joint estimation;
- K-part segmentation;
- Eulerian grid optimization;
- DINO-based static/dynamic classification;
- optical-flow supervision;
- external learned motion models;
- pretrained diffusion models;
- Stable Diffusion / VAE / image diffusion;
- Stage 2 / Stage 3 joint discovery;
- permanent modification of canonical Gaussian geometry.

This is a deliberately isolated experiment to answer one question:

> Can a self-supervised learned denoiser infer, Gaussian-by-Gaussian, which frame-0 Gaussians must move so that the resulting 3DGS renders the real multiview frame-1 observations?

---

# 1. Existing repository assumptions

Repository root:

```text
~/Joint3DGS/RigEuler3DGS
```

Example input sequence:

```text
/home/samuelemara/Joint3DGS/RigEuler3DGS/input/USB_100109
```

Static trained model:

```text
/home/samuelemara/Joint3DGS/RigEuler3DGS/output/USB_100109
```

Static PLY:

```text
/home/samuelemara/Joint3DGS/RigEuler3DGS/output/USB_100109/point_cloud/iteration_30000/point_cloud.ply
```

The repository already has:

- `scene.Scene`
- `scene.gaussian_model.GaussianModel`
- `gaussian_renderer.render`
- the Gaussian rasterizer CUDA extension
- dynamic-sequence camera loading
- multiview RGB frame loading
- optional depth loading
- reverse accumulation of pixel features to Gaussians

The CUDA reverse accumulator has already been extended so that it returns:

```python
rendered
feature_sum
valid_weight
count
raw_support
visible_mass
radii
```

and the high-level wrapper exposes quantities equivalent to:

```python
raw_support
visible_mass
valid_visible_mass
visibility_ratio
valid_fraction
projected
alpha_visible
evidence_valid
```

Do not rewrite the rasterizer unless a real incompatibility is found.

---

# 2. Core idea

The network does not directly learn a motion label from RGB.

Instead:

1. start with the known static Gaussian positions at frame 0;
2. inject random 3D positional noise into the Gaussians;
3. render the noisy Gaussians into all selected frame-1 cameras;
4. compare those renders with the real frame-1 RGB-D observations;
5. reverse-project the multiview residual evidence back onto individual Gaussians using the CUDA responsibility accumulator;
6. feed the noisy Gaussian state plus per-Gaussian render evidence to a learned denoising network;
7. predict a per-Gaussian position correction;
8. render again;
9. train the denoiser so the corrected Gaussians match frame 1;
10. repeat this denoising operation for multiple iterative denoising steps;
11. repeat the entire experiment from multiple independent noise realizations;
12. infer a per-Gaussian dynamic probability from the noise-invariant recovered displacement.

The central desired identity is:

\[
\boxed{
\Delta\boldsymbol{\mu}_i^{net}
\approx
-\boldsymbol{\epsilon}_i
+
\mathbf d_i^{0\rightarrow1}
}
\]

where:

- \(\boldsymbol{\epsilon}_i\) is the artificial positional noise;
- \(-\boldsymbol{\epsilon}_i\) removes the artificial perturbation;
- \(\mathbf d_i^{0\rightarrow1}\) is the actual displacement required to explain frame 1.

Therefore the final corrected position is

\[
\hat{\boldsymbol{\mu}}_{i,1}
=
\boldsymbol{\mu}_{i,0}
+
\mathbf d_i^{0\rightarrow1}.
\]

Static Gaussian:

\[
\mathbf d_i^{0\rightarrow1}\approx0.
\]

Dynamic Gaussian:

\[
\mathbf d_i^{0\rightarrow1}\neq0.
\]

---

# 3. Important distinction: artificial noise vs physical displacement

Let the canonical frame-0 Gaussian position be:

\[
\boldsymbol{\mu}_{i,0}.
\]

Sample artificial noise:

\[
\boldsymbol{\epsilon}_i
\sim
\mathcal N(0,\sigma_x^2I).
\]

Noisy starting position:

\[
\tilde{\boldsymbol{\mu}}_i^{(0)}
=
\boldsymbol{\mu}_{i,0}
+
\boldsymbol{\epsilon}_i.
\]

The denoiser predicts a correction:

\[
\Delta\boldsymbol{\mu}_i^{(r)}
=
D_\theta(
 h_i^{(r)}
)
\]

at iterative denoising step \(r\).

Update:

\[
\tilde{\boldsymbol{\mu}}_i^{(r+1)}
=
\tilde{\boldsymbol{\mu}}_i^{(r)}
+
\Delta\boldsymbol{\mu}_i^{(r)}.
\]

After \(R\) denoising iterations:

\[
\hat{\boldsymbol{\mu}}_{i,1}
=
\tilde{\boldsymbol{\mu}}_i^{(R)}.
\]

Recovered physical displacement relative to the canonical frame-0 state:

\[
\boxed{
\hat{\mathbf d}_i
=
\hat{\boldsymbol{\mu}}_{i,1}
-
\boldsymbol{\mu}_{i,0}
}
\]

NOT relative to the noisy initialization.

This is critical.

Do not confuse:

```text
network correction
```

with:

```text
physical displacement
```

The network correction contains both noise cancellation and true motion.

---

# 4. Per-Gaussian learning — no hard cell labels

The final dynamic/static variable is defined for every Gaussian:

\[
P_i^{dyn}.
\]

Do not collapse Gaussians into one shared cell label.

Spatial coherence must instead be imposed through local Gaussian-neighbor message passing.

Each Gaussian retains its own:

```text
position
noise
render residual evidence
visibility
denoised displacement
dynamic probability
```

A local graph may be used for efficiency and coherence, but all outputs remain per Gaussian.

---

# 5. Frame selection

This implementation must use exactly one temporal transition:

```text
frame 0 → frame 1
```

Use all synchronized cameras available for those frames unless a debug CLI option requests a subset.

Frame 0 provides:

- canonical/static Gaussian geometry;
- optional frame-0 images only for diagnostics.

Frame 1 provides the actual supervision:

- RGB GT from every selected camera;
- depth GT where available.

Do not use any later frames.

Determine the actual timestamp difference:

\[
\Delta t_{01}.
\]

Prefer timestamps from the dataset if available.

Otherwise:

\[
\Delta t_{01}=\frac{1}{fps}.
\]

Save this value to the result file.

---

# 6. What is frozen and what is trainable

## Frozen

For this experiment freeze all canonical 3DGS parameters:

- canonical Gaussian position `_xyz`;
- SH/color features;
- opacity;
- scale;
- Gaussian rotation;
- semantic field;
- DINO field;
- camera intrinsics/extrinsics.

The canonical static PLY must remain unchanged.

## Trainable

Only the denoising network parameters \(\theta\) are trainable.

The temporary Gaussian positions used during rendering are differentiable tensors, but they are not persistent model parameters.

Do not add them to the Gaussian optimizer.

---

# 7. Initial implementation target

Add a standalone test/training entry point, preferably:

```text
tools/test_frame01_gaussian_denoiser.py
```

Add the denoising model in a self-contained module, preferably:

```text
scene/gaussian_motion_denoiser.py
```

If useful, add helper logic to:

```text
scene/render_guided_motion_probe.py
```

but do not overload unrelated Stage-2/Stage-3 code.

Do not make `train.py` mandatory for this experiment.

---

# 8. Denoiser architecture

Start simple.

Use a lightweight residual graph denoiser over Gaussians.

Recommended design:

```text
per-Gaussian feature encoder
        ↓
KNN spatial message passing
        ↓
residual MLP block
        ↓
KNN spatial message passing
        ↓
output MLP
        ↓
3D correction Δμ_i
```

A PyTorch implementation is sufficient.

Avoid adding heavyweight external graph libraries unless already installed.

KNN can be precomputed once from the canonical Gaussian positions.

Recommended:

```text
K = 16
```

or:

```text
K = 24
```

for the first experiment.

### Canonical graph

Build the neighborhood graph using frame-0 canonical positions:

\[
\mathcal N(i)
=
KNN(\boldsymbol{\mu}_{i,0}).
\]

Keep this graph fixed during denoising.

Reason:

- avoids graph topology changing because of artificial noise;
- preserves canonical surface adjacency;
- allows neighboring Gaussians to exchange motion evidence;
- preserves per-Gaussian outputs.

---

# 9. Denoiser input features

For each Gaussian \(i\) and denoising iteration \(r\), construct a feature vector.

At minimum:

\[
h_i^{(r)}
=
[
\delta_i^{(r)},
\sigma_r,
e_i^{rgb},
e_i^{depth},
g_i^{rgb},
S_i^{raw},
S_i^{vis},
S_i^{valid},
r_i^{vis},
f_i^{valid}
]
\]

where:

\[
\delta_i^{(r)}
=
\tilde{\mu}_i^{(r)}
-
\mu_{i,0}.
\]

The terms are:

### Current offset from canonical

```python
current_offset = current_xyz - canonical_xyz
```

3 values.

### Current noise/denoising level

Scalar:

```python
sigma_r
```

Use a normalized scalar or timestep embedding.

### Per-Gaussian RGB residual evidence

Compute from frame-1 image residuals and reverse accumulation.

### Per-Gaussian depth residual evidence

When depth is enabled.

### Per-Gaussian position gradient

Very useful:

\[
g_i^{rgb}
=
\frac{\partial L_{render}}{\partial \tilde{\mu}_i}.
\]

Pass a detached copy of the current render gradient to the network as an input feature.

This gives the network a direct local indication of which position change would reduce the frame-1 render error.

### Visibility features

Use:

```text
raw_support
visible_mass
valid_visible_mass
visibility_ratio
valid_fraction
```

Normalize masses robustly before passing them to the network.

### Optional static geometry features

The following may also be included:

```text
canonical xyz
log scale
opacity
Gaussian normal/covariance orientation if already easily available
```

Do not use DINO initially.

Do not use GT moving-part labels.

---

# 10. Render residual → Gaussian evidence

For each denoising iteration, render the current noisy/denoised Gaussian positions into all selected frame-1 cameras.

For camera \(c\):

\[
\hat I_c^{(r)}
=
\mathcal R(
\tilde X^{(r)},
C_c
).
\]

Compute robust RGB residual:

\[
R_c^{rgb}(u)
=
\rho(
\hat I_c^{(r)}(u)-I_{1,c}(u)
).
\]

Recommended first version:

```python
rgb_residual = torch.abs(render_rgb - gt_rgb).mean(dim=0)
```

Optionally later use Charbonnier.

For depth:

\[
R_c^D(u)
=
\rho_D(
\hat D_c^{(r)}(u)-D_{1,c}(u)
)
\]

only on valid GT depth pixels.

Reverse accumulate the residuals using the existing CUDA responsibility function.

For RGB residual:

```python
pixel_features = rgb_residual[None, ...]
pixel_confidence = valid_rgb_mask
```

For depth residual:

```python
pixel_features = depth_residual[None, ...]
pixel_confidence = valid_depth_mask
```

Use Gaussian-level grouping:

```python
group_ids = torch.arange(num_gaussians, device="cuda")
num_groups = num_gaussians
```

This is important: the output must remain Gaussian-by-Gaussian.

For Gaussian \(i\):

\[
E_i^{rgb}
=
\frac{N_i^{rgb}}
{S_i^{valid}+\epsilon}.
\]

Do the same for depth.

---

# 11. Multi-view aggregation

The same Gaussian is shared by all cameras.

Aggregate evidence across cameras at each denoising iteration.

For scalar evidence:

\[
E_i
=
\frac{
\sum_c
S_{i,c}^{valid} E_{i,c}
}{
\sum_c S_{i,c}^{valid}+\epsilon
}.
\]

For gradients:

\[
\mathbf g_i
=
\frac{
\sum_c
w_{i,c}\mathbf g_{i,c}
}{
\sum_cw_{i,c}+\epsilon
}
\]

with:

\[
w_{i,c}
=
S_{i,c}^{valid}.
\]

Also estimate multiview directional agreement:

\[
C_i^{view}
=
\frac{
\left\|
\sum_c
w_{i,c}
\hat{\mathbf g}_{i,c}
\right\|
}{
\sum_cw_{i,c}+\epsilon
}.
\]

This may be passed to the network or saved as diagnostic evidence.

Occluded/unprojected cameras must not count as static evidence.

---

# 12. Visibility handling

Use the native CUDA values.

For each Gaussian/camera:

\[
S_i^{raw}
\]

\[
S_i^{vis}
\]

\[
S_i^{valid}.
\]

Compute:

\[
r_i^{vis}
=
\frac{S_i^{vis}}
{S_i^{raw}+\epsilon}
\]

and:

\[
f_i^{valid}
=
\frac{S_i^{valid}}
{S_i^{vis}+\epsilon}.
\]

Interpretation:

```text
raw ≈ 0
→ not projected
→ no evidence

raw > 0, visible/raw small
→ occluded/deep
→ weak or no supervision

visible > 0, valid/visible small
→ geometrically visible but invalid observation
→ weak or no supervision

valid_visible_mass significant
→ valid render supervision
```

Do not classify an unsupported Gaussian as static merely because its displacement is small.

---

# 13. Iterative denoising

This experiment must contain repeated denoising steps for the same sampled artificial-noise realization.

Example:

```text
canonical X0
    ↓
sample noise
    ↓
X^0
    ↓ render → evidence
Dθ
    ↓
X^1
    ↓ render → evidence
Dθ
    ↓
X^2
    ↓
...
    ↓
X^R
```

Use the same denoiser weights at every denoising iteration.

This is recurrent/iterative denoising.

Recommended first value:

```text
R = 6 denoising iterations
```

Do not use 20 initially.

We first need to verify that the network learns stable iterative corrections without exploding.

---

# 14. Noise schedule

Use positional noise in world coordinates.

Do not use one fixed sigma for every denoising iteration.

Use a descending schedule:

\[
\sigma_0 > \sigma_1 > \dots > \sigma_{R-1}.
\]

Set it relative to local Gaussian spacing.

Compute:

```python
median_nn_distance
```

from the canonical KNN graph.

Recommended initial schedule:

\[
\sigma_{max}
=
0.15\,d_{NN}
\]

\[
\sigma_{min}
=
0.02\,d_{NN}.
\]

Use geometric interpolation across denoising steps.

For example, if:

```text
median_nn_distance = h
```

then:

```text
sigma:
0.15h
0.10h
0.07h
0.05h
0.03h
0.02h
```

The artificial noise must be large enough to make denoising nontrivial, but not so large that Gaussian correspondences/geometry are destroyed.

Save the actual derived values to JSON.

---

# 15. Training strategy

We have no GT clean frame-1 Gaussian positions.

Therefore do not use supervised noise-prediction loss against a known clean \(X_1\).

The training loss comes from rendering the progressively denoised Gaussian configuration against the real frame-1 multiview RGB-D data.

For each training step:

1. sample a fresh artificial noise field;
2. initialize noisy Gaussian positions;
3. perform \(R\) iterative denoising steps;
4. render the final result;
5. optionally render intermediate denoising states;
6. backpropagate the frame-1 rendering loss through all denoising steps into \(D_\theta\).

This makes the denoiser learn:

```text
given:
- current noisy Gaussian state
- render residual
- local neighbor context
- visibility evidence

predict:
- a correction that removes artificial noise
- plus any real frame0→frame1 motion needed to explain frame 1
```

---

# 16. Training losses

## 16.1 Final multiview RGB loss

\[
L_{rgb}
=
\frac1{N_C}
\sum_c
\|
\hat I_{1,c}-I_{1,c}
\|_1.
\]

Recommended:

```text
lambda_rgb = 1.0
```

## 16.2 SSIM

Use existing repository SSIM implementation.

\[
L_{ssim}
=
1-\text{SSIM}.
\]

Recommended:

```text
lambda_ssim = 0.2
```

## 16.3 Depth loss

If depth is available and convention is verified.

Use the same depth convention as the existing dynamic pipeline.

Recommended:

```text
lambda_depth = 0.1
```

Do not silently compare metric depth to inverse depth.

## 16.4 Intermediate denoising render loss

Useful to ensure every recurrent step improves rather than only the last one.

For denoising iteration \(r\):

\[
L_{render}^{(r)}.
\]

Weight later steps more strongly.

Example:

\[
L_{iter}
=
\sum_{r=1}^{R}
\gamma_r
L_{render}^{(r)}
\]

with:

\[
\gamma_r
\propto
\frac{r}{R}.
\]

Recommended:

```text
lambda_iter = 0.25
```

in addition to the final loss.

## 16.5 Spatial correction smoothness

Use the canonical Gaussian graph.

For neighboring Gaussians:

\[
L_{spatial}
=
\frac{1}{|E|}
\sum_{(i,j)}
w_{ij}
\|
\hat d_i-\hat d_j
\|_1.
\]

Use weak weight only.

Recommended:

```text
lambda_spatial = 0.005
```

This is only an inductive bias.

It must not force both sides of an articulation boundary to move together.

## 16.6 Motion sparsity

Most Gaussians are expected to be static.

Apply weak sparsity on recovered physical displacement:

\[
L_{motion}
=
\frac1N
\sum_i
\rho(
\|\hat d_i\|
).
\]

Recommended:

```text
lambda_motion = 0.002
```

Use a robust Charbonnier/pseudo-Huber penalty.

Do not make this large enough to suppress actual motion.

---

# 17. Very important anti-collapse constraints

We previously observed that unconstrained render optimization makes many static Gaussians move slightly to compensate for ordinary reconstruction/calibration errors.

This implementation must explicitly detect that failure.

Track:

```text
mean recovered displacement
median recovered displacement
90th percentile
max displacement
fraction above 0.25 * median_nn_distance
fraction above 0.50 * median_nn_distance
```

Also compare:

```text
static canonical render loss at frame 1
final denoised render loss
```

A useful denoiser should produce meaningful render improvement without moving most of the entire scene.

Do not simply decrease a threshold to manufacture a static/dynamic split.

---

# 18. Multiple independent noising trials

After training the denoiser, run inference with:

```text
S = 8
```

independent artificial noise realizations.

For trial \(s\):

\[
\epsilon_i^{(s)}
\]

produces final recovered displacement:

\[
\hat d_i^{(s)}.
\]

Compute:

\[
\bar d_i
=
\frac1S
\sum_s
\hat d_i^{(s)}.
\]

Mean recovered speed:

\[
\bar v_i
=
\frac{\bar d_i}{\Delta t_{01}}.
\]

Compute displacement magnitude:

\[
m_i
=
\|\bar d_i\|.
\]

Compute seed/noise consistency:

\[
C_i^{seed}
=
\frac{
\|\bar d_i\|
}{
\frac1S
\sum_s
\|\hat d_i^{(s)}\|+\epsilon
}.
\]

This quantity must be clamped to:

\[
[0,1].
\]

Interpretation:

```text
all trials recover same nonzero displacement
→ consistency ≈ 1

different random residual motions cancel
→ consistency small
```

For a truly static Gaussian:

```text
all recovered d_i^(s) should be near zero
```

so do not rely on directional consistency for the static decision.

Also compute:

\[
M_i^{static}
=
\frac1S
\sum_s
\|\hat d_i^{(s)}\|.
\]

Static evidence is strong when this is small.

---

# 19. Render-improvement evidence per Gaussian

This must be included.

Dynamic probability must not come only from displacement magnitude.

For each final denoised trial, compare:

## Static hypothesis

Render canonical frame-0 Gaussian positions directly against frame 1.

Per-Gaussian responsibility-weighted residual:

\[
E_i^{static}.
\]

## Denoised/moved hypothesis

Render final denoised positions.

\[
E_i^{moved}.
\]

Then compute:

\[
\boxed{
G_i
=
\frac{
E_i^{static}
-
E_i^{moved}
}{
E_i^{static}+\epsilon
}
}
\]

for Gaussians with sufficient valid visibility.

Interpretation:

```text
G_i > 0
→ allowing this Gaussian's region to move improved its frame-1 rendering evidence

G_i ≈ 0
→ moving it provided little benefit

G_i < 0
→ moving it made its attributed residual worse
```

This is necessary to distinguish real motion from arbitrary render-error compensation.

Average `G_i` across noise trials.

---

# 20. Dynamic probability

The experiment must produce a continuous per-Gaussian probability.

Do not use hard cell classes internally.

For this first implementation use an interpretable evidence-based probability after denoiser inference.

Define normalized quantities.

### Motion magnitude evidence

Let:

\[
d_{scale}
=
0.10\,d_{NN}.
\]

Then:

\[
M_i
=
1-
\exp(
-\|\bar d_i\|/d_{scale}
).
\]

### Noise-invariant motion consistency

\[
C_i^{seed}
\in[0,1].
\]

### Render improvement

Map:

\[
G_i
\]

to:

\[
R_i
=
\sigma(
(G_i-G_0)/T_G
).
\]

Recommended:

```text
G0 = 0.03
TG = 0.02
```

### Visibility confidence

Use:

\[
V_i
=
\text{robust normalized valid_visible_mass}
\]

aggregated over all frame-1 cameras.

### Initial probability

Use:

\[
\boxed{
P_i^{dyn}
=
V_i
\,
M_i
\,
C_i^{seed}
\,
R_i
}
\]

clamped to `[0, 1]`.

This is the initial diagnostic probability.

Do not hide its components.

Save every factor individually.

This probability is NOT meant to be the final production formulation; it is for validating whether learned noising/denoising produces meaningful Gaussian-level motion evidence.

If later required, a learned probability head can be added.

---

# 21. Static probability

Also save:

\[
P_i^{static}.
\]

Define static evidence only when the Gaussian is visible.

A simple first version:

\[
S_i
=
\exp(
-
M_i^{static}/d_{static}
)
\]

with:

\[
d_{static}
=
0.03\,d_{NN}.
\]

Multiply by lack of meaningful render improvement:

\[
R_i^{static}
=
1-
R_i.
\]

Then:

\[
P_i^{static}
=
V_i
S_i
R_i^{static}.
\]

Do not force:

\[
P_i^{dyn}+P_i^{static}=1
\]

because unsupported/ambiguous Gaussians should remain uncertain.

Define uncertainty:

\[
P_i^{unknown}
=
1-
\max(
P_i^{dyn},
P_i^{static}
).
\]

Clamp to `[0,1]`.

---

# 22. Network-training/inference separation

Training:

```text
many random noising realizations
→ learn D_theta
```

Inference:

```text
freeze D_theta
→ run S independent noise trials
→ estimate displacement distribution
→ estimate P_dynamic per Gaussian
```

Do not continue changing network weights during the final probability-estimation trials.

---

# 23. Recommended initial training parameters

Start with these values.

```text
frame_from                 = 0
frame_to                   = 1

training_iterations        = 1000

denoise_iterations         = 6

knn                        = 16

hidden_dim                 = 96
message_passing_layers     = 2

train_noise_sigma_max      = 0.15 * median_nn_distance
train_noise_sigma_min      = 0.02 * median_nn_distance

optimizer                  = AdamW
learning_rate              = 1e-3
weight_decay               = 1e-5

lambda_rgb                 = 1.0
lambda_ssim                = 0.2
lambda_depth               = 0.1
lambda_iter                = 0.25
lambda_spatial             = 0.005
lambda_motion              = 0.002

inference_noise_trials     = 8

render_gain_center         = 0.03
render_gain_temperature    = 0.02

dynamic_motion_scale       = 0.10 * median_nn_distance
static_motion_scale        = 0.03 * median_nn_distance
```

These are starting values, not universal physical constants.

All spatial values must be derived from the actual static Gaussian nearest-neighbor scale.

---

# 24. Camera use

Training should preferably use all available synchronized cameras at frame 1.

If VRAM is too high:

```text
sample 4 cameras per training iteration
```

but periodically evaluate using all cameras.

At final inference/evaluation:

```text
use all cameras
```

for render residual and responsibility aggregation.

Add CLI:

```text
--train-cameras-per-iter
--eval-all-cameras
```

but give them sensible defaults.

---

# 25. Memory-efficient implementation

Do not create a full dense `[N,N]` graph.

Use fixed KNN indices:

```python
knn_idx: [N, K]
```

and gather neighbor features.

Do not keep every rasterization graph alive simultaneously.

For recurrent denoising:

```text
render
→ collect evidence
→ network correction
→ next state
```

Only preserve differentiability needed for the current training objective.

If full backpropagation through all 6 renders exceeds memory, support truncated backpropagation as a fallback, but first implement the full differentiable version.

---

# 26. Canonical geometry safety

Before training:

```python
canonical_xyz_copy = gaussians._xyz.detach().clone()
```

After every training iteration and after the whole experiment verify:

```python
max_abs_change = (
    gaussians._xyz.detach() - canonical_xyz_copy
).abs().max()
```

This must remain exactly or numerically zero.

Print:

```text
canonical geometry max abs change: ...
```

and assert below a tiny tolerance.

Use temporary renderer state rather than overwriting `_xyz`.

---

# 27. Required outputs

Save under:

```text
output/<scene>_frame01_denoiser/
```

At minimum:

## Model

```text
denoiser_frame0_to_1.pt
```

Include:

```text
state_dict
network configuration
KNN size
noise schedule
median_nn_distance
delta_t
training configuration
```

## Per-Gaussian results

```text
per_gaussian_frame01.pt
```

with:

```python
{
    "canonical_xyz": ...,
    "mean_predicted_xyz_frame1": ...,
    "mean_displacement": ...,
    "mean_velocity": ...,
    "mean_displacement_norm": ...,
    "trial_displacements": ...,
    "seed_consistency": ...,
    "render_gain": ...,
    "static_residual": ...,
    "moved_residual": ...,
    "raw_support": ...,
    "visible_mass": ...,
    "valid_visible_mass": ...,
    "visibility_ratio": ...,
    "valid_fraction": ...,
    "dynamic_probability": ...,
    "static_probability": ...,
    "unknown_probability": ...,
    "delta_t": ...,
}
```

## JSON summary

```text
frame01_summary.json
```

Include:

```text
number of Gaussians
number of cameras
delta_t
median nearest-neighbor distance
noise schedule
training iterations
final RGB loss
final SSIM loss
final depth loss
canonical frame1 static-render loss
denoised frame1 render loss
mean displacement
median displacement
P10/P50/P90/P99 displacement
dynamic-probability quantiles
static-probability quantiles
visibility quantiles
canonical geometry max change
```

---

# 28. Required visual diagnostics

These are mandatory.

## 28.1 Per-camera image comparison

For every frame-1 camera save:

```text
GT RGB
static frame-0 3DGS render
initial noisy render
final denoised render
static absolute residual
denoised absolute residual
```

Prefer one tiled PNG per camera.

Example:

```text
debug/cam_000_frame01.png
```

## 28.2 3D Gaussian displacement visualization

Create HTML or PLY showing:

```text
canonical Gaussian position
predicted frame-1 position
displacement vector
```

Color Gaussians by:

```text
dynamic_probability
```

## 28.3 Probability visualization

Create a separate 3D visualization:

```text
blue   → static
red    → dynamic
gray   → uncertain / unsupported
```

But keep the underlying probability continuous.

## 28.4 Denoising trajectory visualization

For a small sampled subset of Gaussians save:

```text
x^0
x^1
...
x^R
```

to verify that the iterative denoiser genuinely moves from noisy states toward stable final states.

---

# 29. Required W&B / console metrics

W&B is optional for the standalone test but should be supported if already configured.

Log:

```text
train/rgb
train/ssim
train/depth
train/total

train/displacement_mean
train/displacement_median
train/displacement_p90

train/correction_norm_mean
train/correction_norm_p90

train/valid_visible_fraction

eval/static_frame1_rgb
eval/denoised_frame1_rgb
eval/render_gain

eval/dynamic_probability_mean
eval/dynamic_probability_p90

debug/canonical_xyz_max_change
```

Also log one multiview diagnostic image periodically.

---

# 30. Critical validation experiments

The script must automatically perform these sanity checks.

## A. Identity/static-target test

Optional debug mode:

Use frame 0 images as both source and target.

Then the correct physical motion is:

\[
d_i=0.
\]

Inject artificial noise and train/evaluate denoising.

Expected:

```text
mean recovered physical displacement → near zero
dynamic probabilities → near zero for visible Gaussians
```

Add CLI:

```text
--target-same-as-source
```

This is an essential unit test.

## B. Frame-1 test

Normal experiment:

```text
frame 0 geometry
→ frame 1 GT
```

Expected:

```text
some spatially coherent subset retains nonzero physical displacement
static regions return near canonical position
frame-1 render loss improves significantly
```

## C. Noise-repeatability test

For the same trained denoiser, use 8 random seeds.

A true moving Gaussian should recover approximately the same physical displacement from different artificial noises.

Static Gaussian:

\[
d_i^{(s)}\approx0.
\]

Dynamic Gaussian:

\[
d_i^{(1)}
\approx
d_i^{(2)}
\approx
\dots
\approx
d_i.
\]

---

# 31. Failure conditions

Print explicit warnings or fail if:

### Numerical

```text
NaN/Inf in network output
NaN/Inf in positions
NaN/Inf in renderer
NaN/Inf in probability
```

### Canonical corruption

```text
canonical _xyz changed
```

### Probability bug

```text
dynamic_probability outside [0,1]
static_probability outside [0,1]
seed_consistency outside [0,1]
```

### Motion explosion

Warn if:

```text
median displacement > 0.25 * median_nn_distance
```

or:

```text
>50% of visible Gaussians move more than 0.25 * median_nn_distance
```

for this one-frame experiment.

### No learning

Warn if the denoised frame-1 render does not improve over the static frame-0 render.

---

# 32. CLI

The standalone script should expose at least:

```text
--root-scene
--static-model-dir
--iteration
--output-dir

--source-frame
--target-frame

--training-iterations
--denoise-iterations
--knn
--hidden-dim

--noise-max-ratio
--noise-min-ratio

--lr

--lambda-rgb
--lambda-ssim
--lambda-depth
--lambda-iter
--lambda-spatial
--lambda-motion

--train-cameras-per-iter
--inference-trials

--use-depth
--seed
--verbose

--target-same-as-source
```

Defaults:

```text
source_frame          = 0
target_frame          = 1
training_iterations   = 1000
denoise_iterations    = 6
knn                   = 16
hidden_dim            = 96
noise_max_ratio       = 0.15
noise_min_ratio       = 0.02
lr                    = 1e-3
inference_trials      = 8
```

---

# 33. Minimal expected command for USB_100109

The final implementation should support:

```bash
cd ~/Joint3DGS/RigEuler3DGS
conda activate rigeuler3dgs

CUDA_LAUNCH_BLOCKING=1 python tools/test_frame01_gaussian_denoiser.py \
  --root-scene /home/samuelemara/Joint3DGS/RigEuler3DGS/input/USB_100109 \
  --static-model-dir /home/samuelemara/Joint3DGS/RigEuler3DGS/output/USB_100109 \
  --iteration 30000 \
  --output-dir /home/samuelemara/Joint3DGS/RigEuler3DGS/output/USB_100109_frame01_denoiser \
  --source-frame 0 \
  --target-frame 1 \
  --use-depth
```

Do not invent additional mandatory flags.

---

# 34. Recommended implementation sequence for Codex

Implement incrementally and verify each stage before continuing.

## Step 1 — data/camera smoke test

Load:

```text
static PLY
frame-0 cameras
frame-1 cameras
frame-1 RGB
frame-1 depth
```

Render canonical static 3DGS into frame 1.

Save debug images.

## Step 2 — Gaussian-level CUDA evidence test

For canonical/static frame-1 render:

```text
compute RGB residual
reverse accumulate residual Gaussian-by-Gaussian
save residual/support tensors
```

Verify:

\[
0
\le
S^{valid}
\le
S^{vis}
\le
S^{raw}.
\]

## Step 3 — noise/render test

Inject Gaussian positional noise.

Render frame 1.

Verify render loss becomes worse than canonical/static render on average.

## Step 4 — denoiser forward pass

Implement the Gaussian graph denoiser.

Run one forward correction without training.

Verify shapes/gradients.

## Step 5 — train on identity test first

Use:

```text
frame 0 → frame 0
```

Train the network.

It must learn to cancel artificial noise and recover canonical positions.

Do not proceed until this works.

## Step 6 — frame 0 → frame 1

Train against real frame-1 observations.

## Step 7 — multiple independent noise inference

Run 8 noise seeds.

Compute:

```text
mean displacement
velocity
seed consistency
render gain
dynamic/static probability
```

## Step 8 — visual diagnostics

Generate all required images and 3D probability/displacement visualization.

---

# 35. Conceptual success criterion

This experiment is successful only if the following behavior emerges.

## Static Gaussian

For different artificial perturbations:

\[
\epsilon_i^{(1)},
\epsilon_i^{(2)},
\dots
\]

the network predicts:

\[
\Delta\mu_i^{(s)}
\approx
-\epsilon_i^{(s)}.
\]

Therefore:

\[
\hat d_i^{(s)}
=
\epsilon_i^{(s)}
+
\Delta\mu_i^{(s)}
\approx0.
\]

## Dynamic Gaussian

For different artificial perturbations:

\[
\Delta\mu_i^{(s)}
\approx
-\epsilon_i^{(s)}
+
d_i.
\]

Therefore:

\[
\hat d_i^{(s)}
\approx
d_i
\]

independently of artificial noise.

This noise-invariant residual displacement is the central self-supervised motion signal.

The CUDA renderer supplies the supervision because the correct \(d_i\) is the one that causes the denoised Gaussians to render the real frame-1 multiview RGB-D observations.

---

# 36. Do not overinterpret the result yet

This one-step implementation is only meant to establish:

```text
noising
→ learned denoising
→ frame-1 render improvement
→ reproducible Gaussian-level residual motion
→ Gaussian-level dynamic probability
```

It is NOT yet the final dynamic reconstruction pipeline.

Only after this experiment works should the recovered:

```text
P_dynamic_i
d_i
v_i
```

be passed to the later rigid-motion / Euler / joint-discovery stages.

For now, keep the implementation completely isolated and easy to debug.
