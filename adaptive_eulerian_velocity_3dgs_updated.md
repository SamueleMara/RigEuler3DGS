# Adaptive Eulerian Velocity Fields for Motion-Aware Dynamic 3D Gaussian Splatting

**Working title:** Motion-Adaptive Eulerian 3D Gaussian Splatting  
**Purpose:** Theoretical formulation and implementation blueprint for learning a physically interpretable 3D velocity field that transports canonical 3D Gaussians through video time, with adaptive refinement only where Gaussian motion becomes incoherent.  
**Date:** 2026-07-06

---

## 1. Executive Summary

The core idea is to start from a canonical 3D Gaussian Splatting (3DGS) reconstruction at an initial time, then learn a continuous Eulerian velocity field that transports the Gaussians through the following frames of a video. Instead of predicting an independent deformation for each Gaussian at each timestamp, every Gaussian queries a shared motion field:

```text
d mu_i(t) / dt = v_theta(mu_i(t), t)
```

The Gaussian center is updated by integrating the velocity field over time:

```text
mu_i(t_2) = mu_i(t_1) + integral_{t_1}^{t_2} v_theta(mu_i(tau), tau) d tau
```

The method is not simply a 4DGS-style deformation MLP. The key distinction is that the model is trained as a **transport process**. This means that motion must be temporally compositional, forward/backward consistent, locally rigid where appropriate, and explainable by low-dimensional SE(3) or joint-like motion in coherent regions.

The adaptive part is also central: the velocity field is represented on a sparse grid or hash-grid defined only near the occupied support of the initial Gaussians. The grid is refined only where neighboring Gaussians cannot be explained by coherent motion. This creates a compact, interpretable motion model: coherent rigid regions remain coarse, while motion boundaries, articulated joints, occlusion regions, and non-rigid zones receive higher motion resolution.

---

## 2. Motivation

Dynamic 3DGS methods can render moving scenes by learning time-dependent Gaussian deformations. However, good dynamic rendering does not necessarily imply correct 3D motion. If the objective is only RGB/SSIM reconstruction, a model can produce visually convincing frames while learning physically wrong Gaussian trajectories.

For articulated object understanding, this is not enough. We need motion that is useful for:

- estimating static and dynamic parts;
- extracting coherent rigid regions;
- discovering revolute or prismatic joints;
- estimating joint axes and centers;
- tracking long-lived scene elements through occlusions;
- separating true motion from rendering-only deformation;
- obtaining velocity and acceleration signals for later physical reasoning.

The proposed representation therefore shifts the goal from:

```text
render frame t well
```

to:

```text
transport the canonical Gaussian scene through time using a coherent 3D velocity field, while still matching the rendered video frames.
```

---

## 3. Key Difference from a Standard 4DGS-Style MLP

A typical 4DGS/deformation-field formulation predicts a deformation directly from canonical space to a target time:

```text
mu_i(t) = mu_i(0) + Delta_theta(mu_i(0), t)
```

This is a direct canonical-to-time mapping. It can render dynamic scenes efficiently, but it does not necessarily impose that the motion from time 0 to time 2 is equal to the composition of motion from 0 to 1 and 1 to 2.

The Eulerian velocity-field formulation instead defines instantaneous motion:

```text
d mu_i(t) / dt = v_theta(mu_i(t), t)
```

The position at a later time is obtained by integration. This introduces a flow map:

```text
Phi_{a -> b}(x) = x + integral_a^b v_theta(Phi_{a -> tau}(x), tau) d tau
```

A meaningful velocity field should satisfy temporal composition:

```text
Phi_{0 -> 2}(x) approx Phi_{1 -> 2}(Phi_{0 -> 1}(x))
```

This is the real theoretical key. If this constraint is not enforced, the velocity model can degenerate into a disguised 4D deformation MLP.

### Practical interpretation

- 4DGS-style deformation: “Where should this Gaussian be at time t?”
- Eulerian velocity field: “How does space move at this location and time, and where does the Gaussian arrive if it follows that motion?”

The second formulation is better suited for motion understanding, tracking, and joint estimation.

---

## 4. Problem Setting

Assume a dynamic RGB-D or multiview RGB-D video:

```text
{I_{c,k}, D_{c,k}, K_c, T_{cw,c,k}} for cameras c = 1..C and frames k = 0..T
```

where:

- `I_{c,k}` is the RGB image from camera `c` at frame `k`;
- `D_{c,k}` is the depth map, if available;
- `K_c` is the camera intrinsic matrix;
- `T_{cw,c,k}` is the camera-to-world or world-to-camera pose, depending on convention;
- cameras may be static or moving, but poses are assumed known or estimated.

The initial goal is to reconstruct a canonical Gaussian scene at time `t_0`:

```text
G_0 = {g_i}_{i=1}^N
```

Each Gaussian has:

```text
g_i = {mu_i^0, Sigma_i^0, R_i^0, s_i^0, alpha_i, color_i or SH_i}
```

where:

- `mu_i^0 in R^3` is the center;
- `Sigma_i^0` is the covariance;
- `R_i^0` and `s_i^0` parameterize orientation and scale;
- `alpha_i` is opacity;
- `color_i` or `SH_i` are color/spherical-harmonic appearance parameters.

The dynamic model should produce, for every frame `k`, a deformed Gaussian set:

```text
G_k = T_motion(G_0, t_k)
```

which is rendered into the available cameras and compared to the observed frames.

---

## 5. Canonical Gaussian Scene

The canonical scene can be obtained in different ways:

1. Train a static 3DGS from the first frame or first short static window.
2. Train a static 3DGS from all frames after masking dynamic areas.
3. Back-project RGB-D frames into a fused point cloud at `t_0`, then initialize Gaussians from this cloud.
4. Use a pre-existing Gaussian reconstruction and optimize only motion.

For the proposed method, the cleanest starting point is:

```text
Stage A: train/fix canonical Gaussians at t_0.
Stage B: freeze most appearance and geometry parameters.
Stage C: learn motion field while rendering later frames.
Stage D: optionally refine geometry and appearance after motion is stable.
```

Freezing appearance at the beginning is important. If color and opacity are too free, the model may explain motion by changing appearance instead of moving geometry.

---

## 6. Occupancy-Supported Motion Domain

The velocity field should not be defined densely everywhere in 3D space. Most space is empty. Use the initial Gaussians to define an occupied support:

```text
Omega_G = {x in R^3 : rho_G(x) > tau_occ}
```

where `rho_G(x)` is a Gaussian density or opacity-derived occupancy function:

```text
rho_G(x) = sum_i alpha_i exp(-0.5 (x - mu_i)^T Sigma_i^{-1} (x - mu_i))
```

Then build a sparse voxel grid, hash grid, octree, or set of motion anchors only inside or near `Omega_G`.

This has three advantages:

1. **Efficiency:** no motion capacity is wasted in empty space.
2. **Regularization:** the field is tied to observed geometry.
3. **Interpretability:** each cell or anchor corresponds to an occupied local region of the object/scene.

Occupancy can also be projected into image space to build visibility masks and to associate rendering residuals with specific 3D cells.

---

## 7. Eulerian Velocity Field

The motion field is defined as:

```text
v_theta: R^3 x R -> R^3
```

where:

```text
v_i(t) = v_theta(mu_i(t), t)
```

A practical parameterization is:

```text
v_theta(x,t) = MLP([gamma_x(x), gamma_t(t), F_grid(x,t)])
```

where:

- `gamma_x(x)` is positional encoding of 3D position;
- `gamma_t(t)` is temporal encoding;
- `F_grid(x,t)` is an interpolated feature from a sparse trainable grid/hash grid;
- the MLP is small and acts as a decoder from local motion features to velocity.

A more structured version decomposes the field as:

```text
v_theta(x,t) = v_global(t) + v_local(x,t)
```

where `v_global(t)` captures global camera/object drift and `v_local(x,t)` captures local motion.

For articulated scenes, a useful decomposition is:

```text
v_theta(x,t) = p_static(x) * 0 + sum_m p_m(x,t) * v_m(x,t)
```

where:

- `p_static(x)` is the probability of being static;
- `p_m(x,t)` is the soft assignment to motion component `m`;
- `v_m` can be a local SE(3), revolute, prismatic, or generic velocity field.

---

## 8. Gaussian Transport

### 8.1 Center transport

The Gaussian center evolves according to:

```text
d mu_i(t) / dt = v_theta(mu_i(t), t)
```

Discrete Euler integration:

```text
mu_i^{k+1} = mu_i^k + Delta_t * v_theta(mu_i^k, t_k)
```

Higher-quality integration can use midpoint or RK4:

```text
k1 = v(mu_i^k, t_k)
k2 = v(mu_i^k + 0.5 Delta_t k1, t_k + 0.5 Delta_t)
k3 = v(mu_i^k + 0.5 Delta_t k2, t_k + 0.5 Delta_t)
k4 = v(mu_i^k + Delta_t k3, t_k + Delta_t)
mu_i^{k+1} = mu_i^k + Delta_t/6 * (k1 + 2 k2 + 2 k3 + k4)
```

For the first implementation, Euler or midpoint integration is sufficient.

### 8.2 Rotation and covariance transport

Moving only centers is the simplest version, but anisotropic Gaussians should rotate with the local motion. There are three levels of complexity.

#### Level 1: center-only motion

```text
mu_i(t) changes, Sigma_i fixed
```

Good for a prototype, but can produce artifacts when elongated Gaussians should rotate with the object.

#### Level 2: predicted angular velocity

The field predicts both linear and angular velocity:

```text
[v_theta(x,t), omega_theta(x,t)] = MLP(...)
```

Then update orientation:

```text
R_i^{k+1} = Exp(Delta_t * omega_i^k) R_i^k
Sigma_i^{k+1} = R_i^{k+1} S_i (R_i^{k+1})^T
```

where `S_i` is the diagonal scale covariance in the local Gaussian frame.

#### Level 3: local deformation gradient

Compute the Jacobian of the flow:

```text
J_v(x,t) = partial v_theta(x,t) / partial x
```

The covariance evolves approximately as:

```text
Sigma_i^{k+1} approx A_i Sigma_i^k A_i^T
A_i = I + Delta_t * J_v(mu_i^k,t_k)
```

This is theoretically elegant but more expensive and more unstable. For articulated rigid objects, predicting a local angular velocity is usually cleaner.

---

## 9. Rendering Supervision

For each frame and camera, render the transported Gaussian set:

```text
I_hat_{c,k} = Render(G_k, K_c, T_{cw,c,k})
```

Photometric loss:

```text
L_rgb = sum_{c,k} || I_hat_{c,k} - I_{c,k} ||_1
```

SSIM loss:

```text
L_ssim = sum_{c,k} (1 - SSIM(I_hat_{c,k}, I_{c,k}))
```

Combined rendering loss:

```text
L_render = lambda_rgb L_rgb + lambda_ssim L_ssim
```

If depth is available, add depth rendering supervision:

```text
L_depth = sum_{c,k} M_{c,k} || D_hat_{c,k} - D_{c,k} ||_1
```

where `M_{c,k}` masks valid depth pixels and/or dynamic object regions.

If masks are available:

```text
L_mask = BCE(A_hat_{c,k}, M_{c,k})
```

where `A_hat` is rendered alpha/accumulated opacity.

For motion learning, depth and masks are extremely valuable because RGB alone is underconstrained.

---

## 10. Transport-Specific Losses

The following losses are what make this approach different from a generic deformation MLP.

### 10.1 Rollout rendering loss

Instead of always mapping from canonical time directly to target time, explicitly roll the Gaussians forward:

```text
G_0 -> G_1 -> G_2 -> ... -> G_k
```

Then render every state. The loss becomes:

```text
L_rollout = sum_k L_render(Render(G_k), I_k)
```

This forces the learned field to act as a time-consistent transport process.

### 10.2 Forward-backward cycle consistency

Integrate forward from `t_a` to `t_b`, then backward from `t_b` to `t_a`:

```text
x_b = Phi_{a -> b}(x_a)
x_hat_a = Phi_{b -> a}(x_b)
```

Cycle loss:

```text
L_cycle = sum_i || x_hat_a_i - x_a_i ||_2^2
```

This helps prevent drift and unstable trajectories.

### 10.3 Flow composition consistency

For three times `t_a < t_b < t_c`:

```text
Phi_{a -> c}(x) approx Phi_{b -> c}(Phi_{a -> b}(x))
```

Loss:

```text
L_comp = sum_i || Phi_{a -> c}(mu_i) - Phi_{b -> c}(Phi_{a -> b}(mu_i)) ||_2^2
```

This is a central theoretical constraint. A standard canonical-to-time deformation MLP does not automatically satisfy it.

### 10.4 Local rigidity loss

For neighboring Gaussians that should belong to the same rigid region, preserve pairwise distances:

```text
L_rigid = sum_{(i,j) in E} w_ij ( ||mu_i(t)-mu_j(t)||_2 - ||mu_i(0)-mu_j(0)||_2 )^2
```

The graph `E` can be a kNN graph in canonical Gaussian space. The weight `w_ij` can depend on:

- spatial proximity;
- color similarity;
- DINO/semantic feature similarity;
- opacity confidence;
- current dynamic/static confidence;
- whether the pair belongs to the same adaptive motion cell.

This loss is essential for articulated objects. Without it, the velocity field may become rubber-like.

### 10.5 Local SE(3) explainability loss

Inside each motion grid cell `c`, estimate the best local rigid motion parameters:

```text
v_c(x,t) = omega_c(t) x (x - a_c) + u_c(t)
```

where:

- `omega_c(t)` is local angular velocity;
- `a_c` is a local pivot/reference point;
- `u_c(t)` is local translation velocity.

Then penalize deviation from locally rigid velocity:

```text
L_se3 = sum_c sum_{i in c} || v_theta(mu_i,t) - [omega_c x (mu_i - a_c) + u_c] ||_2^2
```

This makes each cell prefer coherent rigid motion unless evidence says otherwise.

### 10.6 Static prior

For background or fixed object parts, velocity should be zero:

```text
L_static = sum_i (1 - p_dyn_i) || v_theta(mu_i,t) ||_2^2
```

If static anchors can be identified from long-lived points or background masks, they should receive a stronger static penalty.

### 10.7 Smooth acceleration loss

Velocity should not change violently unless the video supports it:

```text
L_acc = sum_i || v_theta(mu_i(t_{k+1}), t_{k+1}) - v_theta(mu_i(t_k), t_k) ||_2^2
```

For a continuous-time formulation, this approximates acceleration regularization.

### 10.8 Occupancy and visibility consistency

If a Gaussian is not visible at frame `k`, its photometric supervision is weak or absent. Use rendered alpha/depth visibility to weight losses:

```text
w_vis(i,k,c) = visibility of Gaussian i in camera c at frame k
```

Then apply stronger photometric or depth residuals only where visible. For occluded Gaussians, use rollout/cycle/rigidity/static losses to prevent drift.

---

## 11. Adaptive Motion Grid

The velocity field should not have uniform resolution. Use a coarse-to-fine motion grid over the occupied support.

Each grid cell `c` stores a learnable motion feature:

```text
z_c(t) or z_c with temporal encoding
```

The MLP decodes local velocity from interpolated cell features:

```text
v_theta(x,t) = MLP(gamma_x(x), gamma_t(t), InterpGrid(z, x))
```

### 11.1 Cell splitting criterion

A cell should split if the Gaussians inside it cannot be explained by coherent motion.

Define a cell score:

```text
S_c = lambda_photo E_photo(c)
    + lambda_vel Var_{i in c}(v_i)
    + lambda_rigid E_rigid(c)
    + lambda_se3 E_se3(c)
    + lambda_depth E_depth(c)
```

where:

- `E_photo(c)` is the rendering residual attributed to Gaussians in cell `c`;
- `Var(v_i)` is velocity variance within the cell;
- `E_rigid(c)` is distance-preservation error;
- `E_se3(c)` is local SE(3) fitting error;
- `E_depth(c)` is rendered depth mismatch for projections of that cell.

Split if:

```text
S_c > tau_split
```

and the cell contains enough Gaussians.

### 11.2 Cell merging criterion

Merge adjacent cells if they are motion-compatible:

```text
|| mean_v(c1) - mean_v(c2) || < tau_v
E_se3(c1 union c2) < tau_se3
E_photo(c1 union c2) does not increase significantly
```

This keeps the representation compact.

### 11.3 Cell freezing criterion

Freeze a cell if it is consistently static or consistently well explained:

```text
mean ||v_i|| < tau_static
E_photo(c) < tau_photo
E_rigid(c) < tau_rigid
```

Frozen cells save computation and reduce overfitting.

### 11.4 Relation to 3DGS densification

Standard 3DGS adaptively clones, splits, and prunes Gaussians based on reconstruction needs. Here, the same philosophy is applied to the **motion field**, not only geometry:

```text
3DGS densification: refine geometry where image reconstruction requires it.
Proposed method: refine motion where coherent transport fails.
```

The novelty is therefore not merely adding more Gaussians, but adapting the resolution of the motion representation to the complexity of local dynamics.

---

## 12. Dynamic Static/Dynamic Decomposition

A useful extension is to learn a dynamic probability per Gaussian or per grid cell:

```text
p_dyn_i in [0,1]
```

Then the velocity can be gated:

```text
v_i(t) = p_dyn_i * v_dynamic(mu_i,t) + (1 - p_dyn_i) * v_static(mu_i,t)
```

with:

```text
v_static approx 0
```

Regularize `p_dyn` with a binary/entropy loss:

```text
L_binary = sum_i p_dyn_i (1 - p_dyn_i)
```

and a spatial smoothness loss:

```text
L_dyn_smooth = sum_{(i,j)} w_ij |p_dyn_i - p_dyn_j|
```

This directly supports the goal of separating fixed and moving parts before joint estimation.

---

## 13. Articulated Object Prior

For an articulated object with a fixed part and a moving part, the learned velocity field should eventually collapse to a low-dimensional joint model.

### 13.1 Revolute joint velocity

For a revolute joint with axis direction `e` and point `a` on the axis:

```text
v_rev(x,t) = q_dot(t) * e x (x - a)
```

where `q(t)` is joint angle.

### 13.2 Prismatic joint velocity

For a prismatic joint with axis direction `e`:

```text
v_pris(x,t) = q_dot(t) * e
```

### 13.3 Joint explainability loss

After an initial velocity field is learned, fit candidate joint models to the dynamic region and penalize residuals:

```text
L_joint = min over joint params sum_i || v_theta(mu_i,t) - v_joint(mu_i,t; params) ||_2^2
```

This gives a natural bridge from dynamic 3DGS to joint estimation:

1. Learn a flexible velocity field.
2. Segment coherent motion regions.
3. Fit revolute/prismatic candidate joints.
4. Feed the joint model back into training as a stronger prior.
5. Refine ambiguous points by checking which joint model best explains them.

This matches the desired iterative loop: motion field -> joint hypothesis -> relabel uncertain regions -> motion field refinement.

---

## 14. Full Objective

A practical total loss is:

```text
L_total =
    lambda_rgb    L_rgb
  + lambda_ssim   L_ssim
  + lambda_depth  L_depth
  + lambda_mask   L_mask
  + lambda_roll   L_rollout
  + lambda_cycle  L_cycle
  + lambda_comp   L_comp
  + lambda_rigid  L_rigid
  + lambda_se3    L_se3
  + lambda_static L_static
  + lambda_acc    L_acc
  + lambda_bin    L_binary
  + lambda_dyn    L_dyn_smooth
  + lambda_joint  L_joint
```

For a first implementation, do not activate all terms at once. Start with:

```text
L_total_mvp = L_rgb + L_ssim + L_depth + L_rigid + L_static + L_acc
```

Then add rollout/cycle/composition losses once basic training works.

---


## 14A. Loss-to-Parameter Routing and Gradient Control

A central implementation detail is that losses must not be allowed to update every parameter indiscriminately. The method should explicitly route gradients so that motion-related losses train the Eulerian velocity field, static/dynamic losses train the dynamic labels, and grid-adaptation signals modify only the adaptive grid structure.

This separation avoids degenerate solutions in which the canonical Gaussians absorb motion, the dynamic mask labels everything as moving, DINO features are treated as trainable correspondences, or the grid splits everywhere to reduce local residuals.

### 14A.1 Parameter groups

```text
Theta_G    = canonical Gaussian parameters: centers, rotations, scales, opacities, colors/SH
Theta_v    = Eulerian velocity field: MLP weights, velocity-grid feature codes, temporal embeddings
Theta_dyn  = static/dynamic logits or probabilities
Theta_grid = adaptive grid structure: active cells, levels, gates, split/merge state, Gaussian-to-cell assignments
```

During the main motion-learning phase, the canonical geometry should be frozen or strongly regularized. The Gaussian centers at time `t` are obtained by integrating the velocity field, so image and motion losses should primarily update `Theta_v`.

### 14A.2 Routing table

| Loss / mechanism | Updates | Should not update | Detached quantities / notes |
|---|---|---|---|
| RGB, SSIM, depth rendering | `Theta_v` during motion training; optional late fine-tune of colors/opacities | `Theta_grid` topology; usually `Theta_G` geometry | Canonical geometry frozen or strongly regularized. |
| Rollout rendering | `Theta_v` | `Theta_grid` topology; `Theta_G` | Backpropagate through the integration chain, but not through split/merge decisions. |
| Cycle consistency | `Theta_v` | `Theta_G`; `Theta_grid` | Target position is stop-gradient: `||mu_hat_a - sg(mu_a)||`. |
| Speed / acceleration / temporal coherence | `Theta_v` | `Theta_G`; `Theta_grid` | Smooth the velocity field, not the canonical scene. |
| DINO-weighted rigidity | `Theta_v` | DINO backbone/features; `Theta_G` | DINO weights and canonical pairwise distances are stop-gradient. |
| Local SE(3) coherence | `Theta_v` | `Theta_G`; fitted SE(3) variables | Best-fit `R_c,u_c` are estimated and detached; residual trains velocity field. |
| Static velocity penalty | `Theta_v` | `Theta_dyn` | Use `sg(1-s_i)` so the label cannot escape the penalty. |
| Dynamic sparsity / binarization | `Theta_dyn` | `Theta_v`; `Theta_G`; `Theta_grid` | Motion-derived pseudo-labels are detached if used. |
| Grid incoherence score | `Theta_grid` only | `Theta_v`; `Theta_G`; `Theta_dyn` | Velocity, rigidity, SE(3), DINO, and image residuals are diagnostics and are detached. |
| Split / merge / prune / freeze | `Theta_grid` only | `Theta_v`; `Theta_G` | Topology/capacity control, not a direct motion loss. |
| DINO feature extraction | None | All trainable model groups | DINO is frozen and only provides semantic affinity hints. |

### 14A.3 Practical stop-gradient rules

- Detach DINO features and DINO affinity weights before using them inside rigidity or grid-refinement losses.
- Detach canonical pairwise distances in rigidity losses so the canonical shape cannot shrink or stretch to reduce error.
- Detach fitted local SE(3) transforms if they are computed with Kabsch/Procrustes and used only as consistency targets.
- Detach grid incoherence scores before split/merge decisions so grid refinement does not backpropagate into the velocity field.
- Detach static/dynamic probabilities inside the static velocity penalty, and train the dynamic branch separately with sparsity, binarization, and detached pseudo-labels.

The core rule is: rendering, rollout, rigidity, speed, temporal, cycle, and SE(3) losses train the velocity field; dynamic-label losses train the static/dynamic logits; grid-adaptation losses or policies update only the grid structure.

## 14B. Incremental Loss Activation and Ablation Schedule

The full objective should not be activated at full strength from the beginning. If all constraints are applied immediately, the optimization may become too constrained before the velocity field has learned any useful motion. The method should therefore be implemented as an incremental ablation schedule, adding one contribution at a time and measuring its effect.

### 14B.1 Initial objective

The first complete motion-training objective should be:

```text
L = L_rgb + L_ssim + L_depth
    + lambda_rollout L_rollout
    + lambda_static L_static
    + lambda_rigid L_rigid-DINO
```

This initial objective gives the model frame-level supervision through RGB/SSIM/depth, forces motion to be generated by rollout through the velocity field, anchors static regions, and introduces the first piecewise-rigid prior through DINO-weighted rigidity.

Only after this version is stable should the additional losses be enabled:

```text
Then add: L_cycle
Then add: L_SE(3)
Finally enable: adaptive grid split/merge
```

### 14B.2 Recommended step-by-step implementation

1. Train the canonical 3DGS reconstruction from the initial frame or a short canonical window.
2. Freeze or strongly regularize canonical geometry and appearance so motion must be explained by the velocity field.
3. Train a coarse occupied-space Eulerian velocity field using rollout rendering supervision.
4. Add static anchoring so background and static object parts keep near-zero velocity.
5. Add DINO-weighted rigidity so spatially close and semantically similar Gaussians preserve pairwise distances.
6. Add cycle consistency to reduce drift and preserve Gaussian identity over forward/backward transport.
7. Add local SE(3) explainability so each coherent cell is encouraged to behave as a rigid body.
8. Enable adaptive grid refinement using detached incoherence scores for split, merge, prune, and freeze decisions.
9. Optionally fine-tune selected Gaussian parameters and the velocity field jointly for rendering quality, with strong regularization to the canonical reconstruction.

### 14B.3 Ablation stages and expected effects

| Stage | Active additions | Trainable parameters | Expected effect / diagnostic |
|---|---|---|---|
| A0 Canonical 3DGS | Static 3DGS rendering only | `Theta_G` | Provides the canonical Gaussian support and occupancy domain. |
| A1 Coarse velocity rollout | RGB + SSIM + depth + `L_rollout` | `Theta_v` | Shows whether the Eulerian field can transport Gaussians to match later frames. |
| A2 Static anchoring | + `L_static` | `Theta_v`; optionally `Theta_dyn` separately | Reduces fake global/background motion; improves static/dynamic separation. |
| A3 DINO rigidity | + `L_rigid-DINO` | `Theta_v` | Improves coherent part motion and reduces elastic deformation. |
| A4 Cycle consistency | + `L_cycle` | `Theta_v` | Reduces drift and improves forward/backward trajectory identity. |
| A5 Local SE(3) | + `L_SE(3)` | `Theta_v` | Makes cells locally rigid and produces a useful signal for articulated boundaries. |
| A6 Adaptive grid | + split/merge/prune/freeze policy | `Theta_grid` only | Increases capacity only where motion incoherence persists. |
| A7 Final fine-tune | Optional joint fine-tuning | `Theta_v` plus restricted `Theta_G` | Improves visual quality without destroying the learned motion model. |

### 14B.4 Why this schedule matters

RGB, SSIM, and depth tell the model what each frame should look like. The remaining losses tell the model what kind of motion is allowed. Therefore, the velocity field should first learn a coarse explanation of the observed motion, and only afterwards should stronger physical and structural constraints be introduced.

The adaptive grid should come last because the split/merge mechanism should respond to persistent, meaningful motion incoherence. If it is enabled too early, the grid may simply overfit early photometric errors, occlusions, or untrained velocity noise.

## 15. Training Schedule

### Stage 0: Data preparation

- Fuse multiview RGB-D point clouds per frame.
- Estimate or load camera intrinsics and poses.
- Build masks if object/background segmentation is available.
- Train or load canonical 3DGS at `t_0`.

### Stage 1: Canonical Gaussian stabilization

- Train static Gaussians on first frame/window.
- Freeze or weakly regularize color, opacity, and scale.
- Build Gaussian occupancy support.

### Stage 2: Coarse velocity field

- Initialize coarse sparse grid over occupied support.
- Train velocity field with short time windows.
- Use center-only Gaussian transport initially.
- Losses: RGB, SSIM, depth, static prior, acceleration smoothness.

### Stage 3: Rollout consistency

- Increase temporal window length.
- Train by rolling forward frame by frame.
- Add forward/backward cycle and composition consistency.

### Stage 4: Local rigidity and SE(3) coherence

- Build kNN graph over canonical Gaussians.
- Add rigidity loss.
- Fit local SE(3) per grid cell and penalize incoherent motion.

### Stage 5: Adaptive grid refinement

- Compute cell-level motion incoherence scores.
- Split high-error/high-incoherence cells.
- Merge or freeze low-error coherent cells.
- Continue training after refinement.

### Stage 6: Optional articulated prior

- Estimate dynamic/static labels.
- Fit revolute/prismatic candidates.
- Add joint explainability loss.
- Iterate motion field and joint refinement.

---

## 16. Minimal Viable Prototype

A robust first prototype should avoid too many moving parts.

### MVP representation

- Fixed canonical 3DGS.
- Sparse occupied voxel grid.
- Small MLP predicting only linear velocity.
- Euler integration.
- Fixed covariance and appearance.

### MVP losses

```text
L = L_rgb + 0.2 L_ssim + lambda_depth L_depth + lambda_rigid L_rigid + lambda_acc L_acc
```

### MVP adaptive mechanism

Every `N` iterations:

1. Compute average photometric/depth residual per cell.
2. Compute velocity variance per cell.
3. Split cells whose combined score is high.
4. Freeze cells that are static and low-error.

### MVP diagnostics

Render and log:

- RGB reconstruction video;
- depth reconstruction video;
- Gaussian trajectories;
- velocity vectors;
- per-cell split scores;
- static/dynamic probability map;
- local rigidity error heatmap;
- estimated motion clusters.

---

## 17. Pseudocode

```python
# Inputs:
#   canonical_gaussians G0
#   video frames I[c,k], D[c,k]
#   camera intrinsics K[c], poses T[c,k]
#   occupied grid built from G0

initialize velocity_grid over occupied Gaussian support
initialize velocity_mlp

for iteration in range(num_iters):
    frame_window = sample_temporal_window()
    cameras = sample_cameras()

    G = copy(G0)
    rollout_states = []

    for k in frame_window:
        # Query velocity at current Gaussian centers
        v = velocity_mlp(G.positions, time=k, grid=velocity_grid)

        # Advect Gaussian centers
        G.positions = G.positions + dt * v

        # Optional: update Gaussian rotations
        # omega = angular_mlp(...)
        # G.rotations = exp(dt * omega) @ G.rotations

        rollout_states.append(G.copy())

    loss = 0

    for Gk, k in zip(rollout_states, frame_window):
        for c in cameras:
            pred_rgb, pred_depth, pred_alpha = render(Gk, K[c], T[c,k])
            loss += rgb_loss(pred_rgb, I[c,k])
            loss += ssim_loss(pred_rgb, I[c,k])
            loss += depth_loss(pred_depth, D[c,k])

    loss += rigidity_loss(rollout_states, knn_graph)
    loss += acceleration_loss(rollout_states, velocity_mlp)
    loss += static_prior_loss(G0, velocity_mlp)
    loss += cycle_loss_if_enabled(...)
    loss += composition_loss_if_enabled(...)

    loss.backward()
    optimizer.step()
    optimizer.zero_grad()

    if iteration % refinement_interval == 0:
        scores = compute_cell_scores(
            photometric_residuals,
            depth_residuals,
            velocity_variance,
            rigidity_error,
            se3_fit_error,
        )
        split_high_score_cells(scores)
        merge_coherent_neighbor_cells(scores)
        freeze_static_low_error_cells(scores)
```

---

## 18. Pros Compared to Per-Gaussian Motion MLP

### 18.1 Better inductive bias

A shared velocity field naturally encourages neighboring Gaussians to move coherently.

### 18.2 Continuous-time trajectories

Velocity and acceleration are available, which is useful for physics and joint estimation.

### 18.3 Temporal composition

The field can be trained so that long-horizon motion is the composition of short-horizon motion.

### 18.4 Compactness

Motion is represented by grid/MLP parameters rather than by storing a separate trajectory for every Gaussian.

### 18.5 Adaptive expressivity

The model can remain simple in rigid regions and become expressive near motion boundaries.

### 18.6 Better connection to articulation

Revolute and prismatic joint velocities can be directly compared to the learned field.

---

## 19. Cons and Risks

### 19.1 Harder optimization

An ODE/rollout model is harder to train than direct canonical-to-time deformation.

### 19.2 Photometric ambiguity

RGB/SSIM alone cannot guarantee correct 3D motion. Depth, masks, feature consistency, or point-cloud supervision are strongly recommended.

### 19.3 Occlusion drift

Occluded Gaussians receive weak direct supervision. Cycle, rigidity, static priors, and temporal windows are needed.

### 19.4 Smooth fields blur motion boundaries

A continuous velocity field can average motion across articulated boundaries. Adaptive cell splitting or motion-component gating is necessary.

### 19.5 Topology limitations

A smooth ODE flow does not naturally create new surfaces. Newly visible regions may require Gaussian densification or activation of hidden/low-opacity Gaussians.

### 19.6 Covariance rotation complexity

Correctly transporting anisotropic Gaussian covariances is harder than moving centers. Start center-only, then add angular velocity.

---

## 20. Failure Modes and Mitigations

| Failure mode | Cause | Mitigation |
|---|---|---|
| Good RGB, wrong motion | Photometric ambiguity | Add depth, masks, feature/point-cloud consistency, rigidity |
| Rubber-like deformation | Field too free | Add local rigidity and SE(3) explainability |
| Static points become dynamic | Motion bleeding near boundaries | Static anchors, dynamic gating, sharper cell splitting |
| Dynamic points become static | Motion underfitting | Increase local grid resolution, reduce static penalty locally |
| Occluded points drift | No visibility supervision | Cycle loss, trajectory smoothness, long-lived anchors |
| Motion boundary is blurred | Smooth MLP interpolation | Adaptive grid split, mixture-of-motion components |
| Training unstable | Too many losses too early | Stage losses progressively |
| New surfaces missing | Initial Gaussians do not cover them | Allow geometry densification after motion stabilizes |

---

## 21. Recommended Research Claim

The strongest formulation is not:

```text
We use an MLP to deform 3D Gaussians over time.
```

That is too close to existing dynamic Gaussian methods.

The stronger claim is:

```text
We learn a continuous Eulerian velocity field that transports canonical 3D Gaussians through time. The field is defined only on the occupied Gaussian support and is adaptively refined where local Gaussian motion becomes incoherent. This yields a compact, transport-consistent motion representation that supports rendering, tracking, static/dynamic segmentation, and articulated joint estimation.
```

The key contributions would be:

1. **Transport-based dynamic 3DGS:** Gaussian motion is produced by integrating an Eulerian velocity field, not by direct timestamp deformation.
2. **Motion-adaptive velocity grid:** the grid is refined only where local motion coherence fails.
3. **Rigidity-aware supervision:** local distance preservation and SE(3) explainability make the learned motion useful for articulated objects.
4. **Bridge to joint estimation:** the learned velocity field can be projected onto revolute/prismatic models and used to refine static/dynamic labels.

---

## 22. Relation to Prior Work

- **3D Gaussian Splatting** introduced a real-time radiance-field representation based on anisotropic Gaussians with interleaved optimization and density control.
- **Dynamic 3D Gaussians** models dynamic scenes with persistent Gaussians that move and rotate over time, using local rigidity to obtain dense tracking from rendering supervision.
- **4D Gaussian Splatting** combines 3D Gaussians with 4D neural voxels and a lightweight MLP to predict Gaussian deformations at novel timestamps for efficient dynamic rendering.
- **Deformable 3D Gaussians** learns canonical Gaussians plus a deformation field for monocular dynamic scene reconstruction.
- **EulerFlow / Neural Eulerian Scene Flow Fields** formulates scene flow as a continuous space-time ODE, directly motivating the proposed velocity-field transport formulation.
- **FlowNet3D and FLOT** are important point-cloud scene-flow baselines, but they are closer to per-point/correspondence-based displacement estimation than to dynamic Gaussian transport.

The proposed method sits between these lines of work: it uses the rendering power of dynamic 3DGS, the transport formulation of Eulerian scene flow, and the adaptive refinement philosophy of Gaussian densification.

---

## 23. References

1. Kerbl, B., Kopanas, G., Leimkuehler, T., Drettakis, G. **3D Gaussian Splatting for Real-Time Radiance Field Rendering.** SIGGRAPH 2023. arXiv:2308.04079. https://arxiv.org/abs/2308.04079
2. Luiten, J., Kopanas, G., Leibe, B., Ramanan, D. **Dynamic 3D Gaussians: Tracking by Persistent Dynamic View Synthesis.** arXiv:2308.09713. https://arxiv.org/abs/2308.09713
3. Wu, G., Yi, T., Fang, J., Xie, L., Zhang, X., Wei, W., Liu, W., Tian, Q., Wang, X. **4D Gaussian Splatting for Real-Time Dynamic Scene Rendering.** arXiv:2310.08528. https://arxiv.org/abs/2310.08528
4. Yang, Z., Gao, X., Zhou, W., Jiao, S., Zhang, Y., Jin, X. **Deformable 3D Gaussians for High-Fidelity Monocular Dynamic Scene Reconstruction.** CVPR 2024. arXiv:2309.13101. https://arxiv.org/abs/2309.13101
5. Vedder, K., Peri, N., Chodosh, N., Khatri, I., Eaton, E., Jayaraman, D., Ramanan, D. **Neural Eulerian Scene Flow Fields.** ICLR 2025. arXiv:2410.02031. https://arxiv.org/abs/2410.02031
6. Li, X., Kaesemodel Pontes, J., Lucey, S. **Neural Scene Flow Prior.** arXiv:2111.01253. https://arxiv.org/abs/2111.01253
7. Liu, X., Qi, C. R., Guibas, L. J. **FlowNet3D: Learning Scene Flow in 3D Point Clouds.** CVPR 2019. arXiv:1806.01411. https://arxiv.org/abs/1806.01411
8. Puy, G., Boulch, A., Marlet, R. **FLOT: Scene Flow on Point Clouds Guided by Optimal Transport.** ECCV 2020. arXiv:2007.11142. https://arxiv.org/abs/2007.11142

---

## 24. Final Takeaway

The key is not simply to learn a dynamic MLP. The key is to learn a **transport-consistent Eulerian velocity field** over the occupied Gaussian support. Gaussians should be moved by integrating this field, and the field should be refined only where coherent local motion is insufficient. This gives a representation that can still render well, but is much more useful for understanding motion, separating parts, and estimating articulated joints.
