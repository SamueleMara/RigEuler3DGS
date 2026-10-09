# Eulerian Velocity-Gradient Rigidity — v6

> **Historical patch note.** References to retained Lagrangian branches or old
> command aliases no longer describe the executable repository. See
> [`PIPELINE.md`](PIPELINE.md) for the current fixed-curl-only implementation.

## Scope

This patch replaces the Eulerian trainable rigid-basis formulation with a continuum-kinematics formulation centered on the **cell-centre Eulerian velocity field**.

The public command

```text
--phase1_motion_representation eulerian_curl_grid
```

now routes to

```text
eulerian_curl_strain
```

The corrected Eulerian Phase 1 **does not optimize** angular velocity, translational twist, joint axis, rotation center, screw pitch, or any per-Gaussian velocity. Those quantities are either absent from Phase 1 or derived from the learned Eulerian velocity field.

The Lagrangian shared-SE(3) branch is intentionally left available and unchanged by this Eulerian rewrite.

---

# 1. Source kinematics used for the rewrite

The implementation follows the local first-order continuum expansion described in *Teoria 3. Cinematica v1.3* (pp. 8–13):

\[
\mathbf v(\mathbf X_c+\delta\mathbf X)
\simeq
\mathbf v_c + G_c\,\delta\mathbf X,
\qquad
G_c = \nabla \mathbf v_c.
\]

The local velocity gradient is decomposed as

\[
G_c = D_c + \Omega_c,
\]

with

\[
D_c = \frac12(G_c+G_c^T),
\qquad
\Omega_c = \frac12(G_c-G_c^T).
\]

`D` is the deformation-rate tensor.  A locally rigid motion has

\[
D_c=0.
\]

The antisymmetric part represents rigid rotation.  Angular velocity is derived as

\[
\boldsymbol\omega_c=\frac12\nabla\times\mathbf v_c.
\]

**The curl magnitude is not minimized.**  A rigid revolute motion has non-zero curl.  The implementation instead minimizes deformation `D` and spatial disagreement of the **derived** angular velocity inside likely-same-part material regions.

---

# 2. Trainable Eulerian state

## 2.1 Motion

The only trainable motion state is the temporal vector potential

\[
A(t)=\sum_m B_m(t) A_m,
\]

where `B_m(t)` are cubic B-spline temporal basis functions.

Physical cell-centre velocity is

\[
\boxed{\mathbf v_c(t)=\operatorname{curl}_h A(t)}.
\]

The discrete construction preserves

\[
\operatorname{div}_h(\operatorname{curl}_h A)=0
\]

up to numerical precision.

There are no trainable `omega`, `b`, `X_axis`, or rotation-center parameters in the Eulerian Phase 1 optimizer.

## 2.2 Material assignment

Every retained foreground Gaussian owns persistent trainable probabilities over

\[
\{\text{static},\text{moving part 1},\ldots,\text{moving part K}\}.
\]

These are material variables shared by all frames.  They are initialized from the continuous gradient-motion evidence and optimized contextually.

The motion-physics loss uses **detached** assignment probabilities, so the motion field cannot reduce deformation by simply changing labels.  A separate assignment-only kinematic loss uses **detached motion descriptors**, so assignments can change without corrupting the velocity field.

---

# 3. Fixed Eulerian world grid

The grid remains fixed in world coordinates.  Gaussian centers are material tracers.

For an interval `k -> k+1`, source cell IDs are

\[
c_i^k=\operatorname{cell}(\mathbf x_i^k).
\]

Every Gaussian in a source cell receives the exact same cell velocity

\[
\mathbf v_i^k=\mathbf v_{c_i^k}(t_{mid}).
\]

Transport is

\[
\mathbf x_i^{k+1}=\mathbf x_i^k+\Delta t_k\mathbf v_{c_i^k}(t_{mid}).
\]

Gaussian positions are never optimizer parameters.

---

# 4. Observed multiview optical flow -> direct observed 3-D cell velocity

The v5 observed-flow inversion is retained.

For every visible Gaussian `i`, source camera `j`, and source cell `c`, the **observed optical-flow vector** supplies

\[
J_{\pi,ij}\mathbf v_c
=
\frac{\mathbf f^{obs}_{ij}}{\Delta t}.
\]

All Gaussian-camera observations sharing the same Eulerian source cell are accumulated into one weighted 3-D least-squares problem:

\[
H_c=\sum_{ij}w_{ij}J_{ij}^TJ_{ij},
\qquad
\mathbf g_c=\sum_{ij}w_{ij}J_{ij}^T\frac{\mathbf f^{obs}_{ij}}{\Delta t},
\]

\[
(H_c+\lambda I)\mathbf v_c^{obs}=\mathbf g_c.
\]

The displacement vector is always optical flow.  Forward/backward consistency and DINO cosine only weight the equations.

The compact `3x3` systems are solved/conditioned on CPU in float64 to avoid CUDA batched-eigensolver workspace OOM; only the compact results return to GPU.

The direct cell losses are

\[
L_{obs,v}=\operatorname{mean}_w\|\mathbf v_c-\mathbf v_c^{obs}\|^2
\]

and, for cells with significant observed speed,

\[
L_{obs,dir}=\operatorname{mean}_w[1-\cos(\mathbf v_c,\mathbf v_c^{obs})].
\]

Defaults:

```text
lambda_observed_flow_velocity  = 20
lambda_observed_flow_direction = 10
```

The original dense observed-flow reprojection endpoint loss also remains active.

---

# 5. Material local velocity-gradient tensor

A raw Cartesian finite difference through empty grid cells is **not** used for rigidity.

For each occupied cell `c`, a material-neighbour graph is built from nearby occupied cells.  The local gradient is fitted through the Taylor relation

\[
\mathbf v_d-\mathbf v_c
\simeq
G_c(\mathbf X_d-\mathbf X_c).
\]

The implemented weighted solve is

\[
G_c
=
\arg\min_G
\sum_{d\in\mathcal N(c)}
 a_{cd}
\|\Delta\mathbf v_{cd}-G\Delta\mathbf X_{cd}\|^2
+
\lambda_G\|G\|_F^2.
\]

With

\[
M_c=\sum_d a_{cd}\Delta X\Delta X^T,
\qquad
C_c=\sum_d a_{cd}\Delta v\Delta X^T,
\]

the code solves

\[
G_c=C_c(M_c+\lambda I)^{-1}.
\]

Default neighborhood settings:

```text
curl_material_neighbors                    = 12
curl_material_neighbor_radius_cells        = 3.5
eulerian_velocity_gradient_min_neighbors   = 3
eulerian_velocity_gradient_ridge           = 1e-4
```

The ridge gives the minimum-norm extension along weakly sampled directions of surface-like Gaussian support.

---

# 6. Exact tensor decomposition implemented

Code convention is

\[
G_{ij}=\frac{\partial v_i}{\partial x_j}.
\]

The implementation computes

\[
D=\frac12(G+G^T),
\qquad
\Omega=\frac12(G-G^T),
\]

and

\[
\omega_x=\frac12(G_{zy}-G_{yz}),
\]

\[
\omega_y=\frac12(G_{xz}-G_{zx}),
\]

\[
\omega_z=\frac12(G_{yx}-G_{xy}).
\]

For an exact rigid field

\[
\mathbf v=\mathbf b+\boldsymbol\omega\times\mathbf X,
\]

this gives `D=0` and recovers the correct `omega` independently of any chosen axis point.

---

# 7. New canonical Eulerian rigidity loss

The principal rigidity loss is now

\[
\boxed{L_D=\operatorname{mean}\|D_c\|_F^2.}
\]

The implementation converts the gradient scale `[1/s]` to a local relative-velocity scale using the median material-neighbour distance before normalization by `eulerian_velocity_scale`.

Default:

```text
lambda_eulerian_strain_rate = 100
```

This suppresses extension, compression, and shear while preserving both rigid translation and rigid rotation.

## 7.1 Derived angular-velocity coherence

`omega_c` is not trainable.  It is derived from the antisymmetric component / half-curl.

Likely-same-part neighboring cells minimize

\[
L_{\omega}
=
\sum_{(c,d)}a_{cd}
\|\omega_c-\omega_d\|^2
\]

with spatial scaling by the cell-pair distance.

Default:

```text
lambda_eulerian_omega_coherence = 25
```

Again: **curl magnitude is never minimized.**

## 7.2 Static speed

The static probability is used only as a detached soft prior:

\[
L_{static}=\sum_c P_c(static)\|v_c\|^2.
\]

Default:

```text
lambda_eulerian_static_speed_strain = 5
```

## 7.3 Local rigid residual diagnostic

The code also reports

\[
\Delta v_{cd}
-
\frac{\omega_c+\omega_d}{2}\times\Delta X_{cd}.
\]

It is redundant with `D=0`, so its default optimizer weight is zero:

```text
lambda_eulerian_local_rigid_residual = 0
```

---

# 8. Same-part material weighting and boundary behavior

For cell part probabilities `P_c(r)`, same-part probability is

\[
s_{cd}=\sum_r P_c(r)P_d(r).
\]

The motion-physics edge weight is

\[
a_{cd}=a_{floor}+(1-a_{floor})s_{cd}.
\]

The probabilities are detached for the motion loss.  The floor anneals from the existing early value to the late value, keeping the continuum graph connected while the initial static/dynamic labels are still uncertain and allowing real boundaries to sharpen later.

---

# 9. Assignment-only contextual correction

The assignment loss receives **detached** kinematics.

## 9.1 False-static correction from observed flow

For cells with a valid observed 3-D velocity, nonzero observed speed penalizes the static hypothesis:

\[
L_{static,obs}
\propto
P_c(static)\|v_c^{obs}\|^2.
\]

Thus an occlusion-induced false static can become moving because the observed multiview motion says that the occupied cell is not stationary.

## 9.2 Kinematic boundary evidence

For neighboring cells, the assignment sees detached incompatibility based on

- local rigid residual; and
- derived `omega` disagreement.

High kinematic disagreement makes a same-part labeling expensive and allows a material-part boundary to emerge.

Default:

```text
lambda_eulerian_assignment_kinematic = 2
```

Existing asymmetric initialization prior, spatial material coherence, and entropy hardening remain.

---

# 10. Gaussian covariance rotation is derived from the SAME Eulerian field

The old corrected branch translated centers using the CurlGrid but rotated anisotropic Gaussians using a separate learned rigid basis.  That inconsistency is removed.

For each source cell the code derives

\[
\omega_c=\frac12\nabla\times v_c
\]

from the local velocity-gradient fit, gathers that `omega_c` for the Gaussian, and integrates the world angular velocity into its quaternion over the real `dt`.

Therefore:

```text
translation <- v_c = curl_h(A)
rotation    <- omega_c = 1/2 curl(v_c)
```

Both now come from one Eulerian field.

---

# 11. Temporal training schedule

The existing overlapping-window schedule is retained.  With the current default/USB command:

```text
phase1_window_frames  = 6
phase1_window_stride  = 5
free_field_iters      = 25
```

and 60 frames, there are 13 temporal windows.

The existing stage budget remains:

```text
observation x1
bootstrap   x1
assignment  x1
joint       x3
```

for `6 * 13 * 25 = 1950` updates.

Unlike the old trainable-basis Eulerian branch, `D=0` continuum rigidity is active in **every stage in which motion is trainable**, including the observation stage.  There is no longer a zero-basis trap requiring an unconstrained motion warmup.

Window order alternates forward/reverse between global passes.

---

# 12. One motion update, exactly

For every target frame in the current window:

1. Replay current Eulerian geometry from canonical frame to the source frame under `no_grad`.
2. Build source-cell IDs and occupied material cells.
3. Build occupied-cell material neighbor graph.
4. Aggregate persistent Gaussian part probabilities to current cells.
5. Query `A(t_mid)`, compute `v_cells = curl_h(A)`.
6. Gather one `v_c` per source Gaussian.
7. Transport centers with real `dt`.
8. Fit local `G_c` from occupied material neighbors.
9. Decompose `G_c -> D_c, Omega_c, omega_c`.
10. Integrate Gaussian quaternion using derived `omega_c`.
11. Compute `L_D`, derived-omega coherence, static speed, optional local residual.
12. Recompute/use cached observed optical-flow-derived `v_c_obs`.
13. Apply direct observed 3-D magnitude and direction losses to `v_c`.
14. If assignment is trainable, apply detached kinematic assignment loss.
15. Render every synchronized target camera.
16. Compute foreground RGB/SSIM, depth, DINO feature-field loss, and dense observed 2-D flow endpoint loss.
17. Average cameras and frames in the window.
18. Backpropagate.

After the window contributions:

19. Apply assignment prior/spatial/entropy if assignments are trainable.
20. Apply temporal B-spline acceleration regularization to `A` if motion is trainable.
21. Clip gradients.
22. Step `A` optimizer and/or assignment optimizer depending on the stage.
23. Run additional `A`-only projection steps using exactly the same observed-velocity, `D`, derived-omega-coherence and static-speed objectives.  No rigid-basis or axis fit occurs.

---

# 13. Current total Eulerian motion objective

Conceptually:

\[
L_{motion}
=
\lambda_{obs,v}L_{obs,v}
+
\lambda_{obs,dir}L_{obs,dir}
+
\lambda_D L_D
+
\lambda_{\omega}L_{\omega-coh}
+
\lambda_{static}L_{static}
+
\lambda_{render}L_{RGBD/DINO/2D-flow}
+
\lambda_A L_{A-temporal}.
\]

Current defaults:

```text
lambda_observed_flow_velocity         20
lambda_observed_flow_direction        10
lambda_eulerian_strain_rate          100
lambda_eulerian_omega_coherence       25
lambda_eulerian_static_speed_strain    5
lambda_eulerian_local_rigid_residual   0
lambda_curl_temporal_acceleration      0.10
```

There is no Eulerian `lambda_basis`, trained twist, or trained axis in this path.

---

# 14. New debug contract

The corrected Eulerian run writes:

```text
dynamic_debug/
  phase1_eulerian_velocity_gradient_contract.json
  phase1_eulerian_velocity_gradient_audit.pt
  phase1_eulerian_velocity_gradient_summary.json
  phase1_eulerian_curl_grid_final.pt
  phase1_material_assignment_final.pt
  phase1_gaussian_tracks_final.pt
  phase1_observed_3d_flow_cells_*.pt
  phase1_eulerian_curl_strain_transported.html
  phase1_part_dino_flow_eulerian_curl_strain_final_*.png/.pt/.json
```

The per-frame velocity-gradient audit saves:

- occupied cell IDs;
- current cell part probabilities;
- `v_c`;
- fitted `G_c`;
- deformation tensor `D_c`;
- antisymmetric tensor `Omega_c`;
- derived `omega_c`;
- `curl_vector = 2 omega_c`;
- `trace(G)`;
- `||D||_F`;
- valid-gradient mask;
- material-cell graph and same-part weights;
- strain loss;
- omega-coherence loss;
- static-speed loss;
- local rigid residual diagnostic;
- discrete continuity residual;
- outside-grid tracer count.

Known incompatible stale Eulerian rigid-basis/legacy-flow debug products are deleted at trainer start if a model output directory is reused.

The existing DINO/flow/K-part camera diagnostic remains.  DINO part clustering/prototype classification is still **diagnostic only**; DINO does not create the motion vector and is not silently promoted to a K-part training target in this patch.

---

# 15. File-by-file modifications

## `arguments/__init__.py`

- Added public `eulerian_curl_strain` representation.
- Public `eulerian_curl_grid` is documented/routed to strain formulation.
- Added local gradient fit controls.
- Added `D`, derived-omega coherence, static-speed and assignment-kinematic weights.
- Left rigid-basis parameters because the corrected Lagrangian branch still uses them; Eulerian strain mode does not.

## `scene/gaussian_model.py`

- `eulerian_curl_grid -> eulerian_curl_strain` alias.
- Eulerian strain initialization creates only:
  - `MaterialAssignmentField`;
  - temporal `FixedCurlGrid`.
- No `TemporalRigidBasis` is instantiated for Eulerian strain.
- Motion optimizer owns only CurlGrid vector-potential parameters.
- Assignment optimizer owns only material logits.
- Dynamic checkpoint schema v15 stores CurlGrid + assignments + strain config, no Eulerian rigid basis.
- Runtime replay and exported dynamic states derive Gaussian angular velocity from the CurlGrid velocity gradient.

## `utils/dynamic_loss_utils.py`

Added:

- `_same_part_pair_weights`;
- `fit_local_velocity_gradient`;
- `decompose_velocity_gradient`;
- `eulerian_velocity_gradient_rigidity_loss`;
- `eulerian_assignment_from_kinematics_loss`.

## `utils/unified_dynamic_training.py`

- Added `train_eulerian_velocity_gradient_phase1`.
- Added `_eulerian_strain_replay_source`.
- Corrected Eulerian dispatch no longer goes through the trainable rigid-basis trainer.
- Direct observed 3-D cell velocity and direction losses retained.
- Added local continuum rigidity and assignment-only kinematic path.
- Gaussian orientation now comes from derived `omega_c`.
- Added new velocity-gradient contract/audit/summary outputs.
- A-only projection no longer fits/optimizes a common rigid basis.

## `utils/motion_correspondence.py`

Retains v5.1:

- observed optical-flow `J_pi v = f_obs/dt` inversion;
- DINO/cycle confidence weighting only;
- CPU float64 solve of compact `3x3` normal systems to avoid CUDA eigensolver OOM;
- source raster uses source orientation.

## `scene/fixed_curl_grid.py`

Retained temporal B-spline vector potential and exact discrete curl/divergence construction used by the new trainer.

## `scene/material_assignment_field.py`

Retained only because the Lagrangian shared-SE(3) comparison still needs it.  Corrected Eulerian strain mode does not instantiate or optimize it.

## `scene/lagrangian_frame_velocity.py`

Retained for explicit legacy ablation support only.

## `train.py`

- Corrected update-count handling recognizes explicit `eulerian_curl_strain`.
- Public `eulerian_curl_grid` keeps the same windowed corrected Phase-1 budget.

---

# 16. Verification performed on the release logic

`VERIFY_EULERIAN_VELOCITY_GRADIENT.py` checks:

1. rigid translation -> `D=0`, `omega=0`;
2. rigid rotation around an arbitrary unknown axis -> `D≈0`, derived `omega` is correct;
3. divergence-free shear -> `D` is non-zero, proving incompressibility alone is insufficient;
4. derived `omega` rotates Gaussian quaternion correctly;
5. `div_h(curl_h A)` stays at numerical zero and the `D` loss backpropagates into `A`;
6. public Eulerian alias routes to the strain formulation and no trainable Eulerian basis is present;
7. trainer contains observed `v_c` supervision + `D=sym(grad v)` and does not emit a rigid-basis output.

The assembled repository was also compiled recursively and imported from `train.py` downward with stubs only for unavailable external compiled CUDA/PLY dependencies.

A real USB/CUDA training run was **not** executed in this container.
