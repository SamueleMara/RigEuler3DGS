# Fixed-curl Eulerian Phase-1 pipeline

This repository now has one supported dynamic training path:

`eulerian_curl_strain` = a fixed finite-volume Eulerian grid whose only motion
parameters are temporal cubic B-spline controls of an edge vector potential
`A(t)`. Cell velocity is the discrete curl `v(t) = curl_h(A(t))`. Per-Gaussian
static/moving material logits are the only other dynamic parameters. Angular
velocity, rigid centers, joint axes, twists, and Gaussian trajectories are
derived quantities, never optimizer parameters.

## Code map

| File | Responsibility |
| --- | --- |
| `train.py` | Process entry point, static 3DGS loading/training, dynamic orchestration, final save. |
| `arguments/__init__.py` | Canonical 3DGS arguments plus the single fixed-curl Phase-1 configuration surface. |
| `scene/__init__.py` | Loads the canonical COLMAP scene and expands fixed cameras over the dynamic sequence. |
| `scene/dynamic_sequence_dataset.py` | Discovers RGB/depth/DINO/mask frames, clones calibrated cameras, and assigns physical timestamps. |
| `scene/gaussian_model.py` | Canonical Gaussian state, semantic compaction, dynamic state/checkpoint schema, replay, and optimizers. |
| `scene/fixed_curl_grid.py` | B-spline vector potential, finite-volume curl, cell lookup, material support, and exact discrete continuity diagnostic. |
| `scene/material_assignment_field.py` | Persistent static/moving logits and their initialization, prior, and checkpoint state. |
| `gaussian_renderer/__init__.py` | RGB/depth rendering, fixed-geometry projected-velocity feature splatting, and CUDA observed-flow normal-equation accumulation. |
| `utils/gradient_motion_evidence.py` | Full-scene canonical-vs-future gradient attribution used only to initialize material logits. |
| `utils/motion_correspondence.py` | Optical-flow/DINO correspondence cache, rasterized 3-D observed-velocity targets, and the projected-velocity optical-flow loss. |
| `utils/dynamic_loss_utils.py` | Material graphs, local velocity-gradient fit, strain/affine/coherence losses, kinematic assignment loss, and quaternion integration. |
| `utils/unified_dynamic_training.py` | Window schedule, fixed-curl optimization loop, deterministic final replay, audits, and exports. |

## Supplied command: effective execution

```bash
CUDA_LAUNCH_BLOCKING=1 python train.py \
  -s /home/samuelemara/Joint3DGS/Sapien/output/partnet_subset_fixed_render_linear/USB_100109 \
  -m /home/samuelemara/Joint3DGS/RigEuler3DGS/output/USB_100109_projected_velocity_flow_smoke \
  --start_ply /home/samuelemara/Joint3DGS/RigEuler3DGS/output/USB_100109/point_cloud/iteration_30000/point_cloud.ply \
  --dynamic_motion \
  --phase1_motion_representation eulerian_curl_strain \
  --phase1_only \
  --use_semantic_feature_field \
  --use_dino_feature_field \
  --dynamic_use_depth \
  --semantic_dynamic_threshold 0.2 \
  --free_field_iters 4 \
  --phase1_window_frames 2 \
  --phase1_window_stride 20 \
  --curl_grid_target_cells 4096 \
  --motion_lr 1e-3 \
  --observed_flow_target_refresh_iters 1 \
  --observed_flow_velocity_min_cameras 3 \
  --observed_flow_velocity_min_observations 6 \
  --observed_flow_velocity_max_reprojection_px 0.75 \
  --lambda_observed_flow_velocity 20 \
  --lambda_observed_flow_direction 10 \
  --lambda_motion_projected_velocity_flow 2.0 \
  --lambda_eulerian_strain_rate 100 \
  --lambda_eulerian_velocity_spatial_coherence 10 \
  --lambda_eulerian_speed_spatial_coherence 2 \
  --velocity_spatial_coherence_robust_delta 0.25 \
  --lambda_eulerian_spatial_gradient_coherence 50 \
  --lambda_eulerian_local_affine_coherence 50 \
  --lambda_eulerian_static_speed_strain 5 \
  --lambda_eulerian_assignment_kinematic 2 \
  --eulerian_static_gate_threshold 0.85 \
  --lambda_eulerian_hard_static_speed 250 \
  --lambda_assignment_dino_boundary 0.20 \
  --dino_boundary_rigidity_strength 1.0 \
  --disable_viewer
```

The supplied dataset currently contains 8 fixed cameras and 60 frames per
camera. It contains metric `depth_npy`, masks, and DINO targets. It does not
contain a `flow` directory or a timestamp file, so this run computes OpenCV DIS
forward/backward flow on demand and uses `frame_id / 30` seconds.

The start PLY exists at iteration 30,000. Because `static_train_iters` and
`start_ply_iteration` both default to 30,000, canonical static optimization is
skipped after loading and rebinding the optimizer to the PLY tensors.

`CUDA_LAUNCH_BLOCKING=1` makes CUDA launches synchronous for accurate failure
locations; it does not change the objective and normally slows the run.
`--disable_viewer` disables only the network viewer. W&B remains enabled by
default; use `--no_wandb` if this machine is not authenticated.

The dense optical-flow objective has one coefficient,
`lambda_motion_projected_velocity_flow`, and the command sets it explicitly.
There is no second trajectory/depth flow loss or outer flow multiplier.

### Exact update budget

`phase1_window_frames=2` and the requested stride `20` do not leave frames
unsupervised. The scheduler clamps the effective step to the window width, then
adds the final full-width window. For 60 frames this produces 31 windows and
covers every target frame 1 through 59.

The simplified schedule has five joint motion/material passes and one
assignment-only pass. Therefore:

```text
31 windows × (5 motion passes + 1 assignment pass) × 4 iterations = 744 updates
final checkpoint iteration = 30,000 + 744 = 30,744
```

Within a two-frame target window, one optimizer update accumulates both adjacent
frame intervals, and each interval averages all 8 synchronized cameras before
back-propagation.

## Implementation steps

1. **Parse and validate configuration.** `train.py` constructs the canonical
   model/optimization/renderer groups, adds the fixed-curl arguments, and calls
   `apply_dynamic_defaults`. Dynamic mode requires `--phase1_only`, semantic
   features, physical seconds, and exactly `eulerian_curl_strain`.

2. **Prepare the output and logging.** The model directory is created, the
   resolved arguments are saved, TensorBoard is opened when installed, and W&B
   is initialized unless disabled.

3. **Resolve the sequence and DINO targets.** The source directory contains a
   `colmap` child, so the canonical reader uses `ROOT/colmap` while dynamic data
   uses `ROOT`. Existing `ROOT/dinov3/**/*.pt` files are reused; otherwise the
   extractor is launched before scene construction.

4. **Load the canonical scene.** `Scene` reads COLMAP intrinsics, extrinsics,
   images, and the sparse point cloud. `GaussianModel` allocates canonical
   position, rotation, scale, opacity, spherical-harmonic appearance, semantic,
   and DINO fields.

5. **Load the iteration-30,000 PLY.** Loading replaces canonical `Parameter`
   objects, so `training_setup` is called again to bind a fresh static optimizer.
   The static loop has zero remaining steps for this command. The loaded scene
   can still be saved into the new output directory at iteration 30,000.

6. **Expand calibrated cameras over the video.** Each fixed COLMAP camera is
   cloned for every matching `rgb/cam_XXX/frame_XXXXXX.png`. Metric NPY depth is
   preferred, semantic masks and DINO tensors are attached, and timestamps are
   rebased to start at zero without changing physical frame deltas. Incomplete
   synchronized frame groups are rejected.

7. **Compute the exact Phase-1 budget.** Cameras are grouped by frame and
   physical camera ID. The coverage-preserving windows and six-pass schedule
   yield 744 updates; this replaces approximate accounting based only on
   `free_field_iters`.

8. **Measure full-scene motion evidence.** Before deleting background
   Gaussians, the initializer renders canonical frame-0 geometry against frame 0
   and selected future targets for every camera. It differentiates RGB,
   optional depth, and DINO residuals with respect to position, screen-space
   mean, opacity, appearance, DINO, scale, and rotation. Foreground-normalized
   channels are combined, and positive future-minus-frame0 responsibility is
   used as a weak moving-material prior. No optimizer step is taken here.

9. **Compact to semantic foreground.** Semantic probability is thresholded at
   `0.2`; background Gaussian rows are physically removed from every canonical
   tensor. Original row indices are retained so full-scene evidence can be
   gathered into the compact model. Fewer than the configured minimum number of
   foreground points is a hard error.

10. **Initialize material assignments.** `MaterialAssignmentField` creates one
    static class and `num_moving_parts` moving classes (one moving class by
    default). Continuous gradient evidence initializes probabilities between
    the configured floor and ceiling. The logits remain trainable throughout
    Phase 1; the initial hard mask is diagnostic and prior-weight metadata.

11. **Create the fixed curl grid.** The compact foreground bounding box is
    padded and divided into an aspect-ratio-aware grid near 4,096 cells. Three
    oriented edge tensors `A_x`, `A_y`, and `A_z`, each with cubic B-spline time
    controls, are allocated. Face fluxes are finite differences of `A`, and
    cell-centered velocity is the average of adjacent face fluxes. This gives
    `div_h(curl_h(A)) = 0` up to floating-point error.

12. **Create the two dynamic optimizers.** Adam with `motion_lr=1e-3` updates
    only the three vector-potential tensors. A separate Adam updates only the
    material logits. Canonical geometry, appearance, semantic logits, DINO
    features, and exposure are frozen.

13. **Build correspondences lazily.** For each adjacent synchronized camera
    pair, `MotionCorrespondenceCache` first looks for precomputed flow. This
    dataset has none, so deterministic OpenCV DIS forward and backward flow is
    computed. Forward/backward cycle consistency, image bounds, foreground
    masks, and frozen DINO cosine similarity form the confidence map. Results
    are cached on CPU and in a bounded device cache.

14. **Replay the current field to each source frame.** Every interval starts
    from canonical Gaussians and deterministically replays all earlier physical
    intervals. Gaussians are reassigned to fixed world cells as they move.
    Current point probabilities and DINO descriptors are aggregated into
    occupied cells, and an occupied-cell material-neighbor graph is rebuilt.

15. **Apply exact static gating.** A cell is hard static only when its aggregated
    static probability is at least `0.85` and at least the configured fraction
    of member Gaussians are individually static. Forward velocity is exactly
    zero in such cells. Backward propagation uses a sigmoid straight-through
    surrogate so rendering and flow can undo an incorrect static decision. A
    weight-250 penalty simultaneously drives the raw curl velocity toward zero.

16. **Evaluate and transport the Eulerian field.** At an interval midpoint,
    raw cell velocity is `curl_h(A(t_mid))`; the static gate produces the actual
    tracer velocity. Gaussian translation is explicit physical-time advection
    `x_end = x_start + dt v_cell`.

17. **Derive continuum kinematics.** For every occupied cell, weighted least
    squares over neighboring occupied material cells fits
    `v_d - v_c ≈ G_c (X_d - X_c)`. The code derives:

    ```text
    D_c     = 0.5 (G_c + G_c^T)       deformation/strain rate
    Omega_c = 0.5 (G_c - G_c^T)       infinitesimal rotation tensor
    omega_c = 0.5 curl(v_c)            angular velocity
    ```

    Quaternion rotations are integrated from derived `omega`; no angular
    velocity, pivot, center, axis, twist, or joint parameter is learned.

18. **Infer observed 3-D cell velocity through the CUDA rasterizer.** For each
    source camera, the same Gaussian `T·alpha` visibility weights used for
    rendering accumulate per-cell normal equations
    `J_pi^T J_pi` and `J_pi^T flow/dt`. Compact 3×3 systems are solved in CPU
    float64 to avoid large CUDA eigensolver workspaces. A target is accepted
    only with at least 6 observations, 3 distinct cameras, adequate conditioning,
    and at most 0.75 px moving-model RMS reprojection error.

19. **Render each target camera and accumulate data terms.** The temporary
    Gaussian state is set to transported position and derived rotation. The
    target foreground supplies RGB L1 + DSSIM, inverse-depth consistency, and
    frozen DINO feature-field supervision.

20. **Apply the rasterized projected-velocity flow loss.** For every Gaussian,
    the code computes the camera-aware displacement
    `project_target(x + v dt) - project_source(x)`. The new feature rasterizer
    splats that two-channel vector and a validity channel using detached
    source-time geometry, rotation, scale, and opacity. After alpha
    normalization, the dense prediction is compared with confident observed
    flow using a resolution-normalized robust endpoint error. Its backward path
    reaches only cell velocity and therefore the vector potential. The single
    `lambda_motion_projected_velocity_flow` coefficient is applied inside this
    loss exactly once.

21. **Apply observed-velocity and physics losses.** The configured objective
    combines direct 3-D velocity magnitude and direction, strain `||D||²`, robust
    vector-velocity and speed coherence, neighboring-gradient coherence, local
    affine consistency, soft and hard static speed, and temporal second
    differences of the B-spline controls. Curl magnitude is never minimized.

22. **Apply assignment losses.** Material logits receive the annealed evidence
    prior, canonical-neighbor coherence, DINO-aware boundary coherence, late
    entropy sharpening, and a kinematic hypothesis loss comparing static and
    moving explanations of observed flow. Assignment gradients also pass
    through the straight-through static gate in data and physics terms.

23. **Step according to the simplified schedule.** Five passes step both Adam
    optimizers; one assignment pass freezes `A` and steps only the material
    logits. Window order alternates forward/reverse. Gradients are clipped and
    logits are bounded. After motion updates, three default A-only projection
    steps reapply rigidity, static, and observed-velocity constraints.

24. **Replay the final state deterministically.** With final hard material
    labels, the code replays all 59 intervals from the canonical state, derives
    rotations from the final velocity gradient, and records positions,
    velocities, angular velocities, cell IDs, gates, gradients, strain, and
    continuity diagnostics for every frame.

25. **Save results.** The output contains the canonical/transported PLY save,
    a full checkpoint with dynamic schema version 20, final Gaussian tracks,
    material assignments, curl-grid state, gradient audits and summaries,
    attribution evidence, the explicit physics contract, and an HTML transported
    Gaussian visualization. For this command the final iteration is 30,744.

## Pruned implementation families

The source tree no longer contains the alternative adaptive octree/grid,
Lagrangian per-frame/MLP, direct motion-field, rigid-twist/basis, candidate
joint, hard-joint Phase-2, ScrewAIR outer-loop, or obsolete flow/joint debug
export implementations. Their parser flags, checkpoint branches, trainer
dispatch, stale repair/validation scripts, and duplicate projected-flow patch
directory were removed with them. The legacy transported-position trajectory
rasterizer, trajectory/depth loss, and their three redundant weights were also
removed in favor of the single projected-velocity flow path. The retained
source has no unresolved Python names under `ruff --select F,E9`.
