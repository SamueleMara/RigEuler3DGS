# EulerRig training implementation recap

This document describes the current training implementation in this repository. It is based on the active code paths in `train.py`, `arguments/__init__.py`, `scene/gaussian_model.py`, and the dynamic utilities under `utils/`.

Ground truth is not an algorithm input. The only GT paths in the training code are post-solve validation/audit paths guarded by `--rigid_analytic_validation`.

## Entry point

The executable training entry is:

```bash
python train.py ...
```

Main control flow:

1. `train.py::main()`
   - Creates `ModelParams`, `OptimizationParams`, `PipelineParams`.
   - Adds dynamic arguments through `arguments.add_dynamic_args()`.
   - Normalizes/validates them through `arguments.apply_dynamic_defaults()`.
   - Starts the viewer unless `--disable_viewer`.
   - Calls `training(...)`.

2. `train.py::training(...)`
   - Creates the output directory and `cfg_args`.
   - Initializes W&B unless `--no_wandb` or `--wandb_mode disabled`.
   - Ensures DINO targets exist when `--use_dino_feature_field`.
   - Builds the canonical `Scene` and `GaussianModel`.
   - Loads `--start_ply` or checkpoint if requested.
   - Runs/static-skips Phase 0.
   - Enables the dynamic sequence.
   - Selects one dynamic branch:
     - staged K-part rigid discovery and joints;
     - single rigid Killing phase;
     - fixed-curl Eulerian free-field phase.

## High-level phases

| Phase | Code | Purpose | Main trainable state |
| --- | --- | --- | --- |
| 0 | `train_static()` | Canonical 3DGS RGB/depth/semantic/DINO training or start-Ply load | Canonical Gaussian xyz, scale, rotation, opacity, SH, semantic logits, DINO features |
| 1A | `initialize_*motion_labels()` or depth/visibility pretrain | Initialize dynamic/material state before compaction | Usually no optimizer step for K-part path |
| 1 | `run_kpart_rigid_discovery()` → `run_velocity_field_kpart_discovery()` | Discover static + K rigid moving parts | Soft material responsibilities; rigid velocity fields are refit each cycle |
| 2 | `fit_joint_graph()` | Fit revolute/prismatic/screw candidates and select by held-out rendering | Candidate joint geometry and candidate `delta_q` during comparison |
| 3 | `optimize_joint_trajectory()` | Optimize final absolute joint coordinates and exact SE(3) replay | Per-frame or spline joint coordinates; optional small joint-geometry refinement |
| Final | `gaussians.set_kpart_joint_state()` + `scene.save()` | Install frozen final dynamic state and save checkpoint/PLY | Stored replay state, not free dynamic optimizer |

## Phase 0: canonical 3DGS

Code locations:

- `train.py::train_static`
- `train.py::_static_visual_loss`
- `scene/gaussian_model.py::training_setup`
- `scene/gaussian_model.py::load_ply`
- `scene/gaussian_model.py::save_ply`

Behavior:

1. If `--start_ply` is given, the model loads the existing canonical PLY.
2. Loading a PLY replaces parameter objects, so `gaussians.training_setup(opt)` is called again to bind the optimizer to the loaded tensors.
3. If the loaded iteration is already at or beyond `--static_train_iters`, canonical optimization is skipped.
4. Otherwise, the static loop samples canonical cameras, renders RGB/depth, backpropagates, densifies/prunes until `densify_until_iter`, and logs losses.

Static losses:

- RGB L1.
- DSSIM through `ssim`.
- Optional inverse-depth L1 when reliable depth exists.
- Optional semantic feature-field loss.
- Optional DINO feature-field loss.

Phase 0 is the only phase where canonical Gaussian geometry/appearance should be freely optimized. Dynamic phases freeze canonical state before motion discovery.

## Dynamic sequence setup

Code locations:

- `train.py::_sequence_root`
- `scene.Scene.enable_dynamic_sequence`
- `scene/dynamic_sequence_dataset.py`

The dynamic dataset is resolved from `--dynamic_sequence_root` or from the source path. The loader expands fixed calibrated cameras over the synchronized video frames:

- RGB from `--dynamic_rgb_dir`.
- Metric depth from `--dynamic_depth_npy_dir` or depth PNG.
- Masks from `--semantic_mask_dir`.
- DINO tensors from `--dinov3_root`.
- Timestamps from `--dynamic_timestamps_file`, otherwise `frame_id / fps`.

The dynamic camera data device is temporarily switched to `--dynamic_camera_data_device` while sequence cameras are loaded, then restored.

## Dynamic initialization before Phase 1

Code locations:

- `train.py::training`
- `utils/unified_dynamic_training.py::initialize_neutral_motion_labels`
- `utils/unified_dynamic_training.py::initialize_gradient_motion_labels`
- `utils/temporal_visibility_pretraining.py`
- `scene/gaussian_model.py::init_dynamic_motion`

The initialization depends on the selected branch.

For the current staged K-part branch:

```bash
--dynamic_motion
--phase1_motion_representation rigid_killing_eulerian|rigid_killing_lagrangian
--kpart_rigid_discovery
```

the code uses neutral initialization:

- every semantic-foreground Gaussian starts with uniform static/dynamic responsibility;
- no GT labels are used;
- depth/sequence pretraining is explicitly disallowed for this branch by `apply_dynamic_defaults()`;
- support type is inferred:
  - `rigid_killing_eulerian` → `args.kpart_support = eulerian`;
  - `rigid_killing_lagrangian` → `args.kpart_support = lagrangian`.

Then `GaussianModel.init_dynamic_motion()`:

1. Compacts to semantic foreground with `compact_to_semantic_foreground()`.
2. Creates `MaterialAssignmentField`.
3. For rigid Killing representations, initializes rigid state but does not create a curl grid or motion optimizer.
4. Freezes canonical geometry and appearance through `freeze_canonical(True, True)`.

## Active Phase 1: velocity-field K-part discovery

Code locations:

- `utils/kpart_rigid_discovery.py::run_kpart_rigid_discovery`
- `utils/kpart_velocity_field_discovery.py::run_velocity_field_kpart_discovery`
- `utils/kpart_soft_responsibility.py`
- `utils/kpart_field_merge.py`
- `utils/kpart_initial_oversegmentation.py`
- `utils/kpart_counterfactual_rendering.py`
- `utils/kpart_dino_boundary.py`
- `utils/motion_correspondence.py`

`run_kpart_rigid_discovery()` now dispatches to `run_velocity_field_kpart_discovery()` by default because `--kpart_velocity_field_segmentation` defaults to true.

### Support backends

The Eulerian and Lagrangian versions share the same high-level Stage 1 loop. The support backend changes only how observations are pooled.

| Backend | Flag | Support unit | Main effect |
| --- | --- | --- | --- |
| Eulerian | `--phase1_motion_representation rigid_killing_eulerian` | Occupied spatial supports/cells, optionally split by DINO/responsibility | More stable pooled observations, but mixed supports must be controlled |
| Lagrangian | `--phase1_motion_representation rigid_killing_lagrangian` | One Gaussian per support | No cell mixing, heavier/noisier observations |

The code states this explicitly in `phase1_kpart_summary.json` with `observation_backend_only_difference: true`.

### Observation construction

Code:

- `utils/kpart_velocity_field_discovery.py::_build_observation_pack`
- `utils/motion_correspondence.py::MotionCorrespondenceCache`
- `utils/motion_correspondence.py::observed_velocity_targets`

For each interval and support, the rasterized flow backend accumulates observed 3D velocity evidence from all physical cameras:

- observed velocity;
- confidence;
- validity;
- distinct camera count;
- normal matrix;
- static-vs-moving reprojection diagnostics.

Static is represented as an exact zero-velocity model. Moving parts are represented by full rigid velocity fields.

### Initial regions and fields

The velocity-field Stage 1 does not first trust a hard dynamic mask. It builds provisional regions and rigid fields:

1. Optional oversegmentation through `initialize_provisional_regions`.
2. Static render-failure bootstrap from later-frame canonical render errors.
3. Region rigid-field fitting through `fit_region_rigid_field`.
4. Merge-to-K through `merge_until_k`, using shared-field, render, and DINO terms.

This is why the current implementation avoids the earlier bad idea of merging arbitrary K′ clusters only by label entropy: merge decisions must be backed by velocity-field compatibility.

### Cycle loop

Per cycle in `run_velocity_field_kpart_discovery()`:

1. Freeze current hard pooling from `responsibilities.argmax(-1)`.
2. Build a static observation pack.
3. Build one observation pack per moving part under its current rigid mode.
4. Refit each moving-part rigid velocity field with confidence-weighted observations.
5. Compute static-vs-part unaries with `compute_velocity_field_unaries`.
6. Normalize field/render unary families.
7. Optionally evaluate uncertain components by counterfactual rendering.
8. Build DINO 3D affinity and optional rendered 2D seam regularization.
9. Optimize responsibility logits for a few inner steps.
10. Mark low-confidence/low-margin/high-entropy points ambiguous.
11. Save cycle PT and HTML diagnostics.

### Responsibility objective

Code:

- `utils/kpart_soft_responsibility.py::optimize_responsibility_logits`

The optimized variable is the responsibility logit table. Responsibilities are:

```text
r_i = softmax(logits_i)
```

The inner objective is:

```text
L = mean_i sum_k r_ik * unary_ik
  + temperature * mean_i sum_k r_ik log(r_ik)
  + kpart_dino_3d_weight * L_DINO_3D
  + kpart_dino_2d_seam_weight * L_DINO_2D_seam
```

where:

- `unary_ik` is the compatibility of point/support `i` with static zero field or moving rigid field `k`;
- entropy keeps assignment soft early;
- DINO 3D regularization keeps nearby same-appearance neighbors coherent;
- DINO 2D seam loss discourages rendered part boundaries away from DINO image boundaries.

Important: the rigid fields are not arbitrary per-Gaussian transforms. They are solved/refit as physical rigid velocity fields and then used as unaries for responsibility optimization.

### Phase 1 outputs

Written under:

```text
<model>/dynamic_debug/
```

Main files:

- `phase1_kpart_responsibility_history.pt`
- `phase1_kpart_responsibility_final.pt`
- `phase1_kpart_assignment_final.pt`
- `phase1_kpart_rigid_velocity_fields.pt`
- `phase1_kpart_coarse_rigid_modes.pt`
- `phase1_kpart_coarse_trajectories.pt`
- `phase1_static_vs_part_energies.pt`
- `phase1_kpart_ablation_labels.pt`
- `phase1_kpart_summary.json`
- `phase1_kpart_coarse_gaussian_tracks.pt`
- `phase1_kpart_coarse_motion.html`
- `phase1_kpart_cycle_XXX.html`

For Eulerian support, additional support diagnostics are saved:

- `phase1_eulerian_contrastive_supports.pt`
- `phase1_eulerian_mixed_support_history.pt`

## Phase 2: physical joint fitting

Code locations:

- `utils/joint_model_fitting.py::fit_joint_graph`
- `utils/joint_model_fitting.py::fit_revolute_joint`
- `utils/joint_model_fitting.py::fit_prismatic_joint`
- `utils/joint_model_fitting.py::fit_screw_joint`
- `utils/joint_model_fitting.py::_fit_rendered_candidate`

Enabled by:

```bash
--fit_joints
--joint_types auto
```

`apply_dynamic_defaults()` currently requires `--joint_types auto` when `--fit_joints` is active. The rendered competition always fits all three candidate families:

- revolute;
- prismatic;
- screw.

For each moving part:

1. Initialize candidates from the Stage 1 rigid mode and coarse trajectory.
2. Split physical cameras into train and held-out validation cameras.
3. Optimize each candidate in blocks:
   - trajectory block;
   - geometry block;
   - trajectory block.
4. Render the full dynamic scene for train cameras.
5. Score held-out cameras.
6. Select the lowest held-out total loss.
7. If screw only wins by less than `--joint_compare_screw_margin`, prefer revolute.

Stage 2 render loss:

```text
L_candidate = L_rgb + depth_l1_weight_final * L_depth
            + lambda_dino_feature_field_dynamic * L_DINO
            + 1e-4 * trajectory_acceleration
```

The candidate optimizer modifies only candidate joint parameters and candidate `delta_q`; canonical Gaussians and Stage 1 labels are frozen.

Main Phase 2 outputs:

- `phase2_part_<part>_revolute_candidate.pt`
- `phase2_part_<part>_prismatic_candidate.pt`
- `phase2_part_<part>_screw_candidate.pt`
- `phase2_joint_summary.json`
- `phase2_joint_graph.pt`
- optional `phase2_joint_candidates_gt_evaluation.json` if GT validation is enabled.

## Phase 3: absolute joint trajectory optimization

Code locations:

- `utils/joint_trajectory_optimization.py::optimize_joint_trajectory`
- `utils/joint_trajectory_optimization.py::optimize_framewise_q`
- `utils/joint_trajectory_optimization.py::optimize_spline_q`
- `utils/joint_trajectory_optimization.py::refine_joint_geometry_small`
- `utils/joint_trajectory_optimization.py::render_joint_constrained_motion`
- `utils/rigid_transform_utils.py`

Enabled by:

```bash
--optimize_joint_trajectory
```

Stage 3 starts from Stage 2 selected joints and coarse coordinates. It optimizes absolute joint coordinates with gauge:

```text
q[:, 0] = 0
```

Two trajectory parameterizations exist:

- `--joint_trajectory_mode per_frame`: optimize all frame coordinates except the first.
- `--joint_trajectory_mode spline`: optimize cubic B-spline controls and sample per-frame coordinates.

Optional:

```bash
--joint_geometry_refine
```

adds a small geometry refinement pass for axis/pivot/pitch, regularized toward Stage 2 geometry.

Stage 3 loss:

```text
L = L_render + 1e-4 * L_temporal_acceleration
```

where `L_render` contains RGB, depth, and DINO terms over complete dynamic cameras. The function also computes a final flow objective/audit with static-flow calibration.

Main outputs:

- `phase3_static_flow_calibration.pt`
- `phase3_joint_coordinates.pt`
- `phase3_joint_velocity.pt`
- `phase3_joint_acceleration.pt`
- `phase3_final_transforms.pt`
- `phase3_final_gaussian_tracks.pt`
- `phase3_selected_joint_trajectory.pt`
- `phase3_selected_joint_tracks.pt`
- `phase3_metric_motion_summary.json`
- `phase3_final_joint_motion.html`
- `phase3_joint_trajectory.html`
- optional `phase3_joint_trajectory_gt_evaluation.html` and `phase3_joint_gt_validation.json`.

Quaternion/covariance transport is audited. If the audit fails, Stage 3 raises instead of silently saving inconsistent motion.

## Final dynamic state installation

Code:

- `scene/gaussian_model.py::set_kpart_joint_state`

After Stage 1/2/3, the final state is installed into `GaussianModel`:

- joints;
- absolute coordinates `q`;
- velocity `qdot`;
- frame times;
- soft responsibilities;
- hard/ambiguous part labels;
- diagnostics.

The model then saves:

- final scene PLY/checkpoint at `args.iterations`;
- dynamic state inside the checkpoint capture;
- final replay can be produced by `GaussianModel.dynamic_state_at(t)`.

## Legacy fixed-curl Eulerian free-field branch

Code locations:

- `utils/unified_dynamic_training.py::run_eulerian_phase1_pipeline`
- `scene/fixed_curl_grid.py`
- `scene/material_assignment_field.py`
- `utils/dynamic_loss_utils.py`
- `utils/motion_correspondence.py::rasterized_projected_velocity_flow_loss`

Selected when:

```bash
--dynamic_motion
--phase1_motion_representation eulerian_curl_strain
```

and `--kpart_rigid_discovery` is not used.

This branch creates a fixed finite-volume Eulerian curl grid:

```text
v(t) = curl_h(A(t))
```

where `A(t)` is a cubic B-spline vector-potential field. Trainable state:

- vector-potential controls;
- material-assignment logits.

Canonical Gaussian geometry/appearance remains frozen.

Main losses:

- observed 3D flow velocity magnitude;
- observed 3D flow direction;
- rasterized projected-velocity optical-flow loss;
- Eulerian strain-rate loss `||sym(grad v)||`;
- velocity/speed spatial coherence;
- spatial gradient coherence;
- local affine coherence;
- static speed/strain;
- hard static speed;
- assignment prior/spatial/DINO/entropy;
- assignment kinematic evidence;
- curl temporal acceleration.

This branch is useful for free-field debugging. The staged K-part branch is currently the intended path for joint estimation.

## Pure-depth geometry handoff script

Code:

- `depth_geometry_relative_hungarian.py`

This is separate from `train.py`. It builds static/dynamic/uncertain anchors directly from the input depth sequence and exports an initial PLY label field. It is useful as a diagnostic or seed source for Gaussian-level assignment.

Current keyframe-joint implementation in that script:

1. Samples symmetric keyframes only; it no longer explores all frames when `--keyframe-joint-optimization` is active.
2. Optimizes all selected keyframe pairs, not only consecutive pairs.
   - For 3 keyframes `[0, M, L]`, it trains `0→M`, `M→L`, and `0→L`.
3. Replaces per-point trainable `delta` with a local SE(3) twist:

   ```text
   xi_i = [v_i, omega_i]
   T_i = exp(xi_i)
   ```

4. Initializes every local transform from weighted Kabsch over the graph-matched neighborhood.
5. Computes rigidity weights from relative-distance preservation:

   ```text
   w_ij = exp(-((d_ij^a - d_pi(i)pi(j)^b)^2) / (2 sigma_r^2))
   ```

6. Uses transform smoothness:

   ```text
   L_T = sum_ij w_ij ||log(T_i^-1 T_j)||^2
   ```

7. Computes physical model evidence:

   ```text
   E_static  = ||x_target - x_source||
   E_dynamic = ||x_target - T_i x_source||
   margin    = E_static - E_dynamic
   ```

8. Trains dynamic logits directly from that model margin.
9. Keeps binarization confidence-gated and late-annealed, with default `--keyframe-binarization-weight 0.05`.
10. Adds skip-pair composition loss when shorter-pair mappings agree:

    ```text
    T_i^(0→L) ≈ T_(pi_0M(i))^(M→L) T_i^(0→M)
    ```

11. Uses triplet composition confidence to down-weight final track evidence.
12. Keeps weak or contradictory evidence uncertain instead of hard-overwriting labels.

New debug outputs:

- `keyframe_joint_history.csv`
- `keyframe_joint_point_labels.csv`
- `keyframe_joint_pair_stats.csv`
- `keyframe_joint_composition.csv`
- `keyframe_joint_evidence_histograms.csv`
- `keyframe_joint_evidence_summary.csv`
- `keyframe_joint_*.ply`

The per-pair PLYs now include:

- `dynamic`
- `p_dynamic`
- `E_static`
- `E_dynamic`
- `evidence_margin`
- `composition_confidence`
- displacement debug `dx,dy,dz`
- twist debug `twist_vx,twist_vy,twist_vz,twist_wx,twist_wy,twist_wz`

The final exported point-cloud label contract is:

```text
0.0 = confident static
0.5 = uncertain
1.0 = confident dynamic
```

## Logging

Code:

- `train.py::_setup_wandb`
- `train.py::log_dynamic`

W&B is enabled by default unless disabled:

```bash
--no_wandb
```

or:

```bash
--wandb_mode disabled
```

Dynamic logs preserve existing prefixes when the key starts with known stage names:

- `visibility`
- `flow_calibration`
- `assignment_pretrain`
- `rigid_fit`
- `assignment_refine`
- `sequence`
- `camera`
- `eulerian`
- `lagrangian`
- `comparison`
- `depth_pretrain`
- `neutral_discovery`
- `responsibility`
- `eulerian_support`
- `part_discovery`
- `joint_fit`
- `joint_compare`
- `joint_selected`
- `joint_trajectory`
- `quaternion_transport`

Other stats are written under:

```text
diagnostic/<stage>/<key>
```

## Current intended test commands

Eulerian K-part + joint trajectory:

```bash
python train.py \
  -s /home/samuelemara/Joint3DGS/Sapien/output/partnet_subset_fixed_render_linear/USB_100109 \
  -m /home/samuelemara/Joint3DGS/RigEuler3DGS/output/USB_100109_eulerrig_eulerian_recap_test \
  --start_ply /home/samuelemara/Joint3DGS/RigEuler3DGS/output/USB_100109/point_cloud/iteration_30000/point_cloud.ply \
  --dynamic_motion \
  --phase1_motion_representation rigid_killing_eulerian \
  --kpart_rigid_discovery \
  --num_moving_parts 1 \
  --kpart_anneal_cycles 12 \
  --kpart_plateau_cycles 8 \
  --fit_joints \
  --joint_types auto \
  --joint_topology known \
  --joint_compare_iters 40 \
  --joint_compare_batch_size 8 \
  --joint_compare_validation_cameras 2 \
  --optimize_joint_trajectory \
  --joint_trajectory_iters 300 \
  --joint_trajectory_batch_size 8 \
  --dynamic_use_depth \
  --use_semantic_feature_field \
  --use_dino_feature_field \
  --semantic_dynamic_threshold 0.2 \
  --curl_grid_target_cells 4096 \
  --rigid_analytic_validation \
  --wandb \
  --wandb_mode online \
  --wandb_project RigEuler3DGS \
  --wandb_name USB_100109_eulerrig_eulerian_recap_test \
  --wandb_log_interval 1 \
  --disable_viewer
```

Lagrangian K-part + joint trajectory changes only:

```bash
--phase1_motion_representation rigid_killing_lagrangian
-m /home/samuelemara/Joint3DGS/RigEuler3DGS/output/USB_100109_eulerrig_lagrangian_recap_test
--wandb_name USB_100109_eulerrig_lagrangian_recap_test
```

Pure-depth keyframe SE(3) labeler 500-iteration test:

```bash
python depth_geometry_relative_hungarian.py \
  /home/samuelemara/Joint3DGS/RigEuler3DGS/input/USB_100109 \
  --output-dir /home/samuelemara/Joint3DGS/RigEuler3DGS/input/USB_100109/USB_100109_keyframe_twist_500 \
  --force \
  --keyframe-joint-optimization \
  --keyframe-count 3 \
  --keyframe-iters 500 \
  --keyframe-binarization-weight 0.05 \
  --target-points 4300 \
  --no-visual-debug
```

## Practical status

The repository currently has many modified, deleted, and untracked files. The active dynamic implementation is not a clean small patch over upstream 3DGS; it is a research branch with multiple experimental paths still present.

The stable path to focus on for joint discovery is:

```text
train.py
  → run_kpart_rigid_discovery()
  → run_velocity_field_kpart_discovery()
  → fit_joint_graph()
  → optimize_joint_trajectory()
  → set_kpart_joint_state()
```

The pure-depth script is a separate pre-handoff/static-anchor diagnostic path. It should not be confused with the online Gaussian training loop.
