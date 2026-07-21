# Joint single-rigid-part dynamic training patch

This patch is built on top of the high-debug + adaptive-grid-octomap branch.
It removes the hard dynamic-discovery / rigid-motion phase split by default.

## Model

For each foreground Gaussian and time:

```text
v_free(x,t)  = current Eulerian MLP + adaptive-grid proposal
v_rigid(x,t) = omega(t) x (x - center) + u(t)
p_i          = sigmoid(dynamic_logit_i)

v_transport = p_i.detach() * v_rigid + (1 - p_i.detach()) * v_free
```

The detached probability gives EM-like gradient routing: rendering trains the
motion proposal and explicit twist, while label losses train the assignment.
Early motion is bootstrapped by `v_free`; as labels become confident, the moving
part is handed over to the explicit rigid field.

## Trainable parameters during every dynamic iteration

- `motion_field` MLP: free motion proposal
- `motion_grid.features`: local proposal conditioning
- `rigid_twist_field`: explicit bounded `omega(t), u(t)`
- `_dynamic_logit`: per-Gaussian static/dynamic assignment
- exposure parameters

Canonical Gaussian geometry/appearance remains frozen under the existing
configuration.

## Active joint losses

- `L_joint_rigid`: high-motion or dynamic points: `v_free -> v_rigid`
- `L_joint_static`: low-motion static points: `v_free -> 0`
- `L_joint_label`: BCE to identity-vs-rigid residual target
- `L_joint_cluster`: cosine coherence of `[1-p,p]` inside initial XYZ+DINO supernodes
- `L_joint_residual`: suppress remaining free transport as the rigid handover matures
- `L_joint_twist_smooth`: temporal first/second differences of `omega(t),u(t)`
- `L_joint_label_binary`: weak binary sharpening
- existing RGB/depth/DINO rendering losses
- existing cycle consistency

Legacy `L_global_twist`, model-target, discovery sparsity, contrastive-label, and
hard phase-specific losses are disabled by default in joint mode.

## Initial clustering

At dynamic initialization, foreground canonical Gaussians are over-clustered in:

```text
joint_cluster_xyz_weight * normalized_xyz
+
joint_cluster_dino_weight * normalized_DINO
```

The default is 64 fixed supernodes. The final static/dynamic clustering remains
trainable through the per-Gaussian binary assignment and cosine assignment loss.
There is exactly one explicit moving rigid model and the complementary static
identity model.

## Stability protections

- bounded `omega(t)` and `u(t)` via `tanh`
- fixed canonical foreground center in the cross product
- total velocity norm safety clamp
- robust pseudo-Huber velocity losses
- gradient clipping
- no unbounded external velocity teacher
- grid split/merge preserves Adam state for unchanged MLP/twist/logit parameters

## High-debug exports

Snapshots are now:

```text
joint_early
joint_mid
joint_final
```

Each contains the existing tracks, velocity HTML, adaptive-grid octomap HTML,
per-point/per-cell diagnostics, plus:

```text
rigid_twist_trajectory.html
rigid_twist_trajectory.summary.json
rigid_omega / rigid_translation in gaussian_tracks.pt
rigid_velocities / residual_velocities in gaussian_tracks.pt
joint_cluster_id in gaussian_tracks.pt
```

## Apply

```bash
cd ~/Joint3DGS/RigEuler3DGS

cp /mnt/data/joint_single_rigid_training_patch/train.py train.py
cp /mnt/data/joint_single_rigid_training_patch/arguments/__init__.py arguments/__init__.py
cp /mnt/data/joint_single_rigid_training_patch/utils/dynamic_loss_utils.py utils/dynamic_loss_utils.py
cp /mnt/data/joint_single_rigid_training_patch/utils/loss_utils.py utils/loss_utils.py
cp /mnt/data/joint_single_rigid_training_patch/utils/visualize_motion_grid_octomap_html.py utils/visualize_motion_grid_octomap_html.py
cp /mnt/data/joint_single_rigid_training_patch/utils/visualize_rigid_twist_html.py utils/visualize_rigid_twist_html.py
cp /mnt/data/joint_single_rigid_training_patch/scene/gaussian_model.py scene/gaussian_model.py
cp /mnt/data/joint_single_rigid_training_patch/scene/adaptive_motion_grid.py scene/adaptive_motion_grid.py
cp /mnt/data/joint_single_rigid_training_patch/scene/rigid_twist_field.py scene/rigid_twist_field.py
cp /mnt/data/joint_single_rigid_training_patch/gaussian_renderer/__init__.py gaussian_renderer/__init__.py
mkdir -p tools
cp /mnt/data/joint_single_rigid_training_patch/tools/analyze_phase_debug_snapshots.py tools/analyze_phase_debug_snapshots.py
cp /mnt/data/joint_single_rigid_training_patch/tools/visualize_motion_grid_octomap.py tools/visualize_motion_grid_octomap.py
```

## Run

```bash
python train.py \
  --root_scene /home/samuelemara/Joint3DGS/Sapien/output/partnet_subset_fixed_render_linear/USB_100109 \
  -m output/USB_100109_joint_single_rigid_high_debug \
  --dynamic_motion \
  --filter_sem \
  --semantic_dynamic_threshold 0.2 \
  --start_ply output/USB_100109_rgb_depth_sem_dino_from_scratch/point_cloud/iteration_30000/point_cloud.ply \
  --start_ply_iteration 30000
```

Joint mode is enabled by default. Use `--disable_joint_single_rigid_training` to
run the legacy separated phases.
