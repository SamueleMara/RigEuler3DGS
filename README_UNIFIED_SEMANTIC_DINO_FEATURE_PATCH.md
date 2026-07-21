# Unified Semantic + DINO Feature Field Patch

This patch wires arbitrary-channel feature rendering all the way through:

1. CUDA diff-gaussian-rasterization
2. PyTorch bindings/autograd wrapper
3. `gaussian_renderer/__init__.py`
4. `scene/gaussian_model.py`
5. `train.py`
6. dynamic/staged semantic + DINO losses

## What it adds

### CUDA / binding side

New bindings:

- `_C.rasterize_gaussian_features`
- `_C.rasterize_gaussian_features_backward`

They render a per-Gaussian tensor:

```python
features: [N, C]
```

into:

```python
rendered_features: [C, H, W]
```

using exactly the same alpha-composited splat equation as color:

```cpp
C[ch] += features[gaussian_id * total_channels + channel_offset + ch] * alpha * T;
```

Large feature dimensions are rendered/backpropagated in channel chunks of 32.

### Python renderer side

New functions:

```python
from gaussian_renderer import render_feature_field, render_semantic_field
```

- `render_feature_field(...)` renders arbitrary `[N,C]` Gaussian descriptors.
- `render_semantic_field(...)` renders a one-channel foreground probability from `sigmoid(gaussians.semantic_logits)`.

### GaussianModel side

New per-Gaussian fields:

```python
_semantic_logits: [N,1]
_dino_feature_field: [N,D]
```

They are:
- initialized with the Gaussians,
- optimized by the main Gaussian optimizer,
- pruned/densified together with Gaussians,
- saved as sidecars:
  - `semantic_logits.pt`
  - `dino_feature_field.pt`

The PLY remains vanilla-compatible.

### Training side

Static and dynamic stages now can include:

```text
L_semantic_feature_field:
  render semantic logits -> binary foreground image
  compare against GT mask with BCE/Dice

L_dino_feature_field:
  render per-Gaussian DINO descriptors
  compare against dense DINO feature map with cosine/L1
```

These losses are used in static training to learn stable fields and in dynamic training to supervise transported Gaussians in later frames.

## Apply

From repo root:

```bash
cd ~/Joint3DGS/RigEuler3DGS

# Rasterizer C++/CUDA/Python package
cp /mnt/data/unified_semantic_dino_feature_patch/submodules/diff-gaussian-rasterization/setup.py submodules/diff-gaussian-rasterization/setup.py
cp /mnt/data/unified_semantic_dino_feature_patch/submodules/diff-gaussian-rasterization/ext.cpp submodules/diff-gaussian-rasterization/ext.cpp
cp /mnt/data/unified_semantic_dino_feature_patch/submodules/diff-gaussian-rasterization/rasterize_points.h submodules/diff-gaussian-rasterization/rasterize_points.h
cp /mnt/data/unified_semantic_dino_feature_patch/submodules/diff-gaussian-rasterization/rasterize_points.cu submodules/diff-gaussian-rasterization/rasterize_points.cu
cp /mnt/data/unified_semantic_dino_feature_patch/submodules/diff-gaussian-rasterization/diff_gaussian_rasterization/__init__.py submodules/diff-gaussian-rasterization/diff_gaussian_rasterization/__init__.py
cp /mnt/data/unified_semantic_dino_feature_patch/submodules/diff-gaussian-rasterization/cuda_rasterizer/*.h submodules/diff-gaussian-rasterization/cuda_rasterizer/
cp /mnt/data/unified_semantic_dino_feature_patch/submodules/diff-gaussian-rasterization/cuda_rasterizer/*.cu submodules/diff-gaussian-rasterization/cuda_rasterizer/

# 3DGS Python code
cp /mnt/data/unified_semantic_dino_feature_patch/gaussian_renderer/__init__.py gaussian_renderer/__init__.py
cp /mnt/data/unified_semantic_dino_feature_patch/scene/gaussian_model.py scene/gaussian_model.py
cp /mnt/data/unified_semantic_dino_feature_patch/arguments/__init__.py arguments/__init__.py
cp /mnt/data/unified_semantic_dino_feature_patch/train.py train.py
cp /mnt/data/unified_semantic_dino_feature_patch/utils/dynamic_loss_utils.py utils/dynamic_loss_utils.py
cp /mnt/data/unified_semantic_dino_feature_patch/scene/adaptive_motion_grid.py scene/adaptive_motion_grid.py
```

## Rebuild rasterizer

```bash
conda activate rigeuler3dgs
cd ~/Joint3DGS/RigEuler3DGS/submodules/diff-gaussian-rasterization

pip uninstall -y diff_gaussian_rasterization
TORCH_CUDA_ARCH_LIST="8.9" pip install -v .
```

Quick import check:

```bash
python - <<'PY'
import diff_gaussian_rasterization as dgr
print("OK", dgr.GaussianRasterizer)
PY
```

## Important new flags

Semantic mask field:

```bash
--use_semantic_feature_field
--lambda_semantic_feature_field 0.05
--semantic_feature_lr 1e-3
--semantic_mask_dir mask
--semantic_loss_type bce_dice
```

DINO feature field:

```bash
--use_dino_feature_field
--lambda_dino_feature_field 0.02
--dino_feature_field_dim 32
--dino_feature_lr 1e-3
```

Disable if needed:

```bash
--disable_semantic_feature_field
--disable_dino_feature_field
```

## Expected data

Masks:

```text
ROOT/mask/cam_000/frame_000000.png
ROOT/mask/cam_001/frame_000000.png
...
```

DINO:

```text
ROOT/dinov3/cam_000/frame_000000.pt
ROOT/dinov3/cam_001/frame_000000.pt
...
```

DINO `.pt` can be:
- raw tensor `[C,H,W]`
- raw tensor `[1,C,H,W]`
- dict with key `features`, `dino`, `dino_feat`, `feature`, or `feat`

## Recommended smoke test

```bash
python -u train.py \
  --root_scene /home/samuelemara/Joint3DGS/Sapien/output/partnet_subset_fixed_render_linear/USB_100109 \
  -m output/USB_100109_semantic_dino_smoke \
  --dynamic_motion \
  --static_train_iters 30000 \
  --euler_field_train_iters 1500 \
  --start_ply output/USB_100109_dynamic_v4/point_cloud/iteration_30000/point_cloud.ply \
  --start_ply_iteration 30000 \
  --use_depth \
  --use_semantic_feature_field \
  --lambda_semantic_feature_field 0.05 \
  --semantic_mask_dir mask \
  --use_dino_feature_field \
  --lambda_dino_feature_field 0.02 \
  --dino_feature_field_dim 32 \
  --num_moving_parts 1 \
  --dynamic_discovery_iters 1000 \
  --motion_understanding_from_iter 1000 \
  --lambda_global_twist 0.05 \
  --global_twist_from_iter 1000 \
  --lambda_global_se3 0.0 \
  --lambda_velocity_rigid 0.0 \
  --enable_adaptive_grid \
  --grid_from_iter 500 \
  --grid_update_interval 250 \
  --grid_split_threshold 0.0 \
  --grid_split_percentile 90 \
  --grid_score_ema_decay 0.95 \
  --grid_max_splits_per_update 64 \
  --grid_min_points 4 \
  --grid_merge_threshold 0.0 \
  --grid_lambda_vel 2.0 \
  --grid_lambda_dino 2.0 \
  --grid_lambda_render_grad 2.0 \
  --grid_lambda_dyn_entropy 1.0 \
  --grid_lambda_dyn_var 1.0 \
  --grid_lambda_twist_residual 0.5 \
  --grid_lambda_strain 0.5 \
  --make_track_html \
  --make_velocity_html \
  --wandb_entity samuelemara-my-institution \
  --wandb_project RigEuler3DGS \
  --wandb_name USB_100109_semantic_dino_smoke \
  --disable_viewer
```

## Losses to monitor

```text
loss/static/L_semantic_feature_field
loss/static/L_dino_feature_field
loss/dynamic/L_semantic_feature_field
loss/dynamic/L_dino_feature_field
loss/dynamic/L_global_twist
diagnostic/grid/*
```

