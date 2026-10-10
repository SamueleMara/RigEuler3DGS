# Codex task: surgical Splatfacto-style transparency carving for official Graphdeco 3DGS

## Scope — read this first

Modify **the existing working checkout** of `graphdeco-inria/gaussian-splatting` to train on **RGB images of isolated objects with black backgrounds**, using a **separate binary mask per camera image** (white = object/foreground, black = background). Implement **only** the mechanism used by Nerfstudio Splatfacto for transparent object reconstruction:

1. Treat the binary object mask as **ground-truth alpha**, NOT as a loss-exclusion mask.
2. At **every training iteration**, sample **one random RGB background color** `b = torch.rand(3, device="cuda")` (one color for the entire image).
3. Render the Gaussians **once** with that same `b` using the **original Graphdeco CUDA renderer**.
4. Construct the **ground-truth composite** with the **same background**: `target = foreground_rgb * alpha_gt + b * (1-alpha_gt)` (equivalently `target = premultiplied_foreground_rgb + (1-alpha_gt)*b`).
5. Compare the **unmasked prediction** against the composited target using the **original L1 + SSIM loss** across **the whole image**, including background.

**Strict exclusions:** NO new silhouette/BCE loss. NO depth losses or changes to depth handling. NO normals, depth variance, depth-based initialization, mesh or point-cloud changes. NO CUDA/C++ changes. NO new pruning, thresholding, splat-size caps, or changes to densification. NO `gsplat` migration, no refactors unrelated to this behavior. Keep the usual training, saving, PLY, GUI, evaluation, and checkpoint behavior intact.

**Required deliverable:** Working code modification in the current repository + a brief changed-files report and actual tests. Do not stop after writing a plan. Do not create a new repo.

## Verified upstream code points

- Graphdeco: `train.py` currently samples `bg = torch.rand((3), device="cuda") if opt.random_background else background`, calls `render(...)`, then does `image *= viewpoint_cam.alpha_mask` and compares against `viewpoint_cam.original_image`. **That is the problematic part for masked RGB objects.**
- Graphdeco: `utils/camera_utils.py::loadCam` loads an RGB image and calculates its training resolution before constructing `Camera`.
- Graphdeco: `scene/cameras.py::Camera` already stores `original_image`, `alpha_mask`, camera dimensions and transforms. Keep `alpha_mask` behavior for the legacy path (including special train/test exposure splits).
- Graphdeco: `arguments/__init__.py::ModelParams` has `_images`, `_depths`, `_resolution`; `OptimizationParams` already has `random_background`. Do not duplicate `random_background`.
- Splatfacto: `nerfstudio/models/splatfacto.py`: `get_outputs()` composites `render + (1-alpha)*background`; `composite_with_background()` forms the alpha-composited ground truth. Its generic `batch["mask"]` in `get_loss_dict()` is a **different mechanism** that excludes pixels. **Do not copy that masking behavior.**

References:
- https://github.com/graphdeco-inria/gaussian-splatting/blob/main/train.py
- https://github.com/graphdeco-inria/gaussian-splatting/blob/main/utils/camera_utils.py
- https://github.com/graphdeco-inria/gaussian-splatting/blob/main/scene/cameras.py
- https://github.com/nerfstudio-project/nerfstudio/blob/main/nerfstudio/models/splatfacto.py

## Requested input layout / CLI

Add one optional model parameter in `arguments/__init__.py`, following the repo's existing naming convention:

```python
# inside ModelParams.__init__
self._masks = ""
```

This exposes `--masks` (a directory path, relative to `-s` / `--source_path`, or absolute). Default `""` means **fully unchanged legacy behavior**.

The standard object dataset layout is:

```text
DATASET/
  images/
    cam_0_0001.png
    cam_0_0002.png
    cam_1_0001.png
  masks/
    cam_0_0001.png   # single-channel, values 0 or 255
    cam_0_0002.png
    cam_1_0001.png
  sparse/0/...       # existing calibration / COLMAP assets
```

Require a corresponding mask for **every** training/evaluation image when `--masks` is set; a missing or ambiguous match is an **error**, not a silent all-ones fallback. Do not use `basename` matching in a way that would mix similarly named files from different cameras. For nested directories, mirror the relative paths under the RGB image root:

```text
images/cam_0/0001.jpg  --> masks/cam_0/0001.png
images/cam_1/0001.jpg  --> masks/cam_1/0001.png
```

For standard COLMAP input, the image root is `source_path / args.images` (default `images`). For nonstandard/synthetic layouts outside that root, derive the relative path from `source_path`, or fail with a clear path-resolution error; never guess another camera's mask.

Only mask pixel values `0` and `255` are accepted at the original source resolution. Explicitly validate the masks rather than doing a heuristic threshold. White is foreground, black background. Check mask and original RGB resolution match before resizing.

## Exact implementation steps

### 1. `arguments/__init__.py`

- Add `self._masks = ""` in `ModelParams.__init__`.
- Do not add any other optimization loss weights or depth/geometry flags.
- With no `--masks`, preserve current code path exactly.

### 2. `utils/camera_utils.py` — read aligned RGB and mask

In `loadCam`, after computing the **same** `resolution=(W,H)` as the existing code, load the matching binary mask when `args.masks` is nonempty. Use the already-open `image` (full-res RGB) for construction of the target. Do not alter the image pixels on disk.

**Important border correctness:** Simply downsampling an RGB image that has already been blacked out and separately nearest-neighbor resizing the mask creates black contamination at the object edge. For the new object-mask path, resize **premultiplied color** and **alpha coverage** consistently:

```python
# Conceptual reference, adapt variable names to repo conventions.
# Original image RGB and binary mask have identical H0,W0.
rgb = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
mask_u8 = np.asarray(Image.open(mask_path).convert("L"), dtype=np.uint8)
assert mask_u8.shape == rgb.shape[:2]
assert np.all((mask_u8 == 0) | (mask_u8 == 255))

alpha0 = mask_u8.astype(np.float32) / 255.0
premul0 = rgb * alpha0[..., None]  # discard undefined black-bg RGB

# Both use exactly the same output dimensions and geometric image transform.
# For downsampling use INTER_AREA so edge alpha denotes coverage; for
# upsampling use INTER_LINEAR. No binary threshold after resizing.
interp = cv2.INTER_AREA if resolution[0] < rgb.shape[1] or resolution[1] < rgb.shape[0] else cv2.INTER_LINEAR
premul = cv2.resize(premul0, resolution, interpolation=interp)  # [H,W,3]
alpha = cv2.resize(alpha0, resolution, interpolation=interp)    # [H,W]

# Store as contiguous float32 CHW tensors on Camera's image device.
# premul: [3,H,W]; alpha: [1,H,W], both in [0,1].
```

At equal resolution preserve the arrays exactly, without interpolation. Do **not** use `INTER_NEAREST` for downscaled alpha, and do **not** divide by alpha to recover straight colors: the training target is more stably represented by `premul + (1-alpha)*background`.

If image and mask dimensions differ, emit a clear exception showing both paths and dimensions. If a mask has values other than 0 or 255, emit a clear exception; do not silently convert/threshold. Ensure the mapping is unique in multi-camera datasets.

Pass/store the tensors in the `Camera` instance (e.g. `object_alpha`, `object_rgb_premult`). Use the same `data_device` and dtype as `original_image`. The existing `alpha_mask` is **not** the object-alpha signal for this feature; keep it for legacy train/test exposure behavior.

### 3. `scene/cameras.py` — two optional camera fields only

Give every `Camera` two default-`None` attributes (or equivalent explicit optional constructor parameters), so the training code can distinguish modes:

```python
self.object_alpha = None             # [1,H,W], or None in legacy mode
self.object_rgb_premult = None       # [3,H,W], or None in legacy mode
```

Populate these in `loadCam()` after construction or via two optional constructor parameters. Ensure all shapes, devices and finite values are checked. **Do not modify** the existing `original_image`, `alpha_mask`, `invdepthmap`, `depth_mask`, camera projections, or camera extrinsics.

Do not edit `scene/dataset_readers.py` unless there is a demonstrable input-path incompatibility; the existing `CameraInfo.image_path` and `args.images` should suffice for COLMAP.

### 4. `train.py` — the only optimization change

Keep **one** original call to `gaussian_renderer.render` per training step. Replace only the background/GT compositing behavior and the inappropriate object-mask multiplication. Pseudocode:

```python
object_mode = viewpoint_cam.object_alpha is not None

# Splatfacto-style transparency carving is ACTIVE automatically with --masks.
# For legacy scenes, respect the existing random_background flag exactly.
if object_mode:
    bg = torch.rand((3,), device="cuda")
else:
    bg = torch.rand((3,), device="cuda") if opt.random_background else background

render_pkg = render(
    viewpoint_cam, gaussians, pipe, bg,
    use_trained_exp=dataset.train_test_exp,
    separate_sh=SPARSE_ADAM_AVAILABLE,
)
image, viewspace_point_tensor, visibility_filter, radii = (
    render_pkg["render"], render_pkg["viewspace_points"],
    render_pkg["visibility_filter"], render_pkg["radii"]
)

if object_mode:
    a = viewpoint_cam.object_alpha
    p = viewpoint_cam.object_rgb_premult
    gt_image = p + (1.0 - a) * bg[:, None, None]
    # CRUCIAL: DO NOT do image *= a, or gt_image *= a.
    # Background is supervised by the ordinary full-image RGB loss.
else:
    # Keep the old behavior for legacy datasets (verbatim where feasible).
    if viewpoint_cam.alpha_mask is not None:
        image = image * viewpoint_cam.alpha_mask.cuda()
    gt_image = viewpoint_cam.original_image.cuda()

# UNCHANGED: original L1+SSIM comparison, on full image in object mode.
Ll1 = l1_loss(image, gt_image)
if FUSED_SSIM_AVAILABLE:
    ssim_value = fused_ssim(image.unsqueeze(0), gt_image.unsqueeze(0))
else:
    ssim_value = ssim(image, gt_image)
loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim_value)
```

Retain the existing loss backward, optimizer steps, existing inverse-depth section (if enabled in the checkout), densification, opacity reset, reporting, GUI, checkpointing. **Do not add** separate alpha-render passes or loss terms.

**Special case:** The repo supports `--train_test_exp`, which splits the visible image halves with `alpha_mask`. Do not accidentally defeat that evaluation/training protocol. Preserve the existing split-specific validity masking when it is explicitly enabled, but keep it separate from `object_alpha`: the split mask is a *loss validity* mask, not the foreground mask. Do not apply the object mask as a validity mask. If necessary, reject unsupported mode combinations loudly rather than train incorrectly.

**Exposure consideration:** The original optional trained exposure transform must remain unchanged; do not introduce a second exposure adjustment for the GT.

### 5. Evaluation consistency (minimal)

The normal report uses a fixed `background` (black by default), rather than a random training background. For the new object-mask path, when generating a GT for qualitative images or numerical comparisons, use:

```python
# bg_eval is the actual constant RGB used by evaluation render
reference = camera.object_rgb_premult + (1.0 - camera.object_alpha) * bg_eval[:, None, None]
```

Do not compare an RGB image composited over black against a prediction rendered over white. Keep original metric behavior for legacy datasets. Do not change camera trajectories, test split logic or viewer output.

## Why this specifically addresses the black, oversized border Gaussians

With per-pixel foreground alpha `A` predicted implicitly by standard alpha compositing, color before background is `F = sum_i(T_i * alpha_i * c_i)`. Render against random `b`:

`I_pred(b) = F + (1-A)*b`.

Target for GT foreground `M` and premultiplied object color `P`:

`I_gt(b) = P + (1-M)*b`.

On background `M=0`, `P=0`, so the residual is `F - A*b`. A fixed black Gaussiana cannot fit arbitrary independent background colors; optimization pushes unintended projected occupancy `A` toward zero rather than constructing black splats. This is Splatfacto-style **transparency carving**, accomplished with **ordinary RGB loss and the original CUDA alpha compositing**.

Caveat: it is an occupancy constraint across observed views, not a proof that all 3D covariance scales will be bounded in unseen regions. Do not claim otherwise.

## Tests — implement and actually run

1. **Legacy regression:** without `--masks`, prove that `bg`, GT, RGB loss and existing masking behavior follow the prepatch path. Legacy arguments and CLI still work.
2. **Mask association:** sample with two cameras both containing `0001.png`; confirm `cam_0/0001.png` maps only to `masks/cam_0/0001.png`, and likewise for cam_1. Confirm missing mask fails explicitly.
3. **Mask validation:** reject wrong dimensions and grayscale/intermediate values at source resolution. No auto-thresholding.
4. **Resize edge:** 2x2 or 4x4 synthetic RGB/mask example resized downward; verify `premul + (1-alpha)*b` equals the area-average of the original per-pixel composite within floating-point tolerance. This test catches dark-edge halo bugs.
5. **Background compositing:** use `M=0`, `P=0` everywhere; verify GT equals sampled random `b` for all pixels. Use `M=1`; verify GT equals original foreground RGB. At a fractional coverage pixel verify linear interpolation.
6. **Autograd sanity:** render a tiny scene with an isolated Gaussian in known background pixels; verify the full-image RGB loss backpropagates a **finite, nonzero** gradient to its opacity with random background. In object mode, no post-render object mask may zero these gradients.
7. **Smoke test:** at least a few iterations on a minimal real/synthetic masked camera dataset, with `--disable_viewer`. Confirm no NaNs, no shape/device errors, original CUDA rasterizer works, and one render call per training step.
8. **Optional short A/B check:** same initialization/seed, original black-background objective vs new random-background objective; log a background occupancy diagnostic if easy, **without** modifying the training loss or CUDA. Do not promise quality improvements without measurements.

Do not claim GPU tests passed unless executed. If GPU or dataset is unavailable, clearly separate static/code tests passed from blocked runtime checks.

## Required final Codex report

- Exact files changed and what each change does.
- How matching RGB/mask pairs are resolved, with one example.
- Exact command to train, e.g.:

```bash
python train.py -s /absolute/path/to/DATASET -m /absolute/path/to/OUTPUT --masks masks --disable_viewer
```

- Tests executed and results. Note any that could not run.
- Explicit confirmation: no added depth/geometry/silhouette losses, no CUDA changes, no altered densification/pruning, no unrelated refactors.

## Acceptance conditions

Implementation is accepted only if **a supplied object mask causes compositing over a newly sampled random RGB background during training, both prediction and GT use the exact same background, the full-image original photometric loss is used without object-mask multiplication, binary masks are loaded reliably from every camera, and the original behavior is preserved when no masks are supplied**.
