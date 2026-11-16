# Phase 4 / Phase 5 — Gate 0 repository audit

Date 2026-11-10. Checkout: worktree `~/Joint3DGS/RigEuler3DGS-p45`, branch `phase45`, created from `main` at
`6807d834c0e10ca6eff7cbb2d6255307ff3b1dca` (the commit the specification audits). Nothing upstream was edited:
no dataset file, no Phase 0–3 result, no Phase 1–3 source file.

## 1. Rasterizer build and isolation

| item | finding |
|---|---|
| package | `submodules/diff-gaussian-rasterization` is **vendored in this repository**, not a git submodule (`.gitmodules` lists simple-knn, SIBR, fused-ssim, dinov3, sam3 only) |
| what the environment imports | `rigeuler3dgs` conda env, editable install via `easy-install.pth` → `~/Joint3DGS/RigEuler3DGS/submodules/diff-gaussian-rasterization` (the **main** checkout). Every Phase 0–3 run, from every worktree, loads main's `_C` (built 2026-10-05) |
| consequence | rebuilding in place in main would change the extension under the user's running and future Phase 0–3 jobs. The Phase 4/5 extension is built **in this worktree only** (`python setup.py build_ext --inplace`), and `recovery/__init__.py` puts this checkout's package first on `sys.path` and refuses to run if the loaded module lacks `rasterize_gaussians_rgbd` or comes from another path |
| toolchain | torch 2.11.0+cu128; nvcc `/usr/local/cuda-12.8` (`/usr/local/cuda` → 11.8 is the wrong one); GPU RTX 4070 Laptop, sm_89, 8 GB, shared |
| build command | `cd submodules/diff-gaussian-rasterization && CUDA_HOME=/usr/local/cuda-12.8 PATH=/usr/local/cuda-12.8/bin:$PATH TORCH_CUDA_ARCH_LIST=8.9 python setup.py build_ext --inplace` (47 s) |
| untracked build inputs | `third_party/glm` is git-ignored; the worktree symlinks main's copy. `_C_vis` (from `setup_vis.py`, sources unchanged) is symlinked from main |

## 2. Renderer facts relevant to the new path

* `cuda_rasterizer/forward.cu` `renderCUDA` (≈ l. 281–440): front-to-back loop, `alpha = min(0.99, o·G)`,
  skip `alpha < 1/255`, stop before `T < 1e-4`; legacy depth output is **weighted inverse depth**
  `Σ w_i / z_i` (`expected_invdepth`), not metric depth and not normalised by alpha.
* `cuda_rasterizer/rasterizer_impl.h:53` `ImageState::accum_alpha` is passed to the kernels as `final_T`
  (`rasterizer_impl.cu` forward/backward). It stores the **final transmittance T**, not alpha.
  The new path exposes `alpha = 1 − T_final` as its own output buffer.
* `gaussian_renderer/__init__.py:91-200` `render(...)`: `means3D_override`, `rotations_override`
  (the latter only when `pipe.compute_cov3D_python` is False, l. 132); returns `render`, `radii`,
  `viewspace_points`, `visibility_filter`, and `depth` = the inverse-depth channel above.
* `arguments/__init__.py:106` `antialiasing = False` by default.
* **Pre-existing upstream defect** (found by the Gate 1 oracle, not introduced here):
  `backward.cu` `computeCov2DCUDA`, antialiasing branch (≈ l. 208–246), differentiates
  `det_cov / det_cov_plus_h` with `c_xx, c_yy` that already include the 0.3 low-pass, while the formula it
  cites needs the values before the low-pass. With antialiasing on, scale / rotation gradients of the
  **legacy RGB backward** are off by 0.8–1.6 % (opacity and SH exact). Left untouched (legacy behaviour is
  preserved by specification); Phase 4/5 refuse `antialiasing=True`. Regression-tested in
  `tests/test_rgbd_cuda_forward_backward.py::test_antialiasing_gradient_gap_is_preexisting`.
* Fused SSIM: `utils/loss_utils.py:21` imports `fusedssim` from `diff_gaussian_rasterization._C`, which this
  `_C` does not export; the separately installed `fused_ssim` package is present in the environment.

## 3. Datasets (read only)

Real: `.../Sapien/output/real/final_<scene>_episode_000000`; synthetic:
`.../preparation_8cams/partnet_subset_fixed_render_linear/<scene>`.

| item | convention (verified) |
|---|---|
| RGB | `rgb/<cam>/frame_NNNNNN.png`, 640×480 (real); output frame IDs consecutive from 0 |
| depth | `depth_npy/<cam>/frame_NNNNNN.npy`, float32 **camera-Z in metres**; 0 = missing (34 % of a real fridge view); no NaN observed (`README.txt`: "depth_npy is camera-Z in meters"). `depth_png` is millimetres |
| masks | `mask/<cam>/frame_NNNNNN.png`, uint8 {0, 255}, whole object; loaded strictly by `estimator/masked_observation.py:189` `load_object_mask` (non-binary → refuse) |
| cameras | `metadata/cameras.json`: per camera `fx, fy, cx, cy, width, height`, `pose_matrix_4x4` = `opencv_optical_camera_to_world`, frame `base_link`. `README.txt`: RGB, registered depth and COLMAP share one centred, **undistorted PINHOLE** K → no distortion model |
| time | `metadata/frame_timestamps.csv` (`frame_index, source_frame_index, timestamp_s`); the pipeline's `original_frame` is `frame_index` |
| direction | `metadata/direction/summary.json:recommended_direction` (e.g. red_fridge `FORWARD`) |
| GT (synthetic only) | `metadata/joint_info.json`, `metadata/frame_values.csv`. Never read by Phase 4/5 inference |

## 4. Upstream (Phase 0–3) contract

* `run_bidirectional_all.sh:31-33,477-478`: `--direction auto` uses `recommended_direction` and runs one pass;
  `both` also exists. Phase 4/5 accept only one direction and refuse a run tree with both `forward_joint` and
  `backward_joint` unless `--direction` names one; they never read the other.
* Endpoint: `static_bidi/initial` (forward) or `static_bidi/final` (backward); `point_cloud.ply` + `cfg_args`.
* Phase 2 core: `<dir>_rigid/support_coherence/support.json:members` = row indices into the endpoint PLY
  (`tools/run_joint_phase3.py:230-234`); v6 runs also keep `support_before_refinement.json`.
* Phase 3: `<dir>_joint/summary.json` (`canonical_model`, `canonical_original_frame`,
  `original_frames_processing_order`, `hypotheses`, `joint_types_run`, l. 411-437);
  `<dir>_joint/<type>/geometry.json:geometry_fixed` (`axis`, `axis_point` | `direction`);
  `<dir>_joint/<type>/trajectory.csv` (`processing_index, original_frame, q, qdot, q_phase2_init,
  posterior_sd_q, ...`), row 0 = canonical anchor with `q = 0`.
* **Joint selection status: not machine-readable upstream.** Phase 3 fits both hypotheses. The R/L > 10 rule
  that picks the type lives in `results/rigeuler_tables.py` (evaluation time, documented in `STATUS.md`).
  Phase 4/5 therefore require an explicit `--joint-type`; there is no automatic choice.
* Posing (Phase 3 transport, `tools/run_joint_phase3.py:11-13`): revolute `μ = a + Q(θ)(μ̄ − a)`,
  `R = Q(θ) R̄`; prismatic `μ = μ̄ + s e`. Verified to 1e-9 against `estimator.phase2_model_comparison`
  states on a real fridge run (results/real_gt_axes/render_comparison.py).
* Phase 1–3 render with **frozen DC colour** (`tools/run_iekf_scene.py:153-163`, SH degree 0);
  `gaussian_renderer.render` uses **world-frame SH** for all Gaussians. For moving Gaussians world-frame SH
  does not rotate with the part (spec §3.4); Phase 4/5 offer `world_SH` (parity) and `local_SH`.

## 5. Risks recorded

* Real captures have 2 training cameras (cam_001, cam_002); cam_000 is never used by Phase 0–3 and is not
  used by Phase 4/5 unless a run's own manifest lists it.
* Occluded interior of the real fridge (Phase 1 drift cause) is exactly the "late-visible surface" case of
  Phase 4B; it must be back-transported, not born.
* The shared 8 GB GPU: new stages default to `--frames-on-cpu`, one view on the GPU at a time.
