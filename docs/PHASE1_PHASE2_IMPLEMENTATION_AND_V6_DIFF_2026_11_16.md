# Phase 1 → Phase 2: exact implementation, and what differs from v6 (2026-11-16)

Runs compared on the 8 real two-camera scenes (orange_suitecase: three cameras):

| label | where | when | code |
|---|---|---|---|
| **v6** | `results_11_10_26/ours_v6_two_cams` (orange: `ours_v6_three_cams`) | 2026-11-07/08 | commit 455e362 + uncommitted diff `f3f4367f…` |
| **draw1**, **draw2** | `results_11_10_26/ours_v8_overnight_real/draw{1,2}` | night of 2026-11-15/16 | p45 = commit 6807d83 + 5 uncommitted files |

Phase 4/5 of the overnight run were still going when this was written. They are not covered here.

---

## 0. Bottom line

1. **The code that ran last night is v6's code.**
   - The hash v6 stored for its code (stage signature `code.uncommitted_diff_sha256 = f3f4367f…`) is reproduced exactly by the diff 455e362 → 6807d83. So 6807d83, the base of p45, is v6's tracked code.
   - The only other differences are 5 uncommitted files of mine. At the settings used last night they change nothing, and section 1.3 proves it on the real outputs.
2. **The inputs are v6's inputs.** Every input hash in the stage signatures is identical: endpoint PLY, cfg_args, RGB, depth, masks, sensor model, P0 prior and every config value. The camera poses come from the pre-registration copy, and frame 1 of the tracks agrees with v6 to 0.0001–0.001 mm (median).
3. **Given the same tracks, the support selection and the Phase 2 refinement are exactly v6's.**
   - Re-running the selection on the saved tracks of all 24 runs (8 scenes × v6/draw1/draw2) with v6's file and with the current file gives the same Gaussians both times, identical to what each run recorded.
   - Re-running the free-space assignment on each run's saved evidence reproduces every recorded label and member list (24/24).
4. **What changes between runs is the Phase 1 tracks.**
   - Phase 1 is not reproducible: the CUDA adjoint and Fisher sums use float `atomicAdd`.
   - No frame ever converges: all 150/150 frames of robot_dog stop on the 12-iteration budget in v6, draw1 and draw2.
   - The differences grow from about 0.001 mm at frame 1 to 1–7 cm at the last frame.
   - The `--phase1-deterministic` option, which removes the atomics, exists and was **off** in v6 and last night.
5. **The selection turns that track noise into completely different supports** in the scenes where no candidate is clearly a rigid moving body: robot_dog, black_laptop, black_box and wooden_box.
   - The winner is the maximum of one number, the world held-out ΔRSS per node-frame.
   - In those scenes the top candidates are near-ties: black_laptop draw1 is 0.01506 vs 0.01504.
   - A large drifting static set can win, and the refinement then recruits more nodes from the whole graph.
   - **v6 was one draw of this same process.** Its supports in those 4 scenes were also weak: selected world eta_cv 0.06 (robot_dog), 0.31 (black_laptop), 0.69 (black_box) and 0.57 (wooden_box). The scenes with a clear moving body (red_fridge, light_blue_box, orange_suitecase, matchbox's selection) behave the same in all runs.

---

## 1. Difference from v6

### 1.1 Code

v6's stage manifests (all 24 Phase 1/2/3 manifests) record `commit 455e362`, `uncommitted_diff_sha256 f3f4367f4ef60b98…`.
`tools/stage_signature.py:109-111` hashes `git diff HEAD -- *.py *.sh *.cu *.cpp *.h *.cuh`, stripped. Files that were new and untracked at the time are not in that diff.

| diff hashed | sha256 (first 16) |
|---|---|
| v6 manifests | `f3f4367f4ef60b98` |
| `git diff --diff-filter=M 455e362 79e7933` (modified files) | `f3f4367f4ef60b98` ✔ |
| `git diff --diff-filter=M 455e362 6807d83` | `f3f4367f4ef60b98` ✔ |

So 79e7933, committed on the branch right after the v6 run, and the merge 6807d83 carry exactly v6's modifications to tracked code. The files that were new then (e.g. `estimator/velocity_decay.py`, `estimator/free_space_support.py`, `utils/camera_selection.py`) are not covered by the hash. For `free_space_support.assign` the outputs prove equality (1.3); `evidence()` was not re-run.

**Uncommitted in p45 (mine).** All are opt-in, and their default values were used last night:

| file | change | value last night | effect at that value |
|---|---|---|---|
| `estimator/coherence_extraction.py` | `min_eta_cv` gate in `select_moving_component` | 0 | `weak = []`, the `if min_eta_cv > 0` branch is skipped |
| same | `recruit_ratio` in `refine_from_core`: `em < recruit_ratio * ei` | 1 | `em < 1.0*ei` is the same comparison as `em < ei` |
| same | `rigid_trim` after refinement | `trim_k` 0 | not called |
| `tools/run_motion_coherence_support.py` | 3 CLI flags + `extraction_summary` in `predicted_support.json` | defaults | output-only addition |
| `estimator/free_space_support.py` | `mode` in `assign`: `members &= prev` only when `intersect` | `replace` | line not executed |
| `tools/run_rigid_se3_pipeline.py` | `--support-refine-mode` (default `replace`) | not passed | — |
| `run_bidirectional_all.sh` | pass-through of the 4 flags | `0 1 0 replace` | `COHARGS` empty, `P2RMODE` not passed, signature unchanged |

Phase 1 tracking code (the IEKF, renderer bridge and process model) has **no** change at all.

The CUDA extension is imported from the main checkout (`~/Joint3DGS/RigEuler3DGS/submodules/diff-gaussian-rasterization`, editable install; `_C.so` 2026-10-05, `_C_vis.so` 2026-10-20). That is the same binary for v6 and last night. p45's submodule has uncommitted Phase 4 edits, but they are not built or imported.

### 1.2 Inputs

The Phase 1 stage signatures of v6 and draw1 differ **only** in:
- `source.dir`, `observation.mask_root`, `endpoint.model_dir`: paths. The overnight run reads `~/real_poses_before_20261112/...`, and its endpoint is a copy of v6's.
- `code.*`: see 1.1.

Identical: `endpoint.ply_sha256`, `endpoint.cfg_args_sha256`, `source.rgb_content_sha256`, `source.depth_content_sha256`, `observation.mask_content_sha256`, `config.SENSOR_SHA`, `config.P0_SHA` and every other config value (BOOT 12, IEKF_SN 64, KNN 12, SNCOUNT 192, FOLDS 5, RIGID_ITERS 8, RGBMS 0.04, DEPMS 0.003, P0V, P1PART canonical, P1ALPHA coupled, P1RGBSUP full, VTAU 0.5, P2REFINE free-space/1). draw1 vs draw2 differ only in `endpoint.model_dir`.

**Camera poses are not in the signature.** The overnight run reads the pre-registration COLMAP/cameras files, symlinked from `robot_base_registration_applied_20261112/backups/<scene>/before`. That this is the frame v6 used is shown by frame 1 of the tracks: v6 vs draw1 agree to a median of 0.00012–0.025 mm, the same as draw1 vs draw2. The registered poses had put frame 1 off by 0.020 vs 0.0047 mean |dx|.

### 1.3 Proof on the real outputs that selection and refinement did not change

**Support selection replay.** Script: `scratchpad/replay_v6_vs_now.py`, CPU only. For each run it:
1. loads the saved `phase1_centers/centers_frame_*.npy` and `canonical_xyz.npy`;
2. rebuilds supernodes (192, seed 0), the graph (k 12) and the extraction (5 folds);
3. does this once with `git show HEAD:estimator/coherence_extraction.py` (v6) and once with the current file;
4. maps the moving nodes to Gaussians and compares them with `predicted_support.json`.

| scene | v6 run | draw1 | draw2 |
|---|---|---|---|
| matchbox | 190 = 190 = rec | 163 = 163 = rec | 135 = 135 = rec |
| wooden_box | 3407 ✔ | 5852 ✔ | 6717 ✔ |
| black_laptop | 1464 ✔ | 18869 ✔ | 1468 ✔ |
| robot_dog | 4333 ✔ | 17896 ✔ | 27300 ✔ |
| orange_suitecase | 1016 ✔ | 1014 ✔ | 204 ✔ |
| light_blue_box | 10408 ✔ | 9594 ✔ | 9815 ✔ |
| black_box | 454 ✔ | 645 ✔ | 21610 ✔ |
| red_fridge | 4493 ✔ | 4489 ✔ | 5774 ✔ |

✔ means: v6 code = current code = recorded, as identical index sets.

**Free-space refinement replay.** `free_space_support.assign` was re-run on `support_refine/round_0.npz` (counts, displacement_max, previous) from each run's `forward_rigid`/`backward_rigid` directory. Labels and members are identical to the recorded ones in 24/24 runs:

| scene | v6: P1 → P2 (accepted) | draw1 | draw2 |
|---|---|---|---|
| matchbox | 190 → 190 (no: < 200 moving) | 163 → 163 (no) | 135 → 135 (no) |
| wooden_box | 3407 → 2649 (+2648 −3406) | 5852 → 2811 (+2811 −5852) | 6717 → 2781 |
| black_laptop | 1464 → 6187 (+6120 −1397) | 18869 → 8383 (−10486) | 1468 → 1468 (no) |
| robot_dog | 4333 → 1088 | 17896 → 17896 (no) | 27300 → 27300 (no) |
| orange_suitecase | 1016 → 812 | 1014 → 793 | 204 → 722 |
| light_blue_box | 10408 → 8448 | 9594 → 8441 | 9815 → 8427 |
| black_box | 454 → 4281 (+4128) | 645 → 4582 (+4544) | 21610 → 21610 (no) |
| red_fridge | 4493 → 3474 | 4489 → 3282 | 5774 → 3094 |

Two things in this table matter:
- In wooden_box and black_box **v6's own** final support is almost entirely Gaussians the refinement *added*: wooden_box removed 3406 of 3407. The coherence support was replaced, not refined.
- When the coherence support is a static blob, its fitted motion is near identity. Fewer than 200 Gaussians then get a "moving" label, the refinement is refused, and the blob goes to Phase 3 unchanged: robot_dog draw1/draw2, black_box draw2, black_laptop draw2.

### 1.4 What the runs produced (from `draw_comparison/draw_comparison.md`)

| scene | v6 P1 → P2 | draw1 P1 → P2 | draw2 P1 → P2 | P2 Jaccard v6~d1 / v6~d2 / d1~d2 | rev axis angle v6~d1 / v6~d2 / d1~d2 |
|---|---|---|---|---|---|
| red_fridge | 4493 → 3474 | 4489 → 3282 | 5774 → 3094 | 0.89 / 0.42 / 0.41 | 0.9° / 5.4° / 5.1° |
| light_blue_box | 10408 → 8448 | 9594 → 8441 | 9815 → 8427 | 0.99 / 0.99 / 0.99 | pri 0.5° / 0.8° / 0.3° |
| orange_suitecase | 1016 → 812 | 1014 → 793 | 204 → 722 | 0.93 / 0.68 / 0.72 | 0.1° / 2.2° / 2.3° |
| wooden_box | 3407 → 2649 | 5852 → 2811 | 6717 → 2781 | 0.94 / 0.94 / 0.99 | 2.0° / 2.0° / 0.1° |
| matchbox | 190 → 190 | 163 → 163 | 135 → 135 | 0.58 / 0.71 / 0.77 | 80° / 17° / 84° |
| black_box | 454 → 4281 | 645 → 4582 | 21610 → 21610 | 0.84 / 0.19 / 0.21 | 49° / 41° / 86° |
| black_laptop | 1464 → 6187 | 18869 → 8383 | 1468 → 1468 | 0.66 / 0.00 / 0.00 | 15° / 40° / 37° |
| robot_dog | 4333 → 1088 | 17896 → 17896 | 27300 → 27300 | 0.00 / 0.04 / 0.66 | 33° / 68° / 44° |

GT error (old frame): red_fridge 6.6° / 7.1° / 12.1°; wooden_box 10.3° / 12.1° / 12.1°; light_blue_box 1.6° / 2.2° / 2.5°.

### 1.5 Why the same code picks different supports: the competing candidates

These are the top 5 qualified candidates by the selection key. The key is the world held-out ΔRSS per node-frame, in m²; the largest wins (`coherence_extraction.py:372`, HEAD). Script: `scratchpad/top_candidates.py`.

**robot_dog.** No candidate has world eta above 0.06 in any run. The tracked motion is tiny (keys around 1e-5 to 1e-4), so this is an argmax over noise.
```
v6     cand 152   10 nodes  key 1.87e-04  eta rel 0.060 world 0.061   <- selected, refined to 30 nodes
       cand 158  183 nodes  key 1.79e-05  eta rel 0.923 world 0.010
draw1  cand  94  115 nodes  key 3.04e-06  eta rel 0.892 world 0.016   <- selected, 115 -> 134 nodes
draw2  cand 162  189 nodes  key 5.71e-05  eta rel 0.933 world 0.003   <- selected: 189 of 192 nodes = the scene
```

**black_laptop.** A near-tie between a ~185-node blob (world eta 0.06–0.09) and small candidates.
```
v6     cand 105   19 nodes  key 6.69e-02  rel 0.127 world 0.314   <- selected
       cand 153  189 nodes  key 4.78e-02  rel 0.909 world 0.090
draw1  cand 153  184 nodes  key 1.506e-02 rel 0.917 world 0.062   <- selected (184 -> 191 nodes)
       cand 154  185 nodes  key 1.504e-02 rel 0.938 world 0.062
       cand  64    9 nodes  key 1.292e-02 rel 0.804 world 0.524   <- the most part-like candidate loses
draw2  cand 102   10 nodes  key 5.07e-02  rel 0.164 world 0.209   <- selected
```

**black_box.**
```
v6     cand  78    5 nodes  key 2.38e-01  rel 0.644 world 0.689   <- selected
draw1  cand  96    8 nodes  key 7.85e-02  rel 0.636 world 0.711   <- selected
draw2  cand 146  180 nodes  key 6.26e-03  rel 0.272 world 0.118   <- selected (blob)
       cand 120    6 nodes  key 4.27e-03  rel 0.867 world 0.710   <- part-like, loses by 1.5x
```

**red_fridge** (a clear body). All top candidates are 0.93–0.96 world eta and overlap. A different winner still gives the same door.

**wooden_box.** The top candidates are 9–43-node pieces with world eta 0.57–0.88. Different seeds, but Phase 2 replaces the support anyway (1.3), so the final supports are 0.94–0.99 alike.

So the rule is fine when one body clearly moves rigidly. Where none does, it has two weaknesses:
- It has **no null hypothesis**: it always returns something.
- Its key rewards **any** world displacement that a rigid fit explains a little better than identity. Phase 1's static side drifts 1–10 cm (median, "drift outside" column), so a big drifting static set competes with the part.

The previously recorded memory "coherence selector has no null hypothesis" is the same finding.

---

## 2. Phase 1, part A: tracking (`tools/run_motion_coherence_support.py` → `bootstrap_positions` → `iekf_frame_step`)

Exact command (red_fridge; the others differ in scene, window, cameras and direction), built by `run_bidirectional_all.sh:598-612`:
```
python tools/run_motion_coherence_support.py --scene final_red_fridge_episode_000000 --direction forward
  --model <OUT>/static_bidi/initial --source ~/real_poses_before_20261112/final_red_fridge_episode_000000
  --frames 189 --cameras cam_001,cam_002 --bootstrap-iters 12 --iekf-supernode-size 64
  --coherence-knn 12 --supernode-count 192 --cv-folds 5 --frames-on-cpu
  --observation-mode masked-rgb-expected-quad --rgb-background-distribution uniform01 --mask-source <src>/mask
  --motion-start-frame 100 --motion-end-frame 288 --frame-timestamps auto
  --sensor-policy ideal --sensor-model <v5>/base/calibration/sensor_model.json
  --rgb-model-sigma 0.04 --depth-model-sigma 0.003 --velocity-tau 0.5
  --p0-geometry <OUT>/static_bidi/initial/geometry_information.npz --p0-velocity-sigma 1.0 0.316227766
  --phase1-partition canonical --alpha-precision coupled --rgb-support full --out <OUT>/bidirectional/forward
```
`--deterministic` is **not** passed. `P1DET` defaults to 0 (`run_bidirectional_all.sh:209-210`).

### 2.1 Call graph
```
main()                                                    run_motion_coherence_support.py:80
├ masked_observation.preflight (masks binary, native size, present)      masked_observation.py:406-434
├ [skipped: --deterministic :196-198, first-return :199-203, --phase1-centers reload :220-257]
├ load_scene(model, source)                                run_iekf_scene.py:65-127
│   Scene → readColmapSceneInfo(<source>/colmap/sparse/0/{images,cameras}.bin)   dataset_readers.py:285
│   GaussianModel.load_ply(point_cloud/iteration_<max>/point_cloud.ply)
├ select_cameras("cam_001,cam_002")                        utils/camera_selection.py:68-75
├ state_from_gaussians → mu, R, scale, opacity, colour = clamp(0.2821·DC+0.5) (DC only); v = ω = 0   run_iekf_scene.py:153-173
├ IEKFConfig()                                             sparse_iekf.py:83-229
├ run_config.window → (start 100, 189 frames); check_canonical_endpoint   run_config.py:57-71, 147-168
├ dataset_rgbd_frames: rgb/<cam>/frame_%06d, depth_npy/<cam>/frame_%06d.npy (valid = dz > 0),
│     metadata/frame_timestamps.csv (strictly increasing)  run_iekf_scene.py:256-382
├ run_config.configure: calibrated SceneNoiseModel (overrides --sensor-policy ideal)   run_config.py:74-99
├ masked_observation.from_args → expected-quad model, b̄ = (0.5,0.5,0.5), masks [100..288]   :469-493
└ kpart_pipeline.bootstrap_positions(...)                  kpart_pipeline.py:1739-1970
    ├ bcfg = replace(cfg, use_supernodes=True, supernode_target_size=64, supernode_seed=0, max_iters=12,
    │                covariance_tangent_transport=False)                    :1766-1770
    ├ build_supernodes(truth.mu, target_size=64, knn=8, seed=0)  (fixed partition, hash checked per frame)
    ├ P0 from npz + velocity sigmas                                         :1813-1834
    └ for each frame: ti = 0 → positions[0] = endpoint means, NOT filtered   :1852-1861
                      ti ≥ 1 → iekf_frame_step(prev posterior, obs_k, cams, dt_k, ...)  sparse_iekf.py:472-1179
                      positions.append(state.mu); rotations.append(state.R)       :1938-1939
├ save phase1_centers/centers_frame_{ORIGINAL:04d}.npy, rotations_*, canonical_xyz.npy, manifests   :378-431
└ positions → coherence (part B)                                                     :493-506
```

### 2.2 State
- **Per Gaussian:** centre μ, rotation R (left perturbation), linear velocity v and angular velocity ω, giving a 12-D error state `[dμ | dφ | dv | dω]` (`process.py:13-15`; boxplus `state.py:589-622`).
- **Covariance:** one 12×12 block per Gaussian, stored float32. No correlation between Gaussians is carried from frame to frame (`covariance.py:394-432`).
- **Appearance** is frozen and asserted unchanged (`sparse_iekf.py:555-556, 1166-1172`).
- **Grouped correction.** The per-frame *correction* is solved in C = N // 64 groups.
  - The groups come from a fixed partition: farthest-point seeding (seed 0) plus at most 25 Lloyd iterations on the endpoint means (`supernodes.py:72-153`).
  - Group centroids are recomputed each frame from the predicted means with CUDA `index_add_` (`sparse_iekf.py:664-672`).
  - The expansion is `dμ_i = dt_c + dθ_c × d_i`, `dφ_i = dθ_c`, with the same form for velocities (`iekf_group.cu:56-86`); `Gᵀ` is `reduceGroupAdjointCUDA` (`:91-119`).
  - The state and prior stay per Gaussian. Only the update lies in range(G).

### 2.3 Process model (`process.py:114-207`, `velocity_decay.py:31-62`, `covariance.py:99-210`)
- **Mean (OU twist):** `μ⁻ = μ + v·h`, `R⁻ = Exp(ω·h)·R`, `v⁻ = φ·v`, `ω⁻ = φ·ω`.
  - `φ = exp(−dt/τ)` and `h = τ(1−φ)`. With dt = 0.04 s (from the timestamps) and τ = 0.5: **φ = 0.92312, h = 0.038442 s**.
- **F:** `F_μv = h·I`, `F_φφ = Exp(ωh)`, `F_φω = h·J_l(ωh)`, `F_vv = F_ωω = φ·I`.
- **Q:** process-noise densities are pos 1e-3, rot 1e-3, vel 1e-1 and ω 1e-1 (`sparse_iekf.py:106-109`). The OU integrals are c_xx = 2.010e-5, c_xv = 7.389e-4 and c_vv = 0.036964. This gives `Q_μμ ≈ 2.41e-7`, `Q_μv ≈ 7.39e-6`, `Q_vv ≈ 3.70e-4` (× I3), with the same values for φ/ω.
- **Predict:** `P⁻ = sym(F P Fᵀ + Q)`, then project_psd (float64; floor 3.81e-6 in correlation form), invert the blocks (Cholesky with a jitter ladder), project_psd again, and cast to float32 (`sparse_iekf.py:587-593`). project_psd falls back to CPU `eigh` on CUDA OOM (`covariance.py:362-372`).
- **P0** is used once, for the first filtered frame:
  - the pose block is `P0_geometry` from `geometry_information.npz`;
  - the velocity sd is **1.0 m/s** and the angular-velocity sd is **√0.1 = 0.316 rad/s**;
  - the cross blocks are 0.
  
  After that, the previous posterior is carried forward (`kpart_pipeline.py:1813-1868`).

### 2.4 Observation model (`masked_observation.py:110-174`, `observation_terms.py:91-156`, `measurement.py:280-335`)
- **Render** (`iekf.cu` `renderFromCacheCUDA` :301-390): front-to-back with `α_i = min(0.99, o_i·exp(−½dᵀQd))`, skipping α < 1/255 and stopping at T < 1e-4, on background b̄ = 0.5.
  - `C = Σ c_i α_i T_i + T·b̄`, `A = Σ α_i T_i`, `D = Σ α_i T_i z_i / (A + 1e-6)`.
- **Target:** `Y = M·I_obs + (1−M)·b̄`, where M is the object mask.
- **RGB**, over the full image with fixed quadratic weights: `E = ½ Σ W_c (Y − C)²` with `W_c = 1/(σ²_cal,c + 0.04²)`. Only the diagonal of the calibrated RGB covariance is used.
- **Depth**, on `(d > 0) ∧ M ∧ A > 0.05`: Student-t with ν = 5. `sd² = max(a + b z² + d z⁴, q²/12) + 0.003²` at the observed z, and the IRLS weight is `w = 6/(5 + r²/sd²)`.
- **Silhouette** (alpha coupled): `E_A = ½ Σ κ (M − A)²` with `κ = (1/12) Σ_c W_c`. This is the exact expectation over a Uniform[0,1]³ background.
- **MAP objective:** `J(x) = Σ_cams (E_rgb + E_D + E_A) + ½ (x ⊟ x⁻)ᵀ Λ⁻ (x ⊟ x⁻)`, with a fresh render for every trial (`sparse_iekf.py:420-469`).
- **No velocity Jacobian.** v and ω are corrected only through the prior's cross-covariance (`renderer_bridge.py:224-231`).
- Masks never select Gaussians. They only replace the background, add the silhouette term and restrict depth.

### 2.5 The per-frame update (`sparse_iekf.py:720-1147`, `trust_region.py`)
Up to **12 trust-region iterations** (`--bootstrap-iters 12`). Each iteration:
1. Linearize: `build_linearization` builds the cache, then a CUB radix sort over (tile, depth).
2. Assemble `b = Hᵀ W r − Λ⁻(x ⊟ x⁻)`. Hᵀ comes from `vjpCUDA` (float **atomicAdd** at `iekf.cu:638, 645-647`) and the alpha VJP (`iekf_visibility.cu:277`).
3. Build the operator: `A v = Λ⁻ v + Hᵀ W H v + 1e-3 v`. Fisher blocks come from `fisherCUDA` (atomicAdd at `iekf.cu:794`) and the alpha Fisher (`iekf_visibility.cu:371`).
4. Reduce to groups: `Gᵀ A G` and `Gᵀ b`. `reduceGroupAdjointCUDA` uses atomicAdd (`iekf_group.cu:112-118`). The group preconditioner is built from 12 probes plus 1e-6·trace/12 jitter.
5. Solve with Steihaug PCG: at most 30 iterations, forcing tolerance η = clip(0.5√(‖g‖/‖g0‖), 1e-6, 0.01), and trust-region metric diag(1/σ²) with σ = 0.05, 0.05, 0.5, 0.5.
6. Trial: `x ⊞ G s`, re-render, then `ρ = actual / predicted`.
7. Adjust the radius (the radius is reset to 1.0 every frame):

   | ρ | decision | radius |
   |---|---|---|
   | < 0.1 | reject | ×0.25 |
   | < 0.25 | accept if the actual decrease > 0 | ×0.5 |
   | ≤ 0.75 | accept | unchanged |
   | > 0.75 | accept | ×2 if the step hit the boundary |

8. Convergence requires a relative decrease < 1e-6 **and** a scaled step < 1e-7. **In practice no frame converges.** `phase1_frames.json` gives `stop_reason = iteration_budget` for 150/150 frames of robot_dog in v6, draw1 and draw2. Each frame ends wherever iteration 12 lands.
9. Posterior: `Λ_post,i = Λ⁻_i + [F_i 0; 0 0]` per Gaussian (groups ignored). Invert in float64, inflate weak directions (whitened Fisher eigenvalue < 1e-2) to at least the prior variance, project_psd, cast to float32. This becomes the next frame's prior.

### 2.6 Saved
- `centers_frame_{ORIG:04d}.npy` (float64, cast from float32) and `rotations_*`, named by original frame. The first file (e.g. 0100) is the endpoint itself.
- `canonical_xyz.npy` is the endpoint geometry, which for backward runs is the final model.
- Coherence reads the tracks in **original chronological order** in both directions.

### 2.7 Where two identical runs diverge
- **Float atomics.** `vjpCUDA` and `fisherCUDA` atomicAdd, the alpha VJP and Fisher, the group reductions and the CUDA `index_add_` for the group centroids. These give about 1e-7 relative differences in b, A, the preconditioner and the posterior.
- **Amplifiers.** Those differences flip discrete decisions:
  - trust-region accept/reject at ρ = 0.1/0.25/0.75;
  - PCG exits;
  - the renderer contributor cutoffs (α < 1/255, T_STOP, α clamp);
  - the depth gate A > 0.05.
  
  Since every frame stops on the iteration budget, not at a converged point, each frame's end state depends on that path. The posterior is carried into the next frame, so the differences compound over 150–188 frames.
- **Measured** (v6 vs draw1, median / p99): about 0.0001–0.025 / 0.01–6 mm at step 1; 0.002–0.3 / 0.1–22 mm at step 2; 1.8–75 mm / 0.1–2 m at the last frame.
- **Fix that exists:** `--phase1-deterministic` (`run_bidirectional_all.sh:52, 158, 209-211`) → `estimator/deterministic_ops.enable()`.
  - It JIT-builds copies of `iekf.cu`, `iekf_group.cu` and `iekf_visibility.cu` with two-pass int64 fixed-point sums.
  - It swaps 6 entry points and turns on `torch.use_deterministic_algorithms`.
  - It was off in v6 and last night. Not covered by it: the CPU fallback of project_psd on OOM, which depends on what else is using the shared card.
- `_C.so` was built 2026-10-05, before a refactor of `iekf.cu` on 2026-10-19 that reads as behaviour-preserving. Both v6 and last night ran the same binary, so this is not a v6 difference.

---

## 3. Phase 1, part B: the support selection (same script, lines ~476-600)

Notation: `ce:N` = `git show HEAD:estimator/coherence_extraction.py` line N; `mcs:N` = `estimator/motion_coherence_support.py`; `sn:N` = `estimator/supernodes.py`; `drv:N` = `tools/run_motion_coherence_support.py`.

### 3.1 Supernodes (drv:493-504, sn:71-153)
- **Count:** `n = min(192, N)`.
- **Seeding:** farthest-point on `mu0.float()`, starting from one index drawn with a CPU generator seeded 0.
- **Refinement:** at most 25 k-means iterations (`cdist().argmin`, ties to the lowest index); empty groups are dropped.
- **Determinism:** the partition depends only on the endpoint, so it is the same in every run.
- **Node trajectory:** the plain mean of the member centres per frame, in float64 (mcs:53-84).

### 3.2 Graph and edge coherence (mcs:91-153, 361-384)
- **kNN graph:** k = 12 on the frame-0 node means, symmetrised (j ∈ kNN(i) or i ∈ kNN(j)).
- **Per edge:** `C_ij = sqrt( mean_t log²( l_ij(t) / median_t l_ij ) )`, the RMS log-strain against the edge's own median length.
- The old two-way normalized cut is still computed and written (`component_stats`, `identification`, `cut`, `nodes.csv` classes), **but the v2 extractor does not use it.**

### 3.3 Candidate hierarchy (ce:49-141, 486-503)
- **Single linkage:** edges are taken in increasing C (`argsort`, not a stable sort), with union-find. Two clusters merge on their **second** connecting edge, or on one edge if either side is a single node.
- **Candidates:** every cluster of 3 or more nodes at the moment it dies is a candidate, with persistence = C(death) − C(birth). Survivors get persistence = inf. Member order is `members[a] + members[b]`.
- **Pool:** remove duplicates, bin by `int(log2 size)`, keep the most persistent `max(2, 48 // bins)` per bin, cap at 48. That is about 42 candidates for 192 nodes.

### 3.4 Rigidity tests (ce:189-282, 506-539)
Each pooled candidate is tested twice: in the **world** frame, and **relative**, in the frame of its complement (Kabsch of the complement, `y = Rᵀ(x − p)`).

`cross_validated_rigidity`:
- **Folds** split **nodes** by their position in the member list (`arange(N) % 5`), not time.
- **For each fold and frame:** Kabsch on the training nodes from frame 0 to frame t, predict the held-out nodes, and accumulate `rss_r += ‖x_t − (R x_0 + p)‖²` and `rss_i += ‖x_t − x_0‖²`.
- **Outputs:**
  - `eta_cv = 1 − rss_r/rss_i`;
  - `delta_rss_cv = rss_i − rss_r`;
  - `delta_rss_cv_per_node_frame = (rss_i − rss_r)/(N·T)`;
  - `delta_bic` (diagnostic only);
  - `relative_motion_ratio = sqrt(rss_i_rel/(N·T)) / median member radius`.

### 3.5 Selection, section 17 (ce:289-383)
- **Qualify:** not degenerate, at least one boundary edge, `relative_motion_ratio > sqrt(eps_f64) ≈ 1.5e-8`, **relative** `eta_cv > 0` and **relative** `delta_rss_cv > 0`. No world condition.
- **Pick:** `best = max(qualified, key = world delta_rss_cv_per_node_frame)` (ce:372). There is no sign requirement and no world-eta requirement; ties go to the first in pool order.
- **NO_COHERENT_MOVING_COMPONENT** is returned only if nothing qualifies.

Relative rigidity is **symmetric**. When the part is the complement, the static side is just as rigid relative to it, so the static side usually qualifies with relative eta ≈ 0.9–0.99. The code says so at ce:362-366. The world key is then the only tie-breaker, and in scenes where the part's tracks are weak, it is a comparison of drift and noise.

### 3.6 Refinement, section 19 (ce:390-456, 557-559)
- Fit `T_t` from the selected core alone (world Kabsch per frame).
- For **every other node in the graph**, with no adjacency limit:
  - `E^M = Σ_t ‖x_i(t) − T_t x_i(0)‖²`;
  - `E^I = Σ_t ‖x_i(t) − x_i(0)‖²`;
  - recruit if `E^M < E^I`, with no margin.
- One pass; `core ∪ recruited`. (`sigma_k`/MAD only feeds a diagnostic, `core_residual_scale`.)
- With a near-identity core motion this is a sign test on noise. In the synthetic check, a static core with 2 mm shared drift recruited 89 of 90 static nodes. That is how a seed of 10–115 nodes becomes 17896–27300 Gaussians.

### 3.7 Output (`predicted_support.json`, drv:551-593)
- Moving nodes are mapped to Gaussians via the supernode id: `dynamic_gaussians`.
- Every node not labelled MOVING_RIGID is written as static, **including incoherent ones**.
- `extraction_summary` (selected candidate with its relative and world rigidity, seed nodes, refinement counts) exists only in runs from 2026-11-14 on.

### 3.8 Fixed numbers that change the outcome
192 nodes, k = 12, 25 k-means iterations with seed 0, min 3 nodes, second-edge merge rule, 48-candidate pool with log2 bins, 5 folds by member order, and the float floor. The decisions themselves are threshold-free, but each is a hard comparison at zero: qualification, the argmax and recruitment.

---

## 4. Phase 2: rigid SE(3) + free-space support refinement (`tools/run_rigid_se3_pipeline.py`)

Command:
```
python tools/run_rigid_se3_pipeline.py ... --rigid-iters 8 [same observation/sensor/window/tau flags]
  --coherence-support <OUT>/bidirectional/forward/predicted_support.json
  --support-refine free-space --support-refine-rounds 1
  --phase1-centers <OUT>/bidirectional/forward/phase1_centers --out <OUT>/bidirectional/forward_rigid
```

### 4.1 Order of execution (line numbers of the working tree; HEAD is 3 lower from line 118 on)
1. **Setup:** mask preflight, scene and cameras, RGB-D frames, calibrated sensor model, and masked expected-quad model with b̄ = 0.5 (:157-261).
2. **Model-error map**, computed **once at frame 0 with every Gaussian at canonical** (:262-275, `model_error.py:79-121`):
   - `v_rgb = max((Y0 − C(G0))², 1e-4)`, `v_d = max((d0 − D(G0))², 1e-6)`, then a 3×3 box blur;
   - added **in variance** in every frame of the window, so any pixel that is wrong at canonical is down-weighted for the whole window. It does not depend on the support.
3. **[1/7]** Reuse the Phase-1 centres (:301-342).
4. **[2/7]** `_load_support` (:718-763): `members = dynamic_gaussians` as stored. It refuses if the status is NO_COHERENT; there is no size or compactness check.
5. **[3/7]** Kabsch on the members' Phase-1 tracks (`rigid_part_runtime.py:237-348`, uniform weights, frame 0 = I).
   - **Only the first twist** `(v[0], ω[0])` reaches the filter: from frame 1 on, `predict()` overwrites the Kabsch poses (:169-181, 378-380).
6. **[4/7]** Rigid IEKF over the window, below.
7. **[4b/7]** Free-space refinement (round 0), below. If it changes the support: Kabsch on the new members, then **a fresh full solve** with no warm start (:426-430). With rounds = 1 there is no second refinement.
8. **[5/7]** `joint_reduction.reduce_trajectory` gives `joint_reduction/{prismatic,revolute}/{geometry.json,trajectory.csv}`. **[6/7]** GT evaluation. **[7/7]** Renders and HTML pages.

### 4.2 Rigid filter (`rigid_part_runtime.py`, `rigid_part_se3.py`, `joint_iekf.frame_step`)
- **State:** one SE(3) per frame plus a twist (v, ω), with a 12-D error state and left retraction on the current frame only.
- **Gaussians:** members are moved by `T_t`. **Non-members stay at canonical in every frame.** They are still rendered, so they occlude and contribute alpha, but they have a zero tangent.
- **P0:** `diag(1e-4·I6, 1·I6)`. Frame 0 is never updated.
- **Process:** the same OU model, with φ = 0.92312 and h = 0.038442 s. F uses `Ad(Exp(hξ))`, `h·J_l(hξ)` and `φ·I`; Q uses `s² = 1` (`--sigma-vdot/--sigma-wdot`).
- **Prior:** `½ eᵀ Λ⁻ e` with `e = [Log(T T⁻⁻¹); ξ − ξ⁻]`, Jacobian by central differences (step 1e-6).
- **Likelihood:** the same masked RGB, Student-t depth and silhouette terms as Phase 1. Each variance gets the model-error map added. The mask M is the **whole object**, static body included.
- **Update:** up to **8 trust-region iterations per frame** (`--rigid-iters 8`), with no convergence tolerance. Each iteration:
  1. builds the dense 12×12 operator from 12 probe columns (each a JVP plus a VJP over both cameras);
  2. runs Steihaug CG (40 iterations, η = 1e-2) with the exact inverse as preconditioner;
  3. re-renders the trial;
  4. applies the same ρ rules as Phase 1, accepting a step only if `E_new < E`.
- **Posterior:** `info = JᵀΛ⁻J + H_info`, inverted.
- **No consistency check:** no frame or camera gating and no chi² test. Nothing checks that the support actually moves as one rigid body; `unexplained_fraction` is never thresholded.

### 4.3 Free-space refinement (`estimator/free_space_support.py`, `_refine_support` :766-809)
- **Inputs:** `mu0` for **all** N Gaussians, the first solve's `R_t, p_t`, and the raw depth (`dz > 0`, not masked). `tau = 0.02 m` (`--support-refine-tau`).
- **classify** (:41-64): project the Gaussian **centre** to the nearest pixel and compare its depth `z` with the observed `d`:

  | code | condition | meaning |
  |---|---|---|
  | CONSISTENT | `|z − d| ≤ τ` | on the observed surface |
  | VIOLATED | `z − d < −τ` | the sensor sees past it: free space |
  | HIDDEN | `z − d > τ` | behind the surface; counted but **never scored** |

- **evidence** (:67-103):
  - Two hypotheses for every Gaussian: static `x0` and moving `x_m = R_t x0 + p_t`.
  - **Camera gate:** a camera counts only if `x0` is CONSISTENT in it at the first processed frame.
  - **Frame gate:** a frame counts only if `‖x_m − x0‖ > 2τ = 4 cm`.
  - The counts are summed over frames × cameras.
- **assign** (HEAD :106-120):
  ```python
  score = 2*(V_static - V_moving) + (C_moving - C_static)
  label = moving if score > 4 else static
  label = unobserved if (C_s+V_s+C_m+V_m) < 6
  label = small_motion if max displacement < 2*tau      # overrides unobserved
  members = previous if small_motion else (label == moving)          # "replace"
  if (#moving < 200): members = previous                              # refinement refused
  ```
  - **replace** (what ran): a "moving" Gaussian becomes a member even if it was not in the coherence support, while static and unobserved ones are removed.
  - `intersect` (uncommitted, opt-in): additionally `members &= previous`.
- **Writes:** `support_refine/round_0.{npz,json}`. `support_coherence/support.json` is replaced by the refined set and the coherence set is kept as `support_before_refinement.json`. The first solve's poses are **not** saved.

### 4.4 What Phase 3 reads
- `support_coherence/support.json` → `members`;
- `joint_reduction/{prismatic,revolute}/{geometry.json,trajectory.csv}`.

Nothing else. Phase 3 keeps both fixed, so any support error passes straight into Phase 3.

### 4.5 How a wrong Phase-1 support propagates
- **Mostly static, or a blob covering most of the scene:**
  - The fitted motion is pulled toward identity. If nothing moves by 4 cm, everything is small_motion, there are 0 < 200 moving labels, and **the wrong support is kept as is**: robot_dog draw1/draw2, black_box draw2, black_laptop draw2.
  - If the compromise motion is larger, the test runs under a wrong `T_t`. HIDDEN costs nothing, so Gaussians whose static position is refuted, such as the door that swung away, can be **added** without the motion explaining them.
- **The refinement cannot:** split the support into two bodies, re-estimate the motion inside the test, revisit small_motion Gaussians, act when there are fewer than 200 moving labels, or check the labels against the re-solved motion.
- **Run-to-run:** Phase 2 has the same float-atomic adjoints (`iekf_apply_HT`, `iekf_vis_apply_HT`) and **no** deterministic switch. `deterministic_ops.enable()` is not called by `run_rigid_se3_pipeline.py`. The free-space labels are deterministic given the poses but flip near `|r| = τ`, `score = 4`, displacement = 2τ and the 200 limit.

---

## 5. What follows

1. **The overnight supports are not a code change.** They are a second and third draw of the v6 process. In 4 of 8 scenes that process has no stable answer, because the selector has no rigid moving body to find there.
2. **Phase 1 can be made repeatable with `--phase1-deterministic`.** It is not yet wired into `run_overnight_real.sh`. Determinism makes the result repeatable, not right. A repeatable wrong support in robot_dog is still wrong.
3. **The selection itself needs a world-frame null hypothesis.** That means rejecting qualified candidates whose world eta_cv or world ΔRSS is not clearly positive, and limiting recruitment to graph neighbours with a margin. The cautious options (`min_eta_cv` on min(relative, world), `recruit_ratio`) do part of this. On the replays (`draw_comparison.md`, column "cautious") they stop robot_dog, black_laptop and wooden_box in every run, instead of returning a different blob each time.

Scripts used (read-only, CPU), in `/tmp/claude-1000/-home-samuelemara-Joint3DGS-RigEuler3DGS/90323ff0-f6c6-454e-a004-5e28e27daa35/scratchpad/`:
- `replay_v6_vs_now.py`: section 1.3, selection;
- an inline `assign` replay: section 1.3, refinement;
- `top_candidates.py`: section 1.5.

Sections 2–4 come from reading the code. The line numbers were checked by reading. The equations and defaults are quoted from the code, but apart from the replays above they were not re-executed.
