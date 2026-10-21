# Pipeline code map (2026-10-21)

This map covers every file the production pipeline runs, stage by stage, with file:line
references. It describes the tree as committed. Code that no stage uses was moved to `attic/`,
which stays on disk but is no longer in the repository (see the last section).

```
dataset scene (rgb/ depth_npy/ mask/ colmap/ colmap_final/ pointcloud/ metadata/)
  │
  ├─ static ── train.py ─────────────► <res>/<scene>/static_bidi/{initial,final}/
  │
  ├─ Phase 1 + support (F from initial, B from final)
  │     tools/run_motion_coherence_support.py ──► bidirectional/{forward,backward}/
  ├─ Phase 2 rigid SE(3) + joint reduction
  │     tools/run_rigid_se3_pipeline.py ────────► bidirectional/{forward,backward}_rigid/
  └─ Phase 3 fixed joint, per-frame displacement
        tools/run_joint_phase3.py ──────────────► bidirectional/{forward,backward}_joint/
```

## 0. Drivers

| File | What it does |
|---|---|
| `run_bidirectional_all.sh` | The whole pipeline over a dataset root or one scene folder. It checks the input (`check_scene`, :188), runs the stages static (:346), phase1 (:380), phase2 (:409) and phase3 (:444) in both directions, reuses a stage only through its experiment signature (`stage_sig` :330, `decide` :340, `manifest` :351 -> `tools/stage_signature.py`: reuse / redo own incomplete / REFUSE legacy or differently configured outputs, `FORCE=1` overrides), and forwards the observation options. The default observation is `--observation-mode masked-rgb-expected-quad`; the masks are checked (`tools/check_masks.py`) before any training or deletion; `original` must be explicit, the hard QP needs it. The stage watchdog is `run()` (:291, `STALL_MIN`, default 20 min): a stage that stops reporting progress is killed and retried once. |
| `tools/stage_signature.py` | `compute` (code commit + diff hash, scene, direction, static PLY / cfg_args sha256, observation mode, mask content hash, source listing hashes, config, upstream signature), `check` (exit 0 reuse / 10 absent / 11 own incomplete / 12 refuse with a field diff), `write` (atomic `stage_manifest.json`; `complete` validates the outputs first) |
| `tools/check_masks.py` | Strict mask readiness check (present, unique, native size, binary) used by the driver and `masked_observation.preflight` |
| `scripts/run_object_mask_ab.sh` | FAIR A/B: the same static models copied into two new roots and verified identical (sha256 PLY / cfg_args / cameras.json), then FRESH Phase 1/2/3 in both arms (explicit `original` vs masked), scene by scene, then `tools/compare_object_masks.py`. |
| `tools/compare_object_masks.py` | The A/B table. Checks complete manifests in both arms and that the signatures differ only in the observation block / roots; GT metrics (evaluation only) and GT-free `mask_diagnostics.json`, `stage_resources.json`; exit 1 on any missing file. |
| `tools/repair_report.py` | Machine-readable per-stage report of one or more roots: signature hashes, accepted/rejected steps, MAP split, covariance, depth counts, leakage, wall time, GPU. |
| `tools/benchmark_masked_rgb_observation.py` | One tracked Phase-1 frame per mode, GPU-synchronised timing of renders, JVP/VJP, Fisher, CG, PSD inversions; leakage of the tracked centres AND orientations. |
| `utils/progress.py` | Progress bars on `/dev/tty`, and the `$RIGEULER_HEARTBEAT` beat the watchdog reads. |

## 1. Inputs (wrappers)

| File | Role |
|---|---|
| `tools/make_endpoint_colmap.py:48` | One-image-per-camera endpoint COLMAP model (`colmap/`, `colmap_final/`) built from an all-frames model |
| `scene/dataset_readers.py:261` `readColmapSceneInfo`, `:207` `fetchPly`, `:132` `_load_matching_part_indices` | Read COLMAP and the sparse cloud. GT part labels are attached from `pointcloud/points3D_parts.ply` (initial) or `points3D_parts_final.ply` (final), for evaluation only |
| `scene/dataset_readers.py:238` `_resolve_depths_folder` | Finds `depth_npy/` next to `colmap/` |
| `utils/static_endpoint_scene.py:137` `transport_sparse_gt_parts`, `:310` `build_labeled_frame_static_scene` | Build labelled endpoint scenes |
| `tools/run_iekf_scene.py:65` `load_scene`, `:140` `camera_specs`, `:153` `state_from_gaussians`, `:236` `dataset_rgbd_frames` | Motion-stage inputs: the trained model to Gaussian state, the cameras, and the RGB-D frames (`valid = depth > 0`, :302) in processing order for F/B |
| `utils/sequence_direction.py` | Forward/backward frame order and endpoint-provenance checks |
| `estimator/masked_observation.py:166` `load_object_masks`, `:115` `prepare`, `:337` `preflight` | Strict whole-object mask loading (binary, native size, matched by camera and original frame); masked observation `P = M·I`, `V_used = V ∧ M ∧ finite ∧ depth > 0`; the preflight refuses missing masks before any GPU work |

## 2. Static training (both endpoints)

| File | Role |
|---|---|
| `train.py:1391` `main`, `:744` `training`, `:582` `train_static` | 3DGS training with masks, background carving and object-only depth supervision. `_static_visual_loss` is at :473 and `_validate_carving_options` at :429 |
| `scene/gaussian_model.py:53` `GaussianModel` | Gaussians; `_gt_part_labels` (:87) is inherited through densification (:2487) and written by `save_ply` (:1781) as `gt_part`/`gt_dynamic` |
| `gaussian_renderer/` | The differentiable training renderer |
| `submodules/diff-gaussian-rasterization/` | The training rasterizer (`_C`) and the IEKF operators (`cuda_rasterizer/iekf.cu`) |
| `submodules/diff-gaussian-rasterization/cuda_rasterizer/iekf_visibility.cu` (`_C_vis`) | Visible / accumulated alpha forward, JVP, VJP and the alpha Fisher (`visFisherCUDA`). Built by `setup_vis.py` |

Output: `<scene>/static_bidi/{initial,final}/point_cloud/iteration_30000/point_cloud.ply`, with GT labels.
Carving A/B of two static runs: `scripts/compare_carving_ab.py`.

## 3. Phase 1: free per-Gaussian IEKF + coherence support

The runner is `tools/run_motion_coherence_support.py:80` (`main`).

**Phase 1 trajectories**

- `tools/kpart_pipeline.py:1723` `bootstrap_positions`: the per-frame loop, warm-started from the previous frame. RECURSIVE covariance: P0 once, then each frame's returned posterior is the next frame's `P_prev_post` (`--phase1-covariance-reset` restores the old re-initialisation as a diagnostic); `:1700` `covariance_report` per frame into `phase1_frames.json`; tracked rotations saved as `phase1_centers/rotations_frame_*.npy`. In the masked mode it prepares each frame's observation.
- `estimator/sparse_iekf.py:424` `iekf_frame_step` is one frame:
  - predict: `estimator/process.py:113` `process_predict`, `:136` `error_state_transition`, and `estimator/covariance.py:97` `covariance_predict`, `:299` `project_psd`, `:202` `invert_blocks`;
  - linearize: `estimator/renderer_bridge.py:182` `build_linearization`; the operators are `FrameLinearization.apply_H` (:87), `apply_HT` (:96), `build_block_fisher` (:110), and for alpha `apply_H_alpha` (:140);
  - measurement: `_frame_terms` (:235) -> `estimator/observation_terms.py:73` `camera_terms` (original: Student-t NLL RGB + depth, `estimator/measurement.py:315` `robust_term`, `:295` `student_t_nll`) or `estimator/masked_observation.py:243` `phase1_camera_terms` (masked), `measurement_operators` (:356: RHS, normal operator, Fisher), `evaluate_true_map_energy` (:387, acceptance on the same terms);
  - stopping: `convergence_decision` (:337), relative decrease measured before the state is replaced; every frame logs a `stop` row (`converged`, `trust_radius_floor`, `lm_max`, `iteration_budget`);
  - posterior: `estimator/observability.py:203` `inflate_weak_covariance` raises weak-direction variance by a PSD update (keeps correlations);
  - grouped state: `estimator/supernodes.py:72` `build_supernodes`, `:159` `GroupedObservation` (64-Gaussian groups);
  - step: `estimator/trust_region.py:171` `steihaug_cg`, `:257` `adapt_radius`.
- Optional hard constraint: `estimator/visibility_constraint.py:316` `constrained_step`, `estimator/constrained_qp.py:157` `solve_qp_projection`.

**Support** (`run_motion_coherence_support.py:413` and on)

- `estimator/motion_coherence_support.py`:
  - `:91` `canonical_knn_graph` (192 nodes, 12-NN);
  - `:120` `edge_strain` (`z = log(l / median l)`, C = RMS);
  - `:361` `run_motion_coherence`;
  - `:414` `propagate_to_gaussians`.
- `estimator/coherence_extraction.py`:
  - `:49` `candidate_hierarchy` (single linkage, 2-edge merges);
  - `:220` `cross_validated_rigidity` (5-fold held-out Kabsch);
  - `:289` `select_moving_component`;
  - `:390` `refine_from_core`;
  - `:463` `extract_moving_component` (MOVING_RIGID / STATIC_COMPATIBLE / INCOHERENT_OR_AMBIGUOUS).
- The prediction is written before any GT: `predicted_support.json` (:460).

**GT evaluation (after the prediction)**

- `estimator/optimized_gt.py:46` `find_labeled_model`, `:81` `load_optimized_gt`.
- `estimator/motion_coherence_eval.py`:
  - `:28` `require_prediction_on_disk`;
  - `:95` `classification_metrics`;
  - `:131` `gt_edge_coherence`.
- The Otsu baseline (ablation only): `estimator/moving_support.py`.

**Debug**: `debug/motion_coherence_debug.py`

- `:184` `write_support_3d` (node graph + temporal slider);
- `:529` `write_selector_comparison_3d`;
- `:393` `write_candidate_tree`;
- `:84` `write_finite_strain_heatmap`.

Repairing old pages: `tools/repair_support_3d_slider.py`.

## 4. Phase 2: rigid SE(3) IEKF + joint reduction

The runner is `tools/run_rigid_se3_pipeline.py:75` (`main`). Its steps:

1. `_load_support` (:290): the frozen coherence support. Falling back to Otsu is refused.
2. Kabsch initialisation: `estimator/rigid_part_runtime.py:209` `kabsch_initialize` (runner :310).
3. Rigid window (runner :355):
   - `estimator/rigid_part_runtime.py:327` `run_rigid_window`, with `RigidState` (:60);
   - the rigid map: `estimator/rigid_part_se3.py:185` `state_to_gaussians`, `:223` `rigid_jvp`, `:249` `rigid_vjp`;
   - the IEKF frame: `estimator/joint_iekf.py:190` `frame_step` and `:165` `_energy`, through `estimator/joint_renderer_bridge.py:31` `joint_apply_H` and `:53` `joint_apply_HT`;
   - precision: `estimator/phase2_shared.py:29` `effective_precision` (original), `:92` `effective_precision_masked` (masked), with model error from `estimator/model_error.py:79` `estimate_model_error`.
4. Joint reduction, no renderer (runner :468): `joint_reduction/rigid_trajectory_reduction.py`:
   - `:592` `reduce_trajectory`;
   - `:250` `fit_revolute_axis`;
   - `:204` `fit_prismatic`;
   - `:158` `one_dof_diagnostics`.
5. GT evaluation (runner :487): `joint_reduction/gt_evaluation.py` writes `joint_reduction/gt_evaluation.json`.
6. Debug (runner :503), `debug/rigid_debug.py`:
   - `:421` `write_rigid_phase2_3d`;
   - `:472` `write_rigid_to_joint_3d`;
   - `:600` `write_timelines`;
   - `:710` `write_render_html`.

## 5. Phase 3: fixed joint, per-frame displacement

The runner is `tools/run_joint_phase3.py:194` (`main`).

- `load_initialisation` (:106) reads the Phase 2 joint geometry.
- `estimator/phase2_model_comparison.py:122` `prismatic_state` / `:128` `revolute_state` build the state.
- `estimator/joint_displacement.py:30` `DisplacementOnlyState`: only `(q, qdot)` is estimated. `:103` `geometry_unchanged` is checked bitwise before writing.
- The window: `phase2_model_comparison.py:238` `fit_model` → `estimator/joint_iekf.py:556` `run_window` → `frame_step` (:279). It uses the same measurement model as Phase 2, original or masked (`estimator/phase2_shared.py:92` `measurement_terms`).
- MAP (Phase 2 and 3): `estimator/joint_iekf.py:208` `_FramePrior`: E_total = E_meas + w/2 e^T Λ⁻ e with the exact manifold error (`rigid_part_runtime.py:177` / `joint_displacement.py:96` `prior_error`); the same E_total is solved (b, A) and accepted (pred vs actual); posterior = measurement information + w J^T Λ⁻ J at the accepted state.
- GT: `evaluate_gt` (:160) writes `evaluation_gt/phase3_gt_evaluation.json`.

## 6. Observation modes (Phase 1/2/3)

| Option | Code |
|---|---|
| `--observation-mode original` (explicit control) | Full-frame RGB-D, Student-t NLL: `observation_terms.camera_terms` via `sparse_iekf._frame_terms` and `phase2_shared.measurement_terms` |
| `--observation-mode masked-rgb-expected-quad` (DEFAULT) / `masked-rgb-constant` (ablation) | `estimator/masked_observation.py`. Masked RGB on a known background, fixed W, depth on `V ∧ M`, plus ½κ(A−M)² in the energy, RHS, normal operator and Fisher (`renderer_bridge.apply_H_alpha / apply_HT_alpha / build_alpha_fisher`) |
| `--object-mask-constraint visible-negative-hard` | `estimator/visibility_constraint.py` + `estimator/constrained_qp.py`. The hard negative-ray constraint; it cannot be combined with the masked modes |

Docs:

- `docs/BACKGROUND_MASK_IEKF_AUDIT_2026_10_19.md`
- `docs/MASKED_RGB_FOREGROUND_DEPTH_AUDIT_2026_10_20.md`
- `docs/MASKED_RGB_FOREGROUND_DEPTH_IMPLEMENTATION_2026_10_20.md`
- `docs/MASKED_IEKF_DEFAULT_REPAIR_REPORT_2026_10_21.md` (default mode, MAP acceptance, recursion, signatures, fair A/B)

GT-free per-stage outputs (same definition in every mode): `mask_diagnostics.json` (`estimator/mask_diagnostics.py:103` `MaskDiagnostics`), `stage_resources.json`, Phase 2/3 `model_error_audit.json` (`estimator/model_error.py:165`).

## 7. Result folder layout (per scene)

```
<scene>/static_bidi/{initial,final}/          trained endpoints (GT-labelled PLY)
<scene>/bidirectional/<dir>/                  Phase 1 + support: predicted_support.json, phase1_centers/,
                                              evaluation/, numerical/, debug/*.html, phase1_direction/
<scene>/bidirectional/<dir>_rigid/            Phase 2: support_coherence/, phase2_rigid/, joint_reduction/,
                                              html/, summary.json
<scene>/bidirectional/<dir>_joint/            Phase 3: prismatic/, revolute/, evaluation_gt/, summary.json
<scene>/logs/<scene>_<stage>_<dir>.log        stage logs (the watchdog keeps hung attempts as .hung1/.hung2)
```

## 8. Tests kept

There are 63 tests, all of which pass on this tree, plus the helper module `tests/test_fb_pixel_overlap_core.py` (scene builders the kept tests import). Each touches only code in this map. The main
suites are listed in README §7. They include:

- the renderer operators: `test_unified_cuda_renderer_jvp_vjp`, `test_iekf_cuda_operators`, `test_background_compositing_audit`, `test_visible_alpha_operators`;
- the Phase 1 IEKF: `test_iekf_*`, `test_v2_*`;
- the observation modes: `test_masked_rgb_observation`, `test_constrained_qp`, `test_visibility_constrained_iekf_*`;
- support: `test_motion_coherence_support`;
- Phase 2/3: `test_rigid_se3_phase2`, `test_phase2_parallel_pr`, `test_joint_phase3_runner`;
- static: `test_carving_depth_object_only`, `test_object_mask_transparency_carving`.

## 9. Not in the repository (`attic/`, kept on disk, gitignored)

Moved there, with their paths preserved:

- **Retired experiments:** the fused/unified IEKS, the forward/backward endpoint-correspondence and pixel-overlap diagnostics, the frame-0/frame-01 motion classifiers, flow-matching endpoint transport, the SAM3/DINO affinity fields, and the old drivers (`run_full_pipeline_all.sh`, `run_rigid_se3_all.sh`, `run_support_labelling_all.sh`, `run_complete_joint_debug_all.sh`, `run_real_bidi_phase1.sh`).
- **Withdrawn code:** the `--object-masks` silhouette prototype.
- **Standalone 3DGS scripts:** `render.py`, `metrics.py`, `full_eval.py`, `convert.py`, `lpipsPyTorch`.
- **Old outputs and docs:** the tracked outputs under `results/`, and stale docs.
- **Tests:** 95 tests that only exercise retired code, or that no longer pass against the current code.

`train.py`'s optional DINO/SAM3/dynamic-prior features still launch `tools/extract_dinov3_features.py`, `tools/extract_sam3_features.py`, `tools/extract_sam3_masks.py` and `tools/run_dynamic_prior_comparison.py`. Those four stay so that no `train.py` option breaks.
