# Repository audit — unified bidirectional 3DGS / hierarchical IEKF

Milestone **M0** of `CODEX_IMPLEMENT_UNIFIED_BIDIRECTIONAL_3DGS_IEKF_PIPELINE.md`,
section 1. Everything below was read out of the current working tree or
computed from real artifacts on disk. Where the specification names a module
that does not exist, the actual equivalent is named instead and the
difference is stated.

Audit date: 2026-10-12. Baseline `git status --short`: clean except a
pre-existing dirty `submodules/sam3` that is not mine and is not touched.

---

## 1. Discovered production call graph

The specification's audit template asks for these slots. Filled from the
real tree:

| Slot | Actual location |
|---|---|
| `FORWARD_TOP_LEVEL_ENTRY` | `tools/run_rigid_se3_pipeline.py::main` |
| `PHASE1_ENTRY` | `tools/run_motion_coherence_support.py::main` |
| `PHASE1_FRAME_ITERATOR` | `tools/run_iekf_scene.py::dataset_rgbd_frames` (now takes `direction=`) |
| `PHASE1_IEKF_STEP` | `estimator/sparse_iekf.py::iekf_frame_step` (line 321) |
| `PHASE1_BOOTSTRAP` | `tools/kpart_pipeline.py::bootstrap_positions` (line 1700) |
| `PHASE1_RENDER_FORWARD` | `estimator/renderer_bridge.py::build_linearization` (line 125) |
| `PHASE1_JVP/VJP` | inside `FrameLinearization` (`estimator/renderer_bridge.py`, line 70) |
| `COHERENCE_ENTRY` | `estimator/motion_coherence_support.py`, driven from `tools/run_motion_coherence_support.py` |
| `COHERENCE_SUPPORT_ARTIFACT` | `<out>/predicted_support.json`, `<out>/numerical/{nodes,edges}.csv`, `components.json` |
| `RIGID_SE3_ENTRY` | `estimator/rigid_part_runtime.py::run_rigid_window` (line 327) |
| `RIGID_SE3_STATE` | `estimator/rigid_part_se3.py::RigidPart` + `RIGID_ERROR_DIM = 12`, `TAU=0:6`, `XI=6:12` |
| `RIGID_SE3_JVP/VJP` | `estimator/rigid_part_se3.py::rigid_jvp` (223), `rigid_vjp` (249) |
| `RIGID_SE3_OUTPUT_POSES` | `<out>/phase2_rigid/poses.npy` (original chronological order) |
| `JOINT_REDUCTION_ENTRY` | `joint_reduction/rigid_trajectory_reduction.py::reduce_trajectory` |
| `STATIC_MODEL_LOADER` | `tools/run_iekf_scene.py::load_scene` → `scene.Scene` + `GaussianModel` |
| `MEASUREMENT/INFORMATION` | `estimator/measurement.py` (`build_precision`, `rgbd_residual`, `student_t_weight`, `channel_sigma`) |
| `TRUST REGION` | `estimator/trust_region.py`; acceptance inside `run_rigid_window` |

## 2. Phases 3 and 4 already exist in substance

This is the most consequential finding of the audit, and it changes the
implementation plan for sections 7 and 8.

The specification asks for a new `joint_geometry_iekf.py` (Phase 3,
time-invariant geometry with latent `q(t)`) and `joint_motion_iekf.py`
(Phase 4, geometry frozen, estimate `q`, `qdot`). The repository already
contains that state space, built for
`CODEX_IMPLEMENT_PHASE2_RIGID_SCREW_IEKF.md`:

```
estimator/joint_state.py       JointPart: members, centroid, length,
                               zeta [6] = [u, omega]  <- ONE screw per part,
                               persistent across every frame,
                               q [T], qdot [T]        <- latent coordinate
                               error state dq, dqdot, dtheta_J in R^6
estimator/joint_iekf.py        JointIEKFConfig, frame_step (165),
                               run_window (332), apply_sign_gauge (88),
                               sign_jacobian (116)
estimator/joint_manifold.py    se3_exp/log, se3_left_jacobian, se3_adjoint,
                               revolute_axis, tangent_basis, zeta_to_xi
estimator/joint_kinematics.py  joint-parameterised Gaussian kinematics
estimator/joint_process.py     process model for the joint state
estimator/joint_renderer_bridge.py  joint state against the SAME renderer
estimator/joint_score_blocks.py     information-form blocks
estimator/joint_runtime.py          window driver
```

`joint_state.py` states the invariant the specification asks for, verbatim:
*"The geometry `zeta_k` is ONE screw per part, persistent across every frame:
there is deliberately no `xi_t`, no per-frame `T_t` and no per-Gaussian twist
in the production state. That absence is what makes the parameterisation
rigid by construction."*

**Consequence.** Phase 3 should be built by *parameterising* this existing
filter with independent revolute and prismatic hypotheses, not by writing a
new one. Phase 4 is the same filter with `dtheta_J` held at zero — a masked
error state, not a new module. Writing the two new files the layout suggests
would duplicate a working, tested estimator, which section 10 explicitly
warns against and section 0 forbids.

What is genuinely missing for Phase 3/4 as specified: running revolute and
prismatic as *separate competing hypotheses* with a comparison, and a frozen-
geometry mode. Both are additions to the existing driver.

## 3. The 5,131-Gaussian "support provenance bug" — NOT a bug

Section 1 states: *"A previous rigid run used 5,131 Gaussians, matching the
broad dynamic fraction and suggesting a support provenance bug."*

Audited by rebuilding the deterministic partition from
`phase1_centers/canonical_xyz.npy` with `build_supernodes(num_supernodes=192,
seed=0)` and intersecting it with the exported index sets:

```
rebuilt partition: 192 supernodes over 24582 gaussians
  gaussians in the 13 exported 'dynamic_supernodes' : 5131
  gaussians in the 35 DYNAMIC-labelled supernodes   : 10448
  gaussians actually exported as dynamic_gaussians  : 5131

  exported == 13-node set ?  True   (overlap 5131)
  exported == 35-node set ?  False  (overlap 5099)
```

The exported set is **exactly** the 13-node high-confidence core. The broad
spectral component would have been 10,448 Gaussians and was not used.

The inference in the specification came from a coincidence:
`5131 / 24582 = 20.9%`, which looks like a "broad dynamic fraction", but
`predicted_dynamic_fraction` is *computed from the exported set itself*, so
the agreement is definitional rather than evidential. The supernodes are
unequal in size; the 13 core nodes hold 395 Gaussians each against an average
of 128, because the moving part is densely modelled.

`tools/run_rigid_se3_pipeline.py` line 592 consumes `rec["dynamic_gaussians"]`
and guards on `num_gaussians` matching the current model, so the rigid stage
received the core.

**One real oddity, recorded not fixed.** Of the 13 exported dynamic
supernodes, `nodes.csv` labels 12 `DYNAMIC` and 1 `STATIC`. The MOVING_RIGID
extraction selected a node the spectral labelling calls static. That is the
merge-candidate logic, not obviously wrong, but it should be understood
before the unified Phase 1 reuses this extractor.

## 4. Artifact formats, as they actually are

`predicted_support.json` keys:
`method, status, num_supernodes, num_gaussians, static_supernodes,
dynamic_supernodes, static_gaussians, dynamic_gaussians,
predicted_dynamic_fraction, graph_config, component_stats, identification,
cut, note`.

There is **no** `members` key — code written against one gets an empty list
silently. The Gaussian index lists are `static_gaussians` /
`dynamic_gaussians`.

`numerical/nodes.csv` columns: `node, supernode, label, predicted_class,
C_in, C_out, margin, ambiguous`. It carries **no per-Gaussian membership**;
the Gaussian→supernode map is not persisted and must be rebuilt
deterministically from `canonical_xyz.npy`, `num_supernodes` and `seed`
(recorded in `graph_config`).

`phase1_centers/` (written by the bidirectional work):
`canonical_xyz.npy`, `original_gaussian_indices.npy`,
`centers_frame_NNNN.npy` indexed by **original** frame,
`trajectory_manifest.json` with `direction`, `canonical_endpoint`,
`canonical_original_frame`, `trajectory_order`.

Verified on washer: `canonical_xyz` equals `centers_frame_0059` bitwise and
differs from `centers_frame_0000` (max 0.83), i.e. the canonical geometry is
the final endpoint and is not a trajectory frame.

## 5. Data layout, verified

```
DATASET_ROOT/<scene>/colmap/sparse/0        registers frame_000000  (8 cams)
DATASET_ROOT/<scene>/colmap_final/sparse/0  registers frame_000059  (8 cams)
DATASET_ROOT/<scene>/{rgb,mask,depth_npy}/cam_XXX/frame_NNNNNN.*
DATASET_ROOT/<scene>/pointcloud/points3D_parts.ply        GT, initial
DATASET_ROOT/<scene>/pointcloud/points3D_parts_final.ply  GT, final
RESULT_ROOT/<scene>/static_bidi/{initial,final}/point_cloud/iteration_30000/point_cloud.ply
```

- `colmap/images/` and `colmap_final/images/` are **empty or single-frame**;
  the real frames are in `rgb/`, reached with `--images ../rgb`. The
  symlinks were dropped by exFAT.
- Camera extrinsics in `colmap` and `colmap_final` are **bitwise identical**
  (`max |dq| = max |dt| = 0.0` on washer). The world/camera gauge is shared,
  so no camera alignment is needed between endpoints — this is the section 20
  equivalence check of the previous specification, passing.
- Washer counts: initial model **21,338** Gaussians, final **24,582**. No row
  correspondence; they are separate trainings of different COLMAP models.

### Inputs that do NOT exist yet

- **No forward Phase-1 output exists for washer.** Searching
  `RESULT_ROOT/washer_103776` for `predicted_support.json` finds only
  `bidirectional/backward/`. The specification's `--forward-phase1` argument
  therefore has no value to take for this scene; it must be produced by
  `run_bidirectional_all.sh` (stage `phase1`, direction `forward`) first.
  Per section 10, no example command is given with an invented path.
- `stapler_103111/coherence/` exists but is from the earlier single-direction
  overnight run, not the bidirectional layout.

## 6. Renderer / observation contract

One renderer, reached through `estimator/renderer_bridge.py`:

- `build_linearization(state, cameras, *, background=None, eps_depth=1e-6,
  antialiasing=False, debug=False) -> FrameLinearization`. Appearance is
  frozen — colours, opacity and scales come from the state and are never
  updated by the estimator. The SO(3) state meets the rasterizer's quaternion
  in exactly one place, inside this call.
- `FrameLinearization` carries the per-camera JVP/VJP; it is an **operator**
  rebuilt at the current state, not a cached numeric Jacobian. This is what
  lets all four motion models share `H_G` as section 0.6 requires.
- Residual/precision: `estimator/measurement.py::rgbd_residual`,
  `build_precision`, `channel_sigma`, `student_t_weight`, `depth_validity`.
- The rasterizer returns `color, radii, invdepth`. There is **no alpha
  output**. Accumulated alpha is obtained exactly by rendering an all-ones
  feature over a black background through `render_feature_field`
  (front-to-back compositing gives `A(p)` identically); this is the method
  already used in `utils/carving_diagnostics.py::render_projected_alpha` and
  is sanctioned by the previous specification's section 31.
- The submodule is a **custom fork** carrying `_RasterizeGaussianFeatures`,
  on which the semantic / DINO / SAM3 feature fields depend. It must not be
  swapped for an external fork.

## 7. Rigid Phase-2 contract

```
RigidPart            R, p, v, omega per frame; q/qdot read-only diagnostics
RIGID_ERROR_DIM = 12, TAU = 0:6, XI = 6:12
run_rigid_window(state, mu0, R0, observations, linearize, precision_fn,
                 times=None, cfg=None, label="rigid") -> dict
```

Frame 0 is the canonical gauge and carries no information: `T_0 = I`
whatever the motion is, so the driver deliberately does not update on it.
**For a backward run the canonical frame is original frame 59**, which the
bidirectional work handles by solving in processing order and persisting
chronologically.

Convention: `boxplus` is `R+ = Exp(hat(dphi)) @ R` — LEFT/spatial
perturbation. A right-multiplied fixed gauge transform leaves the pose
covariance block unchanged; inversion requires adjoint transport.

## 8. API limitations that constrain the new work

1. **No per-Gaussian supernode map is persisted.** Any stage needing it must
   rebuild deterministically from `canonical_xyz.npy` + `graph_config`.
2. **No Gaussian correspondence between endpoints.** 21,338 vs 24,582 rows,
   different trainings. Any two-anchor state must carry explicit provenance.
3. **`dataset_rgbd_frames` loads all frames into memory** before solving;
   `--frames-on-cpu` keeps them off the GPU. 60 frames x 8 cameras of
   640x480 RGB+depth is the memory floor for the unified filter.
4. **Supernode sizing has two independent consumers** —
   `IEKFConfig.supernode_target_size` (Phase-1 grouped state) and the
   coherence graph node count. They shared one flag until recently; conflating
   them again makes Phase 1 about 60x slower with correct-looking output.
5. **The COLMAP reader writes `points3D.ply` beside the model it opens**,
   i.e. into the dataset tree. `utils/readonly_source.py` stages the model
   into the run's own cache to avoid this; it is built and tested but not yet
   wired into any runner.
6. **`run_rigid_window` assumes one rigid body.** Multi-body support (the
   foldchair case) is not represented.

## 9. Baseline regression surface

These must keep behaving identically; they are the section 11.A gate.

```
tests/test_motion_coherence_support.py          41 checks
tests/test_rigid_se3_phase2.py                  RSE3-1..9 + P0 gates
tests/test_joint_reduction.py
tests/test_bidirectional_sequence_direction.py  54 checks
tests/test_bidirectional_time_and_source.py     63 checks
tests/test_transparency_carving.py              80 checks
tests/test_object_mask_transparency_carving.py  47 checks
tests/test_static_endpoint_cli.py
```

Static 3DGS training and the rasterizer submodule are out of scope and must
not change.
