# RigEuler3DGS: bidirectional articulated 3DGS with error-state Kalman estimation

This repo recovers the articulation of an object from synchronized, calibrated multiview RGB-D. Two
3D Gaussian Splatting (3DGS) models are trained, one at each end of the motion. Each is then tracked
**independently**, forward from the first frame and backward from the last, through four stages:

```
static 3DGS at t=0 (initial) and t=T (final)
   ├─ forward  (anchored at frame 0):   Phase 1F → support → Phase 2F + joint ID → Phase 3F
   └─ backward (anchored at frame T):   Phase 1B → support → Phase 2B + joint ID → Phase 3B
```

| Stage | What it estimates | Entry point |
|---|---|---|
| Static endpoints | 3DGS at the first and last frame (RGB + metric depth) | `train.py` |
| Phase 1 + support | free per-Gaussian motion (IEKF); static / moving-rigid support from trajectory coherence | `tools/run_motion_coherence_support.py` |
| Phase 2 + joint identification | rigid SE(3) pose of the moving part per frame (RGB-D IEKF); prismatic and revolute joint fits from those poses | `tools/run_rigid_se3_pipeline.py` |
| Phase 3 | joint-specific RGB-D IEKF, one filter per hypothesis (P: s, e; R: θ, e, a) | `tools/run_joint_phase3.py` |

Theory: `assets/Bidirectional_Articulated_3DGS_IEKF_Final_Framework.pdf`.

Conformance of this implementation and where our datasets fail:
[`docs/PIPELINE_VS_THEORY_REPORT.md`](docs/PIPELINE_VS_THEORY_REPORT.md).

**Not part of this pipeline.** Merging the forward and backward Phase-1 models (the fused IEKS,
raw-union experiments, `scripts/run_fused_phase1*.sh`) is not used. The two directions are never
merged: each gives its own estimate.

> **Status (2026-10-16).**
> * Static, Phase 1, Phase 2 and joint identification are the stable pipeline. They produced
>   `results_11_10_26/ours/<scene>/bidirectional/` (tag `bidirectional-batch-2026-10-12`), and the
>   joint is correct in 16 of 20 scene×direction runs.
> * Phase 3 runs in both directions but does not yet improve on the Phase-2 joint (see the report,
>   §6). **Use Phase 2's `joint_reduction/` as the final joint until Phase 3 is re-validated.**
> * No stage chooses the joint type. Both P and R are reported, and `decided_type` is `UNDECIDED`.

---

## 1. Environment

Built for CUDA 12.8, PyTorch 2.9 and an RTX 4070 (8 GB, sm 8.9).

```bash
conda env create -f environment.yml            # creates env "rigeuler3dgs"
conda activate rigeuler3dgs
bash install_after_conda_create.sh "$PWD"      # torch cu128 + the CUDA rasterizer submodules
```

The drivers call the interpreter directly. Override it with `PY=...` if your env lives elsewhere:

```bash
PY=/home/samuelemara/miniconda3/envs/rigeuler3dgs/bin/python
```

## 2. Data layout

Each scene directory (one PartNet-Mobility episode, 8 static cameras, 60 frames) contains:

```
<DATA>/<scene>/
  colmap/sparse/0          cameras + sparse points at frame 0   (static "initial" model)
  colmap_final/sparse/0    the same at the last frame           (static "final" model)
  rgb/cam_XXX/frame_XXXXXX.png
  depth_npy/cam_XXX/frame_XXXXXX.npy    metric Z in metres, 0 = invalid
  mask/cam_XXX/...                      object silhouettes (static training only)
  metadata/, pointcloud/                GT: used ONLY for evaluation
```

The paths used throughout (substitute yours):

```bash
DATA="/media/samuelemara/Extreme SSD/Samuele/Backup_09_10_26/Sapien/output/preparation_8cams/partnet_subset_fixed_render_linear"
RES="/media/samuelemara/Extreme SSD/Samuele/results_11_10_26/ours"
```

The scenes are: `USB_100109 stapler_103111 blade_103706 fridge_10905 storage_45135 laptop_10211
oven_101917 scissor_11100 foldchair_102255 washer_103776`.

## 3. Full pipeline, all scenes, one command

`run_bidirectional_all.sh` runs every stage for both directions:

* It skips any stage whose output already exists, so an interrupted run resumes.
* A failed stage is recorded and the batch moves on.
* `DRY=1` prints the commands without running them.

```bash
cd /home/samuelemara/Joint3DGS/RigEuler3DGS

# everything: static endpoints, Phase 1 + support, Phase 2 + joint ID, Phase 3 -- forward and backward
SRC_ROOT="$DATA" OUT_ROOT="$RES" STAGES="static phase1 phase2 phase3" bash run_bidirectional_all.sh

# the stable part only (the default STAGES)
SRC_ROOT="$DATA" OUT_ROOT="$RES" bash run_bidirectional_all.sh

# one scene, one stage, or a dry run
SCENES="fridge_10905" SRC_ROOT="$DATA" OUT_ROOT="$RES" STAGES="phase3" bash run_bidirectional_all.sh
DRY=1 SRC_ROOT="$DATA" OUT_ROOT="$RES" STAGES="static phase1 phase2 phase3" bash run_bidirectional_all.sh
```

Defaults:

* Static training: `ITERS=30000`.
* Window: `FRAMES=60`, `CAMS=8`.
* Phase 1: `BOOT=12` iterations per frame, `IEKF_SN=64` (grouped-state size).
* Support: `KNN=12`, `SNCOUNT=192` (coherence graph nodes), `FOLDS=5`.
* Phase 2: `RIGID_ITERS=8` iterations per frame.
* Phase 3: `PHASE3_ITERS=5` iterations per frame, `PHASE3_HYP=prismatic,revolute`.
* `FORCE=1` reruns stages whose outputs already exist.

The GPU is an 8 GB card. Don't run two of these at once, and leave other GPU jobs off while static
training runs.

## 4. Stage by stage, one scene, forward and backward

The commands below are exactly what the driver runs. In them:

* `S` is the scene;
* `SRC="$DATA/$S"` is its dataset directory;
* `OUT="$RES/$S"` is its results directory;
* `BIDI="$OUT/bidirectional"` is where the dynamic stages write.

```bash
S=fridge_10905; SRC="$DATA/$S"; OUT="$RES/$S"; BIDI="$OUT/bidirectional"
```

### 4.1 Static endpoints (initial = frame 0, final = frame T)

```bash
$PY train.py -s "$SRC/colmap"       -m "$OUT/static_bidi/initial" --images ../rgb --masks "$SRC/mask" -r 1 \
    --depths depth_npy --sapien_depth_is_metric --iterations 30000 --static_train_iters 30000 \
    --test_iterations 999999 --disable_viewer --no_wandb
$PY train.py -s "$SRC/colmap_final" -m "$OUT/static_bidi/final"   --images ../rgb --masks "$SRC/mask" -r 1 \
    --depths depth_npy --sapien_depth_is_metric --iterations 30000 --static_train_iters 30000 \
    --test_iterations 999999 --disable_viewer --no_wandb
```

### 4.2 Phase 1 (free-motion IEKF) + rigid-coherence support

Forward uses the initial model and processes frames 0 → T; backward uses the final model and
processes T → 0.

```bash
$PY tools/run_motion_coherence_support.py --scene "$S" --direction forward \
    --model "$OUT/static_bidi/initial" --source "$SRC" \
    --frames 60 --cameras 8 --bootstrap-iters 12 --iekf-supernode-size 64 \
    --coherence-knn 12 --supernode-count 192 --cv-folds 5 --frames-on-cpu \
    --out "$BIDI/forward"

$PY tools/run_motion_coherence_support.py --scene "$S" --direction backward \
    --model "$OUT/static_bidi/final" --source "$SRC" \
    --frames 60 --cameras 8 --bootstrap-iters 12 --iekf-supernode-size 64 \
    --coherence-knn 12 --supernode-count 192 --cv-folds 5 --frames-on-cpu \
    --out "$BIDI/backward"
```

Output, in `$BIDI/<dir>/`:

* `phase1_centers/centers_frame_XXXX.npy`: centres only, in original chronological indexing, with a
  `trajectory_manifest.json`;
* `predicted_support.json`: the moving and static Gaussians.

### 4.3 Phase 2 (rigid SE(3) RGB-D IEKF) + joint identification

```bash
$PY tools/run_rigid_se3_pipeline.py --scene "$S" --direction forward \
    --model "$OUT/static_bidi/initial" --source "$SRC" --frames 60 --cameras 8 \
    --rigid-iters 8 --frames-on-cpu \
    --coherence-support "$BIDI/forward/predicted_support.json" \
    --phase1-centers "$BIDI/forward/phase1_centers" --out "$BIDI/forward_rigid"

$PY tools/run_rigid_se3_pipeline.py --scene "$S" --direction backward \
    --model "$OUT/static_bidi/final" --source "$SRC" --frames 60 --cameras 8 \
    --rigid-iters 8 --frames-on-cpu \
    --coherence-support "$BIDI/backward/predicted_support.json" \
    --phase1-centers "$BIDI/backward/phase1_centers" --out "$BIDI/backward_rigid"
```

Output, in `$BIDI/<dir>_rigid/`:

* `phase2_rigid/`
  * `poses_original_order.npy`, `poses_processing_order.npy`, `pose_covariances.npy`
  * `exact_rigidity_gate.json`: the drift must be at machine precision.
* `support_coherence/support.json`: the frozen support.
* `joint_reduction/{prismatic,revolute}/{geometry.json,trajectory.csv}`
  * Both joint hypotheses, in processing order, relative to the anchor frame.
  * Read `axis_radius_over_part_scale`: order 1 means the axis is on the part; a large value means a
    far-axis revolute imitating a translation.
* `joint_reduction/one_dof_diagnostics.json`: `gamma_1dof`.
* `summary.json`: includes Phase 2's own GT check.
* `html/`: the debug pages.

### 4.4 Phase 3 (joint-specific RGB-D IEKF)

Initialised from 4.3's support and joint identification. Both hypotheses are fitted, and backward is
anchored at frame T with the final model.

```bash
$PY tools/run_joint_phase3.py --scene "$S" --direction forward \
    --model "$OUT/static_bidi/initial" --source "$SRC" \
    --phase2-dir "$BIDI/forward_rigid" --frames 60 --cameras 8 --iters 5 \
    --hypotheses prismatic,revolute --frames-on-cpu --evaluate-gt \
    --out "$BIDI/forward_joint"

$PY tools/run_joint_phase3.py --scene "$S" --direction backward \
    --model "$OUT/static_bidi/final" --source "$SRC" \
    --phase2-dir "$BIDI/backward_rigid" --frames 60 --cameras 8 --iters 5 \
    --hypotheses prismatic,revolute --frames-on-cpu --evaluate-gt \
    --out "$BIDI/backward_joint"
```

Output, in `$BIDI/<dir>_joint/`:

* `summary.json`: per hypothesis, the initial and final geometry, coordinate range, energies, and
  `decided_type: UNDECIDED`.
* `{prismatic,revolute}/geometry.json`.
* `{prismatic,revolute}/trajectory.csv`: processing index, original frame, q, q̇, energy per frame.
* `evaluation_gt/phase3_gt_evaluation.json`: only with `--evaluate-gt`, written after inference.

About 2 min per hypothesis per direction at 8 cameras and 2 iterations/frame on fridge (25k
Gaussians); scale linearly with `--iters`.

The window must be the full sequence, `--frames 60`. A shorter backward window would not be anchored
at frame T, and the runner refuses it.

## 5. Reading the results

* **Final joint (stable):** `$BIDI/<dir>_rigid/joint_reduction/`.
  * Both P and R are reported; nothing chooses the type.
  * Prefer the hypothesis with order-1 `axis_radius_over_part_scale` and the much lower pose
    residual. For prismatic parts the energy margin is small (1.15–2.8), so treat type as uncertain.
* **Trust gates, GT-free:** treat a direction as **unresolved** when either holds:
  * `gamma_1dof < 0.91` (in `joint_reduction/one_dof_diagnostics.json`);
  * `support_rigidity.unexplained_fraction_final > 0.5` (in `summary.json`).

  Then use the other direction. On the 10 scenes these gates flag exactly the four failing runs:
  oven F, storage F, washer F and laptop B.

  The thresholds were read off these same runs, so they are post-hoc, not validated. The closest pass
  is scissor F, with γ = 0.924.
* **Forward vs backward:** compare the two estimates in the common world gauge. Don't add their
  information: they reuse the same images, so they are correlated (framework §9).

## 6. Evaluation (GT is never read during inference)

```bash
# Phase 3, if not run with --evaluate-gt: re-run with it (inference outputs are identical)
# Endpoint correspondence diagnosis + GT evaluation (optional, all scenes)
bash scripts/run_fb_endpoint_correspondence_all.sh --results-root "$RES" --dataset-root "$DATA" \
    --tag endpoint_eval_v4 --estimated-joints auto --evaluate-gt --execute --gpu-memory-fraction 0.2
```

Phase 2 writes its own `joint_reduction/gt_evaluation.json` after fitting. The GT scores per run
(support precision and recall, axis and line error, coordinate and pose error) are in
`<scene>/endpoint_correspondence/<tag>/evaluation_gt/`.

## 7. Tests

```bash
$PY tests/test_rigid_se3_phase2.py              # Phase 2: Jacobians, adjoint, transport, CUDA renderer
$PY tests/test_iekf_process_covariance.py       # Phase 1: left-SO(3) process Jacobians, Q
$PY tests/test_motion_coherence_support.py      # support
$PY tests/test_phase2_parallel_pr.py            # P/R joint models (Phase 3 kinematics, FD)
$PY tests/test_joint_phase3_runner.py           # Phase 3 runner: initialisation, direction/window refusals
$PY tests/test_fb_covariance_transport.py       # endpoint transport and ownership
```

## 8. Known limitations

The full list, with file:line references, is in `docs/PIPELINE_VS_THEORY_REPORT.md`.

* **Phase 1:**
  * Covariance is reset every frame.
  * Updates are confined to 64-Gaussian rigid groups that are rebuilt every frame.
  * Only centres are exported.
* **Support:**
  * There is no calibrated "nothing moves" null.
  * AMBIGUOUS is rendered as static.
  * Closed-start forward runs (oven, storage, washer) produce a confident wrong support.
* **Phase 2 and Phase 3:**
  * The process noise is in per-frame units with σ = 1, which makes the prior inert.
  * Phase 3's diag(1) prior lets the axis drift.
  * Two update defects in `estimator/joint_iekf.py` are latent: acceptance is likelihood-only, and
    the information transport has its sides swapped.
* **No type decision, no "unresolved" output** on the standard path.
