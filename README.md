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
| Phase 3 | per-frame joint displacement q_k (and q̇_k) by RGB-D IEKF; joint type and geometry fixed from joint identification | `tools/run_joint_phase3.py` |

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
> * Phase 3 keeps the identified joint fixed and refines only the per-frame displacement. On fridge
>   it reproduces the Phase-2 track to within 0.006 rad (report, §6).
> * No stage chooses the joint type. Joint identification reports both P and R; give Phase 3 the type
>   to run with `--hypotheses`.

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

Each scene directory contains the following. The synthetic PartNet-Mobility episodes have 8 static
cameras and 60 frames; any number of cameras and frames is accepted.

```
<DATA>/<scene>/
  colmap/sparse/0          cameras + sparse points at frame 0   (static "initial" model)
  colmap_final/sparse/0    the same at the last frame           (static "final" model)
  rgb/cam_XXX/frame_XXXXXX.png
  depth_npy/cam_XXX/frame_XXXXXX.npy    metric Z in metres, 0 = invalid
  mask/cam_XXX/...                      object silhouettes (static training only)
  metadata/, pointcloud/                GT: used ONLY for evaluation
```

### 2.1 Set the input and output paths (copy and edit this block first)

**Input** (`--input`, or the environment variable `SRC_ROOT`) can be **either**:

* **one scene folder**, the folder that itself contains `rgb/`, `depth_npy/`, `colmap/` and
  `colmap_final/`. For example:
  `/media/samuelemara/Extreme SSD/Samuele/Backup_09_10_26/Sapien/output/real/fridge_plastic_20261016_163314`.
  Only that scene is run.
* **a dataset root**, a folder whose sub-folders are scene folders, such as
  `$DATA/washer_103776/rgb/`, `$DATA/washer_103776/colmap/`, …. Every scene folder found there is
  run, or only those listed with `--scenes`.

**Output** (`--output`, or `OUT_ROOT`): the **results root**. Each scene is written to
`<output>/<scene>/`, for example `/media/samuelemara/Extreme SSD/Samuele/results_11_10_26/ours`.

* It does not need to be inside the dataset or the repository.
* If you pass `<root>/<scene>` for a single scene, the driver recognises it and does not nest the
  scene twice.
* To reuse previously trained static models, point it at the existing results root. A new, empty
  root needs the `static` stage before motion estimation.

Run the following in **the same terminal** in which you will run the pipeline:

```bash
cd /home/samuelemara/Joint3DGS/RigEuler3DGS
conda activate rigeuler3dgs

# INPUT: a dataset root (its sub-folders are scenes) ...
DATA="/media/samuelemara/Extreme SSD/Samuele/Backup_09_10_26/Sapien/output/preparation_8cams/partnet_subset_fixed_render_linear"
# ... or one scene folder, used directly as --input
SCENE_DIR="/media/samuelemara/Extreme SSD/Samuele/Backup_09_10_26/Sapien/output/real/fridge_plastic_20261016_163314"

# OUTPUT: directory whose immediate children will be scene names.
RES="/media/samuelemara/Extreme SSD/Samuele/results_11_10_26/ours"

# Python executable used by the per-stage commands below.
PY="/home/samuelemara/miniconda3/envs/rigeuler3dgs/bin/python"

# Example single scene and its fully resolved paths.
S="washer_103776"
SRC="$DATA/$S"               # input for this scene
OUT="$RES/$S"                # output for this scene
BIDI="$OUT/bidirectional"   # Phase 1/2/3 output for this scene

# Check the input before launching a GPU job.
if test -d "$SRC/colmap/sparse/0" && test -d "$SRC/colmap_final/sparse/0" \
    && test -d "$SRC/rgb" && test -d "$SRC/depth_npy"; then
  echo "Input layout OK: $SRC"
else
  echo "ERROR: missing required input directories under $SRC; do not start the run." >&2
fi

printf 'INPUT ROOT : %s\nINPUT SCENE: %s\nOUTPUT ROOT: %s\nOUTPUT SCENE: %s\n' \
  "$DATA" "$SRC" "$RES" "$OUT"
```

**Spaces in paths are supported when you keep the double quotes.** Do not write an unquoted path such as `/media/samuelemara/Extreme SSD/...` in a shell command: Bash would treat it as multiple arguments. The variables are shell-local and remain defined in that terminal; opening a new terminal requires setting them again (or sourcing a configuration file you created).

The scene names used in the original dataset are: `USB_100109 stapler_103111 blade_103706 fridge_10905 storage_45135 laptop_10211 oven_101917 scissor_11100 foldchair_102255 washer_103776`.

### 2.2 How path arguments map to scripts

| Interface | Input location | Output location | What is appended automatically? |
|---|---|---|---|
| **Pipeline driver** `run_bidirectional_all.sh` | `--input "$DATA"` (root) or `--input "$DATA/<scene>"` (one scene); or `SRC_ROOT=...` | `--output "$RES"`; or `OUT_ROOT=...` | `/<scene>` for each selected scene |
| **Phase 1 CLI** `run_motion_coherence_support.py` | `--source "$SRC"`, `--model "$OUT/static_bidi/initial"` (F) or `--model "$OUT/static_bidi/final"` (B) | `--out "$BIDI/forward"` or `.../backward` | No scene folder added to `--out` |
| **Phase 2 CLI** `run_rigid_se3_pipeline.py` | `--source "$SRC"`, `--model ...`, plus saved support/centers | `--out "$BIDI/forward_rigid"` or `.../backward_rigid` | No scene folder added |
| **Phase 3 CLI** `run_joint_phase3.py` | `--source "$SRC"`, `--model ...`, `--phase2-dir ...` | `--out "$BIDI/forward_joint"` or `.../backward_joint` | No scene folder added |
| **Optional endpoint diagnostic** `run_fb_endpoint_correspondence_all.sh` | `--dataset-root "$DATA"` or one scene folder | `--results-root "$RES"` | Uses each scene's result tree; diagnostic tag determines subdirectory |

The driver accepts `--input`/`--output` (or `-i`/`-o`), plus `--scenes`, `--stages`, `--frames`,
`--cameras`, `--dry-run`, `--force` and `--help`. The environment variables `SRC_ROOT`, `OUT_ROOT`,
`SCENES`, `STAGES`, `FRAMES`, `CAMS`, `DRY` and `FORCE` do the same; the command-line options win. The
individual Python tools use explicit `--source` (one scene folder), `--model` and `--out` flags.

### 2.3 Input check (automatic, before any GPU work)

For every scene, the driver checks:

* both `colmap/sparse/0` and `colmap_final/sparse/0` exist;
* `rgb/cam_*` and `depth_npy/cam_*` exist with the same number of frames per camera;
* the frame files are numbered `frame_000000, frame_000001, …` without gaps.

A scene that fails is skipped with the reason printed, and the rest of the batch runs. With the
defaults `--frames auto --cameras auto`, every frame and every camera of the scene is used. The
backward run is anchored at the scene's last frame, so it needs the full sequence.

**The real captures do not pass this check yet.** For example,
`real/fridge_plastic_20261016_163314` has 3 cameras and 328 frames per camera, stored as
`frame_000000, frame_000002, …, frame_000654` (every second index). The RGB-D loader reads frames
consecutively, so the run is refused with:

```text
!! input check failed -- skipping: rgb/cam_000: 328 frames numbered 0, 2, 4, ... 654, not 0..327; ...
```

To run it, the frames (and the image names inside the COLMAP models) must be renumbered to be
contiguous, or the loader must learn to read the frame list. It also has no joint GT, so
`--evaluate-gt` reports `GT_UNAVAILABLE`.

For reference, a completed Washer run under the example `RES` root is organised as follows:

```text
$RES/washer_103776/
  static_bidi/
    initial/                  # trained at frame 0
    final/                    # trained at frame T
  bidirectional/
    forward/                  # Phase 1F + support
    backward/                 # Phase 1B + support
    forward_rigid/            # Phase 2F + joint identification
    backward_rigid/           # Phase 2B + joint identification
    forward_joint/            # Phase 3F
    backward_joint/           # Phase 3B
  endpoint_correspondence/   # optional separate diagnostic, if run
```

## 3. Full pipeline, one command (one scene folder or a whole dataset)

`run_bidirectional_all.sh` runs every stage for both directions:

* It skips any stage whose output already exists, so an interrupted run resumes.
* A failed stage is recorded and the batch moves on.
* `DRY=1` prints the commands without running them.

The commands below assume you already ran **Section 2.1** to define `DATA` and `RES` in this shell.
Always look at a `--dry-run` first: it prints the resolved input root, scenes, output root, the input
check, and every command, without running anything.

**One scene folder** (any folder with `rgb/`, `depth_npy/`, `colmap/`, `colmap_final/`), all stages:

```bash
cd /home/samuelemara/Joint3DGS/RigEuler3DGS

bash run_bidirectional_all.sh \
  --input  "/media/samuelemara/Extreme SSD/Samuele/Backup_09_10_26/Sapien/output/preparation_8cams/partnet_subset_fixed_render_linear/washer_103776" \
  --output "/media/samuelemara/Extreme SSD/Samuele/results_11_10_26/ours" \
  --stages "static phase1 phase2 phase3" --dry-run

# then the same command without --dry-run
```

**A whole dataset root**, every scene found in it:

```bash
bash run_bidirectional_all.sh --input "$DATA" --output "$RES" \
  --stages "static phase1 phase2 phase3" --dry-run
bash run_bidirectional_all.sh --input "$DATA" --output "$RES" --stages "static phase1 phase2 phase3"
```

**Existing results root with trained static models** (resume the stable stages, the default
`--stages`):

```bash
bash run_bidirectional_all.sh --input "$DATA" --output "$RES"
```

**Some scenes of a root**, or **only Phase 3** for already processed scenes (needs their Phase 2
outputs under the output root):

```bash
bash run_bidirectional_all.sh -i "$DATA" -o "$RES" --scenes "washer_103776 oven_101917" \
  --stages "static phase1 phase2 phase3"
bash run_bidirectional_all.sh -i "$DATA/fridge_10905" -o "$RES" --stages phase3
```

The same runs with environment variables, as before:

```bash
SCENES="washer_103776" SRC_ROOT="$DATA" OUT_ROOT="$RES" STAGES="static phase1 phase2 phase3" \
  bash run_bidirectional_all.sh
```

**Paths:**

* Keep the double quotes, so path segments such as `Extreme SSD` survive.
* A trailing `/` is ignored.
* With a dataset root, `--scenes` names its sub-folders. With a scene folder, `--scenes` may be
  omitted; if given, it must name that folder.

Defaults:

* Static training: `ITERS=30000`.
* Window: `--frames auto`, `--cameras auto`, i.e. every frame and camera of the scene (60 and 8 on
  the synthetic data).
* Phase 1: `BOOT=12` iterations per frame, `IEKF_SN=64` (grouped-state size).
* Support: `KNN=12`, `SNCOUNT=192` (coherence graph nodes), `FOLDS=5`.
* Phase 2: `RIGID_ITERS=8` iterations per frame.
* Phase 3: `PHASE3_ITERS=5` iterations per frame, `PHASE3_HYP=prismatic,revolute`.
* `FORCE=1` reruns stages whose outputs already exist.

The GPU is an 8 GB card. Don't run two of these at once, and leave other GPU jobs off while static
training runs.

## 4. Stage by stage, one scene, forward and backward

The individual commands below bypass the all-scenes driver and let you set the **exact scene input and output directory**. They use the variables from Section 2.1. To switch to a different scene, recompute all three dependent paths:

```bash
S="fridge_10905"
SRC="$DATA/$S"              # exact input scene directory, used by --source
OUT="$RES/$S"               # exact output scene directory
BIDI="$OUT/bidirectional"  # exact dynamic-results directory

# Alternatively, for manual per-stage commands ONLY, SRC and OUT may be
# unrelated full paths, without following a shared parent/scene layout:
# SRC="/another/disk/input/my_scene"
# OUT="/another/disk/output/experiment_a"
# BIDI="$OUT/bidirectional"
```

The `--model` flag points to an **existing trained static 3DGS directory** under `OUT/static_bidi/`, whereas `--source` points to the **RGB-D scene directory** under `SRC`. `--out` points to the **exact desired directory** for that stage; the Python tools do not infer the output root from `--source`. The static training step writes the models with `train.py -s <input COLMAP directory> -m <output model directory>`.

The lines in Sections 4.1–4.4 use `$PY`; define it as shown in Section 2.1 before running the individual commands.

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

### 4.4 Phase 3 (per-frame joint displacement, joint fixed)

Phase 3 **does not change the joint**:

* The type and the geometry (prismatic direction, or revolute axis and axis point) are taken from
  4.3's `joint_reduction/` and held fixed. A run checks that they are bitwise unchanged before it
  writes anything.
* It re-estimates only the displacement q_k (and its rate) frame by frame on the RGB-D. It starts
  from joint identification's q track, moves the support Gaussians by the fixed joint, and keeps the
  static ones in the render.
* Choose the type with `--hypotheses revolute` or `--hypotheses prismatic`. The default runs both,
  each with its own fixed geometry.
* Backward is anchored at frame T with the final model.

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

* `summary.json`: per type, the fixed geometry (`geometry_unchanged_bitwise`), the q range against
  joint identification's, the largest change in q, and energies.
* `{prismatic,revolute}/geometry.json`: the same per type.
* `{prismatic,revolute}/trajectory.csv`: processing index, original frame, q, q̇, the Phase-2 q,
  posterior sd of q, and energy per frame.
* `evaluation_gt/phase3_gt_evaluation.json`: only with `--evaluate-gt`, written after inference.

About 1 min per type per direction at 8 cameras and 2 iterations/frame on fridge (25k Gaussians);
scale linearly with `--iters`. A run reading from the external SSD has occasionally stalled for tens
of minutes on I/O, so re-run it if that happens.

The window must be the full sequence, `--frames 60`. A shorter backward window would not be anchored
at frame T, and the runner refuses it.

## 5. Reading the results

* **Final joint:**
  * Type and geometry: `$BIDI/<dir>_rigid/joint_reduction/<type>/geometry.json`.
  * Displacement per frame: `$BIDI/<dir>_joint/<type>/trajectory.csv` (Phase 3), or
    `joint_reduction/<type>/trajectory.csv` if Phase 3 was not run.
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
# Uses the SAME input and output roots configured in Section 2.1.
# Optional endpoint correspondence diagnosis + post-run GT evaluation (all scenes):
bash scripts/run_fb_endpoint_correspondence_all.sh \
    --dataset-root "$DATA" --results-root "$RES" \
    --tag endpoint_eval_v4 --estimated-joints auto \
    --evaluate-gt --execute --gpu-memory-fraction 0.2
```

This auxiliary script uses `--dataset-root` and `--results-root`.

* `--dataset-root` may also be **one scene folder**, for example `--dataset-root "$DATA/fridge_10905"`;
  only that scene is diagnosed.
* Its outputs appear under `$RES/<scene>/endpoint_correspondence/<tag>/` and in the batch summary
  directory, not under `$BIDI`.

Phase 2 writes its own `joint_reduction/gt_evaluation.json` after fitting. The GT scores per run
(support precision and recall, axis and line error, coordinate and pose error) are in
`<scene>/endpoint_correspondence/<tag>/evaluation_gt/`.

## 7. Tests

```bash
$PY tests/test_rigid_se3_phase2.py              # Phase 2: Jacobians, adjoint, transport, CUDA renderer
$PY tests/test_iekf_process_covariance.py       # Phase 1: left-SO(3) process Jacobians, Q
$PY tests/test_motion_coherence_support.py      # support
$PY tests/test_phase2_parallel_pr.py            # P/R joint models (Phase 3 kinematics, FD)
$PY tests/test_joint_phase3_runner.py           # Phase 3: init, refusals, fixed joint (H = q-column, geometry bitwise)
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
  * Two update defects in `estimator/joint_iekf.py` are latent: acceptance is likelihood-only, and
    the information transport has its sides swapped.
* **No type decision, no "unresolved" output** on the standard path.
