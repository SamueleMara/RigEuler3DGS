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
  mask/cam_XXX/...                      object silhouettes (static training; motion stages with --object-mask-constraint)
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
| **Fair A/B** `scripts/run_object_mask_ab.sh` | `--dataset-root "$DATA"`, `--static-root "$RES/ours"` (static models, read only) | `--original-root`, `--masked-root` (default `$RES/ours_fairab_original`, `$RES/ours_fairab_masked`) | `/<scene>` per scene; table in `<masked-root>_vs_original/` |

The driver accepts `--input`/`--output` (or `-i`/`-o`), plus `--scenes`, `--stages`, `--frames`,
`--cameras`, `--dry-run` and `--help`. The environment variables `SRC_ROOT`, `OUT_ROOT`, `SCENES`,
`STAGES`, `FRAMES`, `CAMS` and `DRY` do the same; the command-line options win. The individual Python
tools use explicit `--source` (one scene folder), `--model` and `--out` flags.

A `--dry-run` never writes or deletes anything. It only prints the resolved paths, the input check
and the commands.

### 2.3 Input check (automatic, before any GPU work)

For every scene, the driver checks:

* `colmap/sparse/0` and `colmap_final/sparse/0` exist, and **each holds exactly one image per
  camera**. An endpoint model is one instant seen by every camera.
* `rgb/cam_*` and `depth_npy/cam_*` exist, with the same number of frames per camera.
* The frame files are numbered `frame_000000, frame_000001, …` without gaps. The RGB-D loader reads
  them consecutively.

A scene that fails is skipped with the reason printed, and the rest of the batch runs.

* `--frames auto` (the default) uses every frame of the scene. The backward run is anchored at the
  last frame, so it needs the full sequence.
* `--cameras` sets how many cameras Phase 1/2/3 use. The default is 8, as in the stable results;
  `--cameras auto` uses all of them.
* Static training always uses every camera in the COLMAP model.

**Example of a failing scene.** `real/fridge_plastic_20261016_163314` has an initial model
(`colmap/sparse/0`) holding **984 images**, every frame of its 3 cameras, while `colmap_final` holds
3. The static "initial" model would then be trained on a moving object, and loading all 984 images is
what ran the GPU out of memory. It is refused with:

```text
!! input check failed -- skipping: colmap/sparse/0 holds 984 images for 3 cameras; an endpoint model
   must hold exactly one image per camera (its endpoint frame)
```

Fix it by keeping only the frame-0 image of each camera, with the same cameras and the same 3D
points:

```bash
# writes <scene>/colmap_frame000000/ and checks it (3 images, poses identical to the source)
$PY tools/make_endpoint_colmap.py --scene "$SCENE_DIR" --frame 0
# or put it in place directly: colmap/ -> colmap_allframes/ (kept), new model -> colmap/
$PY tools/make_endpoint_colmap.py --scene "$SCENE_DIR" --frame 0 --replace
```

Then run the driver as usual. The longer-term fix is for the converter to export `colmap/` this way,
as it already does for `colmap_final/`.

### 2.4 Scenes without ground truth

The pipeline needs no GT. If `metadata/joint_info.json`, `metadata/frame_values.csv`,
`pointcloud/joint_metadata.json` or `pointcloud/points3D_parts*.ply` are missing (as in the real
captures), every stage still runs. Only the GT metrics are skipped:

| Stage | Without GT |
|---|---|
| Phase 1 | `evaluation/optimized_3dgs_gt.json` says `OPTIMIZED_LABELED_3DGS_GT_NOT_FOUND`, and support metrics are not computed |
| Phase 2 | `joint_reduction/gt_evaluation.json` says `GT_NOT_AVAILABLE`; the debug page shows "no GT available" |
| Phase 3 | `evaluation_gt/phase3_gt_evaluation.json` has `gt_status` `GT_UNAVAILABLE`, or `GT_INCOMPLETE` when only `metadata/cameras.json` exists, and no fit is scored |

### 2.5 Object masks in the motion stages (`--object-mask-constraint`, opt-in)

Code audit and proofs: [docs/BACKGROUND_MASK_IEKF_AUDIT_2026_10_19.md](docs/BACKGROUND_MASK_IEKF_AUDIT_2026_10_19.md).

**The problem.** Phase 1/2/3 fit full-frame RGB and the valid depth over a constant render background
(black). The renderer's background derivative is correct, and was checked numerically on four
backgrounds. Even so, a Gaussian outside the object can sit where nothing penalises it:

* on the synthetic scenes the observed background is black and has no depth, so a dark Gaussian
  outside the mask leaves no residual;
* on real captures the background is textured and has depth, so object Gaussians can fit it.

**What the constraint does.** The per-frame object mask is a measurement of where the object is
**not**. For each such ray, the object's visible opacity must stay at most ε:

    A_vis = 1 − Π_i (1 − g_i α_i) ≤ ε

* The contributors are the renderer's own, with the same cut-off and clamp.
* For a ray with no surface in front of the object (`trusted-full-ray`), g_i = 1.
* For a ray in front of a measured non-object surface (`known-nonobject-depth`),
  g_i = 1[splat depth < z_obs − δz]. Only splats in front of that surface count, and each is judged
  by its own depth, never by the composited depth.

Every IEKF iteration then works as follows:

1. **Solve.** The step solves the constrained MAP problem: the trust-region step is projected onto
   `A_vis + C_s δ ≤ ε` (a dual active-set solver, `solve_qp_projection` in `estimator/constrained_qp.py`). C_s is the visible-alpha
   Jacobian (native CUDA JVP/VJP, `_C_vis`) lifted through the stage's own state, so the same code
   serves Phase 1 (free Gaussians), Phase 2 (rigid SE(3)) and Phase 3 (per-frame joint
   displacement, with the joint geometry and type fixed).
2. **Re-render and audit.** The trial state is re-rendered and audited on every negative ray:
   * no new violation;
   * the total of the violations already present at the frame start does not grow.
   A failing step is rejected.
3. **Likelihood.** The RGB-D likelihood uses object pixels only (mask = 1).

**Where it stops.** The constraint never invents a direction. A Gaussian that is stranded far outside
the mask, offscreen, or behind measured depth has no information pulling it back, so it stays on its
anchored trajectory. The frame is then reported as not `FEASIBLE`. If keeping the rays dark would
contradict the RGB-D, for example a whole rigid part held back to hide one misowned Gaussian, those
rays are released and the frame is reported as `MODEL_OWNERSHIP_INFEASIBLE`.

**Output.** Each stage writes `visibility_constraint.json`:

* contents: the resolved policy, the per-frame status (`FEASIBLE`, `MODEL_OWNERSHIP_INFEASIBLE`,
  `UNRESOLVED_INFEASIBLE`, `INDISTINGUISHABLE`), ray counts, the audits, and the accepted and
  rejected steps;
* location: Phase 1 in `<dir>/`, Phase 2 in `<dir>_rigid/phase2_rigid/`, Phase 3 in
  `<dir>_joint/<type>/`.

**Options:**

| option | default | meaning |
|---|---|---|
| `--object-mask-constraint {off,visible-negative-hard}` | `off` | turn the constraint on |
| `--mask-ray-semantics {trusted-full-ray,known-nonobject-depth,conservative}` | `trusted-full-ray` | how a negative ray is read: `trusted-full-ray` for the synthetic SAPIEN scenes (no occluders); `known-nonobject-depth` for real captures with measured surfaces; `conservative` counts the rays but constrains none |
| `--mask-source DIR` | `<source>/mask` | the mask directory itself (`<cam>/frame_NNNNNN.png`); in `run_bidirectional_all.sh` use `--mask-subdir NAME` for `<scene>/NAME` |
| `--mask-depth-noise-m S` | none | required by `known-nonobject-depth`, which is refused without it; δz = 3S |
| `--mask-boundary-px N` | 2 | rays within N px of the mask edge are left free (antialiasing, Gaussian tails) |
| `--mask-alpha-epsilon E` | 0.05 | ε |

The driver forwards these flags to Phase 1, 2 and 3, and only when the mode is not `off`:

```bash
bash run_bidirectional_all.sh -i "$SCENE_DIR" -o "$RES" --stages "phase1 phase2 phase3" \
  --object-mask-constraint visible-negative-hard --mask-ray-semantics trusted-full-ray
```

The hard constraint is a measurement model of the **original** observation: since 2026-10-21 it
needs `--observation-mode original` explicitly (the default masked mode refuses it). The A/B script
now compares original vs masked observation (section 2.6); a hard-constraint A/B is run by hand with
two driver calls into two new roots.

**Tests** (real CUDA renderer):

| test file | checks |
|---|---|
| `tests/test_background_compositing_audit.py` | 11 |
| `tests/test_visible_alpha_operators.py` | 12 |
| `tests/test_constrained_qp.py` | 21 |
| `tests/test_visibility_constrained_iekf_phase1.py` | 12 |
| `tests/test_visibility_constrained_iekf_phase23.py` | 8 |

**Building `_C_vis`.** This is a separate extension; `_C` is unchanged:

```bash
cd submodules/diff-gaussian-rasterization && python setup_vis.py build_ext --inplace --build-temp build_vis
```


### 2.6 Masked-RGB observation (`--observation-mode`, the DEFAULT since 2026-10-21)

Implementation and evidence:

- [docs/MASKED_RGB_FOREGROUND_DEPTH_AUDIT_2026_10_20.md](docs/MASKED_RGB_FOREGROUND_DEPTH_AUDIT_2026_10_20.md)
- [docs/MASKED_RGB_FOREGROUND_DEPTH_IMPLEMENTATION_2026_10_20.md](docs/MASKED_RGB_FOREGROUND_DEPTH_IMPLEMENTATION_2026_10_20.md)

**Default.** Every motion entry point (driver, Phase 1/2/3 runners) uses `masked-rgb-expected-quad`
unless `--observation-mode original` is passed explicitly; the masks are checked BEFORE any model
load, training or output deletion and a scene without complete binary masks is refused (driver:
`SKIPPED(masks not ready)`), never silently run in the original mode. `masked-rgb-constant` is an
ablation. Static training is unaffected. Repair report:
[docs/MASKED_IEKF_DEFAULT_REPAIR_REPORT_2026_10_21.md](docs/MASKED_IEKF_DEFAULT_REPAIR_REPORT_2026_10_21.md).

**Why.** The original mode compares the whole image with the object rendered over black.
On a real capture the grey, textured background is about 98% of that cost and pulls object Gaussians
outward.

**What the masked modes compute.** For every frame and camera:

- **RGB:** the recorded object, premultiplied by the whole-object mask and composited on a known
  artificial background b: `Y = M·I + (1 − M)·b`. The scene is rendered on the same b, and the error
  is taken over the whole image.
- **Depth:** the recorded depth only where it is valid and inside the mask. Every other pixel has
  exactly zero precision, and its stored value is never read.
- **Expected-quad only:** b is uniform in [0,1]³. The expected quadratic RGB error is evaluated
  exactly, as one render at b̄ = (½, ½, ½) plus an analytic accumulated-alpha term ½κ(A − M)², with
  κ = trace(W)/12 (= 1/(4σ²) in Phase 1). No random backgrounds, no extra renders, no QP.
- **Precision:** the RGB precision W is fixed:
  - Phase 1: 1/σ_rgb²;
  - Phase 2/3: 1/(σ² + model-error variance).

  It is not reweighted with Student-t, which the exact expectation needs. Depth keeps its robust
  precision.
- **Where the term enters:** the alpha term is in the energy, the right-hand side, the normal
  operator, the Fisher preconditioner and the posterior of Phase 1, 2 and 3. Each stage uses its own
  lift: supernodes, rigid SE(3), or the fixed-joint displacement.

| option | default | meaning |
|---|---|---|
| `--observation-mode original\|masked-rgb-constant\|masked-rgb-expected-quad` | `masked-rgb-expected-quad` | `original` is the full-frame observation, an explicit control |
| `--rgb-background-distribution uniform01` | `uniform01` | expected-quad only |
| `--rgb-background-constant R G B` | none | required by `masked-rgb-constant`, refused otherwise |
| `--mask-source DIR` / driver `--mask-subdir NAME` | `<scene>/mask` | binary masks at native resolution; non-binary, wrong-size, missing or ambiguous files are refused |

**Rules:**

- The masked modes refuse to run together with `--object-mask-constraint` (hard QP: explicit
  `--observation-mode original` only).
- Depth is a measurement only where it is valid, finite and > 0 (positive camera Z); masks remove
  depth, they never overwrite its value.
- Every stage writes `stage_manifest.json` (`tools/stage_signature.py`): the driver reuses a stage
  only if it is complete with an identical signature (code, scene, direction, static PLY / cfg_args
  hashes, observation mode, mask content hash, source listing, config, upstream stage) and REFUSES a
  legacy or differently configured output (`FORCE=1` overwrites). Use new output roots after
  changing anything.
- Each stage also writes `mask_diagnostics.json` (GT-free background leakage of the tracked centres
  AND orientations, same definition in every mode), `stage_resources.json` (wall time, peak GPU),
  Phase 1 `phase1_frames.json` (covariance, stop reasons) and Phase 2/3 `model_error_audit.json`.
- Static training never receives these options.
- Each stage writes `masked_observation.json`: the mode, the background, the precision, coverage, and
  the per-frame energy terms.

**Tests** (real CUDA): `tests/test_masked_rgb_observation.py`, 46 checks. They cover:

- the composite;
- coverage-correct resizing;
- invariance to the depth values outside the mask;
- the exact expectation against an 8-point cubature: energy, gradient and normal;
- the alpha Jacobian and the new alpha Fisher kernel;
- all three stage lifts;
- the dense normal matrix;
- acceptance on the exact objective;
- no QP and one render per camera per evaluation;
- conflicting flags;
- identifiability;
- end-to-end Phase 1/2/3 windows.

**Tests:** `tests/test_masked_default_repair.py` (73 checks) covers the 2026-10-21 repairs.

**Commands:**

```bash
# FAIR A/B: the same static models copied (verified sha256) into two NEW roots, then FRESH
# Phase 1/2/3 in BOTH arms (explicit --observation-mode original vs masked), then the table with a
# manifest diff (only the observation block may differ) and GT-free leakage diagnostics
bash scripts/run_object_mask_ab.sh --dry-run          # preview
bash scripts/run_object_mask_ab.sh --scenes "fridge_10905"   # one object first
bash scripts/run_object_mask_ab.sh                    # all ten synthetic scenes

# one scene or root, directly (masked-rgb-expected-quad is the default)
bash run_bidirectional_all.sh -i "$SCENE_DIR" -o "$RES_NEW" --stages "phase1 phase2 phase3"

# benchmark: one tracked frame, original vs masked, timing and leakage (writes only into a NEW --out)
python tools/benchmark_masked_rgb_observation.py --model "$RES/fridge_10905/static_bidi/initial" \
  --source "$DATA/fridge_10905" --out "$RES/../benchmark_masked_rgb_$(date +%Y%m%d_%H%M)"
```

**Not addressed by this mode:**

- Gaussians that no camera constrains, including ones hidden behind the visible surface, still
  drift with the constant-velocity motion model.
- A Gaussian that is fully inside a negative region gets a cost but no in-image restoring direction.

### 2.7 GPU memory (8 GB card)

* **Static training** keeps every camera image, depth map and mask on the CPU (`--data_device
  cpu`) and moves only the view being trained.
* **Phase 1/2/3** keep the RGB-D sequence on the CPU (`--frames-on-cpu`), as well as the endpoint
  camera images (unused there).
* The Phase-1 covariance repair falls back to the CPU if the GPU cannot allocate its workspace.
* The driver sets `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`.
* What stays on the GPU is the render linearization of every camera used for the motion stages, and
  it grows with the camera count. 8 cameras fit; 64 cameras (`--cameras auto` on
  `preparation_64cams`) ran out of memory on some scenes even with the above.

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
* Window: `--frames auto` (every frame of the scene, 60 on the synthetic data); `--cameras 8` for
  Phase 1/2/3 (`--cameras auto` = all).
* Phase 1: `BOOT=12` iterations per frame, `IEKF_SN=64` (grouped-state size).
* Support: `KNN=12`, `SNCOUNT=192` (coherence graph nodes), `FOLDS=5`.
* Phase 2: `RIGID_ITERS=8` iterations per frame.
* Phase 3: `PHASE3_ITERS=5` iterations per frame, `PHASE3_HYP=prismatic,revolute`.

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
    --test_iterations 999999 --disable_viewer --no_wandb --data_device cpu
$PY train.py -s "$SRC/colmap_final" -m "$OUT/static_bidi/final"   --images ../rgb --masks "$SRC/mask" -r 1 \
    --depths depth_npy --sapien_depth_is_metric --iterations 30000 --static_train_iters 30000 \
    --test_iterations 999999 --disable_viewer --no_wandb --data_device cpu
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

Every stage writes its prediction to disk first and only then scores it against GT. GT comes
from the dataset's `pointcloud/` and `metadata/` files and the `gt_part`/`gt_dynamic` labels that
static training attaches to every Gaussian. Without GT the stages still run and only the scores
are missing (section 2.4).

| Stage | Prediction (GT-free) | GT score |
|---|---|---|
| Phase 1 + support | `<dir>/predicted_support.json` | `<dir>/evaluation/confusion_matrix.json`, `gt_coherence.json`, `optimized_3dgs_gt.json` (`estimator/motion_coherence_eval.py`, `estimator/optimized_gt.py`) |
| Phase 2 + joint | `<dir>_rigid/phase2_rigid/`, `<dir>_rigid/joint_reduction/` | `<dir>_rigid/joint_reduction/gt_evaluation.json` |
| Phase 3 | `<dir>_joint/<type>/trajectory.csv`, `summary.json` | `<dir>_joint/evaluation_gt/phase3_gt_evaluation.json` (`--evaluate-gt`) |

Fair A/B of two results roots (original vs masked observation; both must be fresh runs with complete
stage manifests), GT used for scoring only; and the machine-readable stage report:

```bash
$PY tools/compare_object_masks.py --original "$RES_A" --masked "$RES_B" --out "$RES_B"_vs_original
$PY tools/repair_report.py --root "$RES_A" --root "$RES_B" --out "$RES_B"_vs_original/stage_report.json
```

## 7. Tests

The complete stage-by-stage code map with file:line references is in
[docs/PIPELINE_CODE_MAP.md](docs/PIPELINE_CODE_MAP.md).

All on the real CUDA renderer where relevant; each prints `N checks passed`.

```bash
# static training
$PY tests/test_carving_depth_object_only.py tests/test_object_mask_transparency_carving.py
# renderer operators (RGB-D JVP/VJP/Fisher, background term, visible / accumulated alpha)
$PY tests/test_unified_cuda_renderer_jvp_vjp.py; $PY tests/test_iekf_cuda_operators.py
$PY tests/test_background_compositing_audit.py; $PY tests/test_visible_alpha_operators.py
# Phase 1 IEKF (state, process, PCG, trust region, grouped solve, end to end)
$PY tests/test_iekf_manifold_state.py; $PY tests/test_iekf_process_covariance.py
$PY tests/test_iekf_pcg_measurement.py; $PY tests/test_v2_trust_region_and_lines.py
$PY tests/test_v2_grouped_solve.py; $PY tests/test_iekf_end_to_end.py
# masked observation / hard constraint
$PY tests/test_masked_rgb_observation.py; $PY tests/test_constrained_qp.py
$PY tests/test_visibility_constrained_iekf_phase1.py; $PY tests/test_visibility_constrained_iekf_phase23.py
# support (coherence) and frame order
$PY tests/test_motion_coherence_support.py; $PY tests/test_bidirectional_sequence_direction.py
# Phase 2 (rigid SE(3), joint) and Phase 3 (fixed joint, displacement)
$PY tests/test_rigid_se3_phase2.py; $PY tests/test_phase2_parallel_pr.py; $PY tests/test_joint_phase3_runner.py
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
