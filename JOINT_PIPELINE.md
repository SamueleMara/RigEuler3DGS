# Joint discovery: the default pipeline

```
Phase 1 free motion
  -> coherence MOVING_RIGID support
  -> rigid SE(3) RGB-D Phase 2          six pose numbers, no joint geometry
  -> joint reduction P + R              no renderer
  -> pose-space model comparison
```

The renderer is asked one question: **which rigid pose `T_t` best explains the
moving part at this frame?** The joint is inferred afterwards, from the
recovered finite rotations. No axis direction, axis point or joint type is
ever a parameter of the image optimiser.

## Run it

From a raw input scene, all three stages, one command:

```bash
SCENES="scissor_11100 laptop_10211 foldchair_102255 oven_101917 washer_103776" \
  ./run_full_pipeline_all.sh
```

It trains the static 3DGS endpoints, runs the coherence selector, then rigid
SE(3) Phase 2 and the reduction. Phase 1 is solved once, in stage 2, and
stage 3 reuses its centers. A scene whose static model already exists skips
stage 1, so re-running after a failure resumes rather than retrains.

```bash
STAGES="2 3" ./run_full_pipeline_all.sh      # scenes already trained
SCENES="laptop_10211" ./run_full_pipeline_all.sh
ITERS=1500 FRAMES=16 CAMS=3 OUTROOT=/tmp/smoke ./run_full_pipeline_all.sh
```

`OUTROOT` is honoured by every stage, so a reduced-budget run never touches
the production tree.

When the static models and the supports already exist, the last two stages
have their own runners:

```bash
./run_support_labelling_all.sh     # stage 2 only, plus the Otsu comparison
./run_rigid_se3_all.sh             # stage 3 only
python tools/run_rigid_se3_pipeline.py --scene stapler_103111 \
    --frames 60 --cameras 8 --max-gaussians 0 --frames-on-cpu
```

Any set of finished runs tabulates with:

```bash
python tools/summarize_pipeline.py --all
```

The reduction is renderer-free, so it can be re-derived from saved poses in
seconds without repeating Phase 2:

```bash
python tools/rereduce_rigid_trajectory.py output/rigid_se3_*
```

## Why this is the default

Measured on all five datasets, 60 frames, 8 cameras, every Gaussian. GT axis
angle error in degrees, against `pointcloud/joint_metadata.json`:

| scene | GT type | legacy direct-axis | **this pipeline** |
|---|---|---|---|
| stapler_103111 | revolute | 89.22 | **0.23** |
| USB_100109 | revolute | 1.48 | **0.37** |
| fridge_10905 | revolute | 6.84 | **0.07** |
| storage_45135 | prismatic | 72.91 | **0.23** |
| blade_103706 | prismatic | 0.14 | **0.08** |

The old parameterisation let the optimiser trade a distant axis against a
small angle, because over a short sweep those produce nearly the same pixels.
The rigid pose has no such valley; the axis is then recovered from the finite
rotations by two linear-algebra problems that stay conditioned at `cond = 1.0`
even when the axis is 100 part-radii away.

## Output

```
output/rigid_se3_<scene>/
    support_coherence/support.json
    phase1/centers/
    phase2_rigid/     poses.npy  velocities.npy  pose_covariances.npy
                      state_trace.csv  phase1_kabsch_vs_rgbd.csv
                      exact_rigidity_gate.json  renders/
    joint_reduction/  one_dof_diagnostics.json  gt_evaluation.json
                      prismatic/  revolute/
    html/             rigid_phase2_debug.html      rigid reconstruction alone
                      rigid_to_joint_debug.html    + the axes, as overlays
                      rigid_timelines.html         section 41
                      rigid_renders.html           GT/P1/Kabsch/P2 per frame
    summary.json
```

## Reading the result

Three numbers, in this order.

1. **`exact_rigidity_gate.json`** — `max_pairwise_distance_drift` must be at
   machine precision. It is a property of the shared SE(3) parameterisation,
   so anything else means something is broken, not noisy.
2. **`phase1_kabsch_vs_rgbd.csv`** — how far the RGB-D stage moved from the
   Phase-1 rigid projection. A large divergence is a reason to inspect the
   RGB-D objective *before* trusting any axis.
3. **`joint_reduction/revolute/geometry.json`** →
   `axis_radius_over_part_scale`. Order 1 means the axis lies on the part.
   A large value means the revolute fit is approximating a translation with a
   distant axis.

That third number matters because the pose energy alone is **not** a
classifier. On storage (GT prismatic) `E_R/E_P = 0.86`, so R fits marginally
better — while its axis sits 62 part-radii away. The energies are evidence;
section 35 asks for them to be reported, not resolved.

## Dataset prerequisites

Each `input/<scene>/` needs `colmap/`, `rgb/<cam>/frame_*.png`, `depth_npy/`,
`pointcloud/points3D_parts.ply` (GT part labels, for the static trainer),
`pointcloud/joint_metadata.json` (GT axis, for scoring only) and
`metadata/joint_info.json`.

One easy-to-miss item: `colmap/images/<cam>/frame_000000.png` must exist. The
working scenes hold a relative symlink there:

```bash
ln -s ../../../rgb/cam_000/frame_000000.png \
      input/<scene>/colmap/images/cam_000/frame_000000.png
```

Without it the COLMAP reader raises `FileNotFoundError` before training
starts.

## Legacy

`./run_complete_joint_debug_all.sh` and `tools/run_complete_joint_debug.py`
run the superseded generic-screw, direct-P and direct-R RGB-D estimators.
They are kept for regression and paper ablations and no longer define
production joint geometry. Both announce this on every run.

Full write-up: `RIGID_SE3_PHASE2_AND_JOINT_REDUCTION_REPORT.md`.
