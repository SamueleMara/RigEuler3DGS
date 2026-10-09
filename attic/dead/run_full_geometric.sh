#!/usr/bin/env bash
# Full geometric rigid-motion joint discovery, all iterations, complete debug.
#
#   CODEX_IMPLEMENT_GEOMETRIC_RIGID_MOTION_JOINT_DISCOVERY.md  §43, §51
#   CODEX_ADD_VISUAL_AND_NUMERICAL_DEBUG_GEOMETRIC_PIPELINE.md §51
#
# Both datasets run the IDENTICAL algorithm and the identical flags. Nothing
# below is dataset-specific except the model and the output directory.
set -euo pipefail

# MEASURED on this machine (RTX 4070 Laptop, 8 GiB), with these exact settings:
#
#   USB      11,320 Gaussians,  60 frames x 8 cameras  ->  32.1 min, 1.86 GiB
#   storage  40,521 Gaussians,  60 frames x 8 cameras  ->  not yet measured;
#            3.6x the Gaussians, so budget roughly 1.5-3x the USB time and
#            expect a higher peak (it should still fit in 8 GiB -- the window
#            itself is staged on the host, --frames-on-cpu being the default).
#
# Phase 2 dominates and scales as discovery_outer x frames x rigid_refine_iters.
# If a run is too slow, halve --discovery-outer and --rigid-refine-iters first;
# they are the two that multiply. On USB, --discovery-outer 8 had margin: the
# section 28 criteria were all met at outer 5.
#
# The iteration counts are BUDGETS -- every stage stops on its own convergence
# criterion (section 28) and reports which one fired.

export PYTHONPATH=.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_HOME=/usr/local/cuda-12.8
PY=/home/samuelemara/miniconda3/envs/rigeuler3dgs/bin/python

COMMON=(
  --observation-source dataset_rgbd
  --sequence-start 0 --sequence-length 60     # all 60 dataset time steps
  --all-cameras                               # all 8 cameras, asserted
  --full-gaussians                            # every Gaussian in the model
  --supernodes                                # grouped state (needed at scale)
  --window-discovery --num-moving-parts 1     # K is the only structural prior

  # ---- iteration budgets, outermost first -------------------------------
  --discovery-outer      8    # §10  z <-> {T_k,t} outer alternation
  --rigid-refine-iters  25    # §9   Levenberg iters per frame for T_{k,t}
  --reopen-support-rounds 4   # §26  z <-> T <-> joint geometry, after the fit
  --bootstrap-iters     12    # §25.1 IEKF iters per frame, motion evidence
  --iters                8    # IEKF Gauss-Newton iters per frame step
  --pcg-iters           80    # PCG iters inside each IEKF step

  --depth-sigma constant:0.05:0.0
  --no-visualizations

  # ---- complete debug package (§50) -------------------------------------
  --debug-all
  --debug-frame-stride 5          # HTML/PLY/panels every 5th frame;
                                  # every NUMERICAL artifact still covers all 60
  --debug-gaussian-subsample 6000 # Gaussians drawn in the HTML
)

run () {   # run <model> <out>
  echo "=============================================================="
  echo "  $1  ->  $2"
  echo "=============================================================="
  "$PY" tools/run_iekf_scene.py --model "$1" "${COMMON[@]}" --out "$2"
}

run output/merged_USB_100109    output/geometric_usb_full
run output/merged_storage_45135 output/geometric_storage_full

echo
echo "ALL DONE.  Start with:"
for d in output/geometric_usb_full output/geometric_storage_full; do
  echo "  $d/debug_numeric/failure_summary.md"
  echo "  $d/geometric_motion_debug.html"
done
