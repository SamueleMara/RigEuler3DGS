#!/usr/bin/env bash
# Per scene: Phase 4 (4A on ALL Gaussians x ALL views; 4B / 4C on every RECOVERY_STRIDE-th frame), then -- only if
# 4C completed -- Phase 5 on ALL views as a separate invocation. One scene finishes completely before the next.
#
#   RUN_ID=full_s4 RECOVERY_STRIDE=4 bash run_phase45_then_5_synthetic.sh
# Env: SCENES (default: all 10 synthetic scenes), RUN_ID (required), RECOVERY_STRIDE (default 4), MODE, P4_ARGS, P5_ARGS
set -u
cd "$(dirname "$(readlink -f "$0")")" || exit 2
[ -n "${RUN_ID:-}" ] || { echo "RUN_ID is required (the same id names the Phase-4 and Phase-5 runs)"; exit 2; }
SCENES=${SCENES:-"foldchair_102255 laptop_10211 storage_45135 oven_101917 scissor_11100 washer_103776 blade_103706 fridge_10905 stapler_103111 USB_100109"}
RECOVERY_STRIDE=${RECOVERY_STRIDE:-4}
for s in $SCENES; do
  SCENES="$s" STAGE=all RUN_ID="$RUN_ID" RECOVERY_STRIDE="$RECOVERY_STRIDE" EXTRA_ARGS="${P4_ARGS:-}" \
      bash run_phase45_synthetic.sh
  SCENES="$s" RUN_ID="$RUN_ID" MODE="${MODE:-gaussian_rgbd}" EXTRA_ARGS="${P5_ARGS:-}" bash run_phase5_synthetic.sh
done
