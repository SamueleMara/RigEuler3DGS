#!/usr/bin/env bash
# Multi-scene batch of the corrected Phase 4 (real, then synthetic). The 2026-11-11 directive forbids a multi-scene
# batch before ONE real scene's complete Phase 4 has been reviewed, so this refuses unless CONFIRM_BATCH=1.
# Phase 5 is not run. Logs: ~/phase4_real.log, ~/phase4_synthetic.log (tee, line-buffered; no `tail`).
set -u
cd "$(dirname "$(readlink -f "$0")")" || exit 2
if [ "${CONFIRM_BATCH:-0}" != 1 ]; then
  echo "refusing: review one real scene first (SCENES=red_fridge bash run_phase45_real.sh); then CONFIRM_BATCH=1"
  exit 2
fi
echo "batch start $(date)"
SCENES="${REAL_SCENES:-red_fridge wooden_box light_blue_box}" bash run_phase45_real.sh 2>&1 | tee ~/phase4_real.log
SCENES="${SYN_SCENES:-foldchair_102255 laptop_10211 storage_45135 oven_101917 scissor_11100 washer_103776 blade_103706 fridge_10905 stapler_103111 USB_100109}" \
  bash run_phase45_synthetic.sh 2>&1 | tee ~/phase4_synthetic.log
echo "batch end $(date)"
