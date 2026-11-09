#!/usr/bin/env bash
# STANDARD synthetic pipeline (status 2026-11-09, STATUS.md).
#
# The recipe that produced the v5 synthetic results (results_11_10_26/ours_v5_canonical_synthetic + washer
# in ours_v5_phase1_gates/washer_103776_canonical; commit 8273cd0, clean tree, identical config on all 10):
#   calibrated-bootstrap orchestrator, ideal sensor, covariance-surface endpoint bootstrap (30000
#   Gaussians, 1000 refinement steps), one direction per scene from metadata/direction/summary.json,
#   masked-RGB expected-quad observation, canonical Phase-1 partition, Phases 1-3 on every camera.
# The v6 Phase 1-3 options (--velocity-tau, --phase2-support-refine) are OFF here, as in v5.
# GT is read only by the evaluation after Phase 3.
#
#   bash run_standard_synthetic.sh 2>&1 | tee ~/standard_synthetic.log
#   SCENES="USB_100109" bash run_standard_synthetic.sh        one scene
#   DRY=1 bash run_standard_synthetic.sh                      validate inputs and print the plan, no GPU
#
# Env overrides: SYN (dataset root), OUT, SCENES, PY, DRY
set -u
cd "$(dirname "$(readlink -f "$0")")" || exit 2
PY=${PY:-~/miniconda3/envs/rigeuler3dgs/bin/python}
SYN=${SYN:-"/media/samuelemara/Extreme SSD/Samuele/Backup_09_10_26/Sapien/output/preparation_8cams/partnet_subset_fixed_render_linear"}
OUT=${OUT:-"/media/samuelemara/Extreme SSD/Samuele/results_11_10_26/ours_standard_synthetic"}
SCENES=${SCENES:-"foldchair_102255 laptop_10211 storage_45135 oven_101917 scissor_11100 washer_103776 blade_103706 fridge_10905 stapler_103111 USB_100109"}
DRYARG=(); [ "${DRY:-0}" = 1 ] && DRYARG=(--dry-run)

echo "standard synthetic: $(pwd) ($(git rev-parse --short HEAD)$(git diff --quiet HEAD -- . ':!submodules' || echo ' + uncommitted')) -> $OUT"
STATUS=()
for s in $SCENES; do
  echo "=== $s  $(date +%T)"
  "$PY" tools/run_noise_aware_rigeuler.py --source "$SYN/$s" --dataset-kind synthetic \
    --motion-start-frame 0 --motion-end-frame 59 --bootstrap-mode covariance_surface --sensor-noise-model ideal \
    --stages "bootstrap phase1 phase2 phase3" --phase1-partition canonical \
    --out "$OUT/$s" "${DRYARG[@]}" 2>&1 | tail -4
  STATUS+=("$s rc=${PIPESTATUS[0]}")
done
echo; echo "=== summary  $(date)"; printf '  %s\n' "${STATUS[@]}"
