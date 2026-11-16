#!/usr/bin/env bash
# Phase 5 (unified spec PART II) on completed Phase-4 runs of the synthetic scenes. A SEPARATE program: it never runs
# Phase 4, only consumes <OUT>/<scene>/phase4/<RUN_ID> (4A + 4C complete.json, hash-verified) and writes
# <OUT>/<scene>/phase5/<RUN_ID>. Trains on ALL views; joint frozen; RGB = 3DGS l1_loss + fused_ssim (lambda_dssim 0.2).
#
#   SCENES="fridge_10905" RUN_ID=full_s4 bash run_phase5_synthetic.sh
#   MODE=sh_only ...        colour only (geometry bitwise frozen)
#   EXTRA_ARGS="--loss-use-depth false --loss-use-mask false"    RGB + SSIM only
# Env: RES, OUT, SCENES, RUN_ID (required), MODE, PY, DRY, EXTRA_ARGS, COMPARISON (default 1)
set -u
cd "$(dirname "$(readlink -f "$0")")" || exit 2
PY=${PY:-~/miniconda3/envs/rigeuler3dgs/bin/python}
RES=${RES:-"/media/samuelemara/Extreme SSD/Samuele/results_11_10_26"}
OUT=${OUT:-"$RES/ours_phase4_corrected_synthetic"}
SCENES=${SCENES:-"USB_100109"}
MODE=${MODE:-gaussian_rgbd}
[ -n "${RUN_ID:-}" ] || { echo "RUN_ID of a completed Phase-4 run is required"; exit 2; }
DRYARG=(); [ "${DRY:-0}" = 1 ] && DRYARG=(--dry-run)
read -r -a XARGS <<< "${EXTRA_ARGS:-}"
echo "phase5 synthetic: $(pwd) ($(git rev-parse --short HEAD)$(git diff --quiet HEAD -- . ':!submodules/sam3' || echo ' + uncommitted')) mode $MODE run $RUN_ID"
STATUS=()
for s in $SCENES; do
  P4="$OUT/$s/phase4/$RUN_ID"; P5="$OUT/$s/phase5/${RUN_ID}_$MODE"
  if [ ! -f "$P4/4c_accepted_moving/complete.json" ]; then STATUS+=("$s: SKIPPED, no complete 4C in $P4"); continue; fi
  echo "=== $s  $(date +%T)"
  [ "${DRY:-0}" = 1 ] || mkdir -p "$P5"
  "$PY" -u tools/train_phase5_rgbd_gaussians.py --phase4-dir "$P4" --output "$P5" --train-mode "$MODE" \
      "${DRYARG[@]}" "${XARGS[@]}" 2>&1 | grep --line-buffered -v Warning \
      | { if [ "${DRY:-0}" = 1 ]; then cat; else tee -a "$P5/console.log"; fi; }
  STATUS+=("$s phase5 $MODE rc=${PIPESTATUS[0]} -> $P5")
  if [ "${DRY:-0}" != 1 ] && [ -f "$P5/complete.json" ] && [ "${COMPARISON:-1}" = 1 ]; then
    # FreeArtGS-style comparison (all frames x all cameras, joint axes, GT if the dataset has it) -> $P5/comparison
    "$PY" -u tools/phase5_comparison.py --phase5-dir "$P5" 2>&1 | grep --line-buffered -v "Warning\|^Reading camera" \
        | tee -a "$P5/console.log"
    STATUS+=("$s comparison rc=${PIPESTATUS[0]} -> $P5/comparison")
  fi
done
echo; echo "=== summary  $(date)"; printf '  %s\n' "${STATUS[@]}"
