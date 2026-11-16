#!/usr/bin/env bash
# Unified Phase 4 (spec 2026-11-11: 4A -> 4B -> 4C as separate stages) on the v5 standard synthetic runs. Reads the Phase 0-3 result trees and the
# datasets, writes ONLY under $OUT. Phase 4 only, by stage; Phase 5 is NOT run by this script.
#
#   SCENES="USB_100109" bash run_phase45_synthetic.sh                 4A on one scene
#   SCENES="USB_100109" STAGE=all bash run_phase45_synthetic.sh       4A -> 4B -> 4C
#   DRY=1 bash run_phase45_synthetic.sh                                contracts + planned resources of all 10
#
# Joint type: the documented evaluation rule (results/rigeuler_tables.py); direction: the one in the run tree.
#   SCENES="USB_100109" STAGE=all RECOVERY_STRIDE=4 bash run_phase45_synthetic.sh   4A on all views, 4B/4C every 4th frame
# Env: SYN, RES, OUT, SCENES, STAGE, RUN_ID, RECOVERY_STRIDE (4B/4C frame stride, default 1), PY, DRY, EXTRA_ARGS
set -u
cd "$(dirname "$(readlink -f "$0")")" || exit 2
PY=${PY:-~/miniconda3/envs/rigeuler3dgs/bin/python}
SYN=${SYN:-"/media/samuelemara/Extreme SSD/Samuele/Backup_09_10_26/Sapien/output/preparation_8cams/partnet_subset_fixed_render_linear"}
RES=${RES:-"/media/samuelemara/Extreme SSD/Samuele/results_11_10_26"}
OUT=${OUT:-"$RES/ours_phase4_corrected_synthetic"}
SCENES=${SCENES:-"USB_100109"}
STAGE=${STAGE:-4a}
if [ -z "${RUN_ID:-}" ]; then
  case "$STAGE" in 4b|4c) echo "STAGE=$STAGE needs RUN_ID of a run whose 4A (and 4B) completed"; exit 2;; esac
  RUN_ID=run_$(date +%Y%m%d_%H%M%S)
fi
DRYARG=(); [ "${DRY:-0}" = 1 ] && DRYARG=(--dry-run)
RECOVERY_STRIDE=${RECOVERY_STRIDE:-1}
STRIDEARG=(); [ "$STAGE" != 4a ] && STRIDEARG=(--recovery-frame-stride "$RECOVERY_STRIDE")
read -r -a XARGS <<< "${EXTRA_ARGS:-}"

echo "phase4 synthetic: $(pwd) ($(git rev-parse --short HEAD)$(git diff --quiet HEAD -- . ':!submodules/sam3' || echo ' + uncommitted')) -> $OUT  stage $STAGE  run $RUN_ID"
STATUS=()
for s in $SCENES; do
  if [ "$s" = washer_103776 ]; then R="$RES/ours_v5_phase1_gates/washer_103776_canonical/pipeline/$s"
  else R="$RES/ours_v5_canonical_synthetic/$s/pipeline/$s"; fi
  B="$R/bidirectional"
  if [ -d "$B/forward_joint" ]; then DIR=forward; END=initial; else DIR=backward; END=final; fi
  JT=$("$PY" -c "import sys; sys.path.insert(0, 'results'); import rigeuler_tables as t; from pathlib import Path; \
print(t.joint_geometry(Path(sys.argv[1]))['type_decision'])" "$B/${DIR}_rigid" 2>/dev/null)
  if [ -z "$JT" ]; then STATUS+=("$s: no joint type (Phase-2 outputs missing?)"); continue; fi
  O="$OUT/$s/phase4"
  [ "${DRY:-0}" = 1 ] || mkdir -p "$O"
  echo "=== $s  $DIR  $JT  $(date +%T)"
  P4="$O/$RUN_ID"; [ "${DRY:-0}" = 1 ] || mkdir -p "$P4"
  case "$STAGE" in
    4a|all) SARGS=(--stage "$STAGE" --source "$SYN/$s" --direction "$DIR" --model "$R/static_bidi/$END" \
            --phase2-dir "$B/${DIR}_rigid" --phase3-dir "$B/${DIR}_joint" --joint-type "$JT" --results-root "$RES" --output "$P4");;
    4b) SARGS=(--stage 4b --phase4a "$P4/4a_original_membership");;
    4c) SARGS=(--stage 4c --phase4a "$P4/4a_original_membership" --phase4b "$P4/4b_new_surface_proposals");;
    *) echo "STAGE must be 4a, 4b, 4c or all"; exit 2;;
  esac
  "$PY" -u tools/run_phase4_recovery.py "${SARGS[@]}" "${STRIDEARG[@]}" "${DRYARG[@]}" "${XARGS[@]}" \
      2>&1 | grep --line-buffered -v Warning | { if [ "${DRY:-0}" = 1 ]; then cat; else tee -a "$P4/console_$STAGE.log"; fi; }
  RC=${PIPESTATUS[0]}
  if [ "${DRY:-0}" != 1 ]; then      # membership debug page (CPU, seconds): <run>/html/phase4_membership_debug.html
    "$PY" tools/phase4_debug_html.py --phase4-dir "$P4" 2>&1 | grep -v Warning | tee -a "$P4/console_$STAGE.log"
  fi
  STATUS+=("$s phase4 stage $STAGE rc=$RC -> $O/$RUN_ID (4a_original_membership / 4b_new_surface_proposals / 4c_accepted_moving)")
done
echo; echo "=== summary  $(date)"; printf '  %s\n' "${STATUS[@]}"
