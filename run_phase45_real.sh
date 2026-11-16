#!/usr/bin/env bash
# Unified Phase 4 (spec 2026-11-11: 4A -> 4B -> 4C as separate stages) on the v6 real two-camera runs (cam_001, cam_002 -- the contract refuses any other
# set, because those are the cameras Phase 0-3 used). Reads the Phase 0-3 trees, writes ONLY under $OUT.
# Phase 4 only, by stage; Phase 5 is NOT run by this script (the directive: review Phase 4 first).
#
#   SCENES="red_fridge" bash run_phase45_real.sh                      4A on one scene (the directive's next step)
#   SCENES="red_fridge" STAGE=all bash run_phase45_real.sh            4A -> 4B -> 4C (only after 4A was reviewed)
#   SCENES="red_fridge" ARMS="C majority unanimous" bash run_phase45_real.sh   other Phase-0 arms
#   DRY=1 bash run_phase45_real.sh                                    contracts + planned resources only
#
# Joint type per run: the documented R/L rule (results/rigeuler_tables.py), no GT.
# Env: REAL, RES, OUT, SCENES (short names), ARMS, STAGE (4a|4b|4c|all), RUN_ID, PY, DRY, EXTRA_ARGS,
#      P03 (Phase 0-3 root, default $RES/ours_v6_two_cams), CAMS (must equal the cameras Phase 0-3 used, default cam_001,cam_002)
#   P03="$RES/ours_v6_three_cams" CAMS=cam_000,cam_001,cam_002 SCENES=orange_suitecase ...   a 3-camera Phase 0-3 tree
# Every run: <OUT>/<scene>/<arm>/phase4/<RUN_ID>/ with phase4.log (line-buffered) + console log next to it.
set -u
cd "$(dirname "$(readlink -f "$0")")" || exit 2
PY=${PY:-~/miniconda3/envs/rigeuler3dgs/bin/python}
REAL=${REAL:-"/media/samuelemara/Extreme SSD/Samuele/Backup_09_10_26/Sapien/output/real"}
RES=${RES:-"/media/samuelemara/Extreme SSD/Samuele/results_11_10_26"}
OUT=${OUT:-"$RES/ours_phase4_corrected_real"}
SCENES=${SCENES:-"red_fridge"}
ARMS=${ARMS:-"unanimous"}
STAGE=${STAGE:-4a}
if [ -z "${RUN_ID:-}" ]; then
  case "$STAGE" in 4b|4c) echo "STAGE=$STAGE needs RUN_ID of a run whose 4A (and 4B) completed"; exit 2;; esac
  RUN_ID=run_$(date +%Y%m%d_%H%M%S)
fi
CAMS=${CAMS:-cam_001,cam_002}
P03=${P03:-"$RES/ours_v6_two_cams"}
DRYARG=(); [ "${DRY:-0}" = 1 ] && DRYARG=(--dry-run)
read -r -a XARGS <<< "${EXTRA_ARGS:-}"

echo "phase4 real: $(pwd) ($(git rev-parse --short HEAD)$(git diff --quiet HEAD -- . ':!submodules/sam3' || echo ' + uncommitted')) -> $OUT  stage $STAGE  run $RUN_ID  cameras $CAMS  from $P03"
STATUS=()
for sc in $SCENES; do
  name="final_${sc}_episode_000000"
  for arm in $ARMS; do
    R="$P03/$name/$arm/pipeline/$name"
    B="$R/bidirectional"
    if [ -d "$B/forward_joint" ]; then DIR=forward; END=initial; else DIR=backward; END=final; fi
    JT=$("$PY" -c "import sys; sys.path.insert(0, 'results'); import rigeuler_tables as t; from pathlib import Path; \
print(t.joint_geometry(Path(sys.argv[1]))['type_decision'])" "$B/${DIR}_rigid" 2>/dev/null)
    if [ -z "$JT" ]; then STATUS+=("$sc $arm: no joint type"); continue; fi
    O="$OUT/$sc/$arm/phase4"
    [ "${DRY:-0}" = 1 ] || mkdir -p "$O"
    echo "=== $sc $arm  $DIR  $JT  $(date +%T)"
    P4="$O/$RUN_ID"; [ "${DRY:-0}" = 1 ] || mkdir -p "$P4"
    case "$STAGE" in
      4a|all) SARGS=(--stage "$STAGE" --source "$REAL/$name" --direction "$DIR" --model "$R/static_bidi/$END" \
              --phase2-dir "$B/${DIR}_rigid" --phase3-dir "$B/${DIR}_joint" --joint-type "$JT" --cameras "$CAMS" --results-root "$RES" --output "$P4");;
      4b) SARGS=(--stage 4b --phase4a "$P4/4a_original_membership");;
      4c) SARGS=(--stage 4c --phase4a "$P4/4a_original_membership" --phase4b "$P4/4b_new_surface_proposals");;
      *) echo "STAGE must be 4a, 4b, 4c or all"; exit 2;;
    esac
    "$PY" -u tools/run_phase4_recovery.py "${SARGS[@]}" "${DRYARG[@]}" "${XARGS[@]}" \
        2>&1 | grep --line-buffered -v Warning | { if [ "${DRY:-0}" = 1 ]; then cat; else tee -a "$P4/console_$STAGE.log"; fi; }
    RC=${PIPESTATUS[0]}
    if [ "${DRY:-0}" != 1 ]; then    # membership debug page (CPU, seconds): <run>/html/phase4_membership_debug.html
      "$PY" tools/phase4_debug_html.py --phase4-dir "$P4" 2>&1 | grep -v Warning | tee -a "$P4/console_$STAGE.log"
    fi
    STATUS+=("$sc $arm phase4 stage $STAGE rc=$RC -> $O/$RUN_ID (4a_original_membership / 4b_new_surface_proposals / 4c_accepted_moving)")
  done
done
echo; echo "=== summary  $(date)"; printf '  %s\n' "${STATUS[@]}"
