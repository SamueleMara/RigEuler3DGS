#!/usr/bin/env bash
# OVERNIGHT real batch (2026-11-15): Phases 1-5 on the 8 real captures + run-to-run study of Phase 1-3.
#
#   A  draw1:   Phases 1-3, all 8 scenes                                       (~1.3 h)
#   B  draw2..: REPEATS more independent Phase 1-3 runs, identical settings    (~1.3 h each)
#   D  comparison (CPU, read-only): v6, the cautious run, draw1..N             (~15 min)
#        -> $OUT/draw_comparison/draw_comparison.md   (tools/compare_phase_draws.py --replay)
#   C  Phase 4 + Phase 5 on draw1, all 8 scenes, red_fridge last               (~9-10 h)
# D runs before C, so the run-to-run report is ready in the morning even if Phase 4 is still going.
#
# Same settings as v6 (threshold-free coherence selection, so no scene stops), on the same Phase-0 endpoints.
# Camera poses: 2-camera scenes read the pre-registration SHADOW dataset (symlink tree; the v6 endpoints are in the
# old camera frame), orange_suitecase reads the dataset as it is with its 3-camera Phase 0 of 2026-11-14 (new frame).
# The comparison replays the cautious selection (0.8 / 0.25 / 3) on every draw, so its effect is measured without
# making scenes stop. COH_ETA / COH_RECRUIT / COH_TRIM apply it for real.
#
#   bash run_overnight_real.sh 2>&1 | tee -a ~/overnight_real.log
#   PARTS="A B D" ...        Phase 1-3 study only;   PARTS="C" ...   Phase 4/5 only (needs draw1)
#   SCENE_FILTER="red_fridge wooden_box" ...    DRY=1 ... (print the plan; builds the shadow only)
# Env: OUT, OUT45, SHADOW, PARTS, REPEATS, RUN_ID, SCENE_FILTER, COH_ETA, COH_RECRUIT, COH_TRIM, VTAU, ROUNDS, PY, DRY
set -u
cd "$(dirname "$(readlink -f "$0")")" || exit 2
PY=${PY:-~/miniconda3/envs/rigeuler3dgs/bin/python}
REAL=${REAL:-"/media/samuelemara/Extreme SSD/Samuele/Backup_09_10_26/Sapien/output/real"}
RES=${RES:-"/media/samuelemara/Extreme SSD/Samuele/results_11_10_26"}
BACKUP="$RES/robot_base_registration_applied_20261112/backups"
SHADOW=${SHADOW:-~/real_poses_before_20261112}
V5="$RES/ours_v5_two_cams"; V6="$RES/ours_v6_two_cams"; V6_3="$RES/ours_v6_three_cams"
CAUTIOUS="$RES/ours_v7_cautious_real_v2"          # 2026-11-15 cautious run, compared as one more draw
OUT=${OUT:-"$RES/ours_v8_overnight_real"}          # $OUT/draw1, $OUT/draw2, ...
OUT45=${OUT45:-"$RES/ours_phase45_overnight_real"}
PARTS=${PARTS:-"A B D C"}
REPEATS=${REPEATS:-1}
RUN_ID=${RUN_ID:-overnight_rs2}
SCENE_FILTER=${SCENE_FILTER:-}
COH_ETA=${COH_ETA:-0}; COH_RECRUIT=${COH_RECRUIT:-1}; COH_TRIM=${COH_TRIM:-0}
VTAU=${VTAU:-0.5}; ROUNDS=${ROUNDS:-1}
DRY=${DRY:-0}
ARM=unanimous
MIS=(--rgb-model-sigma 0.04 --depth-model-sigma 0.003)
# scene  direction  motion [start end]   -- identical to ~/run_v6_two_cams.sh; C runs in this order (fridge last)
TABLE="matchbox forward 61 207
wooden_box forward 38 127
black_laptop forward 55 228
robot_dog forward 36 186
orange_suitecase forward 51 162
light_blue_box forward 35 200
black_box backward 52 228
red_fridge forward 100 288"
has() { [[ " $PARTS " == *" $1 "* ]]; }
want() { [ -z "$SCENE_FILTER" ] || [[ " $SCENE_FILTER " == *" $1 "* ]]; }

shadow() {   # shadow NAME: the pre-registration dataset of NAME (built once; symlinks only, the dataset is untouched)
  local n=$1 d="$SHADOW/$1"
  if [ ! -d "$d" ]; then
    [ -d "$BACKUP/$n/before" ] || { echo "no pre-registration backup $BACKUP/$n/before" >&2; return 1; }
    mkdir -p "$SHADOW"; rm -rf "$d.tmp"
    cp -rs "$REAL/$n" "$d.tmp" || return 1
    (cd "$BACKUP/$n/before" && find . -type f) | while read -r f; do ln -sfn "$BACKUP/$n/before/${f#./}" "$d.tmp/${f#./}"; done
    mv "$d.tmp" "$d"
  fi
  [ "$(readlink "$d/colmap/sparse/0/images.bin")" = "$BACKUP/$n/before/colmap/sparse/0/images.bin" ] ||
    { echo "$d/colmap/sparse/0/images.bin is not the pre-registration file" >&2; return 1; }
}

config() {   # config SCENE -> name END CAMS EPROOT CAL SRCROOT
  name="final_${1}_episode_000000"
  [ "$dir" = forward ] && END=initial || END=final
  if [ "$1" = orange_suitecase ]; then
    CAMS=cam_000,cam_001,cam_002; EPROOT="$V6_3"; CAL="$V6_3/$name/base/calibration/sensor_model.json"; SRCROOT="$REAL"
  else
    CAMS=cam_001,cam_002; EPROOT="$V6"; CAL="$V5/$name/base/calibration/sensor_model.json"
    shadow "$name" || return 1; SRCROOT="$SHADOW"
  fi
  [ -f "$CAL" ] || { echo "no sensor model $CAL" >&2; return 1; }
}

phase13() {   # phase13 DRAWROOT SCENE
  local PO="$1/$name/$ARM/pipeline" SRC_EP="$EPROOT/$name/$ARM/pipeline/$name/static_bidi/$END"
  local DST="$1/$name/$ARM/pipeline/$name/static_bidi/$END"
  [ -d "$SRC_EP" ] || { STATUS+=("$2 $(basename "$1") SKIPPED (no endpoint $SRC_EP)"); return; }
  if [ ! -d "$DST" ] && [ "$DRY" != 1 ]; then
    mkdir -p "$(dirname "$DST")"; rm -rf "$DST.tmp"; cp -r "$SRC_EP" "$DST.tmp" && mv "$DST.tmp" "$DST"
  fi
  echo; echo "=== $(basename "$1")  $2  $dir  motion [$ms, $me]  $CAMS  poses from $SRCROOT  $(date +%T)"
  DRY="$DRY" ITERS=1000 bash run_bidirectional_all.sh -i "$SRCROOT" -o "$PO" --scenes "$name" \
    --stages "phase1 phase2 phase3" --cameras "$CAMS" --motion-start-frame "$ms" --motion-end-frame "$me" \
    --direction "$dir" --frame-timestamps auto --sensor-model "$CAL" --bootstrap-p0 \
    --p0-velocity-sigma 1.0 0.316227766 --phase1-partition canonical "${MIS[@]}" \
    --velocity-tau "$VTAU" --phase2-support-refine free-space --phase2-support-refine-rounds "$ROUNDS" \
    --coherence-min-eta-cv "$COH_ETA" --coherence-recruit-ratio "$COH_RECRUIT" --coherence-trim-k "$COH_TRIM" \
    < /dev/null 2>&1 | tail -4
  STATUS+=("$2 $(basename "$1") phase1-3 rc=${PIPESTATUS[0]}")
  local PS="$PO/$name/bidirectional/$dir/predicted_support.json"
  [ -f "$PS" ] && "$PY" -c "import json,sys; d=json.load(open(sys.argv[1])); e=d.get('extraction_summary') or {}; \
c=e.get('selected_candidate') or {}; r=c.get('rigidity',{}); w=c.get('rigidity_world',{}); \
print('  support:', d['status'], len(d['dynamic_gaussians']), 'Gaussians; seed eta_cv rel %.3f world %.3f' \
% (r.get('eta_cv', float('nan')), w.get('eta_cv', float('nan'))))" "$PS" < /dev/null
}

echo "overnight real: $(pwd) ($(git rev-parse --short HEAD)$(git diff --quiet HEAD -- . ':!submodules' || echo ' + uncommitted'))"
echo "  parts: $PARTS, repeats: $REPEATS, coherence eta>=$COH_ETA recruit $COH_RECRUIT trim $COH_TRIM;" \
     "Phase 1-3 -> $OUT/draw*; Phase 4/5 -> $OUT45 ($RUN_ID)"
STATUS=()
DRAWS=(draw1); for ((i = 2; i <= REPEATS + 1; i++)); do DRAWS+=("draw$i"); done
for D in "${DRAWS[@]}"; do
  if [ "$D" = draw1 ]; then has A || continue; else has B || continue; fi
  while read -r sc dir ms me <&3; do
    [ -z "$sc" ] || ! want "$sc" && continue
    config "$sc" || { STATUS+=("$sc SKIPPED (config)"); continue; }
    phase13 "$OUT/$D" "$sc"
  done 3<<< "$TABLE"
done

if has D && [ "$DRY" != 1 ]; then
  echo; echo "=== D  run-to-run comparison  $(date +%T)"
  DA=(--draw v6="$V6" --override v6:orange_suitecase="$V6_3")
  [ -d "$CAUTIOUS" ] && DA+=(--draw cautious="$CAUTIOUS")
  for D in "${DRAWS[@]}"; do [ -d "$OUT/$D" ] && DA+=(--draw "$D=$OUT/$D"); done
  SCN=""; while read -r sc _ <&3; do want "$sc" && SCN+="$sc "; done 3<<< "$TABLE"
  CUDA_VISIBLE_DEVICES= "$PY" tools/compare_phase_draws.py "${DA[@]}" --scenes "$SCN" --replay \
    --out "$OUT/draw_comparison" < /dev/null 2>&1 | grep -v Warning | tail -3
  STATUS+=("comparison rc=${PIPESTATUS[0]} -> $OUT/draw_comparison/draw_comparison.md")
fi

if has C; then
  while read -r sc dir ms me <&3; do
    [ -z "$sc" ] || ! want "$sc" && continue
    config "$sc" || { STATUS+=("$sc SKIPPED (config)"); continue; }
    echo; echo "=== C  $sc  Phase 4/5 on draw1  $(date +%T)"
    REAL="$SRCROOT" P03="$OUT/draw1" CAMS="$CAMS" OUT="$OUT45" STAGE=all RUN_ID="$RUN_ID" SCENES="$sc" ARMS="$ARM" \
      DRY="$DRY" EXTRA_ARGS="--recovery-frame-stride 2" bash run_phase45_real.sh < /dev/null 2>&1 | grep -a \
      --line-buffered -E "^===|COMPLETE|INCOMPLETE|core check|rc=|no joint type|Error|Traceback"
    if [ "$DRY" != 1 ]; then
      OUT="$OUT45" RUN_ID="$RUN_ID" SCENES="$sc/$ARM" bash run_phase5_synthetic.sh < /dev/null 2>&1 | grep -a \
        --line-buffered -E "^===|done in|masked PSNR|SKIPPED|rc=|Error|Traceback"
    fi
  done 3<<< "$TABLE"
fi
echo; echo "=== summary  $(date)"; printf '  %s\n' "${STATUS[@]}"
