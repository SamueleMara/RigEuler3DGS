#!/usr/bin/env bash
# CAUTIOUS real batch: Phases 1-5 on the 8 real captures. The ONLY difference from v6 is the stricter
# motion-coherence selection (three opt-in flags of the coherence extractor, run at the end of the Phase-1 tool).
# Phase 0 (calibration, bootstrap, Phase-0 repair) is NOT redone: endpoint and sensor models are copied from v6.
#
# CAMERA POSES (2026-11-15): on 2026-11-12 23:44 a robot-base registration was applied to every final_* dataset
# (metadata/robot_base_registration.json): every camera moved by one shared transform, 3.62 deg and 5.6-6.7 cm. The
# v5/v6 Phase-0 models were trained BEFORE that, in the old frame. Running them against the re-exported poses
# misaligns every view (this is what broke the 2026-11-14 cautious run and the Phase-4 runs started after Nov 12).
# So every scene whose endpoint comes from v6 (2 cameras) reads a SHADOW dataset: a symlink tree of the dataset
# with the pre-registration files from robot_base_registration_applied_20261112/backups/<scene>/before linked
# over the re-exported ones. The dataset itself is never touched. orange_suitecase uses the 3-camera Phase 0 built
# on 2026-11-14 with the NEW poses, so it reads the dataset as it is.
#
# Cautious coherence settings (defaults of run_bidirectional_all.sh reproduce v6 exactly):
#   --coherence-min-eta-cv 0.8        the seed must be a confidently rigid MOVING body: held-out eta_cv >= 0.8 both
#                                     relative to its complement and in the world frame; otherwise Phase 1 reports
#                                     NO_COHERENT_MOVING_COMPONENT and that scene stops (no joint is invented)
#   --coherence-recruit-ratio 0.25    a node joins only if the core's motion explains >= 75% of its displacement
#   --coherence-trim-k 3              the support is trimmed to ONE rigid body (median + 3 MAD, iterated)
# Phase 2 free-space refinement: v6 setting (replace). P2RMODE=intersect makes it remove-only (not the default).
#
#   bash run_cautious_real.sh 2>&1 | tee -a ~/cautious_real_v2.log
#   SCENE_FILTER="red_fridge" bash run_cautious_real.sh            one scene
#   PARTS="p13" bash run_cautious_real.sh                          Phases 1-3 only (p13 | p4 | p5, any subset)
#   DRY=1 bash run_cautious_real.sh                                print the plan, write nothing (builds the shadow)
# Env: OUT, OUT45, SHADOW, ARMS, RUN_ID, PARTS, SCENE_FILTER, COH_ETA, COH_RECRUIT, COH_TRIM, P2RMODE, VTAU,
#      ROUNDS, PY, DRY
set -u
cd "$(dirname "$(readlink -f "$0")")" || exit 2
PY=${PY:-~/miniconda3/envs/rigeuler3dgs/bin/python}
REAL=${REAL:-"/media/samuelemara/Extreme SSD/Samuele/Backup_09_10_26/Sapien/output/real"}
RES=${RES:-"/media/samuelemara/Extreme SSD/Samuele/results_11_10_26"}
BACKUP="$RES/robot_base_registration_applied_20261112/backups"
SHADOW=${SHADOW:-~/real_poses_before_20261112}   # local ext4: the SSD cannot hold symlinks
V5="$RES/ours_v5_two_cams"            # 2-camera sensor models
V6="$RES/ours_v6_two_cams"            # 2-camera Phase-0 endpoints (old poses)
V6_3="$RES/ours_v6_three_cams"        # orange_suitecase, 3 cameras: endpoint + sensor model (new poses)
OUT=${OUT:-"$RES/ours_v7_cautious_real_v2"}
OUT45=${OUT45:-"$RES/ours_phase45_cautious_real_v2"}
ARMS=${ARMS:-unanimous}
RUN_ID=${RUN_ID:-cautious_rs2}
PARTS=${PARTS:-"p13 p4 p5"}
SCENE_FILTER=${SCENE_FILTER:-}
COH_ETA=${COH_ETA:-0.8}; COH_RECRUIT=${COH_RECRUIT:-0.25}; COH_TRIM=${COH_TRIM:-3}; P2RMODE=${P2RMODE:-replace}
VTAU=${VTAU:-0.5}; ROUNDS=${ROUNDS:-1}
DRY=${DRY:-0}
MIS=(--rgb-model-sigma 0.04 --depth-model-sigma 0.003)
# scene  direction  motion [start end]   -- identical to ~/run_v6_two_cams.sh
TABLE="black_box backward 52 228
black_laptop forward 55 228
light_blue_box forward 35 200
matchbox forward 61 207
orange_suitecase forward 51 162
red_fridge forward 100 288
robot_dog forward 36 186
wooden_box forward 38 127"
has() { [[ " $PARTS " == *" $1 "* ]]; }

shadow() {   # shadow NAME -> the pre-registration dataset of NAME (built once; symlinks only)
  local n=$1 d="$SHADOW/$1"
  if [ ! -d "$d" ]; then
    [ -d "$BACKUP/$n/before" ] || { echo "no pre-registration backup $BACKUP/$n/before" >&2; return 1; }
    mkdir -p "$SHADOW"; rm -rf "$d.tmp"
    cp -rs "$REAL/$n" "$d.tmp" || return 1
    (cd "$BACKUP/$n/before" && find . -type f) | while read -r f; do ln -sfn "$BACKUP/$n/before/${f#./}" "$d.tmp/${f#./}"; done
    mv "$d.tmp" "$d"
  fi
  # refuse a shadow whose poses are not the backup's
  [ "$(readlink "$d/colmap/sparse/0/images.bin")" = "$BACKUP/$n/before/colmap/sparse/0/images.bin" ] ||
    { echo "$d/colmap/sparse/0/images.bin is not the pre-registration file" >&2; return 1; }
}

echo "cautious real: $(pwd) ($(git rev-parse --short HEAD)$(git diff --quiet HEAD -- . ':!submodules' || echo ' + uncommitted'))"
echo "  eta_cv >= $COH_ETA (relative and world), recruit ratio $COH_RECRUIT, trim k $COH_TRIM, refine $P2RMODE," \
     "velocity-tau $VTAU; parts: $PARTS; Phase 1-3 -> $OUT; Phase 4/5 -> $OUT45 ($RUN_ID)"
STATUS=()
# fd 3: the scene table must not be readable by the children (ffmpeg in Phase 5 reads stdin and swallowed lines)
while read -r sc dir ms me <&3; do
  [ -z "$sc" ] && continue
  if [ -n "$SCENE_FILTER" ] && [[ " $SCENE_FILTER " != *" $sc "* ]]; then continue; fi
  name="final_${sc}_episode_000000"
  [ "$dir" = forward ] && END=initial || END=final
  if [ "$sc" = orange_suitecase ]; then
    CAMS=cam_000,cam_001,cam_002; EPROOT="$V6_3"; CAL="$V6_3/$name/base/calibration/sensor_model.json"
    SRCROOT="$REAL"; POSES="current (robot base) poses"
  else
    CAMS=cam_001,cam_002; EPROOT="$V6"; CAL="$V5/$name/base/calibration/sensor_model.json"
    shadow "$name" || { STATUS+=("$sc SKIPPED (no pre-registration shadow)"); continue; }
    SRCROOT="$SHADOW"; POSES="pre-registration poses ($SHADOW)"
  fi
  [ -f "$CAL" ] || { STATUS+=("$sc SKIPPED (no sensor model $CAL)"); continue; }
  for ARM in $ARMS; do
    echo; echo "=== $sc  $ARM  $dir  motion [$ms, $me]  $CAMS  $POSES  $(date +%T)"
    PO="$OUT/$name/$ARM/pipeline"
    if has p13; then
      SRC_EP="$EPROOT/$name/$ARM/pipeline/$name/static_bidi/$END"
      DST="$PO/$name/static_bidi/$END"
      [ -d "$SRC_EP" ] || { STATUS+=("$sc $ARM SKIPPED (no endpoint $SRC_EP)"); continue; }
      if [ ! -d "$DST" ] && [ "$DRY" != 1 ]; then
        mkdir -p "$(dirname "$DST")"; rm -rf "$DST.tmp"
        cp -r "$SRC_EP" "$DST.tmp" && mv "$DST.tmp" "$DST"
      fi
      DRY="$DRY" ITERS=1000 bash run_bidirectional_all.sh -i "$SRCROOT" -o "$PO" --scenes "$name" \
        --stages "phase1 phase2 phase3" --cameras "$CAMS" --motion-start-frame "$ms" --motion-end-frame "$me" \
        --direction "$dir" --frame-timestamps auto --sensor-model "$CAL" --bootstrap-p0 \
        --p0-velocity-sigma 1.0 0.316227766 --phase1-partition canonical "${MIS[@]}" \
        --velocity-tau "$VTAU" --phase2-support-refine free-space --phase2-support-refine-rounds "$ROUNDS" \
        --phase2-support-refine-mode "$P2RMODE" --coherence-min-eta-cv "$COH_ETA" \
        --coherence-recruit-ratio "$COH_RECRUIT" --coherence-trim-k "$COH_TRIM" < /dev/null 2>&1 | tail -6
      STATUS+=("$sc $ARM phase1-3 rc=${PIPESTATUS[0]} -> $PO/$name")
      PS="$PO/$name/bidirectional/$dir/predicted_support.json"
      if [ -f "$PS" ]; then
        "$PY" -c "import json,sys; d=json.load(open(sys.argv[1])); e=d.get('extraction_summary') or {}; \
s=e.get('selection',{}); c=e.get('selected_candidate') or {}; r=c.get('rigidity',{}); w=c.get('rigidity_world',{}); \
print('  support:', d['status'], len(d['dynamic_gaussians']), 'Gaussians;', \
('seed eta_cv rel %.3f world %.3f' % (r.get('eta_cv', float('nan')), w.get('eta_cv', float('nan')))) if c else s.get('reason'))" "$PS" < /dev/null
      fi
    fi
    if has p4; then
      REAL="$SRCROOT" P03="$OUT" CAMS="$CAMS" OUT="$OUT45" STAGE=all RUN_ID="$RUN_ID" SCENES="$sc" ARMS="$ARM" DRY="$DRY" \
        EXTRA_ARGS="--recovery-frame-stride 2" bash run_phase45_real.sh < /dev/null 2>&1 | grep -a --line-buffered -E \
        "^===|COMPLETE|INCOMPLETE|core check|rc=|no joint type|Error|Traceback"
    fi
    if has p5 && [ "$DRY" != 1 ]; then
      OUT="$OUT45" RUN_ID="$RUN_ID" SCENES="$sc/$ARM" bash run_phase5_synthetic.sh < /dev/null 2>&1 | grep -a --line-buffered -E \
        "^===|done in|masked PSNR|SKIPPED|rc=|Error|Traceback"
    fi
  done
done 3<<< "$TABLE"
echo; echo "=== summary  $(date)"; printf '  %s\n' "${STATUS[@]}"
