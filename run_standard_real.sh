#!/usr/bin/env bash
# STANDARD real-capture pipeline (status 2026-11-09, STATUS.md).
#
# One script for the recipe behind the v6 real results (results_11_10_26/ours_v6_two_cams):
#   1. calibration + endpoint bootstrap  (as ~/run_v5_two_cams.sh: calibrated RGB-D sensor model from the
#      static window, covariance-surface bootstrap of the stationary endpoint, 30000 Gaussians, 1000 steps)
#   2. Phase-0 geometry-consistency repair of that endpoint, one model per arm:
#        C          no repair
#        majority   tools/phase0_geometry_consistency.py --rule majority
#        unanimous  tools/phase0_geometry_consistency.py --rule unanimous
#      evidence = the 20 stationary frames next to the endpoint
#   3. Phases 1-3 with the v6 options (as ~/run_v6_two_cams.sh):
#        --velocity-tau 0.5                    mean-reverting velocity (estimator/velocity_decay.py)
#        --phase2-support-refine free-space    free-space support refinement, 1 round
#                                              (estimator/free_space_support.py)
#      declared model-mismatch sds rgb 0.04 / depth 0.003, canonical Phase-1 partition.
# v6 itself reused the v5 endpoints (steps 1-2), which this code produces unchanged: steps 1-2 are the
# same code in v5 and v6. Every stage sees only $CAMS. Real captures have no joint GT.
#
#   bash run_standard_real.sh 2>&1 | tee ~/standard_real.log
#   SCENE_FILTER="red_fridge" ARMS="C" bash run_standard_real.sh     one scene, one arm
#   DRY=1 bash run_standard_real.sh                                  print the commands only
#
# Env overrides: REAL, OUT, CAMS, ARMS, PHASES, NG, RI, VTAU (0 = off), REFINE (off|free-space), ROUNDS,
#                SCENE_FILTER (space-separated short names), TABLE, PY, DRY
set -u
cd "$(dirname "$(readlink -f "$0")")" || exit 2
PY=${PY:-~/miniconda3/envs/rigeuler3dgs/bin/python}
REAL=${REAL:-"/media/samuelemara/Extreme SSD/Samuele/Backup_09_10_26/Sapien/output/real"}
OUT=${OUT:-"/media/samuelemara/Extreme SSD/Samuele/results_11_10_26/ours_standard_real"}
CAMS=${CAMS:-cam_001,cam_002}
ARMS=${ARMS:-"C majority unanimous"}
PHASES=${PHASES:-"phase1 phase2 phase3"}
NG=${NG:-30000}
RI=${RI:-1000}
VTAU=${VTAU:-0.5}
REFINE=${REFINE:-free-space}
ROUNDS=${ROUNDS:-1}
MIS=(--rgb-model-sigma 0.04 --depth-model-sigma 0.003)   # declared, from the fridge residual report 80-99
export STALL_MIN=${STALL_MIN:-20}
X=(); [ "${DRY:-0}" = 1 ] && X=(echo)

# scene  direction  static [start end]  motion [start end]
# direction: metadata/direction/summary.json. red_fridge: the window of the calibrated bootstrap.
# The others: tools/estimate_motion_interval.py --cameras cam_001,cam_002 --lag 19.
TABLE=${TABLE:-"
black_box        backward 0 47 52 228
black_laptop     forward  0 50 55 228
light_blue_box   forward  0 30 35 200
matchbox         forward  0 56 61 207
orange_suitecase forward  0 46 51 162
red_fridge       forward  0 95 100 288
robot_dog        forward  0 31 36 186
wooden_box       forward  0 33 38 127
"}

echo "standard real: $(pwd) ($(git rev-parse --short HEAD)$(git diff --quiet HEAD -- . ':!submodules' || echo ' + uncommitted')) -> $OUT"
echo "cameras $CAMS, arms $ARMS, velocity-tau $VTAU, support refine $REFINE/$ROUNDS"
STATUS=()
while read -r sc dir cs ce ms me; do
  [ -z "$sc" ] && continue
  if [ -n "${SCENE_FILTER:-}" ] && [[ " $SCENE_FILTER " != *" $sc "* ]]; then continue; fi
  name="final_${sc}_episode_000000"
  ROOT="$OUT/$name"
  echo "=== $name  $dir  static [$cs, $ce]  motion [$ms, $me]  $(date +%T)"
  "${X[@]}" "$PY" tools/run_noise_aware_rigeuler.py --source "$REAL/$name" --dataset-kind real \
    --rgbd-noise-calibration --calibration-start-frame "$cs" --calibration-end-frame "$ce" \
    --motion-start-frame "$ms" --motion-end-frame "$me" --direction "$dir" --cameras "$CAMS" \
    --bootstrap-mode covariance_surface --sensor-noise-model calibrated \
    --num-gaussians "$NG" --refine-iters "$RI" --stages "calibration bootstrap" --out "$ROOT/base" 2>&1 | tail -3
  rc=${PIPESTATUS[0]}
  if [ "$rc" != 0 ]; then STATUS+=("$name base FAILED rc=$rc"); continue; fi
  CAL="$ROOT/base/calibration/sensor_model.json"
  if [ "$dir" = forward ]; then END=initial; E0=$(( ms > 20 ? ms - 20 : 0 )); E1=$ms
  else END=final; E0=$(( me - 20 )); E1=$me; fi
  EP="$ROOT/base/pipeline/$name/static_bidi/$END"
  for ARM in $ARMS; do
    PO="$ROOT/$ARM/pipeline"
    DST="$PO/$name/static_bidi/$END"
    if [ ! -d "$DST" ] && [ -z "${X[*]}" ]; then
      mkdir -p "$(dirname "$DST")"
      rm -rf "$DST.tmp"
      if [ "$ARM" = C ]; then
        cp -r "$EP" "$DST.tmp" && mv "$DST.tmp" "$DST"
      else
        "$PY" tools/phase0_geometry_consistency.py --model "$EP" --source "$REAL/$name" --cameras "$CAMS" \
          --frames "$E0" "$E1" --direction "$dir" --sensor-model "$CAL" --frame-timestamps auto "${MIS[@]}" \
          --p0-geometry "$EP/geometry_information.npz" --rule "$ARM" \
          --report "$ROOT/$ARM/phase0_repair.json" --out-model "$DST.tmp" > "$ROOT/$ARM/phase0_repair.log" 2>&1
        rc=$?
        if [ "$rc" != 0 ]; then tail -3 "$ROOT/$ARM/phase0_repair.log"; STATUS+=("$name $ARM repair FAILED rc=$rc"); continue; fi
        mv "$DST.tmp" "$DST"
      fi
    fi
    echo "--- $name, arm $ARM  $(date +%T)"
    "${X[@]}" env ITERS="$RI" bash run_bidirectional_all.sh -i "$REAL" -o "$PO" --scenes "$name" --stages "$PHASES" \
      --cameras "$CAMS" --motion-start-frame "$ms" --motion-end-frame "$me" --direction "$dir" \
      --frame-timestamps auto --sensor-model "$CAL" --bootstrap-p0 --p0-velocity-sigma 1.0 0.316227766 \
      --phase1-partition canonical "${MIS[@]}" \
      --velocity-tau "$VTAU" --phase2-support-refine "$REFINE" --phase2-support-refine-rounds "$ROUNDS" 2>&1 | tail -4
    STATUS+=("$name $ARM rc=${PIPESTATUS[0]}")
  done
done <<< "$TABLE"

echo; echo "=== summary  $(date)"; printf '  %s\n' "${STATUS[@]}"
