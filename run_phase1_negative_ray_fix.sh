#!/usr/bin/env bash
# robot_dog silhouette-induced static drift: ablation, corrected run, gate, then black_laptop (2026-11-17)
#
#   T  ablation (read-only diagnostic, tools/diagnose_phase1_negative_ray.py) on robot_dog: six arms
#      {legacy, certified q=1, calibrated q} x {legacy, observable group}, first 40 frames from the SAME
#      endpoint / P0 / sensor model / deterministic reductions, forces decomposed per camera and channel,
#      frozen-state replays at processing indices 5 15 30                 -> $OUT/ablation_robot_dog/report.md
#   A  robot_dog Phases 1-3 with the corrected Phase 1 (occlusion-consistent, calibrated q, observable)
#                                                                         -> $OUT/occ1/...
#   G  gate (CPU): static displacement on one fixed static set vs v6 / draw1 / draw2 / certified (real3);
#      occlusion drift; supports and joints                              -> $OUT/comparison/
#   B  black_laptop Phases 1-3 + the same comparison -- ONLY when robot_dog passed G (FORCE=1 overrides)
#
#   bash run_phase1_negative_ray_fix.sh 2>&1 | tee -a ~/phase1_negative_ray_fix.log
# Env: OUT (default on the SSD, under $RES), PARTS ("T A G B"), FORCE, PY
set -u
cd "$(dirname "$(readlink -f "$0")")" || exit 2
HERE=$(pwd)
PY=${PY:-~/miniconda3/envs/rigeuler3dgs/bin/python}
RES=${RES:-"/media/samuelemara/Extreme SSD/Samuele/results_11_10_26"}
SHADOW=${SHADOW:-~/real_poses_before_20261112}
V5="$RES/ours_v5_two_cams"; V6="$RES/ours_v6_two_cams"; V8="$RES/ours_v8_overnight_real"
CERT="$RES/ours_v9_phase1_occlusion_real3/occ1"
OUT=${OUT:-"$RES/ours_v10b_negative_ray_legacy_group"}
PARTS=${PARTS:-"T A G B"}
FORCE=${FORCE:-0}
# group update of the corrected Phase 1 (parts A/B). The 2026-11-17 ablation (ours_v10_negative_ray_fix/
# ablation_robot_dog) isolated the drift to --phase1-group-update observable (legacy/legacy 3.5 mm ->
# legacy/observable 22.8 mm; calibrated/legacy 2.3 mm), so the default is now legacy.
P1GRP=${P1GRP:-legacy}
case "$P1GRP" in legacy|observable) ;; *) echo "P1GRP must be legacy or observable" >&2; exit 2;; esac
# an earlier corrected run shown beside the new one in the gate tables (not a reference of the gate)
PREV=${PREV:-"$HOME/results_local/ours_v10_negative_ray_fix/occ1"}
PREVARGS=(); PREVDRAW=()
if [ -d "$PREV" ] && [ "$(readlink -f "$PREV")" != "$(readlink -f "$OUT/occ1" 2>/dev/null)" ]; then
  PREVARGS=(--run prev="$PREV"); PREVDRAW=(--draw prev="$PREV")
fi
has() { [[ " $PARTS " == *" $1 "* ]]; }
mkdir -p "$OUT" || exit 2

# one copy at a time (a second launch would empty the first one's stage folders)
exec 9>"$OUT/.run.lock"
flock -n 9 || { echo "another run_phase1_negative_ray_fix.sh is running on $OUT -- refusing" >&2; exit 1; }
# stale JIT build locks (a killed build leaves one and the next run waits forever): removed only when
# no Phase-1 process of this worktree is running
if ! pgrep -f "$HERE/tools/(run_motion_coherence_support|diagnose_phase1_negative_ray)" > /dev/null &&
   ! pgrep -f "tools/(run_motion_coherence_support|diagnose_phase1_negative_ray).py" > /dev/null; then
  find submodules/diff-gaussian-rasterization -path '*/.build_*/*' -name lock -print -delete
fi
FREE=$(df -BG --output=avail "$OUT" | tail -1 | tr -dc 0-9)
[ "${FREE:-0}" -ge 12 ] || { echo "only ${FREE} GB free under $OUT; need >= 12" >&2; exit 2; }

{ echo "date $(date -Is)"; echo "worktree $HERE"; echo "commit $(git rev-parse HEAD) branch $(git branch --show-current)"
  echo "tracked diff sha256 $(git diff HEAD -- '*.py' '*.sh' '*.cu' '*.cpp' '*.h' '*.cuh' | sha256sum | cut -c1-64)"
  echo "untracked files:"; git status --short --untracked-files=all | grep '^??' | grep -E '\.(py|sh|cu|cuh|h)$' |
    while read -r _ f; do echo "  $(sha256sum "$f" | cut -c1-64)  $f"; done
} > "$OUT/provenance_$(date +%Y%m%d_%H%M%S).txt"
echo "negative-ray fix: $HERE ($(git rev-parse --short HEAD) on $(git branch --show-current) + uncommitted) -> $OUT; parts $PARTS"
STATUS=()

gate() {   # gate SCENE -> rc of the static-displacement gate
  local sc=$1
  mkdir -p "$OUT/comparison"
  CUDA_VISIBLE_DEVICES= "$PY" tools/compare_static_displacement.py --run v6="$V6" --run draw1="$V8/draw1" \
    --run draw2="$V8/draw2" --run certified="$CERT" "${PREVARGS[@]}" --run fix="$OUT/occ1" --reference v6 draw1 draw2 \
    --candidate fix --static-from v6 certified fix --scenes "$sc" --out "$OUT/comparison/$sc" < /dev/null 2>&1 |
    grep -E "gate|WARNING|->"
  local rc=${PIPESTATUS[0]}
  CUDA_VISIBLE_DEVICES= "$PY" tools/compare_phase1_occlusion_drift.py --run v6="$V6" --run draw1="$V8/draw1" \
    --run certified="$CERT" "${PREVARGS[@]}" --run fix="$OUT/occ1" --scenes "$sc" --out "$OUT/comparison/$sc" < /dev/null 2>&1 | tail -1
  CUDA_VISIBLE_DEVICES= "$PY" tools/compare_phase_draws.py --draw v6="$V6" --draw draw1="$V8/draw1" --draw draw2="$V8/draw2" \
    --draw certified="$CERT" "${PREVDRAW[@]}" --draw fix="$OUT/occ1" --scenes "$sc" --replay --out "$OUT/comparison/$sc" < /dev/null 2>&1 |
    grep -v Warning | tail -2
  return "$rc"
}

if has T; then
  n=final_robot_dog_episode_000000
  EP="$V6/$n/unanimous/pipeline/$n/static_bidi/initial"
  echo; echo "=== T  ablation robot_dog  $(date +%T)"
  "$PY" tools/diagnose_phase1_negative_ray.py --model "$EP" --source "$SHADOW/$n" \
    --sensor-model "$V5/$n/base/calibration/sensor_model.json" --sensor-policy ideal \
    --p0-geometry "$EP/geometry_information.npz" --p0-velocity-sigma 1.0 0.316227766 --frame-timestamps auto \
    --rgb-model-sigma 0.04 --depth-model-sigma 0.003 --velocity-tau 0.5 --motion-start-frame 36 --motion-end-frame 186 \
    --cameras cam_001,cam_002 --deterministic --max-frames 40 --frozen-frames 5 15 30 \
    --static-exclude-support "$V6/$n/unanimous/pipeline/$n/bidirectional/forward_rigid/support_coherence/support.json" \
                             "$CERT/$n/unanimous/pipeline/$n/bidirectional/forward_rigid/support_coherence/support.json" \
    --out "$OUT/ablation_robot_dog" < /dev/null 2>&1 | grep -v "motion evidence"
  STATUS+=("ablation robot_dog rc=${PIPESTATUS[0]} -> $OUT/ablation_robot_dog/report.md")
fi
if has A; then
  OUT="$OUT" PARTS=A SCENE_FILTER=robot_dog P1LIK=occlusion-consistent P1GRP="$P1GRP" P1REL=calibrated \
    bash run_phase1_occlusion_real.sh 2>&1 | grep -v "motion evidence" | tail -8
  STATUS+=("robot_dog Phases 1-3 (calibrated, group $P1GRP) rc=${PIPESTATUS[0]}")
fi
GATE_RD=1
if has G; then
  echo; echo "=== G  robot_dog gate  $(date +%T)"
  gate robot_dog; GATE_RD=$?
  STATUS+=("robot_dog gate rc=$GATE_RD ($([ $GATE_RD = 0 ] && echo PASS || echo 'FAIL / not evaluated')) -> $OUT/comparison/robot_dog/")
fi
if has B; then
  if [ "$GATE_RD" = 0 ] || [ "$FORCE" = 1 ]; then
    OUT="$OUT" PARTS=A SCENE_FILTER=black_laptop P1LIK=occlusion-consistent P1GRP="$P1GRP" P1REL=calibrated \
      bash run_phase1_occlusion_real.sh 2>&1 | grep -v "motion evidence" | tail -8
    STATUS+=("black_laptop Phases 1-3 (calibrated, group $P1GRP) rc=${PIPESTATUS[0]}")
    echo; echo "=== G  black_laptop gate  $(date +%T)"
    gate black_laptop; r=$?
    STATUS+=("black_laptop gate rc=$r -> $OUT/comparison/black_laptop/")
  else
    STATUS+=("black_laptop NOT RUN: robot_dog did not pass the gate (FORCE=1 runs it anyway)")
  fi
fi
echo; echo "=== summary  $(date)"; printf '  %s\n' "${STATUS[@]}"
