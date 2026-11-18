#!/usr/bin/env bash
# Phase-1 occlusion fix on the real captures (2026-11-16)
# RigEuler3DGS_PHASE1_COHERENCE_CODEX_IMPLEMENTATION_SPEC_2026-11-16.md, Part I only.
#
#   A  occ1:          Phases 1-3 with --phase1-likelihood occlusion-consistent
#                     --phase1-group-update observable --phase1-deterministic
#                     (coherence selection, Phase 2 and Phase 3 are the v6/v8 code, unchanged)
#   R  occ1_repeat:   Phase 1 only, identical settings: is the new Phase 1 bitwise repeatable?
#   D  comparison (CPU, read-only):
#        occlusion drift of static Gaussians while covered: v6 / draw1 / occ1
#            -> $OUT/comparison/occlusion_drift.md   (tools/compare_phase1_occlusion_drift.py)
#        supports and joints: v6 / draw1 / draw2 / occ1
#            -> $OUT/comparison/draw_comparison.md   (tools/compare_phase_draws.py --replay)
#        Phase-1 repeatability: max |x_occ1 - x_occ1_repeat| per scene -> $OUT/comparison/repeatability.txt
#
# Same endpoints, cameras, sensor model, motion windows and Phase 2/3 settings as the 2026-11-15 overnight run
# (run_overnight_real.sh), so the only change in A is Phase 1.
#
#   bash run_phase1_occlusion_real.sh 2>&1 | tee -a ~/phase1_occlusion_real.log
#   PARTS="A D" ...            skip the repeat;   SCENE_FILTER="robot_dog wooden_box" ...;   DRY=1 ... (plan only)
# Env: OUT, PARTS, SCENE_FILTER, VTAU, ROUNDS, PY, DRY, P1LIK, P1GRP, P1REL (calibrated|certified|none)
set -u
cd "$(dirname "$(readlink -f "$0")")" || exit 2
PY=${PY:-~/miniconda3/envs/rigeuler3dgs/bin/python}
REAL=${REAL:-"/media/samuelemara/Extreme SSD/Samuele/Backup_09_10_26/Sapien/output/real"}
RES=${RES:-"/media/samuelemara/Extreme SSD/Samuele/results_11_10_26"}
BACKUP="$RES/robot_base_registration_applied_20261112/backups"
SHADOW=${SHADOW:-~/real_poses_before_20261112}
V5="$RES/ours_v5_two_cams"; V6="$RES/ours_v6_two_cams"; V6_3="$RES/ours_v6_three_cams"
V8="$RES/ours_v8_overnight_real"
OUT=${OUT:-"$RES/ours_v9_phase1_occlusion_real"}
PARTS=${PARTS:-"A R D"}
SCENE_FILTER=${SCENE_FILTER:-}
VTAU=${VTAU:-0.5}; ROUNDS=${ROUNDS:-1}
P1LIK=${P1LIK:-occlusion-consistent}; P1GRP=${P1GRP:-observable}; P1REL=${P1REL:-calibrated}
RELARGS=(); [ "$P1LIK" = occlusion-consistent ] && RELARGS=(--negative-ray-reliability "$P1REL")
DRY=${DRY:-0}
ARM=unanimous
MIS=(--rgb-model-sigma 0.04 --depth-model-sigma 0.003)
# scene  direction  motion [start end]: the four scenes whose support failed, then two controls (fridge last)
TABLE="robot_dog forward 36 186
black_laptop forward 55 228
black_box backward 52 228
wooden_box forward 38 127
light_blue_box forward 35 200
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

run13() {   # run13 DRAWROOT SCENE STAGES
  local PO="$1/$name/$ARM/pipeline" SRC_EP="$EPROOT/$name/$ARM/pipeline/$name/static_bidi/$END"
  local DST="$1/$name/$ARM/pipeline/$name/static_bidi/$END"
  [ -d "$SRC_EP" ] || { STATUS+=("$2 $(basename "$1") SKIPPED (no endpoint $SRC_EP)"); return; }
  if [ ! -d "$DST" ] && [ "$DRY" != 1 ]; then
    mkdir -p "$(dirname "$DST")"; rm -rf "$DST.tmp"; cp -r "$SRC_EP" "$DST.tmp" && mv "$DST.tmp" "$DST"
  fi
  echo; echo "=== $(basename "$1")  $2  $dir  motion [$ms, $me]  $CAMS  stages [$3]  $(date +%T)"
  DRY="$DRY" ITERS=1000 bash run_bidirectional_all.sh -i "$SRCROOT" -o "$PO" --scenes "$name" \
    --stages "$3" --cameras "$CAMS" --motion-start-frame "$ms" --motion-end-frame "$me" \
    --direction "$dir" --frame-timestamps auto --sensor-model "$CAL" --bootstrap-p0 \
    --p0-velocity-sigma 1.0 0.316227766 --phase1-partition canonical "${MIS[@]}" \
    --velocity-tau "$VTAU" --phase2-support-refine free-space --phase2-support-refine-rounds "$ROUNDS" \
    --phase1-likelihood "$P1LIK" --phase1-group-update "$P1GRP" "${RELARGS[@]}" --phase1-deterministic \
    < /dev/null 2>&1 | tail -4
  STATUS+=("$2 $(basename "$1") [$3] rc=${PIPESTATUS[0]}")
  local PS="$PO/$name/bidirectional/$dir/predicted_support.json"
  [ -f "$PS" ] && "$PY" -c "import json,sys; d=json.load(open(sys.argv[1])); e=d.get('extraction_summary') or {}; \
c=e.get('selected_candidate') or {}; r=c.get('rigidity',{}); w=c.get('rigidity_world',{}); \
print('  support:', d['status'], len(d['dynamic_gaussians']), 'Gaussians; seed eta_cv rel %.3f world %.3f' \
% (r.get('eta_cv', float('nan')), w.get('eta_cv', float('nan'))))" "$PS" < /dev/null
  local OB="$PO/$name/bidirectional/$dir/phase1_observability/frames.jsonl"
  [ -f "$OB" ] && "$PY" -c "import json,sys; R=[json.loads(l) for l in open(sys.argv[1])]; \
s=lambda k: sum((r.get(k) or 0) for r in R); \
st={}; [st.__setitem__(c['camera'], {kk: st.get(c['camera'],{}).get(kk,0)+c[kk] for kk in ('OBJECT_SUPPORTED','KNOWN_NONOBJECT_DEPTH','UNKNOWN')}) for r in R for c in r.get('ray_status',[])]; \
print('  phase1: %d frames, stop reasons %s, prediction-only member-frames %d' % (len(R), \
{x: sum(1 for r in R if r['stop_reason']==x) for x in set(r['stop_reason'] for r in R)}, s('prediction_only_members'))); \
print('  ray status totals:', st)" "$OB" < /dev/null
}

echo "phase1 occlusion real: $(pwd) ($(git rev-parse --short HEAD) on $(git branch --show-current)$(git diff --quiet HEAD -- . ':!submodules' || echo ' + uncommitted'))"
echo "  parts: $PARTS, Phase 1: --phase1-likelihood $P1LIK --phase1-group-update $P1GRP ${RELARGS[*]} --phase1-deterministic -> $OUT"
if [ "$DRY" != 1 ]; then
  mkdir -p "$OUT"
  { echo "date $(date -Is)"; echo "worktree $(pwd)"; echo "commit $(git rev-parse HEAD) branch $(git branch --show-current)"
    echo "tracked diff sha256 $(git diff HEAD -- '*.py' '*.sh' '*.cu' '*.cpp' '*.h' '*.cuh' | sha256sum | cut -c1-64)"
    echo "untracked files:"; git status --short --untracked-files=all | grep '^??' | grep -E '\.(py|sh|cu|cuh|h)$' | while read -r _ f; do
      echo "  $(sha256sum "$f" | cut -c1-64)  $f"; done
  } > "$OUT/provenance_$(date +%Y%m%d_%H%M%S).txt"
fi
STATUS=()
if has A; then
  while read -r sc dir ms me <&3; do
    [ -z "$sc" ] || ! want "$sc" && continue
    config "$sc" || { STATUS+=("$sc SKIPPED (config)"); continue; }
    run13 "$OUT/occ1" "$sc" "phase1 phase2 phase3"
  done 3<<< "$TABLE"
fi
if has R; then
  while read -r sc dir ms me <&3; do
    [ -z "$sc" ] || ! want "$sc" && continue
    config "$sc" || { STATUS+=("$sc SKIPPED (config)"); continue; }
    run13 "$OUT/occ1_repeat" "$sc" "phase1"
  done 3<<< "$TABLE"
fi

if has D && [ "$DRY" != 1 ]; then
  echo; echo "=== D  comparisons  $(date +%T)"
  SCN=""; while read -r sc _ <&3; do want "$sc" && SCN+="$sc "; done 3<<< "$TABLE"
  mkdir -p "$OUT/comparison"
  CUDA_VISIBLE_DEVICES= "$PY" tools/compare_phase1_occlusion_drift.py --run v6="$V6" --run draw1="$V8/draw1" \
    --run occ1="$OUT/occ1" --scenes "$SCN" --out "$OUT/comparison" < /dev/null 2>&1 | grep -v Warning | tail -2
  STATUS+=("occlusion drift rc=${PIPESTATUS[0]} -> $OUT/comparison/occlusion_drift.md")
  DA=(--draw v6="$V6")
  for D in draw1 draw2; do [ -d "$V8/$D" ] && DA+=(--draw "$D=$V8/$D"); done
  DA+=(--draw occ1="$OUT/occ1")
  CUDA_VISIBLE_DEVICES= "$PY" tools/compare_phase_draws.py "${DA[@]}" --scenes "$SCN" --replay \
    --out "$OUT/comparison" < /dev/null 2>&1 | grep -v Warning | tail -3
  STATUS+=("supports rc=${PIPESTATUS[0]} -> $OUT/comparison/draw_comparison.md")
  if [ -d "$OUT/occ1_repeat" ]; then
    CUDA_VISIBLE_DEVICES= "$PY" - "$OUT" $SCN > "$OUT/comparison/repeatability.txt" <<'EOF'
import glob, sys
from pathlib import Path
import numpy as np
out, scenes = Path(sys.argv[1]), sys.argv[2:]
for sc in scenes:
    n = f"final_{sc}_episode_000000"
    a = sorted(glob.glob(str(out / "occ1" / n / "unanimous/pipeline" / n / "bidirectional/*/phase1_centers/centers_frame_*.npy")))
    b = sorted(glob.glob(str(out / "occ1_repeat" / n / "unanimous/pipeline" / n / "bidirectional/*/phase1_centers/centers_frame_*.npy")))
    if not a or len(a) != len(b):
        print(f"{sc}: missing ({len(a)} vs {len(b)} frames)"); continue
    d = max(float(np.abs(np.load(x) - np.load(y)).max()) for x, y in zip(a, b))
    print(f"{sc}: {len(a)} frames, max |x_occ1 - x_repeat| = {d:.3e} m {'(bitwise identical)' if d == 0 else ''}")
EOF
    STATUS+=("repeatability -> $OUT/comparison/repeatability.txt"); cat "$OUT/comparison/repeatability.txt"
  fi
fi
echo; echo "=== summary  $(date)"; printf '  %s\n' "${STATUS[@]}"
