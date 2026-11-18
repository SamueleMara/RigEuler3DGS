#!/usr/bin/env bash
# FULL real pipeline with the corrected Phase 1 (2026-11-18), all 8 real captures:
#   p13   Phases 1-3 from THIS worktree: --phase1-likelihood occlusion-consistent --negative-ray-reliability calibrated
#         --phase1-group-update legacy --phase1-deterministic (the configuration of ours_v10b_negative_ray_legacy_group;
#         no cautious coherence flags), same endpoints / cameras / sensor models / motion windows as v6 / cautious
#   p4    Phase 4 (4A -> 4B -> 4C, --recovery-frame-stride 2) from RigEuler3DGS-p45 (as ours_phase45_cautious_real_v2)
#   p5    Phase 5 from RigEuler3DGS-p45 (branch phase5-sim-asset-export): its last step writes assets/<scene>/
#         (two 3DGS PLYs + asset.yaml + object.yaml; no physics -> BLOCKED_MISSING_PHYSICS)
#   urdf  PREVIEW URDF / USD / MJCF: tools/make_preview_physics_spec.py (convex hulls of the Gaussians, declared
#         density, observed-range joint limits, declared joint dynamics, unverified gravity) + the post-hoc exporter
#         -> <P5>/assets/<scene>.rev-preview/  status PHYSICS_PREVIEW_ASSUMPTIONS (labelled; nothing measured)
#
#   bash run_full_real_negray.sh 2>&1 | tee -a ~/full_real_negray.log
#   SCENE_FILTER="robot_dog red_fridge" ...      PARTS="p4 p5 urdf" ...      DRY=1 ... (plan only)
# A scene whose Phase 3 already finished in $OUT13 is not re-run (resume after an interruption); FORCE13=1 re-runs it.
# Env: OUT13, OUT45, RUN_ID, PARTS, SCENE_FILTER, FORCE13, VTAU, ROUNDS, PY, P45, DRY,
#      PREVIEW_DENSITY, PREVIEW_VELOCITY, PREVIEW_EFFORT, PREVIEW_DAMPING, PREVIEW_FRICTION
set -u
cd "$(dirname "$(readlink -f "$0")")" || exit 2
HERE=$(pwd)
PY=${PY:-~/miniconda3/envs/rigeuler3dgs/bin/python}
if [ -f "$HERE/tools/export_phase5_sim_asset.py" ]; then P45=${P45:-"$HERE"}   # merged tree
else P45=${P45:-"$HOME/Joint3DGS/RigEuler3DGS-p45"}; fi
REAL=${REAL:-"/media/samuelemara/Extreme SSD/Samuele/Backup_09_10_26/Sapien/output/real"}
RES=${RES:-"/media/samuelemara/Extreme SSD/Samuele/results_11_10_26"}
BACKUP="$RES/robot_base_registration_applied_20261112/backups"
SHADOW=${SHADOW:-~/real_poses_before_20261112}
V5="$RES/ours_v5_two_cams"; V6="$RES/ours_v6_two_cams"; V6_3="$RES/ours_v6_three_cams"
OUT13=${OUT13:-"$RES/ours_v11_negray_real"}
OUT45=${OUT45:-"$RES/ours_phase45_v11_negray_real"}
RUN_ID=${RUN_ID:-negray_rs2}
PARTS=${PARTS:-"p13 p4 p5 urdf"}
SCENE_FILTER=${SCENE_FILTER:-}
FORCE13=${FORCE13:-0}
VTAU=${VTAU:-0.5}; ROUNDS=${ROUNDS:-1}
DRY=${DRY:-0}
PREVIEW_DENSITY=${PREVIEW_DENSITY:-300}; PREVIEW_VELOCITY=${PREVIEW_VELOCITY:-1.0}; PREVIEW_EFFORT=${PREVIEW_EFFORT:-10}
PREVIEW_DAMPING=${PREVIEW_DAMPING:-0.1}; PREVIEW_FRICTION=${PREVIEW_FRICTION:-0.0}
export RIGEULER_SIM_PYTHON=${RIGEULER_SIM_PYTHON:-"$HOME/.venvs/rigeuler-sim-export/bin/python"}
export RIGEULER_SAPIEN_PYTHON=${RIGEULER_SAPIEN_PYTHON:-"$HOME/miniconda3/envs/parttwin_sim/bin/python"}
ARM=unanimous
MIS=(--rgb-model-sigma 0.04 --depth-model-sigma 0.003)
TABLE="robot_dog forward 36 186
black_laptop forward 55 228
black_box backward 52 228
light_blue_box forward 35 200
matchbox forward 61 207
orange_suitecase forward 51 162
red_fridge forward 100 288
wooden_box forward 38 127"
has() { [[ " $PARTS " == *" $1 "* ]]; }

[ -f "$P45/tools/export_phase5_sim_asset.py" ] || { echo "$P45 is not the phase5-sim-asset-export worktree" >&2; exit 2; }
if [ "$DRY" != 1 ]; then
  mkdir -p "$OUT13" "$OUT45" || exit 2
  exec 9>"$OUT13/.run.lock"
  flock -n 9 || { echo "another run_full_real_negray.sh is running on $OUT13 -- refusing" >&2; exit 1; }
  # stale JIT build locks (a killed build leaves one; the next run would wait forever): only when nothing runs
  if ! pgrep -f "tools/(run_motion_coherence_support|diagnose_phase1_negative_ray|train_phase5_rgbd_gaussians|run_phase4_recovery).py" > /dev/null; then
    find submodules/diff-gaussian-rasterization "$P45/submodules/diff-gaussian-rasterization" -path '*/.build_*/*' \
      -name lock -print -delete 2>/dev/null
  fi
  FREE=$(df -BG --output=avail "$OUT13" | tail -1 | tr -dc 0-9)
  [ "${FREE:-0}" -ge 60 ] || { echo "only ${FREE} GB free under $OUT13; need >= 60" >&2; exit 2; }
  { echo "date $(date -Is)"
    for W in "$HERE" "$P45"; do
      echo "worktree $W commit $(git -C "$W" rev-parse HEAD) branch $(git -C "$W" branch --show-current)"
      echo "  tracked diff sha256 $(git -C "$W" diff HEAD -- '*.py' '*.sh' '*.cu' '*.cpp' '*.h' '*.cuh' | sha256sum | cut -c1-64)"
    done
    echo "preview physics: density $PREVIEW_DENSITY velocity $PREVIEW_VELOCITY effort $PREVIEW_EFFORT damping $PREVIEW_DAMPING friction $PREVIEW_FRICTION"
  } > "$OUT13/provenance_$(date +%Y%m%d_%H%M%S).txt"
fi
echo "full real (new Phase 1): Phases 1-3 $HERE -> $OUT13; Phases 4-5 + assets $P45 -> $OUT45 ($RUN_ID); parts $PARTS"
echo "  live progress of every stage (Phases 1-3 tqdm, Phase-4 ETA, Phase-5 tqdm), in another terminal:"
echo "    OUT13=\"$OUT13\" OUT45=\"$OUT45\" bash $HERE/watch_full_real.sh"

shadow() {   # shadow NAME -> the pre-registration dataset of NAME (built once; symlinks only)
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

STATUS=()
while read -r sc dir ms me <&3; do              # fd 3: children (ffmpeg in Phase 5) must not read the table
  [ -z "$sc" ] && continue
  if [ -n "$SCENE_FILTER" ] && [[ " $SCENE_FILTER " != *" $sc "* ]]; then continue; fi
  name="final_${sc}_episode_000000"
  [ "$dir" = forward ] && END=initial || END=final
  if [ "$sc" = orange_suitecase ]; then
    CAMS=cam_000,cam_001,cam_002; EPROOT="$V6_3"; CAL="$V6_3/$name/base/calibration/sensor_model.json"; SRCROOT="$REAL"
  else
    CAMS=cam_001,cam_002; EPROOT="$V6"; CAL="$V5/$name/base/calibration/sensor_model.json"
    shadow "$name" || { STATUS+=("$sc SKIPPED (no pre-registration shadow)"); continue; }
    SRCROOT="$SHADOW"
  fi
  [ -f "$CAL" ] || { STATUS+=("$sc SKIPPED (no sensor model $CAL)"); continue; }
  PO="$OUT13/$name/$ARM/pipeline"
  P5="$OUT45/$sc/$ARM/phase5/${RUN_ID}_gaussian_rgbd"
  echo; echo "=== $sc  $dir  motion [$ms, $me]  $CAMS  $(date +%T)"
  if has p13; then
    if [ -f "$PO/$name/bidirectional/${dir}_joint/summary.json" ] && [ "$FORCE13" != 1 ]; then
      echo "  Phases 1-3 already finished in $PO (FORCE13=1 re-runs)"; STATUS+=("$sc phase1-3 reused")
    else
      SRC_EP="$EPROOT/$name/$ARM/pipeline/$name/static_bidi/$END"
      DST="$PO/$name/static_bidi/$END"
      [ -d "$SRC_EP" ] || { STATUS+=("$sc SKIPPED (no endpoint $SRC_EP)"); continue; }
      if [ ! -d "$DST" ] && [ "$DRY" != 1 ]; then
        mkdir -p "$(dirname "$DST")"; rm -rf "$DST.tmp"; cp -r "$SRC_EP" "$DST.tmp" && mv "$DST.tmp" "$DST"
      fi
      DRY="$DRY" ITERS=1000 bash run_bidirectional_all.sh -i "$SRCROOT" -o "$PO" --scenes "$name" \
        --stages "phase1 phase2 phase3" --cameras "$CAMS" --motion-start-frame "$ms" --motion-end-frame "$me" \
        --direction "$dir" --frame-timestamps auto --sensor-model "$CAL" --bootstrap-p0 \
        --p0-velocity-sigma 1.0 0.316227766 --phase1-partition canonical "${MIS[@]}" \
        --velocity-tau "$VTAU" --phase2-support-refine free-space --phase2-support-refine-rounds "$ROUNDS" \
        --phase1-likelihood occlusion-consistent --negative-ray-reliability calibrated \
        --phase1-group-update legacy --phase1-deterministic < /dev/null 2>&1 | grep -v "motion evidence" | tail -6
      STATUS+=("$sc phase1-3 rc=${PIPESTATUS[0]} -> $PO/$name")
    fi
    PS="$PO/$name/bidirectional/$dir/predicted_support.json"
    [ -f "$PS" ] && "$PY" -c "import json,sys; d=json.load(open(sys.argv[1])); e=d.get('extraction_summary') or {}; \
c=e.get('selected_candidate') or {}; r=c.get('rigidity',{}); w=c.get('rigidity_world',{}); \
print('  support:', d['status'], len(d['dynamic_gaussians']), 'Gaussians; seed eta_cv rel %.3f world %.3f' \
% (r.get('eta_cv', float('nan')), w.get('eta_cv', float('nan'))))" "$PS" < /dev/null
  fi
  if has p4; then
    REAL="$SRCROOT" P03="$OUT13" CAMS="$CAMS" OUT="$OUT45" STAGE=all RUN_ID="$RUN_ID" SCENES="$sc" ARMS="$ARM" \
      DRY="$DRY" EXTRA_ARGS="--recovery-frame-stride 2" bash "$P45/run_phase45_real.sh" < /dev/null 2>&1 |
      grep -a --line-buffered -E "^===|COMPLETE|INCOMPLETE|core check|ETA this sweep|4A batch|rc=|no joint type|Error|Traceback"
    STATUS+=("$sc phase4 -> $OUT45/$sc/$ARM/phase4/$RUN_ID")
  fi
  if has p5 && [ "$DRY" != 1 ]; then
    OUT="$OUT45" RUN_ID="$RUN_ID" SCENES="$sc/$ARM" bash "$P45/run_phase5_synthetic.sh" < /dev/null 2>&1 |
      grep -a --line-buffered -E "^===|phase5 stage|done in|masked PSNR|SKIPPED|asset export|rc=|Error|Traceback"
    STATUS+=("$sc phase5 $([ -f "$P5/complete.json" ] && echo complete || echo NOT complete) -> $P5")
  fi
  if has urdf && [ "$DRY" != 1 ]; then
    if [ -f "$P5/complete.json" ]; then
      SPEC="$OUT45/$sc/$ARM/sim_asset_preview_inputs/physics_spec.yaml"
      (cd "$P45" && CUDA_VISIBLE_DEVICES= "$PY" tools/make_preview_physics_spec.py --phase5-dir "$P5" --out "$SPEC" \
         --density "$PREVIEW_DENSITY" --velocity-limit "$PREVIEW_VELOCITY" --effort-limit "$PREVIEW_EFFORT" \
         --damping "$PREVIEW_DAMPING" --coulomb-friction "$PREVIEW_FRICTION" < /dev/null &&
       CUDA_VISIBLE_DEVICES= "$PY" tools/export_phase5_sim_asset.py --phase5-dir "$P5" --asset-physics-spec "$SPEC" \
         --asset-revision preview < /dev/null) 2>&1 | grep -E "asset export|failed|Error|Traceback"
      rc=${PIPESTATUS[0]}
      STATUS+=("$sc preview URDF/USD/MJCF rc=$rc -> $P5/assets/$name.rev-preview")
    else
      STATUS+=("$sc preview URDF SKIPPED (Phase 5 not complete)")
    fi
  fi
done 3<<< "$TABLE"
echo; echo "=== summary  $(date)"; printf '  %s\n' "${STATUS[@]}"
