#!/usr/bin/env bash
#
# Full bidirectional pipeline for every scene in the two-endpoint dataset.
#
#   per scene:  static 30k initial + final
#               Phase 1 + coherence, forward and backward
#               rigid SE(3) Phase 2, forward and backward
#
# Every stage is skipped when its output is already on disk, so the script is
# re-runnable and an interrupted run resumes where it stopped. A stage that
# fails is recorded and the scene continues to the next stage only when the
# next stage does not depend on it; otherwise the scene is abandoned and the
# run moves on. One scene must not take the batch down.
#
#   bash run_bidirectional_all.sh                  # everything
#   SCENES="washer_103776" bash run_bidirectional_all.sh
#   STAGES="static" bash run_bidirectional_all.sh  # static | phase1 | phase2
#   DRY=1 bash run_bidirectional_all.sh            # print the commands only
#
set -u
cd "$(dirname "$0")"

PY=${PY:-/home/samuelemara/miniconda3/envs/rigeuler3dgs/bin/python}
SRC_ROOT=${SRC_ROOT:-"/media/samuelemara/Extreme SSD/Samuele/Backup_09_10_26/Sapien/output/preparation_8cams/partnet_subset_fixed_render_linear"}
OUT_ROOT=${OUT_ROOT:-"/media/samuelemara/Extreme SSD/Samuele/results_11_10_26/ours"}

SCENES=${SCENES:-"USB_100109 stapler_103111 blade_103706 fridge_10905 storage_45135 laptop_10211 oven_101917 scissor_11100 foldchair_102255 washer_103776"}
STAGES=${STAGES:-"static phase1 phase2"}
ITERS=${ITERS:-30000}
FRAMES=${FRAMES:-60}
CAMS=${CAMS:-8}
BOOT=${BOOT:-12}            # Phase-1 IEKF iterations per frame
RIGID_ITERS=${RIGID_ITERS:-8}
KNN=${KNN:-12}
SNCOUNT=${SNCOUNT:-192}     # coherence GRAPH node count
IEKF_SN=${IEKF_SN:-64}      # Phase-1 IEKF grouped-state size -- NOT the graph.
                            # These were one flag once; 0 there means one group
                            # per Gaussian and Phase 1 slows ~60x.
FOLDS=${FOLDS:-5}
DRY=${DRY:-0}
# FORCE=1 retrains/reruns even when the output exists. The final endpoint
# models trained before the points3D_parts_final.ply lookup landed carry no
# GT labels, so their debug pages have empty GT and metric panels; FORCE=1
# is how they get rebuilt. Nothing about thealgorithm depends on it.
FORCE=${FORCE:-0}
FORCE_STATIC=${FORCE_STATIC:-$FORCE}

has_stage() { case " $STAGES " in *" $1 "*) return 0;; *) return 1;; esac; }

# A trained static model counts as present only when its checkpoint is a
# plausible PLY. "The file exists" is not enough: an interrupted write leaves
# a short or empty file that passes -f and then fails at load time, halfway
# through a batch, with an error that points at the loader rather than at the
# truncated file.
static_ready() {
  local ply="$1/point_cloud/iteration_${ITERS}/point_cloud.ply"
  [ -f "$ply" ] || return 1
  [ "$(stat -c %s "$ply" 2>/dev/null || echo 0)" -gt 100000 ] || return 1
  head -c 400 "$ply" 2>/dev/null | grep -qi "^ply" || return 1
  return 0
}
run() {
  local log="$1"; shift
  if [ "$DRY" = "1" ]; then printf '  + %s\n' "$*"; return 0; fi
  mkdir -p "$(dirname "$log")"
  "$@" > "$log" 2>&1
}

STATUS=()
T_ALL=$(date +%s)

for S in $SCENES; do
  SRC="$SRC_ROOT/$S"
  OUT="$OUT_ROOT/$S"
  BIDI="$OUT/bidirectional"
  LOGS="$OUT/logs"
  echo "=============================================================="
  echo " $S   $(date '+%F %T')"
  echo "=============================================================="

  if [ ! -d "$SRC/colmap/sparse/0" ] || [ ! -d "$SRC/colmap_final/sparse/0" ]; then
    echo "  !! missing colmap or colmap_final -- skipping"
    STATUS+=("$S SKIPPED(no endpoints)"); continue
  fi

  # ---- static 30k, both endpoints ------------------------------------
  if has_stage static; then
    for END in initial final; do
      [ "$END" = initial ] && CM="colmap" || CM="colmap_final"
      M="$OUT/static_bidi/$END"
      if [ "$FORCE_STATIC" != "1" ] && static_ready "$M"; then
        echo "  [static/$END] present, reusing  ($(du -h "$M/point_cloud/iteration_${ITERS}/point_cloud.ply" | cut -f1))"
      else
        echo "  [static/$END] training ${ITERS} iters  $(date +%T)"
        rm -rf "$M"; mkdir -p "$M"
        run "$M/train.log" \
          "$PY" train.py -s "$SRC/$CM" -m "$M" --images ../rgb \
            --masks "$SRC/mask" -r 1 \
            --depths depth_npy --sapien_depth_is_metric \
            --iterations "$ITERS" --static_train_iters "$ITERS" \
            --test_iterations 999999 --disable_viewer --no_wandb
        RC=$?
        if [ "$DRY" != "1" ] && [ $RC -ne 0 ]; then
          echo "    !! static/$END failed rc=$RC"; tail -5 "$M/train.log"
          STATUS+=("$S FAILED(static/$END)"); continue 2
        fi
      fi
    done
  fi

  # Everything below needs both models.
  for END in initial final; do
    if ! static_ready "$OUT/static_bidi/$END"; then
      echo "  !! static/$END missing -- cannot run the dynamic stages"
      STATUS+=("$S SKIPPED(static/$END absent)"); continue 3
    fi
  done

  # ---- Phase 1 + coherence, both directions ---------------------------
  if has_stage phase1; then
    for DIR in forward backward; do
      [ "$DIR" = forward ] && END=initial || END=final
      P1="$BIDI/$DIR"
      if [ "$FORCE" != "1" ] && [ -f "$P1/predicted_support.json" ]; then
        echo "  [phase1/$DIR] present, reusing"
      else
        echo "  [phase1/$DIR] from static_bidi/$END  $(date +%T)"
        rm -rf "$P1"; mkdir -p "$P1"
        run "$LOGS/${S}_phase1_${DIR}.log" \
          "$PY" tools/run_motion_coherence_support.py \
            --scene "$S" --direction "$DIR" \
            --model "$OUT/static_bidi/$END" --source "$SRC" \
            --frames "$FRAMES" --cameras "$CAMS" --bootstrap-iters "$BOOT" \
            --iekf-supernode-size "$IEKF_SN" \
            --coherence-knn "$KNN" --supernode-count "$SNCOUNT" \
            --cv-folds "$FOLDS" --frames-on-cpu --out "$P1"
        RC=$?
        if [ "$DRY" != "1" ] && [ $RC -ne 0 ]; then
          echo "    !! phase1/$DIR failed rc=$RC"
          tail -5 "$LOGS/${S}_phase1_${DIR}.log"
          STATUS+=("$S FAILED(phase1/$DIR)")
        fi
      fi
    done
  fi

  # ---- rigid SE(3) Phase 2, both directions ---------------------------
  if has_stage phase2; then
    for DIR in forward backward; do
      [ "$DIR" = forward ] && END=initial || END=final
      P1="$BIDI/$DIR"
      P2="$BIDI/${DIR}_rigid"
      if [ ! -f "$P1/predicted_support.json" ]; then
        echo "  [phase2/$DIR] no support from phase1 -- skipping"
        STATUS+=("$S SKIPPED(phase2/$DIR, no support)"); continue
      fi
      if [ "$FORCE" != "1" ] && [ -f "$P2/phase2_rigid/poses.npy" ]; then
        echo "  [phase2/$DIR] present, reusing"; continue
      fi
      echo "  [phase2/$DIR]  $(date +%T)"
      rm -rf "$P2"; mkdir -p "$P2"
      run "$LOGS/${S}_phase2_${DIR}.log" \
        "$PY" tools/run_rigid_se3_pipeline.py \
          --scene "$S" --direction "$DIR" \
          --model "$OUT/static_bidi/$END" --source "$SRC" \
          --frames "$FRAMES" --cameras "$CAMS" \
          --rigid-iters "$RIGID_ITERS" --frames-on-cpu \
          --coherence-support "$P1/predicted_support.json" \
          --phase1-centers "$P1/phase1_centers" \
          --out "$P2"
      RC=$?
      if [ "$DRY" != "1" ] && [ $RC -ne 0 ]; then
        echo "    !! phase2/$DIR failed rc=$RC"
        tail -5 "$LOGS/${S}_phase2_${DIR}.log"
        STATUS+=("$S FAILED(phase2/$DIR)")
      fi
    done
  fi

  STATUS+=("$S done")
done

echo
echo "=============================================================="
echo " SUMMARY   $(( ($(date +%s) - T_ALL) / 60 )) min"
echo "=============================================================="
for line in "${STATUS[@]}"; do echo "  $line"; done
