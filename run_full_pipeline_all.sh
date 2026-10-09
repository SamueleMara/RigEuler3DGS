#!/usr/bin/env bash
# The complete joint-discovery pipeline, from raw input scene to fitted joint.
#
#   [1] static 3DGS endpoints        train.py via tools/train_all_static_endpoints.py
#   [2] coherence static/dynamic     Phase 1 + trajectory metric coherence
#   [3] rigid SE(3) Phase 2          RGB-D pose per frame, no joint geometry
#       + post-hoc joint reduction   P and R, from the finite rotations
#
# Phase 1 is computed ONCE, in stage 2, and stage 3 reuses its centers. Stage 1
# is skipped when a trained endpoint model already exists, so re-running this
# after a failure resumes rather than retrains.
#
#   ./run_full_pipeline_all.sh                               # all five new scenes
#   SCENES="laptop_10211" ./run_full_pipeline_all.sh
#   STAGES="2 3" ./run_full_pipeline_all.sh                  # skip 3DGS training
#   ITERS=200 FRAMES=6 CAMS=2 ./run_full_pipeline_all.sh     # smoke
#
set -u
PY=${PY:-/home/samuelemara/miniconda3/envs/rigeuler3dgs/bin/python}
cd "$(dirname "$0")"

SCENES=${SCENES:-"scissor_11100 laptop_10211 foldchair_102255 oven_101917 washer_103776"}
STAGES=${STAGES:-"1 2 3"}
ITERS=${ITERS:-14000}         # static 3DGS iterations per endpoint
FRAMES=${FRAMES:-60}
CAMS=${CAMS:-8}
NG=${NG:-0}                   # 0 = every Gaussian
CAMERA=${CAMERA:-1}
BOOT=${BOOT:-12}              # Phase-1 IEKF iterations per frame
KNN=${KNN:-12}                # ONE global coherence default; never per scene
FOLDS=${FOLDS:-5}
RIGID_ITERS=${RIGID_ITERS:-8}
CG=${CG:-40}
POINTS=${POINTS:-6000}
RENDERS=${RENDERS:-1}
OUTROOT=${OUTROOT:-output}
LOGDIR=${LOGDIR:-$OUTROOT/full_pipeline_logs}
TAG=${TAG:-static_endpoints_14k}
mkdir -p "$LOGDIR"
RUN_START=$(date +%s)

has_stage() { case " $STAGES " in *" $1 "*) return 0;; *) return 1;; esac }

if ! $PY -c "import torch,sys; sys.exit(0 if torch.cuda.is_available() else 1)" 2>/dev/null; then
  echo "!! CUDA is not available. Every stage here drives the rasterizer."
  echo "   Check:  lsmod | grep nvidia ; ls /dev/nvidia*"
  exit 1
fi

echo "=============================================================="
echo " COMPLETE PIPELINE: static 3DGS -> coherence -> rigid SE(3) -> joint"
echo "   scenes : $SCENES"
echo "   stages : $STAGES"
echo "   static : $ITERS iterations per endpoint (tag $TAG)"
echo "   budget : $FRAMES frames, $CAMS cameras, gaussians $([ "$NG" = 0 ] && echo ALL || echo "$NG")"
echo "   phase1 : $BOOT iters    coherence knn $KNN, $FOLDS folds"
echo "   phase2 : rigid SE(3), $RIGID_ITERS iters, $CG CG"
echo "=============================================================="

STATUS=()
for S in $SCENES; do
  T_SCENE=$(date +%s)
  MODEL="$OUTROOT/${S}_${TAG}/shared/frame0_static"
  COH="$OUTROOT/motion_coherence_$S"
  RIG="$OUTROOT/rigid_se3_$S"
  echo ""
  echo "############################ $S"

  # --- [1] static 3DGS ------------------------------------------------
  if has_stage 1; then
    if ls "$MODEL"/point_cloud/iteration_*/point_cloud.ply >/dev/null 2>&1; then
      echo "  [1/3] static 3DGS: already trained, reusing $MODEL"
    else
      echo "  [1/3] static 3DGS endpoints ($ITERS iters)  -> $LOGDIR/${S}_static.log"
      $PY tools/train_all_static_endpoints.py \
          --only "$S" --iterations "$ITERS" --tag "$TAG" \
          --output-root "$OUTROOT" \
          --log-dir "$LOGDIR" > "$LOGDIR/${S}_static.log" 2>&1
      RC=$?
      if [ $RC -ne 0 ]; then
        STATUS+=("$S FAILED(static rc=$RC)")
        echo "    !! static training failed -- last 25 lines:"; tail -25 "$LOGDIR/${S}_static.log"
        continue
      fi
    fi
  fi
  if ! ls "$MODEL"/point_cloud/iteration_*/point_cloud.ply >/dev/null 2>&1; then
    STATUS+=("$S SKIPPED(no static model)")
    echo "  !! no trained model at $MODEL -- run stage 1"
    continue
  fi

  # --- [2] coherence static/dynamic -----------------------------------
  if has_stage 2; then
    echo "  [2/3] coherence static/dynamic  -> $LOGDIR/${S}_coherence.log"
    $PY tools/run_motion_coherence_support.py \
        --scene "$S" --model "$MODEL" --source "input/$S" \
        --frames "$FRAMES" --cameras "$CAMS" \
        --max-gaussians "$NG" --bootstrap-iters "$BOOT" \
        --coherence-knn "$KNN" --cv-folds "$FOLDS" --frames-on-cpu \
        --out "$COH" > "$LOGDIR/${S}_coherence.log" 2>&1
    RC=$?
    if [ $RC -ne 0 ]; then
      STATUS+=("$S FAILED(coherence rc=$RC)")
      echo "    !! coherence failed -- last 25 lines:"; tail -25 "$LOGDIR/${S}_coherence.log"
      continue
    fi
    grep -E '^  GT:|^  \[' "$LOGDIR/${S}_coherence.log" | head -4
  fi
  if [ ! -f "$COH/predicted_support.json" ]; then
    STATUS+=("$S SKIPPED(no support)")
    echo "  !! no $COH/predicted_support.json -- run stage 2"
    continue
  fi

  # --- [3] rigid SE(3) + joint reduction ------------------------------
  if has_stage 3; then
    EXTRA=()
    [ "$RENDERS" = "0" ] && EXTRA+=(--no-renders)
    # Phase 1 was already solved in stage 2; reuse it rather than paying twice.
    [ -d "$COH/phase1_centers" ] && EXTRA+=(--phase1-centers "$COH/phase1_centers")
    echo "  [3/3] rigid SE(3) + joint reduction  -> $LOGDIR/${S}_rigid.log"
    $PY tools/run_rigid_se3_pipeline.py \
        --scene "$S" --model "$MODEL" --source "input/$S" \
        --frames "$FRAMES" --cameras "$CAMS" --camera "$CAMERA" \
        --max-gaussians "$NG" --bootstrap-iters "$BOOT" \
        --rigid-iters "$RIGID_ITERS" --rigid-cg-iters "$CG" \
        --max-debug-points "$POINTS" --frames-on-cpu \
        --coherence-support "$COH/predicted_support.json" \
        "${EXTRA[@]}" --out "$RIG" > "$LOGDIR/${S}_rigid.log" 2>&1
    RC=$?
    if [ $RC -ne 0 ]; then
      STATUS+=("$S FAILED(rigid rc=$RC)")
      echo "    !! rigid stage failed -- last 25 lines:"; tail -25 "$LOGDIR/${S}_rigid.log"
      continue
    fi
    grep -E 'exact rigidity|vs Phase-1 Kabsch|gamma_1DOF|E_P |R axis |GT type' "$LOGDIR/${S}_rigid.log"
  fi
  STATUS+=("$S OK $(( $(date +%s) - T_SCENE ))s")
done

echo ""
echo "=============================================================="
echo " RESULTS   (total $(( $(date +%s) - RUN_START ))s)"
echo "=============================================================="
for s in "${STATUS[@]}"; do echo "  $s"; done
echo ""
$PY tools/summarize_pipeline.py --outroot "$OUTROOT" $SCENES
