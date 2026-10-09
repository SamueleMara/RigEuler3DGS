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
#   WAIT_FOR_GPU=3600 ./run_full_pipeline_all.sh             # share the card
#
# SHARING THE GPU. Static 3DGS wants several GiB and this is an 8 GiB card.
# If something else is using it, each scene dies somewhere different --
# set_device, densify_and_split, a 2 MiB allocation inside the semantic loss
# -- and the run reports unrelated-looking failures instead of one cause. The
# gate below checks first, names the process holding the card, and either
# skips the scene or waits for it. Nothing partial is written either way, so a
# re-run starts clean.
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

# --- GPU contention gate -------------------------------------------------
#
# Static 3DGS needs several GiB. When something else holds the card, train.py
# dies in a different place each time -- set_device, densify_and_split, a 2 MiB
# allocation inside the semantic loss -- and the run reports three unrelated
# looking failures instead of one cause. This checks first and says who.
MIN_FREE_MIB=${MIN_FREE_MIB:-5000}
WAIT_FOR_GPU=${WAIT_FOR_GPU:-0}      # seconds to wait for the card; 0 = do not

# The WORST of several samples, not one reading. A neighbour that cycles --
# a render job stepping through scenes, say -- frees the card between its
# scenes, and a single sample taken in that gap reports the whole card free
# moments before it is taken again. Measured here: one read said 8045 MiB
# free while the card was at 0.
GPU_SAMPLES=${GPU_SAMPLES:-5}
GPU_SAMPLE_GAP=${GPU_SAMPLE_GAP:-2}

gpu_free_mib() {
  local lo="" v i
  for ((i = 0; i < GPU_SAMPLES; i++)); do
    v=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits \
        2>/dev/null | head -1 | tr -d ' ')
    [ -z "$v" ] && { echo ""; return; }
    { [ -z "$lo" ] || [ "$v" -lt "$lo" ]; } && lo="$v"
    [ $((i + 1)) -lt "$GPU_SAMPLES" ] && sleep "$GPU_SAMPLE_GAP"
  done
  echo "$lo"
}

gpu_free_now() {
  nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits 2>/dev/null \
    | head -1 | tr -d ' '
}

gpu_holders() {
  nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader 2>/dev/null \
    | while IFS=, read -r pid mem; do
        pid=$(echo "$pid" | tr -d ' ')
        cmd=$(ps -o cmd= -p "$pid" 2>/dev/null | cut -c1-90)
        [ -z "$cmd" ] && cmd="(not visible from this user -- another container?)"
        echo "        pid $pid  $(echo "$mem" | tr -d ' ')  $cmd"
      done
}

# Returns 0 when there is room, 1 when there is not.
gpu_ready() {
  local free deadline
  free=$(gpu_free_mib)
  [ -z "$free" ] && return 0                   # no nvidia-smi: do not block
  [ "$free" -ge "$MIN_FREE_MIB" ] && return 0
  echo "  !! GPU has ${free} MiB free, this stage wants ${MIN_FREE_MIB} MiB."
  echo "     Held by:"
  gpu_holders
  if [ "$WAIT_FOR_GPU" -le 0 ]; then
    echo "     Set WAIT_FOR_GPU=<seconds> to wait for it, or MIN_FREE_MIB to"
    echo "     lower the bar if you know the scene is small."
    return 1
  fi
  deadline=$(( $(date +%s) + WAIT_FOR_GPU ))
  echo "     waiting up to ${WAIT_FOR_GPU}s for the card..."
  while [ "$(date +%s)" -lt "$deadline" ]; do
    sleep 30
    # cheap check first, then confirm it is not just a gap between a
    # neighbour's scenes before committing a 20 minute training run
    free=$(gpu_free_now)
    if [ -n "$free" ] && [ "$free" -ge "$MIN_FREE_MIB" ]; then
      free=$(gpu_free_mib)
      if [ -n "$free" ] && [ "$free" -ge "$MIN_FREE_MIB" ]; then
        echo "     ${free} MiB free and steady, continuing"
        return 0
      fi
      echo "     ${free} MiB at the low sample -- a gap, not a free card"
    fi
  done
  echo "     still only ${free} MiB free after ${WAIT_FOR_GPU}s"
  return 1
}

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
echo "   gpu    : $(gpu_free_now) MiB free now, need $MIN_FREE_MIB$([ "$WAIT_FOR_GPU" -gt 0 ] && echo ", will wait up to ${WAIT_FOR_GPU}s" || echo "")"
echo "=============================================================="

STATUS=()
for S in $SCENES; do
  T_SCENE=$(date +%s)
  MODEL="$OUTROOT/${S}_${TAG}/shared/frame0_static"
  COH="$OUTROOT/motion_coherence_$S"
  RIG="$OUTROOT/rigid_se3_$S"
  echo ""
  echo "############################ $S"
  if ! gpu_ready; then
    STATUS+=("$S SKIPPED(gpu busy)")
    continue
  fi

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
        DETAIL=""
        for L in "$LOGDIR/${S}_static.log" "$LOGDIR/${S}_frame0_${ITERS}.log"; do
          if [ -f "$L" ] && grep -qiE "out of memory|OutOfMemoryError|cudaErrorMemoryAllocation" "$L"; then
            DETAIL=" -- CUDA OUT OF MEMORY"
            echo "    !! static training ran out of GPU memory. This is"
            echo "       contention, not a defect in the scene. Who held the card:"
            grep -oE "Process [0-9]+ has [0-9.]+ [GM]iB memory in use" "$L" | tail -1 | sed 's/^/         /'
            gpu_holders
            echo "       Re-run when the card is free; nothing was written, so"
            echo "       this scene starts clean. WAIT_FOR_GPU=<seconds> makes"
            echo "       the pipeline wait instead of failing."
            break
          fi
        done
        STATUS+=("$S FAILED(static rc=$RC)$DETAIL")
        if [ -z "$DETAIL" ]; then
          echo "    !! static training failed -- last 25 lines:"
          tail -25 "$LOGDIR/${S}_static.log"
        fi
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
