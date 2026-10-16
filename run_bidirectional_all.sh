#!/usr/bin/env bash
#
# Full bidirectional pipeline for every scene in the two-endpoint dataset.
#
#   per scene:  static 30k initial + final
#               Phase 1 + coherence, forward and backward
#               rigid SE(3) Phase 2 + joint identification, forward and backward
#               Phase 3 per-frame joint displacement, joint fixed (opt-in: "phase3" in STAGES)
#
# Every stage is skipped when its output is already on disk, so the script is
# re-runnable and an interrupted run resumes where it stopped. A stage that
# fails is recorded and the scene continues to the next stage only when the
# next stage does not depend on it; otherwise the scene is abandoned and the
# run moves on. One scene must not take the batch down.
#
#   bash run_bidirectional_all.sh --input DATASET_ROOT --output RESULTS_ROOT
#   bash run_bidirectional_all.sh --input /path/to/one/scene_folder --output RESULTS_ROOT
#   bash run_bidirectional_all.sh -i IN -o OUT --stages "phase3" --dry-run
#
# --input / -i   EITHER a dataset root whose sub-folders are scenes (every
#                scene folder found there is run, unless --scenes narrows it)
#                OR one scene folder itself (it contains rgb/, depth_npy/,
#                colmap/, colmap_final/). Quote paths with spaces.
# --output / -o  the results ROOT: each scene is written to <output>/<scene>/.
#                Passing <root>/<scene> for a single scene is recognised and
#                not nested twice.
# --scenes "A B" restrict a dataset root to these scenes
# --stages "..." static | phase1 | phase2 | phase3   (default: static phase1 phase2)
# --frames N|auto   frames per sequence (default auto = every frame of the scene;
#                   the backward run needs the whole sequence, ending at frame T)
# --cameras N|auto  cameras (default auto = every rgb/cam_* folder)
# --dry-run      print the commands only        --force   rerun existing stages
#
# The same settings can be given as environment variables (SRC_ROOT, OUT_ROOT,
# SCENES, STAGES, FRAMES, CAMS, DRY, FORCE), e.g.
#   SCENES="washer_103776" STAGES="phase3" bash run_bidirectional_all.sh
#
# Every scene is checked before any GPU work: both COLMAP endpoints, rgb/ and
# depth_npy/ per camera, the same frame count per camera, and frame files
# numbered frame_000000, frame_000001, ... without gaps (the RGB-D loader
# reads them consecutively). A scene that fails the check is skipped with the
# reason; the rest of the batch runs.
#
set -u
cd "$(dirname "$0")"

usage() { sed -n '3,46p' "$0" | sed 's/^# \{0,1\}//'; }
INPUT="" ; OUTPUT=""
while [ $# -gt 0 ]; do
  case "$1" in
    -i|--input)   INPUT="$2"; shift 2;;
    -o|--output)  OUTPUT="$2"; shift 2;;
    --scenes)     SCENES="$2"; shift 2;;
    --stages)     STAGES="$2"; shift 2;;
    --frames)     FRAMES="$2"; shift 2;;
    --cameras)    CAMS="$2"; shift 2;;
    --dry-run)    DRY=1; shift;;
    --force)      FORCE=1; shift;;
    -h|--help)    usage; exit 0;;
    *) echo "unknown argument: $1 (see --help)" >&2; exit 2;;
  esac
done

PY=${PY:-/home/samuelemara/miniconda3/envs/rigeuler3dgs/bin/python}
INPUT=${INPUT:-${SRC_ROOT:-"/media/samuelemara/Extreme SSD/Samuele/Backup_09_10_26/Sapien/output/preparation_8cams/partnet_subset_fixed_render_linear"}}
OUTPUT=${OUTPUT:-${OUT_ROOT:-"/media/samuelemara/Extreme SSD/Samuele/results_11_10_26/ours"}}
INPUT=${INPUT%/}; OUTPUT=${OUTPUT%/}

STAGES=${STAGES:-"static phase1 phase2"}
ITERS=${ITERS:-30000}
FRAMES=${FRAMES:-auto}
CAMS=${CAMS:-auto}
BOOT=${BOOT:-12}            # Phase-1 IEKF iterations per frame
RIGID_ITERS=${RIGID_ITERS:-8}
KNN=${KNN:-12}
SNCOUNT=${SNCOUNT:-192}     # coherence GRAPH node count
IEKF_SN=${IEKF_SN:-64}      # Phase-1 IEKF grouped-state size -- NOT the graph.
                            # These were one flag once; 0 there means one group
                            # per Gaussian and Phase 1 slows ~60x.
FOLDS=${FOLDS:-5}
PHASE3_ITERS=${PHASE3_ITERS:-5}   # Phase-3 IEKF iterations per frame
PHASE3_HYP=${PHASE3_HYP:-prismatic,revolute}
DRY=${DRY:-0}
# FORCE=1 retrains/reruns even when the output exists. The final endpoint
# models trained before the points3D_parts_final.ply lookup landed carry no
# GT labels, so their debug pages have empty GT and metric panels; FORCE=1
# is how they get rebuilt. Nothing about thealgorithm depends on it.
FORCE=${FORCE:-0}
FORCE_STATIC=${FORCE_STATIC:-$FORCE}

# ---- input: a dataset root or a single scene folder -----------------------
is_scene_dir() { [ -d "$1/rgb" ] && { [ -d "$1/colmap" ] || [ -d "$1/colmap_final" ]; }; }
if [ ! -d "$INPUT" ]; then
  echo "input not found: $INPUT" >&2; exit 2
fi
if is_scene_dir "$INPUT"; then
  SRC_ROOT=$(dirname "$INPUT")
  ONE=$(basename "$INPUT")
  if [ -n "${SCENES:-}" ] && [ "$SCENES" != "$ONE" ]; then
    echo "--input is the scene folder '$ONE' but --scenes says '$SCENES'" >&2; exit 2
  fi
  SCENES="$ONE"
  # <results>/<scene> given as the output of a single scene: use its parent
  if [ "$(basename "$OUTPUT")" = "$ONE" ]; then
    echo "  note: --output ends in the scene name; writing to $OUTPUT"
    OUTPUT=$(dirname "$OUTPUT")
  fi
else
  SRC_ROOT="$INPUT"
  if [ -z "${SCENES:-}" ]; then
    SCENES=""
    for d in "$SRC_ROOT"/*/; do
      d=${d%/}; is_scene_dir "$d" && SCENES="$SCENES $(basename "$d")"
    done
    SCENES=${SCENES# }
  fi
  if [ -z "$SCENES" ]; then
    echo "no scene folders (with rgb/ and colmap*/) under $SRC_ROOT" >&2; exit 2
  fi
fi
OUT_ROOT="$OUTPUT"
for S in $SCENES; do
  case "$S" in */*) echo "scene names must be plain folder names: '$S'" >&2; exit 2;; esac
done

# Layout check of one scene: prints "OK <frames> <cameras>" or "FAIL <reasons>".
check_scene() {
  "$PY" - "$1" <<'PYEOF'
import re, sys
from pathlib import Path
d = Path(sys.argv[1]); errs = []
for k in ("colmap/sparse/0", "colmap_final/sparse/0", "rgb", "depth_npy"):
    if not (d / k).is_dir():
        errs.append(f"missing {k}/")
cams = sorted(p.name for p in (d / "rgb").glob("cam_*") if p.is_dir()) if (d / "rgb").is_dir() else []
if not cams:
    errs.append("no rgb/cam_* folders")
counts = {}
for c in cams:
    idx = sorted(int(m.group(1)) for f in (d / "rgb" / c).iterdir()
                 if (m := re.fullmatch(r"frame_(\d+)\.(png|jpg|jpeg)", f.name, re.I)))
    counts[c] = len(idx)
    if not idx:
        errs.append(f"rgb/{c} has no frame_NNNNNN images")
    elif idx != list(range(len(idx))):
        errs.append(f"rgb/{c}: {len(idx)} frames numbered {', '.join(map(str, idx[:3]))}, ... {idx[-1]}"
                    f", not 0..{len(idx) - 1}; the RGB-D loader reads frame_000000, frame_000001, ... "
                    f"consecutively, so the frames (and the COLMAP image names) must be renumbered "
                    f"to be contiguous")
        break
    if (d / "depth_npy" / c).is_dir():
        nd = len(list((d / "depth_npy" / c).glob("frame_*.npy")))
        if nd != len(idx):
            errs.append(f"depth_npy/{c} has {nd} frames, rgb/{c} has {len(idx)}")
if counts and len(set(counts.values())) > 1:
    errs.append(f"cameras have different frame counts: {counts}")
print("FAIL " + "; ".join(errs) if errs else f"OK {min(counts.values())} {len(cams)}")
PYEOF
}

echo "input root : $SRC_ROOT"
echo "scenes     : $SCENES"
echo "output root: $OUT_ROOT   (each scene -> $OUT_ROOT/<scene>/)"
echo "stages     : $STAGES"
[ "$DRY" = "1" ] || mkdir -p "$OUT_ROOT"

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

  CHK=$(check_scene "$SRC")
  if [ "${CHK%% *}" != "OK" ]; then
    echo "  !! input check failed -- skipping: ${CHK#FAIL }"
    STATUS+=("$S SKIPPED(input check failed)"); continue
  fi
  read -r _ NF_ALL NC_ALL <<< "$CHK"
  if [ "$FRAMES" = "auto" ]; then NF=$NF_ALL; else NF=$FRAMES; fi
  if [ "$CAMS" = "auto" ]; then NC=$NC_ALL; else NC=$CAMS; fi
  if [ "$NF" -gt "$NF_ALL" ]; then
    echo "  !! --frames $NF but the scene has $NF_ALL -- skipping"
    STATUS+=("$S SKIPPED(frames $NF > $NF_ALL)"); continue
  fi
  if [ "$NF" -lt "$NF_ALL" ]; then
    echo "  note: $NF of $NF_ALL frames; the backward run is anchored at the LAST frame and will refuse a shorter window"
  fi
  echo "  input OK: $NF_ALL frames x $NC_ALL cameras  (using $NF x $NC)"

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
            --frames "$NF" --cameras "$NC" --bootstrap-iters "$BOOT" \
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
          --frames "$NF" --cameras "$NC" \
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

  # ---- Phase 3: per-frame joint displacement, both directions -----------
  # Opt-in. Joint type + geometry FIXED from Phase 2 joint identification;
  # only the per-frame displacement q_k is re-estimated on the RGB-D.
  if has_stage phase3; then
    for DIR in forward backward; do
      [ "$DIR" = forward ] && END=initial || END=final
      P2="$BIDI/${DIR}_rigid"
      P3="$BIDI/${DIR}_joint"
      if [ ! -f "$P2/support_coherence/support.json" ] || [ ! -d "$P2/joint_reduction" ]; then
        echo "  [phase3/$DIR] no Phase-2 joint identification -- skipping"
        STATUS+=("$S SKIPPED(phase3/$DIR, no phase2)"); continue
      fi
      if [ "$FORCE" != "1" ] && [ -f "$P3/summary.json" ]; then
        echo "  [phase3/$DIR] present, reusing"; continue
      fi
      echo "  [phase3/$DIR]  $(date +%T)"
      rm -rf "$P3"; mkdir -p "$P3"
      run "$LOGS/${S}_phase3_${DIR}.log" \
        "$PY" tools/run_joint_phase3.py \
          --scene "$S" --direction "$DIR" \
          --model "$OUT/static_bidi/$END" --source "$SRC" \
          --phase2-dir "$P2" --frames "$NF" --cameras "$NC" \
          --iters "$PHASE3_ITERS" --hypotheses "$PHASE3_HYP" --frames-on-cpu \
          --evaluate-gt --out "$P3"
      RC=$?
      if [ "$DRY" != "1" ] && [ $RC -ne 0 ]; then
        echo "    !! phase3/$DIR failed rc=$RC"
        tail -5 "$LOGS/${S}_phase3_${DIR}.log"
        STATUS+=("$S FAILED(phase3/$DIR)")
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
