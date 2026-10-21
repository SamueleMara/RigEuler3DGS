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
# --cameras N|auto  cameras used by Phase 1/2/3 (default 8, the setting of the
#                   stable results; auto = every rgb/cam_* folder). Static
#                   training always uses every camera of the COLMAP model.
#                   Each view keeps its own linearization on the GPU, so many
#                   cameras (e.g. 64) need far more than an 8 GB card.
# --object-mask-constraint off|visible-negative-hard   (default off)
#                hard visible-object-opacity constraint on trusted negative mask
#                rays in Phase 1/2/3 (estimator/visibility_constraint.py); with
#                --mask-ray-semantics trusted-full-ray|known-nonobject-depth|conservative,
#                --mask-depth-noise-m (required for known-nonobject-depth),
#                --mask-boundary-px (2), --mask-alpha-epsilon (0.05),
#                --mask-subdir NAME (masks read from <scene>/NAME; default mask)
# --observation-mode masked-rgb-expected-quad|masked-rgb-constant|original
#                DEFAULT masked-rgb-expected-quad: Phase 1/2/3 observe the masked
#                object RGB composited on a known artificial background (rendered on
#                the same one), with the exact Uniform[0,1]^3-background expectation
#                term, and depth on the object only (estimator/masked_observation.py).
#                The masks (<scene>/mask, or --mask-subdir NAME) are checked before
#                any training or output deletion; missing masks stop the scene, never
#                a silent fallback. masked-rgb-constant needs --rgb-background-constant
#                R G B; original is the explicit unmasked control (and the only mode
#                the hard --object-mask-constraint runs with). Static training
#                receives none of these switches.
# Stage reuse is signature-checked (tools/stage_signature.py): a stage output is
# reused only when its stage_manifest.json is complete and matches the requested
# code / endpoint / observation / masks / frames / cameras / settings / upstream.
# Output of another configuration, or legacy output without a manifest, is never
# reused or deleted: use a new --output root, or FORCE=1 to overwrite.
# --dry-run      print the commands only
#
# The same settings can be given as environment variables (SRC_ROOT, OUT_ROOT,
# SCENES, STAGES, FRAMES, CAMS, DRY), e.g.
#   SCENES="washer_103776" STAGES="phase3" bash run_bidirectional_all.sh
#
# Every scene is checked before any GPU work: both COLMAP endpoints, each
# holding exactly one image per camera (the endpoint frame), rgb/ and
# depth_npy/ per camera, the same frame count per camera, and frame files
# numbered frame_000000, frame_000001, ... without gaps (the RGB-D loader
# reads them consecutively). A scene that fails the check is skipped with the
# reason; the rest of the batch runs. No ground truth is needed: without
# metadata/ or pointcloud/ GT files every stage runs and its GT metrics are
# reported as unavailable.
#
# GPU memory: static training keeps every camera image, depth and mask on the
# CPU (--data_device cpu) and moves one view at a time; the motion stages keep
# the RGB-D frames (--frames-on-cpu) and the endpoint camera images on the CPU.
#
set -u
cd "$(dirname "$0")"

usage() { sed -n '3,/^set -u/p' "$0" | sed '$d' | sed 's/^# \{0,1\}//'; }
INPUT="" ; OUTPUT="" ; OBS_BG=()
while [ $# -gt 0 ]; do
  case "$1" in
    -i|--input)   INPUT="$2"; shift 2;;
    -o|--output)  OUTPUT="$2"; shift 2;;
    --scenes)     SCENES="$2"; shift 2;;
    --stages)     STAGES="$2"; shift 2;;
    --frames)     FRAMES="$2"; shift 2;;
    --cameras)    CAMS="$2"; shift 2;;
    --dry-run)    DRY=1; shift;;
    --object-mask-constraint) VIS_MODE="$2"; shift 2;;
    --mask-ray-semantics)     VIS_SEM="$2"; shift 2;;
    --mask-depth-noise-m)     VIS_DNOISE="$2"; shift 2;;
    --mask-boundary-px)       VIS_BAND="$2"; shift 2;;
    --mask-alpha-epsilon)     VIS_EPS="$2"; shift 2;;
    --mask-subdir)            VIS_SUBDIR="$2"; shift 2;;
    --observation-mode)       OBS_MODE="$2"; shift 2;;
    --rgb-background-constant) OBS_BG=("$2" "$3" "$4"); shift 4;;
    --rgb-background-distribution) OBS_DIST="$2"; shift 2;;
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
CAMS=${CAMS:-8}
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
VIS_MODE=${VIS_MODE:-off}
VISARGS=()
if [ "$VIS_MODE" != "off" ]; then
  VISARGS=(--object-mask-constraint "$VIS_MODE" --mask-ray-semantics "${VIS_SEM:-trusted-full-ray}"
           --mask-boundary-px "${VIS_BAND:-2}" --mask-alpha-epsilon "${VIS_EPS:-0.05}")
  [ -n "${VIS_DNOISE:-}" ] && VISARGS+=(--mask-depth-noise-m "$VIS_DNOISE")
fi
OBS_MODE=${OBS_MODE:-masked-rgb-expected-quad}     # motion default since 2026-10-21
OBSARGS=()
case "$OBS_MODE" in
  original) [ ${#OBS_BG[@]} -eq 0 ] && [ -z "${OBS_DIST:-}" ] || { echo "--rgb-background-* needs a masked --observation-mode" >&2; exit 2; }
    OBSARGS=(--observation-mode original);;     # explicit: the runners' default is the masked mode
  masked-rgb-constant)
    [ ${#OBS_BG[@]} -eq 3 ] || { echo "masked-rgb-constant needs --rgb-background-constant R G B" >&2; exit 2; }
    OBSARGS=(--observation-mode "$OBS_MODE" --rgb-background-constant "${OBS_BG[@]}");;
  masked-rgb-expected-quad)
    [ ${#OBS_BG[@]} -eq 0 ] || { echo "--rgb-background-constant is for masked-rgb-constant" >&2; exit 2; }
    OBSARGS=(--observation-mode "$OBS_MODE" --rgb-background-distribution "${OBS_DIST:-uniform01}");;
  *) echo "unknown --observation-mode $OBS_MODE" >&2; exit 2;;
esac
if [ "$OBS_MODE" != "original" ] && [ "$VIS_MODE" != "off" ]; then
  echo "--object-mask-constraint $VIS_MODE needs --observation-mode original (the default masked-rgb-expected-quad and the hard constraint are different measurement models)" >&2
  exit 2
fi
OBS_DIST_RES=""; [ "$OBS_MODE" = "masked-rgb-expected-quad" ] && OBS_DIST_RES=${OBS_DIST:-uniform01}
# fewer "reserved but unallocated" failures on the shared 8 GB card
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}
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


def colmap_images(sparse):
    """Registered image count of a COLMAP model (images.bin header, else images.txt)."""
    b, t = sparse / "images.bin", sparse / "images.txt"
    if b.is_file():
        import struct
        with open(b, "rb") as fh:
            return struct.unpack("<Q", fh.read(8))[0]
    if t.is_file():
        # two lines per image; the POINTS2D line may be empty
        rows = [r for r in t.read_text().splitlines() if not r.startswith("#")]
        while rows and not rows[-1].strip():
            rows.pop()
        return (len(rows) + 1) // 2
    return None
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
# each endpoint model is ONE instant seen by every camera: a model holding every
# frame would train the "static" endpoint on a moving object (and load every
# image of the sequence)
for m in ("colmap", "colmap_final"):
    sp = d / m / "sparse" / "0"
    n = colmap_images(sp) if sp.is_dir() else None
    if n is not None and cams and n != len(cams):
        errs.append(f"{m}/sparse/0 holds {n} images for {len(cams)} cameras; an endpoint model must "
                    f"hold exactly one image per camera (its endpoint frame)")
print("FAIL " + "; ".join(errs) if errs else f"OK {min(counts.values())} {len(cams)}")
PYEOF
}

echo "input root : $SRC_ROOT"
echo "scenes     : $SCENES"
echo "output root: $OUT_ROOT   (each scene -> $OUT_ROOT/<scene>/)"
echo "stages     : $STAGES"
[ "$VIS_MODE" != "off" ] && echo "mask constraint: ${VISARGS[*]}"
if [ "$OBS_MODE" != "original" ]; then
  echo "observation: ${OBSARGS[*]} (masks <scene>/${VIS_SUBDIR:-mask}); hard QP: off"
else
  echo "observation: original (explicit control, no masks${VIS_MODE:+; hard QP: $VIS_MODE})"
fi
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
# Watchdog. Python stages touch $RIGEULER_HEARTBEAT on every progress event
# (utils/progress.py). A stage whose heartbeat is older than STALL_MIN minutes
# is hung (seen on long runs: the process waits forever on its own GPU work
# while the GPU is idle), so it is killed and retried ONCE; a second hang is a
# failure and the batch moves on. Armed only after the first beat, so stages
# that never report progress (static training) are never touched.
# STALL_MIN=0 turns it off.
STALL_MIN=${STALL_MIN:-20}
CHILD=""
trap '[ -n "$CHILD" ] && kill "$CHILD" 2>/dev/null; exit 130' INT TERM
run() {
  local log="$1"; shift
  if [ "$DRY" = "1" ]; then printf '  + %s\n' "$*"; return 0; fi
  mkdir -p "$(dirname "$log")"
  local hb="$log.heartbeat" attempt rc stalled age
  for attempt in 1 2; do
    rm -f "$hb"; stalled=0
    RIGEULER_HEARTBEAT="$hb" "$@" > "$log" 2>&1 &
    CHILD=$!
    while kill -0 "$CHILD" 2>/dev/null; do
      sleep "${STALL_POLL:-20}"
      if [ "$STALL_MIN" -gt 0 ] && [ -f "$hb" ] && [ "$stalled" = 0 ]; then
        age=$(( $(date +%s) - $(stat -c %Y "$hb" 2>/dev/null || date +%s) ))
        if [ "$age" -gt "${STALL_SEC:-$(( STALL_MIN * 60 ))}" ]; then
          echo "    !! no progress for ${age} s (limit ${STALL_SEC:-$(( STALL_MIN * 60 ))} s, attempt $attempt): killing pid $CHILD  $(date +%T)"
          kill "$CHILD" 2>/dev/null; sleep 10; kill -9 "$CHILD" 2>/dev/null; stalled=1
        fi
      fi
    done
    wait "$CHILD"; rc=$?
    CHILD=""
    rm -f "$hb"
    if [ "$stalled" = 1 ]; then
      cp "$log" "$log.hung$attempt" 2>/dev/null
      if [ "$attempt" = 1 ]; then echo "    retrying the stage once  $(date +%T)"; continue; fi
      return 124
    fi
    return $rc
  done
}

# ---- experiment signatures (tools/stage_signature.py) -----------------------
# A stage is reused only when its stage_manifest.json says "complete", its
# signature (code, endpoint PLY hash, observation mode, masks, source frames and
# cameras, settings, upstream stage) equals the requested one, and its outputs
# still validate. Legacy output without a manifest, or a different signature,
# is never reused and never deleted: the scene stops there unless FORCE=1.
SIGDIR=$(mktemp -d "${TMPDIR:-/tmp}/rigeuler_sig.XXXXXX")
trap 'rm -rf "$SIGDIR"' EXIT
stage_sig() {   # stage_sig STAGE DIR MODEL UPSTREAM_DIR OUTFILE
  "$PY" tools/stage_signature.py compute --stage "$1" --direction "$2" --scene "$S" \
    --source "$SRC" --model "$3" --frames "$NF" --cameras "$NC" --iterations "$ITERS" \
    --observation-mode "$OBS_MODE" ${OBS_DIST_RES:+--rgb-background-distribution "$OBS_DIST_RES"} \
    ${OBS_BG[0]:+--rgb-background-constant "${OBS_BG[*]}"} --object-mask-constraint "$VIS_MODE" \
    --mask-options "${VISARGS[*]}" --mask-source "$SRC/${VIS_SUBDIR:-mask}" ${4:+--upstream "$4"} \
    --config "BOOT=$BOOT" "IEKF_SN=$IEKF_SN" "KNN=$KNN" "SNCOUNT=$SNCOUNT" "FOLDS=$FOLDS" \
             "RIGID_ITERS=$RIGID_ITERS" "PHASE3_ITERS=$PHASE3_ITERS" "PHASE3_HYP=$PHASE3_HYP" > "$5"
}
# decide STAGE_DIR SIGFILE LABEL -> DECISION in reuse | run | refuse
decide() {
  local msg rc
  msg=$("$PY" tools/stage_signature.py check --dir "$1" --signature "$2"); rc=$?
  case $rc in
    0)  DECISION=reuse;  echo "  [$3] reusing: ${msg#*: }";;
    10) DECISION=run;;
    11) DECISION=run;    echo "  [$3] ${msg#*: }";;
    *)  if [ "$FORCE" = "1" ]; then DECISION=run; echo "  [$3] FORCE=1, overwriting: $msg"
        else DECISION=refuse; echo "  !! [$3] $msg"; fi;;
  esac
}
manifest() {    # manifest STAGE_DIR SIGFILE STATUS
  [ "$DRY" = "1" ] && return 0
  "$PY" tools/stage_signature.py write --dir "$1" --signature "$2" --status "$3"
}

STATUS=()
T_ALL=$(date +%s)

for S in $SCENES; do
  SRC="$SRC_ROOT/$S"
  VISSRC=(); { [ "$VIS_MODE" != "off" ] || [ "$OBS_MODE" != "original" ]; } && VISSRC=(--mask-source "$SRC/${VIS_SUBDIR:-mask}")
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
  if [ "$CAMS" = "auto" ] || [ "$CAMS" -gt "$NC_ALL" ]; then NC=$NC_ALL; else NC=$CAMS; fi
  if [ "$NF" -gt "$NF_ALL" ]; then
    echo "  !! --frames $NF but the scene has $NF_ALL -- skipping"
    STATUS+=("$S SKIPPED(frames $NF > $NF_ALL)"); continue
  fi
  if [ "$NF" -lt "$NF_ALL" ]; then
    echo "  note: $NF of $NF_ALL frames; the backward run is anchored at the LAST frame and will refuse a shorter window"
  fi
  echo "  input OK: $NF_ALL frames x $NC_ALL cameras  (using $NF x $NC)"
  if [ "$OBS_MODE" != "original" ] || [ "$VIS_MODE" != "off" ]; then
    # masks are checked BEFORE any training or output deletion; no fallback
    MCHK=$("$PY" tools/check_masks.py --source "$SRC" --mask-source "$SRC/${VIS_SUBDIR:-mask}" \
           --cameras "$NC" --frames "$NF" 2>&1); MRC=$?
    if [ $MRC -ne 0 ]; then
      echo "  !! $MCHK" | head -6
      echo "  !! observation mode $OBS_MODE needs the masks -- skipping (pass --observation-mode original explicitly to run without them)"
      STATUS+=("$S SKIPPED(masks not ready)"); continue
    fi
    echo "  $MCHK"
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
        [ "$DRY" = "1" ] || { rm -rf "$M"; mkdir -p "$M"; }   # never touch disk in a dry run
        run "$M/train.log" \
          "$PY" train.py -s "$SRC/$CM" -m "$M" --images ../rgb \
            --masks "$SRC/mask" -r 1 \
            --depths depth_npy --sapien_depth_is_metric \
            --iterations "$ITERS" --static_train_iters "$ITERS" \
            --test_iterations 999999 --disable_viewer --no_wandb \
            --data_device cpu
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

  declare -A DEC1=() DEC2=()       # per-direction decisions, for the dry-run narration
  # ---- Phase 1 + coherence, both directions ---------------------------
  if has_stage phase1; then
    for DIR in forward backward; do
      [ "$DIR" = forward ] && END=initial || END=final
      P1="$BIDI/$DIR"
      SIG1="$SIGDIR/${S}_phase1_${DIR}.json"
      stage_sig phase1 "$DIR" "$OUT/static_bidi/$END" "" "$SIG1"
      decide "$P1" "$SIG1" "phase1/$DIR"
      DEC1[$DIR]=$DECISION
      if [ "$DECISION" = refuse ]; then
        STATUS+=("$S REFUSED(phase1/$DIR, see message)")
      elif [ "$DECISION" = run ]; then
        echo "  [phase1/$DIR] from static_bidi/$END  $(date +%T)"
        [ "$DRY" = "1" ] || { rm -rf "$P1"; mkdir -p "$P1"; }   # never touch disk in a dry run
        manifest "$P1" "$SIG1" running
        run "$LOGS/${S}_phase1_${DIR}.log" \
          "$PY" tools/run_motion_coherence_support.py \
            --scene "$S" --direction "$DIR" \
            --model "$OUT/static_bidi/$END" --source "$SRC" \
            --frames "$NF" --cameras "$NC" --bootstrap-iters "$BOOT" \
            --iekf-supernode-size "$IEKF_SN" \
            --coherence-knn "$KNN" --supernode-count "$SNCOUNT" \
            --cv-folds "$FOLDS" --frames-on-cpu "${VISARGS[@]}" "${OBSARGS[@]}" "${VISSRC[@]}" --out "$P1"
        RC=$?
        if [ "$DRY" != "1" ] && [ $RC -ne 0 ]; then
          echo "    !! phase1/$DIR failed rc=$RC"
          tail -5 "$LOGS/${S}_phase1_${DIR}.log"
          manifest "$P1" "$SIG1" failed
          STATUS+=("$S FAILED(phase1/$DIR)")
        elif ! manifest "$P1" "$SIG1" complete; then
          STATUS+=("$S FAILED(phase1/$DIR outputs invalid)")
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
      SIG1="$SIGDIR/${S}_phase1_${DIR}.json"
      [ -f "$SIG1" ] || stage_sig phase1 "$DIR" "$OUT/static_bidi/$END" "" "$SIG1"
      if ! "$PY" tools/stage_signature.py check --dir "$P1" --signature "$SIG1" > /dev/null; then
        if [ "$DRY" = "1" ] && [ "${DEC1[$DIR]:-}" = run ]; then
          echo "  [phase2/$DIR] (dry run) would follow the phase1/$DIR run above"
        else
          echo "  [phase2/$DIR] no COMPLETE phase1/$DIR with the requested signature -- skipping"
          STATUS+=("$S SKIPPED(phase2/$DIR, no matching phase1)"); continue
        fi
      fi
      SIG2="$SIGDIR/${S}_phase2_${DIR}.json"
      stage_sig phase2 "$DIR" "$OUT/static_bidi/$END" "$P1" "$SIG2"
      decide "$P2" "$SIG2" "phase2/$DIR"
      DEC2[$DIR]=$DECISION
      [ "$DECISION" = reuse ] && continue
      if [ "$DECISION" = refuse ]; then STATUS+=("$S REFUSED(phase2/$DIR, see message)"); continue; fi
      echo "  [phase2/$DIR]  $(date +%T)"
      [ "$DRY" = "1" ] || { rm -rf "$P2"; mkdir -p "$P2"; }   # never touch disk in a dry run
      manifest "$P2" "$SIG2" running
      run "$LOGS/${S}_phase2_${DIR}.log" \
        "$PY" tools/run_rigid_se3_pipeline.py \
          --scene "$S" --direction "$DIR" \
          --model "$OUT/static_bidi/$END" --source "$SRC" \
          --frames "$NF" --cameras "$NC" \
          --rigid-iters "$RIGID_ITERS" --frames-on-cpu "${VISARGS[@]}" "${OBSARGS[@]}" "${VISSRC[@]}" \
          --coherence-support "$P1/predicted_support.json" \
          --phase1-centers "$P1/phase1_centers" \
          --out "$P2"
      RC=$?
      if [ "$DRY" != "1" ] && [ $RC -ne 0 ]; then
        echo "    !! phase2/$DIR failed rc=$RC"
        tail -5 "$LOGS/${S}_phase2_${DIR}.log"
        manifest "$P2" "$SIG2" failed
        STATUS+=("$S FAILED(phase2/$DIR)")
      elif ! manifest "$P2" "$SIG2" complete; then
        STATUS+=("$S FAILED(phase2/$DIR outputs invalid)")
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
      P1="$BIDI/$DIR"
      SIG2="$SIGDIR/${S}_phase2_${DIR}.json"
      [ -f "$SIG2" ] || stage_sig phase2 "$DIR" "$OUT/static_bidi/$END" "$P1" "$SIG2"
      if ! "$PY" tools/stage_signature.py check --dir "$P2" --signature "$SIG2" > /dev/null; then
        if [ "$DRY" = "1" ] && [ "${DEC2[$DIR]:-}" = run ]; then
          echo "  [phase3/$DIR] (dry run) would follow the phase2/$DIR run above"
        else
          echo "  [phase3/$DIR] no COMPLETE phase2/$DIR with the requested signature -- skipping"
          STATUS+=("$S SKIPPED(phase3/$DIR, no matching phase2)"); continue
        fi
      fi
      SIG3="$SIGDIR/${S}_phase3_${DIR}.json"
      stage_sig phase3 "$DIR" "$OUT/static_bidi/$END" "$P2" "$SIG3"
      decide "$P3" "$SIG3" "phase3/$DIR"
      [ "$DECISION" = reuse ] && continue
      if [ "$DECISION" = refuse ]; then STATUS+=("$S REFUSED(phase3/$DIR, see message)"); continue; fi
      echo "  [phase3/$DIR]  $(date +%T)"
      [ "$DRY" = "1" ] || { rm -rf "$P3"; mkdir -p "$P3"; }   # never touch disk in a dry run
      manifest "$P3" "$SIG3" running
      run "$LOGS/${S}_phase3_${DIR}.log" \
        "$PY" tools/run_joint_phase3.py \
          --scene "$S" --direction "$DIR" \
          --model "$OUT/static_bidi/$END" --source "$SRC" \
          --phase2-dir "$P2" --frames "$NF" --cameras "$NC" \
          --iters "$PHASE3_ITERS" --hypotheses "$PHASE3_HYP" --frames-on-cpu "${VISARGS[@]}" "${OBSARGS[@]}" "${VISSRC[@]}" \
          --evaluate-gt --out "$P3"
      RC=$?
      if [ "$DRY" != "1" ] && [ $RC -ne 0 ]; then
        echo "    !! phase3/$DIR failed rc=$RC"
        tail -5 "$LOGS/${S}_phase3_${DIR}.log"
        manifest "$P3" "$SIG3" failed
        STATUS+=("$S FAILED(phase3/$DIR)")
      elif ! manifest "$P3" "$SIG3" complete; then
        STATUS+=("$S FAILED(phase3/$DIR outputs invalid)")
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
