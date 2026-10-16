#!/usr/bin/env bash
# Endpoint correspondence on EVERY eligible scene, one after another on one GPU,
# then (optionally) the separate, evaluation-only GT scoring of each SEALED result.
#
#   bash scripts/run_fb_endpoint_correspondence_all.sh \
#     --results-root '<results>' --dataset-root '<dataset>' --tag endpoint_eval_v3 \
#     --estimated-joints auto --evaluate-gt --list-scenes | --preflight-only | --execute
#
# Eligible scene = a directory <results-root>/<scene>/ with BOTH trained endpoint
# PLYs (static_bidi/{initial,final}/point_cloud/iteration_30000/point_cloud.ply),
# BOTH Phase-1 manifests (bidirectional/{forward,backward}/phase1_centers/
# trajectory_manifest.json) and a dataset directory <dataset-root>/<scene>/.
# Nothing is hard-coded; every skipped directory is listed with its reason.
#
# Per scene (sequential, never parallel):
#   1. scripts/run_fb_endpoint_correspondence.sh --preflight         (no GPU work)
#   2. ... --execute [--resume] [--gpu-memory-fraction F]           (the GT-free diagnosis)
#   3. only if run_state.json says complete: true, a SEPARATE process
#      tools/evaluate_fb_endpoint_correspondence_gt.py               (EVALUATION ONLY)
# A failed preflight, diagnosis or evaluation is recorded and the batch moves on.
# A failed or missing GT evaluation never marks a successful diagnosis failed.
# Nothing is deleted; an existing tagged experiment is never overwritten (the
# per-scene tool refuses; rerun with --resume, or use a new --tag).
#
# Arguments
#   --results-root DIR / --dataset-root DIR   roots (paths with spaces are fine; quote them).
#                                --dataset-root may also be ONE scene folder (it has rgb/
#                                and colmap*/): that scene alone is run. --results-root
#                                may then also be <root>/<scene>.
#   --tag T                      experiment tag (required except with --list-scenes)
#   --estimated-joints auto|off|require       forwarded (auto: J0 continues when joints are missing)
#   --gpu-memory-fraction F      forwarded to the diagnosis (cap on a shared GPU)
#   --scenes "A B"               restrict to these eligible scenes
#   --list-scenes                print eligible and skipped scenes; nothing else
#   --preflight-only             per-scene preflight only (no rendering)
#   --execute                    preflight + diagnosis (+ GT evaluation)
#   --resume                     continue/reuse matching experiments and evaluations
#   --evaluate-gt                run the GT evaluator after each sealed diagnosis (DEFAULT)
#   --no-evaluate-gt             skip it
#
# Output: <results-root>/<scene>/endpoint_correspondence/<tag>/ per scene (GT
# scores under its evaluation_gt/), and the post-hoc batch tables in
# <results-root>/endpoint_correspondence_batch/<tag>/.
# Stop cleanly between scenes: touch <results-root>/endpoint_correspondence_batch/<tag>/STOP
set -u
RESULTS_ROOT="/media/samuelemara/Extreme SSD/Samuele/results_11_10_26/ours"
DATASET_ROOT="/media/samuelemara/Extreme SSD/Samuele/Backup_09_10_26/Sapien/output/preparation_8cams/partnet_subset_fixed_render_linear"
TAG="" ; JOINTS="auto" ; FRAC="" ; SCENES="" ; MODE="" ; RESUME=0 ; EVAL=1
while [ $# -gt 0 ]; do
  case "$1" in
    --results-root) RESULTS_ROOT="$2"; shift 2;;
    --dataset-root) DATASET_ROOT="$2"; shift 2;;
    --tag) TAG="$2"; shift 2;;
    --estimated-joints) JOINTS="$2"; shift 2;;
    --gpu-memory-fraction) FRAC="$2"; shift 2;;
    --scenes) SCENES="$2"; shift 2;;
    --list-scenes) MODE="list"; shift;;
    --preflight-only) MODE="preflight"; shift;;
    --execute) MODE="execute"; shift;;
    --resume) RESUME=1; shift;;
    --evaluate-gt) EVAL=1; shift;;
    --no-evaluate-gt) EVAL=0; shift;;
    -h|--help) sed -n '2,/^set -u/p' "$0" | sed 's/^# \{0,1\}//' | sed '$d'; exit 0;;
    *) echo "unknown argument: $1" >&2; exit 2;;
  esac
done
[ -n "$MODE" ] || { echo "choose one of --list-scenes, --preflight-only, --execute" >&2; exit 2; }
RESULTS_ROOT="${RESULTS_ROOT%/}"; DATASET_ROOT="${DATASET_ROOT%/}"
# a single scene folder given as the dataset: run that scene only
if [ -d "$DATASET_ROOT/rgb" ] && { [ -d "$DATASET_ROOT/colmap" ] || [ -d "$DATASET_ROOT/colmap_final" ]; }; then
  ONE="$(basename "$DATASET_ROOT")"
  if [ -n "$SCENES" ] && [ "$SCENES" != "$ONE" ]; then
    echo "--dataset-root is the scene folder '$ONE' but --scenes says '$SCENES'" >&2; exit 2
  fi
  SCENES="$ONE"; DATASET_ROOT="$(dirname "$DATASET_ROOT")"
  [ "$(basename "$RESULTS_ROOT")" = "$ONE" ] && RESULTS_ROOT="$(dirname "$RESULTS_ROOT")"
fi
[ -d "$RESULTS_ROOT" ] || { echo "no results root: $RESULTS_ROOT" >&2; exit 2; }
HERE="$(cd "$(dirname "$0")" && pwd)"

eligible_reason() {   # prints "" when eligible, else the reason
  local r="$RESULTS_ROOT/$1"
  [ -f "$r/static_bidi/initial/point_cloud/iteration_30000/point_cloud.ply" ] || { echo "no static_bidi/initial PLY"; return; }
  [ -f "$r/static_bidi/final/point_cloud/iteration_30000/point_cloud.ply" ] || { echo "no static_bidi/final PLY"; return; }
  [ -f "$r/bidirectional/forward/phase1_centers/trajectory_manifest.json" ] || { echo "no forward Phase-1 manifest"; return; }
  [ -f "$r/bidirectional/backward/phase1_centers/trajectory_manifest.json" ] || { echo "no backward Phase-1 manifest"; return; }
  [ -d "$DATASET_ROOT/$1" ] || { echo "no dataset directory $DATASET_ROOT/$1"; return; }
  echo ""
}
ELIG=() ; SKIP=()
for d in "$RESULTS_ROOT"/*/; do
  [ -d "$d" ] || continue
  s="$(basename "$d")"
  why="$(eligible_reason "$s")"
  if [ -z "$why" ]; then ELIG+=("$s"); else SKIP+=("$s: $why"); fi
done
if [ -n "$SCENES" ]; then
  SEL=()
  for s in $SCENES; do
    found=0; for e in "${ELIG[@]}"; do [ "$e" = "$s" ] && found=1; done
    if [ $found = 1 ]; then SEL+=("$s"); else echo "requested scene $s is not eligible" >&2; fi
  done
  ELIG=("${SEL[@]}")
fi
echo "results root: $RESULTS_ROOT"
echo "dataset root: $DATASET_ROOT"
echo "eligible scenes (${#ELIG[@]}):"; for s in "${ELIG[@]}"; do echo "  $s"; done
echo "skipped (${#SKIP[@]}):"; for s in "${SKIP[@]}"; do echo "  $s"; done
[ "$MODE" = "list" ] && exit 0
[ -n "$TAG" ] || { echo "--tag is required" >&2; exit 2; }

. "$HERE/_fused_phase1_python.sh"
cd "$FUSED_REPO" || exit 2
BATCH="$RESULTS_ROOT/endpoint_correspondence_batch/$TAG"
mkdir -p "$BATCH"
STATUS="$BATCH/scenes_status.tsv"
printf "scene\tpreflight\tdiagnosis\tsealed\tgt_evaluation\tminutes\toutput\n" > "$STATUS"
COMMON=(--frames endpoints --cameras all --estimated-joints "$JOINTS" --tag "$TAG"
        --results-root "$RESULTS_ROOT" --dataset-root "$DATASET_ROOT")
EXE=(--execute)
[ "$RESUME" = 1 ] && EXE+=(--resume)
[ -n "$FRAC" ] && EXE+=(--gpu-memory-fraction "$FRAC")
EVR=()
[ "$RESUME" = 1 ] && EVR+=(--resume)
T0=$(date +%s)
for S in "${ELIG[@]}"; do
  if [ -e "$BATCH/STOP" ]; then echo "STOP file present; stopping before $S"; break; fi
  OUT="$RESULTS_ROOT/$S/endpoint_correspondence/$TAG"
  echo "=============================================================="
  echo " $S   $(date '+%F %T')   -> $OUT"
  echo "=============================================================="
  t=$(date +%s)
  bash "$HERE/run_fb_endpoint_correspondence.sh" --scene "$S" "${COMMON[@]}" --preflight > "$BATCH/$S.preflight.log" 2>&1
  pf=$?
  pfs=$([ $pf = 0 ] && echo OK || echo "BLOCKED($pf)")
  echo "  preflight: $pfs   (log $BATCH/$S.preflight.log)"
  diag="-" ; sealed="no" ; ev="-"
  if [ "$MODE" = "execute" ] && [ $pf = 0 ]; then
    bash "$HERE/run_fb_endpoint_correspondence.sh" --scene "$S" "${COMMON[@]}" "${EXE[@]}" 2>&1 | tee "$BATCH/$S.diagnosis.log" \
      | grep --line-buffered -E "^===|cam_0|endpoint pass|ONLY_AT|done in|REFUSED|refused|already holds|holds experiment|Error|error|BLOCKED"
    rc=${PIPESTATUS[0]}
    diag=$([ $rc = 0 ] && echo OK || echo "FAILED($rc)")
    if [ -f "$OUT/run_state.json" ] && "$FUSED_PY" -c "import json,sys; sys.exit(0 if json.load(open(sys.argv[1])).get('complete') else 1)" "$OUT/run_state.json"; then
      sealed="yes"
    fi
    if [ "$EVAL" = 1 ] && [ $rc = 0 ] && [ "$sealed" = yes ]; then
      "$FUSED_PY" -u tools/evaluate_fb_endpoint_correspondence_gt.py --diagnosis-dir "$OUT" \
          --dataset-dir "$DATASET_ROOT/$S" "${EVR[@]}" > "$BATCH/$S.gt_evaluation.log" 2>&1
      erc=$?
      case $erc in 0) ev="EVAL_OK";; 3) ev="NO_GT_OR_INCOMPLETE";; 2) ev="EVAL_REFUSED";; *) ev="EVAL_FAILED($erc)";; esac
      tail -1 "$BATCH/$S.gt_evaluation.log"
    elif [ "$EVAL" = 1 ]; then
      ev="NOT_RUN(diagnosis failed or not sealed)"
    fi
  fi
  m=$(( ($(date +%s) - t) / 60 ))
  printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\n" "$S" "$pfs" "$diag" "$sealed" "$ev" "$m" "$OUT" >> "$STATUS"
  echo "  $S: preflight $pfs, diagnosis $diag, sealed $sealed, GT evaluation $ev ($m min)"
done
"$FUSED_PY" -u tools/aggregate_fb_endpoint_gt_batch.py --results-root "$RESULTS_ROOT" --tag "$TAG" \
    --status "$STATUS" || echo "batch aggregation failed (per-scene results are unaffected)"
echo "=============================================================="
echo " batch done in $(( ($(date +%s) - T0) / 60 )) min"
column -t -s $'\t' "$STATUS"
echo " batch tables: $BATCH"
