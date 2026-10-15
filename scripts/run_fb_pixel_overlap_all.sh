#!/usr/bin/env bash
# The F/B pixel-overlap diagnostic on EVERY scene that has both historical
# Phase-1 directions, one scene after another. Read-only on all inputs.
#
#   bash scripts/run_fb_pixel_overlap_all.sh --tag overlap_coverage_v1 --resume
#
# Eligible scene: <results-root>/<scene>/static_bidi/{initial,final} PLYs and
# bidirectional/{forward,backward}/phase1_centers/trajectory_manifest.json.
# Every frame and every camera of every eligible scene, --mode all. A scene that
# fails is recorded and the batch moves on. --resume makes a rerun continue
# where it stopped (finished scenes are reused in seconds).
#
# Arguments
#   --tag T               experiment name (required)
#   --resume              continue matching experiments
#   --scenes "A B ..."    restrict to these scenes (default: all eligible)
#   --list                print the eligible scenes and exit
#   any other argument is passed to every scene (e.g. --pair-prune 1e-4)
#
# Stop cleanly between scenes: touch <results-root>/fb_pixel_overlap_<tag>.STOP
# Summary: <results-root>/fb_pixel_overlap_<tag>_summary.tsv
set -u
. "$(dirname "$0")/_fused_phase1_python.sh"
cd "$FUSED_REPO" || exit 2
RESULTS_ROOT=${RESULTS_ROOT:-"/media/samuelemara/Extreme SSD/Samuele/results_11_10_26/ours"}
TAG="" ; SCENES="" ; LIST=0 ; PASS=()
while [ $# -gt 0 ]; do
  case "$1" in
    --tag) TAG="$2"; shift 2;;
    --scenes) SCENES="$2"; shift 2;;
    --list) LIST=1; shift;;
    --results-root) RESULTS_ROOT="$2"; PASS+=("$1" "$2"); shift 2;;
    -h|--help) sed -n '2,/^set -u/p' "$0" | sed 's/^# \{0,1\}//' | sed '$d'; exit 0;;
    *) PASS+=("$1"); shift;;
  esac
done
eligible() {
  local s="$1" r="$RESULTS_ROOT/$1"
  ls "$r"/static_bidi/initial/point_cloud/iteration_*/point_cloud.ply >/dev/null 2>&1 || return 1
  ls "$r"/static_bidi/final/point_cloud/iteration_*/point_cloud.ply >/dev/null 2>&1 || return 1
  [ -f "$r/bidirectional/forward/phase1_centers/trajectory_manifest.json" ] || return 1
  [ -f "$r/bidirectional/backward/phase1_centers/trajectory_manifest.json" ] || return 1
}
if [ -z "$SCENES" ]; then
  for d in "$RESULTS_ROOT"/*/; do
    s=$(basename "$d"); eligible "$s" && SCENES="$SCENES $s"
  done
fi
if [ "$LIST" = 1 ]; then echo "eligible:$SCENES"; exit 0; fi
[ -n "$TAG" ] || { echo "--tag is required" >&2; exit 2; }
STOP="$RESULTS_ROOT/fb_pixel_overlap_${TAG}.STOP"
SUM="$RESULTS_ROOT/fb_pixel_overlap_${TAG}_summary.tsv"
echo -e "scene\texit\tminutes\toutput" > "$SUM"
echo "scenes:$SCENES"
echo "stop between scenes: touch \"$STOP\""
T0=$(date +%s)
for S in $SCENES; do
  if [ -e "$STOP" ]; then echo "STOP file present; stopping before $S"; break; fi
  echo "=============================================================="
  echo " $S   $(date '+%F %T')"
  echo "=============================================================="
  t=$(date +%s)
  "$FUSED_PY" -u tools/analyze_fb_pixel_overlap.py --scene "$S" --frames all --cameras all \
      --mode all --tag "$TAG" "${PASS[@]}"
  rc=$?
  m=$(( ($(date +%s) - t) / 60 ))
  echo -e "$S\t$rc\t$m\t$RESULTS_ROOT/$S/observation_overlap_coverage/$TAG" >> "$SUM"
  echo " $S exit $rc after $m min"
done
echo "=============================================================="
echo " all done in $(( ($(date +%s) - T0) / 60 )) min; summary: $SUM"
column -t -s $'\t' "$SUM"
