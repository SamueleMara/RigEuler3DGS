#!/usr/bin/env bash
#
# A/B of --object-masks on the synthetic scenes, then the comparison table.
#
#   without masks:  <nomask-root>/<scene>/  static models and Phase 1/2 LINKED from the
#                   existing stable results (<base-root>), Phase 3 run fresh
#   with masks:     <masks-root>/<scene>/   static models linked, Phase 1, 2 and 3 run
#                   with --object-masks
#   comparison:     tools/compare_object_masks.py -> <masks-root>/../object_masks_ab/
#
# The static models are shared: on the SAPIEN exports depth exists on the object
# only, so the static depth fix (1189024) does not change them, and the masks
# already drove their carving. Nothing under <base-root> is written: every link
# is made per directory (static_bidi/initial, .../final, bidirectional/<dir>), so a
# stage that removes its output directory removes the link, never the original.
#
#   bash scripts/run_object_mask_ab.sh --dry-run            # show what would run
#   bash scripts/run_object_mask_ab.sh                      # everything
#   bash scripts/run_object_mask_ab.sh --scenes "fridge_10905 oven_101917"
#   bash scripts/run_object_mask_ab.sh --compare-only       # just rebuild the table
#
# --base-root / --dataset-root / --nomask-root / --masks-root override the paths.
set -u
cd "$(dirname "$0")/.."

RES="/media/samuelemara/Extreme SSD/Samuele/results_11_10_26"
BASE="$RES/ours"
DATA="/media/samuelemara/Extreme SSD/Samuele/Backup_09_10_26/Sapien/output/preparation_8cams/partnet_subset_fixed_render_linear"
NOMASK="$RES/ours_nomask"
MASKS="$RES/ours_masks"
SCENES="USB_100109 stapler_103111 blade_103706 fridge_10905 storage_45135 laptop_10211 oven_101917 scissor_11100 foldchair_102255 washer_103776"
DRY=0 ; COMPARE_ONLY=0
while [ $# -gt 0 ]; do
  case "$1" in
    --base-root) BASE="${2%/}"; shift 2;;
    --dataset-root) DATA="${2%/}"; shift 2;;
    --nomask-root) NOMASK="${2%/}"; shift 2;;
    --masks-root) MASKS="${2%/}"; shift 2;;
    --scenes) SCENES="$2"; shift 2;;
    --dry-run) DRY=1; shift;;
    --compare-only) COMPARE_ONLY=1; shift;;
    -h|--help) sed -n '2,/^set -u/p' "$0" | sed '$d' | sed 's/^# \{0,1\}//'; exit 0;;
    *) echo "unknown argument: $1" >&2; exit 2;;
  esac
done
PY=${PY:-/home/samuelemara/miniconda3/envs/rigeuler3dgs/bin/python}
CMP="$(dirname "$MASKS")/object_masks_ab"
DRYFLAG=(); [ "$DRY" = "1" ] && DRYFLAG=(--dry-run)

link() {   # link SRC DST: a per-directory symlink, never replacing a real directory
  local src="$1" dst="$2"
  [ -e "$src" ] || { echo "  !! missing $src"; return 1; }
  if [ -L "$dst" ]; then return 0; fi
  if [ -e "$dst" ]; then echo "  !! $dst exists and is not a link -- left as it is"; return 0; fi
  if [ "$DRY" = "1" ]; then echo "  + ln -s $src $dst"; return 0; fi
  mkdir -p "$(dirname "$dst")" && ln -s "$src" "$dst"
}

echo "base (stable, read only): $BASE"
echo "dataset                 : $DATA"
echo "without masks           : $NOMASK"
echo "with masks              : $MASKS"
echo "scenes                  : $SCENES"

if [ "$COMPARE_ONLY" != "1" ]; then
  OK=""
  for S in $SCENES; do
    b="$BASE/$S"
    good=1
    for e in initial final; do
      link "$b/static_bidi/$e" "$NOMASK/$S/static_bidi/$e" || good=0
      link "$b/static_bidi/$e" "$MASKS/$S/static_bidi/$e" || good=0
    done
    for d in forward backward forward_rigid backward_rigid; do
      link "$b/bidirectional/$d" "$NOMASK/$S/bidirectional/$d" || good=0
    done
    [ "$good" = 1 ] && OK="$OK $S" || echo "  !! $S skipped (missing base outputs)"
  done
  OK="${OK# }"
  [ -n "$OK" ] || { echo "no scene ready"; exit 1; }
  echo
  echo "=== 1/2  WITHOUT masks: Phase 3 on the stable Phase 1/2 results"
  bash run_bidirectional_all.sh --input "$DATA" --output "$NOMASK" --scenes "$OK" \
    --stages "phase3" "${DRYFLAG[@]}"
  echo
  echo "=== 2/2  WITH masks: Phase 1, 2 and 3 (--object-masks)"
  bash run_bidirectional_all.sh --input "$DATA" --output "$MASKS" --scenes "$OK" \
    --stages "phase1 phase2 phase3" --object-masks "${DRYFLAG[@]}"
fi

echo
echo "=== comparison -> $CMP"
if [ "$DRY" = "1" ]; then
  echo "  + $PY tools/compare_object_masks.py --baseline $NOMASK --masks $MASKS --out $CMP"
else
  "$PY" tools/compare_object_masks.py --baseline "$NOMASK" --masks "$MASKS" --out "$CMP"
fi
