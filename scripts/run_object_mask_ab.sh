#!/usr/bin/env bash
#
# A/B of the hard negative-ray constraint (--object-mask-constraint
# visible-negative-hard) on the synthetic scenes, then the comparison table.
#
#   without:  <nomask-root>/<scene>/  static models + Phase 1/2 outputs COPIED from the
#             stable results (<base-root>), Phase 3 run fresh, constraint off
#   with:     <masks-root>/<scene>/   the same static models copied, Phase 1, 2 and 3
#             rerun with the constraint on (same static endpoints, same settings)
#   table:    tools/compare_object_masks.py -> <masks-root>/../object_masks_ab/
#
# Copies, not links: the results drive is exFAT (no symlinks / hard links).
# Only the files the stages and the comparison read are copied (no checkpoints,
# no renders); nothing under <base-root> is written. The static models are shared:
# on the SAPIEN exports depth exists on the object only, so the static depth fix
# (1189024) does not change them.
#
#   bash scripts/run_object_mask_ab.sh --dry-run
#   bash scripts/run_object_mask_ab.sh
#   bash scripts/run_object_mask_ab.sh --scenes "fridge_10905 oven_101917"
#   bash scripts/run_object_mask_ab.sh --compare-only
#
# --mask-boundary-px N / --mask-alpha-epsilon E are forwarded (defaults 2 / 0.05).
set -u
cd "$(dirname "$0")/.."

RES="/media/samuelemara/Extreme SSD/Samuele/results_11_10_26"
BASE="$RES/ours"
DATA="/media/samuelemara/Extreme SSD/Samuele/Backup_09_10_26/Sapien/output/preparation_8cams/partnet_subset_fixed_render_linear"
NOMASK="$RES/ours_nomask"
MASKS="$RES/ours_masks"
SCENES="USB_100109 stapler_103111 blade_103706 fridge_10905 storage_45135 laptop_10211 oven_101917 scissor_11100 foldchair_102255 washer_103776"
DRY=0 ; COMPARE_ONLY=0 ; BAND=2 ; EPS=0.05
while [ $# -gt 0 ]; do
  case "$1" in
    --base-root) BASE="${2%/}"; shift 2;;
    --dataset-root) DATA="${2%/}"; shift 2;;
    --nomask-root) NOMASK="${2%/}"; shift 2;;
    --masks-root) MASKS="${2%/}"; shift 2;;
    --scenes) SCENES="$2"; shift 2;;
    --mask-boundary-px) BAND="$2"; shift 2;;
    --mask-alpha-epsilon) EPS="$2"; shift 2;;
    --dry-run) DRY=1; shift;;
    --compare-only) COMPARE_ONLY=1; shift;;
    -h|--help) sed -n '2,/^set -u/p' "$0" | sed '$d' | sed 's/^# \{0,1\}//'; exit 0;;
    *) echo "unknown argument: $1" >&2; exit 2;;
  esac
done
PY=${PY:-/home/samuelemara/miniconda3/envs/rigeuler3dgs/bin/python}
CMP="$(dirname "$MASKS")/object_masks_ab"
DRYFLAG=(); [ "$DRY" = "1" ] && DRYFLAG=(--dry-run)

copy_tree() {   # copy_tree SRC DST [find-args...]: copy matching files, never overwrite
  local src="$1" dst="$2"; shift 2
  [ -e "$src" ] || { echo "  !! missing $src"; return 1; }
  if [ "$DRY" = "1" ]; then echo "  + copy $src -> $dst"; return 0; fi
  ( cd "$src" && find . "$@" -type f -print0 ) | while IFS= read -r -d '' f; do
    [ -e "$dst/$f" ] && continue
    mkdir -p "$(dirname "$dst/$f")" && cp -p "$src/$f" "$dst/$f"
  done
}
static_copy() {  # the trained model without its training checkpoint
  copy_tree "$1" "$2" -not -name "chkpnt*.pth" -not -name "train.log"
}

echo "base (stable, read only): $BASE"
echo "dataset                 : $DATA"
echo "without constraint      : $NOMASK"
echo "with constraint         : $MASKS"
echo "scenes                  : $SCENES"

if [ "$COMPARE_ONLY" != "1" ]; then
  OK=""
  for S in $SCENES; do
    b="$BASE/$S"; good=1
    for e in initial final; do
      static_copy "$b/static_bidi/$e" "$NOMASK/$S/static_bidi/$e" || good=0
      static_copy "$b/static_bidi/$e" "$MASKS/$S/static_bidi/$e" || good=0
    done
    for d in forward backward; do
      copy_tree "$b/bidirectional/$d" "$NOMASK/$S/bidirectional/$d" \
        \( -name "predicted_support.json" -o -path "./evaluation/*" -o -path "./phase1_centers/*" -o -name "config.json" \) || good=0
      copy_tree "$b/bidirectional/${d}_rigid" "$NOMASK/$S/bidirectional/${d}_rigid" \
        \( -path "./support_coherence/*" -o -path "./joint_reduction/*" -o -name "summary.json" \) || good=0
    done
    [ "$good" = 1 ] && OK="$OK $S" || echo "  !! $S skipped (missing base outputs)"
  done
  OK="${OK# }"
  [ -n "$OK" ] || { echo "no scene ready"; exit 1; }
  echo
  echo "=== 1/2  WITHOUT the constraint: Phase 3 on the stable Phase 1/2 results"
  bash run_bidirectional_all.sh --input "$DATA" --output "$NOMASK" --scenes "$OK" \
    --stages "phase3" "${DRYFLAG[@]}"
  echo
  echo "=== 2/2  WITH the constraint: Phase 1, 2 and 3"
  bash run_bidirectional_all.sh --input "$DATA" --output "$MASKS" --scenes "$OK" \
    --stages "phase1 phase2 phase3" --object-mask-constraint visible-negative-hard \
    --mask-ray-semantics trusted-full-ray --mask-boundary-px "$BAND" --mask-alpha-epsilon "$EPS" \
    "${DRYFLAG[@]}"
fi

echo
echo "=== comparison -> $CMP"
if [ "$DRY" = "1" ]; then
  echo "  + $PY tools/compare_object_masks.py --baseline $NOMASK --masks $MASKS --out $CMP"
else
  "$PY" tools/compare_object_masks.py --baseline "$NOMASK" --masks "$MASKS" --out "$CMP"
fi
