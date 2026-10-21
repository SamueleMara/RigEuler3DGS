#!/usr/bin/env bash
#
# FAIR A/B of the motion observation: explicit original vs masked, BOTH arms
# fresh Phase 1, Phase 2 and Phase 3 from the SAME static endpoints
# (CODEX_FIX_MASKED_IEKF_DEFAULT_AND_VALIDATION_2026-10-21.md, P1-4).
#
#   static    <static-root>/<scene>/static_bidi/{initial,final}  READ ONLY; the
#             trained models are copied (exFAT: no links) into both arm roots and
#             the copies are verified byte-identical (point_cloud.ply, cfg_args,
#             cameras.json sha256) before anything runs
#   original  <original-root>: run_bidirectional_all.sh --observation-mode original
#   masked    <masked-root>:   run_bidirectional_all.sh --observation-mode <mode>
#             (default masked-rgb-expected-quad; masked-rgb-constant needs
#             --background R G B)
#   compare   tools/compare_object_masks.py -> <compare-root>: checks that every
#             stage manifest is complete and that the two arms' signatures
#             differ ONLY in the observation block / output roots, then scores
#             GT metrics (evaluation only) and the GT-free mask diagnostics
#
# Nothing from one arm seeds the other; no earlier Phase 1/2/3 output is copied.
# Same scenes, frames, cameras, directions and driver settings in both arms; the
# scenes are run arm after arm, scene by scene, so a partial run stays paired.
# Stage reuse is decided by the driver's experiment signatures: a stage is reused
# only if it is complete with an identical signature, and a foreign/legacy
# output is refused (FORCE=1 overrides) -- so always use NEW roots.
#
#   bash scripts/run_object_mask_ab.sh --dry-run
#   bash scripts/run_object_mask_ab.sh --scenes "fridge_10905"
#   bash scripts/run_object_mask_ab.sh --frames 2 --scenes fridge_10905 --original-root X --masked-root Y
#   bash scripts/run_object_mask_ab.sh --compare-only
set -u
cd "$(dirname "$0")/.."

RES="/media/samuelemara/Extreme SSD/Samuele/results_11_10_26"
STATIC="$RES/ours"
DATA="/media/samuelemara/Extreme SSD/Samuele/Backup_09_10_26/Sapien/output/preparation_8cams/partnet_subset_fixed_render_linear"
ORIG="$RES/ours_fairab_original"
MASKED="$RES/ours_fairab_masked"
CMP=""
MODE="masked-rgb-expected-quad"; BGC=()
SCENES="USB_100109 stapler_103111 blade_103706 fridge_10905 storage_45135 laptop_10211 oven_101917 scissor_11100 foldchair_102255 washer_103776"
FRAMES=""; CAMS=""; ITER=30000
DRY=0 ; COMPARE_ONLY=0
while [ $# -gt 0 ]; do
  case "$1" in
    --static-root|--base-root) STATIC="${2%/}"; shift 2;;
    --dataset-root) DATA="${2%/}"; shift 2;;
    --original-root|--nomask-root) ORIG="${2%/}"; shift 2;;
    --masked-root|--masks-root) MASKED="${2%/}"; shift 2;;
    --compare-root) CMP="${2%/}"; shift 2;;
    --mode|--with) MODE="$2"; shift 2;;
    --background) BGC=("$2" "$3" "$4"); shift 4;;
    --scenes) SCENES="$2"; shift 2;;
    --frames) FRAMES="$2"; shift 2;;
    --cameras) CAMS="$2"; shift 2;;
    --dry-run) DRY=1; shift;;
    --compare-only) COMPARE_ONLY=1; shift;;
    -h|--help) sed -n '2,/^set -u/p' "$0" | sed '$d' | sed 's/^# \{0,1\}//'; exit 0;;
    *) echo "unknown argument: $1" >&2; exit 2;;
  esac
done
case "$MODE" in
  masked-rgb-expected-quad) MODEARGS=(--observation-mode masked-rgb-expected-quad);;
  masked-rgb-constant) [ ${#BGC[@]} -eq 3 ] || { echo "--mode masked-rgb-constant needs --background R G B" >&2; exit 2; }
                       MODEARGS=(--observation-mode masked-rgb-constant --rgb-background-constant "${BGC[@]}");;
  *) echo "--mode must be masked-rgb-expected-quad or masked-rgb-constant (the original arm is always run)" >&2; exit 2;;
esac
[ "$ORIG" != "$MASKED" ] || { echo "the two arms need separate roots" >&2; exit 2; }
for r in "$ORIG" "$MASKED"; do
  [ "$r" != "$STATIC" ] || { echo "an arm root must not be the static source root $STATIC" >&2; exit 2; }
done
CMP=${CMP:-$(dirname "$MASKED")/$(basename "$MASKED")_vs_original}
PY=${PY:-/home/samuelemara/miniconda3/envs/rigeuler3dgs/bin/python}
COMMON=(--stages "phase1 phase2 phase3")
[ -n "$FRAMES" ] && COMMON+=(--frames "$FRAMES")
[ -n "$CAMS" ] && COMMON+=(--cameras "$CAMS")

sha() { if [ -f "$1" ]; then sha256sum "$1" | cut -d' ' -f1; else echo MISSING; fi; }
static_copy() {  # static_copy SRC DST: the trained model without checkpoints / logs, never overwrite
  local src="$1" dst="$2"
  if [ "$DRY" = "1" ]; then echo "  + copy $src -> $dst"; return 0; fi
  ( cd "$src" && find . -type f -not -name "chkpnt*.pth" -not -name "train.log" -print0 ) |
  while IFS= read -r -d '' f; do
    [ -e "$dst/$f" ] && continue
    mkdir -p "$(dirname "$dst/$f")" && cp -p "$src/$f" "$dst/$f"
  done
}
verify_static() {  # verify_static SRC DST: identical PLY, cfg_args, cameras.json
  local src="$1" dst="$2" f a b ok=0
  for f in "point_cloud/iteration_$ITER/point_cloud.ply" cfg_args cameras.json; do
    a=$(sha "$src/$f"); b=$(sha "$dst/$f")
    if [ "$a" = MISSING ] || [ "$a" != "$b" ]; then echo "  !! $dst/$f: sha256 $b, source $a"; ok=1; fi
  done
  return $ok
}

echo "static (read only) : $STATIC"
echo "dataset            : $DATA"
echo "original arm       : $ORIG   (--observation-mode original)"
echo "masked arm         : $MASKED   (${MODEARGS[*]})"
echo "comparison         : $CMP"
echo "scenes             : $SCENES"
echo "frames / cameras   : ${FRAMES:-driver default} / ${CAMS:-driver default}"

DONE=""
if [ "$COMPARE_ONLY" != "1" ]; then
  for S in $SCENES; do
    echo; echo "################ $S"
    good=1
    for e in initial final; do
      src="$STATIC/$S/static_bidi/$e"
      [ -f "$src/point_cloud/iteration_$ITER/point_cloud.ply" ] || { echo "  !! missing $src"; good=0; continue; }
      for arm in "$ORIG" "$MASKED"; do
        static_copy "$src" "$arm/$S/static_bidi/$e"
        [ "$DRY" = "1" ] || verify_static "$src" "$arm/$S/static_bidi/$e" || good=0
      done
    done
    [ "$good" = 1 ] || { echo "  !! $S skipped: static endpoints missing or not identical in both arms"; continue; }
    [ "$DRY" = "1" ] || echo "  static endpoints verified identical in both arms (sha256 PLY, cfg_args, cameras.json)"
    C1=(bash run_bidirectional_all.sh --input "$DATA" --output "$ORIG" --scenes "$S" "${COMMON[@]}"
        --observation-mode original)
    C2=(bash run_bidirectional_all.sh --input "$DATA" --output "$MASKED" --scenes "$S" "${COMMON[@]}"
        "${MODEARGS[@]}")
    if [ "$DRY" = "1" ]; then
      echo "  + $(printf '%q ' "${C1[@]}")"
      echo "  + $(printf '%q ' "${C2[@]}")"
      continue
    fi
    echo "=== $S: ORIGINAL arm (fresh Phase 1/2/3)"
    "${C1[@]}"
    echo "=== $S: MASKED arm (fresh Phase 1/2/3)"
    "${C2[@]}"
    DONE="$DONE $S"
    echo "=== $S: manifest diff (only the observation block and the roots may differ)"
    "$PY" tools/compare_object_masks.py --original "$ORIG" --masked "$MASKED" --scenes "$S" \
      --out "$CMP/per_scene/$S" | sed -n '1,40p'
  done
fi

echo
echo "=== comparison -> $CMP"
CS=(); [ "$COMPARE_ONLY" = "1" ] || CS=(--scenes "${DONE# }")
if [ "$DRY" = "1" ]; then
  echo "  + $PY tools/compare_object_masks.py --original $ORIG --masked $MASKED --out $CMP"
  echo "  (dry run: nothing was copied or run)"
elif [ "$COMPARE_ONLY" != "1" ] && [ -z "${DONE# }" ]; then
  echo "  no scene ran"; exit 1
else
  "$PY" tools/compare_object_masks.py --original "$ORIG" --masked "$MASKED" --out "$CMP" "${CS[@]}"
fi
