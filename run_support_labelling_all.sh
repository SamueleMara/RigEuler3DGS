#!/usr/bin/env bash
# Static/dynamic LABELLING only -- coherence v2 vs the motion (Otsu) baseline.
#
# No Phase 2. No screw, no P, no R, no renders. Phase 1 produces the free
# per-Gaussian trajectories, both selectors read exactly those same
# trajectories, and each is scored against the optimized-3DGS labelled GT.
#
# Section 20/24: the two selectors are never unioned, intersected, averaged or
# voted, and neither seeds the other. The table is evaluation only.
#
#   ./run_support_labelling_all.sh                       # fresh Phase 1, all 5
#   PHASE1_FROM=output/complete_joint_debug ./run_support_labelling_all.sh
#   SCENES="storage_45135" ./run_support_labelling_all.sh
#
set -u
PY=${PY:-/home/samuelemara/miniconda3/envs/rigeuler3dgs/bin/python}
cd "$(dirname "$0")"

# Cheapest scene first (7.5k, 9k, 11k, 28k, 41k Gaussians): an overrun costs
# the last scene rather than all of them.
SCENES=${SCENES:-"blade_103706 stapler_103111 USB_100109 fridge_10905 storage_45135"}
FRAMES=${FRAMES:-60}
CAMS=${CAMS:-8}
NG=${NG:-0}                 # 0 = every Gaussian
BOOT=${BOOT:-12}            # matches the Phase-2 A/B budget
KNN=${KNN:-12}              # ONE global default; never tuned per scene
FOLDS=${FOLDS:-5}
EXTRACTOR=${EXTRACTOR:-v2}
PHASE1_FROM=${PHASE1_FROM:-}   # e.g. output/complete_joint_debug -> reuse centers
OUTROOT=${OUTROOT:-output}
TAG=${TAG:-support_labelling}
LOGDIR=${LOGDIR:-$OUTROOT/${TAG}_logs}
mkdir -p "$LOGDIR"
RUN_START=$(date +%s)

if [ -z "$PHASE1_FROM" ]; then
  if ! $PY -c "import torch,sys; sys.exit(0 if torch.cuda.is_available() else 1)" 2>/dev/null; then
    echo "!! CUDA is not available and no PHASE1_FROM was given."
    echo "   Phase 1 drives the CUDA rasterizer. Either fix the driver"
    echo "   (lsmod | grep nvidia) or reuse stored centers with PHASE1_FROM."
    exit 1
  fi
fi

echo "=============================================================="
echo " STATIC/DYNAMIC LABELLING  --  coherence v2  vs  motion baseline"
echo "   scenes    : $SCENES"
echo "   budget    : $FRAMES frames, $CAMS cameras, gaussians $([ "$NG" = 0 ] && echo ALL || echo "$NG")"
echo "   phase 1   : $([ -n "$PHASE1_FROM" ] && echo "REUSED from $PHASE1_FROM (no renderer)" || echo "fresh, $BOOT iters")"
echo "   coherence : extractor $EXTRACTOR, knn $KNN, $FOLDS CV folds"
echo "   phase 2   : NOT RUN"
echo "=============================================================="

STATUS=()
for S in $SCENES; do
  OUT="$OUTROOT/${TAG}_$S"
  LOG="$LOGDIR/${S}.log"
  CENTERS_ARG=()
  if [ -n "$PHASE1_FROM" ]; then
    C="${PHASE1_FROM}_${S}/phase1/centers"
    if [ ! -d "$C" ]; then
      echo "######## $S  SKIPPED -- no centers at $C"
      STATUS+=("$S SKIPPED(no centers)")
      continue
    fi
    CENTERS_ARG=(--phase1-centers "$C")
  fi
  echo ""
  echo "######## $S  ->  $OUT"
  T0=$(date +%s)
  $PY tools/run_motion_coherence_support.py \
      --scene "$S" \
      --frames "$FRAMES" \
      --cameras "$CAMS" \
      --max-gaussians "$NG" \
      --bootstrap-iters "$BOOT" \
      --coherence-knn "$KNN" \
      --extractor "$EXTRACTOR" \
      --cv-folds "$FOLDS" \
      --frames-on-cpu \
      "${CENTERS_ARG[@]}" \
      --out "$OUT" > "$LOG" 2>&1
  RC=$?
  E=$(( $(date +%s) - T0 ))
  if [ $RC -eq 0 ]; then
    STATUS+=("$S OK ${E}s")
    grep -E '^  GT:|^  \[' "$LOG" | head -4
  else
    # A failure is reported, never papered over.
    STATUS+=("$S FAILED(rc=$RC) ${E}s")
    echo "  !! FAILED -- last 20 lines of $LOG:"
    tail -20 "$LOG"
  fi
done

echo ""
echo "=============================================================="
echo " RESULTS   (total $(( $(date +%s) - RUN_START ))s)"
echo "=============================================================="
for s in "${STATUS[@]}"; do echo "  $s"; done
echo ""

$PY - "$OUTROOT" "$TAG" $SCENES <<'PYEOF'
import json, sys
from pathlib import Path
root, tag, scenes = sys.argv[1], sys.argv[2], sys.argv[3:]
hdr = (f"{'scene':<16}{'selector':<11}{'prec':>7}{'recall':>8}{'F1':>7}"
       f"{'IoU':>7}{'balAcc':>8}{'pred dyn':>10}{'GT dyn':>9}")
print(hdr)
print("-" * len(hdr))
best = {}
for s in scenes:
    p = Path(root) / f"{tag}_{s}" / "numerical" / "gt_metrics.json"
    if not p.exists():
        print(f"{s:<16}-- no gt_metrics.json --")
        continue
    d = json.load(p.open())
    for name in ("current", "coherence"):
        m = d.get(name)
        if not m:
            continue
        def f(x):
            return "   --  " if x != x else f"{x:>7.3f}"
        print(f"{s:<16}{name:<11}{f(m['dynamic_precision'])}"
              f"{f(m['dynamic_recall']):>8}{f(m['dynamic_f1'])}"
              f"{f(m['dynamic_iou'])}{f(m['balanced_accuracy']):>8}"
              f"{100 * m['predicted_dynamic_fraction']:>9.1f}%"
              f"{100 * m['gt_dynamic_fraction']:>8.1f}%")
        best.setdefault(s, {})[name] = m['dynamic_f1']
    print()

# A selector that predicted nothing has an undefined F1. That is an
# UNRESOLVED result, not a loss, and it is reported as one rather than being
# swept into the win count.
def ok(v):
    return len(v) == 2 and all(x == x for x in v.values())
scored = {s: v for s, v in best.items() if ok(v)}
bad = [s for s, v in best.items() if not ok(v)]
won = [s for s, v in scored.items() if v['coherence'] > v['current']]
tied = [s for s, v in scored.items() if v['coherence'] == v['current']]
print(f"coherence F1 > current on {len(won)}/{len(scored)} scored scenes"
      + (f" (tied on {len(tied)})" if tied else ""))
for s, v in scored.items():
    print(f"    {s:<16}{v['current']:.3f} -> {v['coherence']:.3f}"
          f"   ({v['coherence'] - v['current']:+.3f})")
for s in bad:
    print(f"    {s:<16}UNRESOLVED -- a selector predicted no dynamic "
          f"Gaussians, so its F1 is undefined")
print("\nSection 24: evaluation only. The two selectors are never combined,")
print("and no support is produced from this table.")
PYEOF
