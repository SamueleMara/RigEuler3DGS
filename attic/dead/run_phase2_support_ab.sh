#!/usr/bin/env bash
# Section 43 / 29-31: Phase-2 A/B over the two static/dynamic selectors.
#
#     current   support -> Phase2 Screw / P / R
#     coherence support -> Phase2 Screw / P / R
#
# The two supports are NEVER combined (section 20). This runs them as two
# independent experiments and prints one comparison table.
#
# The coherence arm loads the ALREADY-FROZEN predicted_support.json written by
# tools/run_motion_coherence_support.py, so Phase 2 sees exactly the Gaussian
# indices that were scored against the optimized labelled GT (section 41).
#
#   FRAMES=60 CAMS=8 NG=0 CAMERA=1 ./run_phase2_support_ab.sh
#   SCENES="storage_45135" ARMS="coherence" ./run_phase2_support_ab.sh
#
set -u
PY=${PY:-/home/samuelemara/miniconda3/envs/rigeuler3dgs/bin/python}
cd "$(dirname "$0")"

if ! $PY -c "import torch,sys; sys.exit(0 if torch.cuda.is_available() else 1)" 2>/dev/null; then
  echo "!! CUDA is not available -- the NVIDIA kernel module is not loaded."
  echo "   Check:  lsmod | grep nvidia ; ls /dev/nvidia*"
  echo "   Fix:    sudo modprobe nvidia nvidia_uvm nvidia_modeset   (or reboot)"
  echo "   Phase 2 drives the CUDA rasterizer, so this cannot run yet."
  exit 1
fi

# Cheapest scene first (Gaussian count 7.5k, 9k, 11k, 28k, 41k): an overrun
# costs the last scene rather than all of them.
SCENES=${SCENES:-"blade_103706 stapler_103111 USB_100109 fridge_10905 storage_45135"}
ARMS=${ARMS:-"current coherence"}
FRAMES=${FRAMES:-60}
CAMS=${CAMS:-8}
NG=${NG:-0}                 # 0 = every Gaussian; MUST match the coherence run
CAMERA=${CAMERA:-1}
BOOT=${BOOT:-12}
SCREW_ITERS=${SCREW_ITERS:-10}
PR_ITERS=${PR_ITERS:-5}
FOLDS=${FOLDS:-2}
POINTS=${POINTS:-5000}
OUTROOT=${OUTROOT:-output}
LOGDIR=${LOGDIR:-$OUTROOT/phase2_ab_logs}
mkdir -p "$LOGDIR"
RUN_START=$(date +%s)

echo "=============================================================="
echo " PHASE-2 A/B  --  current support  vs  coherence-v2 support"
echo "   scenes : $SCENES"
echo "   arms   : $ARMS"
echo "   budget : $FRAMES frames, $CAMS cameras, gaussians $([ "$NG" = 0 ] && echo ALL || echo $NG)"
echo "   iters  : phase1 $BOOT  screw $SCREW_ITERS  P/R $PR_ITERS  folds $FOLDS"
echo "   render : camera $CAMERA, every frame"
echo "=============================================================="

STATUS=()
for S in $SCENES; do
  for ARM in $ARMS; do
    OUT="$OUTROOT/phase2_ab_${ARM}_$S"
    LOG="$LOGDIR/${S}_${ARM}.log"
    echo ""
    echo "######## $S  [$ARM]  ->  $OUT"
    T0=$(date +%s)
    $PY tools/run_complete_joint_debug.py \
        --scene "$S" \
        --phase2-support-source "$ARM" \
        --camera "$CAMERA" \
        --frames "$FRAMES" \
        --cameras "$CAMS" \
        --max-gaussians "$NG" \
        --bootstrap-iters "$BOOT" \
        --screw-iters "$SCREW_ITERS" \
        --pr-iters "$PR_ITERS" \
        --pr-folds "$FOLDS" \
        --max-debug-points "$POINTS" \
        --frames-on-cpu \
        --out "$OUT" > "$LOG" 2>&1
    RC=$?
    E=$(( $(date +%s) - T0 ))
    if [ $RC -eq 0 ]; then
      STATUS+=("$S/$ARM OK ${E}s")
      grep -E 'frozen moving support|R status:|peak CUDA|fold [0-9]' "$LOG" | head -8
    else
      # Section 45: a failure is reported, never papered over.
      STATUS+=("$S/$ARM FAILED(rc=$RC) ${E}s")
      echo "  !! FAILED -- last 20 lines of $LOG:"
      tail -20 "$LOG"
    fi
  done
done

echo ""
echo "=============================================================="
echo " RESULTS   (total $(( $(date +%s) - RUN_START ))s)"
echo "=============================================================="
for s in "${STATUS[@]}"; do echo "  $s"; done

echo ""
$PY - "$OUTROOT" $SCENES <<'PYEOF'
import json, sys
from pathlib import Path
root, scenes = sys.argv[1], sys.argv[2:]
hdr = (f"{'scene':<15}{'support':<11}{'moving%':>8}{'screw E':>12}{'P E':>12}"
       f"{'R E':>12}{'R status':>26}")
print(hdr); print("-" * len(hdr))
for s in scenes:
    for arm in ("current", "coherence"):
        p = Path(root) / f"phase2_ab_{arm}_{s}" / "comparison" / "summary.json"
        if not p.exists():
            print(f"{s:<15}{arm:<11}{'-- no summary.json --'}")
            continue
        d = json.load(p.open())
        tr = d["train"]
        frac = 100 * d["num_moving_gaussians"] / max(
            d.get("num_gaussians") or 1, 1) if d.get("num_gaussians") else \
            float("nan")
        def e(k):
            v = tr.get(k)
            return f"{v['total']:.4e}" if v else "--"
        rs = d.get("revolute", {}).get("R_fit_status", "?")
        print(f"{s:<15}{arm:<11}{frac:>7.1f}%{e('screw'):>12}"
              f"{e('prismatic'):>12}{e('revolute'):>12}{rs:>26}")
    # held-out, per arm
    for arm in ("current", "coherence"):
        q = Path(root) / f"phase2_ab_{arm}_{s}" / "comparison" / "heldout_metrics.csv"
        if not q.exists():
            continue
        import csv
        rows = list(csv.DictReader(q.open()))
        ph = [float(r["P_heldout"]) for r in rows if r["P_heldout"] not in ("", "nan")]
        rh = [float(r["R_heldout"]) for r in rows if r["R_heldout"] not in ("", "nan")]
        interp = all(r["comparison_interpretable"] == "True" for r in rows)
        print(f"{'':<15}{arm:<11}heldout  P {sum(ph)/len(ph):.4e}"
              if ph else f"{'':<15}{arm:<11}heldout  P --", end="")
        print(f"   R {sum(rh)/len(rh):.4e}" if rh else "   R --",
              f"  interpretable={interp}")
print("\nSection 32: evidence only. The two supports are never combined, and")
print("no P/R classifier is produced from this table.")
PYEOF
