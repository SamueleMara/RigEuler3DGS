#!/usr/bin/env bash
# LEGACY / ABLATION RUNNER.
#
# The production joint-discovery pipeline is ./run_rigid_se3_all.sh --
# Phase 1 -> coherence MOVING_RIGID -> rigid SE(3) RGB-D Phase 2 -> post-hoc
# joint reduction. On all five datasets its axes are better, by up to 89
# degrees (stapler 89.22 -> 0.23, storage 72.91 -> 0.23). This script runs the
# superseded direct-axis estimators and is kept for regression and ablations.
# See RIGID_SE3_PHASE2_AND_JOINT_REDUCTION_REPORT.md section 21.
#
# Section 43: the complete joint debug on every dataset, one shared config.
#
# Section 33 is explicit that the frame/camera budget must be the SAME for
# every scene -- a per-scene budget tuned until the numbers look good is not
# evidence. Override the budget if you like, but override it for ALL scenes:
#
#   FRAMES=60 CAMS=8 NG=0 CAMERA=1 ./run_complete_joint_debug_all.sh
#   SCENES="storage_45135 blade_103706" ./run_complete_joint_debug_all.sh
#
set -u
PY=${PY:-/home/samuelemara/miniconda3/envs/rigeuler3dgs/bin/python}
cd "$(dirname "$0")"

# Ordered CHEAPEST FIRST (by Gaussian count: 7k, 9k, 11k, 28k, 39k).
# Cost is ~linear in Gaussians, and storage alone is 42% of the total work. If
# the run overruns, this ordering means four scenes are complete rather than
# one, and each scene writes its whole artifact tree before the next starts.
SCENES=${SCENES:-"blade_103706 stapler_103111 USB_100109 fridge_10905 storage_45135"}
FRAMES=${FRAMES:-60}          # every dataset frame
CAMS=${CAMS:-8}               # every synchronized camera
NG=${NG:-0}                   # 0 = EVERY Gaussian in the model (7k-39k)
CAMERA=${CAMERA:-1}           # section 23: rendered for EVERY frame
# Iteration caps, chosen from a measurement on this box (see TIME BUDGET
# below), not from taste. They are CAPS: the trust region stops a frame as
# soon as it converges, so the real run lands well under the upper bound.
BOOT=${BOOT:-12}              # Phase-1 iterations per frame
SCREW_ITERS=${SCREW_ITERS:-10}
PR_ITERS=${PR_ITERS:-5}
FOLDS=${FOLDS:-2}             # 8 cameras -> a clean 4 train / 4 held-out split
POINTS=${POINTS:-5000}        # section 26: the 3D page's point budget only
# 60 frames x 8 cameras of 640x480 RGB-D pin ~2.4 GiB on the GPU. On an 8 GiB
# card that leaves too little for the rasterizer at the full Gaussian count,
# so the window is staged in host memory. Set FRAMES_ON_CPU=0 on a larger card.
DEADLINE_H=${DEADLINE_H:-5}
DEADLINE_S=$(echo "$DEADLINE_H*3600" | bc -l | cut -d. -f1)
RUN_START=$(date +%s)
FRAMES_ON_CPU=${FRAMES_ON_CPU:-1}
[ "$FRAMES_ON_CPU" = "0" ] && FRAMES_ON_CPU=""
OUTROOT=${OUTROOT:-output}
LOGDIR=${LOGDIR:-$OUTROOT/complete_joint_debug_logs}
mkdir -p "$LOGDIR"

echo "=============================================================="
echo " COMPLETE JOINT DEBUG -- all scenes, one shared configuration"
echo "   scenes   : $SCENES"
echo "   frames   : $FRAMES      cameras: $CAMS      gaussians: $([ "$NG" = "0" ] && echo 'ALL' || echo "$NG")"
echo "   iters    : phase1 $BOOT   screw $SCREW_ITERS   P/R $PR_ITERS   folds $FOLDS"
echo "   render   : camera $CAMERA, every frame"
echo "   staging  : $([ -n "$FRAMES_ON_CPU" ] && echo 'frames on host RAM' || echo 'frames on GPU')"
echo "   logs     : $LOGDIR"
echo "   order    : cheapest scene first"
echo "   deadline : ${DEADLINE_H}h (remaining scenes are SKIPPED, not truncated)"
echo "--------------------------------------------------------------"
echo " TIME BUDGET  (measured: storage 39405 Gaussians x 8 cameras ="
echo "  3.35 s per frame x iteration; 8.07 s summed over all 5 scenes)"
echo "    fit upper bound = 484 s x (SCREW_ITERS + 2*FOLDS*PR_ITERS)"
printf "                    = %.1f h   + ~0.9 h overhead = %.1f h\n" \
  "$(echo "484*($SCREW_ITERS+2*$FOLDS*$PR_ITERS)/3600" | bc -l)" \
  "$(echo "484*($SCREW_ITERS+2*$FOLDS*$PR_ITERS)/3600+0.9" | bc -l)"
echo "  Raise SCREW_ITERS/PR_ITERS if the early scenes come in fast."
echo "=============================================================="

STATUS=()
for S in $SCENES; do
  OUT="$OUTROOT/complete_joint_debug_$S"
  LOG="$LOGDIR/$S.log"
  SPENT=$(( $(date +%s) - RUN_START ))
  if [ "$SPENT" -ge "$DEADLINE_S" ]; then
    # Section 42: a scene is either complete or reported missing. Starting one
    # that cannot finish would leave a half-written tree that looks like a run.
    echo ""
    echo "######## $S  SKIPPED -- ${DEADLINE_H}h budget already spent (${SPENT}s)"
    STATUS+=("$S SKIPPED(deadline)")
    continue
  fi
  echo ""
  echo "######## $S  ->  $OUT     [${SPENT}s of ${DEADLINE_S}s spent]"
  START=$(date +%s)
  $PY tools/run_complete_joint_debug.py \
      --scene "$S" \
      --camera "$CAMERA" \
      --frames "$FRAMES" \
      --cameras "$CAMS" \
      --max-gaussians "$NG" \
      --bootstrap-iters "$BOOT" \
      --screw-iters "$SCREW_ITERS" \
      --pr-iters "$PR_ITERS" \
      --pr-folds "$FOLDS" \
      --max-debug-points "$POINTS" \
      ${FRAMES_ON_CPU:+--frames-on-cpu} \
      --out "$OUT" > "$LOG" 2>&1
  RC=$?
  ELAPSED=$(( $(date +%s) - START ))
  if [ $RC -eq 0 ]; then
    STATUS+=("$S OK ${ELAPSED}s")
    echo "  [${ELAPSED}s for $S; $(( $(date +%s) - RUN_START ))s of ${DEADLINE_S}s total]"
    tail -14 "$LOG"
  else
    # Section 42: a failed scene is REPORTED, never papered over with an old
    # output or quietly skipped.
    STATUS+=("$S FAILED(rc=$RC) ${ELAPSED}s")
    echo "  !! FAILED (rc=$RC) -- last 25 lines of $LOG:"
    tail -25 "$LOG"
  fi
done

echo ""
echo "=============================================================="
echo " RESULTS"
echo "=============================================================="
for s in "${STATUS[@]}"; do echo "  $s"; done
echo "  total wall: $(( $(date +%s) - RUN_START ))s"
echo ""
for S in $SCENES; do
  OUT="$OUTROOT/complete_joint_debug_$S"
  echo "  $S"
  echo "    dir      $OUT"
  echo "    3D       $OUT/html/joint_motion_debug.html"
  echo "    renders  $OUT/html/chosen_camera_all_frames.html"
done

echo ""
echo "=============================================================="
echo " CROSS-SCENE SUMMARY"
echo "=============================================================="
$PY - "$OUTROOT" $SCENES <<'PYEOF'
import json, sys
from pathlib import Path

root, scenes = sys.argv[1], sys.argv[2:]
GT = {"USB_100109": "revolute (continuous)", "storage_45135": "prismatic",
      "fridge_10905": "revolute", "stapler_103111": "revolute",
      "blade_103706": "prismatic"}
hdr = (f"{'scene':<16}{'GT':<22}{'model':<11}{'total E':>12}"
       f"{'heldout':>12}  geometry status")
print(hdr); print("-" * len(hdr))
for s in scenes:
    p = Path(root) / f"complete_joint_debug_{s}" / "comparison" / "summary.json"
    if not p.exists():
        print(f"{s:<16}{GT.get(s,'?'):<22}{'-- no summary.json (run failed) --'}")
        continue
    d = json.load(p.open())
    ho = {}
    hp = p.parent / "heldout_metrics.csv"
    if hp.exists():
        import csv
        rows = list(csv.DictReader(hp.open()))
        for m, k in (("prismatic", "P_heldout"), ("revolute", "R_heldout")):
            v = [float(r[k]) for r in rows if r[k] not in ("", "nan")]
            ho[m] = sum(v) / len(v) if v else float("nan")
    for m in ("phase1", "screw", "prismatic", "revolute"):
        tr = d["train"].get(m)
        if tr is None:
            st = d.get("revolute", {}).get("R_fit_status", "not fitted")
            print(f"{s:<16}{GT.get(s,'?'):<22}{m:<11}{'--':>12}{'--':>12}  {st}")
            continue
        if m == "screw":
            g = d["screw"]["geometry_consistency"]
            st = (f"trans {100*d['screw']['translation_only_share']:.1f}% "
                  f"cam-cos {g['across_cameras']['median_pairwise_cosine']:+.3f}")
        elif m == "revolute":
            st = d["revolute"].get("R_fit_status", "?")
        elif m == "prismatic":
            st = f"travel {d['prismatic']['total_travel']:.4f}"
        else:
            st = "free"
        h = ho.get(m)
        print(f"{s:<16}{GT.get(s,'?'):<22}{m:<11}{tr['total']:>12.4e}"
              f"{(f'{h:.4e}' if h==h and h is not None else '--'):>12}  {st}")
    print(f"{'':<16}{'':<22}heldout interpretable: "
          f"{d.get('heldout_interpretable')}")
PYEOF
