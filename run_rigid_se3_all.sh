#!/usr/bin/env bash
# Sections 37, 38, 44: the production pipeline on all five datasets.
#
#     Phase 1 -> coherence MOVING_RIGID -> rigid SE(3) Phase 2
#             -> joint reduction P + R -> pose-space comparison
#
# No axis, direction, centre or joint type is a parameter of the RGB-D
# optimiser. Section 38's controls:
#     blade, storage    expected primarily PRISMATIC
#     stapler, USB, fridge   expected REVOLUTE  (stapler is the forensic case)
#
#   ./run_rigid_se3_all.sh
#   PHASE1_FROM=output/complete_joint_debug ./run_rigid_se3_all.sh   # reuse
#   SCENES="stapler_103111" ./run_rigid_se3_all.sh
#
set -u
PY=${PY:-/home/samuelemara/miniconda3/envs/rigeuler3dgs/bin/python}
cd "$(dirname "$0")"

if ! $PY -c "import torch,sys; sys.exit(0 if torch.cuda.is_available() else 1)" 2>/dev/null; then
  echo "!! CUDA is not available. Rigid Phase 2 drives the rasterizer."
  echo "   Check:  lsmod | grep nvidia ; ls /dev/nvidia*"
  exit 1
fi

# Stapler first: section 37 makes it the forensic case, so an overrun costs a
# control rather than the run's whole reason for existing.
SCENES=${SCENES:-"stapler_103111 blade_103706 USB_100109 fridge_10905 storage_45135"}
FRAMES=${FRAMES:-60}
CAMS=${CAMS:-8}
NG=${NG:-0}
CAMERA=${CAMERA:-1}
BOOT=${BOOT:-12}
RIGID_ITERS=${RIGID_ITERS:-8}
CG=${CG:-40}
POINTS=${POINTS:-6000}
RENDERS=${RENDERS:-1}
PHASE1_FROM=${PHASE1_FROM:-}
OUTROOT=${OUTROOT:-output}
# Where the FROZEN coherence supports live. Separate from OUTROOT on purpose:
# redirecting this run's output must not change where its input support is
# looked up, or a scratch run silently reports "no support".
SUPROOT=${SUPROOT:-output}
TAG=${TAG:-rigid_se3}
LOGDIR=${LOGDIR:-$OUTROOT/${TAG}_logs}
mkdir -p "$LOGDIR"
RUN_START=$(date +%s)

echo "=============================================================="
echo " RIGID SE(3) PHASE 2  +  POST-HOC JOINT REDUCTION"
echo "   scenes  : $SCENES"
echo "   budget  : $FRAMES frames, $CAMS cameras, gaussians $([ "$NG" = 0 ] && echo ALL || echo "$NG")"
echo "   phase 1 : $([ -n "$PHASE1_FROM" ] && echo "REUSED from $PHASE1_FROM" || echo "fresh, $BOOT iters")"
echo "   phase 2 : rigid SE(3), $RIGID_ITERS iters, $CG CG"
echo "   support : coherence MOVING_RIGID (frozen, no fallback)"
echo "=============================================================="

STATUS=()
for S in $SCENES; do
  OUT="$OUTROOT/${TAG}_$S"
  LOG="$LOGDIR/${S}.log"
  EXTRA=()
  [ "$RENDERS" = "0" ] && EXTRA+=(--no-renders)
  if [ -n "$PHASE1_FROM" ]; then
    C="${PHASE1_FROM}_${S}/phase1/centers"
    if [ ! -d "$C" ]; then
      echo "######## $S  SKIPPED -- no centers at $C"; STATUS+=("$S SKIPPED"); continue
    fi
    EXTRA+=(--phase1-centers "$C")
  fi
  SUP="$SUPROOT/motion_coherence_$S/predicted_support.json"
  if [ ! -f "$SUP" ]; then
    echo "######## $S  SKIPPED -- no frozen coherence support at $SUP"
    echo "         run ./run_support_labelling_all.sh or"
    echo "         tools/run_motion_coherence_support.py --scene $S first"
    STATUS+=("$S SKIPPED(no support)"); continue
  fi
  echo ""
  echo "######## $S  ->  $OUT"
  T0=$(date +%s)
  $PY tools/run_rigid_se3_pipeline.py \
      --scene "$S" \
      --frames "$FRAMES" \
      --cameras "$CAMS" \
      --camera "$CAMERA" \
      --max-gaussians "$NG" \
      --bootstrap-iters "$BOOT" \
      --rigid-iters "$RIGID_ITERS" \
      --rigid-cg-iters "$CG" \
      --max-debug-points "$POINTS" \
      --frames-on-cpu \
      --coherence-support "$SUP" \
      "${EXTRA[@]}" \
      --out "$OUT" > "$LOG" 2>&1
  RC=$?
  E=$(( $(date +%s) - T0 ))
  if [ $RC -eq 0 ]; then
    STATUS+=("$S OK ${E}s")
    grep -E 'exact rigidity|vs Phase-1 Kabsch|gamma_1DOF|E_P |R axis |GT type' "$LOG"
  else
    STATUS+=("$S FAILED(rc=$RC) ${E}s")
    echo "  !! FAILED -- last 25 lines of $LOG:"
    tail -25 "$LOG"
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
hdr = (f"{'scene':<16}{'GT':<10}{'gamma1DOF':>10}{'E_P':>11}{'E_R':>11}"
       f"{'P ang':>8}{'R ang':>8}{'R line':>9}{'cond':>10}{'rad/L':>9}")
print(hdr); print("-" * len(hdr))
for s in scenes:
    p = Path(root) / f"{tag}_{s}" / "summary.json"
    if not p.exists():
        print(f"{s:<16}-- no summary.json --"); continue
    d = json.load(p.open())
    g = d["gt_evaluation"]
    P, R = d["prismatic"], d["revolute"]
    ep = P["pose_residual"]["weighted_energy"]
    er = R["pose_residual"]["weighted_energy"]
    print(f"{s:<16}{str(g.get('gt_joint_type')):<10}"
          f"{d['one_dof']['gamma_1dof']:>10.4f}{ep:>11.3e}{er:>11.3e}"
          f"{g['prismatic']['axis_angle_deg']:>8.2f}"
          f"{g['revolute']['axis_angle_deg']:>8.2f}"
          f"{g['revolute']['axis_line_error']['distance']:>9.4f}"
          f"{R['finite_axis_solver']['condition_number']:>10.2e}"
          f"{R.get('axis_radius_over_part_scale', float('nan')):>9.1f}"
          if g.get("available") else f"{s:<16}no GT")
print()
hdr2 = (f"{'scene':<16}{'rigidity drift':>16}{'orient defect':>15}"
        f"{'max d_rot(Kabsch)':>19}{'max d_p':>10}")
print(hdr2); print("-" * len(hdr2))
for s in scenes:
    p = Path(root) / f"{tag}_{s}" / "summary.json"
    if not p.exists(): continue
    d = json.load(p.open())
    gate, k = d["exact_rigidity_gate"], d["phase1_kabsch_vs_rgbd"]
    print(f"{s:<16}{gate['max_pairwise_distance_drift']:>16.3e}"
          f"{gate['max_orientation_defect']:>15.3e}"
          f"{k['max_d_rotation_rad']:>19.4f}{k['max_d_translation']:>10.4f}")
print("\nrad/L is the far-axis tell: order 1 means the revolute axis lies on the")
print("part; a large value means R is approximating a translation. It is")
print("reported, not thresholded -- section 26 forbids classifying from one")
print("statistic, and on a near-P scene the pose energy alone can prefer R.")
print("\nSection 35: the P/R comparison happens in POSE space. No direct P/R")
print("RGB-D training loss is used in this branch, and no axis was a")
print("parameter of the image optimiser (section 49).")
PYEOF
