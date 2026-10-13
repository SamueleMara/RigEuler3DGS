#!/usr/bin/env bash
# Real forward / backward / raw-union free-motion Phase 1, end to end.
#
# CODEX_PRIORITY_REAL_BIDIRECTIONAL_PHASE1_MERGE_AND_VISUALIZE.md sections 3-7.
#
# Everything is written under
#   $OUT_ROOT/$SCENE/unified_phase1_experiments/$EXP_TAG/
# and NOTHING outside it: static_bidi/, bidirectional/forward|backward and the
# *_rigid trees are read only. A stage whose output already holds a manifest
# is SKIPPED (resume), never deleted or overwritten.
#
#   SCENE=washer_103776 bash run_real_bidi_phase1.sh
#   SCENE=storage_45135 bash run_real_bidi_phase1.sh
#   STAGES="union report" SCENE=washer_103776 bash run_real_bidi_phase1.sh
#
# Stages, in order (STAGES selects a subset):
#   phase1    re-run the EXISTING free Phase 1, forward and backward, with the
#             same settings run_bidirectional_all.sh used, plus
#             --full-state-out. This is what produces the 12-D warm starts.
#   diag      whitened process innovations of both 12-D exports
#   baseline  render-only: real | F track | B track | raw union
#   fwd       forward-only IEKS ladder, initial source anchored at frame 0
#   bwd       backward-only IEKS ladder, final source anchored at frame 59
#   union     ONE joint raw-union IEKS over both sources, 60 frames, camera
#             ladder 1 -> 2 -> 4 (UNION_MAX_CAMS caps it)
#   report    one comparison over every variant's best accepted state
set -u
cd "$(dirname "$0")"

PY=${PY:-/home/samuelemara/miniconda3/envs/rigeuler3dgs/bin/python}
SCENE=${SCENE:-washer_103776}
SRC_ROOT=${SRC_ROOT:-"/media/samuelemara/Extreme SSD/Samuele/Backup_09_10_26/Sapien/output/preparation_8cams/partnet_subset_fixed_render_linear"}
OUT_ROOT=${OUT_ROOT:-"/media/samuelemara/Extreme SSD/Samuele/results_11_10_26/ours"}
EXP_TAG=${EXP_TAG:-real_bidi_phase1_v1}
STAGES=${STAGES:-"phase1 diag baseline fwd bwd union report"}
UNION_MAX_CAMS=${UNION_MAX_CAMS:-4}
OUTER=${OUTER:-10}
PCG=${PCG:-60}

SRC="$SRC_ROOT/$SCENE"
OUT="$OUT_ROOT/$SCENE"
EXP="$OUT/unified_phase1_experiments/$EXP_TAG"
LOGS="$EXP/logs"
mkdir -p "$LOGS"
STREAM="--stream-frames --frames-on-cpu --process-factors-on-cpu --diagonal-preconditioner"
COMMON=(--scene "$SCENE" --data-root "$SRC" --out-root "$OUT"
        --initial-states "$EXP/free_forward/full_state"
        --final-states "$EXP/free_backward/full_state")

has() { case " $STAGES " in *" $1 "*) return 0;; *) return 1;; esac; }
done_() { [ -f "$1/manifest.json" ] || [ -f "$1/full_state_manifest.json" ]; }
say() { echo "[$(date +%T)] $*" | tee -a "$LOGS/driver.log"; }

for p in "$SRC" "$OUT/static_bidi/initial" "$OUT/static_bidi/final"; do
  [ -e "$p" ] || { echo "missing: $p"; exit 2; }
done
say "scene $SCENE -> $EXP"

run_cli() {   # run_cli <out> <log name> args...
  local o="$1" name="$2"; shift 2
  if done_ "$o"; then say "  skip $name (present)"; return 0; fi
  say "  $name"
  "$PY" tools/run_unified_phase1_real_bidirectional.py "${COMMON[@]}" --out "$o" "$@" \
    > "$LOGS/$name.log" 2>&1
  local rc=$?
  grep -E "status |flags:|OOM" "$LOGS/$name.log" | tail -2 | sed 's/^/      /' | tee -a "$LOGS/driver.log"
  [ $rc -eq 0 ] || say "    !! $name exited $rc (see $LOGS/$name.log)"
}

# ---- 1. the 12-D warm starts ---------------------------------------------
if has phase1; then
  for DIR in forward backward; do
    [ "$DIR" = forward ] && END=initial || END=final
    D="$EXP/free_$DIR"
    if done_ "$D/full_state"; then say "  skip phase1/$DIR (present)"; continue; fi
    [ -e "$D" ] && { say "  !! $D exists without a full-state manifest; move it aside"; exit 3; }
    say "  phase1/$DIR from static_bidi/$END (same settings as run_bidirectional_all.sh)"
    /usr/bin/time -v "$PY" tools/run_motion_coherence_support.py \
      --scene "$SCENE" --direction "$DIR" --canonical-endpoint "$END" \
      --model "$OUT/static_bidi/$END" --source "$SRC" \
      --frames 60 --cameras 8 --bootstrap-iters 12 \
      --iekf-supernode-size 64 --coherence-knn 12 --supernode-count 192 \
      --cv-folds 5 --frames-on-cpu \
      --out "$D" --full-state-out "$D/full_state" \
      > "$LOGS/phase1_$DIR.log" 2>&1 || say "    !! phase1/$DIR failed"
    grep -E "full state:|Elapsed|Maximum resident" "$LOGS/phase1_$DIR.log" | sed 's/^/      /' | tee -a "$LOGS/driver.log"
    # does the re-run reproduce the historical Phase 1 it replaces?
    "$PY" - "$OUT/bidirectional/$DIR/phase1_centers" "$D/phase1_centers" \
          "$EXP/diagnostics/phase1_reproduction_$DIR.json" <<'PYEOF' | tee -a "$LOGS/driver.log"
import sys, json, glob, os
import numpy as np
old, new, out = sys.argv[1:4]
rows = []
for f in sorted(glob.glob(os.path.join(new, "centers_frame_*.npy"))):
    o = os.path.join(old, os.path.basename(f))
    if os.path.isfile(o):
        d = np.linalg.norm(np.load(f) - np.load(o), axis=1)
        rows.append({"frame": os.path.basename(f), "max": float(d.max()),
                     "mean": float(d.mean())})
os.makedirs(os.path.dirname(out), exist_ok=True)
json.dump({"historical": old, "rerun": new, "per_frame": rows,
           "note": "CUDA nondeterminism means a re-run is not bitwise identical"},
          open(out, "w"), indent=1)
if rows:
    print(f"      reproduction vs historical centres: worst max "
          f"{max(r['max'] for r in rows):.3e} m, worst mean "
          f"{max(r['mean'] for r in rows):.3e} m")
PYEOF
  done
fi

# ---- 2. process innovations, 12-D -----------------------------------------
if has diag; then
  for DIR in forward backward; do
    J="$EXP/diagnostics/full_12d/process_innovations_$DIR.json"
    [ -f "$J" ] && { say "  skip diag/$DIR (present)"; continue; }
    say "  diag/$DIR"
    "$PY" tools/diagnose_process_innovations.py --full-state "$EXP/free_$DIR/full_state" \
      --label "$DIR" --out "$EXP/diagnostics/full_12d" > "$LOGS/diag_$DIR.log" 2>&1
    grep -E "chi2 total|share|176" "$LOGS/diag_$DIR.log" | sed 's/^/      /' | tee -a "$LOGS/driver.log"
  done
fi

# ---- 3. render-only baseline ------------------------------------------------
has baseline && run_cli "$EXP/render_baseline_full12d" baseline --mode render_baseline

# ---- 4/5. single-source ladders (each window contains its own anchor) ------
if has fwd; then
  run_cli "$EXP/fwd_0_8_2cam"  fwd_0_8  --mode forward_only --frames 0:8  --camera-ids cam_000,cam_004 --max-outer $OUTER --pcg-iters $PCG
  run_cli "$EXP/fwd_0_16_2cam" fwd_0_16 --mode forward_only --frames 0:16 --camera-ids cam_000,cam_004 --max-outer $OUTER --pcg-iters $PCG
  run_cli "$EXP/fwd_0_60_2cam" fwd_0_60 --mode forward_only --frames 0:60 --camera-ids cam_000,cam_004 --max-outer $OUTER --pcg-iters $PCG $STREAM
fi
if has bwd; then
  run_cli "$EXP/bwd_52_60_2cam" bwd_52_60 --mode backward_only --frames 52:60 --camera-ids cam_000,cam_004 --max-outer $OUTER --pcg-iters $PCG
  run_cli "$EXP/bwd_44_60_2cam" bwd_44_60 --mode backward_only --frames 44:60 --camera-ids cam_000,cam_004 --max-outer $OUTER --pcg-iters $PCG
  run_cli "$EXP/bwd_0_60_2cam"  bwd_0_60  --mode backward_only --frames 0:60  --camera-ids cam_000,cam_004 --max-outer $OUTER --pcg-iters $PCG $STREAM
fi

# ---- 6. the joint raw-union experiment: 60 frames keep BOTH anchors -------
if has union; then
  run_cli "$EXP/union_0_60_1cam" union_1cam --mode raw_union_experiment \
    --representation-policy diagnostic-only --frames 0:60 --camera-ids cam_000 \
    --max-outer $OUTER --pcg-iters $PCG $STREAM
  [ "$UNION_MAX_CAMS" -ge 2 ] && run_cli "$EXP/union_0_60_2cam" union_2cam --mode raw_union_experiment \
    --representation-policy diagnostic-only --frames 0:60 --camera-ids cam_000,cam_004 \
    --max-outer $OUTER --pcg-iters $PCG $STREAM
  [ "$UNION_MAX_CAMS" -ge 4 ] && run_cli "$EXP/union_0_60_4cam" union_4cam --mode raw_union_experiment \
    --representation-policy diagnostic-only --frames 0:60 --camera-ids cam_000,cam_002,cam_004,cam_006 \
    --max-outer $OUTER --pcg-iters $PCG $STREAM
  [ "$UNION_MAX_CAMS" -ge 8 ] && run_cli "$EXP/union_0_60_8cam" union_8cam --mode raw_union_experiment \
    --representation-policy diagnostic-only --frames 0:60 \
    --camera-ids cam_000,cam_001,cam_002,cam_003,cam_004,cam_005,cam_006,cam_007 \
    --max-outer $OUTER --pcg-iters $PCG $STREAM
fi

# ---- 7. one report over every best accepted state --------------------------
if has report; then
  EXTRA=()
  for d in fwd_0_60_2cam bwd_0_60_2cam union_0_60_1cam union_0_60_2cam union_0_60_4cam union_0_60_8cam; do
    [ -f "$EXP/$d/best_accepted/state.npz" ] && EXTRA+=("$EXP/$d")
  done
  run_cli "$EXP/report_all_variants" report --mode render_baseline \
    --extra-variants-from "${EXTRA[@]}"
fi
say "done: $EXP"
