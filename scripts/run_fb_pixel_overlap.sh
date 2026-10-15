#!/usr/bin/env bash
# Forward/backward pixel overlap, same-surface evidence and complementary
# coverage of the EXISTING Phase-1 sources, for ONE scene. Read-only.
#
#   bash scripts/run_fb_pixel_overlap.sh --scene washer_103776 \
#        --frames all --cameras all --mode all --tag overlap_coverage_v1 --resume
#
# Reads static_bidi/{initial,final} and bidirectional/{forward,backward}/
# phase1_centers READ ONLY, renders each source ALONE at every requested
# original (frame, camera) against the dataset's own RGB-D, and writes only to
#   <results-root>/<scene>/observation_overlap_coverage/<tag>/
# It never trains, merges, prunes or optimises anything, and never renders the
# raw F+B union.
#
# Arguments
#   --scene NAME          scene directory name (case preserved)
#   --frames F            all | a:b | comma list of ORIGINAL frame indices
#   --cameras C           all | comma list (cam_000,cam_004)
#   --mode M              coverage | pairs | all  (default all)
#                           coverage: per-pixel categories, per-Gaussian exact
#                                     contribution accounting, maps, sweeps
#                           pairs:    every (F,B) co-contributor pair, merged
#                                     over the sequence (bounded storage, see
#                                     observation_diagnostics/pair_evidence.py)
#                           all:      both, then aggregation, case studies,
#                                     galleries and report/
#   --tag T               experiment name
#   --resume              continue an experiment with the same fingerprint
#                         (inputs, frames, cameras, thresholds, measurement
#                         code); a different fingerprint is refused
#   --preflight           JSON check of inputs, cameras, CUDA, disk; exits
#   --dry-run             print the plan; exits
#   --maps all|none|LIST  frames whose per-pixel PNG/NPZ maps are written
#   --pair-prune TAU      store a view's pair only if its per-view O >= TAU
#                         (default 1e-4; the pruned mass is accounted and bounds
#                         every pair result; 0 = store all, needs much more RAM)
#   --results-root / --dataset-root   override the default mount points
#
# Progress is printed per original frame with an ETA; logs/ keeps a copy.
# Interrupting is safe: --resume continues from the last finished frame
# (coverage) or the last 10-frame checkpoint (pairs).
# Exit: 0 complete (or pilot complete), 3 ran but incomplete, 1 failed, 2 blocked.
set -u
. "$(dirname "$0")/_fused_phase1_python.sh"
cd "$FUSED_REPO" || exit 2
if [ $# -eq 0 ] || [ "${1:-}" = "-h" ] || [ "${1:-}" = "--help" ]; then
  sed -n '2,/^set -u/p' "$0" | sed 's/^# \{0,1\}//' | sed '$d'
  exit 0
fi
exec "$FUSED_PY" -u tools/analyze_fb_pixel_overlap.py "$@"
