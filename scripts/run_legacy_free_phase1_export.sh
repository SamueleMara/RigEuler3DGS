#!/usr/bin/env bash
# OPT-IN: reproduce ONE historical legacy free Phase-1 direction, non-destructively.
#
#   bash scripts/run_legacy_free_phase1_export.sh --scene washer_103776 \
#        --direction backward --tag legacy_export_v1 --preflight
#   ... same with --dry-run, then --execute [--save-full-state]
#
# NOT a prerequisite of the fused Phase 1, and never called by it. Runs the
# SAME tools/run_motion_coherence_support.py with the historical settings into
# <results>/<scene>/unified_phase1_experiments/<tag>/legacy_iekf_<direction>/,
# optionally recording the full mu, R, v, omega state, then compares the
# rerun's centres with the historical ones. Never run_bidirectional_all.sh,
# never static training, never a write inside static_bidi/ or bidirectional/.
#
# Nothing runs without exactly one of --preflight, --dry-run, --execute.
# A rerun takes as long as the original Phase 1 did (minutes per direction).
set -u
if [ "${1:-}" = "--help" ] || [ "${1:-}" = "-h" ]; then
  sed -n '2,/^set -u/p' "$0" | sed 's/^# \{0,1\}//' | sed '$d'
  exit 0
fi
. "$(dirname "${BASH_SOURCE[0]}")/_fused_phase1_python.sh"
cd "$FUSED_REPO" || exit 2
exec "$FUSED_PY" tools/legacy_phase1_rerun.py "$@"
