#!/usr/bin/env bash
# Fused Phase-1 IEKS for EVERY eligible scene, one at a time on the one GPU.
#
#   bash scripts/run_fused_phase1_overnight.sh --all-existing-scenes \
#        --reuse-existing-phase1 --profile overnight --tag fused_phase1_v1 --resume
#
# Discovers scenes under the results root (no frozen list). Eligible = both
# endpoint PLYs, both historical forward/backward phase1_centers with valid
# manifests, matching frame counts, identical camera gauge, RGB-D on the
# dataset disk. Ineligible scenes are listed with the precise reason and are
# NEVER retrained. Each eligible scene runs through scripts/run_fused_phase1.sh
# (the same validated single-scene path), strictly serially; a failure is
# recorded and the next scene starts.
#
# Arguments
#   --all-existing-scenes     required
#   --reuse-existing-phase1   required
#   --profile P               overnight (default) | washer-validation | smoke
#   --tag T                   per-scene experiment name
#   --resume                  continue matching experiments; without it a scene
#                             whose experiment already exists is skipped
#   --scene-time-budget-min M override the profile's per-scene budget
#   --linear-backend B, --preconditioner P, --transfer-policy T,
#   --cache-host-budget-gib G  passed to every scene (see run_fused_phase1.sh)
#   --list-scenes             print eligible / skipped scenes and exit
#   --preflight               write eligible_scenes.json and exit; no GPU work
#   --dry-run                 print the serial queue of commands; no GPU work
#
# Outputs: per scene under <scene>/unified_phase1_experiments/<tag>/; the
# run's own files under <results>/unified_phase1_overnight/<tag>/:
#   eligible_scenes.json, ALL_SCENES_SUMMARY.{md,csv,json}, logs/<scene>.log
# Clean stop: touch <results>/unified_phase1_overnight/<tag>/STOP
# Exit: 0 all converged, 3 some partial, 1 any failed, 2 blocked.
set -u
if [ "${1:-}" = "--help" ] || [ "${1:-}" = "-h" ]; then
  sed -n '2,/^set -u/p' "$0" | sed 's/^# \{0,1\}//' | sed '$d'
  exit 0
fi
. "$(dirname "${BASH_SOURCE[0]}")/_fused_phase1_python.sh"
cd "$FUSED_REPO" || exit 2
echo "fused Phase-1 overnight: shell PID $$, started $(date)"
exec "$FUSED_PY" tools/fused_phase1_driver.py overnight "$@"
