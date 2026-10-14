#!/usr/bin/env bash
# Fused (forward + backward) free-motion Phase-1 IEKS for ONE scene.
#
#   bash scripts/run_fused_phase1.sh --scene washer_103776 \
#        --reuse-existing-phase1 --profile washer-validation \
#        --tag fused_phase1_v1 --resume
#
# Reads the EXISTING static_bidi/{initial,final} models and the EXISTING
# bidirectional/{forward,backward}/phase1_centers READ ONLY, builds the
# two-source candidate union, and runs ONE jointly rendered full-window
# RGB-D IEKS over all original frames. It never runs static training and
# never runs the legacy forward/backward IEKF.
#
# Arguments
#   --scene NAME              case-preserving scene directory name
#   --reuse-existing-phase1   required: consume the historical F/B results
#   --profile P               washer-validation | overnight | smoke
#                               washer-validation / overnight: ALL Gaussians,
#                               ALL original frames, ALL calibrated cameras,
#                               up to 10 outer iterations x 60 PCG;
#                               overnight adds a 180-minute per-scene budget
#                               (the posterior is the same).
#                               smoke: 1,500 Gaussians per source, 1 camera,
#                               written under <tag>__smoke, SMOKE_SUBSAMPLED=YES
#   --linear-backend B        legacy_reference (default) | exact_cached_structured
#                               the SAME system; exact_cached_structured freezes
#                               each accepted state's linearisations in host RAM
#                               instead of re-rendering them on every PCG step
#   --preconditioner P        diagonal | block12 | temporal_block_tridiagonal
#                               a solver choice only: A and b do not change
#   --transfer-policy T       synchronous (default) | double_buffered
#   --cache-host-budget-gib G refuse (CACHE_HOST_BUDGET_EXCEEDED) rather than swap
#   --pcg-tol R / --pcg-iters K   declared linear-solve target / cap per step
#   --tag T                   experiment name under unified_phase1_experiments/
#   --resume                  continue a matching experiment (see below)
#   --time-budget-min M       override the profile's time budget (0 = none)
#   --initial-full-state D    optional full-state export (legacy rerun with
#   --final-full-state D        --save-full-state) instead of the derived guess
#   --preflight               JSON check of inputs, CUDA, disk; no GPU work
#   --dry-run                 print every command and output path; no GPU work
#   --results-root / --dataset-root   override the default mount points
#
# Resume: an experiment is identified by a fingerprint of its sources,
# frames, cameras, population and solver code. Same fingerprint -> continue
# (a CONVERGED optimiser is reused; otherwise ONE new attempt starts from the
# best accepted state so far). Different fingerprint -> refused, never
# overwritten. A Washer test and the overnight run with the same --tag share
# one experiment: the profiles differ only in solve budget.
#
# RAW_UNION_EXPERIMENT vs VALID_CANONICAL_SCENE: the joint IEKS optimises the
# motion of the RAW UNION of two independently trained Gaussian mixtures. That
# is a well-defined conditional posterior, and it is what runs. It is NOT a
# validated canonical scene: the two mixtures represent most surfaces twice,
# and the theory gives no association model to merge them. Reports say
# CANONICAL_PHYSICAL_SCENE_VALIDITY = UNRESOLVED.
#
# Stop cleanly: touch <experiment>/STOP  (ends after the current outer
# iteration; the best accepted state is kept).
# Exit: 0 converged, 3 ran but not converged, 1 failed, 2 blocked.
set -u
if [ "${1:-}" = "--help" ] || [ "${1:-}" = "-h" ]; then
  sed -n '2,/^set -u/p' "$0" | sed 's/^# \{0,1\}//' | sed '$d'
  exit 0
fi
. "$(dirname "${BASH_SOURCE[0]}")/_fused_phase1_python.sh"
cd "$FUSED_REPO" || exit 2
exec "$FUSED_PY" tools/fused_phase1_driver.py scene "$@"
