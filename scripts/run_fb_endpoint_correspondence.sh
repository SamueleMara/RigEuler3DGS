#!/usr/bin/env bash
# Endpoint correspondence, missing geometry and visibility diagnosis for ONE
# scene, with the existing IEKF renderer and the ESTIMATED joints. Read only.
#
#   bash scripts/run_fb_endpoint_correspondence.sh --scene washer_103776 \
#        --frames endpoints --cameras all --estimated-joints auto \
#        --tag endpoint_evidence_v1 --preflight | --dry-run | --execute
#
# Contrasts: A  F0 (initial PLY) vs B0 (backward transport) at frame 0
#            B  FT (forward transport) vs BT (final PLY) at frame T
#            C  F0 vs BT in world coordinates on observed-static pixels
# J0 joint-agnostic and J1 joint-informed (estimated rigid part motion from
# bidirectional/<d>_rigid) are reported side by side. A GT firewall refuses
# GT joints, part labels, masks, meshes and evaluation files process-wide.
#
# Arguments
#   --scene NAME                 scene directory name (case preserved)
#   --frames endpoints           frames 0 and T (the only mode)
#   --cameras all|LIST           e.g. cam_000,cam_004
#   --estimated-joints auto      J1 where a USABLE estimate exists, else J0 only
#                      require   block if no USABLE estimate exists
#                      off       J0 only
#   --tag T                      experiment name
#   --preflight                  metadata, camera and joint-provenance checks; no GPU work
#   --dry-run                    inputs, outputs, joint modes and render counts; no GPU work
#   --execute                    run
#   --resume                     reuse a complete experiment with the same fingerprint
#   --gpu-memory-fraction F      cap this process on a shared GPU (e.g. 0.25)
#   --results-root / --dataset-root / --out-dir   override locations
#   --python PATH                interpreter (default: the rigeuler3dgs env)
#
# Output: <results-root>/<scene>/endpoint_correspondence/<tag>/ (start with
# visuals/overview.html and endpoint_summary.md).
# Exit: 0 done, 1 failed (or a protected input changed / firewall refusal), 2 blocked.
set -u
ARGS=()
while [ $# -gt 0 ]; do
  case "$1" in
    --python) export PY="$2"; shift 2;;
    -h|--help) sed -n '2,/^set -u/p' "$0" | sed 's/^# \{0,1\}//' | sed '$d'; exit 0;;
    *) ARGS+=("$1"); shift;;
  esac
done
. "$(dirname "$0")/_fused_phase1_python.sh"
cd "$FUSED_REPO" || exit 2
exec "$FUSED_PY" -u -W ignore tools/analyze_fb_endpoint_correspondence.py "${ARGS[@]}"
