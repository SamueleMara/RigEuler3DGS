# Sourced by the fused Phase-1 wrappers. Selects and validates the
# rigeuler3dgs interpreter itself, so the user-facing commands need no
# `conda activate` (which does not work in a non-interactive shell).
FUSED_PY="${PY:-/home/samuelemara/miniconda3/envs/rigeuler3dgs/bin/python}"
if [ ! -x "$FUSED_PY" ]; then
  echo "BLOCKED: interpreter not found or not executable: $FUSED_PY" >&2
  echo "         set PY=/path/to/rigeuler3dgs/bin/python" >&2
  exit 2
fi
if ! "$FUSED_PY" -c "import torch, plyfile" >/dev/null 2>&1; then
  echo "BLOCKED: $FUSED_PY cannot import torch/plyfile; wrong environment?" >&2
  exit 2
fi
FUSED_REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
