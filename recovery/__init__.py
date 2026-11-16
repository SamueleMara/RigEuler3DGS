"""Phase 4 (geometry recovery) and Phase 5 (Gaussian-only RGB-D-mask training) for RigEuler3DGS.

Both stages run downstream of an immutable Phase 3 (one selected direction, one canonical endpoint, fixed
joint type / geometry / q). See RigEuler3DGS_PHASE4_PHASE5_CODEX_IMPLEMENTATION_2026-11-10.md.

Importing this package pins the rasterizer extension to THIS checkout's
submodules/diff-gaussian-rasterization, which carries the additive RGB + alpha + metric-depth entry point
(`rasterize_gaussians_rgbd`). The conda environment's editable install points at another checkout; the new
stages must never fall back to it silently, so a missing entry point is an import error.
"""
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
_DGR = REPO / "submodules" / "diff-gaussian-rasterization"

if "diff_gaussian_rasterization" not in sys.modules:
    sys.path.insert(0, str(_DGR))
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

import diff_gaussian_rasterization as _dgr  # noqa: E402

RASTERIZER_PATH = Path(_dgr.__file__).resolve().parent
if not hasattr(_dgr._C, "rasterize_gaussians_rgbd") or RASTERIZER_PATH != (_DGR / "diff_gaussian_rasterization").resolve():
    raise ImportError(
        f"recovery needs the RGB-D rasterizer of {_DGR} (rasterize_gaussians_rgbd); loaded {RASTERIZER_PATH}. "
        "Import `recovery` before anything that imports diff_gaussian_rasterization, and build the extension: "
        "cd submodules/diff-gaussian-rasterization && CUDA_HOME=/usr/local/cuda-12.8 python setup.py build_ext --inplace")
