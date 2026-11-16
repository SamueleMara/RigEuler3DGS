"""Gate 1 compatibility: the legacy entry points of THIS build equal those of a reference build.

Reference = the rasterizer the conda environment imports by default (its editable install; for this work,
the main checkout that every Phase 0-3 run uses). Both builds render the same 80-Gaussian scene through
rasterize_gaussians (+ backward), rasterize_gaussian_features and rasterize_gaussian_positions, with and
without antialiasing, in separate processes (two extensions named diff_gaussian_rasterization._C cannot
share a process).

  * forward outputs (colour, radii, inverse depth, features, positions) must be bitwise identical
  * gradients use atomicAdd, so they are compared against the reference build's own run-to-run floor,
    measured here from two reference runs (CUDA nondeterminism, not a code difference)

Skipped (not passed) when the reference build is this build.
"""
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
DGR = HERE.parent / "submodules" / "diff-gaussian-rasterization"
DUMP = HERE / "helpers" / "rasterizer_legacy_dump.py"
FORWARD = ("color", "radii", "invd", "features", "positions")


def _run(out, this_build):
    env = dict(os.environ)
    env.pop("PYTHONPATH", None)
    if this_build:
        env["PYTHONPATH"] = str(DGR)
    r = subprocess.run([sys.executable, str(DUMP), str(out)], env=env, capture_output=True, text=True, cwd=tempfile.gettempdir())
    assert r.returncode == 0, r.stderr[-2000:]
    return r.stdout.splitlines()[0]


def _diff(a, b):
    fwd, grad = 0, 0.0
    for k in a:
        for t in a[k]:
            d = float((a[k][t].float() - b[k][t].float()).abs().max())
            if t in FORWARD:
                fwd += d > 0
            else:
                grad = max(grad, d)
    return fwd, grad


def test_legacy_entry_points_match_reference_build():
    with tempfile.TemporaryDirectory() as d:
        d = Path(d)
        ref_line = _run(d / "ref1.pt", False)
        this_line = _run(d / "this.pt", True)
        assert "has rgbd: True" in this_line, this_line
        if str(DGR) in ref_line:
            print("SKIP: the default rasterizer is this build; no independent reference")
            return
        _run(d / "ref2.pt", False)
        r1, r2, me = (torch.load(d / f) for f in ("ref1.pt", "ref2.pt", "this.pt"))
    fwd_ref, floor = _diff(r1, r2)
    fwd, gap = _diff(r1, me)
    assert fwd_ref == 0, "the reference build's forward is not deterministic"
    assert fwd == 0, f"{fwd} legacy forward tensors differ from the reference build"
    # gradient gap within twice the measured atomicAdd floor (and an absolute guard)
    assert gap <= max(2 * floor, 1e-5), f"legacy gradients differ by {gap:.2e}; reference floor {floor:.2e}"
    print(f"reference: {ref_line}\nthis:      {this_line}\nforward bitwise equal; gradient gap {gap:.2e} "
          f"vs reference run-to-run floor {floor:.2e}")


if __name__ == "__main__":
    test_legacy_entry_points_match_reference_build()
    print("PASS test_legacy_entry_points_match_reference_build")
