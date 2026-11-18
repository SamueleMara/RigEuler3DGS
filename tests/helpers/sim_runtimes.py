"""Simulator runtimes for the sim-asset tests, from the environment only (never a hard-coded workstation path):
    RIGEULER_SIM_PYTHON / RIGEULER_USD_PYTHON / RIGEULER_MUJOCO_PYTHON   interpreter(s) with pxr / mujoco
    RIGEULER_SAPIEN_PYTHON                                              interpreter with sapien (+ parttwin1d)
A missing runtime makes the engine part of a test report NOT_RUN (printed), never PASS.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from recovery.sim_asset_export.validate import find_runtime  # noqa: E402

_cache = {}


def runtime(kind):
    if kind not in _cache:
        _cache[kind] = find_runtime(kind)["python"]
    return _cache[kind]


def report_not_run(what, kind):
    print(f"  NOT_RUN {what}: no {kind} runtime (set RIGEULER_{kind.upper()}_PYTHON)")


def run_all(globs):
    names = [n for n in globs if n.startswith("test_")]
    for n in names:
        globs[n]()
        print(f"PASS {n}")
    print(f"{len(names)}/{len(names)} tests passed")
