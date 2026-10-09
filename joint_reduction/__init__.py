"""Post-hoc joint reduction.

CODEX_IMPLEMENT_RIGID_SE3_PHASE2_AND_POSTHOC_JOINT_REDUCTION.md, Part II.

Section 24: `rigid_trajectory_reduction` reads the recovered rigid trajectory
and its uncertainty, and nothing else. It must not import the renderer, and a
test enforces that.

The convenience names below are resolved LAZILY. This package predates that
module and holds many others that `estimator.*` imports; an eager
`from .rigid_trajectory_reduction import ...` here would make every one of
those imports also pull in the reduction -- and since the reduction imports
back into `estimator.manifold`, that is a cycle waiting to be tripped by
import order rather than by anything a caller did.
"""

from __future__ import annotations

_LAZY = {
    "PRISMATIC": "rigid_trajectory_reduction",
    "REVOLUTE": "rigid_trajectory_reduction",
    "fit_axis_point": "rigid_trajectory_reduction",
    "fit_prismatic": "rigid_trajectory_reduction",
    "fit_revolute_axis": "rigid_trajectory_reduction",
    "one_dof_diagnostics": "rigid_trajectory_reduction",
    "part_scale": "rigid_trajectory_reduction",
    "pose_residuals": "rigid_trajectory_reduction",
    "prismatic_poses": "rigid_trajectory_reduction",
    "reduce_trajectory": "rigid_trajectory_reduction",
    "refine_prismatic": "rigid_trajectory_reduction",
    "refine_revolute": "rigid_trajectory_reduction",
    "relative_increments": "rigid_trajectory_reduction",
    "relative_twist_covariance": "rigid_trajectory_reduction",
    "revolute_angles": "rigid_trajectory_reduction",
    "revolute_poses": "rigid_trajectory_reduction",
}

__all__ = sorted(_LAZY)


def __getattr__(name: str):
    mod = _LAZY.get(name)
    if mod is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    import importlib
    return getattr(importlib.import_module(f".{mod}", __name__), name)


def __dir__():
    return sorted(set(globals()) | set(_LAZY))
