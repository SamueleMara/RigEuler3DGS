"""Section 37: compose the joint map with the EXISTING renderer. Nothing else.

CODEX_IMPLEMENT_PHASE2_RIGID_SCREW_IEKF.md sections 0, 20-24, 37, 38.

    H_J   = H_G J_{G,J}
    H_J^T = J_{G,J}^T H_G^T

`H_G` is Phase 1's `FrameLinearization.apply_H` / `apply_HT` and is not
reimplemented, not wrapped in autograd and not modified. The CUDA contributor
traversal, the alpha-clamp derivative and the forward/JVP/VJP consistency are
exactly the corrected Phase-1 operators (section 38).

This file deliberately contains no measurement model, no robust weighting and
no solver: those are Phase 1's and stay Phase 1's (section 0).
"""

from __future__ import annotations

import torch
from torch import Tensor

from .joint_kinematics import full_jvp, full_vjp


def _renderer_dtype(linearization):
    """The dtype Phase-1's cached operators expect."""
    return (linearization.views[0].rgb.dtype if linearization.views
            else torch.float32)


def joint_apply_H(linearization, state, frame: int, delta: Tensor,
                  mu0: Tensor, R0: Tensor | None = None, bases=None):
    """H_J v = H_G (J_{G,J} v) -> one RGB-D tangent per camera."""
    # Stage B's physical models carry their own exact joint map; the generic
    # screw keeps `full_jvp`.  Everything from here down -- the renderer, the
    # measurement model, the solver -- is shared and untouched.
    pose = (state.jvp(frame, delta, mu0, R0) if hasattr(state, "jvp")
            else full_jvp(state, frame, delta, mu0, R0, bases))
    # The joint layer works in float64 because its tangent bases and SE(3)
    # series need it; the renderer's cached operators are float32. Casting at
    # this single boundary keeps the precision where it matters without
    # touching the Phase-1 operators.
    return linearization.apply_H(pose.to(_renderer_dtype(linearization)))


def joint_apply_HT(linearization, state, frame: int, pixel_adjoints,
                   mu0: Tensor, R0: Tensor | None = None, bases=None) -> Tensor:
    """H_J^T z = J_{G,J}^T (H_G^T z) -> joint error state [6K]."""
    dt = _renderer_dtype(linearization)
    pose = linearization.apply_HT([(a.to(dt), b.to(dt))
                                   for a, b in pixel_adjoints])
    return (state.vjp(frame, pose, mu0, R0) if hasattr(state, "vjp")
            else full_vjp(state, frame, pose, mu0, R0, bases))


def joint_normal_operator(linearization, state, frame: int, precisions,
                          mu0: Tensor, R0: Tensor | None = None, bases=None):
    """v -> J^T H^T W H J v, matrix-free (section 29).

    Built from the two operators above and the measurement precision Phase 1
    already computes, so the Phase-2 normal equations inherit Phase-1's robust
    weighting and visibility handling rather than re-deriving them.
    """
    def op(v: Tensor) -> Tensor:
        tangents = joint_apply_H(linearization, state, frame, v, mu0, R0, bases)
        weighted = [(pc.to(dc.dtype) * dc, pd.to(dd.dtype) * dd)
                    for (pc, pd), (dc, dd) in zip(precisions, tangents)]
        return joint_apply_HT(linearization, state, frame, weighted, mu0, R0,
                              bases)
    return op
