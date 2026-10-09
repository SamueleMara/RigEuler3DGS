"""Stage-2 multi-frame twist refinement entry point.

This is intentionally gated until progressive Stage 1 has a stable label file.
It exists so callers fail explicitly instead of falling back to the old
closed-loop consensus/K-part machinery.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def monotonic_coordinates_from_raw(raw_increments: torch.Tensor, *, q0: float = 0.0) -> torch.Tensor:
    """Positive-increment q(t) parameterization for the later Stage-4 solver."""
    inc = F.softplus(torch.as_tensor(raw_increments).float())
    return float(q0) + torch.cumsum(inc, dim=-1)


def run_multiframe_twist_refinement(*_, **__):
    raise NotImplementedError(
        "Progressive Stage 2 is gated behind Stage 1 convergence. "
        "Implement shared T_k(t)=exp(q_k(t) xi_k) refinement here after "
        "the progressive Stage-1 acceptance checks pass."
    )
