"""Part-label identity across windows.

CODEX_REQUIRED_MODIFICATIONS_MULTIVIEW_MULTIFRAME_KPART_IEKF.md section 21.

With K > 1 the labels the optimiser assigns are arbitrary: nothing ties "part 2"
in one window to "part 2" in the next. Section 21 requires explicit identity
matching with a cost built from support overlap, axis direction, axis-line
distance, joint-type agreement and trajectory continuity, solved by Hungarian
assignment.
"""

from __future__ import annotations

import math

import numpy as np
import torch

EPS = 1e-12


def _jaccard(a: torch.Tensor, b: torch.Tensor) -> float:
    x, y = a.bool(), b.bool()
    u = int((x | y).sum())
    return 1.0 if u == 0 else int((x & y).sum()) / u


def match_cost(prev_support, cur_support, prev_model, cur_model,
               prev_q=None, cur_q=None, *,
               w_overlap: float = 1.0, w_axis: float = 0.5,
               w_line: float = 0.3, w_type: float = 0.5,
               w_traj: float = 0.2) -> tuple[float, dict]:
    """Section 21's five-term cost between one previous and one current part.

    Every term is normalised to [0,1] so the weights mean what they say, and
    every term is skipped (contributing nothing, with its weight removed from
    the normaliser) when the information it needs is absent -- an unresolved
    model must not be punished as if it had a wrong axis.
    """
    terms, used = {}, 0.0
    cost = 0.0

    ov = _jaccard(prev_support, cur_support)
    terms["support_overlap"] = ov
    cost += w_overlap * (1.0 - ov)
    used += w_overlap

    if prev_model is not None and cur_model is not None:
        c = float(torch.clamp((prev_model.axis @ cur_model.axis).abs(), 0.0, 1.0))
        ang = math.degrees(math.acos(c))
        terms["axis_angle_deg"] = ang
        cost += w_axis * min(ang / 90.0, 1.0)
        used += w_axis

        same = prev_model.kind == cur_model.kind
        terms["type_agreement"] = same
        cost += w_type * (0.0 if same else 1.0)
        used += w_type

        if prev_model.kind == "revolute" and cur_model.kind == "revolute":
            d = prev_model.line.distance_to(cur_model.line)
            terms["axis_line_distance"] = d["line_distance"]
            # saturating normaliser: beyond one unit the lines are simply
            # different, and a larger distance carries no extra information
            cost += w_line * min(d["line_distance"], 1.0)
            used += w_line

    if prev_q is not None and cur_q is not None and len(prev_q) and len(cur_q):
        gap = abs(float(cur_q[0]) - float(prev_q[-1]))
        terms["trajectory_gap"] = gap
        cost += w_traj * min(gap, 1.0)
        used += w_traj

    return cost / max(used, EPS), terms


def match_parts(prev_supports: list, cur_supports: list,
                prev_models: list, cur_models: list,
                prev_q: list | None = None, cur_q: list | None = None) -> dict:
    """Hungarian assignment from current part indices to previous ones.

    Section 21 allows exact matching for small K; Hungarian IS exact and costs
    nothing at this size, so it is used unconditionally rather than branching on
    K and having two code paths to keep correct.
    """
    from scipy.optimize import linear_sum_assignment

    K = len(cur_supports)
    Kp = len(prev_supports)
    C = np.zeros((K, Kp), dtype=np.float64)
    detail = {}
    for i in range(K):
        for j in range(Kp):
            c, terms = match_cost(
                prev_supports[j], cur_supports[i],
                prev_models[j] if j < len(prev_models) else None,
                cur_models[i] if i < len(cur_models) else None,
                prev_q[j] if prev_q and j < len(prev_q) else None,
                cur_q[i] if cur_q and i < len(cur_q) else None)
            C[i, j] = c
            detail[f"{i}->{j}"] = terms
    row, col = linear_sum_assignment(C)
    # permutation maps CURRENT label k (1-based) -> PREVIOUS label (1-based)
    perm = {int(r) + 1: int(c) + 1 for r, c in zip(row, col)}
    return {"permutation": perm,
            "cost_matrix": C.tolist(),
            "total_cost": float(C[row, col].sum()),
            "per_pair": detail,
            "is_identity": all(k == v for k, v in perm.items()),
            "method": "hungarian"}


def apply_permutation(labels: torch.Tensor, perm: dict) -> torch.Tensor:
    """Relabel a support map so its part indices match the previous window.

    Static (0) is never permuted: it is not a part, and letting it move would
    make the mapping meaningless.
    """
    out = labels.clone()
    for cur, prev in perm.items():
        out[labels == int(cur)] = int(prev)
    return out
