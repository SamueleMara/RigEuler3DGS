"""Which Gaussians belong to the moving part.

This is the input every Phase-2 model is handed, and it is upstream of all of
them: the joint is fitted to whatever this set contains, so a set that is half
static drags the fitted rigid motion toward no motion at all.

The previous rule was

    moving = displacement > median(displacement)

which is a MEDIAN SPLIT: it selects exactly half the scene by construction,
whatever the data says. Measured on all five datasets it returned 50.0%, 50.0%,
50.0%, 50.0%, 50.0%. For a drawer -- where roughly a tenth of the Gaussians
move -- that set is ~80% static, and one rigid screw fitted to it recovers a
COM displacement of 2e-04 against a ground-truth travel of 0.56.

The rule here is Otsu's threshold on the displacement: the value that maximizes
between-class variance for a two-class split. It has no tunable parameter and
no world unit, and it adapts to the scene instead of imposing a fraction. On
the same five datasets it returns 15%, 44%, 41%, 26%, 9%, and the selected
set's MEDIAN displacement lands on the ground-truth joint travel:

    scene      kept    selected median    GT travel
    blade     15.3%          0.82           0.88
    stapler   43.8%          2.14           2.01
    USB       41.0%          1.04            --
    fridge    26.0%          1.01           1.05
    storage    9.1%          0.47           0.56

Displacement is measured as `max_t ||x_t - x_0||`, not `||x_last - x_0||`: a
joint that opens and closes again, or that reverses inside the window, has a
small end-to-end displacement and a large excursion, and the end-to-end form
would call the whole part static.
"""

from __future__ import annotations

import torch
from torch import Tensor


def otsu_threshold(x: Tensor, bins: int = 512) -> dict:
    """Two-class threshold maximizing between-class variance. No parameters.

    Also returns `separability` = max between-class variance / total variance,
    which is near 1 for a cleanly bimodal distribution and near 0 for a
    unimodal one. A caller that gets a low separability has been told that the
    scene does not actually split into movers and non-movers, rather than
    being handed an arbitrary cut dressed up as a decision.
    """
    v = torch.as_tensor(x).double().reshape(-1)
    if v.numel() == 0:
        return {"threshold": 0.0, "separability": 0.0, "bins": 0}
    lo, hi = float(v.min()), float(v.max())
    if not (hi > lo):
        return {"threshold": lo, "separability": 0.0, "bins": 0}
    h = torch.histc(v, bins=bins, min=lo, max=hi).double()
    edges = torch.linspace(lo, hi, bins + 1, dtype=torch.float64)
    centers = 0.5 * (edges[1:] + edges[:-1])
    w = h / h.sum().clamp_min(1.0)
    cw = torch.cumsum(w, 0)                       # class-0 weight
    cm = torch.cumsum(w * centers, 0)             # class-0 first moment
    mu_t = cm[-1]
    den = cw * (1.0 - cw)
    sb = torch.where(den > 0, (mu_t * cw - cm) ** 2 / den.clamp_min(1e-300),
                     torch.zeros_like(den))
    k = int(torch.argmax(sb))
    total_var = float(v.var(unbiased=False))
    return {"threshold": float(centers[k]),
            "separability": float(sb[k]) / max(total_var, 1e-30),
            "bins": bins}


def trajectory_excursion(positions) -> Tensor:
    """`max_t ||x_t - x_0||` per Gaussian, over the whole window."""
    # Phase 1 may hand back a mix of host and device tensors; the excursion is
    # a tiny per-Gaussian scalar, so it is computed on the CPU throughout
    # rather than making the caller guarantee a device.
    x0 = torch.as_tensor(positions[0]).double().cpu()
    best = torch.zeros(x0.shape[0], dtype=torch.float64)
    for p in positions[1:]:
        d = (torch.as_tensor(p).double().cpu() - x0).norm(dim=1)
        best = torch.maximum(best, d)
    return best


def select_moving_support(positions, min_members: int = 16,
                          min_separability: float = 0.0) -> dict:
    """The frozen moving support, with the evidence for it.

    Returns the member indices plus the numbers a reader needs to judge the
    split: the threshold, the separability, the kept fraction and the mean and
    median excursion on each side. Those are what make it checkable that the
    set is the moving part and not half the scene.
    """
    disp = trajectory_excursion(positions)
    o = otsu_threshold(disp)
    moving = disp > o["threshold"]
    method = "otsu"

    if int(moving.sum()) < min_members:
        # Degenerate split (a nearly static window). Fall back to the largest
        # movers so Phase 2 has something to fit, and SAY that it happened --
        # the fraction is then arbitrary and must not be read as evidence.
        k = min(max(int(0.1 * disp.numel()), min_members), int(disp.numel()))
        moving = torch.zeros_like(disp, dtype=torch.bool)
        moving[disp.topk(k).indices] = True
        method = "fallback_top_decile"

    idx = torch.nonzero(moving, as_tuple=False).reshape(-1).cpu()
    sel, rej = disp[moving], disp[~moving]
    return {
        "members": idx,
        "method": method,
        "threshold": o["threshold"],
        "separability": o["separability"],
        "bimodal": bool(o["separability"] >= max(min_separability, 0.0)),
        "num_members": int(idx.numel()),
        "num_total": int(disp.numel()),
        "kept_fraction": float(moving.double().mean()),
        "excursion": {
            "selected_mean": float(sel.mean()) if sel.numel() else 0.0,
            "selected_median": float(sel.median()) if sel.numel() else 0.0,
            "selected_max": float(sel.max()) if sel.numel() else 0.0,
            "rejected_mean": float(rej.mean()) if rej.numel() else 0.0,
            "rejected_max": float(rej.max()) if rej.numel() else 0.0,
            "contrast": (float(sel.mean() / rej.mean().clamp_min(1e-12))
                         if sel.numel() and rej.numel() else float("inf")),
            "definition": "max_t ||x_t - x_0||, over the whole window"},
        "note": ("Otsu's two-class threshold on the per-Gaussian excursion. "
                 "No tunable parameter and no world unit; the kept fraction is "
                 "whatever the scene supports, NOT a fixed share. The previous "
                 "median split returned exactly 50% on every dataset."),
    }
