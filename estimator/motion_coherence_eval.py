"""Ground-truth evaluation of a static/dynamic prediction. EVALUATION ONLY.

CODEX_IMPLEMENT_MOTION_COHERENCE_STATIC_DYNAMIC_FILTER.md sections 21-24.

This module is deliberately a separate file from the predictor. Section 21
requires that GT never reaches graph construction, the coherence score, the
affinity, the spectral cut, the static/dynamic identification, the Kabsch fit
or ambiguity handling -- and the cleanest way to guarantee that is for the
predictor to have no path to this code at all. `motion_coherence_support.py`
imports nothing from here, and a regression test asserts it.

`require_prediction_on_disk` is the order guard: GT cannot be loaded until the
prediction has been written, so a future edit that quietly moves evaluation
earlier fails loudly instead of leaking.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import torch
from torch import Tensor


def require_prediction_on_disk(path) -> Path:
    """Section 21's order guard. Call before touching any GT."""
    p = Path(path)
    if not p.is_file():
        raise RuntimeError(
            "section 21: ground truth may only be loaded AFTER the prediction "
            f"has been written to disk, and {p} does not exist. This guard is "
            "what keeps the coherence branch honest -- if evaluation can run "
            "before prediction, GT can leak into it.")
    return p


def load_static_dynamic_gt(source, mu0: Tensor, max_dist: float | None = None) -> dict:
    """Section 22: the real GT representation, mapped to Gaussians.

    The datasets ship a SPARSE labelled point cloud, `pointcloud/
    points3D_parts.ply`, with an integer `part` field. The repository's own
    convention, in `utils/static_endpoint_scene.py`, is `moving = part != 0`;
    that is read from the code rather than guessed.

    The cloud is sparse (360-6234 points against 7k-40k Gaussians), so labels
    are carried to Gaussians by nearest neighbour in the canonical frame. A
    Gaussian farther than `max_dist` from any labelled point is UNMAPPED and is
    excluded from every metric rather than being assigned a guess.
    """
    from plyfile import PlyData

    src = Path(source) / "pointcloud" / "points3D_parts.ply"
    if not src.is_file():
        return {"available": False, "source": str(src),
                "reason": "no points3D_parts.ply in this dataset"}
    v = PlyData.read(str(src))["vertex"].data
    xyz = torch.tensor(np.stack([v["x"], v["y"], v["z"]], 1), dtype=torch.float64)
    part = torch.tensor(np.asarray(v["part"]).astype(np.int64))
    moving = part != 0                       # utils/static_endpoint_scene.py

    x = torch.as_tensor(mu0).double().cpu()
    d = torch.cdist(x, xyz)
    nn = d.argmin(1)
    nnd = d.gather(1, nn[:, None]).squeeze(1)
    if max_dist is None:
        # Scale-free: the typical spacing of the labelled cloud itself, times
        # a fixed factor. Not a tuned per-dataset number.
        spacing = float(torch.cdist(xyz, xyz).fill_diagonal_(
            float("inf")).min(1).values.median())
        max_dist = 4.0 * spacing
    mapped = nnd <= max_dist
    labels = torch.full((x.shape[0],), -1, dtype=torch.long)
    labels[mapped] = moving[nn[mapped]].long()
    return {
        "available": True, "source": str(src),
        "mapping_method": ("nearest neighbour in the canonical frame from the "
                           "sparse labelled point cloud; moving = part != 0 "
                           "(utils/static_endpoint_scene.py); Gaussians "
                           "farther than max_dist are unmapped and excluded"),
        "max_dist": float(max_dist),
        "num_gt_points": int(xyz.shape[0]),
        "num_static": int((labels == 0).sum()),
        "num_dynamic": int((labels == 1).sum()),
        "num_unmapped": int((labels < 0).sum()),
        "coverage": float((labels >= 0).double().mean()),
        "gt_dynamic_fraction_of_mapped": (
            float((labels == 1).sum()) / max(int((labels >= 0).sum()), 1)),
        "labels": labels,
    }


def classification_metrics(pred_dynamic: Tensor, gt_labels: Tensor) -> dict:
    """Section 23. Raw accuracy is deliberately not reported on its own."""
    gt = torch.as_tensor(gt_labels).long().reshape(-1)
    pred = torch.as_tensor(pred_dynamic).bool().reshape(-1)
    m = gt >= 0
    gtd, prd = (gt[m] == 1), pred[m]
    tp = int((prd & gtd).sum())
    fp = int((prd & ~gtd).sum())
    fn = int((~prd & gtd).sum())
    tn = int((~prd & ~gtd).sum())

    def _d(a, b):
        return a / b if b else float("nan")

    pd_, rd = _d(tp, tp + fp), _d(tp, tp + fn)
    ps, rs = _d(tn, tn + fn), _d(tn, tn + fp)
    f1d = _d(2 * pd_ * rd, pd_ + rd) if (pd_ == pd_ and rd == rd) else float("nan")
    f1s = _d(2 * ps * rs, ps + rs) if (ps == ps and rs == rs) else float("nan")
    den = math.sqrt(float((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)))
    return {
        "confusion_matrix": {"tp": tp, "fp": fp, "fn": fn, "tn": tn},
        "dynamic_precision": pd_, "dynamic_recall": rd, "dynamic_f1": f1d,
        "dynamic_iou": _d(tp, tp + fp + fn),
        "static_precision": ps, "static_recall": rs, "static_f1": f1s,
        "balanced_accuracy": (rd + rs) / 2 if (rd == rd and rs == rs)
        else float("nan"),
        "matthews_corrcoef": ((tp * tn - fp * fn) / den) if den > 0
        else float("nan"),
        "predicted_dynamic_fraction": _d(tp + fp, tp + fp + fn + tn),
        "gt_dynamic_fraction": _d(tp + fn, tp + fp + fn + tn),
        "num_evaluated": int(m.sum()), "num_excluded_unmapped": int((~m).sum()),
        "note": ("section 23: raw accuracy is not reported alone; with a 9% "
                 "dynamic class, predicting everything static scores 91%"),
    }


def gt_edge_coherence(C: Tensor, edge_i: Tensor, edge_j: Tensor,
                      node_gt: Tensor, node_pred: Tensor) -> dict:
    """Section 35 / 44: is GT itself metrically coherent?

    If GT-static-to-GT-static edges score as badly as cross edges, the failure
    is in the Phase-1 trajectories, not in the partition that reads them.
    """
    C = torch.as_tensor(C).double().reshape(-1)

    def split(lab):
        li, lj = lab[edge_i], lab[edge_j]
        ok = (li >= 0) & (lj >= 0)
        return ({"within_static": C[ok & (li == 0) & (lj == 0)],
                 "within_dynamic": C[ok & (li == 1) & (lj == 1)],
                 "cross": C[ok & (li != lj)]})

    out = {}
    for name, lab in (("gt", node_gt), ("predicted", node_pred)):
        for key, vals in split(lab).items():
            out[f"{name}_{key}_median_C"] = (float(vals.median())
                                             if vals.numel() else float("nan"))
            out[f"{name}_{key}_count"] = int(vals.numel())
    return out


def compare_selectors(rows: list) -> dict:
    """Section 24: a table, and nothing else. No hybrid support is produced."""
    return {"rows": rows,
            "note": ("section 24: comparison is EVALUATION ONLY. The two "
                     "selectors are never unioned, intersected, averaged or "
                     "voted; neither seeds the other.")}


def write_json(path, payload) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(payload, indent=1, default=str))
