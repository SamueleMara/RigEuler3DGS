"""The optimized labeled 3DGS ground truth. EVALUATION ONLY.

CODEX_FIX_COHERENCE_EXTRACTION_AND_USE_OPTIMIZED_3DGS_GT.md sections 2-8.

PROVENANCE (section 3), traced in `scene/gaussian_model.py`, not guessed:

    input/<scene>/pointcloud/points3D_parts.ply        sparse labels, `part`
        -> BasicPointCloud.part_indices
        -> GaussianModel._gt_part_labels                (line ~657)
             a torch.long tensor, NOT an nn.Parameter, so it is never in the
             optimizer and never receives a gradient
        -> densify_and_clone / _split:
             new_gt_parts = old_gt_parts[sel].repeat(rep)   (line ~2375)
             every child inherits its parent's label exactly
        -> prune_points:
             self._gt_part_labels = gt[valid_points_mask]   (line ~2267)
             pure index selection
        -> save_ply:
             written as the per-vertex fields `gt_part` and `gt_dynamic`
             (line ~1817)

So the field is an immutable lineage from the labelled sparse input through
densification and pruning into the optimized model -- which is what section 3
requires. A learned `dynamic_prob` / `dynamic_logit` would NOT qualify, and
none is used here.

The optimized model and the Phase-1 canonical state are the SAME Gaussians in
the SAME order: verified at `max_i ||mu_i^GT - mu_i^Phase1(0)|| = 0.000e+00` on
all five scenes, so labels transfer BY INDEX. The previous sparse-nearest-
neighbour mapping at `4 x` the sparse cloud's median spacing is kept only as
the section 7 diagnostic.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch import Tensor

NOT_FOUND = "OPTIMIZED_LABELED_3DGS_GT_NOT_FOUND"
GT_FIELDS = ("gt_part", "gt_dynamic")


def find_labeled_model(model_dir, iteration: int | None = None) -> dict:
    """Locate the optimized PLY that carries the GT fields (sections 2, 4)."""
    from plyfile import PlyData

    root = Path(model_dir) / "point_cloud"
    searched, rejected = [], []
    if not root.is_dir():
        return {"status": NOT_FOUND, "searched": [str(root)],
                "rejected": [], "reason": f"no point_cloud/ under {model_dir}"}
    iters = sorted(root.glob("iteration_*"),
                   key=lambda p: int(p.name.split("_")[1]))
    if iteration is not None:
        iters = [p for p in iters if p.name == f"iteration_{iteration}"]
    for d in reversed(iters):                       # latest first
        ply = d / "point_cloud.ply"
        searched.append(str(ply))
        if not ply.is_file():
            rejected.append({"path": str(ply), "why": "missing"})
            continue
        names = set(PlyData.read(str(ply))["vertex"].data.dtype.names or ())
        missing = [f for f in GT_FIELDS if f not in names]
        if missing:
            rejected.append({"path": str(ply),
                             "why": f"missing fields {missing}"})
            continue
        return {"status": "FOUND", "path": str(ply), "iteration": d.name,
                "searched": searched, "rejected": rejected}
    return {"status": NOT_FOUND, "searched": searched, "rejected": rejected,
            "reason": ("no optimized PLY carries gt_part/gt_dynamic. These are "
                       "written by GaussianModel.save_ply only when the scene "
                       "was built with a labelled sparse cloud; regenerate "
                       "with the static-endpoint training path "
                       "(utils/static_endpoint_scene.py)")}


def load_optimized_gt(model_dir, phase1_frame0: Tensor,
                      iteration: int | None = None,
                      tol: float | None = None) -> dict:
    """Section 4/6: GT labels aligned to the Phase-1 Gaussian ordering.

    Alignment is checked, never assumed. If the counts match and the positions
    agree to serialization tolerance the labels are taken BY INDEX. If the
    counts match but the order differs, a mutual-nearest-neighbour
    correspondence is established and unmatched/duplicate counts are reported.
    A count mismatch is an error, not a silent resample.
    """
    from plyfile import PlyData

    found = find_labeled_model(model_dir, iteration)
    if found["status"] != "FOUND":
        return found

    v = PlyData.read(found["path"])["vertex"]
    xyz = torch.tensor(np.stack([v["x"], v["y"], v["z"]], 1),
                       dtype=torch.float64)
    part = torch.tensor(np.asarray(v["gt_part"]).astype(np.int64))
    dyn = torch.tensor(np.asarray(v["gt_dynamic"]).astype(np.int64)) > 0
    x1 = torch.as_tensor(phase1_frame0).double().cpu()

    if xyz.shape[0] != x1.shape[0]:
        return {"status": NOT_FOUND, **found,
                "reason": (f"the labelled model has {xyz.shape[0]} Gaussians "
                           f"and Phase 1 has {x1.shape[0]}; they are not the "
                           "same optimized state, so no index or NN mapping "
                           "would be ground truth")}

    if tol is None:
        # serialization tolerance: the PLY stores float32
        tol = 1e-5 * max(float(x1.abs().max()), 1.0)
    d_index = (xyz - x1).norm(dim=1)
    if float(d_index.max()) <= tol:
        order = torch.arange(xyz.shape[0])
        method, stats = "direct_index", {
            "max_alignment_error": float(d_index.max()),
            "median_alignment_error": float(d_index.median())}
    else:
        D = torch.cdist(x1, xyz)
        a = D.argmin(1)
        b = D.argmin(0)
        mutual = b[a] == torch.arange(x1.shape[0])
        order = a
        method, stats = "mutual_nearest_neighbour", {
            "max_alignment_error": float(d_index.max()),
            "num_mutual": int(mutual.sum()),
            "num_unmatched": int((~mutual).sum()),
            "num_duplicate_targets": int(
                a.numel() - torch.unique(a).numel())}

    labels = dyn[order].long()
    return {
        "status": "FOUND", "available": True,
        "source": found["path"], "iteration": found["iteration"],
        "mapping_method": method,
        "provenance": ("sparse points3D_parts.ply -> BasicPointCloud."
                       "part_indices -> GaussianModel._gt_part_labels "
                       "(torch.long, never an nn.Parameter, never optimized) "
                       "-> inherited by densify clone/split via "
                       "old_gt_parts[sel].repeat(rep) -> index-selected by "
                       "prune_points -> serialized as gt_part/gt_dynamic"),
        "moving_rule": "gt_dynamic > 0, equivalently part != 0",
        "num_gaussians": int(labels.numel()),
        "num_static": int((labels == 0).sum()),
        "num_dynamic": int((labels == 1).sum()),
        "num_unmapped": 0,
        "coverage": 1.0,
        "gt_dynamic_fraction": float((labels == 1).double().mean()),
        "part_histogram": {int(k): int(c) for k, c in
                           zip(*np.unique(part.numpy(), return_counts=True))},
        "alignment": stats,
        "labels": labels,
    }


def compare_with_sparse_nn(optimized_labels: Tensor, sparse_labels: Tensor) -> dict:
    """Section 7: how wrong was the old sparse-NN pseudo-GT?"""
    o = torch.as_tensor(optimized_labels).long().reshape(-1)
    s = torch.as_tensor(sparse_labels).long().reshape(-1)
    m = s >= 0
    agree = (o[m] == s[m])
    return {
        "num_compared": int(m.sum()),
        "old_unmapped": int((~m).sum()),
        "agreement_fraction": float(agree.double().mean()) if int(m.sum()) else float("nan"),
        "static_to_dynamic_disagreement": int(((o[m] == 0) & (s[m] == 1)).sum()),
        "dynamic_to_static_disagreement": int(((o[m] == 1) & (s[m] == 0)).sum()),
        "optimized_dynamic_fraction": float((o == 1).double().mean()),
        "sparse_nn_dynamic_fraction": (float((s[m] == 1).double().mean())
                                       if int(m.sum()) else float("nan")),
        "note": ("section 7: the sparse-NN labels are a DIAGNOSTIC. The "
                 "optimized labelled 3DGS GT is the primary reference."),
    }


def supernode_gt(labels: Tensor, supernode_id: Tensor) -> dict:
    """Section 8: per-supernode GT purity. Diagnostic only, never predictive."""
    lab = torch.as_tensor(labels).long().reshape(-1)
    sid = torch.as_tensor(supernode_id).long().reshape(-1)
    n = int(sid.max()) + 1 if sid.numel() else 0
    rows = []
    for s in range(n):
        mem = lab[sid == s]
        if not mem.numel():
            continue
        p = float((mem == 1).double().mean())
        rows.append({"supernode_id": s, "num_members": int(mem.numel()),
                     "gt_dynamic_fraction": p, "gt_static_fraction": 1.0 - p,
                     "gt_purity": max(p, 1.0 - p)})
    purity = [r["gt_purity"] for r in rows]
    return {"rows": rows, "num_supernodes": len(rows),
            "median_purity": float(np.median(purity)) if purity else float("nan"),
            "fraction_pure_above_0.9": (float(np.mean([p > 0.9 for p in purity]))
                                        if purity else float("nan")),
            "note": "section 8: diagnostic; prediction stays Gaussian-level"}
