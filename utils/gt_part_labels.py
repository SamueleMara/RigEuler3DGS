"""EVALUATION ONLY: static / dynamic ground-truth part labels for Gaussians.

The synthetic SAPIEN inputs ship one labelled surface cloud per endpoint, each
with an integer `part` field (0 = the static base, != 0 = the moving link; the
repository convention of utils/static_endpoint_scene.py):

    <scene>/pointcloud/points3D_parts.ply         initial endpoint (colmap/)
    <scene>/pointcloud/points3D_parts_final.ply   final endpoint (colmap_final/)

Real captures have no such file, and then nothing is labelled -- never a guess.

Two lineages carry the labels to an endpoint model:
  * static 3DGS training (train.py): the COLMAP points ARE the labelled cloud,
    labels attach at initialization (scene/dataset_readers.py) and clone / split
    / prune inherit them exactly (scene/gaussian_model.py);
  * the fixed-count Phase-0 bootstrap (tools/train_tracking_bootstrap.py): the
    Gaussians are seeded from back-projected endpoint depth, not from the
    labelled cloud, so each SEED takes the part of its nearest labelled sample
    of the SAME endpoint. Ids never change during the refinement, so the seed's
    label is the Gaussian's label. A seed farther than `factor` x the cloud's
    median spacing from every labelled sample is UNLABELLED (-1) and excluded
    from every metric.

Nothing here is ever read by an estimator; the labels are written next to the
model only for the evaluation stages.
"""
from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np

LABEL_FILES = {"initial": "points3D_parts.ply", "final": "points3D_parts_final.ply"}
UNLABELLED = -1
DEFAULT_FACTOR = 4.0


def _sha(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def labelled_cloud(source, endpoint: str) -> dict | None:
    """The endpoint's labelled cloud {path, sha256, xyz [M,3], part [M]}, or None."""
    from plyfile import PlyData
    if endpoint not in LABEL_FILES:
        raise ValueError(f"endpoint must be one of {tuple(LABEL_FILES)}, not {endpoint!r}")
    p = Path(source) / "pointcloud" / LABEL_FILES[endpoint]
    if not p.is_file():
        return None
    v = PlyData.read(str(p))["vertex"]
    if "part" not in (v.data.dtype.names or ()):
        return None
    return {"path": str(p), "sha256": _sha(p), "endpoint": endpoint,
            "xyz": np.stack([v["x"], v["y"], v["z"]], 1).astype(np.float64),
            "part": np.asarray(v["part"]).astype(np.int64)}


def median_spacing(xyz: np.ndarray) -> float:
    from scipy.spatial import cKDTree
    if len(xyz) < 2:
        return float("inf")
    d, _ = cKDTree(xyz).query(xyz, k=2)
    return float(np.median(d[:, 1]))


def nn_labels(xyz: np.ndarray, cloud: dict, factor: float = DEFAULT_FACTOR) -> dict:
    """Part label of each row from its nearest labelled sample; -1 beyond factor x spacing."""
    from scipy.spatial import cKDTree
    xyz = np.asarray(xyz, np.float64)
    max_dist = factor * median_spacing(cloud["xyz"])
    d, i = cKDTree(cloud["xyz"]).query(xyz, k=1)
    part = cloud["part"][i].copy()
    part[d > max_dist] = UNLABELLED
    return {"part": part, "nn_distance": d, "max_dist": float(max_dist), "factor": float(factor)}


def dynamic_from_part(part: np.ndarray) -> np.ndarray:
    """1 moving, 0 static, -1 unlabelled."""
    part = np.asarray(part, np.int64)
    return np.where(part < 0, UNLABELLED, (part != 0).astype(np.int64))


def summary(part: np.ndarray, nn_distance: np.ndarray | None = None) -> dict:
    dyn = dynamic_from_part(part)
    out = {"num": int(part.size), "num_static": int((dyn == 0).sum()), "num_dynamic": int((dyn == 1).sum()),
           "num_unlabelled": int((dyn < 0).sum()),
           "coverage": float((dyn >= 0).mean()) if part.size else 0.0,
           "part_histogram": {int(k): int(c) for k, c in zip(*np.unique(part, return_counts=True))}}
    if nn_distance is not None and nn_distance.size:
        out["nn_distance_m_q"] = [float(x) for x in np.quantile(nn_distance, [0.5, 0.9, 0.99])]
    return out
