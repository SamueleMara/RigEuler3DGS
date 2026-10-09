#!/usr/bin/env python3
"""Attach dataset GT part labels to a trained static 3DGS point cloud."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from plyfile import PlyData, PlyElement


def _default_static_ply(model_dir: Path, iteration: int) -> Path:
    return model_dir / "point_cloud" / f"iteration_{int(iteration)}" / "point_cloud.ply"


def _read_xyz(vertex) -> torch.Tensor:
    return torch.from_numpy(
        np.stack((vertex["x"], vertex["y"], vertex["z"]), axis=-1).astype(np.float32, copy=True)
    )


def _nearest_parts(gaussian_xyz: torch.Tensor, gt_xyz: torch.Tensor, gt_part: torch.Tensor, chunk: int) -> dict:
    parts = torch.unique(gt_part, sorted=True)
    nearest_distance = []
    nearest_part = []
    nearest_index = []
    part_distances = []
    for start in range(0, gaussian_xyz.shape[0], int(chunk)):
        query = gaussian_xyz[start : start + int(chunk)]
        d = torch.cdist(query, gt_xyz)
        dist, idx = d.min(dim=1)
        nearest_distance.append(dist)
        nearest_index.append(idx)
        nearest_part.append(gt_part.index_select(0, idx))
        per_part = []
        for part in parts:
            mask = gt_part == part
            per_part.append(d[:, mask].min(dim=1).values)
        part_distances.append(torch.stack(per_part, dim=1))
    nearest_distance_t = torch.cat(nearest_distance)
    nearest_part_t = torch.cat(nearest_part)
    nearest_index_t = torch.cat(nearest_index)
    part_distances_t = torch.cat(part_distances)
    best, order = torch.sort(part_distances_t, dim=1)
    margin = best[:, 1] - best[:, 0] if best.shape[1] > 1 else torch.full_like(best[:, 0], float("inf"))
    return {
        "parts": parts,
        "part_distances": part_distances_t,
        "nearest_distance": nearest_distance_t,
        "nearest_part": nearest_part_t,
        "nearest_index": nearest_index_t,
        "nearest_part_margin": margin,
        "nearest_part_by_distance": parts.index_select(0, order[:, 0]),
    }


def _weighted_vote_parts(
    gaussian_xyz: torch.Tensor,
    gt_xyz: torch.Tensor,
    gt_part: torch.Tensor,
    *,
    knn: int,
    sigma: float,
    chunk: int,
) -> dict:
    parts = torch.unique(gt_part, sorted=True)
    part_index = torch.searchsorted(parts, gt_part)
    k = min(max(1, int(knn)), int(gt_xyz.shape[0]))
    sigma_t = max(float(sigma), 1.0e-6)
    scores = []
    nearest_distance = []
    nearest_index = []
    for start in range(0, gaussian_xyz.shape[0], int(chunk)):
        query = gaussian_xyz[start : start + int(chunk)]
        d = torch.cdist(query, gt_xyz)
        vals, idx = torch.topk(d, k=k, dim=1, largest=False)
        weights = torch.exp(-(vals.square()) / (2.0 * sigma_t * sigma_t))
        local_scores = torch.zeros((query.shape[0], parts.numel()), dtype=torch.float32)
        local_scores.scatter_add_(1, part_index.index_select(0, idx.reshape(-1)).reshape_as(idx), weights)
        scores.append(local_scores)
        nearest_distance.append(vals[:, 0])
        nearest_index.append(idx[:, 0])
    scores_t = torch.cat(scores, dim=0)
    nearest_distance_t = torch.cat(nearest_distance)
    nearest_index_t = torch.cat(nearest_index)
    top = torch.topk(scores_t, k=min(2, scores_t.shape[1]), dim=1)
    if scores_t.shape[1] > 1:
        vote_margin = (top.values[:, 0] - top.values[:, 1]) / scores_t.sum(dim=1).clamp_min(1.0e-8)
    else:
        vote_margin = torch.ones(scores_t.shape[0])
    return {
        "parts": parts,
        "scores": scores_t,
        "vote_labels": parts.index_select(0, scores_t.argmax(dim=1)),
        "vote_margin": vote_margin,
        "nearest_distance": nearest_distance_t,
        "nearest_index": nearest_index_t,
        "nearest_part": gt_part.index_select(0, nearest_index_t),
    }


def _gaussian_knn(xyz: torch.Tensor, *, knn: int, chunk: int) -> torch.Tensor:
    k = min(max(1, int(knn)), max(1, int(xyz.shape[0]) - 1))
    idxs = []
    for start in range(0, xyz.shape[0], int(chunk)):
        d = torch.cdist(xyz[start : start + int(chunk)], xyz)
        _, idx = torch.topk(d, k=k + 1, dim=1, largest=False)
        idxs.append(idx[:, 1:])
    return torch.cat(idxs, dim=0)


def _smooth_vote_labels(
    scores: torch.Tensor,
    nn_idx: torch.Tensor,
    *,
    iterations: int,
    smooth_weight: float,
) -> torch.Tensor:
    labels = scores.argmax(dim=1).long()
    if int(iterations) <= 0 or float(smooth_weight) <= 0.0 or nn_idx.numel() == 0:
        return labels
    prob = scores / scores.sum(dim=1, keepdim=True).clamp_min(1.0e-8)
    unary = -torch.log(prob.clamp_min(1.0e-8))
    num_parts = scores.shape[1]
    for _ in range(int(iterations)):
        neigh = labels.index_select(0, nn_idx.reshape(-1)).reshape_as(nn_idx)
        counts = F.one_hot(neigh, num_classes=num_parts).float().mean(dim=1)
        labels = (unary + float(smooth_weight) * (1.0 - counts)).argmin(dim=1).long()
    return labels


def _cleanup_small_components(labels: torch.Tensor, nn_idx: torch.Tensor, *, min_size: int) -> torch.Tensor:
    min_size = int(min_size)
    if min_size <= 1 or labels.numel() == 0 or nn_idx.numel() == 0:
        return labels
    out = labels.clone()
    n = int(labels.numel())
    neighbors = [set(row.tolist()) for row in nn_idx.cpu()]
    for i, row in enumerate(nn_idx.cpu().tolist()):
        for nb in row:
            neighbors[int(nb)].add(i)
    neighbors = [sorted(item) for item in neighbors]
    visited = torch.zeros(n, dtype=torch.bool)
    for seed in range(n):
        if bool(visited[seed]):
            continue
        label = int(out[seed])
        stack = [seed]
        component = []
        visited[seed] = True
        while stack:
            item = stack.pop()
            component.append(item)
            for nb in neighbors[item]:
                if not bool(visited[nb]) and int(out[nb]) == label:
                    visited[nb] = True
                    stack.append(nb)
        if len(component) >= min_size:
            continue
        component_set = set(component)
        boundary = []
        for item in component:
            boundary.extend(int(out[nb]) for nb in neighbors[item] if nb not in component_set)
        if not boundary:
            continue
        counts = torch.bincount(torch.tensor(boundary, dtype=torch.long), minlength=int(out.max()) + 1)
        replacement = int(counts.argmax())
        if replacement != label:
            out[torch.tensor(component, dtype=torch.long)] = replacement
    return out


def _fill_local_holes(
    labels: torch.Tensor,
    nn_idx: torch.Tensor,
    *,
    min_opposite_fraction: float,
    iterations: int,
) -> torch.Tensor:
    out = labels.clone()
    if int(iterations) <= 0 or nn_idx.numel() == 0:
        return out
    num_parts = int(out.max()) + 1
    needed = max(1, int(round(float(min_opposite_fraction) * nn_idx.shape[1])))
    for _ in range(int(iterations)):
        neigh = out.index_select(0, nn_idx.reshape(-1)).reshape_as(nn_idx)
        counts = F.one_hot(neigh, num_classes=num_parts).sum(dim=1)
        best = counts.argmax(dim=1)
        support = counts.max(dim=1).values
        flip = (best != out) & (support >= needed)
        if not bool(flip.any()):
            break
        out = torch.where(flip, best.to(out.dtype), out)
    return out


def _copy_vertex_with_gt(
    vertex,
    labels: torch.Tensor,
    moving: torch.Tensor,
    distance: torch.Tensor,
    margin: torch.Tensor,
    raw_labels: torch.Tensor,
    changed: torch.Tensor,
):
    old_names = set(vertex.dtype.names or ())
    extra = [
        ("gt_part", "i4"),
        ("gt_dynamic", "u1"),
        ("gt_part_nn_distance", "f4"),
        ("gt_part_margin", "f4"),
        ("gt_part_raw_nn", "i4"),
        ("gt_part_changed_by_graph", "u1"),
    ]
    dtype = list(vertex.dtype.descr) + [item for item in extra if item[0] not in old_names]
    out = np.empty(vertex.shape[0], dtype=dtype)
    for name in vertex.dtype.names:
        out[name] = vertex[name]
    out["gt_part"] = labels.numpy().astype(np.int32)
    out["gt_dynamic"] = moving.numpy().astype(np.uint8)
    out["gt_part_nn_distance"] = distance.numpy().astype(np.float32)
    out["gt_part_margin"] = margin.numpy().astype(np.float32)
    out["gt_part_raw_nn"] = raw_labels.numpy().astype(np.int32)
    out["gt_part_changed_by_graph"] = changed.numpy().astype(np.uint8)
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root-scene", required=True)
    parser.add_argument("--static-model-dir", required=True)
    parser.add_argument("--iteration", type=int, default=30000)
    parser.add_argument("--static-ply", default="")
    parser.add_argument("--gt-ply", default="")
    parser.add_argument("--moving-part", type=int, default=1)
    parser.add_argument("--output-ply", default="")
    parser.add_argument("--sidecar", default="")
    parser.add_argument("--summary", default="")
    parser.add_argument("--trusted-distance", type=float, default=0.03)
    parser.add_argument("--vote-knn", type=int, default=12)
    parser.add_argument("--vote-sigma", type=float, default=0.04)
    parser.add_argument("--smooth-knn", type=int, default=16)
    parser.add_argument("--smooth-iters", type=int, default=8)
    parser.add_argument("--smooth-weight", type=float, default=0.45)
    parser.add_argument("--cleanup-min-size", type=int, default=32)
    parser.add_argument("--hole-fill-iters", type=int, default=2)
    parser.add_argument("--hole-fill-fraction", type=float, default=0.75)
    parser.add_argument("--chunk", type=int, default=1024)
    args = parser.parse_args()

    root = Path(args.root_scene).expanduser().resolve()
    model_dir = Path(args.static_model_dir).expanduser().resolve()
    static_ply = Path(args.static_ply).expanduser().resolve() if args.static_ply else _default_static_ply(model_dir, args.iteration)
    gt_ply = Path(args.gt_ply).expanduser().resolve() if args.gt_ply else root / "pointcloud" / "points3D_parts.ply"
    if not static_ply.is_file():
        raise FileNotFoundError(f"Missing static 3DGS PLY: {static_ply}")
    if not gt_ply.is_file():
        raise FileNotFoundError(f"Missing GT part PLY: {gt_ply}")

    static_data = PlyData.read(str(static_ply))
    gt_data = PlyData.read(str(gt_ply))
    static_vertex = static_data["vertex"].data
    gt_vertex = gt_data["vertex"].data
    if "part" not in (gt_vertex.dtype.names or ()):
        raise ValueError(f"GT PLY has no 'part' vertex property: {gt_ply}")

    gaussian_xyz = _read_xyz(static_vertex).float()
    gt_xyz = _read_xyz(gt_vertex).float()
    gt_part = torch.from_numpy(np.asarray(gt_vertex["part"], dtype=np.int64).copy())
    raw_match = _nearest_parts(gaussian_xyz, gt_xyz, gt_part, max(1, int(args.chunk)))
    vote = _weighted_vote_parts(
        gaussian_xyz,
        gt_xyz,
        gt_part,
        knn=int(args.vote_knn),
        sigma=float(args.vote_sigma),
        chunk=max(1, int(args.chunk)),
    )
    nn_idx = _gaussian_knn(
        gaussian_xyz,
        knn=int(args.smooth_knn),
        chunk=max(1, int(args.chunk)),
    )
    smoothed_index = _smooth_vote_labels(
        vote["scores"],
        nn_idx,
        iterations=int(args.smooth_iters),
        smooth_weight=float(args.smooth_weight),
    )
    cleaned_index = _cleanup_small_components(
        smoothed_index,
        nn_idx,
        min_size=int(args.cleanup_min_size),
    )
    hole_filled_index = _fill_local_holes(
        cleaned_index,
        nn_idx,
        min_opposite_fraction=float(args.hole_fill_fraction),
        iterations=int(args.hole_fill_iters),
    )
    final_index = _cleanup_small_components(
        hole_filled_index,
        nn_idx,
        min_size=int(args.cleanup_min_size),
    )
    labels = vote["parts"].index_select(0, final_index).long()
    moving = labels == int(args.moving_part)
    trusted = vote["nearest_distance"] <= float(args.trusted_distance)
    raw_labels = raw_match["nearest_part"].long()
    changed = labels != raw_labels

    out_ply = Path(args.output_ply).expanduser().resolve() if args.output_ply else static_ply.with_name("point_cloud_gt_parts.ply")
    sidecar = Path(args.sidecar).expanduser().resolve() if args.sidecar else static_ply.parent / "gt_parts.pt"
    summary_path = Path(args.summary).expanduser().resolve() if args.summary else static_ply.parent / "gt_parts_summary.json"
    out_ply.parent.mkdir(parents=True, exist_ok=True)
    sidecar.parent.mkdir(parents=True, exist_ok=True)

    annotated_vertex = _copy_vertex_with_gt(
        static_vertex,
        labels.cpu(),
        moving.cpu(),
        vote["nearest_distance"].cpu(),
        vote["vote_margin"].cpu(),
        raw_labels.cpu(),
        changed.cpu(),
    )
    PlyData([PlyElement.describe(annotated_vertex, "vertex")], text=static_data.text).write(str(out_ply))

    source_counts = {str(int(p)): int((gt_part == int(p)).sum()) for p in torch.unique(gt_part, sorted=True)}
    gaussian_counts = {str(int(p)): int((labels == int(p)).sum()) for p in torch.unique(labels, sorted=True)}
    raw_counts = {str(int(p)): int((raw_labels == int(p)).sum()) for p in torch.unique(raw_labels, sorted=True)}
    vote_counts = {str(int(p)): int((vote["vote_labels"] == int(p)).sum()) for p in torch.unique(vote["vote_labels"], sorted=True)}
    summary = {
        "schema_version": 1,
        "role": "debug_evaluation_gt_port_only_not_algorithm_input",
        "static_ply": str(static_ply),
        "gt_ply": str(gt_ply),
        "output_ply": str(out_ply),
        "sidecar": str(sidecar),
        "moving_part": int(args.moving_part),
        "gaussian_count": int(labels.numel()),
        "gt_source_count": int(gt_part.numel()),
        "source_part_counts": source_counts,
        "raw_nearest_part_counts": raw_counts,
        "weighted_vote_part_counts": vote_counts,
        "ported_part_counts": gaussian_counts,
        "ported_dynamic_count": int(moving.sum()),
        "changed_from_raw_nearest_count": int(changed.sum()),
        "vote_knn": int(args.vote_knn),
        "vote_sigma": float(args.vote_sigma),
        "smooth_knn": int(args.smooth_knn),
        "smooth_iters": int(args.smooth_iters),
        "smooth_weight": float(args.smooth_weight),
        "cleanup_min_size": int(args.cleanup_min_size),
        "hole_fill_iters": int(args.hole_fill_iters),
        "hole_fill_fraction": float(args.hole_fill_fraction),
        "trusted_distance": float(args.trusted_distance),
        "trusted_count": int(trusted.sum()),
        "nearest_distance": {
            "median": float(vote["nearest_distance"].median()),
            "p90": float(vote["nearest_distance"].quantile(0.9)),
            "p99": float(vote["nearest_distance"].quantile(0.99)),
            "max": float(vote["nearest_distance"].max()),
        },
        "vote_part_margin": {
            "median": float(vote["vote_margin"].median()),
            "p10": float(vote["vote_margin"].quantile(0.1)),
            "min": float(vote["vote_margin"].min()),
        },
        "ground_truth_used_as_algorithm_input": False,
    }
    torch.save(
        {
            **summary,
            "gt_part_labels": labels.cpu(),
            "gt_moving_mask": moving.cpu(),
            "gt_trusted_mask": trusted.cpu(),
            "gt_raw_nearest_part_labels": raw_labels.cpu(),
            "gt_vote_part_labels": vote["vote_labels"].long().cpu(),
            "gt_smoothed_part_indices": smoothed_index.cpu(),
            "gt_cleaned_part_indices": cleaned_index.cpu(),
            "gt_hole_filled_part_indices": hole_filled_index.cpu(),
            "gt_final_part_indices": final_index.cpu(),
            "gt_changed_from_raw_nearest": changed.cpu(),
            "gt_nearest_index": vote["nearest_index"].cpu(),
            "gt_nearest_distance": vote["nearest_distance"].cpu(),
            "gt_part_vote_margin": vote["vote_margin"].cpu(),
            "gt_part_scores": vote["scores"].cpu(),
        },
        sidecar,
    )
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
