#!/usr/bin/env python3
"""Extract and diagnose high-semantic Gaussian regions using XYZ + DINO.

Reads ``gaussian_tracks.pt`` and writes:
  - component_diagnostic.json
  - component_metrics.csv
  - component_dino_cosine.csv
  - component_labels.pt
  - component_prototypes.pt
  - semantic_dino_components.ply
  - semantic_dino_components.html
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch
import torch.nn.functional as F

try:
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components
    from scipy.spatial import cKDTree
except ImportError as exc:
    raise SystemExit("Missing scipy. Install with: conda install scipy") from exc

EPS = 1e-8


def require_tensor(data: Any, name: str, ndim: Optional[int] = None) -> torch.Tensor:
    if not torch.is_tensor(data):
        raise KeyError(f"gaussian_tracks.pt is missing tensor '{name}'")
    tensor = data.detach().float().cpu()
    if ndim is not None and tensor.ndim != ndim:
        raise ValueError(f"{name} must be {ndim}D, got {tuple(tensor.shape)}")
    return tensor


def json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if torch.is_tensor(value):
        return json_safe(value.detach().cpu().tolist())
    if isinstance(value, np.ndarray):
        return json_safe(value.tolist())
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, (np.floating, float)):
        number = float(value)
        return number if math.isfinite(number) else None
    return value


def query_knn(xyz: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
    tree = cKDTree(xyz)
    k_query = min(len(xyz), max(2, int(k) + 1))
    try:
        distance, index = tree.query(xyz, k=k_query, workers=-1)
    except TypeError:
        distance, index = tree.query(xyz, k=k_query)
    if distance.ndim == 1:
        distance = distance[:, None]
        index = index[:, None]
    return distance[:, 1:].astype(np.float32), index[:, 1:].astype(np.int64)


def build_graph_components(
    xyz: torch.Tensor,
    dino: torch.Tensor,
    semantic: torch.Tensor,
    knn: int,
    radius_scale: float,
    dino_threshold: float,
) -> tuple[np.ndarray, dict[str, Any]]:
    xyz_np = xyz.numpy().astype(np.float32, copy=False)
    dino_np = F.normalize(dino, dim=-1, eps=EPS).numpy().astype(np.float32, copy=False)
    sem_np = semantic.numpy().astype(np.float32, copy=False)

    distances, neighbors = query_knn(xyz_np, knn)
    nearest = distances[:, 0]
    nearest = nearest[np.isfinite(nearest) & (nearest > 1e-10)]
    if nearest.size:
        median_nn = float(np.median(nearest))
    else:
        diagonal = float(np.linalg.norm(xyz_np.max(0) - xyz_np.min(0)))
        median_nn = max(diagonal / 1000.0, 1e-4)
    radius = max(median_nn * float(radius_scale), median_nn * 1.01, 1e-5)

    n, k_eff = neighbors.shape
    rows = np.repeat(np.arange(n, dtype=np.int64), k_eff)
    cols = neighbors.reshape(-1)
    dist = distances.reshape(-1)
    valid = np.isfinite(dist) & (dist > 0) & (dist <= radius)
    rows, cols, dist = rows[valid], cols[valid], dist[valid]

    cosine = np.einsum("ij,ij->i", dino_np[rows], dino_np[cols])
    cosine = np.clip(cosine, -1.0, 1.0)

    p_i = np.clip(sem_np[rows], 1e-5, 1.0 - 1e-5)
    p_j = np.clip(sem_np[cols], 1e-5, 1.0 - 1e-5)
    logit_i = np.log(p_i / (1.0 - p_i))
    logit_j = np.log(p_j / (1.0 - p_j))
    geometry_affinity = np.exp(-0.5 * (dist / radius) ** 2)
    feature_affinity = np.clip((cosine + 1.0) * 0.5, 0.0, 1.0)
    semantic_confidence = np.sqrt(p_i * p_j)
    semantic_similarity = np.exp(-np.abs(logit_i - logit_j) / 4.0)
    edge_weight = (
        geometry_affinity
        * feature_affinity
        * semantic_confidence
        * semantic_similarity
    )

    accepted = (cosine >= float(dino_threshold)) & (edge_weight >= 0.05)
    rows, cols = rows[accepted], cols[accepted]
    directed = coo_matrix(
        (np.ones(rows.shape[0], dtype=np.uint8), (rows, cols)),
        shape=(n, n),
    ).tocsr()
    adjacency = directed.maximum(directed.transpose())
    count, labels = connected_components(adjacency, directed=False, return_labels=True)
    return labels.astype(np.int64), {
        "candidate_points": n,
        "effective_knn": k_eff,
        "median_nearest_neighbor_distance": median_nn,
        "spatial_edge_radius": radius,
        "dino_cosine_threshold": float(dino_threshold),
        "accepted_directed_edges": int(rows.shape[0]),
        "raw_connected_components": int(count),
    }


def integrated_proposal_stats(
    velocity: Optional[torch.Tensor],
    times: Optional[torch.Tensor],
    n: int,
) -> tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
    if velocity is None or velocity.ndim != 3 or velocity.shape[1] != n:
        return None, None, None
    velocity = velocity.float().cpu()
    speed = torch.linalg.norm(velocity, dim=-1)
    mean_speed = speed.mean(0)
    max_speed = speed.max(0).values
    if times is None or times.numel() != velocity.shape[0] or velocity.shape[0] < 2:
        return None, mean_speed, max_speed
    times = times.float().cpu().reshape(-1)
    cumulative = torch.zeros((n, 3), dtype=torch.float32)
    max_displacement = torch.zeros(n, dtype=torch.float32)
    for frame in range(1, velocity.shape[0]):
        dt = float(times[frame] - times[frame - 1])
        if not math.isfinite(dt) or dt <= 0:
            continue
        cumulative += 0.5 * (velocity[frame - 1] + velocity[frame]) * dt
        max_displacement = torch.maximum(max_displacement, torch.linalg.norm(cumulative, dim=-1))
    return max_displacement, mean_speed, max_speed


def best_fit_se3(
    canonical: torch.Tensor,
    positions: Optional[torch.Tensor],
    max_points: int = 4096,
    seed: int = 12345,
) -> dict[str, Any]:
    empty = {
        "valid": False,
        "fit_point_count": 0,
        "rmse_mean": None,
        "rmse_median": None,
        "rmse_p95": None,
        "rmse_max": None,
        "normalized_rmse_mean": None,
        "translation_norm_max": None,
        "rotation_angle_deg_max": None,
    }
    if positions is None or positions.ndim != 3 or canonical.shape[0] < 3:
        return empty

    if canonical.shape[0] > max_points:
        generator = torch.Generator().manual_seed(seed)
        selected = torch.randperm(canonical.shape[0], generator=generator)[:max_points]
        x = canonical[selected].float().cpu()
        y = positions[:, selected].float().cpu()
    else:
        x = canonical.float().cpu()
        y = positions.float().cpu()

    x_mean = x.mean(0, keepdim=True)
    x_centered = x - x_mean
    y_mean = y.mean(1, keepdim=True)
    y_centered = y - y_mean
    covariance = torch.einsum("ni,tnj->tij", x_centered, y_centered) / x.shape[0]
    u, _, vh = torch.linalg.svd(covariance)
    rotation = u @ vh
    reflected = torch.linalg.det(rotation) < 0
    if reflected.any():
        u = u.clone()
        u[reflected, :, -1] *= -1
        rotation = u @ vh

    prediction = torch.einsum("ni,tij->tnj", x_centered, rotation)
    residual = y_centered - prediction
    rmse = torch.sqrt(torch.mean(torch.sum(residual.square(), dim=-1), dim=-1))
    rotated_mean = torch.einsum("ni,tij->tnj", x_mean, rotation)[:, 0]
    translation = y_mean[:, 0] - rotated_mean
    trace = rotation[:, 0, 0] + rotation[:, 1, 1] + rotation[:, 2, 2]
    angle = torch.acos(((trace - 1) * 0.5).clamp(-1, 1)) * (180 / math.pi)
    radius = torch.sqrt(torch.mean(torch.sum(x_centered.square(), dim=-1))).clamp_min(EPS)
    return {
        "valid": True,
        "fit_point_count": int(x.shape[0]),
        "rmse_mean": float(rmse.mean()),
        "rmse_median": float(torch.quantile(rmse, 0.50)),
        "rmse_p95": float(torch.quantile(rmse, 0.95)),
        "rmse_max": float(rmse.max()),
        "normalized_rmse_mean": float(rmse.mean() / radius),
        "translation_norm_max": float(torch.linalg.norm(translation, dim=-1).max()),
        "rotation_angle_deg_max": float(angle.max()),
    }


def component_nn_median(xyz: torch.Tensor) -> float:
    if xyz.shape[0] < 2:
        return float("nan")
    tree = cKDTree(xyz.numpy())
    try:
        distance, _ = tree.query(xyz.numpy(), k=2, workers=-1)
    except TypeError:
        distance, _ = tree.query(xyz.numpy(), k=2)
    return float(np.median(distance[:, 1]))


def component_metrics(
    component_id: int,
    ids: torch.Tensor,
    xyz: torch.Tensor,
    semantic: torch.Tensor,
    dino: torch.Tensor,
    positions: Optional[torch.Tensor],
    actual_disp: Optional[torch.Tensor],
    proposal_disp: Optional[torch.Tensor],
    proposal_mean_speed: Optional[torch.Tensor],
    proposal_max_speed: Optional[torch.Tensor],
) -> tuple[dict[str, Any], torch.Tensor, torch.Tensor]:
    points = xyz[ids]
    sem = semantic[ids]
    features = dino[ids]
    prototype = F.normalize(
        (features * sem.clamp_min(EPS)[:, None]).sum(0, keepdim=True),
        dim=-1,
        eps=EPS,
    )[0]
    cosine = (features * prototype[None]).sum(-1).clamp(-1, 1)
    centroid = points.mean(0)
    bbox_min, bbox_max = points.min(0).values, points.max(0).values
    bbox = bbox_max - bbox_min
    radius = torch.linalg.norm(points - centroid, dim=-1)
    output: dict[str, Any] = {
        "component_id": component_id,
        "point_count": int(ids.numel()),
        "semantic_mean": float(sem.mean()),
        "semantic_min": float(sem.min()),
        "semantic_median": float(torch.quantile(sem, 0.50)),
        "centroid": centroid.tolist(),
        "bbox_min": bbox_min.tolist(),
        "bbox_max": bbox_max.tolist(),
        "bbox_dimensions": bbox.tolist(),
        "bbox_diagonal": float(torch.linalg.norm(bbox)),
        "bbox_volume": float(torch.prod(bbox.clamp_min(1e-6))),
        "rms_radius": float(torch.sqrt(torch.mean(radius.square()))),
        "p95_radius": float(torch.quantile(radius, 0.95)),
        "median_local_neighbor_distance": component_nn_median(points),
        "dino_prototype": prototype.tolist(),
        "dino_cosine_to_prototype_mean": float(cosine.mean()),
        "dino_cosine_to_prototype_median": float(torch.quantile(cosine, 0.50)),
        "dino_cosine_to_prototype_p10": float(torch.quantile(cosine, 0.10)),
        "dino_cosine_to_prototype_min": float(cosine.min()),
        "best_local_se3": best_fit_se3(
            points,
            positions[:, ids] if positions is not None else None,
            seed=12345 + component_id,
        ),
    }
    for name, values in (
        ("actual_max_displacement", actual_disp),
        ("proposal_max_displacement", proposal_disp),
    ):
        if values is not None:
            selected = values[ids]
            output[f"{name}_mean"] = float(selected.mean())
            output[f"{name}_median"] = float(torch.quantile(selected, 0.50))
            output[f"{name}_max"] = float(selected.max())
    if proposal_mean_speed is not None:
        output["proposal_speed_mean"] = float(proposal_mean_speed[ids].mean())
    if proposal_max_speed is not None:
        output["proposal_speed_max"] = float(proposal_max_speed[ids].max())
    return output, prototype, cosine


def normalize(values: list[float]) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    if len(array) == 0:
        return array
    lo, hi = float(array.min()), float(array.max())
    return np.ones_like(array) if hi - lo < 1e-12 else (array - lo) / (hi - lo)


def rank_components(components: list[dict[str, Any]]) -> None:
    if not components:
        return
    size = normalize([math.log1p(c["point_count"]) for c in components])
    compactness = normalize([
        math.log1p(c["point_count"] / max(c["bbox_diagonal"] ** 3, 1e-12))
        for c in components
    ])
    dino = np.clip(
        (np.asarray([c["dino_cosine_to_prototype_mean"] for c in components]) + 1) * 0.5,
        0,
        1,
    )
    semantic = np.asarray([c["semantic_mean"] for c in components])
    scores = 0.40 * size + 0.25 * compactness + 0.25 * dino + 0.10 * semantic
    for component, score in zip(components, scores.tolist()):
        component["rank_score"] = float(score)


def write_metrics_csv(path: Path, components: list[dict[str, Any]]) -> None:
    columns = [
        "rank", "component_id", "selected_top_k", "rank_score", "point_count",
        "semantic_mean", "semantic_min", "semantic_median",
        "centroid_x", "centroid_y", "centroid_z",
        "bbox_dx", "bbox_dy", "bbox_dz", "bbox_diagonal",
        "median_local_neighbor_distance", "dino_cosine_to_prototype_mean",
        "actual_max_displacement_mean", "actual_max_displacement_max",
        "proposal_max_displacement_mean", "proposal_max_displacement_max",
        "proposal_speed_mean", "proposal_speed_max",
        "se3_rmse_mean", "se3_rmse_p95", "se3_normalized_rmse_mean",
    ]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for component in components:
            center, bbox, se3 = component["centroid"], component["bbox_dimensions"], component["best_local_se3"]
            writer.writerow({
                "rank": component["rank"],
                "component_id": component["component_id"],
                "selected_top_k": int(component["selected_top_k"]),
                "rank_score": component["rank_score"],
                "point_count": component["point_count"],
                "semantic_mean": component["semantic_mean"],
                "semantic_min": component["semantic_min"],
                "semantic_median": component["semantic_median"],
                "centroid_x": center[0], "centroid_y": center[1], "centroid_z": center[2],
                "bbox_dx": bbox[0], "bbox_dy": bbox[1], "bbox_dz": bbox[2],
                "bbox_diagonal": component["bbox_diagonal"],
                "median_local_neighbor_distance": component["median_local_neighbor_distance"],
                "dino_cosine_to_prototype_mean": component["dino_cosine_to_prototype_mean"],
                "actual_max_displacement_mean": component.get("actual_max_displacement_mean"),
                "actual_max_displacement_max": component.get("actual_max_displacement_max"),
                "proposal_max_displacement_mean": component.get("proposal_max_displacement_mean"),
                "proposal_max_displacement_max": component.get("proposal_max_displacement_max"),
                "proposal_speed_mean": component.get("proposal_speed_mean"),
                "proposal_speed_max": component.get("proposal_speed_max"),
                "se3_rmse_mean": se3.get("rmse_mean"),
                "se3_rmse_p95": se3.get("rmse_p95"),
                "se3_normalized_rmse_mean": se3.get("normalized_rmse_mean"),
            })


def write_cosine_csv(path: Path, matrix: torch.Tensor) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["component"] + [f"component_{i}" for i in range(matrix.shape[0])])
        for i, row in enumerate(matrix):
            writer.writerow([f"component_{i}"] + [f"{float(value):.8f}" for value in row])


def write_ply(
    path: Path,
    xyz: torch.Tensor,
    semantic: torch.Tensor,
    labels: torch.Tensor,
    selected: torch.Tensor,
    dino_cohesion: torch.Tensor,
    actual_disp: Optional[torch.Tensor],
    proposal_disp: Optional[torch.Tensor],
) -> None:
    try:
        from plyfile import PlyData, PlyElement
    except ImportError:
        print("[WARN] plyfile unavailable; skipping PLY")
        return
    dtype = [
        ("x", "f4"), ("y", "f4"), ("z", "f4"),
        ("semantic_prob", "f4"), ("component_id", "f4"),
        ("selected_topk", "f4"), ("dino_component_cosine", "f4"),
        ("max_displacement", "f4"), ("proposal_max_displacement", "f4"),
    ]
    vertices = np.empty(xyz.shape[0], dtype=dtype)
    points = xyz.numpy()
    vertices["x"], vertices["y"], vertices["z"] = points[:, 0], points[:, 1], points[:, 2]
    vertices["semantic_prob"] = semantic.numpy()
    vertices["component_id"] = labels.numpy().astype(np.float32)
    vertices["selected_topk"] = selected.numpy().astype(np.float32)
    vertices["dino_component_cosine"] = dino_cohesion.numpy()
    vertices["max_displacement"] = actual_disp.numpy() if actual_disp is not None else 0
    vertices["proposal_max_displacement"] = proposal_disp.numpy() if proposal_disp is not None else 0
    PlyData([PlyElement.describe(vertices, "vertex")], text=False).write(str(path))


def write_html(
    path: Path,
    xyz: torch.Tensor,
    semantic_mask: torch.Tensor,
    labels: torch.Tensor,
    top_k: int,
    max_points: int = 30000,
) -> None:
    try:
        import plotly.graph_objects as go
    except ImportError:
        print("[WARN] plotly unavailable; skipping HTML")
        return
    generator = torch.Generator().manual_seed(12345)
    figure = go.Figure()

    def sample(ids: torch.Tensor, limit: int) -> torch.Tensor:
        if ids.numel() <= limit:
            return ids
        return ids[torch.randperm(ids.numel(), generator=generator)[:limit]]

    def add(ids: torch.Tensor, name: str, size: float, opacity: float) -> None:
        if ids.numel() == 0:
            return
        points = xyz[ids]
        figure.add_trace(go.Scatter3d(
            x=points[:, 0], y=points[:, 1], z=points[:, 2], mode="markers", name=name,
            marker={"size": size, "opacity": opacity}, customdata=ids,
            hovertemplate="gaussian=%{customdata}<br>x=%{x:.4f}<br>y=%{y:.4f}<br>z=%{z:.4f}<extra></extra>",
        ))

    per_group = max(1, max_points // max(top_k + 2, 1))
    add(sample(torch.where(~semantic_mask)[0], per_group), "semantic background", 1.2, 0.12)
    add(sample(torch.where(semantic_mask & (labels < 0))[0], per_group), "candidate/noise", 1.8, 0.35)
    for component_id in range(top_k):
        add(sample(torch.where(labels == component_id)[0], per_group * 2), f"component {component_id}", 2.5, 0.9)
    figure.update_layout(
        title="Semantic + geometry + DINO components",
        scene={"aspectmode": "data"},
        margin={"l": 0, "r": 0, "t": 35, "b": 0},
    )
    figure.write_html(str(path), include_plotlyjs="cdn")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tracks", required=True, type=Path)
    parser.add_argument("--output_dir", type=Path, default=None)
    parser.add_argument("--semantic_threshold", type=float, default=-1.0)
    parser.add_argument("--top_k", type=int, default=3)
    parser.add_argument("--knn", type=int, default=24)
    parser.add_argument("--radius_scale", type=float, default=3.0)
    parser.add_argument("--dino_cosine_threshold", type=float, default=0.65)
    parser.add_argument("--min_component_points", type=int, default=20)
    args = parser.parse_args()

    tracks_path = args.tracks.expanduser().resolve()
    output_dir = (
        args.output_dir.expanduser().resolve()
        if args.output_dir is not None
        else tracks_path.parent / "semantic_dino_component_diagnostic"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    tracks = torch.load(tracks_path, map_location="cpu")

    xyz = require_tensor(tracks.get("canonical_positions"), "canonical_positions", 2)
    semantic = require_tensor(tracks.get("semantic_prob"), "semantic_prob").reshape(-1)
    dino = F.normalize(require_tensor(tracks.get("dino_feature_field"), "dino_feature_field", 2), dim=-1, eps=EPS)
    n = xyz.shape[0]
    if xyz.shape != (n, 3) or semantic.numel() != n or dino.shape[0] != n:
        raise ValueError("Inconsistent Gaussian tensor shapes")

    if args.semantic_threshold >= 0:
        semantic_threshold = float(args.semantic_threshold)
    else:
        stored = tracks.get("dynamic_foreground_threshold", 0.2)
        semantic_threshold = float(stored.item()) if torch.is_tensor(stored) else float(stored)
    semantic_mask = semantic >= semantic_threshold
    candidate_ids = torch.where(semantic_mask)[0]
    if candidate_ids.numel() < 2:
        raise RuntimeError("Too few semantic candidates")

    raw_labels, graph_stats = build_graph_components(
        xyz[candidate_ids], dino[candidate_ids], semantic[candidate_ids],
        args.knn, args.radius_scale, args.dino_cosine_threshold,
    )

    positions = tracks.get("positions")
    positions = positions.detach().float().cpu() if torch.is_tensor(positions) else None
    if positions is not None and (positions.ndim != 3 or positions.shape[1] != n):
        positions = None
    actual_disp = (
        torch.linalg.norm(positions - xyz[None], dim=-1).max(0).values
        if positions is not None else None
    )
    proposal = tracks.get("proposal_velocities", tracks.get("residual_velocities"))
    proposal = proposal.detach().float().cpu() if torch.is_tensor(proposal) else None
    times = tracks.get("times")
    times = times.detach().float().cpu() if torch.is_tensor(times) else None
    proposal_disp, proposal_mean_speed, proposal_max_speed = integrated_proposal_stats(proposal, times, n)

    raw_ids, raw_counts = np.unique(raw_labels, return_counts=True)
    retained = [int(raw_id) for raw_id, count in zip(raw_ids, raw_counts) if count >= args.min_component_points]
    components, prototypes, memberships = [], [], []
    for provisional_id, raw_id in enumerate(retained):
        local = torch.from_numpy(np.where(raw_labels == raw_id)[0]).long()
        global_ids = candidate_ids[local]
        metrics, prototype, cosine = component_metrics(
            provisional_id, global_ids, xyz, semantic, dino, positions,
            actual_disp, proposal_disp, proposal_mean_speed, proposal_max_speed,
        )
        metrics["raw_graph_component_id"] = raw_id
        components.append(metrics)
        prototypes.append(prototype)
        memberships.append((global_ids, cosine))

    rank_components(components)
    order = sorted(range(len(components)), key=lambda index: components[index]["rank_score"], reverse=True)
    components = [components[index] for index in order]
    prototypes = [prototypes[index] for index in order]
    memberships = [memberships[index] for index in order]

    labels = torch.full((n,), -1, dtype=torch.long)
    selected = torch.zeros(n, dtype=torch.bool)
    dino_cohesion = torch.full((n,), -1.0)
    for rank, (component, membership) in enumerate(zip(components, memberships)):
        ids, cosine = membership
        component["rank"] = rank
        component["component_id"] = rank
        component["selected_top_k"] = rank < args.top_k
        labels[ids] = rank
        dino_cohesion[ids] = cosine
        if rank < args.top_k:
            selected[ids] = True

    top_count = min(args.top_k, len(components))
    top_prototypes = torch.stack(prototypes[:top_count]) if top_count else torch.empty((0, dino.shape[1]))
    cosine_matrix = (top_prototypes @ top_prototypes.T).clamp(-1, 1)

    summary = {
        "tracks": str(tracks_path),
        "configuration": {
            "semantic_threshold": semantic_threshold,
            "top_k": args.top_k,
            "knn": args.knn,
            "radius_scale": args.radius_scale,
            "dino_cosine_threshold": args.dino_cosine_threshold,
            "min_component_points": args.min_component_points,
            "ranking": "0.40 point count + 0.25 compactness + 0.25 DINO cohesion + 0.10 semantic mean",
        },
        "global": {
            "num_gaussians": n,
            "semantic_candidate_count": int(candidate_ids.numel()),
            "semantic_background_count": int(n - candidate_ids.numel()),
            "retained_component_count": len(components),
            **graph_stats,
        },
        "top_component_dino_cosine_matrix": cosine_matrix,
        "components": components,
    }
    with (output_dir / "component_diagnostic.json").open("w") as handle:
        json.dump(json_safe(summary), handle, indent=2)
    write_metrics_csv(output_dir / "component_metrics.csv", components)
    write_cosine_csv(output_dir / "component_dino_cosine.csv", cosine_matrix)
    torch.save({
        "component_id_full": labels,
        "semantic_candidate_mask": semantic_mask,
        "selected_top_k_mask": selected,
        "dino_cosine_to_component_prototype": dino_cohesion,
    }, output_dir / "component_labels.pt")
    torch.save({
        "prototypes": torch.stack(prototypes) if prototypes else torch.empty((0, dino.shape[1])),
        "top_k_pairwise_cosine": cosine_matrix,
    }, output_dir / "component_prototypes.pt")
    write_ply(
        output_dir / "semantic_dino_components.ply", xyz, semantic, labels,
        selected, dino_cohesion, actual_disp, proposal_disp,
    )
    write_html(
        output_dir / "semantic_dino_components.html", xyz, semantic_mask,
        labels, top_count,
    )

    print(f"Semantic candidates: {candidate_ids.numel()}/{n}")
    print(
        f"Graph radius={graph_stats['spatial_edge_radius']:.6f}, "
        f"DINO cosine>={args.dino_cosine_threshold:.3f}, "
        f"raw components={graph_stats['raw_connected_components']}, "
        f"retained={len(components)}"
    )
    print("\nrank  points  sem_mean  dino_coh  bbox_diag  nn_median  prop_disp  se3_rmse")
    for component in components[:max(10, top_count)]:
        se3_rmse = component["best_local_se3"].get("rmse_mean")
        print(
            f"{component['rank']:>4}  {component['point_count']:>6}  "
            f"{component['semantic_mean']:>8.4f}  "
            f"{component['dino_cosine_to_prototype_mean']:>8.4f}  "
            f"{component['bbox_diagonal']:>9.4f}  "
            f"{component['median_local_neighbor_distance']:>9.5f}  "
            f"{component.get('proposal_max_displacement_mean', float('nan')):>9.4f}  "
            f"{(se3_rmse if se3_rmse is not None else float('nan')):>8.5f}"
        )
    print("\nTop-component DINO cosine matrix:")
    if top_count:
        print("          " + " ".join(f"c{i:>7}" for i in range(top_count)))
        for i, row in enumerate(cosine_matrix):
            print(f"c{i:<7} " + " ".join(f"{float(value):>8.4f}" for value in row))
    print(f"\nWrote: {output_dir}")


if __name__ == "__main__":
    main()
