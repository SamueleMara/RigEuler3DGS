"""Shared Gaussian-wise static-vs-rigid responsibility for rigid motion."""

from __future__ import annotations

import json
from pathlib import Path

import torch

from utils.dynamic_loss_utils import build_lagrangian_knn_pair_graph
from utils.html_plot_layout import (
    fixed_scene_bounds,
    left_colorbar,
    plotly_3d_config,
    plotly_3d_layout,
)


STATIC, RIGID, AMBIGUOUS = 0, 1, 3


@torch.no_grad()
def compute_gaussian_static_rigid_sequence_energies(
    canonical_points,
    mode,
    interval_speeds,
    frame_times,
    cameras_by_frame,
    correspondence_cache,
    calibrations,
    args,
):
    """Evaluate separate static and exact-rigid hypotheses for every Gaussian."""
    from utils.gaussian_sequence_assignment import (
        UNKNOWN,
        refine_assignment_with_rigid_hypotheses,
    )

    points = torch.as_tensor(canonical_points)
    count = int(points.shape[0])
    result = refine_assignment_with_rigid_hypotheses(
        points,
        mode,
        interval_speeds,
        frame_times,
        cameras_by_frame,
        correspondence_cache,
        calibrations,
        torch.full((count,), UNKNOWN, device=points.device, dtype=torch.uint8),
        torch.zeros(count, device=points.device, dtype=torch.long),
        args,
        one_shot=True,
    )
    effective_weight = result["weight"].to(points) * result["excitation"].to(points)
    result["energy_weight"] = effective_weight.sum(dim=(1, 2)).cpu()
    result["normalized_energy_margin"] = result["margin"].cpu()
    return result


@torch.no_grad()
def update_gaussian_responsibility(
    static_energy,
    rigid_energy,
    *,
    margin: float,
    temperature: float,
    energy_weight=None,
):
    """Return ``sigmoid((E_static-E_rigid-margin)/temperature)`` per Gaussian."""
    static = torch.as_tensor(static_energy).float()
    rigid = torch.as_tensor(rigid_energy, device=static.device).float()
    if static.shape != rigid.shape:
        raise ValueError("static_energy and rigid_energy must have equal shape")
    delta = static - rigid
    if energy_weight is not None:
        weight = torch.as_tensor(energy_weight, device=static.device).float()
        if weight.shape != delta.shape:
            raise ValueError("energy_weight must match the Gaussian energy shape")
        informative = weight > 0
        delta = delta / weight.clamp_min(1.0e-8)
    else:
        informative = torch.isfinite(delta)
    probability = torch.sigmoid(
        (delta - float(margin)) / max(float(temperature), 1.0e-6)
    )
    probability = torch.where(
        informative & torch.isfinite(probability),
        probability,
        torch.full_like(probability, 0.5),
    )
    return {
        "raw_probability": probability,
        "normalized_energy_margin": delta,
        "informative": informative,
    }


@torch.no_grad()
def build_gaussian_dino_graph(
    canonical_points,
    dino_features,
    *,
    neighbors: int,
    sigma: float,
    max_pairs: int = 262144,
    query_chunk_size: int = 512,
):
    """Build a detached canonical KNN graph with DINO material affinities."""
    points = torch.as_tensor(canonical_points).detach()
    features = torch.as_tensor(dino_features, device=points.device).detach().float()
    if features.ndim != 2 or features.shape[0] != points.shape[0]:
        raise ValueError("dino_features must be [N,F] and match canonical_points")
    graph = build_lagrangian_knn_pair_graph(
        points,
        neighbors_per_point=int(neighbors),
        max_pairs=int(max_pairs),
        query_chunk_size=int(query_chunk_size),
    )
    pairs = graph.pairs
    normalized = torch.nn.functional.normalize(features, dim=-1)
    cosine = (
        (normalized[pairs[:, 0]] * normalized[pairs[:, 1]]).sum(dim=-1)
        if pairs.numel()
        else points.new_empty((0,))
    )
    distance = (1.0 - cosine).clamp(0.0, 2.0)
    scale = max(float(sigma), 1.0e-6)
    affinity = torch.exp(-distance / scale)
    # DINO says where a boundary may exist; motion evidence decides each side.
    # One-sigma edges are the deterministic cutoff for responsibility smoothing.
    strong = distance <= scale
    return {
        "pairs": pairs.detach(),
        "spatial_distance": graph.distance.detach(),
        "dino_cosine": cosine.detach(),
        "dino_distance": distance.detach(),
        "affinity": affinity.detach(),
        "strong_affinity": strong.detach(),
        "sigma": scale,
    }


@torch.no_grad()
def smooth_gaussian_responsibility_dino(
    raw_probability,
    dino_graph,
    *,
    strength: float,
):
    """Weakly average responsibility only across strong detached DINO edges."""
    probability = torch.as_tensor(raw_probability).float()
    if dino_graph is None or float(strength) <= 0.0:
        return probability
    pairs = torch.as_tensor(dino_graph["pairs"], device=probability.device).long()
    strong = torch.as_tensor(
        dino_graph["strong_affinity"], device=probability.device
    ).bool()
    pairs = pairs[strong]
    if pairs.numel() == 0:
        return probability
    affinity = torch.as_tensor(
        dino_graph["affinity"], device=probability.device, dtype=probability.dtype
    )[strong]
    neighbor_sum = torch.zeros_like(probability)
    affinity_sum = torch.zeros_like(probability)
    neighbor_sum.index_add_(0, pairs[:, 0], affinity * probability[pairs[:, 1]])
    neighbor_sum.index_add_(0, pairs[:, 1], affinity * probability[pairs[:, 0]])
    affinity_sum.index_add_(0, pairs[:, 0], affinity)
    affinity_sum.index_add_(0, pairs[:, 1], affinity)
    weight = float(strength)
    return (probability + weight * neighbor_sum) / (
        1.0 + weight * affinity_sum
    )


@torch.no_grad()
def classify_final_gaussians(
    probability,
    *,
    static_threshold: float,
    rigid_threshold: float,
):
    """Polarize Gaussian responsibilities without moving ambiguous Gaussians."""
    value = torch.as_tensor(probability).float().clamp(0.0, 1.0)
    if float(static_threshold) >= float(rigid_threshold):
        raise ValueError("static_threshold must be lower than rigid_threshold")
    static = value <= float(static_threshold)
    rigid = value >= float(rigid_threshold)
    ambiguous = ~(static | rigid)
    state = torch.full(value.shape, AMBIGUOUS, device=value.device, dtype=torch.uint8)
    state[static] = STATIC
    state[rigid] = RIGID
    entropy = -(
        value.clamp_min(1.0e-8) * torch.log(value.clamp_min(1.0e-8))
        + (1.0 - value).clamp_min(1.0e-8)
        * torch.log((1.0 - value).clamp_min(1.0e-8))
    )
    return {
        "state": state,
        "static_mask": static,
        "rigid_mask": rigid,
        "ambiguous_mask": ambiguous,
        "entropy": entropy,
        "mean_entropy": float(entropy.mean()) if entropy.numel() else 0.0,
        "ambiguous_fraction": float(ambiguous.float().mean()) if value.numel() else 0.0,
    }


@torch.no_grad()
def dino_boundary_metrics(dino_graph, binary_labels, support_ids=None):
    """Evaluate detached DINO boundary edges against post-solve binary labels."""
    pairs = torch.as_tensor(dino_graph["pairs"]).long().cpu()
    labels = torch.as_tensor(binary_labels).cpu()
    predicted = ~torch.as_tensor(dino_graph["strong_affinity"]).bool().cpu()
    actual = labels[pairs[:, 0]] != labels[pairs[:, 1]]
    tp = int((predicted & actual).sum())
    fp = int((predicted & ~actual).sum())
    fn = int((~predicted & actual).sum())
    result = {
        "true_positive": tp,
        "false_positive": fp,
        "false_negative": fn,
        "precision": tp / (tp + fp) if tp + fp else None,
        "recall": tp / (tp + fn) if tp + fn else None,
        "boundary_edges": int(actual.sum()),
    }
    if support_ids is not None:
        groups = torch.as_tensor(support_ids).long().cpu()
        result["boundary_edges_remaining_inside_support"] = int(
            (actual & (groups[pairs[:, 0]] == groups[pairs[:, 1]])).sum()
        )
    return result


def _boundary_segments(points, pairs):
    xyz = torch.as_tensor(points).detach().cpu().float()
    x, y, z = [], [], []
    for first, second in torch.as_tensor(pairs).detach().cpu().long().tolist():
        for values, axis in ((x, 0), (y, 1), (z, 2)):
            values.extend((float(xyz[first, axis]), float(xyz[second, axis]), None))
    return x, y, z


@torch.no_grad()
def write_gaussian_responsibility_html(
    points,
    probability,
    classification,
    dino_graph,
    output,
    *,
    title,
    max_points=1000,
    max_edges=2000,
):
    """Write the representation-independent per-cycle material diagnostic."""
    xyz = torch.as_tensor(points).detach().cpu().float()
    p = torch.as_tensor(probability).detach().cpu().float()
    state = torch.as_tensor(classification["state"]).detach().cpu().long()
    if xyz.shape[0] > int(max_points):
        selected = torch.linspace(0, xyz.shape[0] - 1, int(max_points)).round().long().unique()
    else:
        selected = torch.arange(xyz.shape[0])
    sampled = xyz[selected]
    hover = [
        f"gaussian={int(index)}<br>p_rigid={float(p[index]):.5f}<br>class={int(state[index])}"
        for index in selected
    ]

    def marker(color, scale, name):
        return {
            "type": "scatter3d",
            "mode": "markers",
            "x": sampled[:, 0].tolist(),
            "y": sampled[:, 1].tolist(),
            "z": sampled[:, 2].tolist(),
            "text": hover,
            "hoverinfo": "text",
            "marker": {
                "size": 3,
                "color": color,
                "colorscale": scale,
                "colorbar": left_colorbar(name),
                "opacity": 0.86,
            },
            "name": name,
        }

    traces = [
        marker(p[selected].tolist(), "Viridis", "Gaussian p(rigid)"),
        marker(state[selected].tolist(), "Portland", "final class"),
    ]
    traces[1]["visible"] = False
    pairs = torch.as_tensor(dino_graph["pairs"]).detach().cpu().long()
    discontinuity = ~torch.as_tensor(
        dino_graph["strong_affinity"]
    ).detach().cpu().bool()
    edge_xyz = _boundary_segments(xyz, pairs[discontinuity][: int(max_edges)])
    traces.append({
        "type": "scatter3d",
        "mode": "lines",
        "x": edge_xyz[0],
        "y": edge_xyz[1],
        "z": edge_xyz[2],
        "line": {"width": 2, "color": "rgba(220,30,30,0.6)"},
        "name": "strong DINO discontinuity",
        "hoverinfo": "skip",
    })
    buttons = [{
        "type": "buttons",
        "direction": "down",
        "buttons": [
            {
                "label": label,
                "method": "update",
                "args": [{"visible": [index == item for index in range(2)] + [True]}],
            }
            for item, label in enumerate(("responsibility", "final class"))
        ],
    }]
    payload = {
        "data": traces,
        "layout": plotly_3d_layout(
            title, buttons=buttons, scene_bounds=fixed_scene_bounds(xyz)
        ),
    }
    path = Path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "<!doctype html><html><head><meta charset='utf-8'>"
        "<script src='https://cdn.plot.ly/plotly-2.35.2.min.js'></script></head>"
        "<body style='margin:0'><div id='plot' style='width:100vw;height:100vh'>"
        "</div>"
        + f"<script>const p={json.dumps(payload)};Plotly.newPlot('plot',p.data,p.layout,{json.dumps(plotly_3d_config())});</script>"
        "</body></html>",
        encoding="utf-8",
    )
    return path
