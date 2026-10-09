"""DINO/responsibility-contrastive Eulerian observation supports."""

from __future__ import annotations

import json
from pathlib import Path

import torch

from utils.html_plot_layout import fixed_scene_bounds, left_colorbar, plotly_3d_config, plotly_3d_layout
from utils.rigid_killing_motion import build_rigid_observation_grid


@torch.no_grad()
def assign_gaussians_to_cells(points, target_cells, *, padding_fraction, min_cells_per_axis):
    return build_rigid_observation_grid(
        points,
        int(target_cells),
        padding_fraction=float(padding_fraction),
        min_cells_per_axis=int(min_cells_per_axis),
    )


@torch.no_grad()
def build_raw_support_state(cell_ids, num_cells, *, min_gaussians=1):
    """Create unsplit Eulerian supports; DINO never partitions cycle zero."""
    ids = torch.as_tensor(cell_ids).detach().cpu().long().reshape(-1)
    if ids.numel() and (int(ids.min()) < 0 or int(ids.max()) >= int(num_cells)):
        raise ValueError("cell_ids must be compact non-negative raw-cell IDs")
    counts = torch.bincount(ids, minlength=int(num_cells))
    occupied = torch.nonzero(counts > 0, as_tuple=False).flatten()
    remap = torch.full((int(num_cells),), -1, dtype=torch.long)
    remap[occupied] = torch.arange(occupied.numel())
    groups = remap[ids]
    support_counts = counts[occupied]
    return {
        "group_ids": groups,
        "num_supports": int(occupied.numel()),
        "parent_cell_id": occupied,
        "parent_support_id": torch.full((occupied.numel(),), -1, dtype=torch.long),
        "split_reason": ["raw_cell"] * int(occupied.numel()),
        "split_cycle": torch.full((occupied.numel(),), -1, dtype=torch.long),
        "weak_support": support_counts < int(min_gaussians),
        "gaussian_count": support_counts,
        "initial_dino_group_ids": groups.clone(),
        "membership_changes": 0,
    }


@torch.no_grad()
def detect_mixed_supports_from_responsibility(
    support_state,
    responsibilities,
    *,
    confidence_threshold,
    min_class_fraction=0.10,
):
    """Detect motion-proven K+1 conflicts before allowing a DINO split."""
    ids = torch.as_tensor(support_state["group_ids"]).detach().cpu().long()
    value = torch.as_tensor(responsibilities).detach().cpu().float()
    if value.ndim != 2 or value.shape[0] != ids.numel():
        raise ValueError("responsibilities must be [N,K+1]")
    confidence, labels = value.max(dim=-1)
    confident = confidence >= float(confidence_threshold)
    count = int(support_state["num_supports"])
    fractions = torch.zeros((count, value.shape[1]), dtype=torch.float32)
    totals = torch.bincount(ids, minlength=count).float().clamp_min(1.0)
    for label in range(value.shape[1]):
        selected = confident & (labels == label)
        fractions[:, label].index_add_(0, ids, selected.float())
    fractions /= totals[:, None]
    significant = fractions >= float(min_class_fraction)
    mixed = significant.sum(dim=-1) >= 2
    return {
        "mixed_support_mask": mixed,
        "class_fraction": fractions,
        "confident_labels": labels,
        "confident_mask": confident,
        "mixed_support_count": int(mixed.sum()),
    }


@torch.no_grad()
def split_mixed_supports_with_dino(
    support_state,
    responsibilities,
    dino_graph,
    *,
    cycle,
    enabled,
    confidence_threshold,
    min_class_fraction=0.10,
    min_gaussians=1,
):
    """Persistently split only responsibility-proven mixed supports.

    Confident responsibility labels are unary seeds. Strong DINO edges vote only
    for ambiguous members inside the already-proven mixed support.
    """
    diagnostic = detect_mixed_supports_from_responsibility(
        support_state,
        responsibilities,
        confidence_threshold=float(confidence_threshold),
        min_class_fraction=float(min_class_fraction),
    )
    if not bool(enabled) or int(cycle) <= 0 or diagnostic["mixed_support_count"] == 0:
        return {**support_state, "membership_changes": 0, "mixed_diagnostic": diagnostic}

    old_ids = torch.as_tensor(support_state["group_ids"]).detach().cpu().long()
    value = torch.as_tensor(responsibilities).detach().cpu().float()
    labels = diagnostic["confident_labels"].clone()
    confident = diagnostic["confident_mask"]
    if dino_graph is None:
        pairs = torch.empty((0, 2), dtype=torch.long)
        affinity = torch.empty(0)
    else:
        pairs = torch.as_tensor(dino_graph["pairs"]).detach().cpu().long()
        strong = torch.as_tensor(dino_graph["strong_affinity"]).detach().cpu().bool()
        affinity = torch.as_tensor(dino_graph["affinity"]).detach().cpu().float()
        pairs, affinity = pairs[strong], affinity[strong]

    new_ids = torch.empty_like(old_ids)
    parent_cell, parent_support, reasons, cycles = [], [], [], []
    next_group = 0
    changed = 0
    for support in range(int(support_state["num_supports"])):
        members = torch.nonzero(old_ids == support, as_tuple=False).flatten()
        do_split = bool(diagnostic["mixed_support_mask"][support])
        local_labels = labels[members].clone()
        if do_split:
            ambiguous = ~confident[members]
            if bool(ambiguous.any()) and pairs.numel():
                member_lookup = torch.full((old_ids.numel(),), -1, dtype=torch.long)
                member_lookup[members] = torch.arange(members.numel())
                inside = (old_ids[pairs[:, 0]] == support) & (old_ids[pairs[:, 1]] == support)
                local_pairs, local_affinity = pairs[inside], affinity[inside]
                votes = torch.zeros((members.numel(), value.shape[1]))
                for edge_index, (first, second) in enumerate(local_pairs.tolist()):
                    a, b = int(member_lookup[first]), int(member_lookup[second])
                    w = float(local_affinity[edge_index])
                    if confident[second]:
                        votes[a, int(labels[second])] += w
                    if confident[first]:
                        votes[b, int(labels[first])] += w
                has_vote = votes.sum(dim=-1) > 0
                local_labels[ambiguous & has_vote] = votes[ambiguous & has_vote].argmax(dim=-1)
            # Responsibilities remain the fallback unary for nodes without a
            # strong DINO connection to a confident seed.
            present = sorted(int(item) for item in local_labels.unique().tolist())
        else:
            local_labels.zero_()
            present = [0]
        if len(present) > 1:
            changed += int(members.numel())
        for label in present:
            selected = members[local_labels == label]
            if selected.numel() == 0:
                continue
            new_ids[selected] = next_group
            parent_cell.append(int(support_state["parent_cell_id"][support]))
            parent_support.append(support if len(present) > 1 else int(support_state["parent_support_id"][support]))
            reasons.append("responsibility+dino" if len(present) > 1 else support_state["split_reason"][support])
            cycles.append(int(cycle) if len(present) > 1 else int(support_state["split_cycle"][support]))
            next_group += 1
    counts = torch.bincount(new_ids, minlength=next_group)
    return {
        **support_state,
        "group_ids": new_ids,
        "num_supports": next_group,
        "parent_cell_id": torch.tensor(parent_cell, dtype=torch.long),
        "parent_support_id": torch.tensor(parent_support, dtype=torch.long),
        "split_reason": reasons,
        "split_cycle": torch.tensor(cycles, dtype=torch.long),
        "weak_support": counts < int(min_gaussians),
        "gaussian_count": counts,
        "membership_changes": changed,
        "mixed_diagnostic": diagnostic,
    }


@torch.no_grad()
def aggregate_gaussian_part_weights_to_support(support_state, responsibilities, weights=None):
    """Return the weighted K+1 responsibility mean for every CUDA group ID."""
    ids = torch.as_tensor(support_state["group_ids"]).long()
    value = torch.as_tensor(responsibilities, device=ids.device).float()
    if value.ndim != 2 or value.shape[0] != ids.numel():
        raise ValueError("responsibilities must be [N,K+1]")
    point_weight = torch.ones(ids.numel(), device=ids.device, dtype=value.dtype)
    if weights is not None:
        point_weight = torch.as_tensor(weights, device=ids.device, dtype=value.dtype).reshape(-1)
    count = int(support_state["num_supports"])
    numerator = torch.zeros((count, value.shape[1]), device=ids.device, dtype=value.dtype)
    denominator = torch.zeros(count, device=ids.device, dtype=value.dtype)
    numerator.index_add_(0, ids, value * point_weight[:, None])
    denominator.index_add_(0, ids, point_weight)
    return numerator / denominator.clamp_min(1.0e-8)[:, None]


@torch.no_grad()
def compute_cell_dino_dispersion(cell_ids, dino_features, num_cells):
    ids = torch.as_tensor(cell_ids).detach().cpu().long()
    features = torch.nn.functional.normalize(
        torch.as_tensor(dino_features).detach().cpu().float(), dim=-1
    )
    if features.ndim != 2 or features.shape[0] != ids.shape[0]:
        raise ValueError("dino_features must be [N,F] and match cell_ids")
    feature_sum = torch.zeros((int(num_cells), features.shape[1]), dtype=features.dtype)
    count = torch.bincount(ids, minlength=int(num_cells)).float()
    feature_sum.index_add_(0, ids, features)
    resultant = torch.linalg.norm(feature_sum / count.clamp_min(1.0)[:, None], dim=-1)
    return torch.where(count > 0, 1.0 - resultant, torch.zeros_like(resultant))


def _connected_components(neighbor_pairs, count):
    parent = list(range(int(count)))

    def find(value):
        while parent[value] != value:
            parent[value] = parent[parent[value]]
            value = parent[value]
        return value

    for first, second in neighbor_pairs:
        a, b = find(int(first)), find(int(second))
        if a != b:
            parent[max(a, b)] = min(a, b)
    roots = [find(index) for index in range(int(count))]
    unique = {root: index for index, root in enumerate(sorted(set(roots)))}
    return torch.tensor([unique[root] for root in roots], dtype=torch.long)


@torch.no_grad()
def split_cells_by_dino_components(
    points,
    cell_ids,
    dino_features,
    *,
    num_cells,
    enabled: bool,
    cosine_threshold: float,
    dispersion_threshold: float,
    min_gaussians: int,
    local_neighbors: int = 8,
):
    """Split only DINO-dispersed raw cells using local spatial graph components."""
    xyz = torch.as_tensor(points).detach().cpu().float()
    raw_ids = torch.as_tensor(cell_ids).detach().cpu().long()
    features = torch.nn.functional.normalize(
        torch.as_tensor(dino_features).detach().cpu().float(), dim=-1
    )
    dispersion = compute_cell_dino_dispersion(raw_ids, features, int(num_cells))
    group_ids = torch.empty_like(raw_ids)
    parent_cell, split_reason, split_cycle = [], [], []
    next_group = 0
    for cell in range(int(num_cells)):
        members = torch.nonzero(raw_ids == cell, as_tuple=False).flatten()
        if members.numel() == 0:
            continue
        components = torch.zeros(members.numel(), dtype=torch.long)
        if (
            bool(enabled)
            and members.numel() > 1
            and float(dispersion[cell]) > float(dispersion_threshold)
        ):
            local_xyz = xyz[members]
            local_dino = features[members]
            distance = torch.cdist(local_xyz, local_xyz)
            distance.fill_diagonal_(float("inf"))
            k = min(max(1, int(local_neighbors)), int(members.numel()) - 1)
            neighbors = torch.topk(distance, k=k, largest=False).indices
            source = torch.arange(members.numel())[:, None].expand_as(neighbors)
            cosine = (local_dino[source] * local_dino[neighbors]).sum(dim=-1)
            kept = torch.stack((source[cosine >= float(cosine_threshold)], neighbors[cosine >= float(cosine_threshold)]), dim=-1)
            components = _connected_components(kept.tolist(), int(members.numel()))
        component_count = int(components.max()) + 1
        reason = "dino" if component_count > 1 else "raw_cell"
        for component in range(component_count):
            selected = members[components == component]
            group_ids[selected] = next_group
            parent_cell.append(cell)
            split_reason.append(reason)
            split_cycle.append(0 if reason == "dino" else -1)
            next_group += 1
    counts = torch.bincount(group_ids, minlength=next_group)
    return {
        "group_ids": group_ids,
        "num_supports": next_group,
        "parent_cell_id": torch.tensor(parent_cell, dtype=torch.long),
        "parent_support_id": torch.full((next_group,), -1, dtype=torch.long),
        "split_reason": split_reason,
        "split_cycle": torch.tensor(split_cycle, dtype=torch.long),
        "weak_support": counts < int(min_gaussians),
        "gaussian_count": counts,
        "cell_dino_dispersion": dispersion,
        "initial_dino_group_ids": group_ids.clone(),
        "membership_changes": 0,
    }


def build_contrastive_group_ids(support_state, *, device=None):
    return torch.as_tensor(support_state["group_ids"], device=device, dtype=torch.long)


@torch.no_grad()
def compute_support_responsibility_stats(
    support_state,
    probability,
    *,
    static_threshold: float,
    rigid_threshold: float,
):
    ids = torch.as_tensor(support_state["group_ids"]).cpu().long()
    value = torch.as_tensor(probability).detach().cpu().float()
    count = int(support_state["num_supports"])
    ones = torch.ones_like(value)
    support_count = torch.zeros(count).index_add_(0, ids, ones)
    total = torch.zeros(count).index_add_(0, ids, value)
    square = torch.zeros(count).index_add_(0, ids, value.square())
    mean = total / support_count.clamp_min(1.0)
    variance = square / support_count.clamp_min(1.0) - mean.square()
    static_fraction = torch.zeros(count).index_add_(
        0, ids, (value <= float(static_threshold)).float()
    ) / support_count.clamp_min(1.0)
    rigid_fraction = torch.zeros(count).index_add_(
        0, ids, (value >= float(rigid_threshold)).float()
    ) / support_count.clamp_min(1.0)
    return {
        "gaussian_count": support_count.long(),
        "mean_p": mean,
        "variance_p": variance.clamp_min(0.0),
        "static_fraction": static_fraction,
        "rigid_fraction": rigid_fraction,
    }


@torch.no_grad()
def split_supports_by_responsibility(
    support_state,
    probability,
    *,
    cycle: int,
    enabled: bool,
    static_threshold: float,
    rigid_threshold: float,
    min_bimodal_fraction: float,
    min_gaussians: int,
):
    """Persistently split bimodal supports; never merge existing subgroups."""
    if not bool(enabled) or int(cycle) <= 0:
        return {**support_state, "membership_changes": 0}
    old_ids = torch.as_tensor(support_state["group_ids"]).cpu().long()
    value = torch.as_tensor(probability).detach().cpu().float()
    stats = compute_support_responsibility_stats(
        support_state,
        value,
        static_threshold=float(static_threshold),
        rigid_threshold=float(rigid_threshold),
    )
    new_ids = torch.empty_like(old_ids)
    parent_cell, parent_support, reasons, cycles = [], [], [], []
    next_group = 0
    changed = 0
    for support in range(int(support_state["num_supports"])):
        members = torch.nonzero(old_ids == support, as_tuple=False).flatten()
        mixed = (
            float(stats["static_fraction"][support]) > float(min_bimodal_fraction)
            and float(stats["rigid_fraction"][support]) > float(min_bimodal_fraction)
        )
        categories = (
            (value[members] <= float(static_threshold)).long()
            + 2 * (value[members] >= float(rigid_threshold)).long()
            if mixed
            else torch.zeros(members.numel(), dtype=torch.long)
        )
        present = sorted(int(item) for item in categories.unique().tolist())
        if len(present) > 1:
            changed += int(members.numel())
        for category in present:
            selected = members[categories == category]
            new_ids[selected] = next_group
            parent_cell.append(int(support_state["parent_cell_id"][support]))
            parent_support.append(
                support
                if len(present) > 1
                else int(support_state["parent_support_id"][support])
            )
            reasons.append("responsibility" if len(present) > 1 else support_state["split_reason"][support])
            cycles.append(int(cycle) if len(present) > 1 else int(support_state["split_cycle"][support]))
            next_group += 1
    counts = torch.bincount(new_ids, minlength=next_group)
    return {
        **support_state,
        "group_ids": new_ids,
        "num_supports": next_group,
        "parent_cell_id": torch.tensor(parent_cell, dtype=torch.long),
        "parent_support_id": torch.tensor(parent_support, dtype=torch.long),
        "split_reason": reasons,
        "split_cycle": torch.tensor(cycles, dtype=torch.long),
        "weak_support": counts < int(min_gaussians),
        "gaussian_count": counts,
        "membership_changes": changed,
    }


@torch.no_grad()
def aggregate_gaussian_responsibility_to_support(
    support_state,
    probability,
    weights=None,
):
    ids = torch.as_tensor(support_state["group_ids"])
    value = torch.as_tensor(probability, device=ids.device).float()
    weight = torch.ones_like(value) if weights is None else torch.as_tensor(weights, device=ids.device).float()
    count = int(support_state["num_supports"])
    numerator = torch.zeros(count, device=ids.device).index_add_(0, ids, weight * value)
    denominator = torch.zeros(count, device=ids.device).index_add_(0, ids, weight)
    return numerator / denominator.clamp_min(1.0e-8)


@torch.no_grad()
def summarize_support_purity(support_state, static_mask, rigid_mask):
    ids = torch.as_tensor(support_state["group_ids"]).cpu().long()
    static = torch.as_tensor(static_mask).cpu().bool()
    rigid = torch.as_tensor(rigid_mask).cpu().bool()
    count = int(support_state["num_supports"])
    total = torch.bincount(ids, minlength=count).float().clamp_min(1.0)
    n_static = torch.zeros(count).index_add_(0, ids, static.float())
    n_rigid = torch.zeros(count).index_add_(0, ids, rigid.float())
    purity = torch.maximum(n_static, n_rigid) / total
    return {
        "support_purity": purity,
        "median_support_purity": float(purity.median()) if purity.numel() else None,
        "p10_support_purity": float(torch.quantile(purity, 0.10)) if purity.numel() else None,
        "fraction_purity_gt_0_90": float((purity > 0.90).float().mean()) if purity.numel() else None,
        "fraction_purity_gt_0_95": float((purity > 0.95).float().mean()) if purity.numel() else None,
    }


@torch.no_grad()
def mixed_cell_diagnostics(raw_cell_ids, support_state, static_mask, rigid_mask):
    raw = torch.as_tensor(raw_cell_ids).cpu().long()
    static = torch.as_tensor(static_mask).cpu().bool()
    rigid = torch.as_tensor(rigid_mask).cpu().bool()

    def mixed_fraction(ids):
        count = int(ids.max()) + 1 if ids.numel() else 0
        has_static = torch.zeros(count, dtype=torch.long).index_add_(0, ids, static.long()) > 0
        has_rigid = torch.zeros(count, dtype=torch.long).index_add_(0, ids, rigid.long()) > 0
        mixed = has_static & has_rigid
        point_mixed = mixed[ids] if ids.numel() else torch.zeros(0, dtype=torch.bool)
        return (
            float(mixed.float().mean()) if count else 0.0,
            float(point_mixed.float().mean()) if ids.numel() else 0.0,
            has_static,
            has_rigid,
            mixed,
        )

    raw_fraction, raw_point_fraction, raw_static, raw_rigid, raw_mixed = mixed_fraction(raw)
    initial_fraction, _, _, _, _ = mixed_fraction(
        torch.as_tensor(support_state["initial_dino_group_ids"]).cpu().long()
    )
    final_fraction, final_point_fraction, _, _, _ = mixed_fraction(
        torch.as_tensor(support_state["group_ids"]).cpu().long()
    )
    return {
        "contains_final_static": raw_static,
        "contains_final_rigid": raw_rigid,
        "mixed_material_cell": raw_mixed,
        "fraction_occupied_cells_mixed": raw_fraction,
        "fraction_gaussians_inside_mixed_cells": raw_point_fraction,
        "fraction_mixed_before_dino_splitting": raw_fraction,
        "fraction_mixed_after_dino_splitting": initial_fraction,
        "fraction_mixed_after_responsibility_refinement": final_fraction,
        "fraction_gaussians_inside_mixed_final_supports": final_point_fraction,
    }


@torch.no_grad()
def summarize_geometric_cell_transitions(position_history, observation_grid):
    """Report actual entry/exit events relative to the fixed geometric grid."""
    positions = torch.stack(
        [torch.as_tensor(value).detach().cpu().double() for value in position_history]
    )
    origin = torch.as_tensor(observation_grid.origin).detach().cpu().double()
    size = torch.as_tensor(observation_grid.cell_size).detach().cpu().double()
    counts = torch.as_tensor(observation_grid.cell_counts).detach().cpu().long()
    coordinate = torch.floor((positions - origin) / size).long()
    inside = ((coordinate >= 0) & (coordinate < counts)).all(dim=-1)
    coordinate = torch.minimum(
        torch.maximum(coordinate, torch.zeros_like(coordinate)), counts - 1
    )
    dense_ids = (
        (coordinate[..., 0] * counts[1] + coordinate[..., 1]) * counts[2]
        + coordinate[..., 2]
    )
    dense_ids = torch.where(inside, dense_ids, torch.full_like(dense_ids, -1))
    changed = dense_ids[1:] != dense_ids[:-1]
    entered = inside[1:] & (~inside[:-1] | changed)
    left = inside[:-1] & (~inside[1:] | changed)
    return {
        "dense_geometric_cell_id_history": dense_ids,
        "inside_grid_history": inside,
        "gaussians_entering_per_frame": entered.sum(dim=1),
        "gaussians_leaving_per_frame": left.sum(dim=1),
        "membership_changes_per_frame": changed.sum(dim=1),
        "gaussians_ever_changing_cell": changed.any(dim=0),
    }


@torch.no_grad()
def support_observation_diagnostics(support_state, observation_states):
    count = int(support_state["num_supports"])
    observations = torch.stack([state["observation_count"].detach().cpu() for state in observation_states])
    cameras = torch.stack([state["distinct_camera_count"].detach().cpu() for state in observation_states])
    condition = torch.stack([state["conditioning"].detach().cpu() for state in observation_states])
    weight = torch.stack([state["weight_sum"].detach().cpu() for state in observation_states])
    valid = torch.stack([state["base_valid"].detach().cpu() for state in observation_states])
    if observations.shape[1] != count:
        raise ValueError("observation support count does not match support_state")
    masked_condition = torch.where(valid, condition, torch.full_like(condition, float("nan")))
    return {
        "observation_count": observations.sum(dim=0),
        "distinct_camera_count": cameras.max(dim=0).values,
        "condition_number": torch.nanmedian(masked_condition.clamp_min(1.0e-12).reciprocal(), dim=0).values,
        "velocity_target_uncertainty": weight.sum(dim=0).clamp_min(1.0e-8).rsqrt(),
        "valid_support_fraction": float(valid.any(dim=0).float().mean()),
        "weak_conditioning_fraction": float((~valid.any(dim=0)).float().mean()),
    }


def _line_segments(points, pairs):
    xyz = points.detach().cpu()
    x, y, z = [], [], []
    for first, second in pairs.detach().cpu().tolist():
        for values, axis in ((x, 0), (y, 1), (z, 2)):
            values.extend((float(xyz[first, axis]), float(xyz[second, axis]), None))
    return x, y, z


@torch.no_grad()
def write_eulerian_support_html(
    points,
    probability,
    classification,
    support_state,
    dino_graph,
    cell_centers,
    cell_size,
    output,
    *,
    title,
    max_points=1000,
    max_edges=2000,
    max_cells=512,
):
    """Compact per-cycle responsibility, boundary, support, and variance view."""
    xyz = torch.as_tensor(points).detach().cpu().float()
    p = torch.as_tensor(probability).detach().cpu().float()
    state = torch.as_tensor(classification["state"]).detach().cpu().long()
    groups = torch.as_tensor(support_state["group_ids"]).cpu().long()
    stats = compute_support_responsibility_stats(
        support_state, p, static_threshold=0.2, rigid_threshold=0.8
    )
    variance = stats["variance_p"][groups]
    if xyz.shape[0] > int(max_points):
        ids = torch.linspace(0, xyz.shape[0] - 1, int(max_points)).round().long().unique()
    else:
        ids = torch.arange(xyz.shape[0])
    sampled = xyz[ids]
    hover = [f"gaussian={int(i)}<br>p_rigid={float(p[i]):.5f}<br>cell={int(torch.as_tensor(support_state['parent_cell_id'])[groups[i]])}<br>support={int(groups[i])}" for i in ids]
    marker = lambda color, scale, name: {
        "type": "scatter3d", "mode": "markers", "x": sampled[:, 0].tolist(),
        "y": sampled[:, 1].tolist(), "z": sampled[:, 2].tolist(), "text": hover,
        "hoverinfo": "text", "marker": {"size": 3, "color": color, "colorscale": scale,
        "colorbar": left_colorbar(name), "opacity": 0.86}, "name": name,
    }
    traces = [
        marker(p[ids].tolist(), "Viridis", "Gaussian p(rigid)"),
        marker(state[ids].tolist(), "Portland", "final class"),
        marker(groups[ids].tolist(), "Turbo", "contrastive support"),
        marker(variance[ids].tolist(), "Magma", "support Var(p)"),
    ]
    for trace in traces[1:]:
        trace["visible"] = False
    pairs = torch.as_tensor(dino_graph["pairs"]).cpu().long()
    boundary = ~torch.as_tensor(dino_graph["strong_affinity"]).cpu().bool()
    pairs = pairs[boundary][: int(max_edges)]
    edge_xyz = _line_segments(xyz, pairs)
    traces.append({"type": "scatter3d", "mode": "lines", "x": edge_xyz[0], "y": edge_xyz[1], "z": edge_xyz[2], "line": {"width": 2, "color": "rgba(220,30,30,0.6)"}, "name": "strong DINO discontinuity", "hoverinfo": "skip"})
    centers = torch.as_tensor(cell_centers).detach().cpu().float()
    if centers.shape[0] > int(max_cells):
        center_ids = torch.linspace(
            0, centers.shape[0] - 1, int(max_cells)
        ).round().long().unique()
        centers = centers[center_ids]
    half = 0.5 * torch.as_tensor(cell_size).detach().cpu().float()
    box_pairs = ((0,1),(1,3),(3,2),(2,0),(4,5),(5,7),(7,6),(6,4),(0,4),(1,5),(2,6),(3,7))
    bx, by, bz = [], [], []
    signs = torch.tensor([[-1,-1,-1],[-1,-1,1],[-1,1,-1],[-1,1,1],[1,-1,-1],[1,-1,1],[1,1,-1],[1,1,1]], dtype=torch.float32)
    for center in centers:
        corners = center + signs * half
        for first, second in box_pairs:
            for values, axis in ((bx,0),(by,1),(bz,2)):
                values.extend((float(corners[first,axis]), float(corners[second,axis]), None))
    traces.append({"type": "scatter3d", "mode": "lines", "x": bx, "y": by, "z": bz, "line": {"width": 1, "color": "rgba(70,70,70,0.18)"}, "name": "raw geometric cells", "hoverinfo": "skip"})
    labels = ("responsibility", "final class", "support id", "support variance")
    buttons = [{"type": "buttons", "direction": "down", "buttons": [{"label": label, "method": "update", "args": [{"visible": [index == item for index in range(4)] + [True, True]}]} for item, label in enumerate(labels)]}]
    payload = {"data": traces, "layout": plotly_3d_layout(title, buttons=buttons, scene_bounds=fixed_scene_bounds(xyz))}
    path = Path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("<!doctype html><html><head><meta charset='utf-8'><script src='https://cdn.plot.ly/plotly-2.35.2.min.js'></script></head><body style='margin:0'><div id='plot' style='width:100vw;height:100vh'></div>" + f"<script>const p={json.dumps(payload)};Plotly.newPlot('plot',p.data,p.layout,{json.dumps(plotly_3d_config())});</script></body></html>", encoding="utf-8")
    return path
