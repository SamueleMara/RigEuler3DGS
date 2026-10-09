"""Multi-frame full-scene counterfactual evidence for uncertain K-parts."""

from __future__ import annotations

import torch
import torch.nn.functional as F



@torch.no_grad()
def extract_uncertain_components(
    pairs,
    same_material_affinity,
    uncertain_mask,
    *,
    min_affinity=0.1,
    min_component_size=2,
):
    """Extract coherent uncertain material regions without singleton ranking bias."""
    edge = torch.as_tensor(pairs).detach().cpu().long()
    affinity = torch.as_tensor(same_material_affinity).detach().cpu().float()
    uncertain = torch.as_tensor(uncertain_mask).detach().cpu().bool()
    adjacency = [[] for _ in range(uncertain.numel())]
    keep = (affinity >= float(min_affinity)) & uncertain[edge[:, 0]] & uncertain[edge[:, 1]]
    for first, second in edge[keep].tolist():
        adjacency[first].append(second)
        adjacency[second].append(first)
    visited = torch.zeros_like(uncertain)
    components = []
    for seed in torch.nonzero(uncertain, as_tuple=False).flatten().tolist():
        if visited[seed]:
            continue
        visited[seed] = True
        stack, members = [seed], []
        while stack:
            node = stack.pop()
            members.append(node)
            for neighbor in adjacency[node]:
                if not visited[neighbor]:
                    visited[neighbor] = True
                    stack.append(neighbor)
        if len(members) >= int(min_component_size):
            components.append(torch.as_tensor(members, dtype=torch.long))
    # Isolated uncertainty is retained only if no coherent region exists, so
    # it can never outrank a real multi-Gaussian component.
    if not components:
        components = [torch.as_tensor([index], dtype=torch.long) for index in torch.nonzero(uncertain).flatten().tolist()]
    return components


@torch.no_grad()
def rank_components_for_rendering(components, entropy, *, image_influence=None, recent_change=None, max_components=None):
    entropy_value = torch.as_tensor(entropy).detach().cpu().float()
    influence = torch.ones_like(entropy_value) if image_influence is None else torch.as_tensor(image_influence).detach().cpu().float()
    change = torch.zeros_like(entropy_value) if recent_change is None else torch.as_tensor(recent_change).detach().cpu().float()
    ranked = []
    for component in components:
        ids = torch.as_tensor(component).detach().cpu().long()
        score = float(ids.numel()) * float(entropy_value[ids].mean()) * float(influence[ids].mean().clamp_min(1.0e-4))
        score *= 1.0 + float(change[ids].mean())
        ranked.append((score, ids))
    ranked.sort(key=lambda item: (item[0], int(item[1].numel())), reverse=True)
    result = [item[1] for item in ranked]
    return result if max_components is None else result[: int(max_components)]


@torch.no_grad()
def select_informative_frame_pairs(
    projected_positions,
    *,
    count=3,
    min_displacement=0.5,
    max_displacement=64.0,
    bidirectional=True,
):
    """Select short/medium/long temporal baselines by actual image motion.

    ``projected_positions`` is ``[F,M,2]`` or ``[F,2]`` for representative
    component points. Fixed frame offsets are deliberately not used.
    """
    projected = torch.as_tensor(projected_positions).detach().cpu().float()
    if projected.ndim == 2:
        projected = projected[:, None]
    candidates = []
    for first in range(projected.shape[0] - 1):
        for second in range(first + 1, projected.shape[0]):
            displacement = torch.linalg.norm(projected[second] - projected[first], dim=-1)
            valid = torch.isfinite(displacement)
            if not bool(valid.any()):
                continue
            amount = float(displacement[valid].median())
            if float(min_displacement) <= amount <= float(max_displacement):
                candidates.append((first, second, amount))
    if not candidates:
        return []
    candidates.sort(key=lambda item: item[2])
    requested = max(1, int(count))
    quantiles = torch.linspace(0, len(candidates) - 1, min(requested, len(candidates))).round().long().unique()
    selected = [candidates[int(index)] for index in quantiles]
    result = []
    for first, second, displacement in selected:
        result.append({"source": first, "target": second, "projected_displacement": displacement})
        if bidirectional:
            result.append({"source": second, "target": first, "projected_displacement": displacement})
    return result


@torch.no_grad()
def render_counterfactual_scene(
    camera,
    gaussians,
    pipe,
    background,
    positions,
    orientations,
    *,
    component_mask,
    dino_features=None,
    use_trained_exp=False,
):
    """Render the complete scene plus component coverage in one feature pass."""
    from gaussian_renderer import render, render_feature_field

    xyz = torch.as_tensor(positions, device=gaussians.get_xyz.device, dtype=gaussians.get_xyz.dtype)
    rotation = torch.as_tensor(orientations, device=xyz.device, dtype=gaussians.get_rotation.dtype)
    gaussians.set_free_working_state(
        xyz,
        float(getattr(camera, "timestamp", 0.0)),
        velocity_override=torch.zeros_like(xyz),
        rotation_override=rotation,
    )
    try:
        rgb = render(camera, gaussians, pipe, background, use_trained_exp=bool(use_trained_exp))
        component = torch.as_tensor(component_mask, device=xyz.device, dtype=xyz.dtype)[:, None]
        ones = torch.ones_like(component)
        if dino_features is None:
            feature = torch.cat((ones, component), dim=-1)
            dino_dim = 0
        else:
            dino = F.normalize(torch.as_tensor(dino_features, device=xyz.device).detach().float(), dim=-1)
            feature = torch.cat((dino, ones, component), dim=-1)
            dino_dim = dino.shape[1]
        field = render_feature_field(
            camera, gaussians, pipe, feature,
            bg_feature=torch.zeros(feature.shape[1], device=xyz.device, dtype=feature.dtype),
            detach_geometry=True,
        )["render"]
        alpha = field[dino_dim].clamp(0, 1)
        component_alpha = field[dino_dim + 1].clamp(0, 1)
        rendered_dino = None
        if dino_dim:
            rendered_dino = F.normalize(field[:dino_dim] / alpha.clamp_min(1.0e-6)[None], dim=0)
        return {
            "rgb": rgb["render"], "depth": rgb["depth"],
            "dino": rendered_dino, "alpha": alpha,
            "component_alpha": component_alpha,
        }
    finally:
        gaussians.clear_time()


def render_full_scene_with_candidate_label(*args, **kwargs):
    """Render the complete scene under one candidate component label."""
    return render_counterfactual_scene(*args, **kwargs)


@torch.no_grad()
def compute_local_affected_region(current_component_alpha, candidate_component_alpha, *, dilation=3, threshold=1.0e-4):
    current = torch.as_tensor(current_component_alpha).float()
    candidate = torch.as_tensor(candidate_component_alpha, device=current.device).float()
    affected = ((current > float(threshold)) | (candidate > float(threshold))).float()[None, None]
    radius = max(0, int(dilation))
    if radius:
        kernel = 2 * radius + 1
        affected = F.max_pool2d(affected, kernel, stride=1, padding=radius)
    return affected[0, 0].bool()


@torch.no_grad()
def compute_counterfactual_rgb_depth_dino_loss(
    rendered,
    target_rgb,
    affected_region,
    *,
    target_depth=None,
    depth_valid=None,
    target_dino=None,
    rgb_weight=1.0,
    depth_weight=0.0,
    dino_weight=0.0,
):
    """Normalized local evidence; unavailable modalities do not vote."""
    rgb = torch.as_tensor(rendered["rgb"])
    target = torch.as_tensor(target_rgb, device=rgb.device, dtype=rgb.dtype)
    region = torch.as_tensor(affected_region, device=rgb.device).bool()
    finite_rgb = torch.isfinite(rgb).all(0) & torch.isfinite(target).all(0)
    rgb_valid = region & finite_rgb
    rgb_loss = (rgb[:, rgb_valid] - target[:, rgb_valid]).abs().mean() if bool(rgb_valid.any()) else rgb.new_tensor(float("nan"))
    depth_loss = rgb.new_tensor(float("nan"))
    if target_depth is not None and rendered.get("depth") is not None:
        predicted_depth = torch.as_tensor(rendered["depth"], device=rgb.device, dtype=rgb.dtype).squeeze()
        depth_target = torch.as_tensor(target_depth, device=rgb.device, dtype=rgb.dtype).squeeze()
        valid = region & torch.isfinite(predicted_depth) & torch.isfinite(depth_target)
        if depth_valid is not None:
            valid &= torch.as_tensor(depth_valid, device=rgb.device).bool().squeeze()
        if bool(valid.any()):
            depth_loss = (predicted_depth[valid] - depth_target[valid]).abs().mean()
    dino_loss = rgb.new_tensor(float("nan"))
    if target_dino is not None and rendered.get("dino") is not None:
        predicted_dino = torch.as_tensor(rendered["dino"], device=rgb.device, dtype=rgb.dtype)
        dino_target = torch.as_tensor(target_dino, device=rgb.device, dtype=rgb.dtype)
        channels = min(predicted_dino.shape[0], dino_target.shape[0])
        valid = region & torch.isfinite(predicted_dino[:channels]).all(0) & torch.isfinite(dino_target[:channels]).all(0)
        if bool(valid.any()):
            dino_loss = (1.0 - (F.normalize(predicted_dino[:channels, valid], dim=0) * F.normalize(dino_target[:channels, valid], dim=0)).sum(0)).mean()
    terms = ((rgb_loss, rgb_weight), (depth_loss, depth_weight), (dino_loss, dino_weight))
    active = [(value, float(weight)) for value, weight in terms if float(weight) > 0 and bool(torch.isfinite(value))]
    total = sum(value * weight for value, weight in active) / max(sum(weight for _, weight in active), 1.0e-8) if active else rgb.new_tensor(float("nan"))
    return {
        "total": total, "rgb": rgb_loss, "depth": depth_loss, "dino": dino_loss,
        "valid_pixels": int(region.sum()), "valid": bool(torch.isfinite(total)),
    }


def compute_rgb_depth_dino_flow_energy(*args, rendered_flow=None, target_flow=None, flow_valid=None, flow_weight=0.0, **kwargs):
    """Counterfactual energy with an optional normalized optical-flow term."""
    result = compute_counterfactual_rgb_depth_dino_loss(*args, **kwargs)
    reference = torch.as_tensor(result["total"])
    flow = reference.new_tensor(float("nan"))
    if rendered_flow is not None and target_flow is not None and float(flow_weight) > 0:
        predicted = torch.as_tensor(rendered_flow, device=reference.device, dtype=reference.dtype)
        target = torch.as_tensor(target_flow, device=reference.device, dtype=reference.dtype)
        valid = torch.isfinite(predicted).all(0) & torch.isfinite(target).all(0)
        if flow_valid is not None:
            valid &= torch.as_tensor(flow_valid, device=reference.device).bool().squeeze()
        if bool(valid.any()):
            flow = torch.linalg.norm(predicted[:, valid] - target[:, valid], dim=0).mean()
    if bool(torch.isfinite(flow)):
        base_weight = sum(float(kwargs.get(name, default)) for name, default in (("rgb_weight", 1.0), ("depth_weight", 0.0), ("dino_weight", 0.0)))
        result["total"] = (result["total"] * base_weight + flow * float(flow_weight)) / max(base_weight + float(flow_weight), 1.0e-8)
        result["valid"] = True
    result["flow"] = flow
    return result


@torch.no_grad()
def update_persistent_render_evidence(state, component, label_energies, *, cycle, momentum=0.5, field_version=None, frame_pair_signature=None):
    """Update tested Gaussian/label rows without resetting evidence elsewhere."""
    evidence = torch.as_tensor(state["evidence"]).detach().cpu().float().clone()
    valid = torch.as_tensor(state.get("valid", torch.zeros_like(evidence, dtype=torch.bool))).detach().cpu().bool().clone()
    age = torch.as_tensor(state.get("last_update", torch.full_like(evidence, -1, dtype=torch.long))).detach().cpu().long().clone()
    ids = torch.as_tensor(component).detach().cpu().long()
    for label, energy in dict(label_energies).items():
        value = float(energy)
        if not torch.isfinite(torch.tensor(value)):
            continue
        old_valid = valid[ids, int(label)]
        previous = evidence[ids, int(label)]
        updated = torch.where(old_valid, float(momentum) * previous + (1.0 - float(momentum)) * value, torch.full_like(previous, value))
        evidence[ids, int(label)] = updated
        valid[ids, int(label)] = True
        age[ids, int(label)] = int(cycle)
    return {
        "evidence": evidence,
        "valid": valid,
        "last_update": age,
        "field_version": field_version,
        "frame_pair_signature": frame_pair_signature,
        "resets_unselected_evidence": False,
    }


@torch.no_grad()
def evaluate_component_all_labels(
    component,
    labels,
    frame_pairs,
    *,
    render_label_fn,
    loss_fn,
):
    """Evaluate all counterfactual labels, retaining direction-wise validity."""
    result = {}
    for label in labels:
        rows = []
        for pair in frame_pairs:
            rendered = render_label_fn(component, int(label), pair)
            loss = loss_fn(rendered, pair)
            if bool(loss.get("valid", False)):
                rows.append(loss)
        summary = {}
        for key in ("total", "rgb", "depth", "dino"):
            values = [float(row[key]) for row in rows if bool(torch.isfinite(torch.as_tensor(row[key])))]
            summary[key] = sum(values) / len(values) if values else None
        summary["valid_comparisons"] = len(rows)
        result[int(label)] = summary
    finite = [(label, row["total"]) for label, row in result.items() if row["total"] is not None]
    winner = min(finite, key=lambda item: item[1])[0] if finite else None
    return {"labels": result, "winner": winner, "valid": bool(finite)}
