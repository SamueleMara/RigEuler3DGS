"""Stage-1 Gaussian-wise discovery of static plus K rigid Killing experts."""

from __future__ import annotations

import json
import math
from pathlib import Path
import torch
from tqdm import tqdm

from utils.eulerian_contrastive_support import (
    aggregate_gaussian_part_weights_to_support,
    assign_gaussians_to_cells,
    build_contrastive_group_ids,
    build_raw_support_state,
    split_mixed_supports_with_dino,
)
from utils.lagrangian_rigid_responsibility import build_gaussian_motion_observations
from utils.kpart_boundary_graph import (
    build_canonical_gaussian_knn,
    combine_dino_motion_affinity,
    compute_dino_affinity,
    compute_dino_cosine_distance,
    compute_local_dino_boundary_saliency,
    extract_uncertain_components,
)
from utils.kpart_counterfactual_rendering import (
    compute_counterfactual_rgb_depth_dino_loss,
    compute_local_affected_region,
    render_counterfactual_scene,
    select_informative_frame_pairs,
)
from utils.kpart_graph_segmentation import (
    assemble_unary_matrix,
    compute_segmentation_energy,
    labels_to_soft_responsibilities,
    normalize_unary_families,
    solve_multilabel_potts,
)
from utils.kpart_motion_signature import (
    compute_gaussian_part_speed_distance,
    compute_gaussian_speed_signature,
    compute_pairwise_motion_compatibility,
    compute_part_speed_signature,
)
from utils.motion_correspondence import MotionCorrespondenceCache
from utils.rigid_killing_motion import (
    aggregate_points_to_rigid_cells,
    integrate_rigid_mode_exact,
    predict_rigid_cell_velocity,
    solve_rigid_killing_mode_all_intervals,
)
from utils.rigid_responsibility_assignment import (
    build_gaussian_dino_graph,
    compute_gaussian_static_rigid_sequence_energies,
)
from utils.rigid_transform_utils import (
    write_quaternion_transport_audit,
)
from utils.temporal_visibility_pretraining import _atomic_torch_save


def initialize_uniform_responsibilities(num_gaussians, num_moving_parts, *, device=None):
    if int(num_gaussians) <= 0 or int(num_moving_parts) <= 0:
        raise ValueError("Stage 1 requires Gaussians and at least one moving part")
    return torch.full(
        (int(num_gaussians), int(num_moving_parts) + 1),
        1.0 / float(int(num_moving_parts) + 1),
        device=device,
        dtype=torch.float32,
    )


@torch.no_grad()
def deterministic_mode_seed_weights(points, num_moving_parts, *, strength=0.35):
    """Soft deterministic spatial diversity used only to break the initial tie."""
    xyz = torch.as_tensor(points).detach().cpu().double()
    k = int(num_moving_parts)
    if k == 1:
        return torch.ones((xyz.shape[0], 1), dtype=torch.float64)
    centered = xyz - xyz.mean(dim=0, keepdim=True)
    covariance = centered.T @ centered / max(1, xyz.shape[0] - 1)
    axis = torch.linalg.eigh(covariance).eigenvectors[:, -1]
    coordinate = centered @ axis
    low, high = torch.quantile(coordinate, torch.tensor([0.05, 0.95], dtype=torch.float64))
    centers = torch.linspace(float(low), float(high), k, dtype=torch.float64)
    scale = max(float(high - low) / max(k - 1, 1), float(coordinate.std()), 1.0e-8)
    seed = torch.softmax(-0.5 * ((coordinate[:, None] - centers[None]) / scale).square(), dim=-1)
    # Mean one preserves each expert's total statistical weight.
    seed = seed * float(k)
    return (1.0 - float(strength)) + float(strength) * seed


@torch.no_grad()
def apply_dino_multiclass_smoothing(responsibilities, dino_graph, *, strength):
    value = torch.as_tensor(responsibilities).float()
    if dino_graph is None or float(strength) <= 0.0:
        return value
    pairs = torch.as_tensor(dino_graph["pairs"], device=value.device).long()
    strong = torch.as_tensor(dino_graph["strong_affinity"], device=value.device).bool()
    affinity = torch.as_tensor(dino_graph["affinity"], device=value.device, dtype=value.dtype)
    pairs, affinity = pairs[strong], affinity[strong]
    if pairs.numel() == 0:
        return value
    total = torch.zeros_like(value)
    denominator = torch.zeros(value.shape[0], device=value.device, dtype=value.dtype)
    total.index_add_(0, pairs[:, 0], affinity[:, None] * value[pairs[:, 1]])
    total.index_add_(0, pairs[:, 1], affinity[:, None] * value[pairs[:, 0]])
    denominator.index_add_(0, pairs[:, 0], affinity)
    denominator.index_add_(0, pairs[:, 1], affinity)
    result = (value + float(strength) * total) / (
        1.0 + float(strength) * denominator[:, None]
    )
    return result / result.sum(dim=-1, keepdim=True).clamp_min(1.0e-8)


@torch.no_grad()
def update_multiclass_responsibilities(
    energies,
    *,
    temperature,
    informative=None,
    previous=None,
    dino_graph=None,
    dino_strength=0.0,
    overlap_strength=0.0,
):
    energy = torch.as_tensor(energies).float()
    if energy.ndim != 2 or energy.shape[1] < 2:
        raise ValueError("energies must be [N,K+1]")
    logits = -energy / max(float(temperature), 1.0e-6)
    if float(overlap_strength) > 0.0 and energy.shape[1] > 2:
        previous_value = (
            torch.as_tensor(previous, device=energy.device, dtype=energy.dtype)
            if previous is not None
            else torch.softmax(logits, dim=-1)
        )
        moving_total = previous_value[:, 1:].sum(dim=-1, keepdim=True)
        logits[:, 1:] -= float(overlap_strength) * (
            moving_total - previous_value[:, 1:]
        )
    raw = torch.softmax(logits, dim=-1)
    result = apply_dino_multiclass_smoothing(
        raw, dino_graph, strength=float(dino_strength)
    )
    if informative is not None:
        valid = torch.as_tensor(informative, device=result.device).bool().reshape(-1)
        fallback = (
            torch.as_tensor(previous, device=result.device, dtype=result.dtype)
            if previous is not None
            else torch.full_like(result, 1.0 / float(result.shape[1]))
        )
        result = torch.where(valid[:, None], result, fallback)
    return {"raw": raw, "responsibilities": result}


@torch.no_grad()
def compute_gaussian_multi_expert_energies(pair_results, *, mad_floor=0.15):
    """Combine K static-vs-rigid evaluations into one auditable K+1 energy."""
    if not pair_results:
        raise ValueError("At least one moving expert is required")
    static_candidates, moving, informative = [], [], []
    for result in pair_results:
        weight = torch.as_tensor(result["energy_weight"]).float()
        valid = weight > 0
        static = torch.as_tensor(result["static_energy"]).detach().cpu().float()
        rigid = torch.as_tensor(result["rigid_energy"]).detach().cpu().float()
        static = static / weight.clamp_min(1.0e-8)
        rigid = rigid / weight.clamp_min(1.0e-8)
        # The perpendicular component first fits a scalar along the candidate
        # projected mode, so Stage 1 does not require its coarse speed magnitude
        # to already be metric. Exact replay residual remains a weak term.
        perpendicular = torch.as_tensor(
            result["speed_perpendicular_residual"]
        ).detach().cpu().float() / max(float(mad_floor), 1.0e-6)
        static_candidates.append(torch.where(valid, static, torch.nan))
        moving.append(torch.where(valid, 0.25 * rigid + 0.75 * perpendicular, torch.nan))
        informative.append(valid)
    static_stack = torch.stack(static_candidates, dim=-1)
    static_energy = torch.nanmedian(static_stack, dim=-1).values
    moving_energy = torch.stack(moving, dim=-1)
    valid_any = torch.stack(informative, dim=-1).any(dim=-1)
    energy = torch.cat((static_energy[:, None], moving_energy), dim=-1)
    energy = torch.where(torch.isfinite(energy), energy, torch.zeros_like(energy))
    # Only energy differences matter; this keeps softmax numerically tame.
    energy -= energy.amin(dim=-1, keepdim=True)
    return {"energies": energy, "informative": valid_any}


@torch.no_grad()
def classify_kpart_responsibilities(responsibilities, *, confidence_threshold):
    value = torch.as_tensor(responsibilities).float()
    confidence, labels = value.max(dim=-1)
    ambiguous = confidence < float(confidence_threshold)
    labels = labels.clone()
    labels[ambiguous] = -1
    entropy = -(value * torch.log(value.clamp_min(1.0e-8))).sum(dim=-1)
    return {
        "labels": labels,
        "confidence": confidence,
        "ambiguous_mask": ambiguous,
        "entropy": entropy,
    }


def check_part_discovery_convergence(
    previous_responsibilities,
    responsibilities,
    *,
    mode_change,
    trajectory_change,
    energy_change,
    responsibility_tolerance=5.0e-3,
    mode_tolerance=1.0e-3,
):
    old = torch.as_tensor(previous_responsibilities).float()
    new = torch.as_tensor(responsibilities).float()
    responsibility_change = float(torch.linalg.norm(new - old, dim=-1).mean())
    old_labels, new_labels = old.argmax(dim=-1), new.argmax(dim=-1)
    class_changes = int((old_labels != new_labels).sum())
    converged = (
        responsibility_change < float(responsibility_tolerance)
        and class_changes == 0
        and float(mode_change) < float(mode_tolerance)
        and float(trajectory_change) < float(mode_tolerance)
        and float(energy_change) < float(responsibility_tolerance)
    )
    return {
        "converged": bool(converged),
        "responsibility_change": responsibility_change,
        "class_changes": class_changes,
    }


def _temperature(cycle, anneal_cycles, start, end):
    if int(anneal_cycles) <= 1 or int(cycle) >= int(anneal_cycles) - 1:
        return float(end)
    fraction = float(cycle) / float(int(anneal_cycles) - 1)
    return float(start) + fraction * (float(end) - float(start))


def _mode_distance(previous, current):
    if previous is None:
        return float("inf")
    a = torch.cat((previous.reference_velocity, previous.angular_mode)).double()
    b = torch.cat((current.reference_velocity, current.angular_mode)).double()
    if torch.dot(a, b) < 0:
        b = -b
    return float(torch.linalg.norm(a - b) / torch.linalg.norm(a).clamp_min(1.0e-12))


def coarse_coordinates(interval_speeds, frame_times):
    speed = torch.as_tensor(interval_speeds).detach().cpu().double()
    time = torch.as_tensor(frame_times).detach().cpu().double()
    return torch.cat((torch.zeros(1, dtype=torch.float64), torch.cumsum(speed * torch.diff(time), 0)))


def _replay(points, rotations, mode, coordinates, frame_index):
    if mode is None or int(frame_index) == 0:
        return points.clone(), rotations.clone()
    result = integrate_rigid_mode_exact(
        points,
        mode.to(device=points.device, dtype=points.dtype),
        coordinates[int(frame_index)].to(points),
        orientations=rotations,
    )
    return result["positions"], result["orientations"]


def _strict_json(value):
    if isinstance(value, dict):
        return {str(key): _strict_json(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_strict_json(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _write_cycle_html(
    points,
    responsibilities,
    classification,
    output,
    *,
    title,
    max_points,
    energies=None,
    dino_graph=None,
    support_ids=None,
    extra_metrics=None,
):
    xyz = torch.as_tensor(points).detach().cpu().float()
    value = torch.as_tensor(responsibilities).detach().cpu().float()
    labels = torch.as_tensor(classification["labels"]).detach().cpu().long()
    entropy = torch.as_tensor(classification["entropy"]).detach().cpu().float()
    ids = torch.arange(xyz.shape[0])
    if xyz.shape[0] > int(max_points):
        ids = torch.linspace(0, xyz.shape[0] - 1, int(max_points)).round().long().unique()
        xyz, value, labels, entropy = xyz[ids], value[ids], labels[ids], entropy[ids]
    colors = ["#9e9e9e", "#377eb8", "#e41a1c", "#4daf4a", "#984ea3", "#ff7f00", "#a65628"]
    marker_colors = ["#bdbdbd" if int(label) < 0 else colors[int(label) % len(colors)] for label in labels]
    hover = [
        f"label={int(labels[i])}<br>entropy={float(entropy[i]):.5f}<br>r={value[i].tolist()}"
        for i in range(xyz.shape[0])
    ]
    traces = [{
        "type": "scatter3d", "mode": "markers",
        "name": "argmax part label", "visible": True,
        "x": xyz[:, 0].tolist(), "y": xyz[:, 1].tolist(), "z": xyz[:, 2].tolist(),
        "marker": {"size": 2.5, "color": marker_colors}, "text": hover, "hoverinfo": "text",
    }]
    metric_names = []
    metrics = []
    for class_index in range(value.shape[1]):
        metric_names.append(f"responsibility class {class_index}")
        metrics.append(value[:, class_index])
    metric_names.append("responsibility entropy")
    metrics.append(entropy)
    if energies is not None:
        energy = torch.as_tensor(energies).detach().cpu().float()[ids]
        best_rigid = energy[:, 1:].amin(dim=-1)
        metric_names.append("static minus best-rigid energy")
        metrics.append(energy[:, 0] - best_rigid)
        if energy.shape[1] > 2:
            ordered = energy[:, 1:].sort(dim=-1).values
            metric_names.append("second-best minus best rigid energy")
            metrics.append(ordered[:, 1] - ordered[:, 0])
    if extra_metrics:
        total_points = torch.as_tensor(points).shape[0]
        for name, metric in extra_metrics.items():
            value_metric = torch.as_tensor(metric).detach().cpu().float().reshape(-1)
            if value_metric.numel() == total_points:
                metric_names.append(str(name))
                metrics.append(value_metric[ids])
    if support_ids is not None:
        metric_names.append("Eulerian contrastive support ID")
        metrics.append(torch.as_tensor(support_ids).detach().cpu().float()[ids])
    for name, metric in zip(metric_names, metrics):
        traces.append({
            "type": "scatter3d", "mode": "markers", "name": name, "visible": False,
            "x": xyz[:, 0].tolist(), "y": xyz[:, 1].tolist(), "z": xyz[:, 2].tolist(),
            "marker": {"size": 2.5, "color": metric.tolist(), "colorscale": "Viridis", "showscale": True},
            "text": hover, "hoverinfo": "text",
        })
    boundary_trace = None
    if dino_graph is not None:
        pairs = torch.as_tensor(dino_graph["pairs"]).detach().cpu().long()
        strong = torch.as_tensor(dino_graph["strong_affinity"]).detach().cpu().bool()
        full_labels = torch.as_tensor(classification["labels"]).detach().cpu().long()
        boundary = pairs[strong & (full_labels[pairs[:, 0]] != full_labels[pairs[:, 1]])]
        if boundary.shape[0] > 1000:
            boundary = boundary[torch.linspace(0, boundary.shape[0] - 1, 1000).round().long().unique()]
        full_xyz = torch.as_tensor(points).detach().cpu().float()
        x, y, z = [], [], []
        for first, second in boundary.tolist():
            x.extend((float(full_xyz[first, 0]), float(full_xyz[second, 0]), None))
            y.extend((float(full_xyz[first, 1]), float(full_xyz[second, 1]), None))
            z.extend((float(full_xyz[first, 2]), float(full_xyz[second, 2]), None))
        boundary_trace = len(traces)
        traces.append({
            "type": "scatter3d", "mode": "lines", "name": "DINO-aligned class boundary",
            "visible": True, "x": x, "y": y, "z": z,
            "line": {"width": 2, "color": "#111111"}, "hoverinfo": "skip",
        })
    buttons = []
    selectable = len(traces) - (1 if boundary_trace is not None else 0)
    for index in range(selectable):
        visible = [False] * len(traces)
        visible[index] = True
        if boundary_trace is not None:
            visible[boundary_trace] = True
        buttons.append({"label": traces[index]["name"], "method": "update", "args": [{"visible": visible}, {"title": f"{title} — {traces[index]['name']}"}]})
    layout = {
        "title": title, "scene": {"aspectmode": "data"},
        "margin": {"l": 0, "r": 0, "t": 75, "b": 0},
        "updatemenus": [{"buttons": buttons, "direction": "down", "x": 0.01, "y": 1.08}],
    }
    document = f"""<!doctype html><html><head><meta charset='utf-8'><script src='https://cdn.plot.ly/plotly-2.35.2.min.js'></script></head><body><div id='plot' style='width:100vw;height:100vh'></div><script>Plotly.newPlot('plot',{json.dumps(traces)},{json.dumps(layout)},{{responsive:true}});</script></body></html>"""
    Path(output).write_text(document, encoding="utf-8")


def _build_coarse_tracks(points, rotations, modes, coordinates, labels, frame_times):
    xyz_frames, rotation_frames, velocity_frames = [], [], []
    part_rotation_frames, part_translation_frames = [], []
    labels_device = torch.as_tensor(labels, device=points.device).long()
    time = torch.as_tensor(frame_times).double()
    q = torch.stack([torch.as_tensor(item).double() for item in coordinates])
    qdot = torch.zeros_like(q)
    if q.shape[1] > 1:
        qdot[:, :-1] = torch.diff(q, dim=-1) / torch.diff(time)[None]
        qdot[:, -1] = qdot[:, -2]
    for frame in range(len(frame_times)):
        xyz, orientation = points.clone(), rotations.clone()
        velocity = torch.zeros_like(points)
        frame_part_rotations, frame_part_translations = [], []
        for part, mode in enumerate(modes, start=1):
            mask = labels_device == part
            result = integrate_rigid_mode_exact(
                points,
                mode.to(device=points.device, dtype=points.dtype),
                q[part - 1, frame].to(points),
                orientations=rotations,
            )
            if bool(mask.any()):
                xyz[mask] = result["positions"][mask]
                orientation[mask] = result["orientations"][mask]
                part_velocity = predict_rigid_cell_velocity(
                    mode.to(device=points.device, dtype=points.dtype),
                    result["positions"],
                    qdot[part - 1, frame].to(points),
                )
                velocity[mask] = part_velocity[mask]
            reference = mode.reference_point.to(points)
            frame_part_rotations.append(result["rotation_quaternion"])
            frame_part_translations.append(
                reference - result["rotation_matrix"] @ reference + result["translation"]
            )
        xyz_frames.append(xyz.cpu())
        rotation_frames.append(orientation.cpu())
        velocity_frames.append(velocity.cpu())
        part_rotation_frames.append(torch.stack(frame_part_rotations).cpu())
        part_translation_frames.append(torch.stack(frame_part_translations).cpu())
    return {
        "times": torch.as_tensor(frame_times, dtype=torch.float32),
        "positions": torch.stack(xyz_frames),
        "rotations": torch.stack(rotation_frames),
        "velocities": torch.stack(velocity_frames),
        "canonical_xyz": points.detach().cpu(),
        "part_labels": torch.as_tensor(labels).cpu(),
        "part_rotation_quaternions": torch.stack(part_rotation_frames),
        "part_translations": torch.stack(part_translation_frames),
    }


@torch.no_grad()
def _physical_and_speed_unaries(pair_results, speeds, *, speed_huber_delta):
    """Direct static/exact-rigid unaries plus full temporal speed signatures."""
    static_candidates, rigid, speed_distance = [], [], []
    signatures, signature_weights, uncertainties = [], [], []
    informative = []
    for result, part_speed in zip(pair_results, speeds):
        energy_weight = torch.as_tensor(result["energy_weight"]).detach().cpu().double()
        valid_energy = energy_weight > 0
        static = torch.as_tensor(result["static_energy"]).detach().cpu().double() / energy_weight.clamp_min(1.0e-8)
        exact_rigid = torch.as_tensor(result["rigid_energy"]).detach().cpu().double() / energy_weight.clamp_min(1.0e-8)
        static_candidates.append(torch.where(valid_energy, static, torch.nan))
        rigid.append(torch.where(valid_energy, exact_rigid, torch.nan))
        observation_weight = (
            torch.as_tensor(result["weight"]).detach().cpu().double()
            * torch.as_tensor(result["excitation"]).detach().cpu().double()
            * torch.as_tensor(result["valid"]).detach().cpu().double()
        )
        signature = compute_gaussian_speed_signature(result["inferred_speed"], observation_weight)
        part_signature = compute_part_speed_signature(part_speed)["signature"]
        distance = compute_gaussian_part_speed_distance(
            signature["signature"], part_signature,
            weights=signature["weight"], uncertainty=signature["uncertainty"],
            huber_delta=float(speed_huber_delta),
        )
        signatures.append(signature["signature"])
        signature_weights.append(signature["weight"])
        uncertainties.append(signature["uncertainty"])
        speed_distance.append(distance["distance"])
        informative.append(valid_energy)
    static = torch.nanmedian(torch.stack(static_candidates, -1), dim=-1).values
    direct = torch.cat((static[:, None], torch.stack(rigid, -1)), -1)
    direct = torch.where(torch.isfinite(direct), direct, torch.nan)
    moving_speed = torch.stack(speed_distance, -1)
    # Static has no Killing coordinate. Zero is a neutral reference; moving
    # labels pay only their normalized temporal incompatibility.
    speed_unary = torch.cat((torch.zeros_like(moving_speed[:, :1]), moving_speed), -1)
    return {
        "direct": direct,
        "speed": speed_unary,
        "signatures": torch.stack(signatures, 1),
        "signature_weights": torch.stack(signature_weights, 1),
        "uncertainty": torch.stack(uncertainties, 1),
        "informative": torch.stack(informative, -1).any(-1),
    }


@torch.no_grad()
def _scene_state_for_labels(points, rotations, labels, modes, coordinates, frame):
    xyz, orientation = points.clone(), rotations.clone()
    material = torch.as_tensor(labels, device=points.device).long()
    for part, mode in enumerate(modes, start=1):
        moved = integrate_rigid_mode_exact(
            points, mode.to(device=points.device, dtype=points.dtype),
            torch.as_tensor(coordinates[part - 1][int(frame)]).to(points),
            orientations=rotations,
        )
        mask = material == part
        xyz[mask], orientation[mask] = moved["positions"][mask], moved["orientations"][mask]
    return xyz, orientation


@torch.no_grad()
def _counterfactual_component_unaries(
    components,
    preliminary_labels,
    base_unary,
    points,
    rotations,
    modes,
    coordinates,
    camera_maps,
    gaussians,
    pipe,
    dataset,
    args,
):
    """Render uncertain components against real frames, all available cameras."""
    from utils.loss_utils import _load_dino_target_for_view
    from utils.motion_correspondence import project_world_points

    render_unary = torch.full_like(torch.as_tensor(base_unary).cpu().float(), float("nan"))
    summaries = []
    background = torch.tensor(
        [1.0, 1.0, 1.0] if bool(dataset.white_background) else [0.0, 0.0, 0.0],
        device=points.device, dtype=points.dtype,
    )
    dino_features = gaussians.get_dino_feature_field if bool(getattr(args, "use_dino_feature_field", False)) else None
    common_cameras = sorted(set.intersection(*(set(mapping) for mapping in camera_maps)))
    if not common_cameras:
        return render_unary, summaries
    representative_camera = common_cameras[0]
    for component_id, component in enumerate(components):
        component = torch.as_tensor(component).long()
        if component.numel() == 0:
            continue
        sample = component[torch.linspace(0, component.numel() - 1, min(64, component.numel())).round().long().unique()].to(points.device)
        current_label = int(torch.mode(torch.as_tensor(preliminary_labels).cpu()[component]).values)
        candidate_hint = int(torch.as_tensor(base_unary).cpu()[component].mean(0).argmin())
        moving_hint = candidate_hint if candidate_hint > 0 else max(1, current_label)
        projected = []
        for frame in range(len(camera_maps)):
            candidate_labels = torch.as_tensor(preliminary_labels).cpu().clone()
            candidate_labels[component] = moving_hint
            xyz, _ = _scene_state_for_labels(points, rotations, candidate_labels, modes, coordinates, frame)
            pixels, _, visible = project_world_points(
                xyz[sample], camera_maps[frame][representative_camera]
            )
            pixels = pixels.clone()
            pixels[~visible] = float("nan")
            projected.append(pixels.cpu())
        frame_pairs = select_informative_frame_pairs(
            torch.stack(projected),
            count=int(args.kpart_render_frame_pairs_per_component),
            min_displacement=float(args.kpart_render_min_projected_displacement),
            max_displacement=float(args.kpart_render_max_projected_displacement),
            bidirectional=True,
        )
        if not frame_pairs:
            continue
        component_mask = torch.zeros(points.shape[0], device=points.device, dtype=torch.bool)
        component_mask[component.to(points.device)] = True
        label_rows = {}
        for candidate_label in range(base_unary.shape[1]):
            modality_rows = []
            candidate_labels = torch.as_tensor(preliminary_labels).cpu().clone()
            candidate_labels[component] = candidate_label
            for pair in frame_pairs:
                target_frame = int(pair["target"])
                current_xyz, current_rotation = _scene_state_for_labels(
                    points, rotations, preliminary_labels, modes, coordinates, target_frame
                )
                candidate_xyz, candidate_rotation = _scene_state_for_labels(
                    points, rotations, candidate_labels, modes, coordinates, target_frame
                )
                for camera_key in common_cameras:
                    camera = camera_maps[target_frame][camera_key]
                    current_render = render_counterfactual_scene(
                        camera, gaussians, pipe, background, current_xyz, current_rotation,
                        component_mask=component_mask, dino_features=dino_features,
                        use_trained_exp=bool(dataset.train_test_exp),
                    )
                    candidate_render = render_counterfactual_scene(
                        camera, gaussians, pipe, background, candidate_xyz, candidate_rotation,
                        component_mask=component_mask, dino_features=dino_features,
                        use_trained_exp=bool(dataset.train_test_exp),
                    )
                    region = compute_local_affected_region(
                        current_render["component_alpha"], candidate_render["component_alpha"]
                    )
                    target_dino = _load_dino_target_for_view(
                        camera, dataset, args, camera.original_image.shape[-2:], device=points.device
                    ) if dino_features is not None else None
                    modality_rows.append(compute_counterfactual_rgb_depth_dino_loss(
                        candidate_render, camera.original_image.to(points.device), region,
                        target_depth=(camera.invdepthmap.to(points.device) if bool(getattr(camera, "depth_reliable", False)) else None),
                        depth_valid=(camera.depth_mask.to(points.device) if bool(getattr(camera, "depth_reliable", False)) else None),
                        target_dino=target_dino,
                        rgb_weight=float(args.kpart_render_rgb_weight),
                        depth_weight=float(args.kpart_render_depth_weight),
                        dino_weight=float(args.kpart_render_dino_weight),
                    ))
            valid = [row for row in modality_rows if row["valid"]]
            label_rows[candidate_label] = {
                key: (sum(float(row[key]) for row in valid if bool(torch.isfinite(torch.as_tensor(row[key])))) / max(1, sum(bool(torch.isfinite(torch.as_tensor(row[key]))) for row in valid)))
                for key in ("total", "rgb", "depth", "dino")
            } if valid else {key: None for key in ("total", "rgb", "depth", "dino")}
            if valid:
                render_unary[component, candidate_label] = label_rows[candidate_label]["total"]
        finite = [(label, row["total"]) for label, row in label_rows.items() if row["total"] is not None]
        summaries.append({
            "component_id": component_id, "gaussian_count": int(component.numel()),
            "current_label": current_label, "candidate_labels": list(range(base_unary.shape[1])),
            "selected_frame_pairs": frame_pairs, "per_label": label_rows,
            "winner": min(finite, key=lambda item: item[1])[0] if finite else None,
        })
    return render_unary, summaries


@torch.no_grad()
def _run_legacy_kpart_rigid_discovery(
    scene,
    gaussians,
    pipe,
    dataset,
    args,
    *,
    log_fn=None,
):
    """Run Stage 1 only; no GT enters fitting, energies, or responsibilities."""
    from utils.unified_dynamic_training import (
        _camera_map,
        _complete_frame_groups,
        _fixed_camera_metadata,
        _ordered_dynamic_cameras,
    )
    from utils.visualize_transported_gaussians_html import write_transported_gaussians_html

    k = int(args.num_moving_parts)
    support_kind = str(args.kpart_support)
    lagrangian = support_kind == "lagrangian"
    cameras = _ordered_dynamic_cameras(scene.getTrainCameras())
    groups = _complete_frame_groups(cameras, args)
    if len(groups) < 2:
        raise RuntimeError("K-part discovery requires at least two complete frames")
    frame_times = [float(item[1]) for item in groups]
    camera_maps = [_camera_map(item[2]) for item in groups]
    points = gaussians.get_canonical_xyz.detach()
    rotations = gaussians.get_rotation.detach().clone()
    responsibilities = initialize_uniform_responsibilities(points.shape[0], k, device=points.device)
    with torch.no_grad():
        gaussians.material_assignment.logits.copy_(torch.log(responsibilities.clamp_min(1.0e-8)))

    dino_features = gaussians.get_dino_feature_field
    dino_graph = boundary_graph = None
    if bool(args.kpart_dino):
        if not torch.is_tensor(dino_features) or dino_features.ndim != 2 or dino_features.shape[0] != points.shape[0]:
            raise RuntimeError("--kpart_dino requires detached per-Gaussian DINO features")
        if bool(args.kpart_graph_segmentation):
            topology = build_canonical_gaussian_knn(
                points,
                neighbors=int(args.kpart_graph_knn),
                max_pairs=int(args.phase1_pair_max_pairs),
                query_chunk_size=int(args.lagrangian_knn_chunk_size),
            )
            dino = compute_dino_cosine_distance(dino_features.detach(), topology["pairs"])
            if bool(args.kpart_dino_local_boundary):
                local = compute_local_dino_boundary_saliency(
                    dino["distance"], topology["pairs"], num_nodes=points.shape[0]
                )
            else:
                scale = max(float(args.kpart_dino_sigma), 1.0e-6)
                local = {
                    "saliency": dino["distance"] / scale,
                    "local_median": torch.zeros_like(dino["distance"]),
                    "local_mad": torch.full_like(dino["distance"], scale),
                }
            dino_affinity = compute_dino_affinity(
                local["saliency"], gamma=float(args.kpart_dino_boundary_gamma)
            )
            boundary_graph = {
                **topology, **dino, **local,
                "dino_affinity": dino_affinity,
            }
            # Preserve the existing support-splitting/HTML interface. The edge
            # threshold is local, not the removed global DINO-distance cutoff.
            dino_graph = {
                "pairs": topology["pairs"], "spatial_distance": topology["spatial_distance"],
                "dino_cosine": dino["cosine"], "dino_distance": dino["distance"],
                "affinity": dino_affinity, "strong_affinity": dino_affinity >= 0.5,
            }
        else:
            dino_graph = build_gaussian_dino_graph(
                points,
                dino_features.detach(),
                neighbors=int(args.kpart_dino_knn),
                sigma=float(args.kpart_dino_sigma),
                max_pairs=int(args.phase1_pair_max_pairs),
                query_chunk_size=int(args.lagrangian_knn_chunk_size),
            )

    grid = support_state = raw_cell_ids = None
    if not lagrangian:
        grid = assign_gaussians_to_cells(
            points,
            int(args.curl_grid_target_cells),
            padding_fraction=float(args.curl_grid_padding),
            min_cells_per_axis=int(args.curl_grid_min_cells_per_axis),
        )
        raw_cell_ids = grid.point_group_ids
        support_state = build_raw_support_state(
            raw_cell_ids, grid.num_cells,
            min_gaussians=int(args.eulerian_min_support_gaussians),
        )

    cache = MotionCorrespondenceCache(dataset, args)
    if not cache.enabled:
        raise RuntimeError("K-part discovery requires rasterized optical-flow observations")
    output = Path(scene.model_path) / "dynamic_debug"
    output.mkdir(parents=True, exist_ok=True)
    if boundary_graph is not None:
        _atomic_torch_save(
            {"schema_version": 1, "pairs": boundary_graph["pairs"], "distance": boundary_graph["distance"]},
            output / "phase1_dino_edge_distance.pt",
        )
        _atomic_torch_save(
            {"schema_version": 1, "saliency": boundary_graph["saliency"], "local_median": boundary_graph["local_median"], "local_mad": boundary_graph["local_mad"]},
            output / "phase1_dino_local_boundary_saliency.pt",
        )
    seed_weights = deterministic_mode_seed_weights(points, k)
    modes = [None] * k
    speeds = [torch.zeros(len(groups) - 1, dtype=torch.float64) for _ in range(k)]
    references = [None] * k
    history, responsibility_history, support_history = [], [], []
    previous_energy = None
    stable_plateau_cycles = 0
    anneal_cycles = int(args.kpart_anneal_cycles)
    discovery_cycles = anneal_cycles + int(args.kpart_plateau_cycles)
    if args.kpart_cycles is not None:
        discovery_cycles = int(args.kpart_cycles)
        anneal_cycles = min(anneal_cycles, discovery_cycles)
    completion_cycles = int(args.kpart_final_completion_passes) if bool(args.kpart_graph_segmentation) else 0
    total_cycles = discovery_cycles + completion_cycles

    progress = tqdm(range(total_cycles), desc=f"Stage 1 K-part {support_kind} discovery", unit="cycle")
    final_energy = None
    graph_payload = None
    ablation_labels = {}
    render_summaries = []
    for cycle in progress:
        old_responsibilities = responsibilities.clone()
        old_modes = list(modes)
        old_q = [coarse_coordinates(speeds[index], frame_times) for index in range(k)]
        if not lagrangian and cycle > 0:
            support_state = split_mixed_supports_with_dino(
                support_state,
                responsibilities.cpu(),
                dino_graph
                if bool(args.eulerian_dino_partition_mixed_supports)
                else None,
                cycle=cycle,
                enabled=bool(args.eulerian_split_mixed_supports),
                confidence_threshold=float(args.kpart_confidence_threshold),
                min_class_fraction=0.10,
                min_gaussians=int(args.eulerian_min_support_gaussians),
            )
        point_group_ids = None if lagrangian else build_contrastive_group_ids(support_state, device=points.device)
        num_supports = points.shape[0] if lagrangian else int(support_state["num_supports"])
        support_part_probability = (
            responsibilities
            if lagrangian
            else aggregate_gaussian_part_weights_to_support(
                support_state, responsibilities.cpu()
            ).to(points)
        )

        cycle_solver = []
        for part in range(k):
            current_q = coarse_coordinates(speeds[part], frame_times)
            observation_states, position_history = [], []
            for target_index in range(1, len(groups)):
                source_index = target_index - 1
                source_xyz, source_rotation = _replay(
                    points, rotations, modes[part], current_q, source_index
                )
                source_cameras = camera_maps[source_index]
                target_cameras = camera_maps[target_index]
                observation_args = (
                    source_xyz,
                    list(source_cameras.values()),
                    list(target_cameras.values()),
                    frame_times[source_index],
                    frame_times[target_index],
                )
                if lagrangian:
                    state = build_gaussian_motion_observations(
                        cache,
                        *observation_args,
                        gaussians=gaussians,
                        pipe=pipe,
                        query_rotations=source_rotation,
                    )
                    support_positions = source_xyz
                else:
                    state = cache.observed_velocity_targets(
                        *observation_args,
                        group_ids=point_group_ids,
                        num_groups=num_supports,
                        gaussians=gaussians,
                        pipe=pipe,
                        query_rotations=source_rotation,
                    )
                    support_positions = aggregate_points_to_rigid_cells(
                        source_xyz, point_group_ids, num_supports
                    )
                observation_states.append(state)
                position_history.append(support_positions.detach().cpu().double())
            positions = torch.stack(position_history)
            velocity = torch.stack([item["velocity"].detach().cpu().double() for item in observation_states])
            confidence = torch.stack([item["confidence"].detach().cpu().double() for item in observation_states])
            valid = torch.stack([item["valid"].detach().cpu().bool() for item in observation_states])
            camera_count = torch.stack([item["distinct_camera_count"].detach().cpu().long() for item in observation_states])
            valid &= camera_count >= int(args.rigid_min_distinct_cameras)
            prior = support_part_probability[:, part + 1].detach().cpu().double()
            if cycle < min(2, anneal_cycles):
                seeded = seed_weights[:, part]
                if not lagrangian:
                    seeded = aggregate_points_to_rigid_cells(
                        seeded, point_group_ids.cpu(), num_supports
                    ).double()
                prior = prior * seeded
            fit_weight = confidence * prior.clamp_min(1.0e-4)[None]
            if references[part] is None:
                support = (fit_weight * valid.double()).sum(dim=0)
                if not bool((support > 0).any()):
                    raise RuntimeError(f"Rigid part {part + 1} has no accepted observations")
                references[part] = positions[0, int(torch.argmax(support))].clone()
            mode, part_speeds, diagnostic = solve_rigid_killing_mode_all_intervals(
                positions,
                velocity,
                weights=fit_weight,
                valid_mask=valid,
                reference_point=references[part],
                max_iters=int(args.rigid_solver_max_iters),
                tolerance=float(args.rigid_solver_tol),
                irls_iters=int(args.rigid_solver_irls_iters),
                huber_delta=float(args.rigid_solver_huber_delta),
                min_valid_cells=int(args.rigid_min_valid_cells),
                angular_eps=float(args.rigid_angular_eps),
            )
            if not mode.valid:
                if modes[part] is None:
                    raise RuntimeError(f"Rigid part {part + 1} produced no valid Killing mode")
                mode, part_speeds = modes[part], speeds[part]
                diagnostic = {**diagnostic, "fallback_to_previous": True}
            modes[part], speeds[part] = mode, part_speeds
            cycle_solver.append({
                "part": part + 1,
                "valid_supports": int(valid.sum()),
                "speed_mean": float(torch.as_tensor(part_speeds).mean()),
                "condition_number": float(mode.diagnostics.get("condition_number", float("inf"))),
                "solver": diagnostic,
            })

        pair_results = [
            compute_gaussian_static_rigid_sequence_energies(
                points, modes[part], speeds[part], frame_times, camera_maps,
                cache, {}, args,
            )
            for part in range(k)
        ]
        legacy_energy = compute_gaussian_multi_expert_energies(
            pair_results, mad_floor=float(args.static_flow_mad_floor)
        )
        temperature = _temperature(
            cycle, anneal_cycles,
            float(args.kpart_temperature_start),
            float(args.kpart_temperature_end),
        )
        legacy_update = update_multiclass_responsibilities(
            legacy_energy["energies"].to(points),
            temperature=temperature,
            informative=legacy_energy["informative"].to(points.device),
            previous=responsibilities,
            dino_graph=dino_graph if bool(args.kpart_dino) else None,
            dino_strength=float(args.kpart_dino_strength),
            overlap_strength=(0.10 if cycle >= max(2, anneal_cycles // 2) else 0.0),
        )
        ablation_labels["current_softmax"] = legacy_update["responsibilities"].argmax(-1).cpu()

        if bool(args.kpart_graph_segmentation):
            if boundary_graph is None:
                raise RuntimeError("Graph K-part segmentation requires the local canonical DINO graph")
            physical = _physical_and_speed_unaries(
                pair_results, speeds,
                speed_huber_delta=float(args.kpart_speed_huber_delta),
            )
            motion = compute_pairwise_motion_compatibility(
                physical["signatures"], boundary_graph["pairs"],
                weights=physical["signature_weights"],
                uncertainty=physical["uncertainty"],
                huber_delta=float(args.kpart_speed_huber_delta),
            )
            combined = combine_dino_motion_affinity(
                boundary_graph["dino_affinity"], motion["distance"]
            )
            speed_valid = torch.isfinite(physical["speed"])
            # Class 0 has no Killing coordinate. Its structural zero is a
            # neutral reference, not a speed observation; including N exact
            # zeros makes the family MAD collapse and rejects every mover.
            speed_valid[:, 0] = False
            normalized = normalize_unary_families({
                "rigid": {"values": physical["direct"], "valid": torch.isfinite(physical["direct"])},
                "speed_signature": {"values": physical["speed"], "valid": speed_valid},
            })
            no_render = assemble_unary_matrix(
                normalized["families"],
                {"rigid": 1.0, "speed_signature": float(args.kpart_speed_signature_weight) if bool(args.kpart_speed_signature) else 0.0},
            )["unary"]
            speed_labels = no_render.argmin(-1)
            ablation_labels["rigid_unary"] = normalized["families"]["rigid"].argmin(-1).cpu()
            preliminary_labels = solve_multilabel_potts(
                no_render, boundary_graph["pairs"], combined["affinity"],
                pairwise_weight=float(args.kpart_boundary_weight),
                initial_labels=responsibilities.argmax(-1).cpu(),
            )
            ablation_labels["speed_signature"] = speed_labels.cpu()
            ablation_labels["dino_graph"] = preliminary_labels.cpu()
            preliminary_soft = torch.softmax(-no_render / max(float(temperature), 1.0e-6), -1)
            ordered = torch.sort(no_render, dim=-1).values
            unary_margin = ordered[:, 1] - ordered[:, 0]
            entropy = -(preliminary_soft * torch.log(preliminary_soft.clamp_min(1.0e-8))).sum(-1)
            boundary_nodes = torch.zeros(points.shape[0], dtype=torch.bool)
            salient_edges = boundary_graph["saliency"] > 1.0
            if bool(salient_edges.any()):
                boundary_nodes[boundary_graph["pairs"][salient_edges].reshape(-1)] = True
            uncertain = (
                (entropy > 0.35)
                | (unary_margin < 0.5)
                | boundary_nodes
                | ~physical["informative"]
            )
            components = extract_uncertain_components(
                boundary_graph["pairs"], combined["affinity"], uncertain,
                max_components=int(args.kpart_render_max_components_per_cycle),
                priority=entropy + (unary_margin + 1.0e-6).reciprocal().clamp_max(20.0),
            )
            render_unary = torch.full_like(no_render, float("nan"))
            cycle_render = []
            if (
                bool(args.kpart_counterfactual_render)
                and int(args.kpart_render_max_components_per_cycle) > 0
                and cycle >= discovery_cycles - 1
            ):
                coordinates_for_render = [coarse_coordinates(speeds[index], frame_times) for index in range(k)]
                render_unary, cycle_render = _counterfactual_component_unaries(
                    components, preliminary_labels, no_render,
                    points, rotations, modes, coordinates_for_render, camera_maps,
                    gaussians, pipe, dataset, args,
                )
                render_summaries.extend({"cycle": cycle + 1, **item} for item in cycle_render)
            all_normalized = normalize_unary_families({
                "rigid": {"values": physical["direct"], "valid": torch.isfinite(physical["direct"])},
                "speed_signature": {"values": physical["speed"], "valid": speed_valid},
                "render": {"values": render_unary, "valid": torch.isfinite(render_unary)},
            })
            assembled = assemble_unary_matrix(
                all_normalized["families"],
                {
                    "rigid": 1.0,
                    "speed_signature": float(args.kpart_speed_signature_weight) if bool(args.kpart_speed_signature) else 0.0,
                    "render": float(args.kpart_render_weight) if bool(args.kpart_counterfactual_render) else 0.0,
                },
            )
            graph_unary = assembled["unary"]
            graph_labels = solve_multilabel_potts(
                graph_unary, boundary_graph["pairs"], combined["affinity"],
                pairwise_weight=float(args.kpart_boundary_weight),
                initial_labels=preliminary_labels,
            )
            graph_energy = compute_segmentation_energy(
                graph_unary, boundary_graph["pairs"], combined["affinity"], graph_labels,
                pairwise_weight=float(args.kpart_boundary_weight),
            )
            responsibilities = labels_to_soft_responsibilities(
                graph_labels, graph_unary,
                temperature=temperature, graph_strength=0.90,
            ).to(points)
            final_energy = {"energies": graph_unary, "informative": physical["informative"]}
            graph_payload = {
                "unary": graph_unary, "labels": graph_labels,
                "motion_distance": motion["distance"], "combined_affinity": combined["affinity"],
                "render_unary": render_unary, "energy": graph_energy,
            }
            if cycle == discovery_cycles - 1:
                ablation_labels["render"] = graph_labels.cpu()
            if cycle >= discovery_cycles:
                ablation_labels["final_completion"] = graph_labels.cpu()
            _atomic_torch_save(
                {"schema_version": 1, "unary": graph_unary, "contributions": assembled["contributions"], "normalization": all_normalized["statistics"]},
                output / f"phase1_graph_unaries_cycle_{cycle + 1:03d}.pt",
            )
            _atomic_torch_save(
                {"schema_version": 1, "labels": graph_labels, "responsibilities": responsibilities.cpu()},
                output / f"phase1_graph_labels_cycle_{cycle + 1:03d}.pt",
            )
            graph_json = {
                **graph_energy,
                "label_changes": int((graph_labels != torch.as_tensor(old_responsibilities).cpu().argmax(-1)).sum()),
                "ambiguous_fraction": float(uncertain.float().mean()),
                "component_render_evaluations": cycle_render,
            }
            (output / f"phase1_graph_energy_cycle_{cycle + 1:03d}.json").write_text(
                json.dumps(_strict_json(graph_json), indent=2, allow_nan=False), encoding="utf-8"
            )
            for part in range(k):
                _atomic_torch_save(
                    {"schema_version": 1, "part": part + 1, "part_speed": torch.as_tensor(speeds[part]), "gaussian_signature": physical["signatures"][:, part], "validity_weight": physical["signature_weights"][:, part], "uncertainty": physical["uncertainty"][:, part]},
                    output / f"phase1_motion_signature_part_{part + 1}.pt",
                )
            _atomic_torch_save(
                {"schema_version": 1, "distance": physical["speed"], "informative": physical["informative"]},
                output / "phase1_gaussian_speed_distance.pt",
            )
            _atomic_torch_save(
                {"schema_version": 1, **motion},
                output / "phase1_pairwise_motion_compatibility.pt",
            )
            _atomic_torch_save(
                {"schema_version": 1, "pairs": boundary_graph["pairs"], "dino_affinity": boundary_graph["dino_affinity"], "motion_affinity": combined["motion_affinity"], "combined_affinity": combined["affinity"]},
                output / "phase1_combined_boundary_affinity.pt",
            )
        else:
            responsibilities = legacy_update["responsibilities"].to(points)
            final_energy = legacy_energy
        classification = classify_kpart_responsibilities(
            responsibilities,
            confidence_threshold=float(args.kpart_confidence_threshold),
        )
        current_q = [coarse_coordinates(speeds[index], frame_times) for index in range(k)]
        mode_change = max(_mode_distance(old_modes[index], modes[index]) for index in range(k))
        trajectory_change = max(
            float(torch.linalg.norm(current_q[index] - old_q[index]) / torch.linalg.norm(old_q[index]).clamp_min(1.0e-8))
            for index in range(k)
        )
        energy_mean = float(final_energy["energies"][final_energy["informative"]].mean()) if bool(final_energy["informative"].any()) else 0.0
        energy_change = float("inf") if previous_energy is None else abs(energy_mean - previous_energy) / max(abs(previous_energy), 1.0e-8)
        convergence = check_part_discovery_convergence(
            old_responsibilities,
            responsibilities,
            mode_change=mode_change,
            trajectory_change=trajectory_change,
            energy_change=energy_change,
            mode_tolerance=max(float(args.rigid_solver_tol), 1.0e-3),
        )
        previous_energy = energy_mean
        confident_labels = classification["labels"]
        counts = [int((confident_labels == label).sum()) for label in range(k + 1)]
        cycle_stats = {
            "cycle": cycle + 1,
            "temperature": temperature,
            "plateau": cycle >= anneal_cycles,
            "responsibility_change": convergence["responsibility_change"],
            "class_changes": convergence["class_changes"],
            "mode_change": mode_change,
            "trajectory_change": trajectory_change,
            "expert_energy_mean": energy_mean,
            "expert_energy_change": energy_change,
            "entropy_mean": float(classification["entropy"].mean()),
            "ambiguous_count": int(classification["ambiguous_mask"].sum()),
            "class_counts": counts,
            "solver": cycle_solver,
            "completion_pass": max(0, cycle - discovery_cycles + 1),
        }
        if graph_payload is not None:
            cycle_stats["graph_energy"] = graph_payload["energy"]
            cycle_stats["render_components_tested"] = len(cycle_render)
        history.append(cycle_stats)
        responsibility_history.append(responsibilities.detach().cpu())
        if not lagrangian:
            support_history.append({
                "cycle": cycle + 1,
                "group_ids": point_group_ids.detach().cpu(),
                "parent_cell_id": support_state["parent_cell_id"].clone(),
                "split_reason": list(support_state["split_reason"]),
                "mixed_support_count": int(support_state.get("mixed_diagnostic", {}).get("mixed_support_count", 0)),
            })
        _write_cycle_html(
            points, responsibilities, classification,
            output / f"phase1_kpart_cycle_{cycle + 1:03d}.html",
            title=f"K-part responsibilities cycle {cycle + 1}",
            max_points=int(args.transported_html_max_points),
            energies=final_energy["energies"],
            dino_graph=dino_graph,
            support_ids=point_group_ids,
        )
        if log_fn is not None:
            logged = {
                "part_discovery/temperature": temperature,
                "part_discovery/responsibility_change": convergence["responsibility_change"],
                "part_discovery/class_changes": convergence["class_changes"],
                "part_discovery/mode_change": mode_change,
                "part_discovery/coarse_trajectory_change": trajectory_change,
                "part_discovery/expert_energy": energy_mean,
                "part_discovery/entropy": float(classification["entropy"].mean()),
                "part_discovery/ambiguous_count": int(classification["ambiguous_mask"].sum()),
            }
            for label, count in enumerate(counts):
                logged[f"part_discovery/class_{label}_count"] = count
            for part, mode in enumerate(modes, start=1):
                logged[f"part_discovery/rigid/part_{part}_axis_norm"] = float(torch.linalg.norm(mode.angular_mode))
                logged[f"part_discovery/rigid/part_{part}_coarse_speed_mean"] = float(torch.as_tensor(speeds[part - 1]).mean())
            if graph_payload is not None:
                graph_labels_logged = graph_payload["labels"]
                same = graph_labels_logged[boundary_graph["pairs"][:, 0]] == graph_labels_logged[boundary_graph["pairs"][:, 1]]
                cut = ~same
                logged.update({
                    "part_discovery/speed_signature/median_pair_distance": float(graph_payload["motion_distance"].median()),
                    "part_discovery/dino_boundary/mean_cut_saliency": float(boundary_graph["saliency"][cut].mean()) if bool(cut.any()) else 0.0,
                    "part_discovery/dino_boundary/mean_same_label_affinity": float(graph_payload["combined_affinity"][same].mean()) if bool(same.any()) else 0.0,
                    "part_discovery/render/components_tested": len(cycle_render),
                    "part_discovery/render/labels_changed_by_render": int((graph_payload["labels"] != preliminary_labels).sum()),
                    "part_discovery/graph/unary_energy": graph_payload["energy"]["unary"],
                    "part_discovery/graph/pairwise_energy": graph_payload["energy"]["pairwise"],
                    "part_discovery/graph/label_changes": int((graph_payload["labels"] != old_responsibilities.cpu().argmax(-1)).sum()),
                })
            log_fn("kpart_discovery", cycle + 1, logged)
        progress.set_postfix(
            energy=f"{energy_mean:.3f}",
            dR=f"{convergence['responsibility_change']:.3g}",
            dmode=f"{mode_change:.3g}",
            entropy=f"{float(classification['entropy'].mean()):.3f}",
            ambiguous=int(classification["ambiguous_mask"].sum()),
        )
        if cycle >= anneal_cycles:
            stable_plateau_cycles = stable_plateau_cycles + 1 if convergence["converged"] else 0
            if stable_plateau_cycles >= 2 and not bool(args.kpart_graph_segmentation):
                break

    classification = classify_kpart_responsibilities(
        responsibilities, confidence_threshold=float(args.kpart_confidence_threshold)
    )
    labels = (
        graph_payload["labels"].to(responsibilities.device)
        if bool(args.kpart_graph_segmentation) and graph_payload is not None
        else classification["labels"]
    )
    if bool(args.kpart_graph_segmentation) and graph_payload is not None:
        classification = {**classification, "labels": labels, "ambiguous_mask": torch.zeros_like(labels, dtype=torch.bool)}
        ablation_labels.setdefault("final_completion", labels.detach().cpu())
    coordinates = [coarse_coordinates(speeds[index], frame_times) for index in range(k)]
    mode_payload = [mode.as_dict() for mode in modes]
    responsibility_payload = {
        "schema_version": 1,
        "responsibilities": responsibilities.detach().cpu(),
        "temperature_schedule": [float(item["temperature"]) for item in history],
        "ground_truth_used_as_algorithm_input": False,
    }
    assignment_payload = {
        "schema_version": 1,
        "part_labels": labels.detach().cpu(),
        "confidence": classification["confidence"].detach().cpu(),
        "ambiguous_mask": classification["ambiguous_mask"].detach().cpu(),
        "entropy": classification["entropy"].detach().cpu(),
        "ground_truth_used_as_algorithm_input": False,
    }
    _atomic_torch_save({"schema_version": 1, "history": responsibility_history}, output / "phase1_kpart_responsibility_history.pt")
    _atomic_torch_save(responsibility_payload, output / "phase1_kpart_responsibility_final.pt")
    _atomic_torch_save(assignment_payload, output / "phase1_kpart_assignment_final.pt")
    _atomic_torch_save({"schema_version": 1, "modes": mode_payload}, output / "phase1_kpart_coarse_rigid_modes.pt")
    _atomic_torch_save({"schema_version": 1, "frame_times": torch.as_tensor(frame_times), "coordinates": torch.stack(coordinates), "interval_speeds": torch.stack([torch.as_tensor(item) for item in speeds])}, output / "phase1_kpart_coarse_trajectories.pt")
    _atomic_torch_save({"schema_version": 1, **final_energy}, output / "phase1_static_vs_part_energies.pt")
    if bool(args.kpart_graph_segmentation):
        _atomic_torch_save(
            {"schema_version": 1, "labels": ablation_labels, "ground_truth_used_as_algorithm_input": False},
            output / "phase1_kpart_ablation_labels.pt",
        )
        (output / "phase1_counterfactual_render_summary.json").write_text(
            json.dumps(_strict_json({"schema_version": 1, "components": render_summaries}), indent=2, allow_nan=False),
            encoding="utf-8",
        )
    if not lagrangian:
        _atomic_torch_save({"schema_version": 1, "raw_cell_ids": raw_cell_ids, "support_state": support_state}, output / "phase1_eulerian_contrastive_supports.pt")
        _atomic_torch_save({"schema_version": 1, "history": support_history}, output / "phase1_eulerian_mixed_support_history.pt")
    summary = {
        "schema_version": 1,
        "support": support_kind,
        "num_moving_parts": k,
        "gaussian_count": int(points.shape[0]),
        "cycles_completed": len(history),
        "anneal_cycles_requested": int(args.kpart_anneal_cycles),
        "plateau_cycles_requested": int(args.kpart_plateau_cycles),
        "confidence_threshold": float(args.kpart_confidence_threshold),
        "class_counts": [int((labels == label).sum()) for label in range(k + 1)],
        "ambiguous_count": int((labels < 0).sum()),
        "history": history,
        "coarse_speed_is_diagnostic_only": True,
        "graph_segmentation": bool(args.kpart_graph_segmentation),
        "speed_signature": bool(args.kpart_speed_signature),
        "local_dino_boundary": bool(args.kpart_dino_local_boundary),
        "counterfactual_rendering": bool(args.kpart_counterfactual_render),
        "final_completion_passes": int(args.kpart_final_completion_passes),
        "ground_truth_used_as_algorithm_input": False,
    }
    (output / "phase1_kpart_summary.json").write_text(
        json.dumps(_strict_json(summary), indent=2, allow_nan=False), encoding="utf-8"
    )
    coarse_tracks = _build_coarse_tracks(points, rotations, modes, coordinates, labels, frame_times)
    coarse_tracks.update({
        "schema_version": 1,
        "motion_membership": responsibilities[:, 1:].sum(dim=-1).cpu(),
        "part_probabilities": responsibilities.cpu(),
        "fixed_cameras": _fixed_camera_metadata(cameras),
    })
    _atomic_torch_save(coarse_tracks, output / "phase1_kpart_coarse_gaussian_tracks.pt")
    write_transported_gaussians_html(
        coarse_tracks,
        output / "phase1_kpart_coarse_motion.html",
        title=f"Stage 1 coarse {support_kind} K-part motion",
        max_points=int(args.transported_html_max_points),
        max_vectors=min(250, int(args.transported_html_max_points)),
    )
    quaternion_audit = write_quaternion_transport_audit(
        coarse_tracks,
        output,
        scales=gaussians.get_scaling.detach().cpu(),
        max_debug_gaussians=min(64, int(args.transported_html_max_points)),
    )
    if log_fn is not None:
        log_fn("quaternion_transport", len(history), {
            "quaternion_transport/max_norm_error": quaternion_audit["max_quaternion_norm_error"],
            "quaternion_transport/spatial_rotation_variance": quaternion_audit["spatial_rigid_rotation_variance_max"],
            "quaternion_transport/part_quaternion_agreement": quaternion_audit["part_quaternion_agreement_error_max"],
            "quaternion_transport/covariance_error": quaternion_audit["covariance_transport_error_max"],
        })
    return {
        "modes": modes,
        "interval_speeds": speeds,
        "coordinates": coordinates,
        "frame_times": frame_times,
        "frame_ids": [int(item[0]) for item in groups],
        "responsibilities": responsibilities,
        "classification": classification,
        "labels": labels,
        "coarse_tracks": coarse_tracks,
        "history": history,
        "ablation_labels": ablation_labels,
    }


def run_kpart_rigid_discovery(scene, gaussians, pipe, dataset, args, *, log_fn=None):
    """Dispatch to the prior-constrained velocity-field Stage 1.

    The default path no longer runs K' oversegmentation/merge.  Diffusion/FM
    priors define the static leftover and the moving clustering domain; the
    fitted velocity fields are only coarse physical evidence for later joint
    estimation.
    """
    if bool(getattr(args, "kpart_velocity_field_segmentation", True)):
        from utils.kpart_velocity_field_discovery import run_velocity_field_kpart_discovery

        return run_velocity_field_kpart_discovery(
            scene, gaussians, pipe, dataset, args, log_fn=log_fn
        )
    return _run_legacy_kpart_rigid_discovery(
        scene, gaussians, pipe, dataset, args, log_fn=log_fn
    )
