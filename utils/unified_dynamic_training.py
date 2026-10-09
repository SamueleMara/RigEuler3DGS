"""Fixed Eulerian curl-strain Phase-1 training."""

from __future__ import annotations

import json
import math
from pathlib import Path
import resource
import time
from typing import Callable, Dict, Optional, Sequence

import numpy as np
import torch
from tqdm import tqdm

from gaussian_renderer import render, render_feature_field
from utils.dynamic_loss_utils import (
    aggregate_point_features_to_cells,
    aggregate_point_probabilities_to_cells,
    assignment_dino_boundary_coherence_loss,
    assignment_spatial_coherence_loss,
    build_lagrangian_knn_pair_graph,
    build_occupied_cell_knn_pair_graph,
    eulerian_assignment_from_kinematics_loss,
    eulerian_velocity_gradient_rigidity_loss,
    get_camera_timestamp,
    integrate_world_angular_velocity_quaternion,
    rigid_killing_gradient_diagnostics,
)
from utils.gradient_motion_evidence import (
    compute_gradient_motion_evidence,
    write_gradient_motion_evidence_html,
)
from utils.loss_utils import (
    _dynamic_root_for_train,
    _load_mask_for_view,
    dino_feature_field_loss,
    l1_loss,
    sam3_feature_field_loss,
    ssim,
)
from utils.motion_correspondence import (
    MotionCorrespondenceCache,
    PairObservation,
    rasterized_projected_velocity_flow_loss,
    velocity_reprojection_rms,
)
from utils.rigid_killing_motion import (
    aggregate_points_to_rigid_cells,
    integrate_rigid_mode_exact,
    predict_rigid_cell_velocity,
    rigid_residuals,
    solve_rigid_assignment,
    solve_rigid_killing_mode_all_intervals,
)
from utils.gaussian_sequence_assignment import (
    DYNAMIC_CANDIDATE,
    DYNAMIC_CORE,
    STATIC_CORE,
    UNKNOWN,
    refine_assignment_with_rigid_hypotheses,
)
from utils.lagrangian_rigid_motion import (
    compute_gaussian_speed_observations,
    compute_lagrangian_gradient_diagnostics,
)
from utils.lagrangian_rigid_responsibility import (
    build_gaussian_motion_observations,
    compute_gaussian_speed_diagnostics,
)
from utils.rigid_responsibility_assignment import (
    build_gaussian_dino_graph,
    classify_final_gaussians,
    compute_gaussian_static_rigid_sequence_energies,
    dino_boundary_metrics,
    smooth_gaussian_responsibility_dino,
    update_gaussian_responsibility,
    write_gaussian_responsibility_html,
)
from utils.eulerian_contrastive_support import (
    aggregate_gaussian_responsibility_to_support,
    assign_gaussians_to_cells,
    build_contrastive_group_ids,
    compute_support_responsibility_stats,
    mixed_cell_diagnostics,
    split_cells_by_dino_components,
    split_supports_by_responsibility,
    summarize_support_purity,
    support_observation_diagnostics,
    summarize_geometric_cell_transitions,
    write_eulerian_support_html,
)
from utils.visualize_transported_gaussians_html import write_transported_gaussians_html
from utils.temporal_visibility_pretraining import _atomic_torch_save
from utils.rigid_transform_utils import (
    write_quaternion_transport_audit,
)


def _strict_json(value):
    if isinstance(value, dict):
        return {str(key): _strict_json(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_strict_json(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _visual_loss(camera, gaussians, pipe, background, dataset, args, opt, stage):
    """Foreground-only visual objective for dynamic motion training.

    Semantic background Gaussians have already been physically removed.  The
    target image must therefore be masked as well; otherwise the optimizer would
    try to explain deliberately absent background pixels by moving foreground
    Gaussians.  Every dynamic view is required to provide a foreground mask.
    """
    package = render(
        camera,
        gaussians,
        pipe,
        background,
        use_trained_exp=dataset.train_test_exp,
    )
    image = package["render"]
    target = camera.original_image.cuda()
    height, width = int(target.shape[-2]), int(target.shape[-1])
    mask = _load_mask_for_view(
        camera, dataset, args, (height, width), device=image.device
    )
    if mask is None:
        raise RuntimeError(
            f"Foreground-only Phase 1 requires a semantic mask for view "
            f"{getattr(camera, 'image_name', '<unknown>')}"
        )
    mask = (mask > 0.5).to(image.dtype)
    if camera.alpha_mask is not None:
        alpha = camera.alpha_mask.to(device=image.device, dtype=image.dtype)
        if alpha.ndim == 3:
            alpha = alpha[0]
        mask = mask * (alpha > 0.5).to(mask.dtype)
    mask3 = mask[None].expand_as(image)
    foreground_denominator = mask3.sum().clamp_min(1.0)
    image_masked = image * mask3
    target_masked = target * mask3
    rgb_l1 = torch.abs(image_masked - target_masked).sum() / foreground_denominator
    # Both tensors are zero outside the object mask.  The mask is fixed, so the
    # SSIM gradient is still routed only through rendered foreground pixels.
    rgb_dssim = 1.0 - ssim(image_masked, target_masked)
    loss = (1.0 - float(opt.lambda_dssim)) * rgb_l1 + float(
        opt.lambda_dssim
    ) * rgb_dssim
    stats = {
        "rgb_l1": float(rgb_l1.detach()),
        "rgb_dssim": float(rgb_dssim.detach()),
        "foreground_pixels": int(mask.sum().detach()),
    }

    depth_weight = float(getattr(opt, "depth_l1_weight_final", 0.0))
    if depth_weight > 0 and getattr(camera, "depth_reliable", False):
        depth = package["depth"]
        target_depth = camera.invdepthmap.cuda()
        depth_mask = camera.depth_mask.cuda()
        if depth_mask.ndim == 3:
            depth_mask = depth_mask[0]
        combined = depth_mask * mask
        depth_denominator = combined.sum().clamp_min(1.0)
        depth_loss = (
            torch.abs(depth - target_depth) * combined
        ).sum() / depth_denominator
        loss = loss + depth_weight * depth_loss
        stats["depth_l1"] = float(depth_loss.detach())
    else:
        stats["depth_l1"] = 0.0

    dino_weight = float(getattr(args, "lambda_dino_feature_field_dynamic", 0.0))
    if bool(getattr(args, "use_dino_feature_field", False)) and dino_weight > 0:
        dino_loss, dino_stats = dino_feature_field_loss(
            viewpoint_cam=camera,
            gaussians=gaussians,
            pipe=pipe,
            background=background,
            dataset=dataset,
            dyn=args,
            stage=stage,
        )
        if dino_loss is not None:
            loss = loss + dino_weight * dino_loss
            stats["dino"] = float(dino_loss.detach())
        stats.update({f"dino/{key}": value for key, value in dino_stats.items()})

    sam3_weight = float(getattr(args, "lambda_sam3_feature_field_dynamic", 0.0))
    if bool(getattr(args, "use_sam3_feature_field", False)) and sam3_weight > 0:
        sam3_loss, sam3_stats = sam3_feature_field_loss(
            viewpoint_cam=camera,
            gaussians=gaussians,
            pipe=pipe,
            background=background,
            dataset=dataset,
            dyn=args,
            stage=stage,
        )
        if sam3_loss is not None:
            loss = loss + sam3_weight * sam3_loss
            stats["sam3"] = float(sam3_loss.detach())
        stats.update({f"sam3/{key}": value for key, value in sam3_stats.items()})
    return loss, stats, package


def _attribution_visual_loss(
    camera, gaussians, pipe, background, dataset, args, opt, stage
):
    """Attribution loss on the full canonical scene.

    This is intentionally separate from the foreground-only Phase-1 training
    loss.  It reproduces the working initializer: RGB + optional depth + DINO
    are evaluated with the complete static scene still present, before semantic
    background compaction.  No optical-flow term, rigidity term, velocity
    update, or integration is part of this probe.
    """
    package = render(
        camera,
        gaussians,
        pipe,
        background,
        use_trained_exp=dataset.train_test_exp,
    )
    image = package["render"]
    if camera.alpha_mask is not None:
        image = image * camera.alpha_mask.cuda()
    target = camera.original_image.cuda()
    rgb_l1 = l1_loss(image, target)
    rgb_dssim = 1.0 - ssim(image, target)
    loss = (1.0 - float(opt.lambda_dssim)) * rgb_l1 + float(
        opt.lambda_dssim
    ) * rgb_dssim
    stats = {
        "rgb_l1": float(rgb_l1.detach()),
        "rgb_dssim": float(rgb_dssim.detach()),
    }

    depth_weight = 0.05
    if (
        depth_weight > 0.0
        and bool(getattr(camera, "depth_reliable", False))
        and package.get("depth", None) is not None
    ):
        depth = package["depth"]
        target_depth = camera.invdepthmap.cuda()
        depth_mask = camera.depth_mask.cuda()
        depth_loss = torch.abs((depth - target_depth) * depth_mask).mean()
        loss = loss + depth_weight * depth_loss
        stats["depth_l1"] = float(depth_loss.detach())
    else:
        stats["depth_l1"] = 0.0

    include_dino = bool(getattr(args, "use_dino_feature_field", False))
    dino_weight = float(args.lambda_dino_feature_field_dynamic)
    if include_dino and dino_weight > 0.0:
        dino_loss, dino_stats = dino_feature_field_loss(
            viewpoint_cam=camera,
            gaussians=gaussians,
            pipe=pipe,
            background=background,
            dataset=dataset,
            dyn=args,
            stage="dynamic",
        )
        if dino_loss is not None:
            loss = loss + dino_weight * dino_loss
            stats["dino"] = float(dino_loss.detach())
        stats.update({f"dino/{key}": value for key, value in dino_stats.items()})

    include_sam3 = bool(getattr(args, "use_sam3_feature_field", False))
    sam3_weight = float(getattr(args, "lambda_sam3_feature_field_dynamic", 0.0))
    if include_sam3 and sam3_weight > 0.0:
        sam3_loss, sam3_stats = sam3_feature_field_loss(
            viewpoint_cam=camera,
            gaussians=gaussians,
            pipe=pipe,
            background=background,
            dataset=dataset,
            dyn=args,
            stage="dynamic",
        )
        if sam3_loss is not None:
            loss = loss + sam3_weight * sam3_loss
            stats["sam3"] = float(sam3_loss.detach())
        stats.update({f"sam3/{key}": value for key, value in sam3_stats.items()})
    return loss, stats, package


def _ordered_dynamic_cameras(cameras: Sequence):
    return sorted(
        cameras,
        key=lambda camera: (
            str(getattr(camera, "dynamic_camera_id", "")),
            int(getattr(camera, "frame_id", 0)),
        ),
    )


def _complete_frame_groups(cameras: Sequence, args):
    """Chronological frames, each containing every physical camera exactly once."""
    expected = sorted(
        {str(getattr(camera, "dynamic_camera_id", "")) for camera in cameras}
    )
    grouped = {}
    for camera in cameras:
        grouped.setdefault(int(getattr(camera, "frame_id", 0)), []).append(camera)
    result = []
    for frame_id in sorted(grouped):
        frame_cameras = sorted(
            grouped[frame_id],
            key=lambda camera: str(getattr(camera, "dynamic_camera_id", "")),
        )
        present = [
            str(getattr(camera, "dynamic_camera_id", "")) for camera in frame_cameras
        ]
        if present != expected:
            raise RuntimeError(
                f"Frame {frame_id} is not a complete multi-view set: expected {expected}, got {present}"
            )
        frame_times = [
            get_camera_timestamp(camera, args.fps, args.time_scale)
            for camera in frame_cameras
        ]
        if max(frame_times) - min(frame_times) > 1.0e-8:
            raise RuntimeError(
                f"Cameras at frame {frame_id} have inconsistent timestamps"
            )
        result.append((frame_id, float(frame_times[0]), frame_cameras))
    return result


def parse_joint_frame_spec(spec, frame_ids):
    """Resolve a --joint_frames spec into positions into the frame-group list.

    Returns strictly increasing positions that always include the first and last
    available frame, so the selection spans the full physical motion whatever
    the spec asks for.
    """
    text = str(spec or "all").strip().lower()
    count = len(frame_ids)
    if count < 2:
        raise ValueError("joint frame selection needs at least two complete frames")
    if text in ("", "all"):
        return list(range(count))
    if text in ("endpoints", "endpoint", "first_last", "2"):
        return [0, count - 1]
    if text.startswith("stride:"):
        stride = int(text.split(":", 1)[1])
        if stride < 1:
            raise ValueError("--joint_frames stride must be a positive integer")
        positions = sorted(set(list(range(0, count, stride)) + [0, count - 1]))
        return positions
    wanted = []
    for token in text.replace(";", ",").split(","):
        token = token.strip()
        if token:
            wanted.append(int(token))
    if not wanted:
        raise ValueError(f"--joint_frames {spec!r} selected no frames")
    lookup = {int(frame): index for index, frame in enumerate(frame_ids)}
    positions = []
    for frame in wanted:
        if frame not in lookup:
            raise ValueError(
                f"--joint_frames requested frame {frame}, which is not a complete "
                f"multi-view frame (available {frame_ids[0]}..{frame_ids[-1]})"
            )
        positions.append(lookup[frame])
    positions = sorted(set(positions + [0, count - 1]))
    return positions


def resolve_joint_frame_selection(groups, args):
    """Subset the frame groups Stage 2 and Stage 3 run on.

    Stage 2 and Stage 3 index the Stage-1 coordinate profile by position in this
    list, so subsetting here reindexes the whole downstream problem: frame 1
    becomes the second selected frame and ``dt`` becomes the physical gap
    between selected frames.  The joint coordinate is a state rather than a path
    integral, so a sparse selection still describes the same joint; only
    correspondence-based objectives care about the gap size.
    """
    frame_ids = [int(item[0]) for item in groups]
    positions = parse_joint_frame_spec(getattr(args, "joint_frames", "all"), frame_ids)
    selected = [groups[index] for index in positions]
    times = [float(item[1]) for item in selected]
    gaps = [times[index + 1] - times[index] for index in range(len(times) - 1)]
    return {
        "positions": positions,
        "groups": selected,
        "frame_ids": [int(item[0]) for item in selected],
        "frame_times": times,
        "spec": str(getattr(args, "joint_frames", "all")),
        "available_frames": len(groups),
        "selected_frames": len(selected),
        "max_gap_seconds": max(gaps) if gaps else 0.0,
        "is_sparse": len(selected) < len(groups),
    }


def subset_stage1_coordinates(stage1, positions):
    """Sample the Stage-1 coordinate profile at the selected frame positions."""
    index = torch.as_tensor(list(positions), dtype=torch.long)
    return [
        torch.as_tensor(coordinate).detach().cpu().double()[index]
        for coordinate in stage1["coordinates"]
    ]


def expand_joint_coordinates(sparse_coordinates, positions, full_profiles):
    """Lift a coordinate solved on a frame subset back onto every frame.

    Stage 2 and Stage 3 may run on as few as two frames, but the final export
    replays the whole video.  The Stage-1 profile is the best available shape
    over all frames, so the expansion keeps that shape and transfers only the
    magnitude the sparse solve recovered: a least-squares scale of the
    full-length profile against the sparse coordinate at the selected
    positions.  With two frames this reduces exactly to matching the endpoint.
    """
    index = torch.as_tensor(list(positions), dtype=torch.long)
    sparse = torch.as_tensor(sparse_coordinates).detach().cpu().double()
    if sparse.ndim == 1:
        sparse = sparse[None]
    expanded, scales = [], []
    for part, profile in enumerate(full_profiles):
        full = torch.as_tensor(profile).detach().cpu().double().reshape(-1)
        full = full - full[0]
        sampled = full[index]
        target = sparse[part] - sparse[part][0]
        denominator = float((sampled * sampled).sum())
        scale = float((sampled * target).sum() / denominator) if denominator > 1.0e-16 else 0.0
        expanded.append(scale * full)
        scales.append(scale)
    return torch.stack(expanded), scales


def _mean_accumulated_stats(sums, counts):
    return {key: sums[key] / max(1, counts[key]) for key in sums}


def _add_stats(sums, counts, values):
    for key, value in values.items():
        if isinstance(value, (int, float)):
            sums[key] = sums.get(key, 0.0) + float(value)
            counts[key] = counts.get(key, 0) + 1


def _fixed_camera_metadata(cameras: Sequence):
    """Serialize one fixed world-space pose and frustum calibration per camera."""
    selected = {}
    for camera in _ordered_dynamic_cameras(cameras):
        name = str(
            getattr(
                camera, "dynamic_camera_id", getattr(camera, "image_name", "camera")
            )
        )
        selected.setdefault(name, camera)
    result = []
    for name, camera in selected.items():
        width, height = int(camera.image_width), int(camera.image_height)
        fx = width / (2.0 * math.tan(0.5 * float(camera.FoVx)))
        fy = height / (2.0 * math.tan(0.5 * float(camera.FoVy)))
        camera_r = (
            camera.R.detach().cpu().numpy()
            if torch.is_tensor(camera.R)
            else np.asarray(camera.R)
        )
        camera_t = (
            camera.T.detach().cpu().numpy()
            if torch.is_tensor(camera.T)
            else np.asarray(camera.T)
        )
        world_to_camera = np.eye(4, dtype=np.float64)
        world_to_camera[:3, :3] = np.asarray(camera_r, dtype=np.float64).T
        world_to_camera[:3, 3] = np.asarray(camera_t, dtype=np.float64)
        camera_to_world = np.linalg.inv(world_to_camera)
        result.append(
            {
                "name": name,
                "width": width,
                "height": height,
                "fx": float(fx),
                "fy": float(fy),
                "cx": 0.5 * width,
                "cy": 0.5 * height,
                "center": camera_to_world[:3, 3].tolist(),
                "r_cw": camera_to_world[:3, :3].tolist(),
            }
        )
    return result


def _camera_map(cameras: Sequence):
    return {str(getattr(camera, "dynamic_camera_id", "")): camera for camera in cameras}


def _phase1_window_ranges(num_frames: int, window_frames: int, stride: int):
    """Target-frame windows; frame 0 is canonical and never a target."""
    if num_frames < 2:
        return []
    width = max(1, int(window_frames))
    requested_step = max(1, int(stride))
    # Every physical target frame must receive direct supervision.  A stride
    # larger than the window width silently left per-frame velocity knots
    # untrained in earlier smoke runs.  Clamp to contiguous/overlapping coverage.
    step = min(requested_step, width)
    last_target = num_frames - 1
    starts = list(range(1, last_target + 1, step))
    windows = []
    for start in starts:
        end = min(last_target + 1, start + width)
        windows.append(tuple(range(start, end)))
    # Ensure the last frame appears in a full-width window when possible.
    final_start = max(1, last_target - width + 1)
    final_window = tuple(range(final_start, last_target + 1))
    if final_window and final_window not in windows:
        windows.append(final_window)
    # Preserve deterministic order and eliminate duplicates.
    unique = []
    seen = set()
    for window in windows:
        if window not in seen:
            unique.append(window)
            seen.add(window)
    return unique


def _assignment_temperature(args, fraction: float) -> float:
    start = float(args.assignment_temperature_start)
    end = float(args.assignment_temperature_end)
    f = min(1.0, max(0.0, float(fraction)))
    # Geometric annealing is smooth in log-temperature and never crosses zero.
    return float(math.exp((1.0 - f) * math.log(start) + f * math.log(end)))


def _straight_through_static_cell_gate(
    cell_probabilities,
    occupied_mask,
    args,
    *,
    point_probabilities=None,
    point_cell_ids=None,
    num_cells=None,
):
    """Exact-static forward gate with gradients to the trainable labels.

    A cell is allowed to become hard static only when it is both high-probability
    static and *pure*: at least ``curl_static_cell_purity`` of its current Gaussian
    centres are individually classified static. This prevents a coarse mixed cell
    from freezing a minority moving part.

    Forward is exactly binary. Backward uses a sigmoid surrogate through the cell
    static probability, so rendering/flow can turn an incorrectly frozen cell back
    into dynamic.
    """
    p_static = cell_probabilities[:, 0]
    threshold = float(args.eulerian_static_gate_threshold)
    temperature = max(float(args.eulerian_static_gate_temperature), 1.0e-4)
    purity_ok = torch.ones_like(p_static, dtype=torch.bool)
    if (
        point_probabilities is not None
        and point_cell_ids is not None
        and num_cells is not None
    ):
        point_static = (point_probabilities[:, 0].detach() >= threshold).to(
            p_static.dtype
        )[:, None]
        fraction_2d, counts = aggregate_point_features_to_cells(
            point_static, point_cell_ids, int(num_cells), normalize=False
        )
        fraction = (
            fraction_2d[:, 0] if fraction_2d is not None else torch.zeros_like(p_static)
        )
        purity_ok = (counts > 0) & (fraction >= float(args.curl_static_cell_purity))
    soft_gate = torch.sigmoid((p_static - threshold) / temperature) * purity_ok.to(
        p_static.dtype
    )
    hard_gate = ((p_static.detach() >= threshold) & purity_ok).to(soft_gate.dtype)
    gate = hard_gate + soft_gate - soft_gate.detach()
    occupied = occupied_mask.detach().to(gate.dtype)
    return gate * occupied, hard_gate * occupied


def _eulerian_strain_replay_source(
    canonical_xyz,
    canonical_rotation,
    curl_grid,
    assignment,
    frame_times,
    source_index: int,
    args,
    temperature: float,
    same_part_floor: float,
    gaussian_dino_features=None,
):
    """Replay Eulerian source geometry with rotation derived from grad(v), not a trained basis."""
    x = canonical_xyz.detach().clone()
    q = canonical_rotation.detach().clone()
    if source_index <= 0:
        return x, q
    cell_centers = curl_grid.cell_centers(device=x.device, dtype=x.dtype)
    soft = assignment.soft_probabilities(float(temperature)).to(x.dtype)
    for interval_index in range(1, source_index + 1):
        t0 = float(frame_times[interval_index - 1])
        t1 = float(frame_times[interval_index])
        dt = t1 - t0
        midpoint = 0.5 * (t0 + t1)
        support = curl_grid.material_support(x)
        ids = support["cell_ids"]
        occupied = support["occupied"]
        max_radius = 0.0
        if float(args.curl_material_neighbor_radius_cells) > 0.0:
            max_radius = float(args.curl_material_neighbor_radius_cells) * float(
                curl_grid.cell_size.max().detach().cpu()
            )
        graph = build_occupied_cell_knn_pair_graph(
            cell_centers,
            occupied,
            neighbors_per_cell=int(args.curl_material_neighbors),
            max_radius=max_radius,
            max_pairs=int(args.phase1_pair_max_pairs),
        )
        cell_prob, _ = aggregate_point_probabilities_to_cells(
            soft, ids, curl_grid.num_cells
        )
        cell_dino, _ = (
            aggregate_point_features_to_cells(
                gaussian_dino_features, ids, curl_grid.num_cells, normalize=True
            )
            if gaussian_dino_features is not None
            else (None, None)
        )
        v_cells_raw = curl_grid.cell_velocities(midpoint)
        static_gate, _ = _straight_through_static_cell_gate(
            cell_prob,
            occupied,
            args,
            point_probabilities=soft,
            point_cell_ids=ids,
            num_cells=curl_grid.num_cells,
        )
        # Exact physical staticness in the forward replay.  The raw curl field is
        # simultaneously driven toward zero in those cells during optimization.
        v_cells = (1.0 - static_gate[:, None]) * v_cells_raw
        grad_state = eulerian_velocity_gradient_rigidity_loss(
            cell_centers,
            v_cells,
            occupied,
            cell_prob,
            graph,
            same_part_floor=float(same_part_floor),
            velocity_scale=float(args.eulerian_velocity_scale),
            min_neighbors=int(args.eulerian_velocity_gradient_min_neighbors),
            relative_ridge=float(args.eulerian_velocity_gradient_ridge),
            static_speed_weight=0.0,
            cell_dino_features=cell_dino,
            dino_boundary_strength=float(args.dino_boundary_rigidity_strength),
            velocity_spatial_robust_delta=float(
                args.velocity_spatial_coherence_robust_delta
            ),
        )
        velocity = curl_grid.gather_cell_velocity(ids, cell_velocity_table=v_cells)
        omega_points = curl_grid.gather_cell_velocity(
            ids, cell_velocity_table=grad_state["omega"]
        )
        q = integrate_world_angular_velocity_quaternion(q, omega_points, dt)
        x = x + dt * velocity
    return x, q


def _observed_velocity_direction_losses(
    predicted_velocity, observation_state, *, velocity_scale=1.0
):
    """Direct magnitude + direction loss against 3-D targets inferred ONLY from observed 2-D flow."""
    target = observation_state["velocity"].detach().to(predicted_velocity)
    confidence = (
        observation_state["confidence"].detach().to(predicted_velocity).clamp_min(0.0)
    )
    valid = observation_state["valid"].detach().to(device=predicted_velocity.device)
    if predicted_velocity.shape != target.shape:
        raise ValueError("predicted/observed velocity tables must have equal shape")
    zero = predicted_velocity.sum() * 0.0
    valid = (
        valid
        & torch.isfinite(target).all(dim=-1)
        & torch.isfinite(predicted_velocity).all(dim=-1)
    )
    if not valid.any():
        return zero, zero, {"observed_flow/valid_velocity_targets": 0}
    w = confidence[valid].clamp_min(1.0e-4)
    scale = max(float(velocity_scale), 1.0e-6)
    diff = predicted_velocity[valid] - target[valid]
    magnitude = (
        w * (diff.square().sum(dim=-1) / (scale * scale))
    ).sum() / w.sum().clamp_min(1.0e-8)
    target_speed = torch.linalg.norm(target[valid], dim=-1)
    adaptive = max(1.0e-5, 0.10 * float(torch.quantile(target_speed.detach(), 0.90)))
    moving = target_speed > adaptive
    if moving.any():
        cos = torch.nn.functional.cosine_similarity(
            predicted_velocity[valid][moving], target[valid][moving], dim=-1, eps=1.0e-8
        ).clamp(-1.0, 1.0)
        wm = w[moving]
        direction = (wm * (1.0 - cos)).sum() / wm.sum().clamp_min(1.0e-8)
        cosine_mean = float((wm * cos.detach()).sum() / wm.sum().clamp_min(1.0e-8))
    else:
        direction = zero
        cosine_mean = 0.0
    return (
        magnitude,
        direction,
        {
            "observed_flow/valid_velocity_targets": int(valid.sum()),
            "observed_flow/velocity_target_speed_median": float(
                target_speed.detach().median()
            ),
            "observed_flow/velocity_target_speed_p90": float(
                torch.quantile(target_speed.detach(), 0.90)
            ),
            "observed_flow/velocity_direction_cosine_mean": cosine_mean,
            "observed_flow/velocity_direction_threshold": adaptive,
            "observed_flow/target_condition_mean": float(
                observation_state["conditioning"][valid].detach().float().mean()
            ),
            "observed_flow/target_observation_count_mean": float(
                observation_state["observation_count"][valid].detach().float().mean()
            ),
        },
    )


def _assignment_regularizers(
    assignment,
    pair_graph,
    args,
    temperature,
    progress_fraction,
    same_part_floor=None,
    dino_features=None,
):
    soft = assignment.soft_probabilities(temperature)
    prior = assignment.asymmetric_prior_loss(
        static_weight=float(args.assignment_prior_static_weight),
        dynamic_weight=float(args.assignment_prior_dynamic_weight),
    )
    spatial = assignment_spatial_coherence_loss(
        soft,
        pair_graph,
        same_part_floor=float(
            args.assignment_same_part_floor
            if same_part_floor is None
            else same_part_floor
        ),
    )
    dino_spatial = assignment_dino_boundary_coherence_loss(
        soft,
        pair_graph,
        dino_features,
    )
    entropy = assignment.entropy_loss(temperature)
    warmup = float(args.assignment_entropy_warmup_fraction)
    entropy_ramp = (
        0.0
        if progress_fraction <= warmup
        else ((progress_fraction - warmup) / max(1.0e-8, 1.0 - warmup))
    )
    prior_scale = (
        1.0
        - progress_fraction
        + progress_fraction * float(args.assignment_prior_final_scale)
    )
    total = (
        float(args.lambda_assignment_prior) * float(prior_scale) * prior
        + float(args.lambda_assignment_spatial) * spatial
        + float(args.lambda_assignment_dino_boundary) * dino_spatial
        + float(args.lambda_assignment_entropy) * float(entropy_ramp) * entropy
    )
    dynamic_probability = 1.0 - soft[:, 0]
    return total, {
        "assignment/prior": float(prior.detach()),
        "assignment/prior_scale": float(prior_scale),
        "assignment/spatial": float(spatial.detach()),
        "assignment/dino_same_material": float(dino_spatial.detach()),
        "assignment/entropy": float(entropy.detach()),
        "assignment/entropy_ramp": float(entropy_ramp),
        "assignment/dynamic_probability_mean": float(
            dynamic_probability.detach().mean()
        ),
        "assignment/dynamic_probability_p10": float(
            torch.quantile(dynamic_probability.detach(), 0.10)
        ),
        "assignment/dynamic_probability_p90": float(
            torch.quantile(dynamic_probability.detach(), 0.90)
        ),
        "assignment/temperature": float(temperature),
    }


def _optimizer_parameters(optimizer):
    if optimizer is None:
        return []
    result = []
    for group in optimizer.param_groups:
        result.extend(group["params"])
    return result


def _annealed_same_part_floor(args, progress_fraction: float) -> float:
    f = min(1.0, max(0.0, float(progress_fraction)))
    start = float(
        getattr(
            args,
            "assignment_same_part_floor_start",
            getattr(args, "assignment_same_part_floor", 0.25),
        )
    )
    end = float(
        getattr(
            args,
            "assignment_same_part_floor_end",
            getattr(args, "assignment_same_part_floor", 0.25),
        )
    )
    return (1.0 - f) * start + f * end


def _phase1_stage_schedule(args):
    result = [("motion", i) for i in range(int(args.phase1_motion_passes))]
    result += [
        ("assignment", i)
        for i in range(int(getattr(args, "phase1_assignment_passes", 1)))
    ]
    return result


def train_eulerian_velocity_gradient_phase1(
    scene,
    gaussians,
    pipe,
    background,
    dataset,
    args,
    opt,
    *,
    visual_loss_fn: Callable,
    log_fn: Optional[Callable[[str, int, Dict], None]] = None,
    correspondence_cache: Optional[MotionCorrespondenceCache] = None,
):
    """Eulerian Phase 1: precise cell-centre velocity + continuum rigid kinematics.

    Trainable motion state
    ----------------------
    * temporal B-spline vector potential A(t) only;
    * v_c(t) = curl_h A(t), hence div_h(v)=0 by construction;
    * persistent Gaussian material assignment logits.

    Derived, NEVER trainable
    ------------------------
    * G_c = grad(v)_c from occupied material neighbours;
    * D_c = 0.5 (G_c + G_c^T), the deformation-rate tensor;
    * Omega_c = 0.5 (G_c - G_c^T), the rigid-rotation tensor;
    * omega_c = 0.5 curl(v)_c.

    No omega, b, rigid center, or joint axis is optimized in this phase.
    """
    representation = str(getattr(gaussians, "motion_representation", ""))
    if representation != "eulerian_curl_strain":
        raise RuntimeError(
            f"train_eulerian_velocity_gradient_phase1 received {representation}"
        )
    if gaussians.curl_grid is None or gaussians.material_assignment is None:
        raise RuntimeError(
            "Eulerian strain mode requires FixedCurlGrid and MaterialAssignmentField"
        )
    if gaussians.motion_optimizer is None or gaussians.membership_optimizer is None:
        raise RuntimeError(
            "Eulerian strain mode requires motion and assignment optimizers"
        )

    cameras = _ordered_dynamic_cameras(scene.getTrainCameras())
    frame_groups = _complete_frame_groups(cameras, args)
    if len(frame_groups) < 2:
        raise RuntimeError("Phase 1 needs at least two synchronized frames")
    frame_times = [float(g[1]) for g in frame_groups]
    deltas = np.diff(np.asarray(frame_times, dtype=np.float64))
    if not np.isfinite(deltas).all() or np.any(deltas <= 0.0):
        raise RuntimeError("Frame timestamps must be finite and strictly increasing")

    canonical_xyz = gaussians.get_canonical_xyz.detach().clone()
    canonical_rotation = gaussians.get_rotation.detach().clone()
    assignment = gaussians.material_assignment
    curl = gaussians.curl_grid
    cell_centers = curl.cell_centers(
        device=canonical_xyz.device, dtype=canonical_xyz.dtype
    )
    material_graph = build_lagrangian_knn_pair_graph(
        canonical_xyz,
        neighbors_per_point=int(args.phase1_pair_neighbors),
        max_pairs=int(args.phase1_pair_max_pairs),
        query_chunk_size=int(getattr(args, "lagrangian_knn_chunk_size", 512)),
    )
    windows = _phase1_window_ranges(
        len(frame_groups),
        int(args.phase1_window_frames),
        int(args.phase1_window_stride),
    )
    # Continuum D=0 physics is active in every pass that updates motion.
    stages = _phase1_stage_schedule(args)
    iterations_per_window = int(args.free_field_iters)
    total_updates = phase1_update_count(cameras, args)
    args.phase1_total_updates = total_updates
    if total_updates <= 0:
        raise RuntimeError("Phase 1 has zero updates")

    output_dir = Path(scene.model_path) / "dynamic_debug"
    output_dir.mkdir(parents=True, exist_ok=True)
    initial_soft = assignment.soft_probabilities(
        float(args.assignment_temperature_start)
    ).detach()
    initial_dynamic = 1.0 - initial_soft[:, 0]
    gaussian_dino_features = gaussians.get_dino_feature_field
    if (
        not torch.is_tensor(gaussian_dino_features)
        or gaussian_dino_features.numel() == 0
        or gaussian_dino_features.shape[0] != canonical_xyz.shape[0]
    ):
        gaussian_dino_features = None
    frame_group_maps = [_camera_map(g[2]) for g in frame_groups]

    projection_optimizer = (
        torch.optim.Adam(
            curl.potential_parameters(), lr=float(args.eulerian_projection_lr)
        )
        if int(args.eulerian_projection_steps) > 0
        and float(args.eulerian_projection_lr) > 0
        else None
    )

    contract = {
        "schema_version": 3,
        "motion_representation": representation,
        "trainable_motion": "temporal_Bspline_vector_potential_A_only",
        "trainable_omega": False,
        "trainable_translation_b": False,
        "trainable_axis_or_center": False,
        "cell_velocity": "v_c(t)=curl_h(A(t))",
        "continuity": "raw A-field satisfies div_h(curl_h(A))=0; optimized hard-static material gate is piecewise during label refinement and raw v_c is strongly driven to zero in static cells",
        "velocity_gradient": "G_c=argmin_G sum_d a_cd ||(v_d-v_c)-G(X_d-X_c)||^2",
        "deformation_tensor": "D_c=0.5*(G_c+G_c^T)",
        "rotation_tensor": "Omega_c=0.5*(G_c-G_c^T)",
        "derived_angular_velocity": "omega_c=0.5*curl(v_c), never optimized",
        "rigidity": "axis-free: minimize ||D_c||_F^2 plus mandatory same-moving-part spatial gradient and local-affine coherence; DO NOT fit omega/b/axis and DO NOT minimize curl magnitude",
        "spatial_coherence": "same-moving-part neighbours minimize ||G_c-G_d||_F^2 and ||(dv)-0.5(G_c+G_d)dx||^2/||dx||^2",
        "omega_coherence": "diagnostic only; derived omega is not optimized during generic Phase 1",
        "gaussian_rotation": "integrate derived omega_c from the same cell velocity field",
        "flow_supervision": "one dense projected-velocity raster loss plus rasterized multiview 3D velocity targets",
        "dino_role": "material_boundary_gate_only: DINO may weaken rigidity only where current motion labels disagree; optical flow supplies displacement",
        "material_evidence": "weak original gradient-motion prior only; static/dynamic/part labels are optimized jointly from the first Phase-1 update",
        "assignment": "persistent trainable material probabilities; hard-static forward gate is straight-through differentiable to labels",
        "static_cells": "if optimized cell P(static) crosses threshold, tracer velocity is exactly zero in forward; A is strongly projected toward v_c=0 there",
        "stages": [stage for stage, _ in stages],
        "window_target_frames": int(args.phase1_window_frames),
        "window_stride": int(args.phase1_window_stride),
        "iterations_per_window": iterations_per_window,
        "frame_times_seconds": frame_times,
        "grid": {
            "cells": int(curl.num_cells),
            "shape": list(map(int, curl.grid_shape)),
            "A_control_knots": int(curl.num_control_knots),
            "A_parameters": int(curl.parameter_count),
        },
        "weights": {
            "projected_velocity_flow": float(
                args.lambda_motion_projected_velocity_flow
            ),
            "observed_velocity": float(args.lambda_observed_flow_velocity),
            "observed_direction": float(args.lambda_observed_flow_direction),
            "strain_rate": float(args.lambda_eulerian_strain_rate),
            "velocity_spatial_coherence": float(
                args.lambda_eulerian_velocity_spatial_coherence
            ),
            "speed_spatial_coherence": float(
                args.lambda_eulerian_speed_spatial_coherence
            ),
            "spatial_gradient_coherence": float(
                args.lambda_eulerian_spatial_gradient_coherence
            ),
            "local_affine_coherence": float(
                args.lambda_eulerian_local_affine_coherence
            ),
            "omega_coherence_legacy": 0.0,
            "static_speed": float(args.lambda_eulerian_static_speed_strain),
            "hard_static_speed": float(args.lambda_eulerian_hard_static_speed),
            "dino_boundary_rigidity_strength": float(
                args.dino_boundary_rigidity_strength
            ),
            "assignment_dino_boundary": float(args.lambda_assignment_dino_boundary),
            "assignment_kinematic": float(args.lambda_eulerian_assignment_kinematic),
            "curl_temporal_acceleration": float(args.lambda_curl_temporal_acceleration),
        },
    }
    with (output_dir / "phase1_eulerian_velocity_gradient_contract.json").open(
        "w", encoding="utf-8"
    ) as stream:
        json.dump(contract, stream, indent=2, allow_nan=False)

    print(
        "[Eulerian velocity-gradient rigidity] "
        f"cells={curl.num_cells} shape={curl.grid_shape} A_mode={curl.temporal_mode} "
        f"A_controls={curl.num_control_knots} A_params={curl.parameter_count} "
        "trainable_omega=0 trainable_b=0 trainable_axis=0 "
        f"initial_Pdyn={float(initial_dynamic.mean()):.3f}"
    )

    progress = tqdm(
        total=total_updates,
        desc="Phase 1 Eulerian: observed v_c + generic spatial coherence",
        unit="update",
        dynamic_ncols=True,
    )
    update_index = 0
    audit_history = []

    for global_pass_index, (stage_name, _) in enumerate(stages):
        train_motion = stage_name == "motion"
        # Labels are contextual variables, not a precomputed initialization.
        # They receive gradients from rendering/flow through the straight-through
        # static gate during every stage; the assignment-only pass simply freezes A.
        train_assignment = True
        pass_windows = (
            windows if global_pass_index % 2 == 0 else list(reversed(windows))
        )
        for window_index, target_indices in enumerate(pass_windows):
            observed_target_cache = {}
            for local_index in range(iterations_per_window):
                update_index += 1
                progress_fraction = (update_index - 1) / max(1, total_updates - 1)
                temperature = _assignment_temperature(args, progress_fraction)
                same_floor = _annealed_same_part_floor(args, progress_fraction)
                gaussians.motion_optimizer.zero_grad(set_to_none=True)
                gaussians.membership_optimizer.zero_grad(set_to_none=True)
                gaussians.clear_time()

                aggregate_stats, aggregate_counts = {}, {}
                frame_count = len(target_indices)

                for target_index in target_indices:
                    source_time = float(frame_times[target_index - 1])
                    target_time = float(frame_times[target_index])
                    dt = target_time - source_time
                    midpoint = 0.5 * (source_time + target_time)
                    source_by_camera = frame_group_maps[target_index - 1]
                    target_by_camera = frame_group_maps[target_index]

                    x_start, q_start = _eulerian_strain_replay_source(
                        canonical_xyz,
                        canonical_rotation,
                        curl,
                        assignment,
                        frame_times,
                        target_index - 1,
                        args,
                        temperature,
                        same_floor,
                        gaussian_dino_features=gaussian_dino_features,
                    )
                    x_start = x_start.detach()
                    q_start = q_start.detach()
                    support = curl.material_support(x_start)
                    source_cell_ids = support["cell_ids"].detach()
                    occupied = support["occupied"].detach()
                    max_radius = 0.0
                    if float(args.curl_material_neighbor_radius_cells) > 0.0:
                        max_radius = float(
                            args.curl_material_neighbor_radius_cells
                        ) * float(curl.cell_size.max().detach().cpu())
                    cell_graph = build_occupied_cell_knn_pair_graph(
                        cell_centers,
                        occupied,
                        neighbors_per_cell=int(args.curl_material_neighbors),
                        max_radius=max_radius,
                        max_pairs=int(args.phase1_pair_max_pairs),
                    )

                    soft_prob = assignment.soft_probabilities(temperature).to(
                        canonical_xyz.dtype
                    )
                    cell_prob, _ = aggregate_point_probabilities_to_cells(
                        soft_prob, source_cell_ids, curl.num_cells
                    )
                    cell_dino, _ = (
                        aggregate_point_features_to_cells(
                            gaussian_dino_features,
                            source_cell_ids,
                            curl.num_cells,
                            normalize=True,
                        )
                        if gaussian_dino_features is not None
                        else (None, None)
                    )

                    # v8 static/dynamic coupling.  The optimized cell label controls
                    # an exact forward static gate. Forward is binary (static means
                    # zero tracer speed), while backward uses a sigmoid surrogate so
                    # RGB/depth/DINO/flow can flip a wrong static decision.
                    static_gate, static_gate_hard = _straight_through_static_cell_gate(
                        cell_prob,
                        occupied,
                        args,
                        point_probabilities=soft_prob,
                        point_cell_ids=source_cell_ids,
                        num_cells=curl.num_cells,
                    )
                    v_cells_raw = curl.cell_velocities(midpoint)
                    v_cells = (1.0 - static_gate[:, None]) * v_cells_raw
                    velocity_end = curl.gather_cell_velocity(
                        source_cell_ids, cell_velocity_table=v_cells
                    )
                    x_end = x_start + dt * velocity_end

                    grad_state = eulerian_velocity_gradient_rigidity_loss(
                        cell_centers,
                        v_cells,
                        occupied,
                        cell_prob,
                        cell_graph,
                        same_part_floor=same_floor,
                        velocity_scale=float(args.eulerian_velocity_scale),
                        min_neighbors=int(
                            args.eulerian_velocity_gradient_min_neighbors
                        ),
                        relative_ridge=float(args.eulerian_velocity_gradient_ridge),
                        static_speed_weight=1.0,
                        cell_dino_features=cell_dino,
                        dino_boundary_strength=float(
                            args.dino_boundary_rigidity_strength
                        ),
                        velocity_spatial_robust_delta=float(
                            args.velocity_spatial_coherence_robust_delta
                        ),
                    )
                    omega_cells = grad_state["omega"]
                    omega_points = curl.gather_cell_velocity(
                        source_cell_ids, cell_velocity_table=omega_cells
                    )
                    q_end = integrate_world_angular_velocity_quaternion(
                        q_start, omega_points, dt
                    )

                    # Do not let the motion optimizer escape a static decision by
                    # changing the labels: the hard mask is detached here.  The
                    # assignment itself is changed only by the contextual data and
                    # assignment losses.
                    hard_static_weight = static_gate_hard.detach() * occupied.to(
                        v_cells_raw.dtype
                    )
                    hard_static_loss = (
                        hard_static_weight
                        * (
                            torch.linalg.norm(v_cells_raw, dim=-1)
                            / max(float(args.eulerian_velocity_scale), 1.0e-6)
                        ).square()
                    ).sum() / hard_static_weight.sum().clamp_min(1.0e-8)

                    if train_motion:
                        physics_loss = (
                            float(args.lambda_eulerian_strain_rate)
                            * grad_state["strain"]
                            + float(args.lambda_eulerian_velocity_spatial_coherence)
                            * grad_state["velocity_spatial_coherence"]
                            + float(args.lambda_eulerian_speed_spatial_coherence)
                            * grad_state["speed_spatial_coherence"]
                            + float(args.lambda_eulerian_spatial_gradient_coherence)
                            * grad_state["spatial_gradient_coherence"]
                            + float(args.lambda_eulerian_local_affine_coherence)
                            * grad_state["local_affine_coherence"]
                            + float(args.lambda_eulerian_static_speed_strain)
                            * grad_state["static_speed"]
                            + float(args.lambda_eulerian_hard_static_speed)
                            * hard_static_loss
                        )
                    else:
                        physics_loss = v_cells.sum() * 0.0

                    observed_motion_loss = x_end.sum() * 0.0
                    assignment_kinematic = x_end.sum() * 0.0
                    observed_stats = {}
                    observed_state = None
                    if (
                        correspondence_cache is not None
                        and correspondence_cache.enabled
                    ):
                        refresh = int(
                            getattr(args, "observed_flow_target_refresh_iters", 5)
                        )
                        cache_key = int(target_index)
                        if (
                            cache_key not in observed_target_cache
                            or (local_index % refresh) == 0
                        ):
                            observed_target_cache[cache_key] = (
                                correspondence_cache.observed_velocity_targets(
                                    x_start.detach(),
                                    list(source_by_camera.values()),
                                    list(target_by_camera.values()),
                                    source_time,
                                    target_time,
                                    group_ids=source_cell_ids.detach(),
                                    num_groups=curl.num_cells,
                                    gaussians=gaussians,
                                    pipe=pipe,
                                    query_rotations=q_start.detach(),
                                )
                            )
                        observed_state = observed_target_cache[cache_key]
                        obs_mag, obs_dir, obs_stats = (
                            _observed_velocity_direction_losses(
                                v_cells,
                                observed_state,
                                velocity_scale=float(args.eulerian_velocity_scale),
                            )
                        )
                        if train_motion:
                            observed_motion_loss = (
                                float(args.lambda_observed_flow_velocity) * obs_mag
                                + float(args.lambda_observed_flow_direction) * obs_dir
                            )
                        if train_assignment:
                            assignment_kinematic, assignment_k_stats = (
                                eulerian_assignment_from_kinematics_loss(
                                    cell_centers,
                                    v_cells,
                                    cell_prob,
                                    occupied,
                                    cell_graph,
                                    grad_state,
                                    observed_state=observed_state,
                                    same_part_floor=0.0,
                                    velocity_scale=float(args.eulerian_velocity_scale),
                                    cell_dino_features=cell_dino,
                                    dino_boundary_strength=float(
                                        args.dino_boundary_rigidity_strength
                                    ),
                                    flow_noise_px=float(args.multiview_flow_noise_px),
                                    flow_model_margin_px=float(
                                        args.multiview_flow_model_margin_px
                                    ),
                                    flow_model_temperature_px=float(
                                        args.multiview_flow_model_temperature_px
                                    ),
                                    flow_fit_max_px=float(
                                        args.observed_flow_velocity_max_reprojection_px
                                    ),
                                )
                            )
                        else:
                            assignment_k_stats = {
                                "assignment_static_observed": assignment_kinematic,
                                "assignment_dynamic_observed": assignment_kinematic,
                                "assignment_kinematic_boundary": assignment_kinematic,
                            }
                        observed_stats = {
                            **obs_stats,
                            "observed_flow/loss_velocity": float(obs_mag.detach()),
                            "observed_flow/loss_direction": float(obs_dir.detach()),
                            "assignment/kinematic": float(
                                assignment_kinematic.detach()
                            ),
                            "assignment/static_observed": float(
                                assignment_k_stats[
                                    "assignment_static_observed"
                                ].detach()
                            ),
                            "assignment/dynamic_observed": float(
                                assignment_k_stats[
                                    "assignment_dynamic_observed"
                                ].detach()
                            ),
                            "assignment/kinematic_boundary": float(
                                assignment_k_stats[
                                    "assignment_kinematic_boundary"
                                ].detach()
                            ),
                        }

                    gaussians.set_free_working_state(
                        x_end,
                        target_time,
                        velocity_override=velocity_end,
                        cell_ids_override=source_cell_ids,
                        rotation_override=q_end,
                    )
                    camera_loss = x_end.sum() * 0.0
                    frame_stats, frame_counts = {}, {}
                    for camera_key in sorted(target_by_camera):
                        source_camera = source_by_camera[camera_key]
                        target_camera = target_by_camera[camera_key]
                        visual, visual_stats, _ = visual_loss_fn(
                            target_camera,
                            gaussians,
                            pipe,
                            background,
                            dataset,
                            args,
                            opt,
                            stage="free",
                        )
                        flow_loss = x_end.sum() * 0.0
                        flow_stats = {
                            "projected_velocity_flow": 0.0,
                            "projected_velocity_flow_total": 0.0,
                            "projected_velocity_flow_valid": 0,
                            "projected_velocity_flow_weight_sum": 0.0,
                        }
                        if (
                            correspondence_cache is not None
                            and correspondence_cache.enabled
                        ):
                            observation = correspondence_cache.get(
                                source_camera, target_camera, x_end.device
                            )
                            flow_loss, flow_stats = (
                                rasterized_projected_velocity_flow_loss(
                                    source_camera,
                                    target_camera,
                                    gaussians,
                                    pipe,
                                    x_start,
                                    v_cells,
                                    observation,
                                    args,
                                    delta_time=dt,
                                    query_rotation=q_start,
                                    group_ids=source_cell_ids.detach(),
                                )
                            )
                        camera_loss = camera_loss + visual + flow_loss
                        _add_stats(frame_stats, frame_counts, visual_stats)
                        _add_stats(frame_stats, frame_counts, flow_stats)
                    camera_loss = camera_loss / float(max(1, len(target_by_camera)))

                    frame_total = (
                        float(args.phase1_data_gradient_scale) * camera_loss
                        + physics_loss
                        + observed_motion_loss
                        + (
                            float(args.lambda_eulerian_assignment_kinematic)
                            * assignment_kinematic
                            if train_assignment
                            else assignment_kinematic * 0.0
                        )
                    ) / float(frame_count)
                    frame_total.backward()

                    speed = torch.linalg.norm(velocity_end.detach(), dim=-1)
                    omega_norm = torch.linalg.norm(omega_points.detach(), dim=-1)
                    D_norm = torch.linalg.norm(
                        grad_state["deformation"].detach(), dim=(-2, -1)
                    )
                    valid_G = grad_state["valid"].detach()
                    local_stats = {
                        **_mean_accumulated_stats(frame_stats, frame_counts),
                        **observed_stats,
                        "loss/data_camera_mean": float(camera_loss.detach()),
                        "physics/strain_rate": float(grad_state["strain"].detach()),
                        "physics/velocity_spatial_coherence": float(
                            grad_state["velocity_spatial_coherence"].detach()
                        ),
                        "physics/speed_spatial_coherence": float(
                            grad_state["speed_spatial_coherence"].detach()
                        ),
                        "physics/velocity_spatial_pairs": int(
                            grad_state["velocity_spatial_valid"].sum().detach()
                        ),
                        "physics/spatial_gradient_coherence": float(
                            grad_state["spatial_gradient_coherence"].detach()
                        ),
                        "physics/local_affine_coherence": float(
                            grad_state["local_affine_coherence"].detach()
                        ),
                        "physics/omega_coherence_diag": float(
                            grad_state["omega_coherence"].detach()
                        ),
                        "physics/static_speed": float(
                            grad_state["static_speed"].detach()
                        ),
                        "physics/hard_static_speed": float(hard_static_loss.detach()),
                        "physics/hard_static_cells": int(
                            (static_gate_hard > 0.5).sum().item()
                        ),
                        "assignment/cell_static_probability_mean": float(
                            cell_prob[:, 0][occupied].detach().mean()
                        )
                        if occupied.any()
                        else 0.0,
                        "physics/local_rigid_residual_diag": float(
                            grad_state["local_rigid_residual"].detach()
                        ),
                        "physics/gradient_valid_cells": int(valid_G.sum()),
                        "physics/deformation_norm_median": float(
                            D_norm[valid_G].median()
                        )
                        if valid_G.any()
                        else 0.0,
                        "physics/deformation_norm_max": float(D_norm[valid_G].max())
                        if valid_G.any()
                        else 0.0,
                        "motion/speed_median": float(speed.median()),
                        "motion/speed_p90": float(torch.quantile(speed, 0.90)),
                        "motion/speed_max": float(speed.max()),
                        "motion/derived_omega_median": float(omega_norm.median()),
                        "motion/derived_omega_max": float(omega_norm.max()),
                        "physics/occupied_cells": int(occupied.sum()),
                        "physics/cell_pairs": int(cell_graph.count),
                        "physics/same_part_floor": float(same_floor),
                        "schedule/delta_time": dt,
                        "grid/outside_points": int(support["outside_points"]),
                    }
                    div = curl.continuity_residual(midpoint)[occupied]
                    local_stats["curl/continuity_abs_max"] = (
                        float(div.detach().abs().max()) if div.numel() else 0.0
                    )
                    _add_stats(aggregate_stats, aggregate_counts, local_stats)
                    gaussians.clear_time()

                assignment_loss, assignment_stats = _assignment_regularizers(
                    assignment,
                    material_graph,
                    args,
                    temperature,
                    progress_fraction,
                    same_part_floor=same_floor,
                    dino_features=gaussian_dino_features,
                )
                curl_temporal = curl.temporal_acceleration_loss()
                global_loss = (
                    assignment_loss if train_assignment else assignment_loss * 0.0
                ) + (
                    float(args.lambda_curl_temporal_acceleration) * curl_temporal
                    if train_motion
                    else curl_temporal * 0.0
                )
                global_loss.backward()

                motion_parameters = _optimizer_parameters(gaussians.motion_optimizer)
                membership_parameters = _optimizer_parameters(
                    gaussians.membership_optimizer
                )
                motion_grad = (
                    torch.nn.utils.clip_grad_norm_(
                        motion_parameters, float(args.curl_grid_grad_clip)
                    )
                    if motion_parameters and float(args.curl_grid_grad_clip) > 0
                    else torch.tensor(0.0)
                )
                assignment_grad = (
                    torch.nn.utils.clip_grad_norm_(membership_parameters, 2.0)
                    if membership_parameters
                    else torch.tensor(0.0)
                )
                if train_motion:
                    gaussians.motion_optimizer.step()
                if train_assignment:
                    gaussians.membership_optimizer.step()
                    with torch.no_grad():
                        assignment.logits.clamp_(-10.0, 10.0)

                # Extra A-only continuum projection: keep the observed v_c target,
                # suppress D, and make derived omega spatially coherent.  No fitted
                # rigid basis or axis appears anywhere in this projection.
                projection_value = 0.0
                if train_motion and projection_optimizer is not None:
                    for _projection_step in range(int(args.eulerian_projection_steps)):
                        projection_optimizer.zero_grad(set_to_none=True)
                        projection_total = curl.potential_x.sum() * 0.0
                        for target_index in target_indices:
                            source_time = float(frame_times[target_index - 1])
                            target_time = float(frame_times[target_index])
                            midpoint = 0.5 * (source_time + target_time)
                            x_source, _ = _eulerian_strain_replay_source(
                                canonical_xyz,
                                canonical_rotation,
                                curl,
                                assignment,
                                frame_times,
                                target_index - 1,
                                args,
                                temperature,
                                same_floor,
                                gaussian_dino_features=gaussian_dino_features,
                            )
                            support_p = curl.material_support(x_source.detach())
                            ids_p = support_p["cell_ids"].detach()
                            occupied_p = support_p["occupied"].detach()
                            max_radius = 0.0
                            if float(args.curl_material_neighbor_radius_cells) > 0.0:
                                max_radius = float(
                                    args.curl_material_neighbor_radius_cells
                                ) * float(curl.cell_size.max().detach().cpu())
                            graph_p = build_occupied_cell_knn_pair_graph(
                                cell_centers,
                                occupied_p,
                                neighbors_per_cell=int(args.curl_material_neighbors),
                                max_radius=max_radius,
                                max_pairs=int(args.phase1_pair_max_pairs),
                            )
                            point_probs_p = (
                                assignment.soft_probabilities(temperature)
                                .detach()
                                .to(canonical_xyz.dtype)
                            )
                            probs_p, _ = aggregate_point_probabilities_to_cells(
                                point_probs_p, ids_p, curl.num_cells
                            )
                            dino_p, _ = (
                                aggregate_point_features_to_cells(
                                    gaussian_dino_features,
                                    ids_p,
                                    curl.num_cells,
                                    normalize=True,
                                )
                                if gaussian_dino_features is not None
                                else (None, None)
                            )
                            # Projection uses the current hard static decision, but
                            # assignment is detached here: this optimizer is A-only.
                            static_gate_p, static_hard_p = (
                                _straight_through_static_cell_gate(
                                    probs_p,
                                    occupied_p,
                                    args,
                                    point_probabilities=point_probs_p,
                                    point_cell_ids=ids_p,
                                    num_cells=curl.num_cells,
                                )
                            )
                            static_gate_p = static_gate_p.detach()
                            static_hard_p = static_hard_p.detach()
                            v_p_raw = curl.cell_velocities(midpoint)
                            v_p = (1.0 - static_gate_p[:, None]) * v_p_raw
                            gs_p = eulerian_velocity_gradient_rigidity_loss(
                                cell_centers,
                                v_p,
                                occupied_p,
                                probs_p,
                                graph_p,
                                same_part_floor=same_floor,
                                velocity_scale=float(args.eulerian_velocity_scale),
                                min_neighbors=int(
                                    args.eulerian_velocity_gradient_min_neighbors
                                ),
                                relative_ridge=float(
                                    args.eulerian_velocity_gradient_ridge
                                ),
                                static_speed_weight=1.0,
                                cell_dino_features=dino_p,
                                dino_boundary_strength=float(
                                    args.dino_boundary_rigidity_strength
                                ),
                                velocity_spatial_robust_delta=float(
                                    args.velocity_spatial_coherence_robust_delta
                                ),
                            )
                            hard_w = static_hard_p * occupied_p.to(v_p_raw.dtype)
                            hard_static_p = (
                                hard_w
                                * (
                                    torch.linalg.norm(v_p_raw, dim=-1)
                                    / max(float(args.eulerian_velocity_scale), 1.0e-6)
                                ).square()
                            ).sum() / hard_w.sum().clamp_min(1.0e-8)
                            p_loss = (
                                float(args.lambda_eulerian_strain_rate) * gs_p["strain"]
                                + float(args.lambda_eulerian_velocity_spatial_coherence)
                                * gs_p["velocity_spatial_coherence"]
                                + float(args.lambda_eulerian_speed_spatial_coherence)
                                * gs_p["speed_spatial_coherence"]
                                + float(args.lambda_eulerian_spatial_gradient_coherence)
                                * gs_p["spatial_gradient_coherence"]
                                + float(args.lambda_eulerian_local_affine_coherence)
                                * gs_p["local_affine_coherence"]
                                + float(args.lambda_eulerian_static_speed_strain)
                                * gs_p["static_speed"]
                                + float(args.lambda_eulerian_hard_static_speed)
                                * hard_static_p
                            )
                            obs_p = observed_target_cache.get(int(target_index), None)
                            if (
                                obs_p is not None
                                and obs_p["velocity"].shape[0] == v_p.shape[0]
                                and bool(obs_p["valid"].any())
                            ):
                                obs_mag_p, obs_dir_p, _ = (
                                    _observed_velocity_direction_losses(
                                        v_p,
                                        obs_p,
                                        velocity_scale=float(
                                            args.eulerian_velocity_scale
                                        ),
                                    )
                                )
                                p_loss = p_loss + (
                                    float(args.lambda_observed_flow_velocity)
                                    * obs_mag_p
                                    + float(args.lambda_observed_flow_direction)
                                    * obs_dir_p
                                )
                            projection_total = projection_total + p_loss / float(
                                frame_count
                            )
                        projection_total.backward()
                        torch.nn.utils.clip_grad_norm_(
                            curl.potential_parameters(), float(args.curl_grid_grad_clip)
                        )
                        projection_optimizer.step()
                        projection_value = float(projection_total.detach())

                stats = {
                    **_mean_accumulated_stats(aggregate_stats, aggregate_counts),
                    **assignment_stats,
                    "loss/assignment_regularizers": float(assignment_loss.detach()),
                    "loss/curl_temporal": float(curl_temporal.detach()),
                    "loss/eulerian_strain_projection": float(projection_value),
                    "motion/gradient_norm": float(motion_grad),
                    "assignment/gradient_norm": float(assignment_grad),
                    "schedule/stage": stage_name,
                    "schedule/global_pass": int(global_pass_index + 1),
                    "schedule/window": int(window_index + 1),
                    "schedule/window_first_target": int(target_indices[0]),
                    "schedule/window_last_target": int(target_indices[-1]),
                    "schedule/local_iteration": int(local_index + 1),
                }
                if log_fn is not None:
                    log_fn("free", update_index, stats)
                if update_index % 5 == 0 or local_index == iterations_per_window - 1:
                    progress.set_postfix(
                        {
                            "stage": stage_name[:4],
                            "Data": f"{stats.get('loss/data_camera_mean', 0.0):.4f}",
                            "Pdyn": f"{stats['assignment/dynamic_probability_mean']:.3f}",
                            "Vmed": f"{stats.get('motion/speed_median', 0.0):.3f}",
                            "D": f"{stats.get('physics/strain_rate', 0.0):.2e}",
                        }
                    )
                progress.update(1)

                if local_index == iterations_per_window - 1:
                    with torch.no_grad():
                        soft = assignment.soft_probabilities(temperature)
                        hard = assignment.hard_labels()
                        audit_history.append(
                            {
                                "stage": stage_name,
                                "global_pass": int(global_pass_index + 1),
                                "window": int(window_index + 1),
                                "target_indices": list(map(int, target_indices)),
                                "update_index": int(update_index),
                                "temperature": float(temperature),
                                "same_part_floor": float(same_floor),
                                "dynamic_probability_mean": float(
                                    (1.0 - soft[:, 0]).mean()
                                ),
                                "hard_dynamic_count": int((hard > 0).sum()),
                                "initial_static_to_dynamic": int(
                                    (
                                        (~assignment.initial_dynamic_mask) & (hard > 0)
                                    ).sum()
                                ),
                                "initial_dynamic_to_static": int(
                                    (
                                        assignment.initial_dynamic_mask & (hard == 0)
                                    ).sum()
                                ),
                            }
                        )

    progress.close()
    gaussians.clear_time()

    # Final deterministic replay: rotations come from the derived local omega of v_c.
    with torch.no_grad():
        final_soft = assignment.soft_probabilities(
            float(args.assignment_temperature_end)
        ).detach()
        hard_labels = assignment.hard_labels().detach()
        final_dynamic = hard_labels > 0
        positions = [canonical_xyz.cpu()]
        rotations = [canonical_rotation.cpu()]
        velocities = [torch.zeros_like(canonical_xyz).cpu()]
        angular_velocities = [torch.zeros_like(canonical_xyz).cpu()]
        cell_ids_history = [torch.full((canonical_xyz.shape[0],), -1, dtype=torch.long)]
        gradient_audit = []
        x = canonical_xyz.clone()
        q = canonical_rotation.clone()
        for target_index in range(1, len(frame_groups)):
            source_time = float(frame_times[target_index - 1])
            target_time = float(frame_times[target_index])
            dt = target_time - source_time
            midpoint = 0.5 * (source_time + target_time)
            support = curl.material_support(x)
            ids = support["cell_ids"]
            occupied = support["occupied"]
            max_radius = (
                float(args.curl_material_neighbor_radius_cells)
                * float(curl.cell_size.max().cpu())
                if float(args.curl_material_neighbor_radius_cells) > 0
                else 0.0
            )
            graph = build_occupied_cell_knn_pair_graph(
                cell_centers,
                occupied,
                neighbors_per_cell=int(args.curl_material_neighbors),
                max_radius=max_radius,
                max_pairs=int(args.phase1_pair_max_pairs),
            )
            point_probs_final = final_soft.to(x.device)
            cell_prob, _ = aggregate_point_probabilities_to_cells(
                point_probs_final, ids, curl.num_cells
            )
            cell_dino_final, _ = (
                aggregate_point_features_to_cells(
                    gaussian_dino_features, ids, curl.num_cells, normalize=True
                )
                if gaussian_dino_features is not None
                else (None, None)
            )
            static_gate_final, static_hard_final = _straight_through_static_cell_gate(
                cell_prob,
                occupied,
                args,
                point_probabilities=point_probs_final,
                point_cell_ids=ids,
                num_cells=curl.num_cells,
            )
            static_gate_final = static_gate_final.detach()
            static_hard_final = static_hard_final.detach()
            v_cells_raw = curl.cell_velocities(midpoint)
            v_cells = (1.0 - static_gate_final[:, None]) * v_cells_raw
            gs = eulerian_velocity_gradient_rigidity_loss(
                cell_centers,
                v_cells,
                occupied,
                cell_prob,
                graph,
                same_part_floor=float(args.assignment_same_part_floor_end),
                velocity_scale=float(args.eulerian_velocity_scale),
                min_neighbors=int(args.eulerian_velocity_gradient_min_neighbors),
                relative_ridge=float(args.eulerian_velocity_gradient_ridge),
                static_speed_weight=1.0,
                cell_dino_features=cell_dino_final,
                dino_boundary_strength=float(args.dino_boundary_rigidity_strength),
                velocity_spatial_robust_delta=float(
                    args.velocity_spatial_coherence_robust_delta
                ),
            )
            v = curl.gather_cell_velocity(ids, cell_velocity_table=v_cells)
            omega = curl.gather_cell_velocity(ids, cell_velocity_table=gs["omega"])
            q = integrate_world_angular_velocity_quaternion(q, omega, dt)
            x = x + dt * v
            positions.append(x.cpu())
            rotations.append(q.cpu())
            velocities.append(v.cpu())
            angular_velocities.append(omega.cpu())
            cell_ids_history.append(curl.cell_ids(x).cpu())

            occ_ids = torch.nonzero(occupied, as_tuple=False).reshape(-1)
            Dnorm = torch.linalg.norm(gs["deformation"], dim=(-2, -1))
            traceG = torch.diagonal(gs["gradient"], dim1=-2, dim2=-1).sum(dim=-1)
            gradient_audit.append(
                {
                    "target_frame_index": int(target_index),
                    "target_time": target_time,
                    "occupied_cell_ids": occ_ids.cpu(),
                    "cell_probabilities": cell_prob[occ_ids].cpu(),
                    "cell_velocity": v_cells[occ_ids].cpu(),
                    "raw_curl_cell_velocity": v_cells_raw[occ_ids].cpu(),
                    "hard_static_gate": static_hard_final[occ_ids].cpu(),
                    "velocity_gradient": gs["gradient"][occ_ids].cpu(),
                    "deformation_tensor": gs["deformation"][occ_ids].cpu(),
                    "rotation_tensor": gs["rotation"][occ_ids].cpu(),
                    "derived_omega": gs["omega"][occ_ids].cpu(),
                    "curl_vector": (2.0 * gs["omega"][occ_ids]).cpu(),
                    "gradient_trace": traceG[occ_ids].cpu(),
                    "deformation_frobenius": Dnorm[occ_ids].cpu(),
                    "gradient_valid": gs["valid"][occ_ids].cpu(),
                    "cell_pairs": graph.pairs.cpu(),
                    "same_part_weight": gs["same_part_weight"].cpu(),
                    "strain_loss": float(gs["strain"].cpu()),
                    "spatial_gradient_coherence_loss": float(
                        gs["spatial_gradient_coherence"].cpu()
                    ),
                    "local_affine_coherence_loss": float(
                        gs["local_affine_coherence"].cpu()
                    ),
                    "velocity_spatial_coherence_loss": float(
                        gs["velocity_spatial_coherence"].cpu()
                    ),
                    "speed_spatial_coherence_loss": float(
                        gs["speed_spatial_coherence"].cpu()
                    ),
                    "omega_coherence_diag": float(gs["omega_coherence"].cpu()),
                    "local_rigid_residual_diag": float(
                        gs["local_rigid_residual"].cpu()
                    ),
                    "static_speed_loss": float(gs["static_speed"].cpu()),
                    "continuity_abs_max": float(
                        curl.continuity_residual(midpoint).abs().max().cpu()
                    ),
                    "outside_points": int(support["outside_points"]),
                }
            )

    gradient_summary = []
    for frame in gradient_audit:
        valid = frame["gradient_valid"].bool()
        dnorm = frame["deformation_frobenius"]
        onorm = torch.linalg.norm(frame["derived_omega"], dim=-1)
        tr = frame["gradient_trace"].abs()
        gradient_summary.append(
            {
                "target_frame_index": int(frame["target_frame_index"]),
                "target_time": float(frame["target_time"]),
                "valid_gradient_cells": int(valid.sum()),
                "deformation_frobenius_median": float(dnorm[valid].median())
                if valid.any()
                else 0.0,
                "deformation_frobenius_p90": float(torch.quantile(dnorm[valid], 0.90))
                if valid.any()
                else 0.0,
                "derived_omega_median": float(onorm[valid].median())
                if valid.any()
                else 0.0,
                "gradient_trace_abs_median": float(tr[valid].median())
                if valid.any()
                else 0.0,
                "strain_loss": float(frame["strain_loss"]),
                "velocity_spatial_coherence_loss": float(
                    frame["velocity_spatial_coherence_loss"]
                ),
                "speed_spatial_coherence_loss": float(
                    frame["speed_spatial_coherence_loss"]
                ),
                "omega_coherence_loss": float(frame["omega_coherence_diag"]),
                "local_rigid_residual_diag": float(frame["local_rigid_residual_diag"]),
                "continuity_abs_max": float(frame["continuity_abs_max"]),
                "outside_points": int(frame["outside_points"]),
            }
        )
    with (output_dir / "phase1_eulerian_velocity_gradient_summary.json").open(
        "w", encoding="utf-8"
    ) as stream:
        json.dump(
            _strict_json({"schema_version": 1, "frames": gradient_summary}),
            stream,
            indent=2,
            allow_nan=False,
        )

    final_tracks = {
        "schema_version": 14,
        "dynamic_phase": "phase1",
        "motion_representation": representation,
        "times": torch.as_tensor(frame_times, dtype=torch.float32),
        "positions": torch.stack(positions),
        "rotations": torch.stack(rotations),
        "velocities": torch.stack(velocities),
        "angular_velocities": torch.stack(angular_velocities),
        "cell_ids": torch.stack(cell_ids_history),
        "canonical_xyz": canonical_xyz.cpu(),
        "foreground_source_indices": gaussians._foreground_source_indices.detach().cpu(),
        "semantic_probability": None
        if gaussians.get_semantic_probability is None
        else gaussians.get_semantic_probability.detach().cpu(),
        "motion_evidence": None
        if gaussians.get_motion_evidence.numel() == 0
        else gaussians.get_motion_evidence.detach().cpu(),
        "initial_dynamic_mask": gaussians.get_initial_dynamic_mask.detach().cpu(),
        "initial_dynamic_probability": initial_dynamic.cpu(),
        "motion_membership": (1.0 - final_soft[:, 0]).cpu(),
        "part_probabilities": final_soft.cpu(),
        "part_labels": hard_labels.cpu(),
        "kinematic_report": None,
        "fixed_cameras": _fixed_camera_metadata(cameras),
        "rigidity_definition": "D=sym(grad(v))=0; omega=0.5*curl(v) derived only",
    }
    torch.save(final_tracks, output_dir / "phase1_gaussian_tracks_final.pt")
    torch.save(
        {
            "schema_version": 4,
            "initial_dynamic_mask": assignment.initial_dynamic_mask.cpu(),
            "initial_dynamic_score": assignment.initial_dynamic_score.cpu(),
            "initial_static_score": assignment.initial_static_score.cpu(),
            "initial_unknown_score": assignment.initial_unknown_score.cpu(),
            "initial_evidence_confidence": assignment.initial_evidence_confidence.cpu(),
            "initial_probabilities": assignment.initial_probabilities.cpu(),
            "initial_evidence_threshold": assignment.initial_evidence_threshold.cpu(),
            "final_probabilities": final_soft.cpu(),
            "final_labels": hard_labels.cpu(),
            "initial_static_to_dynamic": int(
                ((~assignment.initial_dynamic_mask) & final_dynamic).sum()
            ),
            "initial_dynamic_to_static": int(
                (assignment.initial_dynamic_mask & (~final_dynamic)).sum()
            ),
            "history": audit_history,
            "material_graph_pairs": material_graph.pairs.cpu(),
        },
        output_dir / "phase1_material_assignment_final.pt",
    )
    torch.save(curl.capture_state(), output_dir / "phase1_eulerian_curl_grid_final.pt")
    torch.save(
        {
            "schema_version": 3,
            "cell_centers": cell_centers.cpu(),
            "frames": gradient_audit,
        },
        output_dir / "phase1_eulerian_velocity_gradient_audit.pt",
    )

    write_transported_gaussians_html(
        final_tracks,
        output_dir / f"phase1_{representation}_transported.html",
        title="Phase 1 Eulerian - observed cell velocity + velocity-gradient rigidity",
        semantic_threshold=float(args.semantic_dynamic_threshold),
        max_points=int(args.transported_html_max_points),
    )
    return final_tracks, frame_times


def initialize_gradient_motion_labels(
    scene,
    gaussians,
    pipe,
    background,
    dataset,
    args,
    opt,
):
    """Measure full-scene motion evidence before foreground compaction."""

    cameras = _ordered_dynamic_cameras(scene.getTrainCameras())
    # Before dynamic initialization the foreground mask may not yet have been
    # materialized.  Build it from the frozen semantic field without removing
    # any Gaussian.
    try:
        foreground = gaussians.get_dynamic_foreground_mask
        if foreground.numel() != gaussians.get_canonical_xyz.shape[0]:
            raise RuntimeError
    except Exception:
        foreground = gaussians.build_dynamic_foreground_mask(args)

    def attribution_loss(camera, stage):
        gaussians.clear_time()
        return _attribution_visual_loss(
            camera, gaussians, pipe, background, dataset, args, opt, stage
        )

    state = compute_gradient_motion_evidence(
        cameras=cameras,
        gaussians=gaussians,
        visual_loss_fn=attribution_loss,
        foreground_mask=foreground,
        max_future_frames=int(args.motion_evidence_future_frames),
        max_cameras=int(args.motion_evidence_max_cameras),
        min_dynamic_points=max(
            int(args.motion_evidence_min_dynamic_points),
            8 * int(args.num_moving_parts),
        ),
        static_refine_steps=int(args.static_evidence_refine_steps),
        static_refine_lr=float(args.static_evidence_refine_lr),
        static_refine_prior_weight=float(args.static_evidence_refine_prior_weight),
    )

    # v8: no pre-training STATIC/DYNAMIC/UNKNOWN classifier.  The historical
    # gradient attribution is only a weak initialization prior.  Static/dynamic
    # labels are optimized contextually together with the Eulerian velocity field.
    state["schema_version"] = max(int(state.get("schema_version", 26)), 28)

    output_dir = Path(scene.model_path) / "dynamic_debug"
    output_dir.mkdir(parents=True, exist_ok=True)
    torch.save(state, output_dir / "phase1_gradient_motion_evidence.pt")
    write_gradient_motion_evidence_html(
        state,
        output_dir / "static_dynamic_gradient_attribution.html",
        max_points=int(args.motion_evidence_html_max_points),
    )

    return state


@torch.no_grad()
def initialize_neutral_motion_labels(gaussians):
    """Return an exact P(dynamic)=0.5 prior without inferring material labels."""
    canonical_xyz = gaussians.get_canonical_xyz.detach().cpu()
    neutral = torch.full((canonical_xyz.shape[0],), 0.5, dtype=torch.float32)
    return {
        "schema_version": 1,
        "canonical_xyz": canonical_xyz,
        "motion_evidence": neutral,
        "motion_evidence_normalized": neutral.clone(),
        "static_consistency_score": neutral.clone(),
        "initial_dynamic_mask": torch.zeros_like(neutral, dtype=torch.bool),
        "initialization": "neutral_foreground_no_material_labels",
    }


def _rigid_replay_to_frame(
    canonical_xyz,
    canonical_rotation,
    mode,
    interval_speeds,
    frame_times,
    frame_index,
    effective_dynamic_mask,
):
    canonical_rotation = torch.nn.functional.normalize(
        canonical_rotation, dim=-1
    )
    if mode is None or frame_index <= 0:
        velocity = torch.zeros_like(canonical_xyz)
        return canonical_xyz.clone(), canonical_rotation.clone(), velocity
    time_values = torch.as_tensor(frame_times, dtype=torch.float64)
    speeds = torch.as_tensor(interval_speeds, dtype=torch.float64)
    count = min(int(frame_index), int(speeds.numel()))
    delta_time = time_values[1 : count + 1] - time_values[:count]
    total_increment = float((speeds[:count] * delta_time).sum())
    mode_device = mode.to(device=canonical_xyz.device, dtype=canonical_xyz.dtype)
    result = integrate_rigid_mode_exact(
        canonical_xyz,
        mode_device,
        total_increment,
        orientations=canonical_rotation,
        effective_dynamic_mask=effective_dynamic_mask,
    )
    speed_index = min(max(int(frame_index), 0), int(speeds.numel()) - 1)
    current_speed = float(speeds[speed_index]) if speeds.numel() else 0.0
    velocity = predict_rigid_cell_velocity(
        mode_device, result["positions"], current_speed
    )
    velocity = velocity * effective_dynamic_mask[:, None].to(velocity)
    return result["positions"], result["orientations"], velocity


def _smooth_rigid_cell_probability(probability, pair_graph, improvement, margin):
    if pair_graph.count == 0:
        return probability, improvement > float(margin)
    pairs = pair_graph.pairs.to(probability.device)
    total = torch.zeros_like(probability)
    count = torch.zeros_like(probability)
    total.index_add_(0, pairs[:, 0], probability[pairs[:, 1]])
    total.index_add_(0, pairs[:, 1], probability[pairs[:, 0]])
    count.index_add_(0, pairs[:, 0], torch.ones_like(probability[pairs[:, 0]]))
    count.index_add_(0, pairs[:, 1], torch.ones_like(probability[pairs[:, 1]]))
    neighbor_mean = torch.where(count > 0, total / count.clamp_min(1.0), probability)
    smoothed = 0.8 * probability + 0.2 * neighbor_mean
    return smoothed, improvement > float(margin)


def _temporal_speed_regularization(speeds, interval_weights, strength):
    value = torch.as_tensor(speeds).detach().cpu().double()
    if value.numel() < 2 or float(strength) <= 0.0:
        return value
    weight = torch.as_tensor(interval_weights).detach().cpu().double().clamp_min(1.0e-8)
    difference = torch.zeros(
        (value.numel() - 1, value.numel()), dtype=torch.float64
    )
    row = torch.arange(value.numel() - 1)
    difference[row, row] = -1.0
    difference[row, row + 1] = 1.0
    normal = torch.diag(weight) + float(strength) * (difference.T @ difference)
    return torch.linalg.solve(normal, weight * value)


def _rigid_pairwise_distance_error(position_history, effective_dynamic_mask, limit):
    indices = torch.nonzero(effective_dynamic_mask, as_tuple=False).flatten()
    if indices.numel() < 2:
        return 0.0
    if indices.numel() > int(limit):
        sample = torch.linspace(0, indices.numel() - 1, int(limit)).long()
        indices = indices.index_select(0, sample)
    reference = torch.pdist(position_history[0].index_select(0, indices).double())
    maximum = 0.0
    for positions in position_history[1:]:
        current = torch.pdist(positions.index_select(0, indices).double())
        maximum = max(maximum, float((current - reference).abs().max()))
    return maximum


def run_rigid_killing_phase1(
    scene,
    gaussians,
    pipe,
    background,
    dataset,
    args,
    opt,
    *,
    log_fn=None,
):
    """Deterministic EM solve for one rigid spatial mode and scalar speeds."""
    del background, opt
    representation = str(gaussians.motion_representation)
    rigid_representations = {
        "rigid_killing_1d",
        "rigid_killing_eulerian",
        "rigid_killing_lagrangian",
    }
    if representation not in rigid_representations:
        raise RuntimeError("run_rigid_killing_phase1 requires a rigid Killing representation")
    lagrangian = representation == "rigid_killing_lagrangian"
    freeze_depth_cores = bool(
        getattr(args, "phase1_depth_sequence_pretrain", False)
        and getattr(args, "phase1_freeze_depth_cores", False)
    )
    neutral_discovery = bool(
        getattr(args, "phase1_neutral_rigid_discovery", False)
    )
    responsibility_training = bool(
        getattr(args, "rigid_responsibility_training", False)
    )
    dynamic_prior_supplied = bool(str(getattr(args, "dynamic_prior_path", "") or "").strip())
    freeze_prior_labels = bool(getattr(args, "freeze_dynamic_prior_labels", False))
    if freeze_prior_labels and not dynamic_prior_supplied:
        raise RuntimeError("--freeze_dynamic_prior_labels requires --dynamic_prior_path")
    support_type_name = (
        "gaussian"
        if lagrangian
        else (
            "contrastive_eulerian_support"
            if responsibility_training
            else "occupied_cell"
        )
    )
    if gaussians.curl_grid is not None or gaussians.motion_optimizer is not None:
        raise RuntimeError("The rigid branch must not own a curl field or motion optimizer")
    if gaussians.material_assignment is None:
        raise RuntimeError("Rigid Phase 1 requires initialized material assignment")

    cameras = _ordered_dynamic_cameras(scene.getTrainCameras())
    frame_groups = _complete_frame_groups(cameras, args)
    if len(frame_groups) < 2:
        raise RuntimeError("Rigid Phase 1 requires at least two complete frames")
    frame_times = [float(group[1]) for group in frame_groups]
    frame_group_maps = [_camera_map(group[2]) for group in frame_groups]
    canonical_xyz = gaussians.get_canonical_xyz.detach()
    canonical_rotation = gaussians.get_rotation.detach().clone()

    def phase_clock():
        if canonical_xyz.is_cuda:
            torch.cuda.synchronize(canonical_xyz.device)
        return time.perf_counter()

    phase_started = phase_clock()
    runtime = {
        "flow_stat_accumulation_seconds": 0.0,
        "rigid_solver_seconds": 0.0,
        "assignment_seconds": 0.0,
        "final_replay_seconds": 0.0,
    }
    observation_grid = None
    point_group_ids = None
    geometric_cell_ids = None
    cell_graph = None
    eulerian_support_state = None
    gaussian_dino_graph = None
    support_state_history = []
    gaussian_dino_features = gaussians.get_dino_feature_field
    if responsibility_training:
        if (
            not torch.is_tensor(gaussian_dino_features)
            or gaussian_dino_features.ndim != 2
            or gaussian_dino_features.shape[0] != canonical_xyz.shape[0]
            or gaussian_dino_features.shape[1] == 0
        ):
            raise RuntimeError(
                "Soft rigid responsibility training requires detached per-Gaussian DINO features"
            )
        gaussian_dino_features = gaussian_dino_features.detach()
        gaussian_dino_graph = build_gaussian_dino_graph(
            canonical_xyz,
            gaussian_dino_features,
            neighbors=int(args.rigid_responsibility_dino_knn),
            sigma=float(args.rigid_responsibility_dino_sigma),
            max_pairs=int(args.phase1_pair_max_pairs),
            query_chunk_size=int(args.lagrangian_knn_chunk_size),
        )
    if lagrangian:
        num_supports = int(canonical_xyz.shape[0])
        canonical_support_positions = canonical_xyz.detach().cpu().double()
    else:
        observation_grid = assign_gaussians_to_cells(
            canonical_xyz,
            int(args.curl_grid_target_cells),
            padding_fraction=float(args.curl_grid_padding),
            min_cells_per_axis=int(args.curl_grid_min_cells_per_axis),
        )
        geometric_cell_ids = observation_grid.point_group_ids.to(
            device=canonical_xyz.device, dtype=torch.long
        )
        if responsibility_training:
            eulerian_support_state = split_cells_by_dino_components(
                canonical_xyz,
                geometric_cell_ids,
                gaussian_dino_features,
                num_cells=observation_grid.num_cells,
                enabled=bool(args.eulerian_contrastive_dino_split),
                cosine_threshold=float(
                    args.eulerian_contrastive_dino_cosine_threshold
                ),
                dispersion_threshold=float(
                    args.eulerian_contrastive_dino_dispersion_threshold
                ),
                min_gaussians=int(args.eulerian_contrastive_min_gaussians),
                local_neighbors=int(args.rigid_responsibility_dino_knn),
            )
            point_group_ids = build_contrastive_group_ids(
                eulerian_support_state, device=canonical_xyz.device
            )
            num_supports = int(eulerian_support_state["num_supports"])
            canonical_support_positions = aggregate_points_to_rigid_cells(
                canonical_xyz, point_group_ids, num_supports
            ).detach().cpu().double()
        else:
            point_group_ids = geometric_cell_ids
            num_supports = observation_grid.num_cells
            canonical_support_positions = observation_grid.cell_centers.double()
        cell_graph = build_occupied_cell_knn_pair_graph(
            canonical_support_positions.to(
                device=canonical_xyz.device, dtype=canonical_xyz.dtype
            ),
            torch.ones(num_supports, device=canonical_xyz.device, dtype=torch.bool),
            neighbors_per_cell=6,
            max_pairs=int(args.phase1_pair_max_pairs),
        )
    correspondence_cache = MotionCorrespondenceCache(dataset, args)
    if not correspondence_cache.enabled:
        raise RuntimeError("Rigid Phase 1 requires rasterized flow observations")

    initial_point_probability = (
        1.0 - gaussians.material_assignment.soft_probabilities()[:, 0]
    ).detach()
    if (neutral_discovery or responsibility_training) and not torch.allclose(
        initial_point_probability,
        torch.full_like(initial_point_probability, 0.5),
        atol=1.0e-6,
        rtol=0.0,
    ):
        raise RuntimeError("Rigid responsibility discovery requires exact P(rigid)=0.5")
    current_point_probability = initial_point_probability.clone()
    current_effective_mask = gaussians.get_initial_dynamic_mask.detach().clone()
    sequence_enabled = bool(
        getattr(args, "sequence_pretrain_assignment", False)
        and gaussians.sequence_assignment_state is not None
    )
    if freeze_prior_labels and (
        sequence_enabled or neutral_discovery or responsibility_training or freeze_depth_cores
    ):
        raise RuntimeError(
            "--freeze_dynamic_prior_labels is only valid for plain prior-guided rigid Killing"
        )
    if (
        lagrangian
        and not sequence_enabled
        and not neutral_discovery
        and not responsibility_training
    ):
        raise RuntimeError(
            "The Lagrangian rigid branch requires whole-sequence Gaussian assignment"
        )
    sequence_state = None
    fit_core_mask = current_effective_mask.clone()
    consecutive_support_cycles = torch.zeros_like(
        current_effective_mask, dtype=torch.long
    )
    sequence_calibrations = {}
    sequence_identity_masks = {}
    frozen_sequence_state = None
    frozen_probability = None
    frozen_static_control_diagnostics = None
    static_control_group_ids = None
    if neutral_discovery:
        fit_core_mask = torch.ones_like(current_effective_mask)
        current_effective_mask = fit_core_mask.clone()
    if responsibility_training:
        fit_core_mask = torch.ones_like(current_effective_mask)
        current_effective_mask = torch.zeros_like(current_effective_mask)
    if sequence_enabled:
        sequence_state = gaussians.sequence_assignment_state["state"].to(
            device=canonical_xyz.device, dtype=torch.uint8
        )
        if freeze_depth_cores:
            current_point_probability = gaussians.sequence_assignment_state[
                "dynamic_probability"
            ].to(device=canonical_xyz.device, dtype=canonical_xyz.dtype)
            initial_point_probability = current_point_probability.detach().clone()
            frozen_sequence_state = sequence_state.detach().clone()
            frozen_probability = current_point_probability.detach().clone()
            if not lagrangian:
                source_ids = torch.arange(
                    canonical_xyz.shape[0],
                    device=canonical_xyz.device,
                    dtype=torch.long,
                )
                static_control_group_ids = torch.where(
                    sequence_state == STATIC_CORE,
                    source_ids,
                    torch.full_like(source_ids, -1),
                )
        fit_core_mask = sequence_state == DYNAMIC_CORE
        current_effective_mask = fit_core_mask.clone()
        if not bool(fit_core_mask.any()):
            raise RuntimeError("Rigid solve aborted: whole-sequence dynamic core is empty")
    current_mode = None
    current_speeds = torch.zeros(len(frame_groups) - 1, dtype=torch.float64)
    fixed_reference_point = None
    cycle_history = []
    assignment_state_history = []
    final_observations = None
    final_positions = None
    final_rigid_rms_px = None
    final_assignment = None
    final_gaussian_speed_observations = None
    final_gaussian_query_positions = None
    final_gaussian_query_rotations = None
    eulerian_gaussian_reference = None
    static_control_observation_states = []
    static_control_positions = []
    stable_cycles = 0
    previous_sequence_energy = None
    sequence_energy_worsenings = 0

    output_dir = Path(scene.model_path) / "dynamic_debug"
    output_dir.mkdir(parents=True, exist_ok=True)
    calibration_path = output_dir / "phase1_static_flow_calibration.pt"
    if sequence_enabled and not freeze_depth_cores:
        if not calibration_path.exists():
            raise RuntimeError("sequence pretraining calibration artifact is missing")
        sequence_calibrations = torch.load(calibration_path, map_location="cpu")[
            "calibration"
        ]
        visibility_path = output_dir / "phase1_visibility_sequence_pixel.pt"
        if not visibility_path.exists():
            raise RuntimeError("sequence pretraining visibility artifact is missing")
        sequence_identity_masks = torch.load(
            visibility_path, map_location="cpu"
        )["identity_masks"]
        correspondence_cache.measurement_source_id = (
            "pair_observation_occlusion_calibrated_flow_v2"
        )

    def sequence_observation_transform(interval_index):
        if not sequence_enabled or freeze_depth_cores:
            return None

        def transform(source_camera, _target_camera, observation):
            key = str(getattr(source_camera, "dynamic_camera_id", ""))
            calibration = sequence_calibrations[(key, int(interval_index))]
            median = torch.as_tensor(
                calibration["median"],
                device=observation.flow.device,
                dtype=observation.flow.dtype,
            )
            identity = sequence_identity_masks[key][int(interval_index)].to(
                device=observation.flow.device, dtype=observation.flow.dtype
            )
            if tuple(identity.shape) != tuple(observation.flow.shape[-2:]):
                identity = torch.nn.functional.interpolate(
                    identity[None, None],
                    size=observation.flow.shape[-2:],
                    mode="bilinear",
                    align_corners=False,
                )[0, 0]
            corrected = observation.flow - median[:, None, None]
            confidence = observation.confidence * identity[None].clamp(0, 1)
            confidence = confidence * float(calibration["camera_reliability"])
            return PairObservation(
                observation.source_height,
                observation.source_width,
                corrected,
                confidence,
                observation.calibration_confidence,
            )

        return transform

    def save_responsibility_artifacts():
        responsibility_payload = {
            "schema_version": 1,
            "representation": representation,
            "support_type": support_type_name,
            "initial_probability": initial_point_probability.cpu(),
            "gaussian_responsibility": current_point_probability.cpu(),
            "raw_gaussian_responsibility": final_assignment["raw_probability"],
            "static_energy": final_assignment["static_energy"].detach().cpu(),
            "rigid_energy": final_assignment["rigid_energy"].detach().cpu(),
            "normalized_energy_margin": final_assignment[
                "normalized_energy_margin"
            ],
            "state": sequence_state.detach().cpu(),
            "static_mask": classification["static_mask"].detach().cpu(),
            "rigid_mask": classification["rigid_mask"].detach().cpu(),
            "ambiguous_mask": classification["ambiguous_mask"].detach().cpu(),
            "entropy": classification["entropy"].detach().cpu(),
            "temperature_schedule": [
                float(args.rigid_responsibility_temperature_start),
                float(args.rigid_responsibility_temperature_end),
            ],
            "margin": float(args.rigid_responsibility_margin),
            "static_threshold": float(args.rigid_responsibility_static_threshold),
            "rigid_threshold": float(args.rigid_responsibility_rigid_threshold),
            "dino_graph": {
                key: value.detach().cpu() if torch.is_tensor(value) else value
                for key, value in gaussian_dino_graph.items()
            },
            "cycle_history": assignment_state_history,
            "ground_truth_used_as_algorithm_input": False,
        }
        _atomic_torch_save(
            responsibility_payload,
            output_dir / "phase1_rigid_responsibility_final.pt",
        )
        responsibility_summary = {
            "schema_version": 1,
            "representation": representation,
            "support_type": support_type_name,
            "gaussian_count": int(canonical_xyz.shape[0]),
            "static_count": int(classification["static_mask"].sum()),
            "rigid_count": int(classification["rigid_mask"].sum()),
            "ambiguous_count": int(classification["ambiguous_mask"].sum()),
            "ambiguous_fraction": classification["ambiguous_fraction"],
            "responsibility_mean": float(current_point_probability.mean()),
            "responsibility_median": float(current_point_probability.median()),
            "responsibility_entropy": classification["mean_entropy"],
            "cycles": len(cycle_history),
            "dino_role": "boundary_and_material_affinity_only",
            "motion_role": "static_vs_rigid_side_identity",
            "soft_geometry_blending": False,
            "ground_truth_used_as_algorithm_input": False,
        }
        if not lagrangian:
            final_support_stats = compute_support_responsibility_stats(
                eulerian_support_state,
                current_point_probability,
                static_threshold=float(args.eulerian_contrastive_static_threshold),
                rigid_threshold=float(args.eulerian_contrastive_rigid_threshold),
            )
            final_support_observation = support_observation_diagnostics(
                eulerian_support_state, final_observations
            )
            purity = summarize_support_purity(
                eulerian_support_state,
                classification["static_mask"],
                classification["rigid_mask"],
            )
            mixed = mixed_cell_diagnostics(
                geometric_cell_ids,
                eulerian_support_state,
                classification["static_mask"],
                classification["rigid_mask"],
            )
            transitions = summarize_geometric_cell_transitions(
                discovery_positions_history, observation_grid
            )
            subgroup_count = torch.bincount(
                eulerian_support_state["parent_cell_id"],
                minlength=observation_grid.num_cells,
            )
            support_payload = {
                "schema_version": 1,
                "geometric_cell_ids": geometric_cell_ids.detach().cpu(),
                "contrastive_support_ids": point_group_ids.detach().cpu(),
                "parent_cell_id": eulerian_support_state["parent_cell_id"],
                "parent_support_id": eulerian_support_state["parent_support_id"],
                "split_reason": eulerian_support_state["split_reason"],
                "split_cycle": eulerian_support_state["split_cycle"],
                "weak_support": eulerian_support_state["weak_support"],
                "number_of_dino_subgroups": subgroup_count,
                "cell_dino_dispersion": eulerian_support_state[
                    "cell_dino_dispersion"
                ],
                **final_support_stats,
                **final_support_observation,
                **purity,
                "mixed_cell_diagnostics": mixed,
                "geometric_cell_transitions": transitions,
                "pooling_benefit": eulerian_gaussian_reference["comparison"],
                "support_history": support_state_history,
            }
            _atomic_torch_save(
                support_payload,
                output_dir / "phase1_eulerian_contrastive_support_final.pt",
            )
            _atomic_torch_save(
                eulerian_gaussian_reference,
                output_dir / "phase1_eulerian_per_gaussian_pooling_reference.pt",
            )
            responsibility_summary.update({
                "geometric_cell_count": observation_grid.num_cells,
                "contrastive_support_count": num_supports,
                "dino_split_support_count": sum(
                    reason == "dino"
                    for reason in eulerian_support_state["split_reason"]
                ),
                "responsibility_split_support_count": sum(
                    reason == "responsibility"
                    for reason in eulerian_support_state["split_reason"]
                ),
                "mean_subgroups_per_geometric_cell": float(
                    subgroup_count.float().mean()
                ),
                **{
                    key: value
                    for key, value in mixed.items()
                    if not torch.is_tensor(value)
                },
                **{
                    key: value
                    for key, value in purity.items()
                    if key != "support_purity"
                },
                "valid_support_fraction": final_support_observation[
                    "valid_support_fraction"
                ],
                "weak_conditioning_fraction": final_support_observation[
                    "weak_conditioning_fraction"
                ],
                "pooling_observation_count_gain": eulerian_gaussian_reference[
                    "comparison"
                ]["median_observation_count_gain"],
                "pooling_camera_support_gain": eulerian_gaussian_reference[
                    "comparison"
                ]["median_camera_support_gain"],
                "pooling_condition_number_ratio": eulerian_gaussian_reference[
                    "comparison"
                ]["median_condition_number_ratio"],
                "gaussians_ever_changing_geometric_cell": int(
                    transitions["gaussians_ever_changing_cell"].sum()
                ),
            })
        with (output_dir / "phase1_rigid_responsibility_summary.json").open(
            "w", encoding="utf-8"
        ) as stream:
            json.dump(
                _strict_json(responsibility_summary),
                stream,
                indent=2,
                allow_nan=False,
            )
    if responsibility_training:
        progress_name = (
            "Phase 1 soft-responsibility Lagrangian rigid fit"
            if lagrangian
            else "Phase 1 soft-responsibility contrastive Eulerian rigid fit"
        )
    elif neutral_discovery:
        progress_name = (
            "Phase 1 neutral Lagrangian discovery + assignment"
            if lagrangian
            else "Phase 1 neutral Eulerian discovery + assignment"
        )
    elif freeze_depth_cores:
        progress_name = (
            "Phase 1B frozen-core Lagrangian rigid fit"
            if lagrangian
            else "Phase 1B frozen-core Eulerian rigid fit"
        )
    else:
        progress_name = (
            "Phase 1B-D Lagrangian rigid fit/assignment"
            if lagrangian
            else "Phase 1B-D Eulerian rigid fit/assignment"
        )
    rigid_progress = tqdm(
        range(
            int(args.rigid_responsibility_cycles)
            if responsibility_training
            else (
                1
                if freeze_depth_cores or neutral_discovery
                else int(args.free_field_iters)
            )
        ),
        desc=progress_name,
        unit="cycle",
    )
    for cycle in rigid_progress:
        if responsibility_training and not lagrangian:
            eulerian_support_state = split_supports_by_responsibility(
                eulerian_support_state,
                current_point_probability,
                cycle=cycle,
                enabled=bool(args.eulerian_contrastive_responsibility_split),
                static_threshold=float(args.eulerian_contrastive_static_threshold),
                rigid_threshold=float(args.eulerian_contrastive_rigid_threshold),
                min_bimodal_fraction=float(
                    args.eulerian_contrastive_min_bimodal_fraction
                ),
                min_gaussians=int(args.eulerian_contrastive_min_gaussians),
            )
            point_group_ids = build_contrastive_group_ids(
                eulerian_support_state, device=canonical_xyz.device
            )
            num_supports = int(eulerian_support_state["num_supports"])
            canonical_support_positions = aggregate_points_to_rigid_cells(
                canonical_xyz, point_group_ids, num_supports
            ).detach().cpu().double()
            support_state_history.append({
                "cycle": cycle,
                "group_ids": point_group_ids.detach().cpu(),
                "parent_cell_id": eulerian_support_state["parent_cell_id"].clone(),
                "parent_support_id": eulerian_support_state[
                    "parent_support_id"
                ].clone(),
                "split_reason": list(eulerian_support_state["split_reason"]),
                "split_cycle": eulerian_support_state["split_cycle"].clone(),
                "membership_changes": int(
                    eulerian_support_state["membership_changes"]
                ),
            })
        observation_started = phase_clock()
        observation_states = []
        support_positions = []
        gaussian_query_positions = []
        gaussian_query_rotations = []
        for target_index in range(1, len(frame_groups)):
            source_index = target_index - 1
            source_xyz, source_rotation, _ = _rigid_replay_to_frame(
                canonical_xyz,
                canonical_rotation,
                current_mode,
                current_speeds,
                frame_times,
                source_index,
                (
                    torch.ones_like(current_effective_mask)
                    if responsibility_training
                    else current_effective_mask
                ),
            )
            source_supports = (
                source_xyz
                if lagrangian
                else aggregate_points_to_rigid_cells(
                    source_xyz,
                    point_group_ids,
                    num_supports,
                    weights=(
                        fit_core_mask.to(source_xyz.dtype)
                        if sequence_enabled
                        else None
                    ),
                )
            )
            source_by_camera = frame_group_maps[source_index]
            target_by_camera = frame_group_maps[target_index]
            observation_args = (
                source_xyz,
                list(source_by_camera.values()),
                list(target_by_camera.values()),
                frame_times[source_index],
                frame_times[target_index],
            )
            if lagrangian and responsibility_training:
                state = build_gaussian_motion_observations(
                    correspondence_cache,
                    *observation_args,
                    gaussians=gaussians,
                    pipe=pipe,
                    query_rotations=source_rotation,
                    observation_transform=sequence_observation_transform(source_index),
                )
            else:
                state = correspondence_cache.observed_velocity_targets(
                    *observation_args,
                    group_ids=(
                        None
                        if lagrangian
                        else (
                            torch.where(
                                fit_core_mask,
                                point_group_ids,
                                torch.full_like(point_group_ids, -1),
                            )
                            if sequence_enabled
                            else point_group_ids
                        )
                    ),
                    num_groups=None if lagrangian else num_supports,
                    gaussians=gaussians,
                    pipe=pipe,
                    query_rotations=source_rotation,
                    observation_transform=sequence_observation_transform(source_index),
                )
            observation_states.append(state)
            support_positions.append(source_supports.detach().cpu().double())
            gaussian_query_positions.append(source_xyz.detach().cpu())
            gaussian_query_rotations.append(source_rotation.detach().cpu())
            if freeze_depth_cores and not lagrangian:
                static_state = correspondence_cache.observed_velocity_targets(
                    source_xyz,
                    list(source_by_camera.values()),
                    list(target_by_camera.values()),
                    frame_times[source_index],
                    frame_times[target_index],
                    group_ids=static_control_group_ids,
                    num_groups=int(canonical_xyz.shape[0]),
                    gaussians=gaussians,
                    pipe=pipe,
                    query_rotations=source_rotation,
                    observation_transform=sequence_observation_transform(
                        source_index
                    ),
                )
                static_control_observation_states.append(static_state)
                static_control_positions.append(
                    source_xyz.detach().cpu().double()
                )

        positions = torch.stack(support_positions)
        observed_velocity = torch.stack(
            [state["velocity"].detach().cpu().double() for state in observation_states]
        )
        confidence = torch.stack(
            [state["confidence"].detach().cpu().double() for state in observation_states]
        )
        observation_valid = torch.stack(
            [state["valid"].detach().cpu().bool() for state in observation_states]
        )
        camera_count = torch.stack(
            [
                state["distinct_camera_count"].detach().cpu().long()
                for state in observation_states
            ]
        )
        observation_valid &= camera_count >= int(args.rigid_min_distinct_cameras)
        solver_valid = observation_valid.clone()
        if lagrangian:
            if not responsibility_training:
                solver_valid &= fit_core_mask.detach().cpu()[None]
            support_prior = (
                current_effective_mask.detach().cpu().double()
                if freeze_prior_labels
                else current_point_probability.detach().cpu().double()
            )
        elif responsibility_training:
            support_prior = aggregate_gaussian_responsibility_to_support(
                eulerian_support_state,
                current_point_probability.detach().cpu(),
            ).double()
        elif freeze_prior_labels:
            support_prior = aggregate_points_to_rigid_cells(
                current_effective_mask.to(canonical_xyz.dtype),
                point_group_ids,
                num_supports,
            ).detach().cpu().double()
        else:
            support_prior = aggregate_points_to_rigid_cells(
                current_point_probability,
                point_group_ids,
                num_supports,
            ).detach().cpu().double()
        if freeze_prior_labels:
            fit_weight = confidence * support_prior.clamp(0.0, 1.0)[None]
        elif responsibility_training:
            fit_weight = confidence * support_prior.clamp(0.0, 1.0)[None]
        else:
            fit_weight = (
                confidence
                if sequence_enabled or neutral_discovery
                else confidence * support_prior.clamp_min(0.02)[None]
            )
        runtime["flow_stat_accumulation_seconds"] += (
            phase_clock() - observation_started
        )

        if fixed_reference_point is None:
            support = (fit_weight * solver_valid.to(torch.float64)).sum(dim=0)
            if not bool(support.gt(0).any()):
                raise RuntimeError("Rigid solve aborted: no accepted moving observations")
            reference_support = int(torch.argmax(support))
            fixed_reference_point = canonical_support_positions[
                reference_support
            ].clone()

        solver_started = phase_clock()
        mode, speeds, solver_diagnostics = solve_rigid_killing_mode_all_intervals(
            positions,
            observed_velocity,
            weights=fit_weight,
            valid_mask=solver_valid,
            reference_point=fixed_reference_point,
            max_iters=int(args.rigid_solver_max_iters),
            tolerance=float(args.rigid_solver_tol),
            irls_iters=int(args.rigid_solver_irls_iters),
            huber_delta=float(args.rigid_solver_huber_delta),
            min_valid_cells=int(args.rigid_min_valid_cells),
            angular_eps=float(args.rigid_angular_eps),
        )
        if not mode.valid:
            raise RuntimeError(
                "Rigid solve aborted: the shared Killing mode is invalid or does not beat static"
            )
        condition_number = float(mode.diagnostics.get("condition_number", float("inf")))
        if condition_number > float(args.rigid_max_condition_number):
            raise RuntimeError(
                f"Rigid solve aborted: condition number {condition_number:.3e} exceeds limit"
            )
        interval_weight = (confidence * solver_valid.to(torch.float64)).sum(dim=1)
        speeds = _temporal_speed_regularization(
            speeds, interval_weight, float(args.rigid_speed_temporal_weight)
        )
        gaussian_speed_observations = None
        if lagrangian:
            gaussian_speed_observations = (
                compute_gaussian_speed_diagnostics(
                    positions,
                    observed_velocity,
                    observation_states,
                    observation_valid,
                    mode,
                    speeds,
                )
                if responsibility_training
                else compute_gaussian_speed_observations(
                positions,
                observed_velocity,
                torch.stack(
                    [
                        state["normal_matrix"].detach().cpu().double()
                        for state in observation_states
                    ]
                ),
                observation_valid,
                mode,
                speeds,
                )
            )
        runtime["rigid_solver_seconds"] += phase_clock() - solver_started

        assignment_started = phase_clock()
        predicted_support_velocity = predict_rigid_cell_velocity(mode, positions, speeds)
        rigid_rms_px = torch.stack(
            [
                velocity_reprojection_rms(
                    state,
                    predicted_support_velocity[index].to(
                        device=canonical_xyz.device, dtype=canonical_xyz.dtype
                    ),
                    delta_time=frame_times[index + 1] - frame_times[index],
                )
                .detach()
                .cpu()
                .double()
                for index, state in enumerate(observation_states)
            ]
        )
        static_rms_px = torch.stack(
            [state["static_rms_px_mean"].detach().cpu().double() for state in observation_states]
        )
        assignment_valid = torch.stack(
            [state["base_valid"].detach().cpu().bool() for state in observation_states]
        ) & (camera_count >= int(args.rigid_min_distinct_cameras))
        if responsibility_training:
            energy = compute_gaussian_static_rigid_sequence_energies(
                canonical_xyz,
                mode,
                speeds,
                frame_times,
                frame_group_maps,
                correspondence_cache,
                {},
                args,
            )
            schedule_fraction = float(cycle) / float(
                max(int(args.rigid_responsibility_cycles) - 1, 1)
            )
            temperature = (
                float(args.rigid_responsibility_temperature_start)
                + schedule_fraction
                * (
                    float(args.rigid_responsibility_temperature_end)
                    - float(args.rigid_responsibility_temperature_start)
                )
            )
            update = update_gaussian_responsibility(
                energy["static_energy"],
                energy["rigid_energy"],
                margin=float(args.rigid_responsibility_margin),
                temperature=temperature,
                energy_weight=energy["energy_weight"],
            )
            raw_probability = update["raw_probability"].to(canonical_xyz)
            new_point_probability = (
                smooth_gaussian_responsibility_dino(
                    raw_probability,
                    gaussian_dino_graph,
                    strength=float(args.rigid_responsibility_dino_strength),
                )
                if args.rigid_responsibility_dino
                else raw_probability
            )
            classification = classify_final_gaussians(
                new_point_probability,
                static_threshold=float(args.rigid_responsibility_static_threshold),
                rigid_threshold=float(args.rigid_responsibility_rigid_threshold),
            )
            sequence_state = classification["state"].to(
                device=canonical_xyz.device, dtype=torch.uint8
            )
            new_effective_mask = classification["rigid_mask"].to(
                canonical_xyz.device
            )
            fit_core_mask = torch.ones_like(new_effective_mask)
            if lagrangian:
                support_probability = new_point_probability
                effective_support_mask = new_effective_mask
            else:
                support_probability = aggregate_gaussian_responsibility_to_support(
                    eulerian_support_state,
                    new_point_probability.detach().cpu(),
                ).to(canonical_xyz)
                effective_support_mask = (
                    aggregate_points_to_rigid_cells(
                        new_effective_mask.to(canonical_xyz.dtype),
                        point_group_ids,
                        num_supports,
                    )
                    > 0
                )
            assignment = {
                **energy,
                "state": sequence_state.detach().cpu(),
                "fit_core_mask": fit_core_mask.detach().cpu(),
                "effective_dynamic_mask": new_effective_mask.detach().cpu(),
                "dynamic_probability": new_point_probability.detach().cpu(),
                "raw_probability": raw_probability.detach().cpu(),
                "normalized_energy_margin": update[
                    "normalized_energy_margin"
                ].detach().cpu(),
                "temperature": temperature,
                "classification_entropy": classification["entropy"].detach().cpu(),
                "ambiguous_mask": classification["ambiguous_mask"].detach().cpu(),
                "label_changes": int(
                    (new_effective_mask != current_effective_mask).sum()
                ),
                "core_promotions": 0,
                "core_demotions": 0,
                "consecutive_support_cycles": consecutive_support_cycles.cpu(),
            }
            static_control_margin = 0.0
        elif freeze_prior_labels:
            valid_float = assignment_valid.to(static_rms_px.dtype)
            denominator = valid_float.sum(dim=0).clamp_min(1.0)
            support_static_error = (static_rms_px * valid_float).sum(dim=0) / denominator
            support_rigid_error = (rigid_rms_px * valid_float).sum(dim=0) / denominator
            support_improvement = support_static_error - support_rigid_error
            new_effective_mask = current_effective_mask.detach().clone()
            new_point_probability = new_effective_mask.to(canonical_xyz.dtype)
            if lagrangian:
                support_probability = new_point_probability
                effective_support_mask = new_effective_mask
            else:
                support_probability = aggregate_points_to_rigid_cells(
                    new_effective_mask.to(canonical_xyz.dtype),
                    point_group_ids,
                    num_supports,
                )
                effective_support_mask = support_probability >= 0.5
            assignment = {
                "dynamic_probability": support_probability.detach().cpu(),
                "effective_dynamic_mask": effective_support_mask.detach().cpu(),
                "rigid_vs_static_improvement": support_improvement.detach().cpu(),
                "valid_interval_count": assignment_valid.sum(dim=0).detach().cpu(),
                "improvement_per_interval": (static_rms_px - rigid_rms_px).detach().cpu(),
                "static_energy": support_static_error.detach().cpu(),
                "rigid_energy": support_rigid_error.detach().cpu(),
                "label_changes": 0,
                "core_promotions": 0,
                "core_demotions": 0,
                "consecutive_support_cycles": consecutive_support_cycles.cpu(),
                "frozen_prior_labels": True,
            }
            static_control_margin = 0.0
        elif freeze_depth_cores:
            valid_float = assignment_valid.to(static_rms_px.dtype)
            denominator = valid_float.sum(dim=0).clamp_min(1.0)
            support_static_error = (static_rms_px * valid_float).sum(dim=0) / denominator
            support_rigid_error = (rigid_rms_px * valid_float).sum(dim=0) / denominator
            support_margin = support_static_error - support_rigid_error
            support_informative = assignment_valid.any(dim=0)
            support_interval_valid = assignment_valid.T
            if lagrangian:
                point_static_error = support_static_error
                point_rigid_error = support_rigid_error
                point_margin = support_margin
                point_informative = support_informative
                interval_valid = support_interval_valid
                interval_margin = (static_rms_px - rigid_rms_px).T
            else:
                control_positions = torch.stack(static_control_positions)
                predicted_control_velocity = predict_rigid_cell_velocity(
                    mode, control_positions, speeds
                )
                control_rigid_rms = torch.stack(
                    [
                        velocity_reprojection_rms(
                            state,
                            predicted_control_velocity[index].to(
                                device=canonical_xyz.device,
                                dtype=canonical_xyz.dtype,
                            ),
                            delta_time=(
                                frame_times[index + 1] - frame_times[index]
                            ),
                        )
                        .detach()
                        .cpu()
                        .double()
                        for index, state in enumerate(
                            static_control_observation_states
                        )
                    ]
                )
                control_static_rms = torch.stack(
                    [
                        state["static_rms_px_mean"].detach().cpu().double()
                        for state in static_control_observation_states
                    ]
                )
                control_camera_count = torch.stack(
                    [
                        state["distinct_camera_count"].detach().cpu().long()
                        for state in static_control_observation_states
                    ]
                )
                control_valid = torch.stack(
                    [
                        state["base_valid"].detach().cpu().bool()
                        for state in static_control_observation_states
                    ]
                ) & (
                    control_camera_count
                    >= int(args.rigid_min_distinct_cameras)
                )
                control_valid_float = control_valid.to(control_static_rms.dtype)
                control_denominator = control_valid_float.sum(dim=0).clamp_min(1.0)
                point_static_error = (
                    control_static_rms * control_valid_float
                ).sum(dim=0) / control_denominator
                point_rigid_error = (
                    control_rigid_rms * control_valid_float
                ).sum(dim=0) / control_denominator
                point_margin = point_static_error - point_rigid_error
                point_informative = control_valid.any(dim=0)
                interval_valid = control_valid.T
                interval_margin = (
                    control_static_rms - control_rigid_rms
                ).T
            interval_margin = torch.where(
                interval_valid, interval_margin, torch.zeros_like(interval_margin)
            )
            informative_interval_count = interval_valid.sum(dim=1)
            time_persistence = (
                ((interval_margin > 0) & interval_valid).sum(dim=1).float()
                / informative_interval_count.clamp_min(1).float()
            )
            static_control = (
                (sequence_state.cpu() == STATIC_CORE) & point_informative
            )
            static_control_margin = (
                float(point_margin[static_control].median())
                if bool(static_control.any())
                else 0.0
            )
            frozen_static_control_diagnostics = {
                "informative_static_core_points": int(static_control.sum()),
                "static_hypothesis_error_median_px": (
                    float(point_static_error[static_control].median())
                    if bool(static_control.any())
                    else None
                ),
                "rigid_hypothesis_error_median_px": (
                    float(point_rigid_error[static_control].median())
                    if bool(static_control.any())
                    else None
                ),
                "static_minus_rigid_error_median_px": static_control_margin,
                "rigid_better_fraction": (
                    float((point_margin[static_control] > 0).float().mean())
                    if bool(static_control.any())
                    else None
                ),
                "rejection_margin_px": float(
                    args.rigid_assignment_improvement_margin
                ),
                "accepted": static_control_margin
                <= float(args.rigid_assignment_improvement_margin),
            }
            if static_control_margin > float(args.rigid_assignment_improvement_margin):
                raise RuntimeError(
                    "Frozen-core rigid solve rejected: rigid motion explains the "
                    "trusted static core better than the static hypothesis"
                )
            new_point_probability = current_point_probability.detach().clone()
            new_effective_mask = fit_core_mask.detach().clone()
            if lagrangian:
                support_probability = new_point_probability
                effective_support_mask = new_effective_mask
            else:
                support_probability = aggregate_points_to_rigid_cells(
                    new_point_probability, point_group_ids, num_supports
                )
                effective_support_mask = (
                    aggregate_points_to_rigid_cells(
                        new_effective_mask.to(canonical_xyz.dtype),
                        point_group_ids,
                        num_supports,
                    )
                    > 0
                )
            zero = torch.zeros_like(point_margin)
            assignment = {
                "state": sequence_state.detach().cpu(),
                "fit_core_mask": fit_core_mask.detach().cpu(),
                "effective_dynamic_mask": new_effective_mask.detach().cpu(),
                "dynamic_probability": new_point_probability.detach().cpu(),
                "margin": point_margin,
                "interval_margin": interval_margin,
                "camera_margin": torch.zeros(
                    (point_margin.numel(), len(frame_group_maps[0])),
                    dtype=point_margin.dtype,
                ),
                "time_persistence": time_persistence,
                "camera_persistence": zero,
                "speed_correlation": zero,
                "informative_interval_count": informative_interval_count,
                "static_energy": point_static_error,
                "rigid_energy": point_rigid_error,
                "label_changes": 0,
                "core_promotions": 0,
                "core_demotions": 0,
                "consecutive_support_cycles": consecutive_support_cycles.cpu(),
            }
        elif neutral_discovery:
            neutral_state = torch.full(
                (canonical_xyz.shape[0],),
                UNKNOWN,
                device=canonical_xyz.device,
                dtype=torch.uint8,
            )
            assignment = refine_assignment_with_rigid_hypotheses(
                canonical_xyz,
                mode,
                speeds,
                frame_times,
                frame_group_maps,
                correspondence_cache,
                {},
                neutral_state,
                consecutive_support_cycles,
                args,
                one_shot=True,
            )
            sequence_state = assignment["state"].to(
                device=canonical_xyz.device, dtype=torch.uint8
            )
            fit_core_mask = assignment["fit_core_mask"].to(
                canonical_xyz.device
            )
            new_point_probability = assignment["dynamic_probability"].to(
                canonical_xyz
            )
            new_effective_mask = assignment["effective_dynamic_mask"].to(
                canonical_xyz.device
            )
            if lagrangian:
                support_probability = new_point_probability
                effective_support_mask = new_effective_mask
            else:
                support_probability = aggregate_points_to_rigid_cells(
                    new_point_probability, point_group_ids, num_supports
                )
                effective_support_mask = (
                    aggregate_points_to_rigid_cells(
                        new_effective_mask.to(canonical_xyz.dtype),
                        point_group_ids,
                        num_supports,
                    )
                    > 0
                )
            static_control_margin = 0.0
        elif sequence_enabled:
            assignment = refine_assignment_with_rigid_hypotheses(
                canonical_xyz,
                mode,
                speeds,
                frame_times,
                frame_group_maps,
                correspondence_cache,
                sequence_calibrations,
                sequence_state,
                consecutive_support_cycles,
                args,
            )
            static_control = (
                (sequence_state == STATIC_CORE)
                & (assignment["informative_interval_count"].to(sequence_state.device) > 0)
            )
            if bool(static_control.any()):
                static_control_margin = float(
                    assignment["margin"].to(sequence_state.device)[static_control].median()
                )
                if static_control_margin > float(args.core_demotion_margin):
                    raise RuntimeError(
                        "Rigid solve rejected: the rigid hypothesis systematically "
                        "explains the trusted static core better"
                    )
            else:
                static_control_margin = 0.0
            sequence_state = assignment["state"].to(
                device=canonical_xyz.device, dtype=torch.uint8
            )
            consecutive_support_cycles = assignment[
                "consecutive_support_cycles"
            ].to(canonical_xyz.device)
            fit_core_mask = assignment["fit_core_mask"].to(canonical_xyz.device)
            new_point_probability = assignment["dynamic_probability"].to(canonical_xyz)
            new_effective_mask = assignment["effective_dynamic_mask"].to(
                canonical_xyz.device
            )
            if lagrangian:
                support_probability = new_point_probability
                effective_support_mask = new_effective_mask
            else:
                support_probability = aggregate_points_to_rigid_cells(
                    new_point_probability, point_group_ids, num_supports
                )
                effective_support_mask = (
                    aggregate_points_to_rigid_cells(
                        new_effective_mask.to(canonical_xyz.dtype),
                        point_group_ids,
                        num_supports,
                    )
                    > 0
                )
            if int(fit_core_mask.sum()) == 0:
                raise RuntimeError("Rigid solve aborted: trusted dynamic core collapsed")
        else:
            assignment = solve_rigid_assignment(
                static_rms_px,
                rigid_rms_px,
                weights=confidence,
                valid_mask=assignment_valid,
                prior_dynamic_probability=support_prior,
                improvement_margin=float(args.rigid_assignment_improvement_margin),
                temperature=float(args.rigid_assignment_temperature),
                prior_weight=float(args.rigid_assignment_prior_weight),
            )
            support_probability, physical_improvement = _smooth_rigid_cell_probability(
                assignment["dynamic_probability"].to(canonical_xyz),
                cell_graph,
                assignment["rigid_vs_static_improvement"].to(canonical_xyz),
                float(args.rigid_assignment_improvement_margin),
            )
            effective_support_mask = physical_improvement & (support_probability >= 0.5)
            if int(effective_support_mask.sum()) < int(args.rigid_min_valid_cells):
                raise RuntimeError("Rigid solve aborted: dynamic support collapsed")
            new_point_probability = support_probability.index_select(0, point_group_ids)
            new_effective_mask = effective_support_mask.index_select(0, point_group_ids)
            static_control_margin = 0.0
        runtime["assignment_seconds"] += phase_clock() - assignment_started

        previous_labels = current_effective_mask
        label_changes = int((new_effective_mask != previous_labels).sum())
        mode_change = float("inf")
        speed_change = float("inf")
        if current_mode is not None:
            old_generator = torch.cat(
                (current_mode.reference_velocity, current_mode.angular_mode)
            )
            new_generator = torch.cat((mode.reference_velocity, mode.angular_mode))
            mode_change = float(
                torch.linalg.norm(new_generator - old_generator)
                / torch.linalg.norm(old_generator).clamp_min(1.0e-12)
            )
            speed_change = float(
                torch.linalg.norm(speeds - current_speeds)
                / torch.linalg.norm(current_speeds).clamp_min(1.0e-12)
            )
        residual = rigid_residuals(mode, positions, observed_velocity, speeds)
        valid_residual = residual[solver_valid]
        rejected_static_like = sum(
            int(state["rejected_static_like"].sum()) for state in observation_states
        )
        cycle_stats = {
            "cycle": cycle + 1,
            "valid_supports": int(solver_valid.any(dim=0).sum()),
            "valid_cells": int(solver_valid.any(dim=0).sum()),
            "valid_observations": int(solver_valid.sum()),
            "rejected_static_like": rejected_static_like,
            "condition_number": condition_number,
            "rigid_residual_median": float(valid_residual.median()),
            "static_rms_px_mean": float(static_rms_px[assignment_valid].mean()),
            "rigid_rms_px_mean": float(rigid_rms_px[assignment_valid].mean()),
            "mode_change": mode_change,
            "speed_change": speed_change,
            "speed_mean": float(speeds.mean()),
            "speed_std": float(speeds.std(unbiased=False)),
            "speed_min": float(speeds.min()),
            "speed_max": float(speeds.max()),
            "hard_dynamic_supports": int(effective_support_mask.sum()),
            "hard_dynamic_cells": int(effective_support_mask.sum()),
            "hard_dynamic_points": int(new_effective_mask.sum()),
            "static_to_dynamic": int((new_effective_mask & ~previous_labels).sum()),
            "dynamic_to_static": int((~new_effective_mask & previous_labels).sum()),
            "label_changes": label_changes,
            "ambiguous_probabilities": int(
                ((new_point_probability > 0.4) & (new_point_probability < 0.6)).sum()
            ),
        }
        if gaussian_speed_observations is not None:
            local_valid = gaussian_speed_observations["valid"]
            cycle_stats.update({
                "gaussian_speed_observations": int(local_valid.sum()),
                "gaussian_speed_abs_error_median": float(
                    gaussian_speed_observations["speed_absolute_error"][local_valid].median()
                ) if bool(local_valid.any()) else 0.0,
                "gaussian_speed_perpendicular_median": float(
                    gaussian_speed_observations[
                        "perpendicular_velocity_residual"
                    ][local_valid].median()
                ) if bool(local_valid.any()) else 0.0,
            })
        if responsibility_training:
            static_count = int((sequence_state == STATIC_CORE).sum())
            rigid_count = int((sequence_state == DYNAMIC_CORE).sum())
            ambiguous_count = int((sequence_state == UNKNOWN).sum())
            responsibility_stats = {
                "responsibility/temperature": float(assignment["temperature"]),
                "responsibility/mean": float(new_point_probability.mean()),
                "responsibility/median": float(new_point_probability.median()),
                "responsibility/entropy": float(
                    assignment["classification_entropy"].mean()
                ),
                "responsibility/static_gaussians": static_count,
                "responsibility/rigid_gaussians": rigid_count,
                "responsibility/ambiguous_gaussians": ambiguous_count,
                "responsibility/ambiguous_fraction": (
                    ambiguous_count / max(int(canonical_xyz.shape[0]), 1)
                ),
                "responsibility/energy_margin_median": float(
                    assignment["normalized_energy_margin"].median()
                ),
                "responsibility/energy_margin_p10": float(
                    torch.quantile(
                        assignment["normalized_energy_margin"].float(), 0.10
                    )
                ),
                "responsibility/energy_margin_p90": float(
                    torch.quantile(
                        assignment["normalized_energy_margin"].float(), 0.90
                    )
                ),
                "responsibility/static_energy_median": float(
                    assignment["static_energy"].float().median()
                ),
                "responsibility/rigid_energy_median": float(
                    assignment["rigid_energy"].float().median()
                ),
                "responsibility/p10": float(
                    torch.quantile(new_point_probability.float(), 0.10)
                ),
                "responsibility/p90": float(
                    torch.quantile(new_point_probability.float(), 0.90)
                ),
                "responsibility/fit_weight_sum": float(fit_weight.sum()),
            }
            cycle_stats.update(responsibility_stats)
            cycle_payload = {
                "schema_version": 1,
                "cycle": cycle,
                "temperature": float(assignment["temperature"]),
                "gaussian_responsibility": new_point_probability.detach().cpu(),
                "raw_gaussian_responsibility": assignment["raw_probability"],
                "static_energy": assignment["static_energy"].detach().cpu(),
                "rigid_energy": assignment["rigid_energy"].detach().cpu(),
                "normalized_energy_margin": assignment[
                    "normalized_energy_margin"
                ],
                "state": sequence_state.detach().cpu(),
                "static_mask": sequence_state.detach().cpu() == STATIC_CORE,
                "rigid_mask": sequence_state.detach().cpu() == DYNAMIC_CORE,
                "ambiguous_mask": sequence_state.detach().cpu() == UNKNOWN,
                "ground_truth_used_as_algorithm_input": False,
            }
            if lagrangian:
                lagrangian_observation = {
                    "observation_count": torch.stack(
                        [state["observation_count"].detach().cpu() for state in observation_states]
                    ).sum(dim=0),
                    "camera_support": torch.stack(
                        [state["distinct_camera_count"].detach().cpu() for state in observation_states]
                    ).max(dim=0).values,
                    "condition_number": torch.nanmedian(
                        torch.stack(
                            [state["conditioning"].detach().cpu() for state in observation_states]
                        ).clamp_min(1.0e-12).reciprocal(),
                        dim=0,
                    ).values,
                    "observed_velocity": observed_velocity.detach().cpu(),
                    "observed_rigid_speed": gaussian_speed_observations[
                        "observed_gaussian_speed"
                    ].detach().cpu(),
                    "perpendicular_residual": gaussian_speed_observations[
                        "perpendicular_velocity_residual"
                    ].detach().cpu(),
                    "responsibility": new_point_probability.detach().cpu(),
                    "weak_conditioning_fraction": float(
                        (~observation_valid.any(dim=0)).float().mean()
                    ),
                }
                cycle_payload["lagrangian_observation"] = lagrangian_observation
                cycle_stats.update({
                    "lagrangian/valid_gaussian_fraction": float(
                        observation_valid.any(dim=0).float().mean()
                    ),
                    "lagrangian/weak_conditioning_fraction": lagrangian_observation[
                        "weak_conditioning_fraction"
                    ],
                    "lagrangian/observation_count_median": float(
                        lagrangian_observation["observation_count"].float().median()
                    ),
                    "lagrangian/camera_support_median": float(
                        lagrangian_observation["camera_support"].float().median()
                    ),
                })
                write_gaussian_responsibility_html(
                    canonical_xyz,
                    new_point_probability,
                    classification,
                    gaussian_dino_graph,
                    output_dir
                    / f"phase1_responsibility_cycle_{cycle:02d}_lagrangian.html",
                    title=(
                        f"Lagrangian Gaussian responsibility cycle {cycle}: DINO "
                        "locates boundaries; rigid-vs-static evidence identifies sides"
                    ),
                    max_points=int(args.transported_html_max_points),
                )
            else:
                support_probability_stats = compute_support_responsibility_stats(
                    eulerian_support_state,
                    new_point_probability,
                    static_threshold=float(args.eulerian_contrastive_static_threshold),
                    rigid_threshold=float(args.eulerian_contrastive_rigid_threshold),
                )
                support_observation = support_observation_diagnostics(
                    eulerian_support_state, observation_states
                )
                cycle_payload["eulerian_support"] = {
                    **support_state_history[-1],
                    **support_probability_stats,
                    **support_observation,
                    "cell_dino_dispersion": eulerian_support_state[
                        "cell_dino_dispersion"
                    ],
                    "weak_support": eulerian_support_state["weak_support"],
                }
                cycle_stats.update({
                    "eulerian_support/count": num_supports,
                    "eulerian_support/dino_splits": sum(
                        reason == "dino"
                        for reason in eulerian_support_state["split_reason"]
                    ),
                    "eulerian_support/responsibility_splits": sum(
                        reason == "responsibility"
                        for reason in eulerian_support_state["split_reason"]
                    ),
                    "eulerian_support/membership_changes": int(
                        eulerian_support_state["membership_changes"]
                    ),
                    "eulerian_support/variance_median": float(
                        support_probability_stats["variance_p"].median()
                    ),
                    "eulerian_support/valid_fraction": support_observation[
                        "valid_support_fraction"
                    ],
                })
                write_eulerian_support_html(
                    canonical_xyz,
                    new_point_probability,
                    classification,
                    eulerian_support_state,
                    gaussian_dino_graph,
                    observation_grid.cell_centers,
                    observation_grid.cell_size,
                    output_dir
                    / f"phase1_responsibility_cycle_{cycle:02d}_eulerian_support.html",
                    title=(
                        f"Eulerian contrastive support cycle {cycle}: DINO locates "
                        "boundaries; rigid-vs-static evidence identifies each side"
                    ),
                    max_points=int(args.transported_html_max_points),
                )
            _atomic_torch_save(
                cycle_payload,
                output_dir / f"phase1_responsibility_cycle_{cycle:02d}.pt",
            )
            assignment_state_history.append({
                "cycle": cycle,
                "state": sequence_state.detach().cpu(),
                "dynamic_probability": new_point_probability.detach().cpu(),
                "margin": assignment["normalized_energy_margin"],
            })
        elif neutral_discovery:
            cycle_stats.update({
                "neutral_discovery/initial_probability_max_error": float(
                    (initial_point_probability - 0.5).abs().max()
                ),
                "neutral_discovery/fit_foreground_points": int(
                    canonical_xyz.shape[0]
                ),
                "neutral_discovery/label_updates_during_fit": 0,
                "neutral_discovery/postsolve_static_count": int(
                    (sequence_state == STATIC_CORE).sum()
                ),
                "neutral_discovery/postsolve_dynamic_count": int(
                    (sequence_state == DYNAMIC_CORE).sum()
                ),
                "neutral_discovery/postsolve_unknown_count": int(
                    (sequence_state == UNKNOWN).sum()
                ),
                "neutral_discovery/postsolve_label_changes": int(
                    assignment["label_changes"]
                ),
                "neutral_discovery/margin_median": float(
                    assignment["margin"].median()
                ),
                "neutral_discovery/speed_correlation_median": float(
                    assignment["speed_correlation"].median()
                ),
            })
            assignment_state_history.append({
                "cycle": 1,
                "stage": "postsolve_only",
                "state": sequence_state.detach().cpu(),
                "dynamic_probability": new_point_probability.detach().cpu(),
                "margin": assignment["margin"].detach().cpu(),
            })
        elif sequence_enabled and freeze_depth_cores:
            cycle_stats.update({
                "rigid_fit/frozen_dynamic_core_size": int(
                    (sequence_state == DYNAMIC_CORE).sum()
                ),
                "rigid_fit/frozen_static_core_size": int(
                    (sequence_state == STATIC_CORE).sum()
                ),
                "rigid_fit/frozen_unknown_size": int(
                    (sequence_state == UNKNOWN).sum()
                ),
                "rigid_fit/label_changes": label_changes,
                "rigid_fit/static_control_informative_points": (
                    frozen_static_control_diagnostics[
                        "informative_static_core_points"
                    ]
                ),
                "rigid_fit/static_core_rigid_margin_px": static_control_margin,
            })
        elif sequence_enabled:
            assigned_sequence_energy = float(
                torch.where(
                    new_effective_mask,
                    assignment["rigid_energy"].to(new_effective_mask.device),
                    assignment["static_energy"].to(new_effective_mask.device),
                ).sum()
            )
            if (
                previous_sequence_energy is not None
                and assigned_sequence_energy
                > previous_sequence_energy + 1.0e-4 * max(1.0, abs(previous_sequence_energy))
            ):
                sequence_energy_worsenings += 1
            else:
                sequence_energy_worsenings = 0
            previous_sequence_energy = assigned_sequence_energy
            if sequence_energy_worsenings >= 2:
                raise RuntimeError("Rigid solve rejected: sequence energy worsened repeatedly")
            cycle_stats.update({
                "rigid_fit/dynamic_core_size": int((sequence_state == DYNAMIC_CORE).sum()),
                "assignment_refine/dynamic_candidate_size": int((sequence_state == DYNAMIC_CANDIDATE).sum()),
                "assignment_refine/static_core_size": int((sequence_state == STATIC_CORE).sum()),
                "assignment_refine/label_changes": int(assignment["label_changes"]),
                "assignment_refine/core_promotions": int(assignment["core_promotions"]),
                "assignment_refine/core_demotions": int(assignment["core_demotions"]),
                "assignment_refine/temporal_persistence_median": float(assignment["time_persistence"].median()),
                "assignment_refine/camera_persistence_median": float(assignment["camera_persistence"].median()),
                "assignment_refine/speed_correlation_median": float(assignment["speed_correlation"].median()),
                "sequence/static_vs_rigid_energy": float(
                    assignment["static_energy"].sum() - assignment["rigid_energy"].sum()
                ),
                "sequence/energy": assigned_sequence_energy,
                "sequence/static_core_rigid_margin": static_control_margin,
            })
            assignment_state_history.append({
                "cycle": cycle + 1,
                "state": sequence_state.detach().cpu(),
                "dynamic_probability": new_point_probability.detach().cpu(),
                "margin": assignment["margin"].detach().cpu(),
                "time_persistence": assignment["time_persistence"].detach().cpu(),
                "camera_persistence": assignment["camera_persistence"].detach().cpu(),
                "speed_correlation": assignment["speed_correlation"].detach().cpu(),
            })
        cycle_history.append(cycle_stats)
        if log_fn is not None:
            logged_stats = cycle_stats
            if lagrangian:
                logged_stats = {
                    f"lagrangian/{key}": value for key, value in cycle_stats.items()
                }
            log_fn(
                "lagrangian_rigid_killing" if lagrangian else "rigid_killing",
                cycle + 1,
                logged_stats,
            )
        print(
            (
                "[Rigid responsibility] "
                if responsibility_training
                else (
                    "[Rigid Killing neutral discovery] "
                    if neutral_discovery
                    else (
                        "[Rigid Killing frozen-core] "
                        if freeze_depth_cores
                        else (
                            "[Rigid Killing frozen-prior] "
                            if freeze_prior_labels
                            else "[Rigid Killing EM] "
                        )
                    )
                )
            )
            + f"cycle={cycle + 1} valid={cycle_stats['valid_supports']} "
            f"dynamic={cycle_stats['hard_dynamic_supports']} "
            f"rigid_px={cycle_stats['rigid_rms_px_mean']:.4f} "
            f"speed={cycle_stats['speed_mean']:.6f}"
        )
        rigid_progress.set_postfix(
            core=(
                int((sequence_state == DYNAMIC_CORE).sum())
                if sequence_enabled
                else cycle_stats["hard_dynamic_points"]
            ),
            dynamic=cycle_stats["hard_dynamic_points"],
            rigid_px=f"{cycle_stats['rigid_rms_px_mean']:.3f}",
            changes=cycle_stats["label_changes"],
        )

        current_mode, current_speeds = mode, speeds
        current_point_probability = new_point_probability.detach()
        current_effective_mask = new_effective_mask.detach()
        final_observations = observation_states
        final_positions = positions
        final_rigid_rms_px = rigid_rms_px
        final_assignment = assignment
        final_gaussian_speed_observations = gaussian_speed_observations
        final_gaussian_query_positions = gaussian_query_positions
        final_gaussian_query_rotations = gaussian_query_rotations
        if freeze_prior_labels and label_changes == 0:
            stable_cycles += 1
        elif label_changes == 0 and mode_change < float(args.rigid_solver_tol) and speed_change < float(args.rigid_solver_tol):
            stable_cycles += 1
        else:
            stable_cycles = 0
        if stable_cycles >= 1 and not responsibility_training:
            break

    if current_mode is None or final_observations is None:
        raise RuntimeError("Rigid Phase 1 produced no solution")
    if freeze_depth_cores:
        if not torch.equal(sequence_state, frozen_sequence_state):
            raise RuntimeError("Frozen depth-core state changed during rigid fitting")
        if not torch.equal(
            current_effective_mask, frozen_sequence_state == DYNAMIC_CORE
        ):
            raise RuntimeError("Frozen depth-core dynamic mask changed during rigid fitting")
        if not torch.equal(current_point_probability, frozen_probability):
            raise RuntimeError("Frozen depth-core probability changed during rigid fitting")
    if neutral_discovery and len(cycle_history) != 1:
        raise RuntimeError("Neutral rigid discovery must perform exactly one rigid fit")
    if (
        responsibility_training
        and len(cycle_history) != int(args.rigid_responsibility_cycles)
    ):
        raise RuntimeError("Soft rigid responsibility training did not complete its schedule")

    mode_diagnostics = {
        **current_mode.diagnostics,
        "mode_type": current_mode.mode_type,
        "normalization_scale": current_mode.normalization_scale,
        "em_cycles": len(cycle_history),
    }
    gaussians.set_rigid_killing_state(
        current_mode,
        current_speeds,
        torch.as_tensor(frame_times),
        current_point_probability,
        current_effective_mask,
        mode_diagnostics,
    )
    if sequence_enabled or neutral_discovery or responsibility_training:
        gaussians.sequence_assignment_state = {
            "state": sequence_state.detach().cpu(),
            "dynamic_probability": current_point_probability.detach().cpu(),
            "effective_dynamic_mask": current_effective_mask.detach().cpu(),
            "fit_core_mask": fit_core_mask.detach().cpu(),
            "consecutive_support_cycles": consecutive_support_cycles.detach().cpu(),
        }
    measurement_source_id = str(
        final_observations[0]["flow_measurement_source_id"]
    )

    replay_started = phase_clock()
    projected_flow_audit = []
    with torch.no_grad():
        for target_index in range(1, len(frame_groups)):
            source_index = target_index - 1
            source_xyz, source_rotation, point_velocity = _rigid_replay_to_frame(
                canonical_xyz,
                canonical_rotation,
                current_mode,
                current_speeds,
                frame_times,
                source_index,
                current_effective_mask,
            )
            for camera_key in sorted(frame_group_maps[source_index]):
                source_camera = frame_group_maps[source_index][camera_key]
                target_camera = frame_group_maps[target_index][camera_key]
                observation = correspondence_cache.get(
                    source_camera, target_camera, canonical_xyz.device
                )
                transform = sequence_observation_transform(source_index)
                if transform is not None:
                    observation = transform(source_camera, target_camera, observation)
                _, stats = rasterized_projected_velocity_flow_loss(
                    source_camera,
                    target_camera,
                    gaussians,
                    pipe,
                    source_xyz,
                    point_velocity,
                    observation,
                    args,
                    delta_time=frame_times[target_index] - frame_times[source_index],
                    query_rotation=source_rotation,
                    validation_only=True,
                    flow_measurement_source_id=measurement_source_id,
                )
                projected_flow_audit.append(
                    {
                        "interval": source_index,
                        "camera": camera_key,
                        **stats,
                    }
                )

    positions_history = []
    rotations_history = []
    velocities_history = []
    angular_history = []
    discovery_positions_history = []
    discovery_rotations_history = []
    discovery_velocities_history = []
    discovery_angular_history = []
    all_foreground_mask = torch.ones_like(current_effective_mask)
    for frame_index in range(len(frame_groups)):
        xyz, rotation, velocity = _rigid_replay_to_frame(
            canonical_xyz,
            canonical_rotation,
            current_mode,
            current_speeds,
            frame_times,
            frame_index,
            current_effective_mask,
        )
        speed_index = min(frame_index, len(current_speeds) - 1)
        angular = current_mode.angular_mode.to(velocity) * float(
            current_speeds[speed_index]
        )
        angular = angular[None].expand_as(velocity) * current_effective_mask[:, None].to(
            velocity
        )
        positions_history.append(xyz.detach().cpu())
        rotations_history.append(rotation.detach().cpu())
        velocities_history.append(velocity.detach().cpu())
        angular_history.append(angular.detach().cpu())
        if neutral_discovery or responsibility_training:
            all_xyz, all_rotation, all_velocity = _rigid_replay_to_frame(
                canonical_xyz,
                canonical_rotation,
                current_mode,
                current_speeds,
                frame_times,
                frame_index,
                all_foreground_mask,
            )
            all_angular = current_mode.angular_mode.to(all_velocity) * float(
                current_speeds[speed_index]
            )
            discovery_positions_history.append(all_xyz.detach().cpu())
            discovery_rotations_history.append(all_rotation.detach().cpu())
            discovery_velocities_history.append(all_velocity.detach().cpu())
            discovery_angular_history.append(
                all_angular[None].expand_as(all_velocity).detach().cpu()
            )

    if responsibility_training and not lagrangian:
        pooling_started = phase_clock()
        gaussian_reference_states = []
        for target_index in range(1, len(frame_groups)):
            source_index = target_index - 1
            source_xyz = final_gaussian_query_positions[source_index].to(
                canonical_xyz
            )
            source_rotation = final_gaussian_query_rotations[source_index].to(
                canonical_rotation
            )
            source_by_camera = frame_group_maps[source_index]
            target_by_camera = frame_group_maps[target_index]
            gaussian_reference_states.append(
                build_gaussian_motion_observations(
                    correspondence_cache,
                    source_xyz,
                    list(source_by_camera.values()),
                    list(target_by_camera.values()),
                    frame_times[source_index],
                    frame_times[target_index],
                    gaussians=gaussians,
                    pipe=pipe,
                    query_rotations=source_rotation,
                )
            )

        reference_valid = torch.stack([
            state["base_valid"].detach().cpu().bool()
            for state in gaussian_reference_states
        ])
        reference_conditioning = torch.stack([
            state["conditioning"].detach().cpu().float()
            for state in gaussian_reference_states
        ])
        reference_condition_number = torch.nanmedian(
            torch.where(
                reference_valid,
                reference_conditioning.clamp_min(1.0e-12).reciprocal(),
                torch.full_like(reference_conditioning, float("nan")),
            ),
            dim=0,
        ).values
        reference_observation_count = torch.stack([
            state["observation_count"].detach().cpu()
            for state in gaussian_reference_states
        ]).sum(dim=0)
        reference_camera_support = torch.stack([
            state["distinct_camera_count"].detach().cpu()
            for state in gaussian_reference_states
        ]).max(dim=0).values
        reference_uncertainty = torch.stack([
            state["weight_sum"].detach().cpu().float()
            for state in gaussian_reference_states
        ]).sum(dim=0).clamp_min(1.0e-8).rsqrt()
        pooled = support_observation_diagnostics(
            eulerian_support_state, final_observations
        )

        def finite_median(value):
            tensor = torch.as_tensor(value).float().reshape(-1)
            tensor = tensor[torch.isfinite(tensor)]
            return float(tensor.median()) if tensor.numel() else None

        def safe_ratio(numerator, denominator):
            if numerator is None or denominator is None or abs(denominator) < 1.0e-12:
                return None
            return numerator / denominator

        direct_count = finite_median(reference_observation_count)
        pooled_count = finite_median(pooled["observation_count"])
        direct_cameras = finite_median(reference_camera_support)
        pooled_cameras = finite_median(pooled["distinct_camera_count"])
        direct_condition = finite_median(reference_condition_number)
        pooled_condition = finite_median(pooled["condition_number"])
        direct_uncertainty = finite_median(reference_uncertainty)
        pooled_uncertainty = finite_median(pooled["velocity_target_uncertainty"])
        pooling_comparison = {
            "per_gaussian_observation_count_median": direct_count,
            "pooled_support_observation_count_median": pooled_count,
            "median_observation_count_gain": safe_ratio(pooled_count, direct_count),
            "per_gaussian_camera_support_median": direct_cameras,
            "pooled_support_camera_count_median": pooled_cameras,
            "median_camera_support_gain": safe_ratio(pooled_cameras, direct_cameras),
            "per_gaussian_condition_number_median": direct_condition,
            "pooled_support_condition_number_median": pooled_condition,
            "median_condition_number_ratio": safe_ratio(
                direct_condition, pooled_condition
            ),
            "per_gaussian_velocity_uncertainty_median": direct_uncertainty,
            "pooled_support_velocity_uncertainty_median": pooled_uncertainty,
            "median_uncertainty_ratio": safe_ratio(
                direct_uncertainty, pooled_uncertainty
            ),
            "per_gaussian_valid_fraction": float(
                reference_valid.any(dim=0).float().mean()
            ),
            "pooled_support_valid_fraction": pooled["valid_support_fraction"],
        }
        eulerian_gaussian_reference = {
            "schema_version": 1,
            "role": "postsolve_pooling_diagnostic_only",
            "ground_truth_used": False,
            "query_positions": torch.stack(final_gaussian_query_positions),
            "observation_count": reference_observation_count,
            "distinct_camera_count": reference_camera_support,
            "condition_number": reference_condition_number,
            "velocity_target_uncertainty": reference_uncertainty,
            "valid": reference_valid,
            "observed_velocity": torch.stack([
                state["velocity"].detach().cpu()
                for state in gaussian_reference_states
            ]),
            "comparison": pooling_comparison,
        }
        del gaussian_reference_states
        runtime["pooling_diagnostic_seconds"] = phase_clock() - pooling_started

    displacement = torch.linalg.norm(
        positions_history[-1] - positions_history[0], dim=-1
    )
    static_displacement = displacement[~current_effective_mask.cpu()]
    max_static_displacement = (
        float(static_displacement.max()) if static_displacement.numel() else 0.0
    )
    if max_static_displacement > float(args.rigid_static_displacement_tolerance):
        raise RuntimeError(
            f"Effective-static transport contract failed: {max_static_displacement:.3e}"
        )
    pairwise_error = _rigid_pairwise_distance_error(
        positions_history,
        current_effective_mask.cpu(),
        int(args.rigid_pairwise_audit_points),
    )
    if pairwise_error > 1.0e-5:
        raise RuntimeError(
            f"Exact rigid replay failed pairwise-distance preservation: {pairwise_error:.3e}"
        )
    quaternion_norm_error = float(
        torch.stack(rotations_history).norm(dim=-1).sub(1.0).abs().max()
    )
    if quaternion_norm_error > 1.0e-5:
        raise RuntimeError(
            f"Exact rigid replay produced invalid quaternion norms: {quaternion_norm_error:.3e}"
        )

    if lagrangian:
        gradient_diagnostics = compute_lagrangian_gradient_diagnostics(
            canonical_xyz,
            current_mode,
            current_speeds,
            current_effective_mask,
            neighbors_per_gaussian=int(args.curl_material_neighbors),
            max_pairs=int(args.phase1_pair_max_pairs),
            min_neighbors=int(args.eulerian_velocity_gradient_min_neighbors),
            relative_ridge=float(args.eulerian_velocity_gradient_ridge),
        )
    else:
        gradient_diagnostics = rigid_killing_gradient_diagnostics(
            current_mode.angular_mode, current_speeds
        )
    runtime["final_replay_seconds"] = (
        phase_clock()
        - replay_started
        - runtime.get("pooling_diagnostic_seconds", 0.0)
    )
    runtime["rigid_phase_seconds_before_artifact_write"] = (
        phase_clock() - phase_started
    )
    runtime["peak_gpu_memory_bytes"] = (
        int(torch.cuda.max_memory_allocated(canonical_xyz.device))
        if canonical_xyz.is_cuda
        else None
    )
    # Linux reports ru_maxrss in KiB. This repository targets CUDA/Linux.
    runtime["peak_cpu_memory_bytes"] = int(
        resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
    )
    final_probabilities = torch.stack(
        (1.0 - current_point_probability, current_point_probability), dim=-1
    )
    final_labels = current_effective_mask.long()
    final_tracks = {
        "schema_version": 15,
        "dynamic_phase": "phase1",
        "motion_representation": representation,
        "times": torch.as_tensor(frame_times, dtype=torch.float32),
        "positions": torch.stack(positions_history),
        "rotations": torch.stack(rotations_history),
        "velocities": torch.stack(velocities_history),
        "angular_velocities": torch.stack(angular_history),
        "cell_ids": (
            torch.full(
                (len(frame_groups), canonical_xyz.shape[0]), -1, dtype=torch.long
            )
            if lagrangian
            else point_group_ids.detach().cpu()[None].expand(len(frame_groups), -1)
        ),
        "geometric_cell_ids": (
            None if lagrangian else geometric_cell_ids.detach().cpu()
        ),
        "contrastive_support_ids": (
            None
            if lagrangian or not responsibility_training
            else point_group_ids.detach().cpu()
        ),
        "gaussian_ids": gaussians._foreground_source_indices.detach().cpu(),
        "canonical_xyz": canonical_xyz.cpu(),
        "foreground_source_indices": gaussians._foreground_source_indices.detach().cpu(),
        "semantic_probability": None
        if gaussians.get_semantic_probability is None
        else gaussians.get_semantic_probability.detach().cpu(),
        "motion_evidence": gaussians.get_motion_evidence.detach().cpu(),
        "initial_dynamic_mask": gaussians.get_initial_dynamic_mask.detach().cpu(),
        "initial_dynamic_probability": initial_point_probability.cpu(),
        "P_dynamic": current_point_probability.cpu(),
        "effective_dynamic_mask": current_effective_mask.cpu(),
        "motion_membership": current_point_probability.cpu(),
        "prior_labels_frozen": freeze_prior_labels,
        "part_probabilities": final_probabilities.cpu(),
        "part_labels": final_labels.cpu(),
        "kinematic_report": None,
        "fixed_cameras": _fixed_camera_metadata(cameras),
        "rigidity_definition": "v(x,t)=s(t)[u_ref+omega_bar cross (x-x_ref)]",
    }
    cumulative_q = torch.cat((
        torch.zeros(1, dtype=torch.float64),
        torch.cumsum(
            current_speeds * torch.diff(torch.as_tensor(frame_times, dtype=torch.float64)),
            dim=0,
        ),
    ))
    part_quaternions, part_translations = [], []
    mode_device = current_mode.to(device=canonical_xyz.device, dtype=canonical_xyz.dtype)
    for coordinate in cumulative_q:
        transform = integrate_rigid_mode_exact(
            canonical_xyz, mode_device, coordinate.to(canonical_xyz)
        )
        reference = mode_device.reference_point
        world_translation = (
            reference
            - transform["rotation_matrix"] @ reference
            + transform["translation"]
        )
        part_quaternions.append(transform["rotation_quaternion"].cpu())
        part_translations.append(world_translation.cpu())
    final_tracks["part_rotation_quaternions"] = torch.stack(part_quaternions)[:, None]
    final_tracks["part_translations"] = torch.stack(part_translations)[:, None]
    quaternion_audit = write_quaternion_transport_audit(
        final_tracks,
        output_dir,
        scales=gaussians.get_scaling.detach().cpu(),
        max_debug_gaussians=min(64, int(args.transported_html_max_points)),
    )
    if log_fn is not None:
        log_fn("quaternion_transport", len(cycle_history), {
            "quaternion_transport/max_norm_error": quaternion_audit["max_quaternion_norm_error"],
            "quaternion_transport/spatial_rotation_variance": quaternion_audit["spatial_rigid_rotation_variance_max"],
            "quaternion_transport/part_quaternion_agreement": quaternion_audit["part_quaternion_agreement_error_max"],
            "quaternion_transport/covariance_error": quaternion_audit["covariance_transport_error_max"],
        })
    discovery_tracks = None
    if neutral_discovery:
        neutral_probability = torch.full_like(current_point_probability.cpu(), 0.5)
        discovery_tracks = {
            **final_tracks,
            "positions": torch.stack(discovery_positions_history),
            "rotations": torch.stack(discovery_rotations_history),
            "velocities": torch.stack(discovery_velocities_history),
            "angular_velocities": torch.stack(discovery_angular_history),
            "initial_dynamic_probability": neutral_probability,
            "P_dynamic": neutral_probability,
            "effective_dynamic_mask": all_foreground_mask.cpu(),
            "motion_membership": neutral_probability,
            "part_probabilities": torch.full_like(final_probabilities.cpu(), 0.5),
            "part_labels": torch.ones_like(final_labels.cpu()),
        }
    elif responsibility_training:
        discovery_tracks = {
            **final_tracks,
            "positions": torch.stack(discovery_positions_history),
            "rotations": torch.stack(discovery_rotations_history),
            "velocities": torch.stack(discovery_velocities_history),
            "angular_velocities": torch.stack(discovery_angular_history),
            "effective_dynamic_mask": all_foreground_mask.cpu(),
            "part_labels": torch.ones_like(final_labels.cpu()),
        }

    observation_payload = {
        "schema_version": 1,
        "support_type": support_type_name,
        "flow_measurement_source_id": measurement_source_id,
        "support_positions_canonical": canonical_support_positions,
        "cell_centers_canonical": (
            None if lagrangian else observation_grid.cell_centers
        ),
        "point_group_ids": (
            None if lagrangian else point_group_ids.detach().cpu()
        ),
        "geometric_cell_ids": (
            None if lagrangian else geometric_cell_ids.detach().cpu()
        ),
        "gaussian_ids": (
            gaussians._foreground_source_indices.detach().cpu()
            if lagrangian
            else None
        ),
        "support_positions": final_positions,
        "cell_positions": None if lagrangian else final_positions,
        "velocity": torch.stack(
            [state["velocity"].detach().cpu() for state in final_observations]
        ),
        "confidence": torch.stack(
            [state["confidence"].detach().cpu() for state in final_observations]
        ),
        "valid": torch.stack(
            [state["valid"].detach().cpu() for state in final_observations]
        ),
        "accepted": torch.stack(
            [state["valid"].detach().cpu() for state in final_observations]
        ),
        "solver_valid": solver_valid.detach().cpu(),
        "base_valid": torch.stack(
            [state["base_valid"].detach().cpu() for state in final_observations]
        ),
        "observation_count": torch.stack(
            [state["observation_count"].detach().cpu() for state in final_observations]
        ),
        "weight_sum": torch.stack(
            [state["weight_sum"].detach().cpu() for state in final_observations]
        ),
        "distinct_camera_count": torch.stack(
            [state["distinct_camera_count"].detach().cpu() for state in final_observations]
        ),
        "camera_support_mask": torch.stack(
            [state["camera_support_mask"].detach().cpu() for state in final_observations]
        ),
        "conditioning": torch.stack(
            [state["conditioning"].detach().cpu() for state in final_observations]
        ),
        "condition_number": torch.stack(
            [
                state["conditioning"].detach().cpu().clamp_min(1.0e-12).reciprocal()
                for state in final_observations
            ]
        ),
        "static_rms_px": torch.stack(
            [state["static_rms_px_mean"].detach().cpu() for state in final_observations]
        ),
        "moving_rms_px": torch.stack(
            [state["moving_rms_px"].detach().cpu() for state in final_observations]
        ),
        "rigid_rms_px": final_rigid_rms_px,
        "static_improvement_px": torch.stack(
            [state["static_improvement_px"].detach().cpu() for state in final_observations]
        ),
        "static_improvement_relative": torch.stack(
            [
                state["static_improvement_relative"].detach().cpu()
                for state in final_observations
            ]
        ),
        "rejected_static_like": torch.stack(
            [state["rejected_static_like"].detach().cpu() for state in final_observations]
        ),
        "reprojection_rms_px": torch.stack(
            [state["reprojection_rms_px"].detach().cpu() for state in final_observations]
        ),
        "rigid_residual": rigid_residuals(
            current_mode,
            final_positions,
            torch.stack(
                [state["velocity"].detach().cpu().double() for state in final_observations]
            ),
            current_speeds,
        ),
        "normal_matrix": torch.stack(
            [state["normal_matrix"].detach().cpu() for state in final_observations]
        ),
        "rhs": torch.stack(
            [state["rhs"].detach().cpu() for state in final_observations]
        ),
        "target_y2": torch.stack(
            [state["target_y2"].detach().cpu() for state in final_observations]
        ),
        "projected_flow_validation": projected_flow_audit,
    }
    assignment_payload = {
        "schema_version": 1,
        "foreground_source_indices": gaussians._foreground_source_indices.detach().cpu(),
        "initial_dynamic_probability": initial_point_probability.cpu(),
        "P_dynamic": current_point_probability.cpu(),
        "effective_dynamic_mask": current_effective_mask.cpu(),
        "displayed_labels": final_labels.cpu(),
        "support_type": support_type_name,
        "prior_labels_frozen": freeze_prior_labels,
        "support_dynamic_probability": (
            current_point_probability.detach().cpu()
            if lagrangian
            else aggregate_points_to_rigid_cells(
                current_point_probability, point_group_ids, num_supports
            ).detach().cpu()
        ),
        "gaussian_dynamic_probability": current_point_probability.detach().cpu(),
        "cell_dynamic_probability": (
            None
            if lagrangian
            else aggregate_points_to_rigid_cells(
                current_point_probability, point_group_ids, num_supports
            ).detach().cpu()
        ),
        "rigid_vs_static_improvement": (
            final_assignment["margin"].detach().cpu()
            if sequence_enabled or neutral_discovery or responsibility_training
            else final_assignment["rigid_vs_static_improvement"]
        ),
        "persistent_state": (
            sequence_state.detach().cpu()
            if sequence_enabled or neutral_discovery or responsibility_training
            else None
        ),
        "history": cycle_history,
    }
    speed_payload = {
        "schema_version": 1,
        "frame_times": torch.as_tensor(frame_times, dtype=torch.float64),
        "interval_speeds": current_speeds,
        "delta_q": current_speeds
        * torch.diff(torch.as_tensor(frame_times, dtype=torch.float64)),
    }
    mode_payload = {
        "schema_version": 1,
        **current_mode.as_dict(),
        "solver": solver_diagnostics,
        "source_observation_metadata": {
            "flow_measurement_source_id": measurement_source_id,
            "intervals": len(frame_groups) - 1,
            "physical_cameras": len(frame_groups[0][2]),
            "support_type": support_type_name,
            "observation_supports": num_supports,
            "observation_cells": (
                None if lagrangian else observation_grid.num_cells
            ),
            "contrastive_supports": (
                num_supports
                if responsibility_training and not lagrangian
                else None
            ),
            "observation_gaussians": num_supports if lagrangian else None,
        },
    }
    final_summary = {
        "schema_version": 1,
        "representation": representation,
        "support_type": support_type_name,
        "number_of_support_units": num_supports,
        "mode_type": current_mode.mode_type,
        "reference_point": current_mode.reference_point.tolist(),
        "reference_velocity": current_mode.reference_velocity.tolist(),
        "angular_mode": current_mode.angular_mode.tolist(),
        "angular_norm": float(torch.linalg.norm(current_mode.angular_mode)),
        "speed_mean": float(current_speeds.mean()),
        "speed_std": float(current_speeds.std(unbiased=False)),
        "speed_min": float(current_speeds.min()),
        "speed_max": float(current_speeds.max()),
        "effective_dynamic_points": int(current_effective_mask.sum()),
        "effective_static_points": int((~current_effective_mask).sum()),
        "max_static_displacement": max_static_displacement,
        "moving_displacement_median": (
            float(displacement[current_effective_mask.cpu()].median())
            if bool(current_effective_mask.any())
            else None
        ),
        "pairwise_distance_error_max": pairwise_error,
        "quaternion_norm_error_max": quaternion_norm_error,
        "quaternion_transport": quaternion_audit,
        "deformation_frobenius_max": float(
            gradient_diagnostics["deformation_frobenius"][:, gradient_diagnostics["gradient_valid"]].max()
        ) if lagrangian and bool(gradient_diagnostics["gradient_valid"].any()) else float(
            gradient_diagnostics["deformation_frobenius"].max()
        ),
        "omega_spatial_variance_max": float(
            gradient_diagnostics["omega_spatial_variance"].max()
        ),
        "rigidity_matrix_residual_max": (
            float(gradient_diagnostics["rigidity_matrix_residual"].max())
            if lagrangian
            else None
        ),
        "gaussian_speed_observation_count": (
            int(final_gaussian_speed_observations["valid"].sum())
            if lagrangian and final_gaussian_speed_observations is not None
            else None
        ),
        "gaussian_speed_rmse_median": (
            float(
                final_gaussian_speed_observations["speed_rmse"][
                    final_gaussian_speed_observations["informative_interval_count"] > 0
                ].median()
            )
            if lagrangian
            and final_gaussian_speed_observations is not None
            and bool(
                (
                    final_gaussian_speed_observations["informative_interval_count"]
                    > 0
                ).any()
            )
            else None
        ),
        "em_history": cycle_history,
        "sequence_assignment_enabled": sequence_enabled,
        "neutral_rigid_discovery": neutral_discovery,
        "rigid_responsibility_training": responsibility_training,
        "responsibility_cycles": (
            len(cycle_history) if responsibility_training else None
        ),
        "responsibility_mean": (
            float(current_point_probability.mean())
            if responsibility_training
            else None
        ),
        "responsibility_entropy": (
            float(final_assignment["classification_entropy"].mean())
            if responsibility_training
            else None
        ),
        "ambiguous_fraction": (
            float((sequence_state == UNKNOWN).float().mean())
            if responsibility_training
            else None
        ),
        "neutral_initial_probability": 0.5 if neutral_discovery else None,
        "assignment_updates_during_discovery": 0 if neutral_discovery else None,
        "postsolve_assignment_enabled": neutral_discovery,
        "depth_cores_frozen": freeze_depth_cores,
        "prior_labels_frozen": freeze_prior_labels,
        "label_updates_enabled": not (freeze_depth_cores or freeze_prior_labels),
        "frozen_core_label_invariant": (
            bool(
                torch.equal(sequence_state, frozen_sequence_state)
                and torch.equal(
                    current_effective_mask,
                    frozen_sequence_state == DYNAMIC_CORE,
                )
                and torch.equal(current_point_probability, frozen_probability)
            )
            if freeze_depth_cores
            else None
        ),
        "frozen_prior_label_invariant": (
            bool(torch.equal(current_effective_mask, gaussians.get_initial_dynamic_mask.detach()))
            if freeze_prior_labels
            else None
        ),
        "frozen_static_control": frozen_static_control_diagnostics,
        "dynamic_core_points": int((sequence_state == DYNAMIC_CORE).sum()) if sequence_enabled or neutral_discovery or responsibility_training else None,
        "dynamic_candidate_points": int((sequence_state == DYNAMIC_CANDIDATE).sum()) if sequence_enabled or neutral_discovery or responsibility_training else None,
        "static_core_points": int((sequence_state == STATIC_CORE).sum()) if sequence_enabled or neutral_discovery or responsibility_training else None,
        "unknown_points": int((sequence_state == UNKNOWN).sum()) if sequence_enabled or neutral_discovery or responsibility_training else None,
        "runtime": runtime,
    }
    if log_fn is not None:
        namespace = "lagrangian" if lagrangian else "eulerian"
        log_fn(
            "rigid_runtime",
            len(cycle_history),
            {
                f"{namespace}/runtime/{key}": value
                for key, value in runtime.items()
                if value is not None
            },
        )
    contract = {
        "schema_version": 1,
        "motion_representation": representation,
        "observation_support": support_type_name,
        "per_gaussian_observed_speed": lagrangian,
        "per_gaussian_physical_speed": False,
        "spatial_field": "Euclidean Killing field",
        "temporal_degrees_of_freedom_per_interval": 1,
        "reference_point_is_coordinate_gauge": True,
        "physical_axis_parameter": False,
        "physical_center_parameter": False,
        "curl_field": False,
        "motion_optimizer": False,
        "strain_penalty": False,
        "velocity_equality_penalty": False,
        "speed_equality_penalty": False,
        "exact_se3_integration": True,
        "static_transport_exactly_zero": True,
        "dense_flow_role": "validation residual from same PairObservation source",
        "flow_measurement_source_id": measurement_source_id,
        "ground_truth_used_as_solver_input": False,
        "ground_truth_validation": (
            "post_solve_only" if args.rigid_analytic_validation else "disabled"
        ),
        "dino_assignment_prior": False,
        "depth_cores_frozen": freeze_depth_cores,
        "neutral_rigid_discovery": neutral_discovery,
        "rigid_responsibility_training": responsibility_training,
        "persistent_material_variable": (
            "per_canonical_gaussian_p_rigid"
            if responsibility_training
            else None
        ),
        "support_responsibility_is_derived": (
            True if responsibility_training and not lagrangian else None
        ),
        "dino_role": (
            "boundary_and_material_affinity_only"
            if responsibility_training
            else None
        ),
        "motion_evidence_role": (
            "decides_static_vs_rigid_side_identity"
            if responsibility_training
            else None
        ),
        "soft_responsibility_blends_geometry": (
            False if responsibility_training else None
        ),
        "initial_material_probability": (
            0.5 if neutral_discovery or responsibility_training else None
        ),
        "material_labels_used_to_select_discovery_observations": (
            False if neutral_discovery or responsibility_training else None
        ),
        "assignment_updates_during_discovery": False if neutral_discovery else None,
        "postsolve_assignment": neutral_discovery,
        "depth_role": "postsolve_hypothesis_visibility_only" if neutral_discovery else None,
        "moving_vs_static_prefilter_during_discovery": False if neutral_discovery else None,
        "rigid_assisted_reassignment": not freeze_depth_cores and not freeze_prior_labels and not neutral_discovery and not responsibility_training,
        "candidate_promotion": not freeze_depth_cores and not freeze_prior_labels and not neutral_discovery and not responsibility_training,
        "candidate_demotion": not freeze_depth_cores and not freeze_prior_labels and not neutral_discovery and not responsibility_training,
        "outer_em_cycles": not freeze_depth_cores and not freeze_prior_labels and not neutral_discovery and not responsibility_training,
        "speed_correlation_assignment": not freeze_depth_cores and not freeze_prior_labels and not responsibility_training,
        "static_control_observation_support": (
            "independent_per_gaussian_rasterized_flow"
            if freeze_depth_cores and not lagrangian
            else (
                "shared_per_gaussian_rasterized_flow"
                if freeze_depth_cores
                else None
            )
        ),
    }

    if lagrangian:
        speed_observation_payload = {
            "schema_version": 1,
            "layout": "interval_gaussian",
            **{
                key: value.detach().cpu() if torch.is_tensor(value) else value
                for key, value in final_gaussian_speed_observations.items()
            },
            "velocity_gradient_diagnostics": gradient_diagnostics,
        }
        torch.save(
            mode_payload, output_dir / "phase1_lagrangian_rigid_killing_mode.pt"
        )
        torch.save(
            speed_payload, output_dir / "phase1_lagrangian_interval_speeds.pt"
        )
        torch.save(
            observation_payload,
            output_dir / "phase1_lagrangian_observation_audit.pt",
        )
        torch.save(
            speed_observation_payload,
            output_dir / "phase1_lagrangian_gaussian_speed_observations.pt",
        )
        torch.save(
            assignment_payload,
            output_dir / "phase1_lagrangian_assignment_final.pt",
        )
        torch.save(
            final_tracks, output_dir / "phase1_lagrangian_tracks_final.pt"
        )
    else:
        torch.save(mode_payload, output_dir / "phase1_rigid_killing_mode.pt")
        torch.save(speed_payload, output_dir / "phase1_rigid_interval_speeds.pt")
        torch.save(observation_payload, output_dir / "phase1_rigid_observation_audit.pt")
        torch.save(assignment_payload, output_dir / "phase1_rigid_assignment_final.pt")
        torch.save(final_tracks, output_dir / "phase1_rigid_tracks_final.pt")
    if responsibility_training:
        save_responsibility_artifacts()
    if neutral_discovery:
        neutral_payload = {
            "schema_version": 1,
            "initial_dynamic_probability": initial_point_probability.cpu(),
            "state": sequence_state.detach().cpu(),
            "dynamic_probability": current_point_probability.detach().cpu(),
            "effective_dynamic_mask": current_effective_mask.detach().cpu(),
            **{
                key: value.detach().cpu()
                for key, value in final_assignment.items()
                if torch.is_tensor(value)
            },
        }
        torch.save(
            neutral_payload,
            output_dir / "phase1_neutral_postsolve_assignment.pt",
        )
        with (output_dir / "phase1_neutral_discovery_summary.json").open(
            "w", encoding="utf-8"
        ) as stream:
            json.dump(
                _strict_json({
                    "schema_version": 1,
                    "initial_dynamic_probability": 0.5,
                    "foreground_points_used_for_discovery": int(
                        canonical_xyz.shape[0]
                    ),
                    "material_labels_used_for_discovery": False,
                    "assignment_updates_during_discovery": 0,
                    "rigid_fits": len(cycle_history),
                    "postsolve_static_count": int(
                        (sequence_state == STATIC_CORE).sum()
                    ),
                    "postsolve_dynamic_count": int(
                        (sequence_state == DYNAMIC_CORE).sum()
                    ),
                    "postsolve_unknown_count": int(
                        (sequence_state == UNKNOWN).sum()
                    ),
                    "postsolve_margin_median": float(
                        final_assignment["margin"].median()
                    ),
                    "postsolve_speed_correlation_median": float(
                        final_assignment["speed_correlation"].median()
                    ),
                    "rigid_residual_median": cycle_history[0][
                        "rigid_residual_median"
                    ],
                    "speed_mean": final_summary["speed_mean"],
                    "speed_min": final_summary["speed_min"],
                    "speed_max": final_summary["speed_max"],
                    "ground_truth_used_as_algorithm_input": False,
                }),
                stream,
                indent=2,
                allow_nan=False,
            )
    elif sequence_enabled and not freeze_depth_cores:
        hypothesis_keys = (
            "static_energy", "rigid_energy", "margin", "interval_margin",
            "camera_margin", "time_persistence", "static_time_persistence",
            "camera_persistence", "camera_support_count",
            "camera_median_likelihood_ratio", "informative_interval_count",
            "median_likelihood_ratio", "likelihood_ratio_q25",
            "likelihood_ratio_q75", "longest_positive_run",
            "early_dynamic_fraction", "late_dynamic_fraction", "speed_relative_error",
            "speed_sign_consistency", "speed_perpendicular_residual",
            "speed_parallel_residual", "speed_correlation", "static_error", "rigid_error", "weight",
            "excitation",
        )
        hypothesis_payload = {
            "schema_version": 1,
            **{
                key: final_assignment[key].detach().cpu()
                for key in hypothesis_keys
                if key in final_assignment and torch.is_tensor(final_assignment[key])
            },
        }
        history_payload = {
            "schema_version": 1,
            "cycles": cycle_history,
            "assignment_states": assignment_state_history,
            "final_state": sequence_state.detach().cpu(),
            "consecutive_support_cycles": consecutive_support_cycles.detach().cpu(),
        }
        torch.save(
            hypothesis_payload,
            output_dir / (
                "phase1_lagrangian_hypothesis_evidence.pt"
                if lagrangian
                else "phase1_rigid_hypothesis_evidence.pt"
            ),
        )
        torch.save(
            history_payload,
            output_dir / (
                "phase1_lagrangian_assignment_history.pt"
                if lagrangian
                else "phase1_gaussian_assignment_history.pt"
            ),
        )
        torch.save(assignment_payload, output_dir / "phase1_assignment_final.pt")
        from PIL import Image
        diagnostic_features = {
            "rigid_vs_static_margin": torch.sigmoid(final_assignment["margin"]),
            "temporal_persistence": final_assignment["time_persistence"],
            "camera_persistence": final_assignment["camera_persistence"],
            "speed_correlation": 0.5 * (final_assignment["speed_correlation"].clamp(-1, 1) + 1.0),
        }
        diagnostic_camera = frame_group_maps[0][sorted(frame_group_maps[0])[0]]
        for name, feature in diagnostic_features.items():
            image = render_feature_field(
                diagnostic_camera,
                gaussians,
                pipe,
                feature.to(canonical_xyz)[:, None],
                bg_feature=torch.zeros(1, device=canonical_xyz.device),
                detach_geometry=True,
            )["render"][0].detach().cpu().clamp(0, 1)
            Image.fromarray((image * 255).byte().numpy()).save(
                output_dir / f"assignment_{name}.png"
            )
        with (output_dir / "phase1_assignment_summary.json").open(
            "w", encoding="utf-8"
        ) as stream:
            json.dump(
                _strict_json({
                    "dynamic_core_size": final_summary["dynamic_core_points"],
                    "dynamic_candidate_size": final_summary["dynamic_candidate_points"],
                    "static_core_size": final_summary["static_core_points"],
                    "effective_dynamic_size": final_summary["effective_dynamic_points"],
                    "outer_cycles": len(cycle_history),
                    "ground_truth_used_as_algorithm_input": False,
                }),
                stream,
                indent=2,
                allow_nan=False,
            )
    elif sequence_enabled:
        torch.save(
            assignment_payload,
            output_dir / "phase1_depth_frozen_assignment_final.pt",
        )
        with (output_dir / "phase1_depth_frozen_rigid_validation.json").open(
            "w", encoding="utf-8"
        ) as stream:
            json.dump(
                _strict_json({
                    "schema_version": 1,
                    "depth_cores_frozen": True,
                    "single_rigid_fit": len(cycle_history) == 1,
                    "label_updates_enabled": False,
                    "label_changes": sum(
                        int(item["label_changes"]) for item in cycle_history
                    ),
                    "frozen_core_label_invariant": final_summary[
                        "frozen_core_label_invariant"
                    ],
                    "dynamic_core_size": final_summary["dynamic_core_points"],
                    "static_core_size": final_summary["static_core_points"],
                    "unknown_size": (
                        int((sequence_state == UNKNOWN).sum())
                    ),
                    "effective_dynamic_size": final_summary[
                        "effective_dynamic_points"
                    ],
                    "static_control": frozen_static_control_diagnostics,
                    "ground_truth_used_as_algorithm_input": False,
                }),
                stream,
                indent=2,
                allow_nan=False,
            )
    with (output_dir / (
        "phase1_lagrangian_solver_summary.json"
        if lagrangian
        else "phase1_rigid_solver_summary.json"
    )).open(
        "w", encoding="utf-8"
    ) as stream:
        json.dump(_strict_json(final_summary), stream, indent=2, allow_nan=False)
    with (output_dir / (
        "phase1_lagrangian_contract.json"
        if lagrangian
        else "phase1_rigid_contract.json"
    )).open(
        "w", encoding="utf-8"
    ) as stream:
        json.dump(contract, stream, indent=2, allow_nan=False)
    if args.rigid_analytic_validation:
        # Import and load GT only after the measured-flow solve and all primary
        # artifacts are frozen. Nothing below can affect estimation or replay.
        from utils.rigid_killing_validation import (
            load_rigid_dataset_ground_truth,
            validate_frozen_rigid_solution,
        )

        ground_truth = load_rigid_dataset_ground_truth(
            _dynamic_root_for_train(dataset, args),
            canonical_xyz,
            moving_part=int(args.rigid_analytic_moving_part),
            gaussian_part_labels=getattr(gaussians, "_gt_part_labels", None),
        )
        gt_report = validate_frozen_rigid_solution(
            ground_truth,
            current_mode,
            current_speeds,
            [int(group[0]) for group in frame_groups],
            frame_times,
            canonical_xyz,
            positions_history,
            current_effective_mask,
            solver_observation_source_id=measurement_source_id,
            static_tolerance=float(args.rigid_static_displacement_tolerance),
        )
        if responsibility_training:
            gt_moving = ground_truth.moving_mask.detach().cpu().bool()
            predicted_rigid = classification["rigid_mask"].detach().cpu().bool()
            ambiguous = classification["ambiguous_mask"].detach().cpu().bool()
            true_negative = int((~predicted_rigid & ~gt_moving).sum())
            total = max(int(gt_moving.numel()), 1)
            gt_labels = (
                ground_truth.part_labels
                if ground_truth.part_labels is not None
                else gt_moving
            )
            dino_gt = dino_boundary_metrics(gaussian_dino_graph, gt_labels)
            if lagrangian:
                support_boundary = None
            else:
                raw_boundary = dino_boundary_metrics(
                    gaussian_dino_graph, gt_labels, geometric_cell_ids
                )
                initial_boundary = dino_boundary_metrics(
                    gaussian_dino_graph,
                    gt_labels,
                    eulerian_support_state["initial_dino_group_ids"],
                )
                final_boundary = dino_boundary_metrics(
                    gaussian_dino_graph, gt_labels, point_group_ids
                )
                support_boundary = {
                    "raw_geometric_cells": raw_boundary,
                    "after_dino_split": initial_boundary,
                    "after_responsibility_split": final_boundary,
                }
            gt_interval_speeds = torch.as_tensor(
                gt_report["gt_interval_speeds"], dtype=torch.float64
            )
            interval_dt = torch.diff(torch.as_tensor(frame_times, dtype=torch.float64))
            estimated_integrated_angle = float((current_speeds * interval_dt).sum())
            gt_integrated_angle = float((gt_interval_speeds * interval_dt).sum())
            gt_report.update({
                "assignment_true_negative": true_negative,
                "assignment_accuracy": (
                    gt_report["assignment_true_positive"] + true_negative
                )
                / total,
                "assignment_ambiguous_count": int(ambiguous.sum()),
                "assignment_ambiguous_fraction": float(ambiguous.float().mean()),
                "responsibility_entropy": classification["mean_entropy"],
                "estimated_speed_mean": float(current_speeds.mean()),
                "gt_speed_mean": float(gt_interval_speeds.mean()),
                "estimated_integrated_angle": estimated_integrated_angle,
                "gt_integrated_angle": gt_integrated_angle,
                "integrated_angle_absolute_error": abs(
                    estimated_integrated_angle - gt_integrated_angle
                ),
                "dino_part_boundary": dino_gt,
                "gt_boundary_edges_inside_eulerian_support": support_boundary,
                "ground_truth_used_as_algorithm_input": False,
            })
            if log_fn is not None:
                responsibility_gt_metrics = {
                    "responsibility/assignment_accuracy_gt": gt_report[
                        "assignment_accuracy"
                    ],
                    "responsibility/assignment_iou_gt": gt_report[
                        "assignment_iou"
                    ],
                    "responsibility/dino_boundary_precision_gt": dino_gt[
                        "precision"
                    ],
                    "responsibility/dino_boundary_recall_gt": dino_gt[
                        "recall"
                    ],
                }
                log_fn(
                    "gt_validation",
                    len(cycle_history),
                    {
                        key: value
                        for key, value in responsibility_gt_metrics.items()
                        if value is not None
                    },
                )
        if sequence_enabled:
            gt_moving = ground_truth.moving_mask.detach().cpu().bool()
            initial_core = gaussians.get_initial_dynamic_mask.detach().cpu().bool()
            static_core = gaussians.sequence_assignment_state["state"].cpu() == STATIC_CORE
            true_positive = int((initial_core & gt_moving).sum())
            initial_count = int(initial_core.sum())
            gt_count = int(gt_moving.sum())
            static_true = int((static_core & ~gt_moving).sum())
            static_count = int(static_core.sum())
            gt_report.update({
                "initial_dynamic_core_precision": (
                    true_positive / initial_count if initial_count else None
                ),
                "initial_dynamic_core_recall": (
                    true_positive / gt_count if gt_count else None
                ),
                "initial_static_core_precision": (
                    static_true / static_count if static_count else None
                ),
            })
            if log_fn is not None:
                gt_metrics = {}
                if gt_report["initial_dynamic_core_precision"] is not None:
                    gt_metrics["assignment_pretrain/dynamic_core_precision_gt"] = gt_report["initial_dynamic_core_precision"]
                if gt_report["initial_dynamic_core_recall"] is not None:
                    gt_metrics["assignment_pretrain/dynamic_core_recall_gt"] = gt_report["initial_dynamic_core_recall"]
                if gt_metrics:
                    log_fn("gt_validation", len(cycle_history), gt_metrics)
        with (output_dir / (
            "phase1_lagrangian_gt_validation.json"
            if lagrangian
            else "phase1_rigid_gt_validation.json"
        )).open(
            "w", encoding="utf-8"
        ) as stream:
            json.dump(_strict_json(gt_report), stream, indent=2, allow_nan=False)
        print(
            "[Rigid Killing GT validation] "
            f"passed={gt_report['acceptance_passed']} "
            f"direction={gt_report['angular_direction_cosine']} "
            f"speed_max_error={gt_report['interval_speed_error_max']:.6f}"
        )
    html_max_points = int(args.transported_html_max_points)
    html_max_vectors = min(250, html_max_points)
    if responsibility_training:
        write_transported_gaussians_html(
            discovery_tracks,
            output_dir / "phase1_responsibility_all_rigid_hypothesis_motion.html",
            title=(
                "Soft responsibility - common rigid hypothesis applied to all "
                "foreground Gaussians before final polarization"
            ),
            semantic_threshold=float(args.semantic_dynamic_threshold),
            max_points=html_max_points,
            max_vectors=html_max_vectors,
        )
        write_transported_gaussians_html(
            final_tracks,
            output_dir / "phase1_responsibility_final_physical_motion.html",
            title=(
                "Soft responsibility - final physical motion: rigid only; static "
                "and ambiguous remain exactly canonical"
            ),
            semantic_threshold=float(args.semantic_dynamic_threshold),
            max_points=html_max_points,
            max_vectors=html_max_vectors,
        )
    elif neutral_discovery:
        write_transported_gaussians_html(
            discovery_tracks,
            output_dir / "phase1_neutral_discovery_all_gaussians_motion.html",
            title=(
                "Phase 1 neutral discovery - fitted rigid motion applied to all "
                "semantic-foreground Gaussians (before assignment)"
            ),
            semantic_threshold=float(args.semantic_dynamic_threshold),
            max_points=html_max_points,
            max_vectors=html_max_vectors,
        )
        write_transported_gaussians_html(
            final_tracks,
            output_dir / "phase1_neutral_final_assigned_motion.html",
            title=(
                "Phase 1 neutral discovery - final motion after static/dynamic/unknown "
                "assignment"
            ),
            semantic_threshold=float(args.semantic_dynamic_threshold),
            max_points=html_max_points,
            max_vectors=html_max_vectors,
        )
    else:
        write_transported_gaussians_html(
            final_tracks,
            output_dir / (
                "phase1_lagrangian_transported.html"
                if lagrangian
                else "phase1_rigid_killing_1d_transported.html"
            ),
            title=(
                "Phase 1 Lagrangian rigid Killing - exact SE(3) transport"
                if lagrangian
                else "Phase 1 rigid Killing field - exact SE(3) transport"
            ),
            semantic_threshold=float(args.semantic_dynamic_threshold),
            max_points=html_max_points,
            max_vectors=html_max_vectors,
        )
    return {
        "free_tracks": final_tracks,
        "final_tracks": final_tracks,
        "times": frame_times,
    }


def phase1_update_count(cameras, args):
    """Return the exact optimizer-update count for the retained pipeline."""
    if str(args.phase1_motion_representation) in {
        "rigid_killing_1d",
        "rigid_killing_eulerian",
        "rigid_killing_lagrangian",
    }:
        if getattr(args, "kpart_rigid_discovery", False):
            discovery = (
                int(args.kpart_cycles)
                if args.kpart_cycles is not None
                else int(args.kpart_anneal_cycles) + int(args.kpart_plateau_cycles)
            )
            joint_fit = (
                int(args.num_moving_parts)
                * (3 * int(args.joint_compare_iters) + 1)
                if args.fit_joints
                else 0
            )
            trajectory = int(args.joint_trajectory_iters) if args.optimize_joint_trajectory else 0
            if args.optimize_joint_trajectory and args.joint_geometry_refine:
                trajectory += max(3, min(25, int(args.joint_trajectory_iters) // 4))
            return max(1, discovery + joint_fit + trajectory)
        if getattr(args, "enable_nested_killing_part_discovery", False):
            inner = max(1, int(getattr(args, "killing_inner_steps_per_frame", 10)))
            outer = max(1, int(getattr(args, "part_outer_iters", 3)))
            embed = max(1, int(getattr(args, "part_outer_embedding_steps", 10)))
            frame_count = len(_complete_frame_groups(_ordered_dynamic_cameras(cameras), args))
            intervals = max(1, frame_count - 1)
            return max(1, outer * (intervals * inner + embed))
        if getattr(args, "rigid_responsibility_training", False):
            return max(1, int(args.rigid_responsibility_cycles))
        if (
            getattr(args, "phase1_neutral_rigid_discovery", False)
            or getattr(args, "phase1_freeze_depth_cores", False)
        ):
            return 1
        return max(1, int(args.free_field_iters))
    frame_groups = _complete_frame_groups(_ordered_dynamic_cameras(cameras), args)
    windows = _phase1_window_ranges(
        len(frame_groups),
        int(args.phase1_window_frames),
        int(args.phase1_window_stride),
    )
    return len(_phase1_stage_schedule(args)) * len(windows) * int(args.free_field_iters)


def run_eulerian_phase1_pipeline(
    scene,
    gaussians,
    pipe,
    background,
    dataset,
    args,
    opt,
    log_fn=None,
):
    """Train and export the single supported fixed-curl Phase-1 pipeline."""
    correspondence_cache = MotionCorrespondenceCache(dataset, args)
    tracks, times = train_eulerian_velocity_gradient_phase1(
        scene,
        gaussians,
        pipe,
        background,
        dataset,
        args,
        opt,
        visual_loss_fn=_visual_loss,
        log_fn=log_fn,
        correspondence_cache=correspondence_cache,
    )
    return {"free_tracks": tracks, "final_tracks": tracks, "times": times}
