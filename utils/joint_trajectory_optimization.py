"""Stage-3 exact joint-coordinate optimization against RGB/depth/DINO renders."""

from __future__ import annotations

import json
import math
from pathlib import Path

import torch
import torch.nn.functional as F
from tqdm import tqdm

from utils.gaussian_sequence_assignment import hypothesis_visibility
from utils.joint_model_fitting import _render_candidate_view, joint_to_rigid_mode
from utils.motion_correspondence import MotionCorrespondenceCache, _sample_map, project_world_points
from utils.rigid_killing_motion import integrate_rigid_mode_exact, predict_rigid_cell_velocity
from utils.rigid_transform_utils import (
    integrate_joint_transforms,
    quaternion_transport_metrics,
    transport_gaussian_state,
)
from utils.static_flow_calibration import (
    compute_camera_interval_reliability,
    fit_robust_flow_null,
)
from utils.temporal_visibility_pretraining import _atomic_torch_save, inverse_depth_to_metric


def initialize_q_from_coarse_rigid_trajectory(joints):
    if not joints:
        raise ValueError("Stage 3 requires at least one fitted joint")
    return torch.stack([
        torch.as_tensor(joint["coarse_q"]).detach().cpu().double()
        for joint in joints
    ])


def _bspline_level(u, knots, count, degree):
    basis = ((u[:, None] >= knots[:-1]) & (u[:, None] < knots[1:])).to(u.dtype)
    levels = [basis]
    for order in range(1, int(degree) + 1):
        previous = levels[-1]
        columns = knots.numel() - order - 1
        current = torch.zeros((u.numel(), columns), dtype=u.dtype, device=u.device)
        for index in range(columns):
            left_den = knots[index + order] - knots[index]
            right_den = knots[index + order + 1] - knots[index + 1]
            if float(left_den) > 0:
                current[:, index] += (u - knots[index]) / left_den * previous[:, index]
            if float(right_den) > 0:
                current[:, index] += (knots[index + order + 1] - u) / right_den * previous[:, index + 1]
        levels.append(current)
    return levels


def cubic_bspline_basis(frame_times, control_count):
    """Return cubic B, dB/dt, d2B/dt2 at the real acquisition times."""
    times = torch.as_tensor(frame_times).detach().cpu().double()
    if times.numel() < 2 or bool((torch.diff(times) <= 0).any()):
        raise ValueError("frame_times must increase strictly")
    degree = 3
    count = max(degree + 1, int(control_count))
    interior_count = count - degree - 1
    interior = (
        torch.linspace(0.0, 1.0, interior_count + 2, dtype=torch.float64)[1:-1]
        if interior_count > 0
        else torch.empty(0, dtype=torch.float64)
    )
    knots = torch.cat((torch.zeros(degree + 1), interior, torch.ones(degree + 1))).double()
    duration = float(times[-1] - times[0])
    u = (times - times[0]) / duration
    endpoint = u >= 1.0
    levels = _bspline_level(u.clamp_max(1.0 - 1.0e-12), knots, count, degree)
    basis = levels[degree][:, :count]
    basis[endpoint] = 0.0
    basis[endpoint, -1] = 1.0
    lower = levels[degree - 1]
    derivative_u = torch.zeros_like(basis)
    for index in range(count):
        left = knots[index + degree] - knots[index]
        right = knots[index + degree + 1] - knots[index + 1]
        if float(left) > 0:
            derivative_u[:, index] += degree / left * lower[:, index]
        if float(right) > 0:
            derivative_u[:, index] -= degree / right * lower[:, index + 1]
    lower2 = levels[degree - 2]
    first_lower = torch.zeros((u.numel(), count + 1), dtype=torch.float64)
    for index in range(count + 1):
        left = knots[index + degree - 1] - knots[index]
        right = knots[index + degree] - knots[index + 1]
        if float(left) > 0:
            first_lower[:, index] += (degree - 1) / left * lower2[:, index]
        if float(right) > 0:
            first_lower[:, index] -= (degree - 1) / right * lower2[:, index + 1]
    second_u = torch.zeros_like(basis)
    for index in range(count):
        left = knots[index + degree] - knots[index]
        right = knots[index + degree + 1] - knots[index + 1]
        if float(left) > 0:
            second_u[:, index] += degree / left * first_lower[:, index]
        if float(right) > 0:
            second_u[:, index] -= degree / right * first_lower[:, index + 1]
    return basis, derivative_u / duration, second_u / (duration * duration)


def derive_qdot_qddot(coordinates, frame_times):
    q = torch.as_tensor(coordinates).double()
    time = torch.as_tensor(frame_times, device=q.device, dtype=q.dtype)
    if q.ndim != 2 or q.shape[1] != time.numel() or time.numel() < 2:
        raise ValueError("coordinates must be [K,T] and match frame_times")
    if bool((torch.diff(time) <= 0).any()):
        raise ValueError("frame_times must increase strictly")
    velocity = torch.zeros_like(q)
    velocity[:, 0] = (q[:, 1] - q[:, 0]) / (time[1] - time[0])
    velocity[:, -1] = (q[:, -1] - q[:, -2]) / (time[-1] - time[-2])
    if time.numel() > 2:
        velocity[:, 1:-1] = (q[:, 2:] - q[:, :-2]) / (time[2:] - time[:-2])[None]
    acceleration = torch.zeros_like(q)
    acceleration[:, 0] = (velocity[:, 1] - velocity[:, 0]) / (time[1] - time[0])
    acceleration[:, -1] = (velocity[:, -1] - velocity[:, -2]) / (time[-1] - time[-2])
    if time.numel() > 2:
        acceleration[:, 1:-1] = (velocity[:, 2:] - velocity[:, :-2]) / (time[2:] - time[:-2])[None]
    return velocity, acceleration


def derive_qjerk(acceleration, frame_times):
    qddot = torch.as_tensor(acceleration).double()
    time = torch.as_tensor(frame_times, device=qddot.device, dtype=qddot.dtype)
    if qddot.ndim != 2 or qddot.shape[1] != time.numel() or time.numel() < 2:
        raise ValueError("acceleration must be [K,T] and match frame_times")
    jerk = torch.zeros_like(qddot)
    jerk[:, 0] = (qddot[:, 1] - qddot[:, 0]) / (time[1] - time[0])
    jerk[:, -1] = (qddot[:, -1] - qddot[:, -2]) / (time[-1] - time[-2])
    if time.numel() > 2:
        jerk[:, 1:-1] = (qddot[:, 2:] - qddot[:, :-2]) / (time[2:] - time[:-2])[None]
    return jerk


def _trajectory_temporal_terms(q, args):
    time = torch.as_tensor(args._joint_frame_times, device=q.device, dtype=q.dtype)
    if q.shape[1] != time.numel() or time.numel() < 2:
        zero = q.sum() * 0.0
        return {
            "weighted": zero,
            "acceleration": zero,
            "jerk": zero,
            "reverse_velocity": zero,
        }
    dt = torch.diff(time).clamp_min(1.0e-8)
    velocity = torch.diff(q, dim=-1) / dt[None]
    velocity_time = 0.5 * (time[1:] + time[:-1])
    if velocity.shape[1] > 1:
        acceleration = torch.diff(velocity, dim=-1) / torch.diff(velocity_time).clamp_min(1.0e-8)[None]
        acceleration_loss = acceleration.square().mean()
    else:
        acceleration = velocity[:, :0]
        acceleration_loss = q.sum() * 0.0
    if acceleration.shape[1] > 1:
        acceleration_time = 0.5 * (velocity_time[1:] + velocity_time[:-1])
        jerk = torch.diff(acceleration, dim=-1) / torch.diff(acceleration_time).clamp_min(1.0e-8)[None]
        jerk_loss = jerk.square().mean()
    else:
        jerk_loss = q.sum() * 0.0
    direction = torch.sign((q[:, -1] - q[:, 0]).detach())
    direction = torch.where(direction.abs() > 0, direction, torch.ones_like(direction))
    reverse_loss = F.relu(-direction[:, None] * velocity).square().mean()
    accel_weight = float(getattr(args, "joint_trajectory_acceleration_weight", 1.0e-5))
    jerk_weight = float(getattr(args, "joint_trajectory_jerk_weight", 1.0e-8))
    reverse_weight = (
        float(getattr(args, "joint_trajectory_reverse_weight", 1.0e-2))
        if bool(getattr(args, "joint_trajectory_monotonic", True))
        else 0.0
    )
    return {
        "weighted": (
            accel_weight * acceleration_loss
            + jerk_weight * jerk_loss
            + reverse_weight * reverse_loss
        ),
        "acceleration": acceleration_loss,
        "jerk": jerk_loss,
        "reverse_velocity": reverse_loss,
    }


def _append_temporal_log(log, stats):
    for source, target in (
        ("acceleration_loss", "joint_trajectory/acceleration_loss"),
        ("jerk_loss", "joint_trajectory/jerk_loss"),
        ("reverse_velocity_loss", "joint_trajectory/reverse_velocity_loss"),
    ):
        if source in stats:
            log[target] = stats[source]
    return log


def compute_joint_projected_flow(points, joint, q_source, q_target, source_camera, target_camera):
    mode = joint_to_rigid_mode(joint).to(device=points.device, dtype=points.dtype)
    source = integrate_rigid_mode_exact(points, mode, q_source)["positions"]
    target = integrate_rigid_mode_exact(points, mode, q_target)["positions"]
    source_pixel, source_z, valid_source = project_world_points(source, source_camera)
    target_pixel, target_z, valid_target = project_world_points(target, target_camera)
    return {
        "source_positions": source,
        "target_positions": target,
        "source_pixels": source_pixel,
        "target_pixels": target_pixel,
        "source_depth": source_z,
        "target_depth": target_z,
        "projected_flow": target_pixel - source_pixel,
        "valid_projection": valid_source & valid_target,
    }


def _resized_observation(cache, source_camera, target_camera, device):
    observation = cache.get(source_camera, target_camera, device)
    height, width = int(source_camera.image_height), int(source_camera.image_width)
    flow, confidence = observation.flow, observation.confidence
    if tuple(flow.shape[-2:]) != (height, width):
        flow = F.interpolate(flow[None], size=(height, width), mode="bilinear", align_corners=False)[0]
        confidence = F.interpolate(confidence[None], size=(height, width), mode="bilinear", align_corners=False)[0]
    return flow, confidence, height, width


@torch.no_grad()
def estimate_static_flow_calibration(points, labels, camera_maps, cache, args):
    static = torch.as_tensor(points)[torch.as_tensor(labels, device=points.device) == 0]
    if static.shape[0] == 0:
        raise RuntimeError("Stage 3 needs confidently static Gaussians for flow calibration")
    if static.shape[0] > 4096:
        ids = torch.linspace(0, static.shape[0] - 1, 4096, device=static.device).round().long().unique()
        static = static[ids]
    calibration = {}
    for interval in range(len(camera_maps) - 1):
        for camera_key in sorted(camera_maps[interval]):
            source_camera = camera_maps[interval][camera_key]
            target_camera = camera_maps[interval + 1][camera_key]
            flow, confidence, height, width = _resized_observation(cache, source_camera, target_camera, points.device)
            pixels, _, valid = project_world_points(static, source_camera)
            samples = _sample_map(flow, pixels, height, width)
            sample_confidence = _sample_map(confidence, pixels, height, width)[:, 0]
            valid &= sample_confidence >= float(args.correspondence_min_confidence)
            fitted = fit_robust_flow_null(
                samples[valid],
                mad_floor=float(args.static_flow_mad_floor),
                min_samples=64,
            )
            reliability = compute_camera_interval_reliability(
                fitted, reference_scale=1.0, min_samples=64
            )
            calibration[(camera_key, interval)] = {
                "median": fitted["median"].cpu(),
                "mad": fitted["mad"].cpu(),
                "sample_count": int(fitted["sample_count"]),
                "reliable": bool(fitted["reliable"]),
                "camera_reliability": reliability,
            }
    return calibration


def _depth_map(camera, height, width, device, dtype):
    inverse = camera.invdepthmap.to(device=device, dtype=dtype)
    if tuple(inverse.shape[-2:]) != (height, width):
        inverse = F.interpolate(inverse[None], size=(height, width), mode="nearest")[0]
    return inverse_depth_to_metric(inverse)[0]


def _trajectory_flow_objective(
    q,
    joints,
    selected_points,
    camera_maps,
    cache,
    calibration,
    args,
    *,
    return_camera_stats=False,
):
    total = q.sum() * 0.0
    total_weight = q.new_zeros(())
    camera_sum, camera_weight = {}, {}
    for interval in range(q.shape[1] - 1):
        for camera_key in sorted(camera_maps[interval]):
            source_camera = camera_maps[interval][camera_key]
            target_camera = camera_maps[interval + 1][camera_key]
            flow, confidence, height, width = _resized_observation(cache, source_camera, target_camera, q.device)
            calibration_item = calibration[(camera_key, interval)]
            median = torch.as_tensor(calibration_item["median"], device=q.device, dtype=q.dtype)
            mad = torch.as_tensor(calibration_item["mad"], device=q.device, dtype=q.dtype).clamp_min(float(args.static_flow_mad_floor))
            reliability = float(calibration_item["camera_reliability"])
            source_depth_map = _depth_map(source_camera, height, width, q.device, q.dtype)
            target_depth_map = _depth_map(target_camera, height, width, q.device, q.dtype)
            for part_index, (joint, points) in enumerate(zip(joints, selected_points)):
                projected = compute_joint_projected_flow(
                    points, joint, q[part_index, interval], q[part_index, interval + 1],
                    source_camera, target_camera,
                )
                source_pixels = projected["source_pixels"]
                target_pixels = projected["target_pixels"]
                observed = _sample_map(flow, source_pixels.detach(), height, width) - median
                observed_confidence = _sample_map(confidence, source_pixels.detach(), height, width)[:, 0]
                observed_source_depth = _sample_map(source_depth_map[None], source_pixels.detach(), height, width)[:, 0]
                observed_target_depth = _sample_map(target_depth_map[None], target_pixels.detach(), height, width)[:, 0]
                source_vis = hypothesis_visibility(
                    projected["source_depth"], observed_source_depth,
                    projected["valid_projection"],
                    relative_sigma=float(args.visibility_depth_sigma),
                    margin_sigma=float(args.visibility_occlusion_margin),
                )
                target_vis = hypothesis_visibility(
                    projected["target_depth"], observed_target_depth,
                    projected["valid_projection"],
                    relative_sigma=float(args.visibility_depth_sigma),
                    margin_sigma=float(args.visibility_occlusion_margin),
                )
                valid = source_vis["visible"] & target_vis["visible"] & (observed_confidence >= float(args.correspondence_min_confidence))
                weight = valid.to(q.dtype) * observed_confidence.clamp(0, 1) * reliability
                residual = (projected["projected_flow"] - observed) / mad
                robust = F.smooth_l1_loss(residual, torch.zeros_like(residual), reduction="none", beta=1.0).sum(dim=-1)
                total = total + (weight * robust).sum()
                total_weight = total_weight + weight.sum()
                if return_camera_stats:
                    key = str(camera_key)
                    camera_sum[key] = camera_sum.get(key, q.new_zeros(())) + (weight * robust).sum()
                    camera_weight[key] = camera_weight.get(key, q.new_zeros(())) + weight.sum()
    flow_loss = total / total_weight.clamp_min(1.0)
    temporal = _trajectory_temporal_terms(q, args)
    objective = flow_loss + temporal["weighted"]
    stats = {
        "flow_loss": flow_loss,
        "temporal_loss": temporal["weighted"],
        "acceleration_loss": temporal["acceleration"],
        "jerk_loss": temporal["jerk"],
        "reverse_velocity_loss": temporal["reverse_velocity"],
        "valid_weight": total_weight,
        "camera_flow_loss": {
            key: camera_sum[key] / camera_weight[key].clamp_min(1.0)
            for key in camera_sum
        },
    }
    return objective, stats


def _trajectory_render_objective(
    q,
    joints,
    points,
    rotations,
    labels,
    pairs,
    gaussians,
    pipe,
    background,
    dataset,
    args,
    opt,
    *,
    return_camera_stats=False,
):
    trajectories = [
        integrate_joint_transforms(
            joint["joint_type"],
            torch.as_tensor(joint["axis"], device=q.device, dtype=q.dtype),
            torch.diff(q[index]),
            pivot=None if joint.get("pivot") is None else torch.as_tensor(joint["pivot"], device=q.device, dtype=q.dtype),
            pitch=joint.get("pitch", 0.0),
        )
        for index, joint in enumerate(joints)
    ]
    states = {}
    for frame in sorted({int(item[0]) for item in pairs}):
        xyz, orientation = points, F.normalize(rotations, dim=-1)
        for part, trajectory in enumerate(trajectories, start=1):
            moved, rotated = transport_gaussian_state(
                points,
                rotations,
                trajectory["rotations"][frame],
                trajectory["translations"][frame],
                trajectory["rotation_quaternions"][frame],
            )
            mask = labels == part
            xyz = torch.where(mask[:, None], moved, xyz)
            orientation = torch.where(mask[:, None], rotated, orientation)
        states[frame] = (xyz, F.normalize(orientation, dim=-1))
    sums = {key: q.new_zeros(()) for key in ("total", "rgb", "depth", "dino")}
    camera_sum, camera_count = {}, {}
    for frame, camera_key, camera in pairs:
        losses = _render_candidate_view(
            camera, gaussians, pipe, background, dataset, args, opt,
            *states[int(frame)],
        )
        for key in sums:
            sums[key] = sums[key] + losses[key]
        if return_camera_stats:
            key = str(camera_key)
            camera_sum[key] = camera_sum.get(key, q.new_zeros(())) + losses["total"]
            camera_count[key] = camera_count.get(key, 0) + 1
    count = max(1, len(pairs))
    means = {key: value / count for key, value in sums.items()}
    temporal = _trajectory_temporal_terms(q, args)
    objective = means["total"] + temporal["weighted"]
    stats = {
        "render_loss": means["total"],
        "rgb_loss": means["rgb"],
        "depth_loss": means["depth"],
        "dino_loss": means["dino"],
        "temporal_loss": temporal["weighted"],
        "acceleration_loss": temporal["acceleration"],
        "jerk_loss": temporal["jerk"],
        "reverse_velocity_loss": temporal["reverse_velocity"],
        "valid_weight": q.new_tensor(float(len(pairs))),
        "camera_render_loss": {
            key: value / camera_count[key] for key, value in camera_sum.items()
        },
    }
    return objective, stats


def _inverse_softplus(value):
    value = torch.as_tensor(value).clamp_min(1.0e-9)
    return torch.where(value > 20.0, value, torch.log(torch.expm1(value).clamp_min(1.0e-12)))


def _monotonic_increment_initialization(initial_q):
    initial = torch.as_tensor(initial_q)
    if initial.ndim != 2 or initial.shape[1] < 2:
        raise ValueError("monotonic Stage-3 trajectory needs coordinates shaped [K,T>=2]")
    profile = initial - initial[:, :1]
    final = profile[:, -1]
    direction = torch.sign(final)
    direction = torch.where(direction.abs() > 0, direction, torch.ones_like(direction))
    signed = torch.cummax((profile * direction[:, None]).clamp_min(0.0), dim=-1).values
    increments = torch.diff(signed, dim=-1).clamp_min(1.0e-6)
    return direction, increments


def optimize_monotonic_q(initial_q, objective, frame_times, iterations, learning_rate, log_fn=None):
    """Optimize joint displacement with non-negative signed interval speeds."""
    initial = torch.as_tensor(initial_q)
    direction, increments = _monotonic_increment_initialization(initial)
    raw_increments = torch.nn.Parameter(_inverse_softplus(increments))
    optimizer = torch.optim.Adam([raw_increments], lr=float(learning_rate))
    history = []
    time = torch.as_tensor(frame_times, device=initial.device, dtype=initial.dtype)
    dt = torch.diff(time).clamp_min(1.0e-8)
    progress = tqdm(range(int(iterations)), desc="Stage 3 monotonic trajectory", unit="iter")
    for iteration in progress:
        optimizer.zero_grad(set_to_none=True)
        positive_increments = F.softplus(raw_increments)
        q = torch.cat(
            (
                torch.zeros((positive_increments.shape[0], 1), device=initial.device, dtype=initial.dtype),
                torch.cumsum(positive_increments, dim=-1) * direction[:, None],
            ),
            dim=-1,
        )
        loss, stats = objective(q)
        if not bool(torch.isfinite(loss)):
            raise RuntimeError("Stage 3 produced a non-finite monotonic trajectory loss")
        loss.backward()
        optimizer.step()
        signed_velocity = positive_increments.detach() / dt[None]
        row = {
            "iteration": iteration + 1,
            "loss": float(loss.detach()),
            "render_loss": float(stats.get("render_loss", stats.get("flow_loss")).detach()),
            "rgb_loss": float(stats.get("rgb_loss", loss.new_zeros(())).detach()),
            "depth_loss": float(stats.get("depth_loss", loss.new_zeros(())).detach()),
            "dino_loss": float(stats.get("dino_loss", loss.new_zeros(())).detach()),
            "temporal_loss": float(stats["temporal_loss"].detach()),
            "acceleration_loss": float(stats.get("acceleration_loss", loss.new_zeros(())).detach()),
            "jerk_loss": float(stats.get("jerk_loss", loss.new_zeros(())).detach()),
            "reverse_velocity_loss": float(stats.get("reverse_velocity_loss", loss.new_zeros(())).detach()),
            "valid_weight": float(stats["valid_weight"].detach()),
            "min_signed_speed": float(signed_velocity.min()),
            "max_signed_speed": float(signed_velocity.max()),
        }
        history.append(row)
        progress.set_postfix(
            render=f"{row['render_loss']:.4f}",
            jerk=f"{row['jerk_loss']:.2g}",
            vmin=f"{row['min_signed_speed']:.3g}",
        )
        if log_fn is not None:
            log_fn("joint_trajectory", iteration + 1, _append_temporal_log({
                "joint_trajectory/loss": row["loss"],
                "joint_trajectory/render_loss": row["render_loss"],
                "joint_trajectory/rgb_loss": row["rgb_loss"],
                "joint_trajectory/depth_loss": row["depth_loss"],
                "joint_trajectory/dino_loss": row["dino_loss"],
                "joint_trajectory/temporal_loss": row["temporal_loss"],
                "joint_trajectory/valid_weight": row["valid_weight"],
                "joint_trajectory/min_signed_speed": row["min_signed_speed"],
                "joint_trajectory/max_signed_speed": row["max_signed_speed"],
            }, row))
    with torch.no_grad():
        positive_increments = F.softplus(raw_increments.detach())
        q = torch.cat(
            (
                torch.zeros((positive_increments.shape[0], 1), device=initial.device, dtype=initial.dtype),
                torch.cumsum(positive_increments, dim=-1) * direction[:, None],
            ),
            dim=-1,
        )
    payload = {
        "direction": direction.detach().cpu().double(),
        "increments": positive_increments.detach().cpu().double(),
        "initial_positive_increments": increments.detach().cpu().double(),
    }
    qdot, qddot = derive_qdot_qddot(q.detach().cpu(), frame_times)
    return q, qdot, qddot, payload, history


def optimize_framewise_q(initial_q, objective, iterations, learning_rate, log_fn=None):
    initial = torch.as_tensor(initial_q)
    tail = torch.nn.Parameter(initial[:, 1:].clone())
    optimizer = torch.optim.Adam([tail], lr=float(learning_rate))
    history = []
    progress = tqdm(range(int(iterations)), desc="Stage 3 joint trajectory", unit="iter")
    for iteration in progress:
        optimizer.zero_grad(set_to_none=True)
        q = torch.cat((torch.zeros((tail.shape[0], 1), device=tail.device, dtype=tail.dtype), tail), dim=-1)
        loss, stats = objective(q)
        if not bool(torch.isfinite(loss)):
            raise RuntimeError("Stage 3 produced a non-finite trajectory loss")
        loss.backward()
        optimizer.step()
        row = {
            "iteration": iteration + 1,
            "loss": float(loss.detach()),
            "render_loss": float(stats.get("render_loss", stats.get("flow_loss")).detach()),
            "rgb_loss": float(stats.get("rgb_loss", loss.new_zeros(())).detach()),
            "depth_loss": float(stats.get("depth_loss", loss.new_zeros(())).detach()),
            "dino_loss": float(stats.get("dino_loss", loss.new_zeros(())).detach()),
            "temporal_loss": float(stats["temporal_loss"].detach()),
            "acceleration_loss": float(stats.get("acceleration_loss", loss.new_zeros(())).detach()),
            "jerk_loss": float(stats.get("jerk_loss", loss.new_zeros(())).detach()),
            "reverse_velocity_loss": float(stats.get("reverse_velocity_loss", loss.new_zeros(())).detach()),
            "valid_weight": float(stats["valid_weight"].detach()),
        }
        history.append(row)
        progress.set_postfix(render=f"{row['render_loss']:.4f}")
        if log_fn is not None:
            log_fn("joint_trajectory", iteration + 1, _append_temporal_log({
                "joint_trajectory/loss": row["loss"],
                "joint_trajectory/render_loss": row["render_loss"],
                "joint_trajectory/rgb_loss": row["rgb_loss"],
                "joint_trajectory/depth_loss": row["depth_loss"],
                "joint_trajectory/dino_loss": row["dino_loss"],
                "joint_trajectory/temporal_loss": row["temporal_loss"],
                "joint_trajectory/valid_weight": row["valid_weight"],
            }, row))
    return torch.cat((torch.zeros((tail.shape[0], 1), device=tail.device, dtype=tail.dtype), tail.detach()), dim=-1), history


def optimize_profile_scale_q(initial_q, objective, iterations, learning_rate, log_fn=None):
    """Solve one scale per joint on the Stage-1 coordinate profile.

    Stage 1 recovers the shape of ``q(t)`` to within about one percent but
    under-estimates its amplitude, because zero-velocity Gaussians wrongly
    admitted to a moving part dilute the scalar-speed projection.  Releasing one
    coordinate per frame therefore discards the part that is already correct in
    order to recover the part that is not.  This parameterisation keeps the
    profile and solves the one number that is actually unknown:

        q(t) = scale * q1(t)

    A scale is the only affine reparameterisation the ``q[:, 0] = 0`` gauge
    admits, since ``q1(0) = 0`` already: a constant offset would break the
    gauge, and any offset that rides the profile is proportional to the scale
    and so adds no freedom.  A genuine second degree of freedom would have to
    reshape time -- a warp on ``q1`` rather than a term beside it -- which is a
    different model and is not what Stage 3 is short of.  With a two-frame
    selection this reduces to a single scalar: the total joint displacement.
    """
    initial = torch.as_tensor(initial_q)
    profile = initial - initial[:, :1]
    scale = torch.nn.Parameter(torch.ones((profile.shape[0], 1), device=profile.device, dtype=profile.dtype))
    optimizer = torch.optim.Adam([scale], lr=float(learning_rate))
    history = []
    progress = tqdm(range(int(iterations)), desc="Stage 3 profile scale", unit="iter")
    for iteration in progress:
        optimizer.zero_grad(set_to_none=True)
        q = scale * profile
        loss, stats = objective(q)
        if not bool(torch.isfinite(loss)):
            raise RuntimeError("Stage 3 produced a non-finite profile-scale loss")
        loss.backward()
        optimizer.step()
        row = {
            "iteration": iteration + 1,
            "loss": float(loss.detach()),
            "render_loss": float(stats.get("render_loss", stats.get("flow_loss")).detach()),
            "rgb_loss": float(stats.get("rgb_loss", loss.new_zeros(())).detach()),
            "depth_loss": float(stats.get("depth_loss", loss.new_zeros(())).detach()),
            "dino_loss": float(stats.get("dino_loss", loss.new_zeros(())).detach()),
            "temporal_loss": float(stats["temporal_loss"].detach()),
            "acceleration_loss": float(stats.get("acceleration_loss", loss.new_zeros(())).detach()),
            "jerk_loss": float(stats.get("jerk_loss", loss.new_zeros(())).detach()),
            "reverse_velocity_loss": float(stats.get("reverse_velocity_loss", loss.new_zeros(())).detach()),
            "valid_weight": float(stats["valid_weight"].detach()),
            "profile_scale": [float(value) for value in scale.detach().reshape(-1)],
        }
        history.append(row)
        progress.set_postfix(render=f"{row['render_loss']:.4f}", scale=f"{row['profile_scale'][0]:.3f}")
        if log_fn is not None:
            log_fn("joint_trajectory", iteration + 1, _append_temporal_log({
                "joint_trajectory/loss": row["loss"],
                "joint_trajectory/render_loss": row["render_loss"],
                "joint_trajectory/rgb_loss": row["rgb_loss"],
                "joint_trajectory/depth_loss": row["depth_loss"],
                "joint_trajectory/dino_loss": row["dino_loss"],
                "joint_trajectory/temporal_loss": row["temporal_loss"],
                "joint_trajectory/valid_weight": row["valid_weight"],
                "joint_trajectory/profile_scale": row["profile_scale"][0],
            }, row))
    with torch.no_grad():
        q = scale.detach() * profile
    payload = {
        "profile_scale": scale.detach().cpu().double(),
        "profile": profile.detach().cpu().double(),
    }
    return q, payload, history


def optimize_spline_q(initial_q, frame_times, objective, iterations, learning_rate, log_fn=None):
    count = max(4, min(initial_q.shape[1], math.ceil(initial_q.shape[1] / 4) + 3))
    basis, derivative, second = cubic_bspline_basis(frame_times, count)
    basis = basis.to(initial_q)
    control_initial = torch.linalg.lstsq(basis, initial_q.T).solution.T
    control = torch.nn.Parameter(control_initial)
    optimizer = torch.optim.Adam([control], lr=float(learning_rate))
    history = []
    progress = tqdm(range(int(iterations)), desc="Stage 3 spline trajectory", unit="iter")
    for iteration in progress:
        optimizer.zero_grad(set_to_none=True)
        raw = control @ basis.T
        q = raw - raw[:, :1]
        loss, stats = objective(q)
        if not bool(torch.isfinite(loss)):
            raise RuntimeError("Stage 3 produced a non-finite spline loss")
        loss.backward()
        optimizer.step()
        row = {
            "iteration": iteration + 1,
            "loss": float(loss.detach()),
            "render_loss": float(stats.get("render_loss", stats.get("flow_loss")).detach()),
            "rgb_loss": float(stats.get("rgb_loss", loss.new_zeros(())).detach()),
            "depth_loss": float(stats.get("depth_loss", loss.new_zeros(())).detach()),
            "dino_loss": float(stats.get("dino_loss", loss.new_zeros(())).detach()),
            "temporal_loss": float(stats["temporal_loss"].detach()),
            "acceleration_loss": float(stats.get("acceleration_loss", loss.new_zeros(())).detach()),
            "jerk_loss": float(stats.get("jerk_loss", loss.new_zeros(())).detach()),
            "reverse_velocity_loss": float(stats.get("reverse_velocity_loss", loss.new_zeros(())).detach()),
            "valid_weight": float(stats["valid_weight"].detach()),
        }
        history.append(row)
        progress.set_postfix(render=f"{row['render_loss']:.4f}")
        if log_fn is not None:
            log_fn("joint_trajectory", iteration + 1, _append_temporal_log({
                "joint_trajectory/loss": row["loss"],
                "joint_trajectory/render_loss": row["render_loss"],
                "joint_trajectory/rgb_loss": row["rgb_loss"],
                "joint_trajectory/depth_loss": row["depth_loss"],
                "joint_trajectory/dino_loss": row["dino_loss"],
                "joint_trajectory/temporal_loss": row["temporal_loss"],
                "joint_trajectory/valid_weight": row["valid_weight"],
            }, row))
    raw = control.detach() @ basis.T
    q = raw - raw[:, :1]
    qdot = control.detach().cpu().double() @ derivative.T
    qddot = control.detach().cpu().double() @ second.T
    return q, qdot, qddot, control.detach(), history


@torch.no_grad()
def render_joint_constrained_motion(points, rotations, labels, joints, q, qdot, frame_times):
    xyz0 = torch.as_tensor(points)
    rotation0 = torch.as_tensor(rotations)
    label = torch.as_tensor(labels, device=xyz0.device).long()
    coordinates = torch.as_tensor(q, device=xyz0.device, dtype=xyz0.dtype)
    velocity_coordinates = torch.as_tensor(qdot, device=xyz0.device, dtype=xyz0.dtype)
    positions, orientations, velocities = [], [], []
    transforms = []
    part_rotation_quaternions, part_translations = [], []
    for frame in range(coordinates.shape[1]):
        xyz, orientation, velocity = xyz0.clone(), rotation0.clone(), torch.zeros_like(xyz0)
        frame_transforms = []
        frame_part_quaternions, frame_part_translations = [], []
        for part_index, joint in enumerate(joints, start=1):
            mode = joint_to_rigid_mode(joint).to(device=xyz0.device, dtype=xyz0.dtype)
            result = integrate_rigid_mode_exact(
                xyz0, mode, coordinates[part_index - 1, frame], orientations=rotation0
            )
            mask = label == part_index
            xyz[mask] = result["positions"][mask]
            orientation[mask] = result["orientations"][mask]
            exact_velocity = predict_rigid_cell_velocity(
                mode, result["positions"], velocity_coordinates[part_index - 1, frame]
            )
            velocity[mask] = exact_velocity[mask]
            reference = mode.reference_point.to(xyz0)
            transform_translation = reference - result["rotation_matrix"] @ reference + result["translation"]
            frame_transforms.append({
                "rotation": result["rotation_matrix"].cpu(),
                "translation": transform_translation.cpu(),
            })
            frame_part_quaternions.append(result["rotation_quaternion"].cpu())
            frame_part_translations.append(transform_translation.cpu())
        positions.append(xyz.cpu())
        orientations.append(orientation.cpu())
        velocities.append(velocity.cpu())
        transforms.append(frame_transforms)
        part_rotation_quaternions.append(torch.stack(frame_part_quaternions))
        part_translations.append(torch.stack(frame_part_translations))
    return {
        "schema_version": 1,
        "times": torch.as_tensor(frame_times, dtype=torch.float32),
        "positions": torch.stack(positions),
        "rotations": torch.stack(orientations),
        "velocities": torch.stack(velocities),
        "canonical_xyz": xyz0.detach().cpu(),
        "part_labels": label.cpu(),
        "transforms": transforms,
        "part_rotation_quaternions": torch.stack(part_rotation_quaternions),
        "part_translations": torch.stack(part_translations),
    }


def refine_joint_geometry_small(
    coordinates,
    joints,
    selected_points,
    camera_maps,
    cache,
    calibration,
    args,
    *,
    iterations,
    learning_rate,
    log_fn=None,
):
    """Tightly refine axis/pivot/pitch while absolute q and labels stay frozen."""
    q = torch.as_tensor(coordinates).detach()
    parameters, parameter_state = [], []
    for joint in joints:
        axis0 = torch.as_tensor(joint["axis"], device=q.device, dtype=q.dtype).detach()
        axis = torch.nn.Parameter(axis0.clone())
        # Prismatic motion ignores the pivot, so refining it would optimise a
        # parameter no objective depends on.
        pivot0 = (
            None
            if joint.get("pivot") is None or joint["joint_type"] == "prismatic"
            else torch.as_tensor(joint["pivot"], device=q.device, dtype=q.dtype).detach()
        )
        pivot = None if pivot0 is None else torch.nn.Parameter(pivot0.clone())
        pitch0 = torch.as_tensor(float(joint.get("pitch", 0.0)), device=q.device, dtype=q.dtype)
        pitch = torch.nn.Parameter(pitch0.clone()) if joint["joint_type"] == "screw" else None
        parameters.append(axis)
        if pivot is not None:
            parameters.append(pivot)
        if pitch is not None:
            parameters.append(pitch)
        parameter_state.append((axis, axis0, pivot, pivot0, pitch, pitch0))
    optimizer = torch.optim.Adam(parameters, lr=float(learning_rate))
    history = []
    for iteration in tqdm(range(int(iterations)), desc="Stage 3 joint geometry", unit="iter"):
        optimizer.zero_grad(set_to_none=True)
        current = []
        regularizer = q.new_zeros(())
        for joint, state in zip(joints, parameter_state):
            axis, axis0, pivot, pivot0, pitch, pitch0 = state
            normalized_axis = axis / torch.linalg.norm(axis).clamp_min(1.0e-8)
            normalized_initial = axis0 / torch.linalg.norm(axis0).clamp_min(1.0e-8)
            current_joint = {**joint, "axis": normalized_axis}
            regularizer = regularizer + 10.0 * (normalized_axis - normalized_initial).square().sum()
            if pivot is not None:
                current_joint["pivot"] = pivot
                extent = torch.linalg.norm(selected_points[len(current)].amax(dim=0) - selected_points[len(current)].amin(dim=0)).clamp_min(1.0e-6)
                regularizer = regularizer + 10.0 * ((pivot - pivot0) / extent).square().sum()
            if pitch is not None:
                current_joint["pitch"] = pitch
                regularizer = regularizer + 10.0 * (pitch - pitch0).square()
            current.append(current_joint)
        data_loss, stats = _trajectory_flow_objective(
            q, current, selected_points, camera_maps, cache, calibration, args
        )
        loss = data_loss + regularizer
        if not bool(torch.isfinite(loss)):
            raise RuntimeError("Stage 3 produced a non-finite joint-geometry loss")
        loss.backward()
        optimizer.step()
        row = {
            "iteration": iteration + 1,
            "loss": float(loss.detach()),
            "flow_loss": float(stats["flow_loss"].detach()),
            "regularizer": float(regularizer.detach()),
        }
        history.append(row)
        if log_fn is not None:
            log_fn("joint_trajectory", iteration + 1, {
                "joint_trajectory/geometry_loss": row["loss"],
                "joint_trajectory/geometry_flow_loss": row["flow_loss"],
                "joint_trajectory/geometry_regularizer": row["regularizer"],
            })
    refined = []
    for joint, state in zip(joints, parameter_state):
        axis, _, pivot, _, pitch, _ = state
        result = {
            **joint,
            "axis": (axis.detach() / torch.linalg.norm(axis.detach()).clamp_min(1.0e-8)).cpu(),
            "pivot": None if pivot is None else pivot.detach().cpu(),
            "pitch": float(joint.get("pitch", 0.0)) if pitch is None else float(pitch.detach()),
        }
        refined.append(result)
    return refined, history


def write_joint_trajectory_html(
    coordinates,
    velocity,
    frame_times,
    output,
    *,
    gt_coordinates=None,
    gt_velocity=None,
):
    q = torch.as_tensor(coordinates).detach().cpu().double()
    qdot = torch.as_tensor(velocity).detach().cpu().double()
    times = torch.as_tensor(frame_times).detach().cpu().double().tolist()
    traces = []
    colors = ("#e41a1c", "#4daf4a", "#984ea3", "#ff7f00")
    for part in range(q.shape[0]):
        color = colors[part % len(colors)]
        traces.extend((
            {"type": "scatter", "mode": "lines+markers", "name": f"part {part + 1} q", "x": times, "y": q[part].tolist(), "xaxis": "x", "yaxis": "y", "line": {"color": color}},
            {"type": "scatter", "mode": "lines+markers", "name": f"part {part + 1} qdot", "x": times, "y": qdot[part].tolist(), "xaxis": "x2", "yaxis": "y2", "line": {"color": color}},
        ))
    if gt_coordinates is not None:
        gt_q = torch.as_tensor(gt_coordinates).detach().cpu().double().reshape(-1)
        traces.append({"type": "scatter", "mode": "lines", "name": "GT q (evaluation only)", "x": times, "y": gt_q.tolist(), "xaxis": "x", "yaxis": "y", "line": {"color": "#111111", "dash": "dash"}})
    if gt_velocity is not None:
        gt_v = torch.as_tensor(gt_velocity).detach().cpu().double().reshape(-1)
        traces.append({"type": "scatter", "mode": "lines", "name": "GT qdot (evaluation only)", "x": times, "y": gt_v.tolist(), "xaxis": "x2", "yaxis": "y2", "line": {"color": "#111111", "dash": "dash"}})
    layout = {
        "title": "Stage 3 absolute joint trajectory (GT is post-solve evaluation only)",
        "grid": {"rows": 2, "columns": 1, "pattern": "independent"},
        "xaxis": {"title": "time [s]"}, "yaxis": {"title": "q"},
        "xaxis2": {"title": "time [s]"}, "yaxis2": {"title": "qdot [1/s]"},
        "margin": {"l": 70, "r": 20, "t": 60, "b": 55},
    }
    document = f"""<!doctype html><html><head><meta charset='utf-8'><script src='https://cdn.plot.ly/plotly-2.35.2.min.js'></script></head><body><div id='plot' style='width:100vw;height:100vh'></div><script>Plotly.newPlot('plot',{json.dumps(traces)},{json.dumps(layout)},{{responsive:true}});</script></body></html>"""
    Path(output).write_text(document, encoding="utf-8")


def optimize_joint_trajectory(
    scene,
    gaussians,
    pipe,
    background,
    dataset,
    opt,
    args,
    stage1,
    stage2,
    *,
    frame_selection=None,
    log_fn=None,
):
    from utils.unified_dynamic_training import (
        _camera_map,
        _complete_frame_groups,
        _fixed_camera_metadata,
        _ordered_dynamic_cameras,
        resolve_joint_frame_selection,
    )
    from utils.visualize_transported_gaussians_html import write_transported_gaussians_html

    cameras = _ordered_dynamic_cameras(scene.getTrainCameras())
    all_groups = _complete_frame_groups(cameras, args)
    if frame_selection is None:
        frame_selection = resolve_joint_frame_selection(all_groups, args)
    groups = frame_selection["groups"]
    frame_times = frame_selection["frame_times"]
    camera_maps = [_camera_map(item[2]) for item in groups]
    args._joint_frame_times = frame_times
    if frame_selection["is_sparse"]:
        print(
            f"[Stage 3] frame subset '{frame_selection['spec']}': "
            f"{frame_selection['selected_frames']}/{frame_selection['available_frames']} frames, "
            f"max gap {frame_selection['max_gap_seconds']:.3f}s"
        )
    points = gaussians.get_canonical_xyz.detach()
    rotations = gaussians.get_rotation.detach().clone()
    labels = torch.as_tensor(stage1["labels"], device=points.device).long()
    joints = stage2["joints"]
    camera_keys = sorted(set(camera_maps[0]).intersection(*[set(item) for item in camera_maps[1:]]))
    render_pairs = [
        (frame, key, camera_maps[frame][key])
        for frame in range(1, len(camera_maps))
        for key in camera_keys
    ]
    if not render_pairs:
        raise RuntimeError("Stage 3 requires at least one complete target-frame camera")
    cache = MotionCorrespondenceCache(dataset, args)
    output = Path(scene.model_path) / "dynamic_debug"
    # Optical flow is only meaningful between adjacent frames.  A sparse frame
    # selection turns each interval into a multi-second baseline that dense flow
    # cannot track, so the flow-based calibration, the geometry refinement that
    # consumes it, and the final flow diagnostic are all skipped there.  The
    # calibration is also skipped when nothing needs it: it was previously
    # computed on every run and consumed only under --joint_geometry_refine,
    # while being able to abort the whole stage for an all-dynamic assignment.
    flow_usable = not bool(frame_selection["is_sparse"])
    geometry_refine = bool(args.joint_geometry_refine) and flow_usable
    if bool(args.joint_geometry_refine) and not flow_usable:
        print(
            "[Stage 3] --joint_geometry_refine needs adjacent-frame flow; skipped "
            f"for the sparse '{frame_selection['spec']}' selection"
        )
    calibration_state = {"value": None, "skipped": None}

    def flow_calibration(*, required):
        """Fit the static-flow null once, on demand.

        ``required`` distinguishes the geometry refinement, which cannot run
        without it, from the closing flow diagnostic, which is informational and
        must never abort the stage.  An all-dynamic assignment has no confidently
        static Gaussians, and that used to abort Stage 3 through a calibration
        whose only consumer was a refinement that was switched off.
        """
        if calibration_state["value"] is not None or calibration_state["skipped"]:
            return calibration_state["value"]
        if not flow_usable:
            calibration_state["skipped"] = "sparse_frame_selection"
            return None
        try:
            calibration_state["value"] = estimate_static_flow_calibration(
                points, labels, camera_maps, cache, args
            )
        except RuntimeError as error:
            if required:
                raise
            calibration_state["skipped"] = str(error)
            return None
        _atomic_torch_save(
            {"schema_version": 1, "calibration": calibration_state["value"], "ground_truth_used_as_algorithm_input": False},
            output / "phase3_static_flow_calibration.pt",
        )
        return calibration_state["value"]

    selected_points = []
    for part in range(1, len(joints) + 1):
        part_points = points[labels == part]
        if part_points.shape[0] == 0:
            raise RuntimeError(f"Stage 3 part {part} has no confident Gaussians")
        if part_points.shape[0] > 2048:
            ids = torch.linspace(0, part_points.shape[0] - 1, 2048, device=points.device).round().long().unique()
            part_points = part_points[ids]
        selected_points.append(part_points)
    initial_q = initialize_q_from_coarse_rigid_trajectory(joints).to(device=points.device, dtype=points.dtype)

    render_step = {"value": 0}
    render_generator = torch.Generator().manual_seed(20_260_825)
    render_order = torch.randperm(len(render_pairs), generator=render_generator).tolist()

    def objective(q):
        batch_size = min(len(render_pairs), int(args.joint_trajectory_batch_size))
        start = (render_step["value"] * batch_size) % len(render_order)
        batch = [
            render_pairs[render_order[(start + index) % len(render_order)]]
            for index in range(batch_size)
        ]
        render_step["value"] += 1
        return _trajectory_render_objective(
            q, joints, points, rotations, labels, batch,
            gaussians, pipe, background, dataset, args, opt,
        )

    learning_rate = max(float(args.motion_lr), 1.0e-2)
    total_iterations = int(args.joint_trajectory_iters)
    first_iterations = max(1, total_iterations // 2) if geometry_refine else total_iterations

    # A cubic B-spline needs at least four controls, so it cannot describe a
    # short frame selection; fall back to the profile-scale solve rather than
    # silently fitting more controls than there are frames.
    requested_mode = str(args.joint_trajectory_mode)
    trajectory_mode = requested_mode
    mode_fallback = None
    if requested_mode == "spline" and len(frame_times) < 6:
        trajectory_mode, mode_fallback = "profile_scale", "spline_needs_at_least_six_frames"
    if requested_mode == "per_frame" and len(frame_times) < 3:
        trajectory_mode, mode_fallback = "profile_scale", "per_frame_needs_at_least_three_frames"
    if mode_fallback is not None:
        print(f"[Stage 3] trajectory mode '{requested_mode}' -> 'profile_scale' ({mode_fallback})")

    def offset_log(offset):
        if log_fn is None:
            return None
        return lambda stage, local, stats: log_fn(stage, int(offset) + int(local), stats)

    def optimize_coordinates(start, iterations, log_offset):
        if trajectory_mode == "monotonic":
            result_q, result_qdot, result_qddot, result_payload, result_history = optimize_monotonic_q(
                start, objective, frame_times, iterations, learning_rate, offset_log(log_offset),
            )
            return result_q.detach(), result_qdot, result_qddot, result_payload, result_history
        if trajectory_mode == "spline":
            result_q, result_qdot, result_qddot, result_control, result_history = optimize_spline_q(
                start, frame_times, objective, iterations, learning_rate, offset_log(log_offset),
            )
            return result_q.detach(), result_qdot, result_qddot, {"control_values": result_control.cpu()}, result_history
        if trajectory_mode == "profile_scale":
            result_q, result_payload, result_history = optimize_profile_scale_q(
                start, objective, iterations, learning_rate, offset_log(log_offset)
            )
            if bool(getattr(args, "joint_trajectory_release_per_frame", False)) and len(frame_times) >= 3:
                # Only reached on explicit request: the scale solve lands the
                # magnitude first, then per-frame coordinates refine around it.
                released_q, released_history = optimize_framewise_q(
                    result_q.detach(), objective, iterations, learning_rate,
                    offset_log(int(log_offset) + int(iterations)),
                )
                result_q = released_q
                result_history = (
                    [{**item, "substage": "profile_scale"} for item in result_history]
                    + [{**item, "substage": "released_per_frame"} for item in released_history]
                )
            result_qdot, result_qddot = derive_qdot_qddot(result_q.detach().cpu(), frame_times)
            return result_q.detach(), result_qdot, result_qddot, result_payload, result_history
        result_q, result_history = optimize_framewise_q(
            start, objective, iterations, learning_rate, offset_log(log_offset)
        )
        result_qdot, result_qddot = derive_qdot_qddot(result_q.detach().cpu(), frame_times)
        return result_q.detach(), result_qdot, result_qddot, None, result_history

    q, qdot, qddot, spline_payload, history = optimize_coordinates(
        initial_q, first_iterations, 0
    )
    geometry_history = []
    if geometry_refine:
        geometry_iterations = max(3, min(25, total_iterations // 4))
        joints, geometry_history = refine_joint_geometry_small(
            q,
            joints,
            selected_points,
            camera_maps,
            cache,
            flow_calibration(required=True),
            args,
            iterations=geometry_iterations,
            learning_rate=max(1.0e-4, min(float(args.motion_lr), 1.0e-3)),
            log_fn=offset_log(first_iterations),
        )
        remaining = max(1, total_iterations - first_iterations)
        q, qdot, qddot, spline_payload, second_history = optimize_coordinates(
            q, remaining, first_iterations + geometry_iterations
        )
        history = (
            [{**item, "substage": "q_before_geometry"} for item in history]
            + [{**item, "substage": "q_after_geometry"} for item in second_history]
        )
    q_cpu = q.detach().cpu().double()
    qdot_cpu = torch.as_tensor(qdot).detach().cpu().double()
    qddot_cpu = torch.as_tensor(qddot).detach().cpu().double()
    qjerk_cpu = derive_qjerk(qddot_cpu, frame_times)
    trajectory_direction = torch.sign(q_cpu[:, -1] - q_cpu[:, 0])
    trajectory_direction = torch.where(
        trajectory_direction.abs() > 0,
        trajectory_direction,
        torch.ones_like(trajectory_direction),
    )
    signed_velocity = qdot_cpu * trajectory_direction[:, None]
    final_flow_stats = None
    with torch.no_grad():
        diagnostic_calibration = flow_calibration(required=False)
        if diagnostic_calibration is not None:
            _, final_flow_stats = _trajectory_flow_objective(
                q.to(points), joints, selected_points, camera_maps, cache,
                diagnostic_calibration, args, return_camera_stats=True,
            )
        _, final_render_stats = _trajectory_render_objective(
            q.to(points), joints, points, rotations, labels, render_pairs,
            gaussians, pipe, background, dataset, args, opt,
            return_camera_stats=True,
        )
    tracks = render_joint_constrained_motion(
        points, rotations, labels, joints, q_cpu, qdot_cpu, frame_times
    )
    tracks.update({
        "motion_membership": (labels > 0).float().cpu(),
        "part_probabilities": stage1["responsibilities"].detach().cpu(),
        "fixed_cameras": _fixed_camera_metadata(cameras),
        "joint_coordinates": q_cpu,
        "joint_velocity": qdot_cpu,
        "joint_acceleration": qddot_cpu,
        "joint_jerk": qjerk_cpu,
    })
    _atomic_torch_save({"schema_version": 1, "frame_times": torch.as_tensor(frame_times), "coordinates": q_cpu, "gauge": "q[:,0]=0", "mode": trajectory_mode, "frame_ids": frame_selection["frame_ids"], "trajectory_payload": spline_payload, "spline": spline_payload}, output / "phase3_joint_coordinates.pt")
    _atomic_torch_save({"schema_version": 1, "frame_times": torch.as_tensor(frame_times), "velocity": qdot_cpu}, output / "phase3_joint_velocity.pt")
    _atomic_torch_save({"schema_version": 1, "frame_times": torch.as_tensor(frame_times), "acceleration": qddot_cpu}, output / "phase3_joint_acceleration.pt")
    _atomic_torch_save({"schema_version": 1, "frame_times": torch.as_tensor(frame_times), "jerk": qjerk_cpu}, output / "phase3_joint_jerk.pt")
    _atomic_torch_save({"schema_version": 1, "transforms": tracks["transforms"]}, output / "phase3_final_transforms.pt")
    _atomic_torch_save(tracks, output / "phase3_final_gaussian_tracks.pt")
    _atomic_torch_save(
        {
            "schema_version": 2,
            "joints": joints,
            "frame_times": torch.as_tensor(frame_times),
            "delta_q": torch.diff(q_cpu, dim=-1),
            "q": q_cpu,
            "velocity": qdot_cpu,
            "acceleration": qddot_cpu,
            "jerk": qjerk_cpu,
            "ground_truth_used_as_algorithm_input": False,
        },
        output / "phase3_selected_joint_trajectory.pt",
    )
    _atomic_torch_save(tracks, output / "phase3_selected_joint_tracks.pt")
    if args.joint_geometry_refine:
        _atomic_torch_save(
            {"schema_version": 1, "joints": joints, "history": geometry_history},
            output / "phase3_refined_joint_models.pt",
        )
    summary = {
        "schema_version": 2,
        "trajectory_mode": trajectory_mode,
        "trajectory_mode_requested": requested_mode,
        "trajectory_mode_fallback_reason": mode_fallback,
        "released_per_frame": bool(getattr(args, "joint_trajectory_release_per_frame", False)),
        "frame_selection": {
            key: value for key, value in frame_selection.items() if key != "groups"
        },
        "iterations": int(args.joint_trajectory_iters),
        "frame_count": len(frame_times),
        "joint_count": len(joints),
        "final_render_loss": float(final_render_stats["render_loss"].detach()),
        "final_rgb_loss": float(final_render_stats["rgb_loss"].detach()),
        "final_depth_loss": float(final_render_stats["depth_loss"].detach()),
        "final_dino_loss": float(final_render_stats["dino_loss"].detach()),
        "camera_render_loss": {key: float(value.detach()) for key, value in final_render_stats["camera_render_loss"].items()},
        "flow_diagnostic_skipped_reason": calibration_state["skipped"],
        "final_flow_loss": (
            None if final_flow_stats is None else float(final_flow_stats["flow_loss"].detach())
        ),
        "camera_flow_loss": (
            {}
            if final_flow_stats is None
            else {key: float(value.detach()) for key, value in final_flow_stats["camera_flow_loss"].items()}
        ),
        "joint_coordinate_final": q_cpu[:, -1].tolist(),
        "joint_velocity_mean": qdot_cpu.mean(dim=-1).tolist(),
        "joint_velocity_max_abs": qdot_cpu.abs().amax(dim=-1).tolist(),
        "joint_signed_speed_min": signed_velocity.amin(dim=-1).tolist(),
        "joint_reverse_velocity_count": (signed_velocity < -1.0e-8).sum(dim=-1).tolist(),
        "joint_acceleration_max_abs": qddot_cpu.abs().amax(dim=-1).tolist(),
        "joint_jerk_max_abs": qjerk_cpu.abs().amax(dim=-1).tolist(),
        "trajectory_temporal_weights": {
            "acceleration": float(getattr(args, "joint_trajectory_acceleration_weight", 1.0e-5)),
            "jerk": float(getattr(args, "joint_trajectory_jerk_weight", 1.0e-8)),
            "reverse_velocity": float(getattr(args, "joint_trajectory_reverse_weight", 1.0e-2)),
            "monotonic_constraint": bool(getattr(args, "joint_trajectory_monotonic", True)),
        },
        "static_gaussians_exactly_canonical": True,
        "part_assignment_frozen_from_stage1": True,
        "joint_geometry_refined": bool(geometry_refine),
        "ground_truth_used_as_algorithm_input": False,
        "history": history,
    }
    quaternion_metrics = quaternion_transport_metrics(
        tracks["canonical_xyz"], tracks["rotations"][0], tracks["positions"],
        tracks["rotations"], tracks["part_labels"],
        scales=gaussians.get_scaling.detach().cpu(),
        part_rotation_quaternions=tracks["part_rotation_quaternions"],
    )
    summary["quaternion_transport"] = quaternion_metrics
    if (
        quaternion_metrics["max_quaternion_norm_error"] > 1.0e-5
        or not quaternion_metrics["covariance_rotates_consistently"]
        or quaternion_metrics["center_moves_without_required_orientation"]
        or quaternion_metrics["spatial_rigid_rotation_variance_max"]
        > quaternion_metrics["spatial_rotation_tolerance"]
        or quaternion_metrics["part_quaternion_agreement_error_max"]
        > quaternion_metrics["part_quaternion_agreement_tolerance"]
    ):
        raise RuntimeError("Stage 3 quaternion/covariance transport audit failed")
    if log_fn is not None:
        log_fn("quaternion_transport", int(args.joint_trajectory_iters) + 1, {
            "quaternion_transport/max_norm_error": quaternion_metrics["max_quaternion_norm_error"],
            "quaternion_transport/spatial_rotation_variance": quaternion_metrics["spatial_rigid_rotation_variance_max"],
            "quaternion_transport/part_quaternion_agreement": quaternion_metrics["part_quaternion_agreement_error_max"],
            "quaternion_transport/covariance_error": quaternion_metrics["covariance_transport_error_max"],
        })
    (output / "phase3_metric_motion_summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False), encoding="utf-8")
    write_transported_gaussians_html(
        tracks,
        output / "phase3_final_joint_motion.html",
        title="Stage 3 exact joint-constrained Gaussian motion",
        max_points=int(args.transported_html_max_points),
        max_vectors=min(250, int(args.transported_html_max_points)),
    )
    write_joint_trajectory_html(
        q_cpu,
        qdot_cpu,
        frame_times,
        output / "phase3_joint_trajectory.html",
    )
    return {
        "coordinates": q_cpu,
        "velocity": qdot_cpu,
        "acceleration": qddot_cpu,
        "jerk": qjerk_cpu,
        "tracks": tracks,
        "calibration": calibration_state["value"],
        "calibration_skipped_reason": calibration_state["skipped"],
        "summary": summary,
        "joints": joints,
        "frame_selection": {key: value for key, value in frame_selection.items() if key != "groups"},
        "frame_ids": frame_selection["frame_ids"],
        "frame_times": frame_times,
    }
