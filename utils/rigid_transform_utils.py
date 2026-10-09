"""Exact SE(3) and wxyz quaternion transport shared by rigid motion stages."""

from __future__ import annotations

import json
from pathlib import Path

import torch


def normalize_quaternion(quaternion: torch.Tensor) -> torch.Tensor:
    return torch.nn.functional.normalize(torch.as_tensor(quaternion), dim=-1)


def quaternion_multiply(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    """Hamilton product for local-to-world wxyz quaternions."""
    a = torch.as_tensor(left)
    b = torch.as_tensor(right, device=a.device, dtype=a.dtype)
    aw, ax, ay, az = torch.broadcast_tensors(*a.unbind(-1), *b.unbind(-1))[:4]
    bw, bx, by, bz = torch.broadcast_tensors(*a.unbind(-1), *b.unbind(-1))[4:]
    return torch.stack(
        (
            aw * bw - ax * bx - ay * by - az * bz,
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
        ),
        dim=-1,
    )


def axis_angle_to_quaternion(axis: torch.Tensor, angle: torch.Tensor | float) -> torch.Tensor:
    axis = torch.as_tensor(axis)
    axis = axis / torch.linalg.norm(axis, dim=-1, keepdim=True).clamp_min(1.0e-12)
    angle = torch.as_tensor(angle, device=axis.device, dtype=axis.dtype)
    half = 0.5 * angle
    vector = torch.sin(half)[..., None] * axis
    return normalize_quaternion(torch.cat((torch.cos(half)[..., None], vector), dim=-1))


def quaternion_to_matrix(quaternion: torch.Tensor) -> torch.Tensor:
    q = normalize_quaternion(quaternion)
    w, x, y, z = q.unbind(-1)
    return torch.stack(
        (
            1 - 2 * (y * y + z * z),
            2 * (x * y - w * z),
            2 * (x * z + w * y),
            2 * (x * y + w * z),
            1 - 2 * (x * x + z * z),
            2 * (y * z - w * x),
            2 * (x * z - w * y),
            2 * (y * z + w * x),
            1 - 2 * (x * x + y * y),
        ),
        dim=-1,
    ).reshape(q.shape[:-1] + (3, 3))


def skew_symmetric(vector: torch.Tensor) -> torch.Tensor:
    """Batched skew matrix ``[v]_x`` with ``(...,3) -> (...,3,3)``."""
    v = torch.as_tensor(vector)
    x, y, z = v.unbind(-1)
    zero = torch.zeros_like(x)
    return torch.stack(
        (
            zero, -z, y,
            z, zero, -x,
            -y, x, zero,
        ),
        dim=-1,
    ).reshape(v.shape[:-1] + (3, 3))


def se3_exp_spatial(
    angular_velocity: torch.Tensor,
    linear_velocity: torch.Tensor,
    delta_time: torch.Tensor | float,
    *,
    eps: float = 1.0e-8,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Batched exact exponential of a spatial/world-frame Killing twist.

    The convention is column-vector, active, left-action spatial twist:

    ``x_dot = omega x x + b`` and ``T_{t+1} = Exp(dt * xi_hat) @ T_t``.

    Returns ``(R_inc, p_inc, q_inc_wxyz)`` so callers can transport both Gaussian
    centers and local-to-world Gaussian orientation quaternions with the same
    rigid increment.
    """
    omega = torch.as_tensor(angular_velocity)
    linear = torch.as_tensor(linear_velocity, device=omega.device, dtype=omega.dtype)
    dt = torch.as_tensor(delta_time, device=omega.device, dtype=omega.dtype)
    while dt.ndim < omega.ndim - 1:
        dt = dt[..., None]
    phi = omega * dt[..., None]
    rho = linear * dt[..., None]
    phi_hat = skew_symmetric(phi)
    phi_hat2 = phi_hat @ phi_hat
    theta = torch.linalg.norm(phi, dim=-1)
    theta2 = theta.square()
    small = theta < float(eps)
    a = torch.where(
        small,
        1.0 - theta2 / 6.0 + theta2.square() / 120.0,
        torch.sin(theta) / theta.clamp_min(float(eps)),
    )
    b = torch.where(
        small,
        0.5 - theta2 / 24.0 + theta2.square() / 720.0,
        (1.0 - torch.cos(theta)) / theta2.clamp_min(float(eps)),
    )
    c = torch.where(
        small,
        1.0 / 6.0 - theta2 / 120.0 + theta2.square() / 5040.0,
        (theta - torch.sin(theta)) / (theta2 * theta).clamp_min(float(eps)),
    )
    eye = torch.eye(3, dtype=omega.dtype, device=omega.device).expand(phi_hat.shape)
    rotation = eye + a[..., None, None] * phi_hat + b[..., None, None] * phi_hat2
    left_jacobian = eye + b[..., None, None] * phi_hat + c[..., None, None] * phi_hat2
    translation = (left_jacobian @ rho[..., None])[..., 0]
    half = 0.5 * theta
    quat_scale = torch.where(
        small,
        0.5 - theta2 / 48.0 + theta2.square() / 3840.0,
        torch.sin(half) / theta.clamp_min(float(eps)),
    )
    quaternion = normalize_quaternion(
        torch.cat((torch.cos(half)[..., None], phi * quat_scale[..., None]), dim=-1)
    )
    return rotation, translation, quaternion


def matrix_to_quaternion(matrix: torch.Tensor) -> torch.Tensor:
    """Convert an active rotation matrix to a local-to-world wxyz quaternion."""
    value = torch.as_tensor(matrix)
    m00, m01, m02 = value[..., 0, 0], value[..., 0, 1], value[..., 0, 2]
    m10, m11, m12 = value[..., 1, 0], value[..., 1, 1], value[..., 1, 2]
    m20, m21, m22 = value[..., 2, 0], value[..., 2, 1], value[..., 2, 2]
    # Shepperd's branch selection, made autograd-safe.
    #
    # The naive form takes sqrt of each of the four candidates directly. That is
    # finite in the FORWARD pass but its gradient is not: at the identity three
    # of the four candidates are exactly 0, and d/dx sqrt(x) -> inf there, so
    # every near-identity rotation produced NaN gradients. A near-prismatic
    # joint makes every frame near-identity, which is how the whole scalar
    # trajectory ended up non-finite.
    #
    # The four candidates sum to 4, so the largest is always >= 1. Clamping them
    # all to 0.25 before the sqrt therefore never touches the branch that gets
    # selected, while keeping the three discarded branches finite in backward --
    # torch.where propagates a NaN from the unselected side otherwise.
    c0 = 1 + m00 + m11 + m22
    c1 = 1 + m00 - m11 - m22
    c2 = 1 - m00 + m11 - m22
    c3 = 1 - m00 - m11 + m22
    candidates = torch.stack((c0, c1, c2, c3), dim=-1)
    s = torch.sqrt(candidates.clamp_min(0.25))
    s0, s1, s2, s3 = s[..., 0], s[..., 1], s[..., 2], s[..., 3]

    half = 0.5
    branches = torch.stack((
        torch.stack((half * s0, (m21 - m12) / (2 * s0),
                     (m02 - m20) / (2 * s0), (m10 - m01) / (2 * s0)), dim=-1),
        torch.stack(((m21 - m12) / (2 * s1), half * s1,
                     (m01 + m10) / (2 * s1), (m02 + m20) / (2 * s1)), dim=-1),
        torch.stack(((m02 - m20) / (2 * s2), (m01 + m10) / (2 * s2),
                     half * s2, (m12 + m21) / (2 * s2)), dim=-1),
        torch.stack(((m10 - m01) / (2 * s3), (m02 + m20) / (2 * s3),
                     (m12 + m21) / (2 * s3), half * s3), dim=-1),
    ), dim=-2)                                            # [..., 4 branches, 4]

    # The selection itself carries no gradient: which branch is best is a
    # discrete property of the matrix, not a differentiable function of it.
    best = candidates.detach().argmax(dim=-1)
    quat = branches.gather(
        -2, best[..., None, None].expand(*best.shape, 1, 4)).squeeze(-2)
    # q and -q are the same rotation; pin w >= 0 so the output is continuous with
    # the previous convention.
    sign = torch.where(quat[..., :1] < 0, -torch.ones_like(quat[..., :1]),
                       torch.ones_like(quat[..., :1]))
    return normalize_quaternion(quat * sign)


def compose_se3(
    rotation_a: torch.Tensor,
    translation_a: torch.Tensor,
    rotation_b: torch.Tensor,
    translation_b: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compose column-vector transforms as ``T_a @ T_b``."""
    rotation = rotation_a @ rotation_b
    translation = (rotation_a @ translation_b[..., None])[..., 0] + translation_a
    return rotation, translation


def apply_se3_to_points(
    points: torch.Tensor, rotation: torch.Tensor, translation: torch.Tensor
) -> torch.Tensor:
    return torch.as_tensor(points) @ rotation.transpose(-1, -2) + translation


def se3_exp(
    angular_velocity: torch.Tensor,
    linear_velocity: torch.Tensor,
    delta_q: torch.Tensor | float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Exact exponential of a constant spatial twist over scalar ``delta_q``."""
    angular = torch.as_tensor(angular_velocity)
    linear = torch.as_tensor(linear_velocity, device=angular.device, dtype=angular.dtype)
    value = torch.as_tensor(delta_q, device=angular.device, dtype=angular.dtype).reshape(())
    norm = torch.linalg.norm(angular)
    identity = torch.eye(3, device=angular.device, dtype=angular.dtype)
    if float(norm.detach()) <= 1.0e-12:
        identity_q = torch.tensor([1.0, 0.0, 0.0, 0.0], device=angular.device, dtype=angular.dtype)
        return identity, value * linear, identity_q
    axis = angular / norm
    angle = value * norm
    quaternion = axis_angle_to_quaternion(axis, angle)
    rotation = quaternion_to_matrix(quaternion)
    x, y, z = angular
    zero = angular.new_zeros(())
    generator = torch.stack((
        torch.stack((zero, -z, y)),
        torch.stack((z, zero, -x)),
        torch.stack((-y, x, zero)),
    ))
    matrix = value * generator
    theta2 = angle.square()
    small = angle.abs() < 1.0e-4
    safe = torch.where(small, torch.ones_like(angle), angle)
    cosc = torch.where(
        small, 0.5 - theta2 / 24.0 + theta2.square() / 720.0,
        (1.0 - torch.cos(angle)) / safe.square(),
    )
    third = torch.where(
        small, 1.0 / 6.0 - theta2 / 120.0 + theta2.square() / 5040.0,
        (angle - torch.sin(angle)) / safe.pow(3),
    )
    jacobian = value * (identity + cosc * matrix + third * (matrix @ matrix))
    return rotation, jacobian @ linear, quaternion


def apply_rotation_to_gaussian_quaternions(
    orientations: torch.Tensor, rotation_quaternion: torch.Tensor
) -> torch.Tensor:
    orientation = normalize_quaternion(orientations)
    increment = normalize_quaternion(rotation_quaternion).expand_as(orientation)
    return normalize_quaternion(quaternion_multiply(increment, orientation))


def joint_incremental_transform(
    joint_type: str,
    axis: torch.Tensor,
    delta_q: torch.Tensor | float,
    *,
    pivot: torch.Tensor | None = None,
    pitch: torch.Tensor | float = 0.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    axis = torch.as_tensor(axis)
    axis = axis / torch.linalg.norm(axis).clamp_min(1.0e-12)
    value = torch.as_tensor(delta_q, device=axis.device, dtype=axis.dtype).reshape(())
    identity = torch.eye(3, device=axis.device, dtype=axis.dtype)
    identity_q = torch.tensor([1.0, 0.0, 0.0, 0.0], device=axis.device, dtype=axis.dtype)
    if str(joint_type) == "prismatic":
        return identity, value * axis, identity_q
    rotation_q = axis_angle_to_quaternion(axis, value)
    rotation = quaternion_to_matrix(rotation_q)
    point = torch.zeros_like(axis) if pivot is None else torch.as_tensor(pivot).to(axis)
    point = point - torch.dot(point, axis) * axis
    axial = value * torch.as_tensor(pitch, device=axis.device, dtype=axis.dtype) * axis
    translation = point - rotation @ point + axial
    return rotation, translation, rotation_q


def integrate_joint_transforms(
    joint_type: str,
    axis: torch.Tensor,
    delta_q: torch.Tensor,
    *,
    pivot: torch.Tensor | None = None,
    pitch: torch.Tensor | float = 0.0,
) -> dict[str, torch.Tensor]:
    """Integrate exact increments and retain the part quaternion at every frame."""
    increments = torch.as_tensor(delta_q)
    identity = torch.eye(3, device=increments.device, dtype=increments.dtype)
    zero = torch.zeros(3, device=increments.device, dtype=increments.dtype)
    identity_q = torch.tensor([1.0, 0.0, 0.0, 0.0], device=increments.device, dtype=increments.dtype)
    rotations, translations, quaternions = [identity], [zero], [identity_q]
    current_rotation, current_translation, current_quaternion = identity, zero, identity_q
    for value in increments.reshape(-1):
        delta_rotation, delta_translation, delta_quaternion = joint_incremental_transform(
            joint_type, axis, value, pivot=pivot, pitch=pitch
        )
        current_rotation, current_translation = compose_se3(
            current_rotation, current_translation, delta_rotation, delta_translation
        )
        current_quaternion = normalize_quaternion(
            quaternion_multiply(delta_quaternion, current_quaternion)
        )
        rotations.append(current_rotation)
        translations.append(current_translation)
        quaternions.append(current_quaternion)
    return {
        "rotations": torch.stack(rotations),
        "translations": torch.stack(translations),
        "rotation_quaternions": torch.stack(quaternions),
        "q": torch.cat((increments.new_zeros(1), torch.cumsum(increments, dim=0))),
    }


def direct_joint_transform(joint: dict, q: torch.Tensor | float):
    return joint_incremental_transform(
        str(joint["joint_type"]),
        torch.as_tensor(joint["axis"]),
        q,
        pivot=joint.get("pivot"),
        pitch=joint.get("pitch", 0.0),
    )


def transport_gaussian_state(
    positions: torch.Tensor,
    orientations: torch.Tensor,
    rotation: torch.Tensor,
    translation: torch.Tensor,
    rotation_quaternion: torch.Tensor,
    mask: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    moved = apply_se3_to_points(positions, rotation, translation)
    rotated = apply_rotation_to_gaussian_quaternions(orientations, rotation_quaternion)
    if mask is None:
        return moved, rotated
    selected = torch.as_tensor(mask, device=moved.device).bool()
    return (
        torch.where(selected[:, None], moved, positions),
        torch.where(selected[:, None], rotated, normalize_quaternion(orientations)),
    )


def covariance_from_quaternion_scale(
    orientations: torch.Tensor, scales: torch.Tensor
) -> torch.Tensor:
    rotation = quaternion_to_matrix(orientations)
    diagonal = torch.diag_embed(torch.as_tensor(scales).square())
    return rotation @ diagonal @ rotation.transpose(-1, -2)


def quaternion_transport_metrics(
    canonical_positions: torch.Tensor,
    canonical_orientations: torch.Tensor,
    positions: torch.Tensor,
    orientations: torch.Tensor,
    labels: torch.Tensor,
    *,
    scales: torch.Tensor | None = None,
    part_rotation_quaternions: torch.Tensor | None = None,
) -> dict:
    orientation_source = torch.as_tensor(orientations)
    source_epsilon = (
        torch.finfo(orientation_source.dtype).eps
        if orientation_source.is_floating_point()
        else 0.0
    )
    spatial_tolerance = max(1.0e-8, 64.0 * source_epsilon)
    part_agreement_tolerance = max(1.0e-8, 1024.0 * source_epsilon)
    covariance_tolerance = max(1.0e-5, 1024.0 * source_epsilon)
    xyz0 = torch.as_tensor(canonical_positions).detach().cpu().double()
    q0 = normalize_quaternion(torch.as_tensor(canonical_orientations).detach().cpu().double())
    xyz = torch.as_tensor(positions).detach().cpu().double()
    raw_q = torch.as_tensor(orientations).detach().cpu().double()
    q = normalize_quaternion(raw_q)
    label = torch.as_tensor(labels).detach().cpu().long()
    norm_error = (torch.linalg.norm(raw_q, dim=-1) - 1.0).abs()
    moving = label > 0
    center_moved = bool(moving.any() and ((xyz[-1, moving] - xyz0[moving]).norm(dim=-1) > 1.0e-8).any())
    base_rotation = quaternion_to_matrix(q0)
    transported_rotation = quaternion_to_matrix(q)
    orientation_changed = bool(
        moving.any()
        and (
            torch.linalg.norm(
                transported_rotation[-1, moving] @ base_rotation[moving].transpose(-1, -2)
                - torch.eye(3, dtype=torch.float64),
                dim=(-2, -1),
            )
            > 1.0e-8
        ).any()
    )
    spatial_errors = []
    part_agreement_errors = []
    covariance_errors = []
    rotational_part_present = False
    if part_rotation_quaternions is not None:
        part_q = normalize_quaternion(torch.as_tensor(part_rotation_quaternions).detach().cpu().double())
        identity = torch.eye(3, dtype=torch.float64)
        rotational_part_present = bool(
            (torch.linalg.norm(quaternion_to_matrix(part_q) - identity, dim=(-2, -1)) > 1.0e-8).any()
        )
        for part in range(1, int(label.max().item()) + 1 if label.numel() else 1):
            selected = label == part
            if not bool(selected.any()):
                continue
            relative = transported_rotation[:, selected] @ base_rotation[selected].transpose(-1, -2)
            expected = quaternion_to_matrix(part_q[:, part - 1])[:, None]
            spatial_errors.append(
                torch.linalg.norm(relative - relative[:, :1], dim=(-2, -1)).reshape(-1)
            )
            part_agreement_errors.append(
                torch.linalg.norm(relative - expected, dim=(-2, -1)).reshape(-1)
            )
            if scales is not None:
                scale = torch.as_tensor(scales).detach().cpu().double()[selected]
                canonical_covariance = covariance_from_quaternion_scale(q0[selected], scale)
                actual = covariance_from_quaternion_scale(q[:, selected], scale[None].expand(q.shape[0], -1, -1))
                wanted = expected @ canonical_covariance[None] @ expected.transpose(-1, -2)
                covariance_errors.append(torch.linalg.norm(actual - wanted, dim=(-2, -1)).reshape(-1))
    spatial = torch.cat(spatial_errors) if spatial_errors else torch.zeros(1)
    part_agreement = (
        torch.cat(part_agreement_errors)
        if part_agreement_errors
        else torch.zeros(1)
    )
    covariance = torch.cat(covariance_errors) if covariance_errors else torch.zeros(1)
    return {
        "quaternion_convention": "wxyz local-to-world active rotation",
        "quaternion_composition_order": "q_world_delta ⊗ q_gaussian_canonical",
        "killing_replay_updates_xyz": center_moved,
        "killing_replay_updates_quaternion": orientation_changed,
        "rotational_part_present": rotational_part_present,
        "center_moves_without_required_orientation": bool(
            center_moved and rotational_part_present and not orientation_changed
        ),
        "covariance_rotates_consistently": float(covariance.max()) <= covariance_tolerance,
        "part_level_cumulative_quaternion_available": part_rotation_quaternions is not None,
        "max_quaternion_norm_error": float(norm_error.max()),
        "median_quaternion_norm_error": float(norm_error.median()),
        "spatial_rigid_rotation_variance_max": float(spatial.max()),
        "part_quaternion_agreement_error_max": float(part_agreement.max()),
        "covariance_transport_error_max": float(covariance.max()),
        "spatial_rotation_tolerance": spatial_tolerance,
        "part_quaternion_agreement_tolerance": part_agreement_tolerance,
        "covariance_transport_tolerance": covariance_tolerance,
    }


def write_quaternion_transport_audit(
    tracks: dict,
    output_dir: str | Path,
    *,
    scales: torch.Tensor | None = None,
    max_debug_gaussians: int = 64,
) -> dict:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    metrics = quaternion_transport_metrics(
        tracks["canonical_xyz"],
        tracks["rotations"][0],
        tracks["positions"],
        tracks["rotations"],
        tracks["part_labels"],
        scales=scales,
        part_rotation_quaternions=tracks.get("part_rotation_quaternions"),
    )
    if metrics["max_quaternion_norm_error"] > 1.0e-5:
        raise RuntimeError("Rigid transport quaternion norm drift exceeded 1e-5")
    if metrics["center_moves_without_required_orientation"]:
        raise RuntimeError("Rigid centers moved while rotational Gaussian orientations stayed canonical")
    if (
        metrics["spatial_rigid_rotation_variance_max"]
        > metrics["spatial_rotation_tolerance"]
    ):
        raise RuntimeError("Gaussians in one rigid part do not share one rotation")
    if (
        metrics["part_quaternion_agreement_error_max"]
        > metrics["part_quaternion_agreement_tolerance"]
    ):
        raise RuntimeError("Gaussian transport disagrees with the stored part quaternion")
    if not metrics["covariance_rotates_consistently"]:
        raise RuntimeError("Rigid quaternion and covariance transport disagree")
    (output / "killing_quaternion_transport_audit.json").write_text(
        json.dumps(metrics, indent=2, allow_nan=False), encoding="utf-8"
    )
    _write_orientation_debug_html(
        tracks, output / "killing_gaussian_orientation_debug.html", max_debug_gaussians
    )
    print(
        "[Quaternion transport audit] "
        f"xyz={'yes' if metrics['killing_replay_updates_xyz'] else 'no'} "
        f"quaternion={'yes' if metrics['killing_replay_updates_quaternion'] else 'no'} "
        f"covariance={'yes' if metrics['covariance_rotates_consistently'] else 'no'} "
        f"part_quaternion={'yes' if metrics['part_level_cumulative_quaternion_available'] else 'no'} "
        f"part_error={metrics['part_quaternion_agreement_error_max']:.3e} "
        f"convention={metrics['quaternion_convention']} "
        f"composition={metrics['quaternion_composition_order']}"
    )
    return metrics


def _write_orientation_debug_html(tracks: dict, output: Path, max_points: int):
    positions = torch.as_tensor(tracks["positions"]).detach().cpu().float()
    rotations = quaternion_to_matrix(torch.as_tensor(tracks["rotations"]).detach().cpu().float())
    labels = torch.as_tensor(tracks["part_labels"]).detach().cpu().long()
    selected = torch.nonzero(labels > 0, as_tuple=False).flatten()
    if selected.numel() > int(max_points):
        selected = selected[torch.linspace(0, selected.numel() - 1, int(max_points)).round().long()]
    frames = sorted({0, positions.shape[0] // 2, positions.shape[0] - 1})
    traces = []
    colors = ("#e41a1c", "#4daf4a", "#377eb8")
    extent = float(torch.linalg.norm(positions[0].amax(0) - positions[0].amin(0)).clamp_min(1.0e-6))
    length = 0.025 * extent
    for frame in frames:
        for axis in range(3):
            x, y, z = [], [], []
            for index in selected.tolist():
                start = positions[frame, index]
                end = start + length * rotations[frame, index, :, axis]
                x.extend((float(start[0]), float(end[0]), None))
                y.extend((float(start[1]), float(end[1]), None))
                z.extend((float(start[2]), float(end[2]), None))
            traces.append({
                "type": "scatter3d", "mode": "lines", "name": f"frame {frame} axis {axis}",
                "x": x, "y": y, "z": z, "line": {"width": 3, "color": colors[axis]},
                "visible": frame == frames[0],
            })
    buttons = []
    for frame in frames:
        buttons.append({
            "label": f"frame {frame}", "method": "update",
            "args": [{"visible": [trace["name"].startswith(f"frame {frame} ") for trace in traces]}],
        })
    document = f"""<!doctype html><html><head><meta charset='utf-8'><script src='https://cdn.plot.ly/plotly-2.35.2.min.js'></script></head><body><div id='plot' style='width:100vw;height:100vh'></div><script>Plotly.newPlot('plot',{json.dumps(traces)},{json.dumps({'title': 'Killing Gaussian orientation transport', 'scene': {'aspectmode': 'data'}, 'updatemenus': [{'buttons': buttons, 'direction': 'down'}]})},{{responsive:true}});</script></body></html>"""
    output.write_text(document, encoding="utf-8")
