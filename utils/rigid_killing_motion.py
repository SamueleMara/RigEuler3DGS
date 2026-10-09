"""Closed-form rigid Killing-field fitting and exact SE(3) transport.

The fixed ``reference_point`` is only a coordinate gauge chosen from real
motion support.  It is not a physical joint parameter.  All linear algebra in
the fitting path is performed on CPU in float64; returned solver state remains
float64 and can be moved to the rendering device for replay.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from utils.rigid_transform_utils import (
    apply_rotation_to_gaussian_quaternions,
    axis_angle_to_quaternion,
    se3_exp,
)


_EPS64 = 1.0e-12


@dataclass
class RigidKillingMode:
    reference_point: torch.Tensor
    reference_velocity: torch.Tensor
    angular_mode: torch.Tensor
    mode_type: str
    normalization_scale: float
    valid: bool
    diagnostics: dict[str, Any]

    def to(self, *, device=None, dtype=None) -> "RigidKillingMode":
        return RigidKillingMode(
            reference_point=self.reference_point.to(device=device, dtype=dtype),
            reference_velocity=self.reference_velocity.to(device=device, dtype=dtype),
            angular_mode=self.angular_mode.to(device=device, dtype=dtype),
            mode_type=self.mode_type,
            normalization_scale=self.normalization_scale,
            valid=self.valid,
            diagnostics=self.diagnostics,
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "reference_point": self.reference_point.detach().cpu(),
            "reference_velocity": self.reference_velocity.detach().cpu(),
            "angular_mode": self.angular_mode.detach().cpu(),
            "mode_type": self.mode_type,
            "normalization_scale": float(self.normalization_scale),
            "valid": bool(self.valid),
            "diagnostics": self.diagnostics,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "RigidKillingMode":
        """Inverse of :meth:`as_dict`.

        Reconstruction lives next to serialisation so a consumer reloading a
        checkpoint cannot get the field set subtly wrong -- an omitted
        ``normalization_scale`` would silently rescale every predicted velocity.
        """
        return cls(
            reference_point=torch.as_tensor(payload["reference_point"]),
            reference_velocity=torch.as_tensor(payload["reference_velocity"]),
            angular_mode=torch.as_tensor(payload["angular_mode"]),
            mode_type=payload["mode_type"],
            normalization_scale=float(payload["normalization_scale"]),
            valid=bool(payload["valid"]),
            diagnostics=dict(payload.get("diagnostics") or {}),
        )


@dataclass
class RigidIntervalFit:
    speed: torch.Tensor
    predicted_cell_velocity: torch.Tensor
    residual_per_cell: torch.Tensor
    valid_mask: torch.Tensor
    diagnostics: dict[str, Any]


@dataclass
class RigidObservationGrid:
    origin: torch.Tensor
    cell_size: torch.Tensor
    cell_counts: torch.Tensor
    point_group_ids: torch.Tensor
    cell_centers: torch.Tensor
    point_counts: torch.Tensor

    @property
    def num_cells(self) -> int:
        return int(self.cell_centers.shape[0])


#: Shared robust-solver settings.  Both the rigid and the translation-only fit
#: must use identical numerics, or their residuals would differ by the solver
#: rather than by the model and could not be compared.
_DEFAULT_IRLS_ITERS = 5
_DEFAULT_HUBER_DELTA = 0.05
_DEFAULT_MIN_VALID_CELLS = 8


def _float64_cpu(value: torch.Tensor) -> torch.Tensor:
    return torch.as_tensor(value).detach().to(device="cpu", dtype=torch.float64)


def _cross_matrix(vector: torch.Tensor) -> torch.Tensor:
    """Return matrices W satisfying ``W @ y == vector cross y``."""
    x, y, z = vector.unbind(dim=-1)
    zero = torch.zeros_like(x)
    return torch.stack(
        (
            zero,
            -z,
            y,
            z,
            zero,
            -x,
            -y,
            x,
            zero,
        ),
        dim=-1,
    ).reshape(vector.shape[:-1] + (3, 3))


def build_rigid_observation_grid(
    points: torch.Tensor,
    target_cells: int,
    *,
    padding_fraction: float = 0.05,
    min_cells_per_axis: int = 2,
) -> RigidObservationGrid:
    """Build compact occupied observation cells without a trainable field."""
    xyz = _float64_cpu(points)
    if xyz.ndim != 2 or xyz.shape[-1] != 3 or xyz.shape[0] == 0:
        raise ValueError("points must be non-empty [N,3]")
    lower, upper = xyz.amin(dim=0), xyz.amax(dim=0)
    raw_extent = (upper - lower).clamp_min(1.0e-8)
    padding = max(0.0, float(padding_fraction)) * raw_extent.max()
    origin = lower - padding
    extent = (upper + padding - origin).clamp_min(1.0e-8)
    target = max(8, int(target_cells))
    cell_width = (float(torch.prod(extent)) / target) ** (1.0 / 3.0)
    counts = torch.ceil(extent / max(cell_width, _EPS64)).long().clamp_min(
        int(min_cells_per_axis)
    )
    size = extent / counts.to(torch.float64)
    coordinate = torch.floor((xyz - origin) / size).long()
    coordinate = torch.minimum(
        torch.maximum(coordinate, torch.zeros_like(coordinate)), counts - 1
    )
    raw_ids = (coordinate[:, 0] * counts[1] + coordinate[:, 1]) * counts[2] + coordinate[:, 2]
    occupied_raw, compact_ids = torch.unique(raw_ids, sorted=True, return_inverse=True)
    ix = occupied_raw // (counts[1] * counts[2])
    rem = occupied_raw % (counts[1] * counts[2])
    iy, iz = rem // counts[2], rem % counts[2]
    occupied_coordinate = torch.stack((ix, iy, iz), dim=-1)
    centers = origin + (occupied_coordinate.to(torch.float64) + 0.5) * size
    point_counts = torch.bincount(compact_ids, minlength=centers.shape[0])
    return RigidObservationGrid(
        origin=origin,
        cell_size=size,
        cell_counts=counts,
        point_group_ids=compact_ids,
        cell_centers=centers,
        point_counts=point_counts,
    )


def aggregate_points_to_rigid_cells(
    values: torch.Tensor,
    group_ids: torch.Tensor,
    num_cells: int,
    *,
    weights: torch.Tensor | None = None,
) -> torch.Tensor:
    """Weighted mean of point values in compact rigid observation cells."""
    if values.ndim == 1:
        values = values[:, None]
        squeeze = True
    else:
        squeeze = False
    ids = group_ids.to(device=values.device, dtype=torch.long).reshape(-1)
    if values.shape[0] != ids.shape[0]:
        raise ValueError("values and group_ids must have the same leading size")
    w = torch.ones(ids.shape[0], device=values.device, dtype=values.dtype)
    if weights is not None:
        w = weights.to(device=values.device, dtype=values.dtype).reshape(-1)
    out = torch.zeros((num_cells, values.shape[1]), device=values.device, dtype=values.dtype)
    denominator = torch.zeros(num_cells, device=values.device, dtype=values.dtype)
    out.index_add_(0, ids, values * w[:, None])
    denominator.index_add_(0, ids, w)
    out = out / denominator.clamp_min(torch.finfo(values.dtype).eps)[:, None]
    return out[:, 0] if squeeze else out


def _coordinate_neighbor_edges(points: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    indices = torch.nonzero(valid, as_tuple=False).flatten()
    if indices.numel() < 2:
        return torch.empty((0, 2), dtype=torch.long)
    edges = []
    for axis in range(3):
        ordered = indices[torch.argsort(points[indices, axis])]
        edges.append(torch.stack((ordered[:-1], ordered[1:]), dim=-1))
        if ordered.numel() > 2:
            edges.append(torch.stack((ordered[:-2], ordered[2:]), dim=-1))
    return torch.unique(torch.cat(edges, dim=0).sort(dim=-1).values, dim=0)


def _huber_weights(residual: torch.Tensor, delta: float) -> torch.Tensor:
    if residual.numel() == 0:
        return residual
    threshold = float(delta)
    if threshold <= 0.0:
        threshold = max(
            1.0e-8,
            1.4826 * float(torch.median(residual.detach()).clamp_min(1.0e-8)),
        )
    return torch.where(
        residual <= threshold,
        torch.ones_like(residual),
        threshold / residual.clamp_min(_EPS64),
    )


def _solve_weighted_system(
    design: torch.Tensor,
    target: torch.Tensor,
    weights: torch.Tensor,
    *,
    relative_ridge: float = 0.0,
) -> tuple[torch.Tensor, dict[str, float]]:
    sqrt_weight = weights.clamp_min(0.0).sqrt().repeat_interleave(target.shape[-1])
    rows = design.reshape(-1, design.shape[-1])
    rhs_rows = target.reshape(-1)
    weighted_rows = rows * sqrt_weight[:, None]
    weighted_rhs = rhs_rows * sqrt_weight
    normal = weighted_rows.T @ weighted_rows
    rhs = weighted_rows.T @ weighted_rhs
    eigenvalues = torch.linalg.eigvalsh(normal)
    largest = float(eigenvalues[-1]) if eigenvalues.numel() else 0.0
    smallest = float(eigenvalues[0]) if eigenvalues.numel() else 0.0
    ridge = max(0.0, float(relative_ridge)) * max(largest, _EPS64)
    if smallest <= max(largest, 1.0) * 1.0e-12:
        ridge = max(ridge, max(largest, 1.0) * 1.0e-10)
    regularized = normal + ridge * torch.eye(normal.shape[0], dtype=normal.dtype)
    solution = torch.linalg.solve(regularized, rhs)
    condition_number = largest / max(smallest, _EPS64) if largest > 0.0 else float("inf")
    return solution, {
        "condition_number": condition_number,
        "conditioning_ratio": smallest / max(largest, _EPS64),
        "ridge": ridge,
    }


def fit_local_angular_velocity_from_edges(
    cell_positions: torch.Tensor,
    observed_velocity: torch.Tensor,
    *,
    weights: torch.Tensor | None = None,
    valid_mask: torch.Tensor | None = None,
    edges: torch.Tensor | None = None,
    irls_iters: int = 5,
    huber_delta: float = 0.05,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Origin- and translation-independent angular solve from relative velocity."""
    positions, velocity = _float64_cpu(cell_positions), _float64_cpu(observed_velocity)
    if positions.shape != velocity.shape or positions.ndim != 2 or positions.shape[-1] != 3:
        raise ValueError("cell_positions and observed_velocity must both be [C,3]")
    weight = torch.ones(positions.shape[0], dtype=torch.float64)
    if weights is not None:
        weight = _float64_cpu(weights).reshape(-1)
    valid = torch.isfinite(positions).all(dim=-1) & torch.isfinite(velocity).all(dim=-1) & (weight > 0)
    if valid_mask is not None:
        valid &= torch.as_tensor(valid_mask).detach().cpu().bool().reshape(-1)
    pair = _coordinate_neighbor_edges(positions, valid) if edges is None else torch.as_tensor(edges).detach().cpu().long()
    if pair.ndim != 2 or pair.shape[-1] != 2:
        raise ValueError("edges must be [E,2]")
    if pair.numel():
        pair_valid = valid[pair].all(dim=-1) & (torch.linalg.norm(positions[pair[:, 1]] - positions[pair[:, 0]], dim=-1) > 1.0e-10)
        pair = pair[pair_valid]
    if pair.shape[0] < 3:
        return torch.zeros(3, dtype=torch.float64), {
            "valid": False,
            "edge_count": int(pair.shape[0]),
            "condition_number": float("inf"),
            "weighted_rms": float("inf"),
        }
    delta_x = positions[pair[:, 1]] - positions[pair[:, 0]]
    delta_v = velocity[pair[:, 1]] - velocity[pair[:, 0]]
    design = -_cross_matrix(delta_x)
    base_weight = torch.sqrt(weight[pair[:, 0]] * weight[pair[:, 1]])
    robust = torch.ones_like(base_weight)
    omega = torch.zeros(3, dtype=torch.float64)
    solve_diag: dict[str, float] = {}
    for _ in range(max(1, int(irls_iters))):
        omega, solve_diag = _solve_weighted_system(
            design, delta_v, base_weight * robust
        )
        residual = torch.linalg.norm(torch.einsum("eij,j->ei", design, omega) - delta_v, dim=-1)
        robust = _huber_weights(residual, huber_delta)
    effective = base_weight * robust
    rms = torch.sqrt(
        (effective * residual.square()).sum() / effective.sum().clamp_min(_EPS64)
    )
    return omega, {
        "valid": bool(torch.isfinite(omega).all()),
        "edge_count": int(pair.shape[0]),
        "weighted_rms": float(rms),
        "robust_weight_mean": float(robust.mean()),
        **solve_diag,
    }


def predict_rigid_cell_velocity(
    mode: RigidKillingMode,
    cell_positions: torch.Tensor,
    speed: torch.Tensor | float,
) -> torch.Tensor:
    positions = torch.as_tensor(cell_positions)
    reference = mode.reference_point.to(positions)
    linear = mode.reference_velocity.to(positions)
    angular = mode.angular_mode.to(positions)
    spatial_mode = linear + torch.cross(
        angular.expand_as(positions), positions - reference, dim=-1
    )
    scalar = torch.as_tensor(speed, device=positions.device, dtype=positions.dtype)
    if scalar.ndim == 0:
        return scalar * spatial_mode
    if positions.ndim == 2:
        return scalar[:, None, None] * spatial_mode[None]
    return scalar[:, None, None] * spatial_mode


def rigid_residuals(
    mode: RigidKillingMode,
    cell_positions: torch.Tensor,
    observed_velocity: torch.Tensor,
    speeds: torch.Tensor,
) -> torch.Tensor:
    predicted = predict_rigid_cell_velocity(mode, cell_positions, speeds)
    return torch.linalg.norm(torch.as_tensor(observed_velocity).to(predicted) - predicted, dim=-1)


def solve_rigid_field_single_interval(
    cell_positions: torch.Tensor,
    observed_velocity: torch.Tensor,
    *,
    weights: torch.Tensor | None = None,
    valid_mask: torch.Tensor | None = None,
    reference_point: torch.Tensor | None = None,
    edges: torch.Tensor | None = None,
    irls_iters: int = 5,
    huber_delta: float = 0.05,
    min_valid_cells: int = 4,
) -> RigidIntervalFit:
    positions, velocity = _float64_cpu(cell_positions), _float64_cpu(observed_velocity)
    weight = torch.ones(positions.shape[0], dtype=torch.float64)
    if weights is not None:
        weight = _float64_cpu(weights).reshape(-1)
    valid = torch.isfinite(positions).all(dim=-1) & torch.isfinite(velocity).all(dim=-1) & (weight > 0)
    if valid_mask is not None:
        valid &= torch.as_tensor(valid_mask).detach().cpu().bool().reshape(-1)
    predicted = torch.zeros_like(velocity)
    residual = torch.full((positions.shape[0],), float("inf"), dtype=torch.float64)
    if int(valid.sum()) < max(3, int(min_valid_cells)):
        return RigidIntervalFit(
            speed=torch.tensor(0.0, dtype=torch.float64),
            predicted_cell_velocity=predicted,
            residual_per_cell=residual,
            valid_mask=valid,
            diagnostics={"valid": False, "valid_cells": int(valid.sum()), "reason": "insufficient_valid_cells"},
        )
    if reference_point is None:
        reference = positions[torch.argmax(torch.where(valid, weight, torch.zeros_like(weight)))].clone()
    else:
        reference = _float64_cpu(reference_point).reshape(3)
    omega, angular_diag = fit_local_angular_velocity_from_edges(
        positions,
        velocity,
        weights=weight,
        valid_mask=valid,
        edges=edges,
        irls_iters=irls_iters,
        huber_delta=huber_delta,
    )
    relative = positions - reference
    rotational = torch.cross(omega.expand_as(relative), relative, dim=-1)
    cell_robust = torch.ones_like(weight)
    reference_velocity = torch.zeros(3, dtype=torch.float64)
    for _ in range(max(1, int(irls_iters))):
        effective = weight * cell_robust * valid.to(torch.float64)
        reference_velocity = (
            effective[:, None] * (velocity - rotational)
        ).sum(dim=0) / effective.sum().clamp_min(_EPS64)
        predicted = reference_velocity + rotational
        residual = torch.linalg.norm(velocity - predicted, dim=-1)
        cell_robust = _huber_weights(residual, huber_delta)
    effective = weight * cell_robust * valid.to(torch.float64)
    rigid_rms = torch.sqrt(
        (effective * residual.square()).sum() / effective.sum().clamp_min(_EPS64)
    )
    static_rms = torch.sqrt(
        (effective * velocity.square().sum(dim=-1)).sum()
        / effective.sum().clamp_min(_EPS64)
    )
    diagnostics = {
        "valid": bool(angular_diag["valid"] and torch.isfinite(reference_velocity).all()),
        "valid_cells": int(valid.sum()),
        "reference_point": reference.tolist(),
        "reference_velocity": reference_velocity.tolist(),
        "angular_velocity": omega.tolist(),
        "rigid_rms": float(rigid_rms),
        "static_rms": float(static_rms),
        "rigid_improvement": float(static_rms - rigid_rms),
        "robust_weight_mean": float(cell_robust[valid].mean()),
        "angular_solve": angular_diag,
    }
    diagnostics["generator"] = torch.cat((reference_velocity, omega))
    return RigidIntervalFit(
        speed=torch.tensor(1.0, dtype=torch.float64),
        predicted_cell_velocity=predicted,
        residual_per_cell=residual,
        valid_mask=valid,
        diagnostics=diagnostics,
    )


def screw_geometry(
    reference_velocity: torch.Tensor, angular_mode: torch.Tensor
) -> dict[str, float]:
    """Gauge-invariant screw decomposition of one rigid generator.

    ``lever_arm`` is the perpendicular distance from the reference point to the
    instantaneous rotation axis, ``|v_perp| / |omega|``.  It is invariant to the
    overall scale of the generator, so it survives the unit normalization the
    alternating solver applies and is the only magnitude in this record that can
    be compared against a physical length.  ``pitch`` is the axial translation
    per radian.  Both are undefined for a generator with no angular part, which
    is reported as an infinite lever arm.
    """
    linear = _float64_cpu(reference_velocity).reshape(3)
    angular = _float64_cpu(angular_mode).reshape(3)
    angular_norm = float(torch.linalg.norm(angular))
    linear_norm = float(torch.linalg.norm(linear))
    if angular_norm <= _EPS64:
        return {
            "angular_norm": angular_norm,
            "linear_norm": linear_norm,
            "lever_arm": float("inf"),
            "pitch": float("inf") if linear_norm > _EPS64 else 0.0,
        }
    direction = angular / angular_norm
    axial = float(torch.dot(linear, direction))
    perpendicular = float(torch.linalg.norm(linear - axial * direction))
    return {
        "angular_norm": angular_norm,
        "linear_norm": linear_norm,
        "lever_arm": perpendicular / angular_norm,
        "pitch": axial / angular_norm,
    }


def instantaneous_axis(mode: RigidKillingMode) -> dict[str, torch.Tensor] | None:
    """The instantaneous screw axis of one rigid generator, as a line.

    Returns the unit direction and the point on the axis closest to the mode's
    reference point, i.e. exactly the ``(axis, pivot)`` pair a revolute or screw
    joint is fitted with downstream.  For a generator with no angular part the
    axis is at infinity and there is nothing to return.

    The pivot is *not* projected onto ``pivot . axis == 0`` here: that gauge
    slides the point along the axis, which the joint fits want for a canonical
    representation but which is irrelevant to any perpendicular distance.
    """
    angular = _float64_cpu(mode.angular_mode).reshape(3)
    angular_norm = float(torch.linalg.norm(angular))
    if angular_norm <= _EPS64:
        return None
    axis = angular / angular_norm
    reference = _float64_cpu(mode.reference_point).reshape(3)
    velocity = _float64_cpu(mode.reference_velocity).reshape(3) / angular_norm
    perpendicular = velocity - torch.dot(velocity, axis) * axis
    return {
        "axis": axis,
        "pivot": reference + torch.cross(axis, perpendicular, dim=-1),
        "angular_norm": angular_norm,
        "pitch": float(torch.dot(velocity, axis)),
    }


def part_extent(part_points: torch.Tensor) -> float:
    """RMS radius of a point set about its own centroid."""
    points = _float64_cpu(part_points).reshape(-1, 3)
    if points.shape[0] == 0:
        return float("nan")
    centred = points - points.mean(dim=0)
    return float(centred.norm(dim=-1).square().mean().sqrt())


def solve_translation_mode_all_intervals(
    cell_positions: torch.Tensor,
    observed_velocity: torch.Tensor,
    *,
    weights: torch.Tensor | None = None,
    valid_mask: torch.Tensor | None = None,
    irls_iters: int = _DEFAULT_IRLS_ITERS,
    huber_delta: float = _DEFAULT_HUBER_DELTA,
    min_valid_cells: int = _DEFAULT_MIN_VALID_CELLS,
) -> tuple[RigidKillingMode, torch.Tensor, dict[str, Any]]:
    """Fit the same field family with the angular part constrained to zero.

    This is the prismatic hypothesis expressed in the Stage-1 representation: a
    single spatial direction plus one signed scalar speed per interval, with
    ``omega == 0`` so every point of the part moves alike.  It exists so the two
    joint types can be *fitted* and compared on how well each explains the
    observed velocity, instead of being separated by a threshold on some derived
    ratio.  The residual it reports is directly comparable with the one from
    ``solve_rigid_killing_mode_all_intervals``, because both are the same
    confidence-weighted mean square of ``|v_observed - v_predicted|`` over the
    same observations.
    """
    velocity = _float64_cpu(observed_velocity)
    positions = _float64_cpu(cell_positions)
    if velocity.ndim != 3 or velocity.shape[-1] != 3:
        raise ValueError("observed_velocity must be [K,C,3]")
    if positions.ndim == 2:
        positions = positions[None].expand(velocity.shape[0], -1, -1).clone()
    base_weight = torch.ones(velocity.shape[:2], dtype=torch.float64)
    if weights is not None:
        base_weight = _float64_cpu(weights)
    valid = torch.isfinite(velocity).all(dim=-1) & (base_weight > 0)
    if valid_mask is not None:
        valid &= torch.as_tensor(valid_mask).detach().cpu().bool()

    # The direction is the weighted principal axis of the observed velocities:
    # a translating part moves along one line, so all its velocity mass lies on
    # that line whatever the per-interval speed and sign happen to be.
    effective = (base_weight * valid.to(torch.float64)).reshape(-1)
    flat = velocity.reshape(-1, 3)
    robust = torch.ones_like(effective)
    direction = torch.zeros(3, dtype=torch.float64)
    speeds = torch.zeros(velocity.shape[0], dtype=torch.float64)
    residual = torch.zeros(velocity.shape[:2], dtype=torch.float64)
    for _ in range(max(1, int(irls_iters))):
        w = effective * robust
        scatter = (w[:, None] * flat).T @ flat
        eigenvalues, eigenvectors = torch.linalg.eigh(scatter)
        direction = eigenvectors[:, -1]
        if float(torch.dot(direction, (w[:, None] * flat).sum(dim=0))) < 0:
            direction = -direction
        speeds = (
            (base_weight * robust.reshape(velocity.shape[:2]) * valid.to(torch.float64))
            * (velocity @ direction)
        ).sum(dim=1) / (
            base_weight * robust.reshape(velocity.shape[:2]) * valid.to(torch.float64)
        ).sum(dim=1).clamp_min(_EPS64)
        predicted = speeds[:, None, None] * direction
        residual = torch.linalg.norm(velocity - predicted, dim=-1)
        robust = _huber_weights(residual.reshape(-1), huber_delta)

    weighted = base_weight * valid.to(torch.float64)
    translation_rms = torch.sqrt(
        (weighted * residual.square()).sum() / weighted.sum().clamp_min(_EPS64)
    )
    static_rms = torch.sqrt(
        (weighted[:, :, None] * velocity.square()).sum() / weighted.sum().clamp_min(_EPS64)
    )
    # Put the field in the same unit gauge the rigid solve uses, so the two
    # results are interchangeable everywhere downstream.
    # A translation field is the same everywhere, so its reference point is pure
    # gauge.  Take the weighted centroid of the observations that were actually
    # used, which is determined by the data rather than by whichever sample
    # happened to be first.
    weight_total = weighted.sum().clamp_min(_EPS64)
    reference_point = (weighted[:, :, None] * positions).sum(dim=(0, 1)) / weight_total
    mode, scaled_speeds = normalize_rigid_mode(
        reference_point,
        direction,
        torch.zeros_like(direction),
        speeds,
    )
    mode.valid = bool(
        int(valid.any(dim=0).sum()) >= int(min_valid_cells)
        and torch.isfinite(translation_rms)
        and torch.isfinite(torch.as_tensor(speeds)).all()
        and float(torch.linalg.norm(direction)) > _EPS64
    )
    mode.diagnostics.update(
        {
            "constrained": "angular_mode_is_exactly_zero",
            "translation_rms": float(translation_rms),
            "static_rms": float(static_rms),
            "valid_cells": int(valid.any(dim=0).sum()),
            "valid_observations": int(valid.sum()),
        }
    )
    return mode, (speeds if scaled_speeds is None else scaled_speeds), {
        "translation_rms": float(translation_rms),
        "static_rms": float(static_rms),
        "direction": direction.tolist(),
    }


def normalize_rigid_mode(
    reference_point: torch.Tensor,
    reference_velocity: torch.Tensor,
    angular_mode: torch.Tensor,
    speeds: torch.Tensor | None = None,
    *,
    angular_eps: float = 1.0e-4,
    diagnostics: dict[str, Any] | None = None,
) -> tuple[RigidKillingMode, torch.Tensor | None]:
    """Put one generator in unit gauge and record its screw geometry.

    ``mode_type`` names the *gauge* this normalization chose, not the joint
    type: "rotational" means the generator was divided by ``|omega|`` and the
    scalar speeds carry rad/s, "translational" means it was divided by ``|v|``
    and the speeds carry m/s.  Because the alternating solver re-normalizes an
    already-unit generator on every iteration, ``angular_eps`` sees ``|omega|
    == 1`` from the second iteration onward and the gauge is then always
    rotational.  That is correct for integration but carries no information
    about the physical joint; use ``classify_joint_type`` for that.
    """
    reference = _float64_cpu(reference_point).reshape(3)
    linear = _float64_cpu(reference_velocity).reshape(3)
    angular = _float64_cpu(angular_mode).reshape(3)
    geometry = screw_geometry(linear, angular)
    angular_norm = geometry["angular_norm"]
    linear_norm = geometry["linear_norm"]
    if angular_norm > float(angular_eps):
        scale, mode_type = angular_norm, "rotational"
        linear, angular = linear / scale, angular / scale
    elif linear_norm > _EPS64:
        scale, mode_type = linear_norm, "translational"
        linear, angular = linear / scale, torch.zeros_like(angular)
    else:
        scale, mode_type = 0.0, "translational"
    scaled_speeds = None if speeds is None else _float64_cpu(speeds) * scale
    details = dict(diagnostics or {})
    details.update(
        {
            "angular_norm_at_normalization": angular_norm,
            "linear_norm_at_normalization": linear_norm,
            "screw_lever_arm": geometry["lever_arm"],
            "screw_pitch": geometry["pitch"],
        }
    )
    valid = bool(scale > _EPS64 and torch.isfinite(linear).all() and torch.isfinite(angular).all())
    return (
        RigidKillingMode(
            reference_point=reference,
            reference_velocity=linear,
            angular_mode=angular,
            mode_type=mode_type,
            normalization_scale=scale,
            valid=valid,
            diagnostics=details,
        ),
        scaled_speeds,
    )


def solve_scalar_speeds(
    mode: RigidKillingMode,
    cell_positions: torch.Tensor,
    observed_velocity: torch.Tensor,
    *,
    weights: torch.Tensor | None = None,
    valid_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    velocity = _float64_cpu(observed_velocity)
    positions = _float64_cpu(cell_positions)
    if velocity.ndim == 2:
        velocity = velocity[None]
    if positions.ndim == 2:
        positions = positions[None].expand(velocity.shape[0], -1, -1)
    if positions.shape != velocity.shape:
        raise ValueError("positions and velocities must be [K,C,3] or shared [C,3]")
    weight = torch.ones(velocity.shape[:2], dtype=torch.float64)
    if weights is not None:
        weight = _float64_cpu(weights)
    valid = torch.isfinite(positions).all(dim=-1) & torch.isfinite(velocity).all(dim=-1) & (weight > 0)
    if valid_mask is not None:
        valid &= torch.as_tensor(valid_mask).detach().cpu().bool()
    phi = predict_rigid_cell_velocity(mode, positions, 1.0)
    effective = weight * valid.to(torch.float64)
    numerator = (effective[:, :, None] * velocity * phi).sum(dim=(1, 2))
    denominator = (effective[:, :, None] * phi.square()).sum(dim=(1, 2)).clamp_min(_EPS64)
    return numerator / denominator


def solve_rigid_killing_mode_all_intervals(
    cell_positions: torch.Tensor,
    observed_velocity: torch.Tensor,
    *,
    weights: torch.Tensor | None = None,
    valid_mask: torch.Tensor | None = None,
    reference_point: torch.Tensor | None = None,
    max_iters: int = 20,
    tolerance: float = 1.0e-6,
    irls_iters: int = 5,
    huber_delta: float = 0.05,
    min_valid_cells: int = 8,
    angular_eps: float = 1.0e-4,
) -> tuple[RigidKillingMode, torch.Tensor, dict[str, Any]]:
    """Fit one spatial rigid generator and one signed scalar speed per interval."""
    velocity = _float64_cpu(observed_velocity)
    positions = _float64_cpu(cell_positions)
    if velocity.ndim != 3 or velocity.shape[-1] != 3:
        raise ValueError("observed_velocity must be [K,C,3]")
    if positions.ndim == 2:
        positions = positions[None].expand(velocity.shape[0], -1, -1).clone()
    if positions.shape != velocity.shape:
        raise ValueError("cell_positions must be [C,3] or [K,C,3]")
    base_weight = torch.ones(velocity.shape[:2], dtype=torch.float64)
    if weights is not None:
        base_weight = _float64_cpu(weights)
    valid = torch.isfinite(positions).all(dim=-1) & torch.isfinite(velocity).all(dim=-1) & (base_weight > 0)
    if valid_mask is not None:
        valid &= torch.as_tensor(valid_mask).detach().cpu().bool()
    if reference_point is None:
        support = (base_weight * valid).sum(dim=0)
        reference = positions[0, torch.argmax(support)].clone()
    else:
        reference = _float64_cpu(reference_point).reshape(3)

    interval_fits = [
        solve_rigid_field_single_interval(
            positions[k],
            velocity[k],
            weights=base_weight[k],
            valid_mask=valid[k],
            reference_point=reference,
            irls_iters=irls_iters,
            huber_delta=huber_delta,
            min_valid_cells=min_valid_cells,
        )
        for k in range(velocity.shape[0])
    ]
    candidates = [
        (k, fit)
        for k, fit in enumerate(interval_fits)
        if bool(fit.diagnostics.get("valid", False))
        and torch.is_tensor(fit.diagnostics.get("generator"))
    ]
    if not candidates:
        invalid, _ = normalize_rigid_mode(reference, torch.zeros(3), torch.zeros(3))
        invalid.valid = False
        invalid.diagnostics["reason"] = "no_valid_interval_fit"
        return invalid, torch.zeros(velocity.shape[0], dtype=torch.float64), {
            "converged": False,
            "intervals": [fit.diagnostics for fit in interval_fits],
        }
    best_index, best = max(
        candidates,
        key=lambda item: float((base_weight[item[0]] * valid[item[0]]).sum()),
    )
    generator = best.diagnostics["generator"].clone()
    mode, _ = normalize_rigid_mode(
        reference,
        generator[:3],
        generator[3:],
        angular_eps=angular_eps,
    )
    # The alternating loop below re-normalizes an already-unit generator, so its
    # own "at_normalization" magnitudes are all 1.  Capture the raw physical
    # magnitudes of the seed generator here, once, and never overwrite them.
    raw_screw = screw_geometry(generator[:3], generator[3:])
    speeds = solve_scalar_speeds(
        mode, positions, velocity, weights=base_weight, valid_mask=valid
    )
    robust_weight = base_weight.clone()
    history = []
    converged = False
    previous_generator = torch.cat((mode.reference_velocity, mode.angular_mode))
    previous_speeds = speeds.clone()
    previous_loss = float("inf")
    design = torch.cat(
        (
            torch.eye(3, dtype=torch.float64).reshape(1, 1, 3, 3).expand(
                positions.shape[0], positions.shape[1], -1, -1
            ),
            -_cross_matrix(positions - reference),
        ),
        dim=-1,
    )
    solve_diagnostics: dict[str, float] = {}

    for iteration in range(max(1, int(max_iters))):
        effective = robust_weight * valid.to(torch.float64)
        scaled_weight = effective * speeds[:, None].square()
        normal = torch.einsum("kc,kcai,kcaj->ij", scaled_weight, design, design)
        rhs = torch.einsum("kc,k,kcai,kca->i", effective, speeds, design, velocity)
        eigenvalues = torch.linalg.eigvalsh(normal)
        largest, smallest = float(eigenvalues[-1]), float(eigenvalues[0])
        ridge = 0.0
        if smallest <= max(largest, 1.0) * 1.0e-12:
            ridge = max(largest, 1.0) * 1.0e-10
        generator = torch.linalg.solve(
            normal + ridge * torch.eye(6, dtype=torch.float64), rhs
        )
        candidate, _ = normalize_rigid_mode(
            reference,
            generator[:3],
            generator[3:],
            angular_eps=angular_eps,
        )
        speeds = solve_scalar_speeds(
            candidate, positions, velocity, weights=effective, valid_mask=valid
        )
        normalized_generator = torch.cat(
            (candidate.reference_velocity, candidate.angular_mode)
        )
        if torch.dot(normalized_generator, previous_generator) < 0:
            candidate.reference_velocity.neg_()
            candidate.angular_mode.neg_()
            speeds.neg_()
            normalized_generator.neg_()
        residual = rigid_residuals(candidate, positions, velocity, speeds)
        cell_robust = _huber_weights(residual, huber_delta)
        robust_weight = base_weight * cell_robust
        valid_weight = base_weight * valid.to(torch.float64)
        loss = float(
            (valid_weight * residual.square()).sum()
            / valid_weight.sum().clamp_min(_EPS64)
        )
        mode_change = float(
            torch.linalg.norm(normalized_generator - previous_generator)
            / torch.linalg.norm(previous_generator).clamp_min(_EPS64)
        )
        speed_change = float(
            torch.linalg.norm(speeds - previous_speeds)
            / torch.linalg.norm(previous_speeds).clamp_min(_EPS64)
        )
        residual_change = abs(previous_loss - loss) / max(abs(previous_loss), 1.0) if torch.isfinite(torch.tensor(previous_loss)) else float("inf")
        solve_diagnostics = {
            "condition_number": largest / max(smallest, _EPS64),
            "conditioning_ratio": smallest / max(largest, _EPS64),
            "ridge": ridge,
        }
        history.append(
            {
                "iteration": iteration + 1,
                "weighted_residual": loss,
                "mode_change": mode_change,
                "speed_change": speed_change,
                "residual_change": residual_change,
                **solve_diagnostics,
            }
        )
        mode = candidate
        previous_generator = normalized_generator.clone()
        previous_speeds = speeds.clone()
        previous_loss = loss
        if mode_change < tolerance and speed_change < tolerance and residual_change < tolerance:
            converged = True
            break

    largest_speed = int(torch.argmax(speeds.abs())) if speeds.numel() else 0
    if speeds.numel() and speeds[largest_speed] < 0:
        mode.reference_velocity.neg_()
        mode.angular_mode.neg_()
        speeds.neg_()
    residual = rigid_residuals(mode, positions, velocity, speeds)
    valid_weight = base_weight * valid.to(torch.float64)
    rigid_rms = torch.sqrt(
        (valid_weight * residual.square()).sum()
        / valid_weight.sum().clamp_min(_EPS64)
    )
    static_rms = torch.sqrt(
        (valid_weight[:, :, None] * velocity.square()).sum()
        / valid_weight.sum().clamp_min(_EPS64)
    )
    mode.valid = bool(
        mode.valid
        and int(valid.any(dim=0).sum()) >= int(min_valid_cells)
        and float(rigid_rms) < float(static_rms)
        and torch.isfinite(speeds).all()
    )
    mode.diagnostics.update(
        {
            "raw_angular_norm": raw_screw["angular_norm"],
            "raw_linear_norm": raw_screw["linear_norm"],
            "raw_screw_lever_arm": raw_screw["lever_arm"],
            "raw_screw_pitch": raw_screw["pitch"],
            "initial_interval": int(best_index),
            "valid_observations": int(valid.sum()),
            "valid_cells": int(valid.any(dim=0).sum()),
            "rigid_rms": float(rigid_rms),
            "static_rms": float(static_rms),
            "converged": converged,
            "iterations": len(history),
            **solve_diagnostics,
        }
    )
    interval_diagnostics = []
    for k in range(velocity.shape[0]):
        mask = valid[k]
        wk = base_weight[k] * mask.to(torch.float64)
        interval_rigid = torch.sqrt(
            (wk * residual[k].square()).sum() / wk.sum().clamp_min(_EPS64)
        )
        interval_static = torch.sqrt(
            (wk[:, None] * velocity[k].square()).sum() / wk.sum().clamp_min(_EPS64)
        )
        interval_diagnostics.append(
            {
                "interval": k,
                "speed": float(speeds[k]),
                "valid_cells": int(mask.sum()),
                "rigid_rms": float(interval_rigid),
                "static_rms": float(interval_static),
            }
        )
    diagnostics = {
        "converged": converged,
        "history": history,
        "intervals": interval_diagnostics,
        "single_interval_initializers": [
            {key: value for key, value in fit.diagnostics.items() if key != "generator"}
            for fit in interval_fits
        ],
    }
    return mode, speeds, diagnostics


def re_reference_rigid_mode(
    mode: RigidKillingMode, new_reference_point: torch.Tensor
) -> RigidKillingMode:
    """Represent the identical field around a different coordinate gauge."""
    reference = torch.as_tensor(new_reference_point).to(mode.reference_point).reshape(3)
    displacement = reference - mode.reference_point
    reference_velocity = mode.reference_velocity + torch.cross(
        mode.angular_mode, displacement, dim=-1
    )
    return RigidKillingMode(
        reference_point=reference,
        reference_velocity=reference_velocity,
        angular_mode=mode.angular_mode.clone(),
        mode_type=mode.mode_type,
        normalization_scale=mode.normalization_scale,
        valid=mode.valid,
        diagnostics=dict(mode.diagnostics),
    )


def _rigid_transform(
    mode: RigidKillingMode, delta_q: torch.Tensor, *, device, dtype
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    linear = mode.reference_velocity.to(device=device, dtype=dtype)
    angular = (
        torch.zeros(3, device=device, dtype=dtype)
        if mode.mode_type == "translational"
        else mode.angular_mode.to(device=device, dtype=dtype)
    )
    rotation, translation, quaternion = se3_exp(angular, linear, delta_q)
    return rotation, translation, quaternion


def _left_multiply_quaternion(
    quaternion_wxyz: torch.Tensor, angular_mode: torch.Tensor, delta_q: torch.Tensor
) -> torch.Tensor:
    angular = angular_mode.to(quaternion_wxyz)
    norm = torch.linalg.norm(angular)
    axis = angular / norm.clamp_min(1.0e-12)
    increment = axis_angle_to_quaternion(axis, delta_q * norm)
    return apply_rotation_to_gaussian_quaternions(quaternion_wxyz, increment)


def integrate_rigid_mode_exact(
    points: torch.Tensor,
    mode: RigidKillingMode,
    delta_q: torch.Tensor | float,
    *,
    orientations: torch.Tensor | None = None,
    effective_dynamic_mask: torch.Tensor | None = None,
) -> dict[str, torch.Tensor | None]:
    """Apply the exact exponential of one rigid spatial generator."""
    if not mode.valid:
        raise RuntimeError("Cannot integrate an invalid rigid Killing mode")
    xyz = torch.as_tensor(points)
    increment = torch.as_tensor(delta_q, device=xyz.device, dtype=xyz.dtype).reshape(())
    rotation, translation, rotation_quaternion = _rigid_transform(
        mode, increment, device=xyz.device, dtype=xyz.dtype
    )
    reference = mode.reference_point.to(xyz)
    moved = reference + (xyz - reference) @ rotation.T + translation
    mask = None
    if effective_dynamic_mask is not None:
        mask = torch.as_tensor(effective_dynamic_mask, device=xyz.device).bool().reshape(-1)
        if mask.shape[0] != xyz.shape[0]:
            raise ValueError("effective_dynamic_mask must match points")
        moved = torch.where(mask[:, None], moved, xyz)
    updated_orientation = None
    if orientations is not None:
        orientation = torch.nn.functional.normalize(
            torch.as_tensor(orientations), dim=-1
        )
        if mode.mode_type == "rotational":
            updated_orientation = _left_multiply_quaternion(
                orientation, mode.angular_mode, increment
            )
        else:
            updated_orientation = orientation.clone()
        if mask is not None:
            updated_orientation = torch.where(
                mask[:, None], updated_orientation, orientation
            )
    return {
        "positions": moved,
        "orientations": updated_orientation,
        "rotation_matrix": rotation,
        "rotation_quaternion": rotation_quaternion,
        "translation": translation,
    }


@torch.no_grad()
def project_rigid_flow_per_gaussian(
    canonical_points: torch.Tensor,
    mode: RigidKillingMode,
    interval_speeds: torch.Tensor,
    frame_times: torch.Tensor | list[float],
    cameras_by_frame,
) -> dict[str, torch.Tensor]:
    """Project the exact rigid trajectory for every Gaussian/camera/interval."""
    from utils.motion_correspondence import project_world_points

    points = torch.as_tensor(canonical_points)
    times = torch.as_tensor(frame_times, dtype=torch.float64).flatten()
    speeds = torch.as_tensor(interval_speeds, dtype=torch.float64).flatten()
    if times.numel() != speeds.numel() + 1:
        raise ValueError("frame_times must have one more entry than interval_speeds")
    if len(cameras_by_frame) != times.numel():
        raise ValueError("cameras_by_frame must match frame_times")
    keys = sorted(set(cameras_by_frame[0]).intersection(*[set(item) for item in cameras_by_frame[1:]]))
    cumulative = torch.cat((torch.zeros(1, dtype=torch.float64), torch.cumsum(speeds * torch.diff(times), dim=0)))
    positions = [
        integrate_rigid_mode_exact(points, mode, increment)["positions"]
        for increment in cumulative
    ]
    flow = points.new_zeros((points.shape[0], len(keys), speeds.numel(), 2))
    source_depth = points.new_zeros((points.shape[0], len(keys), speeds.numel()))
    target_depth = torch.zeros_like(source_depth)
    valid = torch.zeros_like(source_depth, dtype=torch.bool)
    for interval in range(speeds.numel()):
        for camera_index, key in enumerate(keys):
            source_pixel, z0, valid0 = project_world_points(positions[interval], cameras_by_frame[interval][key])
            target_pixel, z1, valid1 = project_world_points(positions[interval + 1], cameras_by_frame[interval + 1][key])
            flow[:, camera_index, interval] = target_pixel - source_pixel
            source_depth[:, camera_index, interval] = z0
            target_depth[:, camera_index, interval] = z1
            valid[:, camera_index, interval] = valid0 & valid1
    return {
        "flow": flow,
        "source_depth": source_depth,
        "target_depth": target_depth,
        "valid": valid,
        "camera_keys": keys,
        "positions": torch.stack(positions),
    }


@torch.no_grad()
def compute_unit_speed_projected_flow(
    canonical_points: torch.Tensor,
    mode: RigidKillingMode,
    interval_speeds: torch.Tensor,
    frame_times: torch.Tensor | list[float],
    cameras_by_frame,
) -> dict[str, torch.Tensor]:
    """Projected displacement for unit scalar generator speed at each source pose."""
    actual = project_rigid_flow_per_gaussian(
        canonical_points, mode, interval_speeds, frame_times, cameras_by_frame
    )
    from utils.motion_correspondence import project_world_points

    points = torch.as_tensor(canonical_points)
    times = torch.as_tensor(frame_times, dtype=torch.float64).flatten()
    speeds64 = torch.as_tensor(interval_speeds, dtype=torch.float64).flatten()
    cumulative = torch.cat((torch.zeros(1, dtype=torch.float64), torch.cumsum(speeds64 * torch.diff(times), dim=0)))
    unit = torch.zeros_like(actual["flow"])
    for interval in range(speeds64.numel()):
        unit_target = integrate_rigid_mode_exact(
            points, mode, cumulative[interval] + (times[interval + 1] - times[interval])
        )["positions"]
        for camera_index, key in enumerate(actual["camera_keys"]):
            source_pixel, _, _ = project_world_points(
                actual["positions"][interval], cameras_by_frame[interval][key]
            )
            target_pixel, _, _ = project_world_points(
                unit_target, cameras_by_frame[interval + 1][key]
            )
            unit[:, camera_index, interval] = target_pixel - source_pixel
    return {**actual, "unit_flow": unit}


def infer_observation_speed_scalar(*args, **kwargs):
    """Public rigid-motion facade for sequence scalar-speed diagnostics."""
    from utils.gaussian_sequence_assignment import infer_observation_speed_scalar as infer

    return infer(*args, **kwargs)


def solve_rigid_assignment(
    static_error: torch.Tensor,
    rigid_error: torch.Tensor,
    *,
    weights: torch.Tensor | None = None,
    valid_mask: torch.Tensor | None = None,
    prior_dynamic_probability: torch.Tensor | None = None,
    improvement_margin: float = 0.05,
    temperature: float = 0.12,
    prior_weight: float = 0.25,
) -> dict[str, torch.Tensor]:
    """Compete explicit static and rigid-moving hypotheses per material cell."""
    static = _float64_cpu(static_error)
    rigid = _float64_cpu(rigid_error)
    if static.shape != rigid.shape:
        raise ValueError("static_error and rigid_error must have equal shapes")
    if static.ndim == 1:
        static, rigid = static[None], rigid[None]
    weight = torch.ones_like(static)
    if weights is not None:
        weight = _float64_cpu(weights)
    valid = torch.isfinite(static) & torch.isfinite(rigid) & (weight > 0)
    if valid_mask is not None:
        valid &= torch.as_tensor(valid_mask).detach().cpu().bool()
    effective = weight * valid.to(torch.float64)
    denominator = effective.sum(dim=0)
    improvement_per_interval = static - rigid
    improvement = (effective * improvement_per_interval).sum(dim=0) / denominator.clamp_min(_EPS64)
    data_logit = (improvement - float(improvement_margin)) / max(
        float(temperature), 1.0e-6
    )
    if prior_dynamic_probability is not None:
        prior = _float64_cpu(prior_dynamic_probability).reshape(-1).clamp(
            1.0e-5, 1.0 - 1.0e-5
        )
        data_logit = data_logit + float(prior_weight) * torch.logit(prior)
    probability = torch.sigmoid(data_logit)
    has_measurement = denominator > 0
    if prior_dynamic_probability is not None:
        probability = torch.where(has_measurement, probability, prior)
    else:
        probability = torch.where(has_measurement, probability, torch.zeros_like(probability))
    effective_mask = has_measurement & (improvement > float(improvement_margin)) & (probability >= 0.5)
    return {
        "dynamic_probability": probability,
        "effective_dynamic_mask": effective_mask,
        "rigid_vs_static_improvement": improvement,
        "valid_interval_count": valid.sum(dim=0),
        "improvement_per_interval": improvement_per_interval,
    }
