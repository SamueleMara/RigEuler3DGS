"""Per-Gaussian observations for the Lagrangian rigid-Killing comparison."""

from __future__ import annotations

import torch

from utils.dynamic_loss_utils import (
    build_lagrangian_knn_pair_graph,
    decompose_velocity_gradient,
    fit_local_velocity_gradient,
)
from utils.rigid_killing_motion import predict_rigid_cell_velocity


@torch.no_grad()
def compute_gaussian_speed_observations(
    positions: torch.Tensor,
    observed_velocity: torch.Tensor,
    normal_matrix: torch.Tensor,
    valid_mask: torch.Tensor,
    mode,
    global_rigid_speed: torch.Tensor,
    *,
    epsilon: float = 1.0e-10,
) -> dict[str, torch.Tensor]:
    """Project each Gaussian velocity system onto the shared Killing mode."""
    xyz = torch.as_tensor(positions).double()
    velocity = torch.as_tensor(observed_velocity).double()
    normal = torch.as_tensor(normal_matrix).double()
    valid = torch.as_tensor(valid_mask).bool()
    speed = torch.as_tensor(global_rigid_speed).double().reshape(-1)
    if xyz.shape != velocity.shape or xyz.ndim != 3 or xyz.shape[-1] != 3:
        raise ValueError("positions and observed_velocity must both be [K,N,3]")
    if normal.shape != (*xyz.shape[:2], 3, 3) or valid.shape != xyz.shape[:2]:
        raise ValueError("normal_matrix/valid_mask must match [K,N] supports")
    if speed.numel() != xyz.shape[0]:
        raise ValueError("global_rigid_speed must have one value per interval")

    phi = predict_rigid_cell_velocity(mode, xyz, 1.0).double()
    normal_phi = torch.einsum("knij,knj->kni", normal, phi)
    denominator = (phi * normal_phi).sum(dim=-1)
    numerator = torch.einsum("kni,knij,knj->kn", phi, normal, velocity)
    informative = valid & torch.isfinite(numerator) & (denominator > float(epsilon))
    observed = torch.where(
        informative,
        numerator / denominator.clamp_min(float(epsilon)),
        torch.zeros_like(numerator),
    )
    perpendicular = torch.linalg.norm(
        velocity - observed[..., None] * phi, dim=-1
    )
    perpendicular = torch.where(
        informative, perpendicular, torch.zeros_like(perpendicular)
    )
    absolute_error = torch.where(
        informative,
        (observed - speed[:, None]).abs(),
        torch.zeros_like(observed),
    )
    count = informative.sum(dim=0)
    weight = informative.to(torch.float64)
    denominator_count = count.clamp_min(1).to(torch.float64)
    speed_rmse = torch.sqrt(
        ((observed - speed[:, None]).square() * weight).sum(dim=0)
        / denominator_count
    )
    relative_error = (
        (absolute_error * weight).sum(dim=0) / denominator_count
    ) / (
        (speed[:, None].abs() * weight).sum(dim=0) / denominator_count
    ).clamp_min(float(epsilon))
    median_absolute_error = torch.zeros(
        xyz.shape[1], device=xyz.device, dtype=torch.float64
    )
    for gaussian_index in torch.nonzero(count, as_tuple=False).flatten().tolist():
        median_absolute_error[gaussian_index] = absolute_error[
            informative[:, gaussian_index], gaussian_index
        ].median()
    sign_agreement = (
        ((observed * speed[:, None]) >= 0).to(torch.float64) * weight
    ).sum(dim=0) / denominator_count
    observed_mean = (observed * weight).sum(dim=0) / denominator_count
    global_mean = (speed[:, None] * weight).sum(dim=0) / denominator_count
    observed_centered = (observed - observed_mean[None]) * weight
    global_centered = (speed[:, None] - global_mean[None]) * weight
    covariance = (observed_centered * global_centered).sum(dim=0)
    correlation = covariance / torch.sqrt(
        observed_centered.square().sum(dim=0)
        * global_centered.square().sum(dim=0)
    ).clamp_min(float(epsilon))
    correlation = torch.where(count >= 2, correlation, torch.zeros_like(correlation))
    return {
        "observed_gaussian_speed": observed,
        "global_rigid_speed": speed,
        "valid": informative,
        "perpendicular_velocity_residual": perpendicular,
        "speed_absolute_error": absolute_error,
        "informative_interval_count": count,
        "informative_interval_fraction": count.to(torch.float64) / float(xyz.shape[0]),
        "speed_correlation": correlation,
        "speed_rmse": speed_rmse,
        "speed_relative_error": relative_error,
        "speed_sign_agreement": sign_agreement,
        "speed_median_abs_error": median_absolute_error,
    }


@torch.no_grad()
def compute_lagrangian_gradient_diagnostics(
    canonical_xyz: torch.Tensor,
    mode,
    interval_speeds: torch.Tensor,
    moving_mask: torch.Tensor,
    *,
    neighbors_per_gaussian: int = 12,
    max_pairs: int = 262144,
    min_neighbors: int = 3,
    relative_ridge: float = 1.0e-4,
) -> dict[str, torch.Tensor]:
    """Fit material-graph gradients without adding local velocity degrees of freedom."""
    xyz = torch.as_tensor(canonical_xyz)
    moving = torch.as_tensor(moving_mask, device=xyz.device).bool()
    speeds = torch.as_tensor(interval_speeds, device=xyz.device, dtype=xyz.dtype)
    moving_ids = torch.nonzero(moving, as_tuple=False).flatten()
    moving_xyz = xyz.index_select(0, moving_ids)
    graph = build_lagrangian_knn_pair_graph(
        moving_xyz,
        neighbors_per_point=int(neighbors_per_gaussian),
        max_pairs=int(max_pairs),
    )
    pairs = (
        moving_ids.index_select(0, graph.pairs.reshape(-1)).reshape(-1, 2)
        if graph.pairs.numel()
        else graph.pairs
    )
    moving_unit_velocity = predict_rigid_cell_velocity(
        mode, moving_xyz, 1.0
    ).to(xyz)
    local_gradient, counts, weight_sum = fit_local_velocity_gradient(
        moving_xyz,
        moving_unit_velocity,
        graph,
        min_neighbors=int(min_neighbors),
        relative_ridge=float(relative_ridge),
    )
    unit_gradient = xyz.new_zeros((xyz.shape[0], 3, 3)).index_copy(
        0, moving_ids, local_gradient
    )
    valid = torch.zeros_like(moving)
    valid[moving_ids] = (counts >= int(min_neighbors)) & (weight_sum > 0)
    gradient = speeds[:, None, None, None] * unit_gradient[None]
    deformation = 0.5 * (gradient + gradient.transpose(-1, -2))
    unit_deformation, _, unit_omega = decompose_velocity_gradient(unit_gradient)
    del unit_deformation
    omega = speeds[:, None, None] * unit_omega[None]
    omega_variance = torch.zeros_like(speeds)
    if bool(valid.any()):
        centered = omega[:, valid] - omega[:, valid].mean(dim=1, keepdim=True)
        omega_variance = centered.square().sum(dim=-1).mean(dim=-1)

    rigidity_residual = torch.zeros_like(speeds)
    if pairs.numel():
        selected = pairs
        dx = xyz[selected[:, 1]] - xyz[selected[:, 0]]
        unit_velocity = predict_rigid_cell_velocity(mode, xyz, 1.0).to(xyz)
        dv = unit_velocity[selected[:, 1]] - unit_velocity[selected[:, 0]]
        unit_edge_residual = (dx * dv).sum(dim=-1)
        rigidity_residual = speeds.abs() * torch.sqrt(
            unit_edge_residual.square().mean()
        )
    return {
        "knn_pairs": pairs.detach().cpu(),
        "unit_gradient": unit_gradient.detach().cpu(),
        "gradient_valid": valid.detach().cpu(),
        "deformation_frobenius": torch.linalg.norm(
            deformation, dim=(-2, -1)
        ).detach().cpu(),
        "omega_spatial_variance": omega_variance.detach().cpu(),
        "rigidity_matrix_residual": rigidity_residual.detach().cpu(),
    }
