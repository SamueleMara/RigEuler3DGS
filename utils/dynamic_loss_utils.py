"""Minimal losses and temporal utilities for the unified dynamic pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Sequence

import torch


def get_camera_timestamp(camera, fps: float = 30.0, time_scale: float = 1.0):
    if hasattr(camera, "timestamp"):
        return float(camera.timestamp)
    frame = int(getattr(camera, "frame_id", 0))
    return float(frame) / max(float(fps), 1.0e-8) * float(time_scale)


def normalize_camera_timestamps(
    cameras: Sequence,
    fps: float = 30.0,
    time_scale: float = 1.0,
    time_mode: str = "seconds",
):
    if not cameras:
        return cameras
    raw = []
    for camera in cameras:
        if hasattr(camera, "raw_timestamp"):
            raw.append(float(camera.raw_timestamp))
        else:
            frame = int(getattr(camera, "frame_id", 0))
            raw.append(float(frame) / max(float(fps), 1.0e-8))
    lo = min(raw)
    if str(time_mode) == "normalized":
        raise ValueError(
            "Normalized motion time is non-physical. Use seconds so cell and "
            "joint speeds remain coupled to real video displacement."
        )
    if abs(float(time_scale) - 1.0) > 1.0e-8:
        raise ValueError(
            "--time_scale must be 1.0 for physical video-time integration"
        )
    # Rebase the first selected video frame to t=0 without changing any real
    # inter-frame delta. This also makes an explicitly cropped motion window a
    # valid canonical interval rather than integrating from an unobserved gap.
    values = [value - lo for value in raw]
    for camera, value in zip(cameras, values):
        camera.timestamp = float(value)
    return cameras



@dataclass
class PairGraph:
    """Local undirected Gaussian graph.

    ``pairs`` is [P,2] with unique source/destination Gaussian ids.  ``distance``
    stores the detached source-frame distance and is useful for diagnostics.
    """

    pairs: torch.Tensor
    distance: torch.Tensor

    @property
    def count(self) -> int:
        return int(self.pairs.shape[0])



def build_lagrangian_knn_pair_graph(
    positions: torch.Tensor,
    neighbors_per_point: int = 4,
    max_pairs: int = 65536,
    query_chunk_size: int = 512,
) -> PairGraph:
    """Build a fixed material-neighborhood graph without any spatial grid.

    Neighbors are chosen once from canonical Gaussian centers and then kept by
    Gaussian identity, which is the appropriate material graph for assignment
    regularization. Chunked cdist avoids an N x N matrix.
    """
    if positions.ndim != 2 or positions.shape[-1] != 3:
        raise ValueError("positions must be [N,3]")
    count = int(positions.shape[0])
    device = positions.device
    if count < 2:
        return PairGraph(
            pairs=torch.empty((0, 2), device=device, dtype=torch.long),
            distance=torch.empty((0,), device=device, dtype=positions.dtype),
        )
    k = min(max(1, int(neighbors_per_point)), count - 1)
    pair_budget = max(1, int(max_pairs))
    chunk = max(1, int(query_chunk_size))
    pair_chunks = []
    for start in range(0, count, chunk):
        end = min(count, start + chunk)
        distance = torch.cdist(positions[start:end].detach(), positions.detach())
        rows = torch.arange(end - start, device=device)
        cols = torch.arange(start, end, device=device)
        distance[rows, cols] = float("inf")
        destination = torch.topk(distance, k=k, largest=False).indices
        source = torch.arange(start, end, device=device)[:, None].expand_as(destination)
        pair_chunks.append(torch.stack((source.reshape(-1), destination.reshape(-1)), dim=-1))
        if sum(int(value.shape[0]) for value in pair_chunks) >= pair_budget * 2:
            # Enough candidates to fill the undirected budget after deduplication.
            break
    pairs = torch.cat(pair_chunks, dim=0)
    pairs = torch.sort(pairs, dim=-1).values
    pairs = pairs[pairs[:, 0] != pairs[:, 1]]
    pairs = torch.unique(pairs, dim=0)
    if pairs.shape[0] > pair_budget:
        pairs = pairs[:pair_budget]
    pair_distance = torch.linalg.norm(
        positions[pairs[:, 0]].detach() - positions[pairs[:, 1]].detach(), dim=-1
    )
    return PairGraph(pairs=pairs, distance=pair_distance)




@torch.no_grad()
def build_occupied_cell_knn_pair_graph(
    cell_centers: torch.Tensor,
    occupied_mask: torch.Tensor,
    neighbors_per_cell: int = 12,
    max_radius: float = 0.0,
    max_pairs: int = 65536,
) -> PairGraph:
    """Build a material-aware graph between OCCUPIED Eulerian control volumes.

    The fixed grid is only a control-volume discretization; empty cells are not
    material.  This graph therefore never differentiates an occupied velocity
    against an arbitrary empty-space velocity.  Neighbours are selected from
    occupied cell centres only and can optionally be radius limited.
    """
    if cell_centers.ndim != 2 or cell_centers.shape[-1] != 3:
        raise ValueError("cell_centers must be [C,3]")
    occupied = occupied_mask.detach().reshape(-1).bool()
    if occupied.numel() != cell_centers.shape[0]:
        raise ValueError("occupied_mask must match cell_centers")
    ids = torch.nonzero(occupied, as_tuple=False).reshape(-1)
    if ids.numel() < 2:
        return PairGraph(
            pairs=torch.empty((0, 2), device=cell_centers.device, dtype=torch.long),
            distance=torch.empty((0,), device=cell_centers.device, dtype=cell_centers.dtype),
        )
    xyz = cell_centers.detach().index_select(0, ids)
    distance = torch.cdist(xyz, xyz)
    distance.fill_diagonal_(float("inf"))
    k = min(max(1, int(neighbors_per_cell)), int(ids.numel()) - 1)
    values, local_dst = torch.topk(distance, k=k, largest=False, dim=1)
    local_src = torch.arange(ids.numel(), device=ids.device)[:, None].expand_as(local_dst)
    valid = torch.isfinite(values)
    if float(max_radius) > 0.0:
        valid &= values <= float(max_radius)
    src = ids.index_select(0, local_src[valid])
    dst = ids.index_select(0, local_dst[valid])
    if src.numel() == 0:
        return PairGraph(
            pairs=torch.empty((0, 2), device=cell_centers.device, dtype=torch.long),
            distance=torch.empty((0,), device=cell_centers.device, dtype=cell_centers.dtype),
        )
    pairs = torch.stack((src, dst), dim=-1)
    pairs = torch.sort(pairs, dim=-1).values
    pairs = pairs[pairs[:, 0] != pairs[:, 1]]
    pairs = torch.unique(pairs, dim=0)
    if pairs.shape[0] > int(max_pairs):
        pairs = pairs[: int(max_pairs)]
    pair_distance = torch.linalg.norm(
        cell_centers[pairs[:, 0]].detach() - cell_centers[pairs[:, 1]].detach(), dim=-1
    )
    return PairGraph(pairs=pairs, distance=pair_distance)





def fit_local_velocity_gradient(
    positions: torch.Tensor,
    velocities: torch.Tensor,
    pair_graph: PairGraph,
    *,
    pair_weights: Optional[torch.Tensor] = None,
    min_neighbors: int = 3,
    relative_ridge: float = 1.0e-4,
    epsilon: float = 1.0e-8,
):
    """Fit the Eulerian velocity-gradient tensor G = dv/dx at occupied cells.

    The fit is the first-order Taylor relation used in continuum kinematics:

        v_d - v_c ~= G_c (X_d - X_c).

    Only occupied-cell neighbours are present in ``pair_graph``.  The least-
    squares solve is differentiable with respect to the cell velocities; the
    graph geometry is fixed.  For surface-like support, the Tikhonov term picks
    the minimum-norm extension in weakly observed normal directions while still
    constraining the tangential deformation that is actually sampled.
    """
    if positions.shape != velocities.shape or positions.ndim != 2 or positions.shape[-1] != 3:
        raise ValueError("positions and velocities must both be [N,3]")
    n = int(positions.shape[0])
    src, dst = _directed_pairs(pair_graph)
    zero = velocities.sum() * 0.0
    if src.numel() == 0:
        return (
            velocities.new_zeros((n, 3, 3)) + zero,
            torch.zeros(n, device=positions.device, dtype=torch.long),
            velocities.new_zeros((n,)),
        )

    r = (positions[dst] - positions[src]).detach()          # [E,3]
    dv = velocities[dst] - velocities[src]                  # [E,3]
    if pair_weights is None:
        w = torch.ones(src.shape[0], device=positions.device, dtype=positions.dtype)
    else:
        edge_w = pair_weights.reshape(-1).to(device=positions.device, dtype=positions.dtype)
        if edge_w.numel() != pair_graph.pairs.shape[0]:
            raise ValueError("pair_weights must have one value per undirected edge")
        w = torch.cat((edge_w, edge_w), dim=0).clamp_min(0.0)

    # M = sum w r r^T ; C = sum w dv r^T ; G = C M^{-1}.
    rr = r[:, :, None] * r[:, None, :]
    dvr = dv[:, :, None] * r[:, None, :]
    M = velocities.new_zeros((n, 3, 3))
    C = velocities.new_zeros((n, 3, 3))
    M.index_add_(0, src, rr * w[:, None, None])
    C.index_add_(0, src, dvr * w[:, None, None])
    counts = torch.zeros(n, device=positions.device, dtype=torch.long)
    # Count only mechanically supported neighbours.  With moving-part/DINO
    # boundary weights, zero-weight static/cross-part edges must not make an
    # underconstrained local gradient look valid.
    counts.index_add_(0, src, (w > float(epsilon)).to(torch.long))
    weight_sum = velocities.new_zeros((n,))
    weight_sum.index_add_(0, src, w)

    valid = (counts >= max(1, int(min_neighbors))) & (weight_sum > float(epsilon))
    valid_ids = torch.nonzero(valid, as_tuple=False).reshape(-1)
    G = velocities.new_zeros((n, 3, 3)) + zero
    if valid_ids.numel():
        Mv = M.index_select(0, valid_ids)
        Cv = C.index_select(0, valid_ids)
        trace = Mv.diagonal(dim1=-2, dim2=-1).sum(dim=-1) / 3.0
        ridge = float(relative_ridge) * trace.clamp_min(float(epsilon)) + float(epsilon)
        eye = torch.eye(3, device=positions.device, dtype=positions.dtype)[None]
        Mr = Mv + ridge[:, None, None] * eye
        # Solve M^T G^T = C^T.  M is symmetric but this form keeps the algebra explicit.
        Gv = torch.linalg.solve(Mr.transpose(-1, -2), Cv.transpose(-1, -2)).transpose(-1, -2)
        G = G.index_copy(0, valid_ids, Gv)
    return G, counts, weight_sum


def decompose_velocity_gradient(gradient: torch.Tensor):
    """Return deformation tensor D, rigid-rotation tensor Omega and angular velocity.

    Code convention is G_ij = d v_i / d x_j.  For an exact rigid field
    v = b + omega x x, G = [omega]_x is skew-symmetric, therefore D=0 and
    omega = 0.5 curl(v).
    """
    if gradient.ndim != 3 or gradient.shape[-2:] != (3, 3):
        raise ValueError("gradient must be [N,3,3]")
    GT = gradient.transpose(-1, -2)
    deformation = 0.5 * (gradient + GT)
    rotation = 0.5 * (gradient - GT)
    omega = torch.stack((
        0.5 * (gradient[:, 2, 1] - gradient[:, 1, 2]),
        0.5 * (gradient[:, 0, 2] - gradient[:, 2, 0]),
        0.5 * (gradient[:, 1, 0] - gradient[:, 0, 1]),
    ), dim=-1)
    return deformation, rotation, omega


def _same_moving_part_pair_weights(
    probabilities: torch.Tensor,
    pair_graph: PairGraph,
    dino_features: torch.Tensor | None = None,
    *,
    dino_strength: float = 1.0,
) -> torch.Tensor:
    """Axis-free physical coupling for neighbours in the SAME MOVING part.

    Expert 0 is static and is deliberately excluded: static material is handled
    by the exact zero-velocity gate.  Experts 1..K are moving parts.  DINO only
    attenuates coupling across a descriptor boundary using the exact cosine
    distance; it never creates a motion label.
    """
    pairs = pair_graph.pairs
    if pairs.numel() == 0:
        return probabilities.new_empty((0,))
    probs = probabilities.detach()
    src, dst = pairs[:, 0], pairs[:, 1]
    if probs.shape[1] <= 1:
        same_moving = torch.zeros(src.shape[0], device=probs.device, dtype=probs.dtype)
    else:
        same_moving = (probs[src, 1:] * probs[dst, 1:]).sum(dim=-1).clamp(0.0, 1.0)
    if dino_features is None or dino_features.numel() == 0:
        material_affinity = torch.ones_like(same_moving)
    else:
        distance = dino_cosine_distance_pair_values(dino_features, pairs)
        material_affinity = dino_same_material_affinity_from_distance(distance).to(same_moving.dtype)
    strength = min(1.0, max(0.0, float(dino_strength)))
    dino_gate = (1.0 - strength) + strength * material_affinity
    return same_moving * dino_gate


def _same_moving_part_pair_weights_no_dino(
    probabilities: torch.Tensor,
    pair_graph: PairGraph,
    *,
    same_part_floor: float = 0.0,
) -> torch.Tensor:
    """Physical same-moving support that is deliberately independent of DINO.

    DINO is useful for deciding where a material boundary is plausible.  Once
    two neighbours are currently supported as the same moving material, direct
    velocity coherence should be a physical prior rather than an appearance
    prior, so this weight uses only the trainable material probabilities.
    """
    pairs = pair_graph.pairs
    if pairs.numel() == 0:
        return probabilities.new_empty((0,))
    probs = probabilities.detach()
    src, dst = pairs[:, 0], pairs[:, 1]
    if probs.shape[1] <= 1:
        return torch.zeros(src.shape[0], device=probs.device, dtype=probs.dtype)
    same_moving = (probs[src, 1:] * probs[dst, 1:]).sum(dim=-1).clamp(0.0, 1.0)
    moving_pair = ((1.0 - probs[src, 0]) * (1.0 - probs[dst, 0])).clamp(0.0, 1.0)
    floor = min(1.0, max(0.0, float(same_part_floor)))
    if floor <= 0.0:
        return same_moving
    # Early in training, uncertain part IDs should not completely delete the
    # smoothness path across a moving region.  The floor couples only material
    # that is currently moving, never static/background material.
    return torch.maximum(same_moving, floor * moving_pair)


def direct_velocity_spatial_coherence_loss(
    velocities: torch.Tensor,
    occupied_mask: torch.Tensor,
    probabilities: torch.Tensor,
    pair_graph: PairGraph,
    *,
    same_part_floor: float = 0.0,
    velocity_scale: float = 1.0,
    robust_delta: float = 0.25,
    epsilon: float = 1.0e-6,
):
    """Robust first-order same-part smoothness directly on neighbouring v.

    This is intentionally lower-level than the fitted-gradient losses:
    derivative-based D/G terms are good physics, but they can amplify noisy
    local velocity fits.  This term damps independent random arrows directly.
    The vector term controls direction and magnitude; the speed term gives an
    extra rotation-friendly magnitude coherence signal.
    """
    if velocities.ndim != 2 or velocities.shape[-1] != 3:
        raise ValueError("velocities must be [N,3]")
    if probabilities.ndim != 2 or probabilities.shape[0] != velocities.shape[0]:
        raise ValueError("probabilities must be [N,K+1] and match velocities")
    occupied = occupied_mask.detach().reshape(-1).bool()
    if occupied.numel() != velocities.shape[0]:
        raise ValueError("occupied_mask must match velocities")

    zero = velocities.sum() * 0.0
    pairs = pair_graph.pairs
    if pairs.numel() == 0:
        return zero, zero, velocities.new_empty((0,)), torch.zeros((0,), device=velocities.device, dtype=torch.bool)

    weights = _same_moving_part_pair_weights_no_dino(
        probabilities, pair_graph, same_part_floor=same_part_floor
    ).to(device=velocities.device, dtype=velocities.dtype)
    src, dst = pairs[:, 0], pairs[:, 1]
    valid = occupied[src] & occupied[dst] & (weights > float(epsilon))
    if not valid.any():
        return zero, zero, weights, valid

    scale = max(float(velocity_scale), float(epsilon))
    delta = max(float(robust_delta), float(epsilon))
    pw = weights[valid]
    dv = velocities[dst[valid]] - velocities[src[valid]]
    vector_residual = torch.linalg.norm(dv, dim=-1) / scale
    speed_residual = torch.abs(
        torch.linalg.norm(velocities[dst[valid]], dim=-1)
        - torch.linalg.norm(velocities[src[valid]], dim=-1)
    ) / scale

    def charbonnier(value):
        return torch.sqrt(value.square() + delta * delta) - delta

    vector_loss = (pw * charbonnier(vector_residual)).sum() / pw.sum().clamp_min(float(epsilon))
    speed_loss = (pw * charbonnier(speed_residual)).sum() / pw.sum().clamp_min(float(epsilon))
    return vector_loss, speed_loss, weights, valid


def eulerian_velocity_gradient_rigidity_loss(
    cell_centers: torch.Tensor,
    cell_velocity: torch.Tensor,
    occupied_mask: torch.Tensor,
    cell_probabilities: torch.Tensor,
    pair_graph: PairGraph,
    *,
    same_part_floor: float = 0.0,
    velocity_scale: float = 1.0,
    min_neighbors: int = 3,
    relative_ridge: float = 1.0e-4,
    static_speed_weight: float = 1.0,
    cell_dino_features: torch.Tensor | None = None,
    dino_boundary_strength: float = 1.0,
    velocity_spatial_robust_delta: float = 0.25,
    epsilon: float = 1.0e-6,
):
    """Generic axis-free rigidity + mandatory spatial coherence.

    First-stage motion must NOT assume or optimize a joint axis, angular
    velocity, translation twist, pivot, or screw parameters.  The optimized
    quantity is only the velocity representation (curl(A) cells or point v_i).

    Physics is therefore written entirely in v and its local gradient G=dv/dx:

      1. D = sym(G) -> 0                              local non-deformation
      2. G_i ~= G_j                                  spatial gradient smoothness
      3. dv_ij ~= 0.5*(G_i+G_j) dx_ij                local affine consistency

    Terms 2-3 are evaluated only for neighbours likely belonging to the same
    MOVING material part, with DINO used solely as a continuous boundary gate.
    They are axis-free and preserve a generic affine velocity field; together
    with D->0 that includes translation and arbitrary rigid rotation without
    ever fitting omega/b/axis during Phase 1.

    omega=0.5 curl(v) is returned only as a derived diagnostic/orientation cue;
    it is not spatially optimized by this loss.
    """
    occupied = occupied_mask.detach().reshape(-1).bool()
    if cell_velocity.shape != cell_centers.shape:
        raise ValueError("cell_centers and cell_velocity must both be [C,3]")
    if cell_probabilities.ndim != 2 or cell_probabilities.shape[0] != cell_centers.shape[0]:
        raise ValueError("cell_probabilities must be [C,K+1]")
    zero = cell_velocity.sum() * 0.0
    scale = max(float(velocity_scale), float(epsilon))

    pair_weight_motion = _same_moving_part_pair_weights(
        cell_probabilities, pair_graph, cell_dino_features,
        dino_strength=float(dino_boundary_strength),
    )
    G, counts, weight_sum = fit_local_velocity_gradient(
        cell_centers, cell_velocity, pair_graph,
        pair_weights=pair_weight_motion if pair_weight_motion.numel() else None,
        min_neighbors=int(min_neighbors), relative_ridge=float(relative_ridge), epsilon=float(epsilon),
    )
    D, Omega, omega = decompose_velocity_gradient(G)
    valid = occupied & (counts >= max(1, int(min_neighbors))) & (weight_sum > float(epsilon))

    # Local rigidity: no deformation.  No cell spacing or dt attenuation.
    strain_per = D.square().sum(dim=(-2, -1))
    strain_loss = strain_per[valid].mean() if valid.any() else zero

    pairs = pair_graph.pairs
    spatial_gradient_coherence = zero
    local_affine_coherence = zero
    omega_coherence_diag = zero
    if pairs.numel():
        src, dst = pairs[:, 0], pairs[:, 1]
        pair_valid = valid[src] & valid[dst] & (pair_weight_motion > float(epsilon))
        if pair_valid.any():
            sv, dv_ids = src[pair_valid], dst[pair_valid]
            pw = pair_weight_motion[pair_valid]
            dx = (cell_centers[dv_ids] - cell_centers[sv]).detach()
            edge_length = torch.linalg.norm(dx, dim=-1).clamp_min(float(epsilon))

            # Mandatory spatial smoothness: neighbouring local gradients should
            # agree. This is a second-order smoothness of v and therefore does
            # NOT force neighbouring velocity vectors to be parallel/equal.
            dG = G[dv_ids] - G[sv]
            grad_cost = dG.square().sum(dim=(-2, -1))
            spatial_gradient_coherence = (pw * grad_cost).sum() / pw.sum().clamp_min(float(epsilon))

            # Generic local affine integrability. For any affine field, including
            # rigid translation/rotation, this residual is exactly zero.
            Gavg = 0.5 * (G[sv] + G[dv_ids])
            predicted_dv = torch.einsum('eij,ej->ei', Gavg, dx)
            measured_dv = cell_velocity[dv_ids] - cell_velocity[sv]
            affine_rate_residual = torch.linalg.norm(measured_dv - predicted_dv, dim=-1) / edge_length
            local_affine_coherence = (pw * affine_rate_residual.square()).sum() / pw.sum().clamp_min(float(epsilon))

            # Diagnostic only. Never used to optimize the field in v11.
            domega = torch.linalg.norm(omega[dv_ids] - omega[sv], dim=-1)
            omega_coherence_diag = (pw * domega.square()).sum() / pw.sum().clamp_min(float(epsilon))

    velocity_spatial, speed_spatial, velocity_spatial_weight, velocity_spatial_valid = direct_velocity_spatial_coherence_loss(
        cell_velocity, occupied, cell_probabilities, pair_graph,
        same_part_floor=float(same_part_floor),
        velocity_scale=scale,
        robust_delta=float(velocity_spatial_robust_delta),
        epsilon=float(epsilon),
    )

    static_prob = cell_probabilities[:, 0].detach().clamp(0.0, 1.0)
    static_weight_cells = static_prob * occupied.to(static_prob.dtype)
    static_speed = (
        static_weight_cells * (torch.linalg.norm(cell_velocity, dim=-1) / scale).square()
    ).sum() / static_weight_cells.sum().clamp_min(float(epsilon))

    return {
        "strain": strain_loss,
        "spatial_gradient_coherence": spatial_gradient_coherence,
        "local_affine_coherence": local_affine_coherence,
        "velocity_spatial_coherence": velocity_spatial,
        "speed_spatial_coherence": speed_spatial,
        "omega_coherence": omega_coherence_diag,  # diagnostic only
        "static_speed": float(static_speed_weight) * static_speed,
        "gradient": G,
        "deformation": D,
        "rotation": Omega,
        "omega": omega,
        "valid": valid,
        "neighbor_count": counts,
        "same_part_weight": pair_weight_motion,
        "velocity_spatial_weight": velocity_spatial_weight,
        "velocity_spatial_valid": velocity_spatial_valid,
        "local_rigid_residual": local_affine_coherence,  # legacy audit alias
        "pair_length": pair_graph.distance.detach().median().clamp_min(float(epsilon)) if pairs.numel() else cell_centers.new_tensor(1.0),
    }

def eulerian_assignment_from_kinematics_loss(
    cell_centers: torch.Tensor,
    cell_velocity: torch.Tensor,
    cell_probabilities: torch.Tensor,
    occupied_mask: torch.Tensor,
    pair_graph: PairGraph,
    gradient_state: Dict[str, torch.Tensor],
    *,
    observed_state: Optional[Dict[str, torch.Tensor]] = None,
    same_part_floor: float = 0.0,
    velocity_scale: float = 1.0,
    cell_dino_features: torch.Tensor | None = None,
    dino_boundary_strength: float = 1.0,
    flow_noise_px: float = 0.25,
    flow_model_margin_px: float = 0.05,
    flow_model_temperature_px: float = 0.12,
    flow_fit_max_px: float = 0.75,
    epsilon: float = 1.0e-6,
):
    """Assignment evidence without fitting omega/b/axis.

    Static-vs-moving evidence still comes from the multiview observed-flow model
    comparison.  Spatial part-boundary evidence uses only generic local velocity
    gradients and affine-consistency residuals, detached from the motion field.
    """
    occupied = occupied_mask.detach().reshape(-1).bool()
    probs = cell_probabilities
    zero = probs.sum() * 0.0
    scale = max(float(velocity_scale), float(epsilon))

    static_observed = zero
    dynamic_observed = zero
    if observed_state is not None and observed_state.get("velocity", None) is not None:
        conf = observed_state["confidence"].detach().to(cell_velocity).clamp_min(0.0)
        valid_obs = observed_state["valid"].detach().to(device=cell_velocity.device).bool() & occupied
        static_rms = observed_state.get("static_rms_px_max", observed_state.get("static_rms_px_mean", None))
        moving_rms = observed_state.get("reprojection_rms_px", None)
        if static_rms is not None and moving_rms is not None and valid_obs.any():
            static_rms = static_rms.detach().to(cell_velocity)
            moving_rms = moving_rms.detach().to(cell_velocity)
            sigma = max(float(flow_noise_px), 1.0e-4)
            margin = max(0.0, float(flow_model_margin_px))
            temp = max(float(flow_model_temperature_px), 1.0e-4)
            fit_scale = max(float(flow_fit_max_px), sigma)
            static_evidence = torch.exp(-torch.square(static_rms / sigma))
            improvement = torch.sigmoid((static_rms - moving_rms - margin) / temp)
            dynamic_evidence = improvement * (1.0 - static_evidence) * torch.exp(-torch.square(moving_rms / fit_scale))
            w = conf * valid_obs.to(conf.dtype)
            static_observed = (w * probs[:, 0] * dynamic_evidence).sum() / w.sum().clamp_min(float(epsilon))
            dynamic_observed = (w * (1.0 - probs[:, 0]) * static_evidence).sum() / w.sum().clamp_min(float(epsilon))
        elif valid_obs.any():
            target = observed_state["velocity"].detach().to(cell_velocity)
            speed2 = (torch.linalg.norm(target, dim=-1) / scale).square()
            w = conf * valid_obs.to(conf.dtype)
            static_observed = (w * probs[:, 0] * speed2).sum() / w.sum().clamp_min(float(epsilon))

    pairs = pair_graph.pairs
    boundary = zero
    if pairs.numel():
        src, dst = pairs[:, 0], pairs[:, 1]
        grad_valid = gradient_state["valid"].detach().bool()
        pair_valid = grad_valid[src] & grad_valid[dst]
        if pair_valid.any():
            src, dst = src[pair_valid], dst[pair_valid]
            G = gradient_state["gradient"].detach()
            dx = (cell_centers[dst] - cell_centers[src]).detach()
            edge_length = torch.linalg.norm(dx, dim=-1).clamp_min(float(epsilon))
            dv = (cell_velocity[dst] - cell_velocity[src]).detach()
            Gavg = 0.5 * (G[src] + G[dst])
            predicted = torch.einsum('eij,ej->ei', Gavg, dx)
            affine_cost = (torch.linalg.norm(dv - predicted, dim=-1) / edge_length).square()
            gradient_cost = (G[dst] - G[src]).square().sum(dim=(-2, -1))
            kinematic_cost = affine_cost + gradient_cost

            same_prob = (probs[src] * probs[dst]).sum(dim=-1)
            if cell_dino_features is not None and cell_dino_features.numel() > 0:
                pair_subset = torch.stack((src, dst), dim=-1)
                dino_distance = dino_cosine_distance_pair_values(cell_dino_features, pair_subset)
                dino_same = dino_same_material_affinity_from_distance(dino_distance).to(same_prob.dtype)
            else:
                dino_same = torch.ones_like(same_prob)
            strength = min(1.0, max(0.0, float(dino_boundary_strength)))
            required_same = strength * dino_same + (1.0 - strength) * same_prob
            floor = min(1.0, max(0.0, float(same_part_floor)))
            same_weight = floor + (1.0 - floor) * torch.maximum(same_prob, required_same)
            boundary = (same_weight * kinematic_cost).mean()

    return static_observed + dynamic_observed + boundary, {
        "assignment_static_observed": static_observed,
        "assignment_dynamic_observed": dynamic_observed,
        "assignment_kinematic_boundary": boundary,
    }

def _directed_pairs(pair_graph: PairGraph):
    pairs = pair_graph.pairs
    if pairs.numel() == 0:
        return pairs.new_empty((0,)), pairs.new_empty((0,))
    return (
        torch.cat((pairs[:, 0], pairs[:, 1]), dim=0),
        torch.cat((pairs[:, 1], pairs[:, 0]), dim=0),
    )



def integrate_world_angular_velocity_quaternion(
    quaternion_wxyz: torch.Tensor,
    angular_velocity: torch.Tensor,
    delta_time: float,
    epsilon: float = 1.0e-8,
) -> torch.Tensor:
    """Left-integrate a wxyz quaternion using WORLD-frame angular velocity.

    R maps local Gaussian coordinates to world coordinates and obeys
    dR/dt = [omega]_x R, therefore q_new = dq_world (*) q_old.
    """
    if quaternion_wxyz.ndim != 2 or quaternion_wxyz.shape[-1] != 4:
        raise ValueError("quaternion_wxyz must be [N,4]")
    if angular_velocity.shape != quaternion_wxyz[:, 1:].shape:
        raise ValueError("angular_velocity must be [N,3]")
    q = torch.nn.functional.normalize(quaternion_wxyz, dim=-1)
    omega_norm = torch.linalg.norm(angular_velocity, dim=-1)
    half_angle = 0.5 * float(delta_time) * omega_norm
    # sin(0.5*dt*|w|)/|w| with stable zero-speed limit 0.5*dt.
    scale = torch.where(
        omega_norm > float(epsilon),
        torch.sin(half_angle) / omega_norm.clamp_min(float(epsilon)),
        torch.full_like(omega_norm, 0.5 * float(delta_time)),
    )
    dq = torch.cat((torch.cos(half_angle)[:, None], angular_velocity * scale[:, None]), dim=-1)
    aw, ax, ay, az = dq.unbind(dim=-1)
    bw, bx, by, bz = q.unbind(dim=-1)
    out = torch.stack((
        aw*bw - ax*bx - ay*by - az*bz,
        aw*bx + ax*bw + ay*bz - az*by,
        aw*by - ax*bz + ay*bw + az*bx,
        aw*bz + ax*by - ay*bx + az*bw,
    ), dim=-1)
    return torch.nn.functional.normalize(out, dim=-1)


def rigid_killing_gradient_diagnostics(
    angular_mode: torch.Tensor, interval_speeds: torch.Tensor
) -> dict:
    """Analytic velocity-gradient diagnostics for a rigid Killing mode."""
    angular = torch.as_tensor(angular_mode)
    speeds = torch.as_tensor(interval_speeds, device=angular.device, dtype=angular.dtype)
    x, y, z = angular.unbind()
    zero = torch.zeros_like(x)
    skew = torch.stack(
        (zero, -z, y, z, zero, -x, -y, x, zero)
    ).reshape(3, 3)
    gradient = speeds[:, None, None] * skew
    deformation = 0.5 * (gradient + gradient.transpose(-1, -2))
    rotation = 0.5 * (gradient - gradient.transpose(-1, -2))
    omega = speeds[:, None] * angular
    return {
        "gradient": gradient,
        "deformation": deformation,
        "rotation": rotation,
        "omega": omega,
        "deformation_frobenius": torch.linalg.norm(deformation, dim=(-2, -1)),
        "omega_spatial_variance": torch.zeros_like(speeds),
    }



def assignment_spatial_coherence_loss(
    probabilities: torch.Tensor,
    pair_graph: PairGraph,
    same_part_floor: float = 0.25,
    epsilon: float = 1.0e-6,
) -> torch.Tensor:
    """Robust material-neighbour TV on soft static/part probabilities.

    Unlike the old affinity formulation, no edge is deleted by the current label.
    This lets dynamic anchors recruit initially false-static occluded neighbours.
    """
    if probabilities.ndim != 2:
        raise ValueError("probabilities must be [N,E]")
    pairs = pair_graph.pairs
    if pairs.numel() == 0:
        return probabilities.sum() * 0.0
    src, dst = pairs[:, 0], pairs[:, 1]
    delta = probabilities[src] - probabilities[dst]
    same_probability = (probabilities[src] * probabilities[dst]).sum(dim=-1)
    weight = float(same_part_floor) + (1.0 - float(same_part_floor)) * same_probability.detach()
    return (weight * torch.sqrt(delta.square().sum(dim=-1) + float(epsilon) ** 2)).mean()




def dino_cosine_distance_pair_values(
    features: torch.Tensor,
    pairs: torch.Tensor,
) -> torch.Tensor:
    """Exact cosine distance 1-cos(F_i,F_j), with no learned/hand threshold."""
    if pairs.numel() == 0:
        return features.new_empty((0,))
    feat = torch.nn.functional.normalize(features.detach().float(), dim=-1)
    src, dst = pairs[:, 0], pairs[:, 1]
    cosine = (feat[src] * feat[dst]).sum(dim=-1).clamp(-1.0, 1.0)
    return 1.0 - cosine


def dino_same_material_affinity_from_distance(distance: torch.Tensor) -> torch.Tensor:
    """Convert exact cosine distance to a bounded same-material graph weight.

    The measured descriptor boundary quantity remains exactly

        d_DINO = 1 - cos(F_i, F_j) in [0, 2].

    A graph affinity must lie in [0,1], so we use only the linear change of
    coordinates a = 1 - d/2 = (1+cos)/2.  There is no threshold, temperature,
    sigmoid, exponential kernel, or hand-picked feature-distance scale.
    """
    return (1.0 - 0.5 * distance).clamp(0.0, 1.0)

def assignment_dino_boundary_coherence_loss(
    probabilities: torch.Tensor,
    pair_graph: PairGraph,
    dino_features: torch.Tensor | None,
    *,
    epsilon: float = 1.0e-6,
) -> torch.Tensor:
    """DINO-aware label smoothness using exact cosine distance.

    DINO does not decide static/dynamic.  It only determines how strongly a
    canonical neighbour pair should resist a material-label discontinuity.
    No cosine threshold and no sigmoid temperature are used.
    """
    if probabilities.ndim != 2:
        raise ValueError("probabilities must be [N,E]")
    pairs = pair_graph.pairs
    if pairs.numel() == 0 or dino_features is None or dino_features.numel() == 0:
        return probabilities.sum() * 0.0
    if dino_features.ndim != 2 or dino_features.shape[0] != probabilities.shape[0]:
        raise ValueError("dino_features must be [N,F] and match probabilities")
    src, dst = pairs[:, 0], pairs[:, 1]
    distance = dino_cosine_distance_pair_values(dino_features, pairs)
    same_dino = dino_same_material_affinity_from_distance(distance).to(probabilities.dtype)
    delta = probabilities[src] - probabilities[dst]
    label_distance = torch.sqrt(delta.square().sum(dim=-1) + float(epsilon) ** 2)
    return (same_dino * label_distance).sum() / same_dino.sum().clamp_min(float(epsilon))



def aggregate_point_probabilities_to_cells(
    point_probabilities: torch.Tensor,
    point_cell_ids: torch.Tensor,
    num_cells: int,
):
    """Differentiably average persistent material probabilities inside fixed cells."""
    if point_probabilities.ndim != 2:
        raise ValueError("point_probabilities must be [N,E]")
    ids = point_cell_ids.detach().reshape(-1).long()
    if ids.numel() != point_probabilities.shape[0]:
        raise ValueError("point_cell_ids must match points")
    C = int(num_cells)
    E = int(point_probabilities.shape[1])
    accum = torch.zeros((C, E), device=point_probabilities.device, dtype=point_probabilities.dtype)
    counts = torch.zeros((C,), device=point_probabilities.device, dtype=point_probabilities.dtype)
    valid = (ids >= 0) & (ids < C)
    if valid.any():
        accum.index_add_(0, ids[valid], point_probabilities[valid])
        counts.index_add_(0, ids[valid], torch.ones_like(ids[valid], dtype=point_probabilities.dtype))
    probs = accum / counts.clamp_min(1.0)[:, None]
    # Empty cells are explicitly static for bookkeeping only; they are excluded
    # from every material loss by occupied_mask.
    if E > 0:
        empty = counts <= 0
        probs = probs.clone()
        probs[empty] = 0.0
        probs[empty, 0] = 1.0
    return probs, counts



def aggregate_point_features_to_cells(
    point_features: torch.Tensor,
    point_cell_ids: torch.Tensor,
    num_cells: int,
    *,
    normalize: bool = True,
):
    """Average material features (e.g. DINO) inside occupied Eulerian cells."""
    if point_features is None or point_features.numel() == 0:
        return None, None
    if point_features.ndim != 2:
        raise ValueError("point_features must be [N,F]")
    ids = point_cell_ids.detach().reshape(-1).long()
    if ids.numel() != point_features.shape[0]:
        raise ValueError("point_cell_ids must match point_features")
    C = int(num_cells); Fdim = int(point_features.shape[1])
    feat = point_features.detach().float()
    if normalize:
        feat = torch.nn.functional.normalize(feat, dim=-1)
    accum = torch.zeros((C, Fdim), device=feat.device, dtype=feat.dtype)
    counts = torch.zeros((C,), device=feat.device, dtype=feat.dtype)
    valid = (ids >= 0) & (ids < C)
    if valid.any():
        accum.index_add_(0, ids[valid], feat[valid])
        counts.index_add_(0, ids[valid], torch.ones_like(ids[valid], dtype=feat.dtype))
    mean = accum / counts.clamp_min(1.0)[:, None]
    if normalize:
        mean = torch.nn.functional.normalize(mean, dim=-1)
    return mean, counts
