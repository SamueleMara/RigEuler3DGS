from __future__ import annotations

from dataclasses import dataclass

import torch

from .moments import compute_soft_inertia, compute_soft_mass, inertia_invariants
from .rigid_fit import fit_weighted_twist, skew_matrix, weighted_kabsch


@dataclass
class RigidPartSequenceState:
    times: torch.Tensor
    dt: torch.Tensor
    part_prob: torch.Tensor
    mass: torch.Tensor
    com: torch.Tensor
    inertia: torch.Tensor
    rotation: torch.Tensor
    translation: torch.Tensor
    omega: torch.Tensor
    b: torch.Tensor


def huber(value: torch.Tensor, delta: float = 0.01) -> torch.Tensor:
    abs_value = value.abs()
    d = torch.as_tensor(float(delta), dtype=value.dtype, device=value.device)
    return torch.where(abs_value <= d, 0.5 * value.square(), d * (abs_value - 0.5 * d))


@torch.no_grad()
def build_knn_graph(xyz: torch.Tensor, k: int = 16, chunk_size: int = 2048) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    points = torch.as_tensor(xyz).detach().float().cpu()
    n = int(points.shape[0])
    kk = min(max(1, int(k)) + 1, n)
    src, dst, dist = [], [], []
    for start in range(0, n, max(1, int(chunk_size))):
        end = min(n, start + max(1, int(chunk_size)))
        d = torch.cdist(points[start:end], points)
        val, idx = torch.topk(d, k=kk, largest=False, dim=1)
        val, idx = val[:, 1:], idx[:, 1:]
        row = torch.arange(start, end)[:, None].expand_as(idx)
        src.append(row.reshape(-1))
        dst.append(idx.reshape(-1))
        dist.append(val.reshape(-1))
    return torch.cat(src).long(), torch.cat(dst).long(), torch.cat(dist).float()


def build_sequence_state(
    canonical_xyz: torch.Tensor,
    positions: torch.Tensor,
    velocities: torch.Tensor,
    part_prob: torch.Tensor,
    times: torch.Tensor,
    *,
    eps: float = 1.0e-8,
) -> RigidPartSequenceState:
    xyz0 = torch.as_tensor(canonical_xyz)
    pos = torch.as_tensor(positions, dtype=xyz0.dtype, device=xyz0.device)
    vel = torch.as_tensor(velocities, dtype=xyz0.dtype, device=xyz0.device)
    prob = torch.as_tensor(part_prob, dtype=xyz0.dtype, device=xyz0.device)
    t = torch.as_tensor(times, dtype=xyz0.dtype, device=xyz0.device).reshape(-1)
    k1 = int(prob.shape[1])
    masses = compute_soft_mass(prob)
    com_frames, inertia_frames, rotations, translations, omegas, bs = [], [], [], [], [], []
    eye = torch.eye(3, dtype=xyz0.dtype, device=xyz0.device)
    zero = torch.zeros(3, dtype=xyz0.dtype, device=xyz0.device)
    for frame in range(int(pos.shape[0])):
        inertia, com = compute_soft_inertia(pos[frame], prob, normalize=True, eps=eps)
        com_frames.append(com)
        inertia_frames.append(inertia)
        r_frame, t_frame, w_frame, b_frame = [], [], [], []
        for part in range(k1):
            if part == 0:
                r_frame.append(eye)
                t_frame.append(zero)
                w_frame.append(zero)
                b_frame.append(zero)
                continue
            weights = prob[:, part]
            r, trans = weighted_kabsch(xyz0, pos[frame], weights, eps=eps)
            omega, b = fit_weighted_twist(pos[frame], vel[frame], weights, eps=max(eps, 1.0e-6))
            r_frame.append(r)
            t_frame.append(trans)
            w_frame.append(omega)
            b_frame.append(b)
        rotations.append(torch.stack(r_frame))
        translations.append(torch.stack(t_frame))
        omegas.append(torch.stack(w_frame))
        bs.append(torch.stack(b_frame))
    return RigidPartSequenceState(
        times=t,
        dt=torch.diff(t).clamp_min(float(eps)),
        part_prob=prob,
        mass=masses,
        com=torch.stack(com_frames),
        inertia=torch.stack(inertia_frames),
        rotation=torch.stack(rotations),
        translation=torch.stack(translations),
        omega=torch.stack(omegas),
        b=torch.stack(bs),
    )


def twist_assignment_loss(positions: torch.Tensor, velocities: torch.Tensor, state: RigidPartSequenceState) -> torch.Tensor:
    """Raw/free-track twist assignment loss.

    Expects label-independent raw/free positions and velocities.  Do not feed
    rigid-projected replay tracks here: that makes the loss tautological.
    """
    pos = torch.as_tensor(positions, dtype=state.part_prob.dtype, device=state.part_prob.device)
    vel = torch.as_tensor(velocities, dtype=state.part_prob.dtype, device=state.part_prob.device)
    total = pos.new_tensor(0.0)
    denom = pos.new_tensor(0.0)
    for frame in range(int(pos.shape[0])):
        predicted = []
        for part in range(int(state.part_prob.shape[1])):
            if part == 0:
                predicted.append(torch.zeros_like(vel[frame]))
            else:
                predicted.append(torch.cross(state.omega[frame, part][None].expand_as(pos[frame]), pos[frame], dim=-1) + state.b[frame, part])
        pred = torch.stack(predicted, dim=1)
        err = (vel[frame, :, None, :] - pred).square().sum(dim=-1)
        total = total + (state.part_prob * err).sum()
        denom = denom + state.part_prob.sum()
    return total / denom.clamp_min(1.0e-8)


def trajectory_unary(
    canonical_xyz: torch.Tensor,
    raw_positions: torch.Tensor,
    state: RigidPartSequenceState,
) -> torch.Tensor:
    """Per-Gaussian K+1 trajectory residuals from raw/free tracks.

    Output shape is [N, K+1]. Static class 0 is identity; moving classes use
    the weighted Kabsch transforms refit from the current soft labels.
    """
    xyz0 = torch.as_tensor(canonical_xyz, dtype=state.part_prob.dtype, device=state.part_prob.device)
    pos = torch.as_tensor(raw_positions, dtype=xyz0.dtype, device=xyz0.device)
    total = torch.zeros((xyz0.shape[0], state.part_prob.shape[1]), dtype=xyz0.dtype, device=xyz0.device)
    for frame in range(int(pos.shape[0])):
        candidates = []
        for part in range(int(state.part_prob.shape[1])):
            if part == 0:
                candidates.append(xyz0)
            else:
                candidates.append(xyz0 @ state.rotation[frame, part].transpose(0, 1) + state.translation[frame, part])
        pred = torch.stack(candidates, dim=1)
        total = total + (pos[frame, :, None, :] - pred).square().sum(dim=-1)
    return total / max(1, int(pos.shape[0]))


def trajectory_unary_loss(trajectory_energy: torch.Tensor, part_prob: torch.Tensor) -> torch.Tensor:
    """Soft K+1 trajectory unary loss with gradients to labels."""
    energy = torch.as_tensor(trajectory_energy, dtype=part_prob.dtype, device=part_prob.device)
    return (part_prob * energy).sum() / part_prob.sum().clamp_min(1.0e-8)


def twist_unary(
    raw_positions: torch.Tensor,
    raw_velocities: torch.Tensor,
    state: RigidPartSequenceState,
) -> torch.Tensor:
    """Per-Gaussian K+1 twist residuals from raw/free tracks."""
    pos = torch.as_tensor(raw_positions, dtype=state.part_prob.dtype, device=state.part_prob.device)
    vel = torch.as_tensor(raw_velocities, dtype=state.part_prob.dtype, device=state.part_prob.device)
    total = torch.zeros((pos.shape[1], state.part_prob.shape[1]), dtype=pos.dtype, device=pos.device)
    for frame in range(int(pos.shape[0])):
        candidates = []
        for part in range(int(state.part_prob.shape[1])):
            if part == 0:
                candidates.append(torch.zeros_like(vel[frame]))
            else:
                candidates.append(torch.cross(state.omega[frame, part][None].expand_as(pos[frame]), pos[frame], dim=-1) + state.b[frame, part])
        pred = torch.stack(candidates, dim=1)
        total = total + (vel[frame, :, None, :] - pred).square().sum(dim=-1)
    return total / max(1, int(pos.shape[0]))


def twist_unary_loss(twist_energy: torch.Tensor, part_prob: torch.Tensor) -> torch.Tensor:
    """Soft K+1 twist unary loss with gradients to labels."""
    energy = torch.as_tensor(twist_energy, dtype=part_prob.dtype, device=part_prob.device)
    return (part_prob * energy).sum() / part_prob.sum().clamp_min(1.0e-8)


def raw_graph_affinity(
    positions: torch.Tensor,
    velocities: torch.Tensor,
    edge_src: torch.Tensor,
    edge_dst: torch.Tensor,
    edge_d0: torch.Tensor,
    *,
    eps: float = 1.0e-8,
) -> torch.Tensor:
    """Label-independent rigidity affinity computed from raw/free tracks."""
    pos = torch.as_tensor(positions)
    vel = torch.as_tensor(velocities, dtype=pos.dtype, device=pos.device)
    src = torch.as_tensor(edge_src, device=pos.device).long()
    dst = torch.as_tensor(edge_dst, device=pos.device).long()
    d0 = torch.as_tensor(edge_d0, dtype=pos.dtype, device=pos.device)
    dist_errors, rate_errors = [], []
    for frame in range(int(pos.shape[0])):
        delta_x = pos[frame, src] - pos[frame, dst]
        delta_v = vel[frame, src] - vel[frame, dst]
        dist_errors.append((torch.linalg.norm(delta_x, dim=-1) - d0).square())
        rate_errors.append((delta_x * delta_v).sum(dim=-1).square())
    dist_error = torch.stack(dist_errors).mean(dim=0)
    rate_error = torch.stack(rate_errors).mean(dim=0)
    dist_scale = torch.quantile(dist_error.detach(), 0.75).clamp_min(float(eps))
    rate_scale = torch.quantile(rate_error.detach(), 0.75).clamp_min(float(eps))
    return torch.exp(-dist_error / dist_scale) * torch.exp(-rate_error / rate_scale)


def graph_label_coherence_loss(
    part_prob: torch.Tensor,
    edge_src: torch.Tensor,
    edge_dst: torch.Tensor,
    affinity: torch.Tensor,
) -> torch.Tensor:
    """Raw-rigidity-weighted soft-label coherence."""
    src = torch.as_tensor(edge_src, device=part_prob.device).long()
    dst = torch.as_tensor(edge_dst, device=part_prob.device).long()
    weight = torch.as_tensor(affinity, dtype=part_prob.dtype, device=part_prob.device).clamp_min(0.0)
    diff = (part_prob[src] - part_prob[dst]).square().sum(dim=-1)
    return (weight * diff).sum() / weight.sum().clamp_min(1.0e-8)


def pairwise_distance_loss(
    positions: torch.Tensor,
    edge_src: torch.Tensor,
    edge_dst: torch.Tensor,
    edge_d0: torch.Tensor,
    part_prob: torch.Tensor,
    *,
    delta: float = 0.01,
) -> torch.Tensor:
    """Raw/free-track same-label pairwise distance preservation loss."""
    pos = torch.as_tensor(positions, dtype=part_prob.dtype, device=part_prob.device)
    src = torch.as_tensor(edge_src, device=part_prob.device).long()
    dst = torch.as_tensor(edge_dst, device=part_prob.device).long()
    d0 = torch.as_tensor(edge_d0, dtype=part_prob.dtype, device=part_prob.device)
    same = (part_prob[src] * part_prob[dst]).sum(dim=-1)
    total = pos.new_tensor(0.0)
    for frame in range(int(pos.shape[0])):
        dist = torch.linalg.norm(pos[frame, src] - pos[frame, dst], dim=-1)
        total = total + (same * huber(dist - d0, delta=delta)).sum()
    return total / (same.sum() * max(1, int(pos.shape[0]))).clamp_min(1.0e-8)


def distance_rate_loss(
    positions: torch.Tensor,
    velocities: torch.Tensor,
    edge_src: torch.Tensor,
    edge_dst: torch.Tensor,
    part_prob: torch.Tensor,
) -> torch.Tensor:
    """Raw/free-track same-label distance-rate rigidity loss."""
    pos = torch.as_tensor(positions, dtype=part_prob.dtype, device=part_prob.device)
    vel = torch.as_tensor(velocities, dtype=part_prob.dtype, device=part_prob.device)
    src = torch.as_tensor(edge_src, device=part_prob.device).long()
    dst = torch.as_tensor(edge_dst, device=part_prob.device).long()
    same = (part_prob[src] * part_prob[dst]).sum(dim=-1)
    total = pos.new_tensor(0.0)
    for frame in range(int(pos.shape[0])):
        rate = ((pos[frame, src] - pos[frame, dst]) * (vel[frame, src] - vel[frame, dst])).sum(dim=-1)
        total = total + (same * rate.square()).sum()
    return total / (same.sum() * max(1, int(pos.shape[0]))).clamp_min(1.0e-8)


def com_transport_loss(state: RigidPartSequenceState, min_mass: float = 3.0) -> torch.Tensor:
    """COM transport consistency computed from raw/free tracks."""
    valid = state.mass > float(min_mass)
    total = state.com.new_tensor(0.0)
    count = state.com.new_tensor(0.0)
    c0 = state.com[0]
    for frame in range(int(state.com.shape[0])):
        pred = torch.einsum("kij,kj->ki", state.rotation[frame], c0) + state.translation[frame]
        static_pred = pred.clone()
        static_pred[0] = c0[0]
        err = (state.com[frame] - static_pred).square().sum(dim=-1)
        total = total + err[valid].sum()
        count = count + valid.sum()
    return total / count.clamp_min(1.0)


def inertia_transport_loss(state: RigidPartSequenceState, min_mass: float = 3.0) -> torch.Tensor:
    """Inertia transport consistency computed from raw/free tracks."""
    valid = state.mass > float(min_mass)
    total = state.inertia.new_tensor(0.0)
    count = state.inertia.new_tensor(0.0)
    j0 = state.inertia[0]
    for frame in range(int(state.inertia.shape[0])):
        pred = state.rotation[frame] @ j0 @ state.rotation[frame].transpose(-1, -2)
        static_pred = pred.clone()
        static_pred[0] = j0[0]
        err = (state.inertia[frame] - static_pred).square().sum(dim=(-1, -2))
        total = total + err[valid].sum()
        count = count + valid.sum()
    return total / count.clamp_min(1.0)


def com_differential_loss(state: RigidPartSequenceState, min_mass: float = 3.0) -> torch.Tensor:
    if int(state.com.shape[0]) < 2:
        return state.com.new_tensor(0.0)
    valid = state.mass > float(min_mass)
    cdot = (state.com[1:] - state.com[:-1]) / state.dt[:, None, None]
    omega_cross_c = torch.cross(state.omega[:-1], state.com[:-1], dim=-1) + state.b[:-1]
    err = (cdot - omega_cross_c).square().sum(dim=-1)
    return err[:, valid].mean() if bool(valid.any()) else err.new_tensor(0.0)


def inertia_differential_loss(state: RigidPartSequenceState, min_mass: float = 3.0) -> torch.Tensor:
    if int(state.inertia.shape[0]) < 2:
        return state.inertia.new_tensor(0.0)
    valid = state.mass > float(min_mass)
    jdot = (state.inertia[1:] - state.inertia[:-1]) / state.dt[:, None, None, None]
    omega_mat = skew_matrix(state.omega[:-1])
    pred = omega_mat @ state.inertia[:-1] - state.inertia[:-1] @ omega_mat
    err = (jdot - pred).square().sum(dim=(-1, -2))
    return err[:, valid].mean() if bool(valid.any()) else err.new_tensor(0.0)


def inertia_invariant_loss(state: RigidPartSequenceState, min_mass: float = 3.0, eps: float = 1.0e-8) -> torch.Tensor:
    valid = state.mass > float(min_mass)
    i1, i2 = inertia_invariants(state.inertia)
    e1 = ((i1 - i1[:1]) / i1[:1].abs().clamp_min(float(eps))).square()
    e2 = ((i2 - i2[:1]) / i2[:1].abs().clamp_min(float(eps))).square()
    err = e1 + e2
    return err[:, valid].mean() if bool(valid.any()) else err.new_tensor(0.0)


def occupancy_loss(part_prob: torch.Tensor, min_fraction: float) -> torch.Tensor:
    prob = torch.as_tensor(part_prob)
    frac = prob.mean(dim=0)
    return torch.relu(float(min_fraction) - frac).square().sum()
