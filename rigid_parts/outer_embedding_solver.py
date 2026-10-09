from __future__ import annotations

import torch
import torch.nn.functional as F

from .endpoint import _nearest_squared
from .moments import compute_global_com, compute_global_inertia, compute_soft_inertia


def candidate_final_positions(
    canonical_xyz: torch.Tensor,
    final_rotation: torch.Tensor,
    final_translation: torch.Tensor,
) -> torch.Tensor:
    """Legacy rigid-transform candidates (kept for tests and diagnostics).

    FIRM v3 §25 uses per-Gaussian integrated endpoints instead; see
    :func:`candidates_from_local_endpoints`.
    """
    xyz = torch.as_tensor(canonical_xyz)
    rot = torch.as_tensor(final_rotation, dtype=xyz.dtype, device=xyz.device)
    trans = torch.as_tensor(final_translation, dtype=xyz.dtype, device=xyz.device)
    candidates = [xyz]
    for part in range(1, int(rot.shape[0])):
        candidates.append(xyz @ rot[part].transpose(0, 1) + trans[part])
    return torch.stack(candidates, dim=1)


def candidates_from_local_endpoints(
    canonical_xyz: torch.Tensor,
    moving_endpoints: torch.Tensor,
) -> torch.Tensor:
    """FIRM v3 §25: K+1 candidate endpoints, static identity first.

        x_i0^T = x_i^0                     (static identity, always a candidate)
        x_ik^T = local part-k integrated final position of Gaussian i

    ``moving_endpoints`` is [K, N, 3] straight from the local Killing solve.
    Returns [N, K+1, 3].
    """
    xyz = torch.as_tensor(canonical_xyz)
    moving = torch.as_tensor(moving_endpoints, dtype=xyz.dtype, device=xyz.device)
    if moving.ndim != 3 or int(moving.shape[1]) != int(xyz.shape[0]):
        raise ValueError("moving_endpoints must be [K, N, 3] matching the canonical cloud")
    return torch.cat((xyz[:, None, :], moving.transpose(0, 1)), dim=1)


def soft_compose_candidates(candidates: torch.Tensor, part_prob: torch.Tensor) -> torch.Tensor:
    prob = torch.as_tensor(part_prob, dtype=candidates.dtype, device=candidates.device)
    return (prob[:, :, None] * candidates).sum(dim=1)


def candidate_nn_energies(candidates: torch.Tensor, final_xyz: torch.Tensor, chunk_size: int) -> torch.Tensor:
    cand = torch.as_tensor(candidates)
    final = torch.as_tensor(final_xyz, dtype=cand.dtype, device=cand.device)
    values = []
    for part in range(int(cand.shape[1])):
        values.append(_nearest_squared(cand[:, part], final, chunk_size))
    return torch.stack(values, dim=1)


def _soft_inertia_at_positions(points_by_part: torch.Tensor, part_prob: torch.Tensor, eps: float = 1.0e-8) -> tuple[torch.Tensor, torch.Tensor]:
    prob = torch.as_tensor(part_prob, dtype=points_by_part.dtype, device=points_by_part.device)
    centers = []
    inertias = []
    eye = torch.eye(3, dtype=points_by_part.dtype, device=points_by_part.device)
    for part in range(int(points_by_part.shape[1])):
        pts = points_by_part[:, part]
        w = prob[:, part]
        mass = w.sum().clamp_min(float(eps))
        center = (w[:, None] * pts).sum(dim=0) / mass
        rel = pts - center[None]
        term = rel.square().sum(-1)[:, None, None] * eye - rel[:, :, None] * rel[:, None, :]
        inertia = (w[:, None, None] * term).sum(dim=0) / mass
        centers.append(center)
        inertias.append(0.5 * (inertia + inertia.T))
    return torch.stack(inertias), torch.stack(centers)


def final_target_part_assignment(
    final_xyz: torch.Tensor,
    candidates: torch.Tensor,
    *,
    temperature: float,
    chunk_size: int,
) -> torch.Tensor:
    final = torch.as_tensor(final_xyz, dtype=candidates.dtype, device=candidates.device)
    distances = []
    for part in range(int(candidates.shape[1])):
        distances.append(_nearest_squared(final, candidates[:, part].detach(), chunk_size))
    return torch.softmax(-torch.stack(distances, dim=1) / max(float(temperature), 1.0e-8), dim=1).detach()


def build_outer_targets(
    *,
    final_xyz: torch.Tensor,
    candidates: torch.Tensor,
    temperature_final_assignment: float,
    chunk_size: int,
) -> dict[str, torch.Tensor]:
    """FIRM v3 §28: compute the final-cloud targets ONCE per outer iteration.

    q_jk and the target 0th/1st/2nd moments are detached and then held fixed for
    every embedding gradient step, so the target cannot chase a shrinking part.
    """
    with torch.no_grad():
        q = final_target_part_assignment(
            final_xyz,
            candidates,
            temperature=temperature_final_assignment,
            chunk_size=chunk_size,
        )
        target_inertia, target_com = compute_soft_inertia(final_xyz, q, normalize=True)
        target_mass = q.sum(dim=0) / max(int(final_xyz.shape[0]), 1)  # §27 M_k^target
        target_global_com = compute_global_com(final_xyz)
        target_global_inertia = compute_global_inertia(final_xyz, target_global_com)
    return {
        "q": q.detach(),
        "part_com": target_com.detach(),
        "part_inertia": target_inertia.detach(),
        "part_mass": target_mass.detach(),
        "global_com": target_global_com.detach(),
        "global_inertia": target_global_inertia.detach(),
    }


def outer_embedding_losses(
    *,
    canonical_xyz: torch.Tensor,
    final_xyz: torch.Tensor,
    part_prob: torch.Tensor,
    static_anchor_mask: torch.Tensor,
    prototypes: torch.Tensor,
    bbox_diag: torch.Tensor,
    chunk_size: int,
    margin: float,
    candidates: torch.Tensor | None = None,
    targets: dict[str, torch.Tensor] | None = None,
    previous_prob: torch.Tensor | None = None,
    final_rotation: torch.Tensor | None = None,
    final_translation: torch.Tensor | None = None,
    motion_separation: torch.Tensor | None = None,
    temperature_final_assignment: float = 0.01,
) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    """FIRM v3 §31 outer objective.

    ``candidates`` are the per-Gaussian K+1 endpoints (§25). ``targets`` are the
    fixed detached final-cloud targets from :func:`build_outer_targets` (§28).
    Both may be omitted, in which case they are derived from the legacy rigid
    ``final_rotation``/``final_translation`` for backwards compatibility.
    """
    xyz0 = torch.as_tensor(canonical_xyz)
    final = torch.as_tensor(final_xyz, dtype=xyz0.dtype, device=xyz0.device)
    prob = torch.as_tensor(part_prob, dtype=xyz0.dtype, device=xyz0.device)
    if candidates is None:
        if final_rotation is None or final_translation is None:
            raise ValueError("outer_embedding_losses needs candidates or a rigid final transform")
        candidates = candidate_final_positions(xyz0, final_rotation, final_translation)
    if targets is None:
        targets = build_outer_targets(
            final_xyz=final,
            candidates=candidates,
            temperature_final_assignment=temperature_final_assignment,
            chunk_size=chunk_size,
        )

    cand_nn = candidate_nn_energies(candidates, final, chunk_size)
    pred_final = soft_compose_candidates(candidates, prob)

    # §28 candidate NN: (1/N) sum_i sum_k p_ik E_ik   (prob rows sum to 1 => prob.sum() == N)
    cand_loss = (prob * cand_nn).sum() / prob.sum().clamp_min(1.0e-8)
    final_nn = _nearest_squared(pred_final, final, chunk_size).mean() + _nearest_squared(final, pred_final, chunk_size).mean()

    # §33 normalisation: COM by bbox^2, inertia relative, mass dimensionless.
    scale2 = torch.as_tensor(bbox_diag, dtype=xyz0.dtype, device=xyz0.device).square().clamp_min(1.0e-8)
    pred_com = compute_global_com(pred_final)
    global_com = (pred_com - targets["global_com"]).square().sum() / scale2
    pred_inertia = compute_global_inertia(pred_final, pred_com)
    global_inertia = (pred_inertia - targets["global_inertia"]).square().sum() / targets["global_inertia"].square().sum().clamp_min(1.0e-8)

    pred_part_inertia, pred_part_com = _soft_inertia_at_positions(candidates, prob)
    part_com = (pred_part_com - targets["part_com"]).square().sum(dim=-1).sum() / scale2
    part_inertia = (pred_part_inertia - targets["part_inertia"]).square().sum(dim=(-2, -1))
    part_inertia = (part_inertia / targets["part_inertia"].square().sum(dim=(-2, -1)).clamp_min(1.0e-8)).sum()

    # §27 zeroth moment: support/mass fraction. Without this, normalised inertia
    # can always be lowered by shrinking a part into a compact subset.
    pred_mass = prob.sum(dim=0) / max(int(prob.shape[0]), 1)
    part_mass = (pred_mass - targets["part_mass"]).square().sum()

    # §22: anchors are NOT hard-pinned any more, so this gradient is live.
    anchor_mask = torch.as_tensor(static_anchor_mask, device=prob.device).bool()
    static_anchor = (
        -torch.log(prob[anchor_mask, 0].clamp_min(1.0e-8)).mean()
        if bool(anchor_mask.any())
        else prob.sum() * 0.0
    )

    # §24 candidate margin, literal: l != k only.
    diff = cand_nn[:, :, None] - cand_nn[:, None, :]          # [N,K+1,K+1] = E_ik - E_il
    hinge = F.relu(float(margin) + diff)
    k_plus_1 = int(cand_nn.shape[1])
    off_diagonal = ~torch.eye(k_plus_1, dtype=torch.bool, device=prob.device)
    candidate_margin = (prob * (hinge * off_diagonal[None]).sum(dim=2)).sum() / prob.sum().clamp_min(1.0e-8)

    entropy = -(prob.clamp_min(1.0e-8) * prob.clamp_min(1.0e-8).log()).sum(dim=1).mean()
    occupancy = F.relu(1.0 / float(prob.shape[1]) * 0.25 - prob.mean(dim=0)).square().sum()

    # §30 trust region: KL(p_old || p), p_old detached. A proximal term for the
    # alternating optimisation, not a semantic prior.
    if previous_prob is None:
        trust = prob.sum() * 0.0
    else:
        p_old = torch.as_tensor(previous_prob, dtype=prob.dtype, device=prob.device).detach().clamp_min(1.0e-8)
        trust = (p_old * (p_old.log() - prob.clamp_min(1.0e-8).log())).sum(dim=1).mean()

    emb_motion = prototypes.new_tensor(0.0)
    motion_sep_debug = prototypes.new_zeros((int(prob.shape[1]), int(prob.shape[1])))
    if motion_separation is not None:
        sep = torch.as_tensor(motion_separation, dtype=prob.dtype, device=prob.device)
        if sep.shape == (prob.shape[1], prob.shape[1]) and int(prob.shape[1]) > 2:
            moving_sep = sep[1:, 1:]
            motion_sep_debug = sep.detach()
            offdiag = ~torch.eye(moving_sep.shape[0], dtype=torch.bool, device=prob.device)
            if bool(offdiag.any()):
                margin_scale = torch.as_tensor(bbox_diag, dtype=prob.dtype, device=prob.device).square().clamp_min(1.0e-8)
                sim_motion = torch.exp(-moving_sep / margin_scale)[offdiag]
                moving_prob = prob[:, 1:]
                overlap = (moving_prob.T @ moving_prob / max(int(prob.shape[0]), 1))[offdiag]
                proto = torch.as_tensor(prototypes[1:], dtype=prob.dtype, device=prob.device)
                proto_dist = torch.cdist(proto, proto).square()[offdiag]
                proto_close = torch.exp(-proto_dist)
                emb_motion = (sim_motion * (overlap + proto_close)).mean()

    diagnostics = {
        "part_fraction": prob.mean(dim=0).detach(),
        "candidate_nn_energy": cand_nn.detach(),
        "predicted_final": pred_final.detach(),
        "target_part_assignment": targets["q"],
        "global_com_error": torch.linalg.norm(pred_com - targets["global_com"]).detach(),
        "global_inertia_error": torch.linalg.norm(pred_inertia - targets["global_inertia"]).detach(),
        "part_com_error": torch.linalg.norm(pred_part_com - targets["part_com"], dim=-1).detach(),
        "part_inertia_error": torch.linalg.norm(pred_part_inertia - targets["part_inertia"], dim=(-2, -1)).detach(),
        "part_mass_error": (pred_mass - targets["part_mass"]).abs().detach(),
        "predicted_part_mass": pred_mass.detach(),
        "motion_separation": motion_sep_debug.detach(),
    }
    return {
        "candidate_nn": cand_loss,
        "final_nn": final_nn,
        "global_com": global_com,
        "global_inertia": global_inertia,
        "part_mass": part_mass,
        "part_com": part_com,
        "part_inertia": part_inertia,
        "static_anchor": static_anchor,
        "embedding_motion": emb_motion,
        "candidate_margin": candidate_margin,
        "entropy": entropy,
        "occupancy": occupancy,
        "trust_region": trust,
    }, diagnostics


# FIRM v3 §32: the geometry guard. Inertia alone must never buy an update that
# makes endpoint geometry, COM or support worse.
GEOMETRY_GUARD_TERMS = ("candidate_nn", "final_nn", "global_com", "part_com", "part_mass")


def geometry_guard_value(losses: dict[str, torch.Tensor], weights: dict[str, float]) -> torch.Tensor:
    """L_geom = lam_cand L_cand + lam_NN L_finalNN + lam_GC L_globalCOM + lam_PC L_partCOM + lam_PM L_partmass."""
    total = None
    for key in GEOMETRY_GUARD_TERMS:
        if key not in losses:
            continue
        term = float(weights.get(key, 0.0)) * losses[key]
        total = term if total is None else total + term
    if total is None:
        raise ValueError("geometry guard needs at least one geometry term")
    return total
