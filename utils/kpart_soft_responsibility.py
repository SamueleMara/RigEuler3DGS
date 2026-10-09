"""Soft K+1 Gaussian responsibilities driven by full rigid velocity fields."""

from __future__ import annotations

import torch

from utils.rigid_killing_motion import predict_rigid_cell_velocity


def _pack_value(pack, key, default):
    return torch.as_tensor(pack.get(key, default)).detach().cpu().double()


@torch.no_grad()
def compute_velocity_field_unaries(observation_packs, fields, *, huber_delta=0.0):
    """Compute static-zero plus K full 3D field compatibility unaries."""
    packs = list(observation_packs) if isinstance(observation_packs, (list, tuple)) else [observation_packs] * (len(fields) + 1)
    if len(packs) == len(fields):
        packs = [packs[0]] + packs
    if len(packs) != len(fields) + 1:
        raise ValueError("observation_packs must be one common pack or static plus K packs")
    columns, valid_columns, residual_columns = [], [], []
    for label in range(len(fields) + 1):
        pack = packs[label]
        velocity = _pack_value(pack, "velocity", None)
        positions = _pack_value(pack, "positions", None)
        confidence = _pack_value(pack, "confidence", torch.ones(velocity.shape[:2]))
        valid = torch.as_tensor(pack.get("valid", torch.ones(velocity.shape[:2], dtype=torch.bool))).detach().cpu().bool()
        if positions.ndim == 2:
            positions = positions[None].expand_as(velocity)
        predicted = torch.zeros_like(velocity) if label == 0 else predict_rigid_cell_velocity(fields[label - 1]["mode"], positions, fields[label - 1]["temporal_speed"])
        residual_vector = velocity - predicted
        normal = pack.get("normal_matrix")
        weight_sum = _pack_value(pack, "weight_sum", confidence)
        if normal is not None:
            normal_value = torch.as_tensor(normal).detach().cpu().double()
            sample = torch.einsum("tni,tnij,tnj->tn", residual_vector, normal_value, residual_vector) / weight_sum.clamp_min(1.0e-8)
        else:
            norm = torch.linalg.norm(residual_vector, dim=-1)
            delta = float(huber_delta)
            sample = norm.square() if delta <= 0 else torch.where(norm <= delta, 0.5 * norm.square(), delta * (norm - 0.5 * delta))
        effective = confidence * valid.double()
        energy = (effective * sample).sum(0) / effective.sum(0).clamp_min(1.0e-8)
        informative = effective.sum(0) > 0
        energy = torch.where(informative, energy, torch.full_like(energy, float("nan")))
        columns.append(energy)
        valid_columns.append(informative)
        residual_columns.append(torch.linalg.norm(residual_vector, dim=-1))
    unary = torch.stack(columns, -1)
    finite = torch.isfinite(unary)
    if bool(finite.any()):
        fill = torch.quantile(unary[finite], 0.9) + unary[finite].std().clamp_min(1.0e-6)
    else:
        fill = torch.tensor(1.0, dtype=unary.dtype)
    unary = torch.where(finite, unary, fill)
    unary -= unary.amin(-1, keepdim=True)
    return {
        "unaries": unary.float(),
        "informative": torch.stack(valid_columns, -1).any(-1),
        "per_label_valid": torch.stack(valid_columns, -1),
        "residuals": torch.stack(residual_columns, -1),
        "static_is_exact_zero_field": True,
    }


@torch.no_grad()
def initialize_responsibilities_from_unaries(unaries, *, temperature=1.0, previous=None):
    value = torch.as_tensor(unaries).float()
    result = torch.softmax(-value / max(float(temperature), 1.0e-6), dim=-1)
    if previous is not None:
        old = torch.as_tensor(previous, device=result.device, dtype=result.dtype)
        result = 0.75 * result + 0.25 * old
        result /= result.sum(-1, keepdim=True).clamp_min(1.0e-8)
    return result


def compute_soft_entropy(responsibilities):
    value = torch.as_tensor(responsibilities)
    return -(value * torch.log(value.clamp_min(1.0e-8))).sum(-1)


def compute_3d_dino_regularizer(responsibilities, pairs, same_material_affinity):
    value = torch.as_tensor(responsibilities)
    edge = torch.as_tensor(pairs, device=value.device).long()
    affinity = torch.as_tensor(same_material_affinity, device=value.device, dtype=value.dtype)
    if edge.numel() == 0:
        return value.sum() * 0.0
    return (affinity * (value[edge[:, 0]] - value[edge[:, 1]]).square().sum(-1)).sum() / affinity.sum().clamp_min(1.0e-8)


def optimize_responsibility_logits(
    unaries,
    *,
    initial_responsibilities=None,
    temperature=0.25,
    pairs=None,
    same_material_affinity=None,
    dino_3d_weight=0.15,
    sam3_pairs=None,
    sam3_affinity=None,
    sam3_3d_weight=0.0,
    seam_loss_fn=None,
    dino_2d_seam_weight=0.05,
    sam3_2d_seam_weight=0.0,
    steps=8,
    learning_rate=0.3,
):
    unary = torch.as_tensor(unaries).detach().float()
    if initial_responsibilities is None:
        initial = initialize_responsibilities_from_unaries(unary, temperature=temperature)
    else:
        initial = torch.as_tensor(initial_responsibilities, device=unary.device, dtype=unary.dtype)
    logits = torch.log(initial.clamp_min(1.0e-8)).detach().requires_grad_(True)
    optimizer = torch.optim.Adam([logits], lr=float(learning_rate))
    last = {}
    for _ in range(max(1, int(steps))):
        optimizer.zero_grad(set_to_none=True)
        responsibility = torch.softmax(logits, -1)
        data = (responsibility * unary).sum(-1).mean()
        entropy_term = float(temperature) * (responsibility * torch.log(responsibility.clamp_min(1.0e-8))).sum(-1).mean()
        dino3d = responsibility.sum() * 0.0
        if pairs is not None and same_material_affinity is not None and float(dino_3d_weight) > 0:
            dino3d = compute_3d_dino_regularizer(responsibility, pairs, same_material_affinity)
        sam3_3d = responsibility.sum() * 0.0
        if sam3_pairs is not None and sam3_affinity is not None and float(sam3_3d_weight) > 0:
            sam3_3d = compute_3d_dino_regularizer(responsibility, sam3_pairs, sam3_affinity)
        seam = responsibility.sum() * 0.0 if seam_loss_fn is None else seam_loss_fn(responsibility)
        loss = (
            data
            + entropy_term
            + float(dino_3d_weight) * dino3d
            + float(sam3_3d_weight) * sam3_3d
            + (float(dino_2d_seam_weight) + float(sam3_2d_seam_weight)) * seam
        )
        loss.backward()
        optimizer.step()
        last = {
            "free_energy": float(loss.detach()),
            "unary": float(data.detach()),
            "entropy_term": float(entropy_term.detach()),
            "dino_3d": float(dino3d.detach()),
            "sam3_3d": float(sam3_3d.detach()),
            "dino_2d_seam": float(seam.detach()),
            "sam3_2d_seam": float(seam.detach()) if float(sam3_2d_seam_weight) > 0 else 0.0,
        }
    return {"responsibilities": torch.softmax(logits.detach(), -1), "logits": logits.detach(), "diagnostics": last}


@torch.no_grad()
def compute_boundary_ambiguity_mask(responsibilities, *, confidence_threshold=0.8, entropy_threshold=None, margin_threshold=0.15):
    value = torch.as_tensor(responsibilities).float()
    ordered = value.sort(-1, descending=True).values
    confidence = ordered[:, 0]
    margin = ordered[:, 0] - ordered[:, 1]
    entropy = compute_soft_entropy(value)
    if entropy_threshold is None:
        entropy_threshold = 0.5 * float(torch.log(torch.tensor(float(value.shape[1]))))
    ambiguous = (confidence < float(confidence_threshold)) | (margin < float(margin_threshold)) | (entropy > float(entropy_threshold))
    return {"ambiguous_mask": ambiguous, "confidence": confidence, "margin": margin, "entropy": entropy}


@torch.no_grad()
def compute_field_fit_weights(responsibilities, part_index, *, responsibility_power=2.0, observation_confidence=None):
    value = torch.as_tensor(responsibilities).float()[:, int(part_index)].clamp(0, 1).pow(float(responsibility_power))
    if observation_confidence is None:
        return value
    confidence = torch.as_tensor(observation_confidence).float()
    return confidence * value if confidence.ndim == 1 else confidence * value[None]
