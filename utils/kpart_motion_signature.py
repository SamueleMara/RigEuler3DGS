"""Rigid-normalized temporal speed signatures for K-part discovery.

The signature removes the spatial Killing-field magnitude from an observation:
Gaussians on the same rigid part may have very different linear velocities, but
must share the same scalar coordinate speed as a function of time.
"""

from __future__ import annotations

import torch


def _weighted_median(values: torch.Tensor, weights: torch.Tensor, dim: int):
    order = torch.argsort(values, dim=dim)
    ordered_values = torch.gather(values, dim, order)
    ordered_weights = torch.gather(weights, dim, order)
    cumulative = ordered_weights.cumsum(dim=dim)
    half = 0.5 * ordered_weights.sum(dim=dim, keepdim=True)
    index = (cumulative >= half).to(torch.int64).argmax(dim=dim, keepdim=True)
    return torch.gather(ordered_values, dim, index).squeeze(dim)


@torch.no_grad()
def compute_projected_rigid_scalar_speed(
    rigid_velocity,
    observed_velocity,
    *,
    weight=None,
    eps: float = 1.0e-10,
):
    """Return ``(Phi^T W v)/(Phi^T W Phi)`` and its excitation.

    The final dimension is the vector/measurement dimension. ``weight`` may be
    scalar per observation or diagonal per component.
    """
    phi = torch.as_tensor(rigid_velocity).double()
    observed = torch.as_tensor(observed_velocity, device=phi.device).double()
    try:
        phi, observed = torch.broadcast_tensors(phi, observed)
    except RuntimeError as exc:
        raise ValueError("rigid_velocity and observed_velocity must be broadcast-compatible") from exc
    if weight is None:
        weighted_phi = phi
    else:
        value = torch.as_tensor(weight, device=phi.device).double()
        weighted_phi = phi * (value[..., None] if value.ndim == phi.ndim - 1 else value)
    numerator = (weighted_phi * observed).sum(dim=-1)
    denominator = (weighted_phi * phi).sum(dim=-1)
    valid = torch.isfinite(numerator) & torch.isfinite(denominator) & (denominator > float(eps))
    speed = torch.where(valid, numerator / denominator.clamp_min(float(eps)), torch.zeros_like(numerator))
    return {"speed": speed, "excitation": denominator.clamp_min(0.0), "valid": valid}


@torch.no_grad()
def estimate_signature_uncertainty(speed_samples, weights, *, dim: int = -2, eps: float = 1.0e-8):
    """Robust weighted MAD uncertainty for repeated speed observations."""
    speed = torch.as_tensor(speed_samples).double()
    weight = torch.as_tensor(weights, device=speed.device).double()
    if speed.shape != weight.shape:
        raise ValueError("speed_samples and weights must have equal shape")
    usable = torch.isfinite(speed) & torch.isfinite(weight) & (weight > 0)
    safe_speed = torch.where(usable, speed, torch.zeros_like(speed))
    safe_weight = torch.where(usable, weight, torch.zeros_like(weight))
    median = _weighted_median(safe_speed, safe_weight, dim)
    deviation = (safe_speed - median.unsqueeze(dim)).abs()
    mad = _weighted_median(deviation, safe_weight, dim)
    effective = safe_weight.sum(dim=dim)
    uncertainty = 1.4826 * mad + torch.rsqrt(effective.clamp_min(float(eps)))
    return torch.where(effective > 0, uncertainty, torch.full_like(uncertainty, float("inf")))


@torch.no_grad()
def compute_gaussian_speed_signature(inferred_speed, weights, *, eps: float = 1.0e-8):
    """Aggregate camera-wise inferred scalar speeds into ``[N,T]`` signatures.

    Inputs are ``[N,C,T]``. Missing, occluded, or weakly excited observations
    must have zero weight and consequently do not pull the signature.
    """
    speed = torch.as_tensor(inferred_speed).detach().cpu().double()
    weight = torch.as_tensor(weights).detach().cpu().double()
    if speed.shape != weight.shape or speed.ndim != 3:
        raise ValueError("inferred_speed and weights must both be [N,C,T]")
    usable = torch.isfinite(speed) & torch.isfinite(weight) & (weight > 0)
    safe_weight = torch.where(usable, weight, torch.zeros_like(weight))
    safe_speed = torch.where(usable, speed, torch.zeros_like(speed))
    total = safe_weight.sum(dim=1)
    signature = (safe_weight * safe_speed).sum(dim=1) / total.clamp_min(float(eps))
    uncertainty = estimate_signature_uncertainty(safe_speed, safe_weight, dim=1, eps=eps)
    valid = total > 0
    signature = torch.where(valid, signature, torch.zeros_like(signature))
    return {
        "signature": signature,
        "weight": total,
        "valid": valid,
        "uncertainty": uncertainty,
        "camera_observation_count": usable.sum(dim=1),
    }


@torch.no_grad()
def compute_part_speed_signature(interval_speeds):
    """Return the arbitrary signed per-interval coordinate-speed trajectory."""
    speed = torch.as_tensor(interval_speeds).detach().cpu().double().reshape(-1)
    return {"signature": speed, "valid": torch.isfinite(speed)}


def _huber(value: torch.Tensor, delta: float):
    absolute = value.abs()
    threshold = max(float(delta), 1.0e-8)
    return torch.where(
        absolute <= threshold,
        0.5 * absolute.square(),
        threshold * (absolute - 0.5 * threshold),
    )


@torch.no_grad()
def compute_gaussian_part_speed_distance(
    gaussian_signature,
    part_signature,
    *,
    weights=None,
    uncertainty=None,
    huber_delta: float = 1.0,
    eps: float = 1.0e-8,
):
    """Robust full-trajectory distance; stops and sign reversals are retained."""
    gaussian = torch.as_tensor(gaussian_signature).double()
    part = torch.as_tensor(part_signature, device=gaussian.device).double()
    if part.ndim != 1 or gaussian.shape[-1] != part.shape[0]:
        raise ValueError("part_signature must match the last Gaussian time dimension")
    weight = torch.ones_like(gaussian) if weights is None else torch.as_tensor(weights, device=gaussian.device).double()
    if weight.shape != gaussian.shape:
        raise ValueError("weights must match gaussian_signature")
    if uncertainty is not None:
        scale = torch.as_tensor(uncertainty, device=gaussian.device).double()
        if scale.shape != gaussian.shape:
            raise ValueError("uncertainty must match gaussian_signature")
        weight = weight / (1.0 + torch.where(torch.isfinite(scale), scale, torch.full_like(scale, 1.0e6)).square())
    valid = torch.isfinite(gaussian) & torch.isfinite(weight) & torch.isfinite(part)[None]
    effective = torch.where(valid, weight.clamp_min(0.0), torch.zeros_like(weight))
    residual = torch.where(valid, gaussian - part, torch.zeros_like(gaussian))
    distance = (effective * _huber(residual, huber_delta)).sum(dim=-1) / effective.sum(dim=-1).clamp_min(float(eps))
    informative = effective.sum(dim=-1) > 0
    return {"distance": distance, "weight": effective.sum(dim=-1), "informative": informative}


@torch.no_grad()
def compute_pairwise_motion_compatibility(
    signatures,
    pairs,
    *,
    weights=None,
    uncertainty=None,
    huber_delta: float = 1.0,
    eps: float = 1.0e-8,
):
    """Compare normalized temporal signatures on graph edges, never raw speed.

    ``signatures`` can be ``[N,T]`` or candidate-specific ``[N,K,T]``. The
    returned ``distance`` is the best compatible candidate distance per edge.
    """
    signature = torch.as_tensor(signatures).detach().cpu().double()
    edge = torch.as_tensor(pairs).detach().cpu().long()
    if signature.ndim == 2:
        signature = signature[:, None]
    if signature.ndim != 3 or edge.ndim != 2 or edge.shape[-1] != 2:
        raise ValueError("signatures must be [N,T] or [N,K,T], pairs [E,2]")
    weight = torch.ones_like(signature) if weights is None else torch.as_tensor(weights).detach().cpu().double()
    if weight.ndim == 2:
        weight = weight[:, None]
    if weight.shape != signature.shape:
        weight = weight.expand_as(signature)
    pair_weight = torch.minimum(weight[edge[:, 0]], weight[edge[:, 1]])
    if uncertainty is not None:
        unc = torch.as_tensor(uncertainty).detach().cpu().double()
        if unc.ndim == 2:
            unc = unc[:, None]
        unc = unc.expand_as(signature)
        pair_weight = pair_weight / (1.0 + unc[edge[:, 0]].square() + unc[edge[:, 1]].square())
    residual = signature[edge[:, 0]] - signature[edge[:, 1]]
    valid = torch.isfinite(residual) & torch.isfinite(pair_weight)
    effective = torch.where(valid, pair_weight.clamp_min(0.0), torch.zeros_like(pair_weight))
    per_candidate = (effective * _huber(torch.where(valid, residual, torch.zeros_like(residual)), huber_delta)).sum(-1) / effective.sum(-1).clamp_min(float(eps))
    informative = effective.sum(-1) > 0
    masked = torch.where(informative, per_candidate, torch.full_like(per_candidate, float("inf")))
    distance = masked.amin(dim=-1)
    distance = torch.where(torch.isfinite(distance), distance, torch.zeros_like(distance))
    return {"distance": distance, "per_part_distance": per_candidate, "informative": informative.any(dim=-1)}
