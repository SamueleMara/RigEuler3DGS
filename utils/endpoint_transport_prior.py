"""Geometry-only endpoint transport prior.

IMPORTANT:
This module estimates only a static/dynamic prior.  It must remain independent
of rigid Killing estimation, joints, part labels, and ground-truth rigid labels.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from utils.dynamic_prior import DynamicPrior


@dataclass
class EndpointTransportConfig:
    knn: int = 16
    temperature: float = 0.02
    robust_delta: float = 0.02
    motion_scale_mode: str = "local_knn"
    confidence_distance_scale: float = 0.0
    confidence_entropy_scale: float = 1.0
    chunk_size: int = 1024
    prior_prob_bias: float = 1.0
    prior_prob_temperature: float = 0.35
    local_scale_knn: int = 8


@dataclass
class EndpointTransportPriorResult:
    prior: DynamicPrior
    barycentric_target: torch.Tensor
    displacement: torch.Tensor
    nearest_distance: torch.Tensor
    entropy: torch.Tensor
    target_indices: torch.Tensor | None
    transport_weights: torch.Tensor | None
    motion_scale: torch.Tensor


def _softplus_huber_distance(distance: torch.Tensor, delta: float) -> torch.Tensor:
    delta = max(float(delta), 1.0e-8)
    return delta * delta * (torch.sqrt(1.0 + (distance / delta).square()) - 1.0)


def _chunked_target_knn(
    source_xyz: torch.Tensor,
    target_xyz: torch.Tensor,
    *,
    knn: int,
    chunk_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    n = int(source_xyz.shape[0])
    k = min(max(1, int(knn)), int(target_xyz.shape[0]))
    values = []
    indices = []
    for start in range(0, n, max(1, int(chunk_size))):
        chunk = source_xyz[start : start + int(chunk_size)]
        d2 = torch.cdist(chunk, target_xyz).square()
        val, idx = torch.topk(d2, k=k, dim=1, largest=False, sorted=True)
        values.append(val)
        indices.append(idx)
    return torch.cat(values, 0), torch.cat(indices, 0)


def local_source_scale(
    source_xyz: torch.Tensor,
    *,
    knn: int = 8,
    chunk_size: int = 1024,
) -> torch.Tensor:
    n = int(source_xyz.shape[0])
    if n <= 1:
        return torch.ones(n, device=source_xyz.device, dtype=source_xyz.dtype)
    k = min(max(2, int(knn) + 1), n)
    medians = []
    for start in range(0, n, max(1, int(chunk_size))):
        chunk = source_xyz[start : start + int(chunk_size)]
        d = torch.cdist(chunk, source_xyz)
        val = torch.topk(d, k=k, dim=1, largest=False, sorted=True).values[:, 1:]
        medians.append(val.median(dim=1).values)
    scale = torch.cat(medians, 0)
    global_scale = scale[torch.isfinite(scale) & (scale > 0)]
    fallback = global_scale.median() if global_scale.numel() else torch.tensor(1.0, device=source_xyz.device, dtype=source_xyz.dtype)
    return torch.where(scale > 1.0e-8, scale, fallback).clamp_min(1.0e-6)


class EndpointTransportPrior:
    def __init__(self, target_xyz: torch.Tensor, target_meta: dict[str, Any] | None, cfg: EndpointTransportConfig):
        target = torch.as_tensor(target_xyz).detach().float()
        if target.ndim != 2 or target.shape[-1] != 3 or int(target.shape[0]) == 0:
            raise ValueError("target_xyz must be non-empty [M,3]")
        self.target_xyz = target
        self.target_meta = dict(target_meta or {})
        self.cfg = cfg

    def compute(self, source_xyz: torch.Tensor, gaussian_ids: torch.Tensor | None = None) -> EndpointTransportPriorResult:
        source = torch.as_tensor(source_xyz).detach().float()
        if source.ndim != 2 or source.shape[-1] != 3:
            raise ValueError("source_xyz must be [N,3]")
        device = source.device
        target = self.target_xyz.to(device=device, dtype=source.dtype)
        ids = (
            torch.arange(source.shape[0], dtype=torch.long)
            if gaussian_ids is None
            else torch.as_tensor(gaussian_ids).detach().cpu().long().reshape(-1)
        )
        if ids.numel() != source.shape[0]:
            raise ValueError("gaussian_ids must match source_xyz length")

        dist2, idx = _chunked_target_knn(
            source,
            target,
            knn=int(self.cfg.knn),
            chunk_size=int(self.cfg.chunk_size),
        )
        distance = torch.sqrt(dist2.clamp_min(0.0))
        cost = _softplus_huber_distance(distance, float(self.cfg.robust_delta))
        weights = torch.softmax(-cost / max(float(self.cfg.temperature), 1.0e-8), dim=1)
        gathered = target[idx.reshape(-1)].reshape(idx.shape[0], idx.shape[1], 3)
        barycentric = (weights[..., None] * gathered).sum(dim=1)
        displacement = barycentric - source
        nearest = distance[:, 0]
        denom = torch.log(torch.tensor(float(max(2, idx.shape[1])), device=device, dtype=source.dtype))
        entropy = (-(weights * torch.log(weights.clamp_min(1.0e-12))).sum(dim=1) / denom).clamp(0.0, 1.0)

        if self.cfg.motion_scale_mode == "global_nn":
            finite = nearest[torch.isfinite(nearest)]
            scale_value = finite.median() if finite.numel() else torch.tensor(1.0, device=device, dtype=source.dtype)
            scale = torch.full_like(nearest, float(scale_value.clamp_min(1.0e-6)))
        elif self.cfg.motion_scale_mode == "gaussian_scale" and self.target_meta.get("source_scale") is not None:
            source_scale = torch.as_tensor(self.target_meta["source_scale"], device=device, dtype=source.dtype).reshape(-1)
            if source_scale.numel() != source.shape[0]:
                raise ValueError("target_meta['source_scale'] must match source_xyz length")
            scale = source_scale.clamp_min(1.0e-6)
        else:
            scale = local_source_scale(
                source,
                knn=int(self.cfg.local_scale_knn),
                chunk_size=int(self.cfg.chunk_size),
            )
        raw = torch.linalg.norm(displacement, dim=1) / scale.clamp_min(1.0e-6)
        probability = torch.sigmoid(
            (raw - float(self.cfg.prior_prob_bias))
            / max(float(self.cfg.prior_prob_temperature), 1.0e-6)
        )

        distance_scale = float(self.cfg.confidence_distance_scale)
        if distance_scale <= 0.0:
            finite_scale = scale[torch.isfinite(scale) & (scale > 0)]
            distance_scale_t = finite_scale.median() if finite_scale.numel() else torch.tensor(1.0, device=device, dtype=source.dtype)
        else:
            distance_scale_t = torch.tensor(distance_scale, device=device, dtype=source.dtype)
        distance_confidence = torch.exp(-nearest / distance_scale_t.clamp_min(1.0e-6))
        entropy_power = max(0.0, float(self.cfg.confidence_entropy_scale))
        entropy_confidence = (1.0 - entropy).clamp(0.0, 1.0)
        entropy_term = (
            entropy_confidence.pow(entropy_power)
            if entropy_power > 0.0
            else torch.ones_like(entropy_confidence)
        )
        probability_margin = (2.0 * (probability - 0.5).abs()).clamp(0.0, 1.0)
        confidence = (
            distance_confidence * torch.maximum(entropy_term, 0.5 * probability_margin)
        ).clamp(0.0, 1.0)
        valid = torch.isfinite(raw) & torch.isfinite(confidence) & (target.shape[0] > 0)

        prior = DynamicPrior(
            probability=probability.detach().cpu(),
            raw_score=raw.detach().cpu(),
            valid=valid.detach().cpu(),
            confidence=confidence.detach().cpu(),
            method="endpoint_transport",
            gaussian_ids=ids,
            metadata={
                "ground_truth_used_as_algorithm_input": False,
                "uses_killing_solver": False,
                "target_gaussians": int(target.shape[0]),
                "knn": int(idx.shape[1]),
                "temperature": float(self.cfg.temperature),
                "robust_delta": float(self.cfg.robust_delta),
                "motion_scale_mode": str(self.cfg.motion_scale_mode),
                "confidence_model": "distance * max(entropy_term, 0.5 * probability_margin)",
            },
        ).validate()
        return EndpointTransportPriorResult(
            prior=prior,
            barycentric_target=barycentric.detach().cpu(),
            displacement=displacement.detach().cpu(),
            nearest_distance=nearest.detach().cpu(),
            entropy=entropy.detach().cpu(),
            target_indices=idx.detach().cpu(),
            transport_weights=weights.detach().cpu(),
            motion_scale=scale.detach().cpu(),
        )


def save_endpoint_transport_result(result: EndpointTransportPriorResult, output_dir: str | Path) -> None:
    output = Path(output_dir).expanduser()
    output.mkdir(parents=True, exist_ok=True)
    torch.save(result.displacement, output / "displacement.pt")
    torch.save(result.barycentric_target, output / "barycentric_target.pt")
    torch.save(result.entropy, output / "transport_entropy.pt")
    torch.save(result.nearest_distance, output / "nearest_distance.pt")
    torch.save(result.motion_scale, output / "motion_scale.pt")
    if result.target_indices is not None:
        torch.save(result.target_indices, output / "target_indices.pt")
    if result.transport_weights is not None:
        torch.save(result.transport_weights, output / "transport_weights.pt")
