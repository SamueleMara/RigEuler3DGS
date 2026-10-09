"""Neural endpoint flow matching prior.

IMPORTANT:
This module estimates only a static/dynamic prior.  It must remain independent
of rigid Killing estimation, joints, part labels, and ground-truth rigid labels.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch import nn
import torch.nn.functional as F
from tqdm import tqdm

from utils.dynamic_prior import DynamicPrior
from utils.endpoint_transport_prior import EndpointTransportPriorResult, local_source_scale


class EndpointFlowMLP(nn.Module):
    def __init__(self, hidden_dim: int = 128, layers: int = 4, posenc_frequencies: int = 6):
        super().__init__()
        self.posenc_frequencies = max(0, int(posenc_frequencies))
        in_dim = 3 + 1 + 2 * self.posenc_frequencies * 3
        blocks = []
        width = max(8, int(hidden_dim))
        depth = max(1, int(layers))
        last = in_dim
        for _ in range(depth):
            blocks.append(nn.Linear(last, width))
            blocks.append(nn.SiLU())
            last = width
        blocks.append(nn.Linear(last, 3))
        self.net = nn.Sequential(*blocks)

    def _encode_xyz(self, xyz: torch.Tensor) -> torch.Tensor:
        parts = [xyz]
        for i in range(self.posenc_frequencies):
            freq = float(2**i)
            parts.append(torch.sin(freq * xyz))
            parts.append(torch.cos(freq * xyz))
        return torch.cat(parts, dim=-1)

    def forward(self, xyz: torch.Tensor, tau: torch.Tensor) -> torch.Tensor:
        if tau.ndim == 0:
            tau = tau.expand(xyz.shape[0])
        tau = tau.reshape(-1, 1).to(device=xyz.device, dtype=xyz.dtype)
        return self.net(torch.cat((self._encode_xyz(xyz), tau), dim=-1))


@dataclass
class FlowMatchingConfig:
    hidden_dim: int = 128
    layers: int = 4
    lr: float = 1.0e-3
    iters: int = 1000
    batch_size: int = 2048
    identity_weight: float = 0.01
    smooth_weight: float = 0.0
    integration_steps: int = 16
    integrator: str = "euler"
    prior_prob_bias: float = 1.0
    prior_prob_temperature: float = 0.35
    seed: int = 20260904


def _sample_targets(result: EndpointTransportPriorResult, source_ids: torch.Tensor) -> torch.Tensor:
    if result.target_indices is None or result.transport_weights is None:
        raise ValueError("Endpoint transport result must include target_indices and transport_weights")
    idx = result.target_indices[source_ids.cpu()].to(source_ids.device)
    weights = result.transport_weights[source_ids.cpu()].to(source_ids.device).clamp_min(0.0)
    weights = weights / weights.sum(dim=1, keepdim=True).clamp_min(1.0e-8)
    draw = torch.multinomial(weights, num_samples=1).squeeze(1)
    return idx[torch.arange(idx.shape[0], device=idx.device), draw]


def train_endpoint_flow_matching(
    source_xyz: torch.Tensor,
    target_xyz: torch.Tensor,
    transport_result: EndpointTransportPriorResult,
    cfg: FlowMatchingConfig,
    *,
    device: str | torch.device | None = None,
) -> tuple[EndpointFlowMLP, list[dict[str, float]]]:
    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    source = torch.as_tensor(source_xyz).detach().float().to(device)
    target = torch.as_tensor(target_xyz).detach().float().to(device)
    confidence = transport_result.prior.confidence.to(device=device, dtype=source.dtype)
    valid = transport_result.prior.valid.to(device=device)
    if source.ndim != 2 or source.shape[-1] != 3 or target.ndim != 2 or target.shape[-1] != 3:
        raise ValueError("source_xyz and target_xyz must be [N,3] and [M,3]")
    model = EndpointFlowMLP(cfg.hidden_dim, cfg.layers).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=max(float(cfg.lr), 1.0e-8))
    generator = torch.Generator(device=device).manual_seed(int(cfg.seed))
    sample_weight = torch.where(valid, confidence.clamp_min(1.0e-4), torch.zeros_like(confidence))
    if float(sample_weight.sum()) <= 0.0:
        sample_weight = torch.ones(source.shape[0], device=device)
    sample_weight = sample_weight / sample_weight.sum()
    logs: list[dict[str, float]] = []
    progress = tqdm(range(max(1, int(cfg.iters))), desc="Endpoint flow matching prior", unit="iter")
    for iteration in progress:
        batch = min(int(cfg.batch_size), int(source.shape[0]))
        source_ids = torch.multinomial(sample_weight, batch, replacement=True, generator=generator)
        target_ids = _sample_targets(transport_result, source_ids)
        x0 = source[source_ids]
        x1 = target[target_ids]
        tau = torch.rand(batch, device=device, generator=generator)
        xt = x0 + tau[:, None] * (x1 - x0)
        target_velocity = x1 - x0
        pred = model(xt, tau)
        data = F.mse_loss(pred, target_velocity)
        identity = (model(x0, torch.zeros(batch, device=device)).square().sum(dim=1) * (1.0 - confidence[source_ids])).mean()
        loss = data + float(cfg.identity_weight) * identity
        if float(cfg.smooth_weight) > 0.0 and batch > 1:
            jitter = 0.002 * torch.randn_like(xt)
            smooth = F.mse_loss(model(xt + jitter, tau), pred.detach())
            loss = loss + float(cfg.smooth_weight) * smooth
        else:
            smooth = torch.zeros((), device=device)
        if not bool(torch.isfinite(loss)):
            raise RuntimeError("Endpoint flow matching produced a non-finite loss")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        if iteration % 25 == 0 or iteration == int(cfg.iters) - 1:
            row = {
                "iteration": int(iteration),
                "loss": float(loss.detach()),
                "data": float(data.detach()),
                "identity": float(identity.detach()),
                "smooth": float(smooth.detach()),
            }
            logs.append(row)
            progress.set_postfix(loss=f"{row['loss']:.4g}", data=f"{row['data']:.4g}")
    return model, logs


@torch.no_grad()
def integrate_flow(model: EndpointFlowMLP, source_xyz: torch.Tensor, cfg: FlowMatchingConfig) -> torch.Tensor:
    device = next(model.parameters()).device
    x = torch.as_tensor(source_xyz).detach().float().to(device)
    steps = max(1, int(cfg.integration_steps))
    dt = 1.0 / float(steps)
    for step in range(steps):
        tau = torch.full((x.shape[0],), float(step) / float(steps), device=device, dtype=x.dtype)
        if str(cfg.integrator).lower() == "rk4":
            k1 = model(x, tau)
            k2 = model(x + 0.5 * dt * k1, (tau + 0.5 * dt).clamp(0.0, 1.0))
            k3 = model(x + 0.5 * dt * k2, (tau + 0.5 * dt).clamp(0.0, 1.0))
            k4 = model(x + dt * k3, (tau + dt).clamp(0.0, 1.0))
            x = x + dt * (k1 + 2.0 * k2 + 2.0 * k3 + k4) / 6.0
        else:
            x = x + dt * model(x, tau)
    return x.detach().cpu()


def build_dynamic_prior_from_flow(
    source_xyz: torch.Tensor,
    integrated_endpoint: torch.Tensor,
    gaussian_ids: torch.Tensor,
    cfg: FlowMatchingConfig,
    *,
    confidence: torch.Tensor | None = None,
    valid: torch.Tensor | None = None,
) -> DynamicPrior:
    source = torch.as_tensor(source_xyz).detach().cpu().float()
    integrated = torch.as_tensor(integrated_endpoint).detach().cpu().float()
    if source.shape != integrated.shape:
        raise ValueError("source_xyz and integrated_endpoint must have identical shape")
    scale = local_source_scale(source, chunk_size=1024).cpu()
    displacement = integrated - source
    raw = torch.linalg.norm(displacement, dim=1) / scale.clamp_min(1.0e-6)
    probability = torch.sigmoid(
        (raw - float(cfg.prior_prob_bias))
        / max(float(cfg.prior_prob_temperature), 1.0e-6)
    )
    ids = torch.as_tensor(gaussian_ids).detach().cpu().long().reshape(-1)
    if ids.numel() != source.shape[0]:
        raise ValueError("gaussian_ids must match source_xyz length")
    probability_margin = (2.0 * (probability - 0.5).abs()).clamp(0.0, 1.0)
    if confidence is None:
        confidence_t = probability_margin
    else:
        confidence_t = torch.as_tensor(confidence).detach().cpu().float().reshape(-1)
        if confidence_t.numel() != raw.numel():
            raise ValueError("confidence must match source_xyz length")
        confidence_t = torch.maximum(confidence_t.clamp(0.0, 1.0), probability_margin)
    if valid is None:
        valid_t = torch.isfinite(raw) & torch.isfinite(probability)
    else:
        valid_t = torch.as_tensor(valid).detach().cpu().bool().reshape(-1)
        if valid_t.numel() != raw.numel():
            raise ValueError("valid must match source_xyz length")
        valid_t = valid_t & torch.isfinite(raw) & torch.isfinite(probability)
    confidence_t = torch.where(valid_t, confidence_t, torch.zeros_like(confidence_t))
    return DynamicPrior(
        probability=probability,
        raw_score=raw,
        valid=valid_t,
        confidence=confidence_t,
        method="flow_matching",
        gaussian_ids=ids,
        metadata={
            "ground_truth_used_as_algorithm_input": False,
            "uses_killing_solver": False,
            "integrator": str(cfg.integrator),
            "integration_steps": int(cfg.integration_steps),
            "hidden_dim": int(cfg.hidden_dim),
            "layers": int(cfg.layers),
            "confidence_model": "max(endpoint_confidence, probability_margin)",
        },
    ).validate()


def flow_checkpoint_payload(model: EndpointFlowMLP, cfg: FlowMatchingConfig, logs: list[dict[str, float]]) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "state_dict": model.state_dict(),
        "config": cfg.__dict__,
        "training_log": logs,
        "ground_truth_used_as_algorithm_input": False,
        "uses_killing_solver": False,
    }
