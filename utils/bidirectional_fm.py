"""Fresh bidirectional Flow Matching for endpoint correspondence probes.

This is intentionally independent of the older endpoint-transport/progressive
matcher.  It trains one MLP for 0->T and one independent MLP for T->0 from a
plain local soft target sampler, then exposes dynamic probability separately
from the continuous endpoint proposal.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
from torch import nn
import torch.nn.functional as F
from tqdm import tqdm


@dataclass
class SoftTransport:
    target_indices: torch.Tensor
    weights: torch.Tensor
    barycentric_target: torch.Tensor
    displacement: torch.Tensor
    nearest_distance: torch.Tensor
    entropy: torch.Tensor
    local_scale: torch.Tensor
    probability: torch.Tensor
    confidence: torch.Tensor
    valid: torch.Tensor


@dataclass
class FlowTrainConfig:
    knn: int = 16
    temperature: float = 0.02
    robust_delta: float = 0.02
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
    chunk_size: int = 1024
    local_scale_knn: int = 8
    seed: int = 20260914


@dataclass
class FlowResult:
    endpoint: torch.Tensor
    displacement: torch.Tensor
    probability: torch.Tensor
    confidence: torch.Tensor
    valid: torch.Tensor
    local_scale: torch.Tensor
    model: nn.Module
    transport: SoftTransport
    training_log: list[dict[str, float]]


class FlowMLP(nn.Module):
    def __init__(self, hidden_dim: int = 128, layers: int = 4, posenc_frequencies: int = 6):
        super().__init__()
        self.posenc_frequencies = max(0, int(posenc_frequencies))
        in_dim = 3 + 1 + 2 * self.posenc_frequencies * 3
        width = max(16, int(hidden_dim))
        blocks: list[nn.Module] = []
        last = in_dim
        for _ in range(max(1, int(layers))):
            blocks += [nn.Linear(last, width), nn.SiLU()]
            last = width
        blocks.append(nn.Linear(last, 3))
        self.net = nn.Sequential(*blocks)

    def _encode_xyz(self, xyz: torch.Tensor) -> torch.Tensor:
        parts = [xyz]
        for i in range(self.posenc_frequencies):
            freq = float(2**i)
            parts += [torch.sin(freq * xyz), torch.cos(freq * xyz)]
        return torch.cat(parts, dim=-1)

    def forward(self, xyz: torch.Tensor, tau: torch.Tensor) -> torch.Tensor:
        tau = torch.as_tensor(tau, device=xyz.device, dtype=xyz.dtype)
        if tau.ndim == 0:
            tau = tau.expand(xyz.shape[0])
        return self.net(torch.cat((self._encode_xyz(xyz), tau.reshape(-1, 1)), dim=-1))


def chunked_knn(query: torch.Tensor, target: torch.Tensor, k: int, *, chunk_size: int = 1024) -> tuple[torch.Tensor, torch.Tensor]:
    query = torch.as_tensor(query).detach().float()
    target = torch.as_tensor(target, device=query.device, dtype=query.dtype).detach()
    k = min(max(1, int(k)), int(target.shape[0]))
    values = []
    indices = []
    for start in range(0, int(query.shape[0]), max(1, int(chunk_size))):
        d2 = torch.cdist(query[start:start + int(chunk_size)], target).square()
        val, idx = torch.topk(d2, k=k, dim=1, largest=False, sorted=True)
        values.append(val)
        indices.append(idx)
    return torch.cat(values, 0), torch.cat(indices, 0)


def local_spacing(points: torch.Tensor, *, k: int = 8, chunk_size: int = 1024) -> torch.Tensor:
    points = torch.as_tensor(points).detach().float()
    n = int(points.shape[0])
    if n <= 1:
        return torch.ones(n, device=points.device, dtype=points.dtype)
    kk = min(max(2, int(k) + 1), n)
    chunks = []
    for start in range(0, n, max(1, int(chunk_size))):
        d = torch.cdist(points[start:start + int(chunk_size)], points)
        chunks.append(torch.topk(d, k=kk, dim=1, largest=False, sorted=True).values[:, 1:].median(1).values)
    scale = torch.cat(chunks, 0)
    finite = scale[torch.isfinite(scale) & (scale > 0)]
    fallback = finite.median() if finite.numel() else torch.tensor(1.0, device=points.device, dtype=points.dtype)
    return torch.where(scale > 1.0e-8, scale, fallback).clamp_min(1.0e-6)


def target_median_spacing(points: torch.Tensor, *, chunk_size: int = 1024) -> float:
    return float(local_spacing(points, k=1, chunk_size=chunk_size).median().detach().cpu())


def _soft_huber(distance: torch.Tensor, delta: float) -> torch.Tensor:
    delta = max(float(delta), 1.0e-8)
    return delta * delta * (torch.sqrt(1.0 + (distance / delta).square()) - 1.0)


@torch.no_grad()
def build_soft_transport(source_xyz: torch.Tensor, target_xyz: torch.Tensor, cfg: FlowTrainConfig) -> SoftTransport:
    source = torch.as_tensor(source_xyz).detach().float()
    target = torch.as_tensor(target_xyz, device=source.device, dtype=source.dtype).detach()
    dist2, idx = chunked_knn(source, target, cfg.knn, chunk_size=cfg.chunk_size)
    dist = dist2.clamp_min(0.0).sqrt()
    cost = _soft_huber(dist, cfg.robust_delta)
    weights = torch.softmax(-cost / max(float(cfg.temperature), 1.0e-8), dim=1)
    gathered = target[idx.reshape(-1)].reshape(idx.shape[0], idx.shape[1], 3)
    bary = (weights[..., None] * gathered).sum(1)
    displacement = bary - source
    scale = local_spacing(source, k=cfg.local_scale_knn, chunk_size=cfg.chunk_size)
    raw = torch.linalg.norm(displacement, dim=-1) / scale.clamp_min(1.0e-8)
    probability = torch.sigmoid((raw - float(cfg.prior_prob_bias)) / max(float(cfg.prior_prob_temperature), 1.0e-6))
    entropy = -(weights * torch.log(weights.clamp_min(1.0e-12))).sum(1)
    entropy = entropy / torch.log(torch.tensor(float(max(2, idx.shape[1])), device=source.device, dtype=source.dtype))
    nearest = dist[:, 0]
    distance_scale = scale[torch.isfinite(scale) & (scale > 0)].median() if bool((torch.isfinite(scale) & (scale > 0)).any()) else source.new_tensor(1.0)
    probability_margin = (2.0 * (probability - 0.5).abs()).clamp(0.0, 1.0)
    confidence = (torch.exp(-nearest / distance_scale.clamp_min(1.0e-8)) * torch.maximum(1.0 - entropy, 0.5 * probability_margin)).clamp(0.0, 1.0)
    valid = torch.isfinite(raw) & torch.isfinite(confidence)
    return SoftTransport(
        target_indices=idx.detach().cpu(),
        weights=weights.detach().cpu(),
        barycentric_target=bary.detach().cpu(),
        displacement=displacement.detach().cpu(),
        nearest_distance=nearest.detach().cpu(),
        entropy=entropy.detach().cpu(),
        local_scale=scale.detach().cpu(),
        probability=probability.detach().cpu(),
        confidence=confidence.detach().cpu(),
        valid=valid.detach().cpu(),
    )


def _sample_targets(transport: SoftTransport, source_ids: torch.Tensor) -> torch.Tensor:
    idx = transport.target_indices[source_ids.detach().cpu()].to(source_ids.device)
    weights = transport.weights[source_ids.detach().cpu()].to(source_ids.device).clamp_min(0.0)
    weights = weights / weights.sum(1, keepdim=True).clamp_min(1.0e-8)
    draw = torch.multinomial(weights, num_samples=1).squeeze(1)
    return idx[torch.arange(idx.shape[0], device=idx.device), draw]


def train_flow_direction(
    source_xyz: torch.Tensor,
    target_xyz: torch.Tensor,
    cfg: FlowTrainConfig,
    *,
    direction_name: str,
    device: str | torch.device = "cuda",
) -> FlowResult:
    device = torch.device(device if torch.cuda.is_available() or str(device) == "cpu" else "cpu")
    source_cpu = torch.as_tensor(source_xyz).detach().float().cpu()
    target_cpu = torch.as_tensor(target_xyz).detach().float().cpu()
    transport = build_soft_transport(source_cpu.to(device), target_cpu.to(device), cfg)
    source = source_cpu.to(device)
    target = target_cpu.to(device)
    confidence = transport.confidence.to(device)
    valid = transport.valid.to(device)
    sample_weight = torch.where(valid, confidence.clamp_min(1.0e-4), torch.zeros_like(confidence))
    if float(sample_weight.sum()) <= 0:
        sample_weight = torch.ones_like(sample_weight)
    sample_weight = sample_weight / sample_weight.sum()
    model = FlowMLP(cfg.hidden_dim, cfg.layers).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=max(float(cfg.lr), 1.0e-8))
    generator = torch.Generator(device=device).manual_seed(int(cfg.seed))
    logs: list[dict[str, float]] = []
    progress = tqdm(range(max(1, int(cfg.iters))), desc=f"{direction_name} FM", unit="iter")
    for iteration in progress:
        batch = min(int(cfg.batch_size), int(source.shape[0]))
        ids = torch.multinomial(sample_weight, batch, replacement=True, generator=generator)
        target_ids = _sample_targets(transport, ids)
        x0 = source[ids]
        x1 = target[target_ids]
        tau = torch.rand(batch, device=device, generator=generator)
        xt = x0 + tau[:, None] * (x1 - x0)
        target_velocity = x1 - x0
        pred = model(xt, tau)
        data = F.mse_loss(pred, target_velocity)
        identity = (model(x0, torch.zeros(batch, device=device)).square().sum(-1) * (1.0 - confidence[ids])).mean()
        loss = data + float(cfg.identity_weight) * identity
        if float(cfg.smooth_weight) > 0:
            jitter = 0.002 * torch.randn_like(xt)
            smooth = F.mse_loss(model(xt + jitter, tau), pred.detach())
            loss = loss + float(cfg.smooth_weight) * smooth
        else:
            smooth = torch.zeros((), device=device)
        if not bool(torch.isfinite(loss)):
            raise RuntimeError(f"{direction_name} FM produced a non-finite loss")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        if iteration % 25 == 0 or iteration == int(cfg.iters) - 1:
            row = {
                "iteration": int(iteration),
                "loss": float(loss.detach().cpu()),
                "data": float(data.detach().cpu()),
                "identity": float(identity.detach().cpu()),
                "smooth": float(smooth.detach().cpu()),
            }
            logs.append(row)
            progress.set_postfix(loss=f"{row['loss']:.4g}", data=f"{row['data']:.4g}")
    endpoint = integrate_flow(model, source_cpu, cfg, device=device)
    displacement = endpoint - source_cpu
    scale = local_spacing(source_cpu, k=cfg.local_scale_knn, chunk_size=cfg.chunk_size).cpu()
    raw = torch.linalg.norm(displacement, dim=-1) / scale.clamp_min(1.0e-8)
    probability = torch.sigmoid((raw - float(cfg.prior_prob_bias)) / max(float(cfg.prior_prob_temperature), 1.0e-6))
    confidence = torch.maximum(transport.confidence, (2.0 * (probability - 0.5).abs()).clamp(0.0, 1.0))
    valid = transport.valid & torch.isfinite(probability)
    return FlowResult(
        endpoint=endpoint.cpu(),
        displacement=displacement.cpu(),
        probability=probability.cpu(),
        confidence=confidence.cpu(),
        valid=valid.cpu(),
        local_scale=scale.cpu(),
        model=model,
        transport=transport,
        training_log=logs,
    )


@torch.no_grad()
def integrate_flow(model: nn.Module, points: torch.Tensor, cfg: FlowTrainConfig, *, device: str | torch.device | None = None) -> torch.Tensor:
    device = torch.device(device or next(model.parameters()).device)
    model = model.to(device).eval()
    x = torch.as_tensor(points).detach().float().to(device)
    steps = max(1, int(cfg.integration_steps))
    dt = 1.0 / float(steps)
    for step in range(steps):
        tau = torch.full((x.shape[0],), float(step) / float(steps), device=device)
        if str(cfg.integrator).lower() == "rk4":
            k1 = model(x, tau)
            k2 = model(x + 0.5 * dt * k1, (tau + 0.5 * dt).clamp(0.0, 1.0))
            k3 = model(x + 0.5 * dt * k2, (tau + 0.5 * dt).clamp(0.0, 1.0))
            k4 = model(x + dt * k3, (tau + dt).clamp(0.0, 1.0))
            x = x + dt * (k1 + 2.0 * k2 + 2.0 * k3 + k4) / 6.0
        else:
            x = x + dt * model(x, tau)
    return x.detach().cpu()


def save_flow_result(result: FlowResult, output_dir: str | Path, cfg: FlowTrainConfig) -> None:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    torch.save(result.endpoint.cpu(), output / "endpoint.pt")
    torch.save(result.displacement.cpu(), output / "displacement.pt")
    torch.save(result.probability.cpu(), output / "probability.pt")
    torch.save(result.confidence.cpu(), output / "confidence.pt")
    torch.save(result.valid.cpu(), output / "valid.pt")
    torch.save(result.local_scale.cpu(), output / "local_scale.pt")
    torch.save(result.transport.target_indices.cpu(), output / "initial_transport_target_indices.pt")
    torch.save(result.transport.weights.cpu(), output / "initial_transport_weights.pt")
    payload = {
        "schema_version": 1,
        "config": asdict(cfg),
        "training_log": result.training_log,
        "ground_truth_used_as_algorithm_input": False,
    }
    torch.save({"state_dict": result.model.cpu().state_dict(), **payload}, output / "checkpoint.pt")
