"""Small displacement diffusion scorer for bidirectional correspondence candidates."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

import torch
from torch import nn
import torch.nn.functional as F
from tqdm import tqdm


@dataclass
class DisplacementDiffusionConfig:
    hidden_dim: int = 128
    layers: int = 4
    iters: int = 500
    batch_size: int = 4096
    lr: float = 1.0e-3
    trials: int = 4
    schedule: str = "cosine"
    posenc_frequencies: int = 4
    seed: int = 20260914
    chunk_size: int = 65536


class DisplacementDenoiser(nn.Module):
    def __init__(self, cfg: DisplacementDiffusionConfig):
        super().__init__()
        self.posenc_frequencies = max(0, int(cfg.posenc_frequencies))
        position_dim = 3 + 2 * self.posenc_frequencies * 3
        in_dim = 3 + position_dim + 1 + 1 + 1 + 1 + 1 + 3
        width = max(16, int(cfg.hidden_dim))
        layers: list[nn.Module] = []
        last = in_dim
        for _ in range(max(1, int(cfg.layers))):
            layers += [nn.Linear(last, width), nn.SiLU()]
            last = width
        layers.append(nn.Linear(last, 3))
        self.net = nn.Sequential(*layers)

    def _encode_position(self, xyz: torch.Tensor) -> torch.Tensor:
        parts = [xyz]
        for i in range(self.posenc_frequencies):
            freq = float(2**i)
            parts += [torch.sin(freq * xyz), torch.cos(freq * xyz)]
        return torch.cat(parts, dim=-1)

    def forward(
        self,
        noisy_displacement: torch.Tensor,
        diffusion_time: torch.Tensor,
        *,
        xyz: torch.Tensor,
        local_scale: torch.Tensor,
        p_dynamic: torch.Tensor,
        fm_confidence: torch.Tensor,
        cycle_confidence: torch.Tensor,
        fm_displacement: torch.Tensor,
    ) -> torch.Tensor:
        if diffusion_time.ndim == 0:
            diffusion_time = diffusion_time.expand(noisy_displacement.shape[0])
        features = torch.cat(
            (
                noisy_displacement,
                self._encode_position(xyz),
                diffusion_time.reshape(-1, 1).to(noisy_displacement),
                local_scale.reshape(-1, 1).to(noisy_displacement),
                p_dynamic.reshape(-1, 1).to(noisy_displacement),
                fm_confidence.reshape(-1, 1).to(noisy_displacement),
                cycle_confidence.reshape(-1, 1).to(noisy_displacement),
                fm_displacement.to(noisy_displacement),
            ),
            dim=-1,
        )
        return self.net(features)


def _alpha_sigma(t: torch.Tensor, schedule: str) -> tuple[torch.Tensor, torch.Tensor]:
    if str(schedule).lower() != "cosine":
        raise ValueError("only cosine diffusion schedule is implemented")
    alpha = torch.cos(0.5 * torch.pi * t).clamp_min(1.0e-3)
    sigma = torch.sin(0.5 * torch.pi * t).clamp_min(1.0e-5)
    return alpha, sigma


def train_displacement_denoiser(
    *,
    clean_displacement: torch.Tensor,
    xyz: torch.Tensor,
    local_scale: torch.Tensor,
    p_dynamic: torch.Tensor,
    fm_confidence: torch.Tensor,
    cycle_confidence: torch.Tensor,
    fm_displacement: torch.Tensor,
    sample_weight: torch.Tensor,
    cfg: DisplacementDiffusionConfig,
    direction_name: str,
    device: str | torch.device,
) -> tuple[DisplacementDenoiser, list[dict[str, float]]]:
    device = torch.device(device if torch.cuda.is_available() or str(device) == "cpu" else "cpu")
    clean = torch.as_tensor(clean_displacement).detach().float().to(device)
    if clean.ndim != 2 or clean.shape[-1] != 3 or int(clean.shape[0]) == 0:
        raise ValueError("clean_displacement must be non-empty [N,3]")
    cond = {
        "xyz": torch.as_tensor(xyz).detach().float().to(device),
        "local_scale": torch.as_tensor(local_scale).detach().float().reshape(-1).to(device),
        "p_dynamic": torch.as_tensor(p_dynamic).detach().float().reshape(-1).to(device),
        "fm_confidence": torch.as_tensor(fm_confidence).detach().float().reshape(-1).to(device),
        "cycle_confidence": torch.as_tensor(cycle_confidence).detach().float().reshape(-1).to(device),
        "fm_displacement": torch.as_tensor(fm_displacement).detach().float().to(device),
    }
    weight = torch.as_tensor(sample_weight).detach().float().reshape(-1).to(device).clamp_min(0.0)
    if weight.numel() != clean.shape[0] or float(weight.sum()) <= 0:
        weight = torch.ones(clean.shape[0], device=device)
    weight = weight / weight.sum()
    model = DisplacementDenoiser(cfg).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=max(float(cfg.lr), 1.0e-8))
    gen = torch.Generator(device=device).manual_seed(int(cfg.seed))
    logs: list[dict[str, float]] = []
    progress = tqdm(range(max(1, int(cfg.iters))), desc=f"{direction_name} diffusion", unit="iter")
    for iteration in progress:
        batch = min(int(cfg.batch_size), int(clean.shape[0]))
        ids = torch.multinomial(weight, batch, replacement=True, generator=gen)
        t = torch.rand(batch, device=device, generator=gen).clamp(1.0e-4, 0.999)
        alpha, sigma = _alpha_sigma(t, cfg.schedule)
        noise = torch.randn((batch, 3), device=device, generator=gen)
        noisy = alpha[:, None] * clean[ids] + sigma[:, None] * noise
        pred = model(noisy, t, **{k: v[ids] for k, v in cond.items()})
        loss = F.mse_loss(pred, noise)
        if not bool(torch.isfinite(loss)):
            raise RuntimeError(f"{direction_name} diffusion produced a non-finite loss")
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        if iteration % 25 == 0 or iteration == int(cfg.iters) - 1:
            row = {"iteration": int(iteration), "loss": float(loss.detach().cpu())}
            logs.append(row)
            progress.set_postfix(loss=f"{row['loss']:.4g}")
    return model, logs


@torch.no_grad()
def score_displacement_candidates(
    model: DisplacementDenoiser,
    *,
    clean_displacement: torch.Tensor,
    xyz: torch.Tensor,
    local_scale: torch.Tensor,
    p_dynamic: torch.Tensor,
    fm_confidence: torch.Tensor,
    cycle_confidence: torch.Tensor,
    fm_displacement: torch.Tensor,
    cfg: DisplacementDiffusionConfig,
    device: str | torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    device = torch.device(device if torch.cuda.is_available() or str(device) == "cpu" else "cpu")
    clean_all = torch.as_tensor(clean_displacement).detach().float()
    n = int(clean_all.shape[0])
    energy = torch.empty(n, dtype=torch.float32)
    correction = torch.empty((n, 3), dtype=torch.float32)
    gen = torch.Generator(device=device).manual_seed(int(cfg.seed) + 99)
    model = model.to(device).eval()
    for start in range(0, n, max(1, int(cfg.chunk_size))):
        end = min(start + int(cfg.chunk_size), n)
        clean = clean_all[start:end].to(device)
        cond = {
            "xyz": torch.as_tensor(xyz[start:end]).detach().float().to(device),
            "local_scale": torch.as_tensor(local_scale[start:end]).detach().float().reshape(-1).to(device),
            "p_dynamic": torch.as_tensor(p_dynamic[start:end]).detach().float().reshape(-1).to(device),
            "fm_confidence": torch.as_tensor(fm_confidence[start:end]).detach().float().reshape(-1).to(device),
            "cycle_confidence": torch.as_tensor(cycle_confidence[start:end]).detach().float().reshape(-1).to(device),
            "fm_displacement": torch.as_tensor(fm_displacement[start:end]).detach().float().to(device),
        }
        accum_energy = torch.zeros(end - start, device=device)
        accum_corr = torch.zeros(end - start, 3, device=device)
        for _ in range(max(1, int(cfg.trials))):
            t = torch.rand(end - start, device=device, generator=gen).clamp(1.0e-4, 0.999)
            alpha, sigma = _alpha_sigma(t, cfg.schedule)
            noise = torch.randn((end - start, 3), device=device, generator=gen)
            noisy = alpha[:, None] * clean + sigma[:, None] * noise
            eps_hat = model(noisy, t, **cond)
            denoised = (noisy - sigma[:, None] * eps_hat) / alpha[:, None].clamp_min(1.0e-6)
            delta = denoised - clean
            scale = cond["local_scale"].clamp_min(1.0e-8)
            accum_energy += delta.square().sum(-1) / scale.square()
            accum_corr += delta
        energy[start:end] = (accum_energy / max(1, int(cfg.trials))).detach().cpu()
        correction[start:end] = (accum_corr / max(1, int(cfg.trials))).detach().cpu()
    return energy, correction


def save_diffusion_model(model: DisplacementDenoiser, logs: list[dict[str, float]], output_dir: str | Path, cfg: DisplacementDiffusionConfig) -> None:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "schema_version": 1,
            "state_dict": model.cpu().state_dict(),
            "config": asdict(cfg),
            "training_log": logs,
            "ground_truth_used_as_algorithm_input": False,
        },
        output / "checkpoint.pt",
    )
