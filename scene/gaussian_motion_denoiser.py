"""Small per-Gaussian graph denoiser for frame-to-frame motion probes."""

from __future__ import annotations

import torch
from torch import nn


@torch.no_grad()
def build_canonical_knn(points: torch.Tensor, k: int = 16, chunk_size: int = 2048):
    """Return fixed canonical KNN indices [N,K] and nearest-neighbor median."""
    xyz = torch.as_tensor(points).detach().float()
    if xyz.ndim != 2 or xyz.shape[-1] != 3:
        raise ValueError("points must be [N,3]")
    n = int(xyz.shape[0])
    k = max(1, min(int(k), max(1, n - 1)))
    rows = []
    nearest = []
    for start in range(0, n, int(chunk_size)):
        end = min(n, start + int(chunk_size))
        dist = torch.cdist(xyz[start:end], xyz)
        arange = torch.arange(start, end, device=xyz.device)
        dist[torch.arange(end - start, device=xyz.device), arange] = float("inf")
        values, indices = torch.topk(dist, k, largest=False)
        rows.append(indices)
        nearest.append(values[:, 0])
    nn = torch.cat(nearest)
    finite = nn[torch.isfinite(nn) & (nn > 0)]
    median = finite.median() if finite.numel() else xyz.new_tensor(1.0)
    return torch.cat(rows, 0).long(), median


class GaussianMotionDenoiser(nn.Module):
    """Residual graph MLP with mean-neighbor message passing."""

    def __init__(self, input_dim: int, hidden_dim: int = 96, message_layers: int = 2):
        super().__init__()
        self.input_dim = int(input_dim)
        self.hidden_dim = int(hidden_dim)
        self.message_layers = int(message_layers)
        self.encoder = nn.Sequential(
            nn.Linear(self.input_dim, self.hidden_dim),
            nn.SiLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.SiLU(),
        )
        self.blocks = nn.ModuleList(
            nn.Sequential(
                nn.Linear(2 * self.hidden_dim, self.hidden_dim),
                nn.SiLU(),
                nn.Linear(self.hidden_dim, self.hidden_dim),
            )
            for _ in range(max(1, self.message_layers))
        )
        self.out = nn.Sequential(
            nn.SiLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.SiLU(),
            nn.Linear(self.hidden_dim, 3),
        )

    def forward(self, features: torch.Tensor, knn_idx: torch.Tensor, step_scale: float | torch.Tensor):
        if features.ndim != 2 or features.shape[-1] != self.input_dim:
            raise ValueError(f"features must be [N,{self.input_dim}]")
        if knn_idx.ndim != 2 or knn_idx.shape[0] != features.shape[0]:
            raise ValueError("knn_idx must be [N,K]")
        h = self.encoder(features)
        neighbors = knn_idx.to(device=features.device, dtype=torch.long)
        for block in self.blocks:
            message = h[neighbors].mean(1)
            h = h + block(torch.cat((h, message), -1))
        scale = torch.as_tensor(step_scale, device=features.device, dtype=features.dtype)
        return torch.tanh(self.out(h)) * scale
