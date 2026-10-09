from __future__ import annotations

import torch
from torch import nn


class RigidPartLabels(nn.Module):
    """Canonical, time-invariant soft labels for one static + K moving parts."""

    def __init__(
        self,
        num_gaussians: int,
        num_moving_parts: int,
        init_logits: torch.Tensor | None = None,
        temperature: float = 1.0,
    ) -> None:
        super().__init__()
        k1 = int(num_moving_parts) + 1
        if init_logits is None:
            logits = torch.zeros((int(num_gaussians), k1), dtype=torch.float32)
        else:
            logits = torch.as_tensor(init_logits).detach().float()
            if tuple(logits.shape) != (int(num_gaussians), k1):
                raise ValueError(f"init_logits must be [{num_gaussians},{k1}], got {tuple(logits.shape)}")
        self.part_logits = nn.Parameter(logits)
        self.temperature = float(temperature)

    def probabilities(self, temperature: float | None = None) -> torch.Tensor:
        temp = max(float(self.temperature if temperature is None else temperature), 1.0e-6)
        return torch.softmax(self.part_logits / temp, dim=-1)

    def hard_labels(self, temperature: float | None = None) -> torch.Tensor:
        return self.probabilities(temperature).argmax(dim=-1)

    @staticmethod
    def entropy(part_prob: torch.Tensor, eps: float = 1.0e-8) -> torch.Tensor:
        p = torch.as_tensor(part_prob).float().clamp_min(float(eps))
        return -(p * torch.log(p)).sum(dim=-1).mean()
