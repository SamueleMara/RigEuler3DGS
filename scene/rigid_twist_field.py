"""Explicit time-varying rigid velocity field for one moving part.

The field predicts one bounded angular velocity omega(t) and one bounded
translation velocity u(t).  The resulting world velocity is

    v_rigid(x,t) = omega(t) x (x - center) + u(t)

Using a fixed canonical center removes the arbitrary origin dependence and the
bounded outputs prevent the scale explosion caused by unconstrained fitted
velocity teachers.
"""

import math
import torch
from torch import nn


class _FourierTimeEmbedding(nn.Module):
    def __init__(self, num_frequencies=8, include_input=True):
        super().__init__()
        self.num_frequencies = int(num_frequencies)
        self.include_input = bool(include_input)
        if self.num_frequencies > 0:
            freqs = (2.0 ** torch.arange(self.num_frequencies).float()) * math.pi
        else:
            freqs = torch.empty(0).float()
        self.register_buffer("freqs", freqs, persistent=False)

    @property
    def out_dim(self):
        return (1 if self.include_input else 0) + 2 * self.num_frequencies

    def forward(self, t, device, dtype):
        tt = torch.as_tensor(t, device=device, dtype=dtype).reshape(1, 1)
        values = []
        if self.include_input:
            values.append(tt)
        if self.freqs.numel() > 0:
            phase = tt * self.freqs.to(device=device, dtype=dtype).reshape(1, -1)
            values.extend((torch.sin(phase), torch.cos(phase)))
        return torch.cat(values, dim=-1) if values else torch.empty(1, 0, device=device, dtype=dtype)


class TimeVaryingRigidTwist(nn.Module):
    """Small temporal MLP that outputs bounded omega(t), u(t)."""

    def __init__(
        self,
        hidden_dim=64,
        num_layers=3,
        time_frequencies=8,
        max_omega=4.0,
        max_translation=2.0,
    ):
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.num_layers = int(num_layers)
        self.max_omega = float(max_omega)
        self.max_translation = float(max_translation)
        self.time_embed = _FourierTimeEmbedding(time_frequencies, include_input=True)

        layers = []
        dim = self.time_embed.out_dim
        for _ in range(max(1, self.num_layers - 1)):
            layers.append(nn.Linear(dim, self.hidden_dim))
            layers.append(nn.SiLU(inplace=True))
            dim = self.hidden_dim
        layers.append(nn.Linear(dim, 6))
        self.mlp = nn.Sequential(*layers)

        # Start from the identity/static explanation.  The residual Eulerian
        # field bootstraps image motion; the joint rigid loss then transfers it
        # into this explicit twist field.
        last = self.mlp[-1]
        nn.init.zeros_(last.weight)
        nn.init.zeros_(last.bias)

    def forward(self, t):
        ref = next(self.parameters())
        emb = self.time_embed(t, device=ref.device, dtype=ref.dtype)
        raw = self.mlp(emb).reshape(6)
        omega = torch.tanh(raw[:3]) * self.max_omega
        trans = torch.tanh(raw[3:]) * self.max_translation
        return omega, trans

    def velocity(self, x, t, center):
        omega, trans = self.forward(t)
        rel = x - center.reshape(1, 3).to(device=x.device, dtype=x.dtype)
        omega_x = omega.to(dtype=x.dtype).reshape(1, 3).expand_as(rel)
        v = torch.cross(omega_x, rel, dim=-1) + trans.to(dtype=x.dtype).reshape(1, 3)
        return v, omega, trans
