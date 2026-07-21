#
# Adaptive Eulerian velocity-field extension for GraphDeco 3DGS.
# This module is intentionally independent from the CUDA rasterizer: it only
# produces ordinary 3D Gaussian tensors that the original renderer can consume.
#

import math
import torch
from torch import nn


class FourierTimeEmbedding(nn.Module):
    """Small Fourier embedding for scalar timestamps.

    Timestamps are expected to be normalized to a compact range, usually [0, 1].
    The embedding is deterministic and has no trainable parameters.
    """

    def __init__(self, num_frequencies=8, include_input=True):
        super().__init__()
        self.num_frequencies = int(num_frequencies)
        self.include_input = bool(include_input)
        if self.num_frequencies > 0:
            freqs = 2.0 ** torch.arange(self.num_frequencies).float() * math.pi
        else:
            freqs = torch.empty(0).float()
        self.register_buffer("freqs", freqs, persistent=False)

    @property
    def out_dim(self):
        return (1 if self.include_input else 0) + 2 * self.num_frequencies

    def forward(self, t, n, device=None, dtype=None):
        if device is None:
            device = self.freqs.device
        if dtype is None:
            dtype = torch.float32
        t_tensor = torch.as_tensor(t, device=device, dtype=dtype).reshape(1, 1)
        t_tensor = t_tensor.expand(n, 1)
        outs = []
        if self.include_input:
            outs.append(t_tensor)
        if self.freqs.numel() > 0:
            freqs = self.freqs.to(device=device, dtype=dtype).reshape(1, -1)
            phase = t_tensor * freqs
            outs.extend([torch.sin(phase), torch.cos(phase)])
        if not outs:
            return torch.empty(n, 0, device=device, dtype=dtype)
        return torch.cat(outs, dim=-1)


class SineLayer(nn.Module):
    """SIREN sine layer: y = sin(omega_0 * Linear(x)).

    This follows the initialization from Sitzmann et al., "Implicit Neural
    Representations with Periodic Activation Functions". It is deliberately
    isolated to the motion-field MLP; dynamic labels and semantic/DINO heads
    should not use periodic activations.
    """

    def __init__(self, in_features, out_features, is_first=False, omega_0=30.0, bias=True):
        super().__init__()
        self.in_features = int(in_features)
        self.is_first = bool(is_first)
        self.omega_0 = float(omega_0)
        self.linear = nn.Linear(int(in_features), int(out_features), bias=bias)
        self.init_weights()

    def init_weights(self):
        with torch.no_grad():
            if self.is_first:
                bound = 1.0 / max(1, self.in_features)
            else:
                bound = math.sqrt(6.0 / max(1, self.in_features)) / max(self.omega_0, 1e-6)
            self.linear.weight.uniform_(-bound, bound)
            if self.linear.bias is not None:
                self.linear.bias.uniform_(-bound, bound)

    def forward(self, x):
        return torch.sin(self.omega_0 * self.linear(x))


class EulerianVelocityField(nn.Module):
    """World-fixed velocity field v_theta(x, t).

    The field is queried at the current Gaussian world position and integrated
    through time. It can use a sparse/adaptive motion grid to obtain local
    trainable features.

    Activation options:
      - relu: original stable piecewise-linear MLP.
      - siren: SIREN-style sinusoidal MLP for a smoother/high-frequency
        continuous velocity field. ReLU remains the default.
    """

    def __init__(self,
                 grid_feature_dim=32,
                 hidden_dim=64,
                 num_layers=3,
                 time_frequencies=8,
                 max_velocity=0.0,
                 xyz_scale=1.0,
                 activation="relu",
                 siren_omega0=30.0,
                 siren_hidden_omega0=30.0):
        super().__init__()
        self.grid_feature_dim = int(grid_feature_dim)
        self.hidden_dim = int(hidden_dim)
        self.num_layers = int(num_layers)
        self.max_velocity = float(max_velocity)
        self.xyz_scale = float(xyz_scale) if float(xyz_scale) > 0 else 1.0
        self.activation = str(activation or "relu").lower()
        if self.activation in ("sin", "sine"):
            self.activation = "siren"
        if self.activation not in ("relu", "siren"):
            raise ValueError(f"Unsupported motion MLP activation '{activation}'. Use 'relu' or 'siren'.")
        self.siren_omega0 = float(siren_omega0)
        self.siren_hidden_omega0 = float(siren_hidden_omega0)
        self.time_embed = FourierTimeEmbedding(time_frequencies, include_input=True)

        in_dim = 3 + self.time_embed.out_dim + self.grid_feature_dim
        if self.activation == "siren":
            self.mlp = self._build_siren(in_dim)
        else:
            self.mlp = self._build_relu(in_dim)

        # Start from exactly zero velocity. This makes enabling the dynamic layer
        # safe on top of a pretrained static 3DGS scene. Note: this also makes
        # the first updates affect the final layer before earlier layers receive
        # large gradients, which is intentional for stability.
        last = self.mlp[-1]
        if isinstance(last, nn.Linear):
            nn.init.zeros_(last.weight)
            nn.init.zeros_(last.bias)

    def _build_relu(self, in_dim):
        layers = []
        dim = int(in_dim)
        for _ in range(max(1, self.num_layers - 1)):
            layers.append(nn.Linear(dim, self.hidden_dim))
            layers.append(nn.ReLU(inplace=True))
            dim = self.hidden_dim
        layers.append(nn.Linear(dim, 3))
        return nn.Sequential(*layers)

    def _build_siren(self, in_dim):
        layers = []
        dim = int(in_dim)
        hidden_layers = max(1, self.num_layers - 1)
        for layer_idx in range(hidden_layers):
            layers.append(SineLayer(
                dim,
                self.hidden_dim,
                is_first=(layer_idx == 0),
                omega_0=(self.siren_omega0 if layer_idx == 0 else self.siren_hidden_omega0),
            ))
            dim = self.hidden_dim
        final = nn.Linear(dim, 3)
        # SIREN paper-style final-layer scale. It is immediately zeroed by the
        # outer constructor for safe dynamic initialization, but keeping this
        # here makes the module sensible if that policy is later changed.
        with torch.no_grad():
            bound = math.sqrt(6.0 / max(1, dim)) / max(self.siren_hidden_omega0, 1e-6)
            final.weight.uniform_(-bound, bound)
            final.bias.uniform_(-bound, bound)
        layers.append(final)
        return nn.Sequential(*layers)

    def extra_repr(self):
        return (f"activation={self.activation}, hidden_dim={self.hidden_dim}, "
                f"num_layers={self.num_layers}, max_velocity={self.max_velocity}, "
                f"siren_omega0={self.siren_omega0}, "
                f"siren_hidden_omega0={self.siren_hidden_omega0}")

    def query_velocity(self, x, t, grid=None, gaussian_cell_id=None):
        if x.numel() == 0:
            return torch.empty_like(x)
        if grid is None:
            z = torch.zeros(x.shape[0], self.grid_feature_dim, device=x.device, dtype=x.dtype)
        else:
            z = grid.query_features(x, gaussian_cell_id=gaussian_cell_id)
            if z.dtype != x.dtype:
                z = z.to(dtype=x.dtype)
        te = self.time_embed(t, x.shape[0], device=x.device, dtype=x.dtype)
        inp = torch.cat([x / self.xyz_scale, te, z], dim=-1)
        v = self.mlp(inp)
        if self.max_velocity > 0:
            v = torch.tanh(v) * self.max_velocity
        return v

    def integrate(self, x0, t0, t1, steps=4, grid=None, gaussian_cell_id=None, return_path=False):
        """Integrate x' = v(x,t) from t0 to t1 with explicit Euler steps.

        Euler integration is deliberately used as the MVP because it is cheap,
        differentiable, and easy to inspect. RK4 can be added later without
        changing the renderer contract.
        """
        steps = max(1, int(steps))
        x = x0
        dt = (float(t1) - float(t0)) / float(steps)
        last_v = torch.zeros_like(x0)
        path = [x0] if return_path else None
        for s in range(steps):
            ts = float(t0) + (s + 0.5) * dt
            last_v = self.query_velocity(x, ts, grid=grid, gaussian_cell_id=gaussian_cell_id)
            x = x + dt * last_v
            if return_path:
                path.append(x)
        if return_path:
            return x, last_v, path
        return x, last_v
