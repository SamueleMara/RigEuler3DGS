"""Fixed finite-volume Eulerian CurlGrid with a temporal B-spline potential.

The physical Eulerian DOF is one cell-centred velocity v_c(t) per fixed world-space
control volume.  Every Gaussian centre in the same cell receives exactly that v_c
for the whole physical video interval.  Gaussian positions are never optimizer
parameters.

Constant-density continuity is architectural.  A vector potential A lives on
oriented grid edges and a discrete curl produces face fluxes.  Therefore

    div_h(curl_h(A)) = 0

up to floating-point roundoff.

The cubic controls are shared by all frames, so temporal coherence is part of
the representation rather than a weak frame-to-frame loss.
"""

from __future__ import annotations

from typing import Dict, Iterable, List, Optional, Tuple
import math
import torch
from torch import nn


def _balanced_cell_counts(
    extent: torch.Tensor, target_cells: int, min_cells_per_axis: int = 2
) -> torch.Tensor:
    target = max(8, int(target_cells))
    extent = extent.detach().double().clamp_min(1.0e-8)
    volume = float(torch.prod(extent).item())
    h = (volume / float(target)) ** (1.0 / 3.0)
    counts = (
        torch.ceil(extent / max(h, 1.0e-12)).long().clamp_min(int(min_cells_per_axis))
    )
    for _ in range(256):
        product = int(torch.prod(counts).item())
        if product > int(math.ceil(target * 1.30)):
            candidates = []
            for axis in range(3):
                if counts[axis] <= min_cells_per_axis:
                    continue
                trial = counts.clone()
                trial[axis] -= 1
                sizes = extent / trial.double()
                candidates.append(
                    (
                        abs(int(torch.prod(trial).item()) - target),
                        float(sizes.max() / sizes.min()),
                        trial,
                    )
                )
            if not candidates:
                break
            candidates.sort(key=lambda item: (item[0], item[1]))
            counts = candidates[0][2]
        elif product < int(math.floor(target * 0.80)):
            candidates = []
            for axis in range(3):
                trial = counts.clone()
                trial[axis] += 1
                sizes = extent / trial.double()
                candidates.append(
                    (
                        abs(int(torch.prod(trial).item()) - target),
                        float(sizes.max() / sizes.min()),
                        trial,
                    )
                )
            candidates.sort(key=lambda item: (item[0], item[1]))
            counts = candidates[0][2]
        else:
            break
    return counts


class FixedCurlGrid(nn.Module):
    STATE_VERSION = 3

    def __init__(
        self,
        canonical_xyz: torch.Tensor,
        time_values: Iterable[float],
        *,
        target_cells: int = 16384,
        padding_fraction: float = 0.15,
        min_cells_per_axis: int = 2,
        temporal_control_knots: int = 10,
    ):
        super().__init__()
        if canonical_xyz.ndim != 2 or canonical_xyz.shape[-1] != 3:
            raise ValueError("canonical_xyz must be [N,3]")
        times = torch.as_tensor(
            list(time_values), device=canonical_xyz.device, dtype=canonical_xyz.dtype
        ).reshape(-1)
        times = torch.unique(times, sorted=True)
        if times.numel() < 2 or not bool((times[1:] > times[:-1]).all()):
            raise ValueError("time_values must contain strictly increasing video times")

        xyz_min = canonical_xyz.detach().amin(dim=0)
        xyz_max = canonical_xyz.detach().amax(dim=0)
        raw_extent = (xyz_max - xyz_min).clamp_min(1.0e-6)
        pad = max(0.0, float(padding_fraction)) * raw_extent.max()
        origin = xyz_min - pad
        upper = xyz_max + pad
        extent = (upper - origin).clamp_min(1.0e-6)
        counts = _balanced_cell_counts(extent, target_cells, min_cells_per_axis).to(
            canonical_xyz.device
        )
        cell_size = extent / counts.to(extent.dtype)

        self.register_buffer("time_values", times)
        self.register_buffer("origin", origin)
        self.register_buffer("upper", upper)
        self.register_buffer("extent", extent)
        self.register_buffer("cell_counts", counts.long())
        self.register_buffer("cell_size", cell_size)

        nx, ny, nz = [int(v) for v in counts.tolist()]
        self.temporal_mode = "cubic_bspline"
        K = max(4, int(temporal_control_knots))
        self.num_control_knots = K
        device, dtype = canonical_xyz.device, canonical_xyz.dtype
        self.potential_x = nn.Parameter(
            torch.zeros((K, nx, ny + 1, nz + 1), device=device, dtype=dtype)
        )
        self.potential_y = nn.Parameter(
            torch.zeros((K, nx + 1, ny, nz + 1), device=device, dtype=dtype)
        )
        self.potential_z = nn.Parameter(
            torch.zeros((K, nx + 1, ny + 1, nz), device=device, dtype=dtype)
        )

    @property
    def num_cells(self) -> int:
        return int(torch.prod(self.cell_counts).item())

    @property
    def grid_shape(self) -> Tuple[int, int, int]:
        return tuple(int(v) for v in self.cell_counts.tolist())

    @property
    def parameter_count(self) -> int:
        return int(sum(p.numel() for p in self.potential_parameters()))

    def potential_parameters(self) -> List[nn.Parameter]:
        return [self.potential_x, self.potential_y, self.potential_z]

    def _bspline_weights(self, t):
        value = torch.as_tensor(
            t, device=self.time_values.device, dtype=self.time_values.dtype
        ).reshape(())
        u01 = (
            (value - self.time_values[0])
            / (self.time_values[-1] - self.time_values[0]).clamp_min(1.0e-8)
        ).clamp(0.0, 1.0)
        span_max = self.num_control_knots - 3
        s = u01 * float(span_max)
        base = int(torch.floor(s.detach()).long().clamp(0, span_max - 1).item())
        u = (s - float(base)).clamp(0.0, 1.0)
        u2, u3 = u * u, u * u * u
        w = torch.stack(
            (
                (1 - 3 * u + 3 * u2 - u3) / 6,
                (4 - 6 * u2 + 3 * u3) / 6,
                (1 + 3 * u + 3 * u2 - 3 * u3) / 6,
                u3 / 6,
            )
        )
        return base, w

    def _potential_component_at(self, parameter: torch.Tensor, t) -> torch.Tensor:
        base, weights = self._bspline_weights(t)
        return (
            parameter[base : base + 4]
            * weights.reshape((4,) + (1,) * (parameter.ndim - 1))
        ).sum(dim=0)

    def potential_at(self, t):
        return (
            self._potential_component_at(self.potential_x, t),
            self._potential_component_at(self.potential_y, t),
            self._potential_component_at(self.potential_z, t),
        )

    def _cell_coordinates(self, points: torch.Tensor):
        coordinate = (points - self.origin.to(points)) / self.cell_size.to(points)
        counts = self.cell_counts.to(points.device)
        inside = ((coordinate >= 0) & (coordinate < counts.to(points.dtype))).all(
            dim=-1
        )
        safe = torch.minimum(
            torch.maximum(coordinate, torch.zeros_like(coordinate)),
            counts.to(points.dtype) - 1.0e-6,
        )
        return torch.floor(safe).long(), inside

    def cell_centers(self, *, device=None, dtype=None) -> torch.Tensor:
        device = self.origin.device if device is None else device
        dtype = self.origin.dtype if dtype is None else dtype
        nx, ny, nz = self.grid_shape
        axes = [
            self.origin[a].to(device=device, dtype=dtype)
            + (torch.arange(n, device=device, dtype=dtype) + 0.5)
            * self.cell_size[a].to(device=device, dtype=dtype)
            for a, n in enumerate((nx, ny, nz))
        ]
        gx, gy, gz = torch.meshgrid(*axes, indexing="ij")
        return torch.stack((gx, gy, gz), dim=-1).reshape(-1, 3)

    def cell_ids(self, points: torch.Tensor) -> torch.Tensor:
        base, inside = self._cell_coordinates(points.detach())
        _, ny, nz = self.grid_shape
        ids = (base[:, 0] * ny + base[:, 1]) * nz + base[:, 2]
        return torch.where(inside, ids, torch.full_like(ids, -1))

    def face_velocities(self, t):
        Ax, Ay, Az = self.potential_at(t)
        hx, hy, hz = self.cell_size.to(Ax).unbind()
        ux = (Az[:, 1:, :] - Az[:, :-1, :]) / hy - (Ay[:, :, 1:] - Ay[:, :, :-1]) / hz
        uy = (Ax[:, :, 1:] - Ax[:, :, :-1]) / hz - (Az[1:, :, :] - Az[:-1, :, :]) / hx
        uz = (Ay[1:, :, :] - Ay[:-1, :, :]) / hx - (Ax[:, 1:, :] - Ax[:, :-1, :]) / hy
        return ux, uy, uz

    def cell_velocity_grid(self, t):
        ux, uy, uz = self.face_velocities(t)
        return torch.stack(
            (
                0.5 * (ux[:-1] + ux[1:]),
                0.5 * (uy[:, :-1] + uy[:, 1:]),
                0.5 * (uz[:, :, :-1] + uz[:, :, 1:]),
            ),
            dim=-1,
        )

    def cell_velocities(self, t):
        return self.cell_velocity_grid(t).reshape(-1, 3)

    def gather_cell_velocity(
        self,
        cell_ids: torch.Tensor,
        t=None,
        *,
        cell_velocity_table: Optional[torch.Tensor] = None,
    ):
        if cell_velocity_table is None:
            if t is None:
                raise ValueError("t required")
            cell_velocity_table = self.cell_velocities(t)
        ids = cell_ids.detach().long()
        valid = ids >= 0
        safe = ids.clamp(min=0, max=max(0, self.num_cells - 1))
        return (
            cell_velocity_table.index_select(0, safe)
            * valid.to(cell_velocity_table.dtype)[:, None]
        )

    def velocity(self, points, t):
        ids = self.cell_ids(points)
        return self.gather_cell_velocity(ids, t), ids

    def finite_volume_divergence_grid(self, t):
        ux, uy, uz = self.face_velocities(t)
        hx, hy, hz = self.cell_size.to(ux).unbind()
        return (
            (ux[1:] - ux[:-1]) / hx
            + (uy[:, 1:] - uy[:, :-1]) / hy
            + (uz[:, :, 1:] - uz[:, :, :-1]) / hz
        )

    def continuity_residual(self, t):
        return self.finite_volume_divergence_grid(t).reshape(-1)

    def material_support(self, points):
        ids = self.cell_ids(points)
        valid = ids >= 0
        counts = torch.zeros(self.num_cells, device=points.device, dtype=torch.long)
        if valid.any():
            counts.scatter_add_(0, ids[valid], torch.ones_like(ids[valid]))
        occupied = counts > 0
        nx, ny, nz = self.grid_shape
        occ = occupied.reshape(nx, ny, nz)
        neigh = torch.zeros_like(occ, dtype=torch.long)
        neigh[1:] += occ[:-1]
        neigh[:-1] += occ[1:]
        neigh[:, 1:] += occ[:, :-1]
        neigh[:, :-1] += occ[:, 1:]
        neigh[:, :, 1:] += occ[:, :, :-1]
        neigh[:, :, :-1] += occ[:, :, 1:]
        boundary = occ & (neigh < 6)
        return {
            "cell_ids": ids,
            "counts": counts,
            "occupied": occupied,
            "boundary": boundary.reshape(-1),
            "occupied_cells": int(occupied.sum()),
            "boundary_cells": int(boundary.sum()),
            "outside_points": int((~valid).sum()),
        }

    def temporal_acceleration_loss(self):
        losses = []
        for p in self.potential_parameters():
            second = p[2:] - 2 * p[1:-1] + p[:-2]
            losses.append(second.square().mean())
        return torch.stack(losses).mean()

    def capture_state(self) -> Dict:
        return {
            "version": self.STATE_VERSION,
            "time_values": self.time_values.detach().cpu(),
            "origin": self.origin.detach().cpu(),
            "upper": self.upper.detach().cpu(),
            "cell_counts": self.cell_counts.detach().cpu(),
            "temporal_mode": self.temporal_mode,
            "num_control_knots": self.num_control_knots,
            "potential_x": self.potential_x.detach().cpu(),
            "potential_y": self.potential_y.detach().cpu(),
            "potential_z": self.potential_z.detach().cpu(),
        }

    @classmethod
    def from_state(cls, state: Dict, device="cuda"):
        version = int(state.get("version", 0))
        if version != 3:
            raise RuntimeError("Only FixedCurlGrid state version 3 is supported")
        dtype = state["potential_x"].dtype
        dummy = torch.stack((state["origin"], state["upper"]), dim=0).to(
            device=device, dtype=dtype
        )
        temporal_knots = int(
            state.get("num_control_knots", state["potential_x"].shape[0])
        )
        field = cls(
            dummy,
            state["time_values"].tolist(),
            target_cells=max(8, int(torch.prod(state["cell_counts"]).item())),
            padding_fraction=0.0,
            temporal_control_knots=temporal_knots,
        )
        field.origin = state["origin"].to(device=device, dtype=dtype)
        field.upper = state["upper"].to(device=device, dtype=dtype)
        field.extent = field.upper - field.origin
        field.cell_counts = state["cell_counts"].to(device=device).long()
        field.cell_size = field.extent / field.cell_counts.to(field.extent.dtype)
        field.potential_x = nn.Parameter(state["potential_x"].to(device).contiguous())
        field.potential_y = nn.Parameter(state["potential_y"].to(device).contiguous())
        field.potential_z = nn.Parameter(state["potential_z"].to(device).contiguous())
        field.temporal_mode = "cubic_bspline"
        field.num_control_knots = int(field.potential_x.shape[0])
        return field
