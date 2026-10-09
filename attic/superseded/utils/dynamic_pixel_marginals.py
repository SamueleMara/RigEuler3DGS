"""R1 -- project the Gaussian dynamic posterior into image space.

    m_cp = sum_i Omega_cpi q_i        (dynamic pixel mass)
    s_cp = sum_i Omega_cpi (1 - q_i)  (static pixel mass)

Both are computed with a single splat of ``[q, 1-q]``, which is the same
compositing the renderer uses for colour, so the marginals inherit the
renderer's own occlusion handling for free: a Gaussian hidden behind another
contributes to neither marginal at that pixel, which is exactly right.

§8: no hand threshold on dynamic pixels.  The marginals are used soft.  The
only cut applied anywhere is a machine-scale support floor that removes
pixels no Gaussian covers at all, and removing zero is not a decision.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch

from utils.rasterizer_ownership import extract_sparse_gaussian_pixel_ownership

EPS = 1.0e-8


@dataclass
class PixelMarginals:
    """Dynamic / static / total ownership mass for one camera."""

    dynamic_mass: torch.Tensor      # [H, W]
    static_mass: torch.Tensor       # [H, W]
    total_mass: torch.Tensor        # [H, W]
    camera_index: int = 0
    stats: dict[str, Any] = field(default_factory=dict)

    @property
    def dynamic_fraction(self) -> torch.Tensor:
        """``m / (m + s)``: the probability the pixel shows moving geometry."""
        return self.dynamic_mass / (self.dynamic_mass + self.static_mass).clamp_min(EPS)

    def support(self, floor: float = 1.0e-4) -> torch.Tensor:
        """Pixels any geometry covers. A sparsity floor, not a semantic cut."""
        return self.total_mass > float(floor)


@torch.no_grad()
def project_gaussian_dynamic_posterior_to_pixels(
    camera,
    gaussians,
    pipe,
    dynamic_posterior: torch.Tensor,
    *,
    camera_index: int = 0,
    render_feature_fn=None,
) -> PixelMarginals:
    """Both marginals from one splat of ``[q, 1-q]``."""
    q = torch.as_tensor(dynamic_posterior).detach().float().to(gaussians.get_xyz.device).reshape(-1, 1)
    maps = extract_sparse_gaussian_pixel_ownership(
        camera, gaussians, pipe, torch.cat((q, 1.0 - q), dim=1),
        camera_index=int(camera_index), render_feature_fn=render_feature_fn,
    )
    dynamic, static = maps.channels[0], maps.channels[1]
    covered = maps.total_mass > 1.0e-4
    return PixelMarginals(
        dynamic_mass=dynamic, static_mass=static, total_mass=maps.total_mass,
        camera_index=int(camera_index),
        stats={
            "dynamic_mass_fraction": float(
                dynamic[covered].sum() / (dynamic[covered].sum() + static[covered].sum()).clamp_min(EPS)
            ) if bool(covered.any()) else 0.0,
            "covered_pixel_fraction": float(covered.float().mean()),
            "median_dynamic_fraction": float(
                (dynamic[covered] / (dynamic[covered] + static[covered]).clamp_min(EPS)).median()
            ) if bool(covered.any()) else 0.0,
        },
    )


@torch.no_grad()
def project_gaussian_static_posterior_to_pixels(camera, gaussians, pipe, dynamic_posterior, **kwargs):
    """The static marginal, for callers that only need calibration pixels."""
    return project_gaussian_dynamic_posterior_to_pixels(
        camera, gaussians, pipe, dynamic_posterior, **kwargs
    )


@torch.no_grad()
def build_dynamic_pixel_domain(
    marginals: PixelMarginals,
    *,
    support_floor: float = 1.0e-4,
    max_pixels: int = 4096,
    generator: torch.Generator | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """``(flat pixel indices, their dynamic mass)`` for the transport solve.

    Pixels are sampled IN PROPORTION to their dynamic mass rather than by
    taking the top-k.  A top-k cut is a threshold in disguise -- it would
    discard exactly the moderately-dynamic pixels the transport most needs to
    explain, and make the domain depend on a number nobody derived.
    """
    support = marginals.support(support_floor)
    mass = (marginals.dynamic_mass * support).reshape(-1)
    available = int((mass > 0).sum())
    if available == 0:
        empty = torch.zeros(0, dtype=torch.long, device=mass.device)
        return empty, torch.zeros(0, device=mass.device)
    if available <= int(max_pixels):
        index = (mass > 0).nonzero(as_tuple=True)[0]
        return index, mass[index]
    probability = (mass / mass.sum().clamp_min(EPS)).cpu()
    picked = torch.multinomial(
        probability, num_samples=int(max_pixels), replacement=False, generator=generator
    ).to(mass.device)
    return picked, mass[picked]
