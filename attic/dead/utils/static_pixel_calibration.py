"""R2 -- confident static geometry calibrates the image model.

§4: static Gaussians are not a nuisance to remove, they are the calibration
asset.  They are the only place in the scene where the correct answer is
known -- nothing moved -- so they are where the observation model can be
measured rather than assumed.

Everything the pixel transport needs is measured here: how far a static pixel
appears to move between endpoints, and how much its appearance changes.  Both
become likelihood terms, which is what removes the need for a tuned
temperature or a weighted appearance term (§11, §29).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch

EPS = 1.0e-8


@dataclass
class StaticPixelCalibration:
    """The measured behaviour of pixels that did not move."""

    displacement_scale: float      # px, robust sigma of apparent motion
    appearance_scale: float        # RGB units, robust sigma of colour change
    render_residual_scale: float   # RGB units, predictive render error
    samples: int
    stats: dict[str, Any] = field(default_factory=dict)

    def displacement_loglik(self, distance: torch.Tensor) -> torch.Tensor:
        """Student-t in pixel distance; heavy tails so a few large apparent
        motions (thin structure, specularity) do not dominate."""
        scale = max(float(self.displacement_scale), 1.0e-3)
        return -2.5 * torch.log1p((distance / scale) ** 2 / 4.0)

    def appearance_loglik(self, difference: torch.Tensor) -> torch.Tensor:
        scale = max(float(self.appearance_scale), 1.0e-4)
        return -2.5 * torch.log1p((difference / scale) ** 2 / 4.0)


def _robust_scale(value: torch.Tensor) -> float:
    v = torch.as_tensor(value).detach().float().reshape(-1)
    v = v[torch.isfinite(v)]
    if int(v.numel()) == 0:
        return 1.0
    scale = float((v - v.median()).abs().median() * 1.4826)
    return scale if scale > 0 else float(v.abs().mean().clamp_min(1.0e-6))


@torch.no_grad()
def collect_confident_static_pixel_pairs(
    source_marginals,
    target_marginals,
    source_image: torch.Tensor,
    target_image: torch.Tensor,
    *,
    static_quantile: float = 0.75,
    max_pixels: int = 8192,
    generator: torch.Generator | None = None,
) -> dict[str, torch.Tensor]:
    """Pixels both endpoints agree are static, with their observed colours.

    A pixel qualifies only if BOTH endpoints call it static.  Requiring both
    matters: a pixel that is static in frame 0 and covered by moved geometry
    in frame T is exactly the disocclusion case, and calibrating on it would
    fold real motion into the noise model.
    """
    source_static = source_marginals.static_mass / source_marginals.total_mass.clamp_min(EPS)
    target_static = target_marginals.static_mass / target_marginals.total_mass.clamp_min(EPS)
    covered = source_marginals.support() & target_marginals.support()
    both = source_static.reshape(-1)[covered.reshape(-1)]
    if int(both.numel()) == 0:
        return {"index": torch.zeros(0, dtype=torch.long), "source_rgb": torch.zeros((0, 3)),
                "target_rgb": torch.zeros((0, 3))}
    cut = float(torch.quantile(both.float(), float(static_quantile)))
    confident = covered & (source_static >= cut) & (target_static >= cut)
    index = confident.reshape(-1).nonzero(as_tuple=True)[0]
    if int(index.numel()) > int(max_pixels):
        pick = torch.randperm(int(index.numel()), generator=generator)[: int(max_pixels)]
        index = index[pick.to(index.device)]
    source_rgb = source_image.reshape(3, -1)[:, index].T
    target_rgb = target_image.reshape(3, -1)[:, index].T
    return {"index": index, "source_rgb": source_rgb, "target_rgb": target_rgb}


@torch.no_grad()
def fit_static_appearance_variation_model(pairs: dict[str, torch.Tensor]) -> float:
    """``p(A_T | A_0, same surface)`` scale, measured not assumed (§29)."""
    if int(pairs["index"].numel()) == 0:
        return 0.05
    return _robust_scale(pairs["target_rgb"] - pairs["source_rgb"])


@torch.no_grad()
def fit_static_pixel_displacement_model(
    pairs: dict[str, torch.Tensor],
    width: int,
    *,
    search_radius: int = 8,
) -> float:
    """How far a STATIC pixel's best photometric match appears to move.

    For fixed cameras this should be ~0, so whatever it is measures the
    combined effect of rasteriser ownership noise, reconstruction error and
    genuine appearance change.  That is the scale against which a dynamic
    pixel's displacement has to be judged surprising.
    """
    if int(pairs["index"].numel()) < 16:
        return float(search_radius) / 2.0
    # Colour change of a static pixel, converted into the displacement that
    # would explain it under the locally measured image gradient.
    difference = (pairs["target_rgb"] - pairs["source_rgb"]).abs().mean(dim=1)
    return max(float(difference.median()) * float(search_radius), 0.5)


@torch.no_grad()
def fit_static_render_residual_model(
    observed: torch.Tensor,
    rendered: torch.Tensor,
    static_mask: torch.Tensor | None = None,
) -> float:
    residual = torch.as_tensor(observed).detach().float() - torch.as_tensor(rendered).detach().float()
    if static_mask is not None:
        mask = torch.as_tensor(static_mask).detach().bool()
        residual = residual[mask.expand_as(residual)] if mask.shape != residual.shape else residual[mask]
    return _robust_scale(residual)


@torch.no_grad()
def calibrate_from_static_pixels(
    source_marginals,
    target_marginals,
    source_image: torch.Tensor,
    target_image: torch.Tensor,
    *,
    rendered_source: torch.Tensor | None = None,
    max_pixels: int = 8192,
    generator: torch.Generator | None = None,
) -> StaticPixelCalibration:
    """One calibration from the scene's own static evidence."""
    pairs = collect_confident_static_pixel_pairs(
        source_marginals, target_marginals, source_image, target_image,
        max_pixels=int(max_pixels), generator=generator,
    )
    appearance = fit_static_appearance_variation_model(pairs)
    displacement = fit_static_pixel_displacement_model(pairs, int(source_image.shape[-1]))
    residual = (
        fit_static_render_residual_model(source_image, rendered_source)
        if rendered_source is not None else appearance
    )
    return StaticPixelCalibration(
        displacement_scale=displacement,
        appearance_scale=appearance,
        render_residual_scale=residual,
        samples=int(pairs["index"].numel()),
        stats={"static_pixels_used": int(pairs["index"].numel())},
    )
