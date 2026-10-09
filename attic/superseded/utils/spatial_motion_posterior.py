"""P0 -- "did this surface stay where it was?", answered as a posterior.

Stage 1 of the physical hierarchy.  It measures absolute spatial mismatch at
both endpoints and converts it into a motion probability WITHOUT a threshold,
a temperature or a slope: a two-component Gaussian mixture is fitted to the
log mismatch and the posterior of the high-mismatch component is the answer.

The point of the mixture is that the static mode is not a modelling choice --
it is a real, measurable thing.  Two independently trained 3DGS models
disagree by some amount everywhere, including where nothing moved, and that
disagreement is what makes a large mismatch surprising.  Fitting it means the
same code serves a scene where 5% moved and one where 50% did, with nothing
to tune in between.

This stage answers only whether compatible geometry remains locally.  It
does NOT say where anything went: `argmin_j ||x_i - y_j||` is not a dynamic
correspondence and must never be read as one (§11).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import torch

from utils import progress

EPS = 1.0e-8


@dataclass
class MixtureFit:
    """A fitted two-component 1-D Gaussian mixture on log mismatch."""

    static_mean: float
    static_std: float
    dynamic_mean: float
    dynamic_std: float
    static_weight: float
    dynamic_weight: float
    log_likelihood: float
    single_log_likelihood: float
    bimodal: bool
    separation: float
    iterations: int

    def posterior(self, log_value: torch.Tensor) -> torch.Tensor:
        """``P(D | z)`` for new log-mismatch values."""
        log_s = _log_normal(log_value, self.static_mean, self.static_std) + math.log(max(self.static_weight, EPS))
        log_d = _log_normal(log_value, self.dynamic_mean, self.dynamic_std) + math.log(max(self.dynamic_weight, EPS))
        return torch.sigmoid(log_d - log_s)


@dataclass
class MotionPosterior:
    """Both endpoints' mismatch, their mixtures and the resulting posteriors."""

    forward_mismatch: torch.Tensor          # [N0] E_{0->T}
    backward_mismatch: torch.Tensor         # [NT] E_{T->0}
    source_dynamic_posterior: torch.Tensor  # [N0] q^0 = P(D | z)
    target_dynamic_posterior: torch.Tensor  # [NT] q^T
    source_map_dynamic: torch.Tensor        # [N0] bool, MAP class
    target_map_dynamic: torch.Tensor        # [NT] bool
    source_scale: torch.Tensor              # [N0] local length unit
    target_scale: torch.Tensor              # [NT]
    origin: torch.Tensor                    # [1, 3] shared centring origin
    source_mixture: MixtureFit | None = None
    target_mixture: MixtureFit | None = None
    stats: dict[str, Any] = field(default_factory=dict)


def _log_normal(value: torch.Tensor, mean: float, std: float) -> torch.Tensor:
    std = max(float(std), EPS)
    return -0.5 * ((value - float(mean)) / std) ** 2 - math.log(std) - 0.5 * math.log(2.0 * math.pi)


@torch.no_grad()
def _nearest_squared_distance(
    query: torch.Tensor, reference: torch.Tensor, *, chunk_size: int = 1024, desc: str = "",
) -> torch.Tensor:
    out: list[torch.Tensor] = []
    starts = list(range(0, int(query.shape[0]), max(1, int(chunk_size))))
    for start in progress.bar(starts, desc=desc or "mismatch", unit="chunk"):
        d2 = torch.cdist(query[start:start + int(chunk_size)], reference).square()
        out.append(d2.min(dim=1).values)
    return torch.cat(out, 0).clamp_min(0.0)


@torch.no_grad()
def compute_bidirectional_spatial_mismatch(
    source_xyz: torch.Tensor,
    target_xyz: torch.Tensor,
    source_scale: torch.Tensor,
    target_scale: torch.Tensor,
    *,
    chunk_size: int = 1024,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """``(E_{0->T}, E_{T->0}, origin)``, both centred on the same origin.

    §6: the shift is a numerical correction, not a parameter.  ``cdist``
    evaluates ``|a|^2 + |b|^2 - 2a.b``, which cancels catastrophically in
    float32 for a cloud far from the origin -- a COLMAP frame centred 500
    units out reported ~0.96 local scales of mismatch for geometry that had
    not moved at all.  Both clouds are shifted by the same vector, so every
    distance is preserved exactly.
    """
    source_xyz = torch.as_tensor(source_xyz).detach()
    target_xyz = torch.as_tensor(target_xyz).detach().to(source_xyz)
    origin = source_xyz.mean(dim=0, keepdim=True)
    a, b = source_xyz - origin, target_xyz - origin

    forward = _nearest_squared_distance(a, b, chunk_size=int(chunk_size), desc="  mismatch 0->T")
    forward = forward / torch.as_tensor(source_scale).detach().to(forward).reshape(-1).clamp_min(EPS).square()
    backward = _nearest_squared_distance(b, a, chunk_size=int(chunk_size), desc="  mismatch T->0")
    # Normalised by the TARGET's own scale: the two models densify
    # independently, so borrowing the source's spacing would make the two
    # directions incomparable exactly where they differ most.
    backward = backward / torch.as_tensor(target_scale).detach().to(backward).reshape(-1).clamp_min(EPS).square()
    return forward, backward, origin


@torch.no_grad()
def fit_two_component_log_mismatch_mixture(
    mismatch: torch.Tensor,
    *,
    max_iterations: int = 200,
    tolerance: float = 1.0e-6,
    min_std: float = 1.0e-3,
) -> MixtureFit:
    """Deterministic EM for a two-component mixture on ``z = log(E + eps)``.

    Initialised from the 25th and 75th percentiles (§7), so the fit is
    reproducible and carries no random seed.  ``max_iterations`` and
    ``tolerance`` are convergence controls, not physical parameters: the
    result is the EM fixed point either way.

    Bimodality is reported, not assumed.  §8 requires the detector to say it
    is uncertain rather than have a separation manufactured for it, so the
    two-component fit is compared against a single Gaussian by BIC -- if one
    Gaussian explains the data as well, there is no dynamic mode to find.
    """
    value = torch.as_tensor(mismatch).detach().reshape(-1).double()
    z = torch.log(value.clamp_min(0.0) + EPS)
    z = z[torch.isfinite(z)]
    n = int(z.numel())
    if n < 4:
        return MixtureFit(0.0, 1.0, 0.0, 1.0, 0.5, 0.5, 0.0, 0.0, False, 0.0, 0)

    quantiles = torch.quantile(z, torch.tensor([0.25, 0.75], dtype=z.dtype, device=z.device))
    mu = torch.stack((quantiles[0], quantiles[1]))
    sigma = torch.full((2,), float(z.std().clamp_min(min_std)), dtype=z.dtype)
    weight = torch.full((2,), 0.5, dtype=z.dtype)

    previous = -float("inf")
    iterations = 0
    for iterations in range(1, int(max_iterations) + 1):
        log_p = torch.stack([
            _log_normal(z, float(mu[k]), float(sigma[k])) + torch.log(weight[k].clamp_min(EPS))
            for k in range(2)
        ], dim=1)                                                     # [n, 2]
        total = torch.logsumexp(log_p, dim=1)
        responsibility = torch.exp(log_p - total[:, None])
        mass = responsibility.sum(dim=0).clamp_min(EPS)

        weight = mass / float(n)
        mu = (responsibility * z[:, None]).sum(dim=0) / mass
        variance = (responsibility * (z[:, None] - mu[None, :]) ** 2).sum(dim=0) / mass
        sigma = variance.clamp_min(float(min_std) ** 2).sqrt()

        log_likelihood = float(total.sum())
        if abs(log_likelihood - previous) < float(tolerance) * max(1.0, abs(log_likelihood)):
            previous = log_likelihood
            break
        previous = log_likelihood

    order = torch.argsort(mu)                       # the high-mismatch mode is dynamic
    low, high = int(order[0]), int(order[1])
    single_ll = float(_log_normal(z, float(z.mean()), float(z.std().clamp_min(min_std))).sum())
    # BIC: 5 free parameters for the mixture, 2 for a single Gaussian.
    bic_two = -2.0 * previous + 5.0 * math.log(n)
    bic_one = -2.0 * single_ll + 2.0 * math.log(n)
    pooled = float(((sigma[low] ** 2 + sigma[high] ** 2) / 2.0).sqrt().clamp_min(min_std))
    separation = float((mu[high] - mu[low]) / pooled)

    return MixtureFit(
        static_mean=float(mu[low]),
        static_std=float(sigma[low]),
        dynamic_mean=float(mu[high]),
        dynamic_std=float(sigma[high]),
        static_weight=float(weight[low]),
        dynamic_weight=float(weight[high]),
        log_likelihood=previous,
        single_log_likelihood=single_ll,
        bimodal=bool(bic_two < bic_one),
        separation=separation,
        iterations=int(iterations),
    )


@torch.no_grad()
def compute_motion_posterior(
    source_xyz: torch.Tensor,
    target_xyz: torch.Tensor,
    source_scale: torch.Tensor,
    target_scale: torch.Tensor,
    *,
    chunk_size: int = 1024,
) -> MotionPosterior:
    """Stage 1 end to end: mismatch, mixtures, posteriors, MAP sets.

    The two endpoints are fitted INDEPENDENTLY.  They have different point
    counts, densification histories and local scales, so a mixture fitted on
    one and applied to the other would silently make the two directions mean
    different things.
    """
    forward, backward, origin = compute_bidirectional_spatial_mismatch(
        source_xyz, target_xyz, source_scale, target_scale, chunk_size=int(chunk_size)
    )
    source_mixture = fit_two_component_log_mismatch_mixture(forward)
    target_mixture = fit_two_component_log_mismatch_mixture(backward)

    q0 = source_mixture.posterior(torch.log(forward.double().clamp_min(0.0) + EPS)).to(forward.dtype)
    qT = target_mixture.posterior(torch.log(backward.double().clamp_min(0.0) + EPS)).to(backward.dtype)
    return MotionPosterior(
        forward_mismatch=forward,
        backward_mismatch=backward,
        source_dynamic_posterior=q0,
        target_dynamic_posterior=qT,
        # MAP, i.e. P(D|z) > P(S|z). Not a tuned cut: 0.5 is where the two
        # fitted components cross.
        source_map_dynamic=q0 > 0.5,
        target_map_dynamic=qT > 0.5,
        source_scale=torch.as_tensor(source_scale).detach().reshape(-1),
        target_scale=torch.as_tensor(target_scale).detach().reshape(-1),
        origin=origin,
        source_mixture=source_mixture,
        target_mixture=target_mixture,
        stats={
            "source_map_dynamic_fraction": float((q0 > 0.5).float().mean()),
            "target_map_dynamic_fraction": float((qT > 0.5).float().mean()),
            "source_bimodal": bool(source_mixture.bimodal),
            "target_bimodal": bool(target_mixture.bimodal),
            "source_separation": float(source_mixture.separation),
            "target_separation": float(target_mixture.separation),
            "spatial_detector_uncertain": not (source_mixture.bimodal and target_mixture.bimodal),
        },
    )
