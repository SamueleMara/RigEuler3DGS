"""Per-hypothesis SE(3) uncertainty, from the patch pair that produced it.

§5 of the audit, and the single most important correction: the uncertainty of
a transform must come from how well THIS patch pair determines SE(3), never
from the scatter of transforms across the scene.

The scene-wide estimate is circular.  Bad transforms produce a huge empirical
covariance, a huge covariance makes every pair of bad transforms mutually
compatible, and the consistency test then certifies exactly the noise it was
meant to reject.  A per-hypothesis covariance cannot do that: a patch that
does not determine a rotation reports a large variance in that direction and
is believed less, whatever the rest of the scene is doing.

Observability is a spectrum, not a boolean (§4).  A near-planar patch can be
numerically rank 3 and still leave rotation about its normal essentially
unconstrained; ``full_rank = True`` hides precisely the direction that will
break downstream.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch

from utils.se3_hypothesis_consistency import se3_exp, se3_log

EPS = 1.0e-12


@dataclass
class TransformUncertainty:
    """Tangent covariance of one hypothesis, plus what it can and cannot see."""

    covariance: torch.Tensor          # [B, 6, 6] in (omega, v) tangent order
    spectrum: torch.Tensor            # [B, 6] eigenvalues, ascending
    observability_rank: torch.Tensor  # [B] directions actually constrained
    patch_spectrum: torch.Tensor      # [B, 3] source patch covariance eigenvalues
    signed_volume: torch.Tensor       # [B] chirality observability (§9)
    stats: dict[str, Any] = field(default_factory=dict)


@torch.no_grad()
def se3_adjoint(rotation: torch.Tensor, translation: torch.Tensor) -> torch.Tensor:
    """``Ad_T`` for the ``(omega, v)`` tangent ordering used throughout.

    Needed because two hypotheses' covariances live in their own tangent
    frames; adding them without transporting one into the other's frame
    compares quantities expressed in different bases (§18).
    """
    rotation = torch.as_tensor(rotation).detach().float()
    translation = torch.as_tensor(translation).detach().float()
    b = int(rotation.shape[0])
    skew = torch.zeros((b, 3, 3), device=rotation.device, dtype=rotation.dtype)
    skew[:, 0, 1], skew[:, 0, 2] = -translation[:, 2], translation[:, 1]
    skew[:, 1, 0], skew[:, 1, 2] = translation[:, 2], -translation[:, 0]
    skew[:, 2, 0], skew[:, 2, 1] = -translation[:, 1], translation[:, 0]

    adjoint = torch.zeros((b, 6, 6), device=rotation.device, dtype=rotation.dtype)
    adjoint[:, :3, :3] = rotation
    adjoint[:, 3:, 3:] = rotation
    adjoint[:, 3:, :3] = skew @ rotation
    return adjoint


@torch.no_grad()
def compose_inverse_transform(
    rotation_a: torch.Tensor, translation_a: torch.Tensor,
    rotation_b: torch.Tensor, translation_b: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """``T_a^{-1} T_b``, batched."""
    inverse_rotation = rotation_a.transpose(1, 2)
    rotation = inverse_rotation @ rotation_b
    translation = torch.einsum("bij,bj->bi", inverse_rotation, translation_b - translation_a)
    return rotation, translation


@torch.no_grad()
def patch_observability_spectrum(
    points: torch.Tensor,
    valid: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """``(eigenvalues descending, signed-volume magnitude)`` of a patch.

    The signed volume is §9's chirality diagnostic.  Proper rotations preserve
    its sign and reflections flip it, but on a near-planar patch it is ~0 and
    chirality is simply unobservable -- which must be reported, not decided.
    """
    from utils.metric_patch_graph import batched_eigvalsh

    points = torch.as_tensor(points).detach().float()
    weight = valid.float()[:, :, None]
    mean = (points * weight).sum(1) / weight.sum(1).clamp_min(EPS)
    centred = (points - mean[:, None, :]) * weight
    covariance = centred.transpose(1, 2) @ centred / weight.sum(1).clamp_min(EPS)[:, None]
    eigenvalues = batched_eigvalsh(covariance.double()).clamp_min(0.0).float().flip(-1)

    # |det| of the principal frame scaled by its own extent: dimensionless,
    # and ~0 exactly when the patch is too flat to have a handedness.
    volume = (eigenvalues[:, 0] * eigenvalues[:, 1] * eigenvalues[:, 2]).clamp_min(0.0).sqrt()
    normaliser = eigenvalues[:, 0].clamp_min(EPS).pow(1.5)
    return eigenvalues, (volume / normaliser).clamp(0.0, 1.0)


@torch.no_grad()
def bootstrap_transform_covariance(
    source_points: torch.Tensor,
    target_points: torch.Tensor,
    coupling: torch.Tensor,
    rotation: torch.Tensor,
    translation: torch.Tensor,
    *,
    draws: int = 24,
    generator: torch.Generator | None = None,
    ridge: float = 1.0e-9,
) -> torch.Tensor:
    """§6: resample the coupling's own support and see how far the fit moves.

    Resampling the matched support -- not the points, not the scene -- is what
    makes this an observability estimate.  A patch whose transform is pinned
    down by its geometry gives nearly the same fit from any subsample; one
    that is pinned in two directions and loose in the third produces a
    covariance that is small in two eigen-directions and large in the third,
    which is exactly the statement the consistency test needs.
    """
    from utils.local_rigid_hypothesis import weighted_kabsch_from_soft_coupling

    source_points = torch.as_tensor(source_points).detach().float()
    target_points = torch.as_tensor(target_points).detach().float()
    coupling = torch.as_tensor(coupling).detach().float()
    b, m, n = coupling.shape
    device = coupling.device
    generator = generator or torch.Generator(device="cpu").manual_seed(0)

    flat = coupling.reshape(b, m * n).clamp_min(0.0)
    total = flat.sum(dim=1, keepdim=True)
    degenerate = (total <= EPS).reshape(-1)
    probability = torch.where(degenerate[:, None], torch.ones_like(flat) / float(m * n), flat / total.clamp_min(EPS))

    perturbations = []
    for _ in range(max(2, int(draws))):
        # Multinomial over the coupling's own edges: an edge with more mass is
        # resampled more often, so the bootstrap respects the transport plan
        # rather than treating every pair as equally believed.
        picks = torch.multinomial(
            probability.cpu(), num_samples=m * n, replacement=True, generator=generator
        ).to(device)
        resampled = torch.zeros_like(flat)
        resampled.scatter_add_(1, picks, torch.ones_like(picks, dtype=flat.dtype))
        resampled = (resampled / float(m * n)).reshape(b, m, n)

        draw_rotation, draw_translation, _mass, _bad = weighted_kabsch_from_soft_coupling(
            source_points, target_points, resampled
        )
        relative_r, relative_t = compose_inverse_transform(
            rotation, translation, draw_rotation, draw_translation
        )
        perturbations.append(se3_log(relative_r, relative_t))

    stacked = torch.stack(perturbations, dim=0)                     # [D, B, 6]
    centred = stacked - stacked.mean(dim=0, keepdim=True)
    covariance = torch.einsum("dbi,dbj->bij", centred, centred) / float(max(1, stacked.shape[0] - 1))
    identity = torch.eye(6, device=device, dtype=covariance.dtype)[None]
    # A hypothesis whose bootstrap did not move at all still is not infinitely
    # certain; the ridge is machine tolerance, not a tuned prior.
    return covariance + float(ridge) * identity


@torch.no_grad()
def hessian_transform_covariance(
    source_points: torch.Tensor,
    target_points: torch.Tensor,
    coupling: torch.Tensor,
    rotation: torch.Tensor,
    translation: torch.Tensor,
    *,
    ridge: float = 1.0e-9,
) -> torch.Tensor:
    """§7: ``Sigma ~ (J^T W J)^+`` around the optimum -- cheaper than bootstrap.

    For the point-to-point rigid residual the Jacobian of a moved point with
    respect to the tangent is ``[-skew(Rx + t), I]``, so the information
    matrix is available in closed form and no autograd is needed.  The
    pseudoinverse is deliberate: an unobservable direction must keep its huge
    uncertainty rather than be forced invertible (§7).
    """
    source_points = torch.as_tensor(source_points).detach().float()
    target_points = torch.as_tensor(target_points).detach().float()
    coupling = torch.as_tensor(coupling).detach().float()
    b, m, _ = coupling.shape

    moved = torch.einsum("bij,bkj->bki", rotation, source_points) + translation[:, None, :]
    weight = coupling.sum(dim=2)                                    # [B, m]

    skew = torch.zeros((b, m, 3, 3), device=moved.device, dtype=moved.dtype)
    skew[..., 0, 1], skew[..., 0, 2] = -moved[..., 2], moved[..., 1]
    skew[..., 1, 0], skew[..., 1, 2] = moved[..., 2], -moved[..., 0]
    skew[..., 2, 0], skew[..., 2, 1] = -moved[..., 1], moved[..., 0]

    identity = torch.eye(3, device=moved.device, dtype=moved.dtype).expand(b, m, 3, 3)
    jacobian = torch.cat((-skew, identity), dim=-1)                 # [B, m, 3, 6]
    information = torch.einsum("bmij,bm,bmik->bjk", jacobian, weight, jacobian)
    return torch.linalg.pinv(information.double()).float() + float(ridge) * torch.eye(
        6, device=moved.device
    )[None]


@torch.no_grad()
def effective_rank(covariance: torch.Tensor, *, ratio: float = 1.0e-4) -> tuple[torch.Tensor, torch.Tensor]:
    """``(rank, ascending spectrum)`` of each covariance.

    A direction whose variance is enormous relative to the tightest one is
    unobservable, and the chi-square test downstream must use the number of
    directions actually constrained rather than a blanket six (§19).
    """
    from utils.metric_patch_graph import batched_eigvalsh

    spectrum = batched_eigvalsh(torch.as_tensor(covariance).detach().double()).clamp_min(0.0).float()
    smallest = spectrum[:, 0].clamp_min(EPS)
    observable = spectrum <= (smallest[:, None] / float(ratio))
    return observable.sum(dim=1), spectrum


@torch.no_grad()
def estimate_transform_uncertainty(
    source_points: torch.Tensor,
    target_points: torch.Tensor,
    source_valid: torch.Tensor,
    coupling: torch.Tensor,
    rotation: torch.Tensor,
    translation: torch.Tensor,
    *,
    method: str = "bootstrap",
    draws: int = 24,
    generator: torch.Generator | None = None,
) -> TransformUncertainty:
    """Per-hypothesis uncertainty plus the observability it came from."""
    if method == "bootstrap":
        covariance = bootstrap_transform_covariance(
            source_points, target_points, coupling, rotation, translation,
            draws=int(draws), generator=generator,
        )
    elif method == "hessian":
        covariance = hessian_transform_covariance(
            source_points, target_points, coupling, rotation, translation
        )
    else:
        raise ValueError(f"unknown uncertainty method {method!r}")

    rank, spectrum = effective_rank(covariance)
    patch_spectrum, volume = patch_observability_spectrum(source_points, source_valid)
    return TransformUncertainty(
        covariance=covariance,
        spectrum=spectrum,
        observability_rank=rank,
        patch_spectrum=patch_spectrum,
        signed_volume=volume,
        stats={
            "method": method,
            "median_rank": float(rank.float().median()),
            "median_log_volume": float(torch.log(spectrum.clamp_min(EPS)).sum(dim=1).median()),
            "median_chirality": float(volume.median()),
        },
    )
