"""P8 -- "which physical hypothesis explains the actual images?"

The final observation-level test, run through the project's own rasteriser.
It does not compensate for poor metric or rigid consistency (§30): it only
chooses between hypotheses that already passed those, and only by comparing
how well each explains the pixels.

Moving a patch means moving its Gaussians AND turning them.  3DGS stores
orientation as a per-Gaussian quaternion and the covariance is
``R diag(s^2) R^T`` built from it, so rotating the patch means composing the
hypothesis rotation onto those quaternions -- ``Sigma' = R Sigma R^T`` comes
out of that for free.  Translating the means alone would leave every
Gaussian's orientation behind, and a correct transform would then render
visibly wrong and be rejected for a reason that has nothing to do with the
hypothesis being tested.

There is no RGB/depth weight to expose, and that is not a stylistic choice.
Residual scale is FITTED on regions the pipeline is confident are static, so
each channel arrives already expressed in units of its own noise and a
log-likelihood in one is directly comparable with a log-likelihood in
another.  Independent observations multiply, so their logs add.  A tuned
coefficient is what you need when you skip that step.

A robust likelihood family is used rather than a Gaussian because render
residuals have heavy tails: specularities, thin structure and disocclusions
produce large residuals everywhere, including under a correct hypothesis,
and under a Gaussian those few pixels decide the comparison.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Callable, Sequence

import torch

from utils.rigid_transform_utils import matrix_to_quaternion, quaternion_multiply

EPS = 1.0e-12


@dataclass
class ResidualModel:
    """A fitted robust residual model in one observation channel."""

    scale: float                 # robust sigma, in the channel's own units
    degrees_of_freedom: float
    family: str
    samples: int

    def log_likelihood(self, residual: torch.Tensor) -> torch.Tensor:
        residual = torch.as_tensor(residual).detach().float()
        scale = max(float(self.scale), 1.0e-8)
        if self.family == "laplace":
            return -residual.abs() / scale - math.log(2.0 * scale)
        dof = float(self.degrees_of_freedom)
        normaliser = (
            math.lgamma((dof + 1.0) / 2.0) - math.lgamma(dof / 2.0)
            - 0.5 * math.log(dof * math.pi) - math.log(scale)
        )
        return normaliser - (dof + 1.0) / 2.0 * torch.log1p((residual / scale) ** 2 / dof)


@dataclass
class HypothesisComparison:
    """Static versus dynamic, in log-likelihood units."""

    static_log_likelihood: float
    dynamic_log_likelihood: float
    log_likelihood_ratio: float
    prefers_dynamic: bool
    indistinguishable: bool
    per_camera: list[float] = field(default_factory=list)
    channels: dict[str, float] = field(default_factory=dict)
    pixels: int = 0


def _robust_scale(residual: torch.Tensor) -> float:
    value = torch.as_tensor(residual).detach().float().reshape(-1)
    value = value[torch.isfinite(value)]
    if int(value.numel()) == 0:
        return 1.0
    scale = float((value - value.median()).abs().median() * 1.4826)
    return scale if scale > 0 else (float(value.abs().mean()) or 1.0)


# --------------------------------------------------------------------------
# Applying a rigid hypothesis to real Gaussians
# --------------------------------------------------------------------------
@torch.no_grad()
def transform_patch_gaussians(
    xyz: torch.Tensor,
    rotation_quaternion: torch.Tensor,
    patch_mask: torch.Tensor,
    rotation: torch.Tensor,
    translation: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """``(means, quaternions)`` with one patch moved by ``(R, t)``.

    The quaternion composition is what rotates each Gaussian's covariance:
    3DGS builds ``Sigma = R_q diag(s^2) R_q^T`` from the stored quaternion,
    so pre-multiplying by the hypothesis rotation gives exactly
    ``Sigma' = R Sigma R^T`` without ever forming a 3x3 covariance.
    """
    xyz = torch.as_tensor(xyz).detach()
    quaternion = torch.as_tensor(rotation_quaternion).detach().to(xyz.device)
    mask = torch.as_tensor(patch_mask).detach().to(device=xyz.device).bool().reshape(-1)
    rotation = torch.as_tensor(rotation).detach().to(xyz).reshape(3, 3)
    translation = torch.as_tensor(translation).detach().to(xyz).reshape(3)

    moved_xyz = xyz.clone()
    moved_xyz[mask] = xyz[mask] @ rotation.T + translation

    hypothesis_quaternion = matrix_to_quaternion(rotation[None].to(quaternion.dtype))
    moved_quaternion = quaternion.clone()
    moved_quaternion[mask] = quaternion_multiply(
        hypothesis_quaternion.expand(int(mask.sum()), 4), quaternion[mask]
    )
    return moved_xyz, moved_quaternion


@torch.no_grad()
def render_rigid_patch_hypothesis(
    camera,
    gaussians,
    pipe,
    background: torch.Tensor,
    patch_mask: torch.Tensor,
    rotation: torch.Tensor,
    translation: torch.Tensor,
    *,
    render_fn: Callable | None = None,
) -> torch.Tensor:
    """Render the scene through the project rasteriser with one patch moved.

    Returns the rendered image as ``[3, H, W]``.
    """
    if render_fn is None:
        from gaussian_renderer import render as render_fn

    moved_xyz, moved_quaternion = transform_patch_gaussians(
        gaussians.get_xyz.detach(), gaussians.get_rotation.detach(),
        patch_mask, rotation, translation,
    )
    output = render_fn(
        camera, gaussians, pipe, background,
        means3D_override=moved_xyz,
        rotations_override=moved_quaternion,
    )
    return output["render"]


@torch.no_grad()
def render_static(camera, gaussians, pipe, background: torch.Tensor, *, render_fn: Callable | None = None) -> torch.Tensor:
    """The static hypothesis: the scene exactly as reconstructed."""
    if render_fn is None:
        from gaussian_renderer import render as render_fn
    return render_fn(camera, gaussians, pipe, background)["render"]


def observed_image(camera) -> torch.Tensor:
    """The camera's ground-truth image as ``[3, H, W]`` in ``[0, 1]``."""
    image = camera.original_image if hasattr(camera, "original_image") else camera.image
    return torch.as_tensor(image).detach().float().clamp(0.0, 1.0)


# --------------------------------------------------------------------------
# Residual models and the comparison
# --------------------------------------------------------------------------
@torch.no_grad()
def fit_predictive_rgb_residual_model(
    cameras: Sequence,
    gaussians,
    pipe,
    background: torch.Tensor,
    *,
    static_gaussian_mask: torch.Tensor | None = None,
    max_cameras: int = 8,
    family: str = "student_t",
    degrees_of_freedom: float = 4.0,
    render_fn: Callable | None = None,
) -> ResidualModel:
    """§28: calibrate on HELD-OUT views, and only where the scene is static.

    Fitting on the views the endpoint 3DGS was optimised on measures
    reconstruction overfit, not predictive observation noise -- it gave
    2.9e-4 on USB_100109, so small that any displacement produced an
    astronomical penalty and the test could only ever answer "static".

    The frame-T cameras are held out with respect to the frame-0 model, so
    rendering frame-0 Gaussians into them and measuring the residual where
    nothing moved gives what §28 asks for: predictive model error plus sensor
    noise.  Restricting to static Gaussians matters -- calibrating over
    moving regions would fold the motion into the noise model and make every
    hypothesis look equally good.
    """
    residuals: list[torch.Tensor] = []
    for camera in list(cameras)[: int(max_cameras)]:
        rendered = render_static(camera, gaussians, pipe, background, render_fn=render_fn)
        residual = observed_image(camera).to(rendered.device) - rendered
        if static_gaussian_mask is not None:
            # Pixels the moving Gaussians cannot reach: rendering with only
            # the static subset and keeping where that agrees with the full
            # render isolates the regions the motion does not touch.
            static_only = render_static_subset(
                camera, gaussians, pipe, background, static_gaussian_mask, render_fn=render_fn
            )
            untouched = (rendered - static_only).abs().amax(dim=0) <= 1.0e-3
            residual = residual[untouched.expand_as(residual)]
        residuals.append(residual.reshape(-1))
        del rendered
    if not residuals:
        return ResidualModel(1.0, float(degrees_of_freedom), str(family), 0)
    pooled = torch.cat(residuals, 0)
    model = ResidualModel(
        scale=_robust_scale(pooled),
        degrees_of_freedom=float(degrees_of_freedom),
        family=str(family),
        samples=int(pooled.numel()),
    )
    del pooled, residuals
    return model


@torch.no_grad()
def render_static_subset(
    camera, gaussians, pipe, background: torch.Tensor,
    keep_mask: torch.Tensor, *, render_fn: Callable | None = None,
) -> torch.Tensor:
    """Render with the non-kept Gaussians pushed far away (a cheap subset render)."""
    if render_fn is None:
        from gaussian_renderer import render as render_fn
    xyz = gaussians.get_xyz.detach()
    mask = torch.as_tensor(keep_mask).to(device=xyz.device).bool().reshape(-1)
    moved = xyz.clone()
    moved[~mask] = moved[~mask] + 1.0e6
    output = render_fn(camera, gaussians, pipe, background, means3D_override=moved)["render"]
    del moved
    return output


@torch.no_grad()
def compare_contrastive_hypotheses(
    cameras: Sequence,
    source_gaussians,
    pipe,
    background: torch.Tensor,
    patch_mask: torch.Tensor,
    rotation: torch.Tensor,
    translation: torch.Tensor,
    model: ResidualModel,
    *,
    max_cameras: int = 8,
    indistinguishable_nats: float = 2.0,
    footprint_threshold: float = 1.0e-3,
    render_fn: Callable | None = None,
) -> HypothesisComparison:
    """The contrastive test: both hypotheses scored against the SAME held-out images.

    Both renders come from the frame-0 Gaussians; the only difference is
    whether the patch is left where it was or moved by the hypothesis.  Both
    are compared against frame-T observations, which neither render was fitted
    to.  That is what makes the comparison informative: the static hypothesis
    is not a free win from having been trained on these pixels, because it
    was not.

    Scored only on the pixel footprint the two hypotheses disagree about --
    over the whole image a local motion is drowned by ~300k identical
    background pixels and every hypothesis scores the same.
    """
    static_total = dynamic_total = 0.0
    per_camera: list[float] = []
    pixels = 0

    for camera in list(cameras)[: int(max_cameras)]:
        static = render_static(camera, source_gaussians, pipe, background, render_fn=render_fn)
        moved = render_rigid_patch_hypothesis(
            camera, source_gaussians, pipe, background, patch_mask, rotation, translation,
            render_fn=render_fn,
        )
        footprint = (static - moved).abs().amax(dim=0) > float(footprint_threshold)
        if not bool(footprint.any()):
            per_camera.append(0.0)
            del static, moved
            continue
        observed = observed_image(camera).to(static.device)
        static_ll = compute_hypothesis_log_likelihood(observed, static, model, region_mask=footprint)
        dynamic_ll = compute_hypothesis_log_likelihood(observed, moved, model, region_mask=footprint)
        static_total += static_ll
        dynamic_total += dynamic_ll
        per_camera.append(dynamic_ll - static_ll)
        pixels += int(footprint.sum())
        del static, moved, observed, footprint

    ratio = dynamic_total - static_total
    return HypothesisComparison(
        static_log_likelihood=static_total,
        dynamic_log_likelihood=dynamic_total,
        log_likelihood_ratio=ratio,
        prefers_dynamic=bool(ratio > float(indistinguishable_nats)),
        indistinguishable=bool(abs(ratio) <= float(indistinguishable_nats)),
        per_camera=per_camera,
        channels={"rgb": ratio},
        pixels=pixels,
    )


@torch.no_grad()
def fit_static_rgb_residual_model(
    cameras: Sequence,
    gaussians,
    pipe,
    background: torch.Tensor,
    *,
    static_pixel_mask: torch.Tensor | None = None,
    max_cameras: int = 8,
    family: str = "student_t",
    degrees_of_freedom: float = 4.0,
    render_fn: Callable | None = None,
) -> ResidualModel:
    """Fit the residual scale where the reconstruction is already believed.

    Fitted against the STATIC render on regions no hypothesis moves, because
    that is the only place the residual is pure observation noise.  Fitting
    over the whole image would absorb the motion -- the very thing being
    tested -- into the noise model and make every hypothesis look equally
    good.
    """
    residuals: list[torch.Tensor] = []
    for camera in list(cameras)[: int(max_cameras)]:
        rendered = render_static(camera, gaussians, pipe, background, render_fn=render_fn)
        residual = observed_image(camera).to(rendered.device) - rendered
        if static_pixel_mask is not None:
            mask = torch.as_tensor(static_pixel_mask).to(device=residual.device).bool()
            residual = residual[mask.expand_as(residual)]
        residuals.append(residual.reshape(-1))
    if not residuals:
        return ResidualModel(1.0, float(degrees_of_freedom), str(family), 0)
    pooled = torch.cat(residuals, 0)
    return ResidualModel(
        scale=_robust_scale(pooled),
        degrees_of_freedom=float(degrees_of_freedom),
        family=str(family),
        samples=int(pooled.numel()),
    )


@torch.no_grad()
def fit_static_depth_residual_model(
    observed_depth: torch.Tensor,
    rendered_depth: torch.Tensor,
    *,
    family: str = "student_t",
    degrees_of_freedom: float = 4.0,
) -> ResidualModel:
    """The same fit for depth, kept as its OWN sensor likelihood.

    Separate because depth noise has nothing to do with RGB noise; combining
    them before each is normalised is exactly what forces an arbitrary
    coefficient into the comparison.
    """
    residual = torch.as_tensor(observed_depth).detach().float() - torch.as_tensor(rendered_depth).detach().float()
    return ResidualModel(
        scale=_robust_scale(residual),
        degrees_of_freedom=float(degrees_of_freedom),
        family=str(family),
        samples=int(residual.numel()),
    )


@torch.no_grad()
def compute_hypothesis_log_likelihood(
    observed: torch.Tensor,
    rendered: torch.Tensor,
    model: ResidualModel,
    *,
    region_mask: torch.Tensor | None = None,
) -> float:
    """Total log-likelihood of an observation under one hypothesis.

    Scored on a region mask so both hypotheses see exactly the same pixels;
    comparing totals over different supports would let whichever hypothesis
    covers more image win regardless of fit.
    """
    residual = torch.as_tensor(observed).detach().float() - torch.as_tensor(rendered).detach().float()
    if region_mask is not None:
        mask = torch.as_tensor(region_mask).to(device=residual.device).bool()
        residual = residual[mask.expand_as(residual)] if mask.shape != residual.shape else residual[mask]
    if int(residual.numel()) == 0:
        return 0.0
    return float(model.log_likelihood(residual).sum())


@torch.no_grad()
def patch_pixel_footprint(
    camera,
    gaussians,
    pipe,
    background: torch.Tensor,
    patch_mask: torch.Tensor,
    rotation: torch.Tensor,
    translation: torch.Tensor,
    *,
    threshold: float = 1.0e-3,
    render_fn: Callable | None = None,
) -> torch.Tensor:
    """Pixels either hypothesis can touch, as ``[H, W]``.

    The union of where the patch is and where it would move to.  Scoring the
    whole image instead would bury a local motion under a hundred thousand
    identical background pixels, and the likelihood ratio would go to zero
    for every hypothesis alike.
    """
    static = render_static(camera, gaussians, pipe, background, render_fn=render_fn)
    moved = render_rigid_patch_hypothesis(
        camera, gaussians, pipe, background, patch_mask, rotation, translation, render_fn=render_fn
    )
    return (static - moved).abs().amax(dim=0) > float(threshold)


@torch.no_grad()
def compare_static_vs_dynamic_likelihood(
    cameras: Sequence,
    gaussians,
    pipe,
    background: torch.Tensor,
    patch_mask: torch.Tensor,
    rotation: torch.Tensor,
    translation: torch.Tensor,
    model: ResidualModel,
    *,
    max_cameras: int = 8,
    indistinguishable_nats: float = 2.0,
    footprint_threshold: float = 1.0e-3,
    render_fn: Callable | None = None,
) -> HypothesisComparison:
    """§31/§33: accept the moved hypothesis only if it explains the images better.

    ``indistinguishable_nats`` is a fixed significance convention (§46), not
    a tuned weight: a ratio inside it means the images cannot separate the
    two hypotheses, and §33 requires retaining ambiguity rather than picking.
    """
    static_total = dynamic_total = 0.0
    per_camera: list[float] = []
    pixels = 0

    for camera in list(cameras)[: int(max_cameras)]:
        static = render_static(camera, gaussians, pipe, background, render_fn=render_fn)
        moved = render_rigid_patch_hypothesis(
            camera, gaussians, pipe, background, patch_mask, rotation, translation, render_fn=render_fn
        )
        footprint = (static - moved).abs().amax(dim=0) > float(footprint_threshold)
        if not bool(footprint.any()):
            # The hypothesis is invisible from here, so this camera carries no
            # evidence either way and must not contribute a zero that dilutes
            # the cameras that can actually see it.
            per_camera.append(0.0)
            continue
        observed = observed_image(camera).to(static.device)
        static_ll = compute_hypothesis_log_likelihood(observed, static, model, region_mask=footprint)
        dynamic_ll = compute_hypothesis_log_likelihood(observed, moved, model, region_mask=footprint)
        static_total += static_ll
        dynamic_total += dynamic_ll
        per_camera.append(dynamic_ll - static_ll)
        pixels += int(footprint.sum())

    ratio = dynamic_total - static_total
    return HypothesisComparison(
        static_log_likelihood=static_total,
        dynamic_log_likelihood=dynamic_total,
        log_likelihood_ratio=ratio,
        prefers_dynamic=bool(ratio > float(indistinguishable_nats)),
        indistinguishable=bool(abs(ratio) <= float(indistinguishable_nats)),
        per_camera=per_camera,
        channels={"rgb": ratio},
        pixels=pixels,
    )


@torch.no_grad()
def compare_images_only(
    observed: dict[str, torch.Tensor],
    static_render: dict[str, torch.Tensor],
    dynamic_render: dict[str, torch.Tensor],
    models: dict[str, ResidualModel],
    *,
    region_mask: torch.Tensor | None = None,
    indistinguishable_nats: float = 2.0,
) -> HypothesisComparison:
    """The same comparison on images supplied directly, for tests and depth.

    Channels are summed as log-likelihoods, the one combination that needs no
    weight.
    """
    channels: dict[str, float] = {}
    static_total = dynamic_total = 0.0
    for channel, model in models.items():
        if channel not in observed or channel not in static_render or channel not in dynamic_render:
            continue
        static_ll = compute_hypothesis_log_likelihood(
            observed[channel], static_render[channel], model, region_mask=region_mask
        )
        dynamic_ll = compute_hypothesis_log_likelihood(
            observed[channel], dynamic_render[channel], model, region_mask=region_mask
        )
        channels[channel] = dynamic_ll - static_ll
        static_total += static_ll
        dynamic_total += dynamic_ll

    ratio = dynamic_total - static_total
    return HypothesisComparison(
        static_log_likelihood=static_total,
        dynamic_log_likelihood=dynamic_total,
        log_likelihood_ratio=ratio,
        prefers_dynamic=bool(ratio > float(indistinguishable_nats)),
        indistinguishable=bool(abs(ratio) <= float(indistinguishable_nats)),
        channels=channels,
    )
