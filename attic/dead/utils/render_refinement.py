"""R7 -- the renderer refines a transform instead of only judging it.

§20: P8 stops being "accept/reject once the geometry is done" and becomes
"recover precise transform parameters against the actual images".  That is
where dynamic accuracy has to come from: the metric and SE(3) stages fix a
hypothesis to within the patch's own observability, and only the images can
tighten it further.

    T* = argmax_T  log p(O | T, G_static, G_dynamic)

Maximum likelihood, with the likelihood supplied by the calibrated residual
model.  No render lambda is introduced (§20) because nothing is being traded
against anything -- there is one objective.

The search is a local pattern search in se(3) rather than a gradient step.
The rasteriser's gradient with respect to a rigid transform is available but
the likelihood surface is dominated by discrete visibility changes, which
makes small-step gradients unreliable exactly where refinement matters; a
shrinking pattern search follows the surface it actually has.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Sequence

import torch

from utils.render_hypothesis_likelihood import (
    ResidualModel,
    compute_hypothesis_log_likelihood,
    observed_image,
    render_rigid_patch_hypothesis,
    render_static,
)
from utils.se3_hypothesis_consistency import se3_exp, se3_log


@dataclass
class RefinementResult:
    """Before and after, so the gain can be read rather than assumed."""

    rotation: torch.Tensor
    translation: torch.Tensor
    log_likelihood_ratio_before: float
    log_likelihood_ratio_after: float
    iterations: int
    evaluations: int
    improved: bool
    stats: dict[str, Any] = field(default_factory=dict)


@torch.no_grad()
def render_dynamic_hypothesis(camera, gaussians, pipe, background, patch_mask, rotation, translation, **kwargs):
    return render_rigid_patch_hypothesis(
        camera, gaussians, pipe, background, patch_mask, rotation, translation, **kwargs
    )


@torch.no_grad()
def render_static_hypothesis(camera, gaussians, pipe, background, **kwargs):
    return render_static(camera, gaussians, pipe, background, **kwargs)


@torch.no_grad()
def compare_hypotheses_by_llr(
    cameras: Sequence,
    gaussians,
    pipe,
    background: torch.Tensor,
    patch_mask: torch.Tensor,
    rotation: torch.Tensor,
    translation: torch.Tensor,
    model: ResidualModel,
    *,
    max_cameras: int = 4,
    footprint_threshold: float = 1.0e-3,
) -> tuple[float, int]:
    """``(LLR, pixels)`` on the footprint where the hypotheses disagree.

    Restricted to that footprint because over a full frame a local motion is
    swamped by hundreds of thousands of identical background pixels and every
    hypothesis scores alike.
    """
    static_total = dynamic_total = 0.0
    pixels = 0
    for camera in list(cameras)[: int(max_cameras)]:
        static = render_static(camera, gaussians, pipe, background)
        moved = render_rigid_patch_hypothesis(
            camera, gaussians, pipe, background, patch_mask, rotation, translation
        )
        footprint = (static - moved).abs().amax(dim=0) > float(footprint_threshold)
        if not bool(footprint.any()):
            del static, moved
            continue
        observed = observed_image(camera).to(static.device)
        static_total += compute_hypothesis_log_likelihood(observed, static, model, region_mask=footprint)
        dynamic_total += compute_hypothesis_log_likelihood(observed, moved, model, region_mask=footprint)
        pixels += int(footprint.sum())
        del static, moved, observed, footprint
    return dynamic_total - static_total, pixels


@torch.no_grad()
def refine_rigid_transform_by_render_likelihood(
    cameras: Sequence,
    gaussians,
    pipe,
    background: torch.Tensor,
    patch_mask: torch.Tensor,
    rotation: torch.Tensor,
    translation: torch.Tensor,
    model: ResidualModel,
    *,
    max_cameras: int = 4,
    iterations: int = 12,
    initial_step: float = 0.05,
    shrink: float = 0.6,
    patch_scale: float = 1.0,
) -> RefinementResult:
    """Shrinking pattern search in ``se(3)`` on the render log-likelihood.

    Steps are taken in the tangent space so rotation and translation are
    perturbed in one consistent parameterisation -- stepping ``R`` and ``t``
    separately would need a ratio between radians and metres, which is exactly
    the tuned weight this architecture forbids.  The translation half of the
    step is scaled by the patch's own size, so the search is in units the
    patch defines rather than in scene metres.
    """
    device = gaussians.get_xyz.device
    rotation = torch.as_tensor(rotation).detach().float().reshape(1, 3, 3).to(device)
    translation = torch.as_tensor(translation).detach().float().reshape(1, 3).to(device)

    def score(r, t) -> float:
        return compare_hypotheses_by_llr(
            cameras, gaussians, pipe, background, patch_mask, r[0], t[0], model,
            max_cameras=int(max_cameras),
        )[0]

    best_score = score(rotation, translation)
    initial_score = best_score
    step = float(initial_step)
    evaluations = 1
    scale = torch.tensor(
        [1.0, 1.0, 1.0, float(patch_scale), float(patch_scale), float(patch_scale)], device=device
    )

    for _ in range(max(1, int(iterations))):
        improved = False
        for axis in range(6):
            for sign in (1.0, -1.0):
                delta = torch.zeros((1, 6), device=device)
                delta[0, axis] = sign * step * float(scale[axis])
                step_r, step_t = se3_exp(delta)
                trial_r = rotation @ step_r
                trial_t = translation + torch.einsum("bij,bj->bi", rotation, step_t)
                trial = score(trial_r, trial_t)
                evaluations += 1
                if trial > best_score:
                    rotation, translation, best_score = trial_r, trial_t, trial
                    improved = True
        if not improved:
            # Converged at this resolution; halve and keep going rather than
            # stopping, so the search ends at the finest scale it can use.
            step *= float(shrink)
            if step < 1.0e-4:
                break

    return RefinementResult(
        rotation=rotation[0], translation=translation[0],
        log_likelihood_ratio_before=float(initial_score),
        log_likelihood_ratio_after=float(best_score),
        iterations=int(iterations), evaluations=evaluations,
        improved=bool(best_score > initial_score),
        stats={
            "llr_gain": float(best_score - initial_score),
            "final_step": step,
            "tangent_move": float(torch.linalg.norm(se3_log(rotation.cpu(), translation.cpu()))),
        },
    )
