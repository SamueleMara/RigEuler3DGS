"""Losses that carry feedback between the modules of the closed loop.

Two directions of feedback live here.

**Killing and assignment -> flow matching** (spec 12).  FM keeps its own
objective, so it cannot simply copy the physics, but it is additionally pulled
toward the current rigid field where a Gaussian is confidently moving and toward
zero where it is confidently static.  Both pulls are weighted by ``r^gamma`` and
both read the physics through ``detach``: FM learns from Killing, but no
gradient runs back into the Killing solver.

**Renderer -> residual diffusion** (spec 19).  The critic is supervised by how
well the corrected positions render, plus weak regularisers that keep the
correction small and locally coherent without smoothing across a real part
boundary.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

_EPS = 1.0e-8


@dataclass
class FlowMatchingFeedbackWeights:
    base: float = 1.0
    killing: float = 0.1
    physical_3d: float = 0.25
    static: float = 0.1
    gamma: float = 2.0


def flow_matching_feedback_loss(
    *,
    predicted_velocity: torch.Tensor,
    killing_velocity: torch.Tensor,
    responsibilities: torch.Tensor,
    fm_confidence: torch.Tensor | None = None,
    killing_valid=None,
    physical_velocity: torch.Tensor | None = None,
    physical_confidence: torch.Tensor | None = None,
    static_feedback_gate: torch.Tensor | None = None,
    weights: FlowMatchingFeedbackWeights | None = None,
) -> dict:
    """Pull FM toward the current physics, weighted by how sure the label is.

    ``predicted_velocity`` is the only argument that carries gradient.  The
    Killing velocity and the responsibilities are detached here rather than
    relying on the caller, because a gradient reaching the Killing solver would
    let FM reshape the physics to match itself, which is precisely the
    collusion the closed loop has to avoid.

    A part that has not been fitted contributes no pull at all.  Giving it the
    zero-velocity target would silently duplicate the static term and drag
    genuinely moving points toward rest before the first fit.
    """
    weights = weights or FlowMatchingFeedbackWeights()
    velocity = predicted_velocity
    physics = killing_velocity.detach().to(velocity)
    share = responsibilities.detach().to(velocity)
    gamma = float(weights.gamma)
    confidence = (
        torch.ones(velocity.shape[:-1], device=velocity.device, dtype=velocity.dtype)
        if fm_confidence is None
        else fm_confidence.detach().to(velocity).clamp_min(0.0)
    )

    static_term = velocity.sum() * 0.0
    if static_feedback_gate is not None:
        gate = static_feedback_gate.detach().to(device=velocity.device, dtype=velocity.dtype)
        if gate.ndim == 1:
            gate = gate[None]
        static_weight = share[:, 0].pow(gamma)[None] * gate.clamp(0.0, 1.0)
        static_term = (static_weight * velocity.square().sum(-1)).sum() / static_weight.sum().clamp_min(_EPS)

    killing_term = velocity.sum() * 0.0
    total_weight = 0.0
    for part in range(physics.shape[0]):
        if killing_valid is not None and not bool(killing_valid[part]):
            continue
        weight = share[:, part + 1].pow(gamma)[None] * confidence
        difference = velocity - physics[part]
        killing_term = killing_term + (weight * difference.square().sum(-1)).mean()
        total_weight += 1.0
    if total_weight > 0:
        killing_term = killing_term / total_weight

    physical_term = velocity.sum() * 0.0
    if physical_velocity is not None and float(weights.physical_3d) > 0.0:
        target = physical_velocity.detach().to(velocity)
        if target.shape != velocity.shape:
            raise ValueError("physical_velocity must match predicted_velocity")
        pconf = (
            torch.ones(velocity.shape[:-1], device=velocity.device, dtype=velocity.dtype)
            if physical_confidence is None
            else physical_confidence.detach().to(velocity).clamp_min(0.0)
        )
        physical_term = (
            pconf * (velocity - target).square().sum(-1)
        ).sum() / pconf.sum().clamp_min(_EPS)

    total = (
        weights.killing * killing_term
        + weights.static * static_term
        + weights.physical_3d * physical_term
    )
    return {
        "total": total,
        "killing_term": killing_term,
        "physical_3d_term": physical_term,
        "static_term": static_term,
        "weights": {
            "killing": float(weights.killing),
            "physical_3d": float(weights.physical_3d),
            "static": float(weights.static),
            "gamma": gamma,
        },
    }


@dataclass
class ResidualRegularizerWeights:
    magnitude: float = 0.01
    spatial: float = 0.0
    huber_delta: float = 1.0


def residual_regularizers(
    residual: torch.Tensor,
    *,
    neighbour_scale: float,
    knn_idx: torch.Tensor | None = None,
    same_label: torch.Tensor | None = None,
    weights: ResidualRegularizerWeights | None = None,
) -> dict:
    """Keep the correction small and locally coherent, but not across a boundary.

    The magnitude term is Huber rather than quadratic so a genuinely large
    correction -- exactly the signal a mislabelled point should produce -- is
    discouraged but not suppressed.

    The spatial term is masked by ``same_label``: smoothing the residual across
    a real part boundary would erase the discontinuity that identifies the
    boundary, so neighbours with different current labels are simply excluded
    from it rather than being smoothed with a lower weight.
    """
    weights = weights or ResidualRegularizerWeights()
    scale = max(float(neighbour_scale), _EPS)
    normalised = residual / scale
    norm = normalised.norm(dim=-1)
    delta = float(weights.huber_delta)
    magnitude = torch.where(
        norm <= delta, 0.5 * norm.square(), delta * (norm - 0.5 * delta)
    ).mean()

    spatial = residual.sum() * 0.0
    if weights.spatial > 0 and knn_idx is not None:
        neighbours = knn_idx.to(device=residual.device, dtype=torch.long)
        difference = (normalised[:, None] - normalised[neighbours]).norm(dim=-1)
        if same_label is not None:
            mask = same_label.to(device=residual.device, dtype=difference.dtype)
            spatial = (difference * mask).sum() / mask.sum().clamp_min(_EPS)
        else:
            spatial = difference.mean()

    return {
        "total": weights.magnitude * magnitude + weights.spatial * spatial,
        "magnitude": magnitude,
        "spatial": spatial,
    }


def same_label_neighbour_mask(hard_labels: torch.Tensor, knn_idx: torch.Tensor) -> torch.Tensor:
    """``[N,K]`` mask of neighbours currently carrying the same label."""
    labels = hard_labels.reshape(-1, 1)
    return (labels == hard_labels[knn_idx.to(labels.device)]).to(torch.float32)


@dataclass
class RenderLossWeights:
    rgb_l1: float = 0.8
    rgb_dssim: float = 0.2
    depth: float = 0.1


def render_supervision_loss(
    rendered: torch.Tensor,
    target: torch.Tensor,
    *,
    rendered_depth: torch.Tensor | None = None,
    target_depth: torch.Tensor | None = None,
    depth_mask: torch.Tensor | None = None,
    weights: RenderLossWeights | None = None,
) -> dict:
    """RGB L1 + DSSIM, plus inverse-depth consistency where depth is reliable.

    Reuses the repository's existing SSIM so the closed loop is scored the same
    way the static training was, rather than introducing a second definition of
    "renders well" that could not be compared with anything already measured.
    """
    from utils.loss_utils import l1_loss, ssim

    weights = weights or RenderLossWeights()
    rgb_l1 = l1_loss(rendered, target)
    dssim = 1.0 - ssim(rendered, target)
    total = weights.rgb_l1 * rgb_l1 + weights.rgb_dssim * dssim
    depth_term = rendered.sum() * 0.0
    if (
        weights.depth > 0
        and rendered_depth is not None
        and target_depth is not None
    ):
        difference = (rendered_depth - target_depth).abs()
        if depth_mask is not None:
            mask = depth_mask.to(difference)
            depth_term = (difference * mask).sum() / mask.sum().clamp_min(_EPS)
        else:
            depth_term = difference.mean()
        total = total + weights.depth * depth_term
    return {
        "total": total,
        "rgb": rgb_l1,
        "dssim": dssim,
        "depth": depth_term,
    }


def supervised_residual_loss(
    predicted_positions: torch.Tensor, clean_positions: torch.Tensor, *, neighbour_scale: float
) -> torch.Tensor:
    """Direct supervision when the correct hypothesis is known.

    Used only in the corruption-training path of spec 18, where a deliberately
    wrong proposal was built from a known-good one: the critic is then told the
    exact correction it should have produced.  This is what teaches it that a
    mislabelled point needs a *whole displacement* undone rather than a small
    nudge, which renderer supervision alone learns slowly.
    """
    scale = max(float(neighbour_scale), _EPS)
    return ((predicted_positions - clean_positions) / scale).square().sum(-1).mean()
