"""Turning the three motion modalities into ``K+1`` assignment evidence.

Each modality answers the same question in its own units -- "how badly does
label ``l`` explain this Gaussian?" -- and the responsibilities are the softmax
of their combination.

``compute_diffusion_transition_energy``
    The residual diffusion left a correction relative to the *current* label.
    Comparing it to the displacement that would move the point from its current
    label to a candidate one turns the critic directly into a label vote.

``compute_fm_killing_label_energy``
    How well the FM transport agrees with each part's rigid field, or with
    standing still.

``compute_observed_velocity_label_energy``
    The same question against the rasterized multiview velocity, in the metric
    of its own normal matrix.

``compute_assignment_graph_energy``
    Neighbouring Gaussians should usually share a label.

``robust_normalize_and_combine``
    The three data energies live on different scales, so they are each divided
    by their own robust spread before being summed.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

_EPS = 1.0e-8


@dataclass
class AssignmentWeights:
    """Relative trust in each modality, plus the cost of claiming motion."""

    diffusion: float = 1.0
    flow_matching: float = 1.0
    fm_3d: float = 1.0
    fm_render: float = 0.25
    observed: float = 1.0
    obs_render: float = 0.25
    target_render: float = 0.05
    graph: float = 0.05
    #: Component-level counterfactual rendering (spec 28).  The only modality
    #: that can speak for a group of Gaussians none of whose members has its own
    #: velocity observation.
    counterfactual: float = 1.0
    #: Added to every moving label.  If static and moving explain the evidence
    #: equally well, prefer static: an object at rest is the null hypothesis and
    #: inventing motion to fit noise is the failure this pipeline is most prone
    #: to.
    move_activation_cost: float = 0.0


@torch.no_grad()
def compute_diffusion_transition_energy(
    *,
    canonical_fg: torch.Tensor,
    hard_labels: torch.Tensor,
    killing_positions: torch.Tensor,
    residual_mean: torch.Tensor,
    residual_cov: torch.Tensor,
    residual_confidence: torch.Tensor,
    target_frames,
    covariance_aware: bool = True,
    ridge: float = 1.0e-6,
) -> torch.Tensor:
    """Score each candidate label by how well it explains the diffusion residual.

    The residual is defined relative to the current hypothesis, so the
    displacement needed to switch Gaussian ``i`` from its current label ``z_i``
    to candidate ``l`` at frame ``t`` is

        delta = d_l(t) - d_{z_i}(t),      d_0 = 0 exactly.

    A residual that matches that delta is the critic saying "this point wants to
    be label ``l``".  With ``covariance_aware`` the mismatch is measured in the
    inverse-covariance metric of the trials, so a direction the trials disagreed
    about counts for less than one they agreed on -- which is the difference
    between a systematic correction and leftover injected noise.
    """
    points = torch.as_tensor(canonical_fg)
    device, dtype = points.device, points.dtype
    n = int(points.shape[0])
    k = int(killing_positions.shape[0])
    labels = k + 1
    frames = [int(value) for value in target_frames]

    current = torch.as_tensor(hard_labels, device=device).long().clamp(0, k)
    # Displacement of every candidate label at every requested frame.
    displacement = torch.zeros((len(frames), labels, n, 3), device=device, dtype=dtype)
    for column, frame in enumerate(frames):
        for part in range(k):
            displacement[column, part + 1] = (
                killing_positions[part, frame].to(device=device, dtype=dtype) - points
            )
        # label 0 stays exactly zero: static is not a fitted field.

    index = current.reshape(1, 1, n, 1).expand(len(frames), 1, n, 3)
    current_displacement = torch.gather(displacement, 1, index)
    transition = displacement - current_displacement            # [F,K+1,N,3]

    residual = residual_mean[frames].to(device=device, dtype=dtype)[:, None]   # [F,1,N,3]
    mismatch = residual - transition                                            # [F,K+1,N,3]
    confidence = residual_confidence[frames].to(device=device, dtype=dtype)[:, None]

    if covariance_aware:
        cov = residual_cov[frames].to(device=device, dtype=dtype)
        eye = torch.eye(3, device=device, dtype=dtype)
        trace = torch.diagonal(cov, dim1=-2, dim2=-1).sum(-1).clamp_min(0.0)
        # Scale the ridge to the covariance so it regularises rather than
        # dominating when the trials happened to agree very closely.
        damping = (float(ridge) * (trace / 3.0) + float(ridge)).clamp_min(_EPS)
        precision = torch.linalg.inv(cov + damping[..., None, None] * eye)
        quadratic = torch.einsum("flni,fnij,flnj->fln", mismatch, precision, mismatch)
    else:
        quadratic = mismatch.square().sum(-1)
    # [F,K+1,N] summed over frames, then transposed to the [N,K+1] every other
    # energy and the softmax use.
    energy = (confidence * quadratic).sum(0).transpose(0, 1).contiguous()
    if tuple(energy.shape) != (n, labels):
        raise RuntimeError(f"diffusion energy must be [{n},{labels}], got {tuple(energy.shape)}")
    return energy


@torch.no_grad()
def compute_fm_killing_label_energy(
    *,
    fm_velocity: torch.Tensor,
    fm_confidence: torch.Tensor,
    killing_velocity: torch.Tensor,
    killing_valid=None,
) -> torch.Tensor:
    """How well FM transport matches standing still, or each rigid field.

    Static costs ``||v_FM||^2``: a point FM says is moving pays to be called
    static.  Moving part ``k`` costs ``||v_FM - v_K,k||^2``.  Deliberately a
    *vector* comparison rather than a scalar dynamic probability -- the whole
    reason FM is in this pipeline is that it proposes a direction, and collapsing
    it to "how likely is this to be moving" throws that away.

    A part that has not been fitted cannot be compared against, so it is given
    the static cost instead of a free zero.
    """
    velocity = torch.as_tensor(fm_velocity)
    device, dtype = velocity.device, velocity.dtype
    confidence = fm_confidence.to(device=device, dtype=dtype).clamp_min(0.0)
    k = int(killing_velocity.shape[0])
    static = (confidence * velocity.square().sum(-1)).sum(0)
    columns = [static]
    for part in range(k):
        if killing_valid is not None and not bool(killing_valid[part]):
            columns.append(static.clone())
            continue
        difference = velocity - killing_velocity[part].to(device=device, dtype=dtype)
        columns.append((confidence * difference.square().sum(-1)).sum(0))
    energy = torch.stack(columns, dim=-1)
    if tuple(energy.shape) != (velocity.shape[1], k + 1):
        raise RuntimeError("flow-matching energy must be [N_F,K+1]")
    return energy


@torch.no_grad()
def compute_observed_velocity_label_energy(
    *,
    observed_velocity: torch.Tensor,
    observed_normal_matrix: torch.Tensor,
    observed_valid: torch.Tensor,
    killing_velocity: torch.Tensor,
    killing_valid=None,
) -> torch.Tensor:
    """The same comparison against the rasterized multiview velocity.

    Measured in the normal matrix's own metric, so a velocity the cameras only
    pinned down in two directions is only judged in those two.  Invalid
    observations contribute nothing at all rather than a zero velocity, because
    "no measurement" and "measured to be still" are different claims and only
    the second is evidence for static.
    """
    velocity = torch.as_tensor(observed_velocity)
    device, dtype = velocity.device, velocity.dtype
    valid = observed_valid.to(device=device)
    normal = observed_normal_matrix.to(device=device, dtype=dtype)
    normal = torch.where(valid[..., None, None], normal, torch.zeros_like(normal))
    k = int(killing_velocity.shape[0])

    def quadratic(residual: torch.Tensor) -> torch.Tensor:
        return torch.einsum("tni,tnij,tnj->tn", residual, normal, residual).sum(0)

    static = quadratic(velocity)
    columns = [static]
    for part in range(k):
        if killing_valid is not None and not bool(killing_valid[part]):
            columns.append(static.clone())
            continue
        columns.append(quadratic(killing_velocity[part].to(device=device, dtype=dtype) - velocity))
    energy = torch.stack(columns, dim=-1)
    if tuple(energy.shape) != (velocity.shape[1], k + 1):
        raise RuntimeError("observed-velocity energy must be [N_F,K+1]")
    return energy


@torch.no_grad()
def build_foreground_graph(
    canonical_fg: torch.Tensor, knn_idx: torch.Tensor, sigma_scale: float = 1.0
) -> dict[str, torch.Tensor]:
    """Geometric neighbour weights on the foreground-only KNN.

    ``sigma`` is set from the median neighbour distance of this cloud rather
    than as an absolute length, so the same setting behaves the same on scenes
    of different physical size.
    """
    points = torch.as_tensor(canonical_fg)
    neighbours = knn_idx.to(device=points.device, dtype=torch.long)
    offset = points[neighbours] - points[:, None]
    distance = offset.norm(dim=-1)
    finite = distance[distance > 0]
    sigma = float(finite.median()) * float(sigma_scale) if finite.numel() else 1.0
    weight = torch.exp(-distance.square() / max(sigma, _EPS) ** 2)
    return {"neighbours": neighbours, "weight": weight, "sigma": sigma}


@torch.no_grad()
def compute_assignment_graph_energy(
    responsibilities: torch.Tensor, graph: dict[str, torch.Tensor]
) -> torch.Tensor:
    """Per-label cost of disagreeing with the neighbourhood.

    The energy of label ``l`` at point ``i`` is the weighted mass its
    neighbours put on *other* labels, so a point surrounded by static
    neighbours pays to be moving.  Written per-label rather than as a single
    smoothness scalar because the softmax below needs a ``[N,K+1]`` term.
    """
    share = torch.as_tensor(responsibilities)
    neighbours, weight = graph["neighbours"], graph["weight"].to(share)
    total = weight.sum(-1, keepdim=True).clamp_min(_EPS)
    neighbour_mean = (weight[..., None] * share[neighbours]).sum(1) / total
    return 1.0 - neighbour_mean


@torch.no_grad()
def robust_scale(energy: torch.Tensor) -> float:
    """The modality's own unit, taken from the gaps it actually resolves.

    The three modalities are in squared metres per second, in pixel-metric units
    and in dimensionless residual units, so summing them raw lets whichever
    happens to be largest decide everything.  What has to be normalised is not
    the absolute energy but the *gap between labels*, since only differences
    reach the softmax.

    And the gap has to be measured over the Gaussians where the modality says
    anything at all.  On a real scene most Gaussians have no observation in a
    given modality, so all their labels tie at zero: over half the rows are
    exactly zero, the median gap is zero, the median absolute deviation is zero,
    and a fallback to the full range then hands the scale to the single largest
    outlier.  Measured on USB that put the rasterized multiview velocity -- the
    most reliable evidence there is, at SNR 14 -- on a scale of 3.6e6, whose
    normalised contribution was a median of 0.0000 and a 90th percentile of
    0.0085.  It had been divided into irrelevance, leaving the untrained critic
    deciding every label by itself.

    So the scale is a high quantile of the non-degenerate gaps: robust to
    outliers, and immune to rows that carry no information.
    """
    value = torch.as_tensor(energy)
    if value.ndim < 2:
        value = value.reshape(-1, 1)
    finite = value[torch.isfinite(value).all(-1)]
    if finite.numel() == 0:
        return 1.0
    gap = finite.amax(-1) - finite.amin(-1)
    # Exactly zero means this modality does not distinguish the labels for that
    # Gaussian, which is the case to exclude.  Anything strictly positive is
    # information, however small: filtering on an absolute epsilon here would
    # discard every gap of a modality whose energies are uniformly tiny and
    # leave it unscaled and silent.
    informative = gap[gap > 0]
    if informative.numel() == 0:
        return 1.0
    # The 75th percentile of the resolved gaps: a typical discriminative
    # magnitude rather than the largest one.  No absolute floor -- a strictly
    # positive gap is a valid unit however small it is, and clamping to an
    # epsilon would leave a modality whose energies are all tiny unscaled and
    # therefore silent, which is the failure this function exists to prevent.
    scale = float(torch.quantile(informative, 0.75))
    return scale if scale > 0.0 else 1.0


@torch.no_grad()
def _gap_reliability(value: torch.Tensor, min_gap: float) -> torch.Tensor:
    """Reliability from whether a modality separates labels at all."""
    energy = torch.as_tensor(value)
    if energy.ndim != 2:
        energy = energy.reshape(energy.shape[0], -1)
    gap = energy.amax(-1) - energy.amin(-1)
    if float(min_gap) <= 0.0:
        return (gap > 0).to(energy.dtype)
    return (gap / float(min_gap)).clamp(0.0, 1.0)


def _as_reliability(value, like: torch.Tensor) -> torch.Tensor:
    if value is None:
        return torch.ones((like.shape[0], 1), device=like.device, dtype=like.dtype)
    rel = torch.as_tensor(value, device=like.device, dtype=like.dtype)
    if rel.ndim == 0:
        return rel.reshape(1, 1).expand(like.shape[0], 1)
    if rel.ndim == 1:
        if rel.shape[0] != like.shape[0]:
            raise ValueError("per-point reliability must match the energy rows")
        return rel.reshape(-1, 1)
    if rel.shape != like.shape:
        raise ValueError("per-label reliability must match the energy shape")
    return rel


@torch.no_grad()
def robust_normalize_and_combine(
    *,
    energy_diffusion: torch.Tensor | None = None,
    energy_fm: torch.Tensor | None = None,
    energy_fm_3d: torch.Tensor | None = None,
    energy_fm_render: torch.Tensor | None = None,
    energy_observed: torch.Tensor | None = None,
    energy_obs_render: torch.Tensor | None = None,
    energy_target_render: torch.Tensor | None = None,
    energy_graph: torch.Tensor | None = None,
    energy_counterfactual: torch.Tensor | None = None,
    energy_reliability: dict | None = None,
    weights: AssignmentWeights | None = None,
    min_reliable_gap: float = 1.0e-8,
) -> dict:
    """Normalise each modality by its own spread, then sum with the move cost.

    Each energy is also shifted so its per-Gaussian minimum is zero before
    weighting: only *differences* between labels carry information, and leaving
    the offsets in would let a modality with a large constant term dominate the
    sum without ever preferring one label over another.
    """
    weights = weights or AssignmentWeights()
    scales, reliability_report, total = {}, {}, None
    parts = {
        "diffusion": (energy_diffusion, weights.diffusion),
        "fm": (energy_fm, weights.flow_matching),
        "fm_3d": (energy_fm_3d, weights.fm_3d),
        "fm_render": (energy_fm_render, weights.fm_render),
        "observed": (energy_observed, weights.observed),
        "obs_render": (energy_obs_render, weights.obs_render),
        "target_render": (energy_target_render, weights.target_render),
        "graph": (energy_graph, weights.graph),
        "counterfactual": (energy_counterfactual, weights.counterfactual),
    }
    normalized = {}
    energy_reliability = dict(energy_reliability or {})
    for name, (energy, weight) in parts.items():
        if energy is None:
            scales[name] = None
            reliability_report[name] = None
            continue
        value = torch.as_tensor(energy)
        shifted = value - value.amin(dim=-1, keepdim=True)
        gap_reliability = _gap_reliability(shifted, float(min_reliable_gap))
        source_reliability = _as_reliability(energy_reliability.get(name), shifted)
        reliability = (source_reliability * gap_reliability[:, None]).clamp(0.0, 1.0)
        if float(weight) == 0.0:
            # Still report its scale so a zero-weighted modality can be
            # inspected -- that is how a warming-up critic's vote is watched
            # before it is trusted -- but keep it out of the sum.
            scales[name] = robust_scale(shifted)
            reliability_report[name] = float(reliability.mean().detach()) if reliability.numel() else 0.0
            continue
        scale = robust_scale(shifted)
        scales[name] = scale
        reliability_report[name] = float(reliability.mean().detach()) if reliability.numel() else 0.0
        contribution = float(weight) * reliability * (shifted / scale)
        normalized[name] = contribution
        total = contribution if total is None else total + contribution
    if total is None:
        raise ValueError("assignment needs at least one active energy modality")
    activation = torch.zeros_like(total)
    activation[:, 1:] = float(weights.move_activation_cost)
    total = total + activation
    return {
        "energy_total": total,
        "normalized": normalized,
        "scales": scales,
        "reliability": reliability_report,
        "move_activation_cost": float(weights.move_activation_cost),
    }


def assignment_temperature(
    outer_iteration: int, schedule=(1.0, 0.8, 0.6, 0.45, 0.3, 0.2, 0.15)
) -> float:
    """Annealed softmax temperature, holding the final value once reached.

    High early so the assignment stays genuinely soft while the Killing fields
    are still poor, low later so it can commit once the physics has settled.
    Hardening early is the failure mode this schedule exists to prevent: a
    premature hard label feeds a contaminated core into the next Killing fit.
    """
    values = list(schedule)
    if not values:
        raise ValueError("the temperature schedule must not be empty")
    return float(values[min(int(outer_iteration), len(values) - 1)])


@torch.no_grad()
def update_responsibilities(
    energy_total: torch.Tensor, temperature: float, previous: torch.Tensor | None = None
) -> dict:
    """Softmax the combined energy into new responsibilities.

    Returns the change from the previous assignment as well, which is the
    loop's convergence signal.
    """
    energy = torch.as_tensor(energy_total)
    updated = torch.softmax(-energy / max(float(temperature), _EPS), dim=-1)
    delta = (
        float((updated - previous).abs().mean()) if previous is not None else float("inf")
    )
    entropy = -(updated * updated.clamp_min(_EPS).log()).sum(-1)
    return {
        "responsibilities": updated,
        "hard_labels": updated.argmax(-1),
        "assignment_entropy": entropy,
        "responsibility_delta": delta,
        "temperature": float(temperature),
    }


@torch.no_grad()
def ambiguous_mask(
    responsibilities: torch.Tensor, confidence_threshold: float = 0.8, margin: float = 0.15
) -> torch.Tensor:
    """Points no label owns confidently, kept as a first-class population.

    Reported rather than forced: an early cycle is *expected* to have a
    meaningful ambiguous set, and its disappearance is a sign of premature
    hardening rather than of success.
    """
    ordered = torch.as_tensor(responsibilities).sort(-1, descending=True).values
    return (ordered[:, 0] < float(confidence_threshold)) | (
        (ordered[:, 0] - ordered[:, 1]) < float(margin)
    )
