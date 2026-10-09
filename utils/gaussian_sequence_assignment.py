"""Persistent whole-sequence material evidence and rigid-assisted assignment."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import chain

import torch


STATIC_CORE, DYNAMIC_CORE, DYNAMIC_CANDIDATE, UNKNOWN = range(4)
STATE_NAMES = ("static_core", "dynamic_core", "dynamic_candidate", "unknown")
EVIDENCE_CHANNELS = (
    "present", "occluded", "vacated", "static_flow", "dynamic_flow", "informative"
)
DEPTH_EVIDENCE_CHANNELS = (
    "static",
    "dynamic",
    "unknown",
    "observable",
    "occlusion_fraction",
    "vacated_ratio",
    "present_ratio",
)


@dataclass(frozen=True)
class CoreThresholds:
    dynamic_score: float = 0.72
    dynamic_time_fraction: float = 0.60
    dynamic_camera_fraction: float = 0.40
    static_score: float = 0.88
    min_observations: int = 12


@torch.no_grad()
def accumulate_sequence_evidence_to_gaussians(
    gaussians,
    pipe,
    records: list[dict],
    *,
    num_cameras: int,
    num_steps: int,
    min_observation_weight: float = 0.0,
) -> dict[str, torch.Tensor]:
    """Reverse-render pixel evidence records into persistent ``[N,C,T,F]`` tensors."""
    from gaussian_renderer import accumulate_pixel_features_to_gaussians

    n_points = int(gaussians.get_xyz.shape[0])
    iterator = iter(records)
    try:
        first = next(iterator)
    except StopIteration as exc:
        raise ValueError("at least one evidence record is required") from exc
    channels = int(torch.as_tensor(first["pixel_features"]).shape[0])
    device = gaussians.get_xyz.device
    feature_sum = torch.zeros((n_points, num_cameras, num_steps, channels), device=device)
    weight_sum = torch.zeros((n_points, num_cameras, num_steps), device=device)
    count = torch.zeros((n_points, num_cameras, num_steps), device=device, dtype=torch.long)
    for record in chain((first,), iterator):
        camera_index, step_index = int(record["camera_index"]), int(record["step_index"])
        result = accumulate_pixel_features_to_gaussians(
            record["camera"], gaussians, pipe,
            record["pixel_features"], record["pixel_confidence"],
            min_observation_weight=min_observation_weight,
        )
        feature_sum[:, camera_index, step_index] += result["feature_sum"]
        weight_sum[:, camera_index, step_index] += result["weight_sum"]
        count[:, camera_index, step_index] += result["observation_count"]
    mean = feature_sum / weight_sum[..., None].clamp_min(1.0e-8)
    return {
        "feature_mean": mean.detach().cpu(),
        "feature_sum": feature_sum.detach().cpu(),
        "weight_sum": weight_sum.detach().cpu(),
        "observation_count": count.detach().cpu(),
        "channel_names": EVIDENCE_CHANNELS[:channels],
    }


@torch.no_grad()
def accumulate_depth_pixel_classes_to_gaussians(
    gaussians,
    pipe,
    records,
    *,
    num_cameras: int,
) -> dict[str, torch.Tensor]:
    """Transfer one depth-only canonical pixel classification per camera."""
    from gaussian_renderer import accumulate_pixel_features_to_gaussians

    point_count = int(gaussians.get_xyz.shape[0])
    channel_count = len(DEPTH_EVIDENCE_CHANNELS)
    device = gaussians.get_xyz.device
    feature_sum = torch.zeros(
        (point_count, int(num_cameras), channel_count), device=device
    )
    raster_weight_sum = torch.zeros(
        (point_count, int(num_cameras)), device=device
    )
    observation_count = torch.zeros(
        (point_count, int(num_cameras)), device=device, dtype=torch.long
    )
    for record in records:
        camera_index = int(record["camera_index"])
        result = accumulate_pixel_features_to_gaussians(
            record["camera"],
            gaussians,
            pipe,
            record["pixel_features"],
            record["pixel_confidence"],
        )
        feature_sum[:, camera_index] = result["feature_sum"]
        raster_weight_sum[:, camera_index] = result["weight_sum"]
        observation_count[:, camera_index] = result["observation_count"]
    return {
        "schema_version": 1,
        "channel_names": DEPTH_EVIDENCE_CHANNELS,
        "feature_sum": feature_sum.cpu(),
        "raster_weight_sum": raster_weight_sum.cpu(),
        "observation_count": observation_count.cpu(),
    }


def compute_gaussian_camera_votes(
    evidence_state: dict,
    *,
    min_camera_weight: float,
    static_threshold: float,
    dynamic_threshold: float,
) -> dict[str, torch.Tensor]:
    """Apply explicit per-camera votes without collapsing camera identity."""
    feature_sum = torch.as_tensor(evidence_state["feature_sum"]).float()
    names = list(evidence_state.get("channel_names", DEPTH_EVIDENCE_CHANNELS))

    def channel(name):
        return feature_sum[..., names.index(name)]

    observable_weight = channel("observable")
    raster_weight = torch.as_tensor(evidence_state["raster_weight_sum"]).float()
    visible = observable_weight >= float(min_camera_weight)
    static_score = channel("static") / observable_weight.clamp_min(1.0e-8)
    dynamic_score = channel("dynamic") / observable_weight.clamp_min(1.0e-8)
    static_vote = visible & (static_score >= float(static_threshold))
    dynamic_vote = visible & (dynamic_score >= float(dynamic_threshold))
    # Conservative thresholds should already be exclusive; ties remain unknown.
    conflict = static_vote & dynamic_vote
    static_vote &= ~conflict
    dynamic_vote &= ~conflict
    visible_count = visible.sum(dim=1)
    static_count = static_vote.sum(dim=1)
    dynamic_count = dynamic_vote.sum(dim=1)
    denominator = visible_count.clamp_min(1).float()
    return {
        "visible_camera_mask": visible,
        "static_camera_vote": static_vote,
        "dynamic_camera_vote": dynamic_vote,
        "visible_camera_count": visible_count,
        "static_camera_count": static_count,
        "dynamic_camera_count": dynamic_count,
        "ambiguous_camera_count": (visible & ~static_vote & ~dynamic_vote).sum(dim=1),
        "static_camera_fraction": static_count.float() / denominator,
        "dynamic_camera_fraction": dynamic_count.float() / denominator,
        "static_camera_score": static_score,
        "dynamic_camera_score": dynamic_score,
        "observable_weight": observable_weight,
        "mean_present_ratio": channel("present_ratio").sum(dim=1)
        / raster_weight.sum(dim=1).clamp_min(1.0e-8),
        "mean_vacated_ratio": channel("vacated_ratio").sum(dim=1)
        / raster_weight.sum(dim=1).clamp_min(1.0e-8),
        "mean_occlusion_fraction": channel("occlusion_fraction").sum(dim=1)
        / raster_weight.sum(dim=1).clamp_min(1.0e-8),
    }


def build_depth_only_initial_cores(
    votes: dict,
    *,
    min_static_cameras: int,
    min_dynamic_cameras: int,
    min_static_camera_fraction: float,
    min_dynamic_camera_fraction: float,
    max_conflicting_cameras: int,
) -> dict[str, torch.Tensor]:
    """Create only trusted STATIC/DYNAMIC cores; everything else is UNKNOWN."""
    static_count = torch.as_tensor(votes["static_camera_count"])
    dynamic_count = torch.as_tensor(votes["dynamic_camera_count"])
    static_core = (
        (static_count >= int(min_static_cameras))
        & (
            torch.as_tensor(votes["static_camera_fraction"])
            >= float(min_static_camera_fraction)
        )
        & (dynamic_count <= int(max_conflicting_cameras))
    )
    dynamic_core = (
        (dynamic_count >= int(min_dynamic_cameras))
        & (
            torch.as_tensor(votes["dynamic_camera_fraction"])
            >= float(min_dynamic_camera_fraction)
        )
        & (static_count <= int(max_conflicting_cameras))
    )
    conflict = static_core & dynamic_core
    static_core &= ~conflict
    dynamic_core &= ~conflict
    state = torch.full(static_count.shape, UNKNOWN, dtype=torch.uint8)
    state[static_core] = STATIC_CORE
    state[dynamic_core] = DYNAMIC_CORE
    probability = torch.full(static_count.shape, 0.5, dtype=torch.float32)
    probability[static_core] = 0.01
    probability[dynamic_core] = 0.99
    return {
        "state": state,
        "static_core_mask": static_core,
        "dynamic_core_mask": dynamic_core,
        "unknown_mask": ~(static_core | dynamic_core),
        "dynamic_probability": probability,
        "dynamic_score": torch.as_tensor(votes["dynamic_camera_fraction"]).float(),
        "static_score": torch.as_tensor(votes["static_camera_fraction"]).float(),
        **votes,
    }


def _channel(evidence: torch.Tensor, names: tuple[str, ...] | list[str], name: str) -> torch.Tensor:
    try:
        return evidence[..., list(names).index(name)]
    except ValueError:
        return torch.zeros(evidence.shape[:-1], dtype=evidence.dtype, device=evidence.device)


def compute_gaussian_temporal_persistence(
    evidence: torch.Tensor,
    channel_names=EVIDENCE_CHANNELS,
) -> dict[str, torch.Tensor]:
    """Preserve interval decisions, then aggregate over cameras (never last-frame wins)."""
    value = torch.as_tensor(evidence).float()
    if value.ndim != 4:
        raise ValueError("evidence must be [N,cameras,time,channels]")
    dynamic = torch.maximum(
        _channel(value, channel_names, "dynamic_flow"),
        _channel(value, channel_names, "vacated"),
    )
    static = torch.minimum(
        _channel(value, channel_names, "present"),
        _channel(value, channel_names, "static_flow"),
    )
    informative = _channel(value, channel_names, "informative")
    valid = (
        (informative > 0.05)
        | (_channel(value, channel_names, "vacated") > 0.50)
        | (_channel(value, channel_names, "present") > 0.50)
    )
    dynamic_vote = valid & (dynamic > static + 0.15) & (dynamic > 0.35)
    static_vote = valid & (static > dynamic + 0.15) & (static > 0.35)
    flow_valid = informative > 0.05
    # dynamic_flow is already the calibrated r_static>=4 decision, multiplied
    # by identity/flow/reliability confidence. Require meaningful ownership.
    flow_dynamic_vote = flow_valid & (
        _channel(value, channel_names, "dynamic_flow") >= 0.20
    )
    valid_time = valid.any(dim=1)
    dyn_time = dynamic_vote.any(dim=1)
    sta_time = static_vote.any(dim=1)
    denominator = valid_time.sum(dim=-1).clamp_min(1)
    flow_valid_time = flow_valid.any(dim=1)
    flow_denominator = flow_valid_time.sum(dim=-1).clamp_min(1)
    return {
        "dynamic_time_fraction": dyn_time.sum(dim=-1).float() / denominator,
        "static_time_fraction": sta_time.sum(dim=-1).float() / denominator,
        "informative_time_count": valid_time.sum(dim=-1),
        "flow_dynamic_time_fraction": (
            flow_dynamic_vote.any(dim=1).sum(dim=-1).float() / flow_denominator
        ),
        "dynamic_vote": dynamic_vote,
        "static_vote": static_vote,
        "flow_valid": flow_valid,
        "flow_dynamic_vote": flow_dynamic_vote,
    }


def compute_gaussian_camera_persistence(
    evidence: torch.Tensor,
    channel_names=EVIDENCE_CHANNELS,
) -> dict[str, torch.Tensor]:
    value = torch.as_tensor(evidence).float()
    temporal = compute_gaussian_temporal_persistence(value, channel_names)
    dynamic_vote = temporal["dynamic_vote"]
    static_vote = temporal["static_vote"]
    visible = (
        (_channel(value, channel_names, "informative") > 0.05)
        | (_channel(value, channel_names, "vacated") > 0.50)
        | (_channel(value, channel_names, "present") > 0.50)
    ).any(dim=-1)
    dyn_camera = dynamic_vote.float().mean(dim=-1) >= 0.5
    sta_camera = static_vote.float().mean(dim=-1) >= 0.5
    denominator = visible.sum(dim=-1).clamp_min(1)
    flow_valid = temporal["flow_valid"]
    flow_dynamic_vote = temporal["flow_dynamic_vote"]
    flow_visible = flow_valid.any(dim=-1)
    flow_dynamic_camera = (
        flow_dynamic_vote.sum(dim=-1).float()
        / flow_valid.sum(dim=-1).clamp_min(1)
    ) >= 0.10
    flow_camera_denominator = flow_visible.sum(dim=-1).clamp_min(1)
    return {
        "dynamic_camera_fraction": (dyn_camera & visible).sum(dim=-1).float() / denominator,
        "static_camera_fraction": (sta_camera & visible).sum(dim=-1).float() / denominator,
        "visible_camera_count": visible.sum(dim=-1),
        "dynamic_camera_count": (dyn_camera & visible).sum(dim=-1),
        "static_camera_count": (sta_camera & visible).sum(dim=-1),
        "flow_dynamic_camera_fraction": (
            (flow_dynamic_camera & flow_visible).sum(dim=-1).float()
            / flow_camera_denominator
        ),
    }


def build_initial_cores(
    evidence_state: dict,
    thresholds: CoreThresholds | None = None,
) -> dict[str, torch.Tensor]:
    """Build conservative static/dynamic cores and leave weak evidence UNKNOWN."""
    cfg = thresholds or CoreThresholds()
    evidence = torch.as_tensor(evidence_state["feature_mean"]).float()
    names = tuple(evidence_state.get("channel_names", EVIDENCE_CHANNELS))
    observations = torch.as_tensor(evidence_state["observation_count"]).sum(dim=(1, 2))
    temporal = compute_gaussian_temporal_persistence(evidence, names)
    cameras = compute_gaussian_camera_persistence(evidence, names)
    slot_valid = (torch.as_tensor(evidence_state["weight_sum"]) > 0).float()
    slot_denominator = slot_valid.sum(dim=(1, 2)).clamp_min(1.0)
    def sequence_mean(name):
        return (_channel(evidence, names, name) * slot_valid).sum(dim=(1, 2)) / slot_denominator
    present = sequence_mean("present")
    vacated = sequence_mean("vacated")
    static_flow = sequence_mean("static_flow")
    dynamic_flow = sequence_mean("dynamic_flow")
    dynamic_amplitude = (0.50 * dynamic_flow + 0.50 * vacated).clamp(0.0, 1.0)
    static_score = (0.55 * present + 0.45 * static_flow - 0.50 * vacated).clamp(0.0, 1.0)
    # The configurable score threshold is a model-selection probability, not
    # an unnormalized evidence amplitude. Comparing 0.72 to the latter made a
    # non-empty, persistent sequence produce an empty core.
    dynamic_score = dynamic_amplitude / (dynamic_amplitude + static_score).clamp_min(1.0e-6)
    enough = observations >= int(cfg.min_observations)
    dynamic_core = (
        enough
        & (dynamic_score >= float(cfg.dynamic_score))
        & (temporal["flow_dynamic_time_fraction"] >= float(cfg.dynamic_time_fraction))
        & (cameras["flow_dynamic_camera_fraction"] >= float(cfg.dynamic_camera_fraction))
    )
    static_core = (
        enough
        & ~dynamic_core
        & (static_score >= float(cfg.static_score))
        & (temporal["static_time_fraction"] >= float(cfg.dynamic_time_fraction))
        & (cameras["static_camera_fraction"] >= float(cfg.dynamic_camera_fraction))
    )
    state = torch.full_like(observations, UNKNOWN, dtype=torch.uint8)
    state[static_core] = STATIC_CORE
    state[dynamic_core] = DYNAMIC_CORE
    probability = dynamic_score.clamp(0.01, 0.99)
    return {
        "state": state,
        "static_core_mask": static_core,
        "dynamic_core_mask": dynamic_core,
        "unknown_mask": ~(static_core | dynamic_core),
        "dynamic_probability": probability,
        "dynamic_score": dynamic_score,
        "dynamic_evidence_amplitude": dynamic_amplitude,
        "static_score": static_score,
        "observation_count": observations,
        **{
            key: value
            for key, value in temporal.items()
            if not key.endswith("vote") and key != "flow_valid"
        },
        **cameras,
    }


def hypothesis_visibility(
    expected_depth: torch.Tensor,
    observed_depth: torch.Tensor,
    valid_projection: torch.Tensor,
    *,
    relative_sigma: float = 0.02,
    sigma_floor: float = 1.0e-3,
    margin_sigma: float = 2.5,
) -> dict[str, torch.Tensor]:
    expected = torch.as_tensor(expected_depth).float()
    observed = torch.as_tensor(observed_depth).float()
    sigma = (expected.abs() * float(relative_sigma)).clamp_min(float(sigma_floor))
    valid = torch.as_tensor(valid_projection).bool() & (observed > 0) & torch.isfinite(observed)
    delta = observed - expected
    visible = valid & (delta.abs() <= float(margin_sigma) * sigma)
    occluded = valid & (delta < -float(margin_sigma) * sigma)
    vacated = valid & (delta > float(margin_sigma) * sigma)
    return {
        "visible": visible,
        "occluded": occluded,
        "vacated": vacated,
        "valid": valid,
        "normalized_depth_residual": delta.abs() / sigma.clamp_min(1.0e-8),
    }


def infer_observation_speed_scalar(
    unit_projected_flow: torch.Tensor,
    corrected_observed_flow: torch.Tensor,
    expected_speed: torch.Tensor,
    valid_mask: torch.Tensor | None = None,
) -> dict[str, torch.Tensor]:
    """Compute per-Gaussian scalar-speed compatibility over camera/time observations."""
    g = torch.as_tensor(unit_projected_flow).float()
    flow = torch.as_tensor(corrected_observed_flow).float()
    if g.shape != flow.shape or g.shape[-1] != 2:
        raise ValueError("projected and observed flow must have equal [...,2] shape")
    speed = torch.as_tensor(expected_speed, device=g.device, dtype=g.dtype)
    while speed.ndim < g.ndim - 1:
        speed = speed.unsqueeze(0)
    norm2 = g.square().sum(dim=-1)
    inferred = (g * flow).sum(dim=-1) / norm2.clamp_min(1.0e-8)
    predicted_parallel = inferred[..., None] * g
    perpendicular = torch.linalg.norm(flow - predicted_parallel, dim=-1)
    valid = norm2 > 1.0e-4
    if valid_mask is not None:
        valid &= torch.as_tensor(valid_mask).bool()
    expected = speed.expand_as(inferred)
    relative_error = (inferred - expected).abs() / expected.abs().clamp_min(1.0e-4)
    sign = (inferred * expected > 0).float()
    parallel_residual = (inferred - expected).abs() * torch.sqrt(norm2.clamp_min(1.0e-8))
    weights = valid.float()
    denominator = weights.sum(dim=tuple(range(1, weights.ndim))).clamp_min(1.0)
    reduce_dims = tuple(range(1, weights.ndim))
    expected_mean = (expected * weights).sum(dim=reduce_dims) / denominator
    inferred_mean = (inferred * weights).sum(dim=reduce_dims) / denominator
    expected_centered = expected - expected_mean.reshape((-1,) + (1,) * (weights.ndim - 1))
    inferred_centered = inferred - inferred_mean.reshape((-1,) + (1,) * (weights.ndim - 1))
    covariance = (expected_centered * inferred_centered * weights).sum(dim=reduce_dims)
    variance_product = (
        (expected_centered.square() * weights).sum(dim=reduce_dims)
        * (inferred_centered.square() * weights).sum(dim=reduce_dims)
    )
    correlation = covariance / torch.sqrt(variance_product.clamp_min(1.0e-8))
    constant_score = (1.0 - relative_error.clamp_max(1.0)) * sign
    constant_score = (constant_score * weights).sum(dim=reduce_dims) / denominator
    correlation = torch.where(variance_product > 1.0e-8, correlation, constant_score)
    return {
        "inferred_speed": inferred,
        "speed_relative_error": (relative_error * weights).sum(dim=reduce_dims) / denominator,
        "speed_sign_consistency": (sign * weights).sum(dim=reduce_dims) / denominator,
        "speed_perpendicular_residual": (perpendicular * weights).sum(dim=reduce_dims) / denominator,
        "speed_parallel_residual": (parallel_residual * weights).sum(dim=reduce_dims) / denominator,
        "speed_correlation": correlation,
        "valid": valid,
    }


def compute_rigid_hypothesis_unaries(
    static_error: torch.Tensor,
    rigid_error: torch.Tensor,
    *,
    static_visibility: torch.Tensor,
    rigid_visibility: torch.Tensor,
    weights: torch.Tensor | None = None,
    excitation: torch.Tensor | None = None,
    margin: float = 0.25,
) -> dict[str, torch.Tensor]:
    """Aggregate full camera/time physical-hypothesis likelihood ratios."""
    static = torch.as_tensor(static_error).float()
    rigid = torch.as_tensor(rigid_error).float()
    if static.shape != rigid.shape or static.ndim != 3:
        raise ValueError("errors must have equal [N,cameras,time] shape")
    valid = torch.as_tensor(static_visibility).bool() | torch.as_tensor(rigid_visibility).bool()
    weight = torch.ones_like(static) if weights is None else torch.as_tensor(weights).float()
    if excitation is not None:
        weight = weight * torch.as_tensor(excitation).float()
    weight = weight * valid.float()
    likelihood_ratio = static - rigid
    interval_weight = weight.sum(dim=1)
    interval_margin = (likelihood_ratio * weight).sum(dim=1) / interval_weight.clamp_min(1.0e-8)
    total_weight = interval_weight.sum(dim=-1)
    aggregate_margin = (interval_margin * interval_weight).sum(dim=-1) / total_weight.clamp_min(1.0e-8)
    informative = interval_weight > 0
    positive = informative & (interval_margin > float(margin))
    negative = informative & (interval_margin < -float(margin))
    time_den = informative.sum(dim=-1).clamp_min(1)
    camera_weight = weight.sum(dim=-1)
    camera_margin = (likelihood_ratio * weight).sum(dim=-1) / camera_weight.clamp_min(1.0e-8)
    visible_camera = camera_weight > 0
    camera_den = visible_camera.sum(dim=-1).clamp_min(1)
    run = torch.zeros(static.shape[0], device=static.device, dtype=torch.long)
    longest_run = torch.zeros_like(run)
    for interval in range(positive.shape[-1]):
        run = torch.where(positive[:, interval], run + 1, torch.zeros_like(run))
        longest_run = torch.maximum(longest_run, run)
    masked_interval_margin = torch.where(
        informative, interval_margin, torch.full_like(interval_margin, float("nan"))
    )
    split = max(1, interval_margin.shape[-1] // 2)
    early_info, late_info = informative[:, :split], informative[:, split:]
    early_positive, late_positive = positive[:, :split], positive[:, split:]
    return {
        "static_energy": (static * weight).sum(dim=(1, 2)),
        "rigid_energy": (rigid * weight).sum(dim=(1, 2)),
        "margin": aggregate_margin,
        "interval_margin": interval_margin,
        "camera_margin": camera_margin,
        "time_persistence": positive.sum(dim=-1).float() / time_den,
        "static_time_persistence": negative.sum(dim=-1).float() / time_den,
        "camera_persistence": ((camera_margin > margin) & visible_camera).sum(dim=-1).float() / camera_den,
        "static_camera_persistence": ((camera_margin < -margin) & visible_camera).sum(dim=-1).float() / camera_den,
        "camera_support_count": visible_camera.sum(dim=-1),
        "camera_median_likelihood_ratio": torch.nanmedian(
            torch.where(
                visible_camera,
                camera_margin,
                torch.full_like(camera_margin, float("nan")),
            ),
            dim=-1,
        ).values,
        "informative_interval_count": informative.sum(dim=-1),
        "median_likelihood_ratio": torch.nanmedian(masked_interval_margin, dim=-1).values,
        "likelihood_ratio_q25": torch.nanquantile(masked_interval_margin, 0.25, dim=-1),
        "likelihood_ratio_q75": torch.nanquantile(masked_interval_margin, 0.75, dim=-1),
        "longest_positive_run": longest_run,
        "early_dynamic_fraction": early_positive.sum(dim=-1).float() / early_info.sum(dim=-1).clamp_min(1),
        "late_dynamic_fraction": (
            late_positive.sum(dim=-1).float() / late_info.sum(dim=-1).clamp_min(1)
            if late_info.shape[-1]
            else torch.zeros(static.shape[0], device=static.device)
        ),
    }


def classify_rigid_hypotheses_once(
    hypothesis: dict,
    speed_compatible: torch.Tensor,
    *,
    min_margin: float,
    min_time_fraction: float,
    min_camera_fraction: float,
) -> dict[str, torch.Tensor | int]:
    """Create STATIC/DYNAMIC/UNKNOWN labels once after rigid discovery."""
    margin = torch.as_tensor(hypothesis["margin"]).float()
    informative = torch.as_tensor(hypothesis["informative_interval_count"]) > 0
    dynamic = (
        informative
        & (margin >= float(min_margin))
        & (torch.as_tensor(hypothesis["time_persistence"]) >= float(min_time_fraction))
        & (torch.as_tensor(hypothesis["camera_persistence"]) >= float(min_camera_fraction))
        & torch.as_tensor(speed_compatible).bool()
    )
    static = (
        informative
        & (margin <= -float(min_margin))
        & (
            torch.as_tensor(hypothesis["static_time_persistence"])
            >= float(min_time_fraction)
        )
        & (
            torch.as_tensor(hypothesis["static_camera_persistence"])
            >= float(min_camera_fraction)
        )
    )
    state = torch.full(margin.shape, UNKNOWN, dtype=torch.uint8, device=margin.device)
    state[static] = STATIC_CORE
    state[dynamic] = DYNAMIC_CORE
    probability = torch.sigmoid(
        margin / max(float(min_margin), 1.0e-4)
    )
    probability = torch.where(
        informative, probability, torch.full_like(probability, 0.5)
    )
    return {
        "state": state,
        "dynamic_probability": probability,
        "effective_dynamic_mask": dynamic,
        "fit_core_mask": dynamic,
        "static_mask": static,
        "unknown_mask": ~(static | dynamic),
        "static_count": int(static.sum()),
        "dynamic_count": int(dynamic.sum()),
        "unknown_count": int((~(static | dynamic)).sum()),
    }


def update_candidate_states(
    previous_state: torch.Tensor,
    hypothesis: dict,
    *,
    min_margin: float = 0.35,
    min_time_fraction: float = 0.60,
    min_camera_fraction: float = 0.40,
    demotion_margin: float = 0.70,
) -> dict[str, torch.Tensor]:
    previous = torch.as_tensor(previous_state).to(torch.uint8)
    margin = torch.as_tensor(hypothesis["margin"])
    dynamic = (
        (margin >= float(min_margin))
        & (hypothesis["time_persistence"] >= float(min_time_fraction))
        & (hypothesis["camera_persistence"] >= float(min_camera_fraction))
    )
    contradictory = margin <= -float(demotion_margin)
    state = previous.clone()
    state[(previous == UNKNOWN) & dynamic] = DYNAMIC_CANDIDATE
    state[(previous == STATIC_CORE) & dynamic] = DYNAMIC_CANDIDATE
    state[(previous == DYNAMIC_CANDIDATE) & ~dynamic] = UNKNOWN
    state[(previous == DYNAMIC_CORE) & contradictory] = DYNAMIC_CANDIDATE
    return {"state": state, "label_changes": int((state != previous).sum())}


def promote_dynamic_core(
    state: torch.Tensor,
    consecutive_support_cycles: torch.Tensor,
    hypothesis: dict,
    *,
    min_cycles: int = 2,
    min_margin: float = 0.60,
) -> dict[str, torch.Tensor]:
    value = torch.as_tensor(state).to(torch.uint8).clone()
    cycles = torch.as_tensor(consecutive_support_cycles).long()
    promote = (
        (value == DYNAMIC_CANDIDATE)
        & (cycles >= int(min_cycles))
        & (torch.as_tensor(hypothesis["margin"]) >= float(min_margin))
    )
    value[promote] = DYNAMIC_CORE
    return {"state": value, "promoted": promote, "promotion_count": int(promote.sum())}


@torch.no_grad()
def refine_assignment_with_rigid_hypotheses(
    canonical_points: torch.Tensor,
    mode,
    interval_speeds: torch.Tensor,
    frame_times,
    cameras_by_frame: list[dict],
    correspondence_cache,
    calibrations: dict,
    previous_state: torch.Tensor,
    consecutive_support_cycles: torch.Tensor,
    args,
    *,
    one_shot: bool = False,
) -> dict[str, torch.Tensor | int]:
    """Evaluate every Gaussian under static and exact-rigid hypotheses over the video."""
    from utils.motion_correspondence import _sample_map, project_world_points
    from utils.rigid_killing_motion import compute_unit_speed_projected_flow
    from utils.temporal_visibility_pretraining import inverse_depth_to_metric

    points = torch.as_tensor(canonical_points)
    device, dtype = points.device, points.dtype
    trajectory = compute_unit_speed_projected_flow(
        points, mode, interval_speeds, frame_times, cameras_by_frame
    )
    keys = trajectory["camera_keys"]
    n_points = points.shape[0]
    n_intervals = len(cameras_by_frame) - 1
    static_error = torch.zeros((n_points, len(keys), n_intervals), device=device)
    rigid_error = torch.zeros_like(static_error)
    comparison_weight = torch.zeros_like(static_error)
    static_visibility = torch.zeros_like(static_error, dtype=torch.bool)
    rigid_visibility = torch.zeros_like(static_visibility)
    observed_dynamic_flow = torch.zeros((n_points, len(keys), n_intervals, 2), device=device)
    speed_valid = torch.zeros_like(static_visibility)

    for interval in range(n_intervals):
        static_source = points
        static_target = points
        rigid_source = trajectory["positions"][interval]
        rigid_target = trajectory["positions"][interval + 1]
        for camera_index, key in enumerate(keys):
            source_camera = cameras_by_frame[interval][key]
            target_camera = cameras_by_frame[interval + 1][key]
            observation = correspondence_cache.get(source_camera, target_camera, device)
            height, width = int(source_camera.image_height), int(source_camera.image_width)
            flow_map = observation.flow
            confidence_map = observation.confidence
            if tuple(flow_map.shape[-2:]) != (height, width):
                flow_map = torch.nn.functional.interpolate(
                    flow_map[None], size=(height, width), mode="bilinear", align_corners=False
                )[0]
                confidence_map = torch.nn.functional.interpolate(
                    confidence_map[None], size=(height, width), mode="bilinear", align_corners=False
                )[0]

            def evaluate_positions(source_xyz, target_xyz):
                source_pixels, source_z, valid_source_projection = project_world_points(source_xyz, source_camera)
                target_pixels, target_z, valid_target_projection = project_world_points(target_xyz, target_camera)
                source_inv = source_camera.invdepthmap.to(device=device, dtype=dtype)
                target_inv = target_camera.invdepthmap.to(device=device, dtype=dtype)
                if tuple(source_inv.shape[-2:]) != (height, width):
                    source_inv = torch.nn.functional.interpolate(source_inv[None], size=(height, width), mode="nearest")[0]
                if tuple(target_inv.shape[-2:]) != (height, width):
                    target_inv = torch.nn.functional.interpolate(target_inv[None], size=(height, width), mode="nearest")[0]
                source_depth_map = inverse_depth_to_metric(source_inv)[0]
                target_depth_map = inverse_depth_to_metric(target_inv)[0]
                observed_source_z = _sample_map(source_depth_map[None], source_pixels, height, width)[:, 0]
                observed_target_z = _sample_map(target_depth_map[None], target_pixels, height, width)[:, 0]
                source_vis = hypothesis_visibility(
                    source_z, observed_source_z, valid_source_projection,
                    relative_sigma=float(args.visibility_depth_sigma),
                    margin_sigma=float(args.visibility_occlusion_margin),
                )
                target_vis = hypothesis_visibility(
                    target_z, observed_target_z, valid_target_projection,
                    relative_sigma=float(args.visibility_depth_sigma),
                    margin_sigma=float(args.visibility_occlusion_margin),
                )
                visible = source_vis["visible"] & target_vis["visible"]
                vacated = source_vis["vacated"] | target_vis["vacated"]
                occluded = source_vis["occluded"] | target_vis["occluded"]
                depth_error = 0.5 * (
                    source_vis["normalized_depth_residual"]
                    + target_vis["normalized_depth_residual"]
                )
                return source_pixels, target_pixels, visible, vacated, occluded, depth_error

            s_pixel, s_target_pixel, s_visible, s_vacated, s_occluded, s_depth_error = evaluate_positions(
                static_source, static_target
            )
            d_pixel, d_target_pixel, d_visible, d_vacated, d_occluded, d_depth_error = evaluate_positions(
                rigid_source, rigid_target
            )
            static_visibility[:, camera_index, interval] = s_visible
            rigid_visibility[:, camera_index, interval] = d_visible
            calibration = calibrations.get((key, interval), {})
            median = torch.as_tensor(calibration.get("median", (0.0, 0.0)), device=device, dtype=dtype)
            mad = torch.as_tensor(calibration.get("mad", (1.0, 1.0)), device=device, dtype=dtype).clamp_min(
                float(args.static_flow_mad_floor)
            )
            reliability = float(calibration.get("camera_reliability", 1.0))
            static_observed = _sample_map(flow_map, s_pixel, height, width) - median
            dynamic_observed = _sample_map(flow_map, d_pixel, height, width) - median
            static_confidence = _sample_map(confidence_map, s_pixel, height, width)[:, 0]
            dynamic_confidence = _sample_map(confidence_map, d_pixel, height, width)[:, 0]
            rigid_projected = d_target_pixel - d_pixel
            s_flow_error = torch.linalg.norm(static_observed / mad, dim=-1)
            d_flow_error = torch.linalg.norm((dynamic_observed - rigid_projected) / mad, dim=-1)
            # A vacated expected surface is a visibility mismatch, not an occlusion vote.
            s_error = torch.where(s_visible, s_flow_error + 0.25 * s_depth_error, torch.where(s_vacated, torch.full_like(s_flow_error, 4.0), torch.zeros_like(s_flow_error)))
            d_error = torch.where(d_visible, d_flow_error + 0.25 * d_depth_error, torch.where(d_vacated, torch.full_like(d_flow_error, 4.0), torch.zeros_like(d_flow_error)))
            comparable = ~(s_occluded | d_occluded) & ((s_visible | s_vacated) & (d_visible | d_vacated))
            confidence = torch.minimum(static_confidence, dynamic_confidence).clamp(0, 1)
            static_error[:, camera_index, interval] = s_error
            rigid_error[:, camera_index, interval] = d_error
            comparison_weight[:, camera_index, interval] = comparable.float() * confidence * reliability
            observed_dynamic_flow[:, camera_index, interval] = dynamic_observed
            speed_valid[:, camera_index, interval] = d_visible & (dynamic_confidence > 0.05)

    excitation = (torch.linalg.norm(trajectory["flow"], dim=-1) / 1.0).clamp(0.0, 1.0)
    excitation = torch.where(excitation >= 0.10, excitation, torch.zeros_like(excitation))
    hypothesis = compute_rigid_hypothesis_unaries(
        static_error, rigid_error,
        static_visibility=static_visibility,
        rigid_visibility=rigid_visibility,
        weights=comparison_weight,
        excitation=excitation,
        margin=float(args.rigid_assignment_min_margin),
    )
    speed = infer_observation_speed_scalar(
        trajectory["unit_flow"], observed_dynamic_flow,
        torch.as_tensor(interval_speeds, device=device, dtype=dtype),
        speed_valid & (comparison_weight > 0),
    )
    speed_compatible = (
        (speed["speed_correlation"] >= float(args.rigid_assignment_min_speed_correlation))
        | ((speed["speed_relative_error"] <= 0.35) & (speed["speed_sign_consistency"] >= 0.75))
    )
    if one_shot:
        assigned = classify_rigid_hypotheses_once(
            hypothesis,
            speed_compatible,
            min_margin=float(args.rigid_assignment_min_margin),
            min_time_fraction=float(args.rigid_assignment_min_time_fraction),
            min_camera_fraction=float(args.rigid_assignment_min_camera_fraction),
        )
        previous = torch.as_tensor(previous_state, device=device).to(torch.uint8)
        return {
            **hypothesis,
            **speed,
            **assigned,
            "consecutive_support_cycles": torch.zeros(
                n_points, device=device, dtype=torch.long
            ),
            "label_changes": int((assigned["state"] != previous).sum()),
            "core_promotions": 0,
            "core_demotions": 0,
            "one_shot_assignment": True,
            "static_error": static_error.cpu(),
            "rigid_error": rigid_error.cpu(),
            "weight": comparison_weight.cpu(),
            "excitation": excitation.cpu(),
        }
    gated_hypothesis = dict(hypothesis)
    gated_hypothesis["time_persistence"] = torch.where(
        speed_compatible, hypothesis["time_persistence"], torch.zeros_like(hypothesis["time_persistence"])
    )
    updated = update_candidate_states(
        previous_state, gated_hypothesis,
        min_margin=float(args.rigid_assignment_min_margin),
        min_time_fraction=float(args.rigid_assignment_min_time_fraction),
        min_camera_fraction=float(args.rigid_assignment_min_camera_fraction),
        demotion_margin=float(args.core_demotion_margin),
    )
    candidate_support = updated["state"] == DYNAMIC_CANDIDATE
    cycles = torch.where(
        candidate_support,
        torch.as_tensor(consecutive_support_cycles, device=device).long() + 1,
        torch.zeros(n_points, device=device, dtype=torch.long),
    )
    promoted = promote_dynamic_core(
        updated["state"], cycles, gated_hypothesis,
        min_cycles=int(args.core_promotion_min_cycles),
        min_margin=float(args.core_promotion_min_margin),
    )
    state = promoted["state"]
    probability = torch.sigmoid(hypothesis["margin"] / max(float(args.rigid_assignment_min_margin), 1.0e-4))
    return {
        **hypothesis,
        **speed,
        "state": state,
        "dynamic_probability": probability,
        "effective_dynamic_mask": (state == DYNAMIC_CORE) | (state == DYNAMIC_CANDIDATE),
        "fit_core_mask": state == DYNAMIC_CORE,
        "consecutive_support_cycles": cycles,
        "label_changes": updated["label_changes"],
        "core_promotions": promoted["promotion_count"],
        "core_demotions": int(((torch.as_tensor(previous_state) == DYNAMIC_CORE) & (state != DYNAMIC_CORE)).sum()),
        "static_error": static_error.cpu(),
        "rigid_error": rigid_error.cpu(),
        "weight": comparison_weight.cpu(),
        "excitation": excitation.cpu(),
    }
