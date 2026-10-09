"""Velocity-field-centric Stage-1 discovery of static plus K rigid parts."""

from __future__ import annotations

import json
import math
from pathlib import Path

import torch
from tqdm import tqdm

from utils.eulerian_contrastive_support import assign_gaussians_to_cells
from utils.kpart_counterfactual_rendering import (
    compute_local_affected_region,
    compute_rgb_depth_dino_flow_energy,
    extract_uncertain_components,
    rank_components_for_rendering,
    render_full_scene_with_candidate_label,
    select_informative_frame_pairs,
    update_persistent_render_evidence,
)
from utils.kpart_dino_boundary import (
    build_canonical_dino_graph,
    compute_3d_same_material_affinity,
    compute_image_dino_boundary,
    compute_one_sided_dino_seam_penalty,
    compute_rendered_part_boundary,
    render_soft_part_probability_maps,
)
from utils.kpart_sam3_boundary import (
    build_canonical_sam3_graph,
    compute_3d_sam3_same_material_affinity,
)
from utils.kpart_boundary_graph import build_canonical_gaussian_knn
from utils.kpart_field_merge import fit_region_rigid_field
from utils.kpart_graph_segmentation import normalize_unary_families
from utils.kpart_initial_oversegmentation import (
    compute_static_render_failure,
    rasterize_failure_to_gaussians,
)
from utils.kpart_soft_responsibility import (
    compute_boundary_ambiguity_mask,
    compute_velocity_field_unaries,
    initialize_responsibilities_from_unaries,
    optimize_responsibility_logits,
)
from utils.motion_correspondence import MotionCorrespondenceCache, project_world_points
from utils.rigid_killing_motion import integrate_rigid_mode_exact
from utils.rigid_endpoint_transport import (
    EndpointTransportConfig,
    RigidEndpointTransport,
    integrate_endpoint_hypotheses,
    write_endpoint_target_summary,
    write_endpoint_transport_diagnostics,
)
from utils.rigid_transform_utils import write_quaternion_transport_audit
from utils.temporal_visibility_pretraining import _atomic_torch_save


def _strict_json(value):
    if isinstance(value, dict):
        return {str(key): _strict_json(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_strict_json(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _coordinates(speed, frame_times):
    speed = torch.as_tensor(speed).detach().cpu().double()
    time = torch.as_tensor(frame_times).detach().cpu().double()
    return torch.cat((torch.zeros(1, dtype=torch.float64), torch.cumsum(speed * torch.diff(time), 0)))


def _render_denoise_state(gaussians):
    state = getattr(gaussians, "render_discovery_state", None)
    return state if isinstance(state, dict) else None


def _render_denoise_gaussian_field(state, name, count, *, default=0.0):
    aliases = {
        "dynamic_score": ("dynamic_score", "per_gaussian_dynamic_score"),
        "static_score": ("static_score", "per_gaussian_static_score"),
        "unknown_score": ("unknown_score", "per_gaussian_unknown_score"),
        "confidence": ("confidence", "per_gaussian_confidence"),
        "final_dynamic_probability": (
            "final_dynamic_probability",
            "per_gaussian_final_dynamic_probability",
        ),
    }
    for key in aliases.get(name, (name,)):
        if key in state:
            value = torch.as_tensor(state[key]).detach().cpu().float().reshape(-1)
            if value.numel() == int(count):
                return value
    return torch.full((int(count),), float(default), dtype=torch.float32)


def _render_denoise_bootstrap(state, count, confidence_threshold):
    probability = _render_denoise_gaussian_field(
        state,
        "final_dynamic_probability",
        count,
        default=0.5,
    ).clamp(0.0, 1.0)
    unknown = _render_denoise_gaussian_field(state, "unknown_score", count, default=1.0).clamp(0.0, 1.0)
    confidence = _render_denoise_gaussian_field(state, "confidence", count, default=0.0).clamp(0.0, 1.0)
    valid = (confidence >= float(confidence_threshold)) & (unknown < 0.95)
    likelihood = torch.where(valid, probability, torch.zeros_like(probability))
    return {
        "failure": likelihood,
        "dynamic_likelihood": likelihood,
        "normalized_modalities": {"render_denoise": likelihood},
        "available_modalities": ["render_denoise"],
        "valid": valid,
        "render_denoise_probability": probability,
        "render_denoise_unknown_score": unknown,
        "render_denoise_confidence": confidence,
    }


def _dynamic_prior_bootstrap(gaussians, count):
    metadata = getattr(gaussians, "dynamic_prior_metadata", {})
    if not isinstance(metadata, dict) or not metadata.get("method"):
        return None
    assignment = getattr(gaussians, "material_assignment", None)
    if assignment is None or not hasattr(assignment, "initial_probabilities"):
        return None
    probability = assignment.initial_probabilities[:, 1:].sum(-1).detach().cpu().float()
    confidence = assignment.initial_evidence_confidence.detach().cpu().float()
    unknown = assignment.initial_unknown_score.detach().cpu().float()
    if probability.numel() != int(count):
        return None
    valid = torch.isfinite(probability) & torch.isfinite(confidence) & (confidence > 0.0) & (unknown < 0.95)
    if not bool(valid.any()):
        return None
    return {
        "method": str(metadata.get("method")),
        "probability": probability.clamp(0.0, 1.0),
        "confidence": confidence.clamp(0.0, 1.0),
        "unknown": unknown.clamp(0.0, 1.0),
        "valid": valid,
    }


def _fuse_dynamic_prior_bootstrap(bootstrap, prior_bootstrap, weight):
    if prior_bootstrap is None or float(weight) <= 0.0:
        return bootstrap
    old = torch.as_tensor(bootstrap["dynamic_likelihood"]).detach().cpu().float().clamp(0.0, 1.0)
    prob = prior_bootstrap["probability"].to(old).clamp(0.0, 1.0)
    conf = prior_bootstrap["confidence"].to(old).clamp(0.0, 1.0)
    valid = prior_bootstrap["valid"].to(torch.bool)
    blend = (float(weight) * conf / (1.0 + float(weight) * conf)).clamp(0.0, 1.0)
    fused = torch.where(valid, (1.0 - blend) * old + blend * prob, old)
    normalized_modalities = dict(bootstrap.get("normalized_modalities", {}))
    normalized_modalities[f"dynamic_prior_{prior_bootstrap['method']}"] = prob * conf
    available_modalities = list(bootstrap.get("available_modalities", []))
    available_modalities.append(f"dynamic_prior_{prior_bootstrap['method']}")
    return {
        **bootstrap,
        "failure": fused,
        "dynamic_likelihood": fused,
        "normalized_modalities": normalized_modalities,
        "available_modalities": available_modalities,
        "dynamic_prior_probability": prob,
        "dynamic_prior_confidence": conf,
        "dynamic_prior_valid": valid,
    }


def observed_velocity_snr(pack, labels):
    """Separability of the observed 3-D velocity between moving and static sets.

    The rasterized flow inversion is the first stage of the pipeline and the one
    every later stage inherits.  Its useful content is not the magnitude of the
    velocity but the ratio between the speed it reports on genuinely moving
    Gaussians and on static ones: assignment, the scalar-speed solve and the
    joint competition all degrade together as that ratio approaches one.  Only
    accepted observations count, since rejected ones are stored as exact zeros
    and would otherwise be read as "static".
    """
    velocity = torch.as_tensor(pack["velocity"]).detach().cpu().double()
    speed = torch.linalg.norm(velocity, dim=-1)
    valid = torch.as_tensor(pack["valid"]).detach().cpu().bool()
    label = torch.as_tensor(labels).detach().cpu().long()
    moving = (label > 0)[None].expand_as(valid) & valid
    static = (label == 0)[None].expand_as(valid) & valid
    moving_speed = float(speed[moving].mean()) if bool(moving.any()) else float("nan")
    static_speed = float(speed[static].mean()) if bool(static.any()) else float("nan")
    if not (math.isfinite(moving_speed) and math.isfinite(static_speed)) or static_speed <= 0.0:
        snr = float("inf") if math.isfinite(moving_speed) and moving_speed > 0 else float("nan")
    else:
        snr = moving_speed / static_speed
    return {
        "snr": snr,
        "moving_mean_speed": moving_speed,
        "static_mean_speed": static_speed,
        "observation_valid_fraction": float(valid.double().mean()),
        "moving_observations": int(moving.sum()),
        "static_observations": int(static.sum()),
    }


def _raw_free_tracks_from_observed_velocity(points, frame_times, pack):
    """Integrate label-independent observed velocities into raw/free tracks."""
    xyz0 = torch.as_tensor(points).detach().cpu().float()
    times = torch.as_tensor(frame_times).detach().cpu().float()
    velocity = torch.as_tensor(pack["velocity"]).detach().cpu().float()
    valid = torch.as_tensor(pack["valid"]).detach().cpu().bool()
    confidence = torch.as_tensor(pack.get("confidence", torch.ones_like(valid, dtype=velocity.dtype))).detach().cpu().float().clamp(0.0, 1.0)
    measured = torch.where(valid[..., None], velocity, torch.zeros_like(velocity))
    observed = measured * confidence[..., None]
    dt = torch.diff(times).clamp_min(1.0e-8)
    frames = [xyz0]
    for interval in range(int(observed.shape[0])):
        frames.append(frames[-1] + observed[interval] * dt[interval])
    frame_velocity = torch.cat((observed, observed[-1:]), dim=0) if observed.numel() else torch.zeros((times.numel(), xyz0.shape[0], 3))
    return {
        "schema_version": 1,
        "source_of_truth": "label_independent_observed_velocity_integration",
        "times": times,
        "positions": torch.stack(frames),
        "velocities": frame_velocity,
        "observed_velocities": measured,
        "canonical_xyz": xyz0,
        "valid": valid,
        "confidence": confidence,
        "ground_truth_used_as_algorithm_input": False,
    }


def _otsu_threshold(scores):
    """Exact between-class-variance split of a score vector.

    Otsu's criterion is usually evaluated on a histogram, but the bin count is
    then a discretisation choice that changes the answer.  Evaluated on the
    sorted values instead it has a closed form at every candidate split, so the
    result is determined by the data alone:

        variance(k) = w0(k) w1(k) (mean0(k) - mean1(k))^2

    for each of the n - 1 places the sorted values can be cut.  Returns None
    when there is no split to make -- fewer than two distinct values -- so the
    caller states its own fallback rather than receiving an invented number.
    """
    value = torch.as_tensor(scores).detach().cpu().double().reshape(-1)
    value = value[torch.isfinite(value)]
    if value.numel() < 2:
        return None
    ordered, _ = torch.sort(value)
    if float(ordered[-1] - ordered[0]) <= 0.0:
        return None
    count = ordered.numel()
    index = torch.arange(1, count, dtype=torch.float64)
    prefix = torch.cumsum(ordered, dim=0)[:-1]
    total = float(prefix[-1] + ordered[-1])
    weight_low = index / count
    weight_high = 1.0 - weight_low
    mean_low = prefix / index
    mean_high = (total - prefix) / (count - index)
    variance = weight_low * weight_high * (mean_low - mean_high).square()
    best = int(torch.argmax(variance))
    # The split sits between the two values it separates.
    return float(0.5 * (ordered[best] + ordered[best + 1]))


def _prior_anchor_masks(prior_bootstrap, pairs, args):
    if prior_bootstrap is None:
        return None
    probability = torch.as_tensor(prior_bootstrap["probability"]).detach().cpu().float().clamp(0.0, 1.0)
    confidence = torch.as_tensor(prior_bootstrap["confidence"]).detach().cpu().float().clamp(0.0, 1.0)
    valid = torch.as_tensor(prior_bootstrap["valid"]).detach().cpu().bool()
    method = str(prior_bootstrap.get("method", ""))
    requested = max(
        0.5,
        float(getattr(args, "kpart_prior_anchor_threshold", getattr(args, "prior_hard_threshold", 0.5))),
    )
    conf_min = float(getattr(args, "kpart_confidence_threshold", 0.8))
    anchor = valid & (confidence >= conf_min)
    dynamic_seed_only = method in {"strict_closed_loop_endpoint_fm", "strict_closed_loop_rgbd_endpoint_fm"}
    if dynamic_seed_only:
        requested_mode = str(getattr(args, "kpart_prior_threshold_mode", "otsu"))
        mode = "fixed" if requested_mode == "fixed" else "probability_midpoint"
        split = requested if requested_mode == "fixed" else 0.5
        low, high = 0.0, 1.0
        if requested_mode != "fixed" and bool(valid.any()):
            scored = probability[valid]
            low, high = float(scored.min()), float(scored.max())
        margin_fraction = min(1.0, max(0.0, (requested - 0.5) / 0.5))
        dynamic_cut = (
            split + margin_fraction * max(0.0, high - split)
            if requested_mode != "fixed"
            else requested
        )
        static_anchor = torch.zeros_like(anchor)
        dynamic_anchor = anchor & (probability >= dynamic_cut)
        active = (valid & (probability >= split)).clone()
        if not bool(active.any()) and bool(dynamic_anchor.any()):
            active = dynamic_anchor.clone()
        threshold = {
            "mode": "dynamic_seed_only",
            "split_mode": mode,
            "requested_mode": requested_mode,
            "requested": requested,
            "margin_fraction": margin_fraction,
            "score_range": [low, high],
            "split": split,
            "static_cut": None,
            "dynamic_cut": dynamic_cut,
            "active_cut": split,
            "reason": "strict closed-loop prior is reliable as dynamic seed, not as static anchor",
        }
        edge = torch.as_tensor(pairs).detach().cpu().long()
        for _ in range(max(0, int(getattr(args, "kpart_prior_boundary_expansion", 0)))):
            if edge.numel() == 0 or not bool(active.any()):
                break
            grow = torch.zeros_like(active)
            left, right = edge[:, 0], edge[:, 1]
            grow[right[active[left]]] = True
            grow[left[active[right]]] = True
            active |= grow
        active = _cleanup_prior_active_mask(
            active,
            static_anchor,
            dynamic_anchor,
            edge,
            min_size=max(1, int(getattr(args, "kpart_prior_cleanup_min_size", 16))),
        )
        return {
            "static_anchor": static_anchor,
            "dynamic_anchor": dynamic_anchor,
            "active_mask": active,
            "threshold": threshold,
            "confidence_threshold": conf_min,
        }
    # The prior's ranking is far better than its calibration, so derive the
    # split from this scene's own score histogram instead of trusting the
    # nominal probability scale.  The anchors stay symmetric about the split and
    # never reach further into the distribution than the requested threshold.
    requested_mode = str(getattr(args, "kpart_prior_threshold_mode", "otsu"))
    mode = requested_mode
    low, high = 0.0, 1.0
    split = 0.5
    if requested_mode == "otsu" and bool(anchor.any()):
        scored = probability[anchor]
        candidate = _otsu_threshold(scored)
        if candidate is None:
            # A prior with no spread offers no split to find.  Say so and use
            # the nominal probability scale, rather than reporting a
            # data-driven threshold that was not derived from any data.
            mode = "fixed_no_split_available"
        else:
            split = candidate
            low, high = float(scored.min()), float(scored.max())
    # ``requested`` says how far from the split a Gaussian must sit to be
    # anchored, as a fraction of that side's own range rather than as an
    # absolute probability.  A prior whose whole dynamic mode sits at 0.35 (or
    # at 1e-23, which aborted one measured run) then still yields anchors on
    # both sides, which an absolute margin around a data-driven split does not.
    margin_fraction = min(1.0, max(0.0, (requested - 0.5) / 0.5))
    static_cut = split - margin_fraction * (split - low)
    dynamic_cut = split + margin_fraction * (high - split)
    static_anchor = anchor & (probability <= static_cut)
    dynamic_anchor = anchor & (probability >= dynamic_cut) & ~static_anchor
    threshold = {
        "mode": mode,
        "requested_mode": requested_mode,
        "requested": requested,
        "margin_fraction": margin_fraction,
        "score_range": [low, high],
        "split": split,
        "static_cut": static_cut,
        "dynamic_cut": dynamic_cut,
        "active_cut": split,
    }

    # Active K-part domain is the full prior-dynamic side of the scene split,
    # not only the highest-confidence dynamic anchors.  The anchors are frozen;
    # the rest of this mask is where static-vs-moving can still be refined.
    active = valid & (probability >= split) & ~static_anchor
    if not bool(active.any()) and bool(dynamic_anchor.any()):
        active = dynamic_anchor.clone()
    edge = torch.as_tensor(pairs).detach().cpu().long()
    for _ in range(max(0, int(getattr(args, "kpart_prior_boundary_expansion", 0)))):
        if edge.numel() == 0 or not bool(active.any()):
            break
        grow = torch.zeros_like(active)
        left, right = edge[:, 0], edge[:, 1]
        grow[right[active[left]]] = True
        grow[left[active[right]]] = True
        active |= grow & ~static_anchor
    active = _cleanup_prior_active_mask(
        active,
        static_anchor,
        dynamic_anchor,
        edge,
        min_size=max(1, int(getattr(args, "kpart_prior_cleanup_min_size", 16))),
    )

    return {
        "static_anchor": static_anchor,
        "dynamic_anchor": dynamic_anchor,
        "active_mask": active,
        "threshold": threshold,
        "confidence_threshold": conf_min,
    }


def _cleanup_prior_active_mask(active, static_anchor, dynamic_anchor, edge, *, min_size):
    active = torch.as_tensor(active).detach().cpu().bool().clone()
    static_anchor = torch.as_tensor(static_anchor).detach().cpu().bool()
    dynamic_anchor = torch.as_tensor(dynamic_anchor).detach().cpu().bool()
    edge = torch.as_tensor(edge).detach().cpu().long()
    n = int(active.numel())
    if edge.numel() == 0 or n == 0:
        return active

    adjacency = [[] for _ in range(n)]
    for first, second in edge.tolist():
        if 0 <= first < n and 0 <= second < n:
            adjacency[first].append(second)
            adjacency[second].append(first)

    def components(mask):
        seen = torch.zeros(n, dtype=torch.bool)
        starts = torch.nonzero(mask, as_tuple=False).flatten().tolist()
        for start in starts:
            if bool(seen[start]):
                continue
            stack = [start]
            seen[start] = True
            component = []
            while stack:
                node = stack.pop()
                component.append(node)
                for neighbor in adjacency[node]:
                    if bool(mask[neighbor]) and not bool(seen[neighbor]):
                        seen[neighbor] = True
                        stack.append(neighbor)
            yield component

    for component in components(active):
        ids = torch.as_tensor(component, dtype=torch.long)
        if ids.numel() < int(min_size) and not bool(dynamic_anchor[ids].any()):
            active[ids] = False

    fillable = (~active) & ~static_anchor
    for component in components(fillable):
        ids = torch.as_tensor(component, dtype=torch.long)
        if ids.numel() > int(min_size):
            continue
        touches_active = False
        touches_static = False
        for node in component:
            for neighbor in adjacency[node]:
                touches_active = touches_active or bool(active[neighbor])
                touches_static = touches_static or bool(static_anchor[neighbor])
            if touches_active and touches_static:
                break
        if touches_active and not touches_static:
            active[ids] = True
    return active


def _is_uniform_dynamic_prior(prior_bootstrap):
    if prior_bootstrap is None:
        return False
    return str(prior_bootstrap.get("method", "")).lower() in {
        "baseline",
        "all_fg",
        "all_foreground",
    }


def _bootstrap_active_mask(probability, *, minimum):
    value = torch.as_tensor(probability).detach().cpu().float().reshape(-1).clamp(0.0, 1.0)
    if value.numel() == 0:
        return torch.zeros(0, dtype=torch.bool)
    threshold = _otsu_threshold(value)
    active = value >= float(threshold)
    if int(active.sum()) < int(minimum):
        keep = min(value.numel(), max(1, int(minimum)))
        active = torch.zeros_like(active, dtype=torch.bool)
        active[torch.topk(value, keep).indices] = True
    return active


@torch.no_grad()
def _cluster_active_prior_points(points, active_mask, dynamic_probability, num_moving_parts):
    """Exactly-K spatial clusters inside the prior-dynamic domain."""
    xyz = torch.as_tensor(points).detach().cpu().float()
    active = torch.as_tensor(active_mask).detach().cpu().bool().reshape(-1)
    score = torch.as_tensor(dynamic_probability).detach().cpu().float().reshape(-1).clamp(0.0, 1.0)
    k = int(num_moving_parts)
    labels = torch.full((xyz.shape[0],), -1, dtype=torch.long)
    ids = torch.nonzero(active, as_tuple=False).flatten()
    if ids.numel() < k:
        raise RuntimeError(
            f"prior K-part clustering has {int(ids.numel())} active points for K={k}"
        )
    if k == 1:
        labels[ids] = 0
        return {"labels": labels, "centers": xyz[ids].mean(0, keepdim=True)}

    active_xyz = xyz.index_select(0, ids)
    active_score = score.index_select(0, ids).clamp_min(1.0e-4)
    span = torch.linalg.norm(active_xyz.amax(0) - active_xyz.amin(0)).clamp_min(1.0e-8)
    seed_local = [int(torch.argmax(active_score))]
    min_dist = torch.linalg.norm(active_xyz - active_xyz[seed_local[0]], dim=-1) / span
    for _ in range(1, k):
        seed_score = min_dist * (0.25 + active_score)
        seed_score[torch.as_tensor(seed_local)] = -1.0
        seed = int(torch.argmax(seed_score))
        seed_local.append(seed)
        min_dist = torch.minimum(
            min_dist,
            torch.linalg.norm(active_xyz - active_xyz[seed], dim=-1) / span,
        )
    centers = active_xyz[torch.as_tensor(seed_local)].clone()
    local_labels = torch.zeros(active_xyz.shape[0], dtype=torch.long)
    for _ in range(12):
        distance = torch.cdist(active_xyz, centers)
        local_labels = distance.argmin(-1)
        for label in range(k):
            mask = local_labels == label
            if bool(mask.any()):
                w = active_score[mask, None]
                centers[label] = (active_xyz[mask] * w).sum(0) / w.sum().clamp_min(1.0e-8)
            else:
                farthest = int(torch.argmax(distance.amin(-1).values))
                centers[label] = active_xyz[farthest]
                local_labels[farthest] = label
    labels[ids] = local_labels
    return {"labels": labels, "centers": centers}


def _apply_stage1_domain_constraints(responsibilities, domain):
    value = torch.as_tensor(responsibilities).clone()
    active = domain["active_mask"].to(device=value.device).bool()
    static_anchor = domain["static_anchor"].to(device=value.device).bool()
    dynamic_anchor = domain["dynamic_anchor"].to(device=value.device).bool()
    forced_static = (~active) | static_anchor
    if bool(forced_static.any()):
        value[forced_static] = 0.0
        value[forced_static, 0] = 1.0
    if value.shape[1] > 1 and bool(dynamic_anchor.any()):
        dyn = value[dynamic_anchor, 1:].clamp_min(0.0)
        mass = dyn.sum(-1, keepdim=True)
        fallback = torch.full_like(dyn, 1.0 / dyn.shape[1])
        value[dynamic_anchor, 0] = 0.0
        value[dynamic_anchor, 1:] = torch.where(
            mass > 1.0e-8, dyn / mass.clamp_min(1.0e-8), fallback
        )
    value /= value.sum(-1, keepdim=True).clamp_min(1.0e-8)
    return value


def _mask_unary_with_stage1_domain(unary, domain, *, penalty=1.0e4):
    value = torch.as_tensor(unary).clone()
    active = domain["active_mask"].to(device=value.device).bool()
    static_anchor = domain["static_anchor"].to(device=value.device).bool()
    dynamic_anchor = domain["dynamic_anchor"].to(device=value.device).bool()
    forced_static = (~active) | static_anchor
    if value.shape[1] > 1 and bool(forced_static.any()):
        value[forced_static, 1:] = value[forced_static, 1:] + float(penalty)
    if bool(dynamic_anchor.any()):
        value[dynamic_anchor, 0] = value[dynamic_anchor, 0] + float(penalty)
    return value


@torch.no_grad()
def _prior_kpart_initialization(points, bootstrap, prior_bootstrap, prior_anchors, args):
    """Build static plus exactly-K moving clusters directly from prior scores."""
    k = int(args.num_moving_parts)
    n = int(torch.as_tensor(points).shape[0])
    uniform_prior = _is_uniform_dynamic_prior(prior_bootstrap)
    if prior_bootstrap is not None and not uniform_prior:
        probability = torch.as_tensor(prior_bootstrap["probability"]).detach().cpu().float().reshape(-1).clamp(0.0, 1.0)
        confidence = torch.as_tensor(prior_bootstrap["confidence"]).detach().cpu().float().reshape(-1).clamp(0.0, 1.0)
        valid = torch.as_tensor(prior_bootstrap["valid"]).detach().cpu().bool().reshape(-1)
        source = str(prior_bootstrap.get("method", "dynamic_prior"))
    else:
        probability = torch.as_tensor(bootstrap["dynamic_likelihood"]).detach().cpu().float().reshape(-1).clamp(0.0, 1.0)
        confidence = torch.ones_like(probability)
        valid = torch.isfinite(probability)
        source = "bootstrap" if prior_bootstrap is None else f"{prior_bootstrap.get('method', 'prior')}_eligible_bootstrap"
    if probability.numel() != n:
        raise ValueError("prior K-part probability length does not match Gaussian count")

    if prior_anchors is None:
        static_anchor = torch.zeros(n, dtype=torch.bool)
        dynamic_anchor = torch.zeros(n, dtype=torch.bool)
        active = torch.ones(n, dtype=torch.bool) if uniform_prior else _bootstrap_active_mask(
            probability,
            minimum=max(k, int(args.rigid_min_valid_cells)),
        )
    else:
        static_anchor = prior_anchors["static_anchor"].detach().cpu().bool()
        dynamic_anchor = prior_anchors["dynamic_anchor"].detach().cpu().bool()
        active = prior_anchors["active_mask"].detach().cpu().bool()
    active = active & valid & ~static_anchor
    if int(active.sum()) < k:
        candidates = valid & ~static_anchor
        keep = min(int(candidates.sum()), max(k, int(args.rigid_min_valid_cells)))
        if keep < k:
            raise RuntimeError("prior K-part clustering has too few valid non-static candidates")
        active = torch.zeros(n, dtype=torch.bool)
        candidate_ids = torch.nonzero(candidates, as_tuple=False).flatten()
        active[candidate_ids[torch.topk(probability[candidate_ids], keep).indices]] = True

    clusters = _cluster_active_prior_points(points, active, probability, k)
    cluster_labels = clusters["labels"]
    p_dyn = (0.5 + confidence * (probability - 0.5)).clamp(0.02, 0.98)
    responsibilities = torch.zeros((n, k + 1), dtype=torch.float32)
    responsibilities[:, 0] = 1.0
    for label in range(k):
        mask = active & (cluster_labels == label)
        responsibilities[mask, 0] = 1.0 - p_dyn[mask]
        responsibilities[mask, label + 1] = p_dyn[mask]
    domain = {
        "active_mask": active,
        "static_anchor": static_anchor,
        "dynamic_anchor": dynamic_anchor,
    }
    responsibilities = _apply_stage1_domain_constraints(responsibilities, domain)
    return {
        **domain,
        "responsibilities": responsibilities,
        "cluster_labels": cluster_labels,
        "cluster_centers": clusters["centers"],
        "probability": probability,
        "confidence": confidence,
        "valid": valid,
        "source": source,
        "cluster_sizes": [
            int(((cluster_labels == label) & active).sum()) for label in range(k)
        ],
        "uniform_prior": uniform_prior,
    }


def _render_label_unary(state, count, num_moving_parts):
    probability = _render_denoise_gaussian_field(
        state,
        "final_dynamic_probability",
        count,
        default=0.5,
    ).clamp(0.0, 1.0)
    unknown = _render_denoise_gaussian_field(state, "unknown_score", count, default=1.0).clamp(0.0, 1.0)
    confidence = _render_denoise_gaussian_field(state, "confidence", count, default=0.0).clamp(0.0, 1.0)
    trust = confidence * (1.0 - unknown)
    cost = torch.zeros((int(count), int(num_moving_parts) + 1), dtype=torch.float32)
    cost[:, 0] = probability
    cost[:, 1:] = 1.0 - probability[:, None]
    return trust[:, None] * cost


def _compound_support_ids(raw_ids, material_labels, device):
    raw = torch.as_tensor(raw_ids).detach().cpu().long()
    material = torch.as_tensor(material_labels).detach().cpu().long() + 1
    _, inverse = torch.unique(torch.stack((raw, material), -1), dim=0, sorted=True, return_inverse=True)
    return inverse.to(device), int(inverse.max()) + 1


@torch.no_grad()
def _build_observation_pack(
    points,
    rotations,
    frame_times,
    camera_maps,
    cache,
    gaussians,
    pipe,
    args,
    *,
    support_kind,
    raw_cell_ids=None,
    material_labels=None,
    mode=None,
    temporal_speed=None,
    progress_description=None,
):
    """Form Gaussian-wise observations; only pooling differs by backend."""
    lagrangian = support_kind == "lagrangian"
    coordinates = None if mode is None else _coordinates(temporal_speed, frame_times)
    states, positions = [], []
    interval_indices = range(1, len(frame_times))
    if progress_description is not None:
        interval_indices = tqdm(
            interval_indices,
            desc=str(progress_description),
            unit="interval",
            leave=False,
        )
    for target_index in interval_indices:
        source_index = target_index - 1
        if mode is None:
            source_xyz, source_rotation = points, rotations
        else:
            replay = integrate_rigid_mode_exact(
                points,
                mode.to(device=points.device, dtype=points.dtype),
                coordinates[source_index].to(points),
                orientations=rotations,
            )
            source_xyz, source_rotation = replay["positions"], replay["orientations"]
        if lagrangian:
            group_ids = torch.arange(points.shape[0], device=points.device)
            num_groups = points.shape[0]
        else:
            labels = torch.zeros(points.shape[0], dtype=torch.long) if material_labels is None else material_labels
            group_ids, num_groups = _compound_support_ids(raw_cell_ids, labels, points.device)
        state = cache.observed_velocity_targets(
            source_xyz,
            list(camera_maps[source_index].values()),
            list(camera_maps[target_index].values()),
            frame_times[source_index],
            frame_times[target_index],
            group_ids=group_ids,
            num_groups=num_groups,
            gaussians=gaussians,
            pipe=pipe,
            query_rotations=source_rotation,
        )
        expanded = {}
        for key in (
            "velocity", "confidence", "valid", "base_valid", "distinct_camera_count",
            "weight_sum", "normal_matrix", "static_rms_px_mean", "static_rms_px_max",
            "moving_rms_px", "static_improvement_px", "static_improvement_relative",
        ):
            if key in state:
                expanded[key] = state[key][group_ids]
        expanded["valid"] &= expanded["distinct_camera_count"] >= int(args.rigid_min_distinct_cameras)
        states.append(expanded)
        positions.append(source_xyz.detach().cpu().double())
    pack = {"positions": torch.stack(positions)}
    for key in states[0]:
        pack[key] = torch.stack([state[key].detach().cpu() for state in states])
    pack["backend"] = support_kind
    return pack


def _robust_image(value, valid):
    result = torch.zeros_like(value)
    finite = value[valid & torch.isfinite(value)]
    if finite.numel():
        low, high = torch.quantile(finite, torch.tensor([0.1, 0.9], device=finite.device))
        result = ((value - low) / (high - low).clamp_min(1.0e-6)).clamp(0, 1)
    return torch.where(valid, result, torch.zeros_like(result))


def _robust_unit_interval(value, valid):
    tensor = torch.as_tensor(value).detach().cpu().float()
    mask = torch.as_tensor(valid).detach().cpu().bool() & torch.isfinite(tensor)
    result = torch.zeros_like(tensor)
    finite = tensor[mask]
    if finite.numel():
        low, high = torch.quantile(finite, torch.tensor([0.1, 0.9]))
        result = ((tensor - low) / (high - low).clamp_min(1.0e-6)).clamp(0.0, 1.0)
    return torch.where(mask, result, torch.zeros_like(result))


@torch.no_grad()
def _static_image_failure(points, rotations, camera_maps, gaussians, pipe, dataset, args):
    """Render canonical geometry at sampled later frames and reverse its failure."""
    from gaussian_renderer import render, render_feature_field
    from utils.loss_utils import _load_dino_target_for_view

    count = points.shape[0]
    total = torch.zeros(count)
    weight = torch.zeros(count)
    later = torch.linspace(1, len(camera_maps) - 1, min(3, len(camera_maps) - 1)).round().long().unique().tolist()
    common = sorted(set.intersection(*(set(mapping) for mapping in camera_maps)))
    background = torch.tensor(
        [1.0, 1.0, 1.0] if bool(dataset.white_background) else [0.0, 0.0, 0.0],
        device=points.device,
        dtype=points.dtype,
    )
    dino_feature = gaussians.get_dino_feature_field if bool(args.use_dino_feature_field) else None
    views = [(frame, camera_key) for frame in later for camera_key in common]
    for frame, camera_key in tqdm(
        views, desc="Phase A static RGB/depth/DINO bootstrap", unit="view", leave=False
    ):
        camera = camera_maps[int(frame)][camera_key]
        gaussians.set_free_working_state(
            points,
            float(getattr(camera, "timestamp", 0.0)),
            velocity_override=torch.zeros_like(points),
            rotation_override=rotations,
        )
        try:
            rendered = render(camera, gaussians, pipe, background, use_trained_exp=bool(dataset.train_test_exp))
            target_rgb = camera.original_image.to(points.device)
            rgb_valid = torch.isfinite(rendered["render"]).all(0) & torch.isfinite(target_rgb).all(0)
            terms = [_robust_image((rendered["render"] - target_rgb).abs().mean(0), rgb_valid)]
            validity = [rgb_valid]
            if bool(getattr(camera, "depth_reliable", False)):
                predicted_depth = rendered["depth"].squeeze()
                target_depth = camera.invdepthmap.to(points.device).squeeze()
                depth_valid = camera.depth_mask.to(points.device).bool().squeeze() & torch.isfinite(predicted_depth) & torch.isfinite(target_depth)
                terms.append(_robust_image((predicted_depth - target_depth).abs(), depth_valid))
                validity.append(depth_valid)
            if torch.is_tensor(dino_feature):
                feature = torch.cat((torch.nn.functional.normalize(dino_feature.detach().float(), dim=-1), torch.ones((count, 1), device=points.device)), -1)
                field = render_feature_field(camera, gaussians, pipe, feature, bg_feature=torch.zeros(feature.shape[1], device=points.device), detach_geometry=True)["render"]
                alpha = field[-1].clamp(0, 1)
                predicted_dino = torch.nn.functional.normalize(field[:-1] / alpha.clamp_min(1.0e-6)[None], dim=0)
                target_dino = _load_dino_target_for_view(camera, dataset, args, target_rgb.shape[-2:], device=points.device)
                if target_dino is not None:
                    channels = min(predicted_dino.shape[0], target_dino.shape[0])
                    dino_valid = (alpha > 1.0e-4) & torch.isfinite(predicted_dino[:channels]).all(0) & torch.isfinite(target_dino[:channels]).all(0)
                    distance = 1.0 - (predicted_dino[:channels] * target_dino[:channels]).sum(0)
                    terms.append(_robust_image(distance.clamp_min(0), dino_valid))
                    validity.append(dino_valid)
            pixel_valid = torch.stack(validity).any(0)
            pixel_failure = torch.stack(terms).mean(0)
            reverse = rasterize_failure_to_gaussians(
                pixel_failure,
                pixel_valid.float(),
                camera=camera,
                gaussians=gaussians,
                pipe=pipe,
                query_positions=points,
                query_rotations=rotations,
            )
            total += reverse["failure_sum"].cpu()
            weight += reverse["weight_sum"].cpu()
        finally:
            gaussians.clear_time()
    return {"failure": total / weight.clamp_min(1.0e-8), "valid": weight > 0, "weight": weight}


@torch.no_grad()
def _scene_state(points, rotations, hard_labels, fields, frame_times, frame):
    xyz, orientation = points.clone(), rotations.clone()
    labels = torch.as_tensor(hard_labels, device=points.device).long()
    for part, field in enumerate(fields, start=1):
        coordinate = _coordinates(field["temporal_speed"], frame_times)[int(frame)]
        moved = integrate_rigid_mode_exact(
            points,
            field["mode"].to(device=points.device, dtype=points.dtype),
            coordinate.to(points),
            orientations=rotations,
        )
        mask = labels == part
        xyz[mask] = moved["positions"][mask]
        orientation[mask] = moved["orientations"][mask]
    return xyz, orientation


@torch.no_grad()
def _region_field_scene_state(points, rotations, region_labels, region_fields, frame_times, frame):
    xyz, orientation = points.clone(), rotations.clone()
    labels = torch.as_tensor(region_labels, device=points.device).long()
    for region, field in region_fields.items():
        if not bool(field.get("valid", False)):
            continue
        coordinate = _coordinates(field["temporal_speed"], frame_times)[int(frame)]
        moved = integrate_rigid_mode_exact(
            points,
            field["mode"].to(device=points.device, dtype=points.dtype),
            coordinate.to(points),
            orientations=rotations,
        )
        mask = labels == int(region)
        xyz[mask] = moved["positions"][mask]
        orientation[mask] = moved["orientations"][mask]
    return xyz, orientation


def _make_merge_render_evaluator(points, rotations, frame_times, camera_maps, gaussians, pipe, dataset, args):
    from utils.loss_utils import _load_dino_target_for_view

    common = sorted(set.intersection(*(set(mapping) for mapping in camera_maps)))
    background = torch.tensor(
        [1.0, 1.0, 1.0] if bool(dataset.white_background) else [0.0, 0.0, 0.0],
        device=points.device,
        dtype=points.dtype,
    )
    dino_features = gaussians.get_dino_feature_field if bool(args.use_dino_feature_field) else None

    @torch.no_grad()
    def evaluate(region_a, region_b, shared_field, *, context=None):
        if context is None or not common or not bool(shared_field.get("valid", False)):
            return 0.0
        labels = torch.as_tensor(context["labels"]).detach().cpu().long()
        fields = context["fields"]
        if not (fields[int(region_a)]["valid"] and fields[int(region_b)]["valid"]):
            return 0.0
        component = (labels == int(region_a)) | (labels == int(region_b))
        if int(component.sum()) < 2:
            return 0.0
        coordinate = _coordinates(shared_field["temporal_speed"], frame_times).abs()
        informative_frames = [int(item) for item in torch.argsort(coordinate, descending=True).tolist() if int(item) > 0][:2]
        if not informative_frames:
            return 0.0
        merged_labels = labels.clone()
        merged_labels[labels == int(region_b)] = int(region_a)
        merged_fields = dict(fields)
        merged_fields[int(region_a)] = shared_field
        merged_fields.pop(int(region_b), None)
        component_mask = component.to(points.device)
        separate_losses, merged_losses = [], []
        sample = torch.nonzero(component, as_tuple=False).flatten()
        sample = sample[torch.linspace(0, sample.numel() - 1, min(128, sample.numel())).round().long().unique()].to(points.device)
        for frame in informative_frames:
            separate_xyz, separate_rotation = _region_field_scene_state(points, rotations, labels, fields, frame_times, frame)
            merged_xyz, merged_rotation = _region_field_scene_state(points, rotations, merged_labels, merged_fields, frame_times, frame)
            visibility = []
            for camera_key in common:
                _, _, visible = project_world_points(merged_xyz[sample], camera_maps[frame][camera_key])
                visibility.append((int(visible.sum()), camera_key))
            selected_cameras = [item[1] for item in sorted(visibility, reverse=True)[: min(2, len(visibility))]]
            for camera_key in selected_cameras:
                camera = camera_maps[frame][camera_key]
                separate_render = render_full_scene_with_candidate_label(
                    camera, gaussians, pipe, background, separate_xyz, separate_rotation,
                    component_mask=component_mask, dino_features=dino_features,
                    use_trained_exp=bool(dataset.train_test_exp),
                )
                merged_render = render_full_scene_with_candidate_label(
                    camera, gaussians, pipe, background, merged_xyz, merged_rotation,
                    component_mask=component_mask, dino_features=dino_features,
                    use_trained_exp=bool(dataset.train_test_exp),
                )
                affected = compute_local_affected_region(separate_render["component_alpha"], merged_render["component_alpha"])
                target_dino = _load_dino_target_for_view(camera, dataset, args, camera.original_image.shape[-2:], device=points.device) if torch.is_tensor(dino_features) else None
                kwargs = {
                    "target_depth": camera.invdepthmap.to(points.device) if bool(getattr(camera, "depth_reliable", False)) else None,
                    "depth_valid": camera.depth_mask.to(points.device) if bool(getattr(camera, "depth_reliable", False)) else None,
                    "target_dino": target_dino,
                    "rgb_weight": float(args.kpart_render_rgb_weight),
                    "depth_weight": float(args.kpart_render_depth_weight),
                    "dino_weight": float(args.kpart_render_dino_weight),
                }
                target_rgb = camera.original_image.to(points.device)
                separate_loss = compute_rgb_depth_dino_flow_energy(separate_render, target_rgb, affected, **kwargs)
                merged_loss = compute_rgb_depth_dino_flow_energy(merged_render, target_rgb, affected, **kwargs)
                if separate_loss["valid"] and merged_loss["valid"]:
                    separate_losses.append(float(separate_loss["total"]))
                    merged_losses.append(float(merged_loss["total"]))
        if not merged_losses:
            return 0.0
        return sum(merged_losses) / len(merged_losses) - sum(separate_losses) / len(separate_losses)

    return evaluate


@torch.no_grad()
def _evaluate_render_components(
    components,
    responsibilities,
    fields,
    points,
    rotations,
    frame_times,
    camera_maps,
    gaussians,
    pipe,
    dataset,
    args,
    persistent,
    cycle,
):
    from utils.loss_utils import _load_dino_target_for_view

    hard = responsibilities.argmax(-1).detach().cpu()
    common = sorted(set.intersection(*(set(mapping) for mapping in camera_maps)))
    summaries = []
    if not common:
        return persistent, summaries
    background = torch.tensor(
        [1.0, 1.0, 1.0] if bool(dataset.white_background) else [0.0, 0.0, 0.0],
        device=points.device,
        dtype=points.dtype,
    )
    dino_features = gaussians.get_dino_feature_field if bool(args.use_dino_feature_field) else None
    for component_id, component in enumerate(components):
        ids = torch.as_tensor(component).long()
        if ids.numel() == 0:
            continue
        moving_hint = max(1, int(torch.mode(hard[ids]).values))
        sample = ids[torch.linspace(0, ids.numel() - 1, min(64, ids.numel())).round().long().unique()].to(points.device)
        projected = []
        hinted = hard.clone()
        hinted[ids] = moving_hint
        for frame in range(len(frame_times)):
            xyz, _ = _scene_state(points, rotations, hinted, fields, frame_times, frame)
            pixels, _, visible = project_world_points(xyz[sample], camera_maps[frame][common[0]])
            pixels = pixels.clone()
            pixels[~visible] = float("nan")
            projected.append(pixels.cpu())
        pairs = select_informative_frame_pairs(
            torch.stack(projected),
            count=int(args.kpart_render_frame_pairs_per_component),
            min_displacement=float(args.kpart_render_min_projected_displacement),
            max_displacement=float(args.kpart_render_max_projected_displacement),
            bidirectional=True,
        )
        if not pairs:
            continue
        component_mask = torch.zeros(points.shape[0], device=points.device, dtype=torch.bool)
        component_mask[ids.to(points.device)] = True
        label_energy = {}
        label_rows = {}
        current_states = {}
        for pair in pairs:
            frame = int(pair["target"])
            current_states[frame] = _scene_state(points, rotations, hard, fields, frame_times, frame)
        for label in range(responsibilities.shape[1]):
            candidate_labels = hard.clone()
            candidate_labels[ids] = label
            rows = []
            for pair in pairs:
                frame = int(pair["target"])
                current_xyz, current_rotation = current_states[frame]
                candidate_xyz, candidate_rotation = _scene_state(points, rotations, candidate_labels, fields, frame_times, frame)
                for camera_key in common:
                    camera = camera_maps[frame][camera_key]
                    current_render = render_full_scene_with_candidate_label(
                        camera, gaussians, pipe, background, current_xyz, current_rotation,
                        component_mask=component_mask, dino_features=dino_features,
                        use_trained_exp=bool(dataset.train_test_exp),
                    )
                    candidate_render = render_full_scene_with_candidate_label(
                        camera, gaussians, pipe, background, candidate_xyz, candidate_rotation,
                        component_mask=component_mask, dino_features=dino_features,
                        use_trained_exp=bool(dataset.train_test_exp),
                    )
                    affected = compute_local_affected_region(current_render["component_alpha"], candidate_render["component_alpha"])
                    target_dino = _load_dino_target_for_view(camera, dataset, args, camera.original_image.shape[-2:], device=points.device) if torch.is_tensor(dino_features) else None
                    rows.append(compute_rgb_depth_dino_flow_energy(
                        candidate_render,
                        camera.original_image.to(points.device),
                        affected,
                        target_depth=(camera.invdepthmap.to(points.device) if bool(getattr(camera, "depth_reliable", False)) else None),
                        depth_valid=(camera.depth_mask.to(points.device) if bool(getattr(camera, "depth_reliable", False)) else None),
                        target_dino=target_dino,
                        rgb_weight=float(args.kpart_render_rgb_weight),
                        depth_weight=float(args.kpart_render_depth_weight),
                        dino_weight=float(args.kpart_render_dino_weight),
                    ))
            valid = [row for row in rows if row["valid"]]
            if valid:
                energy = sum(float(row["total"]) for row in valid) / len(valid)
                label_energy[label] = energy
                label_rows[label] = {name: sum(float(row[name]) for row in valid if bool(torch.isfinite(torch.as_tensor(row[name])))) / max(1, sum(bool(torch.isfinite(torch.as_tensor(row[name]))) for row in valid)) for name in ("total", "rgb", "depth", "dino")}
        persistent = update_persistent_render_evidence(
            persistent,
            ids,
            label_energy,
            cycle=cycle,
            field_version=cycle,
            frame_pair_signature=[(int(pair["source"]), int(pair["target"])) for pair in pairs],
        )
        summaries.append({
            "cycle": cycle,
            "component_id": component_id,
            "gaussian_count": int(ids.numel()),
            "frame_pairs": pairs,
            "per_label": label_rows,
            "winner": min(label_energy, key=label_energy.get) if label_energy else None,
        })
    return persistent, summaries


def _field_distance(old, new, points, frame_times):
    if old is None or not old["valid"] or not new["valid"]:
        return float("inf")
    sample = torch.linspace(0, points.shape[0] - 1, min(512, points.shape[0])).round().long().unique()
    xyz = points.detach().cpu().double()[sample]
    from utils.rigid_killing_motion import predict_rigid_cell_velocity

    a = predict_rigid_cell_velocity(old["mode"], xyz, old["temporal_speed"])
    b = predict_rigid_cell_velocity(new["mode"], xyz, new["temporal_speed"])
    scale = torch.linalg.norm(a, dim=-1).median().clamp_min(1.0e-8)
    return float(torch.linalg.norm(a - b, dim=-1).median() / scale)


def _visual_classification(labels, num_classes):
    label = torch.as_tensor(labels).detach().cpu().long()
    valid = label >= 0
    safe = label.clamp_min(0)
    responsibility = torch.nn.functional.one_hot(safe, int(num_classes)).float()
    responsibility[~valid] = 1.0 / float(num_classes)
    return responsibility, {
        "labels": label,
        "entropy": -(responsibility * torch.log(responsibility.clamp_min(1.0e-8))).sum(-1),
    }


def _write_seam_html(path, probability, part_boundary, dino_boundary, *, title):
    import torch.nn.functional as F

    part = torch.as_tensor(part_boundary).detach().cpu().float()[None, None]
    dino = torch.as_tensor(dino_boundary).detach().cpu().float()[None, None]
    height, width = part.shape[-2:]
    scale = min(1.0, 192.0 / max(height, width))
    size = (max(1, int(height * scale)), max(1, int(width * scale)))
    part = F.interpolate(part, size=size, mode="bilinear", align_corners=False)[0, 0].clamp(0, 1)
    dino = F.interpolate(dino, size=size, mode="bilinear", align_corners=False)[0, 0].clamp(0, 1)
    moving = torch.as_tensor(probability).detach().cpu().float()[1:].sum(0)[None, None]
    moving = F.interpolate(moving, size=size, mode="bilinear", align_corners=False)[0, 0].clamp(0, 1)
    overlay = torch.stack((part, dino, torch.zeros_like(part)), -1)
    payload = {
        "moving": moving.tolist(),
        "part": part.tolist(),
        "dino": dino.tolist(),
        "overlay": (255.0 * overlay).round().byte().tolist(),
    }
    document = f"""<!doctype html><html><head><meta charset='utf-8'><script src='https://cdn.plot.ly/plotly-2.35.2.min.js'></script></head><body><h2>{title}</h2><div style='display:grid;grid-template-columns:1fr 1fr'><div id='moving'></div><div id='part'></div><div id='dino'></div><div id='overlay'></div></div><script>const p={json.dumps(payload)};Plotly.newPlot('moving',[{{type:'heatmap',z:p.moving,colorscale:'Viridis'}}],{{title:'rendered moving probability'}});Plotly.newPlot('part',[{{type:'heatmap',z:p.part,colorscale:'Magma'}}],{{title:'predicted part seam'}});Plotly.newPlot('dino',[{{type:'heatmap',z:p.dino,colorscale:'Magma'}}],{{title:'image DINO boundary'}});Plotly.newPlot('overlay',[{{type:'image',z:p.overlay}}],{{title:'seam (red) over DINO boundary (green)'}});</script></body></html>"""
    Path(path).write_text(document, encoding="utf-8")


@torch.no_grad()
def run_velocity_field_kpart_discovery(scene, gaussians, pipe, dataset, args, *, log_fn=None):
    """Run Phase A field merge followed by Phase B soft K+1 field EM."""
    from utils.unified_dynamic_training import _camera_map, _complete_frame_groups, _fixed_camera_metadata, _ordered_dynamic_cameras
    from utils.kpart_rigid_discovery import _build_coarse_tracks, _write_cycle_html
    from utils.visualize_transported_gaussians_html import write_transported_gaussians_html

    k = int(args.num_moving_parts)
    support_kind = str(args.kpart_support)
    cameras = _ordered_dynamic_cameras(scene.getTrainCameras())
    groups = _complete_frame_groups(cameras, args)
    if len(groups) < 2:
        raise RuntimeError("velocity-field K-part discovery requires at least two complete frames")
    frame_times = [float(item[1]) for item in groups]
    camera_maps = [_camera_map(item[2]) for item in groups]
    points = gaussians.get_canonical_xyz.detach()
    rotations = gaussians.get_rotation.detach().clone()
    render_denoise_mode = str(getattr(args, "dynamic_discovery_mode", "legacy")) == "render_denoise"
    render_state = _render_denoise_state(gaussians)
    dino_features = gaussians.get_dino_feature_field
    sam3_features = getattr(gaussians, "get_sam3_feature_field", None)
    has_dino_features = (
        torch.is_tensor(dino_features)
        and dino_features.ndim == 2
        and dino_features.shape[0] == points.shape[0]
        and dino_features.numel() > 0
        and bool(getattr(args, "kpart_dino", False))
    )
    has_sam3_features = (
        torch.is_tensor(sam3_features)
        and sam3_features.ndim == 2
        and sam3_features.shape[0] == points.shape[0]
        and sam3_features.numel() > 0
        and bool(getattr(args, "kpart_sam3", False))
    )
    prior_bootstrap = _dynamic_prior_bootstrap(gaussians, points.shape[0])
    prior_bootstrap_enabled = (
        prior_bootstrap is not None
        and float(getattr(args, "kpart_dynamic_prior_weight", 1.0)) > 0.0
    )
    endpoint_enabled = bool(getattr(args, "kpart_endpoint_transport", False))
    if not (
        render_denoise_mode
        or has_dino_features
        or has_sam3_features
        or endpoint_enabled
        or prior_bootstrap_enabled
    ):
        raise RuntimeError(
            "velocity-field K-part discovery requires DINO, SAM3, render-denoise, endpoint transport, or an explicit DynamicPrior"
        )
    if render_denoise_mode and render_state is None:
        raise RuntimeError("render-denoise K-part discovery requires Stage A render evidence")
    output = Path(scene.model_path) / "dynamic_debug"
    output.mkdir(parents=True, exist_ok=True)

    endpoint_transport = None
    endpoint_debug = output / "endpoint_transport"
    if endpoint_enabled:
        endpoint_state = getattr(gaussians, "endpoint_static_state", None)
        if endpoint_state is None:
            raise RuntimeError("--kpart_endpoint_transport requires prepared endpoint_static_state")
        endpoint_transport = RigidEndpointTransport(
            endpoint_state,
            EndpointTransportConfig(
                enabled=True,
                knn=int(args.endpoint_knn),
                radius=float(args.endpoint_radius),
                temperature=float(args.endpoint_transport_temperature),
                geometry_weight=float(args.endpoint_geometry_weight),
                normal_weight=float(args.endpoint_normal_weight),
                sam3_weight=float(args.endpoint_sam3_weight),
                confidence_min=float(args.endpoint_confidence_min),
                robust_delta=float(args.endpoint_robust_delta),
                metric=str(args.endpoint_geometry_metric),
                mode=str(args.endpoint_transport_mode),
            ),
        )
        endpoint_debug.mkdir(parents=True, exist_ok=True)
        write_endpoint_target_summary(endpoint_debug / "target_summary.json", endpoint_transport)

    sam3_graph = None
    sam3_same_material = None
    if has_sam3_features:
        sam3_graph = build_canonical_sam3_graph(
            points,
            sam3_features,
            neighbors=int(args.kpart_initial_supernode_knn),
            max_pairs=int(args.phase1_pair_max_pairs),
            query_chunk_size=int(args.lagrangian_knn_chunk_size),
        )
        sam3_same_material = compute_3d_sam3_same_material_affinity(
            points,
            sam3_graph["pairs"],
            sam3_graph["sam3_distance"],
            sigma=float(args.kpart_sam3_similarity_sigma),
        )

    if render_denoise_mode or (not has_dino_features and not has_sam3_features):
        print("[Stage 1 velocity fields] building canonical spatial graph")
        graph = build_canonical_gaussian_knn(
            points,
            neighbors=int(args.kpart_initial_supernode_knn),
            max_pairs=int(args.phase1_pair_max_pairs),
            query_chunk_size=int(args.lagrangian_knn_chunk_size),
        )
        graph["dino_distance"] = torch.zeros(graph["pairs"].shape[0], dtype=torch.float32)
        same_material = compute_3d_same_material_affinity(
            points,
            graph["pairs"],
            graph["dino_distance"],
            dino_beta=0.0,
        )
    elif has_dino_features:
        print("[Stage 1 velocity fields] building canonical DINO/spatial graph")
        graph = build_canonical_dino_graph(
            points,
            dino_features,
            neighbors=int(args.kpart_initial_supernode_knn),
            max_pairs=int(args.phase1_pair_max_pairs),
            query_chunk_size=int(args.lagrangian_knn_chunk_size),
        )
        same_material = compute_3d_same_material_affinity(points, graph["pairs"], graph["dino_distance"])
    else:
        print("[Stage 1 velocity fields] building canonical SAM3/spatial graph")
        graph = dict(sam3_graph)
        graph["dino_distance"] = sam3_graph["sam3_distance"]
        same_material = sam3_same_material
    raw_cell_ids = None
    if support_kind == "eulerian":
        grid = assign_gaussians_to_cells(
            points,
            int(args.curl_grid_target_cells),
            padding_fraction=float(args.curl_grid_padding),
            min_cells_per_axis=int(args.curl_grid_min_cells_per_axis),
        )
        raw_cell_ids = grid.point_group_ids
    cache = MotionCorrespondenceCache(dataset, args)
    if not cache.enabled:
        raise RuntimeError("velocity-field K-part discovery requires rasterized optical-flow observations")
    static_pack = _build_observation_pack(
        points, rotations, frame_times, camera_maps, cache, gaussians, pipe, args,
        support_kind=support_kind, raw_cell_ids=raw_cell_ids,
        progress_description="Phase A static-flow observations",
    )
    flow_failure = (static_pack["static_rms_px_mean"].double() * static_pack["confidence"].double()).sum(0) / static_pack["confidence"].double().sum(0).clamp_min(1.0e-8)
    if render_denoise_mode:
        bootstrap = _render_denoise_bootstrap(
            render_state,
            points.shape[0],
            float(args.render_denoise_confidence_threshold),
        )
        _atomic_torch_save(
            {
                "schema_version": 2,
                **bootstrap,
                "flow_failure_for_rigid_fit_only": flow_failure,
                "dino_used_for_static_dynamic_discovery": False,
            },
            output / "phase1_static_render_failure.pt",
        )
    else:
        image_failure = _static_image_failure(points, rotations, camera_maps, gaussians, pipe, dataset, args)
        bootstrap = compute_static_render_failure(
            {"flow": flow_failure, "render": image_failure["failure"]},
            weights={"flow": 1.0, "render": float(args.kpart_initial_dynamic_bootstrap_weight)},
            validity={"flow": static_pack["valid"].any(0), "render": image_failure["valid"]},
        )
        _atomic_torch_save({"schema_version": 1, **bootstrap, "flow_failure": flow_failure, "image_failure": image_failure}, output / "phase1_static_render_failure.pt")

    prior_anchor_enabled = prior_bootstrap_enabled and not _is_uniform_dynamic_prior(prior_bootstrap)
    bootstrap = _fuse_dynamic_prior_bootstrap(
        bootstrap,
        prior_bootstrap,
        float(getattr(args, "kpart_dynamic_prior_weight", 1.0)) if prior_anchor_enabled else 0.0,
    )
    prior_anchors = (
        _prior_anchor_masks(prior_bootstrap, graph["pairs"], args)
        if prior_anchor_enabled
        else None
    )
    if prior_bootstrap_enabled:
        _atomic_torch_save(
            {
                "schema_version": 1,
                "method": prior_bootstrap["method"],
                "probability": prior_bootstrap["probability"],
                "confidence": prior_bootstrap["confidence"],
                "valid": prior_bootstrap["valid"],
                "fused_dynamic_likelihood": bootstrap["dynamic_likelihood"],
                "weight": float(getattr(args, "kpart_dynamic_prior_weight", 1.0)),
                "static_anchor": prior_anchors["static_anchor"] if prior_anchors is not None else None,
                "dynamic_anchor": prior_anchors["dynamic_anchor"] if prior_anchors is not None else None,
                "kpart_active_mask": prior_anchors["active_mask"] if prior_anchors is not None else None,
                "anchor_threshold": prior_anchors["threshold"] if prior_anchors is not None else None,
                "anchor_confidence_threshold": prior_anchors["confidence_threshold"] if prior_anchors is not None else None,
                "boundary_expansion": int(getattr(args, "kpart_prior_boundary_expansion", 0)),
                "cleanup_min_size": int(getattr(args, "kpart_prior_cleanup_min_size", 16)),
                "ground_truth_used_as_algorithm_input": False,
            },
            output / "phase1_dynamic_prior_bootstrap.pt",
        )

    endpoint_bootstrap = None
    if endpoint_transport is not None and float(args.endpoint_unary_weight) > 0.0:
        identity_prediction = points[:, None, :].detach()
        endpoint_bootstrap_result = endpoint_transport.evaluate(
            points,
            identity_prediction,
            canonical_sam3=sam3_features if bool(getattr(args, "use_sam3_feature_field", False)) else None,
        )
        write_endpoint_transport_diagnostics(
            endpoint_bootstrap_result,
            points,
            identity_prediction,
            endpoint_debug,
            0,
            max_points=int(args.transported_html_max_points),
        )
        endpoint_static_energy = endpoint_bootstrap_result.energy[:, 0].cpu()
        endpoint_static_valid = endpoint_bootstrap_result.valid[:, 0].cpu()
        endpoint_static_likelihood = _robust_unit_interval(
            endpoint_static_energy,
            endpoint_static_valid,
        )
        confidence = endpoint_bootstrap_result.confidence[:, 0].cpu().clamp(0.0, 1.0)
        cue = endpoint_static_likelihood * confidence
        old = torch.as_tensor(bootstrap["dynamic_likelihood"]).detach().cpu().float()
        valid = endpoint_static_valid & torch.isfinite(cue)
        weight = float(args.endpoint_unary_weight)
        fused = torch.where(valid, (old + weight * cue) / (1.0 + weight), old)
        bootstrap["dynamic_likelihood"] = fused
        bootstrap["failure"] = fused
        normalized_modalities = dict(bootstrap.get("normalized_modalities", {}))
        normalized_modalities["endpoint_static_identity_failure"] = cue
        bootstrap["normalized_modalities"] = normalized_modalities
        available = list(bootstrap.get("available_modalities", []))
        available.append("endpoint_static_identity_failure")
        bootstrap["available_modalities"] = available
        endpoint_bootstrap = {
            "energy": endpoint_static_energy,
            "valid": endpoint_static_valid,
            "confidence": confidence,
            "dynamic_likelihood": cue,
        }
        _atomic_torch_save(
            {"schema_version": 1, **endpoint_bootstrap, "ground_truth_used_as_algorithm_input": False},
            output / "phase1_endpoint_static_bootstrap.pt",
        )

    initialization = _prior_kpart_initialization(
        points,
        bootstrap,
        prior_bootstrap,
        prior_anchors,
        args,
    )
    active = initialization["active_mask"]
    print(
        f"[Stage 1 prior K-part] initialized {k} prior-driven moving cluster(s) "
        f"from {int(active.sum())} active candidates; static leftover={int((~active).sum())}"
    )
    _atomic_torch_save(
        {
            "schema_version": 2,
            **initialization,
            "ground_truth_used_as_algorithm_input": False,
        },
        output / "phase1_prior_kpart_initialization.pt",
    )
    initial_summary = {
        "schema_version": 2,
        "initialization": "exact_k_prior_probability_clusters_plus_static",
        "kprime_discovery_removed": True,
        "num_moving_parts": k,
        "cluster_sizes": initialization["cluster_sizes"],
        "active_count": int(active.sum()),
        "static_leftover_count": int((~active).sum()),
        "static_anchor_count": int(initialization["static_anchor"].sum()),
        "dynamic_anchor_count": int(initialization["dynamic_anchor"].sum()),
        "dynamic_likelihood_quantiles": torch.quantile(bootstrap["dynamic_likelihood"], torch.tensor([0.0, 0.25, 0.5, 0.75, 1.0])).tolist(),
        "material_affinity_quantiles": torch.quantile(same_material["affinity"], torch.tensor([0.0, 0.25, 0.5, 0.75, 1.0])).tolist(),
        "dino_used_for_static_dynamic_discovery": False,
        "sam3_used_for_static_dynamic_discovery": False,
        "dino_used_for_kpart_regularization": bool(
            has_dino_features
            and (
                float(getattr(args, "kpart_dino_3d_weight", 0.0)) > 0.0
                or float(getattr(args, "kpart_dino_2d_seam_weight", 0.0)) > 0.0
            )
        ),
        "sam3_used_for_kpart_regularization": bool(
            has_sam3_features
            and (
                float(getattr(args, "kpart_sam3_3d_weight", 0.0)) > 0.0
                or float(getattr(args, "kpart_sam3_2d_seam_weight", 0.0)) > 0.0
            )
        ),
        "endpoint_bootstrap_used": endpoint_bootstrap is not None,
        "dynamic_prior_bootstrap_used": bool(prior_bootstrap_enabled),
        "dynamic_prior_method": None if prior_bootstrap is None else prior_bootstrap["method"],
        "dynamic_prior_weight": float(getattr(args, "kpart_dynamic_prior_weight", 1.0)),
        "prior_static_anchor_count": 0 if prior_anchors is None else int(prior_anchors["static_anchor"].sum()),
        "prior_dynamic_anchor_count": 0 if prior_anchors is None else int(prior_anchors["dynamic_anchor"].sum()),
        "prior_kpart_active_count": 0 if prior_anchors is None else int(prior_anchors["active_mask"].sum()),
        "prior_boundary_expansion": int(getattr(args, "kpart_prior_boundary_expansion", 0)),
        "prior_cleanup_min_size": int(getattr(args, "kpart_prior_cleanup_min_size", 16)),
        "ground_truth_used_as_algorithm_input": False,
    }
    (output / "phase1_prior_kpart_initialization_summary.json").write_text(json.dumps(initial_summary, indent=2), encoding="utf-8")
    initial_visual_labels = torch.where(
        active,
        initialization["cluster_labels"] + 1,
        torch.zeros_like(initialization["cluster_labels"]),
    )
    initial_responsibility, initial_classification = _visual_classification(
        initial_visual_labels,
        k + 1,
    )
    _write_cycle_html(
        points, initial_responsibility, initial_classification,
        output / "phase1_prior_kpart_initialization.html",
        title="Stage 1 prior-constrained K-part initialization",
        max_points=int(args.transported_html_max_points),
        dino_graph={"pairs": graph["pairs"], "strong_affinity": same_material["affinity"] >= 0.5},
    )

    split_static_pack = _build_observation_pack(
        points, rotations, frame_times, camera_maps, cache, gaussians, pipe, args,
        support_kind=support_kind, raw_cell_ids=raw_cell_ids,
        material_labels=initialization["cluster_labels"],
        progress_description="Phase A prior-cluster flow observations",
    )
    solver_kwargs = {
        "max_iters": int(args.rigid_solver_max_iters),
        "tolerance": float(args.rigid_solver_tol),
        "irls_iters": int(args.rigid_solver_irls_iters),
        "huber_delta": float(args.rigid_solver_huber_delta),
        "min_valid_cells": int(args.rigid_min_valid_cells),
        "angular_eps": float(args.rigid_angular_eps),
    }
    fields = []
    for part in range(k):
        fitted = fit_region_rigid_field(
            split_static_pack,
            initialization["responsibilities"][:, part + 1],
            responsibility_power=float(args.kpart_field_fit_responsibility_power),
            solver_kwargs=solver_kwargs,
        )
        if not fitted["valid"]:
            raise RuntimeError(f"Stage 1 prior K-part field {part + 1} has too little support")
        fields.append(fitted)
    print("[Stage 1 prior K-part] Phase A produced exactly K moving fields")
    _atomic_torch_save(
        {
            "schema_version": 2,
            "labels": initialization["cluster_labels"],
            "fields": [
                {
                    "mode": field["mode"].as_dict(),
                    "temporal_speed": field["temporal_speed"],
                    "cost": field["cost"],
                }
                for field in fields
            ],
            "kprime_discovery_removed": True,
            "ground_truth_used_as_algorithm_input": False,
        },
        output / "phase1_prior_kpart_fields.pt",
    )
    initial_field = compute_velocity_field_unaries([static_pack] + [split_static_pack] * k, fields, huber_delta=float(args.rigid_solver_huber_delta))
    normalized_field = normalize_unary_families({"field": {"values": initial_field["unaries"], "valid": torch.isfinite(initial_field["unaries"])}})["families"]["field"]
    normalized_field = _mask_unary_with_stage1_domain(normalized_field, initialization)
    responsibilities = initialize_responsibilities_from_unaries(normalized_field, temperature=float(args.kpart_soft_temperature)).to(points.device)
    initial_hint = initialization["responsibilities"]
    responsibilities = (0.75 * responsibilities.cpu() + 0.25 * initial_hint).to(points.device)
    responsibilities = _apply_stage1_domain_constraints(responsibilities, initialization)
    if render_denoise_mode:
        render_probability = _render_denoise_gaussian_field(
            render_state,
            "final_dynamic_probability",
            points.shape[0],
            default=0.5,
        )
        render_unknown = _render_denoise_gaussian_field(
            render_state,
            "unknown_score",
            points.shape[0],
            default=1.0,
        )
        render_confidence = _render_denoise_gaussian_field(
            render_state,
            "confidence",
            points.shape[0],
            default=0.0,
        )
        render_hint = torch.zeros_like(responsibilities.cpu())
        render_hint[:, 0] = 1.0 - render_probability
        render_hint[:, 1:] = render_probability[:, None] / max(1, k)
        render_hint /= render_hint.sum(-1, keepdim=True).clamp_min(1.0e-8)
        blend = (render_confidence * (1.0 - render_unknown)).clamp(0.0, 1.0)[:, None]
        responsibilities = ((1.0 - blend) * responsibilities.cpu() + blend * render_hint).to(points.device)
        responsibilities = _apply_stage1_domain_constraints(responsibilities, initialization)
    responsibilities /= responsibilities.sum(-1, keepdim=True)
    gaussians.material_assignment.logits.copy_(torch.log(responsibilities.clamp_min(1.0e-8)))

    persistent = {
        "evidence": torch.full((points.shape[0], k + 1), float("nan")),
        "valid": torch.zeros((points.shape[0], k + 1), dtype=torch.bool),
        "last_update": torch.full((points.shape[0], k + 1), -1, dtype=torch.long),
    }
    ablation_labels = {
        "prior_kpart_initialization": torch.where(active, initialization["cluster_labels"] + 1, torch.zeros_like(initialization["cluster_labels"])),
        "initial_soft": responsibilities.argmax(-1).cpu(),
    }
    history, responsibility_history, render_summaries = [], [], []
    discovery_cycles = int(args.kpart_cycles) if args.kpart_cycles is not None else int(args.kpart_anneal_cycles) + int(args.kpart_plateau_cycles)
    total_cycles = discovery_cycles + int(args.kpart_final_completion_passes)
    stable = 0
    sharpened = False
    converged_at_cycle = None
    previous_render = None
    progress = tqdm(range(total_cycles), desc=f"Stage 1 velocity-field {support_kind} K-part", unit="cycle")
    final_field_energy = initial_field
    last_seam_artifact = None
    for cycle in progress:
        previous_responsibilities = responsibilities.detach().clone()
        previous_fields = fields
        hard_pooling = responsibilities.argmax(-1).detach().cpu()
        static_pack = _build_observation_pack(
            points, rotations, frame_times, camera_maps, cache, gaussians, pipe, args,
            support_kind=support_kind, raw_cell_ids=raw_cell_ids,
            material_labels=hard_pooling,
            progress_description=f"Phase B cycle {cycle + 1} static observations",
        )
        candidate_packs, refit_fields = [], []
        for part in range(k):
            pack = _build_observation_pack(
                points, rotations, frame_times, camera_maps, cache, gaussians, pipe, args,
                support_kind=support_kind, raw_cell_ids=raw_cell_ids,
                material_labels=hard_pooling,
                mode=fields[part]["mode"], temporal_speed=fields[part]["temporal_speed"],
                progress_description=f"Phase B cycle {cycle + 1} part {part + 1} observations",
            )
            candidate_packs.append(pack)
            fitted = fit_region_rigid_field(
                pack,
                responsibilities[:, part + 1].cpu(),
                responsibility_power=float(args.kpart_field_fit_responsibility_power),
                reference_point=fields[part]["mode"].reference_point,
                solver_kwargs=solver_kwargs,
            )
            refit_fields.append(fitted if fitted["valid"] else fields[part])
        fields = refit_fields
        part_field_changes = [
            _field_distance(previous_fields[index], fields[index], points, frame_times)
            for index in range(k)
        ]
        invalidated_render_labels = []
        for part, change in enumerate(part_field_changes, start=1):
            if math.isfinite(change) and change > float(args.kpart_render_field_invalidation_threshold):
                persistent["valid"][:, part] = False
                persistent["evidence"][:, part] = float("nan")
                invalidated_render_labels.append(part)
        field_energy = compute_velocity_field_unaries([static_pack] + candidate_packs, fields, huber_delta=float(args.rigid_solver_huber_delta))
        final_field_energy = field_energy
        field_normalized = normalize_unary_families({"field": {"values": field_energy["unaries"], "valid": torch.isfinite(field_energy["unaries"])}})["families"]["field"]
        endpoint_result = None
        endpoint_normalized = None
        endpoint_velocity_normalized = None
        endpoint_valid = None
        endpoint_extra_metrics = {}
        if endpoint_transport is not None:
            predicted_endpoint = integrate_endpoint_hypotheses(points, fields, frame_times)
            endpoint_result = endpoint_transport.evaluate(
                points,
                predicted_endpoint,
                canonical_sam3=sam3_features if bool(getattr(args, "use_sam3_feature_field", False)) else None,
            )
            endpoint_valid = endpoint_result.valid.cpu()
            endpoint_normalized = normalize_unary_families(
                {"endpoint": {"values": endpoint_result.energy.cpu(), "valid": endpoint_valid}}
            )["families"]["endpoint"]
            endpoint_residual = torch.linalg.norm(
                endpoint_result.barycentric_target.cpu() - predicted_endpoint.detach().cpu(),
                dim=-1,
            )
            endpoint_velocity_normalized = normalize_unary_families(
                {"endpoint_velocity": {"values": endpoint_residual, "valid": endpoint_valid}}
            )["families"]["endpoint_velocity"]
            write_endpoint_transport_diagnostics(
                endpoint_result,
                points,
                predicted_endpoint,
                endpoint_debug,
                cycle + 1,
                max_points=int(args.transported_html_max_points),
            )
            best_moving_energy = (
                endpoint_result.energy[:, 1:].amin(-1).detach().cpu()
                if endpoint_result.energy.shape[1] > 1
                else endpoint_result.energy[:, 0].detach().cpu()
            )
            endpoint_extra_metrics = {
                "endpoint static energy": endpoint_result.energy[:, 0].detach().cpu(),
                "endpoint best moving energy": best_moving_energy,
                "endpoint static minus best moving": endpoint_result.energy[:, 0].detach().cpu() - best_moving_energy,
                "endpoint confidence max": endpoint_result.confidence.max(-1).values.detach().cpu(),
                "endpoint entropy min": endpoint_result.entropy.min(-1).values.detach().cpu(),
            }
        temperature = float(args.kpart_soft_temperature)
        if discovery_cycles > 1:
            fraction = min(cycle, discovery_cycles - 1) / float(discovery_cycles - 1)
            if sharpened:
                # Convergence already fired once, so annealing has nothing left
                # to contribute; hold the final temperature and let the patience
                # counter close the loop instead of burning the cycle budget.
                fraction = 1.0
            temperature = float(args.kpart_temperature_start) * (1.0 - fraction) + float(args.kpart_soft_temperature) * fraction

        boundary = compute_boundary_ambiguity_mask(
            responsibilities,
            confidence_threshold=float(args.kpart_confidence_threshold),
            margin_threshold=float(args.kpart_soft_margin_threshold),
        )
        components = extract_uncertain_components(graph["pairs"], same_material["affinity"], boundary["ambiguous_mask"].cpu(), min_component_size=2)
        influence = static_pack["confidence"].float().mean(0)
        components = rank_components_for_rendering(
            components,
            boundary["entropy"].cpu(),
            image_influence=influence,
            recent_change=(responsibilities.argmax(-1) != previous_responsibilities.argmax(-1)).float().cpu(),
            max_components=int(args.kpart_render_max_components_per_cycle),
        )
        cycle_render = []
        if bool(args.kpart_counterfactual_render) and components:
            persistent, cycle_render = _evaluate_render_components(
                components, responsibilities, fields, points, rotations, frame_times,
                camera_maps, gaussians, pipe, dataset, args, persistent, cycle + 1,
            )
            render_summaries.extend(cycle_render)
        render_unary = persistent["evidence"].clone()
        render_valid = persistent["valid"] & torch.isfinite(render_unary)
        render_normalized = normalize_unary_families({"render": {"values": render_unary, "valid": render_valid}})["families"]["render"]
        unary = field_normalized
        if render_denoise_mode:
            unary = float(args.lambda_velocity_label) * unary
            unary = unary + float(args.lambda_render_label) * _render_label_unary(
                render_state,
                points.shape[0],
                k,
            ).to(unary)
        if endpoint_normalized is not None and float(args.endpoint_unary_weight) > 0.0:
            unary = unary + float(args.endpoint_unary_weight) * torch.where(
                endpoint_valid,
                endpoint_normalized,
                torch.zeros_like(endpoint_normalized),
            )
        if endpoint_velocity_normalized is not None and float(args.endpoint_velocity_weight) > 0.0:
            unary = unary + float(args.endpoint_velocity_weight) * torch.where(
                endpoint_valid,
                endpoint_velocity_normalized,
                torch.zeros_like(endpoint_velocity_normalized),
            )
        unary = unary + float(args.kpart_render_weight) * torch.where(render_valid, render_normalized, torch.zeros_like(render_normalized))
        unary = _mask_unary_with_stage1_domain(unary, initialization)
        render_changed_unary_winner = int((unary.argmin(-1) != field_normalized.argmin(-1)).sum())
        render_mean = float(render_unary[render_valid].mean()) if bool(render_valid.any()) else 0.0
        render_change = float("inf") if previous_render is None else abs(render_mean - previous_render) / max(abs(previous_render), 1.0e-8)
        previous_render = render_mean

        hard_geometry = responsibilities.argmax(-1).detach().cpu()
        seam_frame = 1 + (cycle % max(1, len(frame_times) - 1))
        common = sorted(set.intersection(*(set(mapping) for mapping in camera_maps)))
        seam_camera = camera_maps[seam_frame][common[cycle % len(common)]] if common else None
        target_dino_boundary = None
        seam_family = None
        seam_xyz = seam_rotation = None
        if seam_camera is not None and has_dino_features and float(args.kpart_dino_2d_seam_weight) > 0:
            from utils.loss_utils import _load_dino_target_for_view

            target_dino = _load_dino_target_for_view(seam_camera, dataset, args, seam_camera.original_image.shape[-2:], device=points.device)
            if target_dino is not None:
                target_dino_boundary = compute_image_dino_boundary(target_dino)
                seam_family = "dino"
                seam_xyz, seam_rotation = _scene_state(points, rotations, hard_geometry, fields, frame_times, seam_frame)
        elif seam_camera is not None and has_sam3_features and float(args.kpart_sam3_2d_seam_weight) > 0:
            from utils.loss_utils import _load_sam3_target_for_view

            target_sam3 = _load_sam3_target_for_view(
                seam_camera,
                dataset,
                args,
                seam_camera.original_image.shape[-2:],
                device=points.device,
            )
            if target_sam3 is not None:
                target_dino_boundary = compute_image_dino_boundary(target_sam3)
                seam_family = "sam3"
                seam_xyz, seam_rotation = _scene_state(points, rotations, hard_geometry, fields, frame_times, seam_frame)

        def seam_loss_fn(value):
            if target_dino_boundary is None:
                return value.sum() * 0.0
            rendered = render_soft_part_probability_maps(
                seam_camera, gaussians, pipe, value,
                positions=seam_xyz, orientations=seam_rotation,
            )
            predicted = compute_rendered_part_boundary(rendered["probability_maps"])
            return compute_one_sided_dino_seam_penalty(predicted, target_dino_boundary)

        with torch.enable_grad():
            optimized = optimize_responsibility_logits(
                unary.to(points.device),
                initial_responsibilities=responsibilities,
                temperature=temperature,
                pairs=graph["pairs"],
                same_material_affinity=same_material["affinity"],
                dino_3d_weight=float(args.kpart_dino_3d_weight) if has_dino_features and not render_denoise_mode else 0.0,
                sam3_pairs=None if sam3_graph is None else sam3_graph["pairs"],
                sam3_affinity=None if sam3_same_material is None else sam3_same_material["affinity"],
                sam3_3d_weight=float(args.kpart_sam3_3d_weight) if has_sam3_features else 0.0,
                seam_loss_fn=seam_loss_fn,
                dino_2d_seam_weight=0.0 if render_denoise_mode else float(args.kpart_dino_2d_seam_weight),
                sam3_2d_seam_weight=float(args.kpart_sam3_2d_seam_weight) if has_sam3_features else 0.0,
                steps=int(args.kpart_soft_logit_steps),
            )
        responsibilities = optimized["responsibilities"].to(points.device)
        responsibilities = _apply_stage1_domain_constraints(responsibilities, initialization)
        gaussians.material_assignment.logits.copy_(torch.log(responsibilities.clamp_min(1.0e-8)))
        if target_dino_boundary is not None:
            rendered_soft = render_soft_part_probability_maps(
                seam_camera, gaussians, pipe, responsibilities,
                positions=seam_xyz, orientations=seam_rotation,
            )
            predicted_boundary = compute_rendered_part_boundary(rendered_soft["probability_maps"])
            last_seam_artifact = {
                "frame": seam_frame,
                "camera": str(common[cycle % len(common)]),
                "feature_family": seam_family,
                "part_probability_maps": rendered_soft["probability_maps"].detach().cpu(),
                "predicted_part_boundary": predicted_boundary.detach().cpu(),
                "feature_boundary": target_dino_boundary.detach().cpu(),
                "dino_boundary": target_dino_boundary.detach().cpu(),
                "one_sided_seam_loss": float(compute_one_sided_dino_seam_penalty(predicted_boundary, target_dino_boundary)),
            }
            seam_prefix = "phase1_sam3_rendered_seam" if seam_family == "sam3" else "phase1_dino_rendered_seam"
            _atomic_torch_save(
                {"schema_version": 1, **last_seam_artifact},
                output / f"{seam_prefix}_cycle_{cycle + 1:03d}.pt",
            )
            _write_seam_html(
                output / f"{seam_prefix}_cycle_{cycle + 1:03d}.html",
                rendered_soft["probability_maps"], predicted_boundary, target_dino_boundary,
                title=f"Stage 1 rendered {str(seam_family).upper()} seam cycle {cycle + 1}",
            )
        final_boundary = compute_boundary_ambiguity_mask(
            responsibilities,
            confidence_threshold=float(args.kpart_confidence_threshold),
            margin_threshold=float(args.kpart_soft_margin_threshold),
        )
        labels = responsibilities.argmax(-1)
        forced_static = ((~initialization["active_mask"]) | initialization["static_anchor"]).to(labels.device)
        dynamic_anchor = initialization["dynamic_anchor"].to(labels.device)
        labels[forced_static] = 0
        if responsibilities.shape[1] == 2:
            labels[dynamic_anchor] = 1
        final_boundary["ambiguous_mask"][forced_static | dynamic_anchor] = False
        labels[final_boundary["ambiguous_mask"]] = -1
        responsibility_change = float(torch.linalg.norm(responsibilities - previous_responsibilities, dim=-1).mean())
        field_change = max(part_field_changes)
        cycle_stats = {
            "cycle": cycle + 1,
            "completion_pass": max(0, cycle - discovery_cycles + 1),
            "temperature": temperature,
            "responsibility_change_mean": responsibility_change,
            "responsibility_change_max": float(torch.linalg.norm(responsibilities - previous_responsibilities, dim=-1).max()),
            "field_change": field_change,
            "render_energy_change": render_change,
            "responsibility_mass": responsibilities.sum(0).detach().cpu().tolist(),
            "high_confidence_interior_count": [int(((labels == label) & ~final_boundary["ambiguous_mask"]).sum()) for label in range(k + 1)],
            "ambiguous_count": int(final_boundary["ambiguous_mask"].sum()),
            "field_residual": [float(field["cost"]) for field in fields],
            "counterfactual_components_evaluated": len(cycle_render),
            "counterfactual_component_sizes": [int(item["gaussian_count"]) for item in cycle_render],
            "responsibilities_changed_by_render_evidence": render_changed_unary_winner,
            "render_evidence_invalidated_labels": invalidated_render_labels,
            "endpoint_transport_enabled": endpoint_result is not None,
            "endpoint_confident_entries": int(endpoint_valid.sum()) if endpoint_valid is not None else 0,
            "endpoint_confident_gaussians": int(endpoint_valid.any(-1).sum()) if endpoint_valid is not None else 0,
            "endpoint_energy_mean": float(endpoint_result.energy[endpoint_valid.to(endpoint_result.energy.device)].mean()) if endpoint_result is not None and bool(endpoint_valid.any()) else 0.0,
            "endpoint_delta_mean": float(endpoint_result.delta_vs_static[endpoint_valid.to(endpoint_result.energy.device)].mean()) if endpoint_result is not None and bool(endpoint_valid.any()) else 0.0,
            **optimized["diagnostics"],
        }
        history.append(cycle_stats)
        responsibility_history.append(responsibilities.detach().cpu())
        _atomic_torch_save({"schema_version": 1, "responsibilities": responsibilities.detach().cpu(), "labels": labels.detach().cpu(), "ambiguity": {key: value.detach().cpu() for key, value in final_boundary.items()}, "unaries": unary, "diagnostics": cycle_stats}, output / f"phase1_velocity_field_soft_cycle_{cycle + 1:03d}.pt")
        classification = {"labels": labels, **final_boundary}
        _write_cycle_html(
            points, responsibilities, classification,
            output / f"phase1_kpart_cycle_{cycle + 1:03d}.html",
            title=f"Velocity-field K-part responsibilities cycle {cycle + 1}",
            max_points=int(args.transported_html_max_points),
            energies=unary,
            dino_graph={"pairs": graph["pairs"], "strong_affinity": same_material["affinity"] >= 0.5},
            extra_metrics=endpoint_extra_metrics,
        )
        if cycle == 0:
            if has_dino_features:
                ablation_labels["dino_3d"] = labels.detach().cpu()
            if has_sam3_features:
                ablation_labels["sam3_3d"] = labels.detach().cpu()
            if endpoint_result is not None:
                ablation_labels["endpoint_transport"] = labels.detach().cpu()
        if target_dino_boundary is not None:
            ablation_labels["dino_2d_seam"] = labels.detach().cpu()
        if cycle_render:
            ablation_labels["render"] = labels.detach().cpu()
        if cycle >= discovery_cycles:
            ablation_labels["final_completion"] = labels.detach().cpu()
        if log_fn is not None:
            log = {
                "part_discovery/velocity_field/responsibility_change": responsibility_change,
                "part_discovery/velocity_field/field_change": field_change,
                "part_discovery/velocity_field/render_change": render_change,
                "part_discovery/velocity_field/ambiguous_count": int(final_boundary["ambiguous_mask"].sum()),
                "part_discovery/velocity_field/dino_3d": optimized["diagnostics"]["dino_3d"],
                "part_discovery/velocity_field/sam3_3d": optimized["diagnostics"].get("sam3_3d", 0.0),
                "part_discovery/velocity_field/dino_2d_seam": optimized["diagnostics"]["dino_2d_seam"],
                "part_discovery/velocity_field/sam3_2d_seam": optimized["diagnostics"].get("sam3_2d_seam", 0.0),
                "part_discovery/render/components_tested": len(cycle_render),
            }
            if endpoint_result is not None:
                log.update(
                    {
                        "part_discovery/endpoint/confident_entries": int(endpoint_valid.sum()),
                        "part_discovery/endpoint/confident_gaussians": int(endpoint_valid.any(-1).sum()),
                        "part_discovery/endpoint/energy_mean": cycle_stats["endpoint_energy_mean"],
                        "part_discovery/endpoint/delta_mean": cycle_stats["endpoint_delta_mean"],
                    }
                )
            for part, field in enumerate(fields, start=1):
                log[f"part_discovery/velocity_field/part_{part}_residual"] = float(field["cost"])
                log[f"part_discovery/velocity_field/part_{part}_mass"] = float(responsibilities[:, part].sum())
            log_fn("kpart_velocity_field", cycle + 1, log)
        progress.set_postfix(dR=f"{responsibility_change:.3g}", dfield=f"{field_change:.3g}", entropy=f"{float(final_boundary['entropy'].mean()):.3f}", ambiguous=int(final_boundary["ambiguous_mask"].sum()))
        converged = responsibility_change < float(args.kpart_soft_convergence_tolerance) and field_change < float(args.kpart_soft_convergence_tolerance) and render_change < 5.0 * float(args.kpart_soft_convergence_tolerance)
        if converged and not sharpened:
            # First convergence: finish annealing immediately rather than after
            # the fixed cycle budget, then require patience more converged
            # cycles at the final temperature before exiting.
            sharpened = True
            converged_at_cycle = cycle + 1
            stable = 0
        elif converged:
            stable += 1
        else:
            stable = 0
        patience = max(1, int(getattr(args, "kpart_convergence_patience", 2)))
        if sharpened and stable >= patience and cycle + 1 >= int(args.kpart_final_completion_passes):
            break

    # Required final confidence-weighted field refit; labels remain soft.
    final_fields = []
    hard_pooling = responsibilities.argmax(-1).detach().cpu()
    for part in range(k):
        pack = _build_observation_pack(
            points, rotations, frame_times, camera_maps, cache, gaussians, pipe, args,
            support_kind=support_kind, raw_cell_ids=raw_cell_ids,
            material_labels=hard_pooling,
            mode=fields[part]["mode"], temporal_speed=fields[part]["temporal_speed"],
            progress_description=f"Final part {part + 1} field observations",
        )
        fitted = fit_region_rigid_field(
            pack, responsibilities[:, part + 1].cpu(),
            responsibility_power=float(args.kpart_field_fit_responsibility_power),
            reference_point=fields[part]["mode"].reference_point,
            solver_kwargs=solver_kwargs,
        )
        final_fields.append(fitted if fitted["valid"] else fields[part])
    fields = final_fields
    # Measured on canonical positions with the final labelling, so it reports
    # the raw separability of the observations rather than how well one
    # hypothesis replays them.
    snr_pack = _build_observation_pack(
        points, rotations, frame_times, camera_maps, cache, gaussians, pipe, args,
        support_kind=support_kind, raw_cell_ids=raw_cell_ids,
        material_labels=hard_pooling,
        progress_description="Observed-velocity SNR observations",
    )
    raw_free_tracks = _raw_free_tracks_from_observed_velocity(points, frame_times, snr_pack)
    _atomic_torch_save(raw_free_tracks, output / "raw_free_gaussian_tracks.pt")
    if render_denoise_mode:
        from scene.render_guided_motion_probe import run_static_rigid_gate_probe

        with torch.enable_grad():
            gate_payload = run_static_rigid_gate_probe(
                scene,
                gaussians,
                pipe,
                dataset,
                args,
                fields,
                frame_times,
                log_fn=log_fn,
            )
        if gate_payload is not None:
            gate_state = gate_payload["per_gaussian"]
            gate_probability = torch.as_tensor(
                gate_state["final_dynamic_probability"],
                device=responsibilities.device,
                dtype=responsibilities.dtype,
            ).clamp(0.0, 1.0)
            gate_confidence = torch.as_tensor(
                gate_state["confidence"],
                device=responsibilities.device,
                dtype=responsibilities.dtype,
            ).clamp(0.0, 1.0)
            gate_unknown = torch.as_tensor(
                gate_state["unknown_score"],
                device=responsibilities.device,
                dtype=responsibilities.dtype,
            ).clamp(0.0, 1.0)
            moving_mass = responsibilities[:, 1:].sum(-1, keepdim=True).clamp_min(1.0e-8)
            moving_distribution = responsibilities[:, 1:] / moving_mass
            gate_hint = torch.zeros_like(responsibilities)
            gate_hint[:, 0] = 1.0 - gate_probability
            gate_hint[:, 1:] = gate_probability[:, None] * moving_distribution
            blend = (gate_confidence * (1.0 - gate_unknown)).clamp(0.0, 1.0)[:, None]
            responsibilities = (1.0 - blend) * responsibilities + blend * gate_hint
            responsibilities = _apply_stage1_domain_constraints(responsibilities, initialization)
            responsibilities /= responsibilities.sum(-1, keepdim=True).clamp_min(1.0e-8)
            stored_gate_state = dict(render_state)
            stored_gate_state.update(gate_state)
            gaussians.set_render_discovery_state(stored_gate_state, args)
            gaussians.material_assignment.logits.copy_(torch.log(responsibilities.clamp_min(1.0e-8)))
            ablation_labels["render_static_rigid_gate"] = responsibilities.argmax(-1).detach().cpu()
    final_boundary = compute_boundary_ambiguity_mask(responsibilities, confidence_threshold=float(args.kpart_confidence_threshold), margin_threshold=float(args.kpart_soft_margin_threshold))
    labels = responsibilities.argmax(-1)
    forced_static = ((~initialization["active_mask"]) | initialization["static_anchor"]).to(labels.device)
    dynamic_anchor = initialization["dynamic_anchor"].to(labels.device)
    labels[forced_static] = 0
    if responsibilities.shape[1] == 2:
        labels[dynamic_anchor] = 1
    final_boundary["ambiguous_mask"][forced_static | dynamic_anchor] = False
    labels[final_boundary["ambiguous_mask"]] = -1
    classification = {"labels": labels, **final_boundary}
    coordinates = [_coordinates(field["temporal_speed"], frame_times) for field in fields]
    modes = [field["mode"] for field in fields]
    speeds = [field["temporal_speed"] for field in fields]
    ablation_labels.setdefault("final_completion", labels.detach().cpu())

    velocity_snr = observed_velocity_snr(snr_pack, labels.detach().cpu())
    snr_floor = float(getattr(args, "kpart_min_velocity_snr", 0.0))
    velocity_snr["min_required"] = snr_floor
    velocity_snr["passed"] = bool(
        snr_floor <= 0.0
        or not math.isfinite(velocity_snr["snr"])
        or velocity_snr["snr"] >= snr_floor
    )
    _atomic_torch_save(
        {"schema_version": 1, **velocity_snr, "ground_truth_used_as_algorithm_input": False},
        output / "phase1_observed_velocity_snr.pt",
    )
    message = (
        f"[Stage 1 velocity fields] observed-velocity SNR {velocity_snr['snr']:.2f} "
        f"(moving {velocity_snr['moving_mean_speed']:.4g} / static "
        f"{velocity_snr['static_mean_speed']:.4g}, "
        f"{100.0 * velocity_snr['observation_valid_fraction']:.1f}% of observations accepted)"
    )
    if velocity_snr["passed"]:
        print(message)
    else:
        detail = (
            f"{message} is below the --kpart_min_velocity_snr floor of {snr_floor:.2f}. "
            "The rasterized flow cannot separate this part from its surroundings, so the "
            "assignment, the scalar speeds and any joint fitted from them are unreliable."
        )
        if bool(getattr(args, "kpart_require_velocity_snr", False)):
            raise RuntimeError(detail)
        print(f"[Stage 1 velocity fields] WARNING: {detail}")

    _atomic_torch_save({"schema_version": 1, "history": responsibility_history}, output / "phase1_kpart_responsibility_history.pt")
    _atomic_torch_save({"schema_version": 1, "responsibilities": responsibilities.detach().cpu(), "ground_truth_used_as_algorithm_input": False}, output / "phase1_kpart_responsibility_final.pt")
    _atomic_torch_save({"schema_version": 1, "part_labels": labels.detach().cpu(), "confidence": final_boundary["confidence"].cpu(), "ambiguous_mask": final_boundary["ambiguous_mask"].cpu(), "entropy": final_boundary["entropy"].cpu(), "ground_truth_used_as_algorithm_input": False}, output / "phase1_kpart_assignment_final.pt")
    _atomic_torch_save(
        {
            "schema_version": 3,
            "source_of_truth": "rigid_velocity_field",
            # Both joint hypotheses are fitted on the same observations and
            # weights, so Stage 2 can select the one that explains the velocity
            # rather than deciding from a threshold on a derived ratio.
            "fields": [
                {
                    "spatial_mode": field["mode"].as_dict(),
                    "temporal_speed": field["temporal_speed"],
                    "field_residual": field["cost"],
                    "translation_spatial_mode": (
                        None
                        if field.get("translation_mode") is None
                        else field["translation_mode"].as_dict()
                    ),
                    "translation_temporal_speed": field.get("translation_speed"),
                    "translation_residual": field.get("translation_cost"),
                    "translation_valid": field.get("translation_valid", False),
                }
                for field in fields
            ],
        },
        output / "phase1_kpart_rigid_velocity_fields.pt",
    )
    _atomic_torch_save({"schema_version": 1, "modes": [mode.as_dict() for mode in modes]}, output / "phase1_kpart_coarse_rigid_modes.pt")
    _atomic_torch_save({"schema_version": 2, "source_of_truth": "derived_from_rigid_velocity_fields", "frame_times": torch.as_tensor(frame_times), "coordinates": torch.stack(coordinates), "interval_speeds": torch.stack([torch.as_tensor(item) for item in speeds])}, output / "phase1_kpart_coarse_trajectories.pt")
    _atomic_torch_save({"schema_version": 1, **final_field_energy}, output / "phase1_static_vs_part_energies.pt")
    _atomic_torch_save({"schema_version": 2, "labels": ablation_labels, "ground_truth_used_as_algorithm_input": False}, output / "phase1_kpart_ablation_labels.pt")
    _atomic_torch_save({"schema_version": 1, **persistent}, output / "phase1_counterfactual_render_persistent_evidence.pt")
    if last_seam_artifact is not None:
        seam_prefix = (
            "phase1_sam3_rendered_seam.pt"
            if last_seam_artifact.get("feature_family") == "sam3"
            else "phase1_dino_rendered_seam.pt"
        )
        _atomic_torch_save({"schema_version": 1, **last_seam_artifact}, output / seam_prefix)
    (output / "phase1_counterfactual_render_summary.json").write_text(json.dumps(_strict_json({"schema_version": 2, "persistent": True, "components": render_summaries}), indent=2, allow_nan=False), encoding="utf-8")
    summary = {
        "schema_version": 3,
        "architecture": "prior_constrained_velocity_field_kpart",
        "source_of_truth": "rigid_velocity_field",
        "kprime_discovery_removed": True,
        "initialization": "exact_k_prior_probability_clusters_plus_static",
        "initial_kpart_cluster_sizes": initialization["cluster_sizes"],
        "prior_active_count": int(initialization["active_mask"].sum()),
        "static_leftover_frozen": True,
        "phase1_speed_role": "coarse_physical_consistency_only_not_metric_displacement",
        "phase2_joint_type_candidates": ["revolute", "prismatic"],
        "phase3_metric_displacement_source": "joint_trajectory_optimization",
        "dynamic_discovery_mode": "render_denoise" if render_denoise_mode else "legacy",
        "support": support_kind,
        "observation_backend_only_difference": True,
        "num_moving_parts": k,
        "initial_supernodes": k,
        "gaussian_count": int(points.shape[0]),
        "cycles_completed": len(history),
        "cycles_budgeted": total_cycles,
        "converged_at_cycle": converged_at_cycle,
        "converged": bool(converged_at_cycle is not None),
        "convergence_patience": int(getattr(args, "kpart_convergence_patience", 2)),
        "observed_velocity_snr": velocity_snr,
        "class_counts": [int((labels == label).sum()) for label in range(k + 1)],
        "ambiguous_count": int((labels < 0).sum()),
        "static_is_zero_field": True,
        "raw_speed_primary_unary": False,
        "independent_transform_optimization": False,
        "persistent_render_evidence": True,
        "render_label_weight": float(getattr(args, "lambda_render_label", 0.0)) if render_denoise_mode else 0.0,
        "velocity_label_weight": float(getattr(args, "lambda_velocity_label", 1.0)),
        "dino_used_for_static_dynamic_discovery": False,
        "sam3_used_for_static_dynamic_discovery": False,
        "dino_used_for_kpart_regularization": bool(
            has_dino_features
            and (
                float(getattr(args, "kpart_dino_3d_weight", 0.0)) > 0.0
                or float(getattr(args, "kpart_dino_2d_seam_weight", 0.0)) > 0.0
            )
        ),
        "sam3_used_for_kpart_regularization": bool(
            has_sam3_features
            and (
                float(getattr(args, "kpart_sam3_3d_weight", 0.0)) > 0.0
                or float(getattr(args, "kpart_sam3_2d_seam_weight", 0.0)) > 0.0
            )
        ),
        "endpoint_transport_enabled": endpoint_transport is not None,
        "endpoint_frame": int(getattr(endpoint_transport, "frame", -1)) if endpoint_transport is not None else None,
        "endpoint_model_path": getattr(endpoint_transport, "target_path", "") if endpoint_transport is not None else "",
        "endpoint_transport_mode": str(getattr(args, "endpoint_transport_mode", "local_soft")),
        "endpoint_sam3_enabled": bool(getattr(args, "use_sam3_feature_field", False) and float(getattr(args, "endpoint_sam3_weight", 0.0)) > 0.0),
        "endpoint_unary_weight": float(getattr(args, "endpoint_unary_weight", 0.0)),
        "endpoint_velocity_weight": float(getattr(args, "endpoint_velocity_weight", 0.0)),
        "endpoint_ground_truth_used": False,
        "endpoint_velocity_updates_physical_speeds": False,
        "dynamic_prior_bootstrap_used": bool(prior_bootstrap_enabled),
        "dynamic_prior_method": None if prior_bootstrap is None else prior_bootstrap["method"],
        "dynamic_prior_weight": float(getattr(args, "kpart_dynamic_prior_weight", 1.0)),
        "prior_static_anchor_count": 0 if prior_anchors is None else int(prior_anchors["static_anchor"].sum()),
        "prior_dynamic_anchor_count": 0 if prior_anchors is None else int(prior_anchors["dynamic_anchor"].sum()),
        "prior_kpart_active_count": 0 if prior_anchors is None else int(prior_anchors["active_mask"].sum()),
        "prior_boundary_expansion": int(getattr(args, "kpart_prior_boundary_expansion", 0)),
        "prior_cleanup_min_size": int(getattr(args, "kpart_prior_cleanup_min_size", 16)),
        "sam3_3d_weight": float(getattr(args, "kpart_sam3_3d_weight", 0.0)) if has_sam3_features else 0.0,
        "sam3_2d_seam_weight": float(getattr(args, "kpart_sam3_2d_seam_weight", 0.0)) if has_sam3_features else 0.0,
        "history": history,
        "ground_truth_used_as_algorithm_input": False,
    }
    (output / "phase1_kpart_summary.json").write_text(json.dumps(_strict_json(summary), indent=2, allow_nan=False), encoding="utf-8")

    coarse_tracks = _build_coarse_tracks(points, rotations, modes, coordinates, labels, frame_times)
    coarse_tracks.update({"schema_version": 2, "source_of_truth": "integrated_rigid_velocity_fields", "motion_membership": responsibilities[:, 1:].sum(-1).cpu(), "part_probabilities": responsibilities.cpu(), "fixed_cameras": _fixed_camera_metadata(cameras)})
    _atomic_torch_save(coarse_tracks, output / "phase1_kpart_coarse_gaussian_tracks.pt")
    write_transported_gaussians_html(coarse_tracks, output / "phase1_kpart_coarse_motion.html", title=f"Stage 1 velocity-field {support_kind} K-part motion", max_points=int(args.transported_html_max_points), max_vectors=min(250, int(args.transported_html_max_points)))
    write_quaternion_transport_audit(coarse_tracks, output, scales=gaussians.get_scaling.detach().cpu(), max_debug_gaussians=min(64, int(args.transported_html_max_points)))
    return {
        "modes": modes,
        "interval_speeds": speeds,
        "coordinates": coordinates,
        "frame_times": frame_times,
        "frame_ids": [int(item[0]) for item in groups],
        "responsibilities": responsibilities,
        "classification": classification,
        "labels": labels,
        "coarse_tracks": coarse_tracks,
        "raw_tracks": {
            "times": raw_free_tracks["times"],
            "positions": raw_free_tracks["positions"],
            "canonical_xyz": raw_free_tracks["canonical_xyz"],
            "source_of_truth": raw_free_tracks["source_of_truth"],
        },
        "raw_velocities": raw_free_tracks["velocities"],
        "observed_velocities": raw_free_tracks["observed_velocities"],
        "observed_velocity_confidence": raw_free_tracks["confidence"],
        "observed_velocity_valid": raw_free_tracks["valid"],
        "history": history,
        "ablation_labels": ablation_labels,
        "velocity_fields": fields,
        "dynamic_prior_bootstrap": prior_bootstrap,
    }
