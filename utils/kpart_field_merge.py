"""Rigid-field fitting and mandatory shared-field provisional-region merges."""

from __future__ import annotations

import math

import torch

from utils.rigid_killing_motion import (
    predict_rigid_cell_velocity,
    solve_rigid_killing_mode_all_intervals,
    solve_translation_mode_all_intervals,
)


def _observation_tensors(observations):
    positions = torch.as_tensor(observations["positions"]).detach().cpu().double()
    velocity = torch.as_tensor(observations["velocity"]).detach().cpu().double()
    confidence = torch.as_tensor(observations.get("confidence", torch.ones(velocity.shape[:2]))).detach().cpu().double()
    valid = torch.as_tensor(observations.get("valid", torch.ones(velocity.shape[:2], dtype=torch.bool))).detach().cpu().bool()
    if positions.ndim == 2:
        positions = positions[None].expand(velocity.shape[0], -1, -1).clone()
    if positions.shape != velocity.shape or confidence.shape != velocity.shape[:2] or valid.shape != velocity.shape[:2]:
        raise ValueError("observations require positions/velocity [T,N,3] and confidence/valid [T,N]")
    return positions, velocity, confidence, valid


#: Free parameters in each spatial generator, after its normalisation.  The
#: rigid generator is a 6-vector (three linear, three angular) put in unit
#: gauge, leaving five; the translation generator is a 3-vector direction of
#: unit norm, leaving two.  Both carry the same one speed per interval, so those
#: cancel in the difference and only the spatial freedom is charged.
RIGID_GENERATOR_COMPONENTS = 6
TRANSLATION_GENERATOR_COMPONENTS = 3
UNIT_GAUGE_CONSTRAINTS = 1
RIGID_GENERATOR_DOF = RIGID_GENERATOR_COMPONENTS - UNIT_GAUGE_CONSTRAINTS
TRANSLATION_GENERATOR_DOF = TRANSLATION_GENERATOR_COMPONENTS - UNIT_GAUGE_CONSTRAINTS
EXTRA_RIGID_DOF = RIGID_GENERATOR_DOF - TRANSLATION_GENERATOR_DOF


def compare_rigid_and_translation_evidence(
    rigid_cost, translation_cost, observation_count, residual_dimension
):
    """Which motion model the observed velocity supports, penalised for freedom.

    The rigid family contains the translation family -- set ``omega = 0`` and one
    becomes the other -- so its residual can never be larger, and comparing the
    two residuals directly always favours rotation.  With noisy observations
    that bias alone flips the answer: on synthetic data a *pure* translation
    observed with 5% noise still gave the rigid fit a 0.03% edge.

    The correction is the standard nested-model one.  Under Gaussian residuals
    the log-likelihood ratio of the two fits is ``n log(SSE_t / SSE_r)`` over
    ``n`` scalar residuals, and the Bayesian information criterion charges each
    extra degree of freedom ``log n``:

        delta = n log(cost_translation / cost_rigid) - EXTRA_RIGID_DOF log(n)

    Positive supports revolute, negative prismatic, and ``exp(delta / 2)`` are
    the posterior odds under equal priors.  Every term comes from the data or
    from the two models' own structure: the two residuals, the number of scalar
    residuals behind them, and the parameter counts declared above.  Nothing is
    tuned and nothing is clamped.

    The limits are exact rather than special-cased.  Both residuals zero means
    the two models are indistinguishable on this data, so the likelihood ratio
    is one and only the penalty remains -- the simpler model wins, which is the
    correct nested-model answer.  A zero rigid residual against a positive
    translation residual is unbounded evidence for rotation.  No observations at
    all is not a decision: ``joint_type`` is ``None`` and the caller must say
    what it does instead.

    ``n`` counts correlated observations, so it overstates the independent
    sample size and the magnitude reads as an ordering rather than a calibrated
    probability.  The sign has held with orders of magnitude to spare on every
    synthetic case tried, at noise levels up to 20%.
    """
    rigid = max(float(rigid_cost), 0.0)
    translation = max(float(translation_cost), 0.0)
    dimension = int(residual_dimension)
    count = dimension * int(observation_count)
    if count <= 0 or dimension <= 0:
        # No evidence exists.  Report that, rather than inventing a decision.
        return {
            "rigid_residual": rigid,
            "translation_residual": translation,
            "observation_count": int(observation_count),
            "scalar_residual_count": count,
            "extra_rigid_dof": EXTRA_RIGID_DOF,
            "bic_penalty": None,
            "log_likelihood_ratio": None,
            "log_evidence_difference": None,
            "revolute_probability": None,
            "joint_type": None,
            "reason": "no_valid_velocity_observations",
        }
    if rigid <= 0.0 and translation <= 0.0:
        log_ratio = 0.0
    elif rigid <= 0.0:
        log_ratio = math.inf
    elif translation <= 0.0:
        log_ratio = -math.inf
    else:
        log_ratio = math.log(translation / rigid)
    penalty = EXTRA_RIGID_DOF * math.log(float(count))
    log_likelihood_ratio = (
        math.copysign(math.inf, log_ratio) if math.isinf(log_ratio) else count * log_ratio
    )
    difference = log_likelihood_ratio - penalty
    # torch.sigmoid is exact at both infinities, so the odds need no clamp.
    probability = float(torch.sigmoid(torch.tensor(difference / 2.0, dtype=torch.float64)))
    return {
        "rigid_residual": rigid,
        "translation_residual": translation,
        "observation_count": int(observation_count),
        "scalar_residual_count": count,
        "extra_rigid_dof": EXTRA_RIGID_DOF,
        "bic_penalty": penalty,
        "log_likelihood_ratio": log_likelihood_ratio,
        "log_evidence_difference": difference,
        "revolute_probability": probability,
        "joint_type": "revolute" if difference > 0.0 else "prismatic",
        "reason": None,
    }


@torch.no_grad()
def fit_region_rigid_field(
    observations,
    region_weights,
    *,
    responsibility_power=2.0,
    reference_point=None,
    solver_kwargs=None,
):
    """Fit one factorized Killing field using confidence-weighted interiors."""
    positions, velocity, confidence, valid = _observation_tensors(observations)
    membership = torch.as_tensor(region_weights).detach().cpu().double().reshape(-1).clamp(0.0, 1.0)
    if membership.numel() != velocity.shape[1]:
        raise ValueError("region_weights must have one value per Gaussian")
    field_weight = membership.pow(float(responsibility_power))
    weights = confidence * field_weight[None]
    support = (weights * valid.double()).sum(0)
    active = (field_weight > 0) & valid.any(0)
    if int(active.sum()) < int((solver_kwargs or {}).get("min_valid_cells", 8)):
        active = support > 0
    if not bool(active.any()):
        raise RuntimeError("rigid-field region has no weighted valid observations")
    if reference_point is None and bool((support > 0).any()):
        reference_point = positions[0, int(torch.argmax(support))]
    # Every point outside the region has exactly zero fit weight.  Removing
    # those rows before the dense IRLS solve changes no equation and avoids
    # repeating full-scene work for every provisional merge candidate.
    positions = positions[:, active]
    velocity = velocity[:, active]
    confidence = confidence[:, active]
    valid = valid[:, active]
    weights = weights[:, active]
    kwargs = dict(solver_kwargs or {})
    mode, speed, diagnostics = solve_rigid_killing_mode_all_intervals(
        positions,
        velocity,
        weights=weights,
        valid_mask=valid,
        reference_point=reference_point,
        **kwargs,
    )
    predicted = (
        predict_rigid_cell_velocity(mode, positions, speed)
        if mode.valid
        else torch.zeros_like(velocity)
    )
    residual = torch.linalg.norm(velocity - predicted, dim=-1)
    effective = weights * valid.double()
    cost = (effective * residual.square()).sum() / effective.sum().clamp_min(1.0e-12)
    static_cost = (effective[:, :, None] * velocity.square()).sum() / effective.sum().clamp_min(1.0e-12)
    # Fit the prismatic hypothesis on exactly the same observations and weights,
    # so the two joint types are explored together and can be selected on which
    # one actually explains the velocity rather than on a threshold.  The rigid
    # family contains the translation family, so its cost can never be higher;
    # what matters downstream is by how much the extra rotational freedom pays.
    # The same solver settings the rigid fit was given, so the two residuals
    # differ only by the model, never by the numerics.
    translation_kwargs = {
        key: value
        for key, value in kwargs.items()
        if key in ("irls_iters", "huber_delta", "min_valid_cells")
    }
    translation_mode, translation_speed, translation_diagnostics = (
        solve_translation_mode_all_intervals(
            positions, velocity, weights=weights, valid_mask=valid, **translation_kwargs
        )
    )
    translation_predicted = (
        predict_rigid_cell_velocity(translation_mode, positions, translation_speed)
        if translation_mode.valid
        else torch.zeros_like(velocity)
    )
    translation_residual = torch.linalg.norm(velocity - translation_predicted, dim=-1)
    translation_cost = (effective * translation_residual.square()).sum() / effective.sum().clamp_min(1.0e-12)
    comparison = compare_rigid_and_translation_evidence(
        float(cost),
        float(translation_cost),
        int((valid & (weights > 0)).sum()),
        int(velocity.shape[-1]),
    )
    return {
        "mode": mode,
        "temporal_speed": speed,
        "cost": float(cost),
        "static_cost": float(static_cost),
        "translation_mode": translation_mode,
        "translation_speed": translation_speed,
        "translation_cost": float(translation_cost),
        "translation_valid": bool(translation_mode.valid),
        "translation_diagnostics": translation_diagnostics,
        "joint_type_evidence": comparison,
        "weight_sum": float(effective.sum()),
        "residual": residual,
        "valid": bool(mode.valid and torch.isfinite(cost)),
        "diagnostics": diagnostics,
        "region_weights": membership,
        "active_indices": torch.nonzero(active, as_tuple=False).flatten(),
    }


@torch.no_grad()
def evaluate_field_on_probe_positions(field, probe_positions):
    if not bool(field.get("valid", False)):
        raise RuntimeError("cannot evaluate an invalid provisional rigid field")
    return predict_rigid_cell_velocity(
        field["mode"],
        torch.as_tensor(probe_positions).detach().cpu().double(),
        torch.as_tensor(field["temporal_speed"]).detach().cpu().double(),
    )


@torch.no_grad()
def compute_region_field_distance(field_a, field_b, probe_positions, *, probe_weights=None):
    """Gauge-invariant distance obtained by evaluating physical velocity fields."""
    first = evaluate_field_on_probe_positions(field_a, probe_positions)
    second = evaluate_field_on_probe_positions(field_b, probe_positions)
    distance = torch.linalg.norm(first - second, dim=-1)
    if probe_weights is None:
        weight = torch.ones_like(distance)
    else:
        weight = torch.as_tensor(probe_weights).detach().cpu().double()
        if weight.ndim == 1:
            weight = weight[None].expand_as(distance)
    valid = torch.isfinite(distance) & torch.isfinite(weight) & (weight > 0)
    if not bool(valid.any()):
        return float("inf")
    values = distance[valid]
    robust_limit = torch.quantile(values, 0.9)
    robust = distance.clamp_max(robust_limit)
    return float((robust * weight * valid).sum() / (weight * valid).sum().clamp_min(1.0e-12))


@torch.no_grad()
def fit_shared_field_to_union(observations, weights_a, weights_b, **kwargs):
    union = torch.maximum(
        torch.as_tensor(weights_a).detach().cpu().double(),
        torch.as_tensor(weights_b).detach().cpu().double(),
    )
    return fit_region_rigid_field(observations, union, **kwargs)


@torch.no_grad()
def compute_shared_field_merge_penalty(field_a, field_b, shared_field):
    """Cost increase of one mandatory shared-field refit over separate fits."""
    if not bool(shared_field.get("valid", False)):
        return float("inf")
    wa, wb = float(field_a["weight_sum"]), float(field_b["weight_sum"])
    separate = (wa * float(field_a["cost"]) + wb * float(field_b["cost"])) / max(wa + wb, 1.0e-12)
    scale = max(separate, 1.0e-8)
    return (float(shared_field["cost"]) - separate) / scale


@torch.no_grad()
def evaluate_merge_render_delta(region_a, region_b, shared_field, *, render_evaluator=None, context=None):
    if render_evaluator is None:
        return 0.0
    return float(render_evaluator(region_a, region_b, shared_field, context=context))


def select_best_merge(candidates):
    valid = [row for row in candidates if math.isfinite(float(row.get("shared_field_fit_penalty", float("inf"))))]
    if not valid:
        raise RuntimeError("field merge has no candidate supported by a valid shared-field refit")
    return min(valid, key=lambda row: float(row["total_score"]))


def _region_pairs(labels, graph_pairs, points):
    active_labels = sorted(int(item) for item in torch.unique(labels[labels >= 0]).tolist())
    result = set()
    edge = torch.as_tensor(graph_pairs).detach().cpu().long()
    if edge.numel():
        first, second = labels[edge[:, 0]], labels[edge[:, 1]]
        for a, b in torch.stack((first, second), -1).tolist():
            if a >= 0 and b >= 0 and a != b:
                result.add(tuple(sorted((int(a), int(b)))))
    # Disconnected material components still receive a spatial region-neighbor
    # edge; merge selection itself remains exclusively physical-field based.
    if not result and len(active_labels) > 1:
        centers = torch.stack([torch.as_tensor(points)[labels == label].mean(0) for label in active_labels])
        distance = torch.cdist(centers.float(), centers.float())
        distance.fill_diagonal_(float("inf"))
        for index, label in enumerate(active_labels):
            neighbor = active_labels[int(torch.argmin(distance[index]))]
            result.add(tuple(sorted((label, neighbor))))
    return sorted(result)


@torch.no_grad()
def merge_until_k(
    labels,
    observations,
    graph_pairs,
    points,
    *,
    target_k,
    dynamic_likelihood=None,
    dino_distance=None,
    responsibility_power=2.0,
    shared_field_weight=1.0,
    render_weight=0.25,
    dino_weight=0.05,
    render_evaluator=None,
    solver_kwargs=None,
    show_progress=False,
):
    """Agglomerate K' regions to K using mandatory union-field refits."""
    current = torch.as_tensor(labels).detach().cpu().long().clone()
    target = int(target_k)
    graph = torch.as_tensor(graph_pairs).detach().cpu().long()
    dynamic = torch.ones(current.numel(), dtype=torch.float64) if dynamic_likelihood is None else torch.as_tensor(dynamic_likelihood).detach().cpu().double().clamp(0, 1)
    history = []
    label_snapshots = [current.clone()]
    if target == 1:
        merged = torch.full_like(current, -1)
        active = current >= 0
        merged[active] = 0
        final_field = fit_region_rigid_field(
            observations,
            active.double() * dynamic,
            responsibility_power=responsibility_power,
            solver_kwargs=solver_kwargs,
        )
        if not bool(final_field["valid"]):
            raise RuntimeError("field merge did not produce exactly K valid moving rigid velocity fields")
        history.append(
            {
                "operation": "collapse_to_single_target",
                "input_regions": int(torch.unique(current[active]).numel()),
                "target_regions": 1,
                "shared_field_fit_penalty": 0.0,
                "render_merge_delta": 0.0,
                "dino_boundary_score": 0.0,
                "total_score": 0.0,
                "selected": True,
            }
        )
        label_snapshots.append(merged.clone())
        return {"labels": merged, "fields": [final_field], "history": history, "label_snapshots": label_snapshots}
    progress = None
    if bool(show_progress):
        from tqdm import tqdm

        progress = tqdm(
            total=max(0, int(torch.unique(current[current >= 0]).numel()) - target),
            desc="Phase A shared-field merge",
            unit="merge",
        )
    while int(torch.unique(current[current >= 0]).numel()) > target:
        active_labels = sorted(int(item) for item in torch.unique(current[current >= 0]).tolist())
        fields = {}
        region_weights = {}
        for label in active_labels:
            weights = (current == label).double() * dynamic
            region_weights[label] = weights
            fields[label] = fit_region_rigid_field(
                observations,
                weights,
                responsibility_power=responsibility_power,
                solver_kwargs=solver_kwargs,
            )
        candidate_pairs = _region_pairs(current, graph, points)
        candidates = []
        for first, second in candidate_pairs:
            shared = fit_shared_field_to_union(
                observations,
                region_weights[first],
                region_weights[second],
                responsibility_power=responsibility_power,
                solver_kwargs=solver_kwargs,
            )
            penalty = compute_shared_field_merge_penalty(fields[first], fields[second], shared)
            union_mask = (current == first) | (current == second)
            if fields[first]["valid"] and fields[second]["valid"]:
                field_distance = compute_region_field_distance(
                    fields[first], fields[second], torch.as_tensor(observations["positions"])[:, union_mask],
                )
            else:
                field_distance = float("inf")
            boundary = 0.0
            if dino_distance is not None and graph.numel():
                crossing = ((current[graph[:, 0]] == first) & (current[graph[:, 1]] == second)) | ((current[graph[:, 0]] == second) & (current[graph[:, 1]] == first))
                if bool(crossing.any()):
                    boundary = float(torch.as_tensor(dino_distance).detach().cpu().float()[crossing].mean())
            render_delta = 0.0
            total = float(shared_field_weight) * penalty + float(dino_weight) * boundary
            candidates.append({
                "region_a": first,
                "region_b": second,
                "size_a": int((current == first).sum()),
                "size_b": int((current == second).sum()),
                "field_distance": field_distance,
                "shared_field_fit_penalty": penalty,
                "render_merge_delta": render_delta,
                "dino_boundary_score": boundary,
                "total_score": total,
                "selected": False,
                "shared_field": shared,
            })
        if render_evaluator is not None and float(render_weight) > 0:
            promising = sorted(
                (row for row in candidates if math.isfinite(float(row["shared_field_fit_penalty"]))),
                key=lambda row: float(row["shared_field_fit_penalty"]),
            )[: min(2, len(candidates))]
            for row in promising:
                row["render_merge_delta"] = evaluate_merge_render_delta(
                    row["region_a"], row["region_b"], row["shared_field"],
                    render_evaluator=render_evaluator,
                    context={"labels": current, "fields": fields},
                )
        for row in candidates:
            row["total_score"] = (
                float(shared_field_weight) * float(row["shared_field_fit_penalty"])
                + float(render_weight) * float(row["render_merge_delta"])
                + float(dino_weight) * float(row["dino_boundary_score"])
            )
        selected = select_best_merge(candidates)
        selected["selected"] = True
        for row in candidates:
            history.append({key: value for key, value in row.items() if key != "shared_field"})
        keep, remove = int(selected["region_a"]), int(selected["region_b"])
        current[current == remove] = keep
        label_snapshots.append(current.clone())
        if progress is not None:
            progress.update(1)
            progress.set_postfix(
                regions=int(torch.unique(current[current >= 0]).numel()),
                shared=f"{float(selected['shared_field_fit_penalty']):.3g}",
                render=f"{float(selected['render_merge_delta']):.3g}",
            )
    remaining = sorted(int(item) for item in torch.unique(current[current >= 0]).tolist())
    remap = {old: new for new, old in enumerate(remaining)}
    merged = torch.full_like(current, -1)
    for old, new in remap.items():
        merged[current == old] = new
    final_fields = []
    for label in range(target):
        final_fields.append(fit_region_rigid_field(
            observations,
            (merged == label).double() * dynamic,
            responsibility_power=responsibility_power,
            solver_kwargs=solver_kwargs,
        ))
    if len(final_fields) != target or any(not field["valid"] for field in final_fields):
        if progress is not None:
            progress.close()
        raise RuntimeError("field merge did not produce exactly K valid moving rigid velocity fields")
    if progress is not None:
        progress.close()
    label_snapshots.append(merged.clone())
    return {"labels": merged, "fields": final_fields, "history": history, "label_snapshots": label_snapshots}
