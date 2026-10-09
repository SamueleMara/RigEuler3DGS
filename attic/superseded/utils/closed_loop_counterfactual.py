"""Component-level counterfactual rendering and leave-component-out validation.

Spec 28 and the third safeguard of spec 30.  Both ask the same kind of question
-- what happens if this whole group of Gaussians is treated differently -- and
both need the group tested as a unit rather than point by point.

Why this matters for coverage.  Every per-Gaussian energy in the loop needs that
Gaussian to have its own evidence: the multiview velocity reaches 4542 of 11263
foreground Gaussians on the USB scene, and the remaining 4493 moving ones get no
vote at all and fall to static by default.  A *component* does not have that
problem.  Render the whole component static, render it transported by part k,
and compare the images: a group of flow-invisible Gaussians that belongs to the
moving part still changes the picture when it moves, so the component earns an
energy that none of its members could earn alone.

    delta_L(C, k) = L_render(C static) - L_render(C moved by part k)

Positive means moving explains the images better.  The two renders share every
Gaussian outside the component, so the shared part of the image contributes the
same amount to both losses and cancels in the difference; only the component's
own pixels survive it.  That cancellation is why the *difference* is used rather
than either loss on its own -- a large static background makes both losses big
and their difference is still exactly the component's contribution.

Nothing here optimises free per-Gaussian coordinates.  Only the two discrete
hypotheses are ever rendered, which is the invariant of spec 33.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

_EPS = 1.0e-8


@dataclass
class CounterfactualConfig:
    """What to test, and how much rendering to spend on it."""

    max_components: int = 8
    min_component_size: int = 4
    frames_per_component: int = 2
    confidence_threshold: float = 0.8
    margin_threshold: float = 0.15
    #: Momentum on the persistent evidence table, so a component that was
    #: tested three cycles ago still carries what it learned.
    momentum: float = 0.5
    enabled: bool = True


@torch.no_grad()
def ambiguous_components(
    *,
    responsibilities: torch.Tensor,
    knn_idx: torch.Tensor,
    graph_weight: torch.Tensor,
    config: CounterfactualConfig,
    observed_support: torch.Tensor | None = None,
) -> list[torch.Tensor]:
    """Connected groups of Gaussians no label owns confidently.

    Grouping is by spatial adjacency on the foreground KNN, restricted to edges
    the graph considers strong, so a component is a piece of surface rather than
    a scatter of unrelated points.

    ``observed_support`` biases the ordering toward components the velocity
    could *not* explain.  Those are the ones per-Gaussian energies cannot reach,
    so they are exactly where a component-level test earns its cost.
    """
    share = torch.as_tensor(responsibilities)
    ordered = share.sort(-1, descending=True).values
    ambiguous = (ordered[:, 0] < float(config.confidence_threshold)) | (
        (ordered[:, 0] - ordered[:, 1]) < float(config.margin_threshold)
    )
    if not bool(ambiguous.any()):
        return []

    neighbours = knn_idx.to(device=share.device, dtype=torch.long)
    strong = graph_weight.to(share.device) >= graph_weight.to(share.device).median()
    count = int(share.shape[0])
    # Union-find over strong edges between ambiguous points.
    parent = torch.arange(count, device=share.device)

    def find(node: int) -> int:
        while int(parent[node]) != node:
            parent[node] = parent[int(parent[node])]
            node = int(parent[node])
        return node

    rows = torch.nonzero(ambiguous, as_tuple=False).flatten().tolist()
    ambiguous_set = set(rows)
    for row in rows:
        for column in range(neighbours.shape[1]):
            if not bool(strong[row, column]):
                continue
            other = int(neighbours[row, column])
            if other not in ambiguous_set:
                continue
            a, b = find(row), find(other)
            if a != b:
                parent[a] = b

    groups: dict[int, list[int]] = {}
    for row in rows:
        groups.setdefault(find(row), []).append(row)
    components = [
        torch.tensor(members, device=share.device, dtype=torch.long)
        for members in groups.values()
        if len(members) >= int(config.min_component_size)
    ]
    if not components:
        return []

    entropy = -(share * share.clamp_min(_EPS).log()).sum(-1)

    def priority(component: torch.Tensor) -> float:
        score = float(entropy[component].mean()) * float(component.numel()) ** 0.5
        if observed_support is not None:
            unobserved = float((observed_support[component] <= 0).double().mean())
            # A component the velocity never saw is worth more here, because
            # nothing else in the loop can speak for it.
            score = score * (1.0 + unobserved)
        return score

    components.sort(key=priority, reverse=True)
    return components[: int(config.max_components)]


@torch.no_grad()
def counterfactual_component_energy(
    *,
    state,
    components: list[torch.Tensor],
    target_frames,
    render_component_fn,
    config: CounterfactualConfig,
    persistent: dict | None = None,
    cycle: int = 0,
) -> dict:
    """Render each component under every label and turn the result into energy.

    ``render_component_fn(component, label, frame)`` renders the full scene with
    that component forced to that label and returns the scalar image loss.  Only
    differences between labels are ever read, and the shared background cancels
    in those differences, so the absolute value of the loss never matters.  The
    energies are stored per Gaussian
    so they can be summed with the per-point modalities of spec 24, and they
    persist across cycles with momentum: a component tested two cycles ago still
    carries its verdict, because re-rendering every component every cycle is the
    single most expensive thing the loop could do.
    """
    labels = state.num_labels
    count = state.num_foreground
    device = state.canonical_fg.device
    if persistent is None:
        persistent = {
            "energy": torch.full((count, labels), float("nan"), device=device),
            "tested": torch.zeros(count, dtype=torch.bool, device=device),
            "cycle": torch.full((count,), -1, dtype=torch.long, device=device),
        }
    if not config.enabled or not components:
        return {"persistent": persistent, "summaries": [], "energy": persistent["energy"]}

    frames = [int(value) for value in target_frames]
    chosen = frames[: max(1, int(config.frames_per_component))]
    summaries = []
    for index, component in enumerate(components):
        per_label = []
        for label in range(labels):
            total, seen = 0.0, 0
            for frame in chosen:
                value = render_component_fn(component, label, frame)
                if value is None:
                    continue
                total += float(value)
                seen += 1
            per_label.append(total / seen if seen else float("nan"))
        row = torch.tensor(per_label, device=device)
        if not bool(torch.isfinite(row).any()):
            continue
        # Only differences between labels matter, so anchor the component's
        # cheapest hypothesis at zero.
        row = row - row[torch.isfinite(row)].min()
        previous = persistent["energy"][component]
        blended = torch.where(
            persistent["tested"][component][:, None],
            float(config.momentum) * previous + (1.0 - float(config.momentum)) * row[None],
            row[None].expand_as(previous),
        )
        persistent["energy"][component] = blended
        persistent["tested"][component] = True
        persistent["cycle"][component] = int(cycle)
        finite = [value for value in per_label if value == value]
        summaries.append({
            "component": index,
            "size": int(component.numel()),
            "frames": chosen,
            "per_label": per_label,
            "winner": int(row.argmin()) if finite else None,
            "static_minus_best_moving": (
                None if len(per_label) < 2 or per_label[0] != per_label[0]
                else per_label[0] - min(
                    value for value in per_label[1:] if value == value
                ) if any(value == value for value in per_label[1:]) else None
            ),
        })
    # Untested Gaussians must contribute nothing rather than a filled-in value:
    # "not rendered" is not evidence for any label.
    energy = torch.where(
        persistent["tested"][:, None], persistent["energy"],
        torch.zeros_like(persistent["energy"]),
    )
    energy = torch.nan_to_num(energy, nan=0.0, posinf=0.0, neginf=0.0)
    return {"persistent": persistent, "summaries": summaries, "energy": energy}


@torch.no_grad()
def leave_component_out_validation(
    *,
    canonical_fg: torch.Tensor,
    consensus_velocity: torch.Tensor,
    responsibilities: torch.Tensor,
    core_masks: torch.Tensor,
    component: torch.Tensor,
    part: int,
    gamma: float = 2.0,
    solver_kwargs: dict | None = None,
) -> dict:
    """Third safeguard of spec 30: does the field agree with a component it never saw?

    Before a large uncertain component is promoted into a part, the field is
    refitted with that component excluded and then evaluated *on* it.  A
    component that genuinely belongs moves the way a field fitted without it
    predicts; one that does not, does not.  Fitting on a component and then
    observing that it fits is not evidence, and this is the check that
    distinguishes the two.
    """
    from utils.fm_killing_consensus import refit_killing_fields
    from utils.rigid_killing_motion import predict_rigid_cell_velocity

    cores = torch.as_tensor(core_masks).detach().cpu().bool()
    ids = torch.as_tensor(component).detach().cpu().long()
    without = cores.clone()
    without[part, ids] = False
    minimum = int((solver_kwargs or {}).get("min_valid_cells", 8))
    if int(without[part].sum()) < max(3, minimum):
        return {"available": False, "reason": "core_too_small_without_the_component"}

    fit = refit_killing_fields(
        canonical_fg=canonical_fg, consensus_velocity=consensus_velocity,
        responsibilities=responsibilities, core_masks=without[part : part + 1],
        gamma=gamma, solver_kwargs=solver_kwargs,
    )
    mode = fit["modes"][0]
    if not mode.valid:
        return {"available": False, "reason": "held_out_fit_failed"}

    points = torch.as_tensor(canonical_fg).detach().cpu().double()
    velocity = torch.as_tensor(consensus_velocity).detach().cpu().double()
    predicted = predict_rigid_cell_velocity(
        mode, points[None].expand_as(velocity), fit["interval_speeds"][0]
    )
    residual = torch.linalg.norm(velocity - predicted, dim=-1)
    inside = residual[:, ids]
    remaining = torch.nonzero(without[part], as_tuple=False).flatten()
    outside = residual[:, remaining]
    # Also compare against standing still, so a component that simply has no
    # motion is not mistaken for one the field explains.
    static = torch.linalg.norm(velocity[:, ids], dim=-1)
    return {
        "available": True,
        "component_size": int(ids.numel()),
        "held_out_residual": float(inside.mean()),
        "fitted_residual": float(outside.mean()),
        "static_residual": float(static.mean()),
        # Below one means the excluded component is explained about as well as
        # the points that built the field, i.e. it independently agrees.
        "agreement_ratio": float(inside.mean() / outside.mean().clamp_min(_EPS)),
        # Above one means the field beats standing still on this component.
        "improvement_over_static": float(
            static.mean() / inside.mean().clamp_min(_EPS)
        ),
    }


def build_component_render_fn(
    *,
    state,
    render_scene_fn,
    killing_positions: torch.Tensor,
):
    """Adapt a whole-scene render loss into the component/label/frame signature.

    The component is forced to the candidate label while every other Gaussian
    keeps the label the loop currently believes, so the comparison isolates that
    component's contribution and nothing else.
    """
    canonical = state.canonical_fg

    @torch.no_grad()
    def render_component_fn(component: torch.Tensor, label: int, frame: int):
        frame = int(frame)
        if frame <= 0 or frame >= killing_positions.shape[1]:
            return None
        labels = state.hard_labels.clone()
        labels[component] = int(label)
        positions = canonical.clone()
        for part in range(state.num_moving_parts):
            mask = labels == (part + 1)
            if bool(mask.any()):
                positions[mask] = killing_positions[part, frame][mask].to(positions)
        return render_scene_fn(positions, frame)

    return render_component_fn
