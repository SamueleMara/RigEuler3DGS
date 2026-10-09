"""Hand the converged closed-loop state to joint fitting and q(t) (spec 41).

Stage 2 and Stage 3 were written against the older K-part discovery, which
produced its labels over the compacted Gaussian rows.  The closed loop works on
foreground indices into the full cloud, and it carries more than the old stage
did -- a consensus velocity, core masks, a translation-only alternative for each
part.  This module translates between the two without either side having to know
about the other.

Three things happen here.

**Scatter to the full cloud.**  Background Gaussians are static by definition, so
they take label 0.  That is not a padding convention: a background Gaussian
genuinely does not move, and every downstream transport already treats label 0
as the identity.

**Fit the prismatic alternative.**  The closed loop only ever fits the full rigid
field, because that is what the consensus is projected onto.  Stage 2 decides
revolute versus prismatic by comparing the two fits, so the translation-only fit
is produced here, on the same consensus and the same core weights, and the
nested-model evidence goes with it.

**Derive the joint coordinate.**  The closed loop's interval speeds integrate to
the coordinate Stage 2 renders its candidates against.
"""

from __future__ import annotations

import torch

from utils.fm_killing_consensus import build_discrete_physical_proposal
from utils.kpart_field_merge import compare_rigid_and_translation_evidence
from utils.rigid_killing_motion import (
    predict_rigid_cell_velocity,
    solve_translation_mode_all_intervals,
)

_EPS = 1.0e-12


@torch.no_grad()
def integrate_interval_speeds(interval_speeds: torch.Tensor, frame_times) -> torch.Tensor:
    """Running integral of the per-interval speeds: the joint coordinate q(t)."""
    speeds = torch.as_tensor(interval_speeds).detach().cpu().double()
    times = torch.as_tensor([float(value) for value in frame_times], dtype=torch.float64)
    return torch.cat(
        (torch.zeros((speeds.shape[0], 1), dtype=torch.float64),
         torch.cumsum(speeds * torch.diff(times)[None], dim=-1)),
        dim=-1,
    )


@torch.no_grad()
def fit_translation_alternative(state, *, gamma: float = 2.0, solver_kwargs: dict | None = None) -> list[dict]:
    """Fit the prismatic hypothesis per part, on the loop's own consensus.

    Same observations, same weights and same solver settings as the rigid fit the
    loop already made, so the two residuals differ by the model and nothing else.
    Without that the comparison in Stage 2 would be measuring the numerics.
    """
    points = state.canonical_fg.detach().cpu().double()
    velocity = state.consensus_velocity.detach().cpu().double()
    share = state.responsibilities.detach().cpu().double()
    cores = state.core_masks.detach().cpu().bool()
    kwargs = dict(solver_kwargs or {})
    positions = points[None].expand_as(velocity).contiguous()

    results = []
    for part, mode in enumerate(state.killing_modes):
        weight_row = share[:, part + 1].clamp(0.0, 1.0).pow(float(gamma)) * cores[part].double()
        weights = weight_row[None].expand(velocity.shape[0], -1).contiguous()
        valid = weights > 0
        effective = weights * valid.double()
        observation_count = int(valid.sum())

        rigid_cost = float("nan")
        if mode.valid and observation_count:
            predicted = predict_rigid_cell_velocity(
                mode, positions, state.killing_interval_speeds[part].detach().cpu().double()
            )
            residual = torch.linalg.norm(velocity - predicted, dim=-1)
            rigid_cost = float(
                (effective * residual.square()).sum() / effective.sum().clamp_min(_EPS)
            )

        translation_cost = float("nan")
        translation_mode = translation_speed = None
        if observation_count >= int(kwargs.get("min_valid_cells", 8)):
            translation_mode, translation_speed, _ = solve_translation_mode_all_intervals(
                positions, velocity, weights=weights, valid_mask=valid, **kwargs
            )
            if translation_mode.valid:
                predicted = predict_rigid_cell_velocity(
                    translation_mode, positions, translation_speed
                )
                residual = torch.linalg.norm(velocity - predicted, dim=-1)
                translation_cost = float(
                    (effective * residual.square()).sum() / effective.sum().clamp_min(_EPS)
                )

        evidence = compare_rigid_and_translation_evidence(
            0.0 if rigid_cost != rigid_cost else rigid_cost,
            0.0 if translation_cost != translation_cost else translation_cost,
            observation_count,
            int(velocity.shape[-1]),
        )
        results.append({
            "mode": mode,
            "temporal_speed": state.killing_interval_speeds[part].detach().cpu().double(),
            "cost": rigid_cost,
            "translation_mode": translation_mode,
            "translation_speed": translation_speed,
            "translation_cost": translation_cost,
            "translation_valid": bool(translation_mode is not None and translation_mode.valid),
            "joint_type_evidence": evidence,
            "core_size": int(cores[part].sum()),
        })
    return results


@torch.no_grad()
def closed_loop_state_to_stage1(
    state,
    *,
    total_gaussians: int,
    gamma: float = 2.0,
    solver_kwargs: dict | None = None,
) -> dict:
    """The ``stage1`` dictionary Stage 2 and Stage 3 already know how to consume.

    Labels and responsibilities are scattered to the full cloud with background
    as static, so ``fit_joint_graph`` can render whole scenes and
    ``optimize_joint_trajectory`` can pick out each part, without either of them
    needing to know the loop worked on a subset.
    """
    fields = fit_translation_alternative(state, gamma=gamma, solver_kwargs=solver_kwargs)
    coordinates = integrate_interval_speeds(state.killing_interval_speeds, state.frame_times)

    device = state.canonical_fg.device
    labels_full = torch.zeros(int(total_gaussians), dtype=torch.long, device=device)
    labels_full[state.foreground_idx] = state.hard_labels.to(device)
    responsibilities_full = torch.zeros(
        (int(total_gaussians), state.num_labels), dtype=torch.float32, device=device
    )
    # Background is static with probability one; it is not unknown.
    responsibilities_full[:, 0] = 1.0
    responsibilities_full[state.foreground_idx] = state.responsibilities.to(device)

    return {
        "modes": [item["mode"] for item in fields],
        "interval_speeds": [item["temporal_speed"] for item in fields],
        "coordinates": [coordinates[part] for part in range(state.num_moving_parts)],
        "frame_times": list(state.frame_times),
        "frame_ids": list(state.frame_ids),
        "labels": labels_full,
        "responsibilities": responsibilities_full,
        "velocity_fields": fields,
        "classification": {
            "labels": labels_full,
            "confidence": responsibilities_full.amax(-1),
            "entropy": torch.zeros(int(total_gaussians), device=device),
            "ambiguous_mask": torch.zeros(int(total_gaussians), dtype=torch.bool, device=device),
        },
        "source": "closed_loop_diffusion_fm_killing",
        "foreground_idx": state.foreground_idx,
        "num_foreground": state.num_foreground,
    }


@torch.no_grad()
def joint_transport_positions(
    *, points: torch.Tensor, joints, coordinates, pitch_default: float = 0.0
) -> torch.Tensor:
    """``[K, T, N_F, 3]``: every point carried by every joint's exact transport.

    Same layout as the loop's ``killing_positions``, so everything downstream
    that consumed the coarse transport consumes this without change.  That is
    the whole point of section 41 step 3 -- replace the transport, keep the
    machinery.
    """
    from utils.rigid_transform_utils import joint_incremental_transform

    frames = int(torch.as_tensor(coordinates[0]).numel())
    positions = points[None, None].expand(len(joints), frames, *points.shape).clone()
    for part, joint in enumerate(joints):
        axis = torch.as_tensor(joint["axis"], device=points.device, dtype=points.dtype)
        pivot = (
            None if joint.get("pivot") is None
            else torch.as_tensor(joint["pivot"], device=points.device, dtype=points.dtype)
        )
        row = torch.as_tensor(coordinates[part]).detach().cpu().double().reshape(-1)
        for frame in range(1, frames):
            rotation, translation, _ = joint_incremental_transform(
                joint["joint_type"], axis, float(row[frame] - row[0]),
                pivot=pivot, pitch=joint.get("pitch", pitch_default),
            )
            positions[part, frame] = points @ rotation.T.to(points) + translation.to(points)
    return positions


@torch.no_grad()
def joint_transport_velocity(
    *, positions: torch.Tensor, frame_times
) -> torch.Tensor:
    """``[K, T-1, N_F, 3]`` finite-difference velocity of an exact transport."""
    times = torch.as_tensor([float(v) for v in frame_times], dtype=torch.float64)
    deltas = torch.diff(times).to(positions).clamp_min(1.0e-8)
    return torch.diff(positions, dim=1) / deltas[None, :, None, None]


@torch.no_grad()
def reassign_under_joint_transport(
    state,
    *,
    joints,
    coordinates: torch.Tensor,
    frame_times,
    temperature: float = 0.2,
    move_activation_cost: float = 0.0,
    denoiser=None,
    denoiser_spec=None,
    diffusion_config=None,
    knn_idx=None,
    neighbour_scale: float = 1.0,
    graph: dict | None = None,
    assignment_weights=None,
    target_frames=None,
    render_evidence_fn=None,
    observation_confidence_fn=None,
    counterfactual_energy=None,
) -> dict:
    """Re-decide the assignment against the *exact joint* transport (spec 41).

    Steps 3 to 6 of section 41.  The transport a moving Gaussian actually
    undergoes, once Stage 3 has metric ``q(t)``, is the joint's and not the
    coarse Killing field's -- so the transport is replaced and *the same*
    K+1 energies are recomputed against it.

    Recomputing every modality rather than only the velocity one is not
    optional.  Measured on the USB scene, the velocity observation reaches
    4542 of 11263 foreground Gaussians; an assignment decided on that
    modality alone re-imposes that coverage ceiling and *sheds* correctly
    labelled moving Gaussians that the graph, the critic and the
    component-level rendering had rescued -- Stage 4 dropped F1 from 0.379 to
    0.289 doing exactly that.  Section 41 step 5 says "recompute the K+1
    energies", plural, and this is why.

    The optional arguments are the modalities: whichever are supplied are
    recomputed under the exact transport, and whichever are not are simply
    absent from the combination.  ``counterfactual_energy`` is passed through
    unchanged -- it was earned by rendering, and the transport it was earned
    against is the one being installed.
    """
    from utils.closed_loop_assignment import (
        AssignmentWeights,
        compute_assignment_graph_energy,
        compute_diffusion_transition_energy,
        compute_fm_killing_label_energy,
        compute_observed_velocity_label_energy,
        robust_normalize_and_combine,
        update_responsibilities,
    )

    points = state.canonical_fg
    positions = joint_transport_positions(
        points=points, joints=joints, coordinates=coordinates
    )
    velocity = joint_transport_velocity(positions=positions, frame_times=frame_times)
    valid = [True] * len(joints)

    energies = {
        "observed_velocity": compute_observed_velocity_label_energy(
            observed_velocity=state.observed_velocity,
            observed_normal_matrix=state.observed_normal_matrix,
            observed_valid=state.observed_valid,
            killing_velocity=velocity,
            killing_valid=valid,
        ),
        "fm": None,
        "graph": None,
        "diffusion": None,
        "counterfactual": (
            None if counterfactual_energy is None else counterfactual_energy.detach()
        ),
    }
    if state.fm_velocity is not None and float(state.fm_confidence.abs().sum()) > 0.0:
        energies["fm"] = compute_fm_killing_label_energy(
            fm_velocity=state.fm_velocity, fm_confidence=state.fm_confidence,
            killing_velocity=velocity, killing_valid=valid,
        )
    if graph is not None:
        energies["graph"] = compute_assignment_graph_energy(state.responsibilities, graph)

    # Step 4: one more physics-conditioned residual-diffusion pass, this time
    # against the exact-joint proposal.  The critic is the only channel that
    # speaks for a Gaussian whose velocity was never observed, so leaving it out
    # is what made this stage lose recall.
    diffusion = None
    if denoiser is not None and denoiser_spec is not None and knn_idx is not None:
        from utils.physics_conditioned_diffusion import (
            DiffusionConfig,
            run_physics_conditioned_diffusion,
        )

        proposal = build_discrete_physical_proposal(
            canonical_fg=points, hard_labels=state.hard_labels, killing_positions=positions
        )
        frames = (
            list(range(1, int(positions.shape[1])))
            if target_frames is None else [int(v) for v in target_frames]
        )
        diffusion = run_physics_conditioned_diffusion(
            denoiser, spec=denoiser_spec, state=state, physical_positions=proposal,
            target_frames=frames, knn_idx=knn_idx, neighbour_scale=neighbour_scale,
            config=diffusion_config or DiffusionConfig(),
            render_evidence_fn=render_evidence_fn,
            observation_confidence_fn=observation_confidence_fn,
        )
        energies["diffusion"] = compute_diffusion_transition_energy(
            canonical_fg=points, hard_labels=state.hard_labels,
            killing_positions=positions, residual_mean=diffusion.mean,
            residual_cov=diffusion.cov,
            residual_confidence=diffusion.confidence, target_frames=frames,
        )

    weights = assignment_weights or AssignmentWeights()
    weights.move_activation_cost = float(move_activation_cost)
    combined = robust_normalize_and_combine(
        energy_diffusion=energies["diffusion"], energy_fm=energies["fm"],
        energy_observed=energies["observed_velocity"], energy_graph=energies["graph"],
        energy_counterfactual=energies["counterfactual"], weights=weights,
    )
    update = update_responsibilities(
        combined["energy_total"], temperature, previous=state.responsibilities
    )
    changed = int((update["hard_labels"] != state.hard_labels).sum())
    return {
        **update,
        "joint_positions": positions,
        "joint_velocity": velocity,
        "labels_changed": changed,
        "energies": energies,
        "energy_scales": combined["scales"],
        "modalities_used": sorted(k for k, v in energies.items() if v is not None),
        "diffusion": diffusion,
    }
