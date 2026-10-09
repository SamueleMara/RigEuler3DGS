"""Closed-loop diffusion / flow-matching / Killing pipeline.

    PYTHONPATH=. python tests/test_closed_loop_motion.py

Everything here runs on synthetic data whose answer is known, so a failure names
the module that broke rather than the scene that was unlucky.
"""

import math
import tempfile
from pathlib import Path

import torch

from scene.physics_conditioned_gaussian_denoiser import (
    DenoiserConfig,
    PhysicsConditionedGaussianDenoiser,
    PhysicsFeatureSpec,
    assemble_features,
    build_foreground_knn,
    load_denoiser,
)
from utils.closed_loop_assignment import (
    AssignmentWeights,
    assignment_temperature,
    build_foreground_graph,
    compute_assignment_graph_energy,
    compute_diffusion_transition_energy,
    compute_fm_killing_label_energy,
    compute_observed_velocity_label_energy,
    robust_normalize_and_combine,
    update_responsibilities,
)
from utils.closed_loop_flow_matching import TemporalFlowConfig, TemporalFlowField
from utils.closed_loop_motion_losses import flow_matching_feedback_loss
from utils.closed_loop_motion_state import allocate_closed_loop_state, compute_entropy
from utils.fm_killing_consensus import (
    ConsensusWeights,
    CoreShellConfig,
    bootstrap_moving_core_from_observations,
    build_discrete_physical_proposal,
    compute_consensus_velocity,
    integrate_killing_positions,
    integrate_killing_transforms,
    refit_killing_fields,
    update_core_shell_hysteresis,
)
from utils.physics_conditioned_diffusion import (
    CorruptionConfig,
    corrupt_physical_proposal,
    decompose_residual_against_field,
)

AXIS = torch.tensor([0.0, 0.0, 1.0])


def _hinged_slab(n=300, frames=6, speed=0.8, seed=0):
    """A slab whose top half rotates about +z; the bottom half is static."""
    generator = torch.Generator().manual_seed(seed)
    points = torch.rand(n, 3, generator=generator) * torch.tensor([1.0, 1.0, 0.4])
    points = points - torch.tensor([0.5, 0.5, 0.2])
    moving = points[:, 2] > 0.0
    times = [index / 30.0 for index in range(frames)]
    velocity = torch.zeros(frames - 1, n, 3)
    for interval in range(frames - 1):
        velocity[interval, moving] = speed * torch.cross(
            AXIS.expand(int(moving.sum()), 3), points[moving], dim=-1
        )
    return points, moving, times, velocity


def _state_with_observations(points, moving, times, velocity, *, noise=0.02, drop=0.25, seed=0):
    generator = torch.Generator().manual_seed(seed)
    n, intervals = points.shape[0], velocity.shape[0]
    state = allocate_closed_loop_state(
        points, torch.arange(n), list(range(len(times))), times, 1, static_prior=0.7
    )
    valid = torch.rand(intervals, n, generator=generator) > drop
    state.observed_velocity = velocity + torch.randn(velocity.shape, generator=generator) * noise
    state.observed_valid = valid
    state.observed_normal_matrix = torch.eye(3).expand(intervals, n, 3, 3).clone()
    state.observed_confidence = valid.float() * 0.8
    state.fm_velocity = state.observed_velocity.clone()
    state.fm_confidence = state.observed_confidence.clone()
    state.assignment_entropy = compute_entropy(state.responsibilities)
    state.validate()
    return state


def test_state_enforces_its_own_shapes():
    points, _, times, _ = _hinged_slab(n=120, frames=5)
    state = allocate_closed_loop_state(points, torch.arange(120), list(range(5)), times, 2)
    state.validate()
    assert state.num_labels == 3 and state.num_intervals == 4
    assert torch.allclose(state.responsibilities.sum(-1), torch.ones(120))
    # Background never enters a motion tensor; the only crossing is outward.
    full = state.scatter_to_full(state.canonical_fg, total=500, fill=float("nan"))
    assert bool(torch.isnan(full[200:]).all())
    state.fm_velocity = torch.zeros(3, 120, 3)
    try:
        state.validate()
    except ValueError as error:
        assert "fm_velocity" in str(error)
    else:
        raise AssertionError("a shape drift must be caught, not carried forward")


def test_static_hypothesis_pulls_consensus_to_zero():
    """A static-dominant point resists a flow-matching hallucination."""
    points, moving, times, velocity = _hinged_slab(n=200)
    n, intervals = points.shape[0], velocity.shape[0]
    share = torch.zeros(n, 2)
    share[moving, 1], share[moving, 0] = 0.95, 0.05
    share[~moving, 0], share[~moving, 1] = 0.95, 0.05
    generator = torch.Generator().manual_seed(1)
    fm = velocity.clone()
    fm[:, ~moving] = torch.randn(intervals, int((~moving).sum()), 3, generator=generator) * 0.3

    result = compute_consensus_velocity(
        fm_velocity=fm, fm_confidence=torch.ones(intervals, n),
        observed_velocity=torch.zeros_like(fm),
        observed_normal_matrix=torch.zeros(intervals, n, 3, 3),
        observed_valid=torch.zeros(intervals, n, dtype=torch.bool),
        killing_velocity=torch.zeros(1, intervals, n, 3),
        responsibilities=share, killing_valid=[False], weights=ConsensusWeights(),
    )
    consensus = result["consensus_velocity"]
    assert float(consensus[:, ~moving].norm(dim=-1).mean()) < float(fm[:, ~moving].norm(dim=-1).mean())
    # An unfitted part must not assert v = 0: doing so is the static hypothesis
    # in disguise and halves the consensus on genuinely moving points.
    gated = float(consensus[:, moving].norm(dim=-1).mean())
    ungated = compute_consensus_velocity(
        fm_velocity=fm, fm_confidence=torch.ones(intervals, n),
        observed_velocity=torch.zeros_like(fm),
        observed_normal_matrix=torch.zeros(intervals, n, 3, 3),
        observed_valid=torch.zeros(intervals, n, dtype=torch.bool),
        killing_velocity=torch.zeros(1, intervals, n, 3),
        responsibilities=share, killing_valid=None, weights=ConsensusWeights(),
    )["consensus_velocity"]
    assert gated > 1.5 * float(ungated[:, moving].norm(dim=-1).mean())


def test_bootstrap_breaks_the_cold_start_deadlock():
    """Without this the core, the field and the assignment all wait on each other."""
    points, moving, _, velocity = _hinged_slab(n=400)
    intervals, n = velocity.shape[0], points.shape[0]
    generator = torch.Generator().manual_seed(2)
    valid = torch.rand(intervals, n, generator=generator) > 0.25
    seed = bootstrap_moving_core_from_observations(
        observed_velocity=velocity + torch.randn(velocity.shape, generator=generator) * 0.02,
        observed_confidence=valid.float() * 0.8, observed_valid=valid,
        canonical_fg=points, num_moving_parts=1,
    )
    core = seed["core_masks"][0]
    assert seed["reason"] is None and int(core.sum()) > 0
    precision = float((core & moving).sum() / core.sum())
    assert precision > 0.9, precision
    # A scene with no motion has no split to find, and says so rather than
    # inventing a core out of noise.
    still = bootstrap_moving_core_from_observations(
        observed_velocity=torch.zeros_like(velocity), observed_confidence=valid.float(),
        observed_valid=valid, canonical_fg=points, num_moving_parts=1,
    )
    assert still["reason"] == "no_observed_motion"


def test_core_hysteresis_retains_between_the_thresholds():
    """Enter high and persistent, leave only when low: a band, not a gate."""
    share = torch.tensor([[0.05, 0.95], [0.05, 0.95]])
    streak = torch.zeros(2, 1, dtype=torch.long)
    static_streak = torch.zeros(2, dtype=torch.long)
    config = CoreShellConfig(enter=0.9, exit=0.3, persistence_cycles=2)
    core = None
    for cycle in range(3):
        result = update_core_shell_hysteresis(
            responsibilities=share, moving_streak=streak, static_streak=static_streak,
            previous_core=core, config=config,
        )
        streak, static_streak, core = (
            result["moving_streak"], result["static_streak"], result["core_masks"]
        )
        # One cycle above `enter` is not enough; two are.
        assert int(core.sum()) == (0 if cycle == 0 else 2), (cycle, int(core.sum()))

    # Drifting into the middle of the band must not expel an existing member.
    middle = torch.tensor([[0.5, 0.5], [0.5, 0.5]])
    result = update_core_shell_hysteresis(
        responsibilities=middle, moving_streak=streak, static_streak=static_streak,
        previous_core=core, config=config,
    )
    assert int(result["core_masks"].sum()) == 2, "a hard gate would empty the core here"
    # Falling below `exit` does expel it.
    low = torch.tensor([[0.9, 0.1], [0.9, 0.1]])
    result = update_core_shell_hysteresis(
        responsibilities=low, moving_streak=result["moving_streak"],
        static_streak=result["static_streak"], previous_core=result["core_masks"], config=config,
    )
    assert int(result["core_masks"].sum()) == 0


def test_killing_refit_recovers_the_field_and_keeps_speeds_when_skipped():
    points, moving, times, velocity = _hinged_slab(n=300, speed=0.8)
    n = points.shape[0]
    share = torch.zeros(n, 2)
    share[moving, 1], share[moving, 0] = 0.95, 0.05
    share[~moving, 0], share[~moving, 1] = 0.95, 0.05
    core = torch.zeros(1, n, dtype=torch.bool)
    core[0, moving] = True
    fit = refit_killing_fields(
        canonical_fg=points, consensus_velocity=velocity, responsibilities=share,
        core_masks=core, gamma=2.0, solver_kwargs={"min_valid_cells": 8},
    )
    mode = fit["modes"][0]
    assert mode.valid
    axis = mode.angular_mode.double() / mode.angular_mode.double().norm()
    error = math.degrees(math.acos(min(1.0, abs(float(axis @ AXIS.double())))))
    assert error < 1.0, error
    assert abs(float(fit["interval_speeds"][0].mean()) - 0.8) < 0.01

    # A part whose core has vanished keeps its previous motion.  Returning zero
    # speeds would freeze it at canonical while its axis still looked correct.
    skipped = refit_killing_fields(
        canonical_fg=points, consensus_velocity=velocity, responsibilities=share,
        core_masks=torch.zeros(1, n, dtype=torch.bool), gamma=2.0,
        previous_modes=fit["modes"], previous_speeds=fit["interval_speeds"],
        solver_kwargs={"min_valid_cells": 8},
    )
    assert not skipped["diagnostics"][0]["refitted"]
    assert torch.allclose(skipped["interval_speeds"][0], fit["interval_speeds"][0])


def test_physical_proposal_is_discrete_never_blended():
    points, moving, times, velocity = _hinged_slab(n=200)
    n = points.shape[0]
    share = torch.zeros(n, 2)
    share[moving, 1], share[moving, 0] = 0.6, 0.4      # deliberately soft
    share[~moving, 0], share[~moving, 1] = 0.6, 0.4
    core = torch.zeros(1, n, dtype=torch.bool)
    core[0, moving] = True
    fit = refit_killing_fields(
        canonical_fg=points, consensus_velocity=velocity, responsibilities=share,
        core_masks=core, solver_kwargs={"min_valid_cells": 8},
    )
    tracks = integrate_killing_positions(fit["modes"], fit["interval_speeds"], points, times)
    transforms = integrate_killing_transforms(fit["modes"], fit["interval_speeds"], times)
    assert transforms["rotations"].shape == (1, len(times), 3, 3)
    assert transforms["translations"].shape == (1, len(times), 3)
    assert transforms["rotation_quaternions"].shape == (1, len(times), 4)
    proposal = build_discrete_physical_proposal(
        canonical_fg=points, hard_labels=share.argmax(-1), killing_positions=tracks
    )
    # Static points are *exactly* canonical, not 40% of the way somewhere.
    assert torch.equal(proposal[-1][~moving], points[~moving])
    moved = (proposal[-1][moving] - points[moving]).norm(dim=-1)
    exact = (tracks[0, -1][moving] - points[moving]).norm(dim=-1)
    assert torch.allclose(moved, exact), "a moving point must take the whole transform"


def test_denoiser_layout_is_derived_and_starts_at_no_correction():
    spec = PhysicsFeatureSpec(num_moving_parts=2)
    offsets = spec.offsets()
    assert spec.input_dim == sum(width for _, width in spec.blocks())
    assert offsets["noise_residual"] == (0, 3)
    points = torch.randn(150, 3) * 0.1
    knn, scale = build_foreground_knn(points, k=16)
    share = torch.rand(150, 3)
    share = share / share.sum(-1, keepdim=True)
    features = assemble_features(
        spec, current_positions=points + 0.01, physical_positions=points, canonical_fg=points,
        fm_velocity=torch.randn(150, 3) * 0.05, killing_velocity=torch.randn(150, 3) * 0.05,
        responsibilities=share, assignment_entropy=compute_entropy(share),
        hard_labels=share.argmax(-1), neighbour_scale=scale,
    )
    assert features.shape == (150, spec.input_dim) and bool(torch.isfinite(features).all())
    model = PhysicsConditionedGaussianDenoiser(spec.input_dim, hidden_dim=64, message_layers=2)
    # An untrained critic proposes exactly nothing, so training starts from the
    # hypothesis rather than from a random displacement of it.
    assert float(model(features, knn, step_scale=scale).abs().max()) == 0.0
    payload = model.checkpoint_payload(DenoiserConfig(spec=spec))
    restored, config = load_denoiser(payload)
    assert restored.input_dim == model.input_dim and config.spec.num_moving_parts == 2
    try:
        load_denoiser({"pipeline": "legacy_motion_denoiser", "denoiser_config": {}})
    except ValueError:
        pass
    else:
        raise AssertionError("a legacy checkpoint has a different layout and must be refused")


def test_corruption_produces_genuinely_wrong_hypotheses():
    points, moving, times, velocity = _hinged_slab(n=250)
    n = points.shape[0]
    share = torch.zeros(n, 2)
    share[moving, 1] = 1.0
    share[~moving, 0] = 1.0
    core = torch.zeros(1, n, dtype=torch.bool)
    core[0, moving] = True
    fit = refit_killing_fields(
        canonical_fg=points, consensus_velocity=velocity, responsibilities=share,
        core_masks=core, solver_kwargs={"min_valid_cells": 8},
    )
    labels = share.argmax(-1)
    clean = build_discrete_physical_proposal(
        canonical_fg=points, hard_labels=labels,
        killing_positions=integrate_killing_positions(
            fit["modes"], fit["interval_speeds"], points, times
        ),
    )
    changed = 0
    for seed in range(4):
        broken = corrupt_physical_proposal(
            canonical_fg=points, hard_labels=labels, killing_modes=fit["modes"],
            interval_speeds=fit["interval_speeds"], frame_times=times,
            config=CorruptionConfig(), generator=torch.Generator().manual_seed(seed),
        )
        changed += int((broken["labels"] != labels).sum())
        assert broken["corruptions"], "a corruption pass must record what it broke"
        assert float((broken["positions"] - clean).abs().max()) > 0
    assert changed > 0, "the critic must see wrong labels, not only wrong geometry"
    # Disabled corruption must reproduce the clean proposal exactly.
    identical = corrupt_physical_proposal(
        canonical_fg=points, hard_labels=labels, killing_modes=fit["modes"],
        interval_speeds=fit["interval_speeds"], frame_times=times,
        config=CorruptionConfig(enabled=False),
    )
    assert torch.allclose(identical["positions"], clean)


def test_residual_decomposition_separates_the_three_failures():
    points, moving, times, velocity = _hinged_slab(n=250)
    n = points.shape[0]
    share = torch.zeros(n, 2)
    share[moving, 1] = 1.0
    share[~moving, 0] = 1.0
    core = torch.zeros(1, n, dtype=torch.bool)
    core[0, moving] = True
    mode = refit_killing_fields(
        canonical_fg=points, consensus_velocity=velocity, responsibilities=share,
        core_masks=core, solver_kwargs={"min_valid_cells": 8},
    )["modes"][0]
    field = torch.cross(AXIS.expand(n, 3), points, dim=-1)
    generator = torch.Generator().manual_seed(3)

    def describe(residual):
        return decompose_residual_against_field(
            residual[None].expand(len(times), n, 3), canonical_fg=points, mode=mode,
            responsibilities_for_part=share[:, 1],
        )

    speed_error = describe(0.3 * field)
    axis_error = describe(AXIS.expand(n, 3) * 0.01)
    mislabelling = describe(torch.randn(n, 3, generator=generator) * 0.01)
    assert speed_error["parallel_over_orthogonal"] > 10.0
    assert axis_error["orthogonal_coherence"] > 0.9
    assert mislabelling["orthogonal_coherence"] < 0.3
    assert mislabelling["parallel_over_orthogonal"] < 2.0


def test_diffusion_residual_votes_for_the_label_it_wants():
    """The critic's correction is read directly as a label transition."""
    points, moving, times, _ = _hinged_slab(n=200)
    n, frames = points.shape[0], len(times)
    displacement = torch.zeros(1, frames, n, 3)
    for frame in range(frames):
        displacement[0, frame, :, 1] = 0.05 * frame
    tracks = points[None, None].expand(1, frames, n, 3) + displacement

    for hypothesis, residual_sign in ((0, +1.0), (1, -1.0)):
        labels = torch.full((n,), hypothesis, dtype=torch.long)
        residual = torch.zeros(frames, n, 3)
        # The critic asks for the motion to be applied (from static) or undone
        # (from moving) exactly on the points whose label is wrong.
        wrong = moving if hypothesis == 0 else ~moving
        residual[:, wrong] = residual_sign * displacement[0][:, wrong]
        energy = compute_diffusion_transition_energy(
            canonical_fg=points, hard_labels=labels, killing_positions=tracks,
            residual_mean=residual, residual_cov=torch.eye(3).expand(frames, n, 3, 3) * 1e-6,
            residual_confidence=torch.ones(frames, n), target_frames=[1, frames - 1],
        )
        assert energy.shape == (n, 2)
        assert int(energy[moving].argmin(-1).mode().values) == 1
        assert int(energy[~moving].argmin(-1).mode().values) == 0


def test_energies_survive_wildly_different_scales():
    points, moving, times, velocity = _hinged_slab(n=200)
    n, intervals = points.shape[0], velocity.shape[0]
    killing = torch.zeros(1, intervals, n, 3)
    killing[0] = velocity
    fm = torch.zeros(intervals, n, 3)
    fm[:, moving] = velocity[:, moving]

    energy_fm = compute_fm_killing_label_energy(
        fm_velocity=fm, fm_confidence=torch.ones(intervals, n),
        killing_velocity=killing, killing_valid=[True],
    )
    energy_vel = compute_observed_velocity_label_energy(
        observed_velocity=fm, observed_normal_matrix=torch.eye(3).expand(intervals, n, 3, 3).clone(),
        observed_valid=torch.ones(intervals, n, dtype=torch.bool),
        killing_velocity=killing, killing_valid=[True],
    )
    knn, _ = build_foreground_knn(points, k=8)
    graph = build_foreground_graph(points, knn)
    energy_graph = compute_assignment_graph_energy(torch.full((n, 2), 0.5), graph)

    combined = robust_normalize_and_combine(
        energy_diffusion=None, energy_fm=energy_fm * 1.0e9, energy_observed=energy_vel * 1.0e-9,
        energy_graph=energy_graph, weights=AssignmentWeights(move_activation_cost=0.05),
    )
    # Comparability is the property that matters, not an absolute bound: after
    # scaling by each modality's own resolved gap, the typical discriminative
    # magnitudes must be within an order of magnitude of each other, so no
    # modality decides purely because of the units it happens to be in.
    typical = {
        name: float((value.amax(-1) - value.amin(-1)).quantile(0.9))
        for name, value in combined["normalized"].items()
    }
    resolving = [v for v in typical.values() if v > 0]
    assert resolving, typical
    assert max(resolving) / min(resolving) < 10.0, typical
    update = update_responsibilities(combined["energy_total"], assignment_temperature(0))
    assert float(((update["hard_labels"] > 0) == moving).float().mean()) > 0.95
    # An unfitted part is charged the static cost, never given a free zero.
    unfitted = compute_fm_killing_label_energy(
        fm_velocity=fm, fm_confidence=torch.ones(intervals, n),
        killing_velocity=killing, killing_valid=[False],
    )
    assert torch.allclose(unfitted[:, 0], unfitted[:, 1])


def test_flow_matching_feedback_is_detached_from_the_physics():
    """FM learns from Killing; no gradient may run back into the solver."""
    points, moving, _, velocity = _hinged_slab(n=120)
    n, intervals = points.shape[0], velocity.shape[0]
    killing = torch.zeros(1, intervals, n, 3, requires_grad=True)
    share = torch.rand(n, 2, requires_grad=True)
    normalised = share / share.sum(-1, keepdim=True)
    predicted = torch.zeros(intervals, n, 3, requires_grad=True)
    loss = flow_matching_feedback_loss(
        predicted_velocity=predicted, killing_velocity=killing,
        responsibilities=normalised, killing_valid=[True],
    )
    loss["total"].backward()
    assert predicted.grad is not None
    assert killing.grad is None, "a gradient into Killing would let FM reshape the physics"
    assert share.grad is None


def test_flow_field_starts_still_and_stays_temporal():
    config = TemporalFlowConfig(hidden_dim=32, layers=2, iterations=1)
    field = TemporalFlowField(config)
    points = torch.randn(50, 3) * 0.1
    velocities = field.evaluate_all_intervals(points, num_intervals=4)
    assert velocities.shape == (4, 50, 3)
    assert float(velocities.abs().max()) == 0.0
    # Different intervals must be able to differ, or a stop-start joint would be
    # averaged into one constant velocity matching no interval.
    with torch.no_grad():
        for parameter in field.network[-1].parameters():
            parameter.add_(torch.randn(parameter.shape) * 0.1)
    varied = field.evaluate_all_intervals(points, num_intervals=4)
    assert float((varied[0] - varied[-1]).abs().max()) > 0.0


def test_the_loop_closes_and_recovers_the_hinge():
    """The end-to-end property the whole pipeline exists for."""
    from tools.train_closed_loop_motion import ABLATIONS, build_parser, run_closed_loop

    torch.manual_seed(0)
    points, moving, times, velocity = _hinged_slab(n=400, frames=6, speed=0.8)
    state = _state_with_observations(points, moving, times, velocity)
    knn, scale = build_foreground_knn(points, k=16)
    graph = build_foreground_graph(points, knn)
    spec = PhysicsFeatureSpec(num_moving_parts=1)
    denoiser = PhysicsConditionedGaussianDenoiser(spec.input_dim, hidden_dim=64, message_layers=2)
    field = TemporalFlowField(TemporalFlowConfig(hidden_dim=64, layers=3, iterations=60))
    args = build_parser().parse_args([
        "--root-scene", "x", "--static-model-dir", "x", "--output-dir", "x",
        "--outer-iterations", "7", "--diffusion-trials", "3", "--diffusion-steps", "2",
        "--fm-iterations", "60", "--move-activation-cost", "0.05",
    ])
    with tempfile.TemporaryDirectory() as directory:
        result = run_closed_loop(
            state=state, args=args, ablation=ABLATIONS["G_full_closed_loop"],
            denoiser=denoiser, spec=spec, flow_field=field, knn_idx=knn,
            neighbour_scale=scale, graph=graph, output_root=Path(directory),
            fm_base_velocity=velocity, fm_base_confidence=state.observed_confidence,
            log=lambda message: None,
        )
        first = Path(directory) / "outer_0000"
        for name in ("responsibilities.pt", "killing_modes.pt", "diffusion_residual_mean.pt",
                     "energy_diffusion.pt", "core_masks.pt", "diagnostics.json", "state.html"):
            assert (first / name).is_file(), name

    accuracy = float(((state.hard_labels > 0) == moving).float().mean())
    assert accuracy > 0.9, accuracy
    mode = state.killing_modes[0]
    assert mode.valid
    axis = mode.angular_mode.double() / mode.angular_mode.double().norm()
    error = math.degrees(math.acos(min(1.0, abs(float(axis @ AXIS.double())))))
    assert error < 5.0, error
    assert float(state.killing_interval_speeds[0].mean()) > 0.5

    # Soft early, committing late: an early hard label feeds a contaminated core
    # into the next Killing fit, so the schedule must not collapse immediately.
    entropies = [record["mean_assignment_entropy"] for record in result["records"]]
    assert entropies[0] > entropies[-1] and entropies[0] > 0.3
    # And the loop must have kept re-deciding rather than freezing on cycle one.
    assert sum(1 for r in result["records"] if r["responsibility_delta"] > 1e-3) >= 3


def _converged_state(n=300, frames=6, total=500):
    """A state that looks like the loop has already converged on it."""
    points, moving, times, velocity = _hinged_slab(n=n, frames=frames, speed=0.8)
    from utils.fm_killing_consensus import predict_all_killing_velocities

    index = torch.arange(n) * (total // n)
    state = allocate_closed_loop_state(points, index, list(range(frames)), times, 1)
    share = torch.zeros(n, 2)
    share[moving, 1], share[moving, 0] = 0.95, 0.05
    share[~moving, 0], share[~moving, 1] = 0.95, 0.05
    state.responsibilities = share
    state.hard_labels = share.argmax(-1)
    state.assignment_entropy = compute_entropy(share)
    state.consensus_velocity = velocity
    state.observed_velocity = velocity
    state.observed_valid = torch.ones(frames - 1, n, dtype=torch.bool)
    state.observed_normal_matrix = torch.eye(3).expand(frames - 1, n, 3, 3).clone()
    core = torch.zeros(1, n, dtype=torch.bool)
    core[0, moving] = True
    state.core_masks = core
    fit = refit_killing_fields(
        canonical_fg=points, consensus_velocity=velocity, responsibilities=share,
        core_masks=core, solver_kwargs={"min_valid_cells": 8},
    )
    state.killing_modes = fit["modes"]
    state.killing_interval_speeds = fit["interval_speeds"].float()
    state.killing_velocity = predict_all_killing_velocities(
        state.killing_modes, state.killing_interval_speeds, points
    )
    state.killing_positions = integrate_killing_positions(
        state.killing_modes, state.killing_interval_speeds, points, times
    )
    state.validate()
    return state, points, moving, times, total, index


def test_stage1_hands_over_to_joint_fitting():
    """The section-41 handover: state -> the dict Stage 2 and Stage 3 consume."""
    from utils.closed_loop_to_joint import closed_loop_state_to_stage1

    state, points, moving, times, total, index = _converged_state()
    stage1 = closed_loop_state_to_stage1(state, total_gaussians=total)

    # Scattered to the full cloud, with background static because it is.
    assert stage1["labels"].shape == (total,)
    assert stage1["responsibilities"].shape == (total, 2)
    background = torch.ones(total, dtype=torch.bool)
    background[index] = False
    assert bool((stage1["labels"][background] == 0).all())
    assert bool((stage1["responsibilities"][background, 0] == 1.0).all())
    assert torch.equal(stage1["labels"][index], state.hard_labels)

    # Both hypotheses are fitted, so the type can be selected rather than assumed.
    field = stage1["velocity_fields"][0]
    assert field["translation_valid"]
    assert field["translation_cost"] > field["cost"], "a rotation must fit a hinge better"
    evidence = field["joint_type_evidence"]
    assert evidence["joint_type"] == "revolute"
    assert evidence["log_evidence_difference"] > 0

    # q(t) is the running integral of the interval speeds, starting at zero.
    coordinate = stage1["coordinates"][0]
    assert float(coordinate[0]) == 0.0
    assert float(coordinate[-1]) > 0.0
    for key in ("modes", "coordinates", "labels", "responsibilities", "frame_times"):
        assert key in stage1, key


def test_reassignment_under_exact_joint_transport():
    """Stage 4: the coarse field is replaced by the joint's own transport."""
    from utils.closed_loop_to_joint import (
        integrate_interval_speeds,
        reassign_under_joint_transport,
    )

    state, points, moving, times, _, _ = _converged_state()
    joints = [{
        "joint_type": "revolute",
        "axis": torch.tensor([0.0, 0.0, 1.0]),
        "pivot": torch.zeros(3),
        "pitch": 0.0,
    }]
    result = reassign_under_joint_transport(
        state, joints=joints,
        coordinates=integrate_interval_speeds(state.killing_interval_speeds, times),
        frame_times=times, temperature=0.2, move_activation_cost=0.05,
    )
    accuracy = float(((result["hard_labels"] > 0) == moving).float().mean())
    assert accuracy > 0.9, accuracy
    assert result["joint_velocity"].shape == (1, len(times) - 1, points.shape[0], 3)
    # A prismatic joint transports every point alike, which is a different
    # velocity field and so a different assignment question.
    sliding = [{
        "joint_type": "prismatic",
        "axis": torch.tensor([0.0, 1.0, 0.0]),
        "pivot": None,
        "pitch": 0.0,
    }]
    other = reassign_under_joint_transport(
        state, joints=sliding,
        coordinates=integrate_interval_speeds(state.killing_interval_speeds, times),
        frame_times=times, temperature=0.2,
    )
    assert not torch.equal(other["hard_labels"], result["hard_labels"])


def test_pipeline_runner_resolves_and_plans():
    """The single runner must resolve its own paths without running anything."""
    import run_pipeline

    parser = run_pipeline.build_parser()
    args = parser.parse_args(["--scene", "USB_100109", "--dry-run"])
    paths = run_pipeline.resolve_paths(args)
    for key in ("scene_root", "output", "static", "static_ply", "closed_loop",
                "closed_loop_checkpoint", "joint", "report"):
        assert key in paths, key
    assert paths["scene_name"] == "USB_100109"
    assert paths["scene_root"].is_dir()
    # Every stage is selectable, and an unknown one is refused rather than
    # silently skipped.
    assert run_pipeline.ALL_STAGES == ("0", "1", "2", "3", "4", "5")
    try:
        run_pipeline.main(["--scene", "USB_100109", "--stages", "9", "--dry-run"])
    except SystemExit as error:
        assert "unknown stages" in str(error)
    else:
        raise AssertionError("an unknown stage must be refused")


def test_closed_loop_endpoint_model_auto_discovery():
    """Frame-T endpoint transport should be found next to a frame-0 endpoint run."""
    from argparse import Namespace

    from tools._closed_loop_scene import _endpoint_candidates, _find_model_ply

    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory) / "shared"
        frame0 = root / "frame0_static"
        frame_t = root / "frame_t_static" / "point_cloud" / "iteration_14"
        frame_t.mkdir(parents=True)
        ply = frame_t / "point_cloud.ply"
        ply.write_text("ply\n", encoding="utf-8")

        args = Namespace(static_model_dir=str(frame0), endpoint_static_model_dir="")
        candidates = _endpoint_candidates(args)
        assert root / "frame_t_static" in candidates
        assert _find_model_ply(root / "frame_t_static", 14) == ply.resolve()


def test_ablations_disconnect_what_they_claim_to():
    from tools.train_closed_loop_motion import ABLATIONS

    for name, ablation in ABLATIONS.items():
        assert ablation.get("purpose"), f"{name} must say what it tests"
    assert ABLATIONS["A_diffusion_only"]["use_fm"] is False
    assert ABLATIONS["B_fm_killing_only"]["use_diffusion"] is False
    assert ABLATIONS["C_fixed_prior_killing"]["freeze_assignment"] is True
    assert ABLATIONS["D_no_diffusion_feedback"]["diffusion_into_assignment"] is False
    assert ABLATIONS["E_fixed_killing"]["freeze_killing"] is True
    assert ABLATIONS["F_no_fm_killing_regularization"]["fm_killing_feedback"] is False
    full = ABLATIONS["G_full_closed_loop"]
    assert full["use_fm"] and full["use_diffusion"] and full["fm_killing_feedback"]


def main():
    tests = [value for name, value in sorted(globals().items()) if name.startswith("test_")]
    for test in tests:
        test()
        print(f"ok  {test.__name__}")
    print(f"\n{len(tests)} tests passed")


# --------------------------------------------------------------------------


# --------------------------------------------------------------------------
# spec 28 / 30b / 32 / 40b
# --------------------------------------------------------------------------


def test_ambiguous_components_group_by_adjacency_and_prefer_the_unobserved():
    """Two spatially separate ambiguous blobs must come back as two components,
    and the one the velocity never saw must be tested first."""
    from scene.physics_conditioned_gaussian_denoiser import build_foreground_knn
    from utils.closed_loop_assignment import build_foreground_graph
    from utils.closed_loop_counterfactual import (
        CounterfactualConfig, ambiguous_components,
    )

    generator = torch.Generator().manual_seed(0)
    near = torch.rand(20, 3, generator=generator) * 0.05
    far = torch.rand(20, 3, generator=generator) * 0.05 + torch.tensor([10.0, 0.0, 0.0])
    settled = torch.rand(20, 3, generator=generator) * 0.05 + torch.tensor([0.0, 10.0, 0.0])
    points = torch.cat((near, far, settled))
    share = torch.full((60, 2), 0.5)
    share[40:] = torch.tensor([0.99, 0.01])          # the blob a label already owns
    knn_idx, _ = build_foreground_knn(points, k=6)
    graph = build_foreground_graph(points, knn_idx)

    support = torch.zeros(60)
    support[:20] = 1.0                                # `near` is observed, `far` is not
    components = ambiguous_components(
        responsibilities=share, knn_idx=knn_idx, graph_weight=graph["weight"],
        config=CounterfactualConfig(max_components=8, min_component_size=4),
        observed_support=support,
    )
    assert len(components) == 2, [int(c.numel()) for c in components]
    members = [set(c.tolist()) for c in components]
    assert all(not (m & set(range(40, 60))) for m in members), "a settled blob was tested"
    assert members[0] <= set(range(20, 40)), "the unobserved component was not prioritised"


def test_counterfactual_energy_follows_the_render_and_persists_across_cycles():
    """A component the renderer prefers moving must get lower energy on the
    moving label -- and must keep it on a later cycle it was not re-tested in."""
    from utils.closed_loop_counterfactual import (
        CounterfactualConfig, counterfactual_component_energy,
    )

    state, _, moving, _, _, _ = _converged_state()
    component = torch.nonzero(moving, as_tuple=False).flatten()[:5]
    calls = []

    def render_component_fn(members, label, frame):
        calls.append((int(label), int(frame)))
        return 1.0 if int(label) == 0 else 0.25      # the images prefer "moving"

    config = CounterfactualConfig(max_components=4, min_component_size=1,
                                  frames_per_component=2, momentum=0.5)
    first = counterfactual_component_energy(
        state=state, components=[component], target_frames=[1, 2],
        render_component_fn=render_component_fn, config=config, cycle=0,
    )
    energy = first["energy"]
    assert energy.shape == (state.num_foreground, state.num_labels)
    assert torch.allclose(energy[component, 1], torch.zeros(5)), "moving is not the cheapest"
    assert torch.all(energy[component, 0] > 0.7), energy[component, 0]
    untested = torch.ones(state.num_foreground, dtype=torch.bool)
    untested[component] = False
    assert torch.all(energy[untested] == 0.0), "an untested Gaussian voted anyway"
    assert len(calls) == 4, calls

    # A cycle that tests nothing must still return the verdict from cycle 0.
    second = counterfactual_component_energy(
        state=state, components=[], target_frames=[1, 2],
        render_component_fn=render_component_fn,
        config=config, persistent=first["persistent"], cycle=1,
    )
    assert torch.allclose(second["energy"][component, 0], energy[component, 0])


def test_counterfactual_energy_reaches_a_gaussian_the_velocity_never_saw():
    """The whole point of spec 28: a flow-invisible Gaussian gets a real vote."""
    from utils.closed_loop_assignment import (
        AssignmentWeights, robust_normalize_and_combine,
    )
    from utils.closed_loop_counterfactual import (
        CounterfactualConfig, counterfactual_component_energy,
    )

    state, _, moving, _, _, _ = _converged_state()
    invisible = torch.nonzero(moving, as_tuple=False).flatten()[:5]
    state.observed_valid[:, invisible] = False
    # No per-Gaussian modality can separate the labels here: this is the state a
    # flow-invisible moving Gaussian is actually in.
    observed = torch.zeros((state.num_foreground, state.num_labels))

    result = counterfactual_component_energy(
        state=state, components=[invisible], target_frames=[1],
        render_component_fn=lambda m, label, f: 1.0 if int(label) == 0 else 0.0,
        config=CounterfactualConfig(min_component_size=1, frames_per_component=1),
        cycle=0,
    )
    combined = robust_normalize_and_combine(
        energy_diffusion=None, energy_fm=None, energy_observed=observed,
        energy_graph=None, energy_counterfactual=result["energy"],
        weights=AssignmentWeights(diffusion=0.0, flow_matching=0.0, observed=1.0,
                                  graph=0.0, counterfactual=1.0),
    )
    total = combined["energy_total"]
    assert total[invisible].argmin(-1).eq(1).all(), (
        "the counterfactual vote did not survive normalisation: "
        f"{total[invisible][0].tolist()}"
    )


def test_leave_component_out_rejects_a_component_that_does_not_belong():
    """Fitting on a component and then observing that it fits is not evidence.
    A component moving the wrong way must fail the held-out check that a genuine
    piece of the same part passes."""
    from utils.closed_loop_counterfactual import leave_component_out_validation

    state, _, moving, _, _, _ = _converged_state()
    rows = torch.nonzero(moving, as_tuple=False).flatten()
    generator = torch.Generator().manual_seed(3)
    # Without a little noise both residuals are numerically zero and the ratio
    # measures nothing but rounding.
    state.consensus_velocity = state.consensus_velocity + torch.randn(
        state.consensus_velocity.shape, generator=generator
    ) * 0.01
    genuine, intruder = rows[:4], rows[4:8]
    state.consensus_velocity[:, intruder] = -state.consensus_velocity[:, intruder] * 4.0

    kwargs = dict(canonical_fg=state.canonical_fg,
                  consensus_velocity=state.consensus_velocity,
                  responsibilities=state.responsibilities,
                  core_masks=state.core_masks, part=0,
                  solver_kwargs={"min_valid_cells": 8})
    good = leave_component_out_validation(component=genuine, **kwargs)
    bad = leave_component_out_validation(component=intruder, **kwargs)
    assert good["available"] and bad["available"], (good, bad)
    assert good["agreement_ratio"] < bad["agreement_ratio"], (
        f"the held-out check cannot tell them apart: {good['agreement_ratio']:.3f} "
        f"vs {bad['agreement_ratio']:.3f}"
    )
    assert good["improvement_over_static"] > 1.0, good
    assert bad["improvement_over_static"] < good["improvement_over_static"], bad


def test_leave_component_out_declines_when_it_would_gut_the_core():
    """Excluding the component must leave enough of the core to fit at all; a
    fit on three points that then 'validates' would be worse than no check."""
    from utils.closed_loop_counterfactual import leave_component_out_validation

    state, _, moving, _, _, _ = _converged_state()
    everything = torch.nonzero(moving, as_tuple=False).flatten()
    result = leave_component_out_validation(
        canonical_fg=state.canonical_fg, consensus_velocity=state.consensus_velocity,
        responsibilities=state.responsibilities, core_masks=state.core_masks,
        component=everything, part=0, solver_kwargs={"min_valid_cells": 8},
    )
    assert result["available"] is False
    assert result["reason"] == "core_too_small_without_the_component"


def test_target_frames_are_deterministic_then_resampled():
    """Spec 32: the early cycles must be reproducible, the later ones must not
    keep optimising the same five frames."""
    from tools.train_closed_loop_motion import default_target_frames, sample_target_frames

    generator = torch.Generator().manual_seed(0)
    fixed = default_target_frames(60, 5)
    assert sample_target_frames(60, 5, generator, deterministic=True) == fixed
    draws = {tuple(sample_target_frames(60, 5, generator, deterministic=False))
             for _ in range(8)}
    assert len(draws) > 1, "sampling returned the same frames every time"
    for draw in draws:
        assert len(set(draw)) == 5 and all(0 <= f < 60 for f in draw), draw


def test_failure_response_only_tightens_and_reports_every_change():
    """Spec 40b: a detected failure must change something, in the direction the
    specification names, and say so."""
    from utils.closed_loop_assignment import AssignmentWeights
    from utils.closed_loop_debug import respond_to_failure_modes
    from utils.fm_killing_consensus import ConsensusWeights, CoreShellConfig

    weights = AssignmentWeights(move_activation_cost=0.0)
    consensus = ConsensusWeights(static=1.0, gamma=2.0)
    core = CoreShellConfig()
    before = (consensus.static, core.enter)
    response = respond_to_failure_modes(
        {"warnings": ["everything_moves: ..."]}, weights, consensus, core,
        log=lambda *_: None,
    )
    assert response["applied"], "a detected failure produced no response"
    assert weights.move_activation_cost > 0.0
    assert consensus.static > before[0] and core.enter > before[1]
    assert respond_to_failure_modes(
        {"warnings": []}, weights, consensus, core, log=lambda *_: None
    )["applied"] == []


def test_joint_refinement_keeps_the_gaussians_only_the_graph_can_see():
    """Spec 41 step 5 says recompute the K+1 energies, plural.

    Measured on USB, deciding this refinement on the velocity observation alone
    re-imposed that modality's coverage ceiling and shed correctly-labelled
    moving Gaussians (F1 0.379 -> 0.289).  This pins the mechanism: a moving
    Gaussian with no velocity observation must survive the refinement when the
    graph agrees with its neighbours, and must not when only velocity votes.
    """
    from scene.physics_conditioned_gaussian_denoiser import build_foreground_knn
    from utils.closed_loop_assignment import build_foreground_graph
    from utils.closed_loop_to_joint import reassign_under_joint_transport

    state, points, moving, times, _, _ = _converged_state()
    rows = torch.nonzero(moving, as_tuple=False).flatten()
    invisible = rows[: max(4, rows.numel() // 4)]
    # These moving Gaussians have no velocity evidence at all.
    state.observed_valid[:, invisible] = False
    state.observed_velocity[:, invisible] = 0.0
    state.fm_confidence = torch.zeros_like(state.fm_confidence)

    joints = [{
        "joint_type": "revolute",
        "axis": AXIS.tolist(),
        "pivot": [0.0, 0.0, 0.0],
    }]
    coordinates = torch.zeros(1, len(times), dtype=torch.float64)
    coordinates[0] = torch.cumsum(
        torch.tensor([0.0] + [0.8 / 30.0] * (len(times) - 1), dtype=torch.float64), 0
    )
    knn_idx, _ = build_foreground_knn(points, k=8)
    graph = build_foreground_graph(points, knn_idx)

    velocity_only = reassign_under_joint_transport(
        state, joints=joints, coordinates=coordinates, frame_times=times,
        temperature=0.2,
    )
    every_modality = reassign_under_joint_transport(
        state, joints=joints, coordinates=coordinates, frame_times=times,
        temperature=0.2, graph=graph,
    )
    assert velocity_only["modalities_used"] == ["observed_velocity"], (
        velocity_only["modalities_used"]
    )
    assert "graph" in every_modality["modalities_used"], every_modality["modalities_used"]

    kept_velocity = int((velocity_only["hard_labels"][invisible] > 0).sum())
    kept_all = int((every_modality["hard_labels"][invisible] > 0).sum())
    assert kept_all > kept_velocity, (
        "the graph did not rescue any unobserved moving Gaussian: "
        f"velocity-only kept {kept_velocity}, every modality kept {kept_all} "
        f"of {invisible.numel()}"
    )


def test_joint_transport_is_the_exact_transform_not_a_finite_difference():
    """The transport installed by spec 41 step 3 must be the joint's own
    rotation, so a hinge coordinate of theta puts a point exactly where a
    rotation by theta puts it."""
    from utils.closed_loop_to_joint import (
        joint_transport_positions, joint_transport_velocity,
    )

    points = torch.tensor([[1.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.5, 0.5, 1.0]])
    angle = 0.7
    coordinates = torch.tensor([[0.0, angle]], dtype=torch.float64)
    positions = joint_transport_positions(
        points=points,
        joints=[{"joint_type": "revolute", "axis": [0.0, 0.0, 1.0],
                 "pivot": [0.0, 0.0, 0.0]}],
        coordinates=coordinates,
    )
    cosine, sine = math.cos(angle), math.sin(angle)
    expected = torch.stack((
        points[:, 0] * cosine - points[:, 1] * sine,
        points[:, 0] * sine + points[:, 1] * cosine,
        points[:, 2],
    ), dim=-1)
    assert torch.allclose(positions[0, 0], points, atol=1e-6), "frame 0 must be canonical"
    assert torch.allclose(positions[0, 1], expected, atol=1e-5), positions[0, 1]
    # A rotation about +z through the origin cannot change |xy| or z.
    radius = positions[0, :, :, :2].norm(dim=-1)
    assert torch.allclose(radius[0], radius[1], atol=1e-5)

    velocity = joint_transport_velocity(positions=positions, frame_times=[0.0, 0.5])
    assert velocity.shape == (1, 1, 3, 3)
    assert torch.allclose(velocity[0, 0], (expected - points) / 0.5, atol=1e-5)


if __name__ == "__main__":
    main()
