#!/usr/bin/env python3
"""Closed-loop diffusion + flow-matching + Killing motion discovery.

The loop this drives is the point of the pipeline:

    r -> FM -> Killing -> T_k(t) -> physical proposal -> residual diffusion
      -> K+1 energies -> r -> (next FM and Killing fit)

with two extra arrows that keep it from being four independent stages bolted
together: the Killing field regularises the FM objective, and the
responsibilities weight both the FM objective and the Killing fit.

Run it against a trained canonical 3DGS:

    python tools/train_closed_loop_motion.py \\
        --root-scene input/USB_100109 \\
        --static-model-dir output/USB_100109/shared/frame0_static \\
        --static-iteration 14000 \\
        --output-dir output/USB_100109_closed_loop \\
        --num-moving-parts 1
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parent.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from scene.physics_conditioned_gaussian_denoiser import (  # noqa: E402
    DenoiserConfig,
    PhysicsConditionedGaussianDenoiser,
    PhysicsFeatureSpec,
    build_foreground_knn,
)
from utils.closed_loop_assignment import (  # noqa: E402
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
from utils.closed_loop_counterfactual import (  # noqa: E402
    CounterfactualConfig,
    ambiguous_components,
    build_component_render_fn,
    counterfactual_component_energy,
    leave_component_out_validation,
)
from utils.closed_loop_debug import (  # noqa: E402
    collect_diagnostics,
    detect_failure_modes,
    format_diagnostics_line,
    respond_to_failure_modes,
    save_closed_loop_debug,
    write_closed_loop_html,
)
from utils.closed_loop_flow_matching import (  # noqa: E402
    TemporalFlowConfig,
    TemporalFlowField,
    train_temporal_flow_field,
)
from utils.closed_loop_motion_losses import FlowMatchingFeedbackWeights  # noqa: E402
from utils.closed_loop_motion_state import (  # noqa: E402
    PIPELINE_NAME,
    SCHEMA_VERSION,
    allocate_closed_loop_state,
    compute_entropy,
)
from utils.fm_killing_consensus import (  # noqa: E402
    ConsensusWeights,
    bootstrap_moving_core_from_observations,
    CoreShellConfig,
    build_discrete_physical_proposal,
    compute_candidate_physical_displacements,
    compute_consensus_velocity,
    compute_fm_physical_endpoint_energy,
    cross_fit_core_validation,
    integrate_killing_positions,
    integrate_killing_transforms,
    predict_all_killing_velocities,
    refit_killing_fields,
    update_core_shell_hysteresis,
)
from utils.physics_conditioned_diffusion import (  # noqa: E402
    CorruptionConfig,
    bootstrap_responsibilities_from_neutral_denoiser,
    DiffusionConfig,
    train_physics_conditioned_denoiser,
    decompose_residual_against_field,
    residual_speed_correction,
    run_physics_conditioned_diffusion,
)


#: Which feedback arrows each ablation leaves connected (spec 43).  Every entry
#: names what it is testing, because an ablation whose purpose is not written
#: down gets re-run with a different meaning six months later.
ABLATIONS = {
    "A_diffusion_only": {
        "use_fm": False, "use_observed": False, "use_diffusion": True,
        "fm_killing_feedback": False, "use_counterfactual": False,
        "purpose": "can the residual critic assign labels with no motion evidence at all",
    },
    "B_fm_killing_only": {
        "use_fm": True, "use_observed": True, "use_diffusion": False,
        "fm_killing_feedback": True, "use_counterfactual": False,
        "purpose": "how far transport plus rigid physics gets without the critic",
    },
    "C_fixed_prior_killing": {
        "use_fm": True, "use_observed": True, "use_diffusion": False,
        "fm_killing_feedback": True, "freeze_assignment": True,
        "purpose": "the old sequential pipeline: assign once, then fit physics",
    },
    "D_no_diffusion_feedback": {
        "use_fm": True, "use_observed": True, "use_diffusion": True,
        "fm_killing_feedback": True, "diffusion_into_assignment": False,
        "purpose": "run the critic but do not let it vote, isolating its contribution",
    },
    "E_fixed_killing": {
        "use_fm": True, "use_observed": True, "use_diffusion": True,
        "fm_killing_feedback": True, "freeze_killing": True,
        "purpose": "does reassignment help when the physics cannot be refitted",
    },
    "H_no_counterfactual": {
        "use_fm": True, "use_observed": True, "use_diffusion": True,
        "fm_killing_feedback": True, "use_counterfactual": False,
        "purpose": "is component-level rendering what reaches the flow-invisible Gaussians",
    },
    "F_no_fm_killing_regularization": {
        "use_fm": True, "use_observed": True, "use_diffusion": True,
        "fm_killing_feedback": False,
        "purpose": "is the Killing -> FM arrow load-bearing or is it just collusion",
    },
    "G_full_closed_loop": {
        "use_fm": True, "use_observed": True, "use_diffusion": True,
        "fm_killing_feedback": True,
        "purpose": "everything connected",
    },
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    data = parser.add_argument_group("data")
    data.add_argument("--root-scene", type=str, required=True)
    data.add_argument("--static-model-dir", type=str, required=True)
    data.add_argument("--static-iteration", type=int, default=14000)
    data.add_argument("--output-dir", type=str, required=True)
    data.add_argument("--frames", type=str, default="all",
                      help="'all', 'stride:N', or an explicit comma list of frame ids")
    data.add_argument("--target-frames", type=str, default="",
                      help="frames the diffusion renders against; default is a spread over the sequence")
    data.add_argument("--device", type=str, default="cuda")
    data.add_argument("--fps", type=float, default=30.0)
    data.add_argument("--semantic-dynamic-threshold", type=float, default=0.5,
                      help="semantic probability above which a Gaussian is foreground")
    data.add_argument("--semantic-dynamic-min-points", type=int, default=32)
    data.add_argument("--allow-all-gaussian-foreground", action="store_true",
                      help="run over the whole cloud when the model has no semantic field")
    data.add_argument("--correspondence-flow-max-side", type=int, default=512)
    data.add_argument("--endpoint-static-model-dir", type=str, default="",
                      help="Trained final-frame static 3DGS used to initialise FM endpoint transport. "
                           "If omitted, common static-endpoint locations next to --static-model-dir are tried.")
    data.add_argument("--endpoint-static-iteration", type=int, default=0,
                      help="Iteration of --endpoint-static-model-dir; 0 reuses --static-iteration.")
    data.add_argument("--no-endpoint-fm", action="store_true",
                      help="Disable endpoint 3DGS transport initialisation and fall back to rasterized observations.")
    data.add_argument("--endpoint-transport-knn", type=int, default=16)
    data.add_argument("--endpoint-transport-temperature", type=float, default=0.02)
    data.add_argument("--endpoint-transport-robust-delta", type=float, default=0.02)
    data.add_argument("--endpoint-transport-confidence-distance-scale", type=float, default=0.0)
    data.add_argument("--endpoint-transport-confidence-entropy-scale", type=float, default=1.0)
    data.add_argument("--endpoint-transport-chunk-size", type=int, default=1024)
    data.add_argument("--endpoint-motion-scale", type=str, default="local_knn",
                      choices=("local_knn", "global_nn", "gaussian_scale"))

    loop = parser.add_argument_group("loop")
    loop.add_argument("--num-moving-parts", type=int, default=1)
    loop.add_argument("--outer-iterations", type=int, default=8)
    loop.add_argument("--inner-physical-iterations", type=int, default=2)
    loop.add_argument("--ablation", type=str, default="G_full_closed_loop", choices=sorted(ABLATIONS))
    loop.add_argument("--convergence-responsibility-delta", type=float, default=1.0e-3)
    loop.add_argument("--static-prior", type=float, default=0.7)
    loop.add_argument("--seed", type=int, default=20_260_910)

    weights = parser.add_argument_group("weights (spec 36 debugging defaults)")
    weights.add_argument("--lambda-fm-base", type=float, default=1.0)
    weights.add_argument("--lambda-fm-killing", type=float, default=0.1)
    weights.add_argument("--lambda-fm-static", type=float, default=0.1)
    weights.add_argument("--lambda-consensus-fm", type=float, default=1.0)
    weights.add_argument("--lambda-consensus-obs", type=float, default=1.0)
    weights.add_argument("--lambda-consensus-killing", type=float, default=1.0)
    weights.add_argument("--lambda-consensus-static", type=float, default=1.0)
    weights.add_argument("--lambda-assignment-diff", type=float, default=1.0)
    weights.add_argument("--lambda-assignment-fm", type=float, default=1.0)
    weights.add_argument("--lambda-assignment-obs", type=float, default=1.0)
    weights.add_argument("--lambda-assignment-graph", type=float, default=0.05)
    weights.add_argument("--responsibility-fit-gamma", type=float, default=2.0)
    weights.add_argument("--move-activation-cost", type=float, default=0.0)

    core = parser.add_argument_group("core / shell")
    core.add_argument("--tau-core-enter", type=float, default=0.90)
    core.add_argument("--tau-core-exit", type=float, default=0.30)
    core.add_argument("--core-persistence-cycles", type=int, default=2)
    core.add_argument("--cross-fit-validation", action="store_true")
    core.add_argument("--leave-component-out", action="store_true",
                      help="spec 30: validate a component against a field fitted without it")

    counter = parser.add_argument_group("counterfactual component rendering (spec 28)")
    counter.add_argument("--counterfactual-max-components", type=int, default=8,
                         help="0 disables component-level rendering")
    counter.add_argument("--counterfactual-min-component-size", type=int, default=4)
    counter.add_argument("--counterfactual-frames-per-component", type=int, default=2)
    counter.add_argument("--lambda-assignment-counterfactual", type=float, default=1.0)
    counter.add_argument("--deterministic-target-cycles", type=int, default=2,
                         help="cycles before target frames are resampled (spec 32)")
    counter.add_argument("--adapt-to-failure-modes", action="store_true",
                         help="spec 40: act on detected failures, not only report them")

    schedule = parser.add_argument_group("temperature")
    schedule.add_argument("--assignment-temperature-schedule", type=str,
                          default="1.0,0.8,0.6,0.45,0.3,0.2,0.15")

    denoiser = parser.add_argument_group("denoiser")
    denoiser.add_argument("--denoiser-hidden-dim", type=int, default=128)
    denoiser.add_argument("--denoiser-message-layers", type=int, default=3)
    denoiser.add_argument("--denoiser-knn", type=int, default=16)
    denoiser.add_argument("--diffusion-trials", type=int, default=8)
    denoiser.add_argument("--diffusion-steps", type=int, default=6)
    denoiser.add_argument("--diffusion-noise-sigma-ratio", type=float, default=0.15)
    # Stage A of spec 31.  An untrained critic answers "no correction wanted" to
    # everything, which the transition energy reads as a full-weight vote for
    # the current label; on USB that collapsed all 11263 Gaussians to static.
    # So the critic is trained on deliberately corrupted hypotheses before it
    # is allowed to vote, and its vote ramps in over the first cycles.
    denoiser.add_argument("--denoiser-train-iterations", type=int, default=150)
    denoiser.add_argument("--denoiser-learning-rate", type=float, default=1.0e-3)
    denoiser.add_argument("--neutral-bootstrap-iterations", type=int, default=200,
                          help="spec 5.1 physics-neutral pass that initialises the responsibilities")
    denoiser.add_argument("--diffusion-vote-warmup-cycles", type=int, default=2,
                          help="cycles over which the critic's assignment weight ramps from zero")

    flow = parser.add_argument_group("flow matching")
    flow.add_argument("--fm-init", choices=("legacy", "current_knn"), default="legacy",
                      help="legacy restores endpoint FlowMLP initialization; current_knn keeps the old local-KNN ablation")
    flow.add_argument("--fm-residual-refinement", action=argparse.BooleanOptionalAction, default=True)
    flow.add_argument("--fm-hidden-dim", type=int, default=128)
    flow.add_argument("--fm-layers", type=int, default=4)
    flow.add_argument("--fm-iterations", type=int, default=200)
    flow.add_argument("--fm-learning-rate", type=float, default=1.0e-3)
    flow.add_argument("--fm-prior-prob-bias", type=float, default=1.0)
    flow.add_argument("--fm-prior-prob-temperature", type=float, default=0.35)
    flow.add_argument("--legacy-fm-iters", type=int, default=1000)
    flow.add_argument("--legacy-fm-batch-size", type=int, default=2048)
    flow.add_argument("--legacy-fm-integration-steps", type=int, default=16)
    flow.add_argument("--legacy-fm-integrator", choices=("euler", "rk4"), default="euler")
    flow.add_argument("--legacy-fm-identity-weight", type=float, default=0.01)
    flow.add_argument("--legacy-fm-smooth-weight", type=float, default=0.0)
    flow.add_argument("--force-legacy-fm", action="store_true")
    flow.add_argument("--fm-base-preservation-weight", type=float, default=1.0)
    flow.add_argument("--fm-base-preservation-start", type=float, default=1.0)
    flow.add_argument("--fm-base-preservation-end", type=float, default=0.25)
    flow.add_argument("--fm-physical-3d-weight", type=float, default=0.25)
    flow.add_argument("--fm-rendered-flow-weight", type=float, default=0.25)
    flow.add_argument("--observed-rendered-flow-weight", type=float, default=0.25)
    flow.add_argument("--target-physical-render-weight", type=float, default=0.05)
    flow.add_argument("--fm-feedback-start-cycle", type=int, default=1)
    flow.add_argument("--fm-feedback-ramp-cycles", type=int, default=3)
    flow.add_argument("--fm-max-residual-step", type=float, default=0.25,
                      help="trust-region radius as a multiple of the base FM/scene motion scale; 0 disables")
    flow.add_argument("--rendered-flow-min-support", type=float, default=1.0e-4)
    flow.add_argument("--rendered-flow-min-gap", type=float, default=1.0e-8)
    flow.add_argument("--rendered-flow-cameras-per-frame", type=int, default=1)
    flow.add_argument("--save-rendered-flow-debug", action="store_true")

    stage = parser.add_argument_group("staged activation (spec 31)")
    stage.add_argument("--physical-inner-iterations", dest="inner_physical_iterations",
                       type=int, default=argparse.SUPPRESS)
    stage.add_argument("--physics-alpha-schedule", type=str, default="0.25,0.5,0.75,1.0",
                       help="fraction of the Killing transport the critic is shown per cycle")
    stage.add_argument("--speed-correction-rate", type=float, default=0.0,
                       help="0 disables the residual-driven speed nudge of spec 27")
    return parser


def parse_frame_spec(spec: str, available: list[int]) -> list[int]:
    text = str(spec or "all").strip().lower()
    if text in ("", "all"):
        return list(available)
    if text.startswith("stride:"):
        stride = max(1, int(text.split(":", 1)[1]))
        chosen = sorted(set(available[::stride] + [available[-1]]))
        return chosen
    wanted = [int(item) for item in text.replace(";", ",").split(",") if item.strip()]
    missing = sorted(set(wanted) - set(available))
    if missing:
        raise ValueError(f"frames {missing} are not complete multi-view frames")
    return sorted(set(wanted))


def sample_target_frames(num_frames: int, count: int, generator, deterministic: bool) -> list[int]:
    """Spec 32: temporally diverse targets, deterministic first then random.

    A single target frame teaches the critic that frame's particular occlusions.
    The canonical assignment is time-invariant, so the targets have to span the
    sequence for the residual to be about the label rather than the view.  Early
    cycles use the fixed spread so runs are comparable; later cycles resample,
    which is what stops the critic from overfitting the same few views.
    """
    if num_frames <= 2:
        return list(range(1, num_frames))
    if deterministic:
        return default_target_frames(num_frames, count)
    available = torch.arange(1, num_frames)
    picks = torch.randperm(available.numel(), generator=generator)[: min(count, available.numel())]
    return sorted(int(available[index]) for index in picks)


def default_target_frames(num_frames: int, count: int = 5) -> list[int]:
    """A temporally diverse default rather than one target frame (spec 32).

    Training the critic against a single frame teaches it that frame's
    particular occlusions; the canonical assignment is time-invariant, so the
    targets have to span the sequence for the residual to be about the label
    rather than about the view.
    """
    if num_frames <= 2:
        return list(range(1, num_frames))
    positions = torch.linspace(1, num_frames - 1, min(count, num_frames - 1))
    return sorted({int(round(float(value))) for value in positions})


def schedule_value(schedule: list[float], index: int) -> float:
    return float(schedule[min(int(index), len(schedule) - 1)])


def _integrate_velocity_endpoint(velocity: torch.Tensor, frame_times) -> torch.Tensor:
    dt = torch.diff(torch.tensor(frame_times, device=velocity.device, dtype=velocity.dtype))
    return (velocity * dt[:, None, None]).sum(0)


def _initialise_responsibilities_from_fm(state, probability: torch.Tensor, valid: torch.Tensor | None) -> None:
    """Use FM only for static-vs-moving mass; never for K identity."""
    p = torch.as_tensor(probability, device=state.canonical_fg.device, dtype=state.canonical_fg.dtype).reshape(-1)
    if p.numel() != state.num_foreground:
        raise ValueError("FM probability must match foreground count")
    if valid is not None:
        mask = torch.as_tensor(valid, device=p.device).bool().reshape(-1)
        p = torch.where(mask, p, state.responsibilities[:, 1:].sum(-1))
    p = p.clamp(1.0e-4, 1.0 - 1.0e-4)
    state.responsibilities[:, 0] = 1.0 - p
    state.responsibilities[:, 1:] = p[:, None] / float(state.num_moving_parts)
    state.hard_labels = state.responsibilities.argmax(-1)
    state.assignment_entropy = compute_entropy(state.responsibilities)


def _stats(name: str, value: torch.Tensor) -> dict[str, float]:
    flat = torch.as_tensor(value).detach().cpu().float().reshape(-1)
    if flat.numel() == 0:
        return {f"{name}_{key}": 0.0 for key in ("mean", "median", "p10", "p90")}
    return {
        f"{name}_mean": float(flat.mean()),
        f"{name}_median": float(flat.median()),
        f"{name}_p10": float(torch.quantile(flat, 0.10)),
        f"{name}_p90": float(torch.quantile(flat, 0.90)),
    }


def _write_fm_initialization_check(state, args, output_root: Path) -> None:
    probability = state.fm_base_probability.detach().cpu().float()
    labels = (probability >= 0.5).long()
    confidence = state.fm_base_confidence.mean(0).detach().cpu().float()
    payload = {
        "fm_init": str(getattr(args, "fm_init", "")),
        "ground_truth_used_as_algorithm_input": False,
        "num_foreground": state.num_foreground,
        **_stats("displacement_norm", state.fm_base_displacement.norm(dim=-1)),
        **_stats("velocity_norm", state.fm_base_velocity.norm(dim=-1)),
        "mean_dynamic_probability": float(probability.mean()),
        "median_dynamic_probability": float(probability.median()),
        "mean_confidence": float(confidence.mean()),
        "median_confidence": float(confidence.median()),
        "dynamic_fraction": float(labels.float().mean()),
    }
    parts = Path(args.root_scene).expanduser().resolve() / "pointcloud" / "points3D_parts.ply"
    if parts.is_file():
        try:
            from tools.evaluate_closed_loop_motion import _ground_truth_moving, score_assignment

            gt = _ground_truth_moving(state.canonical_fg.detach().cpu().double(), parts)
            payload.update(score_assignment(labels, gt))
            payload["gt_moving"] = int(gt.sum())
        except Exception as exc:  # noqa: BLE001 - debug-only GT check
            payload["gt_check_error"] = str(exc)
    path = output_root / "fm_legacy_initialization_check.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, allow_nan=False), encoding="utf-8")


def _physical_velocity_from_labels(state) -> torch.Tensor:
    velocity = torch.zeros_like(state.fm_velocity)
    labels = state.hard_labels
    for part in range(state.num_moving_parts):
        mask = labels == (part + 1)
        if bool(mask.any()):
            velocity[:, mask] = state.killing_velocity[part, :, mask]
    return velocity


def _static_feedback_gate(state) -> torch.Tensor:
    support = state.observed_confidence.sum(0)
    speed = (
        state.observed_confidence * state.observed_velocity.norm(dim=-1)
    ).sum(0) / support.clamp_min(1.0e-8)
    observed = support > 0
    still = speed <= speed[observed].median() if bool(observed.any()) else torch.zeros_like(observed)
    gate = (state.responsibilities[:, 0] > 0.9) & state.static_core_mask & observed & still
    return gate[None].expand(state.num_intervals, -1).to(state.fm_velocity)


def _feedback_ramp(outer: int, start: int, length: int) -> float:
    if outer < int(start):
        return 0.0
    return min(1.0, (outer - int(start) + 1) / float(max(1, int(length))))


def _fm_trust_radius(state, neighbour_scale: float, args) -> float:
    ratio = float(getattr(args, "fm_max_residual_step", 0.0))
    if ratio <= 0.0:
        return 0.0
    base = state.fm_base_velocity.norm(dim=-1)
    scale = float(base[base > 0].median()) if bool((base > 0).any()) else float(neighbour_scale)
    return ratio * max(scale, float(neighbour_scale), 1.0e-8)


def run_closed_loop(
    *,
    state,
    args,
    ablation: dict,
    denoiser,
    spec,
    flow_field,
    knn_idx,
    neighbour_scale,
    graph,
    output_root: Path,
    render_evidence_fn=None,
    render_loss_fn=None,
    observation_confidence_fn=None,
    render_transport_feedback_fn=None,
    fm_base_velocity: torch.Tensor | None = None,
    fm_base_confidence: torch.Tensor | None = None,
    log=print,
) -> dict:
    """The outer loop of spec 34, one function so the whole cycle is readable.

    Block-coordinate throughout (spec 29): every arrow crosses a ``detach``, so a
    failure is attributable to one module rather than smeared across a single
    differentiable graph that trained itself into agreement.
    """
    temperature_schedule = [float(v) for v in args.assignment_temperature_schedule.split(",")]
    alpha_schedule = [float(v) for v in args.physics_alpha_schedule.split(",")]
    consensus_weights = ConsensusWeights(
        fm=args.lambda_consensus_fm, observed=args.lambda_consensus_obs,
        killing=args.lambda_consensus_killing, static=args.lambda_consensus_static,
        gamma=args.responsibility_fit_gamma,
    )
    core_config = CoreShellConfig(
        enter=args.tau_core_enter, exit=args.tau_core_exit,
        persistence_cycles=args.core_persistence_cycles,
    )
    diffusion_weight = args.lambda_assignment_diff if ablation.get("use_diffusion", True) else 0.0
    assignment_weights = AssignmentWeights(
        diffusion=diffusion_weight,
        flow_matching=args.lambda_assignment_fm if ablation.get("use_fm", True) else 0.0,
        fm_3d=args.fm_physical_3d_weight if ablation.get("use_fm", True) else 0.0,
        fm_render=args.fm_rendered_flow_weight if ablation.get("use_fm", True) else 0.0,
        observed=args.lambda_assignment_obs if ablation.get("use_observed", True) else 0.0,
        obs_render=args.observed_rendered_flow_weight if ablation.get("use_observed", True) else 0.0,
        target_render=args.target_physical_render_weight,
        graph=args.lambda_assignment_graph,
        counterfactual=args.lambda_assignment_counterfactual,
        move_activation_cost=args.move_activation_cost,
    )
    if not ablation.get("diffusion_into_assignment", True):
        assignment_weights.diffusion = 0.0
    if not ablation.get("use_counterfactual", True):
        assignment_weights.counterfactual = 0.0
    explicit_targets = (
        parse_frame_spec(args.target_frames, list(range(state.num_frames)))
        if args.target_frames
        else None
    )
    target_generator = torch.Generator().manual_seed(int(args.seed) + 31)
    target_frames = explicit_targets or default_target_frames(state.num_frames)
    counterfactual_config = CounterfactualConfig(
        max_components=int(args.counterfactual_max_components),
        min_component_size=int(args.counterfactual_min_component_size),
        frames_per_component=int(args.counterfactual_frames_per_component),
        enabled=bool(args.counterfactual_max_components > 0)
        and ablation.get("use_counterfactual", True),
    )
    counterfactual_state = None
    solver_kwargs = {"min_valid_cells": 8}
    records = []
    base_velocity = (
        state.fm_velocity.detach()
        if fm_base_velocity is None
        else torch.as_tensor(fm_base_velocity, device=state.canonical_fg.device, dtype=state.canonical_fg.dtype).detach()
    )
    base_confidence = (
        state.fm_confidence.detach()
        if fm_base_confidence is None
        else torch.as_tensor(fm_base_confidence, device=state.canonical_fg.device, dtype=state.canonical_fg.dtype).detach()
    )

    # Spec 5.1, before anything else.  The renderer covers far more Gaussians
    # than the multiview flow does, so a physics-neutral pass around canonical
    # is the only initialisation with full coverage.  Without it the loop starts
    # from a flat prior and every Gaussian the flow never saw is locked static
    # by the move-activation cost.
    if (
        ablation.get("use_diffusion", True)
        and denoiser is not None
        and render_loss_fn is not None
        and int(args.neutral_bootstrap_iterations) > 0
        and str(getattr(args, "fm_init", "")) != "legacy"
    ):
        log(f"[neutral bootstrap] spec 5.1, {args.neutral_bootstrap_iterations} steps")
        neutral = bootstrap_responsibilities_from_neutral_denoiser(
            denoiser, spec=spec, state=state, target_frames=target_frames,
            knn_idx=knn_idx, neighbour_scale=neighbour_scale,
            config=DiffusionConfig(
                num_trials=args.diffusion_trials, denoise_steps=args.diffusion_steps,
                noise_sigma_ratio=args.diffusion_noise_sigma_ratio, seed=int(args.seed),
            ),
            render_loss_fn=render_loss_fn,
            render_evidence_fn=render_evidence_fn,
            observation_confidence_fn=observation_confidence_fn,
            iterations=int(args.neutral_bootstrap_iterations),
            learning_rate=float(args.denoiser_learning_rate),
            static_prior=float(args.static_prior),
            num_moving_parts=state.num_moving_parts,
            log=log,
        )
        if neutral["reason"] is None:
            state.responsibilities = neutral["responsibilities"].detach()
            state.hard_labels = state.responsibilities.argmax(-1)
            state.assignment_entropy = compute_entropy(state.responsibilities)
            state.validate()
            log(f"[neutral bootstrap] residual scale {neutral['scale']:.5g}; "
                f"initial moving probability mean "
                f"{float(state.responsibilities[:, 1:].sum(-1).mean()):.4f}, "
                f"argmax-moving {int((state.hard_labels > 0).sum())}/{state.num_foreground}")
        else:
            log(f"[neutral bootstrap] skipped: {neutral['reason']}")
    elif str(getattr(args, "fm_init", "")) == "legacy" and int(args.neutral_bootstrap_iterations) > 0:
        log("[neutral bootstrap] skipped: legacy FM initializes the K+1 static/moving mass")

    # Stage B of spec 31.  The core admits only confident points, confidence
    # needs a field to be confident about, and the field is fitted from the
    # core: nothing can start.  The observed velocity is the one piece of
    # evidence that predates every hypothesis, so the first core is read off it
    # and the hysteresis takes over from the next cycle.
    seed = bootstrap_moving_core_from_observations(
        observed_velocity=state.observed_velocity,
        observed_confidence=state.observed_confidence,
        observed_valid=state.observed_valid,
        canonical_fg=state.canonical_fg,
        num_moving_parts=state.num_moving_parts,
    )
    fm_core = (
        (state.fm_base_probability > 0.75)
        & (state.fm_base_confidence.mean(0) > 0.05)
        & state.fm_base_valid
    )
    if bool(fm_core.any()):
        if state.num_moving_parts == 1:
            state.core_masks[0] |= fm_core
        else:
            # FM only says moving-vs-static; split the initial mass evenly until
            # the velocity/Killing evidence can separate identities.
            ids = torch.nonzero(fm_core, as_tuple=False).flatten()
            for part in range(state.num_moving_parts):
                state.core_masks[part, ids[part::state.num_moving_parts]] = True
        log(f"[bootstrap] legacy FM moving core candidates {int(fm_core.sum())}")
    if seed["reason"] is None:
        state.core_masks |= seed["core_masks"].to(state.core_masks)
        sizes = [int(row.sum()) for row in state.core_masks]
        log(f"[bootstrap] observed-speed split {seed['split']:.5g} -> initial cores {sizes}")
    else:
        log(f"[bootstrap] no initial core: {seed['reason']}")

    for outer in range(int(args.outer_iterations)):
        state.outer_iteration = outer
        # The critic's vote is earned, not assumed: at cycle zero it has had no
        # training at all, so it carries no weight in the assignment.
        warmup = max(0, int(args.diffusion_vote_warmup_cycles))
        assignment_weights.diffusion = diffusion_weight * (
            1.0 if warmup == 0 else min(1.0, outer / float(warmup))
        )
        previous_labels = state.hard_labels.clone()
        previous_modes = list(state.killing_modes)
        # Spec 32: fixed targets while the physics is still settling, resampled
        # afterwards so the critic does not learn a handful of views.
        if explicit_targets is None:
            target_frames = sample_target_frames(
                state.num_frames, 5, target_generator,
                deterministic=outer < max(1, int(args.deterministic_target_cycles)),
            )

        # --- A. inner physical loop: FM <-> consensus <-> Killing ------
        consensus = None
        killing_diagnostics = None
        for _ in range(max(1, int(args.inner_physical_iterations))):
            state.killing_velocity = predict_all_killing_velocities(
                state.killing_modes, state.killing_interval_speeds, state.canonical_fg
            )
            valid = [bool(mode.valid) for mode in state.killing_modes]
            fusion = compute_consensus_velocity(
                fm_velocity=state.fm_velocity.detach(),
                fm_confidence=state.fm_confidence,
                observed_velocity=state.observed_velocity,
                observed_normal_matrix=state.observed_normal_matrix,
                observed_valid=state.observed_valid,
                killing_velocity=state.killing_velocity.detach(),
                responsibilities=state.responsibilities.detach(),
                killing_valid=valid,
                weights=consensus_weights,
            )
            consensus = fusion["consensus_velocity"]
            state.consensus_velocity = consensus.detach()

            if not ablation.get("freeze_killing", False):
                refit = refit_killing_fields(
                    canonical_fg=state.canonical_fg,
                    consensus_velocity=consensus.detach(),
                    responsibilities=state.responsibilities.detach(),
                    core_masks=state.core_masks,
                    informative=fusion["informative"],
                    gamma=args.responsibility_fit_gamma,
                    previous_modes=state.killing_modes,
                    previous_speeds=state.killing_interval_speeds,
                    solver_kwargs=solver_kwargs,
                )
                state.killing_modes = refit["modes"]
                state.killing_interval_speeds = refit["interval_speeds"].to(
                    device=state.canonical_fg.device, dtype=torch.float32
                )
                killing_diagnostics = refit["diagnostics"]

        # --- B. exact physical proposal --------------------------------
        state.killing_velocity = predict_all_killing_velocities(
            state.killing_modes, state.killing_interval_speeds, state.canonical_fg
        )
        state.killing_positions = integrate_killing_positions(
            state.killing_modes, state.killing_interval_speeds,
            state.canonical_fg, state.frame_times,
        )
        killing_transforms = integrate_killing_transforms(
            state.killing_modes,
            state.killing_interval_speeds,
            state.frame_times,
            device=state.canonical_fg.device,
            dtype=state.canonical_fg.dtype,
        )
        # Staged activation: early cycles show the critic only part of the
        # transport, so it is not asked to judge a hypothesis the physics has
        # not earned yet.
        alpha = schedule_value(alpha_schedule, outer)
        staged = state.canonical_fg[None, None] + alpha * (
            state.killing_positions - state.canonical_fg[None, None]
        )
        state.hard_labels = state.responsibilities.argmax(-1)
        physical_positions = build_discrete_physical_proposal(
            canonical_fg=state.canonical_fg,
            hard_labels=state.hard_labels,
            killing_positions=staged,
        )
        physical_positions_exact = build_discrete_physical_proposal(
            canonical_fg=state.canonical_fg,
            hard_labels=state.hard_labels,
            killing_positions=state.killing_positions,
        )
        state.physical_displacement = physical_positions_exact[-1].detach() - state.canonical_fg
        state.physical_velocity = _physical_velocity_from_labels(state).detach()
        state.fm_current_displacement = _integrate_velocity_endpoint(
            state.fm_velocity, state.frame_times
        ).detach()
        state.fm_current_confidence = state.fm_confidence.detach().clone()

        rendered_feedback = None
        if render_transport_feedback_fn is not None and (
            float(args.fm_rendered_flow_weight) > 0.0
            or float(args.observed_rendered_flow_weight) > 0.0
            or float(args.target_physical_render_weight) > 0.0
            or bool(args.save_rendered_flow_debug)
        ):
            rendered_feedback = render_transport_feedback_fn(
                state,
                target_frames,
                output_dir=Path(output_root) / f"outer_{outer:04d}",
            )
            state.energy_fm_render = rendered_feedback["energy_fm_render"].detach()
            state.energy_obs_render = rendered_feedback["energy_obs_render"].detach()
            state.energy_target_render = rendered_feedback["energy_target_render"].detach()

        # FM is a residual over the legacy base.  Cycle 0 keeps it effectively
        # frozen; later cycles allow bounded corrections from physical evidence.
        fm_restored_after_collapse = False
        if (
            ablation.get("use_fm", True)
            and flow_field is not None
            and bool(args.fm_residual_refinement)
        ):
            previous_fm = state.fm_velocity.detach().clone()
            ramp = _feedback_ramp(
                outer, args.fm_feedback_start_cycle, args.fm_feedback_ramp_cycles
            )
            if int(args.outer_iterations) <= 1:
                base_preserve = float(args.fm_base_preservation_start)
            else:
                mix = min(1.0, outer / float(max(1, int(args.outer_iterations) - 1)))
                base_preserve = (
                    (1.0 - mix) * float(args.fm_base_preservation_start)
                    + mix * float(args.fm_base_preservation_end)
                )
            base_preserve *= float(args.fm_base_preservation_weight) * float(args.lambda_fm_base)
            physical_confidence = state.fm_confidence.detach()
            if rendered_feedback is not None:
                rel = rendered_feedback["reliability_fm_render"]
                chosen = rel[
                    torch.arange(state.num_foreground, device=rel.device),
                    state.hard_labels.clamp(0, state.num_labels - 1),
                ]
                physical_confidence = torch.maximum(
                    physical_confidence,
                    chosen[None].expand_as(physical_confidence).to(physical_confidence),
                )
            cycle_feedback = FlowMatchingFeedbackWeights(
                base=base_preserve,
                killing=float(args.lambda_fm_killing) * ramp
                if ablation.get("fm_killing_feedback", True) else 0.0,
                physical_3d=float(args.fm_physical_3d_weight) * ramp,
                static=float(args.lambda_fm_static) * ramp
                if ablation.get("fm_killing_feedback", True) else 0.0,
                gamma=args.responsibility_fit_gamma,
            )
            trust_radius = _fm_trust_radius(state, neighbour_scale, args)
            train_temporal_flow_field(
                flow_field,
                points=state.canonical_fg,
                target_velocity=base_velocity,
                target_confidence=base_confidence,
                killing_velocity=state.killing_velocity.detach(),
                responsibilities=state.responsibilities.detach(),
                killing_valid=valid,
                base_velocity=base_velocity,
                physical_velocity=state.physical_velocity,
                physical_confidence=physical_confidence,
                static_feedback_gate=_static_feedback_gate(state),
                max_residual_step=trust_radius,
                feedback_weights=cycle_feedback,
            )
            state.fm_velocity = flow_field.evaluate_all_intervals(
                state.canonical_fg,
                state.num_intervals,
                base_velocity=base_velocity,
                max_residual_step=trust_radius,
            ).detach()
            state.fm_current_displacement = _integrate_velocity_endpoint(
                state.fm_velocity, state.frame_times
            ).detach()
            base_norm = float(base_velocity.norm(dim=-1).mean())
            current_norm = float(state.fm_velocity.norm(dim=-1).mean())
            if base_norm > 1.0e-8 and current_norm < 0.25 * base_norm and outer > 0:
                state.fm_velocity = previous_fm
                state.fm_current_displacement = _integrate_velocity_endpoint(
                    state.fm_velocity, state.frame_times
                ).detach()
                fm_restored_after_collapse = True

            # Refit Killing once from the corrected FM so the next energies see
            # the current transport, not the pre-feedback one.
            state.killing_velocity = predict_all_killing_velocities(
                state.killing_modes, state.killing_interval_speeds, state.canonical_fg
            )
            fusion = compute_consensus_velocity(
                fm_velocity=state.fm_velocity.detach(),
                fm_confidence=state.fm_confidence,
                observed_velocity=state.observed_velocity,
                observed_normal_matrix=state.observed_normal_matrix,
                observed_valid=state.observed_valid,
                killing_velocity=state.killing_velocity.detach(),
                responsibilities=state.responsibilities.detach(),
                killing_valid=[bool(mode.valid) for mode in state.killing_modes],
                weights=consensus_weights,
            )
            consensus = fusion["consensus_velocity"]
            state.consensus_velocity = consensus.detach()
            if not ablation.get("freeze_killing", False):
                refit = refit_killing_fields(
                    canonical_fg=state.canonical_fg,
                    consensus_velocity=consensus.detach(),
                    responsibilities=state.responsibilities.detach(),
                    core_masks=state.core_masks,
                    informative=fusion["informative"],
                    gamma=args.responsibility_fit_gamma,
                    previous_modes=state.killing_modes,
                    previous_speeds=state.killing_interval_speeds,
                    solver_kwargs=solver_kwargs,
                )
                state.killing_modes = refit["modes"]
                state.killing_interval_speeds = refit["interval_speeds"].to(
                    device=state.canonical_fg.device, dtype=torch.float32
                )
                killing_diagnostics = refit["diagnostics"]
            state.killing_velocity = predict_all_killing_velocities(
                state.killing_modes, state.killing_interval_speeds, state.canonical_fg
            )
            state.killing_positions = integrate_killing_positions(
                state.killing_modes, state.killing_interval_speeds,
                state.canonical_fg, state.frame_times,
            )
            staged = state.canonical_fg[None, None] + alpha * (
                state.killing_positions - state.canonical_fg[None, None]
            )
            physical_positions = build_discrete_physical_proposal(
                canonical_fg=state.canonical_fg,
                hard_labels=state.hard_labels,
                killing_positions=staged,
            )
            physical_positions_exact = build_discrete_physical_proposal(
                canonical_fg=state.canonical_fg,
                hard_labels=state.hard_labels,
                killing_positions=state.killing_positions,
            )
            state.physical_displacement = physical_positions_exact[-1].detach() - state.canonical_fg
            state.physical_velocity = _physical_velocity_from_labels(state).detach()
        candidate_displacements = compute_candidate_physical_displacements(
            state.canonical_fg, state.killing_positions,
        )
        fm3d = compute_fm_physical_endpoint_energy(
            fm_current_displacement=state.fm_current_displacement,
            fm_confidence=state.fm_current_confidence,
            candidate_displacements=candidate_displacements,
            huber_delta=float(args.endpoint_transport_robust_delta),
        )
        state.energy_fm_3d = fm3d["energy"].detach()
        state.fm_physical_3d_residual = (
            state.fm_current_displacement - state.physical_displacement
        ).norm(dim=-1).detach()

        # --- C. physics-conditioned residual diffusion -----------------
        diffusion_diagnostics = None
        if ablation.get("use_diffusion", True) and denoiser is not None:
            # Train before trusting.  The corruption gives the exact correction
            # the critic should have produced, so this needs no renderer and is
            # a far stronger signal than the render residual alone.
            if int(args.denoiser_train_iterations) > 0:
                training = train_physics_conditioned_denoiser(
                    denoiser, spec=spec, state=state,
                    clean_positions=physical_positions.detach(),
                    target_frames=target_frames, knn_idx=knn_idx,
                    neighbour_scale=neighbour_scale,
                    config=DiffusionConfig(
                        num_trials=args.diffusion_trials,
                        denoise_steps=args.diffusion_steps,
                        noise_sigma_ratio=args.diffusion_noise_sigma_ratio,
                        seed=int(args.seed) + outer,
                    ),
                    corruption=CorruptionConfig(),
                    iterations=int(args.denoiser_train_iterations),
                    learning_rate=float(args.denoiser_learning_rate),
                    render_evidence_fn=render_evidence_fn,
                    observation_confidence_fn=observation_confidence_fn,
                    render_loss_fn=render_loss_fn,
                    log_every=max(1, int(args.denoiser_train_iterations) // 2),
                )
                if training["history"]:
                    last = training["history"][-1]
                    log(f"    critic trained {training['iterations']} steps, "
                        f"supervised loss {last['supervised']:.5f}")
            result = run_physics_conditioned_diffusion(
                denoiser, spec=spec, state=state,
                physical_positions=physical_positions.detach(),
                target_frames=target_frames, knn_idx=knn_idx,
                neighbour_scale=neighbour_scale,
                config=DiffusionConfig(
                    num_trials=args.diffusion_trials,
                    denoise_steps=args.diffusion_steps,
                    noise_sigma_ratio=args.diffusion_noise_sigma_ratio,
                    seed=int(args.seed) + outer,
                ),
                render_evidence_fn=render_evidence_fn,
                observation_confidence_fn=observation_confidence_fn,
            )
            state.diffusion_residual_mean = result.mean
            state.diffusion_residual_cov = result.cov
            state.diffusion_confidence = result.confidence
            diffusion_diagnostics = result.diagnostics

        # --- C2. component-level counterfactual rendering (spec 28) -----
        # Per-Gaussian energies need each Gaussian to have its own evidence; on
        # a real scene most do not.  A component is testable as a unit, so this
        # is the only channel that can speak for a group the multiview flow
        # never saw.
        counterfactual = None
        components: list = []
        if counterfactual_config.enabled and render_loss_fn is not None:
            support = state.observed_confidence.sum(0)
            components = ambiguous_components(
                responsibilities=state.responsibilities, knn_idx=knn_idx,
                graph_weight=graph["weight"], config=counterfactual_config,
                observed_support=support,
            )
            if components:
                counterfactual = counterfactual_component_energy(
                    state=state, components=components, target_frames=target_frames,
                    render_component_fn=build_component_render_fn(
                        state=state, render_scene_fn=render_loss_fn,
                        killing_positions=state.killing_positions,
                    ),
                    config=counterfactual_config,
                    persistent=counterfactual_state, cycle=outer,
                )
                counterfactual_state = counterfactual["persistent"]
                tested = int(counterfactual_state["tested"].sum())
                log(f"    counterfactual: {len(components)} components, "
                    f"{tested}/{state.num_foreground} Gaussians carry a verdict")
                if args.leave_component_out:
                    for part in range(state.num_moving_parts):
                        for component in components[:2]:
                            check = leave_component_out_validation(
                                canonical_fg=state.canonical_fg,
                                consensus_velocity=state.consensus_velocity,
                                responsibilities=state.responsibilities,
                                core_masks=state.core_masks, component=component,
                                part=part, gamma=args.responsibility_fit_gamma,
                                solver_kwargs=solver_kwargs,
                            )
                            if check.get("available"):
                                log(f"      leave-component-out part {part + 1} "
                                    f"size {check['component_size']}: agreement "
                                    f"{check['agreement_ratio']:.3f}, beats static by "
                                    f"{check['improvement_over_static']:.3f}")

        # --- D. candidate-label energies -------------------------------
        valid = [bool(mode.valid) for mode in state.killing_modes]
        energies = {
            "diffusion": None,
            "fm": None,
            "fm_3d": state.energy_fm_3d,
            "fm_render": state.energy_fm_render if rendered_feedback is not None else None,
            "observed_velocity": None,
            "obs_render": state.energy_obs_render if rendered_feedback is not None else None,
            "target_render": state.energy_target_render if rendered_feedback is not None else None,
            "graph": None,
            "counterfactual": None,
        }
        energy_reliability = {
            "fm_3d": fm3d["reliability"],
        }
        if rendered_feedback is not None:
            energy_reliability.update({
                "fm_render": rendered_feedback["reliability_fm_render"],
                "obs_render": rendered_feedback["reliability_obs_render"],
                "target_render": rendered_feedback["reliability_target_render"],
            })
        if counterfactual is not None:
            energies["counterfactual"] = counterfactual["energy"].detach()
        # Computed whenever the critic ran, even while its vote is still ramping
        # in: the energy is diagnostic output as much as a term, the debug
        # directory has to be the same shape every cycle, and the acceptance
        # check reads it from cycle zero.  Only the weight ramps.
        if ablation.get("use_diffusion", True) and denoiser is not None:
            energies["diffusion"] = compute_diffusion_transition_energy(
                canonical_fg=state.canonical_fg, hard_labels=state.hard_labels,
                killing_positions=state.killing_positions,
                residual_mean=state.diffusion_residual_mean,
                residual_cov=state.diffusion_residual_cov,
                residual_confidence=state.diffusion_confidence,
                target_frames=target_frames,
            ).detach()
        if assignment_weights.flow_matching > 0:
            energies["fm"] = compute_fm_killing_label_energy(
                fm_velocity=state.fm_velocity, fm_confidence=state.fm_confidence,
                killing_velocity=state.killing_velocity, killing_valid=valid,
            ).detach()
        if assignment_weights.observed > 0:
            energies["observed_velocity"] = compute_observed_velocity_label_energy(
                observed_velocity=state.observed_velocity,
                observed_normal_matrix=state.observed_normal_matrix,
                observed_valid=state.observed_valid,
                killing_velocity=state.killing_velocity, killing_valid=valid,
            ).detach()
        energies["graph"] = compute_assignment_graph_energy(
            state.responsibilities, graph
        ).detach()

        combined = robust_normalize_and_combine(
            energy_diffusion=energies["diffusion"], energy_fm=energies["fm"],
            energy_fm_3d=energies["fm_3d"],
            energy_fm_render=energies["fm_render"],
            energy_observed=energies["observed_velocity"],
            energy_obs_render=energies["obs_render"],
            energy_target_render=energies["target_render"],
            energy_graph=energies["graph"],
            energy_counterfactual=energies["counterfactual"],
            energy_reliability=energy_reliability,
            weights=assignment_weights,
            min_reliable_gap=float(args.rendered_flow_min_gap),
        )
        state.label_energies = combined["energy_total"]

        # --- E. soft assignment update ---------------------------------
        if ablation.get("freeze_assignment", False):
            delta = 0.0
        else:
            update = update_responsibilities(
                combined["energy_total"],
                assignment_temperature(outer, temperature_schedule),
                previous=state.responsibilities,
            )
            state.responsibilities = update["responsibilities"].detach()
            state.hard_labels = update["hard_labels"]
            state.assignment_entropy = update["assignment_entropy"]
            delta = update["responsibility_delta"]

        hysteresis = update_core_shell_hysteresis(
            responsibilities=state.responsibilities,
            moving_streak=state.moving_streak,
            static_streak=state.static_streak,
            previous_core=state.core_masks,
            previous_static_core=state.static_core_mask,
            config=core_config,
        )
        for key, value in hysteresis.items():
            setattr(state, key, value)

        # --- F. optional residual-driven speed correction ---------------
        decomposition = []
        for part, mode in enumerate(state.killing_modes):
            item = decompose_residual_against_field(
                state.diffusion_residual_mean, canonical_fg=state.canonical_fg,
                mode=mode, responsibilities_for_part=state.responsibilities[:, part + 1],
            )
            item["part"] = part + 1
            decomposition.append(item)
            if args.speed_correction_rate > 0 and mode.valid:
                nudge = residual_speed_correction(
                    state.diffusion_residual_mean, canonical_fg=state.canonical_fg,
                    mode=mode, responsibilities_for_part=state.responsibilities[:, part + 1],
                    frame_times=state.frame_times, gamma=args.responsibility_fit_gamma,
                )
                state.killing_interval_speeds[part] += (
                    float(args.speed_correction_rate)
                    * nudge.to(state.killing_interval_speeds)
                )

        if args.cross_fit_validation:
            for part in range(state.num_moving_parts):
                check = cross_fit_core_validation(
                    canonical_fg=state.canonical_fg, consensus_velocity=consensus,
                    responsibilities=state.responsibilities, core_masks=state.core_masks,
                    part=part, gamma=args.responsibility_fit_gamma,
                    solver_kwargs=solver_kwargs,
                )
                if check.get("available"):
                    log(f"    cross-fit part {part + 1}: "
                        f"generalisation ratio {check['generalisation_ratio']:.3f}")

        # --- G. debug and convergence -----------------------------------
        state.validate()
        record = collect_diagnostics(
            state, responsibility_delta=delta, consensus_velocity=consensus,
            killing_diagnostics=killing_diagnostics,
            diffusion_diagnostics=diffusion_diagnostics,
            residual_decomposition=decomposition,
            previous_labels=previous_labels, previous_modes=previous_modes,
            energy_scales=combined["scales"],
        )
        record["energy_reliability"] = combined.get("reliability", {})
        record["fm_base_velocity_norm_mean"] = float(state.fm_base_velocity.norm(dim=-1).mean())
        record["fm_current_velocity_norm_mean"] = float(state.fm_velocity.norm(dim=-1).mean())
        record["fm_current_minus_base_velocity_norm_mean"] = float(
            (state.fm_velocity - state.fm_base_velocity).norm(dim=-1).mean()
        )
        record["fm_base_displacement_norm_mean"] = float(state.fm_base_displacement.norm(dim=-1).mean())
        record["fm_current_displacement_norm_mean"] = float(state.fm_current_displacement.norm(dim=-1).mean())
        record["physical_displacement_norm_mean"] = float(state.physical_displacement.norm(dim=-1).mean())
        record["fm_base_physical_3d_residual_mean"] = float(
            (state.fm_base_displacement - state.physical_displacement).norm(dim=-1).mean()
        )
        record["fm_current_physical_3d_residual_mean"] = float(
            state.fm_physical_3d_residual.mean()
        )
        record["fm_collapse_update_rejected"] = bool(fm_restored_after_collapse)
        for name in ("fm_3d", "fm_render", "observed_velocity", "obs_render", "diffusion", "target_render"):
            energy = energies.get(name)
            if energy is not None and energy.shape[-1] > 1:
                record[f"{name}_moving_preferred_fraction"] = float(
                    (energy[:, 1:].amin(-1) < energy[:, 0]).float().mean()
                )
        if rendered_feedback is not None:
            record["rendered_transport_feedback"] = rendered_feedback["diagnostics"]
        record.update(detect_failure_modes(record))
        record["physics_alpha"] = alpha
        record["target_frames"] = list(target_frames)
        record["counterfactual_components"] = len(components)
        if counterfactual is not None:
            record["counterfactual_summaries"] = counterfactual["summaries"]
        if args.adapt_to_failure_modes:
            record["failure_response"] = respond_to_failure_modes(
                record, assignment_weights, consensus_weights, core_config, log=log
            )["applied"]
        records.append(record)
        state.history.append(record)
        save_closed_loop_debug(
            state, output_root, consensus_velocity=consensus,
            killing_transforms=killing_transforms,
            energies=energies, diagnostics=record,
        )
        write_closed_loop_html(
            state, Path(output_root) / f"outer_{outer:04d}" / "state.html",
            consensus_velocity=consensus, previous_labels=previous_labels,
        )
        log(format_diagnostics_line(record, state.num_moving_parts))
        for warning in record.get("warnings", []):
            log(f"    warning: {warning}")

        if delta < float(args.convergence_responsibility_delta) and outer > 0:
            log(f"  converged: responsibility delta {delta:.6f}")
            break

    return {"records": records, "target_frames": target_frames}


def save_checkpoint(
    path: Path,
    *,
    state,
    denoiser,
    denoiser_config,
    flow_field,
    flow_config,
    args,
    ablation_name,
    input_metadata=None,
):
    """The new checkpoint format of spec 44; legacy denoisers are not reused."""
    payload = {
        "schema_version": SCHEMA_VERSION,
        "pipeline": PIPELINE_NAME,
        "foreground_config": {
            "num_foreground": state.num_foreground,
            "foreground_idx": state.foreground_idx.detach().cpu(),
        },
        "denoiser_state_dict": None if denoiser is None else denoiser.state_dict(),
        # Architecture read off the module, never off the requested config.
        "denoiser_config": None if denoiser is None else {
            **denoiser_config.to_dict(),
            "input_dim": denoiser.input_dim,
            "hidden_dim": denoiser.hidden_dim,
            "message_layers": denoiser.message_layers,
        },
        "fm_state_dict": None if flow_field is None else flow_field.state_dict(),
        "fm_config": None if flow_field is None else asdict(flow_config),
        "assignment_config": {
            "num_moving_parts": state.num_moving_parts,
            "temperature_schedule": args.assignment_temperature_schedule,
            "move_activation_cost": args.move_activation_cost,
            "responsibility_fit_gamma": args.responsibility_fit_gamma,
        },
        "killing_config": {
            "modes": [mode.as_dict() for mode in state.killing_modes],
            "interval_speeds": state.killing_interval_speeds.detach().cpu(),
        },
        "training_config": {**vars(args), "ablation": ablation_name},
        "input_metadata": dict(input_metadata or {}),
        "state": state.to_payload(),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    ablation = ABLATIONS[args.ablation]
    output = Path(args.output_dir).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(int(args.seed))
    print(f"[closed loop] ablation {args.ablation}: {ablation['purpose']}")

    from tools._closed_loop_scene import load_closed_loop_inputs

    inputs = load_closed_loop_inputs(args)
    state = allocate_closed_loop_state(
        inputs["canonical_fg"], inputs["foreground_idx"],
        inputs["frame_ids"], inputs["frame_times"],
        args.num_moving_parts, static_prior=args.static_prior,
        device=inputs["canonical_fg"].device,
    )
    state.observed_velocity = inputs["observed_velocity"]
    state.observed_confidence = inputs["observed_confidence"]
    state.observed_normal_matrix = inputs["observed_normal_matrix"]
    state.observed_valid = inputs["observed_valid"]
    state.fm_base_velocity = inputs["fm_base_velocity"].to(
        device=state.canonical_fg.device, dtype=state.canonical_fg.dtype
    ).clone()
    state.fm_base_confidence = inputs["fm_base_confidence"].to(
        device=state.canonical_fg.device, dtype=state.canonical_fg.dtype
    ).clone()
    state.fm_base_displacement = inputs["fm_base_displacement"].to(
        device=state.canonical_fg.device, dtype=state.canonical_fg.dtype
    ).clone()
    state.fm_base_probability = inputs["fm_base_probability"].to(
        device=state.canonical_fg.device, dtype=state.canonical_fg.dtype
    ).clone()
    state.fm_base_valid = inputs["fm_base_valid"].to(device=state.canonical_fg.device).bool().clone()
    state.fm_velocity = inputs["fm_base_velocity"].to(
        device=state.canonical_fg.device, dtype=state.canonical_fg.dtype
    ).clone()
    state.fm_confidence = inputs["fm_base_confidence"].to(
        device=state.canonical_fg.device, dtype=state.canonical_fg.dtype
    ).clone()
    state.fm_current_displacement = state.fm_base_displacement.clone()
    state.fm_current_confidence = state.fm_confidence.clone()
    _initialise_responsibilities_from_fm(
        state, state.fm_base_probability, state.fm_base_valid,
    )
    state.assignment_entropy = compute_entropy(state.responsibilities)
    state.validate()
    _write_fm_initialization_check(state, args, output)
    print(f"[closed loop] {state.num_foreground} foreground Gaussians, "
          f"{state.num_frames} frames, K={state.num_moving_parts}")

    knn_idx, neighbour_scale = build_foreground_knn(state.canonical_fg, k=args.denoiser_knn)
    graph = build_foreground_graph(state.canonical_fg, knn_idx)
    spec = PhysicsFeatureSpec(num_moving_parts=args.num_moving_parts)
    denoiser_config = DenoiserConfig(
        hidden_dim=args.denoiser_hidden_dim, message_layers=args.denoiser_message_layers,
        knn=args.denoiser_knn, spec=spec,
    )
    denoiser = PhysicsConditionedGaussianDenoiser(
        spec.input_dim, hidden_dim=args.denoiser_hidden_dim,
        message_layers=args.denoiser_message_layers,
    ).to(state.canonical_fg.device)
    flow_config = TemporalFlowConfig(
        hidden_dim=args.fm_hidden_dim, layers=args.fm_layers,
        iterations=args.fm_iterations, learning_rate=args.fm_learning_rate, seed=args.seed,
    )
    flow_field = TemporalFlowField(flow_config).to(state.canonical_fg.device)

    result = run_closed_loop(
        state=state, args=args, ablation=ablation, denoiser=denoiser, spec=spec,
        flow_field=flow_field, knn_idx=knn_idx, neighbour_scale=neighbour_scale,
        graph=graph, output_root=output,
        render_evidence_fn=inputs.get("render_evidence_fn"),
        render_loss_fn=inputs.get("render_loss_fn"),
        observation_confidence_fn=inputs.get("observation_confidence_fn"),
        render_transport_feedback_fn=inputs.get("render_transport_feedback_fn"),
        fm_base_velocity=inputs["fm_base_velocity"],
        fm_base_confidence=inputs["fm_base_confidence"],
    )
    endpoint = inputs.get("endpoint_transport")
    if endpoint is not None:
        endpoint_dir = output / "endpoint_fm"
        endpoint_dir.mkdir(parents=True, exist_ok=True)
        torch.save(endpoint["displacement"].detach().cpu(), endpoint_dir / "endpoint_displacement.pt")
        torch.save(endpoint["confidence"].detach().cpu(), endpoint_dir / "endpoint_confidence.pt")
        torch.save(endpoint["probability"].detach().cpu(), endpoint_dir / "endpoint_dynamic_probability.pt")
        (endpoint_dir / "endpoint_fm_summary.json").write_text(
            json.dumps(endpoint["summary"], indent=2, allow_nan=False),
            encoding="utf-8",
        )
    save_checkpoint(
        output / "closed_loop_checkpoint.pt", state=state, denoiser=denoiser,
        denoiser_config=denoiser_config, flow_field=flow_field, flow_config=flow_config,
        args=args, ablation_name=args.ablation,
        input_metadata=inputs.get("metadata", {}),
    )
    (output / "closed_loop_summary.json").write_text(
        json.dumps({
            "schema_version": SCHEMA_VERSION, "pipeline": PIPELINE_NAME,
            "ablation": args.ablation, "ablation_purpose": ablation["purpose"],
            "target_frames": result["target_frames"],
            "outer_cycles": len(result["records"]),
            "final": result["records"][-1] if result["records"] else None,
            "history": result["records"],
            "ground_truth_used_as_algorithm_input": False,
        }, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    print(f"[closed loop] wrote {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
