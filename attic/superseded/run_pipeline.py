#!/usr/bin/env python3
"""Run the whole articulated-motion pipeline for one scene, end to end.

    python run_pipeline.py --scene USB_100109

One command, five stages, one report.  Each stage is skipped if its output
already exists, so a re-run after a failure resumes rather than restarting.

    0  canonical static 3DGS at frame 0        (train.py, reused if present)
    1  progressive rigid discovery            legacy FM + robust SE(3) + components
    2  joint labelling                         revolute vs prismatic, both fitted
    3  metric joint trajectory q(t)            profile scale on the Stage-1 shape
    4  assignment refinement under exact joint transport, then a final q(t) refit
    5  evaluation, including ground truth where the scene ships it

The legacy closed-loop stage is still available with
``--stage1-mode closed_loop`` for comparison.  The default Stage 1 is the
progressive rigid topology requested in
``CODEX_PROGRESSIVE_RIGID_DISCOVERY_PIPELINE.md``.

Stage 4 is the outer feedback of the older specification's section 41: the coarse
Killing transport that Stage 1 assigned against is replaced by the exact joint
transport, and the assignment is asked again.

Useful variants:

    --stages 1,2,3,5        skip the Stage-4 refinement
    --frames stride:4       cut the observation cost while debugging
    --ablation B_fm_killing_only
    --dry-run               print the plan and the resolved paths, run nothing
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
import traceback
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

ALL_STAGES = ("0", "1", "2", "3", "4", "5")


# ----------------------------------------------------------------------
# paths
# ----------------------------------------------------------------------
def resolve_paths(args) -> dict:
    """Work out every path once, so a stage never guesses where its input is."""
    scene_root = Path(args.scene)
    if not scene_root.is_absolute() and not scene_root.exists():
        scene_root = REPO / "input" / args.scene
    scene_root = scene_root.expanduser().resolve()
    if not scene_root.is_dir():
        raise SystemExit(f"scene not found: {scene_root}")

    name = scene_root.name
    output = (
        Path(args.output_dir).expanduser().resolve()
        if args.output_dir
        else REPO / "output" / f"{name}_pipeline"
    )
    static = (
        Path(args.static_model_dir).expanduser().resolve()
        if args.static_model_dir
        else output / "static"
    )
    return {
        "scene_name": name,
        "scene_root": scene_root,
        "output": output,
        "static": static,
        "static_ply": static / "point_cloud" / f"iteration_{args.static_iterations}" / "point_cloud.ply",
        "closed_loop": output / "closed_loop",
        "closed_loop_checkpoint": output / "closed_loop" / "closed_loop_checkpoint.pt",
        "progressive": output / "stage1_progressive",
        "progressive_checkpoint": output / "stage1_progressive" / "progressive_checkpoint.pt",
        "joint": output / "joint",
        "report": output / "pipeline_report.json",
    }


def find_existing_static(scene_name: str, iterations: int) -> Path | None:
    """Reuse a canonical model that is already lying around.

    Training frame-0 3DGS takes far longer than the rest of the pipeline, and
    most of these scenes already have one from an earlier run.  Only a directory
    whose PLY is actually present at the requested iteration counts.
    """
    for candidate in sorted((REPO / "output").glob(f"{scene_name}*/shared/frame0_static")):
        if (candidate / "point_cloud" / f"iteration_{iterations}" / "point_cloud.ply").is_file():
            return candidate
    for candidate in sorted((REPO / "output").glob(f"{scene_name}*")):
        if (candidate / "point_cloud" / f"iteration_{iterations}" / "point_cloud.ply").is_file():
            return candidate
    return None


def _run_child(command, log) -> None:
    """Run a stage subprocess, echoing its output and keeping the tail on failure.

    ``subprocess.run(check=True)`` alone raises a CalledProcessError whose text
    is only the command line, so the child's actual traceback ends up buried
    above a wall of re-quoted arguments.  The last lines are what say what went
    wrong, so they are attached to the error.
    """
    process = subprocess.Popen(
        command, cwd=REPO, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, bufsize=1,
    )
    tail: list[str] = []
    assert process.stdout is not None
    for line in process.stdout:
        sys.stdout.write(line)
        sys.stdout.flush()
        tail.append(line.rstrip())
        if len(tail) > 40:
            tail.pop(0)
    code = process.wait()
    if code != 0:
        raise RuntimeError(
            f"{Path(command[1]).name} exited {code}. Last output:\n  "
            + "\n  ".join(tail[-12:])
        )


# ----------------------------------------------------------------------
# stages
# ----------------------------------------------------------------------
def resolve_static_model(paths: dict, args, log) -> bool:
    """Point ``paths['static']`` at a usable canonical model, if one exists.

    This runs before the stage loop rather than inside stage 0, because every
    later stage needs the path whether or not stage 0 was asked for.  Resolving
    it inside stage 0 meant ``--stages 1`` handed the closed loop a directory
    that had never been created.

    Returns True when a model is already available.
    """
    if paths["static_ply"].is_file():
        log(f"canonical model: {paths['static']}")
        return True
    if args.static_model_dir:
        # Explicitly requested and missing: say so now rather than after the
        # subprocess fails to read a PLY.
        raise SystemExit(
            f"--static-model-dir has no PLY at iteration {args.static_iterations}: "
            f"{paths['static_ply']}"
        )
    found = find_existing_static(paths["scene_name"], args.static_iterations)
    if found is not None and not args.force_static:
        paths["static"] = found
        paths["static_ply"] = (
            found / "point_cloud" / f"iteration_{args.static_iterations}" / "point_cloud.ply"
        )
        log(f"canonical model: {found} (found, reused)")
        return True
    log("canonical model: none found, stage 0 will train one")
    return False


def stage0_static(paths: dict, args, log) -> dict:
    """Canonical frame-0 3DGS, with the semantic and DINO fields the loop needs."""
    if paths["static_ply"].is_file() and not args.force_static:
        log(f"  reusing {paths['static']}")
        return {"skipped": True, "reason": "already available", "path": str(paths["static"])}

    source = paths["scene_root"] / "colmap"
    command = [
        sys.executable, "-u", str(REPO / "train.py"),
        "-s", str(source if source.is_dir() else paths["scene_root"]),
        "-m", str(paths["static"]),
        "--iterations", str(args.static_iterations),
        "--static_train_iters", str(args.static_iterations),
        "--save_iterations", str(args.static_iterations),
        "--use_semantic_feature_field",
        "--use_dino_feature_field",
        "--dinov3_root", str(paths["scene_root"] / "dinov3"),
        "--disable_viewer", "--no_wandb", "--quiet",
    ]
    log("  " + " ".join(command[:6]) + " ...")
    _run_child(command, log)
    if not paths["static_ply"].is_file():
        raise RuntimeError(f"static training produced no PLY at {paths['static_ply']}")
    return {"skipped": False, "path": str(paths["static"])}


def stage1_closed_loop(paths: dict, args, log) -> dict:
    """Closed-loop motion discovery: the assignment and the rigid fields."""
    if paths["closed_loop_checkpoint"].is_file() and not args.force:
        log(f"  reusing {paths['closed_loop_checkpoint'].name}")
        return {"skipped": True, "reason": "already run"}

    command = [
        sys.executable, "-u", str(REPO / "tools" / "train_closed_loop_motion.py"),
        "--root-scene", str(paths["scene_root"]),
        "--static-model-dir", str(paths["static"]),
        "--static-iteration", str(args.static_iterations),
        "--output-dir", str(paths["closed_loop"]),
        "--num-moving-parts", str(args.num_moving_parts),
        "--outer-iterations", str(args.outer_iterations),
        "--frames", args.frames,
        "--ablation", args.ablation,
        "--move-activation-cost", str(args.move_activation_cost),
        "--semantic-dynamic-threshold", str(args.semantic_dynamic_threshold),
        "--correspondence-flow-max-side", str(args.correspondence_flow_max_side),
        "--fps", str(args.fps),
        "--denoiser-train-iterations", str(args.denoiser_train_iterations),
        "--diffusion-vote-warmup-cycles", str(args.diffusion_vote_warmup_cycles),
        "--neutral-bootstrap-iterations", str(args.neutral_bootstrap_iterations),
        "--counterfactual-max-components", str(args.counterfactual_max_components),
        "--fm-init", args.fm_init,
        "--fm-iterations", str(args.fm_iterations),
        "--legacy-fm-iters", str(args.legacy_fm_iters),
        "--legacy-fm-batch-size", str(args.legacy_fm_batch_size),
        "--legacy-fm-integration-steps", str(args.legacy_fm_integration_steps),
        "--legacy-fm-integrator", args.legacy_fm_integrator,
        "--legacy-fm-identity-weight", str(args.legacy_fm_identity_weight),
        "--legacy-fm-smooth-weight", str(args.legacy_fm_smooth_weight),
        "--fm-base-preservation-weight", str(args.fm_base_preservation_weight),
        "--fm-base-preservation-start", str(args.fm_base_preservation_start),
        "--fm-base-preservation-end", str(args.fm_base_preservation_end),
        "--fm-physical-3d-weight", str(args.fm_physical_3d_weight),
        "--fm-rendered-flow-weight", str(args.fm_rendered_flow_weight),
        "--observed-rendered-flow-weight", str(args.observed_rendered_flow_weight),
        "--target-physical-render-weight", str(args.target_physical_render_weight),
        "--fm-feedback-start-cycle", str(args.fm_feedback_start_cycle),
        "--fm-feedback-ramp-cycles", str(args.fm_feedback_ramp_cycles),
        "--fm-max-residual-step", str(args.fm_max_residual_step),
        "--rendered-flow-min-support", str(args.rendered_flow_min_support),
        "--rendered-flow-min-gap", str(args.rendered_flow_min_gap),
        "--rendered-flow-cameras-per-frame", str(args.rendered_flow_cameras_per_frame),
        "--endpoint-transport-knn", str(args.endpoint_transport_knn),
        "--endpoint-transport-temperature", str(args.endpoint_transport_temperature),
        "--endpoint-transport-robust-delta", str(args.endpoint_transport_robust_delta),
        "--endpoint-transport-confidence-distance-scale",
        str(args.endpoint_transport_confidence_distance_scale),
        "--endpoint-transport-confidence-entropy-scale",
        str(args.endpoint_transport_confidence_entropy_scale),
        "--endpoint-transport-chunk-size", str(args.endpoint_transport_chunk_size),
        "--endpoint-motion-scale", args.endpoint_motion_scale,
        "--fm-prior-prob-bias", str(args.fm_prior_prob_bias),
        "--fm-prior-prob-temperature", str(args.fm_prior_prob_temperature),
    ]
    if args.no_endpoint_fm:
        command.append("--no-endpoint-fm")
    if not args.fm_residual_refinement:
        command.append("--no-fm-residual-refinement")
    if args.force_legacy_fm:
        command.append("--force-legacy-fm")
    if args.save_rendered_flow_debug:
        command.append("--save-rendered-flow-debug")
    if args.endpoint_static_model_dir:
        command.extend([
            "--endpoint-static-model-dir",
            str(Path(args.endpoint_static_model_dir).expanduser().resolve()),
        ])
    if int(args.endpoint_static_iterations) > 0:
        command.extend(["--endpoint-static-iteration", str(int(args.endpoint_static_iterations))])
    if args.adapt_to_failure_modes:
        command.append("--adapt-to-failure-modes")
    if args.leave_component_out:
        command.append("--leave-component-out")
    if args.allow_all_gaussian_foreground:
        command.append("--allow-all-gaussian-foreground")
    if args.cross_fit_validation:
        command.append("--cross-fit-validation")
    log("  " + " ".join(command[:8]) + " ...")
    _run_child(command, log)
    if not paths["closed_loop_checkpoint"].is_file():
        raise RuntimeError("the closed loop produced no checkpoint")
    return {"skipped": False, "output": str(paths["closed_loop"])}


def stage1_progressive_rigid(paths: dict, args, log) -> dict:
    """Spec-compliant Stage 1: legacy FM -> robust finite SE(3) -> components."""
    if paths["progressive_checkpoint"].is_file() and not args.force:
        log(f"  reusing {paths['progressive_checkpoint'].name}")
        return {"skipped": True, "reason": "already run"}
    command = [
        sys.executable, "-u", str(REPO / "tools" / "train_progressive_rigid_discovery.py"),
        "--root-scene", str(paths["scene_root"]),
        "--static-model-dir", str(paths["static"]),
        "--static-iteration", str(args.static_iterations),
        "--output-dir", str(paths["progressive"]),
        "--frames", args.frames,
        "--fps", str(args.fps),
        "--num-moving-parts", str(args.num_moving_parts),
        "--semantic-dynamic-threshold", str(args.semantic_dynamic_threshold),
        "--correspondence-flow-max-side", str(args.correspondence_flow_max_side),
        "--legacy-fm-path", str(args.legacy_fm_path),
        "--legacy-fm-iters", str(args.legacy_fm_iters),
        "--legacy-fm-batch-size", str(args.legacy_fm_batch_size),
        "--legacy-fm-integration-steps", str(args.legacy_fm_integration_steps),
        "--legacy-fm-integrator", args.legacy_fm_integrator,
        "--legacy-fm-identity-weight", str(args.legacy_fm_identity_weight),
        "--legacy-fm-smooth-weight", str(args.legacy_fm_smooth_weight),
        "--endpoint-transport-knn", str(args.endpoint_transport_knn),
        "--endpoint-transport-temperature", str(args.endpoint_transport_temperature),
        "--endpoint-transport-robust-delta", str(args.endpoint_transport_robust_delta),
        "--endpoint-transport-confidence-distance-scale",
        str(args.endpoint_transport_confidence_distance_scale),
        "--endpoint-transport-confidence-entropy-scale",
        str(args.endpoint_transport_confidence_entropy_scale),
        "--endpoint-transport-chunk-size", str(args.endpoint_transport_chunk_size),
        "--endpoint-motion-scale", args.endpoint_motion_scale,
        "--fm-prior-prob-bias", str(args.fm_prior_prob_bias),
        "--fm-prior-prob-temperature", str(args.fm_prior_prob_temperature),
        "--fm-regression-min-valid-fraction", str(args.fm_regression_min_valid_fraction),
        "--fm-regression-min-gt-recall", str(args.fm_regression_min_gt_recall),
        "--fm-core-dynamic-threshold", str(args.fm_core_dynamic_threshold),
        "--fm-core-static-threshold", str(args.fm_core_static_threshold),
        "--fm-core-confidence-threshold", str(args.fm_core_confidence_threshold),
        "--rigid-init-ransac-iters", str(args.rigid_init_ransac_iters),
        "--rigid-init-inlier-threshold", str(args.rigid_init_inlier_threshold),
        "--ransac-threshold-mode", args.ransac_threshold_mode,
        "--candidate-target-count", str(args.candidate_target_count),
        "--candidate-endpoint-knn", str(args.candidate_endpoint_knn),
        "--candidate-chunk-size", str(args.candidate_chunk_size),
        "--correspondence-em-iters", str(args.correspondence_em_iters),
        "--rigid-irls-iters", str(args.rigid_irls_iters),
        "--correspondence-isometry-k", str(args.correspondence_isometry_k),
        "--correspondence-isometry-weight", str(args.correspondence_isometry_weight),
        "--rigid-huber-delta", str(args.rigid_huber_delta),
        "--assignment-temperature", str(args.assignment_temperature),
        "--assignment-margin", str(args.assignment_margin),
        "--responsibility-gamma", str(args.responsibility_gamma),
        "--inner-rigid-iters", str(args.inner_rigid_iters),
        "--inner-rigid-iters-after-expand", str(args.inner_rigid_iters_after_expand),
        "--inner-rigid-lr", str(args.inner_rigid_lr),
        "--inner-trust-region", str(args.inner_trust_region),
        "--lambda-fm", str(args.lambda_fm),
        "--lambda-pcd", str(args.lambda_pcd),
        "--lambda-rgb", str(args.lambda_rgb),
        "--lambda-depth", str(args.lambda_depth),
        "--lambda-diffusion", str(args.lambda_diffusion),
        "--pcd-fit-samples", str(args.pcd_fit_samples),
        "--pcd-target-samples", str(args.pcd_target_samples),
        "--pcd-chunk-size", str(args.pcd_chunk_size),
        "--inner-render-cameras", str(args.inner_render_cameras),
        "--outer-iterations", str(args.outer_iterations),
        "--candidate-components-per-outer", str(args.candidate_components_per_outer),
        "--candidate-max-gaussians-per-outer", str(args.candidate_max_gaussians_per_outer),
        "--candidate-min-size", str(args.candidate_min_size),
        "--candidate-knn", str(args.candidate_knn),
        "--candidate-distance-scale", str(args.candidate_distance_scale),
        "--candidate-fm-scale", str(args.candidate_fm_scale),
        "--component-min-reliability", str(args.component_min_reliability),
        "--component-min-observable-fraction", str(args.component_min_observable_fraction),
        "--render-cameras-per-frame", str(args.render_cameras_per_frame),
        "--render-min-support", str(args.render_min_support),
        "--counterfactual-radius-px", str(args.counterfactual_radius_px),
        "--counterfactual-render-eps", str(args.counterfactual_render_eps),
        "--counterfactual-min-affected-pixels", str(args.counterfactual_min_affected_pixels),
        "--max-html-points", str(args.max_html_points),
    ]
    if args.endpoint_static_model_dir:
        command.extend([
            "--endpoint-static-model-dir",
            str(Path(args.endpoint_static_model_dir).expanduser().resolve()),
        ])
    if int(args.endpoint_static_iterations) > 0:
        command.extend(["--endpoint-static-iteration", str(int(args.endpoint_static_iterations))])
    if args.force_legacy_fm:
        command.append("--force-legacy-fm")
    if args.smoke:
        command.append("--smoke")
    if args.diffusion_hypothesis_test:
        command.append("--diffusion-hypothesis-test")
    if not args.use_depth:
        command.append("--no-use-depth")
    if args.save_debug_html:
        command.append("--save-debug-html")
    if args.strict_stage1_bugfix_validation:
        command.append("--strict-stage1-bugfix-validation")
    log("  " + " ".join(command[:8]) + " ...")
    _run_child(command, log)
    if not paths["progressive_checkpoint"].is_file():
        raise RuntimeError("progressive rigid discovery produced no checkpoint")
    return {"skipped": False, "output": str(paths["progressive"])}


def stage1_motion_discovery(paths: dict, args, log) -> dict:
    if args.stage1_mode == "progressive_rigid":
        return stage1_progressive_rigid(paths, args, log)
    return stage1_closed_loop(paths, args, log)


def _rebuild_scene_and_state(paths: dict, args, log):
    """Reload the scene and the converged loop state for the joint stages.

    Stage 2 renders, so it needs the real scene back.  Rather than keeping it
    alive across a subprocess boundary, it is rebuilt here from the same bridge
    the loop used, and the state comes from the checkpoint.
    """
    from tools._closed_loop_scene import load_closed_loop_inputs
    from utils.closed_loop_motion_state import ClosedLoopMotionState, allocate_closed_loop_state
    from utils.rigid_killing_motion import RigidKillingMode

    bridge_args = argparse.Namespace(
        root_scene=str(paths["scene_root"]),
        static_model_dir=str(paths["static"]),
        static_iteration=args.static_iterations,
        frames=args.frames,
        fps=args.fps,
        semantic_dynamic_threshold=args.semantic_dynamic_threshold,
        semantic_dynamic_min_points=32,
        allow_all_gaussian_foreground=args.allow_all_gaussian_foreground,
        correspondence_flow_max_side=args.correspondence_flow_max_side,
        no_endpoint_fm=True,
    )
    inputs = load_closed_loop_inputs(bridge_args)
    # The canonical model is reused read-only from wherever it was trained, but
    # the joint stages write their debug artifacts under ``scene.model_path``.
    # They belong to this pipeline run, not to the model being borrowed.
    inputs["scene"].model_path = str(paths["output"])

    payload = torch.load(paths["closed_loop_checkpoint"], map_location="cpu", weights_only=False)
    saved = payload["state"]
    device = inputs["canonical_fg"].device
    fields = {}
    for name, value in saved.items():
        fields[name] = value.to(device) if torch.is_tensor(value) else value
    # The modes live in the state snapshot, alongside every other field that is
    # being restored, so they come from there rather than from the separate
    # `killing_config` mirror -- one source, no chance of the two disagreeing.
    serialised_modes = saved.get("killing_modes")
    if serialised_modes is None:
        serialised_modes = payload["killing_config"]["modes"]
    fields["killing_modes"] = [
        RigidKillingMode.from_dict(mode) for mode in serialised_modes
    ]
    template = allocate_closed_loop_state(
        inputs["canonical_fg"], inputs["foreground_idx"], inputs["frame_ids"],
        inputs["frame_times"], int(saved.get("num_moving_parts", payload["assignment_config"]["num_moving_parts"])),
        device=device,
    )
    for name in ClosedLoopMotionState.__dataclass_fields__:
        if name not in fields and hasattr(template, name):
            fields[name] = getattr(template, name)
    allowed = set(ClosedLoopMotionState.__dataclass_fields__)
    state = ClosedLoopMotionState(**{k: v for k, v in fields.items() if k in allowed})
    state.validate()
    log(f"  restored state: {state.num_foreground} foreground, K={state.num_moving_parts}, "
        f"{state.label_counts()}")
    return inputs, state


def stage2_joint_labelling(paths: dict, args, inputs, state, log) -> dict:
    """Label each part's joint, fitting both hypotheses and comparing them."""
    from types import SimpleNamespace

    from utils.closed_loop_to_joint import closed_loop_state_to_stage1
    from utils.joint_model_fitting import fit_joint_graph

    stage1 = closed_loop_state_to_stage1(
        state, total_gaussians=inputs["total_gaussians"],
        gamma=args.responsibility_fit_gamma,
    )
    for part, field in enumerate(stage1["velocity_fields"], start=1):
        evidence = field["joint_type_evidence"]
        log(f"  part {part}: rigid residual {field['cost']:.4g}  "
            f"translation {field['translation_cost']:.4g}  "
            f"-> {evidence['joint_type']} (log-evidence {evidence['log_evidence_difference']})")

    joint_args = SimpleNamespace(
        joint_types="auto", joint_topology="known",
        joint_compare_iters=args.joint_compare_iters, joint_compare_batch_size=4,
        joint_compare_validation_cameras=2, joint_compare_geometry_lr=1.0e-3,
        joint_compare_trajectory_lr=1.0e-2, joint_compare_screw_margin=0.01,
        joint_compare_refit_geometry=False,
        joint_frames=args.joint_frames,
        num_moving_parts=state.num_moving_parts,
        use_dino_feature_field=False, lambda_dino_feature_field_dynamic=0.0,
        transported_html_max_points=20000,
        fps=args.fps, time_scale=1.0, motion_time_mode="seconds",
        dynamic_test_stride=0,
        # The joint render loss is masked to the dynamic foreground, and the mask
        # loader finds that mask from the scene root.  Without these two the
        # loader silently falls back to the camera alpha and the candidate
        # comparison is scored over the whole image, background included.
        root_scene=str(paths["scene_root"]),
        dynamic_sequence_root=str(paths["scene_root"]),
        semantic_mask_dir=args.semantic_mask_dir,
    )
    opt = SimpleNamespace(lambda_dssim=0.2, depth_l1_weight_final=0.01)
    dataset = SimpleNamespace(
        white_background=False, train_test_exp=False,
        source_path=str(paths["scene_root"]),
    )
    background = torch.tensor(
        [0.0, 0.0, 0.0], device=inputs["canonical_fg"].device
    )
    paths["joint"].mkdir(parents=True, exist_ok=True)
    result = fit_joint_graph(
        inputs["scene"], inputs["gaussians"], inputs["pipe"], background, dataset, opt,
        stage1, joint_args, paths["joint"], log_fn=None,
    )
    for summary in result["summary"]["joints"]:
        log(f"  selected {summary['joint_type']} "
            f"(decided by {summary['decided_by']}, confidence {summary['confidence']})")
    return {"stage1": stage1, "stage2": result, "joint_args": joint_args, "opt": opt,
            "dataset": dataset, "background": background}


def stage3_trajectory(paths: dict, args, inputs, state, bundle, log) -> dict:
    """Metric q(t): one scale on the Stage-1 profile, released only if asked."""

    joint_args = bundle["joint_args"]
    for name, value in (
        ("optimize_joint_trajectory", True),
        ("joint_trajectory_mode", args.joint_trajectory_mode),
        ("joint_trajectory_iters", args.joint_trajectory_iters),
        ("joint_trajectory_batch_size", 8),
        ("joint_trajectory_release_per_frame", args.release_per_frame),
        ("joint_geometry_refine", False),
        ("motion_lr", 1.0e-2),
        ("static_flow_mad_floor", 0.25),
        ("visibility_depth_sigma", 0.05),
        ("visibility_occlusion_margin", 2.0),
        ("correspondence_min_confidence", 0.05),
        ("dynamic_flow_dir", ""),
        ("dynamic_sequence_root", str(paths["scene_root"])),
        ("correspondence_flow_max_side", 512),
        ("correspondence_cycle_threshold", 1.5),
        ("correspondence_dino_min_cosine", 0.2),
        ("correspondence_alpha_threshold", 0.05),
        ("use_dino_feature_field", False),
    ):
        setattr(joint_args, name, value)

    return _optimize_trajectory(inputs, bundle, joint_args, bundle["stage1"], log)


def _optimize_trajectory(inputs, bundle, joint_args, stage1, log):
    """One q(t) optimisation. Shared by Stage 3 and Stage 4's step-7 refit."""
    from utils.joint_trajectory_optimization import optimize_joint_trajectory

    result = optimize_joint_trajectory(
        inputs["scene"], inputs["gaussians"], inputs["pipe"], bundle["background"],
        bundle["dataset"], bundle["opt"], joint_args, stage1, bundle["stage2"],
        log_fn=None,
    )
    log(f"  mode {result['summary']['trajectory_mode']}  "
        f"final q {[round(float(v), 5) for v in result['coordinates'][:, -1]]}  "
        f"render loss {result['summary']['final_render_loss']:.5g}")
    return result


def stage4_refine_under_joint(paths: dict, args, inputs, state, bundle, stage3, log) -> dict:
    """The section-41 arrow: reassign under exact joint transport, refit q(t).

    Every modality the closed loop used is rebuilt here and recomputed against
    the exact transport, which is section 41 step 5.  Deciding on the velocity
    observation alone re-imposes its coverage ceiling and sheds moving
    Gaussians the other channels had recovered.
    """
    from scene.physics_conditioned_gaussian_denoiser import (
        PhysicsFeatureSpec,
        build_foreground_knn,
        load_denoiser,
    )
    from utils.closed_loop_assignment import (
        AssignmentWeights,
        build_foreground_graph,
    )
    from utils.closed_loop_to_joint import reassign_under_joint_transport

    payload = torch.load(
        paths["closed_loop_checkpoint"], map_location="cpu", weights_only=False
    )
    # The critic's feature layout is built over a fixed neighbourhood size, so
    # the refinement has to use the one the loop trained with.  A separate
    # default here would silently reinterpret the neighbour columns.
    trained_knn = int(payload["training_config"].get("denoiser_knn", args.denoiser_knn))
    if trained_knn != args.denoiser_knn:
        log(f"  using the trained neighbourhood k={trained_knn}, not the default "
            f"{args.denoiser_knn}")
    knn_idx, neighbour_scale = build_foreground_knn(state.canonical_fg, k=trained_knn)
    graph = build_foreground_graph(state.canonical_fg, knn_idx)

    # The trained critic is the channel that reaches a Gaussian the velocity
    # never saw, so the checkpoint's denoiser is reloaded rather than skipped.
    denoiser = spec = None
    try:
        denoiser, _ = load_denoiser(payload)
        denoiser = denoiser.to(state.canonical_fg.device).eval()
        spec = PhysicsFeatureSpec(num_moving_parts=state.num_moving_parts)
        log(f"  critic reloaded: {denoiser.input_dim}-D input, "
            f"{denoiser.message_layers} message layers")
    except (KeyError, RuntimeError, ValueError) as error:
        log(f"  no reusable critic in the checkpoint ({error}); "
            "the diffusion channel is absent from this refinement")

    # The component-level verdicts were earned by rendering, against a transport
    # the exact joint one is a refinement of -- not a different hypothesis.  They
    # are the one channel that speaks for a Gaussian the velocity never observed,
    # so dropping them here is what keeps recall pinned to the flow's coverage.
    counterfactual_energy = None
    cycles = sorted(paths["closed_loop"].glob("outer_*/energy_counterfactual.pt"))
    if cycles:
        stored = torch.load(cycles[-1], map_location=state.canonical_fg.device)
        if stored.shape == (state.num_foreground, state.num_labels):
            counterfactual_energy = stored
            tested = int((stored.abs().sum(-1) > 0).sum())
            log(f"  carrying {cycles[-1].parent.name}'s component verdicts forward: "
                f"{tested} Gaussians")
        else:
            log(f"  {cycles[-1].name} has shape {tuple(stored.shape)}, expected "
                f"{(state.num_foreground, state.num_labels)}; not carried forward")

    joints = stage3["joints"]
    result = reassign_under_joint_transport(
        state, joints=joints, coordinates=stage3["coordinates"],
        frame_times=state.frame_times,
        temperature=args.refine_temperature,
        move_activation_cost=args.move_activation_cost,
        denoiser=denoiser, denoiser_spec=spec,
        knn_idx=knn_idx, neighbour_scale=neighbour_scale, graph=graph,
        assignment_weights=AssignmentWeights(),
        render_evidence_fn=inputs.get("render_evidence_fn"),
        observation_confidence_fn=inputs.get("observation_confidence_fn"),
        counterfactual_energy=counterfactual_energy,
    )
    log(f"  modalities recomputed under the exact transport: "
        f"{', '.join(result['modalities_used'])}")
    log(f"  labels changed under exact joint transport: {result['labels_changed']} "
        f"of {state.num_foreground}")
    if result["labels_changed"] == 0:
        log("  the assignment is already consistent with the exact transport; no refit needed")
        return {"labels_changed": 0, "refitted": False}

    # The assignment moved, so the coarse fields it was derived from no longer
    # describe it.  Refit them from the same consensus before q(t) is retried.
    from utils.fm_killing_consensus import refit_killing_fields

    state.responsibilities = result["responsibilities"]
    state.hard_labels = result["hard_labels"]
    state.assignment_entropy = result["assignment_entropy"]
    refit = refit_killing_fields(
        canonical_fg=state.canonical_fg, consensus_velocity=state.consensus_velocity,
        responsibilities=state.responsibilities, core_masks=state.core_masks,
        gamma=args.responsibility_fit_gamma, previous_modes=state.killing_modes,
        previous_speeds=state.killing_interval_speeds, solver_kwargs={"min_valid_cells": 8},
    )
    state.killing_modes = refit["modes"]
    state.killing_interval_speeds = refit["interval_speeds"].to(
        device=state.canonical_fg.device, dtype=torch.float32
    )
    state.validate()
    state.save(paths["output"] / "closed_loop_state_refined.pt")

    # Step 7: q(t) was fitted against the assignment Stage 3 saw, and that
    # assignment has just changed.  Refitting it on the new labels is the last
    # arrow of section 41; skipping it leaves a metric trajectory describing a
    # part that no longer has the same members.
    from utils.closed_loop_to_joint import closed_loop_state_to_stage1

    refined_stage1 = closed_loop_state_to_stage1(
        state, total_gaussians=inputs["total_gaussians"],
        gamma=args.responsibility_fit_gamma,
    )
    log("  refitting q(t) on the refined assignment")
    trajectory = _optimize_trajectory(
        inputs, bundle, bundle["joint_args"], refined_stage1, log
    )
    before = [float(v) for v in stage3["coordinates"][:, -1]]
    after = [float(v) for v in trajectory["coordinates"][:, -1]]
    log(f"  final q {before} -> {after}  render loss "
        f"{stage3['summary']['final_render_loss']:.5g} -> "
        f"{trajectory['summary']['final_render_loss']:.5g}")
    torch.save(
        {"coordinates": trajectory["coordinates"], "summary": trajectory["summary"]},
        paths["output"] / "joint_trajectory_refined.pt",
    )
    return {
        "labels_changed": int(result["labels_changed"]),
        "refitted": True,
        "counts": state.label_counts(),
        "q_final_before": before,
        "q_final_after": after,
        "render_loss_before": float(stage3["summary"]["final_render_loss"]),
        "render_loss_after": float(trajectory["summary"]["final_render_loss"]),
        "modalities_used": result["modalities_used"],
        # A modality that did not participate has no scale; recording it as a
        # number would claim it was normalised against something.
        "energy_scales": {
            k: (None if v is None else float(v))
            for k, v in result["energy_scales"].items()
        },
    }


def stage5_evaluate(paths: dict, args, log) -> dict:
    """Acceptance criteria plus ground truth, in one place."""
    if args.stage1_mode == "progressive_rigid":
        summary_path = paths["progressive"] / "progressive_summary.json"
        checkpoint = paths["progressive_checkpoint"]
        if not checkpoint.is_file():
            raise RuntimeError(f"missing progressive checkpoint: {checkpoint}")
        summary = json.loads(summary_path.read_text()) if summary_path.is_file() else {}
        checks = {
            "fm_base_present": (paths["progressive"] / "fm_base.pt").is_file(),
            "fm_regression_present": (paths["progressive"] / "fm_base_regression.json").is_file(),
            "rigid_hypotheses_present": (paths["progressive"] / "rigid_init" / "rigid_hypotheses.pt").is_file(),
            "final_labels_present": (paths["progressive"] / "final_stage1_labels.pt").is_file(),
            "final_status_present": (paths["progressive"] / "final_stage1_status.pt").is_file(),
        }
        for name, passed in checks.items():
            log(f"  [{'PASS' if passed else 'FAIL'}] {name}")
        if summary.get("gt_debug"):
            gt = summary["gt_debug"]["trusted_only"]
            log(f"  trusted GT P {gt['assignment_precision']:.4f}  "
                f"R {gt['assignment_recall']:.4f}  F1 {gt['assignment_f1']:.4f}  "
                f"IoU {gt['assignment_iou']:.4f}")
        return {
            "acceptance": {"passed": all(checks.values()), "checks": checks},
            "summary": summary,
        }

    from tools.evaluate_closed_loop_motion import (
        check_acceptance,
        load_cycles,
        score_against_ground_truth,
    )

    cycles = load_cycles(paths["closed_loop"])
    acceptance = check_acceptance(paths["closed_loop"], cycles)
    log(f"  acceptance: {'PASS' if acceptance['passed'] else 'FAIL'}")
    for name, item in acceptance["checks"].items():
        if not item["passed"]:
            log(f"    FAIL {name}: {item['detail']}")

    truth = score_against_ground_truth(
        paths["closed_loop"], paths["scene_root"],
        refined_state=paths["output"] / "closed_loop_state_refined.pt",
    )
    if truth:
        log(f"  stage 1  P {truth['assignment_precision']:.4f}  "
            f"R {truth['assignment_recall']:.4f}  F1 {truth['assignment_f1']:.4f}  "
            f"IoU {truth['assignment_iou']:.4f}  "
            f"({truth['predicted_moving']} moving of GT {truth['gt_moving']})")
        refined = truth.get("refined")
        if refined:
            log(f"  stage 4  P {refined['assignment_precision']:.4f}  "
                f"R {refined['assignment_recall']:.4f}  F1 {refined['assignment_f1']:.4f}  "
                f"IoU {refined['assignment_iou']:.4f}  "
                f"({refined['predicted_moving']} moving)")
        for key, value in truth.items():
            if key.endswith("axis_error_degrees"):
                log(f"  {key}: {value:.3f}")
    else:
        log("  no ground truth for this scene")

    joint_summary = paths["joint"] / "phase2_joint_fit_summary.json"
    joints = json.loads(joint_summary.read_text()) if joint_summary.is_file() else None
    return {"acceptance": acceptance, "ground_truth": truth, "joints": joints}


# ----------------------------------------------------------------------
# driver
# ----------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--scene", required=True,
                        help="scene name under input/, or a path to a scene root")
    parser.add_argument("--output-dir", default="")
    parser.add_argument("--static-model-dir", default="",
                        help="reuse this canonical model instead of searching or training")
    parser.add_argument("--stages", default="0,1,5")
    parser.add_argument("--stage1-mode", choices=("progressive_rigid", "closed_loop"), default="progressive_rigid",
                        help="new default is the progressive FM -> KxSE(3) Stage-1 path")
    parser.add_argument("--force", action="store_true", help="re-run stages that already have output")
    parser.add_argument("--force-static", action="store_true", help="retrain the canonical model")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--continue-on-error", action="store_true",
                        help="attempt later stages even if one fails")

    parser.add_argument("--num-moving-parts", type=int, default=1)
    parser.add_argument("--static-iterations", type=int, default=14000)
    parser.add_argument("--outer-iterations", type=int, default=8)
    parser.add_argument("--frames", default="endpoints")
    parser.add_argument("--joint-frames", default="all")
    parser.add_argument("--ablation", default="G_full_closed_loop")
    parser.add_argument("--move-activation-cost", type=float, default=0.05)
    parser.add_argument("--responsibility-fit-gamma", type=float, default=2.0)
    parser.add_argument("--cross-fit-validation", action="store_true")
    parser.add_argument("--joint-compare-iters", type=int, default=12)
    parser.add_argument("--joint-trajectory-mode", default="profile_scale",
                        choices=("profile_scale", "per_frame", "spline"))
    parser.add_argument("--joint-trajectory-iters", type=int, default=200)
    parser.add_argument("--release-per-frame", action="store_true")
    parser.add_argument("--refine-temperature", type=float, default=0.2)
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--semantic-dynamic-threshold", type=float, default=0.5)
    parser.add_argument("--allow-all-gaussian-foreground", action="store_true")
    parser.add_argument("--correspondence-flow-max-side", type=int, default=512)
    parser.add_argument("--denoiser-train-iterations", type=int, default=150)
    parser.add_argument("--diffusion-vote-warmup-cycles", type=int, default=2)
    parser.add_argument("--neutral-bootstrap-iterations", type=int, default=200)
    parser.add_argument("--denoiser-knn", type=int, default=16,
                        help="neighbourhood the critic and the graph energy use")
    parser.add_argument("--semantic-mask-dir", default="mask",
                        help="foreground masks under the scene root, used by the joint render loss")
    parser.add_argument("--counterfactual-max-components", type=int, default=8)
    parser.add_argument("--fm-init", choices=("legacy", "current_knn"), default="legacy")
    parser.add_argument("--fm-residual-refinement", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--fm-iterations", type=int, default=200)
    parser.add_argument("--legacy-fm-iters", type=int, default=1000)
    parser.add_argument("--legacy-fm-path", default="")
    parser.add_argument("--legacy-fm-batch-size", type=int, default=2048)
    parser.add_argument("--legacy-fm-integration-steps", type=int, default=16)
    parser.add_argument("--legacy-fm-integrator", choices=("euler", "rk4"), default="euler")
    parser.add_argument("--legacy-fm-identity-weight", type=float, default=0.01)
    parser.add_argument("--legacy-fm-smooth-weight", type=float, default=0.0)
    parser.add_argument("--force-legacy-fm", action="store_true")
    parser.add_argument("--fm-base-preservation-weight", type=float, default=1.0)
    parser.add_argument("--fm-base-preservation-start", type=float, default=1.0)
    parser.add_argument("--fm-base-preservation-end", type=float, default=0.25)
    parser.add_argument("--fm-physical-3d-weight", type=float, default=0.25)
    parser.add_argument("--fm-rendered-flow-weight", type=float, default=0.25)
    parser.add_argument("--observed-rendered-flow-weight", type=float, default=0.25)
    parser.add_argument("--target-physical-render-weight", type=float, default=0.05)
    parser.add_argument("--fm-feedback-start-cycle", type=int, default=1)
    parser.add_argument("--fm-feedback-ramp-cycles", type=int, default=3)
    parser.add_argument("--fm-max-residual-step", type=float, default=0.25)
    parser.add_argument("--rendered-flow-min-support", type=float, default=1.0e-4)
    parser.add_argument("--rendered-flow-min-gap", type=float, default=1.0e-8)
    parser.add_argument("--rendered-flow-cameras-per-frame", type=int, default=1)
    parser.add_argument("--save-rendered-flow-debug", action="store_true")
    parser.add_argument("--endpoint-static-model-dir", default="")
    parser.add_argument("--endpoint-static-iterations", type=int, default=0,
                        help="0 means reuse --static-iterations")
    parser.add_argument("--no-endpoint-fm", action="store_true")
    parser.add_argument("--endpoint-transport-knn", type=int, default=16)
    parser.add_argument("--endpoint-transport-temperature", type=float, default=0.02)
    parser.add_argument("--endpoint-transport-robust-delta", type=float, default=0.02)
    parser.add_argument("--endpoint-transport-confidence-distance-scale", type=float, default=0.0)
    parser.add_argument("--endpoint-transport-confidence-entropy-scale", type=float, default=1.0)
    parser.add_argument("--endpoint-transport-chunk-size", type=int, default=1024)
    parser.add_argument("--endpoint-motion-scale", default="local_knn",
                        choices=("local_knn", "global_nn", "gaussian_scale"))
    parser.add_argument("--fm-prior-prob-bias", type=float, default=1.0)
    parser.add_argument("--fm-prior-prob-temperature", type=float, default=0.35)
    parser.add_argument("--adapt-to-failure-modes", action="store_true")
    parser.add_argument("--leave-component-out", action="store_true")
    parser.add_argument("--smoke", action="store_true",
                        help="small progressive Stage-1 run: fewer Gaussians/iters/components")
    parser.add_argument("--fm-regression-min-valid-fraction", type=float, default=0.05)
    parser.add_argument("--fm-regression-min-gt-recall", type=float, default=0.0)
    parser.add_argument("--strict-stage1-bugfix-validation", action="store_true")
    parser.add_argument("--debug-reference-axis", nargs=3, type=float, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--fm-core-dynamic-threshold", type=float, default=0.65)
    parser.add_argument("--fm-core-static-threshold", type=float, default=0.25)
    parser.add_argument("--fm-core-confidence-threshold", type=float, default=0.25)
    parser.add_argument("--rigid-init-ransac-iters", type=int, default=256)
    parser.add_argument("--rigid-init-inlier-threshold", type=float, default=0.04)
    parser.add_argument("--ransac-threshold-mode", choices=("derived", "manual"), default="derived")
    parser.add_argument("--candidate-target-count", type=int, default=32)
    parser.add_argument("--candidate-endpoint-knn", type=int, default=16)
    parser.add_argument("--candidate-chunk-size", type=int, default=1024)
    parser.add_argument("--correspondence-em-iters", type=int, default=8)
    parser.add_argument("--rigid-irls-iters", type=int, default=5)
    parser.add_argument("--correspondence-isometry-k", type=int, default=8)
    parser.add_argument("--correspondence-isometry-weight", type=float, default=1.0)
    parser.add_argument("--rigid-huber-delta", type=float, default=0.02)
    parser.add_argument("--assignment-temperature", type=float, default=0.05)
    parser.add_argument("--assignment-margin", type=float, default=0.15)
    parser.add_argument("--responsibility-gamma", type=float, default=2.0)
    parser.add_argument("--inner-rigid-iters", type=int, default=120)
    parser.add_argument("--inner-rigid-iters-after-expand", type=int, default=60)
    parser.add_argument("--inner-rigid-lr", type=float, default=1.0e-2)
    parser.add_argument("--inner-trust-region", type=float, default=0.25)
    parser.add_argument("--lambda-fm", type=float, default=1.0)
    parser.add_argument("--lambda-pcd", type=float, default=0.25)
    parser.add_argument("--lambda-rgb", type=float, default=0.02)
    parser.add_argument("--lambda-depth", type=float, default=0.02)
    parser.add_argument("--lambda-diffusion", type=float, default=0.0)
    parser.add_argument("--pcd-fit-samples", type=int, default=4096)
    parser.add_argument("--pcd-target-samples", type=int, default=8192)
    parser.add_argument("--pcd-chunk-size", type=int, default=1024)
    parser.add_argument("--inner-render-cameras", type=int, default=1)
    parser.add_argument("--candidate-components-per-outer", type=int, default=2)
    parser.add_argument("--candidate-max-gaussians-per-outer", type=int, default=4096)
    parser.add_argument("--candidate-min-size", type=int, default=3)
    parser.add_argument("--candidate-knn", type=int, default=16)
    parser.add_argument("--candidate-distance-scale", type=float, default=0.05)
    parser.add_argument("--candidate-fm-scale", type=float, default=0.05)
    parser.add_argument("--component-min-reliability", type=float, default=0.05)
    parser.add_argument("--component-min-observable-fraction", type=float, default=0.1)
    parser.add_argument("--diffusion-hypothesis-test", action="store_true")
    parser.add_argument("--render-cameras-per-frame", type=int, default=2)
    parser.add_argument("--render-min-support", type=float, default=1.0e-5)
    parser.add_argument("--counterfactual-radius-px", type=int, default=4)
    parser.add_argument("--counterfactual-render-eps", type=float, default=1.0e-8)
    parser.add_argument("--counterfactual-min-affected-pixels", type=int, default=1)
    parser.add_argument("--use-depth", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--save-debug-html", action="store_true")
    parser.add_argument("--max-html-points", type=int, default=20000)
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    if args.stage1_mode == "closed_loop" and str(args.frames).lower() in {"endpoints", "endpoint"}:
        args.frames = "all"
    stages = [s.strip() for s in args.stages.split(",") if s.strip()]
    unknown = sorted(set(stages) - set(ALL_STAGES))
    if unknown:
        raise SystemExit(f"unknown stages {unknown}; valid are {', '.join(ALL_STAGES)}")

    paths = resolve_paths(args)
    paths["output"].mkdir(parents=True, exist_ok=True)

    print(f"scene    {paths['scene_root']}")
    print(f"output   {paths['output']}")
    print(f"stages   {', '.join(stages)}")
    print(f"stage 1  {args.stage1_mode}")
    print(f"parts    K={args.num_moving_parts}   ablation {args.ablation}")
    if not args.dry_run:
        available = resolve_static_model(paths, args, print)
        if not available and "0" not in stages:
            raise SystemExit(
                "no canonical model is available and stage 0 was not requested; "
                "add stage 0 or pass --static-model-dir"
            )
    if args.dry_run:
        print("\nresolved paths:")
        for key, value in paths.items():
            print(f"  {key:26s} {value}")
        if args.static_model_dir:
            print(f"\ncanonical model explicitly requested: {paths['static']}")
        elif paths["static_ply"].is_file():
            print(f"\ncanonical model already present at output path: {paths['static']}")
        else:
            existing = find_existing_static(paths["scene_name"], args.static_iterations)
            print(f"\ncanonical model that would be reused: {existing or 'none, would train'}")
        return 0

    report: dict = {
        "scene": paths["scene_name"],
        "output": str(paths["output"]),
        "stages_requested": stages,
        "config": vars(args),
        "results": {},
        "ground_truth_used_as_algorithm_input": False,
    }
    inputs = state = bundle = stage3 = None
    started = time.time()

    def run(name: str, title: str, function):
        if name not in stages:
            return None
        print(f"\n=== stage {name}: {title} ===")
        begin = time.time()
        try:
            value = function()
        except Exception as error:  # noqa: BLE001 - recorded, then re-raised or skipped
            report["results"][name] = {
                "status": "error", "error": str(error),
                "traceback": traceback.format_exc(limit=6),
                "seconds": round(time.time() - begin, 1),
            }
            print(f"  FAILED: {error}")
            paths["report"].write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
            if not args.continue_on_error:
                raise
            return None
        report["results"][name] = {
            "status": "ok", "seconds": round(time.time() - begin, 1),
            "detail": _summarise(value),
        }
        print(f"  done in {time.time() - begin:.1f}s")
        return value

    def run_internal(name: str, title: str, function):
        print(f"\n=== {title} ===")
        begin = time.time()
        try:
            value = function()
        except Exception as error:  # noqa: BLE001 - recorded, then re-raised or skipped
            report["results"][name] = {
                "status": "error", "error": str(error),
                "traceback": traceback.format_exc(limit=6),
                "seconds": round(time.time() - begin, 1),
            }
            print(f"  FAILED: {error}")
            paths["report"].write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
            if not args.continue_on_error:
                raise
            return None
        report["results"][name] = {
            "status": "ok", "seconds": round(time.time() - begin, 1),
            "detail": _summarise(value),
        }
        print(f"  done in {time.time() - begin:.1f}s")
        return value

    def log(message: str) -> None:
        print(message)

    if args.stage1_mode == "progressive_rigid" and any(stage in stages for stage in ("2", "3", "4")):
        raise SystemExit(
            "Progressive Stage 2/3/4 are intentionally gated until Stage 1 converges. "
            "Run --stages 0,1,5 or use --stage1-mode closed_loop for the legacy full pipeline."
        )

    run("0", "canonical static 3DGS", lambda: stage0_static(paths, args, log))
    run("1", "motion discovery", lambda: stage1_motion_discovery(paths, args, log))

    needs_scene = any(stage in stages for stage in ("2", "3", "4"))
    if needs_scene:
        loaded = run_internal("reload", "reload scene and converged state",
                              lambda: _rebuild_scene_and_state(paths, args, log))
        if loaded is not None:
            inputs, state = loaded

    if inputs is not None:
        bundle = run("2", "joint labelling",
                     lambda: stage2_joint_labelling(paths, args, inputs, state, log))
    if bundle is not None:
        stage3 = run("3", "metric joint trajectory",
                     lambda: stage3_trajectory(paths, args, inputs, state, bundle, log))
    if stage3 is not None:
        run("4", "reassign under exact joint transport",
            lambda: stage4_refine_under_joint(
                paths, args, inputs, state, bundle, stage3, log))
    run("5", "evaluation", lambda: stage5_evaluate(paths, args, log))

    report["total_seconds"] = round(time.time() - started, 1)
    paths["report"].write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    print(f"\ntotal {report['total_seconds']:.1f}s")
    print(f"report {paths['report']}")
    failed = [k for k, v in report["results"].items() if v.get("status") == "error"]
    return 1 if failed else 0


def _summarise(value):
    """Keep the report readable: no tensors, no scenes, no models."""
    if value is None:
        return None
    if isinstance(value, dict):
        out = {}
        for key, item in value.items():
            if torch.is_tensor(item):
                out[key] = list(item.shape)
            elif isinstance(item, (str, int, float, bool)) or item is None:
                out[key] = item
            elif isinstance(item, dict) and key in ("counts", "acceptance", "ground_truth", "joints", "summary"):
                out[key] = item
        return out
    return str(type(value))


if __name__ == "__main__":
    raise SystemExit(main())
