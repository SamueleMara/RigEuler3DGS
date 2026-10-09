"""Python entry point for the simple render-matching outer loop.

Builds the argument namespace in Python and calls ``train.training`` directly --
no shell wrapper, no subprocess. Import ``run()`` to drive it from your own code,
or run this file for a CLI.

    from tools.run_render_graph import run
    result = run(scene="USB_100109", out="USB_render_graph_test")

The config is the section 25 experiment: K=1, tau fixed, warm start on, all 8
final cameras, deterministic render-matching outer loop with per-class seeds.
"""

from __future__ import annotations

import argparse
import sys
from argparse import ArgumentParser
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))


# Everything that is not a plain default for this experiment, in one place so a
# caller can see and override the whole configuration from Python.
RENDER_GRAPH_CONFIG: dict = {
    # phase / motion representation coupling required by the dynamic validator
    "dynamic_motion": True,
    "phase1_only": True,
    "use_semantic_feature_field": True,
    "phase1_motion_representation": "rigid_killing_eulerian",
    "enable_nested_killing_part_discovery": True,
    "iterations": 14500,
    "static_train_iters": 14000,
    "start_ply_iteration": 14000,
    # inner solver (unchanged by this spec)
    "num_moving_parts": 1,
    "part_outer_iters": 6,
    "killing_inner_steps_per_frame": 15,
    "static_anchor_use_neighborhood_check": True,
    "lambda_inner_rgb": 1.0,
    "lambda_inner_ssim": 0.2,
    "lambda_inner_depth": 0.1,
    "lambda_inner_strain": 10.0,
    "lambda_inner_omega_consistency": 1.0,
    "lambda_inner_b_consistency": 1.0,
    "lambda_inner_field_separation": 1.0,
    "lambda_inner_observed_velocity": 1.0,
    "observed_velocity_c_min": 0.6,
    "observed_velocity_gamma": 2.0,
    "observed_velocity_rho_max": 0.2,
    "lambda_multiview_view_var": 0.0,
    "killing_warm_start": True,
    # pixel-mediated matching supplies the candidate energies
    "enable_pixel_matching": True,
    "raster_top_contributors": 4,
    "lambda_pixel_overlap": 0.5,
    "lambda_pixel_xyz": 1.0,
    "lambda_pixel_depth": 0.5,
    # the deterministic outer loop
    "outer_label_mode": "render_graph",
    "render_seed_confidence": 0.75,
    "min_support_views": 2,
    "killing_neighbor_k": 16,
    "graph_iterations": 1000,
    "graph_tolerance": 1.0e-4,
    "boundary_aware_graph": False,
    "tau_render_preference": 1.0,
    "per_class_seeds": True,
    "min_seeds_per_class": 32,
    "absolute_margin_floor": 0.05,
}


def build_args(
    scene: str = "USB_100109",
    out: str | None = None,
    *,
    endpoints: str | None = None,
    overrides: dict | None = None,
):
    """Construct the fully-validated argument namespace, in Python."""
    import train as train_module  # noqa: F401  (ensures the module imports cleanly)
    from arguments import (
        ModelParams,
        OptimizationParams,
        PipelineParams,
        add_dynamic_args,
        apply_dynamic_defaults,
    )

    out = out or f"{scene}_render_graph"
    endpoints = endpoints or f"output/{scene}_static_endpoints_14k"
    source = REPO / "input" / scene
    start_ply = REPO / endpoints / "shared/frame0_static/point_cloud/iteration_14000/point_cloud.ply"
    final_dir = REPO / endpoints / "shared/frame_t_static"
    model_path = REPO / "output" / out

    # Fail here rather than twenty minutes into a run.
    for label, path in (("scene input", source), ("frame-0 endpoint ply", start_ply),
                        ("frame-T endpoint model", final_dir)):
        if not path.exists():
            raise FileNotFoundError(f"missing {label}: {path}")

    parser = ArgumentParser()
    model = ModelParams(parser)
    optimization = OptimizationParams(parser)
    pipeline = PipelineParams(parser)
    add_dynamic_args(parser)
    for flag, default in (("--ip", "127.0.0.1"), ("--start_checkpoint", "")):
        parser.add_argument(flag, type=str, default=default)
    parser.add_argument("--port", type=int, default=6009)
    parser.add_argument("--debug_from", type=int, default=-1)
    for flag in ("--disable_viewer", "--detect_anomaly", "--quiet"):
        parser.add_argument(flag, action="store_true")
    for flag in ("--test_iterations", "--save_iterations", "--checkpoint_iterations"):
        parser.add_argument(flag, nargs="+", type=int, default=[])

    args = parser.parse_args([])                       # defaults only; no shell parsing
    args.source_path = str(source)
    args.model_path = str(model_path)
    args.start_ply = str(start_ply)
    args.final_3dgs_path = str(final_dir)
    for key, value in RENDER_GRAPH_CONFIG.items():
        setattr(args, key, value)
    for key, value in (overrides or {}).items():
        if not hasattr(args, key):
            raise AttributeError(f"unknown option: {key}")
        setattr(args, key, value)

    args = apply_dynamic_defaults(args)                 # the coupled-flag validator
    return args, model, optimization, pipeline


def run(
    scene: str = "USB_100109",
    out: str | None = None,
    *,
    endpoints: str | None = None,
    overrides: dict | None = None,
    evaluate: bool = True,
) -> dict:
    """Run the whole experiment from Python and return the evaluation."""
    args, model, optimization, pipeline = build_args(
        scene, out, endpoints=endpoints, overrides=overrides)

    print(f"scene      {scene}")
    print(f"output     {args.model_path}")
    print(f"mode       {args.outer_label_mode}  K={args.num_moving_parts}  "
          f"seeds@{args.render_seed_confidence} per_class={args.per_class_seeds}")
    print(f"debug      {args.model_path}/dynamic_debug/nested_part_embedding_killing/\n")

    from train import training

    training(
        model.extract(args),
        optimization.extract(args),
        pipeline.extract(args),
        args,
        args.start_checkpoint,
        args.save_iterations,
        args.checkpoint_iterations,
        args.test_iterations,
    )

    if not evaluate:
        return {"model_path": args.model_path}
    return evaluate_run(Path(args.model_path).relative_to(REPO).as_posix(), scene=scene)


def evaluate_run(run_dir: str, *, scene: str = "USB_100109") -> dict:
    """Section 23 three-stage evaluation (ground truth used only here)."""
    import runpy

    gt = REPO / "input" / scene / "pointcloud" / "points3D_parts.ply"
    argv = sys.argv
    sys.argv = ["eval_render_graph.py", "--run", run_dir, "--gt_ply", str(gt)]
    try:
        runpy.run_path(str(REPO / "tools" / "eval_render_graph.py"), run_name="__main__")
    finally:
        sys.argv = argv
    import json

    path = (REPO / run_dir / "dynamic_debug" / "nested_part_embedding_killing"
            / "render_graph_evaluation.json")
    return json.loads(path.read_text()) if path.is_file() else {}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--scene", default="USB_100109")
    ap.add_argument("--out", default=None)
    ap.add_argument("--endpoints", default=None)
    ap.add_argument("--parts", type=int, default=None, help="K moving parts")
    ap.add_argument("--outer", type=int, default=None,
                    help="number of outer iterations (default 6)")
    ap.add_argument("--memory", action="store_true",
                    help="enable the temporal credit memory (default off)")
    ap.add_argument("--seed-confidence", type=float, default=None)
    ap.add_argument("--graph-k", type=int, default=None)
    ap.add_argument("--tau-render", type=float, default=None)
    ap.add_argument("--global-seed-threshold", action="store_true",
                    help="disable per-class seeds (the old single global cut-off)")
    ap.add_argument("--init-mode", default=None,
                    choices=["legacy", "endpoint_contrastive", "high_precision_bootstrap"],
                    help="how P^(0) is built before the first Killing solve; "
                         "high_precision_bootstrap seeds two confident cores and leaves "
                         "the rest at [0.5, 0.5] for the outer loop to resolve")
    ap.add_argument("--boundary-split", action="store_true",
                    help="one motion-boundary 3DGS split after OUTER 0, before INNER 1")
    ap.add_argument("--legacy-hard-labels", action="store_true",
                    help="ablation: restore the hard render-graph outer loop")
    ap.add_argument("--init-only", action="store_true",
                    help="spec §28: stop after initialization, before the first Killing solve")
    ap.add_argument("--inner-solver", default=None, choices=["per_frame", "shared_scalar"],
                    help="shared_scalar = the CODEX_DECOUPLED two-stage inner loop "
                         "(joint geometry with analytical alpha*, then scalar trajectory)")
    ap.add_argument("--decoupled-ablation", default=None, choices=["A", "B", "C", "D", "E"],
                    help="CODEX_DECOUPLED §59 ladder for the Stage-B losses: A = joint "
                         "geometry only, B += velocity magnitude, C += whole-video render, "
                         "D += final PCD, E += continuity/flip/jerk")
    ap.add_argument("--render-frame-stride", type=int, default=None,
                    help="Stage-B renders every Nth frame (default 8); 1 = every frame as §23 asks")
    ap.add_argument("-O", "--set", action="append", default=[], metavar="KEY=VALUE",
                    help="override any argument by name, e.g. -O lambda_inner_rgb=0.001 "
                         "-O inner_stage_b_normalize=true (repeatable)")
    ap.add_argument("--no-eval", action="store_true")
    ap.add_argument("--print-config", action="store_true")
    args = ap.parse_args()

    overrides: dict = {}
    if args.parts is not None:
        overrides["num_moving_parts"] = args.parts
    if args.outer is not None:
        overrides["part_outer_iters"] = args.outer
    if args.memory:
        overrides["enable_temporal_credit"] = True
    if args.seed_confidence is not None:
        overrides["render_seed_confidence"] = args.seed_confidence
    if args.graph_k is not None:
        overrides["killing_neighbor_k"] = args.graph_k
    if args.tau_render is not None:
        overrides["tau_render_preference"] = args.tau_render
    if args.global_seed_threshold:
        overrides["per_class_seeds"] = False
    if args.init_mode is not None:
        overrides["init_mode"] = args.init_mode
    if args.boundary_split:
        overrides["enable_motion_boundary_split"] = True
    if args.legacy_hard_labels:
        overrides["outer_soft_labels"] = False
    if args.init_only:
        overrides["init_only"] = True
    if args.inner_solver is not None:
        overrides["inner_solver"] = args.inner_solver
    if args.decoupled_ablation is not None:
        overrides["inner_decoupled_ablation"] = args.decoupled_ablation
    if args.render_frame_stride is not None:
        overrides["inner_render_frame_stride"] = args.render_frame_stride
    for item in args.set:
        if "=" not in item:
            raise SystemExit(f"--set expects KEY=VALUE, got {item!r}")
        key, _, raw = item.partition("=")
        text = raw.strip()
        if text.lower() in ("true", "false"):
            value: object = text.lower() == "true"
        else:
            try:
                value = int(text)
            except ValueError:
                try:
                    value = float(text)
                except ValueError:
                    value = text
        overrides[key.strip()] = value

    if args.print_config:
        built, *_ = build_args(args.scene, args.out, endpoints=args.endpoints, overrides=overrides)
        for key in sorted(set(RENDER_GRAPH_CONFIG) | set(overrides)):
            print(f"  {key:42s} {getattr(built, key)}")
        return

    run(args.scene, args.out, endpoints=args.endpoints,
        overrides=overrides, evaluate=not args.no_eval)


if __name__ == "__main__":
    main()
