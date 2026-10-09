#!/usr/bin/env python3
"""Compare frozen Eulerian and Lagrangian rigid-Killing runs."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import torch


def _debug_dir(path: str) -> Path:
    root = Path(path).expanduser().resolve()
    return root if root.name == "dynamic_debug" else root / "dynamic_debug"


def _first(root: Path, names: tuple[str, ...]) -> Path:
    for name in names:
        path = root / name
        if path.is_file():
            return path
    raise FileNotFoundError(f"None of {names} exists in {root}")


def _optional_first(root: Path, names: tuple[str, ...]) -> Path | None:
    try:
        return _first(root, names)
    except FileNotFoundError:
        return None


def _json(root: Path, names: tuple[str, ...], *, required: bool = True) -> dict:
    path = _first(root, names) if required else _optional_first(root, names)
    return json.loads(path.read_text(encoding="utf-8")) if path is not None else {}


def _tensor_artifact(root: Path, names: tuple[str, ...]) -> dict:
    path = _optional_first(root, names)
    return (
        torch.load(path, map_location="cpu", weights_only=False)
        if path is not None
        else {}
    )


def _quantiles(value: torch.Tensor) -> dict:
    value = torch.as_tensor(value).float().reshape(-1)
    value = value[torch.isfinite(value)]
    if not value.numel():
        return {"p10": None, "p50": None, "p90": None}
    q = torch.quantile(value, torch.tensor((0.1, 0.5, 0.9)))
    return dict(zip(("p10", "p50", "p90"), map(float, q)))


def _median(value, mask=None):
    if value is None:
        return None
    tensor = torch.as_tensor(value).float()
    if mask is not None:
        tensor = tensor[torch.as_tensor(mask).bool()]
    tensor = tensor[torch.isfinite(tensor)]
    return float(tensor.median()) if tensor.numel() else None


def _artifact_scalars(value):
    result = {}
    for key, item in (value or {}).items():
        if torch.is_tensor(item):
            if item.numel() != 1:
                continue
            item = item.item()
        if item is None or isinstance(item, (int, float, bool, str)):
            result[key] = item
    return result


def _per_interval_median(value, valid) -> list[float | None]:
    value, valid = torch.as_tensor(value).float(), torch.as_tensor(valid).bool()
    result = []
    for interval in range(value.shape[0]):
        selected = value[interval][valid[interval] & torch.isfinite(value[interval])]
        result.append(float(selected.median()) if selected.numel() else None)
    return result


def _wandb_runtime(model_root: Path):
    summaries = sorted(model_root.glob("wandb/*/files/wandb-summary.json"))
    if not summaries:
        return None
    return json.loads(summaries[-1].read_text(encoding="utf-8")).get("_runtime")


def _run_metrics(root: Path, lagrangian: bool) -> dict:
    summary = _json(
        root,
        ("phase1_lagrangian_solver_summary.json", "phase1_rigid_solver_summary.json"),
    )
    validation = _json(
        root,
        ("phase1_lagrangian_gt_validation.json", "phase1_rigid_gt_validation.json"),
        required=False,
    )
    visibility = _json(root, ("phase1_visibility_summary.json",), required=False)
    audit = _tensor_artifact(
        root,
        ("phase1_lagrangian_observation_audit.pt", "phase1_rigid_observation_audit.pt"),
    )
    assignment = _tensor_artifact(
        root,
        ("phase1_lagrangian_assignment_final.pt", "phase1_rigid_assignment_final.pt"),
    )
    hypothesis = _tensor_artifact(
        root,
        ("phase1_lagrangian_hypothesis_evidence.pt", "phase1_rigid_hypothesis_evidence.pt"),
    )
    local_speed = _tensor_artifact(
        root, ("phase1_lagrangian_gaussian_speed_observations.pt",)
    )
    responsibility = _tensor_artifact(
        root, ("phase1_rigid_responsibility_final.pt",)
    )
    eulerian_support = _tensor_artifact(
        root, ("phase1_eulerian_contrastive_support_final.pt",)
    )
    speed_artifact = _tensor_artifact(
        root,
        ("phase1_lagrangian_interval_speeds.pt", "phase1_rigid_interval_speeds.pt"),
    )
    if not audit:
        raise FileNotFoundError(f"No rigid observation audit found in {root}")

    valid = torch.as_tensor(audit.get("accepted", audit["valid"])).bool()
    base_valid = torch.as_tensor(audit.get("base_valid", valid)).bool()
    cameras_all = torch.as_tensor(audit["distinct_camera_count"])
    cameras = cameras_all[valid]
    conditioning = torch.as_tensor(audit["conditioning"])
    condition_number = torch.as_tensor(
        audit.get("condition_number", conditioning.clamp_min(1.0e-12).reciprocal())
    )
    rigid_residual = torch.as_tensor(audit["rigid_residual"])
    speed_values = validation.get(
        "estimated_interval_speeds", speed_artifact.get("interval_speeds")
    )
    if speed_values is None:
        raise KeyError(f"No interval speeds found in {root}")
    speed = torch.as_tensor(speed_values).float()
    gt_speed_values = validation.get("gt_interval_speeds")
    gt_speed = torch.as_tensor(gt_speed_values).float() if gt_speed_values is not None else None
    final_mask = torch.as_tensor(
        assignment.get("effective_dynamic_mask", torch.zeros(valid.shape[1], dtype=torch.bool))
    ).bool()
    point_count = int(final_mask.numel())
    history = summary.get("em_history", [])
    final_cycle = history[-1] if history else {}
    tp = validation.get("assignment_true_positive")
    fp = validation.get("assignment_false_positive")
    fn = validation.get("assignment_false_negative")
    tn = validation.get("assignment_true_negative")
    if tn is None and None not in (tp, fp, fn):
        tn = point_count - tp - fp - fn
    accuracy = validation.get("assignment_accuracy")
    if accuracy is None and tn is not None and point_count:
        accuracy = (tp + tn) / point_count

    selected = final_mask
    local_perpendicular = None
    if "perpendicular_velocity_residual" in local_speed:
        local_perpendicular = torch.as_tensor(
            local_speed["perpendicular_velocity_residual"]
        )
        local_valid = torch.as_tensor(
            local_speed.get("valid", torch.ones_like(local_perpendicular))
        ).bool()
        local_perpendicular = local_perpendicular[
            local_valid & selected[None]
        ]
    temporal = {
        "speed_correlation_median": _median(hypothesis.get("speed_correlation"), selected),
        "speed_rmse_median": _median(local_speed.get("speed_rmse"), selected),
        "speed_relative_error_median": _median(hypothesis.get("speed_relative_error"), selected),
        "perpendicular_residual_median": (
            _median(local_perpendicular)
            if local_perpendicular is not None
            else _median(hypothesis.get("speed_perpendicular_residual"), selected)
        ),
        "temporal_persistence_median": _median(hypothesis.get("time_persistence"), selected),
        "camera_persistence_median": _median(hypothesis.get("camera_persistence"), selected),
        "informative_interval_fraction_median": (
            _median(hypothesis["informative_interval_count"].float() / valid.shape[0], selected)
            if "informative_interval_count" in hypothesis
            else None
        ),
        "visibility_fraction_median": None,
        "occlusion_fraction_median": None,
        "vacated_fraction_median": None,
    }

    runtime = dict(summary.get("runtime", {}))
    pretrain_runtime = visibility.get("runtime", {})
    wall_seconds = _wandb_runtime(root.parent)
    measured_total = runtime.get("rigid_phase_seconds_before_artifact_write")
    if measured_total is not None:
        measured_total += pretrain_runtime.get("total_seconds_before_artifact_write", 0.0)
    peak_gpu = max(
        (
            value
            for value in (
                runtime.get("peak_gpu_memory_bytes"),
                pretrain_runtime.get("peak_gpu_memory_bytes"),
            )
            if value is not None
        ),
        default=None,
    )
    angular = torch.as_tensor(summary["angular_mode"]).double()
    reference_velocity = torch.as_tensor(summary["reference_velocity"]).double()
    angular_norm2 = float(angular.square().sum())
    axis_cosine = validation.get("angular_direction_cosine")
    axis_error_degrees = (
        math.degrees(math.acos(max(-1.0, min(1.0, abs(float(axis_cosine))))))
        if axis_cosine is not None
        else None
    )
    observation_count = torch.as_tensor(audit["observation_count"]).float()
    camera_mask = audit.get("camera_support_mask")
    observing_cameras = (
        torch.as_tensor(camera_mask).bool().any(dim=0).sum(dim=-1)
        if camera_mask is not None
        else cameras_all.max(dim=0).values
    )
    rejected_static = audit.get("rejected_static_like")
    core_promotions = sum(
        int(item.get("assignment_refine/core_promotions", 0)) for item in history
    )
    core_demotions = sum(
        int(item.get("assignment_refine/core_demotions", 0)) for item in history
    )

    return {
        "representation": summary["representation"],
        "support_type": summary.get("support_type", "gaussian" if lagrangian else "occupied_cell"),
        "observation": {
            "number_of_support_units": int(valid.shape[1]),
            "number_valid_per_interval": valid.sum(dim=1).tolist(),
            "fraction_valid": float(valid.float().mean()),
            "median_distinct_cameras": _median(cameras),
            "camera_support": _quantiles(cameras),
            "fraction_with_at_least_3_cameras": float((cameras >= 3).float().mean()) if cameras.numel() else 0.0,
            "condition_number": _quantiles(condition_number[valid]),
            "conditioning_ratio": _quantiles(conditioning[valid]),
            "fraction_rejected_after_base_checks": float((base_valid & ~valid).sum() / base_valid.sum()) if bool(base_valid.any()) else None,
            "fraction_rejected_static_like": float(torch.as_tensor(rejected_static).float().mean()) if rejected_static is not None else None,
            "gt_static_acceptance_rate": None,
            "gt_moving_acceptance_rate": None,
            "static_false_positive_observation_rate": None,
            "moving_true_positive_observation_rate": None,
            "velocity_magnitude_ratio_gt": None,
            "velocity_direction_cosine_gt": None,
            "velocity_l2_error_gt": None,
            "projected_pixel_residual_gt": None,
            "static_observed_velocity_magnitude": None,
            "static_false_nonzero_target_rate": None,
            "rigid_residual": final_cycle.get("rigid_residual_median"),
            "rigid_residual_per_interval": _per_interval_median(rigid_residual, valid),
            "solver_observations": final_cycle.get("valid_observations"),
        },
        "rigid_mode": {
            "axis": summary["angular_mode"],
            "axis_cosine_gt": axis_cosine,
            "axis_angular_error_degrees": axis_error_degrees,
            "condition_number": final_cycle.get("condition_number"),
            "rigid_residual": final_cycle.get("rigid_residual_median"),
            "solver_observations": final_cycle.get("valid_observations"),
            "reference_velocity": summary["reference_velocity"],
            "screw_pitch": float(torch.dot(reference_velocity, angular) / angular_norm2) if angular_norm2 > 0 else None,
            "angular_mode_error_gt": validation.get("angular_velocity_error"),
            "translation_mode_error_gt": None,
            "pitch_error_gt": None,
        },
        "speed": {
            "mean": float(speed.mean()),
            "mean_gt": float(gt_speed.mean()) if gt_speed is not None else None,
            "mean_error_gt": (
                float((speed.mean() - gt_speed.mean()).abs())
                if gt_speed is not None
                else None
            ),
            "median": float(speed.median()),
            "std": float(speed.std(unbiased=False)),
            "min": float(speed.min()),
            "max": float(speed.max()),
            "curve": speed.tolist(),
            "mae_gt": float((speed - gt_speed).abs().mean()) if gt_speed is not None else None,
            "max_error_gt": validation.get("interval_speed_error_max"),
            "smoothness": float(torch.diff(speed).square().mean()) if speed.numel() > 1 else 0.0,
            "integrated_motion": float(torch.as_tensor(speed_artifact.get("delta_q", speed / 30.0)).sum()),
            "integrated_angle_gt": validation.get("gt_integrated_angle"),
            "integrated_angle_error_gt": validation.get(
                "integrated_angle_absolute_error"
            ),
            "ratio_vs_fixed_neutral_baseline": None,
        },
        "assignment": {
            "precision": validation.get("assignment_precision"),
            "recall": validation.get("assignment_recall"),
            "iou": validation.get("assignment_iou"),
            "accuracy": accuracy,
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "tn": tn,
            "ambiguous_fraction": summary.get("ambiguous_fraction"),
            "responsibility_entropy": summary.get("responsibility_entropy"),
            "predicted_dynamic_count": int(final_mask.sum()),
            "gt_dynamic_count": tp + fn if tp is not None and fn is not None else None,
            "initial_core_precision": validation.get("initial_dynamic_core_precision"),
            "initial_core_recall": validation.get("initial_dynamic_core_recall"),
            "final_precision": validation.get("assignment_precision"),
            "final_recall": validation.get("assignment_recall"),
            "final_iou": validation.get("assignment_iou"),
            "core_promotions": core_promotions,
            "core_demotions": core_demotions,
            "label_changes_per_outer_cycle": [item.get("label_changes") for item in history],
            "iou_per_outer_cycle": None,
        },
        "temporal_label_quality": temporal,
        "physical_rigidity": {
            "pairwise_distance_error": summary.get("pairwise_distance_error_max"),
            "deformation_frobenius_max": summary.get("deformation_frobenius_max"),
            "omega_spatial_variance_max": summary.get("omega_spatial_variance_max"),
            "rigidity_matrix_residual_max": summary.get("rigidity_matrix_residual_max"),
            "quaternion_norm_error": summary.get("quaternion_norm_error_max"),
        },
        "trajectory": {
            "moving_final_median_error": validation.get("moving_final_position_error_median"),
            "moving_final_p90_error": None,
            "moving_final_max_error": validation.get("moving_final_position_error_max"),
            "moving_trajectory_rmse": None,
            "true_static_median_displacement": None,
            "true_static_max_displacement": validation.get("true_static_final_displacement_max"),
            "moving_gaussians_leaving_valid_support": 0 if lagrangian else None,
        },
        "occlusion_robustness": {
            "bins": None,
            "unavailable_reason": "post-solve per-Gaussian GT labels are not stored",
        },
        "screen_footprint": {
            "mean_raster_ownership_weight": _quantiles(
                torch.as_tensor(audit["weight_sum"]).float().mean(dim=0)
                if "weight_sum" in audit
                else torch.empty(0)
            ),
            "mean_observation_count_per_support": _quantiles(observation_count.mean(dim=0)),
            "number_observing_cameras": _quantiles(observing_cameras),
        },
        "runtime": {
            "wall_seconds": wall_seconds,
            "measured_phase_seconds": measured_total,
            "visibility_pretraining_seconds": pretrain_runtime.get("visibility_seconds"),
            "flow_evidence_pretraining_seconds": pretrain_runtime.get("flow_evidence_seconds"),
            **runtime,
            "peak_gpu_memory_bytes": peak_gpu,
            "debug_artifact_bytes": sum(path.stat().st_size for path in root.iterdir() if path.is_file()),
        },
        "representation_size": {
            "support_units": int(valid.shape[1]),
            "flow_systems": int(valid.numel()),
            "accepted_observations": int(valid.sum()),
            "effective_observation_density": float(valid.float().mean()),
        },
        "dino_boundary": validation.get("dino_part_boundary"),
        "eulerian_support": (
            {
                "mixed_cell": _artifact_scalars(
                    eulerian_support.get("mixed_cell_diagnostics")
                ),
                "pooling_benefit": _artifact_scalars(
                    eulerian_support.get("pooling_benefit")
                ),
                "median_purity": eulerian_support.get("median_support_purity"),
                "p10_purity": eulerian_support.get("p10_support_purity"),
                "fraction_purity_gt_0_90": eulerian_support.get(
                    "fraction_purity_gt_0_90"
                ),
                "fraction_purity_gt_0_95": eulerian_support.get(
                    "fraction_purity_gt_0_95"
                ),
                "support_count": int(
                    torch.as_tensor(eulerian_support["parent_cell_id"]).numel()
                )
                if "parent_cell_id" in eulerian_support
                else None,
            }
            if eulerian_support
            else None
        ),
        "responsibility_artifact_available": bool(responsibility),
        "mode_change_per_outer_cycle": [item.get("mode_change") for item in history],
        "gt_evaluation_available": bool(validation),
        "ground_truth_used_as_solver_input": validation.get("ground_truth_used_as_solver_input", False),
    }


def _get(metrics: dict, path: str):
    value = metrics
    for component in path.split("."):
        if not isinstance(value, dict) or component not in value:
            return None
        value = value[component]
    return value


def _winner(eulerian: dict, lagrangian: dict, path: str, larger: bool) -> str:
    e, l = _get(eulerian, path), _get(lagrangian, path)
    if e is None or l is None or e == l:
        return "tie_or_unavailable"
    return "lagrangian" if (l > e) == larger else "eulerian"


def _delta(eulerian: dict, lagrangian: dict, path: str, *, error: bool = False):
    e, l = _get(eulerian, path), _get(lagrangian, path)
    if e is None or l is None:
        return None
    return e - l if error else l - e


def _plots(output_dir: Path, eulerian: dict, lagrangian: dict) -> tuple[list[str], dict]:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return [], {"all": "matplotlib is unavailable"}
    plots, unavailable = [], {}
    series = (
        ("speed_vs_frame.png", "Interval speed", "speed.curve"),
        ("valid_support_vs_frame.png", "Valid support units", "observation.number_valid_per_interval"),
        ("rigid_residual_vs_frame.png", "Median rigid residual", "observation.rigid_residual_per_interval"),
        ("assignment_iou_vs_outer_cycle.png", "Assignment IoU", "assignment.iou_per_outer_cycle"),
        ("mode_change_vs_outer_cycle.png", "Relative mode change", "mode_change_per_outer_cycle"),
        ("trajectory_error_vs_frame.png", "Trajectory error", "trajectory.error_per_frame"),
    )
    for filename, ylabel, path in series:
        left, right = _get(eulerian, path), _get(lagrangian, path)
        if left is None or right is None:
            unavailable[filename] = "required per-frame/cycle values were not saved"
            continue
        figure, axis = plt.subplots(figsize=(7, 4))
        axis.plot(left, label="Eulerian")
        axis.plot(right, label="Lagrangian")
        axis.set_xlabel("Interval / outer cycle")
        axis.set_ylabel(ylabel)
        axis.legend()
        figure.tight_layout()
        figure.savefig(output_dir / filename, dpi=140)
        plt.close(figure)
        plots.append(filename)
    runtime_keys = (
        "visibility_pretraining_seconds",
        "flow_stat_accumulation_seconds",
        "rigid_solver_seconds",
        "assignment_seconds",
        "final_replay_seconds",
    )
    e_values = [_get(eulerian, f"runtime.{key}") for key in runtime_keys]
    l_values = [_get(lagrangian, f"runtime.{key}") for key in runtime_keys]
    if all(value is not None for value in e_values + l_values):
        x = torch.arange(len(runtime_keys)).numpy()
        figure, axis = plt.subplots(figsize=(9, 4))
        axis.bar(x - 0.2, e_values, width=0.4, label="Eulerian")
        axis.bar(x + 0.2, l_values, width=0.4, label="Lagrangian")
        axis.set_xticks(x, [key.replace("_seconds", "") for key in runtime_keys], rotation=20, ha="right")
        axis.set_ylabel("Seconds")
        axis.legend()
        figure.tight_layout()
        figure.savefig(output_dir / "runtime_breakdown.png", dpi=140)
        plt.close(figure)
        plots.append("runtime_breakdown.png")
    else:
        unavailable["runtime_breakdown.png"] = "older run lacks phase timing fields"
    for filename in (
        "static_false_acceptance_vs_frame.png",
        "moving_velocity_error_vs_frame.png",
    ):
        unavailable[filename] = "requires post-solve per-support GT audit; GT is never loaded by the solver"
    return plots, unavailable


def _scalar_metrics(prefix: str, value, output: dict) -> None:
    if isinstance(value, dict):
        for key, item in value.items():
            _scalar_metrics(f"{prefix}/{key}" if prefix else key, item, output)
    elif isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value):
        output[prefix] = value


def _log_wandb(args, report: dict, plot_dir: Path) -> None:
    if not args.wandb:
        return
    import wandb

    run = wandb.init(
        project=args.wandb_project,
        entity=args.wandb_entity,
        name=args.wandb_name,
        mode=args.wandb_mode,
        job_type="eulerian_lagrangian_comparison",
    )
    payload = {}
    _scalar_metrics("eulerian", report["eulerian"], payload)
    _scalar_metrics("lagrangian", report["lagrangian"], payload)
    _scalar_metrics("comparison", report["deltas"], payload)
    for filename in report["plots"]:
        payload[f"comparison/plots/{Path(filename).stem}"] = wandb.Image(plot_dir / filename)
    run.log(payload)
    run.finish()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--eulerian_run", required=True)
    parser.add_argument("--lagrangian_run", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--fixed_neutral_baseline",
        help="Optional frozen neutral run used only for post-solve speed-drag ratio.",
    )
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--wandb_mode", choices=("online", "offline", "disabled"), default="online")
    parser.add_argument("--wandb_project", default="RigEuler3DGS")
    parser.add_argument("--wandb_entity")
    parser.add_argument("--wandb_name", default="eulerian-vs-lagrangian")
    args = parser.parse_args()
    output = Path(args.output).expanduser().resolve()
    output_file = output if output.suffix == ".json" else output / "phase1_eulerian_vs_lagrangian_comparison.json"
    output_file.parent.mkdir(parents=True, exist_ok=True)
    eulerian = _run_metrics(_debug_dir(args.eulerian_run), False)
    lagrangian = _run_metrics(_debug_dir(args.lagrangian_run), True)
    baseline = None
    if args.fixed_neutral_baseline:
        baseline_summary = _json(
            _debug_dir(args.fixed_neutral_baseline),
            ("phase1_lagrangian_solver_summary.json", "phase1_rigid_solver_summary.json"),
        )
        baseline = float(baseline_summary["speed_mean"])
        if abs(baseline) > 1.0e-12:
            eulerian["speed"]["ratio_vs_fixed_neutral_baseline"] = (
                eulerian["speed"]["mean"] / baseline
            )
            lagrangian["speed"]["ratio_vs_fixed_neutral_baseline"] = (
                lagrangian["speed"]["mean"] / baseline
            )
    report = {
        "schema_version": 2,
        "ground_truth_role": "post_solve_evaluation_only",
        "delta_convention": {
            "larger_is_better": "lagrangian_minus_eulerian",
            "errors": "eulerian_minus_lagrangian; positive means Lagrangian improved",
        },
        "fixed_neutral_baseline_speed_mean": baseline,
        "eulerian": eulerian,
        "lagrangian": lagrangian,
        "deltas": {
            "delta_assignment_precision": _delta(eulerian, lagrangian, "assignment.precision"),
            "delta_assignment_recall": _delta(eulerian, lagrangian, "assignment.recall"),
            "delta_assignment_iou": _delta(eulerian, lagrangian, "assignment.iou"),
            "delta_axis_cosine": _delta(eulerian, lagrangian, "rigid_mode.axis_cosine_gt"),
            "delta_speed_error": _delta(eulerian, lagrangian, "speed.mae_gt", error=True),
            "delta_static_displacement": _delta(eulerian, lagrangian, "trajectory.true_static_max_displacement", error=True),
            "delta_moving_final_error": _delta(eulerian, lagrangian, "trajectory.moving_final_median_error", error=True),
            "delta_rigid_residual": _delta(eulerian, lagrangian, "observation.rigid_residual", error=True),
            "delta_runtime": _delta(eulerian, lagrangian, "runtime.measured_phase_seconds", error=True),
            "delta_peak_memory": _delta(eulerian, lagrangian, "runtime.peak_gpu_memory_bytes", error=True),
        },
        "winners": {
            "observation_quality": _winner(eulerian, lagrangian, "observation.rigid_residual", False),
            "assignment": _winner(eulerian, lagrangian, "assignment.iou", True),
            "rigid_mode": _winner(eulerian, lagrangian, "rigid_mode.axis_cosine_gt", True),
            "trajectory": _winner(eulerian, lagrangian, "trajectory.moving_final_median_error", False),
            "occlusion_robustness": "tie_or_unavailable",
            "runtime": _winner(eulerian, lagrangian, "runtime.measured_phase_seconds", False),
            "memory": _winner(eulerian, lagrangian, "runtime.peak_gpu_memory_bytes", False),
            "overall_physical_reconstruction": _winner(eulerian, lagrangian, "trajectory.moving_final_median_error", False),
        },
    }
    report["plots"], report["unavailable_plots"] = _plots(output_file.parent, eulerian, lagrangian)
    output_file.write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
    _log_wandb(args, report, output_file.parent)
    print(output_file)


if __name__ == "__main__":
    main()
