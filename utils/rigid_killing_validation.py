"""Post-solve ground-truth validation for the rigid Killing branch.

This module is deliberately downstream of motion estimation: it reads a frozen
solution and reports errors. It cannot provide observations, weights, labels,
or initialization to the solver.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
import json
from pathlib import Path
from typing import Sequence

import numpy as np
import torch

from utils.rigid_killing_motion import RigidKillingMode


MEASURED_FLOW_SOURCE_ID = "pair_observation_rasterized_flow_v1"
OCCLUSION_CALIBRATED_FLOW_SOURCE_ID = "pair_observation_occlusion_calibrated_flow_v2"


@dataclass(frozen=True)
class RigidDatasetGroundTruth:
    reference_point: torch.Tensor
    axis: torch.Tensor
    frame_values: dict[int, float]
    moving_mask: torch.Tensor
    joint_name: str
    moving_part: int
    part_labels: torch.Tensor | None = None


def load_rigid_dataset_ground_truth(
    sequence_root: str | Path,
    canonical_xyz: torch.Tensor,
    *,
    moving_part: int,
    gaussian_part_labels: torch.Tensor | None = None,
) -> RigidDatasetGroundTruth:
    """Load GT labels and trajectory for evaluation, never estimation."""
    root = Path(sequence_root).expanduser().resolve()
    metadata_path = root / "pointcloud" / "joint_metadata.json"
    points_path = root / "pointcloud" / "points3D_parts.ply"
    trajectory_path = root / "metadata" / "frame_values.csv"
    for path in (metadata_path, points_path, trajectory_path):
        if not path.is_file():
            raise FileNotFoundError(f"Rigid GT validation file is missing: {path}")

    with metadata_path.open("r", encoding="utf-8") as stream:
        metadata = json.load(stream)
    joint_name = str(metadata["name"])
    with trajectory_path.open("r", encoding="utf-8", newline="") as stream:
        rows = csv.DictReader(stream)
        if rows.fieldnames is None or joint_name not in rows.fieldnames:
            raise ValueError(
                f"Trajectory {trajectory_path} has no {joint_name!r} column"
            )
        frame_values = {
            int(row["frame_index"]): float(row[joint_name]) for row in rows
        }

    from plyfile import PlyData

    vertex = PlyData.read(str(points_path))["vertex"].data
    sparse_xyz = torch.from_numpy(
        np.stack((vertex["x"], vertex["y"], vertex["z"]), axis=-1).copy()
    ).double()
    sparse_part = torch.from_numpy(
        np.asarray(vertex["part"], dtype=np.int64).copy()
    )
    query = torch.as_tensor(canonical_xyz).detach().cpu().double()
    label_tensor = (
        torch.as_tensor(gaussian_part_labels).detach().cpu().long().reshape(-1)
        if gaussian_part_labels is not None
        else None
    )
    if label_tensor is not None and label_tensor.numel() == query.shape[0]:
        matched_parts = label_tensor
    else:
        nearest_parts = []
        for start in range(0, query.shape[0], 1024):
            nearest = torch.cdist(query[start : start + 1024], sparse_xyz).argmin(dim=1)
            nearest_parts.append(sparse_part[nearest])
        matched_parts = torch.cat(nearest_parts)

    axis = torch.as_tensor(metadata["axis_vector"], dtype=torch.float64)
    axis_norm = torch.linalg.norm(axis)
    if not bool(torch.isfinite(axis).all()) or float(axis_norm) <= 1.0e-12:
        raise ValueError("Rigid GT axis is invalid")
    return RigidDatasetGroundTruth(
        reference_point=torch.as_tensor(
            metadata["point_of_application"], dtype=torch.float64
        ),
        axis=axis / axis_norm,
        frame_values=frame_values,
        moving_mask=matched_parts == int(moving_part),
        joint_name=joint_name,
        moving_part=int(moving_part),
        part_labels=matched_parts,
    )


def _rotate_about_reference(
    points: torch.Tensor,
    reference_point: torch.Tensor,
    axis: torch.Tensor,
    angle: float,
) -> torch.Tensor:
    zero = torch.zeros((), dtype=torch.float64)
    x, y, z = axis
    skew = torch.stack(
        (
            torch.stack((zero, -z, y)),
            torch.stack((z, zero, -x)),
            torch.stack((-y, x, zero)),
        )
    )
    value = torch.tensor(float(angle), dtype=torch.float64)
    rotation = (
        torch.eye(3, dtype=torch.float64)
        + torch.sin(value) * skew
        + (1.0 - torch.cos(value)) * (skew @ skew)
    )
    return reference_point + (points - reference_point) @ rotation.T


def validate_frozen_rigid_solution(
    ground_truth: RigidDatasetGroundTruth,
    mode: RigidKillingMode,
    interval_speeds: torch.Tensor,
    frame_ids: Sequence[int],
    frame_times: Sequence[float],
    canonical_xyz: torch.Tensor,
    predicted_positions: Sequence[torch.Tensor],
    predicted_dynamic_mask: torch.Tensor,
    *,
    solver_observation_source_id: str,
    static_tolerance: float = 1.0e-6,
) -> dict:
    """Compare a frozen measured-flow result with GT without changing it."""
    if solver_observation_source_id not in {
        MEASURED_FLOW_SOURCE_ID,
        OCCLUSION_CALIBRATED_FLOW_SOURCE_ID,
    }:
        raise RuntimeError(
            "GT validation requires a solution produced from measured rasterized "
            f"flow, got {solver_observation_source_id!r}"
        )
    if len(frame_ids) != len(frame_times) or len(frame_ids) != len(predicted_positions):
        raise ValueError("Frame ids, times, and predicted positions must align")
    if len(frame_ids) < 2:
        raise ValueError("GT validation requires at least two frames")
    missing = [int(frame) for frame in frame_ids if int(frame) not in ground_truth.frame_values]
    if missing:
        raise ValueError(f"GT trajectory is missing frames: {missing}")

    xyz = torch.as_tensor(canonical_xyz).detach().cpu().double()
    history = torch.stack(
        [torch.as_tensor(value).detach().cpu().double() for value in predicted_positions]
    )
    predicted_mask = torch.as_tensor(predicted_dynamic_mask).detach().cpu().bool()
    gt_mask = ground_truth.moving_mask.detach().cpu().bool()
    if xyz.shape != history.shape[1:] or gt_mask.shape != predicted_mask.shape:
        raise ValueError("GT validation tensors do not match canonical Gaussians")
    if not bool(gt_mask.any()):
        raise ValueError(
            f"Moving GT part {ground_truth.moving_part} has no matched Gaussians"
        )

    values = torch.tensor(
        [ground_truth.frame_values[int(frame)] for frame in frame_ids],
        dtype=torch.float64,
    )
    times = torch.as_tensor(frame_times, dtype=torch.float64)
    dt = torch.diff(times)
    if bool((dt <= 0).any()):
        raise ValueError("GT validation frame times must increase strictly")
    gt_speeds = torch.diff(values) / dt
    speeds = torch.as_tensor(interval_speeds).detach().cpu().double()
    if speeds.shape != gt_speeds.shape:
        raise ValueError("Estimated and GT interval-speed counts differ")

    angular = mode.angular_mode.detach().cpu().double()
    angular_norm = torch.linalg.norm(angular)
    direction_cosine = (
        float(torch.dot(angular / angular_norm, ground_truth.axis))
        if mode.mode_type == "rotational" and float(angular_norm) > 1.0e-12
        else None
    )
    angular_velocity_error = torch.linalg.norm(
        speeds[:, None] * angular[None]
        - gt_speeds[:, None] * ground_truth.axis[None],
        dim=-1,
    )
    speed_error = torch.abs(speeds - gt_speeds)

    final_gt = xyz.clone()
    final_gt[gt_mask] = _rotate_about_reference(
        xyz[gt_mask],
        ground_truth.reference_point,
        ground_truth.axis,
        float(values[-1] - values[0]),
    )
    moving_position_error = torch.linalg.norm(
        history[-1, gt_mask] - final_gt[gt_mask], dim=-1
    )
    true_static_displacement = torch.linalg.norm(
        history[-1, ~gt_mask] - history[0, ~gt_mask], dim=-1
    )

    true_positive = int((predicted_mask & gt_mask).sum())
    false_positive = int((predicted_mask & ~gt_mask).sum())
    false_negative = int((~predicted_mask & gt_mask).sum())
    union = true_positive + false_positive + false_negative
    precision_denominator = true_positive + false_positive
    recall_denominator = true_positive + false_negative
    moving_median = float(moving_position_error.median())
    static_max = (
        float(true_static_displacement.max())
        if true_static_displacement.numel()
        else 0.0
    )
    speed_max = float(speed_error.max())
    criteria = {
        "angular_direction_cosine_gt_0_995": direction_cosine is not None
        and direction_cosine > 0.995,
        "interval_speed_error_lt_0_02": speed_max < 0.02,
        "true_static_displacement_le_tolerance": static_max <= float(static_tolerance),
        "moving_final_median_error_lt_0_02": moving_median < 0.02,
    }
    return {
        "schema_version": 1,
        "role": "post_solve_evaluation_only",
        "ground_truth_used_as_solver_input": False,
        "solver_observation_source_id": solver_observation_source_id,
        "joint_name": ground_truth.joint_name,
        "moving_part": ground_truth.moving_part,
        "frame_ids": [int(value) for value in frame_ids],
        "gt_axis": ground_truth.axis.tolist(),
        "gt_reference_point": ground_truth.reference_point.tolist(),
        "gt_interval_speeds": gt_speeds.tolist(),
        "estimated_interval_speeds": speeds.tolist(),
        "interval_speed_absolute_error": speed_error.tolist(),
        "interval_speed_error_max": speed_max,
        "angular_direction_cosine": direction_cosine,
        "angular_velocity_error": angular_velocity_error.tolist(),
        "moving_final_position_error_median": moving_median,
        "moving_final_position_error_max": float(moving_position_error.max()),
        "true_static_final_displacement_max": static_max,
        "assignment_true_positive": true_positive,
        "assignment_false_positive": false_positive,
        "assignment_false_negative": false_negative,
        "assignment_precision": (
            true_positive / precision_denominator if precision_denominator else None
        ),
        "assignment_recall": (
            true_positive / recall_denominator if recall_denominator else None
        ),
        "assignment_iou": true_positive / union if union else None,
        "acceptance": criteria,
        "acceptance_passed": all(criteria.values()),
    }
