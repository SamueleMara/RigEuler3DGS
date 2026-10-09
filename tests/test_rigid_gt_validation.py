import torch

from utils.rigid_killing_motion import normalize_rigid_mode
from utils.rigid_killing_validation import (
    MEASURED_FLOW_SOURCE_ID,
    RigidDatasetGroundTruth,
    validate_frozen_rigid_solution,
)


def _exact_fixture():
    points = torch.tensor(
        [[1.0, 0.0, 0.0], [0.0, 2.0, 0.0], [3.0, 3.0, 1.0]],
        dtype=torch.float64,
    )
    moving = torch.tensor([True, True, False])
    angle = torch.tensor(0.5, dtype=torch.float64)
    cosine, sine = torch.cos(angle), torch.sin(angle)
    rotation = torch.tensor(
        [[cosine, -sine, 0.0], [sine, cosine, 0.0], [0.0, 0.0, 1.0]],
        dtype=torch.float64,
    )
    final = points.clone()
    final[moving] = points[moving] @ rotation.T
    mode, _ = normalize_rigid_mode(
        torch.zeros(3, dtype=torch.float64),
        torch.zeros(3, dtype=torch.float64),
        torch.tensor([0.0, 0.0, 1.0], dtype=torch.float64),
    )
    ground_truth = RigidDatasetGroundTruth(
        reference_point=torch.zeros(3, dtype=torch.float64),
        axis=torch.tensor([0.0, 0.0, 1.0], dtype=torch.float64),
        frame_values={0: 0.0, 1: 0.5},
        moving_mask=moving,
        joint_name="synthetic_joint",
        moving_part=1,
    )
    return points, final, moving, mode, ground_truth


def test_gt_validation_reads_only_a_frozen_measured_flow_solution():
    points, final, moving, mode, ground_truth = _exact_fixture()
    report = validate_frozen_rigid_solution(
        ground_truth,
        mode,
        torch.tensor([0.5], dtype=torch.float64),
        [0, 1],
        [0.0, 1.0],
        points,
        [points, final],
        moving,
        solver_observation_source_id=MEASURED_FLOW_SOURCE_ID,
    )
    assert report["role"] == "post_solve_evaluation_only"
    assert report["ground_truth_used_as_solver_input"] is False
    assert report["acceptance_passed"] is True


def test_gt_validation_rejects_a_gt_solver_observation_source():
    points, final, moving, mode, ground_truth = _exact_fixture()
    try:
        validate_frozen_rigid_solution(
            ground_truth,
            mode,
            torch.tensor([0.5], dtype=torch.float64),
            [0, 1],
            [0.0, 1.0],
            points,
            [points, final],
            moving,
            solver_observation_source_id="analytic_ground_truth_flow",
        )
    except RuntimeError as error:
        assert "measured rasterized flow" in str(error)
    else:
        raise AssertionError("GT-derived solver observations must be rejected")
