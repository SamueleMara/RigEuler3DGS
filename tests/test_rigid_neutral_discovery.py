import torch

from utils.gaussian_sequence_assignment import (
    DYNAMIC_CORE,
    STATIC_CORE,
    UNKNOWN,
    classify_rigid_hypotheses_once,
)
from utils.unified_dynamic_training import initialize_neutral_motion_labels


class _Gaussians:
    get_canonical_xyz = torch.zeros((4, 3))


def test_neutral_initialization_has_no_hard_material_labels():
    state = initialize_neutral_motion_labels(_Gaussians())
    assert torch.equal(state["motion_evidence"], torch.full((4,), 0.5))
    assert not bool(state["initial_dynamic_mask"].any())


def test_one_shot_assignment_keeps_ambiguous_or_speed_incompatible_points_unknown():
    hypothesis = {
        "margin": torch.tensor((1.0, -1.0, 0.0, 1.0)),
        "informative_interval_count": torch.ones(4, dtype=torch.long),
        "time_persistence": torch.tensor((1.0, 0.0, 0.0, 1.0)),
        "static_time_persistence": torch.tensor((0.0, 1.0, 0.0, 0.0)),
        "camera_persistence": torch.tensor((1.0, 0.0, 0.0, 1.0)),
        "static_camera_persistence": torch.tensor((0.0, 1.0, 0.0, 0.0)),
    }
    result = classify_rigid_hypotheses_once(
        hypothesis,
        torch.tensor((True, True, True, False)),
        min_margin=0.35,
        min_time_fraction=0.6,
        min_camera_fraction=0.4,
    )
    assert result["state"].tolist() == [
        DYNAMIC_CORE,
        STATIC_CORE,
        UNKNOWN,
        UNKNOWN,
    ]
    assert float(result["dynamic_probability"][2]) == 0.5
