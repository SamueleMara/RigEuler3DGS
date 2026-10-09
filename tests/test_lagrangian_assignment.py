import torch

from utils.gaussian_sequence_assignment import (
    DYNAMIC_CANDIDATE,
    DYNAMIC_CORE,
    STATIC_CORE,
    UNKNOWN,
    update_candidate_states,
)


def test_lagrangian_assignment_remains_one_state_per_gaussian():
    previous = torch.tensor((STATIC_CORE, DYNAMIC_CORE, UNKNOWN), dtype=torch.uint8)
    hypothesis = {
        "margin": torch.tensor((1.0, 1.0, 1.0)),
        "time_persistence": torch.ones(3),
        "camera_persistence": torch.ones(3),
    }
    result = update_candidate_states(previous, hypothesis)
    assert result["state"].shape == previous.shape
    assert result["state"].tolist() == [DYNAMIC_CANDIDATE, DYNAMIC_CORE, DYNAMIC_CANDIDATE]

