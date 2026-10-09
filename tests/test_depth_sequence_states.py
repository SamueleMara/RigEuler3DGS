import torch

from utils.temporal_visibility_pretraining import (
    OCCLUDED,
    PRESENT,
    UNKNOWN,
    VACATED,
    classify_depth_states,
)


def test_measured_depth_states_have_physical_meaning():
    reference = torch.ones((1, 4))
    observed = torch.tensor(
        (
            ((1.0, 1.0, 1.0, 1.0),),
            ((1.0, 0.7, 1.3, 0.0),),
        )
    )
    state = classify_depth_states(
        reference,
        observed,
        torch.ones_like(reference, dtype=torch.bool),
        present_abs=0.01,
        present_rel=0.01,
        occlusion_abs=0.05,
        vacated_abs=0.05,
    )
    assert state[0].tolist() == [[PRESENT] * 4]
    assert state[1].tolist() == [[PRESENT, OCCLUDED, VACATED, UNKNOWN]]

