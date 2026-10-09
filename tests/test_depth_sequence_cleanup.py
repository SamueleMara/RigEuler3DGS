import torch

from utils.temporal_visibility_pretraining import (
    OCCLUDED,
    PRESENT,
    UNKNOWN,
    VACATED,
    cleanup_state_sequences,
)


def _sequence(values):
    return torch.tensor(values, dtype=torch.uint8)[:, None, None]


def test_isolated_vacated_flip_is_removed():
    cleaned = cleanup_state_sequences(
        _sequence((PRESENT, PRESENT, VACATED, PRESENT, PRESENT)),
        vacated_min_run=2,
    )
    assert cleaned.flatten().tolist() == [PRESENT] * 5


def test_short_vacated_run_becomes_unknown_but_occlusion_is_preserved():
    cleaned = cleanup_state_sequences(
        _sequence((PRESENT, VACATED, VACATED, OCCLUDED, OCCLUDED)),
        vacated_min_run=3,
        remove_isolated_flips=False,
    )
    assert cleaned.flatten().tolist() == [PRESENT, UNKNOWN, UNKNOWN, OCCLUDED, OCCLUDED]

