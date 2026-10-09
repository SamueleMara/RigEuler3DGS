import torch

from utils.temporal_visibility_pretraining import (
    DYNAMIC_PIXEL,
    OCCLUDED,
    PRESENT,
    STATIC_PIXEL,
    UNKNOWN_PIXEL,
    VACATED,
    classify_pixels_depth_only,
    compute_pixel_sequence_stats,
)


def _classify(values, minimum=3):
    state = torch.tensor(values, dtype=torch.uint8)[:, None, None]
    stats = compute_pixel_sequence_stats(state)
    return int(
        classify_pixels_depth_only(
            stats,
            min_observable_frames=minimum,
            static_present_ratio=0.9,
            dynamic_vacated_ratio=0.3,
            vacated_min_run=2,
        )[0, 0]
    )


def test_temporary_occlusion_does_not_vote_dynamic():
    assert _classify((PRESENT, PRESENT, OCCLUDED, OCCLUDED, PRESENT)) == STATIC_PIXEL


def test_persistent_vacating_votes_dynamic():
    assert _classify((PRESENT, PRESENT, VACATED, VACATED, VACATED)) == DYNAMIC_PIXEL


def test_occlusion_after_vacating_does_not_erase_dynamic_evidence():
    assert _classify(
        (PRESENT, PRESENT, VACATED, VACATED, OCCLUDED, OCCLUDED, VACATED, VACATED)
    ) == DYNAMIC_PIXEL


def test_vacated_surface_may_reappear_and_remain_dynamic():
    assert _classify(
        (PRESENT, PRESENT, VACATED, VACATED, PRESENT, PRESENT)
    ) == DYNAMIC_PIXEL


def test_mostly_occluded_pixel_stays_unknown():
    assert _classify((PRESENT, OCCLUDED, OCCLUDED, OCCLUDED, PRESENT)) == UNKNOWN_PIXEL
