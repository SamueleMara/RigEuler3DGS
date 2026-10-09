import torch

from utils.temporal_pixel_fm import match_foreground_pixels_to_next


def test_local_foreground_search_does_not_scan_whole_image_or_use_background():
    h = w = 5
    source = torch.zeros(3, h, w)
    target = torch.zeros(3, h, w)
    source[:, 2, 2] = 1.0
    target[:, 2, 3] = 1.0
    target[:, 4, 4] = 1.0  # identical appearance, but background and far away
    depth = torch.ones(h, w)
    source_mask = torch.zeros(h, w, dtype=torch.bool)
    target_mask = torch.zeros(h, w, dtype=torch.bool)
    source_mask[2, 2] = True
    target_mask[2, 3] = True
    field = match_foreground_pixels_to_next(
        source,
        target,
        depth,
        depth,
        source_mask,
        target_mask,
        source_pixel_indices=torch.tensor([2 * w + 2]),
        delta_t=1.0,
        candidates=25,
        search_radius=2,
    )
    assert field.source_pixel_indices.tolist() == [2 * w + 2]
    assert field.target_pixel_indices.tolist() == [2 * w + 3]
    assert torch.allclose(field.velocity, torch.tensor([[1.0, 0.0]]))


def test_confidence_requires_appearance_distinctiveness_not_zero_offset():
    h = w = 5
    source = torch.zeros(3, h, w)
    target = torch.zeros(3, h, w)
    depth = torch.ones(h, w)
    source_mask = torch.zeros(h, w, dtype=torch.bool)
    target_mask = torch.zeros(h, w, dtype=torch.bool)
    source_mask[2, 2] = True
    target_mask[2, 2] = True
    target_mask[2, 3] = True
    field = match_foreground_pixels_to_next(
        source,
        target,
        depth,
        depth,
        source_mask,
        target_mask,
        source_pixel_indices=torch.tensor([2 * w + 2]),
        delta_t=1.0,
        candidates=25,
        search_radius=2,
    )
    assert field.target_pixel_indices.tolist() == [2 * w + 2]
    assert float(field.confidence[0]) == 0.0


def main():
    test_local_foreground_search_does_not_scan_whole_image_or_use_background()
    test_confidence_requires_appearance_distinctiveness_not_zero_offset()
    print("PASS temporal_pixel_fm_local_search")


if __name__ == "__main__":
    main()
