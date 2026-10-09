import math
import sys
from tempfile import TemporaryDirectory
from pathlib import Path
from types import SimpleNamespace

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from utils.sam3_gaussian_affinity import (
    SAM3MaskPack,
    SAM3MaskCache,
    _root_has_mask_cache,
    compute_rendered_affinity,
    compute_target_affinity,
    gather_membership_matrix,
    gaussian_affinity,
    prepare_foreground_sam3_masks,
    sample_balanced_pairs,
    sample_foreground_pixels,
)


def _args(**overrides):
    values = {
        "sam3_mask_min_fg_fraction": 0.0,
        "sam3_mask_max_fg_fraction": 1.0,
        "sam3_mask_use_confidence": False,
        "sam3_affinity_pos_threshold": 0.75,
        "sam3_affinity_neg_threshold": 0.25,
        "sam3_affinity_positive_pairs": 8,
        "sam3_affinity_intermediate_pairs": 8,
        "sam3_affinity_negative_pairs": 8,
        "sam3_affinity_hard_pair_fraction": 0.0,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def test_overlapping_membership_affinity_identical_and_disjoint():
    H = torch.tensor([[1.0, 0.0, 1.0], [1.0, 0.0, 1.0], [0.0, 1.0, 0.0]])
    A = compute_target_affinity(H)
    assert torch.allclose(A[0, 1], torch.tensor(1.0), atol=1.0e-6)
    assert torch.allclose(A[0, 2], torch.tensor(0.0), atol=1.0e-6)


def test_partial_overlap_is_between_zero_and_one():
    H = torch.tensor([[1.0, 1.0, 0.0], [1.0, 0.0, 0.0]])
    value = float(compute_target_affinity(H)[0, 1])
    assert 0.0 < value < 1.0
    assert math.isclose(value, 1.0 / math.sqrt(2.0), rel_tol=1.0e-5)


def test_mask_id_permutation_invariant():
    H = torch.tensor([[1.0, 0.0, 1.0], [1.0, 1.0, 0.0], [0.0, 1.0, 0.0]])
    A0 = compute_target_affinity(H)
    A1 = compute_target_affinity(H[:, [2, 0, 1]])
    assert torch.allclose(A0, A1, atol=1.0e-6)


def test_zero_membership_pixel_excluded_by_sampler():
    masks = torch.zeros((1, 3, 3))
    masks[0, 1, 1] = 1.0
    pack = SAM3MaskPack(masks=masks, scores=None, foreground_fraction=None, image_hw=(3, 3), mask_hw=(3, 3))
    prepared = prepare_foreground_sam3_masks(pack, torch.ones((3, 3)), _args())
    pixels = sample_foreground_pixels(prepared.foreground_mask, prepared.valid_mask, 16)
    assert pixels.tolist() == [[1, 1]]
    H = gather_membership_matrix(prepared.masks, prepared.weights, pixels)
    assert torch.equal(torch.linalg.norm(H, dim=-1) > 0, torch.ones(1, dtype=torch.bool))


def test_foreground_exclusion_clips_background_membership():
    masks = torch.ones((1, 2, 2))
    foreground = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    pack = SAM3MaskPack(masks=masks, scores=None, foreground_fraction=None, image_hw=(2, 2), mask_hw=(2, 2))
    prepared = prepare_foreground_sam3_masks(pack, foreground, _args())
    assert prepared.masks[0, 0, 1] == 0
    assert prepared.masks[0, 1, 0] == 0
    assert prepared.valid_mask.tolist() == [[True, False], [False, True]]


def test_camera_local_mask_counts_need_no_shared_namespace():
    H0 = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    H1 = torch.tensor([[1.0, 0.0, 1.0], [0.0, 1.0, 0.0]])
    assert compute_target_affinity(H0).shape == (2, 2)
    assert compute_target_affinity(H1).shape == (2, 2)


def test_balanced_pairs_ignore_diagonal_and_keep_bins():
    A = torch.tensor(
        [
            [1.0, 1.0, 0.5, 0.0],
            [1.0, 1.0, 0.5, 0.0],
            [0.5, 0.5, 1.0, 0.2],
            [0.0, 0.0, 0.2, 1.0],
        ]
    )
    pairs = sample_balanced_pairs(A, torch.ones_like(A, dtype=torch.bool), _args())
    assert pairs.numel() > 0
    assert not torch.any(pairs[:, 0] == pairs[:, 1])
    selected = A[pairs[:, 0], pairs[:, 1]]
    assert bool((selected >= 0.75).any())
    assert bool(((selected > 0.25) & (selected < 0.75)).any())
    assert bool((selected <= 0.25).any())


def test_independent_endpoint_orthogonal_space_is_valid_within_field():
    feat0 = torch.randn(10, 4)
    q, _ = torch.linalg.qr(torch.randn(4, 4))
    feat1 = feat0 @ q
    A0 = compute_rendered_affinity(feat0)
    A1 = compute_rendered_affinity(feat1)
    assert torch.allclose(A0, A1, atol=1.0e-5)
    idx = torch.tensor([0, 1, 2])
    assert torch.allclose(
        gaussian_affinity(feat0, idx, idx + 1),
        gaussian_affinity(feat1, idx, idx + 1),
        atol=1.0e-5,
    )


def test_dense_feature_pt_is_not_mask_cache():
    with TemporaryDirectory() as directory:
        root = Path(directory)
        dense_root = root / "sam3" / "cam_000"
        dense_root.mkdir(parents=True)
        torch.save({"features": torch.randn(4, 8, 8), "image_hw": (8, 8)}, dense_root / "frame_000000.pt")
        assert not _root_has_mask_cache(root / "sam3")
        camera = SimpleNamespace(dynamic_camera_id="cam_000", frame_id=0, image_name="frame_000000")
        try:
            SAM3MaskCache(root / "sam3").get(camera)
        except RuntimeError as exc:
            assert "dense SAM3 feature root" in str(exc)
        else:
            raise AssertionError("dense SAM3 feature payload was accepted as a mask cache")


def test_mask_cache_root_detection_accepts_masks():
    with TemporaryDirectory() as directory:
        root = Path(directory)
        mask_root = root / "sam3_masks" / "cam_000"
        mask_root.mkdir(parents=True)
        torch.save({"masks": torch.ones(1, 4, 4), "image_hw": (4, 4)}, mask_root / "frame_000000.pt")
        assert _root_has_mask_cache(root / "sam3_masks")


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
    print("sam3_gaussian_affinity tests passed")
