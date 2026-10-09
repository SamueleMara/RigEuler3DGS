from __future__ import annotations

import torch

from utils.semantic_foreground_transport import apply_foreground_domain, mask_sparse_pixel_transport, renormalize_foreground_ownership


def test_background_contributes_exactly_zero():
    value = torch.arange(4.0).reshape(2, 2)
    mask = torch.tensor([[1, 0], [0, 1]], dtype=torch.bool)
    out = apply_foreground_domain(value, mask)
    assert out[0, 1].item() == 0.0
    assert out[1, 0].item() == 0.0
    src, tgt, val = mask_sparse_pixel_transport(
        torch.tensor([0, 1, 3]),
        torch.tensor([3, 0, 1]),
        torch.ones(3),
        mask,
        mask,
    )
    assert src.tolist() == [0]
    assert tgt.tolist() == [3]
    assert val.tolist() == [1.0]


def test_foreground_ownership_renormalizes_after_masking():
    own = torch.tensor([[0.2, 0.8], [0.4, 0.6]])
    out = renormalize_foreground_ownership(own, foreground_gaussian_mask=torch.tensor([1, 0], dtype=torch.bool), gaussian_dim=1)
    assert torch.allclose(out[:, 0], torch.ones(2))
    assert torch.allclose(out[:, 1], torch.zeros(2))


if __name__ == "__main__":
    test_background_contributes_exactly_zero()
    test_foreground_ownership_renormalizes_after_masking()
    print("PASS semantic_foreground_transport")
