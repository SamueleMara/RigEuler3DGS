from types import SimpleNamespace
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch

from utils.kpart_velocity_field_discovery import _prior_anchor_masks


def check(condition, message):
    if not condition:
        raise AssertionError(message)


def test_strict_prior_uses_rank_split_for_active_domain():
    prior = {
        "method": "strict_closed_loop_endpoint_fm",
        "probability": torch.tensor([0.43, 0.49, 0.50, 0.53, 0.62, 0.73]),
        "confidence": torch.tensor([0.1, 0.2, 0.05, 0.3, 0.4, 0.5]),
        "valid": torch.ones(6, dtype=torch.bool),
    }
    args = SimpleNamespace(
        kpart_prior_anchor_threshold=0.75,
        kpart_confidence_threshold=0.2,
        kpart_prior_threshold_mode="otsu",
        kpart_prior_boundary_expansion=0,
        kpart_prior_cleanup_min_size=1,
    )
    masks = _prior_anchor_masks(prior, torch.empty(0, 2, dtype=torch.long), args)
    active = masks["active_mask"]
    threshold = masks["threshold"]
    check(int(active.sum()) == 4, "strict priors should use p>=0.5 as the active dynamic side")
    check(threshold["active_cut"] == 0.5, "strict-prior active cut should be the probability midpoint")
    check(not bool(masks["static_anchor"].any()), "strict prior must not create static anchors")


if __name__ == "__main__":
    test_strict_prior_uses_rank_split_for_active_domain()
    print("PASS kpart_prior_thresholds")
