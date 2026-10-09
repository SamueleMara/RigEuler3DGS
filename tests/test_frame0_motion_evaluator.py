#!/usr/bin/env python3
"""CPU self-checks for frame0 null-corrected motion evaluator invariants."""

from pathlib import Path
import sys

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.frame0_motion_evaluator import compute_motion_fields, null_correct, trial_seeds


def main():
    raw = torch.tensor(
        [
            [[2.0, 0.0, 0.0], [0.5, 0.0, 0.0]],
            [[2.0, 0.0, 0.0], [0.5, 0.0, 0.0]],
        ],
        dtype=torch.float32,
    )
    null = torch.tensor(
        [
            [[0.5, 0.0, 0.0], [0.1, 0.0, 0.0]],
            [[0.5, 0.0, 0.0], [0.1, 0.0, 0.0]],
        ],
        dtype=torch.float32,
    )
    corrected = null_correct(raw, null)
    assert torch.allclose(corrected, raw - null)

    fields = compute_motion_fields(corrected, sigma_max=1.0)
    assert fields["label"].dtype == torch.bool
    assert torch.equal(fields["label"], fields["ratio"] > 1.0)
    assert fields["label"].tolist() == [True, False]
    assert torch.equal(trial_seeds(7, 3), torch.tensor([100007, 100008, 100009]))


if __name__ == "__main__":
    main()
