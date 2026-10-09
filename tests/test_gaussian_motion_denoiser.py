#!/usr/bin/env python3
"""Small self-check for the frame01 Gaussian denoiser module."""

from pathlib import Path
import sys

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scene.gaussian_motion_denoiser import GaussianMotionDenoiser, build_canonical_knn


def main():
    torch.manual_seed(0)
    points = torch.randn(32, 3)
    knn, median = build_canonical_knn(points, k=6)
    assert knn.shape == (32, 6)
    assert torch.isfinite(median) and float(median) > 0.0

    model = GaussianMotionDenoiser(input_dim=14, hidden_dim=16, message_layers=1)
    features = torch.randn(32, 14, requires_grad=True)
    delta = model(features, knn, step_scale=0.05)
    assert delta.shape == (32, 3)
    assert float(delta.detach().abs().max()) <= 0.050001
    delta.square().mean().backward()
    assert features.grad is not None and torch.isfinite(features.grad).all()


if __name__ == "__main__":
    main()
