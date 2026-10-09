import torch

from utils.lagrangian_rigid_motion import compute_lagrangian_gradient_diagnostics
from utils.rigid_killing_motion import normalize_rigid_mode


def test_lagrangian_killing_gradient_is_non_deforming():
    generator = torch.Generator().manual_seed(9)
    points = torch.randn((80, 3), generator=generator)
    mode, _ = normalize_rigid_mode(
        points[3].double(),
        torch.tensor((0.1, 0.2, -0.1), dtype=torch.float64),
        torch.tensor((0.2, -0.4, 0.7), dtype=torch.float64),
    )
    result = compute_lagrangian_gradient_diagnostics(
        points,
        mode,
        torch.tensor((0.2, 0.5)),
        torch.ones(points.shape[0], dtype=torch.bool),
        neighbors_per_gaussian=12,
        max_pairs=2000,
    )
    valid = result["gradient_valid"]
    assert bool(valid.any())
    assert float(result["deformation_frobenius"][:, valid].max()) < 2e-3
    assert float(result["rigidity_matrix_residual"].max()) < 1e-5

