import sys
from pathlib import Path
from types import SimpleNamespace

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scene.render_guided_motion_probe import (
    DYNAMIC,
    STATIC,
    UNKNOWN,
    RenderDenoiseConfig,
    _cell_state,
    _seed_consistency,
    _view_consistency,
    material_assignment_from_render_evidence,
)


def _config():
    return RenderDenoiseConfig(
        probe_frames=[1, 3],
        cameras_per_probe=4,
        min_support_cameras=2,
        min_informative_frames=2,
        num_seeds=3,
        denoise_steps=4,
        noise_sigma=0.01,
        lr=1.0e-2,
        max_displacement=0.1,
        lambda_rgb=1.0,
        lambda_ssim=0.2,
        lambda_depth=0.1,
        lambda_zero=0.01,
        lambda_neighbor=0.001,
        support_threshold=0.05,
        seed_consistency_threshold=0.65,
        view_consistency_threshold=0.55,
        dynamic_threshold=0.75,
        static_threshold=0.25,
        confidence_threshold=0.60,
        gate_num_seeds=3,
        gate_steps=4,
        gate_lr=1.0e-2,
        gate_temperature=1.0,
        gate_noise_sigma=0.5,
        lambda_gate_prior=0.05,
        lambda_gate_neighbor=0.001,
        lambda_gate_entropy=0.01,
    )


def test_cell_state_keeps_unsupported_points_unknown():
    cfg = _config()
    dynamic = torch.tensor([0.9, 0.1, 0.9])
    static = torch.tensor([0.1, 0.9, 0.1])
    confidence = torch.tensor([0.9, 0.9, 0.9])
    support = torch.tensor([2, 2, 1])

    probability, state = _cell_state(dynamic, static, confidence, support, cfg)

    assert torch.allclose(probability, torch.tensor([0.9, 0.1, 0.9]))
    assert state.tolist() == [DYNAMIC, STATIC, UNKNOWN]


def test_seed_and_view_consistency_require_repeatable_direction():
    aligned = torch.tensor(
        [
            [[1.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
            [[2.0, 0.0, 0.0], [-1.0, 0.0, 0.0]],
        ]
    )
    weights = torch.ones(2, 2)

    consistency = _seed_consistency(aligned, weights)

    assert consistency[0] > 0.99
    assert consistency[1] < 0.01

    view_consistency, count = _view_consistency(
        aligned,
        support=torch.ones(2, 2),
        min_cameras=2,
        threshold=0.5,
    )

    assert count.tolist() == [2, 2]
    assert view_consistency[0] > 0.99
    assert view_consistency[1] < 0.01


def test_material_assignment_uses_unknown_as_soft_not_static_or_dynamic():
    class DummyGaussians:
        @property
        def get_canonical_xyz(self):
            return torch.zeros(3, 3)

    args = SimpleNamespace(
        num_moving_parts=1,
        render_denoise_dynamic_threshold=0.75,
        eulerian_assignment_init_floor=0.20,
        eulerian_assignment_init_ceiling=0.80,
    )
    assignment = material_assignment_from_render_evidence(
        DummyGaussians(),
        {
            "dynamic_score": torch.tensor([0.95, 0.05, 0.95]),
            "static_score": torch.tensor([0.05, 0.95, 0.05]),
            "unknown_score": torch.tensor([0.0, 0.0, 1.0]),
            "confidence": torch.tensor([1.0, 1.0, 0.0]),
            "state": torch.tensor([DYNAMIC, STATIC, UNKNOWN]),
            "final_dynamic_probability": torch.tensor([0.95, 0.05, 0.95]),
        },
        args,
    )

    probabilities = assignment.soft_probabilities()

    assert probabilities[0, 1] > probabilities[0, 0]
    assert probabilities[1, 0] > probabilities[1, 1]
    assert torch.isclose(probabilities[2, 0], probabilities[2, 1], atol=1.0e-4)


if __name__ == "__main__":
    test_cell_state_keeps_unsupported_points_unknown()
    test_seed_and_view_consistency_require_repeatable_direction()
    test_material_assignment_uses_unknown_as_soft_not_static_or_dynamic()
