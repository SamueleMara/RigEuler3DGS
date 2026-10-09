"""§40 check 7: training-view calibration and predictive calibration differ.

Fitting the render residual on the views the endpoint 3DGS was optimised on
measures reconstruction overfit, not observation noise.  On USB_100109 that
gave 2.9e-4 -- so small that any displacement produced an astronomical
penalty and the likelihood test could only ever answer "static", whatever the
hypothesis.

The correction uses frame-T cameras, which the frame-0 model never saw, and
contrasts the moved Gaussians against those same held-out images.  That makes
the test both predictive and genuinely contrastive: the static hypothesis no
longer wins by having been trained on the pixels it is scored against.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from utils.render_hypothesis_likelihood import (
    ResidualModel,
    compare_images_only,
    compute_hypothesis_log_likelihood,
    transform_patch_gaussians,
)

CHECKS: list[str] = []


def check(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)
    CHECKS.append(message)


def _fit_scale(residual: torch.Tensor) -> float:
    v = residual.reshape(-1)
    return float((v - v.median()).abs().median() * 1.4826)


def test_training_and_predictive_calibration_differ() -> None:
    generator = torch.Generator().manual_seed(1)
    observed = torch.rand((3, 64, 64), generator=generator)

    # A model fitted to these views reproduces them almost exactly...
    training_render = observed + torch.randn(observed.shape, generator=generator) * 0.0003
    # ...but on views it never saw, the error is the real predictive error.
    heldout_observed = torch.rand((3, 64, 64), generator=generator)
    heldout_render = heldout_observed + torch.randn(heldout_observed.shape, generator=generator) * 0.02

    training_scale = _fit_scale(observed - training_render)
    predictive_scale = _fit_scale(heldout_observed - heldout_render)
    check(
        predictive_scale > 10.0 * training_scale,
        f"predictive residual scale is far larger than training-view scale "
        f"({predictive_scale:.5f} vs {training_scale:.5f})",
    )

    # The consequence: under the training-view scale, a modest real motion is
    # rejected with astronomical confidence; under the predictive scale the
    # same evidence is evaluated sensibly.
    static = heldout_observed + 0.05
    dynamic = heldout_render
    tight = ResidualModel(training_scale, 4.0, "student_t", 1)
    fair = ResidualModel(predictive_scale, 4.0, "student_t", 1)
    tight_ratio = compare_images_only(
        {"rgb": heldout_observed}, {"rgb": static}, {"rgb": dynamic}, {"rgb": tight}
    ).log_likelihood_ratio
    fair_ratio = compare_images_only(
        {"rgb": heldout_observed}, {"rgb": static}, {"rgb": dynamic}, {"rgb": fair}
    ).log_likelihood_ratio
    # The Student-t's heavy tails mean the log-likelihood grows only
    # logarithmically in residual/scale, so a ~60x scale error inflates the
    # ratio by ~4x rather than ~60x. That bounds the damage but does not
    # remove it: a 4x-inflated ratio still decides comparisons it should not.
    check(
        abs(tight_ratio) > 3.0 * abs(fair_ratio),
        f"an over-tight scale still inflates the likelihood ratio several-fold "
        f"({tight_ratio:+.3g} vs {fair_ratio:+.3g}, {abs(tight_ratio) / abs(fair_ratio):.1f}x)",
    )
    check(fair_ratio > 0, "and under the predictive scale the correct hypothesis still wins")


def test_contrastive_comparison_scores_both_hypotheses_on_the_same_pixels() -> None:
    generator = torch.Generator().manual_seed(2)
    observed = torch.rand((3, 32, 32), generator=generator)
    static = observed.clone()
    dynamic = observed.clone()
    dynamic[:, 8:16, 8:16] += 0.4                 # only this region differs

    footprint = (static - dynamic).abs().amax(dim=0) > 1.0e-3
    model = ResidualModel(0.02, 4.0, "student_t", 1)
    on_footprint_static = compute_hypothesis_log_likelihood(observed, static, model, region_mask=footprint)
    on_footprint_dynamic = compute_hypothesis_log_likelihood(observed, dynamic, model, region_mask=footprint)
    whole_static = compute_hypothesis_log_likelihood(observed, static, model)
    whole_dynamic = compute_hypothesis_log_likelihood(observed, dynamic, model)

    check(
        abs(on_footprint_static - on_footprint_dynamic) > 1.0,
        "scored on the disagreement footprint, the two hypotheses are clearly separated",
    )
    check(
        abs((whole_static - whole_dynamic) - (on_footprint_static - on_footprint_dynamic)) < 1.0e-3,
        "the difference comes entirely from the footprint, which is why the rest is excluded",
    )
    check(
        int(footprint.sum()) < 0.2 * footprint.numel(),
        f"and the footprint is a small fraction of the image ({int(footprint.sum())}/{footprint.numel()}) -- "
        f"scoring the whole frame would bury the motion in identical background",
    )


def test_moving_a_patch_turns_its_gaussians() -> None:
    from utils.rigid_transform_utils import quaternion_to_matrix

    generator = torch.Generator().manual_seed(3)
    xyz = torch.rand((20, 3), generator=generator)
    quaternion = torch.zeros((20, 4))
    quaternion[:, 0] = 1.0
    mask = torch.zeros(20, dtype=torch.bool)
    mask[:10] = True
    angle = math.pi / 3
    rotation = torch.tensor([
        [math.cos(angle), -math.sin(angle), 0.0], [math.sin(angle), math.cos(angle), 0.0], [0.0, 0.0, 1.0],
    ])
    moved_xyz, moved_q = transform_patch_gaussians(xyz, quaternion, mask, rotation, torch.tensor([1.0, 0.0, 0.0]))
    check(
        float((quaternion_to_matrix(moved_q[mask]) - rotation[None]).abs().max()) < 1.0e-4,
        "the hypothesis rotation reaches the Gaussians' quaternions, so Sigma' = R Sigma R^T",
    )
    check(
        float((moved_xyz[~mask] - xyz[~mask]).abs().max()) < 1.0e-6,
        "and nothing outside the patch is disturbed",
    )


def test_indistinguishable_ratio_retains_ambiguity() -> None:
    observed = torch.rand((3, 16, 16), generator=torch.Generator().manual_seed(4))
    model = ResidualModel(0.05, 4.0, "student_t", 1)
    comparison = compare_images_only(
        {"rgb": observed}, {"rgb": observed + 0.0001}, {"rgb": observed - 0.0001}, {"rgb": model},
        indistinguishable_nats=2.0,
    )
    check(comparison.indistinguishable, "two effectively identical renders are reported as indistinguishable")
    check(not comparison.prefers_dynamic, "and no winner is forced (§33)")


def main() -> None:
    tests = [v for n, v in sorted(globals().items()) if n.startswith("test_") and callable(v)]
    for test in tests:
        test()
        print(f"  ok  {test.__name__}")
    print(f"\n{len(tests)} tests passed, {len(CHECKS)} assertions:")
    for message in CHECKS:
        print(f"  - {message}")


if __name__ == "__main__":
    main()
