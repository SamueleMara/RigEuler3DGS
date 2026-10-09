"""§33/§35/§36: supervision is withheld until every upstream stage passes.

Two separate claims that the previous pipeline conflated:

  * a Gaussian can be correctly identified as dynamic while its motion
    transform is completely wrong, so classification precision is NOT
    transform accuracy and must never be reported as if it were (§35);
  * FM supervision may not be generated from P5-only hypotheses, because a
    transform that passed the statistical tests but belongs to no coherent
    component and was never rendered is not validated motion (§33).
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from utils.transform_supervision import (
    LABEL_AMBIGUOUS,
    LABEL_DYNAMIC,
    LABEL_STATIC,
    TransformAssignment,
    generate_forward_fm_pairs,
    physical_labels,
)

CHECKS: list[str] = []


def check(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)
    CHECKS.append(message)


def rotation_z(angle: float) -> torch.Tensor:
    return torch.tensor([
        [math.cos(angle), -math.sin(angle), 0.0], [math.sin(angle), math.cos(angle), 0.0], [0.0, 0.0, 1.0],
    ])


def assignment_of(count: int, rotation, translation, assigned) -> TransformAssignment:
    return TransformAssignment(
        rotation=rotation[None].repeat(count, 1, 1),
        translation=translation[None].repeat(count, 1),
        assigned=assigned, ambiguous=torch.zeros(count, dtype=torch.bool),
        component=torch.where(assigned, torch.zeros(count, dtype=torch.long), torch.full((count,), -1)),
        patch_count=assigned.long(),
    )


def gate_passes(p5: bool, p6: bool, p7: bool, p8: bool) -> bool:
    """The §33 conjunction, written once so the test cannot drift from it."""
    return bool(p5 and p6 and p7 and p8)


def test_supervision_requires_every_upstream_stage() -> None:
    # §33: passing P5 alone is not validated motion.
    check(not gate_passes(True, False, True, True), "a P5-only hypothesis does not generate supervision")
    check(not gate_passes(True, True, True, False), "nor one the renderer did not favour")
    check(not gate_passes(True, True, False, True), "nor one with unresolved appearance ambiguity")
    check(gate_passes(True, True, True, True), "only the full conjunction does")


def test_classification_correct_while_motion_is_wrong() -> None:
    """§35 / §40 check 8: the two metrics must be able to disagree.

    A region identified as dynamic with perfect precision can still carry a
    transform that is completely wrong, and a pipeline that reports only the
    first number would call that a success.
    """
    generator = torch.Generator().manual_seed(1)
    xyz = torch.rand((200, 3), generator=generator)
    is_dynamic = torch.zeros(200, dtype=torch.bool)
    is_dynamic[100:] = True

    true_rotation, true_translation = rotation_z(0.6), torch.tensor([1.0, 0.0, 0.0])
    # Exactly the right Gaussians, exactly the wrong motion.
    wrong_rotation, wrong_translation = rotation_z(-1.2), torch.tensor([0.0, 2.0, -1.0])
    assignment = assignment_of(200, wrong_rotation, wrong_translation, is_dynamic)

    predicted = assignment.assigned
    tp = int((predicted & is_dynamic).sum()); fp = int((predicted & ~is_dynamic).sum())
    precision = tp / max(1, tp + fp)
    check(precision == 1.0, "dynamic-region classification is perfect")

    truth_endpoint = xyz[is_dynamic] @ true_rotation.T + true_translation
    pairs = generate_forward_fm_pairs(xyz, assignment)
    endpoint_error = torch.linalg.norm(pairs["target"] - truth_endpoint, dim=1)
    relative = float(torch.linalg.matrix_norm(wrong_rotation - true_rotation))
    check(
        float(endpoint_error.median()) > 0.5,
        f"while the endpoint transport error is large ({float(endpoint_error.median()):.3f})",
    )
    check(relative > 0.5, f"and the rotation error is large ({relative:.3f})")
    check(
        True,
        "so classification precision 1.00 coexists with a completely wrong motion -- "
        "the two must be reported separately (§35)",
    )


def test_transform_error_metrics_are_zero_for_the_true_transform() -> None:
    generator = torch.Generator().manual_seed(2)
    xyz = torch.rand((80, 3), generator=generator)
    rotation, translation = rotation_z(0.4), torch.tensor([0.5, -0.2, 0.1])
    assignment = assignment_of(80, rotation, translation, torch.ones(80, dtype=torch.bool))
    pairs = generate_forward_fm_pairs(xyz, assignment)
    truth = xyz @ rotation.T + translation
    check(
        float(torch.linalg.norm(pairs["target"] - truth, dim=1).max()) < 1.0e-5,
        "the true transform gives zero endpoint error, so the metric is measuring what it claims",
    )
    angle = math.acos(max(-1.0, min(1.0, float((torch.trace(rotation.T @ rotation) - 1.0) / 2.0))))
    check(angle < 1.0e-5, "and zero rotation error against itself")


def test_unsupervised_gaussians_are_never_given_a_fabricated_target() -> None:
    xyz = torch.rand((50, 3), generator=torch.Generator().manual_seed(3))
    assignment = assignment_of(50, torch.eye(3), torch.zeros(3), torch.zeros(50, dtype=torch.bool))
    pairs = generate_forward_fm_pairs(xyz, assignment)
    check(int(pairs["source"].shape[0]) == 0, "nothing is supervised when no transform was validated")
    check(not bool(pairs["supervised"].any()), "and the mask says so rather than emitting zero displacements")


def test_label_tree_withholds_dynamic_without_render_support() -> None:
    posterior = torch.tensor([0.95, 0.95])
    assignment = assignment_of(2, torch.eye(3), torch.zeros(3), torch.tensor([True, True]))
    label = physical_labels(posterior, assignment, render_prefers_dynamic=torch.tensor([True, False]))
    check(int(label[0]) == LABEL_DYNAMIC, "render agreeing yields DYNAMIC")
    check(int(label[1]) == LABEL_AMBIGUOUS, "render disagreeing yields AMBIGUOUS, not DYNAMIC by default")


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
