"""§50 tests 1, 3, 4, 6: the rasteriser coupling, on synthetic ground truth.

The identity that carries the whole design is

    P_img = (Omega^0)^T Gamma Omega^T

so it is checked against a hand-built ownership and transport where the
answer is known by construction, not inferred from a run.
"""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from utils.lifted_image_transport import (
    candidate_recall_audit,
    fuse_multiview_gaussian_support,
    lift_pixel_transport_to_gaussian_support,
    score_patch_pairs_from_gaussian_support,
)
from utils.pixel_transport import PixelTransport

CHECKS: list[str] = []


def check(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)
    CHECKS.append(message)


def make_transport(source_index, target_index, coupling) -> PixelTransport:
    source_index = torch.as_tensor(source_index).long()
    target_index = torch.as_tensor(target_index).long()
    coupling = torch.as_tensor(coupling).float()
    return PixelTransport(
        camera_id=0, source_pixel_indices=source_index, target_pixel_indices=target_index,
        coupling_values=coupling, source_dynamic_mass=coupling.sum(dim=1),
        target_dynamic_mass=torch.ones_like(coupling),
        unmatched_source_mass=torch.zeros(len(source_index)),
        unmatched_target_mass=torch.zeros(()),
    )


# --------------------------------------------------------------------------
# Test 1 -- the lift identity
# --------------------------------------------------------------------------
def test_lift_matches_the_explicit_matrix_product() -> None:
    """The patch-level lift must equal the dense triple product exactly.

    Computed both ways: the sparse contraction the code uses, and the literal
    (Omega^0)^T Gamma Omega^T with small dense matrices. They must agree, or
    the efficient form is computing something else.
    """
    height, width = 4, 4
    pixels = height * width
    sources, targets = 3, 2

    generator = torch.Generator().manual_seed(1)
    source_footprints = torch.rand((sources, height, width), generator=generator)
    target_footprints = torch.rand((targets, height, width), generator=generator)

    source_index = torch.tensor([0, 5, 9, 13])
    target_index = torch.tensor([[1, 2], [6, 7], [10, 11], [14, 15]])
    coupling = torch.rand((4, 2), generator=generator)
    transport = make_transport(source_index, target_index, coupling)

    lifted = lift_pixel_transport_to_gaussian_support(transport, source_footprints, target_footprints)

    # Dense reference: Gamma as a full [pixels, pixels] matrix.
    gamma = torch.zeros((pixels, pixels))
    for row, p in enumerate(source_index.tolist()):
        for column, q in enumerate(target_index[row].tolist()):
            gamma[p, q] += coupling[row, column]
    reference = source_footprints.reshape(sources, -1) @ gamma @ target_footprints.reshape(targets, -1).T

    check(
        float((lifted - reference).abs().max()) < 1.0e-5,
        f"the sparse lift equals (Omega^0)^T Gamma Omega^T exactly "
        f"(max difference {float((lifted - reference).abs().max()):.2e})",
    )
    check(tuple(lifted.shape) == (sources, targets), "with one score per source-target patch pair")


def test_lift_gives_zero_support_where_transport_does_not_reach() -> None:
    height = width = 4
    source_footprints = torch.zeros((2, height, width))
    source_footprints[0, 0, 0] = 1.0
    source_footprints[1, 3, 3] = 1.0          # this patch owns no transported pixel
    target_footprints = torch.zeros((2, height, width))
    target_footprints[0, 0, 1] = 1.0
    target_footprints[1, 2, 2] = 1.0

    transport = make_transport([0], [[1]], [[1.0]])
    lifted = lift_pixel_transport_to_gaussian_support(transport, source_footprints, target_footprints)
    check(float(lifted[0, 0]) > 0.5, "a transported pixel pair gives its patch pair support")
    check(float(lifted[1].abs().max()) < 1.0e-6, "a patch outside the transported mass gets none")
    check(float(lifted[0, 1]) < 1.0e-6, "and support does not leak to unrelated target patches")


# --------------------------------------------------------------------------
# Test 4 -- image-supported candidate recall
# --------------------------------------------------------------------------
def test_image_support_ranks_the_true_target_far_above_random() -> None:
    """The claim the whole stage exists to make (§52.1).

    Without it, a source patch draws M targets uniformly from thousands; with
    it, the transported pixels point at the right region. The test measures
    recall@k under both, because "the ordering looks sensible" is not a
    measurement.
    """
    sources, targets = 40, 400
    generator = torch.Generator().manual_seed(2)
    truth = torch.randint(0, targets, (sources,), generator=generator)

    support = torch.rand((sources, targets), generator=generator) * 0.01
    support[torch.arange(sources), truth] += 1.0          # image evidence points at the truth
    lifted = fuse_multiview_gaussian_support([support])

    order, report = score_patch_pairs_from_gaussian_support(lifted, candidates=8)
    image_recall = candidate_recall_audit(order, truth)

    random_order = torch.stack([
        torch.randperm(targets, generator=generator)[:8] for _ in range(sources)
    ])
    random_recall = candidate_recall_audit(random_order, truth)

    check(
        image_recall["recall_at_1"] > 0.9,
        f"the true target is ranked first under image support ({image_recall['recall_at_1']:.2f})",
    )
    check(
        image_recall["recall_at_8"] > 20.0 * max(random_recall["recall_at_8"], 0.005),
        f"recall@8 is far above random search ({image_recall['recall_at_8']:.3f} "
        f"vs {random_recall['recall_at_8']:.3f})",
    )
    check(
        report["image_supported_fraction"] > 0.99,
        "and every anchor had real image support rather than falling back to random",
    )


def test_anchors_without_image_support_are_reported_not_hidden() -> None:
    """An 'image-supported' search that was mostly random is a different
    experiment, so the fallback count is part of the report."""
    sources, targets = 10, 50
    support = torch.zeros((sources, targets))
    support[:5, 3] = 1.0                       # only half the anchors see anything
    lifted = fuse_multiview_gaussian_support([support])
    _order, report = score_patch_pairs_from_gaussian_support(lifted, candidates=4)
    check(
        report["anchors_without_image_support"] == 5,
        f"anchors with no image support are counted ({report['anchors_without_image_support']})",
    )
    check(
        abs(report["image_supported_fraction"] - 0.5) < 1.0e-6,
        "and the supported fraction is reported honestly",
    )


# --------------------------------------------------------------------------
# Multi-view fusion
# --------------------------------------------------------------------------
def test_invisible_pairs_contribute_nothing_rather_than_zero_evidence() -> None:
    """§14/§30: a camera that cannot see a pair must not vote against it."""
    seen = torch.zeros((2, 2)); seen[0, 0] = 0.8
    blind = torch.zeros((2, 2))                # this camera sees nothing
    fused = fuse_multiview_gaussian_support([seen, blind])
    check(
        float(fused.support[0, 0]) > 0.5,
        f"a pair seen by one camera keeps its support ({float(fused.support[0, 0]):.3f}) "
        f"despite another camera being blind to it",
    )
    check(int(fused.view_count[0, 0]) == 1, "and the view count records that only one camera contributed")


def test_agreeing_cameras_reinforce() -> None:
    a = torch.zeros((2, 2)); a[0, 0] = 0.5
    b = torch.zeros((2, 2)); b[0, 0] = 0.5
    single = fuse_multiview_gaussian_support([a])
    both = fuse_multiview_gaussian_support([a, b])
    check(
        int(both.view_count[0, 0]) == 2 and int(single.view_count[0, 0]) == 1,
        "multi-view agreement is recorded as more evidence, not a larger arbitrary weight",
    )


# --------------------------------------------------------------------------
# Test 6 -- projected flow from 3D motion
# --------------------------------------------------------------------------
def test_projected_flow_matches_finite_difference_projection() -> None:
    """The analytic Jacobian must agree with actually moving the points.

    Checked by projecting x and x + dt*v with the same camera model and
    comparing the pixel difference to J_pi(X) R v.
    """
    from utils.projected_pixel_flow import project_gaussian_velocity_to_image_flow

    torch.manual_seed(0)
    count = 64
    xyz = torch.rand((count, 3)) * 2.0 - 1.0
    xyz[:, 2] += 4.0                                    # in front of the camera
    velocity = torch.randn((count, 3)) * 0.01

    view = torch.eye(4)
    view[3, :3] = torch.tensor([0.0, 0.0, 0.0])
    camera = SimpleNamespace(
        world_view_transform=view, image_width=640, image_height=480,
        FoVx=1.0, FoVy=0.8,
    )
    analytic = project_gaussian_velocity_to_image_flow(camera, xyz, velocity).double()

    # The reference is computed in float64 with a CENTRED difference. In
    # float32 the difference of two pixel coordinates in the hundreds at a
    # 1e-5 step loses most of its significant digits, and the reference --
    # not the analytic Jacobian -- is then what is wrong: measured that way
    # it disagreed by 19% while the double-precision check below agrees to
    # 1e-7.
    focal_x = 640.0 / (2.0 * float(torch.tan(torch.tensor(0.5, dtype=torch.float64))))
    focal_y = 480.0 / (2.0 * float(torch.tan(torch.tensor(0.4, dtype=torch.float64))))

    def project(points):
        return torch.stack(
            (focal_x * points[:, 0] / points[:, 2], focal_y * points[:, 1] / points[:, 2]), dim=1
        )

    step = 1.0e-4
    xyz64, velocity64 = xyz.double(), velocity.double()
    finite = (project(xyz64 + step * velocity64) - project(xyz64 - step * velocity64)) / (2.0 * step)
    error = torch.linalg.norm(analytic - finite, dim=1)
    magnitude = torch.linalg.norm(finite, dim=1).clamp_min(1.0e-9)
    check(
        float((error / magnitude).max()) < 1.0e-5,
        f"the analytic projected flow matches finite differences "
        f"(max relative error {float((error / magnitude).max()):.2e})",
    )


def test_points_behind_the_camera_get_no_flow() -> None:
    from utils.projected_pixel_flow import project_gaussian_velocity_to_image_flow

    xyz = torch.tensor([[0.0, 0.0, -2.0], [0.0, 0.0, 4.0]])
    velocity = torch.ones((2, 3)) * 0.1
    view = torch.eye(4)
    camera = SimpleNamespace(world_view_transform=view, image_width=64, image_height=64, FoVx=1.0, FoVy=1.0)
    flow = project_gaussian_velocity_to_image_flow(camera, xyz, velocity)
    check(float(flow[0].abs().max()) == 0.0, "a Gaussian behind the camera produces no flow rather than a sign flip")
    check(float(flow[1].abs().max()) > 0.0, "while one in front does")


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
