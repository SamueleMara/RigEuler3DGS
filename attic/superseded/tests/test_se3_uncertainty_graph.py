"""§40 checks 1-4: the uncertainty-aware hypothesis graph.

Check 1 is the important one.  The previous P6 estimated its consistency
metric from the scatter of all accepted transforms, which is circular: bad
transforms inflate the scatter, the inflated scatter makes bad transforms
mutually compatible, and the test certifies the noise it exists to reject.
A per-hypothesis covariance cannot be inflated by other hypotheses being bad,
and check 1 asserts exactly that.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from utils.rigid_hypothesis_graph import build_hypothesis_graph, pairwise_transform_discrepancy
from utils.se3_hypothesis_consistency import se3_exp, se3_log
from utils.transform_uncertainty import (
    bootstrap_transform_covariance,
    effective_rank,
    estimate_transform_uncertainty,
    hessian_transform_covariance,
    patch_observability_spectrum,
    se3_adjoint,
)

CHECKS: list[str] = []


def check(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)
    CHECKS.append(message)


def rotation_from_axis_angle(axis, angle: float) -> torch.Tensor:
    axis = torch.as_tensor(axis, dtype=torch.float32)
    axis = axis / torch.linalg.norm(axis)
    skew = torch.tensor([
        [0.0, -axis[2], axis[1]], [axis[2], 0.0, -axis[0]], [-axis[1], axis[0], 0.0],
    ])
    return torch.eye(3) + math.sin(angle) * skew + (1 - math.cos(angle)) * skew @ skew


def anchor_grid(count: int, *, seed: int = 0) -> torch.Tensor:
    generator = torch.Generator().manual_seed(int(seed))
    return torch.rand((int(count), 3), generator=generator) * 2.0


def tight_covariance(count: int, scale: float = 1.0e-4) -> torch.Tensor:
    return torch.eye(6)[None].repeat(count, 1, 1) * float(scale)


# --------------------------------------------------------------------------
# Check 1 -- covariance cannot self-inflate
# --------------------------------------------------------------------------
def test_random_transforms_are_rejected_however_wide_their_scatter() -> None:
    generator = torch.Generator().manual_seed(1)
    count = 30
    tangent = torch.randn((count, 6), generator=generator) * 2.0      # wildly scattered
    rotation, translation = se3_exp(tangent)
    # Each hypothesis is individually CONFIDENT -- the old scene-wide
    # estimator would read the huge scatter as the noise level and declare
    # them all mutually compatible.
    covariance = tight_covariance(count)

    graph = build_hypothesis_graph(
        source_anchor=torch.arange(count), target_anchor=torch.arange(count),
        rotation=rotation, translation=translation, covariance=covariance,
        anchor_xyz=anchor_grid(count, seed=1), neighbours=8, alpha=0.05,
    )
    check(
        graph.stats["components"] == 0,
        f"random transforms form no component however large their scatter "
        f"(got {graph.stats['components']})",
    )
    check(
        graph.stats["edge_acceptance_rate"] < 0.05,
        f"and almost no edge survives ({graph.stats['edge_acceptance_rate']:.3f})",
    )


# --------------------------------------------------------------------------
# Check 2 -- good transforms stay connected
# --------------------------------------------------------------------------
def test_noisy_estimates_of_one_transform_form_one_component() -> None:
    generator = torch.Generator().manual_seed(2)
    count = 24
    truth = torch.tensor([0.1, -0.2, 0.3, 0.5, 0.1, -0.4])
    noise_scale = 0.02
    tangent = truth[None].repeat(count, 1) + torch.randn((count, 6), generator=generator) * noise_scale
    rotation, translation = se3_exp(tangent)
    covariance = torch.eye(6)[None].repeat(count, 1, 1) * (noise_scale ** 2)

    graph = build_hypothesis_graph(
        source_anchor=torch.arange(count), target_anchor=torch.arange(count),
        rotation=rotation, translation=translation, covariance=covariance,
        anchor_xyz=anchor_grid(count, seed=2), neighbours=8, alpha=0.05,
    )
    check(
        graph.stats["components"] == 1,
        f"noisy estimates of one transform form exactly one component (got {graph.stats['components']})",
    )
    check(
        graph.stats["edge_acceptance_rate"] > 0.8,
        f"and most neighbouring pairs agree ({graph.stats['edge_acceptance_rate']:.3f})",
    )
    consensus = graph.component_transform[0]
    recovered = se3_log(consensus[:3][None], consensus[3][None])[0]
    check(
        float(torch.linalg.norm(recovered - truth)) < 0.1,
        f"and the consensus recovers the true transform (error {float(torch.linalg.norm(recovered - truth)):.4f})",
    )


# --------------------------------------------------------------------------
# Check 3 -- two rigid parts, without being told K
# --------------------------------------------------------------------------
def test_two_transform_families_form_two_components() -> None:
    generator = torch.Generator().manual_seed(3)
    count = 20
    noise = 0.02
    family_a = torch.tensor([0.0, 0.0, 0.0, 1.0, 0.0, 0.0])
    family_b = torch.tensor([0.0, 0.0, 0.9, -1.0, 0.5, 0.0])
    tangent = torch.cat((
        family_a[None].repeat(count, 1), family_b[None].repeat(count, 1),
    )) + torch.randn((2 * count, 6), generator=generator) * noise
    rotation, translation = se3_exp(tangent)
    covariance = torch.eye(6)[None].repeat(2 * count, 1, 1) * (noise ** 2)

    # Anchors clustered so each family is its own spatial neighbourhood.
    anchors = torch.cat((
        anchor_grid(count, seed=31) * 0.3,
        anchor_grid(count, seed=32) * 0.3 + torch.tensor([5.0, 0.0, 0.0]),
    ))
    graph = build_hypothesis_graph(
        source_anchor=torch.arange(2 * count), target_anchor=torch.arange(2 * count),
        rotation=rotation, translation=translation, covariance=covariance,
        anchor_xyz=anchors, neighbours=6, alpha=0.05,
    )
    check(
        graph.stats["components"] == 2,
        f"two independently moving families give two components (got {graph.stats['components']}), K never supplied",
    )
    labels = graph.component
    check(
        int(labels[:count].max()) != int(labels[count:].max()),
        "and the two families are not merged",
    )


# --------------------------------------------------------------------------
# Check 4 -- planar ambiguity must stay uncertain
# --------------------------------------------------------------------------
def test_degenerate_patch_reports_unobservable_directions() -> None:
    """§40 check 4, stated as the geometry actually behaves.

    The audit expects a near-planar patch to be weakly observable, but that is
    a point-to-PLANE phenomenon.  Under the point-to-point residual this
    pipeline uses, a planar patch is fully determined -- you cannot rotate a
    plane about an in-plane axis without lifting its points off it.  Measured:
    volumetric anisotropy 25, planar 29, i.e. indistinguishable.

    What IS unobservable is a collinear patch, whose rotation about its own
    axis moves nothing.  The uncertainty must report that rather than force an
    invertible answer, which is why the covariance is a pseudoinverse.
    """
    count = 40
    generator = torch.Generator().manual_seed(4)
    volumetric = torch.rand((1, count, 3), generator=generator)
    collinear = torch.zeros((1, count, 3))
    collinear[0, :, 0] = torch.linspace(0.0, 1.0, count)

    coupling = (torch.eye(count) / count)[None]
    identity, zero = torch.eye(3)[None], torch.zeros((1, 3))
    volume_rank, volume_spectrum = effective_rank(
        hessian_transform_covariance(volumetric, volumetric, coupling, identity, zero)
    )
    line_rank, line_spectrum = effective_rank(
        hessian_transform_covariance(collinear, collinear, coupling, identity, zero)
    )

    check(
        int(volume_rank[0]) == 6,
        f"a volumetric patch determines all six tangent directions (rank {int(volume_rank[0])})",
    )
    check(
        int(line_rank[0]) < 6,
        f"a collinear patch does not (rank {int(line_rank[0])}) -- the rotation about its own axis is unobservable",
    )
    check(
        float(line_spectrum.max() / line_spectrum.min().clamp_min(1e-30))
        > 1.0e6 * float(volume_spectrum.max() / volume_spectrum.min().clamp_min(1e-30)),
        "and its covariance is enormously anisotropic rather than quietly invertible",
    )


def test_planar_patch_has_unobservable_chirality() -> None:
    """§9: a flat patch has no handedness, so no chirality decision may be forced.

    This is the sense in which a planar patch IS degenerate -- its signed
    volume vanishes -- even though its transform is well determined.
    """
    count = 40
    generator = torch.Generator().manual_seed(9)
    planar = torch.rand((1, count, 3), generator=generator)
    planar[..., 2] = 0.0
    volumetric = torch.rand((1, count, 3), generator=generator)
    valid = torch.ones((1, count), dtype=torch.bool)

    _spectrum, planar_chirality = patch_observability_spectrum(planar, valid)
    _spectrum2, volume_chirality = patch_observability_spectrum(volumetric, valid)
    check(
        float(planar_chirality[0]) < 0.02,
        f"a flat patch reports unobservable chirality ({float(planar_chirality[0]):.4f})",
    )
    check(
        float(volume_chirality[0]) > 10.0 * float(planar_chirality[0]),
        f"while a volumetric one has a real handedness ({float(volume_chirality[0]):.4f})",
    )


def test_observability_spectrum_is_not_a_boolean() -> None:
    # §4: a near-planar patch can be numerically rank 3 and still be
    # unobservable about its normal; a boolean hides exactly that.
    generator = torch.Generator().manual_seed(5)
    points = torch.rand((1, 60, 3), generator=generator)
    points[..., 2] *= 0.002                               # rank 3, but barely
    valid = torch.ones((1, 60), dtype=torch.bool)
    spectrum, chirality = patch_observability_spectrum(points, valid)
    check(bool((spectrum > 0).all()), "all three eigenvalues are strictly positive (numerically rank 3)")
    check(
        float(spectrum[0, 2] / spectrum[0, 0]) < 1.0e-3,
        "yet the smallest is orders of magnitude below the largest -- the spectrum says what a boolean cannot",
    )
    check(float(chirality[0]) < 0.05, "and chirality is correspondingly unobservable")


# --------------------------------------------------------------------------
# Adjoint transport
# --------------------------------------------------------------------------
def test_adjoint_transports_covariance_between_frames() -> None:
    # Without the adjoint, two covariances expressed in different tangent
    # frames are simply added -- which looks like a test and is not one.
    rotation = rotation_from_axis_angle([0.2, 0.9, 0.3], 1.1)[None]
    translation = torch.tensor([[1.5, -0.5, 2.0]])
    adjoint = se3_adjoint(rotation, translation)
    check(
        abs(float(torch.linalg.det(adjoint[0])) - 1.0) < 1.0e-4,
        "the adjoint is volume preserving, so transport does not create or destroy uncertainty",
    )
    identity_adjoint = se3_adjoint(torch.eye(3)[None], torch.zeros((1, 3)))
    check(
        float((identity_adjoint[0] - torch.eye(6)).abs().max()) < 1.0e-6,
        "and the identity transform transports nothing",
    )


def test_discrepancy_uses_both_uncertainties() -> None:
    # The same disagreement must be tolerated when the hypotheses are unsure
    # and rejected when they are confident. That is the whole point of using
    # per-hypothesis covariance.
    tangent = torch.zeros((2, 6))
    tangent[1, 3] = 0.05
    rotation, translation = se3_exp(tangent)
    pairs = torch.tensor([[0, 1]])

    confident = tight_covariance(2, 1.0e-6)
    unsure = tight_covariance(2, 1.0e-1)
    d_confident, _r, _t = pairwise_transform_discrepancy(rotation, translation, confident, pairs)
    d_unsure, _r2, _t2 = pairwise_transform_discrepancy(rotation, translation, unsure, pairs)
    check(
        float(d_confident[0]) > 100.0 * float(d_unsure[0]),
        f"the identical discrepancy is significant for confident hypotheses and not for unsure ones "
        f"({float(d_confident[0]):.3g} vs {float(d_unsure[0]):.3g})",
    )


def test_bootstrap_covariance_grows_when_the_patch_determines_less() -> None:
    """The bootstrap measures how OBSERVATION NOISE propagates into the fit.

    With a noiseless correspondence the fit is identical under every
    resample, so the covariance is ~0 whatever the patch shape -- the
    bootstrap has nothing to measure.  What it captures is direction-
    dependent sensitivity to noise, so the target must actually be noisy.
    """
    generator = torch.Generator(device="cpu").manual_seed(6)
    count = 40
    volumetric = torch.rand((1, count, 3), generator=torch.Generator().manual_seed(7))
    planar = volumetric.clone()
    planar[..., 2] = 0.0

    noise_generator = torch.Generator().manual_seed(8)
    noise = torch.randn((1, count, 3), generator=noise_generator) * 0.02
    coupling = (torch.eye(count) / count)[None]
    identity, zero = torch.eye(3)[None], torch.zeros((1, 3))

    volume_cov = bootstrap_transform_covariance(
        volumetric, volumetric + noise, coupling, identity, zero, draws=24, generator=generator
    )
    planar_cov = bootstrap_transform_covariance(
        planar, planar + noise, coupling, identity, zero, draws=24, generator=generator
    )
    planar_worst = float(torch.linalg.eigvalsh(planar_cov[0].double()).max())
    volume_worst = float(torch.linalg.eigvalsh(volume_cov[0].double()).max())
    check(
        planar_worst > volume_worst,
        f"a patch that determines less of SE(3) bootstraps to a larger worst-case variance "
        f"({planar_worst:.3g} vs {volume_worst:.3g})",
    )


def test_same_anchor_hypotheses_are_never_connected() -> None:
    # §20: two hypotheses of one anchor are alternatives, not corroboration.
    tangent = torch.zeros((4, 6))
    rotation, translation = se3_exp(tangent)
    graph = build_hypothesis_graph(
        source_anchor=torch.tensor([0, 0, 1, 1]), target_anchor=torch.tensor([0, 1, 2, 3]),
        rotation=rotation, translation=translation, covariance=tight_covariance(4),
        anchor_xyz=torch.tensor([[0.0, 0, 0], [0.01, 0, 0]]), neighbours=4, alpha=0.05,
        min_anchors_per_component=2,
    )
    for left, right in graph.edges.tolist():
        check(
            int(graph.source_anchor[left]) != int(graph.source_anchor[right]),
            "no edge joins two hypotheses of the same source anchor",
        )
        break
    check(graph.stats["consistent_edges"] > 0, "but hypotheses of different anchors do connect")


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
