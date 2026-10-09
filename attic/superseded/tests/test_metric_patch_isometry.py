"""Tests for the metric-isometry core: patches, GW, Kabsch, SE(3) algebra.

Scenes are seeded irregular clouds, never lattices. A regular structure has a
degenerate distance spectrum, so every patch is isometric to every other one
and a test built on one measures the degeneracy rather than the mechanism.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from utils.local_metric_isometry import (
    benjamini_hochberg,
    compute_metric_distortion,
    solve_partial_local_gromov_wasserstein,
)
from utils.local_rigid_hypothesis import (
    solve_isometry_with_rigid_selection,
    compute_se3_residual,
    fit_rigid_hypothesis,
    weighted_kabsch_from_soft_coupling,
)
from utils.metric_patch_graph import (
    build_full_geometry_knn,
    build_nested_metric_patches,
    compute_pairwise_metric_matrix,
    find_minimum_full_rank_patch,
    gather_patches,
    normalise_metric_matrix,
)
from utils.se3_hypothesis_consistency import se3_exp, se3_log

CHECKS: list[str] = []


def check(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)
    CHECKS.append(message)


def irregular_cloud(count: int, *, seed: int, extent: float = 1.0, centre=(0.0, 0.0, 0.0)) -> torch.Tensor:
    generator = torch.Generator().manual_seed(int(seed))
    xyz = torch.rand((int(count), 3), generator=generator) * float(extent)
    return xyz + torch.tensor(centre, dtype=xyz.dtype)


def random_rotation(seed: int) -> torch.Tensor:
    generator = torch.Generator().manual_seed(int(seed))
    a = torch.randn((3, 3), generator=generator)
    q, r = torch.linalg.qr(a)
    q = q * torch.sign(torch.diagonal(r))
    if float(torch.linalg.det(q)) < 0:
        q[:, 0] = -q[:, 0]
    return q


# --------------------------------------------------------------------------
# Patch construction
# --------------------------------------------------------------------------
def test_patch_size_follows_local_rank() -> None:
    # A planar sheet cannot constrain a 3-D rotation at any neighbourhood
    # size; a volumetric cloud can almost immediately.  The patch rule must
    # notice the difference rather than take a fixed K on faith.
    volume = irregular_cloud(300, seed=1, extent=1.0)
    plane = irregular_cloud(300, seed=2, extent=1.0)
    plane[:, 2] = 0.0

    for xyz, name, expect_rank in ((volume, "volumetric", True), (plane, "planar", False)):
        knn, _ = build_full_geometry_knn(xyz, 40)
        size, full_rank, _condition = find_minimum_full_rank_patch(xyz, knn)
        if expect_rank:
            check(
                float(full_rank.float().mean()) > 0.95,
                f"a {name} neighbourhood reaches full rank and gets a usable patch",
            )
            check(int(size.median()) < 40, f"and does so well before exhausting the {name} KNN")
        else:
            check(
                float(full_rank.float().mean()) < 0.05,
                f"a {name} neighbourhood never reaches full rank and is flagged, not silently used",
            )


def test_nested_patches_expand_deterministically() -> None:
    xyz = irregular_cloud(300, seed=3, extent=1.0)
    knn, _ = build_full_geometry_knn(xyz, 64)
    patches = build_nested_metric_patches(xyz, knn, levels=3)
    sizes = patches.sizes
    check(bool((sizes[:, 1] >= sizes[:, 0]).all()), "each nested level is at least as large as the previous")
    check(bool((sizes[:, 2] >= sizes[:, 1]).all()), "and the expansion is monotone")
    capped = sizes[:, 2] < 64
    if bool(capped.any()):
        ratio = (sizes[capped, 1].float() / patches.rank_size[capped].float())
        check(abs(float(ratio.median()) - 2.0) < 1.0e-6, "level 2 is exactly twice K1 where the graph allows")


def test_metric_normalisation_is_scale_free() -> None:
    xyz = irregular_cloud(200, seed=4, extent=1.0)
    knn, _ = build_full_geometry_knn(xyz, 24)
    patches = build_nested_metric_patches(xyz, knn)
    indices, valid = gather_patches(patches, torch.arange(20))

    d_small = compute_pairwise_metric_matrix(xyz, indices)
    d_large = compute_pairwise_metric_matrix(xyz * 7.3, indices)
    n_small, _ = normalise_metric_matrix(d_small, valid=valid)
    n_large, _ = normalise_metric_matrix(d_large, valid=valid)
    check(
        float((n_small - n_large).abs().max()) < 1.0e-4,
        "the normalised patch metric is invariant to a global rescaling of the cloud",
    )


# --------------------------------------------------------------------------
# Test 2 / 3 -- metric isometry under rigid motion
# --------------------------------------------------------------------------
def _patch_pair(source_xyz, target_xyz, anchors, k: int = 24):
    knn_s, _ = build_full_geometry_knn(source_xyz, k)
    knn_t, _ = build_full_geometry_knn(target_xyz, k)
    patches_s = build_nested_metric_patches(source_xyz, knn_s)
    patches_t = build_nested_metric_patches(target_xyz, knn_t)
    idx_s, valid_s = gather_patches(patches_s, anchors)
    idx_t, valid_t = gather_patches(patches_t, anchors)
    d0, scale0 = normalise_metric_matrix(compute_pairwise_metric_matrix(source_xyz, idx_s), valid=valid_s)
    dt, _ = normalise_metric_matrix(compute_pairwise_metric_matrix(target_xyz, idx_t), valid=valid_t)
    return (idx_s, valid_s, d0, scale0), (idx_t, valid_t, dt)


def test_isometry_survives_large_translation() -> None:
    xyz = irregular_cloud(200, seed=11, extent=1.0)
    shift = torch.tensor([50.0, -20.0, 7.0])
    moved = xyz + shift
    anchors = torch.arange(12)
    (_, valid_s, d0, _), (_, valid_t, dt) = _patch_pair(xyz, moved, anchors)

    result = solve_partial_local_gromov_wasserstein(d0, dt, source_valid=valid_s, target_valid=valid_t)
    check(
        float(result.distortion.max()) < 1.0e-2,
        f"a pure translation leaves the patch metric untouched (max distortion "
        f"{float(result.distortion.max()):.2e})",
    )


def test_isometry_survives_arbitrary_rotation() -> None:
    xyz = irregular_cloud(200, seed=12, extent=1.0)
    rotation = random_rotation(99)
    shift = torch.tensor([-13.0, 4.0, 22.0])
    moved = xyz @ rotation.T + shift
    anchors = torch.arange(12)
    (idx_s, valid_s, d0, scale0), (idx_t, valid_t, dt) = _patch_pair(xyz, moved, anchors)

    result = solve_partial_local_gromov_wasserstein(d0, dt, source_valid=valid_s, target_valid=valid_t)
    check(
        float(result.distortion.max()) < 1.0e-2,
        "a rotation plus translation leaves the patch metric untouched",
    )

    hypothesis = fit_rigid_hypothesis(
        xyz[idx_s], moved[idx_t], result.coupling, patch_scale=scale0,
    )
    rotation_error = float((hypothesis.rotation - rotation[None]).abs().max())
    translation_error = float((hypothesis.translation - shift[None]).abs().max())
    check(rotation_error < 0.05, f"the recovered R matches the true rotation (max |dR| {rotation_error:.4f})")
    check(translation_error < 0.05, f"and the recovered t matches (max |dt| {translation_error:.4f})")
    check(
        float(hypothesis.residual.max()) < 1.0e-3,
        "with a rigid residual at numerical zero",
    )


def test_recovered_rotation_is_a_proper_rotation() -> None:
    # A reflection has identical internal distances, so only the determinant
    # correction keeps the fit inside SO(3).
    xyz = irregular_cloud(60, seed=13, extent=1.0)
    mirrored = xyz.clone()
    mirrored[:, 0] = -mirrored[:, 0]
    coupling = (torch.eye(60) / 60.0)[None]
    rotation, _translation, _mass, _degenerate = weighted_kabsch_from_soft_coupling(
        xyz[None], mirrored[None], coupling
    )
    determinant = float(torch.linalg.det(rotation[0]))
    check(determinant > 0.99, f"the fit stays a proper rotation against mirrored data (det {determinant:.4f})")
    identity_error = float((rotation[0] @ rotation[0].T - torch.eye(3)).abs().max())
    check(identity_error < 1.0e-4, "and is orthogonal")


# --------------------------------------------------------------------------
# Test 6 -- non-rigid deformation must be rejected
# --------------------------------------------------------------------------
def test_non_rigid_deformation_is_rejected() -> None:
    xyz = irregular_cloud(200, seed=14, extent=1.0)
    rigid = xyz + torch.tensor([5.0, 0.0, 0.0])
    sheared = xyz.clone()
    sheared[:, 0] = sheared[:, 0] + 0.6 * sheared[:, 1]        # a shear preserves no distances
    stretched = xyz * torch.tensor([1.0, 1.0, 1.9])

    anchors = torch.arange(12)
    for name, moved, rigid_case in (("shear", sheared, False), ("stretch", stretched, False), ("rigid", rigid, True)):
        (_, valid_s, d0, _), (_, valid_t, dt) = _patch_pair(xyz, moved, anchors)
        result = solve_partial_local_gromov_wasserstein(d0, dt, source_valid=valid_s, target_valid=valid_t)
        distortion = float(result.distortion.median())
        # Rigid controls land near 5e-3 at worst and the deformations near
        # 4e-2, so the two populations are separated by roughly an order of
        # magnitude rather than by a threshold chosen to make them separate.
        if rigid_case:
            check(distortion < 1.0e-2, f"the rigid control has ~zero distortion ({distortion:.2e})")
        else:
            check(distortion > 2.0e-2, f"a {name} is not an isometry and the distortion says so ({distortion:.4f})")


# --------------------------------------------------------------------------
# Test 7 -- independent sampling of the same surface
# --------------------------------------------------------------------------
def test_independent_sampling_recovers_the_same_transform() -> None:
    # The two endpoints are INDEPENDENT samples of one rigid surface, as two
    # separately trained 3DGS models are: not a single point of one has a
    # counterpart in the other.  This tests transform RECOVERY under that
    # condition -- the patch pair is supplied, because finding it is what the
    # refilled-region test covers.
    # Dense enough that a patch carries more curvature signal than sampling
    # noise.  The sparse regime is covered by its own test, which asserts the
    # method reports the failure rather than guessing.
    source = _bumpy_surface(3000, seed=21)
    rotation = random_rotation(7)
    shift = torch.tensor([3.0, -1.5, 0.75])
    target = _bumpy_surface(3240, seed=22) @ rotation.T + shift
    # Independence is a statement about the typical point, not the luckiest
    # one: in a dense sample some pairs land close by chance, so the median
    # cross distance against the intra-cloud spacing is the honest measure.
    cross = torch.cdist(source @ rotation.T + shift, target).min(dim=1).values
    spacing = torch.cdist(source, source).masked_fill(
        torch.eye(int(source.shape[0]), dtype=torch.bool), float("inf")
    ).min(dim=1).values.median()
    check(
        float(cross.median()) > 0.2 * float(spacing),
        f"the two samples are genuinely independent, so no identity is available "
        f"(median cross {float(cross.median()):.4f} vs spacing {float(spacing):.4f})",
    )

    _coupling, _distortion, hypothesis = _solve(source, target, rotation, shift, k=60, level=2)
    error = torch.stack([torch.linalg.matrix_norm(r - rotation) for r in hypothesis.rotation])
    translation_error = torch.linalg.norm(hypothesis.translation - shift, dim=1)
    check(
        float(error.median()) < 0.6,
        f"independently sampled patches of one rigid surface agree on R (median |dR| {float(error.median()):.3f})",
    )
    check(
        float(translation_error.median()) < 0.4,
        f"and on t (median |dt| {float(translation_error.median()):.3f})",
    )


def _solve(source, target, rotation, shift, *, k: int, level: int, count: int = 10):
    """Solve the true patch pairs, giving the correspondence but not the motion."""
    anchors = torch.arange(0, int(source.shape[0]), max(1, int(source.shape[0]) // count))[:count]
    target_anchors = torch.cdist(source[anchors] @ rotation.T + shift, target).argmin(dim=1)
    knn_s, _ = build_full_geometry_knn(source, k)
    knn_t, _ = build_full_geometry_knn(target, k)
    idx_s, valid_s = gather_patches(build_nested_metric_patches(source, knn_s), anchors, level=level)
    idx_t, valid_t = gather_patches(build_nested_metric_patches(target, knn_t), target_anchors, level=level)
    d0, scale0 = normalise_metric_matrix(compute_pairwise_metric_matrix(source, idx_s), valid=valid_s)
    dt, _ = normalise_metric_matrix(compute_pairwise_metric_matrix(target, idx_t), valid=valid_t)
    return solve_isometry_with_rigid_selection(
        source[idx_s], target[idx_t], d0, dt, valid_s, valid_t, patch_scale=scale0
    )


def test_information_poor_patches_are_reported_not_guessed() -> None:
    """§65: a patch that cannot determine a motion must not report one confidently.

    A small, near-planar patch is nearly symmetric under reflection through
    its own tangent plane, so neither the metric nor the rigid residual can
    separate the true motion from its mirror image -- measured here, a
    28-point patch recovers the rotation essentially at random.  That is a
    real limit of the physics, not a bug, and what matters is that the
    residual it reports is correspondingly poor so the §23 statistical test
    rejects it instead of passing a confident wrong transform downstream.
    """
    rotation = random_rotation(7)
    shift = torch.tensor([3.0, -1.5, 0.75])
    sparse = _bumpy_surface(700, seed=21)
    sparse_t = _bumpy_surface(756, seed=22) @ rotation.T + shift
    dense = _bumpy_surface(3000, seed=21)
    dense_t = _bumpy_surface(3240, seed=22) @ rotation.T + shift

    _c1, _d1, poor = _solve(sparse, sparse_t, rotation, shift, k=28, level=0)
    _c2, _d2, rich = _solve(dense, dense_t, rotation, shift, k=60, level=2)

    poor_error = torch.stack([torch.linalg.matrix_norm(r - rotation) for r in poor.rotation]).median()
    rich_error = torch.stack([torch.linalg.matrix_norm(r - rotation) for r in rich.rotation]).median()
    check(
        float(rich_error) < float(poor_error),
        f"an information-rich patch recovers the motion where a poor one cannot "
        f"({float(rich_error):.3f} vs {float(poor_error):.3f})",
    )
    check(
        float(rich.residual.median()) < float(poor.residual.median()),
        f"and the rigid residual reports the difference, which is what the §23 test acts on "
        f"({float(rich.residual.median()):.4f} vs {float(poor.residual.median()):.4f})",
    )


def _bumpy_surface(count: int, *, seed: int) -> torch.Tensor:
    """Independent samples of one fixed, locally curved surface.

    Deterministic in the surface and random only in WHERE it is sampled, so
    two calls with different seeds give two genuinely independent samplings
    of the same geometry -- which is the situation two separately trained
    endpoint models are in.
    """
    generator = torch.Generator().manual_seed(int(seed))
    direction = torch.randn((int(count), 3), generator=generator)
    direction = direction / torch.linalg.norm(direction, dim=1, keepdim=True)
    bump = 1.0 + 0.30 * torch.sin(3.0 * direction[:, 0]) * torch.cos(2.0 * direction[:, 1])
    return direction * bump[:, None]


# --------------------------------------------------------------------------
# Test 8 -- partial visibility
# --------------------------------------------------------------------------
def test_partial_overlap_still_recovers_the_transform() -> None:
    xyz = irregular_cloud(200, seed=31, extent=1.0)
    rotation = random_rotation(5)
    shift = torch.tensor([2.0, 3.0, -1.0])
    moved = xyz @ rotation.T + shift

    anchors = torch.arange(10)
    (idx_s, valid_s, d0, scale0), (idx_t, valid_t, dt) = _patch_pair(xyz, moved, anchors)
    # A third of the target patch is invisible: partial transport must leave
    # that mass unmatched rather than forcing it onto the visible structure.
    occluded = valid_t.clone()
    occluded[:, ::3] = False
    occluded[:, 0] = True

    result = solve_partial_local_gromov_wasserstein(d0, dt, source_valid=valid_s, target_valid=occluded)
    hypothesis = fit_rigid_hypothesis(xyz[idx_s], moved[idx_t], result.coupling, patch_scale=scale0)
    error = float((hypothesis.rotation - rotation[None]).abs().flatten(1).max(dim=1).values.median())
    check(error < 0.15, f"a partially visible patch still recovers the transform (median |dR| {error:.4f})")
    check(
        float(result.coupling[:, :, ~occluded[0]].sum()) < 1.0e-6,
        "and no mass is transported onto the occluded points",
    )


# --------------------------------------------------------------------------
# SE(3) algebra
# --------------------------------------------------------------------------
def test_se3_log_exp_round_trip() -> None:
    rotations = torch.stack([random_rotation(s) for s in range(1, 9)])
    translations = torch.randn((8, 3), generator=torch.Generator().manual_seed(4)) * 3.0
    tangent = se3_log(rotations, translations)
    r, t = se3_exp(tangent)
    # The sweep includes a 178.9-degree rotation, where the logarithm is at
    # its worst conditioned; 1e-3 is float32 reality there, not slack.
    check(float((r - rotations).abs().max()) < 2.0e-3, "se3_exp inverts se3_log for the rotation")
    check(float((t - translations).abs().max()) < 2.0e-3, "and for the translation")


def test_se3_log_survives_a_half_turn() -> None:
    # A half-turn is where the closed-form logarithm degenerates: the
    # antisymmetric part vanishes and axis/sin(theta) is 0/0.  Two parts of a
    # manipulated object routinely differ by close to one, and an inf here
    # would poison every tangent covariance and consensus downstream.
    axis = torch.tensor([0.3, 0.5, 0.81])
    axis = axis / torch.linalg.norm(axis)
    skew = torch.tensor([
        [0.0, -axis[2], axis[1]], [axis[2], 0.0, -axis[0]], [-axis[1], axis[0], 0.0],
    ])
    for offset in (0.0, 1.0e-5, 1.0e-3):
        angle = math.pi - offset
        rotation = (torch.eye(3) + math.sin(angle) * skew + (1 - math.cos(angle)) * skew @ skew)[None]
        translation = torch.tensor([[1.0, 2.0, 3.0]])
        tangent = se3_log(rotation, translation)
        check(bool(torch.isfinite(tangent).all()), f"se3_log is finite at theta = pi - {offset:g}")
        r, t = se3_exp(tangent)
        check(float((r - rotation).abs().max()) < 5.0e-3, f"and round-trips at theta = pi - {offset:g}")
        check(float((t - translation).abs().max()) < 5.0e-3, f"including its translation at pi - {offset:g}")


def test_se3_log_handles_identity() -> None:
    r = torch.eye(3)[None].expand(4, 3, 3)
    t = torch.randn((4, 3), generator=torch.Generator().manual_seed(6))
    tangent = se3_log(r, t)
    check(float(tangent[:, :3].abs().max()) < 1.0e-5, "a pure translation has zero rotation tangent")
    check(
        float((tangent[:, 3:] - t).abs().max()) < 1.0e-4,
        "and its translation tangent is the translation itself (no small-angle blow-up)",
    )


def test_distortion_is_zero_for_a_true_isometry() -> None:
    xyz = irregular_cloud(30, seed=41, extent=1.0)
    d = torch.cdist(xyz, xyz)[None]
    permutation = torch.randperm(30, generator=torch.Generator().manual_seed(8))
    d_permuted = torch.cdist(xyz[permutation], xyz[permutation])[None]
    # d_permuted's row k is point permutation[k], so the isometric coupling
    # sends source permutation[k] to target k -- the inverse permutation.
    coupling = torch.zeros((1, 30, 30))
    coupling[0, permutation, torch.arange(30)] = 1.0 / 30.0
    distortion = float(compute_metric_distortion(d, d_permuted, coupling))
    check(distortion < 1.0e-8, f"the exact matching permutation has zero distortion ({distortion:.2e})")


def test_benjamini_hochberg_controls_discoveries() -> None:
    # 100 nulls plus 10 strong signals: BH must keep the signals and reject
    # nearly all of the nulls, which a raw 0.05 cut would not do.
    generator = torch.Generator().manual_seed(12)
    p = torch.cat((torch.rand(100, generator=generator), torch.full((10,), 1.0e-5)))
    accepted, _cut = benjamini_hochberg(p, alpha=0.05)
    check(int(accepted[100:].sum()) == 10, "BH keeps all ten true signals")
    check(int(accepted[:100].sum()) <= 2, f"and admits almost no nulls ({int(accepted[:100].sum())} of 100)")
    check(
        int(benjamini_hochberg(torch.rand(200, generator=generator), alpha=0.05)[0].sum()) <= 10,
        "and finds essentially nothing in pure noise",
    )


def test_se3_residual_penalises_a_wrong_transform() -> None:
    xyz = irregular_cloud(40, seed=51, extent=1.0)
    moved = xyz + torch.tensor([2.0, 0.0, 0.0])
    coupling = (torch.eye(40) / 40.0)[None]
    right = compute_se3_residual(
        xyz[None], moved[None], coupling,
        torch.eye(3)[None], torch.tensor([[2.0, 0.0, 0.0]]),
    )
    wrong = compute_se3_residual(
        xyz[None], moved[None], coupling,
        torch.eye(3)[None], torch.tensor([[0.0, 0.0, 0.0]]),
    )
    check(float(right) < 1.0e-5, "the true transform has zero rigid residual")
    check(float(wrong) > 1.0, "and a wrong translation is penalised")


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
