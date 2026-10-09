"""Tests for the hybrid static-XYZ / dynamic-relational hypothesis stage.

Scenes here are built from seeded random points, never from lattices or
regular grids: a regular structure has a degenerate distance spectrum, so
every descriptor is identical and a test built on one measures the
degeneracy rather than the mechanism.

Covers spec tests 1-6, 9, 10 and 12.  Tests 7, 8 and 11 (shared forward /
reverse dynamic coupling, dynamic-FM inverse consistency, renderer
hypothesis selection) are NOT covered here: they exercise the dynamic FM,
diffusion and renderer branches, which this stage deliberately does not
build until the Stage A.5 probe shows the gate is worth wiring.
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from utils.dynamic_prior import _binary_auc
from utils.hybrid_dynamic_feedback import (
    compute_context_distance_change,
    compute_internal_distance_change,
    compute_relational_dynamic_evidence,
    compute_static_context_anchors,
    saturating_score,
)
from utils.hybrid_motion_hypotheses import (
    HybridWeights,
    build_hybrid_evidence,
    combine_hypothesis_evidence,
    compute_dynamic_gate,
    compute_localisation_confidence,
    compute_relational_confidence,
    compute_static_mismatch,
    update_from_immutable_prior,
)
from utils.relational_coupling import build_relational_coupling
from utils.relational_graph import build_relational_graph

CHECKS: list[str] = []


def check(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)
    CHECKS.append(message)


def irregular_cloud(count: int, *, seed: int, extent: float = 1.0, centre=(0.0, 0.0, 0.0)) -> torch.Tensor:
    """A seeded, irregular cloud -- distinct local geometry per point."""
    generator = torch.Generator().manual_seed(int(seed))
    xyz = torch.rand((int(count), 3), generator=generator) * float(extent)
    return xyz + torch.tensor(centre, dtype=xyz.dtype)


def rotation_matrix(angle: float, axis: int = 2) -> torch.Tensor:
    c, s = float(torch.cos(torch.tensor(angle))), float(torch.sin(torch.tensor(angle)))
    matrix = torch.eye(3)
    a, b = [i for i in range(3) if i != int(axis)]
    matrix[a, a], matrix[a, b], matrix[b, a], matrix[b, b] = c, -s, s, c
    return matrix


def two_part_scene(*, translation=(6.0, 0.0, 0.0), jitter: float = 0.0, seed: int = 0):
    """Static block + rigidly translated block. Returns (xyz0, xyzT, is_dynamic)."""
    static = irregular_cloud(120, seed=10 + seed, extent=1.5, centre=(0.0, 0.0, 0.0))
    moving = irregular_cloud(120, seed=20 + seed, extent=1.5, centre=(4.0, 0.0, 0.0))
    xyz0 = torch.cat((static, moving), dim=0)
    shift = torch.tensor(translation, dtype=xyz0.dtype)
    xyzT = torch.cat((static, moving + shift), dim=0)
    if float(jitter) > 0.0:
        generator = torch.Generator().manual_seed(999 + int(seed))
        xyzT = xyzT + torch.randn(xyzT.shape, generator=generator) * float(jitter)
    is_dynamic = torch.zeros(int(xyz0.shape[0]), dtype=torch.bool)
    is_dynamic[int(static.shape[0]):] = True
    return xyz0, xyzT, is_dynamic


def unit_scaling(count: int, value: float = 0.05) -> torch.Tensor:
    return torch.full((int(count), 3), float(value))


def make_graphs(xyz0, xyzT, *, appearance0=None, appearanceT=None, knn_rig: int = 12, knn_ctx: int = 24):
    source = build_relational_graph(
        xyz0, scaling=unit_scaling(int(xyz0.shape[0])), appearance=appearance0,
        knn_rig=knn_rig, knn_ctx=knn_ctx,
    )
    target = build_relational_graph(
        xyzT, scaling=unit_scaling(int(xyzT.shape[0])), appearance=appearanceT,
        knn_rig=knn_rig, knn_ctx=knn_ctx,
    )
    return source, target


# --------------------------------------------------------------------------
# Test 1 -- a static point stays local and the dynamic branch stays quiet
# --------------------------------------------------------------------------
def test_static_point_remains_local() -> None:
    xyz = irregular_cloud(200, seed=1, extent=2.0)
    source, target = make_graphs(xyz, xyz.clone())
    mismatch = compute_static_mismatch(xyz, xyz, source.effective_scale)
    check(float(mismatch.max()) < 1.0e-4, "identical geometry gives essentially zero static mismatch")

    coupling = build_relational_coupling(source, target, max_candidates=16, relational_sweeps=1)
    evidence = build_hybrid_evidence(
        coupling, xyz, xyz, source.effective_scale,
        static_mismatch=mismatch, knn_indices=source.knn_rig,
    )
    check(
        float(evidence.relational_evidence.max()) < 0.2,
        "with nothing displaced, no point accumulates confident non-local mass",
    )
    check(
        float(evidence.context_change.abs().max()) < 0.5,
        "context distances are preserved when nothing moved",
    )


def test_static_mismatch_survives_a_far_from_origin_scene() -> None:
    # A COLMAP frame routinely sits far from the world origin.  Uncentred,
    # float32 cdist cancellation put E_S at ~0.96 -- a full local scale of
    # pure arithmetic error -- for a cloud that had not moved at all.
    xyz = irregular_cloud(200, seed=2, extent=2.0)
    source, _target = make_graphs(xyz, xyz.clone())
    for offset in (0.0, 500.0, 5000.0):
        shifted = xyz + float(offset)
        error = float(compute_static_mismatch(shifted, shifted, source.effective_scale).max())
        check(error < 1.0e-4, f"static mismatch stays ~0 at offset {offset:g} (got {error:.2e})")


# --------------------------------------------------------------------------
# Test 2 -- a large rigid translation raises E_S and routes to the dynamic branch
# --------------------------------------------------------------------------
def test_rigid_translation_triggers_dynamic_branch() -> None:
    xyz0, xyzT, is_dynamic = two_part_scene()
    source, target = make_graphs(xyz0, xyzT)
    mismatch = compute_static_mismatch(xyz0, xyzT, source.effective_scale)
    check(
        float(mismatch[is_dynamic].median()) > 10.0 * float(mismatch[~is_dynamic].median() + 1.0),
        "the moved block has a far larger static mismatch than the static block",
    )

    gate, _threshold, _temperature = compute_dynamic_gate(mismatch)
    check(
        float(gate[is_dynamic].mean()) > float(gate[~is_dynamic].mean()),
        "the dynamic gate opens preferentially on the moved block",
    )

    coupling = build_relational_coupling(source, target, max_candidates=16, relational_sweeps=2)
    best = (coupling.mass * coupling.valid).argmax(dim=1)
    matched = coupling.tgt_idx.gather(1, best[:, None]).reshape(-1)
    moved_hits = is_dynamic[matched[is_dynamic]]
    check(
        float(moved_hits.float().mean()) > 0.9,
        "the relational graph retrieves the moved region, not the static one",
    )


# --------------------------------------------------------------------------
# Test 3 -- the dynamic relation is invariant to the rigid transform applied
# --------------------------------------------------------------------------
def test_dynamic_relation_is_rigid_invariant() -> None:
    xyz0, _xyzT, _is_dynamic = two_part_scene()
    rotation = rotation_matrix(0.7)
    shift = torch.tensor([3.0, -2.0, 1.5])
    moved = xyz0 @ rotation.T + shift

    plain, _ = make_graphs(xyz0, xyz0)
    transformed, _ = make_graphs(moved, moved)
    descriptor_drift = (plain.distance_descriptor - transformed.distance_descriptor).abs().max()
    check(
        float(descriptor_drift) < 1.0e-3,
        "the distance spectrum is unchanged by a rigid transform",
    )

    coupling = build_relational_coupling(plain, transformed, max_candidates=16, relational_sweeps=2)
    best = (coupling.mass * coupling.valid).argmax(dim=1)
    matched = coupling.tgt_idx.gather(1, best[:, None]).reshape(-1)
    identity = torch.arange(int(xyz0.shape[0]))
    check(
        float((matched == identity).float().mean()) > 0.8,
        "a rigidly transformed copy is recovered by relational matching alone",
    )


# --------------------------------------------------------------------------
# Test 4 -- intrinsically identical regions must stay uncertain
# --------------------------------------------------------------------------
def _duplicated_scene():
    """Two intrinsically IDENTICAL copies -- same points, same jitter, offset."""
    block = irregular_cloud(90, seed=7, extent=1.2)
    copy = block + torch.tensor([8.0, 0.0, 0.0])
    xyz = torch.cat((block, copy), dim=0)
    return xyz, block.shape[0]


def test_symmetric_geometry_remains_uncertain() -> None:
    xyz, half = _duplicated_scene()
    source, target = make_graphs(xyz, xyz.clone())
    coupling = build_relational_coupling(
        source, target, max_candidates=16, relational_sweeps=0, bidirectional_union=False,
    )
    confidence, peak = compute_relational_confidence(coupling)
    margin = compute_localisation_confidence(coupling)

    unique_xyz = irregular_cloud(2 * half, seed=31, extent=2.4)
    unique, unique_t = make_graphs(unique_xyz, unique_xyz.clone())
    unique_coupling = build_relational_coupling(
        unique, unique_t, max_candidates=16, relational_sweeps=0, bidirectional_union=False,
    )
    unique_confidence, unique_peak = compute_relational_confidence(unique_coupling)
    unique_margin = compute_localisation_confidence(unique_coupling)

    check(
        float(margin.median()) < 0.5 * float(unique_margin.median()),
        "duplicated geometry cannot localise: the best candidate has an equally good rival",
    )
    check(
        float(confidence.median()) < float(unique_confidence.median()),
        "duplicated geometry is less confidently localised than unique geometry",
    )
    check(
        float(peak.median()) < float(unique_peak.median()),
        "duplicated geometry spreads its peak mass instead of committing",
    )
    # A tie is not broken arbitrarily: the two copies split the top-1 choice.
    best = (coupling.mass * coupling.valid).argmax(dim=1)
    matched = coupling.tgt_idx.gather(1, best[:, None]).reshape(-1)
    own_copy = (matched < half) == (torch.arange(int(xyz.shape[0])) < half)
    check(
        0.3 < float(own_copy.float().mean()) < 0.7,
        "with identical copies the top-1 choice is at chance rather than confidently wrong",
    )


# --------------------------------------------------------------------------
# Test 5 -- appearance resolves what geometry cannot
# --------------------------------------------------------------------------
def test_appearance_resolves_geometric_ambiguity() -> None:
    xyz, half = _duplicated_scene()
    source, target = make_graphs(xyz, xyz.clone())
    geometry_only = build_relational_coupling(
        source, target, max_candidates=16, relational_sweeps=0, bidirectional_union=False,
    )

    colour = torch.zeros((int(xyz.shape[0]), 3))
    colour[:half] = torch.tensor([1.0, 0.0, 0.0])
    colour[half:] = torch.tensor([0.0, 0.0, 1.0])
    coloured_source, coloured_target = make_graphs(xyz, xyz.clone(), appearance0=colour, appearanceT=colour)
    with_appearance = build_relational_coupling(
        coloured_source, coloured_target, max_candidates=16, relational_sweeps=0,
        lambda_appearance=1.0, bidirectional_union=False,
    )
    check(
        float(compute_localisation_confidence(with_appearance).median())
        > float(compute_localisation_confidence(geometry_only).median()) + 0.5,
        "an appearance cue localises a coupling that geometry alone leaves tied",
    )

    def own_copy_fraction(coupling) -> float:
        best = (coupling.mass * coupling.valid).argmax(dim=1)
        matched = coupling.tgt_idx.gather(1, best[:, None]).reshape(-1)
        return float(((matched < half) == (torch.arange(int(xyz.shape[0])) < half)).float().mean())

    check(
        own_copy_fraction(geometry_only) < 0.7,
        "geometry alone picks between the identical copies at roughly chance",
    )
    check(
        own_copy_fraction(with_appearance) > 0.95,
        "appearance sends each point to its own copy rather than the duplicate",
    )
    # The entropy confidence of §12 is reported but deliberately NOT relied on
    # here: it is dominated by the candidate-pool size and the single global
    # softmax temperature, so it barely moves when the ambiguity is resolved.
    entropy_geometry, _ = compute_relational_confidence(geometry_only)
    entropy_appearance, _ = compute_relational_confidence(with_appearance)
    check(
        abs(float(entropy_appearance.median()) - float(entropy_geometry.median())) < 0.05,
        "normalised entropy is insensitive to tie-breaking -- why the margin gates instead",
    )


def test_appearance_weight_without_appearance_is_an_error() -> None:
    xyz = irregular_cloud(60, seed=5)
    source, target = make_graphs(xyz, xyz.clone())
    try:
        build_relational_coupling(source, target, max_candidates=8, lambda_appearance=1.0)
    except ValueError:
        CHECKS.append("a requested appearance weight with no appearance raises instead of silently dropping the cue")
        return
    raise AssertionError("lambda_appearance > 0 must not silently fall back to geometry only")


# --------------------------------------------------------------------------
# Test 6 -- the combined hypothesis beats either branch alone
# --------------------------------------------------------------------------
def _refilled_scene():
    """The case the hybrid design exists for: the vacated region is refilled.

    A moved part whose old location is reoccupied by other target geometry
    has a SMALL static mismatch despite having moved, so the XYZ hypothesis
    is not merely noisy there -- it is confidently wrong.  This is the only
    regime in which a second hypothesis can add anything, and the spec's
    success criterion is stated for exactly it.
    """
    static = irregular_cloud(120, seed=41, extent=1.5, centre=(0.0, 0.0, 0.0))
    moving = irregular_cloud(120, seed=42, extent=1.5, centre=(4.0, 0.0, 0.0))
    # The two endpoints are INDEPENDENTLY sampled where they are static, as
    # two independently trained 3DGS models are.  Reusing the same points
    # would give every static Gaussian a mismatch of exactly zero and hand
    # the XYZ hypothesis a perfect score no real scene would give it.
    static_t = irregular_cloud(120, seed=141, extent=1.5, centre=(0.0, 0.0, 0.0))
    filler = irregular_cloud(120, seed=43, extent=1.5, centre=(4.0, 0.0, 0.0))
    xyz0 = torch.cat((static, moving), dim=0)
    xyzT = torch.cat((static_t, moving + torch.tensor([6.0, 0.0, 0.0]), filler), dim=0)
    is_dynamic = torch.zeros(int(xyz0.shape[0]), dtype=torch.bool)
    is_dynamic[int(static.shape[0]):] = True
    return xyz0, xyzT, is_dynamic


def test_combined_hypothesis_beats_either_alone() -> None:
    xyz0, xyzT, is_dynamic = _refilled_scene()
    source, target = make_graphs(xyz0, xyzT)
    coupling = build_relational_coupling(source, target, max_candidates=16, relational_sweeps=2)
    evidence = build_hybrid_evidence(
        coupling, xyz0, xyzT, source.effective_scale, knn_indices=source.knn_rig,
    )

    static_only = _binary_auc(evidence.static_mismatch, is_dynamic)
    check(
        static_only < 0.75,
        f"the refilled region defeats the XYZ hypothesis on its own (AUC {static_only:.4f})",
    )

    relational_only = _binary_auc(evidence.relational_evidence, is_dynamic)
    combined, _terms = combine_hypothesis_evidence(
        evidence, weights=HybridWeights(static_mismatch=2.0, relational=2.0, context=2.0),
    )
    combined_auc = _binary_auc(combined, is_dynamic)
    check(
        combined_auc > static_only,
        f"combined AUC {combined_auc:.4f} beats raw static {static_only:.4f} "
        f"(relational alone {relational_only:.4f})",
    )
    check(
        combined_auc > 0.8,
        f"the combined score recovers the moved part the static branch missed (AUC {combined_auc:.4f})",
    )


# --------------------------------------------------------------------------
# Test 9 -- internal vs context change separates the three regimes
# --------------------------------------------------------------------------
def test_context_contrast_separates_static_from_rigid_dynamic() -> None:
    xyz0, xyzT, is_dynamic = two_part_scene()
    source, _target = make_graphs(xyz0, xyzT)
    shift = torch.zeros_like(xyz0)
    shift[is_dynamic] = torch.tensor([6.0, 0.0, 0.0])
    transported = xyz0 + shift

    internal, edge_confidence = compute_internal_distance_change(
        xyz0, transported, source.knn_rig, source.effective_scale,
    )
    anchors = compute_static_context_anchors(compute_static_mismatch(xyz0, xyzT, source.effective_scale))
    context = compute_context_distance_change(xyz0, transported, anchors, source.effective_scale)

    check(
        float(internal[is_dynamic].median()) < 0.1,
        "a rigidly moved block keeps its internal distances (internal change ~ 0)",
    )
    check(
        float(context[is_dynamic].median()) > 10.0 * float(context[~is_dynamic].median() + 1.0e-3),
        "the moved block changes its distance to the static anchors, the static block does not",
    )
    check(
        float(edge_confidence[is_dynamic].mean()) > 0.9,
        "edge rigidity confidence stays high across a genuinely rigid motion",
    )


def test_static_anchors_exclude_moving_points() -> None:
    xyz0, xyzT, is_dynamic = two_part_scene()
    source, _target = make_graphs(xyz0, xyzT)
    mismatch = compute_static_mismatch(xyz0, xyzT, source.effective_scale)
    anchors = compute_static_context_anchors(mismatch)
    check(int(anchors.sum()) > 0, "the anchor set is non-empty")
    check(
        not bool((anchors & is_dynamic).any()),
        "no moving point is admitted as a static context anchor",
    )


# --------------------------------------------------------------------------
# Test 10 -- a confidently wrong match is rejected, not promoted
# --------------------------------------------------------------------------
def test_bad_nonlocal_match_is_rejected() -> None:
    xyz0, xyzT, is_dynamic = two_part_scene()
    source, _target = make_graphs(xyz0, xyzT)
    scale = source.effective_scale

    rigid = xyz0.clone()
    rigid[is_dynamic] = rigid[is_dynamic] + torch.tensor([6.0, 0.0, 0.0])

    # Same points reach the same region, but permuted within it: the static
    # mismatch is identical to the rigid case, so only the relational terms
    # can tell the two apart.
    generator = torch.Generator().manual_seed(4)
    scrambled = rigid.clone()
    moving_idx = is_dynamic.nonzero(as_tuple=True)[0]
    scrambled[moving_idx] = rigid[moving_idx[torch.randperm(int(moving_idx.numel()), generator=generator)]]

    confidence = torch.ones(int(xyz0.shape[0]))
    anchors = compute_static_context_anchors(compute_static_mismatch(xyz0, xyzT, scale))

    rigid_internal, _ = compute_internal_distance_change(xyz0, rigid, source.knn_rig, scale)
    bad_internal, _ = compute_internal_distance_change(xyz0, scrambled, source.knn_rig, scale)
    check(
        float(bad_internal[is_dynamic].median()) > 5.0 * float(rigid_internal[is_dynamic].median() + 1.0e-3),
        "a scrambled match tears the neighbourhood apart while a rigid one does not",
    )

    rigid_evidence = compute_relational_dynamic_evidence(
        compute_context_distance_change(xyz0, rigid, anchors, scale), rigid_internal, confidence,
    )
    bad_evidence = compute_relational_dynamic_evidence(
        compute_context_distance_change(xyz0, scrambled, anchors, scale), bad_internal, confidence,
    )
    check(
        float(bad_evidence[is_dynamic].median()) < float(rigid_evidence[is_dynamic].median()),
        "high static mismatch with poor internal rigidity yields weaker dynamic evidence",
    )


def test_saturating_score_is_absolute_not_ranked() -> None:
    # A rank-based score would place a typical point at 0.5 even in a scene
    # where nothing moved; the §21 claims are absolute, so g must be too.
    quiet = torch.full((64,), 1.0e-4)
    check(float(saturating_score(quiet).max()) < 1.0e-3, "g(~0) is ~0 regardless of the rest of the scene")
    check(abs(float(saturating_score(torch.tensor([1.0]), knee=1.0))) - 0.5 < 1.0e-6, "g(knee) == 0.5")


# --------------------------------------------------------------------------
# Test 12 -- identical evidence must not accumulate
# --------------------------------------------------------------------------
def test_no_repeated_evidence_accumulation() -> None:
    prior = torch.tensor([0.2, 0.5, 0.8])
    evidence = torch.tensor([1.0, -1.0, 0.5])
    current = prior.clone()

    first, proposal_a = update_from_immutable_prior(prior, current, evidence, ema=0.5)
    second, proposal_b = update_from_immutable_prior(prior, first, evidence, ema=0.5)
    check(
        torch.allclose(proposal_a, proposal_b),
        "unchanged evidence gives an unchanged proposal, so nothing compounds",
    )
    check(
        bool((second - proposal_b).abs().max() < (first - proposal_a).abs().max()),
        "only the EMA moves, and it moves toward the fixed proposal",
    )

    for _ in range(40):
        current, proposal = update_from_immutable_prior(prior, current, evidence, ema=0.5)
    check(
        torch.allclose(current, proposal, atol=1.0e-5),
        "iterating to convergence lands on the proposal, not past it",
    )


def test_expected_target_falls_back_to_no_displacement() -> None:
    # An all-dustbin row divided by a clamped denominator would land on the
    # world origin and read downstream as an enormous confident displacement.
    xyz0 = irregular_cloud(40, seed=77, extent=1.0, centre=(50.0, 50.0, 50.0))
    source, target = make_graphs(xyz0, xyz0.clone())
    coupling = build_relational_coupling(source, target, max_candidates=8, relational_sweeps=0)
    coupling.mass = torch.zeros_like(coupling.mass)
    coupling.row_mass = torch.zeros_like(coupling.row_mass)
    coupling.unmatched_src = torch.ones_like(coupling.unmatched_src)

    expected = coupling.expected_target(xyz0, source_xyz=xyz0)
    check(
        torch.allclose(expected, xyz0),
        "a fully unmatched row reports no displacement rather than a jump to the origin",
    )


def main() -> None:
    tests = [value for name, value in sorted(globals().items()) if name.startswith("test_") and callable(value)]
    for test in tests:
        test()
        print(f"  ok  {test.__name__}")
    print(f"\n{len(tests)} tests passed, {len(CHECKS)} assertions:")
    for message in CHECKS:
        print(f"  - {message}")


if __name__ == "__main__":
    main()
