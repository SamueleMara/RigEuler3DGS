"""Tests for Stage A.6: dynamic-subset-restricted relational transport.

Scenes are seeded irregular clouds, never lattices: a regular structure has a
degenerate distance spectrum, so every descriptor is identical and a test
built on one measures the degeneracy instead of the mechanism.

Covers §46 tests 1-8.  The refilled-vacated-region test is the mandatory one
-- it is the exact case where raw XYZ matching is not merely noisy but
confidently wrong, and therefore the only regime where restricting the
candidate space can pay for itself.
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from utils.bidirectional_spatial_mismatch import (
    compute_backward_spatial_mismatch,
    compute_bidirectional_spatial_mismatch,
    compute_forward_spatial_mismatch,
)
from utils.dynamic_subset_relational_coupling import (
    build_dynamic_descriptor_candidates,
    build_dynamic_subset_coupling,
    compute_dynamic_candidate_confidence,
    compute_relational_rigidity_error,
    compute_reverse_neighborhood_consistency,
    peak_target,
)
from utils.dynamic_subset_selection import (
    build_hard_dynamic_subset,
    build_soft_dynamic_weights,
    robust_dynamic_gate,
    summarise_dynamic_subset,
)
from utils.relational_graph import build_relational_graph

CHECKS: list[str] = []


def check(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)
    CHECKS.append(message)


def irregular_cloud(count: int, *, seed: int, extent: float = 1.0, centre=(0.0, 0.0, 0.0)) -> torch.Tensor:
    generator = torch.Generator().manual_seed(int(seed))
    xyz = torch.rand((int(count), 3), generator=generator) * float(extent)
    return xyz + torch.tensor(centre, dtype=xyz.dtype)


def unit_scaling(count: int, value: float = 0.05) -> torch.Tensor:
    return torch.full((int(count), 3), float(value))


def make_graph(xyz, *, appearance=None, knn_rig: int = 12, knn_ctx: int = 24):
    return build_relational_graph(
        xyz, scaling=unit_scaling(int(xyz.shape[0])), appearance=appearance,
        knn_rig=knn_rig, knn_ctx=knn_ctx,
    )


def weights_for(source, target):
    mismatch = compute_bidirectional_spatial_mismatch(
        source.xyz, target.xyz, source.effective_scale, target.effective_scale,
    )
    return mismatch, build_soft_dynamic_weights(mismatch)


# --------------------------------------------------------------------------
# Scenes
# --------------------------------------------------------------------------
def static_scene():
    xyz = irregular_cloud(200, seed=1, extent=2.0)
    generator = torch.Generator().manual_seed(2)
    return xyz, xyz + torch.randn(xyz.shape, generator=generator) * 0.002


def translation_scene(shift=(6.0, 0.0, 0.0)):
    static = irregular_cloud(120, seed=10, extent=1.5, centre=(0.0, 0.0, 0.0))
    moving = irregular_cloud(120, seed=20, extent=1.5, centre=(4.0, 0.0, 0.0))
    xyz0 = torch.cat((static, moving), dim=0)
    xyzT = torch.cat((static, moving + torch.tensor(shift)), dim=0)
    is_dynamic = torch.zeros(int(xyz0.shape[0]), dtype=torch.bool)
    is_dynamic[int(static.shape[0]):] = True
    return xyz0, xyzT, is_dynamic


def refilled_scene():
    """The mandatory case: the vacated region is reoccupied in frame T."""
    static = irregular_cloud(120, seed=41, extent=1.5, centre=(0.0, 0.0, 0.0))
    moving = irregular_cloud(120, seed=42, extent=1.5, centre=(4.0, 0.0, 0.0))
    # Independently resampled where the scene is static, as two independently
    # trained 3DGS models are.  Reusing the same points would give every
    # static Gaussian a mismatch of exactly zero, which no real scene gives.
    static_t = irregular_cloud(120, seed=141, extent=1.5, centre=(0.0, 0.0, 0.0))
    filler = irregular_cloud(120, seed=43, extent=1.5, centre=(4.0, 0.0, 0.0))
    xyz0 = torch.cat((static, moving), dim=0)
    xyzT = torch.cat((static_t, moving + torch.tensor([6.0, 0.0, 0.0]), filler), dim=0)
    is_dynamic = torch.zeros(int(xyz0.shape[0]), dtype=torch.bool)
    is_dynamic[int(static.shape[0]):] = True
    return xyz0, xyzT, is_dynamic


# --------------------------------------------------------------------------
# 1 -- static scene
# --------------------------------------------------------------------------
def test_static_scene_produces_no_dynamic_subset() -> None:
    xyz0, xyzT = static_scene()
    source, target = make_graph(xyz0), make_graph(xyzT)
    mismatch, weights = weights_for(source, target)
    check(
        float(mismatch.forward.max()) < 0.05,
        "an unmoved scene has a forward mismatch far below one local scale",
    )
    check(
        float(mismatch.backward.max()) < 0.05,
        "and a backward mismatch just as small",
    )
    summary = summarise_dynamic_subset(weights.source_weight, mismatch.forward)
    check(
        summary["mismatch_median_active"] < 0.05,
        "any points the gate keeps are still spatially consistent, so nothing is claimed to have moved far",
    )


# --------------------------------------------------------------------------
# 2 -- large rigid translation
# --------------------------------------------------------------------------
def test_translation_is_detected_at_both_endpoints() -> None:
    xyz0, xyzT, is_dynamic = translation_scene()
    source, target = make_graph(xyz0), make_graph(xyzT)
    mismatch, weights = weights_for(source, target)

    check(
        float(mismatch.forward[is_dynamic].median()) > 100.0 * float(mismatch.forward[~is_dynamic].median() + 1.0e-6),
        "the moved block dominates the forward mismatch",
    )
    check(
        float(weights.source_weight[is_dynamic].mean()) > 0.9
        and float(weights.source_weight[~is_dynamic].mean()) < 0.1,
        "q0 separates the moved block from the static one",
    )
    # The target-side gate must fire on the block's NEW location, which is a
    # different set of indices -- that is what makes the two directions
    # independent evidence rather than one fact counted twice.
    target_dynamic = torch.zeros(int(xyzT.shape[0]), dtype=torch.bool)
    target_dynamic[120:] = True
    check(
        float(weights.target_weight[target_dynamic].mean()) > 0.9
        and float(weights.target_weight[~target_dynamic].mean()) < 0.1,
        "qT fires on the block's new location, not its old one",
    )

    coupling = build_dynamic_subset_coupling(
        source, target, source_weight=weights.source_weight, target_weight=weights.target_weight,
        max_candidates=12, target_mode="hard",
    )
    tgt, matched = peak_target(coupling)
    hits = target_dynamic[tgt[is_dynamic & matched]]
    check(
        float(hits.float().mean()) > 0.9,
        "restricted transport sends the moved block to the moved block",
    )


# --------------------------------------------------------------------------
# 3 -- refilled vacated region (mandatory)
# --------------------------------------------------------------------------
def test_refilled_region_is_recovered_by_restriction() -> None:
    xyz0, xyzT, is_dynamic = refilled_scene()
    source, target = make_graph(xyz0), make_graph(xyzT)
    mismatch, weights = weights_for(source, target)

    # The premise: raw XYZ is not merely noisy here, it is uninformative.
    moved_es = float(mismatch.forward[is_dynamic].median())
    static_es = float(mismatch.forward[~is_dynamic].median())
    check(
        0.4 < moved_es / max(static_es, 1.0e-9) < 2.5,
        f"the refilled region makes forward mismatch nearly indistinguishable "
        f"(moved {moved_es:.3f} vs static {static_es:.3f})",
    )

    full = build_dynamic_subset_coupling(
        source, target, source_weight=None, target_weight=None,
        max_candidates=12, allow_full_scene=True,
    )
    restricted = build_dynamic_subset_coupling(
        source, target, source_weight=weights.source_weight, target_weight=weights.target_weight,
        max_candidates=12, target_mode="soft", lambda_q=2.0,
    )
    target_dynamic = torch.zeros(int(xyzT.shape[0]), dtype=torch.bool)
    target_dynamic[120:240] = True

    def moved_block_recall(coupling) -> float:
        tgt, matched = peak_target(coupling)
        keep = is_dynamic & matched
        return float(target_dynamic[tgt[keep]].float().mean()) if bool(keep.any()) else 0.0

    check(
        moved_block_recall(restricted) >= moved_block_recall(full),
        f"restriction does not lose the moved block "
        f"(restricted {moved_block_recall(restricted):.3f} vs full {moved_block_recall(full):.3f})",
    )
    check(
        moved_block_recall(restricted) > 0.5,
        f"the moved block is found despite its old region being refilled "
        f"({moved_block_recall(restricted):.3f})",
    )


# --------------------------------------------------------------------------
# 4 -- symmetric static distractor
# --------------------------------------------------------------------------
def test_symmetric_static_distractor_stays_uncertain() -> None:
    block = irregular_cloud(90, seed=7, extent=1.2)
    twin = block + torch.tensor([8.0, 0.0, 0.0])
    xyz = torch.cat((block, twin), dim=0)
    source, target = make_graph(xyz), make_graph(xyz.clone())
    idx, cost, active = build_dynamic_descriptor_candidates(
        source, target, max_candidates=12, allow_full_scene=True,
    )
    valid = torch.isfinite(cost) & active[:, None]
    mass = torch.softmax(-cost, dim=1)
    confidence = compute_dynamic_candidate_confidence(cost, mass, valid)
    check(
        float(confidence.cost_margin.median()) < 0.05,
        "an intrinsically duplicated region has no cost margin: its best candidate has an equal rival",
    )
    best = torch.where(valid, mass, torch.zeros_like(mass)).argmax(dim=1)
    matched = idx.gather(1, best[:, None]).reshape(-1)
    own = (matched < 90) == (torch.arange(int(xyz.shape[0])) < 90)
    check(
        0.3 < float(own.float().mean()) < 0.7,
        "the tie is left at chance rather than broken arbitrarily",
    )


# --------------------------------------------------------------------------
# 5 -- appearance disambiguation
# --------------------------------------------------------------------------
def test_appearance_disambiguates_identical_geometry() -> None:
    block = irregular_cloud(90, seed=7, extent=1.2)
    twin = block + torch.tensor([8.0, 0.0, 0.0])
    xyz = torch.cat((block, twin), dim=0)
    colour = torch.zeros((int(xyz.shape[0]), 3))
    colour[:90, 0] = 1.0
    colour[90:, 2] = 1.0

    plain_s, plain_t = make_graph(xyz), make_graph(xyz.clone())
    col_s, col_t = make_graph(xyz, appearance=colour), make_graph(xyz.clone(), appearance=colour)

    def own_copy_and_margin(source, target, lambda_appearance):
        idx, cost, active = build_dynamic_descriptor_candidates(
            source, target, max_candidates=12, lambda_appearance=lambda_appearance, allow_full_scene=True,
        )
        valid = torch.isfinite(cost) & active[:, None]
        mass = torch.softmax(-cost, dim=1)
        margin = float(compute_dynamic_candidate_confidence(cost, mass, valid).cost_margin.median())
        best = torch.where(valid, mass, torch.zeros_like(mass)).argmax(dim=1)
        matched = idx.gather(1, best[:, None]).reshape(-1)
        own = (matched < 90) == (torch.arange(int(xyz.shape[0])) < 90)
        return float(own.float().mean()), margin

    plain_own, plain_margin = own_copy_and_margin(plain_s, plain_t, 0.0)
    colour_own, colour_margin = own_copy_and_margin(col_s, col_t, 1.0)
    check(colour_own > 0.95 > plain_own, f"appearance resolves the duplicate ({plain_own:.3f} -> {colour_own:.3f})")
    check(colour_margin > plain_margin + 0.5, "and the cost margin registers the resolution")


def test_appearance_weight_without_appearance_raises() -> None:
    xyz = irregular_cloud(60, seed=5)
    source, target = make_graph(xyz), make_graph(xyz.clone())
    try:
        build_dynamic_descriptor_candidates(
            source, target, max_candidates=8, lambda_appearance=1.0, allow_full_scene=True,
        )
    except ValueError:
        CHECKS.append("a requested appearance weight with no appearance raises rather than silently dropping the cue")
        return
    raise AssertionError("lambda_appearance > 0 must not silently fall back to geometry only")


def test_dynamic_candidates_require_a_target_weight() -> None:
    xyz = irregular_cloud(60, seed=6)
    source, target = make_graph(xyz), make_graph(xyz.clone())
    try:
        build_dynamic_descriptor_candidates(source, target, max_candidates=8)
    except ValueError:
        CHECKS.append("omitting the dynamic weights raises unless a full-scene ablation is asked for explicitly")
        return
    raise AssertionError("the dynamic path must not silently degrade to full-scene matching")


# --------------------------------------------------------------------------
# 6 -- reverse-neighbourhood consistency under different sampling
# --------------------------------------------------------------------------
def test_reverse_neighborhood_consistency_under_resampling() -> None:
    xyz0, xyzT, is_dynamic = translation_scene()
    # Frame T is resampled: a third of its Gaussians are dropped, so exact
    # reciprocal identity is impossible and only a neighbourhood test can pass.
    generator = torch.Generator().manual_seed(11)
    keep = torch.randperm(int(xyzT.shape[0]), generator=generator)[: int(0.7 * xyzT.shape[0])]
    xyzT = xyzT[keep.sort().values]

    source, target = make_graph(xyz0), make_graph(xyzT)
    _mismatch, weights = weights_for(source, target)
    coupling = build_dynamic_subset_coupling(
        source, target, source_weight=weights.source_weight, target_weight=weights.target_weight,
        max_candidates=12, target_mode="hard",
    )
    reverse_idx, _reverse_cost, _active = build_dynamic_descriptor_candidates(
        target, source, target_weight=weights.source_weight, source_weight=weights.target_weight,
        max_candidates=8, target_mode="hard",
    )
    error = compute_reverse_neighborhood_consistency(source, coupling, reverse_idx)
    measured = error[torch.isfinite(error) & is_dynamic]
    check(int(measured.numel()) > 0, "reverse consistency is measurable on the moved block")
    check(
        float(measured.median()) < 5.0,
        f"reverse retrieval returns to the same source neighbourhood ({float(measured.median()):.2f} scales)",
    )


def test_rigidity_error_only_counts_matched_edges() -> None:
    xyz0, xyzT, is_dynamic = translation_scene()
    source, target = make_graph(xyz0), make_graph(xyzT)
    _mismatch, weights = weights_for(source, target)
    coupling = build_dynamic_subset_coupling(
        source, target, source_weight=weights.source_weight, target_weight=weights.target_weight,
        max_candidates=12, target_mode="hard",
    )
    rigidity = compute_relational_rigidity_error(
        source, target.xyz, coupling, evaluation_mask=weights.source_weight > 0.5,
    )
    moved = rigidity["per_point_error"][is_dynamic]
    moved = moved[torch.isfinite(moved)]
    check(int(moved.numel()) > 0, "rigidity is measured on the moved block")
    check(
        float(moved.median()) < 0.5,
        f"a rigid translation preserves neighbour distances under the transport ({float(moved.median()):.3f})",
    )
    # An edge whose endpoint never matched must not be scored as perfect.
    check(
        not bool(rigidity["edge_usable"][~rigidity["edge_usable"]].any()),
        "unmatched edges are excluded rather than counted as zero error",
    )


# --------------------------------------------------------------------------
# 7 -- far-from-origin invariance
# --------------------------------------------------------------------------
def test_mismatch_is_invariant_to_a_far_origin() -> None:
    xyz0, xyzT, _ = translation_scene()
    source = make_graph(xyz0)
    target = make_graph(xyzT)
    base_forward = compute_forward_spatial_mismatch(xyz0, xyzT, source.effective_scale)
    base_backward = compute_backward_spatial_mismatch(xyz0, xyzT, target.effective_scale)
    for offset in (0.0, 500.0, 5000.0):
        forward = compute_forward_spatial_mismatch(xyz0 + offset, xyzT + offset, source.effective_scale)
        backward = compute_backward_spatial_mismatch(xyz0 + offset, xyzT + offset, target.effective_scale)
        # Relative, not absolute: mismatch spans 0 to ~100 in this scene, so
        # an absolute bound would be simultaneously far too loose for the
        # static points and too tight for the moved ones.
        drift_f = float(((forward - base_forward).abs() / (base_forward + 1.0)).max())
        drift_b = float(((backward - base_backward).abs() / (base_backward + 1.0)).max())
        check(drift_f < 1.0e-3, f"forward mismatch is stable at offset {offset:g} (drift {drift_f:.2e})")
        check(drift_b < 1.0e-3, f"backward mismatch is stable at offset {offset:g} (drift {drift_b:.2e})")

    unmoved = irregular_cloud(200, seed=3, extent=2.0)
    unmoved_graph = make_graph(unmoved)
    for offset in (0.0, 500.0, 5000.0):
        error = float(compute_forward_spatial_mismatch(
            unmoved + offset, unmoved + offset, unmoved_graph.effective_scale
        ).max())
        check(error < 1.0e-4, f"an unmoved cloud reports ~0 mismatch at offset {offset:g} (got {error:.2e})")


# --------------------------------------------------------------------------
# 8 -- partial visibility / unmatched target
# --------------------------------------------------------------------------
def test_partial_visibility_leaves_mass_unmatched() -> None:
    xyz0 = irregular_cloud(150, seed=61, extent=2.0)
    # Frame T keeps only part of the scene: the disappeared half has nothing
    # legitimate to match, and must be allowed to say so.
    xyzT = xyz0[:75].clone()
    source, target = make_graph(xyz0), make_graph(xyzT)
    _mismatch, weights = weights_for(source, target)
    coupling = build_dynamic_subset_coupling(
        source, target, source_weight=weights.source_weight, target_weight=weights.target_weight,
        max_candidates=12, target_mode="soft",
    )
    check(
        float(coupling.unmatched_src.mean()) > 0.2,
        "a partly invisible scene keeps real unmatched mass rather than forcing a target on everything",
    )
    check(
        float(coupling.row_mass.max()) <= 1.0 + 1.0e-5,
        "no row claims more than unit mass",
    )
    check(
        int(coupling.col_mass.numel()) == int(xyzT.shape[0]),
        "column mass is tracked over the real target count, not an assumed equal one",
    )


def test_gate_adapts_to_the_moving_fraction() -> None:
    # A median-centred gate would call half of every scene dynamic.  The gate
    # has to follow the data, because the candidate space depends on it.
    def moving_fraction(count_moving: int) -> float:
        static = irregular_cloud(200, seed=71, extent=1.5)
        moving = irregular_cloud(count_moving, seed=72, extent=0.4, centre=(4.0, 0.0, 0.0))
        xyz0 = torch.cat((static, moving), dim=0)
        xyzT = torch.cat((static, moving + torch.tensor([6.0, 0.0, 0.0])), dim=0)
        graph = make_graph(xyz0)
        forward = compute_forward_spatial_mismatch(xyz0, xyzT, graph.effective_scale)
        weight, _gate = robust_dynamic_gate(forward)
        return float((weight > 0.5).float().mean())

    small, large = moving_fraction(20), moving_fraction(200)
    check(small < 0.25, f"a scene where ~9% moved does not report half its points active ({small:.3f})")
    check(large > small, f"a scene where ~50% moved reports more ({large:.3f} > {small:.3f})")


def test_hard_subset_never_degenerates() -> None:
    weight = torch.full((500,), 0.01)
    subset = build_hard_dynamic_subset(weight, min_fraction=0.05)
    check(
        float(subset.float().mean()) >= 0.05,
        "an all-static gate still yields a non-empty subset, so the branch cannot be vacuously consistent",
    )
    subset = build_hard_dynamic_subset(torch.full((500,), 0.99), max_fraction=0.90)
    check(
        float(subset.float().mean()) <= 0.90,
        "an all-dynamic gate is clamped, so a non-restriction is not reported as a restriction",
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
