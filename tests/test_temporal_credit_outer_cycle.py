"""Temporal credit outer cycle: required tests A-L (section 44).

Plus checks for the parts whose failure would be silent: section 6's fixed
normalisation constants, section 11's changed-only credit, section 22's clamp,
section 32's components, and the section 35/36 artifact contracts.
"""

from __future__ import annotations

import inspect
import sys
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from rigid_parts.neighborhood import build_neighbor_graph
from rigid_parts.temporal_credit import (
    LocalScore,
    QualityScore,
    ScoreNormalization,
    TemporalCreditCycle,
    align_part_permutation,
    apply_temporal_memory,
    changed_components,
    combine_global_score,
    combine_local_score,
    combine_reward,
    detect_label_oscillation,
    establish_normalization,
    evaluate_match_score,
    local_transition_reward,
    neighborhood_average,
    temporal_credit_loss,
    transition_type_report,
    unsupported_penalty_from,
    update_temporal_memory,
)

DT = torch.float64


def _graph(n=64, spread=1.0, k=8):
    """A line of points so neighbourhoods are local and predictable."""
    x = torch.linspace(0.0, float(spread), n, dtype=DT)[:, None]
    pts = torch.cat((x, torch.zeros((n, 2), dtype=DT)), dim=1)
    return build_neighbor_graph(pts, k=k), pts


def _local(total, graph=None):
    total = torch.as_tensor(total, dtype=DT)
    if graph is None:
        graph, _ = _graph(int(total.numel()))
    return LocalScore(
        render=total.clone(), match=total.clone(),
        match_valid=torch.ones_like(total, dtype=torch.bool),
        render_support=torch.ones_like(total),
        total=total, smoothed=neighborhood_average(graph, total),
    )


def _quality(total, render=None, match=None):
    r = float(total if render is None else render)
    m = float(total if match is None else match)
    return QualityScore(raw_render=r, raw_pixel_match=m, raw_final_nn=0.0, raw_match=m,
                        normalized_render=r, normalized_match=m, total=float(total))


# ---------------------------------------------------------------------------
# A. Baseline first iteration
# ---------------------------------------------------------------------------


def test_a_baseline_has_zero_memory_and_no_reward():
    graph, _ = _graph()
    n = graph.num_nodes
    cycle = TemporalCreditCycle(n, 2)
    assert float(cycle.memory.abs().max()) == 0.0, "memory did not start at exactly zero"
    assert cycle.previous_score is None, "a previous score existed before any solve"

    z0 = torch.zeros(n, dtype=torch.long)
    record = cycle.observe(outer=0, labels=z0, quality=_quality(1.0),
                           local=_local(torch.ones(n, dtype=DT)), graph=graph)

    assert record["is_baseline"] is True, "outer 0 was not marked as the baseline"
    assert record["score_total"] == 1.0, "S^(0) was not computed"
    assert record["num_changed"] == 0, "the baseline invented label transitions"
    assert float(cycle.memory.abs().max()) == 0.0, \
        "memory was updated at the baseline; §7 forbids any reward for z^(0)"
    assert record["relative_global_reward"] != record["relative_global_reward"], \
        "a fake S^(-1) produced a finite baseline reward"
    assert cycle.previous_score is not None, "the baseline was not stored as S^(0)"


def test_a_no_fake_previous_score_is_created():
    """Only after the FIRST label update may a reward appear (§7, §8)."""
    graph, _ = _graph()
    n = graph.num_nodes
    cycle = TemporalCreditCycle(n, 2)
    z0 = torch.zeros(n, dtype=torch.long)
    cycle.observe(outer=0, labels=z0, quality=_quality(1.0),
                  local=_local(torch.ones(n, dtype=DT)), graph=graph)
    z1 = z0.clone()
    z1[:10] = 1
    record = cycle.observe(outer=1, labels=z1, quality=_quality(0.5),
                           local=_local(torch.full((n,), 0.5, dtype=DT)), graph=graph)
    assert record["is_baseline"] is False
    # R^(1) = S^(0) - S^(1) = 1.0 - 0.5 = +0.5, relative 0.5 up to §9's own eps.
    assert abs(record["relative_global_reward"] - 0.5) < 1.0e-7, \
        f"R_rel^(1) is {record['relative_global_reward']}, expected 0.5"
    assert record["num_changed"] == 10


# ---------------------------------------------------------------------------
# B / C. Positive and negative transition reward
# ---------------------------------------------------------------------------


def _two_step(score_new, local_new_value, changed_slice=slice(0, 10), n=64):
    graph, _ = _graph(n)
    cycle = TemporalCreditCycle(n, 2)
    z0 = torch.zeros(n, dtype=torch.long)
    cycle.observe(outer=0, labels=z0, quality=_quality(1.0),
                  local=_local(torch.ones(n, dtype=DT)), graph=graph)
    z1 = z0.clone()
    z1[changed_slice] = 1
    local_new = torch.ones(n, dtype=DT)
    local_new[changed_slice] = float(local_new_value)
    record = cycle.observe(outer=1, labels=z1, quality=_quality(score_new),
                           local=_local(local_new), graph=graph)
    return cycle, record, z0, z1, graph


def test_b_improving_change_gets_positive_reward():
    cycle, record, z0, z1, _ = _two_step(score_new=0.6, local_new_value=0.4)
    assert record["relative_global_reward"] > 0, "a better score gave a non-positive global reward"
    assert record["mean_local_reward_changed"] > 0, \
        f"an improving change got reward {record['mean_local_reward_changed']}"
    assert record["num_positive_reward_changes"] > 0
    assert record["num_negative_reward_changes"] == 0
    # §20: the NEW class is reinforced.
    assert float(cycle.memory[0, 1]) > 0, "the new class was not reinforced"


def test_c_degrading_change_gets_negative_reward():
    cycle, record, _, _, _ = _two_step(score_new=1.8, local_new_value=2.5)
    assert record["relative_global_reward"] < 0, "a worse score gave a non-negative global reward"
    assert record["mean_local_reward_changed"] < 0, \
        f"a degrading change got reward {record['mean_local_reward_changed']}"
    assert record["num_negative_reward_changes"] > 0
    assert float(cycle.memory[0, 1]) < 0, "a harmful transition still reinforced the new class"


# ---------------------------------------------------------------------------
# D / E. Memory direction
# ---------------------------------------------------------------------------


def test_d_positive_transition_reinforces_new_and_penalises_old():
    n = 8
    memory = torch.zeros((n, 3), dtype=DT)
    old = torch.zeros(n, dtype=torch.long)          # a = 0
    new = torch.full((n,), 2, dtype=torch.long)     # b = 2
    changed = torch.ones(n, dtype=torch.bool)
    reward = torch.full((n,), 0.8, dtype=DT)
    out = update_temporal_memory(memory, old, new, changed, reward, decay=0.8)
    assert float(out[0, 2]) > 0, "M_ib did not increase for a positive reward"
    assert float(out[0, 0]) < 0, "M_ia did not decrease for a positive reward"
    assert abs(float(out[0, 2]) - 0.2 * 0.8) < 1.0e-12, "M_ib is not gamma*M + (1-gamma)*r"
    assert abs(float(out[0, 0]) + 0.2 * 0.8) < 1.0e-12, "M_ia is not gamma*M - (1-gamma)*r"
    assert float(out[0, 1]) == 0.0, "an untouched class was modified"


def test_e_harmful_transition_reinforces_old_and_penalises_new():
    n = 8
    memory = torch.zeros((n, 3), dtype=DT)
    old = torch.ones(n, dtype=torch.long)           # a = 1
    new = torch.full((n,), 2, dtype=torch.long)     # b = 2
    changed = torch.ones(n, dtype=torch.bool)
    reward = torch.full((n,), -0.6, dtype=DT)
    out = update_temporal_memory(memory, old, new, changed, reward, decay=0.8)
    assert float(out[0, 2]) < 0, "M_ib did not decrease for a negative reward"
    assert float(out[0, 1]) > 0, "M_ia did not increase for a negative reward"


def test_memory_stays_clamped_to_pm_one():
    """§22: M is a bounded historical preference, not an unbounded accumulator."""
    n = 4
    memory = torch.zeros((n, 2), dtype=DT)
    old = torch.zeros(n, dtype=torch.long)
    new = torch.ones(n, dtype=torch.long)
    changed = torch.ones(n, dtype=torch.bool)
    for _ in range(200):
        memory = update_temporal_memory(memory, old, new, changed,
                                        torch.ones(n, dtype=DT) * 5.0, decay=0.5)
    assert float(memory.max()) <= 1.0 + 1.0e-12, f"memory exceeded +1: {float(memory.max())}"
    assert float(memory.min()) >= -1.0 - 1.0e-12, f"memory below -1: {float(memory.min())}"


def test_global_decay_is_applied_before_transition_updates():
    """§21: the optional decay runs first, so it never scales the fresh credit."""
    n = 2
    memory = torch.full((n, 2), 0.5, dtype=DT)
    old = torch.zeros(n, dtype=torch.long)
    new = torch.ones(n, dtype=torch.long)
    changed = torch.ones(n, dtype=torch.bool)
    reward = torch.full((n,), 1.0, dtype=DT)
    out = update_temporal_memory(memory, old, new, changed, reward,
                                 decay=0.8, global_decay=0.5)
    # 0.5 -> global decay -> 0.25 -> 0.8*0.25 + 0.2*1.0 = 0.4
    assert abs(float(out[0, 1]) - 0.4) < 1.0e-12, \
        f"global decay was not applied before the transition update: {float(out[0,1])}"
    untouched = update_temporal_memory(memory, old, new, torch.zeros(n, dtype=torch.bool),
                                       reward, decay=0.8, global_decay=0.5)
    assert abs(float(untouched[0, 1]) - 0.25) < 1.0e-12, \
        "global decay did not apply when nothing changed"


# ---------------------------------------------------------------------------
# F. Unchanged points receive no transition credit
# ---------------------------------------------------------------------------


def test_f_unchanged_gaussian_gets_no_transition_update():
    n = 10
    memory = torch.zeros((n, 2), dtype=DT)
    old = torch.zeros(n, dtype=torch.long)
    new = torch.zeros(n, dtype=torch.long)
    new[:3] = 1                                     # only the first three moved
    changed = new != old
    reward = torch.full((n,), 0.9, dtype=DT)
    out = update_temporal_memory(memory, old, new, changed, reward, decay=0.8)
    assert float(out[3:].abs().max()) == 0.0, "an unchanged Gaussian received transition credit"
    assert float(out[:3].abs().max()) > 0.0, "the changed Gaussians received nothing"

    # And the combined reward is zeroed outside C^(m) before it ever reaches memory.
    hat, _ = combine_reward(0.5, torch.full((n,), 0.5, dtype=DT), changed=changed)
    assert float(hat[3:].abs().max()) == 0.0, "reward leaked onto unchanged Gaussians"


def test_f_degenerate_self_transition_is_not_credited():
    """a == b is not a transition; +r and -r on one cell would silently cancel."""
    n = 4
    memory = torch.zeros((n, 2), dtype=DT)
    labels = torch.ones(n, dtype=torch.long)
    out = update_temporal_memory(memory, labels, labels.clone(),
                                 torch.ones(n, dtype=torch.bool),
                                 torch.full((n,), 0.7, dtype=DT), decay=0.8)
    assert float(out.abs().max()) == 0.0, "a self-transition altered the memory"


# ---------------------------------------------------------------------------
# G. Fixed score normalization
# ---------------------------------------------------------------------------


def test_g_normalization_constants_are_fixed_for_the_whole_run():
    n = 32
    local_render = torch.linspace(0.1, 2.0, n, dtype=DT)
    local_match = torch.linspace(0.2, 4.0, n, dtype=DT)
    valid = torch.ones(n, dtype=torch.bool)
    norm = establish_normalization(0.5, 0.8, local_render, local_match, valid, outer=0)
    snapshot = norm.as_dict()

    # Scoring later, very different states must not move the constants.
    for render, match in ((5.0, 9.0), (0.01, 0.02)):
        q = combine_global_score(render, match, match, 0.0, norm)
        assert abs(q.normalized_render - render / norm.c_render) < 1.0e-12
        assert abs(q.normalized_match - match / norm.c_match) < 1.0e-12
    assert norm.as_dict() == snapshot, "the normalisation constants drifted"

    # A cycle establishes them once and then keeps them.
    graph, _ = _graph(n)
    cycle = TemporalCreditCycle(n, 2)
    assert cycle.normalization is None
    cycle.normalization = norm
    cycle.observe(outer=0, labels=torch.zeros(n, dtype=torch.long), quality=_quality(1.0),
                  local=_local(torch.ones(n, dtype=DT)), graph=graph)
    assert cycle.normalization.as_dict() == snapshot, "observe() renormalised"


def test_g_score_difference_is_comparable_across_iterations():
    """§6: the same raw improvement always gives the same score drop.

    The equal-weight expectation is REPLACED by corrections §22: the score is now
    render-dominant (lambda_match defaults to 0.05), because at equal weight the
    final-matching terms could move against F1 and the combined score stopped
    tracking label quality. Comparability across iterations is unaffected -- the
    weights and the normalisation constants are both fixed for the run.
    """
    norm = ScoreNormalization(c_render=2.0, c_match=4.0, c_render_local=1.0, c_match_local=1.0)
    a = combine_global_score(2.0, 4.0, 4.0, 0.0, norm)
    b = combine_global_score(1.0, 4.0, 4.0, 0.0, norm)
    c = combine_global_score(2.0, 2.0, 2.0, 0.0, norm)
    assert abs((a.total - b.total) - 0.5) < 1.0e-12, "render halving did not drop S by 0.5"
    assert abs((a.total - c.total) - 0.5 * 0.05) < 1.0e-12, \
        f"match halving should drop S by 0.025 at the default weight, got {a.total - c.total}"
    # Explicitly: render moves the score 20x more than match.
    assert abs((a.total - b.total) - 20.0 * (a.total - c.total)) < 1.0e-9, \
        "the score is not render-dominant by the expected 20x"
    # And an equal-weight score is still reachable for the §23 ablation, where
    # halving the match term does move S by the full 0.5 again.
    equal_a = combine_global_score(2.0, 4.0, 4.0, 0.0, norm, lambda_match=1.0)
    equal_c = combine_global_score(2.0, 2.0, 2.0, 0.0, norm, lambda_match=1.0)
    assert abs((equal_a.total - equal_c.total) - 0.5) < 1.0e-12, \
        "the §23 equal-weight ablation setting is not reachable"


# ---------------------------------------------------------------------------
# H. Local neighbourhood attribution
# ---------------------------------------------------------------------------


def test_h_improving_region_outscores_a_distant_unchanged_one():
    n, k = 120, 6
    graph, _ = _graph(n, spread=1.0, k=k)
    previous = torch.ones(n, dtype=DT)
    new = torch.ones(n, dtype=DT)
    new[:20] = 0.2                                   # one end improves a lot

    prev_local = _local(previous, graph)
    new_local = _local(new, graph)
    reward = local_transition_reward(prev_local.smoothed, new_local.smoothed)

    near = reward[:15].mean()
    far = reward[-15:].mean()
    assert float(near) > 0.3, f"the improved region got only {float(near):.4f}"
    assert abs(float(far)) < 1.0e-6, f"a distant unchanged region got {float(far):.4f}"
    assert float(near) > float(far), "local attribution did not localise"


def test_h_neighborhood_average_uses_spatial_weights_and_keeps_isolated_points():
    graph, _ = _graph(16, k=4)
    values = torch.zeros(16, dtype=DT)
    values[8] = 1.0
    smoothed = neighborhood_average(graph, values)
    assert float(smoothed[8]) < 1.0, "the self term did not mix with neighbours"
    assert float(smoothed[8]) > float(smoothed[0]), "the spike did not stay local"
    assert float(smoothed[7]) > 0.0 and float(smoothed[9]) > 0.0, "neighbours got nothing"

    # An isolated node keeps its own value rather than collapsing to zero.
    empty = build_neighbor_graph(torch.zeros((1, 3), dtype=DT), k=4)
    alone = float(neighborhood_average(empty, torch.tensor([0.7], dtype=DT))[0])
    assert abs(alone - 0.7) < 1.0e-7, f"an isolated node collapsed to {alone}"


def test_local_score_falls_back_to_render_when_match_is_invalid():
    """§14: an unsupported candidate is never given zero cost."""
    n = 6
    graph, _ = _graph(n, k=2)
    norm = ScoreNormalization(c_render=1.0, c_match=1.0, c_render_local=1.0, c_match_local=1.0)
    render = torch.full((n,), 0.5, dtype=DT)
    match = torch.full((n,), 99.0, dtype=DT)          # would dominate if it counted
    valid = torch.zeros(n, dtype=torch.bool)
    local = combine_local_score(render, match, valid, torch.ones(n, dtype=DT), norm, graph)
    assert torch.allclose(local.total, torch.full((n,), 1.0, dtype=DT)), \
        f"invalid match did not fall back to rescaled render: {local.total[:2].tolist()}"
    assert float(local.total.min()) > 0.0, "an unsupported candidate was given zero cost"


# ---------------------------------------------------------------------------
# I. Memory biases the next candidate choice
# ---------------------------------------------------------------------------


def test_i_memory_breaks_a_near_tie():
    energy = torch.tensor([[1.000, 1.001]], dtype=DT)   # class 0 barely wins
    assert int(energy.argmin(dim=1)) == 0
    memory = torch.tensor([[0.0, 0.9]], dtype=DT)       # history favours class 1
    biased, stats = apply_temporal_memory(energy, memory, lambda_memory=0.25)
    assert int(biased.argmin(dim=1)) == 1, \
        f"memory did not break the tie: {biased.tolist()}"
    assert stats["num_preference_flips"] == 1.0
    # And it is a bias on the score, not a replacement of it.
    assert abs(float(biased[0, 0]) - 1.000) < 1.0e-12, "the unbiased class was altered"


def test_i_zero_memory_is_a_no_op():
    energy = torch.rand((20, 3), dtype=DT)
    biased, stats = apply_temporal_memory(energy, torch.zeros((20, 3), dtype=DT))
    assert torch.equal(biased, energy), "zero memory changed the candidate score"
    assert stats["num_preference_flips"] == 0.0


# ---------------------------------------------------------------------------
# J. Strong current evidence dominates memory
# ---------------------------------------------------------------------------


def test_j_large_render_margin_cannot_be_overturned_by_stale_memory():
    # Class 0 wins by a huge margin; memory strongly favours class 1.
    energy = torch.tensor([[0.0, 10.0]], dtype=DT)
    memory = torch.tensor([[-1.0, 1.0]], dtype=DT)
    margin = torch.tensor([10.0], dtype=DT)
    biased, stats = apply_temporal_memory(energy, memory, raw_margin=margin,
                                          lambda_memory=0.25, margin_cap=0.5)
    assert int(biased.argmin(dim=1)) == 0, \
        f"stale memory overturned overwhelming current evidence: {biased.tolist()}"
    # The cap is |lambda M| <= rho * margin = 0.5 * 10 = 5, and 0.25 is far under it,
    # so this case is decided by the margin itself -- check the cap does bind
    # when the memory bias would otherwise be large.
    big, big_stats = apply_temporal_memory(
        energy, memory, raw_margin=torch.tensor([0.2], dtype=DT),
        lambda_memory=4.0, margin_cap=0.5)
    assert big_stats["fraction_capped"] > 0.0, "the margin cap never bound"
    assert float(big[0, 1]) - float(big[0, 0]) > 0.0, \
        "the cap failed to protect the current winner"
    assert int(big.argmin(dim=1)) == 0, "a capped bias still flipped a clear winner"


def test_j_cap_allows_at_most_a_tie_above_the_floor():
    """The exact reach of §24's cap, which is stronger than it first reads.

    The cap is per class: |lambda M_ik| <= rho * max(m_i, m_floor). With maximal
    opposing memory the two classes can therefore move toward each other by at
    most 2 * rho * m_i = m_i when rho = 0.5 -- exactly closing the gap and never
    crossing it. So above ``m_floor`` memory can equalise a decision but cannot
    strictly flip the argmin; its remaining influence reaches the render graph
    through the seed margin and the propagation preference, not the argmin.

    Below ``m_floor`` the floor exceeds the real margin, and there -- where the
    evidence is genuinely indistinguishable -- memory does decide.
    """
    memory = torch.tensor([[-1.0, 1.0]], dtype=DT)          # maximal, opposing
    for margin, may_flip in ((1.0e-5, True), (1.0e-4, True),
                             (1.0e-3, False), (1.0e-2, False), (1.0e-1, False)):
        energy = torch.tensor([[1.0, 1.0 + margin]], dtype=DT)
        biased, _ = apply_temporal_memory(
            energy, memory, raw_margin=torch.tensor([margin], dtype=DT),
            lambda_memory=0.25, margin_cap=0.5)
        gap = float(biased[0, 1] - biased[0, 0])
        flipped = int(biased.argmin(dim=1)) == 1
        assert flipped == may_flip, (
            f"margin {margin}: flipped={flipped}, expected {may_flip} (gap {gap:+.8f})")
        if not may_flip:
            assert gap >= -1.0e-12, f"margin {margin}: the cap was breached (gap {gap:+.8f})"


# ---------------------------------------------------------------------------
# K. K-part permutation alignment
# ---------------------------------------------------------------------------


def test_k_pure_permutation_produces_zero_label_changes():
    n = 300
    generator = torch.Generator().manual_seed(11)
    previous = torch.randint(0, 3, (n,), generator=generator)
    swap = {0: 0, 1: 2, 2: 1}                       # rename part 1 <-> part 2
    renamed = torch.tensor([swap[int(v)] for v in previous])
    aligned, permutation, permuted = align_part_permutation(previous, renamed, 3)
    assert permuted is True, "the permutation was not detected"
    assert int((aligned != previous).sum()) == 0, \
        f"a pure renaming produced {int((aligned != previous).sum())} false transitions"
    assert permutation[1] == 2 and permutation[2] == 1, f"wrong permutation {permutation}"
    assert permutation[0] == 0, "static class 0 was permuted"


def test_k_real_changes_survive_alignment():
    n = 300
    generator = torch.Generator().manual_seed(12)
    previous = torch.randint(0, 3, (n,), generator=generator)
    new = previous.clone()
    new[:40] = (new[:40] + 1) % 3                    # genuine reassignments
    aligned, _, _ = align_part_permutation(previous, new, 3)
    changed = int((aligned != previous).sum())
    assert changed > 0, "alignment erased genuine label changes"


def test_k_k1_needs_no_permutation():
    previous = torch.tensor([0, 1, 1, 0])
    new = torch.tensor([1, 0, 1, 0])
    aligned, permutation, permuted = align_part_permutation(previous, new, 2)
    assert permuted is False, "K=1 reported a permutation"
    assert torch.equal(aligned, new), "K=1 labels were remapped"
    assert permutation == [0, 1]


# ---------------------------------------------------------------------------
# L. No full rollback
# ---------------------------------------------------------------------------


def test_l_a_worse_score_updates_memory_instead_of_reverting():
    n = 64
    graph, _ = _graph(n)
    cycle = TemporalCreditCycle(n, 2)
    z0 = torch.zeros(n, dtype=torch.long)
    cycle.observe(outer=0, labels=z0, quality=_quality(1.0),
                  local=_local(torch.ones(n, dtype=DT)), graph=graph)
    z1 = z0.clone()
    z1[:12] = 1
    local_new = torch.ones(n, dtype=DT)
    local_new[:12] = 3.0                              # this region got much worse
    record = cycle.observe(outer=1, labels=z1, quality=_quality(2.0),
                           local=_local(local_new), graph=graph)

    assert record["relative_global_reward"] < 0, "a worse state did not produce a negative reward"
    # The labels advance; they are NOT reverted (§38).
    assert torch.equal(cycle.previous_labels, z1), \
        "the cycle reverted the label update instead of learning from it"
    assert float(cycle.memory[0, 1]) < 0, "the harmful transition was not recorded in memory"
    assert float(cycle.memory[0, 0]) > 0, "the old class was not reinforced"
    # And the recorded memory then biases the next proposal against class 1.
    energy = torch.tensor([[1.0, 1.0]], dtype=DT)
    biased, _ = apply_temporal_memory(energy, cycle.memory[:1].to(DT), lambda_memory=0.25)
    assert int(biased.argmin(dim=1)) == 0, "memory did not steer the next decision away"


# ---------------------------------------------------------------------------
# Sections 32, 34, 26, 37: components, oscillation, credit loss, report
# ---------------------------------------------------------------------------


def test_section32_changed_components_are_counted():
    n = 60
    graph, _ = _graph(n, k=2)
    changed = torch.zeros(n, dtype=torch.bool)
    changed[5:12] = True                              # one run
    changed[40:47] = True                             # a second, far away
    cid, count = changed_components(graph, changed)
    assert count == 2, f"expected 2 changed components, got {count}"
    assert int(cid[changed].min()) >= 0 and int(cid[~changed].max()) == -1, \
        "component ids leaked onto unchanged Gaussians"
    assert len(set(cid[5:12].tolist())) == 1, "a contiguous run was split"
    assert cid[5] != cid[40], "two separated runs were merged"
    assert changed_components(graph, torch.zeros(n, dtype=torch.bool))[1] == 0


def test_section33_component_reward_is_off_by_default_and_works_when_on():
    n = 20
    local = torch.zeros(n, dtype=DT)
    local[:10] = 1.0
    changed = torch.ones(n, dtype=torch.bool)
    cid = torch.cat((torch.zeros(10, dtype=torch.long), torch.ones(10, dtype=torch.long)))

    off, comp_off = combine_reward(0.0, local, beta_global=0.0, beta_component=0.0,
                                   component_id=cid, changed=changed)
    assert torch.allclose(off, local), "beta_component=0 did not reduce to §18"
    assert float(comp_off.abs().max()) == 0.0

    on, comp_on = combine_reward(0.0, local, beta_global=0.0, beta_component=1.0,
                                 component_id=cid, changed=changed)
    assert abs(float(comp_on[0]) - 1.0) < 1.0e-12, "component mean is wrong"
    assert abs(float(comp_on[15]) - 0.0) < 1.0e-12
    assert torch.allclose(on, comp_on), "beta_component=1 did not use the component reward"

    try:
        combine_reward(0.0, local, beta_global=0.7, beta_component=0.7)
    except ValueError:
        pass
    else:
        raise AssertionError("beta_global + beta_component > 1 was accepted")


def test_section34_oscillation_detection():
    two_back = torch.tensor([0, 1, 0, 2])
    one_back = torch.tensor([1, 0, 0, 1])
    now = torch.tensor([0, 1, 0, 1])
    osc = detect_label_oscillation(two_back, one_back, now)
    assert bool(osc[0]) and bool(osc[1]), "a -> b -> a was not detected"
    assert not bool(osc[2]), "an unchanged label was called an oscillation"
    assert not bool(osc[3]), "a b -> c move was called an oscillation"
    # With no two-back history nothing can oscillate yet.
    assert not bool(detect_label_oscillation(None, one_back, now).any())


def test_section26_credit_loss_reinforces_the_right_class():
    p = torch.tensor([[0.5, 0.5], [0.5, 0.5]], dtype=DT, requires_grad=True)
    old = torch.tensor([0, 0])
    new = torch.tensor([1, 1])
    changed = torch.ones(2, dtype=torch.bool)

    good = temporal_credit_loss(p, old, new, changed, torch.tensor([1.0, 1.0], dtype=DT))
    good.backward()
    # A positive reward must push probability mass toward the NEW class.
    assert float(p.grad[0, 1]) < 0 < float(p.grad[0, 0]), \
        "a positive reward did not reinforce the new label"

    p.grad = None
    bad = temporal_credit_loss(p, old, new, changed, torch.tensor([-1.0, -1.0], dtype=DT))
    bad.backward()
    assert float(p.grad[0, 0]) < 0 < float(p.grad[0, 1]), \
        "a negative reward did not reinforce the old label"

    # Nothing changed -> no loss at all.
    zero = temporal_credit_loss(p.detach(), old, old, torch.zeros(2, dtype=torch.bool),
                                torch.ones(2, dtype=DT))
    assert float(zero) == 0.0


def test_section37_transition_report_splits_by_type():
    n = 40
    old = torch.zeros(n, dtype=torch.long)
    new = torch.zeros(n, dtype=torch.long)
    new[:10] = 1                                     # 0 -> 1, helpful
    old[20:30] = 1                                   # 1 -> 0, harmful
    changed = new != old
    reward = torch.zeros(n, dtype=DT)
    reward[:10] = 0.5
    reward[20:30] = -0.5
    prev = torch.ones(n, dtype=DT)
    now = torch.ones(n, dtype=DT)
    now[:10] = 0.5
    now[20:30] = 1.5

    rows = transition_type_report(old, new, changed, reward, prev, now, prev, now, prev, now)
    kinds = {row["transition"]: row for row in rows}
    assert set(kinds) == {"0->1", "1->0"}, f"unexpected transition types {set(kinds)}"
    assert kinds["0->1"]["count"] == 10 and kinds["1->0"]["count"] == 10
    assert kinds["0->1"]["mean_reward"] > 0 > kinds["1->0"]["mean_reward"]
    assert kinds["0->1"]["fraction_positive_reward"] == 1.0
    assert kinds["0->1"]["mean_render_improvement"] > 0, "improvement sign is inverted"
    assert kinds["1->0"]["mean_render_improvement"] < 0
    assert transition_type_report(old, old, torch.zeros(n, dtype=torch.bool),
                                 reward, prev, now, prev, now, prev, now) == []


def test_section36_artifact_has_every_required_field():
    n = 64
    graph, _ = _graph(n)
    cycle = TemporalCreditCycle(n, 2)
    z0 = torch.zeros(n, dtype=torch.long)
    cycle.observe(outer=0, labels=z0, quality=_quality(1.0),
                  local=_local(torch.ones(n, dtype=DT)), graph=graph)
    assert cycle.artifact(torch.zeros((n, 2)), torch.zeros((n, 2))) == {}, \
        "the baseline produced a transition artifact"

    z1 = z0.clone()
    z1[:10] = 1
    cycle.observe(outer=1, labels=z1, quality=_quality(0.5),
                  local=_local(torch.full((n,), 0.5, dtype=DT)), graph=graph)
    energy = torch.rand((n, 2), dtype=DT)
    payload = cycle.artifact(energy, energy - 0.1)
    for key in ("old_label", "new_label", "changed_mask", "local_score_old", "local_score_new",
                "local_reward", "final_reward", "memory_before", "memory_after",
                "render_energy", "memory_biased_energy", "label_oscillation_mask",
                "changed_connected_component_id"):
        assert key in payload, f"§36 artifact is missing {key}"
    assert tuple(payload["memory_before"].shape) == (n, 2)
    assert not torch.equal(payload["memory_before"], payload["memory_after"]), \
        "memory_before and memory_after are identical after a credited transition"


def test_section35_history_has_every_required_column():
    n = 32
    graph, _ = _graph(n)
    cycle = TemporalCreditCycle(n, 2)
    z = torch.zeros(n, dtype=torch.long)
    cycle.observe(outer=0, labels=z, quality=_quality(1.0),
                  local=_local(torch.ones(n, dtype=DT)), graph=graph)
    z1 = z.clone()
    z1[:5] = 1
    cycle.observe(outer=1, labels=z1, quality=_quality(0.8),
                  local=_local(torch.full((n,), 0.9, dtype=DT)), graph=graph)
    for row in cycle.history:
        for column in ("outer", "score_total", "score_render", "score_match",
                       "relative_global_reward", "num_changed", "changed_fraction",
                       "mean_local_reward_changed", "median_local_reward_changed",
                       "num_positive_reward_changes", "num_negative_reward_changes",
                       "mean_abs_memory", "max_abs_memory", "num_label_oscillations",
                       "num_static", "num_part_0", "num_part_1"):
            assert column in row, f"§35 column {column} is missing"
    assert len(cycle.history) == 2


def test_match_score_marks_unsupported_candidates_invalid():
    energy = torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=DT)
    views = torch.tensor([[5.0, 5.0], [0.0, 0.0]], dtype=DT)
    labels = torch.tensor([0, 1])
    total, pixel, nn, selected, valid = evaluate_match_score(
        pixel_energy=energy, pixel_num_views=views, final_nn_energy=None,
        labels=labels, min_support_views=2)
    assert bool(valid[0]) and not bool(valid[1]), "support validity is wrong"
    assert abs(float(selected[0]) - 1.0) < 1.0e-12, "the selected class energy is wrong"
    assert abs(float(selected[1]) - 4.0) < 1.0e-12
    assert abs(pixel - 1.0) < 1.0e-12, "the unsupported Gaussian polluted the global score"
    assert nn == 0.0 and abs(total - pixel) < 1.0e-12


def test_section30_evaluated_population_is_identical_every_iteration():
    """The set of SUPPORTED Gaussians shifts as T_k(t) changes; the score must not.

    Measured on a real USB run: the moving class went 7856 -> 6798 valid between
    two outer iterations. Averaging over the valid set alone would make part of
    S^(m) - S^(m-1) a change of population rather than of quality, which is
    exactly what §30 exists to prevent.
    """
    n = 100
    energy = torch.full((n, 2), 1.0, dtype=DT)
    labels = torch.ones(n, dtype=torch.long)

    # Same solution quality at both iterations; only the support coverage moved.
    views_a = torch.full((n, 2), 5.0, dtype=DT)
    views_b = views_a.clone()
    views_b[:30, 1] = 0.0                                 # 30 lost their support

    penalty = unsupported_penalty_from(
        torch.full((n,), 1.0, dtype=DT), torch.ones(n, dtype=torch.bool))

    _, without_a, _, sel, valid_a = evaluate_match_score(
        pixel_energy=energy, pixel_num_views=views_a, final_nn_energy=None, labels=labels)
    _, without_b, _, _, valid_b = evaluate_match_score(
        pixel_energy=energy, pixel_num_views=views_b, final_nn_energy=None, labels=labels)
    assert int(valid_a.sum()) != int(valid_b.sum()), "the fixture did not move the population"

    _, with_a, _, _, _ = evaluate_match_score(
        pixel_energy=energy, pixel_num_views=views_a, final_nn_energy=None,
        labels=labels, unsupported_penalty=penalty)
    _, with_b, _, _, _ = evaluate_match_score(
        pixel_energy=energy, pixel_num_views=views_b, final_nn_energy=None,
        labels=labels, unsupported_penalty=penalty)

    # Here every energy is 1.0 and the penalty is also 1.0, so a score computed
    # over the FIXED population of N is unchanged by the coverage shift.
    assert abs(with_a - with_b) < 1.0e-12, (
        f"the fixed-population score still moved with coverage: {with_a} vs {with_b}")
    assert abs(without_a - without_b) < 1.0e-12, "sanity: equal energies give equal means"

    # And an unsupported candidate is never cheaper than a supported one (§14).
    expensive = unsupported_penalty_from(torch.linspace(0.0, 10.0, n, dtype=DT),
                                         torch.ones(n, dtype=torch.bool))
    assert expensive > 0.0, "the penalty collapsed to zero cost"
    _, costly, _, _, _ = evaluate_match_score(
        pixel_energy=torch.zeros((n, 2), dtype=DT), pixel_num_views=views_b,
        final_nn_energy=None, labels=labels, unsupported_penalty=expensive)
    assert costly > 0.0, "unsupported candidates were given zero cost"


def test_section30_penalty_is_fixed_once_and_carried_forward():
    norm = establish_normalization(
        1.0, 2.0, torch.ones(8, dtype=DT), torch.ones(8, dtype=DT),
        torch.ones(8, dtype=torch.bool), outer=0, unsupported_penalty=3.25)
    assert norm.unsupported_penalty == 3.25, "the penalty was not stored"
    assert "unsupported_penalty" in norm.as_dict(), "the penalty is not reported with the constants"
    snapshot = norm.as_dict()
    combine_global_score(9.0, 9.0, 9.0, 0.0, norm)
    assert norm.as_dict() == snapshot, "scoring mutated the fixed constants"


def test_cli_defaults_match_the_spec():
    """Section 41: the flags exist with exactly the documented defaults."""
    from argparse import ArgumentParser

    from arguments import ModelParams, OptimizationParams, PipelineParams, add_dynamic_args

    parser = ArgumentParser()
    ModelParams(parser)
    OptimizationParams(parser)
    PipelineParams(parser)
    add_dynamic_args(parser)
    args = parser.parse_args([])
    for flag, expected in (
        ("lambda_score_render", 1.0),
        ("lambda_score_match", 1.0),
        ("lambda_score_final_nn", 0.1),
        ("temporal_credit_graph_k", 16),
        ("temporal_global_reward_weight", 0.25),
        ("temporal_reward_clip", 1.0),
        ("temporal_memory_decay", 0.8),
        ("temporal_memory_global_decay", 1.0),
        ("lambda_temporal_memory", 0.25),
        ("temporal_memory_margin_cap", 0.5),
        ("temporal_component_reward_weight", 0.0),
        ("enable_temporal_credit", False),
    ):
        assert hasattr(args, flag), f"--{flag} is not registered"
        assert getattr(args, flag) == expected, \
            f"--{flag} default is {getattr(args, flag)}, spec says {expected}"


def test_temporal_credit_requires_render_graph_and_pixel_matching():
    from argparse import ArgumentParser

    from arguments import (
        ModelParams, OptimizationParams, PipelineParams,
        add_dynamic_args, apply_dynamic_defaults,
    )

    def _args(extra):
        parser = ArgumentParser()
        ModelParams(parser)
        OptimizationParams(parser)
        PipelineParams(parser)
        add_dynamic_args(parser)
        return parser.parse_args(["--enable_temporal_credit", *extra])

    for extra, missing in ((["--outer_label_mode", "learned", "--enable_pixel_matching"],
                            "render_graph"),
                           (["--outer_label_mode", "render_graph"], "pixel")):
        try:
            apply_dynamic_defaults(_args(extra))
        except ValueError as error:
            assert missing in str(error), f"wrong error for {extra}: {error}"
        else:
            raise AssertionError(f"a temporal-credit run missing {missing} was accepted")

    ok = apply_dynamic_defaults(_args(["--outer_label_mode", "render_graph",
                                       "--enable_pixel_matching"]))
    assert ok.enable_temporal_credit is True


if __name__ == "__main__":
    module = sys.modules["__main__"]
    tests = [
        (name, fn)
        for name, fn in sorted(vars(module).items())
        if name.startswith("test_") and inspect.isfunction(fn)
    ]
    for name, fn in tests:
        fn()
        print(f"  ok  {name}")
    print(f"PASS temporal_credit_outer_cycle ({len(tests)} tests)")
