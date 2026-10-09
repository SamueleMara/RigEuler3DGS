"""Soft label outer cycle: required tests A-S (sections 47-50).

Plus the invariants of section 54 and the section 12/43 prohibitions, which are
the ones whose violation would be silent.
"""

from __future__ import annotations

import inspect
import sys
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from rigid_parts.endpoint_contrastive_init import (
    build_pseudo_anchors,
    compute_endpoint_matching,
    endpoint_edge_similarity,
    run_soft_endpoint_contrastive_initialization,
)
from rigid_parts.neighborhood import build_neighbor_graph
from rigid_parts.soft_labels import (
    TemporalOrderGuard,
    apply_memory_in_logit_space,
    assert_no_memory_in_initialization,
    binary_entropy,
    clamp_trusted_static,
    combine_render_and_spatial,
    compose_soft_initial_state,
    pair_confidence,
    pairwise_contrastive_loss,
    polarization_loss,
    render_alpha,
    render_probabilities,
    soft_oscillation,
    soft_outer_update,
    soft_spatial_clustering,
    spatial_probability_propagation,
    train_soft_binary_labels,
)

DT = torch.float64
SEED = 20260919


def _two_blocks(n_static=120, n_dynamic=80, shift=0.6):
    g = torch.Generator().manual_seed(SEED)
    static = torch.rand((n_static, 3), generator=g) * torch.tensor([1.0, 1.0, 0.2])
    dynamic = torch.rand((n_dynamic, 3), generator=g) * torch.tensor([1.0, 1.0, 0.2])
    dynamic[:, 2] += 1.0
    x0 = torch.cat((static, dynamic), 0)
    yT = torch.cat((static, dynamic + torch.tensor([0.0, 0.0, float(shift)])), 0)
    gt = torch.cat((torch.zeros(n_static, dtype=torch.long), torch.ones(n_dynamic, dtype=torch.long)))
    return x0, yT, gt


def _init_pieces(x0, yT, trusted=None, dyn_q=0.95):
    m = compute_endpoint_matching(x0, yT)
    graph = build_neighbor_graph(x0, k=8)
    anchors = build_pseudo_anchors(m.normalized, trusted,
                                   static_quantile=0.10, dynamic_quantile=1.0 - dyn_q)
    similarity, _ = endpoint_edge_similarity(graph, m.normalized)
    return m, graph, anchors, similarity


# ---------------------------------------------------------------------------
# A, B, G, H -- initialization output shape and softness
# ---------------------------------------------------------------------------


def test_a_initialization_returns_normalised_probabilities():
    x0, yT, _ = _two_blocks()
    for k in (1, 2, 3):
        r = run_soft_endpoint_contrastive_initialization(x0, yT, k, log_fn=lambda *_: None)
        p = r.part_probs
        assert tuple(p.shape) == (x0.shape[0], k + 1), f"K={k}: shape {tuple(p.shape)}"
        total = p.sum(dim=1)
        assert torch.allclose(total, torch.ones_like(total), atol=1.0e-6), \
            f"K={k}: rows do not sum to 1 (max err {float((total-1).abs().max()):.2e})"
        assert bool((p >= 0).all()), f"K={k}: negative probability"


def test_b_no_persistent_hard_threshold_is_applied():
    """Section 13/43: nothing is thresholded and nothing is one-hot by construction."""
    x0, yT, _ = _two_blocks()
    r = run_soft_endpoint_contrastive_initialization(x0, yT, 1, log_fn=lambda *_: None)
    p = r.part_probs
    one_hot = ((p > 0.999999) | (p < 1.0e-6)).all(dim=1)
    assert not bool(one_hot.all()), "every row is one-hot; a hard threshold was applied"
    assert float(p[:, 1].min()) > 0.0 and float(p[:, 1].max()) < 1.0, \
        "q_dynamic saturated to exactly {0,1}"
    # A 0.5-thresholded state would put every value at exactly 0 or 1.
    assert float(((p > 0.01) & (p < 0.99)).any(dim=1).float().mean()) > 0.0, \
        "no Gaussian retained an intermediate probability"


def test_g_k1_output_is_exactly_q():
    x0, yT, _ = _two_blocks()
    r = run_soft_endpoint_contrastive_initialization(x0, yT, 1, log_fn=lambda *_: None)
    assert torch.allclose(r.part_probs, r.binary.q), "K=1 did not pass [q^S, q^D] straight through"
    assert r.pi is None, "K=1 ran the soft clustering"


def test_h_dynamic_mass_is_preserved_by_the_clustering():
    """Section 16: sum_{k>=1} P_ik == q^D exactly."""
    x0, yT, _ = _two_blocks()
    for k in (2, 3, 4):
        r = run_soft_endpoint_contrastive_initialization(x0, yT, k, log_fn=lambda *_: None)
        moving_mass = r.part_probs[:, 1:].sum(dim=1)
        assert torch.allclose(moving_mass, r.binary.q_dynamic, atol=1.0e-6), \
            f"K={k}: moving mass != q^D (max err {float((moving_mass-r.binary.q_dynamic).abs().max()):.2e})"
        assert torch.allclose(r.part_probs[:, 0], r.binary.q_static, atol=1.0e-6), \
            f"K={k}: P_i0 != q^S"
        assert r.pi is not None and torch.allclose(
            r.pi.sum(dim=1), torch.ones(x0.shape[0]), atol=1.0e-6), "pi rows do not sum to 1"


# ---------------------------------------------------------------------------
# C -- dynamic anchors are the top 5%
# ---------------------------------------------------------------------------


def test_c_dynamic_anchors_are_the_top_five_percent():
    d = torch.linspace(0.0, 1.0, 1001)
    anchors = build_pseudo_anchors(d, None, static_quantile=0.10, dynamic_quantile=1.0 - 0.95)
    fraction = float(anchors.dynamic.float().mean())
    assert abs(fraction - 0.05) < 0.01, f"dynamic anchors cover {fraction*100:.2f}%, expected ~5%"
    assert float(anchors.tau_dynamic) > 0.94, f"tau_D = {float(anchors.tau_dynamic)}, expected Q_0.95"
    assert bool(anchors.dynamic[-1]) and not bool(anchors.dynamic[0]), "wrong tail"
    # And the default really is 0.95, not the old 0.10-as-top-10% reading.
    ten = build_pseudo_anchors(d, None, dynamic_quantile=0.10)
    assert float(ten.dynamic.float().mean()) > fraction, \
        "the 5% setting is not stricter than the old 10% one"


# ---------------------------------------------------------------------------
# D -- polarization actually reduces entropy from the neutral start
# ---------------------------------------------------------------------------


def test_d_training_reduces_entropy_from_the_neutral_initialization():
    x0, yT, _ = _two_blocks()
    _, graph, anchors, similarity = _init_pieces(x0, yT)
    n = int(x0.shape[0])

    neutral = torch.full((n, 2), 0.5)
    start_entropy = float(binary_entropy(neutral).mean())
    assert abs(start_entropy - torch.log(torch.tensor(2.0)).item()) < 1.0e-6, \
        "the neutral start is not [0.5, 0.5]"

    r = train_soft_binary_labels(
        graph, similarity, anchors.trusted_static, anchors.geometric_static,
        anchors.dynamic, num_gaussians=n, iterations=300, lambda_polar=0.5)
    end_entropy = r.summary["mean_entropy"]
    assert end_entropy < start_entropy, \
        f"entropy did not fall from the neutral start: {start_entropy:.5f} -> {end_entropy:.5f}"
    assert r.summary["fraction_peak_gt_090"] > 0.0, "nothing became strongly polarized"

    # The polarization term is what does it: without it entropy falls far less.
    without = train_soft_binary_labels(
        graph, similarity, anchors.trusted_static, anchors.geometric_static,
        anchors.dynamic, num_gaussians=n, iterations=300, lambda_polar=0.0)
    assert end_entropy < without.summary["mean_entropy"], (
        f"lambda_polar had no effect: {end_entropy:.5f} with vs "
        f"{without.summary['mean_entropy']:.5f} without")


def test_section12_logits_start_at_exactly_zero():
    """Soft-labels §12 with NO endpoint distance: still exactly neutral.

    Corrections §8 supersedes the all-zero start when an endpoint distance is
    supplied -- see test_corrections_c_endpoint_logit_bias_is_weak -- but the
    no-distance path must stay neutral, and the forbidden rank-percentile
    PROBABILITY seeding must not come back either way.
    """
    x0, yT, _ = _two_blocks()
    _, graph, anchors, similarity = _init_pieces(x0, yT)
    r = train_soft_binary_labels(
        graph, similarity, anchors.trusted_static, anchors.geometric_static,
        anchors.dynamic, num_gaussians=int(x0.shape[0]), iterations=0)
    assert torch.allclose(r.logits, torch.zeros_like(r.logits)), "logits did not start at zero"
    assert torch.allclose(r.q, torch.full_like(r.q, 0.5)), "q did not start at [0.5, 0.5]"


def test_section9_superseded_signed_confidence_is_no_longer_used():
    """The old |2s-1|^gamma gave a DISSIMILAR pair maximal weight, which is the
    repulsion corrections §3 rules out. Kept only for this comparison."""
    from rigid_parts.soft_labels import positive_similarity_confidence

    s_vals = torch.tensor([0.0, 0.5, 1.0], dtype=DT)
    old = pair_confidence(s_vals, gamma=2.0)
    new = positive_similarity_confidence(s_vals, gamma=2.0)
    assert float(old[0]) == 1.0, "sanity: the old rule weighted s=0 maximally"
    assert float(new[0]) == 0.0, "the positive-only rule still weights s=0"
    assert float(new[2]) == 1.0, "the positive-only rule dropped s=1"
    assert float(new[1]) < float(new[2]), "confidence is not monotone in s"


def _legacy_test_section9_ambiguous_pairs_are_downweighted():
    s = torch.tensor([0.0, 0.25, 0.5, 0.75, 1.0], dtype=DT)
    c = pair_confidence(s, gamma=2.0)
    assert float(c[2]) == 0.0, "a pair at s=0.5 still carries confidence"
    assert float(c[0]) == 1.0 and float(c[4]) == 1.0, "extreme pairs lost confidence"
    assert float(c[1]) < float(c[0]), "confidence is not monotone away from 0.5"


# ---------------------------------------------------------------------------
# E -- similar-distance neighbours converge in soft label space
# ---------------------------------------------------------------------------


def test_e_similar_neighbours_become_similar_in_soft_space():
    x0, yT, _ = _two_blocks()
    _, graph, anchors, similarity = _init_pieces(x0, yT)
    n = int(x0.shape[0])

    m, _, _, _ = _init_pieces(x0, yT)
    r = train_soft_binary_labels(
        graph, similarity, anchors.trusted_static, anchors.geometric_static,
        anchors.dynamic, num_gaussians=n, normalized_distance=m.normalized, iterations=300)

    similar = similarity > 0.9
    dissimilar = similarity < 0.5
    assert bool(similar.any()) and bool(dissimilar.any()), "fixture lacks both edge kinds"

    def _gap(q, mask):
        return float((q[graph.source[mask]] - q[graph.target[mask]]).abs().sum(dim=1).mean())

    # The neutral start has gap exactly 0 everywhere, so "became more similar"
    # can only be read relative to the dissimilar edges and to a control without
    # the pairwise term -- a uniform state is trivially the most similar of all.
    gap_similar = _gap(r.q, similar)
    gap_dissimilar = _gap(r.q, dissimilar)
    assert gap_dissimilar > gap_similar, (
        f"similar neighbours are not closer than dissimilar ones: "
        f"{gap_similar:.5f} vs {gap_dissimilar:.5f}")

    # REPLACED BY corrections §3-§5. The superseded assertion required the
    # objective to SEPARATE dissimilar edges, which is precisely the repulsion
    # the corrections remove: for one rigid rotating part, two neighbours on the
    # same part can have very different endpoint distances, so a large |d_i-d_j|
    # is not evidence of different classes. The positive-only objective must
    # therefore leave dissimilar pairs unconstrained, never push them apart.
    control = train_soft_binary_labels(
        graph, similarity, anchors.trusted_static, anchors.geometric_static,
        anchors.dynamic, num_gaussians=n, iterations=300, lambda_smooth=0.0)
    assert gap_dissimilar <= _gap(control.q, dissimilar) + 1.0e-6, (
        f"the smoothness term INCREASED the dissimilar-edge gap "
        f"({gap_dissimilar:.5f} with vs {_gap(control.q, dissimilar):.5f} without); "
        "that is repulsion, which corrections §3 forbids")
    assert gap_similar < 0.05, f"similar edges are not tight: {gap_similar:.5f}"


# ---------------------------------------------------------------------------
# F -- trusted static anchors
# ---------------------------------------------------------------------------


def test_f_trusted_static_anchors_are_clamped_to_one_hot_static():
    p = torch.tensor([[0.2, 0.8], [0.6, 0.4], [0.5, 0.5]], dtype=DT)
    mask = torch.tensor([True, False, False])
    out = clamp_trusted_static(p, mask)
    assert torch.allclose(out[0], torch.tensor([1.0, 0.0], dtype=DT)), "anchor not clamped"
    assert torch.allclose(out[1], p[1]) and torch.allclose(out[2], p[2]), \
        "a non-anchor was modified"
    assert torch.allclose(clamp_trusted_static(p, torch.zeros(3, dtype=torch.bool)), p)

    # And the clamp survives the whole outer update (section 31).
    x0, yT, _ = _two_blocks()
    graph = build_neighbor_graph(x0, k=6)
    n = int(x0.shape[0])
    trusted = torch.zeros(n, dtype=torch.bool)
    trusted[:5] = True
    update = soft_outer_update(
        soft_probs=torch.full((n, 2), 0.5), energy=torch.rand((n, 2)),
        normalized_margin=torch.rand(n), comparable=torch.ones(n, dtype=torch.bool),
        graph=graph, trusted_mask=trusted, memory=None)
    assert torch.allclose(update.soft_after[:5, 0], torch.ones(5)), \
        "trusted anchors moved during the outer update"
    assert bool((update.hard_after[:5] == 0).all()), "trusted anchors are not static"


# ---------------------------------------------------------------------------
# I, J, K -- persistent softness
# ---------------------------------------------------------------------------


def test_i_outer_update_leaves_uncertain_points_soft():
    n = 60
    graph = build_neighbor_graph(torch.rand((n, 3), generator=torch.Generator().manual_seed(2)), k=6)
    energy = torch.full((n, 2), 1.0)
    energy[:, 1] = 1.02                                   # nearly tied everywhere
    update = soft_outer_update(
        soft_probs=torch.full((n, 2), 0.5), energy=energy,
        normalized_margin=torch.zeros(n), comparable=torch.ones(n, dtype=torch.bool),
        graph=graph, trusted_mask=torch.zeros(n, dtype=torch.bool), memory=None)
    p = update.soft_after
    assert float(p.max()) < 0.999, f"an ambiguous point became one-hot (max {float(p.max()):.6f})"
    assert float(p.min()) > 0.001, "an ambiguous point was driven to zero"
    total = p.sum(dim=1)
    assert torch.allclose(total, torch.ones_like(total), atol=1.0e-6)


def test_j_argmax_projection_does_not_mutate_the_soft_state():
    n = 40
    p = torch.rand((n, 3), generator=torch.Generator().manual_seed(3))
    p = p / p.sum(dim=1, keepdim=True)
    snapshot = p.clone()
    hard = p.argmax(dim=1)
    one_hot = torch.nn.functional.one_hot(hard, 3).to(p)
    assert torch.equal(p, snapshot), "taking argmax mutated P"
    assert not torch.equal(one_hot, p), "the projection and the state are the same object"
    # The state must not be replaceable by its own one-hot (section 43).
    assert float((p - one_hot).abs().max()) > 0.0


def test_k_soft_state_is_unchanged_between_outer_updates():
    n = 50
    graph = build_neighbor_graph(torch.rand((n, 3), generator=torch.Generator().manual_seed(4)), k=5)
    p = torch.full((n, 2), 0.5)
    update = soft_outer_update(
        soft_probs=p, energy=torch.rand((n, 2)), normalized_margin=torch.rand(n),
        comparable=torch.ones(n, dtype=torch.bool), graph=graph,
        trusted_mask=torch.zeros(n, dtype=torch.bool), memory=None)
    assert torch.allclose(update.soft_before, torch.full((n, 2), 0.5)), \
        "the outer update mutated the incoming state"
    # Repeated argmax projections (one per inner solve) change nothing.
    after = update.soft_after.clone()
    for _ in range(5):
        _ = after.argmax(dim=1)
    assert torch.equal(after, update.soft_after), "the state drifted across projections"


# ---------------------------------------------------------------------------
# L, M, N, O, P -- temporal ordering
# ---------------------------------------------------------------------------


def test_l_memory_is_zero_after_initialization():
    guard = TemporalOrderGuard()
    memory = torch.zeros((10, 2))
    guard.check_memory_is_zero(memory)                       # no solve yet
    assert guard.num_memory_updates == 0
    assert not guard.allow_memory_update()


def test_m_memory_is_still_zero_after_the_baseline_solve():
    guard = TemporalOrderGuard()
    guard.record_solve()                                      # inner solve 0
    guard.check_memory_is_zero(torch.zeros((10, 2)))
    assert not guard.allow_memory_update(), "memory was allowed after one solve"
    try:
        guard.record_memory_update()
    except AssertionError as error:
        assert "section 45" in str(error)
    else:
        raise AssertionError("a memory update was permitted after a single solve")
    # A non-zero memory at this point is a violation and must be caught.
    bad = torch.zeros((10, 2))
    bad[0, 1] = 0.3
    try:
        guard.check_memory_is_zero(bad)
    except AssertionError as error:
        assert "section 44" in str(error)
    else:
        raise AssertionError("non-zero memory before the second solve was accepted")


def test_n_first_outer_proposal_uses_zero_memory():
    n = 30
    graph = build_neighbor_graph(torch.rand((n, 3), generator=torch.Generator().manual_seed(5)), k=5)
    energy = torch.rand((n, 2), generator=torch.Generator().manual_seed(6))
    args = dict(soft_probs=torch.full((n, 2), 0.5), energy=energy,
                normalized_margin=torch.rand(n), comparable=torch.ones(n, dtype=torch.bool),
                graph=graph, trusted_mask=torch.zeros(n, dtype=torch.bool))
    none_memory = soft_outer_update(**args, memory=None)
    zero_memory = soft_outer_update(**args, memory=torch.zeros((n, 2)))
    assert torch.allclose(none_memory.soft_after, zero_memory.soft_after, atol=1.0e-9), \
        "zero memory is not equivalent to no memory in the first proposal"


def test_o_first_memory_update_needs_two_solved_states():
    guard = TemporalOrderGuard()
    guard.record_solve()
    guard.record_solve()                                      # S^(0) and S^(1) exist
    assert guard.allow_memory_update()
    guard.record_memory_update()
    assert guard.as_dict() == {"num_solved_states": 2, "num_temporal_credit_updates": 1}


def test_p_memory_influences_only_the_next_proposal():
    """Section 36: a reward never retroactively alters the solved P^(m)."""
    n = 20
    graph = build_neighbor_graph(torch.rand((n, 3), generator=torch.Generator().manual_seed(7)), k=4)
    p_m = torch.full((n, 2), 0.5)
    energy = torch.full((n, 2), 1.0)
    common = dict(energy=energy, normalized_margin=torch.zeros(n),
                  comparable=torch.ones(n, dtype=torch.bool), graph=graph,
                  trusted_mask=torch.zeros(n, dtype=torch.bool))

    no_mem = soft_outer_update(soft_probs=p_m, memory=None, **common)
    memory = torch.zeros((n, 2))
    memory[:, 1] = 0.8
    with_mem = soft_outer_update(soft_probs=p_m, memory=memory, **common)

    assert torch.allclose(no_mem.soft_before, with_mem.soft_before), \
        "memory changed the state it was computed from"
    assert not torch.allclose(no_mem.soft_after, with_mem.soft_after), \
        "memory had no effect on the next proposal"
    assert float(with_mem.soft_after[:, 1].mean()) > float(no_mem.soft_after[:, 1].mean()), \
        "positive moving memory did not raise the moving probability"


def test_section44_initialization_cannot_receive_memory():
    assert_no_memory_in_initialization()
    names = set(inspect.signature(train_soft_binary_labels).parameters)
    assert not any("memory" in n.lower() for n in names), f"memory leaked into init: {names}"
    init_names = set(inspect.signature(run_soft_endpoint_contrastive_initialization).parameters)
    assert not any("memory" in n.lower() for n in init_names), \
        f"memory leaked into the initializer: {init_names}"

    def bad(memory=None):
        return memory

    try:
        assert_no_memory_in_initialization(bad)
    except AssertionError as error:
        assert "section 44" in str(error)
    else:
        raise AssertionError("the guard accepted a function with a memory argument")


# ---------------------------------------------------------------------------
# Q, R, S -- smooth refinement
# ---------------------------------------------------------------------------


def test_q_small_evidence_change_gives_small_probability_change():
    n = 40
    graph = build_neighbor_graph(torch.rand((n, 3), generator=torch.Generator().manual_seed(8)), k=5)
    base = torch.rand((n, 2), generator=torch.Generator().manual_seed(9))
    p = torch.full((n, 2), 0.5)
    common = dict(soft_probs=p, normalized_margin=torch.full((n,), 0.5),
                  comparable=torch.ones(n, dtype=torch.bool), graph=graph,
                  trusted_mask=torch.zeros(n, dtype=torch.bool), memory=None)

    a = soft_outer_update(energy=base, **common)
    b = soft_outer_update(energy=base + 0.001, **common)
    delta = (a.soft_after - b.soft_after).abs().sum(dim=1).max()
    assert float(delta) < 0.01, \
        f"a 0.001 energy change moved P by {float(delta):.5f}; the update is not smooth"

    # A hard argmin rule would have produced a full 0/1 swing on ties.
    assert float(a.soft_after.max()) < 0.999, "the update saturated to one-hot"


def test_r_memory_shifts_probabilities_smoothly():
    """Section 50 Test R: [0.55, 0.45] + positive moving memory -> ~[0.47, 0.53]."""
    u = torch.tensor([[0.55, 0.45]], dtype=DT)
    memory = torch.tensor([[0.0, 1.0]], dtype=DT)
    out = apply_memory_in_logit_space(u, memory, lambda_memory=0.25)
    assert float(out.sum()) == 1.0 or abs(float(out.sum()) - 1.0) < 1.0e-12
    assert float(out[0, 1]) > float(u[0, 1]), "memory did not raise the moving probability"
    assert float(out[0, 1]) < 0.75, f"memory hard-flipped instead of biasing: {out.tolist()}"
    assert float(out[0, 0]) > 0.2, "the static class was wiped out"
    # It really does cross over for this example, smoothly.
    assert float(out[0, 1]) > 0.5 > float(out[0, 0]), \
        f"expected a smooth crossover, got {out.tolist()}"

    # Zero memory is exactly a no-op.
    assert torch.allclose(apply_memory_in_logit_space(u, torch.zeros_like(memory)), u, atol=1.0e-9)


def test_s_only_argmax_changes_the_temporary_physical_class():
    n = 12
    p = torch.full((n, 3), 1.0 / 3.0)
    p[:, 1] += 0.01
    p = p / p.sum(dim=1, keepdim=True)
    hard = p.argmax(dim=1)
    assert bool((hard == 1).all()), "argmax did not pick the leading class"
    # Nudging a non-leading class leaves the physical class alone...
    q = p.clone()
    q[:, 2] += 0.005
    q = q / q.sum(dim=1, keepdim=True)
    assert bool((q.argmax(dim=1) == 1).all()), "a sub-threshold nudge changed the class"
    # ...and the probabilities moved anyway, which is the point of soft state.
    assert float((q - p).abs().sum()) > 0.0


# ---------------------------------------------------------------------------
# Sections 21-24, 41-42: the update's components
# ---------------------------------------------------------------------------


def test_section21_render_probabilities_respect_support():
    energy = torch.tensor([[1.0, 5.0], [1.0, 5.0]], dtype=DT)
    valid = torch.tensor([[True, True], [False, False]])
    r = render_probabilities(energy, tau_render=1.0, valid=valid)
    assert float(r[0, 0]) > float(r[0, 1]), "lower energy did not get higher probability"
    assert torch.allclose(r[1], torch.full((2,), 0.5, dtype=DT)), \
        "a Gaussian with no supported hypothesis did not fall back to uniform"
    total = r.sum(dim=1)
    assert torch.allclose(total, torch.ones_like(total), atol=1.0e-6)
    # Unsupported hypotheses must not win by having a small unmeasured energy.
    one_sided = render_probabilities(torch.tensor([[5.0, 0.0]], dtype=DT),
                                     valid=torch.tensor([[True, False]]))
    assert int(one_sided.argmax(dim=1)) == 0, "an unsupported hypothesis won"


def test_section23_propagation_is_spatial_and_normalised():
    n = 30
    pts = torch.linspace(0.0, 1.0, n, dtype=DT)[:, None]
    pts = torch.cat((pts, torch.zeros((n, 2), dtype=DT)), dim=1)
    graph = build_neighbor_graph(pts, k=4)
    p = torch.zeros((n, 2), dtype=DT)
    p[:, 0] = 1.0
    p[15] = torch.tensor([0.0, 1.0], dtype=DT)
    out = spatial_probability_propagation(p, graph)
    total = out.sum(dim=1)
    assert torch.allclose(total, torch.ones_like(total), atol=1.0e-6), "N is not a distribution"
    assert float(out[14, 1]) > 0.0 and float(out[16, 1]) > 0.0, "the spike did not spread"
    assert float(out[14, 1]) > float(out[0, 1]), "propagation is not local"

    # An isolated node keeps its own distribution rather than becoming zeros.
    lone = build_neighbor_graph(torch.zeros((1, 3), dtype=DT), k=4)
    single = torch.tensor([[0.3, 0.7]], dtype=DT)
    assert torch.allclose(spatial_probability_propagation(single, lone), single)


def test_section24_alpha_follows_render_confidence():
    margin = torch.tensor([0.0, 0.5, 1.0], dtype=DT)
    a = render_alpha(margin)
    assert float(a[0]) == 1.0, "ambiguous evidence did not give alpha -> 1"
    assert float(a[2]) == 0.0, "confident evidence did not give alpha -> 0"
    # A non-comparable Gaussian must ignore its render evidence entirely.
    a2 = render_alpha(margin, comparable=torch.tensor([True, True, False]))
    assert float(a2[2]) == 1.0, "a non-comparable Gaussian still trusted its render evidence"

    r = torch.tensor([[1.0, 0.0], [1.0, 0.0]], dtype=DT)
    nb = torch.tensor([[0.0, 1.0], [0.0, 1.0]], dtype=DT)
    u = combine_render_and_spatial(r, nb, torch.tensor([0.0, 1.0], dtype=DT))
    assert torch.allclose(u[0], r[0]), "alpha=0 did not take pure render evidence"
    assert torch.allclose(u[1], nb[1]), "alpha=1 did not take the pure neighbourhood"


def test_section42_soft_oscillation_detects_reversal():
    p2 = torch.tensor([[0.5, 0.5]], dtype=DT)
    p1 = torch.tensor([[0.7, 0.3]], dtype=DT)
    forward = torch.tensor([[0.9, 0.1]], dtype=DT)
    back = torch.tensor([[0.5, 0.5]], dtype=DT)
    assert soft_oscillation(forward, p1, p2)["fraction_soft_update_reversal"] == 0.0, \
        "a monotone move was called a reversal"
    assert soft_oscillation(back, p1, p2)["fraction_soft_update_reversal"] == 1.0, \
        "a reversal was not detected"
    nan = soft_oscillation(forward, None, None)["fraction_soft_update_reversal"]
    assert nan != nan, "a reversal fraction was reported with no history"


def test_section15_soft_clustering_keeps_probabilities():
    xyz = torch.cat((torch.rand((40, 3), generator=torch.Generator().manual_seed(11)),
                     torch.rand((40, 3), generator=torch.Generator().manual_seed(12)) + 5.0), 0)
    q_dyn = torch.cat((torch.full((40,), 0.9), torch.full((40,), 0.9)))
    pi, mu = soft_spatial_clustering(xyz, q_dyn, 2, iterations=20, seed=SEED)
    assert tuple(pi.shape) == (80, 2)
    total = pi.sum(dim=1)
    assert torch.allclose(total, torch.ones_like(total), atol=1.0e-6), "pi is not a distribution"
    assert not bool(((pi > 0.999999) | (pi < 1.0e-6)).all()), "pi collapsed to hard assignments"
    # The two separated blobs should still be told apart by the argmax.
    assert int(pi[:40].argmax(dim=1).mode().values) != int(pi[40:].argmax(dim=1).mode().values), \
        "the two blobs were merged"
    a, _ = soft_spatial_clustering(xyz, q_dyn, 2, seed=SEED)
    assert torch.allclose(pi, a), "soft clustering is not deterministic for a fixed seed"


def test_compose_rejects_a_mismatched_pi():
    q = torch.tensor([[0.4, 0.6]], dtype=DT)
    try:
        compose_soft_initial_state(q, torch.ones((1, 2), dtype=DT), 3)
    except ValueError:
        pass
    else:
        raise AssertionError("a pi with the wrong part count was accepted")


# ===========================================================================
# CODEX_soft_label_algorithmic_corrections -- required tests A-Q (§29-§31)
# ===========================================================================


def test_corrections_a_low_similarity_does_not_repel():
    """§29 Test A: s_ij ~ 0 must give ~zero pair gradient, never repulsion."""
    from rigid_parts.soft_labels import positive_smoothness_loss

    pts = torch.tensor([[0.0, 0.0, 0.0], [0.1, 0.0, 0.0]], dtype=DT)
    graph = build_neighbor_graph(pts, k=1)
    logits = torch.tensor([[0.0, 0.0], [0.0, 0.0]], dtype=DT, requires_grad=True)
    q = torch.softmax(logits, dim=1)

    dissimilar = torch.zeros(graph.num_edges, dtype=DT)          # s_ij = 0
    loss = positive_smoothness_loss(q, graph, dissimilar, gamma=2.0)
    assert float(loss) == 0.0, f"a dissimilar pair contributed {float(loss)}"
    loss.backward()
    assert float(logits.grad.abs().max()) == 0.0, \
        "a dissimilar pair produced a non-zero gradient; it is being repelled"

    # The superseded BCE form DID push them apart -- that is what §4 removes.
    logits2 = torch.tensor([[0.2, -0.2], [-0.2, 0.2]], dtype=DT, requires_grad=True)
    old = pairwise_contrastive_loss(torch.softmax(logits2, dim=1), graph, dissimilar, gamma=2.0)
    old.backward()
    assert float(logits2.grad.abs().max()) > 0.0, \
        "sanity: the superseded objective should have had a repulsive gradient"


def test_corrections_b_high_similarity_attracts():
    """§29 Test B: s_ij ~ 1 must reduce ||q_i - q_j||."""
    from rigid_parts.soft_labels import positive_smoothness_loss

    pts = torch.tensor([[0.0, 0.0, 0.0], [0.1, 0.0, 0.0]], dtype=DT)
    graph = build_neighbor_graph(pts, k=1)
    similar = torch.ones(graph.num_edges, dtype=DT)
    logits = torch.tensor([[1.0, -1.0], [-1.0, 1.0]], dtype=DT, requires_grad=True)

    optimizer = torch.optim.Adam([logits], lr=0.1)
    before = float((torch.softmax(logits, 1)[0] - torch.softmax(logits, 1)[1]).abs().sum())
    for _ in range(100):
        optimizer.zero_grad()
        positive_smoothness_loss(torch.softmax(logits, dim=1), graph, similar, gamma=2.0).backward()
        optimizer.step()
    after = float((torch.softmax(logits, 1)[0] - torch.softmax(logits, 1)[1]).abs().sum())
    assert after < before, f"a similar pair did not converge: {before:.5f} -> {after:.5f}"


def test_corrections_c_endpoint_logit_bias_is_weak():
    """§29 Test C: with eps_d = 0.2 no point may start near one-hot."""
    from rigid_parts.soft_labels import weak_endpoint_logits

    d = torch.linspace(0.0, 1.0, 501, dtype=DT)
    logits = weak_endpoint_logits(d, epsilon_d=0.2)
    q = torch.softmax(logits, dim=1)
    assert float(q.max()) < 0.56, f"initial probability reached {float(q.max()):.4f}, not near 0.5"
    assert float(q.min()) > 0.44, f"initial probability fell to {float(q.min()):.4f}"
    # The logit DIFFERENCE is exactly eps_d * r, r in [-1, 1].
    diff = logits[:, 1] - logits[:, 0]
    assert abs(float(diff.max()) - 0.2) < 1.0e-6 and abs(float(diff.min()) + 0.2) < 1.0e-6, \
        f"logit difference range {float(diff.min()):.4f}..{float(diff.max()):.4f}, expected +-0.2"
    # And this is NOT the forbidden rank-percentile PROBABILITY seeding, which
    # would have spread q_dynamic across the whole [0, 1] range.
    assert float(q[:, 1].max() - q[:, 1].min()) < 0.15, \
        "the initialization spans too much probability mass to be a weak bias"


def test_corrections_d_endpoint_ordering_is_preserved():
    """§29 Test D: higher endpoint rank starts slightly more dynamic."""
    from rigid_parts.soft_labels import weak_endpoint_logits

    d = torch.tensor([0.1, 0.5, 0.9, 0.3], dtype=DT)
    q = torch.softmax(weak_endpoint_logits(d, epsilon_d=0.2), dim=1)
    assert torch.equal(q[:, 1].argsort(), d.argsort()), \
        "the initial q_dynamic does not follow the endpoint ordering"
    assert float(q[2, 1]) > float(q[0, 1]), "the largest distance is not the most dynamic"


def test_corrections_e_polarization_ramp():
    """§29 Test E: ~0 at iteration 0, exactly lambda_max at the end."""
    from rigid_parts.soft_labels import polarization_weight

    assert polarization_weight(0.0, lambda_max=0.5) == 0.0, "polarization fired at iteration 0"
    assert polarization_weight(0.20, lambda_max=0.5) == 0.0, "ramp started before r0"
    assert abs(polarization_weight(0.45, lambda_max=0.5) - 0.25) < 1.0e-9, "midpoint is wrong"
    assert polarization_weight(0.70, lambda_max=0.5) == 0.5, "ramp did not reach the max at r1"
    assert polarization_weight(1.0, lambda_max=0.5) == 0.5, "the tail is not full strength"
    # And it is actually applied during training.
    x0, yT, _ = _two_blocks()
    _, graph, anchors, similarity = _init_pieces(x0, yT)
    r = train_soft_binary_labels(
        graph, similarity, anchors.trusted_static, anchors.geometric_static,
        anchors.dynamic, num_gaussians=int(x0.shape[0]), iterations=100,
        lambda_polar=0.5, log_every=10)
    assert r.history[0]["lambda_polar_current"] == 0.0, "iteration 0 had polarization"
    assert abs(r.history[-1]["lambda_polar_current"] - 0.5) < 1.0e-9, \
        "the final iteration was not at full polarization"


def test_corrections_g_zero_valid_candidates_gives_alpha_one():
    """§30 Test G."""
    from rigid_parts.soft_labels import validity_alpha

    r = torch.tensor([[0.5, 0.5]], dtype=DT)
    assert float(validity_alpha(r, torch.tensor([0]))[0]) == 1.0, "v=0 did not give alpha=1"


def test_corrections_h_one_valid_candidate_gives_alpha_zero():
    """§30 Test H: the correction that matters most."""
    from rigid_parts.soft_labels import validity_alpha

    r = torch.tensor([[1.0, 0.0]], dtype=DT)
    assert float(validity_alpha(r, torch.tensor([1]))[0]) == 0.0, "v=1 did not give alpha=0"
    # Even an apparently ambiguous R is overridden by v=1.
    assert float(validity_alpha(torch.tensor([[0.5, 0.5]], dtype=DT), torch.tensor([1]))[0]) == 0.0


def test_corrections_i_j_single_valid_candidate_drives_the_proposal():
    """§30 Tests I and J: R and U become one-hot on the only valid candidate."""
    n = 2
    graph = build_neighbor_graph(torch.tensor([[0.0, 0.0, 0.0], [5.0, 0.0, 0.0]], dtype=DT), k=1)
    energy = torch.tensor([[1.0, 1.0], [1.0, 1.0]], dtype=DT)
    valid = torch.tensor([[True, False], [False, True]])
    prior = torch.tensor([[0.5, 0.5], [0.5, 0.5]], dtype=DT)

    update = soft_outer_update(
        soft_probs=prior, energy=energy, normalized_margin=torch.zeros(n, dtype=DT),
        comparable=torch.zeros(n, dtype=torch.bool), graph=graph,
        trusted_mask=torch.zeros(n, dtype=torch.bool), memory=None, valid=valid)

    assert torch.allclose(update.render_probs[0], torch.tensor([1.0, 0.0], dtype=DT), atol=1.0e-6), \
        f"Test I: R is {update.render_probs[0].tolist()}, expected [1, 0]"
    assert torch.allclose(update.render_probs[1], torch.tensor([0.0, 1.0], dtype=DT), atol=1.0e-6), \
        f"Test J: R is {update.render_probs[1].tolist()}, expected [0, 1]"
    assert torch.allclose(update.proposal[0], torch.tensor([1.0, 0.0], dtype=DT), atol=1.0e-6), \
        f"Test I: U is {update.proposal[0].tolist()}, expected [1, 0]"
    assert torch.allclose(update.proposal[1], torch.tensor([0.0, 1.0], dtype=DT), atol=1.0e-6), \
        f"Test J: U is {update.proposal[1].tolist()}, expected [0, 1]"
    assert float(update.alpha.max()) == 0.0, "alpha was not 0 for one-valid-candidate Gaussians"

    # The superseded rule threw this evidence away entirely.
    legacy = soft_outer_update(
        soft_probs=prior, energy=energy, normalized_margin=torch.zeros(n, dtype=DT),
        comparable=torch.zeros(n, dtype=torch.bool), graph=graph,
        trusted_mask=torch.zeros(n, dtype=torch.bool), memory=None, valid=valid,
        confidence_mode="normalized_margin")
    assert float(legacy.alpha.min()) == 1.0, \
        "sanity: the legacy rule should have ignored the render evidence here"


def test_corrections_k_l_alpha_from_render_entropy():
    """§30 Tests K and L: confident -> alpha << 1, ambiguous -> alpha ~ 1."""
    from rigid_parts.soft_labels import validity_alpha

    two = torch.tensor([2, 2])
    confident = validity_alpha(torch.tensor([[0.99, 0.01]], dtype=DT), two[:1])
    ambiguous = validity_alpha(torch.tensor([[0.51, 0.49]], dtype=DT), two[:1])
    assert float(confident[0]) < 0.15, f"Test K: alpha = {float(confident[0]):.4f}, expected << 1"
    assert float(ambiguous[0]) > 0.99, f"Test L: alpha = {float(ambiguous[0]):.4f}, expected ~ 1"


def test_corrections_m_entropy_normalisation_is_bounded():
    """§30 Test M: 0 <= H(R)/log(K+1) <= 1 for every K."""
    from rigid_parts.soft_labels import validity_alpha

    for k1 in (2, 3, 5, 8):
        uniform = torch.full((1, k1), 1.0 / k1, dtype=DT)
        one_hot = torch.zeros((1, k1), dtype=DT)
        one_hot[0, 0] = 1.0
        rand = torch.rand((20, k1), generator=torch.Generator().manual_seed(k1), dtype=DT)
        rand = rand / rand.sum(dim=1, keepdim=True)
        v = torch.full((1,), k1)
        assert abs(float(validity_alpha(uniform, v)[0]) - 1.0) < 1.0e-6, \
            f"K+1={k1}: a uniform R did not give alpha=1"
        assert float(validity_alpha(one_hot, v)[0]) < 1.0e-6, \
            f"K+1={k1}: a one-hot R did not give alpha=0"
        a = validity_alpha(rand, torch.full((20,), k1))
        assert float(a.min()) >= 0.0 and float(a.max()) <= 1.0, \
            f"K+1={k1}: alpha left [0, 1]"


def test_corrections_n_o_temporal_score_is_render_dominant():
    """§31 Tests N and O."""
    from rigid_parts.temporal_credit import ScoreNormalization, combine_global_score

    norm = ScoreNormalization(c_render=2.0, c_match=4.0, c_render_local=1.0, c_match_local=1.0)
    render_only = combine_global_score(3.0, 8.0, 8.0, 0.0, norm, lambda_match=0.0)
    assert abs(render_only.total - 1.5) < 1.0e-12, \
        f"Test N: match weight 0 did not give the pure render score ({render_only.total})"

    weighted = combine_global_score(3.0, 8.0, 8.0, 0.0, norm, lambda_match=0.05)
    assert abs(weighted.total - (1.5 + 0.05 * 2.0)) < 1.0e-12, \
        f"Test O: S != S_R + 0.05 S_M ({weighted.total})"
    # And the default really is render-dominant.
    default = combine_global_score(3.0, 8.0, 8.0, 0.0, norm)
    assert abs(default.total - weighted.total) < 1.0e-12, "the default match weight is not 0.05"


def test_corrections_p_normalisation_stays_fixed():
    """§31 Test P."""
    from rigid_parts.temporal_credit import ScoreNormalization, combine_global_score

    norm = ScoreNormalization(c_render=2.0, c_match=4.0, c_render_local=1.0,
                              c_match_local=1.0, unsupported_penalty=7.0)
    snapshot = norm.as_dict()
    for render, match in ((1.0, 1.0), (50.0, 90.0), (0.01, 0.02)):
        combine_global_score(render, match, match, 0.0, norm)
    assert norm.as_dict() == snapshot, "the fixed normalisation drifted"


def test_corrections_q_no_reward_before_two_solved_states():
    """§31 Test Q -- the existing guard still holds after the corrections."""
    guard = TemporalOrderGuard()
    guard.record_solve()
    assert not guard.allow_memory_update()
    guard.record_solve()
    assert guard.allow_memory_update()
    guard.record_memory_update()
    assert guard.as_dict()["num_temporal_credit_updates"] == 1


def test_corrections_score_f1_correlation_diagnostic():
    """§25: the diagnostic must flag an anti-correlated score."""
    from rigid_parts.temporal_credit import score_f1_correlation

    # signature: (score_total, score_render, score_match, f1)
    # A tracking score FALLS as F1 rises.
    good = score_f1_correlation(score_total=[2.0, 1.5, 1.0], score_render=[2.0, 1.5, 1.0],
                                score_match=[1.0, 1.0, 1.0], f1=[70.0, 80.0, 90.0])
    assert good["pearson_score_f1"] < 0, "a tracking score was not negatively correlated"
    assert good["tracks_quality"] is True
    assert good["expected_sign"] == "negative"

    # An anti-correlated score RISES as F1 rises -- the USB soft run's failure.
    bad = score_f1_correlation(score_total=[1.0, 1.5, 2.0], score_render=[2.0, 1.5, 1.0],
                               score_match=[1.0, 1.5, 2.0], f1=[70.0, 80.0, 90.0])
    assert bad["pearson_score_f1"] > 0, "an anti-correlated score was not detected"
    assert bad["tracks_quality"] is False, "the diagnostic did not flag a bad score"
    # The subterms are reported separately, so the culprit is identifiable.
    assert bad["pearson_render_f1"] < 0 < bad["pearson_match_f1"], \
        "the diagnostic cannot separate a good render term from a bad match term"


def test_corrections_validity_diagnostics_are_reported():
    """§19: the per-validity-case breakdown is required every outer."""
    n = 6
    graph = build_neighbor_graph(torch.rand((n, 3), generator=torch.Generator().manual_seed(21)), k=2)
    valid = torch.tensor([[False, False], [True, False], [False, True],
                          [True, True], [True, True], [True, True]])
    update = soft_outer_update(
        soft_probs=torch.full((n, 2), 0.5, dtype=DT),
        energy=torch.rand((n, 2), generator=torch.Generator().manual_seed(22), dtype=DT),
        normalized_margin=torch.rand(n, dtype=DT), comparable=torch.ones(n, dtype=torch.bool),
        graph=graph, trusted_mask=torch.zeros(n, dtype=torch.bool), memory=None, valid=valid)
    d = update.diagnostics
    for key in ("num_valid0", "num_valid1", "num_valid2plus", "mean_alpha_all",
                "mean_alpha_valid0", "mean_alpha_valid1", "mean_alpha_valid2plus",
                "static_only_valid", "moving_only_valid", "both_valid", "neither_valid"):
        assert key in d, f"§19 diagnostic {key} is missing"
    assert d["num_valid0"] == 1 and d["num_valid1"] == 2 and d["num_valid2plus"] == 3
    assert d["static_only_valid"] == 1 and d["moving_only_valid"] == 1
    assert d["both_valid"] == 3 and d["neither_valid"] == 1
    assert d["mean_alpha_valid0"] == 1.0 and d["mean_alpha_valid1"] == 0.0
    # §33: the artifact carries the validity fields.
    payload = update.as_dict()
    assert "candidate_valid_mask" in payload and "num_valid_candidates" in payload
    assert "alpha" in payload


def test_cli_defaults_match_the_spec():
    from argparse import ArgumentParser

    from arguments import ModelParams, OptimizationParams, PipelineParams, add_dynamic_args

    parser = ArgumentParser()
    ModelParams(parser)
    OptimizationParams(parser)
    PipelineParams(parser)
    add_dynamic_args(parser)
    args = parser.parse_args([])
    for flag, expected in (
        ("outer_soft_labels", True),           # §46: enabled by default
        ("init_dynamic_anchor_quantile", 0.95),
        ("init_pair_confidence_gamma", 2.0),
        ("init_soft_cluster_iters", 20),
        ("lambda_init_pair", 1.0),
        ("lambda_init_polar", 0.5),
        ("lambda_init_anchor", 1.0),
    ):
        assert hasattr(args, flag), f"--{flag} is not registered"
        assert getattr(args, flag) == expected, \
            f"--{flag} default is {getattr(args, flag)}, spec says {expected}"
    legacy = parser.parse_args(["--no-outer_soft_labels"])
    assert legacy.outer_soft_labels is False, "the legacy hard ablation is not reachable"


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
    print(f"PASS soft_labels_outer_cycle ({len(tests)} tests)")
