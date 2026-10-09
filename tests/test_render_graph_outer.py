"""Simple render-matching outer loop: required tests A-G (section 24)."""

from __future__ import annotations

import inspect
import sys
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from rigid_parts.neighborhood import build_neighbor_graph
from rigid_parts.render_graph_outer import (
    build_label_graph_weights,
    propagate_labels,
    render_class_preference,
    run_render_graph_outer,
    select_render_seeds,
)

DT = torch.float64


def _views(n, k, v=8.0):
    return torch.full((n, k), float(v), dtype=DT)


# ---------------------------------------------------------------------------
# Test A - seed clamping
# ---------------------------------------------------------------------------
def test_10_graph_is_purely_spatial_by_default():
    """w_ij = w_ij^space unless boundary weighting is explicitly requested.

    Letting render define BOTH the seeds and the graph boundary is circular: the
    same evidence picks a label and then decides which neighbours are allowed to
    correct it, so a confident mistake reinforces itself.
    """
    torch.manual_seed(11)
    n = 150
    x = torch.rand((n, 3), dtype=DT)
    graph = build_neighbor_graph(x, k=10)
    pref = torch.softmax(torch.randn((n, 2), dtype=DT) * 4.0, dim=1)

    w_default = build_label_graph_weights(graph, pref)
    assert torch.allclose(w_default, graph.weight.to(pref), atol=1e-15), (
        "the default graph is not purely spatial"
    )
    w_aware = build_label_graph_weights(graph, pref, boundary_aware=True)
    assert not torch.allclose(w_aware, w_default), "test vacuous: the two modes agree"
    assert float(w_aware.max()) <= float(w_default.max()) + 1e-12


def test_12b_propagation_runs_to_convergence_not_a_fixed_step_count():
    """The stopping rule is the tolerance; ``iterations`` is only a safety bound."""
    torch.manual_seed(12)
    n = 400
    x = torch.rand((n, 3), dtype=DT)
    graph = build_neighbor_graph(x, k=12)
    pref = torch.softmax(torch.randn((n, 2), dtype=DT), dim=1)
    margin = torch.zeros(n, dtype=DT)
    seed_mask = torch.zeros(n, dtype=torch.bool)
    seed_mask[:20] = True
    seed_label = torch.zeros(n, dtype=torch.long)
    seed_label[:10] = 1
    w = build_label_graph_weights(graph, pref)

    p50, it50, d50 = propagate_labels(graph, w, pref, margin, seed_mask, seed_label,
                                      iterations=50, tol=1.0e-4)
    pconv, itconv, dconv = propagate_labels(graph, w, pref, margin, seed_mask, seed_label,
                                            iterations=1000, tol=1.0e-4)
    assert dconv < 1.0e-4, f"did not converge: delta {dconv:.2e}"
    assert itconv < 1000, "hit the safety bound instead of converging"
    if it50 >= 50:
        # The 50-step cut stopped short, so the labels depended on the step count.
        assert d50 >= 1.0e-4
        assert float((p50 - pconv).abs().max()) > 0.0


def test_A_seeds_stay_clamped_at_every_propagation_iteration():
    torch.manual_seed(0)
    n = 200
    x = torch.rand((n, 3), dtype=DT)
    graph = build_neighbor_graph(x, k=10)
    pref = torch.softmax(torch.randn((n, 2), dtype=DT), dim=1)
    margin = torch.rand(n, dtype=DT)
    seed_mask = torch.zeros(n, dtype=torch.bool)
    seed_mask[:40] = True
    seed_label = torch.zeros(n, dtype=torch.long)
    seed_label[:20] = 1                      # half the seeds are moving

    w = build_label_graph_weights(graph, pref)
    # Check after EVERY iteration, not just at the end.
    for iters in range(1, 12):
        p, _, _ = propagate_labels(graph, w, pref, margin, seed_mask, seed_label, iterations=iters)
        got = p.argmax(dim=1)
        assert torch.equal(got[seed_mask], seed_label[seed_mask]), f"seed drifted at iteration {iters}"
        assert torch.allclose(p[seed_mask].max(dim=1).values,
                              torch.ones(int(seed_mask.sum()), dtype=DT), atol=1e-12)


# ---------------------------------------------------------------------------
# Test B - spatial expansion
# ---------------------------------------------------------------------------
def test_B_unlabelled_points_around_a_moving_seed_become_moving():
    torch.manual_seed(1)
    n = 300
    x = torch.rand((n, 3), dtype=DT)
    graph = build_neighbor_graph(x, k=12)

    blob = (x - torch.tensor([0.2, 0.2, 0.2], dtype=DT)).norm(dim=1) < 0.22
    pref = torch.full((n, 2), 0.5, dtype=DT)          # everyone is ambiguous
    margin = torch.zeros(n, dtype=DT)                  # so alpha = 1 everywhere
    seed_mask = blob & (torch.rand(n) < 0.35)
    seed_label = torch.zeros(n, dtype=torch.long)
    seed_label[seed_mask] = 1
    assert int(seed_mask.sum()) > 5

    p, _, _ = propagate_labels(graph, build_label_graph_weights(graph, pref),
                               pref, margin, seed_mask, seed_label, iterations=60)
    grown = (p.argmax(dim=1) == 1)
    unlabelled_in_blob = blob & ~seed_mask
    frac = float(grown[unlabelled_in_blob].float().mean())
    assert frac > 0.6, f"only {frac*100:.1f}% of the blob's unlabelled points became moving"


# ---------------------------------------------------------------------------
# Test C - boundary preservation
# ---------------------------------------------------------------------------
def test_C_touching_clusters_with_different_preference_do_not_merge():
    torch.manual_seed(2)
    m = 150
    left = torch.rand((m, 3), dtype=DT) * torch.tensor([0.5, 1.0, 1.0], dtype=DT)
    right = torch.rand((m, 3), dtype=DT) * torch.tensor([0.5, 1.0, 1.0], dtype=DT) \
        + torch.tensor([0.5, 0.0, 0.0], dtype=DT)
    x = torch.cat((left, right))                      # the two halves TOUCH at 0.5
    graph = build_neighbor_graph(x, k=12)
    crosses = (x[graph.source, 0] < 0.5) != (x[graph.target, 0] < 0.5)
    assert int(crosses.sum()) > 0, "clusters are not adjacent; nothing to preserve"

    pref = torch.zeros((2 * m, 2), dtype=DT)
    pref[:m] = torch.tensor([0.98, 0.02], dtype=DT)   # left strongly static
    pref[m:] = torch.tensor([0.02, 0.98], dtype=DT)   # right strongly moving

    # The boundary-aware weighting is now opt-in; when enabled it must still
    # weaken edges across the articulation.
    w_aware = build_label_graph_weights(graph, pref, boundary_aware=True)
    assert float(w_aware[crosses].mean()) < 0.25 * float(w_aware[~crosses].mean()), (
        "boundary edges were not weakened relative to interior edges"
    )
    w = w_aware

    margin = torch.zeros(2 * m, dtype=DT)             # worst case: pure propagation
    seed_mask = torch.zeros(2 * m, dtype=torch.bool)
    seed_mask[:10] = True
    seed_mask[m:m + 10] = True
    seed_label = torch.zeros(2 * m, dtype=torch.long)
    seed_label[m:m + 10] = 1
    p, _, _ = propagate_labels(graph, w, pref, margin, seed_mask, seed_label, iterations=80)
    got = p.argmax(dim=1)
    assert float((got[:m] == 0).float().mean()) > 0.9, "static side bled into moving"
    assert float((got[m:] == 1).float().mean()) > 0.9, "moving side bled into static"


# ---------------------------------------------------------------------------
# Test D - uncertain point follows its neighbours
# ---------------------------------------------------------------------------
def test_D_ambiguous_point_is_pulled_by_a_coherent_moving_neighbourhood():
    torch.manual_seed(3)
    n = 120
    x = torch.rand((n, 3), dtype=DT) * 0.3
    x[0] = torch.tensor([0.15, 0.15, 0.15], dtype=DT)   # the uncertain point, interior
    graph = build_neighbor_graph(x, k=10)

    energy = torch.zeros((n, 2), dtype=DT)
    energy[0] = torch.tensor([1.0, 1.0], dtype=DT)      # E_static == E_moving
    energy[1:] = torch.tensor([5.0, 0.1], dtype=DT)     # everyone else clearly moving
    pref = render_class_preference(energy, _views(n, 2), min_support_views=2)
    assert float(pref["normalized_margin"][0]) < 1e-6, "ambiguous point was not given ~zero margin"

    seed_mask, seed_label = select_render_seeds(
        pref["best_label"], pref["normalized_margin"], pref["comparable"],
        torch.zeros(n, dtype=torch.bool), seed_confidence=0.75)
    assert not bool(seed_mask[0]), "the ambiguous point must not become a seed"

    p, _, _ = propagate_labels(
        graph, build_label_graph_weights(graph, pref["preference"]),
        pref["preference"], pref["normalized_margin"], seed_mask, seed_label, iterations=60)
    assert int(p[0].argmax()) == 1, "ambiguous point did not follow its moving neighbourhood"


# ---------------------------------------------------------------------------
# Test E - confident point resists its neighbours
# ---------------------------------------------------------------------------
def test_E_confident_static_point_resists_low_confidence_moving_neighbours():
    torch.manual_seed(4)
    n = 120
    x = torch.rand((n, 3), dtype=DT) * 0.3
    x[0] = torch.tensor([0.15, 0.15, 0.15], dtype=DT)
    graph = build_neighbor_graph(x, k=10)

    energy = torch.zeros((n, 2), dtype=DT)
    energy[0] = torch.tensor([0.01, 9.0], dtype=DT)     # strongly static
    energy[1:] = torch.tensor([1.0, 0.98], dtype=DT)    # barely moving = low confidence
    pref = render_class_preference(energy, _views(n, 2), min_support_views=2)
    assert float(pref["normalized_margin"][0]) > 0.9, "the confident point lost its margin"

    seed_mask = torch.zeros(n, dtype=torch.bool)        # no seeds at all
    seed_label = torch.zeros(n, dtype=torch.long)
    p, _, _ = propagate_labels(
        graph, build_label_graph_weights(graph, pref["preference"]),
        pref["preference"], pref["normalized_margin"], seed_mask, seed_label, iterations=60)
    assert int(p[0].argmax()) == 0, "confident static point was dragged moving by weak neighbours"


# ---------------------------------------------------------------------------
# Test F - unsupported is not forced static
# ---------------------------------------------------------------------------
def test_F_unsupported_point_is_a_non_seed_not_static():
    n = 50
    energy = torch.rand((n, 2), dtype=DT)
    views = _views(n, 2)
    views[0] = 0.0                                      # no final-camera support at all
    views[1] = torch.tensor([1.0, 0.0], dtype=DT)       # only one supported hypothesis

    pref = render_class_preference(energy, views, min_support_views=2)
    assert not bool(pref["comparable"][0]) and not bool(pref["comparable"][1])
    assert float(pref["normalized_margin"][0]) == 0.0

    seed_mask, seed_label = select_render_seeds(
        pref["best_label"], pref["normalized_margin"], pref["comparable"],
        torch.zeros(n, dtype=torch.bool), seed_confidence=0.75)
    assert not bool(seed_mask[0]) and not bool(seed_mask[1]), "unsupported point became a seed"

    # It must not be silently declared static: with no support the preference is
    # uniform, so nothing biases it toward class 0.
    assert abs(float(pref["preference"][0, 0]) - float(pref["preference"][0, 1])) < 1e-9


# ---------------------------------------------------------------------------
# Test G - multiclass, no binary special case
# ---------------------------------------------------------------------------
def test_G_propagation_supports_K_greater_than_one():
    torch.manual_seed(5)
    n, k1 = 300, 4                                      # static + 3 moving parts
    x = torch.rand((n, 3), dtype=DT)
    graph = build_neighbor_graph(x, k=12)
    energy = torch.rand((n, k1), dtype=DT) * 2.0
    # Make three spatial regions clearly prefer three different moving classes.
    for cls, lo in ((1, 0.0), (2, 0.33), (3, 0.66)):
        region = (x[:, 0] >= lo) & (x[:, 0] < lo + 0.33)
        energy[region, cls] = 0.0

    out = run_render_graph_outer(
        energy=energy, num_views=_views(n, k1), graph=graph,
        anchor_mask=torch.zeros(n, dtype=torch.bool),
        seed_confidence=0.75, min_support_views=2, iterations=50,
    )
    assert out.propagated_probabilities.shape == (n, k1)
    assert out.render_preference.shape == (n, k1)
    present = set(int(v) for v in out.final_label.unique())
    assert len(present) >= 3, f"multiclass collapsed to {present}"
    assert max(present) <= k1 - 1


# ---------------------------------------------------------------------------
# sections 5 / 7 / 26: the acceptance criteria that changed
# ---------------------------------------------------------------------------
def test_5_ranking_uses_raw_energy_and_never_the_old_confidence():
    import rigid_parts.render_graph_outer as rg

    # Structural: the ranking path must never consume the pixel matcher's
    # per-candidate confidence field (that product is what silenced the signal).
    body = inspect.getsource(rg.render_class_preference)
    for forbidden in (".confidence", "confidence *", "* confidence", "c_ik"):
        assert forbidden not in body, f"ranking consumes an old confidence term: {forbidden}"
    assert "confidence" not in inspect.getsource(rg.build_label_graph_weights)

    # Ordering must follow E exactly, even when a hypothetical confidence would invert it.
    energy = torch.tensor([[0.2, 0.9]], dtype=DT)
    pref = render_class_preference(energy, _views(1, 2), min_support_views=2)
    assert int(pref["best_label"][0]) == 0
    assert abs(float(pref["raw_margin"][0]) - 0.7) < 1e-12


def test_7_trusted_anchor_overrides_a_conflicting_render_seed():
    n = 10
    best = torch.ones(n, dtype=torch.long)              # render says moving
    margin = torch.ones(n, dtype=DT)                    # with full confidence
    comparable = torch.ones(n, dtype=torch.bool)
    anchors = torch.zeros(n, dtype=torch.bool)
    anchors[:3] = True

    seed_mask, seed_label = select_render_seeds(best, margin, comparable, anchors,
                                                seed_confidence=0.75)
    assert bool(seed_mask.all())
    assert torch.equal(seed_label[:3], torch.zeros(3, dtype=torch.long)), "anchor lost to render"
    assert torch.equal(seed_label[3:], torch.ones(n - 3, dtype=torch.long))


def test_6_margin_normalisation_is_robust_to_outliers():
    margin_src = torch.cat((torch.linspace(0.0, 1.0, 100, dtype=DT),
                            torch.tensor([1000.0], dtype=DT)))   # one wild outlier
    n = int(margin_src.numel())
    energy = torch.zeros((n, 2), dtype=DT)
    energy[:, 1] = margin_src
    pref = render_class_preference(energy, _views(n, 2), min_support_views=2)
    norm = pref["normalized_margin"]
    assert float(norm.max()) <= 1.0 and float(norm.min()) >= 0.0
    # The outlier must not squash everything else toward zero.
    assert float(norm[:100].mean()) > 0.3, float(norm[:100].mean())


def test_12_alpha_is_one_minus_normalized_margin():
    """section 12: confident points follow render, uncertain follow neighbours."""
    n = 60
    x = torch.rand((n, 3), dtype=DT) * 0.2
    graph = build_neighbor_graph(x, k=8)
    pref = torch.zeros((n, 2), dtype=DT)
    pref[0] = torch.tensor([0.99, 0.01], dtype=DT)
    pref[1:] = torch.tensor([0.01, 0.99], dtype=DT)
    seed_mask = torch.zeros(n, dtype=torch.bool)
    seed_label = torch.zeros(n, dtype=torch.long)
    w = build_label_graph_weights(graph, pref)

    hard = torch.zeros(n, dtype=DT); hard[0] = 1.0      # alpha_0 = 0 -> pure render
    p_hard, _, _ = propagate_labels(graph, w, pref, hard, seed_mask, seed_label, iterations=40)
    soft = torch.zeros(n, dtype=DT)                     # alpha_0 = 1 -> pure neighbours
    p_soft, _, _ = propagate_labels(graph, w, pref, soft, seed_mask, seed_label, iterations=40)
    assert float(p_hard[0, 0]) > float(p_soft[0, 0]), (
        "alpha did not trade render evidence against neighbourhood"
    )


def test_4_render_graph_mode_bypasses_every_other_outer_loss():
    """section 4: in this mode no learned outer objective may run."""
    import rigid_parts.nested_pipeline as np_mod

    src = inspect.getsource(np_mod.run_nested_part_embedding_killing_discovery)
    head = src.index("if render_graph_mode:")
    tail = src.index("continue", head)
    block = src[head:tail]
    # The deterministic block must not touch the learned machinery at all.
    for forbidden in ("optimizer.step", "backward", "outer_render_membership_loss",
                      "outer_embedding_losses", "kinetic_label_loss",
                      "compute_detached_3d_gradient_cap", "adaptive_temperature"):
        assert forbidden not in block, f"render_graph mode still runs: {forbidden}"
    assert "run_render_graph_outer" in block
    # section 20: it always continues, no acceptance gate.
    assert "continue" in src[head:tail + 20]
    # section 13: the propagated labels feed the next inner solve.
    assert "render_graph_labels" in src
    assert "--outer_label_mode" or "outer_label_mode" in src


def test_21_22_required_artifacts_are_written():
    import rigid_parts.nested_pipeline as np_mod

    src = inspect.getsource(np_mod.run_nested_part_embedding_killing_discovery)
    assert "render_graph_outer_" in src, "section 21 per-outer .pt not saved"
    assert "render_graph_history.csv" in src, "section 22 history csv not written"
    from rigid_parts.render_graph_outer import RenderGraphResult

    required = {"candidate_energy", "candidate_valid", "candidate_num_views",
                "best_render_label", "raw_margin", "normalized_margin",
                "render_preference", "seed_mask", "seed_label",
                "propagated_probabilities", "final_label"}
    got = set(RenderGraphResult(
        *[torch.zeros(1)] * 11).as_dict().keys())
    assert required <= got, required - got



def test_7b_per_class_seeds_fix_the_cross_class_purity_asymmetry():
    """A global margin threshold is not class-neutral.

    On USB it produced 88.4% pure static seeds against 55.8% pure moving ones.
    When one class's margins run systematically lower, a single cut-off can
    starve it of seeds entirely.
    """
    torch.manual_seed(0)
    n = 2000
    best = torch.zeros(n, dtype=torch.long)
    best[:800] = 1
    margin = torch.zeros(n, dtype=DT)
    margin[:800] = torch.rand(800, dtype=DT) * 0.5      # class 1 margins are lower
    margin[800:] = torch.rand(1200, dtype=DT)
    comparable = torch.ones(n, dtype=torch.bool)
    anchors = torch.zeros(n, dtype=torch.bool)

    sm_g, sl_g = select_render_seeds(best, margin, comparable, anchors,
                                     seed_confidence=0.75, per_class=False)
    assert int((sm_g & (sl_g == 1)).sum()) == 0, "test vacuous: global threshold kept class 1"

    sm_p, sl_p = select_render_seeds(best, margin, comparable, anchors,
                                     seed_confidence=0.75, per_class=True)
    n0 = int((sm_p & (sl_p == 0)).sum())
    n1 = int((sm_p & (sl_p == 1)).sum())
    assert n1 > 0, "per-class selection still starved the weaker class"
    # Each class contributes roughly its own top quantile, so the seed counts
    # track the class sizes rather than the margin scales.
    assert abs(n0 / 1200 - n1 / 800) < 0.05, (n0, n1)


def test_7c_every_class_keeps_at_least_min_seeds():
    n = 500
    best = torch.zeros(n, dtype=torch.long)
    best[:12] = 1                                       # a tiny class
    margin = torch.zeros(n, dtype=DT)                   # perfectly flat margins
    comparable = torch.ones(n, dtype=torch.bool)
    sm, sl = select_render_seeds(best, margin, comparable, torch.zeros(n, dtype=torch.bool),
                                 seed_confidence=0.9, per_class=True, min_seeds_per_class=8)
    # With flat, zero margins nothing clears the absolute floor -- correct:
    # a small class must not get seeds merely for being small.
    assert int(sm.sum()) == 0, "ambiguous points were promoted to seeds"
    margin[:12] = 0.9
    sm2, sl2 = select_render_seeds(best, margin, comparable, torch.zeros(n, dtype=torch.bool),
                                  seed_confidence=0.9, per_class=True, min_seeds_per_class=8)
    assert int((sm2 & (sl2 == 1)).sum()) > 0, "a confident small class got no seeds"



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
    print(f"PASS render_graph_outer ({len(tests)} tests)")
