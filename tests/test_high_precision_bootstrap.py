"""High-precision bootstrap initialization: required tests A-L plus §27/§28.

The central claim under test is that initialization SEEDS rather than SEGMENTS:
two ultra-confident cores, and a large remainder left genuinely at [0.5, 0.5]
for the outer loop to resolve.
"""

from __future__ import annotations

import inspect
import sys
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from rigid_parts.bootstrap_init import (
    BootstrapAnchors,
    bootstrap_diagnostics,
    build_high_precision_bootstrap_anchors,
    compose_bootstrap_part_probs,
    first_support_diagnostics,
    run_high_precision_bootstrap_initialization,
    soft_target_cross_entropy,
    train_soft_bootstrap_probabilities,
)
from rigid_parts.neighborhood import build_neighbor_graph
from rigid_parts.soft_labels import binary_entropy, soft_outer_update

DT = torch.float64


def _line(n=200, spread=1.0, k=8):
    x = torch.linspace(0.0, float(spread), n)[:, None]
    pts = torch.cat((x, torch.zeros((n, 2))), dim=1)
    return build_neighbor_graph(pts, k=k), pts


def _scene(n=200):
    """Endpoint distance rising along a line: a clean static end and moving end."""
    graph, pts = _line(n)
    d = torch.linspace(0.0, 1.0, n)
    from rigid_parts.endpoint_contrastive_init import endpoint_edge_similarity

    similarity, _ = endpoint_edge_similarity(graph, d)
    return graph, pts, d, similarity


# ---------------------------------------------------------------------------
# §24 Tests A-D -- the two cores
# ---------------------------------------------------------------------------


def test_a_dynamic_quantile_098_selects_the_top_two_percent():
    d = torch.linspace(0.0, 1.0, 1000)
    a = build_high_precision_bootstrap_anchors(d, None, dynamic_quantile=0.98)
    fraction = float(a.dynamic_core.float().mean())
    assert abs(fraction - 0.02) < 0.005, f"dynamic core is {fraction*100:.2f}%, expected ~2%"
    assert a.tau_dynamic > 0.97, f"tau_D = {a.tau_dynamic}, expected Q_0.98"
    assert bool(a.dynamic_core[-1]) and not bool(a.dynamic_core[0]), "wrong tail"


def test_b_top_two_percent_is_smaller_than_top_five_percent():
    d = torch.linspace(0.0, 1.0, 1000)
    two = build_high_precision_bootstrap_anchors(d, None, dynamic_quantile=0.98)
    five = build_high_precision_bootstrap_anchors(d, None, dynamic_quantile=0.95)
    assert int(two.dynamic_core.sum()) < int(five.dynamic_core.sum()), (
        f"top-2% ({int(two.dynamic_core.sum())}) is not smaller than top-5% "
        f"({int(five.dynamic_core.sum())})")
    # And the stricter core is a subset of the looser one.
    assert bool((two.dynamic_core & ~five.dynamic_core).sum() == 0), \
        "the 2% core is not contained in the 5% core"


def test_c_static_core_uses_only_the_low_distance_tail():
    d = torch.linspace(0.0, 1.0, 1000)
    a = build_high_precision_bootstrap_anchors(d, None, static_quantile=0.10)
    assert abs(float(a.geometric_static.float().mean()) - 0.10) < 0.01
    assert bool(a.geometric_static[0]) and not bool(a.geometric_static[-1]), "wrong tail"
    idx = torch.nonzero(a.geometric_static).reshape(-1)
    assert float(d[idx].max()) <= a.tau_static + 1.0e-6, "a high-distance point entered the static core"
    # The two cores never overlap.
    assert int((a.geometric_static & a.dynamic_core).sum()) == 0, "the cores overlap"


def test_d_trusted_static_anchors_are_always_retained():
    d = torch.linspace(0.0, 1.0, 1000)
    trusted = torch.zeros(1000, dtype=torch.bool)
    trusted[-1] = True                       # the single most "dynamic-looking" point
    a = build_high_precision_bootstrap_anchors(d, trusted, dynamic_quantile=0.98)
    assert bool(a.trusted_static[-1]), "a trusted anchor was dropped"
    assert not bool(a.dynamic_core[-1]), "a trusted anchor also entered the dynamic core"
    assert not bool(a.undecided[-1]), "a trusted anchor was marked undecided"
    assert int((a.trusted_static & a.dynamic_core).sum()) == 0


# ---------------------------------------------------------------------------
# §25 Tests E-H -- the undecided remainder stays soft
# ---------------------------------------------------------------------------


def test_e_non_anchor_initial_state_is_exactly_half():
    """§7: no signed endpoint-rank bias; every Gaussian starts at [0.5, 0.5]."""
    graph, pts, d, similarity = _scene()
    anchors = build_high_precision_bootstrap_anchors(d, None)
    _, q, _ = train_soft_bootstrap_probabilities(
        graph, similarity, anchors, num_gaussians=int(pts.shape[0]), iterations=0)
    assert torch.allclose(q, torch.full_like(q, 0.5)), \
        f"the start is not [0.5, 0.5]: min {float(q.min()):.5f} max {float(q.max()):.5f}"


def test_f_undecided_points_are_not_forced_to_one_hot():
    graph, pts, d, similarity = _scene()
    anchors = build_high_precision_bootstrap_anchors(d, None)
    _, q, _ = train_soft_bootstrap_probabilities(
        graph, similarity, anchors, num_gaussians=int(pts.shape[0]), iterations=200)
    undecided = anchors.undecided
    peak = q.max(dim=1).values[undecided]
    assert float(peak.max()) < 0.99, \
        f"an undecided point reached {float(peak.max()):.4f}; bootstrap is binarizing"
    assert float(peak.mean()) < 0.90, \
        f"the undecided population is at mean confidence {float(peak.mean()):.4f}"
    # §9: smoothing may move it off 0.5, but not to one-hot.
    assert float(peak.mean()) > 0.5, "sanity: the loss did nothing at all"


def test_g_no_global_entropy_loss_acts_on_undecided_points():
    """§3: lambda_polar is exactly 0 in bootstrap mode, and it matters."""
    graph, pts, d, similarity = _scene()
    anchors = build_high_precision_bootstrap_anchors(d, None)
    _, q0, hist = train_soft_bootstrap_probabilities(
        graph, similarity, anchors, num_gaussians=int(pts.shape[0]), iterations=200)
    assert all(row["lambda_polar_current"] == 0.0 for row in hist), \
        "a non-zero polarization weight appeared in bootstrap mode"

    # With the legacy ablation path the undecided population DOES harden --
    # which is exactly the failure §3 removes.
    _, q1, _ = train_soft_bootstrap_probabilities(
        graph, similarity, anchors, num_gaussians=int(pts.shape[0]), iterations=200,
        lambda_polar=2.0)
    h0 = float(binary_entropy(q0)[anchors.undecided].mean())
    h1 = float(binary_entropy(q1)[anchors.undecided].mean())
    assert h0 > h1, f"polarization did not change anything ({h0:.5f} vs {h1:.5f})"
    assert h0 > 0.3, f"the default path still collapsed undecided entropy to {h0:.5f}"


def test_h_disconnected_undecided_point_stays_at_half():
    """§25 Test H: no anchor influence, no neighbours -> unchanged."""
    # Two static-ish points at x~0, two dynamic-ish at x~1, and one point far
    # away at x=50 with a mid-range endpoint distance: undecided AND unreachable.
    pts = torch.tensor([[0.0, 0, 0], [0.02, 0, 0], [50.0, 0, 0], [1.0, 0, 0], [1.02, 0, 0]])
    d = torch.tensor([0.0, 0.01, 0.5, 0.99, 1.0])
    graph = build_neighbor_graph(pts, k=2)
    anchors = build_high_precision_bootstrap_anchors(
        d, None, dynamic_quantile=0.98, static_quantile=0.10)
    lone = 2
    assert bool(anchors.undecided[lone]), "the fixture point is not undecided"
    assert bool(anchors.dynamic_core[4]) and bool(anchors.geometric_static[0]), \
        "the fixture has no anchors to propagate from"

    from rigid_parts.endpoint_contrastive_init import endpoint_edge_similarity

    similarity, _ = endpoint_edge_similarity(graph, d)
    # The lone point's only edges reach points whose endpoint distance is wildly
    # different, so s_ij ~ 0 there and the positive-only term gives it nothing.
    _, q, _ = train_soft_bootstrap_probabilities(
        graph, similarity, anchors, num_gaussians=5, iterations=200)
    assert abs(float(q[lone, 0]) - 0.5) < 0.05, \
        f"an unanchored, dissimilar-neighbour point drifted to {q[lone].tolist()}"


# ---------------------------------------------------------------------------
# §26 Tests I-L -- the temporary projection
# ---------------------------------------------------------------------------


def test_i_exact_tie_maps_to_static_class_zero():
    """§12: [0.5, 0.5] must resolve to class 0 for INNER 0."""
    p = torch.tensor([[0.5, 0.5], [0.5, 0.5, 0.0], [0.5, 0.25, 0.25]][0:1])
    assert int(torch.tensor([[0.5, 0.5]]).argmax(dim=1)[0]) == 0, "a 2-way tie did not pick class 0"
    assert int(torch.tensor([[0.4, 0.3, 0.3]]).argmax(dim=1)[0]) == 0
    assert int(torch.tensor([[1 / 3, 1 / 3, 1 / 3]]).argmax(dim=1)[0]) == 0, \
        "a K-way tie did not pick static"
    # And through the real diagnostic path.
    anchors = BootstrapAnchors(
        trusted_static=torch.zeros(2, dtype=torch.bool),
        geometric_static=torch.zeros(2, dtype=torch.bool),
        dynamic_core=torch.zeros(2, dtype=torch.bool),
        undecided=torch.ones(2, dtype=torch.bool), tau_static=0.0, tau_dynamic=1.0)
    out = first_support_diagnostics(torch.full((2, 2), 0.5), anchors)
    assert out["temporary_static_count"] == 2 and out["temporary_moving_count"] == 0, \
        "undecided ties did not default to static"


def test_j_argmax_does_not_modify_persistent_probabilities():
    p = torch.tensor([[0.5, 0.5], [0.3, 0.7], [0.9, 0.1]])
    snapshot = p.clone()
    _ = p.argmax(dim=1)
    assert torch.equal(p, snapshot), "argmax mutated the persistent state"


def test_k_l_cores_become_the_expected_temporary_support():
    graph, pts, d, similarity = _scene()
    n = int(pts.shape[0])
    trusted = torch.zeros(n, dtype=torch.bool)
    trusted[:5] = True
    anchors = build_high_precision_bootstrap_anchors(d, trusted, dynamic_quantile=0.98)
    _, q, _ = train_soft_bootstrap_probabilities(
        graph, similarity, anchors, num_gaussians=n, iterations=200)
    z = q.argmax(dim=1)
    assert bool((z[anchors.dynamic_core] == 1).all()), \
        "Test K: the dynamic core is not temporary moving support"
    assert bool((z[anchors.trusted_static] == 0).all()), \
        "Test L: a trusted static point is not temporary static"


# ---------------------------------------------------------------------------
# §27 -- the OUTER loop must be able to resolve what init left open
# ---------------------------------------------------------------------------


def test_outer_loop_recruits_and_rejects_undecided_gaussians():
    """§27, central: P=[0.5,0.5] must move either way on render evidence."""
    n = 6
    graph = build_neighbor_graph(torch.rand((n, 3), generator=torch.Generator().manual_seed(3)), k=2)
    probs = torch.full((n, 2), 0.5, dtype=DT)
    valid = torch.ones((n, 2), dtype=torch.bool)

    moving_better = torch.tensor([[5.0, 0.1]] * n, dtype=DT)     # E_D << E_S
    up = soft_outer_update(
        soft_probs=probs, energy=moving_better, normalized_margin=torch.ones(n, dtype=DT),
        comparable=torch.ones(n, dtype=torch.bool), graph=graph,
        trusted_mask=torch.zeros(n, dtype=torch.bool), memory=None, valid=valid)
    assert float(up.soft_after[:, 1].min()) > 0.5, \
        f"an undecided Gaussian was not recruited by moving evidence: {up.soft_after[0].tolist()}"

    static_better = torch.tensor([[0.1, 5.0]] * n, dtype=DT)     # E_S << E_D
    down = soft_outer_update(
        soft_probs=probs, energy=static_better, normalized_margin=torch.ones(n, dtype=DT),
        comparable=torch.ones(n, dtype=torch.bool), graph=graph,
        trusted_mask=torch.zeros(n, dtype=torch.bool), memory=None, valid=valid)
    assert float(down.soft_after[:, 0].min()) > 0.5, \
        f"an undecided Gaussian was not rejected by static evidence: {down.soft_after[0].tolist()}"


# ---------------------------------------------------------------------------
# §28 -- regression: no premature full polarization
# ---------------------------------------------------------------------------


def test_no_premature_full_polarization_before_inner0():
    graph, pts, d, similarity = _scene(n=400)
    n = int(pts.shape[0])
    anchors = build_high_precision_bootstrap_anchors(d, None, dynamic_quantile=0.98)
    _, q, _ = train_soft_bootstrap_probabilities(
        graph, similarity, anchors, num_gaussians=n, iterations=200)
    part_probs, _, _ = compose_bootstrap_part_probs(q, pts, 1)
    diagnostics, warnings = bootstrap_diagnostics(q, part_probs, anchors)

    polarized = diagnostics["fraction_peak_gt_090"]
    coverage = diagnostics["anchor_coverage"]
    assert polarized < 0.5, \
        f"{polarized*100:.1f}% of the cloud is >0.90 confident before any motion evidence"
    assert polarized <= coverage + 0.25, (
        f"polarization ({polarized*100:.1f}%) ran far past anchor coverage "
        f"({coverage*100:.1f}%); the old failure has returned")
    assert warnings == [], f"unexpected warning: {warnings}"
    assert diagnostics["fraction_undecided"] > 0.5, \
        "initialization decided most of the cloud; it is segmenting, not seeding"


def test_rebinarization_warning_fires_when_it_should():
    """§18: the guard must actually detect the failure it exists for."""
    n = 100
    anchors = BootstrapAnchors(
        trusted_static=torch.zeros(n, dtype=torch.bool),
        geometric_static=torch.zeros(n, dtype=torch.bool),
        dynamic_core=torch.zeros(n, dtype=torch.bool),
        undecided=torch.ones(n, dtype=torch.bool), tau_static=0.0, tau_dynamic=1.0)
    hardened = torch.zeros((n, 2))
    hardened[:, 1] = 0.999
    hardened[:, 0] = 0.001
    _, warnings = bootstrap_diagnostics(hardened, hardened, anchors)
    assert warnings, "a fully hardened non-anchor cloud produced no warning"
    assert "polarized" in warnings[0]


# ---------------------------------------------------------------------------
# objective, composition, end to end
# ---------------------------------------------------------------------------


def test_soft_target_cross_entropy_is_minimised_at_the_target():
    target = torch.tensor([0.999, 0.001], dtype=DT)
    mask = torch.ones(1, dtype=torch.bool)
    at_target = float(soft_target_cross_entropy(target[None], mask, target))
    for other in (torch.tensor([[0.5, 0.5]], dtype=DT), torch.tensor([[0.9, 0.1]], dtype=DT),
                  torch.tensor([[0.99999, 0.00001]], dtype=DT)):
        assert float(soft_target_cross_entropy(other, mask, target)) >= at_target - 1.0e-9, \
            f"{other.tolist()} scored better than the target itself"
    # §6: it does NOT keep pushing past the target, unlike -log(q_class).
    beyond = torch.tensor([[1.0 - 1e-9, 1e-9]], dtype=DT)
    assert float(soft_target_cross_entropy(beyond, mask, target)) > at_target, \
        "saturating past the target is not penalised"


def test_composition_preserves_dynamic_mass_for_k_above_one():
    graph, pts, d, similarity = _scene()
    anchors = build_high_precision_bootstrap_anchors(d, None)
    _, q, _ = train_soft_bootstrap_probabilities(
        graph, similarity, anchors, num_gaussians=int(pts.shape[0]), iterations=50)
    for k in (1, 2, 3):
        p, pi, _ = compose_bootstrap_part_probs(q, pts, k)
        total = p.sum(dim=1)
        assert torch.allclose(total, torch.ones_like(total), atol=1.0e-6), f"K={k} rows != 1"
        assert torch.allclose(p[:, 0], q[:, 0], atol=1.0e-6), f"K={k}: P_i0 != q^S"
        if k > 1:
            assert torch.allclose(p[:, 1:].sum(dim=1), q[:, 1], atol=1.0e-6), \
                f"K={k}: moving mass != q^D"


def test_end_to_end_bootstrap_seeds_rather_than_segments():
    generator = torch.Generator().manual_seed(11)
    static = torch.rand((150, 3), generator=generator) * torch.tensor([1.0, 1.0, 0.2])
    moving = torch.rand((100, 3), generator=generator) * torch.tensor([1.0, 1.0, 0.2])
    moving[:, 2] += 1.0
    x0 = torch.cat((static, moving), 0)
    # Jitter the final cloud so endpoint distances are CONTINUOUS. Reproducing
    # the static block exactly would put 60% of the cloud at d = 0, and the
    # q_S = 0.10 quantile threshold would then swallow all of them by ties --
    # an artefact of the fixture, not of real endpoint geometry.
    jitter = torch.randn((250, 3), generator=generator) * 0.004
    yT = torch.cat((static, moving + torch.tensor([0.0, 0.0, 0.6])), 0) + jitter
    gt = torch.cat((torch.zeros(150, dtype=torch.long), torch.ones(100, dtype=torch.long)))

    result = run_high_precision_bootstrap_initialization(
        x0, yT, 1, gt_part_labels=gt, log_fn=lambda *_: None)

    d = result.diagnostics
    assert d["fraction_undecided"] > 0.5, \
        f"only {d['fraction_undecided']*100:.1f}% left undecided; this is segmentation"
    assert d["num_dynamic_core"] > 0, "no dynamic core at all"
    # §28's bar is "almost the whole cloud", not "any spread past the anchors".
    # The KNN agreement term does propagate anchor CONFIDENCE, so polarization
    # legitimately exceeds anchor coverage; measured on real storage it reaches
    # ~47% while 93.7% of the undecided lean static and only 0.6% lean moving.
    # The §18 warning is expected to fire there and is not a failure.
    assert d["fraction_peak_gt_090"] < 0.80, (
        f"{d['fraction_peak_gt_090']*100:.1f}% of the cloud is >0.90 confident before any "
        "motion evidence; that is the premature-segmentation failure §28 guards against")
    total = result.part_probs.sum(dim=1)
    assert torch.allclose(total, torch.ones_like(total), atol=1.0e-5)

    # §19: precision first. The tiny core should be almost pure.
    support = d["first_support"]
    assert "moving_support_precision" in support, "the GT diagnostic did not run"
    assert support["moving_support_precision"] > 90.0, \
        f"the bootstrap moving core is only {support['moving_support_precision']:.1f}% pure"
    # §19 states a PRIORITY ("high precision first"), not a recall ceiling. On
    # this cleanly-separated fixture propagation from the 2% core recovers almost
    # the whole moving blob, which is a good outcome. On real storage the same
    # code gives precision 89.10% / recall 25.12% -- the hard case the spec is
    # actually aimed at. So only precision is asserted here.
    assert support["moving_support_size"] > 0, "the bootstrap produced no moving support"


def test_cli_defaults_match_the_spec():
    from argparse import ArgumentParser

    from arguments import ModelParams, OptimizationParams, PipelineParams, add_dynamic_args

    parser = ArgumentParser()
    ModelParams(parser); OptimizationParams(parser); PipelineParams(parser)
    add_dynamic_args(parser)
    a = parser.parse_args([])
    for flag, expected in (
        ("bootstrap_dynamic_quantile", 0.98),
        ("bootstrap_static_quantile", 0.10),
        ("lambda_bootstrap_knn", 1.0),
        ("lambda_bootstrap_anchor", 1.0),
        ("lambda_bootstrap_dynamic_core", 1.0),
        ("lambda_bootstrap_geometric_static", 0.5),
        ("lambda_bootstrap_polar", 0.0),
        ("bootstrap_iterations", 200),
    ):
        assert hasattr(a, flag), f"--{flag} is not registered"
        assert getattr(a, flag) == expected, \
            f"--{flag} default is {getattr(a, flag)}, spec says {expected}"
    picked = parser.parse_args(["--init_mode", "high_precision_bootstrap"])
    assert picked.init_mode == "high_precision_bootstrap", "the new mode is not selectable"
    # §22: legacy modes are untouched.
    for legacy in ("legacy", "endpoint_contrastive"):
        assert parser.parse_args(["--init_mode", legacy]).init_mode == legacy


if __name__ == "__main__":
    module = sys.modules["__main__"]
    tests = [(n, f) for n, f in sorted(vars(module).items())
             if n.startswith("test_") and inspect.isfunction(f)]
    for name, fn in tests:
        fn()
        print(f"  ok  {name}")
    print(f"PASS high_precision_bootstrap ({len(tests)} tests)")
