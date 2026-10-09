"""Endpoint-contrastive static/dynamic initialization: required tests A-I (section 27).

Plus checks for the parts of the spec whose failure would be silent: section 6's
non-random logit init, section 11's anchor-weight ordering, section 16's
artifacts, section 21's exact normalisation, and the round trip through
``PartEmbeddingModel`` that decides what the first Killing solve actually sees.
"""

from __future__ import annotations

import inspect
import sys
import tempfile
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from rigid_parts.endpoint_contrastive_init import (
    EPS,
    build_pseudo_anchors,
    coherence_loss,
    compute_endpoint_matching,
    count_connected_components,
    endpoint_edge_similarity,
    evaluate_binary_against_gt,
    initial_binary_logits,
    optimize_binary_dynamic,
    part_compactness,
    rank_percentile,
    run_endpoint_contrastive_initialization,
    separation_loss,
    split_dynamic_support,
    weighted_kmeans,
)
from rigid_parts.neighborhood import build_neighbor_graph

SEED = 20260919


def _two_blocks(n_static=120, n_dynamic=80, shift=0.6, spacing=0.05):
    """A static slab and a slab that translates by ``shift`` between endpoints."""
    generator = torch.Generator().manual_seed(SEED)
    static = torch.rand((n_static, 3), generator=generator) * torch.tensor([1.0, 1.0, 0.2])
    dynamic = torch.rand((n_dynamic, 3), generator=generator) * torch.tensor([1.0, 1.0, 0.2])
    dynamic[:, 2] += 1.0
    x0 = torch.cat((static, dynamic), dim=0)
    # Final cloud: the static slab is reproduced in place, the dynamic slab is
    # translated away, so nothing of it remains near its initial location.
    yT = torch.cat((static + spacing * 0.0, dynamic + torch.tensor([0.0, 0.0, float(shift)])), dim=0)
    gt = torch.cat((torch.zeros(n_static, dtype=torch.long),
                    torch.ones(n_dynamic, dtype=torch.long)))
    return x0, yT, gt


# ---------------------------------------------------------------------------
# A. Endpoint distance
# ---------------------------------------------------------------------------


def test_a_unchanged_points_have_smaller_endpoint_distance():
    x0, yT, gt = _two_blocks()
    m = compute_endpoint_matching(x0, yT)
    assert m.distance.shape == (x0.shape[0],), "d_i must be [N]"
    assert m.index.shape == (x0.shape[0],), "j*_i must be [N]"
    static_d = m.normalized[gt == 0]
    dynamic_d = m.normalized[gt == 1]
    assert float(static_d.max()) < float(dynamic_d.min()), (
        f"unchanged points must have smaller d_i: static max {float(static_d.max()):.6f} "
        f"vs dynamic min {float(dynamic_d.min()):.6f}")
    # The normalisation is by the initial bbox diagonal, so d_bar is scale free.
    # Tolerance 1e-3: this fixture reproduces the static block *exactly*, and
    # float32 cdist (the ||a||^2+||b||^2-2ab form) leaves ~2e-4 of bbox noise on
    # coincident points -- an order of magnitude under the trusted anchors' own
    # 2e-3 acceptance ratio, and three orders under the dynamic block's 0.42.
    scaled = compute_endpoint_matching(x0 * 7.0, yT * 7.0)
    assert torch.allclose(scaled.normalized, m.normalized, atol=1.0e-3), \
        "normalised endpoint distance is not scale invariant"


def test_a_nearest_index_is_the_actual_argmin():
    generator = torch.Generator().manual_seed(SEED)
    x0 = torch.rand((40, 3), generator=generator)
    yT = torch.rand((57, 3), generator=generator)
    m = compute_endpoint_matching(x0, yT)
    brute = torch.cdist(x0, yT)
    assert torch.equal(m.index, brute.argmin(dim=1)), "endpoint_nn_index is not the argmin"
    assert torch.allclose(m.distance, brute.min(dim=1).values, atol=1.0e-5), \
        "endpoint_nn_distance is not the min distance"


# ---------------------------------------------------------------------------
# B. Similar neighbours become similar
# ---------------------------------------------------------------------------


def test_b_similar_neighbours_are_pulled_together():
    x0, yT, _ = _two_blocks()
    m = compute_endpoint_matching(x0, yT)
    graph = build_neighbor_graph(x0, k=8)
    anchors = build_pseudo_anchors(m.normalized, None)

    logits0 = initial_binary_logits(m.normalized)
    p0 = torch.sigmoid(logits0)
    s, _ = endpoint_edge_similarity(graph, m.normalized)

    result = optimize_binary_dynamic(graph, m.normalized, anchors, iterations=200)
    p1 = result.p_dynamic

    # Restrict to the edges the spec calls "similar endpoint behaviour".
    similar = s > 0.9
    assert bool(similar.any()), "no similar edges in the fixture"
    before = (p0.index_select(0, graph.source[similar])
              - p0.index_select(0, graph.target[similar])).abs().mean()
    after = (p1.index_select(0, graph.source[similar])
             - p1.index_select(0, graph.target[similar])).abs().mean()
    assert float(after) < float(before), (
        f"optimization did not reduce |p_i - p_j| on similar edges: "
        f"{float(before):.6f} -> {float(after):.6f}")


def test_b_coherence_loss_is_minimised_by_equal_probabilities():
    x0, yT, _ = _two_blocks(n_static=30, n_dynamic=20)
    graph = build_neighbor_graph(x0, k=6)
    m = compute_endpoint_matching(x0, yT)
    s, _ = endpoint_edge_similarity(graph, m.normalized)
    flat = torch.full((x0.shape[0],), 0.5)
    varied = torch.rand(x0.shape[0], generator=torch.Generator().manual_seed(1))
    assert float(coherence_loss(flat, graph, s)) == 0.0, "constant p must give L_same = 0"
    assert float(coherence_loss(varied, graph, s)) > 0.0, "varying p must give L_same > 0"


# ---------------------------------------------------------------------------
# C. Dissimilar neighbours can separate
# ---------------------------------------------------------------------------


def test_c_dissimilar_neighbours_may_separate():
    x0, yT, gt = _two_blocks()
    m = compute_endpoint_matching(x0, yT)
    graph = build_neighbor_graph(x0, k=8)
    anchors = build_pseudo_anchors(m.normalized, None)
    s, _ = endpoint_edge_similarity(graph, m.normalized)

    result = optimize_binary_dynamic(graph, m.normalized, anchors, iterations=300,
                                     margin=0.5)
    p = result.p_dynamic
    dissimilar = s < 0.5
    similar = s > 0.9
    assert bool(dissimilar.any()), "no dissimilar edges in the fixture"
    gap_dissimilar = (p.index_select(0, graph.source[dissimilar])
                      - p.index_select(0, graph.target[dissimilar])).abs().mean()
    gap_similar = (p.index_select(0, graph.source[similar])
                   - p.index_select(0, graph.target[similar])).abs().mean()
    assert float(gap_dissimilar) > float(gap_similar), (
        f"dissimilar edges did not separate more than similar ones: "
        f"{float(gap_dissimilar):.6f} vs {float(gap_similar):.6f}")


def test_c_separation_margin_is_soft():
    """Beyond the margin the hinge is exactly zero -- no push to opposite labels."""
    x0, _, _ = _two_blocks(n_static=20, n_dynamic=20)
    graph = build_neighbor_graph(x0, k=5)
    s = torch.zeros(graph.num_edges)                     # every edge maximally dissimilar
    p = torch.zeros(x0.shape[0])
    p[x0[:, 2] > 0.5] = 0.6                              # a 0.6 gap across the boundary
    loss_at_gap = separation_loss(p, graph, s, margin=0.5)
    p_hard = torch.zeros_like(p)
    p_hard[x0[:, 2] > 0.5] = 1.0                         # forced to opposite hard labels
    loss_hard = separation_loss(p_hard, graph, s, margin=0.5)
    boundary = ((x0[graph.source, 2] > 0.5) != (x0[graph.target, 2] > 0.5))
    if bool(boundary.any()):
        assert float(loss_at_gap) <= float(loss_hard) + 1.0e-9, "margin is not monotone"
    # A pair already ``margin`` apart contributes nothing at all.
    two = torch.tensor([0.0, 0.5])
    tiny = build_neighbor_graph(torch.tensor([[0.0, 0.0, 0.0], [0.1, 0.0, 0.0]]), k=1)
    assert float(separation_loss(two, tiny, torch.zeros(tiny.num_edges), margin=0.5)) == 0.0, \
        "a pair exactly at the margin still contributes"


# ---------------------------------------------------------------------------
# D. Trusted static anchor
# ---------------------------------------------------------------------------


def test_d_trusted_static_anchor_stays_static():
    x0, yT, gt = _two_blocks()
    m = compute_endpoint_matching(x0, yT)
    graph = build_neighbor_graph(x0, k=8)

    # Trust a point that the weak rank init would have put on the dynamic side:
    # the highest-rank static point. Only the trusted anchor loss can save it.
    static_idx = torch.nonzero(gt == 0, as_tuple=False).reshape(-1)
    worst = static_idx[m.normalized[static_idx].argmax()]
    trusted = torch.zeros(x0.shape[0], dtype=torch.bool)
    trusted[worst] = True

    anchors = build_pseudo_anchors(m.normalized, trusted)
    assert bool(anchors.trusted_static[worst]), "the trusted mask was dropped"
    assert not bool(anchors.dynamic[worst]), \
        "a trusted static anchor must never also be a dynamic pseudo-anchor"

    result = optimize_binary_dynamic(graph, m.normalized, anchors, iterations=300)
    assert float(result.p_dynamic[worst]) <= 0.5, (
        f"trusted static anchor drifted dynamic: p = {float(result.p_dynamic[worst]):.4f}")
    assert int(result.binary_label[worst]) == 0, "trusted static anchor is not labelled static"


def test_d_trusted_anchors_outrank_geometric_ones():
    """Section 11's ordering must actually be applied, not merely documented."""
    x0, yT, _ = _two_blocks()
    m = compute_endpoint_matching(x0, yT)
    graph = build_neighbor_graph(x0, k=8)
    trusted = torch.zeros(x0.shape[0], dtype=torch.bool)
    trusted[m.normalized.argmax()] = True                # the single worst case
    anchors = build_pseudo_anchors(m.normalized, trusted)

    strong = optimize_binary_dynamic(graph, m.normalized, anchors, iterations=200,
                                     lambda_trusted_static=8.0)
    weak = optimize_binary_dynamic(graph, m.normalized, anchors, iterations=200,
                                   lambda_trusted_static=0.0)
    idx = int(m.normalized.argmax())
    assert float(strong.p_dynamic[idx]) < float(weak.p_dynamic[idx]), (
        "lambda_init_trusted_static has no effect on the trusted anchor: "
        f"{float(strong.p_dynamic[idx]):.4f} vs {float(weak.p_dynamic[idx]):.4f}")


# ---------------------------------------------------------------------------
# E. Dynamic pseudo-anchor orientation
# ---------------------------------------------------------------------------


def test_e_dynamic_pseudo_anchor_pushes_dynamic():
    x0, yT, gt = _two_blocks()
    m = compute_endpoint_matching(x0, yT)
    graph = build_neighbor_graph(x0, k=8)
    anchors = build_pseudo_anchors(m.normalized, None, dynamic_quantile=0.10)
    assert int(anchors.dynamic.sum()) > 0, "no dynamic pseudo-anchors were selected"

    with_anchor = optimize_binary_dynamic(graph, m.normalized, anchors, iterations=200,
                                          lambda_dynamic=2.0)
    without = optimize_binary_dynamic(graph, m.normalized, anchors, iterations=200,
                                      lambda_dynamic=0.0)
    mean_with = float(with_anchor.p_dynamic[anchors.dynamic].mean())
    mean_without = float(without.p_dynamic[anchors.dynamic].mean())
    assert mean_with > mean_without, (
        f"the dynamic pseudo-anchor loss did not push toward dynamic: "
        f"{mean_with:.4f} vs {mean_without:.4f}")
    assert mean_with > 0.5, f"dynamic pseudo-anchors did not reach the dynamic side ({mean_with:.4f})"


def test_e_dynamic_quantile_is_the_upper_tail():
    d = torch.linspace(0.0, 1.0, 101)
    anchors = build_pseudo_anchors(d, None, static_quantile=0.10, dynamic_quantile=0.10)
    assert float(anchors.tau_static) <= 0.11, f"tau_S is not Q_0.10: {float(anchors.tau_static)}"
    assert float(anchors.tau_dynamic) >= 0.89, f"tau_D is not Q_0.90: {float(anchors.tau_dynamic)}"
    assert bool(anchors.dynamic[-1]) and not bool(anchors.dynamic[0]), \
        "the dynamic pseudo-anchors are on the wrong tail"


# ---------------------------------------------------------------------------
# F. No constant collapse
# ---------------------------------------------------------------------------


def test_f_synthetic_regions_do_not_collapse_to_a_constant():
    x0, yT, gt = _two_blocks()
    m = compute_endpoint_matching(x0, yT)
    graph = build_neighbor_graph(x0, k=8)
    anchors = build_pseudo_anchors(m.normalized, None)
    result = optimize_binary_dynamic(graph, m.normalized, anchors, iterations=300)

    fraction = float(result.binary_label.float().mean())
    assert 0.001 < fraction < 0.999, f"binary field collapsed: f_D = {fraction:.6f}"
    assert result.warnings == [], f"unexpected collapse warning: {result.warnings}"
    spread = float(result.p_dynamic.max() - result.p_dynamic.min())
    assert spread > 0.1, f"p_dynamic is effectively constant (spread {spread:.6f})"
    # And it separated the two planted regions, not some arbitrary split.
    assert float(result.p_dynamic[gt == 1].mean()) > float(result.p_dynamic[gt == 0].mean()), \
        "the learned field does not align with the planted static/dynamic regions"


def test_f_collapse_is_reported_not_repaired():
    """Section 14: warn, and do NOT substitute a heuristic partition."""
    x0 = torch.rand((60, 3), generator=torch.Generator().manual_seed(3))
    graph = build_neighbor_graph(x0, k=6)
    d = torch.zeros(60)                                   # every point identical -> no contrast
    anchors = build_pseudo_anchors(d, torch.ones(60, dtype=torch.bool))
    result = optimize_binary_dynamic(graph, d, anchors, iterations=100)
    fraction = float(result.binary_label.float().mean())
    if fraction < 0.001 or fraction > 0.999:
        assert result.warnings, "a collapsed field produced no warning"
        assert torch.equal(result.binary_label, (result.p_dynamic > 0.5).long()), \
            "the collapsed label was silently replaced by a heuristic"


# ---------------------------------------------------------------------------
# G. K=1 bypass
# ---------------------------------------------------------------------------


def test_g_k1_does_not_call_kmeans():
    import rigid_parts.endpoint_contrastive_init as module

    calls = []
    original = module.weighted_kmeans
    module.weighted_kmeans = lambda *a, **k: (_ for _ in ()).throw(
        AssertionError("K-means was called with K == 1"))
    try:
        p = torch.tensor([0.1, 0.9, 0.2, 0.8])
        binary = (p > 0.5).long()
        split = module.split_dynamic_support(torch.rand((4, 3)), p, binary, 1)
    finally:
        module.weighted_kmeans = original

    assert split.kmeans_used is False, "K=1 reported using K-means"
    assert torch.equal(split.class_label, binary), "K=1 did not use the binary result directly"
    assert torch.allclose(split.membership, torch.stack((1.0 - p, p), dim=-1)), \
        "K=1 membership is not [1-p, p]"
    assert calls == []


# ---------------------------------------------------------------------------
# H. K>1 clusters only the predicted dynamic support
# ---------------------------------------------------------------------------


def test_h_static_points_never_enter_the_kmeans_fit():
    import rigid_parts.endpoint_contrastive_init as module

    xyz = torch.cat((
        torch.rand((50, 3), generator=torch.Generator().manual_seed(4)) * 0.1,   # static blob
        torch.rand((30, 3), generator=torch.Generator().manual_seed(5)) + 5.0,   # dynamic A
        torch.rand((30, 3), generator=torch.Generator().manual_seed(6)) - 5.0,   # dynamic B
    ), dim=0)
    p = torch.cat((torch.full((50,), 0.1), torch.full((60,), 0.9)))
    binary = (p > 0.5).long()

    seen = {}
    original = module.weighted_kmeans

    def spy(points, weights, k, **kwargs):
        seen["n"] = int(points.shape[0])
        seen["points"] = points.clone()
        return original(points, weights, k, **kwargs)

    module.weighted_kmeans = spy
    try:
        split = module.split_dynamic_support(xyz, p, binary, 2, seed=SEED)
    finally:
        module.weighted_kmeans = original

    assert seen["n"] == 60, f"K-means was fit on {seen['n']} points, not the 60 dynamic ones"
    assert float(seen["points"].norm(dim=-1).min()) > 1.0, \
        "a static point entered the moving-part K-means fit"
    assert split.kmeans_used is True
    # Static points keep hard label 0 regardless of which centroid is nearest.
    assert int((split.class_label[:50] != 0).sum()) == 0, "a static point got a moving label"
    assert set(split.class_label[50:].tolist()) == {1, 2}, \
        f"moving classes are not exactly 1..K: {sorted(set(split.class_label[50:].tolist()))}"
    # The two well-separated dynamic blobs must land in different clusters.
    assert len(set(split.class_label[50:80].tolist())) == 1, "dynamic blob A was split"
    assert len(set(split.class_label[80:].tolist())) == 1, "dynamic blob B was split"
    assert split.class_label[50] != split.class_label[80], "the two blobs merged"


def test_h_membership_is_exactly_normalised():
    """Section 21: 'Normalize exactly before passing downstream.'"""
    xyz = torch.rand((70, 3), generator=torch.Generator().manual_seed(7)) * 4.0
    p = torch.rand(70, generator=torch.Generator().manual_seed(8))
    binary = (p > 0.5).long()
    for k in (1, 2, 3):
        split = split_dynamic_support(xyz, p, binary, k, seed=SEED)
        total = split.membership.sum(dim=-1)
        assert torch.allclose(total, torch.ones_like(total), atol=1.0e-6), \
            f"K={k} membership rows do not sum to 1 (max err {float((total-1).abs().max()):.2e})"
        assert torch.allclose(split.membership[:, 0], 1.0 - p, atol=1.0e-6), \
            f"K={k}: P_i0 != 1 - p_dynamic"
        assert int(split.membership.shape[1]) == k + 1, f"K={k} membership has wrong width"
        assert bool((split.membership >= 0).all()), "negative membership"
        # The hard label must be the argmax of the membership it ships with.
        assert torch.equal(split.class_label, split.membership.argmax(dim=-1)), \
            f"K={k}: z^(0) disagrees with argmax of P"


# ---------------------------------------------------------------------------
# I. No ground truth
# ---------------------------------------------------------------------------


def test_i_initialization_runs_without_any_ground_truth():
    x0, yT, gt = _two_blocks()
    with tempfile.TemporaryDirectory() as tmp:
        init = run_endpoint_contrastive_initialization(
            x0, yT, 1, trusted_static_mask=None, result_dir=Path(tmp),
            gt_part_labels=None, log_fn=lambda *_: None)
    assert init.gt_evaluation["available"] is False, "GT was consulted when none was supplied"
    assert init.class_label.shape == (x0.shape[0],)
    assert init.membership.shape == (x0.shape[0], 2)


def test_i_ground_truth_changes_nothing_in_the_result():
    """GT must not influence loss, quantiles, the 0.5 threshold or K-means."""
    x0, yT, gt = _two_blocks()
    with tempfile.TemporaryDirectory() as tmp:
        without = run_endpoint_contrastive_initialization(
            x0, yT, 1, result_dir=Path(tmp) / "a", gt_part_labels=None,
            log_fn=lambda *_: None)
        with_gt = run_endpoint_contrastive_initialization(
            x0, yT, 1, result_dir=Path(tmp) / "b", gt_part_labels=gt,
            log_fn=lambda *_: None)
    assert torch.equal(without.class_label, with_gt.class_label), \
        "supplying ground truth changed the initialization"
    assert torch.allclose(without.p_dynamic, with_gt.p_dynamic), \
        "supplying ground truth changed p_dynamic"
    assert with_gt.gt_evaluation["available"] is True, "the GT diagnostic did not run"
    assert with_gt.gt_evaluation["f1"] > 50.0, \
        f"the planted fixture should be easy; got F1 {with_gt.gt_evaluation['f1']:.2f}"


# ---------------------------------------------------------------------------
# Section 6 -- the logits are an ordering, never random
# ---------------------------------------------------------------------------


def test_section6_logits_are_monotone_in_endpoint_distance():
    d = torch.tensor([0.4, 0.1, 0.9, 0.2])
    r = rank_percentile(d)
    assert torch.equal(r.argsort(), d.argsort()), "rank percentile is not order preserving"
    assert float(r.min()) == 0.0 and float(r.max()) == 1.0, "rank percentile is not in [0,1]"
    logits = initial_binary_logits(d)
    assert torch.equal(logits.argsort(), d.argsort()), "l^(0) is not monotone in d_bar"
    assert torch.isfinite(logits).all(), "l^(0) is not finite (the eps clamp is missing)"
    # Deterministic: two calls on the same input agree exactly (not random).
    assert torch.equal(initial_binary_logits(d), logits), "l^(0) is not deterministic"


def test_section7_sigma_d_is_the_edge_median():
    x0, yT, _ = _two_blocks()
    graph = build_neighbor_graph(x0, k=8)
    m = compute_endpoint_matching(x0, yT)
    s, sigma = endpoint_edge_similarity(graph, m.normalized)
    delta = (m.normalized.index_select(0, graph.source)
             - m.normalized.index_select(0, graph.target)).abs()
    assert torch.allclose(sigma, delta.median(), atol=1.0e-6), "sigma_d is not the edge median"
    assert float(s.min()) >= 0.0 and float(s.max()) <= 1.0, "s_ij left [0,1]"
    # s -> 1 for identical endpoint behaviour, s -> 0 for very different.
    assert float(s[delta.argmin()]) > float(s[delta.argmax()]), "s_ij has the wrong sense"


# ---------------------------------------------------------------------------
# Section 16 / 22 / 25 -- artifacts, compactness, visualisation
# ---------------------------------------------------------------------------


def test_section16_artifacts_carry_every_required_field():
    x0, yT, gt = _two_blocks()
    with tempfile.TemporaryDirectory() as tmp:
        init = run_endpoint_contrastive_initialization(
            x0, yT, 1, result_dir=Path(tmp), gt_part_labels=gt, log_fn=lambda *_: None)
        directory = Path(tmp) / "initialization"
        assert (directory / "endpoint_matching.pt").is_file(), "endpoint_matching.pt missing"
        assert (directory / "binary_init_history.csv").is_file(), "history csv missing"

        matching = torch.load(directory / "endpoint_matching.pt", weights_only=False)
        for key in ("endpoint_nn_distance", "endpoint_nn_index",
                    "endpoint_nn_distance_normalized"):
            assert key in matching, f"endpoint_matching.pt is missing {key}"

        state = torch.load(directory / "binary_dynamic_init.pt", weights_only=False)
        for key in ("endpoint_nn_distance", "endpoint_nn_distance_normalized",
                    "binary_logits", "p_dynamic", "binary_label",
                    "static_anchor_mask", "trusted_static_anchor_mask",
                    "dynamic_pseudo_anchor_mask", "graph_knn_indices",
                    "graph_spatial_weights", "graph_endpoint_similarity"):
            assert key in state, f"binary_dynamic_init.pt is missing {key}"
        assert int(state["graph_knn_indices"].shape[1]) == 2, "knn indices are not edge pairs"
        assert int(state["graph_spatial_weights"].numel()) == int(init.graph.num_edges)
        assert int(state["graph_endpoint_similarity"].numel()) == int(init.graph.num_edges)

        history = (directory / "binary_init_history.csv").read_text().splitlines()
        header = history[0].split(",")
        for column in ("loss_total", "loss_same", "loss_diff", "loss_static_anchor",
                       "loss_dynamic_anchor", "mean_p_dynamic", "fraction_p_dynamic_gt_0_5"):
            assert column in header, f"section 13 log column {column} is missing"
        assert len(history) - 1 >= 2, "the history has fewer than two logged iterations"

        for name in ("init_endpoint_distance.ply", "init_p_dynamic.ply",
                     "init_binary_label.ply", "init_class_label.ply"):
            assert (directory / name).is_file(), f"section 25 visualisation {name} missing"


def test_section25_k_above_one_colours_clusters_separately():
    from plyfile import PlyData

    x0, yT, _ = _two_blocks()
    with tempfile.TemporaryDirectory() as tmp:
        init = run_endpoint_contrastive_initialization(
            x0, yT, 3, result_dir=Path(tmp), log_fn=lambda *_: None)
        vertex = PlyData.read(str(Path(tmp) / "initialization" / "init_class_label.ply"))["vertex"]
        parts = torch.as_tensor(vertex["part"].astype("int64"))
        assert torch.equal(parts, init.class_label.cpu()), "the ply part field is not z^(0)"
        colours = {tuple(int(vertex[c][i]) for c in ("red", "green", "blue"))
                   for i in range(len(parts))}
        assert len(colours) == len(set(parts.tolist())), \
            f"{len(set(parts.tolist()))} classes got {len(colours)} colours"
        assert len(init.summary["moving_cluster_sizes"]) == 3, "section 24 lost a cluster size"
        assert len(init.split.diagnostics) == 3, "section 22 lost a part"


def test_section22_compactness_counts_components():
    xyz = torch.cat((torch.rand((30, 3), generator=torch.Generator().manual_seed(9)) * 0.1,
                     torch.rand((30, 3), generator=torch.Generator().manual_seed(10)) * 0.1 + 9.0),
                    dim=0)
    graph = build_neighbor_graph(xyz, k=4)
    z = torch.ones(60, dtype=torch.long)
    membership = torch.zeros((60, 2))
    membership[:, 1] = 1.0
    stats = part_compactness(xyz, z, membership, graph, 1)
    assert len(stats) == 1 and stats[0]["num_gaussians"] == 60
    assert stats[0]["num_connected_components"] == 2, (
        f"two separated blobs must be two components, got "
        f"{stats[0]['num_connected_components']}")
    assert stats[0]["mean_radius"] > 0.0 and all(e > 0 for e in stats[0]["bbox_extent"])
    single = count_connected_components(graph, torch.arange(60) < 30)
    assert single == 1, f"one blob must be one component, got {single}"


def test_kmeans_is_weighted_and_deterministic():
    xyz = torch.cat((torch.zeros((10, 3)), torch.ones((10, 3)) * 10.0), dim=0)
    w = torch.cat((torch.full((10,), 0.9), torch.full((10,), 0.9)))
    a, _ = weighted_kmeans(xyz, w, 2, seed=SEED)
    b, _ = weighted_kmeans(xyz, w, 2, seed=SEED)
    assert torch.allclose(a, b), "weighted K-means is not deterministic for a fixed seed"
    centres = sorted(float(c[0]) for c in a)
    assert abs(centres[0]) < 1.0e-4 and abs(centres[1] - 10.0) < 1.0e-4, \
        f"K-means did not recover the two blobs: {centres}"
    # A near-zero weight must barely move the centroid it belongs to.
    skewed = torch.cat((torch.full((10,), 1.0), torch.full((10,), 1.0e-6)))
    c, assign = weighted_kmeans(torch.cat((torch.zeros((10, 3)), torch.ones((10, 3)))),
                                skewed, 1, seed=SEED)
    assert float(c[0, 0]) < 0.01, f"p_dynamic weighting was ignored: centroid {float(c[0,0]):.4f}"


# ---------------------------------------------------------------------------
# Downstream contract: what the first Killing solve actually receives
# ---------------------------------------------------------------------------


def test_part_logits_reproduce_the_membership_through_the_model():
    from rigid_parts.embeddings import PartEmbeddingModel

    x0, yT, _ = _two_blocks()
    for k in (1, 3):
        init = run_endpoint_contrastive_initialization(
            x0, yT, k, log_fn=lambda *_: None)
        for temperature in (1.0, 0.5, 2.0):
            logits = init.part_logits(temperature)
            model = PartEmbeddingModel(int(x0.shape[0]), k + 1, 4,
                                       temperature=temperature, init_logits=logits)
            realised = model.probabilities(temperature)
            expected = init.membership.clamp_min(EPS)
            expected = expected / expected.sum(dim=-1, keepdim=True)
            assert torch.allclose(realised, expected, atol=1.0e-5), (
                f"K={k} tau={temperature}: the model does not reproduce the membership "
                f"(max err {float((realised-expected).abs().max()):.2e})")
            assert torch.equal(model.hard_labels(temperature), init.class_label), \
                f"K={k} tau={temperature}: the model's hard labels differ from z^(0)"


def test_gt_evaluation_matches_a_hand_computation():
    binary = torch.tensor([1, 1, 0, 0, 1])
    gt = torch.tensor([1, 0, 0, 1, 1])
    anchors = build_pseudo_anchors(torch.tensor([0.0, 0.1, 0.2, 0.3, 0.9]),
                                   torch.tensor([True, False, False, False, False]))
    report = evaluate_binary_against_gt(binary, gt, anchors)
    assert report["available"] is True
    assert abs(report["accuracy"] - 60.0) < 1.0e-4, report["accuracy"]      # 3 of 5
    assert abs(report["precision"] - 200.0 / 3.0) < 1.0e-4                   # tp2 fp1
    assert abs(report["recall"] - 200.0 / 3.0) < 1.0e-4                      # tp2 fn1
    assert abs(report["f1"] - 200.0 / 3.0) < 1.0e-4
    assert abs(report["moving_iou"] - 50.0) < 1.0e-4                         # 2 / 4
    assert report["trusted_static_anchor_purity"]["n"] == 1


def test_cli_defaults_match_the_spec():
    """Section 23: the flags exist with exactly the documented defaults."""
    from argparse import ArgumentParser

    from arguments import ModelParams, OptimizationParams, PipelineParams, add_dynamic_args

    parser = ArgumentParser()
    ModelParams(parser)
    OptimizationParams(parser)
    PipelineParams(parser)
    add_dynamic_args(parser)
    args = parser.parse_args([])
    for flag, expected in (
        ("init_mode", "legacy"),
        ("init_graph_k", 16),
        ("init_binary_iters", 300),
        ("init_binary_lr", 1.0e-2),
        ("init_static_quantile", 0.10),
        ("init_dynamic_quantile", 0.10),
        ("init_contrastive_margin", 0.5),
        ("lambda_init_same", 1.0),
        ("lambda_init_diff", 0.5),
        ("lambda_init_static", 1.0),
        ("lambda_init_dynamic", 0.25),
        ("lambda_init_trusted_static", 2.0),
    ):
        assert hasattr(args, flag), f"--{flag} is not registered"
        assert getattr(args, flag) == expected, \
            f"--{flag} default is {getattr(args, flag)}, spec says {expected}"
    picked = parser.parse_args(["--init_mode", "endpoint_contrastive"])
    assert picked.init_mode == "endpoint_contrastive", "the new mode is not selectable"


def test_anchor_weight_ordering_is_enforced():
    """Section 11's inequality must be a guard, not a comment."""
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
        # --no-outer_soft_labels: this fixture tests the §11 anchor-weight guard of
        # the HARD initializer, and the soft-label cycle has its own coupling
        # contract (render_graph + pixel matching) that is not under test here.
        parsed = parser.parse_args(
            ["--init_mode", "endpoint_contrastive", "--no-outer_soft_labels", *extra])
        # The §11 guard is on --init_mode alone; the nested-discovery coupling
        # (final_3dgs_path etc.) is a separate contract and not what is under test.
        return parsed

    ok = apply_dynamic_defaults(_args([]))
    assert ok.lambda_init_trusted_static >= ok.lambda_init_static >= ok.lambda_init_dynamic

    try:
        apply_dynamic_defaults(_args(["--lambda_init_trusted_static", "0.1"]))
    except ValueError as error:
        assert "§11" in str(error) or "lambda_init_trusted_static" in str(error)
    else:
        raise AssertionError("an inverted anchor-weight ordering was accepted")


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
    print(f"PASS endpoint_contrastive_init ({len(tests)} tests)")
