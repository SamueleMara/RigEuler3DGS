"""Render-response contrastive outer loss: required tests A-Q (§38-§44)."""

from __future__ import annotations

import inspect
import sys
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from rigid_parts.neighborhood import build_neighbor_graph
from rigid_parts.render_contrastive import (
    build_render_contrast_graph,
    build_render_response_fingerprints,
    centred_update,
    preserve_loss,
    refine_soft_labels_with_render_contrast,
    render_contrastive_losses,
)

DT = torch.float64


def _pair_graph(n=2, spacing=0.1):
    pts = torch.arange(n, dtype=torch.float32)[:, None] * spacing
    pts = torch.cat((pts, torch.zeros((n, 2))), dim=1)
    return build_neighbor_graph(pts, k=max(1, n - 1))


def _fingerprints(energy, valid=None, tau_rho=1.0):
    e = torch.as_tensor(energy, dtype=DT)
    v = torch.ones(e.shape[0], e.shape[1], e.shape[2], dtype=torch.bool) if valid is None \
        else torch.as_tensor(valid)
    return build_render_response_fingerprints(e, v, num_classes=int(e.shape[1]), tau_rho=tau_rho)


# ---------------------------------------------------------------------------
# §38 Tests A-D -- fingerprint similarity
# ---------------------------------------------------------------------------


def test_a_identical_fingerprints_give_cosine_one():
    # [N=2, K+1=2, V=4]; r = E_S - E_D identical for both Gaussians.
    e = torch.tensor([[[3.0, 1.0, 4.0, 2.0], [0.0, 0.0, 0.0, 0.0]],
                      [[3.0, 1.0, 4.0, 2.0], [0.0, 0.0, 0.0, 0.0]]], dtype=DT)
    f = _fingerprints(e)
    cos = float((f.vectors[0] * f.vectors[1]).sum())
    assert abs(cos - 1.0) < 1.0e-9, f"identical fingerprints gave cosine {cos}"
    g = build_render_contrast_graph(f, _pair_graph())
    assert g.num_positive == 1 and g.num_negative == 0, "no positive edge for identical response"


def test_b_opposite_fingerprints_give_cosine_minus_one():
    e = torch.tensor([[[3.0, 1.0, 4.0, 2.0], [0.0, 0.0, 0.0, 0.0]],
                      [[0.0, 0.0, 0.0, 0.0], [3.0, 1.0, 4.0, 2.0]]], dtype=DT)
    f = _fingerprints(e)
    cos = float((f.vectors[0] * f.vectors[1]).sum())
    assert abs(cos + 1.0) < 1.0e-9, f"opposite fingerprints gave cosine {cos}"
    g = build_render_contrast_graph(f, _pair_graph())
    assert g.num_negative == 1 and g.num_positive == 0, "no negative edge for opposite response"


def test_c_uncorrelated_fingerprints_create_no_pair():
    """§9: 'not similar' must never be read as 'opposite'."""
    e = torch.tensor([[[1.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]],
                      [[0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]]], dtype=DT)
    f = _fingerprints(e)
    cos = float((f.vectors[0] * f.vectors[1]).sum())
    assert abs(cos) < 1.0e-9, f"the fixture is not orthogonal (cos {cos})"
    g = build_render_contrast_graph(f, _pair_graph())
    assert g.num_positive == 0 and g.num_negative == 0, \
        "an orthogonal pair produced a contrastive edge"
    assert g.diagnostics["num_ambiguous"] >= 1


def test_d_insufficient_shared_observations_create_no_pair():
    e = torch.tensor([[[3.0, 1.0, 4.0, 2.0], [0.0, 0.0, 0.0, 0.0]],
                      [[3.0, 1.0, 4.0, 2.0], [0.0, 0.0, 0.0, 0.0]]], dtype=DT)
    valid = torch.ones((2, 2, 4), dtype=torch.bool)
    valid[0, :, 1:] = False          # Gaussian 0 keeps only observation 0
    valid[1, :, :1] = False          # Gaussian 1 loses observation 0 -> 0 shared
    f = _fingerprints(e, valid)
    g = build_render_contrast_graph(f, _pair_graph(), min_shared_obs=2)
    assert g.num_positive == 0 and g.num_negative == 0, \
        "a pair with no shared observations still produced an edge"
    assert g.diagnostics["num_insufficient_shared_obs"] >= 1


# ---------------------------------------------------------------------------
# §39 Tests E-G -- response confidence
# ---------------------------------------------------------------------------


def test_e_near_zero_response_gives_negligible_weight():
    strong = torch.tensor([[[3.0, 1.0, 4.0, 2.0], [0.0, 0.0, 0.0, 0.0]]] * 2, dtype=DT)
    weak = strong * 1.0e-4
    gs = build_render_contrast_graph(_fingerprints(strong), _pair_graph())
    gw = build_render_contrast_graph(_fingerprints(weak), _pair_graph())
    assert gs.num_positive == 1, "the strong fixture produced no edge"
    ws = float(gs.positive_weight.sum())
    ww = float(gw.positive_weight.sum()) if gw.num_positive else 0.0
    assert ww < 0.01 * ws, f"a near-zero response still carried weight {ww} vs {ws}"


def test_f_stronger_responses_give_larger_weights():
    base = torch.tensor([[[3.0, 1.0, 4.0, 2.0], [0.0, 0.0, 0.0, 0.0]]] * 2, dtype=DT)
    # ||r|| for the unscaled fixture is sqrt(30) ~ 5.48, so anything above
    # scale ~0.18 saturates rho at 1 (§8 clips it). Stay below that.
    weights = []
    for scale in (0.02, 0.06, 0.15):
        g = build_render_contrast_graph(_fingerprints(base * scale), _pair_graph())
        weights.append(float(g.positive_weight.sum()) if g.num_positive else 0.0)
    assert weights[0] < weights[1] < weights[2], f"weights not monotone in response: {weights}"
    # And above the clip they plateau, which is the intended bounded behaviour.
    hi = [float(build_render_contrast_graph(_fingerprints(base * s), _pair_graph()).positive_weight.sum())
          for s in (1.0, 10.0)]
    assert abs(hi[0] - hi[1]) < 1.0e-9, "response confidence is not clipped at 1"


def test_g_no_valid_evidence_gives_zero_confidence():
    e = torch.tensor([[[3.0, 1.0], [0.0, 0.0]], [[3.0, 1.0], [0.0, 0.0]]], dtype=DT)
    valid = torch.zeros((2, 2, 2), dtype=torch.bool)
    f = _fingerprints(e, valid)
    assert float(f.confidence.max()) == 0.0, "a Gaussian with no evidence kept confidence"
    g = build_render_contrast_graph(f, _pair_graph())
    assert g.num_positive == 0 and g.num_negative == 0, "zero-confidence Gaussians formed edges"


# ---------------------------------------------------------------------------
# §40, §41 -- agreement and disagreement losses
# ---------------------------------------------------------------------------


def _two_node_graph(positive: bool, weight=1.0):
    src = torch.tensor([0]); dst = torch.tensor([1])
    w = torch.tensor([float(weight)], dtype=DT)
    empty = torch.zeros(0, dtype=torch.long)
    from rigid_parts.render_contrastive import RenderContrastGraph

    if positive:
        return RenderContrastGraph(src, dst, w, empty, empty, w.new_zeros(0),
                                   torch.ones(2, dtype=DT), {})
    return RenderContrastGraph(empty, empty, w.new_zeros(0), src, dst, w,
                               torch.ones(2, dtype=DT), {})


def _logits_from_delta(d0, d1):
    """Binary logits whose centred update equals the requested delta."""
    old = torch.zeros((2, 2), dtype=DT)
    ref = torch.tensor([[-d0 / 2, d0 / 2], [-d1 / 2, d1 / 2]], dtype=DT)
    return ref, old


def test_agreement_loss_prefers_same_direction():
    """§40."""
    g = _two_node_graph(positive=True)
    same = render_contrastive_losses(*_logits_from_delta(0.5, 0.5), g)["agree"]
    opposite = render_contrastive_losses(*_logits_from_delta(0.5, -0.5), g)["agree"]
    assert float(same) < float(opposite), f"agree: same {float(same)} !< opposite {float(opposite)}"
    both_static = render_contrastive_losses(*_logits_from_delta(-0.5, -0.5), g)["agree"]
    assert float(both_static) < float(opposite), "two static-directed updates were not preferred"
    assert abs(float(same) - float(both_static)) < 1.0e-12, \
        "the loss is not symmetric in update direction"


def test_disagreement_loss_prefers_opposite_direction():
    """§41."""
    g = _two_node_graph(positive=False)
    opposite = render_contrastive_losses(*_logits_from_delta(0.5, -0.5), g)["disagree"]
    same = render_contrastive_losses(*_logits_from_delta(0.5, 0.5), g)["disagree"]
    assert float(opposite) < float(same), \
        f"disagree: opposite {float(opposite)} !< same {float(same)}"


def test_absent_negative_edges_create_no_repulsive_gradient():
    """§41/§48: weak or absent disagreement must never repel."""
    from rigid_parts.render_contrastive import RenderContrastGraph

    empty = torch.zeros(0, dtype=torch.long)
    g = RenderContrastGraph(empty, empty, torch.zeros(0, dtype=DT),
                            empty, empty, torch.zeros(0, dtype=DT),
                            torch.ones(2, dtype=DT), {})
    ref, old = _logits_from_delta(0.5, 0.5)
    ref = ref.clone().requires_grad_(True)
    loss = render_contrastive_losses(ref, old, g)["disagree"]
    assert float(loss) == 0.0, "an empty negative set produced a loss"
    loss.backward()
    assert float(ref.grad.abs().max()) == 0.0, "an empty negative set produced a gradient"


# ---------------------------------------------------------------------------
# §42, §43 -- memory untouched, base preserved
# ---------------------------------------------------------------------------


def _refine(lambda_agree=1.0, lambda_disagree=0.25, steps=10, positive=True, trusted=None):
    base = torch.tensor([[0.6, 0.4], [0.55, 0.45]], dtype=DT)
    old = torch.tensor([[0.5, 0.5], [0.5, 0.5]], dtype=DT)
    return refine_soft_labels_with_render_contrast(
        old_probs=old, base_probs=base, contrast_graph=_two_node_graph(positive),
        trusted_static_mask=trusted, num_steps=steps,
        lambda_agree=lambda_agree, lambda_disagree=lambda_disagree)


def test_h_l_disabled_contrast_returns_the_base_exactly():
    """§42 Test H and §43 Test L."""
    zero_weights = _refine(lambda_agree=0.0, lambda_disagree=0.0)
    assert torch.equal(zero_weights.probs, zero_weights.base_probs), \
        "zero contrast weights did not return P_base exactly"
    assert zero_weights.diagnostics["applied"] is False

    no_steps = _refine(steps=0)
    assert torch.equal(no_steps.probs, no_steps.base_probs), "zero steps altered P_base"

    from rigid_parts.render_contrastive import RenderContrastGraph

    empty = torch.zeros(0, dtype=torch.long)
    no_edges = refine_soft_labels_with_render_contrast(
        old_probs=torch.full((2, 2), 0.5, dtype=DT),
        base_probs=torch.tensor([[0.6, 0.4], [0.55, 0.45]], dtype=DT),
        contrast_graph=RenderContrastGraph(empty, empty, torch.zeros(0, dtype=DT), empty,
                                           empty, torch.zeros(0, dtype=DT),
                                           torch.ones(2, dtype=DT), {}))
    assert torch.equal(no_edges.probs, no_edges.base_probs), "an empty graph altered P_base"


def test_i_memory_tensor_is_untouched_by_the_refinement():
    """§42 Test I: the refinement never sees or writes memory."""
    names = set(inspect.signature(refine_soft_labels_with_render_contrast).parameters)
    assert not any("memory" in n.lower() for n in names), \
        f"the refinement accepts a memory argument: {names}"
    graph_names = set(inspect.signature(build_render_contrast_graph).parameters)
    assert not any("memory" in n.lower() for n in graph_names), \
        "§23: pair construction can see temporal information"
    memory = torch.full((2, 2), 0.3, dtype=DT)
    snapshot = memory.clone()
    _refine()
    assert torch.equal(memory, snapshot), "memory changed"


def test_m_small_weights_give_smooth_changes_not_one_hot():
    """§43 Test M."""
    out = _refine(lambda_agree=0.1, lambda_disagree=0.0, steps=10)
    p = out.probs
    assert float(p.max()) < 0.95, f"the refinement pushed toward one-hot: {p.tolist()}"
    assert float(p.min()) > 0.05, "a class was driven to zero"
    total = p.sum(dim=1)
    assert torch.allclose(total, torch.ones_like(total), atol=1.0e-9)
    assert float((p - out.base_probs).abs().max()) < 0.2, \
        "a small contrast weight produced a large jump"


def test_n_trusted_static_anchors_remain_fixed():
    """§43 Test N / §31."""
    trusted = torch.tensor([True, False])
    out = _refine(trusted=trusted)
    assert torch.allclose(out.probs[0], out.base_probs[0]), \
        f"a trusted anchor moved: {out.probs[0].tolist()} vs {out.base_probs[0].tolist()}"
    assert not torch.allclose(out.probs[1], out.base_probs[1]), \
        "the non-anchor did not move at all, so the test is vacuous"


def test_preserve_loss_anchors_the_refinement_to_the_base():
    base = torch.tensor([[0.6, 0.4]], dtype=DT)
    assert float(preserve_loss(base, base)) < 1.0e-12, "KL to itself is not zero"
    far = torch.tensor([[0.99, 0.01]], dtype=DT)
    near = torch.tensor([[0.65, 0.35]], dtype=DT)
    assert float(preserve_loss(near, base)) < float(preserve_loss(far, base)), \
        "the preserve term does not penalise drifting from the base"


# ---------------------------------------------------------------------------
# §44 Tests O-Q -- K > 1
# ---------------------------------------------------------------------------


def test_o_k_above_one_identical_fingerprints_encourage_similar_updates():
    e = torch.tensor([[[1.0, 2.0], [4.0, 3.0], [2.0, 1.0]],
                      [[1.0, 2.0], [4.0, 3.0], [2.0, 1.0]]], dtype=DT)   # [N=2, K+1=3, V=2]
    f = _fingerprints(e)
    cos = float((f.vectors[0] * f.vectors[1]).sum())
    assert abs(cos - 1.0) < 1.0e-9, f"K>1 identical fingerprints gave cosine {cos}"
    g = build_render_contrast_graph(f, _pair_graph())
    assert g.num_positive == 1, "K>1 identical response produced no positive edge"

    old = torch.zeros((2, 3), dtype=DT)
    same = torch.tensor([[0.3, -0.1, -0.2], [0.3, -0.1, -0.2]], dtype=DT)
    diff = torch.tensor([[0.3, -0.1, -0.2], [-0.2, -0.1, 0.3]], dtype=DT)
    ls = render_contrastive_losses(same, old, g)["agree"]
    ld = render_contrastive_losses(diff, old, g)["agree"]
    assert float(ls) < float(ld), f"K>1 agree: same {float(ls)} !< different {float(ld)}"


def test_p_k_above_one_opposite_fingerprints_encourage_opposite_updates():
    # E_1 = C - E_0 makes the class-centred negative energies exactly opposite,
    # which is what §26's fingerprint compares. Negating only some classes (as an
    # earlier version of this fixture did) gives a partial -0.5, not -1.
    e0 = torch.tensor([[1.0, 2.0], [4.0, 3.0], [2.0, 1.0]], dtype=DT)
    e = torch.stack((e0, 5.0 - e0), dim=0)
    f = _fingerprints(e)
    cos = float((f.vectors[0] * f.vectors[1]).sum())
    assert cos < -0.99, f"the fixture is not strongly anti-correlated (cos {cos})"
    g = build_render_contrast_graph(f, _pair_graph())
    assert g.num_negative == 1, "K>1 opposite response produced no negative edge"

    old = torch.zeros((2, 3), dtype=DT)
    opposite = torch.tensor([[0.3, -0.1, -0.2], [-0.3, 0.1, 0.2]], dtype=DT)
    same = torch.tensor([[0.3, -0.1, -0.2], [0.3, -0.1, -0.2]], dtype=DT)
    lo = render_contrastive_losses(opposite, old, g)["disagree"]
    ls = render_contrastive_losses(same, old, g)["disagree"]
    assert float(lo) < float(ls), f"K>1 disagree: opposite {float(lo)} !< same {float(ls)}"


def test_q_common_logit_shift_does_not_affect_the_comparison():
    """§27/§44 Test Q: centring removes the irrelevant common shift."""
    old = torch.zeros((2, 3), dtype=DT)
    ref = torch.tensor([[0.3, -0.1, -0.2], [0.1, 0.0, -0.1]], dtype=DT)
    shifted = ref + 7.5                                   # same distribution
    assert torch.allclose(centred_update(ref, old), centred_update(shifted, old)), \
        "a common additive shift changed the centred update"
    g = _two_node_graph(positive=True)
    a = render_contrastive_losses(ref, old, g)["agree"]
    b = render_contrastive_losses(shifted, old, g)["agree"]
    assert abs(float(a) - float(b)) < 1.0e-12, "a common shift changed the agreement loss"
    # And softmax agrees they are the same distribution.
    assert torch.allclose(torch.softmax(ref, 1), torch.softmax(shifted, 1))


def test_update_is_measured_against_p_old_not_p_base():
    """§30: the contrast reasons about the whole P_old -> P_ref correction."""
    base = torch.tensor([[0.6, 0.4], [0.6, 0.4]], dtype=DT)
    g = _two_node_graph(positive=True)
    a = refine_soft_labels_with_render_contrast(
        old_probs=torch.tensor([[0.5, 0.5], [0.5, 0.5]], dtype=DT),
        base_probs=base, contrast_graph=g, num_steps=10)
    b = refine_soft_labels_with_render_contrast(
        old_probs=torch.tensor([[0.9, 0.1], [0.2, 0.8]], dtype=DT),
        base_probs=base, contrast_graph=g, num_steps=10)
    assert not torch.allclose(a.probs, b.probs, atol=1.0e-6), \
        "changing P_old had no effect; the update is being measured against P_base"


def test_cli_defaults_match_the_spec():
    from argparse import ArgumentParser

    from arguments import ModelParams, OptimizationParams, PipelineParams, add_dynamic_args

    parser = ArgumentParser()
    ModelParams(parser); OptimizationParams(parser); PipelineParams(parser)
    add_dynamic_args(parser)
    a = parser.parse_args([])
    for flag, expected in (
        ("enable_render_response_contrast", False),
        ("render_contrast_steps", 10),
        ("render_contrast_lr", 1.0e-2),
        ("render_contrast_positive_threshold", 0.70),
        ("render_contrast_negative_threshold", 0.70),
        ("render_contrast_min_shared_obs", 2),
        ("lambda_render_contrast_preserve", 1.0),
        ("lambda_render_contrast_agree", 1.0),
        ("lambda_render_contrast_disagree", 0.25),
        ("render_contrast_sign_temperature", 0.10),
    ):
        assert hasattr(a, flag), f"--{flag} is not registered"
        assert getattr(a, flag) == expected, \
            f"--{flag} default is {getattr(a, flag)}, spec says {expected}"


if __name__ == "__main__":
    module = sys.modules["__main__"]
    tests = [(n, f) for n, f in sorted(vars(module).items())
             if n.startswith("test_") and inspect.isfunction(f)]
    for name, fn in tests:
        fn()
        print(f"  ok  {name}")
    print(f"PASS render_response_contrast ({len(tests)} tests)")
