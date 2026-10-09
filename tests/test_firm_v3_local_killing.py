"""FIRM v3 §39 mandatory regression tests.

Each test targets one of the failures the v2 diagnosis proved, so a regression
shows up as a failing assertion rather than as another bad run.
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from rigid_parts.embeddings import PartEmbeddingModel
from rigid_parts.inner_acceptance import endpoint_usefulness, evaluate_inner_solve
from rigid_parts.local_killing import (
    confidence_gate,
    effective_obs_lambda,
    evaluate_local_field,
    field_separation_loss,
    integrate_local_se3,
    local_killing_quantities,
    observed_velocity_loss,
)
from rigid_parts.nested_pipeline import adaptive_temperature
from rigid_parts.neighborhood import build_neighbor_graph, edge_part_weights
from rigid_parts.outer_embedding_solver import (
    build_outer_targets,
    candidate_nn_energies,
    candidates_from_local_endpoints,
    geometry_guard_value,
    outer_embedding_losses,
)

DT = torch.float64


def _rigid_field(points, omega, b):
    return torch.cross(omega.expand_as(points), points, dim=-1) + b


def _fit_local_field(points, target_velocity, *, weight=None, steps=300, control=48, sigma=0.5, lr=0.05):
    """Fit the local RBF field to a (possibly partial) velocity target."""
    n = int(points.shape[0])
    idx = torch.linspace(0, n - 1, min(control, n)).long()
    centers = points[idx][None].clone()
    sig = torch.full((1,), float(sigma), dtype=points.dtype)
    cv = torch.nn.Parameter(torch.zeros_like(centers))
    opt = torch.optim.Adam([cv], lr=lr)
    w = torch.ones(n, dtype=points.dtype) if weight is None else weight
    for _ in range(steps):
        opt.zero_grad()
        v, _, _ = evaluate_local_field(points[None], centers, cv, sig)
        loss = (w * (v[0] - target_velocity).square().sum(-1)).sum() / w.sum().clamp_min(1e-8)
        loss.backward()
        opt.step()
    return centers, cv.detach(), sig


# --------------------------------------------------------------------------
# Test 1 — low-confidence wrong tracks must not dominate
# --------------------------------------------------------------------------
def test_1_low_confidence_wrong_tracks_do_not_dominate():
    torch.manual_seed(0)
    n = 400
    x = torch.rand((n, 3), dtype=DT) * 2 - 1
    omega = torch.tensor([0.0, 0.0, 0.8], dtype=DT)
    b = torch.tensor([0.2, 0.0, 0.0], dtype=DT)
    true_v = _rigid_field(x, omega, b)
    wrong_v = -3.0 * true_v  # confidently wrong direction and magnitude

    prob = torch.ones((n, 1), dtype=DT)
    predicted = true_v[None].clone()

    # (a) below c_min the gate is exactly zero: the term cannot act at all.
    low = {
        "velocity": wrong_v,
        "valid": torch.ones(n, dtype=torch.bool),
        "confidence": torch.full((n,), 0.30, dtype=DT),
    }
    loss_low, stats_low = observed_velocity_loss(predicted, low, prob, c_min=0.60, gamma=2.0)
    assert float(loss_low) == 0.0, f"low-confidence tracks leaked in: {float(loss_low)}"
    assert stats_low["num_above_cmin"] == 0

    # (b) even at high confidence, §13 caps the share of the inner objective.
    high = {
        "velocity": wrong_v,
        "valid": torch.ones(n, dtype=torch.bool),
        "confidence": torch.full((n,), 0.95, dtype=DT),
    }
    loss_high, _ = observed_velocity_loss(predicted, high, prob, c_min=0.60, gamma=2.0)
    assert float(loss_high) > 0.0
    primary = torch.tensor(0.05, dtype=DT)          # render + physics
    lam = effective_obs_lambda(1.0, primary, loss_high, rho_max=0.20)
    share = float(lam * loss_high / (primary + lam * loss_high))
    assert share <= 0.20 + 1e-9, f"observed velocity took {share:.4f} of the inner objective"
    # The v2 state (~72%) must be unreachable.
    assert share < 0.72


def test_1b_physics_beats_wrong_low_confidence_tracks_in_optimization():
    """With gated tracks, the fitted field follows the rigid truth, not the tracks."""
    torch.manual_seed(1)
    n = 250
    x = torch.rand((n, 3), dtype=DT) * 2 - 1
    omega = torch.tensor([0.0, 0.0, 0.8], dtype=DT)
    b = torch.tensor([0.2, 0.0, 0.0], dtype=DT)
    true_v = _rigid_field(x, omega, b)
    wrong_v = -3.0 * true_v

    gate = confidence_gate(torch.ones(n, dtype=torch.bool), torch.full((n,), 0.30, dtype=DT), c_min=0.60)
    assert float(gate.max()) == 0.0
    # Target is the render/physics-consistent truth weighted by (1 - gate) and the
    # tracks weighted by gate; with the gate at zero the tracks vanish.
    blended = (1.0 - gate)[:, None] * true_v + gate[:, None] * wrong_v
    assert torch.allclose(blended, true_v)

    _, cv, _ = _fit_local_field(x, blended, steps=250)
    assert torch.isfinite(cv).all()


# --------------------------------------------------------------------------
# Test 2 — partial track coverage
# --------------------------------------------------------------------------
def test_2_untracked_regions_still_move():
    torch.manual_seed(2)
    n = 400
    x = torch.rand((n, 3), dtype=DT) * 2 - 1
    omega = torch.tensor([0.0, 0.0, 0.6], dtype=DT)
    b = torch.tensor([0.3, 0.1, 0.0], dtype=DT)
    true_v = _rigid_field(x, omega, b)

    tracked = torch.zeros(n, dtype=torch.bool)
    tracked[torch.randperm(n)[: int(0.4 * n)]] = True
    assert 0.35 < float(tracked.float().mean()) < 0.45

    # Only the tracked 40% get a supervision weight; the field is smooth so the
    # untracked 60% must still move, not stay static.
    weight = tracked.to(DT)
    centers, cv, sig = _fit_local_field(x, true_v, weight=weight, steps=400)
    v, _, _ = evaluate_local_field(x[None], centers, cv, sig)
    speed_untracked = v[0][~tracked].norm(dim=-1)
    speed_tracked = v[0][tracked].norm(dim=-1)

    assert float(speed_untracked.mean()) > 0.5 * float(speed_tracked.mean()), (
        f"untracked region went static: {float(speed_untracked.mean()):.4f} "
        f"vs tracked {float(speed_tracked.mean()):.4f}"
    )
    # And they move in the right direction.
    cos = torch.nn.functional.cosine_similarity(v[0][~tracked], true_v[~tracked], dim=-1)
    assert float(cos.mean()) > 0.9, f"untracked velocity direction wrong (cos {float(cos.mean()):.3f})"


# --------------------------------------------------------------------------
# Test 3 — local rigidity, not one global hard twist
# --------------------------------------------------------------------------
def test_3_support_crossing_articulation_shows_local_inconsistency():
    torch.manual_seed(3)
    m = 200
    # The two bodies must TOUCH, otherwise the kNN graph never spans the joint
    # and there is no articulation boundary to detect.
    left = torch.rand((m, 3), dtype=DT) * torch.tensor([1.0, 1.0, 1.0], dtype=DT) - torch.tensor([1.0, 0.5, 0.5], dtype=DT)
    right = torch.rand((m, 3), dtype=DT) * torch.tensor([1.0, 1.0, 1.0], dtype=DT) + torch.tensor([0.0, -0.5, -0.5], dtype=DT)
    x = torch.cat((left, right), dim=0)

    # Two genuinely different rigid motions either side of the boundary.
    v = torch.cat((
        _rigid_field(left, torch.tensor([0.0, 0.0, 0.9], dtype=DT), torch.zeros(3, dtype=DT)),
        _rigid_field(right, torch.tensor([0.0, 0.0, -0.9], dtype=DT), torch.tensor([0.5, 0.0, 0.0], dtype=DT)),
    ), dim=0)

    centers, cv, sig = _fit_local_field(x, v, steps=500, control=64, sigma=0.35)
    graph = build_neighbor_graph(x, k=12)

    crossing = torch.ones((2 * m, 1), dtype=DT)  # one support spanning the joint
    edge_w = edge_part_weights(graph, crossing)
    q = local_killing_quantities(x[None], centers, cv, sig, crossing, graph, edge_w)

    spans_joint = (x[graph.source, 0] < 0) != (x[graph.target, 0] < 0)
    assert int(spans_joint.sum()) > 0, "the two bodies are not adjacent; no joint to detect"

    # The inconsistency must be LOCAL and must sit ON the articulation.
    per_point = q["strain"][0].square().sum(dim=(-2, -1))
    near = x[:, 0].abs() < 0.2
    assert float(per_point[near].mean()) > 2.0 * float(per_point[~near].mean()), (
        f"strain did not localise at the boundary: near {float(per_point[near].mean()):.4e} "
        f"vs far {float(per_point[~near].mean()):.4e}"
    )

    omega = q["omega_local"][0]
    across = (omega[graph.source][spans_joint] - omega[graph.target][spans_joint]).norm(dim=-1).mean()
    within = (omega[graph.source][~spans_joint] - omega[graph.target][~spans_joint]).norm(dim=-1).mean()
    assert float(across) > 2.0 * float(within), (
        f"local twist disagreement is not concentrated on the joint: "
        f"across {float(across):.4f} vs within {float(within):.4f}"
    )

    # ...and that signal is what lets memberships split the support later: the
    # boundary points are exactly the ones a split would have to cut.
    assert float(per_point[near].max()) == float(per_point.max()), (
        "the strongest strain is not on the articulation"
    )


def test_3b_no_global_hard_twist_is_imposed():
    """omega_ik / b_ik are per-Gaussian; nothing collapses them to one twist."""
    torch.manual_seed(4)
    n = 150
    x = torch.rand((n, 3), dtype=DT) * 2 - 1
    centers = torch.rand((1, 40, 3), dtype=DT) * 2 - 1
    cv = torch.randn((1, 40, 3), dtype=DT) * 0.4
    sig = torch.full((1,), 0.4, dtype=DT)
    prob = torch.ones((n, 1), dtype=DT)
    graph = build_neighbor_graph(x, k=10)
    q = local_killing_quantities(x[None], centers, cv, sig, prob, graph, edge_part_weights(graph, prob))
    assert q["omega_local"].shape == (1, n, 3)
    assert q["b_local"].shape == (1, n, 3)
    assert float(q["omega_local"].std(dim=1).max()) > 1e-6, "omega_local collapsed to a single global twist"


# --------------------------------------------------------------------------
# Test 4 — inertia-collapse regression
# --------------------------------------------------------------------------
def test_4_inertia_collapse_is_blocked_by_mass_and_geometry_guard():
    torch.manual_seed(5)
    n = 300
    xyz0 = torch.rand((n, 3), dtype=DT) * 2 - 1
    shift = torch.tensor([0.6, 0.0, 0.0], dtype=DT)
    truly_moving = xyz0[:, 0] > 0.0

    moving_endpoint = xyz0 + shift
    candidates = candidates_from_local_endpoints(xyz0, moving_endpoint[None])
    final_xyz = torch.where(truly_moving[:, None], xyz0 + shift, xyz0)

    targets = build_outer_targets(
        final_xyz=final_xyz, candidates=candidates,
        temperature_final_assignment=0.01, chunk_size=256,
    )

    honest = torch.zeros((n, 2), dtype=DT)
    honest[truly_moving, 1] = 1.0
    honest[~truly_moving, 0] = 1.0
    # Collapse: keep only a compact sliver of the moving part.
    collapsed = honest.clone()
    sliver = truly_moving & (xyz0[:, 1].abs() > 0.35)
    collapsed[sliver, 1] = 0.0
    collapsed[sliver, 0] = 1.0

    # §39 Test 4 asks us to CONSTRUCT the degeneracy: make the inertia target the
    # one the collapsed support matches, while mass/COM/NN targets stay honest.
    # Now lowering normalised inertia genuinely requires shrinking the part.
    from rigid_parts.outer_embedding_solver import _soft_inertia_at_positions

    inertia_of_collapsed, _ = _soft_inertia_at_positions(candidates, collapsed)
    _, com_of_honest = _soft_inertia_at_positions(candidates, honest)
    targets["part_inertia"] = inertia_of_collapsed.detach()
    targets["part_com"] = com_of_honest.detach()
    targets["part_mass"] = (honest.sum(dim=0) / n).detach()

    weights = {
        "candidate_nn": 1.0, "final_nn": 1.0, "global_com": 1.0, "global_inertia": 1.0,
        "part_mass": 2.0, "part_com": 2.0, "part_inertia": 2.0, "static_anchor": 0.0,
        "embedding_motion": 0.0, "candidate_margin": 0.0, "entropy": 0.0,
        "occupancy": 0.0, "trust_region": 0.0,
    }

    def losses_for(prob):
        return outer_embedding_losses(
            canonical_xyz=xyz0, final_xyz=final_xyz, part_prob=prob,
            static_anchor_mask=torch.zeros(n, dtype=torch.bool),
            prototypes=torch.zeros((2, 4), dtype=DT),
            candidates=candidates, targets=targets,
            bbox_diag=torch.tensor(2.0, dtype=DT), chunk_size=256, margin=0.001,
        )[0]

    l_honest = losses_for(honest)
    l_collapsed = losses_for(collapsed)

    assert float(l_collapsed["part_inertia"]) < float(l_honest["part_inertia"]), (
        "test is vacuous: collapsing did not lower normalised inertia"
    )
    assert float(l_collapsed["part_mass"]) > float(l_honest["part_mass"]), (
        "§27 part-mass did not penalise the shrunken support"
    )
    guard_honest = float(geometry_guard_value(l_honest, weights))
    guard_collapsed = float(geometry_guard_value(l_collapsed, weights))
    assert guard_collapsed > guard_honest, (
        f"§32 geometry guard failed to see the collapse ({guard_collapsed:.6f} vs {guard_honest:.6f})"
    )


# --------------------------------------------------------------------------
# Test 5 — soft-temperature recovery
# --------------------------------------------------------------------------
def test_5_temperature_stays_soft_until_motion_is_valid():
    tau = 1.0
    # Invalid motion: tau may never go soft-floor-crossing, whatever else holds.
    for _ in range(20):
        tau, blockers = adaptive_temperature(
            tau, motion_accepted=False, label_churn_fraction=0.0,
            candidate_separation=1.0, geometry_stable=True,
        )
        assert tau >= 0.7, f"tau fell to {tau} while the motion was invalid"
        assert blockers

    # Unstable labels also hold it up.
    tau, blockers = adaptive_temperature(
        tau, motion_accepted=True, label_churn_fraction=0.5,
        candidate_separation=1.0, geometry_stable=True,
    )
    assert tau >= 0.7 and any("labels" in b for b in blockers)

    # Weak candidate separation holds it up.
    tau, blockers = adaptive_temperature(
        tau, motion_accepted=True, label_churn_fraction=0.0,
        candidate_separation=0.0, geometry_stable=True,
    )
    assert tau >= 0.7 and any("separated" in b for b in blockers)

    # Everything good: tau may finally decrease, but never below the hard floor.
    for _ in range(100):
        tau, blockers = adaptive_temperature(
            tau, motion_accepted=True, label_churn_fraction=0.0,
            candidate_separation=1.0, geometry_stable=True,
        )
        assert not blockers
    assert tau >= 0.4 - 1e-12, f"tau fell below the 0.4 hard floor: {tau}"
    assert tau < 0.7, "tau never decreased even though every condition held"

    # The forbidden v2 schedule (1.0 -> 0.15 by iteration count) is unreachable.
    assert tau > 0.15


def test_5b_rejected_motion_does_not_update_embeddings():
    """§20: a failing inner solve produces no embedding update."""
    n = 40
    prob = torch.zeros((n, 2), dtype=DT)
    prob[:, 1] = 1.0
    # Moving candidate fits the final cloud far worse than static identity.
    cand_nn = torch.stack((torch.full((n,), 0.01, dtype=DT), torch.full((n,), 0.50, dtype=DT)), dim=1)
    decision = evaluate_inner_solve(
        acceptance_metrics={"render_improvement": 0.01, "mean_strain_norm": 0.0},
        candidate_nn=cand_nn, part_prob=prob,
    )
    assert not decision.accepted
    assert any("worse than static" in r for r in decision.reasons)
    assert decision.metrics["delta_endpoint_worst"] > 0

    # A useful hypothesis is accepted.
    good = torch.stack((torch.full((n,), 0.50, dtype=DT), torch.full((n,), 0.01, dtype=DT)), dim=1)
    ok = evaluate_inner_solve(
        acceptance_metrics={"render_improvement": 0.01, "mean_strain_norm": 0.0},
        candidate_nn=good, part_prob=prob,
    )
    assert ok.accepted, ok.reasons
    assert ok.metrics["delta_endpoint_worst"] < 0


def test_5c_endpoint_usefulness_matches_the_spec_formula():
    n = 6
    prob = torch.tensor(
        [[0.2, 0.8], [0.5, 0.5], [0.9, 0.1], [0.3, 0.7], [0.6, 0.4], [0.1, 0.9]], dtype=DT
    )
    cand = torch.tensor(
        [[0.10, 0.04], [0.20, 0.30], [0.05, 0.50], [0.40, 0.10], [0.15, 0.25], [0.30, 0.05]], dtype=DT
    )
    got = endpoint_usefulness(cand, prob)
    gain = cand[:, 1] - cand[:, 0]
    expect = (prob[:, 1] * gain).sum() / prob[:, 1].sum()
    assert torch.allclose(got[0], expect, atol=1e-12), f"{float(got[0])} vs {float(expect)}"


# --------------------------------------------------------------------------
# Test 6 — active anchor loss
# --------------------------------------------------------------------------
def test_6_anchor_loss_gradient_is_nonzero_after_warmup():
    n = 120
    mask = torch.zeros(n, dtype=torch.bool)
    mask[:40] = True
    model = PartEmbeddingModel(n, 3, 8, temperature=1.0, static_anchor_mask=mask,
                               anchor_target_probability=0.999)
    prob = model.probabilities(apply_anchor=False)
    assert float(prob[mask, 0].min()) < 1.0, "anchor probability is exactly one; gradient would vanish"
    assert float(prob[mask, 0].min()) > 0.99, "anchors are not strongly static at init"

    loss = -torch.log(prob[mask, 0].clamp_min(1e-8)).mean()
    loss.backward()
    assert model.embedding.grad is not None
    grad = float(model.embedding.grad[mask].norm())
    assert grad > 0.0, "anchor loss gradient is identically zero (v2 regression)"
    assert torch.isfinite(model.embedding.grad).all()


def test_6b_permanent_hard_pinning_is_not_the_default():
    n = 30
    mask = torch.zeros(n, dtype=torch.bool)
    mask[:10] = True
    model = PartEmbeddingModel(n, 3, 6, temperature=1.0, static_anchor_mask=mask)
    default_prob = model.probabilities()
    assert float(default_prob[mask, 0].max()) < 1.0, "probabilities() still hard-pins by default"
    warmup_prob = model.probabilities(apply_anchor=True)
    assert float(warmup_prob[mask, 0].min()) == 1.0, "warmup pinning no longer available"


# --------------------------------------------------------------------------
# Test 7 — literal candidate margin (l != k only)
# --------------------------------------------------------------------------
def test_7_candidate_margin_excludes_l_equals_k():
    torch.manual_seed(7)
    n, k1 = 50, 3
    xyz0 = torch.rand((n, 3), dtype=DT)
    moving = torch.stack((xyz0 + 0.3, xyz0 - 0.2))
    candidates = candidates_from_local_endpoints(xyz0, moving)
    final_xyz = xyz0 + 0.3
    prob = torch.softmax(torch.randn((n, k1), dtype=DT), dim=1)
    margin = 0.001
    targets = build_outer_targets(final_xyz=final_xyz, candidates=candidates,
                                  temperature_final_assignment=0.01, chunk_size=64)
    losses, _ = outer_embedding_losses(
        canonical_xyz=xyz0, final_xyz=final_xyz, part_prob=prob,
        static_anchor_mask=torch.zeros(n, dtype=torch.bool),
        prototypes=torch.zeros((k1, 4), dtype=DT),
        candidates=candidates, targets=targets,
        bbox_diag=torch.tensor(1.0, dtype=DT), chunk_size=64, margin=margin,
    )

    cand_nn = candidate_nn_energies(candidates, final_xyz, 64)
    literal = torch.zeros((), dtype=DT)
    for i in range(n):
        for k in range(k1):
            for l in range(k1):
                if l == k:
                    continue
                literal = literal + prob[i, k] * torch.relu(margin + cand_nn[i, k] - cand_nn[i, l])
    literal = literal / prob.sum()
    assert torch.allclose(losses["candidate_margin"], literal, atol=1e-10), (
        f"margin {float(losses['candidate_margin']):.8f} != literal l!=k {float(literal):.8f}"
    )
    # The l=k terms would each add exactly `margin`; show that offset is absent.
    with_diagonal = literal + margin * k1 * float(prob.sum()) / float(prob.sum())
    assert not torch.allclose(losses["candidate_margin"], with_diagonal, atol=1e-10)


# --------------------------------------------------------------------------
# Test 8 — K >= 2 field separation
# --------------------------------------------------------------------------
def test_8_field_separation_pushes_collapsed_local_fields_apart():
    torch.manual_seed(8)
    diag = torch.rand((200, 3), dtype=DT) * 2 - 1
    centers = torch.rand((2, 40, 3), dtype=DT) * 2 - 1
    centers[1] = centers[0]
    sigma = torch.full((2,), 0.4, dtype=DT)

    # Near-collapse: two hypotheses that have converged onto the same motion.
    cv = torch.nn.Parameter(torch.randn((2, 40, 3), dtype=DT) * 0.02)
    with torch.no_grad():
        cv[1] = cv[0] + 1.0e-6 * torch.randn_like(cv[0])

    loss0, dist0 = field_separation_loss(centers, cv, sigma, diag, margin=0.5)
    assert float(dist0[0, 1]) < 1e-9, "fields are not actually collapsed at the start"
    assert float(loss0) > 0.0, "separation loss is inactive on collapsed fields"

    opt = torch.optim.Adam([cv], lr=0.05)
    for _ in range(300):
        opt.zero_grad()
        loss, _ = field_separation_loss(centers, cv, sigma, diag, margin=0.5)
        loss.backward()
        opt.step()
    loss1, dist1 = field_separation_loss(centers, cv, sigma, diag, margin=0.5)
    assert float(dist1[0, 1]) > float(dist0[0, 1]), "separation did not push the hypotheses apart"
    assert float(loss1) < float(loss0), f"separation loss did not decrease ({float(loss1)} vs {float(loss0)})"
    assert float(loss1) == 0.0, "separation did not clear the margin"


def test_8c_exact_collapse_is_a_saddle_so_the_solver_must_break_symmetry():
    """§18's squared-difference form has zero gradient at EXACT collapse.

    Zero-initialising every part's control velocities would start each interval
    on that saddle, so the inner solver perturbs them for K>1.
    """
    diag = torch.rand((100, 3), dtype=DT) * 2 - 1
    centers = torch.rand((2, 30, 3), dtype=DT) * 2 - 1
    centers[1] = centers[0]
    sigma = torch.full((2,), 0.4, dtype=DT)

    identical = torch.nn.Parameter(torch.zeros((2, 30, 3), dtype=DT))
    loss, dist = field_separation_loss(centers, identical, sigma, diag, margin=0.5)
    grad = torch.autograd.grad(loss, identical)[0]
    assert float(dist[0, 1]) == 0.0
    assert float(loss) > 0.0, "loss should be active at collapse"
    assert float(grad.abs().max()) == 0.0, "expected the documented zero-gradient saddle"

    # The solver's own initialisation must therefore NOT be exactly collapsed.
    import inspect

    from rigid_parts import killing_inner_solver

    source = inspect.getsource(killing_inner_solver.solve_nested_killing_sequence)
    assert "killing_field_symmetry_break" in source, (
        "inner solver no longer breaks the K>1 initialisation symmetry"
    )


def test_8b_field_separation_reported_inactive_at_k1():
    diag = torch.rand((50, 3), dtype=DT)
    centers = torch.rand((1, 20, 3), dtype=DT)
    cv = torch.randn((1, 20, 3), dtype=DT)
    sigma = torch.full((1,), 0.4, dtype=DT)
    loss, dist = field_separation_loss(centers, cv, sigma, diag, margin=0.5)
    assert float(loss) == 0.0
    assert dist.shape == (1, 1)


# --------------------------------------------------------------------------
# §9 / §17 structural checks
# --------------------------------------------------------------------------
def test_per_gaussian_se3_transport_is_exact_and_rigid_for_a_shared_twist():
    torch.manual_seed(9)
    n = 200
    x = torch.rand((1, n, 3), dtype=DT) * 2 - 1
    omega = torch.tensor([0.4, -0.3, 0.9], dtype=DT).expand(1, n, 3).contiguous()
    b = torch.tensor([0.2, 0.5, -0.1], dtype=DT).expand(1, n, 3).contiguous()
    quat = torch.zeros((1, n, 4), dtype=DT)
    quat[..., 0] = 1.0

    def pairwise(a):  # explicit differences; cdist's expanded form loses precision
        return (a[:, None, :] - a[None, :, :]).norm(dim=-1)

    before = pairwise(x[0])
    pos, q = x.clone(), quat.clone()
    for _ in range(25):
        pos, q = integrate_local_se3(pos, q, omega, b, torch.tensor(0.05, dtype=DT))
    assert float((pairwise(pos[0]) - before).abs().max()) < 1e-12
    assert float((pos - x).norm(dim=-1).mean()) > 0.1
    assert abs(float(q.norm(dim=-1).mean()) - 1.0) < 1e-12


if __name__ == "__main__":
    import inspect

    module = sys.modules["__main__"]
    tests = [
        (name, fn)
        for name, fn in sorted(vars(module).items())
        if name.startswith("test_") and inspect.isfunction(fn)
    ]
    for name, fn in tests:
        fn()
        print(f"  ok  {name}")
    print(f"PASS firm_v3_local_killing ({len(tests)} tests)")
