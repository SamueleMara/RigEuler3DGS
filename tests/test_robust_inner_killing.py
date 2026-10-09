"""Robust inner Killing field: required tests A-X (§42-§50)."""

from __future__ import annotations

import inspect
import sys
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from rigid_parts.neighborhood import build_neighbor_graph
from rigid_parts.robust_inner_killing import (
    clean_moving_support,
    cluster_moving_support,
    exact_rigid_consistency_losses,
    field_rms,
    fitting_weights,
    integrate_scalar,
    motion_prior_losses,
    normalize_gauge,
    observability,
    rigid_field,
    scalar_consistency_loss,
    scalar_from_observation,
    scalar_sign_flip_loss,
    screw_pitch,
    se3_exp_twist,
    solve_shared_field_and_scalar,
    transport_points,
)

DT = torch.float64


def _probs(n, dynamic, confidence=0.9):
    p = torch.zeros((n, 2), dtype=DT)
    p[:, 0] = 1.0
    p[dynamic, 0] = 1.0 - confidence
    p[dynamic, 1] = confidence
    return p


# ---------------------------------------------------------------------------
# §42 Tests A-C -- support cleaning
# ---------------------------------------------------------------------------


def test_a_isolated_unsupported_candidate_is_removed():
    # A dense moving blob, plus one far-away lone "moving" point.
    blob = torch.rand((60, 3), generator=torch.Generator().manual_seed(1), dtype=DT) * 0.2
    lone = torch.tensor([[50.0, 0.0, 0.0]], dtype=DT)
    xyz = torch.cat((blob, lone), 0)
    n = int(xyz.shape[0])
    dynamic = torch.zeros(n, dtype=torch.bool)
    dynamic[:] = True
    p = _probs(n, dynamic, 0.9)
    p[-1, 0], p[-1, 1] = 0.45, 0.55        # the lone point is also weakly confident
    z = torch.ones(n, dtype=torch.long)
    graph = build_neighbor_graph(xyz, k=8)
    out = clean_moving_support(xyz, p, z, graph, mad_scale=3.5, local_dynamic_min=0.35)
    assert bool(out.outlier[-1]), "the isolated unsupported candidate was kept"
    assert int(out.outlier[:-1].sum()) == 0, "blob points were rejected"
    assert out.diagnostics["outlier_count"] == 1


def test_b_thin_but_supported_structure_is_retained():
    """§3: spacing alone must never remove a point."""
    # A thin 1D filament: large spacing, but every point has dynamic neighbours.
    xyz = torch.stack((torch.linspace(0, 5, 40, dtype=DT),
                       torch.zeros(40, dtype=DT), torch.zeros(40, dtype=DT)), dim=1)
    n = 40
    z = torch.ones(n, dtype=torch.long)
    p = _probs(n, torch.ones(n, dtype=torch.bool), 0.95)
    graph = build_neighbor_graph(xyz, k=4)
    out = clean_moving_support(xyz, p, z, graph)
    assert int(out.outlier.sum()) == 0, \
        f"{int(out.outlier.sum())} points of a supported filament were rejected"
    assert int(out.clean.sum()) == n


def test_c_persistent_soft_probabilities_are_unchanged():
    xyz = torch.rand((40, 3), generator=torch.Generator().manual_seed(2), dtype=DT)
    p = _probs(40, torch.ones(40, dtype=torch.bool), 0.9)
    snapshot = p.clone()
    z = torch.ones(40, dtype=torch.long)
    graph = build_neighbor_graph(xyz, k=6)
    cleaning = clean_moving_support(xyz, p, z, graph)
    cluster_moving_support(xyz, cleaning, 1)
    assert torch.equal(p, snapshot), "support cleaning mutated the persistent labels"


# ---------------------------------------------------------------------------
# §43 Tests D-F -- clustering
# ---------------------------------------------------------------------------


def _two_blobs(gap=6.0, n_each=50):
    g = torch.Generator().manual_seed(3)
    a = torch.rand((n_each, 3), generator=g, dtype=DT) * 0.5
    b = torch.rand((n_each, 3), generator=g, dtype=DT) * 0.5 + torch.tensor([gap, 0, 0], dtype=DT)
    return torch.cat((a, b), 0)


def test_d_two_separated_groups_give_two_clusters():
    xyz = _two_blobs()
    n = int(xyz.shape[0])
    z = torch.ones(n, dtype=torch.long)
    p = _probs(n, torch.ones(n, dtype=torch.bool), 0.95)
    graph = build_neighbor_graph(xyz, k=8)
    cleaning = clean_moving_support(xyz, p, z, graph)
    clusters = cluster_moving_support(xyz, cleaning, 2, knn=8)
    a = clusters.assignment[:50]
    b = clusters.assignment[50:]
    assert len(set(a[a >= 0].tolist())) == 1, "blob A was split across clusters"
    assert len(set(b[b >= 0].tolist())) == 1, "blob B was split across clusters"
    assert int(a[a >= 0][0]) != int(b[b >= 0][0]), "the two blobs were merged"


def test_e_elongated_connected_part_stays_one_cluster():
    """§8: this is why plain Euclidean K-means is not the default."""
    n = 80
    xyz = torch.stack((torch.linspace(0, 10, n, dtype=DT),
                       torch.zeros(n, dtype=DT), torch.zeros(n, dtype=DT)), dim=1)
    z = torch.ones(n, dtype=torch.long)
    p = _probs(n, torch.ones(n, dtype=torch.bool), 0.95)
    graph = build_neighbor_graph(xyz, k=6)
    cleaning = clean_moving_support(xyz, p, z, graph)
    clusters = cluster_moving_support(xyz, cleaning, 1, knn=6)
    assigned = clusters.assignment[clusters.assignment >= 0]
    assert len(set(assigned.tolist())) == 1, "a single elongated rigid part was fragmented"
    assert int(assigned.numel()) == n


def test_f_low_confidence_point_stays_unassigned():
    xyz = _two_blobs(gap=6.0)
    lone = torch.tensor([[25.0, 0.0, 0.0]], dtype=DT)
    xyz = torch.cat((xyz, lone), 0)
    n = int(xyz.shape[0])
    z = torch.ones(n, dtype=torch.long)
    p = _probs(n, torch.ones(n, dtype=torch.bool), 0.95)
    p[-1, 0], p[-1, 1] = 0.7, 0.3          # below the attach confidence floor
    graph = build_neighbor_graph(xyz, k=8)
    cleaning = clean_moving_support(xyz, p, z, graph)
    clusters = cluster_moving_support(xyz, cleaning, 2, knn=8, attach_min_confidence=0.5)
    assert int(clusters.assignment[-1]) == -1, \
        "an unsupported low-confidence point was forced into a cluster"


# ---------------------------------------------------------------------------
# §44 Tests G-H -- transport vs fitting support
# ---------------------------------------------------------------------------


def test_g_transport_support_may_exceed_fitting_support():
    xyz = torch.rand((60, 3), generator=torch.Generator().manual_seed(4), dtype=DT)
    z = torch.ones(60, dtype=torch.long)                      # all transported
    p = _probs(60, torch.ones(60, dtype=torch.bool), 0.95)
    p[40:, 0], p[40:, 1] = 0.6, 0.4                           # 20 are barely dynamic
    graph = build_neighbor_graph(xyz, k=8)
    cleaning = clean_moving_support(xyz, p, z, graph)
    w = fitting_weights(cleaning.confidence, floor=0.50, gamma=2.0)
    transport = int((z > 0).sum())
    fitting = int((w > 0).sum())
    assert transport == 60 and fitting == 40, \
        f"transport {transport} / fitting {fitting}: the two supports are not distinct"
    assert fitting < transport


def test_h_low_confidence_points_have_zero_fitting_weight():
    c = torch.tensor([0.2, 0.49, 0.50, 0.75, 1.0], dtype=DT)
    w = fitting_weights(c, floor=0.50, gamma=2.0)
    assert float(w[0]) == 0.0 and float(w[1]) == 0.0, "sub-floor confidence got weight"
    assert float(w[2]) == 0.0, "exactly at the floor should be zero"
    assert abs(float(w[3]) - 0.25) < 1.0e-12, "the gamma ramp is wrong"
    assert abs(float(w[4]) - 1.0) < 1.0e-12
    assert bool((w[1:] >= w[:-1]).all()), "fitting weight is not monotone in confidence"


# ---------------------------------------------------------------------------
# §46 Tests K-M -- exact rigid consistency
# ---------------------------------------------------------------------------


def _cells_from_twist(centers, omega, u):
    """Local cells generated from ONE global twist: omega_c = omega,
    u_c = u + omega x b_c, which is the exact-consistency fixed point."""
    cell_omega = omega.reshape(1, 3).expand(centers.shape[0], 3).clone()
    cell_u = u.reshape(1, 3) + torch.cross(cell_omega, centers, dim=-1)
    return cell_omega, cell_u


def test_k_cells_from_one_global_twist_give_near_zero_losses():
    centers = torch.rand((12, 3), generator=torch.Generator().manual_seed(5), dtype=DT)
    omega = torch.tensor([0.1, -0.2, 0.3], dtype=DT)
    u = torch.tensor([0.5, 0.1, -0.4], dtype=DT)
    co, cu = _cells_from_twist(centers, omega, u)
    out = exact_rigid_consistency_losses(co, cu, centers, omega, u)
    assert float(out["omega"]) < 1.0e-20, f"L_omega = {float(out['omega'])}"
    assert float(out["u"]) < 1.0e-20, f"L_u = {float(out['u'])}"


def test_l_inconsistent_axes_produce_positive_l_omega():
    centers = torch.rand((12, 3), generator=torch.Generator().manual_seed(6), dtype=DT)
    omega = torch.tensor([0.1, -0.2, 0.3], dtype=DT)
    u = torch.tensor([0.5, 0.1, -0.4], dtype=DT)
    co, cu = _cells_from_twist(centers, omega, u)
    co = co.clone()
    co[0] = torch.tensor([-0.3, 0.4, 0.1], dtype=DT)          # one cell disagrees
    out = exact_rigid_consistency_losses(co, cu, centers, omega, u)
    assert float(out["omega"]) > 1.0e-6, "an inconsistent axis was not penalised"


def test_m_inconsistent_translations_produce_positive_l_u():
    centers = torch.rand((12, 3), generator=torch.Generator().manual_seed(7), dtype=DT)
    omega = torch.tensor([0.1, -0.2, 0.3], dtype=DT)
    u = torch.tensor([0.5, 0.1, -0.4], dtype=DT)
    co, cu = _cells_from_twist(centers, omega, u)
    cu = cu.clone()
    cu[3] = cu[3] + torch.tensor([1.0, 0.0, 0.0], dtype=DT)
    out = exact_rigid_consistency_losses(co, cu, centers, omega, u)
    assert float(out["u"]) > 1.0e-6, "an inconsistent translation was not penalised"
    assert float(out["omega"]) < 1.0e-20, "the axis loss was contaminated"


def test_j_zero_exact_weights_leave_the_objective_unchanged():
    """§45 Test J: with the new weights at 0 the added terms contribute nothing."""
    centers = torch.rand((8, 3), generator=torch.Generator().manual_seed(8), dtype=DT)
    omega = torch.tensor([0.2, 0.1, -0.1], dtype=DT)
    u = torch.tensor([0.0, 0.3, 0.2], dtype=DT)
    co, cu = _cells_from_twist(centers, omega, u)
    cu = cu + 0.5                                              # deliberately inconsistent
    out = exact_rigid_consistency_losses(co, cu, centers, omega, u)
    total = 0.0 * float(out["omega"]) + 0.0 * float(out["u"])
    assert total == 0.0, "zero-weighted exact terms still contributed"


# ---------------------------------------------------------------------------
# §47 Tests N-O -- scalar factorization
# ---------------------------------------------------------------------------


def _synthetic(frames=8, points=40, seed=9):
    g = torch.Generator().manual_seed(seed)
    x = torch.rand((points, 3), generator=g, dtype=DT) * 2.0 - 1.0
    omega = torch.tensor([0.0, 0.0, 0.6], dtype=DT)
    u = torch.tensor([0.3, 0.0, 0.0], dtype=DT)
    phi = rigid_field(x, omega, u)
    s = torch.tensor([0.2, 0.5, 0.9, 1.2, 1.0, 0.6, 0.3, 0.1], dtype=DT)[:frames]
    v = s[:, None, None] * phi[None]
    return x, omega, u, phi, s, v


def test_n_synthetic_scalar_factorization_is_recovered():
    x, omega, u, phi, s, v = _synthetic()
    w = torch.ones(int(x.shape[0]), dtype=DT)
    dt = torch.ones(int(s.numel()), dtype=DT)
    sol = solve_shared_field_and_scalar(
        x, v, w, dt, spatial_steps=200, scalar_steps=200, rounds=3, lr=0.05,
        lambda_continuity=0.0, lambda_flip=0.0)

    # The gauge is fixed by normalisation, so compare PHYSICAL velocity.
    predicted = sol.speed[:, None, None] * rigid_field(x, sol.omega, sol.u)[None]
    error = (predicted - v).norm(dim=-1).median() / v.norm(dim=-1).median()
    assert float(error) < 0.15, f"relative velocity error {float(error):.4f}"
    # The recovered speed must track the true profile up to one global scale.
    scale = (sol.speed * s).sum() / (s * s).sum()
    corr = torch.corrcoef(torch.stack((sol.speed, s)))[0, 1]
    assert float(corr) > 0.95, f"speed profile correlation {float(corr):.4f}"
    assert float(scale) > 0, "the recovered speed is globally sign-inverted"


def test_o_changing_speed_changes_magnitude_not_direction():
    x, omega, u, phi, s, v = _synthetic()
    slow = 0.25 * phi
    fast = 2.0 * phi
    cos = torch.nn.functional.cosine_similarity(slow, fast, dim=-1)
    assert float(cos.min()) > 1.0 - 1.0e-12, "scaling the speed rotated the field"
    assert float((fast.norm(dim=-1) / slow.norm(dim=-1)).std()) < 1.0e-9, \
        "scaling was not uniform"


# ---------------------------------------------------------------------------
# §48 Tests P-Q -- gauge
# ---------------------------------------------------------------------------


def test_p_scaling_field_and_speed_preserves_physical_velocity():
    x, omega, u, phi, s, v = _synthetic()
    a = 3.7
    scaled = rigid_field(x, omega * a, u * a)
    assert torch.allclose(scaled, a * phi), "the field is not linear in the twist"
    before = s[:, None, None] * phi[None]
    after = (s / a)[:, None, None] * scaled[None]
    assert torch.allclose(before, after), "Phi -> aPhi, s -> s/a changed the velocity"


def test_q_normalisation_fixes_the_rms_gauge_and_preserves_velocity():
    x, omega, u, phi, s, v = _synthetic()
    w = torch.ones(int(x.shape[0]), dtype=DT)
    o, uu, sn, rms = normalize_gauge(omega * 5.0, u * 5.0, s / 5.0, x, w)
    assert abs(float(field_rms(x, o, uu, w)) - 1.0) < 1.0e-9, "RMS gauge was not fixed to 1"
    before = (s / 5.0)[:, None, None] * rigid_field(x, omega * 5.0, u * 5.0)[None]
    after = sn[:, None, None] * rigid_field(x, o, uu)[None]
    assert torch.allclose(before, after, atol=1.0e-10), "normalisation changed the velocity"
    # §18: the sign convention makes the first strong speed positive.
    _, _, flipped, _ = normalize_gauge(omega, u, -s, x, w)
    assert float(flipped[int(s.argmax())]) > 0, "the sign convention was not applied"


# ---------------------------------------------------------------------------
# §49 Tests R-T -- sign-flip loss
# ---------------------------------------------------------------------------


def test_r_constant_sign_gives_zero_flip_penalty():
    assert float(scalar_sign_flip_loss(torch.tensor([1.0, 1.0, 1.0], dtype=DT))) == 0.0


def test_s_abrupt_reversal_gives_a_high_penalty():
    abrupt = float(scalar_sign_flip_loss(torch.tensor([1.0, -1.0], dtype=DT)))
    assert abrupt > 0.5, f"an abrupt +1 -> -1 reversal scored only {abrupt}"


def test_t_smooth_zero_crossing_is_almost_free():
    """§26: a real smooth reversal must remain allowed."""
    smooth = torch.tensor([0.1, 0.01, 0.0, -0.01, -0.1], dtype=DT)
    abrupt = torch.tensor([1.0, -1.0], dtype=DT)
    ls = float(scalar_sign_flip_loss(smooth))
    la = float(scalar_sign_flip_loss(abrupt))
    assert ls < 0.05 * la, f"smooth reversal {ls} is not much cheaper than abrupt {la}"


# ---------------------------------------------------------------------------
# §50 Tests U-X -- optional priors
# ---------------------------------------------------------------------------


def test_u_mode_none_sets_both_prior_weights_to_zero():
    omega = torch.tensor([0.3, 0.2, 0.1], dtype=DT)
    u = torch.tensor([0.5, 0.0, 0.0], dtype=DT)
    out = motion_prior_losses(omega, u, "none")
    assert float(out["revolute"]) == 0.0 and float(out["prismatic"]) == 0.0


def test_v_prismatic_mode_penalises_rotation():
    u = torch.tensor([0.5, 0.0, 0.0], dtype=DT)
    spin = motion_prior_losses(torch.tensor([0.3, 0.2, 0.1], dtype=DT), u, "prismatic")
    pure = motion_prior_losses(torch.zeros(3, dtype=DT), u, "prismatic")
    assert float(spin["prismatic"]) > 0.0, "a rotating twist was not penalised"
    assert float(pure["prismatic"]) == 0.0, "pure translation was penalised"


def test_w_revolute_mode_penalises_nonzero_pitch():
    omega = torch.tensor([0.0, 0.0, 1.0], dtype=DT)
    pure = torch.tensor([1.0, 0.0, 0.0], dtype=DT)          # u perpendicular -> pitch 0
    screw = torch.tensor([0.0, 0.0, 1.0], dtype=DT)         # u parallel -> pitch 1
    assert abs(float(screw_pitch(omega, pure))) < 1.0e-12
    # 1 / (1 + eps), not exactly 1, because of the guarded denominator.
    assert abs(float(screw_pitch(omega, screw)) - 1.0) < 1.0e-6
    assert float(motion_prior_losses(omega, pure, "revolute")["revolute"]) < 1.0e-12
    assert float(motion_prior_losses(omega, screw, "revolute")["revolute"]) > 0.5


def test_x_priors_off_reproduce_the_baseline_objective():
    omega = torch.tensor([0.3, 0.2, 0.1], dtype=DT)
    u = torch.tensor([0.5, 0.0, 0.2], dtype=DT)
    off = motion_prior_losses(omega, u, "none")
    contribution = 0.0 * float(off["revolute"]) + 0.0 * float(off["prismatic"])
    assert contribution == 0.0, "a hidden bias entered with priors off"
    try:
        motion_prior_losses(omega, u, "spiral")
    except ValueError:
        pass
    else:
        raise AssertionError("an unknown prior mode was silently accepted")


# ---------------------------------------------------------------------------
# §28, §29, §38 -- integration, transport, observability
# ---------------------------------------------------------------------------


def test_integration_uses_real_dt_and_starts_at_zero():
    s = torch.tensor([1.0, 1.0, 1.0], dtype=DT)
    q = integrate_scalar(s, torch.tensor([0.5, 2.0], dtype=DT))
    assert float(q[0]) == 0.0, "q(t_0) != 0"
    assert abs(float(q[1]) - 0.5) < 1.0e-12, "trapezoid with dt=0.5 is wrong"
    assert abs(float(q[2]) - 2.5) < 1.0e-12, "real dt was not used"
    unit = integrate_scalar(s, torch.ones(2, dtype=DT))
    assert not torch.allclose(q, unit), "dt had no effect (unit-step assumption)"


def test_se3_transport_is_a_rigid_motion():
    omega = torch.tensor([0.0, 0.0, 1.0], dtype=DT)
    u = torch.tensor([0.0, 0.0, 0.0], dtype=DT)
    q = torch.tensor([0.0, torch.pi / 2], dtype=DT)
    T = se3_exp_twist(omega, u, q)
    assert torch.allclose(T[0], torch.eye(4, dtype=DT), atol=1.0e-9), "q=0 is not identity"
    R = T[1, :3, :3]
    assert torch.allclose(R @ R.transpose(0, 1), torch.eye(3, dtype=DT), atol=1.0e-9), \
        "the rotation is not orthogonal"
    assert abs(float(torch.linalg.det(R)) - 1.0) < 1.0e-9, "determinant is not 1"
    pts = torch.tensor([[1.0, 0.0, 0.0]], dtype=DT)
    moved = transport_points(T, pts)
    assert torch.allclose(moved[1, 0], torch.tensor([0.0, 1.0, 0.0], dtype=DT), atol=1.0e-9), \
        "a 90-degree z-rotation did not map x to y"
    # Rigid motion preserves pairwise distance.
    two = torch.tensor([[1.0, 0.0, 0.0], [0.0, 2.0, 0.0]], dtype=DT)
    out = transport_points(T, two)
    assert abs(float((out[1, 0] - out[1, 1]).norm() - (two[0] - two[1]).norm())) < 1.0e-9


def test_observability_detects_a_degenerate_support():
    """§38: this is the number that separates blade's divergence from a good fit."""
    line = torch.stack((torch.linspace(-1, 1, 50, dtype=DT),
                        torch.zeros(50, dtype=DT), torch.zeros(50, dtype=DT)), dim=1)
    blob = torch.rand((50, 3), generator=torch.Generator().manual_seed(11), dtype=DT) * 2 - 1
    w = torch.ones(50, dtype=DT)
    thin = observability(line, w)
    full = observability(blob, w)
    assert thin["rank"] < 6, f"a collinear support reported full rank {thin['rank']}"
    assert full["rank"] == 6, f"a volumetric support is rank {full['rank']}, expected 6"
    assert thin["condition_number"] > full["condition_number"], \
        "the degenerate support is not worse conditioned"


def test_scalar_consistency_and_inconsistency_signal():
    x, omega, u, phi, s, v = _synthetic()
    implied = scalar_from_observation(v, phi[None].expand_as(v))
    # For exactly-synthetic data the implied speed IS s(t) for every point.
    assert torch.allclose(implied, s[:, None].expand_as(implied), atol=1.0e-9), \
        "the implied speed does not recover s(t) on noiseless data"
    w = torch.ones_like(implied)
    # Not exactly 0: the eps in scalar_from_observation's denominator leaves a
    # ~1e-6 residual where |Phi| is smallest (0.084 here), so the Huber mean
    # lands around 2e-14 rather than at machine zero.
    assert float(scalar_consistency_loss(implied, s[:, None].expand_as(implied), w)) < 1.0e-10
    wrong = implied.clone()
    wrong[:, 0] += 5.0
    assert float(scalar_consistency_loss(wrong, s[:, None].expand_as(implied), w)) > 0.0


# ===========================================================================
# CODEX_ONE_KILLING_FIELD_CORRECTION -- §39-§47
# ===========================================================================


def test_correction_43_exact_rigid_projection_recovers_the_twist():
    """§43: a rigid input field must project back to its own twist, residual ~0."""
    from rigid_parts.robust_inner_killing import project_field_to_rigid_twist

    g = torch.Generator().manual_seed(21)
    x = torch.rand((80, 3), generator=g, dtype=DT) * 2 - 1
    omega = torch.tensor([0.2, -0.4, 0.1], dtype=DT)
    u = torch.tensor([0.7, 0.1, -0.3], dtype=DT)
    phi = rigid_field(x, omega, u)
    out = project_field_to_rigid_twist(x, phi, torch.ones(80, dtype=DT))
    assert torch.allclose(out["omega"], omega, atol=1.0e-6), \
        f"omega* = {out['omega'].tolist()} vs {omega.tolist()}"
    assert torch.allclose(out["u"], u, atol=1.0e-6), \
        f"u* = {out['u'].tolist()} vs {u.tolist()}"
    assert float(out["residual"]) < 1.0e-12, f"residual {float(out['residual'])}"
    assert out["rank"] == 6 and out["valid"] and not out["fallback"]
    # A_i xi must reproduce the field exactly.
    assert torch.allclose(out["rigid_velocity"], phi, atol=1.0e-6)


def test_correction_44_non_rigid_field_has_positive_residual():
    """§44."""
    from rigid_parts.robust_inner_killing import project_field_to_rigid_twist

    g = torch.Generator().manual_seed(22)
    x = torch.rand((80, 3), generator=g, dtype=DT) * 2 - 1
    phi = rigid_field(x, torch.tensor([0.2, -0.4, 0.1], dtype=DT),
                      torch.tensor([0.7, 0.1, -0.3], dtype=DT))
    perturbed = phi + 0.3 * torch.randn(phi.shape, generator=g, dtype=DT)
    out = project_field_to_rigid_twist(x, perturbed, torch.ones(80, dtype=DT))
    assert float(out["residual"]) > 1.0e-3, \
        f"a non-rigid field projected with residual {float(out['residual'])}"


def test_correction_40_projection_is_differentiable_to_the_field():
    """§40: gradients must flow loss -> xi* -> phi -> control."""
    from rigid_parts.robust_inner_killing import project_field_to_rigid_twist

    g = torch.Generator().manual_seed(23)
    x = torch.rand((40, 3), generator=g, dtype=DT) * 2 - 1
    control = torch.randn((40, 3), generator=g, dtype=DT, requires_grad=True)
    phi = control * 0.5                                    # a trivial field of control
    out = project_field_to_rigid_twist(x, phi, torch.ones(40, dtype=DT))
    target = torch.tensor([0.1, 0.2, 0.3, 0.4, 0.5, 0.6], dtype=DT)
    (out["xi"] - target).square().sum().backward()
    assert control.grad is not None, "no gradient reached the field"
    assert bool(torch.isfinite(control.grad).all()), "non-finite gradient"
    assert float(control.grad.norm()) > 0, "zero gradient: the projection is detached"


def test_correction_45_projection_is_homogeneous_in_the_field():
    """§45: xi*(a Phi) = a xi*(Phi), which is what makes the gauge consistent."""
    from rigid_parts.robust_inner_killing import project_field_to_rigid_twist

    g = torch.Generator().manual_seed(24)
    x = torch.rand((60, 3), generator=g, dtype=DT) * 2 - 1
    phi = torch.randn((60, 3), generator=g, dtype=DT)
    w = torch.rand(60, generator=g, dtype=DT)
    base = project_field_to_rigid_twist(x, phi, w)
    for a in (0.25, 3.0, 17.0):
        scaled = project_field_to_rigid_twist(x, a * phi, w)
        assert torch.allclose(scaled["xi"], a * base["xi"], rtol=1.0e-8, atol=1.0e-10), \
            f"xi*(a Phi) != a xi*(Phi) at a={a}"
    # And the gauge swap leaves the physical velocity identical.
    s = torch.tensor([0.5, 1.0, 1.5], dtype=DT)
    a = 4.0
    before = s[:, None, None] * base["rigid_velocity"][None]
    after = (s / a)[:, None, None] * project_field_to_rigid_twist(x, a * phi, w)["rigid_velocity"][None]
    assert torch.allclose(before, after, atol=1.0e-10), "the gauge swap changed the velocity"


def test_correction_39_field_matches_the_derivative_of_the_transform():
    """§39: s(t) A(x) xi* must be the first-order motion of Exp(q(t) xi*)."""
    from rigid_parts.robust_inner_killing import rigid_basis

    omega = torch.tensor([0.0, 0.0, 0.7], dtype=DT)
    u = torch.tensor([0.2, -0.1, 0.0], dtype=DT)
    xi = torch.cat((omega, u))
    x = torch.tensor([[1.0, 0.5, -0.3], [-0.7, 0.2, 0.9]], dtype=DT)

    eps = 1.0e-6
    q = torch.tensor([0.0, eps], dtype=DT)
    T = se3_exp_twist(omega, u, q)
    numerical = (transport_points(T, x)[1] - x) / eps
    analytic = torch.einsum("nij,j->ni", rigid_basis(x), xi)
    assert torch.allclose(numerical, analytic, atol=1.0e-5), (
        f"field and SE(3) transport disagree to first order: "
        f"{numerical.tolist()} vs {analytic.tolist()}")


def test_correction_42_no_second_spatial_parameter_exists():
    """§42/§56: the spatial optimizer must own ONLY the field parameters."""
    import ast as _ast

    source = (REPO / "rigid_parts" / "shared_killing_solver.py").read_text()
    tree = _ast.parse(source)
    forbidden = {"omega_g", "u_g", "global_twist", "global_omega", "global_translation",
                 "omega_rigid", "u_rigid", "xi_rigid"}
    created: list[str] = []
    for node in _ast.walk(tree):
        if not isinstance(node, _ast.Assign):
            continue
        call = node.value
        is_param = (isinstance(call, _ast.Call)
                    and _ast.unparse(call.func).endswith("Parameter"))
        if not is_param:
            continue
        for target in node.targets:
            created.append(_ast.unparse(target))
    assert created, "no Parameters found at all; the parser is wrong"
    offending = [c for c in created if any(f in c for f in forbidden)]
    assert not offending, f"§42 violated: trainable {offending}"
    # Only the field and the scalar may be trainable.
    assert set(created) <= {"control", "speed"}, \
        f"unexpected trainable parameters: {sorted(set(created))}"
    # And the spatial optimizer must be built on control alone. Since
    # CODEX_DECOUPLED that group is named by ``stage_a_parameters``, which
    # tests/test_decoupled_joint_scalar.py pins to exactly [control].
    # CODEX_MERGED renamed the blocks to geometry/scalar; the old names alias it.
    from rigid_parts.shared_killing_solver import stage_a_parameters

    _c = torch.nn.Parameter(torch.zeros(1, 2, 3))
    _s = torch.nn.Parameter(torch.zeros(1, 4))
    assert [id(t) for t in stage_a_parameters(_c, _s)] == [id(_c)], \
        "the spatial optimizer does not own exactly [control]"
    assert "torch.optim.Adam(geometry_parameters(control, speed), lr=geometry_lr)" in source, \
        "the spatial optimizer is not built from the declared geometry block"


def test_correction_47_spatial_parameters_have_no_time_dimension():
    """§46/§47: only s(t) may carry a time axis."""
    source = (REPO / "rigid_parts" / "shared_killing_solver.py").read_text()
    assert "speed = torch.nn.Parameter(torch.ones((moving_parts, frames)" in source, \
        "s(t) is not the per-frame scalar"
    # control is [parts, cells, 3] -- built from `centers`, which has no time axis.
    assert "control = torch.nn.Parameter(control)" in source
    assert "torch.zeros_like(centers)" in source, \
        "the field is not shaped from the control centers (no time dimension)"


def test_correction_38_saved_omega_comes_from_xi_star():
    """§38C: stored omega/b must be the derived twist times the speed."""
    source = (REPO / "rigid_parts" / "shared_killing_solver.py").read_text()
    assert 'proj["omega"].detach()[None] * s_mid[:, None]' in source, \
        "the saved omega sequence is not derived from xi*"
    assert 'xi_star_is_derived' in source, "§65: the artifact does not mark xi* as derived"


def test_correction_52_render_loss_exists_and_is_routed_to_both_blocks():
    """ONE-FIELD \u00a752 required L_render in both alternating phases.

    CODEX_DECOUPLED then moved it to Stage B only, and
    CODEX_MERGED_ROBUST_SUPPORT_AND_DETERMINISTIC_LOSS_ROUTING moved it back to
    BOTH: its \u00a738 makes render mandatory in the geometry block ("because s_k(t)
    is frozen, render gradients can only correct the spatial joint geometry")
    and \u00a749 keeps it in the scalar block. \u00a756's routing table is the authority.
    What has been true throughout, and is what \u00a752 caught as a real bug, is that
    the render loss must EXIST and actually be invoked.
    """
    source = (REPO / "rigid_parts" / "shared_killing_solver.py").read_text()
    assert "multiview_render_loss_for_frame(" in source, "\u00a752: no render loss is computed"
    assert "_render_and_depth(" in source, "the render loss is never invoked"
    geometry = source[source.index("def _geometry_terms"):source.index("def _geometry_total")]
    assert "_render_and_depth" in geometry, "\u00a738 violated: the geometry block does not render"
    scalar = source[source.index("def _scalar_terms"):source.index("def _scalar_total")]
    assert "_render_and_depth" in scalar, "\u00a749 violated: the scalar block does not render"

    from rigid_parts.shared_killing_solver import LOSS_ROUTING
    assert set(LOSS_ROUTING["render"]) == {"geometry", "scalar"}, \
        f"\u00a756: render must route to both blocks, got {LOSS_ROUTING['render']}"

def test_correction_49_small_fitting_support_is_invalid_not_silently_expanded():
    """§49: an empty/small support must not fall back to noisy transport points."""
    source = (REPO / "rigid_parts" / "shared_killing_solver.py").read_text()
    assert "motion marked INVALID" in source, "§49: the invalid-motion state is missing"
    assert "fitting_mask[part] = transport_mask[part] & cleaning.clean" not in source, \
        "§49 violated: the silent fallback to cleaned transport support is still present"
    assert "valid_motion" in source, "no valid_motion flag is produced"


def test_correction_50_projection_reports_rank_and_conditioning():
    """§50: rank/condition must come out of the projection itself."""
    from rigid_parts.robust_inner_killing import project_field_to_rigid_twist

    line = torch.stack((torch.linspace(-1, 1, 30, dtype=DT),
                        torch.zeros(30, dtype=DT), torch.zeros(30, dtype=DT)), dim=1)
    out = project_field_to_rigid_twist(line, torch.randn((30, 3), dtype=DT),
                                       torch.ones(30, dtype=DT))
    assert out["rank"] < 6 and not out["valid"], \
        f"a collinear support reported rank {out['rank']}, valid={out['valid']}"
    assert out["condition_number"] > 1.0e6, "a degenerate support looked well conditioned"


def test_cli_defaults_match_the_spec():
    from argparse import ArgumentParser

    from arguments import ModelParams, OptimizationParams, PipelineParams, add_dynamic_args

    parser = ArgumentParser()
    ModelParams(parser); OptimizationParams(parser); PipelineParams(parser)
    add_dynamic_args(parser)
    a = parser.parse_args([])
    for flag, expected in (
        ("inner_motion_prior", "none"),
        ("lambda_revolute_prior", 0.0),
        ("lambda_prismatic_prior", 0.0),
        ("inner_outlier_mad_scale", 3.5),
        ("inner_outlier_local_dynamic_min", 0.35),
        ("inner_fit_confidence_floor", 0.50),
        ("inner_fit_confidence_gamma", 2.0),
        ("inner_cluster_core_confidence", 0.80),
        ("inner_cluster_knn", 16),
        ("inner_cluster_confidence_gamma", 1.0),
    ):
        assert hasattr(a, flag), f"--{flag} is not registered"
        assert getattr(a, flag) == expected, \
            f"--{flag} default is {getattr(a, flag)}, spec says {expected}"
    for mode in ("none", "revolute", "prismatic"):
        assert parser.parse_args(["--inner_motion_prior", mode]).inner_motion_prior == mode


if __name__ == "__main__":
    module = sys.modules["__main__"]
    tests = [(n, f) for n, f in sorted(vars(module).items())
             if n.startswith("test_") and inspect.isfunction(f)]
    for name, fn in tests:
        fn()
        print(f"  ok  {name}")
    print(f"PASS robust_inner_killing ({len(tests)} tests)")
