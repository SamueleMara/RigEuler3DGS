"""Synthetic checks for local rigid bidirectional dynamic-label refinement.

Covers the failure cases the spec requires to be tested explicitly: static
regions, a single rigid moving part, two neighbouring parts with different
transforms, a false-dynamic flow artefact, and a false-static point inside a
moving part.

Run:
    PYTHONPATH=. python tests/test_local_rigid_dynamic_refinement.py
"""

from __future__ import annotations

import math

import torch

from utils.bidirectional_local_consistency import evaluate_local_inverse_consistency
from utils.dynamic_label_refinement import (
    LABEL_AMBIGUOUS,
    LABEL_DYNAMIC,
    LABEL_STATIC,
    DynamicRefinementState,
    RefinementWeights,
    assign_discrete_labels,
    compute_classification_metrics,
    compute_prior_logit,
    fit_confidence_weights,
    refine_dynamic_probabilities,
    robust_unit_score,
)
from utils.local_rigid_field import (
    adaptive_huber_delta,
    build_knn_graph,
    closest_rotation,
    compute_effective_local_scale,
    compute_motion_gated_rigid_evidence,
    denoise_displacement_field,
    fit_local_deformation,
    local_scale_from_knn,
)


def _state(prob: torch.Tensor, cap: float = 4.0) -> DynamicRefinementState:
    return DynamicRefinementState(
        fm_prior_prob=prob.clone(),
        fm_prior_logit=compute_prior_logit(prob, cap=cap),
        refined_prob=prob.clone(),
    )


def _block(n: int = 5, spacing: float = 0.02, origin: tuple[float, float, float] = (0.0, 0.0, 0.0)) -> torch.Tensor:
    axis = torch.arange(n, dtype=torch.float32) * spacing
    gx, gy, gz = torch.meshgrid(axis, axis, axis, indexing="ij")
    points = torch.stack((gx.reshape(-1), gy.reshape(-1), gz.reshape(-1)), dim=-1)
    return points + torch.tensor(origin)


def _rotation(axis: torch.Tensor, angle_deg: float) -> torch.Tensor:
    axis = axis / axis.norm()
    angle = math.radians(angle_deg)
    K = torch.tensor([
        [0.0, -float(axis[2]), float(axis[1])],
        [float(axis[2]), 0.0, -float(axis[0])],
        [-float(axis[1]), float(axis[0]), 0.0],
    ])
    return torch.eye(3) + math.sin(angle) * K + (1.0 - math.cos(angle)) * (K @ K)


def _rigid_displacement(xyz: torch.Tensor, R: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    return xyz @ R.transpose(0, 1) + t - xyz


def test_closest_rotation_recovers_known_rotation():
    R = _rotation(torch.tensor([0.3, -0.5, 0.8]), 12.0)
    recovered, singular = closest_rotation(R[None].clone())
    assert torch.allclose(recovered[0], R, atol=1.0e-5), recovered[0]
    assert torch.allclose(singular[0], torch.ones(3), atol=1.0e-5), singular[0]
    assert float(torch.linalg.det(recovered[0])) > 0.0


def test_closest_rotation_rejects_reflection():
    # A determinant-negative input must come back as a PROPER rotation, or a
    # degenerate neighbourhood could inject a mirror "motion".
    reflection = torch.diag(torch.tensor([1.0, 1.0, -1.0]))
    recovered, _ = closest_rotation(reflection[None].clone())
    assert float(torch.linalg.det(recovered[0])) > 0.0


def test_knn_graph_excludes_self():
    points = torch.tensor([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 4.0, 0.0]])
    idx, dist = build_knn_graph(points, k=2)
    assert int(idx[0, 0]) == 1
    assert torch.allclose(dist[0, 0], torch.tensor(1.0), atol=1.0e-5)
    assert (idx[:, 0] != torch.arange(3)).all()


def test_static_region_produces_no_dynamic_evidence():
    # 34.1 -- low strain alone must NOT read as dynamic.
    xyz = _block()
    disp = torch.zeros_like(xyz)
    idx, dist = build_knn_graph(xyz, k=16)
    local = fit_local_deformation(xyz, disp, idx, knn_distance=dist)
    interior = local.valid
    assert bool(interior.any())
    assert float(local.motion_magnitude[interior].max()) < 1.0e-3
    assert float(local.strain_error[interior].max()) < 1.0e-4
    assert float(local.rigid_gain[interior].abs().max()) < 1.0e-8
    assert float(local.static_fit_error[interior].max()) < 1.0e-10


def test_single_rigid_part_recovers_its_own_transform():
    # 34.2
    xyz = _block()
    R = _rotation(torch.tensor([0.0, 0.0, 1.0]), 8.0)
    t = torch.tensor([0.03, -0.01, 0.02])
    disp = _rigid_displacement(xyz, R, t)
    idx, dist = build_knn_graph(xyz, k=16)
    local = fit_local_deformation(xyz, disp, idx, knn_distance=dist)
    good = local.valid
    assert bool(good.any())
    assert torch.allclose(local.R[good], R.expand_as(local.R[good]), atol=1.0e-3)
    assert float(local.rigid_fit_error[good].max()) < 1.0e-8
    assert float(local.static_fit_error[good].min()) > float(local.rigid_fit_error[good].max())
    assert float(local.rigid_gain[good].min()) > 0.0
    assert float(local.strain_error[good].max()) < 1.0e-4
    assert float(local.motion_magnitude[good].min()) > 0.5
    predicted = local.predicted_disp[good]
    assert torch.allclose(predicted, disp[good], atol=1.0e-3)


def test_two_moving_parts_are_not_averaged_together():
    # 34.3 -- each part keeps its own transform; boundaries may degrade, but
    # interiors must not be blended into one compromise motion.
    part_a = _block(origin=(0.0, 0.0, 0.0))
    part_b = _block(origin=(0.5, 0.0, 0.0))
    xyz = torch.cat((part_a, part_b), dim=0)
    R_a = _rotation(torch.tensor([0.0, 0.0, 1.0]), 10.0)
    R_b = _rotation(torch.tensor([1.0, 0.0, 0.0]), -14.0)
    t_a = torch.tensor([0.05, 0.0, 0.0])
    t_b = torch.tensor([0.0, 0.04, -0.02])
    disp = torch.cat((_rigid_displacement(part_a, R_a, t_a), _rigid_displacement(part_b, R_b, t_b)), dim=0)
    idx, dist = build_knn_graph(xyz, k=16)
    local = fit_local_deformation(xyz, disp, idx, knn_distance=dist)

    n_a = part_a.shape[0]
    a_valid = local.valid[:n_a]
    b_valid = local.valid[n_a:]
    assert torch.allclose(local.R[:n_a][a_valid], R_a.expand_as(local.R[:n_a][a_valid]), atol=1.0e-3)
    assert torch.allclose(local.R[n_a:][b_valid], R_b.expand_as(local.R[n_a:][b_valid]), atol=1.0e-3)
    # The two parts are far enough apart that no neighbourhood mixes them, so
    # every fit stays rigid -- the failure this guards against is a single
    # averaged transform, which would show up as large strain everywhere.
    assert float(local.strain_error[local.valid].max()) < 1.0e-3


def test_boundary_neighbourhood_loses_inlier_support():
    # 34.3 -- when a neighbourhood really does straddle two different
    # motions, it must lose support rather than silently average them.
    left = _block(n=5, origin=(0.0, 0.0, 0.0))
    right = _block(n=5, origin=(0.1, 0.0, 0.0))  # adjacent, so KNN crosses
    xyz = torch.cat((left, right), dim=0)
    disp = torch.cat((torch.zeros_like(left), _rigid_displacement(right, torch.eye(3), torch.tensor([0.06, 0.0, 0.0]))), dim=0)
    idx, dist = build_knn_graph(xyz, k=16)
    local = fit_local_deformation(xyz, disp, idx, knn_distance=dist, huber_delta=1.0)
    boundary = (xyz[:, 0] > 0.06) & (xyz[:, 0] < 0.12)
    interior = xyz[:, 0] < 0.02
    assert float(local.inlier_fraction[boundary].mean()) < float(local.inlier_fraction[interior].mean())


def test_false_static_centre_is_recovered_from_moving_neighbours():
    # 34.5 -- the leave-one-centre-out fit is what makes this possible.
    xyz = _block()
    R = torch.eye(3)
    t = torch.tensor([0.05, 0.0, 0.0])
    disp = _rigid_displacement(xyz, R, t)
    centre = int(torch.linalg.norm(xyz - xyz.mean(0), dim=-1).argmin())
    disp[centre] = torch.zeros(3)  # flow matching missed this one

    idx, dist = build_knn_graph(xyz, k=16)
    local = fit_local_deformation(xyz, disp, idx, knn_distance=dist)
    assert bool(local.valid[centre])
    # Neighbours say it moved, even though its own displacement says it did not.
    assert torch.allclose(local.predicted_disp[centre], t, atol=5.0e-3), local.predicted_disp[centre]
    assert float(local.motion_magnitude[centre]) > 0.5
    assert float(local.rigid_gain[centre]) > 0.0

    fm_prob = torch.full((xyz.shape[0],), 0.9)
    fm_prob[centre] = 0.05  # FM called it static
    report = refine_dynamic_probabilities(
        _state(fm_prob), local, None, center_displacement=disp, ema=1.0,
        motion_gate_threshold=0.5, motion_gate_temperature=0.25,
    )
    assert float(report.probability[centre]) > float(fm_prob[centre])


def test_false_dynamic_artifact_is_suppressed():
    # 34.4 -- a big displacement with no locally rigid, supported explanation
    # must lose confidence rather than be trusted.
    xyz = _block()
    disp = torch.zeros_like(xyz)
    centre = int(torch.linalg.norm(xyz - xyz.mean(0), dim=-1).argmin())
    disp[centre] = torch.tensor([0.25, -0.2, 0.15])  # isolated flow artefact

    idx, dist = build_knn_graph(xyz, k=16)
    local = fit_local_deformation(xyz, disp, idx, knn_distance=dist)
    # Its neighbours are static, so the neighbour-predicted motion is ~0 and
    # the rigid model buys nothing over the static one.
    assert float(local.motion_magnitude[centre]) < 1.0e-2
    assert float(local.rigid_gain[centre]) < 1.0e-6

    fm_prob = torch.full((xyz.shape[0],), 0.1)
    fm_prob[centre] = 0.95  # FM was fooled by the artefact
    report = refine_dynamic_probabilities(
        _state(fm_prob), local, None, center_displacement=disp, ema=1.0,
        motion_gate_threshold=0.5, motion_gate_temperature=0.25,
    )
    assert float(report.probability[centre]) < float(fm_prob[centre])


def test_inverse_consistency_is_low_for_a_reversible_rigid_motion():
    xyz0 = _block()
    R = _rotation(torch.tensor([0.0, 0.0, 1.0]), 9.0)
    t = torch.tensor([0.04, 0.01, 0.0])
    forward_disp = _rigid_displacement(xyz0, R, t)
    xyzT = xyz0 + forward_disp
    # The backward field is the true inverse map, evaluated on the frame-T cloud.
    R_inv = R.transpose(0, 1)
    t_inv = -R_inv @ t
    backward_disp = _rigid_displacement(xyzT, R_inv, t_inv)

    idx, dist = build_knn_graph(xyz0, k=16)
    forward_local = fit_local_deformation(xyz0, forward_disp, idx, knn_distance=dist)
    scale = local_scale_from_knn(dist)
    consistency, backward_local = evaluate_local_inverse_consistency(
        xyz0, forward_local, xyzT, backward_disp, local_scale=scale, knn=16, mode="transform",
    )
    good = consistency.reverse_valid
    assert bool(good.any())
    assert float(consistency.cycle_error[good].max()) < 5.0e-2
    assert float(consistency.rotation_cycle_error[good].max()) < 1.0e-3
    assert float(consistency.translation_cycle_error[good].max()) < 1.0e-2
    assert backward_local is not None
    assert torch.allclose(backward_local.R[good], R_inv.expand_as(backward_local.R[good]), atol=1.0e-3)


def test_inverse_consistency_flags_a_non_reversible_field():
    xyz0 = _block()
    forward_disp = _rigid_displacement(xyz0, torch.eye(3), torch.tensor([0.05, 0.0, 0.0]))
    xyzT = xyz0 + forward_disp
    # Backward field points the WRONG way: the cycle should not close.
    backward_disp = torch.full_like(xyzT, 0.0)
    backward_disp[:, 1] = 0.05

    idx, dist = build_knn_graph(xyz0, k=16)
    forward_local = fit_local_deformation(xyz0, forward_disp, idx, knn_distance=dist)
    scale = local_scale_from_knn(dist)
    reversible, _ = evaluate_local_inverse_consistency(
        xyz0, forward_local, xyzT,
        _rigid_displacement(xyzT, torch.eye(3), torch.tensor([-0.05, 0.0, 0.0])),
        local_scale=scale, knn=16, mode="transform",
    )
    broken, _ = evaluate_local_inverse_consistency(
        xyz0, forward_local, xyzT, backward_disp, local_scale=scale, knn=16, mode="transform",
    )
    good = reversible.reverse_valid & broken.reverse_valid
    assert float(broken.cycle_error[good].mean()) > float(reversible.cycle_error[good].mean())


def test_interpolate_cycle_mode_also_closes_on_a_reversible_field():
    xyz0 = _block()
    forward_disp = _rigid_displacement(xyz0, torch.eye(3), torch.tensor([0.05, 0.0, 0.0]))
    xyzT = xyz0 + forward_disp
    backward_disp = _rigid_displacement(xyzT, torch.eye(3), torch.tensor([-0.05, 0.0, 0.0]))
    idx, dist = build_knn_graph(xyz0, k=16)
    forward_local = fit_local_deformation(xyz0, forward_disp, idx, knn_distance=dist)
    consistency, backward_local = evaluate_local_inverse_consistency(
        xyz0, forward_local, xyzT, backward_disp,
        local_scale=local_scale_from_knn(dist), knn=16, mode="interpolate",
    )
    assert backward_local is None
    assert float(consistency.cycle_error[consistency.reverse_valid].max()) < 5.0e-2


def test_probability_update_is_conservative_and_nan_safe():
    xyz = _block()
    disp = _rigid_displacement(xyz, torch.eye(3), torch.tensor([0.05, 0.0, 0.0]))
    idx, dist = build_knn_graph(xyz, k=16)
    local = fit_local_deformation(xyz, disp, idx, knn_distance=dist)
    fm_prob = torch.full((xyz.shape[0],), 0.5)
    report = refine_dynamic_probabilities(_state(fm_prob), local, None, ema=0.15)
    # One conservative step must not jump all the way to the proposal.
    moved = (report.probability - fm_prob).abs()
    proposal_gap = (report.proposal_probability - fm_prob).abs()
    assert float(moved.max()) <= float(proposal_gap.max()) * 0.15 + 1.0e-6
    assert bool(torch.isfinite(report.probability).all())

    local.valid[:] = False  # degenerate everywhere
    held = refine_dynamic_probabilities(_state(fm_prob), local, None, ema=0.9)
    assert torch.allclose(held.probability, fm_prob)


def test_labels_use_the_ambiguous_state_at_boundaries():
    label, ambiguous = assign_discrete_labels(
        torch.tensor([0.95, 0.05, 0.5]), ambiguous_low=0.3, ambiguous_high=0.7
    )
    assert int(label[0]) == LABEL_DYNAMIC
    assert int(label[1]) == LABEL_STATIC
    assert int(label[2]) == LABEL_AMBIGUOUS
    assert bool(ambiguous[2]) and not bool(ambiguous[0])


def test_confidence_weights_keep_a_geometry_only_floor():
    p = torch.tensor([0.0, 0.0, 1.0])
    c = torch.tensor([1.0, 0.0, 1.0])
    valid = torch.tensor([True, True, True])
    w = fit_confidence_weights(p, c, valid, alpha=0.25)
    assert float(w[0]) >= 0.25  # a confidently-static neighbour still supports the fit
    assert float(w[2]) > float(w[0])


def test_robust_unit_score_is_scale_free():
    value = torch.tensor([0.0, 1.0, 2.0, 10.0])
    assert torch.allclose(robust_unit_score(value), robust_unit_score(value * 1000.0), atol=1.0e-6)
    assert float(robust_unit_score(torch.zeros(5)).max()) == 0.0


def test_displacement_denoiser_removes_an_isolated_spike():
    xyz = _block()
    disp = _rigid_displacement(xyz, torch.eye(3), torch.tensor([0.05, 0.0, 0.0]))
    centre = int(torch.linalg.norm(xyz - xyz.mean(0), dim=-1).argmin())
    noisy = disp.clone()
    noisy[centre] = torch.tensor([-0.3, 0.4, 0.2])
    idx, dist = build_knn_graph(xyz, k=16)
    cleaned, uncertainty = denoise_displacement_field(noisy, idx, dist)
    assert torch.linalg.norm(cleaned[centre] - disp[centre]) < torch.linalg.norm(noisy[centre] - disp[centre])
    assert bool(torch.isfinite(uncertainty).all())



# --- Corrections from the debug review (spec section 27) ---------------------


def test_evidence_is_not_accumulated_across_iterations():
    # Section 27 test 1. With fixed evidence the proposal must be identical
    # every iteration; only the EMA state may keep moving. Re-deriving the
    # base logit from the previous posterior would re-add the same evidence.
    xyz = _block()
    disp = _rigid_displacement(xyz, torch.eye(3), torch.tensor([0.05, 0.0, 0.0]))
    idx, dist = build_knn_graph(xyz, k=16)
    local = fit_local_deformation(xyz, disp, idx, knn_distance=dist)
    state = _state(torch.full((xyz.shape[0],), 0.4))

    proposals = []
    refined = []
    for _ in range(4):
        report = refine_dynamic_probabilities(
            state, local, None, center_displacement=disp, ema=0.15,
        )
        state.refined_prob = report.probability
        proposals.append(report.proposal_probability.clone())
        refined.append(report.probability.clone())

    for later in proposals[1:]:
        assert torch.allclose(later, proposals[0], atol=1.0e-6), "proposal drifted on fixed evidence"
    # The EMA state still converges toward that fixed proposal, monotonically.
    gaps = [float((r - proposals[0]).abs().mean()) for r in refined]
    assert gaps == sorted(gaps, reverse=True), gaps


def test_missing_reverse_evidence_is_neutral_not_ambiguous():
    # Section 27 test 2. A strong forward/local point whose backward query is
    # unusable must keep its classification.
    xyz = _block()
    disp = _rigid_displacement(xyz, torch.eye(3), torch.tensor([0.06, 0.0, 0.0]))
    idx, dist = build_knn_graph(xyz, k=16)
    local = fit_local_deformation(xyz, disp, idx, knn_distance=dist)
    xyzT = xyz + disp
    consistency, _ = evaluate_local_inverse_consistency(
        xyz, local, xyzT, _rigid_displacement(xyzT, torch.eye(3), torch.tensor([-0.06, 0.0, 0.0])),
        local_scale=local_scale_from_knn(dist), knn=16, mode="transform",
    )
    consistency.reverse_valid[:] = False  # reverse branch unavailable everywhere

    fm_prob = torch.full((xyz.shape[0],), 0.9)
    report = refine_dynamic_probabilities(
        _state(fm_prob), local, consistency, center_displacement=disp, ema=1.0,
        weights=RefinementWeights(cycle=1.5, rotation_cycle=1.0, jacobian_cycle=1.0),
    )
    assert float(report.logit_terms["cycle"].abs().max()) == 0.0
    assert float(report.logit_terms["rotation_cycle"].abs().max()) == 0.0
    assert int((report.label == LABEL_AMBIGUOUS).sum()) == 0
    assert int((report.label == LABEL_DYNAMIC).sum()) == xyz.shape[0]


def test_prior_logit_cap_keeps_a_saturated_prior_correctable():
    # Section 27 test 3.
    prior = compute_prior_logit(torch.tensor([1.0, 0.0, 0.5]), cap=4.0)
    assert bool(torch.isfinite(prior).all())
    assert float(prior[0]) == 4.0
    assert float(prior[1]) == -4.0
    assert abs(float(prior[2])) < 1.0e-5
    # Uncapped, p=1 is an unmovable infinity in all but name.
    assert float(compute_prior_logit(torch.tensor([1.0]), cap=0.0)[0]) > 10.0


def test_static_noise_does_not_become_dynamic_from_rigid_overfitting():
    # Section 27 test 4. A 6-DOF rigid model fits noise better than the
    # identity, so E_R < E_S alone must not read as motion.
    torch.manual_seed(0)
    xyz = _block()
    disp = 0.001 * torch.randn_like(xyz)  # jitter only, no real motion
    idx, dist = build_knn_graph(xyz, k=16)
    local = fit_local_deformation(xyz, disp, idx, knn_distance=dist)
    good = local.valid
    # The rigid model does win on raw residual...
    assert float(local.rigid_failure_fraction[good].mean()) < 0.5
    # ...but the motion gate removes it as dynamic evidence.
    evidence = compute_motion_gated_rigid_evidence(
        local.motion_magnitude, local.rigid_failure_fraction, threshold=1.0, temperature=0.5
    )
    assert float(evidence[good].abs().max()) < 0.05, float(evidence[good].abs().max())


def test_coherent_moving_neighbourhood_gives_positive_rigid_evidence():
    # Section 27 test 5.
    xyz = _block()
    R = _rotation(torch.tensor([0.0, 0.0, 1.0]), 9.0)
    disp = _rigid_displacement(xyz, R, torch.tensor([0.05, 0.01, 0.0]))
    idx, dist = build_knn_graph(xyz, k=16)
    local = fit_local_deformation(xyz, disp, idx, knn_distance=dist)
    evidence = compute_motion_gated_rigid_evidence(
        local.motion_magnitude, local.rigid_failure_fraction, threshold=1.0, temperature=0.5
    )
    assert float(evidence[local.valid].mean()) > 0.2


def test_effective_scale_suppresses_density_driven_false_dynamics():
    # Section 27 test 6. One uniform physical motion everywhere, but a few
    # points sit in tight clusters, so their raw nearest-neighbour spacing is
    # tiny and the SAME motion normalises to a far larger number there -- a
    # false dynamic manufactured purely by sampling density.
    torch.manual_seed(3)
    base = _block(n=6, spacing=0.02)
    # A genuinely dense patch: enough co-located points that each one's own
    # neighbourhood is entirely inside the patch, so its raw spacing collapses.
    cluster = base[0] + 0.001 * torch.randn(60, 3)
    xyz = torch.cat((base, cluster), dim=0)
    disp = _rigid_displacement(xyz, torch.eye(3), torch.tensor([0.01, 0.0, 0.0]))

    idx, dist = build_knn_graph(xyz, k=16)
    raw = local_scale_from_knn(dist)
    effective = compute_effective_local_scale(
        raw, idx, mode="neighborhood_median", clip_low_quantile=0.10, clip_high_quantile=0.90
    )
    magnitude = torch.linalg.norm(disp, dim=-1)
    raw_motion = magnitude / raw
    effective_motion = magnitude / effective
    # The physical motion is uniform, so any spread here is manufactured by
    # sampling density. The raw unit inflates the dense patch by more than an
    # order of magnitude over the median -- that is the false dynamic.
    assert float(raw_motion.max()) > 10.0 * float(raw_motion.median())
    # Pooling and clipping cannot remove the effect entirely (the patch really
    # is denser), but it must substantially reduce the worst inflation.
    assert float(effective_motion.max()) < 0.8 * float(raw_motion.max())
    raw_inflation = float(raw_motion.max()) / float(raw_motion.median())
    effective_inflation = float(effective_motion.max()) / float(effective_motion.median())
    assert effective_inflation < raw_inflation, (raw_inflation, effective_inflation)


def test_adaptive_irls_rejects_injected_outliers():
    # Section 27 test 7. A fixed threshold accepted everything; the adaptive
    # one must actually reject grossly inconsistent neighbours.
    torch.manual_seed(1)
    xyz = _block()
    t = torch.tensor([0.05, 0.0, 0.0])
    disp = _rigid_displacement(xyz, torch.eye(3), t)
    outliers = torch.arange(0, xyz.shape[0], 17)
    disp[outliers] += torch.tensor([0.0, 0.3, -0.25])

    idx, dist = build_knn_graph(xyz, k=16)
    robust = fit_local_deformation(xyz, disp, idx, knn_distance=dist, irls_iters=2)
    naive = fit_local_deformation(xyz, disp, idx, knn_distance=dist, irls_iters=1,
                                  robust_factor=1.0e6, huber_delta_min=1.0e6)
    assert float(robust.inlier_fraction.median()) < 1.0
    good = robust.valid & naive.valid
    robust_error = (robust.predicted_disp[good] - t).norm(dim=-1).median()
    naive_error = (naive.predicted_disp[good] - t).norm(dim=-1).median()
    assert float(robust_error) < float(naive_error), (float(robust_error), float(naive_error))


def test_classification_metrics_match_their_documented_rules():
    # Section 27 test 8.
    probability = torch.tensor([0.9, 0.1, 0.6, 0.45])
    label = torch.tensor([LABEL_DYNAMIC, LABEL_STATIC, LABEL_AMBIGUOUS, LABEL_AMBIGUOUS])
    gt = torch.tensor([True, False, True, False])
    m = compute_classification_metrics(probability, label, gt)

    # p >= 0.5 predicts [T, F, T, F] -> perfect.
    assert m["probability_metrics"]["f1"] == 1.0
    # Excluding ambiguous leaves [T, F] against [T, F] -> perfect, coverage 0.5.
    assert m["selective_metrics"]["coverage"] == 0.5
    assert m["selective_metrics"]["f1"] == 1.0
    # Ambiguous as static predicts [T, F, F, F]: one false negative.
    assert m["f1_ambiguous_as_static"]["false_negative"] == 1
    # Ambiguous as dynamic predicts [T, F, T, T]: one false positive.
    assert m["f1_ambiguous_as_dynamic"]["false_positive"] == 1
    assert m["ambiguity_metrics"]["ambiguous_fraction"] == 0.5
    assert m["ambiguity_metrics"]["ambiguous_gt_dynamic_fraction"] == 0.5


def test_translation_inverse_never_reaches_the_label_update():
    # Section 27 test 9.
    xyz = _block()
    disp = _rigid_displacement(xyz, torch.eye(3), torch.tensor([0.05, 0.0, 0.0]))
    idx, dist = build_knn_graph(xyz, k=16)
    local = fit_local_deformation(xyz, disp, idx, knn_distance=dist)
    xyzT = xyz + disp
    consistency, _ = evaluate_local_inverse_consistency(
        xyz, local, xyzT, _rigid_displacement(xyzT, torch.eye(3), torch.tensor([-0.05, 0.0, 0.0])),
        local_scale=local_scale_from_knn(dist), knn=16, mode="transform",
    )
    fm_prob = torch.full((xyz.shape[0],), 0.5)
    baseline = refine_dynamic_probabilities(_state(fm_prob), local, consistency,
                                            center_displacement=disp, ema=1.0)
    consistency.translation_cycle_error[:] = 1.0e4  # catastrophically mis-scaled
    after = refine_dynamic_probabilities(_state(fm_prob), local, consistency,
                                         center_displacement=disp, ema=1.0)
    assert torch.allclose(baseline.probability, after.probability)
    assert "translation_cycle" not in baseline.logit_terms


def test_backward_diagnostic_detects_directional_collapse():
    # Section 27 test 10.
    from tools.run_local_rigid_dynamic_refinement_probe import displacement_statistics

    torch.manual_seed(2)
    diverse = torch.randn(512, 3)
    collapsed = torch.tensor([1.0, 0.0, 0.0]).expand(512, 3) * torch.rand(512, 1).clamp_min(0.1)
    assert displacement_statistics(collapsed)["unit_direction_resultant_length"] > 0.95
    assert displacement_statistics(diverse)["unit_direction_resultant_length"] < 0.2


def main() -> None:
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"PASS {name}")


if __name__ == "__main__":
    main()
