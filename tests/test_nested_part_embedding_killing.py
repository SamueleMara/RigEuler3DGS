from __future__ import annotations

import importlib
import sys
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from rigid_parts.embeddings import PartEmbeddingModel
from rigid_parts.killing_inner_solver import _affine_killing_quantities, _local_field_killing_quantities
from rigid_parts.outer_embedding_solver import outer_embedding_losses
from rigid_parts.static_anchor_init import extract_confident_static_anchors
from utils.rigid_transform_utils import se3_exp_spatial


def test_static_anchor_extraction_uses_only_confident_mutual_points():
    xyz0 = torch.tensor(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [2.0, 2.0, 0.0],
        ],
        dtype=torch.float32,
    )
    xyzT = torch.tensor(
        [
            [0.0001, 0.0, 0.0],
            [1.0001, 0.0, 0.0],
            [5.0, 5.0, 0.0],
        ],
        dtype=torch.float32,
    )
    result = extract_confident_static_anchors(
        xyz0,
        xyzT,
        distance_ratio=0.001,
        mutual_nn=True,
        chunk_size=2,
    )
    assert result.mask.tolist() == [True, True, False, False]


def test_static_anchor_membership_is_forced_to_static():
    model = PartEmbeddingModel(
        4,
        3,
        5,
        static_anchor_mask=torch.tensor([True, False, False, True]),
        seed=7,
    )
    prob = model.probabilities(temperature=0.3)
    assert torch.allclose(prob[0], torch.tensor([1.0, 0.0, 0.0]))
    assert torch.allclose(prob[3], torch.tensor([1.0, 0.0, 0.0]))
    assert torch.allclose(prob.sum(dim=1), torch.ones(4), atol=1.0e-6)


def test_affine_killing_quantities_extract_translation_and_rotation():
    xyz = torch.tensor(
        [[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [-1.0, 0.0, 0.0]]],
        dtype=torch.float32,
    )
    prob = torch.tensor(
        [
            [0.0, 1.0],
            [0.0, 1.0],
            [0.0, 1.0],
        ],
        dtype=torch.float32,
    )
    linear = torch.tensor([[0.2, -0.4, 0.7]], dtype=torch.float32)
    q_translation = _affine_killing_quantities(
        torch.zeros((1, 3, 3), dtype=torch.float32),
        linear,
        xyz,
        prob,
    )
    assert torch.allclose(q_translation["omega"], torch.zeros(1, 3), atol=1.0e-6)
    assert torch.allclose(q_translation["b"], linear, atol=1.0e-6)
    assert torch.allclose(q_translation["strain"], torch.zeros(1, 3, 3), atol=1.0e-6)

    w = 0.75
    affine = torch.tensor(
        [[[0.0, -w, 0.0], [w, 0.0, 0.0], [0.0, 0.0, 0.0]]],
        dtype=torch.float32,
    )
    q_rotation = _affine_killing_quantities(
        affine,
        torch.zeros((1, 3), dtype=torch.float32),
        xyz,
        prob,
    )
    assert torch.allclose(q_rotation["omega"], torch.tensor([[0.0, 0.0, w]]), atol=1.0e-6)
    assert torch.allclose(q_rotation["b"], torch.zeros(1, 3), atol=1.0e-6)
    assert torch.allclose(q_rotation["strain"], torch.zeros(1, 3, 3), atol=1.0e-6)


def test_local_differential_field_exposes_pointwise_jacobian():
    centers = torch.tensor(
        [[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [1.0, 1.0, 0.0]]],
        dtype=torch.float32,
    )
    # A non-affine control field: the Jacobian must vary across query points.
    control_velocity = torch.tensor(
        [[[0.0, 0.0, 0.0], [0.2, 0.0, 0.0], [0.0, 0.4, 0.0], [0.4, -0.2, 0.0]]],
        dtype=torch.float32,
    )
    xyz = torch.tensor(
        [[[0.1, 0.1, 0.0], [0.9, 0.1, 0.0], [0.4, 0.8, 0.0]]],
        dtype=torch.float32,
    )
    prob = torch.tensor([[0.0, 1.0], [0.0, 1.0], [0.0, 1.0]], dtype=torch.float32)
    q = _local_field_killing_quantities(
        centers,
        control_velocity,
        torch.tensor([0.5], dtype=torch.float32),
        xyz,
        prob,
    )
    assert q["jacobian"].shape == (1, 3, 3, 3)
    assert torch.linalg.norm(q["jacobian"][0, 0] - q["jacobian"][0, 1]) > 1.0e-3
    assert torch.isfinite(q["strain_loss"])


def test_exact_spatial_se3_moves_centers_and_rotations_consistently():
    omega = torch.tensor([[0.0, 0.0, torch.pi / 2]], dtype=torch.float32)
    b = torch.zeros((1, 3), dtype=torch.float32)
    rotation, translation, quat = se3_exp_spatial(omega, b, 1.0)
    point = torch.tensor([[1.0, 0.0, 0.0]], dtype=torch.float32)
    moved = point @ rotation[0].T + translation
    assert torch.allclose(moved, torch.tensor([[0.0, 1.0, 0.0]]), atol=1.0e-5)
    assert torch.allclose(quat.norm(dim=-1), torch.ones(1), atol=1.0e-6)


def test_two_moving_parts_have_distinct_killing_twists():
    xyz = torch.tensor(
        [
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 2.0, 0.0], [0.0, 3.0, 0.0]],
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 2.0, 0.0], [0.0, 3.0, 0.0]],
        ],
        dtype=torch.float32,
    )
    prob = torch.tensor(
        [
            [0.0, 1.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [0.0, 0.0, 1.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=torch.float32,
    )
    affine = torch.zeros((2, 3, 3), dtype=torch.float32)
    affine[1] = torch.tensor(
        [[0.0, -0.4, 0.0], [0.4, 0.0, 0.0], [0.0, 0.0, 0.0]],
        dtype=torch.float32,
    )
    linear = torch.tensor([[0.3, 0.0, 0.0], [0.0, -0.2, 0.0]], dtype=torch.float32)
    q = _affine_killing_quantities(affine, linear, xyz, prob)
    assert torch.linalg.norm(q["omega"][0] - q["omega"][1]) > 0.1
    assert torch.linalg.norm(q["b"][0] - q["b"][1]) > 0.1
    rotation, translation, _ = se3_exp_spatial(q["omega"], q["b"], 0.5)
    assert torch.linalg.norm(rotation[0] - rotation[1]) > 0.1
    assert torch.linalg.norm(translation[0] - translation[1]) > 0.1


def test_moment_terms_send_gradients_to_embeddings():
    xyz0 = torch.tensor(
        [
            [-1.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 0.2, 0.0],
            [0.0, -0.2, 0.0],
        ],
        dtype=torch.float32,
    )
    final_xyz = torch.tensor(
        [
            [-1.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.2, 0.0],
            [0.0, 0.8, 0.0],
        ],
        dtype=torch.float32,
    )
    model = PartEmbeddingModel(4, 2, 3, seed=11)
    prob = model.probabilities(temperature=0.7)
    losses, _ = outer_embedding_losses(
        canonical_xyz=xyz0,
        final_xyz=final_xyz,
        final_rotation=torch.eye(3).repeat(2, 1, 1),
        final_translation=torch.tensor([[0.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=torch.float32),
        part_prob=prob,
        static_anchor_mask=torch.zeros(4, dtype=torch.bool),
        prototypes=model.prototype,
        temperature_final_assignment=0.05,
        bbox_diag=torch.tensor(2.0),
        chunk_size=8,
        margin=0.01,
    )
    moment_loss = losses["global_com"] + losses["global_inertia"] + losses["part_com"] + losses["part_inertia"]
    moment_loss.backward()
    assert model.embedding.grad is not None
    assert torch.isfinite(model.embedding.grad).all()
    assert float(model.embedding.grad.norm()) > 0.0


def test_embedding_motion_loss_is_not_placeholder_for_collapsed_motions():
    xyz0 = torch.randn((6, 3), dtype=torch.float32)
    final_xyz = xyz0.clone()
    model = PartEmbeddingModel(6, 3, 4, seed=22)
    prob = model.probabilities(temperature=0.8)
    motion_separation = torch.zeros((3, 3), dtype=torch.float32)
    losses, _ = outer_embedding_losses(
        canonical_xyz=xyz0,
        final_xyz=final_xyz,
        final_rotation=torch.eye(3).repeat(3, 1, 1),
        final_translation=torch.zeros((3, 3), dtype=torch.float32),
        part_prob=prob,
        static_anchor_mask=torch.zeros(6, dtype=torch.bool),
        prototypes=model.prototype,
        motion_separation=motion_separation,
        temperature_final_assignment=0.05,
        bbox_diag=torch.tensor(1.0),
        chunk_size=8,
        margin=0.01,
    )
    losses["embedding_motion"].backward()
    assert float(losses["embedding_motion"].detach()) > 0.0
    assert model.prototype.grad is not None
    assert float(model.prototype.grad.norm()) > 0.0


def test_missing_static_points_can_move_toward_static_without_hard_assignment():
    xyz0 = torch.tensor(
        [
            [0.0, 0.0, 0.0],
            [0.2, 0.0, 0.0],
            [0.4, 0.0, 0.0],
            [1.0, 0.0, 0.0],
        ],
        dtype=torch.float32,
    )
    final_xyz = xyz0.clone()
    model = PartEmbeddingModel(
        xyz0.shape[0],
        2,
        4,
        static_anchor_mask=torch.tensor([True, False, False, False]),
        seed=3,
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=0.05)
    identity = torch.eye(3).repeat(2, 1, 1)
    translation = torch.tensor([[0.0, 0.0, 0.0], [0.5, 0.0, 0.0]], dtype=torch.float32)
    before = model.probabilities(temperature=0.5)[:, 0].detach()
    for _ in range(80):
        optimizer.zero_grad(set_to_none=True)
        prob = model.probabilities(temperature=0.5)
        losses, _ = outer_embedding_losses(
            canonical_xyz=xyz0,
            final_xyz=final_xyz,
            final_rotation=identity,
            final_translation=translation,
            part_prob=prob,
            static_anchor_mask=torch.tensor([True, False, False, False]),
            prototypes=model.prototype,
            temperature_final_assignment=0.01,
            bbox_diag=torch.tensor(1.0),
            chunk_size=8,
            margin=0.01,
        )
        loss = losses["candidate_nn"] + losses["final_nn"] + 0.2 * losses["candidate_margin"]
        loss.backward()
        optimizer.step()
    after = model.probabilities(temperature=0.5)[:, 0].detach()
    assert float(after[1:3].mean()) > float(before[1:3].mean()) + 0.05


def test_nested_modules_do_not_depend_on_weighted_kabsch():
    rigid_fit = importlib.import_module("rigid_parts.rigid_fit")
    weighted_kabsch = getattr(rigid_fit, "weighted_kabsch", None)
    if weighted_kabsch is None:
        return
    calls = {"count": 0}

    def forbidden(*args, **kwargs):
        calls["count"] += 1
        raise AssertionError("nested part-embedding Killing path must not call Kabsch")

    setattr(rigid_fit, "weighted_kabsch", forbidden)
    try:
        xyz0 = torch.zeros((3, 3), dtype=torch.float32)
        prob = torch.full((3, 2), 0.5, dtype=torch.float32)
        outer_embedding_losses(
            canonical_xyz=xyz0,
            final_xyz=xyz0,
            final_rotation=torch.eye(3).repeat(2, 1, 1),
            final_translation=torch.zeros((2, 3)),
            part_prob=prob,
            static_anchor_mask=torch.zeros(3, dtype=torch.bool),
            prototypes=torch.zeros((2, 4)),
            temperature_final_assignment=0.01,
            bbox_diag=torch.tensor(1.0),
            chunk_size=2,
            margin=0.0,
        )
    finally:
        setattr(rigid_fit, "weighted_kabsch", weighted_kabsch)
    assert calls["count"] == 0


def test_field_separation_uses_whole_trajectory_not_one_frame():
    """§21: 'Do not force motion separation from one frame only.'

    Two parts can agree at a single instant while tracing completely
    different paths, and can differ at an instant while ending up in the same
    place. A one-frame margin is blind to both, so the separation metric has
    to be the trajectory average
    ``D_kl = (1/F) sum_t ||Log(T_k(t)^-1 T_l(t))||_W^2``.
    """
    from rigid_parts.killing_inner_solver import _sequence_motion_distance

    frames = 8
    scale = torch.tensor(1.0)

    # Two parts that coincide at exactly one frame but diverge over the path.
    identical_r = torch.eye(3)[None, None].repeat(frames, 2, 1, 1)
    diverging_t = torch.zeros((frames, 2, 3))
    for t in range(frames):
        diverging_t[t, 0, 0] = 0.1 * t
        diverging_t[t, 1, 0] = -0.1 * t
    diverging_t[0] = 0.0                      # they agree at t=0
    diverged = _sequence_motion_distance(identical_r, diverging_t, scale=scale)

    # Two parts on the same path throughout.
    same_t = torch.zeros((frames, 2, 3))
    for t in range(frames):
        same_t[t, :, 0] = 0.1 * t
    same = _sequence_motion_distance(identical_r, same_t, scale=scale)

    assert float(diverged[0, 1]) > 10.0 * float(same[0, 1]) + 1.0e-6, (
        f"trajectory separation must see the divergence the single shared frame hides "
        f"({float(diverged[0, 1]):.4g} vs {float(same[0, 1]):.4g})"
    )
    assert float(same[0, 1]) < 1.0e-6, "identical trajectories separate by zero"

    # And the single frame they share carries no information either way.
    shared_frame_only = _sequence_motion_distance(
        identical_r[:1], diverging_t[:1], scale=scale
    )
    assert float(shared_frame_only[0, 1]) < 1.0e-6, (
        "at the shared frame alone the two parts look identical -- which is exactly "
        "why §21 forbids separating on one frame"
    )


def test_inner_history_reports_every_required_metric():
    """FIRM v3 §37 (supersedes v2 §45): required inner diagnostics.

    A stage that silently stops reporting is indistinguishable from a stage
    that silently stops working, which is the failure mode this whole debug
    surface exists to prevent.
    """
    import inspect

    from rigid_parts import killing_inner_solver

    source = inspect.getsource(killing_inner_solver)
    required = [
        # §37 per frame/part
        "render_rgb", "render_ssim", "render_depth",
        "strain_norm", "strain_xx", "strain_yy", "strain_zz",
        "strain_xy", "strain_xz", "strain_yz",
        "loss_omega_local_coherence", "loss_b_local_coherence",
        "loss_temporal", "loss_acceleration",
        "observed_velocity_high_conf", "inner_total",
        "omega_norm", "u_norm", "b_norm", "loss_field_separation",
        # §36 confidence diagnostics
        "observed_velocity_num_valid",
        "observed_velocity_coverage_fraction",
        "observed_velocity_mean_confidence",
        "observed_velocity_median_confidence",
        "observed_velocity_num_above_cmin",
        "observed_velocity_high_conf_coverage",
        "observed_velocity_base_lambda",
        "observed_velocity_effective_lambda",
        "observed_velocity_weighted_fraction_of_inner_loss",
    ]
    missing = [name for name in required if f'"{name}"' not in source]
    assert not missing, f"§36/§37 inner metrics not logged: {missing}"

    # Per-moving-part breakdown, so one stalled part cannot hide behind a mean.
    for prefix in ("omega_norm_part_", "b_norm_part_", "strain_norm_part_"):
        assert prefix in source, f"§37 per-part metric missing: {prefix}*"


def test_pointwise_field_losses_are_normalised_per_part_not_globally():
    """Spec 11/14/16: L = sum_k [ sum_i p_ik l_ik / (sum_i p_ik + eps) ].

    A single global weighted mean would weight each part by its mass, letting a
    small strongly non-rigid part hide behind a large rigid one.
    """
    torch.manual_seed(3)
    n = 202
    prob = torch.zeros((n, 3), dtype=torch.float64)
    prob[:2, 1] = 1.0  # tiny part
    prob[2:, 2] = 1.0  # dominant part
    xyz = torch.rand((2, n, 3), dtype=torch.float64) * 2 - 1
    centers = torch.rand((2, 6, 3), dtype=torch.float64)
    control = torch.randn((2, 6, 3), dtype=torch.float64)
    control[0] *= 0.01
    sigma = torch.full((2,), 0.5, dtype=torch.float64)
    q = _local_field_killing_quantities(centers, control, sigma, xyz, prob)

    weights = prob[:, 1:].transpose(0, 1)
    mass = weights.sum(dim=1)
    for key, pointwise in (
        ("strain_loss", q["strain"].square().sum(dim=(-2, -1))),
        ("omega_consistency", (q["omega_local"] - q["omega"][:, None, :]).square().sum(-1)),
        ("b_consistency", (q["b_local"] - q["b"][:, None, :]).square().sum(-1)),
    ):
        per_part = ((pointwise * weights).sum(dim=1) / mass).sum()
        global_mean = (pointwise * weights).sum() / weights.sum()
        assert torch.allclose(q[key], per_part, atol=1e-12), f"{key} is not the per-part form"
        # Guard against a vacuous comparison: the two forms must actually differ here.
        assert (per_part - global_mean).abs() > 1e-6, f"{key} test does not distinguish the forms"


def test_training_pipeline_runs_with_kabsch_monkeypatched_to_raise():
    """Spec 40: with Kabsch patched to raise, the nested path must still run."""
    rigid_fit = importlib.import_module("rigid_parts.rigid_fit")
    originals = {
        name: getattr(rigid_fit, name)
        for name in dir(rigid_fit)
        if "kabsch" in name.lower() and callable(getattr(rigid_fit, name))
    }
    assert originals, "no Kabsch entry point found to monkeypatch"

    def forbidden(*args, **kwargs):
        raise AssertionError("Kabsch must not be reachable from training")

    for name in originals:
        setattr(rigid_fit, name, forbidden)
    try:
        torch.manual_seed(0)
        n = 64
        xyz0 = torch.randn((n, 3))
        model = PartEmbeddingModel(n, 3, 8, temperature=0.5,
                                   static_anchor_mask=torch.zeros(n, dtype=torch.bool))
        opt = torch.optim.Adam(model.parameters(), lr=1e-2)
        rot = torch.eye(3).repeat(3, 1, 1)
        trans = torch.zeros((3, 3))
        trans[1, 0] = 0.3
        for _ in range(3):
            losses, _ = outer_embedding_losses(
                canonical_xyz=xyz0,
                final_xyz=xyz0 + 0.1,
                final_rotation=rot,
                final_translation=trans,
                part_prob=model.probabilities(),
                static_anchor_mask=torch.zeros(n, dtype=torch.bool),
                prototypes=model.prototype,
                temperature_final_assignment=0.1,
                bbox_diag=torch.tensor(1.0),
                chunk_size=32,
                margin=0.05,
            )
            total = sum(v for v in losses.values() if v.requires_grad)
            opt.zero_grad()
            total.backward()
            opt.step()
    finally:
        for name, fn in originals.items():
            setattr(rigid_fit, name, fn)


def test_G_exact_se3_translation_rotation_screw_and_small_angle():
    """Spec 47 Test G: pure translation, pure rotation, combined screw, small angle.

    Each case is checked against an independent 4x4 matrix exponential, and the
    Euler update x + dt*v is shown to be a different (wrong) answer.
    """
    from utils.rigid_transform_utils import skew_symmetric

    dt = torch.tensor(0.37, dtype=torch.float64)
    cases = {
        "pure_translation": (torch.zeros(3, dtype=torch.float64),
                             torch.tensor([0.7, -0.2, 0.4], dtype=torch.float64)),
        "pure_rotation": (torch.tensor([0.0, 0.0, 1.1], dtype=torch.float64),
                          torch.zeros(3, dtype=torch.float64)),
        "screw": (torch.tensor([0.3, -0.2, 0.5], dtype=torch.float64),
                  torch.tensor([0.7, 0.1, -0.4], dtype=torch.float64)),
        "small_angle": (torch.tensor([1e-9, -2e-9, 5e-10], dtype=torch.float64),
                        torch.tensor([0.2, 0.3, -0.1], dtype=torch.float64)),
    }
    for name, (omega, b) in cases.items():
        rotation, translation, quat = se3_exp_spatial(omega[None], b[None], dt)
        xi = torch.zeros((4, 4), dtype=torch.float64)
        xi[:3, :3] = skew_symmetric(omega * dt)
        xi[:3, 3] = b * dt
        reference = torch.matrix_exp(xi)
        assert torch.allclose(rotation[0], reference[:3, :3], atol=1e-12), f"{name}: rotation"
        assert torch.allclose(translation[0], reference[:3, 3], atol=1e-12), f"{name}: translation"
        assert torch.isfinite(translation).all(), f"{name}: non-finite (small-angle series)"
        assert torch.allclose(quat.norm(dim=-1), torch.ones(1, dtype=torch.float64), atol=1e-12)

    # No Euler integration: for a genuinely rotating case the two must differ.
    omega, b = cases["screw"]
    _, translation, _ = se3_exp_spatial(omega[None], b[None], dt)
    assert float((translation[0] - b * dt).abs().max()) > 1e-3, "translation is an Euler update"

    # Pure translation is the one case where Exp reduces to b*dt exactly.
    omega, b = cases["pure_translation"]
    _, translation, _ = se3_exp_spatial(omega[None], b[None], dt)
    assert torch.allclose(translation[0], b * dt, atol=1e-12)


def test_D_two_moving_parts_have_low_within_part_omega_and_b_variance():
    """Spec 47 Test D: distinct trajectories AND low within-part omega/b variance."""
    torch.manual_seed(5)
    n_per = 40
    pts = torch.rand((3 * n_per, 3), dtype=torch.float64) * 2 - 1
    xyz = pts[None].repeat(2, 1, 1)
    prob = torch.zeros((3 * n_per, 3), dtype=torch.float64)
    prob[:n_per, 0] = 1.0
    prob[n_per:2 * n_per, 1] = 1.0
    prob[2 * n_per:, 2] = 1.0

    # Two exactly-rigid Killing fields sampled onto dense control points, so the
    # RBF field can represent them with negligible strain.
    gt = [
        (torch.tensor([0.0, 0.0, 0.8], dtype=torch.float64),
         torch.tensor([0.3, 0.0, 0.0], dtype=torch.float64)),
        (torch.tensor([0.5, -0.4, 0.0], dtype=torch.float64),
         torch.tensor([0.0, -0.2, 0.1], dtype=torch.float64)),
    ]
    centers = torch.rand((2, 96, 3), dtype=torch.float64) * 2.4 - 1.2
    control = torch.stack([
        torch.cross(om.expand_as(centers[i]), centers[i], dim=-1) + bb
        for i, (om, bb) in enumerate(gt)
    ])
    sigma = torch.full((2,), 0.45, dtype=torch.float64)
    q = _local_field_killing_quantities(centers, control, sigma, xyz, prob)

    weights = prob[:, 1:].transpose(0, 1)
    mass = weights.sum(dim=1)

    # Two distinct moving trajectories.
    sep_omega = float((q["omega"][0] - q["omega"][1]).norm())
    sep_b = float((q["b"][0] - q["b"][1]).norm())
    assert sep_omega > 0.1, f"omega trajectories not distinct ({sep_omega:.4f})"
    assert sep_b > 0.1, f"b trajectories not distinct ({sep_b:.4f})"

    for k, (om, bb) in enumerate(gt):
        # The local RBF field attenuates magnitude (it is a smoother, not an
        # interpolator), so direction is the recoverable quantity here.
        cos = torch.nn.functional.cosine_similarity(q["omega"][k], om, dim=0)
        assert float(cos) > 0.99, f"part {k} omega direction cos={float(cos):.4f}"

        # Low within-part variance: each part must behave as ONE Killing motion,
        # i.e. scatter inside a part must be far below the gap between parts.
        var_omega = float(((q["omega_local"][k] - q["omega"][k]).square().sum(-1) * weights[k]).sum() / mass[k])
        var_b = float(((q["b_local"][k] - q["b"][k]).square().sum(-1) * weights[k]).sum() / mass[k])
        assert var_omega < 0.1 * sep_omega ** 2, f"part {k} omega variance {var_omega:.3e} vs sep^2 {sep_omega**2:.3e}"
        assert var_b < 0.5 * sep_b ** 2, f"part {k} b variance {var_b:.3e} vs sep^2 {sep_b**2:.3e}"



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
    print(f"PASS nested_part_embedding_killing ({len(tests)} tests)")
