"""Frame subsetting, joint-type labelling, and the profile-scale trajectory.

Run as a script from the repository root:

    PYTHONPATH=. python tests/test_joint_frames_and_joint_type.py
"""

import contextlib
import io

import torch

from utils.joint_model_fitting import (
    _finite_json,
    _make_joint_model,
    fit_prismatic_joint,
    fit_revolute_joint,
)
from utils.kpart_field_merge import (
    compare_rigid_and_translation_evidence,
    fit_region_rigid_field,
)
from utils.joint_trajectory_optimization import (
    optimize_monotonic_q,
    optimize_profile_scale_q,
)
from utils.kpart_velocity_field_discovery import (
    _otsu_threshold,
    _prior_anchor_masks,
    observed_velocity_snr,
)
from utils.rigid_killing_motion import (
    RigidKillingMode,
    instantaneous_axis,
    part_extent,
    re_reference_rigid_mode,
    screw_geometry,
    solve_rigid_killing_mode_all_intervals,
    solve_translation_mode_all_intervals,
)
from utils.unified_dynamic_training import (
    expand_joint_coordinates,
    parse_joint_frame_spec,
)


def _mode(linear, angular, reference=(0.0, 0.0, 0.0)):
    return RigidKillingMode(
        reference_point=torch.as_tensor(reference, dtype=torch.float64),
        reference_velocity=torch.as_tensor(linear, dtype=torch.float64),
        angular_mode=torch.as_tensor(angular, dtype=torch.float64),
        mode_type="rotational",
        normalization_scale=1.0,
        valid=True,
        diagnostics={},
    )


def test_frame_spec_parsing():
    ids = list(range(60))
    assert parse_joint_frame_spec("all", ids) == list(range(60))
    assert parse_joint_frame_spec("endpoints", ids) == [0, 59]
    assert parse_joint_frame_spec("stride:20", ids) == [0, 20, 40, 59]
    assert parse_joint_frame_spec("0,29,59", ids) == [0, 29, 59]
    # A spec that omits an endpoint still gets both, so the selection always
    # spans the full physical motion.
    assert parse_joint_frame_spec("29", ids) == [0, 29, 59]
    for bad in ("0,99", "0,-1,99"):
        try:
            parse_joint_frame_spec(bad, ids)
        except ValueError:
            pass
        else:
            raise AssertionError(f"{bad!r} should have been rejected")
    try:
        parse_joint_frame_spec("all", [0])
    except ValueError:
        pass
    else:
        raise AssertionError("a single frame should have been rejected")
    # Non-contiguous frame ids are addressed by id, not by position.
    assert parse_joint_frame_spec("10", [0, 5, 10, 15]) == [0, 2, 3]


def test_coordinate_expansion_round_trip():
    profile = torch.linspace(0.0, 1.0, 60, dtype=torch.float64) ** 1.5
    stage1 = [profile * 0.18]
    # A two-frame Stage 3 recovered an endpoint of 0.532 where Stage 1 ended at
    # 0.18; the expansion must transfer only that magnitude.
    sparse = torch.tensor([[0.0, 0.532]], dtype=torch.float64)
    expanded, scales = expand_joint_coordinates(sparse, [0, 59], stage1)
    assert abs(float(expanded[0, -1]) - 0.532) < 1.0e-9
    assert float(expanded[0, 0]) == 0.0
    assert abs(scales[0] - 0.532 / 0.18) < 1.0e-6
    correlation = float(torch.corrcoef(torch.stack([expanded[0], profile]))[0, 1])
    assert abs(correlation - 1.0) < 1.0e-9
    # Sampling the full profile and expanding it back is the identity.
    positions = [0, 20, 40, 59]
    resampled = stage1[0][torch.tensor(positions)][None]
    back, back_scales = expand_joint_coordinates(resampled, positions, stage1)
    assert abs(back_scales[0] - 1.0) < 1.0e-9
    assert float((back[0] - stage1[0]).abs().max()) < 1.0e-12


def test_screw_geometry_is_gauge_invariant():
    linear = torch.tensor([0.0, 2.0, 0.0], dtype=torch.float64)
    angular = torch.tensor([0.0, 0.0, 2.0], dtype=torch.float64)
    base = screw_geometry(linear, angular)
    assert abs(base["lever_arm"] - 1.0) < 1.0e-12
    # Scaling the whole generator is exactly what the solver's unit gauge does;
    # the lever arm is the quantity that survives it.
    for factor in (1.0e-3, 7.0, 1.0e4):
        scaled = screw_geometry(linear * factor, angular * factor)
        assert abs(scaled["lever_arm"] - base["lever_arm"]) < 1.0e-9
    # Axial motion goes to pitch, not to the lever arm.
    pure_screw = screw_geometry(torch.tensor([0.0, 0.0, 3.0], dtype=torch.float64), angular)
    assert pure_screw["lever_arm"] < 1.0e-12
    assert abs(pure_screw["pitch"] - 1.5) < 1.0e-12


def _observation_pack(points, linear, angular, reference, speeds, noise=0.0, seed=5):
    """Exact Killing velocities for one generator, optionally with noise."""
    generator = torch.Generator().manual_seed(seed)
    positions = points[None].expand(len(speeds), -1, -1).clone()
    velocity = torch.stack(
        [
            speed
            * (
                torch.as_tensor(linear, dtype=torch.float64)
                + torch.cross(
                    torch.as_tensor(angular, dtype=torch.float64).expand_as(points),
                    points - torch.as_tensor(reference, dtype=torch.float64),
                    dim=-1,
                )
            )
            for speed in speeds
        ]
    )
    if noise:
        velocity = velocity + torch.randn(
            velocity.shape, generator=generator, dtype=torch.float64
        ) * noise
    return {
        "positions": positions,
        "velocity": velocity,
        "confidence": torch.ones(velocity.shape[:2], dtype=torch.float64),
        "valid": torch.ones(velocity.shape[:2], dtype=torch.bool),
    }


def test_translation_only_field_recovers_a_pure_slide():
    """The prismatic hypothesis is fitted, not derived from the rigid one."""
    generator = torch.Generator().manual_seed(3)
    points = torch.randn(300, 3, generator=generator, dtype=torch.float64) * 0.1
    speeds = [0.4, -0.5, 0.45, 0.5]
    pack = _observation_pack(points, [0.0, 1.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], speeds)
    mode, solved, diagnostics = solve_translation_mode_all_intervals(
        pack["positions"], pack["velocity"], min_valid_cells=8
    )
    assert mode.valid
    assert float(torch.as_tensor(mode.angular_mode).abs().max()) == 0.0, "omega must be exactly zero"
    assert diagnostics["translation_rms"] < 1.0e-12
    # Direction and signed speeds are recovered, including the reversal.
    assert float((torch.as_tensor(solved) - torch.tensor(speeds, dtype=torch.float64)).abs().max()) < 1.0e-9


def test_joint_type_is_selected_by_nested_model_comparison():
    """Both types are fitted; the one the velocity supports wins.

    The rigid family contains the translation family, so a bare residual
    comparison always favours rotation once there is noise.  The penalty for the
    rigid generator's three extra degrees of freedom is what makes the
    comparison correct, and it is read off the data rather than tuned.
    """
    generator = torch.Generator().manual_seed(5)
    points = torch.randn(400, 3, generator=generator, dtype=torch.float64) * 0.1
    speeds = [0.4, -0.5, 0.45, 0.5]
    weights = torch.ones(points.shape[0], dtype=torch.float64)

    def call(linear, angular, reference, noise):
        pack = _observation_pack(points, linear, angular, reference, speeds, noise=noise)
        field = fit_region_rigid_field(pack, weights, solver_kwargs={"min_valid_cells": 8})
        return field["joint_type_evidence"]

    slide = ([0.0, 1.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0])
    for noise in (0.0, 0.05, 0.2):
        evidence = call(*slide, noise)
        assert evidence["joint_type"] == "prismatic", (noise, evidence)
        assert evidence["log_evidence_difference"] < 0.0
        assert evidence["revolute_probability"] < 0.5
    # A bare residual comparison gets the noisy slide wrong, which is why the
    # penalty is there: the rigid fit is the better fit and the worse model.
    noisy = call(*slide, 0.05)
    assert noisy["rigid_residual"] <= noisy["translation_residual"]

    # Hinges are found at every distance, including ones far outside the part
    # where the rotation looks almost uniform.  A ratio-based test cannot do
    # this; the residual can, because a distant rotation still leaves a velocity
    # gradient across the part that no translation reproduces.
    for lever in (0.0, 0.3, 3.0, 30.0):
        for noise in (0.0, 0.05, 0.2):
            evidence = call([0.0, 0.0, 0.0], [0.0, 0.0, 1.0], [lever, 0.0, 0.0], noise)
            assert evidence["joint_type"] == "revolute", (lever, noise, evidence)
            assert evidence["log_evidence_difference"] > 0.0
            assert evidence["revolute_probability"] > 0.5


def test_nested_model_penalty_has_no_tuned_constant():
    """The penalty is EXTRA_RIGID_DOF * log(n), derived from the two models."""
    import math

    from utils.kpart_field_merge import (
        EXTRA_RIGID_DOF,
        RIGID_GENERATOR_DOF,
        TRANSLATION_GENERATOR_DOF,
    )

    # The charged freedom is the difference between the two representations,
    # not a number someone chose.
    assert RIGID_GENERATOR_DOF == 5 and TRANSLATION_GENERATOR_DOF == 2
    assert EXTRA_RIGID_DOF == RIGID_GENERATOR_DOF - TRANSLATION_GENERATOR_DOF

    # Equal residuals: the extra freedom bought nothing, so the simpler model
    # wins by exactly the penalty and nothing else.
    equal = compare_rigid_and_translation_evidence(1.0, 1.0, 500, 3)
    assert equal["joint_type"] == "prismatic"
    assert equal["scalar_residual_count"] == 1500
    assert abs(equal["log_likelihood_ratio"]) < 1.0e-12
    assert abs(equal["log_evidence_difference"] + EXTRA_RIGID_DOF * math.log(1500)) < 1.0e-9

    # The break-even residual ratio follows from n and the parameter counts
    # alone, so there is nothing to calibrate.
    n = 1500
    ratio = math.exp(EXTRA_RIGID_DOF * math.log(n) / n)
    assert compare_rigid_and_translation_evidence(1.0, ratio * 0.999, 500, 3)["joint_type"] == "prismatic"
    assert compare_rigid_and_translation_evidence(1.0, ratio * 1.001, 500, 3)["joint_type"] == "revolute"

    # More observations make the same relative improvement more convincing.
    small = compare_rigid_and_translation_evidence(1.0, 1.5, 50, 3)
    large = compare_rigid_and_translation_evidence(1.0, 1.5, 5000, 3)
    assert large["log_evidence_difference"] > small["log_evidence_difference"]

    # The residual dimension is read from the data, not assumed to be three.
    assert compare_rigid_and_translation_evidence(1.0, 1.0, 500, 2)["scalar_residual_count"] == 1000


def test_evidence_limits_are_exact_not_placeheld():
    """Zero residuals and empty observations have defined answers, not stand-ins."""
    import math

    # The rotation fits exactly and the translation does not: unbounded support
    # for revolute, reported as such rather than clipped to some large number.
    exact = compare_rigid_and_translation_evidence(0.0, 1.0e-3, 500, 3)
    assert exact["log_evidence_difference"] == math.inf
    assert exact["revolute_probability"] == 1.0
    assert exact["joint_type"] == "revolute"

    # Both fit exactly: the models are indistinguishable, so the likelihood
    # ratio is one and only the penalty remains.  That is the correct nested
    # answer, not a fallback.
    both = compare_rigid_and_translation_evidence(0.0, 0.0, 500, 3)
    assert both["log_likelihood_ratio"] == 0.0
    assert both["joint_type"] == "prismatic"
    assert both["log_evidence_difference"] < 0.0

    # A single observation is weak evidence, reported as weak, not refused by a
    # minimum-count cutoff.
    weak = compare_rigid_and_translation_evidence(1.0e-3, 2.0e-3, 1, 3)
    assert weak["joint_type"] == "prismatic"
    assert 0.0 < weak["revolute_probability"] < 0.5

    # No observations is an absence of evidence.  Nothing is invented: the
    # caller is handed None and must say what it does instead.
    empty = compare_rigid_and_translation_evidence(1.0e-3, 2.0e-3, 0, 3)
    assert empty["joint_type"] is None
    assert empty["revolute_probability"] is None
    assert empty["log_evidence_difference"] is None
    assert empty["reason"] == "no_valid_velocity_observations"

    # Every one of them survives a JSON dump that rejects non-finite numbers.
    import json

    for record in (exact, both, weak, empty):
        json.dumps(_finite_json(record), allow_nan=False)


def test_joint_axis_is_located_per_type():
    """Prismatic anchors at the part centroid, revolute at the rotation centre."""
    generator = torch.Generator().manual_seed(31)
    points = torch.randn(300, 3, generator=generator, dtype=torch.float64) * 0.1
    points = points + torch.tensor([5.0, 0.0, 2.0], dtype=torch.float64)
    centroid = points.mean(dim=0)
    extent = part_extent(points)

    # A hinge about +z whose axis passes through x = centroid_x - extent.
    axis_point = centroid - torch.tensor([extent, 0.0, 0.0], dtype=torch.float64)
    axis_point[2] = 0.0
    omega = torch.tensor([0.0, 0.0, 1.0], dtype=torch.float64)
    hinge = _mode(torch.cross(omega, centroid - axis_point, dim=-1), omega, centroid)
    fitted = fit_revolute_joint(hinge, torch.zeros(4, dtype=torch.float64), points)
    # The pivot lands on the true axis line, slid along it to the part.
    assert float((fitted["pivot"][:2] - axis_point[:2]).abs().max()) < 1.0e-9, fitted["pivot"]
    assert abs(float(fitted["pivot"][2]) - float(centroid[2])) < 1.0e-9
    assert float((fitted["axis"] - omega).abs().max()) < 1.0e-9

    # A slide has no unique axis line, so it is anchored at the part centroid.
    sliding = RigidKillingMode(
        reference_point=torch.zeros(3, dtype=torch.float64),
        reference_velocity=torch.tensor([0.0, 1.0, 0.0], dtype=torch.float64),
        angular_mode=torch.zeros(3, dtype=torch.float64),
        mode_type="translational",
        normalization_scale=1.0,
        valid=True,
        diagnostics={},
    )
    slid = fit_prismatic_joint(sliding, torch.zeros(4, dtype=torch.float64), points)
    assert float((slid["pivot"] - centroid).abs().max()) < 1.0e-9
    expected_axis = torch.tensor([0.0, 1.0, 0.0], dtype=torch.float64)
    assert float((slid["axis"] - expected_axis).abs().max()) < 1.0e-9


def test_instantaneous_axis_is_gauge_free():
    """Re-expressing the field about another point must not move the axis line."""
    generator = torch.Generator().manual_seed(19)
    points = torch.randn(300, 3, generator=generator, dtype=torch.float64) * 0.1
    base = _mode([0.0, 0.3, 0.0], [0.0, 0.0, 1.0], points.mean(dim=0))
    line = instantaneous_axis(base)
    reference = fit_revolute_joint(base, torch.zeros(4, dtype=torch.float64), points)
    for shift in ([5.0, 0.0, 0.0], [0.0, -12.0, 3.0], [80.0, 40.0, -7.0]):
        moved = re_reference_rigid_mode(
            base, base.reference_point + torch.tensor(shift, dtype=torch.float64)
        )
        raw = instantaneous_axis(moved)
        offset = raw["pivot"] - line["pivot"]
        along = torch.dot(offset, line["axis"]) * line["axis"]
        assert float((offset - along).norm()) < 1.0e-9, "the axis line must not move"
        assert float((raw["axis"] - line["axis"]).norm()) < 1.0e-9
        # Anchoring at the part makes the reported pivot itself gauge-free.
        other = fit_revolute_joint(moved, torch.zeros(4, dtype=torch.float64), points)
        assert float((other["pivot"] - reference["pivot"]).abs().max()) < 1.0e-9
        assert float((other["axis"] - reference["axis"]).abs().max()) < 1.0e-9


def test_prismatic_pivot_is_reported_not_optimised():
    """A prismatic transform ignores the pivot, so it must not be a parameter."""
    points = torch.randn(200, 3, dtype=torch.float64) * 0.1
    points = points + torch.tensor([5.0, 0.0, 0.0], dtype=torch.float64)
    centroid = points.mean(dim=0)
    candidate = {
        "joint_type": "prismatic",
        "axis": torch.tensor([0.0, 1.0, 0.0], dtype=torch.float64),
        "pivot": centroid,
        "pitch": 0.0,
        "coarse_q": torch.zeros(4, dtype=torch.float64),
    }
    model = _make_joint_model(candidate, points, anchor=centroid)
    assert model.pivot_raw is None
    assert float((model.pivot - centroid).abs().max()) < 1.0e-12
    model.set_trainable("geometry")
    trainable = {name for name, value in model.named_parameters() if value.requires_grad}
    assert trainable == {"axis_raw"}, trainable
    _, translation, _ = model.incremental_transform(torch.tensor(0.7, dtype=torch.float64))
    expected = 0.7 * torch.tensor([0.0, 1.0, 0.0], dtype=torch.float64)
    assert float((translation - expected).abs().max()) < 1.0e-12

    # A revolute pivot slides along its own axis to the part, not to the origin.
    revolute = {
        "joint_type": "revolute",
        "axis": torch.tensor([0.0, 0.0, 1.0], dtype=torch.float64),
        "pivot": torch.tensor([5.0, 0.0, 900.0], dtype=torch.float64),
        "pitch": 0.0,
        "coarse_q": torch.zeros(4, dtype=torch.float64),
    }
    anchored = _make_joint_model(revolute, points, anchor=centroid)
    assert abs(float(anchored.pivot[2]) - float(centroid[2])) < 1.0e-9
    assert abs(float(_make_joint_model(revolute, points).pivot[2])) < 1.0e-9


def _rigid_observations(linear, angular, reference, points, speeds):
    """Exact Killing velocities for one generator at a list of speeds."""
    positions = points[None].expand(len(speeds), -1, -1).clone()
    velocity = torch.stack(
        [
            speed
            * (
                torch.as_tensor(linear, dtype=torch.float64)
                + torch.cross(
                    torch.as_tensor(angular, dtype=torch.float64).expand_as(points),
                    points - torch.as_tensor(reference, dtype=torch.float64),
                    dim=-1,
                )
            )
            for speed in speeds
        ]
    )
    return positions, velocity


def test_raw_magnitudes_survive_the_unit_gauge():
    """The regression this guards: mode_type was always "rotational".

    ``normalize_rigid_mode`` is called once per solver iteration on a generator
    the previous iteration already normalized, so its own magnitudes read 1.0
    from the second iteration onward.  The absolute ``angular_eps`` test then
    always chose the rotational gauge, and the recorded
    "before_normalization" magnitudes were literally 1.0 in every run.  The
    solver now preserves the seed generator's true magnitudes, and the joint
    type is decided from the two physical speeds instead of the gauge.
    """
    generator = torch.Generator().manual_seed(11)
    points = torch.randn(300, 3, generator=generator, dtype=torch.float64) * 0.1
    speeds = [0.4, 0.5, 0.45, 0.5]

    # A pure translation along +y.
    positions, velocity = _rigid_observations(
        [0.0, 1.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], points, speeds
    )
    mode, solved_speeds, _ = solve_rigid_killing_mode_all_intervals(
        positions, velocity, min_valid_cells=8
    )
    assert mode.valid
    assert "raw_angular_norm" in mode.diagnostics

    # A hinge whose axis passes through the part.
    positions, velocity = _rigid_observations(
        [0.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, 0.0, 0.0], points, speeds
    )
    mode, solved_speeds, _ = solve_rigid_killing_mode_all_intervals(
        positions, velocity, min_valid_cells=8
    )
    assert mode.valid
    # The unit gauge puts the physical magnitude in the speeds either way.
    recovered = torch.as_tensor(solved_speeds).abs()
    assert float((recovered - torch.tensor(speeds, dtype=torch.float64)).abs().max()) < 1.0e-6


def test_profile_scale_recovers_amplitude_only():
    """One scale is the only reparameterisation the q[:, 0] = 0 gauge admits."""
    times = torch.linspace(0.0, 1.0, 60, dtype=torch.float64)
    profile = (times**1.5)[None]
    target = 2.5 * profile

    def objective(q):
        loss = ((q - target) ** 2).mean()
        return loss, {
            "render_loss": loss,
            "temporal_loss": loss * 0.0,
            "valid_weight": torch.tensor(1.0),
        }

    with contextlib.redirect_stderr(io.StringIO()):
        q, payload, history = optimize_profile_scale_q(profile, objective, 600, 0.05)
    assert float(q[0, 0]) == 0.0, "the q[:, 0] = 0 gauge must hold exactly"
    ratio = float(q.abs().max() / profile.abs().max())
    assert abs(ratio - 2.5) < 1.0e-3, ratio
    correlation = float(torch.corrcoef(torch.stack([q[0], target[0]]))[0, 1])
    assert abs(correlation - 1.0) < 1.0e-9, "the profile shape must be preserved"
    assert history[-1]["loss"] < 1.0e-12
    assert set(payload) == {"profile", "profile_scale"}
    assert abs(float(payload["profile_scale"][0, 0]) - 2.5) < 1.0e-3

    # Two frames leave exactly one unknown: the total displacement.
    two = torch.tensor([[0.0, 0.18]], dtype=torch.float64)
    endpoint = torch.tensor([[0.0, 0.532]], dtype=torch.float64)

    def two_frame(q):
        loss = ((q - endpoint) ** 2).mean()
        return loss, {
            "render_loss": loss,
            "temporal_loss": loss * 0.0,
            "valid_weight": torch.tensor(1.0),
        }

    with contextlib.redirect_stderr(io.StringIO()):
        q2, payload2, _ = optimize_profile_scale_q(two, two_frame, 800, 0.02)
    assert float(q2[0, 0]) == 0.0
    assert abs(float(q2[0, 1]) - 0.532) < 1.0e-6
    assert abs(float(payload2["profile_scale"][0, 0]) - 0.532 / 0.18) < 1.0e-4


def test_monotonic_trajectory_removes_phase1_backtracking():
    times = torch.linspace(0.0, 1.0, 8, dtype=torch.float64)
    bad_profile = torch.tensor([[0.0, 0.1, 0.08, 0.22, 0.21, 0.35, 0.34, 0.5]], dtype=torch.float64)
    target = torch.linspace(0.0, 0.6, 8, dtype=torch.float64)[None]

    class Args:
        _joint_frame_times = times
        joint_trajectory_acceleration_weight = 1.0e-5
        joint_trajectory_jerk_weight = 1.0e-8
        joint_trajectory_reverse_weight = 1.0e-2
        joint_trajectory_monotonic = True

    def objective(q):
        loss = ((q - target) ** 2).mean()
        return loss, {
            "render_loss": loss,
            "temporal_loss": loss * 0.0,
            "acceleration_loss": loss * 0.0,
            "jerk_loss": loss * 0.0,
            "reverse_velocity_loss": loss * 0.0,
            "valid_weight": torch.tensor(1.0),
        }

    with contextlib.redirect_stderr(io.StringIO()):
        q, qdot, _, payload, _ = optimize_monotonic_q(bad_profile, objective, times, 300, 0.03)
    assert float(q[0, 0]) == 0.0
    assert bool((torch.diff(q[0]) >= -1.0e-8).all())
    assert bool((qdot[0] >= -1.0e-8).all())
    assert abs(float(q[0, -1]) - 0.6) < 2.0e-2
    assert set(payload) == {"direction", "increments", "initial_positive_increments"}


def test_velocity_snr_ignores_rejected_observations():
    velocity = torch.zeros(3, 5, 3, dtype=torch.float64)
    velocity[:, :2, 0] = 0.12
    velocity[:, 2:, 0] = 0.005
    labels = torch.tensor([1, 1, 0, 0, 0])
    clean = observed_velocity_snr(
        {"velocity": velocity, "valid": torch.ones(3, 5, dtype=torch.bool)}, labels
    )
    assert abs(clean["snr"] - 24.0) < 1.0e-9
    # Rejected observations are stored as exact zeros; reading them as "static"
    # would understate the static baseline and inflate the ratio.
    masked_velocity = velocity.clone()
    masked_velocity[0] = 0.0
    valid = torch.ones(3, 5, dtype=torch.bool)
    valid[0] = False
    masked = observed_velocity_snr({"velocity": masked_velocity, "valid": valid}, labels)
    assert abs(masked["snr"] - clean["snr"]) < 1.0e-9
    assert abs(masked["observation_valid_fraction"] - 2.0 / 3.0) < 1.0e-9


def test_otsu_prior_threshold_rescues_a_miscalibrated_prior():
    class Args:
        kpart_prior_anchor_threshold = 0.6
        kpart_confidence_threshold = 0.5
        kpart_prior_boundary_expansion = 0
        kpart_prior_cleanup_min_size = 1
        kpart_prior_threshold_mode = "otsu"

    args = Args()
    edges = torch.zeros((0, 2), dtype=torch.long)

    def anchors(probability, mode):
        args.kpart_prior_threshold_mode = mode
        bootstrap = {
            "probability": probability,
            "confidence": torch.ones(probability.numel()),
            "valid": torch.ones(probability.numel(), dtype=torch.bool),
        }
        result = _prior_anchor_masks(bootstrap, edges, args)
        assert result["threshold"]["requested_mode"] == mode
        return int(result["static_anchor"].sum()), int(result["dynamic_anchor"].sum())

    # A well-ranked but badly scaled prior: the dynamic mode sits at 0.35, so a
    # fixed 0.5 split calls the whole scene static and there are no dynamic
    # anchors at all.  This is the shape that fully collapsed one measured run.
    skewed = torch.cat([torch.full((900,), 0.02), torch.full((100,), 0.35)])
    assert anchors(skewed, "fixed") == (1000, 0)
    assert anchors(skewed, "otsu") == (900, 100)
    # A prior that really is well calibrated is unaffected.
    calibrated = torch.cat([torch.full((900,), 0.05), torch.full((100,), 0.95)])
    assert anchors(calibrated, "fixed") == (900, 100)
    assert anchors(calibrated, "otsu") == (900, 100)
    # A completely collapsed prior carries no split; falling back to all-static
    # is the honest answer and lets the stage fail loudly.
    assert anchors(torch.full((1000,), 1.0e-23), "otsu") == (1000, 0)

    # The split is exact rather than histogrammed, so no bin count enters it.
    # Two modes a nanometre apart are separated, which a 64-bin histogram over
    # [0, 1] could not do -- both would land in the first bin.
    assert 0.0 <= _otsu_threshold(skewed) <= 1.0
    tight = torch.cat([torch.full((5,), 1.0e-23), torch.full((5,), 1.0e-23 + 1.0e-9)])
    split = _otsu_threshold(tight)
    assert split is not None and 0.0 < split < 1.0e-9
    # And where there is genuinely nothing to split, None is returned so the
    # caller names its own fallback instead of receiving a stand-in value.
    assert _otsu_threshold(torch.full((100,), 1.0e-23)) is None
    assert _otsu_threshold(torch.tensor([0.3])) is None
    assert _otsu_threshold(torch.tensor([])) is None


def test_strict_endpoint_prior_is_dynamic_seed_only():
    class Args:
        kpart_prior_anchor_threshold = 0.8
        kpart_confidence_threshold = 0.5
        kpart_prior_boundary_expansion = 0
        kpart_prior_cleanup_min_size = 1
        kpart_prior_threshold_mode = "fixed"

    bootstrap = {
        "method": "strict_closed_loop_endpoint_fm",
        "probability": torch.tensor([0.01, 0.2, 0.6, 0.85, 0.99]),
        "confidence": torch.ones(5),
        "valid": torch.ones(5, dtype=torch.bool),
    }
    result = _prior_anchor_masks(bootstrap, torch.zeros((0, 2), dtype=torch.long), Args())
    assert int(result["static_anchor"].sum()) == 0
    assert result["dynamic_anchor"].tolist() == [False, False, False, True, True]
    assert torch.equal(result["active_mask"], result["dynamic_anchor"])
    assert result["threshold"]["mode"] == "dynamic_seed_only"

    bootstrap["method"] = "strict_closed_loop_rgbd_endpoint_fm"
    result = _prior_anchor_masks(bootstrap, torch.zeros((0, 2), dtype=torch.long), Args())
    assert int(result["static_anchor"].sum()) == 0
    assert result["dynamic_anchor"].tolist() == [False, False, False, True, True]
    assert result["threshold"]["mode"] == "dynamic_seed_only"


def main():
    tests = [value for name, value in sorted(globals().items()) if name.startswith("test_")]
    for test in tests:
        test()
        print(f"ok  {test.__name__}")
    print(f"\n{len(tests)} tests passed")


if __name__ == "__main__":
    main()
