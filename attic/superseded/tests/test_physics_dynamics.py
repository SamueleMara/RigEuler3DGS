"""Tests for CODEX_IMPLEMENT_SPARSE_ALPHA_AWARE_IEKF section 32 + the `physics/` package of section 23.

Covers the mandatory suite entries T20 (force-enabled prismatic baseline) and
T21 (force-enabled revolute baseline) plus the switch/regressor contract of
32.1, the wrench projections of 32.2/32.3, and the constrained WLS / RLS
conditioning requirements of 32.4.

Everything runs on CPU in float64 and uses finite differences -- never autograd --
as the derivative oracle, matching the autograd policy of section 36.
"""

from __future__ import annotations

import ast
import math
from pathlib import Path

import torch

import physics
from physics.rls import RecursiveLeastSquares, constrained_wls
from physics.viscous_models import (
    PhysicsConfig,
    acceleration_parameter_jacobian,
    acceleration_state_jacobian,
    build_regressor,
    central_difference_rates,
    coulomb_basis,
    inverse_dynamics,
    known_load,
    parameter_bounds,
    parameter_names,
    predict_acceleration,
    regression_target,
)
from physics.wrench import (
    gravity_basis_prismatic,
    gravity_basis_revolute,
    prismatic_generalized_force,
    revolute_generalized_force,
    rotate_about_axis,
    unit_axis,
)

torch.manual_seed(20261004)
DT = torch.float64
FRAME_DT = 1.0 / 30.0  # real video frame spacing -- spec rule 0.11 forbids unit dt

PASSED: list[str] = []


def check(name: str, condition: bool, detail: str = "") -> None:
    if not condition:
        raise AssertionError(f"{name} FAILED {detail}")
    PASSED.append(name)
    print(f"  ok  {name}" + (f"   [{detail}]" if detail else ""))


# ---------------------------------------------------------------------------
# synthetic trajectories on a real time grid
# ---------------------------------------------------------------------------
def trajectory(n: int = 90, dt: float = FRAME_DT) -> tuple[torch.Tensor, ...]:
    """Analytic q(t), qdot(t), qddot(t) -- exact derivatives, so the only error
    in a recovered parameter is the solver's."""
    t = torch.arange(n, dtype=DT) * dt
    q = 0.30 * torch.sin(2.0 * t) + 0.11 * torch.cos(5.0 * t) + 0.05 * t
    qdot = 0.60 * torch.cos(2.0 * t) - 0.55 * torch.sin(5.0 * t) + 0.05
    qddot = -1.20 * torch.sin(2.0 * t) - 2.75 * torch.cos(5.0 * t)
    return t, q, qdot, qddot


def axis(vec=(0.2, -0.5, 0.84)) -> torch.Tensor:
    return unit_axis(torch.tensor(vec, dtype=DT))


# ---------------------------------------------------------------------------
# 32.1 switch defaults
# ---------------------------------------------------------------------------
def test_default_switches_match_spec_32_1():
    cfg = PhysicsConfig()
    check("viscous damping defaults ON", cfg.use_viscous_damping is True)
    check("estimated viscous b defaults ON", cfg.estimate_viscous_b is True)
    check("gravity defaults OFF", cfg.use_gravity is False)
    check("Coulomb/static friction defaults OFF", cfg.use_coulomb_friction is False)
    check("the force path itself is opt-in", cfg.use_wrench_dynamics is False)

    class Args:
        use_wrench_dynamics = True
        use_gravity = True

    from_cli = PhysicsConfig.from_namespace(Args())
    check("from_namespace honours section-33 flags", from_cli.use_gravity and from_cli.use_wrench_dynamics)
    check("from_namespace keeps 32.1 defaults for unset flags",
          from_cli.use_viscous_damping and from_cli.estimate_viscous_b
          and from_cli.use_coulomb_friction is False)


# ---------------------------------------------------------------------------
# 32.2 / 32.3 wrench projection
# ---------------------------------------------------------------------------
def test_wrench_projection():
    e = axis()
    magnitude = 3.7

    f_along = magnitude * e
    u = prismatic_generalized_force(e, f_along)
    check("prismatic: force along the axis gives u == |F|",
          abs(float(u) - magnitude) < 1e-14, f"u={float(u):.16f}")

    perp = torch.linalg.cross(e, torch.tensor([1.0, 0.0, 0.0], dtype=DT), dim=-1)
    perp = 11.0 * perp / torch.linalg.vector_norm(perp)
    check("prismatic: perpendicular force gives u == 0",
          abs(float(prismatic_generalized_force(e, perp))) < 1e-14)
    check("prismatic: projection is linear in F",
          abs(float(prismatic_generalized_force(e, f_along + perp)) - magnitude) < 1e-14)

    batch = torch.stack([f_along, perp, -f_along])
    u_batch = prismatic_generalized_force(e, batch)
    check("prismatic: projection broadcasts over a frame batch",
          u_batch.shape == (3,) and abs(float(u_batch[2]) + magnitude) < 1e-14)

    scaled_axis = 7.5 * e  # a non-unit axis must not rescale the generalized force
    check("prismatic: axis is normalized before projecting",
          abs(float(prismatic_generalized_force(scaled_axis, f_along)) - magnitude) < 1e-13)

    a = torch.tensor([0.4, -1.2, 0.7], dtype=DT)
    p = a + torch.tensor([0.3, 0.9, -0.2], dtype=DT)
    tau_mag = 2.3
    tau = tau_mag * e
    u_tau = revolute_generalized_force(e, a, p, torch.zeros(3, dtype=DT), tau)
    check("revolute: torque about the axis gives u == |tau|",
          abs(float(u_tau) - tau_mag) < 1e-14, f"u={float(u_tau):.16f}")

    tau_perp = tau_mag * perp / torch.linalg.vector_norm(perp)
    check("revolute: torque perpendicular to the axis gives u == 0",
          abs(float(revolute_generalized_force(e, a, p, torch.zeros(3, dtype=DT), tau_perp))) < 1e-14)

    radial = p - a
    f_radial = 5.0 * radial / torch.linalg.vector_norm(radial)
    check("revolute: a purely radial force exerts no generalized torque",
          abs(float(revolute_generalized_force(e, a, p, f_radial))) < 1e-14)

    f_gen = torch.tensor([1.3, -0.7, 2.1], dtype=DT)
    u_a = float(revolute_generalized_force(e, a, p, f_gen))
    u_a2 = float(revolute_generalized_force(e, a + 3.3 * e, p, f_gen))
    check("revolute: result is independent of where on the axis `a` sits",
          abs(u_a - u_a2) < 1e-13, f"{u_a:.12f} vs {u_a2:.12f}")

    lever = torch.linalg.cross(e, radial, dim=-1)
    lever = lever / torch.linalg.vector_norm(lever)
    f_tangential = 4.0 * lever
    expected = 4.0 * float(torch.linalg.vector_norm(torch.linalg.cross(e, radial, dim=-1)))
    check("revolute: tangential force gives |r_perp| * |F|",
          abs(float(revolute_generalized_force(e, a, p, f_tangential)) - expected) < 1e-13)

    check("revolute: generalized_force dispatches by joint type",
          abs(float(physics.generalized_force("revolute", e=e, a=a, p=p, F=f_gen)) - u_a) < 1e-15)


def test_gravity_basis_is_a_wrench_projection():
    e = axis()
    g = torch.tensor([0.0, -9.80665, 0.0], dtype=DT)
    gamma = gravity_basis_prismatic(e, g)
    check("prismatic gravity basis equals -(e . g)",
          abs(float(gamma) + float((e * g).sum())) < 1e-15)
    check("prismatic gravity basis vanishes for a horizontal axis",
          abs(float(gravity_basis_prismatic(torch.tensor([1.0, 0.0, 0.0], dtype=DT), g))) < 1e-15)

    lever0 = torch.tensor([0.42, 0.0, -0.11], dtype=DT)
    q = torch.tensor([0.0, 0.3, -1.1, 2.0], dtype=DT)
    gamma_r = gravity_basis_revolute(e, lever0, q, g)
    mass = 1.0
    for i, qi in enumerate(q):
        lever = rotate_about_axis(e, qi, lever0)
        direct = -mass * float(revolute_generalized_force(
            e, torch.zeros(3, dtype=DT), lever, mass * g))
        check(f"revolute gravity basis matches the wrench projection at q[{i}]",
              abs(float(gamma_r[i]) - direct) < 1e-14)
    check("rotating the lever preserves its length",
          abs(float(torch.linalg.vector_norm(rotate_about_axis(e, q[2], lever0)))
              - float(torch.linalg.vector_norm(lever0))) < 1e-14)


# ---------------------------------------------------------------------------
# 32.1: disabled terms are EXCLUDED from regressor and Jacobian
# ---------------------------------------------------------------------------
def test_disabled_terms_are_excluded_from_regressor_and_jacobian():
    _, _, qdot, qddot = trajectory(40)
    e = axis()
    g = torch.tensor([0.0, -9.80665, 0.0], dtype=DT)
    gamma = gravity_basis_prismatic(e, g).expand_as(qdot)

    base = PhysicsConfig()
    phi, names = build_regressor(qddot, qdot, base, "prismatic")
    check("gravity OFF: the regressor has exactly 2 columns",
          phi.shape == (40, 2) and names == ["m", "b"], f"names={names}")
    check("default columns are [qddot, qdot]",
          torch.equal(phi[:, 0], qddot) and torch.equal(phi[:, 1], qdot))

    grav = PhysicsConfig(use_gravity=True)
    phi_g, names_g = build_regressor(qddot, qdot, grav, "prismatic", gravity_basis=gamma)
    check("gravity ON: the regressor has exactly 3 columns",
          phi_g.shape == (40, 3) and names_g == ["m", "b", "m_g"], f"names={names_g}")
    check("the extra column IS the gravity basis", torch.allclose(phi_g[:, 2], gamma))

    check("Coulomb OFF: no sign(qdot) column exists",
          not any(torch.allclose(phi[:, j], torch.sign(qdot)) for j in range(phi.shape[1])))
    coul = PhysicsConfig(use_coulomb_friction=True)
    phi_c, names_c = build_regressor(qddot, qdot, coul, "prismatic")
    check("Coulomb ON: exactly one column is added",
          phi_c.shape[1] == phi.shape[1] + 1 and names_c == ["m", "b", "f_c"])
    check("the added Coulomb column is sign(qdot)", torch.equal(phi_c[:, 2], torch.sign(qdot)))
    check("revolute names the Coulomb term as a torque",
          parameter_names(coul, "revolute") == ["J_e", "b", "tau_c"])

    both = PhysicsConfig(use_gravity=True, use_coulomb_friction=True)
    phi_b, names_b = build_regressor(qddot, qdot, both, "prismatic", gravity_basis=gamma)
    check("both ON: 4 columns in the documented order",
          phi_b.shape[1] == 4 and names_b == ["m", "b", "m_g", "f_c"])

    no_damp = PhysicsConfig(use_viscous_damping=False)
    phi_nd, names_nd = build_regressor(qddot, qdot, no_damp, "prismatic")
    check("viscous damping OFF: the qdot column is gone",
          phi_nd.shape[1] == 1 and names_nd == ["m"])

    pinned = PhysicsConfig(estimate_viscous_b=False, viscous_b=0.45)
    phi_p, names_p = build_regressor(qddot, qdot, pinned, "prismatic")
    check("pinned b: no b column, but the load is still applied",
          phi_p.shape[1] == 1 and names_p == ["m"]
          and torch.allclose(known_load(qdot, pinned, "prismatic"), 0.45 * qdot))

    # The Jacobian must follow the same gating (spec 32.1).
    theta = torch.tensor([2.0, 0.5], dtype=DT)
    u = inverse_dynamics(theta, qddot, qdot, base, "prismatic")
    jac, jac_names = acceleration_parameter_jacobian(theta, qdot, u, base, "prismatic")
    check("Jacobian has one column per enabled parameter",
          jac.shape == (40, 2) and jac_names == names)
    theta_g = torch.tensor([2.0, 0.5, 1.1], dtype=DT)
    u_g = inverse_dynamics(theta_g, qddot, qdot, grav, "prismatic", gravity_basis=gamma)
    jac_g, _ = acceleration_parameter_jacobian(theta_g, qdot, u_g, grav, "prismatic", gravity_basis=gamma)
    check("turning gravity on widens the Jacobian by exactly one column",
          jac_g.shape[1] == jac.shape[1] + 1)

    try:
        build_regressor(qddot, qdot, grav, "prismatic")
    except ValueError as exc:
        check("gravity ON without a gravity basis is refused, not faked", "gravity_basis" in str(exc))
    else:  # pragma: no cover - guarded by the check above
        raise AssertionError("missing gravity basis must raise")

    try:
        build_regressor(qddot, qdot, base, "screw")
    except ValueError as exc:
        check("non prismatic/revolute joints are rejected (spec 0.13)", "screw" in str(exc))
    else:  # pragma: no cover
        raise AssertionError("screw joints must raise")


# ---------------------------------------------------------------------------
# T20: force-enabled prismatic baseline
# ---------------------------------------------------------------------------
def test_T20_prismatic_identification_from_wrench_data():
    m_true, b_true = 2.5, 0.8
    cfg = PhysicsConfig(use_wrench_dynamics=True)  # viscous ON, gravity OFF, Coulomb OFF
    _, _, qdot, qddot = trajectory()
    e = axis()

    # Build the actuating wrench explicitly, then project it: the identification
    # consumes exactly what the force path would deliver at runtime.
    u_required = m_true * qddot + b_true * qdot
    perp = torch.linalg.cross(e, torch.tensor([0.0, 0.0, 1.0], dtype=DT), dim=-1)
    perp = perp / torch.linalg.vector_norm(perp)
    forces = u_required.unsqueeze(-1) * e + 4.2 * perp  # the perpendicular part must cancel
    u = prismatic_generalized_force(e, forces)
    check("T20 wrench projection reproduces the actuating generalized force",
          float((u - u_required).abs().max()) < 1e-13)

    phi, names = build_regressor(qddot, qdot, cfg, "prismatic")
    lower, upper = parameter_bounds(cfg, "prismatic")
    res = constrained_wls(phi, regression_target(u, qdot, cfg, "prismatic"), lower, upper)
    err_m = abs(float(res.params[0]) - m_true)
    err_b = abs(float(res.params[1]) - b_true)
    check("T20 recovers m to < 1e-6 from noiseless data", err_m < 1e-6, f"|dm|={err_m:.3e}")
    check("T20 recovers b to < 1e-6 from noiseless data", err_b < 1e-6, f"|db|={err_b:.3e}")
    check("T20 recovered parameters are positive", float(res.params[0]) > 0 and float(res.params[1]) >= 0)
    check("T20 reports confident parameters on well-excited data",
          res.confident and not res.rank_deficient and res.rank == 2)
    check("T20 reports finite conditioning and non-zero excitation",
          math.isfinite(res.condition_number) and res.excitation > 0.0,
          f"cond={res.condition_number:.3f} pe={res.excitation:.3e}")
    check("T20 parameter names are [m, b]", names == ["m", "b"])
    params, *_ = res  # the result is a NamedTuple with params first, so `-> params` still holds
    check("T20 the result unpacks params first", torch.equal(params, res.params))

    # Weighted version: IRLS-style weights must not move a noiseless solution.
    weights = torch.linspace(0.2, 3.0, qdot.numel(), dtype=DT)
    res_w = constrained_wls(phi, u, lower, upper, weights)
    check("T20 is invariant to row weights on consistent data",
          float((res_w.params - res.params).abs().max()) < 1e-9)

    # Noisy data: still positive, still close, and the residual is reported.
    noise = 0.01 * torch.randn(u.shape, generator=torch.Generator().manual_seed(3), dtype=DT)
    res_n = constrained_wls(phi, u + noise, lower, upper)
    check("T20 stays within 1% of truth under 1% force noise",
          abs(float(res_n.params[0]) - m_true) / m_true < 0.01
          and abs(float(res_n.params[1]) - b_true) / b_true < 0.05,
          f"m={float(res_n.params[0]):.5f} b={float(res_n.params[1]):.5f}")
    check("T20 reports a non-zero residual rms on noisy data", res_n.residual_rms > 0.0)


# ---------------------------------------------------------------------------
# T21: force-enabled revolute baseline
# ---------------------------------------------------------------------------
def test_T21_revolute_identification_from_wrench_data():
    j_true, b_true = 0.3575, 0.12
    cfg = PhysicsConfig(use_wrench_dynamics=True)
    _, _, qdot, qddot = trajectory()
    e = axis((-0.3, 0.6, 0.74))
    a = torch.tensor([0.25, 1.4, -0.6], dtype=DT)
    p = a + torch.tensor([0.31, -0.22, 0.44], dtype=DT)

    u_required = j_true * qddot + b_true * qdot
    radial = p - a
    tangential = torch.linalg.cross(e, radial, dim=-1)
    lever_len = float(torch.linalg.vector_norm(tangential))
    tangential = tangential / lever_len
    forces = (u_required / lever_len).unsqueeze(-1) * tangential
    torques = 0.5 * u_required.unsqueeze(-1) * torch.linalg.cross(
        e, torch.tensor([1.0, 0.0, 0.0], dtype=DT), dim=-1)  # off-axis torque: must project to 0
    u = revolute_generalized_force(e, a, p, forces, torques)
    check("T21 wrench projection reproduces the actuating generalized torque",
          float((u - u_required).abs().max()) < 1e-12)

    phi, names = build_regressor(qddot, qdot, cfg, "revolute")
    lower, upper = parameter_bounds(cfg, "revolute")
    res = constrained_wls(phi, regression_target(u, qdot, cfg, "revolute"), lower, upper)
    err_j = abs(float(res.params[0]) - j_true)
    err_b = abs(float(res.params[1]) - b_true)
    check("T21 recovers J_e to < 1e-6", err_j < 1e-6, f"|dJ|={err_j:.3e}")
    check("T21 recovers b to < 1e-6", err_b < 1e-6, f"|db|={err_b:.3e}")
    check("T21 recovered parameters are positive",
          float(res.params[0]) > 0 and float(res.params[1]) >= 0)
    check("T21 parameter names are [J_e, b]", names == ["J_e", "b"])
    check("T21 reports confident parameters", res.confident and not res.rank_deficient)


# ---------------------------------------------------------------------------
# 32.4: the box constraints must actually bind
# ---------------------------------------------------------------------------
def test_box_constraints_bind():
    cfg = PhysicsConfig()
    _, _, qdot, qddot = trajectory(80)
    m_true, b_negative = 2.5, -0.9  # physically impossible: damping that injects energy
    u = m_true * qddot + b_negative * qdot
    phi, _ = build_regressor(qddot, qdot, cfg, "prismatic")
    lower, upper = parameter_bounds(cfg, "prismatic")

    free = constrained_wls(phi, u)
    check("the unconstrained solution really is negative in b",
          float(free.params[1]) < -0.5, f"b_free={float(free.params[1]):.6f}")

    res = constrained_wls(phi, u, lower, upper)
    check("constrained b is exactly 0, not negative",
          float(res.params[1]) == 0.0, f"b={float(res.params[1])!r}")
    check("the lower bound is reported as active", bool(res.active_lower[1]))
    check("the solver reports KKT convergence", res.converged)

    # With b pinned at its bound the remaining problem is a 1-column LS: check it.
    reference = float((phi[:, 0] @ u) / (phi[:, 0] @ phi[:, 0]))
    check("m equals the exact reduced-problem optimum",
          abs(float(res.params[0]) - reference) < 1e-12,
          f"{float(res.params[0]):.12f} vs {reference:.12f}")
    check("the constrained optimum is not just the clamped free solution",
          abs(float(res.params[0]) - float(free.params[0])) > 1e-3)

    # m > 0 must bind too: data that wants a negative mass.
    u_neg = -1.5 * qddot + 0.4 * qdot
    res_neg = constrained_wls(phi, u_neg, lower, upper)
    check("m is driven to its positive floor instead of going negative",
          float(res_neg.params[0]) == cfg.min_inertia and float(res_neg.params[0]) > 0.0,
          f"m={float(res_neg.params[0]):.3e}")

    # And a strictly interior solution must be left untouched by the box.
    res_in = constrained_wls(phi, 2.5 * qddot + 0.8 * qdot, lower, upper)
    check("an interior optimum is returned with no active constraints",
          not bool(res_in.active_lower.any()) and not bool(res_in.active_upper.any()))


# ---------------------------------------------------------------------------
# 32.4: rank deficiency must be reported, not papered over
# ---------------------------------------------------------------------------
def test_rank_deficient_regressor_is_reported():
    cfg = PhysicsConfig()
    qdot = torch.full((40,), 0.7, dtype=DT)       # constant speed ...
    qddot = torch.zeros(40, dtype=DT)             # ... so the inertia column is dead
    u = 0.8 * qdot
    phi, _ = build_regressor(qddot, qdot, cfg, "prismatic")
    lower, upper = parameter_bounds(cfg, "prismatic")
    res = constrained_wls(phi, u, lower, upper)

    check("rank deficiency is detected", res.rank_deficient and res.rank == 1)
    check("the condition number is reported as infinite", res.condition_number == math.inf)
    check("the persistent-excitation level is zero", res.excitation == 0.0)
    check("NO confident parameters are returned", res.confident is False)
    check("the returned parameters are still finite and feasible",
          bool(torch.isfinite(res.params).all()) and float(res.params[0]) >= cfg.min_inertia)
    check("the identifiable combination is still fitted",
          abs(float(res.params[1]) - 0.8) < 1e-9, f"b={float(res.params[1]):.12f}")

    # A single sample can never identify two parameters either.
    one = constrained_wls(phi[:1], u[:1], lower, upper)
    check("a single sample is reported as rank deficient", one.rank_deficient and not one.confident)

    # Sanity: the same machinery must say "confident" when excitation is present.
    _, _, qdot_e, qddot_e = trajectory(40)
    phi_e, _ = build_regressor(qddot_e, qdot_e, cfg, "prismatic")
    good = constrained_wls(phi_e, 2.0 * qddot_e + 0.3 * qdot_e, lower, upper)
    check("well-excited data is reported as confident", good.confident and good.rank == 2)

    # An explicit excitation floor must be able to veto a poorly excited fit.
    picky = constrained_wls(phi_e, 2.0 * qddot_e + 0.3 * qdot_e, lower, upper,
                            excitation_min=1.0e3)
    check("an excitation threshold can veto confidence",
          picky.confident is False and picky.rank == 2)


# ---------------------------------------------------------------------------
# 32.4: recursive least squares
# ---------------------------------------------------------------------------
def test_rls_matches_batch_wls():
    cfg = PhysicsConfig()
    m_true, b_true = 2.5, 0.8
    _, _, qdot, qddot = trajectory(120)
    u = m_true * qddot + b_true * qdot
    phi, names = build_regressor(qddot, qdot, cfg, "prismatic")
    lower, upper = parameter_bounds(cfg, "prismatic")
    batch = constrained_wls(phi, u, lower, upper)

    rls = RecursiveLeastSquares(2, lower_bounds=lower, upper_bounds=upper, param_names=names)
    report = rls.update_batch(phi, u)
    check("RLS consumed every sample", rls.n_samples == 120 and rls.effective_samples == 120.0)
    check("RLS converges to the batch WLS answer",
          float((rls.params - batch.params).abs().max()) < 1e-6,
          f"rls={rls.params.tolist()} batch={batch.params.tolist()}")
    check("RLS recovers the true m, b", abs(float(rls.params[0]) - m_true) < 1e-6
          and abs(float(rls.params[1]) - b_true) < 1e-6)
    check("RLS reports the regressor condition number",
          math.isfinite(rls.condition_number) and rls.condition_number > 1.0,
          f"cond={rls.condition_number:.4f}")
    check("RLS and batch agree on conditioning",
          abs(rls.condition_number - batch.condition_number) < 1e-6 * batch.condition_number,
          f"{rls.condition_number:.6f} vs {batch.condition_number:.6f}")
    check("RLS exposes a covariance and an information matrix",
          rls.covariance.shape == (2, 2) and rls.information.shape == (2, 2)
          and bool((torch.linalg.eigvalsh(rls.covariance) > 0).all()))
    check("the covariance shrinks as information accumulates",
          float(torch.linalg.matrix_norm(rls.covariance, 2)) < 1.0)
    check("the last update reports confidence and a vanishing innovation",
          report.confident and abs(report.innovation) < 1e-8)

    exact = rls.constrained_solution()
    check("the RLS constrained solution reproduces batch WLS exactly",
          float((exact.params - batch.params).abs().max()) < 1e-9,
          f"max|d|={float((exact.params - batch.params).abs().max()):.3e}")

    # Rank deficiency during streaming must be reported too.
    cold = RecursiveLeastSquares(2, lower_bounds=lower, upper_bounds=upper)
    first = cold.update(phi[0], float(u[0]))
    check("after one sample RLS reports rank deficiency and no confidence",
          first.rank_deficient and first.confident is False)
    cold.update(phi[1], float(u[1]))
    check("after two independent samples RLS is confident", cold.confident)

    # The box projection must apply to the reported parameters as well.
    bad = RecursiveLeastSquares(2, lower_bounds=lower, upper_bounds=upper)
    bad.update_batch(phi, m_true * qddot - 0.9 * qdot)
    check("raw RLS stays unconstrained (it is the pure recursion)", float(bad.params[1]) < 0.0)
    check("projected RLS parameters respect b >= 0", float(bad.projected_params[1]) == 0.0)
    check("the RLS constrained solve matches the batch constrained solve",
          float((bad.constrained_solution().params
                 - constrained_wls(phi, m_true * qddot - 0.9 * qdot, lower, upper).params
                 ).abs().max()) < 1e-9)

    # Forgetting factor: track a parameter change instead of averaging it away.
    drift = RecursiveLeastSquares(2, forgetting=0.90, lower_bounds=lower, upper_bounds=upper)
    drift.update_batch(phi[:60], u[:60])
    new_m, new_b = 1.1, 0.25
    drift.update_batch(phi[60:], new_m * qddot[60:] + new_b * qdot[60:])
    stale = RecursiveLeastSquares(2, forgetting=1.0, lower_bounds=lower, upper_bounds=upper)
    stale.update_batch(phi[:60], u[:60])
    stale.update_batch(phi[60:], new_m * qddot[60:] + new_b * qdot[60:])
    check("a forgetting factor lets RLS track a parameter change",
          abs(float(drift.params[0]) - new_m) < 1e-2 and abs(float(drift.params[1]) - new_b) < 1e-2,
          f"theta={drift.params.tolist()}")
    check("without forgetting the estimate averages the two regimes instead",
          abs(float(stale.params[0]) - new_m) > 10 * abs(float(drift.params[0]) - new_m),
          f"stale m={float(stale.params[0]):.4f}")
    check("forgetting discounts the effective sample count",
          drift.effective_samples < 60.0, f"n_eff={drift.effective_samples:.3f}")


# ---------------------------------------------------------------------------
# gravity / Coulomb: implemented behind flags, and identifiable when enabled
# ---------------------------------------------------------------------------
def test_optional_terms_identify_when_enabled():
    _, q, qdot, qddot = trajectory(120)
    g = torch.tensor([0.0, 0.0, -9.80665], dtype=DT)

    e = axis((0.0, 0.6, 0.8))
    cfg = PhysicsConfig(use_wrench_dynamics=True, use_gravity=True)
    gamma = gravity_basis_prismatic(e, g).expand_as(qdot)
    theta_true = torch.tensor([2.5, 0.8, 2.5], dtype=DT)  # m_g == m for a consistent body
    u = inverse_dynamics(theta_true, qddot, qdot, cfg, "prismatic", gravity_basis=gamma)
    phi, names = build_regressor(qddot, qdot, cfg, "prismatic", gravity_basis=gamma)
    lower, upper = parameter_bounds(cfg, "prismatic")
    res = constrained_wls(phi, u, lower, upper)
    check("prismatic gravity: [m, b, m_g] all recovered",
          float((res.params - theta_true).abs().max()) < 1e-6,
          f"{res.params.tolist()} names={names}")
    check("gravity load pulls the axis the physical way",
          float(gamma[0]) > 0.0 and res.confident)

    e_r = axis((0.1, 0.2, 0.97))
    cfg_r = PhysicsConfig(use_wrench_dynamics=True, use_gravity=True)
    lever0 = torch.tensor([0.4, 0.05, 0.0], dtype=DT)
    gamma_r = gravity_basis_revolute(e_r, lever0, q, g)
    theta_r = torch.tensor([0.3575, 0.12, 1.8], dtype=DT)
    u_r = inverse_dynamics(theta_r, qddot, qdot, cfg_r, "revolute", gravity_basis=gamma_r)
    phi_r, names_r = build_regressor(qddot, qdot, cfg_r, "revolute", gravity_basis=gamma_r)
    lo_r, hi_r = parameter_bounds(cfg_r, "revolute")
    res_r = constrained_wls(phi_r, u_r, lo_r, hi_r)
    check("revolute gravity: [J_e, b, m_g] all recovered",
          float((res_r.params - theta_r).abs().max()) < 1e-6,
          f"{res_r.params.tolist()} names={names_r}")
    check("the revolute gravity column actually varies with q",
          float(gamma_r.std()) > 1e-3)

    cfg_c = PhysicsConfig(use_wrench_dynamics=True, use_coulomb_friction=True)
    theta_c = torch.tensor([2.5, 0.8, 0.35], dtype=DT)
    u_c = inverse_dynamics(theta_c, qddot, qdot, cfg_c, "prismatic")
    phi_c, _ = build_regressor(qddot, qdot, cfg_c, "prismatic")
    lo_c, hi_c = parameter_bounds(cfg_c, "prismatic")
    res_c = constrained_wls(phi_c, u_c, lo_c, hi_c)
    check("Coulomb ON: [m, b, f_c] all recovered",
          float((res_c.params - theta_c).abs().max()) < 1e-6, f"{res_c.params.tolist()}")
    check("Coulomb friction is constrained non-negative", float(res_c.params[2]) >= 0.0)
    check("the Coulomb basis is exactly sign(qdot) by default",
          torch.equal(coulomb_basis(qdot, cfg_c), torch.sign(qdot)))
    smooth = PhysicsConfig(use_coulomb_friction=True, coulomb_smoothing_eps=0.05)
    check("the smoothed Coulomb basis is bounded and odd",
          float(coulomb_basis(qdot, smooth).abs().max()) <= 1.0
          and abs(float(coulomb_basis(torch.zeros(1, dtype=DT), smooth))) < 1e-15)

    # Fitting Coulomb-contaminated data with Coulomb OFF must bias the fit --
    # proof that the column is really absent rather than silently included.
    cfg_off = PhysicsConfig(use_wrench_dynamics=True)
    phi_off, _ = build_regressor(qddot, qdot, cfg_off, "prismatic")
    lo_off, hi_off = parameter_bounds(cfg_off, "prismatic")
    res_off = constrained_wls(phi_off, u_c, lo_off, hi_off)
    check("with Coulomb OFF the friction cannot be explained away",
          res_off.params.shape[0] == 2 and res_off.residual_rms > 1e-3,
          f"rms={res_off.residual_rms:.4e}")


# ---------------------------------------------------------------------------
# prediction and its Jacobians
# ---------------------------------------------------------------------------
def test_prediction_and_jacobians():
    _, q, qdot, qddot = trajectory(50)
    g = torch.tensor([0.0, -9.80665, 0.0], dtype=DT)
    e = axis()
    gamma = gravity_basis_prismatic(e, g).expand_as(qdot)

    cfg = PhysicsConfig(use_wrench_dynamics=True)
    theta = torch.tensor([2.5, 0.8], dtype=DT)
    u = inverse_dynamics(theta, qddot, qdot, cfg, "prismatic")
    back = predict_acceleration(theta, qdot, u, cfg, "prismatic")
    check("predict_acceleration inverts the dynamics exactly",
          float((back - qddot).abs().max()) < 1e-12)

    cfg_full = PhysicsConfig(use_wrench_dynamics=True, use_gravity=True, use_coulomb_friction=True)
    theta_full = torch.tensor([2.5, 0.8, 2.5, 0.35], dtype=DT)
    u_full = inverse_dynamics(theta_full, qddot, qdot, cfg_full, "prismatic", gravity_basis=gamma)
    back_full = predict_acceleration(theta_full, qdot, u_full, cfg_full, "prismatic", gravity_basis=gamma)
    check("the round trip holds with every optional term enabled",
          float((back_full - qddot).abs().max()) < 1e-12)

    pinned = PhysicsConfig(use_wrench_dynamics=True, estimate_viscous_b=False, viscous_b=0.8)
    u_pinned = inverse_dynamics(torch.tensor([2.5], dtype=DT), qddot, qdot, pinned, "prismatic")
    check("a pinned b produces the same physics as an estimated b",
          float((u_pinned - u).abs().max()) < 1e-12)

    # Parameter Jacobian against central differences (no autograd -- section 36).
    jac, jac_names = acceleration_parameter_jacobian(
        theta_full, qdot, u_full, cfg_full, "prismatic", gravity_basis=gamma)
    h = 1e-6
    for i, name in enumerate(jac_names):
        bump = torch.zeros_like(theta_full)
        bump[i] = h
        plus = predict_acceleration(theta_full + bump, qdot, u_full, cfg_full, "prismatic", gravity_basis=gamma)
        minus = predict_acceleration(theta_full - bump, qdot, u_full, cfg_full, "prismatic", gravity_basis=gamma)
        fd = (plus - minus) / (2 * h)
        err = float((fd - jac[:, i]).abs().max())
        check(f"d qddot/d {name} matches central differences", err < 1e-6, f"err={err:.3e}")

    # State Jacobian: d qddot / d qdot, and d qddot / dq through gravity.
    state = acceleration_state_jacobian(theta, qdot, cfg, "prismatic")
    expected = -float(theta[1]) / float(theta[0])
    check("d qddot/d qdot equals -b/m", float((state.d_dqdot - expected).abs().max()) < 1e-15)
    check("d qddot/dq is exactly zero with gravity off",
          float(state.d_dq.abs().max()) == 0.0)

    lever0 = torch.tensor([0.4, 0.05, 0.0], dtype=DT)
    e_r = axis((0.1, 0.2, 0.97))
    cfg_rg = PhysicsConfig(use_wrench_dynamics=True, use_gravity=True)
    theta_rg = torch.tensor([0.3575, 0.12, 1.8], dtype=DT)
    dq = 1e-6
    dgamma = (gravity_basis_revolute(e_r, lever0, q + dq, g)
              - gravity_basis_revolute(e_r, lever0, q - dq, g)) / (2 * dq)
    state_rg = acceleration_state_jacobian(theta_rg, qdot, cfg_rg, "revolute", dgravity_dq=dgamma)
    gamma_r = gravity_basis_revolute(e_r, lever0, q, g)
    u_rg = inverse_dynamics(theta_rg, qddot, qdot, cfg_rg, "revolute", gravity_basis=gamma_r)
    plus = predict_acceleration(theta_rg, qdot, u_rg, cfg_rg, "revolute",
                                gravity_basis=gravity_basis_revolute(e_r, lever0, q + dq, g))
    minus = predict_acceleration(theta_rg, qdot, u_rg, cfg_rg, "revolute",
                                 gravity_basis=gravity_basis_revolute(e_r, lever0, q - dq, g))
    err = float(((plus - minus) / (2 * dq) - state_rg.d_dq).abs().max())
    check("d qddot/dq matches central differences through the gravity basis",
          err < 1e-6, f"err={err:.3e}")

    try:
        acceleration_state_jacobian(theta_rg, qdot, cfg_rg, "revolute")
    except ValueError as exc:
        check("linearizing gravity without gamma'(q) is refused", "dgravity_dq" in str(exc))
    else:  # pragma: no cover
        raise AssertionError("missing dgravity_dq must raise")


# ---------------------------------------------------------------------------
# spec rule 0.11: real video dt
# ---------------------------------------------------------------------------
def test_real_dt_is_mandatory():
    # Exactness on a quadratic over a deliberately NON-uniform time grid.
    times = torch.tensor([0.0, 0.031, 0.070, 0.098, 0.140, 0.171, 0.210], dtype=DT)
    a0, a1, a2 = 0.3, -1.7, 2.9
    q = a0 + a1 * times + a2 * times ** 2
    rates = central_difference_rates(q, times)
    inner = rates.valid
    check("non-uniform central differences are exact for a quadratic (qdot)",
          float((rates.qdot[inner] - (a1 + 2 * a2 * times[inner])).abs().max()) < 1e-12)
    check("non-uniform central differences are exact for a quadratic (qddot)",
          float((rates.qddot[inner] - 2 * a2).abs().max()) < 1e-12)
    check("endpoints are flagged invalid rather than silently biased",
          not bool(rates.valid[0]) and not bool(rates.valid[-1]) and int(rates.valid.sum()) == 5)

    # End-to-end identification from a sampled trajectory at the real frame rate.
    cfg = PhysicsConfig(use_wrench_dynamics=True)
    m_true, b_true = 2.5, 0.8
    _, q_s, qdot_s, qddot_s = trajectory(200)
    u = m_true * qddot_s + b_true * qdot_s
    fd = central_difference_rates(q_s, FRAME_DT)
    keep = fd.valid
    phi, _ = build_regressor(fd.qddot[keep], fd.qdot[keep], cfg, "prismatic")
    lower, upper = parameter_bounds(cfg, "prismatic")
    res = constrained_wls(phi, u[keep], lower, upper)
    check("identification from finite differences at the real dt is within 1%",
          abs(float(res.params[0]) - m_true) / m_true < 0.01
          and abs(float(res.params[1]) - b_true) / b_true < 0.01,
          f"m={float(res.params[0]):.5f} b={float(res.params[1]):.5f}")

    # The same data with unit frame spacing assumed: parameters off by dt powers.
    wrong = central_difference_rates(q_s, 1.0)
    phi_w, _ = build_regressor(wrong.qddot[keep], wrong.qdot[keep], cfg, "prismatic")
    res_w = constrained_wls(phi_w, u[keep], lower, upper)
    check("assuming unit frame spacing scales m by 1/dt^2 (why rule 0.11 exists)",
          abs(float(res_w.params[0]) / float(res.params[0]) - 1.0 / FRAME_DT ** 2) < 1.0,
          f"ratio={float(res_w.params[0]) / float(res.params[0]):.2f} expected={1.0 / FRAME_DT ** 2:.2f}")

    for bad in (0.0, -FRAME_DT, float("nan")):
        try:
            central_difference_rates(q_s, bad)
        except ValueError:
            pass
        else:  # pragma: no cover
            raise AssertionError(f"dt={bad} must be rejected")
    check("non-positive / non-finite frame spacing is rejected", True)


# ---------------------------------------------------------------------------
# numerical hygiene
# ---------------------------------------------------------------------------
def test_dtype_device_and_input_validation():
    cfg = PhysicsConfig()
    _, _, qdot, qddot = trajectory(30)
    u = 2.5 * qddot + 0.8 * qdot

    phi32, _ = build_regressor(qddot.float(), qdot.float(), cfg, "prismatic")
    res32 = constrained_wls(phi32, u.float(), *parameter_bounds(cfg, "prismatic", dtype=torch.float32))
    check("float32 input returns float32 parameters", res32.params.dtype == torch.float32)
    check("the float32 solve still lands on the right answer (solved in float64)",
          abs(float(res32.params[0]) - 2.5) < 1e-4 and abs(float(res32.params[1]) - 0.8) < 1e-4,
          f"{res32.params.tolist()}")
    phi64, _ = build_regressor(qddot, qdot, cfg, "prismatic")
    check("float64 input stays float64", constrained_wls(phi64, u).params.dtype == torch.float64)
    check("the wrench projection preserves dtype",
          prismatic_generalized_force(axis().float(), torch.ones(3)).dtype == torch.float32)

    for bad_phi, bad_u, label in (
        (phi64.clone().index_put_((torch.tensor([0]), torch.tensor([0])), torch.tensor(float("nan"), dtype=DT)),
         u, "NaN regressor"),
        (phi64, u.clone().index_put_((torch.tensor([3]),), torch.tensor(float("inf"), dtype=DT)), "Inf target"),
    ):
        try:
            constrained_wls(bad_phi, bad_u)
        except ValueError:
            pass
        else:  # pragma: no cover
            raise AssertionError(f"{label} must be rejected")
    check("non-finite regressor/target inputs are rejected", True)

    try:
        constrained_wls(phi64, u, torch.tensor([1.0, 1.0], dtype=DT), torch.tensor([0.0, 0.0], dtype=DT))
    except ValueError as exc:
        check("an empty box is rejected", "empty box" in str(exc))
    else:  # pragma: no cover
        raise AssertionError("inverted bounds must raise")

    try:
        constrained_wls(phi64, u, weights=-torch.ones(30, dtype=DT))
    except ValueError as exc:
        check("negative weights are rejected", "non-negative" in str(exc))
    else:  # pragma: no cover
        raise AssertionError("negative weights must raise")

    zero_w = torch.ones(30, dtype=DT)
    zero_w[:10] = 0.0
    res_zw = constrained_wls(phi64, u, weights=zero_w)
    check("zero-weight rows are dropped without breaking the solve",
          float((res_zw.params - torch.tensor([2.5, 0.8], dtype=DT)).abs().max()) < 1e-8)

    rls = RecursiveLeastSquares(2)
    rls.update(phi64[0], float(u[0]), weight=0.0)
    check("a zero-weight RLS update changes nothing but the sample count",
          rls.n_samples == 1 and float(rls.params.abs().max()) == 0.0)
    try:
        rls.update(torch.zeros(3, dtype=DT), 1.0)
    except ValueError as exc:
        check("a wrong-width RLS row is rejected", "3" in str(exc))
    else:  # pragma: no cover
        raise AssertionError("wrong-width row must raise")
    try:
        unit_axis(torch.zeros(3, dtype=DT))
    except ValueError as exc:
        check("a degenerate joint axis is rejected", "degenerate" in str(exc))
    else:  # pragma: no cover
        raise AssertionError("zero axis must raise")


def test_no_optimizer_or_autograd_in_the_physics_package():
    """Spec 0.4/0.5 and section 36: no Adam, no backward, no autograd here."""
    package = Path(physics.__file__).parent
    sources = sorted(package.glob("*.py"))
    check("the package has exactly the four section-23 modules",
          [p.name for p in sources] == ["__init__.py", "rls.py", "viscous_models.py", "wrench.py"],
          str([p.name for p in sources]))
    # Audit the parsed code, not the prose: docstrings are allowed to *mention*
    # autograd, the executable statements are not allowed to use it.
    forbidden = {"backward", "autograd", "requires_grad", "retain_grad", "grad",
                 "optim", "Adam", "SGD", "LBFGS"}
    for path in sources:
        text = path.read_text(encoding="utf-8")
        tree = ast.parse(text)
        used: set[str] = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Attribute):
                used.add(node.attr)
            elif isinstance(node, ast.Name):
                used.add(node.id)
            elif isinstance(node, ast.Import):
                used.update(alias.name.split(".")[-1] for alias in node.names)
            elif isinstance(node, ast.ImportFrom):
                used.update((node.module or "").split("."))
                used.update(alias.name for alias in node.names)
        offenders = sorted(used & forbidden)
        check(f"{path.name} uses no optimizer/autograd symbol", not offenders, str(offenders))
        for stub in ("\n    pass\n", "TODO", "FIXME", "NotImplementedError"):
            check(f"{path.name} contains no {stub.strip()!r} placeholder", stub not in text)
    imported: set[str] = set()
    for path in sources:
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            if isinstance(node, ast.Import):
                imported.update(alias.name.split(".")[0] for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
                imported.update([node.module.split(".")[0]])
    check("the package imports nothing but torch and the stdlib",
          imported <= {"torch", "math", "dataclasses", "typing", "__future__"}, str(sorted(imported)))


def main() -> None:
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for fn in tests:
        print(f"\n{fn.__name__}")
        fn()
    print(f"\n{len(PASSED)} checks passed across {len(tests)} tests")


if __name__ == "__main__":
    main()
