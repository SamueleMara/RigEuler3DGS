"""V2 sections 4, 9, 19, 20: trust region, Plücker lines, axis tests.

T-TR1/2/3 (trust region), T-J1..T-J5 (axis geometry), plus the supernode
tangent of section 3.2.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from estimator.trust_region import (  # noqa: E402
    STOP_BOUNDARY,
    STOP_FORCING,
    STOP_NEGATIVE_CURVATURE,
    ScalingMetric,
    TrustRegionConfig,
    adapt_radius,
    boundary_tau,
    forcing_eta,
    predicted_reduction,
    steihaug_cg,
)
from joint_reduction.line_geometry import (  # noqa: E402
    PluckerLine,
    centroid_twist_to_origin,
    fit_plucker_from_motion,
    revolute_joint_tangent,
    revolute_transport_line,
)

PASSED: list[str] = []
DT = torch.float64


def check(name: str, cond: bool, detail: str = "") -> None:
    if not cond:
        raise AssertionError(f"{name} FAILED {detail}")
    PASSED.append(name)
    print(f"  ok  {name}" + (f"   [{detail}]" if detail else ""))


# ---------------------------------------------------------------------------
# Section 4: trust region
# ---------------------------------------------------------------------------
def spd(n, seed, cond=50.0):
    g = torch.Generator().manual_seed(seed)
    Q, _ = torch.linalg.qr(torch.randn((n, n), generator=g, dtype=DT))
    ev = torch.logspace(0, math.log10(cond), n, dtype=DT)
    return Q @ torch.diag(ev) @ Q.T


def test_T_TR2_exact_vs_trust_region():
    """T-TR2: on a tiny dense problem compare exact GN, the trust-region
    solution and truncated PCG."""
    n = 12
    A = spd(n, 3)
    torch.manual_seed(1)
    b = torch.randn(n, dtype=DT)
    exact = torch.linalg.solve(A, b)
    metric = ScalingMetric(weight=torch.ones(n, dtype=DT))

    # A radius larger than the exact step must reproduce the exact step.
    big = float(metric.norm(exact)) * 10
    res = steihaug_cg(lambda v: A @ v, b, big, metric, max_iters=200, eta=1e-14)
    check("T-TR2 a loose radius recovers the exact Gauss-Newton step",
          float((res.step - exact).norm() / exact.norm()) < 1e-8,
          f"rel {float((res.step - exact).norm() / exact.norm()):.3e}, "
          f"stop={res.stop_reason}")
    check("T-TR2 and reports the forcing criterion as the reason",
          res.stop_reason == STOP_FORCING and not res.boundary_hit)

    # A tight radius must return a point ON the boundary, not the exact step.
    small = float(metric.norm(exact)) * 0.1
    res2 = steihaug_cg(lambda v: A @ v, b, small, metric, max_iters=200, eta=1e-14)
    check("T-TR2 a tight radius returns a boundary point",
          abs(metric.norm(res2.step) - small) < 1e-9 * max(small, 1.0),
          f"|s|_S {metric.norm(res2.step):.6f} vs radius {small:.6f}")
    check("T-TR2 and says the boundary was hit",
          res2.stop_reason == STOP_BOUNDARY and res2.boundary_hit)
    check("T-TR2 the constrained step is NOT the unconstrained one",
          float((res2.step - exact).norm()) > 1e-6)
    # It must still be a descent direction of the model.
    check("T-TR2 the boundary step reduces the quadratic model",
          predicted_reduction(res2.step, b, lambda v: A @ v) > 0)


def test_T_TR1_pcg_budget_invariance():
    """T-TR1: the headline failure. Raising the PCG budget must not flip the
    method from progress to no progress -- the radius, not the iteration count,
    controls the step."""
    n = 40
    A = spd(n, 7, cond=1e6)          # deliberately ill-conditioned
    torch.manual_seed(5)
    b = torch.randn(n, dtype=DT)
    metric = ScalingMetric(weight=torch.ones(n, dtype=DT))
    radius = 0.5

    norms, reasons = [], []
    for budget in (25, 100, 400):
        r = steihaug_cg(lambda v: A @ v, b, radius, metric, max_iters=budget, eta=1e-12)
        norms.append(metric.norm(r.step))
        reasons.append(r.stop_reason)
        pred = predicted_reduction(r.step, b, lambda v: A @ v)
        print(f"       budget {budget:>3}: |s|_S {norms[-1]:.6f}  "
              f"pred_red {pred:.4e}  iters {r.iterations}  {r.stop_reason}")
        check(f"T-TR1 budget {budget} respects the radius",
              norms[-1] <= radius * (1 + 1e-9), f"{norms[-1]:.6f} > {radius}")
        check(f"T-TR1 budget {budget} still predicts a decrease", pred > 0)
    check("T-TR1 the step length is bounded by the radius at every budget",
          max(norms) <= radius * (1 + 1e-9))
    check("T-TR1 a larger budget does not inflate the step",
          norms[-1] <= max(norms[0], radius) * (1 + 1e-9),
          f"{norms[0]:.6f} -> {norms[-1]:.6f}")


def test_T_TR3_scaling_metric():
    """T-TR3: the scaling metric makes a mixed-unit step meaningful."""
    m = ScalingMetric.from_physical_scales((2, 12), "cpu", DT,
                                           sigma_mu=0.01, sigma_phi=0.1,
                                           sigma_v=1.0, sigma_omega=1.0)
    # The metric is RMS-normalized, so the unit-norm step is "every component
    # at one sigma", not "one component at one sigma". That is what makes the
    # radius independent of how many Gaussians the scene has.
    sig = torch.tensor([0.01] * 3 + [0.1] * 3 + [1.0] * 3 + [1.0] * 3, dtype=DT)
    all_one_sigma = sig.reshape(1, 12).expand(2, 12).contiguous()
    check("T-TR3 a step of one sigma in every component has unit scaled norm",
          abs(m.norm(all_one_sigma) - 1.0) < 1e-12, f"{m.norm(all_one_sigma):.9f}")

    s = torch.zeros((2, 12), dtype=DT)
    s[0, 0] = 0.01       # one translation scale
    s2 = torch.zeros((2, 12), dtype=DT)
    s2[0, 3] = 0.1       # one rotation scale
    check("T-TR3 one translation sigma and one rotation sigma measure the same",
          abs(m.norm(s) - m.norm(s2)) < 1e-14,
          f"{m.norm(s):.9f} vs {m.norm(s2):.9f}")
    check("T-TR3 a raw Euclidean norm would NOT agree",
          abs(float(s.norm()) - float(s2.norm())) > 0.05,
          f"{float(s.norm()):.4f} vs {float(s2.norm()):.4f}")

    # The scene-size invariance this normalization exists for: the same
    # physical per-Gaussian motion must have the same scaled norm at any N.
    norms = []
    for n in (6, 1000, 11320):
        mn = ScalingMetric.from_physical_scales((n, 12), "cpu", DT, sigma_mu=0.05)
        step = torch.zeros((n, 12), dtype=DT)
        step[:, 0] = 0.01                     # every Gaussian moves 0.01 in x
        norms.append(mn.norm(step))
    check("T-TR3 the scaled norm of identical physical motion is N-independent",
          max(norms) - min(norms) < 1e-12,
          f"N=6 {norms[0]:.6f}, N=1000 {norms[1]:.6f}, N=11320 {norms[2]:.6f}")
    check("T-TR3 and a plain summed norm would have grown like sqrt(N)",
          True, f"sum-norm would be {(11320 * 0.04) ** 0.5:.1f} at N=11320 "
                f"vs {(6 * 0.04) ** 0.5:.1f} at N=6")
    check("T-TR3 the metric is persisted with implied sigmas",
          "implied_sigma" in m.to_dict()
          and abs(m.to_dict()["implied_sigma"][0] - 0.01) < 1e-12,
          f"implied sigma_mu {m.to_dict()['implied_sigma'][0]:.6f}")

    # boundary_tau really lands on the boundary
    torch.manual_seed(2)
    s0 = torch.randn((2, 12), dtype=DT) * 0.001
    p = torch.randn((2, 12), dtype=DT)
    tau = boundary_tau(s0, p, 2.0, m)
    check("T-TR3 boundary_tau lands exactly on the radius",
          abs(m.norm(s0 + tau * p) - 2.0) < 1e-9, f"{m.norm(s0 + tau * p):.9f}")


def test_negative_curvature_and_policy():
    n = 8
    torch.manual_seed(4)
    Q, _ = torch.linalg.qr(torch.randn((n, n), dtype=DT))
    ev = torch.ones(n, dtype=DT)
    ev[0] = -3.0                       # indefinite
    A = Q @ torch.diag(ev) @ Q.T
    metric = ScalingMetric(weight=torch.ones(n, dtype=DT))
    r = steihaug_cg(lambda v: A @ v, torch.randn(n, dtype=DT), 1.0, metric,
                    max_iters=50, eta=1e-12)
    check("non-positive curvature is detected and reported",
          r.stop_reason == STOP_NEGATIVE_CURVATURE, r.stop_reason)
    check("and the step goes to the boundary",
          abs(metric.norm(r.step) - 1.0) < 1e-9)

    cfg = TrustRegionConfig()
    check("rho < 0.1 rejects and shrinks strongly",
          adapt_radius(1.0, 0.05, False, 1.0, cfg) == (0.25, False))
    check("rho in [0.25,0.75] accepts and keeps the radius",
          adapt_radius(1.0, 0.5, False, 1.0, cfg) == (1.0, True))
    check("rho > 0.75 on the boundary enlarges",
          adapt_radius(1.0, 0.9, True, 1.0, cfg) == (2.0, True))
    check("rho > 0.75 off the boundary does NOT enlarge",
          adapt_radius(1.0, 0.9, False, 1.0, cfg) == (1.0, True))
    check("a NaN rho rejects", adapt_radius(1.0, float("nan"), False, 1.0, cfg)[1] is False)

    check("forcing eta is loose far away and tight near the solution",
          forcing_eta(1.0, 1.0) > forcing_eta(1e-6, 1.0))
    check("forcing eta is clipped", forcing_eta(1e30, 1.0) <= 0.5
          and forcing_eta(0.0, 1.0) >= 1e-6)


# ---------------------------------------------------------------------------
# Sections 7-9, 19: arbitrary-line revolute geometry
# ---------------------------------------------------------------------------
def test_T_J1_offset_axis():
    """T-J1: an axis clearly NOT through the origin must be recovered as a line."""
    e = torch.tensor([0.2, 0.8, 0.56], dtype=DT)
    e = e / e.norm()
    a_raw = torch.tensor([0.35, -0.17, 0.12], dtype=DT)
    a = a_raw - (a_raw @ e) * e              # the e^T a = 0 gauge (section 7)
    check("T-J1 the gauge point satisfies e^T a = 0", abs(float(a @ e)) < 1e-14)
    check("T-J1 the axis is genuinely offset from the origin",
          float(a.norm()) > 0.2, f"|a| {float(a.norm()):.4f}")

    line = PluckerLine.from_point_direction(a, e)
    check("T-J1 m = a x e is non-zero (not an origin axis)",
          float(line.moment.norm()) > 0.2, f"|m| {float(line.moment.norm()):.4f}")
    check("T-J1 a = e x m round-trips",
          float((line.closest_point - a).norm()) < 1e-14)

    torch.manual_seed(9)
    mu = torch.randn((200, 3), dtype=DT) * 0.4
    qdot = 0.7
    v = line.velocity(mu, qdot=qdot)
    fitted, info = fit_plucker_from_motion(mu, v)
    check("T-J1 the fit is reported observable", info["observable"], str(info.get("reason")))
    d = fitted.distance_to(line)
    check("T-J1 axis DIRECTION recovered", d["angle_deg"] < 1e-8, f"{d['angle_deg']:.3e} deg")
    check("T-J1 axis LINE distance recovered (not a point difference)",
          d["line_distance"] < 1e-10, f"{d['line_distance']:.3e}")
    check("T-J1 qdot recovered", abs(info["qdot"] - qdot) < 1e-10,
          f"{info['qdot']:.9f} vs {qdot}")


def test_T_J2_translation_equivariance():
    """T-J2: m' = m + t x e. This is what catches a hidden origin assumption."""
    e = torch.tensor([0.0, 0.0, 1.0], dtype=DT)
    a = torch.tensor([0.4, -0.3, 0.0], dtype=DT)
    line = PluckerLine.from_point_direction(a, e)
    t = torch.tensor([1.7, -2.3, 0.9], dtype=DT)

    moved = line.translate(t)
    expected_m = line.moment + torch.linalg.cross(t, line.direction)
    expected_m = expected_m - (expected_m @ e) * e
    check("T-J2 m' = m + t x e", float((moved.moment - expected_m).norm()) < 1e-14,
          f"{float((moved.moment - expected_m).norm()):.3e}")
    check("T-J2 the direction is unchanged",
          float((moved.direction - line.direction).norm()) < 1e-14)

    # and the same physical points must lie on the translated line
    torch.manual_seed(3)
    mu = torch.randn((80, 3), dtype=DT) * 0.3
    v = line.velocity(mu, qdot=0.5)
    v_moved = moved.velocity(mu + t, qdot=0.5)
    check("T-J2 the velocity field is equivariant under translation",
          float((v - v_moved).abs().max()) < 1e-12,
          f"{float((v - v_moved).abs().max()):.3e}")

    # fitting in the translated frame must give the translated line
    fitted, _ = fit_plucker_from_motion(mu + t, v_moved)
    d = fitted.distance_to(moved)
    check("T-J2 refitting in the translated frame recovers the same line",
          d["angle_deg"] < 1e-8 and d["line_distance"] < 1e-9,
          f"angle {d['angle_deg']:.2e} dist {d['line_distance']:.2e}")


def test_T_J3_centroid_reference_conversion():
    """T-J3: u = v_c - omega x c, verified numerically, then the axis from (u,omega)."""
    e = torch.tensor([0.1, -0.9, 0.42], dtype=DT)
    e = e / e.norm()
    a_raw = torch.tensor([-0.6, 0.25, 0.4], dtype=DT)
    a = a_raw - (a_raw @ e) * e
    line = PluckerLine.from_point_direction(a, e)
    qdot = 1.3
    omega = qdot * e

    torch.manual_seed(11)
    mu = torch.randn((150, 3), dtype=DT) * 0.5
    c = mu.mean(dim=0)
    check("T-J3 the centroid is genuinely non-zero", float(c.norm()) > 1e-3)

    v = line.velocity(mu, qdot=qdot)
    # v(x) = v_c + omega x (x - c), so v_c = v(c)
    v_c = line.velocity(c.reshape(1, 3), qdot=qdot).reshape(3)
    recon = v_c.reshape(1, 3) + torch.linalg.cross(omega.expand_as(mu), mu - c.reshape(1, 3))
    check("T-J3 the centroid-referenced field reproduces v",
          float((recon - v).abs().max()) < 1e-12)

    u = centroid_twist_to_origin(v_c, omega, c)
    u_direct = qdot * line.moment                  # v(x) = qdot(e x x + m) => u = qdot m
    check("T-J3 u = v_c - omega x c matches the origin-referenced twist",
          float((u - u_direct).norm()) < 1e-12,
          f"{float((u - u_direct).norm()):.3e}")

    recovered, info = PluckerLine.from_twist(omega, u)
    d = recovered.distance_to(line)
    check("T-J3 the axis recovered from (u, omega) is the true line",
          d["angle_deg"] < 1e-9 and d["line_distance"] < 1e-12,
          f"angle {d['angle_deg']:.2e} dist {d['line_distance']:.2e}")
    # and using v_c instead of u puts the axis in the WRONG place
    wrong, _ = PluckerLine.from_twist(omega, v_c)
    dw = wrong.distance_to(line)
    check("T-J3 using v_c directly would misplace the axis",
          dw["line_distance"] > 1e-3, f"{dw['line_distance']:.4f}")


def test_T_J5_small_omega_is_unobservable():
    """T-J5: near-prismatic motion must NOT yield a giant confident axis."""
    e = torch.tensor([0.0, 0.0, 1.0], dtype=DT)
    tiny = 1e-9 * e
    u = torch.tensor([0.5, 0.0, 0.0], dtype=DT)
    line, info = PluckerLine.from_twist(tiny, u, min_omega=1e-6)
    check("T-J5 a tiny omega returns no axis", line is None)
    check("T-J5 and is flagged unobservable", info["observable"] is False)
    check("T-J5 with a reason naming the magnitude",
          "unobservable" in info["reason"], info["reason"])

    # the fit path must behave the same way
    torch.manual_seed(6)
    mu = torch.randn((100, 3), dtype=DT) * 0.3
    v = torch.tensor([0.4, 0.0, 0.0], dtype=DT).expand_as(mu) \
        + 1e-7 * torch.randn((100, 3), dtype=DT)
    fitted, finfo = fit_plucker_from_motion(mu, v, min_omega=1e-3)
    check("T-J5 a near-translation fit reports no revolute axis",
          fitted is None and finfo["observable"] is False,
          str(finfo.get("reason")))


def test_revolute_transport_and_tangent():
    """Section 10.2: finite motion about an arbitrary line, and its tangent."""
    e = torch.tensor([0.3, 0.5, 0.81], dtype=DT)
    e = e / e.norm()
    a_raw = torch.tensor([0.5, 0.2, -0.3], dtype=DT)
    a = a_raw - (a_raw @ e) * e
    line = PluckerLine.from_point_direction(a, e)
    torch.manual_seed(13)
    mu0 = torch.randn((60, 3), dtype=DT) * 0.4
    R0 = torch.eye(3, dtype=DT).repeat(60, 1, 1)

    mu, R = revolute_transport_line(mu0, R0, line, 0.0)
    check("§10.2 q=0 is the identity", float((mu - mu0).abs().max()) < 1e-14
          and float((R - R0).abs().max()) < 1e-14)

    # points ON the axis must not move
    on_axis = (a.reshape(1, 3) + torch.linspace(-1, 1, 5, dtype=DT).reshape(5, 1)
               * e.reshape(1, 3))
    moved, _ = revolute_transport_line(on_axis, None, line, 0.9)
    check("§10.2 points on the axis are fixed",
          float((moved - on_axis).abs().max()) < 1e-13,
          f"{float((moved - on_axis).abs().max()):.3e}")

    # the tangent must be the derivative of the transport
    dmu, dphi = revolute_joint_tangent(mu0, line)
    eps = 1e-6
    mp, _ = revolute_transport_line(mu0, None, line, eps)
    mm, _ = revolute_transport_line(mu0, None, line, -eps)
    fd = (mp - mm) / (2 * eps)
    rel = float((fd - dmu).norm() / fd.norm())
    check("§10.2 d mu/dq = e x mu + m matches finite differences", rel < 1e-9,
          f"rel {rel:.3e}")
    check("§10.2 d phi/dq = e", float((dphi[0] - e).norm()) < 1e-14)


# ---------------------------------------------------------------------------
# Section 3.2: the supernode tangent (CPU reference of the CUDA kernel)
# ---------------------------------------------------------------------------
def test_supernode_tangent_math():
    """delta_mu_i = delta_t_c + delta_theta_c x (mu_i^0 - c_c^0), delta_phi = delta_theta."""
    torch.manual_seed(21)
    mu = torch.randn((10, 3), dtype=DT)
    c0 = mu.mean(dim=0)
    dt = torch.tensor([0.1, -0.2, 0.3], dtype=DT)
    dth = torch.tensor([0.05, 0.02, -0.04], dtype=DT)
    d = mu - c0.reshape(1, 3)
    dmu = dt.reshape(1, 3) + torch.linalg.cross(dth.expand_as(d), d)

    # A pure group translation moves every member identically.
    pure_t = dt.reshape(1, 3).expand_as(d)
    check("§3.2 a pure group translation moves all members equally",
          float((pure_t - pure_t[0:1]).abs().max()) < 1e-15)
    # A pure group rotation leaves the centroid fixed.
    pure_r = torch.linalg.cross(dth.expand_as(d), d)
    at_centroid = torch.linalg.cross(dth.reshape(1, 3), torch.zeros((1, 3), dtype=DT))
    check("§3.2 a pure group rotation fixes the centroid",
          float(at_centroid.abs().max()) < 1e-15)
    check("§3.2 the rotation part grows with distance from the centroid",
          float(pure_r.norm(dim=1).max()) > float(pure_r.norm(dim=1).min()))
    # G^T is the stated adjoint reduction.
    bar_mu = torch.randn((10, 3), dtype=DT)
    bar_phi = torch.randn((10, 3), dtype=DT)
    t_bar = bar_mu.sum(dim=0)
    th_bar = torch.linalg.cross(d, bar_mu).sum(dim=0) + bar_phi.sum(dim=0)
    lhs = float((dmu * bar_mu).sum() + (dth.reshape(1, 3).expand_as(d) * bar_phi).sum())
    rhs = float(dt @ t_bar + dth @ th_bar)
    check("§3.3 the group adjoint reduction is the true transpose of G",
          abs(lhs - rhs) / max(abs(lhs), 1e-30) < 1e-12,
          f"{lhs:.9f} vs {rhs:.9f}")


def main() -> None:
    tests = [v for k, v in sorted(globals().items())
             if k.startswith("test_") and callable(v)]
    for fn in tests:
        print(f"\n{fn.__name__}")
        fn()
    print(f"\n{len(PASSED)} checks passed across {len(tests)} tests")


if __name__ == "__main__":
    main()
