"""Sections 39, 41, 47: the Phase-2 rigid-screw joint parameterization.

CODEX_IMPLEMENT_PHASE2_RIGID_SCREW_IEKF.md.

The identity under test is section 50's:

    Phase 2 = Phase 1 RGB-D IEKF (unchanged) + rigid screw state-to-Gaussian map

so what is checked here is the new map and ONLY the new map: its constraints,
its regularity at the prismatic state, its analytic JVP and VJP against finite
differences, and the adjoint identity that lets it compose with the existing
renderer operators.
"""

from __future__ import annotations

import copy
import math
import sys
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from estimator.joint_kinematics import (current_points,  # noqa: E402
                                        full_jvp, full_vjp, joint_jvp,
                                        joint_vjp, part_transform,
                                        transport_gaussians)
from estimator.joint_manifold import (constraint_values,  # noqa: E402
                                      retract_joint_geometry, revolute_axis,
                                      se3_exp, se3_left_jacobian,
                                      tangent_basis, zeta_to_xi)
from estimator.joint_process import (apply_correction, predict,  # noqa: E402
                                     process_matrix, process_noise)
from estimator.joint_state import (JointPart, JointState,  # noqa: E402
                                   initialise_from_twist)
from estimator.manifold import so3_exp, so3_hat, so3_log  # noqa: E402

PASSED: list[str] = []


def check(name, cond, detail=""):
    if not cond:
        raise AssertionError(f"{name} FAILED {detail}")
    PASSED.append(name)
    print(f"  ok  {name}" + (f"   [{detail}]" if detail else ""))


def cloud(n=150, seed=0, scale=1.0):
    g = torch.Generator().manual_seed(seed)
    return torch.randn((n, 3), generator=g, dtype=torch.float64) * scale


def make_part(xi, n=150, seed=0, T=12, qmax=1.1):
    mu0 = cloud(n, seed)
    return mu0, initialise_from_twist(torch.arange(n), mu0,
                                      torch.tensor(xi, dtype=torch.float64),
                                      torch.linspace(0.0, qmax, T), None)


# ===========================================================================
def test_J1_exact_prismatic():
    """J1: at omega = 0, R(q) = I and t(q) = q v, with no angular epsilon."""
    mu0, p = make_part([0.4, -0.3, 0.2, 0.0, 0.0, 0.0])
    xi = zeta_to_xi(p.zeta, p.length)
    worst_R, worst_t = 0.0, 0.0
    for f in range(p.num_frames):
        q = float(p.q[f])
        R, t = part_transform(q, p.zeta, p.length)
        worst_R = max(worst_R, float((R - torch.eye(3, dtype=torch.float64))
                                     .abs().max()))
        worst_t = max(worst_t, float((t - q * xi[:3]).abs().max()))
    print(f"       |R - I| {worst_R:.3e}   |t - q v| {worst_t:.3e}")
    check("J1 R(q) = I exactly at the prismatic state", worst_R == 0.0,
          "bitwise")
    check("J1 t(q) = q v", worst_t < 1e-15, f"{worst_t:.2e}")


def test_J2_offset_revolute():
    """J2: the screw exponential equals an explicit rotation about the line."""
    axis = torch.tensor([0.2, -0.5, 0.84], dtype=torch.float64)
    axis = axis / axis.norm()
    a = torch.tensor([1.3, -0.7, 0.4], dtype=torch.float64)   # not the COM
    # spatial twist of a pure rotation about the line through a: v = -omega x a
    xi = torch.cat([-torch.linalg.cross(axis, a), axis])
    mu0, p = make_part(xi.tolist(), seed=3)
    worst = 0.0
    for f in range(p.num_frames):
        q = float(p.q[f])
        R, t = part_transform(q, p.zeta, p.length)
        theta = q * float(zeta_to_xi(p.zeta, p.length)[3:].norm())
        Rex = so3_exp((theta * axis).reshape(1, 3))[0].double()
        pred = (mu0 - a) @ Rex.T + a
        got = mu0 @ R.T + t
        worst = max(worst, float((pred - got).abs().max()))
    print(f"       worst |screw exp - explicit rotation about the line| "
          f"{worst:.3e}")
    check("J2 the screw matches an explicit off-COM rotation", worst < 1e-12,
          f"{worst:.2e}")


def test_J3_J4_J5_gauge_and_constraints():
    """J3 axis gauge, J4 zero pitch, J5 normalisation, after random updates."""
    mu0, p = make_part([0.3, 0.1, -0.2, 0.15, -0.4, 0.25], seed=5)
    g = torch.Generator().manual_seed(11)
    worst_c = 0.0
    for _ in range(40):
        d = torch.randn(4, generator=g, dtype=torch.float64) * 0.25
        p.zeta, _G = retract_joint_geometry(p.zeta, d)
        worst_c = max(worst_c, float(constraint_values(p.zeta).abs().max()))
    u, w = p.zeta[:3], p.zeta[3:]
    print(f"       after 40 random retractions: |u.omega| "
          f"{abs(float(u @ w)):.3e}, ||zeta|| - 1 "
          f"{abs(float(p.zeta.norm()) - 1):.3e}")
    check("J4 zero pitch holds at numerical precision",
          abs(float(u @ w)) < 1e-14, f"{abs(float(u @ w)):.2e}")
    check("J5 the gauge |u|^2 + |omega|^2 = 1 holds",
          abs(float(p.zeta.norm()) - 1.0) < 1e-14)
    ax = revolute_axis(p.zeta, p.length, p.centroid)
    print(f"       recovered axis gauge e.(a - c0) = {ax['gauge_residual']:.3e},"
          f" pitch {ax['pitch']:.3e}")
    check("J3 the recovered axis satisfies e.(a - c0) = 0",
          abs(ax["gauge_residual"]) < 1e-12,
          f"{abs(ax['gauge_residual']):.2e}")
    check("J3 and the screw has zero pitch", abs(ax["pitch"]) < 1e-12)


def test_J6_prismatic_regularity():
    """J6: perturb all four geometry directions from exact P. No NaN, rank 4."""
    mu0, p = make_part([1.0, 0.0, 0.0, 0.0, 0.0, 0.0], seed=7)
    B = tangent_basis(p.zeta)
    print(f"       at P: basis rank "
          f"{int(torch.linalg.matrix_rank(B))}, J_c B "
          f"{float((torch.linalg.matrix_rank(B))):.0f} columns")
    ok = True
    for k in range(4):
        for h in (1e-6, -1e-6, 1e-3):
            d = torch.zeros(4, dtype=torch.float64)
            d[k] = h
            z, G = retract_joint_geometry(p.zeta, d)
            ok &= bool(torch.isfinite(z).all() and torch.isfinite(G).all())
            ok &= float(constraint_values(z).abs().max()) < 1e-13
    check("J6 the prismatic state has four regular tangent directions",
          int(torch.linalg.matrix_rank(B)) == 4)
    check("J6 and perturbing each stays finite and on the manifold", ok)


def test_J7_J8_jvp_against_finite_differences():
    """J7 (q) and J8 (geometry): analytic JVP vs central differences."""
    for name, xi, seed in (("revolute", [0.3, -0.2, 0.1, 0.1, 0.4, -0.2], 1),
                           ("prismatic", [0.5, 0.2, -0.1, 0.0, 0.0, 0.0], 2)):
        mu0, p = make_part(xi, seed=seed)
        frame = 7
        mu, Rg, _R, _t = current_points(p, frame, mu0, None)

        def geometry(d):
            q2 = p.q.clone()
            q2[frame] = q2[frame] + d[0]
            z2, _ = retract_joint_geometry(p.zeta, d[2:6])
            p2 = copy.deepcopy(p)
            p2.q, p2.zeta = q2, z2
            m, _rg, _r, _tt = current_points(p2, frame, mu0, None)
            return m

        h = 1e-6
        worst = 0.0
        per = {}
        for k in (0, 2, 3, 4, 5):
            d = torch.zeros(6, dtype=torch.float64)
            d[k] = h
            fd = (geometry(d) - geometry(-d)) / (2 * h)
            e = torch.zeros(6, dtype=torch.float64)
            e[k] = 1.0
            an = joint_jvp(p, frame, e, mu_now=mu)[:, :3]
            rel = float((fd - an).abs().max()
                        / max(float(an.abs().max()), 1e-30))
            per[k] = rel
            worst = max(worst, rel)
        print(f"       {name:10s} dq {per[0]:.2e}   geometry "
              + " ".join(f"{per[k]:.2e}" for k in (2, 3, 4, 5)))
        check(f"J7/J8 {name}: the analytic JVP matches finite differences",
              worst < 1e-6, f"{worst:.2e}")


def test_J9_kinematic_adjoint():
    """J9: <J p, z> == <p, J^T z> for random p and z."""
    mu0, p = make_part([0.3, -0.2, 0.1, 0.1, 0.4, -0.2], seed=9)
    st = JointState([p], torch.arange(p.num_frames, dtype=torch.float64))
    g = torch.Generator().manual_seed(21)
    worst = 0.0
    for frame in (0, 4, 11):
        for _ in range(5):
            v = torch.randn(st.dim, generator=g, dtype=torch.float64)
            z = torch.randn((mu0.shape[0], 6), generator=g,
                            dtype=torch.float64)
            lhs = float((full_jvp(st, frame, v, mu0) * z).sum())
            rhs = float((v * full_vjp(st, frame, z, mu0)).sum())
            worst = max(worst, abs(lhs - rhs)
                        / max(abs(lhs), abs(rhs), 1e-30))
    print(f"       worst relative adjoint mismatch over 15 pairs: {worst:.3e}")
    check("J9 the kinematic VJP is the exact adjoint of the JVP",
          worst < 1e-12, f"{worst:.2e}")


def test_J10_orientation_and_covariance_transport():
    """J10: Sigma_i(t) = R Sigma_i^0 R^T exactly, and R_i(t) = R R_i^0."""
    mu0, p = make_part([0.2, -0.1, 0.4, 0.3, 0.2, -0.1], seed=13)
    g = torch.Generator().manual_seed(31)
    R0 = so3_exp(torch.randn((mu0.shape[0], 3), generator=g,
                             dtype=torch.float64) * 0.5).double()
    S0 = torch.einsum("nij,njk->nik", R0, R0.transpose(1, 2))
    R, t = part_transform(float(p.q[6]), p.zeta, p.length)
    mu, Rg = transport_gaussians(mu0, R0, R, t)
    Sg = torch.einsum("ij,njk,lk->nil", R, S0, R)
    direct = torch.einsum("nij,njk->nik", Rg, Rg.transpose(1, 2))
    print(f"       |R_i(t) - R R_i^0| "
          f"{float((Rg - R @ R0).abs().max()):.3e}   "
          f"|Sigma_i(t) - R Sigma_i^0 R^T| "
          f"{float((direct - Sg).abs().max()):.3e}")
    check("J10 orientation transports as R R_i^0",
          float((Rg - R @ R0).abs().max()) < 1e-14)
    check("J10 covariance transports as R Sigma_i^0 R^T",
          float((direct - Sg).abs().max()) < 1e-13)
    # rigidity is structural: pairwise distances are preserved exactly
    d0 = (mu0[:50, None, :] - mu0[None, :50, :]).norm(dim=-1)
    dt = (mu[:50, None, :] - mu[None, :50, :]).norm(dim=-1)
    check("J10 and the map is exactly rigid, so no rigidity penalty is needed",
          float((d0 - dt).abs().max()) < 1e-12,
          f"{float((d0 - dt).abs().max()):.2e}")


def test_J11_tangent_covariance_reset():
    """J11: the reset Jacobian against finite differences of the local map.

    `G = d delta_new / d delta_old`: perturbing in the OLD basis and reading
    the result back in the NEW one.
    """
    mu0, p = make_part([0.3, 0.1, -0.2, 0.15, -0.4, 0.25], seed=17)
    # The reset Jacobian is the FIRST-ORDER transport, so the right check is
    # that it agrees with the retraction's exact derivative to second order --
    # not that it matches at some arbitrary finite step. Asserting a fixed
    # tolerance at |delta| = 0.1 would be testing the size of a second-order
    # term, which is a property of the step, not of the implementation.
    errs = []
    sizes = (0.2, 0.1, 0.05, 0.025)
    for size in sizes:
        d0 = torch.tensor([1.0, -0.6, 0.4, 0.8], dtype=torch.float64)
        d0 = d0 / d0.norm() * size
        z_new, G = retract_joint_geometry(p.zeta, d0)
        B_new = tangent_basis(z_new)
        h = 1e-6
        cols = []
        for k in range(4):
            e = torch.zeros(4, dtype=torch.float64)
            e[k] = h
            zp, _ = retract_joint_geometry(p.zeta, d0 + e)
            zm, _ = retract_joint_geometry(p.zeta, d0 - e)
            cols.append(B_new.T @ ((zp - zm) / (2 * h)))
        errs.append(float((torch.stack(cols, dim=1) - G).abs().max()))
        print(f"       |delta| {size:5.3f}: |G - dr/ddelta| {errs[-1]:.3e}"
              f"   / |delta|^2 = {errs[-1] / size ** 2:.4f}")
    ratios = [errs[i] / errs[i + 1] for i in range(len(errs) - 1)]
    check("J11 the reset Jacobian is the exact first-order transport "
          "(error is O(|delta|^2))",
          all(3.5 < r < 4.5 for r in ratios),
          "halving the step quarters the error: "
          + ", ".join(f"{r:.2f}" for r in ratios))
    _z, G = retract_joint_geometry(p.zeta, torch.zeros(4, dtype=torch.float64))
    check("J11 and it is the identity for a zero step",
          float((G - torch.eye(4, dtype=torch.float64)).abs().max()) < 1e-12)
    check("J11 and orthogonal, so the transported covariance stays PSD",
          float((G @ G.T - torch.eye(4, dtype=torch.float64)).abs().max())
          < 1e-10)


def test_15_qdot_has_no_direct_measurement():
    """Section 15: H_qdot = 0, exactly, and qdot still moves through F."""
    mu0, p = make_part([0.3, -0.2, 0.1, 0.1, 0.4, -0.2], seed=19)
    st = JointState([p], torch.arange(p.num_frames, dtype=torch.float64))
    g = torch.Generator().manual_seed(41)
    e = torch.zeros(st.dim, dtype=torch.float64)
    e[1] = 1.0                                   # pure dqdot
    out = full_jvp(st, 5, e, mu0)
    z = torch.randn((mu0.shape[0], 6), generator=g, dtype=torch.float64)
    adj = full_vjp(st, 5, z, mu0)
    print(f"       |J e_qdot| {float(out.abs().max()):.3e}   "
          f"adjoint qdot component {float(adj[1].abs()):.3e}")
    check("§15 a qdot perturbation produces no Gaussian motion at this frame",
          float(out.abs().max()) == 0.0)
    check("§15 and the adjoint returns exactly zero for qdot",
          float(adj[1].abs()) == 0.0)
    F = process_matrix(st, 0.5)
    check("§15 while the process model couples q to qdot through dt",
          float(F[0, 1]) == 0.5)
    Q = process_noise(st, 0.5)
    check("§15 and the static geometry gets exactly zero process noise",
          float(Q[2:6, 2:6].abs().max()) == 0.0)


def test_16_covariance_transport_preserves_cross_terms():
    """Section 16: the reset is applied to the WHOLE matrix, not block-wise."""
    mu0, p = make_part([0.3, 0.1, -0.2, 0.15, -0.4, 0.25], seed=23)
    st = JointState([p], torch.arange(p.num_frames, dtype=torch.float64))
    g = torch.Generator().manual_seed(51)
    A = torch.randn((6, 6), generator=g, dtype=torch.float64)
    st.covariance = A @ A.T + torch.eye(6, dtype=torch.float64)
    before = st.covariance.clone()
    cross_before = float(before[1, 2:6].abs().sum())
    d = torch.zeros(6, dtype=torch.float64)
    d[2:6] = torch.tensor([0.08, -0.05, 0.02, 0.06], dtype=torch.float64)
    st, G = apply_correction(st, 3, d)
    cross_after = float(st.covariance[1, 2:6].abs().sum())
    print(f"       qdot-geometry cross-covariance: {cross_before:.4f} before, "
          f"{cross_after:.4f} after the reset")
    check("§16 the covariance stays symmetric positive definite",
          bool((torch.linalg.eigvalsh(st.covariance) > 0).all()))
    check("§16 and the qdot-geometry cross terms survive the basis change",
          cross_after > 1e-6 and abs(cross_after - cross_before) > 1e-12,
          "transported, not dropped and not left unchanged")
    check("§16 the transport is exactly G P G^T",
          float((st.covariance - G @ before @ G.T).abs().max()) < 1e-12)


def test_25_reduces_to_the_theory_note_jacobians():
    """Section 25: the unified map must reduce to the original P/R Jacobians."""
    # --- prismatic: d mu / ds = e_P ---------------------------------------
    mu0, p = make_part([0.6, 0.0, 0.8, 0.0, 0.0, 0.0], seed=29)
    xi = zeta_to_xi(p.zeta, p.length)
    e_P = xi[:3] / xi[:3].norm()
    mu, _rg, _R, _t = current_points(p, 5, mu0, None)
    e = torch.zeros(6, dtype=torch.float64)
    e[0] = 1.0
    dmu_dq = joint_jvp(p, 5, e, mu_now=mu)[:, :3]
    # q is normalised; physical travel is s = q |v|, so d mu / ds = dmu_dq/|v|
    dmu_ds = dmu_dq / float(xi[:3].norm())
    print(f"       prismatic: max |d mu/ds - e_P| "
          f"{float((dmu_ds - e_P[None, :]).abs().max()):.3e}")
    check("§25 prismatic reduces to d mu / ds = e_P",
          float((dmu_ds - e_P[None, :]).abs().max()) < 1e-12)
    check("§25 and the prismatic orientation derivative is exactly zero",
          float(joint_jvp(p, 5, e, mu_now=mu)[:, 3:].abs().max()) == 0.0)

    # --- revolute: d mu / dtheta = e_R x (mu - a_R), d phi / dtheta = e_R ---
    axis = torch.tensor([0.0, 0.0, 1.0], dtype=torch.float64)
    a = torch.tensor([0.9, -0.5, 0.0], dtype=torch.float64)
    xi_r = torch.cat([-torch.linalg.cross(axis, a), axis])
    mu0, p = make_part(xi_r.tolist(), seed=31)
    ax = revolute_axis(p.zeta, p.length, p.centroid)
    mu, _rg, _R, _t = current_points(p, 6, mu0, None)
    dmu_dq = joint_jvp(p, 6, e, mu_now=mu)
    om = zeta_to_xi(p.zeta, p.length)[3:]
    scale = float(om.norm())
    dmu_dth = dmu_dq[:, :3] / scale
    dphi_dth = dmu_dq[:, 3:] / scale
    want_mu = torch.linalg.cross(ax["axis"][None, :].expand_as(mu),
                                 mu - ax["axis_point"][None, :])
    print(f"       revolute: max |d mu/dtheta - e_R x (mu - a_R)| "
          f"{float((dmu_dth - want_mu).abs().max()):.3e}   "
          f"|d phi/dtheta - e_R| "
          f"{float((dphi_dth - ax['axis'][None, :]).abs().max()):.3e}")
    check("§25 revolute reduces to d mu / dtheta = e_R x (mu - a_R)",
          float((dmu_dth - want_mu).abs().max()) < 1e-10)
    check("§25 and d phi / dtheta = e_R",
          float((dphi_dth - ax["axis"][None, :]).abs().max()) < 1e-12)


def test_47M_static_gaussians_have_zero_derivative():
    """Acceptance gate M: non-member Gaussians are exactly unaffected."""
    mu0 = cloud(200, 37)
    members = torch.arange(0, 80)
    p = initialise_from_twist(members, mu0,
                              torch.tensor([0.3, -0.2, 0.1, 0.2, 0.1, -0.3]),
                              torch.linspace(0.0, 1.0, 10), None)
    st = JointState([p], torch.arange(10, dtype=torch.float64))
    g = torch.Generator().manual_seed(61)
    v = torch.randn(st.dim, generator=g, dtype=torch.float64)
    out = full_jvp(st, 5, v, mu0)
    static = out[80:]
    print(f"       moving block |J v| {float(out[:80].abs().max()):.3e}, "
          f"static block {float(static.abs().max()):.3e}")
    check("gate M static Gaussians have exactly zero joint derivative",
          float(static.abs().max()) == 0.0)
    from estimator.joint_kinematics import state_to_gaussians
    mu, _ = state_to_gaussians(st, 5, mu0, None)
    check("gate M and their positions are bitwise unchanged",
          bool((mu[80:] == mu0[80:]).all()))


def test_S1_S2_variable_speed_sequences():
    """Sections 41 S1/S2: one constant screw with a VARIABLE q(t).

    This is the structural claim of section 8: the motion magnitude varies
    frame to frame while the geometry does not, so a correct implementation
    reproduces a non-uniform trajectory without any per-frame twist.
    """
    for name, xi in (("prismatic", [0.5, -0.2, 0.3, 0.0, 0.0, 0.0]),
                     ("offset revolute", [0.4, 0.9, 0.0, 0.0, 0.0, 1.0])):
        mu0 = cloud(120, 43)
        tt = torch.linspace(0.0, 1.0, 20, dtype=torch.float64)
        q = 1.3 * tt ** 2 - 0.4 * tt ** 3          # deliberately non-uniform
        p = initialise_from_twist(torch.arange(120), mu0,
                                  torch.tensor(xi, dtype=torch.float64), q,
                                  None)
        # rebuild the sequence from the single screw and compare with a direct
        # per-frame transform of the same screw
        worst = 0.0
        for f in range(p.num_frames):
            R, t = part_transform(float(p.q[f]), p.zeta, p.length)
            direct, _ = se3_exp((float(p.q[f]) * p.xi).reshape(1, 6))
            worst = max(worst, float((R - direct[0]).abs().max()))
        d0 = (mu0[:40, None] - mu0[None, :40]).norm(dim=-1)
        mu_last, _rg, _R, _t = current_points(p, p.num_frames - 1, mu0, None)
        dT = (mu_last[:40, None] - mu_last[None, :40]).norm(dim=-1)
        print(f"       {name:16s} q range {float(q.min()):.3f}..{float(q.max()):.3f}"
              f"   one-screw consistency {worst:.2e}"
              f"   rigidity drift {float((d0 - dT).abs().max()):.2e}")
        check(f"S1/S2 {name}: one constant screw reproduces a variable-speed "
              f"trajectory", worst < 1e-14)
        check(f"S1/S2 {name}: and it stays exactly rigid",
              float((d0 - dT).abs().max()) < 1e-12)


def test_S5_exact_P_without_fake_curvature():
    """Section 41 S5: exact P data must not acquire an angular component."""
    mu0, p = make_part([0.7, 0.1, -0.3, 0.0, 0.0, 0.0], seed=47)
    worst_phi = 0.0
    for f in range(p.num_frames):
        R, _t = part_transform(float(p.q[f]), p.zeta, p.length)
        worst_phi = max(worst_phi,
                        float(so3_log(R.reshape(1, 3, 3)).abs().max()))
    print(f"       max |log R| over the sequence: {worst_phi:.3e}")
    check("S5 an exact prismatic screw produces no rotation at all",
          worst_phi == 0.0, "bitwise zero")
    check("S5 and omega stays exactly zero in the state",
          float(p.zeta[3:].abs().max()) == 0.0)


def main():
    tests = [test_J1_exact_prismatic,
             test_J2_offset_revolute,
             test_J3_J4_J5_gauge_and_constraints,
             test_J6_prismatic_regularity,
             test_J7_J8_jvp_against_finite_differences,
             test_J9_kinematic_adjoint,
             test_J10_orientation_and_covariance_transport,
             test_J11_tangent_covariance_reset,
             test_15_qdot_has_no_direct_measurement,
             test_16_covariance_transport_preserves_cross_terms,
             test_25_reduces_to_the_theory_note_jacobians,
             test_47M_static_gaussians_have_zero_derivative,
             test_S1_S2_variable_speed_sequences,
             test_S5_exact_P_without_fake_curvature]
    for fn in tests:
        print(f"\n{fn.__name__}")
        fn()
    print(f"\n{len(PASSED)} checks passed across {len(tests)} tests")


if __name__ == "__main__":
    main()
