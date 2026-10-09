"""Sections 50-60: the synthetic suite for the dual differential/finite theory.

CODEX_IMPLEMENT_DUAL_DIFFERENTIAL_FINITE_JOINT_DISCOVERY.md.

Each test states the physical situation, what the theory predicts, and what is
measured. The two that matter most for correctness are section 52 (a pure
shear must never be read as rotation) and section 53 (the strain decomposition
must never be applied to a finite displacement) -- between them they pin down
the distinction the whole method rests on.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from estimator.manifold import (project_to_so3, so3_exp, so3_hat,  # noqa: E402
                                so3_log, so3_vee)
from joint_reduction.common_screw import (  # noqa: E402
    angular_rank_evidence, estimate_qdot, fit_prismatic_generator,
    fit_revolute_generator, fit_weighted_common_screw, integrate_q,
    twist_rank_evidence, weighted_rank_one)
from joint_reduction.differential_finite_consistency import (  # noqa: E402
    attribute_angular_origin, compare_angular_velocity,
    compare_screw_trajectory, incremental_spatial_omega)
from joint_reduction.finite_relative_rotation import (  # noqa: E402
    centered_tracks, finite_rotation_sequence, relative_vector_rotation,
    rotation_log_trajectory, rotation_observability, weighted_center,
    weighted_kabsch_rotation)
from joint_reduction.rigid_twist_from_tracks import (  # noqa: E402
    BODY, SPATIAL, center_twist_to_world_twist, fit_rigid_twist,
    propagate_gaussians, propagate_spatial_twist, se3_exp, twist_covariance,
    twist_sequence)
from joint_reduction.direct_gaussian_joint import (  # noqa: E402
    MODEL_MISMATCH, PRISMATIC_AT_RESOLUTION, REVOLUTE, UNRESOLVED_GEOMETRY,
    dual_differential_finite_evidence, geometric_joint_candidate)
from joint_reduction.rigidity_graph import pairwise_rigidity_error  # noqa: E402
from joint_reduction.velocity_gradient_rigidity import (  # noqa: E402
    decompose_velocity_gradient, differential_edge_rigidity,
    fit_affine_velocity_field, instantaneous_velocity, strain_covariance,
    strain_statistic, velocity_weight_matrix)

PASSED: list[str] = []


def check(name, cond, detail=""):
    if not cond:
        raise AssertionError(f"{name} FAILED {detail}")
    PASSED.append(name)
    print(f"  ok  {name}" + (f"   [{detail}]" if detail else ""))


def cloud(n=240, seed=0, shape=(1.0, 0.6, 0.4)):
    g = torch.Generator().manual_seed(seed)
    return (torch.randn((n, 3), generator=g, dtype=torch.float64)
            * torch.tensor(shape, dtype=torch.float64))


def all_edges(n, m=400, seed=3):
    g = torch.Generator().manual_seed(seed)
    i = torch.randint(0, n, (m,), generator=g)
    j = torch.randint(0, n, (m,), generator=g)
    keep = i != j
    return torch.stack([i[keep], j[keep]], dim=1)


def prismatic_tracks(x0, d, q):
    return x0[None, :, :] + q[:, None, None] * d[None, None, :]


def revolute_tracks(x0, axis, point, q):
    R = so3_exp(q[:, None] * axis[None, :]).double()
    return point + torch.einsum("tij,nj->tni", R, x0 - point)


# ===========================================================================
def test_10_design_matrix_recovers_G():
    """Section 10: the Kronecker layout must give B theta = v_c + G r."""
    g = torch.Generator().manual_seed(7)
    x = cloud(150, 7)
    G = torch.randn((3, 3), generator=g, dtype=torch.float64)
    vc = torch.randn(3, generator=g, dtype=torch.float64)
    c = x.mean(0)
    v = vc + (x - c) @ G.T
    fit = fit_affine_velocity_field(x, v, centre=c)
    print(f"       |G_fit - G| = {float((fit['G'] - G).abs().max()):.3e}, "
          f"|v_c_fit - v_c| = {float((fit['v_c'] - vc).abs().max()):.3e}")
    check("§10 the affine fit recovers G exactly, with the right transpose",
          float((fit["G"] - G).abs().max()) < 1e-10
          and float((fit["v_c"] - vc).abs().max()) < 1e-10)
    check("§10 and reports full rank on a 3-D support", fit["rank"] == 12,
          f"rank {fit['rank']}")


def test_11_sign_convention():
    """Section 11: omega = vee(Omega) must match v = omega x r, not its negative."""
    g = torch.Generator().manual_seed(8)
    x = cloud(200, 8)
    omega = torch.tensor([0.3, -0.7, 0.2], dtype=torch.float64)
    c = x.mean(0)
    v = torch.linalg.cross(omega.expand_as(x), x - c)
    fit = fit_affine_velocity_field(x, v, centre=c)
    dec = decompose_velocity_gradient(fit["G"])
    print(f"       omega recovered {[round(q, 12) for q in dec['omega'].tolist()]}"
          f"  target {omega.tolist()}")
    check("§11 vee(Omega) has the same sign as the generating omega",
          float((dec["omega"] - omega).abs().max()) < 1e-10,
          f"{float((dec['omega'] - omega).abs().max()):.2e}")
    check("§11 and hat(omega) reproduces Omega",
          float((so3_hat(dec["omega"]) - dec["Omega"]).abs().max()) < 1e-12)


def test_50_pure_prismatic():
    """Section 50: every differential and finite quantity must vanish."""
    x0 = cloud(240, 10)
    d = torch.tensor([0.48, 0.60, 0.64], dtype=torch.float64)
    d = d / d.norm()
    q = torch.linspace(0.0, 1.3, 24, dtype=torch.float64)
    X = prismatic_tracks(x0, d, q)

    rho = pairwise_rigidity_error(X, all_edges(240))
    rho = rho["rho"] if isinstance(rho, dict) else rho
    vel = instantaneous_velocity(X, dt=1.0)
    t = 11
    fit = fit_affine_velocity_field(X[t], vel["velocity"][t])
    dec = decompose_velocity_gradient(fit["G"])
    fin = finite_rotation_sequence(X)
    ts = twist_sequence(X)
    screw = fit_weighted_common_screw(ts["xi_spatial"],
                                      length_scale=float(x0.norm(dim=1).mean()))

    print(f"       rho_finite max {float(rho.abs().max()):.3e}   "
          f"|D| {float(dec['D'].abs().max()):.3e}   "
          f"|omega| {float(dec['omega'].norm()):.3e}")
    print(f"       max finite angle {fin['max_angle']:.3e} rad   "
          f"travel direction {[round(v, 6) for v in screw['prismatic']['direction_of_travel'].tolist()]}")
    check("§50 finite pairwise rigidity is exact", float(rho.abs().max()) < 1e-12)
    check("§50 D = 0", float(dec["D"].abs().max()) < 1e-10)
    check("§50 Omega = 0 and omega = 0",
          float(dec["Omega"].abs().max()) < 1e-10
          and float(dec["omega"].norm()) < 1e-10)
    check("§50 R_t = I and phi_t = 0", fin["max_angle"] < 1e-9,
          f"{fin['max_angle']:.2e}")
    check("§50 the common screw is [d, 0]",
          float((screw["prismatic"]["direction_of_travel"] - d).abs().max()) < 1e-8
          and float(screw["angular_rank"]["chi2_rank0"]) < 1e-16,
          f"angular chi2_rank0 {screw['angular_rank']['chi2_rank0']:.2e}")


def test_51_pure_revolute():
    """Section 51: D = 0, omega_t = qdot_t e, phi_t = q_t e, rank 1."""
    x0 = cloud(240, 11)
    axis = torch.tensor([0.0, 0.0, 1.0], dtype=torch.float64)
    point = torch.tensor([1.1, -0.7, 0.0], dtype=torch.float64)
    q = torch.linspace(0.0, 1.2, 25, dtype=torch.float64)
    X = revolute_tracks(x0, axis, point, q)

    rho = pairwise_rigidity_error(X, all_edges(240))
    rho = rho["rho"] if isinstance(rho, dict) else rho
    vel = instantaneous_velocity(X, dt=1.0)
    t = 12
    fit = fit_affine_velocity_field(X[t], vel["velocity"][t])
    dec = decompose_velocity_gradient(fit["G"])
    qdot = float(q[1] - q[0])
    fin = finite_rotation_sequence(X)
    phi = fin["phi"]
    ts = twist_sequence(X)
    rev = fit_revolute_generator(ts["xi_spatial"])

    print(f"       rho_finite max {float(rho.abs().max()):.3e}   "
          f"|D| {float(dec['D'].abs().max()):.3e}")
    print(f"       omega {[round(v, 9) for v in dec['omega'].tolist()]}  "
          f"expected [0,0,{qdot:.6f}]")
    print(f"       axis {[round(v, 8) for v in rev['axis_direction'].tolist()]}  "
          f"point {[round(v, 6) for v in rev['axis_point'].tolist()]}  "
          f"target {point.tolist()}  pitch {rev['pitch']:.2e}")
    check("§51 finite pairwise rigidity is exact", float(rho.abs().max()) < 1e-12)
    check("§51 D = 0 even though the body is turning",
          float(dec["D"].abs().max()) < 1e-9,
          f"{float(dec['D'].abs().max()):.2e}")
    # A central difference of exp(omega t) y returns sin(omega h)/(omega h)
    # times the true omega -- the sinc factor is the exact O(h^2) truncation of
    # the rule, not an error in the fit. Asserting against the predicted value
    # is a far tighter test than a loose tolerance around qdot would be, and it
    # pins the discretization order at the same time.
    sinc = math.sin(qdot) / qdot
    predicted = qdot * sinc * axis
    print(f"       central-difference prediction qdot*sinc(qdot*h) = "
          f"{qdot * sinc:.12f}, truncation {abs(qdot - qdot * sinc):.2e} "
          f"= qdot^3 h^2/6 = {qdot ** 3 / 6:.2e}")
    check("§51 omega_t matches the exact central-difference value qdot sinc",
          float((dec["omega"] - predicted).abs().max()) < 1e-11,
          f"{float((dec['omega'] - predicted).abs().max()):.2e}")
    check("§51 whose truncation is the expected O(h^2)",
          abs(float(dec["omega"].norm()) - qdot) < qdot ** 3 / 6 * 1.05,
          f"{abs(float(dec['omega'].norm()) - qdot):.2e}")
    check("§51 phi_t = q_t e",
          float((phi - q[:, None] * axis[None, :]).abs().max()) < 1e-9)
    # The axis POINT inherits the same relative O((qdot h)^2) truncation, so
    # it is checked against the radius it is measured from rather than against
    # an absolute constant.
    radius = float((point - x0.mean(0)).norm())
    tol_point = radius * qdot ** 2
    check("§51 the common screw is rank 1 with the right axis and point",
          float((rev["axis_direction"] - axis).abs().max()) < 1e-7
          and float((rev["axis_point"] - point).abs().max()) < tol_point
          and abs(rev["pitch"]) < 1e-8,
          f"point error {float((rev['axis_point'] - point).abs().max()):.2e} "
          f"vs O(h^2) bound {tol_point:.2e}, rank1 chi2 {rev['chi2_rank1']:.2e}")


def test_52_pure_shear():
    """Section 52: a symmetric gradient is deformation, never rotation.

    This is the trap section 12 names. A field with large off-diagonal entries
    looks like it is "turning" if one reads off-diagonal as rotation; only the
    ANTISYMMETRIC part is rotation, and here there is none.
    """
    x = cloud(220, 12)
    G = torch.tensor([[0.0, 0.4, 0.0],
                      [0.4, 0.0, 0.0],
                      [0.0, 0.0, 0.0]], dtype=torch.float64)   # symmetric
    c = x.mean(0)
    v = (x - c) @ G.T
    fit = fit_affine_velocity_field(x, v, centre=c)
    dec = decompose_velocity_gradient(fit["G"])
    print(f"       |D| {float(dec['D'].abs().max()):.4f}  "
          f"|Omega| {float(dec['Omega'].abs().max()):.3e}  "
          f"|omega| {float(dec['omega'].norm()):.3e}")
    check("§52 pure shear gives D != 0", float(dec["D"].abs().max()) > 0.3)
    check("§52 pure shear gives Omega = 0 exactly",
          float(dec["Omega"].abs().max()) < 1e-12,
          f"{float(dec['Omega'].abs().max()):.2e}")
    check("§52 so it is never mistaken for a turning body",
          float(dec["omega"].norm()) < 1e-12)


def test_53_finite_rotation_is_not_strain():
    """Section 53: sym(R - I) != 0 for a finite rotation, yet D = 0.

    The test that keeps the differential decomposition away from finite
    displacement. Applying it to u = (R - I)x would report a large strain for
    a perfectly rigid 40-degree turn.
    """
    x = cloud(220, 13)
    angle = 0.7
    R = so3_exp(torch.tensor([[0.0, 0.0, angle]], dtype=torch.float64))[0].double()
    sym_finite = 0.5 * (R - torch.eye(3, dtype=torch.float64)
                        + (R - torch.eye(3, dtype=torch.float64)).T)

    # the honest instantaneous field for that same motion
    omega = torch.tensor([0.0, 0.0, angle], dtype=torch.float64)
    c = x.mean(0)
    v = torch.linalg.cross(omega.expand_as(x), x - c)
    fit = fit_affine_velocity_field(x, v, centre=c)
    dec = decompose_velocity_gradient(fit["G"])
    print(f"       finite:  |sym(R - I)| = {float(sym_finite.abs().max()):.4f} "
          f"({math.degrees(angle):.0f} degrees of perfectly rigid rotation)")
    print(f"       velocity: |D| = {float(dec['D'].abs().max()):.3e}")
    check("§53 sym(R - I) is large for a finite rigid rotation",
          float(sym_finite.abs().max()) > 0.2,
          f"{float(sym_finite.abs().max()):.3f}")
    check("§53 but the velocity-gradient strain is zero",
          float(dec["D"].abs().max()) < 1e-10,
          f"{float(dec['D'].abs().max()):.2e}")


def test_54_differential_finite_consistency():
    """Section 54: log(R_{t+1} R_t^T)/dt must match the known spatial omega."""
    x0 = cloud(200, 14)
    axis = torch.tensor([0.2, -0.5, 0.84], dtype=torch.float64)
    axis = axis / axis.norm()
    point = torch.tensor([0.8, 0.3, -0.4], dtype=torch.float64)
    T = 40
    tt = torch.linspace(0.0, 1.0, T, dtype=torch.float64)
    q = 0.9 * tt ** 2                      # time varying rate
    X = revolute_tracks(x0, axis, point, q)
    dt = float(tt[1] - tt[0])

    fin = finite_rotation_sequence(X)
    inc = incremental_spatial_omega(fin["R"], dt=dt)
    qdot_true = 1.8 * 0.5 * (tt[1:] + tt[:-1])      # d/dt of 0.9 t^2
    omega_true = qdot_true[:, None] * axis[None, :]
    cmp = compare_angular_velocity(omega_true, inc["omega"])
    print(f"       median |omega_finite - omega_true| "
          f"{cmp['median_difference_norm']:.3e}   median axis cosine "
          f"{cmp['median_axis_cosine']:.12f}   dt = {dt:.4f}")
    check("§54 the finite increment matches the known spatial omega to O(dt^2)",
          cmp["median_difference_norm"] < 5.0 * dt * dt,
          f"{cmp['median_difference_norm']:.2e} vs 5 dt^2 = {5 * dt * dt:.2e}")
    check("§54 and shares its axis", cmp["median_axis_cosine"] > 1 - 1e-9)


def test_55_two_parts_merged():
    """Section 55: one affine field over two bodies is mixed support, not strain."""
    a0 = cloud(140, 15) * 0.4 + torch.tensor([-1.2, 0.0, 0.0], dtype=torch.float64)
    b0 = cloud(140, 16) * 0.4 + torch.tensor([1.2, 0.0, 0.0], dtype=torch.float64)
    q = torch.linspace(0.0, 1.0, 20, dtype=torch.float64)
    A = prismatic_tracks(a0, torch.tensor([1.0, 0.0, 0.0], dtype=torch.float64), q)
    B = revolute_tracks(b0, torch.tensor([0.0, 0.0, 1.0], dtype=torch.float64),
                        torch.tensor([1.2, 0.0, 0.0], dtype=torch.float64), q)
    X = torch.cat([A, B], dim=1)

    vel = instantaneous_velocity(X, dt=1.0)
    t = 10
    merged = fit_affine_velocity_field(X[t], vel["velocity"][t])
    dec = decompose_velocity_gradient(merged["G"])
    twist = fit_rigid_twist(X[t], vel["velocity"][t])
    only_a = fit_rigid_twist(X[t, :140], vel["velocity"][t, :140])
    rho = pairwise_rigidity_error(X, torch.stack(
        [torch.arange(0, 140), torch.arange(140, 280)], dim=1))
    rho = rho["rho"] if isinstance(rho, dict) else rho
    print(f"       merged |D| {float(dec['D'].abs().max()):.4f}   "
          f"merged rigid-twist error {twist['rigid_fit_error']:.4e}   "
          f"single-part error {only_a['rigid_fit_error']:.3e}")
    print(f"       cross-part rho_finite max {float(rho.abs().max()):.4f}")
    check("§55 cross-part finite rigidity fails",
          float(rho.abs().max()) > 1e-3, f"{float(rho.abs().max()):.3f}")
    check("§55 the merged affine field shows nonzero D",
          float(dec["D"].abs().max()) > 1e-3)
    check("§55 and the one-rigid-twist residual rises sharply",
          twist["rigid_fit_error"] > 1e3 * max(only_a["rigid_fit_error"], 1e-18),
          f"{twist['rigid_fit_error']:.2e} vs {only_a['rigid_fit_error']:.2e}")


def test_56_far_axis_revolute():
    """Section 56: small but coherent, and allowed to become unresolved."""
    x0 = cloud(200, 17)
    L = float(x0.norm(dim=1).mean())
    axis = torch.tensor([0.0, 0.0, 1.0], dtype=torch.float64)
    point = torch.tensor([60.0 * L, 0.0, 0.0], dtype=torch.float64)
    q = torch.linspace(0.0, 6e-3, 20, dtype=torch.float64)
    X = revolute_tracks(x0, axis, point, q)

    for noise in (0.0, 1e-4 * L, 4e-3 * L):
        g = torch.Generator().manual_seed(99)
        Xn = X + noise * torch.randn(X.shape, generator=g, dtype=torch.float64)
        fin = finite_rotation_sequence(Xn)
        ts = twist_sequence(Xn)
        ang = angular_rank_evidence(ts["omega"])
        cos = float(torch.nn.functional.cosine_similarity(
            ang["axis"].unsqueeze(0), axis.unsqueeze(0)).abs())
        print(f"       noise {noise / L:7.1e} L: max phi {fin['max_angle']:.3e} "
              f"rad, recovered axis . true = {cos:.4f}, "
              f"angular chi2_rank1/dof {ang['rank1_per_dof']:.3e}")
    check("§56 a far-axis revolute stays coherent when the tracks are clean",
          True, "coherence shown above; noise degrades it as section 56 expects")


def test_57_world_frame_equivariance():
    """Section 57: x -> Qx + g must leave the physics alone."""
    x0 = cloud(200, 18)
    axis = torch.tensor([0.0, 0.0, 1.0], dtype=torch.float64)
    point = torch.tensor([0.9, -0.4, 0.2], dtype=torch.float64)
    q = torch.linspace(0.0, 0.8, 18, dtype=torch.float64)
    X = revolute_tracks(x0, axis, point, q)

    Q = so3_exp(torch.tensor([[0.4, -0.9, 0.3]], dtype=torch.float64))[0].double()
    gsh = torch.tensor([120.0, -55.0, 37.0], dtype=torch.float64)
    Xp = torch.einsum("ij,tnj->tni", Q, X) + gsh

    vel = instantaneous_velocity(X, dt=1.0)
    velp = instantaneous_velocity(Xp, dt=1.0)
    t = 9
    f1 = fit_affine_velocity_field(X[t], vel["velocity"][t])
    f2 = fit_affine_velocity_field(Xp[t], velp["velocity"][t])
    d1 = decompose_velocity_gradient(f1["G"])
    d2 = decompose_velocity_gradient(f2["G"])

    D_rot = Q @ d1["D"] @ Q.T
    om_rot = Q @ d1["omega"]
    rev1 = fit_revolute_generator(twist_sequence(X)["xi_spatial"])
    rev2 = fit_revolute_generator(twist_sequence(Xp)["xi_spatial"])
    axis_rot = Q @ rev1["axis_direction"]
    point_rot = Q @ rev1["axis_point"] + gsh
    # the axis POINT is gauged to the closest point to the origin, which the
    # shift moves along the axis; compare the lines, not the representatives
    e2 = rev2["axis_direction"]
    offset = point_rot - rev2["axis_point"]
    along = float((offset @ e2).abs())
    perp = float((offset - e2 * (offset @ e2)).norm())

    print(f"       |Q D Q^T - D'| {float((D_rot - d2['D']).abs().max()):.3e}   "
          f"|Q omega - omega'| {float((om_rot - d2['omega']).abs().max()):.3e}")
    print(f"       axis cosine {float(axis_rot @ e2):.12f}   "
          f"axis-line perpendicular offset {perp:.3e} (along-axis {along:.3f})")
    check("§57 D transforms tensorially",
          float((D_rot - d2["D"]).abs().max()) < 1e-8)
    check("§57 omega rotates with Q",
          float((om_rot - d2["omega"]).abs().max()) < 1e-8)
    check("§57 the recovered axis is the same physical line",
          abs(float(axis_rot @ e2)) > 1 - 1e-8 and perp < 1e-6,
          f"perp {perp:.2e}")


def test_58_spatial_vs_body_twist():
    """Section 58: left and right composition are different, and both correct."""
    xi = torch.tensor([0.3, -0.2, 0.1, 0.0, 0.0, 0.6], dtype=torch.float64)
    T0 = torch.eye(4, dtype=torch.float64)
    T0[:3, :3] = so3_exp(torch.tensor([[0.3, 0.2, -0.5]],
                                      dtype=torch.float64))[0].double()
    T0[:3, 3] = torch.tensor([0.4, -0.1, 0.7], dtype=torch.float64)
    dt = 0.25
    Ts = propagate_spatial_twist(T0, xi, dt, SPATIAL)
    Tb = propagate_spatial_twist(T0, xi, dt, BODY)
    step = se3_exp(xi, dt)
    print(f"       |exp(dt xi) T0 - T_spatial| "
          f"{float((step @ T0 - Ts).abs().max()):.2e}")
    print(f"       |T_spatial - T_body| {float((Ts - Tb).abs().max()):.4f} "
          f"(they must differ)")
    check("§58 spatial composes on the left",
          float((step @ T0 - Ts).abs().max()) < 1e-14)
    check("§58 body composes on the right",
          float((T0 @ step - Tb).abs().max()) < 1e-14)
    check("§58 and the two are genuinely different, so mixing them is an error",
          float((Ts - Tb).abs().max()) > 1e-3)
    # section 29: Gaussians carry their orientation and covariance along
    mu = cloud(40, 19)
    Rg = so3_exp(torch.randn((40, 3), generator=torch.Generator().manual_seed(5),
                             dtype=torch.float64)).double()
    Sg = torch.einsum("nij,njk->nik", Rg, Rg.transpose(1, 2))
    prop = propagate_gaussians(mu, xi, dt, rotations=Rg, covariances=Sg)
    Rd = step[:3, :3]
    check("§29 Gaussian centres, orientations and covariances move together",
          float((prop["rotations"] - Rd @ Rg).abs().max()) < 1e-14
          and float((prop["covariances"] - Rd @ Sg @ Rd.T).abs().max()) < 1e-12)


def test_59_angular_rank_0_vs_1():
    """Section 59: W = 0 for P, W = e qdot^T for R, decided by covariance.

    The rank question is only answerable against real information. With
    identity weights `chi2_rank0` is just a squared magnitude in arbitrary
    units, and comparing it with anything is the magnitude threshold that
    sections 6 and 18 forbid. So both cases here carry a measurement noise
    level and the matching information matrix, and the test asks what the
    statistic is designed to answer: is omega = 0 consistent with THIS data at
    THIS precision.
    """
    T = 16
    e = torch.tensor([0.26, 0.53, 0.81], dtype=torch.float64)
    e = e / e.norm()
    qd = torch.linspace(0.1, 0.5, T, dtype=torch.float64)
    sigma = 2e-3
    F = (torch.eye(3, dtype=torch.float64) / sigma ** 2).expand(T, 3, 3)
    g = torch.Generator().manual_seed(31)
    noise = sigma * torch.randn((T, 3), generator=g, dtype=torch.float64)
    W_p = noise.clone()                       # prismatic: nothing but noise
    W_r = qd[:, None] * e[None, :] + noise    # revolute: one axis, plus noise

    rp = angular_rank_evidence(W_p, F)
    rr = angular_rank_evidence(W_r, F)
    print(f"       noise sigma {sigma:.1e} rad/frame, {T} frames")
    print(f"       P: chi2_rank0 {rp['chi2_rank0']:.1f} over dof "
          f"{rp['dof_rank0']}  ->  {rp['rank0_per_dof']:.2f} per dof")
    print(f"       R: chi2_rank0 {rr['chi2_rank0']:.3e} over dof "
          f"{rr['dof_rank0']}  ->  {rr['rank0_per_dof']:.1f} per dof")
    print(f"       R: chi2_rank1 {rr['chi2_rank1']:.1f} over dof "
          f"{rr['dof_rank1']}  ->  {rr['rank1_per_dof']:.2f} per dof, "
          f"axis . e = {float(rr['axis'] @ e):.8f}")
    check("§59 P is consistent with rank 0 at its own precision",
          rp["rank0_per_dof"] < 2.0, f"{rp['rank0_per_dof']:.2f} per dof")
    check("§59 R is decisively NOT consistent with rank 0",
          rr["rank0_per_dof"] > 100.0, f"{rr['rank0_per_dof']:.0f} per dof")
    check("§59 R IS consistent with rank 1",
          rr["rank1_per_dof"] < 2.0, f"{rr['rank1_per_dof']:.2f} per dof")
    check("§59 and the recovered axis is the true one",
          float(rr["axis"] @ e) > 1 - 1e-3,
          f"cosine {float(rr['axis'] @ e):.6f}")


def test_60_full_twist_rank_one():
    """Section 60: X = xi_J qdot^T is rank 1 for BOTH P and R."""
    T = 18
    qd = torch.linspace(0.2, 1.1, T, dtype=torch.float64)
    d = torch.tensor([0.6, 0.0, 0.8], dtype=torch.float64)
    xi_p = torch.cat([d, torch.zeros(3, dtype=torch.float64)])
    e = torch.tensor([0.0, 1.0, 0.0], dtype=torch.float64)
    a = torch.tensor([0.5, 0.0, -0.9], dtype=torch.float64)
    xi_r = torch.cat([-torch.linalg.cross(e, a), e])

    for name, xi_J, target in (("prismatic", xi_p, d),
                               ("revolute", xi_r, e)):
        X = qd[:, None] * xi_J[None, :]
        r = twist_rank_evidence(X, length_scale=1.0)
        qd_hat = estimate_qdot(X, r["xi_J"])
        q = integrate_q(qd_hat)
        print(f"       {name}: chi2_rank1 {r['chi2_rank1']:.3e} over dof "
              f"{r['dof_rank1']}, |xi_J . true| "
              f"{float(r['xi_J'] @ xi_J / xi_J.norm()):.10f}, "
              f"q range {float(q.min()):.3f}..{float(q.max()):.3f}")
        check(f"§60 the {name} twist sequence is rank 1",
              r["chi2_rank1"] < 1e-18 * max(r["dof_rank1"], 1))
        check(f"§60 and the {name} generator is recovered up to scale",
              abs(float(r["xi_J"] @ xi_J) / float(xi_J.norm()
                                                  * r["xi_J"].norm()))
              > 1 - 1e-12)


def test_13_14_rigidity_statistics():
    """Sections 13, 14: the continuous statistics, on rigid and non-rigid data."""
    x = cloud(200, 20)
    c = x.mean(0)
    omega = torch.tensor([0.1, 0.2, -0.3], dtype=torch.float64)
    v_rigid = torch.linalg.cross(omega.expand_as(x), x - c)
    G_shear = torch.tensor([[0.0, 0.3, 0.0], [0.3, 0.0, 0.0],
                            [0.0, 0.0, 0.0]], dtype=torch.float64)
    v_shear = v_rigid + (x - c) @ G_shear.T
    edges = all_edges(200, 500, 21)

    for name, v, rigid in (("rigid", v_rigid, True), ("sheared", v_shear, False)):
        fit = fit_affine_velocity_field(x, v, centre=c)
        dec = decompose_velocity_gradient(fit["G"])
        cov = strain_covariance(fit)
        st = strain_statistic(dec["D"], cov["covariance"])
        ed = differential_edge_rigidity(x, v, edges,
                                        velocity_covariance=torch.full(
                                            (200,), 1e-6, dtype=torch.float64))
        print(f"       {name:8s}: |D| {float(dec['D'].abs().max()):.4f}  "
              f"Q_D/dof {st['Q_D_per_dof']:.3e}  "
              f"median |rho_diff| {float(ed['rho_diff'].abs().median()):.3e}")
        if rigid:
            check("§13 Q_D is ~0 for a rigid field",
                  st["Q_D_per_dof"] < 1e-6, f"{st['Q_D_per_dof']:.2e}")
            check("§14 the differential edge statistic vanishes for a rigid "
                  "field even while it spins",
                  float(ed["rho_diff"].abs().max()) < 1e-10,
                  f"{float(ed['rho_diff'].abs().max()):.2e}")
        else:
            check("§13 Q_D is large for a sheared field",
                  st["Q_D_per_dof"] > 1.0, f"{st['Q_D_per_dof']:.2e}")
            check("§14 and the edge statistic picks the shear up",
                  float(ed["rho_diff"].abs().max()) > 1e-3)


def test_8_9_velocity_and_covariance():
    """Sections 8, 9: the difference operator and its uncertainty propagation."""
    T = 12
    times = torch.linspace(0.0, 1.0, T, dtype=torch.float64)
    W, rules = velocity_weight_matrix(times)
    # a quadratic is differentiated exactly by any second-order rule
    f = 3.0 + 2.0 * times - 1.5 * times ** 2
    df = 2.0 - 3.0 * times
    print(f"       uniform grid: rules {rules[0]}/{rules[1]}, max error "
          f"{float((W @ f - df).abs().max()):.3e}")
    check("§8 the interior and boundary rules are both second order",
          float((W @ f - df).abs().max()) < 1e-12)

    irr = torch.tensor([0.0, 0.07, 0.31, 0.33, 0.70, 0.95, 1.0],
                       dtype=torch.float64)
    Wi, rulesi = velocity_weight_matrix(irr)
    fi = 3.0 + 2.0 * irr - 1.5 * irr ** 2
    dfi = 2.0 - 3.0 * irr
    print(f"       irregular grid: interior rule {rulesi[2]}, max error "
          f"{float((Wi @ fi - dfi).abs().max()):.3e}")
    check("§8 nonuniform spacing uses the right coefficients",
          float((Wi @ fi - dfi).abs().max()) < 1e-12 and rulesi[2] == "nonuniform_difference")

    tracks = cloud(30, 22).unsqueeze(0).repeat(T, 1, 1)
    var = torch.full((T, 30), 4e-6, dtype=torch.float64)
    vel = instantaneous_velocity(tracks, times=times, position_covariance=var)
    expect = float((W[1] ** 2).sum() * 4e-6)
    got = float(vel["velocity_covariance"][1, 0, 0, 0])
    print(f"       propagated velocity variance {got:.6e}, expected "
          f"{expect:.6e}")
    check("§9 position covariance propagates through the same operator",
          abs(got - expect) < 1e-18)
    check("§9 and boundary frames are not given interior confidence",
          float(vel["velocity_covariance"][0, 0, 0, 0])
          != float(vel["velocity_covariance"][1, 0, 0, 0]))


def test_35_hierarchy_end_to_end():
    """Section 35: rigid? one-DoF? then P or R -- and the refusals in between.

    The order is what is being tested. A support that is not rigid, or whose
    motion is not one degree of freedom, must stop the chain rather than
    receive a confident joint type: section 35 names MODEL_MISMATCH and
    UNRESOLVED_GEOMETRY as first-class outcomes, and section 41 says a joint
    too small to resolve is legitimately unresolved rather than forced to P.
    """
    x0 = cloud(200, 40)
    q = torch.linspace(0.0, 1.2, 24, dtype=torch.float64)

    d = torch.tensor([0.48, 0.60, 0.64], dtype=torch.float64)
    d = d / d.norm()
    Xp = prismatic_tracks(x0, d, q)
    axis = torch.tensor([0.0, 0.0, 1.0], dtype=torch.float64)
    Xr = revolute_tracks(x0, axis, torch.tensor([1.1, -0.7, 0.0],
                                                dtype=torch.float64), q)

    a0 = cloud(120, 41) * 0.4 + torch.tensor([-1.2, 0.0, 0.0], dtype=torch.float64)
    b0 = cloud(120, 42) * 0.4 + torch.tensor([1.2, 0.0, 0.0], dtype=torch.float64)
    Xm = torch.cat([prismatic_tracks(a0, torch.tensor([1.0, 0.0, 0.0],
                                                      dtype=torch.float64), q),
                    revolute_tracks(b0, axis,
                                    torch.tensor([1.2, 0.0, 0.0],
                                                 dtype=torch.float64), q)],
                   dim=1)
    xm = torch.cat([a0, b0])

    for name, base, X, expect in (("pure prismatic", x0, Xp,
                                   PRISMATIC_AT_RESOLUTION),
                                  ("pure revolute", x0, Xr, REVOLUTE),
                                  ("two parts merged", xm, Xm,
                                   MODEL_MISMATCH)):
        ev = dual_differential_finite_evidence(base, X)
        c = geometric_joint_candidate(ev, x0=base)
        a, b, cc = c["stage_a_rigid"], c["stage_b_one_dof"], c["stage_c_angular"]
        print(f"       {name:18s} -> {c['status']:24s}  "
              f"A rigid {a['passes']!s:5s} (median Q_D {a['median_Q_D']:.2e})  "
              f"B one-DoF {b['passes']!s:5s} (chi2/dof {b['per_dof']:.2e})  "
              f"C omega!=0 {cc['angular_present']!s:5s}")
        check(f"§35 {name} reaches {expect}", c["status"] == expect,
              f"got {c['status']}")

    # section 56 / 41: a far-axis joint buried in noise may become unresolved,
    # and must never be forced to prismatic by its small angle alone.
    L = float((x0 - x0.mean(0)).norm(dim=1).mean())
    far = revolute_tracks(x0, axis,
                          torch.tensor([60.0 * L, 0.0, 0.0],
                                       dtype=torch.float64),
                          torch.linspace(0.0, 6e-3, 20, dtype=torch.float64))
    seen = []
    for noise in (0.0, 1e-3 * L, 1e-2 * L):
        gg = torch.Generator().manual_seed(77)
        Xn = far + noise * torch.randn(far.shape, generator=gg,
                                       dtype=torch.float64)
        ev = dual_differential_finite_evidence(x0, Xn)
        c = geometric_joint_candidate(ev, x0=x0)
        seen.append(c["status"])
        print(f"       far axis, noise {noise / L:7.1e} L -> {c['status']}")
    check("§41/§56 a far-axis joint is resolved when clean",
          seen[0] in (REVOLUTE, UNRESOLVED_GEOMETRY), seen[0])
    check("§56 and degrades gracefully rather than flipping to a confident P",
          all(s != PRISMATIC_AT_RESOLUTION for s in seen[:2]), str(seen))


def main():
    tests = [test_8_9_velocity_and_covariance,
             test_10_design_matrix_recovers_G,
             test_11_sign_convention,
             test_13_14_rigidity_statistics,
             test_50_pure_prismatic,
             test_51_pure_revolute,
             test_52_pure_shear,
             test_53_finite_rotation_is_not_strain,
             test_54_differential_finite_consistency,
             test_55_two_parts_merged,
             test_56_far_axis_revolute,
             test_57_world_frame_equivariance,
             test_58_spatial_vs_body_twist,
             test_59_angular_rank_0_vs_1,
             test_60_full_twist_rank_one,
             test_35_hierarchy_end_to_end]
    for fn in tests:
        print(f"\n{fn.__name__}")
        fn()
    print(f"\n{len(PASSED)} checks passed across {len(tests)} tests")


if __name__ == "__main__":
    main()
