"""Tests for sections 28-30 of CODEX_IMPLEMENT_SPARSE_ALPHA_AWARE_IEKF.

Covers the reduced forms of the mandatory tests T17 (rigid reduction
consistency), T18 (prismatic reduction) and T19 (revolute reduction), plus the
primitives they rest on:

  * section 28 -- exact twist recovery, pure translation and the honest report of
    rank deficiency, IRLS robustness to gross outliers, membership compatibility,
    static-vs-moving evidence, debug export;
  * section 29.1/29.2 -- transport/fit round trips and, the key consistency
    check, the pose tangents ``G`` against a central difference of the transport
    functions with respect to ``q``;
  * section 29.3 -- reduced information is PSD and monotone in the number of
    Gaussians;
  * section 30 -- ``predict(0)`` is the identity, constant ``qdot`` integrates
    linearly in *real* (non-uniform) ``dt``, and the scalar normal system
    ``I_q dq = b_q`` recovers a known ``q``.

Autograd appears nowhere: section 36 allows it only as an oracle, and here the
oracle is finite differencing, which also validates the small-angle branches.

Run with:
    cd /home/samuelemara/Joint3DGS/RigEuler3DGS && PYTHONPATH=. python tests/test_joint_reduction.py
"""

from __future__ import annotations

import ast
import math
import tempfile
from pathlib import Path

import torch

import joint_reduction as jr
from joint_reduction import joint_filter, prismatic, revolute, rigid_fit

torch.manual_seed(20261004)
DT = torch.float64
PASSED: list[str] = []


def check(name: str, condition: bool, detail: str = "") -> None:
    if not condition:
        raise AssertionError(f"{name} FAILED {detail}")
    PASSED.append(name)
    print(f"  ok  {name}" + (f"   [{detail}]" if detail else ""))


def make_points(n: int = 64, seed: int = 11, spread: float = 2.0) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    return (torch.rand((n, 3), generator=g, dtype=DT) * 2.0 - 1.0) * spread


def random_rotations(n: int, seed: int = 5) -> torch.Tensor:
    """Random SO(3) via QR of a Gaussian matrix, determinant fixed to +1."""
    g = torch.Generator().manual_seed(seed)
    a = torch.randn((n, 3, 3), generator=g, dtype=DT)
    q, r = torch.linalg.qr(a)
    sign = torch.sign(torch.diagonal(r, dim1=-2, dim2=-1))
    q = q * sign[:, None, :]
    flip = torch.linalg.det(q) < 0
    q[flip, :, 0] = -q[flip, :, 0]
    return q


def rigid_velocity(mu: torch.Tensor, v0: torch.Tensor, omega: torch.Tensor,
                   c: torch.Tensor) -> torch.Tensor:
    return v0 + torch.linalg.cross(omega.expand_as(mu), mu - c, dim=-1)


# ---------------------------------------------------------------------------
# Section 28 -- rigid motion compression
# ---------------------------------------------------------------------------
def test_design_matrix_matches_the_twist_field():
    mu = make_points(32)
    c = mu.mean(dim=0)
    v0 = torch.tensor([0.3, -0.7, 1.1], dtype=DT)
    omega = torch.tensor([-0.2, 0.9, 0.4], dtype=DT)
    twist = torch.cat((v0, omega))
    direct = rigid_velocity(mu, v0, omega, c)
    through_design = torch.einsum("nai,i->na", rigid_fit.rigid_design(mu, c), twist)
    check("A_i xi == v0 + omega x (mu_i - c)",
          torch.allclose(direct, through_design, atol=1e-14),
          f"max {float((direct - through_design).abs().max()):.3e}")
    check("hat(v) w == cross(v, w)",
          torch.allclose(rigid_fit.hat(v0) @ omega, torch.linalg.cross(v0, omega), atol=1e-15))


def test_fit_rigid_twist_recovers_a_known_twist_exactly():
    """T17: noiseless rigid field -> the exact (v0, omega), no residual."""
    mu = make_points(128, seed=3)
    c_true = torch.tensor([0.4, -0.25, 0.8], dtype=DT)
    v0 = torch.tensor([0.11, -0.52, 0.37], dtype=DT)
    omega = torch.tensor([0.63, 0.21, -0.44], dtype=DT)
    v = rigid_velocity(mu, v0, omega, c_true)

    fit = jr.fit_rigid_twist(mu, v)
    # The fit centre is the centroid, so v0 must be re-expressed there before
    # comparing: v0(c') = v0(c) + omega x (c' - c).
    v0_at_fit = v0 + torch.linalg.cross(omega, fit["center"] - c_true)
    check("omega recovered exactly", torch.allclose(fit["omega"], omega, atol=1e-12),
          f"err {float((fit['omega'] - omega).abs().max()):.3e}")
    check("v0 recovered exactly (re-expressed at the fit centre)",
          torch.allclose(fit["v0"], v0_at_fit, atol=1e-12),
          f"err {float((fit['v0'] - v0_at_fit).abs().max()):.3e}")
    check("weighted residual is zero", fit["residual"] < 1e-12, f"{fit['residual']:.3e}")
    check("per-point residual is zero",
          float(fit["per_point_residual"].max()) < 1e-12)
    check("normal matrix is full rank on a generic cloud", fit["rank"] == 6)
    check("not flagged rank deficient", fit["rank_deficient"] is False)
    check("scaled condition number is sane", fit["condition_number_scaled"] < 1e4,
          f"{fit['condition_number_scaled']:.3e}")
    check("a moving part gets static_evidence ~ 0", fit["static_evidence"] < 1e-20,
          f"{fit['static_evidence']:.3e}")
    check("velocity_fit reproduces the data",
          torch.allclose(fit["velocity_fit"], v, atol=1e-12))
    # T17 in reduced form: the compressed 6-vector reproduces the free
    # per-Gaussian velocities of points that were never part of the fit.
    held_out = make_points(40, seed=77)
    pred = rigid_velocity(held_out, fit["v0"], fit["omega"], fit["center"])
    truth = rigid_velocity(held_out, v0, omega, c_true)
    check("T17: compressed twist predicts held-out Gaussian velocities",
          torch.allclose(pred, truth, atol=1e-11),
          f"max {float((pred - truth).abs().max()):.3e}")


def test_fit_rigid_twist_reports_rank_deficiency_on_pure_translation():
    """Pure translation: omega ~ 0, and the unobservable rotation is reported."""
    mu = make_points(96, seed=19)
    v0 = torch.tensor([0.5, -0.2, 0.9], dtype=DT)
    v = v0.expand_as(mu).clone()
    fit = jr.fit_rigid_twist(mu, v)
    check("pure translation: omega ~ 0", float(fit["omega"].abs().max()) < 1e-13,
          f"{float(fit['omega'].abs().max()):.3e}")
    check("pure translation: v0 recovered", torch.allclose(fit["v0"], v0, atol=1e-13))
    check("a generic cloud still observes all 6 directions", fit["rank"] == 6)

    # Collinear Gaussians: rotation about the line itself moves nothing, so one
    # twist direction is genuinely unobservable and must be reported, not hidden.
    line_dir = torch.tensor([1.0, -2.0, 0.5], dtype=DT)
    line_dir = line_dir / line_dir.norm()
    s = torch.linspace(-1.0, 1.0, 50, dtype=DT)[:, None]
    mu_line = s * line_dir + torch.tensor([0.2, 0.1, -0.3], dtype=DT)
    v_line = v0.expand_as(mu_line).clone()
    fit_line = jr.fit_rigid_twist(mu_line, v_line)
    check("collinear + translation: rank is 5, not a silent 6", fit_line["rank"] == 5,
          f"rank {fit_line['rank']}")
    check("collinear + translation: flagged rank deficient",
          fit_line["rank_deficient"] is True)
    check("collinear + translation: condition number reported as infinite",
          math.isinf(fit_line["condition_number"])
          and math.isinf(fit_line["condition_number_scaled"]))
    check("collinear + translation: omega stays 0 instead of exploding",
          float(fit_line["omega"].abs().max()) < 1e-12,
          f"{float(fit_line['omega'].abs().max()):.3e}")
    check("collinear + translation: v0 still exact",
          torch.allclose(fit_line["v0"], v0, atol=1e-12))
    # The null direction is "rotate about the line", i.e. [0; line_dir].
    evals, evecs = torch.linalg.eigh(fit_line["normal_matrix"])
    null = evecs[:, 0]
    check("collinear: the null eigenvector is rotation about the line",
          float(null[:3].abs().max()) < 1e-10
          and abs(abs(float(null[3:] @ line_dir)) - 1.0) < 1e-10,
          f"null {[round(float(x), 6) for x in null]}")
    check("collinear: the data is still fit exactly", fit_line["residual"] < 1e-13)


def test_fit_rigid_twist_separates_static_from_moving():
    mu = make_points(64, seed=23)
    static = jr.fit_rigid_twist(mu, torch.zeros_like(mu))
    check("a static group fits the zero twist",
          float(static["twist"].abs().max()) < 1e-15)
    check("a static group gets static_evidence == 1",
          abs(static["static_evidence"] - 1.0) < 1e-12,
          f"{static['static_evidence']}")

    g = torch.Generator().manual_seed(101)
    jitter = 1e-3 * torch.randn(mu.shape, generator=g, dtype=DT)
    noisy_static = jr.fit_rigid_twist(mu, jitter)
    check("a jittering static group still looks static",
          noisy_static["static_evidence"] > 0.5,
          f"{noisy_static['static_evidence']:.3f}")

    omega = torch.tensor([0.0, 0.0, 1.3], dtype=DT)
    moving = jr.fit_rigid_twist(
        mu, rigid_velocity(mu, torch.zeros(3, dtype=DT), omega, mu.mean(dim=0)) + jitter)
    check("a rotating group does not look static", moving["static_evidence"] < 1e-5,
          f"{moving['static_evidence']:.3e}")


def test_fit_rigid_twist_is_robust_to_gross_outliers():
    """10% gross outliers: both the weighted and the IRLS paths survive."""
    n = 200
    mu = make_points(n, seed=31)
    c = mu.mean(dim=0)
    v0 = torch.tensor([0.25, 0.4, -0.15], dtype=DT)
    omega = torch.tensor([-0.5, 0.35, 0.6], dtype=DT)
    g = torch.Generator().manual_seed(997)
    v = rigid_velocity(mu, v0, omega, c) + 1e-4 * torch.randn((n, 3), generator=g, dtype=DT)

    n_out = n // 10
    outlier = torch.zeros(n, dtype=torch.bool)
    outlier[torch.randperm(n, generator=g)[:n_out]] = True
    v = v.clone()
    v[outlier] += 5.0 * torch.randn((n_out, 3), generator=g, dtype=DT) + 3.0

    plain = jr.fit_rigid_twist(mu, v)
    check("without robustness the outliers do bias the fit",
          float((plain["omega"] - omega).abs().max()) > 1e-2,
          f"err {float((plain['omega'] - omega).abs().max()):.3e}")

    # (a) sensible externally supplied weights (what membership_compatibility
    #     produces on the next EM sweep).
    weights = torch.where(outlier, torch.full((n,), 1e-6, dtype=DT), torch.ones(n, dtype=DT))
    weighted = jr.fit_rigid_twist(mu, v, weights)
    check("weighted fit recovers omega despite 10% outliers",
          float((weighted["omega"] - omega).abs().max()) < 5e-4,
          f"err {float((weighted['omega'] - omega).abs().max()):.3e}")
    # v0 is attached to the fit centre (the *weighted* centroid here), so it is
    # compared after re-expression -- the rigid field itself is what must match.
    v0_at_fit = v0 + torch.linalg.cross(omega, weighted["center"] - c)
    check("weighted fit recovers v0 despite 10% outliers",
          float((weighted["v0"] - v0_at_fit).abs().max()) < 5e-4,
          f"err {float((weighted['v0'] - v0_at_fit).abs().max()):.3e}")

    # (b) IRLS from uniform weights -- implemented as Huber-seeded Tukey IRLS.
    for kernel, tol in (("huber", 5e-2), ("tukey", 5e-4)):
        rob = jr.fit_rigid_twist(mu, v, robust=kernel, irls_iters=12)
        err = float((rob["omega"] - omega).abs().max())
        check(f"IRLS ({kernel}) recovers omega from uniform weights", err < tol,
              f"err {err:.3e}")
    rob = jr.fit_rigid_twist(mu, v, robust="tukey", irls_iters=12)
    check("IRLS reports the weights it used", rob["robust_weights"].shape == (n,))
    check("IRLS rejects the outliers", float(rob["robust_weights"][outlier].max()) < 1e-3,
          f"max outlier weight {float(rob['robust_weights'][outlier].max()):.3e}")
    check("IRLS keeps the inliers", float(rob["robust_weights"][~outlier].min()) > 0.5,
          f"min inlier weight {float(rob['robust_weights'][~outlier].min()):.3f}")
    check("IRLS ran more than one linearisation", rob["iterations"] >= 12)


def test_fit_rigid_twist_uses_velocity_covariance():
    """Section 28: fuse filtered velocities *with their uncertainties*."""
    n = 60
    mu = make_points(n, seed=47)
    c = mu.mean(dim=0)
    v0 = torch.zeros(3, dtype=DT)
    omega = torch.tensor([0.0, 0.7, 0.0], dtype=DT)
    v = rigid_velocity(mu, v0, omega, c)
    g = torch.Generator().manual_seed(5)
    bad = torch.zeros(n, dtype=torch.bool)
    bad[::5] = True
    v = v.clone()
    v[bad] += 0.4 * torch.randn((int(bad.sum()), 3), generator=g, dtype=DT)
    var = torch.where(bad, torch.full((n,), 1e4, dtype=DT), torch.full((n,), 1e-4, dtype=DT))
    cov = var[:, None, None] * torch.eye(3, dtype=DT)

    with_cov = jr.fit_rigid_twist(mu, v, cov=cov)
    without = jr.fit_rigid_twist(mu, v)
    check("covariance-weighted fit beats the unweighted one",
          float((with_cov["omega"] - omega).abs().max())
          < 0.05 * float((without["omega"] - omega).abs().max()),
          f"{float((with_cov['omega'] - omega).abs().max()):.3e}"
          f" vs {float((without['omega'] - omega).abs().max()):.3e}")
    check("information-weighted normal matrix scales with 1/sigma^2",
          float(torch.diagonal(with_cov["normal_matrix"])[:3].min()) > 1e3,
          f"{float(torch.diagonal(with_cov['normal_matrix'])[:3].min()):.3e}")


def test_membership_compatibility_is_a_bounded_membership():
    n = 80
    mu = make_points(n, seed=53)
    c = mu.mean(dim=0)
    v0 = torch.tensor([0.1, 0.2, -0.3], dtype=DT)
    omega = torch.tensor([0.4, -0.1, 0.25], dtype=DT)
    v = rigid_velocity(mu, v0, omega, c)
    v = v.clone()
    v[:8] += 2.0
    twist = torch.cat((v0, omega))
    comp = jr.membership_compatibility(mu, v, twist, c, 0.05)
    check("compatibility is in [0,1]",
          bool(((comp >= 0.0) & (comp <= 1.0)).all()))
    check("inliers are fully compatible", float(comp[8:].min()) > 1.0 - 1e-12)
    check("gross outliers are rejected", float(comp[:8].max()) < 1e-6,
          f"{float(comp[:8].max()):.3e}")
    per_point_sigma = torch.full((n,), 0.05, dtype=DT)
    per_point_sigma[:8] = 1e4  # a Gaussian we trust not at all is compatible again
    comp2 = jr.membership_compatibility(mu, v, twist, c, per_point_sigma)
    check("per-point sigma is honoured", float(comp2[:8].min()) > 0.99,
          f"{float(comp2[:8].min()):.6f}")
    # Feeding compatibility back as weights is the EM step of section 28.
    refit = jr.fit_rigid_twist(mu, v, comp)
    check("compatibility weights repair the fit",
          torch.allclose(refit["omega"], omega, atol=1e-10),
          f"err {float((refit['omega'] - omega).abs().max()):.3e}")


def test_rigid_fit_debug_export():
    mu = make_points(24, seed=61)
    v = rigid_velocity(mu, torch.tensor([0.1, 0.0, 0.0], dtype=DT),
                       torch.tensor([0.0, 0.0, 0.5], dtype=DT), mu.mean(dim=0))
    fit = jr.fit_rigid_twist(mu, v, robust="huber")
    report = jr.rigid_fit_report(fit)
    for token in ("v0", "omega", "rank", "static_evidence", "cond(jacobi)"):
        check(f"report mentions {token}", token in report)
    with tempfile.TemporaryDirectory() as tmp:
        paths = jr.export_rigid_fit(fit, str(Path(tmp) / "fit"))
        check("export wrote the npz", Path(paths["npz"]).is_file())
        check("export wrote the json", Path(paths["json"]).is_file())
        import json

        import numpy as np
        blob = np.load(paths["npz"])
        check("npz round-trips the twist",
              np.allclose(blob["twist"], fit["twist"].numpy(), atol=0))
        check("npz carries the per-Gaussian residuals",
              blob["per_point_residual"].shape == (24,))
        meta = json.loads(Path(paths["json"]).read_text(encoding="utf-8"))
        check("json carries the accept/reject scalars",
              meta["rank"] == fit["rank"]
              and abs(meta["static_evidence"] - fit["static_evidence"]) < 1e-15)


# ---------------------------------------------------------------------------
# Section 29.1 / 29.2 -- prismatic and revolute reduction
# ---------------------------------------------------------------------------
def test_prismatic_transport_then_fit_recovers_the_axis():
    """T18: known prismatic axis, rigid transport, exact recovery."""
    mu0 = make_points(90, seed=71)
    R0 = random_rotations(90, seed=72)
    e_true = torch.tensor([0.3, -0.9, 0.4], dtype=DT)
    e_true = e_true / e_true.norm()
    q_true = torch.tensor(0.7314, dtype=DT)

    mu_t, R_t = jr.prismatic_transport(mu0, R0, e_true, q_true)
    check("prismatic transport translates by q*e",
          torch.allclose(mu_t - mu0, (q_true * e_true).expand_as(mu0), atol=1e-15))
    check("prismatic transport leaves orientation untouched",
          torch.equal(R_t, R0))

    fit = jr.fit_prismatic_axis(mu0, mu_t)
    check("T18: prismatic axis recovered to <1e-9",
          float((fit["axis"] - e_true).abs().max()) < 1e-9,
          f"err {float((fit['axis'] - e_true).abs().max()):.3e}")
    check("T18: q recovered to <1e-9", abs(float(fit["q"] - q_true)) < 1e-9,
          f"err {abs(float(fit['q'] - q_true)):.3e}")
    check("T18: prismatic residual is zero", fit["residual"] < 1e-12,
          f"{fit['residual']:.3e}")
    check("off-axis energy is zero for an exactly prismatic motion",
          fit["offaxis_ratio"] < 1e-12, f"{fit['offaxis_ratio']:.3e}")
    check("not flagged degenerate", fit["degenerate"] is False)

    # Negative slide: the sign convention points e along the motion.
    mu_b, _ = jr.prismatic_transport(mu0, None, e_true, -q_true)
    fit_b = jr.fit_prismatic_axis(mu0, mu_b)
    check("sign convention: q >= 0 and e flips instead",
          float(fit_b["q"]) > 0.0
          and torch.allclose(fit_b["axis"], -e_true, atol=1e-9))
    mu_round, _ = jr.prismatic_transport(mu0, None, fit_b["axis"], fit_b["q"])
    check("the flipped fit still transports onto the data",
          torch.allclose(mu_round, mu_b, atol=1e-12))


def test_revolute_transport_then_fit_recovers_axis_and_point():
    """T19: known revolute axis/point, exact recovery up to the axis slide."""
    mu0 = make_points(120, seed=81)
    R0 = random_rotations(120, seed=82)
    e_true = torch.tensor([-0.2, 0.5, 0.84], dtype=DT)
    e_true = e_true / e_true.norm()
    a_true = torch.tensor([0.35, -0.6, 0.15], dtype=DT)
    q_true = torch.tensor(0.6123, dtype=DT)

    mu_t, R_t = jr.revolute_transport(mu0, R0, e_true, a_true, q_true)
    rot = jr.rotation_about_axis(e_true, q_true)
    check("revolute transport matches a + R(mu0 - a)",
          torch.allclose(mu_t, a_true + (mu0 - a_true) @ rot.T, atol=1e-15))
    check("revolute transport left-multiplies the orientations",
          torch.allclose(R_t, rot @ R0, atol=1e-15))
    check("R(e,q) is a rotation",
          torch.allclose(rot @ rot.T, torch.eye(3, dtype=DT), atol=1e-15)
          and abs(float(torch.linalg.det(rot)) - 1.0) < 1e-15)
    check("every displacement is orthogonal to the axis (the fit's premise)",
          float(((mu_t - mu0) @ e_true).abs().max()) < 1e-15)

    fit = jr.fit_revolute_axis(mu0, mu_t)
    check("T19: revolute axis direction recovered to <1e-9",
          float((fit["axis"] - e_true).abs().max()) < 1e-9,
          f"err {float((fit['axis'] - e_true).abs().max()):.3e}")
    check("T19: q recovered to <1e-9", abs(float(fit["q"] - q_true)) < 1e-9,
          f"err {abs(float(fit['q'] - q_true)):.3e}")
    # Screw-axis convention: the point is only defined up to a slide along e.
    offset = fit["point"] - a_true
    slide = offset @ fit["axis"]
    perp = offset - slide * fit["axis"]
    check("T19: the fitted point lies on the true axis line (slide removed)",
          float(perp.abs().max()) < 1e-9, f"perp {float(perp.abs().max()):.3e}")
    check("T19: the returned point is the canonical one (closest to origin)",
          abs(float(fit["point"] @ fit["axis"])) < 1e-12)
    check("T19: revolute residual is zero", fit["residual"] < 1e-12,
          f"{fit['residual']:.3e}")
    check("axis scatter has a clean null direction", fit["axis_gap"] < 1e-14,
          f"{fit['axis_gap']:.3e}")
    check("axis-point system has the expected rank 2", fit["axis_rank"] == 2)
    check("not flagged degenerate", fit["degenerate"] is False)

    # T19 also asks for the Gaussian centre/orientation update: refitting and
    # re-transporting must reproduce both.
    mu_round, R_round = jr.revolute_transport(mu0, R0, fit["axis"], fit["point"], fit["q"])
    check("T19: refit transport reproduces the centres",
          torch.allclose(mu_round, mu_t, atol=1e-12),
          f"max {float((mu_round - mu_t).abs().max()):.3e}")
    check("T19: refit transport reproduces the orientations",
          torch.allclose(R_round, R_t, atol=1e-12),
          f"max {float((R_round - R_t).abs().max()):.3e}")

    # Negative angle: (e, q) -> (-e, -q) is the same motion; q >= 0 resolves it.
    mu_b, _ = jr.revolute_transport(mu0, None, e_true, a_true, -q_true)
    fit_b = jr.fit_revolute_axis(mu0, mu_b)
    check("revolute sign convention keeps q >= 0",
          float(fit_b["q"]) > 0.0 and torch.allclose(fit_b["axis"], -e_true, atol=1e-9))
    mu_b_round, _ = jr.revolute_transport(mu0, None, fit_b["axis"], fit_b["point"], fit_b["q"])
    check("the flipped revolute fit still transports onto the data",
          torch.allclose(mu_b_round, mu_b, atol=1e-12))


def test_so3_log_matrix_inverts_rotation_about_axis():
    e = torch.tensor([0.1, -0.3, 0.9], dtype=DT)
    e = e / e.norm()
    for angle in (0.0, 1e-9, 1e-3, 0.9, 3.0, math.pi - 1e-6):
        rot = jr.rotation_about_axis(e, angle)
        log = jr.so3_log_matrix(rot)
        recovered = float(log.norm())
        check(f"so3_log norm at theta={angle:.6g}", abs(recovered - angle) < 1e-9,
              f"{recovered:.12f}")
        if angle > 1e-6:
            axis = log / log.norm()
            check(f"so3_log axis at theta={angle:.6g}",
                  float((axis - e).abs().max()) < 1e-7,
                  f"{float((axis - e).abs().max()):.3e}")
    batched = jr.so3_log_matrix(torch.stack([
        jr.rotation_about_axis(e, 0.4), jr.rotation_about_axis(e, math.pi - 1e-7)]))
    check("so3_log is batched", batched.shape == (2, 3))
    check("so3_log near pi stays finite", bool(torch.isfinite(batched).all()))


def test_pose_tangents_match_finite_differences_of_the_transports():
    """The key 29.1/29.2 consistency check: G == d(transport)/dq.

    Central difference of the centres, and of the *left* rotation increment
    ``Log(R(q+h) R(q)^-1)`` -- the convention of section 4's retraction
    ``R^+ = Exp([delta phi]_x) R^-``.
    """
    n = 48
    mu0 = make_points(n, seed=91)
    R0 = random_rotations(n, seed=92)
    h = 1e-6

    def finite_difference(transport, q):
        mu_p, R_p = transport(q + h)
        mu_m, R_m = transport(q - h)
        d_mu = (mu_p - mu_m) / (2.0 * h)
        d_phi = jr.so3_log_matrix(R_p @ R_m.transpose(-1, -2)) / (2.0 * h)
        return torch.cat((d_mu, d_phi), dim=-1)

    e_p = torch.tensor([0.6, 0.2, -0.77], dtype=DT)
    e_p = e_p / e_p.norm()
    for q in (0.0, 0.83, -1.7):
        fd = finite_difference(
            lambda qq: jr.prismatic_transport(mu0, R0, e_p, qq), torch.tensor(q, dtype=DT))
        g = jr.prismatic_pose_tangent(e_p, n)[:, :, 0]
        err = float((fd - g).abs().max())
        check(f"prismatic G == d(transport)/dq at q={q}", err < 1e-9, f"err {err:.3e}")
    check("prismatic G has the form [e; 0]",
          torch.allclose(jr.prismatic_pose_tangent(e_p, n)[:, :3, 0], e_p.expand(n, 3))
          and float(jr.prismatic_pose_tangent(e_p, n)[:, 3:, 0].abs().max()) == 0.0)

    e_r = torch.tensor([-0.45, 0.61, 0.65], dtype=DT)
    e_r = e_r / e_r.norm()
    a_r = torch.tensor([0.2, 0.9, -0.4], dtype=DT)
    for q in (0.0, 0.41, -1.25, 2.9):
        qt = torch.tensor(q, dtype=DT)
        fd = finite_difference(
            lambda qq: jr.revolute_transport(mu0, R0, e_r, a_r, qq), qt)
        mu_q, _ = jr.revolute_transport(mu0, R0, e_r, a_r, qt)
        g = jr.revolute_pose_tangent(e_r, a_r, mu_q)[:, :, 0]
        err = float((fd - g).abs().max())
        check(f"revolute G == d(transport)/dq at q={q}", err < 1e-8, f"err {err:.3e}")
    mu_q, _ = jr.revolute_transport(mu0, R0, e_r, a_r, torch.tensor(0.41, dtype=DT))
    g = jr.revolute_pose_tangent(e_r, a_r, mu_q)[:, :, 0]
    check("revolute G has the form [e x (mu - a); e]",
          torch.allclose(g[:, :3], torch.linalg.cross(e_r.expand(n, 3), mu_q - a_r), atol=1e-15)
          and torch.allclose(g[:, 3:], e_r.expand(n, 3), atol=1e-15))


def test_reduced_pixel_jacobian_matches_the_boxed_formulas():
    """T18/T19 ask for H_pq: verify the boxes of 29.1 and 29.2 term by term."""
    n, p = 20, 7
    g = torch.Generator().manual_seed(123)
    H_mu = torch.randn((p, n, 3), generator=g, dtype=DT)
    H_phi = torch.randn((p, n, 3), generator=g, dtype=DT)
    H_pose = torch.cat((H_mu, H_phi), dim=-1)
    mu = make_points(n, seed=124)

    e = torch.tensor([0.2, 0.3, -0.93], dtype=DT)
    e = e / e.norm()
    hq_p = joint_filter.reduce_pixel_jacobian(H_pose, jr.prismatic_pose_tangent(e, n))
    expect_p = torch.einsum("pna,a->p", H_mu, e)[:, None]
    check("H^P_{p,q} == sum_i H^mu_{p,i} e",
          torch.allclose(hq_p, expect_p, atol=1e-13),
          f"max {float((hq_p - expect_p).abs().max()):.3e}")

    a = torch.tensor([0.1, -0.2, 0.45], dtype=DT)
    hq_r = joint_filter.reduce_pixel_jacobian(H_pose, jr.revolute_pose_tangent(e, a, mu))
    expect_r = (torch.einsum("pna,na->p", H_mu, torch.linalg.cross(e.expand(n, 3), mu - a))
                + torch.einsum("pna,a->p", H_phi, e))[:, None]
    check("H^R_{p,q} == sum_i [H^mu (e x (mu-a)) + H^phi e]",
          torch.allclose(hq_r, expect_r, atol=1e-13),
          f"max {float((hq_r - expect_r).abs().max()):.3e}")


# ---------------------------------------------------------------------------
# Section 29.3 -- reduced uncertainty
# ---------------------------------------------------------------------------
def test_reduced_information_is_psd_and_grows_with_more_gaussians():
    n = 60
    mu = make_points(n, seed=131)
    e = torch.tensor([0.0, 0.0, 1.0], dtype=DT)
    a = torch.tensor([0.1, -0.1, 0.0], dtype=DT)
    G = jr.revolute_pose_tangent(e, a, mu)

    g = torch.Generator().manual_seed(17)
    root = torch.randn((n, 6, 6), generator=g, dtype=DT)
    lam = root @ root.transpose(-1, -2) + 1e-3 * torch.eye(6, dtype=DT)  # SPD by construction

    full = jr.reduced_information(G, lam)
    check("reduced information is [m,m]", full.shape == (1, 1))
    check("reduced information is symmetric",
          torch.allclose(full, full.T, atol=0.0))
    check("reduced information is PSD",
          float(torch.linalg.eigvalsh(full).min()) >= 0.0,
          f"min eig {float(torch.linalg.eigvalsh(full).min()):.3e}")
    half = jr.reduced_information(G[: n // 2], lam[: n // 2])
    check("more Gaussians -> more information (Loewner order)",
          float(torch.linalg.eigvalsh(full - half).min()) > 0.0,
          f"{float(full):.4f} vs {float(half):.4f}")
    check("information is additive over disjoint subsets",
          torch.allclose(full, half + jr.reduced_information(G[n // 2:], lam[n // 2:]),
                         atol=1e-10))
    check("explicit sum_i G^T Lambda G matches the einsum",
          torch.allclose(full, sum(G[i].T @ lam[i] @ G[i] for i in range(n)), atol=1e-10))

    # 29.3: uncertainty must not be discarded. A Gaussian constrained only
    # orthogonally to its own tangent must contribute (almost) nothing.
    e_only = torch.zeros(6, 6, dtype=DT)
    e_only[3:, 3:] = torch.eye(3, dtype=DT) * 1e-9
    e_only[:3, :3] = torch.eye(3, dtype=DT) * 1e-9
    blind = jr.reduced_information(G, e_only.expand(n, 6, 6))
    check("a near-uninformed part yields near-zero reduced information",
          float(blind) < 1e-5, f"{float(blind):.3e}")
    check("identity information is the unweighted contraction",
          torch.allclose(jr.reduced_information(G), torch.einsum("nia,nib->ab", G, G),
                         atol=1e-12))

    cov = jr.reduced_covariance(full)
    check("reduced covariance inverts the reduced information",
          abs(float(cov @ full) - 1.0) < 1e-10, f"{float(cov @ full):.12f}")
    singular = jr.reduced_covariance(torch.zeros(1, 1, dtype=DT))
    check("a singular reduction gives zero, not inf", float(singular) == 0.0)

    # Prismatic reduction of an isotropic pose information is N * |e|^2 = N.
    prism = jr.reduced_information(jr.prismatic_pose_tangent(e, n),
                                   torch.eye(6, dtype=DT))
    check("prismatic reduction of identity information is N",
          abs(float(prism) - n) < 1e-10, f"{float(prism)}")


# ---------------------------------------------------------------------------
# Section 30 -- reduced joint-state filter
# ---------------------------------------------------------------------------
def test_joint_state_predict_uses_real_dt():
    state = jr.JointState.initial(q=0.25, qdot=1.5, sigma_q=0.1, sigma_qdot=0.2)
    same = state.predict(0.0, accel_psd=3.0)
    check("predict(dt=0) leaves q untouched", float(same.q) == float(state.q))
    check("predict(dt=0) leaves qdot untouched", float(same.qdot) == float(state.qdot))
    check("predict(dt=0) leaves the covariance untouched",
          torch.equal(same.cov, state.cov))

    # Real, non-uniform video spacing (30 fps with a dropped frame, then 24 fps).
    dts = [1.0 / 30.0, 2.0 / 30.0, 1.0 / 24.0, 1.0 / 30.0, 3.0 / 30.0]
    walker = state
    for dt in dts:
        walker = walker.predict(dt, accel_psd=0.0)
    expected_q = float(state.q) + float(state.qdot) * sum(dts)
    check("constant qdot integrates linearly in real dt",
          abs(float(walker.q) - expected_q) < 1e-14,
          f"{float(walker.q):.15f} vs {expected_q:.15f}")
    check("constant qdot is preserved by the transition",
          abs(float(walker.qdot) - float(state.qdot)) < 1e-15)
    one_shot = state.predict(sum(dts), accel_psd=0.0)
    check("the transition composes over the real intervals",
          abs(float(one_shot.q) - float(walker.q)) < 1e-14)
    check("unit frame spacing is NOT assumed",
          abs(float(state.predict(1.0, accel_psd=0.0).q) - float(walker.q)) > 1e-3)

    noisy = state.predict(1.0 / 30.0, accel_psd=2.5)
    check("process noise inflates the covariance",
          float(noisy.cov[0, 0]) > float(state.cov[0, 0])
          and float(noisy.cov[1, 1]) > float(state.cov[1, 1]))
    check("predicted covariance stays PSD",
          float(torch.linalg.eigvalsh(noisy.cov).min()) > 0.0)
    dt = 1.0 / 30.0
    q_expect = 2.5 * torch.tensor([[dt ** 3 / 3.0, dt ** 2 / 2.0], [dt ** 2 / 2.0, dt]],
                                  dtype=DT)
    F = torch.tensor([[1.0, dt], [0.0, 1.0]], dtype=DT)
    check("covariance follows F P F^T + Q exactly",
          torch.allclose(noisy.cov, F @ state.cov @ F.T + q_expect, atol=1e-15))
    try:
        state.predict(-0.01)
        ok = False
    except ValueError:
        ok = True
    check("negative dt is rejected", ok)


def test_joint_state_update_solves_the_scalar_normal_system():
    prior = jr.JointState.initial(q=0.0, qdot=0.0, sigma_q=10.0, sigma_qdot=10.0)
    # Synthetic reduced measurement: P pixels, H_pq from a real pose tangent.
    n, p = 30, 50
    mu = make_points(n, seed=141)
    e = torch.tensor([0.0, 1.0, 0.0], dtype=DT)
    a = torch.tensor([0.0, 0.0, 0.3], dtype=DT)
    G = jr.revolute_pose_tangent(e, a, mu)
    g = torch.Generator().manual_seed(31)
    H_pose = torch.randn((p, n, 6), generator=g, dtype=DT)
    H_q = joint_filter.reduce_pixel_jacobian(H_pose, G)              # [P,1]
    r_inv = torch.full((p,), 1.0 / 0.01 ** 2, dtype=DT)              # 1/sigma_pixel^2

    q_true = torch.tensor(0.37, dtype=DT)
    resid = (H_q[:, 0] * q_true)                                     # y - h(0) for linear h
    info = (H_q[:, 0] * r_inv * H_q[:, 0]).sum().reshape(1, 1)
    b = (H_q[:, 0] * r_inv * resid).sum().reshape(1)

    posterior = prior.update(info, b)
    check("section 30 update recovers q from the reduced normal system",
          abs(float(posterior.q) - float(q_true)) < 1e-6,
          f"{float(posterior.q):.9f} vs {float(q_true):.9f}")
    check("a position-only measurement does not move qdot",
          abs(float(posterior.qdot)) < 1e-12)
    check("the posterior covariance shrinks",
          float(posterior.cov[0, 0]) < float(prior.cov[0, 0]))
    expected_cov = 1.0 / (1.0 / prior.cov[0, 0] + info[0, 0])
    check("P^+ == (P^-1 + sum H^T R^-1 H)^-1 on the q block",
          abs(float(posterior.cov[0, 0]) - float(expected_cov)) < 1e-15,
          f"{float(posterior.cov[0, 0]):.12e}")
    check("the untouched qdot block keeps its prior variance",
          abs(float(posterior.cov[1, 1]) - float(prior.cov[1, 1])) < 1e-12)

    delta, total = jr.solve_reduced_normal(info, b)
    check("solve_reduced_normal without a prior gives the plain LS solution",
          abs(float(delta[0]) - float(q_true)) < 1e-9, f"{float(delta[0]):.9f}")
    check("the total information is the measurement information when no prior",
          torch.allclose(total, info, atol=0.0))

    # Rule 8: with the prior held fixed, re-linearising at the new point must
    # return the *same* posterior, not one that counted the same frame twice.
    resid2 = resid - H_q[:, 0] * posterior.q
    b2 = (H_q[:, 0] * r_inv * resid2).sum().reshape(1)
    second = posterior.update(info, b2, prior=prior)
    check("a fixed-prior inner iteration is a no-op at the optimum",
          abs(float(second.q) - float(posterior.q)) < 1e-12,
          f"{float(second.q):.12f} vs {float(posterior.q):.12f}")
    check("a fixed-prior inner iteration does not shrink the covariance again",
          abs(float(second.cov[0, 0]) - float(posterior.cov[0, 0])) < 1e-18)

    # Two independent frames must add information.
    third = posterior.update(info, torch.zeros(1, dtype=DT))
    check("a second frame adds information",
          float(third.cov[0, 0]) < float(posterior.cov[0, 0]))

    scalar_state = prior.update(float(info[0, 0]), float(b[0]))
    check("scalar information/b arguments are accepted",
          abs(float(scalar_state.q) - float(posterior.q)) < 1e-12)
    try:
        prior.update(torch.eye(3, dtype=DT), torch.zeros(3, dtype=DT))
        ok = False
    except ValueError:
        ok = True
    check("a mis-shaped reduced system is rejected", ok)


def test_joint_model_drives_the_whole_part_from_the_reduced_state():
    """Section 30: after reduction no Gaussian moves except through the joint."""
    n = 64
    mu0 = make_points(n, seed=151)
    R0 = random_rotations(n, seed=152)
    e = torch.tensor([0.0, 0.0, 1.0], dtype=DT)
    a = torch.tensor([0.4, -0.2, 0.0], dtype=DT)
    q_true = torch.tensor(0.5, dtype=DT)
    mu_t, R_t = jr.revolute_transport(mu0, R0, e, a, q_true)

    model, fit = jr.JointModel.fit("revolute", mu0, mu_t)
    check("JointModel.fit identifies the revolute geometry",
          model.joint_type == "revolute"
          and float((model.axis - e).abs().max()) < 1e-9)
    mu_m, R_m = model.transport(mu0, R0, fit["q"])
    check("JointModel.transport reproduces the part pose",
          torch.allclose(mu_m, mu_t, atol=1e-12) and torch.allclose(R_m, R_t, atol=1e-12))
    check("JointModel.pose_tangent == revolute_pose_tangent",
          torch.allclose(model.pose_tangent(mu_m),
                         jr.revolute_pose_tangent(model.axis, model.point, mu_m), atol=0.0))

    pmodel, pfit = jr.JointModel.fit(
        "prismatic", mu0, jr.prismatic_transport(mu0, None, e, 0.25)[0])
    check("JointModel.fit identifies the prismatic geometry",
          abs(float(pfit["q"]) - 0.25) < 1e-12
          and float((pmodel.axis - e).abs().max()) < 1e-12)
    check("prismatic JointModel.pose_tangent is the constant [e;0]",
          torch.allclose(pmodel.pose_tangent(mu0), jr.prismatic_pose_tangent(e, n), atol=0.0))
    try:
        jr.JointModel("revolute", e)
        ok = False
    except ValueError:
        ok = True
    check("a revolute joint without an axis point is rejected", ok)
    try:
        jr.JointModel("screw", e)
        ok = False
    except ValueError:
        ok = True
    check("rule 13: screw/helical joints are refused", ok)

    # Reduced filter end to end on a real 30 fps clip: predict with real dt,
    # update from the reduced normal system, recover a moving joint.
    dt = 1.0 / 30.0
    qdot_true = 1.2
    state = jr.JointState.initial(q=0.0, qdot=0.0, sigma_q=1.0, sigma_qdot=1.0)
    lam = 1e4 * torch.eye(6, dtype=DT)
    for k in range(1, 9):
        state = state.predict(dt, accel_psd=1e-2)
        q_k = torch.tensor(qdot_true * k * dt, dtype=DT)
        mu_k, _ = model.transport(mu0, None, q_k)
        G = model.pose_tangent(mu_k)
        info = jr.reduced_information(G, lam)
        # Linearised innovation of the reduced measurement: G maps dq onto pose,
        # Lambda whitens it -- exactly section 30 with H = G and R^-1 = Lambda.
        err = (q_k - state.q)
        b = info @ err.reshape(1)
        state = state.update(info, b)
    check("the reduced filter tracks the true joint angle",
          abs(float(state.q) - qdot_true * 8 * dt) < 1e-6,
          f"{float(state.q):.9f} vs {qdot_true * 8 * dt:.9f}")
    check("the reduced filter picks up the joint speed",
          abs(float(state.qdot) - qdot_true) < 0.2, f"{float(state.qdot):.6f}")
    check("the reduced state stays a 2-vector with a 2x2 covariance",
          state.vector.shape == (2,) and state.cov.shape == (2, 2))


# ---------------------------------------------------------------------------
# Sections 0 and 36 -- execution contract
# ---------------------------------------------------------------------------
def test_package_obeys_the_autograd_and_no_stub_contract():
    package = Path(jr.__file__).parent
    # Scanned on the AST, not the text: the prose explaining *why* there is no
    # autograd here would otherwise trip a plain substring search.
    banned = {"backward", "autograd", "requires_grad", "grad", "Adam", "optim",
              "retain_graph"}
    for path in sorted(package.glob("*.py")):
        src = path.read_text(encoding="utf-8")
        check(f"{path.name} has no TODO/FIXME",
              "TODO" not in src and "FIXME" not in src)
        tree = ast.parse(src)
        used = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Attribute):
                used.add(node.attr)
            elif isinstance(node, ast.Name):
                used.add(node.id)
            elif isinstance(node, ast.Import):
                used.update(alias.name.split(".")[-1] for alias in node.names)
            elif isinstance(node, ast.ImportFrom):
                used.update(alias.name for alias in node.names)
                used.update((node.module or "").split("."))
        check(f"{path.name} uses no autograd/optimiser symbol",
              not (used & banned), str(sorted(used & banned)))
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                body = [s for s in node.body if not isinstance(s, ast.Expr)
                        or not isinstance(s.value, ast.Constant)]
                check(f"{path.name}:{node.name} is not a stub",
                      bool(body) and not all(isinstance(s, ast.Pass) for s in body))
        imports = {n.module for n in ast.walk(tree) if isinstance(n, ast.ImportFrom)}
        check(f"{path.name} does not import estimator/",
              not any(m and m.startswith("estimator") for m in imports))


def test_dtype_and_device_are_preserved():
    mu = make_points(40, seed=161).to(torch.float32)
    v = rigid_velocity(mu, torch.tensor([0.1, 0.2, 0.0]), torch.tensor([0.0, 0.3, 0.1]),
                       mu.mean(dim=0))
    fit = jr.fit_rigid_twist(mu, v)
    check("float32 input keeps float32 output", fit["twist"].dtype == torch.float32)
    check("float32 twist is still accurate",
          float((fit["omega"] - torch.tensor([0.0, 0.3, 0.1])).abs().max()) < 1e-5,
          f"{float((fit['omega'] - torch.tensor([0.0, 0.3, 0.1])).abs().max()):.3e}")
    e = torch.tensor([1.0, 0.0, 0.0], dtype=torch.float32)
    g = jr.prismatic_pose_tangent(e, 40)
    check("tangents inherit the axis dtype", g.dtype == torch.float32)
    mu32, _ = jr.revolute_transport(mu, None, e, torch.zeros(3), 0.3)
    check("transport keeps the input dtype", mu32.dtype == torch.float32)
    state = jr.JointState.initial(dtype=torch.float32)
    check("the reduced filter honours its dtype",
          state.predict(0.04, accel_psd=1.0).cov.dtype == torch.float32)
    check("float32 single-precision fit still reports conditioning",
          fit["rank"] == 6 and math.isfinite(fit["condition_number_scaled"]))

    if not torch.cuda.is_available():
        check("cuda unavailable -- device preservation exercised on cpu only", True)
        return
    dev = torch.device("cuda")
    mu64 = make_points(64, seed=162).to(dev)
    axis = torch.tensor([0.0, 0.0, 1.0], dtype=DT, device=dev)
    point = torch.tensor([0.1, 0.2, 0.0], dtype=DT, device=dev)
    mu_t, _ = jr.revolute_transport(mu64, None, axis, point, 0.3)
    rev = jr.fit_revolute_axis(mu64, mu_t)
    check("cuda: revolute fit stays on the device and is exact",
          rev["axis"].device.type == "cuda" and rev["residual"] < 1e-12
          and float((rev["axis"] - axis).abs().max()) < 1e-12)
    twist = jr.fit_rigid_twist(mu64, (mu_t - mu64) / 0.1, robust="tukey")
    check("cuda: robust twist fit runs and stays on the device",
          twist["twist"].device.type == "cuda" and twist["rank"] == 6)
    info = jr.reduced_information(jr.revolute_pose_tangent(axis, point, mu_t),
                                  torch.eye(6, dtype=DT, device=dev))
    state = jr.JointState.initial(dtype=DT, device=dev).predict(1.0 / 30.0, accel_psd=1.0)
    state = state.update(info, torch.ones(1, dtype=DT, device=dev))
    check("cuda: the reduced filter stays on the device",
          info.device.type == "cuda" and state.cov.device.type == "cuda"
          and bool(torch.isfinite(state.vector).all()))


def main() -> None:
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for fn in tests:
        print(f"\n{fn.__name__}")
        fn()
    print(f"\n{len(PASSED)} checks passed across {len(tests)} tests")


if __name__ == "__main__":
    main()
