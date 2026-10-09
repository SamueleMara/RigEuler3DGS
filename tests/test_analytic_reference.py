"""Acceptance tests T1-T5, T7 and the adjoint identity for the CPU analytic reference.

    PYTHONPATH=. python tests/test_analytic_reference.py

Every derivative in ``estimator/analytic_reference.py`` is a hand-written closed
form, so the only honest oracle is a central finite difference of the *same*
nonlinear chain.  Everything runs in float64 on CPU, where a central difference
has a visible truncation/round-off U-shape: each test sweeps the step size and
asserts on the best step, which is what spec 35 (T6) asks for.

Autograd appears nowhere, not even here -- finite differences are enough and they
also check the forward model, which autograd would not.
"""

import math

import torch

from estimator.analytic_reference import (
    ALPHA_CLAMP,
    LOW_PASS_VARIANCE,
    alpha_pose_jacobian,
    composite_forward,
    composite_jvp,
    composite_vjp_alpha_sensitivities,
    dJ_pi,
    dS_and_dQ,
    dSigma_world,
    dSigma_world_basis,
    dalpha,
    depth_jacobian,
    du_from_dmu,
    extend_block_to_12d,
    gaussian_alpha,
    invert_sym2,
    measurement_jacobian_blocks,
    pose_jacobian_block,
    project,
    project_gaussian,
    projected_conic,
    rotate_covariance,
    skew,
)

DTYPE = torch.float64
# Steps bracketing the float64 central-difference optimum (~h^2 + eps_mach/h).
STEPS = [1e-2, 1e-3, 1e-4, 1e-5, 1e-6]


# --------------------------------------------------------------------------
# helpers: SO(3) exponential and a sweeping central-difference oracle
# --------------------------------------------------------------------------


def _so3_exp(phi):
    """Rodrigues formula; only used to build finite-difference perturbations."""
    theta = torch.linalg.norm(phi)
    eye = torch.eye(3, dtype=phi.dtype)
    if float(theta) < 1e-12:
        return eye + skew(phi)
    axis = phi / theta
    K = skew(axis)
    return eye + torch.sin(theta) * K + (1.0 - torch.cos(theta)) * (K @ K)


def _central(fn, step):
    """(f(+h) - f(-h)) / 2h for any tensor-valued f of a scalar offset."""
    return (fn(step) - fn(-step)) / (2.0 * step)


def _sweep_error(fn, analytic, steps=STEPS):
    """Best relative error over the step sweep, plus the whole sweep for reporting."""
    scale = max(float(analytic.abs().max()), 1e-12)
    errors = []
    for step in steps:
        numeric = _central(fn, step)
        errors.append(float((numeric - analytic).abs().max()) / scale)
    return min(errors), errors


def _random_camera(generator):
    """A camera looking roughly down +z with a non-trivial rotation."""
    phi = torch.randn(3, generator=generator, dtype=DTYPE) * 0.3
    R_cw = _so3_exp(phi)
    t_cw = torch.tensor([0.1, -0.2, 4.0], dtype=DTYPE)
    fx, fy, cx, cy = 512.0, 498.0, 320.0, 240.0
    return R_cw, t_cw, fx, fy, cx, cy


def _world_from_camera(mu_c, R_cw, t_cw):
    """Inverse of mu_c = R_cw mu + t_cw, so tests can place depth layers directly."""
    return (mu_c - t_cw) @ R_cw


def _random_gaussian(generator, scale=0.08):
    """World center plus an anisotropic SPD covariance R diag(s^2) R^T."""
    mu = torch.randn(3, generator=generator, dtype=DTYPE) * 0.4
    axes = _so3_exp(torch.randn(3, generator=generator, dtype=DTYPE))
    sigmas = scale * (0.4 + torch.rand(3, generator=generator, dtype=DTYPE))
    Sigma = axes @ torch.diag(sigmas ** 2) @ axes.T
    return mu, Sigma


# --------------------------------------------------------------------------
# spec 9.1 / T1
# --------------------------------------------------------------------------


def test_T1_projection_differential_matches_central_differences():
    """du = J_pi R_cw dmu, swept over the step size, to ~1e-9 in float64."""
    generator = torch.Generator().manual_seed(1)
    for _ in range(8):
        R_cw, t_cw, fx, fy, cx, cy = _random_camera(generator)
        mu, _ = _random_gaussian(generator)
        dmu = torch.randn(3, generator=generator, dtype=DTYPE)

        mu_c, _, J_pi, depth = project(mu, R_cw, t_cw, fx, fy, cx, cy)
        assert float(depth) > 0.5, "test scene must stay in front of the camera"
        analytic = du_from_dmu(J_pi, R_cw @ dmu)

        def u_of(step):
            return project(mu + step * dmu, R_cw, t_cw, fx, fy, cx, cy)[1]

        best, sweep = _sweep_error(u_of, analytic)
        assert best < 1e-9, f"du relative error sweep {sweep}"

        # The sweep must actually be a U: too large a step is truncation-limited.
        assert sweep[0] > best, f"no truncation branch in the sweep {sweep}"

        # Rotation about the Gaussian's own center cannot move u (spec 9.1): the
        # whole rotation dependence of the chain sits in Sigma, not in mu.
        _, Sigma = _random_gaussian(generator)
        dphi = torch.randn(3, generator=generator, dtype=DTYPE)
        R_g = _so3_exp(0.3 * dphi)
        rotated = project_gaussian(mu, R_g @ Sigma @ R_g.T, R_cw, t_cw, fx, fy, cx, cy)
        plain = project_gaussian(mu, Sigma, R_cw, t_cw, fx, fy, cx, cy)
        assert torch.allclose(rotated["u"], plain["u"], atol=0.0)
        assert not torch.allclose(rotated["Q"], plain["Q"], atol=1e-6)
        assert mu_c.shape == (3,)


# --------------------------------------------------------------------------
# spec 9.2 / T2
# --------------------------------------------------------------------------


def test_T2_perspective_jacobian_differential_matches_central_differences():
    """dJ_pi is the term that is usually dropped; it must match FD exactly."""
    generator = torch.Generator().manual_seed(2)
    for _ in range(8):
        R_cw, t_cw, fx, fy, cx, cy = _random_camera(generator)
        mu, _ = _random_gaussian(generator)
        dmu = torch.randn(3, generator=generator, dtype=DTYPE)
        dmu_c = R_cw @ dmu

        mu_c, _, _, _ = project(mu, R_cw, t_cw, fx, fy, cx, cy)
        analytic = dJ_pi(mu_c, dmu_c, fx, fy)
        assert float(analytic.abs().max()) > 1e-6, "degenerate test direction"

        def J_of(step):
            return project(mu + step * dmu, R_cw, t_cw, fx, fy, cx, cy)[2]

        best, sweep = _sweep_error(J_of, analytic)
        assert best < 1e-9, f"dJ_pi relative error sweep {sweep}"

        # The two structurally zero entries must stay exactly zero.
        assert float(analytic[0, 1]) == 0.0 and float(analytic[1, 0]) == 0.0


# --------------------------------------------------------------------------
# spec 9.3 / T3
# --------------------------------------------------------------------------


def test_T3_covariance_rotation_differential_matches_central_differences():
    """dSigma = [dphi]_x Sigma - Sigma [dphi]_x under R -> Exp([dphi]_x) R."""
    generator = torch.Generator().manual_seed(3)
    for _ in range(8):
        axes = _so3_exp(torch.randn(3, generator=generator, dtype=DTYPE))
        Lambda = torch.diag(0.05 + torch.rand(3, generator=generator, dtype=DTYPE))
        Sigma = axes @ Lambda @ axes.T
        dphi = torch.randn(3, generator=generator, dtype=DTYPE)

        analytic = dSigma_world(Sigma, dphi)
        assert torch.allclose(analytic, analytic.T, atol=1e-14), "dSigma must be symmetric"

        def Sigma_of(step):
            R = _so3_exp(step * dphi) @ axes
            return R @ Lambda @ R.T

        best, sweep = _sweep_error(Sigma_of, analytic)
        assert best < 1e-9, f"dSigma relative error sweep {sweep}"

        # The three basis derivatives must reassemble the directional one.
        basis = dSigma_world_basis(Sigma)
        assert basis.shape == (3, 3, 3)
        recombined = (dphi.view(3, 1, 1) * basis).sum(dim=0)
        assert torch.allclose(recombined, analytic, atol=1e-12)

        # An isotropic Lambda is rotation-blind (spec 27, near-isotropic case).
        iso = 0.25 * torch.eye(3, dtype=DTYPE)
        assert float(dSigma_world(iso, dphi).abs().max()) < 1e-15


# --------------------------------------------------------------------------
# spec 9.4 / T4
# --------------------------------------------------------------------------


def test_T4_projected_covariance_and_conic_differential():
    """dS and dQ = -Q dS Q, with the renderer's 0.3 low-pass term present."""
    generator = torch.Generator().manual_seed(4)
    for _ in range(8):
        R_cw, t_cw, fx, fy, cx, cy = _random_camera(generator)
        mu, Sigma = _random_gaussian(generator)
        dmu = torch.randn(3, generator=generator, dtype=DTYPE) * 0.5
        dphi = torch.randn(3, generator=generator, dtype=DTYPE) * 0.5

        cache = project_gaussian(mu, Sigma, R_cw, t_cw, fx, fy, cx, cy)
        S, Q = cache["S"], cache["Q"]
        eye2 = torch.eye(2, dtype=DTYPE)
        assert torch.allclose(S @ Q, eye2, atol=1e-10), "Q must invert the low-passed S"
        raw = cache["J_pi"] @ cache["Sigma_c"] @ cache["J_pi"].T
        assert torch.allclose(S - raw, LOW_PASS_VARIANCE * eye2, atol=1e-12), (
            "the 0.3 screen-space low-pass term of forward.cu is missing from S"
        )

        dmu_c = R_cw @ dmu
        dSigma_c = rotate_covariance(R_cw, dSigma_world(Sigma, dphi))
        dS_a, dQ_a = dS_and_dQ(
            cache["J_pi"], dJ_pi(cache["mu_c"], dmu_c, fx, fy), cache["Sigma_c"], dSigma_c, Q
        )

        def perturbed(step):
            R_g = _so3_exp(step * dphi)
            return project_gaussian(
                mu + step * dmu, R_g @ Sigma @ R_g.T, R_cw, t_cw, fx, fy, cx, cy
            )

        best_S, sweep_S = _sweep_error(lambda h: perturbed(h)["S"], dS_a)
        best_Q, sweep_Q = _sweep_error(lambda h: perturbed(h)["Q"], dQ_a)
        assert best_S < 1e-8, f"dS relative error sweep {sweep_S}"
        assert best_Q < 1e-8, f"dQ relative error sweep {sweep_Q}"

        # dQ = -Q dS Q is the identity that makes the alpha differential cheap.
        assert torch.allclose(dQ_a, -(Q @ dS_a @ Q), atol=1e-12)
        # The 2x2 closed-form inverse must agree with a general solver.
        assert torch.allclose(invert_sym2(S), torch.linalg.inv(S), atol=1e-12)


# --------------------------------------------------------------------------
# spec 9.5 / T5
# --------------------------------------------------------------------------


def _alpha_at(mu, Sigma, opacity, pixel, R_cw, t_cw, intr, clamp=True):
    fx, fy, cx, cy = intr
    cache = project_gaussian(mu, Sigma, R_cw, t_cw, fx, fy, cx, cy)
    return gaussian_alpha(opacity, pixel, cache["u"], cache["Q"], clamp=clamp)


def test_T5_alpha_differential_for_translation_and_rotation():
    """dalpha = alpha[e^T Q du - 0.5 e^T dQ e] for both tangent halves."""
    generator = torch.Generator().manual_seed(5)
    checked = 0
    for _ in range(12):
        R_cw, t_cw, fx, fy, cx, cy = _random_camera(generator)
        intr = (fx, fy, cx, cy)
        # Large, strongly anisotropic Gaussians make the dQ term non-negligible.
        mu, Sigma = _random_gaussian(generator, scale=0.25)
        opacity = torch.tensor(0.55, dtype=DTYPE)

        cache = project_gaussian(mu, Sigma, R_cw, t_cw, fx, fy, cx, cy)
        # Sample a pixel ~1 sigma off the projected center so both the du and the
        # dQ term are active (at e = 0 the alpha differential vanishes exactly).
        offset = torch.randn(2, generator=generator, dtype=DTYPE)
        pixel = cache["u"] + torch.linalg.cholesky(cache["S"]) @ offset
        state = gaussian_alpha(opacity, pixel, cache["u"], cache["Q"])
        if float(state["alpha"]) < 1e-3 or bool(state["clamped"]):
            continue
        checked += 1

        for translating in (True, False):
            dmu = torch.randn(3, generator=generator, dtype=DTYPE) * 0.3
            dphi = torch.randn(3, generator=generator, dtype=DTYPE) * 0.3
            if translating:
                dphi = torch.zeros(3, dtype=DTYPE)
            else:
                dmu = torch.zeros(3, dtype=DTYPE)

            dmu_c = R_cw @ dmu
            du = du_from_dmu(cache["J_pi"], dmu_c)
            dSigma_c = rotate_covariance(R_cw, dSigma_world(Sigma, dphi))
            _, dQ = dS_and_dQ(
                cache["J_pi"],
                dJ_pi(cache["mu_c"], dmu_c, fx, fy),
                cache["Sigma_c"],
                dSigma_c,
                cache["Q"],
            )
            analytic = dalpha(state["alpha"], state["e"], cache["Q"], du, dQ)

            def alpha_of(step):
                R_g = _so3_exp(step * dphi)
                return _alpha_at(
                    mu + step * dmu, R_g @ Sigma @ R_g.T, opacity, pixel, R_cw, t_cw, intr
                )["alpha"]

            best, sweep = _sweep_error(alpha_of, analytic.reshape(1))
            label = "translation" if translating else "rotation"
            assert best < 1e-8, f"dalpha ({label}) relative error sweep {sweep}"

        # The 6D Jacobian assembled from the basis must reproduce both directions.
        J = alpha_pose_jacobian(mu, Sigma, opacity, pixel, R_cw, t_cw, fx, fy, cx, cy)
        assert J["J_alpha"].shape == (6,)
        dmu = torch.randn(3, generator=generator, dtype=DTYPE) * 0.2
        dphi = torch.randn(3, generator=generator, dtype=DTYPE) * 0.2
        tangent = torch.cat((dmu, dphi))

        def alpha_of_both(step):
            R_g = _so3_exp(step * dphi)
            return _alpha_at(
                mu + step * dmu, R_g @ Sigma @ R_g.T, opacity, pixel, R_cw, t_cw, intr
            )["alpha"]

        best, sweep = _sweep_error(alpha_of_both, (J["J_alpha"] @ tangent).reshape(1))
        assert best < 1e-8, f"J_alpha . v relative error sweep {sweep}"

    assert checked >= 6, f"only {checked} usable (pixel, Gaussian) samples"


def test_alpha_clamp_has_exactly_zero_derivative():
    """Above the renderer's min(0.99, .) the measurement is constant, so H is 0."""
    generator = torch.Generator().manual_seed(6)
    R_cw, t_cw, fx, fy, cx, cy = _random_camera(generator)
    mu, Sigma = _random_gaussian(generator, scale=0.3)
    # An opacity near 1 at the projected center saturates the clamp.
    opacity = torch.tensor(0.999, dtype=DTYPE)
    cache = project_gaussian(mu, Sigma, R_cw, t_cw, fx, fy, cx, cy)
    # Sit just off the peak: e != 0 (so the unclamped differential is nonzero --
    # at e = 0 alpha is stationary and the test would be vacuous) but still deep
    # inside the region where o exp(-q/2) > 0.99.
    offset = math.sqrt(0.008 / float(cache["Q"][0, 0]))
    pixel = cache["u"] + torch.tensor([offset, 0.0], dtype=DTYPE)

    state = gaussian_alpha(opacity, pixel, cache["u"], cache["Q"])
    assert bool(state["clamped"]), "test setup failed to saturate alpha"
    assert float(state["alpha"]) == ALPHA_CLAMP
    assert float(state["alpha_raw"]) > ALPHA_CLAMP

    clamped_J = alpha_pose_jacobian(
        mu, Sigma, opacity, pixel, R_cw, t_cw, fx, fy, cx, cy, clamp=True
    )["J_alpha"]
    assert float(clamped_J.abs().max()) == 0.0, "clamped alpha must give a zero Jacobian"

    # Without the clamp the same configuration has a genuinely nonzero Jacobian
    # along the depth direction, so the zero above is the clamp, not a bug.
    free_J = alpha_pose_jacobian(
        mu, Sigma, opacity, pixel, R_cw, t_cw, fx, fy, cx, cy, clamp=False
    )["J_alpha"]
    assert float(free_J.abs().max()) > 1e-6, free_J

    # Finite differences of the clamped chain are zero on both sides of the step.
    def alpha_of(step):
        return _alpha_at(
            mu + step * torch.tensor([0.0, 0.0, 1e-3], dtype=DTYPE),
            Sigma, opacity, pixel, R_cw, t_cw, (fx, fy, cx, cy),
        )["alpha"].reshape(1)

    assert float(_central(alpha_of, 1e-4).abs().max()) == 0.0


# --------------------------------------------------------------------------
# spec 8 / 10 / T7
# --------------------------------------------------------------------------


def _layered_pixel(generator, m=4, front_alpha=None):
    alphas = 0.15 + 0.6 * torch.rand(m, generator=generator, dtype=DTYPE)
    if front_alpha is not None:
        alphas[0] = front_alpha
    colors = torch.rand(m, 3, generator=generator, dtype=DTYPE)
    depths = torch.cumsum(0.5 + torch.rand(m, generator=generator, dtype=DTYPE), dim=0)
    c_bg = torch.rand(3, generator=generator, dtype=DTYPE)
    return alphas, colors, depths, c_bg


def test_composite_forward_matches_an_explicit_loop():
    """The cumprod prefix transmittance must equal the literal recurrence of spec 8."""
    generator = torch.Generator().manual_seed(7)
    alphas, colors, depths, c_bg = _layered_pixel(generator, m=5)
    out = composite_forward(alphas, colors, depths, c_bg, eps_D=1e-6)

    T = 1.0
    C = torch.zeros(3, dtype=DTYPE)
    A = torch.zeros((), dtype=DTYPE)
    Z = torch.zeros((), dtype=DTYPE)
    for i in range(alphas.shape[0]):
        assert abs(float(out["T"][i]) - T) < 1e-14
        w = T * float(alphas[i])
        C = C + w * colors[i]
        A = A + w
        Z = Z + w * depths[i]
        T = T * (1.0 - float(alphas[i]))
    C = C + T * c_bg
    assert torch.allclose(out["C"], C, atol=1e-14)
    assert abs(float(out["A"]) - float(A)) < 1e-14
    assert abs(float(out["Z"]) - float(Z)) < 1e-14
    assert abs(float(out["T_bg"]) - T) < 1e-14
    assert torch.allclose(out["D"], Z / (A + 1e-6), atol=1e-14)


def test_T7_compositor_jvp_matches_central_differences_with_occlusion():
    """Four depth layers: the tangent must carry front/rear visibility coupling."""
    generator = torch.Generator().manual_seed(8)
    eps_D = 1e-6
    for _ in range(6):
        alphas, colors, depths, c_bg = _layered_pixel(generator, m=4)
        dalphas = torch.randn(4, generator=generator, dtype=DTYPE)
        ddepths = torch.randn(4, generator=generator, dtype=DTYPE)

        jvp = composite_jvp(alphas, dalphas, colors, depths, ddepths, c_bg, eps_D)
        base = composite_forward(alphas, colors, depths, c_bg, eps_D)
        for key in ("C", "A", "Z", "D"):
            assert torch.allclose(jvp[key], base[key], atol=1e-14), key

        def forward_of(step):
            out = composite_forward(
                alphas + step * dalphas, colors, depths + step * ddepths, c_bg, eps_D
            )
            return torch.cat((out["C"], out["A"].reshape(1), out["Z"].reshape(1), out["D"].reshape(1)))

        analytic = torch.cat(
            (jvp["dC"], jvp["dA"].reshape(1), jvp["dZ"].reshape(1), jvp["dD"].reshape(1))
        )
        best, sweep = _sweep_error(forward_of, analytic)
        assert best < 1e-9, f"compositor JVP relative error sweep {sweep}"

    # Occlusion coupling: perturbing ONLY the front alpha must move the output,
    # and must do so by more than the naive "no dT" answer would predict.
    alphas = torch.tensor([0.4, 0.5, 0.6], dtype=DTYPE)
    colors = torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]], dtype=DTYPE)
    depths = torch.tensor([1.0, 2.0, 3.0], dtype=DTYPE)
    c_bg = torch.zeros(3, dtype=DTYPE)
    front_only = torch.tensor([1.0, 0.0, 0.0], dtype=DTYPE)
    jvp = composite_jvp(alphas, front_only, colors, depths, None, c_bg, 1e-6)
    # Channels 1 and 2 belong to the two *rear* layers; they can only respond
    # through dT, so a compositor that ignored dT would return exactly zero.
    assert float(jvp["dC"][1].abs()) > 1e-3, jvp["dC"]
    assert float(jvp["dC"][2].abs()) > 1e-3, jvp["dC"]
    assert float(jvp["dC"][1]) < 0.0 and float(jvp["dC"][2]) < 0.0, (
        "occluding more must dim the layers behind"
    )


# --------------------------------------------------------------------------
# spec 11 / T8 (adjoint identity, CPU half)
# --------------------------------------------------------------------------


def test_vjp_sensitivities_reproduce_the_jvp_by_the_adjoint_identity():
    """sum_i (dX/dalpha_i) dalpha_i == dX from the tangent compositor, for any dalpha."""
    generator = torch.Generator().manual_seed(9)
    eps_D = 1e-6
    for _ in range(8):
        m = int(torch.randint(2, 7, (1,), generator=generator))
        alphas, colors, depths, c_bg = _layered_pixel(generator, m=m)
        dalphas = torch.randn(m, generator=generator, dtype=DTYPE)
        ddepths = torch.randn(m, generator=generator, dtype=DTYPE)

        sens = composite_vjp_alpha_sensitivities(alphas, colors, depths, c_bg, eps_D)
        jvp = composite_jvp(alphas, dalphas, colors, depths, ddepths, c_bg, eps_D)

        dC = (sens["dC_dalpha"] * dalphas.unsqueeze(-1)).sum(dim=0)
        dA = (sens["dA_dalpha"] * dalphas).sum()
        dZ = (sens["dZ_dalpha"] * dalphas).sum() + (sens["dZ_dd"] * ddepths).sum()
        dD = (sens["dD_dalpha"] * dalphas).sum() + (sens["dD_dd"] * ddepths).sum()

        assert torch.allclose(dC, jvp["dC"], atol=1e-12), (dC, jvp["dC"])
        assert torch.allclose(dA, jvp["dA"], atol=1e-12)
        assert torch.allclose(dZ, jvp["dZ"], atol=1e-12)
        assert torch.allclose(dD, jvp["dD"], atol=1e-12)

        # The behind-suffix of the last contributor is the background itself.
        assert torch.allclose(sens["B"][-1], c_bg, atol=1e-14)
        assert float(sens["Ab"][-1]) == 0.0 and float(sens["Zb"][-1]) == 0.0

        # Full adjoint pairing <Hv, z> == <v, H^T z> for a random pixel covector.
        z = torch.randn(4, generator=generator, dtype=DTYPE)
        Hv = torch.cat((jvp["dC"], jvp["dD"].reshape(1)))
        HTz_alpha = sens["dC_dalpha"] @ z[:3] + sens["dD_dalpha"] * z[3]
        HTz_depth = sens["dZ_dd"] * 0.0 + sens["dD_dd"] * z[3]
        pairing = float(HTz_alpha @ dalphas + HTz_depth @ ddepths)
        assert abs(pairing - float(Hv @ z)) < 1e-11, (pairing, float(Hv @ z))


def test_suffixes_are_built_by_reverse_traversal_not_by_dividing_out_alpha():
    """alpha = 1 on the front layer kills T behind it; the reference must stay finite."""
    alphas = torch.tensor([1.0, 0.5, 0.3], dtype=DTYPE)
    colors = torch.tensor([[0.2, 0.4, 0.6], [0.9, 0.1, 0.1], [0.3, 0.3, 0.9]], dtype=DTYPE)
    depths = torch.tensor([1.5, 2.5, 3.5], dtype=DTYPE)
    c_bg = torch.tensor([0.05, 0.05, 0.05], dtype=DTYPE)

    sens = composite_vjp_alpha_sensitivities(alphas, colors, depths, c_bg, 1e-6)
    for key, value in sens.items():
        assert torch.isfinite(value).all(), f"{key} is not finite: {value}"
    # Everything behind an opaque front layer is invisible.
    assert float(sens["T"][1]) == 0.0 and float(sens["T"][2]) == 0.0
    assert float(sens["dC_dalpha"][1:].abs().max()) == 0.0
    assert torch.allclose(sens["C"], colors[0], atol=1e-14)
    # The suffixes behind the opaque layer are still well defined and nonzero.
    assert float(sens["Ab"][0]) > 0.0 and float(sens["Zb"][0]) > 0.0


def test_saturated_front_alpha_stays_finite_end_to_end():
    """alpha = 0.99 (the renderer's clamp) on the front layer: no NaN/Inf anywhere."""
    generator = torch.Generator().manual_seed(10)
    alphas, colors, depths, c_bg = _layered_pixel(generator, m=4, front_alpha=ALPHA_CLAMP)
    dalphas = torch.randn(4, generator=generator, dtype=DTYPE)
    ddepths = torch.randn(4, generator=generator, dtype=DTYPE)

    forward = composite_forward(alphas, colors, depths, c_bg, 1e-6)
    jvp = composite_jvp(alphas, dalphas, colors, depths, ddepths, c_bg, 1e-6)
    sens = composite_vjp_alpha_sensitivities(alphas, colors, depths, c_bg, 1e-6)
    for name, group in (("forward", forward), ("jvp", jvp), ("vjp", sens)):
        for key, value in group.items():
            assert torch.isfinite(value).all(), f"{name}[{key}] = {value}"

    # A fully transparent pixel must not divide by zero either (spec 27).
    zeros = torch.zeros(4, dtype=DTYPE)
    empty = composite_forward(zeros, colors, depths, c_bg, 1e-6)
    assert torch.isfinite(empty["D"]).all() and float(empty["D"]) == 0.0
    empty_sens = composite_vjp_alpha_sensitivities(zeros, colors, depths, c_bg, 1e-6)
    for key, value in empty_sens.items():
        assert torch.isfinite(value).all(), f"empty[{key}] = {value}"


# --------------------------------------------------------------------------
# spec 11.3: the assembled 4x6 measurement block
# --------------------------------------------------------------------------


def test_pose_jacobian_block_matches_finite_differences_of_the_full_pixel():
    """H_{p,i} vs central FD of (C, D) under a 6D error-state step of Gaussian i.

    This is the end-to-end statement: projection, dJ_pi, dSigma, dQ, dalpha, the
    compositor sensitivities, and the direct dD/dd_i term all have to be right
    simultaneously or the block misses.
    """
    generator = torch.Generator().manual_seed(11)
    eps_D = 1e-6
    R_cw, t_cw, fx, fy, cx, cy = _random_camera(generator)
    c_bg = torch.rand(3, generator=generator, dtype=DTYPE)

    # Three layers stacked along the *camera* optical axis so they project onto
    # nearly the same pixel (a real occlusion stack) and the depth order cannot
    # flip under the finite-difference step -- the renderer is only piecewise
    # smooth across a sort change, so FD would be meaningless there.
    mus = _world_from_camera(
        torch.tensor([
            [0.05, -0.02, 3.2],
            [0.02, 0.03, 4.0],
            [-0.03, 0.01, 4.8],
        ], dtype=DTYPE),
        R_cw,
        t_cw,
    )
    Sigmas = torch.stack([_random_gaussian(generator, scale=0.22)[1] for _ in range(3)])
    opacity = torch.tensor([0.5, 0.6, 0.45], dtype=DTYPE)
    colors = torch.rand(3, 3, generator=generator, dtype=DTYPE)

    cache = project_gaussian(mus, Sigmas, R_cw, t_cw, fx, fy, cx, cy)
    assert torch.all(cache["depth"][1:] > cache["depth"][:-1]), "layers must be sorted"
    pixel = cache["u"].mean(dim=0) + torch.tensor([1.3, -0.9], dtype=DTYPE)

    blocks = measurement_jacobian_blocks(
        mus, Sigmas, opacity, colors, pixel, R_cw, t_cw, fx, fy, cx, cy, eps_D=eps_D, c_bg=c_bg
    )
    assert blocks["H"].shape == (3, 4, 6)
    assert not bool(blocks["clamped"].any()), "clamp must be inactive in this scene"
    assert float(blocks["alpha"].min()) > 1e-3, blocks["alpha"]

    def render(mus_now, Sigmas_now):
        state = project_gaussian(mus_now, Sigmas_now, R_cw, t_cw, fx, fy, cx, cy)
        alpha = gaussian_alpha(opacity, pixel, state["u"], state["Q"])["alpha"]
        out = composite_forward(alpha, colors, state["depth"], c_bg, eps_D)
        return torch.cat((out["C"], out["D"].reshape(1)))

    worst = 0.0
    for i in range(3):
        # Guard against a vacuous test: anisotropic covariances must give the
        # rotation half of the block real (if much smaller) content, otherwise
        # "H_phi = 0" would trivially match the finite differences.
        assert float(blocks["H"][i, :, 3:].abs().max()) > 1e-6, blocks["H"][i]
        for k in range(6):
            tangent = torch.zeros(6, dtype=DTYPE)
            tangent[k] = 1.0

            def measurement_of(step):
                mus_now = mus.clone()
                mus_now[i] = mus_now[i] + step * tangent[:3]
                Sigmas_now = Sigmas.clone()
                R_g = _so3_exp(step * tangent[3:])
                Sigmas_now[i] = R_g @ Sigmas_now[i] @ R_g.T
                return render(mus_now, Sigmas_now)

            analytic = blocks["H"][i, :, k]
            best, sweep = _sweep_error(measurement_of, analytic)
            # Translation columns land at ~1e-10.  The rotation columns are two
            # to four orders of magnitude smaller than the measurement they are
            # differenced out of, so their float64 FD floor is ~1e-8; 1e-6 keeps
            # the test meaningful without chasing round-off.
            assert best < 1e-6, f"H[{i}][:, {k}] relative error sweep {sweep}"
            worst = max(worst, best)

        # Rotation columns of the depth row are structurally zero *through d_i*:
        # only the alpha path can carry them (spec 11.2, dd/dphi = 0).
        expected_depth_rot = (
            blocks["dD_dalpha"][i] * blocks["J_alpha"][i, 3:]
        )
        assert torch.allclose(blocks["H"][i, 3, 3:], expected_depth_rot, atol=1e-14)

    # J^d = e_z^T R_cw, and dD/dd = w/(A + eps).
    assert torch.allclose(depth_jacobian(R_cw), R_cw[2, :], atol=1e-15)
    assert torch.allclose(
        blocks["dD_dd"], blocks["w"] / (blocks["A"] + eps_D), atol=1e-14
    )

    # The 12D lift must be the pose block padded with structural zeros.
    lifted = extend_block_to_12d(blocks["H"])
    assert lifted.shape == (3, 4, 12)
    assert torch.allclose(lifted[..., :6], blocks["H"], atol=0.0)
    assert float(lifted[..., 6:].abs().max()) == 0.0
    print(f"    worst relative H_(p,i) error over 18 columns: {worst:.3e}")


def test_single_pair_block_builder_agrees_with_the_batched_path():
    """pose_jacobian_block on one (pixel, Gaussian) pair == the batched assembly."""
    generator = torch.Generator().manual_seed(12)
    R_cw, t_cw, fx, fy, cx, cy = _random_camera(generator)
    mus = _world_from_camera(
        torch.tensor([[0.0, 0.0, 3.4], [0.04, -0.03, 4.6]], dtype=DTYPE), R_cw, t_cw
    )
    Sigmas = torch.stack([_random_gaussian(generator, scale=0.2)[1] for _ in range(2)])
    opacity = torch.tensor([0.5, 0.7], dtype=DTYPE)
    colors = torch.rand(2, 3, generator=generator, dtype=DTYPE)
    c_bg = torch.zeros(3, dtype=DTYPE)

    cache = project_gaussian(mus, Sigmas, R_cw, t_cw, fx, fy, cx, cy)
    pixel = cache["u"][0] + torch.tensor([0.7, 0.4], dtype=DTYPE)
    blocks = measurement_jacobian_blocks(
        mus, Sigmas, opacity, colors, pixel, R_cw, t_cw, fx, fy, cx, cy, c_bg=c_bg
    )

    for i in range(2):
        single = alpha_pose_jacobian(
            mus[i], Sigmas[i], opacity[i], pixel, R_cw, t_cw, fx, fy, cx, cy
        )
        assert torch.allclose(single["J_alpha"], blocks["J_alpha"][i], atol=1e-13)
        block = pose_jacobian_block(
            single["J_alpha"],
            colors[i],
            blocks["T"][i],
            blocks["B"][i],
            blocks["dD_dalpha"][i],
            blocks["dD_dd"][i],
            R_cw,
        )
        assert block.shape == (4, 6)
        assert torch.allclose(block, blocks["H"][i], atol=1e-13)


def test_reference_is_device_and_dtype_preserving():
    """float32 must work too: the CUDA kernels are float32 and share these formulas."""
    generator = torch.Generator().manual_seed(13)
    R_cw, t_cw, fx, fy, cx, cy = _random_camera(generator)
    mu, Sigma = _random_gaussian(generator, scale=0.2)
    opacity = torch.tensor(0.6, dtype=DTYPE)
    pixel = project(mu, R_cw, t_cw, fx, fy, cx, cy)[1] + 0.8

    ref = alpha_pose_jacobian(mu, Sigma, opacity, pixel, R_cw, t_cw, fx, fy, cx, cy)
    low = alpha_pose_jacobian(
        mu.float(), Sigma.float(), opacity.float(), pixel.float(),
        R_cw.float(), t_cw.float(), fx, fy, cx, cy,
    )
    assert low["J_alpha"].dtype == torch.float32
    assert low["J_alpha"].device == mu.device
    scale = max(float(ref["J_alpha"].abs().max()), 1e-12)
    # A few tens of float32 ulps: the 2x2 conic inverse and the dS product rule
    # each lose a couple of digits, which is exactly the budget the CUDA kernels
    # will have to live within.
    assert float((low["J_alpha"].double() - ref["J_alpha"]).abs().max()) / scale < 1e-4

    # projected_conic / rotate_covariance keep dtype as well.
    S32, Q32 = projected_conic(ref["J_pi"].float(), ref["Sigma_c"].float())
    assert S32.dtype == torch.float32 and Q32.dtype == torch.float32
    assert rotate_covariance(R_cw.float(), Sigma.float()).dtype == torch.float32
    assert math.isfinite(float(Q32.sum()))


def main():
    tests = [value for name, value in sorted(globals().items()) if name.startswith("test_")]
    for test in tests:
        test()
        print(f"ok  {test.__name__}")
    print(f"\n{len(tests)} tests passed")


if __name__ == "__main__":
    main()
