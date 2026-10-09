"""Closed-form analytic differentials of the 3DGS measurement chain (spec 9, 10, 11).

This is the CPU/PyTorch *reference*: the CUDA kernels added to
``submodules/diff-gaussian-rasterization`` are validated against the functions in
this file, so every convention here is copied from the production rasterizer
(``cuda_rasterizer/forward.cu``) rather than from a textbook:

* ``mu_c = R_cw mu + t_cw`` (the repo's flat ``viewmatrix`` is the transposed
  column-major form of the same map; this module takes ``R_cw``/``t_cw``
  explicitly so the reference never has to guess a storage order);
* ``J_pi = [[fx/Z, 0, -fx X/Z^2], [0, fy/Z, -fy Y/Z^2]]``;
* ``S = J_pi Sigma_c J_pi^T + 0.3 I_2`` -- the ``h_var = 0.3f`` low-pass term that
  ``preprocessCUDA`` adds to the 2D covariance *before* inverting it.  It is a
  constant, so it drops out of ``dS``, but it must be present in ``S`` or ``Q``
  (and therefore ``alpha`` and every derivative) is wrong;
* ``alpha = min(0.99, o exp(-0.5 e^T Q e))`` with ``e = x_p - u``.  The renderer's
  ``d = xy - pixf`` is ``-e``; the quadratic form is symmetric so the sign of
  ``e`` does not matter for ``alpha``, but it does matter for ``d alpha``.

No autograd is used anywhere in this module (spec 36): every derivative is an
explicit closed form.  All functions preserve device/dtype, broadcast over
arbitrary leading batch dimensions, and run in float64 for the finite-difference
acceptance tests.

Two production culling rules of the rasterizer are deliberately *not* reproduced
here: the ``1.3 * tan_fov`` frustum clamp applied to ``t.x``/``t.y`` inside
``computeCov2D`` (inactive for the in-frustum Gaussians the estimator
linearizes around, and non-differentiable when active), and the
``alpha < 1/255`` / ``T < 1e-4`` early-outs.  Those decisions define *which*
contributors enter the list; this module takes the already-sorted contributor
list as given (spec 12).
"""

from __future__ import annotations

from typing import Dict, Optional, Union

import torch

Tensor = torch.Tensor
Scalar = Union[float, Tensor]

# ``h_var`` in cuda_rasterizer/forward.cu: isotropic screen-space low-pass filter
# that keeps sub-pixel Gaussians invertible.
LOW_PASS_VARIANCE = 0.3
# ``min(0.99f, ...)`` in the render kernel.  Above this the measurement is
# constant, hence its derivative is exactly zero (not merely small).
ALPHA_CLAMP = 0.99
# Guard for D = Z / (A + eps_D) in fully transparent pixels (spec 8).
EPS_DEPTH = 1e-6


def _like(value: Scalar, reference: Tensor) -> Tensor:
    """Promote a python/tensor scalar onto the reference's device and dtype."""
    if isinstance(value, Tensor):
        return value.to(device=reference.device, dtype=reference.dtype)
    return torch.as_tensor(value, device=reference.device, dtype=reference.dtype)


def skew(v: Tensor) -> Tensor:
    """Hat operator ``[v]_x`` with ``[v]_x a = v x a``.  Shape ``(..., 3) -> (..., 3, 3)``."""
    zero = torch.zeros_like(v[..., 0])
    vx, vy, vz = v[..., 0], v[..., 1], v[..., 2]
    row0 = torch.stack((zero, -vz, vy), dim=-1)
    row1 = torch.stack((vz, zero, -vx), dim=-1)
    row2 = torch.stack((-vy, vx, zero), dim=-1)
    return torch.stack((row0, row1, row2), dim=-2)


def invert_sym2(S: Tensor) -> Tensor:
    """Closed-form inverse of a symmetric 2x2 batch.

    Written out instead of ``torch.linalg.inv`` to mirror the rasterizer's
    ``conic = {cov.z, -cov.y, cov.x} / det`` exactly, and because the 2x2
    adjugate is what the CUDA kernels will use.
    """
    a = S[..., 0, 0]
    # Symmetrize the off-diagonal: S is symmetric in exact arithmetic, but
    # J Sigma_c J^T accumulates asymmetric round-off.
    b = 0.5 * (S[..., 0, 1] + S[..., 1, 0])
    d = S[..., 1, 1]
    inv_det = 1.0 / (a * d - b * b)
    row0 = torch.stack((d * inv_det, -b * inv_det), dim=-1)
    row1 = torch.stack((-b * inv_det, a * inv_det), dim=-1)
    return torch.stack((row0, row1), dim=-2)


def _quadratic(e_left: Tensor, M: Tensor, e_right: Tensor) -> Tensor:
    """``e_left^T M e_right`` with broadcasting over leading dims."""
    return (e_left.unsqueeze(-2) @ M @ e_right.unsqueeze(-1)).squeeze(-1).squeeze(-1)


# --------------------------------------------------------------------------- #
# Spec 7 / 9.1: projection and its differential
# --------------------------------------------------------------------------- #


def project(
    mu: Tensor,
    R_cw: Tensor,
    t_cw: Tensor,
    fx: Scalar,
    fy: Scalar,
    cx: Scalar,
    cy: Scalar,
):
    """Project world centers into the image.

    Returns ``(mu_c, u, J_pi, depth)`` with shapes ``(...,3)``, ``(...,2)``,
    ``(...,2,3)``, ``(...,)``.  ``depth`` is the camera-space z the rasterizer
    stores in ``depths[idx] = p_view.z``; callers must have culled ``Z <= 0``
    before linearizing (the perspective Jacobian is singular there).
    """
    mu_c = (R_cw @ mu.unsqueeze(-1)).squeeze(-1) + t_cw
    X, Y, Z = mu_c[..., 0], mu_c[..., 1], mu_c[..., 2]
    fx_t, fy_t = _like(fx, mu_c), _like(fy, mu_c)
    cx_t, cy_t = _like(cx, mu_c), _like(cy, mu_c)

    inv_z = 1.0 / Z
    u = torch.stack((fx_t * X * inv_z + cx_t, fy_t * Y * inv_z + cy_t), dim=-1)

    zero = torch.zeros_like(Z)
    inv_z2 = inv_z * inv_z
    row0 = torch.stack((fx_t * inv_z, zero, -fx_t * X * inv_z2), dim=-1)
    row1 = torch.stack((zero, fy_t * inv_z, -fy_t * Y * inv_z2), dim=-1)
    J_pi = torch.stack((row0, row1), dim=-2)
    return mu_c, u, J_pi, Z


def du_from_dmu(J_pi: Tensor, dmu_c: Tensor) -> Tensor:
    """Spec 9.1: ``du = J_pi R_cw dmu``; a rotation about the center leaves ``u`` fixed."""
    return (J_pi @ dmu_c.unsqueeze(-1)).squeeze(-1)


def dJ_pi(mu_c: Tensor, dmu_c: Tensor, fx: Scalar, fy: Scalar) -> Tensor:
    """Spec 9.2: differential of the perspective Jacobian.

    Omitting this term is the classic 3DGS Jacobian bug: moving a center changes
    the *shape* of the projected conic, not just its position, so ``dS`` would
    lose two of its three contributions and ``d alpha`` would be wrong for every
    off-center pixel.
    """
    X, Y, Z = mu_c[..., 0], mu_c[..., 1], mu_c[..., 2]
    dX, dY, dZ = dmu_c[..., 0], dmu_c[..., 1], dmu_c[..., 2]
    fx_t, fy_t = _like(fx, mu_c), _like(fy, mu_c)

    inv_z2 = 1.0 / (Z * Z)
    inv_z3 = inv_z2 / Z
    a00 = -fx_t * dZ * inv_z2
    a02 = -fx_t * dX * inv_z2 + 2.0 * fx_t * X * dZ * inv_z3
    a11 = -fy_t * dZ * inv_z2
    a12 = -fy_t * dY * inv_z2 + 2.0 * fy_t * Y * dZ * inv_z3
    # mu_c and dmu_c may broadcast against each other (e.g. one Gaussian against
    # six tangent directions), so the structural zeros must take the joint shape.
    zero = torch.zeros(
        torch.broadcast_shapes(a00.shape, a02.shape, a11.shape, a12.shape),
        device=mu_c.device,
        dtype=mu_c.dtype,
    )
    row0 = torch.stack(torch.broadcast_tensors(a00, zero, a02), dim=-1)
    row1 = torch.stack(torch.broadcast_tensors(zero, a11, a12), dim=-1)
    return torch.stack(torch.broadcast_tensors(row0, row1), dim=-2)


# --------------------------------------------------------------------------- #
# Spec 9.3 / 9.4: covariance differentials and the inverse conic
# --------------------------------------------------------------------------- #


def dSigma_world(Sigma: Tensor, dphi: Tensor) -> Tensor:
    """Spec 9.3: ``dSigma = [dphi]_x Sigma - Sigma [dphi]_x``.

    This is the first-order effect of the *left* retraction
    ``R+ = Exp([dphi]_x) R`` on ``Sigma = R Lambda R^T``; the two terms are the
    transpose of each other, so the result is symmetric by construction.
    """
    hat = skew(dphi)
    return hat @ Sigma - Sigma @ hat


def rotate_covariance(R: Tensor, Sigma: Tensor) -> Tensor:
    """``R Sigma R^T`` -- used for both ``Sigma_c`` and ``dSigma_c`` (spec 9.3)."""
    return R @ Sigma @ R.transpose(-1, -2)


def dSigma_world_basis(Sigma: Tensor) -> Tensor:
    """The three basis derivatives ``dSigma/d phi_l = E_l Sigma - Sigma E_l`` (spec 9.3).

    Returns ``(..., 3, 3, 3)`` indexed ``[l, row, col]``.
    """
    eye = torch.eye(3, device=Sigma.device, dtype=Sigma.dtype)
    return dSigma_world(Sigma.unsqueeze(-3), eye)


def dS_and_dQ(
    J_pi: Tensor,
    dJ: Tensor,
    Sigma_c: Tensor,
    dSigma_c: Tensor,
    Q: Tensor,
):
    """Spec 9.4: ``dS`` (all three product-rule terms) and ``dQ = -Q dS Q``.

    The ``+0.3 I`` low-pass term of ``S`` is constant and therefore absent from
    ``dS``; it is nevertheless baked into ``Q``, which is why ``Q`` is an input
    rather than something recomputed here.
    """
    J_T = J_pi.transpose(-1, -2)
    dJ_T = dJ.transpose(-1, -2)
    dS = dJ @ Sigma_c @ J_T + J_pi @ dSigma_c @ J_T + J_pi @ Sigma_c @ dJ_T
    dQ = -(Q @ dS @ Q)
    return dS, dQ


def projected_conic(
    J_pi: Tensor,
    Sigma_c: Tensor,
    low_pass: Scalar = LOW_PASS_VARIANCE,
):
    """``S = J_pi Sigma_c J_pi^T + low_pass * I_2`` and ``Q = S^{-1}`` (spec 7)."""
    S_raw = J_pi @ Sigma_c @ J_pi.transpose(-1, -2)
    eye = torch.eye(2, device=S_raw.device, dtype=S_raw.dtype)
    S = S_raw + _like(low_pass, S_raw) * eye
    return S, invert_sym2(S)


def project_gaussian(
    mu: Tensor,
    Sigma: Tensor,
    R_cw: Tensor,
    t_cw: Tensor,
    fx: Scalar,
    fy: Scalar,
    cx: Scalar,
    cy: Scalar,
    low_pass: Scalar = LOW_PASS_VARIANCE,
) -> Dict[str, Tensor]:
    """Full forward projection of a Gaussian: the linearization cache of spec 20.2."""
    mu_c, u, J_pi, depth = project(mu, R_cw, t_cw, fx, fy, cx, cy)
    Sigma_c = rotate_covariance(R_cw, Sigma)
    S, Q = projected_conic(J_pi, Sigma_c, low_pass)
    return {
        "mu_c": mu_c,
        "u": u,
        "J_pi": J_pi,
        "depth": depth,
        "Sigma_c": Sigma_c,
        "S": S,
        "Q": Q,
    }


# --------------------------------------------------------------------------- #
# Spec 9.5: the alpha differential
# --------------------------------------------------------------------------- #


def gaussian_alpha(
    opacity: Tensor,
    pixel: Tensor,
    u: Tensor,
    Q: Tensor,
    clamp: bool = True,
    alpha_max: Scalar = ALPHA_CLAMP,
) -> Dict[str, Tensor]:
    """``alpha = o exp(-0.5 e^T Q e)`` with ``e = x_p - u``.

    Both the unclamped value and the clamp mask are returned: the renderer takes
    ``min(0.99, .)``, and on the clamped side the measurement is *constant*, so
    the Jacobian there is exactly zero rather than small.  Callers that forget
    this get a spurious information contribution from saturated splats.
    """
    e = pixel - u
    power = -0.5 * _quadratic(e, Q, e)
    alpha_raw = opacity * torch.exp(power)
    cap = _like(alpha_max, alpha_raw)
    clamped = alpha_raw > cap
    alpha = torch.minimum(alpha_raw, cap) if clamp else alpha_raw
    return {
        "e": e,
        "power": power,
        "alpha_raw": alpha_raw,
        "alpha": alpha,
        "clamped": clamped if clamp else torch.zeros_like(clamped),
    }


def dalpha(
    alpha: Tensor,
    e: Tensor,
    Q: Tensor,
    du: Tensor,
    dQ: Tensor,
    clamped: Optional[Tensor] = None,
) -> Tensor:
    """Spec 9.5: ``d alpha = alpha [ e^T Q du - 0.5 e^T dQ e ]``.

    Derivation: ``de = -du``, so ``d(-0.5 e^T Q e) = e^T Q du - 0.5 e^T dQ e``.
    Pass ``clamped`` (from :func:`gaussian_alpha`) to zero the rows where the
    0.99 saturation is active.
    """
    linear = _quadratic(e, Q, du) - 0.5 * _quadratic(e, dQ, e)
    out = alpha * linear
    if clamped is not None:
        out = torch.where(clamped, torch.zeros_like(out), out)
    return out


def dalpha_full_chain(
    cache: Dict[str, Tensor],
    Sigma: Tensor,
    alpha: Tensor,
    e: Tensor,
    dmu: Tensor,
    dphi: Tensor,
    R_cw: Tensor,
    fx: Scalar,
    fy: Scalar,
    clamped: Optional[Tensor] = None,
) -> Tensor:
    """``d alpha`` for an explicit ``(dmu, dphi)`` tangent, running the whole chain.

    ``cache`` is the dict from :func:`project_gaussian`.  Leading dims of
    ``dmu``/``dphi`` broadcast against the cache, which is how
    :func:`alpha_pose_jacobian` evaluates all six basis directions at once.
    """
    dmu_c = (R_cw @ dmu.unsqueeze(-1)).squeeze(-1)
    du = du_from_dmu(cache["J_pi"], dmu_c)
    dJ = dJ_pi(cache["mu_c"], dmu_c, fx, fy)
    dSigma_c = rotate_covariance(R_cw, dSigma_world(Sigma, dphi))
    _, dQ = dS_and_dQ(cache["J_pi"], dJ, cache["Sigma_c"], dSigma_c, cache["Q"])
    return dalpha(alpha, e, cache["Q"], du, dQ, clamped=clamped)


def alpha_pose_jacobian(
    mu: Tensor,
    Sigma: Tensor,
    opacity: Tensor,
    pixel: Tensor,
    R_cw: Tensor,
    t_cw: Tensor,
    fx: Scalar,
    fy: Scalar,
    cx: Scalar,
    cy: Scalar,
    low_pass: Scalar = LOW_PASS_VARIANCE,
    clamp: bool = True,
) -> Dict[str, Tensor]:
    """``J^alpha_{p,i} = [d alpha / d dmu , d alpha / d dphi] in R^{1x6}`` (spec 11.3).

    Evaluated by pushing the six error-state basis vectors through the closed
    forms of 9.1-9.5 in one batched sweep -- no finite differences, no autograd.
    """
    cache = project_gaussian(mu, Sigma, R_cw, t_cw, fx, fy, cx, cy, low_pass)
    alpha_state = gaussian_alpha(opacity, pixel, cache["u"], cache["Q"], clamp=clamp)

    eye = torch.eye(3, device=mu.device, dtype=mu.dtype)
    zero = torch.zeros_like(eye)
    # Rows 0-2 translate the center, rows 3-5 rotate it about its own center.
    basis_dmu = torch.cat((eye, zero), dim=0)
    basis_dphi = torch.cat((zero, eye), dim=0)

    # Insert a tangent-direction axis so the cached per-Gaussian quantities
    # broadcast against the six basis directions in a single sweep.
    cache_b = {
        "mu_c": cache["mu_c"].unsqueeze(-2),
        "J_pi": cache["J_pi"].unsqueeze(-3),
        "Sigma_c": cache["Sigma_c"].unsqueeze(-3),
        "Q": cache["Q"].unsqueeze(-3),
    }
    J_alpha = dalpha_full_chain(
        cache_b,
        Sigma.unsqueeze(-3),
        alpha_state["alpha"].unsqueeze(-1),
        alpha_state["e"].unsqueeze(-2),
        basis_dmu,
        basis_dphi,
        R_cw,
        fx,
        fy,
        clamped=alpha_state["clamped"].unsqueeze(-1),
    )

    out = dict(cache)
    out.update(alpha_state)
    out["J_alpha"] = J_alpha
    return out


def depth_jacobian(R_cw: Tensor) -> Tensor:
    """``J^d_i = e_z^T R_cw`` (spec 11.2); ``d d_i / d dphi_i = 0``.

    Returns ``(..., 3)`` -- the row of ``R_cw`` that reads camera z.
    """
    return R_cw[..., 2, :]


# --------------------------------------------------------------------------- #
# Spec 8 / 10: front-to-back compositor and its JVP
# --------------------------------------------------------------------------- #


def _background(c_bg: Optional[Tensor], colors: Tensor) -> Tensor:
    if c_bg is None:
        return torch.zeros_like(colors[..., 0, :])
    return c_bg.to(device=colors.device, dtype=colors.dtype)


def composite_forward(
    alphas: Tensor,
    colors: Tensor,
    depths: Tensor,
    c_bg: Optional[Tensor] = None,
    eps_D: Scalar = EPS_DEPTH,
) -> Dict[str, Tensor]:
    """Front-to-back compositing over a fixed depth-sorted contributor list (spec 8).

    Shapes: ``alphas (..., m)``, ``colors (..., m, 3)``, ``depths (..., m)``,
    ``c_bg (..., 3)``.  ``T`` is the *prefix* transmittance ``T_i`` seen by
    contributor ``i`` (``T_1 = 1``), built with a shifted ``cumprod`` so it never
    divides by ``1 - alpha`` (spec 27: that division blows up at saturation).
    """
    one_minus = 1.0 - alphas
    running = torch.cumprod(one_minus, dim=-1)
    ones = torch.ones_like(running[..., :1])
    T = torch.cat((ones, running[..., :-1]), dim=-1)
    T_bg = running[..., -1]

    w = T * alphas
    bg = _background(c_bg, colors)
    C = (w.unsqueeze(-1) * colors).sum(dim=-2) + T_bg.unsqueeze(-1) * bg
    A = w.sum(dim=-1)
    Z = (w * depths).sum(dim=-1)
    D = Z / (A + _like(eps_D, A))
    return {"C": C, "A": A, "Z": Z, "D": D, "T": T, "T_bg": T_bg, "w": w}


def composite_jvp(
    alphas: Tensor,
    dalphas: Tensor,
    colors: Tensor,
    depths: Tensor,
    ddepths: Optional[Tensor] = None,
    c_bg: Optional[Tensor] = None,
    eps_D: Scalar = EPS_DEPTH,
) -> Dict[str, Tensor]:
    """Exact tangent compositor of spec 10: the matrix-free ``v -> H v``.

    The recurrence is written out literally (not as a vectorized shortcut)
    because it is the specification of the CUDA JVP tile pass, and because
    ``dT`` is what carries front/rear occlusion coupling: a change in a front
    splat's alpha reaches *every* contributor behind it through ``dT``.

    Colors are frozen during motion estimation (spec 3.1), so there is no
    ``dc`` term; depths carry a tangent because ``d_i = e_z^T mu_i^c`` moves with
    the center.
    """
    m = alphas.shape[-1]
    bg = _background(c_bg, colors)

    T = torch.ones_like(alphas[..., 0])
    dT = torch.zeros_like(alphas[..., 0])
    C = torch.zeros_like(colors[..., 0, :])
    dC = torch.zeros_like(colors[..., 0, :])
    A = torch.zeros_like(alphas[..., 0])
    dA = torch.zeros_like(alphas[..., 0])
    Z = torch.zeros_like(alphas[..., 0])
    dZ = torch.zeros_like(alphas[..., 0])

    for i in range(m):
        a = alphas[..., i]
        da = dalphas[..., i]
        c = colors[..., i, :]
        d = depths[..., i]
        dd = None if ddepths is None else ddepths[..., i]

        # dw_i = dT_i alpha_i + T_i dalpha_i is the shared factor of dC/dA/dZ.
        dw = dT * a + T * da
        w = T * a

        C = C + w.unsqueeze(-1) * c
        dC = dC + dw.unsqueeze(-1) * c
        A = A + w
        dA = dA + dw
        Z = Z + w * d
        dZ = dZ + dw * d
        if dd is not None:
            dZ = dZ + w * dd

        dT = dT * (1.0 - a) - T * da
        T = T * (1.0 - a)

    # Background is composited with the final transmittance, so its tangent is dT.
    C = C + T.unsqueeze(-1) * bg
    dC = dC + dT.unsqueeze(-1) * bg

    denom = A + _like(eps_D, A)
    D = Z / denom
    dD = (dZ - D * dA) / denom
    return {
        "C": C,
        "A": A,
        "Z": Z,
        "D": D,
        "T_bg": T,
        "dC": dC,
        "dA": dA,
        "dZ": dZ,
        "dD": dD,
        "dT_bg": dT,
    }


# --------------------------------------------------------------------------- #
# Spec 11: closed-form compositor sensitivities (VJP)
# --------------------------------------------------------------------------- #


def composite_vjp_alpha_sensitivities(
    alphas: Tensor,
    colors: Tensor,
    depths: Tensor,
    c_bg: Optional[Tensor] = None,
    eps_D: Scalar = EPS_DEPTH,
) -> Dict[str, Tensor]:
    """Per-contributor sensitivities ``dC/dalpha_i``, ``dD/dalpha_i``, ``dD/dd_i``.

    The behind-suffixes ``B``, ``A^b``, ``Z^b`` of spec 11.1/11.2 are built by an
    explicit *reverse* traversal.  They are deliberately not recovered from the
    forward prefix accumulators by dividing out ``(1 - alpha_{i+1})``: that
    division is unbounded as alpha saturates (spec 27), and with the renderer's
    0.99 clamp -- or an externally supplied alpha of exactly 1 -- it is an
    outright division by zero.
    """
    m = alphas.shape[-1]
    forward = composite_forward(alphas, colors, depths, c_bg=c_bg, eps_D=eps_D)
    T, w, A, Z, D = (forward[k] for k in ("T", "w", "A", "Z", "D"))
    bg = _background(c_bg, colors)

    B = torch.empty_like(colors)
    Ab = torch.empty_like(alphas)
    Zb = torch.empty_like(depths)
    # Behind the last contributor there is only background.
    B[..., m - 1, :] = bg
    Ab[..., m - 1] = 0.0
    Zb[..., m - 1] = 0.0
    for i in range(m - 2, -1, -1):
        a_next = alphas[..., i + 1]
        keep = 1.0 - a_next
        B[..., i, :] = (
            a_next.unsqueeze(-1) * colors[..., i + 1, :]
            + keep.unsqueeze(-1) * B[..., i + 1, :]
        )
        Ab[..., i] = a_next + keep * Ab[..., i + 1]
        Zb[..., i] = a_next * depths[..., i + 1] + keep * Zb[..., i + 1]

    denom = (A + _like(eps_D, A)).unsqueeze(-1)
    dC_dalpha = T.unsqueeze(-1) * (colors - B)
    dA_dalpha = T * (1.0 - Ab)
    dZ_dalpha = T * (depths - Zb)
    dD_dalpha = (dZ_dalpha - D.unsqueeze(-1) * dA_dalpha) / denom
    dD_dd = w / denom
    return {
        "T": T,
        "w": w,
        "A": A,
        "Z": Z,
        "D": D,
        "C": forward["C"],
        "T_bg": forward["T_bg"],
        "B": B,
        "Ab": Ab,
        "Zb": Zb,
        "dC_dalpha": dC_dalpha,
        "dA_dalpha": dA_dalpha,
        "dZ_dalpha": dZ_dalpha,
        "dD_dalpha": dD_dalpha,
        "dD_dd": dD_dd,
        "dZ_dd": w,
    }


def pose_jacobian_block(
    J_alpha: Tensor,
    color: Tensor,
    transmittance: Tensor,
    behind_color: Tensor,
    dD_dalpha: Tensor,
    dD_dd: Tensor,
    R_cw: Tensor,
) -> Tensor:
    """The ``4x6`` block ``H_{p,i}`` of spec 11.3 for one (pixel, Gaussian) pair.

    Rows 0-2 are ``H^RGB = T_i (c_i - B_i) J^alpha``; row 3 is
    ``H^D = (dD/dalpha_i) J^alpha + (w_i/(A+eps)) [J^d_i, 0_{1x3}]``.  The second
    depth term exists because the center depth ``d_i = e_z^T mu_i^c`` itself moves
    with ``dmu``; dropping it makes depth residuals look blind to motion along
    the optical axis.

    Columns are ordered ``[dmu (3) | dphi (3)]``, matching the 12D error state of
    spec 4 truncated to its pose half (see :func:`extend_block_to_12d`).
    """
    H_rgb = (transmittance.unsqueeze(-1) * (color - behind_color)).unsqueeze(-1) * J_alpha.unsqueeze(-2)

    Jd = depth_jacobian(R_cw)
    zeros3 = torch.zeros_like(Jd)
    Jd_row = torch.cat((Jd, zeros3), dim=-1)
    H_depth = dD_dalpha.unsqueeze(-1) * J_alpha + dD_dd.unsqueeze(-1) * Jd_row
    return torch.cat((H_rgb, H_depth.unsqueeze(-2)), dim=-2)


def extend_block_to_12d(H_pose: Tensor) -> Tensor:
    """``bar H_{p,i} = [H_{p,i}, 0_{4x6}]`` (spec 11.3).

    A single RGB-D frame carries no direct information about ``(v, omega)``;
    velocity is corrected only through the temporal prior's cross-information,
    so these columns are *structurally* zero, never a placeholder.
    """
    return torch.cat((H_pose, torch.zeros_like(H_pose)), dim=-1)


def measurement_jacobian_blocks(
    mu: Tensor,
    Sigma: Tensor,
    opacity: Tensor,
    colors: Tensor,
    pixel: Tensor,
    R_cw: Tensor,
    t_cw: Tensor,
    fx: Scalar,
    fy: Scalar,
    cx: Scalar,
    cy: Scalar,
    low_pass: Scalar = LOW_PASS_VARIANCE,
    eps_D: Scalar = EPS_DEPTH,
    c_bg: Optional[Tensor] = None,
    clamp: bool = True,
) -> Dict[str, Tensor]:
    """Assemble ``H_{p,i}`` for every contributor of one pixel (spec 11.3).

    ``mu (m,3)``, ``Sigma (m,3,3)``, ``opacity (m,)``, ``colors (m,3)`` must
    already be in the renderer's front-to-back depth order; the caller owns the
    sort (spec 12).  Returns the blocks plus the forward render of the pixel, so
    a residual can be formed without a second pass.
    """
    state = alpha_pose_jacobian(
        mu, Sigma, opacity, pixel, R_cw, t_cw, fx, fy, cx, cy, low_pass, clamp
    )
    sens = composite_vjp_alpha_sensitivities(
        state["alpha"], colors, state["depth"], c_bg=c_bg, eps_D=eps_D
    )
    blocks = pose_jacobian_block(
        state["J_alpha"],
        colors,
        sens["T"],
        sens["B"],
        sens["dD_dalpha"],
        sens["dD_dd"],
        R_cw,
    )
    out = dict(sens)
    out["alpha"] = state["alpha"]
    out["clamped"] = state["clamped"]
    out["J_alpha"] = state["J_alpha"]
    out["depth"] = state["depth"]
    out["H"] = blocks
    return out
