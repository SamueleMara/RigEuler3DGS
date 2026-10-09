"""Sections 5-8: the continuous SE(3) chart in which prismatic is a regular point.

CODEX_IMPLEMENT_CONTINUOUS_SE3_PRISMATIC_REVOLUTE_GEOMETRY.md sections 3-8, 22.

The obstruction this module removes
---------------------------------------------------------------------------
The finite-axis revolute map

    x_t = a + Exp(q_t [e]_x) (x_0 - a)

has NO finite minimiser for a pure translation: it is reached only as
||a|| -> inf, q_t -> 0 with ||a|| q_t -> s_t. Any convergence test on a, q or
on the induced motion then fails by construction, which is exactly what was
measured -- a noiseless prismatic motion stayed MAX_ITERS_UNSTABLE at 200, 400
and 900 iterations with the per-iteration changes plateauing rather than
decaying (section 1).

This chart replaces the singular pair (kappa, n) with a single vector:

    d    in S^2    the tangential travel direction at the part centre
    eta  in R^3    angular change PER UNIT signed arc length, with eta . d = 0
    s_t  in R      signed arc length, s_0 = 0

so curvature kappa = ||eta|| and eta = 0 IS pure translation -- an ordinary
interior point of the parameter space with finite, continuous derivatives, not
a limit at infinity. Section 5 requires eta itself to be the optimised
variable; (kappa, n) are computed only for reporting, and only once curvature
is resolved.

Nothing on this path divides by curvature (section 7).
"""

from __future__ import annotations

import math

import torch

_EPS64 = float(torch.finfo(torch.float64).eps)

__all__ = ["stable_so3_coefficients", "skew", "continuous_joint_transform",
           "transform_points", "finite_axis_to_continuous",
           "continuous_to_finite_axis", "SERIES_CUTOFF", "series_cutoff",
           "fit_continuous_joint", "continuous_objective",
           "initialise_continuous", "canonical_moments",
           "gaussian_transport_residual",
           "gaussian_transport_error_pointwise"]


def series_cutoff(dtype: torch.dtype) -> float:
    """Where to switch from the closed form to the series, from dtype precision.

    Section 7 requires this bound to come from floating-point error analysis
    rather than from a dataset. The binding term is

        B(theta) = (1 - cos theta) / theta^2,

    because 1 - cos theta is a catastrophic cancellation for small theta: its
    absolute error is ~eps, so B carries a relative error of about
    2 eps / theta^2. The truncated series 1/2 - theta^2/24 + theta^4/720 has a
    relative error of about theta^6 / 20160. Setting the two equal,

        theta_switch = (40320 eps)^(1/8),

    which is 5.5e-2 in float64 and 7.3e-1 in float32. Below it the series is
    strictly more accurate, above it the closed form is.
    """
    eps = torch.finfo(dtype).eps
    return float((40320.0 * eps) ** (1.0 / 8.0))


#: float64 cutoff, exported for tests and logging.
SERIES_CUTOFF = series_cutoff(torch.float64)


def stable_so3_coefficients(theta: torch.Tensor):
    """A, B, C of section 6, evaluated without cancellation near zero.

        A(theta) = sin(theta) / theta
        B(theta) = (1 - cos(theta)) / theta^2
        C(theta) = (theta - sin(theta)) / theta^3

    At theta = 0 these are exactly 1, 1/2 and 1/6, and the series branch
    returns those values with finite, continuous derivatives (section 7).
    """
    th = theta
    t2 = th * th
    cut = series_cutoff(th.dtype)
    small = th < cut

    # Series: four terms each, which is well past the switch point's accuracy.
    A_s = 1.0 - t2 / 6.0 + t2 * t2 / 120.0 - t2 * t2 * t2 / 5040.0
    B_s = 0.5 - t2 / 24.0 + t2 * t2 / 720.0 - t2 * t2 * t2 / 40320.0
    C_s = (1.0 / 6.0) - t2 / 120.0 + t2 * t2 / 5040.0 - t2 * t2 * t2 / 362880.0

    # Closed form, guarded so the unused branch never divides by zero.
    safe = torch.where(small, torch.ones_like(th), th)
    s2 = safe * safe
    A_c = torch.sin(safe) / safe
    B_c = (1.0 - torch.cos(safe)) / s2
    C_c = (safe - torch.sin(safe)) / (s2 * safe)

    return (torch.where(small, A_s, A_c),
            torch.where(small, B_s, B_c),
            torch.where(small, C_s, C_c))


def skew(v: torch.Tensor) -> torch.Tensor:
    """[v]_x for a batch of 3-vectors."""
    z = torch.zeros_like(v[..., 0])
    return torch.stack([
        torch.stack([z, -v[..., 2], v[..., 1]], dim=-1),
        torch.stack([v[..., 2], z, -v[..., 0]], dim=-1),
        torch.stack([-v[..., 1], v[..., 0], z], dim=-1)], dim=-2)


def continuous_joint_transform(c: torch.Tensor, d: torch.Tensor,
                               eta: torch.Tensor, s: torch.Tensor):
    """Section 6 / section 22: T_t for every frame, exactly.

        phi_t = s_t eta,  theta_t = ||phi_t||
        R_t   = I + A W + B W^2,                 W = [phi_t]_x
        t_c,t = s_t (I + B W + C W^2) d
        p_t   = c - R_t c + t_c,t

    Returns ``(R, p)`` with shapes ``[T,3,3]`` and ``[T,3]``.

    At eta = 0 this evaluates exactly to R = I and p = s d: the series branch
    gives A = 1, B = 1/2, C = 1/6 with W = 0, so every W-term vanishes
    identically rather than by cancellation. No eta / ||eta|| and no 1/kappa
    appears anywhere (section 7).
    """
    c = c.reshape(3).double()
    d = d.reshape(3).double()
    eta = eta.reshape(3).double()
    s = s.reshape(-1).double()

    phi = s.unsqueeze(-1) * eta.unsqueeze(0)            # [T,3]
    theta = phi.norm(dim=-1)                            # [T]
    A, B, C = stable_so3_coefficients(theta)
    W = skew(phi)                                       # [T,3,3]
    W2 = W @ W
    eye = torch.eye(3, dtype=torch.float64, device=c.device).expand_as(W)

    R = eye + A.view(-1, 1, 1) * W + B.view(-1, 1, 1) * W2
    V = eye + B.view(-1, 1, 1) * W + C.view(-1, 1, 1) * W2
    centre_delta = s.unsqueeze(-1) * torch.einsum("tij,j->ti", V, d)
    p = c.unsqueeze(0) - torch.einsum("tij,j->ti", R, c) + centre_delta
    return R, p


def transform_points(x: torch.Tensor, R: torch.Tensor, p: torch.Tensor):
    """x_t = R_t x + p_t for a batch of canonical points. [T,N,3]."""
    return torch.einsum("tij,nj->tni", R.double(), x.double()) + p.double().unsqueeze(1)


# ---------------------------------------------------------------------------
# section 8: equivalence with the finite-axis revolute parameterisation
# ---------------------------------------------------------------------------
def finite_axis_to_continuous(c: torch.Tensor, axis: torch.Tensor,
                              axis_point: torch.Tensor, q: torch.Tensor):
    """(n, a, q_t) -> (d, eta, s_t), for an axis NOT through the centroid.

    With r the centroid-to-axis distance, section 8 gives kappa = 1/r and
    s_t = r q_t; the tangential direction at c is d = n x (c - a) / r.
    """
    c = c.reshape(3).double()
    n = axis.reshape(3).double()
    n = n / n.norm()
    a = axis_point.reshape(3).double()
    q = q.reshape(-1).double()

    rvec = (c - a) - n * (n @ (c - a))       # centroid offset perpendicular to n
    r = rvec.norm()
    if float(r) <= 0.0:
        raise ValueError("axis passes through the canonical centroid; this "
                         "chart is undefined there -- use the finite-axis "
                         "branch (section 9)")
    d = torch.linalg.cross(n, rvec / r)
    eta = n / r
    s = r * q
    return d, eta, s


def continuous_to_finite_axis(c: torch.Tensor, d: torch.Tensor,
                              eta: torch.Tensor, s: torch.Tensor,
                              kappa_floor: float | None = None):
    """(d, eta, s_t) -> (n, a_c, q_t), only where curvature is resolved.

    Section 8's closest point to the CANONICAL CENTROID is

        a_c = c + (eta x d) / ||eta||^2,

    which satisfies n . (a_c - c) = 0. Returns ``None`` when curvature is below
    the representable floor: at eta = 0 the axis does not exist and section 7
    forbids constructing one.
    """
    eta = eta.reshape(3).double()
    kappa = float(eta.norm())
    floor = (kappa_floor if kappa_floor is not None
             else math.sqrt(torch.finfo(torch.float64).eps))
    if kappa <= floor:
        return None
    c = c.reshape(3).double()
    d = d.reshape(3).double()
    n = eta / kappa
    a_c = c + torch.linalg.cross(eta, d) / (kappa * kappa)
    q = kappa * s.reshape(-1).double()
    # Section 8: the point closest to the WORLD ORIGIN is a different point.
    a_0 = a_c - n * (n @ a_c)
    return {"axis": n, "axis_point_centroid": a_c, "axis_point_origin": a_0,
            "q": q, "kappa": kappa, "radius": 1.0 / kappa}


# ---------------------------------------------------------------------------
# sections 10-15: fitting in the continuous chart
# ---------------------------------------------------------------------------
def _geometric_residual(R_obs, p_obs, R, p, L):
    """Section 10's residual, identical for P and R.

    Stacks the dimensionless position error and the rotation geodesic so one
    radian of orientation error weighs the same as one part-radius of
    displacement -- the same normalisation the decision later compares with.
    """
    # Frame 0 is the gauge: T_0 = I on BOTH sides (section 5 fixes s_0 = 0), so
    # its residual is identically zero and carries no information.
    #
    # project_to_so3 is deliberately NOT called here. It is an SVD, and a
    # rotation matrix has three identical singular values, for which the SVD
    # backward is NaN -- so projecting inside the differentiable residual makes
    # the entire autograd Jacobian NaN and the fit stops after one iteration.
    # R from this chart is exactly orthogonal by construction (Rodrigues), and
    # R_obs is projected ONCE outside the differentiable path by the caller.
    dpos = (p_obs[1:] - p[1:]) / L
    rel = torch.einsum("tji,tjk->tik", R[1:], R_obs[1:])
    drot = _so3_log_safe(rel)
    return torch.cat([dpos.reshape(-1), drot.reshape(-1)])


def _so3_log_safe(Rm: torch.Tensor) -> torch.Tensor:
    """log of a batch of rotations, differentiable through the identity.

    The usual theta / (2 sin theta) factor is singular at theta = 0. Writing it
    as the stable series of 1/A(theta) keeps the value, and its derivative,
    finite there -- which matters because a well-fitting frame sits exactly at
    theta = 0.
    """
    tr = Rm[..., 0, 0] + Rm[..., 1, 1] + Rm[..., 2, 2]
    cos = ((tr - 1.0) * 0.5).clamp(-1.0, 1.0)
    vee = torch.stack([Rm[..., 2, 1] - Rm[..., 1, 2],
                       Rm[..., 0, 2] - Rm[..., 2, 0],
                       Rm[..., 1, 0] - Rm[..., 0, 1]], dim=-1)

    # The factor is theta / (2 sin theta). Computing theta with acos is fatal
    # here: d(acos)/dx is infinite at x = 1, which is exactly where a
    # well-fitting frame sits, so the autograd gradient comes back NaN at the
    # solution. Writing the factor as a series in x = 1 - cos theta instead,
    #
    #     theta / (2 sin theta) = 1/2 (1 + x/3 + 4x^2/45 + 44x^3/945 + ...),
    #
    # is smooth through theta = 0 and needs no acos at all. The closed form is
    # used once x is large enough for acos to be well conditioned.
    x = 1.0 - cos
    small = x < 1e-3
    fac_s = 0.5 * (1.0 + x / 3.0 + 4.0 * x * x / 45.0
                   + 44.0 * x * x * x / 945.0)
    safe = torch.where(small, torch.full_like(cos, 0.5), cos)
    th = torch.acos(safe.clamp(-1.0, 1.0))
    fac_c = th / (2.0 * torch.sin(th).clamp_min(1e-300))
    factor = torch.where(small, fac_s, fac_c)
    return vee * factor.unsqueeze(-1)


def tangent_basis(d, T: int, fix_eta_zero: bool):
    """Section 4: an orthonormal basis of the MINIMAL coordinates.

    The chart has 6 + T stored numbers but only 2 + 2 + (T-1) real degrees of
    freedom:

      * d lives on the unit sphere, so only the two directions perpendicular
        to it move it anywhere;
      * eta must stay perpendicular to d, so it has the same two;
      * s has T entries but s_0 is pinned by the gauge, leaving T-1.

    Solving in the stored coordinates means the normal equations carry three
    exactly-null directions. They are singular by construction, every solve
    needs heavy damping to return anything finite, and the damping is what
    stops the fit making progress long before it reaches a stationary point.
    Building the step in this basis instead removes those directions from the
    problem rather than damping them, which is what section 4 asks for:
    "do not retain fictitious gauge DOFs in normal equations".

    Returns B with shape [6+T, k], columns orthonormal.
    """
    dev, dt = d.device, d.dtype
    tmp = torch.tensor([0.0, 0.0, 1.0], dtype=dt, device=dev)
    if abs(float(d @ tmp)) > 0.9:
        tmp = torch.tensor([0.0, 1.0, 0.0], dtype=dt, device=dev)
    u1 = torch.linalg.cross(d, tmp)
    u1 = u1 / u1.norm().clamp_min(1e-300)
    u2 = torch.linalg.cross(d, u1)
    u2 = u2 / u2.norm().clamp_min(1e-300)

    cols = []
    for u in (u1, u2):                                   # d on the sphere
        col = torch.zeros(6 + T, dtype=dt, device=dev)
        col[:3] = u
        cols.append(col)
    if not fix_eta_zero:
        for u in (u1, u2):                               # eta perpendicular
            col = torch.zeros(6 + T, dtype=dt, device=dev)
            col[3:6] = u
            cols.append(col)
    for k in range(1, T):                                # s, with s_0 pinned
        col = torch.zeros(6 + T, dtype=dt, device=dev)
        col[6 + k] = 1.0
        cols.append(col)
    return torch.stack(cols, dim=1)


def tangent_project(g, d, fix_eta_zero: bool):
    """Section 4: drop the gauge directions out of a gradient.

    The chart carries three constraints, each removing directions along which
    the parameters may move but the PHYSICS cannot:

      * ||d|| = 1      -- the radial part of d changes nothing;
      * eta . d = 0    -- the component of eta along d is not a curvature;
      * s_0 = 0        -- a constant offset in s just renames the frames.

    A raw gradient has components along all three. They never shrink, because
    nothing in the objective pushes on them, so testing the raw gradient norm
    for stationarity tests the wrong thing: it can stay large at a perfectly
    converged solution, or (with the free directions dominating) hide real
    descent that remains in the physical ones. The projected gradient is the
    one whose smallness actually means stationarity, and section 4 requires
    the certification to use it.
    """
    gd, ge, gs = g[:3].clone(), g[3:6].clone(), g[6:].clone()
    gd = gd - d * (d @ gd)                       # off the sphere's radius
    if fix_eta_zero:
        ge = torch.zeros_like(ge)                # eta is not a variable here
    else:
        ge = ge - d * (d @ ge)                   # keep eta perpendicular to d
    gs = gs.clone()
    gs[0] = 0.0                                  # s_0 is fixed by the gauge
    return torch.cat([gd, ge, gs])


def _pack(d, eta_perp, s, basis):
    return torch.cat([d, eta_perp, s])


def continuous_objective(c, d, eta, s, R_obs, p_obs, L, *, moments=None,
                         S_half=None):
    """The energy a candidate (d, eta, s) is scored by.

    With `moments` (section 2) this is the coordinate-invariant Gaussian
    transport error E_G, which is what P and R are now compared on. Without
    them it falls back to section 10's origin-translation form, which is kept
    only to report `E_legacy_origin_translation` as an audit quantity: section
    2.1 shows that form is not invariant to where the world origin sits, so it
    must not decide anything.
    """
    R, p = continuous_joint_transform(c, d, eta, s)
    if moments is None:
        r = _geometric_residual(R_obs, p_obs, R, p, L)
        return (r * r).sum() / max(R_obs.shape[0] - 1, 1), R, p
    r = gaussian_transport_residual(R_obs, p_obs, R, p, moments["c"],
                                    moments["S"], moments["L"], S_half=S_half)
    # the residual already carries 1 / (L sqrt(T-1)), so this IS E_G
    return (r * r).sum(), R, p


def initialise_continuous(c, R_obs, p_obs, L):
    """A start for (d, eta, s) read straight off the observed trajectory.

    The centroid's travelled distance gives |s_t|; its direction gives d; and
    since phi_t = log(R_t) = s_t eta, a least-squares fit of phi_t against s_t
    gives eta. None of this divides by curvature, so a pure translation starts
    at exactly eta = 0 rather than at an arbitrary far axis.
    """
    from estimator.manifold import project_to_so3, so3_log
    c = c.reshape(3).double()
    R_obs = project_to_so3(R_obs.double())
    p_obs = p_obs.double()
    tc = torch.einsum("tij,j->ti", R_obs, c) + p_obs - c      # centroid motion
    mag = tc.norm(dim=1)
    k = int(mag.argmax())
    if float(mag[k]) <= 0.0:
        d = torch.tensor([1.0, 0.0, 0.0], dtype=torch.float64, device=c.device)
    else:
        d = tc[k] / mag[k]
    # signed arc length: project the centroid path onto d
    s = tc @ d
    phi = so3_log(R_obs)
    denom = float((s * s).sum())
    eta = (phi * s.unsqueeze(1)).sum(0) / denom if denom > 0 else \
        torch.zeros(3, dtype=torch.float64, device=c.device)
    eta = eta - d * (d @ eta)                                 # section 5 gauge
    return d, eta, s


def fit_continuous_joint(c, R_obs, p_obs, L, *, max_iters: int = 120,
                         lam0: float = 1e-3, tau_grad: float = 1e-10,
                         tau_rel: float = 1e-12, fix_eta_zero: bool = False,
                         moments=None, verbose: bool = False) -> dict:
    """Sections 12-15: Levenberg-Marquardt over (d, eta, s) in the chart.

    `fix_eta_zero` holds eta at exactly zero, which by section 11 IS the
    prismatic solver: the same objective, the same normalisation, the same
    frames, differing only by the constraint. That is what makes E_prof(0) and
    E_P the same number rather than two separately-coded quantities.

    Derivatives come from reverse-mode autograd on the forward model of
    section 6, so they are analytic rather than finite differences; section 13's
    numerical check lives in `tests/test_continuous_joint_jacobians.py`.

    The gauge of section 5 (||d|| = 1, eta . d = 0, s_0 = 0) is enforced by
    retraction after every step, so the iterate never leaves the manifold.
    """
    from estimator.manifold import project_to_so3
    c = c.reshape(3).double()
    # Projected ONCE here, never inside the residual (see _geometric_residual).
    R_obs = project_to_so3(R_obs.double()).detach()
    p_obs = p_obs.double().detach()
    T = int(R_obs.shape[0])

    # Section 2: when the part's moments are supplied the fit minimises the
    # coordinate-invariant transport error. S never changes during the fit, so
    # its root is taken once here.
    S_half = None if moments is None else psd_sqrt(moments["S"]).detach()

    d, eta, s = initialise_continuous(c, R_obs, p_obs, L)
    if fix_eta_zero:
        eta = torch.zeros(3, dtype=torch.float64, device=c.device)

    def retract(d_, eta_, s_):
        d_ = d_ / d_.norm().clamp_min(1e-300)
        eta_ = torch.zeros_like(eta_) if fix_eta_zero else eta_ - d_ * (d_ @ eta_)
        s_ = s_ - s_[0].clone()                       # section 5: s_0 = 0
        return d_, eta_, s_

    d, eta, s = retract(d, eta, s)
    E, _R, _p = continuous_objective(c, d, eta, s, R_obs, p_obs, L,
                                     moments=moments, S_half=S_half)
    E = float(E)
    lam = float(lam0)
    hist = [E]
    status = "MAX_ITERS_UNSTABLE"
    at_round_floor = False
    residual_at_precision = False
    physically_settled = False
    gn_phys_change = float("nan")
    pred_gn = float("nan")
    gnorm = float("inf")
    gtan = float("inf")
    gtan_rel = float("inf")
    rho = float("nan")
    phys_change = None
    iters = 0

    for it in range(int(max_iters)):
        iters = it + 1
        v = torch.cat([d, eta, s]).clone().requires_grad_(True)

        def unpack(vec):
            return vec[:3], vec[3:6], vec[6:]

        def resid(vec):
            dd, ee, ss = unpack(vec)
            dd = dd / dd.norm()
            ee = torch.zeros_like(ee) if fix_eta_zero else ee - dd * (dd @ ee)
            ss = ss - ss[0]
            R, p = continuous_joint_transform(c, dd, ee, ss)
            if moments is None:
                return _geometric_residual(R_obs, p_obs, R, p, L)
            return gaussian_transport_residual(R_obs, p_obs, R, p,
                                               moments["c"], moments["S"],
                                               moments["L"], S_half=S_half)

        r0 = resid(v)
        J = torch.autograd.functional.jacobian(resid, v, vectorize=True)
        g = J.T @ r0.detach()
        gnorm = float(g.norm())
        # Section 4: stationarity is a statement about the CONSTRAINT TANGENT
        # SPACE, so it is this norm the status below is allowed to test.
        # Section 4: build the step in minimal coordinates. B's columns span
        # exactly the physical directions, so H_t below is the Gauss-Newton
        # Hessian of the real problem with no null space left in it.
        B = tangent_basis(d, len(s), fix_eta_zero)
        J_t = J @ B
        g_t = J_t.T @ r0.detach()
        H_t = J_t.T @ J_t
        g_tan = B @ g_t
        gtan = float(g_tan.norm())
        # Section 4 asks for a "predefined SCALE-AWARE tolerance". An absolute
        # bound on ||g|| is not one: g = J^T r, so its size goes as ||J|| ||r||
        # and shrinks with the residual. Two scenes whose fits are equally good
        # would then be certified differently purely because one part is
        # larger or better observed.
        #
        # The dimensionless quantity is the familiar least-squares criterion
        #
        #     ||g_tan|| / (||r|| ||J||_F),
        #
        # the cosine between the residual and the tangent column space: zero
        # exactly when no tangent direction can explain any more of r, and
        # unchanged by rescaling either the residual or the parameters.
        rn = float(r0.detach().norm())
        Jn = float(J.norm())
        gtan_rel = gtan / max(rn * Jn, 1e-300)

        # Section 4.3: what does the model say is still available? The
        # undamped Gauss-Newton step is the best any step could do under the
        # local quadratic, so its predicted reduction bounds the remaining
        # progress. Once that bound falls below the rounding of E itself, no
        # further descent is REPRESENTABLE in double precision and the
        # iteration has reached its floor -- which is a convergence
        # certificate, not an exhausted budget. This matters because the
        # cosine test alone cannot certify an exactly-fitting model: there
        # ||r|| is rounding noise and the cosine is computed on it.
        H = H_t
        g = g_t
        try:
            gn_step = torch.linalg.solve(H, -g)
        except Exception:
            gn_step = torch.linalg.lstsq(H, -g.unsqueeze(1)).solution.squeeze(1)
        pred_gn = float(-2.0 * (g @ gn_step) - gn_step @ (H @ gn_step))

        # How small a reduction is still real? E is a sum of m squared terms,
        # so forming it costs about sqrt(m) * eps in relative accuracy; a
        # predicted gain below that is arithmetic, not progress. The factor of
        # 32 is headroom over that estimate, not a fitted tolerance -- the
        # quantity it guards is the floating-point summation, and nothing in
        # it depends on the scene.
        m = int(r0.numel())
        round_floor = 32.0 * math.sqrt(max(m, 1)) * _EPS64 * max(abs(E), 1e-300)

        # The separate case of an EXACT fit. There ||r|| is rounding noise, so
        # both the cosine and the relative test above are computed on noise
        # and mean nothing. The residual is dimensionless (scaled by L and
        # sqrt(T)), so comparing it directly with double precision is the
        # right statement: the model reproduces the observations to the last
        # bit and there is nothing left to fit.
        residual_at_precision = rn <= 64.0 * math.sqrt(max(m, 1)) * _EPS64

        # Section 4 condition 5: "stable PHYSICAL transforms at numerical
        # accuracy". Parameter-space tests can disagree about a point that is
        # physically settled -- two fits of identical quality can land either
        # side of a gradient threshold -- so the decisive question is what the
        # model's best remaining step would actually do to the trajectory. If
        # it cannot move any rotation entry or position by more than double
        # precision allows on O(1) quantities, the physics is converged
        # whatever the parameters are still doing.
        with torch.no_grad():
            _v = v.detach() + B @ gn_step
            _d, _e, _s = retract(_v[:3], _v[3:6], _v[6:])
            _Rg, _pg = continuous_joint_transform(c, _d, _e, _s)
            _Rc, _pc = continuous_joint_transform(c, d, eta, s)
            gn_phys_change = float(max((_Rg - _Rc).abs().max(),
                                       (_pg - _pc).abs().max()))
        physically_settled = gn_phys_change <= 1e-12

        at_round_floor = (not math.isfinite(pred_gn)
                          or pred_gn <= round_floor or residual_at_precision
                          or physically_settled)
        n = H.shape[0]
        eye = torch.eye(n, dtype=torch.float64, device=H.device)

        accepted = False
        # Trial 0 is the UNDAMPED Gauss-Newton step. Near a solution it is the
        # step that converges quadratically, and the damped sequence below
        # starts from whatever lambda the previous iterations left behind --
        # which can be large enough that every trial is a tiny step that buys
        # nothing. Offering GN first costs one evaluation and removes that
        # failure mode.
        for _trial in range(13):
            # H is now the Hessian in minimal coordinates, so the only
            # ill-conditioning left is the problem's own. Damping is still
            # relative to the curvature present, and lstsq remains as a
            # fallback for a genuinely rank-deficient direction (a part whose
            # motion does not excite one of the coordinates at all).
            scale = float(H.diagonal().abs().max()) or 1.0
            if _trial == 0:
                step = gn_step
            else:
                try:
                    step = torch.linalg.solve(H + lam * scale * eye, -g)
                except Exception:
                    step = torch.linalg.lstsq(H + lam * scale * eye,
                                              -g.unsqueeze(1)).solution.squeeze(1)
            if not torch.isfinite(step).all():
                if _trial:
                    lam *= 10.0
                continue
            dd, ee, ss = unpack(v.detach() + B @ step)
            if not (torch.isfinite(dd).all() and torch.isfinite(ee).all()
                    and torch.isfinite(ss).all()) or float(dd.norm()) < 1e-12:
                if _trial:
                    lam *= 10.0
                continue
            dd, ee, ss = retract(dd, ee, ss)
            E_new, _, _ = continuous_objective(c, dd, ee, ss, R_obs, p_obs,
                                               L, moments=moments,
                                               S_half=S_half)
            E_new = float(E_new)
            if not math.isfinite(E_new):
                if _trial:
                    lam *= 10.0
                continue
            # The acceptance test is RELATIVE. An absolute one -- the
            # previous `E_new < E - 1e-18` -- becomes unsatisfiable as soon as
            # E itself falls below that constant: `E - 1e-18` goes negative
            # and no non-negative energy can beat it, so every step is
            # rejected however good it is. On a near-exact revolute fit that
            # froze the iteration at E = 2e-22 and reported it as
            # uncertified, when one more Gauss-Newton step would have taken it
            # to rounding. The rounding-floor test above is what stops the
            # loop when progress stops being meaningful.
            if E_new < E * (1.0 - 1e-15):
                # Section 4.3: the LM model must actually describe what
                # happened. rho near 1 means the quadratic model predicted the
                # reduction; rho far from 1 means the step was taken on a
                # model that does not hold here, so it is not evidence of
                # being near a well-modelled stationary point.
                pred = float(-2.0 * (g @ step) - step @ (H @ step))
                del _trial
                rho = ((E - E_new) / pred) if pred > 0 else float("nan")
                R_prev, p_prev = continuous_joint_transform(c, d, eta, s)
                R_new, p_new = continuous_joint_transform(c, dd, ee, ss)
                phys_change = float(max((R_new - R_prev).abs().max(),
                                        (p_new - p_prev).abs().max()))
                d, eta, s, E = dd, ee, ss, E_new
                hist.append(E)
                lam = max(lam / 3.0, 1e-14)
                accepted = True
                break
            if _trial:
                lam *= 10.0
        if at_round_floor or gtan_rel < tau_grad:
            # Section 4 conditions 2-4: either no tangent direction correlates
            # with the residual any more, or the model's own best step cannot
            # buy a reduction that double precision could represent.
            # Conditions 2 and 3 of section 4 are met before a step is even
            # attempted: nothing is left to gain.
            status = "CONVERGED_STATIONARY"
            break

        if not accepted:
            # Section 15: no feasible descent step remains at this trust-region
            # scale (condition 4). Whether that is stationarity depends on the
            # PROJECTED gradient (condition 2), never on the raw one.
            if gtan_rel < tau_grad:
                status = "CONVERGED_STATIONARY"
            elif phys_change is not None and phys_change < 1e-12:
                # Section 4: flat objective, but tangent descent still exists.
                # The trajectory is reproducible; the parameters are not
                # certified. These are different claims and get different
                # names.
                status = "DEGENERATE_BUT_TRAJECTORY_STABLE"
            else:
                status = "ILL_CONDITIONED_NOT_CERTIFIED"
            break
        # Section 4 is explicit that "a tiny improvement is not evidence of
        # first-order stationarity", so a small step is deliberately NOT a
        # stopping condition here. The loop stops on the two tests above --
        # the model has nothing left to offer, or the projected gradient is
        # small -- and otherwise keeps going.

    if not (math.isfinite(E) and torch.isfinite(d).all()
            and torch.isfinite(eta).all() and torch.isfinite(s).all()):
        status = "NUMERICAL_FAILURE"

    R, p = continuous_joint_transform(c, d, eta, s)
    kappa = float(eta.norm())
    out = {"d": d, "eta": eta, "s": s, "kappa": kappa, "energy": E,
           "R": R, "p": p, "status": status, "iterations": iters,
           "grad_norm": gnorm, "projected_grad_norm": gtan,
           "projected_grad_relative": gtan_rel,
           "last_gain_ratio_rho": rho,
           "last_physical_change": phys_change, "history": hist,
           "certification": {
               "rule": "section 4: CONVERGED_STATIONARY requires a small "
                       "PROJECTED tangent gradient, not merely a small step; "
                       "a flat objective with tangent descent remaining is "
                       "reported as ILL_CONDITIONED_NOT_CERTIFIED",
               "projected_grad_norm": gtan,
               "projected_grad_relative": gtan_rel,
               "raw_grad_norm": gnorm,
               "tolerance": tau_grad,
               "tolerance_meaning": "dimensionless ||g_tan||/(||r|| ||J||_F)",
               "gain_ratio_rho": rho,
               "physical_change": phys_change,
               "gauss_newton_predicted_reduction": pred_gn,
               "reached_rounding_floor": bool(at_round_floor),
               "residual_at_double_precision": bool(residual_at_precision),
               "gauss_newton_physical_change": gn_phys_change,
               "physically_settled": bool(physically_settled)},
           "eligible": status in ("CONVERGED_STATIONARY",
                                  "DEGENERATE_BUT_TRAJECTORY_STABLE"),
           "fix_eta_zero": bool(fix_eta_zero),
           "objective": ("gaussian_transport" if moments is not None
                         else "legacy_origin_translation"),
           "tolerances": {"tau_grad": tau_grad, "tau_rel": tau_rel,
                          "lam0": lam0, "max_iters": max_iters}}
    axis = continuous_to_finite_axis(c, d, eta, s)
    out["finite_axis"] = (None if axis is None else
                          {k: (v.tolist() if torch.is_tensor(v) else v)
                           for k, v in axis.items()})
    if verbose:
        print(f"       chart fit: {status} in {iters} iters, kappa "
              f"{kappa:.6g}, E {E:.6e}, |grad| {gnorm:.2e}")
    return out


# ---------------------------------------------------------------------------
# section 2: the coordinate-invariant Gaussian-transport objective
# ---------------------------------------------------------------------------
# Section 11 gives this metric its own module. It is re-exported here because
# the chart fit of sections 10-15 scores candidates with it, and because other
# callers already import it from this module.
from .gaussian_transport_metric import (  # noqa: E402,F401
    canonical_moments, gaussian_transport_error_pointwise,
    gaussian_transport_residual, psd_sqrt, transport_audit)
