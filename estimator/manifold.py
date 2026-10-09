"""SO(3) manifold primitives for the sparse alpha-aware error-state IEKF.

Specification: ``CODEX_IMPLEMENT_SPARSE_ALPHA_AWARE_IEKF.md`` section 4.

Convention, used by *every* module of the estimator package: the rotational
error state acts on the **left**,

    R+ = Exp(hat(dphi)) @ R-,            dphi = Log(R_a @ R_b^T),

so the Jacobian that appears in the process model and in the covariance blocks
is the **left** Jacobian ``J_l``.  Switching to the right-multiplied convention
silently transposes/conjugates every rotation block of P and of H, which does
not blow up loudly -- it just makes the filter inconsistent -- so the choice is
pinned here and nothing in this file exposes the right Jacobian.

Autograd policy (spec section 36): these are value-only routines.  No gradient
is ever taken through them in the production estimator, which is why the
small-angle handling optimises for *value* accuracy and for the absence of
NaNs rather than for differentiability at ``theta = 0``.  A single NaN in one
Gaussian propagates through the matrix-free normal operator and destroys the
whole PCG solve, so every branch below is evaluated on a sanitised argument.

Everything is batched over arbitrary leading dimensions; the canonical shape in
the estimator is a leading ``[N]`` over Gaussians.
"""

from __future__ import annotations

import torch
from torch import Tensor

__all__ = [
    "so3_hat",
    "so3_vee",
    "so3_exp",
    "so3_log",
    "so3_left_jacobian",
    "so3_left_jacobian_inverse",
    "so3_adjoint",
    "is_rotation",
    "project_to_so3",
    "rotation_angle",
    "small_angle_cutoff_sq",
]


# ---------------------------------------------------------------------------
# argument checking / numerical constants
# ---------------------------------------------------------------------------
def _as_float_tensor(value: Tensor, where: str) -> Tensor:
    tensor = torch.as_tensor(value)
    if not tensor.is_floating_point():
        raise TypeError(f"{where}: expected a floating-point tensor, got dtype {tensor.dtype}")
    return tensor


def _check_vec3(phi: Tensor, where: str) -> None:
    if phi.ndim < 1 or phi.shape[-1] != 3:
        raise ValueError(f"{where}: expected shape (..., 3), got {tuple(phi.shape)}")


def _check_mat3(matrix: Tensor, where: str) -> None:
    if matrix.ndim < 2 or matrix.shape[-2:] != (3, 3):
        raise ValueError(f"{where}: expected shape (..., 3, 3), got {tuple(matrix.shape)}")


def small_angle_cutoff_sq(dtype: torch.dtype) -> float:
    """``theta**2`` below which the Taylor branches are taken.

    The cutoff is ``theta < eps**(1/4)`` (so ``theta**2 < eps**(1/2)``).  Below
    it the first dropped series term is O(theta^6) << eps, while the closed
    forms -- which divide by ``theta**3`` after a ``theta - sin(theta)``
    cancellation -- have already lost most of their significant digits.  Making
    it dtype-aware matters: the same fixed 1e-4 that is generous in float64 is
    three orders of magnitude *inside* the float32 cancellation zone.
    """
    return float(torch.finfo(dtype).eps) ** 0.5


def _eps(dtype: torch.dtype) -> float:
    return float(torch.finfo(dtype).eps)


# ---------------------------------------------------------------------------
# hat / vee
# ---------------------------------------------------------------------------
def so3_hat(phi: Tensor) -> Tensor:
    """``(..., 3) -> (..., 3, 3)`` skew-symmetric generator ``[phi]_x``."""
    phi = _as_float_tensor(phi, "so3_hat(phi)")
    _check_vec3(phi, "so3_hat(phi)")
    x, y, z = phi.unbind(-1)
    zero = torch.zeros_like(x)
    return torch.stack(
        (
            zero, -z, y,
            z, zero, -x,
            -y, x, zero,
        ),
        dim=-1,
    ).reshape(phi.shape[:-1] + (3, 3))


def so3_vee(skew: Tensor) -> Tensor:
    """``(..., 3, 3) -> (..., 3)`` inverse of :func:`so3_hat`.

    The antisymmetric part is read out rather than three raw entries: callers
    hand us matrices that are only skew up to round-off (``R - R^T`` after a
    long chain of products), and the symmetric part is exactly the component
    that is *not* a tangent vector.  For an exactly skew input this is bitwise
    identical to reading the three entries directly.
    """
    skew = _as_float_tensor(skew, "so3_vee(S)")
    _check_mat3(skew, "so3_vee(S)")
    anti = 0.5 * (skew - skew.transpose(-1, -2))
    return torch.stack((anti[..., 2, 1], anti[..., 0, 2], anti[..., 1, 0]), dim=-1)


# ---------------------------------------------------------------------------
# exponential
# ---------------------------------------------------------------------------
def so3_exp(phi: Tensor) -> Tensor:
    """``(..., 3) -> (..., 3, 3)`` Rodrigues exponential ``Exp(hat(phi))``."""
    phi = _as_float_tensor(phi, "so3_exp(phi)")
    _check_vec3(phi, "so3_exp(phi)")

    theta_sq = phi.square().sum(-1)
    small = theta_sq < small_angle_cutoff_sq(phi.dtype)

    # Both branches are evaluated on a SAFE argument (theta == 1 on the rows the
    # other branch owns) so that no 0/0 is ever formed, not even in a value that
    # torch.where would later discard.
    theta_sq_safe = torch.where(small, torch.ones_like(theta_sq), theta_sq)
    theta = torch.sqrt(theta_sq_safe)
    sin_half = torch.sin(0.5 * theta)

    # (1 - cos t)/t^2 is written as 2 sin^2(t/2)/t^2: the literal form loses all
    # precision around t ~ 1e-4 in float64 (1 - cos t is ~5e-9 with ~1e-16 of
    # absolute rounding error) and is simply wrong in float32.
    coef_a_big = torch.sin(theta) / theta
    coef_b_big = 2.0 * sin_half * sin_half / theta_sq_safe

    t2 = torch.where(small, theta_sq, torch.zeros_like(theta_sq))
    coef_a_small = 1.0 + t2 * (-1.0 / 6.0 + t2 * (1.0 / 120.0 - t2 / 5040.0))
    coef_b_small = 0.5 + t2 * (-1.0 / 24.0 + t2 * (1.0 / 720.0 - t2 / 40320.0))

    coef_a = torch.where(small, coef_a_small, coef_a_big)[..., None, None]
    coef_b = torch.where(small, coef_b_small, coef_b_big)[..., None, None]

    k = so3_hat(phi)
    eye = torch.eye(3, dtype=phi.dtype, device=phi.device).expand(k.shape)
    # phi == 0 gives coef_a == 1, coef_b == 0.5 and k == 0 exactly, hence R == I
    # bitwise.  state.boxplus relies on that: a zero delta must be a no-op.
    return eye + coef_a * k + coef_b * (k @ k)


# ---------------------------------------------------------------------------
# logarithm
# ---------------------------------------------------------------------------
def rotation_angle(rotation: Tensor) -> Tensor:
    """``(..., 3, 3) -> (...)`` principal rotation angle in ``[0, pi]``.

    Computed as ``atan2(|vee(R - R^T)|/2, (tr R - 1)/2)`` instead of
    ``acos((tr R - 1)/2)``.  acos is catastrophically ill-conditioned at both
    ends of the range we care about: near 0 the trace is ``3 - theta^2`` and
    the subtraction destroys every significant digit of theta, and near pi the
    derivative of acos is infinite.  atan2 of the (sin, cos) pair is accurate
    over the whole interval.
    """
    rotation = _as_float_tensor(rotation, "rotation_angle(R)")
    _check_mat3(rotation, "rotation_angle(R)")
    sin_axis = so3_vee(rotation)  # = sin(theta) * axis
    sin_theta = torch.linalg.norm(sin_axis, dim=-1)
    trace = rotation[..., 0, 0] + rotation[..., 1, 1] + rotation[..., 2, 2]
    cos_theta = (0.5 * (trace - 1.0)).clamp(-1.0, 1.0)
    return torch.atan2(sin_theta, cos_theta)


def so3_log(rotation: Tensor) -> Tensor:
    """``(..., 3, 3) -> (..., 3)`` principal logarithm, ``|result| <= pi``.

    Three branches, because no single closed form is usable over the whole
    range:

    * ``theta -> 0``   : ``theta/sin(theta)`` by Taylor series (the quotient is
      1 at the limit but 0/0 numerically);
    * ``theta -> pi``  : the antisymmetric part ``vee(R - R^T) = 2 sin(theta) u``
      vanishes, so the axis must be recovered from the SYMMETRIC part via
      ``u u^T = (sym(R) - cos(theta) I)/(1 - cos(theta))``.  Any formula that
      divides by ``sin(theta)`` (or that uses asin on the off-diagonals) returns
      noise here, and at exactly pi it returns 0/0;
    * otherwise the direct ``theta/(2 sin theta) * vee(R - R^T)``.

    At exactly ``theta = pi`` the axis sign is genuinely ambiguous (``Exp(pi u)
    == Exp(-pi u)``).  We return the deterministic representative whose
    largest-magnitude component is positive; callers must compare rotations,
    not log vectors, there.

    Angles beyond pi wrap, as the principal logarithm must: ``Log(Exp(phi))``
    for ``|phi|`` just under ``2 pi`` returns ``(|phi| - 2 pi) * u``, and
    ``|phi| = 2 pi`` gives exactly zero -- the input is the identity.  This is
    correct, not a failure, so the round-trip identity to assert past pi is on
    ``R``, never on ``phi``.
    """
    rotation = _as_float_tensor(rotation, "so3_log(R)")
    _check_mat3(rotation, "so3_log(R)")
    dtype = rotation.dtype
    eps = _eps(dtype)

    sin_axis = so3_vee(rotation)  # = sin(theta) * axis
    sin_theta = torch.linalg.norm(sin_axis, dim=-1)
    trace = rotation[..., 0, 0] + rotation[..., 1, 1] + rotation[..., 2, 2]
    cos_theta = (0.5 * (trace - 1.0)).clamp(-1.0, 1.0)
    theta = torch.atan2(sin_theta, cos_theta)
    theta_sq = theta.square()

    small = theta_sq < small_angle_cutoff_sq(dtype)
    # The pi branch is selected on sin(theta), which is what actually controls
    # the conditioning of the axis: the relative error of `axis` from the
    # antisymmetric part is ~eps/sin(theta).  cos < 0 disambiguates it from the
    # near-zero rows, which have a small sine as well.
    near_pi = (cos_theta < 0.0) & (sin_theta < small_angle_cutoff_sq(dtype) ** 0.5)
    generic = ~small & ~near_pi

    # --- generic branch -----------------------------------------------------
    sin_safe = torch.where(generic, sin_theta, torch.ones_like(sin_theta))
    scale_generic = theta / sin_safe

    # --- small branch: theta/sin(theta) = 1 + t^2/6 + 7 t^4/360 + 31 t^6/15120
    t2 = torch.where(small, theta_sq, torch.zeros_like(theta_sq))
    scale_small = 1.0 + t2 * (1.0 / 6.0 + t2 * (7.0 / 360.0 + t2 * (31.0 / 15120.0)))

    scale = torch.where(small, scale_small, scale_generic)[..., None]
    phi_axis_part = scale * sin_axis

    # --- near-pi branch: symmetric part --------------------------------------
    sym = 0.5 * (rotation + rotation.transpose(-1, -2))
    eye = torch.eye(3, dtype=dtype, device=rotation.device).expand(sym.shape)
    # 1 - cos(theta) ~ 2 on the rows this branch owns; forcing 1 elsewhere keeps
    # the discarded rows finite instead of 0/0 at the identity.
    denom = torch.where(near_pi, 1.0 - cos_theta, torch.ones_like(cos_theta))
    outer = (sym - cos_theta[..., None, None] * eye) / denom[..., None, None]

    diag = torch.diagonal(outer, dim1=-2, dim2=-1)
    pivot = diag.argmax(dim=-1)  # for a unit axis max_k u_k^2 >= 1/3: never tiny
    pivot_col = torch.gather(
        outer, -1, pivot[..., None, None].expand(outer.shape[:-1] + (1,))
    ).squeeze(-1)
    pivot_val = torch.sqrt(torch.gather(diag, -1, pivot[..., None]).clamp_min(0.0))
    axis = pivot_col / pivot_val.clamp_min(eps)
    axis = axis / torch.linalg.norm(axis, dim=-1, keepdim=True).clamp_min(eps)

    # sin(theta) > 0 on (0, pi), so vee(R - R^T) and the axis must agree in
    # sign.  Only flip on a *meaningful* negative projection: at exactly pi the
    # antisymmetric part is pure round-off and flipping on its sign would make
    # the output non-deterministic for bitwise-identical inputs.
    projection = (axis * sin_axis).sum(-1, keepdim=True)
    sign = torch.where(projection < -4.0 * eps, -torch.ones_like(projection), torch.ones_like(projection))
    phi_pi = (sign * axis) * theta[..., None]

    return torch.where(near_pi[..., None], phi_pi, phi_axis_part)


# ---------------------------------------------------------------------------
# left Jacobian and its inverse
# ---------------------------------------------------------------------------
def so3_left_jacobian(phi: Tensor) -> Tensor:
    """``(..., 3) -> (..., 3, 3)``.

    ``J_l(phi) = I + ((1 - cos t)/t^2) [phi]_x + ((t - sin t)/t^3) [phi]_x^2``,
    defined by ``Exp(phi + dphi) ~= Exp(J_l(phi) dphi) Exp(phi)`` to first order
    in ``dphi``.  This is the Jacobian the constant-twist process model of spec
    section 5 needs for ``dphi_{k+1} ~= Ad_Exp(Omega) dphi_k + J_l(Omega) dt
    domega_k``.
    """
    phi = _as_float_tensor(phi, "so3_left_jacobian(phi)")
    _check_vec3(phi, "so3_left_jacobian(phi)")

    theta_sq = phi.square().sum(-1)
    small = theta_sq < small_angle_cutoff_sq(phi.dtype)
    theta_sq_safe = torch.where(small, torch.ones_like(theta_sq), theta_sq)
    theta = torch.sqrt(theta_sq_safe)
    sin_half = torch.sin(0.5 * theta)

    coef_b_big = 2.0 * sin_half * sin_half / theta_sq_safe          # (1 - cos t)/t^2
    coef_c_big = (theta - torch.sin(theta)) / (theta_sq_safe * theta)  # (t - sin t)/t^3

    t2 = torch.where(small, theta_sq, torch.zeros_like(theta_sq))
    coef_b_small = 0.5 + t2 * (-1.0 / 24.0 + t2 * (1.0 / 720.0 - t2 / 40320.0))
    coef_c_small = 1.0 / 6.0 + t2 * (-1.0 / 120.0 + t2 * (1.0 / 5040.0 - t2 / 362880.0))

    coef_b = torch.where(small, coef_b_small, coef_b_big)[..., None, None]
    coef_c = torch.where(small, coef_c_small, coef_c_big)[..., None, None]

    k = so3_hat(phi)
    eye = torch.eye(3, dtype=phi.dtype, device=phi.device).expand(k.shape)
    return eye + coef_b * k + coef_c * (k @ k)


def so3_left_jacobian_inverse(phi: Tensor) -> Tensor:
    """``(..., 3) -> (..., 3, 3)`` exact inverse of :func:`so3_left_jacobian`.

    ``J_l^{-1}(phi) = I - 0.5 [phi]_x + g(t) [phi]_x^2`` with
    ``g(t) = (1 - (t/2) cot(t/2)) / t^2``, series ``1/12 + t^2/720 +
    t^4/30240 + t^6/1209600``.  The cot form is used instead of
    ``1/t^2 - (1 + cos t)/(2 t sin t)``: they are algebraically identical but
    the latter divides by ``sin t``, which vanishes at ``t = pi`` where ``J_l``
    is perfectly invertible (``det J_l = 2(1 - cos t)/t^2 = 4/pi^2``).

    ``J_l`` is genuinely singular at ``t = 2 pi k``.  Rather than emit a NaN
    that would silently poison every subsequent PCG iteration, the vanishing
    ``sin(t/2)`` is clamped away from zero with its sign preserved, so a
    degenerate input yields a large finite matrix that the caller's residual
    check can see.
    """
    phi = _as_float_tensor(phi, "so3_left_jacobian_inverse(phi)")
    _check_vec3(phi, "so3_left_jacobian_inverse(phi)")
    eps = _eps(phi.dtype)

    theta_sq = phi.square().sum(-1)
    small = theta_sq < small_angle_cutoff_sq(phi.dtype)
    theta_sq_safe = torch.where(small, torch.ones_like(theta_sq), theta_sq)
    theta = torch.sqrt(theta_sq_safe)

    half = 0.5 * theta
    sin_half = torch.sin(half)
    sign_half = torch.where(sin_half < 0.0, -torch.ones_like(sin_half), torch.ones_like(sin_half))
    sin_half_safe = torch.where(sin_half.abs() < eps, sign_half * eps, sin_half)
    coef_g_big = (1.0 - half * torch.cos(half) / sin_half_safe) / theta_sq_safe

    t2 = torch.where(small, theta_sq, torch.zeros_like(theta_sq))
    coef_g_small = 1.0 / 12.0 + t2 * (1.0 / 720.0 + t2 * (1.0 / 30240.0 + t2 / 1209600.0))

    coef_g = torch.where(small, coef_g_small, coef_g_big)[..., None, None]

    k = so3_hat(phi)
    eye = torch.eye(3, dtype=phi.dtype, device=phi.device).expand(k.shape)
    return eye - 0.5 * k + coef_g * (k @ k)


# ---------------------------------------------------------------------------
# misc helpers the rest of the estimator needs
# ---------------------------------------------------------------------------
def so3_adjoint(rotation: Tensor) -> Tensor:
    """``Ad_R`` for SO(3), which is ``R`` itself.

    Trivial, but named: spec section 5 writes the rotational error propagation
    as ``Ad_Exp(Omega) dphi``, and having the symbol in the code is what keeps
    somebody from "simplifying" it into a transpose when they port the formula.
    """
    rotation = _as_float_tensor(rotation, "so3_adjoint(R)")
    _check_mat3(rotation, "so3_adjoint(R)")
    return rotation


def is_rotation(rotation: Tensor, tol: float | None = None) -> Tensor:
    """``(..., 3, 3) -> (...)`` boolean: orthonormal with ``det = +1``."""
    rotation = _as_float_tensor(rotation, "is_rotation(R)")
    _check_mat3(rotation, "is_rotation(R)")
    if tol is None:
        tol = 1.0e4 * _eps(rotation.dtype)
    eye = torch.eye(3, dtype=rotation.dtype, device=rotation.device).expand(rotation.shape)
    orthonormal = (rotation @ rotation.transpose(-1, -2) - eye).abs().amax(dim=(-2, -1)) <= tol
    # A reflection also satisfies R R^T = I, so the determinant sign is a
    # separate, non-redundant test.
    proper = (torch.linalg.det(rotation) - 1.0).abs() <= tol
    return orthonormal & proper


def project_to_so3(matrix: Tensor) -> Tensor:
    """Nearest rotation in Frobenius norm, via SVD with a reflection fix.

    Needed because a long chain of ``Exp(dphi) @ R`` products drifts off SO(3)
    at ~sqrt(#updates) * eps; the estimator calls this periodically rather than
    letting the drift grow until the orthonormality guard in ``state.py`` fires.
    """
    matrix = _as_float_tensor(matrix, "project_to_so3(M)")
    _check_mat3(matrix, "project_to_so3(M)")
    # linalg.svd has no half kernels; promote, then restore the caller's dtype.
    work = matrix if matrix.dtype in (torch.float32, torch.float64) else matrix.float()
    u, _, vh = torch.linalg.svd(work)
    det = torch.linalg.det(u @ vh)
    # Flip the last singular direction when the polar factor is a reflection,
    # otherwise det would come out -1 and the result would not be a rotation.
    flip = torch.ones(u.shape[:-1], dtype=u.dtype, device=u.device)
    flip[..., -1] = torch.where(det < 0.0, -torch.ones_like(det), torch.ones_like(det))
    return ((u * flip[..., None, :]) @ vh).to(matrix.dtype)
