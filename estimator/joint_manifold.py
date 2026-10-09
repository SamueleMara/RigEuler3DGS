"""Sections 8-13, 17, 18: the constrained joint-geometry manifold.

CODEX_IMPLEMENT_PHASE2_RIGID_SCREW_IEKF.md.

One screw per moving part, shared by every member Gaussian and every frame:

    T(q) = Exp_SE3(q xi^),     xi = [v; omega].

CONVENTION. Phase-1's `boxplus` is ``R+ = Exp(hat(dphi)) @ R`` -- a LEFT
multiplication, so `dphi` is a spatial (world-frame) rotation error. Everything
here is written in that same spatial convention: `xi` is a spatial twist and
geometry perturbations compose on the left,

    Exp((eta + delta_eta)^) = Exp((J_l(eta) delta_eta)^) Exp(eta^).

Two constraints define the family (sections 9, 10):

    zero pitch   u . omega = 0          -- a hinge or a slide, never a helix
    gauge        |u|^2 + |omega|^2 = 1  -- q and xi cannot both be free

and they are held as STATE CONSTRAINTS, never as penalty residuals. The
dimensionless generator `zeta = [u, omega]` with `u = v / L` (section 11) is
what makes "orthonormal tangent basis" meaningful: `v` is a length and `omega`
is not, so a basis built directly on `[v, omega]` would mix units and its
singular values would depend on whether the scene is measured in metres.

Section 12 is the reason this parameterisation exists. At a prismatic state
`omega = 0`, `|u| = 1`, and the constraint Jacobian still has rank 2, so the
manifold is regular there -- no axis at infinity, no division by `|omega|`.
"""

from __future__ import annotations

import math

import torch
from torch import Tensor

from .manifold import so3_exp, so3_hat, so3_left_jacobian, so3_vee

# Below this angle the closed forms lose significance to cancellation and the
# Taylor series is both faster and more accurate. Chosen as eps^(1/4), the
# standard balance point for a series truncated after the quadratic term.
SMALL_ANGLE = float(torch.finfo(torch.float64).eps) ** 0.25


def constraint_values(zeta: Tensor) -> Tensor:
    """c(zeta) = [u.omega, |u|^2 + |omega|^2 - 1]; zero on the manifold."""
    u, w = zeta[..., :3], zeta[..., 3:]
    return torch.stack([(u * w).sum(-1),
                        (u * u).sum(-1) + (w * w).sum(-1) - 1.0], dim=-1)


def constraint_jacobian(zeta: Tensor) -> Tensor:
    """J_c = [[omega^T, u^T], [2u^T, 2omega^T]], shape [..., 2, 6]."""
    u, w = zeta[..., :3], zeta[..., 3:]
    top = torch.cat([w, u], dim=-1)
    bot = 2.0 * torch.cat([u, w], dim=-1)
    return torch.stack([top, bot], dim=-2)


def tangent_basis(zeta: Tensor) -> Tensor:
    """B_zeta in R^{6x4}, orthonormal, with J_c B = 0 (section 11).

    From the SVD of J_c: the right singular vectors beyond its rank span the
    null space exactly. At a prismatic state J_c is
    ``[[0, u^T], [2u^T, 0]]`` with rank 2, so four directions survive -- two
    that turn the slide direction and two angular ones perpendicular to it.
    That is section 12's statement that P is an ordinary interior point.
    """
    Jc = constraint_jacobian(zeta.reshape(-1, 6)).reshape(2, 6).double()
    _u, _s, vh = torch.linalg.svd(Jc, full_matrices=True)
    return vh[2:].T.contiguous()                    # [6,4], orthonormal


def scale_matrix(length: float, device=None) -> Tensor:
    """S_L = diag(L, L, L, 1, 1, 1): dimensionless zeta -> physical xi."""
    s = torch.ones(6, dtype=torch.float64, device=device)
    s[:3] = float(length)
    return torch.diag(s)


def zeta_to_xi(zeta: Tensor, length: float) -> Tensor:
    """xi = [L u; omega]."""
    out = zeta.clone().double()
    out[..., :3] = out[..., :3] * float(length)
    return out


def project_to_manifold(zeta: Tensor, iters: int = 4) -> Tensor:
    """Nearest point satisfying both constraints, by constrained Newton.

    The obvious implementation -- Gram-Schmidt `u <- u - omega (u.omega)/|omega|^2`
    then renormalise -- is WRONG at the prismatic state, and wrong in a way
    that only a finite-difference check catches. Moving along a tangent
    direction from `omega = 0` leaves `u.omega = O(h^2)` (the tangent basis
    makes the first-order term vanish exactly), but `|omega|^2` is also
    `O(h^2)`, so their ratio is `O(1)` and the "correction" shifts `u` by an
    `O(h)` amount. The retraction then disagrees with its own basis at first
    order, which showed up as a relative JVP error of 4.8 against finite
    differences on prismatic data while revolute data passed at 1e-10.

    Newton on the constraints instead,

        zeta <- zeta - J_c^+ c(zeta),

    divides by nothing that vanishes: `J_c` keeps rank 2 at `omega = 0`
    (section 12), so its pseudo-inverse is well conditioned exactly where the
    Gram-Schmidt form breaks down. This is section 17's requirement 5 -- no
    division by `|omega|` on the P path -- taken literally.
    """
    z = zeta.double().reshape(-1, 6).clone()
    for _ in range(int(iters)):
        c = constraint_values(z)                       # [n,2]
        if float(c.abs().max()) < 1e-15:
            break
        Jc = constraint_jacobian(z)                    # [n,2,6]
        JJt = Jc @ Jc.transpose(-1, -2)                # [n,2,2]
        lam = torch.linalg.solve(
            JJt + 1e-300 * torch.eye(2, dtype=torch.float64,
                                     device=z.device), c.unsqueeze(-1))
        z = z - torch.einsum("nji,njk->nik", Jc, lam).squeeze(-1)
    return z.reshape(zeta.shape)


def retract_joint_geometry(zeta: Tensor, delta_theta: Tensor,
                           basis: Tensor | None = None) -> tuple:
    """Section 17: zeta (+) B delta_theta, back onto the manifold.

    Returns ``(zeta_new, reset_jacobian)`` where the reset Jacobian is
    ``d delta_theta_new / d delta_theta_old`` for the NEW basis -- section 16
    needs it to carry the covariance across a change of tangent frame.

    The retraction is "move in the ambient space along B, then project", whose
    first-order behaviour is exactly B (requirement 4) because the projection
    is the identity to first order on the tangent space.
    """
    z = zeta.double().reshape(6)
    B = tangent_basis(z) if basis is None else basis.double()
    dz = B @ delta_theta.double().reshape(-1)
    z_new = project_to_manifold(z + dz)
    B_new = tangent_basis(z_new)
    # The change of tangent coordinates. Dropping it -- keeping the old
    # covariance in the new basis -- silently rotates the uncertainty relative
    # to the state it describes, which section 16 forbids.
    #
    # This is the FIRST-ORDER transport, and deliberately so. The exact
    # derivative of the retraction differs from it by O(|delta|^2) (measured:
    # 0.83 |delta|^2, so 8e-5 at a step of 0.01), because the projection's own
    # derivative picks up a term proportional to the off-manifold offset. An
    # error-state covariance is itself only defined to first order, so
    # transporting it with anything else would be claiming an accuracy the
    # representation does not have. The orthogonality of this form is also
    # what guarantees the transported covariance stays positive definite.
    G = B_new.T @ B
    return z_new, G


# ---------------------------------------------------------------------------
# section 18: a stable SE(3) exponential and its left Jacobian
# ---------------------------------------------------------------------------
def _se3_coeffs(theta: Tensor):
    """Series-safe (A, B, C) for the SO(3) exponential and left Jacobian."""
    t2 = theta * theta
    small = theta < SMALL_ANGLE
    a = torch.where(small, 1.0 - t2 / 6.0 + t2 * t2 / 120.0,
                    torch.sin(theta) / theta.clamp_min(1e-300))
    b = torch.where(small, 0.5 - t2 / 24.0 + t2 * t2 / 720.0,
                    (1.0 - torch.cos(theta)) / t2.clamp_min(1e-300))
    c = torch.where(small, 1.0 / 6.0 - t2 / 120.0 + t2 * t2 / 5040.0,
                    (1.0 - a) / t2.clamp_min(1e-300))
    return a, b, c


def se3_exp(eta: Tensor) -> tuple[Tensor, Tensor]:
    """Exp_SE3(eta^) -> (R, t) for eta = [rho; phi], spatial convention.

    At exactly ``phi = 0`` this returns ``R = I`` bitwise and ``t = rho``
    (section 18): the series branch gives A = 1, B = 1/2, C = 1/6 and
    ``hat(0) = 0``, so no angular epsilon is ever introduced to keep a
    prismatic state alive.
    """
    eta = eta.double().reshape(-1, 6)
    rho, phi = eta[:, :3], eta[:, 3:]
    theta = phi.norm(dim=-1)
    K = so3_hat(phi)
    a, b, c = _se3_coeffs(theta)
    eye = torch.eye(3, dtype=torch.float64,
                    device=eta.device).expand(eta.shape[0], 3, 3)
    R = eye + a[:, None, None] * K + b[:, None, None] * (K @ K)
    Jl = eye + b[:, None, None] * K + c[:, None, None] * (K @ K)
    t = torch.einsum("nij,nj->ni", Jl, rho)
    return R, t


def se3_left_jacobian(eta: Tensor) -> Tensor:
    """The 6x6 SE(3) left Jacobian [[Jl, Q], [0, Jl]] (Barfoot).

    `Q` is what makes a geometry perturbation exact. Perturbing the twist
    moves the translation twice over: directly through `rho`, and indirectly
    because the rotation that carries `rho` has itself changed. Dropping `Q`
    and using the block-diagonal `Jl` alone leaves a first-order error that a
    finite-difference check exposes immediately on any part with a non-trivial
    rotation -- which is why section 39's J8 exists.
    """
    eta = eta.double().reshape(-1, 6)
    rho, phi = eta[:, :3], eta[:, 3:]
    n = eta.shape[0]
    theta = phi.norm(dim=-1)
    t2 = theta * theta
    small = theta < SMALL_ANGLE
    P_ = so3_hat(phi)
    R_ = so3_hat(rho)
    eye = torch.eye(3, dtype=torch.float64,
                    device=eta.device).expand(n, 3, 3)
    _a, b, c = _se3_coeffs(theta)
    Jl = eye + b[:, None, None] * P_ + c[:, None, None] * (P_ @ P_)

    st, ct = torch.sin(theta), torch.cos(theta)
    th3 = (theta ** 3).clamp_min(1e-300)
    th4 = (theta ** 4).clamp_min(1e-300)
    th5 = (theta ** 5).clamp_min(1e-300)
    c1 = torch.where(small, 1.0 / 6.0 - t2 / 120.0, (theta - st) / th3)
    c2 = torch.where(small, 1.0 / 24.0 - t2 / 720.0,
                     (t2 + 2.0 * ct - 2.0) / (2.0 * th4))
    c3 = torch.where(small, 1.0 / 120.0 - t2 / 2520.0,
                     (2.0 * theta - 3.0 * st + theta * ct) / (2.0 * th5))

    PR, RP = P_ @ R_, R_ @ P_
    PRP = P_ @ R_ @ P_
    PP = P_ @ P_
    Q = (0.5 * R_
         + c1[:, None, None] * (PR + RP + PRP)
         + c2[:, None, None] * (PP @ R_ + R_ @ PP - 3.0 * PRP)
         + c3[:, None, None] * (PRP @ P_ + P_ @ PRP))
    out = torch.zeros((n, 6, 6), dtype=torch.float64, device=eta.device)
    out[:, :3, :3] = Jl
    out[:, :3, 3:] = Q
    out[:, 3:, 3:] = Jl
    return out


# ---------------------------------------------------------------------------
# section 13: physical geometry, as a derived output only
# ---------------------------------------------------------------------------
def prismatic_direction(zeta: Tensor, length: float) -> Tensor:
    """e_P = v / |v| (section 13.1)."""
    v = zeta.double().reshape(6)[:3] * float(length)
    return v / v.norm().clamp_min(1e-300)


def revolute_axis(zeta: Tensor, length: float, centroid: Tensor,
                  min_omega: float = 0.0) -> dict | None:
    """Axis direction and its closest point to the COM (section 13.2).

    Returns None when the angular block is zero, because then there IS no
    axis -- not because a threshold was crossed. Section 12 forbids keeping a
    prismatic state alive by dividing here; axis extraction is a derived
    output that is simply skipped when R geometry is not resolved.
    """
    z = zeta.double().reshape(6)
    v, w = z[:3] * float(length), z[3:]
    nw = float(w.norm())
    if nw <= min_omega:
        return None
    e = w / nw
    c0 = centroid.double().reshape(3)
    # The stored twist is spatial (about the world origin): v = -omega x a.
    # Referring it to the COM first keeps the two conventions from mixing,
    # which section 13.2 warns about explicitly.
    u_c = v + torch.linalg.cross(w, c0)
    a = c0 + torch.linalg.cross(w, u_c) / (nw * nw)
    return {"axis": e, "axis_point": a, "omega_norm": nw,
            "pitch": float((w @ v) / (nw * nw)),
            "gauge_residual": float(e @ (a - c0))}


# ---------------------------------------------------------------------------
# The SE(3) logarithm, in the same [rho; phi] spatial convention as `se3_exp`.
#
# Joint reduction (CODEX_IMPLEMENT_RIGID_SE3_PHASE2..., sections 25, 29) reads
# finite relative transforms off the recovered rigid trajectory, which needs
# the inverse of `se3_exp` and nothing else. It lives here so that one file
# owns the convention: `se3_log(*se3_exp(eta)) == eta`.
# ---------------------------------------------------------------------------

def se3_log(R: Tensor, t: Tensor) -> Tensor:
    """Log_SE3(R, t) -> eta = [rho; phi] with `rho = J_l(phi)^-1 t`.

    At exactly `R = I` this returns `[t, 0]` bitwise: `so3_log` gives zero and
    the inverse left Jacobian series starts at the identity, so a pure
    translation survives the round trip without acquiring an angular epsilon.
    """
    from .manifold import so3_log, so3_left_jacobian_inverse
    R = R.double().reshape(-1, 3, 3)
    t = t.double().reshape(-1, 3)
    phi = so3_log(R)
    rho = torch.einsum("nij,nj->ni", so3_left_jacobian_inverse(phi), t)
    return torch.cat([rho, phi], dim=-1)


def se3_adjoint(R: Tensor, t: Tensor) -> Tensor:
    """Ad_T for the [rho; phi] ordering: [[R, hat(t) R], [0, R]].

    This is what moves a spatial twist across a pose, and it is the entire
    coupling in the rigid process model: `Exp(a^) T = T Exp((Ad_{T^-1} a)^)`.
    """
    R = R.double().reshape(-1, 3, 3)
    t = t.double().reshape(-1, 3)
    n = R.shape[0]
    out = torch.zeros((n, 6, 6), dtype=torch.float64, device=R.device)
    out[:, :3, :3] = R
    out[:, :3, 3:] = so3_hat(t) @ R
    out[:, 3:, 3:] = R
    return out
