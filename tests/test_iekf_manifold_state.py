"""CODEX_IMPLEMENT_SPARSE_ALPHA_AWARE_IEKF section 4 + section 23 (`state.py`).

Covers `estimator/manifold.py` (SO(3) exp/log/hat/vee/left-Jacobian) and
`estimator/state.py` (GaussianMotionState, boxplus/boxminus, checkpointing).

Everything runs on CPU in float64 so the numerical tolerances below mean what
they say. The only place autograd appears is test_09, where it is used purely
as an ORACLE for the left-Jacobian identity (spec section 36); the production
modules themselves are asserted to be autograd-free by test_18.
"""

from __future__ import annotations

import math
from pathlib import Path

import torch

from estimator import (
    DELTA_DIM,
    DELTA_SLICE_MU,
    DELTA_SLICE_OMEGA,
    DELTA_SLICE_PHI,
    DELTA_SLICE_V,
    GaussianMotionState,
    boxminus,
    boxplus,
    is_rotation,
    matrix_to_quaternion_wxyz,
    project_to_so3,
    quaternion_wxyz_to_matrix,
    rotation_angle,
    so3_adjoint,
    so3_exp,
    so3_hat,
    so3_left_jacobian,
    so3_left_jacobian_inverse,
    so3_log,
    so3_vee,
    split_delta,
    stack_delta,
)
import estimator.manifold as manifold_mod
import estimator.state as state_mod

DT = torch.float64
PI = math.pi

PASSED: list[str] = []


def check(name: str, condition: bool, detail: str = "") -> None:
    if not condition:
        raise AssertionError(f"{name} FAILED {detail}")
    PASSED.append(name)
    print(f"  ok  {name}" + (f"   [{detail}]" if detail else ""))


# ---------------------------------------------------------------------------
# shared fixtures
# ---------------------------------------------------------------------------
def unit_axes(n: int, seed: int = 0) -> torch.Tensor:
    """n random unit axes plus the three coordinate axes and a few diagonals."""
    g = torch.Generator().manual_seed(seed)
    rand = torch.randn((n, 3), generator=g, dtype=DT)
    fixed = torch.tensor(
        [[1.0, 0, 0], [0, 1.0, 0], [0, 0, 1.0], [1.0, 1.0, 1.0], [-1.0, 2.0, -0.5]], dtype=DT
    )
    axes = torch.cat((fixed, rand))
    return axes / torch.linalg.norm(axes, dim=1, keepdim=True)


# The magnitudes the spec singles out, plus the 2*pi wraparound band.
MAGNITUDES = [
    0.0, 1.0e-12, 1.0e-9, 1.0e-7, 1.0e-4, 1.0e-2, 0.5, 1.0, 2.0,
    PI - 1.0e-6, PI - 1.0e-12, PI, PI + 1.0e-6, 4.0,
    2.0 * PI - 1.0e-6, 2.0 * PI, 2.0 * PI + 1.0e-3, 10.0, 100.0,
]
UNAMBIGUOUS = [m for m in MAGNITUDES if m < PI - 1.0e-9]  # |phi| < pi -> Log(Exp(phi)) == phi


def phi_grid(magnitudes=None) -> torch.Tensor:
    axes = unit_axes(24, seed=3)
    mags = MAGNITUDES if magnitudes is None else magnitudes
    return torch.cat([axes * m for m in mags])


def random_state(n: int = 17, seed: int = 11, dtype: torch.dtype = DT) -> GaussianMotionState:
    g = torch.Generator().manual_seed(seed)
    phi = torch.randn((n, 3), generator=g, dtype=DT) * 0.8
    return GaussianMotionState(
        mu=(torch.randn((n, 3), generator=g, dtype=DT) * 2.0).to(dtype),
        R=so3_exp(phi).to(dtype),
        v=(torch.randn((n, 3), generator=g, dtype=DT) * 0.3).to(dtype),
        omega=(torch.randn((n, 3), generator=g, dtype=DT) * 0.2).to(dtype),
        colors=torch.rand((n, 3), generator=g, dtype=DT).to(dtype),
        opacity=torch.rand((n, 1), generator=g, dtype=DT).to(dtype),
        scales=(torch.rand((n, 3), generator=g, dtype=DT) * 0.1 + 0.01).to(dtype),
    )


# ---------------------------------------------------------------------------
# SECTION 4 - manifold
# ---------------------------------------------------------------------------
def test_01_hat_vee_are_exact_inverses():
    """hat/vee round trip bitwise, and hat is exactly skew-symmetric."""
    phi = phi_grid()
    skew = so3_hat(phi)
    check("§4 so3_hat returns [N,3,3]", tuple(skew.shape) == (phi.shape[0], 3, 3), str(tuple(skew.shape)))
    check("§4 so3_hat output is exactly skew", torch.equal(skew, -skew.transpose(1, 2)))
    check("§4 so3_vee(so3_hat(phi)) == phi bitwise", torch.equal(so3_vee(skew), phi))
    other = phi.roll(1, dims=1)
    cross_err = ((skew @ other[..., None]).squeeze(-1) - torch.cross(phi, other, dim=1)).abs().max()
    check("§4 hat(phi) x == phi cross x", cross_err.item() < 1e-12 * max(1.0, phi.abs().max().item() ** 2),
          f"{cross_err.item():.3e}")
    # vee reads the antisymmetric part, so a symmetric contamination is ignored
    sym = torch.randn((phi.shape[0], 3, 3), generator=torch.Generator().manual_seed(2), dtype=DT)
    sym = sym + sym.transpose(1, 2)
    check("§4 so3_vee discards the symmetric part",
          torch.allclose(so3_vee(skew + sym), phi, atol=1e-14))
    check("§4 so3_hat rejects a wrong trailing dim",
          _raises(lambda: so3_hat(torch.zeros((4, 4), dtype=DT)), ValueError))
    check("§4 so3_vee rejects a wrong matrix shape",
          _raises(lambda: so3_vee(torch.zeros((4, 2, 2), dtype=DT)), ValueError))


def test_02_exp_is_a_rotation_at_every_magnitude():
    """R R^T == I and det == +1 for |phi| from 0 to 100 rad."""
    axes = unit_axes(24, seed=5)
    worst_ortho, worst_det = 0.0, 0.0
    for mag in MAGNITUDES:
        r = so3_exp(axes * mag)
        eye = torch.eye(3, dtype=DT).expand_as(r)
        ortho = (r @ r.transpose(1, 2) - eye).abs().max().item()
        det = (torch.linalg.det(r) - 1.0).abs().max().item()
        worst_ortho, worst_det = max(worst_ortho, ortho), max(worst_det, det)
        check(f"§4 Exp is orthonormal at |phi|={mag:.6g}", ortho < 1e-13, f"{ortho:.3e}")
        check(f"§4 det Exp == +1 at |phi|={mag:.6g}", det < 1e-13, f"{det:.3e}")
    check("§4 no NaN/Inf anywhere in Exp", torch.isfinite(so3_exp(phi_grid())).all().item(),
          f"worst ortho {worst_ortho:.2e}, worst det {worst_det:.2e}")
    check("§4 is_rotation agrees with the explicit test", bool(is_rotation(so3_exp(phi_grid())).all()))
    check("§4 is_rotation rejects a reflection",
          not bool(is_rotation(torch.diag(torch.tensor([1.0, 1.0, -1.0], dtype=DT))[None]).item()))


def test_03_exp_log_round_trip_below_pi():
    """Log(Exp(phi)) == phi for |phi| < pi, including 1e-12 and 1e-7."""
    for mag in UNAMBIGUOUS:
        phi = unit_axes(24, seed=7) * mag
        back = so3_log(so3_exp(phi))
        err = (back - phi).abs().max().item()
        check(f"§4 Log(Exp(phi)) == phi at |phi|={mag:.6g}", err < 1e-14, f"abs err {err:.3e}")
    # zero must come back exactly zero, not a denormal
    check("§4 Log(I) is exactly zero",
          torch.equal(so3_log(torch.eye(3, dtype=DT)[None]), torch.zeros((1, 3), dtype=DT)))


def test_04_log_at_and_near_pi_uses_the_symmetric_branch():
    """The pi case: the axis sign is ambiguous, so R must round trip, not phi."""
    axes = unit_axes(24, seed=13)
    for mag in (PI - 1.0e-6, PI - 1.0e-9, PI - 1.0e-12, PI):
        r = so3_exp(axes * mag)
        back = so3_log(r)
        r_back = so3_exp(back)
        err = (r_back - r).abs().max().item()
        check(f"§4 Exp(Log(R)) == R at |phi|={mag!r}", err < 1e-12, f"abs err {err:.3e}")
        ang = (torch.linalg.norm(back, dim=1) - mag).abs().max().item()
        check(f"§4 the recovered angle is right at |phi|={mag!r}", ang < 1e-9, f"{ang:.3e}")
        check(f"§4 Log is finite at |phi|={mag!r}", torch.isfinite(back).all().item())
    # at exactly pi the two valid answers are +-phi; the axis must be collinear
    r_pi = so3_exp(axes * PI)
    back = so3_log(r_pi)
    axis_back = back / torch.linalg.norm(back, dim=1, keepdim=True)
    collinear = (torch.cross(axis_back, axes, dim=1)).abs().max().item()
    check("§4 at exactly pi the axis is collinear with the truth", collinear < 1e-7, f"{collinear:.3e}")
    check("§4 at exactly pi the sign choice is deterministic",
          torch.equal(so3_log(r_pi), so3_log(r_pi.clone())))
    # the canonical representative puts the largest-magnitude component positive
    pivot = axis_back.abs().argmax(dim=1)
    check("§4 at exactly pi the canonical representative is used",
          bool((torch.gather(axis_back, 1, pivot[:, None]).squeeze(1) > 0).all()))
    # a naive asin/(2 sin theta) formula would blow up here; ours must not
    naive_scale = PI / torch.linalg.norm(so3_vee(r_pi), dim=1).clamp_min(1e-300)
    check("§4 the naive 1/sin(theta) branch really is degenerate at pi",
          bool((naive_scale > 1e8).all()), f"min {naive_scale.min().item():.3e}")


def test_05_log_wraps_past_pi_instead_of_exploding():
    """|phi| in (pi, 2pi] is a 2pi wraparound: R round trips, phi does not."""
    axes = unit_axes(24, seed=17)
    for mag in (PI + 1.0e-6, 4.0, 2.0 * PI - 1.0e-6, 2.0 * PI, 2.0 * PI + 1.0e-3, 10.0, 100.0):
        r = so3_exp(axes * mag)
        back = so3_log(r)
        err = (so3_exp(back) - r).abs().max().item()
        check(f"§4 Exp(Log(R)) == R at |phi|={mag:.6g}", err < 1e-12, f"abs err {err:.3e}")
        check(f"§4 |Log(R)| <= pi at |phi|={mag:.6g}",
              bool((torch.linalg.norm(back, dim=1) <= PI + 1e-12).all()))
        check(f"§4 Log is finite at |phi|={mag:.6g}", torch.isfinite(back).all().item())
    # |phi| = 2pi is literally the identity, so the principal log is zero
    r_2pi = so3_exp(axes * (2.0 * PI))
    check("§4 |phi| = 2pi maps to the identity",
          so3_log(r_2pi).abs().max().item() < 1e-14, f"{so3_log(r_2pi).abs().max().item():.3e}")
    # wraparound sign: Log(Exp((2pi - d) u)) = -d u
    delta = 1.0e-3
    back = so3_log(so3_exp(axes * (2.0 * PI - delta)))
    check("§4 the 2pi wraparound flips the axis",
          (back + axes * delta).abs().max().item() < 1e-11,
          f"{(back + axes * delta).abs().max().item():.3e}")
    check("§4 rotation_angle matches |Log(R)|",
          (rotation_angle(r_2pi) - torch.linalg.norm(so3_log(r_2pi), dim=1)).abs().max().item() < 1e-14)


def test_06_log_is_robust_to_round_off_in_R():
    """Real R's arrive after long product chains and are only nearly orthonormal."""
    g = torch.Generator().manual_seed(23)
    r = so3_exp(unit_axes(24, seed=23) * (PI - 1e-8))
    noisy = r + 1e-13 * torch.randn(r.shape, generator=g, dtype=DT)
    back = so3_log(noisy)
    check("§4 Log of a slightly non-orthonormal R is finite", torch.isfinite(back).all().item())
    err = (so3_exp(back) - r).abs().max().item()
    check("§4 Log of a noisy near-pi R still recovers R", err < 1e-6, f"abs err {err:.3e}")
    # trace slightly outside [-1, 1] must not produce a NaN out of acos/atan2
    over = torch.eye(3, dtype=DT)[None] * (1.0 + 1e-12)
    check("§4 a trace above 3 does not NaN", torch.isfinite(so3_log(over)).all().item())
    under = -torch.eye(3, dtype=DT)[None] * (1.0 + 1e-12)
    check("§4 a trace below -1 does not NaN", torch.isfinite(so3_log(under)).all().item())


def test_07_left_jacobian_times_its_inverse_is_identity():
    for mag in MAGNITUDES:
        if abs(mag - 2.0 * PI) < 1e-2:
            continue  # J_l is genuinely singular at 2*pi; checked separately
        phi = unit_axes(24, seed=29) * mag
        jl, jli = so3_left_jacobian(phi), so3_left_jacobian_inverse(phi)
        eye = torch.eye(3, dtype=DT).expand_as(jl)
        err = (jl @ jli - eye).abs().max().item()
        err_rev = (jli @ jl - eye).abs().max().item()
        check(f"§4 J_l J_l^-1 == I at |phi|={mag:.6g}", err < 1e-12, f"{err:.3e}")
        check(f"§4 J_l^-1 J_l == I at |phi|={mag:.6g}", err_rev < 1e-12, f"{err_rev:.3e}")
    zero = torch.zeros((3, 3), dtype=DT)
    check("§4 J_l(0) == I", torch.allclose(so3_left_jacobian(zero), torch.eye(3, dtype=DT).expand(3, 3, 3)))
    check("§4 J_l^-1(0) == I",
          torch.allclose(so3_left_jacobian_inverse(zero), torch.eye(3, dtype=DT).expand(3, 3, 3)))
    check("§4 det J_l == 2(1-cos t)/t^2",
          _det_matches_closed_form(), "closed form over the whole grid")
    sing = unit_axes(4, seed=31) * (2.0 * PI)
    check("§4 J_l^-1 stays finite at the 2pi singularity",
          torch.isfinite(so3_left_jacobian_inverse(sing)).all().item())


def _det_matches_closed_form() -> bool:
    worst = 0.0
    for mag in [m for m in MAGNITUDES if m > 1e-6 and abs(m - 2 * PI) > 1e-2]:
        phi = unit_axes(8, seed=37) * mag
        det = torch.linalg.det(so3_left_jacobian(phi))
        # 2(1 - cos t) written as 4 sin^2(t/2): the literal form is the ORACLE
        # that loses precision at t ~ 1e-4, not the implementation under test.
        closed = 4.0 * math.sin(0.5 * mag) ** 2 / (mag * mag)
        worst = max(worst, (det - closed).abs().max().item())
    return worst < 1e-12


def test_08_left_jacobian_defining_identity_finite_difference():
    """Exp(phi + d) ~= Exp(J_l(phi) d) Exp(phi), with second-order residual."""
    axes = unit_axes(8, seed=41)
    g = torch.Generator().manual_seed(43)
    direction = torch.randn((axes.shape[0], 3), generator=g, dtype=DT)
    direction = direction / torch.linalg.norm(direction, dim=1, keepdim=True)
    for mag in (1e-6, 1e-3, 0.5, 1.0, 2.0, PI - 1e-3):
        phi = axes * mag
        errs = {}
        for h in (1e-3, 1e-4, 1e-6):
            d = direction * h
            lhs = so3_exp(phi + d)
            rhs = so3_exp((so3_left_jacobian(phi) @ d[..., None]).squeeze(-1)) @ so3_exp(phi)
            errs[h] = (lhs - rhs).abs().max().item()
        check(f"§4 J_l identity holds to first order at |phi|={mag:.6g}",
              errs[1e-6] < 1e-11, f"residual {errs[1e-6]:.3e}")
        # Halving h must quarter the residual: that is what distinguishes a
        # correct J_l from one that merely happens to be small. Measured at
        # h = 1e-3/1e-4, where the O(h^2) term is still far above the 1e-16
        # evaluation floor (at h = 1e-6 it is not, and the ratio saturates).
        ratio = errs[1e-3] / max(errs[1e-4], 1e-300)
        check(f"§4 the J_l residual is second order at |phi|={mag:.6g}",
              30.0 < ratio < 300.0, f"err(1e-3)/err(1e-4) = {ratio:.1f}")


def test_09_left_jacobian_matches_two_independent_oracles():
    """§36: autograd appears ONLY here, as an oracle, never in estimator code.

    Oracle A (autograd): differentiating the retraction identity
    ``Exp(phi + d) = Exp(hat(J_l d)) Exp(phi) + O(|d|^2)`` at ``d = 0`` gives
    ``dR/dd_j = hat(J_l e_j) R``.  The derivative is taken through so3_exp only
    -- NOT through so3_log, whose |vee(R - R^T)| is a genuine 0/0 in the
    gradient at ``R = I`` (values are exact there; derivatives are not, and the
    spec does not ask for them).

    Oracle B (quadrature, no autograd at all): ``J_l(phi) = int_0^1
    Exp(s phi) ds``, integrated with 32-node Gauss-Legendre against
    ``torch.matrix_exp`` -- a completely separate code path from Rodrigues.
    """
    axes = unit_axes(5, seed=47)
    nodes, weights = _gauss_legendre_01(32)
    for mag in (1e-3, 0.7, 2.0, PI - 1e-2, 5.0):
        phi = (axes * mag)[0]
        rotation = so3_exp(phi[None])[0]
        analytic = so3_left_jacobian(phi[None])[0]

        jac = torch.autograd.functional.jacobian(
            lambda d: so3_exp((phi + d)[None])[0], torch.zeros(3, dtype=DT, requires_grad=True)
        )  # [3, 3, 3]: jac[..., j] = dR/dd_j
        predicted = torch.stack(
            [so3_hat(analytic[:, j][None])[0] @ rotation for j in range(3)], dim=-1
        )
        err = (jac - predicted).abs().max().item()
        check(f"§4 J_l matches the autograd oracle at |phi|={mag:.6g}", err < 1e-11, f"{err:.3e}")

        quad = (weights[:, None, None] * torch.matrix_exp(nodes[:, None, None] * so3_hat(phi[None])[0])).sum(0)
        err_q = (quad - analytic).abs().max().item()
        check(f"§4 J_l == int_0^1 Exp(s phi) ds at |phi|={mag:.6g}", err_q < 1e-12, f"{err_q:.3e}")


def _gauss_legendre_01(n: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Gauss-Legendre nodes/weights mapped from [-1, 1] to [0, 1]."""
    import numpy as np

    x, w = np.polynomial.legendre.leggauss(n)
    return (
        torch.tensor(0.5 * (x + 1.0), dtype=DT),
        torch.tensor(0.5 * w, dtype=DT),
    )


def test_10_adjoint_and_projection_helpers():
    phi = phi_grid([0.3, 1.2, 3.0])
    r = so3_exp(phi)
    check("§5 Ad_R == R for SO(3)", torch.equal(so3_adjoint(r), r))
    # Ad_R acts as R on tangent vectors: R hat(w) R^T == hat(R w)
    g = torch.Generator().manual_seed(53)
    w = torch.randn((r.shape[0], 3), generator=g, dtype=DT)
    lhs = r @ so3_hat(w) @ r.transpose(1, 2)
    rhs = so3_hat((r @ w[..., None]).squeeze(-1))
    check("§5 R hat(w) R^T == hat(Ad_R w)", (lhs - rhs).abs().max().item() < 1e-13)
    drifted = r + 1e-6 * torch.randn(r.shape, generator=g, dtype=DT)
    fixed = project_to_so3(drifted)
    check("§4 project_to_so3 returns a rotation", bool(is_rotation(fixed).all()))
    check("§4 project_to_so3 stays near the input", (fixed - r).abs().max().item() < 1e-5)
    reflected = r.clone()
    reflected[:, :, 2] *= -1.0
    check("§4 project_to_so3 repairs a reflection", bool(is_rotation(project_to_so3(reflected)).all()))
    check("§4 project_to_so3 preserves dtype", project_to_so3(r).dtype == DT)


def test_11_manifold_is_batch_and_dtype_agnostic():
    phi = unit_axes(6, seed=59)[:6] * 0.9
    nested = phi.reshape(2, 3, 3)
    check("§4 so3_exp supports extra leading dims",
          torch.allclose(so3_exp(nested).reshape(6, 3, 3), so3_exp(phi)))
    check("§4 so3_log supports extra leading dims",
          torch.allclose(so3_log(so3_exp(nested)).reshape(6, 3), phi, atol=1e-14))
    f32 = phi.float()
    r32 = so3_exp(f32)
    check("§4 float32 is preserved by so3_exp", r32.dtype == torch.float32)
    check("§4 float32 round trip is accurate to float32",
          (so3_log(r32) - f32).abs().max().item() < 1e-6)
    r32_pi = so3_exp(unit_axes(6, seed=61)[:6].float() * float(PI))
    check("§4 the float32 pi branch is finite", torch.isfinite(so3_log(r32_pi)).all().item())
    check("§4 float32 Exp(Log(R)) == R near pi",
          (so3_exp(so3_log(r32_pi)) - r32_pi).abs().max().item() < 1e-5)
    check("§4 an integer tensor is rejected loudly",
          _raises(lambda: so3_exp(torch.zeros((2, 3), dtype=torch.int64)), TypeError))
    check("§4 an empty batch is handled", tuple(so3_exp(torch.zeros((0, 3), dtype=DT)).shape) == (0, 3, 3))
    check("§4 so3_log of an empty batch is handled",
          tuple(so3_log(torch.zeros((0, 3, 3), dtype=DT)).shape) == (0, 3))


# ---------------------------------------------------------------------------
# SECTION 4 / 23 - state
# ---------------------------------------------------------------------------
def test_12_state_layout_and_construction():
    s = random_state()
    check("§4 num_gaussians", s.num_gaussians == 17 and len(s) == 17)
    check("§4 the error state is 12D", DELTA_DIM == 12 and s.state_dim == 12 * 17)
    check("§4 the delta block order is mu|phi|v|omega",
          (DELTA_SLICE_MU, DELTA_SLICE_PHI, DELTA_SLICE_V, DELTA_SLICE_OMEGA)
          == (slice(0, 3), slice(3, 6), slice(6, 9), slice(9, 12)))
    check("§4 R is stored as a rotation", bool(is_rotation(s.R).all()))
    check("§3 Sigma = R diag(s^2) R^T",
          torch.allclose(s.covariance(),
                         s.R @ torch.diag_embed(s.scales.square()) @ s.R.transpose(1, 2), atol=1e-15))
    check("§3 Sigma is symmetric positive definite",
          bool((torch.linalg.eigvalsh(s.covariance()) > 0).all()))
    ident = GaussianMotionState.identity(5)
    check("§4 identity() is at rest at the origin",
          bool((ident.mu == 0).all() and (ident.v == 0).all() and (ident.omega == 0).all()))
    check("§4 identity() rotations are the identity",
          torch.equal(ident.R, torch.eye(3, dtype=DT).expand(5, 3, 3)))
    check("§4 a wrong mu shape is rejected",
          _raises(lambda: GaussianMotionState(
              mu=torch.zeros((4, 2), dtype=DT), R=torch.eye(3, dtype=DT).expand(4, 3, 3),
              v=torch.zeros((4, 3), dtype=DT), omega=torch.zeros((4, 3), dtype=DT),
              colors=torch.zeros((4, 3), dtype=DT), opacity=torch.zeros((4, 1), dtype=DT),
              scales=torch.ones((4, 3), dtype=DT)), ValueError))
    check("§4 a mixed motion dtype is rejected",
          _raises(lambda: GaussianMotionState(
              mu=torch.zeros((4, 3), dtype=DT), R=torch.eye(3, dtype=DT).expand(4, 3, 3),
              v=torch.zeros((4, 3), dtype=torch.float32), omega=torch.zeros((4, 3), dtype=DT),
              colors=torch.zeros((4, 3), dtype=DT), opacity=torch.zeros((4, 1), dtype=DT),
              scales=torch.ones((4, 3), dtype=DT)), TypeError))


def test_13_boxplus_is_the_spec_retraction():
    s = random_state()
    g = torch.Generator().manual_seed(67)
    d = torch.randn((s.num_gaussians, DELTA_DIM), generator=g, dtype=DT) * 0.1
    out = boxplus(s, d)
    dmu, dphi, dv, domega = split_delta(d)
    check("§4 mu+ = mu + dmu", torch.equal(out.mu, s.mu + dmu))
    check("§4 v+ = v + dv", torch.equal(out.v, s.v + dv))
    check("§4 omega+ = omega + domega", torch.equal(out.omega, s.omega + domega))
    check("§4 R+ = Exp(hat(dphi)) @ R  (LEFT multiplication)",
          torch.equal(out.R, so3_exp(dphi) @ s.R))
    # the right-multiplied convention must NOT be what we implemented
    right = s.R @ so3_exp(dphi)
    check("§4 boxplus is not the right-multiplied retraction",
          (out.R - right).abs().max().item() > 1e-6,
          f"left/right differ by {(out.R - right).abs().max().item():.3e}")
    check("§4 boxplus leaves the input state untouched",
          torch.equal(s.mu, random_state().mu) and torch.equal(s.R, random_state().R))
    check("§4 boxplus output is still on SO(3)", bool(is_rotation(out.R).all()))
    check("§4 the method form agrees with the function form",
          torch.equal(s.boxplus(d).R, out.R) and torch.equal(s.boxplus(d).mu, out.mu))
    check("§4 stack_delta/split_delta round trip", torch.equal(stack_delta(dmu, dphi, dv, domega), d))


def test_14_boxplus_with_zero_delta_is_bitwise_identity():
    s = random_state()
    out = boxplus(s, torch.zeros((s.num_gaussians, DELTA_DIM), dtype=DT))
    for name in ("mu", "R", "v", "omega"):
        check(f"§4 zero delta leaves {name} bitwise unchanged",
              torch.equal(getattr(out, name), getattr(s, name)))
    check("§4 Exp(0) is exactly the identity matrix",
          torch.equal(so3_exp(torch.zeros((3, 3), dtype=DT)), torch.eye(3, dtype=DT).expand(3, 3, 3)))
    check("§3 zero delta shares the frozen appearance arrays",
          out.colors is s.colors and out.opacity is s.opacity and out.scales is s.scales)


def test_15_boxplus_boxminus_round_trip():
    s = random_state()
    g = torch.Generator().manual_seed(71)
    for scale, tol in ((1e-9, 1e-15), (1e-3, 1e-14), (0.1, 1e-13), (1.0, 1e-12)):
        d = torch.randn((s.num_gaussians, DELTA_DIM), generator=g, dtype=DT) * scale
        if scale >= 1.0:  # keep |dphi| < pi so the principal log does not wrap
            d[:, DELTA_SLICE_PHI] *= 0.5
        back = boxminus(boxplus(s, d), s)
        err = (back - d).abs().max().item()
        check(f"§4 boxminus(boxplus(x,d), x) == d at scale {scale:g}", err < tol, f"{err:.3e}")
    check("§4 boxminus(x, x) is exactly zero",
          torch.equal(boxminus(s, s), torch.zeros((s.num_gaussians, DELTA_DIM), dtype=DT)))
    # the other direction: boxplus(b, a boxminus b) == a
    a, b = random_state(seed=73), random_state(seed=79)
    # shrink the relative rotation below pi so the retraction is the exact inverse
    small = boxminus(a, b) * 0.25
    rebuilt = boxplus(b, small)
    check("§4 boxplus(b, a boxminus b) reproduces a",
          (boxminus(rebuilt, b) - small).abs().max().item() < 1e-13)
    check("§4 the rotation block of boxminus is Log(R_a R_b^T)",
          torch.equal(boxminus(a, b)[:, DELTA_SLICE_PHI], so3_log(a.R @ b.R.transpose(1, 2))))
    check("§4 boxminus returns [N,12]", tuple(boxminus(a, b).shape) == (17, 12))
    check("§4 a size mismatch in boxminus is rejected",
          _raises(lambda: boxminus(a, random_state(n=3, seed=83)), ValueError))


def test_16_elementwise_rotation_updates_raise():
    """§4: 'the production code must not add rotation matrices elementwise'."""
    s = random_state()
    g = torch.Generator().manual_seed(89)
    check("§4 state + state raises", _raises(lambda: s + s, TypeError))
    check("§4 state - state raises", _raises(lambda: s - s, TypeError))
    check("§4 state * scalar raises", _raises(lambda: s * 2.0, TypeError))
    check("§4 scalar * state raises", _raises(lambda: 2.0 * s, TypeError))

    def _iadd():
        t = random_state()
        t += t
        return t

    check("§4 state += state raises", _raises(_iadd, TypeError))

    def _assign_sum():
        t = random_state()
        t.R = t.R + 0.01 * torch.randn(t.R.shape, generator=g, dtype=DT)

    check("§4 assigning R + dR raises", _raises(_assign_sum, ValueError))
    check("§4 the message names the retraction",
          "Exp(hat(dphi))" in _message(_assign_sum))
    check("§4 a [N,3,3] delta is rejected with a rotation-specific message",
          "rotations are NOT updated by adding matrices"
          in _message(lambda: boxplus(s, torch.zeros((s.num_gaussians, 3, 3), dtype=DT))))
    check("§4 a wrong-width delta is rejected",
          _raises(lambda: boxplus(s, torch.zeros((s.num_gaussians, 6), dtype=DT)), ValueError))
    check("§4 a wrong-length delta is rejected",
          _raises(lambda: boxplus(s, torch.zeros((3, DELTA_DIM), dtype=DT)), ValueError))
    check("§4 constructing from a non-rotation raises",
          _raises(lambda: GaussianMotionState(
              mu=torch.zeros((2, 3), dtype=DT), R=torch.ones((2, 3, 3), dtype=DT),
              v=torch.zeros((2, 3), dtype=DT), omega=torch.zeros((2, 3), dtype=DT),
              colors=torch.zeros((2, 3), dtype=DT), opacity=torch.zeros((2, 1), dtype=DT),
              scales=torch.ones((2, 3), dtype=DT)), ValueError))
    # accumulated round-off must NOT trip the guard, and is repairable
    drift = s.R.clone()
    for _ in range(50):
        drift = so3_exp(torch.randn((s.num_gaussians, 3), generator=g, dtype=DT) * 1e-3) @ drift
    s.R = drift
    check("§4 ordinary Exp-product drift passes the guard", bool(is_rotation(s.R).all()))
    check("§4 reorthonormalize repairs drift", bool(is_rotation(s.reorthonormalize().R).all()))
    check("§4 boxminus rejects a non-state", _raises(lambda: boxminus(s, s.R), TypeError))
    check("§4 boxplus rejects a non-state", _raises(lambda: boxplus(s.R, s.R), TypeError))


def test_17_checkpoint_round_trip_is_bitwise():
    s = random_state(n=9, seed=97)
    sd = s.state_dict()
    restored = GaussianMotionState.from_state_dict(sd)
    for name in ("mu", "R", "v", "omega", "colors", "opacity", "scales"):
        check(f"§23 checkpoint preserves {name} bitwise",
              torch.equal(getattr(restored, name), getattr(s, name)))
    check("§23 the checkpoint carries a version", sd["version"] == state_mod.STATE_DICT_VERSION)
    check("§23 the checkpoint carries the size", sd["num_gaussians"] == 9)
    check("§23 state_dict snapshots are detached", all(
        not sd[k].requires_grad for k in ("mu", "R", "v", "omega", "colors", "opacity", "scales")))
    # a snapshot must not alias the live state
    moved = boxplus(s, torch.full((9, DELTA_DIM), 0.01, dtype=DT))
    check("§23 the snapshot is immune to a later boxplus", torch.equal(sd["mu"], s.mu))
    check("§23 boxplus really did move the state", not torch.equal(moved.mu, s.mu))
    # in-place load
    target = random_state(n=9, seed=101)
    target.load_state_dict(sd)
    for name in ("mu", "R", "v", "omega", "colors", "opacity", "scales"):
        check(f"§23 load_state_dict restores {name} bitwise",
              torch.equal(getattr(target, name), getattr(s, name)))
    # a torch.save/torch.load cycle must survive too
    import io
    buf = io.BytesIO()
    torch.save(sd, buf)
    buf.seek(0)
    round_tripped = GaussianMotionState.from_state_dict(torch.load(buf, weights_only=True))
    check("§23 torch.save/load preserves every field bitwise", all(
        torch.equal(getattr(round_tripped, n), getattr(s, n))
        for n in ("mu", "R", "v", "omega", "colors", "opacity", "scales")))
    check("§23 a truncated checkpoint is rejected",
          _raises(lambda: GaussianMotionState.from_state_dict({"version": 1, "num_gaussians": 9}), KeyError))
    check("§23 a wrong version is rejected",
          _raises(lambda: GaussianMotionState.from_state_dict({**sd, "version": 99}), ValueError))
    check("§23 a size mismatch on in-place load is rejected",
          _raises(lambda: random_state(n=4, seed=103).load_state_dict(sd), ValueError))
    check("§23 a dtype mismatch on in-place load is rejected",
          _raises(lambda: random_state(n=9, seed=107, dtype=torch.float32).load_state_dict(sd), TypeError))


def test_18_device_dtype_detach_clone():
    s = random_state(n=6, seed=109)
    check("§23 .to(dtype) casts every field",
          s.to(dtype=torch.float32).dtype == torch.float32
          and s.to(dtype=torch.float32).colors.dtype == torch.float32)
    check("§23 .to(dtype) positionally is accepted", s.to(torch.float32).dtype == torch.float32)
    check("§23 .to() with nothing to do returns self", s.to() is s)
    check("§23 .to('cpu') stays on cpu", s.to("cpu").device.type == "cpu")
    check("§23 the float32 cast still yields rotations", bool(is_rotation(s.to(torch.float32).R).all()))
    c = s.clone()
    check("§23 clone is bitwise equal", all(
        torch.equal(getattr(c, n), getattr(s, n))
        for n in ("mu", "R", "v", "omega", "colors", "opacity", "scales")))
    check("§23 clone does not alias", c.mu.data_ptr() != s.mu.data_ptr()
          and c.colors.data_ptr() != s.colors.data_ptr())
    attached = GaussianMotionState(
        mu=s.mu.clone().requires_grad_(True), R=s.R, v=s.v, omega=s.omega,
        colors=s.colors, opacity=s.opacity, scales=s.scales)
    check("§36 detach() cuts the autograd graph",
          not attached.detach().mu.requires_grad and attached.mu.requires_grad)
    check("§36 detach keeps the values bitwise", torch.equal(attached.detach().mu, attached.mu))
    check("§23 an opacity of shape [N] instead of [N,1] is rejected",
          _raises(lambda: GaussianMotionState(
              mu=s.mu, R=s.R, v=s.v, omega=s.omega, colors=s.colors,
              opacity=s.opacity.squeeze(1), scales=s.scales), ValueError))
    check("§23 a float32 appearance block alongside a float64 state is allowed",
          GaussianMotionState(mu=s.mu, R=s.R, v=s.v, omega=s.omega,
                              colors=s.colors.float(), opacity=s.opacity.float(),
                              scales=s.scales.float()).dtype == DT)
    if torch.cuda.is_available():
        check("§23 a cross-device state is rejected",
              _raises(lambda: GaussianMotionState(
                  mu=s.mu.cuda(), R=s.R, v=s.v, omega=s.omega,
                  colors=s.colors, opacity=s.opacity, scales=s.scales), ValueError))
        moved_gpu = s.to("cuda")
        check("§23 .to('cuda') moves every field",
              all(getattr(moved_gpu, n).is_cuda for n in
                  ("mu", "R", "v", "omega", "colors", "opacity", "scales")))
    else:
        check("§23 the cross-device guard is present in the source",
              "the measurement operator indexes them together"
              in Path(state_mod.__file__).read_text(encoding="utf-8"))


def test_19_frozen_appearance_is_never_updated():
    """§3: appearance, opacity and scales are frozen during motion estimation."""
    s = random_state(n=8, seed=113)
    g = torch.Generator().manual_seed(127)
    moved = s
    for _ in range(5):
        moved = boxplus(moved, torch.randn((8, DELTA_DIM), generator=g, dtype=DT) * 0.05)
    for name in ("colors", "opacity", "scales"):
        check(f"§3 {name} is bitwise unchanged after 5 boxplus steps",
              torch.equal(getattr(moved, name), getattr(s, name)))
        check(f"§3 {name} is shared, not copied", getattr(moved, name) is getattr(s, name))
    check("§3 assert_appearance_unchanged accepts a motion-only update",
          _succeeds(lambda: s.assert_appearance_unchanged(moved)))
    tampered = moved.clone()
    tampered.colors[0, 0] += 1e-9
    check("§3 assert_appearance_unchanged catches a tampered colour",
          _raises(lambda: s.assert_appearance_unchanged(tampered), AssertionError))
    check("§3 boxminus ignores appearance entirely",
          torch.equal(boxminus(moved, s)[:, DELTA_SLICE_MU], moved.mu - s.mu))
    check("§4 the delta vector has no appearance slots", DELTA_DIM == 12)


def test_20_quaternion_bridge_matches_the_repo_convention():
    """The rasterizer stores wxyz quaternions; the bridge must be exact."""
    phi = phi_grid([0.0, 1e-9, 0.3, 1.5, PI - 1e-7, PI, 4.0])
    r = so3_exp(phi)
    q = matrix_to_quaternion_wxyz(r)
    check("§23 quaternions are unit norm",
          (torch.linalg.norm(q, dim=1) - 1.0).abs().max().item() < 1e-14)
    check("§23 the w >= 0 hemisphere is canonical", bool((q[:, 0] >= 0).all()))
    err = (quaternion_wxyz_to_matrix(q) - r).abs().max().item()
    check("§23 quat -> matrix -> quat reproduces R", err < 1e-13, f"{err:.3e}")
    # cross-check against the repository's own build_rotation convention
    w, x, y, z = q.unbind(-1)
    ref = torch.stack((
        1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y),
        2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x),
        2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y),
    ), dim=-1).reshape(-1, 3, 3)
    conv_err = (quaternion_wxyz_to_matrix(q) - ref).abs().max().item()
    check("§23 the convention matches utils.general_utils.build_rotation",
          conv_err < 1e-14, f"{conv_err:.3e}")
    check("§23 an unnormalised quaternion is normalised on the way in",
          (quaternion_wxyz_to_matrix(q * 7.0) - quaternion_wxyz_to_matrix(q)).abs().max().item() < 1e-14)
    s = GaussianMotionState.from_gaussian_arrays(
        xyz=torch.randn((11, 3), generator=torch.Generator().manual_seed(131), dtype=DT),
        rotation_wxyz=matrix_to_quaternion_wxyz(so3_exp(unit_axes(11, seed=137)[:11] * 0.9)),
        scales=torch.full((11, 3), 0.05, dtype=DT),
        opacity=torch.full((11,), 0.7, dtype=DT),
        colors=torch.full((11, 3), 0.3, dtype=DT),
    )
    check("§23 from_gaussian_arrays builds a valid state", bool(is_rotation(s.R).all()))
    check("§23 from_gaussian_arrays reshapes opacity to [N,1]", tuple(s.opacity.shape) == (11, 1))
    check("§23 from_gaussian_arrays defaults the twist to zero",
          bool((s.v == 0).all() and (s.omega == 0).all()))
    arrays = s.to_gaussian_arrays()
    check("§23 to_gaussian_arrays round trips the rotation",
          (quaternion_wxyz_to_matrix(arrays["rotation_wxyz"]) - s.R).abs().max().item() < 1e-13)
    check("§23 to_gaussian_arrays round trips the centers", torch.equal(arrays["xyz"], s.mu))
    param = torch.randn((4, 3), dtype=DT, requires_grad=True)
    detached = GaussianMotionState.from_gaussian_arrays(
        xyz=param, rotation_wxyz=torch.tensor([[1.0, 0, 0, 0]], dtype=DT).repeat(4, 1),
        scales=torch.ones((4, 3), dtype=DT), opacity=torch.ones((4,), dtype=DT),
        colors=torch.zeros((4, 3), dtype=DT))
    check("§36 from_gaussian_arrays detaches live parameters", not detached.mu.requires_grad)


def test_21_estimator_modules_are_autograd_free():
    """§36: no backward / no Adam / no autograd inside the production path."""
    for module in (manifold_mod, state_mod):
        src = Path(module.__file__).read_text(encoding="utf-8")
        name = Path(module.__file__).name
        code = "\n".join(_strip_comments_and_docstrings(src))
        for forbidden in (".backward(", "torch.autograd", "Adam", "torch.optim", "requires_grad_("):
            check(f"§36 {name} contains no {forbidden!r}", forbidden not in code)
        for forbidden in ("TODO", "FIXME", "NotImplementedError", "raise NotImplemented"):
            check(f"§0 {name} contains no {forbidden!r}", forbidden not in code)
        check(f"§0 {name} has no bare `pass` body", "\n    pass" not in code)
    src = Path(state_mod.__file__).read_text(encoding="utf-8")
    check("§4 boxplus uses the left-multiplied Exp retraction", "so3_exp(dphi) @ state.R" in src)
    check("§4 boxminus uses so3_log(R_a R_b^T)", "so3_log(state_a.R @ other.R.transpose(1, 2))" in src)
    # When this test was written the package held only the manifold/state layer
    # and the guard asserted the later modules were absent. They have since
    # landed, so the invariant it protects -- importing `estimator` must not
    # fail because of a half-wired package -- is now checked the other way
    # round: every section 23 module named in __init__ must actually import.
    init_src = Path(Path(state_mod.__file__).parent / "__init__.py").read_text(encoding="utf-8")
    import importlib

    for name in ("process", "covariance", "measurement", "pcg", "sparse_iekf"):
        if f"from .{name} import" in init_src:
            mod = importlib.import_module(f"estimator.{name}")
            check(f"§23 estimator.{name} imports cleanly", mod is not None)
    check("§23 the package itself imports", importlib.import_module("estimator") is not None)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def _raises(fn, exc) -> bool:
    try:
        fn()
    except exc:
        return True
    except Exception as other:  # wrong exception type is a failure, not a pass
        raise AssertionError(f"expected {exc.__name__}, got {type(other).__name__}: {other}") from other
    return False


def _succeeds(fn) -> bool:
    fn()
    return True


def _message(fn) -> str:
    try:
        fn()
    except Exception as exc:
        return str(exc)
    return ""


def _strip_comments_and_docstrings(src: str) -> list[str]:
    """Code lines only, so the prose in the module docstrings cannot trip the scan."""
    import ast
    import io
    import tokenize

    docstrings = set()
    tree = ast.parse(src)
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            doc = ast.get_docstring(node, clean=False)
            if doc is not None and node.body:
                first = node.body[0]
                docstrings.update(range(first.lineno, (first.end_lineno or first.lineno) + 1))
    out = []
    comment_lines = set()
    for tok in tokenize.generate_tokens(io.StringIO(src).readline):
        if tok.type == tokenize.COMMENT:
            comment_lines.add(tok.start[0])
    for i, line in enumerate(src.splitlines(), start=1):
        if i in docstrings:
            continue
        if i in comment_lines:
            line = line.split("#", 1)[0]
        out.append(line)
    return out


def main() -> None:
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for fn in tests:
        print(f"\n{fn.__name__}")
        fn()
    print(f"\n{len(PASSED)} checks passed across {len(tests)} tests")


if __name__ == "__main__":
    main()
